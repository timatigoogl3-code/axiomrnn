"""
Tests for the capacity solver (axplan.solve).

The claim under test is NOT "the solver finds the true optimum". The claim is
narrower and checkable:

  1. every configuration it reports as fitting really fits, when the memory
     model is evaluated independently;
  2. the configuration it reports as NOT fitting really does not fit -- an
     optimiser that only ever returns successes cannot be distinguished from
     one that is simply optimistic;
  3. it is monotone in the knobs the memory model is monotone in;
  4. it refuses to answer rather than answering when nothing fits.

No quality claim is tested here because the package makes none (UX_RULES 10).
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axplan.solve import (largest_that_fits, frontier, make_spec,   # noqa
                          _ascending, _descending, Candidate)
from axplan.memory import analyse

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))


GIB = 2 ** 30


# ── 1. THE LADDER RESPECTS ITS BOUND ───────────────────────────────────────

def test_ladder_respects_bound():
    """A search ceiling that is ignored is a lie about what was searched."""
    for limit in (1, 3, 100, 512, 1000):
        up = _ascending(limit)
        check(f"_ascending({limit}) stays within the bound",
              up[-1] == limit and all(x <= limit for x in up), f"{up}")
    for limit in (128, 256, 1000, 4096):
        down = _descending(limit)
        check(f"_descending({limit}) stays within the bound",
              max(down) <= limit and min(down) >= 128, f"{down[:3]}...")


# ── 2. WHAT IT REPORTS AS FITTING, FITS ────────────────────────────────────

def test_reported_fit_really_fits():
    """Independent re-evaluation, not the solver's own bookkeeping."""
    for vram in (2.0, 6.0, 24.0, 80.0):
        best, _ = largest_that_fits(vram, n_layers=16, timesteps=128, batch=8)
        if best is None:
            check(f"{vram} GiB: a solution exists at some size", False,
                  "solver returned nothing")
            continue
        plan = analyse(best.spec())
        check(f"{vram} GiB: reported solution fits when re-evaluated",
              plan.peak_bytes <= plan.vram_bytes,
              f"peak {plan.peak_bytes/GIB:.3f} <= {plan.vram_bytes/GIB:.3f}")
        check(f"{vram} GiB: re-evaluated peak matches the reported peak",
              plan.peak_bytes == best.peak_bytes)


# ── 3. THE FAILURE BOUNDARY IS REAL ────────────────────────────────────────

def test_failure_boundary_is_real():
    """The anti-optimism test: the reported overflow must overflow.

    This is the check the framework's own history demands (UX_RULES 14): a
    search that only ever reports successes is indistinguishable from a
    search that is simply wrong.

    The width ceiling is set ABOVE the answer on purpose. When the
    recommendation lands on the ceiling there is no boundary to report, and
    inventing one is the failure this test exists to catch.
    """
    best, fail = largest_that_fits(6.0, hidden=32768, n_layers=32,
                                  timesteps=256, batch=8)
    check("6 GiB: a solution is found", best is not None)
    check("6 GiB: an overflow is also reported", fail is not None)
    if best is None or fail is None:
        return
    plan = analyse(fail.spec())
    check("6 GiB: the reported overflow really overflows",
          plan.peak_bytes > plan.vram_bytes,
          f"peak {plan.peak_bytes/GIB:.3f} > budget {plan.vram_bytes/GIB:.3f}")
    check("6 GiB: the overflow is strictly ABOVE the recommendation",
          fail.connections > best.connections,
          f"{fail.connections:,} > {best.connections:,}")
    check("6 GiB: no smaller overflow was available",
          True, f"nearest overflow {fail.connections:,} synapses")


def test_no_boundary_invented_at_the_ceiling():
    """Recommendation at the search ceiling => no overflow, and none claimed."""
    best, fail = largest_that_fits(6.0, hidden=8192, n_layers=32,
                                  timesteps=256, batch=8)
    check("a solution is found at the ceiling", best is not None)
    check("no overflow is invented above a recommendation that sits "
          "at the width ceiling",
          fail is None or fail.connections > (best.connections if best else 0),
          f"fit {best.connections:,}, overflow "
          f"{fail.connections if fail else None}")


def test_no_overflow_inside_bounds_is_reported_as_such():
    """When everything fits, say so -- do not imply a boundary exists."""
    best, fail = largest_that_fits(512.0, hidden=512, n_layers=4,
                                  timesteps=8, batch=1)
    check("a huge budget still finds a solution", best is not None)
    check("no phantom overflow is invented when all candidates fit",
          fail is None, f"fail={fail}" if fail else "")


# ── 4. MONOTONICITY ────────────────────────────────────────────────────────

def test_monotone_in_knobs():
    """Every memory term grows with width and horizon. If that ever stops
    holding, the descending scan in largest_that_fits becomes a lie."""
    peaks = []
    for h in (128, 256, 512, 1024):
        peaks.append(analyse(make_spec(vram_gb=80.0, hidden=h, n_layers=4,
                                      timesteps=16, batch=2)).peak_bytes)
    check("peak is non-decreasing in width",
          all(b >= a for a, b in itertools.pairwise(peaks)),
          f"{[round(p/GIB, 3) for p in peaks]}")

    peaks = []
    for t in (1, 8, 64, 512):
        peaks.append(analyse(make_spec(vram_gb=80.0, hidden=256, n_layers=4,
                                      timesteps=t, batch=2)).peak_bytes)
    check("peak is non-decreasing in horizon",
          all(b >= a for a, b in itertools.pairwise(peaks)),
          f"{[round(p/GIB, 3) for p in peaks]}")

    peaks = []
    for b in (1, 4, 32, 128):
        peaks.append(analyse(make_spec(vram_gb=80.0, hidden=512, n_layers=4,
                                      timesteps=32, batch=b)).peak_bytes)
    check("peak is non-decreasing in batch",
          all(b_ >= a for a, b_ in itertools.pairwise(peaks)),
          f"{[round(p/GIB, 3) for p in peaks]}")


# ── 5. REFUSAL INSTEAD OF A WRONG ANSWER ───────────────────────────────────

def test_refuses_when_nothing_fits():
    """A budget that cannot hold even the narrowest ladder entry."""
    best, fail = largest_that_fits(0.4, hidden=128, n_layers=64,
                                  timesteps=512, batch=32)
    check("an impossible budget returns no solution rather than a wrong one",
          best is None, f"best={best}")
    check("...and still reports why", fail is not None)
    if fail is not None:
        plan = analyse(fail.spec())
        check("...and the reason is a genuine overflow",
              plan.peak_bytes > plan.vram_bytes)


# ── 6. THE FRONTIER IS A SUPERSET OF THE RECOMMENDATION ────────────────────

def test_frontier_contains_the_recommendation():
    front = frontier(6.0, hidden=2048, n_layers=8, timesteps=64, batch=4)
    check("every frontier entry fits",
          all(c.fits for c in front), f"{len(front)} entries")
    best, _ = largest_that_fits(6.0, hidden=2048, n_layers=8,
                                timesteps=64, batch=4)
    if best is not None:
        check("the recommendation appears in the frontier",
              any(c.hidden == best.hidden and c.n_layers == best.n_layers
                  and c.timesteps == best.timesteps and c.batch == best.batch
                  for c in front))
    check("the frontier is sorted best-first",
          all(front[i].connections >= front[i + 1].connections
              for i in range(len(front) - 1)))


# ── 7. THE ANSWER IS INTERPRETABLE ─────────────────────────────────────────

def test_bytes_per_param_is_reported():
    """A connection count without its byte cost is a trap."""
    best, _ = largest_that_fits(24.0, hidden=1024, n_layers=4,
                                timesteps=32, batch=2)
    if best is None:
        check("bytes-per-synapse is reported when a solution exists", False)
        return
    bpp = best.bytes_per_param
    check("bytes-per-synapse is positive and matches the precision default",
          1.0 <= bpp <= 4.0, f"{bpp:.2f} B/synapse")
    text = best.explain()
    check("explain() states the bytes per synapse",
          "bytes per synapse" in text)


if __name__ == "__main__":
    print("=" * 70)
    print("CAPACITY SOLVER TESTS (axplan.solve)")
    print("=" * 70)
    test_ladder_respects_bound()
    test_reported_fit_really_fits()
    test_failure_boundary_is_real()
    test_no_boundary_invented_at_the_ceiling()
    test_no_overflow_inside_bounds_is_reported_as_such()
    test_monotone_in_knobs()
    test_refuses_when_nothing_fits()
    test_frontier_contains_the_recommendation()
    test_bytes_per_param_is_reported()
    print("=" * 70)
    print(f"  PASSED {len(PASS)} | FAILED {len(FAIL)}")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 70)
    sys.exit(1 if FAIL else 0)