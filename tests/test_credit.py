"""
TESTS FOR THE CREDIT RULES (axplan.credit).

The point here is NOT to check the arithmetic (it is simple), but to check
that we do NOT invent numbers where we have no data.

The most common error in this project was not in the maths but in
substituting unknown values: spike_rate=0.08 was invented. The same trap
sits at extrapolating quality beyond the measured range.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axplan.credit import (
    ADAM_BYTES_PER_PARAM,
    TRACE_BYTES,
    CreditRule,
    NeuronModel,
    compare,
    credit_bytes,
    crossover_batch,
    local_quality,
)

PASS, FAIL = [], []
GIB = 1 << 30


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))


# ── 1. THE BYTES MATCH THE DERIVATION ─────────────────────────────────────

def test_trace_bytes_match_source():
    """Trace bytes match the derivation in the report."""
    check("LIF = 4 B/synapse", TRACE_BYTES[NeuronModel.LIF] == 4)
    check("ALIF = 8 B/synapse", TRACE_BYTES[NeuronModel.ALIF] == 8)
    check("ALIF+reward = 12 B/synapse",
          TRACE_BYTES[NeuronModel.ALIF_REWARD] == 12)
    check("Adam = 8 B/param", ADAM_BYTES_PER_PARAM == 8)
    check("ALIF is exactly 2x LIF",
          TRACE_BYTES[NeuronModel.ALIF] == 2 * TRACE_BYTES[NeuronModel.LIF])


# ── 2. THE MAIN PROPERTY: INDEPENDENT OF T ────────────────────────────────

def test_local_independent_of_T():
    """The local rule must NOT grow with the horizon. That is its point."""
    conn = 12 * 768 * 768
    vals = []
    for T in [8, 64, 512, 4096]:
        p = credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, conn, T,
                         32, 768, 12)
        vals.append(p.bytes_total)
    check("the local rule is independent of T",
          len(set(vals)) == 1,
          f"T=8..4096: {vals[0]:,} B always")

    # and BPTT, on the contrary, MUST grow
    bp = [credit_bytes(CreditRule.BPTT, NeuronModel.LIF, conn, T, 32,
                       768, 12).bytes_total for T in [8, 64, 512]]
    check("BPTT grows with T", bp[2] > bp[1] > bp[0],
          f"T=8:{bp[0]:,}  T=64:{bp[1]:,}  T=512:{bp[2]:,}")


# ── 3. THE PRICE IN BYTES vs ADAM ─────────────────────────────────────────

def test_marginal_vs_adam():
    """Key finding: the local rule costs <= the price of momentum."""
    conn = 70_000_000
    c = compare(NeuronModel.LIF, connections=conn, timesteps=64,
                batch=32, hidden=768, n_layers=12)
    check("the local rule is <= 1.5x Adam",
          c.marginal_vs_adam <= 1.5,
          f"{c.marginal_vs_adam:.2f}x Adam "
          f"({c.local.bytes_total / GIB:.2f} GiB against "
          f"{c.adam_bytes / GIB:.2f} GiB)")
    # There is a saving if and only if BPTT costs more than the local
    # rule. Here it does NOT pay (BPTT 0.071 against local 0.261), and the
    # comparison must admit that rather than invent a saving.
    cheaper_local = c.local.bytes_total < c.bptt.bytes_total
    check("the comparison invents no saving when there is none",
          (c.saving_bytes > 0) == cheaper_local,
          f"BPTT {c.bptt.bytes_total / GIB:.3f} against local "
          f"{c.local.bytes_total / GIB:.3f} GiB; saving = "
          f"{c.saving_bytes / GIB:+.3f} GiB")


# ── 4. THE CROSSOVER BATCH (what the planner most needs) ──────────────────

def test_crossover_is_meaningful():
    """The crossover batch FALLS as the horizon grows.

    The first version of this test asserted the opposite, and that was
    wrong:
        crossover = local_total / bptt_per_batch,  and bptt_per_batch ~ T
    so the LONGER the horizon, the SMALLER the batch needed to switch to
    the local rule. That is the substantive point: a long horizon makes
    BPTT unaffordable.

    That is why growing T is not only a quality question, but the one
    situation where the local rule pays off in memory.
    """
    L, h = 12, 768
    Ts = [16, 64, 256]
    xs = [crossover_batch(T, NeuronModel.LIF, hidden=h, n_layers=L)
          for T in Ts]
    check("the crossover is finite and positive", all(0 < x < 1e6 for x in xs),
          "batch* = " + ", ".join(f"T={t}:{x:.0f}" for t, x in zip(Ts, xs)))
    check("the crossover FALLS as the horizon grows", xs[0] > xs[1] > xs[2],
          " > ".join(f"{x:.1f}" for x in xs)
          + "  (BPTT grows like T, local does not)")

    # below the crossover BPTT is cheaper; above it, the local rule is
    conn = L * h * h
    lo = max(1, int(xs[0]) + 2)       # above the crossover at T=16
    hi = max(1, int(xs[2]) - 1)       # below the crossover at T=256
    c_lo = compare(NeuronModel.LIF, connections=conn, timesteps=16,
                   batch=lo, hidden=h, n_layers=L)
    c_hi = compare(NeuronModel.LIF, connections=conn, timesteps=256,
                   batch=hi, hidden=h, n_layers=L)
    check("above the crossover the local rule is cheaper",
          c_lo.local.bytes_total < c_lo.bptt.bytes_total,
          f"T=16 batch={lo}: local {c_lo.local.bytes_total / GIB:.4f} < "
          f"BPTT {c_lo.bptt.bytes_total / GIB:.4f} GiB")
    check("below the crossover BPTT is cheaper",
          c_hi.bptt.bytes_total < c_hi.local.bytes_total,
          f"T=256 batch={hi}: BPTT {c_hi.bptt.bytes_total / GIB:.4f} < "
          f"local {c_hi.local.bytes_total / GIB:.4f} GiB")


# ── 5. HONESTY: NO EXTRAPOLATION ─────────────────────────────────────────

def test_no_extrapolation():
    """Outside the measured range: None, not an invented number."""
    check("T=60 gives the measured 0.855",
          abs(local_quality(60) - 0.855) < 1e-6, f"{local_quality(60)}")
    check("T=500 gives the measured 0.389",
          abs(local_quality(500) - 0.389) < 1e-6, f"{local_quality(500)}")
    check("T=1024 -> None (no data)", local_quality(1024) is None,
          "and not an extrapolation")
    check("T=100000 -> None", local_quality(100_000) is None)
    mid = local_quality(200)
    check("interpolation between points", 0.389 < mid < 0.855, f"T=200: {mid:.3f}")


# ── 6. BPTT IS THE REFERENCE ─────────────────────────────────────────────

def test_bptt_is_reference():
    conn = 1000
    p = credit_bytes(CreditRule.BPTT, NeuronModel.LIF, conn, 64, 32, 768, 12)
    check("BPTT is the quality reference at 1.0", p.quality_kept == 1.0)
    check("BPTT is flagged T-dependent", p.depends_on_T)
    check("the local rule has depends_on_T=False",
          not credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, conn, 64,
                           32, 768, 12).depends_on_T)


# ── 7. CREDIT TRUNCATION IS FREE ─────────────────────────────────────────

def test_credit_truncation_shrinks_bptt():
    """Truncated credit reduces BPTT memory, and is free in quality."""
    conn = 12 * 768 * 768
    full = credit_bytes(CreditRule.BPTT, NeuronModel.LIF, conn, 512, 32,
                        768, 12).bytes_total
    cut = credit_bytes(CreditRule.BPTT, NeuronModel.LIF, conn, 512, 32,
                       768, 12, credit_k=66).bytes_total
    check("credit truncation reduces BPTT memory", cut < full,
          f"full {full / GIB:.3f} GiB -> truncated {cut / GIB:.3f} GiB "
          f"({cut / full:.1%})")
    loc = credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, conn, 512, 32,
                       768, 12, credit_k=66).bytes_total
    check("truncation does not affect the local rule",
          loc == credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, conn, 512,
                              32, 768, 12).bytes_total)


if __name__ == "__main__":
    print("=" * 74)
    print("CREDIT RULE TESTS (axplan.credit)")
    print("=" * 74)
    test_trace_bytes_match_source()
    test_local_independent_of_T()
    test_marginal_vs_adam()
    test_crossover_is_meaningful()
    test_no_extrapolation()
    test_bptt_is_reference()
    test_credit_truncation_shrinks_bptt()
    print("=" * 74)
    print(f"  PASSED {len(PASS)} | FAILED {len(FAIL)}")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 74)
    sys.exit(1 if FAIL else 0)