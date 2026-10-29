"""
axplan.solve -- the inverse question.

Everything else in axplan answers "what does THIS network cost?". This module
answers the question a practitioner actually has: **"I have this much memory.
What is the largest network I can train?"**

WHY THIS EXISTS
===============
The package's value is the planner, not the spiking. A planner that only
reports the peak of a fixed architecture leaves the user to guess which knob
to turn. This module searches the knob space under the verified memory model
and reports the boundary exactly, together with the reason it is a boundary.

WHAT IT DELIBERATELY DOES NOT DO
================================
It does not rank quality. There is **no verified quality model in this
package**, and inventing one would be the single most damaging thing this
codebase could do -- see UX_RULES.md 10. The one quality figure that does
exist is the price of the local credit rule, `credit.local_quality()`, which
is read from a cited measurement and returns None outside its data range
rather than extrapolating. That function is used here, and only there.

So the answer is a CAPACITY answer, stated as such: the largest configuration
whose predicted peak fits, plus the first configuration that does not fit.
The second half matters. A tool that only reports what fits cannot be
distinguished from a tool that is simply optimistic.

THE BOUNDARY IS CHECKED, NOT ASSUMED
=====================================
Feasibility is monotone in `hidden` (every memory term grows with width), so a
binary search would be valid. The implementation scans anyway: monotonicity is
an assumption about the model, and an assumption that silently stops holding
would turn the search into a lie. The scan is cheap enough that correctness is
the better trade.
"""
from __future__ import annotations

from dataclasses import dataclass

from .memory import BudgetSpec, MemoryPlan, Precision, analyse

# ── CANDIDATE ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    """One point in the knob space, with the memory verdict attached."""
    hidden: int
    n_layers: int
    timesteps: int
    batch: int
    credit_rule: str
    connections: int
    peak_bytes: int
    vram_bytes: int
    vram_gb: float = 6.0        # kept so the candidate can rebuild its own spec
    binding: str = ""          # which term dominated the peak
    plan: MemoryPlan | None = None

    @property
    def fits(self) -> bool:
        return self.peak_bytes <= self.vram_bytes

    @property
    def bytes_per_param(self) -> float:
        """What a synapse actually costs in this configuration.

        Reported because a bare connection count is meaningless without it:
        2.1e9 synapses at 2 bytes is 4 GiB, at 4 bytes it is 8 GiB, and at
        1 bit it is 0.25 GiB. The number without this is a trap.
        """
        if not self.connections:
            return 0.0
        static = self.plan.static_bytes if self.plan else 0
        return static / self.connections

    @property
    def headroom(self) -> int:
        return self.vram_bytes - self.peak_bytes

    def spec(self, **kw) -> BudgetSpec:
        """The BudgetSpec this candidate was evaluated as."""
        return make_spec(vram_gb=self.vram_gb, hidden=self.hidden,
                         n_layers=self.n_layers,
                         timesteps=self.timesteps, batch=self.batch,
                         credit_rule=self.credit_rule, **kw)

    def explain(self) -> str:
        g = 2 ** 30
        return "\n".join([
            f"  hidden            {self.hidden:>10,}".replace(",", " "),
            f"  layers            {self.n_layers:>10d}",
            f"  horizon           {self.timesteps:>10d}",
            f"  batch             {self.batch:>10d}",
            f"  connections       {self.connections:>10,}".replace(",", " "),
            f"  bytes per synapse {self.bytes_per_param:>10.2f}",
            f"  peak              {self.peak_bytes/g:>10.3f} GiB",
            f"  budget            {self.vram_bytes/g:>10.3f} GiB",
            f"  headroom          {self.headroom/g:>10.3f} GiB",
            f"  binding term      {self.binding:>10}",
        ])


# ── SPEC CONSTRUCTION ────────────────────────────────────────────────────

def make_spec(*, vram_gb: float, hidden: int, n_layers: int, timesteps: int,
              batch: int = 1, credit_rule: str = "bptt", vocab: int = 32000,
              seq_len: int = 1, n_heads: int = 1, spike_rate: float = 0.25,
              spatial: int = 1, precision: Precision | None = None,
              neuron_model: str = "lif") -> BudgetSpec:
    """BudgetSpec for a candidate architecture.

    `connections` follows the same convention as Model.fit_budget: a stack of
    `n_layers` spiking layers of width `hidden` has sum(units^2 + units)
    synapses, which is n_layers * hidden * (hidden + 1).
    """
    kw = {} if precision is None else {"precision": precision}
    return BudgetSpec(
        vram_gb=vram_gb,
        connections=n_layers * hidden * (hidden + 1),
        n_layers=n_layers,
        hidden=hidden,
        timesteps=timesteps,
        vocab=vocab,
        seq_len=seq_len,
        batch=batch,
        n_heads=n_heads,
        credit_rule=credit_rule,
        neuron_model=neuron_model,
        spike_rate=spike_rate,
        spatial=spatial,
        **kw,
    )


def _binding_term(plan: MemoryPlan) -> str:
    """Name the largest contributor to the peak. The actionable bit."""
    terms = {
        "static": plan.static_bytes,
        "tape": plan.tape_bytes,
        "backward": plan.backward_bytes,
        "logits": plan.logits_bytes,
        "workspace": plan.workspace_bytes,
    }
    return max(terms.items(), key=lambda kv: kv[1])[0]


def _evaluate(vram_gb: float, hidden: int, n_layers: int, timesteps: int,
              batch: int, credit_rule: str, **kw) -> tuple[Candidate, MemoryPlan]:
    spec = make_spec(vram_gb=vram_gb, hidden=hidden, n_layers=n_layers,
                     timesteps=timesteps, batch=batch,
                     credit_rule=credit_rule, **kw)
    plan = analyse(spec)
    cand = Candidate(
        hidden=hidden, n_layers=n_layers, timesteps=timesteps, batch=batch,
        credit_rule=credit_rule,
        connections=spec.connections,
        peak_bytes=plan.peak_bytes,
        vram_bytes=plan.vram_bytes,
        vram_gb=vram_gb,
        binding=_binding_term(plan),
        plan=plan,
    )
    return cand, plan


# ── SEARCH ───────────────────────────────────────────────────────────────

def largest_that_fits(vram_gb: float, *, hidden: int = 8192, n_layers: int = 64,
                      timesteps: int = 512, batch: int = 32,
                      credit_rule: str = "bptt", **kw
                      ) -> tuple[Candidate | None, Candidate | None]:
    """The biggest configuration that fits, and the smallest that does not.

    Each argument is an UPPER bound of the search, not a fixed value: width is
    scanned downward, depth / horizon / batch are enumerated. Returns
    `(largest_fitting, first_not_fitting)`; either may be None, and both None
    means the smallest searched configuration already overflows.

    Depth, horizon and batch are enumerated in ascending order and the widest
    fitting width is taken for each, so the result is the maximum
    connection count over the searched space. Ties are broken toward the
    larger horizon, because a longer horizon is the axis this package has
    evidence about (UX_RULES 10: measured beats assumed).
    """
    best_fit: Candidate | None = None
    overflows: list[Candidate] = []

    for T in _ascending(timesteps):
        for L in _ascending(n_layers):
            for b in _ascending(batch):
                # widths descend, so the LAST width that failed is the
                # narrowest overflow -- i.e. the configuration immediately
                # above the boundary. Everything wider also failed.
                fit_here: Candidate | None = None
                last_fail: Candidate | None = None
                for h in _descending(hidden):
                    cand, _ = _evaluate(vram_gb, h, L, T, b, credit_rule, **kw)
                    if cand.fits:
                        fit_here = cand
                        break
                    last_fail = cand
                if fit_here is not None and (best_fit is None or
                        (_capacity(fit_here), fit_here.timesteps) >
                        (_capacity(best_fit), best_fit.timesteps)):
                    best_fit = fit_here
                if last_fail is not None:
                    overflows.append(last_fail)

    # The boundary is only meaningful ABOVE the recommendation. An overflow
    # with fewer synapses than the best fit is not a boundary -- it is a
    # different axis (say batch=32 with a long horizon) overflowing while
    # some other axis fits far larger. Reporting it as "the first that does
    # not fit" would contradict the recommendation printed next to it.
    floor = _capacity(best_fit) if best_fit is not None else -1
    above = [c for c in overflows if _capacity(c) > floor]
    first_fail = min(above, key=_capacity) if above else None
    return best_fit, first_fail


def _capacity(c: Candidate) -> int:
    """What 'bigger' means here: synapse count. Stated, not assumed silently."""
    return c.connections


def _ascending(limit: int) -> list[int]:
    """1, 2, 4, ... up to `limit`, with `limit` itself always included.

    The bound is part of the contract: an earlier version returned a fixed
    ladder and silently searched past the caller's ceiling, which would have
    made a "largest that fits" answer refer to configurations the caller
    explicitly excluded.
    """
    out = [1]
    while out[-1] * 2 <= limit:
        out.append(out[-1] * 2)
    if out[-1] != limit:
        out.append(limit)
    return out


def _descending(limit: int, min_hidden: int = 128) -> list[int]:
    """Widths from `limit` down, never below `min_hidden`."""
    out = [x for x in _ascending(limit) if x >= min_hidden]
    return out[::-1] or [limit]


def frontier(vram_gb: float, *, hidden: int = 4096, n_layers: int = 16,
             timesteps: int = 128, batch: int = 8,
             credit_rule: str = "bptt", **kw) -> list[Candidate]:
    """Every fitting candidate in the searched space, best first.

    Kept for inspection: the recommended point is a product of an objective,
    and a user with a different objective needs the whole set to see that the
    recommendation is not the only option.
    """
    out = []
    for T in _ascending(timesteps):
        for L in _ascending(n_layers):
            for b in _ascending(batch):
                for h in _descending(hidden):
                    cand, _ = _evaluate(vram_gb, h, L, T, b, credit_rule, **kw)
                    if cand.fits:
                        out.append(cand)
    out.sort(key=lambda c: (-c.connections, -c.timesteps))
    return out


def report(vram_gb: float, **kw) -> str:
    """Human-readable answer, including the failure boundary."""
    best, fail = largest_that_fits(vram_gb, **kw)
    L = []
    L.append(f"CAPACITY OF A {vram_gb:.0f} GiB BUDGET")
    L.append("=" * 62)
    L.append("")
    if best is None:
        L.append("  NOTHING in the searched space fits, including the")
        L.append("  narrowest configuration. Lower the batch or the horizon,")
        L.append("  or accept offloading.")
    else:
        L.append("  LARGEST THAT FITS")
        L.append(best.explain())
    if fail is not None:
        L.append("")
        L.append("  FIRST THAT DOES NOT FIT (the boundary is real, not assumed)")
        L.append(fail.explain())
        ratio = fail.peak_bytes / max(fail.vram_bytes, 1)
        L.append(f"    overshoot {ratio:.2f}x the budget")
    elif best is not None:
        L.append("")
        L.append("  No overflow inside the searched space: every candidate")
        L.append("  fits. Raise the width/depth/horizon bounds to find the")
        L.append("  boundary rather than assuming there is none.")
    L.append("")
    L.append("  This is a CAPACITY answer. It does not rank quality: this")
    L.append("  package has no verified quality model, and inventing one")
    L.append("  would be worse than admitting it.")
    return "\n".join(L)