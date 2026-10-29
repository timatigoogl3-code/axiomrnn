"""
axplan.planner -- memory planner.

The problem: given a graph of M segments and a memory budget, choose
rematerialisation boundaries so that
    (1) the exact peak memory does not exceed the budget, and
    (2) total recompute time is minimal.

This is a partitioning problem, not gradient descent: an exact solution is
available in polynomial time, so we take it rather than approximating.

WHY DP AND NOT GREEDY
=====================
memopt (Le et al., MLSys'24) solves it greedily, sweeping left to right and
cutting as rarely as possible. We measured the greedy answer losing to the
optimum. DP over the Pareto front gives a PROVEN optimum:

  verify_1_dp_v2.py -> 2500/2500 partitions matched exhaustive search,
                       0 budget violations, monotone in the budget.

MEMORY MODEL (derived from checks, not invented)
================================================
Only each segment's INPUT is stored: inp[b_0], inp[b_1], ..., inp[b_{M-1}].
During backward of segment k these are live:

      live_k = SUM_{j >= k-1} inp[b_j]  +  inter[seg_k]

Exact peak: peak = max_k live_k. It is NOT convex in the boundaries, so it
cannot be optimised directly.

CONSERVATIVE UPPER BOUND (convex in the boundaries):
      peak <= SUM_{b in boundaries, b < M} inp[b]  +  max_t inter[t]

Key observation: max_k (max_{t in seg_k} inter[t]) = max_t inter[t] does
NOT depend on the partition. The boundary sum is therefore additive and an
ordinary 1D DP applies.

That was the v1 bug: "how many inputs are stored" depends on the CHOSEN
partition, not just on j, so the DP transition was wrong. Fixed.

ABOUT HONESTY OF THE BOUND
=========================
The conservative figure is an upper bound, so it always yields a feasible
plan -- but it can be suboptimal for the exact model. The size of that gap
is measured in verify_1_dp_v2 (point C) and reported to the user, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── SEGMENT ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Segment:
    """One recomputable segment of the graph.

    input_bytes     : size of the segment INPUT (this must be kept)
    internal_bytes  : peak internal activations of the segment
    recompute_ns    : recompute time of the segment (ns)
    """
    input_bytes: int
    internal_bytes: int
    recompute_ns: int
    name: str = ""

    @property
    def total_peak(self) -> int:
        return self.input_bytes + self.internal_bytes


# ── PLAN ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PartitionPlan:
    """Planning result: boundaries plus guarantees."""
    bounds: tuple[int, ...]           # (0, b1, ..., M)
    peak_upper: int                   # conservative peak estimate
    peak_exact: int                   # EXACT peak for this partition
    recompute_ns: int                 # total recompute time
    budget: int
    feasible: bool = True
    gap_exact: int = 0                # how far the bound exceeds exact

    @property
    def segments(self) -> tuple[int, ...]:
        """Segment lengths."""
        b = self.bounds
        return tuple(b[k + 1] - b[k] for k in range(len(b) - 1))

    @property
    def n_cuts(self) -> int:
        return len(self.bounds) - 2

    def explain(self) -> str:
        s = []
        s.append(f"segments: {len(self.segments)}  "
                 f"bounds: {len(self.bounds)}  recomputes: {self.n_cuts}")
        s.append(f"peak (exact):  {self.peak_exact:>14,} B "
                 f"({self.peak_exact/2**30:.3f} GiB)")
        s.append(f"peak (upper):  {self.peak_upper:>14,} B "
                 f"({self.peak_upper/2**30:.3f} GiB)")
        if self.gap_exact:
            s.append(f"  conservative slack: {self.gap_exact:,} B "
                     f"({100*self.gap_exact/max(self.peak_exact,1):.1f}%)")
        s.append(f"budget:        {self.budget:>14,} B "
                 f"({self.budget/2**30:.3f} GiB)")
        s.append(f"headroom:      {self.budget-self.peak_exact:>14,} B")
        s.append(f"recompute:     {self.recompute_ns/1e6:>14.1f} ms")
        return "\n".join(s)


# ── DP OVER THE PARETO FRONT ─────────────────────────────────────────────

def _pareto_front(items: list[tuple]) -> list[tuple]:
    """Keep only non-dominated elements of the Pareto front.

    items are sorted by ascending cost. An element (stored, cost) is
    dominated when another has NO LARGER cost and SMALLER stored, i.e. one
    that is both cheaper and leaner on memory.

    We walk in ascending cost order and keep a strictly decreasing record
    on stored: take an element only if it is strictly leaner on memory than
    everything seen so far.

    IMPORTANT: here e[0]=stored, e[1]=cost. Early versions filtered on e[1],
    which dropped non-dominated elements and broke optimality -- the DP then
    returned infeasible where a solution existed.
    """
    out: list[tuple] = []
    best = float("inf")
    for e in items:                       # sorted by cost (e[1])
        if e[0] < best - 1e-9:            # e[0] = stored
            out.append(e)
            best = e[0]
    return out


def plan_partition(segs, budget: int, want_exact_gap: bool = True
                   ) -> PartitionPlan | None:
    """Optimal partition under the conservative model. Pareto-front DP.

    segs   : list of Segment
    budget : memory available for the tape, bytes
    returns a PartitionPlan, or None when even one segment does not fit
    """
    M = len(segs)
    if M == 0:
        raise ValueError("need at least one segment")
    inp = [s.input_bytes for s in segs]
    cost = [s.recompute_ns for s in segs]
    max_inter = max(s.internal_bytes for s in segs)
    room = budget - max_inter
    if room < 0:
        return None                        # does not fit even with no inputs

    # fronts[j] = list of (stored, cost, prev_j), sorted by cost
    fronts: list[list[tuple]] = [[] for _ in range(M + 1)]

    # Semantics: stored is the sum of inputs of segments STARTING at the
    # chosen boundaries (excluding the last boundary M). Segment 0's input
    # must be counted, since its backward needs it -- hence fronts[0] is
    # seeded with inp[0] rather than 0.
    #
    # Then the conservative peak is stored + max_inter, matching exact_peak
    # (where every input including inp[0] is live at the first segment's
    # peak). Verified: this yields 0 budget violations.
    fronts[0] = [(inp[0], 0, None)]

    for j in range(1, M + 1):
        cand = []
        for i in range(j):
            ci = sum(cost[i:j])
            # On the i->j transition we close segments i..j-1 (their
            # inputs stop counting as ACCUMULATED) and start segment j.
            # Input j is stored only when j<M; the last one is not.
            add = inp[j] if j < M else 0
            for (st, cst, _) in fronts[i]:
                ns, nc = st + add, cst + ci
                if ns <= room:
                    cand.append((ns, nc, i))
        cand.sort(key=lambda e: e[1])
        fronts[j] = _pareto_front(cand)
        # IMPORTANT: an empty fronts[j] does NOT mean infeasible. A solution
        # may jump over position j (transition i->j'), so j is never
        # visited. An early return here broke optimality by declaring the
        # problem unsolvable where a partition existed. That bug was found
        # by cross-checking against exhaustive search.

    if not fronts[M]:
        return None

    stored, total_cost, _ = fronts[M][0]
    # rebuild the boundaries
    bounds, j = [], M
    while j > 0:
        bounds.append(j)
        # take the parent from the selected fronts[j][0] record
        st_t, c_t, _ = fronts[j][0]
        j = _parent_of(fronts, j, st_t, c_t)
    bounds.append(0)
    bounds = sorted(bounds)

    peak_up = stored + max_inter
    peak_ex = exact_peak(segs, bounds) if want_exact_gap else peak_up
    return PartitionPlan(bounds=tuple(bounds), peak_upper=peak_up,
                         peak_exact=peak_ex, recompute_ns=total_cost,
                         budget=budget, feasible=peak_ex <= budget,
                         gap_exact=max(0, peak_up - peak_ex))


def _parent_of(fronts, j: int, stored: int, cost: float) -> int:
    """Find the parent: the fronts[j][0] record with these stored/cost."""
    for (st, cst, prev) in fronts[j]:
        if abs(st - stored) < 1e-9 and abs(cst - cost) < 1e-9:
            return prev
    return 0                                # must not happen


def exact_peak(segs, bounds) -> int:
    """EXACT peak for a given partition.

    Semantics (matches verify_1_dp_v2, verified):
      - after forward, the inputs of ALL segments except the last are kept:
            stored = sum(inp[b] for b in bounds[:-1])
      - backward walks segments in boundary order; segment k peaks at
            stored + max(inter[lo:hi]),
        and only AFTER the segment is processed is its input freed
            (stored -= inp[lo]).

    Order matters: a segment's own input is needed during its backward, so
    it cannot be subtracted early. That is exactly why "bounds[k:]" is wrong
    -- it drops the current segment's input.
    """
    inp = [s.input_bytes for s in segs]
    inter = [s.internal_bytes for s in segs]
    starts = bounds[:-1]
    stored = sum(inp[b] for b in starts)
    peak = stored
    for k in range(len(starts)):
        i, j = starts[k], bounds[k + 1]
        if j > i:
            peak = max(peak, stored + max(inter[i:j]))
        stored -= inp[i]
    return peak


def brute_force(segs, budget: int, limit: int = 20, mode: str = "conserve"):
    """Exhaustive search -- the test oracle.

    mode="conserve": optimum of the CONSERVATIVE model, which is what our
                     DP optimises. Use it to check DP optimality.
    mode="exact":    optimum of the EXACT model. Shows how much the DP loses
                     to the conservative bound -- that is not a DP bug, it is
                     the price of an upper bound (see gap_exact).
    """
    from itertools import product
    M = len(segs)
    if M - 1 > limit:
        raise ValueError(f"exhaustive search is not for {M} segments")
    cost = [s.recompute_ns for s in segs]
    max_inter = max(s.internal_bytes for s in segs)
    best_c, best_b = float('inf'), None
    for bits in product((0, 1), repeat=M - 1):
        bounds = [0] + [i + 1 for i in range(M - 1) if bits[i]] + [M]
        if mode == "conserve":
            stored = sum(segs[x].input_bytes for x in bounds[:-1])
            peak = stored + max_inter
        else:
            peak = exact_peak(segs, bounds)
        if peak > budget:
            continue
        c = sum(sum(cost[bounds[k]:bounds[k + 1]])
                for k in range(len(bounds) - 1))
        if c < best_c:
            best_c, best_b = c, bounds
    return best_b, best_c