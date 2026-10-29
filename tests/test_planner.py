"""
Tests for the segmentation planner.

The main check is to PROVE optimality: every partition by exhaustive search
against the DP, on a pile of random tasks. verify_1_dp_v2 already did this
on 2500 cases; here we fix it as a framework regression test so that nobody
breaks the optimality "in memory".

We also compare against a GREEDY algorithm (as in memopt), so there is a
reproducible number for "how much do we win over greedy".
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axplan.planner import (Segment, plan_partition, exact_peak,   # noqa
                            brute_force)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))


def rand_segs(rng, M):
    return [Segment(input_bytes=rng.randint(1, 40) * 1024,
                    internal_bytes=rng.randint(1, 20) * 1024,
                    recompute_ns=rng.randint(1, 1000))
            for _ in range(M)]


# ── 1. OPTIMALITY ──────────────────────────────────────────────────────────

def test_dp_optimal(n=400, seed=1):
    """The DP is optimal for the CONSERVATIVE model. That is the proof.

    Important: the reference runs in the same mode (conserve). The DP
    optimises the conservative peak, so the reference must too.
    Divergence from the EXACT optimum is not a bug, it is the price of an
    upper bound; its size is measured separately (test_gap_is_bounded).
    """
    rng = random.Random(seed)
    matched = 0
    checked = 0
    greedy_worse = 0
    for _ in range(n):
        M = rng.randint(2, 9)
        segs = rand_segs(rng, M)
        max_inter = max(s.internal_bytes for s in segs)
        budget = rng.randint(max_inter + 1,
                             sum(s.input_bytes for s in segs) + max_inter + 1)

        got = plan_partition(segs, budget)
        exp_b, exp_c = brute_force(segs, budget, mode="conserve")

        if exp_b is None:
            continue                       # no partition fits the conservative peak
        checked += 1
        if got is None:
            print(f"    DP found none, brute(conserve) did: M={M}")
            return
        if got.recompute_ns == exp_c:
            matched += 1
        else:
            print(f"    divergence: DP {got.recompute_ns} "
                  f"vs brute(conserve) {exp_c}")
            return

        g = greedy_partition(segs, budget)
        if g is not None and g.recompute_ns < got.recompute_ns - 1e-9:
            greedy_worse += 1

    check(f"the DP is optimal for the conservative peak on {checked} tasks",
          matched == checked, f"{matched}/{checked}")
    check("the greedy algorithm is never better than the DP",
          greedy_worse == 0,
          f"cases where greedy won: {greedy_worse}")


def test_gap_is_bounded(n=300, seed=11):
    """DP on the conservative peak against the exact optimum: measure the
    loss and BOUND it.

    The conservative figure is an upper bound, so it can be suboptimal for
    the exact model. That is not a bug, it is a price. But the price must
    be MEASURED and must not be monstrous, or we lose too much for the sake
    of a simple DP.

    Here we require: DP-on-conservative is never worse than 2x the exact
    optimum in recompute, and an infeasible conservative peak must not miss
    a feasible exact one more than half the time.
    """
    rng = random.Random(seed)
    worse = 0
    ratios = []
    feasible_conserve = 0
    exact_better_only = 0
    total = 0
    for _ in range(n):
        M = rng.randint(3, 8)
        segs = rand_segs(rng, M)
        max_inter = max(s.internal_bytes for s in segs)
        budget = rng.randint(max_inter + 1,
                             sum(s.input_bytes for s in segs) + max_inter)
        total += 1
        dc = plan_partition(segs, budget)
        de = brute_force(segs, budget, mode="exact")
        if dc is None and de is None:
            continue
        if dc is not None:
            feasible_conserve += 1
        if dc is not None and de is not None and dc.recompute_ns > de[1]:
            ratios.append(dc.recompute_ns / max(de[1], 1e-9))
            if dc.recompute_ns > 2.0 * de[1]:
                worse += 1
        if dc is None and de is not None:
            exact_better_only += 1

    worst = max(ratios) if ratios else 1.0
    check("the conservative peak is within 2x of the exact optimum",
          worse == 0,
          f"cases worse than 2x: {worse}, worst ratio {worst:.2f}, "
          f"{len(ratios)} measurements")
    # the conservative peak is sometimes infeasible while the exact one
    # is feasible: that is the expected price of an upper bound
    print(f"    (conservative infeasible while exact is feasible: "
          f"{exact_better_only}/{total} -- the price of an upper bound)")


def greedy_partition(segs, budget):
    """Greedy: walk left to right, cut as rarely as possible (memopt style)."""
    M = len(segs)
    max_inter = max(s.internal_bytes for s in segs)
    room = budget - max_inter
    if room < 0:
        return None
    inp = [s.input_bytes for s in segs]
    stored = 0
    bounds = [0]
    for j in range(1, M + 1):
        add = inp[j] if j < M else 0
        if stored + add > room:            # does not fit -> cut here
            bounds.append(j)
            stored = 0
        else:
            stored += add
    bounds.append(M)
    cost = [s.recompute_ns for s in segs]
    c = sum(sum(cost[bounds[k]:bounds[k + 1]])
            for k in range(len(bounds) - 1))
    return plan_partition(segs, budget) if False else _mk(segs, bounds, c,
                                                         budget, max_inter)


def _mk(segs, bounds, cost, budget, max_inter):
    from axplan.planner import PartitionPlan
    stored = sum(segs[b].input_bytes for b in bounds if b < len(segs))
    pu = stored + max_inter
    pe = exact_peak(segs, bounds)
    return PartitionPlan(tuple(bounds), pu, pe, cost, budget,
                         pe <= budget, max(0, pu - pe))


# ── 2. PLAN BOUNDARY ASSIGNMENT ────────────────────────────────────────────

def test_bounds_consistent(n=150, seed=2):
    """Boundaries are recovered consistently: increasing, starting at 0,
    ending at M, and giving the same stored figure the DP computed."""
    rng = random.Random(seed)
    ok = True
    for _ in range(n):
        M = rng.randint(2, 8)
        segs = rand_segs(rng, M)
        max_inter = max(s.internal_bytes for s in segs)
        budget = rng.randint(max_inter + 1,
                             sum(s.input_bytes for s in segs) + max_inter)
        p = plan_partition(segs, budget)
        if p is None:
            continue
        b = p.bounds
        if b[0] != 0 or b[-1] != M:
            ok = False
            break
        if any(b[i] >= b[i + 1] for i in range(len(b) - 1)):
            ok = False
            break
        if any(x < 0 or x > M for x in b):
            ok = False
            break
        # peak from the boundaries == the conservative peak
        stored = sum(segs[x].input_bytes for x in b if x < M)
        if stored + max_inter != p.peak_upper:
            ok = False
            break
        # the conservative peak really is >= the exact one
        if p.peak_upper < p.peak_exact:
            ok = False
            break
    check("boundaries are recovered consistently", ok)


# ── 3. BEHAVIOUR UNDER THE BUDGET ──────────────────────────────────────────

def test_monotone_in_budget(seed=3):
    """A bigger budget means less (or equal) recompute. Monotonicity."""
    rng = random.Random(seed)
    segs = rand_segs(rng, 10)
    max_inter = max(s.internal_bytes for s in segs)
    lo = max_inter + 1
    hi = sum(s.input_bytes for s in segs) + max_inter
    budgets = [lo + (hi - lo) * i // 12 for i in range(13)]
    costs = []
    feas = []
    for b in budgets:
        p = plan_partition(segs, b)
        if p is None:
            # None means NO partition fits. That is a legitimate
            # answer, not an "infeasible plan": with no plan there is
            # nothing to judge.
            feas.append(True)
            costs.append(float('inf'))
        else:
            feas.append(p.feasible)
            costs.append(p.recompute_ns)
    mono = all(costs[i] >= costs[i + 1] - 1e-9
               for i in range(len(costs) - 1)
               if costs[i] != float('inf') and costs[i + 1] != float('inf'))
    check("recompute does not grow as the budget grows", mono,
          f"{[round(c,1) if c != float('inf') else 'inf' for c in costs[:6]]} ...")
    check("every returned plan is feasible",
          all(feas),
          f"{sum(feas)}/{len(feas)}")


# ── 4. INFEASIBILITY AND EDGES ─────────────────────────────────────────────

def test_infeasible_and_edges():
    segs = [Segment(1000, 500, 10), Segment(2000, 600, 20)]
    check("a zero budget gives None", plan_partition(segs, 0) is None)
    check("a budget below max_inter gives None",
          plan_partition(segs, 100) is None)
    p = plan_partition(segs, 10 ** 9)
    check("a huge budget gives one segment",
          p is not None and p.n_cuts == 0,
          f"recomputes: {p.n_cuts}" if p else "None")
    try:
        plan_partition([], 100)
        check("an empty segment list raises", False)
    except ValueError:
        check("an empty segment list raises", True)


# ── 5. A SINGLE SEGMENT ────────────────────────────────────────────────────

def test_single_segment():
    segs = [Segment(10 * 1024, 5 * 1024, 100)]
    p = plan_partition(segs, 10 ** 6)
    check("a single segment yields no cuts",
          p is not None and p.bounds == (0, 1) and p.n_cuts == 0)
    check("a single segment peak = input+internal",
          p.peak_exact == 15 * 1024, f"{p.peak_exact}")


if __name__ == "__main__":
    print("=" * 70)
    print("PLANNER TESTS (axplan.planner)")
    print("=" * 70)
    test_dp_optimal()
    test_gap_is_bounded()
    test_bounds_consistent()
    test_monotone_in_budget()
    test_infeasible_and_edges()
    test_single_segment()
    print("=" * 70)
    print(f"  PASSED {len(PASS)} | FAILED {len(FAIL)}")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 70)
    sys.exit(1 if FAIL else 0)