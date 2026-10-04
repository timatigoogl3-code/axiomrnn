"""
02_budget.py -- the budget as an INPUT to design.

Run (TensorFlow is NOT needed):
    python3 examples/02_budget.py

════════════════════════════════════════════════════════════════════════
WHY THIS EXAMPLE
════════════════════════════════════════════════════════════════════════
The first example showed that a network learns. This one is about
something else.

Usually you learn about memory AFTER the fact: you assemble, you train,
it dies with OOM, you shrink it, you retrain. axiomrnn reverses the
order: you say how much memory you have, and you get back an answer about
what can be learned with it.

How this differs from the alternatives:

  Accelerate  takes max_memory, but only for inference
  JAX         demands a manual commitment
  PyTorch     measures post-factum
  ONNX        does not accept a budget at all

Here the budget is the input and the answer is human-readable: "at 6 GB
there is 4.12 usable, this network needs 1.57, credit horizon 66".

════════════════════════════════════════════════════════════════════════
WHY THIS FILE RUNS WITHOUT TENSORFLOW
════════════════════════════════════════════════════════════════════════
That is not a coincidence, it is a check of an architectural boundary.
`axplan` does not import TF at all, so the planner runs on a machine with
no GPU stack. Verify it: run this file with the same system `python3`
that has no TF installed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import axiomrnn as ax
from axplan.planner import Segment, plan_partition

try:
    import tensorflow  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False

GIB = 1 << 30


def gib(n: int) -> str:
    return f"{n / GIB:.2f} GiB"


print("=" * 74)
print("THE BUDGET AS AN INPUT")
print("=" * 74)
print(f"  TensorFlow in this environment: "
      f"{'present' if HAS_TF else 'ABSENT, and not needed'}")
print()

# ═══════════════════════════════════════════════════════ 1. WHAT IT COSTS ═══
spec = ax.Model()
spec.add_spiking(1024, horizon=256)
spec.add_spiking(1024, horizon=256)
spec.add_dense(1000)

print("  Network: 2 spiking layers of 1024, output into 1000 classes, T=256")
print()
print("  " + "-" * 72)
print(f"  {'budget':<20}{'usable':>12}{'needed':>12}{'fits?':>10}")
print("  " + "-" * 72)

for b in (ax.Budget.consumer_6gb(), ax.Budget.pro_24gb(),
          ax.Budget.datacenter_80gb()):
    p = spec.fit_budget(b, batch=32)
    print(f"  {b.name:<20}{gib(b.usable_bytes()):>12}{gib(p.peak_bytes):>12}"
          f"{'yes' if not p.infeasible else 'NO':>10}")
print()
print("  The \"consumer 6 GB\" profile really has 4.12 GiB usable, not six:")
print("  CUDA context, cuBLAS workspaces, allocator fragmentation. The")
print("  planner knows this and promises no more. The number above is")
print("  measured, not a label.")
print()

# ═══════════════════════════════════════════════════ 2. WHEN THE BUDGET BITES ══
#
# Width grows quadratically in the weight count, so a network hits the
# ceiling quickly. We show exactly where it breaks, on the planner's REAL
# numbers rather than an invented toy.
print("  " + "-" * 72)
print("  Now the width grows -- and the budget starts to matter.")
# Stated as a rule, not as a number copied out of the table below, because the
# table moved when the BPTT tape stopped being capped at the credit horizon and
# a hardcoded boundary is how "fits up to 16384" outlived the arithmetic.
print("  Notice: the wall arrives somewhere between width 4096 and 12288,")
print("  and the planner finds it on its own rather than being told.")
print()
print(f"  {'width':<10}{'needed GiB':>12}{'6 GB':>9}{'80 GB':>9}")

rows = []
for width in (1024, 4096, 12288, 16384, 20480, 24576):
    s = ax.Model()
    s.add_spiking(width, horizon=256)
    s.add_spiking(width, horizon=256)
    s.add_dense(1000)
    p6 = s.fit_budget(ax.Budget.consumer_6gb(), batch=32)
    p80 = s.fit_budget(ax.Budget.datacenter_80gb(), batch=32)
    rows.append((width, p6))
    print(f"  {width:<10}{p6.peak_bytes / GIB:>12.2f}"
          f"{'yes' if not p6.infeasible else 'NO':>9}"
          f"{'yes' if not p80.infeasible else 'NO':>9}")

# Take the failure point FROM THE PLANNER rather than rounding by eye.
first_bad = next((w for w, p in rows if p.infeasible), None)
print()
if first_bad is not None:
    # The last width that fit, read back out of the table rather than written
    # here. When the BPTT cap was fixed this line used to name 16384, which had
    # stopped being true, and nothing caught it because the sweep above it was
    # correct and the sentence below it was not.
    ok_widths = [w for w, p in rows if not p.infeasible]
    last_ok = max(ok_widths) if ok_widths else None
    print("  The planner found on its own that at 6 GB the network stops")
    print(f"  fitting from width {first_bad}. Width {last_ok} works -- and")
    print("  that was computed, not eyeballed.")
print()
print("  One and the same line of code. Different budgets give different")
print("  sizes that fit.")
print()

# ══════════════════════════════════════════════ 3. THE PLANNER: WHAT FITS ════
#
# The graph is cut into segments, and for each one a decision is made:
# store or recompute during backward. The choice is made by a DP over the
# Pareto front, not greedily (greedy takes the cheapest step and walks into
# a trap).
#
# What is shown here is REAL behaviour, including where the planner
# refuses. That matters more than a pretty picture.
#
# The cost is honest: each segment has an INPUT (which must be stored to
# reproduce the segment) and INTERNAL activations (the forward peak).
segs = [
    Segment(input_bytes=3 * GIB, internal_bytes=0.5 * GIB,
            recompute_ns=900_000_000, name="attention-1"),
    Segment(input_bytes=3 * GIB, internal_bytes=0.5 * GIB,
            recompute_ns=400_000_000, name="attention-2"),
    Segment(input_bytes=3 * GIB, internal_bytes=0.5 * GIB,
            recompute_ns=300_000_000, name="MLP"),
    Segment(input_bytes=3 * GIB, internal_bytes=0.4 * GIB,
            recompute_ns=200_000_000, name="readout"),
]

print("  Graph segments (input = stored, internal = forward peak):")
for s in segs:
    print(f"    {s.name:<14} input {gib(s.input_bytes):>9}   "
          f"internal {gib(s.internal_bytes):>9}   "
          f"recompute {s.recompute_ns / 1e9:5.2f} s")
print()
print("  Counting from the requirement: all inputs together are 12 GiB, and")
print("  the budget may be smaller. The planner looks for a partition under")
print("  which everything fits.")
print()
print(f"  {'budget':<10}{'peak':>10}{'cuts':>11}{'recompute':>12}{'fits':>9}")
print("  " + "-" * 52)

for gb in (13.0, 8.0, 4.0, 3.0, 2.0):
    plan = plan_partition(segs, int(gb * GIB))
    if plan is None:
        print(f"  {gb:>4.1f} GiB{'':>4}{'-':>10}{'-':>11}{'-':>12}"
              f"{'NO':>9}")
        continue
    print(f"  {gb:>4.1f} GiB{'':>4}{plan.peak_exact / GIB:>8.2f} GiB"
          f"{plan.n_cuts:>11}{plan.recompute_ns / 1e9:>10.2f} s"
          f"{'yes' if plan.feasible else 'NO':>9}")
print()
print("  Read it like this: at a 13 GiB budget you can store everything and")
print("  recompute nothing (0 cuts). At 8 GiB and 4 GiB, still nothing to")
print("  recompute: storing only the graph input plus the peak of one")
print("  segment is enough. At 3 GiB and below there is no solution at all:")
print("  the graph input alone is 3 GiB and cannot be compressed.")
print()
print("  Note the 0 cuts everywhere. That is NOT the planner failing to")
print("  work -- it is the answer. The peak is stored + max_internal, and")
print("  max_internal does not move when you cut, while stored only grows")
print("  because every cut keeps another boundary tensor alive. So cutting")
print("  the graph can only make the peak larger, and the true optimum is")
print("  always \"no cuts\". examples/advanced/05_segmentation_planner.py")
print("  shows both halves of that, including the arithmetic.")
print()

# ═════════════════════════════════════════════════ 4. WHERE THE PLANNER FAILS ═══
#
# Found while writing this example, and the limitation is substantial:
#
#   room = budget - max(segment internal activations)
#
# So INSIDE a segment the planner does not cut. If one segment's internals
# exceed the budget the answer is None, and no partition helps: you have
# to restructure the model yourself, splitting that piece into smaller
# layers.
#
# The bottom rows of the table above are exactly this case: the graph input
# (3 GiB) has to be stored, and at a 2 GiB budget there is no answer at all.
big = [Segment(input_bytes=0.4 * GIB, internal_bytes=6.0 * GIB,
               recompute_ns=900_000_000, name="huge-attention")]
p7 = plan_partition(big, int(7 * GIB))
p4 = plan_partition(big, int(4 * GIB))
print(f"  A segment with 6 GiB of internals at a 7 GiB budget: "
      f"{'allowed' if p7 else 'NO'}")
print(f"  The same at a 4 GiB budget: "
      f"{'allowed' if p4 else 'NO, and no partition helps'}")
print()

# ══════════════════════════════════════════════════════ 5. HONEST LIMITS ══
print("  " + "-" * 72)
print("  HONESTLY ABOUT WHAT THIS CAN DO")
print("  " + "-" * 72)
print("""
  THE PLANNER:
    - does NOT scale the network automatically. It costs what you already
      described and tells you whether it fits;
    - does NOT cut inside a segment (see point 4);
    - is optimal, verified against exhaustive search on 307 tasks, and the
      greedy algorithm wins on none of them.

  THE MEMORY MODEL:
    - was checked by measurement on an RTX 3050: plan/actual = 0.99;
    - on DIFFERENT hardware the ratio will be its own thing. Measure, do
      not trust.

  WHAT WE DO NOT KNOW (and do not invent):
    - the sparse/bit crossover point, measured by nobody, us included;
    - what a spiking network really costs in energy. We have no such
      hardware here to measure it on;
    - whether a spiking network is faster on a GPU. We measured parity in
      quality (0.994 against 0.999) and did NOT measure superiority.
""")

# ══════════════════════════════════════════════════════════ THE FULL REPORT ══
# verbose=False: explain() prints by default AND returns the report, so
# print(explain(...)) emitted the whole thing twice, identically.
print("  " + "=" * 72)
print(spec.explain(ax.Budget.consumer_6gb(), batch=32, verbose=False))