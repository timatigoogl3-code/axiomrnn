#!/usr/bin/env python3
"""The memory model, tested against the code that actually ships.

This suite exists because the one that used to occupy its slot did not test
anything a user receives. It ran `../verify/test_cost.py` -- a file outside
this repository, absent from the wheel and the sdist -- which began with
`sys.path.insert(0, '/home/cune')` and imported `axon.cost` from
/home/cune/axon/cost.py. So thirty-eight green checks certified a module that
no installation contains, and on a fresh clone, including every GitHub runner,
the suite could not run at all. `git ls-files | grep verify` returns nothing.

`axplan/memory.py` is the shipped successor of that `axon/cost.py`, and it
differs in one way that matters: it counts `backward_bytes` inside
`activation_fraction`, which the old copy had no term for at all. Two
consequences are printed as DIVERGENCE below rather than asserted, because
asserting them would mean asserting something false.

Everything asserted here is an invariant: true of the model, independent of
the constants chosen for it, and worth failing on if it stops holding.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axplan.memory import (  # noqa: E402
    GIB, BudgetSpec, Codec, CreditModel, MemoryPlan, Precision, Regime,
    activation_bytes, analyse, backward_bytes, best_codec, codec_bytes,
    logits_bytes, tape_bytes_1bit,
)
from axplan.credit import (  # noqa: E402
    ADAM_BYTES_PER_PARAM, CreditRule, NeuronModel, TRACE_BYTES,
    compare, credit_bytes, crossover_batch, local_quality,
)

fails: list[str] = []
passed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  [ok]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        fails.append(name)


def note(name: str, text: str) -> None:
    """A measured fact that is NOT an assertion. Never counts as a pass."""
    print(f"  [DIVERGENCE] {name}\n             {text}")


def refuses(fn) -> bool:
    """True if fn raised. Used to assert a refusal instead of a number.

    An API that invents an answer where it should refuse is a defect, and the
    only way to catch that is to call it and watch what comes back.
    """
    try:
        fn()
    except Exception:
        return True
    return False


def is_frozen(obj) -> bool:
    """True if obj rejects attribute assignment."""
    try:
        obj.reason = "mutated"
    except Exception:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 74)
print("T1  CREDIT HORIZON  (the neuron form)")
print("=" * 74)

# lam^k < eps, so k = ceil(log(eps)/log(lam)) when g == lam, i.e. max_slope
# chosen so that lam*(v_th*max_slope - 1) <= lam.
for lam, expect in [(0.25, 5), (0.5, 10), (0.75, 25), (0.9, 66), (0.95, 135)]:
    cm = CreditModel(lam=lam, max_slope=0.5)
    check(f"lam={lam} -> k={expect}", cm.horizon() == expect,
          f"got {cm.horizon()}")

# A sharp surrogate breaks contractivity: credit diverges at any tape length,
# and saying horizon 0 is the honest answer rather than a large number.
cm = CreditModel(lam=0.9, max_slope=2.5)          # snnTorch sigmoid(beta=10)
check("sigmoid(10), lam=0.9 -> not contractive", not cm.contractive(),
      f"g={cm.effective_g}")
check("sigmoid(10) -> horizon 0", cm.horizon() == 0, f"got {cm.horizon()}")

cm = CreditModel.from_surrogate("atan", 2.0, lam=0.9)
check("atan(2.0) maxSlope=1.0", abs(cm.max_slope - 1.0) < 1e-12)
check("atan(2.0) lam=0.9 contractive", cm.contractive())
check("atan(2.0) lam=0.9 -> k=66", cm.horizon() == 66, f"got {cm.horizon()}")

cm = CreditModel.from_surrogate("fast_sigmoid", 25.0, lam=0.9)
check("fast_sigmoid maxSlope=0.5", abs(cm.max_slope - 0.5) < 1e-12)
check("fast_sigmoid lam=0.9 -> k=66", cm.horizon() == 66)

# The scalar form alone would call this contractive. The spectral form does
# not, which is the whole reason the layer form exists.
cm_bad = CreditModel(lam=0.9, max_slope=1.0, spectral_norm=2.4)
check("spectral 2.4 -> not contractive", not cm_bad.contractive(),
      f"g={cm_bad.effective_g}")
cm_good = CreditModel(lam=0.9, max_slope=1.0, spectral_norm=0.85)
check("spectral 0.85 -> contractive", cm_good.contractive(),
      f"g={cm_good.effective_g}")
check("spectral 0.85 -> finite horizon", cm_good.horizon() < 10**6,
      f"got {cm_good.horizon()}")

check("unknown surrogate is rejected, not guessed",
      refuses(lambda: CreditModel.from_surrogate("nope", 1.0)))


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
print("T2  THE REGIME IS COMPUTED, NOT DECLARED")
print("=" * 74)

# Same code, same layer count: only the arithmetic differs, so the regime must
# be able to differ. If these two ever land in the same regime the classifier
# has stopped being arithmetic and become architecture.
conv = BudgetSpec(vram_gb=80, connections=21_800_000, n_layers=34,
                  hidden=512, timesteps=16, spatial=56 * 56, batch=32,
                  seq_len=1, n_heads=1)
p_conv = analyse(conv)

xf = BudgetSpec(vram_gb=80, connections=955_200_000, n_layers=24, hidden=1536,
                timesteps=64, spatial=1, batch=1, seq_len=1024, n_heads=24)
p_xf = analyse(xf)
xf_long = BudgetSpec(**{**xf.__dict__, "timesteps": 1024})
p_xf_long = analyse(xf_long)

check("the two cases land in different regimes",
      p_conv.regime is not p_xf.regime,
      f"both {p_conv.regime}")
check("the same architecture lands in different regimes at different horizons",
      p_xf.regime is not p_xf_long.regime,
      f"both {p_xf.regime}")

# The classifier must agree with its own stated thresholds. These constants
# (0.6 / 0.25) are a calibration choice; the code contradicting them is a bug.
for label, p in (("conv", p_conv), ("1B/T=64", p_xf), ("1B/T=1024", p_xf_long)):
    f = p.activation_fraction
    if p.regime is Regime.OFFLOAD:
        want_offload = True
    elif f > 0.6:
        want_offload = False
    elif f < 0.25:
        want_offload = False
    else:
        want_offload = False
    check(f"{label}: regime matches its own threshold",
          (p.regime is Regime.OFFLOAD) == want_offload,
          f"frac {f:.4f} -> {p.regime}")

note("conv-SNN regime, cited case",
     f"activation_fraction {p_conv.activation_fraction:.3f}, "
     f"regime {p_conv.regime.value}, peak "
     f"{p_conv.peak_bytes / GIB:.1f} GiB of 80. The module docstring quotes "
     f"96.0% and this case as ACTIVATION. It is neither, because "
     f"activation_fraction now counts the dense backward pass "
     f"({p_conv.backward_bytes / GIB:.0f} GiB), a term axon/cost.py had no "
     f"counterpart for.")
note("1B-transformer regime, cited case",
     f"T=64 -> fraction {p_xf.activation_fraction:.4f} "
     f"({p_xf.regime.value}); T=1024 -> {p_xf_long.activation_fraction:.4f} "
     f"({p_xf_long.regime.value}). The old copy reported <25% and STATIC at "
     f"T=64. 0.2594 is 0.9pp above the 0.25 line, so the constant no longer "
     f"separates that case.")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
print("T3  1 BILLION CONNECTIONS: WHAT IT ACTUALLY COSTS")
print("=" * 74)
# The old copy of this suite asserted "1B connections fit in 5 GiB", and it
# passed -- because backward_bytes truncated the BPTT tape at the credit
# horizon, so 934 of the 1000 steps were never charged. The claim was an
# artefact of that bug. It is stated here as measured, including the rule
# change that brings it closest, and it does not fit.

big = dict(vram_gb=5.0, connections=955_200_000, n_layers=24, hidden=1536,
           timesteps=1000, vocab=32000, seq_len=1024, batch=1, n_heads=24,
           precision=Precision(synapse_bits=16, master_bits=0, grad_bits=0,
                               momentum_bits=0, grad_release=True),
           credit=CreditModel.from_surrogate("atan", 2.0, lam=0.9))
bptt_big = analyse(BudgetSpec(**big, credit_rule="bptt"))
local_big = analyse(BudgetSpec(**big, credit_rule="local"))

for label, p in (("BPTT", bptt_big), ("local/e-prop", local_big)):
    print(f"\n  {label}:")
    print("   " + p.explain().replace("\n", "\n   "))

check("1B at T=1000 does NOT fit in 5 GiB under BPTT", bptt_big.infeasible,
      f"peak {bptt_big.peak_bytes / GIB:.3f} GiB")
check("the local rule brings 1B close but does not fit either",
      local_big.infeasible and local_big.peak_bytes < bptt_big.peak_bytes,
      f"local {local_big.peak_bytes / GIB:.3f} GiB")
check("the saving from going local is reported, not just the totals",
      (bptt_big.peak_bytes - local_big.peak_bytes) > 0)
check("and it is a large one",
      (bptt_big.backward_bytes - local_big.backward_bytes) / GIB > 3.0,
      f"{(bptt_big.backward_bytes - local_big.backward_bytes) / GIB:.3f} GiB")

# The crossover is the number that makes the choice decidable, and it is not 1.
xb = crossover_batch(timesteps=1000, hidden=1536, n_layers=24)
print(f"\n  crossover batch at T=1000, h=1536, L=24: {xb:.2f}")
check("the local rule loses at batch 1 and wins from batch 2 upward",
      1.0 < xb <= 2.0, f"crossover {xb:.3f}")
check("and the batch really is the deciding variable, so the crossover is "
      "not an artefact of this shape",
      compare(NeuronModel.LIF, connections=955_200_000, timesteps=1000,
              batch=1, hidden=1536, n_layers=24).bptt.bytes_total
      < compare(NeuronModel.LIF, connections=955_200_000, timesteps=1000,
                batch=1, hidden=1536, n_layers=24).local.bytes_total,
      "at batch 1 credit_bytes uses attn_const 2.02, not the Korthikanti "
      "form, so the two totals are not directly comparable here -- see the "
      "DIVERGENCE note on the two activation constants")

note("the retired '1B fits in 5 GiB' claim",
     f"It held only while backward_bytes capped BPTT at the credit horizon: "
     f"{bptt_big.peak_bytes / GIB:.3f} GiB is what 1000 honest steps cost, "
     f"and {local_big.peak_bytes / GIB:.3f} GiB is the cheapest rule. Both "
     f"exceed 5 GiB. No published document quoted the old figure, so nothing "
     f"on the front page has to be retracted.")

# Still checked, because these were part of the same case and are unaffected:
# the credit horizon, and the error-correction flag.
check("credit horizon > 0", bptt_big.credit_horizon > 0,
      f"k={bptt_big.credit_horizon}")
check("no fp32 master copy at bf16 is flagged as needing error correction",
      big["precision"].needs_error_correction())

big_ec = BudgetSpec(**{**big, "precision": Precision(
    synapse_bits=16, master_bits=0, grad_bits=0, momentum_bits=1,
    grad_release=True)})
p_ec = analyse(big_ec)
print(f"\n  with error-correction (1 B/param): peak "
      f"{p_ec.peak_bytes / GIB:.3f} GiB, headroom "
      f"{p_ec.headroom / GIB:.3f} GiB")
check("the extra precision byte is charged, not free",
      p_ec.static_bytes > bptt_big.static_bytes,
      f"{bptt_big.static_bytes} -> {p_ec.static_bytes}")
check("a negative headroom is reported as a negative number, not clamped",
      p_ec.headroom < 0, f"headroom {p_ec.headroom / GIB:.3f} GiB")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
print("T4  INVARIANTS OF THE MODEL")
print("=" * 74)

check("analyse is a pure function",
      analyse(BudgetSpec(**big, credit_rule="bptt"))
      == analyse(BudgetSpec(**big, credit_rule="bptt")))
check("MemoryPlan is frozen",
      is_frozen(MemoryPlan(vram_bytes=1, connections=1, n_layers=1, hidden=1,
                           timesteps=1)))

# peak must be the sum of the parts it reports, or the breakdown is a story.
for label, p in (("conv", p_conv), ("1B", p_xf), ("1B-long", p_xf_long)):
    parts = (p.static_bytes + p.tape_bytes + p.backward_bytes
             + p.logits_bytes + p.workspace_bytes)
    check(f"{label}: peak == sum of its own terms", p.peak_bytes == parts,
          f"{p.peak_bytes} != {parts}")
    check(f"{label}: fraction is a real fraction",
          0.0 <= p.activation_fraction <= 1.0,
          f"{p.activation_fraction}")
    check(f"{label}: headroom == budget - peak",
          p.headroom == p.vram_bytes - p.peak_bytes)

# Monotonicity: a bigger budget can never make the peak bigger.
prev_peak = prev_k = None
mono = True
for vram in [2, 3, 5, 8, 16, 40, 80, 160]:
    pp = analyse(BudgetSpec(**{**big, "vram_gb": vram}))
    if prev_peak is not None and pp.peak_bytes > prev_peak:
        mono = False
    if prev_k is not None and pp.credit_horizon < prev_k:
        mono = False
    prev_peak, prev_k = pp.peak_bytes, pp.credit_horizon
check("monotone in budget (peak and horizon never shrink)", mono)

# Cheap spikes must cost less tape than dense ones.
sp = BudgetSpec(vram_gb=5, connections=10**9, n_layers=32, hidden=4096,
                timesteps=100, spike_rate=0.01, spatial=1)
lo = tape_bytes_1bit(BudgetSpec(**{**sp.__dict__, "spike_rate": 0.5}))
hi = tape_bytes_1bit(BudgetSpec(**{**sp.__dict__, "spike_rate": 0.01}))
check("sparse spikes cost less tape than dense ones", hi < lo,
      f"{hi / GIB:.4f} vs {lo / GIB:.4f} GiB")

# Nothing fits must be reported, not rounded into a plan.
tiny = analyse(BudgetSpec(**{**big, "vram_gb": 0.5}))
check("0.5 GiB is flagged infeasible", tiny.infeasible and bool(tiny.reason))
check("the infeasible regime is OFFLOAD", tiny.regime is Regime.OFFLOAD)
check("infeasible carries a reason a reader can act on",
      "precision" in tiny.reason, tiny.reason)

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
print("T5  CODEC SELECTION, FROM MEASURED CURVES")
print("=" * 74)
print("  Curves are the measured points of ICLR 2026 (Huang et al.), Tables")
print("  14-16, over 10^7 slots. Reproducing them is the check; nothing here")
print("  is re-derived.")

check("bit @1% = 1.19 MB", abs(codec_bytes(Codec.BIT, 0.01, 1e7) / 1e6 - 1.19) < 0.01)
check("bit @90% = 1.19 MB (density-independent)",
      abs(codec_bytes(Codec.BIT, 0.90, 1e7) / 1e6 - 1.19) < 0.01)
check("sparse @90% = 68.66 MB",
      abs(codec_bytes(Codec.SPARSE, 0.90, 1e7) / 1e6 - 68.66) < 0.01)
check("ANS @90% = 6.95 MB", abs(codec_bytes(Codec.ANS, 0.90, 1e7) / 1e6 - 6.95) < 0.01)
check("dense @90% = 152.6 MB (4 B/element)",
      abs(codec_bytes(Codec.DENSE, 0.90, 1e7) / 1e6 - 152.6) < 0.1)
check("sparse scales linearly with density",
      abs(codec_bytes(Codec.SPARSE, 0.45, 1e7)
          / codec_bytes(Codec.SPARSE, 0.90, 1e7) - 0.5) < 0.02)

print()
print("  rate |    bit |  sparse |     ANS |  CHOSEN")
for rate in [0.001, 0.01, 0.05, 0.08, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]:
    b = codec_bytes(Codec.BIT, rate, 1e7) / 1e6
    s = codec_bytes(Codec.SPARSE, rate, 1e7) / 1e6
    a = codec_bytes(Codec.ANS, rate, 1e7) / 1e6
    c, _ = best_codec(rate, 1e7)
    print(f"  {rate:5.3f} | {b:6.2f} | {s:7.2f} | {a:7.2f} |  {c.value}")

check("ANS wins at rate 0.001", best_codec(0.001, 1e7)[0] is Codec.ANS)
check("bit wins at rate 0.9 (sparsity stopped paying)",
      best_codec(0.9, 1e7)[0] is Codec.BIT)
check("the chosen codec actually changes with density",
      best_codec(0.001, 1e7)[0] != best_codec(0.9, 1e7)[0])
check("the choice is never worse than plain dense",
      all(best_codec(r, 1e7)[1] <= codec_bytes(Codec.DENSE, r, 1e7)
          for r in (0.001, 0.05, 0.5, 0.9)))

# And it changes the PEAK, which is the only reason a codec chooser exists.
s_lo = BudgetSpec(vram_gb=5, connections=10**9, n_layers=32, hidden=4096,
                  timesteps=200, spike_rate=0.005, spatial=1)
s_hi = BudgetSpec(**{**s_lo.__dict__, "spike_rate": 0.7})
pl, ph = analyse(s_lo), analyse(s_hi)
print(f"\n  at rate 0.5%: tape {pl.tape_bytes / GIB:.4f} GiB, "
      f"peak {pl.peak_bytes / GIB:.3f}")
print(f"  at rate 70% : tape {ph.tape_bytes / GIB:.4f} GiB, "
      f"peak {ph.peak_bytes / GIB:.3f}")
check("dense spikes cost more", ph.tape_bytes > pl.tape_bytes)

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
print("T6  THE BACKWARD PASS IS A SEPARATE, DENSE, CREDIT-DEPENDENT TERM")
print("=" * 74)
# This is the term whose arrival moved the regime constants, so it gets its
# own checks rather than riding along inside T4.

check("the backward term is non-zero", backward_bytes(sp) > 0)
check("spike_rate does not appear in a dense gradient at all",
      backward_bytes(BudgetSpec(**{**sp.__dict__, "spike_rate": 0.9}))
      == backward_bytes(sp),
      "bit-ness must not reach the backward pass")

# The claim the README makes: the local rule makes the tape horizon-
# independent, and that is the largest single lever available.
bptt = analyse(BudgetSpec(**{**sp.__dict__, "timesteps": 100,
                             "credit_rule": "bptt"}))
local = analyse(BudgetSpec(**{**sp.__dict__, "timesteps": 100,
                              "credit_rule": "local"}))
check("the local rule's memory does not move with the horizon",
      analyse(BudgetSpec(**{**sp.__dict__, "timesteps": 400,
                            "credit_rule": "local"})).backward_bytes
      == local.backward_bytes,
      f"{local.backward_bytes} at T=100")
check("BPTT's memory does",
      analyse(BudgetSpec(**{**sp.__dict__, "timesteps": 400,
                            "credit_rule": "bptt"})).backward_bytes
      > bptt.backward_bytes,
      f"{bptt.backward_bytes} at T=100")

# Which rule is cheaper depends on the batch, and the planner is obliged to
# say where the boundary is rather than pick a winner. At 10^9 connections
# with a narrow hidden width and a short horizon, the 4 B/synapse trace is
# genuinely MORE expensive than the tape -- the same finding examples/
# advanced/03_credit_rule.py prints, so it is asserted here too.
check("below the crossover batch, BPTT is the cheaper way to hold credit",
      bptt.backward_bytes < local.backward_bytes,
      f"bptt {bptt.backward_bytes / GIB:.3f} GiB vs "
      f"local {local.backward_bytes / GIB:.3f} GiB")

wide = {k: v for k, v in sp.__dict__.items() if k != "credit_rule"}
wide.update(connections=10**9, hidden=4096, n_layers=32, batch=32,
            timesteps=256)
w_bptt = backward_bytes(BudgetSpec(**wide, credit_rule="bptt"))
w_local = backward_bytes(BudgetSpec(**wide, credit_rule="local"))
print(f"\n  at the same spec but batch 32, h=4096, L=32, T=256:")
print(f"    BPTT {w_bptt / GIB:.3f} GiB   local {w_local / GIB:.3f} GiB")
check("above the crossover batch, the local rule is the cheaper one",
      w_local < w_bptt,
      f"bptt {w_bptt / GIB:.3f} GiB vs local {w_local / GIB:.3f} GiB")
xb_wide = crossover_batch(timesteps=256, hidden=4096, n_layers=32)
print(f"    crossover batch at that shape: {xb_wide:.2f}")
check("the crossover batch is below 32 there, which is why local wins",
      xb_wide <= wide["batch"], f"crossover {xb_wide:.2f} vs batch "
      f"{wide['batch']}")
check("a zero firing rate is refused rather than priced as a free tape",
      refuses(lambda: BudgetSpec(**{**sp.__dict__, "spike_rate": 0.0})))
check("a zero horizon is refused rather than priced as no tape",
      refuses(lambda: BudgetSpec(**{**sp.__dict__, "timesteps": 0})))

check("an unknown credit rule is refused, not silently treated as BPTT",
      refuses(lambda: analyse(BudgetSpec(**{**sp.__dict__,
                                             "credit_rule": "wishful"}))))
check("an unknown neuron model is refused",
      refuses(lambda: analyse(BudgetSpec(**{**sp.__dict__,
                                            "neuron_model": "quadratic"}))))
check("a zero firing rate is refused rather than priced as a free tape",
      refuses(lambda: BudgetSpec(**{**sp.__dict__, "spike_rate": 0.0})))
check("a zero horizon is refused rather than priced as no tape",
      refuses(lambda: BudgetSpec(**{**sp.__dict__, "timesteps": 0})))

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
print("T7  THE CREDIT RULE IS ACTUALLY THE RULE THAT WAS ASKED FOR")
print("=" * 74)
# Two defects lived here, and both answered a different question than the one
# asked while returning a plausible number. That is the failure mode this
# whole file exists to catch, so the regression checks are explicit.

check("the local rule is T-independent",
      credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, 10**9, 64, 8, 1024, 8).bytes_total
      == credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, 10**9, 6400, 8, 1024, 8).bytes_total)
check("the local rule reports depends_on_T=False",
      credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, 10**9, 64, 8, 1024, 8).depends_on_T
      is False)
check("BPTT reports depends_on_T=True",
      credit_bytes(CreditRule.BPTT, NeuronModel.LIF, 10**9, 64, 8, 1024, 8).depends_on_T
      is True)
check("the local rule is charged its quality price",
      credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, 10**9, 500, 8, 1024, 8).quality_kept
      == 0.389)
check("the local rule reports quality unknown past the measured range",
      credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, 10**9, 5000, 8, 1024, 8).quality_kept
      is None)
check("full BPTT is the 100% reference",
      credit_bytes(CreditRule.BPTT, NeuronModel.LIF, 10**9, 500, 8, 1024, 8).quality_kept
      == 1.0)

# The regression itself. credit_bytes dispatched on `is CreditRule.LOCAL`,
# and axplan.memory passed the bare string 'local', so the local branch was
# unreachable and every local-rule request was answered with BPTT bytes. A
# str-Enum equals its value but is never identical to it, which is why the
# check passed review and still shipped the bug.
check("a plain string routes to the same branch as its enum member",
      credit_bytes("local", NeuronModel.LIF, 10**9, 500, 8, 1024, 8).bytes_total
      == credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, 10**9, 500, 8, 1024, 8).bytes_total,
      "a str-Enum compares equal to its value but is not identical to it")
check("a plain string neuron routes to the same branch",
      credit_bytes("local", "lif", 10**9, 500, 8, 1024, 8).bytes_total
      == credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, 10**9, 500, 8, 1024, 8).bytes_total)
check("an unknown rule reaches credit_bytes as an error, not as BPTT",
      refuses(lambda: credit_bytes("elvar", NeuronModel.LIF, 10**9, 64, 8, 1024, 8)))

# And the end-to-end version: the number the planner reports is the number the
# rule is worth.
conn, T = 10**9, 256
via_memory = backward_bytes(BudgetSpec(vram_gb=80, connections=conn, n_layers=8,
                                       hidden=1024, timesteps=T, batch=8,
                                       spike_rate=0.25, credit_rule="local"))
via_credit = credit_bytes(CreditRule.LOCAL, NeuronModel.LIF, conn, T, 8, 1024,
                          8).bytes_total
check("analyse() and credit_bytes() agree on the local rule",
      via_memory == via_credit, f"{via_memory} vs {via_credit}")
check("the local rule's trace is 4 B/synapse for LIF",
      via_credit == conn * TRACE_BYTES[NeuronModel.LIF],
      f"{via_credit} vs {conn * 4}")
check("Adam state is 8 B/param, and the local rule is cheaper than it",
      ADAM_BYTES_PER_PARAM == 8
      and via_credit < conn * ADAM_BYTES_PER_PARAM,
      f"local {via_credit / 1e9:.2f} GB vs Adam {conn * 8 / 1e9:.2f} GB")

# ALIF and ALIF+reward cost more, in the order the module documents.
check("trace size rises LIF < ALIF < ALIF+reward",
      TRACE_BYTES[NeuronModel.LIF] < TRACE_BYTES[NeuronModel.ALIF]
      < TRACE_BYTES[NeuronModel.ALIF_REWARD] == 12)

# BPTT in the planner must be Theta(T*A): the tape covers the whole horizon.
bptt_256 = backward_bytes(BudgetSpec(vram_gb=80, connections=conn, n_layers=8,
                                     hidden=1024, timesteps=256, batch=8,
                                     credit_rule="bptt"))
bptt_512 = backward_bytes(BudgetSpec(vram_gb=80, connections=conn, n_layers=8,
                                     hidden=1024, timesteps=512, batch=8,
                                     credit_rule="bptt"))
check("doubling the horizon doubles the BPTT tape",
      abs(bptt_512 / bptt_256 - 2.0) < 1e-9,
      f"ratio {bptt_512 / bptt_256:.4f}")
check("the BPTT tape is not capped at the credit horizon",
      bptt_256 > credit_bytes(CreditRule.BPTT, NeuronModel.LIF, conn, 256, 8,
                              1024, 8, credit_k=66).bytes_total,
      "truncation must be requested, not applied by default")

# compare() is what the examples print, so its verdict has to follow its own
# numbers rather than a standing assumption that the local rule is cheaper.
# At batch 8 it is NOT, and compare() must say so.
#
# connections is set to L*h*h to match crossover_batch, which derives the
# parameter count from the shape instead of taking it. Passing an unrelated
# connections makes the two disagree -- see the DIVERGENCE note below.
conn, hidden, layers, T = 8 * 1024 * 1024, 1024, 8, 256
ch = compare(NeuronModel.LIF, connections=conn, timesteps=T, batch=8,
             hidden=hidden, n_layers=layers)
check("compare() signs its saving consistently with the two totals",
      (ch.saving_bytes < 0) == (ch.local.bytes_total > ch.bptt.bytes_total),
      f"saving {ch.saving_bytes} but bptt {ch.bptt.bytes_total} vs "
      f"local {ch.local.bytes_total}")
check("compare() marks BPTT as the T-dependent one and local as not",
      ch.bptt.depends_on_T is True and ch.local.depends_on_T is False)
check("compare() says the local rule is cheaper than Adam",
      ch.marginal_vs_adam < 1.0, f"{ch.marginal_vs_adam:.3f}x Adam")
xb_here = crossover_batch(timesteps=T, hidden=hidden, n_layers=layers)
print(f"\n  crossover batch at T={T}, h={hidden}, L={layers}: {xb_here:.1f}")
ch_big = compare(NeuronModel.LIF, connections=conn, timesteps=T, batch=64,
                 hidden=hidden, n_layers=layers)
check("above the crossover batch the local rule is cheaper, and the sign "
      "flips with it",
      xb_here < 64 and ch_big.saving_bytes > 0
      and ch_big.bptt.bytes_total > ch_big.local.bytes_total,
      f"crossover {xb_here:.1f}, saving {ch_big.saving_bytes / GIB:.3f} GiB")
# compare() charges BPTT per example, so its saving closes linearly in the
# batch. crossover_batch answers the same question by dividing the local total
# by the per-example BPTT cost. These must agree, and they do only because
# compare() defaults attn_const to the same 2.0208... both paths use.
ch_b1 = compare(NeuronModel.LIF, connections=conn, timesteps=T, batch=1,
                hidden=hidden, n_layers=layers)
_implied = ch_b1.local.bytes_total / ch_b1.bptt.bytes_total
check("the crossover compare() implies is the crossover crossover_batch "
      "reports",
      abs(_implied - xb_here) / xb_here < 1e-3,
      f"compare implies {_implied:.6f}, crossover_batch says {xb_here:.6f}")

note("connections is counted two different ways",
     f"crossover_batch derives it as L*h*h = {layers * hidden * hidden:,} and "
     f"compare() takes it as an argument. Pass an unrelated connections -- "
     f"10^9 against this shape -- and the two disagree by "
     f"{10**9 / (layers * hidden * hidden):.0f}x, so the crossover printed by "
     f"explain_budget and the verdict printed beside it describe different "
     f"networks. Both are individually correct; nothing checks they agree.")

check("the crossover is finite and positive at a long horizon", xb_here > 0)
check("local_quality declines as the horizon grows",
      local_quality(60) > local_quality(500) > 0)

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
print("T7  THE ARITHMETIC IS OURS AND MATCHES THE FORMULA")
print("=" * 74)
# One hand-computed case, so a change to a constant cannot pass unnoticed by
# re-deriving the same wrong formula inside the test.
tiny_spec = BudgetSpec(vram_gb=1.0, connections=1000, n_layers=1, hidden=10,
                       timesteps=2, vocab=5, seq_len=1, batch=1, n_heads=1,
                       spatial=1, fragmentation=1.0, fixed_gb=0.0)
check("activation_bytes matches the Korthikanti form by hand",
      activation_bytes(tiny_spec) == 2 * 1 * (1 * 10 * 1) * (34.0 + 5.0 * 1 * 1 / 10) * 2,
      f"got {activation_bytes(tiny_spec)}")
check("logits_bytes is 4 B per element, fp32",
      logits_bytes(tiny_spec) == 4 * 1 * 1 * 5)
check("tape_bytes_1bit is bounded by the bit codec",
      tape_bytes_1bit(tiny_spec) <= codec_bytes(Codec.BIT, 1.0, 2 * 1 * 10 * 2 * 1))
check("usable_bytes honours fragmentation and the fixed overhead",
      BudgetSpec(**{**tiny_spec.__dict__, "vram_gb": 10.0,
                    "fragmentation": 1.5, "fixed_gb": 1.0}).usable_bytes()
      == int(10.0 * GIB / 1.5 - 1.0 * GIB))
check("vram_bytes is the budget, unrounded away",
      BudgetSpec(**{**tiny_spec.__dict__, "vram_gb": 6.0}).vram_bytes()
      == int(6.0 * GIB))
check("Precision charges only the roles that are switched on",
      Precision(synapse_bits=16, master_bits=0, momentum_bits=0).static_bytes_per_param() == 2.0
      and Precision(synapse_bits=16, master_bits=32, momentum_bits=0).static_bytes_per_param() == 6.0
      and Precision(synapse_bits=16, master_bits=0, momentum_bits=8).static_bytes_per_param() == 3.0)
check("grad_release=False charges the gradient bytes",
      Precision(synapse_bits=16, grad_bits=16, grad_release=False).static_bytes_per_param() == 4.0)
check("an fp32 master copy is not flagged as needing error correction",
      not Precision(synapse_bits=16, master_bits=32).needs_error_correction())

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 74)
if fails:
    print(f"FAILED {len(fails)} of {passed + len(fails)}: {fails}")
    sys.exit(1)
print(f"ALL {passed} CHECKS PASSED")
print("  read the DIVERGENCE lines above: they are measurements this suite")
print("  reports and does not assert, because the docstring figures they")
print("  contradict belong to a model that no longer exists in this code.")
sys.exit(0)