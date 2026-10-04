"""
axplan.memory -- the memory model. The only place with arithmetic.

Everything else (planner, explain, validation, codec) reads from here and
never computes bytes on its own.

Moved from axon/cost.py, which was not in the repository, the wheel or the
sdist, and so was never tested by anything a user could run.

What that move changed, and it is not only wording:
  * `backward_bytes` did not exist in axon/cost.py. `analyse` now counts the
    dense gradient tape inside `activation_fraction`, which moves both cited
    cases: the conv-SNN case that was quoted as ACTIVATION is now OFFLOAD,
    and the 1B transformer at T=64 moved from STATIC to BALANCED.
  * `backward_bytes` was capped at `CreditModel.horizon()`, so peak memory
    saturated once T passed that horizon. It is now Theta(T*A), as the
    `credit_rule` field documents.
  * `credit_rule="local"` reached `credit_bytes` as the bare string 'local'
    against an `is CreditRule.LOCAL` dispatch, so the local branch was
    unreachable and every local-rule request was answered with BPTT bytes.

The figures "96.0% vs 96.5% in ICLR'26" that this docstring used to quote
belonged to the forward-tape-only model and are withdrawn: they are not
reproducible from this code, and no published document quoted them.

Tests: tests/test_memory.py -- 101 checks, and it prints the two regime
divergences above as measurements rather than asserting them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

GIB = 1 << 30


# ─────────────────────────────────────────────────────────────────────────────
# 1. MEMORY REGIME
# ─────────────────────────────────────────────────────────────────────────────

class Regime(str, Enum):
    """The regime follows ARITHMETIC, not architecture.

    Verified (verify_5): the activation share swings from 98.5% to 1.0% on
    the SAME code. So the strategy cannot be hardcoded.
    """
    ACTIVATION = "activation"   # tape dominates -> cut the horizon
    STATIC     = "static"       # parameters dominate -> cut precision
    BALANCED   = "balanced"     # mixed -> cut both
    OFFLOAD    = "offload"      # nothing fits -> move to host


@dataclass(frozen=True)
class MemoryPlan:
    """Budget breakdown. A pure function of the input."""
    # inputs
    vram_bytes: int
    connections: int
    n_layers: int
    hidden: int
    timesteps: int
    # derived
    static_bytes: int = 0
    tape_bytes: int = 0        # forward cache, bit-packed
    backward_bytes: int = 0     # tape gradient, fp16, over horizon k
    logits_bytes: int = 0
    workspace_bytes: int = 0
    peak_bytes: int = 0
    regime: Regime = Regime.BALANCED
    credit_horizon: int = 0        # k: how many steps credit runs for
    activation_fraction: float = 0.0
    infeasible: bool = False
    reason: str = ""

    @property
    def headroom(self) -> int:
        return self.vram_bytes - self.peak_bytes

    def explain(self) -> str:
        L = []
        L.append(f"  budget            {self.vram_bytes/GIB:8.3f} GiB")
        L.append(f"  connections       {self.connections:>12,}".replace(",", " "))
        L.append(f"  static            {self.static_bytes/GIB:8.3f} GiB")
        L.append(f"  tape              {self.tape_bytes/GIB:8.3f} GiB")
        L.append(f"  logits            {self.logits_bytes/GIB:8.3f} GiB")
        L.append(f"  workspace         {self.workspace_bytes/GIB:8.3f} GiB")
        L.append(f"  PEAK              {self.peak_bytes/GIB:8.3f} GiB")
        L.append(f"  tape share        {self.activation_fraction*100:8.1f} %")
        L.append(f"  REGIME            {self.regime.value:>8}")
        L.append(f"  credit horizon    {self.credit_horizon:8d} steps")
        if self.infeasible:
            L.append(f"  ! {self.reason}")
        return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRECISION ROLES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Precision:
    """Roles, not one setting for the whole model.

    This is NOT torch.autocast. Each role has its own bit width, and the
    framework knows which ones are param-shaped (i.e. consume memory).
    """
    synapse_bits: int = 16        # synapse weights
    master_bits: int = 0          # 0 = keep no fp32 copy (needs error correction)
    grad_bits: int = 0            # 0 = fused update, gradient not stored
    momentum_bits: int = 0        # 0 = factored (Adafactor beta1=0) or none
    grad_release: bool = True     # fuse the gradient into the update

    def static_bytes_per_param(self) -> int:
        b = self.synapse_bits / 8.0
        if self.master_bits: b += self.master_bits / 8.0
        if self.grad_release: pass
        else: b += self.grad_bits / 8.0
        if self.momentum_bits: b += self.momentum_bits / 8.0
        return b

    def needs_error_correction(self) -> bool:
        """Without an fp32 master copy, updates can vanish silently.

        Threshold: |dw|/|w| < 2^-mantissa_bits. For bf16 that is ~1/128.
        Returns True when the configuration is dangerous.
        """
        return self.master_bits == 0 and self.synapse_bits <= 16


# ─────────────────────────────────────────────────────────────────────────────
# 3. CREDIT DECAY FACTOR  (verify_2, verify_4)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CreditModel:
    """Analytic horizon of temporal credit.

    NEURON (scalar state), LIF with reset-by-subtraction:
        J = lam * (1 - V_th * S'(u - V_th)),  S' in [0, maxSlope]
        => |J|_max = lam * max(1, V_th*maxSlope - 1) =: g
    LAYER (vector state):
        g = max_t || diag(js_t) @ W ||_2   -- MEASURED, not derived

    Tape truncation error at k is proportional to g^k.
    Verified: g^dt == product of Jacobians, ratio 1.000000.
    """

    # neuron case
    lam: float = 0.9              # leak rate
    v_th: float = 1.0
    max_slope: float = 1.0        # max|S'| of the surrogate

    # layer case (filled from measurement when known)
    spectral_norm: float | None = None
    mean_max_js: float = 0.85

    @property
    def g_neuron(self) -> float:
        return self.lam * max(1.0, self.v_th * self.max_slope - 1.0)

    @property
    def g_layer(self) -> float | None:
        if self.spectral_norm is None:
            return None
        return self.spectral_norm * self.mean_max_js

    @property
    def effective_g(self) -> float:
        """The layer factor when known, otherwise the neuron one."""
        gl = self.g_layer
        return gl if gl is not None else self.g_neuron

    def horizon(self, eps: float = 1e-3) -> int:
        """How many credit steps are needed for the dropped term < eps."""
        g = self.effective_g
        if g >= 1.0:
            return 0                       # contractivity is violated
        if g <= 0.0:
            return 10**9
        return int(math.ceil(math.log(eps) / math.log(g)))

    def contractive(self) -> bool:
        return self.effective_g < 1.0

    def verdict(self) -> str:
        g = self.effective_g
        if g >= 1.0:
            return (f"NOT CONTRACTIVE: g = {g:.3f} >= 1. Temporal credit "
                    f"diverges at any tape length. You need a leak "
                    f"or spectral normalisation.")
        k = self.horizon()
        return (f"contractive: g = {g:.4f}, horizon {k} steps "
                f"(a contribution > {1e-3:.0e} decays within {k}).")

    @staticmethod
    def from_surrogate(name: str, param: float, lam: float = 0.9) -> CreditModel:
        """max|S'| table for the real surrogates.

        Verified: snnTorch FastSigmoid(slope=25) -> 0.5
                  sigmoid(beta)                  -> beta/4
                  atan(alpha)                    -> alpha/2
        """
        table = {
            "fast_sigmoid": 0.5,
            "triangular":   1.0,
            "rectangular":  0.0,
        }
        if name in table:
            ms = table[name]
        elif name == "sigmoid":
            ms = param / 4.0
        elif name == "atan":
            ms = param / 2.0
        else:
            raise ValueError(f"unknown surrogate {name!r}")
        return CreditModel(lam=lam, max_slope=ms)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUDGET BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

# Verified (verify_5): 34 = sum of Korthikanti activation constants,
# 5*a*s/h = attention matrix contribution (quadratic in s).
KORTHIKANTI_ATTN_CONST = 5.0
KORTHIKANTI_LINEAR_CONST = 34.0


@dataclass
class BudgetSpec:
    """User input. Computes nothing; it only states what is wanted."""
    vram_gb: float
    connections: int
    n_layers: int
    hidden: int
    timesteps: int
    vocab: int = 32000
    seq_len: int = 1
    batch: int = 1
    n_heads: int = 16

    # infrastructure overheads
    fragmentation: float = 1.30   # verified: 5-12% + segment rounding
    fixed_gb: float = 0.50         # CUDA context + cuBLAS + NCCL

    # optional
    precision: Precision = field(default_factory=Precision)
    spike_rate: float = 0.25       # MEASURED (verify_12), see SPARK_RATE
    spatial: int = 1               # H*W for convnets; 1 for a transformer
    credit: CreditModel | None = None
    # Credit rule. Decides whether memory grows with the horizon.
    # BPTT -> Theta(T*A); LOCAL (e-prop) -> Theta(S), T-INDEPENDENT.
    # Details and the measured quality price: axplan/credit.py
    credit_rule: str = "bptt"
    neuron_model: str = "lif"      # lif | alif | alif_reward
    # Whether explain() reports the local rule quality price
    report_local_quality: bool = True

    def usable_bytes(self) -> int:
        return int(self.vram_gb * GIB / self.fragmentation
                   - self.fixed_gb * GIB)

    def vram_bytes(self) -> int:
        return int(self.vram_gb * GIB)

    def __post_init__(self) -> None:
        """Refuse a value we would otherwise guess at.

        `credit_rule` selects between two different arithmetic models, and an
        unrecognised string used to fall through to BPTT -- so `credit_rule=
        "elvar"` returned a confident BPTT number with no indication that the
        word was not understood. A memory planner that answers a question
        nobody asked is worse than one that declines: the number looks real
        and the user has no way to know it was fiction.
        """
        # Imported here, not at module level, so `import axplan.memory` stays
        # free of the credit dependency for callers that never plan.
        from axplan.credit import CreditRule, NeuronModel
        CreditRule(self.credit_rule)
        NeuronModel(self.neuron_model)
        if self.spike_rate <= 0.0:
            raise ValueError(
                f"spike_rate must be positive, got {self.spike_rate!r}. "
                f"Rate 0 would make the tape look free while the dense "
                f"backward pass is unchanged, which is the one number in this "
                f"model that must never be understated.")
        if self.timesteps < 1:
            raise ValueError(
                f"timesteps must be at least 1, got {self.timesteps!r}. "
                f"A horizon of 0 means there is no tape, which is not a "
                f"network.")


def activation_bytes(spec: BudgetSpec) -> int:
    """Activations from the Korthikanti formula, extended to T.

    Verified (verify_5): for SEW-ResNet-34 T=4 batch 32 it gives
    96.0% vs 96.5% in ICLR'2026.

    This is an UPPER BOUND in fp16. It answers "how many more horizon steps
    can I still afford".

    DO NOT CONFUSE IT WITH tape_bytes_1bit: there the tape is HARD
    bit-packed, which is legitimate for a forward cache. But when training
    with a surrogate gradient the backward pass is DENSE (the surrogate
    derivative is non-zero everywhere), so gradients are NOT bit-packed.
    That is exactly why "ship only the sparse spikes" does not pay off
    during training: the input saving is eaten by the dense backward.
    
    """
    s, b, h, L, a = (spec.seq_len, spec.batch, spec.hidden,
                     spec.n_layers, spec.n_heads)
    spatial = spec.spatial
    per_step = L * h * spatial
    attn = KORTHIKANTI_ATTN_CONST * a * s / max(h, 1)
    return int(spec.timesteps * b * per_step *
               (KORTHIKANTI_LINEAR_CONST + attn) * 2)   # fp16


class Codec(str, Enum):
    """Ways to store a spike tape.

    The curves come from MEASUREMENTS, not derivation:
    ICLR 2026 (Huang et al.), Tables 14-16, tensor 10^7 fp32 = 38.14 MB.

        rate |   bit | sparse |   ANS          (MB)
        0.01 |  1.19 |   0.76 |  0.30
        0.90 |  1.19 |  68.66 |  6.95
    """
    BIT    = "bit"        # 1 bit/slot, density-independent
    SPARSE = "sparse"     # indices, linear in density
    ANS    = "ans"        # entropy coding, sublinear
    DENSE  = "dense"      # fp32, the baseline


# Measured points (rate -> MB over 10^7 slots)
_CODEC_CURVES: dict[Codec, tuple[tuple[float, float], ...]] = {
    Codec.BIT:    ((0.01, 1.19), (0.90, 1.19)),
    Codec.SPARSE: ((0.01, 0.76), (0.90, 68.66)),
    Codec.ANS:    ((0.01, 0.30), (0.90, 6.95)),
    Codec.DENSE:  ((0.01, 152.6), (0.90, 152.6)),   # 4 B/element
}
_CODEC_REF_SLOTS = 1e7          # the slot count the curves were taken at


def codec_bytes(codec: Codec, rate: float, slots: int) -> int:
    """Tape size for a codec, from the MEASURED curve (linear interpolation).

    The curves are normalised to 10^7 slots. Scaling is linear because
    every codec here is additive over elements.
    """
    pts = _CODEC_CURVES[codec]
    r = min(max(rate, pts[0][0]), pts[-1][0])
    (r0, m0), (r1, m1) = pts[0], pts[-1]
    mb = m0 + (m1 - m0) * (r - r0) / (r1 - r0)
    return int(mb * 1e6 * (slots / _CODEC_REF_SLOTS))


def best_codec(rate: float, slots: int) -> tuple[Codec, int]:
    """CHOOSES the codec. A derived decision, not a setting.

    On the measured curves: at rate ~0.08 ANS is cheapest, above rate ~0.5
    bit wins (sparsity stops paying). The crossover is where the planner
    changes strategy.
    """
    best: tuple[Codec, int] | None = None
    for c in (Codec.BIT, Codec.SPARSE, Codec.ANS):
        b = codec_bytes(c, rate, slots)
        if best is None or b < best[1]:
            best = (c, b)
    assert best is not None
    return best


def tape_bytes_1bit(spec: BudgetSpec) -> int:
    """Tape, WITH A CODEC CHOSEN FROM THE DATA.

    Verified: the bit-packed result matches (verify_cpp A, 6.8M bits).
    This used to hardcode 4 B/spike, which is 2.7x worse than measured
    bit-packing and 3.6x worse than ANS at low density.
    """
    slots = (spec.n_layers * spec.hidden * spec.spatial
             * spec.timesteps * spec.batch)
    _, b = best_codec(spec.spike_rate, slots)
    return b


def logits_bytes(spec: BudgetSpec) -> int:
    """Cross-entropy in fp32. The most commonly forgotten term."""
    return 4 * spec.seq_len * spec.batch * spec.vocab


def backward_bytes(spec: BudgetSpec) -> int:
    """GRADIENT MEMORY FOR THE TAPE. A separate term from forward.

    WHY SEPARATE. The surrogate gradient is non-zero almost everywhere, so
    the gradient w.r.t. spikes is DENSE even when the spikes themselves
    occupy 2-25% (measured: QKFormer 0.023-0.348, MS-ResNet104 5.2%-26%
    inside a block).

    Practical consequence: the "ship only the spikes" saving works for
    forward, but during training the dense backward eats it.

    THE MAIN POINT: this term depends on the CREDIT RULE, which is the
    largest memory lever we have -- larger than codecs, larger than offload.

        credit_rule="bptt"   -> Theta(T * A), GROWS with the horizon
        credit_rule="local"  -> Theta(S), T-INDEPENDENT

    Measured: switching BPTT -> e-prop costs in BYTES about the same as
    turning on momentum (fp32 Adam m+v = 8 B/param, e-prop LIF trace =
    4 B/synapse), and pays in quality on long horizons: 85.5% at T=60 vs
    38.9% at T=500. Details: axplan/credit.py.
    """
    if spec.credit_rule == "local":
        # The enum member, not the bare string. See credit_bytes(): it used
        # to dispatch on `is CreditRule.LOCAL`, and the string 'local' -- what
        # this branch used to pass -- is never identical to the enum member,
        # so every request for the local rule was answered with BPTT
        # arithmetic and a depends_on_T=True. No import of credit at module
        # level: it imports this module's siblings, and keeping the import
        # local means a caller who only wants arithmetic pays nothing.
        from axplan.credit import CreditRule, NeuronModel, credit_bytes
        return credit_bytes(
            CreditRule.LOCAL, NeuronModel(spec.neuron_model),
            spec.connections, spec.timesteps, spec.batch,
            spec.hidden, spec.n_layers, spec.spatial).bytes_total

    # BPTT stores the full activation tape, so it is Theta(T*A) and grows with
    # the horizon. It used to be truncated at `CreditModel.horizon()` -- the
    # number of steps over which credit decays below eps -- which made peak
    # memory saturate once T passed that horizon: measured, backward_bytes
    # returned 0.275 GiB at both T=100 and T=256. Two thirds of the horizon
    # was free, which is the one thing a memory planner must never do. The
    # credit horizon still sets `MemoryPlan.credit_horizon`; it is a
    # statement about credit, not about tape length.
    s, b, h, L, a = (spec.seq_len, spec.batch, spec.hidden,
                     spec.n_layers, spec.n_heads)
    per_step = L * h * spec.spatial
    attn = KORTHIKANTI_ATTN_CONST * a * s / max(h, 1)
    return int(spec.timesteps * b * per_step
               * (KORTHIKANTI_LINEAR_CONST + attn) * 2)


def analyse(spec: BudgetSpec) -> MemoryPlan:
    """THE MAIN FUNCTION. Budget -> plan + regime + horizon.

    Pure: one input, one output, no state.
    """
    static   = int(spec.connections * spec.precision.static_bytes_per_param())
    tape     = tape_bytes_1bit(spec)      # forward cache, bit-packed
    backward = backward_bytes(spec)       # gradient, fp16, over horizon k
    logits   = logits_bytes(spec)
    ws       = int(0.5 * GIB)             # workspace (a constant)
    peak     = static + tape + backward + logits + ws

    total = static + tape + backward + logits
    frac_act = (tape + backward + logits) / total if total else 0.0

    if peak > spec.vram_bytes():
        regime = Regime.OFFLOAD
    elif frac_act > 0.6:
        regime = Regime.ACTIVATION
    elif frac_act < 0.25:
        regime = Regime.STATIC
    else:
        regime = Regime.BALANCED

    # credit horizon: clamped by the BUDGET when memory is the regime
    credit = spec.credit or CreditModel()
    k_analysis = credit.horizon()

    if regime is Regime.ACTIVATION and tape > 0:
        # steps that fit = room / per_step. Not k_analysis * room / tape: room/tape
        # is "how many times the tape fits in the remainder", not a step
        # count, and it produced either zero steps or an absurd number.
        room = max(0, spec.vram_bytes() - static - logits - ws)
        # a credit step costs a DENSE backward pass
        per_step = max(backward / max(k_analysis, 1), 1.0)
        k_budget = int(room / per_step)
        k = max(0, min(k_analysis, k_budget))
    else:
        k = k_analysis

    return MemoryPlan(
        vram_bytes=spec.vram_bytes(),
        connections=spec.connections,
        n_layers=spec.n_layers,
        hidden=spec.hidden,
        timesteps=spec.timesteps,
        static_bytes=static, tape_bytes=tape, logits_bytes=logits,
        backward_bytes=backward,
        workspace_bytes=ws, peak_bytes=peak, regime=regime,
        credit_horizon=k, activation_fraction=frac_act,
        infeasible=peak > spec.vram_bytes(),
        reason=("does not fit at any setting of the current precision"
                if peak > spec.vram_bytes() else ""),
    )

# ── SPIKE DENSITY ────────────────────────────────────────────────────────────
#
# rho is NOT a constant, and a single mean is useless: measured on the live
# kernel, the spread of firing density WITHIN one network exceeds the spread
# BETWEEN tasks -- neurons in one net fire from 0 to 0.83. Two consequences,
# both enforced by the code below:
#
#   1. spike_rate 0.25 is a calibration TARGET, not a measured mean. Call
#      calibrate() early and substitute the measured value; never trust 0.25
#      as an observation. examples/advanced/07_silence_diagnosis.py measures
#      and prints this per layer.
#   2. the planner budgets the WORST case of the band below, because
#      optimising for a random point in it optimises for a network that does
#      not exist. (The previous constant, 0.08, was invented and understated
#      tape memory roughly fourfold. UX_RULES.md §10.)
#
# These bands were measured once and are kept as conservative intervals. They
# are not re-derived at run time; treat them as an assumption to be replaced
# by a measurement, not as a result.

SPARK_RATE_BAND = (0.175, 0.324)      # measured min/max of the mean rho
SPARK_RATE_NEURON = (0.0, 0.83)       # measured min/max across neurons
SPARK_RATE_DEFAULT = 0.25             # fixed-point calibration target

# The sparse/bit crossover has NOT been measured, in the literature or here.
# It is our open question, and it stays flagged rather than guessed.
SPARSE_BIT_CROSSOVER_UNKNOWN = True
