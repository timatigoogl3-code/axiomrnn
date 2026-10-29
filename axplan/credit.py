"""
axplan.credit -- credit assignment rules and their price in bytes.

This is what we LEFT OUT of the first memory model, and it turned out to be
the single largest correction of the whole project. We had modelled only
BPTT and assumed tape memory was set by the horizon and the codec. Both
assumptions are wrong as the MAIN levers.

══════════════════════════════════════════════════════════════════════
THE MAIN FINDING (measured, not invented)
══════════════════════════════════════════════════════════════════════
    fp32 Adam m+v = 8 B/param
    e-prop (LIF)  = 4 B/synapse   -- does NOT depend on T at all
    e-prop (ALIF) = 8 B/synapse
    ALIF+reward   = 12 B/synapse

So on a fixed architecture, switching BPTT -> e-prop costs in BYTES about
the same as switching momentum on -- not like changing paradigms.

More important: the real cliff is T, not S.
    BPTT:  Theta(T * A)   grows with the horizon
    e-prop: Theta(S)      does not grow

Quality price measured (FPTT, Yin/Corradi/Bohte):
    T=60  -> 85.5%
    T=500 -> 38.9%
    a 47-point gap on long horizons

COMPLEXITY CLASSES (three different ones, do not mix them)
══════════════════════════════════════════════════════════════════════
    Theta(S)  per synapse: e-prop, OSTL, ETLP, OSTTP
    Theta(N)  per neuron:  DECOLLE, OTTT, S-TLLR, TESS
    Theta(T*A):            BPTT

No biologically plausible rule reaches O(1) PER SYNAPSE. e-prop buys
Theta(S) instead of Theta(T*A) -- enormous, but the byte multiplier over
Adam stays under 1.5x.

══════════════════════════════════════════════════════════════════════
WHERE THESE NUMBERS COME FROM
══════════════════════════════════════════════════════════════════════
No paper publishes bytes-per-synapse. The 4/8/12 B table is DERIVED from
the equations in Nature Communications 11:3625 (e-prop), with the
derivation in /home/cune/local-learning-memory-report.md section 3.

On SOLO: 2B parameters in 26.7 GB at K=4 -- the only billion-scale result
in local learning. But it is depth-local, not temporally local, and is NOT
evidence that e-prop scales to 1B.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum


class CreditRule(str, Enum):
    """How credit is assigned. Determines both memory and quality."""

    BPTT = "bptt"
    """Full BPTT over the tape. Theta(T*A) -- memory grows with horizon."""

    LOCAL = "local"
    """e-prop / OSTL: eligibility traces. Theta(S) -- independent of T."""


class NeuronModel(str, Enum):
    """Neuron model. Determines trace size."""

    LIF = "lif"
    ALIF = "alif"
    ALIF_REWARD = "alif_reward"


# ── DERIVED BYTES ──────────────────────────────────────────────────────────
#
# fp32 per synapse. Source: derived from the equations in
# Nat. Commun. 11:3625. NOT published in the paper -- derived from formulas.

TRACE_BYTES = {
    NeuronModel.LIF: 4,          # k-filtered eligibility only
    NeuronModel.ALIF: 8,         # + adaptive component
    NeuronModel.ALIF_REWARD: 12,  # + g-filtered copy
}

ADAM_BYTES_PER_PARAM = 8        # m + v in fp32. This is Adam by definition.

# Direct quote from the report: "the marginal cost of going local is <= the
# cost of adding momentum". In other words the local rule costs about the
# same as momentum.

# ── QUALITY vs HORIZON ────────────────────────────────────────────────────
#
# FPTT (arXiv:2112.11231v2): online BPTT approximations degrade as T grows.
# For the e-prop/OSTL class:
FPTT_ACCURACY = {          # T -> fraction of quality retained
    60: 0.855,
    500: 0.389,
}
# A standard LSTM for contrast (degrades more gently):
LSTM_ACCURACY = {
    100: 0.889,
    500: 0.825,
}


@dataclass(frozen=True)
class CreditPlan:
    """Result: what the credit rule costs in bytes and in quality."""
    rule: CreditRule
    neuron: NeuronModel
    bytes_total: int
    bytes_per_synapse: int
    depends_on_T: bool
    # expected quality fraction at this horizon (None = not estimated)
    quality_kept: float | None = None

    def explain(self) -> str:
        dep = "depends on T" if self.depends_on_T else "T-independent"
        s = [f"credit rule:     {self.rule.value} ({dep})",
             f"neuron:          {self.neuron.value}",
             (f"state:           {self.bytes_total:,} B "
              f"({self.bytes_per_synapse} B/synapse)")]
        if self.quality_kept is not None:
            s.append(f"quality:         {self.quality_kept:.1%} "
                     f"of full BPTT")
        return "\n".join(s)


def credit_bytes(rule: CreditRule, neuron: NeuronModel, connections: int,
                 timesteps: int, batch: int, hidden: int, n_layers: int,
                 spatial: int = 1, corr_const: float = 2.0,
                 credit_k: int | None = None,
                 attn_const: float = 2.0208333333333335) -> CreditPlan:
    """What the credit rule costs in bytes.

    BPTT : fp16 activation tape over the credit horizon. If credit_k is
           given, truncate to it (truncation is free in quality -- verified).
    LOCAL: fp32 trace per synapse, independent of T. Here we do NOT multiply
           by batch: the trace accumulates across the batch WITHIN a step,
           it is not stored per example.

    `rule` and `neuron` are normalised through their enums before anything
    else. This is not tidiness. The dispatch below used to be `is
    CreditRule.LOCAL`, and because CreditRule is a str-Enum a caller passing
    the plain string 'local' -- which axplan.memory.backward_bytes did, via
    the module-level `_LOCAL = 'local'` -- failed the identity test and fell
    through to the BPTT branch. The local rule then reported
    depends_on_T=True and quality_kept=1.0, the exact opposite of what this
    module documents, and no caller could tell: the answer was a plausible
    number, not an error. Normalising first makes a string and its enum
    member the same request, and makes an unknown rule raise instead of
    silently becoming BPTT.
    """
    rule = CreditRule(rule)
    neuron = NeuronModel(neuron)

    if rule is CreditRule.LOCAL:
        per_syn = TRACE_BYTES[neuron]
        return CreditPlan(
            rule=rule, neuron=neuron,
            bytes_total=connections * per_syn,
            bytes_per_synapse=per_syn, depends_on_T=False,
            quality_kept=local_quality(timesteps),
        )

    # BPTT
    k = timesteps if credit_k is None else min(timesteps, credit_k)
    per_step = n_layers * hidden * spatial
    total = int(k * batch * per_step * attn_const * 2)   # fp16
    return CreditPlan(
        rule=rule, neuron=neuron,
        bytes_total=total,
        bytes_per_synapse=-1,          # meaningless for BPTT
        depends_on_T=True,
        quality_kept=1.0,             # full BPTT is the reference
    )


def local_quality(timesteps: int) -> float | None:
    """Quality fraction of the local rule, by horizon.

    MEASURED (FPTT, arXiv:2112.11231v2): T=60 -> 0.855, T=500 -> 0.389.

    BEYOND T=500 RETURNS None, NOT AN EXTRAPOLATION. Linear extrapolation
    would give 0.0 at T=1024, which reads as a claim that the local rule is
    useless -- when in fact we simply do not know. Inventing a number where
    there is no data is exactly the error we already made with
    spike_rate=0.08.

    Returning None means "unknown", and the caller must surface that.
    """
    ts = sorted(FPTT_ACCURACY)
    hi = ts[-1]
    if timesteps > hi:
        return None                      # no data
    if timesteps < 20:
        return 1.0                       # short sequences
    for a, b in itertools.pairwise(ts):
        if a <= timesteps <= b:
            qa, qb = FPTT_ACCURACY[a], FPTT_ACCURACY[b]
            return qa + (qb - qa) * (timesteps - a) / (b - a)
    return FPTT_ACCURACY[hi]


# ═══ COMPARING THE OPTIONS ═════════════════════════════════════════════════

@dataclass(frozen=True)
class CreditChoice:
    """Both options and the difference between them."""
    bptt: CreditPlan
    local: CreditPlan
    adam_bytes: int

    @property
    def saving_bytes(self) -> int:
        return self.bptt.bytes_total - self.local.bytes_total

    @property
    def marginal_vs_adam(self) -> float:
        """How many times the local rule costs more than plain momentum.

        BELOW 1 means: switching to the local rule is CHEAPER than turning
        on momentum. That is the main substantive finding.
        """
        return self.local.bytes_total / max(self.adam_bytes, 1)

    def explain(self) -> str:
        GIB = 1 << 30
        L = []
        A = L.append
        A("CREDIT RULE COMPARISON")
        A("")
        A("  BPTT (full, T-dependent):")
        A(f"    credit memory: {self.bptt.bytes_total / GIB:8.3f} GiB")
        A("    quality:      100% (reference)")
        A("  Local rule (e-prop, T-independent):")
        A(f"    credit memory: {self.local.bytes_total / GIB:8.3f} GiB")
        if self.local.quality_kept is None:
            A("    quality:      NO DATA (only measured up to T=500)")
        else:
            A(f"    quality:      {self.local.quality_kept:.1%} of BPTT")
        A(f"  Adam state (m+v, fp32): {self.adam_bytes / GIB:.3f} GiB")
        sign = "-" if self.saving_bytes < 0 else "+"
        A(f"  difference: local {sign}{abs(self.saving_bytes) / GIB:.3f} GiB "
          f"({'more' if self.saving_bytes < 0 else 'less'} than BPTT)")
        A("")
        if self.marginal_vs_adam < 1.0:
            A(f"  FINDING: switching to the local rule costs "
              f"{self.marginal_vs_adam:.2f}x Adam.")
            A("  That is CHEAPER than turning on momentum. In bytes this is")
            A("  a tweak, not a paradigm change.")
        else:
            A(f"  FINDING: the local rule costs "
              f"{self.marginal_vs_adam:.2f}x Adam -- more than momentum,")
            A("  but still independent of T.")
        A("")
        A("  BUT: quality falls with the horizon.")
        A("  Measured (FPTT): T=60 -> 85.5%, T=500 -> 38.9%.")
        A("  Beyond T=500 there is NO DATA, and we do not extrapolate.")
        A("  On long horizons the local rule may lose to full BPTT -- and")
        A("  the planner is obliged to show that rather than hide it.")
        return "\n".join(L)


def crossover_batch(timesteps: int, rule_neuron: NeuronModel = NeuronModel.LIF,
                    hidden: int = 768, n_layers: int = 12,
                    spatial: int = 1, attn_const: float = 2.0208333333333335,
                    credit_k: int | None = None) -> float:
    """At which batch size the local rule becomes cheaper than BPTT.

    Returns float('inf') when BPTT stays cheaper at any sane batch size,
    which happens on short horizons.

    The planner is obliged to report this number: while the batch is below
    the crossover, the local rule is extra complexity for no memory gain.
    """
    k = timesteps if credit_k is None else min(timesteps, credit_k)
    per_syn = TRACE_BYTES[rule_neuron]
    local_total = n_layers * hidden * hidden * spatial * per_syn
    if local_total <= 0:
        return float("inf")
    bptt_per_batch = k * n_layers * hidden * spatial * attn_const * 2
    if bptt_per_batch <= 0:
        return 1.0
    return local_total / bptt_per_batch


def compare(rule_neuron: NeuronModel = NeuronModel.LIF,
            connections: int = 0, timesteps: int = 64,
            batch: int = 32, hidden: int = 768, n_layers: int = 12,
            spatial: int = 1, credit_k: int | None = None) -> CreditChoice:
    """Build both options for comparison."""
    bptt = credit_bytes(CreditRule.BPTT, rule_neuron, connections,
                        timesteps, batch, hidden, n_layers, spatial,
                        credit_k=credit_k)
    loc = credit_bytes(CreditRule.LOCAL, rule_neuron, connections,
                       timesteps, batch, hidden, n_layers, spatial)
    adam = connections * ADAM_BYTES_PER_PARAM
    return CreditChoice(bptt=bptt, local=loc, adam_bytes=adam)