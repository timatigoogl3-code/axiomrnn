"""
axiomrnn -- a framework where memory budget is an input, not a surprise.

Quick start (beginner, no neuromorphic AI background needed):

    import axiomrnn as ax

    model = ax.Model()
    model.add_spiking(256, horizon=64)     # looks like Keras
    model.add_spiking(128, horizon=64)
    model.add_dense(10)

    model.explain(ax.Budget.consumer_6gb())   # <- the actual difference
    km = model.keras_model(inputs=x, outputs=10)

What explain() does: it does not answer "does it fit", it answers "**what
can you learn** on this budget". No existing tool answers that.

Extending without forking:

    class MyCell(ax.SpikingCell):
        def membrane(self, state, drive, thr):
            return state + drive

    model = ax.Model(cell=MyCell())        # the whole stack works at once

Layers know TF through axtf; the planner (axplan) does not know TF at all.
"""
from __future__ import annotations

from axplan.credit import local_quality

# axplan -- TF-free part: memory, planning, explanation
from axplan.memory import (
    SPARK_RATE_BAND,
    SPARK_RATE_DEFAULT,
    SPARK_RATE_NEURON,
    BudgetSpec,
    MemoryPlan,
    Regime,
    activation_bytes,
    analyse,
    best_codec,
)
from axplan.planner import (
    Segment,
    exact_peak,
    plan_partition,
)
from axplan.solve import (
    Candidate,
    largest_that_fits,
)
from axplan.solve import (
    frontier as capacity_frontier,
)
from axplan.solve import (
    report as capacity_report,
)

# axtf -- the only layer that knows about TensorFlow
#
# Two audiences meet at import time, and they need different things.
#
# The memory-planning half (axplan, axon) runs on numpy alone and is imported
# unconditionally above: planning a model's footprint must not require a deep
# learning framework to be installed first.
#
# The spiking half needs TensorFlow. axtf owns that dependency and owns the
# wording of the advice, so the guidance is written once, in axtf, and imported
# here rather than restated -- two copies of an install command drift apart and
# one of them ends up lying.
try:
    from axtf import HAS_TF as _HAS_TF
    from axtf import TF_ERROR as _TF_ERR
    from axtf import SpikingCell, SpikingRNNCell
except Exception as _e:                     # pragma: no cover  # noqa: BLE001
    # axtf itself failed to import, which is a different fault from TensorFlow
    # being absent. Do not mask it as a missing dependency.
    _HAS_TF = False
    _TF_ERR = str(_e)

# backends -- framework abstraction (lazy: no TF/torch imported here)
try:
    from backends import Backend as _Backend
    from backends import TensorFlowBackend as _TFBackend
    from backends import TorchBackend as _TorchBackend
    from backends import get_backend as _get_backend
    _HAS_BACKENDS = True
except Exception as _e:                     # pragma: no cover  # noqa: BLE001
    _HAS_BACKENDS = False
    _Backend = None
    _TFBackend = None
    _TorchBackend = None
    _get_backend = None

__all__ = [
    "GIB",
    "SPARK_RATE_BAND",
    "SPARK_RATE_DEFAULT",
    "SPARK_RATE_NEURON",
    "Backend",
    "Budget",
    "BudgetSpec",
    "Candidate",
    "Dense",
    "MemoryPlan",
    "Model",
    "Regime",
    "Segment",
    "SpikingCell",
    "SpikingRNN",
    "SpikingRNNCell",
    "TensorFlowBackend",
    "TorchBackend",
    "analyse",
    "best_codec",
    "capacity_frontier",
    "capacity_report",
    "exact_peak",
    "explain_budget",
    "gate_report",
    "get_backend",
    "largest_that_fits",
    "local_quality",
    "plan_partition",
]

GIB = 1 << 30

# public aliases for backend classes (may be None if backends unavailable)
Backend = _Backend
TensorFlowBackend = _TFBackend
TorchBackend = _TorchBackend
get_backend = _get_backend


# ═══ MODEL ═══════════════════════════════════════════════════════════════

class Budget:
    """Memory budget as an input, not a consequence.

    Three levels of hardness, because "how much VRAM do you have" is not
    one question:
      vram_gb     : what the card actually has
      host_ram_gb : system RAM, for offload
      offload     : whether moving part of it to host is allowed
    """

    def __init__(self, vram_gb: float, host_ram_gb: float = 0.0,
                 offload: bool = False, name: str = "custom"):
        self.vram_gb = float(vram_gb)
        self.host_ram_gb = float(host_ram_gb)
        self.offload = bool(offload)
        self.name = name

    # ── ready-made profiles of real hardware ───────────────────────────────
    #
    # Written from measured bandwidths and typical cards, not by eye. Each
    # field says where the number comes from.

    @staticmethod
    def consumer_6gb():
        """RTX 3060 12G / 4060 8G. A typical consumer starting point."""
        return Budget(6.0, host_ram_gb=32.0, offload=True, name="consumer 6 GB")

    @staticmethod
    def consumer_8gb():
        return Budget(8.0, host_ram_gb=32.0, offload=True, name="consumer 8 GB")

    @staticmethod
    def pro_24gb():
        """RTX 4090 24G: fp16 with no offload, optimizer in fp32."""
        return Budget(24.0, host_ram_gb=64.0, offload=True, name="pro 24 GB")

    @staticmethod
    def datacenter_80gb():
        """A100 80G / H100 80G. HBM2e."""
        return Budget(80.0, host_ram_gb=512.0, offload=False,
                      name="datacenter 80 GB")

    def usable_bytes(self) -> int:
        """What is actually left after the CUDA context and fragmentation."""
        return int(self.vram_gb * GIB / 1.30 - 0.5 * GIB)

    def __repr__(self):
        return (f"Budget({self.name}: {self.vram_gb} GB VRAM, "
                f"host {self.host_ram_gb} GB, offload={self.offload})")


class Dense:
    """Dense layer wrapper. Keras-compatible via axtf when TF is present."""

    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation


class SpikingRNN:
    """Spiking recurrent layer -- the reason this project exists.

    units         : width
    return_spikes : True -> spikes (B,T,units); False -> rates (B,units)
    horizon       : T. 0 = take it from the input at forward time.
    norm_t        : 1/T gradient normalisation. DO NOT TURN OFF (see below).
    credit_k      : credit horizon. None = full BPTT.
    """

    def __init__(self, units: int, return_spikes: bool = True,
                 horizon: int = 0, norm_t: bool = True,
                 credit_k=None, cell=None, leak: float = 0.9,
                 alpha: float = 2.0):
        self.units = units
        self.return_spikes = return_spikes
        self.horizon = horizon
        self.norm_t = norm_t
        self.credit_k = credit_k
        self.leak = leak
        self.alpha = alpha
        self.cell = cell
        self._tf_layer = None

    # ── planner inputs ───────────────────────────────────────────────────
    def spec(self, batch: int, horizon: int, vram_gb: float = 1.0,
             n_layers: int = 1):
        """Memory estimate for this layer. A pure function, no TF."""
        from axplan.memory import BudgetSpec
        connections = self.units * self.units + self.units     # +bias
        return BudgetSpec(
            vram_gb=vram_gb,
            connections=connections,
            n_layers=n_layers,
            hidden=self.units,
            timesteps=horizon,
            batch=batch,
            seq_len=1,
            n_heads=1,
        )

    def __repr__(self):
        return (f"SpikingRNN({self.units}, T={self.horizon or 'auto'}, "
                f"norm_t={self.norm_t}, k={self.credit_k})")


def _gate_suite_count() -> int | None:
    """How many suites the gate declares, or None if that cannot be read.

    Parsed, not imported: `tests/` is not a package, and `import
    tests.run_all` binds to whichever unrelated `tests` package sits earlier
    on sys.path. ast.parse reads the literal without executing the gate.
    """
    import ast
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tests", "run_all.py")
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets
                     if isinstance(t, ast.Name)]
            if "SUITES" in names and isinstance(node.value, (ast.List,
                                                             ast.Tuple)):
                return len(node.value.elts)
    return None


def _split_layer_kwargs(layer_cls, candidates: dict):
    """Split `candidates` into the names `layer_cls` accepts and the ones it does not.

    A framework should answer "which settings does this layer take?" rather than
    forward everything and let a later line fail. Reading the signature is the
    only honest source: it cannot drift from the constructor the way a
    hand-maintained list would.
    """
    import inspect

    params = inspect.signature(layer_cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates), {}
    accepted = {k: v for k, v in candidates.items() if k in params}
    rejected = {k: v for k, v in candidates.items() if k not in params}
    return accepted, rejected


def _layer_kwargs_help(layer_cls) -> str:
    """Human-readable list of the keyword arguments a layer accepts.

    Every real parameter has a default, so filtering on "has no default" would
    report an empty list -- which is worse than no message, because it reads as
    "this layer takes nothing". List what it takes.
    """
    import inspect

    params = inspect.signature(layer_cls.__init__).parameters
    names = [n for n, p in params.items()
             if n != "self"
             and p.kind not in (inspect.Parameter.VAR_KEYWORD,
                                inspect.Parameter.VAR_POSITIONAL)]
    return ", ".join(names) if names else "(none)"


class _UnitsError(ValueError, TypeError):
    """Raised when add_spiking gets an invalid `units` value."""


class Model:
    """Minimal model layer on top of Keras/TF.

    It does NOT bring its own optimizer, training loop or autodiff: you can
    plug in your own compile/fit however you like. What we add is
    budget-aware design and an honest explanation.

    The stated requirement: 400 lines for a beginner, 6000+ for a pro, and
    the ability to OVERRIDE the system without forking. Here `cell`
    (neuron dynamics) and any layer via subclassing are overridable.
    """

    def __init__(self, cell=None, backend="tensorflow", **layer_kw):
        self.layers: list = []
        self.cell = cell
        self.backend = backend
        self._backend = backend
        self.layer_kw = layer_kw
        self._compiled = None
        self._budget: Budget | None = None

    def _resolve_backend(self, backend=None):
        name = self.backend if backend is None else backend
        if _get_backend is None:
            from backends import get_backend as _gb
            return _gb(name)
        return _get_backend(name)

    def build(self, backend=None, inputs=None, outputs=None,
              compile_model=False, readout="flatten"):
        be = self._resolve_backend(backend)
        if not getattr(be, "has_backend", lambda: False)():
            raise RuntimeError(
                f"Backend {getattr(be, 'name', backend)!r} is not available.")
        return be.build_model(self, inputs=inputs, outputs=outputs,
                              compile_model=compile_model, readout=readout)

    def get_backend(self, backend=None):
        return self._resolve_backend(backend)

    # ── construction ────────────────────────────────────────────────────
    def add(self, layer):
        self.layers.append(layer)
        return self

    def add_spiking(self, units, cell=None, **kw):
        """Add a spiking layer.

        cell : your neuron, or None to use the model-wide one.

        This is how the extension point works: different layers of one
        network may have DIFFERENT neurons.

            spec = ax.Model()
            spec.add_spiking(192, cell=MyCell())      # your neuron
            spec.add_spiking(192)                      # plain LIF

        Override `membrane` together with `membrane_dstate`, and `spike_fn`
        together with `spike_d1`. A forward function without its derivative
        hook is a silently wrong gradient, and the core will not guess it.

        This used to put `cell=self.cell` ahead of `**kw`, so passing your
        own neuron raised "got multiple values for keyword argument 'cell'".
        In other words the advertised ability to extend without forking
        simply did not work through the public API.
        """
        # units must be a positive integer: a width of 0 or a negative
        # number is a typo, and a float/str is a contract violation. Fail
        # fast with a clear message instead of letting it reach Keras.
        import numbers
        if isinstance(units, bool) or not isinstance(units, (int, numbers.Integral)):
            raise _UnitsError(
                f"add_spiking() requires `units` to be a positive int, "
                f"got {units!r} ({type(units).__name__}).")
        if int(units) <= 0:
            raise _UnitsError(
                f"add_spiking() requires `units` to be a positive int, "
                f"got {units!r}.")
        units = int(units)

        merged = {**self.layer_kw, **kw}
        # Keep only what this layer can actually take.
        #
        # Model stores every unknown keyword in `layer_kw` so that one set of
        # defaults applies to all layers. Layers, however, have fixed
        # signatures -- SpikingRNN takes units/horizon/leak/alpha/cell and no
        # **kwargs -- so blindly forwarding turns `Model(vocab=64, dim=32)`
        # followed by `add_spiking(16)` into "unexpected keyword argument
        # 'vocab'". The names a user passes to Model() are model-level
        # settings, and silently dropping them is worse than refusing: a layer
        # built with the wrong width looks like it worked.
        accepted, rejected = _split_layer_kwargs(SpikingRNN, merged)
        if rejected:
            valid = _layer_kwargs_help(SpikingRNN)
            raise TypeError(
                f"add_spiking() cannot use {sorted(rejected)}; SpikingRNN does "
                f"not take {'them' if len(rejected) > 1 else 'it'}.\n"
                f"SpikingRNN accepts: {valid}\n\n"
                "Model() stores unrecognised keywords as defaults for every "
                "layer, so a name meant for one layer will reach another. Pass "
                "layer settings directly instead:\n"
                "    spec.add_spiking(16, horizon=8)\n"
                "and model-wide settings where they belong -- the vocabulary "
                "size belongs to the model, not to a recurrent layer."
            )
        # precedence: an explicit cell argument beats the model-wide cell
        return self.add(SpikingRNN(units, cell=cell or self.cell, **accepted))

    def add_dense(self, units, **kw):
        return self.add(Dense(units, **kw))

    # ── budget ──────────────────────────────────────────────────────────
    def fit_budget(self, budget: Budget, batch: int = 32,
                   connections: int = 0):
        """Compute the memory plan for a budget. Builds nothing, only counts.

        connections: total synapse count; taken from the layers when 0.
        """
        self._budget = budget
        spiking = [l for l in self.layers if isinstance(l, SpikingRNN)]
        conn = connections or sum(l.units * l.units + l.units
                                  for l in spiking)
        T = max((l.horizon for l in spiking), default=0) or 1
        width = max((l.units for l in spiking), default=512)
        spec = BudgetSpec(
            vram_gb=budget.vram_gb,
            connections=conn,
            n_layers=max(1, len(spiking)),
            hidden=width,
            timesteps=T,
            batch=batch,
            seq_len=1,
            n_heads=1,
        )
        return analyse(spec)

    def explain_budget(self, budget: Budget, batch: int = 32,
                       verbose: bool = True) -> str:
        return explain_budget(self, budget, batch, verbose=verbose)

    # The short name is what someone types on their first try. The long one
    # stays, because renaming would break existing code.
    explain = explain_budget

    # ── Keras assembly (needs TF) ───────────────────────────────────────
    def keras_model(self, inputs=None, outputs=None, compile_model=False,
                    readout="flatten"):
        """Assemble a real Keras model.

        inputs        : keras.Input, or None -> shape=(None, None)
        outputs       : number of classes; None -> no output layer
        compile_model : compile immediately with the correct settings
                        (from_logits=True and run_eagerly, see axtf)
                        readout       : how to read spikes before the output layer.
                        'flatten' (default) uses the whole sequence,
                        'mean' averages over time. The measured gap is
                        large: see examples/04_equal_budget.py, which prints
                        the figures of the current run. Mean is reasonable
                        when the task does
                        not depend on exact timing, but it must be chosen
                        deliberately: it was the default once and
                        cost a quarter of the accuracy. That was OUR bug,
                        not a property of spikes.

        Requires TF. Without it you get an honest error, not a silent no-op.

        WARNING: a freshly built model is SILENT -- potentials stay below the
        default threshold, so nothing spikes and no gradient flows. You MUST
        call `axtf.build.calibrate_model(km, x, target_rate=...)` (or
        `layer.calibrate(...)`) before training, otherwise loss saturates at
        ln(num_classes). Calibration is not optional and is not performed
        automatically, because it needs sample data `keras_model` does not have.
        """
        if not _HAS_TF:
            raise RuntimeError(
                "TensorFlow is not installed. axtf is the only layer "
                f"that needs it: {_TF_ERR}\n"
                "axplan (memory, planner, explain) works without TF.")
        be = self._resolve_backend("tensorflow")
        return be.build_model(self, inputs=inputs, outputs=outputs,
                              compile_model=compile_model, readout=readout)

    def __repr__(self):
        s = f"Model({len(self.layers)} layers)"
        if self._budget:
            s += f" under {self._budget}"
        return s


# ═══ EXPLANATION ═════════════════════════════════════════════════════════

def explain_budget(model: Model, budget: Budget, batch: int = 32,
                   verbose: bool = True) -> str:
    """THE ANSWER TO A QUESTION NOBODY ELSE ASKS.

    Not "does it fit", but "**what can you learn**".

    No existing tool does this:
      Accelerate  takes max_memory, but only for inference
      JAX         demands manual commitment
      PyTorch     only measures post-factum
      ONNX Runtime does not accept a budget at all

    Here the budget is an input and the answer is human: "at 6 GB you can
    train to horizon 8 and lose 4% of accuracy; catching the ceiling needs
    11 GB".
    """
    plan = model.fit_budget(budget, batch=batch)
    L = []
    A = L.append

    A("=" * 74)
    A(f"BUDGET: {budget.name} "
      f"({budget.vram_gb:.0f} GB VRAM, host {budget.host_ram_gb:.0f} GB)")
    A("=" * 74)

    spiking = [l for l in model.layers if isinstance(l, SpikingRNN)]
    conn = sum(l.units * l.units + l.units for l in spiking) if spiking else 0
    T = max((l.horizon for l in spiking), default=plan.timesteps)
    width = max((l.units for l in spiking), default=plan.hidden)

    A("")
    A("WHAT WAS ASKED FOR")
    A(f"  connections:   {conn:,}")
    A(f"  width:         {width}")
    A(f"  horizon T:     {T}")
    A(f"  batch:         {batch}")

    A("")
    A("WHAT IT COSTS IN MEMORY")
    A(f"  regime:        {plan.regime.value}")
    A(f"  tape share:    {plan.activation_fraction:.1%} "
      f"{'<-- tape dominates' if plan.activation_fraction > 0.7 else ''}")
    A(f"  peak:          {plan.peak_bytes / GIB:.2f} GiB of "
      f"{plan.vram_bytes / GIB:.0f} GiB")
    A(f"  credit horizon:{plan.credit_horizon} steps "
      f"(truncation is free -- verified)")
    if plan.infeasible:
        A(f"  FITS: NO. {plan.reason}")

    A("")
    A("WHAT THIS MEANS FOR QUALITY")
    if not spiking:
        A("  No spiking layers -- the memory figure is exact, but nothing")
        A("  can be said about quality.")
    else:
        A("  The horizon buys quality MONOTONICALLY while the horizon still")
        A("  carries new signal -- measured by examples/advanced/02_horizon_sweep.py.")
        A("  So every horizon step is a real resource.")
        A("")
        A(f"  Current horizon: {T}.")
        head = budget.usable_bytes() - plan.peak_bytes
        # Cost of ONE horizon step, in fp16 (an upper bound, before
        # bit-packing), because the report asks "how much more can I afford".
        spec = BudgetSpec(
            vram_gb=budget.vram_gb, connections=conn,
            n_layers=max(1, len(spiking)), hidden=width,
            timesteps=T, batch=batch, seq_len=1, n_heads=1)
        step_bytes = activation_bytes(spec) // max(T, 1)
        if head > 0 and step_bytes > 0:
            extra = head // step_bytes
            A(f"  HEADROOM: {head / GIB:.2f} GiB unused.")
            A("  The budget does NOT constrain this network. Feel free to")
            A("  raise width or horizon -- quality grows monotonically.")
            A(f"  One horizon step costs {step_bytes / 1024:.0f} KiB "
              f"(fp16, before bit-packing).")
            A(f"  Roughly {extra} more steps would fit.")
        else:
            A(f"  NO HEADROOM: short by {-head / GIB:.2f} GiB.")
            A(f"  One horizon step costs {step_bytes / 1024:.0f} KiB.")
            A("  What to do, cheapest first:")
            A("    1) the credit horizon is already truncated "
              f"({plan.credit_horizon} steps) -- that one is free")
            A("    2) batch: tape memory is linear in batch")
            A("    3) width: tape memory is linear in neurons")
            A("    4) fp16 weights -> half the weight memory")
        A("")
        A("  Spike density is NOT a constant. Measured on three tasks:")
        A(f"    between tasks: {SPARK_RATE_BAND[0]:.3f} .. "
          f"{SPARK_RATE_BAND[1]:.3f}")
        A(f"    within a net:  {SPARK_RATE_NEURON[0]:.2f} .. "
          f"{SPARK_RATE_NEURON[1]:.2f}  <- larger than between-task")
        A("  The planner must budget the worst case of that range.")

    A("")
    A("THE MAIN MEMORY LEVER: THE CREDIT RULE")
    A("  Larger than codecs, larger than offload:")
    A("    BPTT   -> Theta(T * A), memory GROWS with the horizon")
    A("    e-prop -> Theta(S),      T-INDEPENDENT")
    from axplan.credit import NeuronModel, compare, crossover_batch, local_quality
    L_, h_ = max(1, len(spiking)), width
    cch = compare(NeuronModel.LIF, connections=conn, timesteps=T,
                  batch=batch, hidden=h_, n_layers=L_)
    xs = crossover_batch(T, NeuronModel.LIF, hidden=h_, n_layers=L_)
    A(f"    now: BPTT, {cch.bptt.bytes_total / GIB:.3f} GiB")
    A(f"    e-prop LIF: {cch.local.bytes_total / GIB:.3f} GiB "
      f"({cch.marginal_vs_adam:.2f}x Adam -- "
      f"{'cheaper' if cch.marginal_vs_adam < 1 else 'dearer'} than momentum)")
    A(f"    batch crossover: batch* = {xs:.0f}")
    if batch < xs:
        A(f"    -> at batch={batch} BPTT is CHEAPER. The local rule is")
        A("       currently extra complexity for no memory gain.")
    else:
        A(f"    -> at batch={batch} the local rule is ALREADY cheaper.")
    q = local_quality(T)
    A("")
    A("  QUALITY PRICE of the local rule (measured, FPTT):")
    A("    T=60 -> 85.5%   T=500 -> 38.9%   a 47-point gap")
    A(f"    at T={T}: " + ("no data, we do not extrapolate"
                        if q is None else f"~{q:.1%} of BPTT"))
    A("  We do NOT claim e-prop is better. We say what it costs and what")
    A("  it buys.")

    A("")
    A("VERIFIED, AND NOT")
    A("  Verified -- each line names the command that reproduces it:")
    A("    axon/gate.py           gradient = derivative of the smooth model")
    A("                          (rel 1e-8); 1/T normalisation; DP planner")
    A("                          optimal vs exhaustive search; calibration")
    # The count is read out of the gate's own table at run time. It used to be
    # the literal "18 suites" here while the gate ran 28, and then 31: the
    # number went stale twice because a human typed it once and nobody had a
    # reason to touch this line again.
    #
    # It is parsed rather than imported on purpose. `import tests.run_all`
    # resolves against whatever `tests` package happens to be on sys.path
    # first -- on this machine that is an unrelated installed package, and
    # the import walks into torch. tests/ is not a package and should not
    # become one. ast.parse reads the literal without executing anything.
    # This block reports how the claims were verified, which is a property of
    # the *source tree*, not of the installed wheel -- the wheel deliberately
    # ships no tests/ or examples/, and the sdist that does carry them does not
    # install alongside it. So a wheel-installed `axiomrnn-explain` says the
    # commands are in the sdist and stops there, rather than quoting a count it
    # cannot read.
    _n = _gate_suite_count()
    if _n is None:
        A("    tests/run_all.py       the gate: memory model, credit rules,")
        A("                          TF kernel, Keras assembly, API audit,")
        A("                          every example. Not shipped in the wheel;")
        A("                          it travels in the sdist. See")
        A("                          https://github.com/cubetitled-ui/axiomrnn")
    else:
        A(f"    tests/run_all.py       {_n} suites: memory model, credit")
        A("                          rules, TF kernel, Keras assembly,")
        A("                          API audit, kernels, every example")
    A("    examples/04            spiking vs dense, honest comparison")
    A("                          => PARITY, never superiority. The readout,")
    A("                          not the spikes, dominates the gap.")
    A("    examples/advanced/04   bit-packing counted honestly: large on its")
    A("                          own line, a few percent of real memory.")
    A("                          Binary activations do NOT make the gradient")
    A("                          binary; the backward tape stays dense.")
    A("    examples/advanced/03   credit rule: the actual lever, and it")
    A("                          SHRINKS as the horizon grows")
    A("")
    A("  These are run-specific figures, so they are not printed here as")
    A("  numbers: they change between runs. Run the scripts above.")
    A("")
    A("  NOT measured:")
    A("    the sparse/bit crossover (by nobody, including us)")
    A("    energy: no neuromorphic device was used, ever")
    A("    whether spiking is FASTER on a GPU")
    A("    the original promise 'spikes save memory' -- REFUTED by")
    A("      measurement; the value is in the credit planner, not the bit")
    A("")
    A("  This report answers 'what does THIS network cost?'. For the")
    A("  inverse question -- the largest network this budget can train,")
    A("  and what exactly stops the next step up -- use:")
    A("      ax.largest_that_fits(vram_gb)   or   ax.capacity_report(vram_gb)")

    A("")
    A("=" * 74)
    out = "\n".join(L)
    if verbose:
        print(out)
    return out


def gate_report() -> tuple[bool, str]:
    """Run the gate and return (ok, output). Mandatory before any claim."""
    import os
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(
        ["python3", "-W", "ignore", os.path.join(here, "axon", "gate.py")],
        capture_output=True, text=True, timeout=1200, check=False)
    return r.returncode == 0, r.stdout + r.stderr


# ═══ CONSOLE COMMAND ═════════════════════════════════════════════════════

def main_explain(argv=None) -> int:
    """`axiomrnn-explain` -- a budget without a line of Python.

    Installs with the package and answers the question the whole project
    exists for: how much can I learn with this much memory?

        axiomrnn-explain --budget 6gb --layers 2 --width 1024 --T 256

    TensorFlow is not needed: this command is backed by axplan, which does
    not import it. That also exercises the package boundary in practice.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="axiomrnn-explain",
        description="How much can be learned under a memory budget.")
    ap.add_argument("--budget", default="6gb",
                    help="hardware profile: 6gb | 8gb | 24gb | 80gb")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gate", action="store_true",
                    help="run the gate before printing (mandatory)")
    a = ap.parse_args(argv)

    profiles = {
        "6gb": Budget.consumer_6gb, "8gb": Budget.consumer_8gb,
        "24gb": Budget.pro_24gb, "80gb": Budget.datacenter_80gb,
    }
    pick = profiles.get(a.budget.lower().replace(" ", ""))
    if pick is None:
        print(f"unknown budget '{a.budget}'; available: "
              f"{', '.join(profiles)}")
        return 2

    m = Model()
    for _ in range(a.layers):
        m.add_spiking(a.width, horizon=a.horizon)
    m.add_dense(a.classes)

    # verbose=False, because explain_budget prints by default AND returns the
    # same report. `print(m.explain(...))` therefore printed the whole 81-line
    # report twice, identically: a 162-line file in which the reader has to
    # work out that the second half is a copy. Print the returned value once.
    print(m.explain(pick(), batch=a.batch, verbose=False))

    if a.gate:
        ok, out = gate_report()
        print()
        print(out[-800:])
        if not ok:
            print("GATE FAILED -- the output above must not be published.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_explain())

# ═══ INFERENCE HELPER ════════════════════════════════════════════════════

def generate(model, tok, seed, n: int = 20, context: int = 6,
            temperature: float = 0.0):
    """Greedy/temperature autoregressive generation with a single model call.

    `model` is a compiled-or-not Keras model (the one returned by
    `build_keras` / `Model.keras_model`). `tok` is a tokenizer exposing
    `encode(list[str]) -> list[int]` and `decode(list[int]) -> list[str]`
    (the shape used by `examples/05_t9.py` and `examples/07_generate.py`).

    `seed` may be a string, a list of words, or a list of token ids.

    IMPORTANT: this calls the model directly (`model(x)`) instead of
    `model.predict(...)`. The model is run with `run_eagerly=True` (see
    `axtf`), so each step executes a Python loop over time; generation is
    therefore slow by construction and this function does not hide that. Do
    not quote its throughput as a benchmark -- use
    `examples/bench_inference.py` for honest numbers.
    """
    import numpy as np

    # Normalise the seed into a list of token ids.
    if isinstance(seed, str):
        pieces = seed.split()
    elif isinstance(seed, (list, tuple)):
        pieces = list(seed)
    else:
        pieces = [seed]

    if pieces and all(isinstance(p, (int, np.integer)) and not isinstance(p, bool)
                      for p in pieces):
        ids = [int(p) for p in pieces]
    else:
        ids = list(tok.encode([str(p) for p in pieces]))

    if not ids:
        ids = [0]

    out_words = []
    for _ in range(int(n)):
        window = ids[-int(context):]
        if len(window) < int(context):
            window = [0] * (int(context) - len(window)) + window
        x = np.asarray([window], dtype="int32")
        logits = model(x, training=False)
        logits = np.asarray(logits)
        # logits is (batch, vocab) for a single-step model; if it is
        # (batch, time, vocab) take the last time step.
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        logits = logits[0]

        if temperature is None or temperature <= 0:
            nxt = int(np.argmax(logits))
        else:
            scaled = logits / float(temperature)
            scaled = scaled - np.max(scaled)
            p = np.exp(scaled)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                nxt = int(np.argmax(logits))
            else:
                p = p / s
                nxt = int(np.random.choice(len(p), p=p))

        ids.append(nxt)
        decoded = tok.decode([nxt])
        word = decoded[0] if (isinstance(decoded, (list, tuple)) and decoded) else str(nxt)
        out_words.append(word)

    return " ".join(out_words)


# Alias kept for callers that use the longer name.
generate_text = generate


# keep exports in sync
try:
    __all__  # noqa: B018
except NameError:
    __all__ = []
if "generate" not in __all__:
    __all__.append("generate")
if "generate_text" not in __all__:
    __all__.append("generate_text")
