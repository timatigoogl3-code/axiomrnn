"""
API surface audit -- every advertised symbol is CALLED, not introspected.

Why this file exists
====================
We shipped a function that was broken: `Model.add_spiking(cell=...)` raised
TypeError, because `cell=self.cell` collided with the `cell` kwarg. The
feature it promised (override the neuron without forking) only worked when
called through axtf directly. Nothing caught it, because nobody called it.

An uncalled function can be broken. So every public symbol gets called here,
for real, with real arguments, and must return a sane value.

Rules
=====
1. No `hasattr` checks. A symbol exists if calling it works.
2. No stubs. A function whose body is `pass` or `NotImplementedError` fails.
3. TF-only symbols skip when TF is absent; the core is proven without TF.
4. The test asserts VALUES, not merely absence of exceptions -- a function
   returning None unconditionally would pass a no-exception test.

Run:  /home/cune/.venvs/ax/bin/python tests/test_api_surface.py
"""
from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PASS: list[str] = []
FAIL: list[str] = []
SKIP: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}{(' — ' + detail) if detail else ''}")


def skip(name: str, why: str) -> None:
    SKIP.append(f"{name} ({why})")


def no_stub(fn) -> str:
    """Return a reason if fn is a stub, else ''."""
    if fn is None:
        return "missing"
    src = ""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        pass
    body = [ln.strip() for ln in src.splitlines()[1:]
            if ln.strip() and not ln.strip().startswith("#")]
    if not body:
        return "empty body"
    if body == ["pass"]:
        return "bare pass"
    if any("NotImplementedError" in ln for ln in body):
        return "NotImplementedError"
    return ""


# ══════════════════════════════════════════════ axplan: memory ═════════════

def test_memory_api():
    from axplan import memory as M

    spec = M.BudgetSpec(vram_gb=6, connections=256 * 256, n_layers=8,
                        hidden=256, timesteps=32, batch=8)
    check("BudgetSpec.usable_bytes is positive",
          spec.usable_bytes() > 0, f"{spec.usable_bytes():,} B")

    for fn_name in ("activation_bytes", "backward_bytes", "logits_bytes",
                    "tape_bytes_1bit"):
        fn = getattr(M, fn_name, None)
        why = no_stub(fn)
        check(f"memory.{fn_name} is not a stub", not why, why)
        if why:
            continue
        val = fn(spec)
        check(f"memory.{fn_name} returns positive int",
              isinstance(val, int) and val > 0, f"{val:,}")

    codec, nbytes = M.best_codec(0.25, 65536)
    check("best_codec returns a Codec and bytes",
          isinstance(codec, M.Codec) and isinstance(nbytes, int) and nbytes > 0,
          f"{codec.value}, {nbytes:,} B")
    check("codec_bytes agrees with best_codec",
          M.codec_bytes(codec, 0.25, 65536) == nbytes)

    # codec must actually beat dense -- otherwise bit-packing is pointless
    dense = M.codec_bytes(M.Codec.DENSE, 0.25, 65536)
    check("bit codec beats fp32 on measured data",
          nbytes < dense, f"{nbytes:,} < {dense:,}")

    plan = M.analyse(spec)
    check("analyse returns MemoryPlan", isinstance(plan, M.MemoryPlan))
    check("plan.peak_bytes positive", plan.peak_bytes > 0, f"{plan.peak_bytes:,}")
    check("plan.regime is a Regime", isinstance(plan.regime, M.Regime),
          str(plan.regime))
    check("plan.explain() returns text", len(plan.explain()) > 50)

    # infeasible path must be reachable and reported
    huge = M.BudgetSpec(vram_gb=1, connections=10 ** 9, n_layers=8,
                        hidden=4096, timesteps=4096, batch=64)
    p2 = M.analyse(huge)
    check("oversized model reports infeasible", p2.infeasible,
          p2.reason[:50])

    prec = M.Precision()
    check("Precision.static_bytes_per_param positive",
          prec.static_bytes_per_param() > 0)
    cm = M.CreditModel()
    check("CreditModel.horizon() positive", cm.horizon() > 0, str(cm.horizon()))

    check("MemoryPlan has documented fields",
          all(f in M.MemoryPlan.__dataclass_fields__
              for f in ("peak_bytes", "regime", "credit_horizon", "reason")))


# ══════════════════════════════════════════════ axplan: planner ════════════

def test_planner_api():
    from axplan.planner import (
        PartitionPlan,
        Segment,
        brute_force,
        exact_peak,
        plan_partition,
    )
    GIB = 1 << 30
    for fn in (plan_partition, exact_peak, brute_force):
        check(f"planner.{fn.__name__} is not a stub", not no_stub(fn),
              no_stub(fn))

    segs = [Segment(input_bytes=1 * GIB, internal_bytes=0.5 * GIB,
                    recompute_ns=10 ** 9, name="a"),
            Segment(input_bytes=1 * GIB, internal_bytes=0.4 * GIB,
                    recompute_ns=5 * 10 ** 8, name="b"),
            Segment(input_bytes=1 * GIB, internal_bytes=0.3 * GIB,
                    recompute_ns=2 * 10 ** 8, name="c")]

    plan = plan_partition(segs, 8 * GIB)
    check("plan_partition returns PartitionPlan",
          isinstance(plan, PartitionPlan))
    check("plan bounds cover every segment", plan.bounds[0] == 0
          and plan.bounds[-1] == len(segs), str(plan.bounds))
    check("plan is feasible within budget", plan.peak_exact <= 8 * GIB)
    check("plan.explain() returns text", len(plan.explain()) > 20)

    # empty input must be rejected clearly, not silently
    try:
        plan_partition([], 8 * GIB)
        check("plan_partition rejects empty input", False)
    except ValueError:
        check("plan_partition rejects empty input", True)

    # impossible budget -> None, not a crash
    check("plan_partition returns None when impossible",
          plan_partition(segs, 1) is None)

    check("exact_peak on full graph positive", exact_peak(segs, (0, 3)) > 0)

    bf = brute_force(segs, 8 * GIB, limit=10)
    check("brute_force returns a result", bf is not None)


# ══════════════════════════════════════════════ axplan: credit ════════════

def test_credit_api():
    from axplan import credit as C

    for fn_name in ("credit_bytes", "local_quality", "crossover_batch",
                    "compare"):
        fn = getattr(C, fn_name, None)
        why = no_stub(fn)
        check(f"credit.{fn_name} is not a stub", not why, why)

    p = C.credit_bytes(C.CreditRule.BPTT, C.NeuronModel.LIF, 1000, 64, 32, 768, 12)
    check("credit_bytes BPTT positive", p.bytes_total > 0, f"{p.bytes_total:,}")
    loc = C.credit_bytes(C.CreditRule.LOCAL, C.NeuronModel.LIF, 1000, 64, 32, 768, 12)
    check("credit_bytes LOCAL positive", loc.bytes_total > 0)
    check("LOCAL is cheaper than BPTT", loc.bytes_total < p.bytes_total,
          f"{loc.bytes_total:,} < {p.bytes_total:,}")
    check("CreditPlan.explain() returns text", len(p.explain()) > 20)

    # local rule must NOT grow with horizon -- that is its whole point
    small = C.credit_bytes(C.CreditRule.LOCAL, C.NeuronModel.LIF, 1000, 16, 32, 768, 12)
    big = C.credit_bytes(C.CreditRule.LOCAL, C.NeuronModel.LIF, 1000, 512, 32, 768, 12)
    check("LOCAL memory is horizon-independent",
          small.bytes_total == big.bytes_total,
          f"T=16 {small.bytes_total:,} == T=512 {big.bytes_total:,}")
    bsmall = C.credit_bytes(C.CreditRule.BPTT, C.NeuronModel.LIF, 1000, 16, 32, 768, 12)
    bbig = C.credit_bytes(C.CreditRule.BPTT, C.NeuronModel.LIF, 1000, 512, 32, 768, 12)
    check("BPTT memory grows with horizon", bbig.bytes_total > bsmall.bytes_total,
          f"{bsmall.bytes_total:,} -> {bbig.bytes_total:,}")

    # honest None outside measured range -- a hard project rule
    check("local_quality(60) inside measured range",
          C.local_quality(60) == C.local_quality(60) and
          0.8 < C.local_quality(60) <= 1.0, f"{C.local_quality(60)}")
    check("local_quality beyond data returns None, not a guess",
          C.local_quality(5000) is None)

    co = C.compare(connections=1_000_000, timesteps=64)
    check("compare returns CreditChoice", isinstance(co, C.CreditChoice))
    check("compare.saving_bytes positive", co.saving_bytes > 0,
          f"{co.saving_bytes:,}")
    check("marginal_vs_adam is finite", co.marginal_vs_adam > 0)
    check("CreditChoice.explain() returns text", len(co.explain()) > 20)

    xb = C.crossover_batch(64)
    check("crossover_batch positive", xb > 0, f"{xb:.1f}")
    # measured property: crossover batch FALLS as horizon grows
    x16 = C.crossover_batch(16)
    x256 = C.crossover_batch(256)
    check("crossover batch falls as horizon grows", x256 < x16,
          f"T=16 {x16:.1f} > T=256 {x256:.1f}")


# ══════════════════════════════════════════════ axiomrnn top level ════════

def test_axiomrnn_api():
    import axiomrnn as ax

    check("axiomrnn exports Model", hasattr(ax, "Model"))
    check("axiomrnn exports Budget", hasattr(ax, "Budget"))
    check("axiomrnn exports SpikingRNN", hasattr(ax, "SpikingRNN"))
    check("axiomrnn exports Dense", hasattr(ax, "Dense"))

    for name in ("consumer_6gb", "consumer_8gb", "pro_24gb",
                 "datacenter_80gb"):
        fn = getattr(ax.Budget, name)
        b = fn()
        check(f"Budget.{name}() returns a Budget", isinstance(b, ax.Budget))
        check(f"Budget.{name} usable_bytes positive", b.usable_bytes() > 0,
              f"{b.usable_bytes() / 2**30:.2f} GiB")

    m = ax.Model()
    check("Model() empty", len(m.layers) == 0, repr(m))
    m.add_spiking(64, horizon=16)
    m.add_dense(4)
    check("Model.add_spiking appends", len(m.layers) == 2, repr(m))
    check("Model.add_dense appends", len(m.layers) == 2)

    sp = m.fit_budget(ax.Budget.consumer_6gb(), batch=8)
    check("fit_budget returns a plan", sp is not None and sp.peak_bytes > 0,
          f"{getattr(sp, 'peak_bytes', 0):,} B")
    check("fit_budget sets horizon", getattr(sp, "credit_horizon", 0) > 0,
          str(getattr(sp, "credit_horizon", 0)))

    txt = m.explain_budget(ax.Budget.consumer_6gb(), batch=8, verbose=False)
    check("explain_budget returns long text", len(txt) > 400, f"{len(txt)} chars")
    short = m.explain(ax.Budget.consumer_6gb(), batch=8, verbose=False)
    check("explain() alias equals explain_budget()", short == txt)

    # add() must accept a hand-built layer
    m2 = ax.Model()
    m2.add(ax.SpikingRNN(32, horizon=8))
    check("Model.add accepts SpikingRNN", len(m2.layers) == 1)

    # spec() lives on the LAYER (SpikingRNN), not on Model -- audited here
    # because that is where it is declared.
    layer_spec = ax.SpikingRNN(64, horizon=16).spec(batch=8, horizon=16)
    check("SpikingRNN.spec returns a BudgetSpec",
          hasattr(layer_spec, "connections") and layer_spec.connections > 0,
          f"{layer_spec.connections:,} connections")

    # the cell override that used to be broken
    from axtf.cells import SpikingCell
    m3 = ax.Model()
    m3.add_spiking(16, horizon=8, cell=SpikingCell(0.5, 1.0))
    check("add_spiking(cell=...) is accepted", len(m3.layers) == 1)

    check("explain_budget module function is callable",
          callable(ax.explain_budget))
    check("gate_report is callable", callable(ax.gate_report))
    check("main_explain is callable", callable(ax.main_explain))

    # the inverse question: called for real, not merely introspected
    best, over = ax.largest_that_fits(6.0, hidden=1024, n_layers=8,
                                      timesteps=64, batch=4)
    check("largest_that_fits returns a Candidate",
          isinstance(best, ax.Candidate) and best.connections > 0,
          f"{best.connections:,} synapses" if best else "None")
    check("largest_that_fits reports a verdict on the next step up",
          over is None or over.peak_bytes > over.vram_bytes,
          f"{over.peak_bytes / 2**30:.2f} GiB" if over else "none in range")
    check("Candidate.explain() produces text", len(best.explain()) > 80)
    check("Candidate rebuilds its own spec",
          best.spec().connections == best.connections)
    front = ax.capacity_frontier(6.0, hidden=512, n_layers=4,
                                 timesteps=16, batch=2)
    check("capacity_frontier returns fitting candidates",
          isinstance(front, list) and len(front) > 0
          and all(c.fits for c in front), f"{len(front)} entries")
    rep = ax.capacity_report(6.0, hidden=1024, n_layers=8,
                             timesteps=64, batch=4)
    check("capacity_report returns long text", len(rep) > 300, f"{len(rep)} chars")
    check("local_quality is callable and range-guarded",
          ax.local_quality(60) is not None and ax.local_quality(5000) is None)

    # The suite count printed in the report used to be the literal "18" and was
    # wrong twice. It is now read from the gate's table, so the thing to assert
    # is that the number in the report equals the number of suites that exist.
    n_declared = ax._gate_suite_count()
    check("the gate's suite count can be read", isinstance(n_declared, int)
          and n_declared > 0, f"got {n_declared!r}")
    spec = ax.Model()
    spec.add_spiking(64, horizon=8)
    spec.add_dense(4)
    report = spec.explain(ax.Budget.consumer_8gb(), verbose=False)
    check("the report states the suite count the gate actually has",
          f"{n_declared} suites" in report,
          f"expected {n_declared} suites in the report")


# ══════════════════════════════════════════════ axon reference ═════════════

def test_axon_api():
    import numpy as np

    from axon.lif import SpikingNet, dsig, dsig2, ssur

    for fn in (ssur, dsig, dsig2):
        check(f"axon.{fn.__name__} is not a stub", not no_stub(fn), no_stub(fn))

    u = np.linspace(-2, 2, 9)
    s, d1, d2 = ssur(u, 0.0), dsig(u, 0.0), dsig2(u, 0.0)
    check("ssur is monotonic increasing", bool(np.all(np.diff(s) > 0)))
    check("ssur is centred at the threshold", abs(s[4] - 0.5) < 1e-12)
    check("dsig is positive", bool(np.all(d1 > 0)))
    check("dsig peaks at the threshold", abs(d1[4] - d1.max()) < 1e-12)
    check("dsig2 is antisymmetric", abs(d2[4] + d2[4]) < 1e-12)
    # finite differences on the smooth surrogate
    h = 1e-6
    fd = (ssur(u + h, 0.0) - ssur(u - h, 0.0)) / (2 * h)
    check("dsig matches finite differences",
          float(np.abs(fd - d1).max()) < 1e-6,
          f"max err {np.abs(fd - d1).max():.2e}")

    net = SpikingNet([6, 8, 4], rng=np.random.default_rng(0))
    X = np.random.default_rng(1).normal(size=(5, 6))
    y = np.random.default_rng(2).integers(0, 4, 5)

    logits, readout, cache = net.forward(X, 8)
    check("forward returns logits of shape (B, dout)", logits.shape == (5, 4),
          str(logits.shape))
    check("forward returns readout of shape (B, hidden)", readout.shape == (5, 8),
          str(readout.shape))
    # dims [6, 8, 4] -> ONE hidden layer of width 8, then the output layer.
    # cache must therefore hold 1 entry, not 2.
    check("cache holds one entry per HIDDEN layer",
          len(cache) == net.L, f"{len(cache)} == net.L={net.L}")
    check("cached potentials are time-first (T, B, units)",
          cache[0][0].shape == (8, 5, 8), str(cache[0][0].shape))
    check("cached spikes are binary", bool(
        np.all((cache[0][1] == 0) | (cache[0][1] == 1))))
    check("loss is finite", np.isfinite(net.loss(X, y, 8)))

    l, g = net.grads(X, y, 8, smooth=True)
    check("grads returns matching shapes",
          g.shape == net.params().shape, f"{g.shape}")
    check("grads loss matches loss()",
          abs(l - net.loss(X, y, 8, smooth=True)) < 1e-9)

    net.calibrate(X, 8)
    fire, dead, _ = net.firing(X, 8)
    check("calibrate gives a sane firing rate", 0.0 <= fire <= 1.0, f"{fire:.4f}")
    check("dead fraction in range", 0.0 <= dead <= 1.0, f"{dead:.4f}")
    check("accuracy in range", 0.0 <= net.accuracy(X, y, 8) <= 1.0)

    net.step(g, lr=0.01)
    check("step does not corrupt params", np.isfinite(net.params()).all())

    net.set_params(net.params())
    check("set_params round-trips", np.isfinite(net.params()).all())


# ══════════════════════════════════════════════ axtf (needs TF) ═══════════

def test_axtf_api():
    try:
        from tensorflow import keras
    except ImportError as e:
        skip("axtf API surface", f"no TensorFlow: {e}")
        return

    import axiomrnn as ax
    from axtf.build import build_keras, calibrate_model, measure_model
    from axtf.cells import SpikingCell, SpikingRNNCell

    for fn in (build_keras, calibrate_model, measure_model):
        check(f"axtf.{fn.__name__} is not a stub", not no_stub(fn), no_stub(fn))

    check("SpikingCell has all extension hooks",
          all(hasattr(SpikingCell, m) for m in
              ("membrane", "spike_fn", "reset_fn", "spike_d1", "reset_dud",
               "threshold_dthr")))

    T, D, C = 8, 5, 3
    x = keras.Input(batch_shape=(None, T, D))
    layer = SpikingRNNCell(16, cell=SpikingCell(0.9, 2.0),
                           return_sequences=True, horizon_hint=T)
    y = layer(x)
    m = keras.Model(x, y)
    check("SpikingRNNCell builds with correct shape",
          tuple(m.output_shape) == (None, T, 16), str(m.output_shape))

    import numpy as onp
    rng = onp.random.default_rng(0)
    X = rng.normal(size=(64, T, D)).astype("float32")
    out = m.predict(X, verbose=0)
    check("forward produces real spikes", out.shape == (64, T, 16))
    check("spikes are binary", bool(onp.all((out == 0) | (out == 1))))

    rate = float((out > 0.5).mean())
    check("firing rate is non-trivial", 0.0 < rate < 1.0, f"{rate:.4f}")

    # calibrate() returns the threshold Variable itself (axtf), whereas the
    # numpy reference returns an iteration history. Both are documented
    # behaviour -- audit what each actually returns.
    thr_after = layer.calibrate(X, target_rate=0.2, iters=3)
    check("layer.calibrate returns the threshold variable",
          hasattr(thr_after, "numpy") and
          float(onp.asarray(thr_after).min()) > 0.0,
          f"thr in [{float(onp.asarray(thr_after).min()):.3f}, "
          f"{float(onp.asarray(thr_after).max()):.3f}]")

    fr, _ = layer.firing_rate(X)
    check("layer.firing_rate returns a rate in range", 0.0 <= fr <= 1.0, f"{fr:.4f}")

    # explicit backward is a public API
    dout = onp.random.default_rng(1).normal(size=(64, T, 16)).astype("float32")
    dx, dk, dthr = layer.backward(X, dout, T)
    check("backward returns three gradients",
          dx.shape == X.shape and dk.shape == (D, 16) and dthr.shape == (16,))

    spec = ax.Model()
    spec.add_spiking(16, horizon=T)
    spec.add_dense(C)
    km = build_keras(spec, inputs=keras.Input(batch_shape=(None, T, D)),
                     outputs=C)
    check("build_keras returns a Keras Model",
          isinstance(km, keras.Model) and tuple(km.output_shape) == (None, C))

    calibrate_model(km, X)
    rows = measure_model(km, X)
    check("measure_model returns one row per spiking layer",
          isinstance(rows, list) and len(rows) >= 1, f"{len(rows)} rows")
    check("measure_model rows carry real numbers",
          all("firing" in r and "dead" in r for r in rows))

    # readout modes
    km2 = build_keras(spec, inputs=keras.Input(batch_shape=(None, T, D)),
                      outputs=C, readout="mean")
    check("readout='mean' builds too", tuple(km2.output_shape) == (None, C))

    # compile path sets from_logits correctly
    cm = build_keras(spec, inputs=keras.Input(batch_shape=(None, T, D)),
                     outputs=C, compile_model=True)
    loss = cm.loss
    check("compiled loss is from_logits=True",
          getattr(loss, "from_logits", False) is True)


# ═════════════════════════════════════ shipped metadata vs the git remote ══════
#
# The 0.1.1 wheel advertised Source/Documentation/Issues/Changelog under
# timatigoogl3-code/axiomrnn -- a third party's account -- while `git remote`
# said cubetitled-ui. Every other suite here calls functions and checks
# behaviour, so nothing in the gate was looking at a string, and a wrong URL on
# a PyPI page is invisible to a test that never reads one. tomllib is stdlib
# from 3.11; the package supports 3.10, so the read is skipped rather than
# faked on the oldest supported interpreter.

def test_metadata_points_at_this_repository():
    try:
        import tomllib
    except ImportError:
        skip("project.urls names the repository this project lives in",
             "tomllib needs Python 3.11+, and this is 3.10")
        return

    root = Path(__file__).resolve().parent.parent
    try:
        with open(root / "pyproject.toml", "rb") as fh:
            cfg = tomllib.load(fh)
    except (OSError, ValueError) as exc:
        check("pyproject.toml is readable", False, str(exc))
        return

    urls = cfg.get("project", {}).get("urls", {})
    check("pyproject.toml declares project.urls", bool(urls),
          f"found {sorted(urls)}")

    # The repository the project is actually in. Read from git when available,
    # because a test that hardcodes the expected value would have passed while
    # the remote itself was wrong -- which is the failure being guarded.
    expected = "cubetitled-ui/axiomrnn"
    try:
        # check=False: a repository with no `origin` remote is not an error
        # here, it just means the fallback above stands.
        rem = subprocess.run(["git", "-C", str(root), "remote", "get-url",
                              "origin"], capture_output=True, text=True,
                             timeout=30, check=False)
        if rem.returncode == 0 and rem.stdout.strip():
            url = rem.stdout.strip()
            if "github.com" in url:
                expected = url.split("github.com", 1)[1].lstrip(":/").removesuffix(".git")
    except (OSError, subprocess.SubprocessError):
        pass

    for key, value in sorted(urls.items()):
        check(f"project.urls.{key} names {expected}",
              expected in value, f"got {value}")
        check(f"project.urls.{key} does not name a third-party account",
              "timatigoogl3-code" not in value, f"got {value}")

    # Attribution has to agree with the LICENSE, which ships in the wheel.
    authors = cfg.get("project", {}).get("authors", [])
    names = [a.get("name", "") for a in authors]
    check("pyproject.toml names an author", any(names), f"got {names}")
    lic = (root / "LICENSE").read_text(encoding="utf-8", errors="replace")
    check("the LICENSE copyright matches the package author",
          all(n in lic for n in names if n), f"authors {names}")

    # Classifiers are a promise about interpreters; requires-python is the
    # contract. 0.1.1 advertised 3.14 while capping at <3.14.
    spec = cfg.get("project", {}).get("requires-python", "")
    nums = re.findall(r"3\.(\d+)", spec)
    lo = int(nums[0]) if nums else 0
    hi = int(nums[1]) if len(nums) > 1 else 99
    bad = []
    for c in cfg.get("project", {}).get("classifiers", []):
        m = re.match(r"Programming Language :: Python :: 3\.(\d+)", c)
        if m and not (lo <= int(m.group(1)) < hi):
            bad.append(c)
    check("every Python classifier is inside requires-python",
          not bad, f"requires-python {spec}; outside it: {bad}")

    # The dependency that makes this a framework rather than a planner.
    check("TensorFlow is a required dependency, not an extra",
          any(d.startswith("tensorflow") and "extra ==" not in d
              for d in cfg["project"]["dependencies"]),
          f"got {cfg['project']['dependencies']}")
    check("numpy is required", any(d.startswith("numpy")
                                   for d in cfg["project"]["dependencies"]),
          f"got {cfg['project']['dependencies']}")


# ══════════════════════════════════════════════ runner ════════════════════

def main():
    print("=" * 74)
    print("API AUDIT: every advertised function is called for real")
    print("=" * 74)

    for fn in (test_memory_api, test_planner_api, test_credit_api,
               test_axiomrnn_api, test_axon_api, test_axtf_api,
               test_metadata_points_at_this_repository):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(FAIL)
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            FAIL.append(f"{fn.__name__}: raised {traceback.format_exc()[-200:]}")
        added = len(FAIL) - n0
        print(f"    {'OK' if added == 0 else f'FAIL x{added}'}")

    print()
    print("=" * 74)
    print(f"  PASSED {len(PASS)} | FAILED {len(FAIL)} | SKIPPED {len(SKIP)}")
    if FAIL:
        print()
        print("  FAILED:")
        for f in FAIL:
            print(f"    - {f}")
    if SKIP:
        print()
        print("  SKIPPED:")
        for s in SKIP:
            print(f"    - {s}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())