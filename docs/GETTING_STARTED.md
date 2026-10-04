# Getting started

Five minutes to a budgeted model and a firing neuron, with no TensorFlow
background and no framework background assumed.

Every figure here was produced on **2026-10-04** by the command shown. See
[`../README.md`](../README.md) for what this project does **not** claim — read
that first.

---

## Install

```bash
# `pip install axiomrnn` works and gets you 0.1.1 -- which carries two
# planner defects that return wrong numbers silently. Read the Install
# section of README.md before trusting a credit-rule figure from it.
git clone <this repository>
cd axiomrnn
```

Nothing to install for the planning path. It runs on a bare `python3` with
NumPy.

If you want to train, you need TensorFlow, and **TensorFlow has no build for
Python 3.14** — so you need a Python 3.12-or-older environment. `MEASURED`:
`/usr/bin/python3 -V` on the reference machine is `Python 3.14.4`; the
TensorFlow environment there is `Python 3.12.13` with TensorFlow `2.21.0`.
TensorFlow is a large install: `1.9 GB` for the `tensorflow` package alone.

Check what you have:

```bash
python3 -c "import axiomrnn as ax; print('TF:', ax._HAS_TF, '| err:', ax._TF_ERR)"
```

`TF: False` is fine and expected. `ax.SpikingCell` is then `None` rather than a
crash — that is the documented soft-fail.

**A locally built wheel exists but is not published.**
`dist/axiomrnn-0.1.0-py3-none-any.whl` (58,298 bytes) installs into a clean
TensorFlow-free environment and its console script `axiomrnn-explain` runs;
both were verified on 2026-10-03/04. Until it is on an index this file does not
tell you to install it, and `pip install axiomrnn` will fail or fetch somebody
else's project.

---

## 1. Your first budget, with no TensorFlow

```python
import axiomrnn as ax

spec = ax.Model()
spec.add_spiking(1024, horizon=256)
spec.add_spiking(1024, horizon=256)
spec.add_dense(1000)

print(spec.explain(ax.Budget.consumer_6gb()))
```

You get, among other things:

```
  WHAT IT COSTS IN MEMORY
    regime:        activation
    tape share:    99.6% <-- tape dominates
    peak:          1.57 GiB of 6 GiB

  THE MAIN MEMORY LEVER: THE CREDIT RULE
    now: BPTT, 0.063 GiB
    e-prop LIF: 0.008 GiB (0.50x Adam -- cheaper than momentum)
    batch crossover: batch* = 4
    -> at batch=32 the local rule is ALREADY cheaper.
```

`MEASURED` — `python3 examples/02_budget.py`, exit 0 in 0.04 s on Python
3.14.4 with no TensorFlow installed. That is arithmetic. Nothing imported
TensorFlow to produce it.

## 1b. The other direction: what should you build?

Section 1 answers "what does this cost?". The inverse question is usually the
one you have:

```python
import axiomrnn as ax

best, too_big = ax.largest_that_fits(6.0)     # vram_gb in GiB
print(best.explain())                        # largest that fits
print(too_big.explain())                     # the boundary, if one exists
```

Three fields earn their keep:

- `binding` names which term dominates — `static`, `tape`, `backward`,
  `logits`, `workspace`. If it says `static`, the **weights** dominate, so
  the credit rule and the spike coding save you nothing on this budget and
  width is the only lever. That single word saves a wasted afternoon.
- The second object is the smallest configuration that does **not** fit, so
  you can see the overshoot before you spend the day discovering it.
- `bytes_per_param` is printed next to the synapse count, because
  "2.1 billion synapses" means nothing without it.

Both may be `None`, and that is not an error: `None` for `best` means nothing
in the searched space fits; `None` for `too_big` means the answer sits on the
search ceiling and there is no boundary above it inside the space. Neither
case is papered over.

It is a **capacity** answer. This package has no verified quality model, so
it does not rank architectures by accuracy — `ax.local_quality()` is the one
quality figure it carries, it is read from a cited measurement, and it
returns `None` outside its measured range instead of extrapolating.

## 2. Your first network

```python
import numpy as np, tensorflow as tf, keras
import axiomrnn as ax

T, C = 16, 4
rng = np.random.default_rng(0)
pat = rng.normal(0, 0.4, (C, T, 12))
for k in range(C):
    pat[k, k * (T // C):(k + 1) * (T // C)] += 2.0
y = rng.integers(0, C, 4000)
X = (pat[y] * (rng.random((4000, T, 12)) < 0.3)
     + rng.normal(0, 0.6, (4000, T, 12))).astype("float32")

spec = ax.Model()
spec.add_spiking(192, horizon=T)
spec.add_spiking(192, horizon=T)
spec.add_dense(C)

model = spec.keras_model(
    inputs=keras.Input(batch_shape=(None, T, 12)), outputs=C,
    compile_model=True)
model.fit(X, y, epochs=12, batch_size=64)
```

`MEASURED` — the equivalent shipped script, `examples/01_minimal.py`, reaches
held-out accuracy **0.9983** against a 0.25 baseline in 75 s on CPU
(`0.9983` standalone, `0.9975` as a gate suite on a busier machine — the
difference is run-to-run variance on a 1,200-example held-out set, which is why
a single run should never be quoted as a constant).

### Three Keras 3 things that will bite you

**Use `batch_shape`, not `shape`.** `Input(shape=(T, D))` leaves the time axis
`None`, and our layer cannot derive its horizon from it. The historical error
was `ValueError: the layer needs a horizon and shape=(16, 12) leaves the time
axis None`; the fix is `batch_shape=(None, T, D)`.

**`from_logits=True` is mandatory.** Our output *is* logits. Keras defaults to
expecting probabilities and applies softmax itself. `MEASURED` — with
`from_logits=False` the loss after one epoch is **9.8438**; the correct
configuration is around 1.44 and never stalls above `ln(4) = 1.386`.

**`run_eagerly=True` is not optional.** Our forward pass is a Python loop over
time, and Keras 3 caches the result instead of recomputing it. The gradient
then applies to stale activations. `MEASURED` — with a call counter: the
forward ran 2 times where 4 were needed, and 0 times on a repeat fit.
`compile_model=True` sets `run_eagerly=True` for you. The proper fix is a
`tf.while_loop`; it is not done, and it costs speed on purpose.

## 3. Calibrate, or it will not train

A fresh spiking network is **silent**: potentials stay below the default
threshold of 1.0, nothing spikes, no gradient flows.

```python
from axtf.build import calibrate_model
calibrate_model(model, X[:512], target_rate=0.2)
```

You can see the problem before you fix it:

```python
from axtf.build import measure_model
measure_model(model, X[:512])
# [{'layer': 'spiking_0', 'firing': 0.0, 'dead': 1.0}, ...]
```

`dead: 1.0` means every neuron is silent. After calibration the firing rate
lands near your target.

## 4. Swap in your own neuron

```python
from axtf.cells import SpikingCell, _atan_surrogate

class SmoothCell(SpikingCell):
    def spike_fn(self, u, thr, smooth=True):
        return _atan_surrogate(u, thr, self.alpha)
    def reset_fn(self, u, spike, smooth=True):
        return u * (1.0 - spike)

spec.add_spiking(192, horizon=T, cell=SmoothCell())
```

Different layers of one network may hold different neurons. The backward
pass, calibration and training loop stay shared, which is the point: they
stay *verified*.

## 5. Check the work before you believe it

```bash
python3 tests/run_all.py
```

`MEASURED` — 2026-10-04 21:38 MSK, exit 0, **11 min 24 s** of suite time,
**31 suites**, all of them run, **385 declared checks** across the nine suites
that report a count.

If the gate cannot find an interpreter with TensorFlow — the normal case, since
it is usually a project venv rather than `./.venv` — it reports every
TensorFlow suite as SKIPPED and exits 2. That is the honest answer. Point it at
the right one:

```bash
AX_TF_PY=/path/to/python-with-tensorflow python3 tests/run_all.py
```

The gate compares the analytic gradient against finite differences on the
*smoothed* model — the live run reports `median rel.err 1.02e-08, worst
3.41e-07` over 74 points — and separately checks that the gradient does not
grow with the horizon (`T=16/T=4 = 1.18` with normalisation, `5.40` without).
It is not ceremony: it is what caught a `norm_t` parameter that had been
threaded through four signatures and never applied, and a backward pass that
read `cell.leak` as a membrane derivative.

Two limits of the gate, so you read it correctly. Twenty-two of the thirty-one
suites prove only that the process exited 0, because the gate filters their
output down to lines matching `PASS`/`ok]`/`Error`/`Traceback`/`SKIPPED`; the
other nine report a count, and 14 suites now declare evidence markers, so a run
that exits 0 without printing what it claims to show is a FAIL. And the gate
covers **every** example now — `examples/04_equal_budget.py` used to be the
exception, which is why a comment claiming otherwise could sit in the source
unnoticed.

## Where to go next

| you want | read |
|---|---|
| plan for real hardware | `examples/02_budget.py`, `examples/advanced/04_budget_and_precision.py` |
| understand the memory lever | `examples/advanced/03_credit_rule.py` |
| a working language model | `examples/05_t9.py` |
| not fool yourself when comparing | `examples/04_equal_budget.py` |
| why this codebase looks like this | `docs/ARCHITECTURE.md`, `docs/UX_RULES.md` |