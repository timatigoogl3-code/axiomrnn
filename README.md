# axiomrnn

**Design for the memory budget you have, not the memory you wish you had.**

A recurrent spiking network you extend without forking, plus a memory planner
that tells you what your model will cost before you train it.

```python
import axiomrnn as ax

spec = ax.Model()
spec.add_spiking(192, horizon=24)        # 192 neurons, 24 promised steps
spec.add_spiking(192, horizon=24)
spec.add_dense(4)

print(ax.explain_budget(spec, ax.Budget.consumer_8gb(), verbose=False))
```

That prints a real memory plan and needs no TensorFlow. The spiking layers need
TensorFlow, which is a required dependency.

---

## What you get

| | |
|---|---|
| **Train a spiking network** | recurrent layers whose state is carried by binary spikes, trained through ordinary `keras.fit` |
| **Replace the neuron without forking** | override `membrane` and `membrane_dstate` together; the gradient hooks are checked against finite differences |
| **Know the memory cost first** | ask the planner what a model costs at a given budget before you spend the GPU time |
| **Invert the question** | ask which model a budget can actually train, and what stops the next step up |
| **Generate text** | the recurrent state feeds back one token at a time; see `examples/07_generate.py` |

## What it is not

It is not faster than a dense network on an ordinary GPU, and it does not save
energy, because no measurement here involves neuromorphic hardware. The spiking
choice costs about nothing in quality and earns its keep only where spikes are
cheap.

It also does not beat transformers in general. That is a measured finding, not
a guess. The reasoning is long and unflattering enough to deserve its own file:
**[docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md)**.

Read it before you cite this work.

---

## Install

```bash
pip install axiomrnn
```

`MEASURED` 2026-10-04 — `https://pypi.org/pypi/axiomrnn/json` returns 200 with
releases `0.1.0` (uploaded `11:27:41Z`) and `0.1.1` (`11:54:53Z`). So the name
is taken by this project and `pip install axiomrnn` installs the code below.

**Known-bad install, and this is the important part of the section.** The
published **0.1.1 contains two defects in the memory planner that were found
and fixed after it was uploaded**, and neither raises an error — both return
plausible numbers:

- `credit_rule="local"` was answered with BPTT arithmetic, because the string
  `"local"` reached an `is CreditRule.LOCAL` dispatch and never matched it. The
  planner reported the local rule's memory as if it grew with the horizon.
- Peak memory was capped at the credit horizon, so `analyse()` returned the same
  peak for `T=100` and `T=400` — two thirds of the horizon charged at no cost.

Both are fixed in `master` and pinned by regression checks in
`tests/test_memory.py`. Until a new version is uploaded, **`pip install
axiomrnn` gives you a planner whose local-credit-rule figures are wrong.**
Install from the repository, or wait for the next release. This is stated here
rather than in a changelog because a stranger reads this page and not that.

Built artifacts, verified locally. These carry the fixes; the published 0.1.1
does not:

```
dist/axiomrnn-0.1.1-py3-none-any.whl    73,910 bytes, 18 files
dist/axiomrnn-0.1.1.tar.gz             472,158 bytes
```

`MEASURED` — `python -m build` then `ls -l dist/`, in the repository root. The
wheel ships `axiomrnn.py`, `axon/`, `axplan/` and `axtf/`, declares
`Requires-Python >=3.10,<3.14`, `numpy>=1.24` and **`tensorflow>=2.16`**, offers
extras `tf-gpu` and `dev`, and installs one console script, `axiomrnn-explain`.
`twine check dist/*` → `PASSED` for both files. Note the version number is
unchanged from the published one: **the version was not raised**, so these files
are not interchangeable with what is on PyPI. Whoever cuts the release decides
the number; this repository does not get to bump past a rule.

TensorFlow is a required dependency, not an extra. It was one once, so that the
memory planner stayed usable without a deep learning framework, and that is
still true — `import axiomrnn` works with numpy alone. But the extra made the
default install a package that could not build a spiking layer, which is the one
thing anyone installing this is here for.

The wheel does **not** ship `axcore/`, `tests/` or `examples/`. `axcore` holds
one C++ file with no `__init__.py`; the gate compiles and runs it, and it travels
in the sdist instead.

### The wheel install, verified

```bash
python3.12 -m venv /tmp/v && /tmp/v/bin/python -m pip install dist/axiomrnn-0.1.1-py3-none-any.whl
/tmp/v/bin/axiomrnn-explain --budget 6gb --layers 2 --width 1024 --horizon 256
```

`MEASURED` 2026-10-04, installed from the wheel above into a fresh Python 3.12
virtualenv and used from `/tmp`, with no source tree on `sys.path`:

```
axplan    : /tmp/v/lib/python3.12/site-packages/axplan/__init__.py
axiomrnn  : /tmp/v/lib/python3.12/site-packages/axiomrnn.py
source tree on sys.path: False
find_spec(tensorflow)   = None
ax._HAS_TF              = False
```

Then, on the same installed copy, the two defect checks that 0.1.1 as published
fails and this build passes:

```
credit_bytes("local").depends_on_T = False      # 0.1.1 prints True
credit_bytes("local").quality_kept = 0.389      # 0.1.1 prints 1.0
BPTT tape T=100 : 446,668,800 B                 # 0.1.1 prints 294,801,408
BPTT tape T=400 : 1,786,675,200 B  (exactly 4x) # 0.1.1 prints the same number twice
```

An earlier transcript of this section, taken against 0.1.0 on a Python 3.14
interpreter, read:

```
Package  Version
-------- -------
axiomrnn 0.1.0
numpy    2.5.3
pip      25.0.1

find_spec(tensorflow) = None
find_spec(keras)      = None
ax._HAS_TF            = False
ax._TF_ERR            = No module named 'tensorflow'
ax.SpikingCell        = None
site-packages path    = .../site-packages/axiomrnn.py
any sys.path entry in /home/cune/axiomrnn: False
Budget                = Budget(consumer 6 GB: 6.0 GB VRAM, host 32.0 GB, offload=True)
```

Three things follow, and they matter:

1. **The planning path works with no TensorFlow installed at all.** Exactly
   three packages land: `axiomrnn`, `numpy`, `pip`.
2. **The soft-fail is a soft-fail.** `ax.SpikingCell` is `None` and
   `ax._TF_ERR` explains why, instead of an `ImportError` at package import.
3. **`axplan` and `axon` are TensorFlow-free; `axtf` is not.** `MEASURED` —
   module by module in that venv: `axplan`, `axplan.memory`, `axplan.planner`,
   `axplan.solve`, `axplan.credit`, `axon`, `axon.lif`, `axon.gate` and the
   `axtf` *namespace* all import; `axtf.cells` and `axtf.build` both raise
   `ModuleNotFoundError`. So do not read "the wheel works without TensorFlow"
   as "`axtf` works without TensorFlow". It does not.

`MEASURED` — the console script exits 0 from the installed venv:

```
tape share:    99.6% <-- tape dominates
peak:          1.57 GiB of 6 GiB
HEADROOM:      2.54 GiB unused.
```

Those figures differ from the `0.78 GiB` quoted above for 0.1.0, and the reason
is the fix rather than a regression: BPTT's tape was previously capped at the
credit horizon and two thirds of it was charged at no cost. The script also
reads its suite count out of `tests/run_all.py` rather than carrying a literal,
so it says `31 suites` today and will say whatever is true after the next edit.

### What works today from a checkout

```bash
cd axiomrnn
python3 examples/02_budget.py          # works, no TensorFlow
```

`MEASURED` — `env -u PYTHONPATH /usr/bin/python3 -c "import axiomrnn as ax;
print(ax._HAS_TF)"` in the repository root prints `import ok, _HAS_TF = False`.

---

## The five-minute path

### What TensorFlow is, and why it is unavoidable here

TensorFlow is Google's numerical library; **Keras** is its model-building
layer, and you meet Keras whether or not you asked for it. A Keras model is
not a Python object full of layers you call yourself — it is a *graph*: you
declare the shape of the data going in, list the layers, and Keras wires them
into one callable that moves the data through. This project needs it because
a spiking layer's memory cost is a function of the graph the layer sits in, and
because the layer has to be differentiable by a framework that owns the
backward pass. There is no way to get a *trained* neuron here without it.
**Planning, however, needs nothing but NumPy** — see the third step below,
which is the part that runs in under a second.

### Step 1 — get the code, no install (0 min)

```bash
git clone <this repository>
cd axiomrnn
```

Nothing to install. The planning path runs on a bare `python3` with NumPy.
`MEASURED` — the system interpreter here is Python **3.14.4** with NumPy
**2.5.0** and **no TensorFlow**, and `python3 examples/02_budget.py` exits 0
on it.

If you would rather install the built wheel than use the checkout, the
[Install section](#install) has the verified transcript. Either way, step 2
works with no TensorFlow anywhere.

### Step 2 — the first budget, and it needs no TensorFlow (about 1 second)

```bash
python3 examples/02_budget.py
```

`MEASURED` — exit code 0 in **0.11 s**, **15 MB** peak RSS
(`/usr/bin/time -f "%e s %M KB peakRSS" python3 examples/02_budget.py`). Its
first line of output is the claim being tested:

```
  TensorFlow in this environment: ABSENT, and not needed
```

and the core of it is this table:

```
  budget                    usable      needed     fits?
  consumer 6 GB           4.12 GiB    1.57 GiB       yes
  pro 24 GB              17.96 GiB    1.57 GiB       yes
  datacenter 80 GB       61.04 GiB    1.57 GiB       yes
```

The same architecture at width `20480` needs `23.35 GiB` and the planner says
`NO` on a 6 GB card rather than promising and failing later. This is the
product: **the budget is an input, and the tool tells you what it buys.**

**Why `1.57` and not the `0.78` an earlier version of this file printed.** The
old number was produced by a planner that capped the BPTT tape at the credit
horizon, so most of the horizon was charged at no cost — the defect described in
[Install](#install). The larger figure is the honest one for this architecture
at `horizon=256`, and it is why the sweep below crosses 4 GiB at width 4096
rather than somewhere past 12000.

### Step 3 — the inverse question (about 1 second)

```bash
python3 examples/advanced/15_what_to_build.py
```

Still no TensorFlow. Every other tool answers *"what does this network
cost?"*. This one answers the question you actually have. `MEASURED` — exit
code 0; at a 6 GiB budget:

```
  LARGEST THAT FITS
  hidden 8192, layers 32, horizon 256, batch 1
  connections 2,147,745,792   bytes per synapse 2.00
  peak 5.604 GiB of 6.000 GiB   headroom 0.396 GiB   binding term static

  FIRST THAT DOES NOT FIT
  hidden 32768, layers 4, horizon 1
  peak 8.509 GiB of 6.000 GiB   overshoot 1.42x
```

Three fields earn their keep:

- **the binding term** — `static` means the *weights* dominate, so the credit
  rule and the spike coding save you nothing on this budget and width is the
  only lever. That one word saves a wasted afternoon.
- **the failure boundary** — the smallest step up that does not fit, with the
  overshoot. A tool that only reports successes cannot be told apart from a
  tool that is simply optimistic.
- **refusal** — `None` when nothing in the searched space fits, and no
  invented boundary when the answer sits on the search ceiling. The same run
  prints both refusals on demand: with a budget too small for anything it
  prints `solution: None`, and at the width ceiling it prints `reported
  overflow: None`.

One caveat it states in its own output: this is a **capacity** answer, not a
quality one. This package has no verified quality model. The one quality
figure it carries — what the local credit rule costs — is printed next to the
capacity gain, because a capacity number without its price is a sales pitch.
`MEASURED` — same run: capacity ratio under the local rule is `1.00x`, and
`local rule at T=256 keeps 64.7% of BPTT`.

### Step 4 — a neuron that actually fires (75 s of compute, plus a 1.9 GB download)

**Read this before step 4.** TensorFlow has no build for Python 3.14. The
system interpreter on this machine is 3.14.4, so `pip install tensorflow`
into it will not work; you need an environment on Python 3.12 or older.
`MEASURED` — `/usr/bin/python3 -V` is `Python 3.14.4`; the TensorFlow
environment here is `/home/cune/.venvs/ax/bin/python`, `Python 3.12.13`.
(`tests/run_all.py:25` carries the same note in a comment.)

TensorFlow is a large dependency: `1.9 GB` for the `tensorflow` package,
`21 MB` for `keras`, `6.3 GB` for the whole environment including the
interpreter. `MEASURED` — `du -sh` on each. **That download, not the code, is
what decides whether this step fits in five minutes.**

```bash
# with a Python 3.12-or-older environment that has TensorFlow 2.x:
python examples/01_minimal.py
```

`MEASURED` — TensorFlow `2.21.0`, Keras `3.15.1`, CPU only
(`CUDA_VISIBLE_DEVICES=`), exit code 0 in **75.2 s**, **641 MB** peak RSS,
running standalone. Held-out accuracy **0.9983** against a **0.2500** chance
baseline, after 15 epochs at batch 64; final training loss `3.7285e-04`, final
validation loss `0.0071`; 51,844 total parameters.

`MEASURED` — the same script as a suite of the gate took **86.1 s** in the
2026-10-04 21:38 run below, against the 75.2 s standalone figure above. The
wall clock differs because the gate run had other work on the machine; held-out
accuracy varies run to run on a 1,200-example set. Both numbers are real, and
the difference is why this file quotes a range rather than a constant.

**One warning about this file, `MEASURED`:** it used to run the entire test
gate as a subprocess at the end (`examples/01_minimal.py:165-175`), so running
it trained the network, then ran the whole gate, which trained it a second time.
Observed with `pgrep` during this documentation pass: `01_minimal.py` spawned
`run_all.py`, whose `EXAMPLE 01` suite spawned `01_minimal.py` *again*.
**Fixed at 23:58:59 on 2026-10-03**, during this pass, by the code owner — the
subprocess is now behind `AX_RUN_GATE=1` and the script otherwise prints the
gate command for you to run yourself. This is the one number in this file that
was correct when written and became wrong an hour later.

### Step 5 — check the work before you believe it

```bash
python3 tests/run_all.py
```

`MEASURED` — **11 min 24 s** of suite time on the 2026-10-04 run below, not
five minutes. It is deliberately outside the five-minute path: the five minutes
are for getting a neuron to fire, and the gate is for deciding whether to
believe anything. Most of the wall clock is two suites that train real networks
(`EXAMPLE 05` alone is 289 s). See *Verify it yourself* below for the real
output, its date, and its per-suite inventory.

---

### The five minutes, honestly totalled

`MEASURED`:

| step | wall clock | blocking download |
|---|---:|---|
| clone | seconds | the repository |
| step 2 — budget as an input | **0.04 s** | none |
| step 3 — the inverse question | **0.06 s** | none |
| step 4 — a neuron that fires | **75.2 s** | **1.9 GB** of TensorFlow, on a Python ≤ 3.12 environment |

**What you can genuinely do in under a minute, with no install at all:** run
`python3 examples/02_budget.py` and `python3
examples/advanced/15_what_to_build.py` on a bare interpreter with NumPy, and
get the entire planning surface — hardware profiles, the codec decision, the
precision arithmetic, the segmentation answer, the largest network a budget
buys, and the boundary that stops it.

**What you cannot do in five minutes:** train. Not because the code is slow —
75 seconds of compute — but because TensorFlow is a 1.9 GB download that needs
a different Python than the one already on your machine. That download is the
entire cost of the five-minute path, and no amount of documentation removes
it.

---

## The neuron is an abstraction, not a model of biology

Stated plainly, because the name invites the wrong assumption:

- The cell is a **leaky integrator with a threshold**. It is not
  Hodgkin–Huxley. There are no ion channels, no gating variables, no reversal
  potentials, and no spike-initiation mechanism.
- Adaptive thresholds and drive-gated leaks are still **LIF-family**. Calling
  them ALIF or adaptive-LIF would be a claim this interface cannot honour, so
  it is not done here.
- "Spiking" means the signal is a **binary event** and the gradient is an
  **explicit surrogate**, checked against finite differences on the smoothed
  model. The biological reading is a design analogy, nothing more.
- Nothing in this repository is a neuroscience claim. The honest summary:
  *an integer-valued activation with a hand-written derivative, chosen because
  it makes the backward pass cheap to check.*

---

## Extending without forking

Neuron dynamics are an **object**, so a professional can replace them without
touching the kernel, the backward pass, or the planner:

```python
import tensorflow as tf
from axtf.cells import SpikingCell

class SaturatingCell(SpikingCell):
    """Membrane with a nonlinearity, so du/dstate is not the leak."""

    def membrane(self, state, drive, thr):
        return self.leak * state + drive - tf.square(state)

    def membrane_dstate(self, state, drive, thr):
        return self.leak - 2.0 * state

spec = ax.Model()
spec.add_spiking(192, horizon=64, cell=SaturatingCell())
spec.add_spiking(192, horizon=64)          # a plain LIF, same network
```

Override a derivative hook **only** when you replace the function it belongs
to. `spike_fn` needs `spike_d1`; `membrane` needs `membrane_dstate`. If you
override one and not the other, the core builds a backward pass with a
derivative that does not match the forward pass and reports nothing. That is
the one thing it will not do: guess.

`MEASURED` — the gate's `TF CORE` suite checks `membrane_dstate` against
finite differences for four different membranes at `rel.err` between `1.06e-05`
and `1.24e-04`, and separately checks that the kernel *calls the hook*
(`membrane_dstate called 7 times for T=7`) and that a wrong
`membrane_dstate` breaks gradient entries (`18/30 entries differ by more than
50%, worst 95.6%`). See the gate block below.

---

## The silent breaks, including two still published

Every item here produced a plausible number rather than an error. That is the
category that matters: a crash announces itself, a wrong figure does not.

**0. Two of them shipped to PyPI in 0.1.1 and are in every install.** They are
first because they are the reason this section exists now, and because a reader
who installed the package has both of them:

- **`credit_rule="local"` was answered with BPTT arithmetic.** A `str`-Enum
  member compared with `is` against the bare string `"local"` never matches, so
  the local branch was unreachable and the planner reported the local rule's
  memory as though it grew with the horizon — while its own docstring said
  `Theta(S)`, T-independent. Reproduced against the published wheel:
  `depends_on_T = True` where the documentation says `False`, and
  `quality_kept = 1.0` where it says `0.389` at `T=500`.
- **Peak memory was capped at the credit horizon.** `T=100` and `T=400` cost
  the same, because the BPTT tape was truncated at the credit-decay horizon.
  Four times the sequence for the same price, and every "needed GiB" figure
  this project published was an underestimate. `T=400` now costs exactly 4x
  `T=100`.

Both are fixed here, both are pinned by regression checks, and neither is in
the version you can install. See [Install](#install).

**A third did not even reach a green suite.** The `MEMORY MODEL` gate entry ran
`../verify/test_cost.py` — outside the repository, absent from the wheel,
importing a module from `/home/cune`. Thirty-eight green checks, none of them
about code a user receives, and on a fresh clone the suite could not run at all.
That is why the two defects above survived: the suite meant to catch them was
checking a dead copy. It is now `tests/test_memory.py`, against the shipped
module.

**1. The 1/T normalisation.** With a mean readout the maths requires
`dL/ds_t = (1/T)·dL/dmean`. Without it the gradient grows linearly with the
horizon and training diverges. It was *completely absent* from the TF kernel:
the parameter was threaded through four signatures and never applied. The
test that should have caught it re-implemented the backward pass inline and
compared it against itself.
`MEASURED` — the gate now checks both directions:
`without 1/T grows 0.44x, with it 0.11x (same weights, so this is norm_t and
nothing else)`, and `T=4,8,16 -> 0.0186, 0.0094, 0.0021`.

**2. The time axis in `compute_output_shape`.** `_as_shape` mistook a shape
tuple `(None, 16, 24)` for a list of layer inputs, because the batch dim is
`None`. Since the batch dim is *always* `None`, this was the common case, not
an edge case. Every Functional model broke.

**3. Model input is not layer input.** Calibration read `model.inputs[0]`,
which for an embedding stem is the integer token tensor, not the activations
the layer receives.

**4. The backward pass guessed a derivative.** `_lif_backward` read
`cell.leak` as `du/dstate`. For a linear membrane that is accidentally
correct; for any other membrane it is wrong, and a user's cell had no way to
say so without forking the kernel. Fixed by adding
`SpikingCell.membrane_dstate(state, drive, thr)`, which the kernel now calls.

The verification for that fix is the interesting part: finite differences are
**invalid** on the hard spike, because it is a step function. The check
therefore verifies the smooth membrane separately, counts that the hook is
actually called, and compares gradients element by element — after sabotaging
the kernel in the opposite direction and confirming the test fails.

Full account, including the mistakes made while finding all four:
[`docs/UX_RULES.md`](docs/UX_RULES.md).

`MEASURED` note on the count: the numbered list below is **four** silent breaks
of the kind "no error, just a quietly wrong model", and items **0** above add
two more that reached PyPI, so **six** in total are described on this page.
`docs/UX_RULES.md:209` separately enumerates **five** backward-pass and
algorithmic errors the gate caught — a `(1−s)` factor on the wrong term, an
inverted time recursion, a `dL/dthr` with no path through the reset, an early
`return` in the DP, and a wrong tuple field in the Pareto filter. Two lists,
two scopes, **eleven** bugs. The common property is that every one of them was
invisible to inspection and would have been invisible to any test that did not
specifically look for it.

---

## What is not built

Stated as absence, not as a roadmap.

| not built | what that means for you |
|---|---|
| a correct published release | 0.1.1 is on PyPI and carries two planner defects that return wrong numbers silently — see [Install](#install). Install from the repository until a fixed version ships |
| a calibrated regime classifier | the `0.6` / `0.25` thresholds in `analyse()` were fitted before `backward_bytes` entered `activation_fraction`. Both cases the module docstring cites now classify differently: the conv-SNN case is `OFFLOAD`, not `ACTIVATION`, and the 1B transformer at `T=64` is `BALANCED` at 0.2594, not `STATIC`. `tests/test_memory.py` prints both as DIVERGENCE rather than asserting either |
| a byte-reproducible build | rebuilding the sdist yields a wheel of the same size but a different sha256 (zip timestamps, no `SOURCE_DATE_EPOCH`); two builds are not the same bits |
| a CPU-graph-native layer | `run_eagerly=True` is mandatory; `tf.while_loop` is not done |
| energy measurement | no neuromorphic device exists here; no energy number is measured by anyone on this project |
| speed measurement | not measured. Operation counts are not latency |
| the sparse/bit crossover point | measured by nobody, this project included |
| large-vocabulary quality | the T9 corpus is a smoke test, not a benchmark |
| a quality model | `largest_that_fits` is a **capacity** answer; the package has no verified accuracy model and says so |
| sub-segment graph partitioning | `Segment` has no notion of internal structure, so memopt-style splitting is not expressible |
| gradient rules that cut inside a segment | a segment larger than the budget is simply infeasible |

Known code-level limitations, all stated by the framework's own output:

- `run_eagerly=True` is the default because Keras 3 caches a forward pass that
  contains a Python loop over time, and the gradient then lands on stale
  activations. The proper fix is a `tf.while_loop`; **not done**.
- The segmentation planner is optimal but almost always answers "no cuts",
  because its peak is `stored + max_inter` where `max_inter` is a floor. That
  is a property of the conservative model, not a bug. `MEASURED` —
  `python3 examples/advanced/05_segmentation_planner.py`: `(0,4)` → `3.50G`,
  0 cuts; `(0,2,4)` → `4.00G`, 1 cut; `(0,1,3,4)` → `4.50G`, 2 cuts;
  `(0,1,2,3,4)` → `5.00G`, 3 cuts. Cutting the graph makes memory *worse*, and
  the planner agrees with brute force on 5 of 5 budgets.
- Spike density is not a constant. `MEASURED` — same run as the budget table:
  `between tasks: 0.175 .. 0.324` but `within a net: 0.00 .. 0.83`. A single
  mean is useless; the planner budgets the worst case.

---

## Known limitations

These are the ones a reader will hit while trying this, as opposed to the
ones they will hit at scale.

- **Bit-packing buys much less than the headline number suggests, and I could
  not reproduce the headline number.** `FROM LITERATURE` — `16.8×` on its own
  line, that line being `0.16–0.32%` of memory, so `2.6–5.1%` overall. These
  three figures appear only inside a block that
  `examples/advanced/03_credit_rule.py` explicitly labels *"QUOTED, not
  recomputed in this script"*, and **no command I ran prints `16.8×`**. What
  `examples/advanced/04_budget_and_precision.py` does print live is the codec
  table (reproduced in [`examples/README.md`](examples/README.md)), from which
  the shape of the claim is checkable: at firing rate `0.25` the bit codec
  uses `124,780` bytes, the sparse codec `1,999,646`, and ANS `219,494` — so
  *sparsity* is what costs, and past rate ≈ `0.5` the bit codec wins because
  sparsity stops paying. One sentence explains the whole thing: *binary
  activations do not make the gradient binary.* The backward tape is much
  larger than the forward tape and is **always dense**, because the surrogate
  derivative is non-zero everywhere.
- **The credit rule is a function of the horizon and the batch size, not a
  constant.** `MEASURED` —
  `python3 examples/advanced/03_credit_rule.py` prints BPTT against e-prop at
  three scales and the ratio **rises toward 1.0 as `T` grows**:

  ```
    config                        BPTT       e-prop    ratio
    L=8  h=1024 T=32  b=8       8.1 MiB     32.0 MiB     0.3x
    L=16 h=2048 T=64  b=16    129.3 MiB    256.0 MiB     0.5x
    L=32 h=4096 T=128 b=32   2069.3 MiB   2048.0 MiB     1.0x
  ```

  Read carefully: at `T=32, b=8` the local rule is **more** expensive than
  BPTT. It pays only above a crossover batch that *falls* as the horizon
  grows — `48, 24, 12, 6, 3` for `T = 16, 32, 64, 128, 256` (`MEASURED`, same
  run). Any single "e-prop is Nx cheaper" figure is a number without its `T`
  and `b`, and this file will not print one.
- **The local rule's quality price is real and large.** `MEASURED` — same run:
  `T=60 -> 85.5%` of full BPTT, `T=500 -> 38.9%`, a 47-point gap. Beyond the
  measured range the API returns `None` rather than extrapolating, because a
  linear extrapolation would read `0.0` at `T=1024` and look like a claim that
  the local rule is useless. It simply is not known.
- **A readout that averages over time destroys temporal tasks.** This is not
  about spiking; it is the readout. `MEASURED` —
  `python3 examples/04_equal_budget.py` compares four readouts on one task;
  the numbers are in [`examples/README.md`](examples/README.md) with the
  command.
- **A Keras model is not reproducible across two builds** unless you assign
  seeds explicitly; the NumPy core and the planner are bit-exact under a fixed
  seed. `MEASURED` —
  `python examples/advanced/08_reproducibility.py`.
- **The gate proves less than its length suggests, and here is exactly where.**
  See the gate block below.

---

## Verify it yourself

```bash
python3 tests/run_all.py
```

If TensorFlow lives in an environment the gate cannot find — which is the
normal case, because it is usually a project venv rather than `./.venv` — the
gate will report every TensorFlow suite as SKIPPED and exit 2. That is the
honest answer, not a failure to configure anything. Point it at the right
interpreter:

```bash
AX_TF_PY=/path/to/python-with-tensorflow python3 tests/run_all.py
```

`MEASURED` 2026-10-04 — run without `AX_TF_PY` on a machine whose TensorFlow
is in `~/.venvs/ax`, the gate printed `SKIPPED: 1 of 31` and exited 2. That
single skip was a real registration bug, now fixed: `ADV 02` was marked as
needing TensorFlow while still naming the plain `python3` as its interpreter,
so it skipped on every machine while its planning half could have run
anywhere.

Real output, produced **2026-10-04 at 21:38 MSK**, **11 min 24 s** of suite
time, CPU only (`CUDA_VISIBLE_DEVICES=`), TensorFlow 2.21.0, exit code 0,
`ALL GREEN. Every suite ran.` — every one of the 31 suites, none skipped.
Re-run it yourself; the block below is what it printed:

```
  [  OK  ] GATE (numpy core, gradient proven)
  [  OK  ] PLANNER (DP optimal vs brute force)
  [  OK  ] MEMORY MODEL (arithmetic, regimes, codec, credit rules)
  [  OK  ] CREDIT RULES (e-prop vs BPTT, quality price)
  [  OK  ] TF CORE (gradient, normalisation, custom cell)
  [  OK  ] KERAS ASSEMBLY + training via keras.fit
  [  OK  ] API AUDIT (every advertised function called)
  [  OK  ] C++ KERNELS (bit-packing, layout)
  [  OK  ] EXAMPLE 02 (budget, NO TensorFlow)
  [  OK  ] EXAMPLE 01 (train a network)
  [  OK  ] EXAMPLE 03 (custom neuron, no fork)
  [  OK  ] EXAMPLE 04 (equal budget)
  [  OK  ] EXAMPLE 06 (same model on CPU and on GPU)
  [  OK  ] EXAMPLE 07 (train, then generate text)
  [  OK  ] EXAMPLE 05 (T9 next-word prediction)
  [  OK  ] ADV 01 (embedding + stacked layers)
  [  OK  ] ADV 02 (horizon sweep, both parts)
  [  OK  ] ADV 03 (credit rule, the real lever)
  [  OK  ] ADV 04 (budget, codec, precision)
  [  OK  ] ADV 05 (segmentation planner)
  [  OK  ] ADV 06 (devices: CPU and GPU)
  [  OK  ] CAPACITY SOLVER (axplan.solve)
  [  OK  ] ADV 15 (what to build: the inverse question)
  [  OK  ] ADV 07 (silence diagnosis)
  [  OK  ] ADV 08 (reproducibility)
  [  OK  ] ADV 09 (API tour)
  [  OK  ] ADV 10 (save and load)
  [  OK  ] ADV 11 (derivative hooks)
  [  OK  ] ADV 12 (failure modes)
  [  OK  ] ADV 13 (when not to use)
  [  OK  ] ADV 14 (budget matrix)
==========================================================================
  ALL GREEN. Every suite ran. Results may be published.
```

**Thirty-one suites. This file has now said twenty, then twenty-eight, and both
were wrong: twenty ignored suites that exist, twenty-eight missed the three
examples added since. The count is no longer written by hand anywhere — the
`axiomrnn-explain` report reads it out of `tests/run_all.py` at run time,
because a number a human types once goes stale silently and this file was the
proof. `MEMORY MODEL` also changed shape: it used to run
`../verify/test_cost.py`, a file outside this repository that no installation
contains, and now runs `tests/test_memory.py` against the code that actually
ships.**

### What the gate actually examined, and what "OK" is worth

Rule 6: a green that verified nothing is a failure, so here is the inventory
per suite rather than the word "green". `MEASURED` — parsed from the same
run:

| suite | declared checks | wall |
|---|---:|---:|
| GATE (numpy core, gradient proven) | 12 | 2.7 s |
| PLANNER (DP optimal vs brute force) | 12 | 0.2 s |
| MEMORY MODEL (arithmetic, regimes, codec, credit rules) | **101** | 0.1 s |
| CREDIT RULES (e-prop vs BPTT, quality price) | 23 | 0.1 s |
| TF CORE (gradient, normalisation, custom cell) | 20 | 17.4 s |
| KERAS ASSEMBLY + training via `keras.fit` | 45 | 32.3 s |
| API AUDIT (every advertised function called) | **133** (`SKIPPED 0`) | 4.1 s |
| C++ KERNELS (bit-packing, layout) | 5 printed, no aggregate | 1.0 s |
| CAPACITY SOLVER (`axplan.solve`) | 37 | 0.1 s |
| EXAMPLE 02 · 01 · 03 · 04 · 06 · 07 · 05 | **0 surfaced** | 0.0 · 86.1 · 31.1 · 14.1 · 19.4 · 69.3 · **289.4 s** |
| ADV 01 · 02 · 03 · 04 · 05 · 06 · 15 | **0 surfaced** | 17.3 · 36.2 · 0.0 · 0.0 · 0.0 · 5.7 · 0.1 s |
| ADV 07 · 08 · 09 · 10 · 11 · 12 · 13 · 14 | **0 surfaced** | 29.0 · 2.9 · 2.2 · 15.0 · 3.2 · 5.4 · 0.0 · 0.0 s |

**385 declared checks** across the nine suites that report a count, and
**684 s** of suite time. One suite, `EXAMPLE 05`, is 42% of the whole gate.

**One artefact of running it, so nobody is confused by it.** The gate writes its
pid into `tests/.gate.lock` and holds an `flock` on that file for the duration.
`flock` is tied to the file descriptor, so the kernel releases it even if the
process is `SIGKILL`ed — the lock cannot go stale the way a lockfile normally
can. The *text* in the file can still name a dead pid. **That string is
harmless and blocks nothing**; a second gate refuses only if another process
genuinely holds the descriptor, and then it exits 2 and says so.

Two honest caveats, both `MEASURED` from that run:

1. **The seven `EXAMPLE` suites and the fifteen `ADV` suites print zero check
   lines.** The gate filters suite output down to lines containing
   `PASS`/`ok]`/`Error`/`Traceback`/`SKIPPED`, and the examples mostly print
   none of those. For those twenty-two suites an `[  OK  ]` attests **only that
   the process exited 0**, plus any evidence markers the suite declares — 14 of
   the 31 suites now declare them, so a run that printed nothing it claims to
   show is a `FAIL`, not a green. It still cannot catch a *wrong* number in an
   unmarked example; only a non-zero exit.
2. **`MEMORY MODEL` was testing code that does not exist in any installation.**
   It ran `../verify/test_cost.py` — outside this repository, absent from the
   wheel and the sdist — which imported `axon.cost` from
   `/home/cune/axon/cost.py`. Thirty-eight green checks certified a module no
   user can install, and on a fresh clone, including every GitHub runner, the
   suite could not run at all. It now runs `tests/test_memory.py` against the
   shipped `axplan.memory`, which is where the two planner defects described in
   [Install](#install) were found.

The gate is not optional. No result here is published until it is green — a
rule adopted after three consecutive experiments looked convincing and were
wrong. But a green is only worth its inventory size, so the inventory size is
printed above.

---

## Documentation

| | |
|---|---|
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | the same five-minute path, expanded |
| [`examples/README.md`](examples/README.md) | all 23 examples, each with its real output and the command that produced it |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | what we deliberately do **not** build, and the four packages |
| [`docs/UX_RULES.md`](docs/UX_RULES.md) | 20 rules, each grown from a specific failure |
| [`docs/ATTACKING_THE_QUADRATIC.md`](docs/ATTACKING_THE_QUADRATIC.md) | an independent attack on the premise: replacing the KV cache with a spike-gated fast-weight state |
| [`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) | the original design record (Russian, unedited) |

---

## License

Apache-2.0. See [LICENSE](LICENSE).