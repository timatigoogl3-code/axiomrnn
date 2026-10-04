# Changelog

All notable changes. Dates are ISO.

## [Unreleased] — 2026-10-04

The published **0.1.1 is defective**. Both planner bugs below return plausible
numbers rather than raising, so they were invisible to every consumer. They
were found by writing a test for the memory model that did not already exist,
and both are now pinned by regression checks. Anyone who installed 0.1.1 should
read this before trusting a credit-rule figure.

### Fixed

- **`credit_rule="local"` was answered with BPTT arithmetic.**
  `axplan/memory.py` passed the bare string `"local"` into `credit_bytes()`,
  which dispatched on `is CreditRule.LOCAL`. `CreditRule` is a `str`-Enum, so
  the string *equals* the member but is never *identical* to it, the test
  failed, and execution fell through to the BPTT branch. The local rule
  reported `depends_on_T=True` and `quality_kept=1.0` — the exact opposite of
  what `axplan/credit.py` documents — and `analyse()` charged BPTT bytes for a
  local-rule request. `credit_bytes()` now normalises `rule` and `neuron`
  through their enums, so a string and its member are the same request and an
  unknown value raises instead of silently becoming BPTT.

  Reproduced against the published artifact, installed from the PyPI wheel into
  a clean venv:

  ```
  $ pip install --no-deps axiomrnn==0.1.1     # the file as uploaded
  >>> credit_bytes("local", NeuronModel.LIF, 10**9, 500, 8, 1024, 8)
  depends_on_T = True     # documented: False, "T-INDEPENDENT"
  quality_kept = 1.0      # documented: 0.389 at T=500
  ```

  and after the fix, same call from a wheel built off this tree:

  ```
  depends_on_T = False
  quality_kept = 0.389
  ```
- **Peak memory was capped at the credit horizon.** `backward_bytes()` truncated
  the BPTT tape at `CreditModel.horizon()`, so `analyse()` returned an identical
  peak for `T=100` and `T=400`. Reproduced against the published 0.1.1 wheel:

  ```
  BPTT tape T=100 : 294,801,408 B  (0.2746 GiB)
  BPTT tape T=400 : 294,801,408 B  (0.2746 GiB)   identical
  ```

  Four times the horizon charged at the same price. After the fix, from a wheel
  built off this tree: `446,668,800 B` at `T=100` and `1,786,675,200 B` at
  `T=400` — exactly 4x. Two thirds of the horizon was free, which is the one
  thing a memory planner must never do, and it contradicted the `credit_rule`
  field's own documentation of BPTT as `Theta(T*A)`. The credit horizon still
  sets `MemoryPlan.credit_horizon`, which is a statement about credit rather
  than about tape length. Retraction: the "1B connections fit in 5 GiB" figure
  that the old test asserted held only because of this bug — measured now, 1B
  at `T=1000` costs 10.23 GiB under BPTT and 5.96 GiB under the local rule, and
  neither fits. No published document quoted it.
- **Every `project.urls` entry named a third party's repository.** The published
  0.1.1 `METADATA` reads
  `Source: https://github.com/timatigoogl3-code/axiomrnn` while `git remote`
  says `cubetitled-ui`. `timatigoogl3-code` is the account this environment's
  `gh` is authenticated as, not the author's. Corrected in `pyproject.toml` and
  verified against the rebuilt wheel's `METADATA`, and `tests/test_api_surface.py`
  now reads the URLs, the remote, the LICENSE and the classifiers so it cannot
  drift again — sabotage-verified: reinstating the old URLs and the 3.14
  classifier turns 9 checks red.
- **The wheel advertised Python 3.14 while refusing to install on it.**
  `requires-python = ">=3.10,<3.14"` with a
  `Programming Language :: Python :: 3.14` classifier. The classifier is the
  one a reader sees first.
- **`ADV 02` skipped on every machine.** Registered as needing TensorFlow while
  still naming plain `python3` as its interpreter, which on this machine is
  3.14 and has no TensorFlow wheel. Separately, the gate's availability guard
  used `os.path.exists("python3")`, which is always false for a bare command
  name resolved through `PATH`, so "is this interpreter runnable" and "is
  TensorFlow available" were the same question. They are now separate.

### Added

- **`tests/test_memory.py`** — 101 checks against the shipped `axplan.memory`.
  It replaces a gate entry that ran `../verify/test_cost.py`: a file outside the
  repository, absent from the wheel and the sdist, which began with
  `sys.path.insert(0, '/home/cune')` and imported `axon.cost` from
  `/home/cune/axon/cost.py`. Its green checks therefore certified a module no
  installation contains, and on a fresh clone — including every GitHub runner —
  the suite could not run at all. It also tested a stale predecessor of
  `axplan/memory.py`, which is how both planner defects above survived.
  The suite prints two `DIVERGENCE` lines it deliberately does **not** assert:
  the `0.6`/`0.25` regime constants were fitted before `backward_bytes` entered
  `activation_fraction`, so both cases cited in the module docstring now
  classify differently.
- **`AX_TF_PY`** — names the interpreter the gate should test TensorFlow with.
  Without it the gate discovers candidates but cannot reach a venv it does not
  know about, and reports SKIP rather than guessing.
- **`BudgetSpec` refuses what it cannot price**: an unknown `credit_rule` or
  `neuron_model`, `spike_rate <= 0`, `timesteps < 1`. Each previously returned a
  confident number for a request that did not mean anything.

### Changed

- `axplan/memory.py` docstring no longer quotes `96.0% vs 96.5% in ICLR'26` or
  a test count. Both figures described the forward-tape-only model that
  `axon/cost.py` implemented, which is not the code in this file.
- The gate reports its own suite count, read from `tests/run_all.py` at run
  time instead of a literal in `axiomrnn.py`. That literal said 18 while the
  gate ran 28, then 31.
- README no longer says the package is unpublished. It is on PyPI as 0.1.0 and
  0.1.1, uploaded 2026-10-04, and the page now states plainly that the
  installed version carries the two planner defects above.

## [0.1.1] — 2026-10-04

Uploaded to PyPI at 11:54:53Z. Caps `requires-python` at `<3.14` (TensorFlow has
no 3.14 wheel outside a release candidate) and stops the gate reporting a pass
when a suite did not run. **Contains the two planner defects listed under
Unreleased above.**

## [0.1.0] — 2026-10-02

First public release. Everything below was found by measurement, and
several items are recorded as bugs we shipped and then fixed.

### Added

- **Budget planner** (`axplan`): memory model, Pareto-front DP for graph
  segmentation, credit rules. Pure Python — no TensorFlow import anywhere
  in the package, so planning runs on a machine with no GPU stack.
- **Spiking kernel** (`axtf`): `SpikingRNNCell`, an explicitly backwards
  layer whose gradient is verified against finite differences to
  `rel.err ≤ 1e-8`.
- **Neuron as an object**: `SpikingCell` with four overridable dynamics
  methods and three derivative hooks, so a user's neuron drops into the
  framework without forking it.
- **`explain()`**: answers "what can you learn on this budget", which no
  other tool does. Also available as the `axiomrnn-explain` console
  command.
- **Verification harness**: `axon/gate.py`, 12 checks, mandatory before
  publishing any result, wrapped by `tests/run_all.py` — 20 suites covering
  the numpy core, the planner, the capacity solver, the memory model, the
  credit rules, the TF kernel, Keras assembly, the API audit, the C++
  kernels, and 12 of the 20 examples.
- **API audit** (`tests/test_api_surface.py`): every advertised symbol is
  called for real, with values asserted — not merely introspected.
- **Capacity solver** (`axplan/solve.py`): the inverse question — given a
  budget, the largest configuration that fits, the smallest step up that does
  not, and which term of the memory model binds. It refuses to answer when
  nothing fits, and reports no boundary when the answer sits on the search
  ceiling. It makes **no quality claim**: this package has no verified quality
  model, and says so in its own output.
- **Twenty examples** (5 core, 15 advanced), 12 of which run inside the
  gate. Each prints what it measured; no example prints a number it did not
  compute.
- **pip packaging** (PEP 621): core needs only numpy; `[tf]` and
  `[tf-gpu]` extras.

### Measured, and worth stating plainly

- Spiking vs dense on a temporal task: **0.9938 vs 0.9988** — parity, not
  superiority. The 0.26 gap we first reported was **our own** readout, not
  a property of spikes.
- Bit-packing: 16.8x on its own line, but that line is 0.16–0.32% of
  memory, so the real effect is **2.6–5.1%**. Binary activations do not
  make the gradient binary.
- Credit rule: **17x** (8.7x at T=128) on the line holding ~97% of memory.
  This is the real lever.
- Memory model vs measured allocation on GPU: **0.99**.

### Fixed

- **`norm_t` was dead.** The 1/T gradient normalisation was threaded
  through four signatures and never applied. The test that should have
  caught it re-implemented the backward pass inline and compared it
  against itself. Caught by attempting to break the code and observing
  that the test stayed green.
- **Time axis lost in `compute_output_shape`.** `_as_shape` mistook a
  shape tuple for a list of inputs because the batch dimension is `None` —
  which made this the common case, not an edge case. Every Functional
  model was affected.
- **Model input taken for layer input.** Calibration read
  `model.inputs[0]`, which for an embedding stem is the integer token
  tensor rather than the activations the layer receives.
- **`add_spiking(cell=...)` raised `TypeError`.** The advertised ability to
  extend the system without forking did not work through the public API;
  it only worked by importing `axtf` directly.
- **The backward pass guessed a derivative.** `_lif_backward` read
  `cell.leak` as `du/dstate`. That is accidentally correct for a linear
  membrane and silently wrong for every nonlinear one, with no way for a
  user's cell to correct it short of forking the kernel. Added
  `SpikingCell.membrane_dstate(state, drive_next)`; the kernel now asks the
  cell instead of assuming. Verified by element-wise gradient comparison and
  a call counter, **not** by finite differences — a hard spike is
  discontinuous, and a finite-difference check on the layer reported
  `rel.err = 0.98` on correct code.
- **Default readout was wrong.** Mean-over-time readout cost 0.24 accuracy
  on temporal tasks and had been the default. Now `flatten`.
- **Old `examples/` were stale** and disagreed with the core. Replaced.

### Documented limitations

- `run_eagerly=True` is the default: Keras 3 caches a forward pass that
  contains a Python loop over time, so gradients would otherwise land on
  stale activations. The real fix is `tf.while_loop`, **not done**.
- The segmentation planner is optimal but almost always returns zero
  cuts: its peak is `stored + max_inter`, and `max_inter` is a floor.
- The planner cannot cut *inside* a segment.

### Not measured

- The sparse/bit crossover — by nobody, us included.
- Energy cost: no neuromorphic hardware on the test machine.
- Whether spiking is faster than dense on a GPU.
