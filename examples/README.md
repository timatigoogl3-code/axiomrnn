# Examples

**Twenty runnable examples — five in the core set, fifteen in `advanced/` —
plus `advanced/_generate.py`, which is a generator and not an example.** A
previous version of this file said "Twelve". It was wrong, and so were most of
its numbers.

Everything below was produced on **2026-10-04** by the command printed above
each block. Nothing here is carried forward from an earlier session's notes.
Numbers move between runs — see *Why these numbers are not promises* at the
bottom.

**Tag key.** `MEASURED` — produced by a command in this file, in this
session. `FROM LITERATURE` — someone else's result, quoted, **not** re-run.

---

## Which examples need TensorFlow

Seven need nothing but NumPy. That is not a trick: `axplan` never imports
TensorFlow, so all budget planning works on a machine with no GPU stack at
all.

| | needs TensorFlow | needs a GPU |
|---|---|---|
| `02_budget.py` | **no** | no |
| `advanced/02_horizon_sweep.py` | **partly** — see below | no |
| `advanced/03_credit_rule.py` | **no** | no |
| `advanced/04_budget_and_precision.py` | **no** | no |
| `advanced/05_segmentation_planner.py` | **no** | no |
| `advanced/13_when_not_to_use.py` | **no** | no |
| `advanced/14_budget_matrix.py` | **no** | no |
| `advanced/15_what_to_build.py` | **no** | no |
| `01_minimal.py` · `03_custom_neuron.py` · `04_equal_budget.py` · `05_t9.py` | yes | no |
| `advanced/01` · `06` · `07` · `08` · `09` · `10` · `11` · `12` | yes | no |

`MEASURED` — for the "no" rows, run with `/usr/bin/python3`, which on this
machine is Python 3.14.4 with NumPy 2.5.0 and **no TensorFlow module at all**.
For the "yes" rows, run with `/home/cune/.venvs/ax/bin/python`, Python 3.12.13,
TensorFlow 2.21.0, Keras 3.15.1.

### `advanced/02_horizon_sweep.py` is half a suite

Its Part 1 is pure planning. Its Part 2 trains, and when TensorFlow is absent
it prints one line and stops:

```
  PART 2 -- MEASURED QUALITY (needs TensorFlow)
  TensorFlow absent; planning above is the whole story here.
```

`MEASURED` — and the gate runs this suite **without** TensorFlow
(`tests/run_all.py:68-69`, `need_tf=False`), so `[  OK  ] ADV 02` certifies
Part 1 and never executes Part 2. That is a green that verified half of what
it is named after.

---

## Reproducing the numbers

```bash
cd axiomrnn

# planning only -- bare python3, no TensorFlow
python3 examples/02_budget.py
python3 examples/advanced/03_credit_rule.py
python3 examples/advanced/04_budget_and_precision.py
python3 examples/advanced/05_segmentation_planner.py
python3 examples/advanced/13_when_not_to_use.py
python3 examples/advanced/14_budget_matrix.py
python3 examples/advanced/15_what_to_build.py

# with TensorFlow -- this machine keeps it in a separate venv
PY=/home/cune/.venvs/ax/bin/python
$PY examples/01_minimal.py
$PY examples/03_custom_neuron.py
$PY examples/04_equal_budget.py
$PY examples/05_t9.py
$PY examples/advanced/02_horizon_sweep.py     # for Part 2
$PY examples/advanced/06_devices.py
```

All numbers in this file were produced with `CUDA_VISIBLE_DEVICES=` set to the
empty string, so **everything below is CPU-only**. The machine has an RTX 3050;
it was hidden deliberately, so that the figures describe what a reader without
a GPU would get.

---

## Core set

| # | file | teaches | wall clock | peak RSS |
|---|---|---|---|---|
| 01 | `01_minimal.py` | train a spiking network | **75.2 s** | **641 MB** |
| 02 | `02_budget.py` | what can you learn under a budget | **0.04 s** | **15 MB** |
| 03 | `03_custom_neuron.py` | replace the neuron, no fork | **76 s** | **640 MB** |
| 04 | `04_equal_budget.py` | compare two nets honestly | **839 s** | **728 MB** |
| 05 | `05_t9.py` | next-word prediction, T9 style | **558 s** | **1,079 MB** |

`MEASURED` — wall clock and peak RSS from `/usr/bin/time -f "%e s  %M KB
peakRSS"`, `CUDA_VISIBLE_DEVICES=`, one process at a time. `01_minimal.py` also
ran as a gate suite at `190.9 s` with other work on the machine; the difference
is contention, not a different script.

### 01 — a spiking network that actually works

```bash
/home/cune/.venvs/ax/bin/python -u examples/01_minimal.py
```

`MEASURED` — exit 0, 75.2 s, 641 MB peak RSS.

```
data: 6000 examples, T=16, channels=12, classes=4
 Total params: 51,844 (202.52 KB)
 Trainable params: 51,460 (201.02 KB)
 Non-trainable params: 384 (1.50 KB)

Epoch 15/15
75/75 - 5s - 66ms/step - accuracy: 1.0000 - loss: 3.7285e-04 - val_accuracy: 0.9983 - val_loss: 0.0071

==================================================================
  held-out accuracy:      0.9983
  chance:                 0.2500
==================================================================
```

Four classes, and the label says *which quarter of the sequence* was active.
A network that averages over time loses the label, which is exactly the case
where a spiking network has to hold on to time.

Two things about this example that the previous version of this file did not
say:

- **It used not to be a five-minute step.** It ran the whole test gate as a
  subprocess at the end (`examples/01_minimal.py:165-175`), so running it
  trained the network, then ran all 28 gate suites, which trained it a second
  time. Observed with `pgrep` during this documentation pass. **Fixed at
  23:58:59 on 2026-10-03**; the subprocess is now behind `AX_RUN_GATE=1` and the
  script otherwise prints the gate command for you to run yourself.
- **A TensorFlow warning is expected and harmless.**
  `RuntimeWarning: SpikingRNNCell: could not infer the output shape from
  KerasTensor; the time axis may become None.`

### 02 — the budget as an input (no TensorFlow)

```bash
python3 -u examples/02_budget.py
```

`MEASURED` — exit 0, 0.04 s, 15 MB peak RSS. The first thing it prints is the
claim being tested:

```
  TensorFlow in this environment: ABSENT, and not needed
```

Network: 2 spiking layers of 1024, output into 1000 classes, `T=256`, batch 32.

```
  budget                    usable      needed     fits?
  consumer 6 GB           4.12 GiB    1.57 GiB       yes
  pro 24 GB              17.96 GiB    1.57 GiB       yes
  datacenter 80 GB       61.04 GiB    1.57 GiB       yes
```

`4.12 GiB` is what a 6 GB card really gives you: the CUDA context, cuBLAS
workspaces and allocator fragmentation are real bytes.

Now the width grows and the budget starts to bind:

```
  width       needed GiB     6 GB    80 GB
  1024              1.57      yes      yes
  4096              4.82      yes      yes
  12288            13.84       NO      yes
  16384            18.53       NO      yes
  20480            23.35       NO      yes
  24576            28.30       NO      yes
```

The planner found the width-12288 wall on its own. One line of code, different
budgets, different answers.

**Every figure in this block was larger before 2026-10-04 and the old ones were
wrong.** `axplan.memory.backward_bytes` capped the BPTT tape at the credit
horizon, so most of `T=256` was charged at no cost. The fix raises these
numbers — width 12288 now needs 13.84 GiB where it previously reported 4.38 —
and it moves the wall where it crosses 6 GiB from 20480 down to 12288. If you
read an older copy of this file, its peaks are underestimates.

And the credit rule, which is the actual lever:

```
    BPTT   -> Theta(T * A), memory GROWS with the horizon
    e-prop -> Theta(S),      T-INDEPENDENT
    now: BPTT, 0.063 GiB
    e-prop LIF: 0.008 GiB (0.50x Adam -- cheaper than momentum)
    batch crossover: batch* = 4
    -> at batch=32 the local rule is ALREADY cheaper.
  QUALITY PRICE of the local rule (measured, FPTT):
    T=60 -> 85.5%   T=500 -> 38.9%   a 47-point gap
    at T=256: ~64.7% of BPTT
```

**Both defects in this example's output were fixed 2026-10-04**, and both are
recorded here because the numbers above are the corrected ones:

- The report block was printed **twice**, identically: `explain()` prints by
  default *and* returns the report, so `print(spec.explain(...))` emitted
  162 lines where the same 81-line report appeared at 1–81 and again at 82–162.
  The example now passes `verbose=False` and prints the return value once.
- The "VERIFIED, AND NOT" block claimed `tests/run_all.py 18 suites` while the
  gate ran 28. That string is no longer hardcoded at all: the report reads the
  count out of `tests/run_all.py` at run time and prints `31 suites`, because a
  number a human types once goes stale silently — this file went stale twice.

### 03 — your neuron, no fork

```bash
/home/cune/.venvs/ax/bin/python -u examples/03_custom_neuron.py
```

`MEASURED` — exit 0, 76 s, 640 MB peak RSS.

```
  TF 2.21.0, Keras 3.15.1
  task: 4000 examples, T=16, classes=4

  1. the layer builds, output shape (None, 16, 128)  expected (None, 16, 128)
  2. spike density 0.0269 OK
  3. gradient norm 0.000067 OK

  4. your neuron: density 0.0317, gradient norm 0.000108

  --------------------------------------------------------------
  your neuron: val_accuracy 0.9925, chance 0.2500
  --------------------------------------------------------------
  loss: 0.5661 -> 0.0024
```

**The previous version of this file reported `val_accuracy 0.9975` and
`gradient norm 0.001069`. Both were wrong.** The live gradient norm is about
**ten times smaller** (`0.000108` against `0.001069`), which matters: a
gradient norm an order of magnitude off is a sign the reported run used a
different configuration, not a sign the neuron got better.

The example also explains why you cannot check a hard spike with finite
differences: the derivative of a step is identically zero, and the differences
return exactly `0` on 18 of 20 coordinates. The gate therefore checks the
*smoothed* model, where `rel.err` reaches `1e-8`.

### 04 — how to compare honestly

```bash
/home/cune/.venvs/ax/bin/python -u examples/04_equal_budget.py
```

`MEASURED` — exit 0, **839 s** (the slowest example here), 728 MB peak RSS.
`T=16`, label = which block of time, 4 classes, chance 0.2500. Width 256,
2 layers, **identical for every arm**. The threshold is scanned and the best is
kept, which is the only fair way to compare a spiking net against a dense one.

```
  run (the threshold is scanned, the best is kept):
    dense                  train 1.0000  VAL 0.9988  <- keep
    spiking thr=0.4        train 0.6947  VAL 0.6813  <- keep
    spiking thr=0.7        train 0.7380  VAL 0.7188  <- keep
    spiking thr=1.0        train 0.7596  VAL 0.7438  <- keep
    spiking_flat thr=0.4   train 1.0000  VAL 0.9912  <- keep
    spiking_flat thr=0.7   train 1.0000  VAL 0.9906
    spiking_flat thr=1.0   train 1.0000  VAL 0.9850
    smooth thr=0.4         train 0.7500  VAL 0.7406  <- keep
    smooth thr=0.7         train 0.7404  VAL 0.7137
    smooth thr=1.0         train 0.7043  VAL 0.6825
```

and then the decomposition, which is the point of the whole file:

```
  DECOMPOSITION BY CAUSE (each is a separate measurement)
    dense                          0.9988
    spiking, readout flatten       0.9912
    spiking, readout mean          0.7438
    spiking smooth, readout mean   0.7406

    the bit AND the gradient       0.0031
    the readout over time          0.2475
    the readout explains 99% of the discrepancy

    CAREFUL: the first row cannot be called "the cost of the
    gradient". The smooth variant differs in the forward value AND
    in the derivative. One measurement cannot separate them. The
    correct wording is "the joint contribution of the bit and the
    gradient".
```

```
  CONCLUSION
  Spiking with a flatten readout trails dense by 0.0075.
  That is within noise on a task like this.
```

**The previous version of this file reported `0.9988 / 0.9919 / 0.7525 /
0.7550`. One of those four was right.** The dense figure reproduces exactly.
`0.9919` does not — the live run says `0.9912`. Neither `0.7525` nor `0.7550`
appears; the live mean-readout figure is `0.7438` and the live smooth
mean-readout figure is `0.7406`. The *conclusion* of the old block survives
untouched, and the example is honest enough to include its own architecture
check that the arms are not identical:

```
  ARCHITECTURE HONESTY CHECK
    spiking             0.28 MiB
    spiking_flat        0.34 MiB
    smooth              0.28 MiB
    identical: False
```

**This is the single most-quoted measurement in the repository, and until this
pass it was not in the test gate at all.** `tests/run_all.py` has no suite for
`examples/04_equal_budget.py`; the comment at `tests/run_all.py:86-89` claims
"every example is now a suite", which is false for this one file. Nineteen of
twenty examples are covered. Reported to the owner; documented here rather than
papered over.

### 05 — next-word prediction, T9 style

```bash
/home/cune/.venvs/ax/bin/python -u examples/05_t9.py
```

`MEASURED` — exit 0, **558 s** (9 min 18 s), **1,079 MB** peak RSS: the
slowest and the largest example in the set. It is also the slowest gate suite,
which is why `tests/run_all.py` runs it last.

```
  TensorFlow 2.21.0   device: CPU
  corpus: corpus_en.txt, 114,959 words

  training
    tokens            114,959
    vocabulary        1,807
    windows           114,953
    loss              5.9423 -> 4.8166
    train accuracy    0.1204
    held-out accuracy    0.0999

  baselines on the same held-out set:
    always the most common word      0.0379
    always the most common bigram    0.0041
    this model                       0.0999
  beats both baselines: True

  partial words, like a phone keyboard:
      'the ' -> of 0.30  and 0.13  i 0.05  which 0.05  the 0.04
    'i will ' -> not 0.10  the 0.09  have 0.06  say 0.06  you 0.05
    'she was' -> and 0.17  to 0.12  in 0.04  the 0.04  of 0.04
       'it ' -> and 0.26  i 0.06  the 0.06  to 0.04  she 0.04
    'and th' -> and 0.19  to 0.07  i 0.06  of 0.04  the 0.03
       'zzz' -> and 0.22  of 0.16  the 0.08  which 0.07  i 0.06
```

`2.6×` the majority-word baseline and `24×` the bigram baseline, on 114,959
words of Pride and Prejudice with a deliberately naive tokenisation.

**The previous version of this file reported `loss 5.9384 -> 4.8147`,
`held-out acc 0.0968`, and `'i will' -> not 0.19, have 0.16, be 0.07`.** The
loss figures are close and still wrong. The held-out accuracy is `0.0999`, not
`0.0968`. The predictions are not close at all — the live `'i will '` gives
`not 0.10, the 0.09, have 0.06`, where the old file claimed `not 0.19, have
0.16`. **The two baselines are the only figures that reproduce exactly**
(`0.0379`, `0.0041`), which makes sense: they are counts, not training
outcomes.

Treat the absolute numbers as a smoke test. The example says so itself:

> large-vocabulary quality. The built-in corpus is small and repetitive, so
> treat these numbers as a smoke test, not a benchmark.

**A stale hardcoded number survives inside this example too,**
`MEASURED` — its closing block still says `0.9919 vs 0.9981, measured
separately in verify_14`, at `examples/05_t9.py:32`, `:185` and `:332`. That is
the same stale quality pair as `adv13`, in a *third* file. The live figure from
`examples/04_equal_budget.py` is `0.9912` against `0.9988`. Reported to the
owner.

---

## Advanced set

| # | file | teaches | TF | wall | peak RSS |
|---|---|---|---|---|---|
| 01 | `advanced/01_embedding_and_layers.py` | embedding stem, stacked layers, widths that differ | yes | **19.8 s** | **641 MB** |
| 02 | `advanced/02_horizon_sweep.py` | what each horizon costs, and what it buys | partly | **0.14 s** Part 1 / **63 s** both | **30 MB** / **818 MB** |
| 03 | `advanced/03_credit_rule.py` | the credit rule is the lever, not the bit | **no** | **0.03 s** | **15 MB** |
| 04 | `advanced/04_budget_and_precision.py` | hardware profiles, codec choice, precision | **no** | **0.03 s** | **15 MB** |
| 05 | `advanced/05_segmentation_planner.py` | keep-or-recompute, and why cuts usually lose | **no** | **0.03 s** | **15 MB** |
| 06 | `advanced/06_devices.py` | CPU and GPU, same code | yes | **4.6 s** | **642 MB** |
| 07 | `advanced/07_silence_diagnosis.py` | a network that will not fire, and why | yes | **38 s** | **666 MB** |
| 08 | `advanced/08_reproducibility.py` | what is deterministic here and what is not | yes | **1.9 s** | **568 MB** |
| 09 | `advanced/09_api_tour.py` | every public name, and whether it needs TF | yes | **2.9 s** | **542 MB** |
| 10 | `advanced/10_save_and_load.py` | train once, serve without a GPU | yes | **16.4 s** | **639 MB** |
| 11 | `advanced/11_derivative_hooks.py` | the seven hooks, and how to check yours | yes | **1.8 s** | **568 MB** |
| 12 | `advanced/12_failure_modes.py` | the six ways this breaks, and the exact message | yes | **2.7 s** | **597 MB** |
| 13 | `advanced/13_when_not_to_use.py` | when *not* to use this | **no** | **0.04 s** | **15 MB** |
| 14 | `advanced/14_budget_matrix.py` | one architecture, every budget, side by side | **no** | **0.03 s** | **15 MB** |
| 15 | `advanced/15_what_to_build.py` | the inverse question | **no** | **0.06 s** | **15 MB** |

`MEASURED` — as above. **All eight TensorFlow-free examples together take
0.39 s**, which is the point of `axplan`: the entire planning surface costs
less time than one `import`.

### adv 01 — embedding stem + stacked spiking layers

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/01_embedding_and_layers.py
```

`MEASURED` — exit 0, 19.8 s, 641 MB peak RSS. Device: **CPU**.

```
  task: which half of 12 tokens has the higher mean (binary)
  a bag-of-words model cannot do this: order is the whole signal
  4,000 sequences, class balance 0.50

  before calibration:
    spiking_0  firing 0.0000  dead 1.0000
    spiking_1  firing 0.0000  dead 1.0000

  after calibration:
    spiking_0  firing 0.0455  dead 0.0000
    spiking_1  firing 0.0000  dead 1.0000

  loss        0.7402 -> 0.1030
  train acc   0.9619
  held-out    0.9387   (chance 0.5037)

  spiking layers in the graph: 2
  widths (differ on purpose):   [128, 96]
```

**The second layer stays dead after calibration**, and the example reports it
rather than hiding it. That is a real finding, not a bug: a wider first layer
does not guarantee a live second layer, and you should measure it. The
previous version of this file reported `firing 0.0468` and `held-out 0.9413`;
the live run gives `0.0455` and `0.9387`. Close, and still wrong — the
structural finding reproduces exactly.

### adv 02 — horizon sweep

```bash
python3 -u examples/advanced/02_horizon_sweep.py
```

`MEASURED` — exit 0, no TensorFlow, 0.14 s, 30 MB peak RSS. **Part 1 only.**

```
  width=256  layers=2  classes=4
  budget: consumer 6 GB

      T   peak GiB  credit k      regime   batches   fits
      8       0.51        66  activation        32    yes
     16       0.52        66  activation        16    yes
     32       0.54        66  activation         8    yes
     64       0.57        66  activation         4    yes
    128       0.57        66  activation         2    yes
```

`batches` is the crossover batch: below it, BPTT is *cheaper* than the local
rule. **The crossover batch falls as the horizon grows** — the opposite of the
usual intuition about when local credit rules pay off.

`MEASURED` — with TensorFlow, the same script takes 63 s and 818 MB, and Part 2
measures quality:

```
  PART 2 -- MEASURED QUALITY
      T     held-out accuracy    vs T=8
      8                0.7483   +0.0000
     16                0.8267   +0.0783
     32                0.9217   +0.1733
```

**Formatting defect, `MEASURED` — fixed 2026-10-04.** The embedded `explain()`
report inside this example was truncated mid-sentence and printed fragments,
identically with and without TensorFlow:

```
    We do NOT claim e-prop is better. We say what it costs and what
      tests/run_all.py       18 suites: memory model, credit rules, TF
```

`We say what it costs and what` stopped mid-sentence because the console filter
in the gate keeps only lines matching `PASS`/`ok]`/`Error`/`Traceback`/
`SKIPPED`, and that report is a few hundred lines of prose. The `18 suites`
line was a second instance of the same stale literal. What this example
actually prints is in the block above; the gate's job here is only to confirm
the run did not crash.

### adv 03 — the credit rule is the lever, not the bit

```bash
python3 -u examples/advanced/03_credit_rule.py
```

`MEASURED` — exit 0, 0.03 s, 15 MB peak RSS, no TensorFlow.

```
  PART 2 -- THE RATIO THAT ACTUALLY MATTERS
  config                             BPTT       e-prop    ratio
  L=8 h=1024 T=32 b=8              8.1 MiB      32.0 MiB     0.3x
  L=16 h=2048 T=64 b=16          129.3 MiB     256.0 MiB     0.5x
  L=32 h=4096 T=128 b=32        2069.3 MiB    2048.0 MiB     1.0x
```

**Read that table carefully, because the prose around it does not match it.**
The ratio is BPTT ÷ e-prop memory. At `T=32, b=8` the local rule is **3×
more expensive** than BPTT, and the advantage only reaches parity at
`T=128`. Two lines later the same script says:

```
  Note how the ratio FALLS at T=128 (17x -> 8.7x). Do not carry
  the headline number over to long horizons without re-measuring.
```

**No run produces `17x` or `8.7x`.** They are quoted strings in the example's
source. The direction of the live table is *rising toward 1.0* as `T` grows,
not falling from 17.

Part 3, the crossover batch, and Part 4, the quality price:

```
       T     crossover batch
      16                  48
      32                  24
      64                  12
     128                   6
     256                   3

       T    quality kept
      16          100.0%
      60           85.5%
     200           70.7%
     500           38.9%
    2000            None   no data -- we do NOT extrapolate
```

### adv 04 — budget, codec, precision

```bash
python3 -u examples/advanced/04_budget_and_precision.py
```

`MEASURED` — exit 0, no TensorFlow.

```
  profile                  vram   host  offload      usable
  consumer 6 GB              6G    32G     True      4.12 GiB
  consumer 8 GB              8G    32G     True      5.65 GiB
  pro 24 GB                 24G    64G     True     17.96 GiB
  datacenter 80 GB          80G   512G    False     61.04 GiB
```

```
     rate    best        bytes          bit       sparse          ANS
     0.01     ans       31,457      124,780       79,691       31,457
     0.05     ans       62,796      124,780      399,684       62,796
     0.08     ans       86,301      124,780      639,678       86,301
     0.25     bit      124,780      124,780    1,999,646      219,494
     0.50     bit      124,780      124,780    3,999,598      415,365
     0.90     bit      124,780      124,780    7,199,522      728,760
```

The planner switches codec on its own, at a density it computes rather than
one you set.

```
  config                                 B/param
  fp32 weights + fp32 master + fp32 Adam        16.0
  fp16 weights + fp32 master + fp32 Adam        14.0
  fp16 weights + fp32 Adam momentum              6.0  needs fp32 master
  fp16 weights, fused update, no momentum        2.0  needs fp32 master
```

fp32 → fp16 on the weights is a **12%** saving, not 2×. Adam's fp32 state is
the floor: 8 B/param whatever you do to the weights.

And the infeasible path, which matters more than the feasible one:

```
  config                            need    have   verdict
  L=8 h=512 T=32 b=8               0.57G      6G      FITS
  L=16 h=2048 T=128 b=64           9.44G      6G    NO FIT
  L=32 h=4096 T=512 b=128         72.63G      6G    NO FIT
  L=64 h=8192 T=1024 b=256       584.92G      6G    NO FIT
```

### adv 05 — segmentation: keep or recompute

```bash
python3 -u examples/advanced/05_segmentation_planner.py
```

`MEASURED` — exit 0, no TensorFlow.

```
  partition               exact peak   cuts
  (0, 4)                      3.50G      0
  (0, 2, 4)                   4.00G      1
  (0, 1, 3, 4)                4.50G      2
  (0, 1, 2, 3, 4)             5.00G      3
```

**Cutting the graph makes memory worse**, every time. The reason is
arithmetic, and the example shows it: the peak is
`stored + max_inter`, `max_inter` is a partition-independent *floor*, and every
cut keeps another boundary tensor alive. So the true optimum is always zero
cuts, and the planner is right to return that. It agrees with exhaustive search
on 5 of 5 budgets.

This is a limitation of the conservative model, stated as one:

- the floor is `input(graph) + largest single internal activation`;
- to go below it you must make a **segment** smaller, not cut the graph;
- memopt-style partitioning needs sub-segment structure that this `Segment`
  type cannot express.

### adv 06 — devices: CPU and GPU, same code

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/06_devices.py
```

`MEASURED` — exit 0, 4.6 s, 642 MB peak RSS. **With `CUDA_VISIBLE_DEVICES=`
set, so this is the CPU half only:**

```
  physical devices
    CPU  /physical_device:CPU:0

  No GPU visible. That is a supported configuration, not a
  degraded one -- everything below runs on CPU unchanged.

  training result
    loss          0.9045 -> 0.0393
    accuracy      0.9995
    wall clock    2.0s for 3 epochs on 2000 sequences
```

**The GPU row of the previous version of this file is not reproduced here, and
this file will not print it.** It read
`GPU: loss 0.8758 -> 0.0310, accuracy 0.9995, 2.4 s, peak 29.9 MiB`. Every
figure in this session was produced with the GPU hidden, so this file contains
no measurement of the GPU path at all. To get one, run it twice:

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/06_devices.py
CUDA_VISIBLE_DEVICES= /home/cune/.venvs/ax/bin/python -u examples/advanced/06_devices.py
```

The planning half is device-independent, and is the only half quoted here:

```
    peak           0.64 GiB
    credit horizon 66
    regime         activation
    None of that number touched TensorFlow.
```

The example also carries the clearest statement in the repository of why
`run_eagerly=True` is a deliberate cost:

> Our forward pass is a PYTHON loop over time. Keras 3 wraps the graph in
> `tf.function` and then reuses the result instead of recomputing it. Measured
> with a call counter: the forward ran 2 times when 4 were needed, and 0 times
> on a repeat fit. The gradient still applied — to STALE activations.

### adv 07 — diagnosing a silent spiking network

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/07_silence_diagnosis.py
```

`MEASURED` — exit 0, 38 s, 666 MB peak RSS. Chance accuracy 0.2500.

```
  CASE 1. THRESHOLD FAR TOO HIGH -- the classic failure
  threshold forced to 8.0 (nothing can reach it)
    firing/dead before  0.0000/1.00  0.0000/1.00
    firing/dead after   0.0000/1.00  0.0000/1.00
    loss 1.3863 -> 1.3859   accuracy 0.2610 (chance 0.2500)

  CASE 2. THE SAME NETWORK, CALIBRATED
  same weights, calibrate_model(target_rate=0.2)
    firing/dead before  0.0289/0.00  0.0058/0.20
    firing/dead after   0.1936/0.00  0.3199/0.00
    loss 0.5211 -> 0.0395   accuracy 0.9953 (chance 0.2500)

  CASE 3. THRESHOLD NEAR THE MIDDLE -- it just works
  threshold 0.5, no calibration
    firing/dead before  0.0656/0.00  0.0605/0.00
    firing/dead after   0.0656/0.00  0.0605/0.00
    loss 0.4217 -> 0.0067   accuracy 0.9997 (chance 0.2500)
```

**`+0.7343` accuracy from one calibration call.** No architecture change, no
hyperparameter change. This is the most useful thing in the example set for a
beginner, because "it does not learn" has a hundred causes and exactly one of
them is a broken gradient.

The reading table, which is the part worth stealing:

```
    firing < 0.01   the layer is silent. Call calibrate_model().
    dead  > 0.5     more than half the neurons never fire.
    0.01 - 0.10     sparse but alive; a slow start.
    0.10 - 0.40     the healthy band; the default target is 0.25.
    > 0.50          everything fires; the spike carries little information.
```

### adv 08 — reproducibility: measured, not assumed

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/08_reproducibility.py
```

`MEASURED` — exit 0, 1.9 s, 568 MB peak RSS.

```
  1. THE PLANNER
     three identical calls, bytes: 685942489, 685942489, 685942489
     bit-exact: True

  2. THE NUMPY CORE
     same seed twice, identical spikes: True
     different seed, different spikes: True
     spikes returned: (8, 4, 16), firing rate 0.2188

  3. A KERAS LAYER, BUILT TWICE
     two builds, identical initial weights: False
     weight difference: 0.932176

     ... with assigned weights, two runs give 0.00618631 and 0.00618631
     identical: True
```

`glorot_uniform` is **unseeded**: two builds of "the same" layer are two
different models, with an initial-weight difference of `0.932176`. Seeds do not
fix it, because in an A/B comparison you are comparing two *objects*, not two
runs of one object. Assign the weights.

And the advice that matters more than any of it:

> For a claim of the form "A scores X, B scores Y", run each arm 3 times and
> report the spread. A single run cannot tell you whether a gap is a result or
> a seed.

This file follows that rule: every quality figure above is one run, and is
labelled as one run.

### adv 09 — API tour: every name, called for real

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/09_api_tour.py
```

`MEASURED` — exit 0, 2.9 s, 542 MB peak RSS, `TensorFlow available: True`.

```
  BUDGETS
    consumer_6gb   Budget(consumer 6 GB: 6.0 GB VRAM, host 32.0 GB, ...)
    usable_bytes()  4.12 GiB usable of 6.00

  MODEL SPEC
    ax.Model()      Model(3 layers)
      .layers       ['SpikingRNN', 'SpikingRNN', 'Dense']
    SpikingRNN.spec(...)  16,512 connections, hidden=128, T=64
      ... then analyse()  peak 0.5205 GiB, regime activation, feasible=True

  PLANNING  -- no TensorFlow anywhere below this line
    Model.fit_budget       peak 0.571 GiB, credit horizon 66
    Model.explain_budget   3499 characters of human-readable report

    Measured spike densities, not a constant:
    SPARK_RATE_BAND        0.175 .. 0.324 across networks
    SPARK_RATE_NEURON      0.000 .. 0.830 within one network
    SPARK_RATE_DEFAULT     0.2500

  THE GATE  -- run this before you believe anything
      PASSED 12  |  FAILED 0
    gate passed: True
```

It also prints the seven `SpikingCell` hooks with their docstrings, and is
explicit about one thing worth knowing: `brute_force` is exported from
`axplan.planner` but deliberately **not** on the top-level surface, because
`2**M` makes it a test tool rather than a user tool.

### adv 10 — train once, serve anywhere

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/10_save_and_load.py
```

`MEASURED` — exit 0, 16.4 s, 639 MB peak RSS.

```
  1. TRAIN AND CALIBRATE
     firing after calibration 0.1938, dead 0.0000
     loss 0.4720 -> 0.0039, held-out 1.0000 (chance 0.2500)

  2. THE THRESHOLD IS PART OF THE MODEL, AND NOT IN THE WEIGHTS
     learned threshold, mean 0.115139, range 0.1151..0.1151
     What that costs, measured:
     predictions changed by resetting the threshold: 0.0000
     predictions identical after restoring it:        1.0000

  3. SAVE IN BOTH FORMATS
     save_weights -> model.weights.h5, 139,936 bytes
     save()         -> model.keras, 141,711 bytes

  4. RELOAD INTO A FRESH MODEL AND COMPARE
     outputs identical to the original: True
     prediction agreement: 1.000000
     accuracy after reload: 1.0000 (was 1.0000)
```

The lesson is a trap, not a workflow: **the threshold is not in the weights**,
so a naive rebuild from config starts at the default `1.0` instead of the
learned `0.1151`, and every prediction changes. Rebuild from config, then load
the weights.

### adv 11 — the seven hooks, and how to check yours

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/11_derivative_hooks.py
```

`MEASURED` — exit 0, 1.8 s, 568 MB peak RSS.

```
  1. spike_d1  --  d spike / d u
     hard 1[u>thr]:        sum 11.0000, d/du sum 0.0000
     smooth surrogate:    sum 14.4263, d/du sum 10.9589
     finite differences vs spike_d1: rel.err 6.29e-04 OK

  2. reset_dud  --  d reset / d u
     finite differences vs reset_dud: rel.err 4.28e-04 OK

  3. membrane_dstate  --  the hook that was missing
     default LIF: d membrane/d state = 43.2000 (that is just leak=0.9)
     finite differences agree to rel.err 2.90e-04

  5. WHAT A WRONG HOOK LOOKS LIKE
     the two cells give IDENTICAL spikes: True
     (spikes: 104 in both)
     ||g|| correct 0.012109, wrong 0.008438
     entries differing by more than 50%: 16/40, worst 91.2%
```

**The last block is the one to remember.** A wrong derivative hook produces
*identical spikes* and a gradient that is wrong on 16 of 40 entries, worst by
91.2%. Every forward check you might write passes.

### adv 12 — the six ways this breaks

```bash
/home/cune/.venvs/ax/bin/python -u examples/advanced/12_failure_modes.py
```

`MEASURED` — exit 0, 2.7 s, 597 MB peak RSS. Each failure is triggered for
real, on purpose.

```
  3. from_logits=False on a model that already outputs logits
     loss after 1 epoch: 9.8438

  4. no calibration, so the network never fires
     accuracy after 3 epochs: 0.2500 (chance 0.2500)

  5. credit_k larger than the horizon
     it did NOT fail, and there is no error message.

  6. overriding a forward hook and leaving the derivative
     it trained without any error. final loss 1.3199
```

> Two of these RAISE and three do not. The three that do not are the ones that
> cost days.

Two of the six are marked **FIXED** — `keras.Input(shape=...)` and the lost time
axis — and the example prints the exact historical exception text for each.

### adv 13 — when not to use this

```bash
python3 -u examples/advanced/13_when_not_to_use.py
```

`MEASURED` — exit 0, no TensorFlow.

```
  config                              BPTT      e-prop   ratio
  L=8 h=1024 T=32 b=8               8.1 MiB     32.0 MiB    0.3x
  L=12 h=1536 T=48 b=12            40.9 MiB    108.0 MiB    0.4x
  L=16 h=2048 T=64 b=16           129.3 MiB    256.0 MiB    0.5x
  L=24 h=3072 T=96 b=24           654.8 MiB    864.0 MiB    0.8x
  L=32 h=4096 T=128 b=32         2069.3 MiB   2048.0 MiB    1.0x
```

The same live table as `adv 03`, at five scales instead of three. And again
the prose underneath it contradicts the table right above it:

```
  The ratio peaks and then FALLS: 17x at T=32, under 9x at T=128.
```

There is no `17x` in the table above it.

**This example hardcodes its quality numbers.** `MEASURED` — it runs on bare
`python3` with no TensorFlow, yet prints, under the heading `MEASURED, same
task, same width, same budget (example 04)`:

```
        dense MLP                          0.9981
        spiking, readout flatten           0.9912
        spiking, readout mean              0.7444
```

A script with no TensorFlow cannot have measured those. The live run of
`examples/04_equal_budget.py` in this session prints `0.9988 / 0.9912 / 0.7438`
— the `0.9912` agrees exactly, the other two differ. **The repository contains
five different versions of this one measurement** and no single command
reproduces all of them. Reported to the owner; this file quotes only the live
run.

### adv 14 — one architecture, every budget

```bash
python3 -u examples/advanced/14_budget_matrix.py
```

`MEASURED` — exit 0, no TensorFlow. 2 spiking layers, `T=128`, 1000 classes,
batch 32.

```
  profile                 vram     usable   overhead
  consumer 6 GB             6G     4.12 GiB     1.88 GiB
  consumer 8 GB             8G     5.65 GiB     2.35 GiB
  pro 24 GB                24G    17.96 GiB     6.04 GiB
  pro 48 GB                48G    36.42 GiB    11.58 GiB
  datacenter 80 GB         80G    61.04 GiB    18.96 GiB
```

```
    width      peak  consumer_6_  consumer_8_    pro_24_GB    pro_48_GB  datacenter_
     1024     0.78G          yes          yes          yes          yes          yes
     2048     1.07G          yes          yes          yes          yes          yes
     4096     1.67G          yes          yes          yes          yes          yes
     8192     2.95G          yes          yes          yes          yes          yes
    12288     4.36G          yes          yes          yes          yes          yes
    16384     5.90G          yes          yes          yes          yes          yes
    20480     7.56G           NO          yes          yes          yes          yes
    24576     9.35G           NO           NO          yes          yes          yes
    32768    13.30G           NO           NO          yes          yes          yes
```

```
  profile                  usable   max width    headroom
  consumer 6 GB            4.12 GiB       16384     -1.79 GiB
  consumer 8 GB            5.65 GiB       20480     -1.91 GiB
  pro 24 GB               17.96 GiB       32768      4.66 GiB
  pro 48 GB               36.42 GiB       32768     23.12 GiB
  datacenter 80 GB        61.04 GiB       32768     47.74 GiB
```

Read the **headroom** column, not the width. The two 32 GiB-plus profiles both
report `32768`, which is the search ceiling, not a discovered maximum — the
tool says so rather than implying the network stops there.

### adv 15 — what should I build? (the inverse question)

```bash
python3 -u examples/advanced/15_what_to_build.py
```

`MEASURED` — exit 0, no TensorFlow, no GPU.

```
1. THE FORWARD QUESTION
  PEAK                 0.643 GiB
  tape share            97.3 %
  REGIME            activation
  credit horizon          66 steps

2. THE INVERSE QUESTION  -- CAPACITY OF A 6 GiB BUDGET
  LARGEST THAT FITS
  hidden 8192, layers 32, horizon 256, batch 1
  connections 2,147,745,792   bytes per synapse 2.00
  peak 5.604 GiB of 6.000 GiB   headroom 0.396 GiB
  binding term static

  FIRST THAT DOES NOT FIT
  hidden 32768, layers 4, horizon 1
  connections 4,295,098,368   bytes per synapse 2.00
  peak 8.509 GiB of 6.000 GiB   overshoot 1.42x
  binding term static
```

Three things a forward-only report cannot tell you:

- **the binding term** — `static` means the *weights* dominate, so the credit
  rule and the spike coding buy you nothing here and width is the only lever;
- **the failure boundary**, with its overshoot factor;
- **refusal** — with a budget too small for anything, the same script prints
  `solution: None`; and at the search ceiling it prints
  `reported overflow: None` rather than inventing a boundary.

And the honest part, which is the most useful sentence in the example:

```
4. THE SAME BUDGET UNDER A LOCAL CREDIT RULE
  capacity ratio: 1.00x
  1.00x, and that is the finding: at THIS budget the weights
  dominate (binding term = static), so the credit rule has
  nothing to save. It pays only where the tape dominates.
  A capacity tool that always showed a win for the local
  rule would be advertising, not measuring.

    local rule at T=256 keeps 64.7% of BPTT
```

Capacity and its price, on adjacent lines. That is the shape of an answer.

And it does both refusals on demand, which is the other thing a forward-only
report cannot do:

```
5. WHEN NOTHING FITS, IT SAYS SO
  solution: None
  smallest searched configuration that overflows: 128 wide, 1 layers, T=1, batch=1

6. THE SEARCH CEILING IS NOT A RESULT
  fit 2,147,745,792 synapses (hidden=8192)
  reported overflow: None
  The answer sits ON the width ceiling, so no boundary exists
  above it inside the searched space. None is invented.
```

---

## Why these numbers are not promises

`MEASURED` — every figure in this file is one run on one machine
(CPU-only, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3 / 2.5.0). The
framework itself says so, in `examples/02_budget.py`:

```
  These are run-specific figures, so they are not printed here as
  numbers: they change between runs. Run the scripts above.
```

The proof is in this session's own output. A previous version of this file
claimed `03_custom_neuron` reached `val_accuracy 0.9975` with `gradient norm
0.001069`; the live run gives `0.9925` and `0.000108`. **Both were wrong, and
one of them was wrong by an order of magnitude.** A figure nobody re-ran is a
guess with a decimal point in it.

---

## Defects this pass found in the examples themselves

`MEASURED` — every row was reproduced with a command. The **state** column is
the honest one: a defect still open is a reason to distrust a figure in this
file, and saying so is cheaper than a reader finding out.

| # | file | defect | state |
|---|---|---|---|
| A | `04_equal_budget.py` | was **not a gate suite** while a comment claimed every example was one | **fixed** — it is `EXAMPLE 04` in the table, with a `DECOMPOSITION BY CAUSE` marker |
| B | `02_horizon_sweep.py` | run with `need_tf=False`, so Part 2 never executed and `[ OK ]` certified half of what the suite is named after | **fixed** — it now runs on the discovered TensorFlow interpreter and requires `PART 2 -- MEASURED QUALITY` in its output |
| B | `02_budget.py`, `02_horizon_sweep.py` | hardcoded string `18 suites`; the gate ran 28, then 31 | **fixed** — the count is read out of `tests/run_all.py` at run time |
| B | `02_horizon_sweep.py` | embedded `explain()` report printed **truncated mid-sentence** — the gate keeps only lines matching `PASS`/`ok]`/`Error`/`Traceback`/`SKIPPED`, and the report is prose | **open, and it is a property of the gate's console filter, not of this file.** The example itself prints correctly; what the gate echoes is a subset |
| C | `axiomrnn.py` (not an example) | `explain()` prints by default *and* returns the report, so `print(spec.explain(...))` printed the whole 81-line report twice — 162 lines | **fixed** — pass `verbose=False` and print the return value once |
| D | `axplan/memory.py` | **`credit_rule="local"` was answered with BPTT arithmetic**, because the bare string `"local"` never satisfied an `is CreditRule.LOCAL` dispatch. The local rule reported `depends_on_T=True`, `quality_kept=1.0` | **fixed** — `credit_bytes()` normalises through the enums; reproduced against the published 0.1.1 wheel in `CHANGELOG.md` |
| E | `axplan/memory.py` | **the BPTT tape was capped at the credit horizon**, so `analyse()` returned an identical peak at `T=100` and `T=400`. Every `needed` figure above was an underestimate | **fixed** — the peaks in this file were re-measured after the fix |
| F | `13_when_not_to_use.py:54-56` | prints `0.9981 / 0.9912 / 0.7444` under a `MEASURED` heading, on bare `python3` with **no TensorFlow**. It cannot have measured them. | **open** — quoted numbers, not recomputed by that script |
| F | `05_t9.py:32, :185, :332` | same stale pair, `0.9919 vs 0.9981`, in a third file | **open** — same |
| — | `03_credit_rule.py`, `13_when_not_to_use.py` | both print a live table of `0.3x`…`1.0x` and then assert `17x -> 8.7x` / `17x at T=32` in prose two lines below it. **No command produces `17x`.** | **open** — the prose contradicts the table directly above it |

**D and E changed published numbers**, which is why the budget tables at the
top of this file are larger than in any earlier copy: width 12288 needs
13.84 GiB where it previously reported 4.38, and the wall on a 6 GB card moved
from width 20480 to 12288. An older copy of this file understates every peak.

None of these changes what the examples *conclude*. Each changes how much a
reader can trust a specific figure, which is the whole subject of this file.

---

## What these examples do not show

- **energy savings** — there is no neuromorphic device on this machine, and no
  energy figure here is measured by anyone;
- **superiority over dense networks** — `examples/04_equal_budget.py` measures
  parity, not a win, and this file will not describe it as one;
- **large-vocabulary quality** — the T9 corpus is a smoke test on 114,959
  words with a deliberately naive tokenisation;
- **faster execution** — not measured, and `run_eagerly=True` is a known,
  deliberate cost;
- **the GPU path** — every run here had the GPU hidden, so this file contains no
  GPU measurement at all;
- **graph partitioning that pays** — the planner is optimal and still answers
  "no cuts", which `adv 05` explains rather than hides.

The honest framing, repeated by the framework's own output: the spiking choice
costs nothing in quality here. It is worth it only on hardware where spikes are
cheap. On a normal GPU it is a curiosity, and the planner is the product.