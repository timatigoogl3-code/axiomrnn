# axiomrnn engineering rules

This file is not a manifesto and not a wish list. Every rule here **grew out
of a specific accident**, and the accident is written next to it. A rule
without a history is a slogan, and slogans get broken under deadline.

A rule counts as binding when code or a test references it.

## Revision note (2026-10-02)

Rewritten in English and moved to third person: the package documentation is
English, and this file contradicted its own no-first-person rule while the
rule it was supposed to enforce. Three substantive changes:

- **§7 amended**, not deleted. "No free knobs" was read as *use identical
  values*, which forbids calibrating each variant correctly. The rule now
  says match the *procedure*, not the values.
- **Stale numbers removed from rule bodies.** §2 quoted accuracy figures from
  one specific run. Rules are durable; run-specific numbers belong in
  `examples/README.md`, which is regenerated.
- **§16, §17, §18 added**, each from an accident that had no rule: a silent
  wrong gradient, an invalid verification method, and a documentation number
  nobody could regenerate.
- **§19 added** after the capacity solver shipped a recommendation and an
  "overflow" that contradicted each other.
- **§20 added** after a research harness was found to contain eight defects
  that all flattered the hypothesis it was built to test.

---

## 1. No silent exceptions

**Rule.** An exception may not be swallowed. Either it propagates, or the user
gets a warning that says **what to do about it**.

**Accident.** `compute_output_spec` contained `except Exception: return spec`.
Shape-parse failures never surfaced: the layer returned `(None, None, units)`
and the crash arrived two files away in `Dense`, saying "Shapes used to
initialize variables must be fully-defined" — which points nowhere near the
cause.

```python
# BAD: the shape error is eaten, and a different layer looks guilty
except Exception:
    return spec

# GOOD: it still works, but the user is told where to look
except Exception:
    warnings.warn(
        "SpikingRNNCell: output shape could not be inferred; the time axis "
        "may become None. If a Flatten/Dense below fails, the cause is here.",
        RuntimeWarning, stacklevel=2)
    return spec
```

**Check:** `tests/test_keras_build.py::test_output_shape_keeps_time_axis`.

---

## 2. The default is the measured best, not the habitual one

**Rule.** When a parameter has a measured correct value, that value becomes
the default. The old default is not deleted — it remains an explicit choice,
with its price stated.

**Accident.** Mean-over-time readout was the default. On a task with temporal
structure it cost a large accuracy gap. Changing it took months, because "it
is the convention in the SNN literature" and nobody measured the convention.

```python
_Readout(mode="flatten")   # default: measured best on temporal tasks
_Readout(mode="mean")      # deliberate choice: T leaves the input size
```

**Check:** `tests/test_keras_build.py::test_readout_modes_differ_and_default_is_flatten`.
Current numbers: `examples/README.md`, example 04.

---

## 3. Every acceptance criterion must be two-sided

**Rule.** Any threshold of the form "less than X" must have a lower bound. A
check that can go green under a hundredfold overestimate is worthless.

**Accident.** The first plan-versus-reality criterion was
`measured/planned <= 1.15`. A model that overestimates memory a hundredfold
produces a ratio of 0.000 and passes exactly like an ideal one. A broken
formula would have been certified as correct.

```python
ratio = measured / planned
ok = CRITERION_LO <= ratio <= CRITERION_HI     # both sides mandatory
```

---

## 4. Numbers in reports are computed, never typed

**Rule.** Any figure in report text that can change between runs is computed
from the current run's data.

**Accident.** An explanatory note carried hardcoded numbers from a previous
launch. Under a different seed the report lied while keeping the appearance of
a plausible table — the most dangerous form of dishonesty.

```python
# BAD: hardcoded, wrong at seed=7, but it looks ordinary
print("gradient contribution is small (0.047) vs readout (0.239)")

# GOOD: computed from this run
print(f"gradient {b_:.4f} vs readout {r_:.4f}")
```

---

## 5. Share and ratio are different quantities, and the confusion is easy

**Rule.** When output says "what percentage of the discrepancy does this
cause", compute the *share* (`read/(read+bit)`), not the *ratio*
(`read/bit`). Both are plausible and they answer different questions.

**Accident.** A report printed "the readout explains 17% of the discrepancy",
which was the ratio 0.255/0.015. The share was 94%. The figure understated the
cause sixfold while looking entirely reasonable.

---

## 6. Comparison fairness is verified, not assumed

**Rule.** When comparing two models, the code must **check** that they are
matched on parameter count, and say out loud when they are not.

**Accident.** Layer widths were not chained: each layer received the whole
network's input dimension. The second layer got the wrong shape, the variants
differed in parameter count, and different networks were being compared.

```python
sp_kinds = [k for k in ("spiking", "spiking_flat", "spiking_smooth") if k in res]
sizes = {k: res[k]["params_bytes"] for k in sp_kinds}
same = len(set(sizes.values())) == 1
print(f"HONESTY CHECK: {sizes} identical: {same}")
```

---

## 7. Match the procedure, not the values

**Rule.** Everything that is not the experimental variable must be handled
**with the same rigor**. Where the value is itself the object of study — a
threshold, a temperature, a firing rate — each variant gets its own tuned
value, and the tuning is reported.

**Accident.** The smoothed variant was shipped **without threshold
calibration**, on the grounds that it has no spike density. What was compared
was not two neurons but two different operating regimes.

**Why this wording replaced the old rule.** The original §7 read "never tune
the compared variants differently", which read as *use identical values* — and
that instruction would have forced skipping calibration for a variant that
needs it. Matching rigor while allowing per-variant tuning is both fairer and
the only consistent reading.

---

## 8. When a factor cannot be separated, say so

**Rule.** If one measurement changes two knobs at once, it cannot attribute
the result. Write that down; do not pick the convenient label.

**Accident.** The smoothed variant differed from the hard variant in **both**
the forward value and its derivative. The difference was labelled "the cost of
the surrogate gradient", which was wrong. Correct wording: "the joint
contribution of the bit and the gradient".

---

## 9. A shortfall is ours unless proven otherwise

**Rule.** If a discrepancy is caused by our own setup, that is stated
plainly, without converting it into a property of the method.

**Accident.** "Spikes are worse than dense by 0.27" turned out to be our
readout. While it stood as a property of spikes, it was a false statement
about physics.

> The breakdown gave 0.25. That was **our** bug, not a property of spikes.

---

## 10. Beyond the measured: "not measured", not "roughly"

**Rule.** The sparse/bit crossover has not been measured, by this project or
by anyone else. The point where spikes win on energy has not been measured.
Real-GPU speed has not been measured. All of these are written with the words
"not measured", never with an approximate figure.

**Why this is a rule and not modesty.** An invented `spike_rate = 0.08` was
low by a factor of 2.2–4.0, and calculations were built on it. One invented
number poisons everything resting on it.

---

## 11. The gate comes before every conclusion

**Rule.** A result that has not passed `tests/run_all.py` is not published.
The gate includes the examples: a user must be able to see that verification
is part of the work, not a formality.

**Accident.** The gate caught five real backward-pass errors that reasoning had
missed: a `(1−s)` factor on the wrong term (~30% error), an inverted time
recursion, a `dL/dthr` with no path through the reset, an early `return` in
the DP, and a wrong tuple field in the Pareto filter.

---

## 12. Report the error where it arose

**Rule.** An error message must name where the error arose, not where it was
noticed.

**Accident.** `--thr None` broke argparse, and the dense model then failed
inside oneDNN — making TensorFlow look guilty.

**Check:** flags that may be `None` are **not passed at all**, rather than
passed as the string `"None"`.

---

## 13. "Model input" ≠ "layer input"

**Rule.** When a layer needs to know **what it was fed**, use its own input
edge (`layer.input`), not `model.inputs`.

**Accident.** `calibrate_model` read `model.inputs[0]` for the first spiking
layer. Without an embedding stem that is genuinely the same tensor, and
everything worked. Once an embedding stem appeared (`tokens → Embedding →
Spiking`), `model.inputs` was still the **integer token tensor**, while the
layer received a rank-3 activation tensor. Calibration measured the wrong
tensor, and the crash surfaced in `tf.matmul` inside the cell — three frames
from the cause.

```python
# BAD: with an embedding stem, this is tokens, not activations
first = model.inputs[0]

# GOOD: each layer's own input edge is always what it receives
taps = [l.input for l in spiking]
```

**Check:** `tests/test_api_surface.py::test_axtf_api`,
`tests/test_keras_build.py::test_measure_gives_real_density`.

---

## 14. A test must not test itself

**Rule.** A test must call **the code it makes a claim about**. A check that
reimplements the formula next to the implementation proves nothing.

**Accident.** `test_norm_t_independence` never called the layer. It rewrote the
backward pass inside the test body and compared it against the same code from
which `1/T` had been removed. The conclusion "normalisation works" was an
artifact of arithmetic inside the test.

`norm_t` was a **dead parameter** the entire time: accepted, threaded through
four signatures, and **never applied**. Nobody noticed, because the test never
touched it.

It was found only by trying to **break the code** and checking that the test
would notice. The first attempt broke nothing: the edit did not apply, the
signature differed, and the test went green on unmodified code.

What followed:

1. the sabotage was applied against the actual source text (`_lif_backward`),
   not from memory;
2. the edit was verified to have landed **before** the test was run;
3. the test was rewritten to call the layer on identical weights
   (`layer.kernel.assign(W)` — `glorot_uniform` has no seed, so two instances
   are otherwise incomparable);
4. the test then failed, showing a `0.44x` growth in both branches, which
   confirmed the diagnosis;
5. after the fix the ratio was exactly `1/T` (0.0625 at T=16).

```python
# BAD: the formula lives in the test, the implementation sits next to it
def grad_norm(T, norm_t):
    base = 0.1 / T if norm_t else 0.1        # checking its own arithmetic

# GOOD: call the layer and compare two of its own branches
layer = SpikingRNNCell(UNITS, cell=SpikingCell(leak, alpha),
                       return_sequences=True, norm_t=norm_t,
                       horizon_hint=T)
layer.build((None, T, DIN))
layer.kernel.assign(W)                        # assign: unseeded initialiser
```

Two consequences:

- **an unseeded initialiser** (`glorot_uniform`) makes runs incomparable. In
  tests, weights are assigned explicitly;
- **break-it-and-look** is mandatory before a test is declared working. A test
  that passes on sabotaged code is worse than no test: it supplies false
  confidence.

**Check:** `tests/test_tf_cells.py::test_norm_t_independence`.

---

## 15. The planner must be able to say "useless"

**Rule.** If the optimum of a problem is "do nothing", the tool must return
that and explain it, rather than performing work.

**Accident.** `plan_partition`, in its conservative model, returns **zero cuts**
almost always. The peak is

```
peak = stored + max_inter
stored    = sum of inputs at the chosen boundaries   (grows with cuts)
max_inter = the largest interior activation          (independent of the split)
```

`max_inter` is a **floor** that does not move. `stored` only grows. Therefore
splitting the graph can only **increase** the peak, and the optimum is
always "no cuts".

Several attempts were spent looking for data on which cuts would pay off. None
worked, because this is not a property of the data but a consequence of the
model. The correct move is to say so plainly.

**What that implies.** To get below the floor, make a segment **smaller** — do
not cut the graph. The planner answers "which segments to recompute", and the
honest answer for a flat segment list is "none".

This is not an implementation defect: the DP is optimal, verified against brute
force. The optimum really is zero.

**Check:** `examples/advanced/05_segmentation_planner.py` prints both sides —
why cuts make it worse, and why tuning does not fix it.

---

## 16. If the core cannot know a derivative, it must ask

**Rule.** When the backward pass needs a quantity the kernel cannot compute
from its own assumptions, the kernel **asks the object** that owns the
dynamics. It never assumes a value. An override point exists even when the
default happens to be correct today.

**Accident.** The LIF backward pass read `cell.leak` as `du/dstate` — the leak
coefficient standing in for the membrane's state derivative. For a linear
leak that is accidentally right. For **any nonlinear membrane it is silently
wrong**, and there was no way for a user's cell to correct it short of forking
the kernel.

The fix added `SpikingCell.membrane_dstate(state, drive_next)`, and the kernel
now calls it. Its arguments are the post-reset state and the next step's
drive, which is what the derivative is actually taken with respect to.

**Why "even when the default is correct".** The old default was correct only
for the one membrane this project happened to ship. A hook that is needed
solely when the default is wrong will be missing on every new cell, because
nobody discovers a missing hook while the default happens to work.

**Check:** `tests/test_tf_cells.py::test_membrane_dstate_hook`,
demonstrated in `examples/advanced/11_derivative_hooks.py`.

---

## 17. Finite differences are invalid across a discontinuity

**Rule.** Finite differences cannot verify a gradient whose forward pass is
discontinuous — a hard spike, a step, a `max`. Verify such a gradient by
comparing it against a **reference implementation of the same quantity**,
element by element, and separately confirm the hook is actually being called.
Never by a scalar relative error on the layer.

**Accident.** A first version of the membrane-derivative check ran finite
differences on the layer and reported `rel.err = 0.98` **on correct code**. The
hard spike is a step function, so the numerical derivative is meaningless
across the threshold crossing. A wrong test nearly deleted a correct
implementation.

The same trap was hit a second time with `reset_dud`: the faulty check
produced `rel.err = 0.82` there.

The replacement verifies three separate things:

1. the **membrane** function alone against finite differences — legal, because
   the membrane is smooth;
2. a **call counter** — proof the hook is invoked, not merely defined;
3. **element-wise** comparison of the gradients, so a wrong hook cannot hide
   behind a good average.

**Check:** `tests/test_tf_cells.py::test_membrane_dstate_hook`. Sabotaging the
hook back to `dstate = cell.leak` makes this test fail; that direction was
tested, not assumed.

---

## 18. A number in the documentation must be reproducible by a committed script

**Rule.** Any figure in `README.md`, `examples/README.md` or `CHANGELOG.md`
that is not a constant of the mathematics must be the output of a script in
this repository. If no committed command prints it, it does not belong in the
documentation.

**Accident.** Both the top-level `README.md` and `examples/README.md` carried
accuracy figures from earlier runs, disagreeing with each other and with the
current code. The example index also claimed a count of examples that the
directory did not contain, so a reader following the list hit files that were
not there and missed files that were.

**Check:** `examples/README.md` measured blocks come from actually running the
named scripts; `tests/run_all.py` output is pasted into the top-level
`README.md`.

---

## 19. Two numbers from one tool must not contradict each other

**Rule.** When a tool reports a recommendation *and* a limit, the limit must
be comparable to the recommendation on the same axis. If the two cannot be
ordered against each other, the tool must not present them as a pair.

**Accident.** The capacity solver in `axplan/solve.py` reported, for a 6 GiB
budget: "largest that fits — 2 147 745 792 synapses" and, immediately below,
"first that does not fit — 1 073 872 896 synapses". The overflow was smaller
than the recommendation, which is incoherent on its face.

The cause was selecting the boundary independently: the solver took the
*smallest* overflow anywhere in the searched space, while the recommendation
was the *largest* fitting point. Those come from different axes — a
`batch=32, T=64` corner overflowing says nothing about the boundary of a
`batch=1` solution twice its size.

The fix filters candidates against the recommendation: a boundary must be
strictly larger than the answer it bounds.

**Second half of the same rule.** When the recommendation lands exactly on the
search ceiling, no boundary exists inside the searched space. The tool must
say so and name the ceiling, rather than inventing a limit elsewhere in the
space.

**Check:** `tests/test_solve.py::test_failure_boundary_is_real` and
`::test_no_boundary_invented_at_the_ceiling`. The sabotaged version of the
filter was applied to confirm the test fails without it, in line with §14.

---

## 20. Agreement is the claim that needs proof, not failure

**Rule.** Verification effort is **asymmetric**. A result you expected to fail
is self-verifying: if the code is broken, it fails. A result you expected to
pass needs adversarial checking before it is believed. Concretely: before
believing that a check agrees, prove the check *can* disagree — and prove that
on the code you actually care about, not on a stripped-down variant of it.

Two corollaries that have both caught real errors here:

- a measurement that returns an identical value everywhere has not measured
  anything, and must not print a precision claim beneath itself;
- a defect tends to point the way the author was hoping. Audit for
  defects that would *create* the expected result before auditing for ones that
  would break it.

**Accident 1 — the framework's own.** The `norm_t` test could not fail by
construction (§14), and the membrane-derivative check reported `rel.err = 0.98`
on **correct** code because finite differences cannot see a discontinuity
(§17). In both cases the check was green-or-red for a reason unrelated to the
property it claimed to test.

**Accident 2 — the wider research programme.** A harness built to test whether
noise helps inference contained **eight defects that flattered that
hypothesis**. One removed the element whose *value* equalled a random index
rather than the element *at* that index, so the working set grew without bound
and reported a figure of `40.0` against a true optimum of `6.6505`. Another
made the "exact optimum" enumerator take ~1.8 days per instance, so **no exact
optimum had ever been computed** and the previous result was not reproducible
at all. A third assumed the objective matrix was symmetric; it was not, which
crippled the *baseline* and flattered the method under test.

**What the author did with this, and why it is the standard here.** The entire
suite was rewritten. Three of the report's own prior claims were falsified by
its own data and corrected in place rather than deleted. Nine of ten
pre-registered kill criteria fired. The negative result that came out is
trustworthy *because* the bias was found first — a bias found after a positive
result is usually rationalised away.

**Check:** this rule is verified by the same mechanism as §14 — sabotage the
implementation in the direction that favours the claim, and confirm the check
fails. `tests/test_solve.py` was validated that way (the boundary filter was
removed and the test failed), and so was the A06 closed-form result (re-run,
byte-identical output, no discrepancy).

---

## 21. A test must exercise the code that ships

`tests/run_all.py` ran its memory-model suite as `../verify/test_cost.py`. That
file is outside the repository, absent from the wheel and the sdist, and began
with `sys.path.insert(0, '/home/cune')` before importing `axon.cost` from
`/home/cune/axon/cost.py`. So it printed 38 `[ok]` lines about a module no
installation contains, and on a fresh clone — including every GitHub runner —
it could not run at all.

It also tested a **stale predecessor** of the shipped `axplan/memory.py`. That
is the part that cost something: the suite meant to catch the two planner
defects in §22 and §23 was checking a dead copy of the arithmetic, so it could
not have caught them.

**Check:** `tests/test_memory.py` imports `axplan.memory` — the module the
wheel ships — and the gate's `MEMORY MODEL` entry names that file. §14 applies:
the two defects were reproduced against the published 0.1.1 wheel and pinned as
regressions, so a reintroduction turns the suite red rather than merely
undocumented.

---

## 22. A value compared by identity is not a value

`CreditRule` is a `str`-Enum. `credit_bytes()` dispatched with
`if rule is CreditRule.LOCAL`, and `axplan/memory.py` passed the bare string
`"local"`. The string *equals* the member — `CreditRule.LOCAL == "local"` is
`True` — but is never *identical* to it, so the test failed and execution fell
through to the BPTT branch.

The failure was silent and confidently wrong. `depends_on_T` came back `True`
for a rule documented as T-independent, `quality_kept` came back `1.0` where
the documentation says `0.389` at `T=500`, and `analyse()` charged BPTT bytes
for a local-rule request. Every number was plausible; none was the number
asked for. `releases 0.1.1` shipped this.

**Check:** `credit_bytes()` normalises `rule` and `neuron` through their enums
before anything else, so a string and its member are the same request and an
unknown value raises `ValueError` instead of becoming BPTT.
`tests/test_memory.py` T7 asserts that `"local"` and `CreditRule.LOCAL` take the
same branch, and that an unknown rule is refused rather than answered.

---

## 23. A cap that is not a decision is an undercount

`backward_bytes()` truncated the BPTT tape at `CreditModel.horizon()` — the
number of steps over which credit decays below `eps`. Measured against the
published wheel:

```
T=100 : 294,801,408 B
T=400 : 294,801,408 B     identical
```

Four times the horizon charged at the same price, which contradicts the
`credit_rule` field's own `Theta(T*A)` and made every published "needed GiB"
figure an underestimate. Truncated BPTT is a legitimate technique — `credit.py`
already exposes it as `credit_k` — but applying it silently by default is a
planner under-reporting, and the difference between the two is exactly the kind
of thing a reader cannot detect.

**Check:** BPTT is `Theta(T*A)` again, so `T=400` costs exactly 4x `T=100`, and
`tests/test_memory.py` asserts both the ratio and that the tape is *not* capped
at the credit horizon. The credit horizon still sets `MemoryPlan.credit_horizon`,
which is a statement about credit rather than about tape length.

---

## 24. A published claim must survive its own first reader

Two things in this repository were true in the source and false in the artifact:

- every `[project.urls]` entry read `github.com/timatigoogl3-code/axiomrnn`,
  a third party's account, while `git remote` said `cubetitled-ui`. Nothing in
  the gate compared a string to a remote, because every other suite calls
  functions and checks behaviour;
- the wheel advertised `Programming Language :: Python :: 3.14` while
  `requires-python` capped it at `<3.14`. The classifier is what a reader sees
  before installing.

**Check:** `tests/test_api_surface.py` reads `pyproject.toml`, resolves the
expected repository from `git remote get-url origin`, and asserts the URLs, the
LICENSE attribution, and that every Python classifier falls inside
`requires-python`. Verified by sabotage: reinstating the old URLs and the 3.14
classifier turns 9 checks red.

---

## Summary: what makes this framework trustworthy to a user

| Property | How it is enforced |
|---|---|
| Never fails silently | §1 warning instead of `except` |
| Works out of the box | §2 default = measured best |
| Checks cannot be fooled | §3 two-sided criteria, §17 element-wise verification |
| Reports can be trusted | §4, §5 computed numbers, shares not confused with ratios |
| Comparisons are fair | §6 matched parameter counts, §7 matched procedure |
| Honest about itself | §8, §9, §10 "not measured" instead of plausible fiction |
| Verifiable | §11 gate before any conclusion, §18 documentation numbers are generated |
| Testable without trusting the test | §14 break-it-and-look, §16 the kernel asks the cell |
| Honest when a tool is useless | §15 the planner says "no cuts" |
| Self-consistent | §19 a limit and its recommendation are on the same axis |
| Agreement is earned | §20 prove the check can disagree, before believing that it agreed |
| Tests the shipped code | §21 the suite imports the module the wheel ships |
| Identity is not equality | §22 normalise through the enum before comparing |
| No silent caps | §23 a truncation that is not a decision is an undercount |
| The artifact is checked too | §24 metadata is part of what you ship |