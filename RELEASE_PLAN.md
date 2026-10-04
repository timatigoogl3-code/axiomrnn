# RELEASE PLAN — axiomrnn

Two artifacts, **independent timelines**. Coupling them would hold a library
release hostage to the integrity of its measuring instrument.

| Artifact | What it is | Tracks |
|---|---|---|
| **axiomrnn** | the product — a framework for neural units with real temporal dynamics | R0–R5 |
| **ClaimLedger** | the instrument — an append-only registry that makes a claim reproducible or refuted | L0–L2 |

ClaimLedger is how axiomrnn's headline numbers get verified. Its own hardening
does **not** gate the library release, and it must not be used as an excuse to
delay one.

---

## What "released" means here, falsifiably

Not "the code is finished". A release exists when all of these are true:

1. `pip install <artifact>` succeeds in a **clean virtualenv with no source tree
   on the path**, and the installed package imports and runs one real function.
2. A stranger with Python — **no TensorFlow knowledge, no framework knowledge** —
   gets a working neuron in under five minutes.
3. Every number in the README and `examples/README.md` was produced by a run,
   and the command that produced it is shown.
4. The test gate exits 0, and the output in the README carries the date it ran.
5. The limitations are stated as findings, not as fine print. **A reader who
   never runs the code can correctly predict where this framework beats a
   transformer.**
6. There are no stubs anywhere. No `TODO`, no `pass` standing in for work.

Any one of these failing means not released. There is no partial credit and no
"ship and fix in the next patch".

---

## Track R — the library

### R0 · TRUTH — *blocking, everything depends on it*

The README currently quotes `0.9988 / 0.9919 / 0.7525 / 0.7550` and calls itself
"Twelve examples" when there are 20 files; it reports 20 gate suites when the
gate runs 28. **Documentation that lies is worse than none, because it spends
the reader's trust and they cannot get it back.**

**Do:** re-measure every published number or mark it `FROM LITERATURE`. Delete
what cannot be substantiated. Nothing downstream may start while an
unsubstantiated number is still on the page.

**Exit gate:** zero un-tagged numbers in `README.md` and `examples/README.md`.
A number with no command beside it fails this gate.

**Owner:** Docs · **Blocks:** R1–R5

---

### R1 · INSTALLABLE — *currently at zero*

**Measured: there is no wheel.** Zero `.whl` files, no `dist/` directory. A
previous session reported a built wheel; it does not exist. **Nothing installs,
so adoption is currently impossible at rung 1** regardless of quality.

**Do:** build wheel and sdist. Verify in a clean venv. Confirm the wheel actually
contains `axcore`, `axon`, `axplan`, `axtf` — a wheel missing a subpackage
installs cleanly and fails later, which is the trap. Verify `axplan` works with
**no TensorFlow installed**, because the planning path must not hard-require TF.

**Exit gate:** clean venv → `pip install` → `import axiomrnn` → run one real
entry point, each step exit 0, commands and exit codes written down.

**The criterion is "every importable package ships" — not a fixed list of
directory names.** An earlier version of this gate named `axcore` explicitly
and was wrong: `axcore/` holds one file, `kernels.cpp`, and `pyproject.toml`
declares no `ext_modules` or `Extension`, so there is no compiled `axcore` to
ship. A wheel that *did* contain it would be shipping unbuilt source as though
it were a package. The real trap is a package that imports fine and is missing
a sibling that only fails on a code path the gate never reaches.

**MEASURED, 2026-10-04, verified by the Orchestrator and not by the builder:**
all four criteria pass against `axiomrnn-0.1.0-py3-none-any.whl` (58 298 B) and
`axiomrnn-0.1.0.tar.gz` (396 507 B).

| Check | Result |
|---|---|
| `pip install --no-index --no-deps` into a fresh venv | exit 0 |
| `import axiomrnn`, `import axplan` from `/tmp`, no source tree on path | exit 0 |
| TensorFlow in `sys.modules` at import time | `False` |
| `ax.Budget(vram_gb=6.0)` builds and plans | exit 0 |
| `axiomrnn-explain` console script from the installed venv | exit 0, prints a real budget |
| Shipped modules | `axiomrnn.py` shim; `axon/{__init__,gate,lif}`; `axplan/{__init__,credit,memory,planner,solve}`; `axtf/{__init__,build,cells}` |

The TensorFlow-free path is real, not aspirational: the shim imports `axplan`
at module level and `axtf` only behind a guard. `examples/02_budget.py` exits 0
in 0.11 s at 15 MB peak RSS with TensorFlow absent — measured by Docs.

**Open, and deliberately not closed:** `axcore/kernels.cpp` is referenced by
`tests/run_all.py`, `axtf/build.py` and `axiomrnn.py`, but nothing builds it
and no extension is declared. That is either dead code or an unfinished build
path, and the project rule is no stubs. It does not block the release — it is
not in the wheel — but it must not be left ambiguous.

**Owner:** Release · **Blocks:** R5

**R1 is now the *only* rung standing between this and a defensible release**,
and it is closer than this section implies. Measured 2026-10-04: the name is
claimed, 0.1.0 and 0.1.1 are uploaded, and the wheel is installable. What is
missing is a release whose *numbers are true*:

| rung | state |
|---|---|
| wheel exists and installs in a clean venv | done, 0.1.1 |
| every shipped subpackage present in the wheel | done — 18 files, all four packages |
| `axplan` works with no TensorFlow installed | done |
| gate exits 0 on a machine with TensorFlow | **not verified** — the run that produced the README block predates three fixes to `tests/run_all.py` and skipped one suite |
| published version free of known defects | **no** — two planner bugs, both fixed on `main`, neither in 0.1.1 |
| a 0.1.2 on PyPI | blocked on a human with the author's GitHub credentials |

---

### R2 · FIVE MINUTES

The user does not know TensorFlow or Keras, and the project requires both. So
TF is explained **in one paragraph at the point of first use**, not in a
footnote and not in a prerequisites list that gets skipped.

**Do:** write the shortest honest path from `pip install` to a firing neuron.
Say plainly what the parameters mean.

**Exit gate:** someone who did not write the code follows it and succeeds. If
the only person who can follow it is the author, it fails.

**Owner:** Docs · **Blocks:** R5

---

### R3 · PROOF

**Do:** run the full gate, every suite, every example included. Record the
real output with its date.

**Exit gate:** `python3 tests/run_all.py` exits 0, output pasted into the README
**with the date**, and no suite is missing. A gate that skipped eight suites
without saying so exits 0 and means nothing.

**Owner:** Orchestrator, verified by Auditor · **Blocks:** R5

---

### R4 · SCOPE — the part everyone skips

The research concluded things that are not flattering, and they belong on the
first page, not in a footnote:

- Neuron-style dynamics **do not** beat transformers in general. Measured
  crossover for rate coding: `n ≥ 2^(2^b − 1)`, i.e. `10^76.8` tokens at int8.
- They **do** win in a bounded regime — measured `22.73×` where a spiking unit's
  own dense state dominates.
- Energy and latency **diverge**: `11.90×` versus `5.94×` worse in one run. An
  exchange rate in a single currency is incomplete, and saying which one flatters
  the result is a choice that must be visible.
- The **strong** form of the rate-coding bound is not certifiable without a
  `TC⁰`-versus-`NC¹` separation. It must not be quietly assumed.

**Exit gate:** exit-5, read alone, tells a reader where this framework wins and
where it loses. If it has to be hedged into uselessness to be publishable, the
research was not honest enough to publish.

**Owner:** Science with Docs · **Blocks:** R5

---

### R5 · RELEASE

**Do:** version, changelog, license, authorship metadata, tag. Then publish —
**only after the decisions below are answered.**

**Exit gate:** installable from a clean machine by someone who has never seen
this repository.

---

## Track L — the instrument

### L0 · JOURNAL INTEGRITY — D001, the serious one

Measured, by direct attack, not reported:

```
Delete all 3 attempts of one record:  reproduced -> proposed
Truncate attempts.jsonl to zero bytes: all 5 -> proposed
ledger list          EXIT=0
ledger verify-chain  EXIT=0
```

`records.jsonl` has a hash chain and a head anchor. **`attempts.jsonl` has
neither.** The gate resists a forged verdict and does not resist an absent one.
It is bypassable by subtraction, and `verify-chain` currently certifies a lie.

**Exit gate:** deleting, truncating, middle-cutting or reformatting the journal is
each **detected with a non-zero exit and a first-broken index**.

**Owners:** Runner writes the chain, Core verifies it on read, neither edits the
other's path. **Auditor attacks the fix** rather than admiring it.

### L1 · CANONICAL — the contract that makes parallel work safe

Three bugs, one root cause: two components computed the same quantity differently
and gave both results the same name — `record_hash`, `want` vs `record.value`,
line-as-read vs canonical digest. The third is silent: any `json.dumps` rewrite
of `records.jsonl` detaches every attempt and nothing reports it.

**Exit gate:** contract ACKed by both sides on the file bus, `used_shell`
resolved, and `CANONICAL.md` amended through the bus rather than by editing the
file.

### L2 · THE LIMIT NO CODE CAN FIX

**A head anchor protects nothing while it lives inside the directory it anchors.**
Until that digest is pinned outside the ledger's control, ClaimLedger is
tamper-**evident** and not tamper-**proof**, and no test changes that.

This goes in the README as a stated limit. It must not be designed around, and no
green test may be allowed to imply otherwise.

---

## Decisions only the human can make

**1. Publish at all, and under which identity.**

**CORRECTION — this section previously named the wrong account.** It said
`timatigoogl3-code/axiomrnn` does not exist and that "the account has zero
followers and no other repo". Both statements were about an account that does
not belong to the author. The `gh` CLI in this environment authenticates as
`timatigoogl3-code`, which is a **third party**. Nothing was ever pushed, and
nothing may be pushed there: it is not the author's account to publish from.

The author's handle is `cubetitled-ui`. The author has a different token from
the one in this environment.

Settled so far:

| Decision | Value |
|---|---|
| Public author name | **Cubetitled** — in the LICENSE copyright line and in the PyPI `authors`/`maintainers` fields |
| Licence | **Apache-2.0** |
| GitHub account | **undecided** — the account in this environment is not the author's |

**2. The PyPI name — RESOLVED 2026-10-04, later than this section was written.**

The table below said `axiomrnn` was available. It no longer is, and the reason
matters: **this project claimed it.**

| Name | Status, measured 2026-10-04 |
|---|---|
| `axiomrnn` | **TAKEN BY US.** `https://pypi.org/pypi/axiomrnn/json` → 200, author `Cubetitled`, releases `0.1.0` (11:27:41Z) and `0.1.1` (11:54:53Z) |
| `axonrnn` | available (`/pypi/axonrnn/json` → 404) |
| `axoncortex` | not re-checked; irrelevant now |

So the gambling risk this section warned about did not materialize, and no
decision is pending. What *is* pending is a **0.1.2**, because 0.1.1 as
published contains two planner defects that return wrong numbers without
raising: `credit_rule="local"` was answered with BPTT arithmetic, and peak
memory was capped at the credit horizon so `T=100` and `T=400` cost the same.
Both are fixed on `main` and pinned by regression checks in
`tests/test_memory.py`. Full detail in `CHANGELOG.md`.

The remaining PyPI blocker is credential-based, not a decision: the upload
proceeds through Trusted Publishing from a GitHub release, and this
environment's `gh` is authenticated as `timatigoogl3-code`, a third party.
Nobody here can cut the tag or create the release that triggers it.

**3. License. RESOLVED — Apache-2.0**, chosen 2026-10-04 over GPLv3.

The reasoning is worth keeping, because it is a product decision and not a legal
one. GPLv3 was considered first and gives two real advantages: an explicit
patent grant, and derivatives that must stay open. It was rejected because every
major AI framework is permissive — PyTorch BSD-3, TensorFlow/JAX/Keras and
HuggingFace Transformers Apache-2.0, scikit-learn and NumPy BSD-3 — and a
copyleft AI framework is used by almost nobody, because no company can put one
in a proprietary stack. That would defeat the purpose, which is that other people
build models with this.

Apache-2.0 keeps the patent grant and the open-derivatives expectation without
the adoption cost.

**4. Ownership metadata. RESOLVED — `Cubetitled`**, in the LICENSE copyright line
and in the `authors` / `maintainers` fields of `pyproject.toml`. The two
attributions are kept in step deliberately: an artifact whose LICENSE and whose
package metadata disagree is worse than one with neither.

---

**5. `pyproject.toml` points at the wrong repository. RESOLVED 2026-10-04.**
All four `[project.urls]` read `https://github.com/timatigoogl3-code/axiomrnn`
while `git remote -v` says `https://github.com/cubetitled-ui/axiomrnn.git`.
`timatigoogl3-code` is the third-party account this environment's `gh` happens
to be authenticated as — the same mistake Decision 1 warns about, committed a
second time in the metadata rather than in a document. Corrected, and the
correction is verified against the wheel's `METADATA` rather than against
`pyproject.toml`, because a source file can look right and still not be what
shipped.

**6. `LICENSE` copyright name.** Confirm it says `Cubetitled` and not the GitHub
handle. Decision 4 covers this; the wheel carries
`License-Expression: Apache-2.0` and `License-File: LICENSE`, so the only thing
left to check by eye is the name inside the file.

---

## The honest strategy note

A 0.1.0 research library does not get adopted through downloads. It gets adopted
when its **finding is citable** and someone else builds on the measurement.

The strongest asset here is not the API. It is: *here is the measured exchange
rate at which neuron-style dynamics beat a transformer, here is the regime where
they win, and here is why the popular intuition that they should is wrong.* That
is a result. The framework is the instrument that produced it, and the instrument
is what makes the result reproducible by a third party.

So the release that matters most is not `pip install axiomrnn`. It is a
statement someone else can check. Both are in this plan; only the first is
usually thought about.
