#!/usr/bin/env python3
"""
ALL FRAMEWORK TESTS, ONE COMMAND.

    python3 axiomrnn/tests/run_all.py

HARD RULE: no result is published until this is green. That rule exists
because three experiments in a row looked convincing while being wrong,
and only this check caught them.

Suites that NEED TensorFlow are marked and skipped with an explicit
notice -- skipping silently is not allowed.
"""
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PY = sys.executable


def find_tensorflow_python():
    """Return an interpreter that can actually import tensorflow, or None.

    This used to be a hardcoded path to one machine's virtualenv. On any other
    machine -- including every GitHub runner -- that path does not exist, so
    every TensorFlow suite was reported SKIPPED and the gate still came back
    green. A check that cannot run was indistinguishable from a check that
    passed, which is the one failure mode this gate exists to prevent.

    So the interpreter is discovered, and a candidate is only accepted once it
    has been seen to import tensorflow. Order matters: an explicit override
    first, then the interpreter running the gate, then the usual virtualenv
    locations, then the per-user ones.

    Returns (path_or_None, places_looked_in). The places matter as much as the
    answer: "no TensorFlow" and "I did not know where to look" are different
    findings, and reporting only the first is how a machine with a working
    TensorFlow install ends up reported as having none.
    """
    import glob
    import subprocess

    home = os.path.expanduser("~")
    candidates = [
        # Explicit override. The only way to name an interpreter this function
        # cannot guess, e.g. a conda env or one created by a colleague.
        os.environ.get("AX_TF_PY"),
        sys.executable,
        os.environ.get("VIRTUAL_ENV") and os.path.join(
            os.environ["VIRTUAL_ENV"], "bin", "python"),
        os.path.join(ROOT, ".venv", "bin", "python"),
        os.path.join(ROOT, "venv", "bin", "python"),
        shutil.which("python3") or "python3",
    ]
    # Per-user venv collections. Sorted so the printed answer is stable and a
    # reader can tell which one was chosen.
    candidates += sorted(glob.glob(os.path.join(home, ".venvs", "*", "bin",
                                               "python")))
    # Keep order, drop repeats: two candidates can resolve to one interpreter.
    seen, ordered = set(), []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    for cand in ordered:
        if not (os.path.isfile(cand) and os.access(cand, os.X_OK)):
            continue
        try:
            done = subprocess.run(
                [cand, "-c", "import tensorflow"],
                capture_output=True, timeout=180, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return cand, ordered
    return None, ordered


TF_PY, TF_CANDIDATES = find_tensorflow_python()

SUITES = [
    ("GATE (numpy core, gradient proven)", ["python3", "-W", "ignore",
                                              "axon/gate.py"], False),
    ("PLANNER (DP optimal vs brute force)", ["python3", "-W", "ignore",
                                                   "tests/test_planner.py"], False),
    # Was `../verify/test_cost.py` -- a file outside this repository, absent
    # from the wheel and the sdist, which began with `sys.path.insert(0,
    # '/home/cune')` and imported `axon.cost` from /home/cune/axon/cost.py.
    # Its green checks therefore certified a module no installation contains,
    # and on a fresh clone, including every GitHub runner, the suite could not
    # run at all. It also tested a stale predecessor of axplan/memory.py, so
    # it missed both credit-rule defects now pinned in tests/test_memory.py.
    #
    # The marker is the suite's own summary line. A count in the label would
    # be a claim about a file that can change without this table changing,
    # which is how "31 tests" outlived the 38 lines the suite printed.
    ("MEMORY MODEL (arithmetic, regimes, codec, credit rules)",
     ["python3", "-W", "ignore", "tests/test_memory.py"], False,
     ["ALL 101 CHECKS PASSED"]),
    ("CREDIT RULES (e-prop vs BPTT, quality price)",
     ["python3", "-W", "ignore", "tests/test_credit.py"], False),
    ("TF CORE (gradient, normalisation, custom cell)",
     [TF_PY, "-W", "ignore", "tests/test_tf_cells.py"], True),
    ("KERAS ASSEMBLY + training via keras.fit",
     [TF_PY, "-W", "ignore", "tests/test_keras_build.py"], True),
    # API audit. It exists because add_spiking(cell=...) was broken and
    # nobody noticed: nobody ever called the function.
    ("API AUDIT (every advertised function called)",
     [TF_PY, "-W", "ignore", "tests/test_api_surface.py"], True),
    ("C++ KERNELS (bit-packing, layout)",
     ["/bin/sh", "-c",
      ("cd axcore && g++ -O3 -march=native -std=c++20 -include cstring "
       "kernels.cpp -o /tmp/axkernels 2>/dev/null && /tmp/axkernels")], False),
    # Examples inside the tests are a necessity, not belt-and-braces:
    # 01_minimal had already drifted from reality (it failed on import), and
    # the tropa files drifted so far from the core that they became
    # untrustworthy. An example nobody runs lies.
    ("EXAMPLE 02 (budget, NO TensorFlow)",
     ["python3", "-W", "ignore", "examples/02_budget.py"], False,
     ["the value is in the credit planner"]),
    ("EXAMPLE 01 (train a network)",
     [TF_PY, "-W", "ignore", "examples/01_minimal.py"], True,
     # The two numbers this example exists to produce. Without them the run
     # could have trained nothing at all and still exited 0.
     ["held-out accuracy:", "chance:"]),
    ("EXAMPLE 03 (custom neuron, no fork)",
     [TF_PY, "-W", "ignore", "examples/03_custom_neuron.py"], True,
     ["membrane_dstate", "cannot guess your maths"]),
    # Was absent from this table entirely, while a comment in this file claimed
    # every example was a suite. It imports tensorflow at module level, so it
    # needs the TF interpreter. An example nobody runs lies -- and this one had
    # been lying since it was written.
    # AX_FAST=1 shrinks the training, not the code path. Without it this suite is
    # four model kinds times a threshold sweep at 6000 steps per epoch, which is
    # a measurement rather than a check; it ran for eight minutes inside the
    # gate and was still going. Run the file yourself, without the variable, to
    # get the real numbers.
    ("EXAMPLE 04 (equal budget)",
     ["/usr/bin/env", "AX_FAST=1", TF_PY, "-W", "ignore",
      "examples/04_equal_budget.py"], True,
     ["DECOMPOSITION BY CAUSE"]),
    # Trains the same model on CPU and on GPU and prints the comparison. The
    # markers are the two devices being reached, because a file that silently
    # fell back to CPU on both passes would otherwise look like a green GPU
    # result. On a machine with no GPU it prints that it skipped, which is also
    # acceptable: the honest "not available" is the point.
    ("EXAMPLE 06 (same model on CPU and on GPU)",
     ["/usr/bin/env", "AX_FAST=1", TF_PY, "-W", "ignore",
      "examples/06_gpu_vs_cpu.py"], True,
     ["TRAINING ON CPU", "COMPARISON"]),
    # Generation, not classification: the model's own output is fed back to it one
    # token at a time. The markers are the accuracy line and the generation
    # section, because a run that trained and then produced no text would
    # otherwise pass on exit code alone.
    #
    # AX_FAST keeps the whole pipeline -- train, sample at three temperatures,
    # score against a unigram baseline -- and only shortens training. The
    # bit-per-word figure it prints is not the one to quote; run the file
    # yourself for that.
    ("EXAMPLE 07 (train, then generate text)",
     ["/usr/bin/env", "AX_FAST=1", TF_PY, "-W", "ignore",
      "examples/07_generate.py"], True,
     ["test accuracy", "unigram bits/word", "GENERATION"]),
    # T9 trains a real language model on 115k words. It is the slowest
    # suite here, so it runs last; if it times out the rest already ran.
    ("EXAMPLE 05 (T9 next-word prediction)",
     [TF_PY, "-W", "ignore", "examples/05_t9.py"], True),
    # Advanced set. The planning-only ones run on python3 with no TF,
    # which also exercises the axplan/axtf boundary from the other side.
    ("ADV 01 (embedding + stacked layers)",
     [TF_PY, "-W", "ignore", "examples/advanced/01_embedding_and_layers.py"],
     True),
    # ADV 02 carries a "Part 2 -- MEASURED QUALITY (needs TensorFlow)" section
    # guarded by its own try/except. It was registered need_tf=False, so the
    # gate ran it on plain python3, the guard fired, and the suite reported
    # [ OK ] having skipped half of itself. A suite that quietly drops a
    # section and still passes green is worse than one that fails.
    #
    # It was then registered need_tf=True while still naming "python3" as its
    # interpreter, which cannot import tensorflow on this machine -- the
    # system python3 is 3.14 and TensorFlow has no 3.14 wheel. So it skipped
    # every run. It now names the discovered interpreter, and the marker
    # asserts that the quality section actually printed.
    ("ADV 02 (horizon sweep, both parts)", [TF_PY, "-W", "ignore",
                                            "examples/advanced/02_horizon_sweep.py"],
     True, ["PART 2 -- MEASURED QUALITY"]),
    ("ADV 03 (credit rule, the real lever)",
     ["python3", "-W", "ignore",
      "examples/advanced/03_credit_rule.py"], False,
     ["We do NOT claim e-prop is better"]),
    ("ADV 04 (budget, codec, precision)",
     ["python3", "-W", "ignore",
      "examples/advanced/04_budget_and_precision.py"], False,
     ["A planner that always returns a plan is lying somewhere"]),
    ("ADV 05 (segmentation planner)",
     ["python3", "-W", "ignore",
      "examples/advanced/05_segmentation_planner.py"], False,
     ["this planner answers \"which segments do I recompute\""]),
    ("ADV 06 (devices: CPU and GPU)",
     [TF_PY, "-W", "ignore", "examples/advanced/06_devices.py"], True),
    ("CAPACITY SOLVER (axplan.solve)",
     ["python3", "-W", "ignore", "tests/test_solve.py"], False),
    ("ADV 15 (what to build: the inverse question)",
     ["python3", "-W", "ignore",
      "examples/advanced/15_what_to_build.py"], False,
     ["None is invented"]),
    # ADV 07-14 were outside this gate until 2026-10-03, and they rotted:
    # 08_reproducibility.py called axon.lif.lif_forward, a function that has
    # never existed, and nothing noticed for the lifetime of the file. An
    # example nobody runs lies -- so every example is now a suite.
    ("ADV 07 (silence diagnosis)",
     [TF_PY, "-W", "ignore",
      "examples/advanced/07_silence_diagnosis.py"], True),
    ("ADV 08 (reproducibility)",
     [TF_PY, "-W", "ignore",
      "examples/advanced/08_reproducibility.py"], True),
    ("ADV 09 (API tour)",
     [TF_PY, "-W", "ignore", "examples/advanced/09_api_tour.py"], True),
    ("ADV 10 (save and load)",
     [TF_PY, "-W", "ignore", "examples/advanced/10_save_and_load.py"], True),
    ("ADV 11 (derivative hooks)",
     [TF_PY, "-W", "ignore",
      "examples/advanced/11_derivative_hooks.py"], True),
    ("ADV 12 (failure modes)",
     [TF_PY, "-W", "ignore",
      "examples/advanced/12_failure_modes.py"], True),
    ("ADV 13 (when not to use)",
     ["python3", "-W", "ignore",
      "examples/advanced/13_when_not_to_use.py"], False,
     ["Measure the crossover yourself"]),
    ("ADV 14 (budget matrix)",
     ["python3", "-W", "ignore",
      "examples/advanced/14_budget_matrix.py"], False,
     ["depends on whether your problem can"]),
    # ADV 15 sits above with the planning examples; the TF set 07-12 follows
    # it so a TF-less machine reports SKIP on one contiguous block instead of
    # on eight scattered lines.
]


def run(name, argv, need_tf, expect=None):
    """Run one suite and decide whether it passed.

    `expect` is a list of substrings that must appear in the suite's output.

    It exists because "exited 0" is not a result. An example that computes the
    wrong answer and returns cleanly used to earn a green [ OK ] here, because
    the only verdict was the process exit code, and the console filter then hid
    whatever it had actually printed. Nineteen of twenty examples were in that
    position. A suite with no `expect` still runs and still passes on exit 0,
    but it now says so in its own line, so a reader can tell which green means
    "checked" and which means only "did not crash".
    """
    expect = expect or []
    print("\n" + "=" * 74)
    print(name)
    print("=" * 74)
    exe = argv[0]
    # Is the interpreter this suite names actually runnable here? Asked
    # separately from the TensorFlow question, because conflating them is how
    # ADV 02 came to skip on every machine: a bare command name like
    # "python3" is resolved through PATH, so os.path.exists() on it is always
    # False, and that answer was being read as "no TensorFlow".
    resolved = exe if (exe and os.path.sep in exe) else shutil.which(exe or "")
    if resolved is None:
        print(f"  SKIPPED: {exe!r} is not a runnable interpreter on this "
              f"machine.")
        print("  This is NOT a pass.")
        return "SKIP"
    if need_tf and TF_PY is None:
        # Wording matters more than usual here. The previous version printed
        # "Only this suite needs TensorFlow", which is true of one suite and
        # false of the twenty that follow: on a machine with no TensorFlow the
        # gate skipped most of itself and still came back green. It now says how
        # many suites are being skipped, because a skipped suite is a check that
        # did not run.
        print(f"  SKIPPED: no interpreter with TensorFlow was found (tried "
              f"{len(TF_CANDIDATES)} interpreter"
              f"{'s' if len(TF_CANDIDATES) != 1 else ''})")
        print("  This is NOT a pass. The suites needing TensorFlow did not run, and")
        print("  the gate below will say how many. Install it with:")
        print("      pip install -e '.[dev]'")
        return "SKIP"
    t0 = time.time()
    # The suite gets its own session so the gate owns its whole process tree.
    # A suite that hangs, or that is abandoned when the gate itself is
    # interrupted, must not outlive it: 200-600 MB of TensorFlow per orphan is
    # what pushed this machine into 8 GB of swap and froze it.
    proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=2400)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        reap_group(proc.pid)
        out, _ = proc.communicate()
        code = "TIMEOUT"
    dt = time.time() - t0
    # print only what matters
    for ln in out.splitlines():
        skip = ("oneDNN", "cpu_feature", "absl::", "TF-TRT",
                "To enable the following", "port.cc", "external/local")
        # Filter keywords cover BOTH scripts' output: some suites still
        # print Russian summary lines, and dropping them would hide a
        # failure. Both wordings are matched deliberately.
        if any(k in ln for k in ("PASS", "FAIL", "PASSED", "FAILED",
                                 "PASSED", "FAILED",
                                 "ok]", "[ok]", "ALL TESTS", "Error",
                                 "error", "Traceback", "SKIPPED")) \
                and not any(k in ln for k in skip):
            print("  " + ln.strip())
    print(f"  -> code {code}, {dt:.1f}s")
    if code != 0:
        return "FAIL"

    # Evidence check. Reaching here means only "it did not crash"; the strings
    # below are what turn that into "it did the right thing".
    missing = [e for e in expect if e not in out]
    if missing:
        print(f"  -> EVIDENCE MISSING ({len(missing)} of {len(expect)}):")
        for m in missing:
            print(f"       expected in output: {m!r}")
        print("     The suite exited 0 but did not print what it claims to show.")
        return "FAIL"
    if expect:
        print(f"  -> evidence verified: {len(expect)}/{len(expect)} markers")
    else:
        print("  -> no evidence markers declared: this green means only "
              "'exited 0', not 'correct'")
    return "OK"


def reap_group(pid):
    """Kill every process in a suite's session. Never raises.

    Killing the direct child is not enough: a suite may have spawned its own
    children, and those are the ones that hold memory.
    """
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def acquire_gate():
    """Take the exclusive gate lock. Returns the fd, or None if a gate is live.

    flock is tied to the file descriptor, not to the file contents, so the
    lock is released by the kernel even if this process is SIGKILLed. There is
    no stale lock to clean up, and no wrong answer to reason about.
    """
    path = os.path.join(ROOT, "tests", ".gate.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            holder = os.read(fd, 128).decode(errors="replace").strip()
        except OSError:
            holder = "unknown"
        os.close(fd)
        print("=" * 74)
        print("REFUSED: another gate is already running")
        print("=" * 74)
        print(f"  holder: {holder or 'unknown pid'}")
        print("  This gate would stack TensorFlow processes on top of a live")
        print("  one and drive the machine into swap. Wait for it to finish,")
        print("  or kill it first. Exit 2.")
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"pid {os.getpid()}\n".encode())
    os.fsync(fd)
    return fd


def main():
    gate = acquire_gate()
    if gate is None:
        return 2
    print(f"python: {PY}")
    # os.path.exists(None) raised TypeError, so a machine with no TensorFlow
    # anywhere -- the exact case find_tensorflow_python() was written for --
    # got a traceback instead of a report, and the gate could not run at all
    # where it most needed to say "20 suites did not run". The interpreter is
    # only stat()ed when one was actually found.
    if TF_PY is None:
        print("TF venv: NONE FOUND. Every suite needing TensorFlow will be "
              "reported SKIPPED, which is NOT a pass.")
        print("  looked in:")
        for c in TF_CANDIDATES:
            print(f"    {c}")
        print("  to point the gate at one: AX_TF_PY=/path/to/python")
    else:
        print(f"TF venv: {TF_PY} (found by importing tensorflow in it)")
    results = []
    # A suite entry is (name, argv, need_tf) or (name, argv, need_tf, expect).
    # The unpacking tolerates both so an entry can gain evidence markers
    # without every other entry being touched at the same time.
    for entry in SUITES:
        name, argv, need_tf = entry[0], entry[1], entry[2]
        expect = entry[3] if len(entry) > 3 else None
        results.append((name, run(name, argv, need_tf, expect)))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for name, st in results:
        mark = {"OK": "  OK  ", "FAIL": " FAIL", "SKIP": "SKIP"}[st]
        print(f"  [{mark}] {name}")
    bad = [n for n, s in results if s == "FAIL"]
    skipped = [n for n, s in results if s == "SKIP"]
    print("=" * 74)
    if skipped:
        # Reported separately from failures because it used to be invisible. A
        # hardcoded interpreter path meant that on any other machine every
        # TensorFlow suite was skipped, and the gate still printed ALL GREEN
        # with exit 0. A suite that did not run is not a suite that passed.
        print(f"  SKIPPED: {len(skipped)} of {len(results)}. These did not run:")
        for n in skipped:
            print(f"      {n}")
        print()
        print("  ALL GREEN below means: everything that ran, passed. It does NOT")
        print("  mean the gate is complete.")
        print("=" * 74)
        if bad:
            print(f"  FAILURES: {len(bad)}. Nothing may be published.")
            return 1
        return 2
    if bad:
        print(f"  FAILURES: {len(bad)}. Nothing may be published.")
        return 1
    print("  ALL GREEN. Every suite ran. Results may be published.")
    return 0


if __name__ == "__main__":
    sys.exit(main())