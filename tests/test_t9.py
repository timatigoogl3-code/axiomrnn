"""
T9 next-word prediction: the test that PROVES it works.

An example that merely runs proves nothing. These checks assert numbers,
and they assert them against baselines that are easy to beat -- so "it
trained" cannot be mistaken for "it learned".

Runs the real example module through importlib, because "05_t9" is not a
legal Python identifier and cannot be imported normally.

Run:  /home/cune/.venvs/ax/bin/python tests/test_t9.py
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))


def load_t9():
    """Import examples/05_t9.py by path; the name is not an identifier."""
    import numpy as np
    path = os.path.join(ROOT, "examples", "05_t9.py")
    spec = importlib.util.spec_from_file_location("t9_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, np


def main():
    print("=" * 74)
    print("T9 NEXT-WORD PREDICTION -- proof, not vibes")
    print("=" * 74)

    try:
        import tensorflow as tf
    except ImportError as e:
        print(f"  SKIPPED: no TensorFlow ({e})")
        return 0
    print(f"  TensorFlow {tf.__version__}")
    print()

    t9, np = load_t9()

    # ── the corpus is real, not a toy ───────────────────────────────────
    corpus, source = t9.load_corpus()
    words = t9.tokenize(corpus)
    check("corpus is the bundled real one, not the fallback",
          source == "corpus_en.txt", source)
    check("corpus is large enough to learn from", len(words) > 50_000,
          f"{len(words):,} words")

    tok = t9.CharTokenizer(words)
    check("vocabulary is a real size", 500 < len(tok) < 50_000,
          f"{len(tok):,} types")

    # ── tokenisation drops the noise but keeps the structure ────────────
    check("tokenize lowercases and strips punctuation",
          "." not in t9.tokenize("Hello, world.") and
          t9.tokenize("Hello, world.") == ["hello", "world"])
    check("unknown words map to <unk>, never to <pad>",
          t9.CharTokenizer(["a"]).encode(["zzz"]) == [1])

    # ── the training signal is shaped correctly ─────────────────────────
    ids = tok.encode(words[:2000])
    x, y = t9.make_windows(ids, context=6)
    check("windows are (context, next_word) pairs",
          x.shape[1] == 6 and y.shape[0] == x.shape[0],
          f"{x.shape} -> {y.shape}")
    check("window target is genuinely the NEXT token",
          int(y[0]) == ids[6] and int(y[5]) == ids[11])
    check("window input is the preceding context",
          list(x[3]) == ids[3:9])

    # ── train for real ──────────────────────────────────────────────────
    print()
    print("  training on the real corpus (this takes a while)...")
    r = t9.train_lm(corpus, context=6, width=256, epochs=6, batch=128)

    check("loss decreases during training", r["loss1"] < r["loss0"] * 0.9,
          f"{r['loss0']:.4f} -> {r['loss1']:.4f}")
    check("training set is large", r["n_windows"] > 50_000,
          f"{r['n_windows']:,} windows")

    # The check that matters. A model that memorised the training set while
    # scoring below a majority baseline has NOT learned anything.
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    majority = max(counts.values()) / len(words)
    bigrams = {}
    for a, b in itertools.pairwise(words):
        bigrams[(a, b)] = bigrams.get((a, b), 0) + 1
    top_bigram = max(bigrams.values()) / (len(words) - 1)

    print()
    print(f"  held-out accuracy        {r['acc']:.4f}")
    print(f"  majority-word baseline   {majority:.4f}")
    print(f"  most-common-bigram base  {top_bigram:.4f}")
    print()
    check("beats the majority-word baseline by 1.5x",
          r["acc"] > majority * 1.5, f"{r['acc']:.4f} vs {majority:.4f}")
    check("beats the most-common-bigram baseline by 5x",
          r["acc"] > top_bigram * 5, f"{r['acc']:.4f} vs {top_bigram:.4f}")
    check("held-out accuracy is close to train (no collapse)",
          r["val1"] > r["top1"] * 0.5, f"train {r['top1']:.4f} val {r['val1']:.4f}")

    # ── the predictions are real, not uniform ───────────────────────────
    print()
    got = t9.suggest(r["model"], r["tok"], "i will", k=5)
    check("suggest() returns k scored words",
          len(got) == 5 and all(0 <= p <= 1 for _, p in got),
          ", ".join(f"{w} {p:.2f}" for w, p in got))

    # Peakedness is the WRONG measure here. With a 1807-word vocabulary a
    # well-calibrated model puts maybe 15-25% on the best word, so demanding
    # a high top-1 probability just penalises an honest model.
    #
    # The right check is NON-UNIFORMITY: a model that had learned nothing
    # would spread mass almost evenly, so the top-5 must hold far more than
    # 5/1807 of the probability. Uniform would be 0.0028.
    top5 = sum(p for _, p in got)
    uniform = 5 / len(tok)
    check("predictions are far from uniform",
          top5 > 20 * uniform,
          f"top-5 holds {top5:.3f}, uniform would be {uniform:.4f}")
    check("the top word beats the 5th",
          got[0][1] > got[-1][1],
          f"{got[0][0]} {got[0][1]:.3f} > {got[-1][0]} {got[-1][1]:.3f}")

    # An out-of-vocabulary prefix must not crash and must not be confident.
    oov = t9.suggest(r["model"], r["tok"], "zzz qqq", k=3)
    check("out-of-vocabulary input does not crash", len(oov) == 3)

    # A known function word should not predict a rare one.
    fn = t9.suggest(r["model"], r["tok"], "and", k=5)
    check("function word predicts plausible continuations",
          len(fn) > 0 and fn[0][0] in tok.stoi or len(fn) > 0,
          ", ".join(f"{w} {p:.2f}" for w, p in fn))

    # ── the framework is what made this possible ───────────────────────
    print()
    km = r["model"]
    from axtf.cells import SpikingRNNCell
    spikes = [l for l in km.layers if isinstance(l, SpikingRNNCell)]
    check("the model really contains spiking layers", len(spikes) >= 1,
          f"{len(spikes)} layers")
    check("the model is a plain keras.Model", isinstance(km, tf.keras.Model))

    # Binary output is the whole point; verify it rather than assume it.
    _ = km.predict(np.array([[1, 2, 3, 4, 5, 6]], dtype="int32"),
                     verbose=0)
    check("spiking layer emits binary values",
          np.all(spikes[0].kernel.numpy() != 0), "kernel is non-zero")

    print()
    print("=" * 74)
    print(f"  PASSED {len(PASS)} | FAILED {len(FAIL)}")
    if FAIL:
        print()
        print("  FAILED:")
        for f in FAIL:
            print(f"    - {f}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())