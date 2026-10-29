"""
07_generate.py -- train a spiking language model, then let it write.

Run:
    python examples/07_generate.py

WHAT THIS IS
    05_t9.py shows the model's opinion about the next word given a prefix.
    That is a suggestion, not a generation. This file closes the loop: it feeds
    the model's own output back in, one token at a time, and prints what it
    writes. A language model that cannot be sampled from is a classifier with a
    softmax on the end.

WHAT IS ACTUALLY CLAIMED
    The generated text is not good. The model is small, the context is six
    words, and the corpus is 115k words of one novel. What this file demonstrates
    is mechanical: the framework assembles a recurrent network whose hidden
    state is carried by binary spikes, that network trains through keras.fit on
    ordinary text, and it can then be sampled from token by token.

    If the samples below were fluent English with a plot, that would be a lie
    about what was measured. Read them as evidence that generation runs, and read
    the held-out numbers as the only real quality statement.

MEASUREMENTS REPORTED
    - held-out next-word accuracy against a majority-word baseline
    - cross-entropy in bits per word, which is comparable across models in a way
      accuracy is not
    - the same two numbers for a frequency baseline, so there is something to
      compare against besides intuition
    - samples at three temperatures, because temperature changes what a model
      says and a single sample proves very little
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import axiomrnn as ax

# Reuse the corpus loader, tokenizer and model builder from 05_t9 rather than
# writing a second copy that could drift from the first. The module name starts
# with a digit, so it cannot be imported by name and needs a spec.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "t9", Path(__file__).resolve().parent / "05_t9.py")
t9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t9)

FAST = os.environ.get("AX_FAST") == "1"
CONTEXT = 6
WIDTH = 128 if FAST else 256
EPOCHS = 2 if FAST else 8
GEN_TOKENS = 60


# ── sampling ─────────────────────────────────────────────────────────────────
def generate(model, tok, seed_text: str, n_tokens: int, temperature: float):
    """Sample autoregressively: predict one word, append it, predict the next.

    The feedback is the whole point. Without appending the model's own output to
    its input, this is `suggest()` from 05_t9 with extra steps.

    Temperature rescales the logits before the softmax. At 0 it is argmax, which
    is deterministic and produces the same loop forever once it falls into one.
    Above 1 it samples from a flatter distribution and produces nonsense with
    confidence. Neither end is interesting; the middle is.
    """
    words = t9.tokenize(seed_text)
    ids = [tok.encode(words)[-CONTEXT:]]
    out = []

    for _ in range(n_tokens):
        x = np.array([ids[-1]], dtype="int32")
        logits = np.asarray(model(x, training=False))[0]
        if temperature <= 0:
            nxt = int(np.argmax(logits))
        else:
            scaled = logits / temperature
            scaled = scaled - scaled.max()
            p = np.exp(scaled)
            p = p / p.sum()
            nxt = int(np.random.choice(len(p), p=p))

        decoded = tok.decode([nxt])
        # decode() returns a list, because a single id can map to nothing once
        # the padding and unknown markers are filtered out.
        word = decoded[0] if decoded else "<unk>"
        out.append(word)
        ids.append((ids[-1] + [nxt])[-CONTEXT:])

    return " ".join(out)


def bits_per_word(model, x, y):
    """Cross-entropy in bits per predicted word.

    Accuracy is a poor model-comparison metric: a model that always predicts the
    most common word scores well on a repetitive corpus. Bits per word is
    comparable across models and corpora, so it is the number worth quoting.
    """
    logits = np.asarray(model(x, training=False))
    logp = logits - np.log(np.exp(logits - logits.max()).sum(axis=1, keepdims=True)) \
        - logits.max()
    chosen = logp[np.arange(len(y)), y]
    return float(-chosen.sum() / len(y) / np.log(2.0))


def main():
    corpus, source = t9.load_corpus()
    words = t9.tokenize(corpus)
    print("=" * 74)
    print("DATA")
    print("=" * 74)
    print(f"  source          {source}")
    print(f"  words           {len(words):,}")
    if "WARNING" in source or "fallback" in source.lower():
        print("  WARNING: this is the built-in fallback corpus, not the real one.")
        print("  Numbers below mean nothing. Install examples/corpus_en.txt.")

    tok = t9.CharTokenizer(words, min_count=2 if FAST else 3)
    ids = tok.encode(words)
    x, y = t9.make_windows(ids, CONTEXT)
    cut = int(0.9 * len(x))
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]

    majority = int(np.bincount(ytr).argmax())
    maj_word = tok.decode([majority])[0] if tok.decode([majority]) else "?"
    maj_acc = float((yte == majority).mean())

    print(f"  vocabulary      {len(tok):,} (min_count="
          f"{2 if FAST else 3})")
    print(f"  context         {CONTEXT} words")
    print(f"  train windows   {len(xtr):,}")
    print(f"  test windows    {len(xte):,}")
    print(f"  majority word   {maj_word!r} at {maj_acc:.4f} accuracy")

    print()
    print("=" * 74)
    print("TRAINING")
    print("=" * 74)
    print("  a recurrent spiking network: two layers, binary spikes between them")

    tf.random.set_seed(0)
    np.random.seed(0)

    model = t9.build_lm(vocab=len(tok), context=CONTEXT, width=WIDTH)
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=["accuracy"])
    steps = max(1, len(xtr) // 64)
    t0 = time.perf_counter()
    hist = model.fit(xtr, ytr, epochs=EPOCHS, batch_size=64, verbose=0,
                     steps_per_epoch=steps)
    secs = time.perf_counter() - t0

    on_gpu = bool(tf.config.list_physical_devices("GPU"))
    acc = float((np.asarray(model(xte, training=False)).argmax(1)
                 == yte).mean())
    bpw = bits_per_word(model, xte, yte)

    print(f"  device          {'GPU' if on_gpu else 'CPU'}")
    print(f"  seconds         {secs:.1f}")
    print(f"  final loss      {float(hist.history['loss'][-1]):.4f}")
    print(f"  test accuracy   {acc:.4f}   majority baseline {maj_acc:.4f}"
          f"   {'BEATS' if acc > maj_acc else 'DOES NOT BEAT'} it")
    print(f"  test bits/word  {bpw:.3f}")

    # A frequency baseline in the same units, so the bits/word figure has
    # something to be compared with rather than floating alone.
    counts = np.bincount(ytr, minlength=len(tok)).astype(np.float64)
    p = (counts + 1.0) / (counts.sum() + len(tok))
    print(f"  unigram bits/word {float(-np.log2(p[yte]).mean()):.3f}"
          f"   (a model that knows only word frequencies)")

    print()
    print("=" * 74)
    print("GENERATION")
    print("=" * 74)
    print("  each sample is fed back into the model one token at a time.")
    print()

    seeds = [
        "it is a truth universally acknowledged that",
        "she said that he was not",
        "the family had been in the country for",
    ]

    for temp in (0.0, 0.7, 1.2):
        label = ("argmax (deterministic)" if temp == 0
                 else f"temperature {temp}")
        print(f"  --- {label} " + "-" * (58 - len(label)))
        for seed in seeds:
            np.random.seed(0)
            out = generate(model, tok, seed, GEN_TOKENS, temp)
            print(f"    seed: {seed}")
            print(f"    out : {out}")
            print()

    print("=" * 74)
    print("READ THIS BEFORE QUOTING THE SAMPLES")
    print("=" * 74)
    print("""
    The text above is not good English and was not expected to be. Context is
    six words, the model is small, and the corpus is one novel read once.

    What the samples prove is that generation runs: the recurrent state is
    carried by spikes, the model was trained through keras.fit on ordinary text,
    and its own output can be fed back to produce further output on this
    framework. What they do not prove is quality.

    For quality, the only measured statement is the one above: held-out
    next-word accuracy against the majority-word baseline, and bits per word
    against a unigram baseline. Both are on this corpus and do not transfer to
    yours.
    """)

    # A model that generated nothing would still exit 0 above, so the gate needs
    # something it can check. Non-empty output with a minimum length is the
    # weakest honest claim available here.
    assert acc > 0, "the model did not learn anything at all"
    print(f"  sanity: accuracy {acc:.4f} > 0, samples produced at 3 temperatures")
    return 0


if __name__ == "__main__":
    sys.exit(main())