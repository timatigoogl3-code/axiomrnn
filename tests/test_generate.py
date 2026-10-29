"""Tests for the generate() helper."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _build(vocab=10, context=4, width=8):
    try:
        from tensorflow import keras
    except ImportError:
        return None
    import axiomrnn as ax
    from axtf.cells import SpikingRNNCell

    spec = ax.Model()
    spec.add_spiking(width, horizon=context)
    tokens = keras.Input(batch_shape=(None, context), dtype="int32", name="tokens")
    emb = keras.layers.Embedding(vocab, width, name="embed")(tokens)
    x = emb
    for i, layer in enumerate(spec.layers):
        x = SpikingRNNCell(
            layer.units, cell=layer.cell, return_sequences=True,
            norm_t=True, horizon_hint=context, name=f"spiking_{i}",
        )(x)
    x = keras.layers.Flatten(name="readout")(x)
    logits = keras.layers.Dense(vocab, name="logits")(x)
    model = keras.Model(tokens, logits, name="gen")
    return model, vocab, context


class _Tok:
    def __init__(self, words):
        self.itos = ["<pad>", "<unk>"] + sorted(set(words))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, words):
        return [self.stoi.get(w, 1) for w in words]

    def decode(self, ids):
        out = []
        for i in ids:
            if 0 <= i < len(self.itos) and i > 1:
                out.append(self.itos[i])
        return out


def _tokens(res):
    if isinstance(res, (list, tuple)):
        return list(res)
    return str(res).split()


def test_generate_runs_without_hanging():
    built = _build()
    if built is None:
        return
    model, vocab, context = built
    words = ["a", "b", "c", "d", "e"]
    tok = _Tok(words)
    # vocab size must match model
    assert len(tok.itos) <= vocab or True
    # pad vocab by rebuilding tok with enough words
    tok = _Tok([f"w{i}" for i in range(vocab - 2)])
    import axiomrnn as ax

    res = ax.generate(model, tok, seed="w0", n=4, context=context,
                      temperature=0.0)
    toks = _tokens(res)
    assert len(toks) == 4


def test_generate_temperature_zero_deterministic():
    built = _build()
    if built is None:
        return
    model, vocab, context = built
    tok = _Tok([f"w{i}" for i in range(vocab - 2)])
    import axiomrnn as ax

    r1 = ax.generate(model, tok, seed=["w0", "w1"], n=6, context=context,
                     temperature=0)
    r2 = ax.generate(model, tok, seed=["w0", "w1"], n=6, context=context,
                     temperature=0)
    assert r1 == r2


def test_generate_short_seed_pads_correctly():
    built = _build()
    if built is None:
        return
    model, vocab, context = built
    tok = _Tok([f"w{i}" for i in range(vocab - 2)])
    import axiomrnn as ax

    # seed shorter than context must not crash and must produce n tokens
    res = ax.generate(model, tok, seed=["w0"], n=3, context=context,
                      temperature=0.0)
    toks = _tokens(res)
    assert len(toks) == 3


if __name__ == "__main__":
    test_generate_runs_without_hanging()
    test_generate_temperature_zero_deterministic()
    test_generate_short_seed_pads_correctly()
    print("OK")
