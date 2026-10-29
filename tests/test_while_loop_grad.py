"""Tests for tf.while_loop based gradients."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np


def test_while_loop_gradients_finite():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return
    from axtf.cells import SpikingCell, SpikingRNNCell

    rng = np.random.default_rng(0)
    B, T, DIN, UNITS = 4, 5, 3, 6
    x = rng.normal(0, 1, (B, T, DIN)).astype(np.float32)
    layer = SpikingRNNCell(UNITS, cell=SpikingCell(0.9, 2.0),
                            norm_t=True, learn_threshold=True)
    layer.build((None, T, DIN))
    w = rng.normal(0, 1 / np.sqrt(DIN), (DIN, UNITS)).astype(np.float32)
    thr = np.abs(rng.normal(0.5, 0.1, UNITS)).astype(np.float32)
    layer.kernel.assign(w)
    layer.thr.assign(thr)

    dout = np.ones((B, T, UNITS), dtype=np.float32)
    dx, dk, dthr = layer.backward(x, dout, T)
    assert np.all(np.isfinite(dx))
    assert np.all(np.isfinite(dk))
    assert np.all(np.isfinite(dthr))


def test_graph_mode_training_step_finite():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        return
    import axiomrnn as ax
    from axtf.build import build_keras

    X = np.random.default_rng(1).normal(size=(16, 6, 4)).astype(np.float32)
    y = np.random.default_rng(2).integers(0, 3, size=(16,)).astype(np.int32)

    spec = ax.Model()
    spec.add_spiking(8, horizon=6)
    spec.add_dense(3)
    m = build_keras(spec, inputs=keras.Input(shape=(6, 4)), outputs=3)
    m.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              run_eagerly=False)

    @tf.function
    def step(xb, yb):
        with tf.GradientTape() as tape:
            logits = m(xb, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(
                yb, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, m.trainable_variables)
        return loss, grads

    loss, grads = step(tf.constant(X), tf.constant(y))
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(g.numpy())) for g in grads if g is not None)


def test_norm_t_normalisation_preserved():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return
    from axtf.cells import SpikingCell, SpikingRNNCell

    rng = np.random.default_rng(3)
    B, T, DIN, UNITS = 3, 4, 3, 5
    x = rng.normal(0, 1, (B, T, DIN)).astype(np.float32)
    w = rng.normal(0, 1 / np.sqrt(DIN), (DIN, UNITS)).astype(np.float32)
    thr = np.abs(rng.normal(0.5, 0.1, UNITS)).astype(np.float32)
    dout = np.ones((B, T, UNITS), dtype=np.float32)

    def make(norm_t):
        layer = SpikingRNNCell(UNITS, cell=SpikingCell(0.9, 2.0),
                                norm_t=norm_t)
        layer.build((None, T, DIN))
        layer.kernel.assign(w)
        layer.thr.assign(thr)
        return layer

    l_on = make(True)
    l_off = make(False)
    _, dk_on, _ = l_on.backward(x, dout, T)
    _, dk_off, _ = l_off.backward(x, dout, T)
    ratio = np.linalg.norm(dk_on) / (np.linalg.norm(dk_off) + 1e-12)
    assert abs(ratio - (1.0 / T)) < 1e-5, (ratio, 1.0 / T)


if __name__ == "__main__":
    test_while_loop_gradients_finite()
    test_graph_mode_training_step_finite()
    test_norm_t_normalisation_preserved()
    print("OK")
