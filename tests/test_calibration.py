"""Tests for model calibration helpers."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np


def _data(n=200, T=8, d=12, c=4, seed=0):
    r = np.random.default_rng(seed)
    blk = max(1, T // c)
    pat = np.zeros((c, T, d))
    for k in range(c):
        lo = min(k * blk, T - 1)
        pat[k, lo:min(lo + 2, T)] = 1.0
        pat[k] += r.normal(0, 0.25, (T, d))
    y = r.integers(0, c, n)
    X = pat[y] * (r.random((n, T, d)) < 0.18) + r.normal(0, 0.5, (n, T, d))
    return X.astype(np.float32), y.astype(np.int32)


def test_calibrate_model_sets_firing_rates():
    try:
        from tensorflow import keras
    except ImportError:
        return
    import axiomrnn as ax
    from axtf.build import build_keras, calibrate_model

    X, _ = _data()
    spec = ax.Model()
    spec.add_spiking(32, horizon=8)
    spec.add_spiking(16, horizon=8)
    spec.add_dense(4)
    m = build_keras(spec, inputs=keras.Input(shape=(8, 12)), outputs=4)

    info = calibrate_model(m, X[:128], target_rate=0.25, iters=8)
    assert len(info) == 2
    for name, fr, dead in info:
        assert 0.0 <= fr <= 1.0
        assert abs(fr - 0.25) < 0.15, (name, fr)
        assert 0.0 <= dead <= 1.0


def test_calibrate_verbose_logging_does_not_crash():
    try:
        from tensorflow import keras
    except ImportError:
        return
    import axiomrnn as ax
    from axtf.build import build_keras, calibrate_model

    X, _ = _data(n=128)
    spec = ax.Model()
    spec.add_spiking(24, horizon=8)
    spec.add_dense(4)
    m = build_keras(spec, inputs=keras.Input(shape=(8, 12)), outputs=4)

    # verbose=True prints per-layer stats; must not raise.
    info = calibrate_model(m, X, target_rate=0.2, iters=4, verbose=True)
    assert isinstance(info, list)


if __name__ == "__main__":
    test_calibrate_model_sets_firing_rates()
    test_calibrate_verbose_logging_does_not_crash()
    print("OK")
