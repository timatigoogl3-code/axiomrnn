"""bench_inference.py -- honest inference latency for an axiomrnn model.

Run:
    python examples/bench_inference.py            # CPU
    AX_FAST=1 python examples/bench_inference.py # smaller model

This measures wall-clock time for a single forward pass of a small trained
spiking model, using the direct call path (model(x)) that `ax.generate` uses.
It does NOT use model.predict, and it does NOT print a pre-baked number:
the number depends on your machine, your TensorFlow build and whether a GPU
is present. Run it yourself and quote your own result.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import axiomrnn as ax
from axtf.build import build_keras, calibrate_model

FAST = os.environ.get("AX_FAST") == "1"
T = 8 if FAST else 16
D = 8 if FAST else 12
N = 200 if FAST else 1000
BATCH = 1 if FAST else 4
WARMUP = 5
ITERS = 20


def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    spec = ax.Model()
    spec.add_spiking(64 if FAST else 128, horizon=T)
    spec.add_spiking(32 if FAST else 64, horizon=T)
    spec.add_dense(4)

    m = build_keras(spec, inputs=tf.keras.Input(shape=(T, D)), outputs=4)
    x = np.random.normal(size=(BATCH, T, D)).astype("float32")
    calibrate_model(m, x, target_rate=0.2, iters=4)

    # warmup (first call compiles the eager graph / allocates)
    for _ in range(WARMUP):
        _ = m(x, training=False)

    # direct call path used by ax.generate
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = m(x, training=False)
    direct_ms = (time.perf_counter() - t0) / ITERS * 1000.0

    # predict path, for comparison only
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = m.predict(x, verbose=0)
    predict_ms = (time.perf_counter() - t0) / ITERS * 1000.0

    print(f"model: 2 spiking layers, T={T}, D={D}, batch={BATCH}")
    print(f"device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print(f"direct call (model(x)): {direct_ms:.2f} ms / pass")
    print(f"model.predict:          {predict_ms:.2f} ms / pass")
    print("Note: run_eagerly=True is required for correct gradients; it makes")
    print("inference slower than a graph-mode model. These numbers are for")
    print("this machine only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
