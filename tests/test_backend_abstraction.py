"""Tests for the backend abstraction layer."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import axiomrnn as ax


def test_backend_classes_import():
    from backends.base import Backend
    from backends.tensorflow import TensorFlowBackend
    from backends.torch import TorchBackend
    from backends import get_backend

    assert Backend is not None
    tfb = TensorFlowBackend()
    assert tfb.name == "tensorflow"
    assert isinstance(tfb.has_backend(), bool)
    tb = TorchBackend()
    assert tb.name == "torch"
    assert isinstance(tb.has_backend(), bool)


def test_backend_selection():
    m = ax.Model(backend="tensorflow")
    assert m.backend == "tensorflow"
    m2 = ax.Model(backend="auto")
    assert m2.backend == "auto"

    from backends import get_backend

    be = get_backend("tensorflow")
    assert be.name == "tensorflow"
    be2 = get_backend("auto")
    assert be2.name in ("torch", "tensorflow")
    try:
        m.get_backend()
    except Exception as e:
        raise AssertionError(str(e))


def test_tf_backend_builds_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        return
    spec = ax.Model()
    spec.add_spiking(16, horizon=4)
    spec.add_dense(3)

    from backends.tensorflow import TensorFlowBackend

    be = TensorFlowBackend()
    assert be.has_backend() is True
    km = be.build_model(spec, inputs=keras.Input(batch_shape=(None, 4, 2)),
                        outputs=3)
    assert isinstance(km, keras.Model)

    km2 = spec.build(inputs=keras.Input(batch_shape=(None, 4, 2)), outputs=3)
    assert isinstance(km2, keras.Model)

    km3 = spec.keras_model(inputs=keras.Input(batch_shape=(None, 4, 2)),
                           outputs=3)
    assert isinstance(km3, keras.Model)


def test_torch_backend_import_does_not_crash():
    import backends.torch as bt
    from backends.torch import TorchBackend

    tb = TorchBackend()
    assert tb.name == "torch"
    assert isinstance(tb.has_backend(), bool)
    try:
        import torch
    except ImportError:
        try:
            tb.build_model(None)
        except Exception:
            pass
        else:
            raise AssertionError("expected failure when torch missing")


if __name__ == "__main__":
    test_backend_classes_import()
    test_backend_selection()
    test_tf_backend_builds_model()
    test_torch_backend_import_does_not_crash()
    print("OK")