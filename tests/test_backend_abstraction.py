"""Tests for the backend abstraction layer."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import axiomrnn as ax


def test_backend_classes_import():
    from backends import get_backend
    from backends.base import Backend
    from backends.tensorflow import TensorFlowBackend
    from backends.torch import TorchBackend

    assert Backend is not None
    tfb = TensorFlowBackend()
    assert tfb.name == "tensorflow"
    assert isinstance(tfb.has_backend(), bool)
    tb = TorchBackend()
    assert tb.name == "torch"
    assert isinstance(tb.has_backend(), bool)
    assert get_backend is not None


def test_default_backend_is_tensorflow():
    m = ax.Model()
    assert m.backend == "tensorflow"
    assert m._backend == "tensorflow"
    be = m.get_backend()
    assert be.name == "tensorflow"


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
    # Resolving a backend that exists must not raise.
    assert m.get_backend().name == "tensorflow"


def test_tf_backend_builds_model():
    try:
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


def test_model_build_tensorflow_trainable():
    try:
        from tensorflow import keras
    except ImportError:
        return
    spec = ax.Model()
    spec.add_spiking(8, horizon=4)
    spec.add_dense(3)
    km = spec.build(backend="tensorflow",
                    inputs=keras.Input(batch_shape=(None, 4, 2)),
                    outputs=3)
    assert isinstance(km, keras.Model)
    assert len(km.trainable_variables) > 0


def test_get_backend_torch_raises_helpful_error():
    from backends import get_backend

    be = get_backend("torch")
    assert be.name == "torch"
    assert isinstance(be.has_backend(), bool)
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        try:
            be.build_model(None)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            assert "torch" in msg or "pytorch" in msg
            assert "pip install" in msg
        else:
            raise AssertionError("expected failure when torch missing")


def test_torch_backend_import_does_not_crash():
    from backends.torch import TorchBackend

    tb = TorchBackend()
    assert tb.name == "torch"
    assert isinstance(tb.has_backend(), bool)
    try:
        import torch  # noqa: F401
    except ImportError:
        raised = False
        try:
            tb.build_model(None)
        except Exception:  # noqa: BLE001
            raised = True
        assert raised, "expected failure when torch missing"


def test_import_axiomrnn_works_without_tf():
    # axplan and axon must import with no TensorFlow dependency.
    import axon
    import axplan

    assert axplan is not None
    assert axon is not None
    # axiomrnn itself must import (it degrades gracefully).
    import axiomrnn as _ax

    assert _ax is not None


if __name__ == "__main__":
    test_backend_classes_import()
    test_default_backend_is_tensorflow()
    test_backend_selection()
    test_tf_backend_builds_model()
    test_model_build_tensorflow_trainable()
    test_get_backend_torch_raises_helpful_error()
    test_torch_backend_import_does_not_crash()
    test_import_axiomrnn_works_without_tf()
    print("OK")
