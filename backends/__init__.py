"""backends -- framework backends for axiomrnn.

Kept import-light: importing this package does not import TensorFlow or
PyTorch. Backend classes are imported lazily where needed.
"""

from .base import Backend
from .tensorflow import TensorFlowBackend
from .torch import TorchBackend


def get_backend(name_or_instance=None):
    """Resolve a backend name or instance into a Backend object."""
    if isinstance(name_or_instance, Backend):
        return name_or_instance
    if name_or_instance is None:
        name_or_instance = "tensorflow"
    name = str(name_or_instance).strip().lower()
    if name == "auto":
        # Prefer torch when it is actually available, else tensorflow.
        try:
            from .torch import TorchBackend
            tb = TorchBackend()
            if tb.has_backend():
                return tb
        except Exception:
            pass
        from .tensorflow import TensorFlowBackend
        return TensorFlowBackend()
    if name in ("torch", "pytorch", "torchbackend"):
        from .torch import TorchBackend
        return TorchBackend()
    if name in ("tf", "tensorflow", "tensorflowbackend"):
        from .tensorflow import TensorFlowBackend
        return TensorFlowBackend()
    raise ValueError(f"unknown backend: {name_or_instance!r}")


__all__ = ["Backend", "TensorFlowBackend", "TorchBackend", "get_backend"]
