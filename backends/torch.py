"""Optional PyTorch backend for axiomrnn (skeleton).

Importable whether or not PyTorch is installed. When PyTorch is absent,
`has_backend()` returns False and every operation raises a clear ImportError.
"""

try:
    import torch as _torch  # noqa: F401
    _HAS_TORCH = True
    _TORCH_ERR = ""
except Exception as exc:  # pragma: no cover - depends on environment  # noqa: BLE001
    _HAS_TORCH = False
    _TORCH_ERR = str(exc)

from .base import Backend


class TorchBackend(Backend):
    """Skeleton backend. PyTorch implementation is not provided yet."""

    @property
    def name(self) -> str:
        return "torch"

    def has_backend(self) -> bool:
        return bool(_HAS_TORCH)

    def _require(self):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is not installed.\n\n"
                "Install it with: pip install torch\n\n"
                f"Underlying import error: {_TORCH_ERR}")
        raise RuntimeError(
            "TorchBackend is a skeleton: the PyTorch implementation is not "
            "available yet. Use backend='tensorflow'.")

    def build_cell(self, units, leak=0.9, alpha=2.0, cell_cls=None):
        self._require()

    def build_model(self, spec, inputs=None, outputs=None, readout="flatten",
                    compile_model=False):
        self._require()

    def compile(self, model, **kwargs):
        self._require()

    def train(self, model, x, y, **kwargs):
        self._require()

    def predict(self, model, x, **kwargs):
        self._require()

    def save(self, model, path):
        self._require()

    def load(self, path):
        self._require()
