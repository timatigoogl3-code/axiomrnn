"""Abstract backend interface for axiomrnn."""

from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract backend. Implementations wrap a deep learning framework."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def has_backend(self) -> bool:
        ...

    @abstractmethod
    def build_cell(self, units, leak=0.9, alpha=2.0, cell_cls=None):
        ...

    @abstractmethod
    def build_model(self, spec, inputs=None, outputs=None, readout="flatten",
                    compile_model=False):
        ...

    @abstractmethod
    def compile(self, model, **kwargs):
        ...

    @abstractmethod
    def train(self, model, x, y, **kwargs):
        ...

    @abstractmethod
    def predict(self, model, x, **kwargs):
        ...

    @abstractmethod
    def save(self, model, path):
        ...

    @abstractmethod
    def load(self, path):
        ...
