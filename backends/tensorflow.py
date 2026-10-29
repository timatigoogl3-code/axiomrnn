"""TensorFlow backend for axiomrnn."""

from .base import Backend


class TensorFlowBackend(Backend):
    @property
    def name(self) -> str:
        return "tensorflow"

    def has_backend(self) -> bool:
        try:
            import tensorflow  # noqa: F401
            return True
        except Exception:
            return False

    def _require(self):
        if not self.has_backend():
            try:
                from axtf import TF_HINT, TF_ERROR
                hint = TF_HINT
                err = TF_ERROR
            except Exception:
                hint = "TensorFlow is required."
                err = ""
            raise RuntimeError(
                "TensorFlow is not installed.\n\n" + hint
                + (f"\n\nUnderlying error: {err}" if err else ""))

    def build_cell(self, units, leak=0.9, alpha=2.0, cell_cls=None):
        self._require()
        if cell_cls is None:
            from axtf.cells import SpikingCell
            cell_cls = SpikingCell
        # cell classes may accept (leak, alpha) positionally or as keywords.
        try:
            return cell_cls(leak=leak, alpha=alpha)
        except TypeError:
            try:
                return cell_cls(leak, alpha)
            except TypeError:
                return cell_cls()

    def build_model(self, spec, inputs=None, outputs=None, readout="flatten",
                    compile_model=False):
        self._require()
        from axtf.build import build_keras
        return build_keras(spec, inputs=inputs, outputs=outputs,
                           compile_model=compile_model, readout=readout)

    def compile(self, model, **kwargs):
        self._require()
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = "adam"
        if "loss" not in kwargs:
            from tensorflow import keras
            kwargs["loss"] = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["accuracy"]
        if "run_eagerly" not in kwargs:
            kwargs["run_eagerly"] = True
        model.compile(**kwargs)
        return model

    def train(self, model, x, y, **kwargs):
        self._require()
        return model.fit(x, y, **kwargs)

    def predict(self, model, x, **kwargs):
        self._require()
        return model.predict(x, **kwargs)

    def save(self, model, path):
        self._require()
        model.save(path)
        return path

    def load(self, path):
        self._require()
        from tensorflow import keras
        return keras.models.load_model(path)
