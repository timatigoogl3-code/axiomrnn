"""axtf -- the only package that knows about TensorFlow and Keras.

Contains the spiking cell, the Keras layer, and model assembly. If a rule
belongs to training or assembly, it lives here and nowhere else.

The cell classes are re-exported here so that exactly one import spelling works:

    from axtf import SpikingRNNCell

`axtf.cells` remains the implementation module and is still importable, but a
user should not have to know which of the two is the public one. When
TensorFlow is absent the names still exist and refuse on use with the command
that fixes it, because a name that vanishes on import teaches nothing and a name
that is silently None produces "TypeError: 'NoneType' object is not callable",
which reads as a bug in this package rather than a missing dependency.
"""

__all__ = ["TF_HINT", "SpikingCell", "SpikingRNNCell", "build", "cells"]

#: The single place the TensorFlow install instructions are written. axiomrnn.py
#: imports this rather than repeating it, so the guidance cannot drift between
#: the two layers that give it.
TF_HINT = (
    "Spiking cells need TensorFlow, which is an optional dependency so that\n"
    "  the memory planner stays usable without a deep learning framework.\n"
    "\n"
    "  Install it with:      pip install 'axiomrnn[tf]'\n"
    "  For a GPU build:      pip install 'axiomrnn[tf-gpu]'\n"
    "\n"
    "The memory planner needs no TensorFlow and works right now:\n"
    "\n"
    "    import axiomrnn as ax\n"
    "    m = ax.Model(vocab=32000, dim=768, layers=12)\n"
    "    b = ax.Budget.consumer_8gb()\n"
    "    print(ax.explain_budget(m, b, verbose=False))"
)

try:
    from axtf.cells import SpikingCell, SpikingRNNCell

    HAS_TF = True
    #: Why the import failed, empty when TensorFlow is present. Kept because
    #: "No module named X" is more useful than a generic refusal.
    TF_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised only without TensorFlow  # noqa: BLE001
    HAS_TF = False
    TF_ERROR = str(exc)

    def _refuse(feature: str):
        """Raise an ImportError that names the command which enables `feature`."""
        raise ImportError(
            f"{feature} requires TensorFlow, which is not installed.\n\n{TF_HINT}"
            + (f"\n\nThe underlying import failed with: {TF_ERROR}" if TF_ERROR else "")
        )

    class SpikingCell:
        """Stand-in for the real cell; raises on use. See axtf.TF_HINT."""

        def __init__(self, *args, **kwargs):
            _refuse("axtf.SpikingCell")

    class SpikingRNNCell:
        """Stand-in for the real cell; raises on use. See axtf.TF_HINT."""

        def __init__(self, *args, **kwargs):
            _refuse("axtf.SpikingRNNCell")