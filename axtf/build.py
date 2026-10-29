"""
axtf.build -- assemble a real Keras model from an axiomrnn spec.

The single place where our layer becomes part of a Keras graph. The user
writes:

    model = ax.Model()
    model.add_spiking(256, horizon=8)
    model.add_spiking(128, horizon=8)
    model.add_dense(10)
    km = model.keras_model(inputs=..., outputs=10)

and gets an ordinary Keras model that compiles and trains like any other.

WHAT WE DO NOT BUILD
====================
Layers? No, Keras has them. Optimizer? No. Training loop? No. Autodiff? No.
Serialization? No.

We add exactly five things:
  1. memory model (axplan)
  2. planner (axplan)
  3. diagnostics
  4. a few kernels
  5. explain()

Reason: TF/Keras are mature, so everything here is something they
fundamentally do not do.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .cells import SpikingCell, SpikingRNNCell


class _BuildError(ValueError, TypeError):
    """Raised when build_keras cannot assemble a model."""
    pass


class _Readout(keras.layers.Layer):
    """Turn spikes into a continuous signal for the output layer.

    Two modes. The choice is not cosmetic -- it is a measured difference.

      'mean'    : average over time. Independent of T, compact.
      'flatten' : whole sequence. T enters the input dimension.

    MEASURED (verify_14, RTX 3050, T=16, width=256, 2 layers, threshold
    scanned, trunk parameters equal):

        dense (MLP)              0.9981
        spiking, readout=flatten 0.9912   <- nearly matches dense
        spiking, readout=mean    0.7356

    So the spiking net DOES solve the task, and the 0.26 gap is not bit-ness
    at all -- it is loss of temporal resolution in the mean. Checked
    separately: replacing the hard 1[u>thr] with a smooth one under mean
    readout gives 0.7475 vs 0.7356, so bit-ness costs 0.012 and everything
    else is the readout.

    Hence the 'flatten' default. 'mean' stays because for tasks that do not
    depend on exact timing it is more compact and keeps T out of the input
    dimension -- but it must be chosen deliberately.

    The 1/T normalisation for 'mean' is the part that needs an explicit
    gradient fix, which SpikingRNNCell (norm_t) does rather than hoping
    Keras infers it.
    """

    def __init__(self, mode="flatten", **kw):
        super().__init__(**kw)
        if mode not in ("mean", "flatten"):
            raise ValueError(
                f"unknown readout mode '{mode}'; expected 'mean' or 'flatten'")
        self.mode = mode

    def call(self, spikes):
        if self.mode == "flatten":
            return tf.reshape(spikes, (tf.shape(spikes)[0], -1))
        return tf.reduce_mean(spikes, axis=1)

    def compute_output_shape(self, input_shape):
        b, t, u = input_shape
        return (b, t * u) if self.mode == "flatten" else (b, u)


def build_keras(spec, inputs=None, outputs=None, compile_model=False,
                readout="flatten"):
    """Build a Keras model from an axiomrnn spec.

    spec      : axiomrnn.Model
    inputs    : keras.Input, or None -> (None, None)
    outputs   : number of classes; None -> no output layer
    readout   : 'flatten' (default, measured better) or 'mean'

    WARNING: a freshly built model is silent until calibrated. Call
    calibrate_model(model, x, ...) before training, or no gradient flows.
    """
    from axiomrnn import Dense, SpikingRNN

    spiking_layers = [l for l in spec.layers if isinstance(l, SpikingRNN)]
    if outputs is not None and not spiking_layers:
        raise _BuildError(
            "build_keras: `outputs` was given but the model has no spiking "
            "layers. Add at least one SpikingRNN layer before attaching an "
            "output head, or pass outputs=None.")

    if inputs is None:
        inputs = keras.Input(shape=(None, None), name="input")

    x = inputs
    for i, layer in enumerate(spec.layers):
        if isinstance(layer, SpikingRNN):
            cell = layer.cell or SpikingCell(leak=layer.leak,
                                             alpha=layer.alpha)
            x = SpikingRNNCell(
                layer.units, cell=cell,
                return_sequences=True,          # always true until readout
                norm_t=layer.norm_t,
                credit_k=layer.credit_k,
                horizon_hint=layer.horizon or None,
                name=f"spiking_{i}",
            )(x)
        elif isinstance(layer, Dense):
            x = keras.layers.Dense(layer.units, name=f"dense_{i}")(
                _Readout(mode=readout)(x))
            if layer.activation:
                x = keras.layers.Activation(layer.activation)(x)
        else:
            raise TypeError(f"unknown layer: {type(layer)}")

    if outputs is not None:
        if not isinstance(outputs, int):
            raise TypeError("outputs must be an integer class count")
        if not isinstance(spec.layers[-1], Dense):
            x = _Readout(mode=readout)(x)
            x = keras.layers.Dense(outputs, name="logits")(x)

    m = keras.Model(inputs, x, name="axiomrnn")
    if compile_model:
        # run_eagerly=True: see the long note in axtf/cells.py. Keras 3
        # caches the layer forward pass, and because our forward is a
        # Python loop over time it gets called once while backward runs per
        # step -- so the gradient is applied to stale activations and the
        # loss saturates at ln(C). Measured with a call counter.
        #
        # from_logits=True is REQUIRED: our output IS logits, but Keras
        # expects probabilities and applies softmax itself. A double
        # softmax measured a loss of 2.4341 against the correct 1.4414,
        # and training settled in a wrong local minimum (5.14 vs 1.17).
        m.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=["accuracy"], run_eagerly=False)
    return m


def _layer_inputs(model: keras.Model, x):
    """Feed one probe batch through the model and capture each layer INPUT.

    Keras 3 offers no supported way to ask a layer what it was fed, and
    threshold calibration needs exactly that.

    We tap the incoming edge of each spiking layer: model.inputs for the
    first one, and that layer's output for every later one.

    The earlier version tapped the graph OUTPUT instead, which is wrong in
    two ways: it measured the wrong tensor (the final logits, not the layer
    input) and it silently zipped fewer values than there were layers. The
    failure surfaced as a matmul shape error deep inside the cell, several
    frames from the actual mistake.
    """
    spiking = [l for l in model.layers if _is_spiking(l)]
    if not spiking:
        return []
    # Each layer's own INPUT edge, not the model input.
    #
    # Using model.inputs looks equivalent and is not: a model with an
    # embedding stem feeds the first spiking layer the EMBEDDING output,
    # while model.inputs is still the integer token tensor. Calibration then
    # measures the wrong tensor and the matmul fails on rank, three frames
    # away from the real mistake.
    taps = [l.input for l in spiking]
    probe = keras.Model(model.inputs, taps)
    outs = probe.predict(x, verbose=0)
    return list(outs) if isinstance(outs, list) else [outs]


def _is_spiking(layer) -> bool:
    """True for our layer, including subclasses defined by users."""
    return isinstance(layer, SpikingRNNCell)


def calibrate_model(model: keras.Model, x, target_rate: float = 0.25,
                    iters: int = 8, verbose: bool = False):
    """Pick a firing rate, then set thresholds so the net actually hits it.

    A fresh net is silent: potentials stay below the default threshold of
    1.0, so nothing spikes and no gradient flows. Calibration is what makes
    a default model trainable at all.

    Returns one (layer_name, firing_rate, dead_fraction) tuple per spiking
    layer so callers can assert on the outcome instead of trusting it.
    """
    spiking = [l for l in model.layers if _is_spiking(l)]
    if not spiking:
        return []
    per_layer = _layer_inputs(model, x)
    out = []
    for layer, inp in zip(spiking, per_layer):
        layer.calibrate(inp, target_rate=target_rate, iters=iters)
        fr, dead = layer.firing_rate(inp)
        out.append((layer.name, fr, dead))
        if verbose:
            print(f"  calibrate: {layer.name}: firing_rate={fr:.4f} "
                  f"dead={dead:.4f} (target {target_rate:.2f})")
    if verbose and not spiking:
        print("  calibrate: no spiking layers found")
    return out


def measure_model(model: keras.Model, x):
    """Read real firing statistics off every spiking layer.

    The planner needs densities so it can budget the worst case over a
    measured range instead of trusting one constant.
    """
    per_layer = _layer_inputs(model, x)
    spiking = [l for l in model.layers if _is_spiking(l)]
    rows = []
    for layer, inp in zip(spiking, per_layer):
        fr, dead = layer.firing_rate(inp)
        rows.append(dict(layer=layer.name, firing=round(fr, 4),
                         dead=round(dead, 4)))
    return rows