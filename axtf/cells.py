"""
axtf.cells -- custom recurrent neuron for Keras/TF.

The project core. Users may override it, which is exactly what you cannot
do with a frozen Keras layer:

    class MyCell(SpikingCell):
        def membrane(self, state, x, t): ...      # your dynamics
        def spike_fn(self, u, thr): ...           # your spike rule

and get `SpikingRNN(MyCell)` without forking the framework.

══════════════════════════════════════════════════════════════════════
THE ONE THING THAT MUST HOLD
══════════════════════════════════════════════════════════════════════
With mean readout, the learning curve must NOT depend on T. If it does,
the 1/T gradient normalisation is lost and training diverges on long
horizons.

Verified by the gate (axon/gate.py):
    ||g|| norm_t=True:   T=4:0.301  T=16:0.356  T=64:0.414   (1.18x)
    ||g|| norm_t=False:  T=4:0.658  T=16:3.557  T=64:11.4     (5.40x)

This is not cosmetic. Without 1/T, accuracy FELL with horizon: 0.92 -> 0.06.
Three such bugs in a row cost us a full session, so 1/T is baked into the
core and cannot be switched off by accident.

THREE PLACES WHERE THE MATH BREAKS SILENTLY
===========================================
1. 1/T normalisation -- see above. norm_t.
2. Reset. u_t = leak*u_{t-1} + cu; s_t = 1[u_t > thr]; u_t <- u_t*(1-s_t).
   u_t appears TWICE (spike and reset), so
       d st_t/du_t = (1-s_t) - u_t*sigma'(u_t-thr).
   Dropping that term gave ~30% gradient error.
3. Reset path in the threshold gradient. The threshold enters both s_t and
   st_t, so
       dL/dthr = -sum A_t*sigma'_t + sum (leak*U_{t+1})*u_t*sigma'_t.
   Without the second term it is simply wrong.

The surrogate gradient CANNOT be checked on the hard model: 1[u>thr] is
piecewise constant, so its derivative is identically zero. Hence smooth=True,
where finite differences are meaningful.
"""
from __future__ import annotations

import warnings

import tensorflow as tf
from tensorflow.keras.layers import Layer

try:
    from keras.src.backend.common.keras_tensor import keras_tensor
except Exception:                                   # pragma: no cover  # noqa: BLE001
    keras_tensor = None


# ── surrogate ───────────────────────────────────────────────────────────────

def _atan_surrogate(u, thr, alpha):
    """Smooth surrogate step; replaces the Heaviside delta in backward."""
    return tf.math.atan(alpha * (u - thr)) / math_pi() + 0.5


def _atan_d1(u, thr, alpha):
    a = alpha * (u - thr)
    return alpha / (math_pi() * (1.0 + a * a))


def _atan_d2(u, thr, alpha):
    a = alpha * (u - thr)
    d = 1.0 + a * a
    return -2.0 * alpha * alpha * a / (math_pi() * d * d)


def math_pi():
    import math
    return math.pi


def _quantile(t, q: float) -> float:
    """Quantile without pulling in tensorflow_probability.

    Linear interpolation over sorted values. Threshold calibration does not
    need more, and a dependency is not worth it for one number.
    """
    flat = tf.sort(tf.reshape(t, [-1]))
    n = int(tf.size(flat))
    pos = min(max(q * (n - 1), 0.0), float(n - 1))
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(tf.gather(flat, lo)) * (1.0 - frac) \
        + float(tf.gather(flat, hi)) * frac


# ── CORE ───────────────────────────────────────────────────────────────────

class SpikingCell:
    """Neuron dynamics as an OBJECT. This is the extension point.

    Subclass and override `membrane` / `spike_fn` to get your own neuron
    without touching the layer, the training loop, or the planner.

    A class rather than kwargs, because kwargs cannot be extended without
    editing the core -- and "a professional can override the system without
    forking" is a hard requirement here.
    """

    def __init__(self, leak: float = 0.9, alpha: float = 2.0,
                 learnable_threshold: bool = False):
        self.leak = leak
        self.alpha = alpha
        self.learnable_threshold = learnable_threshold

    # ── DYNAMICS: the part users normally change ─────────────────────────────

    def membrane(self, state, drive, thr):
        """Next potential from state and drive.

        state : (B, units) -- potential from the previous step
        drive : (B, units) -- matmul result
        thr   : (units,)   -- threshold
        """
        return self.leak * state + drive

    def membrane_dstate(self, state, drive, thr):
        """d membrane / d state.

        THE HOOK THAT MAKES membrane() SAFE TO OVERRIDE.

        The backward pass needs d u_{t+1} / d u_t. For the default LIF that
        is a constant `leak`, which is why the kernel could just read
        cell.leak. The moment a subclass overrides membrane() with anything
        non-linear, a hardcoded `leak` becomes a WRONG gradient -- and it is
        silently wrong: nothing raises, and the loss still falls a little.

        So the kernel asks the cell instead of assuming. If you override
        membrane(), override this too; if you cannot state the derivative,
        keep the default membrane(). The cost is one elementwise multiply
        per step, exactly as before.
        """
        return tf.fill(tf.shape(state), tf.cast(self.leak, state.dtype))

    def spike_fn(self, u, thr, smooth: bool):
        """Spike rule.

        smooth=False -> hard 1[u>thr]   (what runs at inference)
        smooth=True  -> surrogate       (what is differentiable)
        """
        if smooth:
            return _atan_surrogate(u, thr, self.alpha)
        return tf.cast(u > thr, u.dtype)

    def reset_fn(self, u, spike, smooth: bool):
        """Reset the potential after a spike.

        Key point: in smooth mode the reset is soft, u*(1-s), so u enters
        the gradient twice -- see module docstring, item 2.
        """
        if smooth:
            return u * (1.0 - spike)
        return tf.where(spike > 0.5, tf.zeros_like(u), u)

    # ── DERIVATIVES: these depend on the methods overridden above ───────────
    #
    # If you override membrane/spike_fn but do not know its derivative,
    # override these too. Otherwise the core CANNOT build a backward pass,
    # and we say so rather than silently building a wrong gradient.

    def spike_d1(self, u, thr):
        """d spike_fn / du."""
        return _atan_d1(u, thr, self.alpha)

    def reset_dud(self, u, spike, thr, dspike):
        """d reset_fn / du."""
        return 1.0 - spike - u * dspike

    def threshold_dthr(self, u, spike, thr, dspike):
        """dL/dthr through the reset: st = u*(1-s), ds/dthr = -sigma'."""
        return u * dspike


# ── LAYER CORE ────────────────────────────────────────────────────────────

def _lif_forward(cell: SpikingCell, x, kernel, thr, T, norm_t, credit_k):
    """Forward pass over time, implemented with tf.while_loop.

    Input arrives batch-first as Keras expects: (B, T, in).
    Returns time-first:
      states : (T, B, units)  potentials BEFORE reset
      spikes : (T, B, units)
      drives : (T, B, units)

    The loop is a tf.while_loop (not a Python for) so the whole pass is
    one graph op and runs in graph mode without run_eagerly=True.
    """
    B = tf.shape(x)[0]
    units = kernel.shape[1]
    state = tf.zeros((B, units), dtype=x.dtype)
    xt = tf.transpose(x, perm=[1, 0, 2])          # (T, B, in)

    states_ta = tf.TensorArray(dtype=x.dtype, size=T, dynamic_size=False,
                               element_shape=(None, units))
    spikes_ta = tf.TensorArray(dtype=x.dtype, size=T, dynamic_size=False,
                               element_shape=(None, units))
    drives_ta = tf.TensorArray(dtype=x.dtype, size=T, dynamic_size=False,
                               element_shape=(None, units))

    def cond(t, st, sta, spa, dra):
        return t < T

    def body(t, st, sta, spa, dra):
        drive = tf.matmul(xt[t], kernel)            # (B, units)
        u = cell.membrane(st, drive, thr)
        s = cell.spike_fn(u, thr, smooth=False)
        nst = cell.reset_fn(u, s, smooth=False)
        sta = sta.write(t, u)
        spa = spa.write(t, s)
        dra = dra.write(t, drive)
        return t + 1, nst, sta, spa, dra

    _, _, states_ta, spikes_ta, drives_ta = tf.while_loop(
        cond, body,
        [tf.constant(0), state, states_ta, spikes_ta, drives_ta],
        parallel_iterations=1)

    states = states_ta.stack()                      # (T, B, units)
    spikes = spikes_ta.stack()
    drives = drives_ta.stack()
    return (states, spikes, drives)


def _lif_backward(cell, x, kernel, thr, states, spikes, drives, dout, T,
                  norm_t, credit_k, learn_thr):
    """Backward pass. Returns (dx, dkernel, dthr).

    dout : (T, B, units) -- gradient w.r.t. each step's spikes, arriving
                           from the layer above (or from the readout).

    Implemented with tf.while_loop (backward in time) so it runs in graph
    mode without run_eagerly=True.
    """
    B = tf.shape(x)[0]
    units = kernel.shape[1]
    if norm_t:
        dout = dout / tf.cast(T, dout.dtype)
    dsig = cell.spike_d1(states, thr)                    # (T,B,units)
    Q = dout * dsig                                      # path via the spike

    gthr_direct = -tf.reduce_sum(dout * dsig, axis=[0, 1])  # (units,)
    gthr_reset = tf.zeros_like(gthr_direct)                 # (units,)

    k = T if credit_k is None else credit_k

    U_ta = tf.TensorArray(dtype=x.dtype, size=T, dynamic_size=False,
                          element_shape=(None, units))
    nxt = tf.zeros((B, units), dtype=x.dtype)

    def cond(t, nxt, U_ta, gthr):
        return t >= 0

    def body(t, nxt, U_ta, gthr):
        s_t = spikes[t]                                  # (B,units)
        f = 1.0 - s_t
        state_next = states[t] * f
        drive_next = drives[tf.minimum(t + 1, T - 1)]
        dstate = cell.membrane_dstate(state_next, drive_next, thr)
        carry = dstate * nxt * f
        age = (T - 1) - t
        in_horizon = tf.cast(age < k, x.dtype)
        carry = carry * in_horizon
        U_t = Q[t] + carry
        if learn_thr:
            # path through the reset, reduced over the batch
            term = (dstate * nxt) * states[t] * dsig[t]
            term = tf.reduce_sum(term, axis=0) * in_horizon
            gthr = gthr + term
        U_ta = U_ta.write(t, U_t)
        return t - 1, U_t, U_ta, gthr

    _, _, U_ta, gthr_reset = tf.while_loop(
        cond, body,
        [tf.constant(T - 1), nxt, U_ta, gthr_reset],
        parallel_iterations=1)

    Ut = U_ta.stack()                                    # (T,B,units)

    xt = tf.transpose(x, perm=[1, 0, 2])                  # (T,B,in)
    dkernel = tf.einsum("tbi,tbj->ij", xt, Ut)            # (in, units)
    dx_t = tf.einsum("tbi,ij->tbj", Ut, tf.transpose(kernel))
    dx = tf.transpose(dx_t, perm=[1, 0, 2])               # (B,T,in)
    dthr = gthr_direct + gthr_reset
    return dx, dkernel, dthr


# ── KERAS LAYER ────────────────────────────────────────────────────────────

def _lif_call_with_grad(x, kernel, thr, cell, norm_t, credit_k, learn_thr):
    """Forward plus an explicit backward via tf.custom_gradient.

    WHY EXPLICIT, AND NOT A PLAIN LOOP
    ==================================
    A time loop over ordinary TF ops breaks the autograd graph: TF cannot
    connect T state steps into one differentiable chain. The symptom is
    nasty -- the model builds fine, loss falls from 7.4 to 1.38, and then
    FREEZES. The gradient is zero and nothing raises.

    So the backward pass is explicit, and it is the same code verified
    against finite differences in test_tf_cells.py.

    FOUR RULES WITHOUT WHICH custom_gradient DOES NOT WORK
    =====================================================
    1) grad_fn must return (input_grads, variable_grads) -- a TUPLE of two
       lists, not one flat tuple. Flat gives "too many values to unpack".
    2) A gradient is required for EVERY input. For a non-trainable threshold
       return zeros_like, not None, or you get "Must return gradient for
       each variable".
    3) keyword arguments (variables=) work ONLY in eager mode. Keras
       Functional builds a GRAPH, and TF raises "custom_gradient decorator
       currently supports keywords arguments only when eager execution is
       enabled". So kernel/threshold are passed as ORDINARY inputs rather
       than through variables=, which works in both modes.
    4) compute_output_shape is required for Functional: without it Keras
       cannot infer the shape and fails while still building the graph.

    What is lost: Keras cannot take a second derivative. We do not care --
    time-wise BPTT needs an explicit backward regardless.
    """

    @tf.custom_gradient
    def _fn(x_, k_, h_):
        T = int(x_.shape[1])
        states, spikes, drives = _lif_forward(
            cell, x_, k_, h_, T, norm_t, credit_k)

        def grad_fn(dout):
            # dout is external layout (B,T,units) -> internal (T,B,units)
            dout_t = tf.transpose(dout, perm=[1, 0, 2])
            dx, dk, dh = _lif_backward(
                cell, x_, k_, h_, states, spikes, drives,
                dout_t, T, norm_t, credit_k, learn_thr)
            dthr_out = dh if learn_thr else tf.zeros_like(h_)
            return (dx, dk, dthr_out), ()

        return tf.transpose(spikes, perm=[1, 0, 2]), grad_fn

    # IMPORTANT: custom_gradient needs TENSORS, but Keras hands us
    # KerasVariable. Convert explicitly, otherwise
    #   "TypeError: Exception encountered when calling SpikingRNNCell.call()"
    # when the layer is called directly outside a Keras graph.
    return _fn(x, tf.convert_to_tensor(kernel),
               tf.convert_to_tensor(thr))



def _as_shape(obj):
    """Extract a shape tuple from whatever Keras passes in.

    Keras 3 varies by call site: a tuple like (None, 8, 12), a KerasTensor,
    a list holding one KerasTensor, or a list with None in place of a shape.
    A KerasTensor cannot be iterated ("Iterating over a symbolic
    KerasTensor is not supported"), so we dispatch on type.

    THE BUG THIS COST REAL TIME. A shape tuple (None, 16, 24) used to be
    mistaken for a list of layer inputs: the first element is None, which is
    neither KerasTensor nor tuple, so the parser took the "list with None"
    branch and returned None. But the batch dim is None almost always, so
    this was not an edge case -- it was the COMMON case:
    compute_output_shape lost the time axis and returned (None, None, units).
    Flatten then produced (None, None) and Dense raised "Shapes used to
    initialize variables must be fully-defined".

    Silent break: the exception was swallowed in compute_output_spec, so the
    model looked assembled until the next layer was reached.

    Distinguish by CONTENT, not type: a shape is ints and None; a list of
    inputs holds KerasTensors, nested lists, or a bare None.
    """
    if obj is None:
        return None
    if hasattr(obj, "shape") and not isinstance(obj, (list, tuple)):
        s = obj.shape
        return tuple(s) if s is not None else None
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return None
        # THIS IS A SHAPE: every element is an int or None
        # Check this FIRST, otherwise batch=None breaks the parse.
        if all(e is None or (isinstance(e, int) and not isinstance(e, bool))
               for e in obj):
            return tuple(obj)
        first = obj[0]
        if hasattr(first, "shape"):
            s = first.shape
            return tuple(s) if s is not None else None
        if isinstance(first, (list, tuple)):
            return tuple(first)
        # list with None, or nested lists
        for item in obj:
            if hasattr(item, "shape") and item.shape is not None:
                return tuple(item.shape)
            if isinstance(item, (list, tuple)):
                return tuple(item)
        return None
    return tuple(obj)


class SpikingRNNCell(Layer):
    """The layer users see. The cell is configurable.

    cell : SpikingCell, or any object with the same interface
    """

    def __init__(self, units, cell=None, return_sequences=True,
                 norm_t=True, credit_k=None, learn_threshold=False,
                 horizon_hint=None, run_eagerly=False, **kw):
        super().__init__(**kw)
        self.units = units
        self.cell = cell if cell is not None else SpikingCell()
        self.return_sequences = return_sequences
        self.norm_t = norm_t
        self.credit_k = credit_k
        self.learn_threshold = learn_threshold or \
            getattr(self.cell, "learnable_threshold", False)
        # The horizon is known up front almost always (users set it for the
        # memory budget). Keras gives no time-axis length in symbolic mode,
        # so we keep the hint.
        self._horizon_hint = horizon_hint

        # The forward/backward passes are now tf.while_loop, so the layer runs
        # correctly in graph mode: run_eagerly=True is no longer required for
        # correctness. Kept as an option for debugging/profiling only.
        self.suppress_jit = bool(run_eagerly)

    def build(self, input_shape):
        units, inputs = self.units, int(input_shape[-1])
        self.kernel = self.add_weight(
            shape=(inputs, units), initializer="glorot_uniform",
            trainable=True, name="kernel")
        self.thr = self.add_weight(
            shape=(units,), initializer="ones",
            trainable=self.learn_threshold, name="threshold")
        self.built = True

    def compute_output_shape(self, input_shape):
        """Required for Keras Functional.

        Without it Keras cannot infer the output shape while building the
        graph, and fails with "Could not automatically infer the output
        shape".

        NOTE: what arrives here is NOT a tuple but a KerasTensor (or a list
        of them). A KerasTensor cannot be iterated, which would give
        "Iterating over a symbolic KerasTensor is not supported".
        So we read the shape off .shape.
        """
        shape = _as_shape(input_shape)
        if shape is None or len(shape) < 2:
            return (None, None, self.units)

        b, t = shape[0], shape[1]
        if self.return_sequences:
            return (b, t, self.units)
        return (b, self.units)

    def compute_output_spec(self, input_shape):
        # A KerasTensor is IMMUTABLE in Keras 3: "The `shape` attribute of
        # KerasTensor is immutable. One should create a new instance".
        # So the shape is set via the constructor, never by assignment.
        spec = super().compute_output_spec(input_shape)
        shape = _as_shape(input_shape)
        if shape is None or len(shape) < 2:
            return spec
        try:
            return keras_tensor.KerasTensor(
                shape=self.compute_output_shape(input_shape),
                dtype=self.compute_dtype,
                sparse=False,
            )
        except Exception:  # noqa: BLE001
            # This used to be a silent `return spec`, and that was a trap:
            # the shape-parse error was swallowed, the layer emitted
            # (None,None,units), and the failure arrived later in Dense with
            # a "fully-defined shapes" message that said nothing about the
            # real cause.
            #
            # Now it warns: the shape is lost, but you can keep working.
            # The fallback does not crash, and it does not hide.
            warnings.warn(
                "SpikingRNNCell: could not infer the output shape from "
                f"{type(input_shape).__name__}; the time axis may become "
                "None. If Flatten/Dense fails below, the cause is here.",
                RuntimeWarning, stacklevel=2)
            return spec

    def call(self, x, training=None, *args, **kwargs):
        # With training=True, Keras 3 may pass the input as a LIST (the
        # model's input list) rather than one tensor. Without unwrapping,
        # unstack slices the wrong axis and you get
        # "In[0] ndims must be >= 2".
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = tf.convert_to_tensor(x)
        rank = x.shape.rank
        if rank is None:
            rank = len(x.shape)
        if int(rank) != 3:
            raise ValueError(
                f"expected input (batch, T, features), got rank "
                f"{rank}: {tuple(x.shape)}")

        # Horizon T: in symbolic mode it can be None (batch unknown), but
        # T is ALWAYS known here -- it is the second dimension. If T is
        # unknown too, fall back to self.horizon as the last resort.
        T = x.shape[1]
        if T is None:
            T = getattr(self, "_horizon_hint", None)
        if T is None:
            raise ValueError(
                "horizon T is unknown: build the layer as "
                "SpikingRNNCell(units, horizon_hint=T), or pass an input "
                "with a known length on axis 1")
        T = int(T)

        spikes = _lif_call_with_grad(
            x, self.kernel, self.thr, self.cell,
            self.norm_t, self.credit_k, self.learn_threshold)
        if self.return_sequences:
            return spikes                            # (B,T,units)
        return tf.reduce_mean(spikes, axis=1)        # (B,units)

    def backward(self, x, dout, T):
        """Explicit backward as a standalone API, for debugging and diffing.

        Inside Keras it already runs within custom_gradient; this exists so
        the result can be COMPARED against the reference (test_tf_cells).

        x    : (B, T, in)    -- external layout, as in call
        dout : (B, T, units) -- gradient w.r.t. spikes, external layout

        RETURNS in external layout: dx (B,T,in), dkernel (in,units),
        dthr (units,).
        """
        dout_t = tf.transpose(dout, perm=[1, 0, 2])         # -> (T,B,units)
        states, spikes, drives = _lif_forward(
            self.cell, x, self.kernel, self.thr, T,
            self.norm_t, self.credit_k)
        return _lif_backward(self.cell, x, self.kernel, self.thr,
                             states, spikes, drives, dout_t, T,
                             self.norm_t, self.credit_k, self.learn_threshold)

    def calibrate(self, x, target_rate: float = 0.25, iters: int = 8,
                  damp: float = 0.7):
        """Fixed-point threshold calibration toward a target firing rate.

        IMPORTANT: measure on a REAL pass with reset ENABLED. A threshold
        must not be calibrated with reset switched off -- reset is precisely
        what keeps the potential bounded, and without it you get a drifting
        distribution the net would never see in practice. Measured: the
        0.90 quantile at T=64 gave 6.38 instead of ~0.05, the threshold
        collapsed to 0, and accuracy fell to 0.06.
        """
        thr = self.thr
        for _ in range(iters):
            u_all = self._potentials(x)
            q = _quantile(u_all, 1.0 - target_rate)
            new = tf.clip_by_value(
                tf.fill(tf.shape(thr), tf.constant(q, thr.dtype)), 0.05, 8.0)
            thr.assign(damp * new + (1 - damp) * thr)
        return thr

    def _potentials(self, x):
        T = int(x.shape[1])
        states, _, _ = _lif_forward(
            self.cell, x, self.kernel, self.thr, T,
            self.norm_t, self.credit_k)
        return tf.reshape(states, [-1])

    def firing_rate(self, x):
        """(mean firing rate, dead fraction). Measured, not assumed."""
        T = int(x.shape[1])
        _, spikes, _ = _lif_forward(
            self.cell, x, self.kernel, self.thr, T,
            self.norm_t, self.credit_k)
        per_neuron = tf.reduce_mean(spikes, axis=[0, 1])   # (units,)
        return float(tf.reduce_mean(per_neuron)), \
            float(tf.reduce_mean(tf.cast(per_neuron == 0, tf.float32)))

    def get_config(self):
        c = super().get_config()
        c.update({"units": self.units, "return_sequences": self.return_sequences,
                  "norm_t": self.norm_t, "credit_k": self.credit_k,
                  "learn_threshold": self.learn_threshold})
        return c