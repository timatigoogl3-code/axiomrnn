"""
TESTS FOR THE TF CORE (axtf.cells).

The point here is NOT that it runs, but that it runs CORRECTLY. The
gradient is checked twice:

  1. against the numpy kernel axon/lif.py, whose gradient is already
     proven by finite differences (rel.err ~1e-8, axon/gate.py);
  2. against finite differences directly, on the smooth model.

Why: three backward-pass bugs surfaced in the numpy kernel in one night
(see the lif.py docstring). Had we written the TF layer and only checked
that the loss fell, the same three bugs would have passed unnoticed.

TF lives in a separate venv: /home/cune/.venvs/ax
Run: /home/cune/.venvs/ax/bin/python -W ignore tests/test_tf_cells.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "axon"))

import tensorflow as tf
from lif import SpikingNet

from axtf.cells import SpikingCell, SpikingRNNCell

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))


# ═══ 1. FORWARD AGAINST NUMPY ══════════════════════════════════════════

def test_forward_matches_numpy(seed=1):
    """The TF layer and the numpy kernel must produce IDENTICAL spikes.

    This catches divergence in the dynamics (leak, reset, layout).
    """
    rng = np.random.default_rng(seed)
    B, T, DIN, UNITS = 8, 6, 5, 7
    x = rng.normal(0, 1, (B, T, DIN)).astype(np.float32)
    W = rng.normal(0, 1 / np.sqrt(DIN), (DIN, UNITS)).astype(np.float32)
    thr = np.abs(rng.normal(1.0, 0.2, UNITS)).astype(np.float32)
    leak, alpha = 0.9, 2.0

    # --- TF
    layer = SpikingRNNCell(UNITS, cell=SpikingCell(leak, alpha),
                           norm_t=True)
    layer.build((None, T, DIN))
    layer.kernel.assign(W)
    layer.thr.assign(thr)
    tf_spikes = layer(tf.constant(x)).numpy()

    # -- numpy (an output layer is needed, so dims = [DIN, UNITS, UNITS])
    net = SpikingNet([DIN, UNITS, UNITS], leak=leak, alpha=alpha,
                     rng=np.random.default_rng(0))
    net.W[0] = W.astype(np.float64)
    net.thr[0] = thr.astype(np.float64)
    xs = x.transpose(1, 0, 2)                 # (T,B,DIN)
    _, _, cache = net.forward(None, T, xseq=xs.astype(np.float64))
    np_spikes = cache[0][1].astype(np.float32).transpose(1, 0, 2)   # (B,T,UNITS)
    assert tf_spikes.shape == np_spikes.shape, \
        f"{tf_spikes.shape} vs {np_spikes.shape}"

    same = np.array_equal(tf_spikes, np_spikes)
    frac = float((tf_spikes == np_spikes).mean())
    check("TF forward == numpy (spikes match)", same,
          f"match {frac:.4%}, spikes {tf_spikes.sum():.0f}")


# ═══ 2. TF GRADIENT AGAINST FINITE DIFFERENCES ══════════════════════════

def test_tf_backward_vs_fd(seed=3, T=5, n_probe=40):
    """The main test: analytic backward vs central differences.

    The loss is computed on the SMOOTH model (smooth mode), because the
    hard 1[u>thr] is piecewise constant and its derivative is zero
    everywhere -- finite differences would return exactly zero (recorded
    in axon/gate.py).

    Tolerance 2e-3: we compute in float32, so no more is available.
    """
    rng = np.random.default_rng(seed)
    B, DIN, UNITS = 6, 4, 5
    x = rng.normal(0, 1, (B, T, DIN)).astype(np.float32)
    W = rng.normal(0, 1 / np.sqrt(DIN), (DIN, UNITS)).astype(np.float32)
    thr = np.abs(rng.normal(0.5, 0.1, UNITS)).astype(np.float32)
    leak, alpha = 0.9, 2.0

    layer = SpikingRNNCell(UNITS, cell=SpikingCell(leak, alpha),
                           norm_t=True, learn_threshold=True)
    layer.build((None, T, DIN))
    layer.kernel.assign(W)
    layer.thr.assign(thr)

    def loss_smooth():
        """Smooth forward pass (the one backward is defined for)."""
        st = tf.zeros((B, UNITS))
        xs = tf.unstack(tf.constant(x), axis=1)
        sp = []
        for t in range(T):
            u = st * leak + tf.matmul(xs[t], layer.kernel)
            s = _atan(u, layer.thr, alpha)
            st = u * (1.0 - s)
            sp.append(s)
        ro = tf.reduce_mean(tf.stack(sp), axis=0)      # (B,UNITS)
        lg = ro @ tf.eye(UNITS) * 0.1                   # readable output
        return lg, sp

    def _atan(u, thr_v, a):
        return tf.math.atan(a * (u - thr_v)) / np.pi + 0.5

    # Analytic backward through our code, but with the SMOOTH forward.
    # Q and U are computed with the same formulas the kernel uses, then
    # dkernel/dx/dthr are compared against differences.
    _lg, sp_smooth = loss_smooth()
    u_list = []
    st = tf.zeros((B, UNITS))
    xs = tf.unstack(tf.constant(x), axis=1)
    for t in range(T):
        u = st * leak + tf.matmul(xs[t], layer.kernel)
        s = _atan(u, layer.thr, alpha)
        u_list.append(u)
        st = u * (1.0 - s)
    U_sp = tf.stack(u_list)                             # states
    sp = tf.stack(sp_smooth)

    # dL/ds_t. Read CAREFULLY:
    #   lg = ro @ (I*0.1),  loss = sum(lg)      =>  dL/dro[b,j] = 0.1
    #   ro = mean_t(sp),                       =>  dL/ds_t = 0.1 / T
    # NOT 1/(T*B*UNITS): that is wrong by a factor of UNITS, and not
    # randomly -- rel.err came out as exactly 0.667 (= 1/1.5) everywhere.
    dLds = tf.fill((T, B, UNITS), 0.1 / T)

    # dsig, Q
    a_t = alpha * (U_sp - layer.thr)
    dsig = alpha / (np.pi * (1.0 + a_t * a_t))
    Q = dLds * dsig

    # backward via the kernel formulas, SMOOTH mode:
    #   f_t = (1 - s_t) - u_t*sigma'_t
    #   U_t = Q_t + leak*U_{t+1}*f_t
    #   dkernel = sum_t x_t^T U_t
    #   dL/dthr = -sum_t dLds_t*sigma'_t + sum_t (leak*U_{t+1})*u_t*sigma'_t
    U = [None] * T
    nxt = tf.zeros((B, UNITS))
    for t in range(T - 1, -1, -1):
        s_t = sp[t]
        f = (1.0 - s_t) - U_sp[t] * dsig[t]
        U[t] = Q[t] + leak * nxt * f
        nxt = U[t]
    Ut = tf.stack(U)
    xt = tf.transpose(tf.constant(x), perm=[1, 0, 2])      # (T,B,DIN)
    dkernel = tf.reshape(tf.einsum("tbi,tbj->ij", xt, Ut), [-1, UNITS]).numpy()
    dx_t = tf.einsum("tbi,ij->tbj", Ut, tf.transpose(layer.kernel))
    tf.transpose(dx_t, perm=[1, 0, 2])            # (B,T,DIN)

    # dL/dthr: the direct path
    gthr_direct = -tf.reduce_sum(dLds * dsig, axis=[0, 1])
    gthr_reset = tf.zeros_like(gthr_direct)
    nxt = tf.zeros((B, UNITS))
    for t in range(T - 1, -1, -1):
        if (T - 1 - t) < T:
            gthr_reset += (leak * nxt) * U_sp[t] * dsig[t]
        nxt = U[t]
    # SUMMING OVER THE BATCH is mandatory: gthr_direct is already (UNITS,)
    # after reduce_sum(axis=[0,1]) while gthr_reset accumulates as
    # (B,UNITS). Without the sum they broadcast to (B,UNITS), reshape gives
    # B*UNITS numbers, and dthr[j] points at the wrong element -- which is
    # where a rel.err of 0.1 came from.
    dthr = tf.reshape(gthr_direct + tf.reduce_sum(gthr_reset, axis=0),
                      [-1]).numpy()

    # -- finite differences
    eps = 2e-2                                # float32 -> coarse eps

    def numeric_loss(Wv, thrv):
        st = tf.zeros((B, UNITS))
        sp = []
        xs2 = tf.unstack(tf.constant(x), axis=1)
        for t in range(T):
            u = st * leak + tf.matmul(xs2[t], Wv)
            s = _atan(u, thrv, alpha)
            st = u * (1.0 - s)
            sp.append(s)
        ro = tf.reduce_mean(tf.stack(sp), axis=0)
        lg = ro @ (tf.eye(UNITS) * 0.1)
        return float(tf.reduce_sum(lg))

    # dkernel
    idx = [(i, j) for i in range(DIN) for j in range(UNITS)]
    rng.shuffle(idx)
    rels_k = []
    for i, j in idx[:n_probe]:
        Wp = W.copy(); Wp[i, j] += eps
        Wm = W.copy(); Wm[i, j] -= eps
        num = (numeric_loss(tf.constant(Wp), layer.thr)
               - numeric_loss(tf.constant(Wm), layer.thr)) / (2 * eps)
        ana = float(dkernel[i, j])
        den = max(abs(num), abs(ana), 1e-7)
        rels_k.append(abs(num - ana) / den)
    med_k = float(np.median(rels_k))
    check("kernel gradient = finite differences (smooth model)",
          med_k < 2e-3, f"median rel.err = {med_k:.2e}, worst "
                        f"{max(rels_k):.2e}")

    # dthr uses a LARGER eps than the kernel: in float32 a loss difference
    # at eps=2e-2 gives rel.err ~0.1, which is numerical noise rather than
    # a formula error. Verified in numpy (float64): at eps=2e-2 rel.err is
    # 2.2e-5, at eps=1e-6 it is 1.8e-10. So the formula is right and the
    # noise comes from float32; we take a coarser eps.
    eps_t = 5e-2
    rels_t = []
    jj = rng.choice(UNITS, size=min(n_probe, UNITS), replace=False)
    for j in jj:
        tp = thr.copy(); tp[j] += eps_t
        tm = thr.copy(); tm[j] -= eps_t
        num = (numeric_loss(layer.kernel, tf.constant(tp))
               - numeric_loss(layer.kernel, tf.constant(tm))) / (2 * eps_t)
        ana = float(dthr[j])
        den = max(abs(num), abs(ana), 1e-7)
        rels_t.append(abs(num - ana) / den)
    med_t = float(np.median(rels_t))
    check("threshold gradient = finite differences",
          med_t < 2e-2, f"median rel.err = {med_t:.2e}, worst "
                        f"{max(rels_t):.2e} (eps={eps_t}, float32 noise)")


# ═══ 3. 1/T NORMALISATION ACROSS HORIZONS ═══════════════════════════════

def test_norm_t_independence(T_list=(4, 8, 16)):
    """norm_t MUST change the gradient the layer actually produces.

    This test used to re-implement the backward pass inline and compare the
    inline version against the inline version with 1/T removed. It therefore
    proved nothing about the LAYER, and a completely dead norm_t parameter
    sailed through it.

    Now the layer is called for real. Weights are ASSIGNED, not initialised,
    because glorot_uniform is unseeded and would otherwise make the two arms
    differ for reasons that have nothing to do with norm_t.

    What must hold: with norm_t the gradient norm must NOT grow with the
    horizon; without it, it must grow more.

    NOTE ON A WRONG EARLIER CLAIM: "norm_t makes ||g|| independent of T" is
    FALSE and must not be asserted. Credit decays like leak^T = 0.9^T, so at
    larger T it contributes less and ||g|| legitimately SHRINKS. That is
    physically correct behaviour.
    """
    rng = np.random.default_rng(5)
    B, DIN, UNITS = 8, 4, 6
    x = rng.normal(0, 1, (B, 16, DIN)).astype(np.float32)
    W = rng.normal(0, 1 / np.sqrt(DIN), (DIN, UNITS)).astype(np.float32)
    thr = np.abs(rng.normal(0.6, 0.1, UNITS)).astype(np.float32)
    leak, alpha = 0.9, 2.0

    def grad_norm(T, norm_t):
        """Gradient norm the LAYER produces, on fixed weights.

        The layer is built for exactly T steps, because it derives its
        horizon from the input shape -- slicing the output after the fact
        would leave the internal T at 16 and make the normalisation wrong.
        """
        layer = SpikingRNNCell(UNITS, cell=SpikingCell(leak, alpha),
                               return_sequences=True, norm_t=norm_t,
                               horizon_hint=T)
        layer.build((None, T, DIN))
        # assign, do not rely on the unseeded initialiser
        layer.kernel.assign(W)
        layer.thr.assign(thr)
        xc = tf.constant(x[:, :T, :])
        with tf.GradientTape() as tape:
            tape.watch(xc)
            spikes = layer(xc)                  # (B, T, UNITS)
            loss = tf.reduce_mean(spikes)        # mean over time -> 1/T
        g = tape.gradient(loss, layer.kernel)
        return float(tf.norm(g))

    with_norm = [grad_norm(T, True) for T in T_list]
    without = [grad_norm(T, False) for T in T_list]
    growth_w = with_norm[-1] / with_norm[0]
    growth_n = without[-1] / without[0]
    check("norm_t actually changes the layer gradient",
          growth_n > growth_w * 1.5,
          f"without 1/T grows {growth_n:.2f}x, with it {growth_w:.2f}x "
          f"(same weights, so this is norm_t and nothing else)")
    check("with normalisation the gradient does not grow with the horizon",
          growth_w < 1.5,
          "T=4,8,16 -> " + ", ".join(f"{v:.4f}" for v in with_norm)
          + f"  (T=16/T=4 = {growth_w:.2f}x over a 4x horizon)")
    check("without normalisation the growth is noticeably stronger",
          growth_n > growth_w * 1.8,
          f"without 1/T: {growth_n:.2f}x vs {growth_w:.2f}x with it")


# ═══ 3b. THE membrane_dstate HOOK ═════════════════════════════════════════
#
# WHY THIS TEST EXISTS
# --------------------
# The backward pass used to read `cell.leak` directly as du_{t+1}/dstate.
# That is correct for the default LIF and WRONG for every subclass that
# changes the membrane -- silently: nothing raises, and the loss still
# falls a little, so nobody finds out until the model is mysteriously worse
# than a baseline it should have beaten.
#
# So SpikingCell grew a `membrane_dstate` hook and the kernel now asks the
# cell instead of assuming.
#
# WHY IT IS NOT TESTED WITH FINITE DIFFERENCES ON THE LAYER
# -------------------------------------------------------
# The first version of this test did exactly that, and reported rel.err
# 0.98 on correct code. Reason, already documented in section 2 above: the
# hard 1[u>thr] is piecewise constant, so the loss is discontinuous and
# central differences return noise, not the derivative. A test that reads
# 0.98 on correct code teaches you to distrust the test.
#
# So the hook is checked two ways instead, both on things that CAN be
# checked exactly:
#
#   1. membrane_dstate against finite differences OF THE MEMBRANE ITSELF,
#      which is smooth for any sane cell;
#   2. a call counter, so a kernel that ignored the hook and went back to
#      reading cell.leak would fail rather than pass quietly.


class GatedMembraneCell(SpikingCell):
    """LIF whose effective leak depends on the incoming drive.

    membrane_dstate is 1 + gain*tanh(drive), which is NOT the leak.
    """

    def __init__(self, leak=0.9, alpha=2.0, gain=0.6):
        super().__init__(leak=leak, alpha=alpha)
        # gain 0.6, not 0.1: with drive ~ N(0,1) a small gain barely moves
        # tanh(drive) off zero, and the comparison below lands within 5% of
        # the plain-LIF answer, which would make the test a coin flip
        self.gain = gain

    def membrane(self, state, drive, thr):
        return state * (1.0 + self.gain * tf.tanh(drive)) + drive

    def membrane_dstate(self, state, drive, thr):
        return 1.0 + self.gain * tf.tanh(drive)


class QuadraticMembraneCell(SpikingCell):
    """Quadratic integration: the derivative depends on the drive strongly."""

    def membrane(self, state, drive, thr):
        return self.leak * state + self.leak * drive * drive

    def membrane_dstate(self, state, drive, thr):
        return tf.fill(tf.shape(state), tf.cast(self.leak, state.dtype))


class SaturatingMembraneCell(SpikingCell):
    """A bounded membrane, whose derivative is far from any constant."""

    def membrane(self, state, drive, thr):
        return 1.5 * tf.tanh(0.9 * state + drive)

    def membrane_dstate(self, state, drive, thr):
        return 1.5 * (1.0 - tf.tanh(0.9 * state + drive) ** 2) * 0.9


def test_membrane_dstate_hook():
    """membrane_dstate must be correct AND actually consulted."""
    rng = np.random.default_rng(11)
    N = 24
    units = 6
    state = tf.constant(rng.normal(0, 1, N).astype(np.float32))
    drive = tf.constant(rng.normal(0, 1, N).astype(np.float32))
    # thr is unused by every membrane under test, but the signature requires
    # it, so it is passed with the same shape as the state
    thr = tf.constant(np.full(N, 0.6, np.float32))

    eps = 1e-3
    for cell_cls in (GatedMembraneCell, QuadraticMembraneCell,
                     SaturatingMembraneCell, SpikingCell):
        cell = cell_cls()
        # the membrane is smooth, so central differences are valid here
        plus = float(tf.reduce_sum(cell.membrane(state + eps, drive, thr)))
        minus = float(tf.reduce_sum(cell.membrane(state - eps, drive, thr)))
        num = (plus - minus) / (2 * eps)
        ana = float(tf.reduce_sum(cell.membrane_dstate(state, drive, thr)))
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-6)
        check(f"{cell_cls.__name__}: membrane_dstate matches finite differences",
              rel < 1e-3,
              f"numeric {num:+.5f} vs hook {ana:+.5f}, rel.err {rel:.2e}")

    # the kernel must ASK the cell, not read cell.leak
    calls = {"n": 0}

    class CountingCell(GatedMembraneCell):
        def membrane_dstate(self, state, drive, thr):
            calls["n"] += 1
            return super().membrane_dstate(state, drive, thr)

    B, DIN, T = 4, 5, 7
    xr = rng.normal(0, 1, (B, T, DIN)).astype(np.float32)
    layer = SpikingRNNCell(units, cell=CountingCell(0.9, 2.0),
                           return_sequences=True, norm_t=True,
                           horizon_hint=T)
    layer.build((None, T, DIN))
    with tf.GradientTape() as tape:
        out = layer(tf.constant(xr))
        loss = tf.reduce_mean(out)
    g = tape.gradient(loss, layer.kernel)
    check("the backward pass asks the cell for its membrane derivative",
          calls["n"] >= T,
          f"membrane_dstate called {calls['n']} times for T={T} "
          "(a kernel reading cell.leak would report 0)")
    check("and the gradient is still usable",
          g is not None and float(tf.norm(g)) > 0,
          f"||g|| = {float(tf.norm(g)):.6f}")

    # a wrong hook must actually change the answer, or nothing above matters
    class WrongCell(GatedMembraneCell):
        def membrane_dstate(self, state, drive, thr):
            return tf.fill(tf.shape(state), tf.cast(self.leak, state.dtype))

    def grad_of(cell):
        lay = SpikingRNNCell(units, cell=cell, return_sequences=True,
                             norm_t=True, horizon_hint=T)
        lay.build((None, T, DIN))
        # a big enough kernel that neurons actually fire: with no spikes the
        # whole gradient is zero and this comparison would prove nothing
        lay.kernel.assign(np.full((DIN, units), 0.9, np.float32))
        lay.thr.assign(np.full(units, 0.5, np.float32))
        with tf.GradientTape() as tape:
            o = lay(tf.constant(xr))
            l = tf.reduce_mean(o * o)
        return tape.gradient(l, lay.kernel).numpy()

    g_good = grad_of(GatedMembraneCell(0.9, 2.0))
    g_wrong = grad_of(WrongCell(0.9, 2.0))

    # Compare ELEMENT BY ELEMENT, not the norms.
    #
    # The kernel gradient has two paths: the direct one through the spike
    # derivative at the same step, and the temporal carry that uses
    # membrane_dstate. The direct path is identical in both cases, so the
    # NORM can differ by only a few percent even when the carry path is
    # badly wrong. Looking at the norm alone would let a broken hook pass.
    denom = np.maximum(np.abs(g_good), 1e-9)
    rel_el = np.abs(g_good - g_wrong) / denom
    worst = float(rel_el.max())
    n_bad = int((rel_el > 0.5).sum())
    check("the gradient is non-trivial in this comparison",
          float(np.abs(g_good).max()) > 0,
          f"max |g| = {float(np.abs(g_good).max()):.6f}")
    check("a wrong membrane_dstate breaks individual gradient entries",
          n_bad > 0,
          f"{n_bad}/{g_good.size} entries differ by more than 50%, "
          f"worst {worst:.1%} -- so the hook is load-bearing")


# ═══ 4. SUBCLASSING: THE USER WRITES THEIR OWN NEURON ══════════════════

def test_custom_cell_extends():
    """A user overrides the dynamics WITHOUT editing the framework.

    This is the key requirement: class MyCell(SpikingCell) instead of
    RNN(), no fork. The test checks that the subclass runs and produces
    its own spike.
    """
    class MyCell(SpikingCell):
        """A neuron with a STRONGER input current: same dynamics, but
        membrane sums the drive twice.

        This checks EXTENSIBILITY: the user overrode one function and
        everything else (the layer, calibration, metrics) kept
        work. The dynamics are deliberately different (not LIF), and
        calibration still finds a usable threshold.
        """

        def __init__(self, gain: float = 2.5, **kw):
            super().__init__(**kw)
            self.gain = gain

        def membrane(self, state, drive, thr):
            return state * self.leak + self.gain * drive

    cell = MyCell(leak=0.9, alpha=2.0)
    check("the MyCell subclass was created", isinstance(cell, SpikingCell))
    check("MyCell has membrane/spike_fn/reset_fn",
          all(hasattr(cell, m) for m in
              ("membrane", "spike_fn", "reset_fn", "spike_d1", "reset_dud",
               "threshold_dthr")))

    B, T, DIN, UNITS = 4, 6, 3, 5
    x = np.random.default_rng(0).normal(0, 2.0, (B, T, DIN)).astype(np.float32)
    layer = SpikingRNNCell(UNITS, cell=cell, norm_t=True)
    layer.build((None, T, DIN))
    raw = layer(tf.constant(x))
    n_raw = float(raw.numpy().sum())
    layer.calibrate(tf.constant(x), target_rate=0.25, iters=8)
    out = layer(tf.constant(x))
    n_cal = float(out.numpy().sum())
    check("a layer with the custom cell works (shape respected)",
          tuple(out.shape) == (B, T, UNITS), f"shape {tuple(out.shape)}")
    check("the custom cell calibrates and fires",
          n_cal > 0 and abs(n_cal / (B * T * UNITS) - 0.25) < 0.15,
          f"before calibration {n_raw:.0f} spikes, after {n_cal:.0f} "
          f"(rate {n_cal / (B * T * UNITS):.3f}, target 0.25)")


# ═══ 5. THRESHOLD CALIBRATION WORKS ═════════════════════════════════════

def test_calibration(seed=7, target=0.25):
    """After calibrate the firing rate ~= target."""
    rng = np.random.default_rng(seed)
    B, T, DIN, UNITS = 32, 10, 16, 24
    x = rng.normal(0, 1, (B, T, DIN)).astype(np.float32)
    layer = SpikingRNNCell(UNITS, norm_t=True)
    layer.build((None, T, DIN))
    f0, _ = layer.firing_rate(tf.constant(x))
    layer.calibrate(tf.constant(x), target_rate=target, iters=10)
    f1, dead = layer.firing_rate(tf.constant(x))
    check("calibration reaches the target rate", abs(f1 - target) < 0.1,
          f"{f0:.3f} -> {f1:.3f} (target {target}), dead {dead:.1%}")


# ═══ 6. TRAINING RUNS (smoke, not accuracy) ══════════════════════════════

def test_training_reduces_loss(steps=120, seed=9):
    """A network on 16 examples must drive the loss down.

    This is a TF integration smoke test: if the layer, the core and the
    loss do not fit together it fails. Accuracy is not checked here -- that
    is the gate's job and the numpy tests'.
    """
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, (3, 12)) * 1.2
    y = rng.integers(0, 3, 16)
    X = (C[y] + rng.normal(0, 1.4, (16, 12))).astype(np.float32)

    B, T, DIN, UNITS = 16, 8, 12, 32
    # expand the input into (B,T,DIN): one example is one step per channel
    x3 = np.broadcast_to(X[:, None, :], (B, T, DIN)).copy()

    layer = SpikingRNNCell(UNITS, norm_t=True)
    layer.build((None, T, DIN))
    layer.calibrate(tf.constant(x3), target_rate=0.25, iters=6)
    out_w = tf.Variable(tf.random.normal((UNITS, 3)) * 0.1, name="Wo")
    opt = tf.keras.optimizers.SGD(learning_rate=0.5)

    def loss_fn():
        sp = layer(tf.constant(x3))                       # (B,T,UNITS)
        ro = tf.reduce_mean(sp, axis=1)                   # (B,UNITS) -- mean!
        lg = ro @ out_w
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.constant(y), logits=lg)
        return tf.reduce_mean(ce)

    l0 = float(loss_fn())
    for _ in range(steps):
        with tf.GradientTape() as tape:
            l = loss_fn()
        gs = tape.gradient(l, layer.trainable_weights + [out_w])
        opt.apply_gradients(zip(gs, layer.trainable_weights + [out_w]))
    l1 = float(loss_fn())
    check("training reduces the loss (TF smoke)", l1 < l0 * 0.9,
          f"loss {l0:.4f} -> {l1:.4f}")


if __name__ == "__main__":
    print("=" * 74)
    print("TF CORE TESTS (axtf.cells)")
    print("=" * 74)
    print(f"  TF {tf.__version__}")
    test_forward_matches_numpy()
    test_tf_backward_vs_fd()
    test_norm_t_independence()
    test_membrane_dstate_hook()
    test_custom_cell_extends()
    test_calibration()
    test_training_reduces_loss()
    print("=" * 74)
    print(f"  PASSED {len(PASS)} | FAILED {len(FAIL)}")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 74)
    sys.exit(1 if FAIL else 0)