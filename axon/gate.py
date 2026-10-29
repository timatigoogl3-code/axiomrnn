"""
axon.gate -- THE GATE. Nothing gets published until this is green.

Why this file exists: in one session I produced confident nonsense three
times in a row, and every time the bug was in the TEST RIG rather than the
formulas:
  - threshold calibrated with reset switched off -> "accuracy falls with T"
  - gradient not normalised by 1/T               -> "accuracy falls with T"
  - (a third one, an earlier run, also in the rig)

The checks below catch that class in seconds. Run:
    python3 axon/gate.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lif import SpikingNet

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))


# ═══ 1. GRADIENT vs FINITE DIFFERENCES ═══════════════════════════════════

def test_finite_difference(n_probe=80, seed=3):
    """Analytic gradient against central finite differences.

    IMPORTANT, AND THIS WAS A BUG IN THE FIRST GATE: checking the
    surrogate gradient against the HARD forward pass is meaningless --
    1[u>thr] is piecewise constant, so its derivative is zero everywhere
    except at the kinks. Finite differences return EXACTLY 0 for any eps,
    and any correct implementation would "fail".

    The right thing to check is the SMOOTH model (smooth=True): there
    s = sigma(u-thr), the reset is soft, the function is smooth, and our
    backward pass is its true gradient. That is what we check.

    This is not a technicality but a substantive fact about the method: a
    surrogate gradient is NOT the derivative of the hard function (and
    cannot be), it is the gradient of the smoothing. Any check of a
    surrogate must therefore run on the smooth model.

    Catches: a forgotten 1/T, a wrong sign in the reset, a wrong surrogate,
    non-recursive accumulation over time, an error in dL/dthr.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (12, 9))
    y = rng.integers(0, 4, 12)
    T = 5
    # IMPORTANT: credit_k=None. The derivative checked against finite
    # differences is that of FULL BPTT. A truncated credit horizon is
    # deliberately NOT the derivative of any function (it cuts the graph),
    # so comparing it against differences is meaningless. Truncation is
    # checked separately, in test_credit_truncation.
    net = SpikingNet([9, 16, 7, 4], rng=rng, learn_thr=True, credit_k=None)
    for l in range(net.L):
        net.thr[l] = np.full(net.dims[l + 1], 0.37)

    # the smooth and hard losses must be close, otherwise we are checking
    # the wrong object
    lh = net.loss(X, y, T, smooth=False)
    ls = net.loss(X, y, T, smooth=True)
    check("smoothing does not distort the loss",
          abs(lh - ls) / max(abs(lh), 1e-9) < 0.10,
          f"hard {lh:.6f} vs smooth {ls:.6f}")

    v0 = net.params()
    _, g = net.grads(X, y, T, norm_t=True, learn_thr=True, smooth=True)

    kinds = []
    for l in range(net.L):
        kinds += ["W"] * net.W[l].size + ["thr"] * net.thr[l].size
    kinds += ["W"] * net.Wo.size
    assert len(kinds) == len(v0), "coordinate labels disagree with params()"

    eps = 1e-6
    rels = {"W": [], "thr": []}
    idx = rng.choice(len(v0), size=n_probe, replace=False)
    for i in idx:
        vp = v0.copy(); vp[i] += eps
        vm = v0.copy(); vm[i] -= eps
        net.set_params(vp); lp = net.loss(X, y, T, smooth=True)
        net.set_params(vm); lm = net.loss(X, y, T, smooth=True)
        net.set_params(v0)
        num = (lp - lm) / (2 * eps)
        den = max(abs(num), abs(g[i]), 1e-9)
        rels[kinds[i]].append(abs(num - g[i]) / den)

    w_rel = float(np.median(rels["W"])); w_max = float(np.max(rels["W"]))
    t_rel = float(np.median(rels["thr"])); t_max = float(np.max(rels["thr"]))
    check("weight gradient = derivative of the smooth model",
          w_rel < 1e-6, f"median {w_rel:.2e}, worst {w_max:.2e}, "
                        f"{len(rels['W'])} points")
    check("threshold gradient = derivative of the smooth model",
          t_rel < 1e-5, f"median {t_rel:.2e}, worst {t_max:.2e}, "
                         f"{len(rels['thr'])} points")

    # -- explicit demonstration of why the hard mode is uncheckable ----
    hard = 0
    for i in rng.choice(len(v0), size=20, replace=False):
        vp = v0.copy(); vp[i] += eps
        vm = v0.copy(); vm[i] -= eps
        net.set_params(vp); lp = net.loss(X, y, T, smooth=False)
        net.set_params(vm); lm = net.loss(X, y, T, smooth=False)
        net.set_params(v0)
        if abs((lp - lm) / (2 * eps)) == 0.0:
            hard += 1
    check("the hard forward pass is fundamentally uncheckable by differences",
          hard >= 18, f"{hard}/20 coordinates gave exactly 0 -- "
                      f"the function is piecewise constant")


# ═══ 2. T INVARIANCE ═════════════════════════════════════════════════════

def test_t_invariance(seed=5):
    """1/T NORMALISATION: ||g|| must not grow with the horizon.

    This is a direct, decisive check of bug 15, not an indirect one.

    Why: with mean readout (readout = 1/T * sum_t s_t) the maths requires
    dL/ds_t = (1/T) * dL/dreadout. The backward pass sums contributions over
    T steps, so without 1/T the gradient grows LINEARLY in T. At a fixed
    learning rate a larger T means a T-times larger step and training
    diverges. That is where "accuracy falls with T: 0.92 -> 0.06" came from.

    We measure ||g|| on THE SAME weights at different T:
      norm_t=True  -> ||g||(T) is roughly constant
      norm_t=False -> ||g||(T) grows like T

    No training, no calibration -- a pure property of the formula, so the
    test is deterministic and runs in milliseconds.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (64, 16)); y = rng.integers(0, 3, 64)
    TS = [4, 8, 16, 32, 64]

    norms = {}
    for flag in (True, False):
        row = []
        for T in TS:
            net = SpikingNet([16, 24, 3], rng=np.random.default_rng(seed))
            _, g = net.grads(X, y, T, norm_t=flag)
            row.append(float(np.linalg.norm(g)))
        norms[flag] = row

    base4, base16 = norms[True][0], norms[True][2]
    growth = base16 / base4
    check("with normalisation the gradient does not grow with T",
          growth < 1.6,
          " ||g||: " + " ".join(f"T={T}:{v:.3e}" for T, v in zip(TS, norms[True]))
          + f"  (T=16/T=4 = {growth:.2f})")

    n_growth = norms[False][2] / norms[False][0]
    check("control: without normalisation the gradient GROWS like T",
          n_growth > 2.0 and n_growth > growth * 2,
          f"T=16/T=4 = {n_growth:.2f} without vs {growth:.2f} with"
          + f"  (ideal {TS[2] // TS[0]})")

    # third: training at a sane lr must converge at ANY T
    losses = {}
    for T in (4, 32):
        net = SpikingNet([16, 32, 3], rng=np.random.default_rng(seed))
        net.calibrate(X, T, target=0.25)
        for _ in range(150):
            _, gr = net.grads(X, y, T, norm_t=True)
            net.step(gr, lr=0.05)
        losses[T] = net.loss(X, y, T)
    check("with normalisation training converges at T=4 and T=32",
          max(losses.values()) < 0.5,
          f"T=4 -> {losses[4]:.4f}, T=32 -> {losses[32]:.4f}")


# ═════════════════════════════════════════════════ 3. OVERFIT-32 ══════════

def test_overfit_32(steps=400, seed=7, target=0.02):
    """The model must drive 32 examples below loss 0.02.

    Catches: EVERYTHING. If the gradient is wrong there is nothing to drive,
    and we find out in seconds rather than after a 40-minute experiment.
    """
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, (4, 20)) * 1.2
    y = rng.integers(0, 4, 32)
    X = C[y] + rng.normal(0, 1.5, (32, 20))

    T = 16
    net = SpikingNet([20, 64, 4], rng=rng, norm_t=True)
    net.calibrate(X, T, target=0.30)
    l0 = net.loss(X, y, T)
    for i in range(steps):
        _, gr = net.grads(X, y, T, norm_t=True)
        net.step(gr, lr=0.08, mom=0.9)
    l1 = net.loss(X, y, T)
    acc = net.accuracy(X, y, T)
    check("overfit-32: loss falls to ~0",
          l1 < target and l1 < l0 * 0.05,
          f"loss {l0:.4f} -> {l1:.4f}, accuracy on the same 32 = {acc:.0%}")


# ═══ 4. CALIBRATION DOES ITS JOB ════════════════════════════════════════

def test_calibration(seed=11, target=0.25):
    """After calibration the firing rate ~= target.

    Catches: bug 14 (calibrating with reset switched off) -- then the
    threshold drifts and the rate becomes an arbitrary number.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (64, 32))
    T = 16
    net = SpikingNet([32, 48, 5], rng=rng)
    fire0, _, _ = net.firing(X, T)
    net.calibrate(X, T, target=target)
    fire1, _, pl = net.firing(X, T)
    ok = abs(fire1 - target) < 0.12
    check("threshold calibration hits the target firing rate",
          ok, f"{fire0:.3f} -> {fire1:.3f} (target {target}), "
              f"per layer {[round(p, 3) for p in pl]}")


# ═══ CREDIT TRUNCATION ══════════════════════════════════════════════════

def test_credit_truncation(seed=13, T=24, k=4):
    """Truncated credit must match full BPTT in the small-k limit.

    We do not check equality (it does not hold); we check that the RESULTS
    are CLOSE for k >= the theoretical horizon (66 at leak 0.9 -- here k=T).
    Plus: the result must not depend on k once k >= T.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (32, 12)); y = rng.integers(0, 4, 32)
    net = SpikingNet([12, 20, 4], rng=rng)
    net.calibrate(X, T, target=0.25)
    _, g_full = net.grads(X, y, T, credit_k=None)
    _, g_k1 = net.grads(X, y, T, credit_k=1)
    rel_full = np.linalg.norm(g_full) / (np.linalg.norm(g_k1) + 1e-12)
    check("full BPTT and k=1 give different gradients (orb is not broken)",
          rel_full > 1.2, f"||g_full||/||g_k1|| = {rel_full:.2f}")
    _, g_kT = net.grads(X, y, T, credit_k=T)
    same = np.allclose(g_full, g_kT, rtol=1e-12, atol=1e-14)
    check("credit_k=T is identical to full BPTT", same)

    # -- truncation must give a SUBGRADIENT, not garbage --------------------
    # With truncated credit the gradient stops being a derivative, but it
    # must remain a descent direction: g_k^T * (g_full - g_k) >= 0.
    # If that fails, truncation BREAKS optimisation.
    net2 = SpikingNet([12, 20, 4], rng=np.random.default_rng(seed),
                      credit_k=None, norm_t=True)
    net2.calibrate(X, T, target=0.25)
    _, gf = net2.grads(X, y, T, credit_k=None)
    ns, bad = [], 0
    for kk in (1, 2, 3, 4, 6, 8, 12):
        _, gk = net2.grads(X, y, T, credit_k=kk)
        # (g_full - g_k) is the "full information" direction, g_k ours
        inner = float(gk @ (gf - gk))
        ns.append(inner)
        if inner < -1e-14:
            bad += 1
    check("truncation stays a descent direction",
          bad == 0, f"min g_k·(g_full−g_k) = {min(ns):+.3e}, "
                    f"bad k: {bad}/7")


# ═══════════════════════════════════════════════════════════════ MAIN ═════

if __name__ == "__main__":
    print("=" * 74)
    print("THE GATE: can this rig be trusted at all")
    print("=" * 74)
    test_finite_difference()
    test_calibration()
    test_overfit_32()
    test_t_invariance()
    test_credit_truncation()
    print("=" * 74)
    print(f"  PASSED {len(PASS)}  |  FAILED {len(FAIL)}")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
        print()
        print("  No result is published while the gate is red.")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)