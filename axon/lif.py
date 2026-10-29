"""Reference LIF spiking kernel with verified gradients.

numpy only -- no TF. Exists so gradients can be checked against finite
differences without a GPU (see axon/gate.py).

Equations
---------
    u_t  = leak * st_{t-1} + in_t @ W^T ,    u_{-1} = 0
    s_t  = 1[u_t > thr]                     (hard spike)
    st_t = u_t * (1 - s_t)                  (reset)

    G_t := dL/ds_t                 (last layer: G_t = (dl @ Wo) / T)
    Q_t := G_t * sigma'(u_t - thr)
    dL/dthr    = -sum_t Q_t + reset path
    P_t := Q_t*(1 - s_t) + leak*P_{t+1}
    dL/dW      = sum_t in_t^T @ P_t

Three places where the math breaks silently
------------------------------------------
1. 1/T normalisation. readout = (1/T) sum_t s_t, so dL/ds_t carries 1/T.
   Without it the gradient scales with T: at T=64 it is 16x the T=4 value
   at the same lr, and training diverges.
2. Threshold must be calibrated on a pass WITH reset enabled. Reset is what
   keeps u bounded; measuring with it off gives a drifting distribution and
   collapses the threshold to 0.
3. The surrogate gradient through the threshold is non-zero. Skip it and the
   threshold never learns.

u_t appears twice (in s_t and in st_t), which makes the graph ambiguous if
treated as one variable. The reset therefore reads a copy, which makes the
graph a tree and the gradient unique. See grads().
"""
from __future__ import annotations

import numpy as np


def ssur(u, thr, alpha=2.0):
    """Smooth surrogate step, ~1[u>thr]. Used in smooth mode."""
    return np.arctan(alpha * (u - thr)) / np.pi + 0.5


def dsig(u, thr, alpha=2.0):
    """d/du of the surrogate step; replaces the Heaviside delta."""
    a = alpha * (u - thr)
    return alpha / (np.pi * (1.0 + a * a))


def dsig2(u, thr, alpha=2.0):
    """Second derivative of the surrogate step.

    sigma'(x) = alpha / (pi (1 + (alpha x)^2))
    d/dx      = -2 alpha^2 x / (pi (1 + (alpha x)^2)^2)
    """
    a = alpha * (u - thr)
    d = 1.0 + a * a
    return -2.0 * alpha * alpha * a / (np.pi * d * d)


class SpikingNet:
    """Fully connected LIF spiking net.

    dims      : [din, d1, ..., dL, dout]
    leak      : leak coefficient, 0 <= leak < 1
    alpha     : surrogate sharpness
    norm_t    : divide dL/ds_t by T. Required for mean readout.
    credit_k  : credit horizon in steps; None = full BPTT
    learn_thr : train the threshold
    """

    def __init__(self, dims, leak=0.9, alpha=2.0, rng=None, norm_t=True,
                 credit_k=None, learn_thr=False):
        self.dims = list(dims)
        self.L = len(dims) - 2
        self.leak = float(leak)
        self.alpha = float(alpha)
        self.norm_t = bool(norm_t)
        self.credit_k = credit_k
        self.learn_thr = bool(learn_thr)
        rng = np.random.default_rng(0) if rng is None else rng

        self.W, self.Wo, self.thr = [], None, []
        for i in range(self.L + 1):
            w = rng.normal(0, 1.0 / np.sqrt(dims[i]), (dims[i], dims[i + 1])) * 0.9
            if i < self.L:
                self.W.append(w)
                self.thr.append(np.ones(dims[i + 1]))
            else:
                self.Wo = w

        self.mW = [np.zeros_like(w) for w in self.W]
        self.mWo = np.zeros_like(self.Wo)

    # ── parameters ───────────────────────────────────────────────────────────

    def params(self):
        """All parameters as one flat vector, for finite differences."""
        out = []
        for w, t in zip(self.W, self.thr):
            out += [w.ravel(), t.ravel()]
        out.append(self.Wo.ravel())
        return np.concatenate(out)

    def set_params(self, vec):
        o = 0
        for l in range(self.L):
            n = self.W[l].size
            self.W[l] = vec[o:o + n].reshape(self.W[l].shape).copy(); o += n
            n = self.thr[l].size
            self.thr[l] = vec[o:o + n].reshape(self.thr[l].shape).copy(); o += n
        self.Wo = vec[o:o + self.Wo.size].reshape(self.Wo.shape).copy()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, X, T, xseq=None, smooth=False):
        """Return (logits, readout, cache).

        X    : (B, din)        static input, repeated at every step
        xseq : (T, B, din)    time-varying input; overrides X
        smooth=False -> hard spike 1[u>thr], reset to 0. This is what runs
                        at inference, but it is piecewise constant, so its
                        finite-difference derivative is exactly zero.
        smooth=True  -> surrogate in the forward pass, s = sigma(u-thr),
                        soft reset u <- u*(1-s). The backward pass is the
                        SAME code, and is now the true gradient of this
                        smooth function, so it can be verified.

        The split matters: a surrogate gradient is not the derivative of the
        hard function and cannot be. Only the smooth model is checkable.
        cache: per layer (u, spikes); u is taken BEFORE the reset
        """
        if xseq is not None:
            inp = xseq                                # (T, B, din)
            B = xseq.shape[1]
        else:
            if X is None:
                raise ValueError("need either X or xseq")
            B = X.shape[0]
            inp = np.broadcast_to(X, (T, B, X.shape[1]))

        cur, cache = inp, []
        for l in range(self.L):
            st = np.zeros((B, self.dims[l + 1]))
            us, sps = [], []
            for t in range(T):
                cu = cur[t] @ self.W[l]
                u = self.leak * st + cu
                if smooth:
                    s = ssur(u, self.thr[l], self.alpha)
                    st = u * (1.0 - s)              # soft reset
                else:
                    s = (u > self.thr[l]).astype(np.float64)
                    st = np.where(s > 0.0, 0.0, u)  # hard reset
                us.append(u); sps.append(s)
            spikes = np.array(sps)
            cache.append((np.array(us), spikes))
            cur = spikes

        readout = cur.mean(0)                     # (B, dL)
        return readout @ self.Wo, readout, cache

    def inputs(self, X, T, xseq=None):
        if xseq is not None:
            return xseq
        if X is None:
            raise ValueError("need either X or xseq")
        return np.broadcast_to(X, (T, X.shape[0], X.shape[1]))

    # ── calibration ──────────────────────────────────────────────────────────

    def calibrate(self, X, T, target=0.25, iters=8, damp=0.7, xseq=None,
                  verbose=False):
        """Fixed-point threshold calibration toward a target firing rate.

        Potentials are measured on a REAL pass with reset enabled.
        new_thr[l] = quantile(1 - target) of layer l potentials.
        """
        hist = []
        for it in range(iters):
            _, _, cache = self.forward(X, T, xseq=xseq)
            u_all = [c[0] for c in cache]      # u before reset, real pass
            new = []
            for l in range(self.L):
                q = float(np.quantile(u_all[l], 1.0 - target))
                new.append(np.clip(np.full(self.dims[l + 1], q), 0.05, 8.0))
            self.thr = [damp * n + (1 - damp) * o
                        for n, o in zip(new, self.thr)]
            hist.append(float(np.mean([t.mean() for t in self.thr])))
            if verbose:
                print(f"    calib it{it}: thr={hist[-1]:.4f}")
        return hist

    # ── loss ─────────────────────────────────────────────────────────────────

    def loss(self, X, y, T, xseq=None, smooth=False):
        logits, _, _ = self.forward(X, T, xseq=xseq, smooth=smooth)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        n = X.shape[0] if xseq is None else xseq.shape[1]
        return -np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean()

    # ── gradients ────────────────────────────────────────────────────────────

    def grads(self, X, y, T, xseq=None, credit_k=None, norm_t=None,
              learn_thr=None, smooth=False):
        """Analytic gradients -> (loss, dvec) in params() layout.

        One formula covers both modes; the mode only changes the VALUE of s_t.

        Graph
        -----
            u_t  = leak * st_{t-1} + cu_t,   cu_t = in_t @ W
            s_t  = 1[u_t > thr]        (hard) | sigma(u_t - thr) (smooth)
            st_t = (1 - s_t) * u_t

        u_t is used TWICE, so treating it as one variable makes the graph
        cyclic and the derivative ambiguous. The reset reads a COPY, which
        makes it a tree and the gradient unique:

            u_t      -> s_t      (via sigma)
            u_copy_t -> st_t     (via (1 - s_t))

        That single detail separates a correct backward pass from most
        hand-written ones.

        Recursion
        ----------
            d st_t/d u_t = (1 - s_t) - u_t * sigma'(u_t - thr)

        With sg_t = sigma'(u_t - thr),  f_t = d st_t/du_t,  U_t = dL/du_t:

            U_t       = A_t*sg_t + leak * U_{t+1} * f_t      (t = T-1 ... 0)
            dL/dcu_t  = U_t
            dL/dW     = sum_t in_t^T @ U_t
            dL/dthr   = -sum_t A_t*sg_t + reset path
            A_t below = U_t @ W_l^T

        Only f_t differs between modes:
            hard  : s_t constant        => f_t = (1 - s_t)
            smooth: s_t = sigma(u_t-thr) => f_t = (1-s_t) - u_t*sg_t

        Past bug: the (1-s_t) factor was applied to A_t*sg_t instead of to
        leak*U_{t+1}. The error is large and invisible -- only the finite
        difference check catches it.
        """
        norm_t = self.norm_t if norm_t is None else norm_t
        credit_k = self.credit_k if credit_k is None else credit_k
        learn_thr = self.learn_thr if learn_thr is None else learn_thr

        logits, readout, cache = self.forward(X, T, xseq=xseq, smooth=smooth)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        B = X.shape[0] if xseq is None else xseq.shape[1]
        loss = -np.log(np.clip(p[np.arange(B), y], 1e-12, None)).mean()

        dl = p.copy(); dl[np.arange(B), y] -= 1.0; dl /= B
        # Wo is (dL, dout), so the gradient is readout.T @ dl
        gWo = readout.T @ dl

        # time-mean readout -> 1/T normalisation
        G = (dl @ self.Wo.T) / T if norm_t else (dl @ self.Wo.T)
        A = np.repeat(G[None], T, 0).astype(np.float64)   # (T,B,dL)

        inp = self.inputs(X, T, xseq)
        gW = [None] * self.L
        gthr = [None] * self.L
        k = T if credit_k is None else int(credit_k)

        for l in range(self.L - 1, -1, -1):
            u, s = cache[l]
            th = self.thr[l][None, None, :]
            sg = dsig(u, th, self.alpha)          # sigma'(u_t - thr)

            # recursion runs BACKWARD in time
            U = np.empty_like(A)
            gthr_acc = np.zeros((B, self.dims[l + 1]))   # reset path
            nxt = np.zeros((B, self.dims[l + 1]))        # = U_{t+1}
            for t in range(T - 1, -1, -1):
                # d st_t/du_t:
                #   hard reset : s_t constant => (1 - s_t)
                #   soft reset :             => (1-s_t) - u_t*sg_t
                f = (1.0 - s[t]) if not smooth else (1.0 - s[t] - u[t] * sg[t])
                carry = self.leak * nxt * f
                # credit horizon: stop propagating past k steps
                if (T - 1 - t) >= k:
                    carry = 0.0
                U[t] = A[t] * sg[t] + carry
                # threshold, second path: through the reset.
                #   st_t = u_t*(1-s_t), ds_t/dthr = -sigma'
                #   => d st_t/dthr = -u_t*sigma', dL/dst_t = leak*U_{t+1}
                if smooth and (T - 1 - t) < k:
                    gthr_acc += (self.leak * nxt) * u[t] * sg[t]
                nxt = U[t]

            gthr[l] = (-(A * sg).sum(axis=(0, 1)) + gthr_acc.sum(axis=0))

            src = inp if l == 0 else cache[l - 1][1]
            gW[l] = np.einsum("tbi,tbj->ij", src, U)
            if l > 0:
                A = U @ self.W[l].T                # dL/din_{l,t}

        out = []
        for l in range(self.L):
            out += [gW[l].ravel(),
                    gthr[l].ravel() if learn_thr else np.zeros_like(gthr[l])]
        out.append(gWo.ravel())
        return loss, np.concatenate(out)

    # ── SGD ──────────────────────────────────────────────────────────────────

    def step(self, gvec, lr=0.05, mom=0.9, learn_thr=None):
        learn_thr = self.learn_thr if learn_thr is None else learn_thr
        o = 0
        for l in range(self.L):
            n = self.W[l].size
            g = gvec[o:o + n].reshape(self.W[l].shape); o += n
            self.mW[l] = mom * self.mW[l] + g
            self.W[l] = self.W[l] - lr * self.mW[l]
            n = self.thr[l].size
            g = gvec[o:o + n].reshape(self.thr[l].shape); o += n
            if learn_thr:
                self.thr[l] = self.thr[l] - lr * g
        g = gvec[o:o + self.Wo.size].reshape(self.Wo.shape)
        self.mWo = mom * self.mWo + g
        self.Wo = self.Wo - lr * self.mWo

    # ── metrics ──────────────────────────────────────────────────────────────

    def firing(self, X, T, xseq=None):
        """(mean firing rate, dead fraction, per-layer rates)."""
        _, _, cache = self.forward(X, T, xseq=xseq)
        rates = [((u > self.thr[l][None, None, :]).astype(float)).mean(axis=(0, 1))
                 for l, (u, _) in enumerate(cache)]
        dead = float(np.mean([(r == 0).mean() for r in rates]))
        fire = float(np.mean([r.mean() for r in rates]))
        return fire, dead, [float(r.mean()) for r in rates]

    def accuracy(self, X, y, T, xseq=None):
        logits, _, _ = self.forward(X, T, xseq=xseq)
        return float((logits.argmax(1) == y).mean())