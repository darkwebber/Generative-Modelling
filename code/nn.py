"""
The few pieces every model in this folder shares, written out by hand in numpy, like everything else.

    sig, swish, swish_grad     activations (swish(u) = u·σ(u): smooth, so slopes and scores are smooth)
    Adam                       the optimiser, with an optional running average (EMA) of the weights
    MLP                        a layer-spec network: ('lin', a, b), 'lrelu', 'sigmoid'     (GAN, flow couplings)
    SwishMLP                   sizes[0] → … → sizes[-1], swish in between                  (energy, score)
    TimeMLP, temb              x, t → out, a sinusoidal embedding of t added to every hidden layer
                               (diffusion; any model that is told "when" — e.g. flow matching)
    ring, moons                the 2-D toy data sets used across the series
    grad_check                 hand-written gradients vs finite differences

Each model file keeps its own forward pass, loss and backward maths: those are the lesson.
This file is only the plumbing they have in common.
"""
import math

import numpy as np


# ── activations ───────────────────────────────────────────────
def sig(u): return 1 / (1 + np.exp(-np.clip(u, -60, 60)))
def swish(u): return u * sig(u)
def swish_grad(u): s = sig(u); return s * (1 + u * (1 - s))          # d/du [u·σ(u)]


# ── the optimiser ─────────────────────────────────────────────
class Adam:
    """Adam on a list of arrays, updated in place. ema: also keep a running average of the weights.
    square_first: compute (1 − β₂)·g² as (1 − β₂)·(g²) instead of ((1 − β₂)·g)·g — equal maths, different
    last-bit rounding; the GAN and flow nets were trained that way, and keeping it reproduces them exactly."""

    def __init__(self, params, b1=0.9, b2=0.999, ema=None, square_first=False):
        self.params, self.b1, self.b2, self.t, self.ema_rate, self.sq = params, b1, b2, 0, ema, square_first
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]
        self.ema = [p.copy() for p in params] if ema else None

    def step(self, grads, lr):
        self.t += 1; b1, b2 = self.b1, self.b2
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g; self.v[i] = b2 * self.v[i] + ((1 - b2) * g ** 2 if self.sq else (1 - b2) * g * g)
            p -= lr * (self.m[i] / (1 - b1 ** self.t)) / (np.sqrt(self.v[i] / (1 - b2 ** self.t)) + 1e-8)
            if self.ema is not None: self.ema[i] = self.ema_rate * self.ema[i] + (1 - self.ema_rate) * p

    def use_ema(self):
        """Swap the averaged weights in (smoother, usually better samples)."""
        for p, e in zip(self.params, self.ema): p[...] = e


# ── networks ──────────────────────────────────────────────────
class MLP:
    """Layers: ('lin', n_in, n_out), 'lrelu', 'sigmoid'. forward() caches, backward() returns dL/dx."""

    def __init__(self, spec, rng, scale=1.0):
        self.spec, self.P = spec, {}
        for i, s in enumerate(spec):
            if isinstance(s, tuple):
                _, a, b = s
                self.P[f'W{i}'] = (rng.normal(0, 1, (a, b)) * np.sqrt(2 / a) * scale).astype(np.float32)
                self.P[f'b{i}'] = np.zeros(b, np.float32)
        self.opt = Adam(list(self.P.values()), square_first=True)

    def forward(self, x):
        self.cache = []
        for i, s in enumerate(self.spec):
            self.cache.append(x)
            if isinstance(s, tuple):
                x = x @ self.P[f'W{i}'] + self.P[f'b{i}']
            elif s == 'lrelu':
                x = np.where(x > 0, x, 0.2 * x)
            elif s == 'sigmoid':
                x = 1 / (1 + np.exp(-x))
        self.out = x
        return x

    def backward(self, d, accumulate=True):
        self.G = {k: np.zeros_like(v) for k, v in self.P.items()} if not accumulate or not hasattr(self, 'G') else self.G
        for i in reversed(range(len(self.spec))):
            s, x = self.spec[i], self.cache[i]
            if isinstance(s, tuple):
                self.G[f'W{i}'] += x.T @ d
                self.G[f'b{i}'] += d.sum(0)
                d = d @ self.P[f'W{i}'].T
            elif s == 'lrelu':
                d = d * np.where(x > 0, 1, 0.2)
            elif s == 'sigmoid':
                sg = 1 / (1 + np.exp(-x))
                d = d * sg * (1 - sg)
        return d

    def zero_grad(self):
        self.G = {k: np.zeros_like(v) for k, v in self.P.items()}

    def step(self, lr, b1=0.5, b2=0.999):
        self.opt.b1, self.opt.b2 = b1, b2; self.opt.step([self.G[k] for k in self.P], lr)


class SwishMLP:
    """sizes[0] → sizes[1] → … → sizes[-1], swish between layers, linear output."""

    def __init__(self, sizes, seed=0, b1=0.9):
        rng = np.random.default_rng(seed)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(1 / a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.opt = Adam(self.W + self.b, b1=b1)

    def forward(self, x):
        self.h, self.u = [x], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            u = self.h[-1] @ W + b
            if i < len(self.W) - 1:
                self.u.append(u); self.h.append(u * sig(u))
            else:
                return u

    def backward(self, d, params=True):
        """Given dL/d(output), store dL/dθ in self.gW, self.gb (if params) and return dL/d(input)."""
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            if params: gW.insert(0, self.h[i].T @ d); gb.insert(0, d.sum(0))
            d = d @ self.W[i].T
            if i > 0: d = d * swish_grad(self.u[i - 1])
        if params: self.gW, self.gb = gW, gb
        return d

    def step(self, lr, b1=None):
        if b1 is not None: self.opt.b1 = b1
        self.opt.step(self.gW + self.gb, lr)


def temb(t, dim=64):
    """Sinusoidal embedding of a step t ∈ {0 … 999}: sines and cosines at geometrically spaced frequencies."""
    f = np.exp(-math.log(1000) * np.arange(dim // 2) / (dim // 2))
    a = (np.asarray(t, np.float64)[:, None] + 1) * f[None] * 1.0
    return np.concatenate([np.sin(a), np.cos(a)], 1).astype(np.float32)


class TimeMLP:
    """out(x, t):  h₁ = swish(x W₁ + e U₁ + b₁) … out = h W_o + b_o,  e = temb(t) (t in every hidden layer).
    ff > 0 also feeds sin/cos of x at 1 … 64 cycles (Fourier features): lets a small MLP draw sharp detail."""
    freqs = None

    def __init__(self, dim, hidden=(256, 256, 256), ed=64, seed=0, ff=0, out=None, ema=0.999):
        rng = np.random.default_rng(seed); self.dim, self.ed = dim, ed
        self.freqs = (2.0 ** np.linspace(0, 6, ff)).astype(np.float32) if ff else None
        ins = [dim * (1 + 2 * ff)] + list(hidden[:-1])
        self.W = [(rng.standard_normal((a, h)) * np.sqrt(1 / a)).astype(np.float32) for a, h in zip(ins, hidden)]
        self.U = [(rng.standard_normal((ed, h)) * np.sqrt(1 / ed)).astype(np.float32) for h in hidden]
        self.b = [np.zeros(h, np.float32) for h in hidden]
        self.Wo = np.zeros((hidden[-1], out or dim), np.float32); self.bo = np.zeros(out or dim, np.float32)
        self.params = self.W + self.U + self.b + [self.Wo, self.bo]
        self.opt = Adam(self.params, ema=ema)

    def mlp(self, x, t):
        e = temb(t, self.ed)
        if self.freqs is not None:
            xf = (x[:, :, None] * self.freqs[None, None]).reshape(len(x), -1); x = np.concatenate([x, np.sin(xf), np.cos(xf)], 1).astype(np.float32)
        self.e, self.a, self.z = e, [x], []
        for W, U, b in zip(self.W, self.U, self.b):
            z = self.a[-1] @ W + e @ U + b; self.z.append(z); self.a.append(z * sig(z))
        return self.a[-1] @ self.Wo + self.bo

    __call__ = mlp

    def backward(self, d):
        gW, gU, gb = [None] * len(self.W), [None] * len(self.W), [None] * len(self.W)
        gWo, gbo = self.a[-1].T @ d, d.sum(0); d = d @ self.Wo.T
        for i in reversed(range(len(self.W))):
            d = d * swish_grad(self.z[i])
            gW[i] = self.a[i].T @ d; gU[i] = self.e.T @ d; gb[i] = d.sum(0)
            d = d @ self.W[i].T
        self.grads = gW + gU + gb + [gWo, gbo]

    def step(self, lr): self.opt.step(self.grads, lr)
    def use_ema(self): self.opt.use_ema()


# ── 2-D toy data ──────────────────────────────────────────────
def ring(n, rng, k=8, r=2.0, std=0.12):
    """k Gaussian blobs evenly spaced on a circle of radius r."""
    a = rng.integers(0, k, n) * 2 * np.pi / k
    return (np.stack([r * np.cos(a), r * np.sin(a)], 1) + std * rng.standard_normal((n, 2))).astype(np.float32)


def moons(n, rng, noise=0.08):
    """Two interleaved half-moons."""
    th = np.pi * rng.random(n); up = rng.random(n) < 0.5
    x = np.where(up, np.cos(th) - 0.5, 0.5 - np.cos(th)); y = np.where(up, np.sin(th) - 0.25, 0.25 - np.sin(th))
    return (np.stack([x, y], 1) * 1.5 + noise * rng.standard_normal((n, 2))).astype(np.float32)


# ── checking hand-written gradients ───────────────────────────
def grad_check(params, grads, loss, rng, h=1e-6):
    """Nudge one random entry of each parameter array both ways and compare the slope with the gradient.
    params: arrays (edited in place and restored), grads: matching analytic gradients, loss: () → float.
    Returns the worst relative error (≈1e-8 when the backprop is right)."""
    worst = 0
    for P, G in zip(params, grads):
        idx = tuple(rng.integers(0, s) for s in P.shape); old = P[idx]
        P[idx] = old + h; lp = loss(); P[idx] = old - h; lm = loss(); P[idx] = old
        num = (lp - lm) / (2 * h); worst = max(worst, abs(num - G[idx]) / max(1e-9, abs(num) + abs(G[idx])))
    return worst


def report(worst, what=''):
    print(f'{what}worst relative gradient error {worst:.1e} ({"OK" if worst < 1e-4 else "CHECK"})')
