"""
Episode 09 — Diffusion models (DDPM + DDIM) in numpy, backprop by hand.

Forward process (fixed, no learning): add a little Gaussian noise, T = 1000 times.
    x_t = √(1 − β_t)·x_{t−1} + √β_t·ε          →   in one jump:   x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε,   ᾱ_t = Π(1 − β_s)
Training (Ho et al. 2020): pick x₀, a step t and noise ε; make x_t in one jump; regress onto the noise:
    L(θ) = E ‖ ε − ε_θ(x_t, t) ‖²          (episode 8's denoising score matching: s_θ = −ε_θ / √(1 − ᾱ_t))
Sampling, ancestral (DDPM):
    x_{t−1} = ( x_t − β_t/√(1 − ᾱ_t)·ε_θ ) / √(1 − β_t) + σ_t·z
Sampling, DDIM (Song et al. 2021): jump between any two steps, deterministically:
    x̂₀ = ( x_t − √(1 − ᾱ_t)·ε_θ ) / √ᾱ_t,     x_s = √ᾱ_s·x̂₀ + √(1 − ᾱ_s)·ε_θ

The network is an MLP whose every hidden layer also receives a sinusoidal embedding of t. For MNIST it outputs its
guess of the clean image, x̂₀ = tanh(F), and reports ε_θ = (x_t − √ᾱ_t·x̂₀)/√(1 − ᾱ_t): the same model, rearranged,
but an MLP finds a clean image far easier to produce than faint per-pixel noise. Its loss is then ‖ε − ε_θ‖²
weighted by w_t = min(1, 5/SNR_t), SNR_t = ᾱ_t/(1 − ᾱ_t) (Hang et al., 2023), so nearly clean steps don't dominate.

Requires:  pip install numpy          (MNIST is downloaded on first use by vae.py)

Try:
    python diffusion.py --data ring            # 2-D: train, then sample with 1000, 100 and 20 steps
    python diffusion.py --data mnist --steps 20000 --plot
    python diffusion.py --check                # verify the hand-written gradients numerically
"""
import argparse
import math

import numpy as np

T = 1000
BETAS = np.linspace(1e-4, 0.02, T).astype(np.float64)
ABAR = np.cumprod(1 - BETAS)


def temb(t, dim=64):
    """Sinusoidal embedding of the step t ∈ {0 … T−1}."""
    f = np.exp(-math.log(1000) * np.arange(dim // 2) / (dim // 2))
    a = (np.asarray(t, np.float64)[:, None] + 1) * f[None] * 1.0
    return np.concatenate([np.sin(a), np.cos(a)], 1).astype(np.float32)


def _sig(u): return 1 / (1 + np.exp(-np.clip(u, -60, 60)))


class EpsNet:
    """ε_θ(x, t):  h₁ = swish(x W₁ + e U₁ + b₁) … out = h W_o + b_o,  e = temb(t)."""
    x0, gamma, freqs = False, 5.0, None

    def __init__(self, dim, hidden=(256, 256, 256), ed=64, seed=0, ff=0, x0=False, gamma=5.0):
        rng = np.random.default_rng(seed); self.dim, self.ed, self.x0, self.gamma = dim, ed, x0, gamma
        # optional Fourier features of the input (sin/cos at 1 … 64 cycles): lets a small MLP draw sharp detail
        self.freqs = (2.0 ** np.linspace(0, 6, ff)).astype(np.float32) if ff else None
        ins = [dim * (1 + 2 * ff)] + list(hidden[:-1])
        self.W = [(rng.standard_normal((a, h)) * np.sqrt(1 / a)).astype(np.float32) for a, h in zip(ins, hidden)]
        self.U = [(rng.standard_normal((ed, h)) * np.sqrt(1 / ed)).astype(np.float32) for h in hidden]
        self.b = [np.zeros(h, np.float32) for h in hidden]
        self.Wo = np.zeros((hidden[-1], dim), np.float32); self.bo = np.zeros(dim, np.float32)
        self.params = self.W + self.U + self.b + [self.Wo, self.bo]
        self.m = [np.zeros_like(p) for p in self.params]; self.v = [np.zeros_like(p) for p in self.params]; self.k = 0
        self.ema = [p.copy() for p in self.params]

    def __call__(self, x, t):
        e = temb(t, self.ed)
        if self.x0:
            # predict the clean image instead, x̂₀ = tanh(F), and read the noise off it (the denoiser identity):
            #   ε_θ = (x_t − √ᾱ·x̂₀) / √(1 − ᾱ)
            # the same model, rearranged; an MLP finds a clean image far easier to output than faint per-pixel noise
            ab = ABAR[np.asarray(t)].astype(np.float32)[:, None]; self.dk = -np.sqrt(ab / (1 - ab)); self.th = np.tanh(self._mlp(x, e))
            self.snr = ab / (1 - ab)
            return ((x - np.sqrt(ab) * self.th) / np.sqrt(1 - ab)).astype(x.dtype)
        return self._mlp(x, e)

    def _mlp(self, x, e):
        if self.freqs is not None:
            xf = (x[:, :, None] * self.freqs[None, None]).reshape(len(x), -1); x = np.concatenate([x, np.sin(xf), np.cos(xf)], 1).astype(np.float32)
        self.e, self.a, self.z = e, [x], []
        for W, U, b in zip(self.W, self.U, self.b):
            z = self.a[-1] @ W + e @ U + b; self.z.append(z); self.a.append(z * _sig(z))
        return self.a[-1] @ self.Wo + self.bo

    def backward(self, d):
        if self.x0: d = d * self.dk * (1 - self.th ** 2)          # back through ε = (x − √ᾱ·tanh F)/√(1 − ᾱ)
        gW, gU, gb = [None] * len(self.W), [None] * len(self.W), [None] * len(self.W)
        gWo, gbo = self.a[-1].T @ d, d.sum(0); d = d @ self.Wo.T
        for i in reversed(range(len(self.W))):
            s = _sig(self.z[i]); d = d * (s * (1 + self.z[i] * (1 - s)))
            gW[i] = self.a[i].T @ d; gU[i] = self.e.T @ d; gb[i] = d.sum(0)
            d = d @ self.W[i].T
        self.grads = gW + gU + gb + [gWo, gbo]

    def step(self, lr, b1=0.9, b2=0.999, ema=0.999):
        self.k += 1
        for i, (p, g) in enumerate(zip(self.params, self.grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g; self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            p -= lr * (self.m[i] / (1 - b1 ** self.k)) / (np.sqrt(self.v[i] / (1 - b2 ** self.k)) + 1e-8)
            self.ema[i] = ema * self.ema[i] + (1 - ema) * p

    def use_ema(self):
        """Swap in the exponential-moving-average weights (smoother, better samples)."""
        for p, e in zip(self.params, self.ema): p[...] = e


def loss_and_grad(net, x0, rng):
    n = len(x0); t = rng.integers(0, T, n); eps = rng.standard_normal(x0.shape).astype(np.float32)
    ab = ABAR[t].astype(np.float32)[:, None]
    xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
    r = net(xt, t) - eps
    if net.x0:            # per-step weights w_t = min(1, γ/SNR_t) (Hang et al., 2023): the nearly clean steps don't dominate
        w = np.minimum(1, net.gamma / net.snr); return float((w * r ** 2).mean() * x0.shape[1]), 2 * w * r / n
    return float((r ** 2).mean() * x0.shape[1]), 2 * r / n


def train(net, data_fn, steps, batch=256, lr=1e-3, seed=0, log=print, every=None):
    rng = np.random.default_rng(seed); hist = []
    for s in range(1, steps + 1):
        loss, d = loss_and_grad(net, data_fn(batch, rng), rng); net.backward(d)
        net.step(lr * (0.5 * (1 + math.cos(math.pi * s / steps))) + 1e-5); hist.append(loss)
        if s % (every or max(1, steps // 10)) == 0: log(f'step {s:6d}   loss {np.mean(hist[-200:]):.4f}')
    return hist


# ── sampling ──────────────────────────────────────────────────
def ddpm(net, x, rng, keep=None):
    """Ancestral sampling through all T steps. keep: list of t at which to record (x_t, x̂₀)."""
    rec = {}
    for t in range(T - 1, -1, -1):
        eps = net(x, np.full(len(x), t)); ab, b = ABAR[t], BETAS[t]
        if keep is not None and t in keep: rec[t] = (x.copy(), ((x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)).astype(np.float32))
        x = (x - b / np.sqrt(1 - ab) * eps) / np.sqrt(1 - b)
        if t > 0:
            var = b * (1 - ABAR[t - 1]) / (1 - ab)          # the posterior variance β̃_t
            x = x + np.sqrt(var) * rng.standard_normal(x.shape)
        x = x.astype(np.float32)
    return (x, rec) if keep is not None else x


def ddim(net, x, steps, keep_all=False, clip=None):
    """Deterministic DDIM with `steps` evenly spaced steps. clip: bound x̂₀ to [−clip, clip] (the data range)."""
    ts = np.linspace(T - 1, 0, steps).round().astype(int); traj = [x.copy()]
    for i, t in enumerate(ts):
        eps = net(x, np.full(len(x), t)); ab = ABAR[t]
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
        if clip: x0 = np.clip(x0, -clip, clip); eps = (x - np.sqrt(ab) * x0) / np.sqrt(1 - ab)
        abs_ = ABAR[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = (np.sqrt(abs_) * x0 + np.sqrt(1 - abs_) * eps).astype(np.float32)
        if keep_all: traj.append(x.copy())
    return (x, traj) if keep_all else x


# ── data ──────────────────────────────────────────────────────
def ring(n, rng):
    a = rng.integers(0, 8, n) * np.pi / 4
    return (np.stack([2 * np.cos(a), 2 * np.sin(a)], 1) + 0.12 * rng.standard_normal((n, 2))).astype(np.float32) / 2


def mnist_fn():
    import vae
    X, _ = vae.load_mnist('train'); X = X * 2 - 1
    return lambda n, rng: X[rng.integers(0, len(X), n)]


def grad_check():
    for x0 in (False, True): _grad_check(x0)


def _grad_check(x0):
    rng = np.random.default_rng(0); net = EpsNet(3, (8, 8), ed=6, seed=1, x0=x0)
    for i, p in enumerate(net.params): net.params[i][...] = p + 0.3 * rng.standard_normal(p.shape)
    net.params = [p.astype(np.float64) for p in net.params]
    net.W, net.U, net.b = net.params[:2], net.params[2:4], net.params[4:6]; net.Wo, net.bo = net.params[6], net.params[7]
    x = rng.standard_normal((5, 3)); t = rng.integers(0, T, 5); y = rng.standard_normal((5, 3))
    L = lambda: float(((net(x, t) - y) ** 2).sum())
    out = net(x, t); net.backward(2 * (out - y)); G = net.grads; worst = 0
    for P, Gp in zip(net.params, G):
        idx = tuple(rng.integers(0, s) for s in P.shape); old = P[idx]
        P[idx] = old + 1e-6; lp = L(); P[idx] = old - 1e-6; lm = L(); P[idx] = old
        num = (lp - lm) / 2e-6; worst = max(worst, abs(num - Gp[idx]) / max(1e-9, abs(num) + abs(Gp[idx])))
    print(f'{"x̂₀ head" if x0 else "ε head"}: worst relative gradient error {worst:.1e} ({"OK" if worst < 1e-4 else "CHECK"})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', choices=['ring', 'mnist'], default='ring')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    if a.check:
        return grad_check()
    if a.data == 'ring':
        net = EpsNet(2, (256, 256, 256)); train(net, ring, a.steps or 8000, batch=512); net.use_ema()
        rng = np.random.default_rng(1); z = rng.standard_normal((3000, 2)).astype(np.float32)
        for name, x in (('DDPM, 1000 steps', ddpm(net, z.copy(), rng)), ('DDIM, 100 steps', ddim(net, z.copy(), 100, clip=1.5)), ('DDIM, 20 steps', ddim(net, z.copy(), 20, clip=1.5))):
            r = np.linalg.norm(x * 2, axis=1); print(f'{name:18s} radius {r.mean():.2f} ± {r.std():.2f}   (data: 2.00 ± 0.12)')
    else:
        net = EpsNet(784, (1024, 1024), x0=True); train(net, mnist_fn(), a.steps or 30000, batch=128, lr=1e-3); net.use_ema()
        x = ddim(net, np.random.default_rng(1).standard_normal((64, 784)).astype(np.float32), 100, clip=1)
        if a.plot:
            import matplotlib.pyplot as plt
            g = ((x.reshape(8, 8, 28, 28).transpose(0, 2, 1, 3).reshape(224, 224) + 1) / 2).clip(0, 1)
            plt.imshow(g, cmap='gray'); plt.axis('off'); plt.show()


if __name__ == '__main__':
    main()
