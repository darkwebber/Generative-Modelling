"""
Episode 08 — Score matching and (annealed) Langevin dynamics, in numpy, backprop by hand.

The score of a density is the slope of its log:    s(x) = ∇ₓ log p(x)
It never needs the normaliser Z (∇ₓ log Z = 0), and Langevin dynamics samples p using only s:

    x ← x + η·s(x) + √(2η)·ε,    ε ~ N(0, I)

Learning s without knowing p — denoising score matching (Vincent, 2011):
    add noise  x̃ = x + σε;  the way home is  −(x̃ − x)/σ² = −ε/σ;  regress onto it:
    L(θ) = E ‖ σ·s_θ(x̃, σ) + ε ‖²        →  its minimiser is the score of the noisy data, ∇ log p_σ
We write s_θ(x̃, σ) = net(x̃, log σ) / σ, so the network simply predicts −ε.

One network for many noise levels σ₁ > … > σ_L (NCSN, Song & Ermon 2019), sampled by annealed Langevin:
start at the largest σ, run a few Langevin steps, lower σ, repeat.

Requires:  pip install numpy          (matplotlib optional, for --plot)

Try:
    python score.py                      # 8 blobs: train, then compare plain vs annealed Langevin
    python score.py --data twoblob       # an 80 / 20 mixture: plain Langevin gets the weights wrong
    python score.py --check              # verify the hand-written gradients numerically
    python score.py --plot
"""
import argparse
import math

import numpy as np

R = 4.0
SIGMAS = np.geomspace(3.0, 0.05, 10)           # σ₁ … σ_L


# ── data: mixtures of Gaussians, so the true score of every noisy version is known exactly ──
def mixture(name):
    if name == 'ring':
        a = np.arange(8) * np.pi / 4
        return np.stack([2 * np.cos(a), 2 * np.sin(a)], 1), np.full(8, 1 / 8), 0.12
    if name == 'twoblob':
        return np.array([[-1.6, 0.9], [1.6, -0.9]]), np.array([0.8, 0.2]), 0.35
    raise ValueError(name)


def sample(name, n, rng):
    mu, w, s = mixture(name)
    k = rng.choice(len(w), n, p=w)
    return (mu[k] + s * rng.standard_normal((n, 2))).astype(np.float32)


def true_score(name, x, sigma=0.0):
    """∇ log p_σ(x) for the mixture blurred by N(0, σ²I): each blob just gets wider."""
    mu, w, s = mixture(name); v = s * s + sigma * sigma
    d = x[:, None, :] - mu[None]                                    # (N, K, 2)
    lw = np.log(w)[None] - 0.5 * (d ** 2).sum(-1) / v
    r = np.exp(lw - lw.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    return -(r[..., None] * d).sum(1) / v


# ── the network: [x, y, log σ] → 128 → 128 → 128 → 2, swish ──
def _sig(u): return 1 / (1 + np.exp(-np.clip(u, -60, 60)))


class Net:
    def __init__(self, hidden=128, seed=0, sizes=None):
        rng = np.random.default_rng(seed); sizes = sizes or [3, hidden, hidden, hidden, 2]
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(1 / a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.t = 0

    def __call__(self, x, sigma):
        """net(x̃, σ) ≈ −ε. The score is this divided by σ."""
        inp = np.concatenate([x, np.log(np.broadcast_to(np.asarray(sigma, np.float32), (len(x),)))[:, None]], 1).astype(np.float32)
        self.h, self.u = [inp], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            u = self.h[-1] @ W + b
            if i < len(self.W) - 1:
                self.u.append(u); self.h.append(u * _sig(u))
            else:
                return u

    def score(self, x, sigma):
        return self(x, sigma) / np.asarray(sigma, np.float32).reshape(-1, 1) if np.ndim(sigma) else self(x, sigma) / sigma

    def backward(self, d):
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, self.h[i].T @ d); gb.insert(0, d.sum(0))
            d = d @ self.W[i].T
            if i > 0:
                s = _sig(self.u[i - 1]); d = d * (s * (1 + self.u[i - 1] * (1 - s)))
        self.gW, self.gb = gW, gb

    def step(self, lr, b1=0.9, b2=0.999):
        self.t += 1
        for k, (p, g) in enumerate(zip(self.W + self.b, self.gW + self.gb)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g; self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            p -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + 1e-8)


def dsm_loss(net, x, sigma, eps):
    """Denoising score matching with σ² weighting: mean ‖σ·s_θ(x + σε, σ) + ε‖² = mean ‖net + ε‖²."""
    out = net(x + sigma[:, None] * eps, sigma)
    r = out + eps
    return float((r ** 2).sum(1).mean()), 2 * r / len(x)


def train(data='ring', steps=6000, batch=512, lr=2e-3, seed=0, snap=(), snap_sigma=0.3, log=print):
    rng = np.random.default_rng(seed); net = Net(seed=seed); hist, snaps = [], {}
    for s in range(1, steps + 1):
        x = sample(data, batch, rng); sig = SIGMAS[rng.integers(0, len(SIGMAS), batch)].astype(np.float32)
        eps = rng.standard_normal(x.shape).astype(np.float32)
        loss, d = dsm_loss(net, x, sig, eps); net.backward(d)
        net.step(lr * (0.2 if s > 0.85 * steps else 1.0)); hist.append(loss)
        if s in snap or (s == 1 and 0 in snap):
            snaps[s] = [w.copy() for w in net.W] + [b.copy() for b in net.b]
        if s % (steps // 6) == 0: log(f'step {s:5d}   denoising loss {np.mean(hist[-200:]):.3f}')
    return net, hist, snaps


# ── samplers ──────────────────────────────────────────────────
def langevin(score, x, steps, eta, rng, keep=0):
    """x ← x + η·s(x) + √(2η)·ε."""
    traj = [x.copy()] if keep else None
    for t in range(1, steps + 1):
        x = np.clip(x + eta * score(x) + math.sqrt(2 * eta) * rng.standard_normal(x.shape).astype(np.float32), -R, R)
        if keep and t % keep == 0: traj.append(x.copy())
    return (x, traj) if keep else x


def annealed(net, x, rng, T=30, c=0.1, keep=0, sigmas=SIGMAS):
    """Annealed Langevin: T steps at each σᵢ, step size η = c·σᵢ² (large noise first)."""
    traj = [x.copy()] if keep else None
    for sg in sigmas:
        eta = c * sg * sg
        for t in range(1, T + 1):
            x = np.clip(x + eta * net.score(x, sg) + math.sqrt(2 * eta) * rng.standard_normal(x.shape).astype(np.float32), -R, R)
            if keep and t % keep == 0: traj.append(x.copy())
    return (x, traj) if keep else x


def left_share(x):   # twoblob: fraction of samples in the heavier (left) blob
    return float((x[:, 0] < 0).mean())


def grad_check():
    rng = np.random.default_rng(0); net = Net(hidden=8, seed=1)
    net.W = [w.astype(np.float64) for w in net.W]; net.b = [b.astype(np.float64) for b in net.b]
    x = rng.standard_normal((6, 2)); sig = np.array([0.1, 0.5, 1, 2, 0.3, 0.7]); eps = rng.standard_normal((6, 2))
    _, d = dsm_loss(net, x, sig, eps); net.backward(d); G = net.gW + net.gb; worst = 0
    for P, Gp in zip(net.W + net.b, G):
        idx = tuple(rng.integers(0, s) for s in P.shape); old = P[idx]
        P[idx] = old + 1e-6; lp = dsm_loss(net, x, sig, eps)[0]; P[idx] = old - 1e-6; lm = dsm_loss(net, x, sig, eps)[0]; P[idx] = old
        num = (lp - lm) / 2e-6; worst = max(worst, abs(num - Gp[idx]) / max(1e-9, abs(num) + abs(Gp[idx])))
    print(f'worst relative gradient error {worst:.1e} ({"OK" if worst < 1e-4 else "CHECK"})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', choices=['ring', 'twoblob'], default='ring')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    if a.check:
        return grad_check()
    net, _, _ = train(a.data, a.steps)
    rng = np.random.default_rng(1); x0 = rng.uniform(-R, R, (2000, 2)).astype(np.float32)
    plain = langevin(lambda x: net.score(x, SIGMAS[-1]), x0, 300, 0.1 * SIGMAS[-1] ** 2, rng)
    ann = annealed(net, x0, np.random.default_rng(2))
    test = np.random.default_rng(3).uniform(-3, 3, (4000, 2)).astype(np.float32)
    for sg in (SIGMAS[0], SIGMAS[5], SIGMAS[-1]):
        err = np.linalg.norm(net.score(test, sg) - true_score(a.data, test, sg), axis=1) / (np.linalg.norm(true_score(a.data, test, sg), axis=1) + 1e-9)
        print(f'σ = {sg:.2f}: median relative score error {np.median(err):.2f}')
    if a.data == 'twoblob':
        print(f'share in the 80% blob:  plain Langevin {left_share(plain):.2f}   annealed {left_share(ann):.2f}   truth 0.80')
    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for axi, (x, t) in zip(ax, ((plain, 'plain Langevin at the smallest σ'), (ann, 'annealed Langevin'))):
            axi.scatter(*x.T, s=2, c='#c4654a'); axi.set_xlim(-R, R); axi.set_ylim(-R, R); axi.set_title(t)
        plt.show()


if __name__ == '__main__':
    main()
