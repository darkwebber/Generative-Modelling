"""
Episode 07 — Energy-based models in numpy, backprop by hand.

An energy-based model is ANY network that outputs one number per input: its energy E_θ(x).
Low energy = looks like data. The Boltzmann distribution turns energy into probability:

    p_θ(x) = exp(−E_θ(x)) / Z_θ,        Z_θ = ∫ exp(−E_θ(x)) dx        (never computed in training)

Learning = maximum likelihood (episode 1). The awkward log Z term turns into an average over the
model's OWN samples ("fantasies"):

    ∇θ [−log p_θ(x)] = ∇θ E_θ(x_data) − E_{x′ ~ p_θ}[ ∇θ E_θ(x′) ]
                       push energy DOWN on data   push it UP where the model dreams

Sampling = Langevin dynamics: roll downhill, plus a jiggle (no Z needed — only the slope):

    x ← x − η ∇ₓE(x) + √(2η) ε,      ε ~ N(0, I)

Training keeps a replay buffer of fantasies and moves them a few Langevin steps per update
(persistent contrastive divergence), with a small E² penalty so energies stay finite.

Requires:  pip install numpy          (matplotlib optional, for --plot)

Try:
    python ebm.py                         # 8 blobs: sculpt a landscape, report the exact 2-D NLL (≈ 3 min)
    python ebm.py --alpha 0.001           # weaker E² penalty: training drifts, one valley swallows the rest
    python ebm.py --data moons --plot     # two moons: energy, density and Langevin samples
    python ebm.py --sculpt                # 1-D: watch data dig valleys and fantasies raise hills
    python ebm.py --check                 # verify the hand-written gradients numerically
"""
import argparse
import math

import numpy as np

LOG2PI = math.log(2 * math.pi)
R = 3.5  # the box the samplers live in: [-R, R]²


# ── data ──────────────────────────────────────────────────────
def ring(n, rng, k=8, r=2.0, std=0.12):
    a = rng.integers(0, k, n) * 2 * np.pi / k
    return (np.stack([r * np.cos(a), r * np.sin(a)], 1) + std * rng.standard_normal((n, 2))).astype(np.float32)


def moons(n, rng, noise=0.08):
    th = np.pi * rng.random(n); up = rng.random(n) < 0.5
    x = np.where(up, np.cos(th) - 0.5, 0.5 - np.cos(th)); y = np.where(up, np.sin(th) - 0.25, 0.25 - np.sin(th))
    return (np.stack([x, y], 1) * 1.5 + noise * rng.standard_normal((n, 2))).astype(np.float32)


DATA = {'ring': ring, 'moons': moons}


# ── the energy network: 2 → H → H → H → 1, swish activations (smooth, so the slope is smooth) ──
def _sig(u): return 1 / (1 + np.exp(-u))


class Energy:
    def __init__(self, hidden=96, seed=0):
        rng = np.random.default_rng(seed); sizes = [2, hidden, hidden, hidden, 1]
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(1 / a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.t = 0

    def __call__(self, x):
        """E(x) for a batch of points, shape (N,). Caches what backward() needs."""
        self.h, self.u = [x], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            u = self.h[-1] @ W + b
            if i < len(self.W) - 1:
                self.u.append(u); self.h.append(u * _sig(u))       # swish(u) = u·σ(u)
            else:
                return u[:, 0]

    def backward(self, dE, params=True):
        """Given dL/dE (N,), return dL/dx (N, 2); if params, also store dL/dθ in self.gW, self.gb."""
        d = dE[:, None].astype(np.float32); gW, gb = [], []
        for i in reversed(range(len(self.W))):
            if params: gW.insert(0, self.h[i].T @ d); gb.insert(0, d.sum(0))
            d = d @ self.W[i].T
            if i > 0:
                s = _sig(self.u[i - 1]); d = d * (s * (1 + self.u[i - 1] * (1 - s)))   # swish′
        if params: self.gW, self.gb = gW, gb
        return d

    def grad_x(self, x):
        """The slope of the landscape, ∇ₓE(x) — the only thing a Langevin sampler needs."""
        self(x); return self.backward(np.ones(len(x), np.float32), params=False)

    def step(self, lr, b1=0.0, b2=0.999):
        self.t += 1
        for k, (p, g) in enumerate(zip(self.W + self.b, self.gW + self.gb)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g; self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            p -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + 1e-8)


# ── sampling: Langevin dynamics ──────────────────────────────
def langevin(E, x, steps, eta, rng, noise=True, keep=0):
    """x ← x − η∇E(x) + √(2η)ε. noise=False is plain gradient descent (it collapses into the minima).
    keep=k also returns every k-th state (for the video)."""
    x = x.copy(); traj = [x.copy()] if keep else None
    for s in range(1, steps + 1):
        x = x - eta * E.grad_x(x)
        if noise: x = x + math.sqrt(2 * eta) * rng.standard_normal(x.shape).astype(np.float32)
        x = np.clip(x, -R, R)
        if keep and s % keep == 0: traj.append(x.copy())
    return (x, traj) if keep else x


# ── exact likelihood — possible here only because 2-D is small enough to integrate on a grid ──
def log_Z(E, n=400, L=4.5):
    g = np.linspace(-L, L, n, dtype=np.float32); X, Y = np.meshgrid(g, g)
    e = -E(np.stack([X.ravel(), Y.ravel()], 1)).astype(np.float64); m = e.max()
    return float(m + np.log(np.exp(e - m).sum()) + 2 * np.log(g[1] - g[0]))


def nll(E, x):
    return float(E(x).mean() + log_Z(E))


# ── training: persistent contrastive divergence ──────────────
def train(data='ring', steps=4000, batch=256, lr=1e-3, eta=3e-3, k=60, alpha=0.01, hidden=96, seed=0,
          buffer=8192, snapshots=(), log=print):
    """Returns the energy net, a history of (mean E on data, mean E on fantasies) and any snapshots."""
    rng = np.random.default_rng(seed); E = Energy(hidden, seed)
    buf = rng.uniform(-R, R, (buffer, 2)).astype(np.float32); hist, snaps = [], {}
    for s in range(1, steps + 1):
        idx = rng.integers(0, buffer, batch)
        fresh = rng.random(batch) < 0.05                           # 5% restart from pure noise
        buf[idx[fresh]] = rng.uniform(-R, R, (fresh.sum(), 2))
        fake = langevin(E, buf[idx], k, eta, rng); buf[idx] = fake
        x = DATA[data](batch, rng)
        # loss = mean E(data) − mean E(fantasy) + α·(mean E(data)² + mean E(fantasy)²)
        Ed = E(x); E.backward((1 + 2 * alpha * Ed) / batch); gd = (E.gW, E.gb)
        Ef = E(fake); E.backward((-1 + 2 * alpha * Ef) / batch)
        E.gW = [a + b for a, b in zip(gd[0], E.gW)]; E.gb = [a + b for a, b in zip(gd[1], E.gb)]
        E.step(lr * (0.3 if s > 0.8 * steps else 1.0))
        hist.append((float(Ed.mean()), float(Ef.mean())))
        if s in snapshots: snaps[s] = {'fake': fake.copy(), 'data': x.copy(), 'W': [w.copy() for w in E.W], 'b': [b.copy() for b in E.b]}
        if s % (steps // 6) == 0:
            log(f'step {s:5d}   E(data) {Ed.mean():+6.2f}   E(fantasies) {Ef.mean():+6.2f}')
    test = DATA[data](5000, np.random.default_rng(seed + 99))
    res = nll(E, test)
    log(f'\nheld-out NLL (exact, by integrating e^−E over a 400×400 grid): {res:.3f} nats per point')
    return E, hist, res, snaps


# ── 1-D: sculpting a landscape with bumps (linear in θ, so every step is exact) ──
def E_true(x):  # the 1-D landscape from the end of episode 6
    return 0.12 * x ** 4 - 1.1 * x ** 2 + 0.35 * x + 3.0 + 0.4 * np.sin(3 * x)


def sculpt_1d(steps=160, n_data=200, lr=2.0, seed=0, L=3.3, l2=0.0):
    """E_w(x) = Σ w_k φ_k(x) with Gaussian bumps φ_k. Gradient of −log-likelihood w.r.t. w:
    mean φ(data) − mean φ(fantasies); fantasies are drawn exactly (1-D ⇒ Z on a grid)."""
    rng = np.random.default_rng(seed); g = np.linspace(-L, L, 661); dx = g[1] - g[0]
    c = np.linspace(-L, L, 25); phi = lambda x: np.exp(-0.5 * ((np.asarray(x)[:, None] - c) / 0.4) ** 2)
    pt = np.exp(-E_true(g)); cdf = np.cumsum(pt); cdf /= cdf[-1]
    data = np.interp(rng.random(n_data), cdf, g); w = np.zeros(len(c)); frames = []
    for s in range(steps + 1):
        Eg = phi(g) @ w; p = np.exp(-(Eg - Eg.min())); cdf = np.cumsum(p); cdf /= cdf[-1]
        fant = np.interp(rng.random(400), cdf, g)
        frames.append({'E': Eg.copy(), 'fant': fant[:40].copy()})
        grad = phi(data).mean(0) - phi(fant).mean(0) + l2 * w
        w -= lr * grad
    return g, data, frames


# ── gradient check ───────────────────────────────────────────
def grad_check():
    rng = np.random.default_rng(0); E = Energy(hidden=8, seed=1)
    E.W = [w.astype(np.float64) for w in E.W]; E.b = [b.astype(np.float64) for b in E.b]
    x, f = rng.standard_normal((5, 2)), rng.standard_normal((5, 2)); a = 0.1
    L = lambda: float(E(x).mean() - E(f).mean() + a * ((E(x) ** 2).mean() + (E(f) ** 2).mean()))
    Ed = E(x); E.backward((1 + 2 * a * Ed) / 5); gd = [g.copy() for g in E.gW + E.gb]
    Ef = E(f); E.backward((-1 + 2 * a * Ef) / 5); G = [p + q for p, q in zip(gd, E.gW + E.gb)]
    worst = 0
    for P, Gp in zip(E.W + E.b, G):
        idx = tuple(rng.integers(0, s) for s in P.shape); old = P[idx]
        P[idx] = old + 1e-6; lp = L(); P[idx] = old - 1e-6; lm = L(); P[idx] = old
        num = (lp - lm) / 2e-6; worst = max(worst, abs(num - Gp[idx]) / max(1e-9, abs(num) + abs(Gp[idx])))
    gx = E.grad_x(x); i = (3, 1); old = x[i]
    x[i] = old + 1e-6; ep = E(x)[3]; x[i] = old - 1e-6; em = E(x)[3]; x[i] = old
    worst = max(worst, abs((ep - em) / 2e-6 - gx[i]) / max(1e-9, abs(gx[i])))
    print(f'worst relative gradient error {worst:.1e} ({"OK" if worst < 1e-4 else "CHECK"})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', choices=list(DATA), default='ring')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--eta', type=float, default=3e-3, help='Langevin step size η')
    ap.add_argument('--k', type=int, default=60, help='Langevin steps per update')
    ap.add_argument('--alpha', type=float, default=0.01, help='E² penalty')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--sculpt', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        return grad_check()
    if args.sculpt:
        g, data, frames = sculpt_1d()
        for s in (0, 5, 20, 80, 160):
            Eg = frames[s]['E']; print(f'step {s:3d}  energy at data {np.interp(data, g, Eg).mean():+6.2f}   at fantasies {np.interp(frames[s]["fant"], g, Eg).mean():+6.2f}')
        return
    E, hist, _, _ = train(args.data, args.steps, eta=args.eta, k=args.k, alpha=args.alpha, seed=args.seed)
    if args.plot:
        import matplotlib.pyplot as plt
        g = np.linspace(-R, R, 200, dtype=np.float32); X, Y = np.meshgrid(g, g[::-1])
        e = E(np.stack([X.ravel(), Y.ravel()], 1)).reshape(200, 200)
        rng = np.random.default_rng(1); x = langevin(E, rng.uniform(-R, R, (2000, 2)).astype(np.float32), 300, args.eta, rng)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(e, extent=[-R, R, -R, R], cmap='magma_r'); ax[0].set_title('energy E(x)')
        ax[1].imshow(np.exp(-(e - e.min())), extent=[-R, R, -R, R], cmap='Blues'); ax[1].set_title('p(x) ∝ exp(−E)')
        ax[2].scatter(*x.T, s=2, c='#c4654a'); ax[2].set_xlim(-R, R); ax[2].set_ylim(-R, R); ax[2].set_title('Langevin samples from noise')
        plt.show()


if __name__ == '__main__':
    main()
