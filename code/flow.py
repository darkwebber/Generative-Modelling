"""
Episode 06 — Normalizing flows (RealNVP-style affine couplings) in numpy, backprop by hand.

A flow is an invertible map f: data x -> noise z, built from K simple layers.
Change of variables (episode 1) gives the EXACT log-likelihood:

    log p(x) = log N(f(x); 0, I) + sum_k log |det J_k|

Coupling layer (2-D): keep one coordinate a, transform the other b:
    forward   b' = b * exp(s(a)) + t(a)          a' = a
    inverse   b  = (b' - t(a)) * exp(-s(a))
    log|det|  = s(a)                             (the Jacobian is triangular!)
s and t are ordinary neural nets — they never need to be inverted.

Requires:  pip install numpy          (matplotlib optional, for --plot)

Try:
    python flow.py                       # two moons: train by maximum likelihood, report exact NLL
    python flow.py --data ring --steps 12000 --seed 1 --plot   # 8 blobs: the thin "bridges" a flow cannot tear
    python flow.py --layers 2            # too few layers: see what a shallow flow can't bend
    python flow.py --check               # verify the hand-written gradients numerically
"""
import argparse
import math

import numpy as np

from gan import MLP

LOG2PI = math.log(2 * math.pi)


# ── data ──────────────────────────────────────────────────────
def moons(n, rng, noise=0.08):
    th = np.pi * rng.random(n); up = rng.random(n) < 0.5
    x = np.where(up, np.cos(th) - 0.5, 0.5 - np.cos(th)); y = np.where(up, np.sin(th) - 0.25, 0.25 - np.sin(th))
    return (np.stack([x, y], 1) * 1.5 + noise * rng.standard_normal((n, 2))).astype(np.float32)


def ring(n, rng, k=8, r=2.0, std=0.12):
    a = rng.integers(0, k, n) * 2 * np.pi / k
    return (np.stack([r * np.cos(a), r * np.sin(a)], 1) + std * rng.standard_normal((n, 2))).astype(np.float32)


DATA = {'moons': moons, 'ring': ring}


# ── the flow ──────────────────────────────────────────────────
class Flow:
    def __init__(self, layers=8, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.nets = [MLP([('lin', 1, hidden), 'lrelu', ('lin', hidden, hidden), 'lrelu', ('lin', hidden, 2)], rng, scale=0.3)
                     for _ in range(layers)]
        self.masks = [k % 2 for k in range(layers)]          # which coordinate conditions the other

    @staticmethod
    def _st(raw):
        s = 2 * np.tanh(raw[:, 0] / 2)                      # soft-clamped log-scale
        return s, raw[:, 1]

    def forward(self, x, keep=False):
        """data -> noise. Returns z, sum of log|det|, and (optionally) every intermediate stage."""
        stages, logdet, self.cache = [x], np.zeros(len(x), np.float32), []
        for net, m in zip(self.nets, self.masks):
            a, b = x[:, m:m + 1], x[:, 1 - m]
            raw = net.forward(a); s, t = self._st(raw)
            nb = b * np.exp(s) + t
            self.cache.append((a, b, raw, s))
            x = x.copy(); x[:, 1 - m] = nb
            logdet = logdet + s
            if keep: stages.append(x)
        return x, logdet, stages

    def inverse(self, z, keep=False):
        """noise -> data (sampling): undo the layers in reverse order."""
        stages = [z]
        for net, m in zip(reversed(self.nets), reversed(self.masks)):
            a = z[:, m:m + 1]; s, t = self._st(net.forward(a))
            z = z.copy(); z[:, 1 - m] = (z[:, 1 - m] - t) * np.exp(-s)
            if keep: stages.append(z)
        return z, stages

    def log_prob(self, x):
        z, logdet, _ = self.forward(x)
        return -0.5 * (z ** 2).sum(1) - LOG2PI + logdet

    def loss_and_grads(self, x):
        """NLL = mean( ½|z|² + log 2π − Σ s ). Every gradient by hand, layer by layer in reverse."""
        N = len(x)
        z, logdet, _ = self.forward(x)
        loss = float((0.5 * (z ** 2).sum(1) + LOG2PI - logdet).mean())
        dx = z / N                                             # d/dz of ½|z|²
        for net, m, (a, b, raw, s) in zip(reversed(self.nets), reversed(self.masks), reversed(self.cache)):
            dnb = dx[:, 1 - m]                                 # gradient arriving at b'
            db = dnb * np.exp(s)
            ds = dnb * b * np.exp(s) - 1.0 / N                 # −1/N from the −log|det| term
            dt = dnb
            draw = np.stack([ds * (1 - np.tanh(raw[:, 0] / 2) ** 2), dt], 1)
            net.zero_grad(); da = net.backward(draw)[:, 0]
            nd = dx.copy(); nd[:, 1 - m] = db; nd[:, m] = dx[:, m] + da
            dx = nd
        return loss, dx

    def step(self, lr):
        for net in self.nets:
            net.step(lr, b1=0.9)


def train(data='moons', layers=8, steps=4000, batch=256, lr=2e-3, seed=0, log=print):
    rng = np.random.default_rng(seed)
    flow = Flow(layers, seed=seed); hist = []
    for s in range(1, steps + 1):
        x = DATA[data](batch, rng)
        # gradients are accumulated per layer inside loss_and_grads; step each net afterwards
        loss, _ = flow.loss_and_grads(x)
        flow.step(lr * (0.3 if s > 0.7 * steps else 1.0))
        if s % 50 == 0: hist.append(loss)
        if s % (steps // 8) == 0:
            log(f'step {s:5d}   training NLL {loss:.3f} nats per point')
    test = DATA[data](5000, np.random.default_rng(seed + 99))
    nll = float(-flow.log_prob(test).mean())
    mu, C = test.mean(0), np.cov(test, rowvar=False)
    d = test - mu; gauss = float((0.5 * np.einsum('ni,ij,nj->n', d, np.linalg.inv(C), d) + LOG2PI + 0.5 * np.log(np.linalg.det(C))).mean())
    log(f'\nheld-out NLL — flow: {nll:.3f}   best single Gaussian: {gauss:.3f}   (nats per point, lower = better)')
    return flow, hist, nll, gauss


def grad_check():
    rng = np.random.default_rng(0)
    flow = Flow(layers=3, hidden=8, seed=1)
    for net in flow.nets: net.P = {k: v.astype(np.float64) for k, v in net.P.items()}
    x = rng.standard_normal((6, 2))
    loss, dx = flow.loss_and_grads(x)
    grads = [{k: v.copy() for k, v in net.G.items()} for net in flow.nets]
    worst = 0
    for li, net in enumerate(flow.nets):
        for k in net.P:
            idx = tuple(rng.integers(0, s) for s in net.P[k].shape); old = net.P[k][idx]
            net.P[k][idx] = old + 1e-6; lp = flow.loss_and_grads(x)[0]; net.P[k][idx] = old - 1e-6; lm = flow.loss_and_grads(x)[0]; net.P[k][idx] = old
            num = (lp - lm) / 2e-6; an = grads[li][k][idx]
            worst = max(worst, abs(num - an) / max(1e-9, abs(num) + abs(an)))
    i = (2, 1); old = x[i]; x[i] = old + 1e-6; lp = flow.loss_and_grads(x)[0]; x[i] = old - 1e-6; lm = flow.loss_and_grads(x)[0]; x[i] = old
    worst = max(worst, abs((lp - lm) / 2e-6 - dx[i]) / max(1e-9, abs(dx[i])))
    z, _, _ = flow.forward(x); xr, _ = flow.inverse(z)
    print(f'worst relative gradient error {worst:.1e} ({"OK" if worst < 1e-4 else "CHECK"}) · inverse error {np.abs(xr - x).max():.1e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', choices=list(DATA), default='moons')
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        return grad_check()
    flow, _, _, _ = train(args.data, args.layers, args.steps, seed=args.seed)
    if args.plot:
        import matplotlib.pyplot as plt
        g = np.linspace(-3.5, 3.5, 200, dtype=np.float32); X, Y = np.meshgrid(g, g[::-1])
        lp = flow.log_prob(np.stack([X.ravel(), Y.ravel()], 1)).reshape(200, 200)
        x, _ = flow.inverse(np.random.default_rng(1).standard_normal((2000, 2)).astype(np.float32))
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].imshow(np.exp(lp), extent=[-3.5, 3.5, -3.5, 3.5], cmap='Blues'); ax[0].set_title('exact density p(x)')
        ax[1].scatter(*x.T, s=2, c='#c4654a'); ax[1].set_xlim(-3.5, 3.5); ax[1].set_ylim(-3.5, 3.5); ax[1].set_title('samples  x = f⁻¹(z)')
        plt.show()


if __name__ == '__main__':
    main()
