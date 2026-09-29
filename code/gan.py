"""
Episode 05 — GANs from scratch in numpy (backprop written by hand).

    generator      G(z):  noise  ->  sample            never sees a single real example
    discriminator  D(x):  sample ->  logit of "real"   the judge from episode 04, now a teacher

    D step:  maximise  E_data[log D(x)] + E_z[log(1 - D(G(z)))]        (plain binary cross-entropy)
    G step:  maximise  E_z[log D(G(z))]                                 ("non-saturating" loss)
             — its only signal is the gradient of D, flowing back through x = G(z)

Requires:  pip install numpy          (matplotlib optional, for --plot)

Try:
    python gan.py --data ring                  # 8 Gaussians on a ring: watch it find (or lose) the modes
    python gan.py --data ring --seed 3 --lr-g 5e-3   # a less lucky run: mode collapse
    python gan.py --data mnist --epochs 30     # digits, with no pixel ever shown to the generator
    python gan.py --check                      # verify the hand-written gradients numerically
"""
import argparse
import os

import numpy as np

import nn
from nn import MLP, grad_check as check_grads

HERE = os.path.dirname(os.path.abspath(__file__))


# the networks are nn.MLP: a layer list like [('lin', 2, 128), 'lrelu', …], backprop by hand, Adam (β₁ = 0.5)
sig = lambda l: 1 / (1 + np.exp(-l))
softplus = lambda l: np.logaddexp(0, l)


def gan_step(G, D, x_real, z, lr_g, lr_d, saturating=False):
    """One round of the game. Returns (D loss, G loss, D(real), D(fake)) averaged over the batch."""
    B = len(x_real)
    # --- judge: real -> 1, fake -> 0 (binary cross-entropy on the logit l)
    x_fake = G.forward(z)
    D.zero_grad()
    l_real = D.forward(x_real); D.backward((sig(l_real) - 1) / B)          # d/dl softplus(-l)
    l_fake = D.forward(x_fake); D.backward(sig(l_fake) / B)                # d/dl softplus(l)
    d_loss = (softplus(-l_real) + softplus(l_fake)).mean()
    D.step(lr_d)
    # --- forger: make the judge say "real" (non-saturating loss  -log D(G(z)))
    x_fake = G.forward(z)
    l = D.forward(x_fake)
    # original minimax loss  log(1 - D(G(z)))  has gradient -sigmoid(l): ~0 exactly when G is bad (it 'saturates')
    dx = D.backward((-sig(l) if saturating else sig(l) - 1) / B, accumulate=False)   # gradient reaches the pixels…
    G.zero_grad(); G.backward(dx)                                          # …and flows on into G's weights
    G.step(lr_g)
    return float(d_loss), float(softplus(-l).mean()), float(sig(l_real).mean()), float(sig(l).mean())


# ── data ──────────────────────────────────────────────────────
def ring(n, rng, k=8, r=2.0, std=0.08):
    return nn.ring(n, rng, k, r, std)                   # 8 tight blobs on a circle


def modes_covered(x, k=8, r=2.0, std=0.08):
    """How many of the ring's 8 modes get at least 2% of the samples (within 4 std)?"""
    c = np.stack([r * np.cos(np.arange(k) * 2 * np.pi / k), r * np.sin(np.arange(k) * 2 * np.pi / k)], 1)
    d = np.linalg.norm(x[:, None] - c[None], axis=2)
    near = d.min(1) < 4 * std
    counts = np.bincount(d.argmin(1)[near], minlength=k)
    return int((counts > 0.02 * len(x)).sum()), float(near.mean())


def train_2d(steps=6000, seed=0, lr_g=1e-3, lr_d=1e-3, batch=256, snap_every=100, log=print, saturating=False, hidden=128):
    rng = np.random.default_rng(seed)
    G = MLP([('lin', 2, hidden), 'lrelu', ('lin', hidden, hidden), 'lrelu', ('lin', hidden, 2)], rng)
    D = MLP([('lin', 2, 128), 'lrelu', ('lin', 128, 128), 'lrelu', ('lin', 128, 1)], rng)
    z_fixed = rng.standard_normal((600, 2)).astype(np.float32)
    gx = np.linspace(-3, 3, 48, dtype=np.float32)
    grid = np.stack(np.meshgrid(gx, gx[::-1]), -1).reshape(-1, 2)
    snaps = []
    for s in range(steps + 1):
        if s % snap_every == 0:
            xg = G.forward(z_fixed)
            field = sig(D.forward(grid)).reshape(48, 48)
            l = D.forward(xg); dx = D.backward((sig(l) - 1) / len(xg), accumulate=False)
            m, frac = modes_covered(xg)
            snaps.append({'step': s, 'x': xg.copy(), 'field': field, 'grad': -dx[:60] * len(xg), 'modes': m, 'on': frac})
            if s % (snap_every * 10) == 0:
                log(f'step {s:5d}   modes covered {m}/8   samples on a mode {frac:.0%}')
        if s < steps:
            gan_step(G, D, ring(batch, rng), rng.standard_normal((batch, 2)).astype(np.float32), lr_g, lr_d, saturating)
    return G, D, snaps


def train_mnist(epochs=30, seed=0, lr=2e-4, batch=128, zdim=64, log=print):
    from vae import load_mnist
    X, _ = load_mnist('train')
    rng = np.random.default_rng(seed)
    G = MLP([('lin', zdim, 256), 'lrelu', ('lin', 256, 512), 'lrelu', ('lin', 512, 784), 'sigmoid'], rng)
    D = MLP([('lin', 784, 512), 'lrelu', ('lin', 512, 256), 'lrelu', ('lin', 256, 1)], rng)
    z_fixed = rng.standard_normal((16, zdim)).astype(np.float32)
    snaps, hist = [G.forward(z_fixed).copy()], []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(X)); stats = []
        for i in range(0, len(X) - batch + 1, batch):
            stats.append(gan_step(G, D, X[order[i:i + batch]], rng.standard_normal((batch, zdim)).astype(np.float32), lr, lr))
        s = np.mean(stats, 0); hist.append({'epoch': ep, 'd_loss': s[0], 'g_loss': s[1], 'd_real': s[2], 'd_fake': s[3]})
        snaps.append(G.forward(z_fixed).copy())
        log(f'epoch {ep:2d}   D loss {s[0]:.3f}   G loss {s[1]:.3f}   D(real) {s[2]:.2f}   D(fake) {s[3]:.2f}')
    return G, D, snaps, hist


def sample(G, n, zdim=64, seed=1):
    return G.forward(np.random.default_rng(seed).standard_normal((n, zdim)).astype(np.float32))


def grad_check():
    rng = np.random.default_rng(0)
    for spec in ([('lin', 3, 5), 'lrelu', ('lin', 5, 2), 'sigmoid'], [('lin', 3, 4), 'lrelu', ('lin', 4, 1)]):
        net = MLP(spec, rng); net.P = {k: v.astype(np.float64) for k, v in net.P.items()}
        x = rng.standard_normal((4, 3)); w = rng.standard_normal((4, spec[-2][2] if spec[-1] == 'sigmoid' else 1))
        f = lambda: float((net.forward(x) * w).sum())
        f(); net.zero_grad(); dx = net.backward(w)
        worst = check_grads(list(net.P.values()), [net.G[k] for k in net.P], f, rng)
        i = (1, 2); old = x[i]; x[i] = old + 1e-6; a = f(); x[i] = old - 1e-6; b = f(); x[i] = old
        worst = max(worst, abs((a - b) / 2e-6 - dx[i]) / max(1e-9, abs(dx[i])))
        print(f'{len(spec)}-layer net: worst relative gradient error {worst:.1e} ({"OK" if worst < 1e-4 else "CHECK"})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', choices=['ring', 'mnist'], default='ring')
    ap.add_argument('--steps', type=int, default=6000, help='2-D only')
    ap.add_argument('--epochs', type=int, default=30, help='MNIST only')
    ap.add_argument('--lr-g', type=float, default=1e-3)
    ap.add_argument('--lr-d', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        return grad_check()
    if args.data == 'ring':
        G, D, snaps = train_2d(args.steps, args.seed, args.lr_g, args.lr_d)
        if args.plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            for a, s in zip(ax, [snaps[i] for i in np.linspace(0, len(snaps) - 1, 4).astype(int)]):
                a.imshow(s['field'], extent=[-3, 3, -3, 3], cmap='Blues', vmin=0, vmax=1)
                a.scatter(*s['x'].T, s=3, c='#c4654a'); a.set_title(f"step {s['step']} · {s['modes']}/8 modes"); a.axis('off')
            plt.show()
    else:
        G, D, snaps, _ = train_mnist(args.epochs, args.seed)
        if args.plot:
            import matplotlib.pyplot as plt
            x = sample(G, 64).reshape(8, 8, 28, 28).transpose(0, 2, 1, 3).reshape(224, 224)
            plt.imshow(x, cmap='gray'); plt.axis('off'); plt.show()


if __name__ == '__main__':
    main()
