"""
Episode 10 — Flow matching (and rectified flow / reflow) in numpy, backprop by hand.

Generation as motion: move every noise point along a velocity field, dx/dt = v(x, t), from t = 0 (noise) to t = 1 (data).
The simplest road between a noise point x₀ ~ N(0, I) and a data point x₁ is a straight line:

    x_t = (1 − t)·x₀ + t·x₁              velocity  d/dt x_t = x₁ − x₀          (constant: a straight road at constant speed)

Random pairs give roads that cross, so at (x, t) many velocities are possible. Regression learns their average,

    L(θ) = E_{t, x₀, x₁} ‖ v_θ(x_t, t) − (x₁ − x₀) ‖²        →   v*(x, t) = E[ x₁ − x₀ | x_t = x ]

and that averaged field carries N(0, I) exactly onto the data (Lipman et al. 2022; Liu et al. 2022; Albergo &
Vanden-Eijnden 2022). Sampling: Euler steps x ← x + Δt·v_θ(x, t), as few or as many as you like.

Reflow (Liu et al. 2022): run the trained flow to pair each noise point z with the data point it lands on, then train
again on those pairs. The new pairs never cross, so the new roads are nearly straight — and one or two steps suffice.

For MNIST the network guesses the clean image, x̂₁ = tanh(F), and reports v_θ = (x̂₁ − x)/(1 − t): the same model,
rearranged (as in episode 9), because an MLP finds a clean image far easier to produce than a noisy velocity.
Its loss, rewritten on x̂₁, is ‖x̂₁ − x₁‖² / (1 − t)²; we cap that weight at γ so the nearly finished steps don't dominate.

Requires:  pip install numpy          (MNIST is downloaded on first use by vae.py)

Try:
    python flow_matching.py --data ring              # 2-D: train, then sample with 1, 2, 4, 8 and 100 Euler steps; then reflow
    python flow_matching.py --data mnist --steps 20000 --plot
    python flow_matching.py --check                  # verify the hand-written gradients numerically
"""
import argparse
import math

import numpy as np

from nn import TimeMLP, grad_check as check_grads, report, ring  # noqa: F401


class VelocityNet(TimeMLP):
    """v_θ(x, t), t ∈ [0, 1]: an nn.TimeMLP (t embedded into every hidden layer, EMA weights).
    x1: guess the clean point x̂₁ = tanh(F) instead and report v = (x̂₁ − x)/(1 − t).
    cond: number of classes; the label (one-hot, plus a 'no label' slot) is appended to the input — episode 11's knob."""
    x1, gamma, cond = False, 5.0, 0

    def __init__(self, dim, hidden=(256, 256, 256), ed=64, seed=0, ff=0, x1=False, gamma=5.0, cond=0):
        super().__init__(dim + (cond + 1 if cond else 0), hidden, ed, seed, ff, out=dim)
        self.x1, self.gamma, self.cond, self.out_dim = x1, gamma, cond, dim

    def _in(self, x, y):
        if not self.cond: return x
        oh = np.zeros((len(x), self.cond + 1), np.float32); oh[np.arange(len(x)), self.cond if y is None else y] = 1
        return np.concatenate([x, oh], 1)

    def __call__(self, x, t, y=None):
        t = np.broadcast_to(np.asarray(t, np.float32), (len(x),))
        F = self.mlp(self._in(x, y), t * 999)                     # the time embedding expects 0 … 999
        if self.x1:
            self.th = np.tanh(F); self.om = np.maximum(1 - t, 1e-3)[:, None].astype(np.float32)
            return ((self.th - x) / self.om).astype(x.dtype)
        return F

    def backward(self, d):
        if self.x1: d = d / self.om * (1 - self.th ** 2)                # back through v = (tanh F − x)/(1 − t)
        super().backward(d)


def pairs_random(x1, rng):
    """Independent pairs: a fresh noise point for every data point (the roads cross)."""
    return rng.standard_normal(x1.shape).astype(np.float32), x1


def loss_and_grad(net, x0, x1, rng, y=None):
    n = len(x1); t = rng.random(n).astype(np.float32)
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1; u = x1 - x0            # the point on the road, and the road's velocity
    r = net(xt, t, y) - u
    if net.x1:            # ‖v̂ − u‖² = ‖x̂₁ − x₁‖²/(1 − t)²: cap that weight at γ, so nearly finished steps don't dominate
        w = np.minimum(1.0, net.gamma * (1 - t) ** 2)[:, None]; return float((w * r ** 2).mean() * x1.shape[1]), 2 * w * r / n
    return float((r ** 2).mean() * x1.shape[1]), 2 * r / n


def train(net, data_fn, steps, batch=256, lr=1e-3, seed=0, log=print, every=None, pairs=None, labels=None, p_drop=0.1):
    """data_fn(n, rng) → x₁ (or (x₁, y) when the net is conditional). pairs: (Z, X) arrays to train on instead
    (reflow: each noise point with the data point the previous flow sent it to)."""
    rng = np.random.default_rng(seed); hist = []
    for s in range(1, steps + 1):
        y = None
        if pairs is not None:
            i = rng.integers(0, len(pairs[0]), batch); x0, x1 = pairs[0][i], pairs[1][i]
        else:
            b = data_fn(batch, rng)
            if net.cond: b, y = b; y = np.where(rng.random(batch) < p_drop, net.cond, y)     # sometimes hide the label
            x0, x1 = pairs_random(b, rng)
        loss, d = loss_and_grad(net, x0, x1, rng, y); net.backward(d)
        net.step(lr * (0.5 * (1 + math.cos(math.pi * s / steps))) + 1e-5); hist.append(loss)
        if s % (every or max(1, steps // 10)) == 0: log(f'step {s:6d}   loss {np.mean(hist[-200:]):.4f}')
    return hist


# ── sampling: Euler steps along the learned field ─────────────
def euler(net, x, steps, keep_all=False, y=None, w=None):
    """Integrate dx/dt = v_θ(x, t) from t = 0 to 1 with `steps` equal Euler steps.
    y: class labels (conditional nets). w: guidance scale (episode 11): v = v_∅ + w·(v_y − v_∅)."""
    ts = np.linspace(0, 1, steps + 1).astype(np.float32); traj = [x.copy()]
    for k in range(steps):
        t, dt = ts[k], ts[k + 1] - ts[k]
        v = net(x, t, y)
        if w is not None: v = net(x, t, None) + w * (v - net(x, t, None))
        x = (x + dt * v).astype(np.float32)
        if keep_all: traj.append(x.copy())
    return (x, traj) if keep_all else x


def straightness(traj):
    """How straight are the roads? 1 − (mean deviation from the chord) / (chord length); 1 = perfectly straight."""
    T = np.stack(traj); a, b = T[0], T[-1]; u = np.linspace(0, 1, len(T))[:, None, None]
    chord = a[None] + u * (b - a)[None]
    return float(1 - np.linalg.norm(T - chord, axis=2).mean() / (np.linalg.norm(b - a, axis=1).mean() + 1e-9))


def reflow_pairs(net, n, steps=100, seed=0, dim=None, batch=4096):
    """Noise points and where the flow sends them: the non-crossing pairs for the next round."""
    rng = np.random.default_rng(seed); Z = rng.standard_normal((n, dim or net.out_dim)).astype(np.float32)
    X = np.concatenate([euler(net, Z[i:i + batch], steps) for i in range(0, n, batch)])
    return Z, X


# ── data ──────────────────────────────────────────────────────
def ring8(n, rng):
    """The season's 8 blobs on a circle of radius 2 (as in episodes 5–9), scaled by ½."""
    return ring(n, rng) / 2


def ring8_labelled(n, rng):
    a = rng.integers(0, 8, n); th = a * np.pi / 4
    x = (np.stack([2 * np.cos(th), 2 * np.sin(th)], 1) + 0.12 * rng.standard_normal((n, 2))).astype(np.float32) / 2
    return x, a


def mnist_fn(labels=False):
    import vae
    X, y = vae.load_mnist('train'); X = X * 2 - 1
    if labels: return lambda n, rng: (lambda i: (X[i], y[i]))(rng.integers(0, len(X), n))
    return lambda n, rng: X[rng.integers(0, len(X), n)]


def grad_check():
    for x1 in (False, True):
        for cond in (0, 3): _grad_check(x1, cond)


def _grad_check(x1, cond):
    rng = np.random.default_rng(0); net = VelocityNet(3, (8, 8), ed=6, seed=1, x1=x1, cond=cond)
    for i, p in enumerate(net.params): net.params[i][...] = p + 0.3 * rng.standard_normal(p.shape)
    net.params = [p.astype(np.float64) for p in net.params]
    net.W, net.U, net.b = net.params[:2], net.params[2:4], net.params[4:6]; net.Wo, net.bo = net.params[6], net.params[7]
    x = rng.standard_normal((5, 3)); t = rng.random(5) * 0.9; y = rng.standard_normal((5, 3)); lab = rng.integers(0, cond + 1, 5) if cond else None
    L = lambda: float(((net(x, t, lab) - y) ** 2).sum())
    out = net(x, t, lab); net.backward(2 * (out - y))
    report(check_grads(net.params, net.grads, L, rng), f'{"x̂₁ head" if x1 else "v head"}{", conditional" if cond else ""}: ')


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
        net = VelocityNet(2, (256, 256, 256)); train(net, ring8, a.steps or 8000, batch=512); net.use_ema()
        z = np.random.default_rng(1).standard_normal((3000, 2)).astype(np.float32)
        def show(net, name):
            for k in (1, 2, 4, 8, 100):
                x, tr = euler(net, z.copy(), k, keep_all=True); r = np.linalg.norm(x * 2, axis=1)
                print(f'{name:10s} {k:3d} steps   radius {r.mean():.2f} ± {r.std():.2f}   (data: 2.00 ± 0.12)   straightness {straightness(euler(net, z.copy(), 100, True)[1]):.3f}')
        show(net, '1-rectified')
        Z, X = reflow_pairs(net, 50000); net2 = VelocityNet(2, (256, 256, 256), seed=1)
        train(net2, None, a.steps or 8000, batch=512, pairs=(Z, X)); net2.use_ema(); show(net2, 'reflowed')
    else:
        net = VelocityNet(784, (1024, 1024), x1=True); train(net, mnist_fn(), a.steps or 30000, batch=128, lr=1e-3); net.use_ema()
        x = euler(net, np.random.default_rng(1).standard_normal((64, 784)).astype(np.float32), 32)
        if a.plot:
            import matplotlib.pyplot as plt
            g = ((x.reshape(8, 8, 28, 28).transpose(0, 2, 1, 3).reshape(224, 224) + 1) / 2).clip(0, 1)
            plt.imshow(g, cmap='gray'); plt.axis('off'); plt.show()


if __name__ == '__main__':
    main()
