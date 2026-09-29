"""
Episode 11 — Guidance: asking a generator for what you want, in numpy, backprop by hand.

Episode 10's flow draws whatever the dice decide: a sample from p(x). Guidance asks for a sample from p(x | y),
"a 7", "blob 3", "the letter G". Three recipes, each one line of maths on top of the last:

1. Tell the network. Give the label y to the velocity net as an extra input and train exactly as before:
       v_θ(x, t, y) ≈ E[ x₁ − x₀ | x_t = x, y ]            (a flow for each label, one network)

2. Classifier guidance (Dhariwal & Nichol 2021). Bayes' rule, p(x | y) ∝ p(x)·p(y | x), becomes a sum of arrows once
   you take the log and its gradient (the score of episode 8):
       ∇ log p(x | y) = ∇ log p(x) + ∇ log p(y | x)
   Train a classifier that reads *noisy* points x_t, and add its arrow, turned up by w:
       ∇ log p(x) + w·∇ log p(y | x)      = the score of  p(x)·p(y | x)^w
   On our straight roads a score change becomes a velocity change: Δv = (1 − t)/t · Δ(score).

3. Classifier-free guidance (Ho & Salimans 2021). Bayes backwards: the classifier's arrow is the difference between
   the conditional and unconditional arrows, ∇ log p(y | x) = ∇ log p(x | y) − ∇ log p(x). One network trained with the
   label hidden 10% of the time (the "no label" slot, ∅) knows both, so no classifier is needed:
       v = v_∅ + w·(v_y − v_∅)            w = 0: ignore the request · w = 1: plain conditional · w > 1: insist
   Swap v_∅ for another label's arrow and it becomes a negative prompt: v = v_neg + w·(v_y − v_neg).

Requires:  pip install numpy          (MNIST is downloaded on first use by vae.py)

Try:
    python guidance.py --check                  # verify the hand-written gradients numerically
    python guidance.py                          # 8 blobs: conditional flow, classifier guidance, CFG at w = 0, 1, 4
"""
import argparse
import math

import numpy as np

import flow_matching as FM
from nn import TimeMLP, grad_check as check_grads, report


# ── a classifier that reads noisy points ──────────────────────
class NoisyClassifier(TimeMLP):
    """p_φ(y | x_t, t): logits from an nn.TimeMLP. Trained on points part-way along the roads, so its arrow
    ∇ₓ log p(y | x_t) means something at every t, not only on clean data."""

    def __init__(self, dim, k, hidden=(256, 256), ed=64, seed=0):
        super().__init__(dim, hidden, ed, seed, out=k); self.k = k
        self.Wo[...] = (np.random.default_rng(seed + 7).standard_normal(self.Wo.shape) * 0.01).astype(np.float32)

    def logits(self, x, t):
        return self.mlp(x, np.broadcast_to(np.asarray(t, np.float32), (len(x),)) * 999)

    def probs(self, x, t):
        z = self.logits(x, t); z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)

    def grad_log_p(self, x, t, y):
        """∇ₓ log p(y | x_t): back through log-softmax (d/dz = onehot(y) − p), then through the network to x."""
        p = self.probs(x, t); d = -p; d[np.arange(len(x)), y] += 1
        self.backward(d.astype(np.float32)); return self.dx


def train_classifier(clf, data_fn, steps, batch=256, lr=1e-3, seed=0, log=print, every=None):
    """Cross-entropy on x_t = (1 − t)·z + t·x, t ~ U(0, 1): the same noisy points the flow will pass through."""
    rng = np.random.default_rng(seed); hist = []
    for s in range(1, steps + 1):
        x, y = data_fn(batch, rng); z = rng.standard_normal(x.shape).astype(np.float32); t = rng.random(batch).astype(np.float32)
        xt = (1 - t)[:, None] * z + t[:, None] * x
        p = clf.probs(xt, t); loss = float(-np.log(p[np.arange(batch), y] + 1e-12).mean())
        d = p.copy(); d[np.arange(batch), y] -= 1; clf.backward((d / batch).astype(np.float32))
        clf.step(lr * (0.5 * (1 + math.cos(math.pi * s / steps))) + 1e-5); hist.append(loss)
        if s % (every or max(1, steps // 10)) == 0: log(f'step {s:6d}   cross-entropy {np.mean(hist[-200:]):.4f}')
    return hist


# ── sampling with a request ───────────────────────────────────
def guided_velocity(net, x, t, y, w=1.0, neg=None, clf=None, t_min=0.05):
    """The arrow to follow at (x, t) when asking for label y.
    clf given:  classifier guidance,  v = v_∅ + w·(1 − t)/t·∇ log p(y | x_t)
    otherwise:  classifier-free,      v = v_neg + w·(v_y − v_neg)     (neg = None: the 'no label' slot ∅)"""
    if clf is not None:
        return net(x, t, None) + w * (1 - t) / max(t, t_min) * clf.grad_log_p(x, t, y)
    v_neg = net(x, t, neg)
    if w == 0: return v_neg
    return v_neg + w * (net(x, t, y) - v_neg)


def sample(net, x, steps, y, w=1.0, neg=None, clf=None, keep_all=False):
    ts = np.linspace(0, 1, steps + 1).astype(np.float32); traj = [x.copy()]
    for k in range(steps):
        x = (x + (ts[k + 1] - ts[k]) * guided_velocity(net, x, ts[k], y, w, neg, clf)).astype(np.float32)
        if keep_all: traj.append(x.copy())
    return (x, traj) if keep_all else x


# ── 2-D data ──────────────────────────────────────────────────
def two_clouds(n, rng, sep=0.55, std=0.6):
    """Two overlapping classes: a point near the middle could be either. Where guidance's w earns its keep."""
    y = rng.integers(0, 2, n); c = np.where(y[:, None] == 0, [[-sep, 0.0]], [[sep, 0.0]])
    return (c + std * rng.standard_normal((n, 2))).astype(np.float32), y


def two_clouds_density(X, sep=0.55, std=0.6):
    """Exact p(x | y) for both classes on points X (for the 'what w asks for' contours)."""
    g = lambda c: np.exp(-((X - c) ** 2).sum(-1) / (2 * std ** 2)) / (2 * np.pi * std ** 2)
    return np.stack([g(np.array([-sep, 0.0])), g(np.array([sep, 0.0]))], -1)


# ── checks ────────────────────────────────────────────────────
def grad_check():
    rng = np.random.default_rng(0); clf = NoisyClassifier(3, 4, (8, 8), ed=6, seed=1)
    for i, p in enumerate(clf.params): clf.params[i][...] = p + 0.3 * rng.standard_normal(p.shape)
    clf.params = [p.astype(np.float64) for p in clf.params]
    clf.W, clf.U, clf.b = clf.params[:2], clf.params[2:4], clf.params[4:6]; clf.Wo, clf.bo = clf.params[6], clf.params[7]
    x = rng.standard_normal((5, 3)); t = rng.random(5); y = rng.integers(0, 4, 5)
    logp = lambda: float(np.log(clf.probs(x, t)[np.arange(5), y]).sum())
    # parameters: cross-entropy gradient
    p = clf.probs(x, t); d = p.copy(); d[np.arange(5), y] -= 1; clf.backward(d)
    report(check_grads(clf.params, clf.grads, lambda: -logp(), rng), 'noisy classifier, weights: ')
    # the input: ∇ₓ log p(y | x) — the arrow classifier guidance adds
    g = clf.grad_log_p(x, t, y); worst = 0
    for _ in range(6):
        i, j = rng.integers(0, 5), rng.integers(0, 3); x[i, j] += 1e-6; up = logp(); x[i, j] -= 2e-6; dn = logp(); x[i, j] += 1e-6
        num = (up - dn) / 2e-6; worst = max(worst, abs(num - g[i, j]) / max(1e-8, abs(num) + abs(g[i, j])))
    report(worst, 'noisy classifier, ∇ₓ log p(y | x): ')
    FM.grad_check()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true'); ap.add_argument('--steps', type=int, default=8000)
    a = ap.parse_args()
    if a.check: return grad_check()
    rng = np.random.default_rng(0)
    print('a conditional flow on the 8 blobs (label hidden 10% of the time) …')
    net = FM.VelocityNet(2, (256, 256, 256), seed=3, cond=8); FM.train(net, FM.ring8_labelled, a.steps, batch=512); net.use_ema()
    print('a classifier for noisy points …')
    clf = NoisyClassifier(2, 8, (128, 128)); train_classifier(clf, FM.ring8_labelled, a.steps // 2)
    c = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)
    hit = lambda x, y: float((np.linalg.norm(x[:, None] - c[None], axis=2).argmin(1) == y).mean())
    z = rng.standard_normal((1000, 2)).astype(np.float32); y = np.full(1000, 2)
    for nm, kw in [('no request (w = 0)', dict(w=0)), ('conditional (w = 1)', dict(w=1)), ('CFG w = 4', dict(w=4)),
                   ('classifier guidance w = 1', dict(w=1, clf=clf)), ('classifier guidance w = 4', dict(w=4, clf=clf))]:
        print(f'  asked for blob 3 · {nm:28s} → {hit(sample(net, z, 32, y, **kw), 2):5.1%} land on blob 3')


if __name__ == '__main__':
    main()
