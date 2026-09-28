"""
Episode 04 — How do you grade a generator?  Every metric from the video, in numpy.

We evaluate five "models" on MNIST, each producing 2,000 samples:
    real       held-out real digits           (the ceiling: a perfect model)
    vae        our 2-D VAE from episode 03    (honest, a bit blurry)
    copier     2,000 training images          (a photocopier: memorised, not generated)
    ones       2,000 training images of "1"   (a one-trick pony: mode collapse)
    noise      uniform random pixels          (the floor)

and score each with:
    IS         Inception Score       exp( E_x KL( p(y|x) || p(y) ) )            higher = better   (max 10)
    FID        Fréchet distance      |mu_r - mu_g|^2 + Tr(S_r + S_g - 2 (S_r S_g)^1/2)   lower = better
    precision  fraction of samples that land on the real-data manifold          (fidelity)
    recall     fraction of real data covered by the samples' manifold           (diversity)
    NN-train   median distance from a sample to its nearest TRAINING image       (novelty; 0 = copied)
    C2ST       accuracy of a classifier trained to tell samples from real data   (50% = indistinguishable)

"Features" come from a small MNIST classifier we train here (784 -> 256 -> 64 -> 10) — the
MNIST stand-in for the Inception network used on photos.

Requires:  pip install numpy
Run:       python evals.py                  # trains the feature net (+ the VAE if needed), prints the scorecard
"""
import os

import numpy as np

from vae import decode, load_mnist, train as train_vae

HERE = os.path.dirname(os.path.abspath(__file__))
N = 2000


# ── a feature extractor: a small MNIST classifier, backprop by hand ─────────
def train_classifier(X, y, epochs=6, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    he = lambda a, b: (rng.normal(0, 1, (a, b)) * np.sqrt(2 / a)).astype(np.float32)
    P = {'W1': he(784, 256), 'b1': np.zeros(256, np.float32), 'W2': he(256, 64), 'b2': np.zeros(64, np.float32),
         'W3': he(64, 10), 'b3': np.zeros(10, np.float32)}
    m = {k: np.zeros_like(v) for k, v in P.items()}; s = {k: np.zeros_like(v) for k, v in P.items()}; step = 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), 128):
            idx = order[i:i + 128]
            xb, yb = X[idx], y[idx]; B = len(xb)
            a1 = xb @ P['W1'] + P['b1']; h1 = np.maximum(0, a1)
            a2 = h1 @ P['W2'] + P['b2']; h2 = np.maximum(0, a2)
            z = h2 @ P['W3'] + P['b3']; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            dz = p; dz[np.arange(B), yb] -= 1; dz /= B
            G = {'W3': h2.T @ dz, 'b3': dz.sum(0)}
            d2 = (dz @ P['W3'].T) * (a2 > 0); G['W2'] = h1.T @ d2; G['b2'] = d2.sum(0)
            d1 = (d2 @ P['W2'].T) * (a1 > 0); G['W1'] = xb.T @ d1; G['b1'] = d1.sum(0)
            step += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * G[k]; s[k] = 0.999 * s[k] + 0.001 * G[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(s[k] / (1 - 0.999 ** step)) + 1e-8)
    return P


def load_classifier(Xtr, ytr, verbose=True):
    path = os.path.join(HERE, 'data', 'ep04_clf.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    if verbose: print('training the feature network (our "Inception") …')
    P = train_classifier(Xtr, ytr)
    np.savez(path, **P)
    return P


def features(P, X):
    """64-d penultimate activations = a 'perceptual' space where similar-looking digits are close."""
    h = np.maximum(0, X @ P['W1'] + P['b1'])
    return np.maximum(0, h @ P['W2'] + P['b2'])


def class_probs(P, X):
    z = features(P, X) @ P['W3'] + P['b3']; z -= z.max(1, keepdims=True); p = np.exp(z)
    return p / p.sum(1, keepdims=True)


# ── the metrics ───────────────────────────────────────────────────────────
def inception_score(p_yx):
    """exp(E_x KL(p(y|x) || p(y))): sharp per-sample predictions AND a spread-out mix of classes."""
    p_y = p_yx.mean(0, keepdims=True)
    kl = (p_yx * (np.log(p_yx + 1e-12) - np.log(p_y + 1e-12))).sum(1)
    return float(np.exp(kl.mean()))


def _sqrtm_psd(A):
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def fid(f_real, f_gen):
    """Fit a Gaussian to each feature cloud; distance = centres apart + shapes mismatched."""
    mu1, mu2 = f_real.mean(0), f_gen.mean(0)
    S1, S2 = np.cov(f_real, rowvar=False), np.cov(f_gen, rowvar=False)
    r1 = _sqrtm_psd(S1)
    cross = np.sqrt(np.clip(np.linalg.eigvalsh(r1 @ S2 @ r1), 0, None)).sum()   # Tr((S1 S2)^1/2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * cross)


def _pdist(A, B):
    return np.sqrt(np.clip((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0, None))


def precision_recall(f_real, f_gen, k=3):
    """Kynkäänniemi et al. 2019: a set's 'manifold' = union of balls reaching each point's k-th neighbour."""
    def radii(F):
        d = _pdist(F, F); d.sort(1); return d[:, k]
    r_real, r_gen = radii(f_real), radii(f_gen)
    precision = (_pdist(f_gen, f_real) <= r_real[None]).any(1).mean()   # samples inside the real manifold
    recall = (_pdist(f_real, f_gen) <= r_gen[None]).any(1).mean()       # real points inside the sample manifold
    return float(precision), float(recall)


def nn_train_distance(X_gen, X_train, n=300):
    """Median pixel distance from a sample to its nearest training image. 0 means copied."""
    d = np.full(n, np.inf)
    for i in range(0, len(X_train), 10000):
        d = np.minimum(d, _pdist(X_gen[:n], X_train[i:i + 10000]).min(1))
    return float(np.median(d))


def c2st(f_real, f_gen, steps=400, seed=0):
    """Classifier two-sample test: train logistic regression real-vs-sample, report held-out accuracy."""
    rng = np.random.default_rng(seed)
    X = np.vstack([f_real, f_gen]); y = np.r_[np.ones(len(f_real)), np.zeros(len(f_gen))]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    idx = rng.permutation(len(X)); tr, te = idx[:len(X) // 2], idx[len(X) // 2:]
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X[tr] @ w + b))); g = p - y[tr]
        w -= 0.5 * (X[tr].T @ g / len(tr) + 1e-3 * w); b -= 0.5 * g.mean()
    return float((((X[te] @ w + b) > 0) == y[te]).mean())


# ── the five "models" ─────────────────────────────────────────────────────
def load_vae():
    path = os.path.join(HERE, 'data', 'ep03_vae.npz')
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files if k != 'history'}
    print('training the episode-03 VAE first …')
    return train_vae('vae', 30)[0]


def sample_sets(seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr = load_mnist('train'); Xte, yte = load_mnist('test')
    vae = load_vae()
    # shuffle the test set first: its two halves were written by different groups of people,
    # and an unshuffled split would make "real vs real" look distinguishable
    perm = rng.permutation(len(Xte))
    ref, held = Xte[perm[:N]], Xte[perm[N:2 * N]]
    sets = {
        'real': held,
        'vae': decode(vae, rng.standard_normal((N, 2)).astype(np.float32)),
        'copier': Xtr[rng.permutation(len(Xtr))[:N]],
        'ones': Xtr[np.where(ytr == 1)[0][:N]],
        'noise': rng.random((N, 784)).astype(np.float32),
    }
    return sets, Xtr, ytr, ref, Xte, yte


def scorecard(verbose=True):
    sets, Xtr, ytr, ref, Xte, yte = sample_sets()
    clf = load_classifier(Xtr, ytr, verbose)
    acc = float((class_probs(clf, Xte).argmax(1) == yte).mean())
    if verbose: print(f'feature net test accuracy: {acc:.1%}\n')
    f_ref = features(clf, ref)
    rows = {}
    for name, X in sets.items():
        f = features(clf, X)
        p, r = precision_recall(f_ref, f)
        rows[name] = {'IS': inception_score(class_probs(clf, X)), 'FID': fid(f_ref, f), 'precision': p, 'recall': r,
                      'NN_train': nn_train_distance(X, Xtr), 'C2ST': c2st(f_ref, f)}
    if verbose:
        print(f"{'model':8s} {'IS ↑':>7s} {'FID ↓':>9s} {'prec ↑':>8s} {'recall ↑':>9s} {'NN-train':>9s} {'C2ST (50%=best)':>16s}")
        for name, m in rows.items():
            print(f"{name:8s} {m['IS']:7.2f} {m['FID']:9.1f} {m['precision']:8.2f} {m['recall']:9.2f} {m['NN_train']:9.2f} {m['C2ST']:15.0%}")
        print('\nNo single column crowns the right winner. Read them together — and look at the samples.')
    return rows, sets, clf, acc


if __name__ == '__main__':
    scorecard()
