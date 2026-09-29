"""
Episode 01 — a density you can fit and sample from: a mixture of bell curves, by maximum likelihood.

    p(x) = Σₖ πₖ · N(x; μₖ, σₖ²)          a flexible shape; its knobs θ are the weights π, centres μ, widths σ
    fit:     maximise Σᵢ log p(xᵢ)         (EM below: every step can only raise the likelihood)
    sample:  pick bump k with probability πₖ, then x = μₖ + σₖ·z with z ~ N(0, 1)   — simple noise, bent into data

The data is the episode's own: 420 points from two bumps (40% at −1.2, width 0.5; 60% at 1.1, width 0.7).

Requires:  pip install numpy

Try:
    python density.py              # fit 1, 2, 3 and 4 bumps; compare their log-likelihoods; draw new samples
    python density.py --check      # EM never lowers the likelihood; samples follow the fitted p(x)
"""
import argparse
import math

import numpy as np

TRUE = [(0.4, -1.2, 0.5), (0.6, 1.1, 0.7)]            # (weight, centre, width) of the hidden density


def data(n=420, seed=11):
    rng = np.random.default_rng(seed); k = rng.random(n) < TRUE[0][0]
    return np.where(k, TRUE[0][1] + TRUE[0][2] * rng.standard_normal(n), TRUE[1][1] + TRUE[1][2] * rng.standard_normal(n))


def normal(x, m, s): return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def pdf(x, w, m, s):
    """p(x) = Σₖ wₖ N(x; mₖ, sₖ²), for an array of x."""
    return (w * normal(np.asarray(x, float)[:, None], m, s)).sum(1)


def log_likelihood(x, w, m, s): return float(np.log(pdf(x, w, m, s)).sum())


def fit(x, K, steps=200, seed=0):
    """Maximum likelihood by EM. E: how much each bump 'owns' each point. M: refit each bump to what it owns."""
    rng = np.random.default_rng(seed)
    w, m, s = np.full(K, 1 / K), rng.choice(x, K, replace=False), np.full(K, x.std())
    hist = []
    for _ in range(steps):
        r = w * normal(x[:, None], m, s); r /= r.sum(1, keepdims=True)       # E-step: responsibilities
        nk = r.sum(0); w = nk / len(x); m = (r * x[:, None]).sum(0) / nk     # M-step: weighted counts,
        s = np.sqrt((r * (x[:, None] - m) ** 2).sum(0) / nk) + 1e-6          #          weighted means and widths
        hist.append(log_likelihood(x, w, m, s))
    return w, m, s, hist


def sample(n, w, m, s, seed=1):
    """Noise in, data out: choose a bump by its weight, then stretch and shift a standard bell curve."""
    rng = np.random.default_rng(seed); k = rng.choice(len(w), n, p=w)
    return m[k] + s[k] * rng.standard_normal(n)


def check():
    x = data(); w, m, s, hist = fit(x, 2)
    drops = sum(b < a - 1e-9 for a, b in zip(hist, hist[1:]))
    xs = sample(200_000, w, m, s); edges = np.linspace(-4, 4, 41); h, _ = np.histogram(xs, edges, density=True)
    err = np.abs(h - pdf((edges[1:] + edges[:-1]) / 2, w, m, s)).max()
    print(f'EM steps that lowered the likelihood: {drops} (should be 0)')
    print(f'largest gap between the samples\' histogram and p(x): {err:.3f} ({"OK" if err < 0.02 else "CHECK"})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    if ap.parse_args().check: return check()
    x = data(); held = data(2000, seed=99)
    print('bumps   log-likelihood (train, per point)   held-out   fitted (weight, centre, width)')
    for K in (1, 2, 3, 4):
        w, m, s, _ = fit(x, K)
        fitted = '  '.join(f'({a:.2f}, {b:+.2f}, {c:.2f})' for a, b, c in sorted(zip(w, m, s), key=lambda q: q[1]))
        print(f'{K:5d}   {log_likelihood(x, w, m, s) / len(x):+10.3f}{"":24s}{log_likelihood(held, w, m, s) / len(held):+8.3f}   {fitted}')
    w, m, s, _ = fit(x, 2)
    print('\nthe hidden density:', '  '.join(f'({a:.2f}, {b:+.2f}, {c:.2f})' for a, b, c in TRUE))
    print('five new samples from the 2-bump fit:', np.round(sample(5, w, m, s), 2))


if __name__ == '__main__':
    main()
