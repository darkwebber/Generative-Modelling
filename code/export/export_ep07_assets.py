"""
Train the energy-based models shown in episode 07 and export everything to ../video/episodes/ep07-assets.js:
the sculpted landscape at several points in training (with its data and fantasies), Langevin and
gradient-descent marble paths, a long mixing chain, each valley's share of probability, the exact
2-D likelihood, the slope field, and the 1-D sculpting and Metropolis demos.

    python export/export_ep07_assets.py        # a few minutes; the trained model is cached in data/
"""
import base64
import json
import os
import pickle
import re

import numpy as np

from common import DATA, EPISODES, LABS, VIDEO  # noqa: E402,F401  (first: puts code/ on the path)
import ebm

OUT = os.path.join(EPISODES, 'ep07-assets.js')
R = ebm.R
SNAPS = (1, 40, 120, 300, 800, 2000, 4000)
CFG = dict(steps=4000, eta=3e-3, k=60, alpha=0.01, hidden=96, seed=0)
EMAX = 48.0   # heights shown: log(1 + (E − floor)/2) / log(1 + EMAX/2), each map from its own floor


def rnd(a, d=2):
    return np.round(np.asarray(a, dtype=np.float64), d).tolist()


def u8(a):
    return base64.b64encode(np.clip(np.round(a * 255), 0, 255).astype(np.uint8).tobytes()).decode()


def energy_grid(E, n=96):
    g = np.linspace(-R, R, n, dtype=np.float32); X, Y = np.meshgrid(g, g[::-1])
    return E(np.stack([X.ravel(), Y.ravel()], 1)).reshape(n, n).astype(np.float64)


def emap(e):
    lo = e.min()
    return {'n': e.shape[0], 'lo': float(lo), 'u8': u8(np.log1p((e - lo) / 2) / np.log1p(EMAX / 2))}


def with_params(E, snap):
    F = ebm.Energy(CFG['hidden']); F.W, F.b = snap['W'], snap['b']; return F


def trained():
    path = os.path.join(DATA, 'ep07_ring.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    E, hist, nll, snaps = ebm.train('ring', snapshots=SNAPS, **CFG)
    res = {'E': E, 'hist': hist, 'nll': nll, 'snaps': snaps}
    os.makedirs(os.path.dirname(path), exist_ok=True); pickle.dump(res, open(path, 'wb'))
    return res


def sector(x):  # which of the 8 valleys a point is nearest to
    return (np.round(np.arctan2(x[..., 1], x[..., 0]) / (np.pi / 4)).astype(int)) % 8


def main():
    M = trained(); E = M['E']; eta = CFG['eta']
    e = energy_grid(E); floor = e.min()
    # ── the landscape during training (same colour scale for every snapshot)
    train = []
    for s in SNAPS:
        sn = M['snaps'][s]; F = with_params(E, sn)
        train.append({'step': s, 'E': emap(energy_grid(F, 72)), 'fake': rnd(sn['fake'][:160]), 'data': rnd(sn['data'][:160])})
    h = np.array(M['hist']); k = max(1, len(h) // 200)
    hist = rnd(h[:len(h) // k * k].reshape(-1, k, 2).mean(1), 3)
    # ── exact density (2-D only!) and each valley's share of it
    n = 400; g = np.linspace(-4.5, 4.5, n, dtype=np.float32); X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], 1); lp = -E(pts).astype(np.float64); lp -= lp.max(); p = np.exp(lp); p /= p.sum()
    share = [float(p[sector(pts) == j].sum()) for j in range(8)]
    dens = np.exp(-(e - floor)); dens = {'n': e.shape[0], 'u8': u8(np.sqrt(dens / dens.max()))}
    # ── marbles: Langevin vs plain gradient descent, from the same uniform noise
    rng = np.random.default_rng(5); x0 = rng.uniform(-R, R, (420, 2)).astype(np.float32)
    _, lang = ebm.langevin(E, x0, 240, eta, np.random.default_rng(6), keep=4)
    _, gd = ebm.langevin(E, x0, 240, eta, np.random.default_rng(6), noise=False, keep=4)
    # ── a long single chain: how often does it cross a ridge?
    c, chain = ebm.langevin(E, np.array([[2.0, 0.0]], np.float32), 30000, eta, np.random.default_rng(7), keep=30)
    ch = np.concatenate(chain); sec = sector(ch); far = np.linalg.norm(ch, axis=1) > 1.4
    cross = int(((np.diff(sec) != 0) & far[1:]).sum())
    visits = [int((sec[far] == j).sum()) for j in range(8)]
    # ── the slope field (for the hook) and a few probe energies
    q = np.linspace(-3.2, 3.2, 21, dtype=np.float32); QX, QY = np.meshgrid(q, q[::-1]); qp = np.stack([QX.ravel(), QY.ravel()], 1)
    score = -E.grad_x(qp)
    probes = {nm: float(E(np.array([pt], np.float32))[0] - floor) for nm, pt in
              (('blob', [2.0, 0.0]), ('diag', [1.414, 1.414]), ('between', [1.85, 0.77]), ('centre', [0.0, 0.0]), ('corner', [3.0, 3.0]))}
    # ── 1-D: sculpting from data, and a Metropolis chain on the true landscape
    g1, d1, frames = ebm.sculpt_1d()
    sub = slice(None, None, 4)
    sculpt = {'g': rnd(g1[sub], 3), 'data': rnd(d1, 3), 'E': [rnd(f['E'][sub], 3) for f in frames], 'fant': [rnd(f['fant'][:30], 3) for f in frames]}
    mr = np.random.default_rng(11); x = -2.1; met = []
    for _ in range(4000):
        prop = float(np.clip(x + 0.7 * mr.standard_normal(), -3.25, 3.25)); dE = float(ebm.E_true(prop) - ebm.E_true(x))
        u = float(mr.random()); acc = dE <= 0 or u < np.exp(-dE)
        met.append([round(x, 3), round(prop, 3), round(dE, 3), round(u, 3), int(acc)]); x = prop if acc else x
    # ── the flow from episode 6 on the same 8 blobs (for the side-by-side)
    ep6 = open(os.path.join(EPISODES, 'ep06-assets.js')).read()
    ep6 = json.loads(re.search(r'window\.EP6 = (\{.*\});', ep6, re.S).group(1))
    assets = {
        'R': R, 'eta': eta, 'k': CFG['k'], 'steps': CFG['steps'], 'alpha': CFG['alpha'], 'emax': EMAX,
        'E': emap(e), 'Eraw': {'n': 49, 'v': rnd(energy_grid(E, 49) - floor, 1)}, 'dens': dens, 'train': train, 'hist': hist,
        'nll': M['nll'], 'nllFlow': ep6['nllRing'], 'gauss': ep6['gaussRing'], 'perfect': float(np.log(8) + np.log(2 * np.pi * np.e * 0.12 ** 2)),
        'share': rnd(share, 4), 'lang': [rnd(s) for s in lang], 'gd': [rnd(s) for s in gd],
        'chain': rnd(ch), 'cross': cross, 'visits': visits, 'chainSteps': 30000,
        'score': {'n': 21, 'p': rnd(qp), 'v': rnd(score, 3)}, 'probes': probes,
        'sculpt': sculpt, 'metro': met, 'flowRing': ep6['densRing'],
    }
    with open(OUT, 'w') as fh:
        fh.write('// Generated by code/export_ep07_assets.py — a real trained energy-based model for episode 07.\n')
        fh.write('window.EP7 = ' + json.dumps(assets, separators=(',', ':')) + ';\n')
    print(f'wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)')
    print(f"NLL {M['nll']:.3f}  flow {ep6['nllRing']:.3f}  perfect {assets['perfect']:.3f}   shares {np.round(share, 3)}   crossings {cross}  visits {visits}")
    print('probes', {k_: round(v, 2) for k_, v in probes.items()})


if __name__ == '__main__':
    main()
