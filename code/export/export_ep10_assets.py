"""
Train the flow-matching models shown in episode 10 and export:
  ../video/episodes/ep10-assets.js   the title-writing flows (before and after reflow, for the cold open), a 2-D model's
                            straight-line training pairs, learned velocity field, Euler samples at 1–100 steps, the reflow
                            couplings and straightened roads, 64 MNIST digits flowing out of noise with their x̂₁ guesses,
                            steps-vs-quality for flow matching, reflow and episode 9's DDIM (episode 4's FID), and a
                            label-conditioned 2-D flow for the hook into episode 11
  ../video/labs/ep10-model.js      the 2-D flows (both rounds) and the MNIST flow (8-bit), so the playground runs them live

    python export/export_ep10_assets.py            # ~1 h on a laptop CPU; trained nets are cached in data/
    python export/export_ep10_assets.py glyph      # or train one piece: glyph | ring | mnist | cond (run them in parallel)
"""
import base64
import json
import os
import pickle
import subprocess

import numpy as np

from common import DATA, EPISODES, LABS, VIDEO  # noqa: E402,F401  (first: puts code/ on the path)
import flow_matching as FM

TITLE, TITLE_PX, TITLE_SCALE = 'Flow Matching', 190, 1.4


def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
def q16(a, s=4000): return b64(np.clip(np.round(np.asarray(a) * s), -32767, 32767).astype('<i2'))
def q8img(a): return b64(np.clip(np.round((np.asarray(a) + 1) / 2 * 255), 0, 255).astype(np.uint8))       # [-1, 1] → 0…255
def rnd(a, d=3): return np.round(np.asarray(a, np.float64), d).tolist()


DRAFT = {}          # 'draft' run: stand-ins for nets not trained yet, so the scenes can be laid out early (never shipped)


def cached(name, make):
    path = os.path.join(DATA, f'ep10_{name}.pkl')
    if os.path.exists(path): return pickle.load(open(path, 'rb'))
    if name in DRAFT: return DRAFT[name]()
    obj = make(); os.makedirs(os.path.dirname(path), exist_ok=True); pickle.dump(obj, open(path, 'wb')); return obj


def glyph():
    out = subprocess.run(['node', 'src/tools/glyph.mjs', TITLE, str(TITLE_PX)], cwd=VIDEO, capture_output=True, text=True, check=True).stdout
    return np.array(json.loads(out.strip().splitlines()[-1]), np.float32) / TITLE_SCALE


def path_ratio(traj):
    """Straightness of each road: straight-line distance ÷ distance travelled (1 = a perfectly straight road)."""
    T = np.stack(traj); travelled = np.linalg.norm(T[1:] - T[:-1], axis=2).sum(0); chord = np.linalg.norm(T[-1] - T[0], axis=1)
    return float(np.median(chord / np.maximum(travelled, 1e-9)))


def blob_shares(x):
    """Share of samples near each of the 8 blobs (radius-2 ring, scaled ½), and the share that landed nowhere."""
    c = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)
    d = np.linalg.norm(x[:, None] - c[None], axis=2); k = d.argmin(1); near = d.min(1) < 0.25
    return [float(((k == j) & near).mean()) for j in range(8)], float(1 - near.mean())


# ── the models ────────────────────────────────────────────────
def make_glyph():
    G = cached('glyph_pts', glyph)
    def data(n, r): return G[r.integers(0, len(G), n)] + 0.006 * r.standard_normal((n, 2)).astype(np.float32)
    net = FM.VelocityNet(2, (384, 384, 384), seed=1, ff=10); h = FM.train(net, data, 40000, batch=1024, lr=1e-3, every=4000); net.use_ema()
    return {'net': net, 'hist': h}


def make_glyph_reflow():
    n1 = cached('glyph', make_glyph)['net']; Z, X = FM.reflow_pairs(n1, 300000, steps=100, seed=11)
    net = FM.VelocityNet(2, (384, 384, 384), seed=2, ff=10); h = FM.train(net, None, 25000, batch=1024, lr=1e-3, every=2500, pairs=(Z, X)); net.use_ema()
    return {'net': net, 'hist': h}


def make_ring():
    net = FM.VelocityNet(2, (256, 256, 256), seed=2); h = FM.train(net, FM.ring8, 12000, batch=512, lr=1e-3, every=2000); net.use_ema()
    return {'net': net, 'hist': h}


def make_ring_reflow():
    n1 = cached('ring', make_ring)['net']; Z, X = FM.reflow_pairs(n1, 100000, steps=200, seed=12)
    net = FM.VelocityNet(2, (256, 256, 256), seed=3); h = FM.train(net, None, 12000, batch=512, lr=1e-3, every=2000, pairs=(Z, X)); net.use_ema()
    return {'net': net, 'hist': h}


def make_cond():
    net = FM.VelocityNet(2, (256, 256, 256), seed=4, cond=8); h = FM.train(net, FM.ring8_labelled, 12000, batch=512, lr=1e-3, every=2000); net.use_ema()
    return {'net': net, 'hist': h}


def make_mnist():
    net = FM.VelocityNet(784, (1024, 1024), seed=3, x1=True); h = FM.train(net, FM.mnist_fn(), 30000, batch=128, lr=1e-3, every=2000); net.use_ema()
    return {'net': net, 'hist': h}


def make_mnist_reflow():
    import copy
    n1 = cached('mnist', make_mnist)['net']; Z, X = FM.reflow_pairs(n1, 60000, steps=32, seed=13, dim=784, batch=2000)
    from nn import Adam
    net = copy.deepcopy(n1); net.opt = Adam(net.params, ema=0.999)          # warm start from round 1, fresh optimiser
    h = FM.train(net, None, 20000, batch=128, lr=3e-4, every=2000, pairs=(Z, X)); net.use_ema()
    return {'net': net, 'hist': h}


def main(only=None):
    jobs = {'glyph': lambda: (cached('glyph', make_glyph), cached('glyph_reflow', make_glyph_reflow)),
            'ring': lambda: (cached('ring', make_ring), cached('ring_reflow', make_ring_reflow)),
            'cond': lambda: cached('cond', make_cond),
            'mnist': lambda: (cached('mnist', make_mnist), cached('mnist_reflow', make_mnist_reflow))}
    if only == 'draft':
        ring = lambda: {'net': cached('ring', make_ring)['net'], 'hist': [0.0]}
        DRAFT.update(glyph=ring, glyph_reflow=lambda: {'net': cached('ring_reflow', make_ring_reflow)['net'], 'hist': [0.0]},
                     mnist=lambda: {'net': FM.VelocityNet(784, (64, 64), x1=True), 'hist': list(np.linspace(200, 120, 300))},
                     mnist_reflow=lambda: {'net': FM.VelocityNet(784, (64, 64), x1=True), 'hist': [0.0]},
                     judges=lambda: {f'{a}{k}': {'FID': float(v), 'precision': 0.5, 'recall': 0.5, 'NN_train': 5.0, 'C2ST': 0.7}
                                     for a, ks in (('fm', (1, 2, 4, 8, 16, 32, 100)), ('rf', (1, 2, 4, 8, 16)), ('dd', (2, 4, 8, 16, 32, 100)))
                                     for k, v in zip(ks, np.geomspace(110, 6, len(ks)))})
    elif only: jobs[only](); return
    else:
        for f in jobs.values(): f()
    g1, g2 = cached('glyph', make_glyph)['net'], cached('glyph_reflow', make_glyph_reflow)['net']
    r1, r2 = cached('ring', make_ring), cached('ring_reflow', make_ring_reflow); R1, R2 = r1['net'], r2['net']
    cond = cached('cond', make_cond)['net']
    M1, M2 = cached('mnist', make_mnist), cached('mnist_reflow', make_mnist_reflow); m1, m2 = M1['net'], M2['net']

    # ── the cold open: 5000 points, the curved first-round flow (60 steps) and the straightened one
    zt = np.random.default_rng(1).standard_normal((5000, 2)).astype(np.float32)
    _, tfine = FM.euler(g1, zt.copy(), 60, keep_all=True)
    fast = {}
    for k in (1, 2, 3, 4, 6, 8):
        _, tr = FM.euler(g2, zt.copy(), k, keep_all=True); fast[str(k)] = q16(np.stack(tr))
    title = {'n': len(zt), 'scale': TITLE_SCALE, 'fine': q16(np.stack(tfine)), 'fineStages': len(tfine), 'fast': fast,
             'ratio1': path_ratio(tfine), 'ratio2': path_ratio(FM.euler(g2, zt[:1000].copy(), 60, keep_all=True)[1])}
    print('title road straightness', title['ratio1'], '→', title['ratio2'])

    # ── 2-D: data, random pairs, the learned field, samples by step count
    rng = np.random.default_rng(3); n = 600
    x1 = FM.ring8(n, rng); x0 = rng.standard_normal((n, 2)).astype(np.float32)
    z = np.random.default_rng(4).standard_normal((n, 2)).astype(np.float32)
    _, p1 = FM.euler(R1, z.copy(), 60, keep_all=True); _, p2 = FM.euler(R2, z.copy(), 60, keep_all=True)
    grid = np.linspace(-2.2, 2.2, 21); gx, gy = np.meshgrid(grid, grid[::-1]); gp = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    field_ts = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    field = [rnd(R1(gp, np.full(len(gp), t, np.float32)), 3) for t in field_ts]
    steps = {}
    for name, net in (('r1', R1), ('r2', R2)):
        for k in (1, 2, 4, 8):
            x, tr = FM.euler(net, z.copy(), k, keep_all=True); sh, lost = blob_shares(x)
            steps[f'{name}_{k}'] = {'stages': [rnd(s, 3) for s in tr], 'lost': round(lost, 3), 'radius': round(float(np.linalg.norm(x * 2, axis=1).mean()), 3)}
    print({k: (v['radius'], v['lost']) for k, v in steps.items()})
    # one noisy point on the roads: the fan of possible velocities at (x, t), and the average the network learns
    fan_t, fan_x = 0.5, np.array([0.75, 0.3], np.float32)
    big1 = FM.ring8(200000, np.random.default_rng(5)); big0 = np.random.default_rng(6).standard_normal(big1.shape).astype(np.float32)
    xt = (1 - fan_t) * big0 + fan_t * big1; near = np.linalg.norm(xt - fan_x, axis=1) < 0.06
    fan = {'t': fan_t, 'x': rnd(fan_x), 'u': rnd((big1 - big0)[near][:60]), 'x0': rnd(big0[near][:60]), 'x1': rnd(big1[near][:60]),
           'avg': rnd((big1 - big0)[near].mean(0)), 'net': rnd(R1(fan_x[None], np.array([fan_t], np.float32))[0])}
    print('fan', int(near.sum()), 'avg', fan['avg'], 'net', fan['net'])
    # the reflow couplings: noise points joined to where round 1 sends them (they never cross)
    zc = np.random.default_rng(7).standard_normal((160, 2)).astype(np.float32); xc = FM.euler(R1, zc.copy(), 200)
    H1 = np.array(r1['hist']); k1 = len(H1) // 120
    ring = {'x1': rnd(x1), 'x0': rnd(x0), 'z': rnd(z), 'paths1': q16(np.stack(p1)), 'paths2': q16(np.stack(p2)), 'np': len(p1), 'n': n,
            'fieldTs': field_ts, 'grid': rnd(grid, 3), 'field': field, 'steps': steps, 'fan': fan, 'couple': {'z': rnd(zc), 'x': rnd(xc)},
            'ratio1': path_ratio(p1), 'ratio2': path_ratio(p2), 'hist': rnd(H1[:k1 * 120].reshape(120, k1).mean(1), 4)}
    print('ring straightness', ring['ratio1'], '→', ring['ratio2'])

    # ── the hook: one flow, eight labels
    zq = np.random.default_rng(8).standard_normal((300, 2)).astype(np.float32)
    condp = {}
    for lab in (None, 2, 6):
        y = None if lab is None else np.full(len(zq), lab)
        _, tr = FM.euler(cond, zq.copy(), 40, keep_all=True, y=y if y is not None else np.full(len(zq), 8)); condp['none' if lab is None else str(lab)] = q16(np.stack(tr))
    hook = {'z': rnd(zq), 'paths': condp, 'np': 41}

    # ── MNIST: 64 digits flowing out of noise (32 Euler steps) with the network's x̂₁ guesses
    zm = np.random.default_rng(5).standard_normal((64, 784)).astype(np.float32)
    x, tr = FM.euler(m1, zm.copy(), 32, keep_all=True); ts = np.linspace(0, 1, 33)
    x1hat = [((s + (1 - t) * m1(s, np.float32(t)))[:8]) for s, t in zip(tr[:-1], ts[:-1])] + [tr[-1][:8]]
    digits = {'x': [q8img(s) for s in tr], 'x1': [q8img(s) for s in x1hat], 'ts': rnd(ts, 4)}
    cmp = {}
    for k in (1, 2, 4, 8, 32): cmp[f'fm{k}'] = q8img(FM.euler(m1, zm[:16].copy(), k))
    for k in (1, 2, 4): cmp[f'rf{k}'] = q8img(FM.euler(m2, zm[:16].copy(), k))
    import diffusion as D
    dnet = pickle.load(open(os.path.join(DATA, 'ep09_mnist.pkl'), 'rb'))['net']
    for k in (2, 4, 8, 32): cmp[f'dd{k}'] = q8img(D.ddim(dnet, zm[:16].copy(), k, clip=1))

    # ── episode 4's judges: FID against the number of network calls, for all three
    def make_judges():
        import evals as E
        sets, Xtr, ytr, ref, Xte, yte = E.sample_sets()
        clf = E.load_classifier(Xtr, ytr, False); f_ref = E.features(clf, ref)
        zz = np.random.default_rng(9).standard_normal((E.N, 784)).astype(np.float32); out = {}
        def score(X):
            X = np.clip((X + 1) / 2, 0, 1); f = E.features(clf, X); p, r = E.precision_recall(f_ref, f)
            return {'FID': float(E.fid(f_ref, f)), 'precision': float(p), 'recall': float(r), 'NN_train': float(E.nn_train_distance(X, Xtr)), 'C2ST': float(E.c2st(f_ref, f))}
        for k in (1, 2, 4, 8, 16, 32, 100):
            out[f'fm{k}'] = score(np.concatenate([FM.euler(m1, zz[i:i + 500].copy(), k) for i in range(0, E.N, 500)])); print('fm', k, out[f'fm{k}']['FID'])
        for k in (1, 2, 4, 8, 16):
            out[f'rf{k}'] = score(np.concatenate([FM.euler(m2, zz[i:i + 500].copy(), k) for i in range(0, E.N, 500)])); print('rf', k, out[f'rf{k}']['FID'])
        for k in (2, 4, 8, 16, 32, 100):
            out[f'dd{k}'] = score(np.concatenate([D.ddim(dnet, zz[i:i + 500].copy(), k, clip=1) for i in range(0, E.N, 500)])); print('dd', k, out[f'dd{k}']['FID'])
        return out
    J = cached('judges', make_judges)
    J = {k: {m: round(v, 4) for m, v in row.items()} for k, row in J.items()}
    def asset(n): return json.loads(open(os.path.join(EPISODES, f'ep{n:02d}-assets.js')).read().split(' = ', 1)[1].rstrip().rstrip(';'))
    J['real'] = asset(4)['rows']['real']; J['ddpm1000'] = asset(9)['judges']['diffusion']
    Hm = np.array(M1['hist']); km = len(Hm) // 150
    A = {'title': title, 'ring': ring, 'hook': hook, 'digits': digits, 'cmp': cmp, 'judges': J,
         'mhist': rnd(Hm[:km * 150].reshape(150, km).mean(1), 3)}
    with open(os.path.join(EPISODES, 'ep10-assets.js'), 'w') as fh:
        fh.write('// Generated by code/export/export_ep10_assets.py — real flow-matching models for episode 10.\n')
        fh.write('window.EP10 = ' + json.dumps(A, separators=(',', ':')) + ';\n')

    # the networks themselves, for the playground: the 2-D flows in float32, the MNIST flow in 8 bits
    def f32(W): return {'shape': list(W.shape), 'f': b64(np.asarray(W, '<f4'))}
    def net2d(net): return {'W': [f32(w) for w in net.W], 'U': [f32(u) for u in net.U], 'b': [f32(b) for b in net.b], 'Wo': f32(net.Wo), 'bo': f32(net.bo), 'ed': net.ed, 'cond': net.cond}
    def q8w(W):
        s = np.abs(W).max(0) / 127 + 1e-12; return {'shape': list(W.shape), 's': rnd(s, 8), 'q': b64(np.round(W / s).astype(np.int8))}
    Mm = {'W': [q8w(w) for w in m1.W], 'U': [q8w(u) for u in m1.U], 'b': [rnd(b, 5) for b in m1.b], 'Wo': q8w(m1.Wo), 'bo': rnd(m1.bo, 5), 'ed': m1.ed}
    L = {'ring1': net2d(R1), 'ring2': net2d(R2), 'cond': net2d(cond), 'mnist': Mm}
    with open(os.path.join(LABS, 'ep10-model.js'), 'w') as fh:
        fh.write('// Generated by code/export/export_ep10_assets.py — the flow-matching networks for the episode 10 playground.\n')
        fh.write('window.EP10M = ' + json.dumps(L, separators=(',', ':')) + ';\n')
    for f in (os.path.join(EPISODES, 'ep10-assets.js'), os.path.join(LABS, 'ep10-model.js')): print(os.path.basename(f), f'{os.path.getsize(f) / 1e6:.2f} MB')


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
