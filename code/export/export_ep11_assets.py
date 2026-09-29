"""
Train the guidance models shown in episode 11 and export:
  ../video/episodes/ep11-assets.js   the title spelled letter by letter by one label-conditioned flow; the eight blobs
                            asked for one blob (plain conditional, classifier guidance, classifier-free guidance, negative
                            requests) with the arrows behind each; two overlapping classes under the guidance dial w; MNIST
                            digits on request, the same noise asked for every digit, the dial on real digits, and episode 4's
                            judges (and a digit reader) as w grows
  ../video/labs/ep11-model.js      the 2-D nets (float32) and the conditional MNIST net (8-bit) for the playground

    python export/export_ep11_assets.py            # ~1 h on a laptop CPU; trained nets are cached in data/
    python export/export_ep11_assets.py letters    # or train one piece: letters | toys | mnist (run them in parallel)
"""
import base64
import json
import os
import pickle
import subprocess

import numpy as np

from common import DATA, EPISODES, LABS, VIDEO  # noqa: E402,F401  (first: puts code/ on the path)
import flow_matching as FM
import guidance as G

TITLE, TITLE_PX, TITLE_SCALE = 'Guidance', 190, 1.4


def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
def q16(a, s=4000): return b64(np.clip(np.round(np.asarray(a) * s), -32767, 32767).astype('<i2'))
def q8img(a): return b64(np.clip(np.round((np.asarray(a) + 1) / 2 * 255), 0, 255).astype(np.uint8))       # [-1, 1] → 0…255
def rnd(a, d=3): return np.round(np.asarray(a, np.float64), d).tolist()


def cached(name, make):
    path = os.path.join(DATA, f'ep11_{name}.pkl')
    if os.path.exists(path): return pickle.load(open(path, 'rb'))
    obj = make(); os.makedirs(os.path.dirname(path), exist_ok=True); pickle.dump(obj, open(path, 'wb')); return obj


def letters():
    out = subprocess.run(['node', 'src/tools/glyph.mjs', TITLE, str(TITLE_PX), 'letters'], cwd=VIDEO, capture_output=True, text=True, check=True).stdout
    P = np.array(json.loads(out.strip().splitlines()[-1]), np.float32)
    return P[:, :2] / TITLE_SCALE, P[:, 2].astype(int)


# ── the models ────────────────────────────────────────────────
def make_letters():
    """One flow, eight labels: label k draws the k-th letter of the title, where it sits in the word."""
    X, Y = cached('letters_pts', letters)
    def data(n, r): i = r.integers(0, len(X), n); return X[i] + 0.006 * r.standard_normal((n, 2)).astype(np.float32), Y[i]
    net = FM.VelocityNet(2, (384, 384, 384), seed=5, ff=10, cond=8); h = FM.train(net, data, 30000, batch=1024, lr=1e-3, every=3000); net.use_ema()
    return {'net': net, 'hist': h}


def make_ring_clf():
    clf = G.NoisyClassifier(2, 8, (128, 128), seed=2); h = G.train_classifier(clf, FM.ring8_labelled, 6000, batch=512, every=1000); clf.use_ema()
    return {'clf': clf, 'hist': h}


def make_clouds():
    net = FM.VelocityNet(2, (256, 256, 256), seed=6, cond=2); h = FM.train(net, G.two_clouds, 12000, batch=512, lr=1e-3, every=2000); net.use_ema()
    return {'net': net, 'hist': h}


def make_mnist():
    """Conditional MNIST flow, warm-started from episode 10's unconditional net: the ten label inputs (and the
    'no label' slot) start at zero weight, so it begins as episode 10's model and learns to listen."""
    src = pickle.load(open(os.path.join(DATA, 'ep10_mnist.pkl'), 'rb'))['net']
    net = FM.VelocityNet(784, (1024, 1024), seed=3, x1=True, cond=10)
    net.W[0][...] = 0; net.W[0][:784] = src.W[0]
    for a, b in zip(net.params[1:], src.params[1:]): a[...] = b
    h = FM.train(net, FM.mnist_fn(labels=True), 20000, batch=128, lr=3e-4, every=2000); net.use_ema()
    return {'net': net, 'hist': h}


def make_mnist_clf():
    clf = G.NoisyClassifier(784, 10, (512, 256), seed=4); h = G.train_classifier(clf, FM.mnist_fn(labels=True), 8000, batch=256, every=1000); clf.use_ema()
    return {'clf': clf, 'hist': h}


def cond10():
    """Episode 10's label-conditioned flow on the eight blobs (trained with the label hidden 10% of the time)."""
    return pickle.load(open(os.path.join(DATA, 'ep10_cond.pkl'), 'rb'))['net']


BLOBC = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)
def blob_of(x): d = np.linalg.norm(x[:, None] - BLOBC[None], axis=2); return np.where(d.min(1) < 0.3, d.argmin(1), -1)


def export():
    rng = np.random.default_rng(0)
    L8, RC, CL = cached('letters', make_letters)['net'], cached('ring_clf', make_ring_clf)['clf'], cached('clouds', make_clouds)['net']
    MC, MN = cached('mnist_clf', make_mnist_clf)['clf'], cached('mnist', make_mnist)
    MM, RN = MN['net'], cond10()

    # ── the cold open: one noise cloud, drained letter by letter ("y = G", "y = u", …), 40 Euler steps, w = 1;
    #    and the same noise at w = 2.5 for 'the price of listening': each letter shrinks to its most certain strokes
    per = 800; zt = np.random.default_rng(1).standard_normal((8 * per, 2)).astype(np.float32); tl, hard = [], []
    Xl, Yl = cached('letters_pts', letters); cover = []
    for k in range(8):
        zk = zt[k * per:(k + 1) * per]; _, tr = G.sample(L8, zk.copy(), 40, np.full(per, k), w=1.0, keep_all=True); tl.append(np.stack(tr))
        xh = G.sample(L8, zk.copy(), 40, np.full(per, k), w=2.5); hard.append(xh); R = Xl[Yl == k]
        cover.append([float((np.linalg.norm(R[:, None] - x[None], axis=2).min(1) < 0.03).mean()) for x in (tr[-1], xh)])
    print('letter coverage (w = 1, w = 2.5)', np.round(cover, 2).tolist())
    title = {'n': per, 'k': 8, 'word': TITLE, 'scale': TITLE_SCALE, 'paths': [q16(p) for p in tl], 'stages': 41,
             'hard': q16(np.concatenate(hard)), 'cover': rnd(np.mean(cover, 0), 3)}

    # ── the eight blobs: ask for blob 3 (label 2) three ways, and the arrows behind each
    zq = np.random.default_rng(8).standard_normal((300, 2)).astype(np.float32); Y3 = np.full(300, 2)
    flows = {}
    for key, kw in {'none': dict(y=Y3, w=0.0), 'b3': dict(y=Y3, w=1.0), 'b7': dict(y=np.full(300, 6), w=1.0),
                    'cg0': dict(y=Y3, w=0.0, clf=RC), 'cg1': dict(y=Y3, w=1.0, clf=RC), 'cg4': dict(y=Y3, w=4.0, clf=RC),
                    'cfg3': dict(y=Y3, w=3.0), 'negn': dict(y=Y3, w=1.25), 'neg': dict(y=Y3, w=1.25, neg=np.full(300, 1)),
                    'neg2': dict(y=Y3, w=2.0, neg=np.full(300, 1))}.items():
        _, tr = G.sample(RN, zq.copy(), 40, kw.pop('y'), keep_all=True, **kw); flows[key] = q16(np.stack(tr))
        x = tr[-1]; b = blob_of(x); print(f'blobs {key:6s} on blob 3: {(b == 2).mean():.0%}   mean {x.mean(0).round(2)}')
    grid = np.linspace(-1.5, 1.5, 15); gx, gy = np.meshgrid(grid, grid[::-1]); gp = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    hgrid = np.linspace(-1.5, 1.5, 41); hx, hy = np.meshgrid(hgrid, hgrid[::-1]); hp = np.stack([hx.ravel(), hy.ravel()], 1).astype(np.float32)
    fts = [0.3, 0.6, 0.85]; fields = {}
    for t in fts:
        T = np.float32(t); v0 = RN(gp, T, None); vy = RN(gp, T, np.full(len(gp), 2))
        cg = (1 - t) / t * RC.grad_log_p(gp, T, np.full(len(gp), 2))
        fields[str(t)] = {'v0': rnd(v0), 'vy': rnd(vy), 'cg': rnd(cg), 'p': rnd(RC.probs(hp, T)[:, 2], 3)}
    probe_t, probe_x = 0.55, np.array([[0.7, 0.7]], np.float32)
    probe = {'t': probe_t, 'x': rnd(probe_x[0]), 'v0': rnd(RN(probe_x, np.float32(probe_t), None)[0]), 'vy': rnd(RN(probe_x, np.float32(probe_t), np.array([2]))[0]),
             'cg': rnd(((1 - probe_t) / probe_t * RC.grad_log_p(probe_x, np.float32(probe_t), np.array([2])))[0])}
    blobs = {'z': rnd(zq), 'flows': flows, 'stages': 41, 'grid': rnd(grid), 'hgrid': rnd(hgrid), 'fts': fts, 'fields': fields, 'probe': probe}

    # ── two overlapping clouds: class A under the dial
    zc = np.random.default_rng(9).standard_normal((900, 2)).astype(np.float32); ws = [0, 1, 2, 4, 8]; clouds = {'ws': ws, 'paths': {}, 'stats': {}}
    for w in ws:
        _, tr = G.sample(CL, zc.copy(), 32, np.zeros(900, int), w=float(w), keep_all=True); x = tr[-1]
        clouds['paths'][str(w)] = q16(np.stack(tr)); clouds['stats'][str(w)] = {'mean': rnd(x.mean(0)), 'std': rnd(x.std(0)), 'pA': round(float((G.two_clouds_density(x).argmax(1) == 0).mean()), 3)}
        print('clouds w', w, clouds['stats'][str(w)])
    clouds.update(stages=33, n=900, sep=0.55, std=0.6)

    # ── MNIST on request
    import evals as E
    sets, Xtr, ytr, ref, Xte, yte = E.sample_sets(); ev = E.load_classifier(Xtr, ytr, False); f_ref = E.features(ev, ref)
    read = lambda X: E.class_probs(ev, np.clip((X + 1) / 2, 0, 1))
    zg = np.random.default_rng(5).standard_normal((8, 784)).astype(np.float32)
    grid10 = [G.sample(MM, zg.copy(), 32, np.full(8, d), w=1.0) for d in range(10)]
    gread = [read(x).argmax(1).tolist() for x in grid10]
    print('grid read', gread)
    sweep = {str(w): q8img(G.sample(MM, zg.copy(), 32, np.full(8, 7), w=float(w))) for w in (0, 1, 2, 4, 8)}
    cgm = {str(w): q8img(G.sample(MM, zg.copy(), 32, np.full(8, 7), w=float(w), clf=MC)) for w in (1, 4)}
    negm = {}
    for w in (1.25, 1.5, 2.0, 3.0):
        for key, (d, avoid) in {'7': (7, None), '7n1': (7, 1), '4': (4, None), '4n9': (4, 9)}.items():
            x = G.sample(MM, zg.copy(), 32, np.full(8, d), w=w, neg=None if avoid is None else np.full(8, avoid)); negm[f'{key}@{w}'] = q8img(x)
            print('neg', key, w, 'read', read(x).argmax(1).tolist())
    print('cg read', {w: read(G.sample(MM, zg.copy(), 32, np.full(8, 7), w=float(w), clf=MC)).argmax(1).tolist() for w in (1, 4)})

    def make_judges():
        zz = np.random.default_rng(9).standard_normal((E.N, 784)).astype(np.float32); yy = np.arange(E.N) % 10; out = {}
        for w in (0, 0.5, 1, 1.5, 2, 3, 4, 6, 8):
            X = np.concatenate([G.sample(MM, zz[i:i + 500].copy(), 32, yy[i:i + 500], w=float(w)) for i in range(0, E.N, 500)])
            Xc = np.clip((X + 1) / 2, 0, 1); f = E.features(ev, Xc); p, r = E.precision_recall(f_ref, f); pr = E.class_probs(ev, Xc)
            out[str(w)] = {'acc': float((pr.argmax(1) == yy).mean()), 'FID': float(E.fid(f_ref, f)), 'precision': float(p), 'recall': float(r),
                           'conf': float(pr[np.arange(E.N), yy].mean()), 'perClass': [float((pr.argmax(1)[yy == d] == d).mean()) for d in range(10)]}
            print('judges w', w, {k: round(v, 3) for k, v in out[str(w)].items() if k != 'perClass'})
        return out
    J = cached('judges', make_judges)
    Hm = np.array(MN['hist']); km = len(Hm) // 150
    mnist = {'grid': [q8img(x) for x in grid10], 'gridRead': gread, 'sweep': sweep, 'cg': cgm, 'neg': negm, 'judges': J,
             'hist': rnd(Hm[:km * 150].reshape(150, km).mean(1), 3)}

    A = {'title': title, 'blobs': blobs, 'clouds': clouds, 'mnist': mnist}
    with open(os.path.join(EPISODES, 'ep11-assets.js'), 'w') as fh:
        fh.write('// Generated by code/export/export_ep11_assets.py — real guidance models for episode 11.\n')
        fh.write('window.EP11 = ' + json.dumps(A, separators=(',', ':')) + ';\n')

    # the networks, for the playground: the 2-D nets in float32, the MNIST net in 8 bits
    def f32(W): return {'shape': list(W.shape), 'f': b64(np.asarray(W, '<f4'))}
    def net2d(net): return {'W': [f32(w) for w in net.W], 'U': [f32(u) for u in net.U], 'b': [f32(b) for b in net.b], 'Wo': f32(net.Wo), 'bo': f32(net.bo),
                            'ed': net.ed, 'cond': net.cond, 'ff': 0 if net.freqs is None else len(net.freqs)}
    def q8w(W):
        s = np.abs(W).max(0) / 127 + 1e-12; return {'shape': list(W.shape), 's': rnd(s, 8), 'q': b64(np.round(W / s).astype(np.int8))}
    Mm = {'W': [q8w(w) for w in MM.W], 'U': [q8w(u) for u in MM.U], 'b': [rnd(b, 5) for b in MM.b], 'Wo': q8w(MM.Wo), 'bo': rnd(MM.bo, 5), 'ed': MM.ed, 'cond': MM.cond}
    Lb = {'blobs': net2d(RN), 'clouds': net2d(CL), 'letters': net2d(L8), 'mnist': Mm}
    with open(os.path.join(LABS, 'ep11-model.js'), 'w') as fh:
        fh.write('// Generated by code/export/export_ep11_assets.py — the guidance networks for the episode 11 playground.\n')
        fh.write('window.EP11M = ' + json.dumps(Lb, separators=(',', ':')) + ';\n')
    for f in (os.path.join(EPISODES, 'ep11-assets.js'), os.path.join(LABS, 'ep11-model.js')): print(os.path.basename(f), f'{os.path.getsize(f) / 1e6:.2f} MB')


JOBS = {'letters': lambda: cached('letters', make_letters),
        'toys': lambda: (cached('ring_clf', make_ring_clf), cached('clouds', make_clouds)),
        'mnist': lambda: (cached('mnist_clf', make_mnist_clf), cached('mnist', make_mnist))}


def main(only=None):
    if only: JOBS[only](); return
    for f in JOBS.values(): f()
    export()


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
