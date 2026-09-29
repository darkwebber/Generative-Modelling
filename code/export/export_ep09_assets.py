"""
Train the diffusion models shown in episode 09 and export:
  ../video/episodes/ep09-assets.js   the title-writing model's 1000-step reverse process (the cold open), a 2-D model's
                            trajectories and x̂₀ guesses, 64 MNIST digits being born with their x̂₀ guesses, DDPM vs
                            DDIM at several step counts, a noise interpolation, curved paths for the hook
  ../video/labs/ep09-model.js    the MNIST network itself (8-bit), so the playground can generate digits live

    python export/export_ep09_assets.py       # ~40 min on a laptop CPU; trained nets are cached in data/
"""
import base64
import json
import os
import pickle
import subprocess

import numpy as np

from common import DATA, EPISODES, LABS, VIDEO  # noqa: E402,F401  (first: puts code/ on the path)
import diffusion as D



def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
def q16(a, s=4000): return b64(np.clip(np.round(np.asarray(a) * s), -32767, 32767).astype('<i2'))
def q8img(a): return b64(np.clip(np.round((np.asarray(a) + 1) / 2 * 255), 0, 255).astype(np.uint8))       # [-1, 1] → 0…255
def rnd(a, d=3): return np.round(np.asarray(a, np.float64), d).tolist()


def cached(name, make):
    path = os.path.join(DATA, f'ep09_{name}.pkl')
    if os.path.exists(path): return pickle.load(open(path, 'rb'))
    obj = make(); os.makedirs(os.path.dirname(path), exist_ok=True); pickle.dump(obj, open(path, 'wb')); return obj


def glyph(word='Diffusion'):
    out = subprocess.run(['node', 'src/tools/glyph.mjs', word], cwd=VIDEO, capture_output=True, text=True, check=True).stdout
    return np.array(json.loads(out.strip().splitlines()[-1]), np.float32)


def main(only=None):
    rng = np.random.default_rng(0)
    # ── the title model: learns the shape of the word "Diffusion"
    G = cached('glyph', glyph); SCALE = 1.0
    def title_data(n, r): return G[r.integers(0, len(G), n)] / SCALE + 0.008 * r.standard_normal((n, 2)).astype(np.float32)
    def make_title():
        net = D.EpsNet(2, (384, 384, 384), seed=1, ff=10); h = D.train(net, title_data, 40000, batch=1024, lr=1e-3, every=4000); net.use_ema(); return {'net': net, 'hist': h}
    if only in (None, 'title'): title = cached('title', make_title)['net']
    if only == 'title': return
    # ── a 2-D teaching model (8 blobs, as in episodes 5–8)
    def make_ring():
        net = D.EpsNet(2, (256, 256, 256), seed=2); h = D.train(net, D.ring, 10000, batch=512, lr=1e-3, every=2000); net.use_ema(); return {'net': net, 'hist': h}
    def make_mnist():
        net = D.EpsNet(784, (1024, 1024), seed=3, x0=True); h = D.train(net, D.mnist_fn(), 30000, batch=128, lr=1e-3, every=2000); net.use_ema(); return {'net': net, 'hist': h}
    if only in ('ring', 'mnist'):
        if only == 'ring': cached('ring', make_ring)
        else: cached('mnist', make_mnist)
        return
    z = np.random.default_rng(1).standard_normal((5000, 2)).astype(np.float32)
    keep = set(np.round(np.linspace(999, 0, 61)).astype(int).tolist())
    xT, rec = D.ddpm(title, z.copy(), np.random.default_rng(2), keep=keep)
    stages = [rec[t][0] for t in sorted(rec, reverse=True)] + [xT]
    R = cached('ring', make_ring); rnet = R['net']
    zr = np.random.default_rng(3).standard_normal((600, 2)).astype(np.float32)
    kr = set(np.round(np.linspace(999, 0, 41)).astype(int).tolist())
    xr, recr = D.ddpm(rnet, zr.copy(), np.random.default_rng(4), keep=kr)
    ring_x = [recr[t][0] for t in sorted(recr, reverse=True)] + [xr]; ring_x0 = [recr[t][1] for t in sorted(recr, reverse=True)] + [xr]
    _, paths = D.ddim(rnet, zr[:160].copy(), 50, keep_all=True, clip=1.5)
    ring_steps = {str(s): rnd(D.ddim(rnet, zr.copy(), s, clip=1.5), 3) for s in (100, 20, 5)}
    # ── MNIST
    Mn = cached('mnist', make_mnist); mnet = Mn['net']
    zm = np.random.default_rng(5).standard_normal((64, 784)).astype(np.float32)
    km = set(np.round(np.linspace(999, 0, 31)).astype(int).tolist())
    xm, recm = D.ddpm(mnet, zm.copy(), np.random.default_rng(6), keep=km)
    dig_x = [recm[t][0] for t in sorted(recm, reverse=True)] + [xm]; dig_x0 = [recm[t][1][:8] for t in sorted(recm, reverse=True)] + [xm[:8]]
    steps_cmp = {'1000': q8img(xm[:16])}
    for s in (100, 20, 5): steps_cmp[str(s)] = q8img(D.ddim(mnet, zm[:16].copy(), s, clip=1))
    a, b = zm[0], zm[1]; th = np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    slerp = np.stack([(np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th) for u in np.linspace(0, 1, 16)]).astype(np.float32)
    interp = q8img(D.ddim(mnet, slerp, 50, clip=1))
    Hm = np.array(Mn['hist']); k = len(Hm) // 150; hist = rnd(Hm[:k * 150].reshape(150, k).mean(1), 4)
    # ── episode 4's judges on 2,000 digits from the full 1000-step sampler, beside the VAE, GAN and real rows
    def make_judges():
        import evals as E
        sets, Xtr, ytr, ref, Xte, yte = E.sample_sets()
        clf = E.load_classifier(Xtr, ytr, False); f_ref = E.features(clf, ref)
        X = np.clip((D.ddpm(mnet, np.random.default_rng(7).standard_normal((E.N, 784)).astype(np.float32), np.random.default_rng(8)) + 1) / 2, 0, 1)
        f = E.features(clf, X); p, r = E.precision_recall(f_ref, f)
        return {'IS': float(E.inception_score(E.class_probs(clf, X))), 'FID': float(E.fid(f_ref, f)), 'precision': float(p), 'recall': float(r),
                'NN_train': float(E.nn_train_distance(X, Xtr)), 'C2ST': float(E.c2st(f_ref, f))}
    def asset(n): return json.loads(open(os.path.join(EPISODES, f'ep0{n}-assets.js')).read().split(' = ', 1)[1].rstrip().rstrip(';'))
    rows4 = asset(4)['rows']
    judges = {'real': rows4['real'], 'vae': rows4['vae'], 'gan': asset(5)['score'], 'diffusion': cached('judges', make_judges)}
    judges = {k: {m: round(v, 4) for m, v in row.items()} for k, row in judges.items()}
    print('judges', judges['diffusion'])
    A = {
        'T': D.T, 'abar': rnd(D.ABAR[::10], 6), 'titleScale': SCALE,
        'title': {'n': len(z), 'stages': len(stages), 'ts': sorted(rec, reverse=True) + [-1], 'q': q16(np.stack(stages))},
        'ring': {'x': [rnd(s, 3) for s in ring_x], 'x0': [rnd(s, 3) for s in ring_x0], 'ts': sorted(recr, reverse=True) + [-1], 'paths': [rnd(s, 3) for s in paths], 'steps': ring_steps},
        'digits': {'n': 64, 'ts': sorted(recm, reverse=True) + [-1], 'x': [q8img(s) for s in dig_x], 'x0': [q8img(s) for s in dig_x0]},
        'stepsCmp': steps_cmp, 'interp': interp, 'hist': hist, 'judges': judges,
    }
    with open(os.path.join(EPISODES, 'ep09-assets.js'), 'w') as fh:
        fh.write('// Generated by code/export_ep09_assets.py — real diffusion models for episode 09.\n')
        fh.write('window.EP9 = ' + json.dumps(A, separators=(',', ':')) + ';\n')
    # the network itself, 8-bit with one scale per output unit, for the playground
    def q8w(W):
        s = np.abs(W).max(0) / 127 + 1e-12; return {'shape': list(W.shape), 's': rnd(s, 8), 'q': b64(np.round(W / s).astype(np.int8))}
    M = {'W': [q8w(w) for w in mnet.W], 'U': [q8w(u) for u in mnet.U], 'b': [rnd(b, 5) for b in mnet.b], 'Wo': q8w(mnet.Wo), 'bo': rnd(mnet.bo, 5), 'ed': mnet.ed, 'x0': mnet.x0, 'abar': rnd(D.ABAR, 7)}
    with open(os.path.join(LABS, 'ep09-model.js'), 'w') as fh:
        fh.write('// Generated by code/export_ep09_assets.py — the MNIST diffusion network (8-bit weights).\n')
        fh.write('window.EP9M = ' + json.dumps(M, separators=(',', ':')) + ';\n')
    for f in (os.path.join(EPISODES, 'ep09-assets.js'), os.path.join(LABS, 'ep09-model.js')): print(os.path.basename(f), f'{os.path.getsize(f) / 1e6:.2f} MB')


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)      # optional: title | ring | mnist trains just that net (run them in parallel)
