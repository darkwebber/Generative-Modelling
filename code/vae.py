"""
Episode 03 — Autoencoders and VAEs, from scratch in numpy (backprop written by hand).

    encoder    h = relu(x W1 + b1)                      784 -> 256
               AE : z = h Wz + b                        256 -> 2        (one point)
               VAE: mu = h Wmu + b,  logvar = h Wlv + b  (a small Gaussian cloud)
                    z = mu + exp(logvar / 2) * eps,  eps ~ N(0, I)      (reparameterisation)
    decoder    g = relu(z W3 + b3)                      2 -> 256
               x_hat = sigmoid(g W4 + b4)               256 -> 784

    loss       reconstruction  = sum_pixels BCE(x, x_hat)          "rebuild the image"
               VAE adds  beta * KL( N(mu, sigma^2) || N(0, I) )
                         = beta * 1/2 * sum( mu^2 + sigma^2 - log sigma^2 - 1 )   "stay near the origin, stay fuzzy"

Requires:  pip install numpy          (matplotlib optional, for --plot)

Try:
    python vae.py --model ae                  # plain autoencoder: great reconstructions, holey latent space
    python vae.py --model vae                 # VAE: smooth, packed latent space you can sample from
    python vae.py --model vae --beta 4        # more organised, blurrier
    python vae.py --model vae --plot          # latent map + a grid of decoded z's + fresh samples
    python vae.py --check                     # verify every hand-written gradient numerically
"""
import argparse
import gzip
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


# ── data ──────────────────────────────────────────────────────
def load_mnist(split='train', folder=os.path.join(HERE, 'data', 'mnist')):
    """MNIST as float32 pixels in [0, 1], shape (N, 784), plus integer labels."""
    os.makedirs(folder, exist_ok=True)
    pre = 'train' if split == 'train' else 't10k'
    out = []
    for kind, offset in (('images-idx3', 16), ('labels-idx1', 8)):
        name = f'{pre}-{kind}-ubyte.gz'
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            print(f'downloading {name} …')
            urllib.request.urlretrieve(MNIST_URL + name, path)
        with gzip.open(path, 'rb') as f:
            out.append(np.frombuffer(f.read(), np.uint8, offset=offset))
    return out[0].reshape(-1, 784).astype(np.float32) / 255.0, out[1].astype(np.int64)


# ── model ─────────────────────────────────────────────────────
def init(model, hidden=256, latent=2, seed=0):
    rng = np.random.default_rng(seed)
    he = lambda n_in, n_out: (rng.normal(0, 1, (n_in, n_out)) * np.sqrt(2 / n_in)).astype(np.float32)
    P = {'W1': he(784, hidden), 'b1': np.zeros(hidden, np.float32),
         'W3': he(latent, hidden), 'b3': np.zeros(hidden, np.float32),
         'W4': he(hidden, 784) * 0.1, 'b4': np.zeros(784, np.float32)}
    if model == 'ae':
        P.update(Wz=he(hidden, latent) * 0.1, bz=np.zeros(latent, np.float32))
    else:
        P.update(Wmu=he(hidden, latent) * 0.1, bmu=np.zeros(latent, np.float32),
                 Wlv=he(hidden, latent) * 0.1, blv=np.zeros(latent, np.float32))
    return P


def encode(P, x):
    """AE: the code z.   VAE: the cloud (mu, logvar)."""
    h = np.maximum(0, x @ P['W1'] + P['b1'])
    if 'Wz' in P:
        return h @ P['Wz'] + P['bz'], None
    return h @ P['Wmu'] + P['bmu'], h @ P['Wlv'] + P['blv']


def decode(P, z):
    g = np.maximum(0, z @ P['W3'] + P['b3'])
    return 1 / (1 + np.exp(-(g @ P['W4'] + P['b4'])))


def loss_and_grads(P, x, beta=1.0, eps=None, rng=np.random):
    """Forward pass, loss (per image, in nats), and every gradient — by hand."""
    B = len(x)
    a1 = x @ P['W1'] + P['b1']
    h = np.maximum(0, a1)
    vae = 'Wmu' in P
    if vae:
        mu = h @ P['Wmu'] + P['bmu']
        lv = np.clip(h @ P['Wlv'] + P['blv'], -10, 10)
        std = np.exp(0.5 * lv)
        if eps is None:
            eps = rng.standard_normal(mu.shape).astype(np.float32)
        z = mu + std * eps                                  # the reparameterisation trick
    else:
        z = h @ P['Wz'] + P['bz']
    a3 = z @ P['W3'] + P['b3']
    g = np.maximum(0, a3)
    logits = g @ P['W4'] + P['b4']
    # binary cross-entropy, written stably:  softplus(l) - x * l
    recon = (np.logaddexp(0, logits) - x * logits).sum(1).mean()
    kl = 0.5 * (mu ** 2 + std ** 2 - lv - 1).sum(1).mean() if vae else 0.0
    loss = recon + beta * kl

    G = {}
    d_logits = (1 / (1 + np.exp(-logits)) - x) / B         # sigmoid + BCE  ->  (x_hat - x)
    G['W4'], G['b4'] = g.T @ d_logits, d_logits.sum(0)
    d_a3 = (d_logits @ P['W4'].T) * (a3 > 0)
    G['W3'], G['b3'] = z.T @ d_a3, d_a3.sum(0)
    d_z = d_a3 @ P['W3'].T
    if vae:
        d_mu = d_z + beta * mu / B                          # KL pulls mu toward 0
        d_lv = d_z * eps * 0.5 * std + beta * 0.5 * (std ** 2 - 1) / B   # and sigma toward 1
        G['Wmu'], G['bmu'] = h.T @ d_mu, d_mu.sum(0)
        G['Wlv'], G['blv'] = h.T @ d_lv, d_lv.sum(0)
        d_h = d_mu @ P['Wmu'].T + d_lv @ P['Wlv'].T
    else:
        G['Wz'], G['bz'] = h.T @ d_z, d_z.sum(0)
        d_h = d_z @ P['Wz'].T
    d_a1 = d_h * (a1 > 0)
    G['W1'], G['b1'] = x.T @ d_a1, d_a1.sum(0)
    return loss, recon, kl, G


def train(model='vae', epochs=15, beta=1.0, lr=1e-3, batch=128, seed=0, log=print):
    X, _ = load_mnist('train')
    Xte, _ = load_mnist('test')
    P = init(model, seed=seed)
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    step, history = 0, []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(X))
        tot = rec = klt = 0.0
        for i in range(0, len(X), batch):
            xb = X[order[i:i + batch]]
            loss, r, k, G = loss_and_grads(P, xb, beta, rng=rng)
            step += 1
            for n in P:                                     # Adam
                m[n] = 0.9 * m[n] + 0.1 * G[n]
                s[n] = 0.999 * s[n] + 0.001 * G[n] ** 2
                P[n] -= lr * (m[n] / (1 - 0.9 ** step)) / (np.sqrt(s[n] / (1 - 0.999 ** step)) + 1e-8)
            tot += loss * len(xb); rec += r * len(xb); klt += float(k) * len(xb)
        _, te_r, te_k, _ = loss_and_grads(P, Xte[:2000], beta, rng=rng)
        history.append({'epoch': ep, 'recon': float(rec / len(X)), 'kl': float(klt / len(X)), 'test_recon': float(te_r), 'test_kl': float(te_k)})
        log(f'epoch {ep:2d}   reconstruction {rec / len(X):7.2f}   KL {klt / len(X):6.2f}   '
            f'(test: {te_r:6.2f} + {float(te_k):5.2f})   nats per image')
    return P, history


def grad_check():
    rng = np.random.default_rng(1)
    x = (rng.random((5, 784)) > 0.7).astype(np.float32)
    for model in ('ae', 'vae'):
        P = {k: v.astype(np.float64) for k, v in init(model, hidden=16, seed=2).items()}
        eps = rng.standard_normal((5, 2))
        _, _, _, G = loss_and_grads(P, x, 1.0, eps=eps)
        worst = 0
        for n in P:
            for _ in range(3):
                idx = tuple(rng.integers(0, s) for s in P[n].shape)
                old = P[n][idx]
                P[n][idx] = old + 1e-5; lp = loss_and_grads(P, x, 1.0, eps=eps)[0]
                P[n][idx] = old - 1e-5; lm = loss_and_grads(P, x, 1.0, eps=eps)[0]
                P[n][idx] = old
                num = (lp - lm) / 2e-5
                worst = max(worst, abs(num - G[n][idx]) / max(1e-8, abs(num) + abs(G[n][idx])))
        print(f'{model}: worst relative gradient error {worst:.2e}  ({"OK" if worst < 1e-3 else "CHECK"} — ReLU kinks can add tiny errors)')


def plot(P, model):
    import matplotlib.pyplot as plt
    from statistics import NormalDist
    Xte, yte = load_mnist('test')
    z, _ = encode(P, Xte[:5000])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    sc = ax[0].scatter(z[:, 0], z[:, 1], c=yte[:5000], cmap='tab10', s=3)
    fig.colorbar(sc, ax=ax[0]); ax[0].set_title(f'{model.upper()} latent space (colour = true digit)')
    n = 15
    grid = [NormalDist().inv_cdf(p) for p in np.linspace(0.05, 0.95, n)] if model == 'vae' else np.linspace(z.min(0)[0], z.max(0)[0], n)
    gy = grid if model == 'vae' else np.linspace(z.min(0)[1], z.max(0)[1], n)
    canvas = np.zeros((28 * n, 28 * n))
    for i, yv in enumerate(gy[::-1]):
        for j, xv in enumerate(grid):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = decode(P, np.array([[xv, yv]], np.float32)).reshape(28, 28)
    ax[1].imshow(canvas, cmap='gray'); ax[1].set_title('decoding a grid of z values'); ax[1].axis('off')
    samples = decode(P, np.random.standard_normal((64, 2)).astype(np.float32)).reshape(8, 8, 28, 28)
    ax[2].imshow(samples.transpose(0, 2, 1, 3).reshape(224, 224), cmap='gray')
    ax[2].set_title('decode(z),  z ~ N(0, I)'); ax[2].axis('off')
    plt.tight_layout(); plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['ae', 'vae'], default='vae')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--beta', type=float, default=1.0, help='weight on the KL term (VAE only)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', help='save the trained weights to this .npz file')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--check', action='store_true', help='numerically verify the gradients and exit')
    args = ap.parse_args()
    if args.check:
        return grad_check()
    P, _ = train(args.model, args.epochs, args.beta, args.lr, seed=args.seed)
    if args.save:
        np.savez(args.save, **P)
        print('saved', args.save)
    if args.plot:
        plot(P, args.model)


if __name__ == '__main__':
    main()
