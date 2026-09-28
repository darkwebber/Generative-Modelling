"""
Episode 02 — Replace the count table with a neural network (numpy only, backprop by hand).

    e = [C[x_{t-k}], ..., C[x_{t-1}]]        embeddings of the last k letters, concatenated
    h = tanh(e W1 + b1)                        hidden layer
    z = h W2 + b2                              27 logits
    p = softmax(z)                             p_theta(next | context)
    loss = -log p[true next letter]            cross-entropy = maximum likelihood

Requires:  pip install numpy          (matplotlib optional, for --plot)

Try:
    python ar_neural.py                        # train, compare with counting, sample 20 names
    python ar_neural.py --k 5 --hidden 64       # longer memory, bigger network
    python ar_neural.py --tau 0.5              # colder sampling
    python ar_neural.py --emb 2 --plot         # 2-D letter embeddings: do the vowels cluster?
    python ar_neural.py --check                # verify the hand-written gradients numerically
    python ar_neural.py --big                  # ~32k names, if you are online
"""
import argparse
import math
import random

import numpy as np

from ar_counting import CHARS, build, load_words, neg_log_likelihood

V = len(CHARS)
STOI = {c: i for i, c in enumerate(CHARS)}


def make_examples(words, k):
    """Every (last k letters -> next letter) pair, as integer arrays."""
    X, Y = [], []
    for w in words:
        ctx = [0] * k                                   # 0 is '.', the start marker
        for ch in w + '.':
            X.append(ctx)
            Y.append(STOI[ch])
            ctx = ctx[1:] + [STOI[ch]]
    return np.array(X), np.array(Y)


def init(k, emb, hidden, rng):
    return {
        'C': rng.normal(0, 1, (V, emb)),                                  # one vector per letter
        'W1': rng.normal(0, 1, (k * emb, hidden)) / math.sqrt(k * emb),
        'b1': np.zeros(hidden),
        'W2': rng.normal(0, 1, (hidden, V)) * 0.01,                       # start near uniform
        'b2': np.zeros(V),
    }


def forward(P, X):
    e = P['C'][X].reshape(len(X), -1)                   # (B, k*emb)
    h = np.tanh(e @ P['W1'] + P['b1'])                  # (B, hidden)
    z = h @ P['W2'] + P['b2']                           # (B, 27) logits
    z = z - z.max(1, keepdims=True)                     # numerically safe softmax
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return e, h, p


def loss_and_grads(P, X, Y):
    B = len(X)
    e, h, p = forward(P, X)
    loss = -np.log(p[np.arange(B), Y]).mean()
    # backward: the famous softmax + cross-entropy gradient is simply (p - onehot)
    dz = p.copy()
    dz[np.arange(B), Y] -= 1
    dz /= B
    g = {'W2': h.T @ dz, 'b2': dz.sum(0)}
    dh = dz @ P['W2'].T
    da = dh * (1 - h ** 2)                              # tanh'(a) = 1 - tanh(a)^2
    g['W1'] = e.T @ da
    g['b1'] = da.sum(0)
    de = (da @ P['W1'].T).reshape(B, X.shape[1], -1)
    g['C'] = np.zeros_like(P['C'])
    np.add.at(g['C'], X, de)                            # route gradients back to each letter's row
    return loss, g


def grad_check(k=3, emb=4, hidden=16):
    rng = np.random.default_rng(0)
    P = init(k, emb, hidden, rng)
    X = rng.integers(0, V, (8, k))
    Y = rng.integers(0, V, 8)
    _, g = loss_and_grads(P, X, Y)
    for name in P:
        idx = tuple(rng.integers(0, s) for s in P[name].shape)
        old = P[name][idx]
        P[name][idx] = old + 1e-5
        lp, _ = loss_and_grads(P, X, Y)
        P[name][idx] = old - 1e-5
        lm, _ = loss_and_grads(P, X, Y)
        P[name][idx] = old
        num = (lp - lm) / 2e-5
        print(f'  d loss / d {name}{list(idx)}:  backprop {g[name][idx]: .6e}   numerical {num: .6e}')


def sample(P, k, tau=1.0, max_len=20, rng=np.random):
    ctx, out = [0] * k, ''
    while len(out) < max_len:
        _, _, p = forward(P, np.array([ctx]))
        p = p[0] ** (1 / tau)                           # temperature (same as dividing logits by tau)
        p /= p.sum()
        j = rng.choice(V, p=p)                          # roll the dice
        if j == 0:
            break
        out += CHARS[j]
        ctx = ctx[1:] + [j]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, default=3, help='context length')
    ap.add_argument('--emb', type=int, default=4, help='embedding size')
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=3e-3, help='weight decay')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--data')
    ap.add_argument('--big', action='store_true')
    ap.add_argument('--plot', action='store_true', help='scatter-plot the first two embedding dimensions')
    ap.add_argument('--check', action='store_true', help='numerically verify the gradients and exit')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.check:
        return grad_check()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    words = load_words(args.data, args.big)
    random.shuffle(words)
    n_val = max(1, len(words) // 10)
    val, train = words[:n_val], words[n_val:]
    Xtr, Ytr = make_examples(train, args.k)
    Xva, Yva = make_examples(val, args.k)
    print(f'{len(train)} training words ({len(Ytr)} examples) · {len(val)} held-out words')

    P = init(args.k, args.emb, args.hidden, rng)
    m = {n: np.zeros_like(v) for n, v in P.items()}     # Adam moment estimates
    s = {n: np.zeros_like(v) for n, v in P.items()}
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(Ytr), 128)
        loss, g = loss_and_grads(P, Xtr[idx], Ytr[idx])
        lr = args.lr * (0.1 if step > 0.75 * args.steps else 1.0)
        for n in P:                                     # Adam: gradient descent with momentum + scaling
            g[n] += args.wd * P[n]                          # weight decay: keeps the network simple
            m[n] = 0.9 * m[n] + 0.1 * g[n]
            s[n] = 0.999 * s[n] + 0.001 * g[n] ** 2
            P[n] -= lr * (m[n] / (1 - 0.9 ** step)) / (np.sqrt(s[n] / (1 - 0.999 ** step)) + 1e-8)
        if step % max(1, args.steps // 8) == 0 or step == 1:
            vl, _ = loss_and_grads(P, Xva, Yva)
            print(f'step {step:5d}   train loss {loss:.3f}   held-out loss {vl:.3f}')

    # compare with the counting model on the same held-out words
    counts = build(train, args.k)
    count_val = sum(neg_log_likelihood(counts, w, args.k) for w in val) / len(Yva)
    neural_val, _ = loss_and_grads(P, Xva, Yva)
    print(f'\nheld-out loss per letter — counting (k={args.k}): {count_val:.3f}   neural: {neural_val:.3f}'
          f'   uniform: {math.log(V):.3f}')
    print('(lower = less surprised by names it has never seen = better generalisation)\n')

    known = set(words)
    for _ in range(args.n):
        name = sample(P, args.k, args.tau, rng=rng)
        print(f"  {name}{'   (copied from training data)' if name in known else ''}")

    if args.plot:
        import matplotlib.pyplot as plt
        E = P['C'][:, :2]
        plt.figure(figsize=(6, 6))
        plt.scatter(E[:, 0], E[:, 1], s=260, c=['#7a9e7e' if c in 'aeiouy' else '#c4654a' for c in CHARS])
        for i, c in enumerate(CHARS):
            plt.text(E[i, 0], E[i, 1], c, ha='center', va='center', color='white')
        plt.title('learned letter embeddings (green = vowels)')
        plt.show()


if __name__ == '__main__':
    main()
