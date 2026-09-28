"""
Episode 02 — Autoregressive generation by counting.

A character-level n-gram language model in pure Python (no dependencies).

    p(x) = prod_t p(x_t | x_<t)          # the chain rule — exact
    p(x_t | x_<t) ~= p(x_t | last k letters) = N(context, x_t) / N(context)

Try:
    python ar_counting.py                      # bigram names, tau = 1
    python ar_counting.py --k 3 --n 20         # longer memory
    python ar_counting.py --k 3 --tau 0.4      # cold: safe & repetitive
    python ar_counting.py --k 3 --tau 2.0      # hot: wild
    python ar_counting.py --explain            # watch one name being generated, step by step
    python ar_counting.py --score emma xqzzy   # how surprised is the model by these?
    python ar_counting.py --data my_words.txt  # train on anything: cities, pokémon, your own words
    python ar_counting.py --big                # download ~32k names (makemore dataset) if online
"""
import argparse
import math
import os
import random
import urllib.request
from collections import defaultdict

CHARS = '.abcdefghijklmnopqrstuvwxyz'   # '.' marks both the start and the end of a word
HERE = os.path.dirname(os.path.abspath(__file__))
BIG_URL = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'


def load_words(path=None, big=False):
    if big:
        try:
            text = urllib.request.urlopen(BIG_URL, timeout=10).read().decode()
            print(f'downloaded {BIG_URL}')
            return clean(text)
        except Exception as e:  # offline, blocked, ...
            print(f'(could not download the big dataset: {e}; using the bundled names)')
    path = path or os.path.join(HERE, 'names.txt')
    with open(path, encoding='utf-8') as f:
        return clean(f.read())


def clean(text):
    words = (''.join(c for c in w.lower() if c in CHARS[1:]) for w in text.split())
    return sorted({w for w in words if len(w) >= 2})


# ── 1. "training" = counting ──────────────────────────────────
def count(words, k):
    """N[context][next] = how often `next` followed the k-letter `context`."""
    N = defaultdict(lambda: defaultdict(int))
    for w in words:
        s = '.' * k + w + '.'          # pad the start so the first letter has a context
        for i in range(len(s) - k):
            N[s[i:i + k]][s[i + k]] += 1
    return N


def build(words, k_max):
    return {k: count(words, k) for k in range(1, k_max + 1)}


# ── 2. the model: p(next | context) ──────────────────────────
def next_dist(models, context, k, tau=1.0, alpha=0.05, beta=1.0):
    """
    Probability of each of the 27 symbols coming next.
    * start from the 1-letter model, add-alpha smoothed so nothing is impossible
    * then blend in longer contexts:  p_j = (N_j + beta * p_{j-1}) / (N_j_total + beta)
      (rarely-seen contexts lean on the shorter model; well-seen ones trust their counts)
    * tau: temperature;  p_i ∝ exp(log p_i / tau)
    Returns the distribution and the longest context that was actually seen.
    """
    s = '.' * k + context
    row = models[1].get(s[-1:], {})
    total = sum(row.values())
    p = [(row.get(c, 0) + alpha) / (total + alpha * len(CHARS)) for c in CHARS]
    used = s[-1:]
    for j in range(2, k + 1):
        row = models[j].get(s[-j:])
        if row is None:                               # never seen: longer ones won't be either
            break
        total = sum(row.values())
        p = [(row.get(c, 0) + beta * pc) / (total + beta) for c, pc in zip(CHARS, p)]
        used = s[-j:]
    logits = [math.log(pc) / tau for pc in p]
    m = max(logits)
    e = [math.exp(z - m) for z in logits]            # softmax
    total = sum(e)
    return [v / total for v in e], used


# ── 3. generation: look up, roll the dice, append, repeat ────
def sample(models, k, tau=1.0, explain=False, max_len=20):
    out, logp = '', 0.0
    while len(out) < max_len:
        p, ctx = next_dist(models, out, k, tau)
        u = random.random()                           # the dice
        cum, j = 0.0, len(CHARS) - 1
        for i, pi in enumerate(p):
            cum += pi
            if u < cum:
                j = i
                break
        logp += math.log(p[j])
        if explain:
            top = sorted(range(27), key=lambda i: -p[i])[:6]
            bars = '  '.join(f"{CHARS[i]}:{p[i]:.2f}" for i in top)
            print(f"  context '{ctx}'  ->  {bars}   | u={u:.3f} picks '{CHARS[j]}'")
        if CHARS[j] == '.':
            break
        out += CHARS[j]
    return out, logp


# ── 4. scoring: how surprised is the model? ──────────────────
def neg_log_likelihood(models, word, k):
    """-log p(word) = sum over letters of -log p(x_t | x_<t)   (cross-entropy)"""
    nll, ctx = 0.0, ''
    for ch in word + '.':
        p, _ = next_dist(models, ctx, k, tau=1.0)
        nll -= math.log(p[CHARS.index(ch)])
        ctx += ch
    return nll


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, default=1, help='context length (letters of memory)')
    ap.add_argument('--tau', type=float, default=1.0, help='temperature')
    ap.add_argument('--n', type=int, default=15, help='how many samples')
    ap.add_argument('--data', help='a text file of words (default: bundled names.txt)')
    ap.add_argument('--big', action='store_true', help='download ~32k names')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--explain', action='store_true', help='print every step of one generation')
    ap.add_argument('--score', nargs='*', help='words to score')
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    words = load_words(args.data, args.big)
    models = build(words, max(1, args.k))
    known = set(words)
    print(f'{len(words)} training words · context k = {args.k} · temperature τ = {args.tau}')

    n_letters = sum(len(w) + 1 for w in words)
    avg = sum(neg_log_likelihood(models, w, args.k) for w in words) / n_letters
    print(f'training loss: {avg:.3f} nats per letter   (uniform guessing: {math.log(27):.3f})\n')

    if args.explain:
        name, logp = sample(models, args.k, args.tau, explain=True)
        print(f"\n  -> '{name}'   log p = {logp:.2f}\n")

    if args.score:
        for w in args.score:
            nll = neg_log_likelihood(models, w.lower(), args.k)
            print(f'  -log p({w!r}) = {nll:6.2f}    p = {math.exp(-nll):.2e}')
        print()

    new = 0
    for _ in range(args.n):
        name, _ = sample(models, args.k, args.tau)
        tag = '   (copied from training data)' if name in known else ''
        new += name not in known
        print(f'  {name}{tag}')
    print(f'\n{new}/{args.n} samples are brand new')


if __name__ == '__main__':
    main()
