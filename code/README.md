# Episode 02 code — autoregressive generation, by hand

Everything from the video, runnable. Start with counting, then replace the table with a
neural network. Only the neural version needs a dependency (`pip install numpy`).

| file | what it is |
|------|------------|
| `ar_counting.py` | character n-gram model in pure Python: count → normalise → sample, with context length `k`, temperature `τ`, and per-letter surprise scoring |
| `ar_neural.py` | the same model as an MLP (embeddings → tanh → softmax) in numpy, with **hand-written backprop**, a gradient checker, and a held-out comparison against counting |
| `names.txt` | the ~380 first names used in the video (use `--big` to download ~32k) |
| `../video/ar-playground.html` | the interactive playground: open it in a browser |

## Five experiments to try

```bash
# 1. The generation loop, step by step: context → distribution → dice → letter
python ar_counting.py --explain --seed 1

# 2. More memory = better names… until it starts copying the training data
python ar_counting.py --k 1 --n 15
python ar_counting.py --k 3 --n 15
python ar_counting.py --k 5 --n 15      # watch the "(copied from training data)" tags

# 3. Temperature: same model, different boldness
python ar_counting.py --k 3 --tau 0.3
python ar_counting.py --k 3 --tau 2.0

# 4. How surprised is the model?  (-log p = the cross-entropy loss)
python ar_counting.py --k 3 --score emma olivia xqzzy

# 5. Neural net vs counting on names neither has seen
python ar_neural.py --check             # the hand-written gradients match numerical ones
python ar_neural.py                     # held-out loss: counting ≈ 2.8, neural ≈ 2.3
python ar_neural.py --emb 2 --plot      # do the vowels cluster in the learned embedding?
```

Train on anything: `--data cities.txt`, `--data pokemon.txt`, your own words. One word per
line (or separated by spaces); only a–z is kept.

## The math, in four lines

```
chain rule    p(x) = ∏_t p(x_t | x_<t)                      exact, for any sequence
model         p_θ(x_t | x_<t) = softmax(z),  z = f_θ(last k tokens)
learn         L(θ) = −Σ_t log p_θ(x_t | x_<t)   = −log p_θ(x)   (maximum likelihood)
sample        x_t ∼ p_θ(· | x_<t)^{1/τ}, append, repeat
```

Counting with add-α smoothing is the maximum-likelihood solution for the table; for
longer contexts `ar_counting.py` blends in shorter ones
(`p_k = (N_k + β·p_{k−1}) / (N_k,total + β)`), so rare contexts lean on what the model
knows about shorter ones.
