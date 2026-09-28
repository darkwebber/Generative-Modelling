# Code for the series — every model, by hand

Everything from the videos, runnable, with backprop written out by hand so nothing is hidden.

- **Episode 02 · autoregressive:** `ar_counting.py` (pure Python), `ar_neural.py` (numpy)
- **Episode 03 · autoencoders & VAEs:** `vae.py` (numpy; downloads MNIST on first run)

## Episode 02 — autoregressive generation

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

## Episode 03 — autoencoders & VAEs

| file | what it is |
|------|------------|
| `vae.py` | an autoencoder **and** a VAE on MNIST (784 → 256 → 2 → 256 → 784) in numpy, every gradient written by hand, plus a gradient checker. ~5 s per epoch on a laptop CPU |
| `export_ep03_assets.py` | trains both (30 epochs) and exports weights + latent codes to `../video/ep03-assets.js` for the video and playground |
| `../video/vae-playground.html` | drag through the latent map, draw a digit and watch it get encoded, sample, interpolate |

```bash
python vae.py --check                     # every hand-written gradient vs a numerical one
python vae.py --model ae --plot           # autoencoder: gappy, sprawling latent map
python vae.py --model vae --plot          # VAE: packed N(0, I) map, grid of decodes, fresh samples
python vae.py --model vae --beta 4 --plot # a tighter leash: tidier map, blurrier digits
python vae.py --model vae --beta 0.1      # a looser one: sharper, but the map gets holes
```

What you should see (30 epochs, held-out digits, nats per image): the AE rebuilds at ≈ 144;
the VAE at ≈ 144.5 reconstruction + ≈ 6.1 KL — almost the same rebuild quality, but a latent
space you can actually sample from.

```
autoencoder   z = f_φ(x),  x̂ = g_θ(z),  L = Σ_pixels BCE(x, x̂)
VAE encoder   q_φ(z|x) = N(μ(x), σ²(x))
VAE loss      L = E_q[−log p_θ(x|z)] + β·KL(q_φ(z|x) ‖ N(0, I)),   KL = ½ Σ (μ² + σ² − log σ² − 1)
trick         z = μ + σ·ε,  ε ~ N(0, I)          ⇒  gradients flow through μ and σ
bound         log p_θ(x) ≥ −L   (the ELBO, for β = 1)
generate      z ~ N(0, I)  →  x = g_θ(z)
```
