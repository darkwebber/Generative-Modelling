# Code for the series — every model, by hand

Everything from the videos, runnable, with backprop written out by hand so nothing is hidden.

Each model's forward pass, loss and backprop live in its own file: those are the lesson. The plumbing they
share (also hand-written) is in `nn.py`: Adam (+ an EMA of the weights), a layer-list MLP (GAN, flow couplings),
a swish MLP (energy, score), a time-conditioned MLP (diffusion, and the next episodes), the 2-D ring and moons
data, and a finite-difference gradient checker. Episodes 02–04 keep Adam written inline, where it is first explained.

- **Episode 01 · densities:** `density.py` (numpy; a mixture of bell curves fitted by maximum likelihood, and sampled)
- **Episode 02 · autoregressive:** `ar_counting.py` (pure Python), `ar_neural.py` (numpy)
- **Episode 03 · autoencoders & VAEs:** `vae.py` (numpy; downloads MNIST on first run)
- **Episode 04 · evaluation:** `evals.py` (numpy; every metric from scratch)
- **Episode 05 · GANs:** `gan.py` (numpy; 2-D rings and MNIST)
- **Episode 06 · normalizing flows:** `flow.py` (numpy; RealNVP-style couplings in 2-D)
- **Episode 07 · energy-based models:** `ebm.py` (numpy; Langevin sampling and contrastive divergence in 2-D)
- **Episode 08 · score matching:** `score.py` (numpy; denoising score matching, noise-conditional net, annealed Langevin)
- **Episode 09 · diffusion:** `diffusion.py` (numpy; DDPM training and sampling, DDIM, 2-D blobs and MNIST)
- **Episode 10 · flow matching:** `flow_matching.py` (numpy; straight-road regression, Euler sampling, reflow, a label-conditioned net with guidance)

The asset exporters (`export/export_epNN_assets.py`) train the models shown in each episode and write their data next to
the episode page (`../video/episodes/epNN-assets.js`); trained nets are cached in `data/` (not in git). `export/common.py`
holds the shared paths.

## Episode 01 — densities

| file | what it is |
|------|------------|
| `density.py` | a mixture of bell curves, p(x) = Σₖ πₖ N(x; μₖ, σₖ²), fitted by maximum likelihood with EM on the episode's own two-bump data, then sampled: pick a bump, stretch and shift a bell curve of noise |
| `export/export_ep01_assets.py` | the episode-01 trailer: collects real samples from the models of episodes 02–09 → `../video/episodes/ep01-assets.js` |
| `../video/labs/ep01-density-lab.html` | fit bumps by hand or with EM, watch the log-likelihood score, draw samples |

```bash
python density.py            # 1–4 bumps: train vs held-out log-likelihood (two wins; more start memorising)
python density.py --check    # EM never lowers the likelihood; samples follow the fitted density
```

## Episode 02 — autoregressive generation

| file | what it is |
|------|------------|
| `ar_counting.py` | character n-gram model in pure Python: count → normalise → sample, with context length `k`, temperature `τ`, and per-letter surprise scoring |
| `ar_neural.py` | the same model as an MLP (embeddings → tanh → softmax) in numpy, with **hand-written backprop**, a gradient checker, and a held-out comparison against counting |
| `names.txt` | the ~380 first names used in the video (use `--big` to download ~32k) |
| `../video/labs/ep02-ar-playground.html` | the interactive playground: open it in a browser |

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
| `export/export_ep03_assets.py` | trains both (30 epochs) and exports weights + latent codes to `../video/episodes/ep03-assets.js` for the video and playground |
| `../video/labs/ep03-vae-playground.html` | drag through the latent map, draw a digit and watch it get encoded, sample, interpolate |

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

## Episode 04 — evaluating generative models

| file | what it is |
|------|------------|
| `evals.py` | Inception Score, FID, precision/recall (k-NN manifolds), nearest-training-image distance and a classifier two-sample test — all from scratch in numpy — plus a 97.8%-accurate digit network (backprop by hand) used as the feature extractor |
| `export/export_ep04_assets.py` | computes every number shown in the video → `../video/episodes/ep04-assets.js` |
| `export/export_ep04_lab.py` | the feature net + image pools for `../video/labs/ep04-metric-lab.html` |

```bash
python evals.py        # scores five "models" (2,000 samples each) and prints the scorecard
```

What you should see:

```
model       IS ↑     FID ↓   prec ↑  recall ↑  NN-train  C2ST (50%=best)
real        9.52       0.9     0.88      0.88      4.28             50%
vae         6.45      51.9     0.95      0.00      4.38             91%
copier      9.55       1.4     0.88      0.88      0.00             52%
ones        1.03     217.6     0.91      0.12      0.00             94%
noise       1.76     403.9     0.00      0.00     14.95            100%
```

The photocopier (training images passed off as samples) ties real data on five of six
judges; only the nearest-training-image distance catches it. The VAE has *higher* precision
than real data and almost zero recall: every sample is a safe, average-looking digit.

```
IS          exp( E_x KL( p(y|x) || p(y) ) )
FID         ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^1/2)
precision   share of samples inside the real k-NN manifold          (fidelity)
recall      share of real points inside the samples' k-NN manifold  (diversity)
likelihood  a 1%-good / 99%-noise mixture loses only log 100 ≈ 4.6 nats
```

One practical gotcha we hit: MNIST's test set is ordered (its two halves were written by
different groups of people), so always shuffle before splitting "real vs real", or even
real data looks distinguishable from itself.

## Episode 05 — GANs

| file | what it is |
|------|------------|
| `gan.py` | a tiny MLP class with hand-written backprop (checked numerically), the GAN game step (original and non-saturating losses), a 2-D ring-of-Gaussians trainer that tracks modes found, and an MNIST GAN |
| `export/export_ep05_assets.py` | trains the runs shown in the video and scores the MNIST GAN with episode 04's judges → `../video/episodes/ep05-assets.js` |
| `../video/labs/ep05-gan-arena.html` | a GAN training live in the browser; change the learning rates and cause mode collapse yourself |

```bash
python gan.py --check                              # hand-written gradients vs numerical ones
python gan.py --data ring --plot                   # finds the 8 blobs (≈ 5k steps, seconds)
python gan.py --data ring --lr-g 3e-3 --lr-d 1e-4  # fast forger, slow judge: modes go missing and hop
python gan.py --data mnist --epochs 40 --plot      # ≈ 17 s per epoch on a laptop CPU
```

What the video's MNIST GAN scores with episode 04's judges (2,000 samples):

```
          FID ↓   precision ↑   recall ↑   IS ↑    C2ST → 50%   NN-train
GAN        21.3       76%          77%      6.59       65%         4.82
VAE        51.9       95%          <1%      6.45       91%         4.38
real        0.9       88%          88%      9.52       50%         4.28
```

The GAN trades a little precision for an enormous gain in recall (diversity) over the VAE,
and it is not copying (its nearest training image is as far away as a real unseen digit's).

```
game          min_G max_D  E_data[log D(x)] + E_z[log(1 − D(G(z)))]
perfect judge D*(x) = p_data(x) / (p_data(x) + p_g(x))
G minimises   V(G, D*) = −log 4 + 2·JSD(p_data ‖ p_g)
in practice   min_G −log D(G(z))      (non-saturating: strong gradients when fakes are bad)
```

## Episode 06 — normalizing flows

| file | what it is |
|------|------------|
| `flow.py` | a RealNVP-style flow: affine coupling layers (s, t from small MLPs), forward and inverse, exact log-likelihood, and the full backward pass by hand (checked numerically) |
| `export/export_ep06_assets.py` | trains the flows shown in the video (moons 8 and 2 layers, 8 blobs) → `../video/episodes/ep06-assets.js` |
| `../video/labs/ep06-flow-lab.html` | a flow training live in the browser: density, samples, warped grid, a layer scrubber, and a click-to-probe log p(x) |

```bash
python flow.py --check                                   # gradients vs numerical; inverse error
python flow.py --plot                                    # two moons, 8 layers (≈ 1 min)
python flow.py --layers 2 --plot                         # too shallow to bend the moons
python flow.py --data ring --steps 12000 --seed 1 --plot # 8 blobs, joined by thin bridges
```

Held-out −log p(x) of the flows in the video, in nats per point (lower is better) — exact, not a bound:

```
                      flow    best single Gaussian
two moons, 8 layers   1.19          2.70
two moons, 2 layers   1.50          2.70
8 blobs,   8 layers   1.00          3.54      (a perfect model would score ≈ 0.68)
```

```
change of variables   p(x) = p(z) · |det ∂z/∂x|,   z = f(x)
coupling layer        a′ = a,   b′ = b·e^s(a) + t(a)    →   log|det| = s(a)   (triangular Jacobian)
inverse               b = (b′ − t(a))·e^−s(a)            (s, t never need inverting)
training              min  E_x[ ½|f(x)|² + log 2π − Σₖ sₖ(x) ]
```

## Episode 07 — energy-based models

| file | what it is |
|------|------------|
| `ebm.py` | an energy network (2→96→96→96→1, swish) with hand-written backprop for both ∇θ and ∇ₓ; a Langevin sampler; persistent contrastive divergence with a replay buffer; the exact 2-D likelihood (Z summed on a 400 × 400 grid); and the 1-D "sculpting" demo (25 bumps, exact fantasies) |
| `export/export_ep07_assets.py` | trains the model shown in the video and exports landscapes, marble paths, the mixing chain, valley shares and the Metropolis chain → `../video/episodes/ep07-assets.js` |
| `../video/labs/ep07-ebm-lab.html` | an EBM training live in the browser: landscape or density, fantasies, knobs for Langevin steps, η, jiggle and the E² penalty, and click-to-drop marbles |

```bash
python ebm.py --check                  # gradients (θ and x) vs numerical
python ebm.py                          # 8 blobs, the video's settings (≈ 3 min)
python ebm.py --alpha 0.001            # weak E² penalty: one valley swallows most of the probability
python ebm.py --data moons --plot      # energy, density and Langevin samples
python ebm.py --sculpt                 # 1-D: data digs, fantasies raise, the gap closes
```

The model in the video (8 blobs, 4,000 steps, 60 Langevin steps of η = 0.003 per update, α = 0.01), in nats per point:

```
held-out −log p (exact, 2-D grid)   EBM 1.07     flow (episode 6) 1.00     perfect 0.68     one Gaussian 3.54
share of probability per valley     26.0%  21.1%  17.2%  15.8%  11.3%  4.7%  2.8%  1.1%   (data: 12.5% each)
one Langevin chain, 30,000 steps    crossed a ridge once
```

Short-run samples look perfect — every valley gets its marbles — yet the valleys are not equally
deep: the chains never move between valleys, so training never compares them. Training twice as
long (8,000 steps) scored worse (1.37), not better.

```
Boltzmann            p(x) = e^−E(x) / Z,        Z = ∫ e^−E(x) dx
learning             ∇θ[−log p(x)] = ∇θE(x) − E_{x′∼pθ}[∇θE(x′)]      (data down, fantasies up)
Metropolis           accept x′ with probability min(1, e^−(E(x′) − E(x)))  (Z cancels)
Langevin             x ← x − η∇ₓE(x) + √(2η)·ε
the score            ∇ₓ log p(x) = −∇ₓE(x)                              (∇ₓ log Z = 0)
```

## Episode 08 — score matching & Langevin dynamics

| file | what it is |
|------|------------|
| `score.py` | a noise-conditional score network ([x, y, log σ] → 128 → 128 → 128 → 2, swish) trained by denoising score matching with hand-written backprop; plain and annealed Langevin; exact scores of the Gaussian-mixture data for checking |
| `export/export_ep08_assets.py` | trains the nets in the video → `../video/episodes/ep08-assets.js` |
| `../video/labs/ep08-score-lab.html` | train a score network live, look at its arrows at any σ (against the true ones), sample with plain or annealed Langevin, drop particles anywhere |

```bash
python score.py --check            # gradients vs numerical
python score.py                    # 8 blobs: learned vs exact score at three noise levels
python score.py --data twoblob     # 80/20 blobs: plain vs annealed Langevin shares
```

From the video (80/20 blobs, 4,000 particles started uniformly, the same 300 steps each):

```
share in the 80% blob     plain Langevin at σ = 0.05: 50%   (3,000 steps: 49%)     annealed σ 3 → 0.05: 77%
```

```
score                 s(x) = ∇ₓ log p(x)                       (no Z: ∇ₓ log Z = 0)
Langevin              x ← x + η·s(x) + √(2η)·ε                 (stationary: p·s − ∇p = 0)
implicit SM           E_p[ ‖s_θ‖² + 2·tr ∇ₓ s_θ ]               (Hyvärinen, 2005)
denoising SM          E ‖σ·s_θ(x + σε, σ) + ε‖²                 (Vincent, 2011) → s_θ = ∇ log p_σ
Tweedie               E[x | x̃] = x̃ + σ²·∇ log p_σ(x̃)
annealed Langevin     for σ₁ > … > σ_L: T steps with η = c·σ²    (Song & Ermon, 2019)
```

## Episode 09 — diffusion models (DDPM & DDIM)

| file | what it is |
|------|------------|
| `diffusion.py` | a noise predictor ε_θ(x, t): an MLP with a sinusoidal embedding of t added into every hidden layer (swish, EMA weights, optional Fourier features of x), trained on ‖ε − ε_θ(x_t, t)‖² with hand-written backprop (for MNIST the net outputs its clean-image guess x̂₀ = tanh(F) and reports ε_θ = (x_t − √ᾱ·x̂₀)/√(1 − ᾱ), with Min-SNR weights w_t = min(1, 5/SNR_t): an MLP finds clean images far easier to output than faint noise); ancestral DDPM sampling with the posterior variance; deterministic DDIM with any number of steps |
| `export/export_ep09_assets.py` | trains the three nets in the video (the title, the eight blobs, MNIST) → `../video/episodes/ep09-assets.js`, plus the MNIST net in 8 bits → `../video/labs/ep09-model.js`. `python export/export_ep09_assets.py title` (or `ring`, `mnist`) trains just one, so you can run them in parallel |
| `../video/labs/ep09-diffusion-lab.html` | the real MNIST diffusion model running in your browser: watch digits form from static (x_t or the network's guess x̂₀), DDPM vs DDIM at any step count, morph between two noise images, draw something and let the model "fix" it (SDEdit) |

```bash
python diffusion.py --check                  # gradients vs numerical
python diffusion.py --data ring              # 8 blobs: DDPM 1000 steps vs DDIM 100 / 20
python diffusion.py --data mnist --plot      # ~15 min: 64 digits from noise
```

From the video (the eight blobs sit at radius 2.00 ± 0.12; 600 samples from the same noise):

```
DDPM 1000 steps: radius 2.00 ± 0.12     DDIM 100: 1.99 ± 0.10     DDIM 20: 1.95 ± 0.09     DDIM 5: 1.15 ± 0.39 (it cracks)
```

```
schedule        β_t: 1e-4 → 0.02 (linear, T = 1000),   ᾱ_t = Π_{s≤t} (1 − β_s)
forward         x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε                                  (any step, in one jump)
training        E ‖ε − ε_θ(x_t, t)‖²                                           (Ho, Jain & Abbeel, 2020)
= score         ε_θ = −√(1 − ᾱ_t)·s_θ(x_t)                                     (episode 8's denoising score matching)
denoiser        x̂₀ = (x_t − √(1 − ᾱ_t)·ε_θ) / √ᾱ_t                            (Tweedie)
DDPM step       x_{t−1} = (x_t − β_t/√(1 − ᾱ_t)·ε_θ) / √(1 − β_t) + σ_t·z,   σ_t² = β_t(1 − ᾱ_{t−1})/(1 − ᾱ_t)
DDIM step       x_s = √ᾱ_s·x̂₀ + √(1 − ᾱ_s)·ε_θ                               (Song, Meng & Ermon, 2021; no fresh noise)
why it works    the ELBO of a VAE whose 1000 latent layers are the noising chain splits into one Gaussian KL per step
```

## Episode 10 — flow matching and reflow

| file | what it is |
|------|------------|
| `flow_matching.py` | a velocity net v_θ(x, t) (episode 9's time-embedded MLP), trained on ‖v_θ(xₜ, t) − (x₁ − x₀)‖² with xₜ = (1 − t)·x₀ + t·x₁, x₀ noise and x₁ data, with hand-written backprop (for MNIST the net outputs its clean-image guess x̂₁ = tanh(F) and reports v = (x̂₁ − x)/(1 − t), weighted by min(1, 5(1 − t)²)); Euler sampling with any number of steps; reflow (retrain on the pairs (x₀, flow(x₀))); straightness; an optional one-hot label with a null slot (dropped 10% of the time) and guided sampling v = v_∅ + w·(v_y − v_∅), a preview of episode 11 |
| `export/export_ep10_assets.py` | trains the nets in the video (the title and its reflow, the eight blobs and their reflow, the labelled blobs, MNIST and its reflow), scores MNIST with episode 4's judges against episode 9's DDIM at the same number of calls → `../video/episodes/ep10-assets.js`, and writes the lab's nets → `../video/labs/ep10-model.js`. `python export/export_ep10_assets.py glyph` (or `ring`, `cond`, `mnist`) trains just one, so you can run them in parallel |
| `../video/labs/ep10-flow-lab.html` | the real models running in your browser: flow noise onto the ring with 1–64 Euler steps (round 1 vs reflow), scrub the wind through time and trace probe roads, ask the labelled model for a blob and turn up the guidance, make digits in a handful of steps |

```bash
python flow_matching.py --check              # gradients vs numerical (4 variants)
python flow_matching.py --data ring          # 8 blobs, then Euler with 1 / 4 / 32 steps
python flow_matching.py --data mnist --plot  # digits from noise in 32 steps
```

From the video (eight blobs at radius 2; MNIST scored on 2,000 samples, lower FID is better):

```
blobs, 1 Euler step      round 1: radius 0.08 (everything in the middle)    after reflow: radius 2.00
straightness             round 1: 0.545                                     after reflow: 1.000
MNIST FID by calls       1      2      4      8      16     32     100
  flow matching          264    142    48     22.2   13.1   9.8    8.1
  DDIM (episode 9)       —      210    53     21.4   13.1   10.5   9.1        (DDPM, 1,000 calls: 9.8)
  after reflow           19.5   17.9   17.0   16.2   15.7
```

```
path            xₜ = (1 − t)·x₀ + t·x₁                  x₀ ~ N(0, I), x₁ ~ data, t ~ U(0, 1)
target          dxₜ/dt = x₁ − x₀                        (a straight road at constant speed)
loss            E ‖v_θ(xₜ, t) − (x₁ − x₀)‖²              (Lipman et al., 2023; Liu et al., 2023; Albergo & Vanden-Eijnden, 2023)
optimum         v*(x, t) = E[x₁ − x₀ | xₜ = x]          its flow carries N(0, I) to the data, marginal by marginal
sampling        x ← x + Δt·v_θ(x, t),  Δt = 1/K
reflow          new pairs (x₀, flow(x₀)): they never cross, so the new roads are straight
one arrow       x̂₁ = xₜ + (1 − t)·v,   x̂₀ = xₜ − t·v,   score = −x̂₀/(1 − t)
guidance        v = v_∅ + w·(v_y − v_∅)                 (episode 11)
```

