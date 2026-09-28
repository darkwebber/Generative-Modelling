# Generative Modelling — a motion-graphic series

First-principles animated explainers. Each episode is a single deterministic `<canvas>`
animation (seeded, time-driven), so every frame can be re-rendered exactly. Visual style:
the *Midnight Atelier* system — warm darks, film grain, and colours with fixed meanings
(terracotta = data/tokens, ink = distributions, amber = parameters/knobs, rose =
noise/dice, plum = models, sage = insight).

## Episode 01 — What is generative modelling? (~7:10)

`generative-modelling.html`

| # | Scene | Math introduced |
|---|-------|-----------------|
| 00 | Intro: noise → data | the colour language |
| 01 | Data is points (2-pixel image → ℝ², digit → ℝ⁷⁸⁴) | x ∈ ℝᵈ |
| 02 | Discriminative: learn the border | p(y \| x) |
| 03 | Generative: learn where data lives, then sample | p(x), Bayes' rule |
| 04 | What is a density? histogram → curve | ∫p = 1, P(a≤x≤b) = ∫ₐᵇ p |
| 05 | Maximum likelihood with a live score | θ* = argmax Σ log p_θ(xᵢ) |
| 06 | Sampling: bend noise through g | x = g(z), p(x) = p(z)\|dz/dx\| |
| 07 | The catch: random pixels are static | manifolds |
| 08 | The zoo: AR, VAE, GAN, flows, diffusion | each family's core equation |
| 09 | Big picture: 1,400 points flow onto data | z → g_θ(z) ∼ p_θ ≈ p_data |
| 10 | Recap | — |

## Episode 02 — Autoregressive generation (~8:47)

`ep02-autoregressive.html` · playground: `ar-playground.html` · code: `../code/`

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | a sentence types itself; "one question, over and over" — what is it? | — |
| 01 | Where we left off | x for text: 27⁵ = 14M words, 27¹⁰⁰ ≈ 10¹⁴³ sentences > atoms in the universe | why a table of p(x) is impossible |
| 02 | The guessing game | h → he → hel → hell: a 14M-way guess becomes five 27-way guesses | — |
| 03 | The chain rule | nested slices of a probability bar; fractions multiply | p(x) = ∏ p(x_t \| x_<t), exact |
| 04 | The one question | context → p_θ → distribution → sample → append → repeat | p_θ(· \| x_<t); "auto-regressive" |
| 05 | Just count | 2,516 letter pairs from 381 names fill a 27×27 table | p(b\|a) = N(a,b)/Σ N(a,·) = MLE |
| 06 | Let's generate | a dice pointer lands in probability slices; a new name appears | u ∼ U(0,1), inverse CDF |
| 07 | More context | k = 1, 2, 3 samples; copies of training data; 27ᵏ explosion, 95 % empty rows | memorisation vs generalisation |
| 08 | Neural net | embeddings (vowels cluster), logits → softmax, surprise curve | softmax, cross-entropy = −log p_θ(x) |
| 09 | Temperature | the bars sharpen and flatten as τ moves | p ∝ e^{z/τ} |
| 10 | Scaling up | tokens, 128k context, causal attention mask, pixels & audio | same product, same loop |
| 11 | Build it yourself | the 20-line Python model + whole-episode recap | — |
| 12 | The hook | a digit drawn pixel by pixel; 3M steps; squeeze into two knobs… | → *Episode 03: autoencoders, latent spaces & VAEs* |

Every number in the counting scenes (tables, samples, "copied" tags, seen-context counts,
the "emma" surprise bars) is computed live from the bundled `names.js` dataset.

## Episode 03 — Autoencoders & VAEs (~8:42)

`ep03-autoencoders-vae.html` · playground: `vae-playground.html` · code: `../code/vae.py`

Built from `src/ep03-scenes.js` + the shared engine: `python src/build_ep03.py`. Every image,
latent map, sample and interpolation is decoded live in the browser from the **real trained
networks** (`ep03-assets.js`, exported by `code/export_ep03_assets.py`).

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | a digit morphs as two numbers move — "how can 2 numbers hold a picture?" | — |
| 01 | The compression game | "a seven, leaning right": meaning compresses; real data is a thin sliver | 784 → 2 |
| 02 | The autoencoder | encoder → 2-number keyhole → decoder; real rebuilds (and honest mistakes) | z = f(x), x̂ = g(z), BCE |
| 03 | Inside the latent space | 2,500 codes cluster by digit without labels; a probe decodes as it walks | map of meaning |
| 04 | Holes in the map | random codes decode to hybrids/smudges; arbitrary scale, no shape | why AEs don't generate |
| 05 | Fuzzy codes on a leash | points → clouds; the cheat (σ → 0); the KL leash packs them into N(0, I) | q(z\|x) = N(μ, σ²) |
| 06 | The math | two terms, closed-form KL with its minima, log p(x) = log ∫…, ELBO floor, real numbers | KL = ½Σ(μ²+σ²−log σ²−1), ELBO |
| 07 | Reparameterisation | gradients blocked by a dice roll → moved aside | z = μ + σ·ε, ∂z/∂μ = 1, ∂z/∂σ = ε |
| 08 | A map you can sample from | AE vs VAE maps, 13×13 decoded grid, fresh samples, interpolation | z ~ N(0, I) → g(z) |
| 09 | The catch | blur = averaging plausible options; the β tug-of-war | argmin E‖x−x̂‖² = E[x] |
| 10 | Build it | the VAE in a dozen numpy lines + recap | — |
| 11 | Hook → evals | three models: blurry, a photocopier, a one-trick pony — every metric can be fooled | → *Episode 04: evaluation*, then GANs |

## Watch / scrub

Open any episode `.html` in a browser: play/pause (space), scrub, ←/→ to skip 5 s, or
jump by chapter.

## Render the MP4s

Needs Node, `playwright` (Chromium) and `ffmpeg`.

```bash
cd video
./fetch-fonts.sh                               # cache the Google Fonts locally (once)
WORKERS=4 node render.mjs generative-modelling.html 30 generative-modelling.mp4
WORKERS=4 node render.mjs ep02-autoregressive.html 30 ep02-autoregressive.mp4
WORKERS=4 node render.mjs ep03-autoencoders-vae.html 30 ep03-autoencoders-vae.mp4
```

`node render.mjs <episode.html> <fps> <out.mp4> <start-s> <end-s>` renders one section
while iterating. Set `FFMPEG=/path/to/ffmpeg` if ffmpeg isn't on your PATH
(`pip install imageio-ffmpeg` ships one).
