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

## Episode 04 — Evaluating generative models (~9:30)

`ep04-evaluation.html` · playground: `metric-lab.html` · code: `../code/evals.py`

Built from `src/ep04-scenes.js` (`python src/build_ep04.py`). Every score on screen comes from
`ep04-assets.js`, computed by `code/export_ep04_assets.py` on real samples.

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | the three suspects from episode 3; "today we build the judges — and catch them lying" | — |
| 01 | What is "good"? | a classifier has an answer key; "draw a 7" has infinitely many | p_θ ≈ p_data; fidelity · diversity · novelty |
| 02 | Likelihood | held-out −log p; VAE ≤ 150.6 nats (0.28 bits/pixel); copier and one-trick pony → ∞ | bits/dim = NLL / (D ln 2) |
| 03 | How likelihood lies | 99% static + 1% VAE loses only 4.6 nats | log(0.01 p) = log p − log 100 |
| 04 | A better eye | pixels: a shifted 7 is farther than a 1; features fix it (mostly) | feature distance |
| 05 | Inception Score | sharp verdicts × spread classes; copier 9.55 ≈ real 9.52 | IS = exp E KL(p(y\|x) ‖ p(y)) |
| 06 | FID | Gaussian clouds in feature space; copier 1.4 ≈ real 0.9 | ‖μ_r−μ_g‖² + Tr(Σ_r+Σ_g−2(Σ_rΣ_g)^½) |
| 07 | Precision & recall | k-NN territories; VAE 95% / <1%, one-trick pony 91% / 12% | fidelity vs diversity |
| 08 | Is it new? | nearest training image: copier 0.00, real 4.28, VAE 4.38; contamination | memorisation |
| 09 | A judge that learns | classifier two-sample test: VAE 91%, copier 52%, real 50% | C2ST |
| 10 | Language models | perplexity, benchmarks, LLM-as-judge, preference arenas | exp(−(1/T)Σ log p) |
| 11 | Scorecard | the full table; Goodhart's law | — |
| 12 | Hook → GANs | the judge becomes a teacher; forger vs detective; ends at 50% | min_G max_D … → *Episode 05: GANs* |

## Episode 05 — GANs (~8:30)

`ep05-gans.html` · playground: `gan-arena.html` · code: `../code/gan.py`

Built from `src/ep05-scenes.js` (`python src/build_ep05.py`). The 2-D scenes replay real
training runs frame by frame (samples, the judge's field and its gradients), and the digit
scenes use a real MNIST GAN — all trained by `code/export_ep05_assets.py`.

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | digits appear epoch by epoch — from a network that never saw one | — |
| 01 | Two players | G: z → x, D: x → P(real); real data only flows into D | the game |
| 02 | The perfect judge | pointwise: max a·log y + b·log(1−y) | D* = p_d / (p_d + p_g) |
| 03 | What G minimises | plug D* back in; live JSD as p_g slides onto p_data | V = −log 4 + 2·JSD |
| 04 | Learning via the judge | D's field + gradient arrows; saturating vs non-saturating loss | ∂L/∂θ = ∂L/∂x · ∂x/∂θ |
| 05 | Watch it train | real 8-Gaussian run: modes found over time | — |
| 06 | Digits | MNIST GAN epoch by epoch; scored with episode 4's judges vs the VAE | FID, precision, recall, C2ST |
| 07 | Mode collapse | real mode-hopping run (fast forger, slow judge) | — |
| 08 | A game is not a hill | GDA on V = x·y spirals out; JSD flat vs Wasserstein slope | radius × √(1+η²); W = \|θ\| |
| 09 | In the wild | StyleGAN, one-pass sampling, adversarial losses everywhere, deepfakes | — |
| 10 | Build it | the game in numpy + recap | — |
| 11 | Hook → flows | GANs can't answer "how likely?"; an invertible warp could | p(x) = p(z)\|det ∂z/∂x\| → *Episode 06: normalizing flows* |

## Episode 06 — Normalizing flows (~9:50)

`ep06-flows.html` · playground: `flow-lab.html` · code: `../code/flow.py`

Built from `src/ep06-scenes.js` (`python src/build_ep06.py`). Every flow on screen is a real
RealNVP-style flow trained by exact maximum likelihood in `code/export_ep06_assets.py`:
the layer-by-layer positions, warped grids and density maps are its actual outputs.

The spine of the episode is one idea built up in steps: **slope → Jacobian → determinant →
triangular → coupling**. Each step answers the question the previous one raises.

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | noise → moons, then backwards; "p(x) = ?" — the question a GAN can't answer | — |
| 01 | Probability is sand | 1-D: a slice dz holds p(z)dz; zoom in and the curve is a line — the slope is the stretch | dx = f′(z)dz, p(x) = p(z)/\|f′(z)\| |
| 02 | The slope becomes a matrix | a 2-D map; zoom until a tiny square becomes a parallelogram; nudge z₁, then z₂ — the two output arrows are its edges, and side by side they are the Jacobian | Jᵢⱼ = ∂xᵢ/∂zⱼ |
| 03 | The determinant | box-and-cut-corners proof of ad − bc; det 2, ½, 0 (crushed flat → not invertible); sand in 2-D | p(x) = p(z)/\|det ∂x/∂z\| |
| 04 | The price | real warped grid; a general det costs D³ ≈ 10¹⁶ for an image; triangular → product of the diagonal, because a shear (a pushed deck of cards) never changes area | det = d₁·d₂ |
| 05 | The coupling trick | keep a, transform b; fill in the Jacobian nudge by nudge: 1, 0, ∗, e^s(a) → triangular; free inverse; stacking | log\|det\| = s(a) |
| 06 | Exact maximum likelihood | push points through the flow; on-data vs off-data (z far in the tails); real training curve | log p(x) = log N(f(x)) + Σ sₖ |
| 07 | Layer by layer | the trained 8-layer flow, forwards and inverse, with its grid | — |
| 08 | Exact density | the model's p(x) everywhere; 8 layers vs 2 | ∫p = 1 by construction |
| 09 | The catch | 8 blobs → bridges (rubber's neck thins but never breaks); no compression; shackled layers | — |
| 10 | Continuous flows | infinitely many thin layers; teaser for episode 10 | d log p/dt = −tr(∂v/∂x) |
| 11 | Build it | a coupling layer in numpy + recap | — |
| 12 | Hook → EBMs | drop the shackles: any network as an energy | p(x) = e^−E(x) / Z → *Episode 07: energy-based models* |

## Series roadmap

| # | Episode | Question it answers | Hook into the next |
|---|---------|--------------------|--------------------|
| 01 | Generative modelling | what is p(x)? | … |
| 02 | Autoregressive | break p(x) into next-piece guesses | can we learn by compressing? |
| 03 | Autoencoders & VAEs | compress to a latent, sample from it | how do we grade a generator? |
| 04 | Evaluation | how to measure "good" — and how metrics lie | what if the judge could teach? |
| 05 | GANs | learn by being judged | can a generator run backwards? |
| 06 | Normalizing flows | exact likelihood via invertible warps | flows are rigid — what if we skip normalising altogether? |
| 07 | Energy-based models | any function can be an (unnormalised) density | the partition function is intractable — can we avoid it? |
| 08 | Score matching & Langevin | learn ∇ log p instead; sample by noisy hill-climbing | scores are bad far from data — add noise at many scales… |
| 09 | Diffusion | denoise step by step (ties episodes 3, 7, 8 together) | 1,000 steps is slow — straighter paths? |
| 10 | Flow matching | learn a velocity field; straight paths; flows + diffusion unified | how do we steer what gets generated? |
| 11 | Conditioning & guidance | p(x \| y), classifier & classifier-free guidance, latent diffusion (the VAE returns) | — the full picture of a modern text-to-image model |

## Sound

Every episode has its own soundtrack: a soft generative score plus sound effects locked to the
animation. It is synthesised from scratch in numpy (no samples), deterministically, so the HTML
player and the MP4 carry identical audio.

- **Engine** — `src/audio_lib.py`: instruments (pad, sub, music box / celesta / marimba / glass /
  kalimba leads, bell chimes, ticks, typewriter keys, dice rattle, rubber stamp, whoosh, riser,
  heartbeat pulse), reverb and mix (score ~5 dB under the effects, about −22 dBFS overall, leaving
  room for narration).
- **Sync** — cues are written on the scene clock `(scene, local time)`. Motion sounds are generated
  from the animation's own motion curve (same easing as the renderer): loudness follows speed,
  brightness/pitch follows progress. Data-driven moments read their values from the page itself
  (`src/probe.mjs`): each generated letter and dice roll, each counted pair, each particle that
  lands, each mode a GAN finds, each mode-collapse hop.
- **Player** — `src/audio_player.py` adds the ♪ button and keeps the audio locked to the animation
  clock through play / pause / scrub / chapter jumps (**M** mutes).

| Episode | Key & colour | Signature sounds |
|---|---|---|
| 01 | C lydian · celesta | data points and sampled particles sonified by position; the log-likelihood sings as the knobs turn; static for the random tries |
| 02 | A minor · music box + clock | typewriter for every letter; slot-machine settle into "hello"; dice rattle per sampled letter; the τ slider as a tone |
| 03 | E minor · glass | the latent code sings as digits melt; loss falls as a tone; a harp cascade as the decoded grid appears |
| 04 | G minor · marimba | every verdict a rubber stamp (CAUGHT / FOOLED / COPIED); digits dropping into each model; the copier's zero-distance "clink" |
| 05 | F♯ minor · kalimba + heartbeat | each mode the real GAN finds chimes; each collapse hop jumps; the GDA spiral pans as it circles |
| 06 | D dorian · music box | the flow's 8 layer pulses each way; det 2 / ½ / 0 morphs; chains breaking in the hook |

```bash
python src/ep01_audio.py   # … ep06_audio.py  → epNN-audio.mp3 next to the episode HTML
python src/audio_player.py generative-modelling.html ep01-audio.mp3   # (ep01/ep02 pages; builds do this themselves)
```

## Watch / scrub

Open any episode `.html` in a browser: play/pause (space), scrub, ←/→ to skip 5 s, or
jump by chapter. ♪ (or **M**) toggles the sound.

## Render the MP4s

Needs Node, `playwright` (Chromium) and `ffmpeg`.

```bash
cd video
./fetch-fonts.sh                               # cache the Google Fonts locally (once)
WORKERS=4 node render.mjs generative-modelling.html 30 generative-modelling.mp4
WORKERS=4 node render.mjs ep02-autoregressive.html 30 ep02-autoregressive.mp4
WORKERS=4 node render.mjs ep03-autoencoders-vae.html 30 ep03-autoencoders-vae.mp4
WORKERS=4 node render.mjs ep04-evaluation.html 30 ep04-evaluation.mp4
WORKERS=4 node render.mjs ep05-gans.html 30 ep05-gans.mp4
WORKERS=4 node render.mjs ep06-flows.html 30 ep06-flows.mp4
# add the soundtrack (same for every episode)
ffmpeg -i ep06-flows.mp4 -i ep06-audio.mp3 -c:v copy -c:a aac -b:a 192k -shortest ep06-flows-sound.mp4
```

`node render.mjs <episode.html> <fps> <out.mp4> <start-s> <end-s>` renders one section
while iterating. Set `FFMPEG=/path/to/ffmpeg` if ffmpeg isn't on your PATH
(`pip install imageio-ffmpeg` ships one).
