# Generative Modelling — a motion-graphic series

First-principles animated explainers. Each episode is a single deterministic `<canvas>`
animation (seeded, time-driven), so every frame can be re-rendered exactly. Visual style:
the *Midnight Atelier* system — warm darks, film grain, and colours with fixed meanings
(terracotta = data/tokens, ink = distributions, amber = parameters/knobs, rose =
noise/dice, plum = models, sage = insight).

**Start here: open `index.html`**. It is the season hub, with every episode in order, its playground and its code,
and it remembers what you have watched. Pages live in `episodes/`, playgrounds in `labs/`, sources in `src/`.

Every episode ends the same way: the question it leaves open, **NEXT · EPISODE NN**, and a strip of the whole season
with the episodes watched so far lit and the next one pulsing. Every build-it scene names its playground.

## Episode 01 — What is generative modelling? (~8:42)

`episodes/ep01-generative-modelling.html` · playground: `labs/ep01-density-lab.html` · code: `../code/density.py`

The cold open is a trailer for the season made only from its real models (`ep01-assets.js`, collected by
`code/export/export_ep01_assets.py` from episodes 02–09): names typed by the episode-02 counting model, the episode-03
VAE walking its latent space, the episode-05 GAN learning digits epoch by epoch, the episode-06 flow folding noise
onto the moons, episode-07 marbles rolling into energy valleys and episode-09 digits denoised out of static.

| # | Scene | Math introduced |
|---|-------|-----------------|
| 00 | Cold open: six real machines tune in out of static; a bang on the title | the colour language |
| 01 | Data is points (2-pixel image → ℝ², digit → ℝ⁷⁸⁴) | x ∈ ℝᵈ |
| 02 | Discriminative: learn the border | p(y \| x) |
| 03 | Generative: learn where data lives, then sample | p(x), Bayes' rule |
| 04 | What is a density? histogram → curve | ∫p = 1, P(a≤x≤b) = ∫ₐᵇ p |
| 05 | Maximum likelihood with a live score | θ* = argmax Σ log p_θ(xᵢ) |
| 06 | Sampling: bend noise through g | x = g(z), p(x) = p(z)\|dz/dx\| |
| 07 | The catch: random pixels are static | manifolds |
| 08 | The season map: the route from episode 01 to 11, each stop with its idea and equation | each family's core equation |
| 09 | Big picture: 1,400 points flow onto data | z → g_θ(z) ∼ p_θ ≈ p_data |
| 10 | Build it: a mixture of bell curves by EM, and sampling it (`code/density.py`) | Σₖ πₖ N(x; μₖ, σₖ²) |
| 11 | Recap | — |
| 12 | Hook → autoregressive: a sentence types itself as the chain rule grows | → *Episode 02* |

## Episode 02 — Autoregressive generation (~8:47)

`episodes/ep02-autoregressive.html` · playground: `labs/ep02-ar-playground.html` · code: `../code/`

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

`episodes/ep03-autoencoders-vae.html` · playground: `labs/ep03-vae-playground.html` · code: `../code/vae.py`

Built from `src/scenes/ep03.js` + the shared engine: `python src/build.py 3`. Every image,
latent map, sample and interpolation is decoded live in the browser from the **real trained
networks** (`ep03-assets.js`, exported by `code/export/export_ep03_assets.py`).

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

`episodes/ep04-evaluation.html` · playground: `labs/ep04-metric-lab.html` · code: `../code/evals.py`

Built from `src/scenes/ep04.js` (`python src/build.py 4`). Every score on screen comes from
`ep04-assets.js`, computed by `code/export/export_ep04_assets.py` on real samples.

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

`episodes/ep05-gans.html` · playground: `labs/ep05-gan-arena.html` · code: `../code/gan.py`

Built from `src/scenes/ep05.js` (`python src/build.py 5`). The 2-D scenes replay real
training runs frame by frame (samples, the judge's field and its gradients), and the digit
scenes use a real MNIST GAN — all trained by `code/export/export_ep05_assets.py`.

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

`episodes/ep06-flows.html` · playground: `labs/ep06-flow-lab.html` · code: `../code/flow.py`

Built from `src/scenes/ep06.js` (`python src/build.py 6`). Every flow on screen is a real
RealNVP-style flow trained by exact maximum likelihood in `code/export/export_ep06_assets.py`:
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

## Episode 07 — Energy-based models (~10:50)

`episodes/ep07-energy.html` · playground: `labs/ep07-ebm-lab.html` · code: `../code/ebm.py`

Built from `src/scenes/ep07.js` (`python src/build.py 7`). Every landscape, marble and number on
screen comes from a real energy-based model trained by contrastive divergence in
`code/export/export_ep07_assets.py`: its energy map at seven points in training, the Langevin and
gradient-descent marble paths, a 30,000-step mixing chain, each valley's share of probability,
and the exact 2-D likelihood (Z summed over a 400 × 400 grid). The 1-D scenes use a real
25-bump model sculpted from 200 samples, and a real Metropolis chain.

The spine: **energy → e^−E → Z is impossible → its gradient is an average over fantasies → so
we must sample → ratios (Metropolis) → slopes (Langevin) → the slope never needed Z (the score).**

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | marbles dropped on a sculpted landscape roll, jiggle and settle into the data | — |
| 01 | One number per point | every family paid for p(x) with a promise; an EBM is any network → one number; probe the real landscape | E_θ : ℝᴰ → ℝ |
| 02 | Energy → probability | three demands (positive, downhill = likely, energies add ⇒ probabilities multiply) force the exponential; softmax was an EBM | p = e^−E / Z, Z = ∫ e^−E |
| 03 | The price tag | terms in Z: 27 letters, a 400² grid, 256⁷⁸⁴ ≈ 10¹⁸⁸⁸ images; but ratios cancel Z | p(a)/p(b) = e^{E(b)−E(a)} |
| 04 | Push down, pull up | ∇ log Z slides into the integral and becomes pθ — an average over the model's own samples | ∇θ(−log p) = ∇θE(x) − E_{pθ}[∇θE(x′)] |
| 05 | Sculpting (1-D, real) | data digs, fantasies raise; the gap closes; the hidden landscape is recovered | — |
| 06 | Metropolis | propose, flip a biased coin; histogram → e^−E/Z; steps shrink like 1/√D | accept w.p. min(1, e^−ΔE) |
| 07 | Use the slope | the same 420 marbles: gradient descent collapses, Langevin spreads into the valleys | x ← x − η∇E + √(2η)ε |
| 08 | Training in 2-D (real) | landscape snapshots with data and fantasies; mean energies chase each other | contrastive divergence, replay buffer, α·E² |
| 09 | The result | EBM density vs episode 6's flow (no bridges); exact held-out NLL vs flow and perfect | — |
| 10 | The catch | valleys hold unequal probability; one marble crosses few ridges in 30k steps; cost, unknown Z | mixing ~ e^ΔE |
| 11 | In disguise | classifiers & JEM, Hopfield/Boltzmann (Nobel 2024), contrastive learning, next-token softmax | — |
| 12 | Build it | Langevin + training loop in numpy + recap | — |
| 13 | Hook → score | the sampler only used the slope; ∇ₓ log Z = 0; the arrows are the score | s(x) = ∇ₓ log p = −∇ₓE → *Episode 08: score matching & Langevin* |

## Episode 08 — Score matching & Langevin dynamics (~10:12)

`episodes/ep08-score.html` · playground: `labs/ep08-score-lab.html` · code: `../code/score.py`

Built from `src/scenes/ep08.js` (`python src/build.py 8`). The 2-D scenes use real noise-conditional
score networks trained by denoising score matching in `code/export/export_ep08_assets.py`. The data are mixtures of
Gaussians, so every learned arrow is checked against the exact score of the noisy data.

The spine: **the score (no Z) → why Langevin's √(2η) is exact → we can't see the true score → integrate by parts
(Hyvärinen) → add noise and point home (Vincent; the score is a denoiser) → small noise is blind in the desert and to
weights → many noise levels + annealed Langevin → run noise backwards: diffusion.**

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | particles follow a learned arrow field down a ladder of noise levels onto the eight blobs | — |
| 01 | The score | log p, its slope as arrows, a relative slope, Z drops out, a Gaussian is a spring | s(x) = ∇ log p = p′/p |
| 02 | Why √(2η) | three Langevin runs on N(0,1): no kick collapses, double kick is 4× too wide; variance recursion; Fokker–Planck balance | Var′ = (1−η)²Var + 2η; ∂p/∂t = −∇·(p s) + Δp = 0 |
| 03 | The problem | we have samples, not scores; Fisher divergence has an unknown term | E‖s_θ − ∇log p‖² |
| 04 | Integrate by parts | the unknown cancels; live fit of a Gaussian score recovers mean and variance; D backward passes | E[s_θ² + 2 s_θ′] |
| 05 | Point home | shake x, the way home is −ε/σ; the average of all possible homes is the noisy score; Tweedie | E‖σ s_θ + ε‖²; E[x∣x̃] = x̃ + σ² s |
| 06 | Training (real) | the arrow field at σ ≈ 0.5 organising over 6,000 steps; loss levels off above 0 | — |
| 07 | The catch | error map at σ = 0.05 (wrong between the blobs); 80/20 blobs: plain Langevin gives 50% (after 3,000 steps 49%) | local arrows can't see weights |
| 08 | Many scales | the learned field from σ = 3 to 0.05, annealed Langevin down the ladder | η = c·σ² |
| 09 | Weights restored | 80/20: truth 80%, plain 50%, annealed 77% | — |
| 10 | In the wild | NCSN (2019), score SDEs (2021), denoisers as priors, ε-prediction in diffusion models | — |
| 11 | Build it | DSM training + annealed Langevin in 15 lines + recap | — |
| 12 | Hook → diffusion | data dissolves into noise, arrows lead it back; destroy slowly, learn to undo | → *Episode 09: diffusion* |

## Episode 09 — Diffusion models (~9:06)

`episodes/ep09-diffusion.html` · playground: `labs/ep09-diffusion-lab.html` · code: `../code/diffusion.py`

Built from `src/scenes/ep09.js` (`python src/build.py 9`). Everything generated on screen comes from real diffusion
models trained in `code/export/export_ep09_assets.py`, the title included: a 2-D model trained on the pixels of the word
"Diffusion" writes it out of 5,000 points of static in the cold open.

The spine: **destroy data slowly (and jump to any step in one line) → tiny steps are easy to undo, giant ones aren't →
predict the noise (episode 8's score, one net for all levels) → run it backwards → 64 digits from nothing, and what
the net believes along the way → why: a VAE with a thousand layers → DDIM: bigger steps, noise as a latent → the roads
are curved: flow matching.**

| # | Scene | Idea | Math |
|---|-------|------|------|
| 00 | Cold open | 5,000 points of static; a real diffusion model writes the title out of them; the melody emerges from noise with it | — |
| 01 | Destroy it slowly | the eight blobs and a digit dissolve over 1,000 steps; ᾱ curve; variance balance | x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε |
| 02 | Undo one small step | the true one-step posterior in 1-D: one narrow bell for a tiny step, two humps for a giant one | x_{t−1} ≈ (x_t + β_t·s)/√(1−β_t) + √β_t·z |
| 03 | Training | x₀ + ε → x_t → ε_θ, a dial for t, the real loss curve | ‖ε − ε_θ(x_t, t)‖², ε_θ = −√(1−ᾱ)·s_θ, x̂₀ |
| 04 | Running it backwards | 600 points from N(0, I) to the eight blobs, beside the net's guess x̂₀ at every step | the DDPM step |
| 05 | Sixty-four digits | 64 MNIST digits born from static; x̂₀ over time for eight of them: composition first, detail last; then episode 4's judges on 2,000 samples: FID 9.8 (GAN 21.3, VAE 51.9), recall 89% (real data 88%), learned judge 61% | x = g(z), g = 1,000 denoising steps |
| 06 | A VAE with a thousand layers | the noising chain as encoder, the ELBO as one KL per step | Σ KL → Σ w_t‖ε − ε_θ‖² |
| 07 | Fewer steps: DDIM | 1000 / 100 / 20 / 5 steps from the same noise; slerp between two noise images morphs the digit | x_s = √ᾱ_s·x̂₀ + √(1−ᾱ_s)·ε_θ |
| 08 | In the wild | DALL·E 2, Imagen, Stable Diffusion (latent), conditioning, video / audio / weather / proteins | — |
| 09 | Build it | forward, training and sampling in 15 lines + recap | — |
| 10 | Hook → flow matching | the curved roads DDIM took vs straight lines from noise to data | → *Episode 10: flow matching* |

## How it's built

```
video/
  src/engine/shell.html   the page: canvas, play/scrub/chapters, ♪ button, <audio>
  src/engine/core.js      Midnight Atelier tokens (P colours, F fonts), maths & easing, drawing helpers,
                          M('[d|x] + [a_|θ]') colour-coded maths, cards, labels, tags, pills, insights
  src/engine/kit.js       helpers several episodes share: plotSquare/frame/clipTo, marbles, 28×28 images
                          (u8imgs/imgCanvas/drawImg), rubber stamps, gaussian, memo, camera shake
  src/engine/player.js    timeline, scene header, captions, progress bar, grain & vignette, player,
                          audio sync, window.renderAt / DURATION / READY for the renderer
  src/scenes/epNN.js      one episode: its data + SC.push({ dur, chapter, title, sub, caps, draw(t) })
  src/build.py            the episode list; inlines the above into one self-contained episodes/epNN-….html
  src/audio/epNN.py       its soundtrack, on top of src/audio/audio_lib.py → episodes/epNN-audio.mp3
  src/tools/              probe.mjs (read values out of a page for the soundtrack), glyph.mjs (text → points)
  src/make.py             build → audio → render → add sound → 720p preview, in one command
  src/new_episode.py      scaffolds the next episode (scenes, soundtrack, build entry)
  episodes/               every page, with its data (epNN-assets.js) and soundtrack (epNN-audio.mp3): open or publish
  labs/                   the playgrounds (epNN-….html) + lab-kit.js / .css (guide, missions, tactile sound, styles)
  index.html, posters/    the season hub
  render.mjs              renders any page to MP4, frame by frame (deterministic) → renders/ (not in git)
```

Scenes draw in a fixed 1920×1080 frame: the header sits above y ≈ 200, content in y 240–930, captions at
y ≈ 972–1014. Captions are plain text (write ₀ ₜ θ, not `_`); on-screen maths goes through `M()`.
Everything is seeded and time-driven, so any frame can be re-rendered exactly.

### Adding an episode

```bash
python src/new_episode.py 10 ep10-flow-matching.html "Flow Matching"   # scenes + soundtrack skeletons, registered
# … write src/scenes/ep10.js; export real-model data to episodes/ep10-assets.js (window.EP10) and list it in build.py
python src/make.py 10                      # page + soundtrack
python src/make.py 10 --render --preview   # + renders/ep10-…-sound.mp4 and a ~25 MB 720p preview
```

## Playgrounds

Every episode has a hands-on lab in `labs/`, named after its episode (`ep01-density-lab.html` … `ep09-diffusion-lab.html`). They share `lab-kit.js` and `lab-kit.css`:

- a three-step **how to play** guide and a colour legend at the top of each lab;
- **missions** that check themselves off as you play (each is one idea from the episode, with a hint), a progress pill
  that follows you down the page, and a small celebration when you finish;
- **tactile sound**, synthesised live with WebAudio: slider detents pitched by value, button clicks, plus sounds for each
  lab's own events (dice rattles, glass marbles, judges' stamps, chimes when a GAN finds a mode…); ♪ toggles it;
- chunkier sliders and buttons that press in, tap ripples on every canvas, and layouts that work on a phone.

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

- **Engine** — `src/audio/audio_lib.py`: instruments (pad, sub, music box / celesta / marimba / glass /
  kalimba leads, bell chimes, ticks, typewriter keys, dice rattle, rubber stamp, whoosh, riser,
  heartbeat pulse), reverb and mix (score ~5 dB under the effects, about −22 dBFS overall, leaving
  room for narration).
- **Sync** — cues are written on the scene clock `(scene, local time)`. Motion sounds are generated
  from the animation's own motion curve (same easing as the renderer): loudness follows speed,
  brightness/pitch follows progress. Data-driven moments read their values from the page itself
  (`src/tools/probe.mjs`): each generated letter and dice roll, each counted pair, each particle that
  lands, each mode a GAN finds, each mode-collapse hop.
- **Player** — the shared page (`src/engine/player.js`) has the ♪ button and keeps the audio locked to the
  animation clock through play / pause / scrub / chapter jumps (**M** mutes).

| Episode | Key & colour | Signature sounds |
|---|---|---|
| 01 | C lydian · celesta | a trailer that tunes in out of static, each real machine heard doing its own thing (keys for typed names, clacks per GAN epoch, rolling marbles, thinning hiss), a heartbeat, a riser and a bang on the title; data points sonified by position; the log-likelihood sings as the knobs turn; a bell per stop on the season map; a typewriter into episode 02 |
| 02 | A minor · music box + clock | typewriter for every letter; slot-machine settle into "hello"; dice rattle per sampled letter; the τ slider as a tone |
| 03 | E minor · glass | the latent code sings as digits melt; loss falls as a tone; a harp cascade as the decoded grid appears |
| 04 | G minor · marimba | every verdict a rubber stamp (CAUGHT / FOOLED / COPIED); digits dropping into each model; the copier's zero-distance "clink" |
| 05 | F♯ minor · kalimba + heartbeat | each mode the real GAN finds chimes; each collapse hop jumps; the GDA spiral pans as it circles |
| 06 | D dorian · music box | the flow's 8 layer pulses each way; det 2 / ½ / 0 morphs; chains breaking in the hook |
| 07 | C minor · vibraphone | glass marbles poured, rolling and jiggling at their real speed; a chisel tap per sculpting step (as loud as the landscape moved); coin, wood-block (accepted) or thunk (rejected) per Metropolis proposal; a chime each time the long chain hops a ridge |
| 08 | B♭ lydian · plucked harp | noise you can hear: a hiss that follows σ, a harp note per rung of the σ ladder, particles rolling at their real speed, the three √(2η) runs each singing their variance, the probe singing its score |
| 09 | E♭ major · celesta | the melody itself diffuses: in the cold open and the 64-digit reveal it starts as random notes and snaps into tune as the noise falls (its detuning follows √(1−ᾱ_t)); TV static that thins with the noise; a heartbeat, a riser and a bang as the title lands; a tock per judge's bar and stamps on diffusion's wins |

```bash
python src/audio/ep01.py   # … ep09.py  → episodes/epNN-audio.mp3 next to the page (or: python src/make.py N)
```

## Watch / scrub

Open `index.html`, or any page in `episodes/`, in a browser: play/pause (space), scrub, ←/→ to skip 5 s, or
jump by chapter. ♪ (or **M**) toggles the sound.

## Render the MP4s

Needs Node, `playwright` (Chromium) and `ffmpeg`.

```bash
cd video
./fetch-fonts.sh                                  # cache the Google Fonts locally (once)
WORKERS=4 python src/make.py 9 --render --preview # build, soundtrack, render, add sound, 720p preview
# or by hand, for any page:
WORKERS=4 node render.mjs episodes/ep09-diffusion.html 30 renders/ep09-diffusion.mp4
ffmpeg -i renders/ep09-diffusion.mp4 -i episodes/ep09-audio.mp3 -c:v copy -c:a aac -b:a 192k -shortest renders/ep09-diffusion-sound.mp4
```

`node render.mjs episodes/<page>.html <fps> <out.mp4> <start-s> <end-s>` renders one section
while iterating. Set `FFMPEG=/path/to/ffmpeg` if ffmpeg isn't on your PATH
(`pip install imageio-ffmpeg` ships one).
