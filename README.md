# Generative Modelling — a first-principles season

Animated explainers that build generative models from scratch, one idea at a time: how machines turn random
noise into text, digits and images. Every episode shows the maths, runs real models (trained in plain numpy with
hand-written backprop), has a synthesised soundtrack locked to the picture, and ends on the question the next one
answers.

**Start here:** open [`video/index.html`](video/index.html) in a browser. It is the season hub: every episode in order,
its playground and its code, and it remembers what you have watched.

| # | Episode | The question | Playground | Code |
|---|---------|--------------|------------|------|
| 01 | What is generative modelling? | what does it mean to learn p(x)? | `ep01-density-lab` | `density.py` |
| 02 | Autoregressive generation | can a machine write one piece at a time? | `ep02-ar-playground` | `ar_counting.py`, `ar_neural.py` |
| 03 | Autoencoders & VAEs | can a machine learn by compressing? | `ep03-vae-playground` | `vae.py` |
| 04 | Grading the imagination | how do you grade a machine that makes new things? | `ep04-metric-lab` | `evals.py` |
| 05 | GANs | what if the judge could teach? | `ep05-gan-arena` | `gan.py` |
| 06 | Normalizing flows | can a generator run backwards? | `ep06-flow-lab` | `flow.py` |
| 07 | Energy-based models | what if any network could be a density? | `ep07-ebm-lab` | `ebm.py` |
| 08 | Score matching & Langevin | can we learn just the arrows? | `ep08-score-lab` | `score.py` |
| 09 | Diffusion models | destroy it slowly, learn to undo it | `ep09-diffusion-lab` | `diffusion.py` |
| 10 | Flow matching | straight lines, fewer steps | `ep10-flow-lab` | `flow_matching.py` |
| 11 | Guidance | how do we steer what gets generated? | — | — |

## What's where

```
video/
  index.html            the season hub (posters/ holds its title cards)
  episodes/             every episode page, with its data (epNN-assets.js) and soundtrack (epNN-audio.mp3)
  labs/                 the playgrounds, one per episode, sharing lab-kit.js / lab-kit.css
  src/engine/           the animation engine shared by all episodes (page, drawing helpers, player)
  src/scenes/           one file per episode: its scenes, captions and on-screen maths
  src/audio/            one soundtrack script per episode, on a shared synthesiser (audio_lib.py)
  src/build.py          builds the pages · src/make.py builds, scores, renders and previews an episode
  render.mjs            renders any page to MP4, frame by frame → video/renders/ (not in git)
code/
  nn.py                 the shared plumbing: Adam, three network types, toy data, gradient checks
  density.py … diffusion.py   one file per model, every gradient written out by hand
  export/               trains the models shown in each episode and writes their data for the pages
```

Details: [`video/README.md`](video/README.md) (scenes of every episode, how the pages and soundtracks are built,
how to add an episode) and [`code/README.md`](code/README.md) (every model and the numbers it reproduces).

```bash
cd video && python src/build.py                        # rebuild every page
python src/make.py 9 --render --preview                # page, soundtrack, 1080p MP4 with sound, 720p preview
cd ../code && python diffusion.py --check              # any model: verify its hand-written gradients
```
