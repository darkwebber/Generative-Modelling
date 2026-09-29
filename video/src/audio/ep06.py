"""Episode 06 soundtrack — D dorian, 'bending space, exactly'. Music-box lead; every motion sound is
driven by the flow's own layer-by-layer motion (8 pulses per pass).

    python src/audio/ep06.py      # → ep06-audio.mp3
"""

import numpy as np

from audio_lib import Score, ease, prog, scene_durations, scene_index

durs = scene_durations(6)
S = Score('ep06-audio', durs, seed=6)
K = 8  # coupling layers in the trained flow


def stage(f):  # drawPts interpolates layer by layer, easing inside each layer → visible pulses
    f = np.clip(f, 0, K); i = np.minimum(np.floor(f), K - 1); return i + ease(f - i)


CH = {'Dm9': ([50, 57, 60, 64, 65], 38), 'Bbmaj7': ([46, 53, 57, 62, 65], 34), 'Fmaj9': ([53, 57, 60, 64, 67], 41),
      'Gm9': ([55, 58, 62, 65, 69], 43), 'Am7': ([57, 60, 64, 67, 72], 45), 'Cadd9': ([48, 55, 62, 64, 67], 36),
      'A7sus': ([57, 62, 64, 67, 71], 45), 'Ebmaj7#11': ([51, 58, 62, 67, 69], 39), 'Dm': ([50, 57, 62, 65, 69], 38)}
S.score(CH, [
    ([(0, 'Dm9'), (18.8, 'Bbmaj7')], 0.6, .8), ([(0, 'Dm9'), (30, 'Fmaj9')], 0.7, .7), ([(0, 'Bbmaj7'), (30, 'Fmaj9')], 0.9, .8),
    ([(0, 'Gm9'), (31, 'A7sus'), (37.5, 'Dm9')], 0.8, .7), ([(0, 'Am7'), (23, 'Fmaj9')], 0.6, .6), ([(0, 'Fmaj9'), (36.5, 'Cadd9')], 1.1, 1.0),
    ([(0, 'Dm9'), (32, 'Fmaj9')], 0.8, .8), ([(0, 'Bbmaj7'), (24, 'Fmaj9')], 1.0, .9), ([(0, 'Fmaj9')], 0.8, .9),
    ([(0, 'Gm9'), (16, 'A7sus'), (32, 'Dm9')], 0.6, .6), ([(0, 'Ebmaj7#11'), (24, 'Bbmaj7')], 0.9, 1.0), ([(0, 'Fmaj9'), (18, 'Cadd9')], 1.4, 1.0),
    ([(0, 'Dm'), (8, 'Bbmaj7'), (23, 'Gm9'), (31, 'A7sus'), (38.5, 'Dm9')], 0.5, .6),
], [62, 64, 65, 67, 69, 72, 74, 76, 77, 79, 81], lead='musicbox')
S.transitions()
c, m = S.cue, S.motion

# 00 cold open — noise → data, back, and forward again (8 layer pulses each way)
m(0, 0.5, 5.6, lambda t: stage(K * ease(prog(t, 0.5, 5))), tone=(50, 62))
m(0, 6.5, 11.6, lambda t: -stage(K - K * ease(prog(t, 6.5, 5))), tone=(62, 50))
m(0, 12, 15.1, lambda t: stage(K * ease(prog(t, 12, 3))), 0.9, tone=(50, 62))
c(0, 15, 'note', m=81); c(0, 18.8, 'chime', 1.4, m=74); c(0, 18.8, 'chime', 0.9, m=81)
for i in range(5): c(0, 20 + i * 0.3, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 120 * i)
# 01 sand — sound only marks what appears
c(1, 8, 'note', 0.8, m=69); c(1, 15, 'tick'); c(1, 22, 'note', m=76); c(1, 30, 'tick'); c(1, 32, 'chime', m=79); c(1, 38.5, 'tick')
# 02 jacobian — the grid warps, then the zoom
m(2, 2, 6.1, lambda t: ease(prog(t, 2, 4)), 0.8, lo=200, hi=1400)
c(2, 8, 'tick'); m(2, 9, 14.1, lambda t: ease(prog(t, 9, 5)), 0.7, lo=500, hi=4000)
c(2, 15, 'note', 1.2, -0.3, m=81); c(2, 23, 'note', 1.2, 0.3, m=84); c(2, 30, 'chime', m=77)
for i in range(4): c(2, 32 + i * 0.15, 'tick', 0.6, f=1600 + 100 * i)
c(2, 38.5, 'tick'); c(2, 40, 'tick')
# 03 determinant — cut-away pieces, then each matrix morph (det 2, ½, 0, back)
c(3, 7, 'tick')
for i in range(6): c(3, 9 + i * 0.8, 'tick', 0.8, -0.5 + 0.2 * i, f=1200 + 90 * i)
c(3, 17, 'chime', 1.2, m=81)
for (ta, tb, up) in [(20, 22, 1), (26, 27.5, -1), (30, 31.5, -1), (36, 37.5, 1)]:
    m(3, ta, tb + 0.05, lambda t, ta=ta, tb=tb, up=up: up * ease(prog(t, ta, tb - ta)), 0.8, lo=300, hi=1800, tone=(57, 64) if up > 0 else (64, 55))
c(3, 31.5, 'thud', 0.9); c(3, 37.5, 'tick'); c(3, 43, 'chime', m=74)
# 04 the cost — shear (deck of cards) and stretch
c(4, 3, 'tick', pan=-0.4); c(4, 3.3, 'tick', pan=-0.3); c(4, 8, 'tick'); c(4, 15.5, 'thud', 0.8, f=48)
c(4, 23, 'tick'); c(4, 25, 'chime', m=79)
m(4, 32, 35.1, lambda t: ease(prog(t, 32, 3)), 0.8, lo=600, hi=1600)
m(4, 39, 41.6, lambda t: ease(prog(t, 39, 2.5)), 0.8, lo=300, hi=900, tone=(60, 67))
# 05 coupling — one layer bends forth, back, forth; later four layers stack
m(5, 16, 27.6, lambda t: ease(prog(t, 16, 3.5)) - ease(prog(t, 20, 3.5)) + ease(prog(t, 24, 3.5)), 0.8, lo=300, hi=1500)
c(5, 8, 'tick')
for tt, f in [(26, 1300), (28.5, 1200), (34, 1500), (35.5, 1600)]: c(5, tt, 'tick', f=f)
c(5, 36.5, 'chime', 1.4, m=86, bright=1.3); c(5, 37, 'tick'); c(5, 40, 'tick'); c(5, 47, 'tick')
m(5, 48, 55.1, lambda t: stage(1 + 3 * ease(prog(t, 48, 7))), 0.9, tone=(50, 62))
# 06 likelihood — points pushed through the flow (twice)
c(6, 8, 'tick'); m(6, 9, 15.1, lambda t: -stage(K * ease(prog(t, 9, 6))), tone=(62, 50))
c(6, 16, 'tick', pan=-0.3); c(6, 17.5, 'tick', pan=0.3); c(6, 24, 'tick'); c(6, 32, 'chime', m=81)
m(6, 40.5, 45.6, lambda t: -stage(K * ease(prog(t, 40.5, 5))), 0.8, tone=(62, 50))
# 07 layers — forwards (left panel), then backwards (right panel), each layer a pulse
m(7, 5, 21.1, lambda t: -stage(K * ease(prog(t, 5, 16))), pan=-0.35, tone=(62, 50))
m(7, 25, 39.1, lambda t: stage(K * ease(prog(t, 25, 14))), pan=0.35, tone=(50, 62))
# 08 density
c(8, 1, 'chime', 0.8, m=74); c(8, 8, 'chime', m=86); c(8, 16, 'whoosh', 0.8); c(8, 24, 'tick'); c(8, 24.2, 'tick')
# 09 the catch — the rubber neck stretches thin
c(9, 4, 'tick'); c(9, 10, 'note', m=69)
m(9, 17, 22.1, lambda t: ease(prog(t, 17, 5)), 0.9, lo=150, hi=700, tone=(45, 52))
c(9, 32, 'thud', 0.6, f=70); c(9, 40, 'clink', 0.8)
# 10 continuous — the flow runs smoothly (cosine clock, layer pulses)
m(10, 2, 18.1, lambda t: stage(K * (0.5 - 0.5 * np.cos(np.pi * prog(t, 2, 16)))), 0.8, tone=(50, 62))
c(10, 8, 'tick'); c(10, 16, 'tick'); c(10, 31, 'chime', m=79)
# 11 build it — keyboard clicks as code lines appear
for i in range(16): c(11, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): c(11, 18.5 + i * 1.6, 'tick', 0.8, f=1400)
# 12 hook — chains appear, then fall away; balls roll into the valleys
for i in range(3): c(12, 1.5 + i * 0.8, 'clink', 0.8, -0.3 + 0.3 * i, f=1900 + 150 * i)
for i in range(6): c(12, 7.6 + i * 0.09, 'clink', 0.5, S.rng.uniform(-0.7, 0.7), f=S.rng.uniform(1600, 2600))
c(12, 7.5, 'whoosh', 1.2, dur=1.4, f0=3000, f1=250)
m(12, 15, 22.6, lambda t: -sum(ease(np.clip((t - 15 - k * 0.3) / 6, 0, 1)) for k in range(6)), 0.6, lo=120, hi=500)
c(12, 23, 'chime', m=74)
c(12, 38.5, 'thud', 1.1, f=45); c(12, 38.5, 'chime', 1.5, m=62); c(12, 38.55, 'chime', 1.2, m=69); c(12, 40, 'tick'); c(12, 40.5, 'chime', 1.0, m=86)

# the season strip on the end card, the playground pill in the build-it scene
S.season_strip(scene_index(6, 'next'), 41.2, 6)
S.play_pill(scene_index(6, 'build it'), 9)

S.render()
