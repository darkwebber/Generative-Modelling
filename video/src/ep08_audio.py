"""Episode 08 soundtrack — B♭ lydian, 'follow the arrows'. A plucked-harp lead over airy pads.

Signature: noise you can hear. Wherever the picture shows a noise level σ, a soft hiss follows it
(loud at σ = 3, a whisper at σ = 0.05), and every rung of the σ ladder is a harp note, rising as the
picture sharpens (country → city → street). Particles following the arrows roll at their real speed
(from the exported annealed-Langevin paths); the three Langevin runs of scene 2 each sing their variance.

    python src/ep08_audio.py      # → ep08-audio.mp3
"""
import json
import os
import re

import numpy as np

from audio_lib import HERE, ROOT, Score, ease, eout, prog

durs = [float(d) for d in re.findall(r"SC\.push\(\{ dur: ([\d.]+)", open(os.path.join(HERE, 'ep08-scenes.js')).read())]
A = json.loads(re.search(r'window\.EP8 = (\{.*\});', open(os.path.join(ROOT, 'ep08-assets.js')).read(), re.S).group(1))
S = Score('ep08-audio', durs, seed=8)
ANN = np.array(A['ann']); NS = len(ANN) - 1; SIG = np.array(A['sigmas']); PER = A['T'] // A['keep']
LADDER = [70, 72, 74, 76, 77, 79, 81, 82, 84, 86]          # one harp note per σ level, rising as σ shrinks


def speed(traj, fmap, t):
    f = np.clip(fmap(t), 0, len(traj) - 1); i = np.minimum(np.floor(f).astype(int), len(traj) - 2)
    step = np.linalg.norm(traj[1:] - traj[:-1], axis=2).mean(1); return step[i] * np.abs(np.gradient(f, t))


def crossings(fmap, t0, t1, per):
    """times at which the stage index crosses each multiple of `per` (a new σ level)."""
    ts = np.linspace(t0, t1, 20000); f = fmap(ts); out = []
    for k in range(1, len(SIG)):
        i = np.searchsorted(f, k * per)
        if i < len(ts): out.append((k, float(ts[i])))
    return out


CH = {'Bbmaj9': ([46, 53, 57, 60, 62], 34), 'C69': ([48, 55, 57, 62, 64], 36), 'Gm9': ([55, 58, 62, 65, 69], 43), 'Ebmaj9': ([51, 55, 58, 62, 65], 39),
      'Fsus': ([53, 58, 60, 65, 67], 41), 'Dm7': ([50, 57, 60, 62, 65], 38), 'Abmaj7#11': ([44, 51, 55, 60, 62], 32), 'Bb69': ([46, 53, 55, 60, 62], 34)}
S.score(CH, [
    ([(0, 'Bbmaj9'), (18.8, 'C69')], 0.5, .7),                                          # 00 intro
    ([(0, 'Bbmaj9'), (15, 'Gm9'), (29, 'Ebmaj9'), (43, 'C69')], 0.8, .8),              # 01 score
    ([(0, 'Dm7'), (15, 'Gm9'), (24, 'Ebmaj9'), (40, 'Fsus'), (48, 'Bbmaj9')], 0.7, .8),  # 02 balance
    ([(0, 'Abmaj7#11'), (16, 'Ebmaj9'), (24, 'C69')], 0.6, .7),                         # 03 the problem
    ([(0, 'Gm9'), (16, 'Dm7'), (24, 'Ebmaj9'), (42, 'Abmaj7#11')], 0.8, .8),           # 04 hyvärinen
    ([(0, 'Bbmaj9'), (23, 'Gm9'), (39, 'Ebmaj9'), (47, 'C69')], 0.9, .9),              # 05 denoising
    ([(0, 'Dm7'), (16, 'Bbmaj9'), (32, 'Fsus')], 1.0, .9),                              # 06 training
    ([(0, 'Abmaj7#11'), (24, 'Gm9'), (40, 'Dm7')], 0.6, .7),                            # 07 catch
    ([(0, 'Ebmaj9'), (15, 'C69'), (30, 'Bbmaj9'), (46, 'Fsus')], 0.9, 1.0),            # 08 many scales
    ([(0, 'Bbmaj9'), (16, 'Ebmaj9'), (31, 'C69')], 1.1, 1.0),                          # 09 result
    ([(0, 'Gm9'), (18, 'Ebmaj9'), (27, 'C69'), (34, 'Bbmaj9')], 1.0, 1.0),             # 10 in the wild
    ([(0, 'Bb69'), (18, 'Ebmaj9')], 1.3, 1.0),                                          # 11 build it
    ([(0, 'Dm7'), (8, 'Gm9'), (23, 'Abmaj7#11'), (31, 'Fsus'), (38.5, 'Bbmaj9')], 0.5, .7),  # 12 next
], [58, 60, 62, 64, 65, 67, 69, 70, 72, 74, 76, 77], lead='harp')
S.transitions()
c, m = S.cue, S.motion
harp = lambda sc, t, mi, g=1.0, pan=0.0: c(sc, t, 'note', g, pan, m=mi, lead='harp')


def sweep(sc, t0, n=12, dur=1.4, lo=0, gain=0.6, up=True):
    notes = [58, 62, 65, 69, 72, 74, 77, 81, 84, 86, 89, 93][lo:lo + n]
    for i, mi in enumerate(notes if up else notes[::-1]): harp(sc, t0 + i * dur / n, mi, gain, -0.7 + 1.4 * i / max(1, n - 1))


def ladder(sc, fmap, t0, t1, gain=0.9):
    for k, tk in crossings(fmap, t0, t1, PER): harp(sc, tk, LADDER[k], gain, -0.5 + k / 9)


# 00 cold open — arrows sweep in; particles follow them down the σ ladder
sweep(0, 0.5, 12, 1.8)
f0 = lambda t: NS * prog(t, 3.5, 13.5)
harp(0, 3.5, LADDER[0], 1.0); ladder(0, f0, 3.5, 17.1)
S.roll(0, 3.5, 17.2, lambda t: speed(ANN, f0, t), 1.0, seed=1)
S.hiss(0, 3.5, 18.3, lambda t: SIG[np.clip((f0(t) // PER).astype(int), 0, 9)] / 3, 1.0)
c(0, 18.8, 'thud', 1.1, f=44); c(0, 18.8, 'chime', 1.3, m=70); c(0, 18.85, 'chime', 0.9, m=77)
for i in range(6): c(0, 20 + i * 0.3, 'tick', 0.7, -0.6 + 0.24 * i, f=1300 + 110 * i)

# 01 the score — the probe sings its score value
harp(1, 1, 65); c(1, 5, 'whoosh', 0.6, dur=0.9, f0=300, f1=1800); c(1, 8, 'tick'); c(1, 8.5, 'tick', 0.7)
m(1, 9, 14.1, lambda t: ease(prog(t, 9, 5)), 0.6, lo=300, hi=1200)
MIX = [(0.65, -1.3, 0.55), (0.35, 1.5, 0.45)]
N1 = lambda x, mu, s: np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
s1 = lambda x: sum(w * N1(x, mu, s) * (-(x - mu) / s ** 2) for w, mu, s in MIX) / sum(w * N1(x, mu, s) for w, mu, s in MIX)
xp = lambda t: np.where(t < 15, -2.6 + 1.9 * ease(prog(t, 9, 5)), -0.7 + 2.9 * np.sin((t - 15) * 0.22))
S.track(1, 9, 49, lambda t: 67 + 7 * np.tanh(s1(xp(t)) / 3), gain=1.0, pan=0.2)
for i in range(21): harp(1, 15 + i * 0.08, 62 + int(round(6 * np.tanh(s1(-4 + i * 0.4) / 4))), 0.45, -0.8 + 1.6 * i / 20)
c(1, 22, 'tick'); c(1, 29, 'chime', 0.9, m=74); c(1, 36, 'note', 0.8, m=62, lead='harp'); c(1, 43, 'tick')

# 02 balance — three Langevin runs; each run's variance sings (collapse falls, √2η holds, double rises)
rg = np.random.default_rng(81)
for ci, pan in ((0, -0.5), (1, 0.0), (2, 0.5)):
    v0 = 3.0                                       # uniform(−3.2, 3.2) start ≈ variance 3.4
    var = lambda t, ci=ci: np.maximum(1e-3, (ci ** 2 * 2 / (2 - 0.05)) + (3.4 - ci ** 2 * 2 / 1.95) * (0.95 ** 2) ** (200 * np.clip((t - 9) / 14, 0, 1) ** 0.8))
    S.track(2, 8.5, 31, lambda t, var=var: 64 + 5 * np.log2(np.maximum(var(t), 0.1)), gain=0.8, pan=pan)
for k in range(30): c(2, 9 + k * 0.45, 'tick', 0.18, 0, f=1800)
c(2, 23.2, 'thunk', 0.6, -0.5); c(2, 23.4, 'chime', 1.1, 0, m=70); c(2, 23.6, 'thunk', 0.6, 0.5)
c(2, 24, 'tick'); c(2, 26, 'chime', 0.8, m=77); c(2, 32, 'whoosh', 0.5, dur=0.8); c(2, 36, 'tick'); c(2, 40, 'chime', 1.0, m=74); c(2, 44, 'tick'); c(2, 48, 'chime', 1.2, m=82)

# 03 the problem
c(3, 0.8, 'tick'); S.cue(3, 1, 'chime', 0.5, m=65)
for i in range(16): c(3, 9 + i * 0.05, 'clack', 0.25, -0.8 + 0.1 * i, f=2600 + 60 * i)
c(3, 9.2, 'thunk', 0.6); c(3, 24, 'chime', 0.9, -0.3, m=72); c(3, 27, 'chime', 0.9, 0.3, m=76)

# 04 integrate by parts — typed rows, a strike, then the live fit descends to rest
for tt, n in ((1, 9), (8, 8), (16, 8), (24, 7)):
    for i in range(n): c(4, tt + i * 0.07, 'key', 0.35, 0.25 * np.sin(i), f=2500 + 180 * (i % 3))
c(4, 9.5, 'scratch', 1.1); c(4, 18, 'tick'); c(4, 25, 'chime', 1.1, m=77)
S.track(4, 32, 41.5, lambda t: 72 - 10 * ease(prog(t, 33, 8)), gain=0.9, pan=0.4)
for k in range(24): c(4, 33 + 8 * (k / 24) ** 1.3, 'tick', 0.25, 0.4, f=1500 + 30 * k)
c(4, 41.2, 'chime', 1.0, 0.4, m=70); c(4, 42, 'thunk', 0.6)

# 05 denoising — shakes, the ways home, the fan of possible origins, the average, the denoised guess
for i in range(14): c(5, 2 + i * 0.12, 'rattle', 0.25, -0.6 + 0.09 * i, dur=0.35)
for i in range(14): harp(5, 8 + i * 0.1, 70 + (i % 5) * 2, 0.4, -0.6 + 0.09 * i)
c(5, 1, 'tick'); c(5, 15, 'tick'); sweep(5, 30.5, 10, 1.2, lo=1, gain=0.4); c(5, 34, 'chime', 1.2, m=65, bright=1.2)
c(5, 39, 'tick'); c(5, 47, 'tick'); c(5, 47.5, 'chime', 1.3, m=82); c(5, 47.55, 'chime', 0.8, m=89)

# 06 training — the field reorganises (loudness = how much the arrows actually change); loss descends
SN = np.array(A['snaps']); chg = np.abs(np.diff(SN, axis=0)).mean((1, 2)); chg /= chg.max()
trf = lambda t: (len(SN) - 1) * ease(prog(t, 4, 26))
m(6, 4, 30.1, lambda t: trf(t), 0.8, lo=250, hi=1800)
for k in range(len(SIG)): harp(6, 8.3 + k * 0.08, LADDER[k], 0.35, -0.6 + k * 0.13)
H = np.array(A['hist']); S.track(6, 16, 31, lambda t: 60 + 14 * np.interp(np.clip((t - 16) / 14, 0, 1) * (len(H) - 1), np.arange(len(H)), (H - H.min()) / (H.max() - H.min())), gain=0.7, pan=0.3)
c(6, 1, 'tick'); c(6, 24, 'tick'); c(6, 40, 'chime', 1.0, m=77)

# 07 catch — the error map swells in; the 80/20 split lands wrong
c(7, 1.5, 'riser', 0.4, dur=1.4, m0=40, m1=47); c(7, 8, 'tick')
for i in range(12): c(7, 24.5 + i * 0.06, 'clack', 0.3, -0.5 + 0.05 * i, f=2400 + 90 * i)
for i in range(12): c(7, 31.5 + i * 0.06, 'clack', 0.3, -0.5 + 0.05 * i, f=2000 + 90 * i)
c(7, 24, 'tock', 0.9, f=1500); c(7, 32, 'thunk', 0.9); c(7, 36, 'thunk', 0.8); c(7, 40, 'chime', 0.8, m=65)

# 08 many scales — the ladder rises as σ shrinks; hiss follows σ; particles roll down the ladder
fl8 = lambda t: np.where(t < 22, 9 * ease(prog(t, 8, 13)), np.clip(NS * np.clip((t - 22) / 30, 0, 1) / PER, 0, 9))
S.hiss(8, 7.5, 53, lambda t: SIG[np.clip(np.round(fl8(t)).astype(int), 0, 9)] / 3, 1.1, pan=-0.3)
ts = np.linspace(8, 21.2, 8000); lv = np.round(9 * ease(prog(ts, 8, 13))).astype(int)
for k in range(1, 10):
    i = np.searchsorted(lv, k)
    if i < len(ts): harp(8, float(ts[i]), LADDER[k], 0.7, -0.4)
f8 = lambda t: NS * np.clip((t - 22) / 30, 0, 1)
harp(8, 22, LADDER[0], 0.9); ladder(8, f8, 22, 52.1, 1.0)
S.roll(8, 22, 52.5, lambda t: speed(ANN, f8, t), 1.0, -0.3, seed=2)
c(8, 1, 'tick'); c(8, 8, 'tick', 0.7); c(8, 22, 'tick'); c(8, 30, 'tick', 0.7); c(8, 46, 'chime', 1.1, m=82)

# 09 result — rose patter (plain), sage patter (annealed), the three shares
for i in range(14): c(9, 0.2 + i * 0.04, 'clack', 0.3, -0.6 + 0.04 * i, f=2100 + 80 * i)
for i in range(24): c(9, 1.6 + i * 0.1, 'clack', 0.3, 0.6 - 0.02 * i, f=2800 + 60 * (i % 7))
c(9, 2.5, 'tock', 0.9, f=1300); c(9, 3.2, 'thunk', 0.8); c(9, 3.9, 'chime', 1.3, m=77); c(9, 3.95, 'chime', 0.9, m=84)

# 10 in the wild — four cards, then the arrows win
for i, (tt, mm) in enumerate(((1, 70), (9, 74), (18, 77), (27, 81))): c(10, tt, 'chime', 0.9, -0.45 + 0.3 * i, m=mm)
sweep(10, 34, 12, 1.6, gain=0.55)

# 11 build it — keys, recap ticks
for i in range(15): c(11, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): c(11, 18.5 + i * 1.6, 'tick', 0.8, f=1400)

# 12 hook — data dissolves into noise (hiss rises), then the arrows lead it back (hiss falls)
S.hiss(12, 1.5, 16, lambda t: np.power(np.clip((t - 2) / 12, 0, 1), 1.6), 1.3, pan=-0.3)
m(12, 2, 14.1, lambda t: np.power(prog(t, 2, 12), 1.6), 0.5, lo=200, hi=900, tone=(62, 50))
f12 = lambda t: NS * np.clip((t - 17) / 13, 0, 1)
S.hiss(12, 16, 31, lambda t: SIG[np.clip((f12(t) // PER).astype(int), 0, 9)] / 3, 1.0, pan=-0.3)
harp(12, 17, LADDER[0], 0.8); ladder(12, f12, 17, 30.1, 0.9)
S.roll(12, 17, 30.5, lambda t: speed(ANN, f12, t), 0.9, -0.3, seed=3)
c(12, 8, 'tick'); c(12, 15, 'tick'); c(12, 23, 'tick')
c(12, 38.5, 'thud', 1.1, f=45); c(12, 38.5, 'chime', 1.5, m=58); c(12, 38.55, 'chime', 1.2, m=65); c(12, 39.3, 'chime', 1.0, m=70); c(12, 40.5, 'tick'); c(12, 41, 'chime', 1.0, m=82)

S.render()
