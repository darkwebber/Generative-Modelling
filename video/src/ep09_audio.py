"""Episode 09 soundtrack — E♭ major, 'noise, run backwards'. Celesta lead over warm pads.

Signature: music emerging from noise. In the cold open (and again when 64 digits are born) a celesta
motif starts as random pitches — noise — and every note is pulled toward the true melody by exactly
as much as the picture has been denoised, so the tune crystallises with the image. TV-static hiss
follows the noise level √(1 − ᾱ_t) wherever the picture shows noise; the title lands on a boom and a
full E♭ chord.

    python src/ep09_audio.py      # → ep09-audio.mp3
"""
import json
import os
import re
import sys

import numpy as np

from audio_lib import HERE, ROOT, Score, ease, eout, prog

sys.path.insert(0, os.path.join(ROOT, '..', 'code'))
durs = [float(d) for d in re.findall(r"SC\.push\(\{ dur: ([\d.]+)", open(os.path.join(HERE, 'ep09-scenes.js')).read())]
A = json.loads(re.search(r'window\.EP9 = (\{.*\});', open(os.path.join(ROOT, 'ep09-assets.js')).read(), re.S).group(1))
S = Score('ep09-audio', durs, seed=9)
ABAR = np.array(A['abar'])                                       # every 10th step
abar = lambda t: np.where(np.asarray(t) < 0, 1.0, ABAR[np.clip(np.round(np.asarray(t) / 10).astype(int), 0, len(ABAR) - 1)])
noise_of_stage = lambda ts, f: np.sqrt(1 - abar(np.array(ts)[np.clip(np.round(f).astype(int), 0, len(ts) - 1)]))
MOTIF = [63, 67, 70, 75, 74, 70, 72, 67, 68, 72, 75, 79, 77, 75, 74, 70]


def emerging_melody(sc, t0, t1, level, gain=1.0, dt=0.36, lead='celesta', seed=0):
    """Notes of MOTIF whose pitch is pulled off by noise × level(t): chaos → the tune, as the picture denoises."""
    rng = np.random.default_rng(seed); t, k = t0, 0
    while t < t1:
        lv = float(level(np.array([t]))[0]); m = MOTIF[k % len(MOTIF)] + int(round(rng.standard_normal() * 9 * lv))
        S.cue(sc, t, 'note', gain * (0.55 + 0.45 * (1 - lv)), float(rng.uniform(-0.6, 0.6)) * lv, m=int(np.clip(m, 48, 96)), lead=lead)
        t += dt * (1 + 0.6 * lv * rng.uniform(-0.5, 0.5)); k += 1


CH = {'Ebmaj9': ([51, 55, 58, 62, 65], 39), 'Cm9': ([48, 55, 58, 62, 63], 36), 'Abmaj7': ([44, 51, 55, 60, 63], 32), 'Bbsus': ([46, 53, 58, 60, 65], 34),
      'Fm9': ([53, 56, 60, 63, 67], 41), 'Gm7': ([55, 58, 62, 65, 67], 43), 'Dbmaj7#11': ([49, 56, 60, 65, 67], 37), 'Eb': ([51, 58, 63, 67, 70], 39)}
S.score(CH, [
    ([(0, 'Cm9'), (9, 'Abmaj7'), (15, 'Bbsus'), (21, 'Eb'), (27, 'Ebmaj9')], 0.0, .9),    # 00 intro (lead replaced by the emerging melody)
    ([(0, 'Ebmaj9'), (16, 'Cm9'), (32, 'Abmaj7'), (48, 'Bbsus')], 0.7, .8),               # 01 forward
    ([(0, 'Fm9'), (16, 'Dbmaj7#11'), (24, 'Ebmaj9'), (40, 'Bbsus')], 0.7, .8),            # 02 reverse
    ([(0, 'Cm9'), (16, 'Abmaj7'), (32, 'Ebmaj9'), (47, 'Bbsus')], 0.9, .9),               # 03 training
    ([(0, 'Abmaj7'), (24, 'Fm9'), (40, 'Ebmaj9')], 0.8, .9),                              # 04 sampling
    ([(0, 'Cm9'), (8, 'Abmaj7'), (14, 'Bbsus'), (20.2, 'Eb'), (28, 'Ebmaj9')], 0.4, 1.0),  # 05 digits
    ([(0, 'Gm7'), (16, 'Cm9'), (32, 'Abmaj7'), (40, 'Bbsus')], 0.7, .8),                  # 06 why it works
    ([(0, 'Ebmaj9'), (16, 'Fm9'), (24, 'Abmaj7'), (40, 'Bbsus')], 0.9, .9),               # 07 ddim
    ([(0, 'Ebmaj9'), (18, 'Cm9'), (27, 'Abmaj7'), (36, 'Eb')], 1.0, 1.0),                 # 08 in the wild
    ([(0, 'Abmaj7'), (18, 'Ebmaj9')], 1.3, 1.0),                                          # 09 build it
    ([(0, 'Cm9'), (8, 'Fm9'), (16, 'Dbmaj7#11'), (24, 'Bbsus'), (38.5, 'Ebmaj9')], 0.5, .7),  # 10 next
], [63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82], lead='celesta')
S.transitions()
c, m = S.cue, S.motion

# 00 cold open — static, a melody emerging from noise, a riser, the bang
TS = A['title']['ts']; NT = len(TS) - 1
fT = lambda t: (NT) * np.where(t < 9, 0, np.power(ease(prog(t, 9, 12)), 0.9))
lvl0 = lambda t: np.clip(noise_of_stage(TS, fT(t)), 0, 1)
S.hiss(0, 0.2, 21.2, lambda t: np.minimum(prog(t, 0.2, 1.2), 1) * (0.35 + 0.65 * lvl0(t)), 2.2, lo=1800, hi=8000)
for k in range(40): c(0, 0.3 + k * 0.22 + 0.1 * np.sin(k * 7), 'tick', 0.12, float(np.sin(k * 3)) * 0.8, f=2500 + 900 * abs(np.sin(k)))   # crackle
emerging_melody(0, 9.2, 20.8, lvl0, 1.2, seed=1)
c(0, 13, 'riser', 0.8, dur=8.0, m0=39, m1=63); c(0, 18.6, 'whoosh', 1.3, dur=2.4, f0=250, f1=6000)
for k in range(12): c(0, 15 + k * (6 / 12) * (1 - k / 30), 'thud', 0.18 + 0.03 * k, f=55)   # accelerating heartbeat
c(0, 21, 'thud', 1.3, f=36); c(0, 21, 'thud', 0.7, f=52)
for mi, g_ in ((51, 1.0), (58, 0.85), (63, 0.9), (67, 0.8), (70, 0.75), (75, 0.7), (82, 0.55)): c(0, 21.0 + 0.012 * (mi % 5), 'chime', g_, (mi - 66) / 30, m=mi, bright=1.3)
S.hiss(0, 21, 24, lambda t: 0.5 * np.exp(-(t - 21) * 1.6), 1.2, lo=3000, hi=9000)
for i in range(5): c(0, 23.5 + i * 0.3, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 110 * i)

# 01 forward — hiss grows with the noise; the signal tone fades with √ᾱ
tF = lambda t: 999 * ease(prog(t, 2, 20))
S.hiss(1, 2, 23, lambda t: np.sqrt(1 - abar(tF(t))), 1.2, pan=-0.4)
S.track(1, 2, 23, lambda t: 63 + 0 * t, amp=lambda t: np.sqrt(abar(tF(t))), gain=1.2, pan=-0.4)
c(1, 8, 'tick'); c(1, 16, 'tick'); c(1, 24, 'chime', 0.9, m=70); c(1, 32, 'tick')
for i, ts in enumerate((0, 100, 250, 400, 600, 999)):
    c(1, 40.3 + i * 0.3, 'note', 0.7, -0.5 + 0.2 * i, m=75 - 2 * i, lead='celesta'); S.hiss(1, 40.3 + i * 0.3, 40.6 + i * 0.3, lambda t, ts=ts: np.sqrt(1 - abar(ts)) + 0 * t, 0.8, pan=-0.5 + 0.2 * i)

# 02 reverse — one bell (consonant) vs two answers (dissonant)
c(2, 1, 'tick'); c(2, 10, 'chime', 1.1, -0.4, m=70); c(2, 8, 'note', 0.8, -0.4, m=63, lead='celesta')
c(2, 16, 'note', 0.8, 0.4, m=62, lead='celesta'); c(2, 18, 'note', 0.9, 0.4, m=61, lead='celesta'); c(2, 18.05, 'note', 0.9, 0.5, m=66, lead='celesta'); c(2, 18.2, 'thunk', 0.6, 0.4)
c(2, 24, 'chime', 1.0, m=75); c(2, 32, 'tick'); c(2, 40, 'chime', 0.8, m=79)

# 03 training — for each cycle: an image, noise, the dial spins to t, the network answers
for cyc in range(8):
    t0 = 1 + cyc * 7
    if t0 > 53: break
    c(3, t0, 'note', 0.6, -0.6, m=72, lead='celesta')
    c(3, t0 + 0.2, 'rattle', 0.5, 0.4, dur=0.8)
    S.hiss(3, t0 + 0.1, t0 + 0.9, lambda t: 0.6 + 0 * t, 0.5, pan=-0.1)
c(3, 8, 'tick'); c(3, 16, 'chime', 0.9, m=74); c(3, 24, 'chime', 0.9, m=77); c(3, 32, 'chime', 0.9, m=82); c(3, 40, 'tick'); c(3, 47, 'tick')

# 04 sampling — hiss falls with the noise level; particles roll; the guesses commit
RT = A['ring']['ts']; NR = len(RT) - 1; RX = np.array(A['ring']['x'])
fR = lambda t: NR * ease(prog(t, 4, 38))
S.hiss(4, 3.5, 42.5, lambda t: np.clip(noise_of_stage(RT, fR(t)), 0, 1), 1.0, pan=-0.4)
step = np.linalg.norm(RX[1:] - RX[:-1], axis=2).mean(1)
S.roll(4, 4, 42.2, lambda t: step[np.minimum(fR(t).astype(int), NR - 1)] * np.abs(np.gradient(fR(t), t)), 0.9, -0.4, seed=4)
c(4, 1, 'tick'); c(4, 8, 'tick'); c(4, 16, 'tick'); c(4, 24, 'chime', 0.9, 0.4, m=75)
for i in range(3): c(4, 40 + i, 'note', 1.0, 0.3, m=[70, 75, 82][i], lead='celesta')

# 05 THE WOW — 64 static squares; the same melody emerges from noise; the grid lands on a chord
DT = A['digits']['ts']; ND = len(DT) - 1
fD = lambda t: ND * np.power(ease(prog(t, 4, 16)), 0.85)
lvl5 = lambda t: np.clip(noise_of_stage(DT, fD(t)), 0, 1)
S.hiss(5, 0.5, 20.4, lambda t: np.minimum(prog(t, 0.5, 1), 1) * (0.3 + 0.7 * lvl5(t)), 2.0, lo=1800, hi=8000)
emerging_melody(5, 4.3, 20.1, lvl5, 1.1, dt=0.32, seed=5)
c(5, 12, 'riser', 0.7, dur=8.0, m0=43, m1=63); c(5, 18.2, 'whoosh', 1.0, dur=2.0, f0=300, f1=5000)
c(5, 20.2, 'thud', 1.0, f=40)
for mi, g_ in ((51, 0.85), (58, 0.75), (63, 0.8), (67, 0.7), (70, 0.6), (75, 0.6), (79, 0.5)): c(5, 20.2 + 0.01 * (mi % 4), 'chime', g_, (mi - 66) / 30, m=mi, bright=1.2)
for j in range(8): c(5, 28.3 + j * 0.25, 'note', 0.5, -0.6 + 0.17 * j, m=[63, 65, 67, 70, 72, 75, 79, 82][j], lead='celesta')
c(5, 36, 'tick'); c(5, 43, 'chime', 0.9, m=75)

# 06 why it works — forward (rose) and backward (sage) sweeps, typed ELBO
for j in range(6): c(6, 1 + j * 0.3, 'note', 0.4, -0.6 + 0.24 * j, m=79 - 2 * j, lead='celesta')
c(6, 8, 'whoosh', 0.8, dur=1.4, f0=400, f1=3500); c(6, 10, 'whoosh', 0.8, dur=1.4, f0=3500, f1=400)
c(6, 16, 'chime', 0.8, m=63)
for tt, n in ((24, 10), (32, 10), (40, 5)):
    for i in range(n): c(6, tt + i * 0.06, 'key', 0.35, 0.2 * np.sin(i), f=2500 + 180 * (i % 3))

# 07 DDIM — four grids pop in; then the morph sings its noise mix
c(7, 8, 'tick')
for j, t0 in enumerate((1, 16, 18, 20)):
    for q in range(4): c(7, t0 + q * 0.05, 'clack', 0.35, -0.6 + 0.4 * j, f=2600 + 200 * q)
c(7, 20.3, 'thunk', 0.5, 0.6)
for q in range(16): c(7, 24.5 + q * 0.12, 'note', 0.35, -0.7 + q * 0.09, m=63 + (q % 8) * 2, lead='celesta')
S.track(7, 32, 48, lambda t: 67 + 8 * (0.5 + 0.5 * np.sin((t - 32) * 0.9)), gain=0.9, pan=0.2)

# 08 in the wild — four cards
for i, (tt, mm) in enumerate(((1, 70), (9, 75), (18, 79), (27, 82))): c(8, tt, 'chime', 0.9, -0.45 + 0.3 * i, m=mm)
c(8, 36, 'chime', 1.1, m=87)

# 09 build it
for i in range(15): c(9, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): c(9, 18.5 + i * 1.6, 'tick', 0.8, f=1400)

# 10 hook — the curved roads draw themselves; the straight lines arrive; title
m(10, 1, 8.1, lambda t: ease(prog(t, 1, 7)), 0.7, lo=250, hi=1600, tone=(51, 63))
for i in range(10): c(10, 16 + i * 0.1, 'note', 0.35, -0.6 + 0.13 * i, m=63 + 2 * i, lead='celesta')
c(10, 8, 'tick'); c(10, 24, 'chime', 0.9, m=77)
c(10, 38.5, 'thud', 0.8, f=45); c(10, 38.5, 'chime', 1.0, m=63); c(10, 38.55, 'chime', 0.85, m=70); c(10, 39.3, 'chime', 1.0, m=75); c(10, 40.5, 'tick'); c(10, 41, 'chime', 1.0, m=87)

S.render()
