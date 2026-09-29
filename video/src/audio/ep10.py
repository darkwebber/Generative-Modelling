"""Episode 10 soundtrack — D major, 'the straight way from noise to data'. Vibraphone lead over bright, open pads.

Signature: straight lines you can hear. Wherever the picture flies along straight roads, a choir of soft voices starts
on random pitches (noise) and glides in straight lines to the notes of a chord (data): one straight segment per Euler
step, a tick for every call to the network. Four hops in the cold open are four ticks and four glides that land on the
title chord. Curved roads (round one, diffusion) are heard as glides that bend and overshoot; straight ones don't.

    python src/audio/ep10.py      # → episodes/ep10-audio.mp3   (or: python src/make.py 10)
"""
import numpy as np

from audio_lib import Score, ease, eout, prog, scene_durations, scene_index

durs = scene_durations(10)
S = Score('ep10-audio', durs, seed=10)
c, m = S.cue, S.motion
IDX = lambda ch: scene_index(10, ch)


def hopf(t, t0, per, K):
    """The picture's hop-by-hop Euler clock (scenes: hopF) → stage in [0, K]."""
    s = np.clip((np.asarray(t, np.float64) - t0) / per, 0, K); i = np.minimum(np.floor(s), K - 1); return np.where(s >= K, K, i + ease(s - i))


def glides(sc, t0, t1, targets, f, K, gain=1.0, seed=0, spread=10, bend=0.0):
    """One voice per target note: from a random 'noise' pitch to the target, straight in pitch per step of f(t) ∈ [0, K].
    bend > 0 makes the road curve: the voice overshoots sideways mid-way, as a curved road does."""
    rng = np.random.default_rng(seed)
    for j, tgt in enumerate(targets):
        m0 = tgt + rng.uniform(-spread, spread); side = rng.choice([-1, 1]) * bend
        pan = float(np.clip((tgt - np.mean(targets)) / 14, -0.8, 0.8))
        S.track(sc, t0, t1, lambda t, m0=m0, tgt=tgt, side=side: m0 + (tgt - m0) * (f(t) / K) + side * np.sin(np.pi * np.clip(f(t) / K, 0, 1)),
                amp=lambda t: 0.55 + 0.45 * np.clip(f(t) / K, 0, 1), gain=gain / np.sqrt(len(targets)) * 1.6, pan=pan, fade=0.5)


def hop_ticks(sc, t0, per, K, gain=0.8, f0=1800, pan=0.0):
    """A tick for every network call, rising a little each time."""
    for i in range(K): c(sc, t0 + i * per, 'tick', gain, pan, f=f0 + 160 * i)


CH = {'Dmaj9': ([50, 57, 61, 64, 66], 38), 'Bm9': ([47, 54, 57, 61, 62], 35), 'Gmaj7': ([43, 50, 54, 59, 62], 31), 'Asus': ([45, 52, 57, 59, 64], 33),
      'Em9': ([52, 55, 59, 62, 66], 40), 'F#m7': ([54, 57, 61, 64, 66], 42), 'Cmaj7#11': ([48, 55, 59, 64, 66], 36), 'D': ([50, 57, 62, 66, 69], 38)}
HOP0, HOP, TK = 8.4, 1.55, 4; BANG = HOP0 + TK * HOP + 0.25
S.score(CH, [
    ([(0, 'Bm9'), (8.4, 'Gmaj7'), (11.5, 'Asus'), (BANG, 'D'), (26, 'Dmaj9')], 0.0, .9),              # 00 intro (the glides are the lead)
    ([(0, 'Dmaj9'), (16, 'Bm9'), (24, 'Gmaj7'), (32, 'Em9'), (40, 'Asus')], 0.7, .8),                 # 01 motion
    ([(0, 'Dmaj9'), (8, 'F#m7'), (16, 'Gmaj7'), (24, 'Dmaj9'), (32, 'Cmaj7#11'), (40, 'Asus')], 0.7, .8),   # 02 straight roads
    ([(0, 'Bm9'), (8, 'Cmaj7#11'), (16, 'Em9'), (24, 'Gmaj7'), (32, 'Asus'), (40, 'Dmaj9'), (47, 'D')], 0.6, .8),  # 03 crossings
    ([(0, 'Gmaj7'), (16, 'Dmaj9'), (32, 'Bm9'), (40, 'Em9'), (47, 'D')], 0.7, .9),                  # 04 the wind
    ([(0, 'Dmaj9'), (16, 'Bm9'), (24, 'Gmaj7'), (32, 'Asus'), (40, 'D')], 0.9, .9),                  # 05 the loss
    ([(0, 'Em9'), (16, 'Gmaj7'), (24, 'Asus'), (32, 'Dmaj9')], 0.8, .9),                             # 06 training
    ([(0, 'Bm9'), (4, 'Gmaj7'), (10, 'Asus'), (16, 'D'), (25, 'Dmaj9'), (33, 'Gmaj7')], 0.4, 1.0),    # 07 digits
    ([(0, 'Dmaj9'), (16, 'F#m7'), (24, 'Gmaj7'), (32, 'Em9'), (40, 'Asus'), (48, 'Bm9')], 0.8, .9),  # 08 steps
    ([(0, 'Bm9'), (8, 'Gmaj7'), (16, 'Asus'), (24, 'Dmaj9'), (32, 'Cmaj7#11'), (33.5, 'D'), (40, 'Dmaj9'), (48, 'Gmaj7')], 0.7, 1.0),  # 09 reflow
    ([(0, 'Em9'), (8, 'Cmaj7#11'), (16, 'Gmaj7'), (24, 'Asus'), (32, 'Dmaj9')], 0.8, .9),           # 10 one family
    ([(0, 'Dmaj9'), (9, 'Bm9'), (18, 'Gmaj7'), (27, 'Asus'), (34, 'D')], 1.0, 1.0),                  # 11 in the wild
    ([(0, 'Gmaj7'), (18, 'Dmaj9')], 1.3, 1.0),                                                       # 12 build it
    ([(0, 'Bm9'), (8, 'Em9'), (16, 'Gmaj7'), (24, 'Cmaj7#11'), (32, 'Asus'), (38.5, 'Dmaj9')], 0.5, .7),   # 13 next
], [62, 64, 66, 69, 71, 73, 74, 76, 78, 81, 83, 86], lead='vibes')
S.transitions()

# 00 cold open — static; four straight hops, a tick per network call, voices gliding straight onto the title chord; the bang
S.hiss(0, 0.2, HOP0 + 0.4, lambda t: np.minimum(prog(t, 0.2, 1.2), 1) * (0.9 - 0.4 * prog(t, 6, 2.4)), 1.8, lo=1800, hi=8000)
for k in range(28): c(0, 0.4 + k * 0.27 + 0.1 * np.sin(k * 7), 'tick', 0.1, float(np.sin(k * 3)) * 0.8, f=2600 + 800 * abs(np.sin(k)))    # crackle
rng = np.random.default_rng(3)
for k in range(10): c(0, 1.2 + k * 0.62, 'note', 0.28, float(rng.uniform(-0.7, 0.7)), m=int(rng.integers(60, 90)), lead='glass')              # stray, random pitches
f0 = lambda t: hopf(t, HOP0, HOP, TK)
glides(0, HOP0 - 0.6, BANG + 0.2, [50, 57, 62, 66, 69, 74, 78, 81], f0, TK, gain=1.4, seed=1, spread=11)
hop_ticks(0, HOP0, HOP, TK, gain=1.1, f0=1500)
for i in range(TK): c(0, HOP0 + i * HOP, 'tock', 0.7, -0.45 + 0.3 * i, f=900 + 180 * i); c(0, HOP0 + i * HOP, 'whoosh', 0.55, -0.3 + 0.2 * i, dur=HOP * 0.9, f0=400 + 150 * i, f1=2600 + 300 * i)
dF0 = lambda t: np.abs(np.gradient(f0(t), t))
S.roll(0, HOP0, BANG, dF0, 0.8, 0.0, lo=240, hi=2200, seed=0)
c(0, BANG - 4.2, 'riser', 0.6, dur=4.2, m0=45, m1=62)
c(0, BANG, 'thud', 1.3, f=36); c(0, BANG, 'thud', 0.7, f=50)
for mi, g_ in ((50, 1.0), (57, 0.85), (62, 0.9), (66, 0.8), (69, 0.75), (74, 0.7), (81, 0.55)): c(0, BANG + 0.012 * (mi % 5), 'chime', g_, (mi - 66) / 30, m=mi, bright=1.3)
S.hiss(0, BANG, BANG + 3, lambda t: 0.45 * np.exp(-(t - BANG) * 1.6), 1.2, lo=3000, hi=9000)
for i in range(5): c(0, BANG + 2.2 + i * 0.3, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 110 * i)

# 01 motion — particles drift on the wind; Euler's steps; a curved road flies off, a straight one lands
sc = IDX('motion')
S.roll(sc, 9, 23, lambda t: np.abs(np.gradient(ease(prog(t, 9, 14)), t)), 0.5, -0.4, lo=200, hi=1300, seed=1)
c(sc, 1, 'chime', 0.7, 0.3, m=74); c(sc, 8, 'whoosh', 0.6, -0.4, dur=1.5, f0=300, f1=1800); c(sc, 16, 'chime', 0.8, 0.3, m=78); c(sc, 24, 'tick', 0.8, 0.3)
for K, t0, base in ((1, 32.5, 1700), (2, 33.5, 1900), (4, 34.5, 2200)):
    for i in range(K): c(sc, t0 + i * 0.12, 'clack', 0.5, 0.2 + 0.1 * i, f=base + 250 * i)
c(sc, 32.8, 'thunk', 0.4, 0.3); c(sc, 36.6, 'tock', 0.8, 0.5, f=1400); c(sc, 36.8, 'chime', 0.9, 0.5, m=81)

# 02 the simplest road — a straight road sounds like a straight glide
sc = IDX('roads')
c(sc, 3, 'clack', 0.6, -0.5, f=2300); c(sc, 3.3, 'clack', 0.6, -0.2, f=3100)
S.track(sc, 9, 15.2, lambda t: 62 + 12 * (0.5 - 0.5 * np.cos(np.pi * np.clip((t - 9) / 6, 0, 1))), gain=1.3, pan=-0.4)
c(sc, 8, 'tick'); c(sc, 16, 'chime', 0.9, -0.3, m=74)
for k in range(4): S.track(sc, 24 + k * 4, 27.8 + k * 4, lambda t, k=k: 62 + 12 * ((t - 24 - 4 * k) / 4), gain=0.8, pan=-0.4)
c(sc, 32, 'chime', 0.8, 0.4, m=69)
S.roll(sc, 33, 39, lambda t: prog(t, 33, 0.5) * (1 - prog(t, 37, 2)), 0.9, -0.4, lo=300, hi=2600, seed=2)
c(sc, 40, 'scratch', 0.7, -0.4, dur=0.5)

# 03 crossings — the fan: one tick per arrow, then the question; the average lands, and the network agrees
sc = IDX('crossings')
S.roll(sc, 1, 7, lambda t: np.abs(np.gradient(0.5 * ease(prog(t, 1, 6)), t)), 0.6, -0.4, seed=3)
c(sc, 8, 'whoosh', 0.7, -0.3, dur=2.2, f0=1800, f1=400)
for i in range(40): c(sc, 9.5 + i * 0.04 * 1.5, 'tick', 0.18, float(np.sin(i * 2.3)) * 0.6, f=1400 + 60 * (i % 12))
c(sc, 16, 'note', 0.7, 0.4, m=73, lead='vibes'); c(sc, 16.08, 'note', 0.7, 0.4, m=74, lead='vibes')       # a rub: which one?
c(sc, 24, 'tick'); c(sc, 32, 'tick')
c(sc, 34, 'chime', 1.1, 0.4, m=74); c(sc, 34.05, 'chime', 0.8, 0.4, m=81)
c(sc, 41, 'chime', 0.9, 0.5, m=74)                                                                             # the same note: nearly the same arrow
c(sc, 47, 'chime', 0.8, m=78)

# 04 the averaged wind — two crowds in step; crossings scratched in; zero crossings rings clean
sc = IDX('the wind')
tt4 = lambda t: np.where(t < 32, (1 - np.cos(np.pi * np.clip(np.where(((t - 2) / 12) % 2 > 1, 2 - ((t - 2) / 12) % 2, ((t - 2) / 12) % 2), 0, 1))) / 2, ease(prog(t, 32.5, 7)))
S.roll(sc, 2, 31, lambda t: np.abs(np.gradient(tt4(t), t)), 0.5, -0.45, seed=4); S.roll(sc, 2, 31, lambda t: np.abs(np.gradient(tt4(t), t)), 0.5, 0.45, seed=5)
S.track(sc, 32.5, 39.5, lambda t: 62 + 12 * ease(prog(t, 32.5, 7)), gain=0.9, pan=0.45)
for i in range(8): c(sc, 36 + i * 0.09, 'scratch', 0.25, -0.5, dur=0.18)
c(sc, 36.9, 'chime', 0.9, 0.5, m=81); c(sc, 16, 'tick'); c(sc, 24, 'chime', 0.8, m=74); c(sc, 47, 'chime', 0.9, m=78)

# 05 the loss — for each example: two marbles, a glide to the point, the target, the answer, the miss
sc = IDX('the loss')
for cyc in range(7):
    t0 = 1 + cyc * 6.5
    if t0 > 42: break
    c(sc, t0 + 0.1, 'clack', 0.45, -0.6, f=2200); c(sc, t0 + 0.2, 'clack', 0.45, -0.3, f=3000)
    S.track(sc, t0 + 1.0, t0 + 2.6, lambda t, t0=t0: 64 + 7 * ease(prog(t, t0 + 1.0, 1.6)), gain=0.6, pan=-0.4)
    c(sc, t0 + 2.6, 'tock', 0.6, -0.4, f=1250); c(sc, t0 + 3.9, 'thunk', 0.35, -0.4, f=170); c(sc, t0 + 4.0, 'tick', 0.35, -0.4, f=2600)
c(sc, 16, 'chime', 0.8, 0.4, m=74); c(sc, 24, 'tick'); c(sc, 27, 'tick'); c(sc, 32, 'chime', 0.9, 0.4, m=81)

# 06 training — the wind sharpens (a slowly brightening hiss), particles flow, the bars rise
sc = IDX('training')
S.hiss(sc, 9, 30, lambda t: 0.25 + 0.35 * ease(prog(t, 9, 20)), 0.6, lo=900, hi=4000, pan=-0.4)
c(sc, 1, 'chime', 0.7, 0.3, m=74); c(sc, 24, 'tick'); c(sc, 26, 'tick', 0.6)
S.roll(sc, 32.5, 38.5, lambda t: np.abs(np.gradient(ease(prog(t, 32.5, 6)), t)), 0.8, -0.4, seed=6)
for b in range(8): c(sc, 38.8 + b * 0.1, 'note', 0.5, -0.6 + 0.17 * b, m=[62, 64, 66, 69, 71, 74, 76, 78][b], lead='vibes')

# 07 digits — sixty-four digits in thirty-two steps: a clockwork of 32 ticks under eight straight glides; the grid lands
sc = IDX('digits')
fD = lambda t: 32 * ease(prog(t, 4, 12))
glides(sc, 3.4, 16.4, [62, 66, 69, 74, 78, 81, 86, 90], fD, 32, gain=1.2, seed=7, spread=12)
dD = np.diff(np.floor(fD(np.linspace(4, 16, 12000))))
for i in np.nonzero(dD)[0]: c(sc, 4 + 12 * i / 12000, 'tick', 0.22, float(np.sin(i)) * 0.5, f=2200 + 25 * (i % 30))
S.hiss(sc, 0.5, 16, lambda t: np.minimum(prog(t, 0.5, 1), 1) * (1 - np.clip(fD(t) / 32, 0, 1)) ** 1.5, 1.4, lo=1800, hi=8000)
c(sc, 16, 'thud', 0.9, f=42)
for mi, g_ in ((50, 0.8), (57, 0.7), (62, 0.75), (66, 0.65), (69, 0.6), (74, 0.55), (81, 0.5)): c(sc, 16 + 0.01 * (mi % 4), 'chime', g_, (mi - 66) / 30, m=mi, bright=1.2)
for j in range(8): c(sc, 17.3 + j * 0.25, 'note', 0.45, -0.1 + 0.1 * j, m=[62, 64, 66, 69, 71, 74, 78, 81][j], lead='vibes')
c(sc, 25, 'tick'); c(sc, 33, 'chime', 0.9, m=78)

# 08 how few steps — four panels hop 1, 2, 4, 8 times; grids pop; the FID curves draw
sc = IDX('steps')
for j, K in enumerate((1, 2, 4, 8)):
    t0, per = 2 + j * 0.6, 4.5 / K
    for i in range(K): c(sc, t0 + i * per, 'tick', 0.55, -0.75 + 0.5 * j, f=1500 + 300 * j + 40 * i)
    S.roll(sc, t0, t0 + 4.6, lambda t, t0=t0, per=per, K=K: np.abs(np.gradient(hopf(t, t0, per, K), t)) / K, 0.35, -0.75 + 0.5 * j, seed=10 + j)
c(sc, 1, 'thunk', 0.4, -0.75); c(sc, 8, 'chime', 0.8, 0.3, m=74)
for j in range(5):
    for q in range(4): c(sc, 16.3 + j * 0.4 + q * 0.03, 'clack', 0.3, -0.7 + 0.35 * j, f=2600 + 200 * q)
for j in range(4): c(sc, 24 + j * 0.4, 'clack', 0.3, -0.35 + 0.35 * j, f=2000)
c(sc, 32, 'whoosh', 0.7, 0, dur=1.2, f0=500, f1=3000)
S.track(sc, 33.5, 35.5, lambda t: 76 - 10 * ease(prog(t, 33.5, 2)), gain=0.9, pan=0.3)
S.track(sc, 35, 37, lambda t: 76 - 12 * ease(prog(t, 35, 2)), gain=0.9, pan=-0.3)
c(sc, 40, 'chime', 0.9, m=78); c(sc, 48, 'note', 0.7, 0, m=64, lead='vibes')

# 09 reflow — couplings drawn; a curved glide beside a straight one; then ONE step, twice: a dull thunk, then the chord
sc = IDX('reflow')
for i in range(24): c(sc, 2 + i * 0.2, 'tick', 0.2, -0.6 + 0.05 * i, f=1800 + 40 * (i % 10))
c(sc, 8, 'chime', 0.8, 0.4, m=74)
S.track(sc, 16.2, 20, lambda t: 62 + 12 * ease(prog(t, 16.2, 3.8)) + 4 * np.sin(np.pi * prog(t, 16.2, 3.8)), gain=0.9, pan=-0.5)     # round 1: bends
S.track(sc, 17, 20, lambda t: 62 + 12 * prog(t, 17, 3), gain=1.0, pan=0.5)                                                                # after reflow: a ruler
c(sc, 24.5, 'tock', 0.7, -0.3, f=1000); c(sc, 25.3, 'tock', 0.8, 0.3, f=1500); c(sc, 26.3, 'chime', 0.8, 0.3, m=81)
c(sc, 33, 'tick', 0.9, -0.6, f=1500); c(sc, 33.3, 'thunk', 0.8, -0.6, f=140)                                                          # one step, round 1: into the middle
c(sc, 34.2, 'tick', 0.9, 0.2, f=1900); c(sc, 34.4, 'thud', 0.9, 0.2, f=44)
for mi, g_ in ((50, 0.8), (62, 0.8), (66, 0.7), (69, 0.7), (74, 0.6), (81, 0.5)): c(sc, 34.4 + 0.01 * (mi % 4), 'chime', g_, 0.2 + (mi - 66) / 40, m=mi, bright=1.2)
for j in range(3): c(sc, 40 + j * 0.5, 'clack', 0.4, -0.6 + 0.3 * j, f=2400)
c(sc, 48, 'chime', 0.9, m=86)

# 10 one family — curved (rose) and straight (sage) roads; the four names, one arrow
sc = IDX('one family')
S.track(sc, 2, 7, lambda t: 62 + 12 * ease(prog(t, 2, 5)) + 5 * np.sin(np.pi * prog(t, 2, 5)), gain=0.8, pan=-0.4)
S.track(sc, 9, 10, lambda t: 62 + 12 * prog(t, 9, 1), gain=0.9, pan=-0.4)
c(sc, 8, 'tick')
for i, t0 in enumerate((16, 24, 27)): c(sc, t0, 'chime', 0.8, 0.3, m=[74, 78, 81][i])
for i in range(4): c(sc, 32.5 + i * 0.6, 'note', 0.6, -0.5 + 0.33 * i, m=[62, 66, 69, 74][i], lead='vibes')
c(sc, 40, 'chime', 1.0, m=86)

# 11 in the wild — four cards
sc = IDX('wild')
for i, (tt, mm) in enumerate(((1, 69), (9, 74), (18, 78), (27, 81))): c(sc, tt, 'chime', 0.9, -0.45 + 0.3 * i, m=mm)
c(sc, 34, 'chime', 1.1, m=86)

# 12 build it
sc = IDX('build')
for i in range(15): c(sc, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): c(sc, 18.5 + i * 1.6, 'tick', 0.8, f=1400)

# 13 hook — flows to any blob; then the same flow asked for blob 3, then 7; the dial; title
sc = IDX('next')
for t0 in (1.5, 16.3, 20.3): S.roll(sc, t0, t0 + 3.4, lambda t, t0=t0: np.abs(np.gradient(ease(prog(t, t0, 3.2)), t)), 0.7, -0.4, seed=int(t0 * 10))
for k in range(16): c(sc, 1.2 + k * 0.05, 'clack', 0.2, 0.4 + 0.02 * k, f=2200 + 90 * (k % 8))
c(sc, 8, 'tick'); c(sc, 16, 'chime', 0.9, -0.4, m=[62, 64, 66, 69, 71, 74, 76, 78][2]); c(sc, 20, 'chime', 0.9, -0.4, m=[62, 64, 66, 69, 71, 74, 76, 78][6])
c(sc, 24, 'tick'); c(sc, 32, 'riser', 0.5, dur=3.0, m0=50, m1=62); c(sc, 33, 'tick', 0.8, 0.4)
c(sc, 38.5, 'thud', 0.8, f=45); c(sc, 38.5, 'chime', 1.0, m=62); c(sc, 38.55, 'chime', 0.85, m=69); c(sc, 39.3, 'chime', 1.0, m=74); c(sc, 40.5, 'tick'); c(sc, 41, 'chime', 1.0, m=86)
S.season_strip(sc, 41.7, 10)
S.play_pill(IDX('build'), 9)

S.render()
