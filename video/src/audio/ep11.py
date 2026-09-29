"""Episode 11 soundtrack — G major, 'asking a generator for what you want'. Celesta lead over warm, patient pads.

Signature: a request you can hear. Every request is a typewriter key; the noise that answers it is a cluster of voices
that start detuned (noise) and glide onto one chord tone (the answer). The dial is heard as tuning: at w = 0 the voices
wander, at w = 1 they settle into tune, and turned up past the sweet spot they collapse into one thin unison — the
sound of 'obedient but samey'. The finale closes Season 1 on the home chord and rings one clear new note for Season 2.

    python src/audio/ep11.py      # → episodes/ep11-audio.mp3   (or: python src/make.py 11)
"""
import numpy as np

from audio_lib import Score, ease, prog, scene_durations, scene_index

durs = scene_durations(11)
S = Score('ep11-audio', durs, seed=11)
c = S.cue
IDX = lambda ch: scene_index(11, ch)


def swarm(sc, t0, t1, target, n=6, spread=9, gain=1.0, pan=0.0, seed=0, w=1.0, fly=None):
    """n voices from random pitches onto `target` (a MIDI note) over [t0, t1]. w < 1 leaves them loose (they wander
    and never quite agree); w > 1 squeezes them into a thin unison early. fly: optional f(t) ∈ [0, 1] for the glide."""
    rng = np.random.default_rng(seed); f = fly or (lambda t: ease(prog(t, t0, t1 - t0)))
    for j in range(n):
        m0 = target + rng.uniform(-spread, spread); lean = (1 - min(w, 1)) * rng.uniform(-4, 4)
        tight = 1 + max(w - 1, 0) * 0.8
        S.track(sc, t0, t1 + 0.6, lambda t, m0=m0, lean=lean, tight=tight: m0 + (target + lean - m0) * np.clip(f(t) * tight, 0, 1),
                amp=lambda t: 0.5 + 0.5 * np.clip(f(t), 0, 1), gain=gain / np.sqrt(n) * 1.5, pan=pan + 0.15 * (j - n / 2) / n, fade=0.4)


CH = {'Gmaj9': ([43, 50, 54, 57, 59], 31), 'Em9': ([40, 47, 50, 54, 55], 28), 'Cmaj7': ([48, 55, 59, 64, 67], 36), 'Dsus': ([50, 55, 57, 62, 64], 38),
      'Am9': ([45, 52, 55, 59, 60], 33), 'Bm7': ([47, 54, 57, 62, 64], 35), 'Cadd9': ([48, 55, 62, 64, 67], 36), 'G': ([43, 50, 55, 59, 62], 31),
      'D/F#': ([42, 50, 54, 57, 62], 30), 'Ebmaj7#11': ([51, 58, 62, 67, 69], 39)}
K0, KEY, FLY = 6.0, 2.25, 1.7; BANG = K0 + 7 * KEY + FLY + 0.2
S.score(CH, [
    ([(0, 'Em9'), (K0, 'Cmaj7'), (K0 + 3 * KEY, 'Am9'), (K0 + 5 * KEY, 'Dsus'), (BANG, 'G'), (28, 'Gmaj9')], 0.0, .85),   # 00 intro
    ([(0, 'Gmaj9'), (8, 'Em9'), (15, 'Cmaj7'), (23, 'Am9'), (30, 'Dsus'), (38, 'Gmaj9')], 0.7, .8),                    # 01 asking
    ([(0, 'Gmaj9'), (8, 'Bm7'), (15, 'Cmaj7'), (23, 'Em9'), (31, 'Dsus'), (39, 'G')], 0.7, .8),                         # 02 tell it
    ([(0, 'Em9'), (8, 'Cadd9'), (15, 'Gmaj9'), (23, 'Am9'), (31, 'Bm7'), (38, 'Dsus')], 0.6, .85),                       # 03 digits
    ([(0, 'Am9'), (8, 'Em9'), (16, 'Cmaj7'), (24, 'Gmaj9'), (32, 'Am9'), (40, 'Dsus'), (47, 'G')], 0.8, .85),            # 04 bayes
    ([(0, 'Gmaj9'), (8, 'Bm7'), (16, 'Cmaj7'), (26, 'Em9'), (33, 'G'), (40, 'Ebmaj7#11'), (47, 'Dsus')], 0.7, .85),     # 05 classifier guidance
    ([(0, 'Cmaj7'), (9, 'Am9'), (17, 'Em9'), (24, 'Gmaj9'), (31, 'Cadd9'), (39, 'Dsus'), (46, 'G')], 0.8, .9),           # 06 CFG
    ([(0, 'Em9'), (9, 'Gmaj9'), (17, 'Bm7'), (24, 'Cmaj7'), (31, 'Am9'), (40, 'Dsus'), (47, 'Em9')], 0.7, .85),          # 07 the dial
    ([(0, 'Gmaj9'), (8, 'Cmaj7'), (16, 'Em9'), (24, 'Am9'), (32, 'Bm7'), (40, 'Cadd9'), (47, 'Dsus')], 0.7, .85),        # 08 the price
    ([(0, 'Em9'), (8, 'Cmaj7'), (16, 'Gmaj9'), (24, 'Ebmaj7#11'), (31, 'Dsus')], 0.8, .85),                               # 09 saying no
    ([(0, 'Gmaj9'), (9, 'Bm7'), (18, 'Cmaj7'), (27, 'Dsus'), (34, 'G')], 1.0, 1.0),                                      # 10 in the wild
    ([(0, 'Cmaj7'), (18, 'Gmaj9')], 1.3, 1.0),                                                                           # 11 build it
    ([(0, 'Em9'), (8, 'Am9'), (16, 'Cmaj7'), (24, 'D/F#')], 0.6, .75),                                                   # 12 prompts
    ([(0, 'G'), (8, 'Cadd9'), (16, 'Em9'), (23.5, 'Gmaj9')], 0.5, .7),                                                   # 13 season 1 → season 2
], [67, 69, 71, 74, 76, 78, 79, 81, 83, 86, 88, 91], lead='celesta')
S.transitions()

# 00 cold open — a field of static; eight keystrokes, each answered by a swarm of voices landing on one note of the word
S.hiss(0, 0.2, BANG + 0.3, lambda t: np.minimum(prog(t, 0.2, 1.5), 1) * (0.85 - 0.08 * np.clip((t - K0) / KEY, 0, 8)), 1.6, lo=1800, hi=8000)
for k in range(24): c(0, 0.5 + k * 0.22 + 0.08 * np.sin(k * 5), 'tick', 0.08, float(np.sin(k * 2.7)) * 0.8, f=2500 + 900 * abs(np.sin(k * 1.3)))
LETTER = [55, 59, 62, 64, 67, 71, 74, 79]                     # G  u  i  d  a  n  c  e : up the G major scale, one note per request
for k in range(8):
    tk = K0 + k * KEY; pan = -0.75 + 0.21 * k
    c(0, tk, 'key', 0.9, pan, f=2600 + 150 * (k % 3)); c(0, tk + 0.04, 'tick', 0.35, pan, f=3200)
    swarm(0, tk + 0.35, tk + 0.35 + FLY, LETTER[k] + 12, n=5, spread=10, gain=0.8, pan=pan, seed=k)
    c(0, tk + 0.35 + FLY, 'note', 0.7, pan, m=LETTER[k] + 12, lead='celesta'); c(0, tk + 0.35, 'whoosh', 0.35, pan, dur=FLY, f0=500, f1=2600)
c(0, BANG - 3.5, 'riser', 0.5, dur=3.5, m0=43, m1=55)
c(0, BANG, 'thud', 1.2, f=38); c(0, BANG, 'thud', 0.6, f=55)
for mi, g_ in ((43, 1.0), (50, 0.85), (55, 0.9), (59, 0.8), (62, 0.75), (67, 0.7), (74, 0.55)): c(0, BANG + 0.012 * (mi % 5), 'chime', g_, (mi - 60) / 30, m=mi, bright=1.2)
S.hiss(0, BANG, BANG + 3, lambda t: 0.4 * np.exp(-(t - BANG) * 1.6), 1.2, lo=3000, hi=9000)
for i in range(5): c(0, BANG + 2.2 + i * 0.3, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 110 * i)

# 01 what asking means — the blobs colour in (eight soft notes), the filter (seven notes fade, one rings), Bayes, the classifier
sc = IDX('asking')
c(sc, 1, 'chime', 0.7, -0.4, m=79)
for k in range(8): c(sc, 8.6 + k * 0.14, 'note', 0.45, -0.7 + 0.2 * k, m=[67, 69, 71, 74, 76, 78, 79, 81][k], lead='celesta')
c(sc, 15.5, 'tick'); c(sc, 16, 'chime', 0.9, 0.3, m=71)
c(sc, 23.5, 'whoosh', 0.5, -0.4, dur=2.0, f0=2600, f1=500); c(sc, 25.5, 'chime', 1.0, -0.4, m=71, bright=1.3)
c(sc, 30.5, 'tick'); c(sc, 31, 'chime', 0.8, 0.3, m=74)
c(sc, 38.5, 'clink', 0.6, 0.3, f=2400)
for k in range(6): c(sc, 39 + k * 1.1, 'tick', 0.25, 0.3, f=1600 + 200 * (k % 3))       # the probe slides; the bars shift

# 02 tell the network — the box; y drops in; one-hot slots click on; the ∅ slot flickers; flows to none, 3, 7
sc = IDX('tell it')
c(sc, 1, 'thunk', 0.5, -0.5, f=160); c(sc, 2, 'clack', 0.6, -0.4, f=2600); c(sc, 2.4, 'tock', 0.6, -0.4, f=1100)
for i in range(8): c(sc, 8.6 + i * 0.07, 'key', 0.35, -0.8 + 0.08 * i, f=2800 + 100 * (i % 2))
c(sc, 9.3, 'chime', 0.8, -0.6, m=71)
for i in range(6): c(sc, 15.5 + i * 0.83, 'key', 0.3, -0.3, f=3400 if i % 3 == 0 else 2800)
c(sc, 23.5, 'tick'); c(sc, 31, 'chime', 0.8, 0.5, m=74)
for t0, tgt in ((31.5, None), (39.3, 71), (43.5, 79)):
    S.roll(sc, t0, t0 + 3.2, lambda t, t0=t0: np.abs(np.gradient(ease(prog(t, t0, 2.8)), t)), 0.7, 0.5, seed=int(t0))
    if tgt is None:
        for k in range(8): c(sc, t0 + 2.8 + k * 0.05, 'note', 0.35, 0.2 + 0.08 * k, m=[67, 69, 71, 74, 76, 78, 79, 81][k], lead='celesta')
    else: swarm(sc, t0, t0 + 2.8, tgt, n=5, gain=0.8, pan=0.5, seed=int(t0)); c(sc, t0 + 2.8, 'chime', 0.9, 0.5, m=tgt)

# 03 digits on request — ten rows pop in; the column highlight steps; the judge's bars; rose rings (soft thunks)
sc = IDX('digits')
for d in range(10): c(sc, 1.5 + d * 0.35, 'clack', 0.35, -0.6, f=2000 + 120 * d); c(sc, 1.55 + d * 0.35, 'note', 0.3, -0.6, m=[67, 69, 71, 74, 76, 78, 79, 81, 83, 86][d], lead='celesta')
for k in range(3): c(sc, 15 + k * 2.7, 'tock', 0.6, -0.5 + 0.1 * k, f=1300)
c(sc, 23.5, 'whoosh', 0.4, 0.4, dur=0.8, f0=600, f1=2400)
for d in range(10): c(sc, 24 + d * 0.1, 'tick', 0.35, 0.2 + 0.05 * d, f=1500 + 90 * d)
for k in range(10): c(sc, 31 + k * 0.08, 'thunk', 0.18, -0.5, f=180)
c(sc, 38, 'riser', 0.35, dur=7, m0=50, m1=62)

# 04 Bayes, as arrows — each line of the derivation is a stroke; the classifier's arrows (a sweep); the sum bends; the agreement
sc = IDX('bayes')
c(sc, 1, 'chime', 0.7, 0.4, m=74)
for t0 in (8.5, 24.5): c(sc, t0, 'chisel', 0.6, 0.4, f=560); c(sc, t0 + 0.3, 'chime', 0.7, 0.4, m=79)
c(sc, 16.5, 'tick', 0.8, 0.4); c(sc, 25.5, 'clink', 0.5, 0.4, f=2600)
c(sc, 32, 'whoosh', 0.7, -0.5, dur=1.8, f0=300, f1=2200)
S.hiss(sc, 32, 40, lambda t: 0.25 * prog(t, 32, 1) * (1 - prog(t, 39, 1)), 0.6, lo=1200, hi=4000, pan=-0.5)
c(sc, 40, 'whoosh', 0.6, -0.5, dur=1.2, f0=2000, f1=700); c(sc, 41, 'note', 0.7, -0.5, m=71, lead='celesta')
c(sc, 47, 'note', 0.7, -0.5, m=71, lead='vibes'); c(sc, 47.04, 'note', 0.6, -0.5, m=71, lead='celesta')      # the same note twice: 'almost the same wind'

# 05 classifier guidance — noisy digits (hiss steps down); the translation; three runs at w = 0, 1, 4 heard as tuning
sc = IDX('classifier')
c(sc, 1, 'chime', 0.7, 0.4, m=76)
for i, lv in enumerate((0.55, 0.35, 0.18, 0.0)): c(sc, 8.8 + i * 0.4, 'tick', 0.5, 0.3, f=1400 + 200 * i)
S.hiss(sc, 8.5, 16, lambda t: np.maximum(0, 0.4 - 0.1 * np.floor((t - 8.5) / 0.4).clip(0, 4)) * prog(t, 8.5, 0.3) * (1 - prog(t, 15, 1)), 0.6, lo=2000, hi=8000, pan=0.3)
for t0 in (16.5, 19, 21.5): c(sc, t0, 'chisel', 0.5, 0.4, f=600)
for t0, w, pan in ((26.3, 0.0, -0.5), (33.3, 1.0, -0.5), (40.3, 4.0, -0.5)):
    S.roll(sc, t0, t0 + 3.2, lambda t, t0=t0: np.abs(np.gradient(ease(prog(t, t0, 3)), t)), 0.6, pan, seed=int(t0))
    swarm(sc, t0, t0 + 3, 74, n=6, spread=10, gain=0.9, pan=pan, seed=int(t0), w=w)
    c(sc, t0 - 0.3, 'tock', 0.7, 0.5, f=900 + 250 * w)
c(sc, 47, 'thunk', 0.4, 0.4)
for j in range(12): c(sc, 47.3 + j * 0.08, 'clack', 0.25, 0.4, f=2400)

# 06 classifier-free — the two one-hots click; grey and plum arrows (two tones); the sage push; w = 1 then 3 (the tone climbs)
sc = IDX('CFG')
c(sc, 1, 'chime', 0.8, 0.4, m=79); c(sc, 9.5, 'chisel', 0.5, 0.4, f=580)
c(sc, 17.5, 'key', 0.6, 0.4, f=3000); c(sc, 17.9, 'key', 0.6, 0.4, f=2600)
c(sc, 18, 'note', 0.6, -0.5, m=67, lead='vibes'); c(sc, 20, 'note', 0.7, -0.5, m=74, lead='celesta')
c(sc, 24, 'whoosh', 0.4, -0.5, dur=0.8, f0=900, f1=2000)
S.track(sc, 25.5, 28, lambda t: 67 + 7 * ease(prog(t, 25.5, 2.5)), gain=0.9, pan=-0.5)
S.track(sc, 33, 36.2, lambda t: 74 + 12 * ease(prog(t, 33, 3)), gain=0.9, pan=-0.5)
for k in range(3): c(sc, 26 + k * 3.8, 'tick', 0.5, -0.5, f=1500 + 300 * k)
c(sc, 31.5, 'chime', 1.0, 0.4, m=79); c(sc, 31.55, 'chime', 0.8, 0.4, m=86)
c(sc, 39.5, 'tick', 0.8, 0.4); c(sc, 39.8, 'tick', 0.8, 0.4)             # two calls per step

# 07 the dial — the dial clicks round (0 → 1 → 2 → 4 → 8); each run a swarm that tightens; the shading swells
sc = IDX('the dial')
for t0, w in ((1.5, 0), (9.2, 0), (13, 1), (17.2, 2), (21.5, 4), (26, 8)):
    if t0 > 2: c(sc, t0, 'clack', 0.8, 0.4, f=1800 + 160 * w); c(sc, t0 + 0.05, 'tock', 0.5, 0.4, f=900 + 120 * w)
    S.roll(sc, t0, t0 + 2.4, lambda t, t0=t0: np.abs(np.gradient(ease(prog(t, t0, 2.2)), t)), 0.5, -0.5, seed=int(t0 * 3))
for t0, w in ((13, 1), (17.2, 2), (21.5, 4), (26, 8)): swarm(sc, t0, t0 + 2.2, 67, n=6, spread=8, gain=0.7, pan=-0.5, seed=int(t0), w=w)
c(sc, 31, 'whoosh', 0.5, 0.4, dur=1.5, f0=300, f1=1500); c(sc, 32, 'chime', 0.7, 0.4, m=74)
c(sc, 40, 'note', 0.6, 0.4, m=79, lead='celesta'); c(sc, 40.4, 'note', 0.5, 0.4, m=78, lead='celesta')

# 08 the price — five rows of sevens pop; the three judges draw (glides that rise, rise-then-fall, fall); the title again
sc = IDX('the price')
for j in range(5):
    for q in range(8): c(sc, 1 + j * 1.2 + q * 0.03, 'clack', 0.22, -0.8 + 0.05 * q, f=2000 + 150 * j)
S.track(sc, 16.3, 18.3, lambda t: 67 + 12 * np.sqrt(ease(prog(t, 16.3, 2))), gain=0.8, pan=0.5)                          # agreement: rises and saturates
S.track(sc, 24.3, 26.3, lambda t: 72 - 5 * np.sin(np.pi * np.clip(prog(t, 24.3, 2) * 1.6, 0, 1)) * (prog(t, 24.3, 2) < 0.62) + 12 * np.clip((prog(t, 24.3, 2) - 0.4) / 0.6, 0, 1) ** 1.3, gain=0.8, pan=0.5)   # FID: dips then climbs
S.track(sc, 32.3, 34.3, lambda t: 79 - 14 * ease(prog(t, 32.3, 2)), gain=0.8, pan=0.5)                                   # recall falls
c(sc, 26, 'chime', 0.9, 0.6, m=79)                                                                                       # the best w
c(sc, 40, 'whoosh', 0.5, -0.5, dur=1.2, f0=600, f1=2400)
for k in range(8): c(sc, 40.3 + k * 0.06, 'note', 0.3, -0.6, m=LETTER[k] + 12, lead='celesta')
for k in range(8): c(sc, 41.9 + k * 0.06, 'note', 0.3, -0.6, m=LETTER[0] + 12, lead='celesta')            # w = 2.5: the word collapses toward one note

# 09 saying no — ask 3 at 1.25; avoid 2 (a low, detuned thud); the overshoot at w = 2 (a glide that flies past)
sc = IDX('saying no')
c(sc, 1, 'chime', 0.7, 0.4, m=76); c(sc, 8, 'chisel', 0.5, 0.4, f=600); c(sc, 11, 'tick', 0.6, 0.4)
S.roll(sc, 2, 5.2, lambda t: np.abs(np.gradient(ease(prog(t, 2, 3)), t)), 0.6, -0.5, seed=91); swarm(sc, 2, 5, 71, n=5, gain=0.7, pan=-0.5, seed=91, w=1.25)
c(sc, 16, 'thunk', 0.8, -0.2, f=110); c(sc, 16.05, 'note', 0.4, -0.2, m=66, lead='vibes')
S.roll(sc, 16.3, 19.5, lambda t: np.abs(np.gradient(ease(prog(t, 16.3, 3)), t)), 0.6, -0.5, seed=92); swarm(sc, 16.3, 19.3, 71, n=5, gain=0.7, pan=-0.6, seed=92, w=1.25)
c(sc, 19.3, 'chime', 0.8, -0.6, m=71)
S.roll(sc, 24.3, 27.5, lambda t: np.abs(np.gradient(ease(prog(t, 24.3, 3)), t)), 0.7, -0.5, seed=93)
S.track(sc, 24.3, 27.6, lambda t: 64 + 16 * ease(prog(t, 24.3, 3.3)), gain=0.9, pan=-0.7); c(sc, 27.5, 'scratch', 0.6, -0.7, dur=0.4)
c(sc, 31, 'chime', 0.8, 0.4, m=79)

# 10 in the wild — four cards
sc = IDX('wild')
for i, (tt, mm) in enumerate(((1, 71), (9, 74), (18, 79), (27, 83))): c(sc, tt, 'chime', 0.9, -0.45 + 0.3 * i, m=mm)
c(sc, 34, 'chime', 1.1, m=86)

# 11 build it
sc = IDX('build')
for i in range(13): c(sc, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): c(sc, 18.5 + i * 1.8, 'tick', 0.8, f=1400)

# 12 prompts — three one-hots click; the prompt typed key by key; a sea of slots (a rattle); 'reading' (a question)
sc = IDX('prompts')
for i in range(3): c(sc, 1 + i * 1.2, 'key', 0.6, -0.6, f=2800); c(sc, 1.05 + i * 1.2, 'note', 0.4, -0.6, m=[71, 79, 86][i], lead='celesta')
PROMPT = 'a seven, drawn boldly, in red ink'
for i, ch in enumerate(PROMPT):
    if ch != ' ': c(sc, 8.5 + 3.5 * i / len(PROMPT), 'key', 0.5, -0.5 + 0.02 * i, f=2400 + 300 * (i % 4))
c(sc, 12.5, 'rattle', 0.5, -0.3, dur=1.4)
c(sc, 16.5, 'whoosh', 0.5, 0.5, dur=1, f0=400, f1=2200); c(sc, 18, 'note', 0.5, 0.5, m=74, lead='vibes'); c(sc, 18.4, 'note', 0.5, 0.5, m=76, lead='vibes')
c(sc, 24, 'riser', 0.35, dur=6.5, m0=43, m1=55)

# 13 Season 1 complete → Season 2 — eleven dots, the home chord; one new bell for Season 2's first episode; the last line
sc = IDX('next')
c(sc, 0, 'thud', 0.8, f=40)
for mi, g_ in ((43, 0.9), (50, 0.8), (55, 0.85), (59, 0.75), (62, 0.7), (67, 0.6)): c(sc, 0.05 + 0.012 * (mi % 5), 'chime', g_, (mi - 58) / 30, m=mi, bright=1.1)
S.season_strip(sc, 2, 11)
c(sc, 5.5, 'chime', 0.7, 0, m=79)                                                                          # 'complete'
c(sc, 16, 'whoosh', 0.5, 0, dur=1.5, f0=300, f1=2000)
for i in range(10): c(sc, 17.5 + i * 0.12, 'tick', 0.35, -0.7 + 0.15 * i, f=1500 + 120 * i)
c(sc, 17.5, 'chime', 1.0, -0.7, m=86, bright=1.3)                                                         # Season 2, episode 1: tokens
c(sc, 23.5, 'note', 0.8, 0, m=67, lead='celesta'); c(sc, 24.3, 'note', 0.8, 0, m=74, lead='celesta'); c(sc, 25.2, 'note', 0.9, 0, m=79, lead='celesta')
c(sc, 26.5, 'chime', 0.6, 0, m=91, bright=1.3)
S.play_pill(IDX('build'), 9)

S.render()
