"""Episode 04 soundtrack — G minor, 'a detective story about metrics'. Marimba lead; every verdict is a
rubber stamp (CAUGHT / FOOLED / COPIED) landing on the frame the stamp hits.

    python src/audio/ep04.py      # → ep04-audio.mp3
"""
import numpy as np

from audio_lib import Score, ease, eout, prog, probe

(durs, nn_d) = probe('ep04-evaluation.html', ['SC.map(s => s.dur)', '["real","vae","copier"].map(k => PAIRS[k].d)'])
S = Score('ep04-audio', durs, seed=44)
GM = [55, 58, 60, 62, 65, 67, 70, 72, 74, 77, 79]
STAMP = 0.28   # the stamp animation scales down onto the page; impact ≈ 0.28 s after it starts

CH = {'Gm9': ([55, 58, 62, 65, 69], 43), 'Ebmaj7': ([51, 58, 62, 67, 70], 39), 'Cm9': ([48, 55, 58, 62, 63], 36), 'D7sus': ([50, 57, 60, 62, 67], 38),
      'D7': ([50, 54, 57, 60, 66], 38), 'Bbmaj9': ([46, 53, 57, 60, 62], 34), 'Fsus': ([53, 58, 60, 65, 70], 41), 'Gm': ([55, 58, 62, 67, 70], 43)}
S.score(CH, [
    ([(0, 'Gm9'), (12.6, 'Ebmaj7')], 0.5, .7),
    ([(0, 'Bbmaj9'), (7, 'Gm9'), (21, 'Ebmaj7')], 0.6, .8),
    ([(0, 'Cm9'), (15, 'D7sus'), (24, 'Gm9'), (38, 'D7')], 0.6, .7),
    ([(0, 'Ebmaj7'), (15, 'Cm9'), (23, 'D7')], 0.5, .6),
    ([(0, 'Gm9'), (16, 'Bbmaj9')], 0.6, .8),
    ([(0, 'Cm9'), (22, 'Ebmaj7'), (30, 'Gm9'), (38, 'D7')], 0.7, .8),
    ([(0, 'Gm9'), (15, 'Bbmaj9'), (31, 'Ebmaj7'), (39, 'D7sus')], 0.6, .8),
    ([(0, 'Cm9'), (16, 'Ebmaj7'), (30, 'Gm9'), (47, 'D7')], 0.6, .8),
    ([(0, 'Gm9'), (15, 'Fsus'), (22, 'Cm9')], 0.6, .7),
    ([(0, 'Bbmaj9'), (15, 'Gm9'), (30, 'Ebmaj7'), (38, 'D7sus')], 0.6, .8),
    ([(0, 'Ebmaj7'), (15, 'Cm9'), (30, 'Bbmaj9')], 0.7, .8),
    ([(0, 'Gm9'), (16, 'Cm9'), (24, 'D7')], 0.5, .6),
    ([(0, 'Gm'), (15, 'Ebmaj7'), (23, 'Cm9'), (31, 'D7sus'), (37.5, 'Gm9')], 0.6, .7),
], GM, lead='marimba', lead_gain=0.06)
S.transitions()

# 00 cold open — three suspects; two stamps; the title
for p in range(3): S.cue(0, 0.4 + p * 0.4, 'note', 0.7, -0.6 + 0.6 * p, lead='marimba', m=[67, 70, 74][p])
S.cue(0, 7 + STAMP, 'stamp', 1.0, 0.0); S.cue(0, 8.5 + STAMP, 'stamp', 1.0, 0.6)
S.cue(0, 12.6, 'chime', 1.3, m=74); S.cue(0, 12.6, 'thud', 0.4, f=43)
for i in range(5): S.cue(0, 16 + i * 0.35, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 120 * i)

# 01 what does good mean — one right answer ✓ … infinitely many sevens
S.cue(1, 0.6, 'tick'); S.cue(1, 3, 'note', lead='marimba', m=79)
S.cue(1, 7, 'tick', pan=0.4)
for i in range(8): S.cue(1, 8 + i * 0.3 + 0.1, 'drop', 0.7, 0.2 + 0.08 * i, m=GM[2 + i])
S.cue(1, 14, 'tick')
for t0, m in [(21.5, 67), (31.5, 72), (35, 77)]: S.cue(1, t0, 'note', 1.0, lead='marimba', m=m)

# 02 judge 1: likelihood — test digits fall into each model: fine / −∞ / −∞
S.cue(2, 0.6, 'tick'); S.cue(2, 9, 'tick')
for i, (t0, bad) in enumerate([(8, False), (15, True), (24, True)]):
    k = 1
    while t0 + 3 * k < 50:
        S.cue(2, t0 + 3 * k - 0.05, 'drop', 0.55, -0.3 + 0.3 * i, m=[74, 61, 63][i] if bad else 74); k += 1
S.cue(2, 15.3, 'note', 0.8, lead='glass', m=49); S.cue(2, 24.3, 'note', 0.8, lead='glass', m=51)
S.cue(2, 24 + STAMP, 'stamp', 1.0, 0.1); S.cue(2, 25 + STAMP, 'stamp', 1.0, 0.5)
S.cue(2, 30, 'tick'); S.cue(2, 38.5, 'note', 0.8, lead='marimba', m=58); S.cue(2, 45, 'note', 0.9, lead='glass', m=61)

# 03 how likelihood lies — a wall of static tiles, one good digit; the math; the bars
for i in range(100):
    if i % 2 == 0: S.cue(3, 2 + i * 0.05 + 0.05, 'key', 0.22, -0.7 + 0.014 * i, f=1800 + 20 * (i % 10))
S.cue(3, 9, 'chime', 0.9, m=79)
for tt in [15, 17, 19, 20]: S.cue(3, tt, 'tick', 0.7)
for i in range(2): S.motion(3, 23.5 + 0.4 * i, 24.6 + 0.4 * i, lambda t, i=i: eout(prog(t, 23.5 + 0.4 * i, 1)), 0.5, lo=300, hi=1200, tone=(55, 62 + 3 * i))

# 04 a better eye — pixel distance lies, feature distance tells the truth
for i in range(4): S.cue(4, 0.6 + i * 0.3, 'tick', 0.6, -0.6 + 0.4 * i)
S.cue(4, 8, 'note', 0.9, lead='glass', m=61); S.cue(4, 16, 'note', lead='marimba', m=74)

# 05 inception score — confident vs a shrug; the bar chart; CAUGHT … FOOLED
S.cue(5, 1, 'tick'); S.cue(5, 8, 'note', lead='marimba', m=79); S.cue(5, 10, 'note', 0.8, lead='marimba', m=66)
S.cue(5, 15, 'tick'); S.cue(5, 17, 'tick'); S.cue(5, 22, 'chime', m=74)
for i in range(5): S.cue(5, 30 + i * 0.6, 'note', 0.6, lead='marimba', m=GM[3 + i])
S.cue(5, 34 + STAMP, 'stamp', 0.9, 0.4); S.cue(5, 39 + STAMP, 'stamp', 1.0, 0.4)

# 06 FID — clouds of features; bell curves; a sensible ranking … except the copier
for tt in [8, 23, 29, 39]: S.motion(6, tt, tt + 0.9, lambda t, tt=tt: eout(prog(t, tt, 0.8)), 0.45, lo=300, hi=1400, pan=-0.4)
S.cue(6, 15, 'chime', 0.7, m=74); S.cue(6, 23, 'tick')
for i in range(5): S.cue(6, 31 + i * 0.6, 'note', 0.6, 0.4, lead='marimba', m=GM[3 + i])
S.cue(6, 40 + STAMP, 'stamp', 1.0, 0.5)

# 07 precision & recall — territory balls; samples; the table; the copier still fools it
S.motion(7, 8, 9.6, lambda t: eout(prog(t, 8, 1.5)), 0.6, lo=200, hi=900, tone=(55, 62), pan=-0.4)
S.cue(7, 16, 'tick'); S.cue(7, 23, 'tick'); S.motion(7, 23, 23.9, lambda t: eout(prog(t, 23, 0.8)), 0.4, pan=-0.4)
for i in range(5): S.cue(7, 30 + i * 0.15, 'tick', 0.5, 0.4, f=1200 + 100 * i)
S.cue(7, 38, 'note', lead='marimba', m=67); S.cue(7, 48 + STAMP, 'stamp', 1.0, 0.6)

# 08 is it new? — nearest-neighbour pairs; the copier's distance is exactly zero
for r in range(3):
    for i in range(3):
        d = nn_d[r][i]; t0 = 8 + r * 1.2 + i * 0.3
        if d < 0.01: S.cue(8, t0, 'clink', 0.9, -0.4 + 0.3 * i, f=2400)
        else: S.cue(8, t0, 'drop', 0.7, -0.4 + 0.3 * i, m=S.snap(d, 2, 6, GM))
S.cue(8, 15, 'tick'); S.cue(8, 17 + STAMP, 'stamp', 1.0, 0.5); S.cue(8, 30, 'thud', 0.5, f=48)

# 09 a judge that learns — the gold standard; automate the human; 50% = indistinguishable
S.cue(9, 0.6, 'tick'); S.cue(9, 1, 'note', 0.7, lead='marimba', m=70)
S.motion(9, 15, 15.9, lambda t: eout(prog(t, 15, 0.8)), 0.5, lo=400, hi=1500); S.cue(9, 15.5, 'note', lead='marimba', m=74)
S.cue(9, 23, 'tick')
for i in range(5): S.cue(9, 30 + i * 0.6, 'note', 0.6, lead='marimba', m=GM[3 + i])
S.cue(9, 38.5, 'chime', 1.2, m=79)

# 10 the same judges for language models
for i, t0 in enumerate([1, 15, 22, 30]): S.cue(10, t0, 'note', 0.9, -0.3 + 0.2 * i, lead='marimba', m=[67, 70, 74, 77][i])

# 11 the scorecard — rows; the copier's fooled boxes light up; only novelty catches it; Goodhart
for i in range(5): S.cue(11, 1 + i * 0.3, 'tick', 0.6, f=1200 + 80 * i)
for j in range(6): S.cue(11, 8.5 + j * 0.2, 'note', 0.5, -0.5 + 0.2 * j, lead='glass', m=[61, 63, 61, 63, 61, 63][j])
S.cue(11, 10, 'chime', 0.9, m=79); S.cue(11, 24, 'thud', 0.5, f=43); S.cue(11, 24, 'chime', 1.1, m=74)

# 12 hook → GANs — the judge teaches; accuracy slides to 50%; forger and detective
S.cue(12, 0.6, 'note', 0.8, lead='marimba', m=67); S.cue(12, 8, 'note', 0.8, lead='marimba', m=74)
S.motion(12, 15, 16, lambda t: eout(prog(t, 15, 0.8)), 0.5, lo=500, hi=2000, tone=(62, 74))
S.track(12, 23, 35, lambda t: 74 - 12 * ease(prog(t, 23, 12)), lambda t: 0.8, 0.8)
S.cue(12, 31, 'tick'); S.cue(12, 35.5, 'riser', 0.9, dur=2.0, m0=43, m1=55)
S.cue(12, 37.5, 'thud', 1.0, f=43); S.cue(12, 37.5, 'chime', 1.4, m=62); S.cue(12, 37.55, 'chime', 1.0, m=70)
S.cue(12, 39, 'tick'); S.cue(12, 39.5, 'chime', 1.0, m=79)

S.render()
