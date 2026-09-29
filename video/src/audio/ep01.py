"""Episode 01 soundtrack — C lydian, 'wonder'. Celesta lead; every dot, particle and knob is heard.

    python src/audio/ep01.py      # → ep01-audio.mp3
"""
import numpy as np

from audio_lib import Score, ease, eout, prog, probe

HTML = 'ep01-generative-modelling.html'
(durs, cloud_y, moons, kde_y, data1, d5, mu_sd, part, gfit_n, fin_r, names) = probe(HTML, [
    'SC.map(s => s.dur)', 'SC[1].CLOUD.map(p => p.y)', 'MOONS.map(p => [p.r, p.c ? 1 : 0, p.y])', 'KDE_SAMPLES.map(s => s.y)',
    'DATA1.map(d => [d.r, d.x])', 'D5.map(d => [d.r, d.x])', '[MU5, SD5]', 'PART.map(p => [p.s, p.d, p.x])',
    'GFIT.samples.length', 'FIN.map(p => p.r)', 'E1.names'])
S = Score('ep01-audio', durs, seed=11)
PENTA = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84]          # C major pentatonic, for sonified data
LYD = [60, 62, 64, 66, 67, 69, 71, 72, 74, 76, 79]

# ── score ──────────────────────────────────────────────────────
CH = {'Cmaj9': ([48, 55, 59, 62, 64], 36), 'Fmaj7#11': ([53, 57, 60, 64, 71], 41), 'Am9': ([57, 60, 64, 67, 71], 45),
      'G69': ([55, 59, 62, 64, 69], 43), 'Em7': ([52, 55, 59, 62, 67], 40), 'Dm9': ([50, 57, 60, 64, 65], 38), 'D/C': ([48, 54, 57, 62, 66], 36)}
S.score(CH, [
    ([(0, 'Am9'), (8, 'Fmaj7#11'), (15, 'Dm9'), (22.5, 'Cmaj9'), (30, 'Fmaj7#11')], 0.3, .8),   # 00 trailer
    ([(0, 'Cmaj9'), (21.6, 'Fmaj7#11')], 0.6, .7),
    ([(0, 'Em7'), (26, 'Am9')], 0.5, .6),
    ([(0, 'Fmaj7#11'), (21, 'Cmaj9'), (36, 'G69')], 0.8, .9),
    ([(0, 'Dm9'), (20, 'G69'), (27, 'Cmaj9')], 0.6, .7),
    ([(0, 'Am9'), (20, 'Fmaj7#11'), (34, 'Cmaj9')], 0.6, .7),
    ([(0, 'Cmaj9'), (21, 'D/C'), (29, 'Cmaj9')], 0.6, .8),
    ([(0, 'Em7'), (19.5, 'Am9'), (34, 'Fmaj7#11')], 0.4, .5),
    ([(0, 'Cmaj9'), (7.5, 'Am9'), (14, 'Fmaj7#11'), (20.5, 'Em7'), (27, 'Dm9'), (34, 'G69'), (41, 'Cmaj9'), (49, 'Fmaj7#11')], 0.7, .9),   # 08 season map
    ([(0, 'Am9'), (7, 'Fmaj7#11'), (15, 'Cmaj9')], 0.8, .9),
    ([(0, 'Am9'), (18, 'Cmaj9')], 1.2, .9),                                                      # 10 build it
    ([(0, 'Fmaj7#11'), (8, 'G69'), (23, 'Cmaj9')], 0.5, .7),
    ([(0, 'Dm9'), (8, 'G69'), (15, 'Em7'), (21.5, 'Cmaj9')], 0.5, .8),                           # 12 next
], LYD, lead='celesta')
S.transitions()
fin_r = np.array(fin_r)
mean_u = lambda t, a, sp, d: np.mean([ease(prog(t, a + sp * r, d)) for r in fin_r[::8]], axis=0)

# 00 trailer — static; six real machines tune in, each heard doing its own thing; a heartbeat, a riser, the bang
TT = lambda i: 1.5 + i * 3.3; BANG = 22.5
S.hiss(0, 0.1, 3.5, lambda t: np.minimum(prog(t, 0.1, 0.5), 1) * (1 - ease(prog(t, 1, 2.5))), 1.6, lo=1800, hi=8000)
PAN = [-0.6, 0, 0.6, -0.6, 0, 0.6]
for i in range(6):                                                    # each tile tunes in out of its own static
    S.hiss(0, TT(i), TT(i) + 1.8, lambda t, i=i: 1 - ease(prog(t, TT(i) + 0.3, 1.4)), 0.9, lo=2200, hi=8000, pan=PAN[i])
    S.cue(0, TT(i) + 0.5, 'note', 0.9, PAN[i], lead='celesta', m=[72, 74, 76, 79, 81, 84][i])
for kk in range(int((BANG - TT(0)) / 2.2) + 1):                       # tile 0: a key for every letter typed
    w = names[kk % len(names)]
    for j in range(len(w)):
        tl = kk * 2.2 + j / (1.6 * len(w)) * 2.2
        if TT(0) + tl < BANG: S.cue(0, TT(0) + 0.5 + tl, 'key', 0.35, -0.6, f=2800 + 90 * (j % 4))
S.track(0, TT(1) + 0.5, BANG, lambda t: 67 + 5 * np.sin(((t - TT(1)) * 7) % 48 / 48 * 2 * np.pi), lambda t: 0.35 * eout(prog(t, TT(1) + 0.5, 1)), 0.5, pan=0)   # tile 1: the latent loop, sung
for e in range(1, 7): S.cue(0, TT(2) + e / 7 * 4.2, 'clack', 0.45, 0.6, f=1900 + 180 * e)                     # tile 2: each GAN epoch
S.motion(0, TT(3) + 0.4, TT(3) + 3.6, lambda t: ease(prog(t, TT(3) + 0.4, 3.2)), 0.5, lo=300, hi=1800, tone=(52, 64), pan=-0.6)   # tile 3: the flow
S.roll(0, TT(4), TT(4) + 4.8, lambda t: np.gradient(np.clip((t - TT(4)) / 4.5, 0, 1) ** 0.75, t), 0.7, 0.0, seed=1)          # tile 4: marbles
S.hiss(0, TT(5) + 0.3, TT(5) + 4.6, lambda t: 1 - ease(prog(t, TT(5) + 0.3, 4.2)), 0.8, lo=2000, hi=8000, pan=0.6)            # tile 5: noise thinning
S.cue(0, TT(5) + 4.5, 'chime', 0.8, 0.6, m=84)
S.cue(0, 16, 'riser', 0.8, dur=6.5, m0=40, m1=64); S.cue(0, 20, 'whoosh', 1.1, dur=2.5, f0=250, f1=6000)
for k in range(10): S.cue(0, 17.5 + k * 0.5 * (1 - k / 25), 'thud', 0.16 + 0.03 * k, f=55)       # the heartbeat quickens
S.cue(0, BANG, 'thud', 1.3, f=36); S.cue(0, BANG, 'thud', 0.7, f=52)
for mi, g_ in ((48, 1.0), (55, 0.85), (60, 0.9), (64, 0.8), (66, 0.75), (71, 0.7), (76, 0.55)): S.cue(0, BANG + 0.012 * (mi % 5), 'chime', g_, (mi - 62) / 30, m=mi, bright=1.3)
S.hiss(0, BANG, BANG + 3, lambda t: 0.5 * np.exp(-(t - BANG) * 1.6), 1.1, lo=3000, hi=9000)
S.cue(0, 25, 'note', 0.8, lead='celesta', m=79)
for i in range(5): S.cue(0, 31.6 + i * 0.4, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 120 * i)

# 01 data is points — two pixels drive a point (a theremin follows pixel 1), then a cloud, then 784 numbers
b1 = lambda t: 0.5 + 0.4 * np.sin(0.9 * np.minimum(t, 11.5) + 0.3)
S.track(1, 5.5, 11.8, lambda t: 64 + 12 * b1(t), lambda t: eout(prog(t, 5.5, 0.8)), 0.8)
S.cue(1, 0.6, 'tick'); S.cue(1, 3, 'tick'); S.cue(1, 6, 'tick', f=1200)
for i, y in enumerate(cloud_y):
    if i % 2 == 0: S.cue(1, 11.5 + 5 * i / len(cloud_y) + 0.15, 'drop', 0.45, (i % 7 - 3) / 5, m=S.snap(y, 0, 1, PENTA))
S.motion(1, 16.8, 18.4, lambda t: ease(prog(t, 16.8, 1.5)), 0.5, lo=400, hi=1400)
S.motion(1, 25.5, 29.4, lambda t: np.mean([ease(prog(t, 25.5 + 2.6 * i / 784, 1.3)) for i in range(0, 784, 16)], axis=0), 0.9, tone=(55, 67))
S.cue(1, 28.6, 'tick'); S.cue(1, 29.2, 'chime', m=76); S.cue(1, 31, 'note', lead='celesta', m=79); S.cue(1, 34, 'tick')

# 02 discriminative — labelled points pop in; the border learns; the far point breaks it
for r, c, y in moons:
    S.cue(2, 0.5 + 2.5 * r + 0.1, 'drop', 0.3, -0.4 if c else 0.4, m=S.snap(y, -1.3, 1.3, [57, 60, 62, 64, 67]) + (0 if c else 12))
S.motion(2, 7, 13.1, lambda t: ease(prog(t, 7, 6)), 0.8, lo=200, hi=900, tone=(45, 52))
S.cue(2, 12, 'tick'); S.cue(2, 12.5, 'tick', f=1300); S.cue(2, 19, 'note', lead='celesta', m=76)
S.cue(2, 26, 'thud', 0.45, f=55); S.cue(2, 33.5, 'note', 1.1, lead='glass', m=66); S.cue(2, 33.6, 'note', 0.8, lead='glass', m=71)

# 03 generative — the landscape rises; brand-new samples sparkle into existence (on screen, one by one)
S.motion(3, 4.5, 9.6, lambda t: ease(prog(t, 4.5, 5)), 0.8, lo=150, hi=700, tone=(41, 53))
S.cue(3, 7, 'tick'); S.cue(3, 14, 'note', 0.8, lead='glass', m=57); S.cue(3, 16, 'note', lead='celesta', m=74); S.cue(3, 21, 'tick')
for i, y in enumerate(kde_y): S.cue(3, 21.5 + i * 0.17 + 0.05, 'drop', 0.8, (i % 5 - 2) / 3, m=S.snap(y, -1.3, 1.3, PENTA))
S.cue(3, 22, 'tick', 0.7); S.cue(3, 36, 'chime', m=79)

# 04 what is a density — 420 points rain onto the line; bins double; the histogram melts into p(x)
for r, x in data1: S.cue(4, 1 + 5 * r + 0.9, 'drop', 0.22, float(np.clip(x / 3, -0.8, 0.8)), m=S.snap(x, -3, 3, PENTA))
S.motion(4, 7.5, 9.6, lambda t: eout(prog(t, 7.5, 2)), 0.7, lo=200, hi=1200, tone=(52, 59))
for i, tt in enumerate([14.5, 16.5, 18.5]): S.cue(4, tt, 'tick', 0.9, f=1300 + 300 * i); S.cue(4, tt + 0.08, 'tick', 0.5, f=1600 + 300 * i)
S.motion(4, 20.5, 23.1, lambda t: ease(prog(t, 20.5, 2.5)), 0.8, lo=300, hi=2000, tone=(55, 67))
S.cue(4, 23, 'tick'); S.cue(4, 27, 'tick'); S.cue(4, 28.5, 'tick'); S.cue(4, 29, 'chime', m=76)
S.motion(4, 33.5, 34.8, lambda t: ease(prog(t, 33.5, 1.2)), 0.6, lo=500, hi=1500); S.cue(4, 33.6, 'tick'); S.cue(4, 35, 'tick')

# 05 maximum likelihood — data drops; the knobs turn; a tone sings the log-likelihood (higher = better)
MU5, SD5 = mu_sd; D5x = np.array([x for _, x in d5])
for r, x in d5: S.cue(5, 1 + 3.5 * r + 0.7, 'drop', 0.9, float(np.clip(x / 2.5, -0.8, 0.8)), m=S.snap(x, -1, 3, PENTA))
K = [[0, -1.6, .55], [16, -1.6, .55], [22, MU5, .55], [25, MU5, .55], [28, MU5, 1.7], [30, MU5, 1.7], [33, MU5, .3], [35, MU5, .3], [38.5, MU5, SD5]]
def params(t):
    t = np.atleast_1d(t); mu = np.zeros_like(t); sd = np.zeros_like(t)
    for j, tt in enumerate(t):
        for i in range(len(K) - 1):
            if tt < K[i + 1][0] or i == len(K) - 2:
                u = float(ease((tt - K[i][0]) / (K[i + 1][0] - K[i][0]))); mu[j] = K[i][1] + (K[i + 1][1] - K[i][1]) * u; sd[j] = K[i][2] + (K[i + 1][2] - K[i][2]) * u; break
    return mu, sd
def ll(t):
    mu, sd = params(t); return np.array([np.sum(-0.5 * ((D5x - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))) for m, s in zip(mu, sd)])
tt = np.linspace(20, 42, 2200); LL = ll(tt); lo_, hi_ = LL.min(), LL.max()
S.track(5, 20, 42, lambda t: 55 + 17 * (np.interp(t, tt, LL) - lo_) / (hi_ - lo_), lambda t: 0.7 + 0.3 * eout(prog(t, 20, 1)), 0.9)
for a, b in [(16, 22), (25, 28), (30, 33), (35, 38.5)]:
    S.motion(5, a, b + 0.05, lambda t, a=a, b=b: ease(prog(t, a, b - a)), 0.5, lo=300, hi=1100)
S.cue(5, 7, 'tick'); S.cue(5, 7.5, 'note', lead='celesta', m=72); S.cue(5, 13.5, 'tick'); S.cue(5, 20, 'tick')
S.cue(5, 34, 'chime', m=79); S.cue(5, 37, 'tick'); S.cue(5, 38.6, 'note', lead='celesta', m=84); S.cue(5, 42, 'tick')

# 06 sampling — noise rises to g and slides across; each particle 'plinks' when it hits the curve
S.cue(6, 7, 'tick'); S.cue(6, 7.4, 'tick'); S.motion(6, 7, 8.3, lambda t: eout(prog(t, 7, 1.2)), 0.5, lo=200, hi=800)
S.motion(6, 14, 17.1, lambda t: ease(prog(t, 14, 3)), 0.8, lo=300, hi=1500, tone=(55, 64)); S.cue(6, 14.4, 'tick')
for k, (s0, d, x) in enumerate(part):
    rate = 900 / 26 * np.sqrt((k + 1) / 900) + 0.5
    S.cue(6, s0 + 0.45 * d, 'drop', float(min(1.0, 2.2 / np.sqrt(rate))), float(np.clip(x / 3.5, -0.8, 0.8)), m=S.snap(x, -3, 3.2, PENTA))
S.cue(6, 22, 'tick'); S.cue(6, 29.5, 'chime', 0.8, m=72); S.cue(6, 36, 'tick')

# 07 the catch — static roars as the random tries race into the thousands; a single blob misfits the moons
n = int(18.5 * 44100); t = np.arange(n) / 44100 + 0.5
env_ = np.clip((t - 0.6) / 0.6, 0, 1) * (0.35 + 0.65 * ease(prog(t, 7, 6))) * (1 - ease(prog(t, 18, 1)))
S.add(S.sfx, S.at(7, 0.5), (S.band(S.noise(n), lo=1800, hi=7000) * env_).astype(np.float32), 0.02)
S.cue(7, 7, 'note', 0.8, lead='glass', m=57); S.cue(7, 14, 'tick')
S.motion(7, 21, 22.6, lambda t: eout(prog(t, 21, 1.5)), 0.7, lo=120, hi=500, tone=(40, 47)); S.cue(7, 21, 'tick')
for i in range(gfit_n): S.cue(7, 28 + i * 0.12 + 0.1, 'drop', 0.45, (i % 5 - 2) / 3, m=[61, 63, 66, 68, 70][i % 5])
S.cue(7, 34, 'tick')

# 08 season map — a bell per stop as it appears; the traveller sweeps the whole route; every stop it passes chimes
STOPS = [1, 7.5, 10.8, 14, 17.5, 20.5, 24, 27, 30.5, 34, 37.5]
for i, t0 in enumerate(STOPS): S.cue(8, t0 + 0.1, 'chime', 0.7, -0.7 + 0.14 * i, m=LYD[i % len(LYD)] + 12)
tr = np.linspace(41, 48.5, 3000); trav = 10 * np.array([ease(prog(x, 41, 7.5)) for x in tr])
S.motion(8, 41, 48.5, lambda t: ease(prog(t, 41, 7.5)), 0.7, lo=300, hi=2400, tone=(60, 79))
for i in range(1, 11): S.cue(8, float(tr[np.argmax(trav >= i - 0.02)]), 'tick', 0.6, -0.7 + 0.14 * i, f=1400 + 110 * i)
for mi, g_ in ((60, 0.8), (64, 0.7), (67, 0.7), (71, 0.6), (76, 0.55)): S.cue(8, 49 + 0.02 * (mi % 3), 'chime', g_, (mi - 66) / 20, m=mi)

# 09 finale — pure noise flows into the data: the whole idea in one motion
S.cue(9, 4.5, 'riser', 0.8, dur=2.2, m0=43, m1=55)
S.motion(9, 6, 15.2, lambda t: mean_u(t, 6, 2, 7), 1.3, tone=(43, 67))
S.cue(9, 15, 'chime', 1.2, m=72); S.cue(9, 15.05, 'chime', 0.9, m=76); S.cue(9, 15.1, 'chime', 0.7, m=79)

# 10 build it — a key per code line; the right card's lines tick in; the playground pill
for i in range(13): S.cue(10, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): S.cue(10, 9.5 + i * 1.6, 'tick', 0.8, f=1400)
S.play_pill(10, 9)

# 11 recap
S.cue(11, 1, 'tick'); S.cue(11, 8, 'tick'); S.cue(11, 23.5, 'chime', 1.1, m=76); S.cue(11, 26, 'tick')

# 12 next — a typewriter writes episode 02's first sentence; the chain rule grows; the end card and the season strip
TYPE1 = 'Once upon a time, a machine learned to write'
for j in range(1, len(TYPE1) + 1):
    if TYPE1[j - 1] != ' ': S.cue(12, 2 + 16 * j / len(TYPE1), 'key', 0.5, -0.5 + j / len(TYPE1), f=2600 + 150 * (j % 5))
S.cue(12, 8, 'tick'); S.cue(12, 8.1, 'chime', 0.6, m=79)
S.cue(12, 21.5, 'thud', 0.7, f=48)
for mi, g_ in ((60, 0.9), (67, 0.7), (72, 0.8), (76, 0.6)): S.cue(12, 21.5 + 0.02 * (mi % 3), 'chime', g_, (mi - 66) / 20, m=mi)
S.cue(12, 22.5, 'tick', 0.8, f=1400)
S.season_strip(12, 23.7, 1)

S.render()
