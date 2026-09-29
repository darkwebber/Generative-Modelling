"""Episode 07 soundtrack — C minor, 'sculpting a landscape'. Vibraphone lead over warm pads.

Tactile, data-driven effects: every marble sound is computed from the real Langevin / gradient-descent
trajectories in ep07-assets.js (rolling loudness = the marbles' actual mean speed, contact grains
thin out as they settle), each Metropolis proposal gets its coin, tock (accepted) or thunk (rejected),
each sculpting step is a chisel tap as loud as the landscape actually changed, and each time the long
chain crosses a ridge you hear it hop.

    python src/audio/ep07.py      # → ep07-audio.mp3
"""

import numpy as np

from audio_lib import Score, assets, ease, eout, hz, prog, scene_durations

durs = scene_durations(7)
A = assets(7)
S = Score('ep07-audio', durs, seed=7)
LANG, GD, CH = np.array(A['lang']), np.array(A['gd']), np.array(A['chain'])
NL = len(LANG) - 1


def stage_speed(traj, fmap, t):
    """Mean marble speed (screen-agnostic units/s) when the renderer shows stage fmap(t) of traj (linear between records)."""
    f = np.clip(fmap(t), 0, NL); i = np.minimum(np.floor(f).astype(int), NL - 1)
    step = np.linalg.norm(traj[1:] - traj[:-1], axis=2).mean(1)            # mean displacement per record
    df = np.gradient(f, t); return step[i] * np.abs(df)


CH_ = {'Cm9': ([48, 55, 58, 62, 63], 36), 'Abmaj7': ([44, 51, 55, 60, 63], 32), 'Fm9': ([53, 56, 60, 63, 67], 41),
       'Ebmaj9': ([51, 55, 58, 62, 65], 39), 'Bbsus': ([46, 53, 58, 60, 65], 34), 'G7sus': ([55, 60, 62, 65, 67], 43),
       'Dbmaj7#11': ([49, 56, 60, 65, 67], 37), 'Cm': ([48, 55, 60, 63, 67], 36), 'Ab6/9': ([44, 51, 53, 58, 60], 32)}
S.score(CH_, [
    ([(0, 'Cm9'), (18.8, 'Abmaj7')], 0.5, .7),                                   # 00 intro
    ([(0, 'Fm9'), (17, 'Ebmaj9'), (33, 'Abmaj7')], 0.7, .7),                       # 01 energy
    ([(0, 'Cm9'), (26, 'Ebmaj9'), (40, 'Abmaj7'), (47, 'Bbsus')], 0.8, .8),        # 02 boltzmann
    ([(0, 'Dbmaj7#11'), (15, 'G7sus'), (29, 'Ebmaj9')], 0.6, .7),                  # 03 Z
    ([(0, 'Fm9'), (22, 'Dbmaj7#11'), (36, 'Abmaj7'), (50, 'Cm9')], 0.8, .8),       # 04 push & pull
    ([(0, 'Cm9'), (24, 'Abmaj7'), (40, 'Ebmaj9')], 1.0, .9),                       # 05 sculpt
    ([(0, 'Fm9'), (23, 'Cm9'), (39, 'Abmaj7'), (46, 'G7sus')], 0.8, .8),           # 06 metropolis
    ([(0, 'Ebmaj9'), (15, 'G7sus'), (22, 'Cm9'), (37, 'Abmaj7')], 0.9, .9),        # 07 langevin
    ([(0, 'Cm9'), (24, 'Fm9'), (40, 'Ebmaj9')], 1.0, 1.0),                         # 08 training
    ([(0, 'Abmaj7'), (24, 'Fm9'), (32, 'G7sus')], 0.7, .8),                        # 09 result
    ([(0, 'Dbmaj7#11'), (15, 'Cm9'), (32, 'Fm9'), (47, 'Abmaj7')], 0.6, .7),       # 10 catch
    ([(0, 'Ebmaj9'), (17, 'Abmaj7'), (34, 'Bbsus')], 1.1, 1.0),                    # 11 in the wild
    ([(0, 'Ab6/9'), (18, 'Ebmaj9')], 1.3, 1.0),                                    # 12 build it
    ([(0, 'Cm'), (8, 'Fm9'), (23, 'Dbmaj7#11'), (30, 'G7sus'), (38.5, 'Cm9')], 0.5, .7),  # 13 next
], [60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79], lead='vibes')
S.transitions()
c, m = S.cue, S.motion


def pour(sc, starts, gain=1.0, pan=0.0, spread=0.6):
    """A handful of marbles landing: one clack per marble (thinned), at the renderer's own pop-in times."""
    rng = np.random.default_rng(int(sc * 100 + len(starts)))
    for k, t0 in enumerate(sorted(starts)):
        if k % 3: continue
        c(sc, t0 + 0.12, 'clack', gain * rng.uniform(0.35, 0.8), pan + rng.uniform(-spread, spread), f=rng.uniform(2300, 4200))


# 00 cold open — the landscape draws in, marbles pour, roll and settle (speed from the real trajectories)
c(0, 0.5, 'riser', 0.6, dur=2.6, m0=36, m1=48)
pour(0, [3 + (k % 60) * 0.012 + (k // 60) * 0.07 for k in range(420)])
S.roll(0, 4.2, 18.3, lambda t: stage_speed(LANG, lambda u: NL * ease(prog(u, 4.2, 11)) ** 0.8, t), 1.2, seed=1)
c(0, 18.8, 'thud', 1.1, f=44); c(0, 18.8, 'chime', 1.3, m=72); c(0, 18.85, 'chime', 0.9, m=79)
for i in range(6): c(0, 20 + i * 0.3, 'tick', 0.7, -0.6 + 0.24 * i, f=1300 + 110 * i)

# 01 energy — four chains of promises, the network box, the probe marble sings its energy
for i in range(4): c(1, 1.6 + i * 1.3, 'clink', 0.8, -0.3 + 0.2 * i, f=1800 + 140 * i)
c(1, 9, 'whoosh', 1.0, dur=1.0, f0=2600, f1=300); c(1, 9.5, 'note', 1.0, m=67, lead='vibes'); c(1, 12, 'tick'); c(1, 13, 'tick', 0.7)
c(1, 14, 'riser', 0.4, dur=1.5, m0=48, m1=55); c(1, 17, 'tick'); c(1, 23.5, 'clack', 1.2, 0.4)
pp = np.array([[0.0, 0.0], [2.0, 0.0], [1.85, 0.77], [1.414, 1.414], [3.0, 3.0], [-2.0, 0.0]])
n96 = A['Eraw']['n']; Eg = np.array(A['Eraw']['v'])


def e_at(p):
    fi = np.clip((p[..., 0] + A['R']) / (2 * A['R']) * (n96 - 1), 0, n96 - 1.001); fj = np.clip((A['R'] - p[..., 1]) / (2 * A['R']) * (n96 - 1), 0, n96 - 1.001)
    i, j = fi.astype(int), fj.astype(int); u, v = fi - i, fj - j
    return (Eg[j, i] * (1 - u) + Eg[j, i + 1] * u) * (1 - v) + (Eg[j + 1, i] * (1 - u) + Eg[j + 1, i + 1] * u) * v


def probe_pos(t):
    u = np.clip((t - 24) / 13, 0, 1) * (len(pp) - 1); i = np.minimum(np.floor(u).astype(int), len(pp) - 2); w = ease(u - i)
    return pp[i] + (pp[i + 1] - pp[i]) * w[:, None]


S.roll(1, 24, 37.2, lambda t: np.linalg.norm(np.gradient(probe_pos(t), t, axis=0), axis=1), 0.7, 0.4, grains=False)
S.track(1, 23.6, 38, lambda t: 55 + 2.2 * np.minimum(e_at(probe_pos(t)), 14), gain=1.1, pan=0.4)
c(1, 33, 'chime', 0.9, m=74)

# 02 boltzmann — three demands, the exponential, two probes, Z fills in, softmax callback
for tt in (7, 13, 19): c(2, tt, 'tick', 1.0, 0.3, f=1500)
c(2, 26, 'chime', 1.1, m=75); c(2, 30, 'whoosh', 0.8, dur=1.0, f0=1800, f1=400)
c(2, 33, 'clack', 1.1, -0.5, f=2600); c(2, 33.08, 'clack', 1.0, 0.1, f=3300); c(2, 34, 'note', 1.0, m=79, lead='vibes')
c(2, 40, 'riser', 0.5, dur=1.2, m0=51, m1=63); c(2, 41.2, 'chime', 1.0, m=70)
for i in range(6): c(2, 47 + i * 0.09, 'key', 0.4, 0.2 * np.sin(i), f=2600 + 150 * (i % 3))
c(2, 50, 'chime', 0.8, m=79)

# 03 Z — grids of terms appear (bigger each time), the universe, then the ratio gift
for tt, g in ((1, 0.6), (8, 0.8), (10.5, 0.8), (15, 1.0)): c(3, tt, 'tick', g, f=1400)
c(3, 8, 'clack', 0.6, -0.5, f=3000)
for k in range(10): c(3, 10 + k * 0.012, 'clack', 0.25, -0.6 + 0.12 * k, f=2400 + 90 * k)
c(3, 15, 'thud', 0.9, f=40); c(3, 15.05, 'riser', 0.5, dur=2.5, m0=36, m1=60); c(3, 18, 'thud', 0.7, f=34)
c(3, 22, 'tick', 0.8); c(3, 29, 'clack', 1.0, -0.4); c(3, 29.06, 'clack', 0.9, 0.0, f=3400)
c(3, 32.5, 'scratch', 1.2, -0.1); c(3, 32.8, 'scratch', 1.1, 0.2); c(3, 33.5, 'chime', 1.1, m=75); c(3, 36, 'chime', 0.9, m=82)

# 04 push & pull — each line of the derivation, the p_θ highlight, then the pushes
for tt, n in ((1, 7), (8, 8), (15, 9), (22, 8), (29, 6)):
    for i in range(n): c(4, tt + i * 0.07, 'key', 0.35, 0.25 * np.sin(i), f=2500 + 180 * (i % 3))
c(4, 23.5, 'chime', 1.2, m=79, bright=1.2); c(4, 30.5, 'note', 0.9, m=75, lead='vibes')
c(4, 36, 'thud', 0.6, f=55); c(4, 37, 'chisel', 1.0, -0.2, f=380); c(4, 38, 'note', 1.0, 0.2, m=84, lead='vibes')
for i in range(8): c(4, 40.2 + i * 0.1, 'chisel', 0.45, -0.6 + 0.15 * i, f=330 + 30 * i)
for i, tt in enumerate((44.5, 47, 50.5)): c(4, tt, 'tick', 0.9, f=1300 + 200 * i)
m(4, 49, 54.1, lambda t: ease(prog(t, 49, 5)), 0.7, lo=200, hi=900)
c(4, 54, 'chime', 0.8, m=72)

# 05 sculpt — one chisel tap per real training step; loudness = how much the landscape actually moved
Es = np.array(A['sculpt']['E']); NF = len(Es) - 1; data = np.array(A['sculpt']['data']); g = np.array(A['sculpt']['g'])
chg = np.abs(np.diff(Es, axis=0)).mean(1); chg = chg / chg.max()
fmap = lambda t: NF * np.power(prog(t, 4, 35), 1.7)
ts = np.linspace(4, 39.2, 20000); fk = fmap(ts)
for k in range(1, NF + 1):
    i = np.searchsorted(fk, k)
    if i >= len(ts): break
    gap = np.interp(A['sculpt']['fant'][k], g, Es[k]).mean() - np.interp(data, g, Es[k]).mean()
    rate = float(np.gradient(fk, ts)[i])                                      # steps per second at this moment
    c(5, float(ts[i]), 'chisel', 1.1 * chg[k - 1] ** 0.8 / max(1.0, rate / 1.5) ** 0.5, float(np.sin(k * 2.1)) * 0.4, f=float(420 + 260 * np.clip(abs(gap), 0, 1)))
c(5, 2, 'tick', 0.8); c(5, 7, 'tick', 0.6)
c(5, 40, 'chime', 1.2, m=72); c(5, 40.1, 'chime', 0.9, m=79); c(5, 41.5, 'note', 0.9, m=84, lead='vibes')

# 06 metropolis — the slow steps get coin + verdict + roll; then the chain speeds into a texture
MET = A['metro']; MSLOW, T0, DT = 12, 9.0, 1.45
Efn = lambda x: 0.12 * x ** 4 - 1.1 * x ** 2 + 0.35 * x + 3.0 + 0.4 * np.sin(3 * x)
c(6, 1, 'tick', 0.8); c(6, 8, 'clack', 0.9, -0.6)
for k in range(MSLOW):
    x, prop, dE, u, acc = MET[k]; t0 = T0 + k * DT; pan = float(np.clip(prop / 3.3, -1, 1)) * 0.7
    c(6, t0, 'note', 0.5, pan, m=int(84 - 3 * np.clip(Efn(prop), 0, 6)), lead='vibes')
    if dE > 0: c(6, t0 + 0.18 * DT, 'coin', 0.9, pan)
    if acc:
        c(6, t0 + 0.42 * DT, 'tock', 1.0, pan, f=1250 + 60 * (k % 3))
        S.roll(6, t0 + 0.55 * DT, t0 + 0.85 * DT, lambda t, a=t0 + 0.55 * DT: np.sin(np.pi * np.clip((t - a) / (0.3 * DT), 0, 1)), 0.5 * min(1.0, abs(prop - x)), pan, grains=False)
    else:
        c(6, t0 + 0.42 * DT, 'thunk', 1.0, pan)
t1 = T0 + MSLOW * DT
kmap = lambda t: MSLOW + (len(MET) - MSLOW) * prog(t, t1, 16) ** 2
ts = np.linspace(t1, t1 + 16, 40000); kk = kmap(ts)
rate = np.gradient(kk, ts); thin = np.random.default_rng(66)
for k in range(MSLOW, len(MET)):
    i = np.searchsorted(kk, k)
    if i >= len(ts): break
    if thin.random() > min(1.0, 26 / max(rate[i], 1e-6)): continue       # ~26 sounds/s once the chain blurs
    x, prop, dE, u, acc = MET[k]; pan = float(np.clip(prop / 3.3, -1, 1)) * 0.7; g_ = 0.3 * min(1.0, 1.4 - 0.4 * min(1.0, rate[i] / 150))
    if acc: c(6, float(ts[i]), 'tock', g_, pan, f=1100 + 500 * float(np.clip((3 - Efn(prop)) / 3.5, 0, 1)))
    else: c(6, float(ts[i]), 'thunk', 0.4 * g_, pan)
c(6, 23, 'tick', 0.7); c(6, 30, 'whoosh', 0.5, dur=0.8)
c(6, 39, 'chime', 1.0, m=75); c(6, 46, 'thud', 0.6, f=48); c(6, 48, 'tick', 0.7)

# 07 langevin — left: marbles collapse (no jiggle); right: the same marbles, jiggling into shape
c(7, 1, 'tick'); c(7, 8, 'tick'); pour(7, [8 + (k % 40) * 0.02 for k in range(420)], 0.8, -0.45, 0.3)
S.roll(7, 9, 20.2, lambda t: stage_speed(GD, lambda u: NL * np.power(prog(u, 9, 11), 0.75), t), 1.0, -0.45, seed=2)
c(7, 16, 'thud', 0.7, -0.45, f=70); c(7, 16.05, 'clack', 0.8, -0.45, f=2000)
c(7, 22, 'tick', 0.9, 0.45); pour(7, [22 + (k % 40) * 0.02 for k in range(420)], 0.8, 0.45, 0.3)
S.roll(7, 23, 44, lambda t: stage_speed(LANG, lambda u: NL * np.power(prog(u, 23, 14), 0.75), t), 1.1, 0.45, seed=3)
c(7, 37, 'chime', 0.8, m=74); c(7, 45, 'note', 1.0, m=79, lead='vibes'); c(7, 45.2, 'note', 0.8, m=86, lead='vibes')

# 08 training — the ground shifts between snapshots; each swap of fantasies is a small scatter of marbles
NS = len(A['train']) - 1; trf = lambda t: NS * ease(prog(t, 6, 38))
tt = np.linspace(6, 44, 20000); tf = trf(tt)
for s in range(NS):
    i0, i1 = np.searchsorted(tf, s), np.searchsorted(tf, min(s + 1, NS - 1e-6))
    ta, tb = float(tt[min(i0, len(tt) - 1)]), float(tt[min(i1, len(tt) - 1)])
    m(8, ta, tb, lambda t, s=s: ease(np.clip(trf(t) - s, 0, 1)), 0.7, lo=90, hi=420, tone=(36 + 3 * s, 39 + 3 * s), pan=-0.4)
    ih = np.searchsorted(tf, s + 0.5)
    if ih < len(tt):
        for k in range(8): c(8, float(tt[ih]) + k * 0.03, 'clack', 0.3, -0.7 + 0.12 * k, f=2400 + 160 * k)
for i in range(4): c(8, 8.5 + i * 1.2 + (38 if i == 2 else 0), 'tick', 0.8, 0.4, f=1400 + 120 * i)
c(8, 28, 'whoosh', 0.6, 0.4, dur=0.8); c(8, 40, 'chime', 0.9, m=79); c(8, 47, 'note', 0.8, m=75, lead='vibes')

# 09 result — two densities, the bridge, the marbles land, the scores, the uneven valleys
c(9, 1, 'chime', 0.7, -0.5, m=72); c(9, 8, 'chime', 0.7, 0.5, m=67); c(9, 10, 'clink', 0.8, 0.5)
pour(9, [16 + (k % 60) * 0.02 for k in range(420)], 0.7, -0.5, 0.3)
for i in range(4): c(9, 24.5 + i * 0.8, 'tick', 0.9, f=1500 - 120 * i)
SH = A['share']
for j in range(8): c(9, 32 + j * 0.06, 'note', 0.7, -0.6 + j * 0.05, m=int(72 - round((SH[j] - 0.125) * 90)), lead='vibes')

# 10 catch — each valley's share is pitched by its depth; the long chain rolls and hops each ridge
for j, s_ in enumerate(SH): c(10, 1.8 + j * 0.25, 'note', 0.9, float(np.cos(j * np.pi / 4)) * 0.6, m=int(72 - round((s_ - 0.125) * 90)), lead='vibes')
c(10, 8, 'tick', 0.8)
chf = lambda t: np.clip((t - 14) / 18, 0, 1) ** 1.4 * (len(CH) - 1)
tt = np.linspace(14, 32, 30000); fi = chf(tt); step = np.linalg.norm(np.diff(CH, axis=0), axis=1)
S.roll(10, 14, 32, lambda t: step[np.minimum(chf(t).astype(int), len(step) - 1)] * np.gradient(chf(t), t), 0.8, -0.3, seed=4)
sec = (np.round(np.arctan2(CH[:, 1], CH[:, 0]) / (np.pi / 4)).astype(int)) % 8; far = np.linalg.norm(CH, axis=1) > 1.4
for i in range(1, len(CH)):
    if sec[i] != sec[i - 1] and far[i]:
        j = np.searchsorted(fi, i)
        if j < len(tt): c(10, float(tt[j]), 'chime', 1.2, -0.3, m=79 + (sec[i] % 3) * 2, bright=1.3); c(10, float(tt[j]), 'whoosh', 0.5, -0.3, dur=0.5, f0=800, f1=3000)
c(10, 24, 'thud', 0.5, f=60)
for tt_, f_ in ((32, 1300), (40, 1200), (42.5, 1100)): c(10, tt_, 'tick', 0.9, 0.3, f=f_)

# 11 in the wild — five cards, five voices
for i, (tt_, mm) in enumerate(((1, 72), (9, 75), (17, 79), (26, 77), (34, 84))):
    c(11, tt_, 'chime', 0.9, -0.5 + 0.25 * i, m=mm)
for i in range(4): c(11, 17.6 + i * 0.12, 'note', 0.7, 0.1 * i, m=[72, 75, 79, 84][i], lead='vibes')

# 12 build it — keyboard clicks as code lines appear; recap ticks
for i in range(15): c(12, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): c(12, 18.5 + i * 1.6, 'tick', 0.8, f=1400)

# 13 hook — the sampler, Z struck out and vanishing, the arrows appear; title
S.roll(13, 2, 22, lambda t: stage_speed(LANG[:, :160], lambda u: NL * np.clip((u - 2) / 20, 0, 1) ** 0.75, t), 0.7, -0.4, seed=5)
c(13, 1, 'tick'); c(13, 8, 'tick'); c(13, 11, 'tick', 0.7)
c(13, 17, 'scratch', 1.3); c(13, 17.3, 'whoosh', 0.9, dur=1.2, f0=3500, f1=200); c(13, 18, 'note', 0.9, m=63, lead='vibes')
for i in range(12): c(13, 23 + i * 0.11, 'note', 0.35, -0.6 + 0.1 * i, m=[60, 63, 67, 70, 72, 75, 79, 82, 84, 87, 91, 94][i], lead='vibes')
c(13, 30, 'chime', 0.8, m=79)
c(13, 38.5, 'thud', 1.1, f=45); c(13, 38.5, 'chime', 1.5, m=60); c(13, 38.55, 'chime', 1.2, m=67); c(13, 39.3, 'chime', 1.0, m=72); c(13, 40.5, 'tick'); c(13, 41, 'chime', 1.0, m=84)

S.render()
