"""Episode 05 soundtrack — F# minor, 'a forger and a detective'. Kalimba lead, a heartbeat pulse under the
game; real training runs are sonified: each newly found mode chimes, each mode-collapse hop jumps.

    python src/ep05_audio.py      # → ep05-audio.mp3
"""
import numpy as np

from audio_lib import Score, ease, eout, prog, probe

(durs, modes, hop_dom, epochs) = probe('ep05-gans.html', ['SC.map(s => s.dur)', 'A.healthy.map(s => s.modes)',
                                                           'A.hop.map((s, i) => dominant("hop", i).k)', 'EPOCHS'])
S = Score('ep05-audio', durs, seed=55)
FM = [54, 57, 59, 61, 64, 66, 69, 71, 73, 76, 78]
MODE_NOTE = [66, 69, 71, 73, 76, 78, 81, 83]
STAMP = 0.28

CH = {'F#m9': ([54, 61, 64, 68, 69], 42), 'Dmaj7': ([50, 57, 61, 66, 69], 38), 'E6': ([52, 59, 61, 64, 68], 40), 'C#7sus': ([49, 56, 59, 61, 66], 37),
      'C#7': ([49, 53, 56, 59, 65], 37), 'Bm9': ([47, 54, 57, 61, 62], 35), 'Amaj9': ([45, 52, 56, 59, 61], 33), 'F#m': ([54, 61, 66, 69, 73], 42)}
S.score(CH, [
    ([(0, 'F#m9'), (13, 'Dmaj7')], 0.5, .7),
    ([(0, 'Bm9'), (14, 'F#m9'), (28, 'C#7sus')], 0.5, .7),
    ([(0, 'F#m9'), (16, 'Dmaj7'), (32, 'Amaj9')], 0.6, .8),
    ([(0, 'Bm9'), (16, 'E6'), (30, 'Dmaj7'), (39, 'Amaj9')], 0.6, .8),
    ([(0, 'F#m9'), (15, 'Bm9'), (31, 'C#7sus'), (39, 'Amaj9')], 0.6, .8),
    ([(0, 'F#m9'), (16, 'Dmaj7'), (32, 'E6'), (40, 'Amaj9')], 0.8, .9),
    ([(0, 'Amaj9'), (15, 'Dmaj7'), (23, 'F#m9'), (39, 'E6')], 0.7, .9),
    ([(0, 'Bm9'), (16, 'C#7'), (32, 'F#m9')], 0.5, .6),
    ([(0, 'F#m9'), (16, 'C#7sus'), (24, 'Bm9'), (32, 'E6')], 0.5, .7),
    ([(0, 'Dmaj7'), (23, 'Bm9')], 0.7, .8),
    ([(0, 'Amaj9'), (18, 'E6')], 1.1, 1.0),
    ([(0, 'F#m'), (15, 'Dmaj7'), (23, 'Bm9'), (31, 'C#7sus'), (38.5, 'F#m9')], 0.5, .6),
], FM, lead='kalimba', lead_gain=0.06)
S.transitions()

# 00 cold open — digits appear epoch by epoch (each epoch a step up); the title
for e in range(1, epochs + 1): S.cue(0, 0.5 + 10 * e / epochs, 'tick', 0.45, -0.5 + e / epochs, f=900 + 30 * e)
S.cue(0, 13, 'chime', 1.3, m=73); S.cue(0, 13, 'thud', 0.4, f=42)
for i in range(5): S.cue(0, 16 + i * 0.35, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 120 * i)

# 01 two players — the forger appears, then the detective; the duel's heartbeat starts
S.cue(1, 7, 'note', lead='kalimba', m=66, pan=-0.5); S.cue(1, 14, 'note', lead='kalimba', m=73, pan=0.5)
S.pulse(1, 14, 36, 0.75, 42, 0.9, accent=2)
S.cue(1, 21, 'chime', 0.9, m=78); S.cue(1, 28, 'tick')

# 02 the judge's perfect strategy
S.cue(2, 0.8, 'tick'); S.cue(2, 8, 'tick'); S.cue(2, 16, 'note', lead='kalimba', m=69, pan=-0.2); S.cue(2, 16.1, 'note', lead='kalimba', m=73, pan=0.2)
S.cue(2, 24, 'tick'); S.cue(2, 26, 'tick'); S.cue(2, 32, 'chime', m=78)

# 03 what the forger minimises — pg slides onto pdata; the JSD tone falls to rest
S.cue(3, 0.8, 'tick'); S.cue(3, 8, 'tick'); S.cue(3, 16, 'chime', 0.9, m=76)
S.motion(3, 30, 37.1, lambda t: ease(prog(t, 30, 7)), 0.7, lo=1200, hi=300)
S.track(3, 24, 38, lambda t: 61 + 14 * (1 - ease(prog(t, 30, 7))) ** 2, None, 0.8); S.cue(3, 37.2, 'chime', 1.1, m=78)

# 04 how the forger learns — the judge's field; gradient arrows; flat vs steep losses
S.motion(4, 8, 9.1, lambda t: eout(prog(t, 8, 1)), 0.5, lo=200, hi=900, pan=-0.4); S.cue(4, 15, 'note', lead='kalimba', m=73)
S.cue(4, 23, 'tick'); S.cue(4, 31, 'tick'); S.cue(4, 33, 'note', 0.8, lead='glass', m=49); S.cue(4, 39, 'chime', m=78)

# 05 watch a real GAN train — every newly found mode chimes (real run)
n = len(modes); fi = lambda t: (n - 1) * ease(prog(t, 3, 40))
tt = np.linspace(3, 43, 4000); idx = np.floor(fi(tt)).astype(int)
S.pulse(5, 3, 43, 0.75, 42, 0.8, accent=2)
prev = modes[0]
for k in range(1, len(tt)):
    if idx[k] != idx[k - 1]:
        m = modes[idx[k]]
        if m > prev:
            for j in range(prev, m): S.cue(5, tt[k] + 0.08 * (j - prev), 'note', 1.0, -0.6 + 0.17 * j, lead='kalimba', m=MODE_NOTE[j])
        elif m < prev: S.cue(5, tt[k], 'note', 0.5, lead='glass', m=54)
        prev = m
S.motion(5, 3, 43, lambda t: fi(t), 0.5, lo=300, hi=1500)
S.cue(5, 43, 'chime', 1.2, m=78)

# 06 the same game on digits — epochs tick upward; the VAE; episode 4's judges report
for e in range(1, epochs + 1): S.cue(6, 1 + 13 * e / epochs, 'tick', 0.4, -0.5 + e / epochs, f=900 + 30 * e)
S.cue(6, 15, 'note', 0.8, lead='glass', m=66); S.cue(6, 23, 'tick')
for i in range(6): S.cue(6, 23.5 + i * 0.4, 'note', 0.6, 0.4, lead='kalimba', m=FM[3 + i])

# 07 mode collapse — the crowd piles onto a blob … the judge darkens it … the crowd hops
n7 = len(hop_dom); fi7 = lambda t: (n7 - 1) * ease(prog(t, 2, 38))
tt = np.linspace(2, 40, 4000); idx = np.floor(fi7(tt)).astype(int); prev = hop_dom[0]
S.pulse(7, 2, 40, 0.75, 42, 0.7, accent=2)
for k in range(1, len(tt)):
    if idx[k] != idx[k - 1] and hop_dom[idx[k]] != prev:
        S.cue(7, tt[k], 'whoosh', 0.6, -0.7 + 0.2 * hop_dom[idx[k]], dur=0.5, f0=600, f1=2400)
        S.cue(7, tt[k] + 0.1, 'note', 0.7, -0.7 + 0.2 * hop_dom[idx[k]], lead='kalimba', m=MODE_NOTE[hop_dom[idx[k]]]); prev = hop_dom[idx[k]]
S.cue(7, 25 + STAMP, 'stamp', 1.0); S.cue(7, 32, 'tick')

# 08 a game is not a hill — the spiral (pans as it circles, rises as it grows); JSD flat vs W slope
eta = 0.18; sp = [(1.0, 0.2)]
for _ in range(70): x, y = sp[-1]; sp.append((x - eta * y, y + eta * x))
sp = np.array(sp); L = len(sp); ts = np.linspace(16, 23, 1400); k = np.clip((L * ease(prog(ts, 16, 7))).astype(int) - 1, 0, L - 1)
rad = np.hypot(sp[k, 0], sp[k, 1]); S.cue(8, 8, 'tick')
for a, b in zip(range(0, 1400, 100), range(100, 1500, 100)):
    seg_t = ts[a:b]
    if len(seg_t) < 2: continue
    S.track(8, seg_t[0], seg_t[-1], lambda t, a=a, b=b: np.interp(t, ts[a:b], 57 + 8 * rad[a:b]), None, 0.8, pan=float(np.clip(sp[k[a], 0] / 3, -0.9, 0.9)), fade=0.05)
S.cue(8, 16, 'tick', f=1200)
S.track(8, 24, 32, lambda t: np.full_like(t, 57.0), lambda t: 0.6, 0.6)            # JSD: flat, no gradient
S.motion(8, 32, 33.1, lambda t: eout(prog(t, 32, 1)), 0.6, lo=300, hi=1600, tone=(57, 69)); S.cue(8, 33, 'chime', m=78)

# 09 GANs in the wild
for i, t0 in enumerate([1, 8, 15]): S.cue(9, t0, 'note', 0.9, -0.3 + 0.3 * i, lead='kalimba', m=[69, 73, 76][i])
S.cue(9, 23, 'note', 0.9, lead='glass', m=54); S.cue(9, 23, 'thud', 0.3, f=45)

# 10 build it
for i in range(14): S.cue(10, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): S.cue(10, 18.5 + i * 1.6, 'tick', 0.8, f=1400)

# 11 hook → flows — a GAN can't run backwards ✕ … a warp that can
S.cue(11, 0.6, 'note', 0.8, lead='kalimba', m=66); S.cue(11, 8, 'thud', 0.5, f=50); S.cue(11, 8.3, 'note', 0.8, lead='glass', m=55)
S.cue(11, 15, 'tick'); S.motion(11, 23, 29.1, lambda t: ease(prog(t, 23, 6)), 1.0, tone=(54, 66)); S.cue(11, 25, 'tick')
S.cue(11, 31, 'note', lead='kalimba', m=78)
S.cue(11, 36.5, 'riser', 0.9, dur=2.0, m0=42, m1=54)
S.cue(11, 38.5, 'thud', 1.0, f=42); S.cue(11, 38.5, 'chime', 1.4, m=61); S.cue(11, 38.55, 'chime', 1.0, m=69)
S.cue(11, 40, 'tick'); S.cue(11, 40.5, 'chime', 1.0, m=78)

S.render()
