"""Episode 02 soundtrack — A minor / C, 'the machine that writes'. Music-box lead + a soft clock pulse
for the autoregressive loop; typewriter keys for every letter that appears; dice rattles for sampling.

    python src/ep02_audio.py      # → ep02-audio.mp3
"""
import numpy as np

from audio_lib import Score, ease, eout, prog, probe

HTML = 'ep02-autoregressive.html'
VOC = '.abcdefghijklmnopqrstuvwxyz'
(durs, demo, first3, fast, emb_y, ctx_copy, open_len, e3, n_ev) = probe(HTML, [
    'SC.map(s => s.dur)', 'DEMO.name', 'NAMES.slice(0, 3)', 'FAST_NAMES', 'EMB.map(p => p[1])',
    'CTX_NAMES.map(l => l.map(n => NAMESET.has(n)))', 'OPEN_TEXT.length', 'E3', 'EVENTS.length'])
S = Score('ep02-audio', durs, seed=22)
AM = [57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81]                  # A minor pentatonic, for letters
letter_m = lambda c: AM[(VOC.index(c) * 3) % len(AM)] if c in VOC else 69

CH = {'Am9': ([45, 52, 55, 59, 60], 33), 'Fmaj7': ([53, 57, 60, 64, 69], 41), 'Cmaj7': ([48, 55, 59, 64, 67], 36), 'G6': ([55, 59, 62, 64, 71], 43),
      'Dm9': ([50, 57, 60, 64, 65], 38), 'Esus': ([52, 57, 59, 64, 69], 40), 'E7': ([52, 56, 59, 62, 68], 40), 'Bbmaj7': ([46, 53, 57, 62, 65], 34)}
S.score(CH, [
    ([(0, 'Am9'), (15.6, 'Fmaj7')], 0.5, .7),
    ([(0, 'Cmaj7'), (13, 'Dm9'), (20, 'E7'), (28.5, 'Am9')], 0.6, .7),
    ([(0, 'Am9'), (14, 'Fmaj7'), (28, 'Cmaj7'), (35, 'G6')], 0.7, .8),
    ([(0, 'Fmaj7'), (15.5, 'Dm9'), (29, 'Cmaj7'), (37, 'G6')], 0.7, .8),
    ([(0, 'Am9'), (14, 'Fmaj7'), (21, 'Cmaj7')], 0.6, .8),
    ([(0, 'Dm9'), (14, 'Fmaj7'), (28, 'Cmaj7'), (36, 'G6')], 0.7, .8),
    ([(0, 'Am9'), (10, 'Fmaj7'), (23, 'Cmaj7'), (37, 'Esus')], 0.6, .8),
    ([(0, 'Fmaj7'), (14, 'Esus'), (21, 'E7'), (35, 'Am9')], 0.5, .7),
    ([(0, 'Cmaj7'), (17, 'Fmaj7'), (31, 'Dm9'), (46, 'G6')], 0.8, .9),
    ([(0, 'Am9'), (11, 'Dm9'), (18, 'Bbmaj7'), (25, 'Am9')], 0.6, .8),
    ([(0, 'Fmaj7'), (16, 'G6'), (25, 'Cmaj7'), (33, 'Fmaj7')], 0.9, 1.0),
    ([(0, 'Cmaj7'), (16, 'G6')], 1.1, 1.0),
    ([(0, 'Am9'), (21, 'Fmaj7'), (29, 'Dm9'), (35.5, 'E7')], 0.5, .6),
], AM, lead='musicbox')
S.transitions()

# 00 cold open — the machine types, one letter per tick of its clock
for k in range(1, open_len + 1): S.cue(0, 0.8 + k / 3.4, 'key', 0.9, -0.4 + 0.8 * k / open_len, f=2900 + 300 * (k % 3))
S.pulse(0, 0.8, 13.8, 1 / 3.4 * 2, 45, 0.9, accent=4)
S.motion(0, 15, 16.7, lambda t: ease(prog(t, 15, 1.6)), 0.5, lo=300, hi=1400)
S.cue(0, 15.6, 'chime', 1.3, m=76); S.cue(0, 15.6, 'thud', 0.35, f=45)
for i in range(5): S.cue(0, 18.5 + i * 0.35, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 120 * i)

# 01 the problem — slot-machine letters settle into 'hello'; the count explodes
S.cue(1, 0.8, 'tick')
for i in range(5):
    S.cue(1, 7 + i * 0.05, 'rattle', 0.6, -0.5 + 0.25 * i, dur=2.3 + i * 0.6)
    S.cue(1, 9.3 + i * 0.6, 'key', 1.0, -0.5 + 0.25 * i, f=2600); S.cue(1, 9.3 + i * 0.6, 'note', 0.6, lead='musicbox', m=letter_m('hello'[i]))
S.motion(1, 13, 14.1, lambda t: ease(prog(t, 13, 1)), 0.5); S.cue(1, 13.5, 'chime', 0.9, m=81)
S.motion(1, 20.5, 22.6, lambda t: ease(prog(t, 20.5, 2)), 1.0, lo=150, hi=900, tone=(33, 57)); S.cue(1, 22.5, 'thud', 0.7, f=40)
S.motion(1, 28.5, 30.1, lambda t: ease(prog(t, 28.5, 1.5)), 0.6, lo=400, hi=1800); S.cue(1, 29.5, 'chime', m=76)

# 02 the guessing game — bars rise, the pick lands (brighter when the guess is more certain)
T2, P2 = [5, 7.5, 14, 17.5, 21], [0.06, 0.30, 0.25, 0.45, 0.80]
for i, (t0, p) in enumerate(zip(T2, P2)):
    for k in range(6): S.cue(2, t0 + k * 0.08 + 0.2, 'tick', 0.35, -0.5 + 0.2 * k, f=900 + 150 * k)
    S.cue(2, t0 + 2.2, 'note', 0.7 + 0.6 * p, lead='musicbox', m=64 + int(round(p * 20))); S.cue(2, t0 + 2.2, 'key', 0.7)
S.motion(2, 28.5, 29.6, lambda t: eout(prog(t, 28.5, 1)), 0.4); S.cue(2, 35, 'chime', 1.2, m=81)

# 03 the chain rule — each zoom is a step down a staircase of nested slices
for r, t0 in enumerate([3, 9, 15.5, 19, 22.5]):
    S.motion(3, t0 + 0.2, t0 + 1.05, lambda t, t0=t0: eout(prog(t, t0 + 0.2, 0.8)), 0.55, lo=300, hi=1500, pan=-0.3 + 0.15 * r)
    S.cue(3, t0, 'note', 0.9, lead='musicbox', m=[81, 76, 74, 69, 64][r])
S.cue(3, 29, 'chime', 1.1, m=76); S.cue(3, 31, 'tick'); S.cue(3, 37, 'chime', 1.2, m=81)

# 04 the one question — the loop: model → bars → sample flies back → append (clock pulse underneath)
S.cue(4, 5, 'tick'); S.cue(4, 8, 'tick', f=1200)
S.pulse(4, 14, 24, 3.2 / 4, 45, 1.0, accent=4)
for c in range(3):
    t0 = 14 + 3.2 * c
    S.motion(4, t0, t0 + 3.2, lambda t, t0=t0: ease(((t - t0) % 3.2) / 3.2), 0.7, lo=500, hi=2000, tone=(64, 76))
    S.cue(4, t0 + 3.15, 'key', 1.0, f=2600); S.cue(4, t0 + 3.15, 'note', 0.6, lead='musicbox', m=letter_m('lo.'[c]))
S.cue(4, 21, 'chime', m=79)

# 05 counting — the window slides; each counted pair flies into the table; then the table floods
S.cue(5, 1, 'tick')
for i in range(60):
    if i % 3 == 0: S.cue(5, 1.2 + i * 0.02, 'key', 0.25, (i % 4 - 1.5) / 2, f=3400)
S.cue(5, 7, 'tick', f=1200)
for k in range(int(5 * 1.2)): S.cue(5, 9 + k / 1.2, 'key', 0.7, f=2400)
ev = []
for w in first3:
    s = '.' + w + '.'; ev += [(s[i], s[i + 1]) for i in range(len(s) - 1)]
for k, (a, b) in enumerate(ev[:e3]):
    S.cue(5, 14 + 7 * (k + 1) / e3, 'drop', 0.9, (VOC.index(b) / 26 - 0.5), m=letter_m(b))
S.motion(5, 21, 27.1, lambda t: ease(prog(t, 21, 6)), 1.0, lo=300, hi=3000, tone=(57, 69))
S.cue(5, 28, 'tick'); S.cue(5, 28.3, 'note', lead='musicbox', m=69); S.cue(5, 36, 'chime', m=76); S.cue(5, 44, 'chime', 1.2, m=81)

# 06 generate — per letter: look up the row, the dice spin and settle, the letter is appended
name = demo + '.'
Sk = lambda k: 3 if k == 0 else 10 + (k - 1) * 2.6
Dk = lambda k: 7 if k == 0 else 2.6
for k, ch in enumerate(name):
    s0, d = Sk(k), Dk(k)
    S.cue(6, s0, 'tick', 0.6, f=1100)
    S.cue(6, s0 + d - 2.2, 'rattle', 1.0, dur=1.6)
    S.cue(6, s0 + d - 0.6, 'note', 1.0, lead='musicbox', m=letter_m(ch)); S.cue(6, s0 + d, 'key', 0.8)
S.cue(6, 23, 'chime', 1.1, m=81)
for i, nm in enumerate(fast):
    for j in range(len(nm)): S.cue(6, 29.5 + i * 0.95 + j / 14, 'key', 0.35, (i % 5 - 2) / 4, f=3000 + 200 * (j % 3))
    S.cue(6, 29.5 + i * 0.95, 'note', 0.35, lead='musicbox', m=letter_m(nm[0]))
S.cue(6, 37, 'thud', 0.45, f=50)

# 07 more context — names appear; copies get flagged; the table of contexts explodes
for c in range(3):
    for i in range(8):
        S.cue(7, 7 + c * 1.5 + i * 0.18, 'key', 0.4, -0.6 + 0.6 * c, f=3000)
        if ctx_copy[c][i]: S.cue(7, 14.5 + i * 0.1, 'stamp', 0.5, -0.6 + 0.6 * c)
for i in range(4):
    S.cue(7, 21.5 + i * 1.3, 'thud', 0.4 + 0.25 * i, f=70 - 8 * i); S.cue(7, 21.5 + i * 1.3, 'note', 0.6, lead='marimba', m=57 + 5 * i)
S.cue(7, 28, 'tick'); S.cue(7, 35, 'chime', m=76)

# 08 neural net — letters land in embedding space; vowels cluster; logits → softmax; surprise
for j, y in enumerate(emb_y): S.cue(8, 1.5 + j * 0.08 + 0.1, 'drop', 0.55, (j / 26 - 0.5), m=S.snap(y, -0.8, 0.8, AM))
S.cue(8, 3, 'tick'); S.cue(8, 9, 'chime', m=76)
S.cue(8, 17, 'tick'); S.cue(8, 24, 'tick')
S.motion(8, 25, 27.1, lambda t: ease(prog(t, 25, 2)), 0.7, lo=300, hi=2200, tone=(57, 69))
S.motion(8, 28, 29.6, lambda t: ease(prog(t, 28, 1.5)), 0.6, lo=2200, hi=600, tone=(69, 64))
S.motion(8, 31.5, 33.6, lambda t: ease(prog(t, 31.5, 2)), 0.5, lo=300, hi=1200); S.cue(8, 33, 'tick')
S.cue(8, 38, 'note', lead='musicbox', m=81); S.cue(8, 40, 'note', 1.0, lead='glass', m=58)
for i in range(5): S.cue(8, 46.5 + i * 0.35, 'drop', 0.8, m=[64, 67, 69, 72, 76][i])
S.cue(8, 49, 'chime', m=79)

# 09 temperature — the τ slider moves; a tone follows it (cold = low, hot = high)
TK = [[0, 1], [8, 1], [11, 0.3], [15, 0.3], [18, 2.5], [22, 2.5], [25, 1]]
def tau(t):
    t = np.atleast_1d(t); out = np.full_like(t, 1.0)
    for i in range(len(TK) - 1):
        m = (t >= TK[i][0]) & (t < TK[i + 1][0]); u = ease((t[m] - TK[i][0]) / (TK[i + 1][0] - TK[i][0])); out[m] = TK[i][1] + (TK[i + 1][1] - TK[i][1]) * u
    return out
S.track(9, 6.5, 26, lambda t: 64 + 9 * np.log(tau(t)), None, 0.8)
S.cue(9, 7, 'tick')
for c in range(3):
    for i in range(6): S.cue(9, 21.5 + c * 0.6 + i * 0.15, 'key', 0.35, -0.5 + 0.5 * c, f=[2200, 3000, 3900][c])
S.cue(9, 24, 'tick')

# 10 scaling up — tokens merge, the context window explodes, attention rows sweep
for t0 in [1, 8, 16, 25]: S.cue(10, t0, 'note', 0.8, lead='musicbox', m=76 if t0 < 16 else 79)
S.motion(10, 3, 4.6, lambda t: ease(prog(t, 3, 1.5)), 0.5, lo=600, hi=1600)
S.motion(10, 9, 13.1, lambda t: ease(prog(t, 9, 4)), 1.0, lo=200, hi=3000, tone=(45, 69))
for k in range(int((33 - 16) * 1.6)): S.cue(10, 16 + k / 1.6, 'tick', 0.3, f=1300 + 90 * (k % 8))
S.cue(10, 33.5, 'chime', 1.2, m=81)

# 11 build it — code types itself; recap lines land
for i in range(21): S.cue(11, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
S.cue(11, 16, 'tick')
for i in range(4): S.cue(11, 17 + i * 2.2, 'tick', 0.8, f=1400)

# 12 hook — a picture painted one pixel at a time (a tick per row), then: can a machine compress?
for row in range(28): S.cue(12, 1 + 18 * row * 28 / 784, 'key', 0.5, -0.6 + 1.2 * row / 27, f=1600 + 60 * row)
S.cue(12, 7, 'thud', 0.6, f=48); S.cue(12, 14, 'chime', 0.9, m=69)
S.motion(12, 29, 35, lambda t: 0.5 + 0.5 * np.sin((t - 29) * 1.3 - np.pi / 2), 0.5, lo=300, hi=1400, tone=(62, 69))
S.cue(12, 33.5, 'riser', 0.9, dur=2.0, m0=45, m1=57)
S.cue(12, 35.5, 'thud', 1.0, f=45); S.cue(12, 35.5, 'chime', 1.4, m=69); S.cue(12, 35.55, 'chime', 1.0, m=76)
S.cue(12, 37, 'tick'); S.cue(12, 37.5, 'chime', 1.0, m=84)

S.render()
