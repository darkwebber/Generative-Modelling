"""Episode 03 soundtrack — E minor, 'squeezing the world into a few numbers'. Glass lead; the latent
point sings as it wanders; a harp-like cascade as the decoded grid appears.

    python src/ep03_audio.py      # → ep03-audio.mp3
"""
import numpy as np

from audio_lib import Score, ease, eout, prog, probe

(durs, recon) = probe('ep03-autoencoders-vae.html', ['SC.map(s => s.dur)', 'A.histAE.map(h => h.recon)'])
S = Score('ep03-audio', durs, seed=33)
EM = [59, 62, 64, 66, 67, 71, 74, 76, 78, 79, 83]
DIGIT_NOTE = {d: EM[(d * 3) % len(EM)] for d in range(10)}

CH = {'Em9': ([52, 59, 62, 66, 67], 40), 'Cmaj7': ([48, 55, 59, 64, 67], 36), 'G6': ([55, 59, 62, 64, 71], 43), 'D/F#': ([54, 57, 62, 66, 69], 42),
      'Am9': ([57, 60, 64, 67, 71], 45), 'Bsus': ([59, 64, 66, 71, 76], 47), 'B7': ([59, 63, 66, 69, 75], 47), 'Cmaj9#11': ([48, 55, 59, 62, 66], 36)}
S.score(CH, [
    ([(0, 'Em9'), (15.6, 'Cmaj9#11')], 0.5, .7),
    ([(0, 'Em9'), (13, 'Am9'), (28, 'Bsus')], 0.5, .7),
    ([(0, 'Cmaj7'), (13, 'G6'), (35, 'D/F#')], 0.7, .8),
    ([(0, 'Em9'), (15, 'Cmaj9#11'), (31, 'G6')], 0.8, .9),
    ([(0, 'Am9'), (15, 'B7'), (31, 'Em9')], 0.5, .6),
    ([(0, 'Cmaj7'), (16.5, 'B7'), (24, 'G6'), (40, 'D/F#')], 0.6, .8),
    ([(0, 'Em9'), (23, 'Am9'), (31, 'Cmaj7'), (46, 'G6')], 0.6, .8),
    ([(0, 'Am9'), (15, 'Cmaj9#11'), (23, 'G6')], 0.6, .8),
    ([(0, 'Em9'), (15, 'Cmaj9#11'), (24, 'G6'), (40, 'D/F#')], 0.9, 1.0),
    ([(0, 'Am9'), (16, 'B7'), (31, 'Em9')], 0.5, .6),
    ([(0, 'Cmaj7'), (18, 'G6')], 1.1, 1.0),
    ([(0, 'Em9'), (14, 'Am9'), (28, 'B7'), (36, 'Em9')], 0.5, .6),
], EM, lead='glass')
S.transitions()

# 00 cold open — two numbers drive the drawing; a tone follows the code as the digits melt
z1 = lambda t: 1.7 * np.sin(0.33 * t + 0.4)
S.track(0, 0.4, 15.5, lambda t: 64 + 5 * z1(t) + 2.5 * np.sin(0.47 * t + 1.3), lambda t: eout(prog(t, 0.3, 1)) * (1 - ease(prog(t, 14.5, 1))), 1.0)
S.cue(0, 15.6, 'chime', 1.3, m=76); S.cue(0, 15.6, 'thud', 0.35, f=41)
for i in range(5): S.cue(0, 18.5 + i * 0.35, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 120 * i)

# 01 the compression game — reading 784 numbers vs 'a seven, leaning right'; static vs structure
for k in range(int(11.5 * 2)): S.cue(1, 1 + k / 2, 'key', 0.22, 0.3, f=3500)
S.cue(1, 7, 'chime', 0.9, m=79)
n = int(14.5 * 44100); t = np.arange(n) / 44100 + 13
S.add(S.sfx, S.at(1, 13), (S.band(S.noise(n), lo=2000, hi=7000) * np.clip((t - 13) / .8, 0, 1) * (1 - ease(prog(t, 27, 1)))).astype(np.float32), 0.012)
for i in range(3): S.cue(1, 20 + 0.05 * i, 'note', 0.6, lead='glass', m=[71, 74, 78][i])
S.motion(1, 28, 29.6, lambda t: ease(prog(t, 28, 1.5)), 0.9, lo=3000, hi=300, tone=(76, 52)); S.cue(1, 30, 'chime', m=83)

# 02 the autoencoder — each digit squeezes through the keyhole and is rebuilt; loss falls
S.cue(2, 1, 'tick')
for c in range(int((34.5 - 2) / 3.5) + 1):
    t0 = 2 + 3.5 * c
    S.motion(2, t0, t0 + 1.6, lambda t, t0=t0: ease(prog(t, t0, 1.5)), 0.45, lo=2500, hi=400, tone=(71, 59), pan=-0.3)
    S.cue(2, t0 + 1.4, 'note', 0.5, 0.3, lead='glass', m=[71, 74, 76, 78, 79, 83][c % 6])
S.cue(2, 7, 'tick', f=1300)
H = np.array(recon); S.track(2, 13.5, 18.6, lambda t: 64 + 12 * (np.interp(np.clip((t - 13.5) / 5, 0, 1) * (len(H) - 1), np.arange(len(H)), H) - H.min()) / (H.max() - H.min()), None, 0.8)
S.cue(2, 20.5, 'chime', 0.9, m=79)
S.motion(2, 34.5, 35.6, lambda t: ease(prog(t, 34.5, 1)), 0.4)
for i in range(8): S.cue(2, 35.5 + i * 0.12, 'drop', 0.6, -0.7 + 0.2 * i, m=EM[2 + i])

# 03 latent space — clusters; then a walk from digit to digit (a note at each arrival)
S.motion(3, 0.5, 2.1, lambda t: eout(prog(t, 0.5, 1.5)), 0.5, lo=200, hi=1000)
for i in range(3): S.cue(3, 8.1 + i * 0.05, 'tick', 0.6, f=1400 + 200 * i)
order = [1, 7, 9, 4, 6, 0, 2, 3, 5, 8]
for k in range(len(order) - 1):
    t0 = 15 + 2.8 * k
    if t0 > 46: break
    S.motion(3, t0, t0 + 2.8, lambda t, t0=t0: ease(prog(t, t0, 2.8)), 0.4, lo=300, hi=1200)
    S.cue(3, t0 + 2.5, 'note', 0.8, lead='glass', m=DIGIT_NOTE[order[k + 1]])
S.cue(3, 15, 'note', 0.8, lead='glass', m=DIGIT_NOTE[1]); S.cue(3, 31, 'tick')

# 04 the holes — random codes decode to smudges and odd hybrids
for i in range(8): S.cue(4, 1.5 + i * 0.7, 'note', 0.8, -0.6 + 0.17 * i, lead='glass', m=[71, 61, 74, 63, 76, 78, 70, 83][i])
S.cue(4, 23, 'note', 0.9, lead='glass', m=52); S.cue(4, 23, 'thud', 0.35, f=50)

# 05 fuzzy codes on a leash — clouds swell; cheating (σ → 0, codes fly apart); the leash pulls them home
S.cue(5, 3, 'tick'); S.motion(5, 2, 8.1, lambda t: ease(prog(t, 2, 6)), 0.6, lo=200, hi=1600, tone=(59, 66))
S.motion(5, 16.5, 22.1, lambda t: ease(prog(t, 17, 5)), 0.7, lo=400, hi=3000, tone=(66, 61)); S.cue(5, 17, 'thud', 0.4, f=55)
S.motion(5, 25, 31.1, lambda t: ease(prog(t, 25, 6)), 0.8, lo=2400, hi=300, tone=(59, 71)); S.cue(5, 25, 'tick'); S.cue(5, 25.5, 'note', lead='glass', m=76)
S.cue(5, 40, 'chime', m=79)

# 06 the math of the VAE — two terms; the KL bowls; the ELBO floor rises
S.cue(6, 0.6, 'tick'); S.cue(6, 8, 'note', 0.7, lead='glass', m=71); S.cue(6, 15, 'note', 0.7, lead='glass', m=74); S.cue(6, 23, 'chime', 0.9, m=76)
S.motion(6, 24.5, 28.6, lambda t: ease(prog(t, 24.5, 4)), 0.7, lo=1500, hi=300, tone=(71, 59))
S.motion(6, 30.5, 31.6, lambda t: ease(prog(t, 30.5, 1)), 0.4); S.cue(6, 31, 'tick'); S.cue(6, 38, 'chime', m=79)
S.motion(6, 40, 46.1, lambda t: ease(prog(t, 40, 6)), 0.8, lo=300, hi=1800, tone=(55, 67)); S.cue(6, 47, 'chime', m=83)

# 07 the reparameterisation trick — no gradient through a dice roll ✕ … move the dice aside ✓
S.cue(7, 0.6, 'tick'); S.cue(7, 8, 'thud', 0.5, f=58); S.cue(7, 8.5, 'note', 0.7, lead='glass', m=58)
S.motion(7, 15, 17.1, lambda t: ease(prog(t, 15, 2)), 0.8, lo=300, hi=2400, tone=(59, 71)); S.cue(7, 17, 'chime', 1.1, m=83); S.cue(7, 23, 'tick')

# 08 a map you can sample from — the decoded grid cascades in like a harp; samples; interpolation
S.cue(8, 1.5, 'note', 0.7, lead='glass', m=71); S.cue(8, 9, 'tick')
for k in range(169):
    r, c = divmod(k, 13); S.cue(8, 15.2 + k * 0.012 + 0.2, 'drop', 0.32, -0.8 + 1.6 * c / 12, m=EM[(12 - r) % len(EM)] + (12 if c > 6 else 0))
S.motion(8, 23.5, 24.6, lambda t: ease(prog(t, 23.5, 1)), 0.4)
for i in range(16): S.cue(8, 24.5 + i * 0.12 + 0.1, 'drop', 0.7, -0.6 + 0.08 * i, m=EM[(i * 5) % len(EM)])
for k in range(8): S.cue(8, 32.5 + k * 0.2 + 0.1, 'note', 0.7, -0.6 + 0.17 * k, lead='glass', m=EM[k + 1])
S.cue(8, 40, 'chime', m=79)

# 09 the catch — two sharp ones average into a smear
S.cue(9, 0.6, 'tick'); S.cue(9, 8.5, 'note', lead='glass', m=71); S.cue(9, 8.7, 'note', lead='glass', m=78)
S.cue(9, 12, 'tick'); S.cue(9, 16, 'note', 0.9, lead='glass', m=74); S.cue(9, 16, 'note', 0.7, lead='glass', m=75)
S.cue(9, 23, 'tick'); S.cue(9, 33, 'chime', 0.9, m=76)

# 10 build it
for i in range(18): S.cue(10, 0.8 + i * 0.12, 'key', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): S.cue(10, 18.5 + i * 1.6, 'tick', 0.8, f=1400)

# 11 hook — a puzzle: three models; COPIED; ONLY 1s; how do you grade imagination?
for p in range(3): S.cue(11, 1 + p * 0.5, 'tick', 0.8, -0.6 + 0.6 * p)
S.cue(11, 15.2, 'stamp', 1.0, 0.0); S.cue(11, 22.2, 'stamp', 1.0, 0.6); S.cue(11, 28, 'tick')
S.cue(11, 34, 'riser', 0.9, dur=2.0, m0=40, m1=52)
S.cue(11, 36, 'thud', 1.0, f=41); S.cue(11, 36, 'chime', 1.3, m=64); S.cue(11, 36.05, 'chime', 0.9, m=71)
S.cue(11, 42, 'tick'); S.cue(11, 42.5, 'chime', 1.0, m=83); S.cue(11, 44, 'tick', 0.6)

S.render()
