"""
Soundtrack for episode 06 (pilot): a soft generative score + sound effects cued to the animation.

Everything is synthesised from scratch in numpy (no samples, no libraries), deterministically,
so the audio in the HTML player and in the rendered MP4 is identical.

    python src/ep06_audio.py            # → ep06-audio.mp3 (next to the episode HTML)

Scene start times are read from src/ep06-scenes.js, so cues stay in sync if scenes change length.
Cues are written as (scene, local time, sound, params) — the same clock the scene code uses.
"""
import os
import re
import subprocess
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SR = 44100
rng = np.random.default_rng(6)

durs = [float(d) for d in re.findall(r"SC\.push\(\{ dur: ([\d.]+)", open(os.path.join(HERE, 'ep06-scenes.js')).read())]
starts = np.concatenate([[0], np.cumsum(durs)[:-1]])
TOTAL = float(sum(durs))
N = int(TOTAL * SR) + SR
music = np.zeros((N, 2), np.float32)
sfx = np.zeros((N, 2), np.float32)


def at(scene, t):
    return starts[scene] + t


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def env_adsr(n, a, r, sus=1.0):
    e = np.full(n, sus, np.float32)
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    if na: e[:na] = np.linspace(0, sus, na) ** 2
    if nr: e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def add(buf, t0, sig, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= len(buf) or i + len(sig) <= 0: return
    if i < 0: sig, i = sig[-i:], 0
    sig = sig[:len(buf) - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += gain * l * sig
    buf[i:i + len(sig), 1] += gain * r * sig


def fft_filter(x, lo=None, hi=None):
    """Zero-phase band-limit by FFT (fine for short sounds)."""
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); m = np.ones_like(f)
    if hi: m *= 1 / (1 + (f / hi) ** 4)
    if lo: m *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4)
    return np.fft.irfft(X * m, len(x)).astype(np.float32)


def sweep_noise(dur, f0, f1, q=3.0, frame=2048):
    """Noise through a band-pass whose centre glides f0 → f1 (overlap-add of shaped frames)."""
    n = int(dur * SR); hop = frame // 4; out = np.zeros(n + frame, np.float32); win = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1 / SR)
    for k, s in enumerate(range(0, n, hop)):
        u = s / max(1, n); fc = f0 * (f1 / f0) ** u; bw = fc / q
        seg = rng.standard_normal(frame).astype(np.float32) * win
        S = np.fft.rfft(seg) * np.exp(-0.5 * ((freqs - fc) / bw) ** 2)
        out[s:s + frame] += np.fft.irfft(S, frame).astype(np.float32) * win
    out = out[:n]; return out / (np.abs(out).max() + 1e-9)


# ── instruments ────────────────────────────────────────────────
def tone(f, dur, harm=(1, .5, .25), decay=None, vib=0.0):
    t = np.arange(int(dur * SR)) / SR; ph = 2 * np.pi * f * t + (vib * np.sin(2 * np.pi * 5 * t) if vib else 0)
    y = sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harm))
    if decay: y *= np.exp(-t / decay)
    return y.astype(np.float32)


def pluck(f, dur=1.6):            # music-box-ish: bright attack, soft decay
    t = np.arange(int(dur * SR)) / SR
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.55) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.18) + 0.12 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / 0.08)
    return (y * env_adsr(len(t), 0.004, 0.2)).astype(np.float32)


def bell(f, dur=3.0, bright=1.0):  # inharmonic partials → a soft chime
    t = np.arange(int(dur * SR)) / SR; y = np.zeros_like(t)
    for r, a, d in [(1, 1, 1.6), (2.76, .45 * bright, .7), (5.40, .25 * bright, .35), (8.93, .12 * bright, .18), (0.5, .3, 2.2)]:
        y += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d)
    return (y * env_adsr(len(t), 0.003, 0.3)).astype(np.float32)


def blip(f=1400, dur=0.07):       # UI tick
    t = np.arange(int(dur * SR)) / SR; fr = f * (1 - 0.35 * t / dur)
    return (np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.018)).astype(np.float32)


def thud(f=62, dur=0.9):
    t = np.arange(int(dur * SR)) / SR; fr = f * (1 + 1.5 * np.exp(-t / 0.05))
    y = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.28) + 0.25 * fft_filter(rng.standard_normal(len(t)).astype(np.float32), hi=900) * np.exp(-t / 0.03)
    return y.astype(np.float32)


def whoosh(dur=1.1, f0=300, f1=3200, rev=False):
    y = sweep_noise(dur, f0, f1, 2.5); e = np.sin(np.linspace(0, np.pi, len(y))) ** 1.5
    return (y * e)[::-1].copy() if rev else y * e


def gliss(f0, f1, dur, harm=(1, .3), shape=None):
    n = int(dur * SR); fr = f0 * (f1 / f0) ** np.linspace(0, 1, n); ph = 2 * np.pi * np.cumsum(fr) / SR
    y = sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harm)); e = shape if shape is not None else np.sin(np.linspace(0, np.pi, n))
    return (y * e).astype(np.float32)


def sand(dur, density=90, bright=5000):  # grains of sand: sparse filtered clicks
    n = int(dur * SR); y = np.zeros(n, np.float32); k = int(density * dur)
    idx = rng.integers(0, n - 400, k)
    click = fft_filter(rng.standard_normal(400).astype(np.float32), lo=bright * 0.5, hi=bright * 1.6) * np.exp(-np.arange(400) / 60)
    for i, a in zip(idx, rng.uniform(0.2, 1, k)): y[i:i + 400] += a * click
    return y * env_adsr(n, 0.4, 0.8)


def clink(f=2100):
    return (bell(f, 0.6, 1.4) + 0.4 * bell(f * 1.47, 0.6, 1.2)) * 0.6


# ── the score: one chord per scene (D dorian / minor colour) ───
D = 50  # D3
CH = {  # midi notes (pad voicing), bass note
    'Dm9': ([50, 57, 60, 64, 65], 38), 'Bbmaj7': ([46, 53, 57, 62, 65], 34), 'Fmaj9': ([53, 57, 60, 64, 67], 41),
    'Gm9': ([55, 58, 62, 65, 69], 43), 'Am7': ([57, 60, 64, 67, 72], 45), 'Cadd9': ([48, 55, 62, 64, 67], 36),
    'A7sus': ([57, 62, 64, 67, 71], 45), 'Ebmaj7#11': ([51, 58, 62, 67, 69], 39), 'Dm': ([50, 57, 62, 65, 69], 38),
}
PLAN = [  # per scene: list of (local start, chord), pluck density (notes/s), brightness
    ([(0, 'Dm9'), (18.8, 'Bbmaj7')], 0.6, .8), ([(0, 'Dm9'), (30, 'Fmaj9')], 0.7, .7), ([(0, 'Bbmaj7'), (30, 'Fmaj9')], 0.9, .8),
    ([(0, 'Gm9'), (31, 'A7sus'), (37.5, 'Dm9')], 0.8, .7), ([(0, 'Am7'), (23, 'Fmaj9')], 0.6, .6), ([(0, 'Fmaj9'), (36.5, 'Cadd9')], 1.1, 1.0),
    ([(0, 'Dm9'), (32, 'Fmaj9')], 0.8, .8), ([(0, 'Bbmaj7'), (24, 'Fmaj9')], 1.0, .9), ([(0, 'Fmaj9')], 0.8, .9),
    ([(0, 'Gm9'), (16, 'A7sus'), (32, 'Dm9')], 0.6, .6), ([(0, 'Ebmaj7#11'), (24, 'Bbmaj7')], 0.9, 1.0), ([(0, 'Fmaj9'), (18, 'Cadd9')], 1.4, 1.0),
    ([(0, 'Dm'), (8, 'Bbmaj7'), (23, 'Gm9'), (31, 'A7sus'), (38.5, 'Dm9')], 0.5, .6),
]
assert len(PLAN) == len(durs), (len(PLAN), len(durs))
SCALE = [62, 64, 65, 67, 69, 72, 74, 76, 77, 79, 81]  # D dorian-ish, upper register for plucks


def pad_voice(freqs, dur, bright):
    t = np.arange(int(dur * SR)) / SR; L = np.zeros_like(t); R = np.zeros_like(t)
    for f in freqs:
        for det, side in [(-0.18, 0), (0.18, 1)]:
            ff = f * 2 ** (det * 0.45 / 12)  # a few cents of chorus
            ph = 2 * np.pi * ff * t + rng.uniform(0, 6.28)
            y = sum((bright ** k) * np.sin((k + 1) * ph) / (k + 1) ** 1.8 for k in range(4))
            (L if side == 0 else R)[:] += y
    lfo = 0.8 + 0.2 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6))
    return (L * lfo).astype(np.float32), (R * lfo).astype(np.float32)


for sc, (chords, dens, bright) in enumerate(PLAN):
    s0, sd = starts[sc], durs[sc]
    bounds = [c[0] for c in chords] + [sd]
    for (ct, name), cend in zip(chords, bounds[1:]):
        notes, bass = CH[name]; dur = cend - ct + 2.5  # overlap into the next chord / scene
        L, R = pad_voice([hz(m) for m in notes], dur, 0.55 * bright)
        e = env_adsr(len(L), 1.8, 2.5)
        i = int((s0 + ct) * SR); n = min(len(L), N - i)
        music[i:i + n, 0] += 0.035 * (L * e)[:n]; music[i:i + n, 1] += 0.035 * (R * e)[:n]
        sub = tone(hz(bass), dur, (1, .15)) * env_adsr(int(dur * SR), 1.5, 2.5)
        add(music, s0 + ct, sub, 0.08)
        # plucks: a gentle, seeded arpeggio drawn from chord tones + scale
        pool = sorted(set([m + 12 for m in notes] + SCALE))
        t = ct + 0.8
        while t < cend - 0.5:
            m = pool[rng.integers(0, len(pool))]
            add(music, s0 + t, pluck(hz(m)), 0.05 * rng.uniform(0.6, 1), rng.uniform(-0.6, 0.6))
            t += rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]) / dens

# fade the score in at the start and out at the end
music[:int(3 * SR)] *= np.linspace(0, 1, int(3 * SR))[:, None]
tail = int(TOTAL * SR); music[tail - int(6 * SR):tail] *= np.linspace(1, 0, int(6 * SR))[:, None] ** 1.5; music[tail:] = 0

# ── sound effects, cued to the animation ───────────────────────
def cue(scene, t, kind, gain=1.0, pan=0.0, **kw):
    T = at(scene, t)
    if kind == 'tick': add(sfx, T, blip(kw.get('f', 1500)), 0.10 * gain, pan)
    elif kind == 'chime': add(sfx, T, bell(hz(kw.get('m', 81)), 3.0, kw.get('bright', 1)), 0.09 * gain, pan)
    elif kind == 'thud': add(sfx, T, thud(kw.get('f', 60)), 0.35 * gain, pan)
    elif kind == 'whoosh': add(sfx, T, whoosh(kw.get('dur', 1.0), kw.get('f0', 300), kw.get('f1', 3000), kw.get('rev', False)), 0.06 * gain, pan)
    elif kind == 'rise': add(sfx, T, gliss(hz(kw['m0']), hz(kw['m1']), kw['dur'], (1, .25, .1)), 0.035 * gain, pan)
    elif kind == 'sand': add(sfx, T, sand(kw['dur'], kw.get('density', 90)), 0.03 * gain, pan)
    elif kind == 'clink': add(sfx, T, clink(kw.get('f', 2100)), 0.08 * gain, pan)
    elif kind == 'pluck': add(sfx, T, pluck(hz(kw.get('m', 79))), 0.09 * gain, pan)


# scene transitions: a soft whoosh just before each new scene
for sc in range(1, len(durs)): cue(sc, -0.45, 'whoosh', 0.9, dur=0.9, f0=250, f1=2500)
# 00 cold open
cue(0, 0.5, 'rise', m0=50, m1=62, dur=5.2); cue(0, 0.5, 'sand', 0.6, dur=5, density=40)
cue(0, 6.5, 'rise', m0=62, m1=50, dur=5.0); cue(0, 12, 'rise', 0.8, m0=50, m1=62, dur=3.2)
cue(0, 15, 'pluck', m=81); cue(0, 18.8, 'chime', 1.4, m=74); cue(0, 18.8, 'chime', 0.9, m=81)
for i in range(5): cue(0, 20 + i * 0.3, 'tick', 0.7, -0.6 + 0.3 * i, f=1300 + 120 * i)
# 01 sand
cue(1, 8, 'rise', 0.7, m0=57, m1=69, dur=1.6); cue(1, 15, 'tick'); cue(1, 15.2, 'sand', 1.1, dur=6.5)
cue(1, 22, 'whoosh', 0.7, dur=0.8, f0=2500, f1=500); cue(1, 22.5, 'pluck', m=76); cue(1, 30, 'tick'); cue(1, 32, 'chime', m=79)
cue(1, 31, 'sand', 1.2, dur=13, density=110); cue(1, 38.5, 'tick')
# 02 jacobian
cue(2, 2, 'rise', 0.8, m0=55, m1=62, dur=4); cue(2, 8, 'tick'); cue(2, 9, 'whoosh', 0.8, dur=5, f0=400, f1=4000)
cue(2, 15, 'pluck', 1.2, -0.3, m=81); cue(2, 23, 'pluck', 1.2, 0.3, m=84); cue(2, 30, 'chime', m=77)
for i in range(4): cue(2, 32 + i * 0.15, 'tick', 0.6, f=1600 + 100 * i)
cue(2, 38.5, 'tick'); cue(2, 40, 'tick')
# 03 determinant
cue(3, 7, 'tick')
for i in range(6): cue(3, 9 + i * 0.8, 'tick', 0.8, -0.5 + 0.2 * i, f=1200 + 90 * i)
cue(3, 17, 'chime', 1.2, m=81); cue(3, 20.5, 'rise', m0=57, m1=64, dur=2); cue(3, 26.5, 'rise', m0=64, m1=55, dur=2)
cue(3, 31.5, 'thud', 0.9); cue(3, 37.5, 'tick'); cue(3, 43, 'chime', m=74)
# 04 the cost
cue(4, 3, 'tick', pan=-0.4); cue(4, 3.3, 'tick', pan=-0.3); cue(4, 8, 'tick'); cue(4, 15.5, 'thud', 0.8, f=48)
cue(4, 23, 'tick'); cue(4, 25, 'chime', m=79); cue(4, 32, 'whoosh', 0.8, dur=3, f0=900, f1=2200); cue(4, 39, 'rise', m0=60, m1=67, dur=2.5)
# 05 coupling
cue(5, 8, 'tick'); cue(5, 16, 'rise', 0.7, m0=57, m1=64, dur=3.5); cue(5, 20, 'rise', 0.7, m0=64, m1=57, dur=3.5); cue(5, 24, 'rise', 0.7, m0=57, m1=64, dur=3.5)
for tt, f in [(26, 1300), (28.5, 1200), (34, 1500), (35.5, 1600)]: cue(5, tt, 'tick', f=f)
cue(5, 36.5, 'chime', 1.4, m=86, bright=1.3); cue(5, 37, 'tick'); cue(5, 40, 'tick'); cue(5, 47, 'tick'); cue(5, 48, 'rise', m0=50, m1=69, dur=7)
# 06 likelihood
cue(6, 8, 'tick'); cue(6, 9, 'rise', m0=62, m1=50, dur=6); cue(6, 16, 'tick', pan=-0.3); cue(6, 17.5, 'tick', pan=0.3)
cue(6, 24, 'tick'); cue(6, 32, 'chime', m=81); cue(6, 40.5, 'rise', 0.8, m0=62, m1=50, dur=5)
# 07 layers
cue(7, 5, 'rise', m0=62, m1=50, dur=16); cue(7, 5, 'sand', 0.5, dur=16, density=30); cue(7, 24, 'whoosh', dur=1.0); cue(7, 25, 'rise', m0=50, m1=62, dur=14)
# 08 density
cue(8, 1, 'chime', 0.8, m=74); cue(8, 8, 'chime', m=86); cue(8, 16, 'whoosh', 0.8); cue(8, 24, 'tick'); cue(8, 24.2, 'tick')
# 09 the catch
cue(9, 4, 'tick'); cue(9, 10, 'pluck', m=69); cue(9, 17, 'rise', 0.9, m0=45, m1=52, dur=5); cue(9, 24, 'sand', 0.9, dur=6)
cue(9, 32, 'thud', 0.6, f=70); cue(9, 40, 'clink', 0.8)
# 10 continuous
cue(10, 2, 'sand', 0.6, dur=16, density=45); cue(10, 2, 'rise', 0.6, m0=50, m1=62, dur=16); cue(10, 8, 'tick'); cue(10, 16, 'tick'); cue(10, 31, 'chime', m=79)
# 11 build it: soft keyboard clicks as code lines appear
for i in range(16): cue(11, 0.8 + i * 0.12, 'tick', 0.45, 0.2 * np.sin(i), f=2400 + 200 * (i % 3))
for i in range(5): cue(11, 18.5 + i * 1.6, 'tick', 0.8, f=1400)
# 12 hook
for i in range(3): cue(12, 1.5 + i * 0.8, 'clink', 0.8, -0.3 + 0.3 * i, f=1900 + 150 * i)
for i in range(6): cue(12, 7.6 + i * 0.09, 'clink', 0.5, rng.uniform(-0.7, 0.7), f=rng.uniform(1600, 2600))
cue(12, 7.5, 'whoosh', 1.2, dur=1.4, f0=3000, f1=250); cue(12, 8, 'rise', m0=38, m1=45, dur=6)
cue(12, 15, 'sand', 0.8, dur=6, density=50); cue(12, 23, 'chime', m=74); cue(12, 31, 'rise', m0=45, m1=57, dur=7)
cue(12, 38.5, 'thud', 1.1, f=45); cue(12, 38.5, 'chime', 1.5, m=62); cue(12, 38.55, 'chime', 1.2, m=69); cue(12, 40, 'tick'); cue(12, 40.5, 'chime', 1.0, m=86)


# ── reverb (FFT convolution with a synthetic room) + mix ───────
def reverb(x, seconds=2.6, mix=0.28):
    n = int(seconds * SR); t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = (rng.standard_normal(n) * np.exp(-t / (seconds / 6.9))).astype(np.float32); ir[:int(0.012 * SR)] = 0; ir /= np.sqrt((ir ** 2).sum())
        B = 1 << 20  # block convolution keeps memory sane
        wet = np.zeros(len(x) + n, np.float32); F = 1 << int(np.ceil(np.log2(B + n))); IR = np.fft.rfft(ir, F)
        for s in range(0, len(x), B):
            seg = x[s:s + B, ch]; y = np.fft.irfft(np.fft.rfft(seg, F) * IR, F)[:len(seg) + n - 1]
            wet[s:s + len(y)] += y.astype(np.float32)
        out[:, ch] = (1 - mix) * x[:, ch] + mix * wet[:len(x)] * 0.9
    return out


mixd = 0.55 * reverb(music, 3.2, 0.35) + reverb(sfx, 1.6, 0.18)   # score sits ~5 dB under the effects
mixd *= 10 ** (-22 / 20) / np.sqrt((mixd ** 2).mean())              # a soft bed for narration: about −22 dBFS RMS
mixd = np.tanh(mixd * 1.2) / 1.2                                     # gentle limiter for the odd peak
mixd = mixd[:int(TOTAL * SR)]

wav = os.path.join(HERE, '_ep06-audio.wav')
with wave.open(wav, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mixd * 32767).astype('<i2').tobytes())
ff = os.environ.get('FFMPEG') or 'ffmpeg'
out = os.path.join(ROOT, 'ep06-audio.mp3')
subprocess.run([ff, '-y', '-loglevel', 'error', '-i', wav, '-c:a', 'libmp3lame', '-b:a', '128k', out], check=True)
os.remove(wav)
rms = 20 * np.log10(np.sqrt((mixd ** 2).mean()) + 1e-9)
print(f'wrote {out}  ({TOTAL:.0f} s, {os.path.getsize(out) / 1e6:.1f} MB, RMS {rms:.1f} dBFS)')
