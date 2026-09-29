"""
Shared sound engine for the series: a soft generative score + sound effects cued to the animation.

Everything is synthesised from scratch in numpy (no samples, no libraries) and is deterministic,
so the HTML player and the rendered MP4 carry identical audio. Each episode has a small script
(src/epNN_audio.py) that picks a key, a chord per scene and a lead timbre, then places cues on the
same clock the scene code uses: (scene index, local time).

Motion sounds are generated from the animation's own motion curve (same easing as the renderer):
loudness follows speed, brightness/pitch follows progress — so they speed up, slow down and pulse
exactly as the picture does.
"""
import json
import os
import subprocess
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SR = 44100


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# the renderer's easing curves (engine: ease / eout / ein)
def ease(x):
    x = np.clip(x, 0, 1); return np.where(x < .5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)


def eout(x):
    x = np.clip(x, 0, 1); return 1 - (1 - x) ** 3


def ein(x):
    x = np.clip(x, 0, 1); return x ** 3


def prog(t, a, d):
    return np.clip((t - a) / d, 0, 1)


def probe(html, exprs):
    """Evaluate JS expressions in the rendered episode page (headless) → python values."""
    cache = os.path.join(HERE, '.probe-' + os.path.basename(html) + '.json')
    key = json.dumps(exprs)
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(ROOT, html)):
        c = json.load(open(cache))
        if c.get('key') == key: return c['val']
    out = subprocess.run(['node', os.path.join(HERE, 'probe.mjs'), html, key], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    val = json.loads(out.strip().splitlines()[-1])
    json.dump({'key': key, 'val': val}, open(cache, 'w'))
    return val


class Score:
    def __init__(self, name, durs, seed=1):
        self.name, self.durs = name, [float(d) for d in durs]
        self.starts = np.concatenate([[0], np.cumsum(self.durs)[:-1]])
        self.total = float(sum(self.durs)); self.N = int(self.total * SR) + 2 * SR
        self.music = np.zeros((self.N, 2), np.float32); self.sfx = np.zeros((self.N, 2), np.float32)
        self.rng = np.random.default_rng(seed)

    # ── plumbing ───────────────────────────────────────────────
    def at(self, sc, t): return self.starts[sc] + t

    def add(self, buf, T, sig, gain=1.0, pan=0.0):
        i = int(T * SR)
        if i >= len(buf) or i + len(sig) <= 0: return
        if i < 0: sig, i = sig[-i:], 0
        sig = sig[:len(buf) - i]
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        buf[i:i + len(sig), 0] += gain * l * sig; buf[i:i + len(sig), 1] += gain * r * sig

    @staticmethod
    def env(n, a, r, sus=1.0):
        e = np.full(n, sus, np.float32); na, nr = min(n, int(a * SR)), min(n, int(r * SR))
        if na: e[:na] = np.linspace(0, sus, na) ** 2
        if nr: e[-nr:] *= np.linspace(1, 0, nr) ** 2
        return e

    def noise(self, n): return self.rng.standard_normal(n).astype(np.float32)

    @staticmethod
    def band(x, lo=None, hi=None):
        X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); m = np.ones_like(f)
        if hi: m *= 1 / (1 + (f / hi) ** 4)
        if lo: m *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4)
        return np.fft.irfft(X * m, len(x)).astype(np.float32)

    def shaped_noise(self, centre, q=2.2, frame=2048):
        """Noise through a band-pass whose centre follows the per-sample array `centre` (Hz)."""
        n = len(centre); hop = frame // 4; out = np.zeros(n + frame, np.float32); win = np.hanning(frame).astype(np.float32)
        freqs = np.fft.rfftfreq(frame, 1 / SR)
        for s in range(0, n, hop):
            fc = centre[min(s + frame // 2, n - 1)]
            S = np.fft.rfft(self.noise(frame) * win) * np.exp(-0.5 * ((freqs - fc) / (fc / q)) ** 2)
            out[s:s + frame] += np.fft.irfft(S, frame).astype(np.float32) * win
        out = out[:n]; return out / (np.abs(out).max() + 1e-9)

    # ── instruments ────────────────────────────────────────────
    def lead(self, kind, f, dur=1.6):
        t = np.arange(int(dur * SR)) / SR; s = lambda k, d: np.sin(2 * np.pi * k * f * t) * np.exp(-t / d)
        if kind == 'musicbox': y = s(1, .55) + .35 * s(2, .18) + .12 * s(3.01, .08)
        elif kind == 'celesta': y = s(1, .9) + .45 * s(2, .35) + .2 * s(4, .12) + .08 * s(5.98, .05)
        elif kind == 'marimba': y = s(1, .35) + .5 * s(4, .06) + .15 * s(10, .015)
        elif kind == 'glass': y = s(1, 1.3) + .3 * s(2.41, .5) + .18 * s(3.87, .25)
        elif kind == 'kalimba': y = s(1, .7) + .28 * s(5.4, .09) + .1 * s(2, .3)
        else: y = s(1, .6)
        return (y * self.env(len(t), 0.004, 0.25)).astype(np.float32)

    def bell(self, f, dur=3.0, bright=1.0):
        t = np.arange(int(dur * SR)) / SR; y = np.zeros_like(t)
        for r, a, d in [(1, 1, 1.6), (2.76, .45 * bright, .7), (5.40, .25 * bright, .35), (8.93, .12 * bright, .18), (0.5, .3, 2.2)]:
            y += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d)
        return (y * self.env(len(t), 0.003, 0.3)).astype(np.float32)

    def blip(self, f=1500, dur=0.07):
        t = np.arange(int(dur * SR)) / SR; fr = f * (1 - 0.35 * t / dur)
        return (np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.018)).astype(np.float32)

    def thud(self, f=62, dur=0.9):
        t = np.arange(int(dur * SR)) / SR; fr = f * (1 + 1.5 * np.exp(-t / 0.05))
        return (np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.28) + 0.25 * self.band(self.noise(len(t)), hi=900) * np.exp(-t / 0.03)).astype(np.float32)

    def key(self, f=3200):  # a soft typewriter / keyboard key
        n = int(0.05 * SR); t = np.arange(n) / SR
        click = self.band(self.noise(n), lo=f * 0.5, hi=f * 1.5) * np.exp(-t / 0.004)
        body = np.sin(2 * np.pi * (f / 9) * t) * np.exp(-t / 0.012) * 0.5
        return (click + body).astype(np.float32)

    def stamp(self):  # rubber stamp: a low thump + paper slap
        n = int(0.5 * SR); t = np.arange(n) / SR
        thump = np.sin(2 * np.pi * np.cumsum(90 * (1 + 0.8 * np.exp(-t / 0.03))) / SR) * np.exp(-t / 0.09)
        slap = self.band(self.noise(n), lo=600, hi=5000) * np.exp(-t / 0.025)
        return (thump + 0.6 * slap).astype(np.float32)

    def rattle(self, dur, f=2600):  # dice / spinning pointer: clicks that slow down, then settle
        n = int(dur * SR); y = np.zeros(n, np.float32); T = 0.0; gap = 0.035
        while T < dur - 0.02:
            i = int(T * SR); m = min(n - i, int(0.02 * SR)); tt = np.arange(m) / SR
            y[i:i + m] += (self.band(self.noise(m), lo=f * .5, hi=f * 1.4) * np.exp(-tt / 0.004)) * self.rng.uniform(0.5, 1)
            T += gap; gap *= 1.16
        return y

    def whoosh(self, dur=1.0, f0=300, f1=3000):
        n = int(dur * SR); c = f0 * (f1 / f0) ** np.linspace(0, 1, n)
        return self.shaped_noise(c, 2.5) * np.sin(np.linspace(0, np.pi, n)) ** 1.5

    def riser(self, dur, m0, m1):  # tension build: noise swell + rising tone
        n = int(dur * SR); u = np.linspace(0, 1, n); c = 300 * (12) ** u
        tone = np.sin(2 * np.pi * np.cumsum(hz(m0) * (hz(m1) / hz(m0)) ** u) / SR)
        return ((0.6 * self.shaped_noise(c, 3) + 0.5 * tone) * u ** 2.2).astype(np.float32)

    def pulse_note(self, f, dur=0.35):  # soft heartbeat / clock pulse
        t = np.arange(int(dur * SR)) / SR
        return (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.09) + 0.3 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.03)).astype(np.float32)

    # ── score: chords per scene ────────────────────────────────
    def pad_voice(self, freqs, dur, bright):
        t = np.arange(int(dur * SR)) / SR; L = np.zeros_like(t); R = np.zeros_like(t)
        for f in freqs:
            for det, side in [(-0.18, 0), (0.18, 1)]:
                ph = 2 * np.pi * f * 2 ** (det * 0.45 / 12) * t + self.rng.uniform(0, 6.28)
                y = sum((bright ** k) * np.sin((k + 1) * ph) / (k + 1) ** 1.8 for k in range(4))
                (L if side == 0 else R)[:] += y
        lfo = 0.8 + 0.2 * np.sin(2 * np.pi * 0.07 * t + self.rng.uniform(0, 6))
        return (L * lfo).astype(np.float32), (R * lfo).astype(np.float32)

    def score(self, chords, plan, scale, lead='musicbox', pad_gain=0.035, lead_gain=0.05):
        """chords: name → (pad midi notes, bass midi). plan[scene] = ([(t, chord), …], notes/s, brightness)."""
        assert len(plan) == len(self.durs), (len(plan), len(self.durs))
        for sc, (chs, dens, bright) in enumerate(plan):
            s0, sd = self.starts[sc], self.durs[sc]; bounds = [c[0] for c in chs] + [sd]
            for (ct, name), cend in zip(chs, bounds[1:]):
                notes, bass = chords[name]; dur = cend - ct + 2.5
                L, R = self.pad_voice([hz(m) for m in notes], dur, 0.55 * bright); e = self.env(len(L), 1.8, 2.5)
                i = int((s0 + ct) * SR); n = min(len(L), self.N - i)
                self.music[i:i + n, 0] += pad_gain * (L * e)[:n]; self.music[i:i + n, 1] += pad_gain * (R * e)[:n]
                sub = np.sin(2 * np.pi * hz(bass) * np.arange(int(dur * SR)) / SR).astype(np.float32) * self.env(int(dur * SR), 1.5, 2.5)
                self.add(self.music, s0 + ct, sub, 0.08)
                if dens <= 0: continue
                pool = sorted(set([m + 12 for m in notes] + scale)); t = ct + 0.8
                while t < cend - 0.5:
                    self.add(self.music, s0 + t, self.lead(lead, hz(pool[self.rng.integers(0, len(pool))])), lead_gain * self.rng.uniform(0.6, 1), self.rng.uniform(-0.6, 0.6))
                    t += self.rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]) / dens

    def pulse(self, sc, t0, t1, period, midi, gain=1.0, accent=4):
        """A soft ticking/heartbeat ostinato (e.g. the autoregressive loop, the GAN duel)."""
        k = 0; t = t0
        while t < t1:
            self.add(self.music, self.at(sc, t), self.pulse_note(hz(midi + (12 if k % accent == 0 else 0))), 0.05 * gain * (1.0 if k % accent == 0 else 0.6))
            t += period; k += 1

    # ── sound effects ──────────────────────────────────────────
    def cue(self, sc, t, kind, gain=1.0, pan=0.0, **kw):
        T, a, s = self.at(sc, t), self.add, self.sfx
        if kind == 'tick': a(s, T, self.blip(kw.get('f', 1500)), 0.10 * gain, pan)
        elif kind == 'chime': a(s, T, self.bell(hz(kw.get('m', 81)), 3.0, kw.get('bright', 1)), 0.09 * gain, pan)
        elif kind == 'thud': a(s, T, self.thud(kw.get('f', 60)), 0.35 * gain, pan)
        elif kind == 'whoosh': a(s, T, self.whoosh(kw.get('dur', 1.0), kw.get('f0', 300), kw.get('f1', 3000)), 0.06 * gain, pan)
        elif kind == 'clink': a(s, T, (self.bell(kw.get('f', 2100), .6, 1.4) + .4 * self.bell(kw.get('f', 2100) * 1.47, .6, 1.2)) * .6, 0.08 * gain, pan)
        elif kind == 'note': a(s, T, self.lead(kw.get('lead', 'musicbox'), hz(kw.get('m', 79))), 0.09 * gain, pan)
        elif kind == 'key': a(s, T, self.key(kw.get('f', 3200)), 0.10 * gain, pan)
        elif kind == 'stamp': a(s, T, self.stamp(), 0.30 * gain, pan)
        elif kind == 'rattle': a(s, T, self.rattle(kw.get('dur', 1.6)), 0.12 * gain, pan)
        elif kind == 'riser': a(s, T, self.riser(kw['dur'], kw.get('m0', 45), kw.get('m1', 57)), 0.05 * gain, pan)
        elif kind == 'drop': a(s, T, self.lead('kalimba', hz(kw.get('m', 76)), 0.5), 0.05 * gain, pan)
        else: raise ValueError(kind)

    def transitions(self, gain=0.9):
        for sc in range(1, len(self.durs)): self.cue(sc, -0.45, 'whoosh', gain, dur=0.9, f0=250, f1=2500)

    def motion(self, sc, t0, t1, P, gain=1.0, lo=250, hi=2600, tone=None, pan=0.0):
        """P(t) → position of the moving thing. Loudness ∝ speed, brightness (and optional pitch) ∝ progress."""
        t = np.linspace(t0, t1, max(2, int((t1 - t0) * SR))); p = np.asarray(P(t), np.float64)
        v = np.abs(np.gradient(p, t)); amp = (v / (v.max() + 1e-9)) ** 0.85
        u = (p - p.min()) / (np.ptp(p) + 1e-9)
        y = self.shaped_noise(lo * (hi / lo) ** u) * amp
        if tone:
            m0, m1 = tone; fr = hz(m0) * (hz(m1) / hz(m0)) ** u
            y = y + 0.35 * np.sin(2 * np.pi * np.cumsum(fr) / SR) * amp
        self.add(self.sfx, self.at(sc, t0), y.astype(np.float32), 0.075 * gain, pan)

    def track(self, sc, t0, t1, midi, amp=None, gain=1.0, pan=0.0, fade=0.3):
        """A sustained soft tone whose pitch follows midi(t) — sonifies a value changing on screen."""
        t = np.linspace(t0, t1, max(2, int((t1 - t0) * SR))); m = np.asarray(midi(t), np.float64)
        f = 440.0 * 2 ** ((m - 69) / 12); ph = 2 * np.pi * np.cumsum(f) / SR
        y = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
        a = np.ones_like(t) if amp is None else np.asarray(amp(t), np.float64)
        e = np.minimum(1, np.minimum((t - t0) / fade, (t1 - t) / fade))
        self.add(self.sfx, self.at(sc, t0), (y * a * e).astype(np.float32), 0.03 * gain, pan)

    @staticmethod
    def snap(v, lo, hi, scale):
        """Map a value in [lo, hi] onto the nearest note of `scale` (a sorted list of midi notes)."""
        u = float(np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)); return scale[int(round(u * (len(scale) - 1)))]

    # ── mix + write ────────────────────────────────────────────
    def reverb(self, x, seconds=2.6, mix=0.28):
        n = int(seconds * SR); t = np.arange(n) / SR; out = np.zeros_like(x); B = 1 << 20
        for ch in range(2):
            ir = (self.rng.standard_normal(n) * np.exp(-t / (seconds / 6.9))).astype(np.float32); ir[:int(0.012 * SR)] = 0; ir /= np.sqrt((ir ** 2).sum())
            wet = np.zeros(len(x) + n, np.float32); F = 1 << int(np.ceil(np.log2(B + n))); IR = np.fft.rfft(ir, F)
            for s in range(0, len(x), B):
                seg = x[s:s + B, ch]; y = np.fft.irfft(np.fft.rfft(seg, F) * IR, F)[:len(seg) + n - 1]
                wet[s:s + len(y)] += y.astype(np.float32)
            out[:, ch] = (1 - mix) * x[:, ch] + mix * wet[:len(x)] * 0.9
        return out

    def render(self, music_db=-5.0, target_rms_db=-22.0):
        m = self.music; m[:int(3 * SR)] *= np.linspace(0, 1, int(3 * SR))[:, None]
        tail = int(self.total * SR); m[tail - int(6 * SR):tail] *= np.linspace(1, 0, int(6 * SR))[:, None] ** 1.5; m[tail:] = 0
        mixd = 10 ** (music_db / 20) * self.reverb(m, 3.2, 0.35) + self.reverb(self.sfx, 1.6, 0.18)
        mixd *= 10 ** (target_rms_db / 20) / np.sqrt((mixd ** 2).mean())
        mixd = (np.tanh(mixd * 1.2) / 1.2)[:tail]
        wav = os.path.join(HERE, f'_{self.name}.wav'); out = os.path.join(ROOT, f'{self.name}.mp3')
        with wave.open(wav, 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mixd * 32767).astype('<i2').tobytes())
        ff = os.environ.get('FFMPEG') or 'ffmpeg'
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', wav, '-c:a', 'libmp3lame', '-b:a', '128k', out], check=True); os.remove(wav)
        print(f'wrote {out}  ({self.total:.0f} s, {os.path.getsize(out) / 1e6:.1f} MB, RMS {20 * np.log10(np.sqrt((mixd ** 2).mean())):.1f} dBFS)')
