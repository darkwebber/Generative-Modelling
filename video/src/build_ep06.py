"""Assemble ep06-flows.html = shared engine (from ep02) + src/ep06-scenes.js + player."""
import os
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
lines = open(os.path.join(root, 'ep02-autoregressive.html')).read().split('\n')
i_math = [i for i, l in enumerate(lines) if 'The language model maths' in l][0] - 1
i_tail = [i for i, l in enumerate(lines) if 'Timeline, chrome, grain, player' in l][0] - 1
head, tail = '\n'.join(lines[:i_math]), '\n'.join(lines[i_tail:])
for a, b in [('<title>Autoregressive Generation Explained</title>', '<title>Normalizing Flows</title>'),
             ('aria-label="Autoregressive generation explainer animation"', 'aria-label="Normalizing Flows explainer animation"'),
             ('<script src="names.js"></script>', '<script src="ep06-assets.js"></script>'),
             ('(shared with episode 01)', '(shared across episodes)')]:
    head = head.replace(a, b)
tail = tail.replace('`EP 02  ·  SCENE', '`EP 06  ·  SCENE').replace(
    "DIG = { '3': makeDigit('3'), '8': makeDigit('8'), '0': makeDigit('0'), '6': makeDigit('6') };",
    'void 0;')
# ── sound (pilot): a pre-rendered soundtrack (src/ep06_audio.py) that follows the animation clock
head = head.replace('<span class="time" id="time">0:00 / 0:00</span>',
    '<span class="time" id="time">0:00 / 0:00</span>\n    <button id="snd-btn" title="sound on / off (M)">♪ sound on</button>\n    <audio id="snd" src="ep06-audio.mp3" preload="auto"></audio>')
AUDIO = {
    'function loop(ts) { if (!playing) return; if (lastTs != null) now += (ts - lastTs) / 1000; lastTs = ts;':
    "const snd = document.getElementById('snd'), sndBtn = document.getElementById('snd-btn'); let soundOn = true;\n"
    "function syncAudio(force) { if (!playing || !soundOn || capture) { if (!snd.paused) snd.pause(); return; } if (force || Math.abs(snd.currentTime - now) > 0.25) snd.currentTime = now; if (snd.paused) snd.play().catch(() => {}); }\n"
    "sndBtn.onclick = () => { soundOn = !soundOn; sndBtn.textContent = soundOn ? '♪ sound on' : '♪ sound off'; syncAudio(true); };\n"
    "function loop(ts) { if (!playing) return; if (soundOn && !snd.paused && snd.readyState >= 2 && !snd.seeking) now = snd.currentTime; else if (lastTs != null) now += (ts - lastTs) / 1000; lastTs = ts;",
    "draw(); if (playing) requestAnimationFrame(loop); }": "draw(); syncAudio(); if (playing) requestAnimationFrame(loop); }",
    "lastTs = null; if (playing) requestAnimationFrame(loop); }": "lastTs = null; syncAudio(true); if (playing) requestAnimationFrame(loop); }",
    "scrub.oninput = () => { now = scrub.value / 1000 * DURATION; draw(); };": "scrub.oninput = () => { now = scrub.value / 1000 * DURATION; draw(); syncAudio(true); };",
    "if (e.code === 'ArrowRight') { now = Math.min(DURATION, now + 5); draw(); } if (e.code === 'ArrowLeft') { now = Math.max(0, now - 5); draw(); } });":
    "if (e.code === 'ArrowRight') { now = Math.min(DURATION, now + 5); draw(); syncAudio(true); } if (e.code === 'ArrowLeft') { now = Math.max(0, now - 5); draw(); syncAudio(true); } if (e.code === 'KeyM') sndBtn.click(); });",
    "b.onclick = () => { now = s.start; draw(); };": "b.onclick = () => { now = s.start; draw(); syncAudio(true); };",
}
for a, b in AUDIO.items():
    assert a in tail, a[:60]
    tail = tail.replace(a, b)
scenes = open(os.path.join(here, 'ep06-scenes.js')).read()
open(os.path.join(root, 'ep06-flows.html'), 'w').write(head + '\n' + scenes + '\n' + tail)
print('built ep06-flows.html')
