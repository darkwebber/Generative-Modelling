"""
Add the soundtrack to an episode page: an <audio> element + a ♪ button, kept locked to the
animation clock (play / pause / scrub / chapter jumps / ←→ all re-sync; M mutes).

Idempotent: if the page already has the player (e.g. episode 02, which is also the engine source
for later episodes), only the audio file name is swapped.

    python src/audio_player.py generative-modelling.html ep01-audio.mp3   # patch a page in place
"""
import re
import sys

HEAD = ('<span class="time" id="time">0:00 / 0:00</span>',
        '<span class="time" id="time">0:00 / 0:00</span>\n    <button id="snd-btn" title="sound on / off (M)">♪ sound on</button>\n'
        '    <audio id="snd" src="{mp3}" preload="auto"></audio>')
TAIL = {
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


def inject(doc, mp3):
    if 'id="snd"' in doc:
        return re.sub(r'<audio id="snd" src="[^"]+"', f'<audio id="snd" src="{mp3}"', doc)
    assert HEAD[0] in doc, 'player controls not found'
    doc = doc.replace(HEAD[0], HEAD[1].format(mp3=mp3), 1)
    for a, b in TAIL.items():
        assert a in doc, a[:60]
        doc = doc.replace(a, b, 1)
    return doc


if __name__ == '__main__':
    path, mp3 = sys.argv[1], sys.argv[2]
    doc = inject(open(path).read(), mp3)          # read first, then write
    with open(path, 'w') as fh: fh.write(doc)
    print(f'{path}: sound player → {mp3}')
