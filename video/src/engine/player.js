// ─────────────────────────────────────────────────────────────
//  Timeline, chrome, grain, player  (shared; configured by EPISODE = { num, header, chapterLabel })
// ─────────────────────────────────────────────────────────────
let acc = 0; SC.forEach(s => { s.start = acc; acc += s.dur; });
const DURATION = acc;
const GRAIN = (() => { const c = document.createElement('canvas'); c.width = c.height = 256; const g = c.getContext('2d'); const img = g.createImageData(256, 256); const r = rng(1234);
  for (let i = 0; i < 256 * 256; i++) { const v = Math.floor(r() * 255); img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255; } g.putImageData(img, 0, 0); return c; })();
let GRAIN_PAT = null, VIGNETTE = null;
function header(i, s, t) {
  const a = eout(prog(t, 0.15, 0.8));
  text(`${EPISODE.header ?? `EP ${EPISODE.num}  ·  `}SCENE ${String(i).padStart(2, '0')}`, 80, 82, { font: F.serif, italic: true, size: 21, color: P.terracotta, a, ls: 3 });
  text(s.title, 80 + (1 - a) * -16, 142, { font: F.serif, size: 58, color: P.chalk, a, ls: -0.5 });
  text(s.sub, 80, 188, { size: 26, color: P.stone, a: eout(prog(t, 0.5, 0.8)) });
}
function captions(s, t) {
  for (const [a, b, str] of s.caps) {
    if (t < a || t > b) continue;
    const al = Math.min(clamp((t - a) / 0.35), clamp((b - t) / 0.35));
    const lines = wrap(str, 1560, `400 34px ${F.sans}`); const ys = lines.length === 1 ? [992] : [972, 1014];
    ctx.save(); ctx.shadowColor = 'rgba(10,8,7,0.95)'; ctx.shadowBlur = 14;
    lines.slice(0, 2).forEach((l, k) => text(l, 960, ys[k], { size: 34, color: P.chalk, align: 'center', a: al * 0.95 }));
    ctx.restore();
  }
}
function progressBar(T, idx) {
  const x0 = 80, x1 = 1840, y = 1040;
  ctx.fillStyle = P.border; setA(1); ctx.fillRect(x0, y, x1 - x0, 2);
  ctx.fillStyle = rgba(P.terracotta, 0.85); ctx.fillRect(x0, y, (x1 - x0) * T / DURATION, 2);
  SC.forEach((s, j) => { const x = x0 + (x1 - x0) * s.start / DURATION; line(x, y - 4, x, y + 6, j === idx ? P.terracotta : P.borderLight, 1.5);
    text(`${String(j).padStart(2, '0')} ${s.chapter}`.toUpperCase(), x + 6, y + 26, { font: F.mono, size: EPISODE.chapterLabel?.[0] ?? 11, color: j === idx ? P.terracotta : P.dust, ls: EPISODE.chapterLabel?.[1] ?? 0.5 }); });
}
function sceneAt(T) { const i = SC.findIndex(s => T < s.start + s.dur); return i < 0 ? SC.length - 1 : i; }
function render(T) {
  T = clamp(T, 0, DURATION - 1e-3);
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over'; ctx.shadowBlur = 0; GA = 1;
  ctx.fillStyle = P.canvas; ctx.fillRect(0, 0, W, H);
  const i = sceneAt(T), s = SC[i], t = T - s.start;
  ctx.save(); s.draw(t); ctx.restore(); GA = 1;
  if (s.title) header(i, s, t);
  captions(s, t);
  const f = Math.min(clamp(t / 0.7), clamp((s.dur - t) / 0.7));
  if (f < 1) { ctx.globalAlpha = 1 - ease(f); ctx.fillStyle = P.canvas; ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1; }
  if (!VIGNETTE) { VIGNETTE = ctx.createRadialGradient(960, 540, 520, 960, 540, 1180); VIGNETTE.addColorStop(0, 'rgba(8,6,5,0)'); VIGNETTE.addColorStop(1, 'rgba(8,6,5,0.55)'); }
  ctx.fillStyle = VIGNETTE; ctx.fillRect(0, 0, W, H);
  progressBar(T, i);
  if (!GRAIN_PAT) GRAIN_PAT = ctx.createPattern(GRAIN, 'repeat');
  const fr = Math.floor(T * 24), r = rng(fr + 7);
  ctx.save(); ctx.globalCompositeOperation = 'overlay'; ctx.globalAlpha = 0.07; ctx.translate(-Math.floor(r() * 256), -Math.floor(r() * 256));
  ctx.fillStyle = GRAIN_PAT; ctx.fillRect(0, 0, W + 256, H + 256); ctx.restore();
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
}

const capture = new URLSearchParams(location.search).has('capture');
if (capture) document.body.classList.add('capture');
let now = 0, playing = false, lastTs = null;
const playBtn = document.getElementById('play'), scrub = document.getElementById('scrub'), timeEl = document.getElementById('time'), chapEl = document.getElementById('chapters');
const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
SC.forEach((s, j) => { const b = document.createElement('button'); b.textContent = `${String(j).padStart(2, '0')} ${s.chapter}`; b.onclick = () => { now = s.start; draw(); syncAudio(true); }; chapEl.appendChild(b); });
function draw() { render(now); scrub.value = (now / DURATION * 1000).toFixed(1); timeEl.textContent = `${fmt(now)} / ${fmt(DURATION)}`; const i = sceneAt(now); [...chapEl.children].forEach((b, j) => b.classList.toggle('on', j === i)); }
const snd = document.getElementById('snd'), sndBtn = document.getElementById('snd-btn'); let soundOn = true;
// The picture keeps its own clock; the soundtrack follows it. (Reading the clock from the <audio> element froze the picture
// wherever a streamed mp3 reports a stalled currentTime while still sounding, e.g. mobile in-app browsers.)
let lastSeek = -1e9;
function syncAudio(force) { if (!playing || !soundOn || capture) { if (!snd.paused) snd.pause(); return; }
  const drift = Math.abs(snd.currentTime - now), t = performance.now();
  if ((force || (drift > 0.35 && t - lastSeek > 1500)) && !snd.seeking) { try { snd.currentTime = now; } catch (e) {} lastSeek = t; }
  if (snd.paused) snd.play().catch(() => {}); }
sndBtn.onclick = () => { soundOn = !soundOn; sndBtn.textContent = soundOn ? '♪ sound on' : '♪ sound off'; syncAudio(true); };
let drawErr = 0;
function loop(ts) { if (!playing) return;
  if (lastTs != null) now += Math.min(0.1, (ts - lastTs) / 1000);
  // nudge gently toward the soundtrack while it is really advancing, so long plays stay in sync
  if (soundOn && !snd.paused && !snd.seeking && snd.readyState >= 3) { const d = snd.currentTime - now; if (Math.abs(d) < 0.3) now += d * 0.05; }
  lastTs = ts; if (now >= DURATION) { now = DURATION; playing = false; playBtn.textContent = '▶ play'; }
  try { draw(); } catch (e) { if (drawErr++ < 3) console.error(e); }       // one bad frame must never stop the film
  syncAudio(); if (playing) requestAnimationFrame(loop); }
function toggle() { if (now >= DURATION - 0.01) now = 0; playing = !playing; playBtn.textContent = playing ? '❚❚ pause' : '▶ play'; lastTs = null; syncAudio(true); if (playing) requestAnimationFrame(loop); }
playBtn.onclick = toggle;
scrub.oninput = () => { now = scrub.value / 1000 * DURATION; draw(); syncAudio(true); };
window.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); toggle(); } if (e.code === 'ArrowRight') { now = Math.min(DURATION, now + 5); draw(); syncAudio(true); } if (e.code === 'ArrowLeft') { now = Math.max(0, now - 5); draw(); syncAudio(true); } if (e.code === 'KeyM') sndBtn.click(); });
window.DURATION = DURATION;
window.renderAt = T => render(T);
(async () => {
  try { await Promise.race([Promise.all([document.fonts.load(`400 40px Newsreader`), document.fonts.load(`italic 400 40px Newsreader`), document.fonts.load(`400 40px Karla`), document.fonts.load(`400 40px 'JetBrains Mono'`)]), new Promise(r => setTimeout(r, 4000))]); } catch (e) {}
  ON_READY.forEach(f => f());
  draw(); window.READY = true;
})();
