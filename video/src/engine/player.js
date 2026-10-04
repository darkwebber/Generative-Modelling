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
function render(T, poster = false) {
  T = clamp(T, 0, DURATION - 1e-3);
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over'; ctx.shadowBlur = 0; GA = 1;
  ctx.fillStyle = P.canvas; ctx.fillRect(0, 0, W, H);
  const i = sceneAt(T), s = SC[i], t = T - s.start;
  ctx.save(); s.draw(t); ctx.restore(); GA = 1;
  if (s.title) header(i, s, t);
  if (!poster && capture) captions(s, t);       // in the browser, captions are page text (readable at any size)
  const f = Math.min(clamp(t / 0.7), clamp((s.dur - t) / 0.7));
  if (f < 1) { ctx.globalAlpha = 1 - ease(f); ctx.fillStyle = P.canvas; ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1; }
  if (!VIGNETTE) { VIGNETTE = ctx.createRadialGradient(960, 540, 520, 960, 540, 1180); VIGNETTE.addColorStop(0, 'rgba(8,6,5,0)'); VIGNETTE.addColorStop(1, 'rgba(8,6,5,0.55)'); }
  ctx.fillStyle = VIGNETTE; ctx.fillRect(0, 0, W, H);
  if (capture) progressBar(T, i);               // in the browser, the control bar does this job
  if (!GRAIN_PAT) GRAIN_PAT = ctx.createPattern(GRAIN, 'repeat');
  const fr = Math.floor(T * 24), r = rng(fr + 7);
  ctx.save(); ctx.globalCompositeOperation = 'overlay'; ctx.globalAlpha = 0.07; ctx.translate(-Math.floor(r() * 256), -Math.floor(r() * 256));
  ctx.fillStyle = GRAIN_PAT; ctx.fillRect(0, 0, W + 256, H + 256); ctx.restore();
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
}


// ─────────────────────────────────────────────────────────────
//  Player: controls, captions, full screen, resume, end screen
// ─────────────────────────────────────────────────────────────
const capture = new URLSearchParams(location.search).has('capture');
if (capture) document.body.classList.add('capture');
const $ = id => document.getElementById(id);
const player = $('player'), stage = $('stage'), tl = $('tl'), fillEl = $('fill'), knob = $('knob'), tip = $('tip'), timeEl = $('time'), chapEl = $('chapters'),
      chapBtn = $('chap'), menu = $('menu'), capEl = $('cap'), capB = $('capb'), big = $('bigbtn'), playBtn = $('play'), muteBtn = $('mute'), volEl = $('vol'),
      ccBtn = $('cc'), rateBtn = $('rate'), fsBtn = $('fs'), snd = $('snd');
const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const ICON = {
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l10.5-6.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6.5 5h4v14h-4zM13.5 5h4v14h-4z"/></svg>',
  replay: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4v4.5h4.5"/></svg>',
  vol: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z" fill="currentColor"/><path d="M15.5 9a4.2 4.2 0 0 1 0 6M18 6.5a7.6 7.6 0 0 1 0 11"/></svg>',
  mute: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z" fill="currentColor"/><path d="M16 9.5l5 5M21 9.5l-5 5"/></svg>',
  list: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M9 6.5h11M9 12h11M9 17.5h11"/><circle cx="4.5" cy="6.5" r="1.2" fill="currentColor"/><circle cx="4.5" cy="12" r="1.2" fill="currentColor"/><circle cx="4.5" cy="17.5" r="1.2" fill="currentColor"/></svg>',
  hide: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12s3.2-6 9-6c1.6 0 3 .4 4.2 1M21 12s-3.2 6-9 6c-1.6 0-3-.4-4.2-1"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/><path d="M4 20 20 4"/></svg>',
  fs: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/></svg>',
  unfs: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5"/></svg>',
};
$('chbtn').innerHTML = ICON.list; $('hide').innerHTML = ICON.hide;
$('hide').onclick = e => { e.stopPropagation(); hideControls(); };

// preferences, remembered in this browser (storage can be blocked: everything works without it)
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
                set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} } };
const pref = Object.assign({ cc: true, rate: 1, vol: 1, muted: false }, store.get('gm-player', {}));
const RATES = [0.75, 1, 1.25, 1.5, 2];
if (!RATES.includes(pref.rate)) pref.rate = 1;
const savePref = () => store.set('gm-player', pref);
const POS_KEY = `gm-pos-${EPISODE.num}`;

let now = 0, playing = false, started = false, ended = false, lastTs = null, dragging = false, buffering = false;
let soundOn = !pref.muted;
// before the first play, show the title card (the end of the cold open) with a play button instead of a black frame
const POSTER_T = SC[0].dur - 2.5;
let resumeAt = store.get(POS_KEY, 0);
if (!(resumeAt > 15 && resumeAt < DURATION - 20)) resumeAt = 0;

// ── timeline: chapter ticks, hover/drag preview ──
SC.forEach((s, j) => { if (j) { const k = document.createElement('i'); k.className = 'tick'; k.style.left = `${100 * s.start / DURATION}%`; tl.querySelector('.track').appendChild(k); } });
tl.setAttribute('aria-valuemax', Math.round(DURATION));
const chLabel = j => `${String(j).padStart(2, '0')} ${SC[j].chapter}`;
const tAt = x => { const r = tl.getBoundingClientRect(); return clamp((x - r.left) / r.width) * DURATION; };
function showTip(x) { const r = tl.getBoundingClientRect(), T = tAt(x), j = sceneAt(T);
  tip.innerHTML = `<b>${String(j).padStart(2, '0')}</b> ${SC[j].chapter} · ${fmt(T)}`; tip.classList.add('on');
  const w = tip.offsetWidth / 2; tip.style.left = `${clamp(x - r.left, w, r.width - w)}px`; }
tl.addEventListener('pointerdown', e => { if (e.button > 0) return; e.preventDefault(); dragging = true; tl.classList.add('drag'); tl.setPointerCapture(e.pointerId); seekTo(tAt(e.clientX), false); showTip(e.clientX); });
tl.addEventListener('pointermove', e => { if (dragging) seekTo(tAt(e.clientX), false); if (dragging || e.pointerType === 'mouse') showTip(e.clientX); });
const endDrag = () => { if (!dragging) return; dragging = false; tl.classList.remove('drag'); tip.classList.remove('on'); lastTs = null; syncAudio(true); poke(); };
tl.addEventListener('pointerup', endDrag); tl.addEventListener('pointercancel', endDrag);
tl.addEventListener('pointerleave', () => { if (!dragging) tip.classList.remove('on'); });

// ── chapters: the list under the player and the menu inside it ──
SC.forEach((s, j) => {
  const b = document.createElement('button'); b.innerHTML = `${chLabel(j)} <time>${fmt(s.start)}</time>`; b.onclick = () => { seekTo(s.start); if (!playing) toggle(); }; chapEl.appendChild(b);
  const m = document.createElement('button'); m.setAttribute('role', 'menuitem'); m.innerHTML = `<b>${String(j).padStart(2, '0')}</b><span>${s.chapter}</span><time>${fmt(s.start)}</time>`;
  m.onclick = () => { seekTo(s.start); closeMenu(); if (!playing) toggle(); }; menu.appendChild(m); });
const openMenu = () => { player.classList.add('menu-open'); const on = menu.querySelector('.on'); if (on) menu.scrollTop = on.offsetTop - 60; };
const closeMenu = () => player.classList.remove('menu-open');
$('chbtn').onclick = chapBtn.onclick = e => { e.stopPropagation(); player.classList.contains('menu-open') ? closeMenu() : openMenu(); poke(); };

// ── what is on screen ──
let lastScene = -1, lastCap = null, lastBelow = null;
function capsAt(T) { const s = SC[sceneAt(T)], t = T - s.start; let str = '', al = 0;
  for (const [a, b, c] of s.caps) if (t >= a && t <= b) { str = c; al = Math.min(clamp((t - a) / 0.35), clamp((b - t) / 0.35)); }
  return [str, al]; }
function updateUI() {
  const p = 100 * now / DURATION;
  fillEl.style.width = `${p}%`; knob.style.left = `${p}%`;
  tl.setAttribute('aria-valuenow', Math.round(now)); tl.setAttribute('aria-valuetext', `${fmt(now)} of ${fmt(DURATION)}`);
  timeEl.innerHTML = `${fmt(now)} <span>/ ${fmt(DURATION)}</span>`;
  const i = sceneAt(now);
  if (i !== lastScene) { lastScene = i; chapBtn.textContent = `· ${chLabel(i)}`;
    [...chapEl.children].forEach((b, j) => b.classList.toggle('on', j === i)); [...menu.children].forEach((b, j) => b.classList.toggle('on', j === i));
    const b = chapEl.children[i]; if (b && chapEl.scrollWidth > chapEl.clientWidth) chapEl.scrollTo({ left: b.offsetLeft - 16, behavior: started ? 'smooth' : 'auto' }); }
  const [str, al] = started ? capsAt(now) : ['', 0], below = player.classList.contains('below');
  if (str !== lastCap || below !== lastBelow) { lastCap = str; lastBelow = below; (below ? capB : capEl).textContent = str; (below ? capEl : capB).textContent = ''; }
  (below ? capB : capEl).style.opacity = below ? (str ? Math.max(al, 0.25) : 0) : al;
  setUI();
}
function draw() { if (!started && !capture) render(POSTER_T, true); else render(now); updateUI(); }

// ── controls show while paused, after any touch or mouse movement, and while in use; then fade ──
// "hide controls" (button, H, or simply starting playback) clears them at once, even while paused; they come back
// on a real mouse movement (not the jitter of the click itself), a tap, a key, or pausing
let lastPoke = 0, overBar = false, touchMode = false, hidden = false, hideX = 0, hideY = 0, mouseX = 0, mouseY = 0;
const poke = () => { hidden = false; lastPoke = performance.now(); setUI(); };
function hideControls() { hidden = true; lastPoke = 0; overBar = false; hideX = mouseX; hideY = mouseY; closeMenu(); tip.classList.remove('on'); setUI(); }
function setUI() {
  const show = started && !hidden && (!playing || dragging || overBar || player.classList.contains('menu-open') || performance.now() - lastPoke < (touchMode ? 3200 : 2400));
  const c = player.classList;
  c.toggle('started', started); c.toggle('playing', playing); c.toggle('show', show); c.toggle('ended', ended); c.toggle('buffering', buffering && playing);
  c.toggle('idle', started && playing && !show); c.toggle('touch', touchMode); c.toggle('nocc', !pref.cc);
  const ic = playing ? 'pause' : ended ? 'replay' : 'play';
  if (playBtn.dataset.ic !== ic) { playBtn.dataset.ic = big.dataset.ic = ic; playBtn.innerHTML = big.innerHTML = ICON[ic]; playBtn.setAttribute('aria-label', `${ic} (k)`); big.setAttribute('aria-label', ic); }
}
// keep the controls' fade running while paused too (no animation loop then)
setInterval(() => { if (!playing) setUI(); }, 400);
const bar = $('bar');
bar.addEventListener('pointerleave', () => { overBar = false; if (!hidden) poke(); });
bar.addEventListener('pointerdown', e => { touchMode = e.pointerType !== 'mouse'; poke(); e.stopPropagation(); });

// ── clicking and tapping the picture ──
// mouse: click plays/pauses, double-click is full screen.  touch: tap shows/hides the controls,
// double-tap the left or right third skips 10 s, double-tap the middle is full screen.
let clickTimer = null, tapAt = 0, tapX = 0, stageTap = 0;
stage.addEventListener('pointermove', e => { if (e.pointerType !== 'mouse') return; touchMode = false; mouseX = e.clientX; mouseY = e.clientY;
  if (hidden && Math.hypot(mouseX - hideX, mouseY - hideY) < 12) return;
  overBar = !!e.target.closest('.bar'); poke(); });
stage.addEventListener('pointerup', e => {
  if (e.target.closest('.bar, .menu, .end, .bigbtn, .resume')) return;
  if (player.classList.contains('menu-open')) { closeMenu(); poke(); return; }
  if (e.pointerType === 'mouse') { touchMode = false; if (e.button) return;
    if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; toggleFS(); }
    else clickTimer = setTimeout(() => { clickTimer = null; toggle(); flash(); }, started ? 220 : 0);
    return; }
  touchMode = true; stageTap = performance.now();
  const t = performance.now(), r = stage.getBoundingClientRect(), x = (e.clientX - r.left) / r.width;
  if (!started) { toggle(); return; }
  if (t - tapAt < 320) { clearTimeout(clickTimer); clickTimer = null; tapAt = 0; if (x < 0.36) skip(-10); else if (x > 0.64) skip(10); else toggleFS(); return; }
  tapAt = t; tapX = x;
  clearTimeout(clickTimer); clickTimer = setTimeout(() => { clickTimer = null; const shown = player.classList.contains('show');
    if (shown) hideControls(); else poke(); }, 300);
});
// a tap that reveals the controls is followed by a click on whatever just appeared under the finger: ignore that one
big.onclick = e => { e.stopPropagation(); if (performance.now() - stageTap < 450) return; toggle(); };
function flash() { const f = $('flash'); f.innerHTML = ICON[playing ? 'play' : 'pause']; f.classList.remove('go'); void f.offsetWidth; f.classList.add('go'); }
function skip(d) { seekTo(now + d); const f = $(d < 0 ? 'skipl' : 'skipr'); f.classList.remove('go'); void f.offsetWidth; f.classList.add('go'); poke(); }

// ── sound ──
// Clock: while the soundtrack is really advancing, the picture follows it exactly (the ear notices drift first).
// If the <audio> clock stalls (some mobile in-app browsers report a frozen currentTime while still sounding) the picture
// falls back to the wall clock, so it never freezes; the audio is re-seeked only then, and at most every few seconds.
// Slow frames never slow the clock down: time comes from the audio or the wall, not from counting frames.
let lastSeek = -1e9, audT = -1, audWall = 0, waitWall = 0;
snd.volume = pref.vol; snd.playbackRate = pref.rate; volEl.value = pref.vol;
// load the whole soundtrack into memory, so any moment can be jumped to whatever the server supports. If play comes
// first, it streams meanwhile from a separate address (so the two downloads never wait on each other) and switches over.
// Opened as a local file it is seekable as it is.
const AUDIO = snd.dataset.src, inMemory = !capture && location.protocol !== 'file:' && !!(window.fetch && window.URL && URL.createObjectURL);
const streamAudio = () => { if (!snd.getAttribute('src')) { snd.src = AUDIO + (inMemory ? '?stream' : ''); snd.playbackRate = pref.rate; } };
if (inMemory) fetch(AUDIO).then(r => { if (!r.ok) throw new Error(r.status); return r.blob(); }).then(blob => {
    snd.src = URL.createObjectURL(blob); snd.playbackRate = pref.rate;
    snd.addEventListener('loadedmetadata', () => { snd.playbackRate = pref.rate; if (playing && soundOn) { seekAudio(now, true); snd.play().catch(() => {}); } }, { once: true });
  }).catch(streamAudio);
else streamAudio();
// the audio clock is only trusted while it agrees with the picture: a soundtrack that failed to jump (a server without
// range requests, a stream still loading) must never drag the picture back; the picture keeps its time and re-seeks it
function audioAlive(ts) { if (!soundOn || capture || snd.paused || snd.seeking) return false;
  const a = snd.currentTime; if (a !== audT) { audT = a; audWall = ts; }
  return ts - audWall < 500 && Math.abs(audT + (ts - audWall) / 1000 * pref.rate - now) < 1; }
function seekAudio(t, force) { if (snd.seeking && !force) return; try { snd.currentTime = t; } catch (e) {} lastSeek = performance.now(); audT = -1; audWall = performance.now(); }
function syncAudio(force) { if (!playing || !soundOn || capture || dragging) { if (!snd.paused) snd.pause(); return; }
  streamAudio();                                   // pressed play before the soundtrack finished loading
  if (force) seekAudio(now, true);
  if (snd.paused) snd.play().catch(() => {}); }
function setSound(on) { soundOn = on; pref.muted = !on; savePref(); muteBtn.innerHTML = on && pref.vol > 0 ? ICON.vol : ICON.mute; muteBtn.setAttribute('aria-label', on ? 'mute (m)' : 'unmute (m)'); syncAudio(true); }
muteBtn.onclick = () => { if (!soundOn && pref.vol === 0) { pref.vol = 1; volEl.value = 1; snd.volume = 1; } setSound(!soundOn); };
volEl.oninput = () => { pref.vol = +volEl.value; snd.volume = pref.vol; if (pref.vol > 0 && !soundOn) setSound(true); else if (pref.vol === 0 && soundOn) setSound(false); else { savePref(); setSound(soundOn); } };
setSound(soundOn);
function setRate(r) { pref.rate = r; savePref(); snd.playbackRate = r; rateBtn.textContent = `${r}×`; }
rateBtn.onclick = () => { setRate(RATES[(RATES.indexOf(pref.rate) + 1) % RATES.length]); poke(); };
setRate(pref.rate);
ccBtn.onclick = () => { pref.cc = !pref.cc; savePref(); ccBtn.setAttribute('aria-pressed', pref.cc); setUI(); poke(); };
ccBtn.setAttribute('aria-pressed', pref.cc);

// ── playing ──
let drawErr = 0, lastSave = 0;
function loop(ts) { if (!playing) return;
  const dt = lastTs == null ? 0 : Math.min(1, Math.max(0, (ts - lastTs) / 1000)); lastTs = ts;
  buffering = false;
  if (dragging) {}                                                                                      // the finger owns the clock
  else if (audioAlive(ts)) { now = audT + (ts - audWall) / 1000 * pref.rate; waitWall = 0; }            // audio clock, smoothed between its updates
  else if (soundOn && !snd.paused && snd.readyState < 3 && waitWall < 2) { waitWall += dt; buffering = waitWall > 0.25; }   // buffering: hold the picture briefly
  else { now += dt * pref.rate;                                                                          // no usable audio clock: wall clock
    if (soundOn && !snd.paused && Math.abs(snd.currentTime - now) > 0.5 && performance.now() - lastSeek > 4000) seekAudio(now); }
  // sound that can't be lined up with the picture yet (still streaming on a slow connection) stays silent, not wrong
  if (soundOn && !snd.paused) { const off = Math.abs(snd.currentTime - now) > 1; if (off && performance.now() - lastSeek > 1200) snd.muted = true; else if (!off) snd.muted = false; }
  if (now >= DURATION) finish();
  if (ts - lastSave > 3000) { lastSave = ts; savePos(); }
  try { draw(); } catch (e) { if (drawErr++ < 3) console.error(e); }       // one bad frame must never stop the film
  syncAudio(); if (playing) requestAnimationFrame(loop); }
function savePos() { store.set(POS_KEY, ended || now < 15 ? 0 : Math.round(now)); if (now > 0.9 * DURATION) markWatched(); }
function markWatched() { const n = +EPISODE.num, seen = store.get('gm-season-seen', {}) || {}; if (!seen[n]) { seen[n] = true; store.set('gm-season-seen', seen); } }
function finish() { now = DURATION; playing = false; ended = true; markWatched(); store.set(POS_KEY, 0); syncAudio(); draw(); }
function toggle() {
  if (!started && resumeAt) now = resumeAt;
  started = true; closeMenu();
  if (ended || now >= DURATION - 0.01) { now = 0; ended = false; }
  playing = !playing; lastTs = null; if (!playing) savePos();
  syncAudio(true); draw(); if (playing) { requestAnimationFrame(loop); hideControls(); } else poke(); }   // no waiting for the overlay to fade
function seekTo(T, audio = true) { started = true; now = clamp(T, 0, DURATION - 0.05); if (ended) ended = false; draw(); if (audio) syncAudio(true); }
playBtn.onclick = e => { e.stopPropagation(); toggle(); };
document.addEventListener('visibilitychange', () => { if (document.hidden) savePos(); });
window.addEventListener('pagehide', savePos);

// resume where you left off
if (resumeAt) { const r = $('resume'); r.classList.add('on'); r.innerHTML = `resumes at ${fmt(resumeAt)}<button type="button">start over</button>`;
  r.querySelector('button').onclick = e => { e.stopPropagation(); resumeAt = 0; store.set(POS_KEY, 0); toggle(); }; }

// ── full screen: the real thing where the browser allows it, otherwise the player fills the window ──
const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;
const isFS = () => !!fsElement() || player.classList.contains('pseudo');
function setPseudo(on) { player.classList.toggle('pseudo', on); document.documentElement.classList.toggle('noscroll', on); fit(); }
async function toggleFS() {
  if (isFS()) { if (fsElement()) { try { await (document.exitFullscreen || document.webkitExitFullscreen).call(document); } catch (e) {} } else setPseudo(false); return; }
  const req = player.requestFullscreen || player.webkitRequestFullscreen;
  try { if (!req) throw new Error('no fullscreen'); await req.call(player); if (!fsElement()) throw new Error('refused');
        if (touchMode) try { await screen.orientation.lock('landscape'); } catch (e) {} }
  catch (e) { setPseudo(true); }
  poke(); }
fsBtn.onclick = e => { e.stopPropagation(); toggleFS(); };
['fullscreenchange', 'webkitfullscreenchange'].forEach(ev => document.addEventListener(ev, () => { if (!fsElement()) try { screen.orientation.unlock(); } catch (e) {} fit(); }));

// size-dependent layout: caption size (--u = one 1080p pixel), captions under the picture when it is small
function fit() {
  const fs = isFS();
  if (fs) { const w = Math.min(player.clientWidth, player.clientHeight * 16 / 9); stage.style.width = `${w}px`; } else stage.style.width = '';
  const w = stage.clientWidth; player.style.setProperty('--u', `${w / 1920}px`);
  player.classList.toggle('below', !fs && w < 600);
  player.classList.toggle('narrow', w < 560);
  fsBtn.innerHTML = fs ? ICON.unfs : ICON.fs; fsBtn.setAttribute('aria-label', fs ? 'exit full screen (f)' : 'full screen (f)');
  updateUI(); }
window.addEventListener('resize', fit);
if (window.ResizeObserver) new ResizeObserver(() => fit()).observe(player);

// ── keyboard ──
window.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  const k = e.key, go = d => { e.preventDefault(); skip(d); };
  if (k === ' ' || k === 'k') { e.preventDefault(); toggle(); flash(); return; }
  if (k === 'h') { e.preventDefault(); player.classList.contains('show') ? hideControls() : poke(); return; }
  else if (k === 'ArrowRight') go(5); else if (k === 'ArrowLeft') go(-5);
  else if (k === 'l') go(10); else if (k === 'j') go(-10);
  else if (k === 'f') toggleFS();
  else if (k === 'm') muteBtn.click();
  else if (k === 'c') ccBtn.click();
  else if (k === '>' || k === '.') setRate(RATES[Math.min(RATES.length - 1, RATES.indexOf(pref.rate) + 1)]);
  else if (k === '<' || k === ',') setRate(RATES[Math.max(0, RATES.indexOf(pref.rate) - 1)]);
  else if (/^[0-9]$/.test(k)) seekTo(DURATION * +k / 10);
  else if (k === 'Home') seekTo(0); else if (k === 'End') seekTo(DURATION - 1);
  else if (k === 'Escape') { if (player.classList.contains('menu-open')) closeMenu(); else if (player.classList.contains('pseudo')) setPseudo(false); }
  else return;
  poke();
});

// ── title, end screen and lock-screen controls (the season table, labs/season.js, is defined after this file) ──
function season() {
  const G = window.GM, n = +EPISODE.num, e = G && G.eps[n - 1];
  if (!e) return;
  $('info').innerHTML = `<div class="k">episode ${EPISODE.num} · ${fmt(DURATION)}</div><h1>${e.title}</h1><p class="q">${e.q || ''}</p>
    <p class="keys"><kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> 5 s · <kbd>f</kbd> full screen · <kbd>c</kbd> captions · <kbd>m</kbd> mute · <kbd>h</kbd> hide controls · <kbd>&lt;</kbd><kbd>&gt;</kbd> speed</p>`;
  const nx = G.eps[n], nh = nx && G.href('ep', n + 1, '../'), lh = G.href('lab', n, '../'), tgt = G.framed ? ' target="_top"' : '';
  const next = !nx ? '<p>That was the last episode of the season.</p>'
    : nh ? `<a class="primary" href="${nh}"${tgt}>▶ episode ${String(n + 1).padStart(2, '0')} · ${nx.title}</a>`
    : nh === false ? `<span class="soon">episode ${String(n + 1).padStart(2, '0')} · ${nx.title} · coming soon</span>` : '';
  $('end-in').innerHTML = `<div class="k">episode ${EPISODE.num} complete</div><h2>${nx ? `Up next: ${nx.q || nx.title}` : e.title}</h2>
    <p>${lh ? 'Get your hands on it first: everything you just watched is in the playground.' : ''}</p>
    <div class="acts">${next}${lh ? `<a class="lab" href="${lh}"${tgt}>▶ playground · ${e.labName}</a>` : ''}<button class="act" type="button" id="replay">↺ watch again</button></div>`;
  $('replay').onclick = e2 => { e2.stopPropagation(); seekTo(0); ended = false; if (!playing) toggle(); };
  if ('mediaSession' in navigator) try {
    navigator.mediaSession.metadata = new MediaMetadata({ title: `${EPISODE.num} · ${e.title}`, artist: 'Generative Modelling', album: 'Season 1' });
    const ms = navigator.mediaSession;
    ms.setActionHandler('play', () => { if (!playing) toggle(); }); ms.setActionHandler('pause', () => { if (playing) toggle(); });
    ms.setActionHandler('seekbackward', () => skip(-10)); ms.setActionHandler('seekforward', () => skip(10));
    ms.setActionHandler('seekto', d => seekTo(d.seekTime));
  } catch (err) {}
}

window.DURATION = DURATION;
window.renderAt = T => render(T);
(async () => {
  try { await Promise.race([Promise.all([document.fonts.load(`400 40px Newsreader`), document.fonts.load(`italic 400 40px Newsreader`), document.fonts.load(`400 40px Karla`), document.fonts.load(`400 40px 'JetBrains Mono'`)]), new Promise(r => setTimeout(r, 4000))]); } catch (e) {}
  ON_READY.forEach(f => f());
  if (!capture) season();
  fit(); draw(); window.READY = true;
})();
