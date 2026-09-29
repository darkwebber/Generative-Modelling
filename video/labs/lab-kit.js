// lab-kit.js — shared by every playground: a "how to play" guide, missions that tick themselves off,
// tactile sound (synthesised with WebAudio, nothing to download), chunkier controls, tap ripples,
// tooltips and phone-friendly layout.
//
//   LabKit.init({ id: 'ep02', steps: ['…', '…', '…'], legend: [['var(--terracotta)', 'data'], …],
//                 missions: [{ id, title, hint, check: () => bool }, …] });
//   LabKit.sfx.pop(u) / chime(u) / thunk() / rattle() / clack(u) / roll(level) / …   (u in 0..1 → pitch)
//   LabKit.flag('name'), LabKit.flags.name, LabKit.count('name') — for mission checks
(function () {
  const LS = { get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
               set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked: fine */ } } };
  const K = window.LabKit = { flags: {}, counts: {} };
  K.flag = n => { K.flags[n] = true; };
  K.count = (n, by = 1) => (K.counts[n] = (K.counts[n] || 0) + by);
  K.throttle = (fn, ms) => { let last = 0; return (...a) => { const t = performance.now(); if (t - last >= ms) { last = t; fn(...a); } }; };

  // ── sound ─────────────────────────────────────────────────
  let ac = null, out = null, soundOn = LS.get('labkit.sound', true), NB = null;
  function A() {
    if (!soundOn) return null;
    try {
      if (!ac) { ac = new (window.AudioContext || window.webkitAudioContext)(); const comp = ac.createDynamicsCompressor(); comp.threshold.value = -20; comp.ratio.value = 4;
        out = ac.createGain(); out.gain.value = 0.55; out.connect(comp); comp.connect(ac.destination); }
      if (ac.state === 'suspended') ac.resume();
      return ac;
    } catch (e) { return null; }
  }
  function route(c, node, pan) { if (pan && c.createStereoPanner) { const p = c.createStereoPanner(); p.pan.value = Math.max(-1, Math.min(1, pan)); node.connect(p); p.connect(out); } else node.connect(out); }
  function env(g, t, a, peak, d) { g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(Math.max(0.0002, peak), t + a); g.gain.exponentialRampToValueAtTime(0.0001, t + a + d); }
  function osc(type, f, dt, a, peak, d, f2, pan) { const c = A(); if (!c) return; const t = c.currentTime + dt, o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(f, t); if (f2) o.frequency.exponentialRampToValueAtTime(f2, t + a + d); env(g, t, a, peak, d); o.connect(g); route(c, g, pan); o.start(t); o.stop(t + a + d + 0.05); }
  function noiseBuf(c) { if (!NB) { NB = c.createBuffer(1, c.sampleRate, c.sampleRate); const d = NB.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1; } return NB; }
  function nz(dt, d, peak, f, q, type = 'bandpass', pan = 0, f2 = 0, a = 0.002) { const c = A(); if (!c) return; const t = c.currentTime + dt, s = c.createBufferSource(), fl = c.createBiquadFilter(), g = c.createGain();
    s.buffer = noiseBuf(c); fl.type = type; fl.frequency.setValueAtTime(f, t); if (f2) fl.frequency.exponentialRampToValueAtTime(f2, t + d); fl.Q.value = q; env(g, t, a, peak, d);
    s.connect(fl); fl.connect(g); route(c, g, pan); s.start(t, Math.random() * 0.5); s.stop(t + a + d + 0.05); }
  const PENTA = [0, 2, 4, 7, 9];
  const note = u => { const i = Math.round(Math.max(0, Math.min(1, u)) * 14); return 261.63 * 2 ** ((PENTA[i % 5] + 12 * Math.floor(i / 5)) / 12); };   // C4 … C7, pentatonic
  let rollNode = null;
  K.sfx = {
    tick(u = 0.5, pan = 0) { osc('sine', note(u), 0, 0.002, 0.07, 0.05, 0, pan); nz(0, 0.012, 0.035, 4200, 1.4, 'bandpass', pan); },
    click() { nz(0, 0.018, 0.14, 2600, 1.4); osc('sine', 190, 0, 0.002, 0.1, 0.035, 90); },
    key() { nz(0, 0.014, 0.06, 3300, 2); osc('sine', 330, 0, 0.001, 0.025, 0.02); },
    pop(u = 0.5, pan = 0) { const f = note(u); osc('sine', f * 1.5, 0, 0.003, 0.16, 0.1, f, pan); nz(0, 0.008, 0.05, 3000, 1, 'bandpass', pan); },
    chime(u = 0.6, pan = 0, dt = 0) { const f = note(u); [[1, 0.13, 1.1], [2.76, 0.045, 0.45], [5.4, 0.02, 0.22]].forEach(([r, g, d]) => osc('sine', f * r, dt, 0.003, g, d, 0, pan)); },
    success() { [0.5, 0.64, 0.78, 0.93].forEach((u, i) => K.sfx.chime(u, 0, i * 0.085)); },
    thunk(pan = 0) { osc('sine', 150, 0, 0.003, 0.26, 0.16, 60, pan); nz(0, 0.05, 0.08, 500, 0.7, 'lowpass', pan); },
    stamp(pan = 0) { osc('sine', 110, 0, 0.002, 0.3, 0.12, 55, pan); nz(0, 0.035, 0.16, 1800, 0.8, 'bandpass', pan); },
    tock(u = 0.5, pan = 0) { const f = 700 + 900 * u; osc('sine', f, 0, 0.0015, 0.14, 0.05, 0, pan); osc('sine', f * 2.3, 0, 0.001, 0.05, 0.02, 0, pan); },
    clack(u = 0.5, pan = 0, dt = 0) { const f = 2300 + 1800 * u; osc('sine', f, dt, 0.001, 0.07, 0.03, 0, pan); osc('sine', f * 2.71, dt, 0.001, 0.035, 0.012, 0, pan); },
    patter(n = 10, dur = 0.4, pan = 0) { for (let i = 0; i < n; i++) K.sfx.clack(Math.random(), pan + (Math.random() - 0.5) * 0.6, Math.random() * dur); },
    rattle(dur = 0.9) { let t = 0, gap = 0.03; while (t < dur) { nz(t, 0.01, 0.09 * (0.5 + Math.random() * 0.5), 2600, 1.6); t += gap; gap *= 1.17; } },
    whoosh(dur = 0.6, up = true) { nz(0, dur, 0.09, up ? 350 : 3200, 0.9, 'bandpass', 0, up ? 3200 : 350, dur * 0.45); },
    scratch(pan = 0) { nz(0, 0.045, 0.045, 3600, 2.2, 'bandpass', pan); },
    // a continuous rolling / rumbling texture whose loudness follows `level` (0..1); call every frame
    roll(level) { const c = A(); if (!c) return; if (!rollNode) { const s = c.createBufferSource(); s.buffer = noiseBuf(c); s.loop = true; const fl = c.createBiquadFilter(); fl.type = 'bandpass'; fl.Q.value = 0.8; const g = c.createGain(); g.gain.value = 0; s.connect(fl); fl.connect(g); g.connect(out); s.start(); rollNode = { fl, g }; }
      const l = Math.max(0, Math.min(1, level)); rollNode.g.gain.setTargetAtTime(0.06 * l, c.currentTime, 0.05); rollNode.fl.frequency.setTargetAtTime(250 + 1400 * l, c.currentTime, 0.05); },
  };
  K.note = note;
  K.haptic = ms => { try { if (navigator.vibrate) navigator.vibrate(ms); } catch (e) { /* not available */ } };

  // ── styles ────────────────────────────────────────────────
  const css = `
  button { transition: transform .08s ease, box-shadow .15s ease, border-color .15s ease, background .15s ease; box-shadow: 0 2px 0 rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05); touch-action: manipulation; }
  button:hover { transform: translateY(-1px); }
  button:active { transform: translateY(1px); box-shadow: 0 0 0 rgba(0,0,0,.45), inset 0 2px 5px rgba(0,0,0,.45); }
  button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 2px solid var(--amber); outline-offset: 2px; }
  input[type=range] { -webkit-appearance: none; appearance: none; height: 24px; background: transparent; cursor: grab; touch-action: none; min-width: 0; }
  input[type=range]:active { cursor: grabbing; }
  input[type=range]::-webkit-slider-runnable-track { height: 6px; border-radius: 6px; background: linear-gradient(90deg, var(--amber) var(--fill, 50%), var(--border-light) var(--fill, 50%)); }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%; margin-top: -7px; background: var(--chalk); border: 4px solid var(--amber); box-shadow: 0 2px 6px rgba(0,0,0,.55); transition: transform .12s ease, box-shadow .12s ease; }
  input[type=range]:hover::-webkit-slider-thumb { transform: scale(1.1); }
  input[type=range]:active::-webkit-slider-thumb { transform: scale(1.28); box-shadow: 0 0 0 7px color-mix(in srgb, var(--amber) 22%, transparent); }
  input[type=range]::-moz-range-track { height: 6px; border-radius: 6px; background: var(--border-light); }
  input[type=range]::-moz-range-progress { height: 6px; border-radius: 6px; background: var(--amber); }
  input[type=range]::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: var(--chalk); border: 4px solid var(--amber); box-shadow: 0 2px 6px rgba(0,0,0,.55); }
  input[type=checkbox] { accent-color: var(--amber); width: 16px; height: 16px; vertical-align: -2px; cursor: pointer; }
  .lk-bump { animation: lkbump .22s ease; display: inline-block; }
  @keyframes lkbump { 40% { transform: scale(1.18); color: var(--chalk); } }
  .lk-ripple { position: fixed; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%; border: 2px solid var(--amber); pointer-events: none; z-index: 50; animation: lkripple .45s ease-out forwards; }
  @keyframes lkripple { to { transform: scale(4.2); opacity: 0; } }
  .lk-guide { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 0 0 14px; }
  .lk-step { display: flex; gap: 12px; align-items: flex-start; background: color-mix(in srgb, var(--surface) 80%, transparent); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; font-size: 14px; color: var(--stone); line-height: 1.45; min-width: 0; }
  .lk-step b { flex: none; width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; font-family: var(--serif); font-style: italic; font-weight: 400; font-size: 17px; color: var(--canvas); background: var(--amber); }
  .lk-step strong { color: var(--chalk); font-weight: 600; }
  .lk-legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 0 0 16px; font-family: var(--mono); font-size: 12px; color: var(--stone); }
  .lk-legend span { display: inline-flex; align-items: center; gap: 7px; }
  .lk-legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px currentColor; }
  .lk-missions { background: color-mix(in srgb, var(--surface) 90%, transparent); border: 1px solid var(--border); border-radius: 14px; padding: 14px 18px 16px; margin: 0 0 22px; }
  .lk-mhead { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .lk-mhead .lk-title { font-family: var(--mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--sage); }
  .lk-mhead .lk-count { font-family: var(--mono); font-size: 13px; color: var(--chalk); }
  .lk-mhead .lk-track { flex: 1 1 120px; height: 6px; border-radius: 6px; background: var(--border); overflow: hidden; min-width: 80px; }
  .lk-mhead .lk-track i { display: block; height: 100%; width: 0; background: var(--sage); border-radius: 6px; transition: width .5s cubic-bezier(.2,.9,.3,1.2); }
  .lk-mhead .lk-reset { background: none; border: none; box-shadow: none; color: var(--dust); font-family: var(--mono); font-size: 11px; padding: 2px 4px; cursor: pointer; text-decoration: underline; }
  .lk-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 18px; margin-top: 12px; }
  .lk-m { display: grid; grid-template-columns: 22px minmax(0, 1fr) auto; gap: 4px 10px; align-items: start; padding: 8px 10px; border-radius: 10px; border: 1px solid transparent; transition: background .3s, border-color .3s; }
  .lk-m .lk-dot { width: 20px; height: 20px; border-radius: 50%; border: 2px solid var(--border-light); margin-top: 1px; display: grid; place-items: center; font-size: 12px; color: var(--canvas); transition: all .3s; }
  .lk-m .lk-t { color: var(--chalk); font-size: 14.5px; line-height: 1.4; }
  .lk-m .lk-h { grid-column: 2 / 4; font-size: 13px; color: var(--dust); line-height: 1.45; display: none; }
  .lk-m.show .lk-h { display: block; }
  .lk-m .lk-hb { background: none; border: 1px solid var(--border); box-shadow: none; color: var(--dust); font-family: var(--mono); font-size: 10.5px; padding: 1px 7px; border-radius: 10px; }
  .lk-m.done { background: color-mix(in srgb, var(--sage) 9%, transparent); border-color: color-mix(in srgb, var(--sage) 30%, transparent); }
  .lk-m.done .lk-dot { background: var(--sage); border-color: var(--sage); }
  .lk-m.done .lk-dot::after { content: "✓"; font-weight: 700; }
  .lk-m.done .lk-t { color: var(--stone); }
  .lk-m.just { animation: lkjust .9s ease; }
  @keyframes lkjust { 20% { transform: scale(1.03); background: color-mix(in srgb, var(--sage) 25%, transparent); } }
  .lk-pill { position: fixed; right: 16px; bottom: calc(16px + env(safe-area-inset-bottom, 0px)); z-index: 40; background: color-mix(in srgb, var(--elevated) 94%, transparent); border: 1px solid color-mix(in srgb, var(--sage) 45%, transparent); color: var(--chalk); border-radius: 22px; padding: 8px 14px; font-family: var(--mono); font-size: 12px; max-width: min(360px, calc(100vw - 32px)); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; transition: opacity .3s, transform .3s; backdrop-filter: blur(6px); }
  .lk-pill.hide { opacity: 0; transform: translateY(12px); pointer-events: none; }
  .lk-pill b { color: var(--sage); font-weight: 400; }
  .lk-toast { position: fixed; left: 50%; top: calc(18px + env(safe-area-inset-top, 0px)); transform: translate(-50%, -20px); opacity: 0; z-index: 60; background: var(--elevated); border: 1px solid var(--sage); color: var(--chalk); border-radius: 12px; padding: 10px 16px; font-size: 14px; box-shadow: 0 10px 30px rgba(0,0,0,.5); transition: all .35s cubic-bezier(.2,.9,.3,1.2); max-width: calc(100vw - 32px); pointer-events: none; }
  .lk-toast.on { opacity: 1; transform: translate(-50%, 0); }
  .lk-toast small { display: block; color: var(--sage); font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; }
  .lk-sound { position: fixed; top: calc(12px + env(safe-area-inset-top, 0px)); right: 16px; z-index: 41; font-size: 12px; padding: 6px 12px; border-radius: 18px; }
  [data-tip] { position: relative; text-decoration: underline dotted color-mix(in srgb, var(--stone) 60%, transparent); text-underline-offset: 3px; cursor: help; }
  [data-tip]:hover::after, [data-tip]:focus::after, [data-tip].tip-on::after { content: attr(data-tip); position: absolute; left: 0; top: calc(100% + 6px); z-index: 30; width: max-content; max-width: min(300px, 80vw); white-space: normal; background: var(--elevated); color: var(--chalk); border: 1px solid var(--border-light); border-radius: 8px; padding: 8px 10px; font-family: var(--sans); font-size: 13px; line-height: 1.45; text-transform: none; letter-spacing: 0; box-shadow: 0 8px 24px rgba(0,0,0,.5); }
  canvas { touch-action: none; }
  @media (max-width: 760px) { .lk-guide { grid-template-columns: 1fr; } .lk-list { grid-template-columns: 1fr; } }
  @media (max-width: 560px) { .knobs { grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr) 48px !important; gap: 10px 10px !important; font-size: 12px !important; } .mix { grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr) 36px !important; } .lk-sound { position: absolute; top: 10px; } .lk-pill { left: 16px; right: 16px; max-width: none; text-align: center; font-size: 11.5px; padding: 7px 12px; } }
  @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }`;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);

  // ── controls: fill, bump, sounds ──────────────────────────
  function fill(r) { const mn = +r.min || 0, mx = r.max === '' ? 100 : +r.max; r.style.setProperty('--fill', ((+r.value - mn) / ((mx - mn) || 1) * 100).toFixed(1) + '%'); }
  function readoutOf(r) { return (r.id && document.getElementById(r.id + 'v')) || (r.nextElementSibling && /^(SPAN|B)$/.test(r.nextElementSibling.tagName) ? r.nextElementSibling : null); }
  const lastVal = new WeakMap(), rangeTick = K.throttle((u) => K.sfx.tick(u), 28);
  document.addEventListener('input', e => {
    const el = e.target;
    if (el.type === 'range') { fill(el); const mn = +el.min || 0, mx = +el.max || 100, u = (+el.value - mn) / ((mx - mn) || 1), steps = (mx - mn) / (+el.step || 1);
      if (lastVal.get(el) !== el.value) { if (steps <= 60) { K.sfx.tick(u); K.haptic(4); } else rangeTick(u); lastVal.set(el, el.value); }
      const ro = readoutOf(el); if (ro) { ro.classList.remove('lk-bump'); void ro.offsetWidth; ro.classList.add('lk-bump'); } }
    else if (el.tagName === 'TEXTAREA' || el.type === 'text') keyTick();
  }, true);
  const keyTick = K.throttle(() => K.sfx.key(), 25);
  document.addEventListener('change', e => { if (e.target.type === 'checkbox') { K.sfx.click(); K.sfx.tick(e.target.checked ? 0.8 : 0.3); } }, true);
  document.addEventListener('pointerdown', e => {
    const b = e.target.closest('button'); if (b && !b.classList.contains('lk-quiet')) { K.sfx.click(); K.haptic(6); }
    if (e.target.tagName === 'CANVAS') { const r = document.createElement('span'); r.className = 'lk-ripple'; r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px'; document.body.appendChild(r); setTimeout(() => r.remove(), 500); }
    const tip = e.target.closest('[data-tip]'); document.querySelectorAll('.tip-on').forEach(x => { if (x !== tip) x.classList.remove('tip-on'); }); if (tip) tip.classList.toggle('tip-on');
  }, true);
  setInterval(() => document.querySelectorAll('input[type=range]').forEach(fill), 250);

  // ── confetti ──────────────────────────────────────────────
  let fx = null, parts = [];
  function burst(x, y, n = 36) {
    if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!fx) { fx = document.createElement('canvas'); fx.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:55'; document.body.appendChild(fx); }
    fx.width = innerWidth; fx.height = innerHeight; const cols = ['#c4654a', '#7a9e7e', '#6889b1', '#d4a853', '#b5707e', '#8b7bb5'];
    for (let i = 0; i < n; i++) { const a = Math.random() * Math.PI * 2, s = 3 + Math.random() * 6; parts.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 4, life: 1, c: cols[i % cols.length], r: 2 + Math.random() * 3 }); }
    if (parts.length === n) requestAnimationFrame(tick);
  }
  function tick() { const g = fx.getContext('2d'); g.clearRect(0, 0, fx.width, fx.height);
    parts.forEach(p => { p.x += p.vx; p.y += p.vy; p.vy += 0.25; p.vx *= 0.98; p.life -= 0.016; g.globalAlpha = Math.max(0, p.life); g.fillStyle = p.c; g.beginPath(); g.arc(p.x, p.y, p.r, 0, 7); g.fill(); });
    parts = parts.filter(p => p.life > 0); if (parts.length) requestAnimationFrame(tick); else g.clearRect(0, 0, fx.width, fx.height); }
  K.burst = burst;

  // ── toast ─────────────────────────────────────────────────
  let toastEl = null, toastT = null;
  K.toast = (msg, head = 'mission complete') => { if (!toastEl) { toastEl = document.createElement('div'); toastEl.className = 'lk-toast'; toastEl.setAttribute('role', 'status'); document.body.appendChild(toastEl); }
    toastEl.innerHTML = `<small>${head}</small>${msg}`; toastEl.classList.add('on'); clearTimeout(toastT); toastT = setTimeout(() => toastEl.classList.remove('on'), 2800); };

  // ── init: guide, legend, missions ─────────────────────────
  K.init = function (cfg) {
    const anchor = document.querySelector(cfg.anchor || '.sub') || document.querySelector('h1');
    const sb = document.createElement('button'); sb.className = 'lk-sound lk-quiet'; sb.title = 'sound on / off';
    const lab = () => { sb.textContent = soundOn ? '♪ sound on' : '♪ sound off'; }; lab();
    sb.onclick = () => { soundOn = !soundOn; LS.set('labkit.sound', soundOn); lab(); if (soundOn) K.sfx.chime(0.7); else if (rollNode) rollNode.g.gain.value = 0; };
    document.body.appendChild(sb);
    let html = '';
    if (cfg.missions && cfg.missions.length) cfg.steps = cfg.steps.map((x, i) => i === cfg.steps.length - 1 ? x + ' <span style="color:var(--sage)">The ◎ pill (bottom right) tracks your progress.</span>' : x);
    if (cfg.steps) html += `<div class="lk-guide">${cfg.steps.map((s, i) => `<div class="lk-step"><b>${i + 1}</b><span>${s}</span></div>`).join('')}</div>`;
    if (cfg.legend) html += `<div class="lk-legend">${cfg.legend.map(([c, s]) => `<span><i style="background:${c};color:${c}"></i>${s}</span>`).join('')}</div>`;
    const M = cfg.missions || [], key = `labkit.${cfg.id}.done`; let done = new Set(LS.get(key, []));
    if (M.length) html += `<section class="lk-missions" id="lk-missions" aria-label="missions"><div class="lk-mhead"><span class="lk-title">missions</span><span class="lk-count"></span><span class="lk-track"><i></i></span><button class="lk-reset lk-quiet" type="button">reset missions</button></div><div class="lk-list">${M.map(m => `<div class="lk-m" data-id="${m.id}"><span class="lk-dot"></span><span class="lk-t">${m.title}</span><button class="lk-hb" type="button">hint</button><span class="lk-h">${m.hint}</span></div>`).join('')}</div></section>`;
    const wrap = document.createElement('div'); wrap.innerHTML = html; const mEl = wrap.querySelector('#lk-missions'); if (mEl) mEl.remove(); anchor.after(...wrap.childNodes);
    const foot = document.querySelector(cfg.missionsBefore || 'footer'); if (mEl) { if (foot) foot.before(mEl); else anchor.parentElement.appendChild(mEl); mEl.style.marginTop = '22px'; }
    if (!M.length) return;
    const box = document.getElementById('lk-missions'), pill = document.createElement('div'); pill.className = 'lk-pill hide'; pill.onclick = () => box.scrollIntoView({ behavior: 'smooth', block: 'center' }); document.body.appendChild(pill);
    box.querySelectorAll('.lk-hb').forEach(b => b.onclick = () => { b.parentElement.classList.toggle('show'); b.textContent = b.parentElement.classList.contains('show') ? 'hide' : 'hint'; });
    box.querySelector('.lk-reset').onclick = () => { done = new Set(); LS.set(key, []); K.flags = {}; K.counts = {}; render(); };
    let boxVisible = true;
    try { new IntersectionObserver(es => { boxVisible = es[0].isIntersecting; render(); }).observe(box); } catch (e) { boxVisible = false; }
    function render() { const n = done.size; box.querySelector('.lk-count').textContent = `${n} / ${M.length}`; box.querySelector('.lk-track i').style.width = (n / M.length * 100) + '%';
      box.querySelectorAll('.lk-m').forEach(el => el.classList.toggle('done', done.has(el.dataset.id)));
      const next = M.find(m => !done.has(m.id));
      pill.innerHTML = next ? `◎ ${n}/${M.length} · next: <b>${next.title.replace(/<[^>]+>/g, '')}</b>` : `◎ all ${M.length} missions done`;
      pill.classList.toggle('hide', boxVisible); }
    render();
    setInterval(() => {
      for (const m of M) { if (done.has(m.id)) continue; let ok = false; try { ok = !!m.check(); } catch (e) { ok = false; }
        if (!ok) continue;
        done.add(m.id); LS.set(key, [...done]); render();
        const el = box.querySelector(`[data-id="${m.id}"]`); el.classList.remove('just'); void el.offsetWidth; el.classList.add('just');
        const all = done.size === M.length;
        K.toast(all ? `${m.title.replace(/<[^>]+>/g, '')}<br><span style="color:var(--sage)">All ${M.length} missions done. You’ve got this episode.</span>` : m.title.replace(/<[^>]+>/g, ''));
        if (all) K.sfx.success(); else { K.sfx.chime(0.72); K.sfx.chime(0.86, 0, 0.1); }
        K.haptic([10, 40, 10]);
        const r = (boxVisible ? el : pill).getBoundingClientRect(); burst(r.left + 24, r.top + r.height / 2, all ? 90 : 34);
        break;   // one celebration at a time
      }
    }, 300);
  };
})();
