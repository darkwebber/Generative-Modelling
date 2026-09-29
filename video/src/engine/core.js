// ─────────────────────────────────────────────────────────────
//  Midnight Atelier engine — tokens, maths, drawing helpers.
//  Shared by every episode (src/build.py inlines it into each page).
//  Colour-coded maths:  M('[d|x] + [a_|θ] … [k^|2] … [pB|∫]')  d/p/a/n/m/s/k/w/c/t = colour, _ sub, ^ sup, B big
// ─────────────────────────────────────────────────────────────
const P = {
  canvas:"#13100e", surface:"#1c1816", elevated:"#272220", border:"#3a3432", borderLight:"#4d4540",
  chalk:"#ede6db", stone:"#a89f97", dust:"#6b6360",
  terracotta:"#c4654a", sage:"#7a9e7e", ink:"#6889b1", amber:"#d4a853", clay:"#a0735c", rose:"#b5707e", plum:"#8b7bb5",
};
const F = { serif:"'Newsreader', Georgia, serif", sans:"'Karla', sans-serif", mono:"'JetBrains Mono', monospace" };
const W = 1920, H = 1080;
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');

const TAU = Math.PI * 2;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, u) => a + (b - a) * u;
const prog = (t, a, d) => clamp((t - a) / d);
const ease = x => { x = clamp(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eout = x => { x = clamp(x); return 1 - Math.pow(1 - x, 3); };
const ein = x => { x = clamp(x); return x * x * x; };
const pop = x => { x = clamp(x); const c = 1.9; return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };
function rng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

const HX = {};
function hx(h) { return HX[h] || (HX[h] = (n => [n >> 16, (n >> 8) & 255, n & 255])(parseInt(h.slice(1), 16))); }
function rgba(h, a) { const c = hx(h); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; }
function mixc(a, b, u) { const A = hx(a), B = hx(b); return `rgb(${Math.round(A[0] + (B[0] - A[0]) * u)},${Math.round(A[1] + (B[1] - A[1]) * u)},${Math.round(A[2] + (B[2] - A[2]) * u)})`; }

let GA = 1;
const setA = a => { ctx.globalAlpha = clamp(GA * a); };
function text(s, x, y, o = {}) {
  const { font = F.sans, size = 28, color = P.stone, align = 'left', a = 1, italic = false, weight = 400, ls = 0 } = o;
  if (GA * a <= 0.002) return;  // skip invisible draws (some browsers ignore globalAlpha = 0)
  ctx.font = `${italic ? 'italic ' : ''}${weight} ${size}px ${font}`;
  ctx.letterSpacing = ls + 'px'; ctx.textAlign = align; ctx.fillStyle = color; setA(a);
  ctx.fillText(s, x, y); ctx.letterSpacing = '0px'; ctx.textAlign = 'left'; setA(1);
}
function wrap(str, maxW, font) {
  ctx.font = font; const words = str.split(' '); const lines = []; let cur = '';
  for (const w of words) { const test = cur ? cur + ' ' + w : w; if (ctx.measureText(test).width > maxW && cur) { lines.push(cur); cur = w; } else cur = test; }
  if (cur) lines.push(cur); return lines;
}
function para(str, x, y, maxW, o = {}) {
  const { size = 24, lh = 1.45, color = P.stone, a = 1, font = F.sans, italic = false } = o;
  const lines = wrap(str, maxW, `${italic ? 'italic ' : ''}400 ${size}px ${font}`);
  lines.forEach((l, i) => text(l, x, y + i * size * lh, { font, size, color, a, italic }));
  return lines.length * size * lh;
}
const CM = { d:P.terracotta, p:P.ink, a:P.amber, n:P.rose, m:P.plum, s:P.sage, k:P.dust, w:P.chalk, c:P.clay, t:P.stone };
const MCACHE = new Map();
function parseM(str) {
  if (MCACHE.has(str)) return MCACHE.get(str);
  const out = []; const re = /\[([a-z])([_^B]?)\|([^\]]*)\]/g; let last = 0, m;
  while ((m = re.exec(str))) {
    if (m.index > last) out.push([str.slice(last, m.index), P.stone, '']);
    out.push([m[3], CM[m[1]], m[2] === '_' ? 's' : m[2] === '^' ? 'S' : m[2] === 'B' ? 'b' : '']);
    last = re.lastIndex;
  }
  if (last < str.length) out.push([str.slice(last), P.stone, '']);
  MCACHE.set(str, out); return out;
}
function M(str, x, y, size = 34, o = {}) {
  const { align = 'left', a = 1 } = o;
  const parts = parseM(str).map(([s, col, mod]) => {
    const sz = mod === 's' || mod === 'S' ? size * 0.66 : mod === 'b' ? size * 1.3 : size;
    ctx.font = `400 ${sz}px ${F.mono}`;
    return { s, col, sz, dy: mod === 's' ? size * 0.26 : mod === 'S' ? -size * 0.42 : mod === 'b' ? size * 0.1 : 0, w: ctx.measureText(s).width };
  });
  const total = parts.reduce((acc, p) => acc + p.w, 0);
  if (GA * a <= 0.002) return total;
  let cx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
  setA(a); ctx.textAlign = 'left';
  for (const p of parts) { ctx.font = `400 ${p.sz}px ${F.mono}`; ctx.fillStyle = p.col; ctx.fillText(p.s, cx, y + p.dy); cx += p.w; }
  setA(1); return total;
}
function dot(x, y, r, col, a = 1) { if (GA * a <= 0.002) return; setA(a); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, Math.max(0, r), 0, TAU); ctx.fill(); setA(1); }
function glow(x, y, r, col, a = 1) { ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = r * 3.5; dot(x, y, r, col, a); ctx.restore(); }
function ring(x, y, r, col, a = 1, lw = 2) { if (GA * a <= 0.002) return; setA(a); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath(); ctx.arc(x, y, Math.max(0, r), 0, TAU); ctx.stroke(); setA(1); }
function line(x1, y1, x2, y2, col, lw = 2, a = 1, dash = null) { if (GA * a <= 0.002) return; setA(a); ctx.strokeStyle = col; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]); setA(1); }
function arrow(x1, y1, x2, y2, col, lw = 2, a = 1) {
  line(x1, y1, x2, y2, col, lw, a);
  const ang = Math.atan2(y2 - y1, x2 - x1), L = 12;
  setA(a); ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - L * Math.cos(ang - 0.4), y2 - L * Math.sin(ang - 0.4));
  ctx.lineTo(x2 - L * Math.cos(ang + 0.4), y2 - L * Math.sin(ang + 0.4)); ctx.fill(); setA(1);
}
function rbox(x, y, w, h, r, fill, stroke, a = 1, lw = 1.5) { setA(a); ctx.beginPath(); ctx.roundRect(x, y, w, h, r); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } setA(1); }
function card(x, y, w, h, accent, a = 1) {
  if (a <= 0) return;
  ctx.save(); setA(a);
  ctx.beginPath(); ctx.roundRect(x, y, w, h, 18); ctx.fillStyle = rgba(P.surface, 0.86); ctx.fill();
  ctx.strokeStyle = P.border; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.clip(); ctx.fillStyle = rgba(accent, 0.6); ctx.fillRect(x, y, 4, h);
  ctx.restore(); setA(1);
}
function label(s, x, y, a = 1, color = P.terracotta) { text(s.toUpperCase(), x, y, { font: F.mono, size: 16, color, a, ls: 2 }); }
function tag(x, y, color, s, a = 1, size = 19) {
  ctx.font = `400 ${size}px ${F.mono}`; const w = ctx.measureText(s).width + size * 2.1; const h = size * 1.75;
  if (GA * a <= 0.002) return w;
  ctx.save(); setA(a);
  ctx.beginPath(); ctx.roundRect(x, y, w, h, h / 2); ctx.fillStyle = rgba(color, 0.08); ctx.fill();
  ctx.strokeStyle = rgba(color, 0.3); ctx.lineWidth = 1.2; ctx.stroke();
  ctx.shadowColor = color; ctx.shadowBlur = 8; ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x + size * 0.8, y + h / 2, size * 0.22, 0, TAU); ctx.fill();
  ctx.shadowBlur = 0; ctx.fillText(s, x + size * 1.35, y + h * 0.68);
  ctx.restore(); setA(1); return w;
}
function pill(x, y, s, color, a = 1, o = {}) {
  const { size = 20, align = 'left', font = F.mono } = o;
  ctx.font = `400 ${size}px ${font}`; const w = ctx.measureText(s).width + size * 1.2, h = size * 1.7;
  const x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  if (GA * a <= 0.002) return w;
  ctx.save(); setA(a); ctx.beginPath(); ctx.roundRect(x0, y - h / 2, w, h, h / 2);
  ctx.fillStyle = rgba(P.canvas, 0.85); ctx.fill(); ctx.strokeStyle = rgba(color, 0.45); ctx.lineWidth = 1.3; ctx.stroke();
  ctx.fillStyle = color; ctx.fillText(s, x0 + size * 0.6, y + size * 0.35); ctx.restore(); setA(1);
  return w;
}
function insight(x, y, w, s, a = 1, head = 'intuition') {
  if (a <= 0) return 0;
  const lines = wrap(s, w - 48, `400 25px ${F.sans}`); const h = 70 + lines.length * 35;
  ctx.save(); setA(a); ctx.beginPath(); ctx.roundRect(x, y, w, h, 16);
  ctx.fillStyle = rgba(P.sage, 0.07); ctx.fill(); ctx.strokeStyle = rgba(P.sage, 0.3); ctx.lineWidth = 1.3; ctx.stroke(); ctx.restore();
  text(head, x + 24, y + 38, { font: F.serif, italic: true, size: 22, color: P.sage, a });
  lines.forEach((l, i) => text(l, x + 24, y + 76 + i * 35, { size: 25, color: P.stone, a }));
  return h;
}
function gaussFn(r) { return () => { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v); }; }
function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
const npdf = (x, m = 0, s = 1) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(TAU));
function tagsRow(list, x, y, t0, t) { let cx = x; list.forEach(([c, s], i) => { cx += tag(cx, y, c, s, eout(prog(t, t0 + i * 0.25, 0.6))) + 12; }); }
// hooks that run once the fonts have loaded (e.g. glyphs rasterised from a font)
const ON_READY = [];
