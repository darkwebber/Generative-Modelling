// ─────────────────────────────────────────────────────────────
//  Scene kit — helpers several episodes share (src/build.py inlines it after core.js)
// ─────────────────────────────────────────────────────────────
// base64 → bytes (exported assets store images and trajectories this way)
function b64bytes(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
// memoise expensive per-frame objects (image canvases, curves)
const IMC = new Map();
function cached(key, make) { let c = IMC.get(key); if (!c) { if (IMC.size > 4000) IMC.clear(); c = make(); IMC.set(key, c); } return c; }
const MEMO = new Map(); const memo = (k, f) => { if (!MEMO.has(k)) MEMO.set(k, f()); return MEMO.get(k); };
// a square 2-D plot of [−R, R]²: X(v), Y(v) map data → pixels; frame() draws it, clipTo() clips to it
function plotSquare(x0, y0, S, R) { return { X: v => x0 + (v + R) / (2 * R) * S, Y: v => y0 + (R - v) / (2 * R) * S, s: S / (2 * R), x0, y0, S }; }
function frame(m, a = 1) { setA(a); ctx.fillStyle = P.surface; ctx.fillRect(m.x0, m.y0, m.S, m.S); ctx.strokeStyle = P.border; ctx.lineWidth = 1.5; ctx.strokeRect(m.x0, m.y0, m.S, m.S); setA(1); }
function clipTo(m, fn) { ctx.save(); ctx.beginPath(); ctx.rect(m.x0, m.y0, m.S, m.S); ctx.clip(); fn(); ctx.restore(); }
// a glass marble (particles, samples)
function marble(x, y, r, col, a = 1) { if (GA * a <= 0.002) return; dot(x, y, r, col, a); dot(x - r * 0.32, y - r * 0.36, r * 0.36, P.chalk, a * 0.5); }
// 28×28 images: uint8 base64 → [0, 1] floats → tinted canvases
function u8imgs(s) { const u = b64bytes(s), out = []; for (let i = 0; i < u.length / 784; i++) out.push(Float32Array.from(u.subarray(i * 784, (i + 1) * 784), v => v / 255)); return out; }
function imgCanvas(vals, col = P.terracotta) {
  const c = document.createElement('canvas'); c.width = c.height = 28; const g = c.getContext('2d'); const d = g.createImageData(28, 28); const [r, gg, b] = hx(col);
  for (let i = 0; i < 784; i++) { d.data[i * 4] = r; d.data[i * 4 + 1] = gg; d.data[i * 4 + 2] = b; d.data[i * 4 + 3] = Math.round(clamp(vals[i]) * 255); }
  g.putImageData(d, 0, 0); return c;
}
function drawImg(c, x, y, s, a = 1, o = {}) {
  const { bg = true, frame = false, col = P.border } = o; if (a <= 0) return;
  if (bg) { setA(a); ctx.fillStyle = P.elevated; ctx.fillRect(x, y, s, s); }
  setA(a); ctx.imageSmoothingEnabled = s < 100; ctx.drawImage(c, x, y, s, s); ctx.imageSmoothingEnabled = true;
  if (frame) { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.strokeRect(x, y, s, s); }
  setA(1);
}
// a rubber stamp that lands (verdicts)
function stamp(s, x, y, col, a, rot = -0.12) { if (a <= 0) return; ctx.save(); ctx.translate(x, y); ctx.rotate(rot); const sc = lerp(1.6, 1, eout(a)); ctx.scale(sc, sc);
  ctx.font = `400 34px ${F.mono}`; const w = ctx.measureText(s).width + 36; setA(a); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.strokeRect(-w / 2, -30, w, 52); ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.fillText(s, 0, 8); ctx.restore(); setA(1); }
// standard normal sampler from a uniform rng
function gaussian(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(TAU * r()); }
// camera shake after an impact at t0
function shake(t, t0, amp = 10) { if (t < t0 || t > t0 + 0.8) return [0, 0]; const u = (t - t0) / 0.8, d = amp * Math.exp(-5 * u); return [d * Math.sin(u * 60), d * Math.cos(u * 47)]; }
