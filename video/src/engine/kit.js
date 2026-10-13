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
// ── the season, for the end-card strip: watched episodes lit, this one ringed, the next one pulsing
const SEASON = ['what is p(x)', 'autoregressive', 'VAEs', 'evaluation', 'GANs', 'flows', 'energy', 'score', 'diffusion', 'flow matching', 'guidance'];
function seasonStrip(cur, t0, t, y = 830) {
  const a = eout(prog(t, t0, 0.8)); if (GA * a <= 0.002) return;
  const n = SEASON.length, x0 = 500, x1 = 1420, X = i => x0 + (x1 - x0) * i / (n - 1);
  line(x0, y, x1, y, P.border, 1.5, a);
  const fill = eout(prog(t, t0 + 0.2, 0.07 * cur + 0.3)); line(x0, y, lerp(x0, X(cur - 1), fill), y, P.terracotta, 2, a * 0.8);
  SEASON.forEach((name, i) => {
    const ep = i + 1, u = prog(t, t0 + 0.2 + i * 0.07, 0.35), s = pop(u), x = X(i);
    let col = P.dust, r = 4;
    if (ep <= cur) { col = P.terracotta; r = 6; }
    if (ep === cur + 1) { const pl = 0.5 + 0.5 * Math.sin((t - t0) * 4.5); col = P.amber; r = 6; ring(x, y, 10 + 5 * pl, P.amber, a * u * (0.8 - 0.6 * pl), 1.5); }
    dot(x, y, r * s, col, a * u);
    if (ep === cur) ring(x, y, 11 * s, P.chalk, a * u, 1.8);
    text(String(ep).padStart(2, '0'), x, y + 30, { font: F.mono, size: 12, color: ep === cur + 1 ? P.amber : ep <= cur ? P.stone : P.dust, align: 'center', a: a * u });
    text(name, x, y + 50, { size: 14, color: ep === cur + 1 ? P.amber : ep <= cur ? P.stone : P.dust, align: 'center', a: a * u * (ep <= cur + 1 ? 1 : 0.7) });
  });
}
// ── "try it yourself": the episode's playground, named in the header area of the build-it scene
function playPill(file, t0, t) { pill(1840, 150, `▶ playground · labs/${file}`, P.amber, eout(prog(t, t0, 0.6)), { size: 17, align: 'right' }); }

// ── 3-D surfaces ─────────────────────────────────────────────
// surf3d(o) draws a height field z = h(x, y) over the square [−R, R]², lit and depth-sorted, and returns a camera
// for anything that sits on it: { P(x, y, z) → [sx, sy, depth], z(x, y) → surface height }.
//   o.h: Float32Array of n×n heights (0..1, row 0 at y = +R) · o.n · o.R
//   o.cx, o.cy: screen centre · o.scale: pixels per world unit · o.hs: world height of h = 1
//   o.yaw: turn about the vertical (0 = map orientation) · o.pitch: 0 = straight down (a flat map), π/2 = side-on
//   o.lo, o.hi: colours for low and high ground (lit) · o.step: grid subsampling · o.wire: mesh-line alpha
//   o.items: [{ x, y, z, draw(sx, sy, depth) }] drawn in depth order with the surface, so ridges hide them
//   o.gamma: colour contrast by height (below 1 lifts low ground) · o.a: alpha.  At pitch 0 the surface lines up exactly with square(…) maps of the same R, so a flat map can tilt up.
function surfCam(o) {             // the camera alone (to place things on the surface before drawing it)
  const { h, n, R, cx, cy, scale, hs = 1, yaw = 0, pitch = 0, persp = 0.18 } = o;
  const cy_ = Math.cos(yaw), sy_ = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), D = R * (1 / Math.max(persp, 1e-3) + 1);
  const P_ = (x, y, z) => { const x1 = x * cy_ - y * sy_, y1 = x * sy_ + y * cy_, d = z * cp - y1 * sp, f = 1 + (persp ? D / (D - d) - 1 : 0) * sp;   // perspective grows with the tilt
    return [cx + scale * x1 * f, cy - scale * (y1 * cp + z * sp - hs * 0.45 * sp) * f, d]; };
  const H = (i, j) => h[Math.min(n - 1, j) * n + Math.min(n - 1, i)];
  const zAt = (x, y) => { const fi = clamp((x + R) / (2 * R) * (n - 1), 0, n - 1.001), fj = clamp((R - y) / (2 * R) * (n - 1), 0, n - 1.001), i = Math.floor(fi), j = Math.floor(fj), u = fi - i, v = fj - j;
    return ((H(i, j) * (1 - u) + H(i + 1, j) * u) * (1 - v) + (H(i, j + 1) * (1 - u) + H(i + 1, j + 1) * u) * v) * hs; };
  return { P: P_, z: zAt, H };
}
// size and centre a view so the whole surface (heights included) fits the box x0, y0, w, h at its current angle
function surfFit(o, x0, y0, w, h) {
  const c = surfCam({ ...o, cx: 0, cy: 0, scale: 1 }); let a = 1e9, b = -1e9, cmin = 1e9, cmax = -1e9;
  for (let j = 0; j <= 12; j++) for (let i = 0; i <= 12; i++) { const x = -o.R + 2 * o.R * i / 12, y = o.R - 2 * o.R * j / 12;
    for (const z of [0, c.z(x, y)]) { const p = c.P(x, y, z); a = Math.min(a, p[0]); b = Math.max(b, p[0]); cmin = Math.min(cmin, p[1]); cmax = Math.max(cmax, p[1]); } }
  const sc = Math.min(w / (b - a), h / (cmax - cmin));
  return { ...o, scale: sc, cx: x0 + w / 2 - sc * (a + b) / 2, cy: y0 + h / 2 - sc * (cmin + cmax) / 2 };
}
const SURF_COLS = {};
function surf3d(o) {
  const { h, n, R, hs = 1, lo = '#151110', hi = '#7a6ca6', step = 2, wire = 0.16, items = [], a = 1, gamma = 1.3 } = o;
  const cam = surfCam(o), P_ = cam.P, H = cam.H, zAt = cam.z;
  if (GA * a <= 0.002) return cam;
  // o.cache: a key for a surface that is not changing (same heights, camera and alpha). The finished surface is kept
  // as an image and reused, and only the items are drawn each frame (on top: fine for a still camera)
  const C_ = o.cache && SURF_CACHE[o.cache];
  if (C_) { ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.drawImage(C_.img, C_.x, C_.y); ctx.restore();
    items.forEach(it => { const p = P_(it.x, it.y, it.z ?? zAt(it.x, it.y)); it.draw(p[0], p[1], p[2]); }); return cam; }
  const m = Math.floor((n - 1) / step) + 1, X = k => -R + 2 * R * Math.min(n - 1, k * step) / (n - 1), Y = k => R - 2 * R * Math.min(n - 1, k * step) / (n - 1);
  const V = new Array(m * m);
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) V[j * m + i] = P_(X(i), Y(j), H(i * step, j * step) * hs);
  const L = [-0.45, 0.5, 0.74], c0 = hx(lo), c1 = hx(hi), cell = 2 * R * step / (n - 1), list = [], cache = SURF_COLS[lo + hi] || (SURF_COLS[lo + hi] = new Map());
  for (let j = 0; j < m - 1; j++) for (let i = 0; i < m - 1; i++) {
    const z00 = H(i * step, j * step), z10 = H((i + 1) * step, j * step), z01 = H(i * step, (j + 1) * step), z11 = H((i + 1) * step, (j + 1) * step);
    const gx = ((z10 + z11) - (z00 + z01)) / 2 * hs / cell, gy = ((z00 + z10) - (z01 + z11)) / 2 * hs / cell, nl = Math.hypot(gx, gy, 1);
    const lam = Math.max(0, (-gx * L[0] - gy * L[1] + L[2]) / nl), lit = 0.28 + 0.78 * lam + 0.25 * Math.pow(lam, 12), u = Math.pow((z00 + z10 + z01 + z11) / 4, gamma);   // soft key light + a little sheen
    const key = (Math.round(u * 63) << 7) | Math.round(Math.min(lit, 1.6) * 79);       // colours come from a small cache, not a string per tile
    let col = cache.get(key); if (!col) { const uq = (key >> 7) / 63, lq = (key & 127) / 79; col = `rgb(${clamp((c0[0] + (c1[0] - c0[0]) * uq) * lq, 0, 255) | 0},${clamp((c0[1] + (c1[1] - c0[1]) * uq) * lq, 0, 255) | 0},${clamp((c0[2] + (c1[2] - c0[2]) * uq) * lq, 0, 255) | 0})`; cache.set(key, col); }
    const q = [V[j * m + i], V[j * m + i + 1], V[(j + 1) * m + i + 1], V[(j + 1) * m + i]];
    list.push({ d: (q[0][2] + q[1][2] + q[2][2] + q[3][2]) / 4, q, col });
  }
  if (!o.cache) items.forEach(it => { const p = P_(it.x, it.y, it.z ?? zAt(it.x, it.y)); list.push({ d: p[2] + (it.lift ?? 0.02), it, p }); });
  list.sort((A_, B_) => A_.d - B_.d);
  const wc = `rgba(10,8,7,${wire})`; let ga = -1;
  for (const e of list) {
    if (e.it) { e.it.draw(e.p[0], e.p[1], e.p[2]); ga = -1; continue; }                 // items set their own style; ours is restored below
    if (ga < 0) { setA(a); ctx.lineJoin = 'round'; ctx.lineWidth = wire > 0 ? 0.8 : 0.6; ctx.setLineDash([]); ga = 1; }
    const q = e.q; ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]); ctx.lineTo(q[1][0], q[1][1]); ctx.lineTo(q[2][0], q[2][1]); ctx.lineTo(q[3][0], q[3][1]); ctx.closePath();
    ctx.fillStyle = e.col; ctx.fill(); ctx.strokeStyle = wire > 0 ? wc : e.col; ctx.stroke();
  }
  setA(1);
  if (o.cache) {                                    // keep the finished surface (its bounding box, as drawn) for the next frames
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const v of V) { x0 = Math.min(x0, v[0]); y0 = Math.min(y0, v[1]); x1 = Math.max(x1, v[0]); y1 = Math.max(y1, v[1]); }
    x0 = Math.max(0, Math.floor(x0) - 2); y0 = Math.max(0, Math.floor(y0) - 2); x1 = Math.min(cv.width, Math.ceil(x1) + 2); y1 = Math.min(cv.height, Math.ceil(y1) + 2);
    const img = document.createElement('canvas'); img.width = x1 - x0; img.height = y1 - y0; img.getContext('2d').drawImage(cv, x0, y0, img.width, img.height, 0, 0, img.width, img.height);
    for (const k in SURF_CACHE) delete SURF_CACHE[k]; SURF_CACHE[o.cache] = { img, x: x0, y: y0 };
    items.forEach(it => { const p = P_(it.x, it.y, it.z ?? zAt(it.x, it.y)); it.draw(p[0], p[1], p[2]); });
  }
  return cam;
}
const SURF_CACHE = {};
