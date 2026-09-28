// ─────────────────────────────────────────────────────────────
//  The trained networks (weights exported by code/export_ep03_assets.py)
// ─────────────────────────────────────────────────────────────
const A = window.EP3;
function b64bytes(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
function f16(o) {
  const u = b64bytes(o.f16), dv = new DataView(u.buffer), n = u.length / 2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) { const h = dv.getUint16(i * 2, true), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? s * 65504 : s * Math.pow(2, e - 15) * (1 + f / 1024); }
  return { a: out, shape: o.shape };
}
function netOf(o) { const n = {}; for (const k in o) n[k] = f16(o[k]); return n; }
function dense(x, W, b, act) {
  const nin = W.shape[0], nout = W.shape[1], y = new Float32Array(nout);
  for (let j = 0; j < nout; j++) y[j] = b.a[j];
  for (let i = 0; i < nin; i++) { const xi = x[i]; if (xi === 0) continue; const row = i * nout; for (let j = 0; j < nout; j++) y[j] += xi * W.a[row + j]; }
  if (act === 'relu') for (let j = 0; j < nout; j++) y[j] = y[j] > 0 ? y[j] : 0;
  if (act === 'sig') for (let j = 0; j < nout; j++) y[j] = 1 / (1 + Math.exp(-y[j]));
  return y;
}
let AE = null, VAE = null;
const decodeZ = (net, z1, z2) => dense(dense([z1, z2], net.W3, net.b3, 'relu'), net.W4, net.b4, 'sig');
function encodeX(net, x) { const h = dense(x, net.W1, net.b1, 'relu'); if (net.Wz) return [dense(h, net.Wz, net.bz), null]; return [dense(h, net.Wmu, net.bmu), dense(h, net.Wlv, net.blv)]; }
function u8imgs(s) { const u = b64bytes(s), out = []; for (let i = 0; i < u.length / 784; i++) out.push(Float32Array.from(u.subarray(i * 784, (i + 1) * 784), v => v / 255)); return out; }
const TEST = u8imgs(A.test), RECON_AE = u8imgs(A.reconAE), RECON_VAE = u8imgs(A.reconVAE), TRAIN = u8imgs(A.train), ONES = u8imgs(A.ones);
const LAB = A.labels, ZAE = A.zAE, ZVAE = A.zVAE, TL = A.testLabels;

// image rendering: 28×28 → offscreen canvas, cached
const IMC = new Map();
function imgCanvas(vals, col = P.terracotta) {
  const c = document.createElement('canvas'); c.width = c.height = 28; const g = c.getContext('2d'); const d = g.createImageData(28, 28); const [r, gg, b] = hx(col);
  for (let i = 0; i < 784; i++) { d.data[i * 4] = r; d.data[i * 4 + 1] = gg; d.data[i * 4 + 2] = b; d.data[i * 4 + 3] = Math.round(clamp(vals[i]) * 255); }
  g.putImageData(d, 0, 0); return c;
}
function cached(key, make) { let c = IMC.get(key); if (!c) { if (IMC.size > 4000) IMC.clear(); c = make(); IMC.set(key, c); } return c; }
const decImg = (which, z1, z2) => cached(`${which}:${z1.toFixed(3)},${z2.toFixed(3)}`, () => imgCanvas(decodeZ(which === 'ae' ? AE : VAE, z1, z2)));
const rawImg = (set, i, arr) => cached(`${set}:${i}`, () => imgCanvas(arr[i]));
function drawImg(c, x, y, s, a = 1, o = {}) {
  const { bg = true, frame = false, col = P.border } = o; if (a <= 0) return;
  if (bg) { setA(a); ctx.fillStyle = P.elevated; ctx.fillRect(x, y, s, s); }
  setA(a); ctx.imageSmoothingEnabled = s < 100; ctx.drawImage(c, x, y, s, s); ctx.imageSmoothingEnabled = true;
  if (frame) { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.strokeRect(x, y, s, s); }
  setA(1);
}

// latent-space scatter plots: numerals at each code, pre-rendered once
function bounds(Z) { const xs = Z.map(p => p[0]).sort((a, b) => a - b), ys = Z.map(p => p[1]).sort((a, b) => a - b); const q = (arr, f) => arr[Math.floor(f * (arr.length - 1))];
  return [q(xs, 0.005), q(xs, 0.995), q(ys, 0.005), q(ys, 0.995)]; }
let BAE, BVAE = [-3.2, 3.2, -3.2, 3.2];
function mapper(Bd, x, y, w, h) { const sx = w / (Bd[1] - Bd[0]), sy = h / (Bd[3] - Bd[2]), s = Math.min(sx, sy);
  const cx = x + w / 2, cy = y + h / 2, mx = (Bd[0] + Bd[1]) / 2, my = (Bd[2] + Bd[3]) / 2;
  return { X: z => cx + (z - mx) * s, Y: z => cy - (z - my) * s, s, inv: (px, py) => [(px - cx) / s + mx, -(py - cy) / s + my] }; }
const SCAT = new Map();
function scatter(which, x, y, w, h, a = 1, n = 2500) {
  const key = `${which}:${w}x${h}:${n}`;
  const c = cached('scat' + key, () => { const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d');
    const m = mapper(which === 'ae' ? BAE : BVAE, 0, 0, w, h), Z = which === 'ae' ? ZAE : ZVAE;
    g.font = `400 13px ${F.mono}`; g.textAlign = 'center'; g.fillStyle = P.rose; g.globalAlpha = 0.72;
    for (let i = 0; i < n; i++) g.fillText(String(LAB[i]), m.X(Z[i][0]), m.Y(Z[i][1]) + 4); return c; });
  setA(a); ctx.drawImage(c, x, y); setA(1);
  return mapper(which === 'ae' ? BAE : BVAE, x, y, w, h);
}
function centroids(Z) { const s = Array.from({ length: 10 }, () => [0, 0, 0]); Z.forEach((p, i) => { const c = s[LAB[i]]; c[0] += p[0]; c[1] += p[1]; c[2]++; }); return s.map(c => [c[0] / c[2], c[1] / c[2]]); }
const erfinv = p => { // Φ⁻¹ via rational approximation (Acklam)
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239], b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155211877], c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783], d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  const q = p - 0.5, r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
};
function gaussFn(r) { return () => { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v); }; }
const LAST_VAE = A.histVAE[A.histVAE.length - 1], LAST_AE = A.histAE[A.histAE.length - 1];
function hourglass(x0, y, w, h, a = 1) { // encoder (left) + decoder (right) funnels around a bottleneck
  const mid = x0 + w / 2, neck = 34;
  setA(a * 0.14); ctx.fillStyle = P.plum; ctx.beginPath(); ctx.moveTo(x0, y - h / 2); ctx.lineTo(mid - 60, y - neck); ctx.lineTo(mid - 60, y + neck); ctx.lineTo(x0, y + h / 2); ctx.fill();
  ctx.beginPath(); ctx.moveTo(x0 + w, y - h / 2); ctx.lineTo(mid + 60, y - neck); ctx.lineTo(mid + 60, y + neck); ctx.lineTo(x0 + w, y + h / 2); ctx.fill();
  setA(a * 0.7); ctx.strokeStyle = P.plum; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x0, y - h / 2); ctx.lineTo(mid - 60, y - neck); ctx.lineTo(mid - 60, y + neck); ctx.lineTo(x0, y + h / 2); ctx.closePath(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x0 + w, y - h / 2); ctx.lineTo(mid + 60, y - neck); ctx.lineTo(mid + 60, y + neck); ctx.lineTo(x0 + w, y + h / 2); ctx.closePath(); ctx.stroke(); setA(1);
}
function stamp(s, x, y, col, a, rot = -0.12) { if (a <= 0) return; ctx.save(); ctx.translate(x, y); ctx.rotate(rot); const sc = lerp(1.6, 1, eout(a)); ctx.scale(sc, sc);
  ctx.font = `400 34px ${F.mono}`; const w = ctx.measureText(s).width + 36; setA(a); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.strokeRect(-w / 2, -30, w, 52); ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.fillText(s, 0, 8); ctx.restore(); setA(1); }

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open
SC.push({ dur: 24, chapter: 'intro',
  caps: [[0.6, 5, 'Every digit you are watching is drawn by the same machine — from just two numbers.'],
         [5, 10.5, 'Move the two numbers, and the drawing melts smoothly from one digit into another.'],
         [10.5, 15.5, 'Last episode ended on a question: can a machine learn by compressing?'],
         [15.5, 20, 'The answer is yes — and compression turns out to be a deep way to understand data.'],
         [20, 23.6, "Let's squeeze some images."]],
  draw(t) {
    const dim = lerp(1, 0.2, ease(prog(t, 15, 1.5))), a0 = eout(prog(t, 0.3, 1));
    const z1 = 1.7 * Math.sin(0.33 * t + 0.4), z2 = 1.7 * Math.sin(0.47 * t + 1.3);
    GA = dim;
    drawImg(decImg('vae', z1, z2), 560, 250, 480, a0, { frame: true });
    const m = scatter('vae', 1180, 330, 420, 420, a0 * 0.8, 1500);
    ring(m.X(z1), m.Y(z2), 14, P.chalk, a0, 2.5); glow(m.X(z1), m.Y(z2), 6, P.rose, a0);
    text(`z = (${z1.toFixed(2)}, ${z2.toFixed(2)})`, 1390, 800, { font: F.mono, size: 26, color: P.rose, align: 'center', a: a0 });
    text('two numbers', 1390, 300, { font: F.serif, italic: true, size: 26, color: P.dust, align: 'center', a: a0 });
    arrow(1170, 500, 1060, 500, P.dust, 2, a0);
    GA = 1;
    const a = eout(prog(t, 15.6, 1.4));
    text('EPISODE 03', 960, 430, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('Autoencoders & VAEs', 960, 540 + (1 - a) * 16, { font: F.serif, size: 118, color: P.chalk, align: 'center', a, ls: -1 });
    text('squeezing the world into a few numbers', 960, 612, { size: 34, color: P.stone, align: 'center', a: eout(prog(t, 16.4, 1.2)) });
    const tg = [[P.terracotta, 'x  images'], [P.rose, 'z  codes'], [P.amber, 'μ, σ, β  knobs'], [P.ink, 'q, p  distributions'], [P.plum, 'encoder / decoder']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let cx = 960 - tw / 2; tg.forEach(([c, s], i) => { cx += tag(cx, 670, c, s, eout(prog(t, 18.5 + i * 0.35, 0.6)), 21) + 14; });
  } });

// 01 — the compression game
SC.push({ dur: 36, chapter: 'compression', title: 'The compression game', sub: 'Describe this picture using as few numbers as you can.',
  caps: [[1, 7, 'Imagine describing this image over the phone. Reading out all 784 pixel values would take forever.'],
         [7, 13, "You'd just say: 'a seven, leaning a little to the right.' A few words carry almost everything."],
         [13, 20, "That works because real images aren't random. Remember episode 1: random pixels are static."],
         [20, 28, 'Real digits live on a thin sliver of pixel space, where only a few things change: which digit, how slanted, how thick.'],
         [28, 35.6, 'So a handful of numbers should be enough — if a machine can learn which numbers to keep.']],
  draw(t) {
    const a0 = eout(prog(t, 0.5, 0.8));
    drawImg(rawImg('test', 0, TEST), 150, 270, 420, a0, { frame: true });
    text('784 numbers', 360, 740, { font: F.mono, size: 22, color: P.dust, align: 'center', a: a0 });
    // the pixel values, read out
    const ra = a0 * (1 - ease(prog(t, 12.5, 0.8)));
    if (ra > 0) { const off = Math.floor(Math.max(0, t - 1) * 60);
      for (let r = 0; r < 14; r++) for (let c = 0; c < 10; c++) { const i = (off + r * 10 + c) % 784; const v = TEST[0][i];
        text(v.toFixed(2), 700 + c * 108, 300 + r * 36, { font: F.mono, size: 20, color: v > 0.05 ? P.terracotta : P.dust, a: ra * (v > 0.05 ? 1 : 0.5) }); } }
    const ba = eout(prog(t, 7, 0.8)) * (1 - ease(prog(t, 12.5, 0.8)));
    if (ba > 0) { rbox(900, 380, 780, 150, 26, rgba(P.canvas, 0.94), P.sage, ba, 2);
      text('“a seven, leaning right”', 1290, 470, { font: F.serif, italic: true, size: 48, color: P.chalk, align: 'center', a: ba }); }
    // callback: static vs digit
    const ca = eout(prog(t, 13, 0.8)) * (1 - ease(prog(t, 27.5, 0.8)));
    if (ca > 0) {
      const r = rng(Math.floor(t * 4) * 97 + 5), noise = new Float32Array(784); for (let i = 0; i < 784; i++) noise[i] = r();
      drawImg(imgCanvas(noise, P.rose), 760, 290, 300, ca, { frame: true }); text('random pixels', 910, 630, { font: F.mono, size: 20, color: P.rose, align: 'center', a: ca });
      const b = eout(prog(t, 20, 0.8));
      [['which digit', P.rose], ['slant', P.rose], ['thickness', P.rose]].forEach(([s, c], i) => { tag(1170, 320 + i * 80, c, s, b * ca, 24); });
      text('only a few directions of change', 1170, 600, { font: F.serif, italic: true, size: 30, color: P.sage, a: b * ca });
    }
    const da = eout(prog(t, 28, 1));
    if (da > 0) { M('[d|784] numbers  →  [n|2] numbers  ?', 1260, 470, 56, { align: 'center', a: da });
      text('392× compression', 1260, 560, { font: F.serif, italic: true, size: 36, color: P.sage, align: 'center', a: eout(prog(t, 30, 1)) }); }
  } });

// 02 — the autoencoder
const SHOW = [0, 1, 2, 3, 4, 7, 11, 18];
SC.push({ dur: 50, chapter: 'autoencoder', title: 'The autoencoder', sub: 'Squeeze through a keyhole, then rebuild.',
  caps: [[1, 7, 'Here is the machine. An encoder squeezes 784 pixels down to just 2 numbers: the code, z.'],
         [7, 13, 'A decoder then tries to rebuild the whole image from those 2 numbers alone.'],
         [13, 20, 'Training: compare the rebuild with the original, and nudge every weight to shrink the difference.'],
         [20, 27, 'Notice — no labels. Nobody says which digit is which. The image is its own teacher.'],
         [27, 35, 'The only way to pass an exam through a 2-number keyhole is to learn what matters most.'],
         [35, 42, 'After training, the rebuilds look like this: blurry, mostly right — and some honest mistakes, like a 4 that comes back as a 9.'],
         [42, 49.6, "Those 2 numbers are each image's coordinates in a brand-new space: the latent space."]],
  draw(t) {
    const cyc = Math.floor(Math.max(0, t - 2) / 3.5) % SHOW.length, idx = SHOW[cyc];
    const up = ease(prog(t, 34.5, 1)), y = lerp(560, 430, up), sc = lerp(1, 0.72, up);
    const a0 = eout(prog(t, 0.5, 0.8));
    const imgS = 230 * sc;
    drawImg(rawImg('test', idx, TEST), 160, y - imgS / 2, imgS, a0, { frame: true });
    text('x', 160 + imgS / 2, y + imgS / 2 + 36, { font: F.mono, size: 26, color: P.terracotta, align: 'center', a: a0 });
    hourglass(460, y, 1000, 330 * sc, eout(prog(t, 1, 1)));
    text('encoder', 640, y - 170 * sc - 16, { font: F.serif, italic: true, size: 28, color: P.plum, align: 'center', a: a0 });
    text('decoder', 1280, y - 170 * sc - 16, { font: F.serif, italic: true, size: 28, color: P.plum, align: 'center', a: eout(prog(t, 7, 0.8)) });
    text('784 → 256 → 2', 640, y + 170 * sc + 40, { font: F.mono, size: 18, color: P.dust, align: 'center', a: a0 });
    text('2 → 256 → 784', 1280, y + 170 * sc + 40, { font: F.mono, size: 18, color: P.dust, align: 'center', a: eout(prog(t, 7, 0.8)) });
    const z = ZAE[idx];
    rbox(890, y - 50, 140, 100, 14, rgba(P.rose, 0.1), P.rose, a0, 2);
    text(z[0].toFixed(2), 960, y - 10, { font: F.mono, size: 26, color: P.rose, align: 'center', a: a0 }); text(z[1].toFixed(2), 960, y + 30, { font: F.mono, size: 26, color: P.rose, align: 'center', a: a0 });
    text('z', 960, y - 66, { font: F.mono, size: 26, color: P.rose, align: 'center', a: a0 });
    const da = eout(prog(t, 7.5, 0.8));
    drawImg(rawImg('rae', idx, RECON_AE), 1530, y - imgS / 2, imgS, da, { frame: true });
    text('x̂', 1530 + imgS / 2, y + imgS / 2 + 36, { font: F.mono, size: 26, color: P.terracotta, align: 'center', a: da });
    // training curve
    const ca = eout(prog(t, 13, 0.8)) * (1 - ease(prog(t, 34, 0.8)));
    if (ca > 0) {
      card(160, 780, 560, 150, P.plum, ca); const H = A.histAE, mx = H[0].recon, mn = H[H.length - 1].recon;
      text('reconstruction loss per epoch (real run)', 184, 814, { font: F.mono, size: 15, color: P.dust, a: ca });
      setA(ca); ctx.strokeStyle = P.plum; ctx.lineWidth = 3; ctx.beginPath(); const n = Math.max(2, Math.floor(H.length * ease(prog(t, 13.5, 5))));
      H.slice(0, n).forEach((h, i) => { const X = 190 + i / (H.length - 1) * 500, Y = 905 - (h.recon - mn) / (mx - mn) * 70; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
      text(`${H[0].recon.toFixed(0)} → ${H[n - 1].recon.toFixed(0)} nats`, 700, 814, { font: F.mono, size: 15, color: P.plum, align: 'right', a: ca });
    }
    const ma = eout(prog(t, 13, 0.8)) * (1 - ease(prog(t, 34, 0.8)));
    card(780, 780, 980, 150, P.plum, ma);
    M('[n|z] = [m|f][a_|φ]([d|x])     [d|x̂] = [m|g][a_|θ]([n|z])', 816, 836, 28, { a: ma });
    M('[m|L] = [kB|Σ][k_|pixels] BCE([d|x], [d|x̂])   ≈  how different?', 816, 896, 24, { a: ma });
    if (t > 20 && t < 34) text('no labels needed', 960, 300, { font: F.serif, italic: true, size: 34, color: P.sage, align: 'center', a: eout(prog(t, 20.5, 0.8)) * (1 - ease(prog(t, 33, 1))) });
    // gallery of reconstructions
    const ga = eout(prog(t, 35, 1));
    if (ga > 0) SHOW.forEach((k, i) => { const x = 170 + i * 200, ai = eout(prog(t, 35.5 + i * 0.12, 0.5));
      drawImg(rawImg('test', k, TEST), x, 650, 84, ai * ga, { frame: true }); drawImg(rawImg('rae', k, RECON_AE), x + 90, 650, 84, ai * ga, { frame: true }); });
    if (ga > 0) text('original  →  rebuilt from 2 numbers', 960, 790, { font: F.mono, size: 18, color: P.dust, align: 'center', a: ga });
  } });

// 03 — latent space tour
SC.push({ dur: 46, chapter: 'latent space', title: 'Inside the latent space', sub: 'Every image, placed at its 2-number code.',
  caps: [[1, 8, 'Let us look inside. Each of these 2,500 test digits sits at its 2-number code; the symbol shows which digit it really is.'],
         [8, 15, 'Similar digits cluster together. Nobody gave it labels — it discovered the categories on its own.'],
         [15, 23, 'Now walk through the space. At every point, the decoder draws what that code means.'],
         [23, 31, 'Along the way the drawings morph: one digit slides into its neighbour.'],
         [31, 38, 'So the latent space is a map of meaning: nearby points, similar images.'],
         [38, 45.6, 'This is the dream — a map where every location is a drawable image. So… can we just pick a spot and generate?']],
  draw(t) {
    const m = scatter('ae', 120, 240, 900, 700, eout(prog(t, 0.5, 1.5)));
    const C = centroids(ZAE), order = [1, 7, 9, 4, 6, 0, 2, 3, 5, 8];
    if (t > 15) {
      const s = (t - 15) / 2.8, k = Math.min(order.length - 2, Math.floor(s)), u = ease(clamp(s - k));
      const p0 = C[order[k]], p1 = C[order[k + 1]], z1 = lerp(p0[0], p1[0], u), z2 = lerp(p0[1], p1[1], u);
      const pa = eout(prog(t, 15, 0.6));
      ring(m.X(z1), m.Y(z2), 16, P.chalk, pa, 2.5); glow(m.X(z1), m.Y(z2), 6, P.rose, pa);
      drawImg(decImg('ae', z1, z2), 1180, 300, 420, pa, { frame: true });
      M(`[m|decode]([n|${z1.toFixed(1)}], [n|${z2.toFixed(1)}])`, 1390, 780, 28, { align: 'center', a: pa });
      line(m.X(z1) + 16, m.Y(z2), 1170, 510, P.dust, 1, pa * 0.5, [4, 6]);
    }
    if (t > 8 && t < 15) { const ca = eout(prog(t, 8, 0.8)) * (1 - ease(prog(t, 14.2, 0.8))); [1, 0, 7].forEach((d, i) => { const c = C[d]; pill(m.X(c[0]), m.Y(c[1]) - 30, `the ${d}s`, P.chalk, ca, { align: 'center' }); }); }
    insight(1180, 820 - 20, 620, 'nearby codes  →  similar images', eout(prog(t, 31, 0.8)) * (1 - ease(prog(t, 38, 0.6))), 'map of meaning');
  } });

// 04 — the holes
const AE_RAND = (() => { const r = rng(77); return Array.from({ length: 8 }, () => [r(), r()]); })();
SC.push({ dur: 44, chapter: 'the holes', title: 'The trouble: holes in the map', sub: 'Pick random spots and decode them.',
  caps: [[1, 8, "Let's try. Pick random points on the map and decode them."],
         [8, 15, 'Some look fine. Others are smudges or odd hybrids — like B and D — landing in gaps between the clusters.'],
         [15, 23, 'The decoder was only trained on the exact points the encoder produced. Everywhere else, it is guessing.'],
         [23, 31, 'And the map has no natural shape: clusters sprawl in odd directions at arbitrary scales. Where would you even sample from?'],
         [31, 38, 'The autoencoder learned to compress — not to generate. Nobody asked it to keep the space tidy.'],
         [38, 43.6, 'Two fixes would make it generative: fill the gaps, and give the map a known shape.']],
  draw(t) {
    const m = scatter('ae', 120, 240, 900, 700, 1);
    AE_RAND.forEach(([u, v], i) => {
      const z1 = lerp(BAE[0], BAE[1], u), z2 = lerp(BAE[2], BAE[3], v), a = eout(prog(t, 1.5 + i * 0.7, 0.5));
      if (a <= 0) return;
      glow(m.X(z1), m.Y(z2), 7, P.chalk, a); text(String.fromCharCode(65 + i), m.X(z1) + 12, m.Y(z2) - 10, { font: F.mono, size: 20, color: P.chalk, a });
      const x = 1110 + (i % 4) * 180, y = 280 + Math.floor(i / 4) * 230;
      drawImg(decImg('ae', z1, z2), x, y, 160, a, { frame: true }); text(String.fromCharCode(65 + i), x + 8, y + 26, { font: F.mono, size: 20, color: P.chalk, a });
    });
    const ia = eout(prog(t, 23, 0.8));
    if (ia > 0) { card(1110, 760, 700, 150, P.rose, ia); label('where to sample?', 1140, 804, ia, P.rose);
      M(`[n|z₁] ∈ [${BAE[0].toFixed(0)}, ${BAE[1].toFixed(0)}],  [n|z₂] ∈ [${BAE[2].toFixed(0)}, ${BAE[3].toFixed(0)}],  shape: ???`, 1140, 866, 24, { a: ia }); }
  } });

// 05 — fuzzy codes on a leash
const FZ = [[-1.2, 0.9], [1.4, 1.1], [0.2, -1.3], [-1.5, -0.8], [1.3, -0.6], [0.1, 0.4]];
const FZ_EPS = (() => { const r = rng(9), g = gaussFn(r); return FZ.map(() => Array.from({ length: 14 }, () => [g(), g()])); })();
SC.push({ dur: 48, chapter: 'fuzzy codes', title: 'Fix: fuzzy codes on a leash', sub: 'Encode to a cloud, not a point — and keep every cloud near home.',
  caps: [[1, 8, 'Fix one: instead of encoding each image to a single point, encode it to a small cloud of points.'],
         [8, 16, 'The decoder receives a random point from the cloud and must still rebuild the image — so a whole neighbourhood has to mean that image.'],
         [16, 24, 'Left alone, the network cheats: it shrinks every cloud to a dot and spreads them far apart. Back to square one.'],
         [24, 32, 'Fix two: put every cloud on a leash. Pull the centres toward the origin, and keep the clouds fuzzy — like a standard bell curve.'],
         [32, 40, 'Now the clouds overlap and tile the space: no gaps, and a known shape — N(0, I).'],
         [40, 47.6, "That's the Variational Autoencoder: fuzzy codes, plus a leash."]],
  draw(t) {
    const cx = 700, cy = 590, S = 150;
    let sig = 0.02, spread = 1;
    sig = lerp(sig, 0.42, ease(prog(t, 2, 6)));
    sig = lerp(sig, 0.04, ease(prog(t, 16.5, 3))); spread = lerp(spread, 1.9, ease(prog(t, 17, 5)));
    sig = lerp(sig, 0.45, ease(prog(t, 25, 6))); spread = lerp(spread, 0.62, ease(prog(t, 25, 6)));
    const ra = eout(prog(t, 24, 1.2));
    line(cx - 520, cy, cx + 520, cy, P.border, 1.5); line(cx, cy - 360, cx, cy + 360, P.border, 1.5);
    [1, 2].forEach(k => { setA(ra * 0.7); ctx.strokeStyle = P.ink; ctx.lineWidth = 2; ctx.setLineDash([8, 8]); ctx.beginPath(); ctx.arc(cx, cy, k * S, 0, TAU); ctx.stroke(); ctx.setLineDash([]); setA(1); });
    if (ra > 0) text('N(0, I)', cx + 2 * S * 0.72 + 10, cy - 2 * S * 0.72 - 10, { font: F.mono, size: 22, color: P.ink, a: ra });
    const jig = Math.floor(t * 3);
    FZ.forEach(([x, y], i) => {
      const px = cx + x * spread * S, py = cy - y * spread * S, a = eout(prog(t, 0.5 + i * 0.25, 0.6));
      if (t > 24 && t < 36) line(cx, cy, px, py, P.plum, 2, a * eout(prog(t, 24.5, 0.8)) * (1 - ease(prog(t, 34, 1))) * 0.8);
      setA(a * 0.14); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.arc(px, py, Math.max(2, sig * S * 2), 0, TAU); ctx.fill(); setA(1);
      ring(px, py, Math.max(2, sig * S * 2), P.ink, a * 0.6, 1.5);
      const r = rng(jig * 31 + i), g = gaussFn(r);
      if (sig > 0.08) for (let k = 0; k < 10; k++) dot(px + g() * sig * S, py + g() * sig * S, 3.2, P.rose, a * 0.9);
      glow(px, py, 5, P.amber, a);
      drawImg(rawImg('test', [0, 1, 2, 3, 4, 7][i], TEST), px + 14, py - 70, 56, a * 0.95, { frame: true });
    });
    if (t > 16.5 && t < 24.5) pill(cx, 270, 'cheating: σ → 0, codes fly apart', P.rose, eout(prog(t, 17, 0.6)) * (1 - ease(prog(t, 24, 0.5))), { align: 'center', size: 22 });
    if (t > 25 && t < 36) pill(cx, 270, 'the leash pulls every cloud home', P.plum, eout(prog(t, 25.5, 0.6)) * (1 - ease(prog(t, 35, 0.8))), { align: 'center', size: 22 });
    // math
    const c1 = eout(prog(t, 3, 0.8));
    card(1300, 260, 530, 230, P.ink, c1); label('fix one · fuzzy codes', 1334, 306, c1, P.ink);
    M('[p|q][a_|φ]([n|z]|[d|x]) = N([a|μ]([d|x]), [a|σ]²([d|x]))', 1334, 380, 24, { a: c1 });
    text('encoder outputs a centre μ and a width σ', 1334, 440, { size: 22, color: P.stone, a: c1 });
    const c2 = eout(prog(t, 25, 0.8));
    card(1300, 520, 530, 230, P.plum, c2); label('fix two · the leash', 1334, 566, c2, P.plum);
    M('KL( [p|q]([n|z]|[d|x]) ‖ N(0, I) )', 1334, 640, 26, { a: c2 });
    text('penalty for straying from the bell curve', 1334, 700, { size: 22, color: P.stone, a: c2 });
    insight(1300, 780, 530, 'VAE = fuzzy codes + a leash', eout(prog(t, 40, 0.8)), 'in one line');
  } });

// 06 — the math
SC.push({ dur: 58, chapter: 'the math', title: 'The math of the VAE', sub: 'Two terms, two jobs — and a floor under log p(x).',
  caps: [[1, 8, 'Here is the whole VAE objective in one line. Two terms, each with a job.'],
         [8, 15, "Term one, reconstruction: sample z from the image's cloud, decode it, and measure how badly we rebuilt x."],
         [15, 23, 'Term two, the leash: the KL divergence — how far the cloud N(μ, σ²) is from the standard bell curve.'],
         [23, 31, 'For Gaussians it has a closed form. μ² pulls the centre home; σ² − log σ² − 1 is smallest exactly at σ = 1.'],
         [31, 38, 'Why these two terms? Episode 1 says: maximise log p(x). But here p(x) = ∫ p(x | z) p(z) dz — any code might have drawn x.'],
         [38, 46, 'That integral is intractable. So the VAE maximises a floor beneath it: the Evidence Lower BOund, or ELBO.'],
         [46, 57.6, `Push the floor up and log p(x) must rise too. Our trained VAE: about ${LAST_VAE.test_recon.toFixed(0)} nats of reconstruction plus ${LAST_VAE.test_kl.toFixed(1)} nats of leash, per image.`]],
  draw(t) {
    const up = ease(prog(t, 30.5, 1)), fy = lerp(380, 300, up), fs = lerp(40, 32, up);
    const a0 = eout(prog(t, 0.6, 0.8));
    ctx.font = `400 ${fs}px ${F.mono}`;
    const p1 = '[m|L] =', p2 = '[p|E][p_|z∼q(z|x)][ −log [p|p][a_|θ]([d|x]|[n|z]) ]', p3 = '+  KL( [p|q][a_|φ]([n|z]|[d|x]) ‖ N(0, I) )';
    const w1 = M(p1, -9999, 0, fs), w2 = M(p2, -9999, 0, fs), w3 = M(p3, -9999, 0, fs); const gap = 20, total = w1 + w2 + w3 + 2 * gap; let x = 960 - total / 2;
    M(p1, x, fy, fs, { a: a0 }); const x2 = x + w1 + gap; M(p2, x2, fy, fs, { a: a0 }); const x3 = x2 + w2 + gap; M(p3, x3, fy, fs, { a: a0 });
    const b1 = eout(prog(t, 8, 0.8)) * (1 - up), b2 = eout(prog(t, 15, 0.8)) * (1 - up);
    const brace = (xa, xb, y, s, col, a) => { if (a <= 0) return; setA(a); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xa, y); ctx.lineTo(xa, y + 14); ctx.lineTo(xb, y + 14); ctx.lineTo(xb, y); ctx.stroke(); setA(1); text(s, (xa + xb) / 2, y + 52, { font: F.serif, italic: true, size: 30, color: col, align: 'center', a }); };
    brace(x2, x2 + w2, fy + 24, 'reconstruction: rebuild x from a sampled code', P.terracotta, b1);
    brace(x3 + 30, x3 + w3, fy + 24, 'the leash', P.plum, b2);
    // closed form + curves
    const ca = eout(prog(t, 23, 0.8)) * (1 - ease(prog(t, 30.5, 0.8)));
    if (ca > 0) {
      M('KL = ½ [kB|Σ][k_|j] ( [a|μ]²  +  [a|σ]² − log [a|σ]² − 1 )', 960, 580, 36, { align: 'center', a: ca });
      const plot = (x0, fn, lo, hi, xs, label2, pos) => { const W = 520, Hh = 190, y0 = 890; line(x0, y0, x0 + W, y0, P.borderLight, 1.5, ca);
        setA(ca); ctx.strokeStyle = P.amber; ctx.lineWidth = 3.5; ctx.beginPath(); for (let i = 0; i <= 100; i++) { const v = lo + (hi - lo) * i / 100, X = x0 + i / 100 * W, Y = y0 - clamp(fn(v) / 4) * Hh; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); } ctx.stroke(); setA(1);
        const v = pos, X = x0 + (v - lo) / (hi - lo) * W, Y = y0 - clamp(fn(v) / 4) * Hh; glow(X, Y, 8, P.plum, ca); text(label2, x0, y0 + 34, { font: F.mono, size: 18, color: P.amber, a: ca }); };
      const u = ease(prog(t, 24.5, 4));
      plot(330, v => v * v, -2, 2, 0, 'μ²  → minimum at μ = 0', lerp(1.8, 0, u));
      plot(1070, v => v * v - Math.log(v * v) - 1, 0.15, 3, 0, 'σ² − log σ² − 1  → minimum at σ = 1', lerp(2.7, 1, u));
    }
    // ELBO floor
    const ea = eout(prog(t, 31, 1));
    if (ea > 0) {
      M('log [p|p][a_|θ]([d|x]) = log [pB|∫] [p|p][a_|θ]([d|x] | [n|z]) [p|p]([n|z]) d[n|z]', 170, 470, 30, { a: ea });
      text('← every code z could have drawn x: intractable', 170, 520, { font: F.mono, size: 18, color: P.dust, a: ea });
      const fa = eout(prog(t, 38, 1));
      M('log [p|p][a_|θ]([d|x])  ≥  −[m|L]  =  ELBO', 170, 610, 34, { a: fa });
      const g = ease(prog(t, 40, 6)), bx = 1300, base = 860;
      const hTrue = lerp(250, 420, g), hElbo = lerp(120, 360, g);
      setA(fa); ctx.fillStyle = rgba(P.ink, 0.7); ctx.fillRect(bx, base - hTrue, 150, hTrue); ctx.fillStyle = rgba(P.plum, 0.8); ctx.fillRect(bx + 200, base - hElbo, 150, hElbo); setA(1);
      text('log p(x)', bx + 75, base + 32, { font: F.mono, size: 20, color: P.ink, align: 'center', a: fa }); text('ELBO', bx + 275, base + 32, { font: F.mono, size: 20, color: P.plum, align: 'center', a: fa });
      line(bx + 160, base - hTrue, bx + 360, base - hTrue, P.dust, 1.5, fa, [5, 5]);
      text('dashed gap = KL(q ‖ true posterior) ≥ 0', bx + 175, base + 66, { font: F.mono, size: 16, color: P.dust, align: 'center', a: fa });
      text('training ↑', bx + 175, base - hTrue - 40, { font: F.serif, italic: true, size: 26, color: P.sage, align: 'center', a: eout(prog(t, 41, 1)) });
      const na = eout(prog(t, 47, 1));
      card(170, 680, 1000, 200, P.sage, na); label('our trained 2-D VAE, on held-out digits', 204, 726, na, P.sage);
      M(`−ELBO = [d|${LAST_VAE.test_recon.toFixed(1)}] (reconstruction) + [m|${LAST_VAE.test_kl.toFixed(1)}] (leash)  nats per image`, 204, 796, 26, { a: na });
      text(`${A.histVAE.length} epochs on 60,000 digits, backprop written by hand in numpy`, 204, 850, { font: F.mono, size: 16, color: P.dust, a: na });
    }
  } });

// 07 — the reparameterisation trick
SC.push({ dur: 38, chapter: 'reparam', title: 'The reparameterisation trick', sub: "You can't differentiate a dice roll — so move the dice.",
  caps: [[1, 8, 'One engineering snag. Training needs gradients to flow back from the loss all the way to the encoder.'],
         [8, 15, "But in the middle sits a random draw — and you can't take the derivative of a dice roll."],
         [15, 23, 'The trick: roll a standard die ε ~ N(0, 1) off to the side, then compute z = μ + σ · ε.'],
         [23, 30, 'Same distribution for z — but now z is a plain function of μ and σ: ∂z/∂μ = 1 and ∂z/∂σ = ε.'],
         [30, 37.6, 'The randomness becomes just another input, the gradients flow, and ordinary backprop trains the whole thing.']],
  draw(t) {
    const y = 520, a0 = eout(prog(t, 0.6, 0.8)), B = ease(prog(t, 15, 2));
    const node = (x, w, label2, col, a, big = false) => { rbox(x - w / 2, y - 45, w, 90, 16, rgba(col, 0.1), rgba(col, 0.75), a, 2); M(label2, x, y + 10, big ? 30 : 26, { align: 'center', a }); };
    node(170, 120, '[d|x]', P.terracotta, a0); arrow(232, y, 290, y, P.dust, 2, a0);
    node(380, 160, 'encoder', P.plum, a0); arrow(462, y, 520, y, P.dust, 2, a0);
    node(610, 160, '[a|μ], [a|σ]', P.amber, a0); arrow(692, y, 760, y, P.dust, 2, a0);
    // sampler node morphs
    if (B < 1) node(890, 240, '[n|z] ∼ N([a|μ], [a|σ]²)', P.rose, a0 * (1 - B));
    if (B > 0) { node(890, 240, '[n|z] = [a|μ] + [a|σ]·[n|ε]', P.plum, B); rbox(840, y + 120, 100, 70, 14, rgba(P.rose, 0.12), P.rose, B, 2); M('[n|ε]', 890, y + 165, 28, { align: 'center', a: B }); text('ε ~ N(0,1)', 890, y + 230, { font: F.mono, size: 18, color: P.rose, align: 'center', a: B }); arrow(890, y + 118, 890, y + 50, P.rose, 2, B); }
    arrow(1012, y, 1070, y, P.dust, 2, a0); node(1160, 160, 'decoder', P.plum, a0); arrow(1242, y, 1300, y, P.dust, 2, a0);
    node(1370, 110, '[d|x̂]', P.terracotta, a0); arrow(1427, y, 1490, y, P.dust, 2, a0); node(1570, 130, '[m|L]', P.plum, a0);
    // backward pass
    const ga = eout(prog(t, 3, 1)), flow = (t * 0.7) % 1;
    const back = (x1, x2, a, col) => { setA(a); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.setLineDash([10, 10]); ctx.lineDashOffset = flow * 20; ctx.beginPath(); ctx.moveTo(x1, y - 80); ctx.lineTo(x2, y - 80); ctx.stroke(); ctx.setLineDash([]); ctx.lineDashOffset = 0; setA(1); };
    back(1570, 1000, ga, P.sage);
    if (B < 1) { back(1000, 780, ga * (1 - B), P.sage); text('✕', 890, y - 66, { font: F.mono, size: 46, color: P.rose, align: 'center', a: ga * (1 - B) * eout(prog(t, 8, 0.6)) });
      text('no gradient through a random draw', 890, y - 130, { font: F.mono, size: 18, color: P.rose, align: 'center', a: ga * (1 - B) * eout(prog(t, 8.5, 0.6)) }); }
    if (B > 0) back(1000, 300, B, P.sage);
    text('gradients ←', 1570, y - 100, { font: F.mono, size: 18, color: P.sage, align: 'center', a: ga });
    const ma = eout(prog(t, 23, 0.8));
    card(460, 800, 1000, 110, P.amber, ma);
    M('[n|z] = [a|μ] + [a|σ]·[n|ε]    ⇒    ∂[n|z]/∂[a|μ] = 1,    ∂[n|z]/∂[a|σ] = [n|ε]', 960, 868, 30, { align: 'center', a: ma });
  } });

// 08 — the VAE's latent space
const VAE_SAMPLES = (() => { const r = rng(123), g = gaussFn(r); return Array.from({ length: 16 }, () => [g(), g()]); })();
const GRID_N = 13, GRID = Array.from({ length: GRID_N }, (_, i) => erfinv(0.04 + 0.92 * i / (GRID_N - 1)));
SC.push({ dur: 54, chapter: 'vae space', title: 'A map you can sample from', sub: 'Same data, same network, same 2 numbers — plus fuzz and a leash.',
  caps: [[1, 8, 'Now train it for real: same data, same network, same 2-number bottleneck — plus the fuzz and the leash.'],
         [8, 15, "Left, the autoencoder's map: sprawling and gappy. Right, the VAE's: one round, tightly packed cloud centred on zero."],
         [15, 24, "Decode a grid across the VAE's space: every location is a digit, and they flow smoothly into each other."],
         [24, 32, 'Generation is now easy: draw z from N(0, I) — the bell curve we asked for — and decode. Brand-new digits.'],
         [32, 40, 'Walk in a straight line from one digit to another and the drawing morphs along the way — no smudgy gaps.'],
         [40, 47, 'The latent space has become a smooth map you can sample from, walk through, and explore.'],
         [47, 53.6, 'Encoder, decoder, and probability tying them together: compression turned into generation.']],
  draw(t) {
    const A1 = 1 - ease(prog(t, 14.5, 1));
    if (A1 > 0) { GA = A1;
      scatter('ae', 120, 250, 780, 660, eout(prog(t, 0.5, 1))); scatter('vae', 1020, 250, 780, 660, eout(prog(t, 1.5, 1)));
      text('autoencoder', 510, 950 - 20, { font: F.serif, italic: true, size: 30, color: P.dust, align: 'center' });
      text('variational autoencoder', 1410, 950 - 20, { font: F.serif, italic: true, size: 30, color: P.sage, align: 'center', a: eout(prog(t, 2, 1)) });
      const m = mapper(BVAE, 1020, 250, 780, 660); [1, 2].forEach(k => { setA(eout(prog(t, 9, 1)) * 0.7); ctx.strokeStyle = P.ink; ctx.lineWidth = 2; ctx.setLineDash([8, 8]); ctx.beginPath(); ctx.arc(m.X(0), m.Y(0), k * m.s, 0, TAU); ctx.stroke(); ctx.setLineDash([]); setA(1); });
      GA = 1; }
    const B1 = eout(prog(t, 15, 1));
    if (B1 > 0) {
      const shrink = ease(prog(t, 23.5, 1)), size = lerp(660, 600, shrink), gx = lerp(630, 150, shrink), gy = 255, cell = size / GRID_N;
      GRID.forEach((vy, i) => GRID.forEach((vx, j) => { const k = i * GRID_N + j, a = eout(prog(t, 15.2 + k * 0.012, 0.4)) * B1; drawImg(decImg('vae', vx, GRID[GRID_N - 1 - i]), gx + j * cell, gy + i * cell, cell - 2, a, { bg: false }); }));
      text('z₁ →', gx + size - 10, gy + size + 30, { font: F.mono, size: 18, color: P.rose, align: 'right', a: B1 }); text('z₂ ↑', gx - 10, gy + 16, { font: F.mono, size: 18, color: P.rose, align: 'right', a: B1 });
      const sa = eout(prog(t, 24, 0.8));
      if (sa > 0) { text('decode(z),  z ~ N(0, I)', 860, 300, { font: F.mono, size: 20, color: P.rose, a: sa });
        VAE_SAMPLES.forEach(([a, b], i) => drawImg(decImg('vae', a, b), 860 + (i % 8) * 118, 320 + Math.floor(i / 8) * 118, 110, eout(prog(t, 24.5 + i * 0.12, 0.4)) * sa, { frame: true })); }
      const ia = eout(prog(t, 32, 0.8));
      if (ia > 0) { const [a0v] = encodeX(VAE, TEST[3]), [b0v] = encodeX(VAE, TEST[18]);
        text('interpolate:  z = (1 − s)·z_A + s·z_B', 860, 620, { font: F.mono, size: 20, color: P.rose, a: ia });
        for (let k = 0; k < 8; k++) { const s = k / 7; drawImg(decImg('vae', lerp(a0v[0], b0v[0], s), lerp(a0v[1], b0v[1], s)), 860 + k * 118, 640, 110, eout(prog(t, 32.5 + k * 0.2, 0.4)) * ia, { frame: k === 0 || k === 7, col: P.terracotta }); } }
      insight(860, 790, 944, 'Every point of N(0, I) decodes to a plausible digit — so sampling is just: roll z, decode.', eout(prog(t, 40, 0.8)));
    }
  } });

// 09 — the catch: blur
const AVG7 = ONES[0].map((v, i) => (v + ONES[2][i]) / 2);
SC.push({ dur: 42, chapter: 'the catch', title: 'The catch: a little blurry', sub: 'Why the VAE hedges its bets.',
  caps: [[1, 8, "Every model has a catch. The VAE's is visible: its images are blurry."],
         [8, 16, "When the decoder isn't sure which of two possible strokes you meant, the cheapest bet under its loss is… the average."],
         [16, 23, 'And the average of two sharp ones — one slanted, one upright — is a fuzzy smear.'],
         [23, 31, 'There is also a tug-of-war, set by a weight β on the leash: more β, a tidier map but blurrier images; less β, sharper but gappier.'],
         [31, 41.6, 'Bigger latents help, and later models remove the blur. But notice: we keep saying "blurry" and "better". How would we even measure that?']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8));
    if (t < 8.5) { [0, 1, 2, 3].forEach((k, i) => { drawImg(rawImg('test', k, TEST), 260 + i * 360, 300, 150, a0, { frame: true }); drawImg(rawImg('rvae', k, RECON_VAE), 420 + i * 360 - 0, 300, 150, a0, { frame: true }); });
      text('original · VAE rebuild', 960, 520, { font: F.mono, size: 20, color: P.dust, align: 'center', a: a0 }); }
    const b = eout(prog(t, 8.5, 0.8));
    if (b > 0) {
      drawImg(rawImg('ones', 0, ONES), 200, 300, 260, b, { frame: true }); text('+', 530, 450, { font: F.serif, size: 70, color: P.dust, align: 'center', a: b });
      drawImg(rawImg('ones', 2, ONES), 600, 300, 260, b, { frame: true }); text('=', 930, 450, { font: F.serif, size: 70, color: P.dust, align: 'center', a: eout(prog(t, 16, 0.6)) });
      drawImg(cached('avg7', () => imgCanvas(AVG7)), 1000, 300, 260, eout(prog(t, 16, 0.8)), { frame: true });
      text('two plausible ones', 530, 610, { font: F.mono, size: 20, color: P.terracotta, align: 'center', a: b }); text('their average', 1130, 610, { font: F.mono, size: 20, color: P.terracotta, align: 'center', a: eout(prog(t, 16, 0.8)) });
      M('argmin[d_|x̂] E‖[d|x] − [d|x̂]‖²  =  E[[d|x]]   ← the mean', 1330, 380, 22, { a: eout(prog(t, 12, 0.8)) });
    }
    const c = eout(prog(t, 23, 0.8));
    card(200, 700, 1520, 190, P.amber, c); label('the β knob', 236, 746, c, P.amber);
    M('[m|L] = reconstruction + [a|β] · KL', 236, 816, 34, { a: c });
    text('β ↑  tidier map, blurrier images          β ↓  sharper images, gappier map', 236, 866, { font: F.mono, size: 20, color: P.stone, a: c });
    text('“blurry”? “better”? — measured how?', 1280, 816, { font: F.serif, italic: true, size: 32, color: P.sage, a: eout(prog(t, 33, 1)) });
  } });

// 10 — build it
const CODE = [
  '[k|# encoder: an image becomes a small cloud]',
  'h = [p|relu]([d|x] @ [a|W1] + [a|b1])',
  '[a|mu], [a|logvar] = h @ [a|Wmu] + [a|bmu], h @ [a|Wlv] + [a|blv]',
  '',
  '[k|# reparameterisation: move the dice aside]',
  '[n|eps] = np.random.[n|randn](*mu.shape)',
  '[n|z] = [a|mu] + np.[p|exp]([a|logvar] / [a|2]) * [n|eps]',
  '',
  '[k|# decoder: rebuild the image]',
  'logits = [p|relu]([n|z] @ [a|W3] + [a|b3]) @ [a|W4] + [a|b4]',
  '',
  '[k|# loss = reconstruction + β · leash]',
  'recon = (np.[p|logaddexp]([a|0], logits) - [d|x] * logits).[p|sum]([a|1])',
  'kl = [a|0.5] * ([a|mu]**[a|2] + np.[p|exp]([a|logvar]) - [a|logvar] - [a|1]).[p|sum]([a|1])',
  'loss = (recon + [a|beta] * kl).[p|mean]()',
  '',
  '[k|# generate: roll z, decode]',
  '[d|x_new] = [p|sigmoid]([p|relu](np.random.[n|randn]([a|1], [a|2]) @ [a|W3] + [a|b3]) @ [a|W4] + [a|b4])',
];
SC.push({ dur: 32, chapter: 'build it', title: 'Build it yourself', sub: 'The heart of a VAE is a dozen lines of numpy.',
  caps: [[1, 9, 'The whole VAE fits in about a dozen lines of numpy. The full code, with hand-written backprop, is linked below.'],
         [9, 18, 'Train it in a few minutes on a laptop. Then play: change β, give it more latent numbers, or feed it your own images.'],
         [18, 31.6, 'Recap: squeeze to a code, make codes fuzzy, leash them to a bell curve — and every point of the map becomes something you can draw.']],
  draw(t) {
    const a = eout(prog(t, 0.6, 0.8));
    card(100, 240, 1060, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 288 + i * 34, 18, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8));
    card(1200, 240, 620, 690, P.sage, ra); label('the whole episode', 1236, 290, ra, P.sage);
    [['autoencoder', '[n|z] = [m|f]([d|x]),  [d|x̂] = [m|g]([n|z])'], ['fuzzy codes', '[p|q]([n|z]|[d|x]) = N([a|μ], [a|σ]²)'], ['the loss', '[m|L] = recon + [a|β]·KL(q ‖ N(0,I))'], ['the trick', '[n|z] = [a|μ] + [a|σ]·[n|ε]'], ['generate', '[n|z] ∼ N(0, I)  →  [m|g]([n|z])']].forEach(([nm, m], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.6, 0.8)); text(nm, 1236, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(m, 1236, 412 + i * 112, 24, { a: ai }); });
  } });

// 11 — the hook: how do you grade a generator?
SC.push({ dur: 50, chapter: 'next', title: 'A puzzle', sub: 'Three models. Sixteen samples each. Which model is best?',
  caps: [[1, 7, 'Here is a puzzle. Three models, sixteen samples each. Which model is best?'],
         [7, 14, 'A is a bit blurry. B is razor sharp — every sample looks perfectly real. C is sharp and confident too.'],
         [14, 21, 'Most people pick B. But B is a photocopier: every "sample" is copied straight out of the training set.'],
         [21, 28, 'C has a different sickness: it only ever draws ones. Sharp, real-looking — and useless.'],
         [28, 36, 'So which number captures "good"? Sharpness crowns the copier. Realism crowns the one-trick pony. Every metric can be fooled.'],
         [36, 42, 'How do you grade a machine whose whole job is to make something new?'],
         [42, 49.6, 'Next episode: evaluation — likelihood, FID, precision and recall, human judges — and exactly how each one can lie to you.']],
  draw(t) {
    const dimAll = 1 - 0.8 * ease(prog(t, 35.5, 1));
    const panels = [['A', i => decImg('vae', VAE_SAMPLES[i][0] * 0.9, VAE_SAMPLES[i][1] * 0.9)], ['B', i => rawImg('train', i, TRAIN)], ['C', i => rawImg('ones', i, ONES)]];
    panels.forEach(([nm, get], p) => {
      const x = 130 + p * 580, y = 250, a = eout(prog(t, 1 + p * 0.5, 0.8)) * dimAll; if (a <= 0) return;
      card(x, y, 520, 530, P.plum, a); text(`model ${nm}`, x + 34, y + 54, { font: F.serif, size: 36, color: P.chalk, a });
      for (let i = 0; i < 16; i++) drawImg(get(i), x + 52 + (i % 4) * 106, y + 84 + Math.floor(i / 4) * 106, 98, a, { bg: true });
    });
    GA = dimAll;
    stamp('COPIED', 130 + 580 + 260, 520, P.rose, eout(prog(t, 15, 0.6)));
    if (t > 15) { const ta = eout(prog(t, 16, 0.8)); text('identical to training images', 130 + 580 + 260, 812, { font: F.mono, size: 18, color: P.rose, align: 'center', a: ta }); }
    stamp('ONLY 1s', 130 + 1160 + 260, 520, P.rose, eout(prog(t, 22, 0.6)));
    GA = 1;
    const sa = eout(prog(t, 28, 0.8)) * dimAll;
    if (sa > 0) {
      rbox(250, 840, 1420, 66, 14, rgba(P.canvas, 0.95), P.border, sa);
      const rows = [['sharp', '✗', '✓', '✓'], ['looks real', '~', '✓', '✓'], ['diverse', '✓', '✓', '✗'], ['new', '✓', '✗', '~']];
      rows.forEach(([nm, ...v], i) => { const x = 280 + i * 350; text(nm, x, 881, { font: F.mono, size: 18, color: P.dust, a: sa }); v.forEach((m, k) => text(`${'ABC'[k]}${m}`, x + 140 + k * 58, 881, { font: F.mono, size: 18, color: m === '✓' ? P.sage : m === '✗' ? P.rose : P.stone, a: sa })); });
    }
    const qa = eout(prog(t, 36, 1.2));
    if (qa > 0) {
      text('How do you grade a machine', 960, 470, { font: F.serif, size: 70, color: P.chalk, align: 'center', a: qa });
      text('whose job is to make something new?', 960, 555, { font: F.serif, size: 70, color: P.chalk, align: 'center', a: qa });
      text('NEXT  ·  EPISODE 04', 960, 660, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 42, 1)), ls: 4 });
      text('Evaluating generative models — and how every metric can lie', 960, 715, { font: F.serif, italic: true, size: 38, color: P.stone, align: 'center', a: eout(prog(t, 42.5, 1)) });
      text('then: GANs', 960, 780, { font: F.mono, size: 18, color: P.dust, align: 'center', a: eout(prog(t, 44, 1)) });
    }
  } });
