// ─────────────────────────────────────────────────────────────
//  Data sets (all seeded — every render is identical)
// ─────────────────────────────────────────────────────────────
function makeMoons(n, seed, noise = 0.09) {
  const r = rng(seed), g = gaussFn(r), pts = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, th = Math.PI * r(); let x, y;
    if (c === 0) { x = Math.cos(th) - 0.5; y = Math.sin(th) - 0.25; } else { x = 0.5 - Math.cos(th); y = 0.25 - Math.sin(th); }
    pts.push({ x: x + noise * g(), y: y + noise * g(), c, r: r() });
  }
  return pts;
}
const MOONS = makeMoons(320, 7);
const BW = 0.14;
function moonPdf(x, y) { let s = 0; for (const p of MOONS) { const dx = x - p.x, dy = y - p.y; s += Math.exp(-(dx * dx + dy * dy) / (2 * BW * BW)); } return s / (MOONS.length * TAU * BW * BW); }
function makeHeat(nx, ny) {
  const c = document.createElement('canvas'); c.width = nx; c.height = ny;
  const g = c.getContext('2d'); const img = g.createImageData(nx, ny); const vals = new Float32Array(nx * ny); let mx = 0;
  for (let j = 0; j < ny; j++) { const y = 1.5 - (j + .5) / ny * 3; for (let i = 0; i < nx; i++) { const x = -2.2 + (i + .5) / nx * 4.4; const v = moonPdf(x, y); vals[j * nx + i] = v; if (v > mx) mx = v; } }
  const col = hx(P.ink);
  for (let k = 0; k < nx * ny; k++) { img.data[k * 4] = col[0]; img.data[k * 4 + 1] = col[1]; img.data[k * 4 + 2] = col[2]; img.data[k * 4 + 3] = Math.round(Math.pow(vals[k] / mx, 0.6) * 235); }
  g.putImageData(img, 0, 0); return c;
}
let HEAT = null;
function drawHeat(pl, a) { if (a <= 0) return; setA(a); ctx.imageSmoothingEnabled = true; ctx.drawImage(HEAT, pl.cx - 2.2 * pl.s, pl.cy - 1.5 * pl.s, 4.4 * pl.s, 3 * pl.s); setA(1); }
const KDE_SAMPLES = (() => { const r = rng(99), g = gaussFn(r); return Array.from({ length: 46 }, () => { const p = MOONS[Math.floor(r() * MOONS.length)]; return { x: p.x + BW * g(), y: p.y + BW * g() }; }); })();

// 1-D bimodal mixture — the "true" density of scenes 4 & 6
const MIX = [{ w: .4, m: -1.2, s: .5 }, { w: .6, m: 1.1, s: .7 }];
const mixPdf = x => MIX.reduce((a, c) => a + c.w * npdf(x, c.m, c.s), 0);
const mixCdf = x => MIX.reduce((a, c) => a + c.w * Phi((x - c.m) / c.s), 0);
function mixInv(u) { let lo = -6, hi = 6; for (let i = 0; i < 48; i++) { const mid = (lo + hi) / 2; if (mixCdf(mid) < u) lo = mid; else hi = mid; } return (lo + hi) / 2; }
const DATA1 = (() => { const r = rng(11), g = gaussFn(r); return Array.from({ length: 420 }, () => { const c = r() < MIX[0].w ? MIX[0] : MIX[1]; return { x: c.m + c.s * g(), r: r(), j: r() }; }); })();
const HIST = {};
for (const nb of [8, 16, 32, 64]) {
  const lo = -3.5, hi = 3.5, bw = (hi - lo) / nb, cnt = new Array(nb).fill(0);
  for (const d of DATA1) { const k = Math.floor((d.x - lo) / bw); if (k >= 0 && k < nb) cnt[k]++; }
  HIST[nb] = cnt.map(c => c / (DATA1.length * bw));
}

// maximum-likelihood data
const D5 = (() => { const r = rng(21), g = gaussFn(r); return Array.from({ length: 14 }, () => ({ x: 0.8 + 0.6 * g(), r: r() })); })();
const MU5 = D5.reduce((a, d) => a + d.x, 0) / D5.length;
const SD5 = Math.sqrt(D5.reduce((a, d) => a + (d.x - MU5) ** 2, 0) / D5.length);
const ll5 = (mu, sd) => D5.reduce((a, d) => a + Math.log(npdf(d.x, mu, sd)), 0);
const LLMAX = ll5(MU5, SD5);

// noise → data particles
const PART = (() => { const r = rng(31), g = gaussFn(r); const N = 900; return Array.from({ length: N }, (_, k) => { const z = clamp(g(), -2.95, 2.95); const u = (k + 1) / N; return { z, x: mixInv(Phi(z)), s: 14 + 26 * Math.sqrt(u), d: lerp(2.6, 1.0, u) }; }); })();
const GZ = z => mixInv(Phi(z));

// 2-D noise → moons map  g(z)
function moonMap(z1, z2) {
  const s = Phi(z1) * 2; let bx, by, nx, ny;
  if (s < 1) { const th = Math.PI * (1 - s); bx = Math.cos(th) - 0.5; by = Math.sin(th) - 0.25; nx = Math.cos(th); ny = Math.sin(th); }
  else { const th = Math.PI * (s - 1); bx = 0.5 - Math.cos(th); by = 0.25 - Math.sin(th); nx = -Math.cos(th); ny = -Math.sin(th); }
  const k = 0.09 * clamp(z2, -2.5, 2.5); return [bx + k * nx, by + k * ny];
}
const FIN = (() => { const r = rng(41), g = gaussFn(r); return Array.from({ length: 1400 }, () => { const z1 = g(), z2 = g(); const [tx, ty] = moonMap(z1, z2); return { z1, z2, tx, ty, r: r() }; }); })();
function finPos(p, u, sc) {
  const sx = p.z1 * sc, sy = p.z2 * sc; let x = lerp(sx, p.tx, u), y = lerp(sy, p.ty, u);
  const ang = 1.1 * Math.sin(Math.PI * u), c = Math.cos(ang), s = Math.sin(ang);
  return [x * c - y * s, x * s + y * c];
}

// Gaussian fit to moons
const GFIT = (() => {
  const n = MOONS.length; const mx = MOONS.reduce((a, p) => a + p.x, 0) / n, my = MOONS.reduce((a, p) => a + p.y, 0) / n;
  let a = 0, b = 0, c = 0; for (const p of MOONS) { a += (p.x - mx) ** 2; b += (p.x - mx) * (p.y - my); c += (p.y - my) ** 2; } a /= n; b /= n; c /= n;
  const tr = (a + c) / 2, dd = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const l11 = Math.sqrt(a), l21 = b / l11, l22 = Math.sqrt(c - l21 * l21);
  const r = rng(51), g = gaussFn(r);
  const samples = Array.from({ length: 44 }, () => { const z1 = g(), z2 = g(); return { x: mx + l11 * z1, y: my + l21 * z1 + l22 * z2 }; });
  return { mx, my, l1: tr + dd, l2: tr - dd, ang: 0.5 * Math.atan2(2 * b, a - c), samples };
})();

// pixel digit "3" and noise
let DIGIT = null;
function makeDigit() {
  const c = document.createElement('canvas'); c.width = 28; c.height = 28; const g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, 28, 28); g.translate(14, 15); g.rotate(0.12);
  g.fillStyle = '#fff'; g.font = 'bold 25px Georgia, serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('3', 0, 1);
  const d = g.getImageData(0, 0, 28, 28).data; const v = new Float32Array(784);
  for (let i = 0; i < 784; i++) v[i] = d[i * 4] / 255; return v;
}
function pixelGrid(x, y, cell, vals, col, a = 1, gap = 1) {
  setA(a); ctx.fillStyle = P.elevated; ctx.fillRect(x - 4, y - 4, cell * 28 + 8, cell * 28 + 8);
  for (let i = 0; i < 784; i++) { const v = vals(i); if (v < 0.02) continue; setA(a * v); ctx.fillStyle = col; ctx.fillRect(x + (i % 28) * cell, y + Math.floor(i / 28) * cell, cell - gap, cell - gap); }
  setA(1);
}

// plot mapping for 2-D scenes
const mk = (cx, cy, s) => ({ cx, cy, s, X: x => cx + s * x, Y: y => cy - s * y });
const boundary = (x, learn) => lerp(-0.3 * x + 0.1, 0.55 * Math.sin(Math.PI * (x + 1)), learn);
const sigm = v => 1 / (1 + Math.exp(-v));

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open: a trailer for the season, made only from its real models (data: code/export/export_ep01_assets.py)
const E1 = window.EP1, TBANG = 22.5;
const TILES = [
  { ep: '02', name: 'autoregressive', col: P.amber, x: 110, y: 170 }, { ep: '03', name: 'VAE', col: P.plum, x: 690, y: 170 }, { ep: '05', name: 'GAN', col: P.rose, x: 1270, y: 170 },
  { ep: '06', name: 'normalizing flow', col: P.ink, x: 110, y: 530 }, { ep: '07', name: 'energy-based model', col: P.plum, x: 690, y: 530 }, { ep: '09', name: 'diffusion', col: P.sage, x: 1270, y: 530 }];
const TW = 540, TH = 320, TT = i => 1.5 + i * 3.3;                       // tile i tunes in at TT(i)
const T_VAE = memo('t_vae', () => u8imgs(E1.vae)), T_GAN = memo('t_gan', () => E1.gan.x.map(u8imgs)), T_DIF = memo('t_dif', () => E1.diffusion.x.map(u8imgs));
const T_STATIC = memo('t_static', () => { const c = document.createElement('canvas'); c.width = 135; c.height = 80; return { c, g: c.getContext('2d'), k: -1 }; });
function tileStatic(x, y, w, h, t, a, seed) { if (GA * a <= 0.003) return; const S = T_STATIC, k = Math.floor(t * 24) * 7 + seed;
  if (S.k !== k) { S.k = k; const im = S.g.createImageData(135, 80), r = rng(k); for (let i = 0; i < im.data.length; i += 4) { const v = r() * 200; im.data[i] = v; im.data[i + 1] = v * 0.92; im.data[i + 2] = v * 0.86; im.data[i + 3] = 255; } S.g.putImageData(im, 0, 0); }
  setA(a); ctx.imageSmoothingEnabled = false; ctx.drawImage(S.c, x, y, w, h); ctx.imageSmoothingEnabled = true; setA(1); }
function landscape1(map) { return memo('t_land', () => { const n = map.n, h = Float32Array.from(b64bytes(map.u8), v => v / 255), c = document.createElement('canvas'); c.width = c.height = n;
  const g = c.getContext('2d'), im = g.createImageData(n, n), lo = hx('#151110'), hi = hx('#7a6ca6');
  for (let q = 0; q < n * n; q++) { const u = Math.pow(h[q], 1.15); for (let ch = 0; ch < 3; ch++) im.data[q * 4 + ch] = lo[ch] + (hi[ch] - lo[ch]) * u; im.data[q * 4 + 3] = 255; }
  g.putImageData(im, 0, 0); return c; }); }
const lerpStage = (S, f, k) => { const n = S.length - 1, fi = clamp(f, 0, n), i = Math.min(n - 1, Math.floor(fi)), u = fi - i; return [lerp(S[i][k][0], S[i + 1][k][0], u), lerp(S[i][k][1], S[i + 1][k][1], u)]; };
function digitGrid(imgsAt, x, y, cell, a, key) { for (let q = 0; q < 9; q++) drawImg(cached(key + q, () => imgCanvas(imgsAt[q], P.terracotta)), x + (q % 3) * (cell + 6), y + Math.floor(q / 3) * (cell + 6), cell, a, { bg: true }); }
function tileContent(i, x, y, t, a) {
  const tl = t - TT(i), cx = x + TW / 2, cy = y + TH / 2 - 10;
  if (i === 0) {                                                        // names typed by the counting model, letters settling out of noise
    const per = 2.2, k = Math.floor(Math.max(0, tl) / per) % E1.names.length, w = E1.names[k], u = (Math.max(0, tl) % per) / per, n = Math.min(w.length, Math.floor(u * 1.6 * w.length + 1));
    const r = rng(Math.floor(t * 18)), abc = 'abcdefghijklmnopqrstuvwxyz'; let s = '';
    for (let j = 0; j < n; j++) s += (j === n - 1 && u * 1.6 * w.length - j < 0.6) ? abc[Math.floor(r() * 26)] : w[j];
    text(s + (Math.floor(t * 3) % 2 ? '▍' : ' '), cx, cy + 20, { font: F.mono, size: 64, color: P.chalk, align: 'center', a });
    M('[p|p]( next letter | letters so far )', cx, cy + 90, 20, { align: 'center', a: a * 0.9 });
  }
  if (i === 1) {                                                        // the VAE decoder walking a loop through its 2-number code
    const f = (Math.max(0, tl) * 7) % 48, j = Math.floor(f), u = f - j, sz = 210, ang = f / 48 * TAU;
    drawImg(cached('tv' + j, () => imgCanvas(T_VAE[j], P.terracotta)), cx - sz / 2 - 60, cy - sz / 2 + 10, sz, a * (1 - u), { bg: true });
    drawImg(cached('tv' + (j + 1) % 48, () => imgCanvas(T_VAE[(j + 1) % 48], P.terracotta)), cx - sz / 2 - 60, cy - sz / 2 + 10, sz, a * u, { bg: false });
    ring(cx + 170, cy + 10, 55, P.border, a, 1.5); dot(cx + 170 + 55 * Math.cos(ang), cy + 10 + 55 * Math.sin(ang), 7, P.rose, a);
    text('z  (2 numbers)', cx + 170, cy + 95, { font: F.mono, size: 15, color: P.rose, align: 'center', a });
  }
  if (i === 2) {                                                        // the GAN's generator, epoch by epoch
    const s = Math.min(E1.gan.epochs.length - 1, Math.floor(clamp(tl / 4.2) * E1.gan.epochs.length));
    digitGrid(T_GAN[s], cx - 130, y + 34, 80, a, 'tg' + s + '_'); text(`epoch ${E1.gan.epochs[s]}`, cx + 190, cy + 10, { font: F.mono, size: 18, color: P.rose, align: 'center', a });
  }
  if (i === 3) {                                                        // the flow carrying noise onto the moons
    const f = (E1.flow.length - 1) * ease(prog(tl, 0.4, 3.2)), s = 62;
    for (let k = 0; k < E1.flow[0].length; k++) { const [px, py] = lerpStage(E1.flow, f, k); dot(cx + px * s, cy + 10 - py * s, 2.6, mixc(P.rose, P.terracotta, f / 8), a * 0.9); }
  }
  if (i === 4) {                                                        // Langevin marbles rolling into the energy valleys
    const R = E1.ebm.R, S = TH - 40, x0 = cx - S / 2, y0 = y + 14, L = E1.ebm.lang, f = (L.length - 1) * clamp(Math.max(0, tl) / 4.5) ** 0.75;
    setA(a); ctx.drawImage(landscape1(E1.ebm.E), x0, y0, S, S); setA(1);
    for (let k = 0; k < L[0].length; k++) { const [px, py] = lerpStage(L, f, k); dot(x0 + (px + R) / (2 * R) * S, y0 + (R - py) / (2 * R) * S, 2.6, P.terracotta, a * 0.95); }
  }
  if (i === 5) {                                                        // diffusion: nine digits denoised out of static
    const n = T_DIF.length - 1, s = Math.round(n * Math.pow(ease(prog(tl, 0.3, 4.2)), 0.85));
    digitGrid(T_DIF[s], cx - 130, y + 34, 80, a, 'td' + s + '_');
    const ts = E1.diffusion.ts[s]; text(ts < 0 ? 'done' : `t = ${ts}`, cx + 190, cy + 10, { font: F.mono, size: 18, color: P.sage, align: 'center', a });
  }
}
SC.push({ dur: 36, chapter: 'intro',
  caps: [[0.6, 5, 'Every AI that writes, paints or composes is quietly doing one thing:'],
         [5, 10.5, 'turning random noise into something that looks real.'],
         [10.5, 16.5, 'Names, digits, shapes, landscapes: six different machines, and every one of them is real.'],
         [16.5, 22, 'Each is a different answer to one question. This season, we build them all.'],
         [26, 31.5, 'From first principles, with the math, and with code you can run yourself.'],
         [31.5, 35.6, 'Watch the colours: each one means something, and it never changes.']],
  draw(t) {
    const [sx, sy] = shake(t, TBANG, 12); ctx.save(); ctx.translate(sx, sy);
    const dimT = 1 - 0.82 * ease(prog(t, TBANG - 0.4, 0.8)), hot = t >= TBANG ? Math.exp(-(t - TBANG) * 1.3) : 0;
    tileStatic(0, 0, 1920, 1080, t, 0.10 * (1 - ease(prog(t, 1, 2.5))), 3);                   // the whole screen starts as static
    TILES.forEach((T, i) => {
      const a = eout(prog(t, TT(i), 0.6)) * dimT; if (a <= 0) return; GA = 1;
      rbox(T.x, T.y, TW, TH, 16, rgba(P.surface, 0.9), P.border, a);
      ctx.save(); ctx.beginPath(); ctx.roundRect(T.x, T.y, TW, TH, 16); ctx.clip();
      tileContent(i, T.x, T.y, t, a * eout(prog(t, TT(i) + 0.5, 1.2)));
      tileStatic(T.x, T.y, TW, TH, t, a * 0.55 * (1 - ease(prog(t, TT(i) + 0.3, 1.4))), i);   // each tile tunes in out of static
      ctx.restore();
      if (hot > 0.01) rbox(T.x, T.y, TW, TH, 16, null, T.col, hot * 0.9, 2.5);
      text(`EP ${T.ep}`, T.x + 18, T.y + TH - 16, { font: F.mono, size: 15, color: T.col, a, ls: 2 });
      text(T.name, T.x + TW - 18, T.y + TH - 16, { font: F.serif, italic: true, size: 20, color: P.stone, align: 'right', a });
    });
    if (hot > 0.01) ring(960, 500, 180 + (t - TBANG) * 1500, P.chalk, hot * 0.35, 3);
    ctx.restore();
    const a = eout(prog(t, TBANG, 0.9));
    text('EPISODE 01', 960, 370, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a: eout(prog(t, TBANG + 0.5, 1)), ls: 4 });
    text('Generative Modelling', 960, 500 + (1 - a) * 18, { font: F.serif, size: 132, color: P.chalk, align: 'center', a, ls: -1 });
    text('what it is  ·  how it differs  ·  the math underneath', 960, 574, { size: 32, color: P.stone, align: 'center', a: eout(prog(t, TBANG + 1, 1.2)) });
    text('By the end of this season you’ll have built every one of those machines, from scratch.', 960, 660, { font: F.serif, italic: true, size: 30, color: P.chalk, align: 'center', a: eout(prog(t, 25, 1.4)) });
    const list = [[P.terracotta, 'x  data'], [P.ink, 'p  distribution'], [P.amber, 'θ  parameters'], [P.rose, 'z  noise'], [P.plum, 'g  model']];
    ctx.font = `400 22px ${F.mono}`; const tw = list.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 22 * 2.1 + 14, -14);
    let cx = 960 - tw / 2; list.forEach(([c, s], i) => { cx += tag(cx, 740, c, s, eout(prog(t, 31.6 + i * 0.4, 0.6)), 22) + 14; });
  } });

// 01 — data is points
SC.push({ dur: 40, chapter: 'data', title: 'Data is just points in space', sub: 'Any image, sound or sentence can be written as a list of numbers.',
  caps: [[1, 6, "Start tiny: an 'image' with just two pixels, each a brightness between 0 and 1."],
         [6, 11, 'Two numbers give one point on a plane. Change the pixels and the point moves.'],
         [11, 16.5, 'Collect many images and you get a cloud of points. That cloud is your dataset.'],
         [16.5, 21.5, "Notice it has a shape: neighbouring pixels tend to be similar, so the points hug the diagonal."],
         [22, 28, 'A real image works the same way. This 28 × 28 digit is just 784 brightness numbers.'],
         [28, 34, "So one image is one point — in 784-dimensional space. We can't draw that, but the math doesn't care."],
         [34, 39.6, 'Key idea: a dataset is a cloud of points, and the cloud has a shape.']],
  CLOUD: (() => { const r = rng(3), g = gaussFn(r); return Array.from({ length: 260 }, () => { const b = 0.08 + 0.84 * r(); return { x: b, y: clamp(b + 0.09 * g(), 0.02, 0.98), r: r() }; }); })(),
  draw(t) {
    const A = 1 - ease(prog(t, 20.8, 1.2));
    if (A > 0) {
      GA = A;
      const tt = Math.min(t, 11.5);
      const b1 = 0.5 + 0.4 * Math.sin(0.9 * tt + 0.3), b2 = 0.5 + 0.4 * Math.sin(1.3 * tt + 1.2);
      const a0 = eout(prog(t, 0.6, 0.8));
      [[b1, 200, 'pixel 1'], [b2, 400, 'pixel 2']].forEach(([b, x, nm]) => {
        text(nm, x + 85, 318, { size: 22, color: P.dust, align: 'center', a: a0 });
        setA(a0); const c = hx(P.chalk); ctx.fillStyle = `rgb(${Math.round(c[0] * b)},${Math.round(c[1] * b)},${Math.round(c[2] * b)})`; ctx.fillRect(x, 340, 170, 170);
        ctx.strokeStyle = P.borderLight; ctx.lineWidth = 2; ctx.strokeRect(x, 340, 170, 170); setA(1);
        text(b.toFixed(2), x + 85, 558, { font: F.mono, size: 30, color: P.terracotta, align: 'center', a: a0 });
      });
      card(200, 610, 370, 100, P.terracotta, eout(prog(t, 3, 0.8)));
      M('[d|x] = ([d|x][d_|1], [d|x][d_|2]) ∈ ℝ²', 232, 672, 30, { a: eout(prog(t, 3, 0.8)) });
      // axes
      const ox = 900, oy = 830, L = 520, ax = eout(prog(t, 1.5, 1));
      line(ox, oy, ox + L + 20, oy, P.borderLight, 2, ax); line(ox, oy, ox, oy - L - 20, P.borderLight, 2, ax);
      text('pixel 1 brightness  →', ox + L, oy + 40, { size: 22, color: P.dust, align: 'right', a: ax });
      ctx.save(); ctx.translate(ox - 26, oy); ctx.rotate(-Math.PI / 2); text('pixel 2 brightness  →', L, 0, { size: 22, color: P.dust, align: 'right', a: ax }); ctx.restore();
      text('0', ox - 16, oy + 26, { font: F.mono, size: 18, color: P.dust, a: ax }); text('1', ox + L - 5, oy + 26, { font: F.mono, size: 18, color: P.dust, a: ax }); text('1', ox - 24, oy - L + 6, { font: F.mono, size: 18, color: P.dust, a: ax });
      const ar = eout(prog(t, 6, 0.8));
      arrow(620, 425, 840, 425, P.dust, 2, ar); text('two numbers → one point', 730, 405, { font: F.mono, size: 18, color: P.dust, align: 'center', a: ar });
      // cloud
      this.CLOUD.forEach((p, i) => { const ti = 11.5 + 5 * (i / this.CLOUD.length); const u = prog(t, ti, 0.5); if (u > 0) dot(ox + p.x * L, oy - p.y * L, 5 * pop(u), P.terracotta, 0.75 * clamp(u * 2)); });
      // live point
      const px = ox + b1 * L, py = oy - b2 * L, pa = eout(prog(t, 5.5, 0.8));
      line(px, py, px, oy, P.terracotta, 1.5, pa * 0.5, [6, 6]); line(px, py, ox, py, P.terracotta, 1.5, pa * 0.5, [6, 6]);
      glow(px, py, 11, P.terracotta, pa);
      if (t > 16.5) { const u = eout(prog(t, 16.8, 1.5)); line(ox + 30, oy - 30, ox + 30 + (L - 60) * u, oy - 30 - (L - 60) * u, P.sage, 2, 0.7, [10, 8]); text('similar neighbours', ox + L - 40, oy - L + 70, { size: 22, color: P.sage, align: 'right', a: u }); }
      GA = 1;
    }
    const C = ease(prog(t, 21.6, 1));
    if (C > 0) {
      GA = C;
      const gx = 170, gy = 290, cell = 17;
      const fl = ease(prog(t, 25.5, 1));
      pixelGrid(gx, gy, cell, i => DIGIT[i], P.terracotta, 1 - 0.7 * fl);
      text('28 × 28 pixels', gx + 238, gy + 520, { font: F.mono, size: 20, color: P.dust, align: 'center' });
      const sx0 = 760, sy0 = 400, sc = 9.4;
      for (let i = 0; i < 784; i++) {
        const u = ease(prog(t, 25.5 + 2.6 * (i / 784), 1.3)); if (u <= 0) continue;
        const x = lerp(gx + (i % 28) * cell, sx0 + (i % 112) * sc, u), y = lerp(gy + Math.floor(i / 28) * cell, sy0 + Math.floor(i / 112) * sc, u), s = lerp(cell - 1, sc - 1, u);
        setA(0.12 + 0.88 * DIGIT[i]); ctx.fillStyle = DIGIT[i] > 0.02 ? P.terracotta : P.elevated; ctx.fillRect(x, y, s, s);
      }
      setA(1);
      const la = eout(prog(t, 28.6, 0.8));
      text('flatten  →  a list of 784 numbers', sx0, sy0 - 26, { font: F.mono, size: 20, color: P.terracotta, a: la });
      M('[d|x] = ([d|x][d_|1], [d|x][d_|2], … , [d|x][d_|784]) ∈ ℝ⁷⁸⁴', sx0, sy0 + 130, 34, { a: eout(prog(t, 29.2, 0.8)) });
      const pa = eout(prog(t, 31, 0.8));
      arrow(1290, sy0 + 160, 1290, sy0 + 250, P.dust, 2, pa);
      glow(1290, sy0 + 285, 12, P.terracotta, pa);
      text('one image  =  one point', 1320, sy0 + 294, { size: 28, color: P.chalk, a: pa });
      insight(sx0, 770, 1060, 'A dataset is a cloud of points — and the cloud has a shape. Learning that shape is what generative modelling is about.', eout(prog(t, 34, 0.8)));
      GA = 1;
    }
  } });

// 02 — discriminative
SC.push({ dur: 42, chapter: 'discriminative', title: 'Discriminative models learn the border', sub: 'The classic question: given an input, which label?',
  caps: [[1, 6.5, 'Now give each point a label. Say terracotta points are cat photos, clay points are dog photos.'],
         [6.5, 12.5, 'A discriminative model learns just one thing: a border that separates the classes.'],
         [12.5, 19, 'Mathematically it learns p(y | x): given an input x, how likely is each label y?'],
         [19, 26, "Perfect for questions like 'is this a cat?' — and it answers confidently."],
         [26, 33, "But now ask it to draw a cat… and it's stuck."],
         [33, 41.6, 'This faraway point is even MORE "cat" to the model — yet it looks like no cat at all. The border says nothing about where cats actually live.']],
  draw(t) {
    const pl = mk(640, 600, 225); const learn = ease(prog(t, 7, 6)), bA = eout(prog(t, 6.5, 1));
    const xs = []; for (let i = 0; i <= 120; i++) xs.push(-2.2 + 4.4 * i / 120);
    // regions
    setA(bA * 0.09); ctx.fillStyle = P.terracotta; ctx.beginPath(); ctx.moveTo(pl.X(-2.2), pl.Y(1.35)); ctx.lineTo(pl.X(2.2), pl.Y(1.35)); for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(pl.X(xs[i]), pl.Y(boundary(xs[i], learn))); ctx.fill();
    ctx.fillStyle = P.clay; ctx.beginPath(); ctx.moveTo(pl.X(-2.2), pl.Y(-1.35)); ctx.lineTo(pl.X(2.2), pl.Y(-1.35)); for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(pl.X(xs[i]), pl.Y(boundary(xs[i], learn))); ctx.fill(); setA(1);
    // points
    for (const p of MOONS) { const u = prog(t, 0.5 + 2.5 * p.r, 0.5); if (u > 0) dot(pl.X(p.x), pl.Y(p.y), 5 * pop(u), p.c ? P.clay : P.terracotta, 0.9 * clamp(u * 2)); }
    // boundary
    setA(bA); ctx.strokeStyle = P.plum; ctx.lineWidth = 5; ctx.beginPath(); xs.forEach((x, i) => { const X = pl.X(x), Y = pl.Y(boundary(x, learn)); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
    pill(pl.X(1.35), pl.Y(boundary(1.35, learn)) - 40, 'decision boundary', P.plum, eout(prog(t, 12, 0.8)));
    tagsRow([[P.terracotta, 'cat'], [P.clay, 'dog']], 180, 232, 1, t);
    // query point
    const Q = [-0.3, 0.72], qa = eout(prog(t, 19, 0.6)) * (1 - 0.6 * ease(prog(t, 33, 1)));
    if (qa > 0) { const pc = sigm(8 * (Q[1] - boundary(Q[0], learn))); ring(pl.X(Q[0]), pl.Y(Q[1]), 16 + 3 * Math.sin(t * 4), P.chalk, qa, 2.5); pill(pl.X(Q[0]) + 30, pl.Y(Q[1]) - 44, `p(cat | x) = ${pc.toFixed(2)}`, P.ink, qa); }
    const Fp = [-1.75, 1.2], fa = eout(prog(t, 33.5, 0.8));
    if (fa > 0) { const pc = sigm(8 * (Fp[1] - boundary(Fp[0], learn))); ring(pl.X(Fp[0]), pl.Y(Fp[1]), 16 + 3 * Math.sin(t * 4), P.terracotta, fa, 3); dot(pl.X(Fp[0]), pl.Y(Fp[1]), 6, P.terracotta, fa); pill(pl.X(Fp[0]) + 30, pl.Y(Fp[1]) + 4, `p(cat | x) = ${pc.toFixed(3)}  ?!`, P.ink, fa); }
    // right column
    const c1 = eout(prog(t, 12.5, 0.8));
    card(1250, 250, 580, 250, P.plum, c1); label('the discriminative question', 1284, 296, c1);
    M('[p|p]([c|y] | [d|x])', 1284, 382, 64, { a: c1 });
    text('probability of label y, given input x', 1284, 432, { size: 24, color: P.stone, a: c1 });
    tag(1284, 452, P.terracotta, 'x  input', c1, 17); tag(1420, 452, P.clay, 'y  label', c1, 17);
    const c2 = eout(prog(t, 26, 0.8));
    card(1250, 540, 580, 230, P.rose, c2); label('now ask it to create', 1284, 586, c2, P.rose);
    M('[d|x] ∼ ???', 1284, 660, 56, { a: c2 });
    para("p(y | x) never modelled x itself — there is nothing to sample from.", 1284, 718, 520, { size: 23, a: c2 });
  } });

// 03 — generative
SC.push({ dur: 44, chapter: 'generative', title: 'Generative models learn where data lives', sub: 'Not a border between answers — a map of the whole space.',
  caps: [[1, 7, 'A generative model asks a bigger question: where does the data live?'],
         [7, 14, 'It learns p(x) — a probability landscape. High where real data is dense, almost zero everywhere else.'],
         [14, 21, "Now that faraway point gets p(x) ≈ 0. The model knows it isn't a plausible cat."],
         [21, 29, 'And here is the magic: sample from p(x), and brand-new points appear that look like they belong.'],
         [29, 36, "That's generation. New images, sentences, sounds — all are samples from a learned p(x)."],
         [36, 43.6, "Bonus: a one-line formula, Bayes' rule, flips p(x | y) into p(y | x). The landscape contains the border, not the reverse."]],
  draw(t) {
    const pl = mk(640, 600, 225);
    drawHeat(pl, 0.95 * ease(prog(t, 4.5, 5)));
    for (const p of MOONS) dot(pl.X(p.x), pl.Y(p.y), 4.5, P.terracotta, 0.9 * eout(prog(t, 0.2 + p.r, 0.6)));
    tagsRow([[P.terracotta, 'x  data'], [P.ink, 'p(x)  density']], 180, 232, 1, t);
    const Fp = [-1.75, 1.2], fa = eout(prog(t, 14, 0.8)) * (1 - ease(prog(t, 21, 1)));
    if (fa > 0) {
      ring(pl.X(Fp[0]), pl.Y(Fp[1]), 16 + 3 * Math.sin(t * 4), P.terracotta, fa, 3); dot(pl.X(Fp[0]), pl.Y(Fp[1]), 6, P.terracotta, fa);
      pill(pl.X(Fp[0]) + 30, pl.Y(Fp[1]) + 4, `p(x) ≈ ${moonPdf(...Fp).toFixed(3)}`, P.ink, fa);
      const Q = [-0.3, 0.72]; const qa = fa * eout(prog(t, 16, 0.8));
      ring(pl.X(Q[0]), pl.Y(Q[1]), 16 + 3 * Math.sin(t * 4), P.chalk, qa, 2.5); pill(pl.X(Q[0]) + 30, pl.Y(Q[1]) - 44, `p(x) ≈ ${moonPdf(...Q).toFixed(2)}`, P.ink, qa);
    }
    // new samples
    KDE_SAMPLES.forEach((s, i) => { const u = prog(t, 21.5 + i * 0.17, 0.6); if (u <= 0) return; const X = pl.X(s.x), Y = pl.Y(s.y); ring(X, Y, 10 + 18 * eout(u), P.chalk, 0.7 * (1 - u), 2); glow(X, Y, 6.5 * pop(u), P.terracotta, 1); ring(X, Y, 10, P.chalk, 0.55 * clamp(u * 2), 1.5); });
    tag(520, 232, P.chalk, 'new samples', eout(prog(t, 22, 0.6)));
    // right column
    const c1 = eout(prog(t, 7, 0.8));
    card(1250, 250, 580, 230, P.ink, c1); label('the generative question', 1284, 296, c1);
    M('[p|p]([d|x])', 1284, 382, 64, { a: c1 });
    text('how plausible is x — anywhere in space?', 1284, 434, { size: 24, color: P.stone, a: c1 });
    const c2 = eout(prog(t, 21, 0.8));
    card(1250, 510, 580, 160, P.terracotta, c2); label('generate', 1284, 556, c2);
    M('[d|x][d_|new] ∼ [p|p]([d|x])', 1284, 626, 46, { a: c2 });
    const c3 = eout(prog(t, 36, 0.8));
    card(1250, 700, 580, 200, P.sage, c3); label("bayes' rule", 1284, 746, c3, P.sage);
    M('[p|p]([c|y]|[d|x]) = [p|p]([d|x]|[c|y]) [p|p]([c|y]) / [p|p]([d|x])', 1284, 808, 28, { a: c3 });
    text('generative ⇒ discriminative, for free', 1284, 862, { size: 24, color: P.sage, a: c3 });
  } });

// 04 — what is a density
SC.push({ dur: 40, chapter: 'density', title: 'What exactly is p(x)?', sub: 'Building a probability density from a pile of data, in one dimension.',
  caps: [[1, 7, "Let's drop to one dimension. Here's a pile of data points on a line."],
         [7, 13.5, 'Chop the line into bins and stack up how many points land in each: a histogram.'],
         [13.5, 20, 'Scale the bars so their total area is 1, then make the bins thinner and thinner…'],
         [20, 27, '…and the bars melt into a smooth curve. That curve is the probability density, p(x).'],
         [27, 33, "Two rules: it's never negative, and the total area under it is exactly 1."],
         [33, 39.6, 'Probability lives in area: the chance of landing between a and b is the area over that stretch.']],
  draw(t) {
    const x0 = 180, x1 = 1220, base = 780, hs = 1150;
    const X = x => x0 + (x + 3.5) / 7 * (x1 - x0), Y = d => base - d * hs;
    const ax = eout(prog(t, 0.4, 0.8));
    line(x0, base, x1, base, P.borderLight, 2, ax);
    for (let k = -3; k <= 3; k++) { line(X(k), base, X(k), base + 8, P.borderLight, 2, ax); text(String(k), X(k), base + 62, { font: F.mono, size: 18, color: P.dust, align: 'center', a: ax }); }
    text('x', x1 + 20, base + 8, { font: F.mono, size: 26, color: P.terracotta, a: ax });
    // histogram
    const nb = t < 14.5 ? 8 : t < 16.5 ? 16 : t < 18.5 ? 32 : 64; const bw = 7 / nb;
    const grow = eout(prog(t, 7.5, 2)), ha = 1 - 0.7 * ease(prog(t, 21, 2));
    if (grow > 0) {
      HIST[nb].forEach((d, k) => { const xa = X(-3.5 + k * bw), xb = X(-3.5 + (k + 1) * bw), y = Y(d * grow);
        setA(0.42 * ha); ctx.fillStyle = P.terracotta; ctx.fillRect(xa + 1, y, xb - xa - 2, base - y);
        setA(0.9 * ha); ctx.strokeStyle = P.terracotta; ctx.lineWidth = 1.5; ctx.strokeRect(xa + 1, y, xb - xa - 2, base - y); setA(1); });
      text(`bins: ${nb}`, x0, 300, { font: F.mono, size: 22, color: P.dust, a: grow * (1 - ease(prog(t, 20, 1))) });
    }
    // falling data
    for (const d of DATA1) { const u = prog(t, 1 + 5 * d.r, 0.9); if (u <= 0) continue; const yf = base + 16 + d.j * 16; dot(X(d.x), lerp(250 - 60 * d.j, yf, ein(u)), 3.2, P.terracotta, 0.75); }
    // curve
    const cu = ease(prog(t, 20.5, 2.5));
    const pts = []; for (let i = 0; i <= 300; i++) { const x = -3.5 + 7 * i / 300; pts.push([X(x), Y(mixPdf(x)), x]); }
    const area = ease(prog(t, 27.5, 1.2)) * (1 - 0.6 * ease(prog(t, 33, 1)));
    if (area > 0) { setA(0.2 * area); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.moveTo(X(-3.5), base); pts.forEach(p => ctx.lineTo(p[0], p[1])); ctx.lineTo(X(3.5), base); ctx.fill(); setA(1); }
    const ab = ease(prog(t, 33.5, 1.2)), A_ = 0.3, B_ = 1.7;
    if (ab > 0) {
      setA(0.5 * ab); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.moveTo(X(A_), base); pts.filter(p => p[2] >= A_ && p[2] <= B_).forEach(p => ctx.lineTo(p[0], p[1])); ctx.lineTo(X(B_), Y(mixPdf(B_))); ctx.lineTo(X(B_), base); ctx.fill(); setA(1);
      [[A_, 'a'], [B_, 'b']].forEach(([v, s]) => { line(X(v), base + 4, X(v), Y(mixPdf(v)) - 30, P.chalk, 1.5, ab, [5, 5]); text(s, X(v), base + 40, { font: F.mono, size: 26, color: P.chalk, align: 'center', a: ab }); });
      pill(X((A_ + B_) / 2), Y(mixPdf(1)) - 70, `area = ${(mixCdf(B_) - mixCdf(A_)).toFixed(2)}`, P.ink, ab, { align: 'center', size: 22 });
    }
    if (cu > 0) { setA(1); ctx.strokeStyle = P.ink; ctx.lineWidth = 5; ctx.beginPath(); const n = Math.floor(cu * 300); for (let i = 0; i <= n; i++) i ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1]); ctx.stroke(); }
    pill(X(-1.2), Y(mixPdf(-1.2)) - 40, 'p(x)', P.ink, eout(prog(t, 23, 0.8)), { align: 'center', size: 24 });
    pill(X(-2.4), Y(0.14), 'total area = 1', P.ink, eout(prog(t, 28.5, 0.8)) * (1 - ease(prog(t, 33, 0.8))), { align: 'center', size: 22 });
    // rules
    const c = eout(prog(t, 27, 0.8));
    card(1320, 250, 510, 300, P.ink, c); label('the rules of a density', 1352, 296, c);
    M('[p|p]([d|x]) ≥ 0', 1352, 364, 32, { a: eout(prog(t, 27.4, 0.8)) });
    M('[pB|∫] [p|p]([d|x]) d[d|x] = 1', 1352, 436, 32, { a: eout(prog(t, 29, 0.8)) });
    M('P(a ≤ [d|x] ≤ b) = [pB|∫][k_|a][k^|b] [p|p]([d|x]) d[d|x]', 1352, 508, 26, { a: eout(prog(t, 33.6, 0.8)) });
    insight(1320, 590, 510, 'p(x) is probability per unit length, not probability itself. Area turns density into probability.', eout(prog(t, 35, 0.8)));
  } });

// 05 — maximum likelihood
SC.push({ dur: 50, chapter: 'likelihood', title: 'Learning p(x): make the data likely', sub: 'Pick a family of shapes with knobs θ — then turn the knobs.',
  caps: [[1, 7, "We never see the true p(x). So we pick a family of shapes with knobs, θ, and tune them."],
         [7, 13, 'The simplest family: a bell curve. Two knobs — its centre μ and its width σ.'],
         [13, 20, 'How good is a guess? Read off the height of the curve at every data point: its likelihood.'],
         [20, 27, 'Multiply those heights together — in practice, add their logs. Bigger means the model finds the data more plausible.'],
         [27, 34, 'Too wide spreads probability thin. Too narrow misses points entirely. The best setting balances both.'],
         [34, 42, 'This is maximum likelihood: choose θ that makes the data we actually observed as probable as possible.'],
         [42, 49.6, 'Deep models do exactly this with millions of knobs, nudging each one in whichever direction improves the score, over and over: gradient descent.']],
  K: [[0, -1.6, .55], [16, -1.6, .55], [22, MU5, .55], [25, MU5, .55], [28, MU5, 1.7], [30, MU5, 1.7], [33, MU5, .3], [35, MU5, .3], [38.5, MU5, SD5]],
  params(t) { const K = this.K; for (let i = 0; i < K.length - 1; i++) { if (t < K[i + 1][0]) { const u = ease((t - K[i][0]) / (K[i + 1][0] - K[i][0])); return [lerp(K[i][1], K[i + 1][1], u), lerp(K[i][2], K[i + 1][2], u)]; } } return [K[K.length - 1][1], K[K.length - 1][2]]; },
  draw(t) {
    const [mu, sd] = this.params(t);
    const x0 = 170, x1 = 1170, base = 740, hs = 360;
    const X = x => x0 + (x + 3.5) / 7 * (x1 - x0), Y = d => base - d * hs;
    const ax = eout(prog(t, 0.4, 0.8));
    line(x0, base, x1, base, P.borderLight, 2, ax);
    for (let k = -3; k <= 3; k++) { line(X(k), base, X(k), base + 8, P.borderLight, 2, ax); text(String(k), X(k), base + 36, { font: F.mono, size: 18, color: P.dust, align: 'center', a: ax }); }
    // curve
    const ca = eout(prog(t, 7.5, 1));
    if (ca > 0) {
      setA(0.1 * ca); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.moveTo(x0, base);
      for (let i = 0; i <= 300; i++) { const x = -3.5 + 7 * i / 300; ctx.lineTo(X(x), Y(npdf(x, mu, sd))); } ctx.lineTo(x1, base); ctx.fill();
      setA(ca); ctx.strokeStyle = P.ink; ctx.lineWidth = 5; ctx.beginPath();
      for (let i = 0; i <= 300; i++) { const x = -3.5 + 7 * i / 300; i ? ctx.lineTo(X(x), Y(npdf(x, mu, sd))) : ctx.moveTo(X(x), Y(npdf(x, mu, sd))); } ctx.stroke(); setA(1);
      pill(X(mu) + 90, Y(npdf(mu, mu, sd)) - 10, 'pθ(x)', P.ink, ca, { size: 22 });
      // μ and σ
      line(X(mu), base, X(mu), Y(npdf(mu, mu, sd)), P.amber, 2, ca, [7, 6]);
      text('μ', X(mu), base + 64, { font: F.mono, size: 28, color: P.amber, align: 'center', a: ca });
      const ys = Y(npdf(mu + sd, mu, sd));
      line(X(mu - sd), ys, X(mu + sd), ys, P.amber, 3, ca); line(X(mu - sd), ys - 9, X(mu - sd), ys + 9, P.amber, 3, ca); line(X(mu + sd), ys - 9, X(mu + sd), ys + 9, P.amber, 3, ca);
      text('σ', X(mu + sd) + 14, ys + 8, { font: F.mono, size: 26, color: P.amber, a: ca });
    }
    // likelihood bars
    const la = eout(prog(t, 13.5, 1.5));
    for (const d of D5) {
      const u = prog(t, 1 + 3.5 * d.r, 0.7); if (u <= 0) continue;
      if (la > 0) { const h = Y(npdf(d.x, mu, sd)); line(X(d.x), base, X(d.x), h, P.ink, 3, 0.85 * la); dot(X(d.x), h, 5.5, P.ink, la); }
      glow(X(d.x), lerp(300, base, ein(u)), 7, P.terracotta, 1);
    }
    // commentary
    text('too wide → every point gets a little', x0 + 10, 300, { font: F.mono, size: 22, color: P.dust, a: eout(prog(t, 26, .6)) * (1 - eout(prog(t, 30.5, .5))) });
    text('too narrow → far points get almost nothing', x0 + 10, 300, { font: F.mono, size: 22, color: P.dust, a: eout(prog(t, 31, .6)) * (1 - eout(prog(t, 35.5, .5))) });
    text('just right ✓', x0 + 10, 300, { font: F.mono, size: 22, color: P.sage, a: eout(prog(t, 38.5, .6)) });
    // knobs
    const c1 = eout(prog(t, 7, 0.8));
    card(1290, 240, 540, 180, P.amber, c1); M('the knobs   [a|θ] = ([a|μ], [a|σ])', 1322, 286, 20, { a: c1 });
    M(`[a|μ] = ${mu.toFixed(2)}`, 1322, 350, 34, { a: c1 }); M(`[a|σ] = ${sd.toFixed(2)}`, 1322, 396, 34, { a: c1 });
    // score
    const c2 = eout(prog(t, 20, 0.8));
    const ll = ll5(mu, sd), frac = clamp((ll - (-70)) / (LLMAX + 70));
    card(1290, 440, 540, 220, P.plum, c2); label('score · log-likelihood', 1322, 486, c2, P.plum);
    M('[kB|Σ][k_|i] log [p|p][a_|θ]([d|x][d_|i])', 1322, 544, 30, { a: c2 });
    text(ll.toFixed(1), 1322, 612, { font: F.mono, size: 46, color: P.chalk, a: c2 });
    setA(c2); ctx.fillStyle = P.elevated; ctx.fillRect(1322, 630, 476, 10); ctx.fillStyle = P.plum; ctx.fillRect(1322, 630, 476 * frac, 10); setA(1);
    // MLE
    const c3 = eout(prog(t, 34, 0.8));
    card(1290, 680, 540, 180, P.sage, c3); label('maximum likelihood', 1322, 726, c3, P.sage);
    M('[a|θ]* = argmax[a_|θ] [kB|Σ][k_|i] log [p|p][a_|θ]([d|x][d_|i])', 1322, 784, 23, { a: c3 });
    M('≡ minimise KL([p|p][k_|data] ‖ [p|p][a_|θ])', 1322, 832, 22, { a: eout(prog(t, 37, 0.8)) });
    insight(170, 790, 1000, 'A neural network is just a very flexible pθ — millions of knobs instead of two.', eout(prog(t, 42, 0.8)));
  } });

// 06 — sampling
SC.push({ dur: 44, chapter: 'sampling', title: 'Generating: turn noise into data', sub: 'Once we have a model, how do we get a brand-new sample out of it?',
  caps: [[1, 7, 'Having p(x) is half the story. How do we actually produce a new sample?'],
         [7, 14, 'Computers are great at one kind of randomness: simple noise, like a standard bell curve we call z.'],
         [14, 21, 'So we learn a function g that bends that simple noise into the shape of our data.'],
         [21, 29, 'Each noise value rises to the curve, then slides across. Steep parts spread points out; flat parts bunch them up.'],
         [29, 36, "Pile up enough outputs and their histogram matches the target distribution — the one we built earlier."],
         [36, 43.6, 'This is the blueprint of nearly every modern generator: sample noise z, output x = g(z).']],
  draw(t) {
    const zy = 780, xx = 560; const ZX = z => 620 + (z + 3) * 105, XY = x => 780 - (x + 3) * 85;
    const ax = eout(prog(t, 0.5, 1));
    line(xx, zy, 1280, zy, P.borderLight, 2, ax); line(xx, zy, xx, 220, P.borderLight, 2, ax);
    text('z  (noise) →', 1280, zy + 34, { font: F.mono, size: 22, color: P.rose, align: 'right', a: ax });
    text('x  (data) ↑', xx + 14, 236, { font: F.mono, size: 22, color: P.terracotta, a: ax });
    // noise bell
    const nb = eout(prog(t, 7, 1.2));
    if (nb > 0) { setA(0.25 * nb); ctx.fillStyle = P.rose; ctx.beginPath(); ctx.moveTo(ZX(-3), zy + 6); for (let i = 0; i <= 120; i++) { const z = -3 + 6 * i / 120; ctx.lineTo(ZX(z), zy + 6 + npdf(z) * 250); } ctx.lineTo(ZX(3), zy + 6); ctx.fill();
      setA(nb); ctx.strokeStyle = P.rose; ctx.lineWidth = 3; ctx.beginPath(); for (let i = 0; i <= 120; i++) { const z = -3 + 6 * i / 120; i ? ctx.lineTo(ZX(z), zy + 6 + npdf(z) * 250) : ctx.moveTo(ZX(z), zy + 6 + npdf(z) * 250); } ctx.stroke(); setA(1); }
    // curve g
    const gu = ease(prog(t, 14, 3));
    if (gu > 0) { setA(1); ctx.strokeStyle = P.plum; ctx.lineWidth = 5; ctx.beginPath(); const n = Math.floor(200 * gu); for (let i = 0; i <= n; i++) { const z = -2.95 + 5.9 * i / 200; i ? ctx.lineTo(ZX(z), XY(GZ(z))) : ctx.moveTo(ZX(z), XY(GZ(z))); } ctx.stroke(); pill(ZX(2.6), XY(GZ(2.6)) + 40, 'x = g(z)', P.plum, gu, { align: 'center', size: 22 }); }
    // steep / flat notes
    const na = eout(prog(t, 22, 0.8)) * (1 - ease(prog(t, 29, 1)));
    if (na > 0) { pill(ZX(-0.35) + 30, XY(GZ(-0.35)), 'steep → spreads out', P.stone, na); pill(ZX(-1.6) - 10, XY(GZ(-1.6)) - 50, 'flat → bunches up', P.stone, na, { align: 'right' }); }
    // particles
    const bins = new Array(36).fill(0); let arrived = 0; const bwx = 6.4 / 36;
    for (const p of PART) {
      if (t < p.s) continue; const tau = (t - p.s) / p.d;
      if (tau >= 1) { const k = Math.floor((p.x + 3) / bwx); if (k >= 0 && k < 36) bins[k]++; arrived++; continue; }
      let X, Yv, col;
      if (tau < 0.45) { X = ZX(p.z); Yv = lerp(zy, XY(p.x), ease(tau / 0.45)); col = P.rose; }
      else { X = lerp(ZX(p.z), xx, ease((tau - 0.45) / 0.55)); Yv = XY(p.x); col = P.terracotta; }
      dot(X, Yv, 4.2, col, 0.9);
      if (tau > 0.42 && tau < 0.5) ring(ZX(p.z), XY(p.x), 6 + 60 * (tau - 0.42), P.chalk, 0.6);
    }
    if (arrived > 0) bins.forEach((c, k) => { const d = c / (arrived * bwx), yA = XY(-3 + (k + 1) * bwx), yB = XY(-3 + k * bwx); setA(0.5); ctx.fillStyle = P.terracotta; ctx.fillRect(xx - 4 - d * 700, yA + 1, d * 700, yB - yA - 2); setA(1); });
    text(`samples: ${arrived}`, 250, 250, { font: F.mono, size: 20, color: P.dust, a: clamp(arrived / 5) });
    const ta = eout(prog(t, 29.5, 1));
    if (ta > 0) { setA(ta); ctx.strokeStyle = P.ink; ctx.lineWidth = 3; ctx.setLineDash([8, 6]); ctx.beginPath(); for (let i = 0; i <= 150; i++) { const x = -3 + 6.4 * i / 150; i ? ctx.lineTo(xx - 4 - mixPdf(x) * 700, XY(x)) : ctx.moveTo(xx - 4 - mixPdf(x) * 700, XY(x)); } ctx.stroke(); ctx.setLineDash([]); setA(1); pill(260, XY(2.9), 'target p(x)', P.ink, ta, { align: 'center' }); }
    // recipe
    const c = eout(prog(t, 7, 0.8));
    card(1340, 250, 490, 300, P.plum, c); label('the recipe', 1372, 296, c);
    M('[n|z] ∼ N(0, 1)', 1372, 364, 34, { a: eout(prog(t, 7.4, 0.8)) });
    M('[d|x] = [m|g][a_|θ]([n|z])', 1372, 436, 34, { a: eout(prog(t, 14.4, 0.8)) });
    M('[p|p]([d|x]) = [p|p]([n|z]) · |d[n|z] / d[d|x]|', 1372, 508, 24, { a: eout(prog(t, 22, 0.8)) });
    insight(1340, 590, 490, 'Randomness in, structure out. Learning a generator means shaping g.', eout(prog(t, 36, 0.8)));
  } });

// 07 — the catch
SC.push({ dur: 40, chapter: 'the catch', title: 'The catch: real data is a thin sliver', sub: 'Why simple shapes are not enough.',
  caps: [[1, 7, 'Here is why this is hard. Pick a random point in 784-dimensional pixel space…'],
         [7, 14, '…and you get static. Try ten thousand times: still static. Real digits are astronomically rare.'],
         [14, 20, "Real data hugs a thin, curvy sliver of the space — mathematicians call it a manifold."],
         [20, 28, "A single bell curve can't bend. Fit one to the moons and it smears probability everywhere."],
         [28, 34, 'Its samples land in the gaps — like a blurry average of every image.'],
         [34, 39.6, 'We need a far more flexible g. Enter neural networks — plus a few clever tricks.']],
  draw(t) {
    const A = 1 - ease(prog(t, 19, 1));
    if (A > 0) {
      GA = A * eout(prog(t, 0.4, 0.8));
      const seed = Math.floor(3 * t + 900 * Math.pow(ease(prog(t, 7, 6)), 2));
      const r = rng(seed * 7919 + 1), noise = new Float32Array(784); for (let i = 0; i < 784; i++) noise[i] = r();
      pixelGrid(290, 300, 15, i => noise[i], P.rose, 1, 0);
      pixelGrid(1210, 300, 15, i => DIGIT[i], P.terracotta, 1);
      text('random point in ℝ⁷⁸⁴', 500, 280, { font: F.mono, size: 22, color: P.rose, align: 'center' });
      text('a real digit', 1420, 280, { font: F.mono, size: 22, color: P.terracotta, align: 'center' });
      text('vs', 960, 530, { font: F.serif, italic: true, size: 48, color: P.dust, align: 'center' });
      const tries = t < 1 ? 0 : t < 7 ? Math.floor((t - 1) * 3) + 1 : Math.floor(19 + 9981 * ease(prog(t, 7, 6)));
      text(`random tries: ${tries.toLocaleString('en-US')}`, 290, 780, { font: F.mono, size: 24, color: P.chalk, a: eout(prog(t, 1, 0.6)) });
      text('digits found: 0', 290, 818, { font: F.mono, size: 24, color: P.rose, a: eout(prog(t, 7, 0.6)) });
      text('real images: a thin sliver of ℝ⁷⁸⁴', 1420, 780, { size: 24, color: P.stone, align: 'center', a: eout(prog(t, 14, 0.8)) });
      GA = 1;
    }
    const B = ease(prog(t, 19.5, 1));
    if (B > 0) {
      GA = B; const pl = mk(700, 600, 230);
      for (const p of MOONS) dot(pl.X(p.x), pl.Y(p.y), 4.5, P.terracotta, 0.85);
      const ea = eout(prog(t, 21, 1.5));
      for (let k = 3; k >= 1; k--) { setA(ea * 0.1); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.ellipse(pl.X(GFIT.mx), pl.Y(GFIT.my), k * Math.sqrt(GFIT.l1) * pl.s, k * Math.sqrt(GFIT.l2) * pl.s, -GFIT.ang, 0, TAU); ctx.fill(); setA(ea * 0.8); ctx.strokeStyle = P.ink; ctx.lineWidth = 2; ctx.stroke(); setA(1); }
      GFIT.samples.forEach((s, i) => { const u = prog(t, 28 + i * 0.12, 0.5); if (u <= 0) return; ring(pl.X(s.x), pl.Y(s.y), 9 * pop(u), P.chalk, 0.8, 2); dot(pl.X(s.x), pl.Y(s.y), 4, P.terracotta, 1); });
      tagsRow([[P.terracotta, 'real data'], [P.ink, 'one gaussian fit'], [P.chalk, 'its samples']], 180, 232, 20, t);
      const c1 = eout(prog(t, 21, 0.8));
      card(1250, 260, 580, 230, P.ink, c1); label('one bell curve', 1284, 306, c1);
      M('[p|p]([d|x]) = N([d|x]; [a|μ], [a|Σ])', 1284, 376, 32, { a: c1 });
      para("a single blob can't bend around the moons", 1284, 430, 520, { size: 24, a: c1 });
      const c2 = eout(prog(t, 34, 0.8));
      card(1250, 520, 580, 220, P.plum, c2); label('what we need', 1284, 566, c2, P.plum);
      M('[d|x] = [m|g][a_|θ]([n|z])', 1284, 634, 36, { a: c2 });
      para('with g a neural network — millions of θ, able to bend space', 1284, 684, 520, { size: 24, a: c2 });
      GA = 1;
    }
  } });

// 08 — the season map: every way to make a flexible p(x) trainable, in the order we build them
const ROUTE = [
  { ep: '01', name: 'What is p(x)?', tag: 'where the data lives', eq: '[p|p]([d|x])', col: P.terracotta, t: 1 },
  { ep: '02', name: 'Autoregressive', tag: 'one piece at a time', eq: '[p|p]([d|x]) = ∏ [p|p]([d|x][d_|t] | [d|x][d_|<t])', col: P.amber, t: 7.5 },
  { ep: '03', name: 'VAEs', tag: 'squeeze into a code, decode it', eq: 'ELBO ≤ log [p|p]([d|x])', col: P.plum, t: 10.8 },
  { ep: '04', name: 'Evaluation', tag: 'how do you grade a generator?', eq: 'FID · precision · recall', col: P.sage, t: 14 },
  { ep: '05', name: 'GANs', tag: 'a forger and a judge', eq: '[m|D]* = [p|p][k_|d] / ([p|p][k_|d] + [p|p][k_|g])', col: P.rose, t: 17.5 },
  { ep: '06', name: 'Flows', tag: 'invertible warps, exact p(x)', eq: '[p|p]([d|x]) = [p|p]([n|z]) / |det J|', col: P.ink, t: 20.5 },
  { ep: '07', name: 'Energy', tag: 'any landscape at all', eq: '[p|p]([d|x]) ∝ e^−E([d|x])', col: P.plum, t: 24 },
  { ep: '08', name: 'Score', tag: 'learn only the arrows', eq: '[s|s]([d|x]) = ∇ log [p|p]([d|x])', col: P.sage, t: 27 },
  { ep: '09', name: 'Diffusion', tag: 'noise, run backwards', eq: '‖[n|ε] − [m|ε][m_|θ]([d|x][d_|t], t)‖²', col: P.terracotta, t: 30.5 },
  { ep: '10', name: 'Flow matching', tag: 'straight roads, noise → data', eq: 'd[d|x]/dt = [m|v][m_|θ]([d|x], t)', col: P.ink, t: 34 },
  { ep: '11', name: 'Guidance', tag: 'steer it: text → image', eq: '[p|p]([d|x] | [c|caption])', col: P.amber, t: 37.5 }];
const STOP = i => i < 6 ? [150 + i * 300, 330] : [1650 - (i - 6) * 375, 650];
function routePath(u) {                                              // position along the route, u ∈ [0, 10]
  const i = Math.min(9, Math.floor(u)), f = u - i, [x0, y0] = STOP(i), [x1, y1] = STOP(i + 1);
  if (i === 5) { const ang = -Math.PI / 2 + Math.PI * f; return [1650 + 190 * Math.cos(ang), 490 + 160 * Math.sin(ang)]; }   // the bend down to the second row
  return [lerp(x0, x1, f), lerp(y0, y1, f)];
}
SC.push({ dur: 56, chapter: 'season map', title: 'The route: a map of the season', sub: 'Every way to make a flexible p(x) trainable, in the order we will build them.',
  caps: [[1, 7.5, 'There are several clever ways to make a flexible p(x) trainable. This season walks through them, in order.'],
         [7.5, 14, 'Autoregressive models generate one piece at a time. VAEs squeeze data into a small code, then decode it.'],
         [14, 20.5, 'Then a detour: how do you grade a generator? The answer turns a judge into a teacher: GANs.'],
         [20.5, 27, 'Flows warp noise with an invertible map, so p(x) is exact. Energy models let any network be a landscape.'],
         [27, 34, 'Score models learn only the arrows toward likelier data. Diffusion runs noise backwards, one small step at a time.'],
         [34, 41, 'Flow matching straightens the road from noise to data. Guidance steers it: type a caption, get that picture.'],
         [41, 49, 'Each stop fixes a weakness of the one before. By the end, you will know how modern image generators work.'],
         [49, 55.6, 'And you will have built every one of them, from scratch.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8));
    // the track: dashed where we have not been yet, solid terracotta behind the traveller
    const reach = ROUTE.reduce((acc, s, i) => t >= s.t ? i : acc, 0), trav = t < 41 ? reach : 10 * ease(prog(t, 41, 7.5));
    ctx.save(); setA(a0 * 0.8); ctx.strokeStyle = P.borderLight; ctx.lineWidth = 3; ctx.setLineDash([6, 8]); ctx.beginPath();
    for (let u = 0; u <= 10.001; u += 0.05) { const [x, y] = routePath(Math.min(u, 10)); u ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); setA(1);
    ctx.save(); setA(a0); ctx.strokeStyle = P.terracotta; ctx.lineWidth = 4; ctx.beginPath();
    for (let u = 0; u <= trav + 1e-6; u += 0.05) { const [x, y] = routePath(Math.min(u, trav)); u ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); setA(1);
    ROUTE.forEach((s, i) => {
      const a = eout(prog(t, s.t, 0.7)) * a0; if (a <= 0) return; const [x, y] = STOP(i), p = pop(prog(t, s.t, 0.5));
      const lit = t >= 41 && trav >= i - 0.02, here = i === 0;
      dot(x, y, 12 * p, lit || here ? s.col : P.elevated, a); ring(x, y, 12 * p, s.col, a, 2.5);
      if (here) { const pl = 0.5 + 0.5 * Math.sin(t * 3); ring(x, y, 18 + 8 * pl, P.terracotta, a * (0.8 - 0.6 * pl), 2); text('you are here', x, y - 30, { font: F.mono, size: 14, color: P.terracotta, align: 'center', a }); }
      text(s.ep, x, y + 42, { font: F.mono, size: 15, color: s.col, align: 'center', a, ls: 2 });
      text(s.name, x, y + 78, { font: F.serif, size: 30, color: P.chalk, align: 'center', a });
      text(s.tag, x, y + 108, { size: 18, color: P.stone, align: 'center', a });
      M(s.eq, x, y + 142, 16, { align: 'center', a: a * 0.95 });
    });
    if (t >= 41) { const [x, y] = routePath(trav); glow(x, y, 9, P.chalk, eout(prog(t, 41, 0.5)) * (1 - ease(prog(t, 49, 1)))); }
    const fa = eout(prog(t, 49, 1)); if (fa > 0) text('built from scratch, one stop at a time', 960, 910, { font: F.serif, italic: true, size: 30, color: P.amber, align: 'center', a: fa });
  } });

// 09 — finale
SC.push({ dur: 30, chapter: 'big idea', title: 'Noise in, meaning out', sub: 'The whole idea, in two dimensions you can see.',
  caps: [[1, 7, 'Put it all together, in two dimensions you can actually see.'],
         [7, 15, 'Start with pure noise. A trained network g moves every point to where data lives.'],
         [15, 22, 'The result: fresh samples from pθ(x) — close to the real data distribution.'],
         [22, 29.6, 'Scale this up to millions of dimensions, and those points become faces, songs and sentences.']],
  draw(t) {
    const pl = mk(960, 560, 245);
    drawHeat(pl, 0.5 * ease(prog(t, 15, 3)));
    for (const p of FIN) { const u = ease(prog(t, 6 + 2 * p.r, 7)); const [x, y] = finPos(p, u, 0.45); dot(pl.X(x), pl.Y(y), 2.8, mixc(P.rose, P.terracotta, u), 0.85 * eout(prog(t, 0.5 + 2 * p.r, 1))); }
    const ma = eout(prog(t, 15, 1));
    card(330, 845, 1260, 80, P.plum, ma);
    M('[n|z] ∼ N(0, I)   →   [d|x] = [m|g][a_|θ]([n|z])   ∼   [p|p][a_|θ]([d|x]) ≈ [p|p][k_|data]([d|x])', 960, 897, 32, { align: 'center', a: ma });
  } });

// 10 — build it (code/density.py)
const CODE1 = [
  '[k|# a mixture of bell curves: knobs θ = weights w, centres m, widths s]',
  '[m|def] [p|pdf](x, w, m, s):',
  '    [m|return] (w * normal(x[:, [m|None]], m, s)).sum([a|1])',
  '',
  '[k|# fit by maximum likelihood (EM): every step raises Σ log p(xᵢ)]',
  '[m|for] _ [m|in] [p|range]([a|200]):',
  '    r = w * normal(x[:, [m|None]], m, s);  r /= r.sum([a|1], keepdims=[m|True])',
  '    n = r.sum([a|0]);  w = n / [p|len](x);  m = (r * x[:, [m|None]]).sum([a|0]) / n',
  '    s = sqrt((r * (x[:, [m|None]] - m)**[a|2]).sum([a|0]) / n)',
  '',
  '[k|# sample: noise in, data out]',
  'k = choice([p|len](w), size=[a|5], p=w)',
  'x_new = m[k] + s[k] * randn([a|5])',
];
SC.push({ dur: 32, chapter: 'build it', title: 'Build it yourself', sub: 'A density you can fit and sample, in a dozen lines of numpy.',
  caps: [[1, 9, 'Everything in this episode fits in a dozen lines of numpy: a mixture of bell curves, fitted by maximum likelihood.'],
         [9, 18, 'Fit one, two, three, four bumps. Two wins on unseen data; more bumps start memorising the training points.'],
         [18, 31.6, 'Then sample: pick a bump, stretch and shift a bell curve of noise. Noise in, data out. The code is linked below, with a playground to try it live.']],
  draw(t) {
    playPill('ep01-density-lab.html', 9, t);
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1100, 690, P.plum, a);
    CODE1.forEach((l, i) => { if (l) M(l, 136, 300 + i * 46, 19, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 9, 0.8)); card(1240, 240, 580, 690, P.sage, ra); label('the whole episode', 1276, 290, ra, P.sage);
    [['data', 'points [d|x] ∈ ℝᵈ'], ['a density', '[p|p]([d|x]) ≥ 0,  ∫ [p|p] = 1'], ['learning', 'max[a_|θ] Σ log [p|p][a_|θ]([d|x][d_|i])'], ['generating', '[d|x] = [m|g]([n|z]),  [n|z] ~ N(0, 1)'], ['the catch', 'data lives on thin slivers']].forEach(([nm, m], i) => {
      const ai = eout(prog(t, 9.5 + i * 1.6, 0.8)); text(nm, 1276, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(m, 1276, 412 + i * 112, 21, { a: ai }); });
  } });

// 11 — recap
SC.push({ dur: 34, chapter: 'recap', title: 'Two questions, two kinds of model', sub: 'Everything in one frame.',
  caps: [[1, 8, 'Recap. Discriminative models learn p(y | x): the border between answers.'],
         [8, 15, "Generative models learn p(x): the whole landscape of what's plausible."],
         [15, 23, 'We fit it by maximum likelihood, and we generate by bending simple noise: x = g(z).'],
         [23, 33.6, 'Understanding where the data lives — that is what lets a machine imagine.']],
  draw(t) {
    const dimA = 1 - 0.8 * ease(prog(t, 23, 1.2));
    const cols = [
      { x: 110, name: 'Discriminative', acc: P.plum, t0: 1, learns: '[p|p]([c|y] | [d|x])', ans: '“Which label is this?”', pic: 'a border between classes', ex: 'spam filters · image classifiers · fraud detection' },
      { x: 990, name: 'Generative', acc: P.ink, t0: 8, learns: '[p|p]([d|x])   or   [p|p]([d|x], [c|y])', ans: '“What is plausible? Make me a new one.”', pic: 'a landscape over the whole space', ex: 'GPT · Stable Diffusion · music & speech models' },
    ];
    cols.forEach((c, j) => {
      const a = eout(prog(t, c.t0, 0.8)) * dimA; if (a <= 0) return; GA = a;
      card(c.x, 230, 820, 650, c.acc, 1);
      text(c.name, c.x + 36, 290, { font: F.serif, size: 44, color: P.chalk });
      const pl = mk(c.x + 410, 440, 90);
      if (j === 1) drawHeat(pl, 0.9);
      if (j === 0) { setA(0.9); ctx.strokeStyle = P.plum; ctx.lineWidth = 3.5; ctx.beginPath(); for (let i = 0; i <= 80; i++) { const x = -2.2 + 4.4 * i / 80; i ? ctx.lineTo(pl.X(x), pl.Y(boundary(x, 1))) : ctx.moveTo(pl.X(x), pl.Y(boundary(x, 1))); } ctx.stroke(); setA(1); }
      MOONS.forEach((p, k) => { if (k % 2 === 0 || k % 3 === 0) dot(pl.X(p.x), pl.Y(p.y), 2.6, j === 0 && p.c ? P.clay : P.terracotta, 0.9); });
      if (j === 1) KDE_SAMPLES.slice(0, 16).forEach(s => { ring(pl.X(s.x), pl.Y(s.y), 6, P.chalk, 0.7, 1.5); });
      const r0 = 610;
      label('learns', c.x + 36, r0); M(c.learns, c.x + 36, r0 + 48, 34);
      label('answers', c.x + 36, r0 + 100); text(c.ans, c.x + 36, r0 + 140, { size: 27, color: P.chalk });
      label('examples', c.x + 36, r0 + 196); text(c.ex, c.x + 36, r0 + 234, { size: 23, color: P.stone });
      text(c.pic, c.x + 784, 290, { font: F.serif, italic: true, size: 24, color: P.dust, align: 'right' });
      GA = 1;
    });
    const qa = eout(prog(t, 23.5, 1.5));
    if (qa > 0) {
      text('“To create, a model must first understand', 960, 500, { font: F.serif, size: 62, color: P.chalk, align: 'center', a: qa });
      text('where the data lives.”', 960, 580, { font: F.serif, size: 62, color: P.chalk, align: 'center', a: qa });
      M('learn  [p|p][a_|θ]([d|x])   ·   sample  [n|z]   ·   generate  [d|x] = [m|g][a_|θ]([n|z])', 960, 680, 28, { align: 'center', a: eout(prog(t, 26, 1)) });
    }
  } });

// 12 — the hook: one piece at a time (episode 02's cold open types this very sentence)
const TYPE1 = 'Once upon a time, a machine learned to write';
SC.push({ dur: 34, chapter: 'next', title: 'Where we start', sub: 'The simplest trick of all.',
  caps: [[1, 8, 'So which trick comes first? The simplest one: never try to write p(x) down in one go.'],
         [8, 15, 'Break the data into pieces, and guess each piece from the ones before it: a letter, then the next.'],
         [15, 21, 'Multiply those small guesses and you get p(x) exactly. It is how every chatbot writes.'],
         [21.5, 33.6, 'Next episode: autoregressive models. Writing the future, one piece at a time.']],
  draw(t) {
    const dimAll = 1 - ease(prog(t, 21, 1)); GA = dimAll;                     // the demo leaves completely for the end card
    const n = Math.min(TYPE1.length, Math.floor(clamp((t - 2) / 16) * TYPE1.length)), a0 = eout(prog(t, 0.6, 0.8));
    ctx.font = `400 44px ${F.mono}`; const cw = ctx.measureText('m').width, x0 = 960 - cw * TYPE1.length / 2;
    text(TYPE1.slice(0, n) + (Math.floor(t * 3) % 2 && n < TYPE1.length ? '▍' : ''), x0, 420, { font: F.mono, size: 44, color: P.chalk, a: a0 });
    // the chain rule grows one factor per letter: the last few factors, with the newest lit
    if (n > 0) { const k0 = Math.max(0, n - 4), terms = [];
      for (let k = k0; k < n; k++) { const c = TYPE1[k] === ' ' ? '␣' : TYPE1[k], ctxs = k === 0 ? '' : ' | ' + (k > 3 ? '…' : '') + TYPE1.slice(Math.max(0, k - 3), k).replace(/ /g, '␣');
        terms.push(`[p|p]([${k === n - 1 ? 'd' : 't'}|${c}]${ctxs})`); }
      M('[p|p]([d|x]) = ' + (k0 > 0 ? '… · ' : '') + terms.join(' · ') + (n < TYPE1.length ? ' · …' : ''), 960, 540, 28, { align: 'center', a: a0 * eout(prog(t, 8, 1)) }); }
    const ca = eout(prog(t, 8, 0.8)); card(560, 610, 800, 110, P.amber, ca);
    text('each guess only ever picks among a few dozen letters', 960, 676, { font: F.serif, italic: true, size: 28, color: P.chalk, align: 'center', a: ca });
    GA = 1;
    const qa = eout(prog(t, 21.5, 1.2));
    if (qa > 0) {
      text('One piece at a time.', 960, 470, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('NEXT  ·  EPISODE 02', 960, 600, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 22.5, 1)), ls: 4 });
      seasonStrip(1, 23.7, t);
      text('Autoregressive generation — writing the future, one piece at a time', 960, 655, { font: F.serif, italic: true, size: 40, color: P.stone, align: 'center', a: eout(prog(t, 23, 1)) });
    }
  } });

// glyphs and heat maps are rasterised once the fonts have loaded
ON_READY.push(() => { DIGIT = makeDigit(); HEAT = makeHeat(220, 150); });
