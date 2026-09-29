// ─────────────────────────────────────────────────────────────
//  A real trained energy-based model (exported by code/export_ep07_assets.py)
// ─────────────────────────────────────────────────────────────
const A = window.EP7, RR = A.R, NL = A.lang.length - 1;
function b64bytes(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
const OC = new Map();  // per-object caches (maps are objects in the assets, so keys never collide)
function oc(obj, tag, make) { let m = OC.get(obj); if (!m) OC.set(obj, m = {}); return m[tag] || (m[tag] = make()); }
function square(x0, y0, S) { return { X: v => x0 + (v + RR) / (2 * RR) * S, Y: v => y0 + (RR - v) / (2 * RR) * S, s: S / (2 * RR), x0, y0, S }; }
function frame(m, a = 1) { setA(a); ctx.fillStyle = P.surface; ctx.fillRect(m.x0, m.y0, m.S, m.S); ctx.strokeStyle = P.border; ctx.lineWidth = 1.5; ctx.strokeRect(m.x0, m.y0, m.S, m.S); setA(1); }
function clipTo(m, fn) { ctx.save(); ctx.beginPath(); ctx.rect(m.x0, m.y0, m.S, m.S); ctx.clip(); fn(); ctx.restore(); }
const heights = map => oc(map, 'h', () => Float32Array.from(b64bytes(map.u8), v => v / 255));
// the landscape: height-shaded relief (valleys dark, ridges lit plum), lit from the top left
function landCanvas(map) { return oc(map, 'L', () => { const n = map.n, h = heights(map), c = document.createElement('canvas'); c.width = c.height = n;
  const g = c.getContext('2d'), im = g.createImageData(n, n), lo = hx('#151110'), hi = hx('#7a6ca6'), L = [-0.55, -0.62, 0.56], k = n * 0.9;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const q = j * n + i, v = h[q];
    const gx = (h[j * n + Math.min(n - 1, i + 1)] - h[j * n + Math.max(0, i - 1)]) * k, gy = (h[Math.min(n - 1, j + 1) * n + i] - h[Math.max(0, j - 1) * n + i]) * k;
    const nl = Math.hypot(gx, gy, 1), sh = clamp((-gx * L[0] - gy * L[1] + L[2]) / nl / 0.56 * 0.5 + 0.25, 0, 1.3), u = Math.pow(v, 1.15), lit = 0.55 + 0.6 * sh;
    for (let ch = 0; ch < 3; ch++) im.data[q * 4 + ch] = clamp((lo[ch] + (hi[ch] - lo[ch]) * u) * lit, 0, 255);
    im.data[q * 4 + 3] = 255; }
  g.putImageData(im, 0, 0); return c; }); }
function densCanvas(d, col) { return oc(d, 'd' + col, () => { const u = b64bytes(d.u8), n = d.n, c = document.createElement('canvas'); c.width = c.height = n; const g = c.getContext('2d'), im = g.createImageData(n, n), rgb = hx(col);
  for (let k = 0; k < n * n; k++) { im.data[k * 4] = rgb[0]; im.data[k * 4 + 1] = rgb[1]; im.data[k * 4 + 2] = rgb[2]; im.data[k * 4 + 3] = Math.round(u[k] * 0.92); } g.putImageData(im, 0, 0); return c; }); }
// grid samples sit on linspace(-R, R, n): stretch the image by half a pixel each side so pixel centres line up
function drawMap(m, c, n, a = 1) { if (GA * a <= 0.002) return; const px = m.S / (n - 1); setA(a); ctx.imageSmoothingEnabled = true; clipTo(m, () => ctx.drawImage(c, m.x0 - px / 2, m.y0 - px / 2, m.S + px, m.S + px)); setA(1); }
// contour lines by marching squares, in grid units
// heights are log(1 + E/2) / log(1 + emax/2) above each map's floor; contour lines at fixed energies
const hOf = e => Math.log1p(e / 2) / Math.log1p(A.emax / 2), eOf = h => 2 * Math.expm1(h * Math.log1p(A.emax / 2));
const LEVELS = [0.4, 1.2, 2.5, 4.5, 7.5, 12, 18, 27, 38].map(hOf);
function contours(map) { return oc(map, 'c', () => { const n = map.n, h = heights(map);
  return LEVELS.map(Lv => { const out = [];
    for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) { const a = h[j * n + i], b = h[j * n + i + 1], c = h[(j + 1) * n + i + 1], d = h[(j + 1) * n + i], pts = [];
      const e = (p, q, x1, y1, x2, y2) => { if ((p < Lv) !== (q < Lv)) { const u = (Lv - p) / (q - p); pts.push(x1 + (x2 - x1) * u, y1 + (y2 - y1) * u); } };
      e(a, b, i, j, i + 1, j); e(b, c, i + 1, j, i + 1, j + 1); e(c, d, i + 1, j + 1, i, j + 1); e(d, a, i, j + 1, i, j);
      for (let k = 0; k + 3 < pts.length; k += 4) out.push(pts[k], pts[k + 1], pts[k + 2], pts[k + 3]); }
    return out; }); }); }
function drawContours(m, map, a = 1, grow = 1) { if (GA * a <= 0.002) return; const n = map.n, s = m.S / (n - 1), segs = contours(map);
  clipTo(m, () => segs.forEach((sg, li) => { const la = a * clamp(grow * LEVELS.length - li); if (la <= 0) return;
    setA(la * (li < 2 ? 0.75 : 0.42)); ctx.strokeStyle = li < 2 ? P.ink : mixc(P.plum, P.chalk, 0.25); ctx.lineWidth = li < 2 ? 1.4 : 1.1; ctx.beginPath();
    for (let k = 0; k < sg.length; k += 4) { ctx.moveTo(m.x0 + sg[k] * s, m.y0 + sg[k + 1] * s); ctx.lineTo(m.x0 + sg[k + 2] * s, m.y0 + sg[k + 3] * s); } ctx.stroke(); })); setA(1); }
function landscape(m, map, a = 1, grow = 1) { frame(m, a); drawMap(m, landCanvas(map), map.n, a); drawContours(m, map, a, grow); }
// exact energy above the floor (no display cap), from a 49 × 49 grid
function Eraw(x, y) { const V = A.Eraw.v, n = A.Eraw.n, fi = clamp((x + RR) / (2 * RR) * (n - 1), 0, n - 1.001), fj = clamp((RR - y) / (2 * RR) * (n - 1), 0, n - 1.001), i = Math.floor(fi), j = Math.floor(fj), u = fi - i, v = fj - j;
  return (V[j][i] * (1 - u) + V[j][i + 1] * u) * (1 - v) + (V[j + 1][i] * (1 - u) + V[j + 1][i + 1] * u) * v; }
function marble(x, y, r, col, a = 1) { if (GA * a <= 0.002) return; dot(x, y, r, col, a); dot(x - r * 0.32, y - r * 0.36, r * 0.36, P.chalk, a * 0.5); }
// marbles along recorded trajectories (stages every 4 sampler steps), linear between records
function stagePts(S, f) { const n = S.length - 1, fi = clamp(f, 0, n), i = Math.min(n - 1, Math.floor(fi)), u = fi - i; return S[i].map((p, k) => [lerp(p[0], S[i + 1][k][0], u), lerp(p[1], S[i + 1][k][1], u)]); }
function drawMarbles(m, S, f, col, a, r = 3.4, count = 1e9, pop0 = null) { const pts = stagePts(S, f); clipTo(m, () => pts.forEach((p, k) => { if (k >= count) return; const pa = pop0 ? pop0(k) : 1; if (pa > 0) marble(m.X(p[0]), m.Y(p[1]), r * (pa < 1 ? pop(pa) : 1), col, a * clamp(pa * 3)); })); }
const fmt2 = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1);

// the 1-D landscape (the one episode 6 ended on)
const Efn = x => 0.12 * x ** 4 - 1.1 * x ** 2 + 0.35 * x + 3.0 + 0.4 * Math.sin(3 * x);
const Z1 = (() => { let s = 0; for (let x = -3.3; x <= 3.3; x += 0.005) s += Math.exp(-Efn(x)) * 0.005; return s; })();
function axes1(X0, X1, Yb, a) { line(X0, Yb, X1, Yb, P.borderLight, 2, a); }
function curve(fn, x0, x1, X, Y, col, lw, a, n = 240, fill = null, base = null) { if (GA * a <= 0.002) return; ctx.beginPath(); for (let i = 0; i <= n; i++) { const x = x0 + (x1 - x0) * i / n; i ? ctx.lineTo(X(x), Y(fn(x))) : ctx.moveTo(X(x), Y(fn(x))); }
  if (fill) { setA(a * fill[1]); ctx.save(); ctx.lineTo(X(x1), base); ctx.lineTo(X(x0), base); ctx.closePath(); ctx.fillStyle = fill[0]; ctx.fill(); ctx.restore(); ctx.beginPath(); for (let i = 0; i <= n; i++) { const x = x0 + (x1 - x0) * i / n; i ? ctx.lineTo(X(x), Y(fn(x))) : ctx.moveTo(X(x), Y(fn(x))); } }
  setA(a); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke(); setA(1); }
function ball1(X, Y, fn, x, r, col, a = 1) { const e = 0.01, sl = (Y(fn(x + e)) - Y(fn(x - e))) / (X(x + e) - X(x - e)), nx = sl / Math.hypot(1, sl), ny = -1 / Math.hypot(1, sl); marble(X(x) + nx * r, Y(fn(x)) + ny * r, r, col, a); }

// the 1-D sculpting run
const SCU = A.sculpt, NF = SCU.E.length - 1;
const sculptE = f => { const fi = clamp(f, 0, NF), i = Math.min(NF - 1, Math.floor(fi)), u = fi - i; return SCU.E[i].map((v, k) => lerp(v, SCU.E[i + 1][k], u)); };
const interp1 = (xs, ys, x) => { const n = xs.length, fi = clamp((x - xs[0]) / (xs[n - 1] - xs[0]) * (n - 1), 0, n - 1.001), i = Math.floor(fi), u = fi - i; return lerp(ys[i], ys[i + 1], u); };
const SEmin = Math.min(...SCU.E.flat()), SEmax = Math.max(...SCU.E.flat());
const trueShift = (() => { const fin = SCU.E[NF]; let s = 0, n = 0; SCU.g.forEach((x, k) => { if (Math.abs(x) < 2.9) { s += fin[k] - Efn(x); n++; } }); return s / n; })();

// Metropolis on the 1-D landscape
const MET = A.metro, metAcc = MET.filter(s => s[4]).length / MET.length;
const metState = k => (k <= 0 ? MET[0][0] : MET[Math.min(k, MET.length) - 1][4] ? MET[Math.min(k, MET.length) - 1][1] : MET[Math.min(k, MET.length) - 1][0]);

// valleys, chain, numbers
const SH = A.share, shMax = Math.max(...SH), shMin = Math.min(...SH), PB = A.probes;
const CH = A.chain, secOf = p => ((Math.round(Math.atan2(p[1], p[0]) / (Math.PI / 4)) % 8) + 8) % 8;
const CROSS = (() => { const out = [0]; for (let i = 1; i < CH.length; i++) out.push(out[i - 1] + (secOf(CH[i]) !== secOf(CH[i - 1]) && Math.hypot(...CH[i]) > 1.4 ? 1 : 0)); return out; })();

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open
SC.push({ dur: 28, chapter: 'intro',
  caps: [[0.6, 6, 'A landscape of hills and valleys. Nobody drew it: a neural network sculpted it, one number for every point.'],
         [6, 12, 'Drop marbles anywhere. Let them roll downhill — and keep giving them small random kicks.'],
         [12, 18.5, 'They settle into eight valleys: the exact shape of the data. No generator. No decoder. No invertible layers.'],
         [18.5, 27.6, 'Today: energy-based models — the loosest rules in generative modelling, and the strangest way to draw a sample.']],
  draw(t) {
    const dim = lerp(1, 0.14, ease(prog(t, 18.3, 1.4))), m = square(560, 170, 800); GA = dim;
    landscape(m, A.E, eout(prog(t, 0.3, 1.2)), prog(t, 0.5, 3));
    const f = NL * ease(prog(t, 4.2, 11)) ** 0.8;
    drawMarbles(m, A.lang, f, P.terracotta, 0.95, 4, 1e9, k => prog(t, 3 + (k % 60) * 0.012 + Math.floor(k / 60) * 0.07, 0.35));
    GA = 1;
    const a = eout(prog(t, 18.8, 1.2));
    text('EPISODE 07', 960, 430, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('Energy-Based Models', 960, 550 + (1 - a) * 16, { font: F.serif, size: 124, color: P.chalk, align: 'center', a, ls: -1 });
    text('sculpting a landscape', 960, 622, { size: 34, color: P.stone, align: 'center', a: eout(prog(t, 19.4, 1)) });
    const tg = [[P.terracotta, 'x  data'], [P.plum, 'E  energy'], [P.ink, 'p  density'], [P.amber, 'Z  normaliser'], [P.rose, 'x′  fantasies'], [P.sage, '∇E  slope']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let cx = 960 - tw / 2; tg.forEach(([c, s], i) => { cx += tag(cx, 690, c, s, eout(prog(t, 20 + i * 0.3, 0.6)), 21) + 14; });
  } });

// 01 — one number per point
const probePath = [[0.0, 0.0], [2.0, 0.0], [1.85, 0.77], [1.414, 1.414], [3.0, 3.0], [-2.0, 0.0]];
SC.push({ dur: 42, chapter: 'energy', title: 'One number per point', sub: 'The loosest possible rule for a generative model.',
  caps: [[1, 9, 'Every model so far paid for its density with a promise: a fixed order, a bound, invertible layers. The GAN gave up on density altogether.'],
         [9, 17, 'An energy-based model promises almost nothing. Take any network, any architecture, and have it output one number: the energy, E(x).'],
         [17, 25, 'Low energy means “this looks like data”. High energy means “this doesn’t”. That is the whole model: a landscape over every possible x.'],
         [25, 33, `Probe the real trained landscape, measured from its lowest point: on a blob, ${PB.blob.toFixed(1)}. Between two blobs, ${PB.between.toFixed(1)}. Out in the corner, ${PB.corner.toFixed(1)}.`],
         [33, 41.6, 'It is a judge, not an artist — like the critics of episodes 4 and 5. Scoring a drawing is far easier than drawing one. The drawing is where it gets hard.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8));
    // the promises every earlier family had to make
    const pa = a0 * (1 - ease(prog(t, 9, 1)));
    if (pa > 0) { card(100, 250, 900, 520, P.rose, pa); label('the price each family paid for p(x)', 134, 296, pa, P.rose);
      [['02  autoregressive', 'one piece at a time, in a fixed order'], ['03  VAE', 'a lower bound, not p(x) itself'], ['05  GAN', 'no p(x) at all'], ['06  flow', 'invertible layers, cheap determinants']].forEach(([nm, s], i) => {
        const ai = pa * eout(prog(t, 1.6 + i * 1.3, 0.6)), y = 380 + i * 96; text('⛓', 140, y, { size: 28, color: P.rose, a: ai }); text(nm, 190, y, { font: F.mono, size: 22, color: P.chalk, a: ai }); text(s, 470, y, { size: 26, color: P.stone, a: ai }); }); }
    // any network → one number
    const na = eout(prog(t, 9.5, 0.8));
    if (na > 0) { const y = 380; marble(170, y, 16, P.terracotta, na); text('x', 170, y + 58, { font: F.mono, size: 26, color: P.terracotta, align: 'center', a: na });
      arrow(200, y, 300, y, P.dust, 2.5, na); rbox(310, y - 90, 380, 180, 20, rgba(P.plum, 0.12), rgba(P.plum, 0.6), na, 2);
      text('any network', 500, y - 8, { font: F.serif, italic: true, size: 34, color: P.chalk, align: 'center', a: na }); text('MLP · CNN · transformer · …', 500, y + 34, { font: F.mono, size: 16, color: P.dust, align: 'center', a: na });
      arrow(700, y, 800, y, P.dust, 2.5, na); M('[m|E](x)', 815, y + 12, 38, { a: na }); text('one number', 815, y + 58, { font: F.mono, size: 16, color: P.plum, a: na });
      M('[m|E][k_|θ] : ℝ[k^|D] → ℝ', 170, 560, 32, { a: eout(prog(t, 12, 0.8)) }); text('no order · no bound · no inverse · no determinant', 170, 610, { font: F.mono, size: 18, color: P.sage, a: eout(prog(t, 13, 0.8)) });
      const la = eout(prog(t, 17, 0.8)); card(100, 660, 900, 170, P.plum, la);
      text('low energy', 140, 720, { font: F.mono, size: 22, color: P.ink, a: la }); text('→  looks like data', 320, 720, { size: 26, color: P.stone, a: la });
      text('high energy', 140, 780, { font: F.mono, size: 22, color: P.plum, a: la }); text('→  doesn’t', 320, 780, { size: 26, color: P.stone, a: la });
      const ja = eout(prog(t, 33, 0.8)); if (ja > 0) { rbox(100, 855, 900, 70, 14, rgba(P.sage, 0.08), rgba(P.sage, 0.4), ja); text('a judge, not an artist', 140, 900, { font: F.serif, italic: true, size: 30, color: P.sage, a: ja }); } }
    // the real landscape, with a probe marble
    const m = square(1150, 250, 640), la2 = eout(prog(t, 14, 1)); landscape(m, A.E, la2, prog(t, 14, 3));
    text('the real trained landscape', 1150, 922, { font: F.mono, size: 16, color: P.dust, a: la2 });
    if (t > 23) { const u = clamp((t - 24) / 13) * (probePath.length - 1), i = Math.min(probePath.length - 2, Math.floor(u)), w = ease(u - i), p = [lerp(probePath[i][0], probePath[i + 1][0], w), lerp(probePath[i][1], probePath[i + 1][1], w)];
      const pa2 = eout(prog(t, 23.5, 0.6)), X = m.X(p[0]), Y = m.Y(p[1]), e = Eraw(p[0], p[1]);
      marble(X, Y, 11, P.amber, pa2); pill(X + (p[0] > 2 ? -170 : 20), Y - 44, `E = ${e.toFixed(1)}`, P.plum, pa2, { size: 22 }); }
  } });

// 02 — from energy to probability
SC.push({ dur: 54, chapter: 'boltzmann', title: 'From energy to probability', sub: 'Three reasonable demands force a single answer.',
  caps: [[1, 7, 'A landscape is not yet a probability. We need a rule that turns each energy into a likelihood. Let’s derive it.'],
         [7, 13, 'Demand one: probabilities can’t be negative, so p must be a positive function of the energy.'],
         [13, 19, 'Demand two: lower energy should mean more likely. Downhill is where the data lives.'],
         [19, 26, 'Demand three: if x has two independent parts, their energies should add — and their probabilities must multiply.'],
         [26, 33, 'Only one function turns sums into products: the exponential. So p(x) ∝ e^−E(x). Each extra unit of energy makes x e ≈ 2.72 times less likely.'],
         [33, 40, 'Flip the landscape upside down and exponentiate: deep valleys become tall peaks. This valley is ' + Math.exp(Efn(0) - Efn(-2.1)).toFixed(0) + '× more likely than the hilltop.'],
         [40, 47, 'Last step: probabilities must add up to one. So divide by the total area under e^−E — a number called Z, the partition function.'],
         [47, 53.6, 'You have met this before: episode 2’s softmax. Its logits are negative energies, and Z was a sum over 27 letters. Same rule — now over every possible x.']],
  draw(t) {
    const X0 = 150, X1 = 1010, X = x => X0 + (x + 3.3) / 6.6 * (X1 - X0), Yt = e => 560 - (e + 0.6) * 44, a0 = eout(prog(t, 0.6, 0.8));
    axes1(X0, X1, 560, a0); curve(Efn, -3.3, 3.3, X, Yt, P.plum, 4, a0, 240, [P.plum, 0.1], 560); text('energy E(x)', X1 - 150, 290, { font: F.mono, size: 20, color: P.plum, a: a0 });
    // e^{−E} below, then shaded area = Z
    const ea = eout(prog(t, 30, 1)), pmax = Math.exp(-Efn(-2.1)), Yb = v => 900 - v / pmax * 230;
    if (ea > 0) { axes1(X0, X1, 900, ea); const za = eout(prog(t, 40, 1));
      curve(x => Math.exp(-Efn(x)), -3.3, 3.3, X, Yb, P.ink, 4, ea, 240, [za > 0 ? mixc(P.ink, P.amber, za) : P.ink, 0.12 + 0.2 * za], 900);
      text('e^−E(x)', X1 - 150, 680, { font: F.mono, size: 20, color: P.ink, a: ea });
      if (za > 0) M('area = [a|Z]', X(-0.1), 870, 26, { align: 'center', a: za }); }
    // two probes: a valley and the hilltop
    const va = eout(prog(t, 33, 0.8)) * (1 - ease(prog(t, 40, 0.8)));
    if (va > 0) { [[-2.1, P.sage], [0, P.rose]].forEach(([x, col]) => { ball1(X, Yt, Efn, x, 12, col, va); line(X(x), Yt(Efn(x)) + 14, X(x), Yb(Math.exp(-Efn(x))) - 8, col, 1.5, va * 0.6, [4, 5]); dot(X(x), Yb(Math.exp(-Efn(x))), 6, col, va); });
      pill(X(-2.1) + 24, Yb(Math.exp(-Efn(-2.1))) - 16, `${Math.exp(Efn(0) - Efn(-2.1)).toFixed(0)}× more likely`, P.sage, va, { size: 20 }); }
    // the three demands
    const dem = [[7, 'p(x) ≥ 0', 'a positive function of E'], [13, 'lower E  ⇒  higher p', 'downhill = likely'], [19, 'E₁ + E₂  ⇒  p₁ · p₂', 'independent parts: energies add']];
    const da = eout(prog(t, 6.5, 0.8)) * (1 - ease(prog(t, 46.5, 1)));
    card(1080, 250, 750, 440, P.plum, da); label('three demands', 1114, 296, da, P.plum);
    dem.forEach(([t0, f, s], i) => { const ai = da * eout(prog(t, t0, 0.7)), y = 360 + i * 84; text(`${i + 1}`, 1116, y + 4, { font: F.serif, italic: true, size: 34, color: P.amber, a: ai });
      text(f, 1160, y, { font: F.mono, size: 26, color: P.chalk, a: ai }); text(s, 1160, y + 32, { font: F.mono, size: 16, color: P.dust, a: ai }); });
    const xa = eout(prog(t, 26, 0.8)) * (1 - ease(prog(t, 46.5, 1)));
    M('e[m^|−(E₁+E₂)] = e[m^|−E₁] · e[m^|−E₂]', 1114, 640, 28, { a: xa });
    const ba = eout(prog(t, 40.5, 0.8));
    card(1080, 720, 750, 200, P.ink, ba * (1 - ease(prog(t, 46.5, 1)))); M('[p|p]([d|x]) = e[m^|−E(x)] / [a|Z]', 1114, 800, 40, { a: ba * (1 - ease(prog(t, 46.5, 1))) });
    M('[a|Z] = [pB|∫] e[m^|−E(x)] d[d|x]', 1114, 872, 28, { a: eout(prog(t, 43, 0.8)) * (1 - ease(prog(t, 46.5, 1))) });
    // callback: softmax
    const sa = eout(prog(t, 47, 0.8));
    if (sa > 0) { card(1080, 250, 750, 670, P.amber, sa); label('seen it before: episode 2', 1114, 296, sa, P.amber);
      M('softmax:   [p|p](letter) = e[a^|z] / [kB|Σ] e[a^|z]', 1114, 380, 28, { a: sa }); text('logits z = −E   ·   Z = a sum over 27 letters', 1114, 430, { font: F.mono, size: 18, color: P.dust, a: sa });
      M('temperature:   [p|p] ∝ e[m^|−E/τ]', 1114, 520, 28, { a: eout(prog(t, 49, 0.8)) }); text('the Boltzmann distribution of physics', 1114, 570, { font: F.serif, italic: true, size: 26, color: P.sage, a: eout(prog(t, 50, 0.8)) });
      M('[p|p]([d|x]) = e[m^|−E(x)] / [a|Z]', 1114, 720, 44, { a: sa }); }
  } });

// 03 — the price tag: Z
const zRows = [['letters (episode 2)', '27', 'instant', P.sage], ['this 1-D curve', '661 grid points', 'instant', P.sage], ['this episode’s 2-D data', '400 × 400 = 160,000', 'a fraction of a second', P.sage], ['one 28 × 28 image, 256 shades', '256⁷⁸⁴ ≈ 10¹⁸⁸⁸', 'never', P.rose]];
SC.push({ dur: 44, chapter: 'Z', title: 'The price tag: Z', sub: 'A sum over every possible x.',
  caps: [[1, 8, 'Z is the total of e^−E over every possible x. For 27 letters, that’s 27 terms. Easy.'],
         [8, 15, 'For a curve, add up a few hundred grid points. For this episode’s 2-D data, a 400 × 400 grid: 160,000 evaluations. Still fine.'],
         [15, 22, 'Every extra dimension multiplies the count. A 28 × 28 image with 256 shades per pixel has 256⁷⁸⁴ ≈ 10¹⁸⁸⁸ possibilities — far more than the 10⁸⁰ atoms in the universe.'],
         [22, 29, 'Worse: Z depends on the network’s weights. Every training step changes the landscape — and with it, the impossible sum.'],
         [29, 36, 'But here is a gift. Compare two points instead: p(a) / p(b). The Zs cancel. Only the energy difference remains.'],
         [36, 43.6, 'Hold on to that: ratios, and slopes, never need Z. They are how we will learn — and how we will sample.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), ra = a0 * (1 - ease(prog(t, 28.5, 1)));
    // growing grids: 1-D line, 2-D grid, 3-D cube
    const gx = 130, gy = 290;
    if (ra > 0) { const a1 = ra * eout(prog(t, 8, 0.8)), a2 = ra * eout(prog(t, 10, 0.8)), a3 = ra * eout(prog(t, 15, 0.8));
      for (let i = 0; i < 12; i++) dot(gx + i * 26, gy + 20, 4, P.amber, a1); text('1-D: n points', gx, gy + 70, { font: F.mono, size: 16, color: P.dust, a: a1 });
      for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) dot(gx + i * 26, gy + 130 + j * 26, 3.4, P.amber, a2 * 0.8); text('2-D: n² points', gx, gy + 450, { font: F.mono, size: 16, color: P.dust, a: a2 });
      for (let k = 0; k < 7; k++) for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) dot(gx + 400 + i * 30 + k * 16, gy + 190 + j * 30 - k * 16, 3, P.amber, a3 * (0.35 + 0.1 * k));
      text('3-D: n³ …   784-D: n⁷⁸⁴', gx + 400, gy + 450, { font: F.mono, size: 16, color: P.rose, a: a3 }); }
    card(820, 250, 1010, 440, P.amber, ra); label('how many terms in Z?', 854, 296, ra, P.amber);
    zRows.forEach(([nm, n, cost, col], i) => { const ai = ra * eout(prog(t, [1, 8, 10.5, 15][i], 0.7)), y = 370 + i * 80;
      text(nm, 854, y, { size: 26, color: P.chalk, a: ai }); text(n, 1270, y, { font: F.mono, size: 24, color: P.amber, a: ai }); text(cost, 1270, y + 30, { font: F.mono, size: 15, color: col, a: ai }); });
    const ua = ra * eout(prog(t, 18, 0.8)); text('atoms in the universe ≈ 10⁸⁰', 854, 675, { font: F.mono, size: 17, color: P.dust, a: ua });
    const wa = ra * eout(prog(t, 22, 0.8)); card(820, 720, 1010, 150, P.rose, wa);
    M('[a|Z][k_|θ] = [pB|∫] e[m^|−Eθ(x)] d[d|x]     changes with every step', 854, 808, 26, { a: wa });
    // the gift: ratios
    const ga = eout(prog(t, 29, 1));
    if (ga > 0) { const X0 = 150, X1 = 900, X = x => X0 + (x + 3.3) / 6.6 * (X1 - X0), Y = e => 800 - (e + 0.6) * 60;
      axes1(X0, X1, 800, ga); curve(Efn, -3.3, 3.3, X, Y, P.plum, 4, ga, 240, [P.plum, 0.1], 800);
      ball1(X, Y, Efn, -2.1, 13, P.sage, ga); ball1(X, Y, Efn, 0.9, 13, P.rose, ga);
      text('a', X(-2.1), Y(Efn(-2.1)) + 50, { font: F.mono, size: 22, color: P.sage, align: 'center', a: ga }); text('b', X(0.9), Y(Efn(0.9)) + 50, { font: F.mono, size: 22, color: P.rose, align: 'center', a: ga });
      card(1000, 330, 830, 470, P.sage, ga); label('ratios don’t need z', 1034, 376, ga, P.sage);
      M('[p|p](a) / [p|p](b) = ', 1034, 470, 32, { a: ga });
      const fa = eout(prog(t, 30.5, 0.8)); let zx = 1034; const zs = [];
      ['(e[m^|−E(a)] / ', '[a|Z]', ') / (e[m^|−E(b)] / ', '[a|Z]', ')'].forEach((q, i) => { if (i % 2) zs.push(zx); zx += M(q, zx, 540, 28, { a: fa }); });
      const ca = eout(prog(t, 32.5, 0.6)); if (ca > 0) zs.forEach(x => line(x - 3, 548, x + 20, 516, P.rose, 3, ca));
      M(`= e[m^|E(b) − E(a)] = e[m^|${(Efn(0.9) - Efn(-2.1)).toFixed(2)}] ≈ ${Math.exp(Efn(0.9) - Efn(-2.1)).toFixed(1)}`, 1034, 620, 30, { a: eout(prog(t, 33.5, 0.8)) });
      text('only the energy difference matters', 1034, 700, { font: F.serif, italic: true, size: 28, color: P.sage, a: eout(prog(t, 36, 0.8)) }); }
  } });

// 04 — learning: push down, pull up
SC.push({ dur: 56, chapter: 'push & pull', title: 'Learning: push down, pull up', sub: 'Maximum likelihood — and where the gradient of Z hides.',
  caps: [[1, 8, 'Learning is episode 1 again: make the data likely. Take the log: log p(x) = −E(x) − log Z. Energy term, minus the dreaded Z.'],
         [8, 15, 'Differentiate with respect to the weights θ. The first term is easy: backprop through the network. The second is the problem.'],
         [15, 22, 'The gradient of log Z is one over Z times the gradient of Z. And Z is an integral of e^−E, so the gradient slides inside.'],
         [22, 29, 'Look at what appears inside the integral: e^−E divided by Z. That is exactly pθ — the model’s own distribution!'],
         [29, 36, 'An integral against pθ is just an average over samples from the model. The impossible Z has turned into “ask the model to dream”.'],
         [36, 44, 'So the recipe: lower the energy under every data point. Raise it under every fantasy — every sample the model currently believes in.'],
         [44, 50, 'Digging where the data is, and filling in where the model is wrong. The landscape gets sculpted, one push at a time.'],
         [50, 55.6, 'When the fantasies land exactly where the data is, the two pushes cancel. The gradient is zero. Learning is done.']],
  draw(t) {
    const x0 = 110, a0 = eout(prog(t, 0.6, 0.8)), fade = 1 - ease(prog(t, 35.5, 1));
    const rows = [[1, 'log [p|p][k_|θ]([d|x]) = −[m|E][k_|θ]([d|x]) − log [a|Z][k_|θ]'], [8, '∇[k_|θ] log [p|p][k_|θ]([d|x]) = −∇[k_|θ][m|E][k_|θ]([d|x]) − ∇[k_|θ] log [a|Z][k_|θ]'],
      [15, '∇[k_|θ] log [a|Z] = (1/[a|Z]) ∇[k_|θ][a|Z] = (1/[a|Z]) [pB|∫] ∇[k_|θ] e[m^|−E(x′)] d[n|x′]'], [22, '             = [pB|∫] (e[m^|−E(x′)] / [a|Z]) · (−∇_θ[m|E](x′)) d[n|x′]'], [29, '             = − [kB|E][k_|x′∼pθ] [ ∇[k_|θ][m|E]([n|x′]) ]']];
    let hx0 = 0, hw = 0;
    rows.forEach(([t0, s], i) => { const ai = a0 * fade * eout(prog(t, t0, 0.8)), y = 320 + i * 82;
      if (i !== 3) return M(s, x0, y, 28, { a: ai });
      let x = x0; x += M('             = [pB|∫] ', x, y, 28, { a: ai }); hx0 = x; hw = M('(e[m^|−E(x′)] / [a|Z])', x, y, 28, { a: ai }); M(' · (−∇[k_|θ][m|E](x′)) d[n|x′]', x + hw, y, 28, { a: ai }); });
    const ha = eout(prog(t, 23.5, 0.6)) * fade; if (ha > 0) { rbox(hx0 - 8, 320 + 3 * 82 - 38, hw + 16, 58, 12, rgba(P.ink, 0.12), P.ink, ha, 2); pill(hx0 + hw / 2, 320 + 3 * 82 + 52, '= pθ(x′): the model’s own distribution', P.ink, ha, { size: 20, align: 'center' }); }
    const la = eout(prog(t, 30.5, 0.6)) * fade; if (la > 0) text('an average over the model’s own samples: its “fantasies”', x0, 720, { font: F.serif, italic: true, size: 28, color: P.sage, a: la });
    // the result
    const ra = eout(prog(t, 36, 0.9));
    if (ra > 0) { card(100, 270, 1000, 330, P.plum, ra); label('the learning rule', 134, 316, ra, P.plum);
      M('−∇[k_|θ] log [p|p][k_|θ]([d|x]) = ∇[k_|θ][m|E]([d|x]) − [kB|E][k_|x′∼pθ] ∇[k_|θ][m|E]([n|x′])', 134, 400, 28, { a: ra });
      text('push energy DOWN at data', 300, 470, { font: F.mono, size: 20, color: P.terracotta, align: 'center', a: eout(prog(t, 37, 0.6)) });
      text('push energy UP at fantasies', 790, 470, { font: F.mono, size: 20, color: P.rose, align: 'center', a: eout(prog(t, 38, 0.6)) });
      text('(gradient descent on θ moves E against these)', 134, 550, { font: F.mono, size: 15, color: P.dust, a: eout(prog(t, 39, 0.6)) }); }
    // illustration: a real landscape mid-training (1-D), with the pushes
    const ia = eout(prog(t, 40, 1));
    if (ia > 0) { const X0 = 150, X1 = 1060, X = x => X0 + (x + 3.3) / 6.6 * (X1 - X0), fr = t < 49 ? 6 : lerp(6, NF, ease(prog(t, 49, 5))), Eg = sculptE(fr), fn = x => interp1(SCU.g, Eg, x), Y = e => 790 - (e - SEmin) * 26;
      curve(fn, -3.3, 3.3, X, Y, P.plum, 4, ia, 200, [P.plum, 0.1], 900);
      const aa = ia * (t < 49 ? 1 : 1 - 0.6 * ease(prog(t, 49, 5)));
      SCU.data.slice(0, 26).forEach(x => arrow(X(x), Y(fn(x)) - 40, X(x), Y(fn(x)) - 8, P.terracotta, 2, aa));
      SCU.fant[Math.round(fr)].slice(0, 14).forEach(x => arrow(X(x), Y(fn(x)) + 40, X(x), Y(fn(x)) + 8, P.rose, 2, aa));
      text(t < 49 ? 'real 1-D run, early on' : 'the same run, later: pushes balance', X0, 925, { font: F.mono, size: 16, color: P.dust, a: ia }); }
    const ca = eout(prog(t, 44, 0.8));
    if (ca > 0) { card(1160, 270, 670, 650, P.sage, ca); label('what each push does', 1194, 316, ca, P.sage);
      [[P.terracotta, '▼  data', 'dig a valley where real examples live'], [P.rose, '▲  fantasies', 'raise the ground where the model dreams but no data lives'], [P.sage, '=  balance', 'fantasies look like data → pushes cancel → done']].forEach(([c, h, s], i) => {
        const ai = ca * eout(prog(t, 44.5 + i * 2.5 + (i === 2 ? 1.5 : 0), 0.7)); text(h, 1194, 400 + i * 170, { font: F.mono, size: 26, color: c, a: ai }); para(s, 1194, 440 + i * 170, 600, { size: 24, a: ai }); }); }
  } });

// 05 — watch a landscape get sculpted (real 1-D run)
const scF = t => NF * Math.pow(prog(t, 4, 35), 1.7);
SC.push({ dur: 48, chapter: 'sculpt', title: 'Watch a landscape get sculpted', sub: `Real training in 1-D: ${SCU.data.length} data points, a landscape made of 25 bumps.`,
  caps: [[1, 8, `Here is that rule running for real. ${SCU.data.length} data points, drawn from a hidden landscape. The model starts perfectly flat: every x equally likely.`],
         [8, 16, 'Each step: dig down under every data point, and raise the ground under the model’s fantasies — fresh samples from its current landscape.'],
         [16, 24, 'At first the fantasies are everywhere, so the ground rises everywhere except where the data sits. Valleys appear.'],
         [24, 32, 'As the valleys deepen, the fantasies fall into them too — and the two pushes start to cancel.'],
         [32, 40, 'When the fantasies are spread exactly like the data, digging and raising balance out. The landscape stops changing.'],
         [40, 47.6, 'Dashed: the hidden landscape the data came from. The model rebuilt it from samples alone — up to a constant shift, which Z absorbs.']],
  draw(t) {
    const X0 = 160, X1 = 1500, X = x => X0 + (x + 3.3) / 6.6 * (X1 - X0), Y = e => 700 - (e - SEmin) / (SEmax - SEmin) * 400, a0 = eout(prog(t, 0.6, 0.8));
    const f = scF(t), Eg = sculptE(f), fn = x => interp1(SCU.g, Eg, x), fi = Math.min(NF, Math.round(f));
    line(X0, Y(0), X1, Y(0), P.border, 1.2, a0, [5, 6]); text('E = 0', X1 + 12, Y(0) + 6, { font: F.mono, size: 15, color: P.dust, a: a0 });
    curve(fn, -3.3, 3.3, X, Y, P.plum, 4, a0, 240, [P.plum, 0.1], 760);
    // rugs: the data, and this step's fantasies
    const ry = 800; text('data', X0 - 20, ry + 6, { font: F.mono, size: 16, color: P.terracotta, align: 'right', a: a0 }); text('fantasies', X0 - 20, ry + 56, { font: F.mono, size: 16, color: P.rose, align: 'right', a: eout(prog(t, 7, 0.8)) });
    SCU.data.forEach(x => line(X(x), ry - 12, X(x), ry + 12, P.terracotta, 2, a0 * 0.8));
    const fa = eout(prog(t, 7, 0.8)); SCU.fant[fi].forEach(x => line(X(x), ry + 38, X(x), ry + 62, P.rose, 2, fa * 0.9));
    // pushes on the curve
    const pa = eout(prog(t, 8, 0.8)) * (1 - 0.55 * ease(prog(t, 33, 4)));
    SCU.data.filter((_, i) => i % 4 === 0).forEach(x => arrow(X(x), Y(fn(x)) - 36, X(x), Y(fn(x)) - 7, P.terracotta, 2, pa));
    SCU.fant[fi].slice(0, 16).forEach(x => arrow(X(x), Y(fn(x)) + 36, X(x), Y(fn(x)) + 7, P.rose, 2, pa * fa));
    // readouts
    const ca = eout(prog(t, 2, 0.8)); card(1580, 260, 250, 360, P.plum, ca); label('training', 1606, 306, ca, P.plum);
    text(`step ${fi} / ${NF}`, 1606, 360, { font: F.mono, size: 22, color: P.chalk, a: ca });
    const Ed = SCU.data.reduce((s, x) => s + fn(x), 0) / SCU.data.length, Ef = SCU.fant[fi].reduce((s, x) => s + fn(x), 0) / SCU.fant[fi].length;
    text('mean E, data', 1606, 420, { font: F.mono, size: 15, color: P.terracotta, a: ca }); text(fmt2(Ed), 1606, 452, { font: F.mono, size: 26, color: P.terracotta, a: ca });
    text('mean E, fantasies', 1606, 500, { font: F.mono, size: 15, color: P.rose, a: ca }); text(fmt2(Ef), 1606, 532, { font: F.mono, size: 26, color: P.rose, a: ca });
    text(`gap ${(Ef - Ed).toFixed(2)}`, 1606, 590, { font: F.mono, size: 18, color: Math.abs(Ef - Ed) < 0.15 ? P.sage : P.amber, a: ca });
    const ta = eout(prog(t, 40, 1)); if (ta > 0) {
      setA(ta); ctx.setLineDash([8, 8]); ctx.strokeStyle = P.sage; ctx.lineWidth = 2.5; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -2.9 + 5.8 * i / 200; i ? ctx.lineTo(X(x), Y(Efn(x) + trueShift)) : ctx.moveTo(X(x), Y(Efn(x) + trueShift)); } ctx.stroke(); ctx.setLineDash([]); setA(1);
      pill(X(0.1), Y(Efn(0.1) + trueShift) - 44, 'the hidden landscape', P.sage, ta, { align: 'center', size: 18 }); }
  } });

// 06 — Metropolis: sampling with ratios
const MSLOW = 12, mT0 = 9, mDT = 1.45, kAt = t => (t < mT0 ? 0 : t < mT0 + MSLOW * mDT ? (t - mT0) / mDT : MSLOW + (MET.length - MSLOW) * Math.pow(prog(t, mT0 + MSLOW * mDT, 16), 2));
SC.push({ dur: 54, chapter: 'metropolis', title: 'Sampling with ratios', sub: 'Propose a step. Flip a biased coin. No Z required.',
  caps: [[1, 9, 'We can’t normalise. But we can compare. That is enough to sample, with a trick from 1953.'],
         [9, 16, 'From where the marble sits, propose a random step. If it goes downhill, always take it.'],
         [16, 23, 'If it goes uphill by ΔE, take it only with probability e^−ΔE: flip a coin weighted by the ratio of the two probabilities. Z cancels.'],
         [23, 31, 'Accepted, rejected, accepted… Mostly the marble lingers in valleys, but now and then it climbs a hill.'],
         [31, 39, 'Speed it up and keep a record of every place it has been. The histogram grows toward e^−E / Z — the true distribution, with no Z ever computed.'],
         [39, 46, `${MET.length.toLocaleString('en-US')} steps, ${(metAcc * 100).toFixed(0)}% accepted. Metropolis and colleagues invented this at Los Alamos, to simulate atoms on one of the first computers.`],
         [46, 53.6, 'But it guesses blindly. In high dimensions almost every random step goes uphill — the steps must shrink like 1/√D. In 784 dimensions, a crawl.']],
  draw(t) {
    const X0 = 150, X1 = 1060, X = x => X0 + (x + 3.3) / 6.6 * (X1 - X0), Y = e => 560 - (e + 0.6) * 44, a0 = eout(prog(t, 0.6, 0.8));
    axes1(X0, X1, 560, a0); curve(Efn, -3.3, 3.3, X, Y, P.plum, 4, a0, 240, [P.plum, 0.1], 560);
    const kf = kAt(t), k = Math.floor(kf);
    // histogram of visited states vs e^−E/Z
    const ha = eout(prog(t, 30, 1)), NB = 44, bw = 6.6 / NB, hb = 910, sc = 230 / (Math.exp(-Efn(-2.1)) / Z1);
    if (ha > 0 && k > 0) { const cnt = new Array(NB).fill(0), kk = Math.min(k, MET.length); for (let i = 1; i <= kk; i++) cnt[clamp(Math.floor((metState(i) + 3.3) / bw), 0, NB - 1)]++;
      cnt.forEach((c, i) => { const h = Math.min(300, c / kk / bw * sc); setA(ha * 0.55); ctx.fillStyle = P.amber; ctx.fillRect(X(-3.3 + i * bw) + 1, hb - h, X(-3.3 + bw) - X(-3.3) - 2, h); setA(1); });
      curve(x => Math.exp(-Efn(x)) / Z1, -3.3, 3.3, X, v => hb - v * sc, P.ink, 3, ha, 200); line(X0, hb, X1, hb, P.borderLight, 1.5, ha);
      text(`visited positions (${Math.min(k, MET.length)} steps)   ·   line: e^−E / Z`, X0, hb + 34, { font: F.mono, size: 16, color: P.dust, a: ha }); }
    // the marble and the proposal
    if (t >= mT0 - 1) { const cur = metState(k), ma = eout(prog(t, mT0 - 1, 0.6));
      if (k < MSLOW && t >= mT0) { const [x, prop, dE, u, acc] = MET[k], ph = kf - k;
        const ga = eout(clamp(ph / 0.15)) * (acc ? 1 - clamp((ph - 0.7) / 0.1) : 1 - clamp((ph - 0.75) / 0.2));
        const mv = acc ? ease(clamp((ph - 0.55) / 0.3)) : 0, xm = lerp(x, prop, mv);
        if (ga > 0) { ball1(X, Y, Efn, prop, 13, acc ? P.sage : P.rose, ga * 0.45); arrow(X(x), Y(Efn(x)) - 44, X(prop), Y(Efn(prop)) - 44, P.dust, 2, ga);
          const ta = eout(clamp((ph - 0.18) / 0.15)) * ga, pAcc = Math.min(1, Math.exp(-dE));
          text(`ΔE = ${(dE >= 0 ? '+' : '−') + Math.abs(dE).toFixed(2)}`, X(prop), Y(Efn(prop)) - 70, { font: F.mono, size: 18, color: dE <= 0 ? P.sage : P.amber, align: 'center', a: ta });
          text(dE <= 0 ? 'downhill: go' : `coin ${u.toFixed(2)} ${u < pAcc ? '<' : '>'} e^−ΔE = ${pAcc.toFixed(2)}`, X(prop), Y(Efn(prop)) - 96, { font: F.mono, size: 16, color: P.stone, align: 'center', a: ta });
          const va = eout(clamp((ph - 0.42) / 0.12)) * ga; text(acc ? '✓' : '✗', X(prop) + 26, Y(Efn(prop)) + 8, { font: F.mono, size: 30, color: acc ? P.sage : P.rose, a: va }); }
        ball1(X, Y, Efn, xm, 14, P.terracotta, ma);
      } else ball1(X, Y, Efn, cur, 14, P.terracotta, ma); }
    // rule cards
    const c1 = eout(prog(t, 1, 0.8)) * (1 - ease(prog(t, 38.5, 1)));
    card(1120, 250, 710, 300, P.sage, c1); label('the ratio trick', 1154, 296, c1, P.sage);
    M('[p|p]([n|x′]) / [p|p]([d|x]) = e[m^|−(E(x′) − E(x))] = e[m^|−ΔE]', 1154, 370, 26, { a: c1 }); text('Z cancels', 1154, 414, { font: F.mono, size: 16, color: P.sage, a: c1 });
    text('downhill (ΔE ≤ 0):  always move', 1154, 470, { font: F.mono, size: 20, color: P.chalk, a: c1 * eout(prog(t, 9, 0.7)) });
    text('uphill:  move with probability e^−ΔE', 1154, 512, { font: F.mono, size: 20, color: P.chalk, a: c1 * eout(prog(t, 16, 0.7)) });
    const c2 = eout(prog(t, 23, 0.8)) * (1 - ease(prog(t, 38.5, 1)));
    card(1120, 580, 710, 150, P.amber, c2); text(`step ${Math.min(k, MET.length).toLocaleString('en-US')}`, 1154, 640, { font: F.mono, size: 24, color: P.chalk, a: c2 });
    const acc = k > 0 ? MET.slice(0, Math.min(k, MET.length)).filter(s => s[4]).length / Math.min(k, MET.length) : 0; text(`accepted ${(acc * 100).toFixed(0)}%`, 1154, 688, { font: F.mono, size: 20, color: P.amber, a: c2 });
    const c3 = eout(prog(t, 39, 0.8));
    if (c3 > 0) { card(1120, 250, 710, 300, P.plum, c3); label('metropolis et al., 1953', 1154, 296, c3, P.plum);
      para('Invented at Los Alamos to simulate the atoms of a liquid on the MANIAC computer. Still one of the most used algorithms in science.', 1154, 350, 640, { size: 24, a: c3 });
      const c4 = eout(prog(t, 46, 0.8)); card(1120, 580, 710, 330, P.rose, c4); label('the catch: it guesses blindly', 1154, 626, c4, P.rose);
      M('step size ∝ 1 / √[a|D]', 1154, 700, 30, { a: c4 }); text('D = 784  →  steps ~28× shorter', 1154, 748, { font: F.mono, size: 20, color: P.chalk, a: c4 });
      para('Random proposals ignore where the valleys are. But our landscape has a slope…', 1154, 800, 640, { size: 22, a: eout(prog(t, 48, 0.8)) }); }
  } });

// 07 — use the slope: gradient descent vs Langevin
SC.push({ dur: 56, chapter: 'langevin', title: 'Use the slope', sub: 'Roll downhill — and keep jiggling.',
  caps: [[1, 8, 'Our landscape is a neural network, so backprop gives us its slope, ∇E(x), for free — at any point, in any number of dimensions.'],
         [8, 15, 'Obvious idea: roll every marble downhill. x ← x − η∇E(x). Let’s drop the same 420 marbles on the real landscape.'],
         [15, 22, 'They all end up at the very bottoms. A handful of points, not a distribution: that is optimisation, not sampling.'],
         [22, 29, 'Now add a small random kick at every step. Same marbles, same slope — plus a jiggle of size √(2η).'],
         [29, 37, 'The kicks stop the marbles collapsing. They spread across each valley and settle into its shape — the shape of e^−E.'],
         [37, 45, `This is Langevin dynamics: ${NL * 4} steps with η = ${A.eta}. Paul Langevin wrote it in 1908 for a pollen grain in water: a pull, plus random molecular kicks.`],
         [45, 55.6, 'Why this size of kick? Why does slope plus jiggle land on exactly e^−E — no Z anywhere? Hold that thought. It is the heart of episode 8.']],
  draw(t) {
    const L = square(90, 250, 620), Rm = square(740, 250, 620), a0 = eout(prog(t, 0.6, 0.8));
    landscape(L, A.E, a0); const ra = eout(prog(t, 21, 1)); landscape(Rm, A.E, ra);
    const fg = NL * Math.pow(prog(t, 9, 11), 0.75), fl = NL * Math.pow(prog(t, 23, 14), 0.75);
    drawMarbles(L, A.gd, fg, P.terracotta, 0.95, 3.2, 1e9, t < 9 ? k => prog(t, 8 + (k % 40) * 0.02, 0.3) : null);
    if (ra > 0) drawMarbles(Rm, A.lang, fl, P.terracotta, 0.95, 3.2, 1e9, t < 23 ? k => prog(t, 22 + (k % 40) * 0.02, 0.3) : null);
    text('roll downhill (no jiggle)', 90, 910, { font: F.mono, size: 18, color: P.chalk, a: a0 }); text(`step ${Math.round(fg * 4)}`, 710, 910, { font: F.mono, size: 16, color: P.dust, align: 'right', a: a0 * eout(prog(t, 9, 0.5)) });
    text('roll downhill + jiggle', 740, 910, { font: F.mono, size: 18, color: P.chalk, a: ra }); text(`step ${Math.round(fl * 4)}`, 1360, 910, { font: F.mono, size: 16, color: P.dust, align: 'right', a: ra * eout(prog(t, 23, 0.5)) });
    const oa = eout(prog(t, 16, 0.8)) * (1 - ease(prog(t, 30, 1))); if (oa > 0) pill(L.X(0) , L.Y(0) - 20, 'collapsed to the minima', P.rose, oa, { align: 'center', size: 20 });
    const c1 = eout(prog(t, 1, 0.8)); card(1400, 250, 430, 250, P.sage, c1); label('the slope is free', 1428, 296, c1, P.sage);
    M('∇ₓ[m|E]([d|x])', 1428, 364, 30, { a: c1 }); para('backprop through the network — with respect to x instead of θ', 1428, 410, 380, { size: 20, a: c1 });
    const c2 = eout(prog(t, 8, 0.8)); card(1400, 530, 430, 150, P.rose, c2); label('gradient descent', 1428, 576, c2, P.rose);
    M('[d|x] ← [d|x] − η ∇[m|E]', 1428, 636, 26, { a: c2 });
    const c3 = eout(prog(t, 22, 0.8)); card(1400, 710, 430, 200, P.sage, c3); label('langevin dynamics', 1428, 756, c3, P.sage);
    M('[d|x] ← [d|x] − η ∇[m|E] + √(2η) [n|ε]', 1428, 816, 24, { a: c3 }); text('ε ∼ N(0, I): a fresh kick each step', 1428, 860, { font: F.mono, size: 15, color: P.dust, a: c3 });
    const qa = eout(prog(t, 45, 0.8)); if (qa > 0) pill(1615, 950 - 30, 'why √(2η)?  → episode 8', P.amber, qa, { align: 'center', size: 20 });
  } });

// 08 — sculpting in 2-D, for real
const NS = A.train.length - 1, trF = t => NS * ease(prog(t, 6, 38));
SC.push({ dur: 56, chapter: 'training', title: 'Sculpting in 2-D, for real', sub: 'Contrastive divergence: Langevin fantasies from a replay buffer.',
  caps: [[1, 8, 'Now the real thing in two dimensions: a small network sculpting a landscape for the eight blobs that gave the flow so much trouble.'],
         [8, 16, `Every step: take a batch of fantasies from a buffer, let them roll for ${A.k} Langevin steps, then push down on data and up on fantasies.`],
         [16, 24, 'At first the fantasies are scattered everywhere, so the ground rises everywhere else. Eight valleys begin to open.'],
         [24, 32, 'Fantasies that wander between the blobs raise ridges there. Watch them grow into walls.'],
         [32, 40, 'The two energies chase each other: once fantasies sit as low as the data, the pushes cancel.'],
         [40, 47, 'This is contrastive divergence — Hinton’s recipe from 2002 — in its modern form: Langevin fantasies kept in a replay buffer (Du & Mordatch, 2019).'],
         [47, 55.6, 'One small extra: a gentle penalty on E² stops the energies running off to infinity. Training an EBM is a balancing act.']],
  draw(t) {
    const m = square(110, 250, 620), a0 = eout(prog(t, 0.6, 0.8)), f = trF(t), i = Math.min(NS - 1, Math.floor(f)), u = ease(f - i), S0 = A.train[i], S1 = A.train[i + 1];
    frame(m, a0); drawMap(m, landCanvas(S0.E), S0.E.n, a0); drawMap(m, landCanvas(S1.E), S1.E.n, a0 * u); drawContours(m, u < 0.5 ? S0.E : S1.E, a0 * (1 - 2 * Math.min(u, 1 - u)) * 0.9);
    const sn = u < 0.5 ? S0 : S1, pa = a0 * (1 - 2 * Math.min(u, 1 - u) * 0.8);
    clipTo(m, () => { sn.data.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.6, P.terracotta, pa * 0.85)); sn.fake.forEach(p => marble(m.X(p[0]), m.Y(p[1]), 3.4, P.rose, pa * eout(prog(t, 8, 0.8)))); });
    text(`training step ${sn.step}`, 110, 910, { font: F.mono, size: 20, color: P.chalk, a: a0 });
    tag(420, 890, P.terracotta, 'data', a0, 16); tag(520, 890, P.rose, 'fantasies', eout(prog(t, 8, 0.8)), 16);
    // the algorithm
    const c1 = eout(prog(t, 8, 0.8)); card(850, 250, 980, 300, P.plum, c1); label('one training step', 884, 296, c1, P.plum);
    [['1', `fantasies ← ${A.k} Langevin steps from the replay buffer`], ['2', 'loss = mean E(data) − mean E(fantasies)'], ['3', '      + α · (E(data)² + E(fantasies)²)'], ['4', 'backprop, step the weights']].forEach(([n, s], k) => {
      const ai = c1 * eout(prog(t, 8.5 + k * 1.2 + (k === 2 ? 38 : 0), 0.6)); if (k !== 2) text(n, 884, 360 + k * 50, { font: F.serif, italic: true, size: 26, color: P.amber, a: ai }); text(s, 924, 360 + k * 50, { font: F.mono, size: 20, color: k === 2 ? P.amber : P.chalk, a: ai }); });
    // mean energy on data vs fantasies, over training
    const ca = eout(prog(t, 28, 0.8));
    if (ca > 0) { const H = A.hist, x0 = 884, y0 = 880, w = 900, h = 240, rest = H.slice(4).flat(), lo = Math.min(...rest) - 0.3, hi = Math.max(...rest) + 0.3;
      const N = Math.max(2, Math.min(H.length, Math.round(H.length * (lerp(S0.step, S1.step, u) / A.steps))));
      card(850, 580, 980, 350, P.ink, ca); text('mean energy during training (real run)', x0, 624, { font: F.mono, size: 16, color: P.dust, a: ca });
      [[0, P.terracotta, 'data'], [1, P.rose, 'fantasies']].forEach(([j, col, nm]) => { setA(ca); ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.beginPath();
        H.slice(0, N).forEach((v, q) => { const X = x0 + q / (H.length - 1) * w, Y = y0 - clamp((v[j] - lo) / (hi - lo)) * h; q ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
        const last = H[N - 1][j]; text(nm, x0 + (N - 1) / (H.length - 1) * w + 10, y0 - clamp((last - lo) / (hi - lo)) * h + (j ? 18 : -8), { font: F.mono, size: 15, color: col, a: ca }); }); }
    const ha = eout(prog(t, 40, 0.8)); if (ha > 0) pill(1340, 250 - 26, 'contrastive divergence · Hinton, 2002', P.plum, ha, { align: 'center', size: 18 });
  } });

// 09 — the result
SC.push({ dur: 42, chapter: 'result', title: 'Eight valleys, no bridges', sub: 'What an energy can do that rubber can’t.',
  caps: [[1, 8, 'The sculpted landscape’s density, e^−E / Z — computed exactly, because in 2-D we can afford to sum Z over a 400 × 400 grid.'],
         [8, 16, 'Next to it, episode 6’s flow on the same data: thin bridges, because rubber can’t tear. The energy simply raises a mountain between valleys.'],
         [16, 24, 'And Langevin marbles dropped from pure noise land in all eight valleys, evenly, with nothing on the ridges.'],
         [24, 32, `Yet its exact score is ${A.nll.toFixed(2)} nats per point, against the flow’s ${A.nllFlow.toFixed(2)}. A perfect model would score ${A.perfect.toFixed(2)}.`],
         [32, 41.6, 'Look closely at the valleys: they are not equally deep. Something went wrong — and it is the deepest problem with energy-based models.']],
  draw(t) {
    const L = square(110, 250, 620), Rm = square(1190, 250, 620), a0 = eout(prog(t, 0.6, 0.8));
    frame(L, a0); drawMap(L, densCanvas(A.dens, P.ink), A.dens.n, a0);
    const la = eout(prog(t, 16, 1)) * (1 - 0.75 * ease(prog(t, 31, 1.5))); if (la > 0) drawMarbles(L, A.lang, NL, P.terracotta, la * 0.8, 2.4, 1e9, k => prog(t, 16 + (k % 60) * 0.02, 0.4));
    text('energy-based model · e^−E / Z', 110, 910, { font: F.mono, size: 18, color: P.chalk, a: a0 });
    const ra = eout(prog(t, 8, 1)); frame(Rm, ra); drawMap(Rm, densCanvas(A.flowRing, P.ink), A.flowRing.n, ra);
    text('normalizing flow (episode 6)', 1190, 910, { font: F.mono, size: 18, color: P.chalk, a: ra });
    const ba = eout(prog(t, 10, 0.8)) * (1 - ease(prog(t, 24, 1))); if (ba > 0) { ring(Rm.X(0.6), Rm.Y(-1.85), 34, P.rose, ba, 2.5); pill(Rm.X(0.6) + 44, Rm.Y(-1.85) + 50, 'a bridge', P.rose, ba); }
    const na = eout(prog(t, 24, 0.8));
    if (na > 0) { card(770, 330, 380, 400, P.amber, na); label('held-out −log p', 800, 376, na, P.amber);
      [[`${A.nll.toFixed(2)}`, 'EBM (exact, 2-D grid)', P.plum], [`${A.nllFlow.toFixed(2)}`, 'flow', P.ink], [`${A.perfect.toFixed(2)}`, 'perfect model', P.sage], [`${A.gauss.toFixed(2)}`, 'one bell curve', P.dust]].forEach(([v, s, c], i) => {
        const ai = na * eout(prog(t, 24.5 + i * 0.8, 0.5)); text(v, 800, 450 + i * 72, { font: F.mono, size: 34, color: c, a: ai }); text(s, 920, 446 + i * 72, { font: F.mono, size: 16, color: P.stone, a: ai }); });
      text('nats per point · lower is better', 800, 710, { font: F.mono, size: 13, color: P.dust, a: na }); }
    const qa = eout(prog(t, 32, 0.8)); if (qa > 0) { [0, 2, 4, 6].forEach(j => ring(L.X(2 * Math.cos(j * Math.PI / 4)), L.Y(2 * Math.sin(j * Math.PI / 4)), 30, P.amber, qa, 2)); [1, 3, 5, 7].forEach(j => ring(L.X(2 * Math.cos(j * Math.PI / 4)), L.Y(2 * Math.sin(j * Math.PI / 4)), 30, P.rose, qa, 2, [4, 4])); }
  } });

// 10 — the catch: marbles get stuck
const chT = t => clamp((t - 14) / 18) ** 1.4 * (CH.length - 1);
SC.push({ dur: 54, chapter: 'catch', title: 'The catch: marbles get stuck', sub: 'A sampler that rarely crosses ridges can’t compare valleys.',
  caps: [[1, 8, `Measure how much probability each valley really holds: ${(shMax * 100).toFixed(1)}% in the deepest, ${(shMin * 100).toFixed(1)}% in the shallowest. The data puts 12.5% in each.`],
         [8, 15, 'But the training fantasies looked perfect. Why? Each one rolls into whichever valley is nearest — and stays there.'],
         [15, 24, `Follow a single marble for ${A.chainSteps.toLocaleString('en-US')} Langevin steps. It crosses a ridge just ${A.cross === 1 ? 'once' : A.cross + ' times'}.`],
         [24, 32, 'So training never compares one valley with another. The fantasies can’t tell it that some valleys are too deep. Moving between modes — mixing — is the hard part of sampling.'],
         [32, 40, 'Second catch: cost. Every sample takes hundreds of network evaluations. A GAN or a flow needs one.'],
         [40, 47, 'Third: in high dimensions Z stays unknown, so there is no exact likelihood to report. Here, in 2-D, we cheated with a grid.'],
         [47, 53.6, 'Flexible and elegant — but hard to train, and slow to sample. Yet the idea is hiding everywhere.']],
  draw(t) {
    const m = square(110, 250, 660), a0 = eout(prog(t, 0.6, 0.8)); landscape(m, A.E, a0);
    // each valley's share of probability
    const sa = eout(prog(t, 1.5, 0.8)) * (1 - 0.7 * ease(prog(t, 14, 1)));
    SH.forEach((s, j) => { const x = 2 * Math.cos(j * Math.PI / 4), y = 2 * Math.sin(j * Math.PI / 4), ai = sa * eout(prog(t, 1.8 + j * 0.25, 0.5)), col = s > 0.14 ? P.amber : s < 0.11 ? P.rose : P.sage;
      pill(m.X(x * 1.33), m.Y(y * 1.33) + 8, `${(s * 100).toFixed(1)}%`, col, ai, { align: 'center', size: 18 }); });
    // one long chain
    const ca = eout(prog(t, 14, 0.8)) * (1 - ease(prog(t, 32, 1)));
    if (ca > 0) { const f = chT(t), n = Math.floor(f); setA(ca * 0.55); ctx.strokeStyle = P.amber; ctx.lineWidth = 1.2; ctx.beginPath(); for (let i = Math.max(0, n - 400); i <= n; i++) { const p = CH[i]; i > Math.max(0, n - 400) ? ctx.lineTo(m.X(p[0]), m.Y(p[1])) : ctx.moveTo(m.X(p[0]), m.Y(p[1])); } ctx.stroke(); setA(1);
      const p = CH[n], q = CH[Math.min(CH.length - 1, n + 1)], u = f - n; marble(m.X(lerp(p[0], q[0], u)), m.Y(lerp(p[1], q[1], u)), 8, P.terracotta, ca);
      card(850, 250, 980, 200, P.amber, ca); label('one marble, one long run', 884, 296, ca, P.amber);
      text(`step ${(Math.round(n / (CH.length - 1) * A.chainSteps)).toLocaleString('en-US')}`, 884, 360, { font: F.mono, size: 28, color: P.chalk, a: ca });
      text(`ridges crossed: ${CROSS[n]}`, 1300, 360, { font: F.mono, size: 28, color: P.amber, a: ca });
      text('valleys visited: ' + new Set(CH.slice(0, n + 1).filter(p => Math.hypot(...p) > 1.4).map(secOf)).size + ' of 8', 884, 410, { font: F.mono, size: 18, color: P.stone, a: ca }); }
    const ma = eout(prog(t, 8, 0.8)) * (1 - ease(prog(t, 12.8, 1)));
    if (ma > 0) { card(850, 250, 980, 300, P.rose, ma); label('why training didn’t notice', 884, 296, ma, P.rose); para('Short chains start from noise, roll into the nearest valley and stay. Every valley gets its fantasies, so each one looks “right” — but no fantasy ever compares two valleys.', 884, 350, 900, { size: 26, a: ma }); }
    const xa = eout(prog(t, 24, 0.8)) * (1 - ease(prog(t, 32, 1)));
    if (xa > 0) { card(850, 480, 980, 180, P.plum, xa); label('mixing', 884, 526, xa, P.plum); para('Crossing a ridge of height ΔE takes about e^ΔE tries. High walls = marbles trapped.', 884, 576, 900, { size: 26, a: xa }); }
    // the other catches
    const k2 = eout(prog(t, 32, 0.8));
    if (k2 > 0) { [['slow sampling', `${NL * 4} network evaluations per sample (GAN or flow: 1)`, P.rose, 32], ['no likelihood in high-D', 'Z is still an impossible sum; here, 2-D let us cheat', P.rose, 40], ['finicky training', 'step sizes, chain lengths, energy penalties…', P.rose, 42.5]].forEach(([h, s, c, t0], i) => {
      const ai = eout(prog(t, t0, 0.8)); card(850, 250 + i * 215, 980, 190, c, ai); label(h, 884, 296 + i * 215, ai, c); para(s, 884, 350 + i * 215, 900, { size: 26, a: ai }); }); }
  } });

// 11 — energies in disguise
SC.push({ dur: 42, chapter: 'in the wild', title: 'Energies in disguise', sub: 'Once you see them, you see them everywhere.',
  caps: [[1, 9, 'Every classifier is an energy-based model over its labels: logits are negative energies, and the softmax denominator is Z — a sum over classes.'],
         [9, 17, 'In 2019, “JEM” turned that around: the same logits also define an energy over images, so one network can classify and generate.'],
         [17, 26, 'The idea is old. Hopfield networks (1982) stored memories as valleys in an energy; Boltzmann machines (1985) learned them. In 2024, Hopfield and Hinton won the Nobel Prize in Physics.'],
         [26, 34, 'Contrastive learning, like CLIP, scores matching image–caption pairs with a softmax over the batch: the other captions play the fantasies.'],
         [34, 41.6, 'And every token a language model writes comes from a softmax over its vocabulary: an energy-based model where Z has only about 100,000 terms.']],
  draw(t) {
    const cards = [[1, 'every classifier', P.ink, '[p|p](y | [d|x]) = e[a^|f_y(x)] / [kB|Σ][k_|y′] e[a^|f_y′(x)]', 'logits = −energies · Z = sum over classes'],
      [9, 'JEM (2019)', P.plum, '[m|E]([d|x]) = −log [kB|Σ][k_|y] e[a^|f_y(x)]', 'the same network: a classifier and a generator'],
      [17, 'hopfield · boltzmann', P.amber, 'memories = valleys of E', '1982 · 1985 · Nobel Prize in Physics 2024'],
      [26, 'contrastive learning', P.sage, 'softmax over the batch', 'other examples in the batch = the fantasies'],
      [34, 'next-token prediction', P.terracotta, 'softmax over ~10⁵ tokens', 'an EBM whose Z is small enough to sum']];
    cards.forEach(([t0, h, c, m1, s], i) => { const a = eout(prog(t, t0, 0.8)), x = i < 3 ? 100 + i * 590 : 395 + (i - 3) * 590, y = i < 3 ? 260 : 610, w = 560, hh = 300;
      card(x, y, w, hh, c, a); label(h, x + 32, y + 46, a, c); if (m1[0] === '[' || m1.includes('[')) M(m1, x + 32, y + 140, 22, { a }); else text(m1, x + 32, y + 140, { font: F.mono, size: 24, color: P.chalk, a });
      para(s, x + 32, y + 210, w - 60, { size: 22, a: a * eout(prog(t, t0 + 1, 0.8)) }); });
  } });

// 12 — build it
const CODE = [
  '[k|# any network E(x) → one number. sample by rolling downhill + jiggling]',
  '[m|def] [p|langevin](E, x, steps, eta):',
  '    [m|for] _ [m|in] [p|range](steps):',
  '        x = x - eta*E.[p|grad_x](x) + np.sqrt([a|2]*eta)*np.random.randn(*x.shape)',
  '    [m|return] x',
  '',
  'buf = np.random.uniform(-R, R, ([a|8192], [a|2]))          [k|# the fantasies]',
  '[m|for] step [m|in] [p|range]([a|' + A.steps + ']):',
  '    i = np.random.randint([a|0], [a|8192], [a|256])',
  '    buf[i] = fake = [p|langevin](E, buf[i], [a|' + A.k + '], [a|' + A.eta + '])',
  '    Ed, Ef = E([d|x_batch()]), E([n|fake])',
  '    [k|# push down on data, up on fantasies; keep E small]',
  '    E.[p|backward](( [a|1] + [a|2]*α*Ed) / [a|256])',
  '    E.[p|backward]((-[a|1] + [a|2]*α*Ef) / [a|256])',
  '    E.[p|step](lr)',
];
SC.push({ dur: 32, chapter: 'build it', title: 'Build it yourself', sub: 'A sampler and a training loop, in a dozen lines.',
  caps: [[1, 9, 'The whole method fits on a screen: a Langevin sampler, and a loop that pushes energy down on data and up on fantasies. Full code, with hand-written backprop, is linked below.'],
         [9, 18, 'Try moons instead of blobs, drop the jiggle and watch samples collapse, or shorten the chains and watch training wobble.'],
         [18, 31.6, 'Recap: energy becomes probability through e^−E / Z. Z is impossible, but its gradient is an average over fantasies. Sample with ratios or with slopes. The price: slow, poorly mixing chains.']],
  draw(t) {
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1060, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 296 + i * 40, 18, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8)); card(1200, 240, 620, 690, P.sage, ra); label('the whole episode', 1236, 290, ra, P.sage);
    [['energy → probability', '[p|p]([d|x]) = e[m^|−E(x)] / [a|Z]'], ['learning: push & pull', '∇[k_|θ][m|E]([d|x]) − [kB|E][k_|pθ] ∇[k_|θ][m|E]([n|x′])'], ['sampling by ratios', 'accept with min(1, e[m^|−ΔE])'], ['sampling by slopes', '[d|x] ← [d|x] − η∇[m|E] + √(2η)[n|ε]'], ['limits', 'slow · poor mixing · unknown Z']].forEach(([nm, m], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.6, 0.8)); text(nm, 1236, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(m, 1236, 412 + i * 112, 22, { a: ai }); });
  } });

// 13 — hook → score matching & Langevin
SC.push({ dur: 48, chapter: 'next', title: 'Only the slope?', sub: 'Where Z quietly disappears.',
  caps: [[1, 8, 'Look again at the sampler: slope, plus a jiggle. It never once used the energy itself — only its slope.'],
         [8, 16, 'Take the slope of log p with respect to x: the slope of −E, minus the slope of log Z.'],
         [16, 23, 'But Z is a single number, a total over all x. It doesn’t change when x moves, so its slope is zero. Z vanishes.'],
         [23, 30, 'What remains, ∇ₓ log p(x), has a name: the score. At every point, an arrow toward more probable x.'],
         [30, 38, 'So why sculpt a landscape at all? What if a network learned the arrows directly — no energy, no Z, no fantasies?'],
         [38, 47.6, 'Next episode: score matching — and why slope plus jiggle lands on exactly the right distribution. Follow the arrows.']],
  draw(t) {
    const dimAll = 1 - 0.9 * ease(prog(t, 38.5, 1)); GA = dimAll;
    const m = square(110, 250, 660), a0 = eout(prog(t, 0.6, 0.8)); landscape(m, A.E, a0 * (1 - 0.55 * ease(prog(t, 23, 2))));
    // the score field
    const sa = eout(prog(t, 23, 1.5));
    if (sa > 0) { const Sf = A.score, big = 0.34; clipTo(m, () => Sf.p.forEach((p, k) => { const v = Sf.v[k], n = Math.hypot(v[0], v[1]); if (n < 1e-6) return; const L = big * Math.tanh(n / 6), d = [v[0] / n * L, v[1] / n * L];
      arrow(m.X(p[0] - d[0] / 2), m.Y(p[1] - d[1] / 2), m.X(p[0] + d[0] / 2), m.Y(p[1] + d[1] / 2), P.sage, 1.6, sa * 0.85); })); }
    const fl = NL * clamp((t - 2) / 20) ** 0.75; drawMarbles(m, A.lang, fl, P.terracotta, 0.9 * eout(prog(t, 1.5, 0.8)), 3, 160);
    const c1 = eout(prog(t, 1, 0.8)); card(850, 250, 980, 150, P.sage, c1);
    M('[d|x] ← [d|x] − η [s|∇ₓE(x)] + √(2η) [n|ε]', 884, 340, 32, { a: c1 });
    const c2 = eout(prog(t, 8, 0.8)); card(850, 430, 980, 400, P.ink, c2); label('the slope of log p', 884, 476, c2, P.ink);
    M('∇ₓ log [p|p]([d|x]) = ∇ₓ( −[m|E]([d|x]) − log [a|Z] )', 884, 550, 28, { a: c2 });
    const w1 = M('             = −∇ₓ[m|E]([d|x]) − ', 884, 616, 28, { a: eout(prog(t, 11, 0.8)) }), w2 = M('∇ₓ log [a|Z]', 884 + w1, 616, 28, { a: eout(prog(t, 11, 0.8)) });
    const za = eout(prog(t, 17, 0.6)); if (za > 0) { line(884 + w1 - 4, 624, 884 + w1 - 4 + (w2 + 8) * za, 596, P.rose, 3, za); text('= 0 : Z doesn’t depend on x', 884 + w1, 670, { font: F.mono, size: 18, color: P.rose, a: za }); }
    const sc = eout(prog(t, 23, 0.8)); M('[s|s](x) = ∇ₓ log [p|p]([d|x]) = −∇ₓ[m|E]([d|x])', 884, 750, 30, { a: sc }); text('“the score”: no Z anywhere', 884, 796, { font: F.mono, size: 18, color: P.sage, a: sc });
    const qa2 = eout(prog(t, 30, 0.8)); if (qa2 > 0) text('learn the arrows directly?', 884, 900, { font: F.serif, italic: true, size: 40, color: P.chalk, a: qa2 });
    GA = 1;
    const qa = eout(prog(t, 38.5, 1.2));
    if (qa > 0) {
      text('Forget the landscape.', 960, 440, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('Learn the arrows.', 960, 530, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: eout(prog(t, 39.3, 1.2)) });
      text('NEXT  ·  EPISODE 08', 960, 640, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 40.5, 1)), ls: 4 });
      text('Score matching & Langevin dynamics — follow the arrows', 960, 695, { font: F.serif, italic: true, size: 42, color: P.stone, align: 'center', a: eout(prog(t, 41, 1)) });
    }
  } });
