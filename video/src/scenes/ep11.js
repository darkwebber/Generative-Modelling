// ─────────────────────────────────────────────────────────────
//  Real guidance models (exported by code/export/export_ep11_assets.py)
//  Notation all season: z = noise (rose), x = data (terracotta), t runs from noise (0) to data (1) as in episode 10.
//  Today: y = the request (ink), w = the guidance dial (amber), v∅ = the wind with no request (dust), v_y = with it (plum),
//  v_y − v∅ = the request's push (sage).
// ─────────────────────────────────────────────────────────────
const A = window.EP11, BL = A.blobs, CLD = A.clouds, MN = A.mnist, J = MN.judges;
const sq = (x0, y0, S, R) => plotSquare(x0, y0, S, R);
const pct = v => (v * 100).toFixed(0) + '%';
function stages(b, n) { const u = b64bytes(b), v = new Int16Array(u.buffer, u.byteOffset, u.length / 2); return { v, n, S: v.length / (2 * n) }; }
function ptAt(st, k, f) { const S = st.S - 1, fi = clamp(f, 0, S), i = Math.min(S - 1, Math.floor(fi)), u = fi - i, a = (i * st.n + k) * 2, b = ((i + 1) * st.n + k) * 2;
  return [lerp(st.v[a], st.v[b], u) / 4000, lerp(st.v[a + 1], st.v[b + 1], u) / 4000]; }
const TL = A.title.paths.map(p => stages(p, A.title.n));                 // eight groups of noise, each asked for one letter
const HARD = memo('hard', () => { const u = b64bytes(A.title.hard); return new Int16Array(u.buffer, u.byteOffset, u.length / 2); });   // the same noise at w = 2.5
const FL = Object.fromEntries(Object.entries(BL.flows).map(([k, v]) => [k, stages(v, BL.z.length)]));
const CP = Object.fromEntries(Object.entries(CLD.paths).map(([k, v]) => [k, stages(v, CLD.n)]));
// digits: uint8 images (from [−1, 1]); the season's terracotta-on-dark
const IMG = new Map();          // keyed by the base64 string itself: different digit sets can share long runs of blank pixels
const imgs = (b, n) => { let v = IMG.get(b); if (!v) { const u = b64bytes(b); v = Array.from({ length: n }, (_, i) => u.subarray(i * 784, (i + 1) * 784)); IMG.set(b, v); } return v; };
const COLD = hx(P.terracotta);
function digitCanvas(px, key) { return memo('c' + key, () => { const c = document.createElement('canvas'); c.width = c.height = 28; const g = c.getContext('2d'), im = g.createImageData(28, 28);
  for (let i = 0; i < 784; i++) { const v = px[i] / 255; im.data[i * 4] = COLD[0] * v + 28 * (1 - v); im.data[i * 4 + 1] = COLD[1] * v + 24 * (1 - v); im.data[i * 4 + 2] = COLD[2] * v + 22 * (1 - v); im.data[i * 4 + 3] = 255; } g.putImageData(im, 0, 0); return c; }); }
function drawDigit(px, key, x, y, s, a = 1) { if (GA * a <= 0.002) return; setA(a); ctx.imageSmoothingEnabled = false; ctx.drawImage(digitCanvas(px, key), x, y, s, s); ctx.imageSmoothingEnabled = true; setA(1); }
function digitRow(b, key, n, x, y, cell, gap, a, from = 0) { const ims = imgs(b, 8); for (let q = 0; q < n; q++) drawDigit(ims[from + q], key + (from + q), x + q * (cell + gap), y, cell, a); }
// the eight blobs (the season's ring, scaled ½): centres, colours and names 1 … 8
const BLOB = k => [Math.cos(k * Math.PI / 4), Math.sin(k * Math.PI / 4)];
const BLOBCOL = [P.terracotta, P.amber, P.sage, P.ink, P.plum, P.rose, P.clay, P.chalk];
const nearestBlob = p => { let b = 0, d = 1e9; for (let k = 0; k < 8; k++) { const q = BLOB(k), e = Math.hypot(p[0] - q[0], p[1] - q[1]); if (e < d) { d = e; b = k; } } return d < 0.35 ? b : -1; };
const DATA8 = memo('data8', () => { const r = rng(11), g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); return Array.from({ length: 800 }, (_, i) => { const k = i % 8, c = BLOB(k); return [c[0] + 0.06 * g(), c[1] + 0.06 * g(), k]; }); });
// exact p(y | x) for the blobs (8 equal Gaussians of spread 0.06, scaled up for the illustration)
function blobPost(x, s = 0.32) { const l = BLOBCOL.map((_, k) => { const c = BLOB(k); return -((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2) / (2 * s * s); }), m = Math.max(...l), e = l.map(v => Math.exp(v - m)), S = e.reduce((a, b) => a + b); return e.map(v => v / S); }
function blobRings(m, a, want = -1, labels = true) { for (let k = 0; k < 8; k++) { const b = BLOB(k), on = want === k; ring(m.X(b[0]), m.Y(b[1]), 22, BLOBCOL[k], a * (want < 0 || on ? 0.9 : 0.3), on ? 3 : 1.5);
  if (labels) text(String(k + 1), m.X(b[0] * 1.38), m.Y(b[1] * 1.38) + 6, { font: F.mono, size: 17, color: BLOBCOL[k], align: 'center', a: a * (want < 0 || on ? 1 : 0.4) }); } }
function flowDots(m, st, f, a, r = 3, colFinal = true) { clipTo(m, () => { for (let k = 0; k < st.n; k++) { const p = ptAt(st, k, f), done = f >= st.S - 1.2, b = done && colFinal ? nearestBlob(p) : -1;
  marble(m.X(p[0]), m.Y(p[1]), r, b >= 0 ? BLOBCOL[b] : mixc(P.rose, P.terracotta, f / (st.S - 1)), a); } }); }
// arrow fields on the blobs' 15 × 15 grid (rows top → bottom)
function fieldArrows(m, F, a, col, scale = 0.09, lw = 1.5, add = null, addW = 0, step = 2) { if (GA * a <= 0.002) return; const g = BL.grid, n = g.length;
  clipTo(m, () => { for (let r = 0; r < n; r += step) for (let c = 0; c < n; c += step) { const i = r * n + c, x = g[c], y = g[n - 1 - r]; let v = F[i].slice(); if (add) { v[0] += addW * add[i][0]; v[1] += addW * add[i][1]; }
    const L = Math.hypot(v[0], v[1]); if (L < 1e-4) continue; const s = scale * Math.min(1, 2.5 / L); arrow(m.X(x - v[0] * s * 0.3), m.Y(y - v[1] * s * 0.3), m.X(x + v[0] * s), m.Y(y + v[1] * s), col, lw, a * clamp(0.35 + L / 4, 0, 0.9)); } }); }
function heatCanvas(key, n, vals, col) { return memo('h' + key, () => { const c = document.createElement('canvas'); c.width = c.height = n; const g = c.getContext('2d'), im = g.createImageData(n, n), C = hx(col);
  for (let i = 0; i < n * n; i++) { im.data[i * 4] = C[0]; im.data[i * 4 + 1] = C[1]; im.data[i * 4 + 2] = C[2]; im.data[i * 4 + 3] = Math.round(255 * clamp(vals[i], 0, 1)); } g.putImageData(im, 0, 0); return c; }); }
function heatDraw(m, key, n, vals, lo, hi, a, col = P.ink) { if (GA * a <= 0.002) return; const c = heatCanvas(key, n, vals, col); clipTo(m, () => { setA(a * 0.55); ctx.imageSmoothingEnabled = true; ctx.drawImage(c, m.X(lo), m.Y(hi), m.X(hi) - m.X(lo), m.Y(lo) - m.Y(hi)); setA(1); }); }
function heatP(m, p, a, col = P.ink) { const g = BL.hgrid, d = (g[g.length - 1] - g[0]) / (g.length - 1); return heatDraw(m, 'p' + FT, g.length, p, g[0] - d / 2, g[g.length - 1] + d / 2, a, col); }
// a one-hot request: a row of slots with one lit
function oneHot(x, y, k, n, a, lit = P.ink, s = 26, gap = 6, labels = null, nullSlot = true) { const N = n + (nullSlot ? 1 : 0);
  for (let i = 0; i < N; i++) { const on = i === k, X = x + i * (s + gap), isNull = nullSlot && i === n;
    rbox(X, y, s, s, 5, on ? rgba(lit, 0.85) : P.surface, on ? lit : P.border, a); text(isNull ? '∅' : on ? '1' : '0', X + s / 2, y + s * 0.68, { font: F.mono, size: s * 0.52, color: on ? P.bg : P.dust, align: 'center', a });
    if (labels) text(isNull ? 'none' : labels[i], X + s / 2, y + s + 18, { font: F.mono, size: 12, color: on ? lit : P.dust, align: 'center', a }); } }
const hopF = (t, t0, per) => ease(prog(t, t0, per));

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open: one network spells the title, one request at a time
const WORD = A.title.word, K0 = 6.0, KEY = 2.25, FLY = 1.7, BANG = K0 + 7 * KEY + FLY + 0.2;
SC.push({ dur: 36, chapter: 'intro',
  caps: [[1, 6, 'Last episode ended with a promise: ask, and it will draw. So let’s ask.'],
         [6.2, 13.2, 'One network, one cloud of noise. We type a request, y, and some of the noise flies to that letter.'],
         [13.4, 20.4, 'Same network every time, same kind of noise. The only thing that changes is what we ask for.'],
         [20.6, 25.6, 'Eight requests, eight letters, one word.'],
         [27, 35.6, 'Real models, trained by hand in numpy. Today: guidance, how to ask a generator for exactly what you want.']],
  draw(t) {
    const [sx, sy] = shake(t, BANG, 9); ctx.save(); ctx.translate(sx, sy);
    const cx = 960, cy = 470, sc = 330, a0 = eout(prog(t, 0.2, 1.2)), hot = t >= BANG ? Math.exp(-(t - BANG) * 1.3) : 0, r = rng(Math.floor(t * 24) + 3);
    for (let k = 0; k < 8; k++) { const st = TL[k], t0 = K0 + k * KEY + 0.35, f = (st.S - 1) * hopF(t, t0, FLY), moving = f > 0 && f < st.S - 1, done = f >= st.S - 1;
      ctx.fillStyle = done ? P.chalk : moving ? mixc(P.rose, P.chalk, f / (st.S - 1)) : P.rose;
      for (let i = 0; i < st.n; i++) { const p = ptAt(st, i, f), j = f === 0 ? 0.012 : 0, X = cx + (p[0] + j * (r() - 0.5)) * sc, Y = cy - (p[1] + j * (r() - 0.5)) * sc;
        setA(a0 * (done ? 0.9 : moving ? 0.85 : 0.55)); ctx.fillRect(X - 1.3, Y - 1.3, 2.6, 2.6); } }
    setA(1);
    if (hot > 0.01) { ctx.save(); ctx.globalCompositeOperation = 'lighter'; for (const [rr, al] of [[8, 0.05], [4, 0.12], [2, 0.3]]) { ctx.fillStyle = rgba(P.terracotta, al * hot);
      TL.forEach(st => { for (let i = 0; i < st.n; i += 2) { const p = ptAt(st, i, st.S - 1); ctx.fillRect(cx + p[0] * sc - rr, cy - p[1] * sc - rr, 2 * rr, 2 * rr); } }); } ctx.restore(); ring(cx, cy, 200 + (t - BANG) * 1400, P.chalk, hot * 0.3, 3); }
    ctx.restore();
    // the request line: "y = G", typed key by key
    const ra = eout(prog(t, K0 - 0.8, 0.6)) * (1 - ease(prog(t, BANG + 1, 1)));
    if (ra > 0) { const k = clamp(Math.floor((t - K0) / KEY), 0, 7), ch = WORD[k], kp = pop(prog(t, K0 + k * KEY, 0.25));
      text('request', 960, 780, { font: F.mono, size: 16, color: P.dust, align: 'center', a: ra, ls: 3 });
      M(`[p|y] = “${t >= K0 ? ch : ' '}”`, 900, 836, 40, { a: ra });
      for (let i = 0; i < 8; i++) { const x = 960 + (i - 3.5) * 40, on = t >= K0 + i * KEY; text(WORD[i], x, 900, { font: F.serif, size: 28, color: on ? (i === k ? P.ink : P.stone) : P.border, align: 'center', a: ra * (i === k ? 0.6 + 0.4 * kp : 1) }); } }
    const a = eout(prog(t, BANG + 0.6, 1.2));
    text('EPISODE 11', 960, 300, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('asking a generator for what you want', 960, 680, { size: 36, color: P.stone, align: 'center', a: eout(prog(t, BANG + 1.4, 1.2)) });
    const tg = [[P.rose, 'z  noise'], [P.terracotta, 'x  data'], [P.ink, 'y  the request'], [P.plum, 'v  the wind'], [P.amber, 'w  the dial']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let tx = 960 - tw / 2; tg.forEach(([c, s], i) => { tx += tag(tx, 760, c, s, eout(prog(t, BANG + 2.2 + i * 0.3, 0.6)), 21) + 14; });
  } });

// 01 — what asking means: p(x) and p(x | y)
SC.push({ dur: 46, chapter: 'asking', title: 'What asking means', sub: 'From p(x) to p(x | y): the same data, filtered by a label.',
  caps: [[1, 8, 'Every model this season learned one thing, p(x): how likely each data point is. Sampling it gives whatever it gives.'],
         [8, 15, 'Now give every point a label, y. Here the label just says which blob a point belongs to, one to eight.'],
         [15, 23, 'Asking for blob 3 means sampling from p(x | y = 3). Read it as: the probability of x, given that y is 3.'],
         [23, 30, 'Think of it as a filter. Keep only the points labelled 3, and rescale what’s left so it still adds up to one.'],
         [30, 38, 'Bayes’ rule links the two: p(x | y) = p(y | x) · p(x) / p(y). It turns the question around.'],
         [38, 45.6, 'p(y | x) looks at a point and says how likely each label is. A machine that does that is called a classifier.']],
  draw(t) {
    const m = sq(110, 250, 640, 1.7), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const lab = eout(prog(t, 8.5, 1.2)), filt = ease(prog(t, 23.5, 2)), want = t > 15.5 ? 2 : -1;
    clipTo(m, () => DATA8.forEach(([x, y, k], i) => { const keep = k === 2 ? 1 : 1 - 0.88 * filt, col = mixc(P.terracotta, BLOBCOL[k], lab); dot(m.X(x), m.Y(y), k === 2 && filt > 0 ? 2.8 + filt : 2.6, col, a0 * 0.9 * keep); void i; }));
    blobRings(m, a0 * lab, want);
    text(filt > 0.5 ? 'p(x | y = 3): only blob 3, rescaled' : lab > 0.5 ? 'coloured by label y' : 'p(x): the eight blobs', m.x0, m.y0 + m.S + 30, { font: F.mono, size: 18, color: filt > 0.5 ? P.sage : P.stone, a: a0 });
    // a probe point sliding between blobs 3 and 2, with the classifier's answer
    const ca = eout(prog(t, 38.5, 0.8));
    if (ca > 0) { const u = 0.5 + 0.5 * Math.sin((t - 38.5) * 0.9), c3 = BLOB(2), c2 = BLOB(1), p = [lerp(c3[0], c2[0], u) + 0.08, lerp(c3[1], c2[1], u) + 0.08], pr = blobPost(p);
      ring(m.X(p[0]), m.Y(p[1]), 9, P.chalk, ca, 2); dot(m.X(p[0]), m.Y(p[1]), 4, P.chalk, ca);
      const bx = 1180, by = 870; text('p(y | x) for this point', bx, by - 150, { font: F.mono, size: 16, color: P.stone, a: ca });
      pr.forEach((v, k) => { const h = 110 * v; setA(ca); ctx.fillStyle = BLOBCOL[k]; ctx.fillRect(bx + k * 62, by - h, 44, h); setA(1); text(String(k + 1), bx + k * 62 + 22, by + 24, { font: F.mono, size: 15, color: BLOBCOL[k], align: 'center', a: ca }); }); }
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 130, P.ink, c1); label('the season so far', 864, 296, c1, P.ink);
    M('[d|x]  ∼  p([d|x])', 864, 350, 30, { a: c1 }); text('whatever the dice decide', 1200, 350, { font: F.mono, size: 16, color: P.dust, a: c1 });
    const c2 = eout(prog(t, 15.5, 0.8)); if (c2 > 0) { card(830, 400, 1000, 150, P.ink, c2 * (1 - ease(prog(t, 37.5, 0.8)))); label('a request', 864, 446, c2 * (1 - ease(prog(t, 37.5, 0.8))), P.ink);
      M('[d|x]  ∼  p([d|x] | [p|y] = 3)', 864, 504, 32, { a: c2 * (1 - ease(prog(t, 37.5, 0.8))) }); text('“the probability of x, given y = 3”', 1300, 504, { font: F.mono, size: 16, color: P.dust, a: c2 * (1 - ease(prog(t, 37.5, 0.8))) }); }
    const c3 = eout(prog(t, 30.5, 0.8)); if (c3 > 0) { card(830, 570, 1000, 130, P.sage, c3 * (1 - ease(prog(t, 37.5, 0.8)))); label('Bayes’ rule', 864, 616, c3 * (1 - ease(prog(t, 37.5, 0.8))), P.sage);
      M('p([d|x] | [p|y]) = p([p|y] | [d|x]) · p([d|x]) / p([p|y])', 864, 670, 30, { a: c3 * (1 - ease(prog(t, 37.5, 0.8))) }); }
    if (ca > 0) { card(830, 400, 1000, 190, P.plum, ca); label('a classifier', 864, 446, ca, P.plum); M('p([p|y] | [d|x])', 864, 504, 34, { a: ca });
      para('Point in, one probability per label out. Here: the exact answer for our blobs, drawn a little blurred.', 864, 548, 930, { size: 22, a: ca }); }
  } });

// 02 — recipe 1: tell the network
const LBL8 = ['1', '2', '3', '4', '5', '6', '7', '8'];
SC.push({ dur: 48, chapter: 'tell it', title: 'Recipe 1: tell the network', sub: 'The request becomes one more input.',
  caps: [[1, 8, 'The most direct idea: tell the network what we want. The request y becomes one more input, next to x and t.'],
         [8, 15, 'A network reads numbers, so y becomes a one-hot list: eight slots, all zero, with a single one in slot y.'],
         [15, 23, 'Add a ninth slot, ∅, meaning no request. While training, we hide the label one time in ten and light ∅ instead.'],
         [23, 31, 'Everything else is episode 10’s regression, unchanged: a random road from noise to data, a random time, match its velocity.'],
         [31, 39, 'Averaging only roads that share a label gives one wind per request: v(x, t, y) = E[ x − z | xₜ = x, y ].'],
         [39, 47.6, 'Same noise, three requests. Ask for ∅ and it goes everywhere. Ask for 3, or 7, and it goes only there.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8));
    // the network, drawn as a box with its inputs
    const bx = 330, by = 430, na = a0;
    rbox(bx, by, 300, 170, 14, P.surface, P.plum, na); text('vθ', bx + 150, by + 100, { font: F.serif, italic: true, size: 56, color: P.plum, align: 'center', a: na });
    [['x', P.terracotta, 'where the point is', 0], ['t', P.amber, 'how far along', 1]].forEach(([s, c, d, i]) => { const y = by + 45 + i * 60; arrow(150, y, bx - 8, y, P.border, 2, na); text(s, 120, y + 9, { font: F.serif, italic: true, size: 30, color: c, align: 'center', a: na }); text(d, 150, y - 14, { font: F.mono, size: 13, color: P.dust, a: na }); });
    arrow(bx + 308, by + 85, bx + 440, by + 85, P.plum, 3, na); text('velocity', bx + 380, by + 70, { font: F.mono, size: 14, color: P.plum, align: 'center', a: na });
    const ya = eout(prog(t, 2, 0.8)), lit = t < 15 ? 2 : t < 23 ? (Math.floor((t - 15) * 1.2) % 10 === 0 ? 8 : 2) : 2;
    if (ya > 0) { arrow(bx + 150, 330, bx + 150, by - 8, P.ink, 2.5, ya); text('y', bx + 150, 300, { font: F.serif, italic: true, size: 32, color: P.ink, align: 'center', a: ya });
      const oa = eout(prog(t, 8.5, 0.8)); if (oa > 0) { oneHot(120, 670, lit, 8, oa * (t < 15.5 ? 1 : 1), P.ink, 44, 8, LBL8, t > 15.5); text(t > 15.5 ? 'one-hot request, with the “none” slot ∅' : 'one-hot request: y = 3', 120, 790, { font: F.mono, size: 16, color: P.stone, a: oa }); } }
    // the loss, recalled
    const la = eout(prog(t, 23.5, 0.8)); if (la > 0) { card(110, 820, 720, 110, P.sage, la); M('‖ [m|v][m_|θ]([d|x][d_|t], t, [p|y]) − ([d|x] − [n|z]) ‖²', 140, 885, 28, { a: la }); }
    // the flows: none, 3, 7
    const m = sq(1190, 250, 640, 2.0), fa = eout(prog(t, 30.5, 0.8)); if (fa > 0) { frame(m, fa);
      const key = t < 39 ? 'none' : t < 43.3 ? 'b3' : 'b7', t0 = t < 39 ? 31.5 : t < 43.3 ? 39.3 : 43.5, st = FL[key], f = (st.S - 1) * hopF(t, t0, 2.8), want = key === 'b3' ? 2 : key === 'b7' ? 6 : -1;
      blobRings(m, fa, want); flowDots(m, st, f, fa);
      M(want < 0 ? '[p|y] = ∅' : `[p|y] = ${want + 1}`, m.x0, m.y0 + m.S + 34, 26, { a: fa }); }
    const ea = eout(prog(t, 31.5, 0.8)); if (ea > 0) { card(640, 640, 520, 160, P.plum, ea); label('one wind per request', 668, 684, ea, P.plum); M('[m|v]([d|x], t, [p|y]) = [kB|E][ [d|x] − [n|z] | [d|x][d_|t] = [d|x], [p|y] ]', 668, 750, 20, { a: ea }); }
  } });

// 03 — digits on request
const ACC1 = J['1'].acc;
const MISS = MN.gridRead.map((row, d) => row.map(r => r !== d));
SC.push({ dur: 46, chapter: 'digits', title: 'Digits on request', sub: 'Ten labels, one network: episode 10’s digit flow, now with a request.',
  caps: [[1, 8, 'Same recipe on handwritten digits: ten label slots plus ∅, starting from episode 10’s network, trained twenty thousand more steps.'],
         [8, 15, 'Each row asks for one digit. Each column starts from the very same noise.'],
         [15, 23, 'Look down a column. The same noise keeps its slant and its thickness in every digit. The noise carries style, the label carries content.'],
         [23, 31, `Did it listen? Episode 4 gave us a digit reader, a classifier trained on real digits. It agrees with the request ${pct(ACC1)} of the time.`],
         [31, 38, 'Good, but not perfect. A few come out ambiguous, like a 4 that could pass for a 9.'],
         [38, 45.6, 'The model hears the request, but softly. We want a way to make it listen harder, and for that, we need Bayes.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), cell = 62, gap = 5, x0 = 170, y0 = 250;
    for (let d = 0; d < 10; d++) { const ra = a0 * eout(prog(t, 1.5 + d * 0.35, 0.5)), y = y0 + d * (cell + gap);
      text(String(d), 130, y + cell * 0.66, { font: F.mono, size: 22, color: P.ink, align: 'center', a: ra });
      digitRow(MN.grid[d], `g${d}_`, 8, x0, y, cell, gap, ra);
      const hl = t > 15 && t < 23 ? 1 : 0; if (hl) { const col = Math.floor((t - 15) / 2.7) % 8; rbox(x0 + col * (cell + gap) - 3, y0 - 6, cell + 6, 10 * (cell + gap) + 6, 8, 'rgba(0,0,0,0)', P.sage, a0 * 0.8); }
      const ma = eout(prog(t, 31, 0.8)); if (ma > 0) MISS[d].forEach((bad, c) => { if (bad) ring(x0 + c * (cell + gap) + cell / 2, y + cell / 2, cell * 0.58, P.rose, ma, 2.5); }); }
    text('same noise down each column  →', x0, 236, { font: F.mono, size: 15, color: P.dust, a: a0 * eout(prog(t, 8, 0.8)) });
    const c1 = eout(prog(t, 23.5, 0.8)); if (c1 > 0) { card(830, 250, 1000, 420, P.ink, c1); label('the digit reader agrees with the request', 864, 296, c1, P.ink);
      const pc = J['1'].perClass; pc.forEach((v, d) => { const x = 880 + d * 92, h = 220 * v * eout(prog(t, 24 + d * 0.1, 0.6)); setA(c1); ctx.fillStyle = v > 0.9 ? P.sage : P.amber; ctx.fillRect(x, 610 - h, 56, h); setA(1);
        text(String(d), x + 28, 640, { font: F.mono, size: 18, color: P.ink, align: 'center', a: c1 }); text(pct(v), x + 28, 600 - h, { font: F.mono, size: 14, color: P.stone, align: 'center', a: c1 }); });
      M(`overall  [s|${pct(ACC1)}]      (w = 1: the plain conditional model)`, 864, 346, 22, { a: c1 }); }
    const c2 = eout(prog(t, 31.5, 0.8)); if (c2 > 0) { card(830, 690, 1000, 160, P.rose, c2); label('rose rings: the reader disagrees', 864, 736, c2, P.rose);
      para('The request went in. The network heard it, softly.', 864, 790, 930, { size: 26, a: c2 }); }
  } });

// 04 — Bayes' rule, as arrows
const FT = '0.6', FF = BL.fields[FT];
const SUM = FF.v0.map((v, i) => [v[0] + FF.cg[i][0], v[1] + FF.cg[i][1]]);
const AGREE = (() => { let s = 0, n = 0; SUM.forEach((u, i) => { const v = FF.vy[i], a = Math.hypot(...u), b = Math.hypot(...v); if (a < 0.05 || b < 0.05) return; s += Math.acos(clamp((u[0] * v[0] + u[1] * v[1]) / (a * b), -1, 1)); n++; }); return Math.round(s / n * 180 / Math.PI); })();
SC.push({ dur: 54, chapter: 'bayes', title: 'Bayes’ rule, as arrows', sub: 'Take the log, then the gradient: a product becomes a sum of arrows.',
  caps: [[1, 8, 'To make it listen harder, go back to Bayes’ rule and do two things to it. First, take the logarithm of both sides.'],
         [8, 16, 'Logarithms turn multiplying into adding, and dividing into subtracting. The rule becomes a plain sum.'],
         [16, 24, 'Second, take the gradient, ∇: the arrow that points where a quantity grows fastest. Episode 8 called ∇ log p(x) the score.'],
         [24, 32, 'p(y) is the same wherever x is, so its arrow is zero. What remains is two arrows, added.'],
         [32, 40, 'The new one is the classifier’s arrow: from every point, the way to look more like a 3. Here it is from a real classifier we trained.'],
         [40, 47, 'Add it to the wind with no request, and the wind bends toward blob 3.'],
         [47, 53.6, `Our network trained with labels learned almost the same wind on its own: on average the two arrows differ by ${AGREE}°.`]],
  draw(t) {
    const m = sq(110, 250, 620, 1.6), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    clipTo(m, () => DATA8.forEach(([x, y, k]) => dot(m.X(x), m.Y(y), 2.2, BLOBCOL[k], a0 * 0.35))); blobRings(m, a0 * 0.8, t > 32 ? 2 : -1);
    const ha = eout(prog(t, 32, 1)); heatP(m, FF.p, ha);
    const ga = ha * (1 - ease(prog(t, 40, 0.8))); fieldArrows(m, FF.cg, ga, P.sage, 0.1, 2.2);
    const sa = eout(prog(t, 40, 0.8)) * (1 - ease(prog(t, 47, 0.8))); fieldArrows(m, FF.v0, sa * 0.5, P.dust, 0.1, 1.6); fieldArrows(m, SUM, sa, P.plum, 0.1, 2.4);
    const va = eout(prog(t, 47, 0.8)); fieldArrows(m, FF.vy, va, P.plum, 0.1, 2.4);
    const cap = t < 32 ? 'the eight blobs' : t < 40 ? 'ink: p(y = 3 | x) · sage: the classifier’s arrow' : t < 47 ? 'dust: no request · plum: plus the classifier' : 'plum: the labelled network’s own wind for y = 3';
    text(cap, m.x0, m.y0 + m.S + 30, { font: F.mono, size: 16, color: P.stone, a: a0 }); text('all at t = 0.6, drawn as velocities (next scene: why)', m.x0, m.y0 + m.S + 54, { font: F.mono, size: 13, color: P.dust, a: a0 * eout(prog(t, 32, 1)) });
    // the derivation, line by line
    const L = [[1, 'p([d|x] | [p|y]) = p([p|y] | [d|x]) · p([d|x]) / p([p|y])', 'Bayes’ rule'],
               [8.5, 'log p([d|x] | [p|y]) = log p([d|x]) + log p([p|y] | [d|x]) − log p([p|y])', 'take the log'],
               [24.5, '∇ log p([d|x] | [p|y]) = ∇ log p([d|x]) + ∇ log p([p|y] | [d|x]) − [k|0]', 'take the gradient']];
    card(830, 250, 1000, 470, P.sage, a0); label('two moves', 864, 296, a0, P.sage);
    L.forEach(([t0, s, nm], i) => { const a = eout(prog(t, t0, 0.8)); text(nm, 864, 356 + i * 120, { font: F.serif, italic: true, size: 20, color: P.dust, a }); M(s, 864, 396 + i * 120, 26, { a }); });
    const ua = eout(prog(t, 25.5, 0.8)); if (ua > 0) { const y = 660; text('the score (episode 8)', 1080, y, { font: F.mono, size: 14, color: P.terracotta, align: 'center', a: ua }); text('the classifier’s arrow', 1370, y, { font: F.mono, size: 14, color: P.sage, align: 'center', a: ua });
      text('p(y) has no x in it', 1640, y, { font: F.mono, size: 14, color: P.dust, align: 'center', a: ua }); }
    const na = eout(prog(t, 16.5, 0.8)); if (na > 0) { card(830, 740, 1000, 180, P.terracotta, na); label('∇, the gradient', 864, 786, na, P.terracotta);
      para('At every point, the arrow toward where a quantity rises fastest. ∇ log p(x) points toward more likely data: episode 8’s score.', 864, 830, 930, { size: 23, a: na }); }
  } });

// 05 — recipe 2: classifier guidance
const NZ = memo('nz', () => { const r = rng(5), g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); return Float32Array.from({ length: 784 }, g); });
const NOISY = memo('noisy', () => { const x = imgs(MN.grid[7], 8)[1]; return [0.2, 0.45, 0.7, 1].map(tt => Uint8Array.from(x, (v, i) => clamp(Math.round((((1 - tt) * NZ[i] + tt * (v / 127.5 - 1)) + 1) * 127.5), 0, 255))); });
SC.push({ dur: 52, chapter: 'classifier', title: 'Recipe 2: classifier guidance', sub: 'Follow the wind, plus a classifier’s arrow, turned up by w.',
  caps: [[1, 8, 'So here is a second recipe: follow the usual wind, and add the classifier’s arrow. It is called classifier guidance, published in 2021.'],
         [8, 16, 'One catch: halfway along a road a point is still part noise, so the classifier must learn to read noisy points at every t.'],
         [16, 26, 'Another: the flow is written in velocities, the classifier in scores. Episode 10’s formulas translate: a change in score, times (1 − t)/t, is a change in velocity.'],
         [26, 33, 'Now the dial: multiply the classifier’s arrow by w. At w = 0 the request is ignored: noise goes to every blob.'],
         [33, 40, 'At w = 1 it is plain Bayes, exactly the conditional distribution. Everything lands on blob 3.'],
         [40, 47, 'Above 1 we ask for p(x) times p(y | x) to the power w: only the points the classifier is most sure about.'],
         [47, 51.6, 'It works, but needs a second, noise-trained network, and can chase whatever fools it.']],
  draw(t) {
    const m = sq(110, 250, 640, 2.0), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const phase = t < 33 ? 'cg0' : t < 40 ? 'cg1' : 'cg4', t0 = t < 33 ? 26.3 : t < 40 ? 33.3 : 40.3, st = FL[phase], f = t < 26 ? 0 : (st.S - 1) * hopF(t, t0, 3), w = { cg0: 0, cg1: 1, cg4: 4 }[phase];
    blobRings(m, a0, t > 26 ? 2 : -1); flowDots(m, st, f, a0);
    if (t > 26) M(`asking for [p|y] = 3 ·  [a|w] = ${w}`, m.x0, m.y0 + m.S + 34, 24, { a: a0 });
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 150, P.sage, c1); label('classifier guidance · Dhariwal & Nichol, 2021', 864, 296, c1, P.sage);
    M('[m|v]  =  [m|v][m_|∅]  +  [a|w] · ∇ log p([p|y] | [d|x][d_|t])       (in scores, for now)', 864, 356, 26, { a: c1 });
    const c2 = eout(prog(t, 8.5, 0.8)) * (1 - ease(prog(t, 16, 0.6))); if (c2 > 0) { card(830, 420, 1000, 300, P.ink, c2); label('it must read noisy points', 864, 466, c2, P.ink);
      NOISY.forEach((px, i) => { drawDigit(px, 'nz' + i, 880 + i * 180, 500, 130, c2); text(`t = ${[0.2, 0.45, 0.7, 1][i]}`, 945 + i * 180, 660, { font: F.mono, size: 16, color: P.amber, align: 'center', a: c2 }); }); }
    const c3 = eout(prog(t, 16.5, 0.8)) * (1 - ease(prog(t, 26, 0.6))); if (c3 > 0) { card(830, 420, 1000, 300, P.terracotta, c3); label('scores to velocities (episode 10)', 864, 466, c3, P.terracotta);
      M('[d|x̂] = [d|x][d_|t] + (1 − t)·[m|v]      [n|ẑ] = [d|x][d_|t] − t·[m|v]      s = −[n|ẑ] / (1 − t)', 864, 530, 24, { a: c3 });
      M('so   [m|v] = ( [d|x][d_|t] + (1 − t)·s ) / t', 864, 600, 28, { a: eout(prog(t, 19, 0.8)) });
      M('and  Δ[m|v] = (1 − t)/t · Δs', 864, 660, 30, { a: eout(prog(t, 21.5, 0.8)) }); }
    const c4 = eout(prog(t, 26.5, 0.8)); if (c4 > 0) { card(830, 420, 1000, 300, P.amber, c4); label('what w asks for', 864, 466, c4, P.amber);
      M('[m|v] = [m|v][m_|∅] + [a|w] · (1 − t)/t · ∇ log p([p|y] | [d|x][d_|t])', 864, 530, 26, { a: c4 });
      M('samples from roughly   p([d|x]) · p([p|y] | [d|x])^[a|w]', 864, 600, 26, { a: eout(prog(t, 40.5, 0.8)) });
      text('w = 0: ignore the request · w = 1: Bayes · w > 1: only what the classifier is sure of', 864, 660, { font: F.mono, size: 16, color: P.dust, a: c4 }); }
    const c5 = eout(prog(t, 47, 0.8)); if (c5 > 0) { card(830, 740, 1000, 180, P.rose, c5); label('on digits, asking for 7', 864, 786, c5, P.rose);
      ['1', '4'].forEach((w, j) => { text(`w = ${w}`, 880 + j * 470, 836, { font: F.mono, size: 15, color: P.amber, a: c5 }); digitRow(MN.cg[w], `cg${w}_`, 6, 960 + j * 470, 806, 56, 3, c5); }); }
  } });

// 06 — recipe 3: classifier-free guidance
const PR = BL.probe;
SC.push({ dur: 52, chapter: 'CFG', title: 'Recipe 3: no classifier needed', sub: 'The classifier’s arrow is a difference of two winds we already have.',
  caps: [[1, 9, 'Now the idea that put guidance everywhere. Rearrange Bayes’ rule once more, and solve for the classifier’s arrow.'],
         [9, 17, 'It is just a difference: the wind with the request, minus the wind without it. And we already have both.'],
         [17, 24, 'Remember the ∅ slot? Hiding the label one time in ten taught one network both winds: with the request, and without.'],
         [24, 31, 'At one point: grey is the wind without the request, plum the wind with it. The sage arrow between is the push.'],
         [31, 39, 'w = 1 follows that push once and lands on the plum arrow. w = 3 follows it three times. This is classifier-free guidance, from 2021.'],
         [39, 46, 'The price is two network calls per step: one with the request, one with ∅.'],
         [46, 51.6, 'Same formula as classifier guidance. The classifier is still there, hiding inside the difference.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), v0 = PR.v0, vy = PR.vy, d = [vy[0] - v0[0], vy[1] - v0[1]];
    const tip = w => [v0[0] + w * d[0], v0[1] + w * d[1]], pts = [[0, 0], v0, vy, tip(3.2)];
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), bx = [Math.min(...xs), Math.max(...xs)], by = [Math.min(...ys), Math.max(...ys)];
    const S = Math.min(540 / (bx[1] - bx[0] + 1e-6), 500 / (by[1] - by[0] + 1e-6)), O = [450 - (bx[0] + bx[1]) / 2 * S, 600 + (by[0] + by[1]) / 2 * S], V = v => [O[0] + v[0] * S, O[1] - v[1] * S];
    const wv = t < 24 ? 0 : t < 31 ? ease(prog(t, 25.5, 2.5)) : 1 + 2 * ease(prog(t, 33, 3));
    card(110, 250, 680, 670, P.plum, a0); label(`one point, at t = ${PR.t}`, 144, 296, a0, P.plum);
    const da = eout(prog(t, 24, 0.8));
    if (da > 0) { const b0 = V(v0), e = V(tip(3.2)); line(b0[0], b0[1], e[0], e[1], P.sage, 1.5, da * 0.5, [6, 7]);
      [1, 2, 3].forEach(k => { const p = V(tip(k)); dot(p[0], p[1], 4, P.sage, da * 0.8); text(`w = ${k}`, p[0] + 12, p[1] + 5, { font: F.mono, size: 14, color: P.sage, a: da * 0.8 }); }); }
    const a1 = eout(prog(t, 17.5, 0.8)), a2 = a1 * eout(prog(t, 19.5, 0.8)), e0 = V(v0), ey = V(vy);
    arrow(O[0], O[1], e0[0], e0[1], P.stone, 4, a1); M('[k|v][k_|∅]', e0[0] - 64, e0[1] + 6, 24, { a: a1 });
    arrow(O[0], O[1], ey[0], ey[1], P.plum, 4, a2); M('[m|v][m_|y]', ey[0] + 12, ey[1] + 22, 24, { a: a2 });
    if (da > 0) arrow(e0[0], e0[1], ey[0], ey[1], P.sage, 3, da);
    if (t > 24) { const eg = V(tip(wv)); arrow(O[0], O[1], eg[0], eg[1], P.amber, 5, eout(prog(t, 24.5, 0.6))); M(`[a|w] = ${wv.toFixed(1)}`, 144, 890, 26, { a: a0 }); }
    marble(O[0], O[1], 8, P.rose, a0);
    // the formulas
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 180, P.sage, c1); label('Bayes, solved for the classifier', 864, 296, c1, P.sage);
    M('∇ log p([p|y] | [d|x]) = ∇ log p([d|x] | [p|y]) − ∇ log p([d|x])', 864, 352, 26, { a: c1 });
    M('in velocities:   (1 − t)/t · ∇ log p([p|y] | [d|x][d_|t]) = [m|v][m_|y] − [k|v][k_|∅]', 864, 404, 22, { a: eout(prog(t, 9.5, 0.8)) });
    const c2 = eout(prog(t, 17.5, 0.8)); if (c2 > 0) { card(830, 450, 1000, 200, P.ink, c2); label('one network, two winds', 864, 496, c2, P.ink);
      oneHot(864, 520, 2, 8, c2, P.ink, 30, 6); M('→  [m|v][m_|y]', 1200, 544, 22, { a: c2 });
      oneHot(864, 562, 8, 8, c2, P.dust, 30, 6); M('→  [k|v][k_|∅]', 1200, 586, 22, { a: c2 });
      text(`the push here: network (${d.map(v => v.toFixed(2)).join(', ')}) · our trained classifier (${PR.cg.map(v => v.toFixed(2)).join(', ')})`, 864, 628, { font: F.mono, size: 14, color: P.dust, a: eout(prog(t, 20, 0.8)) }); }
    const c3 = eout(prog(t, 31.5, 0.8)); if (c3 > 0) { card(830, 670, 1000, 150, P.amber, c3); label('classifier-free guidance · Ho & Salimans, 2021', 864, 716, c3, P.amber);
      M('[m|v] = [k|v][k_|∅] + [a|w] · ( [m|v][m_|y] − [k|v][k_|∅] )', 864, 778, 34, { a: c3 }); }
    const c4 = eout(prog(t, 39.5, 0.8)); if (c4 > 0) { card(830, 840, 1000, 80, P.dust, c4); text('cost: 2 network calls per step, instead of 1', 864, 890, { font: F.mono, size: 20, color: P.stone, a: c4 }); }
  } });

// 07 — the dial, on two overlapping classes
const WS = CLD.ws, SEP = CLD.sep, SD = CLD.std;
const dens = (x, y, w) => { const g = cx => Math.exp(-((x - cx) ** 2 + y * y) / (2 * SD * SD)), pa = g(-SEP), pb = g(SEP), px = (pa + pb) / 2, pA = pa / (pa + pb + 1e-30); return px * Math.pow(pA, w); };
const DGRID = memo('dgrid', () => { const out = {}; for (const w of WS) { const n = 56, v = []; let mx = 0; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const x = -2.3 + 4.6 * (c + 0.5) / n, y = 2.3 - 4.6 * (r + 0.5) / n, d = dens(x, y, w); v.push(d); mx = Math.max(mx, d); } out[w] = v.map(d => d / mx); } return out; });
const MARG = memo('marg', () => { const out = {}; for (const w of WS) { const xs = Array.from({ length: 120 }, (_, i) => -2.3 + 4.6 * i / 119), v = xs.map(x => { let s = 0; for (let j = 0; j < 60; j++) { const y = -2.3 + 4.6 * j / 59; s += dens(x, y, w); } return s; }), S = v.reduce((a, b) => a + b) * (4.6 / 119); out[w] = v.map(u => u / S); } return out; });
SC.push({ dur: 52, chapter: 'the dial', title: 'What the dial does', sub: 'Two overlapping classes, and what p(x) · p(y | x)^w asks for.',
  caps: [[1, 9, 'What does w really do? Take two classes that overlap: A on the left, B on the right. Near the middle, a point could be either.'],
         [9, 17, 'At w = 0 there is no request, so we get both clouds. At w = 1, class A, overlap and all: a fair sample of A.'],
         [17, 24, 'Turn it up. The points in the overlap are the ones the classifier is unsure about, and the power w shrinks them fastest.'],
         [24, 31, 'So the samples retreat from B, toward what is unmistakably A. The cloud narrows and drifts left.'],
         [31, 40, 'The shading is what p(x) · p(A | x)^w asks for. Our samples overshoot it: a push at every step adds up to more than the target.'],
         [40, 47, 'More w: more certain, less varied. Fidelity up, diversity down.'],
         [47, 51.6, 'Now watch that trade happen on real digits.']],
  draw(t) {
    const m = sq(110, 250, 640, 2.3), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const wi = t < 13 ? 0 : t < 17 ? 1 : t < 21 ? 2 : t < 26 ? 3 : 4, w = WS[wi], tw = [9.2, 13, 17.2, 21.5, 26][wi], st = CP[String(w)], f = t < 1.5 ? 0 : (st.S - 1) * hopF(t, wi ? tw : 1.5, 2.2);
    const ha = eout(prog(t, 31, 1)); if (ha > 0) heatDraw(m, 'd' + w, 56, DGRID[w], -2.3, 2.3, ha * 0.9);
    [[-SEP, 'A', P.terracotta], [SEP, 'B', P.ink]].forEach(([cx, nm, c]) => { ring(m.X(cx), m.Y(0), SD * m.S / 4.6 * 2, c, a0 * 0.6, 1.5, [5, 6]); text(nm, m.X(cx), m.Y(0) - SD * m.S / 4.6 * 2 - 10, { font: F.serif, italic: true, size: 30, color: c, align: 'center', a: a0 }); });
    const pre = t < 9 ? 1 : 0;
    clipTo(m, () => { for (let k = 0; k < st.n; k++) { const p = ptAt(st, k, f); dot(m.X(p[0]), m.Y(p[1]), 2.6, pre ? P.rose : mixc(P.rose, P.terracotta, f / (st.S - 1)), a0 * 0.85); } });
    // the dial
    const da = eout(prog(t, 8.5, 0.8)), x0 = 900, x1 = 1780, y = 330, X = v => x0 + (x1 - x0) * Math.log2(1 + v) / Math.log2(9);
    card(830, 250, 1000, 160, P.amber, da); line(x0, y + 30, x1, y + 30, P.border, 3, da);
    WS.forEach(v => { text(String(v), X(v), y + 70, { font: F.mono, size: 18, color: v === w ? P.amber : P.dust, align: 'center', a: da }); line(X(v), y + 22, X(v), y + 38, P.border, 2, da); });
    const kx = X(w); dot(kx, y + 30, 14 * (0.9 + 0.1 * pop(prog(t, tw, 0.3))), P.amber, da); text('guidance dial  w', x0, y - 20, { font: F.mono, size: 16, color: P.amber, a: da });
    const S = CLD.stats[String(w)], sa = eout(prog(t, 17, 0.8)); if (sa > 0) { card(830, 430, 1000, 170, P.sage, sa); label(`samples at w = ${w}`, 864, 476, sa, P.sage);
      M(`on A’s side  [s|${pct(S.pA)}]      spread  ${S.std[0].toFixed(2)}      centre  ${S.mean[0].toFixed(2)}`, 864, 540, 24, { a: sa }); }
    const pa = eout(prog(t, 31, 0.8)); if (pa > 0) { card(830, 620, 1000, 300, P.ink, pa); label('across the panel: what p(x) · p(A | x)^w asks for, and what we got', 864, 666, pa, P.ink);
      const gx0 = 880, gx1 = 1780, gy = 890, H = 170, xs = i => gx0 + (gx1 - gx0) * i / 119;
      const C = MARG[w], bins0 = new Array(30).fill(0); for (let k = 0; k < st.n; k++) { const p = ptAt(st, k, st.S - 1), b = Math.floor((p[0] + 2.3) / 4.6 * 30); if (b >= 0 && b < 30) bins0[b]++; }
      const mx = 1.08 * Math.max(...C, ...bins0.map(c => c / st.n / (4.6 / 30))); setA(pa); ctx.strokeStyle = P.ink; ctx.lineWidth = 2.5; ctx.beginPath(); C.forEach((v, i) => { const Y = gy - H * v / mx; i ? ctx.lineTo(xs(i), Y) : ctx.moveTo(xs(i), Y); }); ctx.stroke(); setA(1);
      const bins = new Array(30).fill(0); for (let k = 0; k < st.n; k++) { const p = ptAt(st, k, st.S - 1), b = Math.floor((p[0] + 2.3) / 4.6 * 30); if (b >= 0 && b < 30) bins[b]++; }
      bins.forEach((c, b) => { const v = c / st.n / (4.6 / 30), h = H * v / mx, xa = gx0 + (gx1 - gx0) * b / 30; setA(pa * 0.7); ctx.fillStyle = P.terracotta; ctx.fillRect(xa + 1, gy - h, (gx1 - gx0) / 30 - 2, h); setA(1); });
      line(gx0, gy, gx1, gy, P.border, 1.5, pa); text('ink line: the target · bars: our samples', gx1, gy + 26, { font: F.mono, size: 13, color: P.dust, align: 'right', a: pa }); }
  } });

// 08 — the price: the dial on real digits, and episode 4's judges
const JW = Object.keys(J).map(Number).sort((a, b) => a - b), JB = JW.reduce((b, w) => J[w].FID < J[b].FID ? w : b, JW[0]);
const SWW = ['0', '1', '2', '4', '8'];
SC.push({ dur: 54, chapter: 'the price', title: 'The price of listening', sub: 'Same noise, one request, the dial turned up; and episode 4’s judges.',
  caps: [[1, 8, 'Same eight noise images, all asking for a 7, with the dial at 0, 1, 2, 4 and 8.'],
         [8, 16, 'At 0, whatever digit the noise leads to. At 1, sevens. By 4, bolder, more obvious sevens. At 8, a caricature.'],
         [16, 24, `Episode 4’s judges, 2,000 digits per setting. The digit reader agrees with the request: ${pct(J['0'].acc)} at w = 0, pure chance; ${pct(J['1'].acc)} at 1; ${pct(J['4'].acc)} at 4.`],
         [24, 32, `FID, the distance from real digits (lower is better): ${J['0'].FID.toFixed(1)} at w = 0, best at w = ${JB} with ${J[JB].FID.toFixed(1)}, then ${J['8'].FID.toFixed(1)} at 8.`],
         [32, 40, `Recall, the share of real variety our digits cover, falls from ${pct(J['1'].recall)} at w = 1 to ${pct(J['8'].recall)} at 8. Every 7 drifts toward the typical 7.`],
         [40, 47, 'Remember the title? That was w = 1. At w = 2.5, each letter keeps only the strokes no other letter could claim.'],
         [47, 53.6, 'Small w: faithful but loose. Large w: obedient but samey. That trade-off is why every generator hands you this dial.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)) * (1 - ease(prog(t, 39.5, 0.8))), cell = 70, gap = 5, x0 = 220, y0 = 270;
    SWW.forEach((w, j) => { const a = a0 * eout(prog(t, 1 + j * 1.2, 0.6)), y = y0 + j * 132; M(`[a|w] = ${w}`, 110, y + 44, 24, { a }); digitRow(MN.sweep[w], `s${w}_`, 8, x0, y, cell, gap, a); });
    text('asking for 7 · each column: the same noise', x0, y0 + 5 * 132 + 10, { font: F.mono, size: 15, color: P.dust, a: a0 });
    // the title again: w = 1 against w = 2.5
    const ta = eout(prog(t, 40, 0.8)); if (ta > 0) { const sc = 150, cx = 450;
      [[TL.map(st => [st, st.S - 1]), 1, 400, A.title.cover[0]], [null, 2.5, 700, A.title.cover[1]]].forEach(([_, w, cy, cov], j) => { const a = ta * eout(prog(t, 40.3 + j * 1.6, 0.8));
        M(`[a|w] = ${w}`, 110, cy - 150, 24, { a }); text(`${pct(cov)} of each letter drawn`, 250, cy - 150, { font: F.mono, size: 15, color: j ? P.rose : P.sage, a });
        ctx.fillStyle = P.chalk; setA(a * 0.9);
        if (j === 0) TL.forEach(st => { for (let i = 0; i < st.n; i += 2) { const p = ptAt(st, i, st.S - 1); ctx.fillRect(cx + p[0] * sc - 1, cy - p[1] * sc - 1, 2, 2); } });
        else { const v = HARD; for (let i = 0; i < v.length; i += 4) ctx.fillRect(cx + v[i] / 4000 * sc - 1, cy - v[i + 1] / 4000 * sc - 1, 2, 2); }
        setA(1); }); }
    // three judges against w
    const charts = [[16, 'digit reader agrees with the request', 'acc', P.sage, 0, 1, v => pct(v)], [24, 'FID  (lower is better)', 'FID', P.rose, 0, Math.max(...JW.map(w => J[w].FID)) * 1.1, v => v.toFixed(1)],
                    [32, 'recall  (variety covered)', 'recall', P.ink, 0, 1, v => pct(v)]];
    charts.forEach(([t0, nm, key, col, lo, hi, fmt], i) => { const a = eout(prog(t, t0, 0.8)); if (a <= 0) return; const x0c = 900, x1c = 1780, yb = 405 + i * 225, H = 110, X = w => x0c + (x1c - x0c) * w / 8, Y = v => yb - H * (v - lo) / (hi - lo);
      card(830, yb - H - 60, 1000, H + 95, col, a); label(nm, 864, yb - H - 26, a, col); line(x0c, yb, x1c, yb, P.border, 1.5, a);
      [0, 1, 2, 4, 8].forEach(w => text(String(w), X(w), yb + 24, { font: F.mono, size: 13, color: P.dust, align: 'center', a }));
      const g = eout(prog(t, t0 + 0.3, 2)); setA(a); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath(); JW.forEach((w, k) => { const p = [X(w), Y(J[w][key])]; if (k / (JW.length - 1) > g + 1e-6) return; k ? ctx.lineTo(...p) : ctx.moveTo(...p); }); ctx.stroke(); setA(1);
      JW.forEach((w, k) => { if (k / (JW.length - 1) > g + 1e-6) return; dot(X(w), Y(J[w][key]), 5, col, a); });
      [0, 1, 4, 8].forEach(w => { if (J[w] && g > 0.95) text(fmt(J[w][key]), X(w), Y(J[w][key]) - 12, { font: F.mono, size: 13, color: P.stone, align: 'center', a }); });
      if (key === 'FID' && g > 0.95) { ring(X(JB), Y(J[JB].FID), 11, P.amber, a, 2); text(`best: w = ${JB}`, X(JB) + 14, Y(J[JB].FID) - 18, { font: F.mono, size: 14, color: P.amber, a }); } });
  } });

// 09 — saying no: negative requests
const FIN = k => { const st = FL[k]; return Array.from({ length: st.n }, (_, i) => ptAt(st, i, st.S - 1)); };
const ON3 = k => FIN(k).filter(p => nearestBlob(p) === 2);
const DIST2 = k => { const b2 = BLOB(1), pts = ON3(k); return pts.reduce((s, p) => s + Math.hypot(p[0] - b2[0], p[1] - b2[1]), 0) / Math.max(1, pts.length); };
const NEGS = memo('negs', () => ({ n: DIST2('negn'), g: DIST2('neg'), on: ON3('neg').length / FL.neg.n, on2: ON3('neg2').length / FL.neg2.n }));
SC.push({ dur: 40, chapter: 'saying no', title: 'Saying no', sub: 'Swap the no-request wind for a wind to avoid.',
  caps: [[1, 8, 'One more trick hides in the formula. Replace the wind with no request by the wind for something we do not want.'],
         [8, 16, 'Start from the wind toward what we avoid; push from it toward the request. At w = 1 the avoided wind drops out; above 1, it repels.'],
         [16, 24, `Ask the blobs for 3 while avoiding 2, at w = 1.25. ${pct(NEGS.on)} still land on blob 3, but on its far side, away from 2.`],
         [24, 31, `Turn it up to 2 and it overshoots: pushed so hard from blob 2 that ${NEGS.on2 < 0.05 ? 'every point flies' : 'most points fly'} right past blob 3.`],
         [31, 39.6, 'Image generators call this the negative prompt: ask for a portrait, and push away from “blurry”. Used gently, it works.']],
  draw(t) {
    const m = sq(110, 250, 620, 1.6), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const k = t < 16 ? 'negn' : t < 24 ? 'neg' : 'neg2', st = FL[k], f = (st.S - 1) * hopF(t, t < 16 ? 2 : t < 24 ? 16.3 : 24.3, 3);
    blobRings(m, a0, 2); if (t > 16) { const b = BLOB(1); ring(m.X(b[0]), m.Y(b[1]), 34, P.rose, a0, 3, [6, 5]); text('avoid', m.X(b[0]) + 40, m.Y(b[1]) - 10, { font: F.mono, size: 16, color: P.rose, a: a0 }); }
    flowDots(m, st, f, a0, 3);
    M(t < 16 ? 'ask [p|3],  [a|w] = 1.25' : t < 24 ? 'ask [p|3],  avoid [n|2],  [a|w] = 1.25' : 'ask [p|3],  avoid [n|2],  [a|w] = 2', m.x0, m.y0 + m.S + 34, 24, { a: a0 });
    if (t > 19 && t < 24) text(`average distance from blob 2: ${NEGS.n.toFixed(2)} → ${NEGS.g.toFixed(2)}`, m.x0, m.y0 + m.S + 56, { font: F.mono, size: 14, color: P.dust, a: eout(prog(t, 19, 0.8)) });
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 250, P.rose, c1); label('a negative request', 864, 296, c1, P.rose);
    M('[m|v] = [m|v][m_|neg] + [a|w] · ( [m|v][m_|y] − [m|v][m_|neg] )', 864, 364, 32, { a: eout(prog(t, 8, 0.8)) });
    M('[a|w] = 1:   [m|v] = [m|v][m_|y]      the negative cancels out', 864, 436, 22, { a: eout(prog(t, 11, 0.8)) });
    const c2 = eout(prog(t, 31, 0.8)); if (c2 > 0) { card(830, 520, 1000, 200, P.ink, c2); label('in image generators', 864, 566, c2, P.ink);
      M('prompt:  “a portrait”        negative prompt:  “blurry”', 864, 630, 24, { a: c2 }); text('same formula: the avoided wind comes from the words to avoid', 864, 680, { font: F.mono, size: 15, color: P.dust, a: c2 }); }
  } });

// 10 — in the wild
SC.push({ dur: 40, chapter: 'wild', title: 'Guidance, everywhere', sub: 'The dial behind today’s image, video and audio generators.',
  caps: [[1, 9, 'Classifier-free guidance became the default almost at once. OpenAI’s GLIDE, in 2021, found people preferred it to guiding with a separate model.'],
         [9, 18, 'Google’s Imagen, in 2022, turned w up high, and added a fix called dynamic thresholding to stop pixels from saturating.'],
         [18, 27, 'Stable Diffusion put the dial in front of everyone: the “guidance scale”, 7.5 by default, plus the negative prompt box.'],
         [27, 34, 'And FLUX.1 [dev], in 2024, was distilled to take w as an input: one network call per step does the work of two.'],
         [34, 39.6, 'One formula, from toy blobs to the models you use.']],
  draw(t) {
    const cards = [[1, 'glide · openai, 2021', P.terracotta, 'classifier-free guidance, for text', 'preferred by people over guiding with a separate model'],
      [9, 'imagen · google, 2022', P.plum, 'large w + dynamic thresholding', 'turn the dial up without burning out pixels'],
      [18, 'stable diffusion · 2022', P.sage, 'guidance scale 7.5 · negative prompts', 'the dial, in everyone’s hands'],
      [27, 'flux.1 [dev] · 2024', P.amber, 'guidance distillation', 'w becomes an input: one call per step']];
    cards.forEach(([t0, h, c, m1, s], i) => { const a = eout(prog(t, t0, 0.8)), x = 100 + (i % 2) * 880, y = 260 + Math.floor(i / 2) * 330;
      card(x, y, 840, 300, c, a); label(h, x + 32, y + 46, a, c); text(m1, x + 32, y + 140, { font: F.mono, size: 26, color: P.chalk, a }); para(s, x + 32, y + 206, 780, { size: 24, a: a * eout(prog(t, t0 + 1, 0.8)) }); });
  } });

// 11 — build it
const CODE = [
  '[k|# training: episode 10’s loss, but sometimes hide the request]',
  'y = [p|where]([a|rand](n) < [a|0.1], NULL, y)',
  'xt = ([a|1] - t)*z + t*x',
  'loss = [p|mean](([p|net](xt, t, y) - (x - z))**[a|2])',
  '',
  '[k|# sampling: two calls, one dial]',
  '[m|for] k [m|in] [p|range](K):',
  '    v_null = [p|net](x, k/K, NULL)',
  '    v_y    = [p|net](x, k/K, y)',
  '    x = x + ([a|1]/K) * (v_null + w * (v_y - v_null))',
  '',
  '[k|# a negative request: swap NULL for what to avoid]',
  '    v_neg  = [p|net](x, k/K, avoid)',
];
SC.push({ dur: 34, chapter: 'build', title: 'Build it yourself', sub: 'One line in training, two calls in sampling.',
  caps: [[1, 9, 'In training, one new line: hide the request one time in ten. In sampling, call the network twice and mix the answers.'],
         [9, 18, 'In the playground, ask for a blob or a digit, drag w from 0 to 8, and try a negative request.'],
         [18, 33.6, 'Recap: a request enters as an input. Bayes turns asking into adding arrows. The difference between two winds is a classifier for free, and w decides how hard to listen.']],
  draw(t) {
    playPill('ep11-guidance-lab.html', 9, t);
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1100, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 296 + i * 44, 19, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8)); card(1240, 240, 580, 690, P.sage, ra); label('the whole episode', 1276, 290, ra, P.sage);
    [['asking', 'p([d|x] | [p|y])'], ['Bayes, as arrows', '∇log p(x|y) = ∇log p(x) + ∇log p(y|x)'], ['classifier guidance', '[m|v][m_|∅] + [a|w]·(1 − t)/t·∇log p(y|xₜ)'], ['classifier-free', '[m|v][m_|∅] + [a|w]·([m|v][m_|y] − [m|v][m_|∅])'], ['the price', 'certainty up, variety down']].forEach(([nm, mm], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.8, 0.8)); text(nm, 1276, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(mm, 1276, 412 + i * 112, 19, { a: ai }); });
  } });

// 12 — every request was a number
const S2 = ['tokens', 'embeddings', 'memory', 'attention', 'position', 'the block', 'training', 'enc / dec', 'inside', 'beyond'];
const PROMPT = 'a seven, drawn boldly, in red ink';
SC.push({ dur: 31, chapter: 'prompts', title: 'Every request was a number', sub: 'Blob 3. The letter G. The digit 7. Real prompts are sentences.',
  caps: [[1, 8, 'Look back at every request today: blob 3, the letter G, the digit 7. Each was one choice from a short, fixed list.'],
         [8, 16, 'Real prompts are sentences. “A seven, drawn boldly, in red ink.” No list could give every possible sentence its own slot.'],
         [16, 24, 'So y must become a list of numbers that carries what the sentence means, where similar meanings get similar numbers.'],
         [24, 30.6, 'Turning words into meaning is reading. Our generators draw on request, but none of them can read a single word.']],
  draw(t) {
    {
      // the three requests from today, each a one-hot
      [['blob 3', 2, 8, P.sage], ['“G”', 0, 8, P.chalk], ['digit 7', 7, 10, P.terracotta]].forEach(([nm, k, n, c], i) => { const a = eout(prog(t, 1 + i * 1.2, 0.6)), y = 270 + i * 95;
        text(nm, 110, y + 30, { font: F.mono, size: 22, color: c, a }); oneHot(330, y + 6, k, n, a, P.ink, 32, 6, null, false); });
      // the prompt that doesn't fit
      const pa = eout(prog(t, 8, 0.6)); if (pa > 0) { const n = Math.floor(clamp((t - 8.5) / 3.5, 0, 1) * PROMPT.length), s = PROMPT.slice(0, n);
        rbox(110, 580, 1000, 80, 12, P.surface, P.ink, pa); text(s + (Math.floor(t * 2) % 2 && n < PROMPT.length ? '▌' : ''), 140, 632, { font: F.serif, italic: true, size: 34, color: P.chalk, a: pa });
        const ea = eout(prog(t, 12.5, 0.8)); if (ea > 0) { const r = rng(4); for (let i = 0; i < 44; i++) { const x = 130 + i * 22, on = i === 21; rbox(x, 700 + Math.sin(t * 2 + i) * 2, 16, 16, 3, on ? P.rose : P.surface, P.border, ea * (0.35 + 0.65 * r())); }
          text('a slot for every sentence?  there would be more slots than atoms', 130, 760, { font: F.mono, size: 16, color: P.rose, a: ea }); } }
      // sentence → meaning → a vector y
      const va = eout(prog(t, 16.5, 0.8)); if (va > 0) { card(1180, 250, 640, 520, P.ink, va); label('what y has to become', 1216, 296, va, P.ink);
        text('“a seven, drawn boldly”', 1216, 360, { font: F.serif, italic: true, size: 26, color: P.chalk, a: va });
        arrow(1500, 380, 1500, 440, P.ink, 2.5, va); rbox(1380, 450, 240, 70, 12, P.surface, P.plum, va); text('reading', 1500, 494, { font: F.serif, italic: true, size: 28, color: P.plum, align: 'center', a: va });
        arrow(1500, 530, 1500, 580, P.ink, 2.5, va); const r = rng(9); for (let i = 0; i < 24; i++) { const h = 10 + 60 * r(); setA(va * eout(prog(t, 18 + i * 0.05, 0.4))); ctx.fillStyle = r() > 0.5 ? P.ink : P.sage; ctx.fillRect(1240 + i * 22, 680 - h, 16, h); setA(1); }
        text('a vector of meaning: similar sentences, similar numbers', 1216, 740, { font: F.mono, size: 14, color: P.dust, a: va }); text('?', 1650, 494, { font: F.serif, italic: true, size: 44, color: P.amber, a: va * (0.6 + 0.4 * Math.sin(t * 3)) }); }
    }
  } });

// 13 — the end of Season 1, and the hook into Season 2
SC.push({ dur: 31, chapter: 'next',
  caps: [[0.5, 8, 'That closes Season 1. We asked what it means to learn p(x), built machine after machine to draw from it, and learned to ask.'],
         [8, 16, 'Season 2: machines that read. How a network turns text into meaning, and learns to predict the next word.'],
         [16, 23, 'It starts where every sentence starts: with tokens.'],
         [23.5, 30.6, 'Before a machine can draw what we say, it must learn to read.']],
  draw(t) {
    t += 31;          // the timings below were laid out on the combined clock
    // Season 1, complete
    const sa = eout(prog(t, 31, 1)) * (1 - ease(prog(t, 46.5, 1)));
    if (sa > 0) { text('SEASON 1', 960, 380, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: sa, ls: 6 }); text('Generative modelling', 960, 460, { font: F.serif, size: 64, color: P.chalk, align: 'center', a: sa });
      text('how do you turn noise into data?', 960, 520, { font: F.serif, italic: true, size: 30, color: P.stone, align: 'center', a: sa * eout(prog(t, 32, 1)) }); GA = sa; seasonStrip(11, 33, t, 680); GA = 1;
      text('complete', 960, 800, { font: F.mono, size: 18, color: P.sage, align: 'center', a: sa * eout(prog(t, 36.5, 0.8)), ls: 5 }); }
    // Season 2, next
    const na = eout(prog(t, 47, 1.2));
    if (na > 0) { text('Before a machine can draw what we say,', 960, 330, { font: F.serif, size: 56, color: P.chalk, align: 'center', a: na * eout(prog(t, 54.5, 1)) });
      text('it must learn to read.', 960, 410, { font: F.serif, size: 56, color: P.chalk, align: 'center', a: na * eout(prog(t, 55.3, 1)) });
      text('NEXT  ·  SEASON 2', 960, 520, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: na, ls: 4 });
      text('Machines that read', 960, 590, { font: F.serif, italic: true, size: 50, color: P.stone, align: 'center', a: na * eout(prog(t, 47.5, 1)) });
      const x0 = 520, x1 = 1400, y = 740, X = i => x0 + (x1 - x0) * i / (S2.length - 1); line(x0, y, x1, y, P.border, 1.5, na);
      S2.forEach((nm, i) => { const u = eout(prog(t, 48.5 + i * 0.12, 0.4)), first = i === 0, pl = 0.5 + 0.5 * Math.sin((t - 48) * 4.5);
        if (first) ring(X(i), y, 10 + 5 * pl, P.amber, na * u * (0.8 - 0.6 * pl), 1.5); dot(X(i), y, first ? 6 : 4, first ? P.amber : P.dust, na * u);
        text(String(i + 1).padStart(2, '0'), X(i), y + 30, { font: F.mono, size: 12, color: first ? P.amber : P.dust, align: 'center', a: na * u }); text(nm, X(i), y + 50, { size: 14, color: first ? P.amber : P.dust, align: 'center', a: na * u }); }); }
  } });
