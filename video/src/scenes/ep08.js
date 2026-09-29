// ─────────────────────────────────────────────────────────────
//  Real noise-conditional score networks (exported by code/export_ep08_assets.py)
// ─────────────────────────────────────────────────────────────
const A = window.EP8, SIG = A.sigmas, NS = A.ann.length - 1, GN = A.grid.n, GL = A.grid.L, WR = 4;
function square(x0, y0, S) { return plotSquare(x0, y0, S, WR); }
function arr(x1, y1, x2, y2, col, lw, a, hd = 5) { if (GA * a <= 0.002) return; const L = Math.hypot(x2 - x1, y2 - y1); if (L < 0.5) return; setA(a); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const an = Math.atan2(y2 - y1, x2 - x1), h = Math.min(hd, L * 0.45); ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - h * Math.cos(an - 0.45), y2 - h * Math.sin(an - 0.45)); ctx.lineTo(x2 - h * Math.cos(an + 0.45), y2 - h * Math.sin(an + 0.45)); ctx.fill(); setA(1); }
const gpt = k => [-GL + (k % GN) * 2 * GL / (GN - 1), GL - Math.floor(k / GN) * 2 * GL / (GN - 1)];
// an arrow field: F[k] = σ·s at grid point k (≈ −ε: the same scale at every noise level)
function field(m, F, a, col = P.sage, len = 26, G = null) { if (GA * a <= 0.002) return; clipTo(m, () => F.forEach((v, k) => { const [x, y] = gpt(k), n = Math.hypot(v[0], v[1]); if (n < 1e-6) return;
  const L = len * Math.tanh(n / 1.2), dx = v[0] / n * L, dy = -v[1] / n * L, X = m.X(x), Y = m.Y(y); arr(X - dx / 2, Y - dy / 2, X + dx / 2, Y + dy / 2, col, 1.6, a * (G ? G(k) : 0.85)); })); }
function lerpField(F0, F1, u) { return F0.map((v, k) => [lerp(v[0], F1[k][0], u), lerp(v[1], F1[k][1], u)]); }
function stagePts(S, f) { const n = S.length - 1, fi = clamp(f, 0, n), i = Math.min(n - 1, Math.floor(fi)), u = fi - i; return S[i].map((p, k) => [lerp(p[0], S[i + 1][k][0], u), lerp(p[1], S[i + 1][k][1], u)]); }
function particles(m, S, f, col, a, r = 3, count = 1e9) { const pts = stagePts(S, f); clipTo(m, () => pts.forEach((p, k) => { if (k < count) marble(m.X(p[0]), m.Y(p[1]), r, col, a); })); }
const levelOf = f => clamp(Math.floor((clamp(f, 0, NS) - 1e-9) / (A.T / A.keep)), 0, SIG.length - 1);
function fieldAtLevel(fl) { const i = clamp(Math.floor(fl), 0, SIG.length - 1), j = Math.min(SIG.length - 1, i + 1); return lerpField(A.fields[i], A.fields[j], ease(clamp(fl - i))); }
const errCanvas = (() => { let c = null; return () => { if (c) return c; const n = A.err.n, u = b64bytes(A.err.u8); c = document.createElement('canvas'); c.width = c.height = n; const g = c.getContext('2d'), im = g.createImageData(n, n), rgb = hx(P.rose);
  for (let k = 0; k < n * n; k++) { im.data[k * 4] = rgb[0]; im.data[k * 4 + 1] = rgb[1]; im.data[k * 4 + 2] = rgb[2]; im.data[k * 4 + 3] = Math.round(Math.pow(u[k] / 255, 0.7) * 235); } g.putImageData(im, 0, 0); return c; }; })();
// seeded normals for the scene-local simulations
const pct = v => (v * 100).toFixed(0) + '%';

// 1-D pieces
const N1 = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(TAU));
const MIX = [[0.65, -1.3, 0.55], [0.35, 1.5, 0.45]];
const p1 = x => MIX.reduce((a, [w, m, s]) => a + w * N1(x, m, s), 0);
const dp1 = x => MIX.reduce((a, [w, m, s]) => a + w * N1(x, m, s) * (-(x - m) / (s * s)), 0);
const s1 = x => dp1(x) / p1(x);
function plot(fn, x0, x1, X, Y, col, lw, a, fill = null, base = null, n = 240) { if (GA * a <= 0.002) return; const path = () => { ctx.beginPath(); for (let i = 0; i <= n; i++) { const x = x0 + (x1 - x0) * i / n; i ? ctx.lineTo(X(x), Y(fn(x))) : ctx.moveTo(X(x), Y(fn(x))); } };
  if (fill) { path(); ctx.lineTo(X(x1), base); ctx.lineTo(X(x0), base); ctx.closePath(); setA(a * fill[1]); ctx.fillStyle = fill[0]; ctx.fill(); }
  path(); setA(a); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke(); setA(1); }

// scene 02: three Langevin runs on N(0, 1) with kick multipliers 0, 1, 2
const BAL = (() => { const r = gaussian(rng(81)), u = rng(82), n = 500, T = 200, eta = 0.05, out = [0, 1, 2].map(() => new Float32Array(n * (T + 1)));
  [0, 1, 2].forEach((c, ci) => { const X = out[ci]; for (let k = 0; k < n; k++) X[k] = (u() * 2 - 1) * 3.2;
    for (let t = 1; t <= T; t++) for (let k = 0; k < n; k++) X[t * n + k] = X[(t - 1) * n + k] * (1 - eta) + c * Math.sqrt(2 * eta) * r(); });
  return { n, T, eta, X: out }; })();
const balVar = (ci, t) => { const n = BAL.n, X = BAL.X[ci]; let s = 0; for (let k = 0; k < n; k++) s += X[t * n + k] ** 2; return s / n; };

// scene 04: fit a bell-curve score by implicit score matching:  L(μ, v) = mean((x − μ)²)/v² − 2/v
const HYV = (() => { const r = gaussian(rng(41)), xs = Array.from({ length: 300 }, () => 1.2 + 0.7 * r()), mean = xs.reduce((a, b) => a + b, 0) / 300, vr = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / 300;
  let mu = -1.5, v = 2.2; const path = [[mu, v]]; for (let i = 0; i < 160; i++) { const m2 = xs.reduce((a, b) => a + (b - mu) ** 2, 0) / 300; mu += 0.06 * (mean - mu); v -= 0.05 * (2 * v - 2 * m2); path.push([mu, v]); }
  const loss = (mu, v) => xs.reduce((a, b) => a + (b - mu) ** 2, 0) / 300 / (v * v) - 2 / v; return { xs, mean, vr, path, loss }; })();

// scene 05: one noisy point, and the posterior-weighted arrows home
const DSMP = [1.05, 0.55], DSMS = 0.55;
const DSMW = (() => { const w = A.data.map(p => Math.exp(-((p[0] - DSMP[0]) ** 2 + (p[1] - DSMP[1]) ** 2) / (2 * DSMS * DSMS))), t = w.reduce((a, b) => a + b, 0); return w.map(v => v / t); })();
const DSMAVG = A.data.reduce((a, p, i) => [a[0] + DSMW[i] * (p[0] - DSMP[0]), a[1] + DSMW[i] * (p[1] - DSMP[1])], [0, 0]);   // = σ²·∇ log p_σ(x̃) on the samples
const SHAKE = (() => { const r = gaussian(rng(55)); return Array.from({ length: 14 }, (_, i) => ({ x: A.data[i * 27 % 400], e: [r(), r()] })); })();

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open
SC.push({ dur: 28, chapter: 'intro',
  caps: [[0.6, 6, 'Last time we sculpted a landscape, then used only its slope. So here are just the arrows.'],
         [6, 12, 'Drop particles anywhere. Each one follows the arrows, with a small random kick at every step.'],
         [12, 18.5, 'As the noise level falls the arrows change: first a broad pull, then sharp detail. The particles land on the data.'],
         [18.5, 27.6, 'Today: the score. How to learn it from data alone, how to sample with it, and the last idea before diffusion models.']],
  draw(t) {
    const dim = lerp(1, 0.14, ease(prog(t, 18.3, 1.4))), m = square(560, 170, 800); GA = dim;
    const f = NS * prog(t, 3.5, 13.5), fl = t < 3.5 ? 0 : clamp(f / (A.T / A.keep), 0, SIG.length - 1);
    frame(m, eout(prog(t, 0.3, 1))); field(m, fieldAtLevel(fl), eout(prog(t, 0.5, 2)), P.sage, 30);
    particles(m, A.ann, f, P.terracotta, eout(prog(t, 3, 0.6)), 3.2);
    const sa = eout(prog(t, 3.5, 0.6)); if (sa > 0) pill(m.x0 + m.S - 20, m.y0 + 30, `σ = ${SIG[levelOf(f)].toFixed(2)}`, P.amber, sa, { align: 'right', size: 20 });
    GA = 1;
    const a = eout(prog(t, 18.8, 1.2));
    text('EPISODE 08', 960, 430, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('Score Matching & Langevin', 960, 550 + (1 - a) * 16, { font: F.serif, size: 112, color: P.chalk, align: 'center', a, ls: -1 });
    text('follow the arrows', 960, 622, { size: 34, color: P.stone, align: 'center', a: eout(prog(t, 19.4, 1)) });
    const tg = [[P.terracotta, 'x  data'], [P.sage, 's(x)  score'], [P.rose, 'ε  noise'], [P.amber, 'σ  noise level'], [P.plum, 'η  step'], [P.ink, 'p  density']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let cx = 960 - tw / 2; tg.forEach(([c, s], i) => { cx += tag(cx, 690, c, s, eout(prog(t, 20 + i * 0.3, 0.6)), 21) + 14; });
  } });

// 01 — what is a score?
SC.push({ dur: 50, chapter: 'score', title: 'The score: an arrow at every point', sub: 'The slope of log p, and why Z never shows up.',
  caps: [[1, 8, 'Start with a density p(x). Take its log: the peaks stay peaks, and the tiny tails turn into gentle, usable slopes.'],
         [8, 15, 'At any x, the slope of log p is the score: s(x) = d/dx log p(x). Positive means “more probable to the right”.'],
         [15, 22, 'Draw it as an arrow at every point. The arrows point toward likely data, and grow longer the further out you go.'],
         [22, 29, 'Another reading: s = p′(x) / p(x), how fast p grows as a fraction of itself. A relative slope.'],
         [29, 36, 'And Z? log p = log p̃ − log Z; Z is constant, so its slope is zero. The score never needs normalising.'],
         [36, 43, 'For a bell curve, the score is −(x − μ)/σ²: a spring pulling back to the centre, stiffer when the bell is narrow.'],
         [43, 49.6, 'In two dimensions the score is a gradient: an arrow field over the plane. That field is what we want to learn.']],
  draw(t) {
    const X0 = 140, X1 = 1040, X = x => X0 + (x + 4) / 8 * (X1 - X0), a0 = eout(prog(t, 0.6, 0.8));
    const Yp = v => 440 - v / 0.5 * 170, Yl = v => 540 + (-v) / 17 * 190;
    line(X0, 440, X1, 440, P.borderLight, 1.5, a0); plot(p1, -4, 4, X, Yp, P.ink, 3.5, a0, [P.ink, 0.14], 440); text('p(x)', X0, 262, { font: F.mono, size: 20, color: P.ink, a: a0 });
    const la = eout(prog(t, 5, 1)); if (la > 0) { line(X0, 540, X1, 540, P.border, 1, la, [4, 6]); plot(x => Math.max(-17, Math.log(p1(x))), -4, 4, X, Yl, P.plum, 3, la); text('log p(x)', X0, 520, { font: F.mono, size: 20, color: P.plum, a: la }); }
    // a probe sliding along, with the tangent of log p
    const pa = eout(prog(t, 8, 0.8)), xp = t < 15 ? -2.6 + 1.9 * ease(prog(t, 9, 5)) : -0.7 + 2.9 * Math.sin((t - 15) * 0.22);
    if (pa > 0) { const lp = Math.log(p1(xp)), sl = s1(xp), px = X(xp), py = Yl(lp), dx = 1.1;
      const tx = X(xp + dx) - X(xp - dx), ty = Yl(lp + sl * dx) - Yl(lp - sl * dx), tl = Math.hypot(tx, ty) || 1; line(px - tx / tl * 120, py - ty / tl * 120, px + tx / tl * 120, py + ty / tl * 120, P.amber, 2.5, pa); dot(px, py, 7, P.amber, pa); line(px, Yp(p1(xp)), px, 760, P.dust, 1, pa * 0.5, [3, 5]);
      text(`slope = ${sl.toFixed(2)}`, px + 14, py - 16, { font: F.mono, size: 17, color: P.amber, a: pa }); }
    // the score as arrows on a line
    const sa = eout(prog(t, 15, 1)); if (sa > 0) { const y = 830; line(X0, y, X1, y, P.borderLight, 1.5, sa); text('score s(x) as arrows', X0, y + 48, { font: F.mono, size: 17, color: P.sage, a: sa });
      for (let i = 0; i <= 20; i++) { const x = -4 + i * 0.4, s = s1(x), L = 62 * Math.tanh(s / 4); arr(X(x) - L / 2, y, X(x) + L / 2, y, P.sage, 3.2, sa * eout(prog(t, 15 + i * 0.08, 0.4)), 10); }
      if (pa > 0) marble(X(xp), 830, 7, P.amber, pa); }
    // cards
    const c1 = eout(prog(t, 8.5, 0.8)); card(1100, 250, 730, 170, P.sage, c1); label('the score', 1134, 296, c1, P.sage);
    M('[s|s]([d|x]) = d/d[d|x] log [p|p]([d|x]) = [p|p]′([d|x]) / [p|p]([d|x])', 1134, 360, 28, { a: c1 }); text('a relative slope: growth as a fraction of p itself', 1134, 400, { font: F.mono, size: 15, color: P.dust, a: eout(prog(t, 22, 0.8)) });
    const c2 = eout(prog(t, 29, 0.8)); card(1100, 440, 730, 170, P.amber, c2); label('no z', 1134, 486, c2, P.amber);
    M('log [p|p] = log [p|p̃] − log [a|Z]   ⇒   [s|s] = ∇ log [p|p̃] − [a|0]', 1134, 548, 24, { a: c2 }); text('any unnormalised p̃ gives the exact score', 1134, 588, { font: F.mono, size: 15, color: P.dust, a: c2 });
    const c3 = eout(prog(t, 36, 0.8)); card(1100, 630, 730, 170, P.ink, c3); label('a bell curve', 1134, 676, c3, P.ink);
    M('[s|s]([d|x]) = −([d|x] − μ) / [a|σ]²', 1134, 740, 30, { a: c3 }); text('a spring to the centre, stiffer when σ is small', 1134, 780, { font: F.mono, size: 15, color: P.dust, a: c3 });
    const c4 = eout(prog(t, 43, 0.8)); if (c4 > 0) { card(1100, 820, 730, 110, P.plum, c4); M('2-D:  [s|s]([d|x]) = ∇ₓ log [p|p]([d|x])  — a vector field', 1134, 885, 24, { a: c4 }); }
  } });

// 02 — why the jiggle is √(2η)
SC.push({ dur: 58, chapter: 'balance', title: 'Why the jiggle is √(2η)', sub: 'A pull and a kick, in perfect balance.',
  caps: [[1, 8, 'Last episode asked: why is the kick exactly √(2η)? Test it on the simplest target: a bell curve with variance 1.'],
         [8, 15, 'Its score is −x: each step shrinks x by (1 − η), then kicks. Three runs: no kick, √(2η), and double.'],
         [15, 24, 'No kick: everything collapses onto the peak. Double kick: the cloud spreads four times too wide. The √(2η) kick lands exactly on the bell.'],
         [24, 32, 'Why? Track the variance. The pull multiplies it by (1 − η)², losing about 2η per step. The kick adds back exactly 2η. Balance at 1.'],
         [32, 40, 'The same balance holds for any p, in any dimension. The arrows push probability uphill; the jiggle spreads it back out.'],
         [40, 48, 'The uphill flow is p·s = p·∇log p = ∇p. The spreading flow is −∇p. They cancel exactly, so p stops changing: it is the stationary distribution.'],
         [48, 57.6, 'So Langevin needs only the score: no energy, no Z. If we can learn the arrows, we can sample.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), st = Math.round(BAL.T * clamp((t - 9) / 14) ** 0.8), X0 = 130, X1 = 990, X = x => X0 + (x + 4) / 8 * (X1 - X0);
    [['no kick', 0, P.rose], ['kick √(2η)', 1, P.sage], ['kick 2·√(2η)', 2, P.rose]].forEach(([nm, ci, col], r) => {
      const base = 400 + r * 225, a = a0 * eout(prog(t, 8 + r * 0.5, 0.8)), n = BAL.n, Xs = BAL.X[ci]; if (a <= 0) return;
      line(X0, base, X1, base, P.borderLight, 1.5, a); plot(x => N1(x, 0, 1), -4, 4, X, v => base - v * 330, P.ink, 2.5, a * 0.9, [P.ink, 0.1], base);
      const nb = 48, bw = 8 / nb, cnt = new Array(nb).fill(0); for (let k = 0; k < n; k++) { const x = Xs[st * n + k]; if (x > -4 && x < 4) cnt[Math.floor((x + 4) / bw)]++; }
      cnt.forEach((c, i) => { const h = Math.min(170, c / n / bw * 330); if (c) { setA(a * 0.5); ctx.fillStyle = col; ctx.fillRect(X(-4 + i * bw) + 1, base - h, X(-4 + bw) - X(-4) - 2, h); setA(1); } });
      for (let k = 0; k < 60; k++) { const x = Xs[st * n + k]; if (x > -4 && x < 4) dot(X(x), base + 14, 3, P.terracotta, a * 0.8); }
      text(nm, X0, base - 160, { font: F.mono, size: 18, color: col, a }); text(`variance ${balVar(ci, st).toFixed(2)}`, X1, base - 160, { font: F.mono, size: 18, color: P.chalk, align: 'right', a });
      if (t > 23) text(ci === 1 ? '✓ matches the bell' : ci === 0 ? '✗ collapsed' : '✗ four times too wide', X1, base - 134, { font: F.mono, size: 15, color: col, align: 'right', a: a * eout(prog(t, 23, 0.6)) });
    });
    const c1 = eout(prog(t, 8, 0.8)) * (1 - ease(prog(t, 31.5, 1)));
    card(1060, 250, 770, 330, P.plum, c1); label('one langevin step on n(0, 1)', 1094, 296, c1, P.plum);
    M('[d|x]′ = [d|x] + η·[s|(−x)] + √(2η)·[n|ε]', 1094, 360, 28, { a: c1 }); M('   = (1 − η)·[d|x] + √(2η)·[n|ε]', 1094, 412, 28, { a: c1 });
    M('Var′ = (1 − η)²·Var + 2η', 1094, 480, 28, { a: c1 * eout(prog(t, 24, 0.8)) });
    M('fixed point:  Var = 2η / (1 − (1 − η)²) = 2/(2 − η) ≈ [s|1]', 1094, 540, 22, { a: c1 * eout(prog(t, 26, 0.8)) });
    const c2 = eout(prog(t, 32, 0.8));
    if (c2 > 0) { card(1060, 250, 770, 470, P.sage, c2); label('any p, any dimension', 1094, 296, c2, P.sage);
      M('uphill flow:     [p|p]·[s|s] = [p|p]·∇log [p|p] = ∇[p|p]', 1094, 372, 24, { a: c2 });
      M('spreading flow:  −∇[p|p]', 1094, 426, 24, { a: eout(prog(t, 36, 0.8)) });
      M('∂[p|p]/∂t = −∇·([p|p]·[s|s]) + Δ[p|p] = −Δ[p|p] + Δ[p|p] = [s|0]', 1094, 500, 24, { a: eout(prog(t, 40, 0.8)) });
      text('(the Fokker–Planck equation)', 1094, 540, { font: F.mono, size: 15, color: P.dust, a: eout(prog(t, 41, 0.8)) });
      para('The kick of size √(2η) is exactly what makes the two flows cancel at p. Any other size settles somewhere else.', 1094, 600, 700, { size: 23, a: eout(prog(t, 44, 0.8)) }); }
    const c3 = eout(prog(t, 48, 0.8)); if (c3 > 0) { card(1060, 750, 770, 150, P.amber, c3); text('needs only the score — no E, no Z', 1094, 840, { font: F.serif, italic: true, size: 32, color: P.amber, a: c3 }); }
  } });

// 03 — learning arrows we can't see
SC.push({ dur: 40, chapter: 'the problem', title: 'Learning arrows we can’t see', sub: 'We have samples, not scores.',
  caps: [[1, 8, 'So: learn a network sθ(x) with an arrow at every x, matching the true arrows on average over the data.'],
         [8, 16, 'Minimise E‖sθ(x) − ∇ₓ log p(x)‖². But the true arrows are exactly what we don’t know. We only have samples.'],
         [16, 24, 'It’s like being asked to copy a map nobody will show you. Two tricks get around it.'],
         [24, 32, 'The first, from Aapo Hyvärinen in 2005, removes the unknown term with calculus. The second, from Pascal Vincent in 2011, uses noise.'],
         [32, 39.6, 'Both give a loss you can compute from samples alone.']],
  draw(t) {
    const m = square(120, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    clipTo(m, () => A.data.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.6, P.terracotta, a0 * 0.85)));
    const qa = eout(prog(t, 9, 0.8)); if (qa > 0) clipTo(m, () => A.fields[A.mid].forEach((v, k) => { if (k % 2) return; const [x, y] = gpt(k); text('?', m.X(x), m.Y(y) + 6, { font: F.mono, size: 16, color: P.rose, align: 'center', a: qa * 0.6 }); }));
    text('what we have: samples  ·  what we need: arrows', 120, 925, { font: F.mono, size: 16, color: P.dust, a: a0 });
    const c1 = eout(prog(t, 1, 0.8)); card(840, 250, 990, 260, P.sage, c1); label('the goal (fisher divergence)', 874, 296, c1, P.sage);
    const w1 = M('min  [kB|E][k_|x∼p] ‖ [s|s][k_|θ]([d|x]) − ', 874, 380, 32, { a: c1 }), w2 = M('[n|∇ₓ log p(x)]', 874 + w1, 380, 32, { a: c1 }); M(' ‖²', 874 + w1 + w2, 380, 32, { a: c1 });
    const ua = eout(prog(t, 9, 0.6)); if (ua > 0) { rbox(874 + w1 - 8, 344, w2 + 16, 52, 12, rgba(P.rose, 0.1), P.rose, ua, 2); pill(874 + w1 + w2 / 2, 436, 'unknown!', P.rose, ua, { align: 'center', size: 22 }); }
    const tr = [[24, 'trick 1 · Hyvärinen, 2005', 'integrate by parts: the unknown term cancels', P.amber], [27, 'trick 2 · Vincent, 2011', 'add noise: then we know the way home', P.ink]];
    tr.forEach(([t0, h, s, c], i) => { const a = eout(prog(t, t0, 0.8)); card(840, 540 + i * 190, 990, 160, c, a); label(h, 874, 586 + i * 190, a, c); text(s, 874, 640 + i * 190, { size: 28, color: P.chalk, a }); });
  } });

// 04 — Hyvärinen: integrate by parts
SC.push({ dur: 54, chapter: 'hyvärinen', title: 'Trick 1: integrate by parts', sub: 'The unknown score cancels out.',
  caps: [[1, 8, 'Write the goal out in one dimension. The squared error splits into three averages; the last doesn’t depend on θ.'],
         [8, 16, 'The troublesome middle term: p times sθ times p′/p. The p’s cancel, leaving sθ times p′.'],
         [16, 24, 'Integrate by parts: move the derivative from p onto sθ. The boundary term vanishes because p fades to zero far away.'],
         [24, 32, 'What’s left is the average of sθ² + 2·sθ′ over the data. No true score anywhere: only samples.'],
         [32, 42, 'Try it: fit a bell-curve score to 300 samples using this loss alone. It lands exactly on their mean and variance.'],
         [42, 53.6, 'The catch: in D dimensions, sθ′ becomes the trace of a D × D Jacobian, one backward pass per dimension. For a 256 × 256 colour image, about 200,000 passes per step.']],
  draw(t) {
    const x0 = 110, a0 = eout(prog(t, 0.6, 0.8));
    const rows = [[1, 'J(θ) = [kB|E][k_|p][(s[k_|θ] − [n|s])²] = [kB|E][k_|p][s[k_|θ]²] − 2 [kB|E][k_|p][s[k_|θ] [n|s]] + [kB|E][k_|p][[n|s]²]'],
      [8, '[kB|E][k_|p][s[k_|θ] [n|s]] = [pB|∫] [p|p] · s[k_|θ] · ([p|p]′/[p|p]) dx = [pB|∫] s[k_|θ] · [p|p]′ dx'],
      [16, '          = [s[k_|θ] [p|p]] − [pB|∫] s[k_|θ]′ · [p|p] dx = − [kB|E][k_|p][s[k_|θ]′]'],
      [24, 'J(θ) = [kB|E][k_|p][ s[k_|θ](x)² + 2·s[k_|θ]′(x) ] + const']];
    rows.forEach(([t0, s], i) => M(s, x0, 320 + i * 96, 27, { a: a0 * eout(prog(t, t0, 0.8)) }));
    const ka = eout(prog(t, 9.5, 0.6)); if (ka > 0) { const w1 = M('[kB|E][k_|p][s[k_|θ] [n|s]] = [pB|∫] [p|p] · s[k_|θ] · (', x0, 416, 27, { a: 0 }); line(x0 + w1 + 8, 424, x0 + w1 + 40, 396, P.rose, 3, ka); }
    const ba = eout(prog(t, 18, 0.6)); if (ba > 0) text('= 0 (p → 0 at ±∞)', x0 + 250, 548, { font: F.mono, size: 15, color: P.rose, a: ba });
    const ga = eout(prog(t, 25, 0.6)); if (ga > 0) { rbox(x0 - 14, 572, 820, 54, 12, rgba(P.sage, 0.08), P.sage, ga, 2); pill(x0 + 830, 599, 'samples only', P.sage, ga, { size: 20 }); }
    const da = eout(prog(t, 42, 0.8)); if (da > 0) { card(100, 700, 1000, 190, P.rose, da); label('in d dimensions', 134, 746, da, P.rose);
      M('[kB|E][k_|p][ ‖s[k_|θ]‖² + 2·tr ∇ₓs[k_|θ] ]:  tr = Σ over D diagonal entries', 134, 806, 24, { a: da }); text('D backward passes per step: fine for 2-D, hopeless for images', 134, 856, { font: F.mono, size: 17, color: P.rose, a: eout(prog(t, 45, 0.8)) }); }
    // live demo: fit s(x) = −(x − μ)/v with the implicit loss
    const ea = eout(prog(t, 31.5, 1));
    if (ea > 0) { const bx = 1180, bw = 650, X = x => bx + (x + 2) / 6 * bw, base = 620, it = Math.round((HYV.path.length - 1) * ease(prog(t, 33, 8))), [mu, v] = HYV.path[it];
      card(1150, 250, 710, 640, P.amber, ea); label('live: fit with J(θ) alone', 1184, 296, ea, P.amber);
      const nb = 30, bw1 = 6 / nb, cnt = new Array(nb).fill(0); HYV.xs.forEach(x => { if (x > -2 && x < 4) cnt[Math.floor((x + 2) / bw1)]++; });
      cnt.forEach((c, i) => { const h = c / 300 / bw1 * 330; setA(ea * 0.45); ctx.fillStyle = P.terracotta; ctx.fillRect(X(-2 + i * bw1) + 1, base - h, X(-2 + bw1) - X(-2) - 2, h); setA(1); });
      plot(x => N1(x, mu, Math.sqrt(v)), -2, 4, X, y => base - y * 330, P.ink, 3, ea); line(bx, base, bx + bw, base, P.borderLight, 1.5, ea);
      for (let i = 0; i <= 12; i++) { const x = -2 + i * 0.5, s = -(x - mu) / v, L = 26 * Math.tanh(s / 2); arr(X(x), base + 34, X(x) + L, base + 34, P.sage, 2, ea, 6); }
      text(`step ${it}`, 1184, 720, { font: F.mono, size: 18, color: P.chalk, a: ea });
      text(`μ = ${mu.toFixed(3)}   (sample mean ${HYV.mean.toFixed(3)})`, 1184, 762, { font: F.mono, size: 18, color: P.amber, a: ea });
      text(`v = ${v.toFixed(3)}   (sample variance ${HYV.vr.toFixed(3)})`, 1184, 800, { font: F.mono, size: 18, color: P.amber, a: ea });
      text(`J = ${HYV.loss(mu, v).toFixed(3)}`, 1184, 842, { font: F.mono, size: 18, color: P.sage, a: ea }); }
  } });

// 05 — denoising score matching
SC.push({ dur: 58, chapter: 'denoising', title: 'Trick 2: add noise, point home', sub: 'Denoising score matching.',
  caps: [[1, 8, 'Take a clean data point x and shake it: x̃ = x + σε. Now we know something exact: the way home.'],
         [8, 15, 'Home is (x − x̃)/σ² = −ε/σ. That is exactly the score of the little noise bell centred on x.'],
         [15, 23, 'So train the network to point home: minimise ‖sθ(x̃) + ε/σ‖². Multiply through by σ and it simply predicts the noise that was added.'],
         [23, 31, 'But a noisy point could have come from many clean points. The network can’t know which, so it learns the average of all their arrows home.'],
         [31, 39, 'Here is one noisy point, with an arrow to every data point that could have produced it, weighted by how likely. Their average is the bold arrow.'],
         [39, 47, 'That average is exactly the score of the noisy data, ∇ log pσ. The unknown score, recovered by an ordinary regression.'],
         [47, 57.6, 'Flip it around: x̃ + σ²·s(x̃) is the best guess of the clean point. The score is a denoiser. Remember that: it is the key to episode 9.']],
  draw(t) {
    const m = square(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    clipTo(m, () => A.data.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.4, P.terracotta, a0 * 0.7)));
    // shaken copies with their way home
    const sa = eout(prog(t, 1.5, 0.8)) * (1 - ease(prog(t, 30, 1)));
    if (sa > 0) SHAKE.forEach(({ x, e }, i) => { const ph = clamp((t - 2 - i * 0.12) / 1.2), sg = 0.45 * (1 + 0.15 * Math.sin(t * 0.8 + i)), xn = [x[0] + sg * e[0] * ease(ph), x[1] + sg * e[1] * ease(ph)];
      arr(m.X(x[0]), m.Y(x[1]), m.X(xn[0]), m.Y(xn[1]), P.rose, 1.6, sa * ph * 0.8, 5); marble(m.X(xn[0]), m.Y(xn[1]), 4.5, P.rose, sa * ph);
      const ha = sa * eout(prog(t, 8 + i * 0.1, 0.5)); arr(m.X(xn[0]), m.Y(xn[1]), m.X(xn[0] + (x[0] - xn[0]) * 0.8), m.Y(xn[1] + (x[1] - xn[1]) * 0.8), P.sage, 2.4, ha, 7); });
    // one noisy point: posterior-weighted arrows home and their average
    const pa = eout(prog(t, 30.5, 1));
    if (pa > 0) { const X = m.X(DSMP[0]), Y = m.Y(DSMP[1]), mx = Math.max(...DSMW);
      A.data.forEach((p, i) => { const w = DSMW[i] / mx; if (w > 0.02) line(X, Y, m.X(p[0]), m.Y(p[1]), P.sage, 1.2, pa * w * 0.7); });
      const av = eout(prog(t, 34, 0.8)); arr(X, Y, m.X(DSMP[0] + DSMAVG[0]), m.Y(DSMP[1] + DSMAVG[1]), P.sage, 5, pa * av, 14); marble(X, Y, 8, P.rose, pa);
      if (t > 47) { const d = eout(prog(t, 47.5, 0.8)); ring(m.X(DSMP[0] + DSMAVG[0]), m.Y(DSMP[1] + DSMAVG[1]), 12, P.amber, d, 2.5); pill(m.X(DSMP[0] + DSMAVG[0]) - 18, m.Y(DSMP[1] + DSMAVG[1]) + 40, 'x̃ + σ²·s(x̃): the denoised guess', P.amber, d, { size: 17, align: 'right' }); } }
    text(t < 30 ? `shaken with σ ≈ 0.45 · rose: noise ε · sage: the way home` : `one noisy point x̃ (σ = ${DSMS}): arrows to every possible origin`, 110, 922, { font: F.mono, size: 16, color: P.dust, a: a0 });
    // cards
    const c1 = eout(prog(t, 1, 0.8)); card(840, 250, 990, 190, P.rose, c1); label('shake, then point home', 874, 296, c1, P.rose);
    M('x̃ = [d|x] + [a|σ][n|ε]      home: ([d|x] − x̃)/[a|σ]² = −[n|ε]/[a|σ]', 874, 360, 26, { a: c1 }); text('= ∇ log N(x̃; x, σ²I): the score of one little bell', 874, 404, { font: F.mono, size: 15, color: P.dust, a: eout(prog(t, 9, 0.8)) });
    const c2 = eout(prog(t, 15, 0.8)); card(840, 460, 990, 190, P.sage, c2); label('the loss', 874, 506, c2, P.sage);
    M('[kB|E] ‖ [s|s][k_|θ](x̃, [a|σ]) + [n|ε]/[a|σ] ‖²   ×σ²   →   [kB|E] ‖ [a|σ]·[s|s][k_|θ] + [n|ε] ‖²', 874, 570, 24, { a: c2 }); text('the network just predicts the noise it was shown', 874, 616, { font: F.mono, size: 15, color: P.dust, a: eout(prog(t, 18, 0.8)) });
    const c3 = eout(prog(t, 39, 0.8)); card(840, 670, 990, 120, P.ink, c3); M('best possible [s|s][k_|θ] = average arrow home = ∇ log [p|p][k_|σ](x̃)', 874, 742, 26, { a: c3 });
    const c4 = eout(prog(t, 47, 0.8)); card(840, 810, 990, 110, P.amber, c4); M('Tweedie:  [kB|E][[d|x] | x̃] = x̃ + [a|σ]²·[s|s](x̃)   the score is a denoiser', 874, 876, 24, { a: c4 });
  } });

// 06 — training it for real
const trF = t => (A.snaps.length - 1) * ease(prog(t, 4, 26));
SC.push({ dur: 48, chapter: 'training', title: 'Training it for real', sub: `One network, ${SIG.length} noise levels, ${A.snapSteps[A.snapSteps.length - 1].toLocaleString('en-US')} steps.`,
  caps: [[1, 8, 'Now train it on the eight blobs. One network sees the noisy point and the noise level σ, and predicts the noise.'],
         [8, 16, `Each step: take clean points, pick a noise level between ${SIG[0].toFixed(1)} and ${SIG[SIG.length - 1].toFixed(2)}, shake, and regress onto the noise.`],
         [16, 24, `Watch the arrows at σ ≈ ${SIG[A.mid].toFixed(1)} organise themselves: random at first, then all pointing toward the ring.`],
         [24, 32, 'The loss never reaches zero, and it shouldn’t: when shaken points overlap, nobody could tell which one a noisy point came from.'],
         [32, 40, `Checked against the exact scores of this mixture, the median error is ${pct(A.relErr[SIG[0].toFixed(2)])} at σ = ${SIG[0].toFixed(0)} and ${pct(A.relErr[SIG[SIG.length - 1].toFixed(2)])} at σ = ${SIG[SIG.length - 1].toFixed(2)}.`],
         [40, 47.6, 'The arrows are learned. Now let’s sample with them.']],
  draw(t) {
    const m = square(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)), f = trF(t), i = Math.min(A.snaps.length - 2, Math.floor(f)), u = ease(f - i);
    frame(m, a0); clipTo(m, () => A.data.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.3, P.terracotta, a0 * 0.6))); field(m, lerpField(A.snaps[i], A.snaps[i + 1], u), a0, P.sage, 26);
    const sn = A.snapSteps[u < 0.5 ? i : i + 1]; text(`training step ${sn.toLocaleString('en-US')}  ·  arrows at σ = ${SIG[A.mid].toFixed(2)}`, 110, 922, { font: F.mono, size: 18, color: P.chalk, a: a0 });
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 200, P.plum, c1); label('a noise-conditional score network', 864, 296, c1, P.plum);
    M('(x̃, log [a|σ])  →  [m|net]  →  ≈ −[n|ε]        [s|s][k_|θ](x̃, [a|σ]) = [m|net] / [a|σ]', 864, 360, 24, { a: c1 }); text('3 inputs · 3 hidden layers of 128 · 2 outputs · swish', 864, 408, { font: F.mono, size: 15, color: P.dust, a: c1 });
    const la = eout(prog(t, 8, 0.8)); if (la > 0) { card(830, 470, 1000, 110, P.amber, la); SIG.forEach((s, k) => pill(864 + k * 94, 528, s.toFixed(2), P.amber, la * eout(prog(t, 8.3 + k * 0.08, 0.3)), { size: 17 })); }
    const ca = eout(prog(t, 16, 0.8));
    if (ca > 0) { const H = A.hist, x0 = 880, y0 = 870, w = 900, h = 200, lo = Math.min(...H), hi = Math.max(...H.slice(2)), n = Math.max(2, Math.round(H.length * clamp((t - 16) / 14)));
      card(830, 600, 1000, 320, P.ink, ca); text('denoising loss during training (real run)', x0, 644, { font: F.mono, size: 16, color: P.dust, a: ca });
      setA(ca); ctx.strokeStyle = P.plum; ctx.lineWidth = 2.5; ctx.beginPath(); H.slice(0, n).forEach((v, q) => { const X = x0 + q / (H.length - 1) * w, Y = y0 - clamp((v - lo) / (hi - lo)) * h; q ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
      const za = eout(prog(t, 24, 0.8)); if (za > 0) text(`levels off near ${H[H.length - 1].toFixed(2)}, never 0: noisy points are ambiguous`, x0, 690, { font: F.mono, size: 16, color: P.sage, a: za }); }
  } });

// 07 — the catch
SC.push({ dur: 56, chapter: 'catch', title: 'The catch: arrows in the desert', sub: 'Small noise is sharp, but blind far from the data, and blind to weights.',
  caps: [[1, 8, 'Here is the smallest-noise field, σ = 0.05, checked against the exact answer. Dark means accurate; bright rose means wrong.'],
         [8, 16, 'Near the blobs, where training points land, it is excellent. In the empty channels between them, where no data ever lands, it is guessing.'],
         [16, 24, 'At small noise there is simply nothing out there to learn from. And a sampler starting from random noise begins exactly out there.'],
         [24, 32, 'The second catch is subtler. Take two blobs: the left one holds 80% of the data, the right one 20%.'],
         [32, 40, `Start particles everywhere and run Langevin at the smallest noise. Result: ${pct(A.two.plainShare)} on the left, not 80%. Even after 3,000 steps: ${pct(A.two.longShare)}.`],
         [40, 48, 'Why? Near each blob the arrows look the same whether it holds 80% or 20%. The weight shows only in the far-away arrows: the ones nobody learned.'],
         [48, 55.6, 'And particles can’t hop from one blob to the other to fix it. It is last episode’s mixing problem, back again.']],
  draw(t) {
    const m = square(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)), pA = a0 * (1 - ease(prog(t, 23.5, 1)));
    if (pA > 0) { frame(m, pA); const n = A.err.n, px = m.S / (n - 1); setA(pA * eout(prog(t, 1.5, 1.5))); ctx.imageSmoothingEnabled = true; clipTo(m, () => ctx.drawImage(errCanvas(), m.X(-A.err.L) - px / 2, m.Y(A.err.L) - px / 2, (A.err.L / WR) * m.S + px, (A.err.L / WR) * m.S + px)); setA(1);
      clipTo(m, () => A.data.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.2, P.terracotta, pA * 0.8))); field(m, A.fields[SIG.length - 1], pA * 0.7, P.sage, 20);
      text('rose: where the learned arrows are wrong (σ = 0.05)', 110, 922, { font: F.mono, size: 16, color: P.dust, a: pA });
      const c1 = eout(prog(t, 8, 0.8)) * (1 - ease(prog(t, 23.5, 1))); card(830, 250, 1000, 260, P.rose, c1); label('arrows in the desert', 864, 296, c1, P.rose);
      para('Training points only ever land near the data. With tiny noise, the space between the blobs never sees a single example, so the network can only extrapolate there.', 864, 350, 920, { size: 25, a: c1 }); }
    // the 80/20 experiment
    const bA = eout(prog(t, 24, 1));
    if (bA > 0) { const T = A.two, L = square(110, 250, 640), X = L.X, Y = L.Y; frame(L, bA);
      const show = t < 31 ? T.data : T.plain, col = t < 31 ? P.terracotta : P.rose, sw = eout(prog(t, t < 31 ? 24.5 : 31.5, 1.2));
      clipTo(L, () => show.forEach((p, k) => dot(X(p[0]), Y(p[1]), 2.4, col, bA * sw * (k < 700 ? 0.85 : 0))));
      text(t < 31 ? 'the data: 80% left, 20% right' : 'plain Langevin at σ = 0.05, from everywhere', 110, 922, { font: F.mono, size: 16, color: P.dust, a: bA });
      card(830, 250, 1000, 300, P.amber, bA); label('share in the left blob', 864, 296, bA, P.amber);
      [['truth', T.w[0], P.sage, 24], ['plain Langevin, 300 steps', T.plainShare, P.rose, 32], ['plain Langevin, 3,000 steps', T.longShare, P.rose, 36]].forEach(([nm, v, c, t0], i) => { const a = eout(prog(t, t0, 0.8)); const y = 360 + i * 64;
        text(nm, 864, y, { font: F.mono, size: 18, color: P.stone, a }); rbox(1200, y - 20, 520, 26, 8, rgba(P.border, 0.6), null, a); rbox(1200, y - 20, 520 * v * ease(prog(t, t0, 1.2)), 26, 8, rgba(c, 0.75), null, a); text(pct(v), 1740, y, { font: F.mono, size: 20, color: c, a }); });
      const c2 = eout(prog(t, 40, 0.8)); if (c2 > 0) { card(830, 580, 1000, 330, P.rose, c2); label('local arrows can’t see weights', 864, 626, c2, P.rose);
        M('[s|s]([d|x]) = ∇ log (0.8·[p|p₁] + 0.2·[p|p₂]) ≈ ∇ log [p|p₁]  near blob 1', 864, 690, 23, { a: c2 });
        para('Multiply a blob by 0.8 or 0.2 and the log only shifts by a constant: its slope, the arrow, doesn’t change. The weights matter only far away, where the two blobs compete.', 864, 750, 920, { size: 23, a: eout(prog(t, 42, 0.8)) }); } }
  } });

// 08 — noise at many scales
const anF = t => NS * clamp((t - 22) / 30);
SC.push({ dur: 60, chapter: 'many scales', title: 'Noise at many scales', sub: 'Blur first, sharpen later: annealed Langevin.',
  caps: [[1, 8, 'The fix: train the network at many noise levels at once. That’s why it takes σ as an input.'],
         [8, 15, 'At σ = 3 the data blurs into one broad hill. Its arrows reach everywhere, and noisy points cover every region: reliable.'],
         [15, 22, 'Step the noise down and the field sharpens: first a ring, then eight separate valleys.'],
         [22, 30, 'Annealed Langevin: start particles from noise at the largest σ, take a few Langevin steps, lower σ, and repeat.'],
         [30, 38, 'Big steps while the noise is big, tiny careful steps at the end: the step size shrinks like σ².'],
         [38, 46, 'Early on, the broad arrows move particles toward the data as a whole, carrying the right weights. Later, sharp arrows settle them into place.'],
         [46, 53, 'Like finding a house: first the country, then the city, then the street.'],
         [53, 59.6, 'Song and Ermon, 2019: noise-conditional score networks. The last step before diffusion.']],
  draw(t) {
    const m = square(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)), f = anF(t);
    const fl = t < 22 ? (SIG.length - 1) * ease(prog(t, 8, 13)) : clamp(f / (A.T / A.keep), 0, SIG.length - 1), sg = SIG[Math.min(SIG.length - 1, Math.round(fl))];
    frame(m, a0); const r = rng(3), g = gaussian(r); clipTo(m, () => A.data.forEach(p => { const e1 = g(), e2 = g(); dot(m.X(p[0] + sg * e1), m.Y(p[1] + sg * e2), 2.2, P.terracotta, a0 * 0.35); }));
    field(m, fieldAtLevel(fl), a0, P.sage, 26);
    if (t > 21.5) particles(m, A.ann, f, P.chalk, eout(prog(t, 21.5, 0.6)), 3);
    text(`σ = ${sg.toFixed(2)}   ·   faint dots: the data blurred by that much noise`, 110, 922, { font: F.mono, size: 16, color: P.dust, a: a0 });
    // the σ ladder
    const la = eout(prog(t, 2, 0.8)); card(790, 250, 150, 670, P.amber, la); label('σ', 850, 290, la, P.amber);
    SIG.forEach((s, k) => { const y = 330 + k * 58, on = Math.round(fl) === k; if (on && la > 0) rbox(806, y - 22, 118, 36, 10, rgba(P.amber, 0.18), P.amber, la, 1.5); text(s.toFixed(2), 865, y + 4, { font: F.mono, size: 18, color: on ? P.chalk : P.dust, align: 'center', a: la }); });
    const c1 = eout(prog(t, 8, 0.8)); card(970, 250, 860, 200, P.ink, c1); label('big σ vs small σ', 1004, 296, c1, P.ink);
    text('big σ: blurry but trustworthy everywhere — sees the weights', 1004, 350, { size: 22, color: P.chalk, a: c1 }); text('small σ: sharp detail — but only near the data', 1004, 396, { size: 22, color: P.chalk, a: eout(prog(t, 15, 0.8)) });
    const c2 = eout(prog(t, 22, 0.8)); card(970, 470, 860, 290, P.sage, c2); label('annealed langevin (ncsn, 2019)', 1004, 516, c2, P.sage);
    M('for [a|σ] in [a|σ₁] > [a|σ₂] > … > [a|σ][k_|L]:', 1004, 572, 23, { a: c2 }); M('    η = c·[a|σ]²', 1004, 614, 23, { a: eout(prog(t, 30, 0.8)) });
    M(`    repeat ${A.T}×:  [d|x] ← [d|x] + η·[s|s][k_|θ]([d|x], [a|σ]) + √(2η)·[n|ε]`, 1004, 656, 23, { a: c2 }); text(`${SIG.length} levels × ${A.T} steps = ${SIG.length * A.T} steps`, 1004, 716, { font: F.mono, size: 15, color: P.dust, a: c2 });
    const c3 = eout(prog(t, 46, 0.8)); if (c3 > 0) { card(970, 780, 860, 140, P.amber, c3); text('country → city → street', 1004, 866, { font: F.serif, italic: true, size: 38, color: P.amber, a: c3 }); }
  } });

// 09 — weights restored
SC.push({ dur: 40, chapter: 'result', title: 'Weights restored', sub: 'The same two blobs, annealed.',
  caps: [[1, 8, `Back to the 80/20 blobs. Same kind of network, same starting particles, same 300 steps — but annealed: ${pct(A.two.annShare)} land on the left.`],
         [8, 16, 'The broad early arrows see both blobs at once, and the bigger one pulls harder. The weights are settled before the detail arrives.'],
         [16, 24, 'No energy, no Z, no fantasies during training. Just a regression onto noise, and a sampler that follows the arrows.'],
         [24, 31, 'Not perfect: a little short of 80%, because each noise level only gets 30 steps. More steps get closer.'],
         [31, 39.6, 'The recipe is complete: noise the data, learn to point home at every noise level, then follow the arrows from noise to data.']],
  draw(t) {
    const L = square(110, 250, 620), Rm = square(1210, 250, 620), T = A.two, a0 = eout(prog(t, 0.6, 0.8)), ra = eout(prog(t, 1.5, 1));
    frame(L, a0); clipTo(L, () => T.plain.forEach(p => dot(L.X(p[0]), L.Y(p[1]), 2.4, P.rose, a0 * 0.85))); text('plain Langevin (σ = 0.05)', 110, 902, { font: F.mono, size: 18, color: P.chalk, a: a0 });
    frame(Rm, ra); clipTo(Rm, () => T.ann.forEach((p, k) => dot(Rm.X(p[0]), Rm.Y(p[1]), 2.4, P.sage, ra * 0.85 * clamp((t - 1.5) * 3 - k / 300)))); text('annealed Langevin (σ = 3 → 0.05)', 1210, 902, { font: F.mono, size: 18, color: P.chalk, a: ra });
    const ba = eout(prog(t, 2, 0.8)); card(760, 330, 420, 360, P.amber, ba); label('share on the left', 790, 376, ba, P.amber);
    [['truth', T.w[0], P.chalk], ['plain', T.plainShare, P.rose], ['annealed', T.annShare, P.sage]].forEach(([nm, v, c], i) => { const y = 450 + i * 80, a = ba * eout(prog(t, 2.5 + i * 0.7, 0.6)); text(nm, 790, y, { font: F.mono, size: 18, color: P.stone, a }); text(pct(v), 1150, y + 4, { font: F.mono, size: 34, color: c, align: 'right', a }); rbox(790, y + 18, 360 * v, 8, 4, rgba(c, 0.7), null, a); });
  } });

// 10 — in the wild
SC.push({ dur: 40, chapter: 'in the wild', title: 'Arrows everywhere', sub: 'From toy blobs to every modern image generator.',
  caps: [[1, 9, 'In 2019, noise-conditional score networks with annealed Langevin produced strikingly good images, competitive with GANs, with no adversary at all.'],
         [9, 18, 'In 2021, Yang Song and colleagues let the noise level vary continuously: infinitely many σ’s, one stochastic differential equation. Score models and diffusion became one idea.'],
         [18, 27, 'Because the score is a denoiser, any good denoiser hides a score: “plug-and-play” priors use one to deblur photos and reconstruct MRI scans.'],
         [27, 34, 'And every image and video diffusion model trains a network that, underneath, is a noise-conditional score network.'],
         [34, 39.6, 'The arrows won.']],
  draw(t) {
    const cards = [[1, 'ncsn · 2019', P.sage, 'many σ + annealed Langevin', 'images from arrows alone — no discriminator'], [9, 'score SDEs · 2021', P.ink, 'σ(t): a continuous noise schedule', 'score models and diffusion unified'],
      [18, 'denoisers as priors', P.amber, 'x̃ + σ²·s(x̃) = E[x | x̃]', 'deblurring, super-resolution, MRI'], [27, 'modern generators', P.plum, 'ε-prediction = σ·score', 'the network inside diffusion models']];
    cards.forEach(([t0, h, c, m1, s], i) => { const a = eout(prog(t, t0, 0.8)), x = 100 + (i % 2) * 880, y = 260 + Math.floor(i / 2) * 330;
      card(x, y, 840, 300, c, a); label(h, x + 32, y + 46, a, c); text(m1, x + 32, y + 140, { font: F.mono, size: 26, color: P.chalk, a }); para(s, x + 32, y + 206, 780, { size: 24, a: a * eout(prog(t, t0 + 1, 0.8)) }); });
  } });

// 11 — build it
const CODE = [
  '[k|# denoising score matching: the network predicts the noise]',
  '[m|for] step [m|in] [p|range]([a|6000]):',
  '    x = [d|data_batch]([a|512])',
  '    sig = SIGMAS[randint([a|0], [a|10], [a|512])]          [k|# 3.0 … 0.05]',
  '    eps = randn([a|512], [a|2])',
  '    out = [p|net](x + sig[:, [m|None]]*eps, log(sig))      [k|# s[k_|θ] = out / σ]',
  '    loss = ((out + eps)**[a|2]).sum([a|1]).mean()',
  '    net.[p|backward]([a|2]*(out + eps)/[a|512]);  net.[p|step](lr)',
  '',
  '[k|# annealed Langevin: follow the arrows while the noise shrinks]',
  'x = uniform(-[a|4], [a|4], (n, [a|2]))',
  '[m|for] sig [m|in] SIGMAS:',
  '    eta = [a|0.1] * sig**[a|2]',
  '    [m|for] _ [m|in] [p|range]([a|30]):',
  '        x += eta * [p|net](x, log(sig))/sig + sqrt([a|2]*eta)*randn(n, [a|2])',
];
SC.push({ dur: 32, chapter: 'build it', title: 'Build it yourself', sub: 'Training and sampling, in fifteen lines.',
  caps: [[1, 9, 'Training is a plain regression onto noise. Sampling is a double loop over noise levels and Langevin steps. The full code, with hand-written backprop, is linked below.'],
         [9, 18, 'Try the 80/20 blobs: sample with only the smallest σ, then with all ten, and compare the shares yourself.'],
         [18, 31.6, 'Recap: the score is the slope of log p, with no Z. Langevin balances pull and kick exactly. Noise makes the score learnable. Many noise levels make it samplable.']],
  draw(t) {
    playPill('ep08-score-lab.html', 9, t);
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1060, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 296 + i * 40, 18, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8)); card(1200, 240, 620, 690, P.sage, ra); label('the whole episode', 1236, 290, ra, P.sage);
    [['the score', '[s|s](x) = ∇ log [p|p](x)   (no Z)'], ['why langevin works', 'pull ∇p + spread −∇p = 0'], ['score matching', '[kB|E][ s[k_|θ]² + 2 s[k_|θ]′ ]  (Hyvärinen)'], ['denoising sm', '[kB|E]‖σ·[s|s][k_|θ](x̃, σ) + [n|ε]‖²'], ['annealing', 'σ: 3 → 0.05,  η = c·σ²']].forEach(([nm, m], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.6, 0.8)); text(nm, 1236, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(m, 1236, 412 + i * 112, 21, { a: ai }); });
  } });

// 12 — hook → diffusion
const FWD = (() => { const g = gaussian(rng(99)); return A.data.map(p => [p[0], p[1], g(), g()]); })();
SC.push({ dur: 48, chapter: 'next', title: 'Noise, in slow motion', sub: 'What annealed Langevin was secretly doing.',
  caps: [[1, 8, 'Look at annealed Langevin backwards. Going down the noise ladder turns noise into data. Going up it turns data into noise.'],
         [8, 15, 'Destroying data is easy: add a little noise, then a little more, a thousand times, until nothing but static is left.'],
         [15, 23, 'Undoing one small step is also easy, if you know which way is home. That is exactly what the score, the denoiser, tells you.'],
         [23, 31, 'So define a slow, fixed noising process. Train one network to undo each tiny step. Then run it starting from pure static.'],
         [31, 38, 'Chain enough tiny denoising steps and static becomes data. That idea, made precise, powers nearly every image generator you have seen.'],
         [38, 47.6, 'Next episode: diffusion models. Noise, run backwards.']],
  draw(t) {
    const dimAll = 1 - 0.9 * ease(prog(t, 38.5, 1)); GA = dimAll;
    const m = square(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const fwd = t < 16, sg = fwd ? 3 * Math.pow(prog(t, 2, 12), 1.6) : 0;
    if (fwd) clipTo(m, () => FWD.forEach(p => marble(m.X(p[0] + sg * p[2]), m.Y(p[1] + sg * p[3]), 3, mixc(P.terracotta, P.rose, clamp(sg / 3)), a0)));
    else { const f = NS * clamp((t - 17) / 13); field(m, fieldAtLevel(clamp(f / (A.T / A.keep), 0, SIG.length - 1)), eout(prog(t, 16, 1)) * 0.7, P.sage, 22); particles(m, A.ann, f, P.terracotta, eout(prog(t, 16, 0.6)), 3); }
    text(fwd ? `forward: add noise · σ = ${sg.toFixed(2)}` : 'backward: follow the arrows home', 110, 922, { font: F.mono, size: 18, color: fwd ? P.rose : P.sage, a: a0 });
    const c1 = eout(prog(t, 8, 0.8)); card(830, 250, 1000, 150, P.rose, c1); label('destroy slowly', 864, 296, c1, P.rose); M('[d|x₀] → [d|x₁] → … → [d|x₁₀₀₀] ≈ pure [n|noise]      (easy)', 864, 352, 24, { a: c1 });
    const c2 = eout(prog(t, 15, 0.8)); card(830, 420, 1000, 150, P.sage, c2); label('undo each step', 864, 466, c2, P.sage); M('[d|x][k_|t−1] ≈ [d|x][k_|t] + (a little)·[s|s][k_|θ]([d|x][k_|t], t) + (a little noise)', 864, 522, 24, { a: c2 });
    const c3 = eout(prog(t, 23, 0.8)); card(830, 590, 1000, 150, P.ink, c3); label('then run it from static', 864, 636, c3, P.ink); text('one network · one tiny step at a time', 864, 694, { size: 26, color: P.chalk, a: c3 });
    GA = 1;
    const qa = eout(prog(t, 38.5, 1.2));
    if (qa > 0) {
      text('Destroy it slowly.', 960, 440, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('Learn to undo it.', 960, 530, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: eout(prog(t, 39.3, 1.2)) });
      text('NEXT  ·  EPISODE 09', 960, 640, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 40.5, 1)), ls: 4 });
      seasonStrip(8, 41.7, t);
      text('Diffusion models — noise, run backwards', 960, 695, { font: F.serif, italic: true, size: 42, color: P.stone, align: 'center', a: eout(prog(t, 41, 1)) });
    }
  } });
