// ─────────────────────────────────────────────────────────────
//  Real GAN runs (exported by code/export_ep05_assets.py)
// ─────────────────────────────────────────────────────────────
const A = window.EP5, R4 = window.EP4.rows;
const MSNAP = A.mnistSnaps.map(u8imgs), GAN64 = u8imgs(A.gan64), VAE16 = u8imgs(A.vae16), SC5 = A.score, HIST = A.mnistHist;
const EPOCHS = MSNAP.length - 1;
const pct = v => `${Math.round(v * 100)}%`;

// 2-D arena: [-3, 3]², the judge's field as a heat map, fakes, real modes
const MODES = Array.from({ length: 8 }, (_, k) => [2 * Math.cos(k * Math.PI / 4), 2 * Math.sin(k * Math.PI / 4)]);
function fieldCanvas(run, i) { return cached(`f${run}${i}`, () => { const u = b64bytes(A[run][i].f), c = document.createElement('canvas'); c.width = c.height = 48; const g = c.getContext('2d'), d = g.createImageData(48, 48), col = hx(P.ink);
  for (let k = 0; k < 48 * 48; k++) { d.data[k * 4] = col[0]; d.data[k * 4 + 1] = col[1]; d.data[k * 4 + 2] = col[2]; d.data[k * 4 + 3] = Math.round(u[k] * 0.8); } g.putImageData(d, 0, 0); return c; }); }
function arena(run, fi, x0, y0, S, o = {}) {
  const { a = 1, field = 1, fakes = 1, grads = 0, real = 1, hud = true } = o; const snaps = A[run], n = snaps.length;
  const f = clamp(fi, 0, n - 1), i = Math.floor(f), j = Math.min(n - 1, i + 1), u = f - i, sA = snaps[i], sB = snaps[j];
  const X = v => x0 + (v + 3) / 6 * S, Y = v => y0 + (3 - v) / 6 * S;
  setA(a); ctx.fillStyle = P.surface; ctx.fillRect(x0, y0, S, S); ctx.strokeStyle = P.border; ctx.lineWidth = 1.5; ctx.strokeRect(x0, y0, S, S);
  if (field > 0) { ctx.imageSmoothingEnabled = true; setA(a * field * (1 - u)); ctx.drawImage(fieldCanvas(run, i), x0, y0, S, S); setA(a * field * u); ctx.drawImage(fieldCanvas(run, j), x0, y0, S, S); }
  setA(1);
  if (real > 0) MODES.forEach(m => { ring(X(m[0]), Y(m[1]), 0.24 * S / 6, P.chalk, a * real * 0.6, 1.5); });
  if (fakes > 0) for (let k = 0; k < sA.x.length; k++) { const px = lerp(sA.x[k][0], sB.x[k][0], u), py = lerp(sA.x[k][1], sB.x[k][1], u); if (Math.abs(px) < 3 && Math.abs(py) < 3) dot(X(px), Y(py), 2.6, P.terracotta, a * fakes * 0.85); }
  if (grads > 0) for (let k = 0; k < sA.g.length; k++) { const px = sA.x[k][0], py = sA.x[k][1], g = sA.g[k], L = Math.hypot(g[0], g[1]) + 1e-9, s = Math.min(0.5, L * 4) / L;
    if (Math.abs(px) < 3 && Math.abs(py) < 3) arrow(X(px), Y(py), X(px + g[0] * s), Y(py + g[1] * s), P.sage, 2, a * grads); }
  if (hud) { const s = u < 0.5 ? sA : sB; text(`step ${Math.round(lerp(sA.step, sB.step, u))}`, x0 + 16, y0 + 30, { font: F.mono, size: 18, color: P.chalk, a });
    text(`modes found ${s.modes}/8`, x0 + S - 16, y0 + 30, { font: F.mono, size: 18, color: s.modes === 8 ? P.sage : P.rose, align: 'right', a }); }
  return { X, Y };
}
function dominant(run, i) { const x = A[run][i].x, c = new Array(8).fill(0); x.forEach(p => { let b = 0, bd = 1e9; MODES.forEach((m, k) => { const d = Math.hypot(p[0] - m[0], p[1] - m[1]); if (d < bd) { bd = d; b = k; } }); if (bd < 0.4) c[b]++; }); const mx = Math.max(...c); return { k: c.indexOf(mx), share: mx / x.length }; }
// 1-D densities for the math scenes
const pData = x => 0.5 * npdf(x, -1, 0.5) + 0.5 * npdf(x, 1.3, 0.6);
function jsd(shift) { let s = 0; const dx = 0.01; for (let x = -6; x < 7; x += dx) { const a = pData(x), b = pData(x - shift), m = (a + b) / 2; if (a > 1e-12) s += 0.5 * a * Math.log(a / m) * dx; if (b > 1e-12) s += 0.5 * b * Math.log(b / m) * dx; } return s; }
function curve1d(fn, x0, x1, base, sx, sy, col, lw, a, dash) { setA(a); ctx.strokeStyle = col; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash); ctx.beginPath(); for (let i = 0; i <= 240; i++) { const x = -3.5 + 7 * i / 240, X = x0 + (x + 3.5) * sx, Yv = base - fn(x) * sy; i ? ctx.lineTo(X, Yv) : ctx.moveTo(X, Yv); } ctx.stroke(); ctx.setLineDash([]); setA(1); }

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open
SC.push({ dur: 24, chapter: 'intro',
  caps: [[0.6, 6, 'Watch this network learn to draw digits, one training epoch at a time.'],
         [6, 12, 'Here is the twist: it has never seen a digit. Not one pixel of real data ever reaches it.'],
         [12, 18, 'All it ever gets is criticism — from a judge who has.'],
         [18, 23.6, 'This is a GAN: a generative adversarial network.']],
  draw(t) {
    const dim = lerp(1, 0.14, ease(prog(t, 12.5, 1.5))), e = Math.min(EPOCHS, Math.floor(prog(t, 0.5, 10) * EPOCHS));
    GA = dim;
    for (let i = 0; i < 16; i++) drawImg(cached(`m${e}_${i}`, () => imgCanvas(MSNAP[e][i])), 560 + (i % 4) * 202, 230 + Math.floor(i / 4) * 170, 160, eout(prog(t, 0.3, 1)));
    text(`epoch ${e}`, 960, 950 - 20, { font: F.mono, size: 26, color: P.chalk, align: 'center', a: eout(prog(t, 0.3, 1)) });
    text('generated from the same 16 noise vectors', 960, 222, { font: F.mono, size: 16, color: P.dust, align: 'center', a: eout(prog(t, 0.3, 1)) });
    GA = 1;
    const a = eout(prog(t, 13, 1.4));
    text('EPISODE 05', 960, 430, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('GANs', 960, 560 + (1 - a) * 16, { font: F.serif, size: 150, color: P.chalk, align: 'center', a, ls: -1 });
    text('learning by being judged', 960, 632, { size: 34, color: P.stone, align: 'center', a: eout(prog(t, 13.8, 1.2)) });
    const tg = [[P.terracotta, 'x  data & fakes'], [P.rose, 'z  noise'], [P.plum, 'G, D  players'], [P.ink, 'p, D(x)  probabilities'], [P.sage, 'gradients & verdicts']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let cx = 960 - tw / 2; tg.forEach(([c, s], i) => { cx += tag(cx, 690, c, s, eout(prog(t, 16 + i * 0.35, 0.6)), 21) + 14; });
  } });

// 01 — two players
SC.push({ dur: 36, chapter: 'the game', title: 'Two players, one game', sub: 'A forger and a detective.',
  caps: [[1, 7, 'Last episode we built a learned judge: a classifier trained to tell real images from generated ones.'],
         [7, 14, 'A GAN puts that judge to work. Player one, the generator G, is episode 1’s g: it turns random noise z into samples.'],
         [14, 21, "Player two, the discriminator D, sees real images and G's fakes, and outputs the probability that its input is real."],
         [21, 28, 'Notice where the real data flows: only into D. The generator learns about digits entirely second-hand.'],
         [28, 35.6, 'D is trained to catch fakes. G is trained to fool D. Forger versus detective.']],
  draw(t) {
    const y = 560, a1 = eout(prog(t, 7, 0.8)), a2 = eout(prog(t, 14, 0.8)), last = MSNAP[EPOCHS];
    // noise → G
    const r = rng(Math.floor(t * 2)); for (let i = 0; i < 4; i++) { const v = new Float32Array(784); for (let k = 0; k < 784; k++) v[k] = r(); drawImg(imgCanvas(v, P.rose), 130 + (i % 2) * 74, y - 70 + Math.floor(i / 2) * 74, 66, a1); }
    text('z ~ N(0, I)', 200, y + 110, { font: F.mono, size: 18, color: P.rose, align: 'center', a: a1 });
    arrow(290, y, 360, y, P.dust, 2, a1);
    rbox(370, y - 90, 260, 180, 20, rgba(P.plum, 0.12), rgba(P.plum, 0.75), a1, 2.5); text('generator', 500, y + 8, { font: F.serif, size: 38, color: P.chalk, align: 'center', a: a1 }); M('[m|G][a_|θ]([n|z])', 500, y + 56, 22, { align: 'center', a: a1 });
    arrow(640, y, 710, y, P.dust, 2, a1);
    for (let i = 0; i < 4; i++) drawImg(cached(`l${i}`, () => imgCanvas(last[i])), 720 + (i % 2) * 74, y - 70 + Math.floor(i / 2) * 74, 66, a1);
    text('fakes', 790, y + 110, { font: F.mono, size: 18, color: P.terracotta, align: 'center', a: a1 });
    // real data → D
    for (let i = 0; i < 4; i++) drawImg(cached(`rl${i}`, () => imgCanvas(u8imgs(window.EP4.samples.real)[i])), 720 + (i % 2) * 74, 270 + Math.floor(i / 2) * 74, 66, a2);
    text('real data', 790, 260, { font: F.mono, size: 18, color: P.stone, align: 'center', a: a2 });
    arrow(880, 340, 1010, 500, P.dust, 2, a2); arrow(880, y, 1010, y, P.dust, 2, a2);
    rbox(1020, y - 90, 280, 180, 20, rgba(P.plum, 0.12), rgba(P.plum, 0.75), a2, 2.5); text('discriminator', 1160, y + 8, { font: F.serif, size: 36, color: P.chalk, align: 'center', a: a2 }); M('[m|D][a_|φ]([d|x]) ∈ (0, 1)', 1160, y + 56, 22, { align: 'center', a: a2 });
    arrow(1310, y, 1380, y, P.dust, 2, a2); text('P(real)', 1400, y + 8, { font: F.mono, size: 26, color: P.ink, a: a2 });
    const wa = eout(prog(t, 21, 0.8)); if (wa > 0) { rbox(700, 238, 180, 190, 14, null, P.sage, wa, 2.5); text('the only door for real data', 790, 460 - 10, { font: F.mono, size: 16, color: P.sage, align: 'center', a: wa }); }
    const ca = eout(prog(t, 28, 0.8));
    card(360, 790, 1200, 120, P.plum, ca);
    M('[m|D]: catch fakes   ([m|D]([d|x]) → 1,  [m|D]([m|G]([n|z])) → 0)          [m|G]: fool [m|D]   ([m|D]([m|G]([n|z])) → 1)', 960, 862, 24, { align: 'center', a: ca });
  } });

// 02 — the judge's best move
SC.push({ dur: 46, chapter: 'the judge', title: "The judge's perfect strategy", sub: 'What would an ideal discriminator output?',
  caps: [[1, 8, "Let's find the judge's best possible strategy. Picture both distributions in one dimension: the real data, and the generator's current guess."],
         [8, 16, 'The judge scores log D on real points, plus log(1 − D) on fakes. It wants both high.'],
         [16, 24, 'Zoom in on a single point x. Real data lands there with density a, fakes with density b. The judge must pick one number, y = D(x).'],
         [24, 32, 'Maximise a·log y + b·log(1 − y). Set the derivative a/y − b/(1 − y) to zero, and y = a / (a + b).'],
         [32, 39, 'So the perfect judge is D*(x) = pdata(x) / (pdata(x) + pg(x)). Only real there? 1. Only fakes? 0.'],
         [39, 45.6, 'And where the two overlap equally, even the perfect judge can only shrug: one half.']],
  draw(t) {
    const x0 = 150, base = 700, sx = 150, sy = 520, sh = 1.4, pg = x => pData(x - sh), Dstar = x => { const a = pData(x), b = pg(x); return a + b < 1e-6 ? 0.5 : a / (a + b); };
    const a0 = eout(prog(t, 0.8, 1));
    line(x0, base, x0 + 7 * sx, base, P.borderLight, 2, a0);
    curve1d(pData, x0, 0, base, sx, sy, P.ink, 4, a0); curve1d(pg, x0, 0, base, sx, sy, P.terracotta, 4, a0, [10, 8]);
    text('p_data', x0 + 20, 300, { font: F.mono, size: 22, color: P.ink, a: a0 }); text('p_g  (the fakes)', x0 + 20, 334, { font: F.mono, size: 22, color: P.terracotta, a: a0 });
    const pa = eout(prog(t, 16, 0.8)) * (1 - ease(prog(t, 32, 0.8))), px = 0.6, PX = x0 + (px + 3.5) * sx;
    if (pa > 0) { line(PX, base, PX, base - Math.max(pData(px), pg(px)) * sy - 40, P.chalk, 1.5, pa, [5, 5]); dot(PX, base - pData(px) * sy, 7, P.ink, pa); dot(PX, base - pg(px) * sy, 7, P.terracotta, pa);
      text(`a = ${pData(px).toFixed(2)}`, PX - 16, base - pData(px) * sy + 30, { font: F.mono, size: 20, color: P.ink, align: 'right', a: pa }); text(`b = ${pg(px).toFixed(2)}`, PX - 16, base - pg(px) * sy - 12, { font: F.mono, size: 20, color: P.terracotta, align: 'right', a: pa }); }
    const da = eout(prog(t, 32, 1));
    if (da > 0) { const sy2 = 330; curve1d(Dstar, x0, 0, base, sx, sy2, P.plum, 5, da);
      line(x0, base - 0.5 * sy2, x0 + 7 * sx, base - 0.5 * sy2, P.dust, 1, da, [4, 8]); text('½', x0 + 7 * sx + 12, base - 0.5 * sy2 + 8, { font: F.mono, size: 22, color: P.dust, a: da });
      text('D*(x)', x0 + 7 * sx - 120, base - Dstar(3) * sy2 - 16, { font: F.mono, size: 24, color: P.plum, a: da }); }
    const c1 = eout(prog(t, 8, 0.8));
    card(1250, 250, 580, 180, P.plum, c1); label("the judge's objective", 1284, 296, c1, P.plum);
    M('[p|E][p_|data] log [m|D]([d|x]) + [p|E][p_|g] log(1 − [m|D]([d|x]))', 1284, 372, 20, { a: c1 });
    const c2 = eout(prog(t, 24, 0.8));
    card(1250, 460, 580, 200, P.ink, c2); label('one point at a time', 1284, 506, c2, P.ink);
    M('max[k_|y]  a·log y + b·log(1 − y)', 1284, 566, 24, { a: c2 }); M('a/y − b/(1−y) = 0  ⇒  y = a/(a+b)', 1284, 620, 24, { a: eout(prog(t, 26, 0.8)) });
    const c3 = eout(prog(t, 32, 0.8));
    card(1250, 690, 580, 150, P.sage, c3);
    M('[m|D]*([d|x]) = [p|p][p_|data] / ([p|p][p_|data] + [p|p][p_|g])', 1284, 780, 25, { a: c3 });
  } });

// 03 — what the generator really minimises
SC.push({ dur: 46, chapter: 'the target', title: 'What the forger is really minimising', sub: 'Plug the perfect judge back in.',
  caps: [[1, 8, 'Now plug the perfect judge back into the game. What is the generator really minimising?'],
         [8, 16, 'A little algebra: compare each distribution with m, the average of the two. Two KL divergences appear — episode 1’s measure of mismatch.'],
         [16, 24, 'Their average is the Jensen–Shannon divergence. So the game’s value is −log 4, plus twice the JSD between data and generator.'],
         [24, 32, "JSD is zero exactly when the generator's distribution equals the data's. That is the only way for G to win."],
         [32, 39, 'Watch: as pg slides onto pdata, the divergence melts to zero and the perfect judge flattens to one half everywhere.'],
         [39, 45.6, 'One half — the 50% from our two-sample test. The finish line of the game is the metric’s perfect score.']],
  draw(t) {
    const a0 = eout(prog(t, 0.8, 0.8));
    M('V([m|G], [m|D]*) = [p|E][p_|data] log [p|p][p_|d]/([p|p][p_|d]+[p|p][p_|g])  +  [p|E][p_|g] log [p|p][p_|g]/([p|p][p_|d]+[p|p][p_|g])', 960, 300, 26, { align: 'center', a: a0 });
    const b = eout(prog(t, 8, 0.8)); M('= −log 4 + KL([p|p][p_|d] ‖ [p|m]) + KL([p|p][p_|g] ‖ [p|m]),     [p|m] = ½([p|p][p_|d] + [p|p][p_|g])', 960, 360, 26, { align: 'center', a: b });
    const c = eout(prog(t, 16, 0.8)); M('= −log 4 + 2 · JSD([p|p][p_|data] ‖ [p|p][p_|g])', 960, 430, 34, { align: 'center', a: c });
    // sliding demo
    const sh = t < 30 ? 2.2 : lerp(2.2, 0, ease(prog(t, 30, 7))), da = eout(prog(t, 24, 1));
    if (da > 0) { const x0 = 150, base = 880, sx = 140, sy = 330, pg = x => pData(x - sh), Ds = x => { const a = pData(x), bb = pg(x); return a + bb < 1e-6 ? 0.5 : a / (a + bb); };
      line(x0, base, x0 + 7 * sx, base, P.borderLight, 2, da); curve1d(pData, x0, 0, base, sx, sy, P.ink, 4, da); curve1d(pg, x0, 0, base, sx, sy, P.terracotta, 4, da, [10, 8]); curve1d(Ds, x0, 0, base, sx, 300, P.plum, 3, da * 0.8);
      const J = jsd(sh); card(1240, 560, 590, 320, P.sage, da); label('live', 1274, 606, da, P.sage);
      text(`JSD = ${J.toFixed(3)} nats`, 1274, 680, { font: F.mono, size: 34, color: P.chalk, a: da });
      text(`V = −log 4 + 2·JSD = ${(-Math.log(4) + 2 * J).toFixed(3)}`, 1274, 740, { font: F.mono, size: 22, color: P.stone, a: da });
      text(`D*(x) at the peak: ${Ds(1.3).toFixed(2)}`, 1274, 790, { font: F.mono, size: 22, color: P.plum, a: da });
      text(`max possible: log 2 = ${Math.log(2).toFixed(3)}`, 1274, 840, { font: F.mono, size: 16, color: P.dust, a: da }); }
  } });

// 04 — how the forger learns
SC.push({ dur: 48, chapter: 'the gradient', title: 'How the forger learns without seeing data', sub: "Follow the slope of the judge's opinion.",
  caps: [[1, 8, 'So how does the generator improve, if it never sees data? Through the judge’s gradient.'],
         [8, 15, "Colour the plane by the judge's opinion: bright where D says 'real', dark where it says 'fake'. A real snapshot from training."],
         [15, 23, 'For each fake point, the gradient of D says which direction looks more real. The arrows point uphill.'],
         [23, 31, "Backprop carries that signal through x = G(z) into G's weights. The judge's opinion becomes the forger's lesson."],
         [31, 39, 'One trap. The original loss, log(1 − D(G(z))), goes almost flat when the judge easily spots fakes — early on, exactly when G needs help most.'],
         [39, 47.6, 'So in practice G maximises log D(G(z)) instead: the same goal, but steep gradients when the fakes are bad.']],
  draw(t) {
    const A1 = 1 - ease(prog(t, 30.5, 1));
    if (A1 > 0) { GA = A1; const fi = 3;
      arena('healthy', fi, 150, 240, 680, { a: eout(prog(t, 0.5, 1)), field: eout(prog(t, 8, 1)), grads: eout(prog(t, 15, 1)), fakes: 1 });
      text('bright = judge says “real”', 150, 950 - 20, { font: F.mono, size: 16, color: P.ink, a: eout(prog(t, 8, 1)) });
      const ca = eout(prog(t, 23, 0.8)); card(920, 300, 900, 260, P.sage, ca); label('the chain rule', 956, 346, ca, P.sage);
      M('[d|x] = [m|G][a_|θ]([n|z])', 956, 420, 30, { a: ca });
      M('∂L/∂[a|θ] = (∂L/∂[d|x]) · (∂[d|x]/∂[a|θ])', 956, 490, 30, { a: ca }); text('judge’s arrow      ×   how G’s knobs move x', 956, 535, { font: F.mono, size: 16, color: P.dust, a: ca });
      GA = 1; }
    const B = eout(prog(t, 31, 1));
    if (B > 0) { const x0 = 220, y0 = 820, W = 760, Hh = 480;
      line(x0, y0, x0 + W, y0, P.borderLight, 2, B); line(x0, y0, x0, y0 - Hh, P.borderLight, 2, B);
      text('D(G(z)): how real the judge finds the fake →', x0 + W, y0 + 40, { font: F.mono, size: 16, color: P.dust, align: 'right', a: B });
      text('0', x0, y0 + 24, { font: F.mono, size: 16, color: P.dust, align: 'center', a: B }); text('1', x0 + W, y0 + 24, { font: F.mono, size: 16, color: P.dust, align: 'center', a: B });
      const plot = (fn, col, a) => { setA(a); ctx.strokeStyle = col; ctx.lineWidth = 4; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const d = 0.005 + 0.99 * i / 200, v = clamp(fn(d) / 5.3, -0.1, 1); i ? ctx.lineTo(x0 + d * W, y0 - v * Hh) : ctx.moveTo(x0 + d * W, y0 - v * Hh); } ctx.stroke(); setA(1); };
      plot(d => Math.log(1 / (1 - d)) , P.rose, B); plot(d => -Math.log(d), P.sage, eout(prog(t, 39, 1)));
      text('minimise log(1 − D): flat near 0 — no signal when fakes are bad', x0 + 30, y0 - 380, { font: F.mono, size: 18, color: P.rose, a: B });
      text('minimise −log D: steep near 0 — strong signal when fakes are bad', x0 + 30, y0 - 420, { font: F.mono, size: 18, color: P.sage, a: eout(prog(t, 39, 1)) });
      ring(x0 + 0.03 * W, y0 - clamp(Math.log(1 / 0.97) / 5.3) * Hh, 22, P.rose, B * eout(prog(t, 33, 0.6)), 2.5);
      card(1080, 380, 740, 300, P.plum, B); label('the two forger losses', 1114, 426, B, P.plum);
      M('original:   min [m_|G]  log(1 − [m|D]([m|G]([n|z])))', 1114, 500, 24, { a: B }); M('in practice: min [m_|G]  −log [m|D]([m|G]([n|z]))', 1114, 560, 24, { a: eout(prog(t, 39, 1)) });
      text('same fixed point, much better gradients', 1114, 630, { font: F.serif, italic: true, size: 26, color: P.sage, a: eout(prog(t, 40, 1)) }); }
  } });

// 05 — watch it train
SC.push({ dur: 52, chapter: 'training', title: 'Watch a real GAN train', sub: 'Eight blobs on a ring. Every frame below comes from an actual run.',
  caps: [[1, 8, 'Now let’s watch a real GAN train — every frame is from an actual run. The target: eight blobs arranged on a ring.'],
         [8, 16, "At first the generator's points are a tangle in the middle, and the judge easily lights up the real blobs."],
         [16, 24, 'The fakes chase the bright regions. The judge adapts, darkening wherever fakes pile up.'],
         [24, 32, 'Push, adapt, push again. Mode by mode, the generator spreads out.'],
         [32, 40, "Eventually the fakes sit on the blobs, and the judge's picture washes out toward grey: it can barely tell anymore."],
         [40, 51.6, 'Nobody ever told the generator where the blobs were. It found them by following the judge’s disapproval.']],
  draw(t) {
    const n = A.healthy.length, fi = lerp(0, n - 1, ease(prog(t, 3, 40)));
    arena('healthy', fi, 560, 240, 700, { a: eout(prog(t, 0.5, 1)), field: 1, fakes: 1, grads: 0 });
    // mode count over time
    const x0 = 1330, y0 = 700, w = 480, h = 180, a = eout(prog(t, 2, 1)); rbox(x0 - 20, y0 - h - 60, w + 40, h + 110, 14, rgba(P.surface, 0.9), P.border, a);
    text('modes found', x0, y0 - h - 24, { font: F.mono, size: 16, color: P.dust, a });
    setA(a); ctx.strokeStyle = P.sage; ctx.lineWidth = 3; ctx.beginPath(); for (let i = 0; i <= Math.floor(fi); i++) { const X = x0 + i / (n - 1) * w, Yv = y0 - A.healthy[i].modes / 8 * h; i ? ctx.lineTo(X, Yv) : ctx.moveTo(X, Yv); } ctx.stroke(); setA(1);
    line(x0, y0, x0 + w, y0, P.borderLight, 1.5, a); text('8', x0 - 16, y0 - h + 6, { font: F.mono, size: 14, color: P.dust, a }); text('0', x0 - 16, y0 + 6, { font: F.mono, size: 14, color: P.dust, a });
    text('fakes (G)', 150, 300, { font: F.mono, size: 18, color: P.terracotta, a }); text('real blobs', 150, 330, { font: F.mono, size: 18, color: P.chalk, a }); text('judge: P(real)', 150, 360, { font: F.mono, size: 18, color: P.ink, a });
  } });

// 06 — digits
const GS = SC5, VS = R4.vae;
SC.push({ dur: 48, chapter: 'digits', title: 'The same game on handwritten digits', sub: `Two small MLPs, ${EPOCHS} passes over MNIST, backprop by hand.`,
  caps: [[1, 8, `Same recipe on handwritten digits: two small networks, ${EPOCHS} passes over the data.`],
         [8, 15, 'Epoch by epoch, noise becomes strokes, and strokes become digits.'],
         [15, 23, 'Next to our VAE from episode 3, the difference jumps out: crisp strokes, no averaging. The judge punishes blur instantly.'],
         [23, 32, `Call in episode 4's judges. FID: ${GS.FID.toFixed(1)}, against the VAE's ${VS.FID.toFixed(1)}. Precision ${pct(GS.precision)}, recall ${GS.recall < 0.01 ? 'under 1%' : pct(GS.recall)} — the VAE managed under 1%.`],
         [32, 39, `The learned judge spots these fakes ${pct(GS.C2ST)} of the time, versus ${pct(VS.C2ST)} for the VAE.`],
         [39, 47.6, `And the novelty check: the nearest training image is ${GS.NN_train.toFixed(2)} away — about as far as a real, unseen digit (${R4.real.NN_train.toFixed(2)}). It is not copying.`]],
  draw(t) {
    const e = Math.min(EPOCHS, Math.floor(prog(t, 1, 13) * EPOCHS)), a0 = eout(prog(t, 0.5, 0.8));
    const A1 = 1 - ease(prog(t, 22.5, 1));
    if (A1 > 0) { GA = A1;
      for (let i = 0; i < 16; i++) drawImg(cached(`m${e}_${i}`, () => imgCanvas(MSNAP[e][i])), 150 + (i % 4) * 150, 250 + Math.floor(i / 4) * 150, 140, a0);
      text(`GAN · epoch ${e}`, 440, 880, { font: F.mono, size: 22, color: P.chalk, align: 'center', a: a0 });
      // D(real), D(fake) curves
      const x0 = 800, y0 = 560, w = 400, h = 220, ca = eout(prog(t, 3, 1)); rbox(x0 - 20, y0 - h - 60, w + 40, h + 110, 14, rgba(P.surface, 0.9), P.border, ca);
      text('judge’s average verdict', x0, y0 - h - 24, { font: F.mono, size: 16, color: P.dust, a: ca });
      [['d_real', P.ink, 'on real'], ['d_fake', P.terracotta, 'on fakes']].forEach(([k, col, nm], j) => { setA(ca); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath(); HIST.slice(0, Math.max(1, e)).forEach((hh, i) => { const X = x0 + i / (EPOCHS - 1) * w, Yv = y0 - hh[k] * h; i ? ctx.lineTo(X, Yv) : ctx.moveTo(X, Yv); }); ctx.stroke(); setA(1); text(nm, x0 + w + 10, y0 - HIST[Math.max(0, e - 1)][k] * h + 6, { font: F.mono, size: 15, color: col, a: ca }); });
      line(x0, y0 - 0.5 * h, x0 + w, y0 - 0.5 * h, P.dust, 1, ca, [4, 6]); text('½', x0 - 18, y0 - 0.5 * h + 6, { font: F.mono, size: 15, color: P.dust, a: ca });
      const va = eout(prog(t, 15, 0.8));
      for (let i = 0; i < 16; i++) drawImg(cached(`v${i}`, () => imgCanvas(VAE16[i])), 1320 + (i % 4) * 124, 250 + Math.floor(i / 4) * 124, 116, va);
      text('VAE (episode 3)', 1564, 880, { font: F.mono, size: 22, color: P.stone, align: 'center', a: va });
      GA = 1; }
    const B = eout(prog(t, 23, 1));
    if (B > 0) {
      for (let i = 0; i < 36; i++) drawImg(cached(`g${i}`, () => imgCanvas(GAN64[i])), 150 + (i % 6) * 104, 250 + Math.floor(i / 6) * 104, 98, B);
      const rows = [['FID ↓', 'FID', v => v.toFixed(1)], ['precision ↑', 'precision', pct], ['recall ↑', 'recall', v => v < 0.01 ? '<1%' : pct(v)], ['IS ↑', 'IS', v => v.toFixed(2)], ['C2ST → 50%', 'C2ST', pct], ['NN-train', 'NN_train', v => v.toFixed(2)]];
      const x0 = 900; card(x0, 250, 920, 620, P.plum, B); label("episode 4's judges, 2,000 samples each", x0 + 34, 296, B, P.plum);
      [['GAN', 360], ['VAE', 560], ['real', 740]].forEach(([nm, dx]) => text(nm, x0 + dx, 350, { font: F.mono, size: 18, color: nm === 'GAN' ? P.terracotta : P.dust, a: B }));
      rows.forEach(([nm, k, f], i) => { const y = 420 + i * 72, ai = B * eout(prog(t, 23.5 + i * 0.4, 0.5)); text(nm, x0 + 34, y, { font: F.mono, size: 20, color: P.stone, a: ai });
        text(f(GS[k]), x0 + 360, y, { font: F.mono, size: 24, color: P.chalk, a: ai }); text(f(VS[k]), x0 + 560, y, { font: F.mono, size: 22, color: P.dust, a: ai }); text(f(R4.real[k]), x0 + 740, y, { font: F.mono, size: 22, color: P.dust, a: ai }); });
    }
  } });

// 07 — mode collapse
SC.push({ dur: 48, chapter: 'collapse', title: 'Mode collapse, caught in the act', sub: 'Same networks — a fast forger and a slow judge.',
  caps: [[1, 8, 'Now a less lucky run: the same networks, but a fast forger and a slow judge.'],
         [8, 16, 'The generator piles almost everything onto a few blobs. Why not? The judge currently likes those spots.'],
         [16, 24, 'The judge slowly catches on and darkens them — so the crowd jumps to another blob. Then another.'],
         [24, 32, "This is mode collapse — the one-trick pony's disease from episodes 3 and 4, caught in the act. Our recall judge would flag it instantly."],
         [32, 40, "Nothing in the generator's objective rewards variety. Each step only asks: what fools the judge right now?"],
         [40, 47.6, 'Fixes exist — minibatch discrimination, Wasserstein losses, spectral normalisation — but this remains the GAN’s signature disease.']],
  draw(t) {
    const n = A.hop.length, fi = lerp(0, n - 1, ease(prog(t, 2, 38)));
    arena('hop', fi, 560, 240, 700, { a: eout(prog(t, 0.5, 1)), field: 1, fakes: 1 });
    // timeline of the dominant mode
    const x0 = 1330, y0 = 300, a = eout(prog(t, 8, 1)); text('most crowded blob, over time', x0, y0 - 20, { font: F.mono, size: 16, color: P.dust, a });
    for (let i = 0; i <= Math.floor(fi); i += 1) { const d = dominant('hop', i), X = x0 + i / (n - 1) * 480, Yv = y0 + 20 + d.k * 36; dot(X, Yv, 4, P.rose, a * clamp(d.share * 3)); }
    for (let k = 0; k < 8; k++) text(`#${k + 1}`, x0 - 36, y0 + 26 + k * 36, { font: F.mono, size: 13, color: P.dust, a });
    const i = Math.round(fi), d = dominant('hop', i);
    text(`${Math.round(d.share * 100)}% of fakes on blob #${d.k + 1}`, x0, y0 + 340, { font: F.mono, size: 20, color: P.rose, a });
    insight(1330, 700, 490, "G's objective only asks “does this fool D now?” — never “is there variety?”", eout(prog(t, 32, 0.8)));
    stamp('5 MODES MISSING', 910, 610, P.rose, eout(prog(t, 25, 0.6)));
  } });

// 08 — a game, not a hill
const SPIRAL = (() => { const p = [[1, 0.2]]; let x = 1, y = 0.2; const eta = 0.18; for (let i = 0; i < 70; i++) { const nx = x - eta * y, ny = y + eta * x; x = nx; y = ny; p.push([x, y]); } return p; })();
SC.push({ dur: 46, chapter: 'the game', title: 'A game is not a hill', sub: 'Why GAN training is so temperamental.',
  caps: [[1, 8, "Why is training so temperamental? Because a GAN isn't rolling downhill. It's a game — two players pulling in different directions."],
         [8, 16, 'The simplest game: V = x · y. Player x wants V small, player y wants it big. The fair outcome sits in the middle, at zero.'],
         [16, 24, 'Take gradient steps for both at once, and instead of settling, they circle — and spiral outward. Every step overshoots.'],
         [24, 32, "Second problem: if real and fake don't overlap at all, the JSD is stuck at log 2, no matter how far apart they are. Flat. No gradient."],
         [32, 45.6, "Measure how far the mass has to move instead — the Wasserstein, or earth-mover's, distance — and the slope comes back. That idea powers the WGAN."]],
  draw(t) {
    const A1 = 1 - ease(prog(t, 23.5, 1));
    if (A1 > 0) { GA = A1; const cx = 560, cy = 590, s = 150, a0 = eout(prog(t, 8, 1));
      line(cx - 360, cy, cx + 360, cy, P.border, 1.5, a0); line(cx, cy - 330, cx, cy + 330, P.border, 1.5, a0); text('x (minimises)', cx + 360, cy + 30, { font: F.mono, size: 16, color: P.dust, align: 'right', a: a0 }); text('y (maximises)', cx + 12, cy - 310, { font: F.mono, size: 16, color: P.dust, a: a0 });
      glow(cx, cy, 7, P.sage, a0); text('equilibrium', cx + 14, cy + 26, { font: F.mono, size: 15, color: P.sage, a: a0 });
      const k = Math.floor(SPIRAL.length * ease(prog(t, 16, 7))); setA(a0); ctx.strokeStyle = P.rose; ctx.lineWidth = 2.5; ctx.beginPath(); SPIRAL.slice(0, Math.max(1, k)).forEach(([x, y], i) => { const X = cx + x * s, Y = cy - y * s; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
      if (k > 0) { const [x, y] = SPIRAL[k - 1]; glow(cx + x * s, cy - y * s, 7, P.rose, a0); }
      const ca = eout(prog(t, 8, 0.8)); card(1150, 300, 670, 330, P.plum, ca); label('simultaneous gradient steps', 1184, 346, ca, P.plum);
      M('V(x, y) = x · y', 1184, 410, 30, { a: ca }); M('x ← x − η·∂V/∂x = x − η·y', 1184, 470, 24, { a: eout(prog(t, 16, 0.8)) }); M('y ← y + η·∂V/∂y = y + η·x', 1184, 520, 24, { a: eout(prog(t, 16, 0.8)) });
      text('each step multiplies the radius by √(1 + η²) > 1', 1184, 585, { font: F.mono, size: 16, color: P.rose, a: eout(prog(t, 18, 0.8)) });
      GA = 1; }
    const B = eout(prog(t, 24, 1));
    if (B > 0) { const x0 = 200, y0 = 800, W = 800, Hh = 400;
      line(x0, y0, x0 + W, y0, P.borderLight, 2, B); line(x0 + W / 2, y0, x0 + W / 2, y0 - Hh - 20, P.borderLight, 1.5, B);
      text('θ: how far apart the real and fake point masses sit →', x0 + W, y0 + 40, { font: F.mono, size: 16, color: P.dust, align: 'right', a: B });
      setA(B); ctx.strokeStyle = P.rose; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(x0, y0 - Math.log(2) / 3 * Hh); ctx.lineTo(x0 + W / 2 - 4, y0 - Math.log(2) / 3 * Hh); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x0 + W / 2 + 4, y0 - Math.log(2) / 3 * Hh); ctx.lineTo(x0 + W, y0 - Math.log(2) / 3 * Hh); ctx.stroke(); setA(1); dot(x0 + W / 2, y0, 6, P.rose, B);
      text('JSD: flat at log 2', x0 + 20, y0 - Math.log(2) / 3 * Hh - 16, { font: F.mono, size: 18, color: P.rose, a: B });
      const wa = eout(prog(t, 32, 1)); setA(wa); ctx.strokeStyle = P.sage; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(x0, y0 - Hh); ctx.lineTo(x0 + W / 2, y0); ctx.lineTo(x0 + W, y0 - Hh); ctx.stroke(); setA(1);
      text('Wasserstein = |θ|: always a slope', x0 + W - 20, y0 - Hh - 14, { font: F.mono, size: 18, color: P.sage, align: 'right', a: wa });
      card(1150, 380, 670, 300, P.sage, wa); label("earth mover's distance", 1184, 426, wa, P.sage);
      para('How much “dirt” must move, and how far, to reshape the fakes into the data — even when they don’t overlap at all.', 1184, 486, 600, { size: 24, a: wa });
      text('→ the Wasserstein GAN (2017)', 1184, 640, { font: F.mono, size: 18, color: P.sage, a: wa }); }
  } });

// 09 — GANs in the world
SC.push({ dur: 36, chapter: 'in the wild', title: 'GANs in the wild', sub: 'Temperamental — and transformative.',
  caps: [[1, 8, 'Despite the tantrums, GANs changed everything. By 2019, StyleGAN was drawing faces of people who never existed.'],
         [8, 15, 'Sampling is a single forward pass — no 784 sequential steps as in autoregression — so GANs are fast.'],
         [15, 23, 'And the adversarial idea spread everywhere: super-resolution, image translation, even the image compressor inside Stable Diffusion is trained with a GAN judge.'],
         [23, 35.6, 'The same power gave us deepfakes — and a brand-new job for episode 4’s learned judges: detecting them.']],
  draw(t) {
    const cards = [['photorealism', 'StyleGAN (2019): faces of people who never existed; “style” knobs for age, pose, hair', 1],
                   ['speed', 'one forward pass per sample — vs 784 sequential steps for a pixel-by-pixel model', 8],
                   ['the idea spreads', 'super-resolution · pix2pix · and the autoencoder inside Stable Diffusion trains with a GAN-style judge', 15],
                   ['the dark side', 'deepfakes — and learned detectors: episode 4’s two-sample test, deployed', 23]];
    cards.forEach(([nm, d, t0], i) => { const x = 150 + (i % 2) * 830, y = 260 + Math.floor(i / 2) * 330, a = eout(prog(t, t0, 0.8)); card(x, y, 780, 290, i === 3 ? P.rose : P.plum, a);
      text(nm, x + 36, y + 70, { font: F.serif, size: 44, color: P.chalk, a }); para(d, x + 36, y + 130, 700, { size: 25, a }); });
  } });

// 10 — build it
const CODE = [
  '[k|# the judge: real → 1, fake → 0   (binary cross-entropy on logits)]',
  'x_fake = G.forward([n|z])',
  'l_real, l_fake = D.forward([d|x_real]), D.forward(x_fake)',
  'd_loss = softplus(-l_real).mean() + softplus(l_fake).mean()',
  'D.backward(...);  D.step([a|lr])',
  '',
  '[k|# the forger: make the judge say "real"   (-log D(G(z)))]',
  'l = D.forward(G.forward([n|z]))',
  'g_loss = softplus(-l).mean()',
  '',
  '[k|# the judge’s gradient flows back through the pixels…]',
  'dx = D.backward((sigmoid(l) - [a|1]) / B)',
  '[k|# …and on into the generator’s weights]',
  'G.backward(dx);  G.step([a|lr])',
];
SC.push({ dur: 32, chapter: 'build it', title: 'Build it yourself', sub: 'The whole game in a few dozen lines of numpy.',
  caps: [[1, 9, 'The whole game fits in a few dozen lines of numpy. The full code, with hand-written backprop and gradient checks, is linked below.'],
         [9, 18, 'Try the ring and change the learning rates to watch mode collapse happen; train on digits in about ten minutes.'],
         [18, 31.6, 'Recap: the judge learns pdata / (pdata + pg), the forger minimises the Jensen–Shannon divergence, and the finish line is a judge stuck at one half.']],
  draw(t) {
    playPill('ep05-gan-arena.html', 9, t);
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1060, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 296 + i * 42, 20, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8)); card(1200, 240, 620, 690, P.sage, ra); label('the whole episode', 1236, 290, ra, P.sage);
    [['the game', 'min[m_|G] max[m_|D] [p|E] log [m|D]([d|x]) + [p|E] log(1 − [m|D]([m|G]([n|z])))'], ['perfect judge', '[m|D]* = [p|p][p_|d] / ([p|p][p_|d] + [p|p][p_|g])'], ['what G minimises', '2·JSD([p|p][p_|d] ‖ [p|p][p_|g]) − log 4'], ['in practice', 'min[m_|G] −log [m|D]([m|G]([n|z]))'], ['watch out', 'mode collapse · oscillation']].forEach(([nm, m], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.6, 0.8)); text(nm, 1236, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(m, 1236, 412 + i * 112, i === 0 ? 16 : 22, { a: ai }); });
  } });

// 11 — hook → normalizing flows
function flowMap(z1, z2, u) { // an invertible 2-step affine coupling, drawn part-way (u ∈ [0,1])
  const s1 = 0.4 * Math.sin(z1), a2 = z2 * Math.exp(s1) + 0.7 * Math.cos(1.2 * z1);
  const s2 = 0.25 * Math.cos(a2), a1 = z1 * Math.exp(s2) + 0.12 * a2 * a2 - 0.3;
  return [[lerp(z1, a1, u), lerp(z2, a2, u)], s1 + s2];
}
const FLOWPTS = (() => { const r = rng(17), out = []; for (let i = 0; i < 900; i++) { const u = 1 - r(), v = r(), rr = Math.sqrt(-2 * Math.log(u)); out.push([rr * Math.cos(TAU * v), rr * Math.sin(TAU * v)]); } return out; })();
SC.push({ dur: 48, chapter: 'next', title: 'The question a GAN cannot answer', sub: 'How likely is this image?',
  caps: [[1, 8, 'GANs make gorgeous samples. But ask a GAN one simple question — how likely is this image? — and it has no answer.'],
         [8, 15, "Episode 4's first judge, likelihood, can't even be computed. G maps noise to images, but it can't be run backwards."],
         [15, 23, 'Remember episode 1? When a function bends noise into data, density changes by exactly how much it stretches space: p(x) = p(z) · |dz/dx|.'],
         [23, 31, 'So what if the generator were perfectly reversible — every image traced back to exactly one noise point?'],
         [31, 38, 'Then we could track the stretching exactly, compute p(x) for any image, and train by plain maximum likelihood. No judge needed.'],
         [38, 47.6, 'Generators that run both ways, and bend space exactly: next episode, normalizing flows.']],
  draw(t) {
    const dimAll = 1 - 0.9 * ease(prog(t, 38.5, 1));
    GA = dimAll;
    const a0 = eout(prog(t, 0.6, 0.8));
    rbox(150, 300, 250, 150, 20, rgba(P.plum, 0.12), rgba(P.plum, 0.75), a0, 2.5); text('G', 275, 395, { font: F.serif, size: 60, color: P.chalk, align: 'center', a: a0 });
    arrow(150 - 90, 375, 140, 375, P.rose, 2.5, a0); text('z', 40, 385, { font: F.mono, size: 28, color: P.rose, a: a0 }); arrow(410, 375, 500, 375, P.terracotta, 2.5, a0); text('x', 515, 385, { font: F.mono, size: 28, color: P.terracotta, a: a0 });
    const na = eout(prog(t, 8, 0.8)); arrow(500, 440, 410, 440, P.dust, 2, na); text('✕', 455, 450, { font: F.mono, size: 34, color: P.rose, align: 'center', a: na }); text('p(x) = ?', 275, 520, { font: F.mono, size: 30, color: P.rose, align: 'center', a: na });
    // the warp
    const wa = eout(prog(t, 15, 1)), u = t < 23 ? 0 : 0.5 - 0.5 * Math.cos(Math.PI * clamp((t - 23) / 7)) * (t < 31 ? 1 : Math.cos(Math.PI * clamp((t - 31) / 3)));
    const uu = t < 23 ? 0 : t < 31 ? ease(prog(t, 23, 6)) : t < 34 ? 1 - ease(prog(t, 31, 2.5)) * 0 : 1;
    const cx = 1330, cy = 590, s = 95;
    if (wa > 0) { setA(wa * 0.5); ctx.strokeStyle = P.border; ctx.lineWidth = 1.2;
      for (let g = -2.5; g <= 2.51; g += 0.5) { ctx.beginPath(); for (let k = 0; k <= 60; k++) { const [[x, y]] = flowMap(g, -2.5 + 5 * k / 60, uu); const X = cx + x * s, Y = cy - y * s; k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); } ctx.stroke();
        ctx.beginPath(); for (let k = 0; k <= 60; k++) { const [[x, y]] = flowMap(-2.5 + 5 * k / 60, g, uu); const X = cx + x * s, Y = cy - y * s; k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); } ctx.stroke(); } setA(1);
      FLOWPTS.forEach(([z1, z2]) => { const [[x, y], ld] = flowMap(z1, z2, uu); dot(cx + x * s, cy - y * s, 2.4, mixc(P.rose, P.terracotta, uu), wa * 0.85); });
      text(uu < 0.5 ? 'noise z' : 'data x = f(z)', cx, 250, { font: F.mono, size: 20, color: uu < 0.5 ? P.rose : P.terracotta, align: 'center', a: wa });
      if (t > 31) text('…and back again:  z = f⁻¹(x)', cx, 950 - 20, { font: F.mono, size: 20, color: P.sage, align: 'center', a: eout(prog(t, 31, 0.8)) }); }
    const ma = eout(prog(t, 15, 0.8)); card(150, 600, 700, 250, P.ink, ma); label('episode 1, change of variables', 184, 646, ma, P.ink);
    M('[p|p][p_|x]([d|x]) = [p|p][p_|z]([n|z]) · |d[n|z]/d[d|x]|', 184, 716, 30, { a: ma });
    M('in many dimensions:  · |det ∂[n|z]/∂[d|x]|', 184, 786, 24, { a: eout(prog(t, 25, 0.8)) });
    GA = 1;
    const qa = eout(prog(t, 38.5, 1.2));
    if (qa > 0) {
      text('Can a generator', 960, 440, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('run backwards?', 960, 530, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('NEXT  ·  EPISODE 06', 960, 640, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 40, 1)), ls: 4 });
      seasonStrip(5, 41.2, t);
      text('Normalizing flows — bending space, exactly', 960, 695, { font: F.serif, italic: true, size: 42, color: P.stone, align: 'center', a: eout(prog(t, 40.5, 1)) });
    }
  } });
