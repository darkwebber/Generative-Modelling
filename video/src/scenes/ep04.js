// ─────────────────────────────────────────────────────────────
//  Real measurements (exported by code/export_ep04_assets.py)
// ─────────────────────────────────────────────────────────────
const A = window.EP4, R = A.rows;
const S = {}; for (const k in A.samples) S[k] = u8imgs(A.samples[k]);
const SEVENS = u8imgs(A.sevens), PER = u8imgs(A.percept.imgs);
const EXS = u8imgs(A.examples.sharpImg)[0], EXA = u8imgs(A.examples.ambImg)[0];
const PAIRS = {}; for (const k in A.nnPairs) PAIRS[k] = { s: u8imgs(A.nnPairs[k].s), n: u8imgs(A.nnPairs[k].n), d: A.nnPairs[k].d };
const img = (key, vals, col) => cached(key, () => imgCanvas(vals, col));
function gridOf(key, arr, x, y, n, cell, a = 1) { for (let i = 0; i < n * n && i < arr.length; i++) drawImg(img(`${key}${i}`, arr[i]), x + (i % n) * cell, y + Math.floor(i / n) * cell, cell - 6, a); }
const NOISE = (() => { const r = rng(4), out = []; for (let k = 0; k < 100; k++) { const v = new Float32Array(784); for (let i = 0; i < 784; i++) v[i] = r(); out.push(v); } return out; })();
const NAMES = { real: 'real (held-out)', vae: 'A · VAE', copier: 'B · photocopier', ones: 'C · one-trick pony', noise: 'static' };
const ORDER = ['real', 'vae', 'copier', 'ones', 'noise'];
const ELBO = A.vaeTest.recon + A.vaeTest.kl, BPD = ELBO / (784 * Math.LN2);
const pct = v => `${Math.round(v * 100)}%`;
// horizontal bar chart of one metric across models
function barChart(metric, x, y, w, o = {}) {
  const { a = 1, max, log = false, fmt = v => v.toFixed(1), better = 'high', rowH = 56, reveal = 1, hl = null, mark = null, models = ORDER } = o;
  models.forEach((k, i) => {
    const ai = a * eout(clamp(reveal * models.length - i)); if (ai <= 0) return;
    const v = R[k][metric], yy = y + i * rowH, frac = log ? Math.log10(1 + v) / Math.log10(1 + max) : v / max;
    text(NAMES[k], x, yy + 24, { font: F.mono, size: 18, color: k === hl ? P.chalk : P.stone, a: ai });
    rbox(x + 230, yy + 4, w - 330, 28, 6, P.surface, P.border, ai, 1);
    setA(ai); ctx.fillStyle = k === hl ? P.terracotta : rgba(P.plum, 0.85); ctx.fillRect(x + 230, yy + 4, (w - 330) * clamp(frac), 28); setA(1);
    text(fmt(v), x + w - 10, yy + 26, { font: F.mono, size: 20, color: P.chalk, align: 'right', a: ai });
  });
  if (mark != null) { const X = x + 230 + (w - 330) * mark[0]; line(X, y - 10, X, y + models.length * rowH, P.sage, 2, a, [6, 6]); text(mark[1], X, y + models.length * rowH + 22, { font: F.mono, size: 16, color: P.sage, align: 'center', a }); }
  text(better === 'high' ? 'higher is better ↑' : 'lower is better ↓', x + w - 10, y - 18, { font: F.mono, size: 15, color: P.dust, align: 'right', a });
}
// 2-D feature-space scatter (PCA of the 64-d features; metrics are computed in 64-d)
const PCA_B = (() => { const all = A.pca.ref.concat(A.pca.real, A.pca.vae, A.pca.copier, A.pca.ones); const xs = all.map(p => p[0]).sort((a, b) => a - b), ys = all.map(p => p[1]).sort((a, b) => a - b); const q = (arr, f) => arr[Math.floor(f * (arr.length - 1))]; return [q(xs, .01), q(xs, .99), q(ys, .01), q(ys, .99)]; })();
function pmap(x, y, w, h) { const b = PCA_B, s = Math.min(w / (b[1] - b[0]), h / (b[3] - b[2])) * 0.9, cx = x + w / 2, cy = y + h / 2, mx = (b[0] + b[1]) / 2, my = (b[2] + b[3]) / 2; return { X: v => cx + (v - mx) * s, Y: v => cy - (v - my) * s, s, x, y, w, h }; }
function clip(m, fn) { ctx.save(); ctx.beginPath(); ctx.rect(m.x, m.y, m.w, m.h); ctx.clip(); fn(); ctx.restore(); }
function gaussEllipse(pts) { const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n; let a = 0, b = 0, c = 0; for (const p of pts) { a += (p[0] - mx) ** 2; b += (p[0] - mx) * (p[1] - my); c += (p[1] - my) ** 2; } a /= n; b /= n; c /= n;
  const tr = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b); return { mx, my, l1: tr + d, l2: Math.max(1e-6, tr - d), ang: 0.5 * Math.atan2(2 * b, a - c) }; }
function drawEllipse(m, e, col, a, k = 1) { setA(a * 0.1); ctx.fillStyle = col; ctx.beginPath(); ctx.ellipse(m.X(e.mx), m.Y(e.my), k * Math.sqrt(e.l1) * m.s, k * Math.sqrt(e.l2) * m.s, -e.ang, 0, TAU); ctx.fill(); setA(a); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.stroke(); setA(1); glow(m.X(e.mx), m.Y(e.my), 6, col, a); }
const kth = (pts, k = 3) => pts.map((p, i) => { const d = pts.map(q => Math.hypot(p[0] - q[0], p[1] - q[1])).sort((a, b) => a - b); return d[k]; });
const REF150 = A.pca.ref.slice(0, 150), REF_R = kth(REF150);
const GEN_R = {}; ['vae', 'ones', 'copier'].forEach(k => GEN_R[k] = kth(A.pca[k].slice(0, 150)));

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open
SC.push({ dur: 24, chapter: 'intro',
  caps: [[0.6, 6, 'Last time: three models. A is a little blurry. B is razor sharp. C is sharp and confident.'],
         [6, 12, 'B was a photocopier, and C only ever draws ones. How would a computer have caught them?'],
         [12, 18, 'Today we build the judges — one at a time, from first principles.'],
         [18, 23.6, "And we'll catch every single one of them lying."]],
  draw(t) {
    const dim = lerp(1, 0.16, ease(prog(t, 12, 1.5)));
    GA = dim;
    [['vae', 'A'], ['copier', 'B'], ['ones', 'C']].forEach(([k, nm], p) => {
      const x = 170 + p * 560, a = eout(prog(t, 0.4 + p * 0.4, 0.8));
      card(x, 250, 480, 560, P.plum, a); text(`model ${nm}`, x + 30, 300, { font: F.serif, size: 34, color: P.chalk, a });
      gridOf(k, S[k], x + 40, 330, 4, 102, a);
    });
    stamp('COPIED', 170 + 560 + 240, 560, P.rose, eout(prog(t, 7, 0.6))); stamp('ONLY 1s', 170 + 1120 + 240, 560, P.rose, eout(prog(t, 8.5, 0.6)));
    GA = 1;
    const a = eout(prog(t, 12.6, 1.4));
    text('EPISODE 04', 960, 430, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('Grading the Imagination', 960, 540 + (1 - a) * 16, { font: F.serif, size: 112, color: P.chalk, align: 'center', a, ls: -1 });
    text('how to evaluate generative models — and how every metric can lie', 960, 612, { size: 32, color: P.stone, align: 'center', a: eout(prog(t, 13.4, 1.2)) });
    const tg = [[P.terracotta, 'x  samples & data'], [P.ink, 'p  distributions'], [P.plum, 'judges & models'], [P.sage, 'verdicts'], [P.rose, 'caught lying']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let cx = 960 - tw / 2; tg.forEach(([c, s], i) => { cx += tag(cx, 670, c, s, eout(prog(t, 16 + i * 0.35, 0.6)), 21) + 14; });
  } });

// 01 — what does "good" mean?
SC.push({ dur: 40, chapter: 'good?', title: 'What does “good” even mean?', sub: 'Grading a classifier is easy. Grading a generator is not.',
  caps: [[1, 7, 'Grading a classifier is easy: there is an answer key. This image is a 7 — did you say 7?'],
         [7, 14, "A generator has no answer key. Ask for 'a seven' and there are infinitely many right answers."],
         [14, 21, 'We want the whole distribution to match — pθ ≈ pdata — and we only ever see samples of each.'],
         [21, 29, 'So every metric is a way of comparing two clouds of points. In practice, we check three things.'],
         [29, 39.6, 'Fidelity: does each sample look real? Diversity: do samples cover everything real data does? Novelty: are they new, or copies?']],
  draw(t) {
    const A1 = 1 - ease(prog(t, 20.5, 1));
    if (A1 > 0) { GA = A1;
      const a = eout(prog(t, 0.6, 0.8));
      card(150, 260, 700, 440, P.sage, a); label('classifier', 184, 306, a, P.sage);
      drawImg(img('seven0', SEVENS[0]), 190, 340, 200, a, { frame: true }); arrow(410, 440, 500, 440, P.dust, 2, a);
      text('“7”', 590, 460, { font: F.mono, size: 60, color: P.chalk, align: 'center', a }); text('✓', 740, 460, { font: F.mono, size: 60, color: P.sage, align: 'center', a: eout(prog(t, 3, 0.6)) });
      text('one right answer', 500, 640, { font: F.serif, italic: true, size: 30, color: P.sage, align: 'center', a });
      const b = eout(prog(t, 7, 0.8));
      card(1070, 260, 700, 440, P.plum, b); label('generator', 1104, 306, b, P.plum);
      text('“draw me a 7”', 1420, 370, { font: F.mono, size: 30, color: P.chalk, align: 'center', a: b });
      SEVENS.forEach((s, i) => drawImg(img('seven' + i, s), 1110 + (i % 4) * 160, 400 + Math.floor(i / 4) * 150, 140, b * eout(prog(t, 8 + i * 0.3, 0.5))));
      text('all correct — so what is “the” answer?', 1420, 740, { font: F.serif, italic: true, size: 26, color: P.dust, align: 'center', a: b });
      const m = eout(prog(t, 14, 0.8)); M('goal:  [p|p][a_|θ]  ≈  [p|p][k_|data]      but we only see samples of each', 960, 830, 30, { align: 'center', a: m });
      GA = 1; }
    const B = eout(prog(t, 21, 1));
    if (B > 0) {
      const cards = [['fidelity', 'does each sample look real?', 21.5], ['diversity', 'do samples cover all of the real data?', 31.5], ['novelty', 'are they new — or copies?', 35]];
      cards.forEach(([nm, q, t0], i) => { const x = 170 + i * 540, a = eout(prog(t, t0, 0.8));
        card(x, 330, 480, 360, P.sage, a); text(nm, x + 40, 420, { font: F.serif, size: 54, color: P.chalk, a }); para(q, x + 40, 490, 400, { size: 28, a }); });
      text('three questions every judge must answer', 960, 790, { font: F.serif, italic: true, size: 30, color: P.dust, align: 'center', a: B });
    }
  } });

// 02 — judge 1: likelihood
SC.push({ dur: 50, chapter: 'likelihood', title: 'Judge 1 · Likelihood', sub: 'How probable does the model find real data it has never seen?',
  caps: [[1, 8, 'Judge one comes from episode 1: likelihood. Show the model real images it has never seen, and ask how probable they are.'],
         [8, 15, `Our VAE: about ${ELBO.toFixed(0)} nats per image — ${BPD.toFixed(2)} bits per pixel, under its own pixel model. The bits it needs per pixel.`],
         [15, 23, "Now the photocopier. Its 'distribution' is just a pile of training images. A brand-new test digit gets probability zero."],
         [23, 30, 'log 0 = −∞: infinitely bad. Likelihood catches the copier instantly — and the one-trick pony too.'],
         [30, 38, 'Why? Likelihood measures coverage. Every real image must get some probability, or you pay without limit.'],
         [38, 45, 'Two catches. Many great generators — including next episode’s — cannot compute a likelihood at all.'],
         [45, 49.6, 'And even when they can, likelihood can lie.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8));
    card(150, 250, 800, 230, P.ink, a0); label('held-out likelihood', 184, 296, a0, P.ink);
    M('score = [kB|Σ][k_|x ∈ test] −log [p|p][a_|θ]([d|x])   (lower = better)', 184, 370, 26, { a: a0 });
    M(`bits per pixel = −log [p|p][a_|θ]([d|x]) / (D · ln 2)`, 184, 430, 24, { a: eout(prog(t, 9, 0.8)) });
    // falling test digits into models
    const models = [['A · VAE', 'vae', 8], ['B · photocopier', 'copier', 15], ['C · one-trick pony', 'ones', 24]];
    models.forEach(([nm, k, t0], i) => {
      const x = 1060 + i * 260, a = eout(prog(t, t0, 0.8)); if (a <= 0) return;
      rbox(x, 380, 220, 110, 16, rgba(P.plum, 0.12), rgba(P.plum, 0.7), a, 2); text(nm, x + 110, 445, { font: F.mono, size: 17, color: P.chalk, align: 'center', a });
      const fall = ((t - t0) % 3) / 3; drawImg(img('seven3', SEVENS[3]), x + 80, lerp(250, 360, ein(fall)), 60, a * (1 - fall));
      const v = k === 'vae' ? `≤ ${ELBO.toFixed(1)}` : '= ∞'; const col = k === 'vae' ? P.chalk : P.rose;
      text(`−log p ${v}`, x + 110, 540, { font: F.mono, size: 22, color: col, align: 'center', a });
      if (k !== 'vae') text('p(x_test) = 0', x + 110, 575, { font: F.mono, size: 16, color: P.rose, align: 'center', a });
      if (k === 'vae') text(`${BPD.toFixed(2)} bits / pixel`, x + 110, 575, { font: F.mono, size: 16, color: P.dust, align: 'center', a });
    });
    stamp('CAUGHT', 1060 + 260 + 110, 660, P.sage, eout(prog(t, 24, 0.6)), 0.08); stamp('CAUGHT', 1060 + 520 + 110, 660, P.sage, eout(prog(t, 25, 0.6)), -0.06);
    const ca = eout(prog(t, 30, 0.8));
    insight(150, 560, 800, 'Likelihood measures coverage: skip any part of the real data and your score goes to infinity.', ca * (1 - ease(prog(t, 38, 0.6))));
    const cb = eout(prog(t, 38.5, 0.8));
    if (cb > 0) { card(150, 560, 800, 300, P.rose, cb); label('two catches', 184, 606, cb, P.rose);
      para('1 · Many generators (GANs, next episode) have no likelihood at all: you can sample, but not score.', 184, 666, 730, { size: 25, a: cb });
      para('2 · Even when you can compute it… it can lie.', 184, 776, 730, { size: 25, a: eout(prog(t, 45, 0.8)) }); }
  } });

// 03 — how likelihood lies
SC.push({ dur: 42, chapter: 'the lie', title: 'How likelihood lies', sub: 'A model that is 99% garbage — and scores almost as well.',
  caps: [[1, 8, 'Build a sneaky model: flip a coin weighted 99 to 1. On 99, output static. On 1, run our VAE.'],
         [8, 15, 'Its samples are 99% garbage. Look at them.'],
         [15, 23, "Its likelihood? pmix(x) ≥ 0.01 · pVAE(x). Take logs, and we lose at most log 100 ≈ 4.6 nats."],
         [23, 31, `${ELBO.toFixed(1)} becomes at most ${(ELBO + Math.log(100)).toFixed(1)} nats per image: about 3% worse, for samples that are 99% noise.`],
         [31, 41.6, 'Likelihood rewards covering the data; it barely notices the junk you spray everywhere else. We need judges that look at samples.']],
  draw(t) {
    const g = eout(prog(t, 1, 1)), goodIdx = 57;
    for (let i = 0; i < 100; i++) { const x = 150 + (i % 10) * 64, y = 250 + Math.floor(i / 10) * 64, a = g * eout(prog(t, 2 + i * 0.05, 0.3));
      drawImg(i === goodIdx ? img('vae3', S.vae[3]) : img('nz' + i, NOISE[i], P.rose), x, y, 58, a, { frame: i === goodIdx && t > 9, col: P.sage }); }
    text('99 × static  +  1 × VAE', 470, 930 - 20, { font: F.mono, size: 20, color: P.dust, align: 'center', a: g });
    const ma = eout(prog(t, 15, 0.8));
    card(900, 250, 920, 330, P.ink, ma); label('the math', 936, 296, ma, P.ink);
    M('[p|p][k_|mix]([d|x]) = 0.01·[p|p][k_|VAE]([d|x]) + 0.99·[p|p][k_|static]([d|x])', 936, 366, 28, { a: ma });
    M('               ≥ 0.01·[p|p][k_|VAE]([d|x])', 936, 420, 28, { a: eout(prog(t, 17, 0.8)) });
    M('−log [p|p][k_|mix]([d|x]) ≤ −log [p|p][k_|VAE]([d|x]) + log 100', 936, 480, 28, { a: eout(prog(t, 19, 0.8)) });
    text(`log 100 = ${Math.log(100).toFixed(2)} nats`, 936, 540, { font: F.mono, size: 20, color: P.dust, a: eout(prog(t, 20, 0.8)) });
    const ba = eout(prog(t, 23, 0.8));
    if (ba > 0) { const x = 900, y = 640, W = 920; text('−log p per test image (lower = better)', x, y, { font: F.mono, size: 17, color: P.dust, a: ba });
      [['VAE', ELBO, P.plum], ['99% garbage', ELBO + Math.log(100), P.rose]].forEach(([nm, v, c], i) => { const yy = y + 30 + i * 70; text(nm, x, yy + 30, { font: F.mono, size: 20, color: P.chalk, a: ba });
        setA(ba); ctx.fillStyle = c; ctx.fillRect(x + 190, yy + 8, (W - 330) * (v / 170) * eout(prog(t, 23.5 + i * 0.4, 1)), 30); setA(1); text(`≤ ${v.toFixed(1)}`, x + W - 10, yy + 32, { font: F.mono, size: 20, color: P.chalk, align: 'right', a: ba }); }); }
  } });

// 04 — a better eye: features
SC.push({ dur: 40, chapter: 'features', title: 'Judging samples needs a better eye', sub: 'Pixels are a terrible way to compare images.',
  caps: [[1, 8, 'To judge samples, we must measure how similar images are. Pixels are the obvious choice — and they are terrible.'],
         [8, 15, 'Slide this seven three pixels to the right. In pixel space, it is now farther from itself than this one is!'],
         [15, 23, 'So we borrow an eye: a network trained to recognise digits. Its inner activations — features — capture what an image is, not where each pixel sits.'],
         [23, 31, 'In feature space, the other seven is close and the one is far away. For photos, people use the Inception network in exactly this way.'],
         [31, 39.6, 'A warning, though: the shifted seven still looks oddly far. Every feature-based metric inherits the blind spots of the network it looks through.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), names = ['a 7', 'same 7, moved 3 px', 'another 7', 'a 1'];
    PER.forEach((v, i) => { const x = 200 + i * 400; drawImg(img('per' + i, v), x, 250, 210, a0 * eout(prog(t, 0.6 + i * 0.3, 0.6)), { frame: true }); text(names[i], x + 105, 500, { font: F.mono, size: 20, color: i === 0 ? P.chalk : P.stone, align: 'center', a: a0 }); });
    const tbl = (x, title, vals, col, a, note) => { if (a <= 0) return; card(x, 560, 760, 300, col, a); label(title, x + 34, 606, a, col); const mx = Math.max(...vals);
      vals.forEach((v, i) => { const y = 650 + i * 62; text(`vs ${names[i + 1]}`, x + 34, y + 26, { font: F.mono, size: 18, color: P.stone, a });
        setA(a); ctx.fillStyle = rgba(col, 0.85); ctx.fillRect(x + 300, y + 6, 330 * v / mx, 28); setA(1); text(v.toFixed(1), x + 730, y + 28, { font: F.mono, size: 20, color: P.chalk, align: 'right', a }); });
      if (note) text(note, x + 34, 845, { font: F.mono, size: 15, color: P.dust, a }); };
    tbl(160, 'distance in pixels', A.percept.pixel, P.rose, eout(prog(t, 8, 0.8)), 'the moved 7 is farther than the 1 ✗');
    tbl(1000, `distance in features (digit net, ${(A.featAcc * 100).toFixed(1)}% accurate)`, A.percept.feature, P.sage, eout(prog(t, 16, 0.8)), 'the 1 is now far away ✓   the moved 7? still far-ish');
  } });

// 05 — Inception Score
SC.push({ dur: 48, chapter: 'IS', title: 'Judge 2 · Inception Score', sub: 'Confident verdicts, spread across every class.',
  caps: [[1, 8, 'Judge number two: the Inception Score. Feed every sample to the digit network and look at its verdicts.'],
         [8, 15, 'A good sample gets a confident verdict — surely a 7. A mushy sample gets a shrug, spread across several classes.'],
         [15, 22, 'And across all the samples, the verdicts should be spread over every class — not all ones.'],
         [22, 30, 'IS rewards both at once: e to the average KL between each verdict and the overall mix. The best possible score for 10 classes is 10.'],
         [30, 38, `Real digits score ${R.real.IS.toFixed(1)}. The VAE ${R.vae.IS.toFixed(1)} — its blur earns shrugs. The one-trick pony: ${R.ones.IS.toFixed(2)}. Caught!`],
         [38, 47.6, `But the photocopier scores ${R.copier.IS.toFixed(2)}, just like real data. IS never even looks at the real data. Ten perfect images, one per class, would score nearly 10.`]],
  draw(t) {
    const hist = (x, y, w, h, p, col, a, title2, lab = true) => { if (a <= 0) return; const bw = w / 10, mx = Math.max(...p, 0.3);
      for (let j = 0; j < 10; j++) { const bh = p[j] / mx * h; setA(a * 0.85); ctx.fillStyle = col; ctx.fillRect(x + j * bw + 3, y + h - bh, bw - 6, bh); if (lab) text(String(j), x + j * bw + bw / 2, y + h + 22, { font: F.mono, size: 15, color: P.dust, align: 'center', a }); }
      setA(1); line(x, y + h, x + w, y + h, P.borderLight, 1.5, a); text(title2, x, y - 12, { font: F.mono, size: 16, color: P.stone, a }); };
    const A1 = 1 - ease(prog(t, 29.5, 1));
    if (A1 > 0) { GA = A1;
      const a = eout(prog(t, 1, 0.8)), b = eout(prog(t, 8, 0.8));
      drawImg(img('exs', EXS), 160, 290, 150, b, { frame: true }); hist(340, 290, 360, 130, A.examples.sharp, P.ink, b, 'p(y | x): confident ✓');
      drawImg(img('exa', EXA), 160, 500, 150, b, { frame: true }); hist(340, 500, 360, 130, A.examples.amb, P.ink, eout(prog(t, 10, 0.8)), 'p(y | x): a shrug (a VAE sample)');
      const c = eout(prog(t, 15, 0.8));
      hist(820, 290, 440, 130, A.py.real, P.ink, c, 'p(y) over real samples: spread ✓'); hist(820, 500, 440, 130, A.py.ones, P.rose, eout(prog(t, 17, 0.8)), 'p(y) over model C: all ones ✗');
      const f = eout(prog(t, 22, 0.8)); card(1320, 280, 500, 360, P.ink, f); label('inception score', 1352, 326, f, P.ink);
      M('IS = exp( [p|E][p_|x] KL( [p|p]([c|y]|[d|x]) ‖ [p|p]([c|y]) ) )', 1352, 400, 21, { a: f });
      para('big when each verdict is sharp AND the verdicts differ from sample to sample', 1352, 460, 440, { size: 22, a: f });
      text('range: 1 … number of classes (10)', 1352, 610, { font: F.mono, size: 16, color: P.dust, a: f });
      text('computed on 2,000 samples per model', 160, 760, { font: F.mono, size: 16, color: P.dust, a });
      GA = 1; }
    const B = eout(prog(t, 30, 1));
    if (B > 0) { barChart('IS', 260, 300, 1400, { a: B, max: 10, reveal: prog(t, 30, 3), hl: t > 38 ? 'copier' : null, mark: [1, 'max 10'], fmt: v => v.toFixed(2), rowH: 80 });
      stamp('FOOLED', 1360, 478, P.rose, eout(prog(t, 39, 0.6)), -0.04); stamp('CAUGHT', 1360, 558, P.sage, eout(prog(t, 34, 0.6)), 0.04); }
  } });

// 06 — FID
SC.push({ dur: 56, chapter: 'FID', title: 'Judge 3 · Fréchet Inception Distance', sub: 'Compare the cloud of samples with the cloud of real data.',
  caps: [[1, 8, 'Judge number three is the industry standard: FID. Finally, a judge that compares samples against real data.'],
         [8, 15, 'Map real images and generated images into feature space. Each becomes a cloud of points.'],
         [15, 23, 'Summarise each cloud by a bell curve — a centre μ and a spread Σ — then measure how far apart the two bell curves are.'],
         [23, 31, 'The distance between centres, plus a penalty when the shapes differ. That is the Fréchet distance. Lower is better.'],
         [31, 39, `Real vs real: ${R.real.FID.toFixed(1)}. VAE: ${R.vae.FID.toFixed(0)}. One-trick pony: ${R.ones.FID.toFixed(0)}. Static: ${R.noise.FID.toFixed(0)}. Finally a sensible ranking…`],
         [39, 47, `…except the photocopier scores ${R.copier.FID.toFixed(1)}: practically perfect. FID cannot tell memory from imagination either.`],
         [47, 55.6, 'And it squashes everything into one number. Is a score bad because samples look wrong, or because they lack variety? FID cannot say.']],
  draw(t) {
    const m = pmap(120, 240, 860, 690), seq = [['vae', 8, 23], ['ones', 23, 29], ['copier', 39, 56]];
    const cur = seq.find(([, a, b]) => t >= a && t < b) || (t >= 29 && t < 39 ? ['ones', 29, 39] : null);
    clip(m, () => {
      A.pca.ref.slice(0, 400).forEach(p => dot(m.X(p[0]), m.Y(p[1]), 3, P.stone, 0.45 * eout(prog(t, 8, 1))));
      if (cur) A.pca[cur[0]].slice(0, 400).forEach(p => dot(m.X(p[0]), m.Y(p[1]), 3, P.terracotta, 0.7 * eout(prog(t, cur[1], 0.8))));
      if (t > 15) { drawEllipse(m, gaussEllipse(A.pca.ref.slice(0, 400)), P.ink, eout(prog(t, 15, 1)));
        if (cur) drawEllipse(m, gaussEllipse(A.pca[cur[0]].slice(0, 400)), P.plum, eout(prog(t, Math.max(15.5, cur[1]), 1))); }
    });
    text('real', 150, 280, { font: F.mono, size: 18, color: P.stone, a: eout(prog(t, 8, 1)) });
    if (cur) text(NAMES[cur[0]], 150, 310, { font: F.mono, size: 18, color: P.terracotta, a: eout(prog(t, cur[1], 0.8)) });
    text('feature space, drawn in 2-D (FID is computed in all 64 dimensions)', 130, 920, { font: F.mono, size: 14, color: P.dust, a: eout(prog(t, 8, 1)) });
    const fa = eout(prog(t, 15, 0.8));
    card(1040, 250, 780, 260, P.ink, fa); label('fréchet distance', 1074, 296, fa, P.ink);
    M('FID = ‖[a|μ][p_|r] − [a|μ][p_|g]‖²', 1074, 366, 28, { a: fa });
    M('    + Tr( [a|Σ][p_|r] + [a|Σ][p_|g] − 2([a|Σ][p_|r][a|Σ][p_|g])[k^|½] )', 1074, 420, 24, { a: eout(prog(t, 23, 0.8)) });
    text('centres apart          shapes mismatched', 1074, 470, { font: F.mono, size: 16, color: P.dust, a: eout(prog(t, 24, 0.8)) });
    const ba = eout(prog(t, 31, 0.8));
    if (ba > 0) { barChart('FID', 1040, 590, 780, { a: ba, max: 450, log: true, better: 'low', reveal: prog(t, 31, 3), hl: t > 39 ? 'copier' : null, rowH: 58 });
      stamp('FOOLED', 1500, 724, P.rose, eout(prog(t, 40, 0.6)), -0.04); }
  } });

// 07 — precision & recall
SC.push({ dur: 56, chapter: 'precision/recall', title: 'Judge 4 · Precision & recall', sub: 'Split one number into two questions: quality and coverage.',
  caps: [[1, 8, "Judge number four splits FID's single number in two: precision and recall."],
         [8, 16, "Around each real point, draw a ball reaching its third-nearest real neighbour. Together, the balls outline where real data lives."],
         [16, 23, 'Precision: what fraction of samples land inside real territory? That is fidelity.'],
         [23, 30, "Recall: flip it around. What fraction of real points land inside the samples' territory? That is diversity."],
         [30, 38, `The one-trick pony: ${pct(R.ones.precision)} precision, ${pct(R.ones.recall)} recall. Sharp, but covers only the ones. Now we can see exactly what went wrong.`],
         [38, 47, `Our VAE: ${pct(R.vae.precision)} precision — higher than real data! — but recall under 1%. Every sample is a safe, average-looking digit: episode 3's blur, measured.`],
         [47, 55.6, `And the photocopier? ${pct(R.copier.precision)} and ${pct(R.copier.recall)} — identical to real data. Still fooled.`]],
  draw(t) {
    const m = pmap(120, 240, 860, 690);
    const cur = t < 30 ? (t > 16 ? 'vae' : null) : t < 38 ? 'ones' : t < 47 ? 'vae' : 'copier';
    clip(m, () => {
      const ra = eout(prog(t, 8, 1.5));
      REF150.forEach((p, i) => { setA(ra * 0.06); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.arc(m.X(p[0]), m.Y(p[1]), REF_R[i] * m.s, 0, TAU); ctx.fill(); setA(1); });
      REF150.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 3, P.stone, 0.6 * eout(prog(t, 1, 1))));
      if (cur) { const g = A.pca[cur].slice(0, 150), ga = eout(prog(t, cur === 'vae' && t < 30 ? 16 : [30, 38, 47][['ones', 'vae', 'copier'].indexOf(cur)], 0.8));
        if (t > 23) g.forEach((p, i) => { setA(ga * 0.08); ctx.fillStyle = P.terracotta; ctx.beginPath(); ctx.arc(m.X(p[0]), m.Y(p[1]), GEN_R[cur][i] * m.s, 0, TAU); ctx.fill(); setA(1); });
        g.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 3.5, P.terracotta, ga)); }
    });
    text('real data + its territory', 150, 280, { font: F.mono, size: 18, color: P.ink, a: eout(prog(t, 8, 1)) });
    if (cur) text(NAMES[cur], 150, 310, { font: F.mono, size: 18, color: P.terracotta });
    text('drawn in 2-D; measured in all 64 feature dimensions (k = 3)', 130, 920, { font: F.mono, size: 14, color: P.dust, a: eout(prog(t, 8, 1)) });
    const fa = eout(prog(t, 16, 0.8));
    card(1040, 250, 780, 230, P.sage, fa); label('two questions', 1074, 296, fa, P.sage);
    M('precision = share of [d|samples] inside real territory', 1074, 360, 22, { a: fa });
    M('recall    = share of [d|real data] inside sample territory', 1074, 410, 22, { a: eout(prog(t, 23, 0.8)) });
    text('fidelity                    diversity', 1074, 455, { font: F.mono, size: 16, color: P.dust, a: eout(prog(t, 24, 0.8)) });
    const ta = eout(prog(t, 30, 0.8));
    if (ta > 0) { const x = 1040, y = 540; card(x, y, 780, 380, P.plum, ta);
      text('model', x + 34, y + 50, { font: F.mono, size: 16, color: P.dust, a: ta }); text('precision', x + 400, y + 50, { font: F.mono, size: 16, color: P.dust, a: ta }); text('recall', x + 600, y + 50, { font: F.mono, size: 16, color: P.dust, a: ta });
      ORDER.forEach((k, i) => { const yy = y + 100 + i * 56, hl = k === cur;
        text(NAMES[k], x + 34, yy, { font: F.mono, size: 20, color: hl ? P.chalk : P.stone, a: ta });
        text(pct(R[k].precision), x + 400, yy, { font: F.mono, size: 22, color: hl ? P.terracotta : P.stone, a: ta });
        text(R[k].recall < 0.01 ? '<1%' : pct(R[k].recall), x + 600, yy, { font: F.mono, size: 22, color: hl ? P.terracotta : P.stone, a: ta }); });
      stamp('FOOLED', 1700, 790, P.rose, eout(prog(t, 48, 0.6))); }
  } });

// 08 — novelty
SC.push({ dur: 42, chapter: 'novelty', title: 'Judge 5 · Is it new?', sub: 'Catching the photocopier: distance to the nearest training image.',
  caps: [[1, 8, 'Every judge so far crowned the photocopier. To catch it, ask a different question: is this sample new?'],
         [8, 15, 'For each sample, find its nearest neighbour in the training set — all 60,000 images — and measure the distance.'],
         [15, 22, `Real held-out digits sit about ${R.real.NN_train.toFixed(1)} away from anything in training. So do the VAE's. The photocopier: exactly zero.`],
         [22, 30, 'Memorisation is not just a bad score: it is how models leak private photos, copyrighted art, or personal data.'],
         [30, 41.6, "And when test questions leak into a language model's training data, benchmarks become photocopier contests. It's called contamination."]],
  draw(t) {
    const rows = [['real', 'real (held-out)'], ['vae', 'A · VAE'], ['copier', 'B · photocopier']];
    rows.forEach(([k, nm], r) => { const y = 270 + r * 190, a = eout(prog(t, 1 + r * 0.5, 0.8)); text(nm, 150, y + 70, { font: F.mono, size: 20, color: P.chalk, a });
      for (let i = 0; i < 3; i++) { const x = 420 + i * 300, ai = a * eout(prog(t, 8 + r * 1.2 + i * 0.3, 0.6));
        drawImg(img(`ps${k}${i}`, PAIRS[k].s[i]), x, y, 120, a, { frame: true }); drawImg(img(`pn${k}${i}`, PAIRS[k].n[i]), x + 130, y, 120, ai, { frame: true });
        text(`d = ${PAIRS[k].d[i].toFixed(1)}`, x + 125, y + 150, { font: F.mono, size: 18, color: PAIRS[k].d[i] < 0.01 ? P.rose : P.stone, align: 'center', a: ai }); } });
    text('sample   |   nearest training image', 420, 250, { font: F.mono, size: 16, color: P.dust, a: eout(prog(t, 8, 0.8)) });
    const ha = eout(prog(t, 15, 0.8));
    if (ha > 0) { const x = 1330, y = 300, w = 480, h = 380; card(x - 20, y - 50, w + 40, h + 130, P.sage, ha); label('distance to nearest training image', x, y - 10, ha, P.sage);
      const bins = 20, mx = 10; [['real', P.stone], ['vae', P.plum], ['copier', P.rose]].forEach(([k, col], j) => { const cnt = new Array(bins).fill(0); A.nnDist[k].forEach(d => cnt[Math.min(bins - 1, Math.floor(d / mx * bins))]++); const cm = 300;
        cnt.forEach((c, b) => { const bh = Math.min(h, c / cm * h); setA(ha * 0.7); ctx.fillStyle = col; ctx.fillRect(x + b * w / bins + j * 7, y + h - bh, 7, bh); }); setA(1); });
      line(x, y + h, x + w, y + h, P.borderLight, 1.5, ha); text('0', x, y + h + 24, { font: F.mono, size: 15, color: P.dust, a: ha }); text('10', x + w, y + h + 24, { font: F.mono, size: 15, color: P.dust, align: 'right', a: ha });
      [['real', P.stone], ['VAE', P.plum], ['copier: all at 0', P.rose]].forEach(([s, c], i) => text(s, x + 160, y + 40 + i * 30, { font: F.mono, size: 17, color: c, a: ha })); }
    stamp('CAUGHT', 1570, 820, P.sage, eout(prog(t, 17, 0.6)));
    const ca = eout(prog(t, 30, 0.8)); if (ca > 0) { rbox(420, 850 - 10, 1390, 70, 14, rgba(P.canvas, 0.95), P.rose, ca, 1.5); text('test data inside training data = contamination: the benchmark becomes a memory test', 1115, 885, { font: F.mono, size: 20, color: P.rose, align: 'center', a: ca }); }
  } });

// 09 — the learned judge
SC.push({ dur: 48, chapter: 'the judge', title: 'Judge 6 · A judge that learns', sub: 'Humans are the gold standard. Can we automate them?',
  caps: [[1, 8, 'The gold standard is still people: show humans real and generated images and see if they can tell the difference.'],
         [8, 15, 'But humans are slow and expensive, and from a handful of samples nobody notices a model that is missing half the data.'],
         [15, 23, 'So automate the human: train a classifier to tell real from generated, then test it on images it has never seen.'],
         [23, 30, 'Stuck at 50%? The samples are indistinguishable from the real thing. The further above 50%, the easier the fakes are to spot.'],
         [30, 38, `Real: ${pct(R.real.C2ST)}. VAE: ${pct(R.vae.C2ST)} — even a simple linear judge spots the blur. The one-trick pony: ${pct(R.ones.C2ST)}. Static: ${pct(R.noise.C2ST)}.`],
         [38, 47.6, 'Hold on to this idea — a judge that learns to spot fakes. It is about to become much more than a metric.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), y = 420;
    const A1 = 1 - ease(prog(t, 29.5, 1));
    if (A1 > 0) { GA = A1;
      for (let i = 0; i < 4; i++) { drawImg(img(`jr${i}`, S.real[i]), 160 + (i % 2) * 110, 280 + Math.floor(i / 2) * 110, 100, a0); drawImg(img(`jv${i}`, S.vae[i]), 160 + (i % 2) * 110, 560 + Math.floor(i / 2) * 110, 100, a0); }
      text('real', 265, 270, { font: F.mono, size: 18, color: P.stone, align: 'center', a: a0 }); text('generated', 265, 550, { font: F.mono, size: 18, color: P.terracotta, align: 'center', a: a0 });
      const ja = eout(prog(t, 15, 0.8));
      arrow(400, 390, 600, 500, P.dust, 2, ja); arrow(400, 670, 600, 560, P.dust, 2, ja);
      rbox(620, 430, 300, 200, 20, rgba(P.plum, 0.12), rgba(P.plum, 0.75), ja, 2.5); text('judge', 770, 520, { font: F.serif, size: 40, color: P.chalk, align: 'center', a: ja }); text('real or fake?', 770, 570, { font: F.mono, size: 18, color: P.plum, align: 'center', a: ja });
      arrow(930, 530, 1040, 530, P.dust, 2, ja); text('held-out accuracy', 1060, 520, { font: F.mono, size: 20, color: P.sage, a: ja });
      const ha = eout(prog(t, 1, 0.8)) * (1 - ease(prog(t, 14, 0.8)));
      if (ha > 0) { card(1100, 260, 700, 260, P.sage, ha); label('humans', 1134, 306, ha, P.sage); para('gold standard — but slow, costly, noisy, and blind to missing variety in a few samples', 1134, 360, 630, { size: 25, a: ha }); }
      const sa = eout(prog(t, 23, 0.8));
      if (sa > 0) { card(1060, 620, 760, 220, P.sage, sa); label('the scale', 1094, 666, sa, P.sage); M('50% = indistinguishable      100% = obvious fake', 1094, 730, 24, { a: sa }); text('(a classifier two-sample test)', 1094, 790, { font: F.mono, size: 16, color: P.dust, a: sa }); }
      GA = 1; }
    const B = eout(prog(t, 30, 1));
    if (B > 0) { barChart('C2ST', 260, 330, 1400, { a: B, max: 1, reveal: prog(t, 30, 3), better: 'low', fmt: pct, hl: 'vae', mark: [0.5, 'indistinguishable (50%)'] });
      text('closer to 50% is better', 960, 670, { font: F.serif, italic: true, size: 30, color: P.dust, align: 'center', a: B });
      text('a judge that learns to spot fakes…', 960, 800, { font: F.serif, italic: true, size: 44, color: P.chalk, align: 'center', a: eout(prog(t, 38.5, 1)) }); }
  } });

// 10 — the same judges for language models
SC.push({ dur: 40, chapter: 'LLMs', title: 'The same judges, for language models', sub: 'Different names, same ideas.',
  caps: [[1, 8, 'The same judges grade today’s language models, under different names.'],
         [8, 15, 'Likelihood becomes perplexity: e to the average surprise. Our episode-2 name model scores 3.4 — as if choosing among 3.4 letters each step.'],
         [15, 22, 'Answer-key benchmarks — maths, code checked by unit tests — work when there is one right answer. Beware contamination.'],
         [22, 30, 'The learned judge becomes “LLM-as-a-judge”: fast and cheap, but biased toward long, confident-sounding answers.'],
         [30, 39.6, 'And human judgment becomes preference arenas: people vote between two answers, and the votes become Elo ratings, like chess.']],
  draw(t) {
    const cards = [['likelihood', 'perplexity = exp( −(1/T) Σ log p(xₜ | x<ₜ) )', 'our name model: e^1.21 ≈ 3.4', 1],
                   ['answer keys', 'benchmarks: exact match, unit tests', 'catch: contamination (the copier)', 15],
                   ['learned judge', 'LLM-as-a-judge', 'catch: prefers long, confident answers', 22],
                   ['humans', 'pairwise preference → Elo ratings', 'catch: slow, and style beats substance', 30]];
    cards.forEach(([nm, what, note, t0], i) => { const x = 150 + (i % 2) * 830, y = 260 + Math.floor(i / 2) * 330, a = eout(prog(t, t0, 0.8));
      card(x, y, 780, 290, P.ink, a); text(nm, x + 36, y + 70, { font: F.serif, size: 44, color: P.chalk, a }); M(what, x + 36, y + 150, 24, { a }); text(note, x + 36, y + 230, { font: F.mono, size: 20, color: i === 0 ? P.sage : P.rose, a }); });
  } });

// 11 — the scorecard
SC.push({ dur: 38, chapter: 'scorecard', title: 'The full scorecard', sub: 'Every number computed on real models, on 2,000 samples each.',
  caps: [[1, 8, 'Here is the full scorecard. Every number was computed on real models.'],
         [8, 16, 'Count the columns where the photocopier ties real data. Only novelty — and likelihood — catch it.'],
         [16, 24, 'No single number is the truth. Each judge answers one question; you need the whole panel — and your own eyes.'],
         [24, 37.6, "And remember Goodhart's law: when a measure becomes a target, it ceases to be a good measure. Optimise any one judge hard enough, and it will be fooled."]],
  draw(t) {
    playPill('ep04-metric-lab.html', 16, t);
    const cols = [['IS ↑', 'IS', v => v.toFixed(2)], ['FID ↓', 'FID', v => v.toFixed(1)], ['precision ↑', 'precision', pct], ['recall ↑', 'recall', v => v < 0.01 ? '<1%' : pct(v)], ['NN-train', 'NN_train', v => v.toFixed(2)], ['C2ST → 50%', 'C2ST', pct], ['likelihood', null, null]];
    const x0 = 130, y0 = 280, cw = 225, rh = 90, a0 = eout(prog(t, 0.6, 0.8));
    card(x0 - 20, y0 - 60, 1700, 560, P.plum, a0);
    cols.forEach(([h], j) => text(h, x0 + 290 + j * 200, y0, { font: F.mono, size: 17, color: P.dust, a: a0 }));
    ORDER.forEach((k, i) => { const y = y0 + 60 + i * rh, a = eout(prog(t, 1 + i * 0.3, 0.6));
      text(NAMES[k], x0, y, { font: F.mono, size: 21, color: k === 'copier' ? P.chalk : P.stone, a });
      cols.forEach(([, key, f], j) => { const x = x0 + 290 + j * 200;
        const v = key ? f(R[k][key]) : (k === 'vae' ? `${ELBO.toFixed(0)} nats` : k === 'real' ? '—' : k === 'noise' ? 'poor' : '−∞');
        const fooled = k === 'copier' && key && key !== 'NN_train', caught = k === 'copier' && (key === 'NN_train' || !key);
        if (t > 8 && fooled) rbox(x - 12, y - 32, 180, 48, 10, null, P.rose, eout(prog(t, 8.5 + j * 0.2, 0.5)), 2);
        if (t > 8 && caught) rbox(x - 12, y - 32, 180, 48, 10, rgba(P.sage, 0.1), P.sage, eout(prog(t, 10, 0.5)), 2);
        text(v, x, y, { font: F.mono, size: 22, color: P.chalk, a }); }); });
    text('real-data "likelihood" is the thing we are estimating — it has no score of its own', x0, y0 + 510, { font: F.mono, size: 14, color: P.dust, a: a0 });
    const g = eout(prog(t, 24, 1));
    if (g > 0) { rbox(260, 820, 1400, 100, 18, rgba(P.canvas, 0.95), P.sage, g, 1.5);
      text('“When a measure becomes a target, it ceases to be a good measure.” — Goodhart', 960, 882, { font: F.serif, italic: true, size: 34, color: P.chalk, align: 'center', a: g }); }
  } });

// 12 — hook → GANs
SC.push({ dur: 46, chapter: 'next', title: 'What if the judge could teach?', sub: 'A metric that becomes a teacher.',
  caps: [[1, 8, `Remember the learned judge? It spotted the VAE's blur ${pct(R.vae.C2ST)} of the time.`],
         [8, 15, 'Here is a wild idea: what if the judge did not just grade the generator… but taught it?'],
         [15, 23, 'Every time the judge catches a fake, its gradient tells the generator exactly which way to change to look more real.'],
         [23, 31, 'Then retrain the judge on the better fakes. Forger and detective, locked in an arms race — each making the other better.'],
         [31, 37, 'The game ends when the judge is stuck at 50%: the perfect score from our two-sample test.'],
         [37, 45.6, 'And our one-trick pony? Its disease has a name — mode collapse — and it will haunt us. Next episode: GANs.']],
  draw(t) {
    const y = 520, a0 = eout(prog(t, 0.6, 0.8)), dim = 1 - 0.92 * ease(prog(t, 37.5, 1));
    GA = dim;
    rbox(150, y - 90, 260, 180, 20, rgba(P.plum, 0.12), rgba(P.plum, 0.75), a0, 2.5); text('generator', 280, y + 12, { font: F.serif, size: 38, color: P.chalk, align: 'center', a: a0 }); M('[m|G]([n|z])', 280, y + 60, 22, { align: 'center', a: a0 });
    const q = ease(prog(t, 23, 12)); // fakes improve as the arms race goes on
    for (let i = 0; i < 4; i++) { const x = 480 + (i % 2) * 110, yy = y - 110 + Math.floor(i / 2) * 110; drawImg(img(`hn${i}`, NOISE[i], P.rose), x, yy, 100, a0 * (1 - q)); drawImg(img(`hv${i}`, S.vae[4 + i]), x, yy, 100, a0 * q * (1 - ease(prog(t, 31, 2))), { bg: false }); drawImg(img(`hr${i}`, S.real[4 + i]), x, yy, 100, a0 * ease(prog(t, 31, 2)), { bg: false }); }
    arrow(420, y, 470, y, P.dust, 2, a0); arrow(710, y, 800, y, P.dust, 2, a0);
    const ja = eout(prog(t, 8, 0.8));
    rbox(820, y - 90, 280, 180, 20, rgba(P.plum, 0.12), rgba(P.plum, 0.75), ja, 2.5); text('judge', 960, y + 12, { font: F.serif, size: 38, color: P.chalk, align: 'center', a: ja }); M('[m|D]([d|x])', 960, y + 60, 22, { align: 'center', a: ja });
    arrow(1110, y, 1190, y, P.dust, 2, ja);
    const acc = t < 23 ? 1 : lerp(1, 0.5, ease(prog(t, 23, 12)));
    text(`“fake” with ${Math.round(acc * 100)}% accuracy`, 1210, y + 8, { font: F.mono, size: 24, color: acc > 0.55 ? P.rose : P.sage, a: ja });
    const ga = eout(prog(t, 15, 0.8));
    if (ga > 0) { const flow = (t * 0.8) % 1; setA(ga); ctx.strokeStyle = P.sage; ctx.lineWidth = 3; ctx.setLineDash([10, 10]); ctx.lineDashOffset = flow * 20; ctx.beginPath(); ctx.moveTo(960, y + 110); ctx.bezierCurveTo(960, y + 250, 280, y + 250, 280, y + 110); ctx.stroke(); ctx.setLineDash([]); setA(1);
      text('gradient: “change this way to look more real”', 620, y + 250, { font: F.mono, size: 20, color: P.sage, align: 'center', a: ga }); }
    const ra = eout(prog(t, 23, 0.8)); if (ra > 0) { const x = 1210, w = 520; rbox(x, y + 60, w, 16, 8, P.surface, P.border, ra); setA(ra); ctx.fillStyle = acc > 0.55 ? P.rose : P.sage; ctx.fillRect(x, y + 60, w * acc, 16); setA(1); line(x + w * 0.5, y + 50, x + w * 0.5, y + 86, P.sage, 2, ra); text('50%', x + w * 0.5, y + 110, { font: F.mono, size: 16, color: P.sage, align: 'center', a: ra }); }
    const fa = eout(prog(t, 31, 0.8));
    M('min[m_|G] max[m_|D]  [p|E][p_|x∼data][ log [m|D]([d|x]) ] + [p|E][p_|z][ log(1 − [m|D]([m|G]([n|z]))) ]', 960, 300, 26, { align: 'center', a: fa });
    GA = 1;
    const qa = eout(prog(t, 37.5, 1.2));
    if (qa > 0) {
      text('A forger and a detective,', 960, 470, { font: F.serif, size: 72, color: P.chalk, align: 'center', a: qa });
      text('teaching each other.', 960, 555, { font: F.serif, size: 72, color: P.chalk, align: 'center', a: qa });
      text('NEXT  ·  EPISODE 05', 960, 660, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 39, 1)), ls: 4 });
      seasonStrip(4, 40.2, t);
      text('GANs — learning by being judged', 960, 715, { font: F.serif, italic: true, size: 42, color: P.stone, align: 'center', a: eout(prog(t, 39.5, 1)) });
    }
  } });
