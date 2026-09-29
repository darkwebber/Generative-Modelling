// ─────────────────────────────────────────────────────────────
//  Real diffusion models (exported by code/export_ep09_assets.py)
// ─────────────────────────────────────────────────────────────
const A = window.EP9, WR = 2.1;
function b64bytes(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
const MEMO = new Map(); const memo = (k, f) => { if (!MEMO.has(k)) MEMO.set(k, f()); return MEMO.get(k); };
function square(x0, y0, S, R = WR) { return { X: v => x0 + (v + R) / (2 * R) * S, Y: v => y0 + (R - v) / (2 * R) * S, s: S / (2 * R), x0, y0, S }; }
function frame(m, a = 1) { setA(a); ctx.fillStyle = P.surface; ctx.fillRect(m.x0, m.y0, m.S, m.S); ctx.strokeStyle = P.border; ctx.lineWidth = 1.5; ctx.strokeRect(m.x0, m.y0, m.S, m.S); setA(1); }
function clipTo(m, fn) { ctx.save(); ctx.beginPath(); ctx.rect(m.x0, m.y0, m.S, m.S); ctx.clip(); fn(); ctx.restore(); }
function marble(x, y, r, col, a = 1) { if (GA * a <= 0.002) return; dot(x, y, r, col, a); dot(x - r * 0.32, y - r * 0.36, r * 0.36, P.chalk, a * 0.5); }
const abar = t => t < 0 ? 1 : A.abar[Math.min(A.abar.length - 1, Math.round(t / 10))];
function gaussian(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(TAU * r()); }
const lerpPts = (S, f) => { const n = S.length - 1, fi = clamp(f, 0, n), i = Math.min(n - 1, Math.floor(fi)), u = fi - i; return S[i].map((p, k) => [lerp(p[0], S[i + 1][k][0], u), lerp(p[1], S[i + 1][k][1], u)]); };

// the cold open: 5000 particles × 62 stages, int16
const TITLE = (() => { const T = A.title, u = b64bytes(T.q), v = new Int16Array(u.buffer); return { n: T.n, S: T.stages, v, ts: T.ts }; })();
function titlePt(k, f) { const S = TITLE.S - 1, fi = clamp(f, 0, S), i = Math.min(S - 1, Math.floor(fi)), u = fi - i, n = TITLE.n, a = (i * n + k) * 2, b = ((i + 1) * n + k) * 2, s = A.titleScale / 4000;
  return [lerp(TITLE.v[a], TITLE.v[b], u) * s, lerp(TITLE.v[a + 1], TITLE.v[b + 1], u) * s]; }

// digits: uint8 images in [-1, 1] → 0…255
const imgs = (b, n) => memo('i' + b, () => { const u = b64bytes(b); return Array.from({ length: n }, (_, i) => u.subarray(i * 784, (i + 1) * 784)); });
const COLD = hx(P.terracotta);
function digitCanvas(px, key) { return memo('c' + key, () => { const c = document.createElement('canvas'); c.width = c.height = 28; const g = c.getContext('2d'), im = g.createImageData(28, 28);
  for (let i = 0; i < 784; i++) { const v = px[i] / 255; im.data[i * 4] = COLD[0] * v + 28 * (1 - v); im.data[i * 4 + 1] = COLD[1] * v + 24 * (1 - v); im.data[i * 4 + 2] = COLD[2] * v + 22 * (1 - v); im.data[i * 4 + 3] = 255; } g.putImageData(im, 0, 0); return c; }); }
function drawDigit(px, key, x, y, s, a = 1) { if (GA * a <= 0.002) return; setA(a); ctx.imageSmoothingEnabled = false; ctx.drawImage(digitCanvas(px, key), x, y, s, s); ctx.imageSmoothingEnabled = true; setA(1); }
const DS = A.digits, NDS = DS.x.length - 1;
const digitAt = (stage, i) => imgs(DS.x[stage], 64)[i];
// the forward process on one digit, computed here: xₜ = √ᾱ·x₀ + √(1−ᾱ)·ε
const FWD_EPS = (() => { const g = gaussian(rng(19)); return Float32Array.from({ length: 784 }, g); })();
function noisyDigit(t) { const x0 = digitAt(NDS, 0), ab = abar(t), out = new Uint8Array(784); for (let i = 0; i < 784; i++) { const v = Math.sqrt(ab) * (x0[i] / 127.5 - 1) + Math.sqrt(1 - ab) * FWD_EPS[i]; out[i] = clamp((v + 1) / 2) * 255; } return out; }
// the forward process on the 8 blobs, computed here
const RING0 = A.ring.x[A.ring.x.length - 1], RING_EPS = (() => { const g = gaussian(rng(23)); return RING0.map(() => [g(), g()]); })();
const ringFwd = t => { const ab = abar(t), a = Math.sqrt(ab), b = Math.sqrt(1 - ab); return RING0.map((p, k) => [a * p[0] + b * RING_EPS[k][0], a * p[1] + b * RING_EPS[k][1]]); };
// 1-D reverse-step illustration
const N1 = (x, m, v) => Math.exp(-0.5 * (x - m) ** 2 / v) / Math.sqrt(TAU * v);
const P1 = y => 0.5 * N1(y, -1.107, 0.2265) + 0.5 * N1(y, 1.107, 0.2265);
function postr(x, b) { const g = [], n = 240; let s = 0; for (let i = 0; i <= n; i++) { const y = -3 + 6 * i / n, v = N1(x, Math.sqrt(1 - b) * y, b) * P1(y); g.push(v); s += v; } return g.map(v => v / (s * 6 / n)); }
const pct = v => (v * 100).toFixed(0) + '%';
// full-frame TV static, re-rolled 24 times a second
const STATIC = (() => { const c = document.createElement('canvas'); c.width = 320; c.height = 180; return { c, g: c.getContext('2d'), im: c.getContext('2d').createImageData(320, 180), k: -1 }; })();
function tvStatic(t, a) { if (GA * a <= 0.003) return; const k = Math.floor(t * 24); if (k !== STATIC.k) { STATIC.k = k; const r = rng(k * 7 + 1), d = STATIC.im.data;
    for (let i = 0; i < d.length; i += 4) { const v = r() * 255; d[i] = v; d[i + 1] = v * 0.93; d[i + 2] = v * 0.88; d[i + 3] = 255; } STATIC.g.putImageData(STATIC.im, 0, 0); }
  setA(a); ctx.imageSmoothingEnabled = false; ctx.drawImage(STATIC.c, 0, 0, 1920, 1080); ctx.imageSmoothingEnabled = true; setA(1); }
function shake(t, t0, amp = 10) { if (t < t0 || t > t0 + 0.8) return [0, 0]; const u = (t - t0) / 0.8, d = amp * Math.exp(-5 * u); return [d * Math.sin(u * 60), d * Math.cos(u * 47)]; }

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open: the title writes itself out of static
const BANG = 21;
SC.push({ dur: 36, chapter: 'intro',
  caps: [[1, 7, 'Five thousand points of pure static. No shape. No meaning.'],
         [7, 13, 'A small network looks at them and asks one question, again and again: which part of this is noise?'],
         [13, 19.5, 'It removes a tiny bit, then asks again. A thousand times.'],
         [27.5, 35.6, 'Everything you just watched, the title included, was drawn by a real diffusion model. Today: how noise becomes a picture.']],
  draw(t) {
    const [sx, sy] = shake(t, BANG, 14); ctx.save(); ctx.translate(sx, sy);
    const f = (TITLE.S - 1) * (t < 9 ? 0 : Math.pow(ease(prog(t, 9, 12)), 0.9)), jit = t < 9 ? 1 : 1 - prog(t, 9, 3), r = rng(Math.floor(t * 24) + 3);
    const cx = 960, cy = 530, sc = 300, done = clamp((f - (TITLE.S - 8)) / 7), hot = t >= BANG ? Math.exp(-(t - BANG) * 1.2) : 0;
    const col = mixc(P.rose, P.chalk, done), a0 = eout(prog(t, 0.2, 1.2)), tNow = TITLE.ts[Math.min(TITLE.S - 1, Math.round(f))], nz = Math.sqrt(1 - abar(tNow));
    tvStatic(t, a0 * 0.10 * nz ** 3 * (t < BANG ? 1 : 0));
    ctx.fillStyle = col;
    for (let k = 0; k < TITLE.n; k++) { const [x, y] = titlePt(k, f), jx = jit * (r() - 0.5) * 0.02, jy = jit * (r() - 0.5) * 0.02, X = cx + (x + jx) * sc, Y = cy - (y + jy) * sc;
      if (X < -10 || X > 1930 || Y < -10 || Y > 1090) continue; setA(a0 * (0.8 + 0.2 * done)); ctx.fillRect(X - 1.5, Y - 1.5, 3, 3); }
    setA(1);
    if (hot > 0.01) { ctx.save(); ctx.globalCompositeOperation = 'lighter';          // glow: soft additive halos, no shadowBlur (fast)
      for (const [r, al] of [[9, 0.05], [5, 0.12], [2.5, 0.35]]) { ctx.fillStyle = rgba(P.terracotta, al * hot);
        for (let k = 0; k < TITLE.n; k += 2) { const [x, y] = titlePt(k, f); ctx.fillRect(cx + x * sc - r, cy - y * sc - r, 2 * r, 2 * r); } } ctx.restore();
      ring(cx, cy, 200 + (t - BANG) * 1400, P.chalk, hot * 0.35, 3); }
    ctx.restore();
    const a = eout(prog(t, BANG + 0.6, 1.2));
    text('EPISODE 09', 960, 330, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('noise, run backwards', 960, 760, { size: 36, color: P.stone, align: 'center', a: eout(prog(t, BANG + 1.4, 1.2)) });
    const tg = [[P.terracotta, 'x₀  data'], [P.rose, 'ε  noise'], [P.amber, 't  step · ᾱ signal left'], [P.plum, 'εθ  the network'], [P.sage, 'x̂₀  its guess']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let tx = 960 - tw / 2; tg.forEach(([c, s], i) => { tx += tag(tx, 830, c, s, eout(prog(t, 23.5 + i * 0.3, 0.6)), 21) + 14; });
  } });

// 01 — destroy slowly: the forward process
const tF1 = t => 999 * ease(prog(t, 2, 20));
SC.push({ dur: 56, chapter: 'forward', title: 'Destroy it slowly', sub: 'The forward process: a thousand pinches of noise.',
  caps: [[1, 8, 'First, the easy direction: destroying data. Take the eight blobs and add a pinch of Gaussian noise. Then another. A thousand times.'],
         [8, 16, 'Each step shrinks every point a tiny bit toward the centre and adds a small kick: xₜ = √(1 − βₜ)·xₜ₋₁ + √βₜ·ε.'],
         [16, 24, 'Why shrink? Episode 8’s balance again: shrinking removes variance, the kick adds it back, and the cloud settles at exactly N(0, I). Static with a known shape.'],
         [24, 32, 'Gaussians add up: a thousand small kicks make one big kick. So we can jump straight to any step: xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε.'],
         [32, 40, 'ᾱₜ is how much of the original signal survives. It starts at 1 and slides down to almost 0.'],
         [40, 48, 'The same on a digit: a little noise, then more, and by step 1000 nothing of it is left.'],
         [48, 55.6, 'No learning so far. This direction is fixed and free. All the magic is in going back.']],
  draw(t) {
    const m = square(110, 250, 560), a0 = eout(prog(t, 0.6, 0.8)), tt = tF1(t); frame(m, a0);
    clipTo(m, () => ringFwd(tt).forEach(p => marble(m.X(p[0]), m.Y(p[1]), 2.6, mixc(P.terracotta, P.rose, clamp(1 - abar(tt))), a0)));
    text(`step t = ${Math.round(tt)}   ·   ᾱₜ = ${abar(tt).toFixed(3)}`, 110, 850, { font: F.mono, size: 20, color: P.chalk, a: a0 });
    // ᾱ curve
    const ca = eout(prog(t, 32, 0.8)); if (ca > 0) { const x0 = 110, y0 = 930, w = 560, h = 50; line(x0, y0, x0 + w, y0, P.border, 1, ca);
      setA(ca); ctx.strokeStyle = P.amber; ctx.lineWidth = 2.5; ctx.beginPath(); A.abar.forEach((v, i) => { const X = x0 + i / (A.abar.length - 1) * w, Y = y0 - v * h; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
      dot(x0 + tt / 999 * w, y0 - abar(tt) * h, 5, P.amber, ca); text('ᾱₜ: signal left', x0 + w, y0 - h + 4, { font: F.mono, size: 14, color: P.amber, align: 'right', a: ca }); }
    const c1 = eout(prog(t, 8, 0.8)); card(730, 250, 1100, 150, P.rose, c1); label('one step', 764, 296, c1, P.rose);
    M('[d|x][d_|t] = √(1 − [a|β][a_|t])·[d|x][d_|t−1] + √[a|β][a_|t]·[n|ε]        [a|β][a_|t]: 0.0001 → 0.02', 764, 356, 26, { a: c1 });
    const c2 = eout(prog(t, 16, 0.8)); card(730, 420, 1100, 150, P.sage, c2); label('why shrink', 764, 466, c2, P.sage);
    M('Var[t_|t] = (1 − [a|β])·Var[t_|t−1] + [a|β]   →   1          (episode 8’s balance)', 764, 526, 26, { a: c2 });
    const c3 = eout(prog(t, 24, 0.8)); card(730, 590, 1100, 170, P.amber, c3); label('jump to any step', 764, 636, c3, P.amber);
    M('[d|x][d_|t] = √[a|ᾱ][a_|t]·[d|x₀] + √(1 − [a|ᾱ][a_|t])·[n|ε]        [a|ᾱ][a_|t] = Π(1 − [a|β][a_|s])', 764, 700, 28, { a: c3 });
    const da = eout(prog(t, 40, 0.8)); if (da > 0) { card(730, 780, 1100, 150, P.terracotta, da);
      [0, 100, 250, 400, 600, 999].forEach((ts, i) => { const x = 770 + i * 172; drawDigit(noisyDigit(ts), 'f' + ts, x, 796, 100, da * eout(prog(t, 40.3 + i * 0.3, 0.4))); text(`t = ${ts}`, x + 50, 916, { font: F.mono, size: 13, color: P.dust, align: 'center', a: da }); }); }
  } });

// 02 — undo one small step
SC.push({ dur: 52, chapter: 'reverse', title: 'Undo one small step', sub: 'Tiny steps are easy to reverse. Giant ones aren’t.',
  caps: [[1, 8, 'Going back: given a noisy point xₜ, where was it one step earlier? Look at one step in one dimension.'],
         [8, 16, 'Because the step is tiny, the answer is a narrow bell curve, nudged slightly toward where the data is. Easy to describe.'],
         [16, 24, 'Take one giant step instead and the answer splits into two possibilities. No single bell curve fits. That is why diffusion takes many small steps.'],
         [24, 32, 'And the nudge? It points toward more likely x: it is the score from episode 8.'],
         [32, 40, 'So every reverse step is a small Langevin-style move: follow the arrows a little, then add a little fresh noise.'],
         [40, 51.6, 'All we need is the score at every noise level. And we already know how to learn that: predict the noise.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), xt = 0.35 + 0.25 * Math.sin(t * 0.25);
    [[8, 0.02, 'one tiny step  (β = 0.02)', P.sage], [16, 0.6, 'one giant step  (β = 0.6)', P.rose]].forEach(([t0, b, nm, col], r) => {
      const a = a0 * eout(prog(t, r ? t0 : 1, 0.8)), X0 = 130 + r * 870, X1 = X0 + 760, X = y => X0 + (y + 3) / 6 * (X1 - X0), base = 700, Y = v => base - v * 240;
      if (a <= 0) return;
      line(X0, base, X1, base, P.borderLight, 1.5, a); setA(a * 0.15); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.moveTo(X(-3), base); for (let i = 0; i <= 240; i++) { const y = -3 + 6 * i / 240; ctx.lineTo(X(y), Y(P1(y))); } ctx.lineTo(X(3), base); ctx.fill(); setA(1);
      setA(a); ctx.strokeStyle = P.ink; ctx.lineWidth = 2.5; ctx.beginPath(); for (let i = 0; i <= 240; i++) { const y = -3 + 6 * i / 240; i ? ctx.lineTo(X(y), Y(P1(y))) : ctx.moveTo(X(y), Y(P1(y))); } ctx.stroke(); setA(1);
      const po = postr(xt, b), pm = Math.max(...po), sc = Math.min(1, 1.6 / pm);
      setA(a); ctx.strokeStyle = col; ctx.lineWidth = 3.5; ctx.beginPath(); po.forEach((v, i) => { const y = -3 + 6 * i / 240; i ? ctx.lineTo(X(y), Y(v * sc * 0.55)) : ctx.moveTo(X(y), Y(v * sc * 0.55)); }); ctx.stroke(); setA(1);
      line(X(xt), base + 10, X(xt), 380, P.amber, 2, a, [4, 5]); marble(X(xt), base + 18, 8, P.amber, a); text('xₜ', X(xt), base + 52, { font: F.mono, size: 18, color: P.amber, align: 'center', a });
      text(nm, X0, 360, { font: F.mono, size: 20, color: col, a }); text('ink: data one step earlier · colour: where xₜ came from', X0, 750 + 40, { font: F.mono, size: 14, color: P.dust, a });
      if (t > (r ? 18 : 10)) text(r ? '✗ two separate answers' : '✓ one narrow bell', X1, 360, { font: F.mono, size: 18, color: col, align: 'right', a: a * eout(prog(t, r ? 18 : 10, 0.6)) });
    });
    const c1 = eout(prog(t, 24, 0.8)); card(130, 840, 1700, 100, P.sage, c1);
    M('[d|x][d_|t−1] ≈ ( [d|x][d_|t] + [a|β][a_|t]·[s|s]([d|x][d_|t]) ) / √(1 − [a|β][a_|t]) + √[a|β][a_|t]·[n|z]      follow the arrows a little, add a little noise', 164, 902, 24, { a: c1 });
  } });

// 03 — training: one regression, a thousand noise levels
const TRAIN_T = [612, 244, 874, 403, 96, 733];
SC.push({ dur: 54, chapter: 'training', title: 'One regression, a thousand noise levels', sub: 'Predict the noise. That’s the whole training loop.',
  caps: [[1, 8, 'Training is almost embarrassingly simple. Take a real image x₀. Pick a random step t, which means a random amount of noise.'],
         [8, 16, 'Draw fresh noise ε and jump straight to xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε. One line, no loop.'],
         [16, 24, 'Show the network xₜ and t, and ask it to predict ε: the noise that was added. The loss is ‖ε − εθ(xₜ, t)‖².'],
         [24, 32, 'That is episode 8’s denoising score matching, with one network for all thousand noise levels. The predicted noise is the score, rescaled: εθ = −√(1 − ᾱₜ)·sθ.'],
         [32, 40, 'Knowing the noise means knowing the clean image: x̂₀ = (xₜ − √(1 − ᾱₜ)·εθ)/√ᾱₜ. A denoiser, exactly as Tweedie promised.'],
         [40, 47, 'The network here: two hidden layers of 1,024, with t fed into each, trained 30,000 steps on MNIST. It outputs its guess x̂₀ and reads the noise off it: the same thing, rearranged.'],
         [47, 53.6, 'Ho, Jain and Abbeel, 2020: “Denoising Diffusion Probabilistic Models”. This one loss is why they work.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), cyc = Math.floor(clamp(t - 1, 0, 60) / 7) % TRAIN_T.length, tt = TRAIN_T[cyc], u = ((t - 1) % 7) / 7;
    const x0 = digitAt(NDS, cyc % 8), ab = abar(tt), eps = FWD_EPS;
    const xt = memo('xt' + cyc, () => { const o = new Uint8Array(784); for (let i = 0; i < 784; i++) o[i] = clamp(((Math.sqrt(ab) * (x0[i] / 127.5 - 1) + Math.sqrt(1 - ab) * eps[i]) + 1) / 2) * 255; return o; });
    const epsImg = memo('eps', () => Uint8Array.from(eps, v => clamp((v / 2.5 + 1) / 2) * 255));
    const cols = [[x0, 'x₀  a real image', P.terracotta, 1], [epsImg, 'ε  fresh noise', P.rose, 8], [xt, `xₜ  (t = ${tt})`, P.amber, 8], [epsImg, 'εθ(xₜ, t)  predicted noise', P.plum, 16]];
    cols.forEach(([img, nm, col, t0], i) => { const x = 130 + i * 300, a = a0 * eout(prog(t, t0, 0.6)); drawDigit(img, `tr${i}${cyc}`, x, 290, 220, a); text(nm, x, 550, { font: F.mono, size: 16, color: col, a }); });
    const ops = [['+', 8], ['→', 8], ['≈', 16]]; ops.forEach(([s, t0], i) => text(s, 380 + i * 300 + 1, 410, { font: F.mono, size: 40, color: P.dust, align: 'center', a: a0 * eout(prog(t, t0, 0.6)) }));
    const da = eout(prog(t, 3, 0.6)); if (da > 0) { const X = 1360, Y = 400, R = 90; ring(X, Y, R, P.border, da, 3); const ang = -Math.PI / 2 + TAU * (tt / 1000) * ease(clamp(u * 3)); line(X, Y, X + R * Math.cos(ang), Y + R * Math.sin(ang), P.amber, 4, da); dot(X, Y, 6, P.amber, da);
      text(`t = ${tt}`, X, Y + R + 40, { font: F.mono, size: 22, color: P.amber, align: 'center', a: da }); text(`ᾱ = ${ab.toFixed(2)}`, X, Y + R + 70, { font: F.mono, size: 16, color: P.dust, align: 'center', a: da }); }
    const c1 = eout(prog(t, 16, 0.8)); card(130, 600, 1220, 110, P.plum, c1); M('loss = ‖ [n|ε] − [m|ε][m_|θ]( √[a|ᾱ][a_|t]·[d|x₀] + √(1 − [a|ᾱ][a_|t])·[n|ε],  [a|t] ) ‖²', 164, 668, 28, { a: c1 });
    const c2 = eout(prog(t, 24, 0.8)); card(130, 730, 1220, 90, P.sage, c2); M('[m|ε][m_|θ] = −√(1 − [a|ᾱ][a_|t])·[s|s][s_|θ]         (episode 8: “the network predicts the noise”)', 164, 786, 24, { a: c2 });
    const c3 = eout(prog(t, 32, 0.8)); card(130, 840, 1220, 90, P.ink, c3); M('[s|x̂₀] = ( [d|x][d_|t] − √(1 − [a|ᾱ][a_|t])·[m|ε][m_|θ] ) / √[a|ᾱ][a_|t]        a denoiser', 164, 896, 24, { a: c3 });
    const ha = eout(prog(t, 40, 0.8)); if (ha > 0) { const H = A.hist, x0p = 1420, y0 = 900, w = 400, h = 200, lo = Math.min(...H), hi = Math.max(...H.slice(3));
      card(1390, 600, 440, 330, P.amber, ha); text('training loss (real run)', x0p, 640, { font: F.mono, size: 15, color: P.dust, a: ha });
      setA(ha); ctx.strokeStyle = P.plum; ctx.lineWidth = 2.5; ctx.beginPath(); H.forEach((v, i) => { const X = x0p + i / (H.length - 1) * w, Y = y0 - clamp((v - lo) / (hi - lo)) * h; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1); }
  } });

// 04 — running it backwards (2-D)
const RX = A.ring.x, RX0 = A.ring.x0, NR = RX.length - 1, fR = t => NR * ease(prog(t, 4, 38));
SC.push({ dur: 52, chapter: 'sampling', title: 'Running it backwards', sub: 'Predict the noise, remove a little, add a little back.',
  caps: [[1, 8, 'Now run it backwards. Start from pure static: 600 points drawn from N(0, I).'],
         [8, 16, 'At each step: predict the noise, remove a little of it, rescale, and add a little fresh noise back.'],
         [16, 24, 'The fresh noise keeps it a sampler, not a single-answer denoiser. Without it, every point would slide to the same average.'],
         [24, 32, 'Right: the network’s guess of the finished picture, x̂₀, at every step. Early on, the best guess is the average of everything: a blur at the centre.'],
         [32, 40, 'As the noise falls, the guesses commit: first to a region, then to a blob, then to a precise spot.'],
         [40, 47, 'Coarse first, fine last: the same country–city–street order as annealed Langevin. This is annealed Langevin, done carefully.'],
         [47, 51.6, 'Now let’s point it at something much harder.']],
  draw(t) {
    const L = square(110, 250, 600), Rm = square(760, 250, 600), a0 = eout(prog(t, 0.6, 0.8)), f = fR(t), i = Math.min(NR, Math.round(f)), tt = A.ring.ts[Math.min(NR, Math.floor(f))];
    frame(L, a0); clipTo(L, () => lerpPts(RX, f).forEach(p => marble(L.X(p[0]), L.Y(p[1]), 2.8, mixc(P.rose, P.terracotta, f / NR), a0)));
    text(`xₜ   ·   t = ${tt < 0 ? 0 : tt}`, 110, 885, { font: F.mono, size: 20, color: P.chalk, a: a0 });
    const ra = eout(prog(t, 24, 1)); frame(Rm, ra); clipTo(Rm, () => lerpPts(RX0, f).forEach(p => dot(Rm.X(p[0]), Rm.Y(p[1]), 2.4, P.sage, ra * 0.8)));
    text('x̂₀: the network’s current guess', 760, 885, { font: F.mono, size: 20, color: P.sage, a: ra });
    const c1 = eout(prog(t, 8, 0.8)); card(1400, 250, 430, 360, P.plum, c1); label('one reverse step', 1428, 296, c1, P.plum);
    M('[d|x][d_|t−1] =', 1428, 352, 22, { a: c1 }); M('  ( [d|x][d_|t] − [a|β][a_|t]/√(1−[a|ᾱ][a_|t])·[m|ε][m_|θ] )', 1428, 392, 20, { a: c1 }); M('  / √(1 − [a|β][a_|t])', 1428, 428, 20, { a: c1 }); M('  + [a|σ][a_|t]·[n|z]', 1428, 468, 22, { a: eout(prog(t, 16, 0.8)) });
    text('remove a little noise · add a little back', 1428, 520, { font: F.mono, size: 14, color: P.dust, a: c1 }); text(`1000 steps`, 1428, 570, { font: F.mono, size: 18, color: P.amber, a: c1 });
    const c2 = eout(prog(t, 40, 0.8)); if (c2 > 0) { card(1400, 640, 430, 250, P.amber, c2); text('country', 1428, 710, { font: F.serif, italic: true, size: 34, color: P.amber, a: c2 }); text('→ city', 1500, 770, { font: F.serif, italic: true, size: 34, color: P.amber, a: eout(prog(t, 41, 0.6)) }); text('→ street', 1560, 830, { font: F.serif, italic: true, size: 34, color: P.amber, a: eout(prog(t, 42, 0.6)) }); }
  } });

// 05 — the wow: 64 digits from static
const fD = t => NDS * Math.pow(ease(prog(t, 4, 16)), 0.85);
SC.push({ dur: 50, chapter: 'digits', title: 'Sixty-four digits from nothing', sub: 'The same recipe, now in 784 dimensions.',
  caps: [[1, 8, 'Sixty-four images of pure static. Same network, same recipe, now in 784 dimensions.'],
         [8, 19, 'A thousand steps. Watch.'],
         [20, 28, 'Sixty-four handwritten digits, from nothing but noise and a network that learned to remove it.'],
         [28, 36, 'Here is what the network believes at each step, x̂₀, for the first eight. At first, a ghostly average of all digits.'],
         [36, 43, 'Then it commits to which digit, surprisingly early. The strokes, the thickness, the little details come last.'],
         [43, 49.6, 'Composition first, detail last. That order is a big part of why diffusion images look so coherent.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), f = fD(t), i = Math.min(NDS - 1, Math.floor(f)), u = f - i, done = t > 20.2;
    const gx = t < 27 ? 614 : 110, gy = 250, cell = t < 27 ? 82 : 76, pad = 4, ga = a0 * (t < 27 ? 1 : 1);
    for (let k = 0; k < 64; k++) { const x = gx + (k % 8) * (cell + pad), y = gy + Math.floor(k / 8) * (cell + pad);
      drawDigit(digitAt(i, k), `d${i}_${k}`, x, y, cell, ga * (1 - u)); if (u > 0) drawDigit(digitAt(i + 1, k), `d${i + 1}_${k}`, x, y, cell, ga * u); }
    const tt = A.digits.ts[Math.min(NDS, Math.round(f))]; text(t < 21 ? `t = ${tt < 0 ? 0 : tt}` : 'done: 1,000 steps each', t < 27 ? gx - 24 : gx, t < 27 ? 280 : 250 + 8 * (cell + pad) + 30, { font: F.mono, size: 20, color: P.amber, align: t < 27 ? 'right' : 'left', a: a0 });
    const ga2 = t > 20.2 ? Math.exp(-(t - 20.2) * 1.5) : 0; if (ga2 > 0.01) { ctx.save(); ctx.globalCompositeOperation = 'lighter'; setA(ga2 * 0.35); ctx.fillStyle = P.terracotta; ctx.fillRect(gx - 10, gy - 10, 8 * (cell + pad) + 16, 8 * (cell + pad) + 16); ctx.restore(); setA(1); }
    // watch it think: x̂₀ over time for eight digits
    const wa = eout(prog(t, 28, 1));
    if (wa > 0) { const cols = [0, 5, 10, 15, 20, 25, 29, NDS], x0 = 780, y0 = 280, c = 118;
      text('x̂₀ at step →', x0, 262, { font: F.mono, size: 15, color: P.sage, a: wa });
      cols.forEach((s, j) => { const ts = A.digits.ts[s]; text(ts < 0 ? 'final' : `${ts}`, x0 + j * (c + 12) + c / 2, y0 + 8 * 0 - 0, { font: F.mono, size: 13, color: P.dust, align: 'center', a: wa }); });
      for (let r = 0; r < 5; r++) cols.forEach((s, j) => { const im = imgs(DS.x0[s], 8)[r]; drawDigit(im, `x0${s}_${r}`, x0 + j * (c + 12), y0 + 14 + r * (c + 10), c, wa * eout(prog(t, 28.3 + j * 0.25 + r * 0.1, 0.5))); }); }
  } });

// 06 — why it works: a VAE with a thousand layers
const CHAIN = [NDS, 26, 20, 14, 8, 0];
SC.push({ dur: 48, chapter: 'why it works', title: 'A VAE with a thousand layers', sub: 'Why such a simple loss is maximum likelihood in disguise.',
  caps: [[1, 8, 'Why does such a simple loss work? Remember episode 3’s VAE: an encoder squeezes x into a latent z, a decoder rebuilds it, and the ELBO keeps it honest.'],
         [8, 16, 'Diffusion is a VAE with a thousand latent layers. The encoder is the fixed noising process: nothing to learn.'],
         [16, 24, 'The last layer, x₁₀₀₀, is exactly N(0, I) by construction. The VAE’s leash comes for free.'],
         [24, 32, 'Write out the ELBO and it splits into one term per step: how well does the learned reverse step match the true one, given x₀?'],
         [32, 40, 'Both are Gaussians of the same width, so each term is just a squared distance between their centres, which works out to ‖ε − εθ‖².'],
         [40, 47.6, 'Drop the per-step weights and you get the simple loss. A bound on maximum likelihood once again, split into a thousand easy pieces.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), y = 420, c = 150;
    CHAIN.forEach((s, j) => { const x = 150 + j * 290, a = a0 * eout(prog(t, 1 + j * 0.3, 0.5)); drawDigit(digitAt(s, 3), `ch${s}`, x, y - c / 2, c, a);
      const ts = A.digits.ts[s]; text(j === 0 ? 'x₀' : j === CHAIN.length - 1 ? 'x₁₀₀₀ ~ N(0, I)' : 'x' + String(ts).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]), x + c / 2, y + c / 2 + 34, { font: F.mono, size: 18, color: j === CHAIN.length - 1 ? P.rose : P.terracotta, align: 'center', a });
      if (j < CHAIN.length - 1) { const fa = a0 * eout(prog(t, 8, 0.8)), ba = a0 * eout(prog(t, 10, 0.8)), xa = x + c + 16, xb = x + 290 - 16;
        arrow(xa, y - 40, xb, y - 40, P.rose, 2.5, fa); arrow(xb, y + 40, xa, y + 40, P.sage, 2.5, ba); } });
    text('q: add noise (fixed, the “encoder”)', 960, 300, { font: F.mono, size: 20, color: P.rose, align: 'center', a: a0 * eout(prog(t, 8, 0.8)) });
    text('pθ: remove noise (learned, the “decoder”)', 960, 590, { font: F.mono, size: 20, color: P.sage, align: 'center', a: a0 * eout(prog(t, 10, 0.8)) });
    const c1 = eout(prog(t, 24, 0.8)); card(150, 650, 1620, 270, P.plum, c1); label('the elbo, one term per step', 184, 696, c1, P.plum);
    M('−log [p|p]([d|x₀]) ≤ [kB|Σ][k_|t] KL( q([d|x][d_|t−1] | [d|x][d_|t], [d|x₀])  ‖  [s|p][s_|θ]([d|x][d_|t−1] | [d|x][d_|t]) ) + …', 184, 766, 26, { a: c1 });
    M('   = [kB|Σ][k_|t] w[t_|t] · ‖ [n|ε] − [m|ε][m_|θ]([d|x][d_|t], t) ‖² + const          (two Gaussians, same width)', 184, 830, 26, { a: eout(prog(t, 32, 0.8)) });
    M('   set w[t_|t] = 1  →  the simple loss', 184, 890, 26, { a: eout(prog(t, 40, 0.8)) });
  } });

// 07 — fewer steps: DDIM
SC.push({ dur: 50, chapter: 'fewer steps', title: 'Fewer steps: DDIM', sub: 'Jump between noise levels using the network’s own guess.',
  caps: [[1, 8, 'A thousand network calls per image is slow. Can we take bigger steps?'],
         [8, 16, 'DDIM, 2021: use the network’s guess x̂₀ to jump directly between any two noise levels, with no fresh noise at all.'],
         [16, 24, 'Same network, no retraining. 1,000 steps, 100, 20: the quality holds up surprisingly well. At 5 steps it starts to crack.'],
         [24, 32, 'Without fresh noise, sampling is deterministic: each noise image maps to exactly one digit. The noise has become a latent code, like the VAE’s z.'],
         [32, 40, 'So slide smoothly between two noise images, and the digit morphs smoothly from one to another.'],
         [40, 49.6, 'Big steps work because the path from noise to data is smooth. But it isn’t straight. Hold that thought.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), fade = 1 - ease(prog(t, 23.5, 1));
    const c1 = eout(prog(t, 8, 0.8)) * fade; card(110, 250, 1720, 110, P.sage, c1); M('[d|x][d_|s] = √[a|ᾱ][a_|s]·[s|x̂₀] + √(1 − [a|ᾱ][a_|s])·[m|ε][m_|θ]([d|x][d_|t], t)        any s < t, no fresh noise', 144, 318, 28, { a: c1 });
    if (fade > 0) [['1000', 'DDPM · 1000 steps', 1], ['100', 'DDIM · 100', 16], ['20', 'DDIM · 20', 18], ['5', 'DDIM · 5', 20]].forEach(([k, nm, t0], j) => {
      const a = a0 * fade * eout(prog(t, t0, 0.6)), x0 = 110 + j * 440; if (a <= 0) return; const ims = imgs(A.stepsCmp[k], 16);
      for (let q = 0; q < 16; q++) drawDigit(ims[q], `sc${k}_${q}`, x0 + (q % 4) * 98, 400 + Math.floor(q / 4) * 98, 94, a); text(nm, x0, 810, { font: F.mono, size: 20, color: j ? P.sage : P.chalk, a }); });
    const ia = eout(prog(t, 24, 1));
    if (ia > 0) { const ims = imgs(A.interp, 16), w = 104;
      text('slide between two noise images → the digit slides too', 110, 300, { font: F.mono, size: 20, color: P.amber, a: ia });
      for (let q = 0; q < 16; q++) drawDigit(ims[q], `ip${q}`, 110 + q * (w + 4), 340, w, ia * eout(prog(t, 24.5 + q * 0.12, 0.4)));
      const u = 0.5 + 0.5 * Math.sin((t - 32) * 0.9), qi = clamp(Math.round(u * 15), 0, 15);
      if (t > 32) { drawDigit(ims[qi], `ip${qi}`, 760, 500, 380, ia * eout(prog(t, 32, 0.8))); text(`noise mix ${(u * 100).toFixed(0)}%`, 950, 920, { font: F.mono, size: 18, color: P.amber, align: 'center', a: ia * eout(prog(t, 32, 0.8)) }); } }
  } });

// 08 — in the wild
SC.push({ dur: 44, chapter: 'in the wild', title: 'Diffusion, everywhere', sub: 'From digits to the images, video and molecules of today.',
  caps: [[1, 9, 'Scale this recipe up and you get the image generators of 2022: DALL·E 2, Imagen, Stable Diffusion.'],
         [9, 18, 'Stable Diffusion runs diffusion not on pixels but in the latent space of an autoencoder, 48 times smaller: episode 3’s idea, back for the big stage.'],
         [18, 27, 'Condition the noise predictor on a caption and it draws what you ask for. How that steering works is episode 11.'],
         [27, 36, 'The same recipe generates video, audio, weather forecasts, and new protein structures that have been made in the lab.'],
         [36, 43.6, 'All of it: add noise, learn to predict it, and run the process backwards.']],
  draw(t) {
    const cards = [[1, 'images · 2022', P.terracotta, 'DALL·E 2 · Imagen · Stable Diffusion', 'text-to-image at photographic quality'],
      [9, 'latent diffusion', P.plum, 'diffuse in a VAE’s latent space', '512×512×3 → 64×64×4: 48× fewer numbers'],
      [18, 'conditioning', P.amber, 'εθ(xₜ, t, caption)', 'steering with text: episode 11'],
      [27, 'beyond images', P.sage, 'video · audio · weather · proteins', 'e.g. RFdiffusion designs protein binders']];
    cards.forEach(([t0, h, c, m1, s], i) => { const a = eout(prog(t, t0, 0.8)), x = 100 + (i % 2) * 880, y = 260 + Math.floor(i / 2) * 330;
      card(x, y, 840, 300, c, a); label(h, x + 32, y + 46, a, c); text(m1, x + 32, y + 140, { font: F.mono, size: 26, color: P.chalk, a }); para(s, x + 32, y + 206, 780, { size: 24, a: a * eout(prog(t, t0 + 1, 0.8)) }); });
  } });

// 09 — build it
const CODE = [
  '[k|# the forward process: jump to any step in one line]',
  'betas = np.linspace([a|1e-4], [a|0.02], [a|1000]);  abar = np.cumprod([a|1] - betas)',
  '',
  '[k|# training: predict the noise]',
  '[m|for] step [m|in] [p|range]([a|30000]):',
  '    x0 = [d|batch]([a|128]);  t = randint([a|0], [a|1000], [a|128]);  eps = randn(*x0.shape)',
  '    xt = sqrt(abar[t])[:, [m|None]]*x0 + sqrt([a|1]-abar[t])[:, [m|None]]*eps',
  '    loss = ((net(xt, t) - eps)**[a|2]).mean();  net.[p|backward]();  net.[p|step]()',
  '',
  '[k|# sampling: run it backwards]',
  'x = randn(n, [a|784])',
  '[m|for] t [m|in] [p|range]([a|999], -[a|1], -[a|1]):',
  '    e = net(x, t)',
  '    x = (x - betas[t]/sqrt([a|1]-abar[t])*e) / sqrt([a|1]-betas[t])',
  '    [m|if] t > [a|0]: x += sqrt(betas[t]*([a|1]-abar[t-[a|1]])/([a|1]-abar[t])) * randn(n, [a|784])',
];
SC.push({ dur: 32, chapter: 'build it', title: 'Build it yourself', sub: 'The forward process, training and sampling, in fifteen lines.',
  caps: [[1, 9, 'The whole method fits on a screen: a noise schedule, a regression onto the noise, and a loop that runs it backwards. The full code, with hand-written backprop, is linked below.'],
         [9, 18, 'Train it on the eight blobs in a minute, or on MNIST in about fifteen minutes. Then swap the sampler for DDIM and count how few steps you need.'],
         [18, 31.6, 'Recap: destroy data slowly with Gaussian noise; learn to predict that noise at every level; run the chain backwards from static.']],
  draw(t) {
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1100, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 296 + i * 40, 18, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8)); card(1240, 240, 580, 690, P.sage, ra); label('the whole episode', 1276, 290, ra, P.sage);
    [['forward (fixed)', '[d|x]_t = √[a|ᾱ]·[d|x₀] + √(1−[a|ᾱ])·[n|ε]'], ['training', '‖[n|ε] − [m|εθ]([d|x]_t, t)‖²'], ['a denoiser', '[s|x̂₀] = ([d|x]_t − √(1−[a|ᾱ])[m|εθ])/√[a|ᾱ]'], ['sampling', '1000 small steps, or DDIM: 20–100'], ['why', 'a VAE with 1000 layers (ELBO)']].forEach(([nm, m], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.6, 0.8)); text(nm, 1276, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(m, 1276, 412 + i * 112, 20, { a: ai }); });
  } });

// 10 — hook → flow matching
const PATHS = A.ring.paths, NP = PATHS.length - 1;
SC.push({ dur: 48, chapter: 'next', title: 'Curved roads', sub: 'Why so many steps?',
  caps: [[1, 8, 'Look at the roads the particles took from noise to data. Each one bends and curves as the network steers it.'],
         [8, 16, 'Curved roads are why we need many small steps: take one big step along a curve and you fly off it.'],
         [16, 24, 'Now imagine a straight line from each noise point to its data point. One big step would get you there.'],
         [24, 32, 'What if we trained a network not to predict noise, but to push points along straight lines: a velocity field?'],
         [32, 39, 'Continuous flows from episode 6, diffusion from today, and straight roads all meet in one idea.'],
         [39, 47.6, 'Next episode: flow matching. The straight way from noise to data.']],
  draw(t) {
    const dimAll = 1 - 0.9 * ease(prog(t, 38.5, 1)); GA = dimAll;
    const m = square(110, 250, 640, 2.4), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const grow = NP * ease(prog(t, 1, 7));
    clipTo(m, () => PATHS[0].forEach((_, k) => { if (k % 2) return; setA(a0 * 0.75); ctx.strokeStyle = mixc(P.rose, P.terracotta, 0.5); ctx.lineWidth = 1.4; ctx.beginPath();
      for (let s = 0; s <= Math.min(NP, grow); s++) { const p = PATHS[s][k]; s ? ctx.lineTo(m.X(p[0]), m.Y(p[1])) : ctx.moveTo(m.X(p[0]), m.Y(p[1])); } ctx.stroke(); setA(1);
      const e = PATHS[Math.min(NP, Math.floor(grow))][k]; dot(m.X(e[0]), m.Y(e[1]), 2.6, P.terracotta, a0);
      const sa = eout(prog(t, 16, 1)); if (sa > 0) { const p0 = PATHS[0][k], p1 = PATHS[NP][k]; line(m.X(p0[0]), m.Y(p0[1]), m.X(p0[0] + (p1[0] - p0[0]) * sa), m.Y(p0[1] + (p1[1] - p0[1]) * sa), P.sage, 1.6, sa * 0.9, [5, 4]); } }));
    text('rose: the roads DDIM took · sage dashed: straight lines', 110, 925, { font: F.mono, size: 16, color: P.dust, a: a0 });
    const c1 = eout(prog(t, 8, 0.8)); card(830, 250, 1000, 170, P.rose, c1); label('curved roads', 864, 296, c1, P.rose); para('A big step along a curve flies off it, so we take many small steps.', 864, 350, 920, { size: 26, a: c1 });
    const c2 = eout(prog(t, 16, 0.8)); card(830, 440, 1000, 190, P.sage, c2); label('straight roads', 864, 486, c2, P.sage);
    M('[d|x][d_|t] = (1 − t)·[n|noise] + t·[d|data]       velocity: [d|data] − [n|noise]', 864, 550, 26, { a: c2 }); text('one big step would do', 864, 600, { font: F.mono, size: 16, color: P.dust, a: c2 });
    const c3 = eout(prog(t, 24, 0.8)); card(830, 650, 1000, 150, P.plum, c3); text('learn the velocity field instead of the noise?', 864, 738, { font: F.serif, italic: true, size: 32, color: P.chalk, a: c3 });
    GA = 1;
    const qa = eout(prog(t, 38.5, 1.2));
    if (qa > 0) {
      text('Straight lines.', 960, 440, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('Fewer steps.', 960, 530, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: eout(prog(t, 39.3, 1.2)) });
      text('NEXT  ·  EPISODE 10', 960, 640, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 40.5, 1)), ls: 4 });
      text('Flow matching — the straight way from noise to data', 960, 695, { font: F.serif, italic: true, size: 42, color: P.stone, align: 'center', a: eout(prog(t, 41, 1)) });
    }
  } });
