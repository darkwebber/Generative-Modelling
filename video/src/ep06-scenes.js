// ─────────────────────────────────────────────────────────────
//  Real trained flows (exported by code/export_ep06_assets.py)
// ─────────────────────────────────────────────────────────────
const A = window.EP6, K = A.layers, RR = A.R;
function b64bytes(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
const IMC = new Map();
function cached(key, make) { let c = IMC.get(key); if (!c) { c = make(); IMC.set(key, c); } return c; }
function densCanvas(d, col) { return cached('d' + d.max + ':' + d.u8.length + col, () => { const u = b64bytes(d.u8), n = d.n, c = document.createElement('canvas'); c.width = c.height = n; const g = c.getContext('2d'), im = g.createImageData(n, n), rgb = hx(col);
  for (let k = 0; k < n * n; k++) { im.data[k * 4] = rgb[0]; im.data[k * 4 + 1] = rgb[1]; im.data[k * 4 + 2] = rgb[2]; im.data[k * 4 + 3] = Math.round(u[k] * 0.92); } g.putImageData(im, 0, 0); return c; }); }
function square(x0, y0, S) { return { X: v => x0 + (v + RR) / (2 * RR) * S, Y: v => y0 + (RR - v) / (2 * RR) * S, s: S / (2 * RR), x0, y0, S }; }
function frame(m, a = 1) { setA(a); ctx.fillStyle = P.surface; ctx.fillRect(m.x0, m.y0, m.S, m.S); ctx.strokeStyle = P.border; ctx.lineWidth = 1.5; ctx.strokeRect(m.x0, m.y0, m.S, m.S); setA(1); }
function clipTo(m, fn) { ctx.save(); ctx.beginPath(); ctx.rect(m.x0, m.y0, m.S, m.S); ctx.clip(); fn(); ctx.restore(); }
function stageAt(stages, f) { const n = stages.length, fi = clamp(f, 0, n - 1), i = Math.floor(fi), j = Math.min(n - 1, i + 1), u = ease(fi - i); return [stages[i], stages[j], u]; }
function drawPts(m, stages, f, col, a, r = 2.6, colFn = null) { const [S0, S1, u] = stageAt(stages, f); for (let k = 0; k < S0.length; k++) { const x = lerp(S0[k][0], S1[k][0], u), y = lerp(S0[k][1], S1[k][1], u); if (Math.abs(x) < RR && Math.abs(y) < RR) dot(m.X(x), m.Y(y), r, colFn ? colFn(f) : col, a); } }
function drawGrid(m, f, a, col = P.borderLight) { const G = A.grid, [S0, S1, u] = stageAt(G.stages, f); setA(a); ctx.strokeStyle = col; ctx.lineWidth = 1.2;
  for (let l = 0; l < G.n; l++) { ctx.beginPath(); for (let k = 0; k < G.m; k++) { const i = l * G.m + k, x = lerp(S0[i][0], S1[i][0], u), y = lerp(S0[i][1], S1[i][1], u); k ? ctx.lineTo(m.X(x), m.Y(y)) : ctx.moveTo(m.X(x), m.Y(y)); } ctx.stroke(); } setA(1); }
const EX = A.example, sgn = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open
SC.push({ dur: 24, chapter: 'intro',
  caps: [[0.6, 6, 'Pure noise goes in. Two moons come out. So far, nothing new…'],
         [6, 12, '…except that this time, run the machine backwards and every point returns exactly where it started.'],
         [12, 18, "A generator that runs both ways can answer the question a GAN couldn't: exactly how likely is this point?"],
         [18, 23.6, 'Today: normalizing flows.']],
  draw(t) {
    const dim = lerp(1, 0.16, ease(prog(t, 12.5, 1.5))), m = square(560, 200, 800);
    const f = t < 6 ? lerp(0, K, ease(prog(t, 0.5, 5))) : lerp(K, 0, ease(prog(t, 6.5, 5)));
    GA = dim; clipTo(m, () => { drawGrid(m, f, eout(prog(t, 0.3, 1)) * 0.6); drawPts(m, A.inv, f, null, 0.85, 2.6, ff => mixc(P.rose, P.terracotta, ff / K)); });
    text(t < 6 ? 'noise  →  data' : 'data  →  noise', 960, 175, { font: F.mono, size: 22, color: P.dust, align: 'center', a: eout(prog(t, 0.5, 1)) }); GA = 1;
    const a = eout(prog(t, 13, 1.4));
    text('EPISODE 06', 960, 430, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('Normalizing Flows', 960, 550 + (1 - a) * 16, { font: F.serif, size: 124, color: P.chalk, align: 'center', a, ls: -1 });
    text('bending space, exactly', 960, 622, { size: 34, color: P.stone, align: 'center', a: eout(prog(t, 13.8, 1.2)) });
    const tg = [[P.terracotta, 'x  data'], [P.rose, 'z  noise'], [P.ink, 'p  density'], [P.amber, 's, t, det  stretch'], [P.plum, 'f  the flow']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let cx = 960 - tw / 2; tg.forEach(([c, s], i) => { cx += tag(cx, 690, c, s, eout(prog(t, 16 + i * 0.35, 0.6)), 21) + 14; });
  } });

// 01 — conservation of sand (1-D)
const f1 = z => z + 0.7 * Math.sin(1.3 * z), df1 = z => 1 + 0.91 * Math.cos(1.3 * z);
const finv1 = x => { let lo = -8, hi = 8; for (let i = 0; i < 50; i++) { const mid = (lo + hi) / 2; if (f1(mid) < x) lo = mid; else hi = mid; } return (lo + hi) / 2; };
const nz = z => Math.exp(-z * z / 2) / Math.sqrt(TAU), px1 = x => { const z = finv1(x); return nz(z) / df1(z); };
SC.push({ dur: 44, chapter: 'sand', title: 'Probability is sand', sub: 'Move it around all you like — you can never create or destroy it.',
  caps: [[1, 8, 'Back to basics, in one dimension. Pour probability like sand along a line: a bell curve of noise, z.'],
         [8, 16, 'Now bend the line with a function x = f(z). Every grain moves, but none is created or destroyed.'],
         [16, 24, 'Take a thin slice of width dz. It holds p(z)·dz of sand. After the bend, the same sand sits in a slice of width dx.'],
         [24, 32, 'Same sand, so p(x)·dx = p(z)·dz, and p(x) = p(z)·|dz/dx|. Where f stretches space, the density thins; where it squeezes, it piles up.'],
         [32, 43.6, "That's episode 1's change of variables. But to use it on a data point x, we have to get back to its z — so f must be invertible."]],
  draw(t) {
    const zy = 780, xx = 620, ZX = z => 680 + (z + 3) * 100, XY = x => 780 - (x + 3.4) * 78, a0 = eout(prog(t, 0.6, 0.8));
    line(xx, zy, 1300, zy, P.borderLight, 2, a0); line(xx, zy, xx, 220, P.borderLight, 2, a0);
    text('z  (noise) →', 1300, zy + 34, { font: F.mono, size: 20, color: P.rose, align: 'right', a: a0 }); text('x  (data) ↑', xx + 14, 238, { font: F.mono, size: 20, color: P.terracotta, a: a0 });
    // bell curve of z (below the axis)
    setA(a0 * 0.25); ctx.fillStyle = P.rose; ctx.beginPath(); ctx.moveTo(ZX(-3), zy + 6); for (let i = 0; i <= 120; i++) { const z = -3 + 6 * i / 120; ctx.lineTo(ZX(z), zy + 6 + nz(z) * 260); } ctx.lineTo(ZX(3), zy + 6); ctx.fill(); setA(1);
    const ca = eout(prog(t, 8, 1.5));
    if (ca > 0) { setA(ca); ctx.strokeStyle = P.plum; ctx.lineWidth = 4; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const z = -3 + 6 * i / 200; i ? ctx.lineTo(ZX(z), XY(f1(z))) : ctx.moveTo(ZX(z), XY(f1(z))); } ctx.stroke(); setA(1); pill(ZX(2.4), XY(f1(2.4)) + 40, 'x = f(z)', P.plum, ca, { align: 'center' });
      // resulting density on the x axis (to the left)
      setA(ca * 0.25); ctx.fillStyle = P.ink; ctx.beginPath(); ctx.moveTo(xx - 6, XY(-3.4)); for (let i = 0; i <= 200; i++) { const x = -3.4 + 6.8 * i / 200; ctx.lineTo(xx - 6 - px1(x) * 300, XY(x)); } ctx.lineTo(xx - 6, XY(3.4)); ctx.fill();
      setA(ca); ctx.strokeStyle = P.ink; ctx.lineWidth = 3; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -3.4 + 6.8 * i / 200; i ? ctx.lineTo(xx - 6 - px1(x) * 300, XY(x)) : ctx.moveTo(xx - 6 - px1(x) * 300, XY(x)); } ctx.stroke(); setA(1);
      text('p(x)', xx - 250, XY(3.2), { font: F.mono, size: 22, color: P.ink, a: ca }); }
    // the slice
    const sa = eout(prog(t, 16, 0.8));
    if (sa > 0) { const z0 = t < 24 ? 0.5 : lerp(-1.8, 1.9, 0.5 - 0.5 * Math.cos((t - 24) * 0.5)), dz = 0.3, x0 = f1(z0), x1 = f1(z0 + dz);
      setA(sa * 0.55); ctx.fillStyle = P.amber; ctx.fillRect(ZX(z0), zy + 6, ZX(z0 + dz) - ZX(z0), nz(z0) * 260); ctx.fillRect(xx - 6 - px1((x0 + x1) / 2) * 300, XY(x1), px1((x0 + x1) / 2) * 300, XY(x0) - XY(x1)); setA(sa * 0.12);
      ctx.beginPath(); ctx.moveTo(ZX(z0), zy); ctx.lineTo(ZX(z0), XY(x0)); ctx.lineTo(xx, XY(x0)); ctx.lineTo(xx, XY(x1)); ctx.lineTo(ZX(z0 + dz), XY(x1)); ctx.lineTo(ZX(z0 + dz), zy); ctx.fill(); setA(1);
      text(`dz = ${dz.toFixed(2)}`, ZX(z0 + dz / 2), zy + 36, { font: F.mono, size: 16, color: P.amber, align: 'center', a: sa });
      text(`dx = ${(x1 - x0).toFixed(2)}`, xx + 12, XY((x0 + x1) / 2) + 6, { font: F.mono, size: 16, color: P.amber, a: sa });
      text(df1(z0 + dz / 2) > 1 ? 'stretched → thinner' : 'squeezed → denser', ZX(z0 + dz / 2) + 30, XY((x0 + x1) / 2) - 20, { font: F.mono, size: 16, color: P.stone, a: sa * eout(prog(t, 24, 0.8)) }); }
    const ma = eout(prog(t, 24, 0.8));
    card(1360, 300, 470, 260, P.ink, ma); label('sand is conserved', 1394, 346, ma, P.ink);
    M('[p|p]([d|x])·d[d|x] = [p|p]([n|z])·d[n|z]', 1394, 416, 24, { a: ma });
    M('[p|p]([d|x]) = [p|p]([n|z]) · |d[n|z]/d[d|x]|', 1394, 480, 26, { a: eout(prog(t, 26, 0.8)) });
    text('needs z = f⁻¹(x)', 1394, 532, { font: F.mono, size: 18, color: P.rose, a: eout(prog(t, 33, 0.8)) });
  } });

// 02 — the determinant
const Jm = [[1.4, 0.6], [0.25, 0.9]], DETJ = Jm[0][0] * Jm[1][1] - Jm[0][1] * Jm[1][0];
SC.push({ dur: 44, chapter: 'determinant', title: 'In 2-D, stretch is a determinant', sub: 'How much does a tiny square grow?',
  caps: [[1, 8, 'In two dimensions, the thin slice becomes a tiny square. Push it through a map and it becomes a tiny parallelogram.'],
         [8, 16, 'How much did its area change? For a linear map with matrix J, the answer is its determinant: ad − bc.'],
         [16, 24, 'A curvy map looks linear up close. Zoom in anywhere, and its Jacobian — the matrix of partial derivatives — tells you the local stretch.'],
         [24, 32, "So the density rule becomes p(x) = p(z) · |det ∂z/∂x|. In a million dimensions it's the same formula — with a million-by-million determinant."],
         [32, 43.6, 'And that is the catch. A general determinant costs about D³ operations. For a 256 × 256 colour image, that is roughly 10¹⁶ — per image, per step.']],
  draw(t) {
    const A1 = 1 - ease(prog(t, 15.5, 1));
    if (A1 > 0) { GA = A1; const cx = 600, cy = 620, s = 150, u = ease(prog(t, 2, 4)), a0 = eout(prog(t, 0.6, 0.8));
      line(cx - 380, cy, cx + 420, cy, P.border, 1.5, a0); line(cx, cy + 220, cx, cy - 380, P.border, 1.5, a0);
      const M2 = [[lerp(1, Jm[0][0], u), lerp(0, Jm[0][1], u)], [lerp(0, Jm[1][0], u), lerp(1, Jm[1][1], u)]], P2 = (x, y) => [cx + (M2[0][0] * x + M2[0][1] * y) * s, cy - (M2[1][0] * x + M2[1][1] * y) * s];
      const q = [P2(0, 0), P2(1, 0), P2(1, 1), P2(0, 1)]; setA(a0 * 0.25); ctx.fillStyle = P.amber; ctx.beginPath(); q.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.fill(); setA(a0); ctx.strokeStyle = P.amber; ctx.lineWidth = 3; ctx.stroke(); setA(1);
      const det = M2[0][0] * M2[1][1] - M2[0][1] * M2[1][0]; text(`area = ${det.toFixed(2)}`, q[2][0] + 20, q[2][1], { font: F.mono, size: 24, color: P.amber, a: a0 });
      const ma = eout(prog(t, 8, 0.8)); card(1150, 300, 670, 300, P.amber, ma); label('a linear map', 1184, 346, ma, P.amber);
      M(`J = [ ${Jm[0][0]}  ${Jm[0][1]} ;  ${Jm[1][0]}  ${Jm[1][1]} ]`, 1184, 420, 28, { a: ma });
      M(`det J = ad − bc = ${Jm[0][0]}·${Jm[1][1]} − ${Jm[0][1]}·${Jm[1][0]} = ${DETJ.toFixed(2)}`, 1184, 490, 22, { a: eout(prog(t, 10, 0.8)) });
      text('= how much every area grows', 1184, 550, { font: F.serif, italic: true, size: 26, color: P.sage, a: eout(prog(t, 11, 0.8)) });
      GA = 1; }
    const B = eout(prog(t, 16, 1));
    if (B > 0) { const m = square(150, 240, 700); frame(m, B); clipTo(m, () => drawGrid(m, K, B * 0.8, P.stone));
      text('a real trained flow: the noise grid, warped into data space', 170, 920, { font: F.mono, size: 16, color: P.dust, a: B });
      const ca = eout(prog(t, 18, 0.8)); ring(m.X(-0.8), m.Y(1.0), 40, P.sage, ca, 2.5); ring(m.X(0), m.Y(0), 40, P.amber, ca, 2.5);
      text('tiny cells: squeezed → high density', m.X(-0.8) - 60, m.Y(1.0) - 60, { font: F.mono, size: 16, color: P.sage, a: ca });
      text('big cells: stretched → low density', m.X(0) + 20, m.Y(0) + 190, { font: F.mono, size: 16, color: P.amber, a: ca });
      const fa = eout(prog(t, 24, 0.8)); card(930, 300, 890, 250, P.ink, fa); label('in D dimensions', 964, 346, fa, P.ink);
      M('[p|p]([d|x]) = [p|p]([n|z]) · |det ∂[n|z]/∂[d|x]|', 964, 420, 32, { a: fa });
      M('∂[n|z]/∂[d|x]  is a  D × D  matrix (the Jacobian)', 964, 490, 22, { a: eout(prog(t, 26, 0.8)) });
      const ca2 = eout(prog(t, 32, 0.8)); card(930, 590, 890, 230, P.rose, ca2); label('the cost', 964, 636, ca2, P.rose);
      M('general determinant: ~[a|D]³ operations', 964, 700, 28, { a: ca2 });
      M('256×256×3 image:  D ≈ 2×10⁵   →   D³ ≈ 10¹⁶', 964, 764, 24, { a: eout(prog(t, 34, 0.8)) }); }
  } });

// 03 — the coupling layer
SC.push({ dur: 52, chapter: 'coupling', title: 'The coupling trick', sub: 'Make the determinant free — and the inverse too.',
  caps: [[1, 8, 'The trick behind RealNVP and Glow is almost cheeky. Split the coordinates into two halves, a and b.'],
         [8, 16, "Leave a completely alone. Transform b with a stretch and a shift that depend only on a:  b′ = b · e^s(a) + t(a)."],
         [16, 24, 'Watch one real layer: every point slides along one axis only, by an amount that depends on its position along the other.'],
         [24, 32, 'The Jacobian is triangular — zeros above the diagonal — so its determinant is just the product of the diagonal: e^s(a). Cost: nothing.'],
         [32, 40, 'Inverting is just as easy:  b = (b′ − t(a)) · e^−s(a). And s and t can be any neural networks at all — we never have to invert them.'],
         [40, 51.6, 'One layer can only bend one half. So stack layers, swapping the roles of a and b each time.']],
  draw(t) {
    const m = square(150, 240, 700), a0 = eout(prog(t, 0.6, 0.8)), f = t < 16 ? 0 : t < 40 ? lerp(0, 1, 0.5 - 0.5 * Math.cos((t - 16) * 0.55)) : lerp(0, 3, ease(prog(t, 41, 8)));
    frame(m, a0); clipTo(m, () => { drawGrid(m, f, a0 * 0.8, P.stone); drawPts(m, A.inv, f, P.rose, 0.7, 2.2); });
    text(t < 40 ? 'one real coupling layer' : `layers stacked: ${Math.floor(f) + 1}`, 170, 280, { font: F.mono, size: 18, color: P.chalk, a: a0 });
    const c1 = eout(prog(t, 1, 0.8)); card(930, 250, 890, 200, P.plum, c1); label('coupling layer', 964, 296, c1, P.plum);
    M('[d|a]′ = [d|a]', 964, 360, 28, { a: c1 });
    M('[d|b]′ = [d|b] · e[a^|s(a)] + [a|t]([d|a])', 964, 414, 28, { a: eout(prog(t, 8, 0.8)) });
    const c2 = eout(prog(t, 24, 0.8)); card(930, 480, 430, 240, P.amber, c2); label('jacobian', 964, 526, c2, P.amber);
    M('∂([d|a]′,[d|b]′)/∂([d|a],[d|b]) =', 964, 580, 18, { a: c2 });
    M('[ 1        0  ]', 994, 630, 24, { a: c2 }); M('[ ∗   e[a^|s(a)] ]', 994, 676, 24, { a: c2 });
    text('det = e^s(a)', 964, 710 - 2, { font: F.mono, size: 16, color: P.sage, a: eout(prog(t, 27, 0.8)) });
    const c3 = eout(prog(t, 32, 0.8)); card(1390, 480, 430, 240, P.sage, c3); label('inverse', 1424, 526, c3, P.sage);
    M('[d|b] = ([d|b]′ − [a|t]) · e[a^|−s]', 1424, 590, 22, { a: c3 }); para('s, t: any neural nets — never inverted', 1424, 640, 370, { size: 22, a: c3 });
    const c4 = eout(prog(t, 40, 0.8)); card(930, 750, 890, 110, P.plum, c4);
    M('log |det| of the whole flow  =  [kB|Σ][k_|k] [a|s][a_|k]  — just add them up', 964, 815, 22, { a: c4 });
  } });

// 04 — exact maximum likelihood
SC.push({ dur: 48, chapter: 'likelihood', title: 'Training: exact maximum likelihood', sub: 'No judge. No bound. The real thing.',
  caps: [[1, 8, 'Training is pure episode 1: maximum likelihood. No judge like the GAN, no lower bound like the VAE — the exact quantity.'],
         [8, 16, "Push a data point through all the layers to get its z. Its log-probability is the bell curve's score for z, plus every layer's log-stretch."],
         [16, 24, `A point on the moons: ${EX.on.logN.toFixed(2)} ${sgn(EX.on.logdet)} = ${EX.on.logp.toFixed(2)}. A point out in the void: ${EX.off.logN.toFixed(2)} ${sgn(EX.off.logdet)} = ${EX.off.logp.toFixed(2)}.`],
         [24, 32, 'Average −log p over the data, differentiate through every layer, and take a step. This is the real training curve.'],
         [32, 40, `On held-out points the flow scores ${A.nll.toFixed(2)} nats per point; the best single bell curve manages ${A.gauss.toFixed(2)}. Exact numbers, not estimates.`],
         [40, 47.6, "Why “normalizing”? Because the trained flow turns the data into a normal distribution."]],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8));
    M('log [p|p]([d|x]) = log N([m|f]([d|x]); 0, I) + [kB|Σ][k_|k] [a|s][a_|k]', 960, 300, 36, { align: 'center', a: a0 });
    text('bell-curve score of z            every layer’s log-stretch', 960, 350, { font: F.mono, size: 18, color: P.dust, align: 'center', a: eout(prog(t, 8, 0.8)) });
    const ea = eout(prog(t, 16, 0.8));
    [['on', 'on the moons', P.sage], ['off', 'out in the void', P.rose]].forEach(([k, nm, col], i) => { const x = 150 + i * 460, y = 430, e = EX[k], a = ea * eout(prog(t, 16 + i * 1.5, 0.8));
      card(x, y, 420, 260, col, a); label(nm, x + 30, y + 44, a, col);
      M(`[d|x] = (${e.x[0].toFixed(2)}, ${e.x[1].toFixed(2)})`, x + 30, y + 92, 20, { a }); M(`  →  [n|z] = (${e.z[0].toFixed(2)}, ${e.z[1].toFixed(2)})`, x + 30, y + 124, 20, { a });
      M(`log N([n|z]) = ${e.logN.toFixed(2)}`, x + 30, y + 166, 22, { a }); M(`[kB|Σ] [a|s] = ${sgn(e.logdet)}`, x + 30, y + 200, 22, { a });
      M(`log [p|p]([d|x]) = ${e.logp.toFixed(2)}`, x + 30, y + 236, 24, { a }); });
    const ca = eout(prog(t, 24, 0.8));
    if (ca > 0) { const x0 = 1100, y0 = 700, w = 700, h = 250, H = A.hist, mx = Math.max(...H.slice(2)), mn = Math.min(...H);
      card(x0 - 30, y0 - h - 70, w + 60, h + 130, P.plum, ca); text('training −log p (nats per point, real run)', x0, y0 - h - 30, { font: F.mono, size: 16, color: P.dust, a: ca });
      setA(ca); ctx.strokeStyle = P.plum; ctx.lineWidth = 2.5; ctx.beginPath(); const n = Math.max(2, Math.floor(H.length * ease(prog(t, 24.5, 6))));
      H.slice(0, n).forEach((v, i) => { const X = x0 + i / (H.length - 1) * w, Y = y0 - clamp((v - mn) / (mx - mn)) * h; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
      const ra = eout(prog(t, 32, 0.8)); text(`held-out: flow ${A.nll.toFixed(2)}   vs   best Gaussian ${A.gauss.toFixed(2)}`, x0, y0 + 40, { font: F.mono, size: 20, color: P.sage, a: ra }); }
    const na = eout(prog(t, 40, 0.8)); if (na > 0) text('data  →  f  →  a normal distribution', 150, 780, { font: F.serif, italic: true, size: 40, color: P.chalk, a: na });
  } });

// 05 — layer by layer
SC.push({ dur: 50, chapter: 'layer by layer', title: 'The real flow, one layer at a time', sub: `${K} coupling layers, trained on two moons.`,
  caps: [[1, 8, `Here is the real trained flow, one layer at a time: ${K} coupling layers, each a gentle, invertible bend.`],
         [8, 16, 'Forwards, data → noise: each layer nudges the moons a little closer to a round cloud.'],
         [16, 24, "After the last layer, the moons have become a standard bell curve. That's the “normalizing”."],
         [24, 32, 'Backwards, noise → data: start from fresh random points and a square grid, and undo the layers in reverse order.'],
         [32, 40, 'The grid shows the warp: squares stretched thin where the density must drop, squeezed where it must pile up.'],
         [40, 49.6, 'Every point takes the same road, forwards or back. Nothing is lost. That is the whole promise of a flow.']],
  draw(t) {
    const L = square(150, 250, 640), Rm = square(1030, 250, 640), a0 = eout(prog(t, 0.6, 0.8));
    const ff = lerp(0, K, ease(prog(t, 5, 16))), fi = lerp(0, K, ease(prog(t, 25, 14)));
    frame(L, a0); clipTo(L, () => drawPts(L, A.fwd, ff, null, 0.85, 2.6, f => mixc(P.terracotta, P.rose, f / K)));
    text(`data → noise   ·   layer ${Math.round(ff)} / ${K}`, 150, 930, { font: F.mono, size: 20, color: P.chalk, a: a0 });
    const ra = eout(prog(t, 24, 1));
    frame(Rm, ra); clipTo(Rm, () => { drawGrid(Rm, fi, ra * 0.7, P.stone); drawPts(Rm, A.inv, fi, null, ra * 0.85, 2.4, f => mixc(P.rose, P.terracotta, f / K)); });
    text(`noise → data   ·   layer ${Math.round(fi)} / ${K}`, 1030, 930, { font: F.mono, size: 20, color: P.chalk, a: ra });
    arrow(810, 570, 1010, 570, P.dust, 2, ra); text('f⁻¹', 910, 555, { font: F.mono, size: 22, color: P.plum, align: 'center', a: ra });
  } });

// 06 — the exact density
SC.push({ dur: 40, chapter: 'exact density', title: 'A density you can actually read', sub: 'Exact p(x), everywhere.',
  caps: [[1, 8, "Because every step is exact, we can paint the model's density everywhere — a map a GAN could never draw."],
         [8, 16, 'Bright on the moons, dark in the void — and it integrates to exactly one, guaranteed by construction.'],
         [16, 24, "Depth matters. With only two layers, the flow can't bend enough: the moons blur into one smeared S, and density leaks where no data lives."],
         [24, 31, `Held-out score: ${A.nllShallow.toFixed(2)} nats per point with two layers, ${A.nll.toFixed(2)} with eight.`],
         [31, 39.6, 'Flows gave the best exact likelihoods of their generation — and in 2018, Glow used this recipe to draw convincing faces.']],
  draw(t) {
    const L = square(150, 250, 640), Rm = square(1030, 250, 640), a0 = eout(prog(t, 0.6, 0.8));
    frame(L, a0); setA(a0); ctx.imageSmoothingEnabled = true; ctx.drawImage(densCanvas(A.dens, P.ink), L.x0, L.y0, L.S, L.S); setA(1);
    text(`${K} layers · exact p(x)`, 150, 930, { font: F.mono, size: 20, color: P.chalk, a: a0 });
    const ra = eout(prog(t, 16, 1));
    frame(Rm, ra); setA(ra); ctx.drawImage(densCanvas(A.densShallow, P.ink), Rm.x0, Rm.y0, Rm.S, Rm.S); setA(1);
    text('2 layers · exact p(x)', 1030, 930, { font: F.mono, size: 20, color: P.chalk, a: ra });
    const sa = eout(prog(t, 24, 0.8)); if (sa > 0) { pill(470, 225, `held-out NLL ${A.nll.toFixed(2)}`, P.sage, sa, { align: 'center' }); pill(1350, 225, `held-out NLL ${A.nllShallow.toFixed(2)}`, P.rose, sa, { align: 'center' }); }
    const ia = eout(prog(t, 8, 0.8)); if (ia > 0 && t < 16) M('[pB|∫] [p|p]([d|x]) d[d|x] = 1   exactly', 1350, 560, 34, { align: 'center', a: ia * (1 - ease(prog(t, 15, 1))) });
  } });

// 07 — the catch: topology
SC.push({ dur: 50, chapter: 'the catch', title: 'The catch: rubber can’t tear', sub: 'What an invertible, continuous map can never do.',
  caps: [[1, 8, 'Now the catch. Ask a flow to model eight separate blobs.'],
         [8, 16, `It does well — ${A.nllRing.toFixed(2)} nats against a bell curve's ${A.gaussRing.toFixed(2)} — but look closely: thin bridges connect the blobs.`],
         [16, 24, 'A flow is a continuous, invertible warp. Like rubber, it can stretch and squeeze, but never tear. One connected blob cannot become eight separate islands.'],
         [24, 32, 'So some samples always land on the bridges, in places no real data lives.'],
         [32, 40, 'Second catch: no compression. The latent z must have exactly as many numbers as the data — 784 in, 784 out. Episode 3’s VAE squeezed digits into 2.'],
         [40, 49.6, "Third: every layer is shackled. It must be invertible, with a determinant we can afford — which rules out most of the architectures you'd love to use."]],
  draw(t) {
    const m = square(150, 240, 700), a0 = eout(prog(t, 0.6, 0.8)), da = eout(prog(t, 4, 1.5));
    frame(m, a0); setA(da); ctx.drawImage(densCanvas(A.densRing, P.ink), m.x0, m.y0, m.S, m.S); setA(1);
    const sa = eout(prog(t, 24, 1)); if (sa > 0) clipTo(m, () => A.ringSamples.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.2, P.terracotta, sa * 0.8)));
    const ba = eout(prog(t, 10, 0.8)); if (ba > 0 && t < 32) { ring(m.X(0.6), m.Y(-1.85), 34, P.rose, ba, 2.5); pill(m.X(0.6) + 44, m.Y(-1.85) + 50, 'a bridge', P.rose, ba); }
    // rubber sheet cartoon
    const ra = eout(prog(t, 16, 0.8)) * (1 - ease(prog(t, 31.5, 1)));
    if (ra > 0) { card(930, 260, 890, 360, P.plum, ra); label('continuous + invertible = rubber', 964, 306, ra, P.plum);
      const cx = 1150, cy = 480, st = ease(prog(t, 17, 5)); setA(ra * 0.35); ctx.fillStyle = P.rose; ctx.beginPath(); ctx.ellipse(cx, cy, 90 + 120 * st, 70 - 40 * st, 0, 0, TAU); ctx.fill(); setA(1);
      ring(cx - 120 * st, cy, 30, P.chalk, ra * st, 1.5); ring(cx + 120 * st, cy, 30, P.chalk, ra * st, 1.5);
      text('stretch ✓', 1450, 430, { font: F.mono, size: 22, color: P.sage, a: ra }); text('squeeze ✓', 1450, 480, { font: F.mono, size: 22, color: P.sage, a: ra }); text('tear ✗', 1450, 530, { font: F.mono, size: 22, color: P.rose, a: ra }); }
    const ca = eout(prog(t, 32, 0.8));
    if (ca > 0) { card(930, 260, 890, 280, P.rose, ca); label('no compression', 964, 306, ca, P.rose);
      M('dim([n|z]) = dim([d|x]):   784 in → 784 out', 964, 380, 28, { a: ca }); M('VAE (episode 3):            784 in →   2 out', 964, 440, 28, { a: ca });
      const c2 = eout(prog(t, 40, 0.8)); card(930, 580, 890, 240, P.rose, c2); label('shackled layers', 964, 626, c2, P.rose);
      para('every layer must be invertible, with a cheap determinant — no ordinary convolutions, attention or pooling without special tricks', 964, 686, 820, { size: 24, a: c2 }); }
  } });

// 08 — continuous flows (a teaser for episode 10)
SC.push({ dur: 40, chapter: 'continuous', title: 'Infinitely many, infinitely thin layers', sub: 'File this away — it returns in episode 10.',
  caps: [[1, 8, 'One more idea — file it away, it comes back in episode 10. What if we used infinitely many, infinitely thin layers?'],
         [8, 16, 'Then each point simply follows a velocity field: dx/dt = v(x, t). The warp becomes a smooth flow, like dye drifting in water.'],
         [16, 24, 'And the costly determinant becomes something cheap: the density changes at a rate set by the trace, d log p / dt = −tr(∂v/∂x).'],
         [24, 31, 'Now any network can be the velocity field — no coupling tricks required. These are continuous normalizing flows.'],
         [31, 39.6, 'Training them was painfully slow… until a much simpler recipe arrived. We will get there: flow matching.']],
  draw(t) {
    const m = square(150, 240, 700), a0 = eout(prog(t, 0.6, 0.8)), f = lerp(0, K, 0.5 - 0.5 * Math.cos(Math.PI * clamp((t - 2) / 16)));
    frame(m, a0); clipTo(m, () => {
      // velocity arrows between consecutive stages
      const [S0, S1] = stageAt(A.inv, Math.min(K - 0.001, f)); for (let k = 0; k < 160; k++) { const p = S0[k], q = S1[k]; if (Math.abs(p[0]) < RR && Math.abs(p[1]) < RR) arrow(m.X(p[0]), m.Y(p[1]), m.X(p[0] + (q[0] - p[0]) * 1.6), m.Y(p[1] + (q[1] - p[1]) * 1.6), P.sage, 1.4, a0 * 0.6); }
      drawPts(m, A.inv, f, null, 0.85, 2.4, ff => mixc(P.rose, P.terracotta, ff / K)); });
    text('arrows: the velocity each point follows', 170, 920, { font: F.mono, size: 16, color: P.sage, a: a0 });
    const c1 = eout(prog(t, 8, 0.8)); card(930, 260, 890, 220, P.plum, c1); label('a flow as an ODE', 964, 306, c1, P.plum);
    M('d[d|x]/dt = [m|v]([d|x], t)', 964, 380, 34, { a: c1 }); text('(infinitely many tiny coupling-free steps)', 964, 440, { font: F.mono, size: 16, color: P.dust, a: c1 });
    const c2 = eout(prog(t, 16, 0.8)); card(930, 510, 890, 220, P.ink, c2); label('instantaneous change of variables', 964, 556, c2, P.ink);
    M('d log [p|p]([d|x]([k|t])) / dt = −tr( ∂[m|v]/∂[d|x] )', 964, 630, 30, { a: c2 }); text('a trace (sum of D numbers), not a determinant', 964, 690, { font: F.mono, size: 16, color: P.sage, a: c2 });
    const c3 = eout(prog(t, 31, 0.8)); if (c3 > 0) text('→ episode 10: flow matching', 964, 800, { font: F.serif, italic: true, size: 36, color: P.sage, a: c3 });
  } });

// 09 — build it
const CODE = [
  '[k|# one affine coupling layer (2-D): keep a, transform b]',
  '[m|def] [p|forward](x, net, m):',
  '    a, b = x[:, m:m+[a|1]], x[:, [a|1]-m]',
  '    raw = net(a);  s = [a|2]*np.tanh(raw[:, [a|0]]/[a|2]);  t = raw[:, [a|1]]',
  '    x[:, [a|1]-m] = b * np.exp([a|s]) + [a|t]         [k|# b′ = b·e^s + t]',
  '    [m|return] x, [a|s]                               [k|# log|det| = s]',
  '',
  '[m|def] [p|inverse](y, net, m):',
  '    a = y[:, m:m+[a|1]];  s, t = st(net(a))',
  '    y[:, [a|1]-m] = (y[:, [a|1]-m] - [a|t]) * np.exp(-[a|s])',
  '    [m|return] y',
  '',
  '[k|# exact maximum likelihood: push data to z, add the log-stretches]',
  '[n|z], logdet = [d|x], [a|0]',
  '[m|for] k, net [m|in] [p|enumerate](nets): [n|z], s = [p|forward]([n|z], net, k % [a|2]);  logdet += s',
  'nll = ([a|0.5]*([n|z]**[a|2]).sum([a|1]) + np.log([a|2]*np.pi) - logdet).mean()',
];
SC.push({ dur: 32, chapter: 'build it', title: 'Build it yourself', sub: 'A coupling layer is a handful of lines.',
  caps: [[1, 9, 'A coupling layer is a handful of lines of numpy. The full code, with hand-written backprop and gradient checks, is linked below.'],
         [9, 18, 'Train it on two moons in a few minutes — then try eight blobs and find the bridges, or drop to two layers and watch it fail to bend.'],
         [18, 31.6, 'Recap: sand is conserved, so p(x) = p(z)·|det ∂z/∂x|. Couplings make that determinant free, and training is exact maximum likelihood.']],
  draw(t) {
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1060, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 290 + i * 38, 18, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8)); card(1200, 240, 620, 690, P.sage, ra); label('the whole episode', 1236, 290, ra, P.sage);
    [['sand is conserved', '[p|p]([d|x]) = [p|p]([n|z])·|det ∂[n|z]/∂[d|x]|'], ['coupling', '[d|b]′ = [d|b]·e[a^|s(a)] + [a|t]([d|a])'], ['free determinant', 'log|det| = [kB|Σ] [a|s]'], ['exact likelihood', 'log [p|p]([d|x]) = log N([m|f]([d|x])) + [kB|Σ][k_|k] [a|s][a_|k]'], ['limits', 'no tearing · no compression']].forEach(([nm, m], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.6, 0.8)); text(nm, 1236, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(m, 1236, 412 + i * 112, 20, { a: ai }); });
  } });

// 10 — hook → energy-based models
const Efn = x => 0.12 * x ** 4 - 1.1 * x ** 2 + 0.35 * x + 3.0 + 0.4 * Math.sin(3 * x);
SC.push({ dur: 48, chapter: 'next', title: 'Throw away the shackles?', sub: 'What if any network could define a density?',
  caps: [[1, 8, 'Flows buy exact likelihood with shackles: every layer invertible, every determinant cheap, no tearing allowed.'],
         [8, 15, 'What if we threw the shackles away? Let any network at all output a single number for each x: its energy, E(x).'],
         [15, 23, 'Low energy for things that look like data, high energy for everything else. Any landscape you like — valleys, ridges, islands.'],
         [23, 31, 'Turn energy into probability the way physics does: p(x) = e^−E(x) / Z. Valleys become likely; mountains become rare.'],
         [31, 38, 'Total freedom. Just one tiny problem: Z — the sum of e^−E over every possible x. For images, an integral over 784 dimensions.'],
         [38, 47.6, 'How do you learn, or even sample from, a distribution you cannot normalise? Next episode: energy-based models.']],
  draw(t) {
    const dimAll = 1 - 0.9 * ease(prog(t, 38.5, 1)); GA = dimAll;
    const x0 = 160, base = 640, sx = 150, X = x => x0 + (x + 3.3) * sx, Yv = e => base - 30 - (e + 0.5) * 55, a0 = eout(prog(t, 8, 1));
    // energy landscape
    if (a0 > 0) { setA(a0 * 0.15); ctx.fillStyle = P.plum; ctx.beginPath(); ctx.moveTo(X(-3.3), base); for (let i = 0; i <= 200; i++) { const x = -3.3 + 6.6 * i / 200; ctx.lineTo(X(x), Math.min(base, Yv(Efn(x)))); } ctx.lineTo(X(3.3), base); ctx.fill();
      setA(a0); ctx.strokeStyle = P.plum; ctx.lineWidth = 4; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -3.3 + 6.6 * i / 200; i ? ctx.lineTo(X(x), Math.min(base, Yv(Efn(x)))) : ctx.moveTo(X(x), Yv(Efn(x))); } ctx.stroke(); setA(1);
      text('energy E(x)', X(-0.3), Yv(Efn(-0.3)) - 40, { font: F.mono, size: 20, color: P.plum, a: a0 });
      // balls settling into valleys
      for (let k = 0; k < 6; k++) { const xs = -3 + k * 1.2, tt = clamp((t - 15 - k * 0.3) / 6), xb = lerp(xs, xs < -0.3 ? -2.2 : 2.0, ease(tt)) + 0.05 * Math.sin(t * 3 + k); glow(X(xb), Math.min(base, Yv(Efn(xb))) - 10, 8, P.terracotta, a0 * eout(prog(t, 15, 0.8))); } }
    const pa = eout(prog(t, 23, 1));
    if (pa > 0) { const Z = (() => { let s = 0; for (let x = -3.3; x < 3.3; x += 0.01) s += Math.exp(-Efn(x)) * 0.01; return s; })();
      setA(pa * 0.3); ctx.fillStyle = P.ink; ctx.beginPath(); const pb = 930, pmx = Math.exp(-Efn(-2.2)) / Z;
      ctx.moveTo(X(-3.3), pb); for (let i = 0; i <= 200; i++) { const x = -3.3 + 6.6 * i / 200; ctx.lineTo(X(x), pb - Math.exp(-Efn(x)) / Z / pmx * 190); } ctx.lineTo(X(3.3), pb); ctx.fill(); setA(1);
      line(X(-3.3), pb, X(3.3), pb, P.border, 1.5, pa);
      M('[p|p]([d|x]) = e[m^|−E(x)] / [a|Z]', X(0.2), 790, 24, { a: pa }); }
    const ma = eout(prog(t, 23, 0.8));
    card(1250, 300, 580, 320, P.ink, ma); label('the boltzmann distribution', 1284, 346, ma, P.ink);
    M('[p|p]([d|x]) = e[m^|−E(x)] / [a|Z]', 1284, 420, 38, { align: 'left', a: ma });
    M('[a|Z] = [pB|∫] e[m^|−E(x)] d[d|x]   = ?', 1284, 500, 30, { a: eout(prog(t, 31, 0.8)) });
    text('over all 784-pixel images…', 1284, 570, { font: F.mono, size: 18, color: P.rose, a: eout(prog(t, 32, 0.8)) });
    GA = 1;
    const qa = eout(prog(t, 38.5, 1.2));
    if (qa > 0) {
      text('What if any network', 960, 440, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('could be a density?', 960, 530, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('NEXT  ·  EPISODE 07', 960, 640, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 40, 1)), ls: 4 });
      text('Energy-based models — sculpting a landscape', 960, 695, { font: F.serif, italic: true, size: 42, color: P.stone, align: 'center', a: eout(prog(t, 40.5, 1)) });
    }
  } });
