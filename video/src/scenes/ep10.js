// ─────────────────────────────────────────────────────────────
//  Real flow-matching models (exported by code/export/export_ep10_assets.py)
//  Notation all season: z = noise (rose), x = data (terracotta); today xₜ = (1 − t)·z + t·x runs from noise (t = 0) to data (t = 1).
// ─────────────────────────────────────────────────────────────
const A = window.EP10, WR = 2.3, RG = A.ring, J = A.judges;
const sq = (x0, y0, S, R = WR) => plotSquare(x0, y0, S, R);
const pct = v => (v * 100).toFixed(0) + '%';
// int16 trajectories: S stages × n points × (x, y), /4000
function stages(b, n) { const u = b64bytes(b), v = new Int16Array(u.buffer, u.byteOffset, u.length / 2); return { v, n, S: v.length / (2 * n) }; }
function ptAt(st, k, f) { const S = st.S - 1, fi = clamp(f, 0, S), i = Math.min(S - 1, Math.floor(fi)), u = fi - i, a = (i * st.n + k) * 2, b = ((i + 1) * st.n + k) * 2;
  return [lerp(st.v[a], st.v[b], u) / 4000, lerp(st.v[a + 1], st.v[b + 1], u) / 4000]; }
const P1 = stages(RG.paths1, RG.n), P2 = stages(RG.paths2, RG.n);                 // 600 particles, 61 stages: round 1 (curved) and after reflow
const TF = stages(A.title.fine, A.title.n), TK = 4, TFAST = stages(A.title.fast[TK], A.title.n);
// episode 9's ᾱ as a function of today's clock (t = 0 noise … 1 data), tabulated at 61 points
const AB9 = (() => { const b = i => 1e-4 + (0.02 - 1e-4) * i / 999, c = [1]; for (let i = 0; i < 1000; i++) c.push(c[i] * (1 - b(i))); return Array.from({ length: 61 }, (_, j) => c[Math.round(999 * (1 - j / 60)) + 1]); })();
const EP9_ABAR = s => AB9[clamp(Math.round(s * 60), 0, 60)];
// the learned wind on a 21 × 21 grid at a few times, bilinear in space and linear in time
function windAt(x, y, t) { const ts = RG.fieldTs, g = RG.grid, n = g.length, j = clamp((t - ts[0]) / (ts[ts.length - 1] - ts[0]), 0, 1) * (ts.length - 1);
  let k = 0; while (k < ts.length - 2 && t > ts[k + 1]) k++; const w = clamp((t - ts[k]) / (ts[k + 1] - ts[k]), 0, 1);
  const fx = clamp((x - g[0]) / (g[n - 1] - g[0]), 0, 1) * (n - 1), fy = clamp((g[n - 1] - y) / (g[n - 1] - g[0]), 0, 1) * (n - 1), i0 = Math.min(n - 2, Math.floor(fx)), r0 = Math.min(n - 2, Math.floor(fy)), a = fx - i0, b = fy - r0;
  const at = (F, r, c) => F[r * n + c], out = [0, 0];
  for (const [F, wt] of [[RG.field[k], 1 - w], [RG.field[k + 1], w]]) for (let d = 0; d < 2; d++)
    out[d] += wt * ((at(F, r0, i0)[d] * (1 - a) + at(F, r0, i0 + 1)[d] * a) * (1 - b) + (at(F, r0 + 1, i0)[d] * (1 - a) + at(F, r0 + 1, i0 + 1)[d] * a) * b);
  void j; return out; }
function windField(m, t, a, col = P.sage, step = 1, scale = 0.11) { if (GA * a <= 0.002) return; const g = RG.grid, n = g.length;
  clipTo(m, () => { for (let r = 0; r < n; r += step) for (let c = 0; c < n; c += step) { const x = g[c], y = g[n - 1 - r], v = windAt(x, y, t), L = Math.hypot(v[0], v[1]); if (L < 1e-4) continue;
    const s = scale * Math.min(1, 2.2 / L) * Math.min(L, 2.2) / L; arrow(m.X(x - v[0] * s * 0.3), m.Y(y - v[1] * s * 0.3), m.X(x + v[0] * s), m.Y(y + v[1] * s), col, 1.5, a * clamp(0.35 + L / 3, 0, 0.85)); } }); }
// digits: uint8 images (from [−1, 1]); the season's terracotta-on-dark
const imgs = (b, n) => memo('i' + b.length + b.slice(b.length >> 1, (b.length >> 1) + 96) + b.slice(-48), () => { const u = b64bytes(b); return Array.from({ length: n }, (_, i) => u.subarray(i * 784, (i + 1) * 784)); });
const COLD = hx(P.terracotta);
function digitCanvas(px, key) { return memo('c' + key, () => { const c = document.createElement('canvas'); c.width = c.height = 28; const g = c.getContext('2d'), im = g.createImageData(28, 28);
  for (let i = 0; i < 784; i++) { const v = px[i] / 255; im.data[i * 4] = COLD[0] * v + 28 * (1 - v); im.data[i * 4 + 1] = COLD[1] * v + 24 * (1 - v); im.data[i * 4 + 2] = COLD[2] * v + 22 * (1 - v); im.data[i * 4 + 3] = 255; } g.putImageData(im, 0, 0); return c; }); }
function drawDigit(px, key, x, y, s, a = 1) { if (GA * a <= 0.002) return; setA(a); ctx.imageSmoothingEnabled = false; ctx.drawImage(digitCanvas(px, key), x, y, s, s); ctx.imageSmoothingEnabled = true; setA(1); }
function digitGrid(key, n, x, y, cols, cell, gap, a) { const ims = imgs(A.cmp[key], 16); for (let q = 0; q < n; q++) drawDigit(ims[q], key + q, x + (q % cols) * (cell + gap), y + Math.floor(q / cols) * (cell + gap), cell, a); }
// Euler hops: stage index for a hop-by-hop animation, with a little ease inside each hop
const hopF = (t, t0, per, K) => { const s = (t - t0) / per; if (s <= 0) return 0; if (s >= K) return K; const i = Math.floor(s); return i + ease(s - i); };
// the eight blobs of the ring, for colouring by destination
const BLOB = k => [Math.cos(k * Math.PI / 4), Math.sin(k * Math.PI / 4)];
const BLOBCOL = [P.terracotta, P.amber, P.sage, P.ink, P.plum, P.rose, P.clay, P.chalk];
const nearestBlob = p => { let b = 0, d = 1e9; for (let k = 0; k < 8; k++) { const q = BLOB(k), e = Math.hypot(p[0] - q[0], p[1] - q[1]); if (e < d) { d = e; b = k; } } return b; };
const DEST1 = RG.z.map((_, k) => nearestBlob(ptAt(P1, k, P1.S - 1)));
function roadLine(m, z, x, col, a, lw = 1.2, dash = null) { line(m.X(z[0]), m.Y(z[1]), m.X(x[0]), m.Y(x[1]), col, lw, a, dash); }
const onRoad = (z, x, t) => [(1 - t) * z[0] + t * x[0], (1 - t) * z[1] + t * x[1]];

// ─────────────────────────────────────────────────────────────
//  SCENES
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open: the title flies out of static in four straight hops
const HOP0 = 8.4, HOP = 1.55, BANG = HOP0 + TK * HOP + 0.25;
SC.push({ dur: 34, chapter: 'intro',
  caps: [[1, 7.6, 'Last episode, turning static into a picture took a thousand small steps.'],
         [7.8, 14.2, 'Watch this. Every point knows where it is going, and flies there in a straight line.'],
         [14.4, 20.4, `${['One', 'Two', 'Three', 'Four', 'Five', 'Six'][TK - 1]} steps. ${['One', 'Two', 'Three', 'Four', 'Five', 'Six'][TK - 1]} calls to the network. The whole title.`],
         [26.6, 33.6, 'Real models, trained by hand in numpy. Today: flow matching, the straight way from noise to data.']],
  draw(t) {
    const [sx, sy] = shake(t, BANG, 12); ctx.save(); ctx.translate(sx, sy);
    const f = hopF(t, HOP0, HOP, TK), cx = 960, cy = 470, sc = 300, a0 = eout(prog(t, 0.2, 1.2)), done = clamp(f / TK), hot = t >= BANG ? Math.exp(-(t - BANG) * 1.3) : 0;
    const r = rng(Math.floor(t * 24) + 5), jit = t < HOP0 ? 1 : 0, col = mixc(P.rose, P.chalk, done ** 2);
    // faint straight trails of the hops already taken
    if (t > HOP0 && t < BANG + 3) { const ta = a0 * 0.16 * (1 - prog(t, BANG, 3)); ctx.strokeStyle = P.sage; ctx.lineWidth = 0.8; setA(ta); ctx.beginPath();
      for (let k = 0; k < TFAST.n; k += 7) { const [x0, y0] = ptAt(TFAST, k, 0), [x1, y1] = ptAt(TFAST, k, f); ctx.moveTo(cx + x0 * sc, cy - y0 * sc); ctx.lineTo(cx + x1 * sc, cy - y1 * sc); } ctx.stroke(); setA(1); }
    ctx.fillStyle = col;
    for (let k = 0; k < TFAST.n; k++) { const [x, y] = ptAt(TFAST, k, f), X = cx + (x + jit * (r() - 0.5) * 0.02) * sc, Y = cy - (y + jit * (r() - 0.5) * 0.02) * sc;
      if (X < -10 || X > 1930 || Y < -10 || Y > 1090) continue; setA(a0 * (0.75 + 0.25 * done)); ctx.fillRect(X - 1.4, Y - 1.4, 2.8, 2.8); }
    setA(1);
    if (hot > 0.01) { ctx.save(); ctx.globalCompositeOperation = 'lighter';
      for (const [rr, al] of [[9, 0.05], [5, 0.12], [2.5, 0.32]]) { ctx.fillStyle = rgba(P.terracotta, al * hot);
        for (let k = 0; k < TFAST.n; k += 2) { const [x, y] = ptAt(TFAST, k, TK); ctx.fillRect(cx + x * sc - rr, cy - y * sc - rr, 2 * rr, 2 * rr); } } ctx.restore();
      ring(cx, cy, 200 + (t - BANG) * 1400, P.chalk, hot * 0.35, 3); }
    ctx.restore();
    // the step counter: one pip per network call
    const ca = eout(prog(t, HOP0 - 0.8, 0.6)) * (1 - ease(prog(t, BANG + 1.5, 1)));
    if (ca > 0) { text('network calls', 960, 820, { font: F.mono, size: 18, color: P.dust, align: 'center', a: ca });
      for (let i = 0; i < TK; i++) { const on = t >= HOP0 + i * HOP, x = 960 + (i - (TK - 1) / 2) * 46; ring(x, 856, 12, on ? P.amber : P.border, ca, 2); if (on) dot(x, 856, 7 * pop(prog(t, HOP0 + i * HOP, 0.3)), P.amber, ca); }
      text('last episode: 1,000', 960, 900, { font: F.mono, size: 16, color: P.rose, align: 'center', a: ca * eout(prog(t, HOP0 + 1, 0.8)) * 0.8 }); }
    const a = eout(prog(t, BANG + 0.6, 1.2));
    text('EPISODE 10', 960, 300, { font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 });
    text('the straight way from noise to data', 960, 680, { size: 36, color: P.stone, align: 'center', a: eout(prog(t, BANG + 1.4, 1.2)) });
    const tg = [[P.rose, 'z  noise'], [P.terracotta, 'x  data'], [P.amber, 't  time, 0 → 1'], [P.sage, 'x − z  a road’s velocity'], [P.plum, 'vθ  the learned wind']];
    ctx.font = `400 21px ${F.mono}`; const tw = tg.reduce((acc, [, s]) => acc + ctx.measureText(s).width + 21 * 2.1 + 14, -14);
    let tx = 960 - tw / 2; tg.forEach(([c, s], i) => { tx += tag(tx, 760, c, s, eout(prog(t, BANG + 2.2 + i * 0.3, 0.6)), 21) + 14; });
  } });

// 01 — generation is motion
const CURV = (() => { const out = {}; for (const K of [1, 2, 4, 32]) { let p = [1, 0]; const pts = [p]; for (let k = 0; k < K; k++) { const w = Math.PI / 2 / K; p = [p[0] - w * p[1], p[1] + w * p[0]]; pts.push(p); } out[K] = pts; } return out; })();
SC.push({ dur: 50, chapter: 'motion', title: 'Generation is motion', sub: 'A wind that blows noise into data.',
  caps: [[1, 8, 'Every generator this season is episode 1’s x = g(z): noise in, data out. Today, g is a journey.'],
         [8, 16, 'Picture a wind that changes over time. Drop a noise point in and let it drift from t = 0 to t = 1.'],
         [16, 24, 'The wind is a velocity field, v(x, t). Episode 6 filed this away: dx/dt = v(x, t), a flow of infinitely thin layers.'],
         [24, 32, 'A computer follows it in small steps: read the arrow, move a little along it, repeat. That is Euler’s method.'],
         [32, 40, 'On a straight road one big step lands exactly. On a curved road, a big step flies off the bend.'],
         [40, 49.6, 'Diffusion’s roads were curved, so it needed many steps. What if we could choose the roads?']],
  draw(t) {
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const tw = t < 8 ? 0 : ease(prog(t, 9, 14));                       // the clock of the wind
    windField(m, tw, a0 * eout(prog(t, 8, 1)), P.sage, 2, 0.12);
    const pa = eout(prog(t, 1.5, 0.8));
    clipTo(m, () => { for (let k = 0; k < RG.n; k += 2) { const p = ptAt(P1, k, tw * (P1.S - 1)); marble(m.X(p[0]), m.Y(p[1]), 3, mixc(P.rose, P.terracotta, tw), pa); } });
    text(`t = ${tw.toFixed(2)}`, 110, 925, { font: F.mono, size: 20, color: P.amber, a: a0 });
    text('sage: the wind at this moment', 750, 925, { font: F.mono, size: 16, color: P.dust, align: 'right', a: a0 * eout(prog(t, 8, 1)) });
    const c0 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 130, P.plum, c0); M('[n|z] ∼ N(0, I)   →   [d|x] = [m|g]([n|z])', 864, 322, 30, { a: c0 }); text('episode 1: noise in, data out', 864, 360, { font: F.mono, size: 16, color: P.dust, a: c0 });
    const c1 = eout(prog(t, 16, 0.8)); card(830, 400, 1000, 150, P.sage, c1); label('a flow (episode 6)', 864, 446, c1, P.sage);
    M('d[d|x] / dt = [m|v]([d|x], t)       t: 0 (noise) → 1 (data)', 864, 510, 30, { a: c1 });
    const c2 = eout(prog(t, 24, 0.8)); card(830, 570, 1000, 130, P.amber, c2); label('euler’s method', 864, 616, c2, P.amber);
    M('[d|x] ← [d|x] + Δt · [m|v]([d|x], t)       K steps of size Δt = 1/K', 864, 670, 28, { a: c2 });
    // the curved-road demo: a quarter circle, followed with 1, 2, 4 and 32 Euler steps
    const c3 = eout(prog(t, 32, 0.8));
    if (c3 > 0) { card(830, 720, 1000, 210, P.rose, c3); const O = [950, 908], R = 100, X = p => O[0] + p[0] * R, Y = p => O[1] - p[1] * R;
      setA(c3 * 0.6); ctx.strokeStyle = P.dust; ctx.lineWidth = 2; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.arc(O[0], O[1], R, -Math.PI / 2, 0); ctx.stroke(); ctx.setLineDash([]); setA(1);
      [[1, P.rose, 32.5], [2, P.amber, 33.5], [4, P.sage, 34.5]].forEach(([K, col, t0]) => { const pts = CURV[K], a = c3 * eout(prog(t, t0, 0.6));
        pts.forEach((p, i) => { if (i) arrow(X(pts[i - 1]), Y(pts[i - 1]), X(p), Y(p), col, 2.5, a); }); dot(X(pts[K]), Y(pts[K]), 5, col, a); });
      marble(X([0, 1]), Y([0, 1]), 7, P.terracotta, c3); text('target', X([0, 1]) - 14, Y([0, 1]) + 5, { font: F.mono, size: 15, color: P.terracotta, align: 'right', a: c3 }); marble(X([1, 0]), Y([1, 0]), 5, P.rose, c3);
      M('curved road:  [n|1 step] · [a|2 steps] · [s|4 steps]', 1120, 790, 18, { a: c3 });
      const s2 = eout(prog(t, 36, 0.6)); line(1120, 880, 1760, 820, P.dust, 2, s2, [5, 5]); arrow(1120, 880, 1760, 820, P.sage, 3, s2 * eout(prog(t, 36.6, 0.5)));
      text('straight road: 1 step, exact', 1120, 910, { font: F.mono, size: 17, color: P.sage, a: s2 }); }
  } });

// 02 — the simplest road
const PAIR = { z: [-1.35, -0.95], x: [0.72, 0.68] };
SC.push({ dur: 46, chapter: 'roads', title: 'The simplest road', sub: 'A straight line from a noise point to a data point.',
  caps: [[1, 8, 'The simplest road there is: pick a noise point z and a data point x, and join them with a line.'],
         [8, 16, 'At time t, go a fraction t of the way: xₜ = (1 − t)·z + t·x. Noise at t = 0, data at t = 1.'],
         [16, 24, 'Its velocity is the derivative: x − z. The same arrow at every moment, so the point glides at constant speed.'],
         [24, 32, 'If every noise point knew its data point, we could learn these arrows, and one step would reach anywhere.'],
         [32, 40, 'But nothing pairs them. So we pair them at random: any noise point with any data point.'],
         [40, 45.6, 'Six hundred random pairs, six hundred straight roads. And they cross everywhere.']],
  draw(t) {
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    clipTo(m, () => RG.x1.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.2, P.terracotta, a0 * 0.35)));
    const oneA = a0 * (1 - ease(prog(t, 32, 1.5)));
    if (oneA > 0) { const tt = t < 8 ? 0 : 0.5 - 0.5 * Math.cos(Math.PI * clamp((t - 9) / 6)) * (t < 24 ? 1 : 1), la = eout(prog(t, 3, 1));
      roadLine(m, PAIR.z, PAIR.x, P.stone, oneA * la, 2, [7, 6]);
      marble(m.X(PAIR.z[0]), m.Y(PAIR.z[1]), 9, P.rose, oneA); text('z', m.X(PAIR.z[0]) - 26, m.Y(PAIR.z[1]) + 8, { font: F.mono, size: 26, color: P.rose, a: oneA });
      marble(m.X(PAIR.x[0]), m.Y(PAIR.x[1]), 9, P.terracotta, oneA); text('x', m.X(PAIR.x[0]) + 16, m.Y(PAIR.x[1]) - 12, { font: F.mono, size: 26, color: P.terracotta, a: oneA });
      if (t > 8) { const tp = t < 24 ? tt : (t - 24) % 4 / 4, p = onRoad(PAIR.z, PAIR.x, tp), pa = oneA * eout(prog(t, 8, 0.6)); marble(m.X(p[0]), m.Y(p[1]), 8, P.amber, pa);
        text(`xₜ   t = ${tp.toFixed(2)}`, m.X(p[0]) + 16, m.Y(p[1]) + 30, { font: F.mono, size: 18, color: P.amber, a: pa });
        const va = oneA * eout(prog(t, 16, 0.8)); if (va > 0) { const u = [PAIR.x[0] - PAIR.z[0], PAIR.x[1] - PAIR.z[1]]; arrow(m.X(p[0]), m.Y(p[1]), m.X(p[0] + u[0] * 0.3), m.Y(p[1] + u[1] * 0.3), P.sage, 4, va);
          text('x − z', m.X(p[0] + u[0] * 0.3) + 10, m.Y(p[1] + u[1] * 0.3) - 12, { font: F.mono, size: 20, color: P.sage, a: va }); } } }
    // six hundred random roads
    const ma = eout(prog(t, 32.5, 1.2));
    if (ma > 0) clipTo(m, () => RG.x0.forEach((z, k) => { const u = eout(prog(t, 33 + (k % 60) * 0.05, 0.9)); if (u <= 0) return; const x = RG.x1[k];
      line(m.X(z[0]), m.Y(z[1]), m.X(lerp(z[0], x[0], u)), m.Y(lerp(z[1], x[1], u)), mixc(P.rose, P.terracotta, 0.5), 0.9, ma * 0.33); if (u > 0.98) dot(m.X(x[0]), m.Y(x[1]), 2.2, P.terracotta, ma * 0.8); }));
    const c1 = eout(prog(t, 8, 0.8)); card(830, 250, 1000, 180, P.amber, c1); label('a point on the road', 864, 296, c1, P.amber);
    M('[d|x][d_|t] = (1 − [a|t])·[n|z] + [a|t]·[d|x]', 864, 370, 40, { a: c1 }); text('t = 0: the noise · t = 1: the data', 864, 414, { font: F.mono, size: 16, color: P.dust, a: c1 });
    const c2 = eout(prog(t, 16, 0.8)); card(830, 450, 1000, 160, P.sage, c2); label('its velocity', 864, 496, c2, P.sage);
    M('d[d|x][d_|t] / dt = [d|x] − [n|z]', 864, 556, 34, { a: c2 }); text('constant: a straight road at steady speed', 864, 594, { font: F.mono, size: 16, color: P.dust, a: c2 });
    const c3 = eout(prog(t, 24, 0.8)) * (1 - ease(prog(t, 31.5, 1))); if (c3 > 0) { card(830, 630, 1000, 150, P.plum, c3); para('If each z knew its x, learning the arrows would be easy — and one Euler step would be exact.', 864, 684, 930, { size: 26, a: c3 }); }
    const c4 = eout(prog(t, 32, 0.8)); if (c4 > 0) { card(830, 630, 1000, 300, P.rose, c4); label('but who pairs them?', 864, 676, c4, P.rose);
      M('[n|z] ∼ N(0, I)   and   [d|x] ∼ data,   independently', 864, 740, 28, { a: c4 }); para('Nothing tells us which noise point belongs to which data point. So we pair them at random — and the roads cross.', 864, 800, 930, { size: 25, a: eout(prog(t, 34, 0.8)) }); }
  } });

// 03 — when roads cross: the fan, and its average
const FAN = RG.fan;
SC.push({ dur: 54, chapter: 'crossings', title: 'When roads cross', sub: 'One place, one moment, many directions.',
  caps: [[1, 8, 'Slide all six hundred points along their roads together. Halfway there, many roads pass through the same spot.'],
         [8, 16, 'Stand at one spot at t = ½. The roads through it head off in many different directions.'],
         [16, 24, 'But a network is told only where it is and when: vθ(x, t). It can give only one arrow here. Which one?'],
         [24, 32, 'Train it with a plain regression: make vθ(xₜ, t) close to x − z, over random pairs and random times.'],
         [32, 40, 'A regression’s best answer to a question with many right answers is their average. Episode 8’s denoiser was the same trick.'],
         [40, 47, 'The average of these arrows, and what our trained network says here: nearly the same arrow.'],
         [47, 53.6, 'So the network learns an averaged wind: v(x, t) = E[ x − z | xₜ = x ].']],
  draw(t) {
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)), zoom = ease(prog(t, 8, 2.5)); frame(m, a0);
    const tt = t < 8 ? 0.5 * ease(prog(t, 1, 6)) : 0.5, F0 = FAN.x;
    clipTo(m, () => { const fa = a0 * (1 - 0.7 * zoom);
      RG.x0.forEach((z, k) => { const x = RG.x1[k]; if (k % 3 === 0) roadLine(m, z, x, mixc(P.rose, P.terracotta, 0.5), fa * 0.18, 0.8); const p = onRoad(z, x, tt); dot(m.X(p[0]), m.Y(p[1]), 2.5, mixc(P.rose, P.terracotta, tt), fa * 0.85); });
      if (zoom > 0) { FAN.x0.forEach((z, i) => { const x = FAN.x1[i]; roadLine(m, z, x, P.stone, zoom * a0 * 0.45, 1.2); }); } });
    if (zoom > 0) { const X = m.X(F0[0]), Y = m.Y(F0[1]); ring(X, Y, 0.06 * m.s * 1.0 + 10, P.chalk, zoom * a0, 2); marble(X, Y, 7, P.amber, zoom * a0);
      text(`t = ½`, X + 16, Y - 16, { font: F.mono, size: 18, color: P.amber, a: zoom * a0 }); }
    // the fan, magnified on the right
    const fa = eout(prog(t, 9, 1)); if (fa > 0) { const C = [1400, 590], R = 250; card(1030, 250, 800, 640, P.sage, fa); label('the fan at one spot', 1066, 296, fa, P.sage);
      const sc = 90; FAN.u.forEach((u, i) => { const ai = fa * eout(prog(t, 9.5 + i * 0.04, 0.4)) * (1 - 0.55 * ease(prog(t, 33, 1)));
        arrow(C[0], C[1], C[0] + u[0] * sc, C[1] - u[1] * sc, mixc(P.stone, P.sage, 0.3), 1.6, ai * 0.7); });
      marble(C[0], C[1], 9, P.amber, fa);
      const qa = eout(prog(t, 16, 0.8)) * (1 - ease(prog(t, 32, 0.8))); if (qa > 0) text('?', C[0] + 30, C[1] - 30, { font: F.serif, italic: true, size: 64, color: P.chalk, a: qa * (0.6 + 0.4 * Math.sin(t * 4)) });
      const Z = 3, av = eout(prog(t, 34, 1)); if (av > 0) arrow(C[0], C[1], C[0] + FAN.avg[0] * sc * Z * av, C[1] - FAN.avg[1] * sc * Z * av, P.sage, 6, av);
      const nv = eout(prog(t, 41, 0.8)); if (nv > 0) { arrow(C[0], C[1], C[0] + FAN.net[0] * sc * Z, C[1] - FAN.net[1] * sc * Z, P.plum, 3, nv); }
      if (av > 0) { text('sage: the average of the fan', 1540, 790, { font: F.mono, size: 17, color: P.sage, a: av }); text('(both drawn 3× longer)', 1540, 850, { font: F.mono, size: 15, color: P.dust, a: av }); }
      if (nv > 0) text(`plum: the network, vθ(x, ½)`, 1540, 818, { font: F.mono, size: 17, color: P.plum, a: nv });
      const ca = eout(prog(t, 24, 0.8)); if (ca > 0) { M('min[a_|θ]  [kB|E] ‖ [m|v][m_|θ]([d|x][d_|t], t) − ([d|x] − [n|z]) ‖²', 1066, 360, 26, { a: ca * (1 - ease(prog(t, 47, 0.8))) }); }
      const ra = eout(prog(t, 47, 0.8)); if (ra > 0) M('[m|v]([d|x], t) = [kB|E][ [d|x] − [n|z] | [d|x][d_|t] = [d|x] ]', 1066, 360, 28, { a: ra }); }
  } });

// 04 — the averaged wind still delivers the data
const HB = 24, hist = (pts, lo = -2.3, hi = 2.3) => { const h = new Array(HB).fill(0); pts.forEach(p => { const i = Math.floor((p[0] - lo) / (hi - lo) * HB); if (i >= 0 && i < HB) h[i]++; }); return h.map(v => v / pts.length); };
const TRAILK = Array.from({ length: 36 }, (_, i) => i * 16 + 3);
const CROSSES = memo('crosses', () => { const out = []; for (let i = 0; i < TRAILK.length; i++) for (let j = i + 1; j < TRAILK.length; j++) {
  const a = RG.x0[TRAILK[i]], b = RG.x1[TRAILK[i]], c = RG.x0[TRAILK[j]], d = RG.x1[TRAILK[j]], den = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0]); if (Math.abs(den) < 1e-9) continue;
  const u = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / den, v = ((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / den;
  if (u > 0 && u < 1 && v > 0 && v < 1) out.push([a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])]); } return out; });
SC.push({ dur: 52, chapter: 'the wind', title: 'The averaged wind still delivers', sub: 'Different journeys, the same crowd at every moment.',
  caps: [[1, 8, 'Now follow the averaged wind instead. Left: points on their random straight roads. Right: points riding the learned wind.'],
         [8, 16, 'At every moment the two clouds match: same shape, same spread. Only the individual journeys differ.'],
         [16, 24, 'Why? Think of sand, as in episode 6. At every spot, the averaged wind moves exactly as much sand as the crossing roads did.'],
         [24, 32, 'The same sand through every doorway means the same cloud at every moment, from noise at t = 0 to data at t = 1.'],
         [32, 40, 'And the wind’s roads never cross. At any place and time there is only one arrow, so two points can never swap.'],
         [40, 47, 'They bend: early on every road heads for the middle, then turns off. Remember that bend.'],
         [47, 51.6, 'This is flow matching: learn the average of simple roads.']],
  draw(t) {
    const L = sq(110, 290, 560), Rm = sq(1000, 290, 560), a0 = eout(prog(t, 0.6, 0.8));
    const tt = t < 32 ? (t < 2 ? 0 : (1 - Math.cos(Math.PI * clamp((t - 2) / 12 % 2 > 1 ? 2 - (t - 2) / 12 % 2 : (t - 2) / 12 % 2))) / 2) : ease(prog(t, 32.5, 7));
    frame(L, a0); frame(Rm, a0);
    const left = RG.x0.map((z, k) => onRoad(z, RG.x1[k], tt)), right = RG.z.map((_, k) => ptAt(P1, k, tt * (P1.S - 1)));
    const trails = t > 32 ? eout(prog(t, 32, 1)) : 0;
    if (trails > 0) { clipTo(L, () => TRAILK.forEach(k => { const z = RG.x0[k], x = onRoad(z, RG.x1[k], tt); line(L.X(z[0]), L.Y(z[1]), L.X(x[0]), L.Y(x[1]), P.stone, 1.3, trails * 0.6); }));
      clipTo(Rm, () => TRAILK.forEach(k => { setA(trails * 0.7); ctx.strokeStyle = P.sage; ctx.lineWidth = 1.5; ctx.beginPath(); const n = Math.round(tt * (P1.S - 1));
        for (let s = 0; s <= n; s++) { const p = ptAt(P1, k, s); s ? ctx.lineTo(Rm.X(p[0]), Rm.Y(p[1])) : ctx.moveTo(Rm.X(p[0]), Rm.Y(p[1])); } ctx.stroke(); setA(1); }));
      const xa = eout(prog(t, 36, 0.8)); if (xa > 0 && tt > 0.95) clipTo(L, () => CROSSES.forEach(p => { line(L.X(p[0]) - 5, L.Y(p[1]) - 5, L.X(p[0]) + 5, L.Y(p[1]) + 5, P.rose, 2, xa); line(L.X(p[0]) - 5, L.Y(p[1]) + 5, L.X(p[0]) + 5, L.Y(p[1]) - 5, P.rose, 2, xa); })); }
    const col = mixc(P.rose, P.terracotta, tt);
    clipTo(L, () => left.forEach(p => dot(L.X(p[0]), L.Y(p[1]), 2.6, col, a0 * 0.9))); clipTo(Rm, () => right.forEach(p => dot(Rm.X(p[0]), Rm.Y(p[1]), 2.6, col, a0 * 0.9)));
    text('random straight roads', 110, 276, { font: F.mono, size: 18, color: P.stone, a: a0 }); text('riding the learned wind', 1000, 276, { font: F.mono, size: 18, color: P.sage, a: a0 });
    text(`t = ${tt.toFixed(2)}`, 960, 560, { font: F.mono, size: 28, color: P.amber, align: 'center', a: a0 });
    // the two crowds, measured: a histogram of positions across each panel
    const ha = eout(prog(t, 8, 1)); if (ha > 0) { const hl = hist(left), hr = hist(right), mx = 0.16;
      [[L, hl, P.stone], [Rm, hr, P.sage]].forEach(([m, h, c]) => h.forEach((v, i) => { const w = m.S / HB, bh = Math.min(1, v / mx) * 70; setA(ha * 0.8); ctx.fillStyle = c; ctx.fillRect(m.x0 + i * w + 1, 930 - bh, w - 2, bh); setA(1); }));
      text('← crowd across each panel →', 835, 925, { font: F.mono, size: 13, color: P.dust, align: 'center', a: ha }); }
    if (trails > 0) { text(`${CROSSES.length} crossings`, L.x0 + L.S, 276, { font: F.mono, size: 18, color: P.rose, align: 'right', a: eout(prog(t, 36, 0.8)) * (tt > 0.95 ? 1 : 0) });
      text('0 crossings', Rm.x0 + Rm.S, 276, { font: F.mono, size: 18, color: P.sage, align: 'right', a: eout(prog(t, 36, 0.8)) * (tt > 0.95 ? 1 : 0) }); }
    const c1 = eout(prog(t, 16, 0.8)) * (1 - ease(prog(t, 31.5, 0.8))); if (c1 > 0) { card(700, 610, 280, 250, P.ink, c1); label('sand (ep 6)', 722, 650, c1, P.ink);
      para('same flux through every doorway → same cloud at every t', 722, 700, 240, { size: 20, a: c1 }); }
  } });

// 05 — the training loop
const EX_T = [0.2, 0.6, 0.8, 0.4, 0.95, 0.0], EXI = [11, 207, 404, 90, 333, 520];
SC.push({ dur: 44, chapter: 'the loss', title: 'One regression', sub: 'A random road, a random moment, its velocity.',
  caps: [[1, 8, 'Here is the whole training loop. Take a real data point x and a fresh noise point z. Pick a random time t.'],
         [8, 16, 'Stand at xₜ = (1 − t)·z + t·x. The road’s velocity there is simply x − z. That is the target.'],
         [16, 24, 'Ask the network for its arrow, vθ(xₜ, t), and score the squared miss. Step the knobs. Repeat.'],
         [24, 32, 'Each target is one random road, but the misses average out: the leftover term has zero mean at every spot.'],
         [32, 40, 'So training on single random roads finds exactly the averaged wind. No simulation, no noise schedule, no chain of steps.'],
         [40, 43.6, 'Lipman, Liu, Albergo and colleagues, 2022.']],
  draw(t) {
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    clipTo(m, () => RG.x1.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2.2, P.terracotta, a0 * 0.3)));
    const cyc = Math.floor(clamp(t - 1, 0, 60) / 6.5) % EX_T.length, u = (clamp(t - 1, 0, 60) % 6.5) / 6.5, k = EXI[cyc], z = RG.x0[k], x = RG.x1[k], tt = EX_T[cyc];
    const ra = a0 * eout(clamp(u * 5)) * (1 - ease(clamp((u - 0.9) * 10)));
    roadLine(m, z, x, P.stone, ra * 0.8, 1.6, [6, 5]); marble(m.X(z[0]), m.Y(z[1]), 8, P.rose, ra); marble(m.X(x[0]), m.Y(x[1]), 8, P.terracotta, ra);
    const p = onRoad(z, x, tt * ease(clamp((u - 0.15) * 4))); marble(m.X(p[0]), m.Y(p[1]), 8, P.amber, ra * eout(clamp((u - 0.12) * 5)));
    const ua = ra * eout(clamp((u - 0.4) * 5)), sc = 0.33, U = [x[0] - z[0], x[1] - z[1]], V = windAt(p[0], p[1], tt), va = ra * eout(clamp((u - 0.6) * 5));
    arrow(m.X(p[0]), m.Y(p[1]), m.X(p[0] + U[0] * sc), m.Y(p[1] + U[1] * sc), P.sage, 3.5, ua); arrow(m.X(p[0]), m.Y(p[1]), m.X(p[0] + V[0] * sc), m.Y(p[1] + V[1] * sc), P.plum, 3.5, va);
    if (va > 0) line(m.X(p[0] + U[0] * sc), m.Y(p[1] + U[1] * sc), m.X(p[0] + V[0] * sc), m.Y(p[1] + V[1] * sc), P.rose, 2, va, [3, 3]);
    text(`t = ${tt.toFixed(2)}`, 110, 925, { font: F.mono, size: 20, color: P.amber, a: a0 }); text('sage: x − z · plum: vθ · rose: the miss', 750, 925, { font: F.mono, size: 16, color: P.dust, align: 'right', a: a0 });
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 120, P.amber, c1);
    M('[d|x] ∼ data,   [n|z] ∼ N(0, I),   [a|t] ∼ U(0, 1),   [d|x][d_|t] = (1 − [a|t])[n|z] + [a|t][d|x]', 864, 322, 25, { a: c1 });
    const c2 = eout(prog(t, 16, 0.8)); card(830, 390, 1000, 150, P.plum, c2); label('the loss', 864, 436, c2, P.plum);
    M('[m|L](θ) = [kB|E] ‖ [m|v][m_|θ]([d|x][d_|t], t) − ([d|x] − [n|z]) ‖²', 864, 500, 34, { a: c2 });
    const c3 = eout(prog(t, 24, 0.8)); card(830, 560, 1000, 200, P.sage, c3); label('why one random road is enough', 864, 606, c3, P.sage);
    M('‖[m|v] − [s|u]‖² = ‖[m|v] − ū‖² + 2([m|v] − ū)·(ū − [s|u]) + ‖ū − [s|u]‖²', 864, 664, 24, { a: c3 });
    M('ū = [kB|E][[s|u] | [d|x][d_|t]]  →  middle term averages to 0  →  same best [m|v]', 864, 716, 22, { a: eout(prog(t, 27, 0.8)) });
    const c4 = eout(prog(t, 32, 0.8)); card(830, 780, 1000, 150, P.rose, c4); para('no simulation during training · no noise schedule · a straight road you choose yourself', 864, 836, 930, { size: 25, a: c4 });
  } });

// 06 — training it for real
SC.push({ dur: 44, chapter: 'training', title: 'Training it for real', sub: 'Eight blobs, one network, twelve thousand steps.',
  caps: [[1, 8, 'Now for real: the eight blobs from the last five episodes. One network, three layers of 256, twelve thousand steps.'],
         [8, 16, 'The learned wind at t = 0: every arrow points at the middle, the average of the data. No blob is chosen yet.'],
         [16, 24, 'As time runs on it sharpens: arrows converge on the eight blobs, and the gaps between them go quiet.'],
         [24, 32, 'The loss never reaches zero, and it shouldn’t: the random roads really do disagree. Its floor is the width of the fan.'],
         [32, 40, 'Drop six hundred noise points in and follow the wind to t = 1. Every blob gets its share.'],
         [40, 43.6, 'Now: how few steps can we take?']],
  draw(t) {
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const tw = t < 8 ? 0 : t < 32 ? 0.95 * ease(prog(t, 9, 20)) : 0; windField(m, tw, a0 * (1 - ease(prog(t, 31, 1))), P.sage, 1, 0.1);
    clipTo(m, () => RG.x1.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 2, P.terracotta, a0 * 0.25 * (1 - ease(prog(t, 31, 1))))));
    const fa = eout(prog(t, 32, 0.8)); if (fa > 0) { const f = (P1.S - 1) * ease(prog(t, 32.5, 6));
      clipTo(m, () => RG.z.forEach((_, k) => { const p = ptAt(P1, k, f); marble(m.X(p[0]), m.Y(p[1]), 3, f > P1.S - 2 ? BLOBCOL[DEST1[k]] : mixc(P.rose, P.terracotta, f / (P1.S - 1)), fa); })); }
    text(t < 32 ? `the wind at t = ${tw.toFixed(2)}` : 'following the wind, 60 small steps', 110, 925, { font: F.mono, size: 20, color: t < 32 ? P.sage : P.chalk, a: a0 });
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 150, P.plum, c1); label('the network', 864, 296, c1, P.plum);
    M('([d|x][d_|t], [a|t])  →  3 × 256 swish, t in every layer  →  [m|v][m_|θ]', 864, 356, 26, { a: c1 }); text('nn.TimeMLP, as in episode 9 · backprop by hand', 864, 388, { font: F.mono, size: 15, color: P.dust, a: c1 });
    const ha = eout(prog(t, 24, 0.8)); if (ha > 0) { const H = RG.hist, x0 = 880, y0 = 640, w = 900, h = 150, lo = Math.min(...H), hi = Math.max(...H.slice(2));
      card(830, 420, 1000, 260, P.ink, ha); text('training loss (real run)', x0, 460, { font: F.mono, size: 15, color: P.dust, a: ha });
      setA(ha); ctx.strokeStyle = P.plum; ctx.lineWidth = 2.5; ctx.beginPath(); H.forEach((v, i) => { const X = x0 + i / (H.length - 1) * w, Y = y0 - clamp((v - lo) / (hi - lo)) * h; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke(); setA(1);
      text(`levels off near ${H[H.length - 1].toFixed(2)}: the width of the fan, not a failure`, x0, 670 - 0, { font: F.mono, size: 15, color: P.sage, a: eout(prog(t, 26, 0.8)) }); }
    const sa = eout(prog(t, 38.6, 0.8)); if (sa > 0) { const cnt = new Array(8).fill(0); DEST1.forEach(b => cnt[b]++); card(830, 700, 1000, 230, P.sage, sa); label('share of points per blob (data: 12.5% each)', 864, 746, sa, P.sage);
      cnt.forEach((c, b) => { const x = 880 + b * 116, hh = c / RG.n / 0.2 * 120, ai = sa * eout(prog(t, 38.8 + b * 0.1, 0.4)); setA(ai * 0.9); ctx.fillStyle = BLOBCOL[b]; ctx.fillRect(x, 900 - hh, 70, hh); setA(1);
        text(pct(c / RG.n), x + 35, 900 - hh - 10, { font: F.mono, size: 15, color: P.stone, align: 'center', a: ai }); }); }
  } });

// 07 — the wow: 64 digits flow out of noise
const DG = A.digits, NDG = DG.x.length - 1, fDg = t => NDG * ease(prog(t, 4, 12));
SC.push({ dur: 42, chapter: 'digits', title: 'Sixty-four digits, thirty-two steps', sub: 'The same recipe in 784 dimensions.',
  caps: [[1, 8, 'Sixty-four squares of static. The same recipe, now on handwritten digits: 784 numbers each.'],
         [8, 16, 'Thirty-two Euler steps. Last episode’s sampler took a thousand.'],
         [17, 25, 'Here is the network’s guess of the finished digit at each step, x̂ = xₜ + (1 − t)·vθ: where the wind is heading.'],
         [25, 33, 'Early on the guess is the average of all digits. Then it commits to a digit, and the details come last.'],
         [33, 41.6, 'Composition first, detail last, exactly as in diffusion. The roads are different; the destination is the same.']],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), f = fDg(t), i = Math.min(NDG - 1, Math.floor(f)), u = f - i, sm = ease(prog(t, 16.5, 1.2));
    const gx = lerp(632, 110, sm), cell = lerp(78, 76, sm), pad = 4, gy = 250;
    for (let k = 0; k < 64; k++) { const x = gx + (k % 8) * (cell + pad), y = gy + Math.floor(k / 8) * (cell + pad), ims0 = imgs(DG.x[i], 64), ims1 = imgs(DG.x[i + 1], 64);
      drawDigit(ims0[k], `g${i}_${k}`, x, y, cell, a0 * (1 - u)); if (u > 0) drawDigit(ims1[k], `g${i + 1}_${k}`, x, y, cell, a0 * u); }
    text(`step ${Math.round(f)} / ${NDG}   ·   t = ${DG.ts[Math.min(NDG, Math.round(f))].toFixed(2)}`, gx, gy + 8 * (cell + pad) + 24, { font: F.mono, size: 19, color: P.amber, a: a0 });
    const g2 = t > 16 ? Math.exp(-(t - 16) * 1.5) : 0; if (g2 > 0.01) { ctx.save(); ctx.globalCompositeOperation = 'lighter'; setA(g2 * 0.3); ctx.fillStyle = P.terracotta; ctx.fillRect(gx - 10, gy - 10, 8 * (cell + pad) + 16, 8 * (cell + pad) + 16); ctx.restore(); setA(1); }
    const wa = eout(prog(t, 17, 1)); if (wa > 0) { const cols = [0, 4, 8, 12, 16, 20, 26, NDG], x0 = 790, y0 = 280, c = 112;
      text('x̂ at step →', x0, 262, { font: F.mono, size: 15, color: P.sage, a: wa });
      cols.forEach((s, j) => text(s === NDG ? 'final' : `${s}`, x0 + j * (c + 14) + c / 2, y0, { font: F.mono, size: 13, color: P.dust, align: 'center', a: wa }));
      for (let r = 0; r < 5; r++) cols.forEach((s, j) => drawDigit(imgs(DG.x1[s], 8)[r], `h${s}_${r}`, x0 + j * (c + 14), y0 + 14 + r * (c + 12), c, wa * eout(prog(t, 17.3 + j * 0.25 + r * 0.08, 0.5)))); }
  } });

// 08 — how few steps?
const SK = [1, 2, 4, 8], FIDK = ['1', '2', '4', '8', '16', '32', '100'];
SC.push({ dur: 54, chapter: 'steps', title: 'How few steps?', sub: 'Euler with big strides, and episode 4’s judge.',
  caps: [[1, 8, 'Big strides now. With one step, every point jumps along its first arrow, and lands in the middle: on the average of everything.'],
         [8, 16, 'Two steps, four, eight: the blobs appear fast. The straighter the roads, the fewer steps you need.'],
         [16, 24, 'On digits: one step gives a blur, the mean digit. By eight steps they are readable.'],
         [24, 32, 'Episode 9’s diffusion model with the same budget of network calls, for comparison.'],
         [32, 40, 'Episode 4’s judge, FID, against the number of network calls. Lower is better.'],
         [40, 48, `With few calls the straight roads win: ${J.fm2.FID.toFixed(0)} against ${J.dd2.FID.toFixed(0)} at two, ${J.fm4.FID.toFixed(0)} against ${J.dd4.FID.toFixed(0)} at four. From eight on, it is neck and neck.`],
         [48, 53.6, `But one step is still a blur, FID ${J.fm1.FID.toFixed(0)}. The roads are not quite straight yet.`]],
  draw(t) {
    const a0 = eout(prog(t, 0.6, 0.8)), A1 = 1 - ease(prog(t, 15.5, 1));
    if (A1 > 0) { GA = A1; SK.forEach((K, j) => { const m = sq(110 + j * 440, 300, 400), st = RG.steps[`r1_${K}`].stages, a = a0 * eout(prog(t, 1 + j * 0.6, 0.6)), f = hopF(t, 2 + j * 0.6, 4.5 / K, K);
        frame(m, a); clipTo(m, () => { RG.x1.forEach(p => dot(m.X(p[0]), m.Y(p[1]), 1.6, P.terracotta, a * 0.2));
          st[0].forEach((_, k) => { if (k % 2) return; const i = Math.min(K - 1, Math.floor(f)), uu = f - i, p0 = st[i][k], p1 = st[i + 1][k], p = [lerp(p0[0], p1[0], uu), lerp(p0[1], p1[1], uu)];
            if (k % 10 === 0) { setA(a * 0.35); ctx.strokeStyle = P.sage; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.X(st[0][k][0]), m.Y(st[0][k][1])); for (let s = 1; s <= i; s++) ctx.lineTo(m.X(st[s][k][0]), m.Y(st[s][k][1])); ctx.lineTo(m.X(p[0]), m.Y(p[1])); ctx.stroke(); setA(1); }
            dot(m.X(p[0]), m.Y(p[1]), 2.3, mixc(P.rose, P.terracotta, f / K), a); }); });
        text(`${K} step${K > 1 ? 's' : ''}`, m.x0, 740, { font: F.mono, size: 22, color: P.chalk, a }); text(`radius ${RG.steps[`r1_${K}`].radius.toFixed(2)}  (data 2.00)`, m.x0, 772, { font: F.mono, size: 15, color: P.dust, a }); }); GA = 1; }
    const B1 = eout(prog(t, 16, 0.8)) * (1 - ease(prog(t, 31.5, 1)));
    if (B1 > 0) { GA = B1; [['fm1', 'flow matching · 1'], ['fm2', '2'], ['fm4', '4'], ['fm8', '8'], ['fm32', '32']].forEach(([k, nm], j) => { const x = 110 + j * 350; digitGrid(k, 16, x, 300, 4, 76, 4, eout(prog(t, 16.3 + j * 0.4, 0.5))); text(nm, x, 650, { font: F.mono, size: 18, color: P.terracotta, a: eout(prog(t, 16.3 + j * 0.4, 0.5)) }); });
      [['dd2', 'DDIM (episode 9) · 2'], ['dd4', '4'], ['dd8', '8'], ['dd32', '32']].forEach(([k, nm], j) => { const x = 110 + (j + 1) * 350, a = eout(prog(t, 24 + j * 0.4, 0.5)); digitGrid(k, 8, x, 690, 4, 76, 4, a); text(nm, x, 880, { font: F.mono, size: 18, color: P.rose, a }); });
      GA = 1; }
    const C = eout(prog(t, 32, 0.9));
    if (C > 0) { const x0 = 300, y0 = 860, w = 1300, h = 520, X = k => x0 + Math.log2(k) / Math.log2(100) * w, top = 300, Y = v => y0 - clamp(1 - Math.log(v) / Math.log(top)) * h;
      line(x0, y0, x0 + w, y0, P.borderLight, 1.5, C); line(x0, y0, x0, y0 - h, P.borderLight, 1.5, C);
      [1, 2, 4, 8, 16, 32, 100].forEach(k => text(String(k), X(k), y0 + 30, { font: F.mono, size: 16, color: P.dust, align: 'center', a: C }));
      text('network calls per sample', x0 + w, y0 + 60, { font: F.mono, size: 16, color: P.dust, align: 'right', a: C }); text('FID ↓ (log scale)', x0 - 20, y0 - h - 20, { font: F.mono, size: 16, color: P.dust, a: C });
      [2, 5, 10, 30, 100, 300].forEach(v => { line(x0 - 6, Y(v), x0, Y(v), P.borderLight, 1.5, C); text(String(v), x0 - 12, Y(v) + 5, { font: F.mono, size: 14, color: P.dust, align: 'right', a: C }); });
      let LY = null; const curve = (keys, col, t0, nm) => { const pts = keys.filter(k => J[k]).map(k => [X(+k.replace(/\D/g, '')), Y(J[k].FID)]), g = eout(prog(t, t0, 2)); if (!pts.length) return;
        setA(C * g); ctx.strokeStyle = col; ctx.lineWidth = 3.5; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke(); setA(1);
        pts.forEach(p => dot(p[0], p[1], 5, col, C * g)); const e = pts[pts.length - 1]; let ly = e[1] + 6; if (LY != null && Math.abs(ly - LY) < 24) ly = ly < LY ? LY - 24 : LY + 24; LY = ly; text(nm, e[0] + 14, ly, { font: F.mono, size: 17, color: col, a: C * g }); };
      curve(['dd2', 'dd4', 'dd8', 'dd16', 'dd32', 'dd100'], P.rose, 33.5, 'DDIM (ep 9)'); curve(FIDK.map(k => 'fm' + k), P.terracotta, 35, 'flow matching');
      const rl = Y(J.real.FID); line(x0, rl, x0 + w, rl, P.sage, 1.5, C * 0.7, [6, 6]); text('real data', x0 + w, rl - 10, { font: F.mono, size: 14, color: P.sage, align: 'right', a: C }); }
  } });

// 09 — reflow: straighten the roads
const CP = RG.couple;
SC.push({ dur: 56, chapter: 'reflow', title: 'Straighten the roads', sub: 'Reflow: pair each noise point with where it actually lands.',
  caps: [[1, 8, 'The roads bend because random pairs cross. So find better pairs: run the trained wind, and note where each noise point lands.'],
         [8, 16, 'These new pairs never cross: they came from a flow, and flow lines cannot cross. Now train again, on these pairs.'],
         [16, 24, 'Straight roads between pairs that never cross leave nothing to average. The new wind blows in straight lines.'],
         [24, 32, `Measured as straight distance over distance travelled: ${RG.ratio1.toFixed(3)} before, ${RG.ratio2.toFixed(3)} after. Almost perfectly straight.`],
         [32, 40, 'And now a single step: the whole ring, from one call to the network. Before, one step fell into the middle.'],
         [40, 48, `On digits, one step goes from FID ${J.fm1.FID.toFixed(0)} to ${J.rf1.FID.toFixed(0)}. The price: with many steps, reflow levels off near ${J.rf16.FID.toFixed(0)}.`],
         [48, 55.6, 'This is reflow, or rectified flow. Repeat it, or distil it, and generators take one step or a few.']],
  draw(t) {
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0 * (1 - ease(prog(t, 31.2, 0.8))));
    const ca = eout(prog(t, 1.5, 1)) * (1 - ease(prog(t, 15.5, 1)));
    if (ca > 0) clipTo(m, () => CP.z.forEach((z, k) => { const x = CP.x[k], u = eout(prog(t, 2 + k * 0.03, 1)); dot(m.X(z[0]), m.Y(z[1]), 2.6, P.rose, ca); if (u <= 0) return;
      line(m.X(z[0]), m.Y(z[1]), m.X(lerp(z[0], x[0], u)), m.Y(lerp(z[1], x[1], u)), P.sage, 1.2, ca * 0.6); if (u > 0.98) dot(m.X(x[0]), m.Y(x[1]), 2.6, P.terracotta, ca); }));
    const pa = eout(prog(t, 16, 1)) * (1 - ease(prog(t, 31.5, 1)));
    if (pa > 0) { clipTo(m, () => { for (let k = 0; k < RG.n; k += 15) { for (const [PP, col, al] of [[P1, P.rose, 0.7], [P2, P.sage, 0.95]]) { setA(pa * al); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
        const n = Math.round((PP.S - 1) * (PP === P2 ? ease(prog(t, 17, 3)) : 1)); for (let s = 0; s <= n; s++) { const p = ptAt(PP, k, s); s ? ctx.lineTo(m.X(p[0]), m.Y(p[1])) : ctx.moveTo(m.X(p[0]), m.Y(p[1])); } ctx.stroke(); setA(1); }
        const z = ptAt(P1, k, 0), x = ptAt(P1, k, P1.S - 1); dot(m.X(z[0]), m.Y(z[1]), 3, P.rose, pa); dot(m.X(x[0]), m.Y(x[1]), 3.5, P.terracotta, pa); } });
      text('rose: round 1 (dives to the middle, then turns)  ·  sage: after reflow', m.x0, m.y0 + m.S + 26, { font: F.mono, size: 15, color: P.dust, a: pa }); }
    const oa = eout(prog(t, 32, 0.8));
    if (oa > 0) { GA = oa; const L = sq(110, 250, 300), Rm = sq(450, 250, 300);
      [[L, 'r1_1', 'round 1 · 1 step', P.rose], [Rm, 'r2_1', 'after reflow · 1 step', P.sage]].forEach(([mm, key, nm, col], j) => { const st = RG.steps[key].stages, f = hopF(t, 33 + j * 1.2, 1.4, 1);
        frame(mm, 1); clipTo(mm, () => st[0].forEach((_, k) => { const p = [lerp(st[0][k][0], st[1][k][0], f), lerp(st[0][k][1], st[1][k][1], f)]; dot(mm.X(p[0]), mm.Y(p[1]), 1.8, col, 1); }));
        text(nm, mm.x0, mm.y0 + mm.S + 28, { font: F.mono, size: 16, color: col }); });
      const da = eout(prog(t, 40, 0.8)); if (da > 0) { [['fm1', 'round 1 · 1 step', P.rose], ['rf1', 'reflow · 1 step', P.sage], ['rf2', 'reflow · 2 steps', P.sage]].forEach(([k, nm, col], j) => {
        const x = 110 + j * 220, a = da * eout(prog(t, 40 + j * 0.5, 0.5)); digitGrid(k, 4, x, 640, 2, 96, 4, a); text(nm, x, 860, { font: F.mono, size: 15, color: col, a }); text(`FID ${J[k].FID.toFixed(1)}`, x, 886, { font: F.mono, size: 15, color: P.dust, a }); }); }
      GA = 1; }
    const c1 = eout(prog(t, 1, 0.8)); card(830, 250, 1000, 170, P.sage, c1); label('reflow', 864, 296, c1, P.sage);
    M('[n|z] ∼ N(0, I)   →   [d|x̃] = flow([n|z])      new pairs ([n|z], [d|x̃])', 864, 356, 24, { a: c1 }); M('then train again:  ‖ [m|v][m_|θ](xₜ, t) − ([d|x̃] − [n|z]) ‖²', 864, 400, 24, { a: eout(prog(t, 8, 0.8)) });
    const c2 = eout(prog(t, 24, 0.8)); card(830, 440, 1000, 170, P.amber, c2); label('straightness (1 = a ruler)', 864, 486, c2, P.amber);
    [['round 1', RG.ratio1, P.rose], ['after reflow', RG.ratio2, P.sage]].forEach(([nm, v, col], i) => { const y = 530 + i * 40, g = eout(prog(t, 24.5 + i * 0.8, 1)); text(nm, 864, y + 8, { font: F.mono, size: 17, color: col, a: c2 });
      rbox(1060, y - 10, 600, 20, 6, P.surface, P.border, c2); setA(c2); ctx.fillStyle = col; ctx.fillRect(1060, y - 10, 600 * v * g, 20); setA(1); text(v.toFixed(3), 1680, y + 8, { font: F.mono, size: 17, color: P.chalk, a: c2 * g }); });
    const c3 = eout(prog(t, 48, 0.8)); if (c3 > 0) { card(830, 630, 1000, 150, P.plum, c3); para('Reflow once, then distil into a one-step network: InstaFlow (2023) drew text-to-image pictures in a single step this way.', 864, 684, 930, { size: 24, a: c3 }); }
  } });

// 10 — one family
const FAMILY_K = [5, 60, 150, 260, 340, 420, 480, 555];
SC.push({ dur: 48, chapter: 'one family', title: 'It was one idea all along', sub: 'Velocity, noise, clean image, score: four names for one arrow.',
  caps: [[1, 8, 'Step back. Diffusion’s DDIM was also a wind carrying noise to data, a deterministic flow. Only the roads differ.'],
         [8, 16, 'Diffusion’s road from z to x is a curve, √ᾱ·x + √(1 − ᾱ)·z. Flow matching’s road is a straight line. Same start, same finish.'],
         [16, 24, 'And the network’s outputs are interchangeable. Knowing the velocity at xₜ means knowing the clean guess, x̂ = xₜ + (1 − t)·v.'],
         [24, 32, 'And the noise guess, ẑ = xₜ − t·v. And episode 8’s score, s = −ẑ / (1 − t). Four names, one arrow.'],
         [32, 40, 'Flows in episode 6, scores in 8, diffusion in 9, flow matching today: one family, seen from four sides.'],
         [40, 47.6, 'What flow matching changed is the road: the simplest one there is.']],
  draw(t) {
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const g = ease(prog(t, 2, 5)), sa = eout(prog(t, 9, 1));
    clipTo(m, () => FAMILY_K.forEach(k => { const z = RG.x0[k], x = RG.x1[k];
      setA(a0 * 0.85); ctx.strokeStyle = P.rose; ctx.lineWidth = 2.2; ctx.beginPath();
      for (let i = 0; i <= 60 * g; i++) { const s = i / 60, ab = EP9_ABAR(s), p = [Math.sqrt(ab) * x[0] + Math.sqrt(1 - ab) * z[0], Math.sqrt(ab) * x[1] + Math.sqrt(1 - ab) * z[1]]; i ? ctx.lineTo(m.X(p[0]), m.Y(p[1])) : ctx.moveTo(m.X(p[0]), m.Y(p[1])); }
      ctx.stroke(); setA(1); if (sa > 0) roadLine(m, z, [lerp(z[0], x[0], sa), lerp(z[1], x[1], sa)], P.sage, a0, 2.2);
      marble(m.X(z[0]), m.Y(z[1]), 5, P.rose, a0); marble(m.X(x[0]), m.Y(x[1]), 5, P.terracotta, a0); }));
    text('rose: diffusion’s road · sage: the straight road', 110, 925, { font: F.mono, size: 16, color: P.dust, a: a0 });
    const c1 = eout(prog(t, 8, 0.8)); card(830, 250, 1000, 150, P.ink, c1); label('two roads, same ends', 864, 296, c1, P.ink);
    M('diffusion:  √[a|ᾱ]·[d|x] + √(1 − [a|ᾱ])·[n|z]        flow matching:  (1 − [a|t])·[n|z] + [a|t]·[d|x]', 864, 360, 22, { a: c1 });
    const rows = [[16, '[s|x̂] = [d|x][d_|t] + (1 − t)·[m|v]', 'the clean guess (episode 9’s x̂₀)'], [24, '[n|ẑ] = [d|x][d_|t] − t·[m|v]', 'the noise guess (episode 9’s εθ)'], [27, '[s|s] = −[n|ẑ] / (1 − t)', 'the score (episode 8)']];
    const c2 = eout(prog(t, 16, 0.8)); card(830, 420, 1000, 260, P.sage, c2); label('one arrow, four names', 864, 466, c2, P.sage);
    rows.forEach(([t0, eq, nm], i) => { const a = eout(prog(t, t0, 0.8)); M(eq, 864, 526 + i * 56, 28, { a }); text(nm, 1420, 526 + i * 56, { font: F.mono, size: 15, color: P.dust, a }); });
    const c3 = eout(prog(t, 32, 0.8)); if (c3 > 0) { card(830, 700, 1000, 230, P.plum, c3); const fam = [['06', 'flows', 'dx/dt = v'], ['08', 'scores', '∇ log p'], ['09', 'diffusion', 'εθ'], ['10', 'flow matching', 'vθ']];
      fam.forEach(([n, nm, eq], i) => { const x = 900 + i * 240, a = c3 * eout(prog(t, 32.5 + i * 0.6, 0.5)); dot(x, 780, 9, i === 3 ? P.terracotta : P.stone, a); if (i) line(x - 240 + 12, 780, x - 12, 780, P.border, 2, a);
        text(n, x, 820, { font: F.mono, size: 16, color: i === 3 ? P.terracotta : P.dust, align: 'center', a }); text(nm, x, 850, { size: 20, color: P.chalk, align: 'center', a }); text(eq, x, 882, { font: F.mono, size: 16, color: P.sage, align: 'center', a }); }); }
  } });

// 11 — in the wild
SC.push({ dur: 40, chapter: 'wild', title: 'Straight roads, everywhere', sub: 'From toy blobs to the image, video and speech models of today.',
  caps: [[1, 9, 'In 2024, Stable Diffusion 3 switched from predicting noise to rectified flow: straight roads, trained at scale.'],
         [9, 18, 'Flux, among the strongest open image models, is a rectified-flow transformer too.'],
         [18, 27, 'Meta’s Movie Gen makes video with flow matching, and its Voicebox model makes speech with it.'],
         [27, 34, 'And reflow with distillation pushes the steps toward one: InstaFlow drew images in a single step.'],
         [34, 39.6, 'Straight roads, a plain regression, fewer steps.']],
  draw(t) {
    const cards = [[1, 'stable diffusion 3 · 2024', P.terracotta, 'rectified flow, at scale', 'straight roads replace the noise schedule'],
      [9, 'flux · 2024', P.plum, 'a rectified-flow transformer', 'open image model, text to image'],
      [18, 'movie gen · voicebox', P.sage, 'flow matching for video and speech', 'the same regression, other kinds of data'],
      [27, 'instaflow · 2023', P.amber, 'reflow + distillation', 'text-to-image in one step']];
    cards.forEach(([t0, h, c, m1, s], i) => { const a = eout(prog(t, t0, 0.8)), x = 100 + (i % 2) * 880, y = 260 + Math.floor(i / 2) * 330;
      card(x, y, 840, 300, c, a); label(h, x + 32, y + 46, a, c); text(m1, x + 32, y + 140, { font: F.mono, size: 26, color: P.chalk, a }); para(s, x + 32, y + 206, 780, { size: 24, a: a * eout(prog(t, t0 + 1, 0.8)) }); });
  } });

// 12 — build it
const CODE = [
  '[k|# training: regress the velocity of a random straight road]',
  '[m|for] step [m|in] [p|range]([a|12000]):',
  '    x = [d|data_batch]([a|512]);  z = [n|randn]([a|512], [a|2]);  t = [a|rand]([a|512], [a|1])',
  '    xt = ([a|1] - t)*z + t*x                      [k|# a point on the road]',
  '    err = [p|net](xt, t) - (x - z)                [k|# vs the road’s velocity]',
  '    net.[p|backward]([a|2]*err/[a|512]);  net.[p|step](lr)',
  '',
  '[k|# sampling: K Euler steps along the learned wind]',
  'x = [n|randn](n, [a|2])',
  '[m|for] k [m|in] [p|range](K):',
  '    x = x + ([a|1]/K) * [p|net](x, k/K)',
  '',
  '[k|# reflow: pair each noise point with where it lands, train again]',
  'Z = [n|randn]([a|100000], [a|2]);  X = [p|sample](Z, K=[a|200])',
  'net2 = [p|train](pairs=(Z, X))              [k|# nearly straight: 1–2 steps]',
];
SC.push({ dur: 32, chapter: 'build', title: 'Build it yourself', sub: 'Training in five lines, sampling in two.',
  caps: [[1, 9, 'Training is five lines: a random road, a point on it, a regression onto its velocity. Sampling is a two-line loop.'],
         [9, 18, 'In the playground, drag the step count from 64 down to 1, and watch reflow rescue the one-step samples.'],
         [18, 31.6, 'Recap: join noise to data with random straight roads, learn their average velocity, follow it. Straighten the roads, and one step is enough.']],
  draw(t) {
    playPill('ep10-flow-lab.html', 9, t);
    const a = eout(prog(t, 0.6, 0.8)); card(100, 240, 1100, 690, P.plum, a);
    CODE.forEach((l, i) => { if (l) M(l, 136, 296 + i * 40, 18, { a: a * eout(prog(t, 0.8 + i * 0.12, 0.4)) }); });
    const ra = eout(prog(t, 18, 0.8)); card(1240, 240, 580, 690, P.sage, ra); label('the whole episode', 1276, 290, ra, P.sage);
    [['the road', '[d|x]ₜ = (1 − t)·[n|z] + t·[d|x]'], ['the loss', '‖[m|vθ]([d|x]ₜ, t) − ([d|x] − [n|z])‖²'], ['what it learns', 'the average velocity'], ['sampling', 'Euler: [d|x] += Δt·[m|vθ]'], ['reflow', 'new pairs → straight → 1 step']].forEach(([nm, mm], i) => {
      const ai = eout(prog(t, 18.5 + i * 1.6, 0.8)); text(nm, 1276, 370 + i * 112, { font: F.serif, italic: true, size: 24, color: P.dust, a: ai }); M(mm, 1276, 412 + i * 112, 22, { a: ai }); });
  } });

// 13 — hook → guidance
const HK = A.hook, HPN = stages(HK.paths.none, HK.z.length), HP2 = stages(HK.paths['2'], HK.z.length), HP6 = stages(HK.paths['6'], HK.z.length);
SC.push({ dur: 46, chapter: 'next', title: 'Which one?', sub: 'Generating is easy. Asking is harder.',
  caps: [[1, 8, 'We can turn noise into digits in a handful of steps. But which digit? We get whatever the dice decide.'],
         [8, 16, 'Everything you type into an image generator is a request: a cat, in watercolour, at sunset. How does a model listen?'],
         [16, 24, 'First idea: hand the network the label too, vθ(x, t, y). Here one real flow is asked for blob 3, then blob 7.'],
         [24, 32, 'It works, more or less. But real models listen too softly, and one famous trick turns up the volume.'],
         [32, 39, 'Compare what the wind does with the request and without it, then push further in that direction. How far is a dial.'],
         [39, 45.6, 'Next episode: guidance. Steering the flow toward what we ask for.']],
  draw(t) {
    const dimAll = 1 - 0.9 * ease(prog(t, 38.5, 1)); GA = dimAll;
    const m = sq(110, 250, 640), a0 = eout(prog(t, 0.6, 0.8)); frame(m, a0);
    const phase = t < 16 ? 0 : t < 24 ? 1 : 2, PP = [HPN, HP2, HP6][t < 16 ? 0 : t < 20 ? 1 : 2], t0 = t < 16 ? 1.5 : t < 20 ? 16.3 : 20.3, f = (PP.S - 1) * ease(prog(t, t0, 3.2)), want = t < 16 ? -1 : t < 20 ? 2 : 6;
    clipTo(m, () => { for (let k = 0; k < 8; k++) { const b = BLOB(k); ring(m.X(b[0]), m.Y(b[1]), 26, BLOBCOL[k], a0 * (want === k ? 1 : 0.35), want === k ? 3 : 1.5); text(String(k + 1), m.X(b[0] * 1.35), m.Y(b[1] * 1.35) + 6, { font: F.mono, size: 18, color: BLOBCOL[k], align: 'center', a: a0 * (want === k || want < 0 ? 1 : 0.4) }); }
      for (let k = 0; k < PP.n; k++) { const p = ptAt(PP, k, f); marble(m.X(p[0]), m.Y(p[1]), 3, f >= PP.S - 1.5 ? BLOBCOL[nearestBlob(p)] : mixc(P.rose, P.terracotta, f / (PP.S - 1)), a0); } });
    text(phase === 0 ? 'no request: any blob' : `asked for blob ${want + 1}`, 110, 925, { font: F.mono, size: 20, color: phase === 0 ? P.stone : BLOBCOL[want], a: a0 });
    void phase;
    const c1 = eout(prog(t, 1, 0.8)) * (1 - ease(prog(t, 15.5, 0.8))); if (c1 > 0) { card(830, 250, 1000, 380, P.terracotta, c1); label('whatever the dice decide', 864, 296, c1);
      digitGrid('fm32', 16, 880, 330, 8, 60, 6, c1); para('Sixteen samples, sixteen surprises: nothing in the recipe says which digit to draw.', 864, 520, 930, { size: 26, a: c1 }); }
    const c2 = eout(prog(t, 16, 0.8)); card(830, 250, 1000, 160, P.amber, c2 * (t > 15.5 ? 1 : 0)); if (t > 15.5) { label('a request as an input', 864, 296, c2, P.amber);
      M('[m|v][m_|θ]([d|x], t)   →   [m|v][m_|θ]([d|x], t, [a|y])', 864, 352, 28, { a: c2 }); text('y = “blob 3”, “a 7”, “a cat in watercolour”', 864, 390, { font: F.mono, size: 16, color: P.dust, a: c2 }); }
    const c3 = eout(prog(t, 24, 0.8)); if (c3 > 0) { card(830, 430, 1000, 200, P.plum, c3); label('how hard should it listen?', 864, 476, c3, P.plum);
      M('[m|v] = [m|v][m_|∅] + [a|w] · ( [m|v][m_|y] − [m|v][m_|∅] )', 864, 546, 34, { a: eout(prog(t, 32, 0.8)) }); text('w: the guidance dial', 864, 596, { font: F.mono, size: 16, color: P.amber, a: eout(prog(t, 33, 0.8)) }); }
    GA = 1;
    const qa = eout(prog(t, 38.5, 1.2));
    if (qa > 0) {
      text('Ask,', 960, 440, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: qa });
      text('and it will draw.', 960, 530, { font: F.serif, size: 76, color: P.chalk, align: 'center', a: eout(prog(t, 39.3, 1.2)) });
      text('NEXT  ·  EPISODE 11', 960, 640, { font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 40.5, 1)), ls: 4 });
      seasonStrip(10, 41.7, t);
      text('Guidance — steering generation toward what we ask for', 960, 695, { font: F.serif, italic: true, size: 42, color: P.stone, align: 'center', a: eout(prog(t, 41, 1)) });
    }
  } });
