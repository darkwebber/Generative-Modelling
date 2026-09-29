// season.js — the season's table of contents, shared by the hub, every episode page (inlined by src/build.py) and
// every playground. GM.href() links pages to each other: relative files when opened from this folder, and the
// published Claude artifacts (the `url` fields) when viewed online, where the other files are not next door.
window.GM = (function () {
  const eps = [
    { n: 1, title: 'Generative Modelling', page: 'ep01-generative-modelling.html', lab: 'ep01-density-lab.html', labName: 'Density Lab', code: 'density.py' },
    { n: 2, title: 'Autoregressive Generation', page: 'ep02-autoregressive.html', lab: 'ep02-ar-playground.html', labName: 'Autoregressive Playground', code: 'ar_counting.py' },
    { n: 3, title: 'Autoencoders & VAEs', page: 'ep03-autoencoders-vae.html', lab: 'ep03-vae-playground.html', labName: 'Latent Space Playground', code: 'vae.py' },
    { n: 4, title: 'Grading the Imagination', page: 'ep04-evaluation.html', lab: 'ep04-metric-lab.html', labName: 'Metric Lab', code: 'evals.py' },
    { n: 5, title: 'GANs', page: 'ep05-gans.html', lab: 'ep05-gan-arena.html', labName: 'GAN Arena', code: 'gan.py' },
    { n: 6, title: 'Normalizing Flows', page: 'ep06-flows.html', lab: 'ep06-flow-lab.html', labName: 'Flow Lab', code: 'flow.py' },
    { n: 7, title: 'Energy-Based Models', page: 'ep07-energy.html', lab: 'ep07-ebm-lab.html', labName: 'Energy Lab', code: 'ebm.py' },
    { n: 8, title: 'Score Matching & Langevin', page: 'ep08-score.html', lab: 'ep08-score-lab.html', labName: 'Score Lab', code: 'score.py' },
    { n: 9, title: 'Diffusion Models', page: 'ep09-diffusion.html', lab: 'ep09-diffusion-lab.html', labName: 'Diffusion Lab', code: 'diffusion.py' },
    { n: 10, title: 'Flow Matching', page: 'ep10-flow-matching.html', lab: 'ep10-flow-lab.html', labName: 'Flow Matching Lab', code: 'flow_matching.py' },
    { n: 11, title: 'Guidance', page: 'ep11-guidance.html', lab: 'ep11-guidance-lab.html', labName: 'Guidance Lab', code: 'guidance.py' },
  ];
  // published artifacts (claude.ai), filled in as they are published
  const URL = {
    hub: null,
    ep: { 1: 'https://claude.ai/artifact/JpqRF6pKvGRi8cK3cPsfZP', 2: 'https://claude.ai/artifact/3Dif6UYkDxky2pavqoqwvM', 3: 'https://claude.ai/artifact/EE1JD2Qq7PzuJWvpZBFLKi',
          4: 'https://claude.ai/artifact/CPUcuRDmkKUVm4jHXkqx4q', 5: 'https://claude.ai/artifact/9LbD4PaxKsHCW7q6wM9uSY', 6: 'https://claude.ai/artifact/P3MyVYi2RyGTMNkwZvnHX8',
          7: 'https://claude.ai/artifact/RBYTbCT4qpnViT2rrP1WSK', 8: 'https://claude.ai/artifact/5NC1JR7LE3kz6N3B3diM3P', 9: 'https://claude.ai/artifact/TkYw57ufRvnqXUbibDUvzD',
          10: 'https://claude.ai/artifact/8c8CcQLFogi2LmZELP3nCj', 11: 'https://claude.ai/artifact/HQ2ebzwJmVfQ9ksAf6EG1U' },
    lab: {},
  };
  const local = /^(file:)$/.test(location.protocol) || /^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])$/.test(location.hostname);
  // kind: 'hub' | 'ep' | 'lab' | 'code';  base: path from this page to the video/ folder ('' or '../')
  function href(kind, n, base) {
    const e = eps[n - 1];
    if (local) return kind === 'hub' ? base + 'index.html' : kind === 'ep' ? base + 'episodes/' + e.page : kind === 'lab' ? base + 'labs/' + e.lab : base + '../code/' + e.code;
    return kind === 'hub' ? URL.hub : kind === 'ep' ? URL.ep[n] || null : kind === 'lab' ? URL.lab[n] || null : null;
  }
  // a small row of links; skips any that have no destination here
  function nav(el, items) {
    el.innerHTML = items.filter(([, h]) => h).map(([label, h, cls]) => `<a href="${h}"${local ? '' : ' target="_top"'} class="${cls || ''}">${label}</a>`).join('');
    el.hidden = !el.innerHTML;
  }
  return { eps, URL, local, href, nav };
})();
