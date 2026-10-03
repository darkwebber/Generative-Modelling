// season.js — the season's table of contents, shared by the hub, every episode page (inlined by src/build.py) and
// every playground. GM.href() links pages to each other: relative files when opened from this folder, the website's
// own pages when site/build.py has built it (SITE below), and otherwise the published Claude artifacts (URL below).
window.GM = (function () {
  const eps = [
    { n: 1, title: 'Generative Modelling', page: 'ep01-generative-modelling.html', lab: 'ep01-density-lab.html', labName: 'Density Lab', code: 'density.py', q: 'What does it mean to learn p(x)?' },
    { n: 2, title: 'Autoregressive Generation', page: 'ep02-autoregressive.html', lab: 'ep02-ar-playground.html', labName: 'Autoregressive Playground', code: 'ar_counting.py', q: 'Can a machine write by guessing one piece at a time?' },
    { n: 3, title: 'Autoencoders & VAEs', page: 'ep03-autoencoders-vae.html', lab: 'ep03-vae-playground.html', labName: 'Latent Space Playground', code: 'vae.py', q: 'Can a machine learn by compressing?' },
    { n: 4, title: 'Grading the Imagination', page: 'ep04-evaluation.html', lab: 'ep04-metric-lab.html', labName: 'Metric Lab', code: 'evals.py', q: 'How do you grade a machine whose job is to make something new?' },
    { n: 5, title: 'GANs', page: 'ep05-gans.html', lab: 'ep05-gan-arena.html', labName: 'GAN Arena', code: 'gan.py', q: 'What if the judge could teach?' },
    { n: 6, title: 'Normalizing Flows', page: 'ep06-flows.html', lab: 'ep06-flow-lab.html', labName: 'Flow Lab', code: 'flow.py', q: 'Can a generator run backwards?' },
    { n: 7, title: 'Energy-Based Models', page: 'ep07-energy.html', lab: 'ep07-ebm-lab.html', labName: 'Energy Lab', code: 'ebm.py', q: 'What if any network could be a density?' },
    { n: 8, title: 'Score Matching & Langevin', page: 'ep08-score.html', lab: 'ep08-score-lab.html', labName: 'Score Lab', code: 'score.py', q: 'Forget the landscape: can we learn just the arrows?' },
    { n: 9, title: 'Diffusion Models', page: 'ep09-diffusion.html', lab: 'ep09-diffusion-lab.html', labName: 'Diffusion Lab', code: 'diffusion.py', q: 'Can we destroy data slowly, then learn to undo it?' },
    { n: 10, title: 'Flow Matching', page: 'ep10-flow-matching.html', lab: 'ep10-flow-lab.html', labName: 'Flow Matching Lab', code: 'flow_matching.py', q: 'Can the road from noise to data be straight?' },
    { n: 11, title: 'Guidance', page: 'ep11-guidance.html', lab: 'ep11-guidance-lab.html', labName: 'Guidance Lab', code: 'guidance.py', q: 'How do we steer what gets generated?' },
  ];
  // published artifacts (claude.ai), filled in as they are published
  const URL = {
    hub: 'https://claude.ai/artifact/WVgy6kYirop3zktVEdREvY',
    ep: { 1: 'https://claude.ai/artifact/JpqRF6pKvGRi8cK3cPsfZP', 2: 'https://claude.ai/artifact/3Dif6UYkDxky2pavqoqwvM', 3: 'https://claude.ai/artifact/EE1JD2Qq7PzuJWvpZBFLKi',
          4: 'https://claude.ai/artifact/CPUcuRDmkKUVm4jHXkqx4q', 5: 'https://claude.ai/artifact/9LbD4PaxKsHCW7q6wM9uSY', 6: 'https://claude.ai/artifact/P3MyVYi2RyGTMNkwZvnHX8',
          7: 'https://claude.ai/artifact/RBYTbCT4qpnViT2rrP1WSK', 8: 'https://claude.ai/artifact/5NC1JR7LE3kz6N3B3diM3P', 9: 'https://claude.ai/artifact/TkYw57ufRvnqXUbibDUvzD',
          10: 'https://claude.ai/artifact/8c8CcQLFogi2LmZELP3nCj', 11: 'https://claude.ai/artifact/HQ2ebzwJmVfQ9ksAf6EG1U' },
    lab: { 1: 'https://claude.ai/artifact/4FgUUwtcBoxGfngx4GyN1x', 2: 'https://claude.ai/artifact/CnmK6jBMAPP87tAvUFW7fP', 3: 'https://claude.ai/artifact/H2KdSuiVqno9C2awDe632j',
           4: 'https://claude.ai/artifact/JTkMZ3MwnCbCebchVQ8hRV', 5: 'https://claude.ai/artifact/6ENTFejgxbYVHcVpEJoec5', 6: 'https://claude.ai/artifact/9jn1d8p7hDu7daXV95scGe',
           7: 'https://claude.ai/artifact/9QWRawZm2ETNyQbcPPkai9', 8: 'https://claude.ai/artifact/LpMp41VXM1eX7qtmhUZ9oe', 9: 'https://claude.ai/artifact/XYi8ABXwLBGkW4o9BUr3zF',
           10: 'https://claude.ai/artifact/YaESNgKaqG5KgZF9gddoPv', 11: 'https://claude.ai/artifact/Rw9UJxfeuEXedezAbu7ReN' },
  };
  // site/build.py replaces the next line with what the website shows: { target: 'public' | 'private',
  // show: {n: {ep, lab, code}} (each 'public' | 'redacted' | 'hidden'), plan: the public site's plan, for private badges }
  const SITE = null;
  const local = !SITE && (/^(file:)$/.test(location.protocol) || /^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])$/.test(location.hostname));
  const framed = !local && !SITE;   // inside a claude.ai artifact frame, links must open in the top window
  // 'public' | 'redacted' (listed as coming soon, no page) | 'hidden' (not mentioned); always public off the website
  function state(kind, n) { return !SITE || kind === 'hub' ? 'public' : (SITE.show[n] || {})[kind] || 'hidden'; }
  // kind: 'hub' | 'ep' | 'lab' | 'code';  base: path from this page to the video/ folder ('' or '../')
  // returns a link, or false for a redacted page (show it as coming soon), or null when there is nothing to show
  function href(kind, n, base) {
    const e = eps[n - 1];
    if (SITE) {
      const s = state(kind, n); if (s !== 'public') return s === 'redacted' ? false : null;
      return kind === 'hub' ? base || './' : kind === 'ep' ? base + 'episodes/' + e.page.replace(/\.html$/, '') : kind === 'lab' ? base + 'labs/' + e.lab.replace(/\.html$/, '') : base + 'code/' + e.code.replace(/\.py$/, '');
    }
    if (local) return kind === 'hub' ? base + 'index.html' : kind === 'ep' ? base + 'episodes/' + e.page : kind === 'lab' ? base + 'labs/' + e.lab : base + '../code/' + e.code;
    return kind === 'hub' ? URL.hub : kind === 'ep' ? URL.ep[n] || null : kind === 'lab' ? URL.lab[n] || null : null;
  }
  // a small row of links; a redacted page shows as 'soon', and any with no destination here are left out
  function nav(el, items) {
    el.innerHTML = items.filter(([, h]) => h || h === false).map(([label, h, cls]) => h === false
      ? `<span class="soon ${cls || ''}" title="coming soon">${label} · soon</span>`
      : `<a href="${h}"${framed ? ' target="_top"' : ''} class="${cls || ''}">${label}</a>`).join('');
    el.hidden = !el.innerHTML;
  }
  return { eps, URL, SITE, local, framed, state, href, nav };
})();
