"""
Build episode pages: one self-contained HTML per episode =
    engine/shell.html  (page, controls, <audio>)
  + engine/core.js     (Midnight Atelier tokens, maths, drawing helpers, M() colour maths)
  + engine/kit.js      (helpers several episodes share: plots, marbles, 28×28 images, stamps…)
  + scenes/epNN.js     (the episode: its data and its SC.push({...}) scenes)
  + engine/player.js   (timeline, header, captions, grain, player, audio sync, renderAt for render.mjs)

    python src/build.py            # every episode
    python src/build.py 9 10       # just these

To add an episode: write src/scenes/epNN.js (copy the skeleton in README → "Adding an episode"),
add a line to EPISODES below, then run this script. Assets (epNN-assets.js) and the soundtrack
(epNN-audio.mp3) sit next to the page in video/episodes/ and are loaded by name.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# num: (page, <title>, data scripts loaded before the engine, extra EPISODE options)
EPISODES = {
    1: ('ep01-generative-modelling.html', 'Ep 01 · Generative Modelling', ['ep01-assets.js'], {}),
    2: ('ep02-autoregressive.html', 'Ep 02 · Autoregressive Generation', ['names.js'], {}),
    3: ('ep03-autoencoders-vae.html', 'Ep 03 · Autoencoders & VAEs', ['ep03-assets.js'], {}),
    4: ('ep04-evaluation.html', 'Ep 04 · Grading the Imagination', ['ep04-assets.js'], {}),
    5: ('ep05-gans.html', 'Ep 05 · GANs', ['ep04-assets.js', 'ep05-assets.js'], {}),
    6: ('ep06-flows.html', 'Ep 06 · Normalizing Flows', ['ep06-assets.js'], {}),
    7: ('ep07-energy.html', 'Ep 07 · Energy-Based Models', ['ep07-assets.js'], {}),
    8: ('ep08-score.html', 'Ep 08 · Score Matching & Langevin', ['ep08-assets.js'], {}),
    9: ('ep09-diffusion.html', 'Ep 09 · Diffusion Models', ['ep09-assets.js'], {}),
    10: ('ep10-flow-matching.html', 'Ep 10 · Flow Matching', ['ep10-assets.js'], {}),
    11: ('ep11-guidance.html', 'Ep 11 · Guidance', ['ep11-assets.js'], {}),
}
ARIA = {1: 'Generative modelling explainer animation', 2: 'Autoregressive generation explainer animation'}


def read(*p): return open(os.path.join(HERE, *p), encoding='utf-8').read()


def build(n):
    page, title, scripts, opts = EPISODES[n]
    num = f'{n:02d}'
    config = {'num': num, **opts}
    body = '\n'.join([
        read('engine', 'core.js').rstrip(),
        read('engine', 'kit.js').rstrip(),
        f'const EPISODE = {json.dumps(config, ensure_ascii=False)};',
        '',
        read('scenes', f'ep{num}.js').rstrip(),
        '',
        read('engine', 'player.js').rstrip(),
        '',
        '// ── links to the season hub, this episode\'s playground and code, and its neighbours (labs/season.js)',
        open(os.path.join(ROOT, 'labs', 'season.js'), encoding='utf-8').read().rstrip(),
        f'(() => {{ const n = {n}, e = GM.eps[n - 1], h = (k, i) => GM.href(k, i, "../");',
        "  GM.nav(document.getElementById('links'), [n > 1 && [`◂ ep ${String(n - 1).padStart(2, '0')} · ${GM.eps[n - 2].title}`, h('ep', n - 1)], ['season hub', h('hub')],",
        "    [`▶ playground · ${e.labName}`, h('lab', n), 'lab'], [`code · ${e.code}`, h('code', n)], n < GM.eps.length && [`ep ${String(n + 1).padStart(2, '0')} · ${GM.eps[n].title} ▸`, h('ep', n + 1), 'next']].filter(Boolean)); })();",
    ])
    doc = read('engine', 'shell.html')
    for k, v in {'{{TITLE}}': title, '{{ARIA}}': ARIA.get(n, f'{title} explainer animation'), '{{AUDIO}}': f'ep{num}-audio.mp3',
                 '{{SCRIPTS}}': ''.join(f'<script src="{s}"></script>' for s in scripts)}.items():
        doc = doc.replace(k, v)
    doc = doc.replace('{{BODY}}', body)          # last, so nothing inside the scenes is ever templated
    with open(os.path.join(ROOT, 'episodes', page), 'w', encoding='utf-8') as fh: fh.write(doc)
    print(f'built episodes/{page}')


if __name__ == '__main__':
    for n in ([int(a) for a in sys.argv[1:]] or sorted(EPISODES)): build(n)
