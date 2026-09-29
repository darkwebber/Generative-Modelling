"""
Start a new episode with the series' conventions already in place:

    python src/new_episode.py 10 ep10-flow-matching.html "Flow Matching"

writes src/ep10-scenes.js (a cold open, one worked scene, the hook), src/ep10_audio.py (a score + cues on the same
clock) and registers the page in src/build.py, then builds it. Fill in the scenes; export any real-model data
to video/ep10-assets.js (window.EP10 = {…}) and add it to the episode's scripts in build.py.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SCENES = r'''// ─────────────────────────────────────────────────────────────
//  Episode {num} — {title}
//  Data from code/export_ep{num}_assets.py arrives as window.EP{n} (add 'ep{num}-assets.js' in build.py).
//  Engine: core.js (P colours, F fonts, text/M/card/label/tag/pill/insight/arrow/dot…, ease/eout/prog/pop, rng)
//          kit.js  (plotSquare/frame/clipTo, marble, u8imgs/imgCanvas/drawImg, stamp, gaussian, memo, shake)
//  Layout: header above y≈200 · content y 240–930 · captions at y≈972–1014 (plain text: use ₀ ₜ θ, not _).
// ─────────────────────────────────────────────────────────────
const SC = [];

// 00 — cold open: no header, the title lands with a bang
SC.push({{ dur: 20, chapter: 'intro',
  caps: [[1, 8, 'Last time: … (pick up the previous episode’s hook).'],
         [8, 15, 'Today: … (the one question this episode answers).']],
  draw(t) {{
    const a = eout(prog(t, 15, 1.2));
    text('EPISODE {num}', 960, 380, {{ font: F.serif, italic: true, size: 26, color: P.terracotta, align: 'center', a, ls: 4 }});
    text('{title}', 960, 520, {{ font: F.serif, size: 110, color: P.chalk, align: 'center', a }});
  }} }});

// 01 — one idea per scene: a picture, the maths in a card, captions that say it in words
SC.push({{ dur: 30, chapter: 'idea', title: 'The idea', sub: 'One sentence that frames the scene.',
  caps: [[1, 10, 'Say what the picture shows.'], [10, 20, 'Then why it matters.'], [20, 29.6, 'Then the formula, read aloud.']],
  draw(t) {{
    const a0 = eout(prog(t, 0.6, 0.8)), m = plotSquare(110, 250, 600, 3); frame(m, a0);
    clipTo(m, () => {{ const r = rng(1), g = gaussian(r); for (let i = 0; i < 400; i++) marble(m.X(g()), m.Y(g()), 3, P.terracotta, a0); }});
    const c1 = eout(prog(t, 20, 0.8)); card(780, 250, 1040, 150, P.sage, c1); label('the formula', 814, 296, c1, P.sage);
    M('[d|x] ~ [p|p]([d|x])   ([k|colour-coded: d data · a knobs · n noise · m model · s guesses])', 814, 356, 26, {{ a: c1 }});
  }} }});

// last — the hook into the next episode
SC.push({{ dur: 20, chapter: 'next', title: 'What’s next', sub: 'The question this episode leaves open.',
  caps: [[1, 10, 'The limitation, shown.'], [10, 19.6, 'Next episode: …']],
  draw(t) {{
    const qa = eout(prog(t, 10, 1.2));
    text('The open question?', 960, 480, {{ font: F.serif, size: 72, color: P.chalk, align: 'center', a: qa }});
    text('NEXT  ·  EPISODE {next}', 960, 590, {{ font: F.mono, size: 22, color: P.terracotta, align: 'center', a: eout(prog(t, 11, 1)), ls: 4 }});
  }} }});
'''

AUDIO = r'''"""Episode {num} soundtrack — {title}. Pick a key, a chord per scene and a lead; cue sounds on the scenes' clock.

    python src/ep{num}_audio.py      # → ep{num}-audio.mp3   (or: python src/make.py {n})
"""
from audio_lib import Score, ease, prog, scene_durations  # noqa: F401

S = Score('ep{num}-audio', scene_durations({n}), seed={n})
CH = {{'Cmaj9': ([48, 55, 59, 62, 64], 36), 'Am9': ([45, 52, 55, 59, 60], 33), 'Fmaj7': ([41, 48, 53, 57, 60], 29), 'G6': ([43, 50, 55, 59, 64], 31)}}
S.score(CH, [
    ([(0, 'Am9'), (15, 'Cmaj9')], 0.0, .9),       # 00 intro     (chord changes at scene-local seconds, lead density, level)
    ([(0, 'Cmaj9'), (15, 'Fmaj7')], 0.8, .9),     # 01 idea
    ([(0, 'Fmaj7'), (10, 'G6')], 0.6, .8),        # last: hook
], [60, 62, 64, 67, 69, 72, 74, 76, 79], lead='celesta')
S.transitions()
c = S.cue
c(0, 15, 'thud', 1.0, f=40); c(0, 15, 'chime', 1.0, m=72)                  # the title lands
for i in range(4): c(1, 20 + i * 0.08, 'key', 0.4, 0.2 * i, f=2400)        # the formula card types in
S.render()
'''


def main():
    n, page, title = int(sys.argv[1]), sys.argv[2], sys.argv[3]; num = f'{n:02d}'
    for path, tpl in ((f'ep{num}-scenes.js', SCENES), (f'ep{num}_audio.py', AUDIO)):
        full = os.path.join(HERE, path)
        if os.path.exists(full): print('exists, left alone:', path); continue
        open(full, 'w', encoding='utf-8').write(tpl.format(num=num, n=n, title=title, next=f'{n + 1:02d}')); print('wrote', path)
    b = os.path.join(HERE, 'build.py'); src = open(b, encoding='utf-8').read()
    if not re.search(rf"^\s+{n}: \(", src, re.M):
        line = f"    {n}: ('{page}', '{title}', [], {{}}),\n"
        i = src.index('}\nARIA'); src = src[:i] + line + src[i:]; open(b, 'w', encoding='utf-8').write(src); print('registered in build.py')
    sys.path.insert(0, HERE)
    import build
    build.build(n)


if __name__ == '__main__':
    main()
