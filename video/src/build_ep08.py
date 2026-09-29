"""Assemble ep08-score.html = shared engine (from ep02) + src/ep08-scenes.js + player."""
import os
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
lines = open(os.path.join(root, 'ep02-autoregressive.html')).read().split('\n')
i_math = [i for i, l in enumerate(lines) if 'The language model maths' in l][0] - 1
i_tail = [i for i, l in enumerate(lines) if 'Timeline, chrome, grain, player' in l][0] - 1
head, tail = '\n'.join(lines[:i_math]), '\n'.join(lines[i_tail:])
for a, b in [('<title>Autoregressive Generation Explained</title>', '<title>Score Matching & Langevin</title>'),
             ('aria-label="Autoregressive generation explainer animation"', 'aria-label="Score Matching & Langevin explainer animation"'),
             ('<script src="names.js"></script>', '<script src="ep08-assets.js"></script>'),
             ('(shared with episode 01)', '(shared across episodes)')]:
    head = head.replace(a, b)
tail = tail.replace('`EP 02  ·  SCENE', '`EP 08  ·  SCENE').replace(
    "DIG = { '3': makeDigit('3'), '8': makeDigit('8'), '0': makeDigit('0'), '6': makeDigit('6') };",
    'void 0;')
scenes = open(os.path.join(here, 'ep08-scenes.js')).read()
from audio_player import inject
open(os.path.join(root, 'ep08-score.html'), 'w').write(inject(head + '\n' + scenes + '\n' + tail, 'ep08-audio.mp3'))
print('built ep08-score.html')
