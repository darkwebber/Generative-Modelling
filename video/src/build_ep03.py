"""Assemble ep03-autoencoders-vae.html = shared engine (from ep02) + src/ep03-scenes.js + player."""
import os
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
lines = open(os.path.join(root, 'ep02-autoregressive.html')).read().split('\n')
i_math = [i for i, l in enumerate(lines) if 'The language model maths' in l][0] - 1
i_tail = [i for i, l in enumerate(lines) if 'Timeline, chrome, grain, player' in l][0] - 1
head, tail = '\n'.join(lines[:i_math]), '\n'.join(lines[i_tail:])
for a, b in [('<title>Autoregressive Generation Explained</title>', '<title>Autoencoders and VAEs</title>'),
             ('aria-label="Autoregressive generation explainer animation"', 'aria-label="Autoencoders and VAEs explainer animation"'),
             ('<script src="names.js"></script>', '<script src="ep03-assets.js"></script>'),
             ('(shared with episode 01)', '(shared across episodes)')]:
    head = head.replace(a, b)
tail = tail.replace('`EP 02  ·  SCENE', '`EP 03  ·  SCENE').replace(
    "DIG = { '3': makeDigit('3'), '8': makeDigit('8'), '0': makeDigit('0'), '6': makeDigit('6') };",
    'AE = netOf(A.ae); VAE = netOf(A.vae); BAE = bounds(ZAE);')
scenes = open(os.path.join(here, 'ep03-scenes.js')).read()
open(os.path.join(root, 'ep03-autoencoders-vae.html'), 'w').write(head + '\n' + scenes + '\n' + tail)
print('built ep03-autoencoders-vae.html')
