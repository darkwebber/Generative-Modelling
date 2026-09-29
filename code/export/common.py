"""Paths shared by the export scripts (and it puts code/ on the import path, so `import vae` etc. work)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)                                   # the models: vae.py, gan.py, diffusion.py …
DATA = os.path.join(CODE, 'data')                              # downloads and trained nets (not in git)
VIDEO = os.path.join(CODE, '..', 'video')
EPISODES = os.path.join(VIDEO, 'episodes')                     # epNN-assets.js, next to each episode page
LABS = os.path.join(VIDEO, 'labs')                             # data only a playground needs
sys.path.insert(0, CODE)
