"""
One command from sources to finished files for an episode:

    python src/make.py 9                  # build the page + the soundtrack
    python src/make.py 9 --render         # … + render the 1080p MP4 and add the sound  (renders/epNN…-sound.mp4)
    python src/make.py 9 --render --preview   # … + a ~25 MB 720p preview (renders/epNN…-720p-sound.mp4)
    python src/make.py 9 --only preview   # just one stage: build | audio | render | mux | preview

Env: WORKERS=n renders n segments in parallel (default 3); FFMPEG=/path/to/ffmpeg (else ffmpeg on PATH,
else the one bundled with `pip install imageio-ffmpeg`).
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build import EPISODES, build  # noqa: E402


def ffmpeg():
    if os.environ.get('FFMPEG'): return os.environ['FFMPEG']
    if shutil.which('ffmpeg'): return 'ffmpeg'
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(*cmd, env=None):
    print('→', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True, env={**os.environ, **(env or {})})


def duration(path):
    out = subprocess.run([ffmpeg(), '-i', path], capture_output=True, text=True).stderr
    h, m, s = out.split('Duration: ')[1].split(',')[0].split(':'); return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', type=int)
    ap.add_argument('--render', action='store_true', help='also render the MP4 and add the soundtrack')
    ap.add_argument('--preview', action='store_true', help='also make a ~25 MB 720p preview')
    ap.add_argument('--only', choices=['build', 'audio', 'render', 'mux', 'preview'])
    ap.add_argument('--fps', type=int, default=30)
    a = ap.parse_args()
    n = a.episode; num = f'{n:02d}'; page = EPISODES[n][0]; stem = page[:-5]
    stages = [a.only] if a.only else ['build', 'audio'] + (['render', 'mux'] if a.render else []) + (['preview'] if a.preview else [])
    ff = ffmpeg(); mp4, snd, prev, mp3 = (f'renders/{stem}.mp4', f'renders/{stem}-sound.mp4', f'renders/{stem}-720p-sound.mp4', f'episodes/ep{num}-audio.mp3')
    os.makedirs(os.path.join(ROOT, 'renders'), exist_ok=True)
    for st in stages:
        if st == 'build': build(n)
        if st == 'audio': run(sys.executable, os.path.join(HERE, 'audio', f'ep{num}.py'), env={'FFMPEG': ff})
        if st == 'render': run('node', 'render.mjs', f'episodes/{page}', a.fps, mp4, env={'FFMPEG': ff, 'WORKERS': os.environ.get('WORKERS', '3')})
        if st == 'mux': run(ff, '-loglevel', 'error', '-y', '-i', mp4, '-i', mp3, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', snd)
        if st == 'preview':
            kbps = max(200, int(25 * 8000 / duration(os.path.join(ROOT, snd))) - 72)       # ≈25 MB in total
            run(ff, '-loglevel', 'error', '-y', '-i', snd, '-vf', 'scale=1280:720', '-c:v', 'libx264', '-preset', 'slow', '-b:v', f'{kbps}k',
                '-maxrate', f'{2 * kbps}k', '-bufsize', f'{4 * kbps}k', '-c:a', 'aac', '-b:a', '64k', prev)
    print('done:', ', '.join(stages))


if __name__ == '__main__':
    main()
