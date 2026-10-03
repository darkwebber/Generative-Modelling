"""Build generative.grasp.how (and its private twin) for Cloudflare Pages.

    python3 site/build.py public     # what everyone sees: only the parts publish.json marks public
    python3 site/build.py private    # everything, with badges showing what the public sees

Writes a plain static site to dist/. Nothing runs on the server: a page that is not public is simply
not uploaded, so it cannot be reached by guessing its address. Standard library only (Pages build image).
"""
import html
import io
import json
import keyword
import os
import re
import shutil
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO, CODE, DIST = os.path.join(ROOT, 'video'), os.path.join(ROOT, 'code'), os.path.join(ROOT, 'dist')
STATES = ('public', 'redacted', 'hidden')
PARTS = {'episode': 'ep', 'playground': 'lab', 'code': 'code'}   # publish.json name -> season.js name
MAX_FILE = 25 * 1024 * 1024                                      # Cloudflare Pages' per-file limit
SITE_LINE = '  const SITE = null;'


def load_plan():
    with open(os.environ.get('PUBLISH_FILE') or os.path.join(ROOT, 'publish.json'), encoding='utf-8') as f:   # override for testing
        eps = json.load(f)['episodes']
    plan = {}
    for n in range(1, 12):
        e = eps.get(str(n), {})
        plan[n] = {}
        for name, key in PARTS.items():
            s = e.get(name, 'hidden')                             # anything left out stays off the site
            if s not in STATES:
                sys.exit(f'publish.json: episode {n} {name} is {s!r}; use one of {", ".join(STATES)}')
            plan[n][key] = s
    return plan


def season_meta():
    """Episode titles and file names, read from labs/season.js so there is one source of truth."""
    src = open(os.path.join(VIDEO, 'labs', 'season.js'), encoding='utf-8').read()
    rows = re.findall(r"\{ n: (\d+), title: '([^']*)', page: '([^']*)', lab: '([^']*)', labName: '([^']*)', code: '([^']*)'[^}]*\}", src)
    assert len(rows) == 11, 'could not read the 11 episodes from season.js'
    return {int(n): dict(title=t, page=p, lab=l, labName=ln, code=c) for n, t, p, l, ln, c in rows}


class Site:
    def __init__(self, target, plan):
        self.target, self.plan = target, plan
        self.show = {n: {k: 'public' for k in PARTS.values()} for n in plan} if target == 'private' else plan
        site = {'target': target, 'show': self.show, 'plan': plan if target == 'private' else None}
        self.site_js = '  const SITE = ' + json.dumps(site, separators=(',', ':')) + ';'
        self.written = []

    def public(self, n, kind):
        return self.show[n][kind] == 'public'

    def write(self, rel, data):
        path = os.path.join(DIST, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data.encode('utf-8') if isinstance(data, str) else data)
        if len(data) > MAX_FILE:
            sys.exit(f'{rel} is {len(data) / 2**20:.1f} MiB; Cloudflare Pages takes at most 25 MiB per file')
        self.written.append(rel)

    def with_site(self, text, rel):
        if text.count(SITE_LINE) != 1:
            sys.exit(f'{rel}: expected one "{SITE_LINE.strip()}" line (rebuild the episodes with video/src/build.py)')
        return text.replace(SITE_LINE, self.site_js)

    def copy_page(self, rel):
        """Copy a page from video/ and every local file it loads (src=/href=), keeping relative paths."""
        text = open(os.path.join(VIDEO, rel), encoding='utf-8').read()
        if SITE_LINE in text:
            text = self.with_site(text, rel)
        self.write(rel, text)
        here = os.path.dirname(rel)
        for ref in sorted(set(re.findall(r'(?:src|href)="([^"#?:$]+)"', text))):
            dep = os.path.normpath(os.path.join(here, ref))
            if dep.startswith('..') or dep in self.written:
                continue
            if dep == os.path.join('labs', 'season.js'):
                self.write(dep, self.with_site(open(os.path.join(VIDEO, dep), encoding='utf-8').read(), dep))
            elif not dep.endswith('.html'):
                self.write(dep, open(os.path.join(VIDEO, dep), 'rb').read())


def highlight(src):
    """Python source -> HTML with comments, strings, numbers and keywords coloured."""
    lines, out, last = src.splitlines(keepends=True), [], (1, 0)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    pos = lambda rc: offs[rc[0] - 1] + rc[1]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            a, b = pos(tok.start), pos(tok.end)
            out.append(html.escape(src[pos(last):a]))
            cls = {tokenize.COMMENT: 'c', tokenize.STRING: 's', tokenize.NUMBER: 'n'}.get(tok.type)
            if tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
                cls = 'k'
            text = html.escape(src[a:b])
            out.append(f'<span class="{cls}">{text}</span>' if cls else text)
            last = tok.end
    except (tokenize.TokenError, IndentationError):
        return html.escape(src)
    out.append(html.escape(src[pos(last):]))
    return ''.join(out)


CODE_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{file} · Generative Modelling</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{ --canvas:#13100e; --surface:#1c1816; --border:#3a3432; --chalk:#ede6db; --stone:#a89f97; --dust:#6b6360; --amber:#d4a853; --sage:#8fa58a; --rose:#c98a8a; --terracotta:#c4654a; }}
  body {{ margin:0; background:var(--canvas); color:var(--chalk); font-family:Inter,system-ui,sans-serif; }}
  main {{ max-width:980px; margin:0 auto; padding:28px 16px 60px; }}
  nav {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-family:'JetBrains Mono',monospace; font-size:12.5px; margin-bottom:18px; }}
  nav a {{ color:var(--stone); text-decoration:none; border-bottom:1px solid var(--border); }} nav a:hover {{ color:var(--chalk); border-color:var(--terracotta); }}
  h1 {{ font-family:'JetBrains Mono',monospace; font-weight:500; font-size:22px; margin:0 0 6px; }}
  p {{ color:var(--stone); font-size:14px; line-height:1.55; margin:0 0 16px; }} p a {{ color:var(--amber); }}
  pre {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; overflow-x:auto; font:13px/1.55 'JetBrains Mono',monospace; tab-size:4; }}
  .c {{ color:var(--dust); font-style:italic; }} .s {{ color:var(--sage); }} .n {{ color:var(--rose); }} .k {{ color:var(--amber); }}
</style></head>
<body><main>
<nav>{nav}</nav>
<h1>{file}</h1>
<p>{about} Plain numpy, backprop written by hand. <a href="{raw}" download>download {file}</a>{uses}</p>
<pre><code>{code}</code></pre>
</main></body></html>
"""


def local_imports(file):
    src = open(os.path.join(CODE, file), encoding='utf-8').read()
    mods = re.findall(r'^\s*(?:from\s+(\w+)\s+import|import\s+(\w+))', src, re.M)
    return sorted({m + '.py' for pair in mods for m in pair if m and os.path.exists(os.path.join(CODE, m + '.py'))} - {file})


def build_code(site, meta):
    """One page per published code file, plus the local modules it imports (nn.py and friends)."""
    owner = {m['code']: n for n, m in meta.items()}
    todo = [m['code'] for n, m in meta.items() if site.public(n, 'code')]   # plus what they import, below
    done = set()
    while todo:
        file = todo.pop()
        if file in done:
            continue
        done.add(file)
        uses = local_imports(file)
        todo += uses
        stem, n = file[:-3], owner.get(file)
        links = [('season hub', '../')]
        if n and site.public(n, 'ep'):
            links.insert(0, (f'◂ episode {n:02d} · {meta[n]["title"]}', f'../episodes/{meta[n]["page"][:-5]}'))
        if n and site.public(n, 'lab'):
            links.append((f'▶ playground · {meta[n]["labName"]}', f'../labs/{meta[n]["lab"][:-5]}'))
        about = f'The model behind episode {n:02d}, {meta[n]["title"]}.' if n else 'Shared building blocks for the season\'s models.'
        use = ''.join(f' · uses <a href="{u[:-3]}">{u}</a>' for u in uses)
        src = open(os.path.join(CODE, file), encoding='utf-8').read()
        site.write(f'code/{file}', src)
        site.write(f'code/{stem}.html', CODE_PAGE.format(file=file, raw=file, about=about, uses=use, code=highlight(src),
                                                         nav=''.join(f'<a href="{h}">{html.escape(t)}</a>' for t, h in links)))


NOT_FOUND = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not here yet · Generative Modelling</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#13100e;color:#ede6db;font-family:Georgia,serif;text-align:center;padding:16px}
p{color:#a89f97;font-family:system-ui,sans-serif;font-size:15px}a{color:#d4a853}</style></head>
<body><div><h1 style="font-weight:400;font-style:italic">This page isn&rsquo;t here (yet).</h1>
<p>It may not be published, or the address may have changed.<br><a href="/">Back to the season</a></p></div></body></html>
"""


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('public', 'private'):
        sys.exit(__doc__)
    target, plan, meta = sys.argv[1], load_plan(), season_meta()
    site = Site(target, plan)
    shutil.rmtree(DIST, ignore_errors=True)

    site.copy_page('index.html')
    for n, m in meta.items():
        if site.public(n, 'ep'):
            site.write(f'posters/ep{n:02d}.jpg', open(os.path.join(VIDEO, 'posters', f'ep{n:02d}.jpg'), 'rb').read())
        if site.public(n, 'ep'):
            site.copy_page(f'episodes/{m["page"]}')
        if site.public(n, 'lab'):
            site.copy_page(f'labs/{m["lab"]}')
    build_code(site, meta)

    # short links: /3 opens episode 3, /play/3 its playground
    redirects = [f'/{n} /episodes/{m["page"][:-5]} 302' for n, m in meta.items() if site.public(n, 'ep')]
    redirects += [f'/play/{n} /labs/{m["lab"][:-5]} 302' for n, m in meta.items() if site.public(n, 'lab')]
    site.write('_redirects', '\n'.join(redirects) + '\n')
    headers = ['/*', '  X-Content-Type-Options: nosniff', '  Referrer-Policy: strict-origin-when-cross-origin']
    if target == 'private':
        headers.append('  X-Robots-Tag: noindex, nofollow')
    headers += ['/code/*.py', '  Content-Type: text/plain; charset=utf-8']
    site.write('_headers', '\n'.join(headers) + '\n')
    site.write('robots.txt', 'User-agent: *\n' + ('Disallow: /\n' if target == 'private' else 'Allow: /\n'))
    site.write('404.html', NOT_FOUND)

    # what went where, for the build log
    print(f'{target} site -> dist/  ({len(site.written)} files, {sum(os.path.getsize(os.path.join(DIST, f)) for f in site.written) / 2**20:.0f} MiB)')
    for n, m in meta.items():
        p = plan[n]
        print(f'  ep {n:02d} {m["title"]:<28} episode {p["ep"]:<8} playground {p["lab"]:<8} code {p["code"]}')
    if target == 'private':
        print('  (private: every part is included; the states above are what the public site shows)')


if __name__ == '__main__':
    main()
