# The website: generative.grasp.how

Two static sites built from this repo by Cloudflare Pages. No Workers and no Pages Functions; nothing runs on a server.

| site | address | who can open it | what it shows |
|---|---|---|---|
| public | `generative.grasp.how` | everyone | only what [`publish.json`](../publish.json) marks public |
| private | `generative-private.pages.dev` (or `generative-private.grasp.how`, see below) | you and the people you invite (Cloudflare Access) | everything, with badges showing what the public sees |

## Choosing what is public

Edit [`publish.json`](../publish.json) (on GitHub, the pencil icon works fine), commit to `main`, and both sites rebuild in about a minute. Each episode has three parts, and each part is set on its own:

| part | what it is |
|---|---|
| `episode` | the video page (`/episodes/…`, short link `/3`) |
| `playground` | the playground (`/labs/…`, short link `/play/3`) |
| `code` | the model's source, shown as a page (`/code/vae`) with a download link |

| state | on the public site |
|---|---|
| `public` | live |
| `redacted` | listed as "coming soon": title and question only. The page itself is not uploaded, so its address shows a 404 |
| `hidden` | not uploaded and not mentioned anywhere: no card, no dot on the season map, no link from other episodes |

Examples:

```json
"1": { "title": "Generative Modelling", "episode": "public",   "playground": "public",   "code": "public" },
"2": { "title": "Autoregressive Generation", "episode": "public", "playground": "redacted", "code": "hidden" },
"3": { "title": "Autoencoders & VAEs", "episode": "redacted", "playground": "redacted", "code": "redacted" },
```

- **To publish** a part, set it to `public`.
- **To unpublish** a part, set it to `redacted` to keep a "coming soon" card, or `hidden` to remove every trace. See [Unpublishing](#unpublishing) for what this does and doesn't remove.
- **Links between pages follow the plan.** A neighbouring episode that is redacted shows as "ep 03 · … · soon" in the link bar. One that is hidden is left out.
- **A public code file brings its imports.** `diffusion.py` imports `nn.py` and `vae.py`, so publishing it publishes those two as well. Otherwise the page would be incomplete.
- **Mistakes fail safe.** A misspelt state stops the build, so the site stays as it was. An episode or part missing from the file counts as `hidden`.

To see what a plan produces before pushing:

```bash
python3 site/build.py public     # prints each part's state, writes dist/
cd dist && python3 -m http.server 8000   # rough preview; Pages also serves /episodes/x for x.html
```

## One-time setup in Cloudflare

`grasp.how`'s DNS is hosted outside Cloudflare, which works fine for the public site: a CNAME record at your DNS provider points `generative` to the Pages project.

The only limit is that Cloudflare Access can lock a custom domain only if that domain's DNS is on Cloudflare. So the private site lives at its `pages.dev` address, which Access can lock without any DNS changes. If you later move `grasp.how`'s nameservers to Cloudflare (free), you can add `generative-private.grasp.how` as well. Check that every existing record (e.g. `diffusion`, email) is carried over before switching.

### 1. The public project
1. Go to **Workers & Pages → Create → Pages → Connect to Git**. Pick the **Pages** option, not Workers, and choose `darkwebber/generative-modelling`. The first time, allow Cloudflare's GitHub app to access the repo.
2. Enter the build settings:
   - Production branch: `main`
   - Framework preset: None
   - Build command: `python3 site/build.py public`
   - Build output directory: `dist`
3. Save and deploy.
4. Under **Custom domains**, add `generative.grasp.how` and choose **My DNS provider → Begin CNAME setup**. At your DNS provider, add the record it shows: CNAME `generative` → `generative-grasp.pages.dev`. Then wait for **Active**; this takes minutes to a few hours.
5. Under **Settings → General**, enable the **access policy for preview deployments**. Every deployment also gets its own address (`<id>.<project>.pages.dev`), and old deployments keep whatever they had when they were built, so this keeps unpublished work out of view.

### 2. The private project
1. Create a second Pages project from the **same repo** and branch, with:
   - Build command: `python3 site/build.py private`
   - Build output directory: `dist`
   - Project name: e.g. `generative-private`. The `pages.dev` address takes this name.
2. Skip the custom domain while `grasp.how`'s DNS is outside Cloudflare (see above).
3. Under **Settings → Builds → Branch control**, turn off preview deployments. The private site only needs `main`.

### 3. Lock the private site (Cloudflare Access, free for up to 50 people)
1. Go to **Zero Trust → Access → Applications → Add an application → Self-hosted**.
2. Add every address the private site can be reached at:
   - `generative-private.pages.dev`, using your project's actual `pages.dev` name
   - `*.generative-private.pages.dev`
   - `generative-private.grasp.how`, only once `grasp.how` is on Cloudflare DNS

   Enabling the project's **Access policy** (Settings → General) creates this application for you, covering the `*.` address. Edit it to add the plain `generative-private.pages.dev` address too.

   Leaving out the `pages.dev` addresses would leave a way round the lock.
3. Add a policy: **Allow**, *Include → Emails*, followed by your address and the people you invite.
4. For the login method, use **One-time PIN**: people type their email and get a code, so no accounts are needed. A session length of a month is comfortable.
5. **To check it works,** open `generative-private.pages.dev` in a private window. You should see the Cloudflare sign-in page, and nothing from the season until you enter a code.

**Inviting or removing someone** means editing the email list in that policy. The change takes effect at their next sign-in.

## Unpublishing

Setting a part to `redacted` or `hidden` removes it from the next deployment. From then on, its address on `generative.grasp.how` shows the "not here yet" page.

What that doesn't reach:
- **Older deployments** still contain it at their own addresses (`<id>.<project>.pages.dev`).
  - The preview access policy from setup step 1.5 keeps those addresses to you.
  - To remove it for good, delete those deployments under **Deployments** (⋯ → Delete).
- **Copies people already have** stay with them, whether they downloaded the page or it is sitting in a cache.

## What the build does

`site/build.py` is standard-library Python, run by the Pages build image. It copies each included page from `video/` together with the files it loads: episode data, soundtrack, playground models. Anything not included is never uploaded.

It also writes these files:
- `labs/season.js` and each episode's link bar, with the plan filled in (`const SITE = …`), so links point to this site's own pages.
- `code/*.html` pages for the published code.
- `_redirects` for the short links `/N` and `/play/N`.
- `_headers`: the private site sends `noindex`.
- `robots.txt` and `404.html`.

Pages has a 25 MiB limit per file; the build checks it (the largest file today is about 10 MB).

Outside the website, the pages still work as before. Opened from the repo, they link to each other's files; published as claude.ai artifacts, they use the artifact addresses in `season.js`.
