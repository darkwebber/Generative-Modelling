// Render an episode (a deterministic canvas animation) to MP4, frame by frame.
// Usage: node render.mjs <episodes/page.html> [fps=30] [out.mp4 = renders/<page>.mp4] [start=0] [end=DURATION]
//   e.g. node render.mjs episodes/ep01-generative-modelling.html
//        node render.mjs episodes/ep02-autoregressive.html 30 renders/ep02.mp4 120 170   # just one section
//   (python src/make.py N --render does this, then adds the soundtrack)
// Env:   WORKERS=n   render n segments in parallel, then join them (default: 1)
//        FFMPEG=/path/to/ffmpeg   CHROMIUM_PATH=/path/to/chrome
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = process.argv[2] || 'episodes/ep01-generative-modelling.html';
const fps = Number(process.argv[3] || 30);
const out = path.resolve(here, process.argv[4] || path.join('renders', path.basename(html).replace(/\.html$/, '.mp4')));
fs.mkdirSync(path.dirname(out), { recursive: true });
const FFMPEG = process.env.FFMPEG || 'ffmpeg';
const workers = Number(process.env.WORKERS || 1);
const run = (cmd, args, opts = {}) => new Promise((res, rej) => { const p = spawn(cmd, args, { stdio: 'inherit', ...opts }); p.on('close', c => c === 0 ? res() : rej(new Error(`${cmd} exited ${c}`))); });

async function openPage() {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
  const page = await browser.newPage({ ignoreHTTPSErrors: true, viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  // Serve Google Fonts from a local cache (video/.fonts, filled by fetch-fonts.sh) when present —
  // headless Chromium behind a proxy often can't reach fonts.gstatic.com itself.
  const fontDir = path.join(here, '.fonts');
  if (fs.existsSync(path.join(fontDir, 'gf.css'))) {
    await page.route('https://fonts.googleapis.com/**', r => r.fulfill({ path: path.join(fontDir, 'gf.css'), contentType: 'text/css' }));
    await page.route('https://fonts.gstatic.com/**', r => r.fulfill({ path: path.join(fontDir, path.basename(new URL(r.request().url()).pathname)), contentType: 'font/woff2' }));
  }
  page.on('pageerror', e => console.error('page error:', e.message));
  await page.goto(pathToFileURL(path.join(here, html)).href + '?capture');
  await page.waitForFunction(() => window.READY === true, null, { timeout: 60000 });
  return { browser, page };
}

const { browser, page } = await openPage();
const duration = await page.evaluate(() => window.DURATION);
const f0 = Math.round(Number(process.argv[5] || 0) * fps), f1 = Math.round(Number(process.argv[6] || duration) * fps);

if (workers > 1) {
  // split [f0, f1) into segments, render each in its own process, then concatenate losslessly
  await browser.close();
  const per = Math.ceil((f1 - f0) / workers), parts = [];
  const jobs = [];
  for (let w = 0; w < workers; w++) {
    const a = f0 + w * per, b = Math.min(f1, a + per); if (a >= b) break;
    const part = `${out}.part${w}.mp4`; parts.push(part);
    jobs.push(run(process.execPath, [fileURLToPath(import.meta.url), html, String(fps), part, String(a / fps), String(b / fps)], { env: { ...process.env, WORKERS: '1' } }));
  }
  await Promise.all(jobs);
  const list = `${out}.parts.txt`; fs.writeFileSync(list, parts.map(p => `file '${p}'`).join('\n'));
  await run(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', out]);
  parts.forEach(p => fs.unlinkSync(p)); fs.unlinkSync(list);
  console.log('wrote', out);
  process.exit(0);
}

const ff = spawn(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
  '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', String(fps), '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });

for (let f = f0; f < f1; f++) {
  // draw the frame and pull it straight out of the canvas (much faster than a screenshot)
  const b64 = await page.evaluate(T => { window.renderAt(T); return document.getElementById('c').toDataURL('image/jpeg', 0.94).slice(23); }, f / fps);
  if (!ff.stdin.write(Buffer.from(b64, 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
  if ((f - f0) % (fps * 10) === 0) console.log(`${path.basename(out)}: frame ${f - f0}/${f1 - f0}  (${(f / fps).toFixed(0)}s)`);
}
ff.stdin.end();
await new Promise(r => ff.on('close', r));
await browser.close();
console.log('wrote', out);
