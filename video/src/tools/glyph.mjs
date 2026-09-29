// Render a word in the series' serif font and print the filled pixels as JSON [[x, y], …] in
// centred units (height ≈ 1). Used by code/export_ep09_assets.py to train the title-generating model.
// Usage (from video/): node src/tools/glyph.mjs "Diffusion" [font px, default 230]
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';

const word = process.argv[2] || 'Diffusion', px = +(process.argv[3] || 230), here = process.cwd(), fontDir = path.join(here, '.fonts');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const page = await browser.newPage();
if (fs.existsSync(path.join(fontDir, 'gf.css'))) {
  await page.route('https://fonts.googleapis.com/**', r => r.fulfill({ path: path.join(fontDir, 'gf.css'), contentType: 'text/css' }));
  await page.route('https://fonts.gstatic.com/**', r => r.fulfill({ path: path.join(fontDir, path.basename(new URL(r.request().url()).pathname)), contentType: 'font/woff2' }));
}
await page.setContent(`<html><head><link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;1,6..72,400&display=swap" rel="stylesheet"></head><body><canvas id="c" width="2400" height="360"></canvas></body></html>`);
await page.evaluate(px => document.fonts.load(`400 ${px}px 'Newsreader'`), px);
await page.waitForTimeout(500);
const pts = await page.evaluate(([w, px]) => {
  const c = document.getElementById('c'), g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, c.width, c.height); g.fillStyle = '#fff'; g.font = `400 ${px}px 'Newsreader'`; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
  g.fillText(w, 1200, 250);
  const d = g.getImageData(0, 0, c.width, c.height).data, out = [];
  for (let y = 0; y < c.height; y += 2) for (let x = 0; x < c.width; x += 2) if (d[(y * c.width + x) * 4] > 128) out.push([(x - 1200) / 180, (170 - y) / 180]);
  return out;
}, [word, px]);
console.log(JSON.stringify(pts));
await browser.close();
