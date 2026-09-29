// Evaluate JS expressions inside a rendered episode page and print the results as JSON.
// Used by the audio scripts to read data-dependent timings (scene lengths, generated names, …).
// Usage (from video/): node src/probe.mjs <episode.html> '["SC.map(s => s.dur)", "DEMO.name"]'
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const [, , html, exprs] = process.argv;
const here = process.cwd(), fontDir = path.join(here, '.fonts');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const page = await browser.newPage({ ignoreHTTPSErrors: true });
if (fs.existsSync(path.join(fontDir, 'gf.css'))) {
  await page.route('https://fonts.googleapis.com/**', r => r.fulfill({ path: path.join(fontDir, 'gf.css'), contentType: 'text/css' }));
  await page.route('https://fonts.gstatic.com/**', r => r.fulfill({ path: path.join(fontDir, path.basename(new URL(r.request().url()).pathname)), contentType: 'font/woff2' }));
}
await page.goto(pathToFileURL(path.join(here, html)).href + '?capture');
await page.waitForFunction(() => window.READY === true, null, { timeout: 60000 });
const out = await page.evaluate(list => list.map(e => (0, eval)(e)), JSON.parse(exprs));
console.log(JSON.stringify(out));
await browser.close();
