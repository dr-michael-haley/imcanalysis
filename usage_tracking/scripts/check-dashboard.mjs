// Read-only smoke test against a running dashboard, including production.
import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from '../../web_viewer/node_modules/playwright/index.mjs';

const base = process.env.SBT_USAGE_URL || 'https://usage.spatialbiology.app';
const output = new URL('../test-results/', import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome' });
try {
  for (const [size, width, height] of [['desktop', 1280, 900], ['mobile', 375, 812]]) {
    const page = await browser.newPage({ viewport: { width, height } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(base);
    await page.locator('#report').waitFor();
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('#apps tr').count(), 3);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: fileURLToPath(new URL('dashboard-' + size + '.png', output)), fullPage: true });
    await page.getByRole('combobox', { name: 'App', exact: true }).selectOption('mcd');
    await page.getByRole('combobox', { name: 'Affiliation' }).selectOption('external');
    await page.getByRole('combobox', { name: 'Reporting period' }).selectOption('7');
    await page.getByRole('button', { name: 'Apply filters' }).click();
    await page.waitForFunction(() => document.querySelector('#apps')?.rows.length === 1 && !document.querySelector('#report').hidden);
    assert.match(await page.locator('#apps').innerText(), /MCD Explorer/);
    assert.equal(await page.locator('#days tr').count(), 7);
    await page.getByRole('combobox', { name: 'Measure' }).selectOption('figure_exported');
    await page.getByText('Daily totals', { exact: true }).click();
    const downloaded = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download daily CSV' }).click();
    const csv = await downloaded;
    assert.match(csv.suggestedFilename(), /daily-mcd-external/);
    assert.equal((await readFile(await csv.path(), 'utf8')).trimEnd().split('\r\n').length, 8);
    await page.route('**/api/v1/usage?*', route => route.fulfill({ status: 503, body: '{}' }));
    await page.getByRole('button', { name: 'Apply filters' }).click();
    await page.getByRole('alert').waitFor();
    await page.unroute('**/api/v1/usage?*');
    await page.getByRole('button', { name: 'Try again' }).click();
    await page.locator('#report').waitFor();
    assert.deepEqual(errors, []);
    await page.close();
    console.log(`${size}: filters, chart, daily totals, CSV and retry passed`);
  }
} finally { await browser.close(); }
