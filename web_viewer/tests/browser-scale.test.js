// Opt-in: python tests/make_scale_fixture.py, then node --test tests/browser-scale.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { tabbedPage } from './tabbed-page.js';

test('five million cells: identity indexing, ROI navigation, marker reads and cell metadata', { timeout: 180000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = tabbedPage(await browser.newPage());
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const ready = roi => page.waitForFunction(name => document.getElementById('status').textContent.startsWith(`${name} ready`), roi, { timeout: 120000 });
  try {
    await page.goto(process.env.SBT_VIEWER_URL || 'http://127.0.0.1:8765/');
    const start = performance.now();
    await page.locator('#anndata-file').setInputFiles(fileURLToPath(new URL('../fixtures/scale.h5ad', import.meta.url)));
    await ready('ROI_0000');
    const indexingMs = performance.now() - start;
    assert.equal(await page.locator('#roi option').count(), 1000);
    assert.match(await page.locator('#roi-summary').textContent(), /5,000 cells/);
    assert.match(await page.locator('#legend').textContent(), /T cell/);
    await page.locator('#colour-by').selectOption('var:row_number');
    await ready('ROI_0000');
    assert.match(await page.locator('.range-label').textContent(), /^0\.000 — 4999/);
    const navigate = performance.now();
    await page.locator('#roi').selectOption('ROI_0999');
    await ready('ROI_0999');
    const navigationMs = performance.now() - navigate;
    assert.equal(await page.locator('.range-label').textContent(), '4.995e+6 — 5.000e+6');
    // The embedding links a point from the last ROI back to its original obs row.
    await page.locator('#embedding-canvas').click({ position: { x: 14, y: 14 } });
    await page.waitForFunction(() => document.getElementById('cell-details').textContent.includes('row_number (X)'));
    const fields = await page.locator('#cell-details dd').allTextContents();
    assert.ok(fields.includes('ROI_0999'));
    const row = Number(fields[0]);
    assert.ok(row >= 4_995_000 && row < 5_000_000);
    assert.equal(Number(fields.at(-1)), row);
    const rankingStart = performance.now();
    await page.locator('#order-mode').selectOption('marker');
    await page.locator('#order-field').selectOption('row_number');
    await page.locator('#apply-order').click();
    await page.waitForFunction(() => document.getElementById('order-status').textContent.includes('mean row_number'), null, { timeout: 120000 });
    assert.equal(await page.locator('#roi option').first().getAttribute('value'), 'ROI_0999');
    const rankingMs = performance.now() - rankingStart;
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ cells: 5_000_000, rois: 1000, indexingMs: Math.round(indexingMs), navigationMs: Math.round(navigationMs), rankingMs: Math.round(rankingMs) }));
  } catch (error) {
    console.error('Scale check status:', await page.locator('#status').textContent(), '\nWarning:', await page.locator('#warning').textContent());
    throw error;
  } finally { await browser.close(); }
});
