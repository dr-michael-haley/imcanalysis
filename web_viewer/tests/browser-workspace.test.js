import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { tabbedPage } from './tabbed-page.js';
const base = process.env.SBT_VIEWER_URL || 'http://127.0.0.1:8765/';
const input = name => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const ready = (page, roi = 'ROI_A') => page.waitForFunction(name => document.getElementById('status').textContent.startsWith(`${name} ready`), roi);
const restored = page => page.waitForFunction(() => document.getElementById('workspace-status').textContent.startsWith('Workspace restored'));
async function save(page) {
  const download = page.waitForEvent('download'); await page.locator('#save-workspace').click();
  const file = await download; assert.match(file.suggestedFilename(), /\.workspace\.json$/);
  return JSON.parse(await readFile(await file.path(), 'utf8'));
}
const load = (page, payload) => page.locator('#workspace-file').setInputFiles({ name: 'review.workspace.json', mimeType: 'application/json', buffer: Buffer.from(typeof payload === 'string' ? payload : JSON.stringify(payload)) });
const pixels = page => page.locator('#tissue').evaluate(canvas => canvas.toDataURL());

test('workspace restores dictionary, six channels, selected cell, panels and camera from one root', { timeout: 90000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = tabbedPage(await browser.newPage({ viewport: { width: 1440, height: 1100 } }));
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message)); page.on('request', request => requests.push({ method: request.method(), url: request.url() }));
  try {
    await page.goto(base); assert.equal(await page.locator('#save-workspace').isDisabled(), true);
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks')); await ready(page);
    await page.locator('#image-files').setInputFiles(input('six_channels')); await ready(page);
    for (let i = 3; i < 6; i++) { await page.locator('#add-channel').click(); await ready(page); }
    await page.locator('#normalization-file').setInputFiles(input('normalization_valid.csv')); await ready(page);
    await page.locator('#next-roi').click(); await ready(page, 'ROI_A_2');
    await page.locator('#low-0').fill('0.15'); await page.locator('#low-0').dispatchEvent('change');
    await page.locator('#high-0').fill('0.8'); await page.locator('#high-0').dispatchEvent('change');
    await page.locator('#channel-colour-4').fill('#8f41e9'); await page.locator('#channel-colour-4').dispatchEvent('input');
    await page.locator('#channel-enabled-2').uncheck(); await ready(page, 'ROI_A_2');
    await page.locator('#population').selectOption(JSON.stringify('B cell'));
    const rect = await page.locator('#tissue').boundingBox(), scale = Math.min((rect.width - 48) / 256, (rect.height - 48) / 256);
    await page.locator('#tissue').click({ position: { x: rect.width / 2 - 96 * scale, y: rect.height / 2 - 96 * scale } });
    await page.waitForFunction(() => document.getElementById('cell-details').textContent.includes('cell_1'));
    await page.locator('#toggle-overlay').click(); await page.locator('#zoom-in').click();
    await page.locator('#review-panels > summary').click(); await page.locator('#section-channels > summary').click();
    await page.waitForTimeout(900);
    const saved = await save(page), before = await pixels(page);
    assert.equal(saved.view.selected, 300); assert.deepEqual(saved.view.normalization.dictionary, { Ch1: { vmax: 180, lower_threshold: 20 } });
    await page.reload(); await load(page, saved);
    await page.waitForFunction(() => !document.getElementById('workspace-root-label').hidden);
    assert.equal(await page.locator('#save-workspace').isDisabled(), true);
    await page.locator('#workspace-root').setInputFiles(input('')); await restored(page);
    const again = await save(page);
    for (const key of ['presets', 'normalization', 'camera', 'panels']) assert.deepEqual(again.view[key], saved.view[key]);
    assert.equal(await pixels(page), before); assert.equal(await page.locator('#cell-badge').textContent(), 'Object 300');
    assert.deepEqual(errors, []); assert.ok(requests.every(request => request.method === 'GET' && new URL(request.url).origin === new URL(base).origin));
  } finally { await browser.close(); }
});

test('invalid workspace and wrong root preserve the view; cancel and multipage restore remain usable', { timeout: 60000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = tabbedPage(await browser.newPage());
  try {
    await page.goto(base);
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#image-files').setInputFiles(input('stacks')); await ready(page);
    assert.equal(await page.locator('#plane-0').count(), 0);
    await page.locator('#remove-channel-1').click(); await ready(page);
    await page.locator('#channel-2').selectOption(''); await ready(page);
    const saved = await save(page), before = await pixels(page);
    for (const payload of ['{broken', { ...saved, version: 99 }, { ...saved, view: { ...saved.view, channels: [] } }]) {
      await load(page, payload); await page.waitForFunction(() => !document.getElementById('workspace-error').hidden);
      assert.equal(await pixels(page), before); assert.equal(await page.locator('#save-workspace').isDisabled(), false);
    }
    await load(page, saved); await page.locator('#workspace-root').setInputFiles(input('images'));
    await page.waitForFunction(() => document.getElementById('workspace-error').textContent.includes('Could not find dense.h5ad'));
    assert.equal(await pixels(page), before);
    await page.locator('#cancel-workspace').click();
    await page.waitForFunction(() => document.getElementById('status').textContent === 'Ready to open a local dataset.');
    assert.equal(await page.locator('#save-workspace').isDisabled(), true);
    await load(page, saved); await page.locator('#workspace-root').setInputFiles(input('')); await restored(page);
    assert.equal(await page.locator('#plane-0').count(), 0); assert.deepEqual((await save(page)).view.presets, saved.view.presets);
  } finally { await browser.close(); }
});
