import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import { mkdir, readFile } from 'node:fs/promises';
const base = process.env.SBT_VIEWER_URL || 'http://127.0.0.1:8765/';
const input = name => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const ready = (page, roi = 'ROI_A') => page.waitForFunction(name => document.getElementById('status').textContent.startsWith(`${name} ready`), roi);
const tab = (page, name) => page.locator(`#tab-${name}`).click();
async function save(page) {
  await tab(page, 'setup');
  if (!await page.locator('#workspace-file-tools').evaluate(node => node.open)) await page.locator('#workspace-file-tools > summary').click();
  const pending = page.waitForEvent('download'); await page.locator('#save-workspace').click();
  return JSON.parse(await readFile(await (await pending).path(), 'utf8'));
}
async function load(page, workspace, root = '') {
  await tab(page, 'setup');
  await page.locator('#workspace-file').setInputFiles({ name: 'review.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(workspace)) });
  await page.waitForFunction(() => !document.getElementById('workspace-root-label').hidden);
  await page.locator('#workspace-root').setInputFiles(input(root));
  await page.waitForFunction(() => document.getElementById('workspace-status').textContent.startsWith('Workspace restored'), null, { timeout: 30000 });
}

test('four independent named views, linked pan/zoom, ROI recipes and root-folder workspace restoration', { timeout: 120000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base);
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks')); await ready(page);
    await page.locator('#image-files').setInputFiles(input('six_channels')); await ready(page);
    await tab(page, 'views');
    assert.equal(await page.locator('#tab-views').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#tab-setup').getAttribute('aria-selected'), 'false');
    await page.locator('#view-name').fill('Markers');
    await page.locator('#overlay-style').selectOption('off');
    await page.locator('#low-0').fill('10'); await page.locator('#low-0').dispatchEvent('change');
    await page.locator('#add-view').click(); await ready(page);
    await page.locator('#view-name').fill('Cells');
    await page.locator('#view-menu-trigger').click(); await page.locator('#quick-images').click(); await page.locator('#view-menu-trigger').click();
    await page.locator('#overlay-style').selectOption('fill');
    await page.locator('#add-view').click(); await ready(page);
    await page.locator('#view-name').fill('CD20');
    await page.locator('#colour-by').selectOption('var:CD20'); await ready(page);
    await page.locator('#expression-source').selectOption('layers/counts'); await ready(page);
    await page.locator('#add-view').click(); await ready(page);
    await page.locator('#view-name').fill('Context');
    await page.locator('#view-menu-trigger').click(); await page.locator('#quick-images').click(); await page.locator('#view-menu-trigger').click();
    assert.equal(await page.locator('.view-pane').count(), 4);
    assert.equal(await page.locator('#add-view').isDisabled(), true);
    assert.equal(await page.locator('.view-pane.selected').getAttribute('data-view'), '3');
    const before = await page.locator('.tissue-canvas').evaluateAll(nodes => nodes.map(node => node.toDataURL()));
    assert.notEqual(before[0], before[1]); assert.notEqual(before[1], before[2]);
    await page.locator('#tissue-1').hover(); await page.mouse.wheel(0, -120);
    await page.waitForTimeout(100);
    const zoomed = await page.locator('.tissue-canvas').evaluateAll(nodes => nodes.map(node => node.toDataURL()));
    for (let i = 0; i < 4; i++) assert.notEqual(zoomed[i], before[i], `Pane ${i + 1} zooms with the others`);
    assert.equal(await page.locator('.view-pane.selected').getAttribute('data-view'), '1');
    const rect = await page.locator('#tissue-2').boundingBox();
    await page.mouse.move(rect.x + rect.width / 2, rect.y + rect.height / 2); await page.mouse.down(); await page.mouse.move(rect.x + rect.width / 2 + 27, rect.y + rect.height / 2 + 13); await page.mouse.up();
    const panned = await page.locator('.tissue-canvas').evaluateAll(nodes => nodes.map(node => node.toDataURL()));
    for (let i = 0; i < 4; i++) assert.notEqual(panned[i], zoomed[i]);
    await page.locator('#quick-next').click(); await ready(page, 'ROI_A_2');
    await page.locator('#view-select').selectOption('0');
    assert.equal(await page.locator('#low-0').inputValue(), '10');
    await page.locator('#view-select').selectOption('2');
    assert.equal(await page.locator('#expression-source').inputValue(), 'layers/counts');
    await page.waitForTimeout(900);
    const saved = await save(page);
    assert.deepEqual(saved.view.presets.map(view => view.name), ['Markers', 'Cells', 'CD20', 'Context']);
    assert.equal(saved.version, 11);
    const rendered = await page.locator('.tissue-canvas').evaluateAll(nodes => nodes.map(node => node.toDataURL()));
    await page.reload(); await load(page, saved);
    const again = await save(page);
    assert.deepEqual(again.view.presets, saved.view.presets);
    assert.deepEqual(again.view.camera, saved.view.camera);
    assert.equal(again.view.selectedView, saved.view.selectedView);
    assert.deepEqual(await page.locator('.tissue-canvas').evaluateAll(nodes => nodes.map(node => node.toDataURL())), rendered);
    await tab(page, 'views');
    await page.waitForTimeout(200);
    await mkdir('test-results', { recursive: true }); await page.screenshot({ path: 'test-results/four-views.png', fullPage: true });
    assert.deepEqual(errors, []);
  } catch (error) {
    console.error(await page.locator('#status').textContent(), await page.locator('#warning').textContent(), await page.locator('#workspace-error').textContent(), errors);
    throw error;
  } finally { await browser.close(); }
});

test('ROI ranking, top-X shortcuts, recipe-specific visits, reset and legacy workspace migration', { timeout: 90000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = await browser.newPage();
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base);
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks')); await ready(page);
    await tab(page, 'rois');
    await page.locator('#order-mode').selectOption('population');
    await page.locator('#order-field').selectOption('population');
    await page.locator('#order-value').selectOption(JSON.stringify('B cell'));
    await page.locator('#apply-order').click();
    await page.waitForFunction(() => document.getElementById('order-status').textContent.includes('B cell'));
    assert.deepEqual(await page.locator('#roi option').evaluateAll(nodes => nodes.map(node => node.value)), ['ROI_A_2', 'ROI_A']);
    assert.match(await page.locator('.roi-shortcut').first().textContent(), /8$/);
    await page.locator('#top-rois').fill('1'); await page.locator('#top-rois').dispatchEvent('change');
    assert.equal(await page.locator('.roi-shortcut').count(), 1);
    await page.locator('.roi-shortcut').click(); await ready(page, 'ROI_A_2');
    await page.waitForFunction(() => document.querySelector('.roi-shortcut').classList.contains('visited'));
    await tab(page, 'views'); await page.locator('#add-view').click(); await ready(page, 'ROI_A_2');
    await page.locator('#view-menu-trigger').click(); await page.locator('#quick-overlay').click(); await tab(page, 'rois');
    await page.waitForFunction(() => document.querySelector('.roi-shortcut').classList.contains('partial'));
    await page.waitForFunction(() => document.querySelector('.roi-shortcut').classList.contains('visited'));
    await page.locator('#reset-visits').click();
    assert.equal(await page.locator('.roi-shortcut.unvisited').count(), 1);
    await page.locator('#order-mode').selectOption('marker'); await page.locator('#order-field').selectOption('CD20');
    await page.locator('#order-source').selectOption('layers/counts'); await page.locator('#apply-order').click();
    await page.waitForFunction(() => document.getElementById('order-status').textContent.includes('mean CD20'));
    assert.match(await page.locator('.roi-shortcut').textContent(), /98$/);
    const saved = await save(page);
    await page.reload(); await load(page, saved);
    await tab(page, 'rois'); assert.match(await page.locator('#order-status').textContent(), /mean CD20.*layers\/counts.*descending/);
    await page.locator('#reset-order').click();
    assert.deepEqual(await page.locator('#roi option').evaluateAll(nodes => nodes.map(node => node.value)), ['ROI_A', 'ROI_A_2']);
    const legacy = { ...saved, version: 1 };
    await page.reload(); await load(page, legacy);
    assert.equal(await page.locator('.view-pane').count(), 1);
    assert.deepEqual(errors, []);
  } catch (error) {
    console.error('Ranking failure:', await page.locator('#roi').inputValue(), await page.locator('#status').textContent(), await page.locator('#warning').textContent(), await page.locator('#order-status').textContent(), errors);
    throw error;
  } finally { await browser.close(); }
});
