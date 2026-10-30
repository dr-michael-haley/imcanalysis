import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile } from 'node:fs/promises';
import { tabbedPage } from './tabbed-page.js';

const base = process.env.SBT_VIEWER_URL || 'http://127.0.0.1:8765/';
const fixtures = fileURLToPath(new URL('../fixtures/', import.meta.url));
const output = fileURLToPath(new URL('../test-results/', import.meta.url));
const input = name => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
async function fingerprints() {
  const entries = await readdir(fixtures, { recursive: true, withFileTypes: true });
  return Promise.all(entries.filter(entry => entry.isFile()).map(async entry => {
    const path = `${entry.parentPath}/${entry.name}`;
    return [path, createHash('sha256').update(await readFile(path)).digest('hex')];
  }));
}
async function ready(page, roi = 'ROI_A') {
  await page.waitForFunction(name => document.getElementById('status').textContent.startsWith(`${name} ready`), roi, { timeout: 25000 });
}

test('browser opens local inputs, links cells, changes ROI and reads CSC without uploads', { timeout: 120000 }, async () => {
  await mkdir(output, { recursive: true });
  const before = await fingerprints();
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = tabbedPage(await context.newPage());
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push({ url: request.url(), method: request.method() }));
  try {
    await page.goto(base);
    await page.screenshot({ path: `${output}/empty.png`, fullPage: true });
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad'));
    await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks'));
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('256 × 256'));
    await page.locator('#image-files').setInputFiles(input('images'));
    await page.waitForFunction(() => document.getElementById('roi-summary').textContent.includes('3 image files'));
    await ready(page);
    assert.match(await page.locator('#warning').textContent(), /1 mask objects/);
    assert.equal(await page.locator('#channel-0 option').count(), 4);
    assert.match(await page.locator('#legend').textContent(), /T cell/);
    const bounds = await page.locator('#viewport').boundingBox();
    const scale = Math.min((bounds.width - 48) / 256, (bounds.height - 48) / 256);
    await page.locator('#tissue').click({ position: { x: bounds.width / 2 - 96 * scale, y: bounds.height / 2 - 96 * scale } });
    await page.waitForFunction(() => document.getElementById('cell-details').textContent.includes('cell_0'));
    assert.match(await page.locator('#cell-badge').textContent(), /300/);
    await page.screenshot({ path: `${output}/loaded.png`, fullPage: true });
    await page.locator('#colour-by').selectOption('var:CD20');
    await ready(page);
    await page.locator('#expression-source').selectOption('layers/counts');
    await ready(page);
    await page.locator('#tissue').click({ position: { x: bounds.width / 2 - 96 * scale, y: bounds.height / 2 - 96 * scale } });
    await page.waitForFunction(() => document.getElementById('cell-details').textContent.includes('CD20 (layers/counts)'));
    assert.equal(await page.locator('#cell-details dd').last().textContent(), '2');
    await page.locator('#next-roi').click(); await ready(page, 'ROI_A_2');
    assert.equal(await page.locator('#view-title').textContent(), 'ROI_A_2');
    await page.locator('#anndata-file').setInputFiles(input('csc.h5ad')); await ready(page);
    await page.locator('#colour-by').selectOption('var:CD20'); await ready(page);
    assert.equal(await page.locator('.range-label').textContent(), '1.000 — 91.00');
    await page.locator('#anndata-file').setInputFiles(input('lzf.h5ad'));
    await page.waitForFunction(() => document.getElementById('warning').textContent.includes('Unsupported HDF5 compression'));
    // Explicitly reject unsupported/invalid scientific mappings.
    await page.locator('#anndata-file').setInputFiles(input('duplicate.h5ad'));
    await page.waitForFunction(() => document.getElementById('warning').textContent.includes('Duplicate cell identity'));
    assert.equal(await page.locator('#empty-state').isVisible(), true);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.deepEqual(errors, []);
    assert.ok(requests.every(request => request.method === 'GET' && new URL(request.url).origin === new URL(base).origin), JSON.stringify(requests));
    assert.deepEqual(await fingerprints(), before, 'Source files must remain byte-for-byte unchanged.');
  } catch (error) {
    await page.screenshot({ path: `${output}/failure.png`, fullPage: true });
    console.error('Viewer status:', await page.locator('#status').textContent(), '\nWarning:', await page.locator('#warning').textContent());
    throw error;
  } finally { await browser.close(); }
});

test('mismatched rasters fail visibly and new multipage scalar images use their first plane', { timeout: 60000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = tabbedPage(await browser.newPage());
  try {
    await page.goto(base);
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks')); await ready(page);
    await page.locator('#image-files').setInputFiles(input('bad_images'));
    await page.waitForFunction(() => document.getElementById('warning').textContent.includes('dimensions differ'));
    assert.equal(await page.locator('#empty-state').isVisible(), true);
    await page.locator('#image-files').setInputFiles(input('stacks')); await ready(page);
    assert.equal(await page.locator('#plane-0').count(), 0);
    assert.equal(await page.locator('#high-range-0').isVisible(), true);
    // Rapid region changes must finish on the last requested region.
    await page.locator('#roi').selectOption('ROI_A_2');
    await page.locator('#roi').selectOption('ROI_A'); await ready(page);
    assert.equal(await page.locator('#view-title').textContent(), 'ROI_A');
  } finally { await browser.close(); }
});

test('six pure-colour channels compose correctly and collapsible panels preserve the view', { timeout: 90000 }, async () => {
  await mkdir(output, { recursive: true });
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = tabbedPage(await browser.newPage({ viewport: { width: 1440, height: 900 } }));
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const colours = ['#ff0000', '#00ff00', '#0000ff', '#00ffff', '#ff00ff', '#ffff00'];
  const pixels = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]];
  const stripePixel = index => page.locator('#tissue').evaluate(async (canvas, i) => {
    // ResizeObserver refits the canvas after the details panel changes layout.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const rect = document.getElementById('viewport').getBoundingClientRect();
    const scale = Math.min((rect.width - 48) / 256, (rect.height - 48) / 256);
    const x = rect.width / 2 + (48 + i * 32 - 128) * scale;
    return [...canvas.getContext('2d').getImageData(Math.round(x * devicePixelRatio), Math.round(rect.height / 2 * devicePixelRatio), 1, 1).data].slice(0, 3);
  }, index);
  try {
    await page.goto(base);
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks')); await ready(page);
    await page.locator('#image-files').setInputFiles(input('six_channels')); await ready(page);
    await page.locator('#overlay-style').selectOption('off');
    for (let i = 3; i < 6; i++) { await page.locator('#add-channel').click(); await ready(page); }
    assert.equal(await page.locator('#add-channel').isDisabled(), true);
    assert.equal(await page.locator('.channel:visible').count(), 6);
    for (let i = 0; i < 6; i++) {
      assert.equal(await page.locator(`#channel-colour-${i}`).inputValue(), colours[i]);
      assert.deepEqual(await stripePixel(i), pixels[i], `Channel ${i + 1} must contribute its pure colour.`);
    }
    const cyanFile = await page.locator('#channel-3').inputValue();
    await page.locator('#channel-enabled-3').uncheck(); await ready(page);
    assert.deepEqual(await stripePixel(3), [0, 0, 0]);
    assert.equal(await page.locator('#channel-3').inputValue(), cyanFile);
    await page.locator('#channel-enabled-3').check(); await ready(page);
    assert.deepEqual(await stripePixel(3), pixels[3]);
    await page.locator('#remove-channel-1').click(); await ready(page);
    assert.equal(await page.locator('.channel:visible').count(), 5);
    assert.deepEqual(await stripePixel(1), [0, 0, 0]);
    assert.deepEqual(await stripePixel(5), pixels[5]);
    await page.locator('#add-channel').click(); await ready(page);
    assert.deepEqual(await stripePixel(1), pixels[1]);
    await page.locator('#next-roi').click(); await ready(page, 'ROI_A_2');
    for (let i = 0; i < 6; i++) assert.deepEqual(await stripePixel(i), pixels[i]);

    for (const id of ['section-data', 'section-roi', 'section-channels', 'section-normalization', 'section-overlay']) {
      await page.locator(`#${id} > summary`).click();
      assert.equal(await page.locator(`#${id}`).evaluate(node => node.open), false);
    }
    await page.waitForFunction(() => document.getElementById('collapse-sections').textContent === 'Expand all');
    const selection = await page.locator('#channel-5').inputValue();
    await page.locator('#collapse-sections').click();
    assert.equal(await page.locator('#channel-5').inputValue(), selection);
    await page.locator('#section-data > summary').click();
    await page.locator('#section-overlay > summary').click();
    await page.locator('#section-channels > summary').scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${output}/six-channels.png`, fullPage: true });

    const before = await page.locator('#viewport').boundingBox();
    await page.locator('#review-panels > summary').click();
    await page.waitForFunction(height => document.getElementById('viewport').getBoundingClientRect().height > height + 200, before.height);
    assert.deepEqual(await stripePixel(5), pixels[5]);
    assert.ok((await page.locator('#viewport').boundingBox()).height > 700);
    await page.screenshot({ path: `${output}/viewer-expanded.png`, fullPage: true });
    await page.locator('#review-panels > summary').click();
    await page.waitForFunction(height => Math.abs(document.getElementById('viewport').getBoundingClientRect().height - height) < 2, before.height);
    assert.deepEqual(await stripePixel(5), pixels[5]);
    // Folding panels must preserve a user's manual zoom and image centre.
    await page.locator('#zoom-in').click();
    const zoom = await page.locator('#view-scale').textContent();
    const centrePixel = () => page.locator('#tissue').evaluate(async canvas => {
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return [...canvas.getContext('2d').getImageData(Math.floor(canvas.width / 2), Math.floor(canvas.height / 2), 1, 1).data];
    });
    const centre = await centrePixel();
    await page.locator('#review-panels > summary').click();
    assert.deepEqual(await centrePixel(), centre);
    assert.equal(await page.locator('#view-scale').textContent(), zoom);
    await page.locator('#review-panels > summary').click();
    assert.deepEqual(await centrePixel(), centre);
    assert.equal(await page.locator('#view-scale').textContent(), zoom);
    await page.locator('#fit').click();
    assert.equal(await page.evaluate(() => document.querySelector('.workspace').scrollHeight <= document.querySelector('.workspace').clientHeight + 1), true);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: `${output}/compact-mobile.png`, fullPage: true });
    assert.deepEqual(errors, []);
  } catch (error) {
    await page.screenshot({ path: `${output}/layout-failure.png`, fullPage: true });
    throw error;
  } finally { await browser.close(); }
});

test('local dictionaries scale images, preserve AnnData and reject invalid replacement files', { timeout: 60000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = tabbedPage(await browser.newPage({ viewport: { width: 1440, height: 900 } }));
  const before = await fingerprints();
  const requests = [], errors = [];
  page.on('request', request => requests.push({ url: request.url(), method: request.method() }));
  page.on('pageerror', error => errors.push(error.message));
  const redPixel = () => page.locator('#tissue').evaluate(async canvas => {
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const rect = document.getElementById('viewport').getBoundingClientRect();
    const scale = Math.min((rect.width - 48) / 256, (rect.height - 48) / 256);
    return [...canvas.getContext('2d').getImageData(Math.round((rect.width / 2 - 80 * scale) * devicePixelRatio), Math.round(rect.height / 2 * devicePixelRatio), 1, 1).data].slice(0, 3);
  });
  try {
    await page.goto(base);
    assert.equal(await page.locator('.brand-logo').evaluate(image => image.complete && image.naturalWidth > 0), true);
    // Loading before any dataset must work, and defaults should become 0–1.
    await page.locator('#normalization-file').setInputFiles(input('normalization_valid.csv'));
    await page.waitForFunction(() => document.getElementById('normalization-status').textContent.includes('1 marker limits'));
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks')); await ready(page);
    await page.locator('#image-files').setInputFiles(input('six_channels')); await ready(page);
    await page.locator('#overlay-style').selectOption('off');
    for (let i = 3; i < 6; i++) { await page.locator('#add-channel').click(); await ready(page); }
    assert.equal(await page.locator('#low-0').inputValue(), '0');
    assert.equal(await page.locator('#high-0').inputValue(), '1');
    assert.match(await page.locator('#normalization-status').textContent(), /1\/6 visible channels matched; 5 use fallback/);
    assert.deepEqual(await redPixel(), [128, 0, 0]); // (100 - 20) / (180 - 20) = 0.5
    assert.match(await page.locator('#channel-normalization-1').textContent(), /Fallback/);
    await page.locator('#normalization-file').setInputFiles(input('normalization.json'));
    await page.waitForFunction(() => document.getElementById('normalization-name').textContent === 'normalization.json'); await ready(page);
    assert.deepEqual(await redPixel(), [128, 0, 0]);
    await page.locator('#normalization-file').setInputFiles(input('normalization.csv'));
    await page.waitForFunction(() => !document.getElementById('normalization-error').hidden);
    assert.equal(await page.locator('#normalization-name').textContent(), 'normalization.json');
    assert.deepEqual(await redPixel(), [128, 0, 0]);
    await page.locator('#colour-by').selectOption('var:CD20'); await ready(page);
    assert.equal(await page.locator('.range-label').textContent(), '1.000 — 91.00', 'AnnData expression must not be normalised by the image dictionary.');
    await page.locator('#high-0').fill('0.5'); await page.locator('#high-0').dispatchEvent('change');
    assert.deepEqual(await redPixel(), [255, 0, 0]);
    await page.locator('#next-roi').click(); await ready(page, 'ROI_A_2');
    assert.equal(await page.locator('#high-0').inputValue(), '0.5', 'Manual normalised contrast survives ROI changes.');
    await page.locator('#auto-0').click(); await ready(page, 'ROI_A_2');
    assert.deepEqual(await redPixel(), [128, 0, 0]);
    await page.locator('#section-data > summary').click();
    await page.locator('#section-channels > summary').click();
    await page.locator('#section-normalization > summary').scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${output}/normalization.png`, fullPage: true });
    await page.locator('#clear-normalization').click(); await ready(page, 'ROI_A_2');
    assert.deepEqual(await redPixel(), [255, 0, 0]);
    assert.equal(await page.locator('#high-0').inputValue(), '100');
    assert.deepEqual(errors, []);
    assert.ok(requests.every(request => request.method === 'GET' && new URL(request.url).origin === new URL(base).origin));
    assert.deepEqual(await fingerprints(), before);
  } catch (error) {
    await page.screenshot({ path: `${output}/normalization-failure.png`, fullPage: true });
    throw error;
  } finally { await browser.close(); }
});

test('section eyes independently hide images and masks without losing settings or selection', { timeout: 60000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true });
  const page = tabbedPage(await browser.newPage({ viewport: { width: 1440, height: 900 } }));
  const tissuePixels = () => page.locator('#tissue').evaluate(canvas => {
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let i = 0; i < data.length; i += 4) if (data[i] || data[i + 1] || data[i + 2]) count++;
    return count;
  });
  try {
    await page.goto(base);
    await page.locator('#anndata-file').setInputFiles(input('dense.h5ad')); await ready(page);
    await page.locator('#mask-files').setInputFiles(input('masks')); await ready(page);
    await page.locator('#image-files').setInputFiles(input('images')); await ready(page);
    await page.locator('#channel-enabled-1').uncheck(); await ready(page);
    await page.locator('#overlay-style').selectOption('fill');
    const settings = await page.locator('#channel-0, #low-0, #high-0, #channel-colour-0, #opacity').evaluateAll(nodes => nodes.map(node => node.value));
    const bounds = await page.locator('#viewport').boundingBox();
    const scale = Math.min((bounds.width - 48) / 256, (bounds.height - 48) / 256);
    await page.locator('#tissue').click({ position: { x: bounds.width / 2 - 96 * scale, y: bounds.height / 2 - 96 * scale } });
    await page.waitForFunction(() => document.getElementById('cell-details').textContent.includes('cell_0'));
    await page.locator('#toggle-images').click();
    assert.equal(await page.locator('#section-channels').evaluate(node => node.open), true);
    assert.equal(await page.locator('#toggle-images .eye-closed').isVisible(), true);
    assert.ok(await tissuePixels() > 0, 'The overlay remains visible with images hidden.');
    await page.locator('#toggle-overlay').click();
    assert.equal(await tissuePixels(), 0, 'No image, mask or selected outline remains visible.');
    assert.equal(await page.locator('#cell-badge').textContent(), 'Object 300');
    assert.equal(await page.locator('#section-overlay').evaluate(node => node.open), true);
    for (const section of ['section-data', 'section-roi', 'section-normalization', 'section-channels', 'section-overlay']) await page.locator(`#${section} > summary`).click();
    await page.screenshot({ path: `${output}/eyes-hidden.png`, fullPage: true });
    await page.locator('#toggle-images').press('Enter');
    assert.equal(await page.locator('#section-channels').evaluate(node => node.open), false);
    assert.equal(await page.locator('#toggle-images .eye-open').isVisible(), true);
    assert.ok(await tissuePixels() > 0, 'Images can be restored while the overlay stays hidden.');
    await page.locator('#toggle-overlay').press('Space');
    assert.equal(await page.locator('#section-overlay').evaluate(node => node.open), false);
    assert.equal(await page.locator('#toggle-overlay').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#channel-enabled-1').isChecked(), false);
    assert.equal(await page.locator('#overlay-style').inputValue(), 'fill');
    assert.equal(await page.locator('#cell-badge').textContent(), 'Object 300');
    assert.deepEqual(await page.locator('#channel-0, #low-0, #high-0, #channel-colour-0, #opacity').evaluateAll(nodes => nodes.map(node => node.value)), settings);
    await page.screenshot({ path: `${output}/eyes-visible.png`, fullPage: true });
  } finally { await browser.close(); }
});
