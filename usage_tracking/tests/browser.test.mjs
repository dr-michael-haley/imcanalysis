import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from '../../web_viewer/node_modules/playwright/index.mjs';
import { readFile, mkdir } from 'node:fs/promises';
import { extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDemoFile } from '../../mcd_viewer/src/demo.js';

const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome', headless: true, args: ['--enable-unsafe-swiftshader'] });
test.after(() => browser.close());
const endpoint = 'https://usage.spatialbiology.app/record';
const builds = { imc: '../../SpatialBiologyToolkit/browser_viewer/static/', mcd: '../../mcd_viewer/dist/', merscope: '../../spatialdata_viewer/dist/' };
async function fixture(tool, options = {}) {
  const context = await browser.newContext({ viewport: options.mobile ? { width: 375, height: 812 } : { width: 1280, height: 900 } });
  if (options.noStorage) await context.addInitScript(() => {
    Storage.prototype.getItem = Storage.prototype.setItem = () => { throw new Error('Storage disabled'); };
  });
  const page = await context.newPage();
  const events = [];
  const actions = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await context.route(`https://${tool}.spatialbiology.app/**`, async route => {
    if (process.env.SBT_USAGE_LIVE_UI === '1') return route.continue();
    const path = new URL(route.request().url()).pathname.slice(1) || 'index.html';
    const build = process.env.SBT_MERSCOPE_BUILD && tool === 'merscope' ? new URL(process.env.SBT_MERSCOPE_BUILD)
      : new URL(process.env.SBT_USAGE_STAGED ? `../deployment/${tool}/updated/` : builds[tool], import.meta.url);
    const file = path === 'usage.js' ? new URL('../client/usage.js', import.meta.url) : new URL(path, build);
    const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.json': 'application/json', '.png': 'image/png' };
    try {
      const headersFile = await readFile(new URL('_headers', build), 'utf8');
      const csp = headersFile.match(/Content-Security-Policy: (.+)/)?.[1]?.trim();
      await route.fulfill({ body: await readFile(file), contentType: types[extname(path)] || 'application/octet-stream', headers: { 'Content-Security-Policy': csp } });
    } catch { await route.fulfill({ status: 404, body: '' }); }
  });
  await context.route(endpoint, async route => {
    events.push(route.request().postDataJSON());
    if (options.offline) return route.abort();
    await route.fulfill({ status: 204, headers: { 'Access-Control-Allow-Origin': `https://${tool}.spatialbiology.app` } });
  });
  await context.route('https://usage.spatialbiology.app/actions', async route => {
    actions.push(route.request().postDataJSON());
    await route.fulfill({ status: 204, headers: { 'Access-Control-Allow-Origin': `https://${tool}.spatialbiology.app` } });
  });
  await page.goto(`https://${tool}.spatialbiology.app/`);
  await page.getByRole('heading', { name: 'Help us keep these tools free' }).waitFor();
  return { page, context, events, actions, errors };
}
for (const tool of ['imc', 'mcd', 'merscope']) {
  test(`${tool}: first visit, validation, remembered name, editing and anonymous choice`, async () => {
    const { page, context, events, errors } = await fixture(tool);
    try {
      assert.equal(events.length, 0);
      await page.getByRole('button', { name: 'Continue to tool', exact: true }).click();
      await page.getByRole('alert').filter({ hasText: 'Enter your name' }).waitFor();
      await page.getByLabel('Your name', { exact: true }).fill('Test Researcher');
      await page.getByLabel('University of Manchester', { exact: true }).check();
      await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
      await page.screenshot({ path: new URL(`../test-results/${tool}-desktop.png`, import.meta.url).pathname.replace(/^\/(\w:)/, '$1') });
      await page.getByRole('button', { name: 'Continue to tool', exact: true }).click();
      await page.waitForFunction(() => JSON.parse(localStorage.getItem(`sbt-usage-receipt-v1:${document.querySelector('script[data-usage-tool]').dataset.usageTool}`))?.sent);
      assert.equal(events.length, 1);
      assert.equal(events[0].name, 'Test Researcher');
      assert.deepEqual(Object.keys(events[0]).sort(), ['affiliation', 'eventId', 'name', 'tool']);
      await page.reload();
      await page.getByRole('button', { name: 'Usage preferences', exact: true }).click();
      assert.equal(await page.getByLabel('Your name', { exact: true }).inputValue(), 'Test Researcher');
      assert.equal(events.length, 1);
      await page.getByLabel('Remain anonymous').check();
      assert(await page.getByLabel('Your name', { exact: true }).isDisabled());
      await page.getByLabel('External', { exact: true }).check();
      await page.getByRole('button', { name: 'Save preferences' }).click();
      await page.waitForFunction(() => JSON.parse(localStorage.getItem(`sbt-usage-receipt-v1:${document.querySelector('script[data-usage-tool]').dataset.usageTool}`))?.sent);
      assert.equal(events.length, 2);
      assert.equal(events[1].name, null);
      assert.equal(events[1].affiliation, 'external');
      assert(!(await page.evaluate(() => JSON.stringify(localStorage))).includes('Test Researcher'));
      await page.getByRole('button', { name: 'Usage preferences', exact: true }).click();
      await page.getByRole('button', { name: 'Save preferences' }).click();
      await page.reload();
      assert.equal(events.length, 2, 'unchanged preferences/reload must not create another anonymous event');
      assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}
test('mobile, keyboard, anonymous with unavailable storage and endpoint', async () => {
  const { page, context, events } = await fixture('mcd', { mobile: true, noStorage: true, offline: true });
  try {
    await page.screenshot({ path: new URL('../test-results/mobile.png', import.meta.url).pathname.replace(/^\/(\w:)/, '$1') });
    const bounds = await page.getByRole('dialog').boundingBox();
    assert(bounds.x >= 0 && bounds.x + bounds.width <= 375);
    await page.getByLabel('Remain anonymous').check();
    await page.getByLabel('External', { exact: true }).check();
    await page.getByRole('button', { name: 'Continue to tool', exact: true }).focus();
    await page.keyboard.press('Enter');
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(events[0]?.name, null);
    await page.getByRole('button', { name: 'Usage preferences', exact: true }).click();
    await page.keyboard.press('Escape');
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
  } finally { await context.close(); }
});
test('Not now and Escape send nothing', async () => {
  const { page, context, events } = await fixture('imc');
  try {
    await page.getByRole('button', { name: 'Not now' }).click();
    await page.getByRole('button', { name: 'Usage preferences', exact: true }).click();
    await page.keyboard.press('Escape');
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(events.length, 0);
  } finally { await context.close(); }
});

for (const tool of Object.keys(builds)) {
  test(`${tool}: a successful local open sends one anonymous action after opt-in`, async () => {
    const { page, context, actions } = await fixture(tool);
    try {
      await page.evaluate(() => window.sbtRecordAction('dataset_opened'));
      assert.equal(actions.length, 0);
      await page.getByLabel('Your name', { exact: true }).fill('Private Test Name');
      await page.getByLabel('External', { exact: true }).check();
      await page.getByLabel('Share anonymous counts', { exact: false }).check();
      await page.getByRole('button', { name: 'Continue to tool', exact: true }).click();
      const actionReceived = kind => page.waitForRequest(req => req.url().endsWith('/actions') && req.postDataJSON().action === kind);
      const expected = actionReceived('dataset_opened');
      if (tool === 'imc') {
        await page.locator('#anndata-file').setInputFiles(fileURLToPath(new URL('../../web_viewer/fixtures/dense.h5ad', import.meta.url)));
      } else if (tool === 'mcd') {
        const buffer = Buffer.from(await createDemoFile().arrayBuffer());
        await page.locator('#file').setInputFiles({ name: 'private-slide.mcd', mimeType: 'application/octet-stream', buffer });
      } else {
        const manifest = JSON.parse(await readFile(new URL('../../spatialdata_viewer/fixtures/fixture.json', import.meta.url)));
        await page.locator('input[type=file][accept*="zip"]').first().setInputFiles(fileURLToPath(new URL('../../spatialdata_viewer/fixtures/' + manifest.zip, import.meta.url)));
      }
      const req = await expected;
      assert.deepEqual(Object.keys(req.postDataJSON()).sort(), ['action','affiliation','eventId','tool']);
      assert.equal(req.postDataJSON().tool, tool);
      assert.equal(req.postDataJSON().affiliation, 'external');
      assert.doesNotMatch(req.postData(), /private|\.mcd|\.zip|h5ad/i);
      if (tool === 'mcd') {
        const demo = actionReceived('demo_opened');
        await page.locator('#demo').evaluate(button => button.click());
        await demo;
      }
      if (tool === 'merscope') {
        // Opening metadata precedes mounting the render canvas used for export.
        await page.locator('.viewer-pane canvas').first().waitFor();
        await page.getByRole('button', { name: 'Figure', exact: true }).click();
        await page.getByRole('button', { name: 'Figure mode', exact: true }).click();
      } else {
        if (tool === 'imc') {
          await page.locator('#mask-files').setInputFiles(fileURLToPath(new URL('../../web_viewer/fixtures/masks', import.meta.url)));
          await page.locator('#quick-export').click();
        } else {
          await page.locator('#export-section > summary').click();
        }
        await page.locator('#figure-mode').click();
      }
      const [exported] = await Promise.all([
        actionReceived('figure_exported'),
        tool === 'merscope' ? page.getByRole('button', { name: 'Export figure', exact: true }).click() : page.locator('#figure-export').click(),
      ]);
      assert.deepEqual(Object.keys(exported.postDataJSON()).sort(), ['action','affiliation','eventId','tool']);
      await page.getByRole('button', { name: 'Usage preferences', exact: true }).click();
      await page.getByLabel('Share anonymous counts', { exact: false }).uncheck();
      await page.getByRole('button', { name: 'Save preferences' }).click();
      await page.evaluate(() => window.sbtRecordAction('figure_exported'));
      assert.deepEqual(actions.map(x => x.action), tool === 'mcd' ? ['dataset_opened', 'demo_opened', 'figure_exported'] : ['dataset_opened','figure_exported']);
    } finally { await context.close(); }
  });
}
