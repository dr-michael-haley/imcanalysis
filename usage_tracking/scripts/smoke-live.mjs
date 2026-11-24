// Explicit production smoke test. Records three anonymous events; generates exact cleanup SQL.
import { chromium } from '../../web_viewer/node_modules/playwright/index.mjs';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const output = new URL('../deployment/', import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-swiftshader'] });
const ids = [];
try {
  for (const tool of ['imc', 'mcd', 'merscope']) {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    try {
      const page = await context.newPage();
      await page.goto(`https://${tool}.spatialbiology.app/`);
      await page.getByRole('heading', { name: 'Help us keep these tools free' }).waitFor();
      await page.getByLabel('Remain anonymous').check();
      await page.getByLabel('External', { exact: true }).check();
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.screenshot({ path: fileURLToPath(new URL(`${tool}-live.png`, output)) });
      const responsePromise = page.waitForResponse('https://usage.spatialbiology.app/record');
      await page.getByRole('button', { name: 'Continue to tool', exact: true }).click();
      const response = await responsePromise;
      const event = response.request().postDataJSON();
      assert.match(event.eventId, /^[0-9a-f-]{36}$/);
      ids.push(event.eventId);
      await writeFile(new URL('smoke-records.json', output), JSON.stringify(ids));
      await writeFile(new URL('smoke-cleanup.sql', output), `DELETE FROM usage_events WHERE name IS NULL AND event_id IN (${ids.map(id => `'${id}'`).join(',')});`);
      assert.equal(response.status(), 204);
      assert.equal(event.name, null);
      await page.waitForFunction(tool => JSON.parse(localStorage.getItem(`sbt-usage-receipt-v1:${tool}`))?.sent, tool);
      await page.reload();
      await page.getByRole('button', { name: 'Usage preferences', exact: true }).waitFor();
      assert.equal(await page.getByRole('dialog').count(), 0);
      console.log(`${tool}: live prompt, anonymous POST 204, and remembered choice verified.`);
    } finally { await context.close(); }
  }
} finally { await browser.close(); }
