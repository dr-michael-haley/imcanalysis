import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';

test('analysis folder gates navigation and reconnects only to the matching project', { timeout: 60000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome' });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    // Use real, cloneable directory handles in an isolated browser profile.
    // Only the native picker and its permission prompts are simulated.
    await page.addInitScript(() => {
      FileSystemDirectoryHandle.prototype.queryPermission = async () => sessionStorage.getItem('grant') || 'granted';
      FileSystemDirectoryHandle.prototype.requestPermission = async () => {
        sessionStorage.removeItem('grant');
        return 'granted';
      };
      window.showDirectoryPicker = async ({ mode }) => {
        if (mode !== 'readwrite') throw Error('Expected writable analysis folder');
        if (sessionStorage.getItem('cancel')) throw new DOMException('Cancelled', 'AbortError');
        const root = await navigator.storage.getDirectory();
        return root.getDirectoryHandle('project-output', { create: true });
      };
    });
    await page.goto(process.env.SBT_VIEWER_URL || 'http://127.0.0.1:8765/');
    const tab = page.locator('#tab-analyses'), section = page.locator('#section-analysis-folder');
    const sidebarTabs = page.getByRole('tablist', { name: 'Viewer sections' }).getByRole('tab');
    const settings = async () => {
      await page.locator('#tab-setup').click();
      if (!await section.evaluate(node => node.open)) await section.locator('summary').click();
    };
    assert.deepEqual(await sidebarTabs.allTextContents(), ['Setup', 'ROIs', 'Views']);
    await settings();
    assert.equal(await section.locator('.step').innerText(), '04');
    await page.evaluate(() => sessionStorage.setItem('cancel', '1'));
    await page.locator('#choose-analysis-folder').click();
    assert.equal(await tab.isHidden(), true);
    await page.evaluate(() => sessionStorage.removeItem('cancel'));
    await page.locator('#choose-analysis-folder').click();
    await tab.waitFor();
    await page.locator('#tab-views').press('End');
    assert.equal(await tab.getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#analysis-project-name').innerText(), 'project-output');
    await page.locator('#manage-analysis-folder').click();

    // Opening a dataset for this empty workspace keeps its configured folder.
    await page.locator('#anndata-file').setInputFiles(fileURLToPath(new URL('../fixtures/dense.h5ad', import.meta.url)));
    await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('ROI_A ready'));
    await page.locator('#workspace-file-tools summary').click();
    const download = page.waitForEvent('download');
    await page.locator('#save-workspace').click();
    const json = await readFile(await (await download).path(), 'utf8');
    const project = JSON.parse(json).analysisProject;
    assert.equal(project.folderName, 'project-output');

    await page.reload();
    await tab.waitFor();
    await tab.click();
    await page.evaluate(() => { sessionStorage.setItem('grant', 'prompt'); window.dispatchEvent(new Event('focus')); });
    await tab.waitFor({ state: 'hidden' });
    assert.equal(await page.locator('#tab-setup').getAttribute('aria-selected'), 'true');
    await settings();
    await page.locator('#reconnect-analysis-folder').click();
    await tab.waitFor();
    await page.locator('#duplicate-local-workspace').click();
    await tab.waitFor({ state: 'hidden' });
    await page.locator('#choose-analysis-folder').click();
    await page.getByText(/belongs to another project/).waitFor();
    assert.equal(await tab.isHidden(), true);

    // Portable imports remember the project identity but require a new local grant.
    await page.locator('#workspace-file').setInputFiles({ name: 'import.workspace.json', mimeType: 'application/json', buffer: Buffer.from(json) });
    await page.locator('#workspace-root').setInputFiles(fileURLToPath(new URL('../fixtures/', import.meta.url)));
    await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('ROI_A ready'));
    await settings();
    await page.getByRole('button', { name: 'Locate folder', exact: true }).click();
    await tab.waitFor();
    await page.locator('#disconnect-analysis-folder').click();
    await tab.waitFor({ state: 'hidden' });
    const savedId = await page.evaluate(async () => {
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('project-output');
      const file = await (await dir.getFileHandle('imc-explorer.analysis.json')).getFile();
      return JSON.parse(await file.text()).id;
    });
    assert.equal(savedId, project.id);
    assert.deepEqual(errors, []);

    const limited = await browser.newPage();
    await limited.addInitScript(() => { window.showDirectoryPicker = undefined; });
    await limited.goto(page.url());
    await limited.locator('#section-analysis-folder summary').click();
    await limited.getByText('Use desktop Chrome or Edge to connect a writable analysis folder.').waitFor();
    assert.equal(await limited.locator('#tab-analyses').isHidden(), true);
    assert.equal(await limited.locator('#choose-analysis-folder').isHidden(), true);
  } finally { await browser.close(); }
});
