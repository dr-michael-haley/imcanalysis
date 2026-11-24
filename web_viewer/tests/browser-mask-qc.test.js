import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { fromArrayBuffer } from 'geotiff';
import { cleanedTiff } from '../src/mask-qc.js';

test('ordinary Views and ROI navigation work with mask files alone', { timeout: 30000 }, async () => {
  const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome' });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(process.env.SBT_VIEWER_URL || 'http://127.0.0.1:8765/');
    await page
      .locator('#mask-files')
      .setInputFiles(fileURLToPath(new URL('../fixtures/masks/', import.meta.url)));
    await page.locator('#tab-views').click();
    await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('ROI_A ready'));
    assert.equal(await page.locator('#tab-analyses').isVisible(), false);
    assert.equal(await page.locator('#viewer-toolbar').isVisible(), true);
    await page.locator('#quick-next').click();
    await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('ROI_A_2 ready'));
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test(
  'mask-only project trains, reuses features, exports original object IDs and reopens from disk',
  { timeout: 120000 },
  async () => {
    const browser = await chromium.launch({ channel: process.env.SBT_BROWSER_CHANNEL || 'chrome' });
    try {
      const page = await browser.newPage({ viewport: { width: 1400, height: 980 } }),
        errors = [];
      page.setDefaultTimeout(8000);
      page.on('pageerror', e => errors.push(e.message));
      const ids = [0, 1, 1, 0, 70000, 70000, 0, 3, 3, 0, 1, 1, 0, 70000, 70000, 0, 3, 3];
      const bytes = data => Array.from(new Uint8Array(cleanedTiff({ width: 9, height: 2, data }, new Set())));
      await page.addInitScript(() => {
        const get = FileSystemFileHandle.prototype.getFile;
        window.sourceReads = [];
        FileSystemFileHandle.prototype.getFile = function () {
          if (/\.tiff?$/i.test(this.name)) window.sourceReads.push(this.name);
          return get.call(this);
        };
        FileSystemDirectoryHandle.prototype.queryPermission = async () => 'granted';
        FileSystemDirectoryHandle.prototype.requestPermission = async () => 'granted';
        window.showDirectoryPicker = async ({ id, mode }) => {
          const root = await navigator.storage.getDirectory();
          return root.getDirectoryHandle(
            mode === 'readwrite' ? 'Output' : id.includes('mask') ? 'Masks' : 'Images',
            { create: true }
          );
        };
      });
      await page.goto(process.env.SBT_VIEWER_URL || 'http://127.0.0.1:8765/');
      await page.evaluate(
        async ([mask, dna]) => {
          const root = await navigator.storage.getDirectory();
          const masks = await root.getDirectoryHandle('Masks', { create: true }),
            images = await root.getDirectoryHandle('Images', { create: true });
          async function write(dir, name, bytes) {
            const f = await dir.getFileHandle(name, { create: true }),
              w = await f.createWritable();
            await w.write(new Uint8Array(bytes));
            await w.close();
          }
          for (const roi of ['ROI_A', 'ROI_B', 'ROI_C']) {
            await write(masks, roi + '.tif', mask);
            if (roi !== 'ROI_C') {
              const channels = await images.getDirectoryHandle(roi, { create: true });
              await write(channels, roi + '_DNA3.tif', dna);
              await write(channels, roi + '_Membrane.tif', dna);
            }
          }
        },
        [bytes(ids), bytes(ids.map(id => (id === 70000 ? 100 : id ? 1 : 0)))]
      );
      const click = id => page.locator('#' + id).click();
      async function status(pattern) {
        try {
          await page.locator('#qc-status').filter({ hasText: pattern }).waitFor();
        } catch (error) {
          throw Error(
            'Expected ' + pattern + '. Status was ' + (await page.locator('#qc-status').textContent()),
            { cause: error }
          );
        }
      }
      const tab = name => page.locator('[data-qc-tab="' + name + '"]').click();
      const select = (id, value) => page.locator('#' + id).selectOption(value);
      async function markObject(id, action, expected) {
        await select('qc-target', 'object');
        await page.locator('#qc-object').fill(String(id));
        await click('qc-label-' + action);
        await status(expected);
      }
      const readSavedRun = () => page.evaluate(async () => {
        const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('Output');
        const read = async (dir, name) => JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text());
        const catalog = await read(dir, 'analyses.json');
        return read(await dir.getDirectoryHandle(catalog.analyses[0].directory), 'run.json');
      });
      for (const label of ['Masks', 'Images'])
        await page.getByRole('button', { name: new RegExp('^' + label + ' Choose folder') }).click();
      await page.locator('#section-analysis-folder summary').click();
      await click('choose-analysis-folder');
      await select('mode-analysis', 'mask-qc');
      assert.equal(await page.locator('.review-heading > span').first().innerText(), 'Mask QC');
      assert.deepEqual(await page.locator('[data-qc-tab]').allTextContents(), ['01 Inputs', '02 Label cells', '03 Train and review', '04 Export', 'Display']);
      await click('mode-review');
      assert.equal(await page.locator('#qc-panel').isHidden(), true);
      await select('mode-analysis', 'mask-qc');
      await click('qc-read-setup');
      await status('Matched 3');
      await page.locator('#qc-name').fill('Primary QC');
      await page.locator('#qc-metrics').setInputFiles({
        name: 'CellposeSAM_cell_metrics.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from('ROI,ObjectNumber,mask_area,cellpose_flow_error\nROI_A,70000,4,999')
      });
      await click('qc-create');
      await status('Opened Primary QC');
      assert.deepEqual(await page.evaluate(() => window.sourceReads), []);
      await click('qc-open-roi');
      await status('ROI_A · 3 objects');
      const canvas = page.locator('#tissue'),
        bounds = await canvas.boundingBox();
      const scale = Math.min((bounds.width - 48) / 9, (bounds.height - 48) / 2);
      const firstCell = { x: (bounds.width - 9 * scale) / 2 + 1.5 * scale, y: (bounds.height - 2 * scale) / 2 + 0.5 * scale };
      assert.equal(await page.locator('#channel-0 option:checked').textContent(), 'DNA3');
      assert.equal(await page.locator('#channel-colour-0').inputValue(), '#ffffff');
      assert.equal(await page.locator('#qc-clear-after-mark').isChecked(), true);
      await canvas.click({ position: firstCell });
      const selectedPixel = await canvas.evaluate((el, point) => {
        const bounds = el.getBoundingClientRect();
        const pixel = el.getContext('2d').getImageData(point.x * el.width / bounds.width, point.y * el.height / bounds.height, 1, 1);
        return [...pixel.data];
      }, firstCell);
      assert.deepEqual(selectedPixel, [255, 255, 0, 255]);
      await click('qc-label-good');
      await status('Marked 1 cell as good.');
      let saved = await readSavedRun();
      assert.deepEqual(saved.labels, { 0: { 1: 'good' } });
      assert.equal(Object.hasOwn(saved, 'staged'), false);
      assert.equal(await page.locator('#selection-count').textContent(), '0');
      assert.ok(await page.locator('#qc-label-good').isDisabled());
      await page.locator('#qc-clear-after-mark').uncheck();
      await canvas.click({ position: firstCell });
      await click('qc-label-good');
      await status('No labels changed.');
      assert.equal(await page.locator('#selection-count').textContent(), '1');
      assert.equal((await readSavedRun()).revision, saved.revision);
      await click('qc-label-remove');
      await status('Removed 1 manual label.');
      assert.deepEqual((await readSavedRun()).labels, {});
      await markObject(70000, 'good', 'Marked 1 cell as good.');
      await markObject(70000, 'artefact', 'Marked 1 cell as artefact.');
      await markObject(1, 'good', 'Marked 1 cell as good.');
      saved = await readSavedRun();
      assert.deepEqual(saved.labels, { 0: { 1: 'good', 70000: 'artefact' } });
      await markObject(4, 'artefact', 'Object 4 is absent');
      assert.equal((await readSavedRun()).revision, saved.revision);
      await select('qc-target', 'selection');
      await tab('model');
      await click('qc-train');
      await status('Trained model saved');
      await page.locator('#qc-threshold').fill('0.7');
      await page.locator('#qc-threshold').press('Enter');
      await status('Preview updated');
      await tab('export');
      await select('qc-export-scope', 'all');
      await click('qc-export');
      await status('Saved 3 ROI(s)');
      assert.deepEqual(await page.evaluate(() => window.sourceReads.sort()), [
        'ROI_A.tif',
        'ROI_A_DNA3.tif',
        'ROI_B.tif',
        'ROI_B_DNA3.tif',
        'ROI_C.tif'
      ]);
      const output = await page.evaluate(async () => {
        const f = await (await navigator.storage.getDirectory()).getDirectoryHandle('Output');
        async function json(dir, name) {
          return JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text());
        }
        const cat = await json(f, 'analyses.json'),
          source = cat.sources[0];
        const table = await json(await f.getDirectoryHandle('features-' + source.id), 'roi-0.json');
        const run = await f.getDirectoryHandle(cat.analyses[0].directory);
        for await (const [name, dir] of run.entries())
          if (name.startsWith('export-')) {
            const mask = await (await dir.getFileHandle('ROI_A.tif')).getFile();
            const csv = async roi => (await (await dir.getFileHandle(roi + '.qc.csv')).getFile()).text();
            return {
              manifest: await json(dir, 'export.json'),
              csv: await csv('ROI_A'),
              unscored: await csv('ROI_C'),
              file: Array.from(new Uint8Array(await mask.arrayBuffer())),
              columns: table.columns
            };
          }
      });
      assert.ok(output.columns.includes('mask_area'));
      assert.ok(!output.columns.includes('cellpose_flow_error'));
      assert.equal(output.manifest.status, 'complete');
      assert.equal(output.manifest.threshold, 0.7);
      assert.match(output.csv, /,70000,artefact,[\d.]+,0.7,0,confirmed/);
      assert.match(output.unscored, /,70000,,,0.7,1,unscored/);
      const tiff = await fromArrayBuffer(Uint8Array.from(output.file).buffer);
      assert.deepEqual(
        [...(await (await tiff.getImage()).readRasters())[0]],
        ids.map(id => (id === 70000 ? 0 : id))
      );

      await tab('display');
      await page.getByLabel('Manual good style', { exact: true }).selectOption('outline');
      await status('Display settings saved');
      await page.getByLabel('Predicted good', { exact: true }).uncheck();
      await select('qc-display-mode', 'uncertainty');
      await select('qc-metric-cmap', 'magma');
      await select('qc-metric-style', 'outline');
      await page.locator('#qc-metric-max').fill('0.8');
      await page.locator('#qc-metric-max').press('Enter');
      await page.waitForFunction(() => document.querySelector('.floating-legend-range')?.textContent.includes('0.800'));
      const storedDisplay = (await readSavedRun()).display;
      assert.equal(storedDisplay.cmap, 'magma');
      assert.equal(storedDisplay.classes.predictedGood.visible, false);
      await tab('label');
      const roiRow = name => page.locator('#qc-roi-rows tr').filter({ has: page.getByRole('button', { name, exact: true }) });
      assert.match(await roiRow('ROI_B').innerText(), /Not scored/);
      await page.evaluate(() => (window.sourceReads.length = 0));
      await click('qc-score-rois'); await status('ROI scores saved');
      assert.deepEqual(await page.evaluate(() => window.sourceReads), []);
      const review = (await readSavedRun()).roiReview;
      assert.equal(review[0].artefact + review[0].good, 1);
      assert.equal(review[1].artefact + review[1].good, 3);
      assert.equal(review[2], undefined);
      await select('qc-roi-filter', 'labelled');
      assert.equal(await page.locator('#qc-roi-rows button').count(), 1);
      await select('qc-roi-filter', 'all');
      await select('qc-roi-sort', 'uncertainty');
      await click('qc-next-review-roi'); await status('ROI_B · 3 objects');
      await roiRow('ROI_A').getByRole('button').click(); await status('ROI_A · 3 objects');
      await tab('model');
      assert.equal(await page.locator('#qc-model-state').innerText(), 'Classifier up to date');
      const trained = (await readSavedRun()).model.id;
      await page.locator('#qc-threshold').fill('0.6'); await page.locator('#qc-threshold').press('Enter');
      await status('Preview updated');
      assert.equal(await page.locator('#qc-model-state').innerText(), 'Classifier up to date');
      await tab('label');
      assert.match(await roiRow('ROI_B').innerText(), /Not scored/);
      await tab('model'); await page.locator('#qc-threshold').fill('0.7'); await page.locator('#qc-threshold').press('Enter');
      await status('Preview updated');
      await tab('label');
      await markObject(1, 'remove', 'Removed 1 manual label');
      await tab('model');
      assert.equal(await page.locator('#qc-model-state').innerText(), 'Retraining needed');
      assert.equal(await page.locator('#qc-train').isDisabled(), true);
      assert.equal((await readSavedRun()).model.id, trained);
      await tab('label'); await markObject(1, 'good', 'Marked 1 cell as good.');
      await select('qc-target', 'selection');
      await tab('model'); await click('qc-train'); await status('Trained model saved');
      assert.equal(await page.locator('#qc-model-state').innerText(), 'Classifier up to date');
      await tab('label');
      assert.match(await roiRow('ROI_B').innerText(), /Not scored/);
      await click('qc-score-rois'); await status('ROI scores saved');

      await click('tab-analyses'); await click('qc-new');
      await page.locator('#qc-name').fill('Independent trial');
      await click('qc-create');
      await status('Opened Independent trial');
      assert.match(await page.locator('#qc-counts').textContent(), /^0 good · 0 artefact/);
      await page.evaluate(() => (window.sourceReads.length = 0));
      await tab('model');
      await click('qc-extract');
      await status('3 / 3');
      assert.deepEqual(await page.evaluate(() => window.sourceReads), []);
      await page.reload();
      await click('tab-analyses');
      await status('Opened Independent trial');
      assert.deepEqual(await page.evaluate(() => window.sourceReads), []);
      await page.locator('#qc-run').selectOption({ label: 'Primary QC' });
      await status('Opened Primary QC');
      assert.match(await page.locator('#qc-counts').textContent(), /^1 good · 1 artefact/);
      assert.doesNotMatch(await roiRow('ROI_B').innerText(), /Not scored/);
      await tab('display');
      assert.equal(await page.locator('#qc-display-mode').inputValue(), 'uncertainty');
      assert.equal(await page.locator('#qc-metric-max').inputValue(), '0.8');
      await select('qc-display-mode', 'labels'); await tab('label');
      await click('qc-open-roi');
      await status('ROI_A · 3 objects');
      assert.deepEqual(await page.evaluate(() => window.sourceReads.sort()), ['ROI_A.tif', 'ROI_A_DNA3.tif']);
      await tab('model');
      assert.equal(await page.locator('#qc-threshold').inputValue(), '0.7');
      assert.equal(await page.locator('#qc-clear-after-mark').isChecked(), false);
      assert.match(await page.locator('#qc-model-info').innerText(), /1 of 3 objects/);
      await click('tab-setup');
      await page.locator('#section-analysis-folder').evaluate(el => {
        el.open = true;
      });
      await click('disconnect-analysis-folder');
      await page.locator('#tab-analyses').waitFor({ state: 'hidden' });
      await click('open-analysis-project');
      await click('tab-analyses');
      await status('Opened Independent trial');
      assert.equal(await page.locator('#qc-run option').count(), 3);
      // Reopen and compare the same tissue using two ordinary Views panes.
      await page.locator('#qc-run').selectOption({ label: 'Primary QC' });
      await status('Opened Primary QC');
      await click('qc-open-roi');
      await status('ROI_A · 3 objects');
      await click('tab-views');
      await page.locator('#view-name').fill('Decisions');
      await click('add-view');
      await page.waitForFunction(() =>
        document.querySelector('#status').textContent.includes('2 linked views')
      );
      await page.locator('#view-name').fill('DNA reference');
      assert.ok(
        (await page.locator('#channel-1 option').allTextContents()).some(name => name.includes('Membrane'))
      );
      await page.locator('#qc-overlay').uncheck();
      await page.waitForFunction(
        () =>
          !document
            .querySelector('#tissue-1')
            .closest('.view-pane')
            .querySelector('.view-legend')
            .textContent.includes('Primary QC')
      );
      assert.match(await page.locator('.view-pane').first().locator('.view-legend').innerText(), /Primary QC/);
      assert.ok(await page.locator('#qc-panel').isVisible());
      const preset = page.locator('#saved-views-root details');
      await preset.locator('summary').click();
      await page.locator('#preset-name').fill('DNA only');
      await click('preset-save');
      await page.locator('#preset-rows select').selectOption('F2');
      await page.locator('#qc-overlay').check();
      await page.locator('#viewport').focus();
      await page.keyboard.press('F2');
      await page.waitForFunction(() => !document.querySelector('#qc-overlay').checked);

      await click('tab-rois');
      await click('next-roi');
      await status('ROI_B · 3 objects');
      await tab('label');
      if (await page.locator('#selection-lasso').isHidden()) await click('selection-mode-trigger');
      await click('selection-lasso');
      const b = await page.locator('#tissue-1').boundingBox();
      const z = Math.min((b.width - 48) / 9, (b.height - 48) / 2);
      const left = b.x + (b.width - 9 * z) / 2,
        top = b.y + (b.height - 2 * z) / 2;
      // Include centres at x=2 and x=5, excluding the third object at x=8.
      await page.mouse.move(left + z, top + 0.3 * z);
      await page.mouse.down();
      for (const [x, y] of [
        [6, 0.3],
        [6, 1.8],
        [1, 1.8],
        [1, 0.3]
      ])
        await page.mouse.move(left + x * z, top + y * z);
      await page.mouse.up();
      await click('qc-label-good');
      await status('Marked 2 cells as good.');
      assert.match(await page.locator('#qc-counts').innerText(), /^3 good · 1 artefact/);
      await click('quick-previous');
      await status('ROI_A · 3 objects');

      // Add a cell in ROI A to the existing two-cell lasso in ROI B.
      await click('selection-mode-trigger');
      await click('selection-click');
      const activeCanvas = page.locator('#tissue-1');
      const area = await activeCanvas.boundingBox();
      const zoom = Math.min((area.width - 48) / 9, (area.height - 48) / 2);
      await activeCanvas.click({ position: {
        x: (area.width - 9 * zoom) / 2 + 1.5 * zoom,
        y: (area.height - 2 * zoom) / 2 + 0.5 * zoom
      } });
      const beforeBatch = await readSavedRun();
      await click('qc-label-remove');
      await status('Removed 3 manual labels.');
      const afterBatch = await readSavedRun();
      assert.deepEqual(afterBatch.labels, { 0: { 70000: 'artefact' } });
      assert.equal(afterBatch.generation, beforeBatch.generation + 1);
      await page.locator('#qc-clear-after-mark').check();
      await markObject(999, 'artefact', 'Object 999 is absent');
      assert.equal(await page.locator('#selection-count').textContent(), '3');
      await select('qc-target', 'selection');
      await click('qc-label-good');
      await status('Marked 3 cells as good.');
      assert.equal(await page.locator('#selection-count').textContent(), '0');
      assert.deepEqual((await readSavedRun()).labels, beforeBatch.labels);

      await click('quick-next'); await status('ROI_B · 3 objects');
      await click('quick-next'); await status('ROI_C · 3 objects');
      await click('qc-open-roi');
      assert.match(await page.locator('#channel-0 option:checked').textContent(), /DNA3.*unavailable/);
      await click('quick-previous'); await status('ROI_B · 3 objects');
      await click('quick-previous'); await status('ROI_A · 3 objects');
      assert.equal(await page.locator('#channel-0 option:checked').textContent(), 'DNA3');

      await click('quick-bookmark');
      await page.locator('#bookmark-rows tr').first().waitFor();
      await page.reload();
      await click('tab-analyses');
      await status('Opened Independent trial');
      await page.locator('#qc-run').selectOption({ label: 'Primary QC' });
      await status('Opened Primary QC');
      await click('qc-open-roi');
      await status('ROI_A · 3 objects');
      assert.equal(await page.locator('#preset-rows select').inputValue(), 'F2');
      await click('mode-review');
      await click('review-tab-bookmarks');
      await page.locator('#bookmark-rows button').first().click();
      await page.waitForFunction(() =>
        document.querySelector('#bookmark-status').textContent.startsWith('Returned to ')
      );
      assert.deepEqual(await page.locator('.pane-label').allTextContents(), ['Decisions', 'DNA reference']);

      // Attach the same saved project to an AnnData view. The observation
      // overlay remains usable and an object missing from AnnData can be labelled.
      await click('tab-setup');
      await page
        .locator('#anndata-file')
        .setInputFiles(fileURLToPath(new URL('../fixtures/dense.h5ad', import.meta.url)));
      await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('ROI_A ready'));
      await click('tab-analyses');
      await click('qc-open-roi');
      await status('ROI_A · 3 objects');
      await click('tab-views');
      await select('view-select', '0');
      await select('colour-by', 'obs:population');
      await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('ROI_A ready'));
      assert.equal(await page.locator('#colour-by').inputValue(), 'obs:population');
      await tab('label');
      await click('selection-mode-trigger');
      await click('selection-click');
      const a = await page.locator('#tissue').boundingBox();
      const az = Math.min((a.width - 48) / 9, (a.height - 48) / 2);
      await page
        .locator('#tissue')
        .click({ position: { x: (a.width - 9 * az) / 2 + 4.5 * az, y: (a.height - 2 * az) / 2 + 0.5 * az } });
      assert.ok(
        await page.locator('#qc-label-good').isEnabled(),
        (await page.evaluate(() =>
          [
            'status',
            'warning',
            'cell-badge',
            'selection-count',
            'selection-mode-name',
            'qc-status',
            'qc-view-roi',
            'view-scale',
            'population-summary'
          ]
            .map(id => id + '=' + document.getElementById(id).textContent)
            .join('\n')
        )) +
          ' Errors: ' +
          errors.join(',')
      );
      await click('qc-label-good');
      await status('Marked 1 cell as good.');
      assert.match(await page.locator('#qc-counts').innerText(), /^4 good · 0 artefact/);

      // Opening an already attached AnnData view moves its DNA slot back to first.
      const waitReady = () => page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('ROI_A ready'));
      await select('channel-0', { label: 'Membrane' });
      await waitReady();
      await select('channel-1', { label: 'DNA3' });
      await waitReady();
      await click('qc-open-roi');
      await waitReady();
      assert.equal(await page.locator('#channel-0 option:checked').textContent(), 'DNA3');
      assert.equal(await page.locator('#channel-1 option:checked').textContent(), 'Membrane');
      assert.equal(await page.locator('#channel-colour-0').inputValue(), '#ffffff');

      // Reopen a run saved by the older workbench with a pending correction.
      await page.evaluate(async () => {
        const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('Output');
        const catalog = JSON.parse(await (await (await dir.getFileHandle('analyses.json')).getFile()).text());
        const runDir = await dir.getDirectoryHandle(catalog.analyses[0].directory);
        const file = await runDir.getFileHandle('run.json');
        const run = JSON.parse(await (await file.getFile()).text());
        const writer = await file.createWritable();
        await writer.write(JSON.stringify({ ...run, staged: {} }));
        await writer.close();
        const db = await new Promise((resolve, reject) => {
          const req = indexedDB.open('sbt-tissue-explorer', 1);
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => reject(req.error);
        });
        await new Promise((resolve, reject) => {
          const tx = db.transaction('settings', 'readwrite');
          tx.objectStore('settings').put(
            { generation: run.generation, staged: { 0: { 70000: 'clear' } } },
            'analysis-draft:' + catalog.analyses[0].id
          );
          tx.oncomplete = resolve;
          tx.onerror = () => reject(tx.error);
        });
        db.close();
      });
      await click('tab-analyses');
      await click('qc-reload');
      await status('Opened Primary QC');
      assert.equal((await readSavedRun()).labels[0][70000], 'good');
      await click('qc-import-edits');
      await status('Earlier label edits applied.');
      assert.equal((await readSavedRun()).labels[0][70000], undefined);
      await tab('model');
      assert.match(await page.locator('#qc-model-info').innerText(), /1 of 3 objects marked for removal/);
      await click('qc-reload');
      await status('Opened Primary QC');
      assert.equal(await page.locator('#qc-import-edits').isVisible(), false);
      assert.deepEqual(errors, []);
    } finally {
      await browser.close();
    }
  }
);
