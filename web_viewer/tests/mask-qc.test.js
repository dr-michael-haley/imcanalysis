import test from 'node:test';
import assert from 'node:assert/strict';
import { fromArrayBuffer } from 'geotiff';
import { measureObjects, readMetrics, cleanedTiff, decision, roiScoreSummary } from '../src/mask-qc.js';
import { composeView } from '../src/views.js';
import { trainBinary, predictBinary } from '../src/analysis-model.js';
import { mapROIs } from '../src/analysis-project.js';
import { decisionOverlay, matchInputs, qcPalette, sourceFiles } from '../src/analysis-tissue.js';

test('reopening older QC sources exposes the recorded DNA path without listing the folder', () => {
  const data = {source:{maskFolder:'Masks',imageFolder:'Images',rois:[{name:'A',mask:'A.tif',dna:'A/DNA.tif'}]}, handles:{}};
  assert.deepEqual(sourceFiles(data,'images').map(file => file.webkitRelativePath), ['Images/A/DNA.tif']);
  data.source.rois[0].images = ['A/DNA.tif','A/CD3.tif'];
  assert.deepEqual(sourceFiles(data,'images').map(file => file.name), ['DNA.tif','CD3.tif']);
});

test('QC colours show who made each decision and follow the current threshold', () => {
  const rows = [1, 2, 3, 4, 5].map(id => ({ id }));
  const labels = { 1: 'good', 2: 'artefact' };
  const scores = new Map([[1, 1], [2, 0], [3, 0.2], [4, 0.6]]);
  const overlay = decisionOverlay('Trial', rows, labels, scores, 0.5);
  assert.deepEqual([...overlay.colours], [
    [1, [0, 255, 0]], [2, [255, 0, 0]], [3, [71, 137, 99]], [4, [170, 88, 95]]
  ]);
  assert.deepEqual(overlay.legend, Object.values(qcPalette));
  assert.deepEqual(decisionOverlay('Trial', rows, labels, scores, 0.7).colours.get(4), overlay.colours.get(3));
  assert.equal(decisionOverlay('Trial', rows, labels, null, 0.5).colours.size, 2);
});

test('QC display respects visibility, outline interiors, selection and clipped continuous limits', () => {
  const rows = [{ id: 1 }], scores = new Map([[1, 0.5]]);
  const view = { channels: [], scene: {}, region: { ids: [1], values: ['a'] }, colours: { colour: () => [80, 80, 80] },
    overlayVisible: true, overlay: { maskQC: true, opacity: 1, style: 'fill' } };
  let image;
  view.scene.getContext = () => ({ createImageData: () => ({ data: new Uint8ClampedArray(36) }), putImageData: value => { image = value.data; } });
  const render = (settings, selected) => {
    const overlay = decisionOverlay('QC', rows, { 1: 'good' }, scores, 0.5, settings);
    composeView(view, { width: 3, height: 3, selected, mask: { data: new Uint32Array(9).fill(1) }, analysisOverlay: overlay });
    return { edge: [...image.slice(0, 3)], centre: [...image.slice(16, 19)], overlay };
  };
  const outline = render({ classes: { manualGood: { style: 'outline', colour: '#123456' } } });
  assert.deepEqual(outline.edge, [18, 52, 86]); assert.deepEqual(outline.centre, [0, 0, 0]);
  const hidden = render({ classes: { manualGood: { visible: false } } });
  assert.deepEqual(hidden.edge, [0, 0, 0]); assert.deepEqual(hidden.centre, [0, 0, 0]);
  assert.ok(!hidden.overlay.legend.some(row => row.name === 'Manual good'));
  assert.deepEqual(render({ classes: { manualGood: { visible: false } } }, 1).edge, [255, 255, 0]);
  const continuous = render({ mode: 'uncertainty', cmap: 'grey', min: 0.2, max: 0.8, style: 'outline' });
  assert.deepEqual(continuous.edge, [255, 255, 255]); assert.deepEqual(continuous.centre, [0, 0, 0]);
  assert.deepEqual(continuous.overlay.scale.range, [0.2, 0.8]);
  scores.set(1, 0.1);
  assert.deepEqual(render({ mode: 'score', cmap: 'grey', min: 0.2, max: 0.8 }).centre, [0, 0, 0]);
});

test('ROI summaries count unlabelled, finite scores and distinguish missing uncertainty from zero', () => {
  const scores = new Map([[1, 1], [2, null], [3, 0], [4, 0.5]]);
  assert.deepEqual(roiScoreSummary(scores, { 1: 'good' }, 0.5), { good: 1, artefact: 1, uncertainty: 0.5 });
  assert.deepEqual(roiScoreSummary(scores, { 1: 'good' }, 0.6), { good: 2, artefact: 0, uncertainty: 0.5 });
  assert.deepEqual(roiScoreSummary(scores, { 1: 'good', 3: 'good', 4: 'good' }, 0.5), { good: 0, artefact: 0, uncertainty: null });
});

test('two identically named mask directories do not silently share decisions', async () => {
  const files = [{name:'A.tif',webkitRelativePath:'Masks/A.tif'}];
  files.workspaceHandle = {name:'Masks'};
  const data = {source:{maskFolder:'Masks',rois:[{mask:'A.tif'}]},handles:{masks:{isSameEntry:async () => false}},local:{}};
  await assert.rejects(matchInputs(data,'masks',files),/different masks folder/);
});

test('measures disconnected and sparse labels, excludes background and preserves missing values', () => {
  const mask = { width: 5, height: 3, data: [1, 1, 1, 0, 70000, 1, 1, 1, 0, 0, 1, 1, 1, 0, 70000] };
  const dna = { ...mask, data: [1, 1, 1, 90, NaN, 1, 3, 1, 90, 0, 1, 1, 1, 90, NaN] };
  const table = measureObjects(mask, dna);
  assert.deepEqual(
    table.rows.map(r => r.id),
    [1, 70000]
  );
  assert.deepEqual(table.rows[0].values.slice(0, 6), [9, 1, 1, 1, 12, 1]);
  assert.equal(table.rows[0].values.at(-1), 3);
  assert.equal(table.rows[1].values[6], null);
  assert.throws(() => measureObjects(mask, { ...dna, width: 4 }), /dimensions/);
  assert.throws(() => measureObjects({ ...mask, data: mask.data.map(v => (v === 1 ? -1 : v)) }, dna), /uint32/);
});
test('imports only mask and DNA columns using original object and ROI keys', () => {
  const { columns, groups } = readMetrics(
    'ROI,ObjectNumber,mask_edge_touching,dna_raw_mean,cellpose_flow_error,CellID\n"R,1",70000,True,2.5,999,key'
  );
  assert.deepEqual(columns, ['mask_edge_touching', 'dna_raw_mean']);
  assert.deepEqual(groups.get('R,1')[70000], [1, 2.5]);
  assert.throws(() => readMetrics('ROI,ObjectNumber,mask_area\nA,1,2\nA,1,3'), /Duplicate/);
});
test('boosting separates examples with missing data and survives JSON serialization', () => {
  const model = JSON.parse(
    JSON.stringify(
      trainBinary(
        [
          [1, null],
          [2, 5],
          [9, 5],
          [10, null]
        ],
        [0, 0, 1, 1],
        ['a', 'b']
      )
    )
  );
  assert.ok(predictBinary(model, [2, 5]) < 0.2);
  assert.ok(predictBinary(model, [9, null]) > 0.8);
  assert.equal(predictBinary(model, [null, null]), null);
  assert.throws(() => trainBinary([[1]], [1], ['a']), /good and artefact/i);
  assert.equal(decision('good', 1, 0.5).keep, true);
  assert.equal(decision('artefact', 0, 0.5).keep, false);
  assert.equal(decision(null, 0.5, 0.5).keep, false);
  assert.deepEqual(decision(null, null, 0.5), { keep: true, reason: 'unscored' });
});
test('exported TIFFs round-trip original uint16 and uint32 object IDs', async () => {
  for (const id of [100, 4294967295]) {
    const mask = { width: 3, height: 2, data: [0, 5, id, 5, id, 0] };
    const tiff = await fromArrayBuffer(cleanedTiff(mask, new Set([5])));
    const image = await tiff.getImage();
    assert.deepEqual([...(await image.readRasters())[0]], [0, 0, id, 0, id, 0]);
    assert.deepEqual(mask.data, [0, 5, id, 5, id, 0]);
  }
});
test('ROI concurrency waits for active writes after a failure', async () => {
  const finished = [];
  await assert.rejects(
    mapROIs([0, 1, 2], 2, async x => {
      if (!x) throw Error('failed');
      await new Promise(r => setTimeout(r, 10));
      finished.push(x);
    }),
    /failed/
  );
  assert.deepEqual(finished, [1]);
});
