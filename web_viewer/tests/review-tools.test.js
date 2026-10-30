import { defaultScaleBar } from '../src/figure.js';
import test from 'node:test';
import { densityWindow } from '../src/density-curve.js';
import { representativeKey, defaultRepresentative, validateRepresentative } from '../src/representative-panel.js';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import h5wasm from 'h5wasm/node';
import { AnnDataReader } from '../src/anndata.js';
import { representativeMarkers, roiMetadata, filterROIs, validateROIFilters } from '../src/review-data.js';
import { assignFlags, validateFlags, applyFlags, flagsCSV, cellCentres, pointInPolygon } from '../src/flags.js';
import { colourmap, COLOURMAPS } from '../src/colourmaps.js';
import { channelName, channelAliases } from '../src/core.js';
import { parseWorkspace, PANEL_IDS } from '../src/workspace.js';
import { densityCurves, densityPath } from '../src/density-curve.js';

test('density curves share axes and unit area despite unequal population sizes', () => {
  const selected = Array.from({ length: 64 }, (_, i) => i < 10 ? i + 1 : 0);
  const others = selected.map(n => n * 1000);
  const result = densityCurves([selected, others], 2, 18);
  const area = values => values.slice(1).reduce((sum, value, i) => sum + (value + values[i]) / 2 * (result.x[i + 1] - result.x[i]), 0);
  for (const values of result.curves) {
    assert.ok(Math.abs(area(values) - 1) < 1e-12);
    assert.ok(values.every(value => Number.isFinite(value) && value >= 0));
  }
  result.curves[0].forEach((value, i) => assert.ok(Math.abs(value - result.curves[1][i]) < 1e-12));
  assert.equal(result.x[0], 2); assert.equal(result.x.at(-1), 18);
});

test('curves retain separate low and high expression modes, including sparse zeros', () => {
  const low = Array(64).fill(0), high = [...low]; low[0] = 20; high[63] = 2000;
  const result = densityCurves([low, high], 0, 10);
  assert.ok(result.curves[0][0] > result.curves[0][128]);
  assert.ok(result.curves[1].at(-1) > result.curves[1][128]);
  result.curves[0].forEach((value, i) => assert.ok(Math.abs(value - result.curves[1].at(-1 - i)) < 1e-12));
  assert.ok(!/NaN|Infinity/.test(densityPath(result.curves[0], result.maximum)));
  assert.notEqual(densityPath(result.curves[0], result.maximum), densityPath(result.curves[1], result.maximum));
  assert.ok(densityCurves([Array(64).fill(0)], 0, 10).curves[0].every(v => v === 0));
});

await h5wasm.ready;
function withReader(name, run) { const file = new h5wasm.File(fileURLToPath(new URL(`../test-results/review-inputs/review_${name}.h5ad`, import.meta.url)), 'r'); try { return run(new AnnDataReader(file)); } finally { file.close(); } }
for (const encoding of ['dense', 'csr', 'csc']) test(`${encoding}: representative ranking, ROI scope, implicit zeros and finite histograms`, () => withReader(encoding, reader => {
  reader.configure('ROI', 'ObjectNumber'); const region = reader.region('ROI_A', 'obs:population', 'X', '');
  const request = { source: 'X', roiOnly: true, rows: region.rows, values: region.values, population: '"T cell"', colourBy: 'obs:population', top: 6 };
  const result = representativeMarkers(reader, request);
  assert.deepEqual(result.totals, [8, 8]); assert.equal(result.markers.length, 6);
  const cd3 = result.markers.find(item => item.marker === 'CD3'); assert.equal(cd3.score, 2); assert.deepEqual(cd3.means, [20, 2]);
  assert.ok(!result.markers.some(item => ['DNA', 'CD20'].includes(item.marker)));
  for (const item of result.markers) assert.deepEqual(item.histogram.map(bins => bins.reduce((a, b) => a + b, 0)), item.counts);
  assert.deepEqual(result.markers.find(item => item.marker === 'CD4').counts, [7, 8]);
  const ki = result.markers.find(item => item.marker === 'Ki67'); assert.equal(ki.min, 0); assert.equal(ki.histogram[1][0], 8);
  const all = representativeMarkers(reader, { ...request, roiOnly: false, source: 'layers/counts' });
  assert.deepEqual(all.totals, [8, 24]); assert.deepEqual(all.markers.find(item => item.marker === 'CD3').means, [40, 4]);
  assert.throws(() => representativeMarkers(reader, { ...request, population: '"Absent"' }), /both/);
}));

test('ROI metadata is constant within each ROI and filters compose without changing rank', () => withReader('dense', reader => {
  const { fields } = roiMetadata(reader, { roiKey: 'ROI', idKey: 'ObjectNumber' });
  assert.ok(fields.some(f => f.name === 'donor')); assert.ok(fields.some(f => f.name === 'batch')); assert.ok(!fields.some(f => ['population', 'quality', 'ObjectNumber'].includes(f.name)));
  const ranking = [{ name: 'ROI_A_2' }, { name: 'ROI_A' }];
  assert.deepEqual(filterROIs(ranking, fields, []), ranking);
  assert.deepEqual(filterROIs(ranking, fields, [{ field: 'donor', values: ['"Donor A"'] }]), [{ name: 'ROI_A' }]);
  assert.deepEqual(filterROIs(ranking, fields, [{ field: 'nullable', values: ['null'] }]), [{ name: 'ROI_A_2' }]);
  assert.deepEqual(filterROIs(ranking, fields, [{ field: 'batch', values: ['"1"'] }]), []);
  assert.throws(() => validateROIFilters([{ field: 'donor', values: ['[]'] }]));
}));
test('flag edits preserve ROI identity, undo exactly and export typed object IDs', () => {
  const field = { id: 'flags', name: 'QC, review', categories: [{ label: 'Check "cell"', colour: '#ffcc00' }], assignments: [] };
  const undo = assignFlags(field, 'ROI_A', [300, 600], 'Check "cell"'); assignFlags(field, 'ROI_A_2', [300], 'Check "cell"');
  assert.deepEqual(applyFlags({ ids: [300, 900] }, field, 'ROI_A').values, ['Check "cell"', null]);
  undo(); assert.deepEqual(field.assignments, [['ROI_A_2', 300, 'Check "cell"']]);
  assert.ok(flagsCSV(field, { roi: 'ROI', id: 'ObjectNumber' }).includes('"ROI_A_2","ObjectNumber","300","QC, review","Check ""cell"""'));
  assert.deepEqual(validateFlags([field]), [field]); assert.throws(() => validateFlags([{ ...field, assignments: [['ROI_A', 300, 'invalid']] }]));
  withReader('dense', reader => { reader.configure('ROI', 'ObjectNumber'); assert.equal(reader.validateFlagCells([field]), true); assert.throws(() => reader.validateFlagCells([{ ...field, assignments: [['ROI_A', 123, 'Check "cell"']] }]), /absent/); });
});
test('lasso uses mask centres and ignores cells without AnnData identities', () => {
  const centres = cellCentres(new Uint32Array([1, 1, 0, 2, 2, 99]), 3, new Map([[1, 0], [2, 1]]));
  assert.deepEqual(centres.get(1), [1, .5]); assert.equal(centres.has(99), false);
  const polygon = [[0, 0], [3, 0], [3, 1], [0, 1]];
  assert.equal(pointInPolygon(...centres.get(1), polygon), true); assert.equal(pointInPolygon(...centres.get(2), polygon), false);
});
test('unlisted DNA filenames keep a stable channel name and colourmaps reverse exactly', () => {
  const aliases = channelAliases({ markers: [], channelNames: [], channelLabels: [] });
  for (const name of ['44_90_Ir193_DNA3.tiff', '55_20_Ir193_DNA3.ome.tif', 'DNA3.tif']) assert.equal(channelName({ name }, 'ROI_A', aliases), 'DNA3');
  for (const key of Object.keys(COLOURMAPS)) { assert.deepEqual(colourmap(0, key), colourmap(1, key, true)); assert.ok(colourmap(.37, key).every(v => Number.isInteger(v) && v >= 0 && v <= 255)); }
});
test('version 5 workspace retains flags, colourmaps, ROI filters and representative settings; v4 migrates', () => {
  const channels = Array.from({ length: 6 }, () => ({ active: false, enabled: true, name: '', path: '', plane: 0, low: 0, high: 1, colour: '#ff0000', auto: true }));
  const overlay = { visible: true, colourBy: 'flag:qc', source: 'X', population: '', style: 'fill', opacity: .8, colourmap: 'magma', reverse: true };
  const input = { format: 'sbt-tissue-explorer-workspace', version: 5, sources: { anndata: { name: 'test.h5ad', size: 123 }, masks: null, images: null }, view: {
    mapping: { roi: 'ROI', id: 'ObjectNumber' }, roi: 'ROI_A', channels, overlay, imagesVisible: true, presets: [{ name: 'QC', channels, overlay, imagesVisible: true }], selectedView: 0, tab: 'views',
    order: { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 }, visits: [], embedding: '', selected: null,
    normalization: { dictionary: null, name: '', quantile: .999, minimum: .1 }, camera: { fitted: true, scale: 1, centreX: 0, centreY: 0 }, panels: Object.fromEntries(PANEL_IDS.map(id => [id, true])),
    flags: [{ id: 'qc', name: 'QC', categories: [{ label: 'Flagged', colour: '#ffcc00' }], assignments: [['ROI_A', 300, 'Flagged']] }], activeFlag: 'qc', review: { tab: 'flags', height: 350 }, roiFilters: [{ field: 'donor', values: ['"Donor A"'] }], representative: { top: 6, roiOnly: true, source: 'X' },
  } };
  const restored = parseWorkspace(JSON.stringify(input));
  for (const key of ['flags', 'activeFlag', 'roiFilters', 'review']) assert.deepEqual(restored.view[key], input.view[key]);
  assert.deepEqual(restored.view.representative, { ...input.view.representative, lowerQuantile: 0, upperQuantile: 1 });
  assert.deepEqual(restored.view.presets, input.view.presets.map(pane => ({ ...pane, channels: pane.channels.map(c => ({ ...c, showInLegend: true })), overlay: { ...pane.overlay, showInLegend: true, populations: [], vmin: null, vmax: null }, layers: [], legendVisible: true, legendScale: 1, scaleBar: defaultScaleBar() })));
  input.version = 4; input.view.overlay.colourBy = 'obs:population'; delete input.view.overlay.colourmap; delete input.view.overlay.reverse;
  for (const key of ['flags', 'activeFlag', 'roiFilters', 'representative']) delete input.view[key];
  for (const id of ['section-roi-filter', 'section-roi-sort', 'section-representative']) delete input.view.panels[id];
  const older = parseWorkspace(JSON.stringify(input)); assert.equal(older.view.presets[0].overlay.colourmap, 'legacy'); assert.deepEqual(older.view.flags, []); assert.equal(older.view.representative.top, 6);
});

test('quantile windows pool cell counts, retain full-density units and reject invalid bounds', () => {
  const histogram = [[8, 0], [0, 2]], density = densityCurves(histogram, 0, 10);
  const window = densityWindow(density, histogram, .5, .9);
  assert.equal(window.x[0], 3.125); assert.equal(window.x.at(-1), 7.5);
  assert.deepEqual(densityWindow(density, histogram).curves, density.curves);
  assert.equal(window.curves.length, 2);
  assert.throws(() => densityWindow(density, histogram, .9, .1));
  assert.throws(() => validateRepresentative({ ...defaultRepresentative(), lowerQuantile: NaN }));
});

test('marker analysis survives image refreshes, RGB assignments, legends and axis changes but invalidates changed populations', () => {
  const state = { dataset: 1, mapping: { roi: 'ROI', id: 'ObjectNumber' }, selectedView: 0, representative: defaultRepresentative(), remaps: [], views: [{ overlay: { colourBy: 'obs:population', population: '"T"' }, channels: [] }] };
  const key = representativeKey(state, 'A');
  state.ready = false; state.views[0].channels = [{ name: 'CD3', colour: '#ff0000' }]; state.views[0].legendVisible = false; state.representative.upperQuantile = .99;
  assert.equal(representativeKey(state, 'A'), key);
  assert.notEqual(representativeKey(state, 'B'), key);
  state.views[0].overlay.population = '"B"'; assert.notEqual(representativeKey(state, 'A'), key);
  state.representative.roiOnly = false; assert.equal(representativeKey(state, 'A'), representativeKey(state, 'B'));
});
