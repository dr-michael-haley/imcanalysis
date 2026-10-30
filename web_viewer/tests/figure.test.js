import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultFigure, defaultScaleBar, validateFigure, validateScaleBar, figureLayout, scaleBarGeometry, paneFurniture } from '../src/figure.js';
import { validatePopulations, populationFilter } from '../src/populations.js';
import { includedInView, composeView } from '../src/views.js';
import { defaultLayer, labelEdgeDistance, prepareLayer, blendLayer } from '../src/aligned-layers.js';
import { resolvePickedPaths } from '../src/workspace-paths.js';
import { parseWorkspace, PANEL_IDS, WORKSPACE_VERSION } from '../src/workspace.js';
import { defaultFigureFilename, figureFilename, validateFigureFilename } from '../src/figure-filename.js';

test('scale bars use calibration and camera zoom, choose physical nice lengths and report overflow', () => {
  const fixed = { ...defaultScaleBar(), automatic: false, length: 100 };
  assert.deepEqual(scaleBarGeometry(fixed, .5, 2, 800), { length: 100, pixels: 400, fits: true });
  assert.equal(scaleBarGeometry(fixed, .5, 2, 300).fits, false);
  const auto = scaleBarGeometry(defaultScaleBar(), 1, 2, 800);
  assert.equal(auto.length, 100); assert.equal(auto.pixels, 200);
  assert.throws(() => validateFigure({ umPerPixel: 0 }), /umPerPixel/);
  assert.throws(() => validateScaleBar({ length: -5 }), /scale bar/);
});

test('physical lock overrides aspect while all pane dimensions and gaps remain exact', () => {
  const s = { ...defaultFigure(), physicalLock: true, physicalWidth: 500, physicalHeight: 250, aspect: '1:1', paneWidth: 1000, gap: 40 };
  const layout = figureLayout(4, 1200, 700, s);
  assert.equal(layout.paneWidth / layout.paneHeight, 2);
  assert.equal(layout.outputWidth, 2040); assert.equal(layout.outputHeight, 1040);
  const zoom = layout.paneWidth * .5 / 500;
  assert.ok(Math.abs(layout.paneHeight / zoom * .5 - 250) < 1e-8);
  assert.ok(layout.width <= 1200 && layout.height <= 700);
});

test('multiple population identities remain typed, combine filtering and preserve absent-ROI choices', () => {
  const view = { overlay: { population: '' }, region: { values: ['T', 'B', 'S', 1, '1', null] } };
  populationFilter(view.overlay, ['"T"', '"B"', '1', 'null']);
  assert.deepEqual(view.region.values.map((_, i) => includedInView(view, i)), [true, true, false, true, false, true]);
  assert.equal(view.overlay.population, '');
  view.region.values = ['S']; assert.equal(includedInView(view, 0), false);
  assert.equal(validatePopulations(view.overlay).length, 4);
  populationFilter(view.overlay, ['"B"']); assert.equal(view.overlay.population, '"B"');
  assert.throws(() => validatePopulations({ populations: ['{}'] }), /population/);
});

test('annotation thickness stays inside labels and transparent component exports retain alpha', () => {
  const data = new Uint16Array(81).fill(1), raster = { data };
  const distances = labelEdgeDistance(raster, 9, 9);
  assert.equal(distances[40], 5); assert.equal(distances[10], 2);
  const source = { kind: 'labels', mapping: [{ value: 1, color: '#ff0000' }] };
  const layer = prepareLayer({ ...defaultLayer('a', 'labels'), style: 'outline', outlineWidth: 2, raster }, source), pixels = new Uint8ClampedArray(4);
  blendLayer(pixels, 0, layer, 10, 9, 9, true); assert.deepEqual([...pixels], [255, 0, 0, 128]);
  pixels.fill(0); blendLayer(pixels, 0, layer, 40, 9, 9, true); assert.deepEqual([...pixels], [0, 0, 0, 0]);
  data[40] = 2; const adjacent = labelEdgeDistance({ data }, 9, 9); assert.equal(adjacent[39], 1); assert.equal(adjacent[40], 1);
});

test('image and cell SVG components reconstruct the same compositing order without opaque cell backgrounds', () => {
  let output; const scene = { getContext: () => ({ createImageData: () => ({ data: new Uint8ClampedArray(4) }), putImageData: img => { output = [...img.data]; } }) };
  const view = { scene, imagesVisible: true, overlayVisible: true, layers: [], channels: [{ active: true, enabled: true, low: 0, high: 1, colour: '#ff0000', raster: { data: [1] } }], region: { ids: [1], values: ['T'] }, overlay: { style: 'fill', opacity: .5, population: '' }, colours: { colour: () => [0, 255, 0] } };
  const dimensions = { width: 1, height: 1, mask: { data: [1] }, selected: null };
  composeView(view, { ...dimensions, component: 'images' }); assert.deepEqual(output, [255, 0, 0, 255]);
  composeView(view, { ...dimensions, component: 'cells' }); assert.deepEqual(output, [0, 255, 0, 128]);
  composeView(view, dimensions); assert.deepEqual(output, [128, 128, 0, 255]);
});

test('nested folder handles resolve exact relative paths without listing the root or siblings', async () => {
  const handle = { name: 'niches' }, sources = { anndata: { name: 'data.h5ad' }, layers: [{ id: 'n', name: 'Niches', kind: 'labels', folder: { name: 'niches' } }] };
  const root = { async resolve(value) { assert.equal(value, handle); return ['HE_annotations', 'niches']; }, entries() { throw new Error('must not scan'); } };
  await resolvePickedPaths(root, sources, new Map([['layer:n', handle]]));
  assert.equal(sources.layers[0].folder.path, 'HE_annotations/niches');
  await assert.rejects(resolvePickedPaths({ resolve: async () => null }, sources, new Map([['layer:n', handle]])), /common parent/);
});

test('figure furniture escapes names and retains editable scale and legend groups', () => {
  const s = { ...defaultFigure(), mode: 'figure' }, view = { name: '<script>&', legendVisible: true, scaleBar: defaultScaleBar() };
  const svg = paneFurniture(view, [{ title: 'Cells', rows: [{ name: 'T & B', colour: [255, 0, 0] }] }], s, 1024, 1024, 2);
  assert.ok(svg.includes('&lt;script&gt;&amp;')); assert.ok(svg.includes('T &amp; B'));
  assert.ok(svg.includes('pane-scale-bar')); assert.ok(svg.includes('pane-legend')); assert.ok(!svg.includes('<script>'));
});

test('workspace v9 migrates and current schema round-trips multi-populations, scale bars, figure framing and nested locations', () => {
  const channels = Array.from({ length: 6 }, () => ({ active: false, enabled: true, name: '', path: '', plane: 0, low: 0, high: 1, colour: '#ff0000', auto: true }));
  const overlay = { visible: true, colourBy: 'obs:population', source: 'X', population: '"T"', style: 'outline', opacity: .8 };
  const input = { format: 'sbt-tissue-explorer-workspace', version: 9, sources: { anndata: { name: 'data.h5ad', size: 10 }, masks: null, images: null }, view: { mapping: { roi: 'ROI', id: 'ID' }, roi: 'A', channels, overlay, imagesVisible: true, presets: [{ name: 'One', channels, overlay, imagesVisible: true }], selectedView: 0, selected: null, embedding: '', tab: 'views', order: { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 }, visits: [], normalization: { dictionary: null, name: '', quantile: .999, minimum: .1 }, camera: { fitted: false, scale: 2, centreX: 20, centreY: 30 }, panels: Object.fromEntries(PANEL_IDS.map(id => [id, true])) } };
  const migrated = parseWorkspace(JSON.stringify(input)); assert.equal(migrated.version, WORKSPACE_VERSION); assert.equal(migrated.view.figure.umPerPixel, 1); assert.deepEqual(migrated.view.presets[0].overlay.populations, ['"T"']);
  migrated.view.figure = { ...defaultFigure(), mode: 'figure', physicalLock: true, physicalWidth: 500, physicalHeight: 500 };
  migrated.view.figure.filename = { ...defaultFigureFilename(), views: true, metadata: ['donor'] };
  migrated.view.figure.scaleFontSize = 40;
  populationFilter(migrated.view.presets[0].overlay, ['"T"', '"B"']); migrated.view.presets[0].scaleBar.length = 50;
  migrated.sources.images = { name: 'niches', path: 'HE_annotations/niches', count: 2, signature: 'a'.repeat(64) };
  const parsed = parseWorkspace(JSON.stringify(migrated)); assert.deepEqual(parsed, migrated);
});

test('scale text and stroke vary independently of names, lengths and other pane visibility', () => {
  const settings = { ...defaultFigure(), mode: 'figure', fontSize: 18, scaleFontSize: 44 };
  const pane = { name: 'Tissue', legendVisible: false, scaleBar: { ...defaultScaleBar(), automatic: false, length: 50, thickness: 8 } };
  const svg = paneFurniture(pane, [], settings, 1024, 600, 2);
  assert.match(svg, /pane-name[\s\S]*?font-size="18"/);
  assert.match(svg, /pane-scale-bar[\s\S]*?h100" stroke="#[a-f0-9]+" stroke-width="8"/);
  assert.match(svg, /pane-scale-bar[\s\S]*?font-size="44"/);
  assert.doesNotMatch(paneFurniture({ ...pane, scaleBar: { ...pane.scaleBar, visible: false } }, [], settings, 1024, 600, 2), /pane-scale-bar/);
  const legacy = validateFigure({ fontSize: 36 }); assert.equal(legacy.scaleFontSize, 36); assert.deepEqual(legacy.filename, defaultFigureFilename());
  assert.throws(() => validateFigure({ scaleFontSize: 0 }), /scaleFontSize/);
});

test('figure filenames combine selected ROI metadata and view names without reading or mutating source data', () => {
  const settings = { ...defaultFigure(), filename: { label: 'Review', roi: true, views: true, timestamp: false, metadata: ['donor', 'dose', 'present', 'optional'] } };
  const context = { roi: 'ROI/A', viewNames: ['H&E', 'DNA'], metadataLoaded: true, metadata: [{ name: 'donor', values: [['ROI/A', 'Patient 7'], ['Other', 'Patient 9']] }, { name: 'dose', values: [['ROI/A', 0]] }, { name: 'present', values: [['ROI/A', false]] }, { name: 'optional', values: [['ROI/A', null]] }] };
  assert.equal(figureFilename(settings, context), 'Review-ROI_A-donor-Patient_7-dose-0-present-false-optional-Missing-H&E-DNA.png');
  assert.throws(() => figureFilename(settings, { ...context, metadataLoaded: false }), /Read ROI metadata/);
  assert.throws(() => figureFilename(settings, { ...context, roi: 'Absent' }), /unavailable/);
  assert.equal(figureFilename({ ...settings, format: 'jpeg', filename: { ...defaultFigureFilename(), label: '', roi: false, timestamp: false } }, context), 'figure.jpg');
});

test('figure filenames stay safe and bounded while preserving the timestamp and format', () => {
  const date = new Date('2026-10-28T12:34:56.789Z'), context = { roi: '../ROI:one', viewNames: ['A'], metadataLoaded: false, metadata: [] };
  const settings = { ...defaultFigure(), format: 'svg', filename: { ...defaultFigureFilename(), label: 'CON', roi: false } };
  assert.equal(figureFilename(settings, context, date), '_CON-2026-10-28T12-34-56-789Z.svg');
  settings.filename = { ...defaultFigureFilename(), label: '☀'.repeat(100), views: true };
  const filename = figureFilename(settings, { ...context, viewNames: ['🧬'.repeat(200)] }, date);
  assert.ok(filename.length < 200); assert.doesNotMatch(filename, /[<>:"/\\|?*\x00-\x1f]/); assert.ok(filename.endsWith('-2026-10-28T12-34-56-789Z.svg'));
  assert.throws(() => validateFigureFilename({ metadata: ['donor', 'donor'] }), /filename/);
});
