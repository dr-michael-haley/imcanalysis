import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import h5wasm from 'h5wasm/node';
import { AnnDataReader } from '../src/anndata.js';
import { readPlot } from '../src/plot-reader.js';
import { defaultPlots, validatePlots, plottedDataCSV, scaleMeans, normaliseCounts, labelCategories } from '../src/plots.js';
import { createRemap, editRemap } from '../src/remaps.js';
import { plotSVG } from '../src/plot-svg.js';
import { parseWorkspace, PANEL_IDS } from '../src/workspace.js';

await h5wasm.ready;
function withReader(name, run) { const file = new h5wasm.File(fileURLToPath(new URL(`../fixtures/${name}.h5ad`, import.meta.url)), 'r'); try { return run(new AnnDataReader(file)); } finally { file.close(); } }
const spec = patch => ({ settings: { ...defaultPlots(), groupBy: 'obs:population', scope: 'all', markers: ['CD3', 'CD20'], sampleBy: 'ROI', ...patch }, roiKey: 'ROI', currentROI: 'ROI_A' });

test('plot colour limits reach heatmap exports without changing aggregates or CSV values', () => withReader('dense', reader => {
  for (const type of ['matrix', 'composition', 'comparison']) {
    const base = { type, compareBy: 'obs:nullable' };
    const automatic = readPlot(reader, spec(base)), manual = readPlot(reader, spec({ ...base, vmin: 10, vmax: 20 }));
    assert.deepEqual(manual.values, automatic.values); assert.deepEqual(manual.counts, automatic.counts);
    const svg = plotSVG(manual);
    assert.ok(svg.includes('<title>20</title>20</text>')); assert.ok(svg.includes('<title>10</title>10</text>'));
    assert.ok(plottedDataCSV(manual).includes('""vmin"":10,""vmax"":20'));
    if (type === 'matrix') assert.match(svg, /fill="#185788"[^>]*><title>T cell · CD20:/);
  }
  const zscore = readPlot(reader, spec({ scale: 'zscore', vmin: -0.5, vmax: 2 }));
  const svg = plotSVG(zscore); assert.ok(svg.includes('<title>-0.5</title>-0.5</text>')); assert.ok(svg.includes('<title>2</title>2</text>'));
  assert.throws(() => validatePlots({ vmin: 4, vmax: 2 }), /Lower colour/);
}));

for (const encoding of ['dense', 'csr', 'csc']) test(`${encoding}: exact means, zero handling, remaps, scope and expression source`, () => withReader(encoding, reader => {
  const progress = [], result = readPlot(reader, spec({}), p => progress.push(p.fraction));
  assert.deepEqual(result.rows.map(row => row.value), ['T cell', 'B cell', 'Stroma', null]);
  assert.deepEqual(result.means, [[0, 43], [0, 46], [0, 49], [0, 52]]);
  assert.deepEqual(result.finiteCounts, Array.from({ length: 4 }, () => [8, 8]));
  assert.equal(progress.at(-1), 1);
  assert.ok(progress.every((v, i) => !i || v >= progress[i - 1]));
  assert.deepEqual(readPlot(reader, spec({ source: 'layers/counts' })).means, [[0, 86], [0, 92], [0, 98], [0, 104]]);
  const current = readPlot(reader, spec({ scope: 'current' }));
  assert.deepEqual(current.rows.map(row => row.value), ['T cell', 'Stroma']); assert.equal(current.selectedCells, 16);
  const selected = readPlot(reader, spec({ scope: 'selected', rois: ['ROI_A_2'], populations: ['null'] }));
  assert.equal(selected.selectedCells, 8); assert.deepEqual(selected.means, [[0, 52]]);
  const plan = editRemap(createRemap('merged', 'Reviewed', 'population', reader.populationSummary('population')), ['"T cell"', '"B cell"'], { label: 'Lymphocytes', colour: '#00ffff' });
  const merged = readPlot(reader, { ...spec({ groupBy: 'remap:merged' }), remaps: [plan] });
  assert.deepEqual(merged.means[0], [0, 44.5]); assert.equal(merged.rows[0].count, 16); assert.equal(merged.rows[0].colour, '#00ffff');
}));

for (const encoding of ['dense', 'csr', 'csc']) test(`${encoding}: missing values excluded, unequal merged populations weighted by cells`, () => withReader(`plot_${encoding}`, reader => {
  const request = spec({ markers: ['signal', 'zero'] });
  const result = readPlot(reader, request);
  assert.deepEqual(result.rows.map(row => row.value), ['C', 'B', 'A']);
  assert.deepEqual(result.means, [[2, 0], [6, 0], [3, 0]]);
  assert.deepEqual(result.finiteCounts, [[1, 1], [2, 1], [2, 3]]);
  const plan = editRemap(createRemap('merged', 'Review', 'population', reader.populationSummary('population')), ['"A"', '"C"'], { label: 'AC' });
  const merged = readPlot(reader, { ...request, settings: { ...request.settings, groupBy: 'remap:merged' }, remaps: [plan] });
  assert.equal(merged.means[0][0], 8 / 3);
  const filtered = readPlot(reader, spec({ markers: ['signal'], filterBy: 'condition', filterValues: ['"x"'] }));
  assert.equal(filtered.selectedCells, 3);
  assert.deepEqual(filtered.rows.map(row => row.value), ['B', 'A']);
  assert.deepEqual(filtered.means, [[9], [0]]);
}));

test('composition percentages, typed missing categories, filtering and label comparison', () => withReader('dense', reader => {
  const bars = readPlot(reader, spec({ type: 'bars' }));
  assert.deepEqual(bars.values, [[50, 0], [0, 50], [50, 0], [0, 50]]);
  assert.deepEqual(readPlot(reader, spec({ type: 'composition', measure: 'count' })).values, [[8, 0], [0, 8], [8, 0], [0, 8]]);
  const filtered = readPlot(reader, spec({ type: 'bars', populations: ['"T cell"', '"B cell"'], order: 'count-desc', sortPopulation: '"B cell"' }));
  assert.equal(filtered.columns[0].label, 'ROI_A_2');
  assert.deepEqual(filtered.values.map(row => row.reduce((a, b) => a + b)), [100, 100]);
  const plan = editRemap(createRemap('merged', 'Review', 'population', reader.populationSummary('population')), ['"T cell"', '"B cell"'], { label: 'Lymphocytes' });
  const comparison = readPlot(reader, { ...spec({ type: 'comparison', groupBy: 'remap:merged', compareBy: 'obs:population', normalization: 'column_percent' }), remaps: [plan] });
  assert.deepEqual(comparison.values, [[50, 0, 0], [50, 0, 0], [0, 100, 0], [0, 0, 100]]);
  assert.equal(comparison.counts.flat().reduce((a, b) => a + b), 32);
}));

test('display scaling and counts have defined constant/empty behaviour', () => {
  assert.deepEqual(scaleMeans([[0, 1], [0, 3], [null, null]], 'minmax'), [[0, 0], [0, 1], [null, null]]);
  assert.deepEqual(scaleMeans([[0, 1], [0, 3]], 'zscore'), [[0, -1], [0, 1]]);
  assert.deepEqual(normaliseCounts([[0, 0], [1, 3]], 'row_percent'), [[0, 0], [25, 75]]);
  assert.deepEqual(labelCategories([{ value: 1 }, { value: '1' }, { value: null }, { value: '(Missing)' }]).map(row => row.label), ['1 [number]', '1 [string]', '(Missing) [missing]', '(Missing) [string]']);
});

test('invalid requests fail clearly instead of silently changing the plot scope', () => withReader('dense', reader => {
  assert.throws(() => readPlot(reader, spec({ markers: [] })), /markers/);
  assert.throws(() => readPlot(reader, spec({ scope: 'selected' })), /ROI/);
  assert.throws(() => readPlot(reader, spec({ populations: ['"absent"'] })), /No cells/);
  assert.throws(() => readPlot(reader, spec({ type: 'comparison', compareBy: 'obs:population' })), /different/);
  assert.throws(() => validatePlots({ ...defaultPlots(), markers: Array(101).fill('a') }), /markers/);
  assert.throws(() => validatePlots({ ...defaultPlots(), filterValues: ['{}'] }), /filter/);
}));

test('CSV exports exact means/counts and snapshot settings; SVG escapes user labels', () => withReader('dense', reader => {
  const result = readPlot(reader, spec({ title: '<script>"unsafe" & label</script>', scale: 'zscore' }));
  const csv = plottedDataCSV(result), svg = plotSVG(result);
  assert.ok(csv.includes('"43","8","32","X"'));
  assert.ok(csv.includes('"null","(Missing)"'));
  assert.ok(svg.startsWith('<svg')); assert.ok(!svg.includes('<script>')); assert.ok(svg.includes('&lt;script&gt;'));
  assert.ok(svg.includes('Marker-wise z-score'));
  assert.ok(plotSVG({ ...result, settings: { ...result.settings, swapAxes: true } }).includes('CD20'));
}));

test('version 4 workspace preserves plot settings and older workspaces get defaults', () => {
  const channels = Array.from({ length: 6 }, () => ({ active: false, enabled: true, name: '', path: '', plane: 0, low: 0, high: 1, colour: '#ff0000', auto: true }));
  const overlay = { visible: true, colourBy: '', source: 'X', population: '', style: 'outline', opacity: .8 };
  const plots = { ...defaultPlots(), type: 'matrix', groupBy: 'obs:population', markers: ['CD20'], scope: 'selected', rois: ['ROI_A'], filterBy: 'nullable', filterValues: ['null'] };
  const input = { format: 'sbt-tissue-explorer-workspace', version: 1, sources: { anndata: { name: 'dense.h5ad', size: 1 }, images: null, masks: null }, view: { mapping: { roi: 'ROI', id: 'ObjectNumber' }, roi: 'ROI_A', channels, imagesVisible: true, overlay, embedding: '', selected: null, normalization: { dictionary: null, name: '', quantile: .99, minimum: .1 }, camera: { fitted: true, scale: 1, centreX: 0, centreY: 0 }, panels: Object.fromEntries(PANEL_IDS.map(id => [id, true])) } };
  const upgraded = parseWorkspace(JSON.stringify(input)); assert.deepEqual(upgraded.view.plots, defaultPlots());
  upgraded.view.plots = plots; upgraded.view.review.tab = 'plots';
  assert.deepEqual(parseWorkspace(JSON.stringify(upgraded)).view.plots, plots);
  upgraded.version = 3; delete upgraded.view.plots;
  assert.deepEqual(parseWorkspace(JSON.stringify(upgraded)).view.plots, defaultPlots());
});
