import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import h5wasm from 'h5wasm/node';
import { AnnDataReader } from '../src/anndata.js';
import { CellSelection, assignSelection } from '../src/selection.js';
import { validateNavigation, validateSnapshot, validateEmbedding, defaultEmbedding, presetForKey } from '../src/navigation.js';
import { parseWorkspace, WORKSPACE_FORMAT, WORKSPACE_VERSION, PANEL_IDS } from '../src/workspace.js';
import { readEmbeddingContext, readEmbeddingCurrent, sampledRows } from '../src/embedding-reader.js';
import { readPlot } from '../src/plot-reader.js';
import { defaultPlots, plottedDataCSV } from '../src/plots.js';
import { alluvialLayout } from '../src/alluvial.js';
import { defaultLayer } from '../src/aligned-layers.js';
import { plotSVG } from '../src/plot-svg.js';

const pane = () => ({ name: 'RGB', imagesVisible: true, legendVisible: true, legendScale: 1.2, layers: [], channels: Array.from({ length: 6 }, (_, i) => ({ active: i < 3, enabled: true, name: `marker${i}`, sourceId: '', path: '', plane: 0, low: 0, high: 10, colour: '#ff0000', auto: false })), overlay: { colourBy: 'obs:population', source: 'X', visible: true, population: '"T cell"', style: 'outline', opacity: .8, colourmap: 'viridis', reverse: false } });
const snapshot = () => ({ roi: 'ROI_A', panes: [pane(), { ...pane(), name: 'Other' }], selectedView: 1, selected: 300, camera: { fitted: false, scale: 3, centreX: 25, centreY: -10 }, normalization: { dictionary: null, name: '', quantile: .999, minimum: .1 }, embedding: 'X_umap', embeddingOptions: { ...defaultEmbedding(), context: true, square: true } });
const navigation = () => ({ bookmarks: [{ id: 'b', name: 'Interesting cells', comment: 'Review T cells', createdAt: '2026-10-27T00:00:00Z', viewer: snapshot() }], savedViews: [{ id: 'p', name: 'Lymphocytes', hotkey: 'F2', settings: pane() }] });

test('v12 migrates legend defaults and v13 retains per-pane flattening in workspaces, bookmarks and presets', () => {
  const input = document(12), old = parseWorkspace(JSON.stringify(input));
  assert.equal(old.view.presets[0].channels[0].showInLegend, true); assert.equal(old.view.presets[0].overlay.showInLegend, true);
  input.sources.layers = [{ id: 'r', name: 'Regions', kind: 'labels', folder: { name: 'Regions', count: 1, signature: 'a'.repeat(64) } }];
  input.view.layers = [defaultLayer('r', 'labels')];
  const panes = [...input.view.presets, ...input.view.bookmarks[0].viewer.panes, input.view.savedViews[0].settings];
  for (const [i, pane] of panes.entries()) {
    pane.channels[0].showInLegend = false; pane.overlay.showInLegend = false;
    pane.layers = [{ ...defaultLayer('r', 'labels'), flatten: i % 2 === 0, flatLabel: 'All tissue', flatColour: '#ff0000', showInLegend: false }];
  }
  const restored = parseWorkspace(JSON.stringify(input));
  const saved = [...restored.view.presets, ...restored.view.bookmarks[0].viewer.panes, restored.view.savedViews[0].settings];
  for (const [i, pane] of saved.entries()) {
    assert.equal(pane.channels[0].showInLegend, false); assert.equal(pane.overlay.showInLegend, false); assert.deepEqual(pane.layers, panes[i].layers);
  }
  assert.deepEqual(parseWorkspace(JSON.stringify(restored)), restored);
  input.view.presets[0].channels[0].showInLegend = 'no'; assert.throws(() => parseWorkspace(JSON.stringify(input)), /legend visibility/);
});
function document(version = WORKSPACE_VERSION) { const v = snapshot(); return { format: WORKSPACE_FORMAT, version, sources: { anndata: { name: 'data.h5ad', size: 100 }, images: null, masks: null, layers: [] }, view: { ...v.panes[0], ...navigation(), mapping: { roi: 'ROI', id: 'ObjectNumber' }, roi: v.roi, presets: v.panes, selectedView: 1, selected: v.selected, tab: 'views', order: { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 }, visits: [], normalization: v.normalization, camera: v.camera, embedding: v.embedding, embeddingOptions: v.embeddingOptions, panels: Object.fromEntries(PANEL_IDS.map(k => [k, true])), review: { tab: 'bookmarks', height: 320 } } }; }
test('bookmarks and pane presets round-trip without losing camera, comments or overlay settings', () => {
  const parsed = parseWorkspace(JSON.stringify(document()));
  assert.deepEqual(parsed.view.bookmarks[0].viewer, validateSnapshot(snapshot()));
  assert.equal(parsed.view.bookmarks[0].comment, 'Review T cells'); assert.equal(parsed.view.savedViews[0].hotkey, 'F2'); assert.equal(parsed.view.review.tab, 'bookmarks');
  assert.deepEqual(parsed.view.embeddingOptions, snapshot().embeddingOptions);
  const old = document(8); delete old.view.bookmarks; delete old.view.savedViews; delete old.view.embeddingOptions;
  const migrated = parseWorkspace(JSON.stringify(old)); assert.deepEqual(migrated.view.bookmarks, []); assert.deepEqual(migrated.view.savedViews, []); assert.deepEqual(migrated.view.embeddingOptions, defaultEmbedding());
});
test('saved states reject invalid cameras, channels, duplicate IDs and conflicting hotkeys', () => {
  const n = navigation(); assert.throws(() => validateNavigation(n.bookmarks, [...n.savedViews, { ...n.savedViews[0], id: 'p2' }]), /hotkey/);
  assert.throws(() => validateNavigation([...n.bookmarks, n.bookmarks[0]]), /duplicate/);
  const v = snapshot(); v.camera.scale = 0; assert.throws(() => validateSnapshot(v), /camera/);
  const bad = snapshot(); bad.panes[0].channels[0].high = -1; assert.throws(() => validateSnapshot(bad), /channel/);
  assert.throws(() => validateEmbedding({ contextLimit: 5000000 }), /sample/);
  assert.throws(() => validateNavigation([], [{ ...n.savedViews[0], hotkey: 'F5' }]), /hotkey/);
});

test('colour limits persist independently in panes, embedding, plots, bookmarks and presets', () => {
  const input = structuredClone(document());
  input.view.overlay = { ...input.view.overlay };
  Object.assign(input.view.overlay, { vmin: -1, vmax: 5 });
  Object.assign(input.view.presets[0].overlay, { vmin: 0, vmax: 20 });
  Object.assign(input.view.embeddingOptions, { vmin: null, vmax: 15 });
  input.view.plots = { ...defaultPlots(), vmin: -2, vmax: 2 };
  Object.assign(input.view.bookmarks[0].viewer.panes[0].overlay, { vmin: 3, vmax: 9 });
  Object.assign(input.view.savedViews[0].settings.overlay, { vmin: 1, vmax: null });
  const parsed = parseWorkspace(JSON.stringify(input)), again = parseWorkspace(JSON.stringify(parsed));
  assert.equal(parsed.version, WORKSPACE_VERSION); assert.deepEqual(again, parsed);
  assert.equal(parsed.view.overlay.vmin, -1); assert.equal(parsed.view.presets[0].overlay.vmax, 20);
  assert.equal(parsed.view.embeddingOptions.vmin, null); assert.equal(parsed.view.embeddingOptions.vmax, 15);
  assert.equal(parsed.view.plots.vmin, -2); assert.equal(parsed.view.bookmarks[0].viewer.panes[0].overlay.vmin, 3);
  assert.equal(parsed.view.savedViews[0].settings.overlay.vmin, 1);
  const old = parseWorkspace(JSON.stringify(document(11)));
  assert.equal(old.view.overlay.vmin, null); assert.equal(old.view.overlay.vmax, null);
  input.view.presets[0].overlay.vmax = 0; assert.throws(() => parseWorkspace(JSON.stringify(input)), /Lower colour/);
});
test('preset hotkeys ignore typing, repeats and modified shortcuts', () => {
  const presets = navigation().savedViews; assert.equal(presetForKey(presets, { key: 'F2' }), presets[0]);
  for (const event of [{ key: 'F2', ctrlKey: true }, { key: 'F2', repeat: true }, { key: 'F2', target: { closest: () => ({}) } }, { key: 'F3' }]) assert.equal(presetForKey(presets, event), null);
});
test('cross-ROI selection and flag undo preserve identity even when mask IDs are reused', () => {
  const s = new CellSelection(); s.add('A', 1); s.add('B', 1); s.add('B', 2); s.toggle('A', 1); assert.equal(s.size, 2); assert.equal(s.has('A', 1), false);
  const field = { categories: [{ label: 'Review' }], assignments: [['A', 1, 'Review'], ['B', 2, 'Review']] };
  const undo = assignSelection(field, s, 'Review'); assert.deepEqual(field.assignments, [['A', 1, 'Review'], ['B', 1, 'Review'], ['B', 2, 'Review']]);
  s.clear(); undo(); assert.deepEqual(field.assignments, [['A', 1, 'Review'], ['B', 2, 'Review']]);
  assert.throws(() => s.add('A', -1), /identity/);
});
test('context sampling stays bounded, deterministic and covers interleaved row patterns', () => {
  const rows = sampledRows(5000000, 50000); assert.equal(rows.length, 50000); assert.equal(new Set(rows).size, 50000);
  assert.ok(rows.every((v, i) => v >= 0 && v < 5000000 && (!i || v > rows[i - 1]))); assert.deepEqual(rows, sampledRows(5000000, 50000));
  const even = rows.filter(r => r % 2 === 0).length; assert.ok(even > 20000 && even < 30000); assert.deepEqual(sampledRows(3, 10), [0, 1, 2]);
});
await h5wasm.ready;
for (const encoding of ['dense', 'csr', 'csc']) test(`${encoding}: embedding identities and colours match exact AnnData rows and sources`, () => {
  const file = new h5wasm.File(fileURLToPath(new URL(`../fixtures/${encoding}.h5ad`, import.meta.url)), 'r');
  try {
    const reader = new AnnDataReader(file), req = { embedding: 'X_umap', roiKey: 'ROI', idKey: 'ObjectNumber', colourBy: 'var:CD20', source: 'layers/counts', limit: 1000 };
    const context = readEmbeddingContext(reader, req); assert.equal(context.ids.length, 32); assert.equal(context.coordinates.length, 64); assert.equal(context.sampled, false);
    assert.equal(context.rois[0], 'ROI_A'); assert.equal(context.rois[1], 'ROI_A_2'); assert.deepEqual(context.values, Array.from({ length: 32 }, (_, r) => (r * 3 + 1) * 2));
    const current = readEmbeddingCurrent(reader, { rows: [0, 2, 6], colourBy: 'var:CD20', source: 'X' }); assert.deepEqual(Array.from(current.values), [1, 7, 19]);
    const categorical = readEmbeddingContext(reader, { ...req, colourBy: 'obs:population' }); assert.deepEqual(categorical.categories, ['T cell', 'B cell', 'Stroma']); assert.equal(categorical.values[3], null); assert.equal(categorical.palette[0], '#ff6666');
  } finally { file.close(); }
});
test('alluvial ribbons conserve counts at every node and escape labels in SVG', () => {
  const counts = [[3, 4, 0], [7, 1, 5]], layout = alluvialLayout(counts);
  assert.equal(layout.total, 20);
  for (const [nodes, key] of [[layout.left, 'source'], [layout.right, 'target']]) nodes.forEach((n, i) => {
    assert.equal(layout.flows.filter(f => f[key] === i).reduce((s, f) => s + f.count, 0), n.count);
    assert.ok(Math.abs(layout.flows.filter(f => f[key] === i).reduce((s, f) => s + f.height, 0) - n.height) < 1e-10);
    assert.ok(n.y >= 0 && n.y + n.height <= layout.height + 1e-10);
  });
  const file = new h5wasm.File(fileURLToPath(new URL('../fixtures/dense.h5ad', import.meta.url)), 'r');
  try {
    const result = readPlot(new AnnDataReader(file), { settings: { ...defaultPlots(), type: 'alluvial', groupBy: 'obs:population', compareBy: 'obs:ROI', scope: 'all', title: '<script>bad</script>' } });
    assert.equal(result.counts.flat().reduce((a, b) => a + b), 32); assert.deepEqual(result.values, result.counts); assert.equal(result.valueLabel, 'Cell count');
    const svg = plotSVG(result); assert.ok(svg.includes('alluvial-flow')); assert.ok(!svg.includes('<script>')); assert.ok(svg.includes('&lt;script&gt;')); assert.ok(plottedDataCSV(result).includes('alluvial'));
  } finally { file.close(); }
});
