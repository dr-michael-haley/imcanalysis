import { defaultScaleBar } from '../src/figure.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRemap, editRemap, applyRemap, remapCSV, remapRecipe, sourceKey, validateRemaps, validateRemapSource } from '../src/remaps.js';
import { parseWorkspace, PANEL_IDS, WORKSPACE_VERSION } from '../src/workspace.js';
import { recipeFingerprint } from '../src/views.js';

const summary = { rows: [{ source: 'T', count: 8 }, { source: 'B', count: 7 }, { source: null, count: 1 }], palette: ['#ff0000', '#00ff00', '#808080'] };
const draft = () => createRemap('draft-1', 'immune_reviewed', 'population', summary);
const keys = (...values) => values.map(sourceKey);

test('typed sources and missing values stay distinct until an explicit merge', () => {
  const plan = createRemap('typed', 'Typed', 'cluster', { rows: [1, '1', null, '(Missing)', '', false].map(source => ({ source, count: 1 })) });
  assert.equal(new Set(plan.rows.map(row => row.label)).size, 6);
  const region = { values: [1, '1', null, '', false], rows: [0, 1, 2, 3, 4] };
  assert.equal(new Set(applyRemap(region, plan).values).size, 5);
  assert.deepEqual(region.values, [1, '1', null, '', false]);
});

test('merges harmonise colours and preserve notes; recolouring propagates to all contributors', () => {
  let plan = editRemap(draft(), keys('B'), { notes: 'Inspect CD20\nLow confidence, "review"' });
  plan = editRemap(plan, keys('T', 'B'), { label: 'Lymphocytes' });
  assert.equal(plan.rows[0].colour, plan.rows[1].colour);
  plan = editRemap(plan, keys('B'), { colour: '#123ABC' });
  assert.deepEqual(plan.rows.slice(0, 2).map(row => row.colour), ['#123abc', '#123abc']);
  assert.equal(plan.rows[1].notes, 'Inspect CD20\nLow confidence, "review"');
  assert.equal(plan.rows[0].notes, '');
  const displayed = applyRemap({ values: ['B', 'T', null] }, plan);
  assert.deepEqual(displayed.values, ['Lymphocytes', 'Lymphocytes', '(Missing)']);
  assert.deepEqual(displayed.categories, ['Lymphocytes', '(Missing)']);
  assert.deepEqual(displayed.palette, ['#123abc', '#808080']);
  assert.throws(() => applyRemap({ values: ['unknown'] }, plan), /no entry/);
});

test('renaming into an existing group adopts its colour without losing source identity', () => {
  const plan = editRemap(draft(), keys('T'), { label: 'B' });
  assert.deepEqual(plan.rows.slice(0, 2).map(row => row.colour), ['#00ff00', '#00ff00']);
  assert.deepEqual(plan.rows.map(row => row.source), ['T', 'B', null]);
  assert.throws(() => editRemap(plan, keys('B'), { label: ' ' }), /Invalid remap/);
});

test('malformed or ambiguous saved remaps fail validation', () => {
  const plan = draft();
  assert.throws(() => validateRemaps([plan, plan]), /unique/);
  assert.throws(() => validateRemaps([{ ...plan, rows: [plan.rows[0], plan.rows[0]] }]), /Duplicate source/);
  assert.throws(() => validateRemaps([{ ...plan, rows: plan.rows.map(row => ({ ...row, label: 'Merged' })) }]), /conflicting colours/);
  assert.throws(() => validateRemaps([{ ...plan, rows: [{ ...plan.rows[0], source: {} }] }]), /source value/);
  assert.throws(() => validateRemaps([{ ...plan, rows: [{ ...plan.rows[0], notes: 'x'.repeat(4001) }] }]), /notes/);
  assert.throws(() => validateRemaps([{ ...plan, rows: [{ ...plan.rows[0], colour: 'red' }] }]), /RRGGBB/);
});

test('restoration validates full-dataset populations and counts, independent of row order', () => {
  validateRemapSource(draft(), { rows: [...summary.rows].reverse() });
  assert.throws(() => validateRemapSource(draft(), { rows: summary.rows.slice(1) }), /does not match/);
  assert.throws(() => validateRemapSource(draft(), { rows: summary.rows.map(row => ({ ...row, count: row.count + 1 })) }), /does not match/);
});

test('CSV quotes source identity, Unicode, commas, quotes and multiline notes losslessly', () => {
  const plan = editRemap(draft(), keys('T'), { label: 'T, α', notes: 'First "line"\nSecond' });
  const csv = remapCSV(plan);
  assert.ok(csv.startsWith('"source_obs","source_value","source_value_json","proposed_name","colour","notes","remap_name"\r\n'));
  assert.ok(csv.includes('"population","T","""T""","T, α","#ff0000","First ""line""\nSecond","immune_reviewed"'));
  assert.ok(csv.includes('"population","(Missing)","null"'));
});

function workspace(version = 3) {
  const channels = Array.from({ length: 6 }, () => ({ active: false, enabled: true, name: '', path: '', plane: 0, low: 0, high: 1, colour: '#ff0000', auto: true }));
  const overlay = { visible: true, colourBy: 'remap:draft-1', source: 'X', population: '', style: 'outline', opacity: .8 };
  return { format: 'sbt-tissue-explorer-workspace', version, sources: { anndata: { name: 'data.h5ad', size: 123 }, masks: null, images: null }, view: {
    mapping: { roi: 'ROI', id: 'ObjectNumber' }, roi: 'A', channels, imagesVisible: true, overlay,
    presets: [{ name: 'Proposed', channels, imagesVisible: true, overlay }], selectedView: 0, tab: 'views',
    order: { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 }, visits: [],
    embedding: '', selected: null, normalization: { dictionary: null, name: '', quantile: .999, minimum: .1 },
    camera: { fitted: true, scale: 1, centreX: 10, centreY: 10 }, panels: Object.fromEntries(PANEL_IDS.map(id => [id, true])),
    remaps: [editRemap(draft(), keys('T'), { notes: 'Proposed annotation' })], activeRemap: 'draft-1', review: { tab: 'remap', height: 460 },
  } };
}
test('workspace roundtrip preserves remaps, exact types, notes, overlays and bottom-panel settings', () => {
  const input = workspace(), restored = parseWorkspace(JSON.stringify(input));
  assert.equal(restored.version, WORKSPACE_VERSION);
  for (const field of ['remaps', 'activeRemap', 'review']) assert.deepEqual(restored.view[field], input.view[field]);
  assert.deepEqual(restored.view.presets[0], { ...input.view.presets[0], channels: input.view.presets[0].channels.map(c => ({ ...c, showInLegend: true })), layers: [], legendVisible: true, legendScale: 1, scaleBar: defaultScaleBar(), overlay: { ...input.view.presets[0].overlay, showInLegend: true, populations: [], colourmap: 'legacy', reverse: false, vmin: null, vmax: null } });
  input.view.remaps = [];
  assert.throws(() => parseWorkspace(JSON.stringify(input)), /remap/);
});
test('version 1 and 2 workspaces acquire safe defaults for the new controls', () => {
  for (const version of [1, 2]) {
    const input = workspace(version); delete input.view.remaps; delete input.view.activeRemap; delete input.view.review;
    input.view.overlay.colourBy = 'obs:population';
    const parsed = parseWorkspace(JSON.stringify(input));
    assert.deepEqual(parsed.view.remaps, []); assert.deepEqual(parsed.view.review, { tab: 'embedding', height: 280 });
    assert.equal(parsed.view.presets[0].overlay.colourBy, 'obs:population');
  }
});
test('visit recipes change with proposed names and colours, but not notes or remap titles', async () => {
  const plan = draft();
  const view = { name: 'Test', channels: [], imagesVisible: true, overlayVisible: true, overlay: { colourBy: 'remap:draft-1' }, remapRecipe: remapRecipe(plan) };
  const before = await recipeFingerprint(view, null);
  view.legendScale = 1.8; view.legendVisible = false;
  assert.equal(await recipeFingerprint(view, null), before);
  view.remapRecipe = remapRecipe(editRemap(plan, keys('T'), { notes: 'Only notes changed' }));
  assert.equal(await recipeFingerprint(view, null), before);
  view.remapRecipe = remapRecipe(editRemap(plan, keys('T'), { colour: '#abcdef' }));
  assert.notEqual(await recipeFingerprint(view, null), before);
});
