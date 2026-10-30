import test from 'node:test';
import assert from 'node:assert/strict';
import { indexLayer, parseLabelMapping, defaultLayer, prepareLayer, blendLayer, labelDictionary, labelEdgeDistance, validateLayerSettings } from '../src/aligned-layers.js';
import { recipeFingerprint } from '../src/views.js';
import { defaultFigure, paneFurniture } from '../src/figure.js';
import { folderReference, resolveWorkspaceRoot, parseWorkspace, WORKSPACE_VERSION, PANEL_IDS } from '../src/workspace.js';
import { legendGroups } from '../src/view-legend.js';
import { resizedLegend } from '../src/view-legend.js';
import { imageCatalogue, channelChoices } from '../src/image-channels.js';
import { rasterIdentity } from '../src/raster.js';

const file = (path, contents = 'test') => { const f = new File([contents], path.split('/').at(-1)); Object.defineProperty(f, 'webkitRelativePath', { value: path }); return f; };
test('aligned assets match whole ROI names, retain absent ROIs, and reject ambiguous images', () => {
  const files = [file('H/ROI_A.png'), file('H/ROI_A_2.ome.tif'), file('H/not_ROI_A.tif')];
  const map = indexLayer(files, ['ROI_A', 'ROI_A_2', 'ROI_B']);
  assert.equal(map.size, 2); assert.equal(map.get('ROI_A'), files[0]); assert.equal(map.get('ROI_A_2'), files[1]); assert.equal(map.has('ROI_B'), false);
  assert.throws(() => indexLayer([...files, file('H/ROI_A.jpg')], ['ROI_A']), /More than one/);
});
test('Napari label CSV preserves quoted names, last duplicate wins, and zero stays transparent', () => {
  const mapping = parseLabelMapping('\uFEFFvalue,label,color\r\n0,Background,#ffffff\r\n1,"A, B",#00ffff\r\n2,Old,#ffffff\r\n2,New,#ff0000');
  assert.deepEqual(mapping, [{ value: 0, label: 'Background', color: 'transparent' }, { value: 1, label: 'A, B', color: '#00ffff' }, { value: 2, label: 'New', color: '#ff0000' }]);
  for (const bad of ['value\n1', 'value,label,color\n-1,A,#ffffff', 'value,label,color\n1.5,A,#ffffff', 'value,label,color\n1,A,junk']) assert.throws(() => parseLabelMapping(bad));
});
test('IMC collections use original image folder rules without AnnData channel aliases', () => {
  const files = [file('Extra/ROI_A/44_90_Ir193_DNA3.tiff'), file('Extra/ROI_A/Other.tif'), file('Extra/ROI_A_2/55_20_Ir193_DNA3.tiff')];
  const index = indexLayer(files, ['ROI_A', 'ROI_A_2'], 'imc');
  assert.equal(index.get('ROI_A')[0].name, 'DNA3'); assert.equal(index.get('ROI_A_2')[0].name, 'DNA3'); assert.equal(index.get('ROI_A')[1].name, 'Other');
});
test('different selected files with identical relative names and sizes never share decoded pixels', () => {
  const a = file('Images/ROI_A/DNA3.tiff'), b = file('Images/ROI_A/DNA3.tiff');
  assert.notEqual(rasterIdentity(a), rasterIdentity(b)); assert.equal(rasterIdentity(a), rasterIdentity(a));
});
test('external annotation dictionaries preserve original IDs and distinguish merged-label columns', () => {
  const csv = 'original_num,name,label\n0,*Background*,0\n1,MVP,4\n2,CT,3\n3,CT_other,3';
  const parsed = labelDictionary(csv);
  assert.deepEqual(parsed.columns, { value: 'original_num', label: 'name', color: '' });
  assert.deepEqual(parsed.mapping.map(item => [item.value, item.label]), [[0, '*Background*'], [1, 'MVP'], [2, 'CT'], [3, 'CT_other']]);
  assert.match(parsed.mapping[2].color, /^#[a-f0-9]{6}$/); assert.equal(parsed.mapping[0].color, 'transparent');
  assert.throws(() => labelDictionary(csv, { value: 'label', label: 'name', color: '' }), /multiple names/);
});
test('compositing preserves RGB alpha, scalar ranges, and label background / unmapped transparency', () => {
  const render = (kind, data, settings = {}, mapping = []) => {
    const pixels = new Uint8ClampedArray([100, 100, 100, 255]);
    const layer = prepareLayer({ ...defaultLayer('x', kind), ...settings, raster: { data } }, { kind, mapping });
    if (layer) blendLayer(pixels, 0, layer, 0, 1, 1);
    return [...pixels];
  };
  assert.deepEqual(render('rgb', new Uint8Array([200, 0, 100, 0])), [100, 100, 100, 255]);
  assert.deepEqual(render('rgb', new Uint8Array([200, 0, 100, 255]), { opacity: .5 }), [150, 50, 100, 255]);
  assert.deepEqual(render('scalar', [5], { low: 0, high: 10 }), [128, 128, 128, 255]);
  assert.deepEqual(render('scalar', [NaN]), [100, 100, 100, 255]);
  const mapping = [{ value: 1, label: 'A', color: '#ff0000' }];
  assert.deepEqual(render('labels', [0], {}, mapping), [100, 100, 100, 255]);
  assert.deepEqual(render('labels', [2], {}, mapping), [100, 100, 100, 255]);
  assert.deepEqual(render('labels', [1], {}, mapping), [178, 50, 50, 255]);
  assert.deepEqual(render('labels', [1], { visible: false }, mapping), [100, 100, 100, 255]);
});
test('one root restores mixed formats and annotation mappings; omitted or changed mapping fails', async () => {
  const csv = 'value,label,color\n1,Tissue,#ff0000';
  const sources = { anndata: { name: 'data.h5ad', size: 4 }, masks: null, images: null, layers: [
    { id: 'h', name: 'Histology', kind: 'rgb', folder: await folderReference([file('H/ROI_A.png')]) },
    { id: 'r', name: 'Regions', kind: 'labels', folder: await folderReference([file('R/ROI_A.tif'), file('R/label_mapping.csv', csv)]) },
  ] };
  const files = [file('root/data.h5ad'), file('root/H/ROI_A.png'), file('root/R/ROI_A.tif'), file('root/R/label_mapping.csv', csv), file('root/R/notes.csv')];
  const restored = await resolveWorkspaceRoot(files, sources);
  assert.equal(restored.layers.length, 2); assert.equal(restored.layers[1].mapping[0].label, 'Tissue'); assert.equal(restored.layers[1].files.length, 2);
  await assert.rejects(resolveWorkspaceRoot(files.filter(f => f.name !== 'label_mapping.csv'), sources), /Regions/);
  const external = file('root/dictionaries/custom.csv', 'original_num,name,label\n0,Background,0\n1,Example region,7');
  sources.layers[1].dictionary = { file: { name: external.name, size: external.size }, columns: { value: 'original_num', label: 'name', color: '' } };
  const separate = await resolveWorkspaceRoot([...files, external], sources);
  assert.equal(separate.layers[1].mapping[1].label, 'Example region'); assert.equal(separate.layers[1].dictionary.columns.value, 'original_num');
  await assert.rejects(resolveWorkspaceRoot(files, sources), /custom.csv/);
});
test('v6 retains layer source order and independent pane settings, rejects dangling references, and migrates v5', async () => {
  const channels = Array.from({ length: 6 }, () => ({ active: false, enabled: true, name: '', path: '', plane: 0, low: 0, high: 1, colour: '#ff0000', auto: true }));
  const overlay = { visible: true, colourBy: '', source: 'X', population: '', style: 'fill', opacity: .8 };
  const input = { format: 'sbt-tissue-explorer-workspace', version: 5, sources: { anndata: { name: 'test.h5ad', size: 123 }, masks: null, images: null }, view: {
    mapping: { roi: 'ROI', id: 'ObjectNumber' }, roi: 'ROI_A', channels, overlay, imagesVisible: true, presets: [{ name: 'QC', channels, overlay, imagesVisible: true }], selectedView: 0, tab: 'views',
    order: { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 }, visits: [], embedding: '', selected: null,
    normalization: { dictionary: null, name: '', quantile: .999, minimum: .1 }, camera: { fitted: true, scale: 1, centreX: 0, centreY: 0 }, panels: Object.fromEntries(PANEL_IDS.map(id => [id, true])),
  } };
  const older = parseWorkspace(JSON.stringify(input)); assert.deepEqual(older.sources.layers, []); assert.deepEqual(older.view.presets[0].layers, []);
  input.version = WORKSPACE_VERSION;
  input.sources.layers = [{ id: 'h', name: 'H&E', kind: 'rgb', folder: await folderReference([file('H/ROI_A.png')]) }];
  input.view.layers = [defaultLayer('h', 'rgb')];
  input.view.presets = [0, 1].map(i => ({ ...input.view.presets[0], name: `View ${i}`, layers: [{ ...defaultLayer('h', 'rgb'), opacity: i ? .3 : 1, visible: !i }] }));
  const restored = parseWorkspace(JSON.stringify(input)); assert.deepEqual(restored.sources.layers, input.sources.layers); assert.deepEqual(restored.view.presets.map(p => p.layers), input.view.presets.map(p => p.layers));
  input.sources.layers.push({ id: 'extra', name: 'Extra IMC', kind: 'imc', folder: await folderReference([file('Extra/ROI_A/DNA3.tiff')]) });
  input.view.presets[0].channels[0] = { ...channels[0], sourceId: 'extra', name: 'DNA3', active: true }; input.view.presets[0].legendVisible = false;
  input.view.presets[0].legendScale = 1.6;
  input.sources.anndata.path = 'analysis/test.h5ad'; input.sources.layers[0].folder.path = 'aligned/H';
  input.view.representative = { top: 6, source: 'X', roiOnly: true, lowerQuantile: .01, upperQuantile: .99 };
  const latest = parseWorkspace(JSON.stringify(input)); assert.equal(latest.view.presets[0].channels[0].sourceId, 'extra'); assert.equal(latest.view.presets[0].legendVisible, false); assert.equal(latest.view.representative.upperQuantile, .99);
  assert.equal(latest.view.presets[0].legendScale, 1.6); assert.equal(latest.view.presets[1].legendScale, 1);
  assert.equal(latest.sources.anndata.path, 'analysis/test.h5ad'); assert.equal(latest.sources.layers[0].folder.path, 'aligned/H');
  input.sources.anndata.path = '../test.h5ad'; assert.throws(() => parseWorkspace(JSON.stringify(input)), /relative path/); input.sources.anndata.path = 'analysis/test.h5ad';
  input.view.presets[0].legendScale = 2.1; assert.throws(() => parseWorkspace(JSON.stringify(input)), /legend size/); input.view.presets[0].legendScale = 1.6;
  input.view.presets[1].layers[0].id = 'missing'; assert.throws(() => parseWorkspace(JSON.stringify(input)), /sources/);
});

test('floating legends include only visible content and honour the selected population', () => {
  const view = { imagesVisible: true, channels: [{ active: true, enabled: true, raster: {}, name: 'DNA3', sourceId: 'imc', colour: '#ff0000' }], layers: [], overlayVisible: true, overlay: { style: 'fill', colourBy: 'obs:population', population: '"T"' }, region: {}, colours: { entries: [['T', [255, 0, 0]], ['B', [0, 255, 0]]] } };
  const groups = legendGroups(view, [{ id: 'imc', name: 'Extra' }]);
  assert.equal(groups[0].rows[0].name, 'Extra · DNA3'); assert.deepEqual(groups[1].rows.map(row => row.name), ['T']);
  view.imagesVisible = false; view.overlayVisible = false; assert.deepEqual(legendGroups(view, []), []);
});

test('extra IMC channels stay selectable across ROIs and do not collide with original markers', () => {
  const original = { name: 'DNA3', key: 'Images/ROI_A/DNA3.tiff', file: {} };
  const extra = { name: 'DNA3', sourceId: 'extra', key: 'extra:ROI_B/DNA3.tiff', file: {} };
  const index = new Map([['ROI_A', { images: [original] }], ['ROI_B', { images: [extra] }]]);
  const catalogue = imageCatalogue(index), a = channelChoices(index, catalogue, 'ROI_A'), b = channelChoices(index, catalogue, 'ROI_B');
  assert.equal(catalogue.length, 2); assert.equal(a[0], original); assert.equal(a[1].name, 'DNA3'); assert.equal(a[1].file, null);
  assert.equal(a[1].sourceId, 'extra'); assert.equal(b[1], extra); assert.equal(b[0].file, null);
});

test('legend size buttons stop at the supported bounds', () => {
  assert.equal(resizedLegend(undefined, 1), 1.1); assert.equal(resizedLegend(.7, -1), .7); assert.equal(resizedLegend(2, 1), 2);
});

test('flattening unites all nonzero labels without changing pixels, and outlines the union', () => {
  const data = Uint32Array.from({ length: 81 }, (_, i) => i % 9 < 4 ? 1 : 4294967295); data[0] = 0;
  const original = data.slice(), raster = { data }, source = { kind: 'labels', mapping: [{ value: 1, color: 'transparent' }] };
  const settings = { ...defaultLayer('a', 'labels'), raster, flatten: true, flatColour: '#ff0000', opacity: .5 };
  const pixel = (i, changes = {}) => { const out = new Uint8ClampedArray(4); blendLayer(out, 0, prepareLayer({ ...settings, ...changes }, source), i, 9, 9, true); return [...out]; };
  assert.deepEqual(pixel(0), [0, 0, 0, 0]);
  assert.deepEqual(pixel(39), [255, 0, 0, 128]); // Includes a dictionary-hidden ID.
  assert.deepEqual(pixel(40), [255, 0, 0, 128]); // Includes an unmapped uint32 ID.
  assert.equal(labelEdgeDistance(raster, 9, 9)[40], 1);
  assert.equal(labelEdgeDistance(raster, 9, 9, true)[40], 5);
  assert.equal(labelEdgeDistance(raster, 9, 9)[40], 1); // Separate normal/flat caches.
  for (const outlineWidth of [1, 2]) assert.deepEqual(pixel(40, { style: 'outline', outlineWidth }), [0, 0, 0, 0]);
  assert.deepEqual(pixel(13, { style: 'outline', outlineWidth: 2 }), [255, 0, 0, 128]);
  assert.deepEqual(data, original);
  const hole = { data: data.slice() }; hole.data[40] = 0;
  assert.equal(labelEdgeDistance(hole, 9, 9, true)[39], 1); // Keep holes in the union.
});

test('per-element legend choices affect viewer and figure legends without hiding tissue content', () => {
  const view = { name: 'Test', legendVisible: true, imagesVisible: true, channels: [{ name: 'Hidden marker', colour: '#ff0000', active: true, enabled: true, raster: {}, showInLegend: false }, { name: 'DNA', colour: '#0000ff', active: true, enabled: true, raster: {} }],
    layers: [{ ...defaultLayer('h', 'rgb'), raster: {}, showInLegend: false }, { ...defaultLayer('r', 'labels'), raster: {}, flatten: true, flatLabel: 'Tissue union', flatColour: '#00ff00' }],
    overlayVisible: true, overlay: { style: 'fill', colourBy: 'obs:cell_type', showInLegend: false }, region: {}, colours: { entries: [['T cells', [255, 0, 0]]] } };
  const sources = [{ id: 'h', kind: 'rgb', name: 'Histology' }, { id: 'r', kind: 'labels', name: 'Regions', mapping: [{ value: 1, label: 'Old category', color: '#ffffff' }] }];
  const groups = legendGroups(view, sources);
  assert.deepEqual(groups.map(g => g.title), ['Image channels', 'Regions']);
  assert.deepEqual(groups[0].rows.map(r => r.name), ['DNA']);
  assert.deepEqual(groups[1].rows, [{ name: 'Tissue union', colour: '#00ff00' }]);
  const svg = paneFurniture(view, groups, { ...defaultFigure(), mode: 'figure', showScaleBars: false }, 1024, 1024, 1);
  assert.match(svg, /Tissue union/); assert.doesNotMatch(svg, /Hidden marker|Histology|T cells|Old category/);
  assert.equal(view.channels[0].enabled, true); assert.equal(view.layers[0].visible, true); assert.equal(view.overlayVisible, true);
  view.layers[1].showInLegend = false; view.channels[1].showInLegend = false; assert.deepEqual(legendGroups(view, sources), []);
});

test('legend changes preserve visit history while flattening changes the tissue recipe', async () => {
  const view = { name: 'Test', imagesVisible: true, channels: [{ active: true, enabled: true, name: 'DNA' }], overlay: { visible: true, population: '' }, layers: [defaultLayer('a', 'labels')] };
  const before = await recipeFingerprint(view, null);
  view.channels[0].showInLegend = false; view.overlay.showInLegend = false; view.layers[0].showInLegend = false;
  view.layers[0].flatLabel = 'Renamed'; view.layers[0].flatColour = '#0000ff';
  assert.equal(await recipeFingerprint(view, null), before);
  view.layers[0].flatten = true; assert.notEqual(await recipeFingerprint(view, null), before);
  for (const changes of [{ flatten: 'yes' }, { showInLegend: 0 }, { flatColour: 'red' }, { flatLabel: '' }]) assert.throws(() => validateLayerSettings([{ ...defaultLayer('a', 'labels'), ...changes }]));
});
