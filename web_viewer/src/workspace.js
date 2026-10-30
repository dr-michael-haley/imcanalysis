import { parseNormalization } from './normalization.js';
import { validateRemaps } from './remaps.js';
import { validateReview } from './review-panel.js';
import { validatePlots } from './plots.js';
import { validateColourLimits } from './colour-limits.js';
import { COLOURMAPS } from './colourmaps.js';
import { validateFlags } from './flags.js';
import { validateROIFilters } from './review-data.js';
import { validateRepresentative } from './representative-panel.js';
import { LAYER_TYPES, MAX_LAYERS, IMAGE_FILE, layerFile, validateLayerSettings, parseLabelMapping, labelDictionary } from './aligned-layers.js';
import { assetPath } from './workspace-paths.js';
import { validateNavigation, validateEmbedding } from './navigation.js';
import { validateFigure, validateScaleBar } from './figure.js';
import { validatePopulations } from './populations.js';

export const WORKSPACE_FORMAT = 'sbt-tissue-explorer-workspace';
export const WORKSPACE_VERSION = 13;
export const MAX_WORKSPACE_BYTES = 10 * 1024 * 1024;
const NEW_PANELS = ['section-roi-filter', 'section-roi-sort', 'section-representative', 'section-aligned-setup', 'section-aligned-view'];
export const PANEL_IDS = ['section-data', 'section-roi', 'section-channels', 'section-normalization', 'section-overlay', 'identity-settings', 'normalization-settings', 'review-panels', ...NEW_PANELS];

// Folder roots may move or be renamed. Match their relative TIFF paths and sizes,
// without reading image contents or serialising every file into the workspace.
export const relativeFilePath = file => file.webkitRelativePath?.split('/').slice(1).join('/') || file.name;
const savedPath = value => value?.path === undefined ? {} : { path: assetPath(value.path) };
export const fileReference = file => file ? { name: file.name, size: file.size, ...(file.webkitRelativePath ? { path: relativeFilePath(file) } : {}) } : null;
const folderCache = new WeakMap();
export function folderReference(files) {
  if (!files.length) return Promise.resolve(null);
  if (!folderCache.has(files)) folderCache.set(files, (async () => {
    const entries = files.map(file => [relativeFilePath(file), file.size]).sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] - b[1]);
    const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(entries)));
    return { name: files[0].webkitRelativePath?.split('/')[0] || 'TIFF files', count: files.length, signature: Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join(''), ...(files.workspacePath ? { path: files.workspacePath } : {}) };
  })());
  return folderCache.get(files);
}
export function sameFile(expected, actual) { return Boolean(expected && actual && expected.name === actual.name && expected.size === actual.size); }
export function sameFolder(expected, actual) { return Boolean(expected && actual && expected.count === actual.count && expected.signature === actual.signature); }

function invalid(field) { throw new Error(`Invalid workspace: ${field}.`); }
function record(value, field) { if (!value || typeof value !== 'object' || Array.isArray(value)) invalid(field); return value; }
function string(value, field, nonempty = false) { if (typeof value !== 'string' || value.length > 16384 || (nonempty && !value)) invalid(field); return value; }
function boolean(value, field) { if (typeof value !== 'boolean') invalid(field); return value; }
function number(value, field, min = -Infinity, max = Infinity) { if (!Number.isFinite(value) || value < min || value > max) invalid(field); return value; }
function integer(value, field, min, max = Number.MAX_SAFE_INTEGER) { number(value, field, min, max); if (!Number.isSafeInteger(value)) invalid(field); return value; }
function category(value, field) {
  string(value, field);
  if (value) {
    let parsed; try { parsed = JSON.parse(value); } catch { invalid(field); }
    if (parsed !== null && !['string', 'number', 'boolean'].includes(typeof parsed)) invalid(field);
  }
  return value;
}
function folder(value, field) {
  if (value === null) return null;
  record(value, field);
  if (!/^[a-f0-9]{64}$/.test(value.signature)) invalid(`${field} signature`);
  return { name: string(value.name, field, true), count: integer(value.count, field, 1), signature: value.signature, ...savedPath(value) };
}

export function parseWorkspace(text) {
  if (new TextEncoder().encode(text).length > MAX_WORKSPACE_BYTES) throw new Error('Choose a workspace smaller than 10 MB.');
  let input;
  try { input = JSON.parse(text.replace(/^\uFEFF/, '')); } catch { throw new Error('Invalid workspace JSON.'); }
  record(input, 'document');
  if (input.format !== WORKSPACE_FORMAT) throw new Error('This is not an SBT IMC Explorer workspace.');
  if (![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, WORKSPACE_VERSION].includes(input.version)) throw new Error(`Unsupported workspace version: ${String(input.version)}. This viewer supports versions 1–${WORKSPACE_VERSION}.`);
  const sources = record(input.sources, 'sources'), anndata = record(sources.anndata, 'AnnData reference');
  const layerSources = sources.layers ?? [];
  if (!Array.isArray(layerSources) || layerSources.length > MAX_LAYERS) invalid('aligned layer sources');
  const seenLayers = new Set();
  const layers = layerSources.map(item => {
    record(item, 'aligned layer source');
    const id = string(item.id, 'layer ID', true), name = string(item.name, 'layer name', true);
    if (id.length > 100 || name.length > 80 || seenLayers.has(id) || !Object.hasOwn(LAYER_TYPES, item.kind) || !item.folder) invalid('aligned layer source');
    let dictionary;
    if (item.dictionary) {
      if (item.kind !== 'labels') invalid('only categorical layers use label dictionaries');
      const raw = record(item.dictionary, 'label dictionary'), file = record(raw.file, 'dictionary file'), columns = record(raw.columns, 'dictionary columns');
      dictionary = { file: { name: string(file.name, 'dictionary filename', true), size: integer(file.size, 'dictionary size', 1, 2 * 1024 * 1024), ...savedPath(file) }, columns: { value: string(columns.value, 'dictionary ID column', true), label: string(columns.label, 'dictionary name column', true), color: string(columns.color, 'dictionary colour column') } };
      if (dictionary.columns.value === dictionary.columns.label) invalid('dictionary ID and name columns must differ');
    }
    seenLayers.add(id); return { id, name, kind: item.kind, folder: folder(item.folder, 'aligned layer folder'), ...(dictionary ? { dictionary } : {}) };
  });
  const view = record(input.view, 'view'), mapping = record(view.mapping, 'cell identity columns');
  const overlay = record(view.overlay, 'overlay'), camera = record(view.camera, 'camera');
  const normalization = record(view.normalization, 'normalisation');
  const dictionary = normalization.dictionary === null ? null : Object.fromEntries(parseNormalization(JSON.stringify(normalization.dictionary), 'workspace.json'));
  if (!Array.isArray(view.channels) || view.channels.length !== 6) invalid('six channel slots are required');
  const channels = view.channels.map((raw, i) => {
    const channel = record(raw, `channel ${i + 1}`);
    if (!/^#[0-9a-f]{6}$/i.test(channel.colour)) invalid(`channel ${i + 1} colour`);
    const low = number(channel.low, 'contrast low'), high = number(channel.high, 'contrast high');
    if (low >= high) invalid('contrast high must exceed low');
    if (dictionary && channel.active && channel.path && !channel.auto && (low < 0 || high > 1)) invalid('normalised contrast must be within 0–1');
    return {
      showInLegend: boolean(channel.showInLegend ?? true, 'channel legend visibility'), active: boolean(channel.active, 'channel active'), enabled: boolean(channel.enabled, 'channel enabled'),
      name: string(channel.name, 'channel name'), path: string(channel.path, 'channel path'),
      ...(channel.sourceId ? { sourceId: string(channel.sourceId, 'channel source ID', true) } : {}),
      plane: integer(channel.plane, 'channel plane', 0), low, high, colour: channel.colour,
      auto: boolean(channel.auto, 'automatic contrast'),
    };
  });
  if (!['outline', 'fill', 'off'].includes(overlay.style)) invalid('overlay style');
  if (overlay.colourmap !== undefined && !Object.hasOwn(COLOURMAPS, overlay.colourmap)) invalid('marker colourmap');
  const panels = record(view.panels, 'panels');
  const result = {
    format: WORKSPACE_FORMAT, version: WORKSPACE_VERSION,
    sources: { anndata: { name: string(anndata.name, 'AnnData name', true), size: integer(anndata.size, 'AnnData size', 1), ...savedPath(anndata) }, masks: folder(sources.masks, 'Masks'), images: folder(sources.images, 'Images'), layers },
    view: {
      mapping: { roi: string(mapping.roi, 'ROI column', true), id: string(mapping.id, 'object ID column', true) },
      roi: string(view.roi, 'ROI', true), channels, layers: validateLayerSettings(view.layers), imagesVisible: boolean(view.imagesVisible, 'image visibility'),
      legendVisible: view.legendVisible === undefined ? true : boolean(view.legendVisible, 'legend visibility'),
      legendScale: view.legendScale === undefined ? 1 : number(view.legendScale, 'legend size', 0.7, 2),
      scaleBar: validateScaleBar(view.scaleBar), figure: validateFigure(view.figure),
      overlay: { ...validateColourLimits(overlay), showInLegend: boolean(overlay.showInLegend ?? true, 'overlay legend visibility'), visible: boolean(overlay.visible, 'overlay visibility'), colourBy: string(overlay.colourBy, 'colour by'), source: string(overlay.source, 'expression source'), populations: validatePopulations(overlay), population: validatePopulations(overlay).length === 1 ? validatePopulations(overlay)[0] : '', style: overlay.style, opacity: number(overlay.opacity, 'opacity', 0, 1), colourmap: overlay.colourmap ?? 'legacy', reverse: overlay.reverse === undefined ? false : boolean(overlay.reverse, 'reverse colourmap') },
      embedding: string(view.embedding, 'embedding'), selected: view.selected === null ? null : integer(view.selected, 'selected cell', 1, 0xffffffff),
      normalization: { dictionary, name: string(normalization.name, 'dictionary name'), quantile: number(normalization.quantile, 'quantile', Number.MIN_VALUE, 1), minimum: number(normalization.minimum, 'minimum counts', 0) },
      camera: { fitted: boolean(camera.fitted, 'fitted view'), scale: number(camera.scale, 'zoom', Number.MIN_VALUE), centreX: number(camera.centreX, 'view centre'), centreY: number(camera.centreY, 'view centre') },
      panels: Object.fromEntries(PANEL_IDS.map(id => [id, boolean(panels[id] ?? (NEW_PANELS.includes(id) ? true : undefined), `${id} open`)])),
    },
  };
  const panes = input.version === 1 ? [{ name: 'View 1', channels: result.view.channels, imagesVisible: result.view.imagesVisible, overlay: result.view.overlay }] : view.presets;
  if (!Array.isArray(panes) || panes.length < 1 || panes.length > 4) invalid('one to four view presets are required');
  result.view.presets = panes.map(pane => {
    record(pane, 'view preset');
    const parsed = input.version === 1 ? result.view : parseWorkspace(JSON.stringify({ ...input, version: 1, view: { ...view, ...pane, bookmarks: [], savedViews: [] } })).view;
    return { name: string(pane.name, 'view name', true).slice(0, 80), channels: parsed.channels, layers: parsed.layers, legendVisible: parsed.legendVisible, legendScale: parsed.legendScale, scaleBar: parsed.scaleBar, imagesVisible: parsed.imagesVisible, overlay: parsed.overlay };
  });
  result.view.selectedView = input.version === 1 ? 0 : integer(view.selectedView, 'selected view', 0, panes.length - 1);
  result.view.tab = input.version === 1 ? 'setup' : ['setup', 'rois', 'views'].includes(view.tab) ? view.tab : invalid('sidebar tab');
  const order = input.version === 1 ? { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 } : record(view.order, 'ROI ordering');
  if (!['name', 'cells', 'population', 'obs', 'marker'].includes(order.mode) || !['asc', 'desc'].includes(order.direction)) invalid('ROI ordering');
  result.view.order = { mode: order.mode, direction: order.direction, field: string(order.field, 'ordering field'), value: category(order.value, 'ordering value'), source: string(order.source, 'ordering source'), top: integer(order.top, 'top ROIs', 1, 100) };
  const visits = input.version === 1 ? [] : view.visits;
  if (!Array.isArray(visits) || visits.length > 10000) invalid('visit history');
  result.view.visits = visits.map(entry => {
    if (!Array.isArray(entry) || entry.length !== 2 || !Array.isArray(entry[1]) || entry[1].length > 100000) invalid('visit history entry');
    return [string(entry[0], 'view signature'), entry[1].map(roi => string(roi, 'visited ROI'))];
  });
  result.view.remaps = validateRemaps(view.remaps ?? []);
  result.view.activeRemap = view.activeRemap == null ? '' : string(view.activeRemap, 'active remap');
  if (result.view.activeRemap && !result.view.remaps.some(plan => plan.id === result.view.activeRemap)) invalid('active remap is missing');
  result.view.review = validateReview(view.review);
  result.view.plots = validatePlots(view.plots);
  result.view.flags = validateFlags(view.flags ?? []);
  result.view.activeFlag = view.activeFlag == null ? '' : string(view.activeFlag, 'active flag field');
  if (result.view.activeFlag && !result.view.flags.some(field => field.id === result.view.activeFlag)) invalid('active flag field is missing');
  result.view.roiFilters = validateROIFilters(view.roiFilters);
  result.view.representative = validateRepresentative(view.representative);
  Object.assign(result.view, validateNavigation(view.bookmarks, view.savedViews));
  result.view.embeddingOptions = validateEmbedding(view.embeddingOptions);
  for (const ref of [result.view.plots.groupBy, result.view.plots.compareBy]) if (ref.startsWith('remap:') && !result.view.remaps.some(plan => `remap:${plan.id}` === ref)) invalid('plot references a missing remap');
  for (const pane of result.view.presets) {
    const overlays = layers.filter(source => source.kind !== 'imc');
    if (pane.layers.length !== overlays.length || pane.layers.some(layer => !overlays.some(source => source.id === layer.id))) invalid('aligned layer settings must match their sources');
    if (pane.channels.some(channel => channel.sourceId && !layers.some(source => source.id === channel.sourceId && source.kind === 'imc'))) invalid('channel references a missing IMC source');
    if (pane.overlay.colourBy.startsWith('flag:') && !result.view.flags.some(field => `flag:${field.id}` === pane.overlay.colourBy)) invalid('overlay references a missing flag field');
    if (pane.overlay.colourBy.startsWith('remap:') && !result.view.remaps.some(plan => `remap:${plan.id}` === pane.overlay.colourBy)) invalid('overlay references a missing remap');
  }
  return result;
}

// Resolve a single user-selected root, using the existing metadata signatures.
// Ambiguous copies are reported rather than silently selecting a dataset.
export async function resolveWorkspaceRoot(files, sources, progress = () => {}) {
  progress(`Checking ${files.length.toLocaleString()} files for the saved AnnData…`);
  await new Promise(resolve => setTimeout(resolve, 0));
  const matchesFile = (file, ref) => sameFile(ref, fileReference(file)) && (!ref.path || relativeFilePath(file) === ref.path);
  const candidates = files.filter(file => matchesFile(file, sources.anndata));
  if (candidates.length !== 1) throw new Error(candidates.length ? `Multiple copies of ${sources.anndata.name} found. Choose a more specific root folder.` : `Could not find ${sources.anndata.name} with its saved size under this root.`);
  const groups = new Map();
  let scanned = 0;
  for (const file of files.filter(file => IMAGE_FILE.test(file.name) || file.name.toLowerCase() === 'label_mapping.csv')) {
    if (++scanned % 2000 === 0) { progress(`Indexing asset folders · ${scanned.toLocaleString()} files…`); await new Promise(resolve => setTimeout(resolve, 0)); }
    const parts = (file.webkitRelativePath || file.name).split('/');
    for (let i = 1; i < parts.length; i++) {
      const folder = parts.slice(0, i).join('/');
      if (!groups.has(folder)) groups.set(folder, []);
      groups.get(folder).push(file);
    }
  }
  const scope = (items, path) => Object.assign(items.map(file => {
    const scoped = new File([file], file.name, { type: file.type, lastModified: file.lastModified });
    Object.defineProperty(scoped, 'webkitRelativePath', { value: `${path.split('/').at(-1)}/${file.webkitRelativePath.slice(path.length + 1)}` });
    return scoped;
  }), { workspacePath: path.split('/').slice(1).join('/') });
  const find = async (expected, label, accepts = file => /\.tiff?$/i.test(file.name)) => {
    if (!expected) return [];
    const matches = [];
    for (const [path, all] of groups) {
      if (expected.path ? path.split('/').slice(1).join('/') !== expected.path : path.split('/').at(-1) !== expected.name) continue;
      const items = all.filter(accepts); if (items.length !== expected.count) continue;
      progress(`Matching ${label} · ${items.length.toLocaleString()} files in ${path}…`);
      await new Promise(resolve => setTimeout(resolve, 0));
      const scoped = scope(items, path);
      if (sameFolder(expected, await folderReference(scoped))) matches.push({ path, files: scoped });
    }
    // Nested single-child folders can describe identical data; prefer the saved root name.
    const named = matches.filter(item => item.path.split('/').at(-1) === expected.name);
    const choices = named.length ? named : matches;
    if (choices.length !== 1) throw new Error(choices.length ? `Multiple matching ${label} folders found. Choose a more specific root folder.` : `Could not find the saved ${label} folder (${expected.name}, ${expected.count} files) under this root.`);
    return choices[0].files;
  };
  const result = { anndata: candidates[0], masks: await find(sources.masks, 'Masks'), images: await find(sources.images, 'Images'), layers: [] };
  for (const layer of sources.layers ?? []) {
    const items = await find(layer.folder, layer.name, file => layerFile(file, layer.kind));
    const mappings = items.filter(file => file.name.toLowerCase() === 'label_mapping.csv');
    if (mappings.length > 1 || mappings[0]?.size > 2 * 1024 * 1024) throw new Error(`Invalid label mapping in ${layer.name}. Choose one annotation set with a mapping smaller than 2 MB.`);
    progress(`Reading ${layer.name} layer settings…`);
    let mapping = mappings.length ? parseLabelMapping(await mappings[0].text()) : [], dictionary, dictionaryText;
    if (layer.dictionary) {
      const matches = files.filter(file => matchesFile(file, layer.dictionary.file));
      if (matches.length !== 1) throw new Error(`${matches.length ? 'Multiple copies of' : 'Could not find'} ${layer.dictionary.file.name} for ${layer.name}. Select a root containing one matching dictionary CSV.`);
      dictionaryText = await matches[0].text();
      const parsed = labelDictionary(dictionaryText, layer.dictionary.columns); mapping = parsed.mapping;
      dictionary = { file: matches[0], columns: parsed.columns, headers: parsed.headers };
    }
    result.layers.push({ ...layer, files: items, mapping, dictionary, dictionaryText });
  }
  return result;
}
