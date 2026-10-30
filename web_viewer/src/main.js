import './style.css';
import { ReaderClient } from './rpc.js';
import { loadRaster, rasterIdentity } from './raster.js';
import { channelAliases, channelName, discoverFiles, validateMask, contrast, colourScale, extent, stem, TIFF } from './core.js';
import { parseNormalization, findNormalization } from './normalization.js';
import { WORKSPACE_FORMAT, WORKSPACE_VERSION, MAX_WORKSPACE_BYTES, PANEL_IDS, parseWorkspace, fileReference, folderReference, sameFile, sameFolder, relativeFilePath } from './workspace.js';
import { newView, viewSettings, composeView, composeViewAsync, includedInView } from './views.js';
import { createLoadingState, yieldToBrowser } from './loading.js';
import { createROIBrowser } from './roi-browser.js';
import { applyRemap, remapRecipe, validateRemapSource } from './remaps.js';
import { createRemapEditor } from './remap-editor.js';
import { createReviewPanel, defaultReview } from './review-panel.js';
import { defaultPlots } from './plots.js';
import { createPlotPanel } from './plot-panel.js';
import { COLOURMAPS, colourmapGradient } from './colourmaps.js';
import { applyFlags, validateFlags } from './flags.js';
import { createFlagPanel } from './flag-panel.js';
import { createRepresentativePanel, defaultRepresentative } from './representative-panel.js';
import { createROIMetadataPanel } from './roi-metadata-panel.js';
import { createAlignedPanel } from './aligned-panel.js';
import { renderViewLegends } from './view-legend.js';
import { createWorkspacePanel } from './workspace-panel.js';
import { imageCatalogue, channelChoices } from './image-channels.js';
import { CellSelection } from './selection.js';
import { defaultEmbedding, validateSnapshot, validatePane } from './navigation.js';
import { createNavigationPanel } from './navigation-panel.js';
import { createColourLimitControls } from './colour-limit-controls.js';
import { validateColourLimits } from './colour-limits.js';
import { createEmbeddingPanel } from './embedding-panel.js';
import { defaultFigure } from './figure.js';
import { createFigurePanel } from './figure-panel.js';
import { createFigureExport } from './figure-export.js';
import { figureFilename } from './figure-filename.js';
import { bindFolderPicker, bindFilePicker } from './folder-picker.js';
import { populationKeys, populationFilter } from './populations.js';
import { createViewerToolbar } from './viewer-toolbar.js';
import { defaultLayer } from './aligned-layers.js';
import logoUrl from '../../images/Logo.svg';

const $ = id => document.getElementById(id);
let workspacePanel = null, loadingActive = false, navigationPanel = null, embeddingPanel = null, viewerToolbar = null, figurePanel = null, roiMetadataPanel = null;
const loading = createLoadingState(progress => {
  loadingActive = Boolean(progress);
  $('loading-overlay').hidden = !progress || Boolean(workspace.pending);
  $('viewport').classList.toggle('is-loading', Boolean(progress));
  $('pane-grid')?.setAttribute('aria-busy', String(Boolean(progress)));
  if (progress) {
    $('loading-title').textContent = progress.title;
    $('loading-detail').textContent = progress.detail;
    if (workspace.pending) $('workspace-status').textContent = `${progress.title} · ${progress.detail}`;
  }
  if (!progress) workspacePanel?.mark();
});
const state = {
  reader: null, header: null, anndata: null, mapping: null, rois: [], files: new Map(), masks: [], images: [],
  region: null, mask: null, channels: [], cache: new Map(), selected: null,
  imagesVisible: true, overlayVisible: true,
  normalization: null, normalizationQuantile: 0.999, normalizationMinimum: 0.1,
  revision: 0, dataset: 0, ready: false, fitted: true, scale: 1, x: 0, y: 0, width: 0, height: 0,
};
const workspace = { pending: null, attempt: 0, reading: 0, applying: false, saving: false };
state.views = [newView('View 1')]; state.selectedView = 0;
state.layers = [];
state.channelCatalogue = [];
state.remaps = []; state.activeRemap = ''; state.review = defaultReview();
state.plots = defaultPlots();
state.flags = []; state.activeFlag = ''; state.selection = new CellSelection();
Object.defineProperty(state, 'flagSelection', { get: () => state.selection.forROI($('roi').value) });
state.bookmarks = []; state.savedViews = []; state.embeddingOptions = defaultEmbedding();
state.figure = defaultFigure();
state.representative = defaultRepresentative(); state.roiMetadata = []; state.roiFilters = [];
state.visits = new Map(); state.tab = 'setup';
state.order = { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 };
for (const key of ['channels', 'region', 'colours', 'imagesVisible', 'overlayVisible']) Object.defineProperty(state, key, { get: () => activeView()[key], set: value => { activeView()[key] = value; } });
function activeView() { return state.views[state.selectedView]; }
const canvas = $('tissue');
function viewportRect() { return canvas.parentElement.getBoundingClientRect(); }
const channelColours = ['#ff0000', '#00ff00', '#0000ff', '#00ffff', '#ff00ff', '#ffff00'];
const MAX_RASTERS = 37; // 24 channels, one mask and up to 12 shared aligned layers; also byte-budgeted below.
for (const logo of document.querySelectorAll('.sbt-logo')) logo.src = logoUrl;
$('viewer-version').textContent = `v${__VIEWER_VERSION__}`;

function status(message) { $('status').textContent = message; updateWorkspaceActions(); viewerToolbar?.sync(); }
function warning(message = '') { $('warning').textContent = message; $('warning').hidden = !message; }
function fail(error) { status('Could not complete this view.'); warning(error.message ?? String(error)); }
function option(value, text) { const node = document.createElement('option'); node.value = value; node.textContent = text; return node; }
function fillSelect(select, items, value = '') {
  select.replaceChildren(...items.map(item => option(item.value, item.text)));
  if (items.some(item => item.value === value)) select.value = value;
  select.disabled = items.length === 0;
}
function clearSelection() {
  state.selected = null;
  $('cell-details').replaceChildren(); $('cell-badge').textContent = 'No cell selected';
  $('cell-hint').hidden = false;
  $('cell-hint').textContent = 'Click a segmented cell to see its AnnData observations.';
}
function clearView() {
  state.ready = false; state.region = null; state.mask = null; state.width = 0; state.height = 0;
  for (const view of state.views) { view.scene.width = 0; view.scene.height = 0; view.region = null; }
  clearSelection(); draw(); drawEmbedding();
  $('empty-state').hidden = false; $('legend').replaceChildren();
  document.querySelector('.workspace').classList.toggle('welcome-only', !state.header);
  renderViewLegends(state);
}

function makeChannel(index) {
  const box = document.createElement('div'); box.className = 'channel'; box.id = `channel-box-${index}`;
  box.hidden = index >= 3;
  box.innerHTML = `<div class="channel-heading"><input type="checkbox" id="channel-enabled-${index}" aria-label="Show channel ${index + 1}" checked><input type="color" id="channel-colour-${index}" aria-label="Channel ${index + 1} colour" value="${channelColours[index]}"><label for="channel-${index}">Channel ${index + 1}</label><button id="remove-channel-${index}" class="remove-channel quiet" aria-label="Remove channel ${index + 1}" title="Remove channel">×</button></div><select id="channel-${index}" aria-label="Image channel ${index + 1}"><option value="">None</option></select><div class="contrast"><label>Low<input type="number" id="low-${index}" aria-label="Channel ${index + 1} low exact value" step="any" value="0"><input type="range" id="low-range-${index}" aria-label="Channel ${index + 1} low slider" min="0" max="1" step="0.001" value="0"></label><label>High<input type="number" id="high-${index}" aria-label="Channel ${index + 1} high exact value" step="any" value="1"><input type="range" id="high-range-${index}" aria-label="Channel ${index + 1} high slider" min="0" max="1" step="0.001" value="1"></label><button id="auto-${index}" title="Auto contrast" aria-label="Auto contrast channel ${index + 1}">Auto</button></div>`;
  $('channels').append(box);
  const source = document.createElement('p'); source.id = `channel-normalization-${index}`; source.className = 'channel-source'; source.hidden = true;
  box.insertBefore(source, box.querySelector('.contrast'));
  const legendLabel = document.createElement('label'); legendLabel.className = 'check-label element-legend-control';
  legendLabel.innerHTML = `<input type="checkbox" id="channel-legend-${index}" aria-label="Show channel ${index + 1} in legend" checked>Show in legend`;
  box.append(legendLabel);
  $(`channel-legend-${index}`).onchange = event => { state.channels[index].showInLegend = event.target.checked; updateViewLegends(); };
  state.channels.push({ active: index < 3, enabled: true, showInLegend: true, file: null, raster: null, name: '', plane: 0, low: 0, high: 1, colour: channelColours[index], auto: true });
  $(`channel-enabled-${index}`).addEventListener('change', event => {
    state.channels[index].enabled = event.target.checked;
    box.classList.toggle('channel-muted', !event.target.checked); refresh();
  });
  $(`remove-channel-${index}`).addEventListener('click', () => {
    Object.assign(state.channels[index], { active: false, file: null, raster: null, name: '', sourceId: '' });
    box.hidden = true; updateChannelActions(); refresh(); $('add-channel').focus();
  });
  $(`channel-${index}`).addEventListener('change', () => {
    const entry = channelChoices(state.files, state.channelCatalogue, $('roi').value).find(item => item.key === $(`channel-${index}`).value);
    Object.assign(state.channels[index], { file: entry?.file ?? null, name: entry?.name ?? '', sourceId: entry?.sourceId ?? '', plane: 0, auto: true });
    refresh();
  });
  const setContrast = event => {
    const low = Number($(`low-${index}`).value), high = Number($(`high-${index}`).value);
    if (!$(`low-${index}`).value.trim() || !$(`high-${index}`).value.trim() || !Number.isFinite(low) || !Number.isFinite(high) || high <= low) { if (event.type === 'change') warning('Contrast high must be greater than low.'); return; }
    if (state.normalization && (low < 0 || high > 1)) { if (event.type === 'change') warning('Normalised display contrast must satisfy 0 ≤ low < high ≤ 1.'); return; }
    Object.assign(state.channels[index], { low, high, auto: false }); warning(); syncContrast(index); compose();
  };
  for (const bound of ['low', 'high']) for (const event of ['input', 'change']) $(`${bound}-${index}`).addEventListener(event, setContrast);
  for (const bound of ['low', 'high']) $(`${bound}-range-${index}`).addEventListener('input', event => {
    const channel = state.channels[index], slider = event.target, gap = Number(slider.step);
    const value = Number(slider.value); channel[bound] = bound === 'low' ? Math.min(value, channel.high - gap) : Math.max(value, channel.low + gap);
    channel.auto = false; $(`${bound}-${index}`).value = String(channel[bound]); slider.value = String(channel[bound]);
    cancelAnimationFrame(channel.contrastFrame); channel.contrastFrame = requestAnimationFrame(compose);
  });
  $(`channel-colour-${index}`).addEventListener('input', event => { state.channels[index].colour = event.target.value; compose(); });
  $(`auto-${index}`).addEventListener('click', () => { state.channels[index].auto = true; refresh(); });
}
for (let i = 0; i < channelColours.length; i++) makeChannel(i);
function syncContrast(index) {
  const channel = state.channels[index], bounds = channel.raster?.displayExtent ?? [0, Math.max(1, channel.high)];
  const low = state.normalization ? 0 : Math.min(bounds[0], channel.low), high = state.normalization ? 1 : Math.max(bounds[1], channel.high);
  for (const bound of ['low', 'high']) { const slider = $(`${bound}-range-${index}`); slider.min = String(low); slider.max = String(high); slider.step = String((high - low) / 1000); slider.value = String(channel[bound]); }
}
function updateChannelActions() {
  const count = state.channels.filter(channel => channel.active).length;
  $('channel-count').textContent = `${count} / ${channelColours.length} channels`;
  $('add-channel').disabled = count === channelColours.length;
}
updateChannelActions();
$('add-channel').addEventListener('click', () => {
  const index = state.channels.findIndex(channel => !channel.active);
  if (index < 0) return;
  const images = state.files.get($('roi').value)?.images ?? [];
  const entry = images.find(item => !state.channels.some(channel => channel.active && channel.file === item.file));
  Object.assign(state.channels[index], { active: true, enabled: true, showInLegend: true, file: entry?.file ?? null, raster: null, name: entry?.name ?? '', plane: 0, low: 0, high: 1, colour: channelColours[index], auto: true });
  const box = $(`channel-box-${index}`); box.hidden = false; box.classList.remove('channel-muted');
  $(`channel-legend-${index}`).checked = true; $(`channel-enabled-${index}`).checked = true; $(`channel-colour-${index}`).value = channelColours[index];
  $(`low-${index}`).value = '0'; $(`high-${index}`).value = '1'; syncContrast(index);
  fillSelect($(`channel-${index}`), [{ value: '', text: 'None' }, ...images.map(item => ({ value: item.key, text: item.name }))], entry?.key ?? '');
  updateChannelActions(); refresh(); $(`channel-${index}`).focus();
});

setupSidebar();
const controlSections = [...document.querySelectorAll('.control-section')];
const updateSectionAction = () => { $('collapse-sections').textContent = controlSections.some(section => section.open) ? 'Collapse all' : 'Expand all'; };
for (const section of controlSections) section.addEventListener('toggle', updateSectionAction);
$('collapse-sections').addEventListener('click', () => {
  const open = !controlSections.some(section => section.open);
  for (const section of controlSections) section.open = open;
  updateSectionAction();
});

for (const [id, key, label] of [['toggle-images', 'imagesVisible', 'all image channels'], ['toggle-overlay', 'overlayVisible', 'cell overlay']]) {
  const button = $(id);
  button.addEventListener('click', event => {
    // A visibility click must not also expand/collapse the containing section.
    event.preventDefault(); event.stopPropagation();
    state[key] = !state[key];
    button.setAttribute('aria-pressed', String(state[key]));
    button.title = `${state[key] ? 'Hide' : 'Show'} ${label}`;
    compose();
  });
  button.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') event.stopPropagation(); });
}

let normalizationRequest = 0;
function normalizationError(message = '') {
  $('normalization-error').textContent = message; $('normalization-error').hidden = !message;
}
function updateNormalizationStatus() {
  $('clear-normalization').disabled = !state.normalization;
  if (!state.normalization) { $('normalization-status').textContent = 'Optional Nimbus / NapariSBT dictionary. Image display only.'; return; }
  const visible = state.channels.filter(channel => channel.active && channel.enabled && channel.raster);
  const matched = visible.filter(channel => channel.raster.normalization?.marker).length;
  $('normalization-status').textContent = `${state.normalization.size} marker limits loaded. ${visible.length ? `${matched}/${visible.length} visible channels matched; ${visible.length - matched} use fallback. ` : ''}Display range 0–1.`;
}
function refreshNormalization(resetContrast = true) {
  cancelPendingRasters(); state.cache.clear();
  for (const view of state.views) for (const channel of view.channels) { channel.raster = null; if (resetContrast) channel.auto = true; }
  updateNormalizationStatus(); refresh();
}
$('normalization-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  const request = ++normalizationRequest;
  try {
    if (file.size > 5 * 1024 * 1024) throw new Error('Choose a normalisation dictionary smaller than 5 MB.');
    const mapping = parseNormalization(await file.text(), file.name);
    if (request !== normalizationRequest) return;
    state.normalization = mapping; $('normalization-name').textContent = file.name;
    normalizationError(); refreshNormalization();
  } catch (error) { if (request === normalizationRequest) normalizationError(`${error.message} The previous display settings are unchanged.`); }
  finally { if (request === normalizationRequest) event.target.value = ''; }
});
$('clear-normalization').addEventListener('click', () => {
  normalizationRequest++; state.normalization = null; $('normalization-file').value = '';
  $('normalization-name').textContent = 'Choose CSV / JSON'; normalizationError(); refreshNormalization();
});
$('apply-normalization-settings').addEventListener('click', () => {
  const quantile = Number($('normalization-quantile').value), minimum = Number($('normalization-minimum').value);
  if (!(quantile > 0 && quantile <= 1) || !Number.isFinite(minimum) || minimum < 0 || $('normalization-minimum').value.trim() === '') {
    normalizationError('Quantile must be above 0 and at most 1; minimum pixel counts must be finite and non-negative.'); return;
  }
  state.normalizationQuantile = quantile; state.normalizationMinimum = minimum; normalizationError();
  if (state.normalization) refreshNormalization(false);
});
function imageNormalization(channel) {
  if (!state.normalization || !channel.file) return null;
  const bounds = findNormalization(state.normalization, [channel.name.split(' · ')[0], stem(channel.file)], state.header ? channelAliases(state.header) : new Map());
  return { bounds, quantile: state.normalizationQuantile, minimum: state.normalizationMinimum };
}

function indexFiles() {
  if (!state.header || !state.rois.length) return;
  state.files = discoverFiles(state.masks, state.images, state.rois.map(roi => roi.name), channelAliases(state.header));
  alignedPanel.index();
  for (const source of state.layers.filter(source => source.kind === 'imc')) for (const [roi, images] of source.rois) {
    state.files.get(roi)?.images.push(...images.map(image => ({ ...image, sourceId: source.id, key: `${source.id}:${image.key}` })));
  }
  state.channelCatalogue = imageCatalogue(state.files);
}
function updateChannels(first = false) {
  const images = state.files.get($('roi').value)?.images ?? [];
  for (const view of state.views) view.channels.forEach((channel, i) => {
    if (!channel.active) return;
    const stableName = channel.name.includes(' · ') ? channel.name : channelName({ name: channel.name }, $('roi').value, channelAliases(state.header));
    const candidates = images.filter(item => (item.sourceId ?? '') === (channel.sourceId ?? ''));
    const entry = candidates.find(item => item.name === channel.name) ?? candidates.find(item => item.name === stableName) ?? (first ? images[i] : null);
    // Keep the recipe, including plane and manual contrast, across ROIs. A missing
    // marker is not replaced by a different channel and can reconnect in the next ROI.
    Object.assign(channel, { file: entry?.file ?? null, raster: null });
    if (entry) channel.sourceId = entry.sourceId ?? '';
    if (entry && entry.name !== channel.name) Object.assign(channel, { name: entry.name, ...(first ? { plane: 0, auto: true } : {}) });
    else if (!entry && first) channel.name = '';
  });
  syncViewControls();
}
function rasterKey(file, plane, mask = false, normalization = null, kind = '') {
  return `${rasterIdentity(file)}|${file.webkitRelativePath || file.name}|${file.size}|${file.lastModified}|${plane}|${mask}|${JSON.stringify(normalization)}|${kind}`;
}
function raster(file, plane, mask = false, normalization = null, kind = '') {
  const key = rasterKey(file, plane, mask, normalization, kind);
  if (!state.cache.has(key)) {
    const controller = new AbortController();
    const entry = { controller, settled: false, promise: loadRaster(file, plane, mask, controller.signal, normalization, kind) };
    state.cache.set(key, entry);
    entry.promise.then(() => { entry.settled = true; }, () => { if (state.cache.get(key) === entry) state.cache.delete(key); });
  }
  const result = state.cache.get(key);
  state.cache.delete(key); state.cache.set(key, result);
  while (state.cache.size > MAX_RASTERS) state.cache.delete(state.cache.keys().next().value);
  return result.promise;
}
function cancelPendingRasters() {
  for (const [key, entry] of state.cache) if (!entry.settled) { entry.controller.abort(); state.cache.delete(key); }
}

async function configure() {
  if (!state.reader || !state.header) return;
  if (state.flags.some(field => field.assignments.length) && state.mapping && (state.mapping.roi !== $('roi-column').value || state.mapping.id !== $('id-column').value)) { warning('Save or remove the existing flag fields before changing cell identity columns. Flags refer to the current ROI and object ID mapping.'); return; }
  const dataset = state.dataset;
  const revision = ++state.revision, mapping = { roi: $('roi-column').value, id: $('id-column').value };
  clearView(); warning(); status('Checking cell identities…');
  const activity = loading.begin('Preparing AnnData', 'Checking cell identities and indexing ROIs…');
  try {
    const rois = await state.reader.callProgress('configure', p => activity.update(p.detail), mapping.roi, mapping.id);
    if (dataset !== state.dataset || revision !== state.revision) return;
    state.rois = rois; state.mapping = mapping;
    flagPanel.clearSelection();
    roiMetadataPanel.reset();
    plotPanel.sync();
    fillSelect($('roi'), rois.map(roi => ({ value: roi.name, text: roi.name })));
    $('previous-roi').disabled = false; $('next-roi').disabled = false;
    state.order = { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: 10 };
    state.ranking = rois; syncOrderControls();
    indexFiles(); updateChannels(true); await refresh(true);
  } catch (error) { if (dataset === state.dataset && revision === state.revision) fail(error); }
  finally { activity.finish(); }
}

bindFilePicker($('anndata-file'), '.h5ad', 'AnnData', async file => {
  lockViewer(true);
  try { await workspacePanel.beforeDataset(); await openAnnData(file); } catch (error) { workspaceError(error.message); }
  finally { lockViewer(Boolean(workspace.pending)); }
}, error => workspaceError(error.message));
async function openAnnData(file) {
  if (!workspace.pending) { state.figure = defaultFigure(); figurePanel?.resetReturnCamera(); }
  const dataset = ++state.dataset; workspace.attempt++; state.anndata = null; state.visits.clear();
  state.remaps = []; state.activeRemap = '';
  state.flags = []; state.activeFlag = ''; state.representative = defaultRepresentative();
  state.bookmarks = []; state.savedViews = []; state.embeddingOptions = defaultEmbedding();
  navigationPanel?.sync(); embeddingPanel?.reset();
  state.revision++; state.reader?.stop(); state.reader = new ReaderClient();
  state.header = null; state.mapping = null; state.rois = []; cancelPendingRasters(); state.cache.clear(); clearView(); warning();
  remapEditor.reset();
  plotPanel.reset();
  flagPanel.reset(); representativePanel.reset(); roiMetadataPanel.reset();
  for (const id of ['roi', 'roi-column', 'id-column', 'colour-by', 'embedding']) fillSelect($(id), []);
  $('apply-mapping').disabled = true; $('previous-roi').disabled = true; $('next-roi').disabled = true;
  $('anndata-name').textContent = file.name; status('Reading AnnData metadata…');
  const activity = loading.begin('Opening AnnData', `Reading ${file.name}…`);
  try {
    const header = await state.reader.call('open', file);
    if (dataset !== state.dataset) return;
    state.header = header; state.anndata = file; embeddingPanel.sync();
    document.querySelector('.workspace').classList.remove('welcome-only');
    remapEditor.sync();
    plotPanel.sync();
    flagPanel.sync(); representativePanel.sync();
    const columns = header.obsKeys.map(key => ({ value: key, text: key }));
    const choose = names => header.obsKeys.find(key => names.includes(key.toLowerCase())) ?? '';
    fillSelect($('roi-column'), [{ value: '', text: 'Choose ROI column' }, ...columns], choose(['roi', 'region', 'sample']));
    fillSelect($('id-column'), [{ value: '', text: 'Choose object ID column' }, ...columns], choose(['objectnumber', 'object_id', 'cell_id']));
    $('apply-mapping').disabled = false;
    const defaultObs = header.obsKeys.find(key => /celltype|cell_type|population|leiden/i.test(key));
    fillSelect($('colour-by'), [{ value: '', text: 'Cell outlines' }, ...header.obsKeys.map(key => ({ value: `obs:${key}`, text: `Observation · ${key}` })), ...header.markers.map(key => ({ value: `var:${key}`, text: `Marker · ${key}` }))], defaultObs ? `obs:${defaultObs}` : '');
    fillSelect($('expression-source'), header.sources.map(key => ({ value: key, text: key })), 'X');
    fillSelect($('embedding'), [{ value: '', text: 'No embedding' }, ...header.embeddings.map(key => ({ value: key, text: key }))], header.embeddings.find(key => /umap/i.test(key)) ?? '');
    for (const view of state.views) {
      if (![...$('colour-by').options].some(item => item.value === view.overlay.colourBy)) view.overlay.colourBy = $('colour-by').value;
      if (!header.sources.includes(view.overlay.source)) view.overlay.source = $('expression-source').value;
    }
    storeViewControls(); syncOrderControls();
    if (workspace.pending) await restoreWorkspace();
    else if ($('roi-column').value && $('id-column').value) await configure();
    else { $('section-data').open = true; document.querySelector('.mapping').open = true; status('Choose the ROI and mask object ID columns, then apply the mapping.'); }
  } catch (error) { if (dataset === state.dataset) { if (workspace.pending) { workspaceError(error.message); $('workspace-status').textContent = 'AnnData could not be opened. Check the error below and retry.'; } else fail(error); } }
  finally { activity.finish(); }
}

for (const [input, key, label] of [['mask-files', 'masks', 'masks-name'], ['image-files', 'images', 'images-name']]) {
  bindFolderPicker($(input), file => TIFF.test(file.name), files => {
    state[key] = files.filter(file => TIFF.test(file.name));
    if (files.workspaceHandle) state[key].workspaceHandle = files.workspaceHandle;
    $(label).textContent = `${state[key].length.toLocaleString()} TIFF files`;
    cancelPendingRasters(); state.cache.clear();
    if (workspace.pending) { restoreWorkspace(); return; }
    try { indexFiles(); updateChannels(key === 'images'); refresh(true); } catch (error) { state.revision++; clearView(); fail(error); }
  }, fail);
}
$('apply-mapping').addEventListener('click', configure);
$('roi').addEventListener('change', () => { storeViewControls(); roiMetadataPanel.summary(); cancelPendingRasters(); state.cache.clear(); updateChannels(); refresh(state.fitted); });
for (const [id, delta] of [['previous-roi', -1], ['next-roi', 1]]) $(id).addEventListener('click', () => {
  const select = $('roi'), index = Math.max(0, Math.min(select.options.length - 1, select.selectedIndex + delta));
  if (!select.options.length || index === select.selectedIndex) return;
  select.selectedIndex = index;
  select.dispatchEvent(new Event('change'));
});
for (const id of ['colour-by', 'expression-source', 'embedding']) $(id).addEventListener('change', () => {
  if (id === 'colour-by') $('population').value = '';
  refresh();
});
for (const id of ['overlay-style', 'opacity']) $(id).addEventListener('input', compose);
$('population').addEventListener('change', () => { compose(); syncViewControls(); drawEmbedding(); representativePanel.changed(); });
$('population-all').onclick = () => { for (const option of $('population').options) option.selected = false; $('population').dispatchEvent(new Event('change')); };
fillSelect($('marker-colourmap'), Object.entries(COLOURMAPS).map(([value, [text]]) => ({ value, text })), 'viridis');
const overlayLimits = createColourLimitControls($('overlay-colour-limits'), 'overlay', limits => {
  storeViewControls(); const view = activeView(); Object.assign(view.overlay, limits);
  if (view.region) view.colours = colourScale(view.region.values, view.region.categories, view.region.palette, view.overlay.colourmap, view.overlay.reverse, view.overlay);
  updateLegend(); compose(); drawEmbedding(); workspacePanel?.mark();
});
for (const id of ['marker-colourmap', 'marker-reverse']) $(id).addEventListener('change', () => { storeViewControls(); const view = activeView(); if (view.region) view.colours = colourScale(view.region.values, view.region.categories, view.region.palette, view.overlay.colourmap, view.overlay.reverse, view.overlay); updateLegend(); compose(); drawEmbedding(); });
$('reset').addEventListener('click', () => window.location.reload());
$('welcome-setup').onclick = () => {
  showTab('setup'); $('section-data').open = true;
  $('section-data').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  $('section-data').querySelector('summary').focus({ preventScroll: true });
};
$('welcome-workspace').onclick = () => {
  showTab('setup'); $('workspace-controls').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  $('local-workspace-select').focus({ preventScroll: true });
};

function setupSidebar() {
  for (const [tab, ids] of [['setup', ['workspace-controls', 'section-data', 'section-normalization', 'section-aligned-setup']], ['rois', ['section-roi', 'section-roi-filter', 'section-roi-sort']], ['views', ['view-controls', 'section-channels', 'section-overlay', 'section-representative', 'section-aligned-view']]]) {
    ids.forEach(id => $(`panel-${tab}`).append($(id)));
    let step = 1;
    for (const label of $(`panel-${tab}`).querySelectorAll('.step')) label.textContent = String(step++).padStart(2, '0');
    $(`tab-${tab}`).addEventListener('click', () => showTab(tab));
    $(`tab-${tab}`).addEventListener('keydown', event => {
      const tabs = ['setup', 'rois', 'views'], index = tabs.indexOf(tab);
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault(); const next = event.key === 'Home' ? 'setup' : event.key === 'End' ? 'views' : tabs[(index + (event.key === 'ArrowRight' ? 1 : 2)) % 3];
        showTab(next); $(`tab-${next}`).focus();
      }
    });
  }
  document.querySelector('.sidebar-tabs').after(document.querySelector('.sidebar-toolbar'));
  showTab('setup');
}
function showTab(tab) {
  state.tab = tab;
  for (const name of ['setup', 'rois', 'views']) {
    $(`panel-${name}`).hidden = name !== tab; $(`tab-${name}`).setAttribute('aria-selected', String(name === tab)); $(`tab-${name}`).tabIndex = name === tab ? 0 : -1;
  }
  if (state.header && tab === 'views') { representativePanel.activate(); syncViewControls(); }
}
function updateViewLegends() {
  renderViewLegends(state); figurePanel?.draw();
  document.dispatchEvent(new Event('workspace-changed'));
}
$('overlay-legend').onchange = event => { activeView().overlay.showInLegend = event.target.checked; updateViewLegends(); };
function storeViewControls() {
  const view = activeView(); if (!view) return;
  const populations = [...$('population').selectedOptions].map(option => option.value);
  view.overlay = { ...validateColourLimits(view.overlay), showInLegend: $('overlay-legend').checked, populations, colourBy: $('colour-by').value, source: $('expression-source').value, population: populations.length === 1 ? populations[0] : '', style: $('overlay-style').value, opacity: Number($('opacity').value), colourmap: $('marker-colourmap').value || 'viridis', reverse: $('marker-reverse').checked };
}
function syncViewControls() {
  const view = activeView(), images = channelChoices(state.files, state.channelCatalogue, $('roi').value);
  fillSelect($('view-select'), state.views.map((pane, i) => ({ value: String(i), text: pane.name })), String(state.selectedView));
  $('view-name').value = view.name; $('add-view').disabled = state.views.length === 4; $('remove-view').disabled = state.views.length === 1;
  for (const [id, value] of [['colour-by', view.overlay.colourBy], ['expression-source', view.overlay.source], ['overlay-style', view.overlay.style], ['opacity', view.overlay.opacity]]) $(id).value = String(value);
  $('expression-label').hidden = !view.overlay.colourBy.startsWith('var:');
  $('colourmap-controls').hidden = !view.overlay.colourBy.startsWith('var:') && !(view.overlay.colourBy && view.region && !view.region.categories);
  overlayLimits.sync(view.overlay);
  $('overlay-legend').checked = view.overlay.showInLegend !== false;
  $('marker-colourmap').value = view.overlay.colourmap ?? 'viridis'; $('marker-reverse').checked = view.overlay.reverse ?? false;
  const keys = populationKeys(view.overlay), categories = new Map((view.region?.categories ?? []).map(value => [JSON.stringify(value), value]));
  for (const key of keys) if (!categories.has(key)) categories.set(key, JSON.parse(key));
  $('population').replaceChildren(...[...categories].map(([key, value]) => { const item = option(key, value == null ? view.overlay.colourBy.startsWith('flag:') ? 'Unassigned' : 'Missing' : String(value)); item.selected = keys.includes(key); return item; }));
  $('population').disabled = !view.region?.categories;
  $('population-summary').textContent = keys.length ? `${keys.length} population${keys.length === 1 ? '' : 's'} selected` : 'All populations';
  view.channels.forEach((channel, i) => {
    const box = $(`channel-box-${i}`); box.hidden = !channel.active; box.classList.toggle('channel-muted', !channel.enabled);
    $(`channel-legend-${i}`).checked = channel.showInLegend !== false; $(`channel-enabled-${i}`).checked = channel.enabled; $(`channel-colour-${i}`).value = channel.colour;
    const selected = images.find(item => item.name === channel.name && (item.sourceId ?? '') === (channel.sourceId ?? ''));
    const select = $(`channel-${i}`);
    select.replaceChildren(option('', channel.name && !selected ? `Missing: ${channel.name}` : 'None'));
    for (const source of [{ id: '', name: 'Images' }, ...state.layers.filter(source => source.kind === 'imc')]) {
      const entries = images.filter(item => (item.sourceId ?? '') === source.id); if (!entries.length) continue;
      const group = document.createElement('optgroup'); group.label = source.name;
      for (const item of entries) group.append(option(item.key, `${source.id ? source.name + ' · ' : ''}${item.name}${item.file ? '' : ' (unavailable in this ROI)'}`));
      select.append(group);
    }
    select.disabled = false; select.value = selected?.key ?? '';
    syncContrast(i);
    for (const bound of ['low', 'high']) {
      const input = $(`${bound}-${i}`); input.value = String(channel[bound]);
      if (state.normalization) { input.min = '0'; input.max = '1'; input.step = '0.01'; }
      else { input.removeAttribute('min'); input.removeAttribute('max'); input.step = 'any'; }
    }
    const normalized = channel.raster?.normalization, label = $(`channel-normalization-${i}`); label.hidden = !normalized;
    label.textContent = normalized ? `0–1 · ${normalized.marker ? `Dictionary: ${normalized.marker}` : 'Fallback quantile'} · ${normalized.lower_threshold.toPrecision(4)}–${normalized.vmax.toPrecision(4)}` : '';
    $(`auto-${i}`).textContent = state.normalization ? 'Reset' : 'Auto';
  });
  for (const [id, visible, label] of [['toggle-images', view.imagesVisible, 'all image channels'], ['toggle-overlay', view.overlayVisible, 'cell overlay']]) { $(id).setAttribute('aria-pressed', String(visible)); $(id).title = `${visible ? 'Hide' : 'Show'} ${label}`; }
  updateChannelActions(); updateNormalizationStatus(); syncQuickTools();
  for (const [i, pane] of state.views.entries()) { pane.element?.classList.toggle('selected', i === state.selectedView); pane.label?.setAttribute('aria-pressed', String(i === state.selectedView)); if (pane.label) pane.label.textContent = pane.name; }
  if (view.region) updateLegend(); else $('legend').replaceChildren();
  alignedPanel.sync();
  renderViewLegends(state);
}
function selectView(index) {
  if (index === state.selectedView || !state.views[index]) return;
  storeViewControls(); state.selectedView = index; syncViewControls(); drawEmbedding();
  representativePanel.changed();
  if (state.selected) selectCell(state.selected);
}
function remapFor(colourBy) { return state.remaps.find(plan => `remap:${plan.id}` === colourBy); }
function syncRemapOptions() {
  const selected = activeView().overlay.colourBy;
  fillSelect($('colour-by'), [{ value: '', text: 'Cell outlines' },
    ...(state.header?.obsKeys ?? []).map(key => ({ value: `obs:${key}`, text: `Observation · ${key}` })),
    ...(state.header?.markers ?? []).map(key => ({ value: `var:${key}`, text: `Marker · ${key}` })),
    ...state.remaps.map(plan => ({ value: `remap:${plan.id}`, text: `Remap · ${plan.name}` })),
    ...state.flags.map(field => ({ value: `flag:${field.id}`, text: `Flags · ${field.name}` }))], selected);
}
function remapsChanged(redraw) {
  workspacePanel?.mark();
  plotPanel.changed();
  storeViewControls();
  let affected = false;
  for (const view of state.views) if (view.overlay.colourBy.startsWith('remap:')) {
    affected = true;
    const plan = remapFor(view.overlay.colourBy);
    if (!plan) { view.overlay.colourBy = ''; populationFilter(view.overlay, []); }
    else populationFilter(view.overlay, populationKeys(view.overlay).filter(key => plan.rows.some(row => JSON.stringify(row.label) === key)));
    if (redraw) view.fingerprint = '';
  }
  syncRemapOptions(); syncViewControls();
  if (affected && redraw) refresh();
}
function flagsChanged() {
  workspacePanel?.mark();
  storeViewControls();
  for (const view of state.views) if (view.overlay.colourBy.startsWith('flag:')) {
    const field = state.flags.find(item => `flag:${item.id}` === view.overlay.colourBy);
    if (!field) { view.overlay.colourBy = ''; populationFilter(view.overlay, []); view.remapRecipe = null; if (view.region) { view.region = { ...view.region, values: view.region.ids, categories: null }; view.colours = { colour: () => [100, 216, 188], entries: [], range: null }; } }
    else if (view.region) {
      view.region = applyFlags(view.region, field, $('roi').value); view.remapRecipe = field;
      populationFilter(view.overlay, populationKeys(view.overlay).filter(key => view.region.categories.some(value => JSON.stringify(value) === key)));
      view.colours = colourScale(view.region.values, view.region.categories, view.region.palette);
    }
  }
  syncRemapOptions(); syncViewControls(); compose(); drawEmbedding();
}
function rebuildPanes() {
  const container = $('pane-grid') ?? Object.assign(document.createElement('div'), { id: 'pane-grid' });
  if (!container.parentNode) $('viewport').prepend(container);
  container.replaceChildren(); container.dataset.count = String(state.views.length);
  state.views.forEach((view, i) => {
    const pane = document.createElement('div'); pane.className = 'view-pane'; pane.dataset.view = String(i);
    const target = i === 0 ? canvas : document.createElement('canvas'); target.id = i === 0 ? 'tissue' : `tissue-${i}`; target.className = 'tissue-canvas'; target.setAttribute('aria-label', `${view.name}: tissue and segmentation`);
    const label = document.createElement('button'); label.className = 'pane-label'; label.textContent = view.name; label.title = 'Select this view'; label.addEventListener('click', () => { selectView(i); showTab('views'); });
    const legend = document.createElement('div'); legend.className = 'view-legend'; legend.setAttribute('role', 'region');
    view.canvas = target; view.element = pane; view.label = label; view.legend = legend;
    const furniture = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); furniture.classList.add('pane-furniture'); furniture.setAttribute('aria-hidden', 'true'); view.furniture = furniture;
    pane.append(target, label, legend, furniture); container.append(pane); bindPane(target, i);
  });
  // Establish the frame before refresh/export can read it, even in a background tab.
  syncViewControls(); resizeCanvas();
}
$('view-select').addEventListener('change', event => selectView(Number(event.target.value)));
$('view-name').addEventListener('input', event => {
  activeView().name = event.target.value.trim() || `View ${state.selectedView + 1}`;
  activeView().label.textContent = activeView().name; $('view-select').options[state.selectedView].textContent = activeView().name;
  activeView().canvas.setAttribute('aria-label', `${activeView().name}: tissue and segmentation`); syncQuickTools(); renderViewLegends(state);
});
$('add-view').addEventListener('click', () => {
  if (state.views.length >= 4) return;
  storeViewControls(); state.views.push(newView(`View ${state.views.length + 1}`, viewSettings(activeView()))); state.selectedView = state.views.length - 1;
  rebuildPanes(); updateChannels(); refresh(state.fitted);
});
$('remove-view').addEventListener('click', () => {
  if (state.views.length <= 1) return;
  state.views.splice(state.selectedView, 1); state.selectedView = Math.min(state.selectedView, state.views.length - 1);
  rebuildPanes(); refresh(state.fitted);
});
function syncQuickTools() {
  viewerToolbar?.sync(); figurePanel?.sync();
}
$('quick-legend').addEventListener('click', () => { activeView().legendVisible = !activeView().legendVisible; renderViewLegends(state); syncQuickTools(); });
async function loadWorkspaceAssets(found, payload) {
  if (!workspace.pending) return false;
  workspace.pending = payload;
  const pending = payload;
  workspaceError(); workspace.resolvingRoot = true;
  try {
    state.masks = found.masks; state.images = found.images; state.layers = found.layers;
    $('masks-name').textContent = `${state.masks.length.toLocaleString()} TIFF files`; $('images-name').textContent = `${state.images.length.toLocaleString()} TIFF files`;
    workspace.rootRequired = false; await openAnnData(found.anndata);
    return !workspace.pending;
  } catch (error) { if (pending === workspace.pending) { workspaceError(error.message); $('workspace-status').textContent = 'Assets found, but the saved view could not be opened. Check the message below and retry.'; } }
  finally { workspace.resolvingRoot = false; updateWorkspaceActions(); }
  return false;
}

function workspaceError(message = '') { $('workspace-error').textContent = message; $('workspace-error').hidden = !message; if (message) { showTab('setup'); $('workspace-locations').open = true; } }
function updateWorkspaceActions() {
  $('save-workspace').disabled = !state.ready || !state.anndata || Boolean(workspace.pending) || workspace.saving || workspace.applying;
  $('write-workspace-file').disabled = $('save-workspace').disabled;
  $('workspace-file').disabled = workspace.applying || workspace.resolvingRoot;
  $('workspace-root').disabled = workspace.applying || workspace.resolvingRoot;
  $('choose-workspace-root').disabled = workspace.applying || workspace.resolvingRoot;
  $('cancel-workspace').hidden = !workspace.pending;
  $('cancel-workspace').disabled = workspace.applying;
  workspacePanel?.controls();
}
function lockViewer(locked) {
  if (locked) viewerToolbar?.close();
  for (const node of [...controlSections, $('view-controls'), document.querySelector('.sidebar-tabs'), $('collapse-sections'), $('viewport'), $('viewer-toolbar'), $('review-panels')]) node.inert = locked;
}
function lockWorkspace(locked) {
  workspace.applying = locked;
  lockViewer(locked || Boolean(workspace.pending));
  updateWorkspaceActions();
}
function captureView() {
  storeViewControls();
  const rect = viewportRect();
  return {
    mapping: { ...state.mapping }, roi: $('roi').value,
    layers: viewSettings(activeView()).layers,
    legendVisible: activeView().legendVisible,
    legendScale: activeView().legendScale, scaleBar: { ...activeView().scaleBar }, figure: { ...state.figure },
    presets: state.views.map(view => { const preset = viewSettings(view); return { ...preset, channels: preset.channels.map((channel, i) => ({ ...channel, path: view.channels[i].file ? relativeFilePath(view.channels[i].file) : '' })) }; }),
    selectedView: state.selectedView, tab: state.tab, order: { ...state.order }, visits: [...state.visits].map(([key, rois]) => [key, [...rois]]),
    remaps: structuredClone(state.remaps), activeRemap: state.activeRemap, review: { ...state.review }, plots: structuredClone(plotPanel.capture()),
    flags: structuredClone(state.flags), activeFlag: state.activeFlag, representative: representativePanel.capture(), roiFilters: structuredClone(state.roiFilters),
    channels: state.channels.map(({ active, enabled, showInLegend = true, name, file, plane, low, high, colour, auto, sourceId }) => ({ active, enabled, showInLegend, name, path: file ? relativeFilePath(file) : '', plane, low, high, colour, auto, ...(sourceId ? { sourceId } : {}) })),
    imagesVisible: state.imagesVisible,
    overlay: { ...activeView().overlay, visible: state.overlayVisible },
    embedding: $('embedding').value, selected: state.selected,
    bookmarks: structuredClone(state.bookmarks), savedViews: structuredClone(state.savedViews), embeddingOptions: { ...state.embeddingOptions },
    normalization: { dictionary: state.normalization ? Object.fromEntries(state.normalization) : null, name: state.normalization ? $('normalization-name').textContent : '', quantile: state.normalizationQuantile, minimum: state.normalizationMinimum },
    camera: { fitted: state.fitted, scale: state.scale, centreX: (rect.width / 2 - state.x) / state.scale, centreY: (rect.height / 2 - state.y) / state.scale },
    panels: Object.fromEntries(PANEL_IDS.map(id => [id, $(id).open])),
  };
}
function captureSnapshot() {
  storeViewControls(); const rect = viewportRect();
  return validateSnapshot({ figure: { ...state.figure }, roi: $('roi').value, mapping: { ...state.mapping }, panes: state.views.map(viewSettings), selectedView: state.selectedView, selected: state.selected,
    embedding: $('embedding').value, embeddingOptions: state.embeddingOptions,
    normalization: { dictionary: state.normalization ? Object.fromEntries(state.normalization) : null, name: state.normalization ? $('normalization-name').textContent : '', quantile: state.normalizationQuantile, minimum: state.normalizationMinimum },
    camera: { fitted: state.fitted, scale: state.scale, centreX: (rect.width / 2 - state.x) / state.scale, centreY: (rect.height / 2 - state.y) / state.scale },
  });
}
function availablePane(raw) {
  const pane = validatePane(raw);
  requireOption('colour-by', pane.overlay.colourBy); requireOption('expression-source', pane.overlay.source);
  if (pane.channels.some(c => c.sourceId && !state.layers.some(s => s.id === c.sourceId && s.kind === 'imc')) || pane.layers.some(l => !state.layers.some(s => s.id === l.id))) throw new Error('This saved view refers to an aligned layer that has been removed. Reconnect that layer or save a new preset.');
  for (const source of state.layers.filter(s => s.kind !== 'imc')) if (!pane.layers.some(l => l.id === source.id)) pane.layers.push({ ...defaultLayer(source.id, source.kind), visible: false });
  return pane;
}
async function restoreSnapshot(raw) {
  const snapshot = validateSnapshot(raw), panes = snapshot.panes.map(availablePane);
  if (snapshot.mapping && (snapshot.mapping.roi !== state.mapping.roi || snapshot.mapping.id !== state.mapping.id)) throw new Error('This bookmark uses different cell identity columns. Restore its ROI and object ID mapping in Setup first.');
  if (!state.rois.some(r => r.name === snapshot.roi)) throw new Error(`Bookmark ROI is unavailable: ${snapshot.roi}.`);
  requireOption('embedding', snapshot.embedding);
  if (snapshot.embeddingOptions.colourBy !== 'view') requireOption('colour-by', snapshot.embeddingOptions.colourBy);
  requireOption('expression-source', snapshot.embeddingOptions.source);
  const activity = loading.begin('Opening bookmark', snapshot.roi); lockViewer(true);
  try {
    storeViewControls();
    if (![...$('roi').options].some(o => o.value === snapshot.roi)) { state.roiFilters = []; roiMetadataPanel.apply(false, snapshot.roi); }
    $('roi').value = snapshot.roi;
    state.figure = { ...snapshot.figure }; figurePanel?.resetReturnCamera();
    state.views = panes.map(p => newView(p.name, p)); state.selectedView = snapshot.selectedView;
    state.normalization = snapshot.normalization.dictionary ? new Map(Object.entries(snapshot.normalization.dictionary)) : null;
    state.normalizationQuantile = snapshot.normalization.quantile; state.normalizationMinimum = snapshot.normalization.minimum;
    $('normalization-name').textContent = snapshot.normalization.name || 'Choose CSV / JSON'; $('normalization-quantile').value = String(state.normalizationQuantile); $('normalization-minimum').value = String(state.normalizationMinimum);
    $('embedding').value = snapshot.embedding; state.embeddingOptions = { ...snapshot.embeddingOptions }; embeddingPanel.reset();
    cancelPendingRasters(); state.cache.clear(); updateChannels(); rebuildPanes();
    if (!await refresh(true)) throw new Error('Could not load the bookmarked ROI.');
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    resizeCanvas(); restoreCamera(snapshot.camera); await selectCell(snapshot.selected); compose(); drawEmbedding();
  } finally { lockViewer(false); activity.finish(); }
}
function restoreCamera(camera) {
  if (state.figure.mode === 'figure' && state.figure.physicalLock) { resizeCanvas(); const rect = viewportRect(); state.x = rect.width / 2 - camera.centreX * state.scale; state.y = rect.height / 2 - camera.centreY * state.scale; draw(); return; }
  if (camera.fitted) { fitView(); return; }
  const rect = viewportRect(); state.fitted = false; state.scale = camera.scale;
  state.x = rect.width / 2 - camera.centreX * state.scale; state.y = rect.height / 2 - camera.centreY * state.scale; draw();
}
async function applyViewPreset(raw) {
  const settings = availablePane(raw); storeViewControls();
  const camera = captureSnapshot().camera, name = activeView().name;
  const activity = loading.begin('Applying view preset', settings.name); lockViewer(true);
  try {
    state.views[state.selectedView] = newView(name, settings); updateChannels(); rebuildPanes();
    if (!await refresh(false)) throw new Error('Could not load the view preset.');
    restoreCamera(camera);
  } finally { lockViewer(false); activity.finish(); }
}
async function captureWorkspace() {
  if (!remapEditor.flush()) throw new Error('Fix the highlighted remap error before saving the workspace.');
  alignedPanel.flush(); flagPanel.flush(); plotPanel.capture(); representativePanel.capture(); validateFlags(state.flags);
  // Capture settings and source lists together, before the metadata digest yields.
  const view = captureView(), anndata = fileReference(state.anndata), masks = state.masks, images = state.images;
  const layerSources = state.layers.map(({ id, name, kind, files, dictionary }) => ({ id, name, kind, files, dictionary: dictionary ? { file: fileReference(dictionary.file), columns: { ...dictionary.columns } } : null }));
  const [maskReference, imageReference] = await Promise.all([folderReference(masks), folderReference(images)]);
  const layers = await Promise.all(layerSources.map(async ({ files, ...source }) => ({ ...source, folder: await folderReference(files) })));
  const payload = { format: WORKSPACE_FORMAT, version: WORKSPACE_VERSION, sources: { anndata, masks: maskReference, images: imageReference, layers }, view };
  workspacePanel?.applyLocations(payload);
  await workspacePanel?.resolveLocations(payload);
  parseWorkspace(JSON.stringify(payload)); // The same validation guards local saves and portable files.
  return payload;
}
async function exportWorkspace(fileHandle = null) {
  workspace.saving = true; updateWorkspaceActions(); workspaceError();
  try {
    const payload = await captureWorkspace(), name = workspacePanel.name;
    const text = JSON.stringify({ ...payload, name, applicationVersion: __VIEWER_VERSION__ }, null, 2);
    parseWorkspace(text);
    if (fileHandle) {
      const stream = await fileHandle.createWritable();
      try { await stream.write(text + '\n'); await stream.close(); } catch (error) { await stream.abort().catch(() => {}); throw error; }
      $('workspace-status').textContent = `Workspace written to ${fileHandle.name}.`;
      return;
    }
    const url = URL.createObjectURL(new Blob([text + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url;
    link.download = `${name.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')}.workspace.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    $('workspace-status').textContent = 'Workspace exported. Automatic local saving continues.';
  } catch (error) { workspaceError(error.message); }
  finally { workspace.saving = false; updateWorkspaceActions(); }
}
$('save-workspace').addEventListener('click', () => exportWorkspace());
let workspaceFileHandle = null;
$('write-workspace-file').hidden = !window.showSaveFilePicker;
$('write-workspace-file').addEventListener('click', async () => {
  try {
    if (workspaceFileHandle) {
      if (await workspaceFileHandle.requestPermission({ mode: 'readwrite' }) !== 'granted') return;
    } else workspaceFileHandle = await window.showSaveFilePicker({ suggestedName: `${workspacePanel.name}.workspace.json`, types: [{ description: 'IMC Explorer workspace', accept: { 'application/json': ['.json'] } }] });
    await exportWorkspace(workspaceFileHandle); $('write-workspace-file').textContent = `Update ${workspaceFileHandle.name}`;
  } catch (error) { if (error.name !== 'AbortError') workspaceError(error.message); }
});
function beginWorkspaceRestore(parsed) {
  workspace.pending = parsed; workspace.rootRequired = true; workspaceError();
  document.body.classList.add('workspace-connecting'); $('loading-overlay').hidden = true;
  showTab('setup'); lockViewer(true); $('workspace-locations').open = true;
  $('workspace-root-label').hidden = false; $('workspace-root').value = ''; updateWorkspaceActions();
  $('workspace-locations').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}
$('workspace-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  const request = ++workspace.reading;
  try {
    if (file.size > MAX_WORKSPACE_BYTES) throw new Error('Choose a workspace smaller than 10 MB.');
    const text = await file.text(), parsed = parseWorkspace(text), metadata = JSON.parse(text.replace(/^\uFEFF/, ''));
    if (request !== workspace.reading) return;
    const name = typeof metadata.name === 'string' && metadata.name.trim() ? metadata.name.trim().slice(0, 120) : file.name.replace(/(?:\.workspace)?\.json$/i, '');
    await workspacePanel.import(parsed, name);
  } catch (error) { if (request === workspace.reading) workspaceError(error.message); }
  finally { if (request === workspace.reading) event.target.value = ''; }
});
$('cancel-workspace').addEventListener('click', () => {
  workspace.reading++; workspace.attempt++; workspace.pending = null; workspace.rootRequired = false; $('workspace-root-label').hidden = true; workspaceError(); updateWorkspaceActions();
  $('workspace-progress').hidden = true;
  $('workspace-status').textContent = 'Workspace restore cancelled.';
  workspacePanel.cancel();
});
function requireOption(id, value) {
  if (![...$(id).options].some(option => option.value === value)) throw new Error(`Workspace ${id} is unavailable in this dataset: ${value || '(none)'}.`);
}
async function restoreWorkspace() {
  const saved = workspace.pending; if (!saved || workspace.rootRequired) return;
  const attempt = ++workspace.attempt, current = () => attempt === workspace.attempt && saved === workspace.pending;
  const { sources, view } = saved;
  const activity = loading.begin('Restoring workspace', 'Checking local files and saved view settings…');
  try {
    const [masks, images] = await Promise.all([folderReference(state.masks), folderReference(state.images)]);
    if (!current()) return;
    const missing = [];
    if (!state.header || !sameFile(sources.anndata, fileReference(state.anndata))) missing.push(`AnnData: ${sources.anndata.name}`);
    if (sources.masks && !sameFolder(sources.masks, masks)) missing.push(`Masks: ${sources.masks.name} (${sources.masks.count.toLocaleString()} TIFF files)`);
    if (sources.images && !sameFolder(sources.images, images)) missing.push(`Images: ${sources.images.name} (${sources.images.count.toLocaleString()} TIFF files)`);
    for (const source of sources.layers) {
      const layer = state.layers.find(item => item.id === source.id);
      if (!layer || !sameFolder(source.folder, await folderReference(layer.files))) missing.push(`Aligned layer: ${source.name}`);
      if (!current()) return;
    }
    if (missing.length) {
      workspaceError(`The following assets no longer match: ${missing.join(', ')}. Locate them in the checklist and retry.`);
      updateWorkspaceActions(); return;
    }
    const summaries = new Map();
    for (const plan of view.remaps) {
      activity.update(`Checking populations for ${plan.name}…`);
      if (!summaries.has(plan.source)) summaries.set(plan.source, await state.reader.call('populationSummary', plan.source));
      if (!current()) return;
      validateRemapSource(plan, summaries.get(plan.source));
    }
    state.remaps = structuredClone(view.remaps); state.activeRemap = view.activeRemap;
    state.flags = structuredClone(view.flags); state.activeFlag = view.activeFlag; flagPanel.reset();
    state.bookmarks = structuredClone(view.bookmarks); state.savedViews = structuredClone(view.savedViews); state.embeddingOptions = { ...view.embeddingOptions };
    navigationPanel.sync(); embeddingPanel.reset();
    state.representative = { ...view.representative }; representativePanel.reset();
    remapEditor.reset(); syncRemapOptions();
    for (const [id, value] of [['roi-column', view.mapping.roi], ['id-column', view.mapping.id], ['colour-by', view.overlay.colourBy], ['expression-source', view.overlay.source], ['embedding', view.embedding]]) requireOption(id, value);
    for (const pane of view.presets) { requireOption('colour-by', pane.overlay.colourBy); requireOption('expression-source', pane.overlay.source); }
    if (view.embeddingOptions.colourBy !== 'view') requireOption('colour-by', view.embeddingOptions.colourBy);
    requireOption('expression-source', view.embeddingOptions.source);
    workspace.lockedAttempt = attempt; lockWorkspace(true); workspaceError(); $('workspace-status').textContent = 'Restoring workspace…';
    state.revision++; normalizationRequest++; cancelPendingRasters(); state.cache.clear(); clearView();
    $('roi-column').value = view.mapping.roi; $('id-column').value = view.mapping.id;
    const rois = await state.reader.callProgress('configure', p => activity.update(p.detail), view.mapping.roi, view.mapping.id);
    if (!current()) return;
    state.rois = rois; state.mapping = { ...view.mapping };
    activity.update('Checking saved flags against cell identities…');
    await state.reader.call('validateFlagCells', state.flags);
    if (!current()) return;
    fillSelect($('roi'), rois.map(roi => ({ value: roi.name, text: roi.name })));
    $('previous-roi').disabled = false; $('next-roi').disabled = false;
    // A workspace without a source must not accidentally use an unrelated open folder.
    for (const [key, input, label] of [['masks', 'mask-files', 'masks-name'], ['images', 'image-files', 'images-name']]) {
      if (!sources[key]) { state[key] = []; $(input).value = ''; $(label).textContent = 'Choose folder'; }
    }
    state.layers = sources.layers.map(source => state.layers.find(item => item.id === source.id));
    indexFiles(); requireOption('roi', view.roi); $('roi').value = view.roi;
    const entries = state.files.get(view.roi)?.images ?? [];
    state.figure = { ...view.figure }; figurePanel?.resetReturnCamera();
    state.views = view.presets.map(preset => newView(preset.name, preset)); state.selectedView = view.selectedView;
    const channelFiles = state.views.map(pane => pane.channels.map(channel => {
      if (!channel.active) return null;
      const matchingSource = entries.filter(item => (item.sourceId ?? '') === (channel.sourceId ?? ''));
      if (!channel.path) return matchingSource.find(item => item.name === channel.name || (!channel.name.includes(' · ') && item.name === channelName({ name: channel.name }, view.roi, channelAliases(state.header)))) ?? null;
      const entry = matchingSource.find(item => relativeFilePath(item.file) === channel.path);
      if (!entry) throw new Error(`Workspace channel is unavailable in ${view.roi}: ${channel.path}.`);
      return entry;
    }));
    state.normalization = view.normalization.dictionary ? parseNormalization(JSON.stringify(view.normalization.dictionary), 'workspace.json') : null;
    state.normalizationQuantile = view.normalization.quantile; state.normalizationMinimum = view.normalization.minimum;
    $('normalization-name').textContent = view.normalization.name || 'Choose CSV / JSON';
    $('normalization-quantile').value = String(state.normalizationQuantile); $('normalization-minimum').value = String(state.normalizationMinimum);
    normalizationError(); updateNormalizationStatus();
    state.views.forEach((pane, index) => pane.channels.forEach((channel, i) => {
      const entry = channelFiles[index][i];
      Object.assign(channel, { file: entry?.file ?? null, raster: null });
      if (entry) channel.name = entry.name;
    }));
    $('embedding').value = view.embedding;
    state.order = view.order; state.visits = new Map(view.visits.map(([key, rois]) => [key, new Set(rois)]));
    activity.update('Restoring ROI order…');
    state.ranking = await state.reader.call('rank', state.order);
    if (!current()) return;
    syncOrderControls(); fillSelect($('roi'), state.ranking.map(item => ({ value: item.name, text: item.name })), view.roi);
    state.roiFilters = view.roiFilters; roiMetadataPanel.apply(false, view.roi);
    rebuildPanes(); syncViewControls(); showTab(view.tab);
    const loaded = await refresh(true);
    if (!current()) return;
    if (!loaded) throw new Error($('warning').textContent || 'The saved ROI could not be loaded.');
    // Selected categories may be absent in an ROI; retain them for the next ROI.
    for (const id of PANEL_IDS) $(id).open = view.panels[id];
    state.review = { ...view.review }; reviewPanel.sync();
    plotPanel.reset(view.plots);
    updateSectionAction();
    // Let the restored panel layout settle before positioning the image centre.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    if (!current()) return;
    resizeCanvas();
    if (view.camera.fitted) fitView();
    else {
      const rect = viewportRect();
      state.fitted = false; state.scale = view.camera.scale;
      state.x = rect.width / 2 - view.camera.centreX * state.scale; state.y = rect.height / 2 - view.camera.centreY * state.scale;
    }
    if (state.figure.mode === 'figure' && state.figure.physicalLock) restoreCamera(view.camera);
    await selectCell(view.selected);
    if (!current()) return;
    compose(); drawEmbedding(); workspace.pending = null; document.body.classList.remove('workspace-connecting'); $('workspace-root-label').hidden = true; $('workspace-progress').hidden = true;
    $('workspace-status').textContent = `Workspace restored · ${view.roi}.`;
    await workspacePanel.restored();
  } catch (error) {
    if (current()) { showTab('setup'); $('workspace-locations').open = true; workspaceError(error.message); $('workspace-status').textContent = 'Workspace could not be fully restored. Check the assets below and retry, or cancel.'; }
  } finally { activity.finish(); if (workspace.lockedAttempt === attempt) lockWorkspace(false); }
}

async function refresh(fit = false) {
  plotPanel.roiChanged();
  storeViewControls();
  const roi = $('roi').value;
  if (!state.reader || !state.rois.some(item => item.name === roi)) return;
  const revision = ++state.revision;
  cancelPendingRasters();
  const entry = state.files.get(roi) ?? { masks: [], images: [] };
  state.ready = false; clearSelection(); warning(); status(`Loading ${roi}…`);
  flagPanel.selectionChanged(); representativePanel.changed();
  $('view-title').textContent = roi; $('roi-summary').textContent = 'Loading images and cells…';
  roiBrowser.update(); drawEmbedding();
  const activity = loading.begin(`Loading ${roi}`, 'Preparing images and cell overlays…');
  const current = () => revision === state.revision;
  try {
    await yieldToBrowser();
    if (!current()) return;
    const snapshots = state.views.map(view => ({ view, channels: view.channels.map(channel => ({ ...channel, file: channel.active && channel.enabled ? channel.file : null, normalization: imageNormalization(channel) })), overlay: { ...view.overlay } }));
    const needed = new Set(snapshots.flatMap(item => item.channels.filter(channel => channel.file).map(channel => rasterKey(channel.file, channel.plane, false, channel.normalization))));
    for (const source of state.layers) {
      if (source.kind === 'imc') continue;
      const file = source.rois.get(roi);
      if (file && state.views.some(view => view.layers.find(layer => layer.id === source.id)?.visible)) needed.add(rasterKey(file, 0, false, null, source.kind));
    }
    if (entry.masks[0]) needed.add(rasterKey(entry.masks[0], 0, true));
    for (const [key, cached] of state.cache) if (!needed.has(key)) { cached.controller.abort(); state.cache.delete(key); }
    state.mask = null;
    for (const view of state.views) for (const channel of view.channels) channel.raster = null;
    for (const view of state.views) for (const layer of view.layers) layer.raster = null;
    activity.update('Reading segmentation mask…');
    const mask = entry.masks[0] ? await raster(entry.masks[0], 0, true) : null;
    if (revision !== state.revision) return;
    let basis = mask, bytes = mask?.data.byteLength ?? 0;
    const unique = new Set(mask ? [mask] : []), regions = new Map();
    const extraRasters = new Map();
    for (const source of state.layers) {
      if (source.kind === 'imc') continue;
      const file = source.rois.get(roi);
      if (!file || !state.views.some(view => view.layers.find(layer => layer.id === source.id)?.visible)) continue;
      activity.update(`Reading ${source.name} · ${file.name}…`);
      const image = await raster(file, 0, false, null, source.kind);
      if (!current()) return;
      basis ??= image;
      if (image.width !== basis.width || image.height !== basis.height) throw new Error(`${source.name}: image dimensions ${image.width} × ${image.height} differ from tissue dimensions ${basis.width} × ${basis.height}. Use an image aligned on the same pixel grid.`);
      if (!unique.has(image)) { unique.add(image); bytes += image.data.byteLength; }
      if (bytes + basis.width * basis.height * 8 * state.views.length > 512 * 1024 * 1024) throw new Error('These panes exceed the 512 MB display budget. Hide some layers or use fewer views.');
      extraRasters.set(source.id, image);
    }
    const loaded = [];
    // Decode sequentially and share cached planes between panes to cap transient RAM.
    for (const [index, snapshot] of snapshots.entries()) {
      const label = `View ${index + 1} of ${snapshots.length} · ${snapshot.view.name}`;
      activity.update(`${label} · Reading cell data…`);
      const { colourBy, source } = snapshot.overlay, plan = remapFor(colourBy), flag = state.flags.find(field => `flag:${field.id}` === colourBy);
      if (colourBy.startsWith('remap:') && !plan) throw new Error('This population remap is no longer available.');
      if (colourBy.startsWith('flag:') && !flag) throw new Error('This flag field is no longer available.');
      const readColour = flag ? '' : plan ? `obs:${plan.source}` : colourBy, key = JSON.stringify([readColour, source]);
      if (!regions.has(key)) regions.set(key, await state.reader.call('region', roi, readColour, source, $('embedding').value));
      if (revision !== state.revision) return;
      const rasters = [];
      for (const channel of snapshot.channels) {
        if (channel.file) activity.update(`${label} · Reading ${channel.name}…`);
        const image = channel.file ? await raster(channel.file, channel.plane, false, channel.normalization) : null;
        if (revision !== state.revision) return;
        if (image && !unique.has(image)) { unique.add(image); bytes += image.data.byteLength; }
        basis ??= image;
        if (basis && image && (image.width !== basis.width || image.height !== basis.height)) throw new Error('Image and mask dimensions differ. Select aligned images; the viewer does not resize scientific data.');
        if (bytes + (basis?.width ?? 0) * (basis?.height ?? 0) * 8 * state.views.length > 512 * 1024 * 1024) throw new Error('These panes exceed the 512 MB display budget. Use fewer channels or views for this ROI.');
        rasters.push(image);
      }
      loaded.push({ ...snapshot, region: flag ? applyFlags(regions.get(key), flag, roi) : plan ? applyRemap(regions.get(key), plan) : regions.get(key), recipe: flag ?? remapRecipe(plan), rasters });
    }
    const coverage = mask ? validateMask(mask.data, loaded[0].region.ids) : null;
    state.mask = mask; state.idToIndex = new Map(loaded[0].region.ids.map((id, i) => [id, i]));
    for (const { view, region, rasters, overlay, recipe } of loaded) {
      view.region = region;
      view.remapRecipe = recipe;
      view.layerRecipe = state.layers.map(({ id, kind, mapping }) => ({ id, kind, mapping }));
      for (const layer of view.layers) {
        layer.raster = extraRasters.get(layer.id) ?? null;
        if (layer.raster && layer.auto && state.layers.find(source => source.id === layer.id)?.kind === 'scalar') {
          [layer.low, layer.high] = contrast(layer.raster.data); layer.auto = false;
        }
      }
      view.colours = overlay.colourBy ? colourScale(region.values, region.categories, region.palette, overlay.colourmap, overlay.reverse, overlay) : { colour: () => [100, 216, 188], entries: [], range: null };
      rasters.forEach((image, i) => {
        const channel = view.channels[i]; channel.raster = image;
        if (image && !image.displayExtent) image.displayExtent = image.normalization ? [0, 1] : extent(image.data);
        if (image && channel.auto) { [channel.low, channel.high] = image.normalization ? [0, 1] : contrast(image.data); channel.auto = false; }
      });
    }
    const dimensionsChanged = state.width !== (basis?.width ?? 0) || state.height !== (basis?.height ?? 0);
    state.width = basis?.width ?? 0; state.height = basis?.height ?? 0;
    for (const [index, view] of state.views.entries()) {
      activity.update(`Rendering view ${index + 1} of ${state.views.length} · ${view.name}…`);
      if (!await composeViewAsync(view, state.figure.mode === 'figure' ? { ...state, selected: null, flagSelection: null } : state, current)) return;
    }
    if (!current()) return;
    state.ready = true;
    $('view-title').textContent = roi;
    $('roi-summary').textContent = `${state.region.rows.length.toLocaleString()} cells · ${entry.images.length} image files`;
    $('empty-state').hidden = Boolean(basis);
    syncViewControls(); draw(); scheduleVisits(); if (fit || dimensionsChanged) fitView(); drawEmbedding();
    flagPanel.sync(); navigationPanel.sync(); embeddingPanel.sync(); roiMetadataPanel.summary(); representativePanel.changed();
    const notes = [];
    if (!mask) notes.push('No matching mask: tissue cell picking is unavailable. Expected ROI.tif or ROI.tiff.');
    for (const pane of state.views) {
      const missing = pane.channels.filter(channel => channel.active && channel.enabled && channel.name && !channel.file);
      if (missing.length) notes.push(`${pane.name}: missing ${missing.map(channel => channel.name).join(', ')}.`);
      const missingLayers = state.layers.filter(source => pane.layers.find(layer => layer.id === source.id)?.visible && !source.rois.has(roi));
      if (missingLayers.length) notes.push(`${pane.name}: no aligned image for ${missingLayers.map(source => source.name).join(', ')} in this ROI.`);
    }
    if (coverage?.missing) notes.push(`${coverage.missing} AnnData cells are absent from this mask.`);
    if (coverage?.context) notes.push(`${coverage.context} mask objects have no AnnData row and appear grey.`);
    warning(notes.join(' '));
    status(`${roi} ready · ${state.region.rows.length.toLocaleString()} cells · ${basis ? `${basis.width} × ${basis.height} px · ` : ''}${state.views.length} linked view${state.views.length === 1 ? '' : 's'}`);
    return true;
  } catch (error) { if (revision === state.revision) { cancelPendingRasters(); state.cache.clear(); clearView(); fail(error); } }
  finally { activity.finish(); }
}

function included(i) { return includedInView(activeView(), i); }

function updateLegend() {
  $('legend').replaceChildren();
  const counts = new Map();
  for (const value of state.region.values) counts.set(value, (counts.get(value) || 0) + 1);
  for (const [category, colour] of state.colours.entries.slice(0, 100)) {
    const row = document.createElement('div'); row.className = 'legend-row';
    const swatch = document.createElement('i'); swatch.style.backgroundColor = `rgb(${colour})`;
    const name = document.createElement('span'); name.textContent = category == null ? activeView().overlay.colourBy.startsWith('flag:') ? 'Unassigned' : 'Missing' : String(category);
    const count = document.createElement('small'); count.textContent = (counts.get(category) || 0).toLocaleString();
    row.append(swatch, name, count); $('legend').append(row);
  }
  if (state.colours.entries.length > 100) { const more = document.createElement('p'); more.textContent = 'First 100 categories shown. Use the population selector for all categories.'; $('legend').append(more); }
  if (state.colours.range) {
    const bar = document.createElement('div'); bar.className = 'gradient';
    bar.style.background = colourmapGradient(activeView().overlay.colourmap, activeView().overlay.reverse);
    const text = document.createElement('div'); text.className = 'range-label';
    text.textContent = state.colours.range.map(value => value.toPrecision(4)).join(' — ');
    $('legend').append(bar, text);
  }
}

function compose() {
  storeViewControls();
  if (!state.ready || !state.width) return;
  for (const view of state.views) composeView(view, state.figure.mode === 'figure' ? { ...state, selected: null, flagSelection: null } : state);
  draw(); scheduleVisits(); syncQuickTools(); renderViewLegends(state);
}

let viewportSize = null;
function resizeCanvas() {
  figurePanel?.layout();
  const rect = viewportRect();
  for (const view of state.views) if (view.canvas) { view.canvas.width = Math.round(rect.width * devicePixelRatio); view.canvas.height = Math.round(rect.height * devicePixelRatio); }
  if (state.figure.mode === 'figure' && state.figure.physicalLock && state.width) {
    const centreX = ((viewportSize?.width ?? rect.width) / 2 - state.x) / state.scale, centreY = ((viewportSize?.height ?? rect.height) / 2 - state.y) / state.scale;
    state.scale = rect.width * state.figure.umPerPixel / state.figure.physicalWidth; state.fitted = false;
    state.x = rect.width / 2 - centreX * state.scale; state.y = rect.height / 2 - centreY * state.scale;
  } else if (state.fitted && state.width) fitView();
  else if (viewportSize) { state.x += (rect.width - viewportSize.width) / 2; state.y += (rect.height - viewportSize.height) / 2; }
  viewportSize = { width: rect.width, height: rect.height }; draw(); drawEmbedding();
}
function draw() {
  for (const view of state.views) {
    const target = view.canvas; if (!target) continue;
    const context = target.getContext('2d');
    context.setTransform(1, 0, 0, 1, 0, 0); context.clearRect(0, 0, target.width, target.height);
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); context.imageSmoothingEnabled = false;
    if (view.scene.width) context.drawImage(view.scene, state.x, state.y, state.width * state.scale, state.height * state.scale);
    if (state.lasso?.view === view && state.lasso.points.length) {
      context.beginPath(); state.lasso.points.forEach(([x, y], i) => context[i ? 'lineTo' : 'moveTo'](state.x + x * state.scale, state.y + y * state.scale)); context.closePath(); context.strokeStyle = '#ffffff'; context.lineWidth = 1.5; context.fillStyle = '#64d8bc22'; context.fill(); context.stroke();
    }
  }
  $('view-scale').textContent = state.width ? `${Math.round(state.scale * 100)}%` : '—';
  figurePanel?.draw();
}

function fitView() {
  if (state.figure.mode === 'figure' && state.figure.physicalLock) { resizeCanvas(); return; }
  state.fitted = true;
  if (!state.width) return;
  const rect = viewportRect();
  state.scale = Math.min((rect.width - 48) / state.width, (rect.height - 48) / state.height);
  state.x = (rect.width - state.width * state.scale) / 2; state.y = (rect.height - state.height * state.scale) / 2; draw();
}
function zoom(factor, x, y) {
  if (state.figure.mode === 'figure' && state.figure.physicalLock) return;
  if (!state.width) return;
  state.fitted = false;
  const scale = Math.min(64, Math.max(0.01, state.scale * factor));
  state.x = x - (x - state.x) * scale / state.scale; state.y = y - (y - state.y) * scale / state.scale; state.scale = scale; draw();
}
$('fit').addEventListener('click', fitView);
for (const [id, factor] of [['zoom-in', 1.4], ['zoom-out', 1 / 1.4]]) $(id).addEventListener('click', () => { const rect = viewportRect(); zoom(factor, rect.width / 2, rect.height / 2); });
function bindPane(target, index) {
  target.sbtListeners?.abort(); target.sbtListeners = new AbortController();
  const listen = (name, handler, options = {}) => target.addEventListener(name, handler, { ...options, signal: target.sbtListeners.signal });
  let drag;
  target.cancelGesture = () => { drag = null; };
  const imagePoint = event => { const rect = target.getBoundingClientRect(); return [(event.clientX - rect.left - state.x) / state.scale, (event.clientY - rect.top - state.y) / state.scale]; };
  listen('wheel', event => { event.preventDefault(); selectView(index); const rect = target.getBoundingClientRect(); zoom(Math.exp(-event.deltaY * 0.001), event.clientX - rect.left, event.clientY - rect.top); }, { passive: false });
  listen('pointerdown', event => { if (event.button !== 0 || !state.ready) return; selectView(index); flagPanel.space('tissue'); drag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, mode: flagPanel.mode() }; if (drag.mode === 'lasso' && state.mask) state.lasso = { view: activeView(), points: [imagePoint(event)] }; target.setPointerCapture(event.pointerId); });
  listen('pointermove', event => {
    if (!drag) return;
    if (state.lasso) { state.lasso.points.push(imagePoint(event)); draw(); return; }
    if (event.clientX !== drag.x || event.clientY !== drag.y) state.fitted = false;
    state.x += event.clientX - drag.x; state.y += event.clientY - drag.y; drag.x = event.clientX; drag.y = event.clientY; draw();
  });
  listen('pointerup', event => {
    if (!drag) return;
    const distance = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY), mode = drag.mode; drag = null;
    if (state.lasso) { const points = state.lasso.points; state.lasso = null; if (state.ready && points.length >= 3) flagPanel.lasso(points); draw(); return; }
    if (distance > 4 || !state.ready || !state.mask) return;
    const rect = target.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left - state.x) / state.scale), y = Math.floor((event.clientY - rect.top - state.y) / state.scale);
    if (x >= 0 && y >= 0 && x < state.width && y < state.height) { const id = state.mask.data[y * state.width + x]; if (mode === 'click') flagPanel.click(id); else selectCell(id); }
  });
  listen('pointercancel', () => { drag = null; state.lasso = null; draw(); });
}


async function selectCell(id) {
  if (!state.ready) return;
  clearSelection(); state.selected = id || null; compose(); drawEmbedding();
  if (!id) return;
  const index = state.idToIndex.get(id);
  $('cell-badge').textContent = `Object ${id}`;
  if (index === undefined) { $('cell-hint').textContent = 'This mask object has no AnnData row in this ROI.'; return; }
  $('cell-hint').textContent = 'Reading observations…';
  const revision = state.revision;
  try {
    const fields = await state.reader.call('cell', state.region.rows[index]);
    if (revision !== state.revision || state.selected !== id) return;
    if ($('colour-by').value.startsWith('var:')) fields.push([`${$('colour-by').value.slice(4)} (${$('expression-source').value})`, state.region.values[index]]);
    const plan = remapFor($('colour-by').value);
    if (plan) fields.push([`Proposed · ${plan.name}`, state.region.values[index]]);
    for (const field of state.flags) fields.push([`Flags · ${field.name}`, field.assignments.find(entry => entry[0] === $('roi').value && entry[1] === id)?.[2] ?? null]);
    $('cell-details').replaceChildren(); $('cell-hint').hidden = true;
    for (const [key, value] of fields) {
      const term = document.createElement('dt'); term.textContent = key;
      const detail = document.createElement('dd'); detail.textContent = value == null ? 'Missing' : String(value);
      $('cell-details').append(term, detail);
    }
  } catch (error) { if (revision === state.revision) fail(error); }
}

function drawEmbedding() { embeddingPanel?.draw(); viewerToolbar?.sync(); }
const roiBrowser = createROIBrowser(state, $, fillSelect, warning, () => roiMetadataPanel.apply(), () => viewerToolbar?.sync());
const alignedPanel = createAlignedPanel(state, $, { refresh, compose, updateLegends: updateViewLegends, showChannels() { showTab('views'); $('section-channels').open = true; $('channel-0').focus(); }, sourcesChanged() { try { indexFiles(); updateChannels(); return refresh(); } catch (error) { fail(error); } } });
alignedPanel.sync();
const reviewPanel = createReviewPanel(state, $, drawEmbedding);
const remapEditor = createRemapEditor(state, $, {
  changed: remapsChanged,
  useOverlay(plan) {
    storeViewControls(); activeView().overlay.colourBy = `remap:${plan.id}`; populationFilter(activeView().overlay, []); activeView().overlayVisible = true;
    syncRemapOptions(); syncViewControls(); showTab('views'); refresh();
  },
});
remapEditor.sync();
const plotPanel = createPlotPanel(state, $, () => remapEditor.flush());
plotPanel.sync();
const flagPanel = createFlagPanel(state, $, { changed: flagsChanged, draw() { compose(); drawEmbedding(); }, included, selectEmbedding() { embeddingPanel.selectVisible(); },
  selectionChanged() { viewerToolbar?.sync(); },
  modeChanged() { state.lasso = null; for (const view of state.views) view.canvas?.cancelGesture?.(); embeddingPanel?.cancelGesture(); viewerToolbar?.sync(); draw(); },
  use(field) { storeViewControls(); activeView().overlay.colourBy = `flag:${field.id}`; populationFilter(activeView().overlay, []); activeView().overlayVisible = true; syncRemapOptions(); syncViewControls(); showTab('views'); refresh(); },
});
flagPanel.sync();
embeddingPanel = createEmbeddingPanel(state, $, {
  included, mode: () => flagPanel.mode(), space: value => flagPanel.space(value),
  selected() { flagPanel.selectionChanged(); compose(); drawEmbedding(); }, changed() { workspacePanel?.mark(); },
  async inspect(roi, id) {
    if (roi !== $('roi').value) { if (![...$('roi').options].some(o => o.value === roi)) { state.roiFilters = []; roiMetadataPanel.apply(false, roi); } $('roi').value = roi; updateChannels(); if (!await refresh(state.fitted)) return; }
    await selectCell(id);
  },
});
embeddingPanel.sync();
viewerToolbar = createViewerToolbar(state, $, { mode: flagPanel.mode, setMode: flagPanel.setMode, scope: flagPanel.scope });
navigationPanel = createNavigationPanel(state, $, {
  snapshot: captureSnapshot, restore: restoreSnapshot, preset: applyViewPreset,
  view() { storeViewControls(); return viewSettings(activeView()); },
  open() { reviewPanel.select('bookmarks'); }, changed() { workspacePanel?.mark(); },
  error: fail, locked: () => loadingActive || workspace.applying || Boolean(workspace.pending),
});
const figureFilenameContext = () => ({ roi: $('roi').value, viewNames: state.views.map(view => view.name), metadata: state.roiMetadata, metadataLoaded: roiMetadataPanel?.isLoaded() ?? false });
figurePanel = createFigurePanel(state, $, {
  filenameContext: figureFilenameContext,
  async readMetadata() { await roiMetadataPanel.load(); if (!roiMetadataPanel.isLoaded()) throw new Error($('roi-metadata-status').textContent); },
  camera: () => captureSnapshot().camera, restoreCamera,
  modeChanged() { flagPanel.setMode('inspect'); compose(); }, layout() { resizeCanvas(); syncQuickTools(); }, draw,
  changed() { workspacePanel?.mark(); }, open() { reviewPanel.select('figure'); },
  async export(progress) {
    if (loadingActive || !state.ready) throw new Error('Wait for the ROI to finish loading.');
    if (roiMetadataPanel?.isLoading()) throw new Error('Wait for ROI metadata to finish loading before exporting.');
    if (state.figure.autoBookmark && state.bookmarks.length >= 500) throw new Error('Remove a bookmark or turn off automatic bookmarking before exporting.');
    const snapshot = captureSnapshot(), rect = viewportRect();
    const filename = figureFilename(state.figure, figureFilenameContext());
    const activity = loading.begin('Exporting figure', 'Rendering panes and overlays…'); lockViewer(true);
    try {
      const blob = await createFigureExport({ ...state, exportROI: $('roi').value }, rect, message => { progress(message); activity.update(message); });
      if (state.figure.autoBookmark) navigationPanel.addExportBookmark(snapshot, filename);
      const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
      return filename;
    } finally { lockViewer(false); activity.finish(); }
  },
});
const representativePanel = createRepresentativePanel(state, $, (name, index) => {
  const entry = state.files.get($('roi').value)?.images.find(item => item.name === name); if (!entry) return;
  const channel = state.channels[index], same = channel.name === name && (channel.sourceId ?? '') === (entry.sourceId ?? '');
  Object.assign(channel, { active: true, enabled: true, file: entry.file, raster: same ? channel.raster : null, name, sourceId: entry.sourceId ?? '', plane: 0, colour: channelColours[index], auto: same ? channel.auto : true });
  state.imagesVisible = true; syncViewControls(); refresh();
});
representativePanel.sync();
roiMetadataPanel = createROIMetadataPanel(state, $, fillSelect, changedROI => {
  roiBrowser.update();
  figurePanel?.sync();
  if (changedROI && !workspace.applying) {
    if ($('roi').value) $('roi').dispatchEvent(new Event('change'));
    else { state.revision++; clearView(); status('No ROIs match these filters. Change or reset the filters in the ROIs tab.'); representativePanel.changed(); }
  }
});
for (const [quick, target] of [['quick-previous', 'previous-roi'], ['quick-next', 'next-roi'], ['quick-images', 'toggle-images'], ['quick-overlay', 'toggle-overlay']]) $(quick).addEventListener('click', () => $(target).click());
function syncOrderControls() { return roiBrowser.sync(); }
function scheduleVisits() { return roiBrowser.scheduleVisits(); }
rebuildPanes();
new ResizeObserver(resizeCanvas).observe($('viewport'));
new ResizeObserver(drawEmbedding).observe($('embedding-canvas'));
resizeCanvas();
workspacePanel = createWorkspacePanel($, {
  assetHandles: () => new Map([['anndata', state.anndata?.workspaceHandle], ['masks', state.masks.workspaceHandle], ['images', state.images.workspaceHandle], ...state.layers.flatMap(source => [[`layer:${source.id}`, source.files.workspaceHandle], [`dictionary:${source.id}`, source.dictionary?.file.workspaceHandle]])].filter(([, handle]) => handle)),
  capture: captureWorkspace,
  hasData: () => Boolean(state.anndata && state.mapping),
  canSave: () => state.ready && Boolean(state.anndata) && !loadingActive && !workspace.pending && !workspace.applying && !workspace.resolvingRoot && !workspace.saving && !document.querySelector('#aligned-add.is-loading, #remap-panel.is-counting'),
  beginRestore: beginWorkspaceRestore,
  loadFiles: loadWorkspaceAssets,
  assetValues: () => new Map([['anndata', state.anndata], ['masks', state.masks], ['images', state.images], ...state.layers.flatMap(source => [[`layer:${source.id}`, source.files], [`dictionary:${source.id}`, source.dictionary?.file]])].filter(([, value]) => value)),
  showSetup: () => showTab('setup'),
  activated() { workspaceFileHandle = null; $('write-workspace-file').textContent = 'Save to file…'; },
  freeze(locked) { lockViewer(locked || Boolean(workspace.pending) || workspace.applying); },
  rootBusy: () => workspace.applying || workspace.resolvingRoot,
});
