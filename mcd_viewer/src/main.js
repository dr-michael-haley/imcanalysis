import { parseSchema, imageRange, readOpticalImage } from './mcd.js';
import { createDemoFile } from './demo.js';
import { ImageViewer, composite } from './viewer.js';
import { CORE_COLOURS, DEFAULT_COLOURS as colours, adjacentChannel, orderedChannels } from './channel-controls.js';
import { panoramaProjection, slideProjection, coordinateOverview } from './overview.js';
import { slideImageSources } from './optical-sources.js';
import { OverviewNavigation } from './overview-navigation.js';
import { autoBounds, contrastControl } from './contrast-control.js';
import { COLOUR_MAPS, mapName, colourGradient } from './colormaps.js';
import { HistogramPanel } from './histogram-panel.js';
import { ExportPanel } from './export-panel.js';

const $ = id => document.getElementById(id);
const state = { file: null, model: null, acquisition: null, slots: [], raster: null, fileEpoch: 0, renderEpoch: 0, mapEpoch: 0, mode: 'channels', optical: null, controller: null, backdrop: 'panorama', overviewImage: null };
let exportPanel = null;
const viewer = new ImageViewer($('canvas'), { onZoom: scale => { $('zoom-label').textContent = `${Math.round(scale * 100)}%`; }, onPixel: inspectPixel, onDraw: () => exportPanel?.draw() });
const overviewNavigation = new OverviewNavigation($('map'), scale => { $('map-zoom').textContent = `${Math.round(scale * 100)}%`; });
const histogramPanel = new HistogramPanel(() => state, runTask);
const channelList = a => orderedChannels(a.channels, $('channel-order').value);
const option = (label, value) => new Option(label, value);
function node(tag, text, className) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; }
function message(text) { $('message-text').textContent = text; $('message').hidden = !text; }
function status(text) { $('status').textContent = text; }
function busy(text = '', fraction = null) {
  $('busy').hidden = !text;
  $('busy-label').textContent = text;
  if (fraction === null) $('progress').removeAttribute('value'); else $('progress').value = fraction;
}
function errorMessage(error) { if (error.name !== 'AbortError') { message(error.message || String(error)); status('Could not complete the request'); } }
function abortLoad() { state.controller?.abort(); state.controller = null; state.renderEpoch++; state.renderPending = false; }
function runTask(payload, signal, showProgress = true) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./reader.worker.js', import.meta.url), { type: 'module' });
    const cleanup = () => { worker.terminate(); signal?.removeEventListener('abort', cancel); };
    const cancel = () => { cleanup(); reject(new DOMException('Loading cancelled', 'AbortError')); };
    if (signal?.aborted) { cancel(); return; }
    signal?.addEventListener('abort', cancel, { once: true });
    worker.onmessage = ({ data }) => {
      if (data.progress) { if (showProgress) busy(data.progress.stage, data.progress.fraction); return; }
      cleanup(); if (data.error) reject(new Error(data.error)); else resolve(data.result);
    };
    worker.onerror = event => { cleanup(); reject(new Error(event.message || 'The browser could not start the MCD reader. Open this app using its local server.')); };
    worker.postMessage(payload);
  });
}
function controls() {
  const a = state.acquisition;
  $('add-channel').disabled = !a || state.slots.length >= Math.min(6, a.channels.length);
  $('auto-all').disabled = !a;
  $('transform').disabled = !a;
  $('channel-count').textContent = a ? `${state.slots.length} / ${a.channels.length}` : '—';
  $('viewer-tools').hidden = !viewer.source;
  $('image-mode-wrap').hidden = !a;
  $('map-blend-note').hidden = !state.slots.some(s => s.map && s.map !== 'pure');
  histogramPanel.sync();
  exportPanel?.sync();
}
function reset() {
  exportPanel?.sourceChanging(true);
  abortLoad(); state.fileEpoch++; state.mapEpoch++;
  state.file = null; state.model = null; state.acquisition = null; state.raster = null; state.slots = []; state.mode = 'channels';
  state.backdrop = 'panorama'; state.overviewImage = null;
  state.optical?.close?.(); state.optical = null; viewer.clear();
  $('welcome').hidden = false; $('file-name').textContent = 'Choose a file or drop it anywhere.'; $('file-size').textContent = '';
  $('file').value = ''; $('close-file').hidden = true; $('channels').replaceChildren();
  $('acquisition').replaceChildren(option('No acquisitions', '')); $('slide').replaceChildren(option('Open a file to begin', ''));
  $('acquisition').disabled = $('slide').disabled = $('search').disabled = $('previous').disabled = $('next').disabled = true;
  $('search').value = ''; $('acquisition-count').textContent = '—'; $('acquisition-summary').textContent = 'Browse regions directly from your MCD.';
  $('overview').hidden = true; $('map').replaceChildren(); $('legend').hidden = true; $('pixel-panel').hidden = true;
  $('metadata').replaceChildren(node('dt', 'Open an MCD to inspect its metadata.'));
  $('view-title').textContent = 'Your tissue, a closer look.'; $('view-context').textContent = 'IMAGING MASS CYTOMETRY';
  message(''); busy(); controls(); status('Ready to open an MCD file');
}
async function openFile(file) {
  if (!file) return;
  if (!/\.mcd$/i.test(file.name)) return message('Choose an imaging mass cytometry .mcd file.');
  reset(); state.file = file;
  const epoch = state.fileEpoch, controller = new AbortController(); state.controller = controller;
  $('welcome').hidden = true; $('file-name').textContent = file.name;
  $('file-size').textContent = file.size >= 1e9 ? `${(file.size / 1e9).toFixed(2)} GB` : `${(file.size / 1e6).toFixed(1)} MB`;
  $('close-file').hidden = false; busy('Finding MCD metadata'); status('Opening local file…');
  try {
    const xml = await runTask({ type: 'schema', file }, controller.signal);
    if (epoch !== state.fileEpoch) return;
    state.model = parseSchema(xml);
    const slides = state.model.slides.length ? state.model.slides : [{ id: '', name: 'All acquisitions' }];
    $('slide').replaceChildren(option('All slides', '*'), ...slides.map(s => option(s.name, s.id)));
    $('slide').disabled = slides.length < 2; $('search').disabled = false;
    $('acquisition-count').textContent = state.model.acquisitions.length;
    refreshList();
    await selectAcquisition(state.model.acquisitions[0].id);
  } catch (error) { if (epoch === state.fileEpoch) { busy(); errorMessage(error); } }
}
function filteredAcquisitions() {
  const text = $('search').value.trim().toLowerCase(), slide = $('slide').value;
  return (state.model?.acquisitions || []).filter(a => (slide === '*' || a.slideId === slide) && `${a.name} ${a.id}`.toLowerCase().includes(text));
}
function refreshList() {
  const list = filteredAcquisitions();
  $('acquisition').replaceChildren(...list.map(a => option(`${a.id} · ${a.name}`, a.id)));
  if (!list.length) $('acquisition').add(option('No matching acquisitions', ''));
  else if (list.some(a => a.id === state.acquisition?.id)) $('acquisition').value = state.acquisition.id;
  $('acquisition').disabled = !list.length;
  const index = list.findIndex(a => a.id === $('acquisition').value);
  $('previous').disabled = index <= 0; $('next').disabled = index < 0 || index >= list.length - 1;
}
async function filterChanged() {
  refreshList();
  const list = filteredAcquisitions();
  if (list.length && !list.some(a => a.id === state.acquisition?.id)) await selectAcquisition(list[0].id);
}
function initialSlots(a) {
  const previous = state.slots;
  if (previous.length) {
    const matched = previous.map(s => { const c = a.channels.find(c => c.key === s.key); return c ? { ...s, index: c.index, low: null, high: null } : null; }).filter(Boolean);
    if (matched.length) return matched;
  }
  const preferred = channelList(a);
  return preferred.slice(0, 3).map((c, i) => ({ index: c.index, key: c.key, colour: colours[i], enabled: true, low: null, high: null }));
}
async function selectAcquisition(id, restoredSlots = null) {
  const a = state.model?.acquisitions.find(a => a.id === id); if (!a) return;
  exportPanel?.sourceChanging();
  abortLoad(); message(''); state.slots = restoredSlots || initialSlots(a); state.acquisition = a; state.mode = 'channels'; state.raster = null;
  state.optical?.close?.(); state.optical = null;
  viewer.clear(); viewer.fitted = true; $('pixel-panel').hidden = true; $('legend').hidden = true;
  $('acquisition').value = a.id; refreshList();
  $('view-title').textContent = a.name; $('view-context').textContent = `${state.file.name} / ACQUISITION ${a.id}`;
  $('acquisition-summary').textContent = `${a.width.toLocaleString()} × ${a.height.toLocaleString()} px · ${a.channels.length} channels`;
  $('metadata').replaceChildren(...Object.entries(a.metadata).flatMap(([key, value]) => [node('dt', key), node('dd', value || '—')]));
  const p = state.model.panoramas.find(p => p.id === a.panoramaId);
  state.images = [
    { id: 'panorama', label: 'Panorama', range: p && imageRange(p.metadata) },
    { id: 'before', label: 'Before ablation', range: imageRange(a.metadata, 'BeforeAblationImage') },
    { id: 'after', label: 'After ablation', range: imageRange(a.metadata, 'AfterAblationImage') },
    ...slideImageSources(state.model, a.slideId),
  ].filter(i => i.range);
  $('image-mode').replaceChildren(option('Channel composite', 'channels'), ...state.images.map(i => option(i.label, i.id)));
  renderChannels(); controls(); drawMap();
  await loadChannels();
}
async function loadChannels() {
  if (!state.acquisition) return;
  if (!state.slots.length) { busy(); message('This acquisition contains no image channels. Embedded optical images can still be viewed when available.'); return; }
  abortLoad();
  const controller = new AbortController(); state.controller = controller;
  const a = state.acquisition, file = state.file;
  const existing = state.raster;
  const requested = [...new Set(state.slots.map(s => s.index))];
  if (existing && requested.every(index => existing.planes.some(p => p.index === index))) {
    existing.planes = existing.planes.filter(p => requested.includes(p.index));
    busy(); applyLimits(); renderChannels(); await render(); return;
  }
  state.raster = null; viewer.clear(); controls(); $('legend').hidden = true; busy('Reading acquisition', 0); status('Reading channel data locally…');
  try {
    const raster = await runTask({ type: 'acquisition', file, acquisition: a, selected: requested }, controller.signal);
    if (controller.signal.aborted) return;
    state.raster = raster; applyLimits(); renderChannels();
    message(raster.warnings.join(' '));
    busy(); await render(); if (controller.signal.aborted) return; controls();
    status(`${raster.filled.toLocaleString()} pixels read · ${a.channels.length} channels · ${file.name}`);
  } catch (error) { if (!controller.signal.aborted) { busy(); errorMessage(error); controls(); } }
}
function applyLimits() {
  for (const s of state.slots) {
    const p = state.raster?.planes.find(p => p.index === s.index);
    if (!p) continue;
    if (s.low === null || s.high === null) Object.assign(s, autoBounds(p.limits));
    s.max = Math.max(p.limits.max, s.high);
  }
}
function changeChannel(slot, channel) {
  if (!channel || slot.index === channel.index) return;
  slot.index = channel.index; slot.key = channel.key; slot.low = slot.high = null;
  state.mode = 'channels'; $('image-mode').value = 'channels'; $('pixel-panel').hidden = true;
  renderChannels(); loadChannels();
}
function renderChannels() {
  $('channels').replaceChildren();
  const a = state.acquisition; if (!a) return;
  state.slots.forEach((s, slotIndex) => {
    const channel = a.channels.find(c => c.index === s.index);
    const card = node('div', undefined, `channel${s.enabled ? '' : ' disabled'}`);
    const top = node('div', undefined, 'channel-top');
    const check = document.createElement('input'); check.type = 'checkbox'; check.checked = s.enabled; check.ariaLabel = `Show ${channel.label}`;
    check.onchange = () => { s.enabled = check.checked; card.classList.toggle('disabled', !s.enabled); controls(); render(); };
    const select = document.createElement('select'); select.ariaLabel = `Channel ${slotIndex + 1}`;
    select.replaceChildren(...channelList(a).map(c => option(c.label, c.index))); select.value = s.index;
    select.onchange = () => changeChannel(s, a.channels.find(c => c.index === Number(select.value)));
    const navigation = node('div', undefined, 'channel-stepper');
    const previous = node('button', '‹', 'channel-arrow'), next = node('button', '›', 'channel-arrow');
    for (const [button, direction, label] of [[previous, -1, 'Previous'], [next, 1, 'Next']]) {
      button.ariaLabel = `${label} channel in slot ${slotIndex + 1}`;
      button.title = `${label} channel (wraps at the end)`; button.disabled = a.channels.length < 2;
      button.onclick = () => changeChannel(s, adjacentChannel(channelList(a), s.index, direction));
    }
    navigation.append(previous, select, next);
    const remove = node('button', '×', 'remove-channel'); remove.ariaLabel = `Remove ${channel.label}`; remove.disabled = state.slots.length <= 1;
    remove.onclick = () => { state.slots.splice(slotIndex, 1); renderChannels(); controls(); render(); };
    top.append(check, navigation, remove); card.append(top, node('div', channel.name, 'channel-label'));
    const palette = node('div', undefined, 'channel-palette'); palette.setAttribute('role', 'group'); palette.ariaLabel = `Colour for channel slot ${slotIndex + 1}`;
    for (const colour of CORE_COLOURS) {
      const swatch = node('button', undefined, 'colour-swatch'); swatch.style.backgroundColor = colour.hex;
      swatch.ariaLabel = `${colour.name} for channel slot ${slotIndex + 1}`; swatch.title = `${colour.name} (${colour.hex})`;
      swatch.setAttribute('aria-pressed', String((!s.map || s.map === 'pure') && s.colour === colour.hex));
      swatch.onclick = () => {
        s.colour = colour.hex; s.map = 'pure'; mapSelect.value = 'pure'; gradient.style.background = colourGradient(s);
        for (const button of palette.children) button.setAttribute('aria-pressed', String(button === swatch));
        controls(); render();
      };
      palette.append(swatch);
    }
    card.append(palette);
    const mapLabel = node('label', undefined, 'colour-map-label'); mapLabel.append(document.createTextNode('Colour map'));
    const mapSelect = document.createElement('select'); mapSelect.ariaLabel = `Colour map for channel slot ${slotIndex + 1}`;
    mapSelect.replaceChildren(option('Pure colour', 'pure'), ...COLOUR_MAPS.map(id => option(mapName(id), id))); mapSelect.value = s.map || 'pure';
    const gradient = node('div', undefined, 'colour-gradient'); gradient.style.background = colourGradient(s); gradient.ariaHidden = 'true';
    mapSelect.onchange = () => {
      s.map = mapSelect.value; gradient.style.background = colourGradient(s);
      for (const button of palette.children) button.setAttribute('aria-pressed', String(s.map === 'pure' && button.title.endsWith(`(${s.colour})`)));
      controls(); render();
    };
    mapLabel.append(mapSelect); card.append(mapLabel, gradient);
    card.append(contrastControl(s, channel.label, !!state.raster, render, () => { s.low = s.high = null; applyLimits(); renderChannels(); render(); }));
    $('channels').append(card);
  });
  controls();
}
async function render() {
  const epoch = ++state.renderEpoch;
  if (state.mode !== 'channels' || !state.raster) return;
  state.renderPending = true; exportPanel?.sync();
  const snapshot = state.slots.map(s => ({ ...s }));
  const cofactor = Number($('cofactor').value);
  const surface = await composite(state.raster, snapshot, $('transform').value, Number.isFinite(cofactor) && cofactor > 0 ? cofactor : 5, () => epoch !== state.renderEpoch);
  if (!surface || epoch !== state.renderEpoch) return;
  state.renderPending = false;
  viewer.setSource(surface); controls();
  $('legend').replaceChildren(...snapshot.filter(s => s.enabled).map(s => {
    const label = state.acquisition.channels.find(c => c.index === s.index)?.label;
    const span = node('span'); const dot = node('i'); dot.style.background = colourGradient(s); dot.className = 'legend-gradient'; span.append(dot, document.createTextNode(`${label}${s.map && s.map !== 'pure' ? ` · ${mapName(s.map)}` : ''}`)); return span;
  }));
  $('legend').hidden = !snapshot.some(s => s.enabled);
}
async function imageModeChanged() {
  exportPanel?.sourceChanging();
  state.mode = $('image-mode').value; state.renderEpoch++;
  $('pixel-panel').hidden = true; $('legend').hidden = true;
  state.optical?.close?.(); state.optical = null;
  viewer.clear();
  if (state.mode === 'channels') { abortLoad(); busy(); viewer.fitted = true; if (state.raster) await render(); else await loadChannels(); return; }
  abortLoad();
  const controller = new AbortController(); state.controller = controller;
  const item = state.images.find(i => i.id === state.mode); if (!item) return;
  busy(`Opening ${item.label.toLowerCase()}`); viewer.clear(); controls();
  try {
    const blob = await runTask({ type: 'image', file: state.file, range: item.range }, controller.signal);
    const bitmap = await createImageBitmap(blob);
    if (controller.signal.aborted) { bitmap.close(); return; }
    state.optical = bitmap; viewer.setSource(bitmap, true); busy(); controls(); status(`${item.label} · ${bitmap.width} × ${bitmap.height} px`);
  } catch (error) { if (!controller.signal.aborted) { busy(); errorMessage(error); } }
}
function inspectPixel(x, y) {
  const raster = state.raster;
  if (!raster || state.mode !== 'channels' || x < 0 || y < 0 || x >= raster.width || y >= raster.height) return;
  const i = y * raster.width + x;
  $('pixel-coordinate').textContent = `x ${x} · y ${y}`;
  $('pixel-values').replaceChildren(...state.slots.map(s => {
    const channel = state.acquisition.channels.find(c => c.index === s.index);
    const value = raster.planes.find(p => p.index === s.index)?.data[i];
    const span = node('span', channel.label);
    span.append(node('b', Number.isFinite(value) ? Number(value.toPrecision(6)).toLocaleString(undefined, { maximumSignificantDigits: 6 }) : 'Not acquired / missing')); return span;
  }));
  $('pixel-panel').hidden = false;
}
async function drawMap() {
  const epoch = ++state.mapEpoch, a = state.acquisition, file = state.file;
  if (!a) return;
  const p = state.model.panoramas.find(p => p.id === a.panoramaId);
  const panoramaRange = p && imageRange(p.metadata);
  const slideImages = slideImageSources(state.model, a.slideId);
  const choices = [
    { id: 'panorama', name: 'Panorama image', available: !!panoramaRange },
    ...(slideImages.length ? slideImages.map(s => ({ id: s.id, name: s.label, available: true })) : [{ id: 'slide', name: 'Whole-slide photo', available: false }]),
    { id: 'none', name: 'Coordinates only', available: true },
  ];
  $('overview-backdrop').replaceChildren(...choices.map(c => {
    const item = option(`${c.name}${c.available ? '' : ' (not present)'}`, c.id); item.disabled = !c.available; return item;
  }));
  const selected = choices.find(c => c.id === state.backdrop && c.available)?.id || choices.find(c => c.available).id;
  $('overview-backdrop').value = selected;
  const slideImage = slideImages.find(s => s.id === selected);
  const list = state.model.acquisitions.filter(item => item.slideId === a.slideId && (slideImage || item.panoramaId === a.panoramaId) && item.points.length >= 3 && item.points.every(point => point.every(Number.isFinite)));
  $('map').replaceChildren(); $('overview').hidden = !list.length && !panoramaRange && !slideImages.length;
  if ($('overview').hidden) return;
  let { width, height, project } = coordinateOverview(list);
  let note = '', photo = false;
  const ns = 'http://www.w3.org/2000/svg';
  const range = slideImage?.range || (selected === 'panorama' ? panoramaRange : null);
  if (range) {
    try {
      const key = `${range.start}:${range.end}`;
      let thumbnail = state.overviewImage;
      if (!thumbnail || thumbnail.key !== key) {
        const blob = await readOpticalImage(file, range);
        const bitmap = await createImageBitmap(blob);
        if (epoch !== state.mapEpoch) { bitmap.close(); return; }
        try {
          const canvas = document.createElement('canvas');
          const scale = Math.min(1, 4096 / Math.max(bitmap.width, bitmap.height));
          canvas.width = Math.max(1, Math.round(bitmap.width * scale)); canvas.height = Math.max(1, Math.round(bitmap.height * scale));
          canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
          thumbnail = { key, width: bitmap.width, height: bitmap.height, url: canvas.toDataURL() };
          state.overviewImage = thumbnail;
        } finally { bitmap.close(); }
      }
      if (epoch !== state.mapEpoch) return;
      width = thumbnail.width; height = thumbnail.height;
      project = slideImage ? (slideImage.projection === 'panorama' ? panoramaProjection : slideProjection)(slideImage.metadata, width, height) : panoramaProjection(p.metadata, width, height);
      const image = document.createElementNS(ns, 'image'); image.setAttribute('href', thumbnail.url); image.setAttribute('width', width); image.setAttribute('height', height); $('map').append(image);
      photo = true;
      if (!project) note = 'Photo shown without ROI outlines: positioning metadata is missing. Use Coordinates only to navigate.';
      else if (!list.length) note = 'This slide has no recorded ROI outlines. Use the acquisition list to navigate.';
    } catch (error) { note = `Photo could not be displayed. Showing coordinates instead. ${error.message}`; }
  }
  if (epoch !== state.mapEpoch) return;
  width = Math.max(1, width); height = Math.max(1, height);
  const pad = Math.max(width, height) * .055;
  const mapKey = `${state.fileEpoch}/${a.slideId}/${selected}/${range ? range.start : a.panoramaId}`;
  overviewNavigation.setBounds(mapKey, { x: -pad, y: -pad, width: width + pad * 2, height: height + pad * 2 });
  $('overview-title').textContent = photo ? (slideImage?.name || p.name) : 'Acquisition map';
  $('overview-action').textContent = project && list.length ? 'Click a region' : 'Photo overview';
  $('overview-note').textContent = note; $('overview-note').hidden = !note;
  if (!project) return;
  for (const item of list) {
    const polygon = document.createElementNS(ns, 'polygon'); const coords = item.points.map(project);
    polygon.setAttribute('points', coords.map(p => p.join(',')).join(' ')); polygon.classList.toggle('active', item.id === a.id);
    polygon.setAttribute('tabindex', '0'); polygon.setAttribute('role', 'button'); polygon.setAttribute('aria-label', `Open ${item.name}`);
    const open = () => { $('slide').value = '*'; $('search').value = ''; refreshList(); selectAcquisition(item.id); };
    polygon.onclick = open; polygon.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } };
    const title = document.createElementNS(ns, 'title'); title.textContent = item.name; polygon.append(title); $('map').append(polygon);
    const text = document.createElementNS(ns, 'text'); text.textContent = item.id;
    text.setAttribute('x', coords.reduce((sum, p) => sum + p[0], 0) / coords.length); text.setAttribute('y', coords.reduce((sum, p) => sum + p[1], 0) / coords.length); text.setAttribute('font-size', Math.max(width, height) / 24); $('map').append(text);
  }
  overviewNavigation.apply();
}

const chooseFile = () => $('file').click();
$('open-file').onclick = $('welcome-open').onclick = chooseFile;
$('file').onchange = () => openFile($('file').files[0]);
$('close-file').onclick = reset;
$('demo').onclick = () => openFile(createDemoFile());
$('dismiss-message').onclick = () => message('');
$('cancel').onclick = () => { if (!state.model) reset(); else { abortLoad(); busy(); message('Loading cancelled. Click Auto all to load this acquisition again, or select another region.'); status('Loading cancelled'); } };
$('acquisition').onchange = () => selectAcquisition($('acquisition').value);
$('slide').onchange = filterChanged; $('search').oninput = filterChanged;
$('overview-backdrop').onchange = () => { state.backdrop = $('overview-backdrop').value; drawMap(); };
for (const [id, direction] of [['previous', -1], ['next', 1]]) $(id).onclick = () => { const list = filteredAcquisitions(), index = list.findIndex(a => a.id === state.acquisition?.id); if (list[index + direction]) selectAcquisition(list[index + direction].id); };
$('add-channel').onclick = () => {
  const c = channelList(state.acquisition).find(c => !state.slots.some(s => s.index === c.index)); if (!c) return;
  state.slots.push({ index: c.index, key: c.key, colour: colours[state.slots.length], enabled: true, low: null, high: null });
  state.mode = 'channels'; $('image-mode').value = 'channels'; renderChannels(); loadChannels();
};
$('auto-all').onclick = () => { for (const s of state.slots) s.low = s.high = null; state.mode = 'channels'; $('image-mode').value = 'channels'; loadChannels(); };
$('transform').onchange = () => { $('cofactor-wrap').hidden = $('transform').value !== 'arcsinh'; render(); };
$('cofactor').onchange = () => { if (!(Number($('cofactor').value) > 0)) $('cofactor').value = 5; render(); };
$('channel-order').onchange = renderChannels;
$('map-zoom-in').onclick = () => overviewNavigation.zoom(1.4);
$('map-zoom-out').onclick = () => overviewNavigation.zoom(1 / 1.4);
$('map-fit').onclick = () => overviewNavigation.fit();
$('image-mode').onchange = imageModeChanged;
$('fit').onclick = () => viewer.fit(); $('one-to-one').onclick = () => viewer.oneToOne(); $('zoom-in').onclick = () => viewer.zoom(1.3); $('zoom-out').onclick = () => viewer.zoom(1 / 1.3);
$('close-pixel').onclick = () => { $('pixel-panel').hidden = true; };
let dragDepth = 0;
window.addEventListener('dragenter', e => { if ([...e.dataTransfer.types].includes('Files')) { e.preventDefault(); dragDepth++; $('drop-overlay').hidden = false; } });
window.addEventListener('dragover', e => { e.preventDefault(); });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('drop-overlay').hidden = true; } });
window.addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; $('drop-overlay').hidden = true; const files = [...e.dataTransfer.files]; if (files.length !== 1) message('Drop one MCD file at a time.'); else openFile(files[0]); });
window.addEventListener('keydown', e => { if (e.target.closest('input, select, textarea, button')) return; if (e.key.toLowerCase() === 'f') viewer.fit(); });
exportPanel = new ExportPanel(viewer, () => state, async snapshot => {
  const epoch = state.fileEpoch;
  if (!state.model?.acquisitions.some(a => a.id === snapshot.acquisitionId)) throw new Error('This bookmark belongs to another MCD file.');
  $('transform').value = snapshot.transform; $('cofactor').value = snapshot.cofactor; $('cofactor-wrap').hidden = snapshot.transform !== 'arcsinh';
  await selectAcquisition(snapshot.acquisitionId, snapshot.slots);
  if (state.fileEpoch !== epoch || state.acquisition?.id !== snapshot.acquisitionId || !state.raster) throw new Error('Bookmark restoration was interrupted.');
  if (snapshot.mode !== 'channels') { $('image-mode').value = snapshot.mode; await imageModeChanged(); }
});
controls();
