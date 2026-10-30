import { defaultFigure, defaultScaleBar, validateFigure, validateScaleBar, figureLayout, paneFurniture, scaleBarGeometry } from './figure.js';
import { figureFilename, validateFigureFilename } from './figure-filename.js';
import { createImageExport, exportGeometry } from './image-export.js';
import { colourTable, mapName } from './colormaps.js';

const $ = id => document.getElementById(id);
const clone = value => structuredClone(value);
const fieldValue = input => input.type === 'checkbox' ? input.checked : input.type === 'number' ? input.valueAsNumber : input.value;

export class ExportPanel {
  constructor(viewer, context, restore) {
    this.viewer = viewer; this.context = context; this.restore = restore;
    this.settings = defaultFigure(); this.bar = defaultScaleBar(); this.title = ''; this.bookmarks = []; this.epoch = 0;
    $('figure-root').innerHTML = `
      <div class="figure-mode" role="group" aria-label="Viewer or Figure mode"><button id="figure-viewer" aria-pressed="true">Viewer</button><button id="figure-mode" aria-pressed="false">Figure</button></div>
      <p id="figure-hint" class="muted">Switch to Figure to preview and export.</p>
      <p id="figure-calibration" class="muted">IMC calibration · 1 pixel = 1 µm</p>
      <fieldset><legend>Frame &amp; physical size</legend>
        <label>Aspect ratio<select id="figure-aspect"><option value="free">Available space</option><option>1:1</option><option>4:3</option><option>3:2</option><option>16:9</option></select></label>
        <label class="checkbox-label"><input id="figure-physicalLock" type="checkbox">Lock field of view in µm</label>
        <div class="two-columns"><label>Width (µm)<input id="figure-physicalWidth" type="number" min="0.1" max="10000000" step="0.1"></label><label>Height (µm)<input id="figure-physicalHeight" type="number" min="0.1" max="10000000" step="0.1"></label></div>
        <p class="muted">Pan to frame your image. Locked size replaces zoom and aspect. Viewer restores your previous framing.</p>
      </fieldset>
      <fieldset><legend>Labels &amp; colour scheme</legend>
        <label>Image title<input id="figure-title" type="text" maxlength="100" placeholder="Acquisition name"></label>
        <label>Colour scheme<select id="figure-theme"><option value="dark">White on black</option><option value="light">Black on white</option><option value="custom">Custom</option></select></label>
        <div class="two-columns figure-colours"><label>Background<input id="figure-background" type="color"></label><label>Text / scale bars<input id="figure-foreground" type="color"></label></div>
        <label class="checkbox-label"><input id="figure-showNames" type="checkbox">Image title</label>
        <label class="checkbox-label"><input id="figure-showLegends" type="checkbox">Channel legend</label>
        <label class="checkbox-label"><input id="figure-overlayBackground" type="checkbox">Background behind labels</label>
        <div class="two-columns"><label>Title text (px)<input id="figure-fontSize" type="number" min="8" max="96"></label><label>Legend text (px)<input id="figure-legendSize" type="number" min="8" max="72"></label></div>
        <label>Maximum legend entries<input id="figure-legendRows" type="number" min="1" max="100"></label>
      </fieldset>
      <fieldset><legend>Scale bar</legend>
        <label class="checkbox-label"><input id="figure-showScaleBars" type="checkbox">Show scale bar</label>
        <label class="checkbox-label"><input id="scale-automatic" type="checkbox">Automatic length</label>
        <div class="two-columns"><label>Length (µm)<input id="scale-length" type="number" min="0.1" max="10000000" step="0.1"></label><label>Line width (px)<input id="scale-thickness" type="number" min="1" max="20" step="0.1"></label></div>
        <label>Position<select id="scale-position"><option value="bottom-left">Bottom left</option><option value="bottom-right">Bottom right</option></select></label>
        <label>Export scale text (px)<input id="figure-scaleFontSize" type="number" min="8" max="96"></label>
        <p id="scale-note" class="muted">Length is measured in µm, independent of export resolution.</p>
      </fieldset>
      <fieldset><legend>Export</legend>
        <label>Image width (px)<input id="figure-paneWidth" type="number" min="128" max="8192"></label>
        <label>Format<select id="figure-format"><option value="png">PNG</option><option value="jpeg">JPEG</option><option value="svg">SVG · separate layers</option></select></label>
        <details class="export-filename"><summary>File name</summary>
          <label>Label / prefix<input id="figure-filename-label" type="text" maxlength="100"></label>
          <label class="checkbox-label"><input id="figure-filename-roi" type="checkbox">Acquisition name</label>
          <label class="checkbox-label"><input id="figure-filename-views" type="checkbox">Image title</label>
          <label class="checkbox-label"><input id="figure-filename-timestamp" type="checkbox">Timestamp (UTC)</label>
          <label>Acquisition metadata fields<select id="figure-filename-metadata" multiple size="4" aria-describedby="filename-meta-help"></select></label>
          <p id="filename-meta-help" class="muted">Ctrl / Cmd to choose multiple fields. Metadata comes from the open MCD.</p>
          <button id="figure-filename-clear" class="quiet">Clear metadata fields</button>
        </details>
        <output id="figure-filename-preview" aria-label="Filename preview"></output>
        <label class="checkbox-label"><input id="figure-autoBookmark" type="checkbox">Bookmark each export in this tab</label>
        <p id="figure-dimensions" class="muted"></p>
        <button id="figure-export" class="primary" disabled>Export picture</button>
        <a id="figure-download" class="export-download" hidden>Download prepared image</a>
        <p class="muted">SVG keeps the image raster and editable vector title, legend and scale bar in separate layers. Sizes are export pixels.</p>
      </fieldset>
      <details id="export-bookmarks"><summary>Export bookmarks <span id="export-bookmark-count">0</span></summary><p class="muted">Restore the acquisition, channels and framing. Bookmarks last until you close this file or reload the tab.</p><div id="export-bookmark-list"></div></details>
      <p id="figure-status" class="muted" role="status"></p>`;
    this.fields = Object.keys(this.settings).filter(key => key !== 'mode' && $(`figure-${key}`));
    for (const key of this.fields) {
      const input = $(`figure-${key}`);
      const change = () => {
        if (!input.reportValidity()) return;
        try { this.settings = validateFigure({ ...this.settings, [key]: fieldValue(input) }); this.sync(); this.layout(); }
        catch (error) { this.status(error.message); }
      };
      input.onchange = change; if (input.type === 'number') input.onblur = change;
    }
    for (const key of ['automatic', 'length', 'thickness', 'position']) {
      const input = $(`scale-${key}`), change = () => {
        if (!input.reportValidity()) return;
        try { this.bar = validateScaleBar({ ...this.bar, [key]: fieldValue(input) }); this.sync(); this.draw(); }
        catch (error) { this.status(error.message); }
      };
      input.onchange = change; if (input.type === 'number') input.onblur = change;
    }
    $('figure-title').oninput = () => { this.title = $('figure-title').value; this.sync(); this.draw(); };
    $('figure-theme').onchange = () => {
      if ($('figure-theme').value !== 'custom') {
        const light = $('figure-theme').value === 'light'; Object.assign(this.settings, { background: light ? '#ffffff' : '#000000', foreground: light ? '#000000' : '#ffffff' }); this.sync(); this.layout();
      }
    };
    for (const key of ['label', 'roi', 'views', 'timestamp']) $(`figure-filename-${key}`).oninput = event => {
      this.settings.filename = validateFigureFilename({ ...this.settings.filename, [key]: fieldValue(event.target) }); this.sync();
    };
    $('figure-filename-metadata').onchange = () => { this.settings.filename.metadata = [...$('figure-filename-metadata').selectedOptions].map(o => o.value); this.sync(); };
    $('figure-filename-clear').onclick = () => { this.settings.filename.metadata = []; this.sync(); };
    $('figure-viewer').onclick = () => this.mode('viewer'); $('figure-mode').onclick = () => this.mode('figure');
    $('figure-export').onclick = () => this.export();
    new ResizeObserver(() => this.layout()).observe($('viewport'));
    this.sync();
  }
  status(text) { $('figure-status').textContent = text; }
  name() { const c = this.context(); return this.title.trim() || (c.mode === 'channels' ? c.acquisition?.name : c.images?.find(i => i.id === c.mode)?.label) || 'Image'; }
  filenameContext() {
    const a = this.context().acquisition;
    return { roi: a?.name || '', viewNames: [this.name()], metadataLoaded: !!a, metadata: Object.entries(a?.metadata || {}).map(([name, value]) => ({ name, values: [[a.name, value]] })) };
  }
  groups() {
    const c = this.context(); if (c.mode !== 'channels' || !c.acquisition) return [];
    const groups = [], rows = [];
    for (const s of c.slots.filter(s => s.enabled)) {
      const name = c.acquisition.channels.find(channel => channel.index === s.index)?.label || s.key;
      if (!s.map || s.map === 'pure') rows.push({ name, colour: s.colour });
      else if (Number.isFinite(s.low) && Number.isFinite(s.high)) {
        const table = colourTable(s), gradient = [0, 32, 64, 96, 128, 160, 192, 224, 255].map(i => '#' + table[i].map(n => n.toString(16).padStart(2, '0')).join('')).join(',');
        groups.push({ title: `${name} · ${mapName(s.map)}`, range: [s.low, s.high], gradient });
      }
    }
    if (rows.length) groups.unshift({ title: 'Channels', rows }); return groups;
  }
  view() { return { name: this.name(), legendVisible: true, scaleBar: { ...this.bar, visible: this.settings.showScaleBars && this.context().mode === 'channels' } }; }
  mode(mode) {
    if (mode === this.settings.mode || this.busy || (mode === 'figure' && !this.viewer.source)) return;
    if (mode === 'figure') {
      this.returnCamera = this.viewer.camera();
      if (!this.initialisedSize) { this.settings.physicalWidth = this.viewer.source.width; this.settings.physicalHeight = this.viewer.source.height; this.initialisedSize = true; }
      if (this.context().mode !== 'channels') this.settings.physicalLock = false;
    }
    this.settings.mode = mode; this.layout();
    if (mode === 'viewer' && this.returnCamera) { this.viewer.restoreCamera(this.returnCamera); this.returnCamera = null; }
    this.sync(); this.draw();
  }
  sourceChanging(reset = false) {
    this.epoch++; this.busy = false;
    if (this.downloadURL) URL.revokeObjectURL(this.downloadURL);
    this.downloadURL = null; $('figure-download').hidden = true; $('figure-download').removeAttribute('href');
    if (this.settings.mode === 'figure') this.mode('viewer');
    this.returnCamera = null; this.status('');
    if (reset) { this.settings = defaultFigure(); this.bar = defaultScaleBar(); this.title = ''; this.bookmarks = []; this.initialisedSize = false; this.renderBookmarks(); }
  }
  sync() {
    const c = this.context(), ready = this.ready(), figure = this.settings.mode === 'figure', imc = c.mode === 'channels';
    for (const key of this.fields) {
      const input = $(`figure-${key}`);
      if (document.activeElement !== input) { if (input.type === 'checkbox') input.checked = this.settings[key]; else input.value = this.settings[key]; }
      input.disabled = !ready || this.busy || !figure;
    }
    $('figure-showScaleBars').disabled = !ready || this.busy || !imc;
    $('figure-aspect').disabled ||= this.settings.physicalLock;
    $('figure-physicalLock').disabled ||= !imc;
    for (const key of ['physicalWidth', 'physicalHeight']) $(`figure-${key}`).disabled ||= !this.settings.physicalLock || !imc;
    for (const key of ['automatic', 'length', 'thickness', 'position']) {
      const input = $(`scale-${key}`); if (document.activeElement !== input) { if (input.type === 'checkbox') input.checked = this.bar[key]; else input.value = this.bar[key]; }
      input.disabled = !ready || this.busy || !imc || !this.settings.showScaleBars || (key === 'length' && this.bar.automatic);
    }
    $('figure-calibration').textContent = imc ? 'IMC calibration · 1 pixel = 1 µm' : 'Optical photograph · no physical scale calibration';
    $('figure-theme').value = this.settings.background === '#000000' && this.settings.foreground === '#ffffff' ? 'dark' : this.settings.background === '#ffffff' && this.settings.foreground === '#000000' ? 'light' : 'custom';
    $('figure-title').disabled = $('figure-theme').disabled = !ready || !figure || this.busy;
    if (document.activeElement !== $('figure-title')) $('figure-title').value = this.title;
    for (const mode of ['viewer', 'figure']) { const button = $(mode === 'viewer' ? 'figure-viewer' : 'figure-mode'); button.disabled = !ready || this.busy; button.setAttribute('aria-pressed', String(this.settings.mode === mode)); }
    $('figure-hint').textContent = figure ? 'Figure preview · pan to frame, then export.' : 'Switch to Figure to preview and export.';
    const filename = this.settings.filename;
    for (const key of ['label', 'roi', 'views', 'timestamp']) {
      const input = $(`figure-filename-${key}`); input.disabled = !ready || !figure || this.busy;
      if (document.activeElement !== input) { if (input.type === 'checkbox') input.checked = filename[key]; else input.value = filename[key]; }
    }
    const names = Object.keys(c.acquisition?.metadata || {}), metadataKey = JSON.stringify([names, filename.metadata]);
    if (metadataKey !== this.metadataKey) {
      this.metadataKey = metadataKey;
      $('figure-filename-metadata').replaceChildren(...[...new Set([...names, ...filename.metadata])].sort().map(name => new Option(names.includes(name) ? name : `${name} (unavailable)`, name, false, filename.metadata.includes(name))));
    }
    $('figure-filename-metadata').disabled = $('figure-filename-clear').disabled = !ready || !figure || this.busy;
    let valid = true;
    try { $('figure-filename-preview').textContent = figureFilename(this.settings, this.filenameContext()); }
    catch (error) { valid = false; $('figure-filename-preview').textContent = error.message; }
    this.filenameValid = valid;
    $('figure-export').disabled = !ready || !figure || this.busy || !valid;
    $('viewport').classList.toggle('figure-mode-active', figure);
    for (const id of ['zoom-in', 'zoom-out', 'one-to-one', 'fit']) $(id).disabled = figure && this.settings.physicalLock;
    this.draw();
  }
  layout() {
    const figure = this.settings.mode === 'figure', viewer = this.viewer, stage = $('image-stage');
    viewer.figure = figure; viewer.physicalWidth = figure && this.settings.physicalLock && this.context().mode === 'channels' ? this.settings.physicalWidth : null;
    viewer.background = this.settings.background;
    if (figure) {
      const rect = $('viewport').getBoundingClientRect(), l = figureLayout(1, Math.max(1, rect.width - 32), Math.max(1, rect.height - 80), this.settings);
      stage.style.cssText = `inset:auto;left:${(rect.width - l.width) / 2}px;top:${(rect.height - 56 - l.height) / 2}px;width:${l.width}px;height:${l.height}px`;
    } else stage.style.cssText = '';
    viewer.resize();
  }
  ready() {
    const c = this.context();
    return !!this.viewer.source && !c.renderPending && (c.mode !== 'channels' || (c.raster && c.slots.every(s => Number.isFinite(s.low) && Number.isFinite(s.high) && c.raster.planes.some(p => p.index === s.index))));
  }
  draw() {
    const viewer = this.viewer, overlay = $('figure-overlay');
    if (!viewer.source || !this.context().acquisition || !(viewer.width > 0)) { overlay.replaceChildren(); $('figure-dimensions').textContent = ''; $('figure-export').disabled = true; return; }
    const figure = this.settings.mode === 'figure', view = this.view();
    overlay.setAttribute('viewBox', `0 0 ${viewer.width} ${viewer.height}`);
    overlay.innerHTML = paneFurniture(view, this.groups(), this.settings, viewer.width, figure ? viewer.height : Math.max(1, viewer.height - 60), viewer.scale, 'preview');
    const calibrated = this.context().mode === 'channels';
    try {
      const g = exportGeometry(this.settings.paneWidth, viewer, viewer);
      $('figure-dimensions').textContent = `${g.width} × ${g.height} px export · ${(viewer.width / viewer.scale).toFixed(1)} × ${(viewer.height / viewer.scale).toFixed(1)} ${calibrated ? 'µm' : 'image pixels'} field of view`;
      const measure = scaleBarGeometry(this.bar, 1, g.zoom, g.width);
      const fits = !calibrated || !this.settings.showScaleBars || measure.fits;
      $('scale-note').textContent = fits ? 'Length is measured in µm, independent of export resolution.' : 'Scale bar is too long for this frame. Shorten it or use Automatic length.';
      $('figure-export').disabled = !this.ready() || !figure || this.busy || !this.filenameValid || !fits;
    } catch (error) { $('figure-dimensions').textContent = error.message; $('figure-export').disabled = true; }
  }
  snapshot() {
    const c = this.context();
    return { acquisitionId: c.acquisition.id, mode: c.mode, slots: clone(c.slots), transform: $('transform').value, cofactor: Number($('cofactor').value),
      settings: clone(this.settings), bar: clone(this.bar), title: this.title, camera: { ...this.viewer.camera(), fieldWidth: this.viewer.width / this.viewer.scale } };
  }
  async export() {
    if ($('figure-export').disabled) return;
    const c = this.context(), epoch = ++this.epoch, snapshot = this.snapshot(), filename = figureFilename(this.settings, this.filenameContext());
    if (this.settings.autoBookmark && this.bookmarks.length >= 100) { this.status('Export bookmark limit reached. Turn off automatic bookmarking to continue.'); return; }
    this.busy = true; this.sync(); this.status('Preparing export…');
    try {
      const blob = await createImageExport({ source: this.viewer.source, rect: { width: this.viewer.width, height: this.viewer.height }, camera: { x: this.viewer.x, y: this.viewer.y, scale: this.viewer.scale },
        settings: snapshot.settings, view: this.view(), groups: this.groups(), calibrated: c.mode === 'channels',
        metadata: { title: this.name(), file: c.file.name, acquisition: c.acquisition.id, source: c.mode, channels: snapshot.slots, transform: snapshot.transform, cofactor: snapshot.cofactor } }, () => epoch === this.epoch);
      if (epoch !== this.epoch) return;
      if (this.downloadURL) URL.revokeObjectURL(this.downloadURL);
      const link = $('figure-download'); this.downloadURL = URL.createObjectURL(blob);
      link.href = this.downloadURL; link.download = filename; link.textContent = `Download ${filename}`; link.hidden = false; link.click();
      if (snapshot.settings.autoBookmark) { this.bookmarks.unshift({ filename, snapshot }); this.renderBookmarks(); }
      this.status(`Ready: ${filename}${snapshot.settings.autoBookmark ? ' · bookmark saved' : ''}.`);
    } catch (error) { if (epoch === this.epoch) this.status(error.message); }
    finally { if (epoch === this.epoch) { this.busy = false; this.sync(); } }
  }
  renderBookmarks() {
    $('export-bookmark-count').textContent = this.bookmarks.length;
    $('export-bookmark-list').replaceChildren(...this.bookmarks.map(bookmark => {
      const button = document.createElement('button'); button.className = 'quiet'; button.textContent = `Restore ${bookmark.filename}`;
      button.onclick = async () => {
        if (this.busy) return;
        const fileEpoch = this.context().fileEpoch;
        this.sourceChanging(); this.status('Restoring exported view…');
        try {
          await this.restore(clone(bookmark.snapshot));
          if (fileEpoch !== this.context().fileEpoch) return;
          const s = bookmark.snapshot;
          this.settings = clone(s.settings); this.bar = clone(s.bar); this.title = s.title; this.returnCamera = this.viewer.camera();
          this.layout(); this.viewer.restoreCamera({ ...s.camera, scale: this.viewer.width / (this.settings.physicalLock ? this.settings.physicalWidth : s.camera.fieldWidth) }); this.sync(); this.status(`Restored ${bookmark.filename}.`);
        } catch (error) { this.status(error.message); }
      }; return button;
    }));
  }
}
