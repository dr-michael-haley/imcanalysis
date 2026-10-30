import { validateFigure, validateScaleBar, figureLayout, paneFurniture } from './figure.js';
import { legendGroups } from './view-legend.js';
import { figureFilename, validateFigureFilename } from './figure-filename.js';

export function createFigurePanel(state, $, actions) {
  $('figure-root').innerHTML = `<div class="figure-mode" role="group" aria-label="Viewer or Figure mode"><button id="figure-viewer">Viewer</button><button id="figure-mode">Figure</button><span id="figure-mode-hint">Switch to Figure to compose and export.</span></div>
    <div class="figure-settings">
      <fieldset><legend>Frame &amp; real size</legend><label>Aspect per pane<select id="figure-aspect"><option value="free">Available space</option><option>1:1</option><option>4:3</option><option>3:2</option><option>16:9</option></select></label><label class="check-label"><input id="figure-physicalLock" type="checkbox">Lock field of view in µm</label><div class="two-columns"><label>Width (µm)<input id="figure-physicalWidth" type="number" min="0.001" max="10000000" step="any"></label><label>Height (µm)<input id="figure-physicalHeight" type="number" min="0.001" max="10000000" step="any"></label></div><p class="muted">Pan remains available. Locked size overrides aspect and zoom. Leaving Figure restores the viewer camera and layout.</p></fieldset>
      <fieldset><legend>Figure overlays</legend><label>Colour scheme<select id="figure-theme"><option value="dark">White on black</option><option value="light">Black on white</option><option value="custom">Custom</option></select></label><div class="figure-colours"><label>Background<input id="figure-background" type="color"></label><label>Text / scale bars<input id="figure-foreground" type="color"></label></div><label class="check-label"><input id="figure-showNames" type="checkbox">View names</label><label class="check-label"><input id="figure-showLegends" type="checkbox">Legends for enabled views</label><label class="check-label"><input id="figure-showScaleBars" type="checkbox">Scale bars</label><label class="check-label"><input id="figure-overlayBackground" type="checkbox">Background behind labels</label><div class="two-columns"><label>Name / scale text (px)<input id="figure-fontSize" type="number" min="8" max="96"></label><label>Legend text (px)<input id="figure-legendSize" type="number" min="8" max="72"></label></div><label>Maximum legend entries<input id="figure-legendRows" type="number" min="1" max="100"></label></fieldset>
      <fieldset><legend>Export</legend><div class="two-columns"><label>Pane width (px)<input id="figure-paneWidth" type="number" min="128" max="8192"></label><label>Pane spacing (px)<input id="figure-gap" type="number" min="0" max="500"></label></div><label>Format<select id="figure-format"><option value="png">PNG</option><option value="jpeg">JPEG</option><option value="svg">SVG · separate layers</option></select></label><label class="check-label"><input id="figure-autoBookmark" type="checkbox">Automatically bookmark each export</label><p id="figure-dimensions" class="muted"></p><button id="figure-export" class="primary-action">Export picture</button><p class="muted">SVG retains image, annotation and cell layers as embedded rasters, with editable vector text and scale bars. Additive annotation layers use plus-lighter blending; PNG/JPEG preserve the composed pixels. Sizes are export pixels.</p></fieldset>
    </div><p id="figure-status" role="status" class="muted"></p>`;
  $('figure-fontSize').closest('label').firstChild.textContent = 'View name text (px)';
  $('figure-fontSize').closest('label').insertAdjacentHTML('afterend', '<label>Scale-bar text (px)<input id="figure-scaleFontSize" type="number" min="8" max="96"></label>');
  $('figure-background').closest('.figure-colours').after($('figure-fontSize').closest('.two-columns'));
  $('figure-showScaleBars').closest('label').insertAdjacentHTML('afterend', `<details class="figure-scale-options" open><summary>Scale bars · choose panes</summary><div class="action-row"><button id="figure-scales-all">All panes</button><button id="figure-scales-none">No panes</button></div><div id="figure-scale-panes"></div><p class="muted">Length is in µm; line width is in export pixels. Pane choices, length and width also appear in the View menu.</p></details>`);
  $('figure-format').closest('label').insertAdjacentHTML('afterend', `<details class="figure-filename-options" open><summary>File name</summary><label>Label / prefix<input id="figure-filename-label" type="text" maxlength="100"></label><div class="two-columns"><label class="check-label"><input id="figure-filename-roi" type="checkbox">ROI name</label><label class="check-label"><input id="figure-filename-views" type="checkbox">View names</label><label class="check-label"><input id="figure-filename-timestamp" type="checkbox">Timestamp (UTC)</label></div><button id="figure-filename-read">Read ROI metadata</button><p id="figure-filename-metadata-status" role="status" class="muted"></p><label>Metadata fields (Ctrl / Cmd for several)<select id="figure-filename-metadata" multiple size="3"></select></label><button id="figure-filename-clear" class="quiet">Clear metadata fields</button><label>Filename preview<output id="figure-filename-preview"></output></label><p id="figure-filename-error" role="status" class="muted"></p></details>`);
  const fields = Object.keys(state.figure).filter(k => !['mode', 'umPerPixel', 'filename'].includes(k));
  let returnCamera = null, busy = false, readingMetadata = false, scaleRows = [], metadataKey = '';

  function syncScalePanes() {
    if (scaleRows.length !== state.views.length || scaleRows.some((row, i) => row.view !== state.views[i])) {
      $('figure-scale-panes').replaceChildren();
      scaleRows = state.views.map((view, i) => {
        const card = document.createElement('div'); card.className = 'figure-scale-pane';
        card.innerHTML = `<label class="check-label"><input type="checkbox" data-bar="visible"><span></span></label><label class="check-label"><input type="checkbox" data-bar="automatic">Automatic length</label><div class="two-columns"><label>Length (µm)<input type="number" min="0.001" max="10000000" step="any" data-bar="length"></label><label>Line width (px)<input type="number" min="1" max="20" step="any" data-bar="thickness"></label></div>`;
        const inputs = [...card.querySelectorAll('input')];
        for (const input of inputs) {
          input.onchange = () => {
            if (!input.reportValidity()) return;
            try {
              const key = input.dataset.bar, value = input.type === 'checkbox' ? input.checked : input.valueAsNumber;
              if (view.scaleBar[key] === value) return;
              view.scaleBar = validateScaleBar({ ...view.scaleBar, [key]: value }); sync(); actions.draw(); actions.changed();
            } catch (error) { $('figure-status').textContent = error.message; }
          };
          if (input.type === 'number') input.onblur = input.onchange;
        }
        $('figure-scale-panes').append(card); return { view, card, inputs, index: i };
      });
    }
    for (const { view, card, inputs, index } of scaleRows) {
      const name = `Pane ${index + 1} · ${view.name}`; card.querySelector('span').textContent = name;
      for (const input of inputs) {
        const key = input.dataset.bar;
        input.setAttribute('aria-label', `${name} · ${key === 'visible' ? 'Show scale bar' : key === 'automatic' ? 'Automatic length' : key === 'length' ? 'Scale length (µm)' : 'Scale line width (px)'}`);
        if (input.type === 'checkbox') input.checked = view.scaleBar[key]; else if (document.activeElement !== input) input.value = view.scaleBar[key];
        input.disabled = state.figure.mode !== 'figure' || busy || !state.figure.showScaleBars || (key !== 'visible' && !view.scaleBar.visible) || (key === 'length' && view.scaleBar.automatic);
      }
    }
    for (const id of ['figure-scales-all', 'figure-scales-none']) $(id).disabled = state.figure.mode !== 'figure' || busy || !state.figure.showScaleBars;
  }
  for (const [id, visible] of [['figure-scales-all', true], ['figure-scales-none', false]]) $(id).onclick = () => {
    for (const view of state.views) view.scaleBar.visible = visible;
    sync(); actions.draw(); actions.changed();
  };
  function syncFilename() {
    const s = state.figure.filename, context = actions.filenameContext(), disabled = state.figure.mode !== 'figure' || busy;
    for (const key of ['label', 'roi', 'views', 'timestamp']) {
      const input = $(`figure-filename-${key}`); input.disabled = disabled;
      if (document.activeElement !== input) { if (input.type === 'checkbox') input.checked = s[key]; else input.value = s[key]; }
    }
    const names = context.metadata.map(field => field.name), key = JSON.stringify([names, s.metadata.filter(name => !names.includes(name)), context.metadataLoaded]);
    if (key !== metadataKey) {
      metadataKey = key; $('figure-filename-metadata').replaceChildren();
      for (const name of [...new Set([...names, ...s.metadata])]) {
        const option = document.createElement('option'); option.value = name; option.textContent = names.includes(name) ? name : `${name} · ${context.metadataLoaded ? 'unavailable' : 'read metadata'}`; option.selected = s.metadata.includes(name); $('figure-filename-metadata').append(option);
      }
    }
    for (const option of $('figure-filename-metadata').options) option.selected = s.metadata.includes(option.value);
    $('figure-filename-metadata').disabled = disabled || readingMetadata || (!names.length && !s.metadata.length);
    $('figure-filename-clear').disabled = disabled || !s.metadata.length;
    $('figure-filename-read').disabled = disabled || readingMetadata || !state.ready || context.metadataLoaded;
    $('figure-filename-read').textContent = readingMetadata ? 'Reading metadata…' : context.metadataLoaded ? 'Metadata loaded' : 'Read ROI metadata';
    $('figure-filename-metadata-status').textContent = readingMetadata ? 'Checking observations for one value per ROI. This may take time; you can cancel in ROIs → Filtering.' : context.metadataLoaded ? `${names.length} ROI metadata fields available.` : 'Optional. Reads metadata only when requested; no scan runs on reload.';
    try { $('figure-filename-preview').textContent = figureFilename(state.figure, context); $('figure-filename-error').textContent = ''; return true; }
    catch (error) { $('figure-filename-preview').textContent = 'Waiting for metadata'; $('figure-filename-error').textContent = error.message; return false; }
  }
  function saveFilename(value) {
    try { state.figure.filename = validateFigureFilename(value); sync(); actions.changed(); }
    catch (error) { $('figure-filename-error').textContent = error.message; }
  }
  for (const key of ['label', 'roi', 'views', 'timestamp']) $(`figure-filename-${key}`).oninput = event => saveFilename({ ...state.figure.filename, [key]: event.target.type === 'checkbox' ? event.target.checked : event.target.value });
  $('figure-filename-metadata').onchange = () => saveFilename({ ...state.figure.filename, metadata: [...$('figure-filename-metadata').selectedOptions].map(option => option.value) });
  $('figure-filename-clear').onclick = () => saveFilename({ ...state.figure.filename, metadata: [] });
  $('figure-filename-read').onclick = async () => {
    readingMetadata = true; sync();
    try { await actions.readMetadata(); }
    catch (error) { $('figure-status').textContent = error.message; }
    finally { readingMetadata = false; sync(); }
  };
  function mode(value) {
    if (value === state.figure.mode) return;
    if (value === 'figure') returnCamera = actions.camera();
    state.figure.mode = value; actions.modeChanged(); sync(); actions.layout();
    if (value === 'viewer' && returnCamera) { actions.restoreCamera(returnCamera); returnCamera = null; }
    actions.changed();
  }
  $('figure-viewer').onclick = () => mode('viewer'); $('figure-mode').onclick = () => mode('figure');
  $('quick-export').onclick = () => actions.open();
  for (const key of fields) $(`figure-${key}`).onchange = event => {
    const node = event.target; if (!node.reportValidity()) return;
    try {
      state.figure = validateFigure({ ...state.figure, [key]: node.type === 'checkbox' ? node.checked : node.type === 'number' ? node.valueAsNumber : node.value });
      sync(); actions.layout(); actions.changed();
    } catch (error) { $('figure-status').textContent = error.message; }
  };
  for (const key of fields) if ($(`figure-${key}`).type === 'number') $(`figure-${key}`).onblur = $(`figure-${key}`).onchange;
  $('figure-theme').onchange = () => {
    if ($('figure-theme').value === 'custom') return;
    const light = $('figure-theme').value === 'light'; Object.assign(state.figure, { background: light ? '#ffffff' : '#000000', foreground: light ? '#000000' : '#ffffff' }); sync(); actions.layout(); actions.changed();
  };
  for (const [id, key] of [['scale-visible', 'visible'], ['scale-auto', 'automatic'], ['scale-length', 'length'], ['scale-thickness', 'thickness'], ['scale-font', 'fontSize'], ['scale-colour', 'colour'], ['scale-position', 'position']]) $(id).onchange = event => {
    const node = event.target; if (!node.reportValidity()) return;
    try { const view = state.views[state.selectedView]; view.scaleBar = validateScaleBar({ ...view.scaleBar, [key]: node.type === 'checkbox' ? node.checked : node.type === 'number' ? node.valueAsNumber : node.value }); sync(); actions.draw(); actions.changed(); } catch (error) { $('figure-status').textContent = error.message; }
  };
  $('scale-calibration').onchange = () => {
    if (!$('scale-calibration').reportValidity()) return;
    state.figure = validateFigure({ ...state.figure, umPerPixel: $('scale-calibration').valueAsNumber }); actions.layout(); actions.changed();
  };
  for (const id of ['scale-calibration', 'scale-length', 'scale-thickness', 'scale-font']) $(id).onblur = $(id).onchange;
  $('figure-export').onclick = async () => {
    if (busy || !state.ready) return; busy = true; sync(); $('figure-status').textContent = 'Preparing export…';
    try { const filename = await actions.export(message => { $('figure-status').textContent = message; }); $('figure-status').textContent = `Exported ${filename}${state.figure.autoBookmark ? ' · bookmark saved' : ''}.`; }
    catch (error) { $('figure-status').textContent = error.message; }
    finally { busy = false; sync(); }
  };
  function sync() {
    const s = state.figure, figure = s.mode === 'figure', bar = state.views[state.selectedView].scaleBar;
    $('figure-viewer').setAttribute('aria-pressed', String(!figure)); $('figure-mode').setAttribute('aria-pressed', String(figure));
    $('figure-mode-hint').textContent = figure ? 'Figure preview · panes and overlays only. Return to Viewer to select cells.' : 'Switch to Figure to compose and export.';
    for (const key of fields) { const input = $(`figure-${key}`); if (document.activeElement !== input) { if (input.type === 'checkbox') input.checked = s[key]; else input.value = s[key]; } input.disabled = !figure || busy; }
    $('figure-aspect').disabled ||= s.physicalLock;
    for (const key of ['physicalWidth', 'physicalHeight']) $(`figure-${key}`).disabled ||= !s.physicalLock;
    $('figure-theme').value = s.background === '#000000' && s.foreground === '#ffffff' ? 'dark' : s.background === '#ffffff' && s.foreground === '#000000' ? 'light' : 'custom';
    $('figure-theme').disabled = !figure || busy;
    const filenameReady = syncFilename(); syncScalePanes();
    $('figure-export').disabled = !figure || !state.ready || busy || readingMetadata || !filenameReady; $('quick-export').disabled = !state.ready;
    $('figure-viewer').disabled = $('figure-mode').disabled = busy;
    for (const [id, key] of [['scale-visible', 'visible'], ['scale-auto', 'automatic'], ['scale-length', 'length'], ['scale-thickness', 'thickness'], ['scale-font', 'fontSize'], ['scale-colour', 'colour'], ['scale-position', 'position']]) { const input = $(id); if (input.type === 'checkbox') input.checked = bar[key]; else if (document.activeElement !== input) input.value = bar[key]; }
    $('scale-length').disabled = bar.automatic;
    if (document.activeElement !== $('scale-calibration')) $('scale-calibration').value = s.umPerPixel;
  }
  function layout() {
    const grid = $('pane-grid'); if (!grid) return;
    const figure = state.figure.mode === 'figure'; $('viewport').classList.toggle('figure-mode-active', figure);
    grid.style.cssText = '';
    if (figure) {
      const rect = $('viewport').getBoundingClientRect(), l = figureLayout(state.views.length, rect.width, rect.height, state.figure);
      Object.assign(grid.style, { inset: 'auto', width: `${l.width}px`, height: `${l.height}px`, left: `${(rect.width - l.width) / 2}px`, top: `${(rect.height - l.height) / 2}px`, gap: `${l.gap}px`, background: state.figure.background });
    }
  }
  function draw() {
    const s = state.figure;
    for (const [i, view] of state.views.entries()) {
      if (!view.furniture) continue; const rect = view.canvas.getBoundingClientRect();
      view.furniture.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
      view.furniture.innerHTML = state.ready ? paneFurniture(view, legendGroups(view, state.layers, state), s, rect.width, rect.height, state.scale, `preview-${i}`) : '';
    }
    if (state.ready && state.views[0].canvas) {
      const rect = state.views[0].canvas.getBoundingClientRect(), l = figureLayout(state.views.length, rect.width * gridShapeWidth(), rect.height * (state.views.length === 4 ? 2 : 1), { ...s, aspect: 'free', physicalLock: false });
      $('figure-dimensions').textContent = `${(rect.width / state.scale * s.umPerPixel).toFixed(1)} × ${(rect.height / state.scale * s.umPerPixel).toFixed(1)} µm per pane · ${Math.round(l.outputWidth)} × ${Math.round(l.outputHeight)} px export`;
    }
  }
  const gridShapeWidth = () => state.views.length === 4 ? 2 : state.views.length;
  sync(); return { sync, layout, draw, resetReturnCamera() { returnCamera = null; } };
}
