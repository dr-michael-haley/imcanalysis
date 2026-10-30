import { LAYER_TYPES, MAX_LAYERS, layerFile, indexLayer, defaultLayer, parseLabelMapping, labelColour, labelDictionary } from './aligned-layers.js';
import { COLOURMAPS } from './colourmaps.js';
import { contrast } from './core.js';
import { bindFolderPicker, bindFilePicker } from './folder-picker.js';

export function createAlignedPanel(state, $, { refresh, compose, sourcesChanged, showChannels, updateLegends }) {
  const setup = $('aligned-sources'), controls = $('aligned-controls');
  const node = (tag, text, className) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; };
  const message = text => { $('aligned-error').textContent = text; $('aligned-error').hidden = !text; };
  const labelled = (parent, text, input) => { const label = node('label', text); input.setAttribute('aria-label', text); label.append(input); parent.append(label); return input; };
  const select = (items, value) => { const el = node('select'); for (const [key, text] of items) { const option = node('option', text); option.value = key; el.append(option); } el.value = value; return el; };
  const check = (parent, text, checked, action) => {
    const label = node('label', undefined, 'check-label element-legend-control'), input = node('input'); input.type = 'checkbox'; input.checked = checked; input.setAttribute('aria-label', text);
    input.onchange = () => action(input.checked); label.append(input, node('span', text)); parent.append(label);
  };
  const showFlattenOption = () => { $('aligned-flatten-option').hidden = $('aligned-kind').value !== 'labels'; };
  $('aligned-kind').addEventListener('change', showFlattenOption); showFlattenOption();
  function index() {
    const maps = state.layers.map(source => indexLayer(source.files, state.rois.map(roi => roi.name), source.kind));
    state.layers.forEach((source, i) => { source.rois = maps[i]; });
    sync();
  }
  bindFolderPicker($('aligned-files'), file => layerFile(file, $('aligned-kind').value), async files => {
    const kind = $('aligned-kind').value, dataset = state.dataset, flatten = kind === 'labels' && $('aligned-flatten').checked;
    if (!files.length) return;
    $('aligned-add').classList.add('is-loading'); $('aligned-files').disabled = true; message('');
    try {
      if (state.layers.length >= MAX_LAYERS) throw new Error(`Use at most ${MAX_LAYERS} aligned layers.`);
      const selected = files.filter(file => layerFile(file, kind));
      if (files.workspaceHandle) selected.workspaceHandle = files.workspaceHandle;
      if (files.workspacePath) selected.workspacePath = files.workspacePath;
      if (!selected.some(file => /\.(tiff?|png|jpe?g|bmp|gif|webp)$/i.test(file.name))) throw new Error('No supported images found in that folder.');
      const mappings = selected.filter(file => file.name.toLowerCase() === 'label_mapping.csv');
      if (mappings.length > 1) throw new Error('Choose one annotation set folder, containing at most one label_mapping.csv.');
      if (mappings[0]?.size > 2 * 1024 * 1024) throw new Error('Choose a label_mapping.csv smaller than 2 MB.');
      const mapping = mappings.length ? parseLabelMapping(await mappings[0].text()) : [];
      if (dataset !== state.dataset) return;
      const rois = indexLayer(selected, state.rois.map(roi => roi.name), kind);
      if (state.rois.length && !rois.size) throw new Error(kind === 'imc' ? 'No ROI folders match your AnnData. Use the same ROI folder names as the original Images folder.' : 'No filenames match your ROI names. Use ROI_name.tiff (or .png / .jpg for colour images).');
      const source = { id: crypto.randomUUID(), name: ($('aligned-name').value.trim() || files[0].webkitRelativePath.split('/')[0] || LAYER_TYPES[kind]).slice(0, 80), kind, files: selected, mapping, rois };
      state.layers.push(source);
      if (kind !== 'imc') for (const view of state.views) view.layers.push({ ...defaultLayer(source.id, kind), flatten });
      $('aligned-name').value = ''; sync(); await sourcesChanged();
    } catch (error) { message(error.message); }
    finally { $('aligned-files').value = ''; $('aligned-files').disabled = false; $('aligned-add').classList.remove('is-loading'); }
  }, error => message(error.message));
  function sync() {
    setup.replaceChildren(); controls.replaceChildren();
    $('section-aligned-view').hidden = !state.layers.some(source => source.kind !== 'imc');
    $('aligned-empty').hidden = Boolean(state.layers.length);
    for (const [index, source] of state.layers.entries()) {
      const row = node('div', undefined, 'aligned-source');
      const name = labelled(row, 'Layer name', node('input')); name.value = source.name; name.maxLength = 80;
      name.addEventListener('input', () => { if (name.value.trim()) source.name = name.value.trim(); });
      row.append(node('p', `${LAYER_TYPES[source.kind]} · ${source.rois?.size ?? 0} of ${state.rois.length} ROIs`, 'muted'));
      if (source.kind === 'imc') {
        const count = new Set([...(source.rois?.values() ?? [])].flatMap(images => images.map(item => item.name))).size;
        row.append(node('p', `${count} channels available in Views → 01 Image channels. AnnData marker names are not required.`, 'muted'));
        const choose = node('button', 'Choose image channels', 'quiet'); choose.onclick = showChannels; row.append(choose);
      }
      if (source.kind === 'labels') {
        const picker = node('label', source.dictionary ? source.dictionary.file.name : 'Load dictionary CSV (optional)', 'file-picker');
        const input = node('input'); input.type = 'file'; input.accept = '.csv'; input.setAttribute('aria-label', `Load ${source.name} dictionary CSV`); picker.append(input); row.append(picker);
        bindFilePicker(input, '.csv', `Load ${source.name} dictionary CSV`, async file => {
          const dataset = state.dataset;
          try {
            if (file.size > 2 * 1024 * 1024) throw new Error('Choose a dictionary CSV smaller than 2 MB.');
            const text = await file.text(), parsed = labelDictionary(text);
            if (dataset !== state.dataset || !state.layers.includes(source)) return;
            source.mapping = parsed.mapping; source.dictionary = { file, columns: parsed.columns, headers: parsed.headers }; source.dictionaryText = text;
            message(''); sync(); await refresh();
          } catch (error) { message(error.message); }
        }, error => message(error.message));
        if (source.dictionary) {
          const inputs = {};
          for (const [key, text] of [['value', 'Label IDs'], ['label', 'Label names'], ['color', 'Colours']]) {
            inputs[key] = labelled(row, `${source.name} ${text}`, select([...(key === 'color' ? [['', 'Automatic colours']] : []), ...source.dictionary.headers.map(name => [name, name])], source.dictionary.columns[key]));
          }
          for (const input of Object.values(inputs)) input.onchange = () => {
            try {
              const columns = Object.fromEntries(Object.entries(inputs).map(([key, input]) => [key, input.value]));
              const parsed = labelDictionary(source.dictionaryText, columns); source.mapping = parsed.mapping; source.dictionary.columns = parsed.columns; message(''); refresh();
            } catch (error) { message(error.message); sync(); }
          };
          const clear = node('button', 'Remove separate dictionary', 'quiet');
          clear.onclick = async () => {
            try {
              const embedded = source.files.find(file => file.name.toLowerCase() === 'label_mapping.csv');
              const mapping = embedded ? parseLabelMapping(await embedded.text()) : [];
              if (!state.layers.includes(source)) return;
              source.mapping = mapping; source.dictionary = null; source.dictionaryText = null; sync(); refresh();
            } catch (error) { message(error.message); }
          }; row.append(clear);
        }
      }
      const actions = node('div', undefined, 'workspace-actions');
      for (const [text, delta] of [['↓ Lower', -1], ['↑ Raise', 1]]) {
        const button = node('button', text, 'quiet'); button.disabled = !state.layers[index + delta];
        button.setAttribute('aria-label', `${text.slice(2)} ${source.name}`);
        button.onclick = () => { [state.layers[index], state.layers[index + delta]] = [state.layers[index + delta], state.layers[index]]; sync(); refresh(); };
        actions.append(button);
      }
      const remove = node('button', 'Remove', 'quiet'); remove.setAttribute('aria-label', `Remove ${source.name}`);
      remove.onclick = () => {
        state.layers = state.layers.filter(item => item.id !== source.id);
        for (const view of state.views) {
          view.layers = view.layers.filter(item => item.id !== source.id);
          for (const channel of view.channels) if (channel.sourceId === source.id) Object.assign(channel, { sourceId: '', name: '', file: null, raster: null });
        }
        sync(); sourcesChanged();
      };
      actions.append(remove); row.append(actions); setup.append(row);
      const view = state.views[state.selectedView], layer = view.layers.find(item => item.id === source.id);
      if (!layer) continue;
      const card = node('div', undefined, 'aligned-card'), heading = node('label', undefined, 'check-label');
      const visible = node('input'); visible.type = 'checkbox'; visible.checked = layer.visible; visible.setAttribute('aria-label', `Show ${source.name}`);
      visible.onchange = () => { layer.visible = visible.checked; refresh(); };
      heading.append(visible, node('strong', source.name)); card.append(heading);
      check(card, `Show ${source.name} in legend`, layer.showInLegend !== false, value => { layer.showInLegend = value; updateLegends(); });
      const file = source.rois?.get($('roi').value);
      card.append(node('p', file ? `${LAYER_TYPES[source.kind]} · ${file.name}` : 'Not available in this ROI · settings retained', 'muted'));
      const opacity = labelled(card, `${source.name} opacity`, node('input')); opacity.type = 'range'; opacity.min = '0'; opacity.max = '1'; opacity.step = '0.05'; opacity.value = layer.opacity;
      opacity.oninput = () => { layer.opacity = Number(opacity.value); compose(); };
      const blend = labelled(card, `${source.name} blending`, select([['over', 'Over layers below'], ['add', 'Additive']], layer.blend));
      blend.onchange = () => { layer.blend = blend.value; compose(); };
      if (source.kind === 'scalar') {
        const colour = labelled(card, `${source.name} colourmap`, select(Object.entries(COLOURMAPS).map(([key, [name]]) => [key, name]), layer.colourmap));
        colour.onchange = () => { layer.colourmap = colour.value; compose(); };
        const range = node('div', undefined, 'two-columns');
        const inputs = ['low', 'high'].map(bound => { const input = labelled(range, `${source.name} ${bound}`, node('input')); input.type = 'number'; input.step = 'any'; input.value = layer[bound]; return input; });
        for (const input of inputs) input.oninput = () => {
          const [low, high] = inputs.map(el => el.valueAsNumber), valid = Number.isFinite(low) && Number.isFinite(high) && low < high;
          inputs.forEach(el => el.setCustomValidity(valid ? '' : 'High must be greater than low.'));
          if (valid) { Object.assign(layer, { low, high, auto: false }); compose(); }
        };
        card.append(range);
        const auto = node('button', 'Auto contrast', 'quiet'); auto.disabled = !layer.raster;
        auto.onclick = () => { [layer.low, layer.high] = contrast(layer.raster.data); layer.auto = false; sync(); compose(); }; card.append(auto);
      }
      if (source.kind === 'labels') {
        check(card, `Flatten ${source.name} labels`, layer.flatten ?? false, value => { layer.flatten = value; sync(); compose(); });
        if (layer.flatten) {
          const flatName = labelled(card, `${source.name} flattened label name`, node('input')); flatName.type = 'text'; flatName.value = layer.flatLabel ?? 'Foreground'; flatName.maxLength = 80;
          flatName.oninput = () => { layer.flatLabel = flatName.value.trim() || 'Foreground'; card.querySelector('.aligned-legend .legend-row span').textContent = layer.flatLabel; updateLegends(); };
          flatName.onchange = () => { flatName.value = layer.flatLabel; };
          const flatColour = labelled(card, `${source.name} flattened label colour`, node('input')); flatColour.type = 'color'; flatColour.value = layer.flatColour ?? '#81d8c1';
          flatColour.oninput = () => { layer.flatColour = flatColour.value; card.querySelector('.aligned-legend .legend-row i').style.backgroundColor = layer.flatColour; compose(); };
          card.append(node('p', 'All non-zero IDs form one label, including unmapped IDs. Zero stays transparent. Original image files are unchanged.', 'muted'));
        }
        const style = labelled(card, `${source.name} appearance`, select([['fill', 'Filled regions'], ['outline', 'Outlines']], layer.style));
        style.onchange = () => { layer.style = style.value; sync(); compose(); };
        if (layer.style === 'outline') {
          const weight = labelled(card, `${source.name} outline width (px)`, node('input')); weight.type = 'number'; weight.min = '1'; weight.max = '20'; weight.step = '1'; weight.value = layer.outlineWidth ?? 1;
          weight.onchange = () => { if (weight.checkValidity() && layer.outlineWidth !== weight.valueAsNumber) { layer.outlineWidth = weight.valueAsNumber; compose(); } };
          weight.onblur = weight.onchange;
        }
        const legend = node('div', undefined, 'legend aligned-legend');
        const entries = layer.flatten ? [{ value: 1, label: layer.flatLabel ?? 'Foreground' }] : source.mapping.length ? source.mapping : (layer.raster?.labels ?? []).map(value => ({ value, label: `Label ${value}` }));
        for (const entry of entries.filter(item => item.value).slice(0, 100)) {
          const colour = layer.flatten ? layer.flatColour ?? '#81d8c1' : labelColour(entry.value, source.mapping); if (!colour) continue;
          const row = node('div', undefined, 'legend-row'), swatch = node('i'); swatch.style.backgroundColor = Array.isArray(colour) ? `rgb(${colour})` : colour;
          row.append(swatch, node('span', entry.label), node('small', layer.flatten ? 'All > 0' : String(entry.value))); legend.append(row);
        }
        card.append(legend);
        if (!layer.flatten) card.append(node('p', source.mapping.length ? 'CSV colours · zero and unmapped IDs are transparent.' : 'Colours follow label IDs. Showing up to 100 labels; zero is transparent.', 'muted'));
      }
      controls.append(card);
    }
  }
  function flush() {
    if (controls.querySelector('input:invalid')) throw new Error('Fix the aligned layer contrast: high must be greater than low.');
  }
  return { index, sync, flush };
}
