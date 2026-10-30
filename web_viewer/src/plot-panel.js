import { createColourLimitControls } from './colour-limit-controls.js';
import { ReaderClient } from './rpc.js';
import { defaultPlots, validatePlots, PLOT_NAMES, plottedDataCSV, labelCategories } from './plots.js';
import { sourceKey, sourceLabel, remapRecipe } from './remaps.js';
import { plotSVG } from './plot-svg.js';

export function createPlotPanel(state, $, flushRemaps) {
  $('plot-root').innerHTML = `<div class="plot-actions"><button id="plot-generate" disabled>Generate / update</button><button id="plot-cancel" hidden>Cancel</button><progress id="plot-progress" max="1" value="0" hidden aria-label="Plot calculation progress"></progress><span id="plot-status" role="status">Open AnnData to create a plot.</span><button id="plot-expand" class="quiet" aria-pressed="false">Expand</button><button id="plot-svg" disabled>SVG</button><button id="plot-png" disabled>PNG</button><button id="plot-csv" disabled>Data CSV</button></div>
  <p id="plot-error" role="alert" class="input-error" hidden></p><div class="plot-layout"><fieldset id="plot-controls" disabled aria-label="Plot settings">
  <label>Plot type<select id="plot-type"></select></label><label>Population labels<select id="plot-groupBy"></select></label>
  <label>Cell scope<select id="plot-scope"><option value="current">Current ROI</option><option value="selected">Selected ROIs</option><option value="all">All cells / ROIs</option></select></label>
  <label id="plot-rois-wrap" hidden>ROIs<select id="plot-rois" multiple size="4"></select></label>
  <div data-plot="matrix"><label>Expression matrix<select id="plot-source"></select></label><label>Find markers<input id="plot-marker-search" type="search" placeholder="Filter marker list"></label><label>Markers (up to 100)<select id="plot-markers" multiple size="5"></select></label><div class="plot-mini-actions"><button type="button" id="plot-markers-all">Select visible</button><button type="button" id="plot-markers-clear">Clear markers</button></div><label>Colour scaling<select id="plot-scale"><option value="none">Unscaled means</option><option value="zscore">Marker-wise z-score</option><option value="minmax">Marker-wise 0–1</option></select></label></div>
  <div data-plot="bars composition"><label>Group samples by<select id="plot-sampleBy"></select></label><label>Measure<select id="plot-measure"><option value="percent">Within-group percentage</option><option value="count">Cell count</option></select></label></div>
  <div data-plot="comparison alluvial"><label>Compare against / destination<select id="plot-compareBy"></select></label><label data-plot="comparison">Values<select id="plot-normalization"><option value="row_percent">Within-row percentage</option><option value="column_percent">Within-column percentage</option><option value="count">Cell count</option></select></label></div>
  <details><summary>Population &amp; metadata filters</summary><button type="button" id="plot-load-populations">Load population choices</button><label>Include populations<select id="plot-populations" multiple size="5"></select></label><button type="button" id="plot-populations-clear">All populations</button><label>Filter observation<select id="plot-filterBy"></select></label><button type="button" id="plot-load-filter">Load filter values</button><label>Include values<select id="plot-filterValues" multiple size="4"></select></label><button type="button" id="plot-filter-clear">All values</button><p class="muted">An empty selection includes all values. Percentages use cells remaining after these filters.</p></details>
  <div id="plot-colour-limits" data-plot="matrix composition comparison"></div><details><summary>Ordering &amp; appearance</summary><label>Order groups<select id="plot-order"><option value="data">Dataset order</option><option value="name">Name</option><option value="count-desc">Cell count, descending</option><option value="count-asc">Cell count, ascending</option></select></label><div data-plot="bars composition"><label>Sort samples by population<select id="plot-sortPopulation"><option value="">Total selected cells</option></select></label><p class="muted">Applies to cell-count ordering; load population choices to choose a population.</p></div><label>Colour map<select id="plot-palette"><option value="automatic">Automatic</option><option value="blue">Blue</option><option value="viridis">Viridis</option><option value="diverging">Blue–red</option></select></label><label class="plot-check"><input id="plot-swapAxes" type="checkbox"> Swap heatmap axes</label><label>Title<input id="plot-title" maxlength="200" placeholder="Automatic title"></label></details>
  <p class="muted">Use Ctrl/Cmd-click for multiple selections. Plots are snapshots; regenerate after changing settings.</p></fieldset>
  <div id="plot-output" tabindex="0" aria-label="Generated plot"><p class="plot-empty">Choose a plot and select Generate / update.<br>Plots use your local AnnData and proposed remaps.</p></div></div>`;
  let job = 0, client = null, busy = false, result = null, svg = '', signature = '', populationChoices = [], filterChoices = [];
  const fields = Object.keys(defaultPlots()).filter(key => !['vmin', 'vmax'].includes(key));
  const limits = createColourLimitControls($('plot-colour-limits'), 'plot', value => { Object.assign(state.plots, value); error(); markStale(); });
  function options(id, items, selected, multiple = false) {
    const values = new Set(multiple ? selected : [selected]);
    const all = [...items];
    for (const value of values) if (value && !all.some(item => item.value === value)) all.push({ value, label: value });
    $(id).replaceChildren(...all.map(item => { const option = document.createElement('option'); option.value = item.value; option.textContent = item.label; option.selected = values.has(item.value); return option; }));
  }
  const arrayOptions = values => values.map(value => ({ value, label: value }));
  function error(message = '') { $('plot-error').textContent = message; $('plot-error').hidden = !message; }
  function capture() {
    const value = {};
    for (const field of fields) {
      const node = $(`plot-${field}`);
      value[field] = node.multiple ? [...node.selectedOptions].map(option => option.value) : node.type === 'checkbox' ? node.checked : node.value;
    }
    state.plots = validatePlots({ ...value, ...limits.read() }); return state.plots;
  }
  function request() {
    const settings = capture();
    const refs = [settings.groupBy, ...(['comparison', 'alluvial'].includes(settings.type) ? [settings.compareBy] : [])];
    return { settings, roiKey: state.mapping?.roi ?? '', currentROI: $('roi').value, remaps: state.remaps.filter(plan => refs.includes(`remap:${plan.id}`)) };
  }
  function fingerprint() {
    const req = request();
    return JSON.stringify([state.dataset, req.settings, req.settings.scope !== 'all' ? req.roiKey : '', req.settings.scope === 'current' ? req.currentROI : '', req.remaps.map(plan => [plan.name, remapRecipe(plan)])]);
  }
  function markStale() {
    if (busy || !result) return;
    let current = false; try { current = signature === fingerprint(); } catch (err) { error(err.message); }
    $('plot-status').textContent = current ? result.subtitle : 'Settings, ROI or remap changed · showing the previous snapshot. Generate / update to refresh.';
  }
  function visibility() {
    for (const node of $('plot-controls').querySelectorAll('[data-plot]')) node.hidden = !node.dataset.plot.split(' ').includes(state.plots.type);
    $('plot-rois-wrap').hidden = state.plots.scope !== 'selected';
    $('plot-swapAxes').disabled = ['bars', 'alluvial'].includes(state.plots.type); $('plot-palette').disabled = state.plots.type === 'alluvial';
    $('plot-load-filter').disabled = !state.plots.filterBy;
    $('plot-generate').disabled = !state.anndata || busy;
  }
  function sync() {
    const settings = state.plots, header = state.header;
    limits.sync(settings);
    if (!settings.groupBy && header) settings.groupBy = `obs:${header.obsKeys.find(name => /population|celltype|leiden/i.test(name)) ?? header.obsKeys[0]}`;
    if (!settings.sampleBy) settings.sampleBy = state.mapping?.roi ?? '';
    const labels = [...(header?.obsKeys ?? []).map(name => ({ value: `obs:${name}`, label: `Observation · ${name}` })), ...state.remaps.map(plan => ({ value: `remap:${plan.id}`, label: `Remap · ${plan.name}` }))];
    options('plot-type', Object.entries(PLOT_NAMES).map(([value, label]) => ({ value, label })), settings.type);
    options('plot-groupBy', [{ value: '', label: 'Choose labels' }, ...labels], settings.groupBy);
    options('plot-compareBy', [{ value: '', label: 'Choose comparison labels' }, ...labels], settings.compareBy);
    options('plot-source', arrayOptions(header?.sources ?? []), settings.source);
    options('plot-sampleBy', [{ value: '', label: 'Choose grouping' }, ...arrayOptions(header?.obsKeys ?? [])], settings.sampleBy);
    options('plot-filterBy', [{ value: '', label: 'No additional filter' }, ...arrayOptions(header?.obsKeys ?? [])], settings.filterBy);
    options('plot-rois', arrayOptions(state.rois.map(roi => roi.name)), settings.rois, true);
    options('plot-markers', arrayOptions(header?.markers ?? []), settings.markers, true);
    const fallback = values => labelCategories(values.map(key => ({ key, value: JSON.parse(key) }))).map(row => ({ value: row.key, label: row.label }));
    options('plot-populations', populationChoices.length ? populationChoices : fallback(settings.populations), settings.populations, true);
    options('plot-sortPopulation', [{ value: '', label: 'Total selected cells' }, ...populationChoices], settings.sortPopulation);
    options('plot-filterValues', filterChoices.length ? filterChoices : fallback(settings.filterValues), settings.filterValues, true);
    for (const field of fields) if (!['type', 'groupBy', 'compareBy', 'source', 'sampleBy', 'filterBy', 'rois', 'markers', 'populations', 'filterValues', 'sortPopulation'].includes(field)) {
      const node = $(`plot-${field}`); if (node.type === 'checkbox') node.checked = settings[field]; else node.value = settings[field];
    }
    $('plot-controls').disabled = !header || busy; visibility(); filterMarkers(); markStale();
    if (header && !busy && !result && $('plot-status').textContent === 'Open AnnData to create a plot.') $('plot-status').textContent = 'Choose plot settings, then Generate / update.';
  }
  function filterMarkers() {
    const query = $('plot-marker-search').value.toLocaleLowerCase();
    for (const option of $('plot-markers').options) option.hidden = !option.textContent.toLocaleLowerCase().includes(query);
  }
  function setBusy(value) { busy = value; $('plot-controls').disabled = value || !state.header; $('plot-cancel').hidden = !value; $('plot-progress').hidden = !value; $('plot-output').setAttribute('aria-busy', String(value)); visibility(); }
  async function run(title, action) {
    const current = ++job, dataset = state.dataset;
    client?.stop(); client = new ReaderClient(); const worker = client;
    setBusy(true); error(); $('plot-status').textContent = title; $('plot-progress').removeAttribute('value');
    try {
      await worker.call('open', state.anndata);
      const output = await action(worker, progress => { if (current === job) { $('plot-status').textContent = progress.detail; $('plot-progress').value = progress.fraction; } });
      if (current === job && dataset === state.dataset) return output;
    } catch (err) { if (current === job) { error(err.message); $('plot-status').textContent = 'Plot could not be generated. Check the settings and try again.'; } }
    finally { worker.stop(); if (current === job) { client = null; setBusy(false); } }
  }
  function cancel() { job++; client?.stop(); client = null; setBusy(false); $('plot-status').textContent = 'Cancelled. The previous plot is unchanged.'; }
  $('plot-generate').addEventListener('click', async () => {
    if (!flushRemaps()) { error('Correct the population remap before plotting.'); return; }
    let req, key; try { req = request(); key = fingerprint(); } catch (err) { error(err.message); return; }
    const output = await run('Opening local AnnData…', (worker, progress) => worker.callProgress('plotSummary', progress, req));
    if (!output) return;
    result = output; signature = key; svg = plotSVG(output); $('plot-output').innerHTML = svg;
    for (const id of ['plot-svg', 'plot-png', 'plot-csv']) $(id).disabled = false;
    markStale();
  });
  $('plot-cancel').addEventListener('click', cancel);
  async function loadChoices(kind) {
    capture(); const ref = kind === 'populations' ? state.plots.groupBy : `obs:${state.plots.filterBy}`;
    const plan = state.remaps.find(plan => `remap:${plan.id}` === ref);
    const output = plan ? { rows: [...new Set(plan.rows.map(row => row.label))].map(source => ({ source })) } : await run('Reading available values…', worker => worker.call('populationSummary', ref.slice(4)));
    if (!output) return;
    const choices = labelCategories(output.rows.map(row => ({ value: row.source, key: sourceKey(row.source) }))).map(row => ({ value: row.key, label: row.label }));
    if (kind === 'populations') populationChoices = choices; else filterChoices = choices;
    sync(); if (!result) $('plot-status').textContent = 'Choices loaded. Select values, then generate the plot.';
  }
  $('plot-load-populations').addEventListener('click', () => loadChoices('populations'));
  $('plot-load-filter').addEventListener('click', () => loadChoices('filter'));
  for (const field of fields) $(`plot-${field}`).addEventListener(field === 'title' ? 'input' : 'change', () => {
    try {
      capture(); error();
      if (field === 'groupBy') { state.plots.populations = []; state.plots.sortPopulation = ''; populationChoices = []; sync(); }
      else if (field === 'filterBy') { state.plots.filterValues = []; filterChoices = []; sync(); }
      visibility(); markStale();
    } catch (err) { error(err.message); }
  });
  $('plot-marker-search').addEventListener('input', filterMarkers);
  for (const field of ['markers', 'populations', 'filterValues']) $(`plot-${field === 'filterValues' ? 'filter' : field}-clear`).addEventListener('click', () => { for (const option of $(`plot-${field}`).options) option.selected = false; capture(); markStale(); });
  $('plot-markers-all').addEventListener('click', () => {
    const options = [...$('plot-markers').options], proposed = options.filter(option => !option.hidden || option.selected);
    if (proposed.length > 100) { error('Select at most 100 markers. Narrow the search first.'); return; }
    proposed.forEach(option => { option.selected = true; }); capture(); error(); markStale();
  });
  function expand(value) { $('review-panels').classList.toggle('plot-expanded', value); $('plot-expand').textContent = value ? 'Restore size' : 'Expand'; $('plot-expand').setAttribute('aria-pressed', String(value)); }
  $('plot-expand').addEventListener('click', () => expand(!$('review-panels').classList.contains('plot-expanded')));
  $('review-panels').addEventListener('toggle', () => { if (!$('review-panels').open) expand(false); });
  for (const tab of document.querySelectorAll('.review-tabs [role=tab]')) for (const event of ['click', 'keydown']) tab.addEventListener(event, () => { if (state.review.tab !== 'plots') expand(false); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') expand(false); });
  function download(blob, suffix) {
    const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = `${result.title.replace(/[^a-zA-Z0-9._-]/g, '_')}.${suffix}`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $('plot-svg').addEventListener('click', () => download(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), 'svg'));
  $('plot-csv').addEventListener('click', () => download(new Blob(['\uFEFF' + plottedDataCSV(result)], { type: 'text/csv;charset=utf-8' }), 'csv'));
  $('plot-png').addEventListener('click', async () => {
    const output = svg, snapshot = result; $('plot-png').disabled = true;
    const url = URL.createObjectURL(new Blob([output], { type: 'image/svg+xml' }));
    try {
      const image = new Image(); image.src = url; await image.decode();
      const scale = Math.min(2, 8192 / image.width, 8192 / image.height, Math.sqrt(24000000 / (image.width * image.height)));
      const canvas = document.createElement('canvas'); canvas.width = Math.ceil(image.width * scale); canvas.height = Math.ceil(image.height * scale);
      canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('PNG export failed. Use SVG for a large plot.');
      if (result === snapshot) download(blob, 'png');
    } catch (err) { error(err.message); } finally { URL.revokeObjectURL(url); $('plot-png').disabled = !result; }
  });
  return {
    sync, capture, changed() { try { capture(); } catch (err) { error(err.message); } for (const key of ['groupBy', 'compareBy']) if (state.plots[key].startsWith('remap:') && !state.remaps.some(plan => `remap:${plan.id}` === state.plots[key])) { state.plots[key] = ''; if (key === 'groupBy') { state.plots.populations = []; populationChoices = []; } } populationChoices = []; sync(); },
    roiChanged: markStale,
    reset(settings = defaultPlots()) { cancel(); result = null; svg = ''; signature = ''; populationChoices = []; filterChoices = []; state.plots = validatePlots(settings); error(); expand(false); $('plot-output').innerHTML = '<p class="plot-empty">Select Generate / update to create a plot from your local data.</p>'; for (const id of ['plot-svg', 'plot-png', 'plot-csv']) $(id).disabled = true; $('plot-status').textContent = state.header ? 'Plot settings ready. Select Generate / update.' : 'Open AnnData to create a plot.'; sync(); },
  };
}
