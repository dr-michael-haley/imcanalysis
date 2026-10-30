import { ReaderClient } from './rpc.js';
import { densityCurves, densityPath, densityWindow } from './density-curve.js';

export const defaultRepresentative = () => ({ top: 6, roiOnly: true, source: 'X', lowerQuantile: 0, upperQuantile: 1 });
export function validateRepresentative(value = defaultRepresentative()) {
  if (!value || !Number.isInteger(value.top) || value.top < 1 || value.top > 30 || typeof value.roiOnly !== 'boolean' || typeof value.source !== 'string') throw new Error('Invalid representative marker settings.');
  const lowerQuantile = value.lowerQuantile ?? 0, upperQuantile = value.upperQuantile ?? 1;
  if (!Number.isFinite(lowerQuantile) || !Number.isFinite(upperQuantile) || lowerQuantile < 0 || upperQuantile > 1 || lowerQuantile >= upperQuantile) throw new Error('Use 0 ≤ lower quantile < upper quantile ≤ 1.');
  return { top: value.top, roiOnly: value.roiOnly, source: value.source, lowerQuantile, upperQuantile };
}
export function representativeKey(state, roi) {
  const current = state.views[state.selectedView], settings = state.representative;
  const plan = state.remaps.find(item => `remap:${item.id}` === current.overlay.colourBy);
  return JSON.stringify([state.dataset, state.mapping, settings.roiOnly ? roi : null, current.overlay.colourBy, current.overlay.population, settings.top, settings.roiOnly, settings.source, plan ? [plan.source, plan.rows.map(row => [row.source, row.label])] : null]);
}
export function createRepresentativePanel(state, $, assign) {
  let worker, request = 0, timer, lastKey = '', result = null;
  const view = () => state.views[state.selectedView];
  $('representative-root').innerHTML = `<div class="quantile-controls"><strong>X-axis quantiles (0–1)</strong><div class="two-columns"><label>Lower<input id="representative-lower" type="number" min="0" max="1" step="0.01" value="0" aria-label="Lower X-axis quantile"></label><label>Upper<input id="representative-upper" type="number" min="0" max="1" step="0.01" value="1" aria-label="Upper X-axis quantile"></label></div><small>For each marker, pooled across both groups. Approximate quantiles; changes only the displayed range.</small></div><div class="two-columns"><label>Top markers<input id="representative-top" type="number" min="1" max="30" value="6"></label><label>Expression<select id="representative-source"></select></label></div>
    <label class="check-label"><input id="representative-roi" type="checkbox" checked> Current ROI only (faster)</label>
    <div class="action-row"><button id="representative-update">Find markers</button><button id="representative-cancel" hidden>Cancel</button></div>
    <p id="representative-status" class="muted" role="status">Select one population in Cell overlay.</p><progress id="representative-progress" max="1" hidden></progress>
    <p class="muted">Ranked by (population mean − other-cell mean) / all-cell standard deviation. Descriptive enrichment, not a significance test. Curves compare the selected population with all other cells in this scope. Both use the same smoothing and each has unit area; missing expression is excluded.</p><div id="representative-results"></div>`;
  function stop() { request++; worker?.stop(); worker = null; clearTimeout(timer); $('representative-cancel').hidden = true; $('representative-progress').hidden = true; }
  const axisError = document.createElement('p'); axisError.className = 'input-error'; axisError.hidden = true; axisError.setAttribute('role', 'alert'); $('representative-root').querySelector('.quantile-controls').append(axisError);
  function capture() { state.representative = validateRepresentative({ top: Number($('representative-top').value), roiOnly: $('representative-roi').checked, source: $('representative-source').value || 'X', lowerQuantile: $('representative-lower').valueAsNumber, upperQuantile: $('representative-upper').valueAsNumber }); return state.representative; }
  function sync() {
    const saved = state.representative, select = $('representative-source'); select.replaceChildren(...(state.header?.sources ?? []).map(name => { const o = document.createElement('option'); o.value = name; o.textContent = name; return o; }));
    select.value = (state.header?.sources ?? []).includes(saved.source) ? saved.source : state.header?.sources[0] ?? '';
    $('representative-top').value = saved.top; $('representative-roi').checked = saved.roiOnly;
    $('representative-lower').value = saved.lowerQuantile ?? 0; $('representative-upper').value = saved.upperQuantile ?? 1; changed();
  }
  function render() {
    const root = $('representative-results'); root.replaceChildren();
    for (const item of result?.markers ?? []) {
      const card = document.createElement('div'); card.className = 'representative-card';
      const title = document.createElement('strong'); title.textContent = item.marker;
      const score = document.createElement('small'); score.textContent = `score ${item.score.toFixed(2)} · means ${item.means.map(v => v.toPrecision(3)).join(' / ')}`;
      const actions = document.createElement('div'); actions.className = 'marker-actions';
      const match = (state.files.get($('roi').value)?.images ?? []).find(image => image.name === item.marker);
      ['R', 'G', 'B'].forEach((label, i) => { const button = document.createElement('button'); button.textContent = label; button.className = `marker-${label.toLowerCase()}`; button.disabled = !match; button.title = match ? `Show ${item.marker} in ${['red', 'green', 'blue'][i]} (channel ${i + 1})` : 'No matching image in this ROI'; button.setAttribute('aria-label', button.title); button.onclick = () => assign(item.marker, i); actions.append(button); });
      const selectedLabel = String(JSON.parse(result.population) ?? 'Missing'), labels = [selectedLabel, 'All other cells'];
      const legend = document.createElement('div'); legend.className = 'density-legend';
      labels.forEach((label, g) => { const entry = document.createElement('span'); entry.className = g ? 'density-others' : 'density-selected'; entry.textContent = `${label} (n=${item.counts[g].toLocaleString()})`; entry.title = `${g ? 'All cells outside the selected population' : `Selected population: ${label}`} · ${item.counts[g].toLocaleString()} finite values · ${result.roiOnly ? 'Current ROI' : 'All ROIs'}`; legend.append(entry); });
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('viewBox', '0 0 260 70'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${item.marker} expression density curves: ${selectedLabel} versus all other cells`);
      const fullDensity = item.density ??= densityCurves(item.histogram, item.min, item.max);
      const density = densityWindow(fullDensity, item.histogram, state.representative.lowerQuantile, state.representative.upperQuantile);
      const baseline = document.createElementNS(svg.namespaceURI, 'path'); baseline.setAttribute('d', 'M2,64 L258,64'); baseline.setAttribute('stroke', '#41545d'); svg.append(baseline);
      const colours = ['#64d8bc', '#ffb86b'];
      // Draw fills first, then lines, so neither group's fill hides the other.
      const paths = density.curves.map(values => densityPath(values, density.maximum));
      paths.forEach((d, g) => { const area = document.createElementNS(svg.namespaceURI, 'path'); area.setAttribute('d', `${d} L258,64 L2,64 Z`); area.setAttribute('fill', colours[g]); area.setAttribute('opacity', '0.09'); svg.append(area); });
      paths.forEach((d, g) => { const path = document.createElementNS(svg.namespaceURI, 'path'); path.classList.add('density-line'); path.setAttribute('d', d); path.setAttribute('fill', 'none'); path.setAttribute('stroke', colours[g]); path.setAttribute('stroke-width', '2'); path.setAttribute('stroke-linejoin', 'round'); if (g) path.setAttribute('stroke-dasharray', '5 3'); const hint = document.createElementNS(svg.namespaceURI, 'title'); hint.textContent = `${labels[g]} · ${item.counts[g].toLocaleString()} cells · smoothed density (unit area)`; path.append(hint); svg.append(path); });
      const range = document.createElement('small'); range.className = 'histogram-range'; range.textContent = `Expression ${density.x[0].toPrecision(3)} — ${density.x.at(-1).toPrecision(3)} · density`;
      card.append(title, actions, score, legend, svg, range); if (!match) { const hint = document.createElement('small'); hint.textContent = 'Image unavailable in this ROI'; card.append(hint); } root.append(card);
    }
  }
  async function generate() {
    stop(); const mine = request, current = view(), dataset = state.dataset;
    try {
      const settings = capture();
      if (!state.ready || !current.region?.categories || !current.overlay.population || (!current.overlay.colourBy.startsWith('obs:') && !current.overlay.colourBy.startsWith('remap:'))) throw new Error('Select one observation or remap population in Cell overlay.');
      lastKey = representativeKey(state, $('roi').value);
      const workerHere = new ReaderClient(); worker = workerHere;
      $('representative-status').textContent = 'Opening expression data…'; $('representative-cancel').hidden = false; $('representative-progress').hidden = false; $('representative-progress').removeAttribute('value');
      result = null; render();
      await workerHere.call('open', state.anndata); if (mine !== request) return;
      result = await workerHere.callProgress('representativeMarkers', p => { if (mine === request) { $('representative-status').textContent = p.detail; $('representative-progress').value = p.fraction; } }, {
        ...settings, population: current.overlay.population, colourBy: current.overlay.colourBy,
        rows: settings.roiOnly ? current.region.rows : null, values: settings.roiOnly ? current.region.values : null,
        remap: state.remaps.find(plan => `remap:${plan.id}` === current.overlay.colourBy),
      });
      if (mine !== request || dataset !== state.dataset) return;
      render(); $('representative-status').textContent = `${result.totals[0].toLocaleString()} selected / ${result.totals[1].toLocaleString()} other cells · ${settings.roiOnly ? $('roi').value : 'All ROIs'} · ${result.markers.length ? 'Top enriched markers' : 'No positively enriched markers in this scope'}`;
    } catch (error) { if (mine === request) { result = null; render(); $('representative-status').textContent = error.message; } }
    finally { if (mine === request) { worker?.stop(); worker = null; $('representative-progress').hidden = true; $('representative-cancel').hidden = true; } }
  }
  function changed() {
    const current = view(), key = representativeKey(state, $('roi').value);
    if (key === lastKey) { if (result && state.ready) render(); return; }
    stop(); result = null; render(); lastKey = state.ready ? key : '';
    if (!state.ready || !current.region?.categories || !current.overlay.population) { $('representative-status').textContent = 'Select one population in Cell overlay.'; return; }
    $('representative-status').textContent = 'Ready to rank markers for this population.';
    if ($('section-representative').open && state.tab === 'views') timer = setTimeout(generate, 180);
  }
  for (const id of ['representative-top', 'representative-roi', 'representative-source']) $(id).onchange = () => { try { capture(); changed(); } catch (error) { $('representative-status').textContent = error.message; } };
  for (const id of ['representative-lower', 'representative-upper']) $(id).oninput = () => {
    try { capture(); for (const key of ['representative-lower', 'representative-upper']) $(key).setCustomValidity(''); axisError.hidden = true; render(); }
    catch (error) { $(id).setCustomValidity(error.message); axisError.textContent = error.message; axisError.hidden = false; }
  };
  $('representative-update').onclick = generate;
  $('representative-cancel').onclick = () => { stop(); $('representative-status').textContent = 'Cancelled. Press Find markers to retry.'; };
  $('section-representative').addEventListener('toggle', () => { if ($('section-representative').open && !result && !worker && state.tab === 'views') changedForce(); });
  function changedForce() { lastKey = ''; changed(); }
  return { sync, capture, changed, activate() { if (!result && !worker) lastKey = ''; changed(); }, reset() { stop(); lastKey = ''; result = null; sync(); } };
}
