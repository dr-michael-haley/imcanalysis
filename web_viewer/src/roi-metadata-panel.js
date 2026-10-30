import { ReaderClient } from './rpc.js';
import { sourceKey, sourceLabel } from './remaps.js';
import { filterROIs } from './review-data.js';

export function createROIMetadataPanel(state, $, fillSelect, changed) {
  let worker, request = 0, loaded = false, pending;
  const idleMessage = 'Metadata is not loaded. Press Read metadata to populate it; large datasets may take time.';
  $('roi-filter-root').innerHTML = `<p id="roi-metadata-status" class="muted" role="status">${idleMessage}</p><progress id="roi-metadata-progress" max="1" hidden></progress><div class="action-row"><button id="roi-metadata-load">Read metadata</button><button id="roi-metadata-cancel" hidden>Cancel</button></div>
    <label>Metadata field<select id="roi-filter-field"></select></label><label>Include values (Ctrl / ⌘ for several)<select id="roi-filter-values" multiple size="4"></select></label><button id="roi-filter-add">Apply field filter</button><button id="roi-filter-reset" class="quiet">Reset all filters</button><div id="roi-filter-active"></div><p id="roi-filter-count" class="muted" role="status"></p>`;
  function choices() {
    const field = state.roiMetadata.find(item => item.name === $('roi-filter-field').value);
    const values = new Map((field?.values ?? []).map(([, value]) => [sourceKey(value), value]));
    fillSelect($('roi-filter-values'), [...values].map(([value, label]) => ({ value, text: sourceLabel(label) })));
    const selected = state.roiFilters.find(filter => filter.field === field?.name)?.values ?? [];
    for (const option of $('roi-filter-values').options) option.selected = selected.includes(option.value);
    $('roi-filter-add').disabled = !field;
  }
  function summary() {
    $('roi-metadata-summary').replaceChildren();
    if (!loaded) { const hint = document.createElement('dd'); hint.textContent = 'Sample metadata: use Read metadata in Filtering to populate.'; $('roi-metadata-summary').append(hint); return; }
    for (const field of state.roiMetadata) {
      const value = field.values.find(([roi]) => roi === $('roi').value)?.[1];
      const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = field.name; dd.textContent = sourceLabel(value ?? null); $('roi-metadata-summary').append(dt, dd);
    }
  }
  function apply(notify = true, preferred = $('roi').value) {
    const ranking = state.ranking ?? state.rois;
    const filtersReady = loaded && state.roiFilters.every(filter => state.roiMetadata.some(field => field.name === filter.field));
    state.filteredRanking = state.roiFilters.length && filtersReady ? filterROIs(ranking, state.roiMetadata, state.roiFilters) : ranking;
    fillSelect($('roi'), state.filteredRanking.map(item => ({ value: item.name, text: item.name })), preferred);
    $('roi-filter-count').textContent = `${state.filteredRanking.length.toLocaleString()} / ${state.rois.length.toLocaleString()} ROIs included${state.filteredRanking.length ? '' : ' · No ROIs match. Reset or change filters.'}`;
    if (state.roiFilters.length && !filtersReady) $('roi-filter-count').textContent += loaded ? ' · Saved filters are not applied: one or more fields are unavailable. Remove them or reset filters.' : ' · Saved filters are pending. Press Read metadata to apply them.';
    $('roi-filter-active').replaceChildren();
    state.roiFilters.forEach(filter => { const row = document.createElement('div'); row.className = 'filter-chip'; const text = document.createElement('span'); text.textContent = `${filter.field}: ${filter.values.map(key => sourceLabel(JSON.parse(key))).join(', ')}`; const button = document.createElement('button'); button.textContent = '×'; button.setAttribute('aria-label', `Remove ${filter.field} filter`); button.onclick = () => { state.roiFilters = state.roiFilters.filter(item => item !== filter); apply(); choices(); }; row.append(text, button); $('roi-filter-active').append(row); });
    summary(); if (notify) changed(preferred !== $('roi').value);
  }
  async function load() {
    if (loaded || !state.anndata || !state.mapping) return;
    if (pending) return pending;
    const mine = ++request, current = new ReaderClient(); worker = current;
    $('roi-metadata-load').disabled = true; $('roi-metadata-cancel').hidden = false; $('roi-metadata-progress').hidden = false; $('roi-metadata-progress').removeAttribute('value'); $('roi-metadata-status').textContent = 'Opening AnnData for sample metadata…';
    pending = (async () => {
      try {
        await current.call('open', state.anndata);
        const result = await current.callProgress('roiMetadata', p => { if (mine === request) { $('roi-metadata-status').textContent = p.detail; $('roi-metadata-progress').value = p.fraction; } }, { roiKey: state.mapping.roi, idKey: state.mapping.id });
        if (mine !== request) return;
        state.roiMetadata = result.fields; loaded = true;
        fillSelect($('roi-filter-field'), result.fields.map(field => ({ value: field.name, text: field.name })));
        choices(); apply(); $('roi-metadata-status').textContent = `${result.fields.length} observations constant within every ROI.${result.skipped.length ? ` Could not read: ${result.skipped.join('; ')}` : ''}`;
      } catch (error) { if (mine === request) $('roi-metadata-status').textContent = error.message; }
      finally { current.stop(); if (mine === request) { worker = null; pending = null; $('roi-metadata-load').disabled = false; $('roi-metadata-cancel').hidden = true; $('roi-metadata-progress').hidden = true; } }
    })(); return pending;
  }
  $('roi-filter-field').onchange = choices;
  $('roi-filter-add').onclick = () => { const field = $('roi-filter-field').value, values = [...$('roi-filter-values').selectedOptions].map(o => o.value); if (!field || !values.length) { $('roi-metadata-status').textContent = 'Select one or more values to include.'; return; } state.roiFilters = state.roiFilters.filter(f => f.field !== field).concat({ field, values }); apply(); };
  $('roi-filter-reset').onclick = () => { state.roiFilters = []; apply(); choices(); };
  $('roi-metadata-load').onclick = () => { loaded = false; load(); };
  function stop() { request++; worker?.stop(); worker = null; pending = null; $('roi-metadata-load').disabled = false; $('roi-metadata-cancel').hidden = true; $('roi-metadata-progress').hidden = true; }
  $('roi-metadata-cancel').onclick = () => { stop(); $('roi-metadata-status').textContent = 'Metadata reading cancelled. Press Read metadata to retry.'; };
  choices();
  return { apply, summary, load, isLoaded: () => loaded, isLoading: () => Boolean(pending), reset() { stop(); loaded = false; state.roiMetadata = []; state.roiFilters = []; state.filteredRanking = null; fillSelect($('roi-filter-field'), []); choices(); $('roi-filter-active').replaceChildren(); summary(); $('roi-filter-count').textContent = ''; $('roi-metadata-status').textContent = idleMessage; } };
}
