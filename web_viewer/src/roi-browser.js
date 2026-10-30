import { recipeFingerprint } from './views.js';

export function createROIBrowser(state, $, fillSelect, warning, applyFilters, navigationChanged = () => {}) {
  let orderRequest = 0, visitRequest = 0, visitTimer, ordering = false;
  async function fields(field = '', value = '') {
    const request = ++orderRequest, mode = $('order-mode').value; ordering = false;
    $('order-field-label').hidden = !['population', 'obs', 'marker'].includes(mode);
    $('order-value-label').hidden = mode !== 'population'; $('order-source-label').hidden = mode !== 'marker';
    const names = mode === 'marker' ? state.header?.markers ?? [] : state.header?.obsKeys ?? [];
    fillSelect($('order-field'), names.map(key => ({ value: key, text: key })), field || (mode === 'population' ? names.find(key => /population|celltype|leiden/i.test(key)) : names[0]));
    fillSelect($('order-source'), (state.header?.sources ?? []).map(key => ({ value: key, text: key })), state.order.source);
    $('apply-order').disabled = !state.header;
    if (mode === 'population' && state.header) {
      $('apply-order').disabled = true;
      try {
        const categories = await state.reader.call('categories', $('order-field').value);
        if (request !== orderRequest) return;
        fillSelect($('order-value'), categories.map(item => ({ value: JSON.stringify(item), text: String(item) })), value);
        $('apply-order').disabled = !categories.length;
      } catch (error) { if (request === orderRequest) { fillSelect($('order-value'), []); $('order-status').textContent = error.message; } }
    }
  }
  async function sync() {
    const spec = state.order;
    $('order-mode').value = spec.mode; $('order-direction').value = spec.direction; $('top-rois').value = String(spec.top);
    await fields(spec.field, spec.value); $('order-source').value = spec.source; update();
  }
  function update() {
    const spec = state.order;
    const title = { name: 'ROI name', cells: 'total cells', population: `${spec.field} = ${spec.value ? JSON.parse(spec.value) : ''} · cell count`, obs: `mean ${spec.field}`, marker: `mean ${spec.field} · ${spec.source}` }[spec.mode];
    if (!ordering) $('order-status').textContent = `Ordered by ${title} · ${spec.direction === 'desc' ? 'descending' : 'ascending'}.`;
    const visitClass = roi => { const count = state.views.filter(view => view.fingerprint && state.visits.get(view.fingerprint)?.has(roi)).length; return count === state.views.length ? 'visited' : count ? 'partial' : 'unvisited'; };
    const container = $('roi-shortcuts'), existing = new Map([...container.children].map(button => [button.dataset.roi, button]));
    const buttons = (state.filteredRanking ?? state.ranking ?? state.rois).slice(0, spec.top).map((item, i) => {
      const button = existing.get(item.name) ?? document.createElement('button'), kind = visitClass(item.name);
      button.className = `roi-shortcut ${kind}`; button.dataset.roi = item.name; button.setAttribute('aria-current', String(item.name === $('roi').value));
      button.textContent = `${i + 1}. ${item.name}${item.score == null ? '' : ` · ${Number.isInteger(item.score) ? item.score.toLocaleString() : item.score.toPrecision(4)}`}`;
      button.title = `${kind === 'visited' ? 'Viewed with all current pane settings' : kind === 'partial' ? 'Viewed with some current pane settings' : 'Not viewed with current pane settings'} · ${item.count.toLocaleString()} cells`;
      button.onclick = () => { if ($('roi').value !== item.name) { $('roi').value = item.name; $('roi').dispatchEvent(new Event('change')); } }; return button;
    });
    // Keep existing buttons attached when only visit colours or scores change,
    // including a number input's blur/change between pointerdown and pointerup.
    buttons.forEach((button, i) => { if (container.children[i] !== button) container.insertBefore(button, container.children[i] ?? null); });
    while (container.children.length > buttons.length) container.lastChild.remove();
    for (const item of $('roi').options) item.dataset.visited = visitClass(item.value);
    for (const id of ['previous-roi', 'quick-previous']) $(id).disabled = $('roi').selectedIndex <= 0;
    for (const id of ['next-roi', 'quick-next']) $(id).disabled = !$('roi').options.length || $('roi').selectedIndex >= $('roi').options.length - 1;
    navigationChanged();
  }
  function scheduleVisits() {
    const request = ++visitRequest, roi = $('roi').value; clearTimeout(visitTimer);
    const normalization = { dictionary: state.normalization ? [...state.normalization].sort(([a], [b]) => a.localeCompare(b)) : null, quantile: state.normalizationQuantile, minimum: state.normalizationMinimum };
    const views = [...state.views];
    Promise.all(views.map(view => recipeFingerprint(view, normalization))).then(keys => {
      if (request !== visitRequest || !state.ready) return;
      views.forEach((view, i) => { view.fingerprint = keys[i]; }); update();
      visitTimer = setTimeout(() => {
        if (request !== visitRequest || !state.ready || roi !== $('roi').value) return;
        views.forEach(view => {
          if (view.channels.some(channel => channel.active && channel.enabled && channel.name && !channel.raster)) return;
          if (!state.mask && !view.channels.some(channel => channel.raster)) return;
          if (!state.visits.has(view.fingerprint)) state.visits.set(view.fingerprint, new Set());
          state.visits.get(view.fingerprint).add(roi);
          document.dispatchEvent(new Event('workspace-changed'));
        });
        update();
      }, 700);
    }).catch(error => warning(error.message));
  }
  $('order-mode').addEventListener('change', () => { $('order-direction').value = $('order-mode').value === 'name' ? 'asc' : 'desc'; fields(); });
  $('order-field').addEventListener('change', () => fields($('order-field').value));
  $('apply-order').addEventListener('click', async () => {
    const dataset = state.dataset, request = ++orderRequest;
    const spec = { mode: $('order-mode').value, field: $('order-field').value, value: $('order-value').value, source: $('order-source').value, direction: $('order-direction').value, top: Math.max(1, Math.min(100, Number($('top-rois').value) || 10)) };
    ordering = true; $('apply-order').disabled = true; $('order-status').textContent = 'Calculating ROI order from AnnData…';
    try {
      const ranking = await state.reader.call('rank', spec);
      if (dataset !== state.dataset || request !== orderRequest) return;
      ordering = false; state.order = spec; state.ranking = ranking;
      applyFilters(); update();
      document.dispatchEvent(new Event('workspace-changed'));
    } catch (error) { if (dataset === state.dataset && request === orderRequest) $('order-status').textContent = error.message; }
    finally { if (request === orderRequest) { ordering = false; $('apply-order').disabled = !state.header; } }
  });
  $('reset-order').addEventListener('click', () => {
    orderRequest++; state.order = { mode: 'name', field: '', value: '', source: 'X', direction: 'asc', top: state.order.top };
    state.ranking = [...state.rois]; applyFilters(); sync();
  });
  $('top-rois').addEventListener('change', () => { state.order.top = Math.max(1, Math.min(100, Math.trunc(Number($('top-rois').value) || 10))); $('top-rois').value = String(state.order.top); update(); });
  $('reset-visits').addEventListener('click', () => { visitRequest++; clearTimeout(visitTimer); for (const view of state.views) state.visits.delete(view.fingerprint); update(); });
  return { sync, update, scheduleVisits };
}
