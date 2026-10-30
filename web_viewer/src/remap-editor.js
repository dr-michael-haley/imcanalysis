import { createRemap, editRemap, remapCSV, sourceKey, sourceLabel, validateRemaps } from './remaps.js';

export function createRemapEditor(state, $, { changed, useOverlay }) {
  let selected = new Set(), page = 0, request = 0, busy = false, history = [], editTimer;
  const active = () => state.remaps.find(plan => plan.id === state.activeRemap);
  const message = (text = '') => { $('remap-error').textContent = text; $('remap-error').hidden = !text; };
  function remember() { history.push({ remaps: state.remaps, active: state.activeRemap }); if (history.length > 20) history.shift(); }
  function options(id, items, chosen) {
    $(id).replaceChildren(...items.map(([value, label]) => { const option = document.createElement('option'); option.value = value; option.textContent = label; return option; }));
    if (items.some(([value]) => value === chosen)) $(id).value = chosen;
  }
  function actions() {
    const plan = active();
    for (const id of ['remap-name', 'remap-export', 'remap-use', 'remap-remove', 'remap-search']) $(id).disabled = !plan || busy;
    $('remap-create').disabled = !state.header || busy || state.remaps.length >= 20;
    $('remap-source').disabled = !state.header || busy;
    $('remap-select').disabled = !state.remaps.length || busy;
    $('remap-undo').disabled = !history.length || busy;
    $('remap-merge').disabled = !selected.size || busy;
    $('remap-selected').textContent = `${selected.size} selected`;
    $('remap-panel').classList.toggle('is-counting', busy);
    $('remap-table-wrap').setAttribute('aria-busy', String(busy));
    $('remap-table-wrap').inert = busy;
  }
  function summary() {
    const plan = active();
    $('remap-summary').textContent = busy ? 'Reading populations from AnnData…' : plan ? `${plan.source} · ${plan.rows.length.toLocaleString()} source populations → ${new Set(plan.rows.map(row => row.label)).size.toLocaleString()} proposed populations · ${plan.rows.reduce((sum, row) => sum + row.count, 0).toLocaleString()} cells across all ROIs` : 'Create a remap from an observation to begin. Source AnnData stays read only.';
    actions();
  }
  function decorate(input, row) {
    input.style.borderLeft = `5px solid ${row.colour}`;
  }
  function commit(keys, patch) {
    const plan = active(); if (!plan) return;
    if (!flush()) return;
    try {
      const updated = editRemap(active(), keys, patch);
      remember(); state.remaps = state.remaps.map(item => item.id === plan.id ? updated : item);
      message(); syncRows(); summary(); changed(patch.notes === undefined);
    } catch (error) { message(error.message); syncRows(); }
  }
  function flush() {
    clearTimeout(editTimer);
    const plan = active(); if (!plan) return true;
    try {
      let updated = plan, redraw = false;
      const original = new Map(plan.rows.map(row => [sourceKey(row.source), row]));
      for (const tr of $('remap-rows').children) {
        const row = original.get(tr.dataset.key); if (!row) continue;
        const patch = {};
        for (const field of ['label', 'colour', 'notes']) {
          const input = tr.querySelector(`[data-field="${field}"]:not([type=color])`);
          if (input.value !== row[field]) patch[field] = input.value;
        }
        if (Object.keys(patch).length) { updated = editRemap(updated, [tr.dataset.key], patch); redraw ||= patch.label !== undefined || patch.colour !== undefined; }
      }
      const name = $('remap-name').value.trim();
      if (name !== plan.name) updated = { ...updated, name };
      if (updated !== plan) {
        const next = validateRemaps(state.remaps.map(item => item.id === plan.id ? updated : item));
        remember(); state.remaps = next; syncRows(); summary();
        options('remap-select', state.remaps.map(item => [item.id, item.name]), state.activeRemap);
        changed(redraw);
      }
      message(); return true;
    } catch (error) { message(error.message); return false; }
  }
  function queueEdit() { clearTimeout(editTimer); editTimer = setTimeout(flush, 300); }
  function syncRows() {
    const plan = active(); if (!plan) return;
    const rows = new Map(plan.rows.map(row => [sourceKey(row.source), row]));
    for (const tr of $('remap-rows').children) {
      const row = rows.get(tr.dataset.key); if (!row) continue;
      for (const input of tr.querySelectorAll('[data-field]')) {
        input.value = row[input.dataset.field];
        if (input.dataset.field === 'label') decorate(input, row);
      }
    }
  }
  function renderRows() {
    const plan = active(), query = $('remap-search').value.trim().toLocaleLowerCase();
    const names = new Map();
    for (const row of plan?.rows ?? []) names.set(sourceLabel(row.source), (names.get(sourceLabel(row.source)) ?? 0) + 1);
    const rows = plan?.rows.filter(row => `${sourceLabel(row.source)} ${row.label} ${row.notes}`.toLocaleLowerCase().includes(query)) ?? [];
    page = Math.max(0, Math.min(page, Math.ceil(rows.length / 100) - 1));
    const visible = rows.slice(page * 100, (page + 1) * 100);
    $('remap-rows').replaceChildren(...visible.map(row => {
      const tr = document.createElement('tr'), key = sourceKey(row.source); tr.dataset.key = key;
      const name = sourceLabel(row.source) + (names.get(sourceLabel(row.source)) > 1 ? ` [${row.source === null ? 'missing' : typeof row.source}]` : '');
      const cell = node => { const td = document.createElement('td'); td.append(node); tr.append(td); };
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = selected.has(key); checkbox.setAttribute('aria-label', `Select ${name}`);
      checkbox.addEventListener('change', () => { checkbox.checked ? selected.add(key) : selected.delete(key); actions(); }); cell(checkbox);
      const source = document.createElement('span'); source.textContent = name; source.title = `Original value: ${key}`; cell(source);
      const count = document.createElement('span'); count.textContent = row.count.toLocaleString(); cell(count);
      for (const [field, label] of [['label', 'Proposed name'], ['colour', 'Colour'], ['notes', 'Notes']]) {
        const input = document.createElement(field === 'notes' ? 'textarea' : 'input'); input.dataset.field = field; input.value = row[field];
        input.setAttribute('aria-label', `${label} for ${name}`);
        input.maxLength = field === 'notes' ? 4000 : field === 'label' ? 256 : 7;
        if (field === 'notes') input.rows = 1;
        if (field === 'label') decorate(input, row);
        input.addEventListener('input', queueEdit);
        input.addEventListener('change', flush);
        if (field === 'colour') {
          const wrap = document.createElement('div'); wrap.className = 'remap-colour';
          const picker = document.createElement('input'); picker.type = 'color'; picker.dataset.field = field; picker.value = row.colour; picker.setAttribute('aria-label', `Pick colour for ${name}`);
          picker.addEventListener('input', () => { input.value = picker.value; queueEdit(); });
          picker.addEventListener('change', () => { input.value = picker.value; flush(); }); wrap.append(picker, input); cell(wrap);
        } else cell(input);
      }
      return tr;
    }));
    $('remap-page').textContent = rows.length ? `${page * 100 + 1}–${Math.min((page + 1) * 100, rows.length)} of ${rows.length.toLocaleString()}` : 'No populations';
    $('remap-previous').disabled = page === 0; $('remap-next').disabled = (page + 1) * 100 >= rows.length;
    $('remap-select-page').disabled = !visible.length;
    $('remap-select-page').onclick = () => { if (!flush()) return; for (const row of visible) selected.add(sourceKey(row.source)); renderRows(); actions(); };
    $('remap-table-wrap').hidden = !plan;
    summary();
  }
  function sync() {
    options('remap-source', (state.header?.obsKeys ?? []).map(name => [name, name]), $('remap-source').value || state.header?.obsKeys.find(name => /population|celltype|leiden/i.test(name)));
    options('remap-select', state.remaps.map(plan => [plan.id, plan.name]), state.activeRemap);
    state.activeRemap = $('remap-select').value;
    $('remap-name').value = active()?.name ?? '';
    renderRows();
  }
  $('remap-create').addEventListener('click', async () => {
    if (busy || !state.header || !flush()) return;
    const current = ++request, dataset = state.dataset, source = $('remap-source').value;
    busy = true; message(); summary();
    try {
      const result = await state.reader.call('populationSummary', source);
      if (current !== request || dataset !== state.dataset) return;
      let name = `${source}_reviewed`, suffix = 2;
      while (state.remaps.some(plan => plan.name === name)) name = `${source}_reviewed_${suffix++}`;
      const plan = createRemap(crypto.randomUUID(), name.slice(0, 120), source, result);
      remember(); state.remaps = [...state.remaps, plan]; state.activeRemap = plan.id; selected.clear(); page = 0;
      $('remap-search').value = ''; changed(false);
    } catch (error) { if (current === request && dataset === state.dataset) message(error.message); }
    finally { if (current === request) { busy = false; sync(); } }
  });
  $('remap-select').addEventListener('change', () => { const id = $('remap-select').value; if (!flush()) { $('remap-select').value = state.activeRemap; return; } state.activeRemap = id; selected.clear(); page = 0; message(); sync(); });
  $('remap-name').addEventListener('input', queueEdit);
  $('remap-name').addEventListener('change', flush);
  $('remap-merge').addEventListener('click', () => { commit([...selected], { label: $('remap-merge-name').value }); });
  $('remap-clear-selection').addEventListener('click', () => { if (!flush()) return; selected.clear(); renderRows(); });
  $('remap-search').addEventListener('input', () => { if (!flush()) return; page = 0; renderRows(); });
  for (const [id, delta] of [['remap-previous', -1], ['remap-next', 1]]) $(id).addEventListener('click', () => { if (!flush()) return; page += delta; renderRows(); });
  $('remap-undo').addEventListener('click', () => { clearTimeout(editTimer); const previous = history.pop(); if (!previous) return; state.remaps = previous.remaps; state.activeRemap = previous.active; selected.clear(); message(); sync(); changed(true); });
  $('remap-remove').addEventListener('click', () => { if (!active()) return; clearTimeout(editTimer); remember(); state.remaps = state.remaps.filter(plan => plan.id !== state.activeRemap); state.activeRemap = state.remaps[0]?.id ?? ''; selected.clear(); sync(); changed(true); });
  $('remap-use').addEventListener('click', () => { if (active() && flush()) useOverlay(active()); });
  $('remap-export').addEventListener('click', () => {
    if (!flush()) return;
    const plan = active(); if (!plan) return;
    const url = URL.createObjectURL(new Blob(['\uFEFF' + remapCSV(plan)], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = `${plan.name.replace(/[^a-zA-Z0-9._-]/g, '_')}.remap.csv`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  return { sync, flush, reset() { clearTimeout(editTimer); request++; busy = false; history = []; selected.clear(); page = 0; message(); sync(); } };
}
