import { assignSelection } from './selection.js';
import { flagsCSV, validateFlags, cellCentres, pointInPolygon } from './flags.js';

export function createFlagPanel(state, $, actions) {
  let history = [], centres, mask;
  let tool = 'inspect', space = 'tissue';
  $('flag-root').innerHTML = `<div class="flag-toolbar">
    <label>Label fields<select id="flag-field"></select></label><button id="flag-create">+ New label field</button>
    <label>Field name<input id="flag-name" maxlength="120"></label><button id="flag-use">Use in selected view</button><button id="flag-export">Export CSV</button><button id="flag-delete" class="quiet">Remove field</button>
    </div><div class="flag-toolbar"><strong id="flag-selected">0 cells selected</strong><span class="muted">Use the viewer quick toolbar to select cells in tissue or embedding.</span>
    </div><div class="flag-toolbar"><label>Category<select id="flag-category"></select></label><button id="flag-add-category">+ Category</button>
    <label>Category name<input id="flag-category-name" maxlength="256"></label><input id="flag-colour" type="color" aria-label="Flag category colour" value="#ffcc00">
    <button id="flag-apply" class="primary-action">Apply to selected cells</button><button id="flag-clear">Unassign selected</button><button id="flag-undo">Undo</button></div>
    <p id="flag-status" class="muted" role="status"></p><p id="flag-error" class="input-error" role="alert" hidden></p>
    <p class="muted">Use the quick toolbar: click toggles a cell; lasso adds cells inside the outline in tissue or embedding. Selections span ROIs and remain until cleared. Tissue selection follows the pane’s population filter; embedding context selects plotted points only. CSV exports assigned cells with ROI + mask object ID. Labels save automatically to the workspace.</p>`;
  const field = () => state.flags.find(item => item.id === state.activeFlag);
  const message = error => { $('flag-error').textContent = error?.message ?? ''; $('flag-error').hidden = !error; };
  function options(node, items, value) { node.replaceChildren(...items.map(([id, label]) => { const o = document.createElement('option'); o.value = id; o.textContent = label; return o; })); if (items.some(item => item[0] === value)) node.value = value; }
  function sync() {
    const item = field(); options($('flag-field'), state.flags.map(f => [f.id, f.name]), state.activeFlag);
    $('flag-name').value = item?.name ?? ''; const selected = $('flag-category').value;
    options($('flag-category'), (item?.categories ?? []).map(c => [c.label, c.label]), selected);
    const category = item?.categories.find(c => c.label === $('flag-category').value);
    $('flag-category-name').value = category?.label ?? ''; $('flag-colour').value = category?.colour ?? '#ffcc00';
    for (const id of ['flag-field', 'flag-name', 'flag-use', 'flag-export', 'flag-delete', 'flag-category', 'flag-add-category', 'flag-category-name', 'flag-colour']) $(id).disabled = !item;
    $('flag-create').disabled = !state.header || state.flags.length >= 30;
    selectionChanged(); $('flag-status').textContent = item ? `${item.assignments.length.toLocaleString()} assigned cells across all ROIs · ${item.categories.length} categories` : 'Create a label field to begin flagging cells.';
  }
  function selectionChanged() {
    $('flag-selected').textContent = `${state.selection.size.toLocaleString()} cells selected`;
    for (const id of ['flag-apply', 'flag-clear']) $(id).disabled = !field() || !state.selection.size || !state.ready;
    $('selection-count').textContent = state.selection.size.toLocaleString(); $('selection-clear').disabled = !state.selection.size; $('selection-visible').disabled = !state.ready; $('flag-undo').disabled = !history.length;
    actions.selectionChanged?.();
  }
  function change(redraw = true) { message(); sync(); actions.changed(redraw); }
  function record(undo) { history.push(undo); if (history.length > 30) history.shift(); }
  function guarded(fn) { try { fn(); } catch (error) { message(error); } }
  $('flag-create').onclick = () => { let n = 1; while (state.flags.some(f => f.name === `Flags ${n}`)) n++; const item = { id: crypto.randomUUID(), name: `Flags ${n}`, categories: [{ label: 'Flagged', colour: '#ffcc00' }], assignments: [] }; state.flags.push(item); state.activeFlag = item.id; record(() => { state.flags = state.flags.filter(f => f.id !== item.id); state.activeFlag = state.flags[0]?.id ?? ''; }); setMode('click'); change(); };
  $('flag-field').onchange = () => { state.activeFlag = $('flag-field').value; sync(); };
  $('flag-category').onchange = sync;
  function editName() { const item = field(); if (!item) return; const old = item.name, name = $('flag-name').value.trim(); if (name === old) return; validateFlags(state.flags.map(f => f === item ? { ...f, name } : f)); item.name = name; record(() => { item.name = old; }); change(false); }
  $('flag-name').onchange = $('flag-name').onblur = () => guarded(editName);
  $('flag-add-category').onclick = () => guarded(() => { const item = field(); if (item.categories.length >= 100) throw new Error('Use at most 100 categories.'); let n = 1; while (item.categories.some(c => c.label === `Category ${n}`)) n++; item.categories.push({ label: `Category ${n}`, colour: '#ffcc00' }); record(() => item.categories.pop()); sync(); $('flag-category').value = `Category ${n}`; change(); });
  function editCategory() { const item = field(); if (!item) return; const old = $('flag-category').value, label = $('flag-category-name').value.trim(), colour = $('flag-colour').value;
    if (label === old && item.categories.find(c => c.label === old)?.colour === colour) return;
    if (!label || item.categories.some(c => c.label === label && c.label !== old)) throw new Error('Use a non-empty, unique category name.');
    const before = structuredClone(item); item.categories = item.categories.map(c => c.label === old ? { label, colour } : c); item.assignments = item.assignments.map(([r, i, l]) => [r, i, l === old ? label : l]);
    record(() => Object.assign(item, before)); sync(); $('flag-category').value = label; change(); }
  $('flag-category-name').onchange = $('flag-category-name').onblur = () => guarded(editCategory); $('flag-colour').onchange = () => guarded(editCategory);
  for (const [id, clear] of [['flag-apply', false], ['flag-clear', true]]) $(id).onclick = () => guarded(() => { record(assignSelection(field(), state.selection, clear ? null : $('flag-category').value)); change(); });
  $('flag-undo').onclick = () => { history.pop()?.(); change(); };
  $('flag-delete').onclick = () => { const item = field(), index = state.flags.indexOf(item); state.flags.splice(index, 1); state.activeFlag = state.flags[0]?.id ?? ''; record(() => { state.flags.splice(index, 0, item); state.activeFlag = item.id; }); change(); };
  function setMode(value) { const changed = tool !== value; tool = value; for (const mode of ['inspect', 'click', 'lasso']) $(`selection-${mode}`).setAttribute('aria-pressed', String(mode === tool)); if (changed) actions.modeChanged?.(); }
  const scope = () => space === 'embedding' && state.review.tab === 'embedding' && $('review-panels').open ? 'embedding' : 'tissue';
  for (const mode of ['inspect', 'click', 'lasso']) $(`selection-${mode}`).onclick = () => setMode(mode);
  $('selection-clear').onclick = () => { state.selection.clear(); selectionChanged(); actions.draw(); };
  $('selection-visible').onclick = () => { if (!state.ready) return; if (scope() === 'embedding') { actions.selectEmbedding(); return; } state.region.ids.forEach((id, i) => { if (actions.included(i)) state.flagSelection.add(id); }); selectionChanged(); actions.draw(); };
  $('flag-use').onclick = () => actions.use(field());
  function flush() { editName(); editCategory(); validateFlags(state.flags); }
  $('flag-export').onclick = () => guarded(() => { flush(); const item = field(), url = URL.createObjectURL(new Blob([flagsCSV(item, state.mapping)], { type: 'text/csv;charset=utf-8' })); const link = document.createElement('a'); link.href = url; link.download = `${item.name.replace(/[^a-z0-9_-]/gi, '_')}.flags.csv`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
  return { sync, flush, selectionChanged, setMode, scope, mode: () => tool, space(value) { space = value; actions.selectionChanged?.(); },
    click(id) { const i = state.idToIndex?.get(id); if (i === undefined || !actions.included(i)) return; if (state.flagSelection.has(id)) state.flagSelection.delete(id); else state.flagSelection.add(id); selectionChanged(); actions.draw(); },
    lasso(polygon) { if (mask !== state.mask) { mask = state.mask; centres = cellCentres(mask.data, state.width, state.idToIndex); } for (const [id, [x, y]] of centres) if (pointInPolygon(x, y, polygon) && actions.included(state.idToIndex.get(id))) state.flagSelection.add(id); selectionChanged(); actions.draw(); },
    clearSelection() { state.selection.clear(); selectionChanged(); },
    reset() { history = []; mask = null; centres = null; state.selection.clear(); setMode('inspect'); message(); sync(); },
  };
}
