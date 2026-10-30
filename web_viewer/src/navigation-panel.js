import { PRESET_KEYS, presetForKey, validateNavigation } from './navigation.js';

export function createNavigationPanel(state, $, actions) {
  $('bookmark-root').innerHTML = `<div class="bookmark-heading"><button id="bookmark-add" class="primary-action">＋ Bookmark current viewer</button><span class="muted">Return to an ROI, all views and their shared zoom and position.</span></div><p id="bookmark-status" class="muted" role="status"></p><div class="bookmark-scroll"><table class="bookmark-table"><thead><tr><th>Name / return to view</th><th>ROI · views</th><th>Comments</th><th></th></tr></thead><tbody id="bookmark-rows"></tbody></table></div>`;
  $('saved-views-root').innerHTML = `<details><summary>Saved view presets</summary><p class="muted">Apply to the selected pane. ROI and camera stay in place.</p><div class="preset-save"><input id="preset-name" maxlength="120" aria-label="New view preset name" placeholder="Preset name"><button id="preset-save">Save current view</button></div><div id="preset-rows"></div><p class="muted">Assign an F-key for quick switching. F5, F11 and F12 remain browser shortcuts. Hotkeys are inactive while typing.</p></details>`;
  let busy = false;
  const status = message => { $('bookmark-status').textContent = message; };
  const button = (label, title, action) => { const b = document.createElement('button'); b.textContent = label; b.title = title; b.setAttribute('aria-label', title); b.onclick = action; return b; };
  const input = (value, label, max, change, multiline = false) => { const node = document.createElement(multiline ? 'textarea' : 'input'); node.value = value; node.maxLength = max; node.setAttribute('aria-label', label); node.oninput = node.onchange = () => { const next = node.value.trim(); if (!multiline && !next) { node.value = value; return; } change(next); actions.changed(); }; return node; };
  async function run(fn, label) {
    if (busy || !state.ready || actions.locked()) return;
    busy = true; sync();
    try { await fn(); status(label); actions.changed(); } catch (error) { status(error.message); actions.error(error); }
    finally { busy = false; sync(); }
  }
  function addBookmark() {
    if (!state.ready || busy || actions.locked()) return;
    try {
      if (state.bookmarks.length >= 500) throw new Error('Use at most 500 bookmarks per workspace.');
      const viewer = actions.snapshot();
      const item = { id: crypto.randomUUID(), name: `${viewer.roi} · ${state.bookmarks.length + 1}`, comment: '', createdAt: new Date().toISOString(), viewer };
      validateNavigation([item]); state.bookmarks.push(item); actions.open(); sync(); actions.changed(); status('Bookmark saved. Add an optional comment in the table.');
      $('bookmark-rows').lastElementChild?.querySelector('textarea')?.focus();
    } catch (error) { actions.error(error); }
  }
  function sync() {
    $('bookmark-add').disabled = $('quick-bookmark').disabled = !state.ready || busy;
    $('preset-save').disabled = !state.ready || busy;
    $('bookmark-rows').replaceChildren(...state.bookmarks.map(item => {
      const row = document.createElement('tr');
      const name = document.createElement('td'); name.append(button('↩ Return', `Return to bookmark ${item.name}`, () => run(() => actions.restore(item.viewer), `Returned to ${item.name}.`)), input(item.name, 'Bookmark name', 120, v => { item.name = v; }));
      const region = document.createElement('td'); region.textContent = `${item.viewer.roi} · ${item.viewer.panes.map(p => p.name).join(', ')}`; region.title = item.createdAt;
      const comments = document.createElement('td'); comments.append(input(item.comment, `Comments for ${item.name}`, 10000, v => { item.comment = v; }, true));
      const remove = document.createElement('td'); remove.append(button('×', `Remove bookmark ${item.name}`, () => { state.bookmarks = state.bookmarks.filter(b => b !== item); sync(); actions.changed(); }));
      row.append(name, region, comments, remove); return row;
    }));
    $('preset-rows').replaceChildren(...state.savedViews.map(item => {
      const row = document.createElement('div'); row.className = 'preset-row';
      const apply = button(item.name, `Apply ${item.name} to selected pane`, () => run(() => actions.preset(item.settings), `Applied ${item.name}.`)); apply.disabled = busy || !state.ready;
      const key = document.createElement('select'); key.setAttribute('aria-label', `Hotkey for ${item.name}`);
      for (const value of ['', ...PRESET_KEYS]) { const opt = document.createElement('option'); opt.value = value; opt.textContent = value || 'No key'; opt.disabled = Boolean(value && state.savedViews.some(p => p !== item && p.hotkey === value)); key.append(opt); } key.value = item.hotkey;
      key.onchange = () => { item.hotkey = key.value; sync(); actions.changed(); };
      row.append(apply, key, button('×', `Remove preset ${item.name}`, () => { state.savedViews = state.savedViews.filter(p => p !== item); sync(); actions.changed(); })); return row;
    }));
    $('bookmark-root').classList.toggle('is-empty', !state.bookmarks.length);
  }
  $('bookmark-add').onclick = $('quick-bookmark').onclick = addBookmark;
  $('preset-save').onclick = () => {
    if (!state.ready || busy || actions.locked()) return;
    try {
      if (state.savedViews.length >= 100) throw new Error('Use at most 100 view presets.');
      const settings = actions.view(), name = $('preset-name').value.trim() || settings.name;
      const item = { id: crypto.randomUUID(), name, hotkey: '', settings };
      validateNavigation([], [item]); state.savedViews.push(item); $('preset-name').value = ''; sync(); actions.changed();
    } catch (error) { actions.error(error); }
  };
  document.addEventListener('keydown', event => {
    const preset = presetForKey(state.savedViews, event);
    if (!preset) return;
    event.preventDefault();
    if (busy || !state.ready || actions.locked()) return;
    run(() => actions.preset(preset.settings), `Applied ${preset.name} (${preset.hotkey}).`);
  });
  sync(); return { sync, addExportBookmark(viewer, name) {
    if (state.bookmarks.length >= 500) throw new Error('Use at most 500 bookmarks per workspace.');
    const item = { id: crypto.randomUUID(), name: name.slice(0, 120), comment: `Figure export: ${name}`, createdAt: new Date().toISOString(), viewer };
    validateNavigation([item]); state.bookmarks.push(item); sync(); actions.changed();
  } };
}
