import { WorkspaceAssets } from './workspace-assets.js';
import { assetPath } from './workspace-paths.js';
import { relativeFilePath } from './workspace.js';

export function createWorkspaceConnection($, actions) {
  let session = null, payload = null, pending = false, working = false, revision = 0, fallbackFiles = null;
  const nodes = new Map();
  const labels = { waiting: 'Waiting', checking: 'Checking…', ready: 'Registered', cached: 'Cached list', error: 'Needs attention' };
  const symbols = { waiting: '○', checking: '◌', ready: '✓', cached: '◷', error: '!' };
  function status(text, failure = '') {
    $('workspace-status').textContent = text;
    $('workspace-error').textContent = failure; $('workspace-error').hidden = !failure;
  }
  function render(row) {
    if (!session) return;
    for (const item of row ? [row] : session.rows) {
      const node = nodes.get(item.key); if (!node) continue;
      if (node.card.dataset.state !== item.state) node.card.open = item.state === 'checking' || item.state === 'error';
      node.card.dataset.state = item.state; node.badge.textContent = `${symbols[item.state]} ${labels[item.state]}`;
      node.detail.textContent = item.detail; node.path.textContent = item.path; node.path.title = item.path;
      if (document.activeElement !== node.input) node.input.value = item.path;
    }
    $('workspace-asset-count').textContent = `${session.connected}/${session.rows.length}`;
    $('workspace-progress').max = session.rows.length; $('workspace-progress').value = session.connected;
    $('workspace-progress').hidden = !working;
    $('workspace-retry-assets').hidden = !pending || session.complete || !session.root && !fallbackFiles;
    $('workspace-open-connected').hidden = !pending || !session.complete;
    $('workspace-root-label').hidden = !pending;
    $('choose-workspace-root').textContent = session.root || fallbackFiles ? 'Change data folder' : 'Choose data folder';
    controls();
  }
  function controls(locked = false) {
    locked ||= working;
    for (const node of $('workspace-location-fields').querySelectorAll('button, input')) node.disabled = locked;
    for (const id of ['workspace-check-mode', 'workspace-retry-assets', 'workspace-open-connected']) $(id).disabled = locked;
  }
  function summary() {
    if (session.complete) status(`${session.connected} of ${session.rows.length} assets registered. Opening the saved view…`);
    else status(`${session.connected} of ${session.rows.length} assets registered. Locate or retry the items marked “Needs attention”. Successful checks are kept.`);
  }
  async function persistCache() { if (session) await actions.saveCache(session.cache).catch(() => {}); }
  async function finish() {
    if (!session?.complete || !pending) return;
    summary(); const mine = revision;
    $('workspace-progress').hidden = false; $('workspace-progress').removeAttribute('value');
    const success = await actions.load(session.result(), payload);
    if (mine !== revision) return;
    if (success) { pending = false; status(`Workspace connected · ${session.connected} assets. ${session.rows.some(r => r.state === 'cached') ? 'Cached images are checked when opened.' : 'All asset file lists checked.'}`); }
    render();
  }
  async function withWork(action) {
    working = true; render();
    try { await action(); await persistCache(); if (session.complete) await finish(); else summary(); }
    finally {
      working = false; render();
      if (session?.rows.some(row => row.state === 'error')) $('workspace-connection').scrollIntoView({ block: 'start', behavior: 'smooth' });
    }
  }
  function build() {
    nodes.clear(); $('workspace-location-fields').replaceChildren();
    for (const row of session.rows) {
      const card = document.createElement('details'); card.className = 'workspace-asset';
      const heading = document.createElement('summary'); heading.className = 'workspace-asset-heading';
      const name = document.createElement('strong'); name.textContent = row.label;
      const badge = document.createElement('span'); badge.className = 'workspace-asset-badge'; heading.append(name, badge);
      const path = document.createElement('div'); path.className = 'workspace-asset-path';
      const detail = document.createElement('p'); detail.className = 'workspace-asset-detail';
      const buttons = document.createElement('div'); buttons.className = 'workspace-asset-actions';
      const locate = document.createElement('button'); locate.textContent = row.kind === 'folder' ? 'Locate folder' : 'Locate file'; locate.setAttribute('aria-label', `Locate ${row.label}`);
      const retry = document.createElement('button'); retry.textContent = 'Retry'; retry.setAttribute('aria-label', `Retry ${row.label}`);
      retry.onclick = () => actions.run(async () => {
        await ensurePending(); const current = session.rows.find(item => item.key === row.key);
        await withWork(async () => { if (session.root) await session.connect(session.root, { keys: [row.key], forceFull: true }); else if (fallbackFiles) await session.suppliedRow(current, fallbackFiles); else throw new Error('Choose a data folder or locate this asset first.'); });
      });
      const edit = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Edit path';
      const label = document.createElement('label'); label.textContent = 'Path relative to the data folder';
      const input = document.createElement('input'); input.type = 'text'; input.setAttribute('aria-label', `${row.label} relative path`);
      input.onchange = async () => {
        try { input.setCustomValidity(''); const path = assetPath(input.value); session.setPath(row.key, path); await actions.path(row.key, path); status('Location saved. Retry this asset to check it.'); }
        catch (error) { input.setCustomValidity(error.message); status('Check the relative path.', error.message); }
      };
      label.append(input); edit.append(summary, label);
      const fallback = document.createElement('input'); fallback.type = 'file'; fallback.className = 'visually-hidden'; fallback.tabIndex = -1;
      fallback.setAttribute('aria-label', `Choose ${row.label}`);
      if (row.kind === 'folder') { fallback.webkitdirectory = true; fallback.multiple = true; } else fallback.accept = row.key === 'anndata' ? '.h5ad' : '.csv';
      fallback.onchange = () => {
        const files = Array.from(fallback.files); fallback.value = ''; if (!files.length) return;
        actions.run(async () => {
          await ensurePending(); await withWork(async () => {
            let resolvedPath = false;
            if (row.kind === 'file' && fallbackFiles) {
              const matches = fallbackFiles.filter(file => file.name === files[0].name && file.size === files[0].size);
              if (matches.length === 1) { const path = relativeFilePath(matches[0]); session.setPath(row.key, path); await actions.path(row.key, path); resolvedPath = true; }
            }
            await session.supply(files, row.key);
            const current = session.rows.find(item => item.key === row.key);
            if (current.state === 'ready' && !resolvedPath) { current.detail += ' · selected for this session; check Edit path for future loads'; render(current); }
          });
        });
      };
      locate.onclick = () => {
        const picker = row.kind === 'folder' ? window.showDirectoryPicker : window.showOpenFilePicker;
        if (!picker || !session.root && (fallbackFiles || !actions.root())) { fallback.click(); return; }
        const picked = row.kind === 'folder' ? picker({ mode: 'read', id: 'sbt-locate-asset' }) : picker({ multiple: false, types: [{ description: row.label, accept: { 'application/octet-stream': [row.key === 'anndata' ? '.h5ad' : '.csv'] } }] });
        actions.run(async () => {
          const selected = await picked.catch(error => { if (error.name !== 'AbortError') throw error; }); if (!selected) return;
          const handle = row.kind === 'folder' ? selected : selected[0];
          await ensurePending(); const root = session.root ?? actions.root();
          if (!root) throw new Error('Choose the common data folder first, or use this asset’s compatibility chooser.');
          const parts = await root.resolve(handle);
          if (!parts?.length) throw new Error(`Choose this asset inside ${root.name}, or change the data folder to a common parent.`);
          const path = parts.join('/'); session.setPath(row.key, path); await actions.path(row.key, path);
          await withWork(() => session.connect(root, { keys: [row.key], forceFull: true }));
        });
      };
      const compatibility = document.createElement('button'); compatibility.className = 'asset-compatibility'; compatibility.textContent = 'Compatibility chooser'; compatibility.onclick = () => fallback.click();
      buttons.append(locate, retry); heading.append(path); card.append(heading, detail, buttons, edit, compatibility, fallback);
      $('workspace-location-fields').append(card); nodes.set(row.key, { card, badge, path, detail, input });
    }
    if (payload.view.normalization.dictionary) {
      const note = document.createElement('p'); note.className = 'workspace-embedded'; note.textContent = '✓ Image normalisation dictionary is included in this workspace.'; $('workspace-location-fields').append(note);
    }
    render();
  }
  async function prepare(next, isPending = true, values = null) {
    session?.cancel(); const mine = ++revision; payload = next; pending = isPending; fallbackFiles = null;
    const cache = await actions.cache().catch(() => ({})); if (mine !== revision) return;
    session = new WorkspaceAssets(next.sources, { cache: cache ?? {}, mode: $('workspace-check-mode').value, changed(row) {
      if (mine !== revision) return;
      render(row);
      if (row.state === 'checking') status(`${session.connected} of ${session.rows.length} registered · ${row.label}: ${row.detail}`);
    } });
    if (!isPending) session.root = actions.root();
    build();
    if (values) for (const row of session.rows) if (values.has(row.key)) { try { await session.validate(row, values.get(row.key)); } catch (error) { session.update(row, 'error', error.message); } }
    status(isPending ? 'Choose the common data folder. Each saved asset is checked separately; missing items can be located below.' : `${session.connected} of ${session.rows.length} assets registered for this session.`);
  }
  async function ensurePending() {
    if (pending) return;
    const old = session, oldFallback = fallbackFiles, next = await actions.capture();
    await prepare(next, true, actions.values()); session.root = old?.root ?? (oldFallback ? null : actions.root()); fallbackFiles = oldFallback;
    actions.begin(next); $('workspace-locations').open = true;
  }
  $('workspace-check-mode').onchange = () => {
    if (session) session.mode = $('workspace-check-mode').value;
    $('workspace-check-help').textContent = $('workspace-check-mode').value === 'fast' ? 'Reuses a previously checked file list. Opens AnnData and dictionaries; checks images as used. New or changed locations get a full check.' : 'Checks all files in the saved asset folders. Recommended for shared or changed datasets.';
  };
  $('workspace-retry-assets').onclick = () => actions.run(async () => withWork(async () => { if (session.root) await session.connect(session.root); else await session.supply(fallbackFiles); }));
  $('workspace-open-connected').onclick = () => actions.run(() => withWork(async () => {}));
  $('workspace-root').onchange = event => {
    const files = Array.from(event.target.files); event.target.value = ''; if (!files.length || !session) return;
    actions.run(async () => {
      await ensurePending();
      fallbackFiles = files; session.root = null;
      for (const row of session.rows) { row.value = null; session.update(row, 'waiting', 'Data folder selected · waiting to check'); }
      await actions.compatibilityRoot(files[0].webkitRelativePath?.split('/')[0] || 'Selected folder');
      await withWork(() => session.supply(files));
    });
  };
  return {
    prepare, controls, status, get payload() { return payload; }, get pending() { return pending; },
    reset() { revision++; session?.cancel(); session = null; payload = null; pending = false; fallbackFiles = null; nodes.clear(); $('workspace-location-fields').replaceChildren(); $('workspace-asset-count').textContent = ''; },
    async connect(root, next) {
      if (!session || payload !== next) await prepare(next);
      pending = true; actions.begin(next); $('workspace-locations').open = true;
      await withWork(async () => { await session.connect(root); await actions.rememberRoot(root); });
    },
    get hasSession() { return Boolean(session); },
  };
}
