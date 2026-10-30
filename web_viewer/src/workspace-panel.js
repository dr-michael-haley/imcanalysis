import { WorkspaceLibrary, WorkspaceConflict, createAutosave, nextWorkspaceName } from './workspace-library.js';
import { createWorkspaceConnection } from './workspace-connection.js';
import { parseWorkspace } from './workspace.js';
import { assetPath, sourceLocations, referenceIdentity, resolvePickedPaths } from './workspace-paths.js';

export function createWorkspacePanel($, { capture, canSave, hasData, beginRestore, loadFiles, showSetup, assetValues = () => new Map(), assetHandles = () => new Map(), activated = () => {}, freeze = () => {}, rootBusy = () => false }) {
  const library = new WorkspaceLibrary();
  let active, records = [], available = false, loaded = false, busy = false, operation = false, lastPayload = '', queue = Promise.resolve(), requestedPersistence = false, pendingName = null, nameTimer;
  const locations = new Map(); let candidateRoot = null, fallbackRootName = '';
  function applyLocations(payload) {
    for (const item of sourceLocations(payload.sources)) {
      const saved = locations.get(item.key);
      if (saved?.identity === referenceIdentity(item.ref)) item.ref.path = assetPath(saved.path);
    }
    return payload;
  }
  async function showLocations() {
    try {
      if (connection.pending) return;
      const payload = loaded && hasData() ? await capture() : null;
      if (payload && JSON.stringify(payload.sources) !== JSON.stringify(connection.payload?.sources)) await connection.prepare(payload, false, assetValues());
    } catch (failure) { error(failure); }
  }
  $('workspace-locations').addEventListener('toggle', () => { if ($('workspace-locations').open) showLocations(); });
  const enqueue = task => { const result = queue.then(task); queue = result.catch(() => {}); return result; };
  const message = (text, kind = '', detail = text) => { $('local-save-status').textContent = text; $('local-save-status').title = detail; $('local-save-status').dataset.state = kind; };
  const error = value => { if (value && connection.pending) { connection.status('Connection needs attention.', value.message || String(value)); return; } $('local-workspace-error').textContent = value?.message || String(value || ''); $('local-workspace-error').hidden = !value; };
  const autosave = createAutosave({
    canSave: () => available && active && loaded && !busy && canSave(),
    save: async () => {
      const id = active.id, payload = await capture();
      if (id !== active?.id || !loaded) return;
      const text = JSON.stringify(payload);
      if (!requestedPersistence) { requestedPersistence = true; navigator.storage?.persist?.().catch(() => {}); }
      if (text !== lastPayload) await enqueue(async () => {
        const name = !active.payload && /^Untitled workspace(?: \d+)?$/.test(active.name) ? payload.sources.anndata.name.replace(/\.h5ad$/i, '').slice(0, 120) : active.name;
        active = await library.put({ ...active, name, payload: text, dataset: payload.sources.anndata.name }, active.revision);
        if (pendingName === null && document.activeElement !== $('local-workspace-name')) $('local-workspace-name').value = name;
        lastPayload = text; await refreshList();
      });
    },
    changed: (kind, failure) => {
      if (failure) { error(failure); message('Not saved', 'error', 'See the error message below.'); }
      else if (kind === 'pending') message('Saving…', 'pending');
      else if (kind === 'saved') message(`Saved locally · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`, 'saved');
      $('retry-local-save').hidden = !autosave.paused;
    },
  });
  function controls() {
    const locked = busy || operation || !available;
    for (const id of ['local-workspace-select', 'open-local-workspace', 'new-local-workspace', 'local-workspace-name', 'duplicate-local-workspace', 'delete-local-workspace']) $(id).disabled = locked || !active;
    $('connect-workspace-root').disabled = locked || !active || !loaded || !canSave();
    $('open-local-workspace').disabled ||= !records.some(record => record.id === $('local-workspace-select').value);
    $('local-workspace-current').textContent = active ? loaded ? 'Active' : 'Resume' : 'Name';
    const root = candidateRoot ?? active?.rootHandle;
    if (!connection.pending) $('root-connection').textContent = root?.name || (fallbackRootName ? `${fallbackRootName} · this session` : 'No folder linked');
    $('root-connection').title = root ? `Remembered read-only folder: ${root.name}` : 'Connect the common folder containing the saved assets.';
    $('connect-workspace-root').textContent = root ? 'Change folder' : 'Connect folder';
    $('connect-workspace-root').hidden = connection.pending || !window.showDirectoryPicker;
    for (const id of ['choose-workspace-root', 'workspace-root', 'workspace-file']) $(id).disabled = busy || operation || rootBusy();
    connection.controls(busy || operation || rootBusy());
  }
  async function refreshList(selection = active?.id) {
    records = await library.list();
    const previous = $('local-workspace-select').value;
    $('local-workspace-select').replaceChildren(...records.map(record => {
      const option = document.createElement('option'); option.value = record.id;
      option.textContent = `${record.name}${record.id === active?.id && loaded ? ' · active' : ''}${record.hasData ? '' : ' · new'}`;
      option.title = record.dataset ? `${record.dataset} · saved ${new Date(record.updatedAt).toLocaleString()}` : 'Choose local data to get started'; return option;
    }));
    $('local-workspace-select').value = records.some(record => record.id === previous) ? previous : selection;
    controls();
  }
  async function create(name, payload = null, rootHandle = null) {
    const record = await library.put({ id: crypto.randomUUID(), name: name.slice(0, 120), payload, rootHandle, createdAt: Date.now(), dataset: payload ? parseWorkspace(payload).sources.anndata.name : '' });
    await library.select(record.id); return record;
  }
  async function activate(record, isLoaded) {
    clearTimeout(nameTimer); pendingName = null;
    locations.clear(); candidateRoot = null; fallbackRootName = ''; connection.reset(); $('workspace-locations').open = false;
    if (record.payload) for (const item of sourceLocations(parseWorkspace(record.payload).sources)) if (item.ref.path) locations.set(item.key, { identity: referenceIdentity(item.ref), path: item.ref.path });
    active = record; loaded = isLoaded; lastPayload = record.payload || ''; autosave.reset();
    activated();
    await library.select(record.id); await refreshList(record.id); $('local-workspace-select').value = record.id;
    $('local-workspace-name').value = record.name; error(); controls();
  }
  async function flush() {
    await initialized;
    if (!available) return;
    await flushName();
    if (autosave.dirty && (!loaded || !canSave())) throw new Error('Wait for the current view to finish loading before switching workspaces.');
    if (autosave.paused) throw new Error('Your latest changes have not been saved. Retry saving or export JSON before switching workspaces.');
    do { await autosave.flush(); await queue; } while (autosave.dirty && canSave() && !autosave.paused);
    if (autosave.dirty) throw new Error('Wait for the current view to finish saving before switching workspaces.');
  }
  async function flushName() {
    clearTimeout(nameTimer); if (pendingName === null) return;
    const name = pendingName.trim(), id = active.id;
    if (!name) throw new Error('Give this workspace a name.');
    pendingName = null;
    try { await enqueue(async () => { if (active.id === id) active = await library.put({ ...active, name }, active.revision); }); }
    catch (failure) { if (pendingName === null) pendingName = name; message('Name not saved', 'error'); throw failure; }
    await refreshList(); message(autosave.dirty ? 'Saving…' : 'Name saved locally', autosave.dirty ? 'pending' : 'saved');
  }
  async function run(action) {
    if (operation) return;
    operation = true; freeze(true); controls();
    try { await initialized; await action(); }
    catch (failure) { error(failure); }
    finally { operation = false; busy = false; freeze(false); controls(); }
  }
  function mark() {
    if (available && active && loaded && !busy && hasData()) autosave.mark();
  }
  const connection = createWorkspaceConnection($, {
    run, capture, values: assetValues, begin: beginRestore, load: loadFiles,
    root: () => candidateRoot ?? active?.rootHandle,
    cache: () => available && active ? library.assetCache(active.id) : Promise.resolve({}),
    saveCache: cache => available && active ? library.cacheAssets(active.id, cache) : Promise.resolve(),
    async path(key, path) {
      const item = sourceLocations(connection.payload.sources).find(item => item.key === key);
      locations.set(key, { identity: referenceIdentity(item.ref), path });
      if (loaded && !connection.pending) mark();
      else if (active?.payload) await enqueue(async () => { active = await library.put({ ...active, payload: JSON.stringify(applyLocations(parseWorkspace(active.payload))) }, active.revision); });
    },
    async rememberRoot(handle) {
      candidateRoot = handle; fallbackRootName = ''; $('root-connection').textContent = handle.name;
      if (available && active) {
        await enqueue(async () => { active = await library.put({ ...active, rootHandle: handle, payload: JSON.stringify(applyLocations(connection.payload)) }, active.revision); });
        await refreshList();
      }
    },
    async compatibilityRoot(name) {
      candidateRoot = null; fallbackRootName = name; $('root-connection').textContent = `${name} · this session`;
      if (available && active?.rootHandle) await enqueue(async () => { active = await library.put({ ...active, rootHandle: null }, active.revision); });
    },
  });
  async function readRoot(handle, restore) {
    candidateRoot = handle; busy = true; controls(); error(); message('Connecting', 'pending');
    try {
      const payload = restore && connection.pending ? connection.payload : applyLocations(await capture());
      if (!restore) await resolvePickedPaths(handle, payload.sources, assetHandles(), true);
      for (const item of sourceLocations(payload.sources)) locations.set(item.key, { identity: referenceIdentity(item.ref), path: item.ref.path ?? item.ref.name });
      await connection.connect(handle, payload);
    } finally { busy = false; controls(); }
    mark();
  }
  async function resume(record) {
    if (!record?.payload) throw new Error('This workspace has no dataset yet. Open it using New workspace, then choose its files.');
    await flush(); busy = true; controls();
    await activate(record, false); const payload = parseWorkspace(record.payload);
    await connection.prepare(payload); beginRestore(payload); showSetup();
    message('Reconnect folder', 'pending', 'Reconnect the data folder to resume this workspace.');
    if (record.rootHandle) {
      let permission = await record.rootHandle.queryPermission({ mode: 'read' });
      if (permission !== 'granted') permission = await record.rootHandle.requestPermission({ mode: 'read' });
      if (permission === 'granted') await readRoot(record.rootHandle, true);
      else message('Choose data folder', 'pending', 'Folder access was not granted. Choose the data folder below to reconnect.');
    }
  }
  const initialized = (async () => {
    try {
      records = await library.list(); const last = await library.last();
      const record = await library.get(records.some(record => record.id === last) ? last : records[0]?.id || '__none__');
      available = true; await activate(record || await create('Untitled workspace'), !record?.payload);
      message(record?.payload ? 'Open to resume' : 'Autosave ready', '', record?.payload ? 'Choose Open to resume your saved view.' : 'Changes save automatically once your data is open.');
    } catch (failure) { available = false; error(failure); message('Saving unavailable', 'error', 'Use JSON export to keep your work.'); controls(); }
  })();

  $('local-workspace-select').addEventListener('change', controls);
  $('open-local-workspace').addEventListener('click', () => run(async () => {
    const id = $('local-workspace-select').value;
    if (id === active?.id && loaded) { showSetup(); return; }
    await flush(); const record = await library.get(id); if (!record) throw new WorkspaceConflict();
    if (!record.payload) { await activate(record, true); location.reload(); return; }
    await resume(record);
  }));
  $('new-local-workspace').addEventListener('click', () => run(async () => {
    await flush(); busy = true; controls(); await create(nextWorkspaceName(records)); location.reload();
  }));
  for (const type of ['input', 'change']) $('local-workspace-name').addEventListener(type, event => {
    pendingName = event.target.value; clearTimeout(nameTimer);
    nameTimer = setTimeout(() => { flushName().catch(error); }, type === 'change' ? 0 : 300);
  });
  $('duplicate-local-workspace').addEventListener('click', () => run(async () => {
    await flush(); busy = true; controls();
    const copy = await create(nextWorkspaceName(records, `${active.name.slice(0, 105)} copy`), active.payload, active.rootHandle);
    await activate(copy, loaded); message('Copy saved locally', 'saved');
  }));
  $('delete-local-workspace').addEventListener('click', () => {
    $('delete-workspace-confirm').hidden = false;
    $('delete-workspace-name').textContent = active.name;
  });
  $('keep-local-workspace').addEventListener('click', () => { $('delete-workspace-confirm').hidden = true; });
  $('confirm-delete-workspace').addEventListener('click', () => run(async () => {
    busy = true; clearTimeout(nameTimer); pendingName = null;
    await autosave.suspend(); await queue;
    try { await library.remove(active.id, active.revision); } catch (failure) { autosave.retry(); throw failure; }
    autosave.reset();
    active = null; loaded = false; location.reload();
  }));
  $('retry-local-save').addEventListener('click', () => { error(); autosave.retry(); });
  $('connect-workspace-root').addEventListener('click', () => {
    if (!window.showDirectoryPicker) { error(new Error('This browser cannot remember folder access. Reopen saved workspaces using the folder chooser when prompted.')); return; }
    // Start the picker directly in the gesture, before any async save or scan.
    const picked = window.showDirectoryPicker({ mode: 'read', id: 'sbt-workspace-root' });
    run(async () => {
      const handle = await picked.catch(failure => { if (failure.name !== 'AbortError') throw failure; }); if (!handle) return;
      await flush(); await readRoot(handle, false);
    });
  });
  $('choose-workspace-root').addEventListener('click', () => {
    if (!window.showDirectoryPicker) { $('workspace-root').click(); return; }
    const picked = window.showDirectoryPicker({ mode: 'read', id: 'sbt-workspace-root' });
    run(async () => {
      const handle = await picked.catch(failure => { if (failure.name !== 'AbortError') throw failure; }); if (handle) await readRoot(handle, true);
    });
  });
  $('connect-workspace-root').hidden = !window.showDirectoryPicker;
  // Native directory inputs remain available in every browser, including imports.
  $('workspace-root-fallback').hidden = false;
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') autosave.flush().catch(() => {}); });
  document.addEventListener('workspace-changed', mark);
  window.addEventListener('beforeunload', event => { if (autosave.dirty || pendingName !== null) { event.preventDefault(); event.returnValue = ''; } });
  for (const type of ['input', 'change', 'click', 'pointerup', 'wheel', 'keydown', 'toggle']) document.addEventListener(type, event => {
    if (event.target.closest?.('#workspace-controls, .brand-area')) return;
    if (type === 'keydown' && !['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    mark();
  }, type === 'toggle');
  window.addEventListener('focus', () => { if (available && !busy) refreshList().catch(error); });

  return {
    initialized, mark, flush, controls, applyLocations,
    async resolveLocations(payload) {
      if (active?.rootHandle) await resolvePickedPaths(active.rootHandle, payload.sources, assetHandles());
      for (const item of sourceLocations(payload.sources)) if (item.ref.path) locations.set(item.key, { identity: referenceIdentity(item.ref), path: item.ref.path });
    },
    async beforeDataset() {
      await initialized;
      if (!available) return;
      await flush();
      if (active.payload) await activate(await create(nextWorkspaceName(records)), true);
    },
    async import(payload, name) {
      freeze(true);
      try {
        await flush();
        if (available) await activate(await create(nextWorkspaceName(records, name), JSON.stringify(payload)), false);
        await connection.prepare(payload); beginRestore(payload); showSetup(); message('Choose data folder', 'pending', 'Workspace imported. Choose its data folder below.');
      } finally { freeze(false); }
    },
    async restored() {
      loaded = true; controls();
      if (available) { error(); autosave.mark(); }
    },
    cancel() { connection.reset(); autosave.reset(); loaded = false; location.reload(); },
    get name() { return pendingName?.trim() || active?.name || 'Workspace'; },
    get available() { return available; },
    get loaded() { return loaded; },
  };
}
