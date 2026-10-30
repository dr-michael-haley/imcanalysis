import { assetPath, sourceLocations } from './workspace-paths.js';
// Keep only workspace recipes and explicitly granted handles here, never file bytes.
export class WorkspaceConflict extends Error {
  constructor() { super('This workspace changed in another tab, or was deleted. Export your current view as JSON, then reload the page to reopen the saved version.'); this.name = 'WorkspaceConflict'; }
}

export function workspaceSummary(record) {
  return { id: record.id, name: record.name, revision: record.revision, createdAt: record.createdAt, updatedAt: record.updatedAt, hasData: Boolean(record.payload), dataset: record.dataset || '', hasRoot: Boolean(record.rootHandle) };
}
export function nextWorkspaceName(records, base = 'Untitled workspace') {
  const names = new Set(records.map(record => record.name)); let name = base.slice(0, 120), suffix = 2;
  while (names.has(name)) { const ending = ` ${suffix++}`; name = base.slice(0, 120 - ending.length) + ending; }
  return name;
}
export function checkRevision(current, expected) {
  if (expected === null ? Boolean(current) : !current || current.revision !== expected) throw new WorkspaceConflict();
}

export class WorkspaceLibrary {
  constructor(factory = globalThis.indexedDB, name = 'sbt-tissue-explorer') { this.factory = factory; this.name = name; }
  open() {
    return this.connection ??= new Promise((resolve, reject) => {
      if (!this.factory) { reject(new Error('Local workspace storage is unavailable in this browser. You can still import and export JSON files.')); return; }
      const request = this.factory.open(this.name, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        db.createObjectStore('workspaces', { keyPath: 'id' });
        db.createObjectStore('summaries', { keyPath: 'id' });
        db.createObjectStore('settings');
      };
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('Close other IMC Explorer tabs to finish opening local workspace storage.'));
      request.onsuccess = () => { request.result.onversionchange = () => request.result.close(); resolve(request.result); };
    });
  }
  async read(store, key) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(store), request = key === undefined ? tx.objectStore(store).getAll() : tx.objectStore(store).get(key);
      tx.oncomplete = () => resolve(request.result); tx.onabort = tx.onerror = () => reject(tx.error || request.error);
    });
  }
  list() { return this.read('summaries').then(records => records.sort((a, b) => b.updatedAt - a.updatedAt)); }
  get(id) { return this.read('workspaces', id); }
  last() { return this.read('settings', 'last'); }
  assetCache(id) { return this.read('settings', `asset-index:${id}`); }
  async cacheAssets(id, cache) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['workspaces', 'settings'], 'readwrite');
      tx.objectStore('workspaces').get(id).onsuccess = event => { if (event.target.result) tx.objectStore('settings').put(cache, `asset-index:${id}`); };
      tx.oncomplete = resolve; tx.onabort = tx.onerror = () => reject(tx.error);
    });
  }
  async select(id) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('settings', 'readwrite'); tx.objectStore('settings').put(id, 'last');
      tx.oncomplete = resolve; tx.onabort = tx.onerror = () => reject(tx.error);
    });
  }
  async put(record, expectedRevision = null) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['workspaces', 'summaries'], 'readwrite'), store = tx.objectStore('workspaces'); let saved, failure;
      store.get(record.id).onsuccess = event => {
        try {
          checkRevision(event.target.result, expectedRevision);
          saved = { ...record, revision: (expectedRevision ?? 0) + 1, updatedAt: Date.now() };
          store.put(saved); tx.objectStore('summaries').put(workspaceSummary(saved));
        } catch (error) { failure = error; tx.abort(); }
      };
      tx.oncomplete = () => resolve(saved); tx.onabort = tx.onerror = () => reject(failure || tx.error);
    });
  }
  async remove(id, revision) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['workspaces', 'summaries', 'settings'], 'readwrite'), store = tx.objectStore('workspaces'); let failure;
      store.get(id).onsuccess = event => {
        try { checkRevision(event.target.result, revision); store.delete(id); tx.objectStore('summaries').delete(id); tx.objectStore('settings').delete(`asset-index:${id}`); }
        catch (error) { failure = error; tx.abort(); }
      };
      tx.oncomplete = resolve; tx.onabort = tx.onerror = () => reject(failure || tx.error);
    });
  }
}

// getFile() obtains lazy File references. It does not read H5AD or pixel buffers.
export async function directoryFiles(root, sources, progress = () => {}, cancelled = () => false) {
  const files = new Map(); let seen = 0;
  const check = () => { if (cancelled()) throw new DOMException('Folder scan cancelled.', 'AbortError'); };
  async function add(handle, path) {
    check();
    if (files.has(path)) return;
    const file = await handle.getFile();
    Object.defineProperty(file, 'webkitRelativePath', { value: `${root.name}/${path}` }); files.set(path, file);
  }
  async function walk(directory, path, accepts) {
    for await (const [name, handle] of directory.entries()) {
      check();
      if (handle.kind === 'directory') await walk(handle, `${path}/${name}`, accepts);
      else if (accepts({ name })) await add(handle, `${path}/${name}`);
      if (++seen % 100 === 0) { progress(`Reading saved asset folders · ${files.size.toLocaleString()} files…`); await new Promise(resolve => setTimeout(resolve, 0)); }
    }
  }
  for (const item of sourceLocations(sources)) {
    check();
    const path = assetPath(item.ref.path ?? item.ref.name), parts = path.split('/');
    progress(`Opening ${item.label} · ${path}…`);
    try {
      let directory = root;
      for (const name of parts.slice(0, -1)) { check(); directory = await directory.getDirectoryHandle(name); }
      check();
      if (item.kind === 'file') await add(await directory.getFileHandle(parts.at(-1)), path);
      else await walk(await directory.getDirectoryHandle(parts.at(-1)), path, item.accepts);
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      throw new Error(`Could not open ${item.label} at “${path}”. Check Asset locations relative to “${root.name}”, then choose the folder again. ${error.message}`);
    }
  }
  return [...files.values()];
}

// Serialises saves, and retains edits made while an earlier write is in flight.
export function createAutosave({ save, canSave, changed = () => {}, delay = 700, retry = 600 }) {
  let revision = 0, saved = 0, timer, pending, paused = false, started = 0;
  const schedule = () => { clearTimeout(timer); if (!paused && revision !== saved) timer = setTimeout(() => { flush().catch(() => {}); }, Math.min(delay, Math.max(0, 2500 - (Date.now() - started)))); };
  async function flush() {
    clearTimeout(timer);
    if (pending) { await pending; if (revision !== saved && !paused) return flush(); return; }
    if (paused || revision === saved) return;
    if (!canSave()) { timer = setTimeout(() => { flush().catch(() => {}); }, retry); return; }
    started = Date.now();
    const target = revision;
    pending = Promise.resolve().then(save).then(() => { saved = target; changed(revision !== saved ? 'pending' : 'saved'); }).catch(error => { paused = true; changed('error', error); throw error; });
    try { await pending; } finally { pending = null; schedule(); }
  }
  return {
    mark() { if (revision === saved) started = Date.now(); revision++; changed(paused ? 'error' : 'pending'); schedule(); }, flush,
    reset() { clearTimeout(timer); revision = saved = 0; paused = false; started = 0; },
    async suspend() { paused = true; clearTimeout(timer); await pending?.catch(() => {}); },
    retry() { paused = false; schedule(); },
    get dirty() { return revision !== saved; }, get paused() { return paused; },
  };
}
