import { sourceLocations, assetPath, referenceIdentity } from './workspace-paths.js';
import { folderReference, fileReference, sameFile, sameFolder, relativeFilePath } from './workspace.js';
import { labelDictionary, parseLabelMapping } from './aligned-layers.js';
import { deferredAssetFile, materializeAssetFile } from './asset-files.js';

const ready = row => ['ready', 'cached'].includes(row.state);
const abort = () => new DOMException('Asset check cancelled.', 'AbortError');
const pause = () => new Promise(resolve => setTimeout(resolve, 0));
const missing = error => error.name === 'NotFoundError' ? 'Not found at this location. Locate it or edit the relative path.' : error.name === 'NotAllowedError' ? 'Folder access was not granted. Choose the data folder again.' : error.message;

export class WorkspaceAssets {
  constructor(sources, { cache = {}, mode = 'full', changed = () => {} } = {}) {
    this.sources = sources; this.cache = { ...cache }; this.mode = mode; this.changed = changed; this.root = null; this.cancelled = false;
    this.rows = sourceLocations(sources).map(item => ({ ...item, path: assetPath(item.ref.path ?? item.ref.name), state: 'waiting', detail: 'Waiting to check', value: null }));
  }
  get complete() { return this.rows.every(ready); }
  get connected() { return this.rows.filter(ready).length; }
  check() { if (this.cancelled) throw abort(); }
  cancel() { this.cancelled = true; }
  update(row, state, detail) { this.check(); row.state = state; row.detail = detail; this.changed(row, this); }
  setPath(key, path) {
    const row = this.rows.find(row => row.key === key); path = assetPath(path);
    row.path = path; row.ref.path = path; row.value = null; this.update(row, 'waiting', 'Location changed · check this asset');
  }
  async openPath(root, path, kind) {
    const parts = assetPath(path).split('/'); let directory = root;
    for (const part of parts.slice(0, -1)) { this.check(); directory = await directory.getDirectoryHandle(part); }
    this.check(); return kind === 'folder' ? directory.getDirectoryHandle(parts.at(-1)) : directory.getFileHandle(parts.at(-1));
  }
  async validate(row, value, cached = false) {
    this.check();
    if (row.kind === 'file') {
      if (!sameFile(row.ref, fileReference(value))) throw new Error(`File differs from the saved reference. Expected ${row.ref.name} (${row.ref.size.toLocaleString()} bytes); selected ${value.name} (${value.size.toLocaleString()} bytes).`);
      if (row.key.startsWith('dictionary:')) {
        const layer = this.sources.layers.find(layer => `dictionary:${layer.id}` === row.key);
        const text = await value.text(), parsed = labelDictionary(text, layer.dictionary.columns);
        row.dictionary = { text, ...parsed };
      }
    } else {
      const actual = await folderReference(value);
      if (!sameFolder(row.ref, actual)) throw new Error(`Folder differs from the saved file list: expected ${row.ref.count.toLocaleString()} files, found ${value.length.toLocaleString()}. Choose the matching folder.`);
      const mappings = value.filter(file => file.name.toLowerCase() === 'label_mapping.csv');
      if (mappings.length > 1 || mappings[0]?.size > 2 * 1024 * 1024) throw new Error('Choose one label_mapping.csv smaller than 2 MB.');
      row.mapping = mappings.length ? parseLabelMapping(await (await materializeAssetFile(mappings[0])).text()) : [];
    }
    this.check(); row.value = value; row.ref.path = row.path;
    this.update(row, cached ? 'cached' : 'ready', row.kind === 'file' ? row.dictionary ? `${row.dictionary.mapping.length.toLocaleString()} label entries · dictionary read` : 'File found · name and size match' : `${value.length.toLocaleString()} files · ${cached ? 'saved list; checked when used' : 'file list and sizes match'}`);
  }
  async scanFolder(row, handle) {
    const files = []; let seen = 0;
    const walk = async (directory, prefix) => {
      for await (const [name, child] of directory.entries()) {
        this.check();
        if (child.kind === 'directory') await walk(child, `${prefix}/${name}`);
        else if (row.accepts({ name })) {
          const file = await child.getFile(); this.check();
          Object.defineProperty(file, 'webkitRelativePath', { value: `${prefix}/${name}` }); files.push(file);
        }
        if (++seen % 50 === 0) { this.update(row, 'checking', `Checking files · ${files.length.toLocaleString()} / ${row.ref.count.toLocaleString()} expected`); await pause(); }
      }
    };
    await walk(handle, handle.name); files.workspaceHandle = handle; files.workspacePath = row.path;
    return files;
  }
  async cachedFolder(row, handle, cached) {
    const directories = new Map([['', Promise.resolve(handle)]]);
    const openDirectory = path => {
      if (!directories.has(path)) {
        const parts = path.split('/'), name = parts.pop();
        directories.set(path, openDirectory(parts.join('/')).then(dir => dir.getDirectoryHandle(name)));
      }
      return directories.get(path);
    };
    const files = cached.files.map(([path, size, lastModified]) => {
      assetPath(path); const parts = path.split('/'), name = parts.pop();
      return deferredAssetFile({ name, size, lastModified, type: '', webkitRelativePath: `${handle.name}/${path}` }, async () => {
        try {
          const file = await (await (await openDirectory(parts.join('/'))).getFileHandle(name)).getFile();
          if (file.size !== size) throw new Error('File size changed since the last full check.');
          return file;
        } catch (error) {
          const message = `${row.label}: ${path}. ${missing(error)} Reconnect this asset with a full check.`;
          if (!this.cancelled) this.update(row, 'error', message);
          throw new Error(message);
        }
      });
    });
    files.workspaceHandle = handle; files.workspacePath = row.path; return files;
  }
  async nativeRow(row, forceFull = false) {
    this.update(row, 'checking', `Opening ${row.path}…`); await pause();
    try {
      const handle = await this.openPath(this.root, row.path, row.kind); this.check();
      if (row.kind === 'file') {
        const file = await handle.getFile(); file.workspaceHandle = handle;
        Object.defineProperty(file, 'webkitRelativePath', { value: `${this.root.name}/${row.path}` });
        await this.validate(row, file);
      } else {
        const cache = this.cache[row.key];
        const fast = !forceFull && this.mode === 'fast' && cache?.identity === referenceIdentity(row.ref) && cache.path === row.path && Array.isArray(cache.files) && cache.files.length === row.ref.count;
        if (this.mode === 'fast' && !fast) this.update(row, 'checking', 'No matching saved list · performing the first full check');
        const files = fast ? await this.cachedFolder(row, handle, cache) : await this.scanFolder(row, handle);
        await this.validate(row, files, fast);
        if (!fast) this.cache[row.key] = { identity: referenceIdentity(row.ref), path: row.path, checkedAt: Date.now(), files: files.map(file => [relativeFilePath(file), file.size, file.lastModified]) };
      }
    } catch (error) { this.check(); if (error.name === 'AbortError') throw error; row.value = null; this.update(row, 'error', missing(error)); }
  }
  async connect(root, { keys = null, forceFull = false } = {}) {
    if (this.root && this.root !== root) for (const row of this.rows) { row.value = null; this.update(row, 'waiting', 'Data folder changed · waiting to check'); }
    this.root = root;
    for (const row of this.rows) if (keys ? keys.includes(row.key) : !ready(row)) await this.nativeRow(row, forceFull);
    return this.complete;
  }
  async suppliedRow(row, files, direct = false) {
    this.update(row, 'checking', 'Checking selected files…'); await pause();
    try {
      if (row.kind === 'file') {
        const choices = direct ? files : files.filter(file => relativeFilePath(file) === row.path);
        if (choices.length !== 1) throw new Error('File not found at this location. Use Locate file to select it.');
        await this.validate(row, choices[0]);
      } else {
        const prefix = `${row.path}/`;
        const chosen = files.filter(file => row.accepts(file) && (direct || relativeFilePath(file).startsWith(prefix)));
        const scoped = chosen.map(file => {
          const copy = new File([file], file.name, { type: file.type, lastModified: file.lastModified });
          const path = direct ? file.webkitRelativePath || `${row.path.split('/').at(-1)}/${file.name}` : `${row.path.split('/').at(-1)}/${relativeFilePath(file).slice(prefix.length)}`;
          Object.defineProperty(copy, 'webkitRelativePath', { value: path }); return copy;
        });
        scoped.workspacePath = row.path; await this.validate(row, scoped);
        this.cache[row.key] = { identity: referenceIdentity(row.ref), path: row.path, checkedAt: Date.now(), files: scoped.map(file => [relativeFilePath(file), file.size, file.lastModified]) };
      }
    } catch (error) { this.check(); row.value = null; this.update(row, 'error', missing(error)); }
  }
  async supply(files, key = null) {
    for (const row of this.rows) if (key ? key === row.key : !ready(row)) await this.suppliedRow(row, files, Boolean(key));
    return this.complete;
  }
  result() {
    if (!this.complete) throw new Error('Reconnect every listed asset before opening this workspace.');
    const get = key => this.rows.find(row => row.key === key);
    return { anndata: get('anndata').value, masks: get('masks')?.value ?? [], images: get('images')?.value ?? [], layers: (this.sources.layers ?? []).map(layer => {
      const folder = get(`layer:${layer.id}`), dictionary = get(`dictionary:${layer.id}`);
      return { ...layer, files: folder.value, mapping: dictionary?.dictionary.mapping ?? folder.mapping ?? [], dictionary: dictionary ? { file: dictionary.value, columns: dictionary.dictionary.columns, headers: dictionary.dictionary.headers } : null, dictionaryText: dictionary?.dictionary.text };
    }) };
  }
}
