import test from 'node:test';
import assert from 'node:assert/strict';
import { WorkspaceAssets } from '../src/workspace-assets.js';
import { fileReference, folderReference } from '../src/workspace.js';
import { materializeAssetFile } from '../src/asset-files.js';

async function fixture() {
  const reads = [], lists = [], opens = [];
  const contents = { 'analysis/a.h5ad': 'anndata', 'Images/ROI/DNA.tiff': 'pixels', 'Masks/ROI.tiff': 'mask', 'Labels/ROI.tiff': 'labels', 'dict/labels.csv': 'original_num,name\n0,Background\n1,Tumour' };
  const file = path => {
    if (!(path in contents)) throw new DOMException('Missing', 'NotFoundError');
    const f = new File([contents[path]], path.split('/').at(-1), { lastModified: 123 });
    Object.defineProperty(f, 'webkitRelativePath', { value: `data/${path}` }); return f;
  };
  const dir = (path = '') => ({ name: path.split('/').at(-1) || 'data', kind: 'directory',
    async getDirectoryHandle(name) {
      const next = path ? `${path}/${name}` : name; opens.push(next);
      if (!Object.keys(contents).some(key => key.startsWith(`${next}/`))) throw new DOMException('Missing', 'NotFoundError');
      return dir(next);
    },
    async getFileHandle(name) {
      const next = path ? `${path}/${name}` : name; opens.push(next);
      if (!(next in contents)) throw new DOMException('Missing', 'NotFoundError');
      return { kind: 'file', async getFile() { reads.push(next); const source = file(next); return new File([source], source.name, { lastModified: source.lastModified }); } };
    },
    async *entries() {
      assert.ok(path, 'Never list the common root'); lists.push(path);
      const names = new Set(Object.keys(contents).filter(key => key.startsWith(`${path}/`)).map(key => key.slice(path.length + 1).split('/')[0]));
      for (const name of names) yield [name, `${path}/${name}` in contents ? await this.getFileHandle(name) : await this.getDirectoryHandle(name)];
    },
  });
  const folder = async path => {
    const files = Object.keys(contents).filter(key => key.startsWith(`${path}/`)).map(key => {
      const f = new File([contents[key]], key.split('/').at(-1)); Object.defineProperty(f, 'webkitRelativePath', { value: key }); return f;
    }); return { ...await folderReference(files), path };
  };
  const sources = { anndata: { ...fileReference(file('analysis/a.h5ad')), path: 'analysis/a.h5ad' }, masks: await folder('Masks'), images: await folder('Images'), layers: [{ id: 'labels', name: 'Annotations', kind: 'labels', folder: await folder('Labels'), dictionary: { file: { ...fileReference(file('dict/labels.csv')), path: 'dict/labels.csv' }, columns: { value: 'original_num', label: 'name', color: '' } } }] };
  return { sources, reads, lists, opens, contents, root: dir(), files: () => Object.keys(contents).map(file) };
}

test('full connection continues past missing assets, then retries only the repaired dictionary', async () => {
  const f = await fixture(); f.sources.layers[0].dictionary.file.path = 'missing/labels.csv'; const events = [];
  const session = new WorkspaceAssets(f.sources, { changed: row => events.push([row.key, row.state]) });
  assert.equal(await session.connect(f.root), false); assert.equal(session.connected, 4);
  assert.equal(session.rows.at(-1).state, 'error'); assert.match(session.rows.at(-1).detail, /Locate/);
  assert.ok(events.some(([key, state]) => key === 'images' && state === 'checking'));
  const reads = f.reads.length, lists = f.lists.length;
  session.setPath('dictionary:labels', 'dict/labels.csv');
  assert.equal(await session.connect(f.root), true);
  assert.equal(f.lists.length, lists); assert.deepEqual(f.reads.slice(reads), ['dict/labels.csv']);
  assert.equal(session.result().layers[0].mapping[1].label, 'Tumour');
  assert.equal(session.sources.layers[0].dictionary.file.path, 'dict/labels.csv');
});

test('quick reconnect checks exact folders without listing them or opening pixels until used', async () => {
  const f = await fixture(), full = new WorkspaceAssets(f.sources); await full.connect(f.root);
  f.reads.length = 0; f.lists.length = 0;
  const fast = new WorkspaceAssets(f.sources, { mode: 'fast', cache: full.cache }); await fast.connect(f.root);
  assert.equal(fast.complete, true); assert.deepEqual(f.lists, []);
  assert.deepEqual(f.reads, ['analysis/a.h5ad', 'dict/labels.csv']);
  assert.equal(fast.rows.find(row => row.key === 'images').state, 'cached');
  const image = fast.result().images[0]; assert.equal(image.name, 'DNA.tiff');
  const actual = await materializeAssetFile(image); assert.ok(actual instanceof File); assert.equal(await actual.text(), 'pixels');
  await materializeAssetFile(image); assert.equal(f.reads.filter(path => path.endsWith('DNA.tiff')).length, 1);
});

test('quick reconnect without a matching cached path falls back to full checking', async () => {
  const f = await fixture(), first = new WorkspaceAssets(f.sources); await first.connect(f.root);
  first.cache.images.path = 'OldImages'; f.lists.length = 0;
  const next = new WorkspaceAssets(f.sources, { cache: first.cache, mode: 'fast' }); await next.connect(f.root);
  assert.deepEqual(f.lists, ['Images', 'Images/ROI']); assert.equal(next.rows.find(row => row.key === 'images').state, 'ready');
  const fresh = new WorkspaceAssets(f.sources, { mode: 'fast' }); await fresh.connect(f.root); assert.ok(fresh.rows.every(row => row.state === 'ready'));
});

test('cached assets report changed or missing files on use and remain retryable', async () => {
  const f = await fixture(), full = new WorkspaceAssets(f.sources); await full.connect(f.root);
  const fast = new WorkspaceAssets(f.sources, { mode: 'fast', cache: full.cache }); await fast.connect(f.root);
  const value = fast.result().images[0]; f.contents['Images/ROI/DNA.tiff'] = 'changed file';
  await assert.rejects(materializeAssetFile(value), /size changed/); assert.equal(fast.complete, false);
  f.contents['Images/ROI/DNA.tiff'] = 'pixels';
  await fast.connect(f.root, { keys: ['images'], forceFull: true }); assert.equal(fast.complete, true);
  const second = new WorkspaceAssets(f.sources, { mode: 'fast', cache: full.cache }); await second.connect(f.root);
  const vanished = second.result().images[0]; delete f.contents['Images/ROI/DNA.tiff'];
  await assert.rejects(materializeAssetFile(vanished), /Not found/);
});

test('compatibility folder import supports nested assets and direct dictionary repair', async () => {
  const f = await fixture(); f.sources.layers[0].dictionary.file.path = 'wrong/labels.csv';
  const session = new WorkspaceAssets(f.sources); await session.supply(f.files());
  assert.equal(session.connected, 4); assert.throws(() => session.result(), /Reconnect every/);
  await session.supply([f.files().find(file => file.name === 'labels.csv')], 'dictionary:labels');
  assert.equal(session.complete, true); assert.equal(session.result().layers[0].dictionary.file.name, 'labels.csv');
  assert.equal(session.cache.images.files[0][0], 'ROI/DNA.tiff');
});

test('wrong folder inventory and dictionary columns cannot show as registered', async () => {
  const f = await fixture(); f.contents['Images/ROI/DNA.tiff'] += 'changed'; f.sources.layers[0].dictionary.columns.label = 'absent';
  const session = new WorkspaceAssets(f.sources); await session.connect(f.root);
  assert.equal(session.rows.find(row => row.key === 'images').state, 'error');
  assert.equal(session.rows.at(-1).state, 'error'); assert.equal(session.connected, 3);
});

test('cancelling an asset check stops subsequent reads and status updates', async () => {
  const f = await fixture(); let session;
  session = new WorkspaceAssets(f.sources, { changed: row => { if (row.key === 'masks' && row.state === 'checking') session.cancel(); } });
  await assert.rejects(session.connect(f.root), { name: 'AbortError' });
  assert.deepEqual(f.reads, ['analysis/a.h5ad']);
});
