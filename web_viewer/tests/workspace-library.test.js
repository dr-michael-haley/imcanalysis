import test from 'node:test';
import assert from 'node:assert/strict';
import { WorkspaceConflict, WorkspaceLibrary, workspaceSummary, nextWorkspaceName, checkRevision, createAutosave, directoryFiles } from '../src/workspace-library.js';

test('workspace revisions reject stale tabs and deleted records instead of recreating them', () => {
  checkRevision(undefined, null); checkRevision({ revision: 3 }, 3);
  assert.throws(() => checkRevision({ revision: 4 }, 3), WorkspaceConflict);
  assert.throws(() => checkRevision(undefined, 3), WorkspaceConflict);
  assert.throws(() => checkRevision({ revision: 1 }, null), WorkspaceConflict);
});
test('recent workspace summaries contain no payload or file handles and copies get distinct names', () => {
  const record = { id: 'one', name: 'Review', revision: 2, createdAt: 1, updatedAt: 2, dataset: 'a.h5ad', payload: 'private notes', rootHandle: { name: 'data' } };
  const summary = workspaceSummary(record);
  assert.equal(summary.hasData, true); assert.equal(summary.hasRoot, true);
  assert.equal('payload' in summary, false); assert.equal('rootHandle' in summary, false);
  assert.equal(nextWorkspaceName([{ name: 'Review' }, { name: 'Review 2' }], 'Review'), 'Review 3');
  assert.equal(nextWorkspaceName([{ name: 'x'.repeat(120) }], 'x'.repeat(120)), `${'x'.repeat(118)} 2`);
});
test('unsupported local storage fails explicitly without pretending it saved', async () => {
  await assert.rejects(new WorkspaceLibrary(null).open(), /unavailable/);
});
test('autosave retains edits that arrive during an in-flight write', async () => {
  let release, writes = 0;
  const saver = createAutosave({ canSave: () => true, delay: 5000, save: () => { writes++; return writes === 1 ? new Promise(resolve => { release = resolve; }) : Promise.resolve(); } });
  saver.mark(); const first = saver.flush(); await Promise.resolve();
  saver.mark(); release(); await first;
  assert.equal(saver.dirty, true);
  await saver.flush(); assert.equal(writes, 2); assert.equal(saver.dirty, false); saver.reset();
});
test('autosave waits for an ROI to finish loading, then saves the final state', async () => {
  let ready = false, writes = 0;
  const saver = createAutosave({ canSave: () => ready, delay: 5000, retry: 5000, save: () => { writes++; } });
  saver.mark(); await saver.flush(); assert.equal(writes, 0); assert.equal(saver.dirty, true);
  ready = true; await saver.flush(); assert.equal(writes, 1); assert.equal(saver.dirty, false); saver.reset();
});
test('failed writes keep dirty state, report errors, and resume only on retry', async () => {
  let fail = true, last;
  const saver = createAutosave({ canSave: () => true, delay: 5000, save: () => { if (fail) throw new Error('Storage full'); }, changed: (kind, error) => { last = [kind, error]; } });
  saver.mark(); await assert.rejects(saver.flush(), /Storage full/);
  assert.equal(saver.paused, true); assert.equal(saver.dirty, true); assert.equal(last[0], 'error');
  fail = false; saver.retry(); await saver.flush(); assert.equal(saver.dirty, false); assert.equal(last[0], 'saved'); saver.reset();
});
test('suspending before deletion waits for any in-flight save without starting another', async () => {
  let release, writes = 0;
  const saver = createAutosave({ canSave: () => true, delay: 5000, save: () => { writes++; return new Promise(resolve => { release = resolve; }); } });
  saver.mark(); const write = saver.flush(); await Promise.resolve(); saver.mark();
  let settled = false; const stopped = saver.suspend().then(() => { settled = true; });
  await Promise.resolve(); assert.equal(settled, false); release(); await Promise.all([write, stopped]);
  assert.equal(saver.paused, true); assert.equal(writes, 1); saver.reset(); assert.equal(saver.dirty, false);
});
test('reconnecting opens exact saved paths without listing the root, siblings or unrelated file metadata', async () => {
  const reads = [], listed = [];
  const file = name => ({ kind: 'file', async getFile() { reads.push(name); const value = new File(['not read'], name); value.arrayBuffer = () => { throw new Error('Do not read image bytes'); }; return value; } });
  const directory = (name, entries, mayList = true) => ({ kind: 'directory', name,
    async getDirectoryHandle(key) { const value = entries.find(([name, handle]) => name === key && handle.kind === 'directory')?.[1]; if (!value) throw new Error('Not found'); return value; },
    async getFileHandle(key) { const value = entries.find(([name, handle]) => name === key && handle.kind === 'file')?.[1]; if (!value) throw new Error('Not found'); return value; },
    async *entries() { assert.ok(mayList, `Must not list ${name}`); listed.push(name); yield* entries; },
  });
  const root = directory('dataset', [['archive', directory('archive', [], false)], ['analysis', directory('analysis', [['a.h5ad', file('a.h5ad')], ['other.h5ad', file('other.h5ad')]], false)], ['Images', directory('Images', [['notes.csv', file('notes.csv')], ['ROI', directory('ROI', [['DNA3.tiff', file('DNA3.tiff')]])]])]], false);
  const sources = { anndata: { name: 'a.h5ad', path: 'analysis/a.h5ad' }, images: { name: 'Images' }, layers: [] };
  const files = await directoryFiles(root, sources);
  assert.deepEqual(reads, ['a.h5ad', 'DNA3.tiff']); assert.deepEqual(listed, ['Images', 'ROI']);
  assert.deepEqual(files.map(file => file.webkitRelativePath), ['dataset/analysis/a.h5ad', 'dataset/Images/ROI/DNA3.tiff']);
  await assert.rejects(directoryFiles(root, sources, () => {}, () => true), { name: 'AbortError' });
  await assert.rejects(directoryFiles(root, { anndata: { name: 'missing.h5ad' } }), /Asset locations/);
  await assert.rejects(directoryFiles(root, { anndata: { name: 'a.h5ad', path: '../a.h5ad' } }), /relative path/);
});
