// Copy alongside workspace-library.js into an isolated local test web root.
import { WorkspaceLibrary, WorkspaceConflict, directoryFiles } from './workspace-library.js';
document.querySelector('#run').onclick = async () => {
  const output = document.querySelector('#result'); output.textContent = 'Running…';
  const name = `sbt-persistence-test-${crypto.randomUUID()}`, db = new WorkspaceLibrary(indexedDB, name);
  const assert = (ok, message) => { if (!ok) throw new Error(message); output.textContent += `\nPASS ${message}`; };
  let root;
  try {
    const first = await db.put({ id: 'review', name: 'Review A', payload: 'original', dataset: 'synthetic.h5ad', createdAt: Date.now() });
    await db.select(first.id);
    assert(await db.last() === 'review', 'last workspace persists');
    assert((await db.list())[0].hasData, 'recent-workspace summary commits with its record');
    const writes = await Promise.allSettled([db.put({ ...first, payload: 'tab A' }, first.revision), db.put({ ...first, payload: 'tab B' }, first.revision)]);
    assert(writes.filter(result => result.status === 'fulfilled').length === 1 && writes.find(result => result.status === 'rejected').reason instanceof WorkspaceConflict, 'simultaneous tabs cannot overwrite one another');
    const current = await db.get('review');
    assert((await db.list())[0].revision === current.revision, 'record and summary retain the same revision');
    const reopened = new WorkspaceLibrary(indexedDB, name);
    assert((await reopened.get('review')).payload === current.payload, 'new database connection reopens the committed payload');
    root = await navigator.storage.getDirectory();
    const folder = await root.getDirectoryHandle(name, { create: true });
    const file = await folder.getFileHandle('synthetic.h5ad', { create: true }), writer = await file.createWritable();
    await writer.write('synthetic test only'); await writer.close();
    const withRoot = await db.put({ ...current, rootHandle: folder }, current.revision);
    const restored = await reopened.get('review');
    assert(await restored.rootHandle.isSameEntry(folder), 'directory handles survive IndexedDB structured cloning');
    const files = await directoryFiles(restored.rootHandle, { anndata: { name: 'synthetic.h5ad' } });
    assert(files.length === 1 && files[0].webkitRelativePath === `${name}/synthetic.h5ad`, 'restored directory handles enumerate usable file references');
    assert(await files[0].text() === 'synthetic test only', 'file references read the original bytes on demand');
    await db.put({ ...withRoot, id: 'copy', name: 'Review B' });
    await db.put({ ...withRoot, payload: 'edited original' }, withRoot.revision);
    assert((await db.get('copy')).payload === current.payload, 'duplicate workspaces keep independent payloads');
    const latest = await db.get('review'); await db.remove('review', latest.revision);
    assert(!await db.get('review') && !(await db.list()).some(item => item.id === 'review'), 'deletion removes record and summary atomically');
    let rejected = false; try { await db.put(latest, latest.revision); } catch (error) { rejected = error instanceof WorkspaceConflict; }
    assert(rejected, 'stale saves cannot resurrect a deleted workspace');
    (await reopened.open()).close(); output.textContent += '\nAll 11 browser persistence checks passed.';
  } catch (error) { output.textContent += `\nFAIL ${error.name}: ${error.message}`; }
  finally {
    (await db.open()).close(); indexedDB.deleteDatabase(name);
    if (root) await root.removeEntry(name, { recursive: true }).catch(() => {});
  }
};
