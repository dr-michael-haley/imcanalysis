import { discoverFiles, filePath, stem } from './core.js';
import { assetPath } from './workspace-paths.js';
import { WorkspaceLibrary } from './workspace-library.js';

const library = new WorkspaceLibrary();
export async function readJson(folder, name, fallback) {
  try {
    return JSON.parse(await (await (await folder.getFileHandle(name)).getFile()).text());
  } catch (error) {
    if (error.name === 'NotFoundError' && fallback !== undefined) return fallback;
    throw error;
  }
}
export async function writeFile(folder, name, value) {
  const handle = await folder.getFileHandle(name, { create: true }),
    out = await handle.createWritable();
  try {
    await out.write(value);
    await out.close();
  } catch (error) {
    await out.abort().catch(() => {});
    throw error;
  }
}
export function writeJson(folder, name, value) {
  return writeFile(folder, name, JSON.stringify(value));
}
export const blankCatalog = () => ({ version: 1, revision: 0, sources: [], analyses: [] });
export const blankRun = () => ({
  version: 1,
  revision: 0,
  labels: {},
  threshold: 0.5,
  model: null,
  generation: 0
});

// Run and project metadata have independent revisions. Only consult the small
// file being updated, and serialize the check and commit across browser tabs.
export async function commit(folder, file, value, projectId) {
  const save = async () => {
    const disk = await readJson(folder, file, null);
    if ((disk?.revision ?? 0) !== value.revision)
      throw Error(
        'Another tab changed this analysis project. Choose Reload analysis list to load the saved changes.'
      );
    const next = { ...value, revision: value.revision + 1 };
    await writeJson(folder, file, next);
    return next;
  };
  return navigator.locks.request('imc-analysis-write-' + projectId, save);
}

export function sourceInventory(masks, images) {
  if (!masks.length || !images.length) throw Error('Choose the masks and images folders in Setup first.');
  const map = discoverFiles(masks, images, [...new Set(masks.map(stem))], new Map());
  const channels = new Set();
  for (const entry of map.values()) for (const image of entry.images) channels.add(image.name);
  if (!channels.size)
    throw Error(
      'No image matches a mask ROI. Use ROI-named mask TIFFs and ROI image subfolders or ROI-prefixed image names.'
    );
  return { map, channels: [...channels].sort() };
}
export function relativeInput(file) {
  const parts = filePath(file).replaceAll('\\', '/').split('/');
  return assetPath(parts.length > 1 ? parts.slice(1).join('/') : parts[0]);
}
export function buildSource(inventory, channel, folders, metrics) {
  const rois = [];
  for (const [name, entry] of inventory.map) {
    const image = entry.images.find(item => item.name === channel);
    rois.push({
      name,
      mask: relativeInput(entry.masks[0]),
      dna: image ? relativeInput(image.file) : null,
      images: entry.images.map(item => relativeInput(item.file))
    });
  }
  rois.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const firstFolder = files => files.workspaceHandle?.name || filePath(files[0]).split('/')[0];
  return {
    id: crypto.randomUUID(),
    recipe: 'mask-dna-v1',
    channel,
    rois,
    maskFolder: firstFolder(folders.masks),
    imageFolder: firstFolder(folders.images),
    metrics: metrics ? { name: metrics.name, columns: metrics.columns } : null
  };
}
export async function inputFile(root, path) {
  const parts = assetPath(path).split('/'),
    name = parts.pop();
  let dir = root;
  for (const part of parts) dir = await dir.getDirectoryHandle(part);
  return (await dir.getFileHandle(name)).getFile();
}
async function writeSetting(key, value) {
  const db = await library.open();
  await new Promise((resolve, reject) => {
    const tx = db.transaction('settings', 'readwrite');
    tx.objectStore('settings').put(value, key);
    tx.oncomplete = resolve;
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}
export const rememberInputs = (id, handles) => writeSetting('analysis-inputs:' + id, handles);
export const savedInputs = id => library.read('settings', 'analysis-inputs:' + id);
export const rememberAnalysisNavigation = (id, navigation) =>
  writeSetting('analysis-navigation:' + id, navigation);
export const savedAnalysisNavigation = id => library.read('settings', 'analysis-navigation:' + id);
export const savedDraft = id => library.read('settings', 'analysis-draft:' + id);

// One job per ROI. Stop admitting new jobs after a failure and wait for all
// in-flight writes before returning control to the workspace.
export async function mapROIs(items, workers, job) {
  let next = 0,
    error;
  async function work() {
    while (!error && next < items.length) {
      const i = next++;
      try {
        await job(items[i], i);
      } catch (e) {
        error ??= e;
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(workers, items.length) }, work));
  if (error) throw error;
}
