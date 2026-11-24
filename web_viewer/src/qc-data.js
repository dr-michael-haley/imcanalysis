import { inputFile, relativeInput, readJson, writeJson, savedInputs } from './analysis-project.js';
import { materializeAssetFile } from './asset-files.js';
import { qcColumns } from './mask-qc.js';

export function computeJob(job, signal) {
  signal.throwIfAborted();
  const worker = new Worker(new URL('./analysis.worker.js', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    const close = () => {
      worker.terminate();
      signal.removeEventListener('abort', cancel);
    };
    const cancel = () => {
      close();
      reject(signal.reason);
    };
    signal.addEventListener('abort', cancel, { once: true });
    worker.onerror = e => {
      close();
      reject(Error(e.message || 'Analysis worker failed.'));
    };
    worker.onmessage = ({ data }) => {
      close();
      if (data.error) reject(Error(data.error));
      else resolve(data.result);
    };
    try {
      worker.postMessage(job);
    } catch (e) {
      close();
      reject(e);
    }
  });
}
export function localInputMap(files) {
  return new Map(files.map(f => [relativeInput(f), f]));
}
export async function openSource(project, source, local) {
  return {
    source,
    folder: await project.getDirectoryHandle('features-' + source.id),
    handles: (await savedInputs(source.id)) ?? {},
    local: local ?? {},
    prepared: new Set()
  };
}
export async function asset(data, kind, path) {
  if (data.local[kind]?.has(path)) return materializeAssetFile(data.local[kind].get(path));
  const handle = data.handles[kind];
  if (!handle) throw Error('Use Locate ' + kind + ' to reconnect the ' + kind + ' folder.');
  try {
    return await inputFile(handle, path);
  } catch (error) {
    throw Error(
      'Could not open ' +
        path +
        ' in ' +
        handle.name +
        '. Use Locate ' +
        kind +
        ' to reconnect. ' +
        error.message
    );
  }
}
export async function roiFeatures(data, index, settings) {
  if (settings.ensure && data.prepared.has(index)) return;
  return navigator.locks.request(
    'imc-features-' + data.source.id + '-' + index,
    { signal: settings.signal },
    () => readFeatures(data, index, settings)
  );
}
async function readFeatures(data, index, { tissue, ensure = false, retainMask = false, maxPixels, signal }) {
  const roi = data.source.rois[index],
    filename = 'roi-' + index + '.json';
  let handle;
  try {
    handle = await data.folder.getFileHandle(filename);
  } catch (error) {
    if (error.name !== 'NotFoundError') throw error;
  }
  if (handle && ensure) {
    data.prepared.add(index);
    return;
  }
  const file = handle ? await handle.getFile() : null;
  // A terminated first writer can leave an empty placeholder on a network drive.
  const cached = file?.size ? JSON.parse(await file.text()) : null;
  if (cached) data.prepared.add(index);
  if (cached) return retainMask ? { table: cached } : cached;
  const [mask, dna] = await Promise.all([
    tissue ? null : asset(data, 'masks', roi.mask),
    roi.dna && !tissue ? asset(data, 'images', roi.dna) : null
  ]);
  const extra = data.source.metrics?.columns ?? [];
  const imported =
    !cached && extra.length
      ? { columns: extra, byId: await readJson(data.folder, 'metrics-' + index + '.json', {}) }
      : null;
  const result = await computeJob(
    {
      op: 'extract',
      decodedMask: tissue?.mask,
      decodedDNA: tissue && roi.dna ? await tissue.dna() : null,
      retainMask,
      columns: qcColumns.concat(extra),
      mask,
      dna,
      imported,
      extract: !cached,
      maxPixels
    },
    signal
  );
  signal.throwIfAborted();
  if (!cached) await writeJson(data.folder, filename, result.table);
  data.prepared.add(index);
  return retainMask ? result : result.table;
}
