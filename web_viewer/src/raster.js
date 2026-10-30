import { materializeAssetFile } from './asset-files.js';
const fileIds = new WeakMap(); let nextFileId = 0;
export function rasterIdentity(file) {
  if (!fileIds.has(file)) fileIds.set(file, ++nextFileId);
  return fileIds.get(file);
}
export async function loadRaster(file, plane = 0, mask = false, signal = null, normalization = null, kind = '') {
  if (signal?.aborted) throw new Error('Image loading cancelled.');
  file = await materializeAssetFile(file);
  if (signal?.aborted) throw new Error('Image loading cancelled.');
  const worker = new Worker(new URL('./raster.worker.js', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    const cancel = () => { worker.terminate(); reject(new Error('Image loading cancelled.')); };
    if (signal?.aborted) { cancel(); return; }
    signal?.addEventListener('abort', cancel, { once: true });
    worker.onmessage = ({ data }) => {
      signal?.removeEventListener('abort', cancel);
      worker.terminate();
      if (data.error) reject(new Error(`${file.name}: ${data.error}`));
      else resolve(data.result);
    };
    worker.onerror = error => { signal?.removeEventListener('abort', cancel); worker.terminate(); reject(new Error(error.message || 'TIFF decoding failed.')); };
    worker.postMessage({ file, plane, mask, normalization, kind });
  });
}
