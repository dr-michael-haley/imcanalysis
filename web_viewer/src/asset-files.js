// Cached inventories contain metadata only. Resolve an image File just before
// passing it to a decoding worker, never during a fast folder reconnection.
const deferred = new WeakMap();
export function deferredAssetFile(metadata, open) {
  const file = { ...metadata }; deferred.set(file, { open, pending: null }); return file;
}
export async function materializeAssetFile(file) {
  const entry = deferred.get(file);
  if (!entry) return file;
  if (!entry.pending) entry.pending = entry.open().catch(error => { entry.pending = null; throw error; });
  return entry.pending;
}
