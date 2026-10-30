// Locations are relative to the one folder the user grants read-only access to.
export function assetPath(value) {
  if (typeof value !== 'string') throw new Error('Enter a path relative to the workspace folder.');
  const path = value.trim().replace(/\\/g, '/');
  if (!path || path.length > 4096 || /[\x00-\x1f:*?"<>|]/.test(path) || path.split('/').some(part => !part || part === '.' || part === '..')) throw new Error('Use a relative path such as Images or analysis/data.h5ad, without .. or a drive letter.');
  return path;
}

export function sourceLocations(sources) {
  return [
    { key: 'anndata', label: 'AnnData file', ref: sources.anndata, kind: 'file' },
    { key: 'masks', label: 'Masks folder', ref: sources.masks, kind: 'folder', accepts: file => /\.tiff?$/i.test(file.name) },
    { key: 'images', label: 'Images folder', ref: sources.images, kind: 'folder', accepts: file => /\.tiff?$/i.test(file.name) },
    ...(sources.layers ?? []).flatMap(layer => [
      { key: `layer:${layer.id}`, label: `${layer.name} folder`, ref: layer.folder, kind: 'folder', accepts: file => layer.kind === 'rgb' ? /\.(tiff?|png|jpe?g|bmp|gif|webp)$/i.test(file.name) : /\.tiff?$/i.test(file.name) || (layer.kind === 'labels' && file.name.toLowerCase() === 'label_mapping.csv') },
      ...(layer.dictionary ? [{ key: `dictionary:${layer.id}`, label: `${layer.name} dictionary`, ref: layer.dictionary.file, kind: 'file' }] : []),
    ]),
  ].filter(item => item.ref);
}

export const referenceIdentity = ref => JSON.stringify([ref.name, ref.size, ref.count, ref.signature]);

const resolvedHandles = new WeakMap();
export async function resolvePickedPaths(root, sources, handles, fresh = false) {
  if (!root?.resolve) return;
  if (!resolvedHandles.has(root) || fresh) resolvedHandles.set(root, new WeakMap());
  const cache = resolvedHandles.get(root);
  for (const item of sourceLocations(sources)) {
    const handle = handles.get(item.key); if (!handle) continue;
    const parts = cache.has(handle) ? cache.get(handle) : await root.resolve(handle);
    if (!parts?.length) throw new Error(`${item.label} must be inside the connected workspace root. Choose their common parent folder.`);
    cache.set(handle, parts);
    item.ref.path = assetPath(parts.join('/'));
  }
}
