import { TIFF, filePath, isBoundary, rgb, discoverFiles } from './core.js';
import { csvRows } from './normalization.js';
import { COLOURMAPS, colourmap } from './colourmaps.js';

export const LAYER_TYPES = { rgb: 'Colour / reference image', scalar: 'Scalar image', labels: 'Categorical labels', imc: 'Additional IMC channels' };
export const IMAGE_FILE = /\.(tiff?|png|jpe?g|bmp|gif|webp)$/i;
export const MAX_LAYERS = 12;
export function layerFile(file, kind) { return kind === 'rgb' ? IMAGE_FILE.test(file.name) : TIFF.test(file.name) || (kind === 'labels' && file.name.toLowerCase() === 'label_mapping.csv'); }
export function indexLayer(files, rois, kind = '') {
  if (kind === 'imc') return new Map([...discoverFiles([], files, rois, new Map())].filter(([, entry]) => entry.images.length).map(([roi, entry]) => [roi, entry.images]));
  const names = new Set(rois), result = new Map();
  for (const file of files) {
    if (!IMAGE_FILE.test(file.name)) continue;
    const roi = file.name.replace(IMAGE_FILE, '').replace(/\.ome$/i, '');
    if (!names.has(roi)) continue;
    if (result.has(roi)) throw new Error(`More than one aligned image matches ${roi}: ${filePath(result.get(roi))} and ${filePath(file)}.`);
    result.set(roi, file);
  }
  return result;
}
export function labelDictionary(text, selected = null) {
  const [header = [], ...rows] = csvRows(text.replace(/^\uFEFF/, ''), 'label mapping');
  const headers = header.map(value => value.trim()), lower = headers.map(value => value.toLowerCase());
  if (new Set(lower).size !== lower.length) throw new Error('CSV dictionary column names must be unique.');
  const choose = names => headers[names.map(key => lower.indexOf(key)).find(i => i >= 0)] ?? '';
  const columns = selected ?? { value: choose(['value', 'original_num', 'label_value', 'id', 'label']), label: choose(['name', 'label_name', 'label', 'category']), color: choose(['color', 'colour']) };
  if (!columns.value || !columns.label || columns.value === columns.label || !headers.includes(columns.value) || !headers.includes(columns.label) || (columns.color && !headers.includes(columns.color))) throw new Error('Choose distinct label-ID and name columns in the CSV dictionary.');
  const indices = [columns.value, columns.label, columns.color].map(name => name ? headers.indexOf(name) : -1);
  const result = new Map();
  for (const row of rows) {
    const [raw, label, suppliedColor] = indices.map(i => row[i]?.trim() ?? ''), value = Number(raw);
    if (!raw || !Number.isSafeInteger(value) || value < 0 || value > 0xffffffff || !label) throw new Error('Label mappings need non-negative integer values and non-empty labels.');
    const color = suppliedColor || (value ? '#' + labelColour(value, []).map(c => c.toString(16).padStart(2, '0')).join('') : 'transparent');
    if (value && !/^#[a-f0-9]{6}$/i.test(color) && color !== 'transparent') throw new Error(`Label ${value}: use a #RRGGBB colour or transparent.`);
    if (label.length > 500) throw new Error('Label names must be at most 500 characters.');
    if (columns.value.toLowerCase() !== 'value' && result.has(value) && result.get(value).label !== label) throw new Error(`ID ${value} has multiple names. Choose the original image-ID column (for example original_num).`);
    result.set(value, { value, label, color: value ? color : 'transparent' });
  }
  if (result.size > 10000) throw new Error('At most 10,000 mapped labels are supported.');
  if (!result.size) throw new Error('The label dictionary contains no rows.');
  return { mapping: [...result.values()].sort((a, b) => a.value - b.value), columns, headers };
}
export function parseLabelMapping(text, columns = null) { return labelDictionary(text, columns).mapping; }
export function defaultLayer(id, kind) { return { id, visible: true, showInLegend: true, flatten: false, flatLabel: 'Foreground', flatColour: '#81d8c1', opacity: kind === 'labels' ? 0.5 : 1, blend: 'over', colourmap: 'grey', low: 0, high: 1, auto: true, style: 'fill', outlineWidth: 1 }; }
export function layerSettings(layer) {
  const { id, visible, opacity, blend, colourmap, low, high, auto, style } = layer;
  return { id, visible, opacity, blend, colourmap, low, high, auto, style, outlineWidth: layer.outlineWidth ?? 1, showInLegend: layer.showInLegend ?? true, flatten: layer.flatten ?? false, flatLabel: layer.flatLabel ?? 'Foreground', flatColour: layer.flatColour ?? '#81d8c1' };
}
export function validateLayerSettings(value = []) {
  if (!Array.isArray(value) || value.length > MAX_LAYERS) throw new Error('Invalid workspace aligned layer settings.');
  const seen = new Set();
  return value.map(item => {
    if (!item || typeof item.id !== 'string' || !item.id || item.id.length > 100 || seen.has(item.id) || typeof item.visible !== 'boolean' || typeof item.auto !== 'boolean' || !Number.isFinite(item.opacity) || item.opacity < 0 || item.opacity > 1 || !['over', 'add'].includes(item.blend) || !Object.hasOwn(COLOURMAPS, item.colourmap) || !['fill', 'outline'].includes(item.style) || !Number.isFinite(item.low) || !Number.isFinite(item.high) || item.low >= item.high) throw new Error('Invalid workspace aligned layer settings.');
    if (!Number.isInteger(item.outlineWidth ?? 1) || (item.outlineWidth ?? 1) < 1 || (item.outlineWidth ?? 1) > 20) throw new Error('Annotation outline width must be 1–20 pixels.');
    if (typeof (item.showInLegend ?? true) !== 'boolean' || typeof (item.flatten ?? false) !== 'boolean' || typeof (item.flatLabel ?? 'Foreground') !== 'string' || !(item.flatLabel ?? 'Foreground').trim() || (item.flatLabel ?? '').length > 80 || !/^#[a-f0-9]{6}$/i.test(item.flatColour ?? '#81d8c1')) throw new Error('Invalid aligned layer legend or flattened label settings.');
    seen.add(item.id); return layerSettings(item);
  });
}
export function labelColour(value, mapping) {
  if (!value) return null;
  if (mapping?.length) { const colour = mapping.find(item => item.value === value)?.color; return !colour || colour === 'transparent' ? null : rgb(colour); }
  // Stable by label ID across ROIs, including additional segmentation masks.
  return colourmap(((Math.imul(value, 2654435761) >>> 0) % 1000) / 999, 'viridis');
}
export function prepareLayer(layer, source) {
  if (!layer.visible || !layer.raster) return null;
  const prepared = { ...layer, kind: source.kind };
  if (source.kind === 'scalar') prepared.lut = Array.from({ length: 256 }, (_, i) => colourmap(i / 255, layer.colourmap));
  if (source.kind === 'labels') {
    if (layer.flatten) { prepared.flatTint = rgb(layer.flatColour ?? '#81d8c1'); return prepared; }
    prepared.colours = source.mapping.length ? new Map(source.mapping.map(item => [item.value, item.color === 'transparent' ? null : rgb(item.color)])) : null;
    prepared.lut = Array.from({ length: 1000 }, (_, i) => colourmap(i / 999, 'viridis'));
  }
  return prepared;
}
export function blendLayer(pixels, offset, layer, i, width, height, transparent = false) {
  const { raster, kind } = layer; let color, alpha = layer.opacity;
  if (kind === 'rgb') {
    alpha *= raster.data[i * 4 + 3] / 255;
    if (transparent) { for (let c = 0; c < 3; c++) pixels[offset + c] = raster.data[i * 4 + c]; pixels[offset + 3] = alpha * 255; return; }
    for (let c = 0; c < 3; c++) pixels[offset + c] = layer.blend === 'add' ? Math.min(255, pixels[offset + c] + raster.data[i * 4 + c] * alpha) : pixels[offset + c] * (1 - alpha) + raster.data[i * 4 + c] * alpha;
    return;
  }
  else if (kind === 'labels') {
    const value = raster.data[i];
    if (!value) return;
    if (layer.style === 'outline') {
      if ((layer.outlineWidth ?? 1) === 1) { if (!labelBoundary(raster.data, i, width, height, layer.flatten)) return; }
      else { layer.edgeDistance ??= labelEdgeDistance(raster, width, height, layer.flatten); if (layer.edgeDistance[i] > layer.outlineWidth) return; }
    }
    color = layer.flatten ? layer.flatTint : layer.colours ? layer.colours.get(value) : layer.lut[(Math.imul(value, 2654435761) >>> 0) % 1000];
  } else {
    const value = raster.data[i]; if (!Number.isFinite(value)) return;
    color = layer.lut[Math.round(Math.max(0, Math.min(1, (value - layer.low) / (layer.high - layer.low))) * 255)];
  }
  if (!color) return;
  if (transparent) { for (let c = 0; c < 3; c++) pixels[offset + c] = color[c]; pixels[offset + 3] = alpha * 255; return; }
  for (let c = 0; c < 3; c++) pixels[offset + c] = layer.blend === 'add' ? Math.min(255, pixels[offset + c] + color[c] * alpha) : pixels[offset + c] * (1 - alpha) + color[c] * alpha;
}

function labelBoundary(data, i, width, height, flatten = false) {
  if (!flatten) return isBoundary(data, i, width, height);
  const x = i % width, y = Math.floor(i / width);
  return data[i] !== 0 && (x === 0 || y === 0 || x === width - 1 || y === height - 1 || !data[i - 1] || !data[i + 1] || !data[i - width] || !data[i + width]);
}
const edgeCache = new WeakMap(), flatEdgeCache = new WeakMap();
export function labelEdgeDistance(raster, width, height, flatten = false) {
  const cache = flatten ? flatEdgeCache : edgeCache;
  if (cache.has(raster)) return cache.get(raster);
  // Separate caches preserve both views without copying or modifying label pixels.
  const data = raster.data, distance = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) if (data[i]) distance[i] = labelBoundary(data, i, width, height, flatten) ? 1 : Math.min(21, distance[i - 1] + 1, distance[i - width] + 1);
  for (let i = data.length - 1; i >= 0; i--) if (distance[i] > 1) distance[i] = Math.min(distance[i], distance[i + 1] + 1, distance[i + width] + 1);
  cache.set(raster, distance); return distance;
}
