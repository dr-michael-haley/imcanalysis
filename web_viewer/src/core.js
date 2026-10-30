import { colourmap } from './colourmaps.js';
import { colourRange } from './colour-limits.js';
export const TIFF = /\.(tif|tiff)$/i;
export const filePath = file => file.webkitRelativePath || file.name;
export const stem = file => file.name.replace(TIFF, '').replace(/\.ome$/i, '');
const alias = value => String(value ?? '').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');

export function channelAliases(header) {
  const result = new Map();
  const ambiguous = new Set();
  header.markers.forEach((name, i) => {
    for (const candidate of [name, header.channelNames[i], header.channelLabels[i],
      header.channelNames[i] && header.channelLabels[i] ? `${header.channelNames[i]}_${header.channelLabels[i]}` : null]) {
      const key = alias(candidate);
      if (!key) continue;
      if (result.has(key) && result.get(key) !== name) ambiguous.add(key);
      result.set(key, name);
    }
  });
  for (const key of ambiguous) result.delete(key);
  return result;
}

export function channelName(file, roi, aliases) {
  const name = stem(file);
  const short = name.startsWith(`${roi}_`) ? name.slice(roi.length + 1) : name;
  const stripIsotope = value => value.replace(/^(?:\d{2,3}[A-Za-z]{1,2}|[A-Za-z]{1,2}\d{2,3})_/, '');
  const parts = short.split('_');
  const candidates = [stripIsotope(short), short];
  if (parts.length >= 4) candidates.push(parts.slice(3).join('_'), parts[2], `${parts[2]}_${parts.slice(3).join('_')}`);
  candidates.push(name, name.split('_').at(-1));
  for (const value of candidates) if (aliases.has(alias(value))) return aliases.get(alias(value));
  // Unlisted image channels (often DNA) have ROI-specific prefixes. Their
  // final filename token is the stable channel identity used between ROIs.
  return name.split('_').at(-1) || name;
}

export function discoverFiles(maskFiles, imageFiles, rois, aliases) {
  const index = new Map(rois.map(roi => [roi, { masks: [], images: [] }]));
  for (const file of maskFiles.filter(file => TIFF.test(file.name))) {
    if (index.has(stem(file))) index.get(stem(file)).masks.push(file);
  }
  // Longest ROI first prevents ROI_A stealing ROI_A_2's flat files.
  const ordered = [...rois].sort((a, b) => b.length - a.length);
  for (const file of imageFiles.filter(file => TIFF.test(file.name))) {
    const parents = filePath(file).split('/').slice(0, -1);
    const nested = parents.filter(parent => index.has(parent));
    if (new Set(nested).size > 1) throw new Error(`Ambiguous ROI folders in ${filePath(file)}.`);
    const roi = nested[0] ?? ordered.find(name => stem(file) === name || stem(file).startsWith(`${name}_`));
    if (!roi) continue;
    index.get(roi).images.push({ file, name: channelName(file, roi, aliases), key: filePath(file) });
  }
  for (const [roi, entry] of index) {
    if (entry.masks.length > 1) throw new Error(`Multiple masks match ${roi}. Select a folder containing one mask per ROI.`);
    entry.images.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    const counts = new Map();
    for (const item of entry.images) counts.set(item.name, (counts.get(item.name) || 0) + 1);
    for (const item of entry.images) if (counts.get(item.name) > 1) item.name += ` · ${item.key}`;
  }
  return index;
}

export function validateMask(data, ids) {
  const present = new Set();
  for (const value of data) {
    if (!Number.isInteger(value) || value < 0 || value > 4294967295) throw new Error('Masks must contain non-negative integer object IDs (up to uint32).');
    if (value) present.add(value);
  }
  const known = new Set(ids);
  return { missing: ids.filter(id => !present.has(id)).length, context: [...present].filter(id => !known.has(id)).length };
}

export function extent(values) {
  let min = Infinity, max = -Infinity;
  for (const value of values) if (value != null && Number.isFinite(Number(value))) {
    min = Math.min(min, Number(value)); max = Math.max(max, Number(value));
  }
  return min === Infinity ? [0, 1] : [min, max === min ? min + 1 : max];
}

export function contrast(values) {
  const sample = [];
  const stride = Math.max(1, Math.floor(values.length / 50000));
  for (let i = 0; i < values.length; i += stride) if (Number.isFinite(values[i])) sample.push(values[i]);
  sample.sort((a, b) => a - b);
  const low = sample[0] ?? 0;
  const high = sample[Math.floor((sample.length - 1) * 0.995)] ?? 1;
  return [low, high > low ? high : low + 1];
}

const PALETTE = ['#64d8bc', '#ffb86b', '#91a7ff', '#f38ba8', '#89dceb', '#d6b5ff', '#d4df80', '#f2a88c', '#a5c5ad', '#d7a6cf'];
export function rgb(hex, fallback = '#89949e') {
  const match = /^#([0-9a-f]{6})$/i.exec(String(hex));
  const value = parseInt((match ? match[1] : fallback.slice(1)), 16);
  return [value >> 16, (value >> 8) & 255, value & 255];
}

export function colourScale(values, categories, palette = [], map = 'viridis', reverse = false, limits = {}) {
  if (categories) {
    const colours = new Map(categories.map((category, i) => [category, rgb(palette[i] || PALETTE[i % PALETTE.length])]));
    return { colour: value => colours.get(value) ?? [137, 148, 158], entries: [...colours], range: null };
  }
  const range = colourRange(extent(values), limits);
  return { colour: value => {
    if (value == null || !Number.isFinite(Number(value))) return [137, 148, 158];
    const t = Math.max(0, Math.min(1, (Number(value) - range[0]) / (range[1] - range[0])));
    return colourmap(t, map, reverse);
  }, entries: [], range };
}

export function isBoundary(mask, i, width, height) {
  const id = mask[i], x = i % width, y = Math.floor(i / width);
  return id !== 0 && (x === 0 || y === 0 || x === width - 1 || y === height - 1 ||
    mask[i - 1] !== id || mask[i + 1] !== id || mask[i - width] !== id || mask[i + width] !== id);
}
