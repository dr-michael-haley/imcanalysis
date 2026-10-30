import { readColumn } from './anndata.js';
import { readExpression } from './plot-reader.js';
import { sourceKey } from './remaps.js';

const BATCH = 8192, DISTRIBUTION_BINS = 64;
export function representativeMarkers(reader, request, progress = () => {}) {
  const header = reader.header(), { source, population, roiOnly, rows, values, colourBy, remap } = request;
  if (!header.sources.includes(source)) throw new Error('Choose an available expression source.');
  if (!population) throw new Error('Select a single population in Cell overlay.');
  const groups = new Int8Array(reader.count).fill(-1), totals = [0, 0];
  if (roiOnly) {
    if (!rows?.length || rows.length !== values?.length) throw new Error('Load an ROI with population labels.');
    rows.forEach((row, i) => { const group = sourceKey(values[i]) === population ? 0 : 1; groups[row] = group; totals[group]++; });
  } else {
    const key = remap?.source ?? (colourBy.startsWith('obs:') ? colourBy.slice(4) : '');
    if (!header.obsKeys.includes(key)) throw new Error('Select an observation or remap population.');
    const mapping = remap && new Map(remap.rows.map(row => [sourceKey(row.source), row.label]));
    for (let start = 0; start < reader.count; start += BATCH) {
      const batch = Array.from({ length: Math.min(BATCH, reader.count - start) }, (_, i) => start + i);
      const labels = readColumn(reader.obs.get(key), batch).values;
      labels.forEach((value, i) => { if (mapping) value = mapping.get(sourceKey(value)); const group = sourceKey(value) === population ? 0 : 1; groups[start + i] = group; totals[group]++; });
      progress({ fraction: start / reader.count * 0.1, detail: `Reading population labels · ${(start + batch.length).toLocaleString()} cells` });
    }
  }
  if (!totals[0] || !totals[1]) throw new Error('Ranking needs both the selected population and other cells in this scope. Try all ROIs.');
  const stats = header.markers.map(() => [0, 1].map(() => ({ sum: 0, square: 0, invalid: 0, seen: 0, min: Infinity, max: -Infinity })));
  readExpression(reader, source, header.markers, groups, (group, col, value) => {
    const s = stats[col][group]; s.seen++;
    if (!Number.isFinite(value)) { s.invalid++; return; }
    s.sum += value; s.square += value * value; s.min = Math.min(s.min, value); s.max = Math.max(s.max, value);
  }, p => progress({ fraction: 0.1 + p.fraction * 0.65, detail: `Ranking markers · ${p.detail}` }));
  const ranked = stats.map((pair, col) => {
    const counts = pair.map((s, g) => totals[g] - s.invalid);
    pair.forEach((s, g) => { if (s.seen < totals[g]) { s.min = Math.min(0, s.min); s.max = Math.max(0, s.max); } });
    const means = pair.map((s, g) => counts[g] ? s.sum / counts[g] : null), count = counts[0] + counts[1];
    const mean = (pair[0].sum + pair[1].sum) / count;
    const variance = Math.max(0, (pair[0].square + pair[1].square) / count - mean * mean);
    const score = counts.every(n => n > 0) && variance > 0 ? (means[0] - means[1]) / Math.sqrt(variance) : null;
    return { marker: header.markers[col], means, counts, score, min: Math.min(...pair.map(s => s.min)), max: Math.max(...pair.map(s => s.max)) };
  }).filter(item => Number.isFinite(item.score) && item.score > 0).sort((a, b) => b.score - a.score || a.marker.localeCompare(b.marker)).slice(0, Math.max(1, Math.min(30, request.top ?? 6)));
  const histograms = ranked.map(item => { if (item.max === item.min) item.max = item.min + 1; return { bins: [new Uint32Array(DISTRIBUTION_BINS), new Uint32Array(DISTRIBUTION_BINS)], seen: [0, 0] }; });
  const bin = (item, value) => Math.max(0, Math.min(DISTRIBUTION_BINS - 1, Math.floor((value - item.min) / (item.max - item.min) * DISTRIBUTION_BINS)));
  if (ranked.length) readExpression(reader, source, ranked.map(item => item.marker), groups, (group, col, value) => {
    const h = histograms[col]; h.seen[group]++; if (Number.isFinite(value)) h.bins[group][bin(ranked[col], value)]++;
  }, p => progress({ fraction: 0.75 + p.fraction * 0.25, detail: `Building expression distributions · ${p.detail}` }));
  ranked.forEach((item, i) => { const h = histograms[i]; [0, 1].forEach(g => { h.bins[g][bin(item, 0)] += totals[g] - h.seen[g]; }); item.histogram = h.bins.map(b => Array.from(b)); });
  return { markers: ranked, totals, source, roiOnly, population };
}

// A sample-level field must have one value per ROI, including missing values.
// Read in bounded batches and stop a field as soon as an ROI varies within it.
export function roiMetadata(reader, { roiKey, idKey }, progress = () => {}) {
  if (!reader.obsKeys.includes(roiKey)) throw new Error('Choose an ROI column first.');
  const names = [], lookup = new Map(), rowROI = new Uint32Array(reader.count);
  for (let start = 0; start < reader.count; start += BATCH) {
    const rows = Array.from({ length: Math.min(BATCH, reader.count - start) }, (_, i) => start + i);
    readColumn(reader.obs.get(roiKey), rows).values.forEach((value, i) => { const name = String(value); if (!lookup.has(name)) { lookup.set(name, names.length); names.push(name); } rowROI[start + i] = lookup.get(name); });
    progress({ fraction: 0.1 * (start + rows.length) / reader.count, detail: 'Indexing ROI metadata…' });
  }
  const fields = [], skipped = [], keys = reader.obsKeys.filter(key => ![roiKey, idKey].includes(key));
  for (const [index, key] of keys.entries()) {
    const values = new Map(); let mixed = false;
    try {
      for (let start = 0; start < reader.count && !mixed; start += BATCH) {
        const rows = Array.from({ length: Math.min(BATCH, reader.count - start) }, (_, i) => start + i);
        const data = readColumn(reader.obs.get(key), rows).values;
        for (let i = 0; i < data.length; i++) {
          const roi = rowROI[start + i], value = data[i] == null || (typeof data[i] === 'number' && !Number.isFinite(data[i])) ? null : data[i];
          if (values.has(roi) && sourceKey(values.get(roi)) !== sourceKey(value)) { mixed = true; break; }
          values.set(roi, value);
        }
      }
      if (!mixed) fields.push({ name: key, values: names.map((name, i) => [name, values.get(i) ?? null]) });
    } catch (error) { skipped.push(`${key}: ${error.message}`); }
    progress({ fraction: 0.1 + 0.9 * (index + 1) / keys.length, detail: `Checking metadata · ${key} (${index + 1}/${keys.length})` });
  }
  return { fields, skipped };
}

export function filterROIs(ranking, metadata, filters) {
  const lookup = new Map(metadata.map(field => [field.name, new Map(field.values)]));
  return ranking.filter(roi => filters.every(filter => { const column = lookup.get(filter.field); return column?.has(roi.name) && filter.values.includes(sourceKey(column.get(roi.name))); }));
}
export function validateROIFilters(filters = []) {
  if (!Array.isArray(filters) || filters.length > 50) throw new Error('Invalid ROI filters.');
  const used = new Set();
  return filters.map(filter => {
    if (!filter || typeof filter.field !== 'string' || !filter.field || used.has(filter.field) || !Array.isArray(filter.values) || !filter.values.length || filter.values.length > 100000) throw new Error('Invalid ROI filter.');
    used.add(filter.field);
    for (const key of filter.values) { const value = JSON.parse(key); if (value !== null && !['number', 'string', 'boolean'].includes(typeof value)) throw new Error('Invalid ROI filter value.'); }
    return { field: filter.field, values: [...new Set(filter.values)] };
  });
}
