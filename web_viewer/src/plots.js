import { sourceKey, sourceLabel } from './remaps.js';
import { validateColourLimits } from './colour-limits.js';

export const PLOT_NAMES = { matrix: 'Marker matrix plot', bars: 'Population composition bars', composition: 'Population composition heatmap', comparison: 'Compare labels', alluvial: 'Categorical alluvial plot' };
export const defaultPlots = () => ({ type: 'matrix', groupBy: '', source: 'X', markers: [], scope: 'current', rois: [], populations: [], filterBy: '', filterValues: [], sampleBy: '', compareBy: '', scale: 'none', measure: 'percent', normalization: 'row_percent', order: 'data', sortPopulation: '', swapAxes: false, palette: 'automatic', title: '', vmin: null, vmax: null });
export function validatePlots(input = defaultPlots()) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Invalid plot settings.');
  const value = { ...defaultPlots(), ...input }, result = {};
  const enums = { type: Object.keys(PLOT_NAMES), scope: ['current', 'selected', 'all'], scale: ['none', 'zscore', 'minmax'], measure: ['count', 'percent'], normalization: ['count', 'row_percent', 'column_percent'], order: ['data', 'name', 'count-desc', 'count-asc'], palette: ['automatic', 'blue', 'viridis', 'diverging'] };
  for (const [key, choices] of Object.entries(enums)) { if (!choices.includes(value[key])) throw new Error(`Invalid plot ${key}.`); result[key] = value[key]; }
  for (const key of ['groupBy', 'source', 'filterBy', 'sampleBy', 'compareBy', 'sortPopulation', 'title']) {
    if (typeof value[key] !== 'string' || value[key].length > (key === 'title' ? 200 : 16384)) throw new Error(`Invalid plot ${key}.`);
    result[key] = value[key];
  }
  for (const key of ['markers', 'rois', 'populations', 'filterValues']) {
    if (!Array.isArray(value[key]) || value[key].length > (key === 'markers' ? 100 : 10000) || value[key].some(v => typeof v !== 'string' || v.length > 16384)) throw new Error(`Invalid plot ${key}.`);
    result[key] = [...new Set(value[key])];
  }
  for (const key of [...result.populations, ...result.filterValues, ...(result.sortPopulation ? [result.sortPopulation] : [])]) {
    let raw; try { raw = JSON.parse(key); } catch { throw new Error('Invalid plot population/filter value.'); }
    if (raw !== null && !['string', 'number', 'boolean'].includes(typeof raw)) throw new Error('Invalid plot population/filter value.');
  }
  if (typeof value.swapAxes !== 'boolean') throw new Error('Invalid plot axis arrangement.');
  result.swapAxes = value.swapAxes;
  Object.assign(result, validateColourLimits(value));
  return result;
}

export function labelCategories(categories) {
  const counts = new Map();
  for (const item of categories) counts.set(sourceLabel(item.value), (counts.get(sourceLabel(item.value)) ?? 0) + 1);
  return categories.map(item => ({ ...item, label: sourceLabel(item.value) + (counts.get(sourceLabel(item.value)) > 1 ? ` [${item.value === null ? 'missing' : typeof item.value}]` : '') }));
}

// Display scaling is applied to population means, never to the source matrix.
export function scaleMeans(means, scale) {
  const output = means.map(row => [...row]);
  for (let col = 0; col < (means[0]?.length ?? 0); col++) {
    const values = means.map(row => row[col]).filter(Number.isFinite);
    if (!values.length || scale === 'none') continue;
    const average = values.reduce((sum, v) => sum + v, 0) / values.length;
    const low = Math.min(...values), high = Math.max(...values);
    const spread = scale === 'zscore' ? Math.sqrt(values.reduce((sum, v) => sum + (v - average) ** 2, 0) / values.length) : high - low;
    for (let row = 0; row < means.length; row++) if (Number.isFinite(means[row][col])) output[row][col] = (means[row][col] - (scale === 'zscore' ? average : low)) / (spread || 1);
  }
  return output;
}
export function normaliseCounts(counts, mode) {
  const columns = Array.from({ length: counts[0]?.length ?? 0 }, (_, col) => counts.reduce((sum, row) => sum + row[col], 0));
  return counts.map(row => { const total = row.reduce((sum, v) => sum + v, 0); return row.map((v, col) => mode === 'count' ? v : (mode === 'row_percent' ? total : columns[col]) ? v * 100 / (mode === 'row_percent' ? total : columns[col]) : 0); });
}

export function plottedDataCSV(result) {
  const q = v => `"${String(v ?? '').replaceAll('"', '""')}"`;
  const rows = [['plot_type', 'row_field', 'row_value_json', 'row_label', 'column_field', 'column_value_json', 'column_label', 'display_value', 'cell_count', 'mean', 'finite_count', 'selected_cells', 'matrix_source', 'settings_json']];
  const settings = JSON.stringify(result.settings);
  result.rows.forEach((row, r) => result.columns.forEach((col, c) => rows.push([result.type, result.rowTitle, row.key, row.label, result.columnTitle, col.key, col.label, result.values[r][c], result.counts[r][c], result.means?.[r][c], result.finiteCounts?.[r][c], result.selectedCells, result.type === 'matrix' ? result.settings.source : '', r === 0 && c === 0 ? settings : ''])));
  return rows.map(row => row.map(q).join(',')).join('\r\n') + '\r\n';
}
export const plotKey = sourceKey;
