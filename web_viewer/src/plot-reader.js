import { readColumn, attribute, checkFilters } from './anndata.js';
import { colourScale } from './core.js';
import { sourceKey, sourceLabel, validateRemaps } from './remaps.js';
import { validatePlots, labelCategories, scaleMeans, normaliseCounts, PLOT_NAMES } from './plots.js';

const BATCH = 8192, MAX_AXIS = 200;
const canonical = value => value == null || (typeof value === 'number' && !Number.isFinite(value)) ? null : value;
const hex = rgb => '#' + rgb.map(v => v.toString(16).padStart(2, '0')).join('');

// This is run in a dedicated, cancellable worker. Only aggregates reach the UI.
export function readPlot(reader, request, progress = () => {}) {
  const settings = validatePlots(request.settings), header = reader.header();
  const plans = validateRemaps(request.remaps ?? []), planMap = new Map(plans.map(plan => [`remap:${plan.id}`, plan]));
  const obs = key => { if (!header.obsKeys.includes(key)) throw new Error(`Observation unavailable: ${key || '(choose one)'}.`); return reader.obs.get(key); };
  function labels(ref) {
    const plan = planMap.get(ref);
    if (ref.startsWith('remap:') && !plan) throw new Error('The selected population remap is unavailable.');
    if (!plan && !ref.startsWith('obs:')) throw new Error('Choose population labels.');
    const key = plan ? plan.source : ref.slice(4), node = obs(key);
    const mapping = plan && new Map(plan.rows.map(row => [sourceKey(row.source), row]));
    const categories = plan ? [...new Set(plan.rows.map(row => row.label))] : attribute(node, 'encoding-type') === 'categorical' ? readColumn(node, []).categories : [];
    const stored = !plan && reader.file.get(`uns/${key}_colors`);
    const palette = plan ? [...new Map(plan.rows.map(row => [row.label, row.colour])).values()] : stored ? readColumn(stored).values : [];
    const colours = colourScale([], categories, palette);
    return { key, title: plan ? plan.name : key, node, categories,
      convert(value) { if (!mapping) return canonical(value); const row = mapping.get(sourceKey(value)); if (!row) throw new Error(`Remap does not cover ${sourceLabel(value)}.`); return row.label; },
      colour(value) { return hex(colours.colour(value)); } };
  }
  const primary = labels(settings.groupBy), comparison = ['comparison', 'alluvial'].includes(settings.type) ? labels(settings.compareBy) : null;
  if (comparison && settings.groupBy === settings.compareBy) throw new Error('Choose two different label columns or remaps.');
  const roiNode = settings.scope !== 'all' ? obs(request.roiKey) : null;
  const roiNames = new Set(settings.scope === 'current' ? [request.currentROI] : settings.rois);
  if (settings.scope !== 'all' && (!roiNames.size || roiNames.has('') || roiNames.has(undefined))) throw new Error('Choose at least one ROI.');
  const filterNode = settings.filterBy ? obs(settings.filterBy) : null, filters = new Set(settings.filterValues), populations = new Set(settings.populations);
  const composition = ['bars', 'composition'].includes(settings.type), sampleNode = composition ? obs(settings.sampleBy) : null;
  if (settings.type === 'matrix' && (!settings.markers.length || !header.sources.includes(settings.source) || settings.markers.some(name => !header.markers.includes(name)))) throw new Error('Choose available markers and an expression matrix.');
  const groups = new Map(), bins = new Map(), pairCounts = new Map();
  const rowGroups = settings.type === 'matrix' ? new Int32Array(reader.count).fill(-1) : null;
  let selectedCells = 0;
  const ensure = (map, value, colour) => {
    const key = sourceKey(value);
    if (!map.has(key)) { if (map.size >= MAX_AXIS) throw new Error('Choose filters to keep each plot axis to at most 200 groups.'); map.set(key, { key, value, colour, count: 0, index: map.size }); }
    return map.get(key);
  };
  for (let start = 0; start < reader.count; start += BATCH) {
    const rows = Array.from({ length: Math.min(BATCH, reader.count - start) }, (_, i) => start + i), cache = new Map();
    const column = node => { if (!node) return null; if (!cache.has(node)) cache.set(node, readColumn(node, rows).values); return cache.get(node); };
    const values = column(primary.node), rois = column(roiNode), filter = column(filterNode), samples = column(sampleNode), other = column(comparison?.node);
    for (let i = 0; i < rows.length; i++) {
      if (rois && !roiNames.has(String(rois[i]))) continue;
      if (filter && filters.size && !filters.has(sourceKey(filter[i]))) continue;
      const value = primary.convert(values[i]);
      if (populations.size && !populations.has(sourceKey(value))) continue;
      const group = ensure(groups, value, primary.colour(value)); group.count++; selectedCells++;
      if (rowGroups) rowGroups[start + i] = group.index;
      else {
        const binValue = comparison ? comparison.convert(other[i]) : canonical(samples[i]);
        const bin = ensure(bins, binValue, comparison?.colour(binValue)); bin.count++;
        const key = `${group.index}:${bin.index}`; pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
      }
    }
    progress({ fraction: (start + rows.length) / reader.count * (rowGroups ? 0.25 : 0.9), detail: `Read ${Math.min(start + BATCH, reader.count).toLocaleString()} / ${reader.count.toLocaleString()} cells` });
  }
  if (!selectedCells) throw new Error('No cells match these plot filters.');
  const ordered = (map, categories = [], sortKey = '') => {
    const list = [...map.values()], ranks = new Map(categories.map((value, i) => [sourceKey(value), i]));
    const named = (a, b) => sourceLabel(a.value).localeCompare(sourceLabel(b.value), undefined, { numeric: true });
    if (settings.order === 'name') list.sort(named);
    else if (settings.order.startsWith('count')) {
      const group = groups.get(sortKey);
      const score = item => group ? pairCounts.get(`${group.index}:${item.index}`) ?? 0 : item.count;
      list.sort((a, b) => (score(a) - score(b)) * (settings.order === 'count-desc' ? -1 : 1) || named(a, b));
    } else if (categories.length) list.sort((a, b) => (ranks.get(a.key) ?? Infinity) - (ranks.get(b.key) ?? Infinity));
    return labelCategories(list);
  };
  const groupList = ordered(groups, primary.categories), binList = ordered(bins, comparison?.categories, composition ? settings.sortPopulation : '');
  for (const [items, definition] of [[groupList, primary], [binList, comparison]]) if (definition && !definition.categories.length) {
    const colours = colourScale([], items.filter(item => item.value !== null).map(item => item.value));
    items.forEach(item => { item.colour = hex(colours.colour(item.value)); });
  }
  let rows, columns, counts, means, finiteCounts, values, rowTitle, columnTitle, valueLabel;
  if (rowGroups) {
    rows = groupList; columns = settings.markers.map(value => ({ key: sourceKey(value), value, label: value }));
    if (rows.length * columns.length > 20000) throw new Error('Limit the plot to 20,000 matrix cells.');
    const sums = Array.from({ length: groups.size }, () => new Float64Array(columns.length));
    const finite = [...groups.values()].map(group => new Float64Array(columns.length).fill(group.count));
    readExpression(reader, settings.source, settings.markers, rowGroups, (group, col, value) => { if (Number.isFinite(value)) sums[group][col] += value; else finite[group][col]--; }, progress);
    counts = rows.map(group => columns.map(() => group.count));
    finiteCounts = rows.map(group => Array.from(finite[group.index]));
    means = rows.map(group => columns.map((_, col) => finite[group.index][col] ? sums[group.index][col] / finite[group.index][col] : null));
    if (means.some(row => row.some(v => v !== null && !Number.isFinite(v)))) throw new Error('Expression values exceed the numeric range for plotting.');
    values = scaleMeans(means, settings.scale); rowTitle = primary.title; columnTitle = 'Marker';
    valueLabel = { none: 'Mean intensity', zscore: 'Marker-wise z-score', minmax: 'Marker-wise 0–1' }[settings.scale];
  } else if (composition) {
    rows = groupList; columns = binList; rowTitle = primary.title; columnTitle = settings.sampleBy;
    counts = rows.map(group => columns.map(bin => pairCounts.get(`${group.index}:${bin.index}`) ?? 0));
    values = normaliseCounts(counts, settings.measure === 'percent' ? 'column_percent' : 'count');
    valueLabel = settings.measure === 'percent' ? 'Selected cells within group (%)' : 'Cell count';
  } else {
    rows = binList; columns = groupList; rowTitle = comparison.title; columnTitle = primary.title;
    counts = rows.map(bin => columns.map(group => pairCounts.get(`${group.index}:${bin.index}`) ?? 0));
    values = normaliseCounts(counts, settings.type === 'alluvial' ? 'count' : settings.normalization);
    valueLabel = settings.type === 'alluvial' ? 'Cell count' : { count: 'Cell count', row_percent: 'Within row (%)', column_percent: 'Within column (%)' }[settings.normalization];
  }
  if (rows.length * columns.length > 20000) throw new Error('Limit the plot to 20,000 matrix cells using filters.');
  const scope = settings.scope === 'all' ? 'All ROIs' : settings.scope === 'current' ? request.currentROI : `${roiNames.size} selected ROIs`;
  const snapshot = { ...settings, roiKey: request.roiKey, resolvedROIs: settings.scope === 'current' ? [request.currentROI] : settings.rois, remaps: plans };
  progress({ fraction: 1, detail: 'Plot ready' });
  return { type: settings.type, title: settings.title || PLOT_NAMES[settings.type], subtitle: `${selectedCells.toLocaleString()} selected cells · ${scope}${settings.type === 'matrix' ? ` · ${settings.source}` : ''}`, selectedCells, rows, columns, rowTitle, columnTitle, values, counts, means, finiteCounts, valueLabel, settings: snapshot };
}

export function readExpression(reader, source, markers, groups, consume, progress = () => {}) {
  const node = reader.file.get(source), encoding = attribute(node, 'encoding-type', 'array');
  const shape = node.shape ?? Array.from(attribute(node, 'shape', []));
  if (shape.length !== 2 || Number(shape[0]) !== reader.count || Number(shape[1]) !== reader.markers.length) throw new Error('Expression matrix shape does not match AnnData.');
  const selected = new Map(markers.map((name, i) => [reader.markers.indexOf(name), i]));
  const update = (done, total) => progress({ fraction: 0.25 + 0.75 * done / total, detail: `Reading marker values · ${Math.round(100 * done / total)}%` });
  if (encoding === 'array') {
    checkFilters(node);
    if (![0, 1].includes(node.metadata.type)) throw new Error('Expression values must be numeric.');
    for (let start = 0; start < reader.count; start += BATCH) {
      const end = Math.min(start + BATCH, reader.count);
      if (groups.subarray(start, end).some(group => group >= 0)) for (const [column, target] of selected) {
        const values = node.slice([[start, end], [column, column + 1]]);
        for (let i = 0; i < values.length; i++) if (groups[start + i] >= 0) consume(groups[start + i], target, Number(values[i]));
      }
      update(end, reader.count);
    }
    return;
  }
  if (!['csr_matrix', 'csc_matrix'].includes(encoding)) throw new Error('Unsupported expression encoding.');
  const pointers = node.get('indptr'), indices = node.get('indices'), data = node.get('data');
  const major = encoding === 'csr_matrix' ? reader.count : reader.markers.length, minor = encoding === 'csr_matrix' ? reader.markers.length : reader.count;
  if (pointers?.shape?.[0] !== major + 1 || indices?.shape?.[0] !== data?.shape?.[0] || ![0, 1].includes(data?.metadata?.type)) throw new Error('Invalid sparse matrix structure.');
  [pointers, indices, data].forEach(checkFilters);
  const pointer = value => { const v = Number(value); if (!Number.isSafeInteger(v) || v < 0 || v > data.shape[0]) throw new Error('Invalid sparse pointers.'); return v; };
  function entries(start, end, visit) {
    if (end < start) throw new Error('Invalid sparse pointers.');
    for (let at = start; at < end; at += 65536) {
      const until = Math.min(at + 65536, end), index = indices.slice([[at, until]]), values = data.slice([[at, until]]);
      for (let i = 0; i < index.length; i++) { const j = Number(index[i]); if (!Number.isInteger(j) || j < 0 || j >= minor) throw new Error('Invalid sparse index.'); visit(j, Number(values[i]), at + i); }
    }
  }
  if (encoding === 'csc_matrix') {
    // One dense column (at most 40 MB at 5M cells), never a dense marker matrix.
    const values = new Float64Array(reader.count); let done = 0;
    for (const [col, target] of selected) {
      values.fill(0); const ptr = Array.from(pointers.slice([[col, col + 2]]), pointer);
      entries(ptr[0], ptr[1], (row, value) => { values[row] += value; });
      for (let row = 0; row < reader.count; row++) if (groups[row] >= 0) consume(groups[row], target, values[row]);
      update(++done, selected.size);
    }
  } else {
    for (let start = 0; start < reader.count; start += BATCH) {
      const end = Math.min(start + BATCH, reader.count), ptr = Array.from(pointers.slice([[start, end + 1]]), pointer);
      for (let i = 0; i < end - start; i++) if (ptr[i + 1] < ptr[i]) throw new Error('Invalid sparse pointers.');
      if (groups.subarray(start, end).some(group => group >= 0)) {
        let position = 0; const row = new Map();
        const flush = () => { for (const [col, value] of row) consume(groups[start + position], selected.get(col), value); row.clear(); };
        entries(ptr[0], ptr.at(-1), (col, value, offset) => {
          while (offset >= ptr[position + 1]) { flush(); position++; }
          if (groups[start + position] >= 0 && selected.has(col)) row.set(col, (row.get(col) ?? 0) + value);
        });
        flush();
      }
      update(end, reader.count);
    }
  }
}
