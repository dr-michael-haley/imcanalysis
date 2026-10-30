// AnnData decoding is independent of the browser so real HDF5 fixtures can test it.
// A total-row allocation guard, not an HDF5 limit. ROI indexing still uses RAM.
const MAX_CELLS = 5_000_000;
const MAX_VALUES = 8_000_000;

export function checkFilters(node) {
  // Reject unavailable filters before HDF5 enters its native abort path. The
  // current upstream LZF plugin cannot link against this WebAssembly runtime.
  for (const filter of node.filters ?? []) {
    if (filter.id > 6) throw new Error(`Unsupported HDF5 compression: ${filter.name || filter.id}. This version reads uncompressed or standard HDF5 filters such as gzip; LZF/Blosc plugins are not enabled.`);
  }
}

export function attribute(node, key, fallback = undefined) {
  return node?.attrs?.[key]?.value ?? fallback;
}

function scalar(value) {
  if (typeof value === 'bigint') {
    const number = Number(value);
    if (!Number.isSafeInteger(number)) throw new Error('An integer exceeds JavaScript’s exact range.');
    return number;
  }
  if (value !== null && !['number', 'string', 'boolean'].includes(typeof value)) throw new Error('Compound or reference-valued AnnData columns are not supported.');
  return value;
}

function array(value) {
  if (value == null) throw new Error('HDF5 could not decode this dataset. Check its compression and encoding.');
  return Array.from(value, scalar);
}

function runs(rows, size = 2048) {
  const result = [];
  for (let i = 0; i < rows.length;) {
    let j = i + 1;
    while (j < rows.length && rows[j] === rows[j - 1] + 1 && j - i < size) j++;
    result.push([rows[i], rows[j - 1] + 1, i]);
    i = j;
  }
  return result;
}

function readVector(node, rows = null) {
  if (!node || node.shape?.length !== 1) throw new Error('Expected a one-dimensional AnnData column.');
  checkFilters(node);
  if (rows == null) {
    if (node.shape[0] > MAX_VALUES) throw new Error('This column exceeds the viewer’s memory limit.');
    return array(node.value);
  }
  const output = [];
  for (const [start, end] of runs(rows)) output.push(...array(node.slice([[start, end]])));
  return output;
}

export function readColumn(node, rows = null) {
  const encoding = attribute(node, 'encoding-type', 'array');
  if (encoding === 'categorical') {
    const categories = readVector(node.get('categories'));
    const codes = readVector(node.get('codes'), rows);
    const values = codes.map(code => {
      if (code === -1) return null;
      if (!Number.isInteger(code) || code < 0 || code >= categories.length) throw new Error('Invalid categorical code.');
      return categories[code];
    });
    return { values, categories };
  }
  if (['nullable-integer', 'nullable-boolean', 'nullable-string-array'].includes(encoding)) {
    const values = readVector(node.get('values'), rows);
    const mask = readVector(node.get('mask'), rows);
    if (values.length !== mask.length) throw new Error('Invalid nullable column.');
    const decoded = values.map((value, i) => mask[i] ? null : value);
    return { values: decoded, categories: encoding === 'nullable-string-array' ? [...new Set(decoded.filter(value => value != null))] : null };
  }
  if (!['array', 'string-array'].includes(encoding)) throw new Error(`Unsupported AnnData column encoding: ${encoding}`);
  const values = readVector(node, rows);
  return { values, categories: values.some(value => typeof value === 'string') ? [...new Set(values.filter(value => value != null))] : null };
}

export function matrixColumn(node, column, rows, expectedShape) {
  if (!node) throw new Error('Expression source is missing.');
  const encoding = attribute(node, 'encoding-type', 'array');
  const shape = node.shape ?? array(attribute(node, 'shape'));
  if (shape.length !== 2 || shape.some((value, i) => Number(value) !== expectedShape[i])) {
    throw new Error('Expression matrix shape does not match obs and var.');
  }
  if (!Number.isInteger(column) || column < 0 || column >= Number(shape[1])) throw new Error('Marker is outside the expression matrix.');
  const result = new Float64Array(rows.length);
  if (encoding === 'array') {
    if (![0, 1].includes(node.metadata.type)) throw new Error('Expression matrices must contain real numeric values.');
    checkFilters(node);
    for (const [start, end, offset] of runs(rows)) result.set(array(node.slice([[start, end], [column, column + 1]])), offset);
    return result;
  }
  if (!['csr_matrix', 'csc_matrix'].includes(encoding)) throw new Error(`Unsupported expression encoding: ${encoding}`);
  const pointers = node.get('indptr');
  const indices = node.get('indices');
  const data = node.get('data');
  const major = Number(shape[encoding === 'csr_matrix' ? 0 : 1]);
  const minor = Number(shape[encoding === 'csr_matrix' ? 1 : 0]);
  if (pointers?.shape?.length !== 1 || pointers.shape[0] !== major + 1 || indices?.shape?.length !== 1 || data?.shape?.length !== 1 || indices.shape[0] !== data.shape[0]) throw new Error('Invalid sparse matrix structure.');
  if (![0, 1].includes(data.metadata.type)) throw new Error('Expression matrices must contain real numeric values.');
  for (const dataset of [pointers, indices, data]) checkFilters(dataset);
  const readEntries = (start, end) => {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > data.shape[0] || end > indices.shape[0]) throw new Error('Invalid sparse matrix pointers.');
    if (end - start > MAX_VALUES) throw new Error('Sparse slice exceeds the viewer’s memory limit.');
    if (end === start) return [[], []];
    const index = array(indices.slice([[start, end]]));
    if (index.some(value => !Number.isInteger(value) || value < 0 || value >= minor)) throw new Error('Invalid sparse matrix index.');
    return [index, array(data.slice([[start, end]]))];
  };
  if (encoding === 'csc_matrix') {
    const [start, end] = array(pointers.slice([[column, column + 2]]));
    const [index, values] = readEntries(start, end);
    if (rows.length && rows.at(-1) - rows[0] + 1 === rows.length) {
      index.forEach((row, i) => { if (row >= rows[0] && row <= rows.at(-1)) result[row - rows[0]] += values[i]; });
    } else {
      const positions = new Map(rows.map((row, i) => [row, i]));
      index.forEach((row, i) => { if (positions.has(row)) result[positions.get(row)] += values[i]; });
    }
  } else {
    for (const [first, last, offset] of runs(rows)) {
      const ptr = array(pointers.slice([[first, last + 1]]));
      const start = ptr[0];
      const [index, values] = readEntries(start, ptr.at(-1));
      for (let r = 0; r < last - first; r++) {
        if (ptr[r + 1] < ptr[r]) throw new Error('Invalid sparse matrix pointers.');
        for (let j = ptr[r] - start; j < ptr[r + 1] - start; j++) if (index[j] === column) result[offset + r] += values[j];
      }
    }
  }
  return result;
}

export class AnnDataReader {
  constructor(file) {
    this.file = file;
    this.obs = file.get('obs');
    this.var = file.get('var');
    if (!this.obs || !this.var) throw new Error('This file does not contain AnnData obs and var tables.');
    this.obsIndex = attribute(this.obs, '_index', '_index');
    this.varIndex = attribute(this.var, '_index', '_index');
    const index = this.obs.get(this.obsIndex);
    this.count = index?.shape?.[0] ?? index?.get?.('codes')?.shape?.[0];
    if (!Number.isInteger(this.count) || this.count < 1) throw new Error('AnnData has no readable cell index.');
    if (this.count > MAX_CELLS) throw new Error(`This first version supports up to ${MAX_CELLS.toLocaleString()} cells.`);
    this.markers = readColumn(this.var.get(this.varIndex)).values.map(String);
    if (new Set(this.markers).size !== this.markers.length) throw new Error('AnnData variable names must be unique.');
    this.obsKeys = this.obs.keys().filter(key => key !== this.obsIndex);
    this.rois = new Map();
  }

  header() {
    if (this.cachedHeader) return this.cachedHeader;
    const optionalVar = key => this.var.get(key) ? readColumn(this.var.get(key)).values : [];
    const obsm = this.file.get('obsm');
    const embeddings = obsm ? obsm.keys().filter(key => {
      const shape = obsm.get(key).shape;
      return shape?.length === 2 && shape[0] === this.count && shape[1] >= 2;
    }) : [];
    this.cachedHeader = { count: this.count, obsKeys: this.obsKeys, markers: this.markers,
      channelNames: optionalVar('channel_name'), channelLabels: optionalVar('channel_label'),
      sources: [...(this.file.get('X') ? ['X'] : []), ...(this.file.get('layers')?.keys() ?? []).map(key => `layers/${key}`)], embeddings };
    for (const labels of [this.cachedHeader.channelNames, this.cachedHeader.channelLabels]) {
      if (labels.length && labels.length !== this.markers.length) throw new Error('Channel metadata length does not match var.');
    }
    return this.cachedHeader;
  }

  configure(roiKey, idKey, progress = () => {}) {
    if (!this.obsKeys.includes(roiKey) || !this.obsKeys.includes(idKey) || roiKey === idKey) throw new Error('Choose separate ROI and object ID observation columns.');
    progress({ detail: `Reading ROI column · ${roiKey}…` });
    const rois = readColumn(this.obs.get(roiKey)).values;
    progress({ detail: `Reading cell identity column · ${idKey}…` });
    const ids = readColumn(this.obs.get(idKey)).values;
    if (rois.length !== this.count || ids.length !== this.count) throw new Error('Identity columns do not match the number of cells.');
    const groups = new Map();
    for (let i = 0; i < this.count; i++) {
      if (i % 100000 === 0) progress({ detail: `Indexing cell identities · ${i.toLocaleString()} / ${this.count.toLocaleString()}` });
      if (rois[i] == null || String(rois[i]).trim() === '') throw new Error(`Missing ROI at row ${i}.`);
      const roi = String(rois[i]);
      const id = typeof ids[i] === 'string' && /^\d+$/.test(ids[i]) ? Number(ids[i]) : ids[i];
      if (!Number.isSafeInteger(id) || id <= 0 || id > 4294967295) throw new Error(`Invalid mask object ID at row ${i}; expected an integer from 1 to 4294967295.`);
      if (!groups.has(roi)) groups.set(roi, { rows: [], ids: [], seen: new Set() });
      const group = groups.get(roi);
      if (group.seen.has(id)) throw new Error(`Duplicate cell identity: ROI ${roi}, object ${id}.`);
      group.seen.add(id); group.rows.push(i); group.ids.push(id);
    }
    this.rois = groups;
    this.rankCache = new Map();
    this.roiNames = [...groups.keys()];
    this.rowROI = new Uint32Array(this.count);
    this.roiNames.forEach((name, index) => { for (const row of groups.get(name).rows) this.rowROI[row] = index; });
    for (const group of groups.values()) delete group.seen;
    return [...groups].map(([name, group]) => ({ name, count: group.rows.length })).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  }

  validateFlagCells(fields) {
    const cache = new Map();
    for (const field of fields) for (const [roi, id] of field.assignments) {
      if (!cache.has(roi)) cache.set(roi, new Set(this.rois.get(roi)?.ids ?? []));
      if (!cache.get(roi).has(id)) throw new Error(`Flagged cell is absent from this dataset: ${roi}, object ${id}.`);
    }
    return true;
  }

  region(roi, colourBy, source, embedding) {
    const group = this.rois.get(roi);
    if (!group) throw new Error(`Unknown ROI: ${roi}`);
    const { rows, ids } = group;
    let values = ids;
    let categories = null;
    let palette = [];
    if (colourBy?.startsWith('obs:')) {
      const key = colourBy.slice(4);
      if (!this.obsKeys.includes(key)) throw new Error('Unknown observation.');
      const node = this.obs.get(key);
      const encoding = attribute(node, 'encoding-type');
      if (encoding === 'string-array' || encoding === 'nullable-string-array' || node.metadata?.type === 3) {
        // A global category order keeps string labels the same colour in every ROI.
        // Keep only the current string column, not every observation in memory.
        if (this.stringColumn?.key !== key) this.stringColumn = { key, ...readColumn(node) };
        values = rows.map(row => this.stringColumn.values[row]);
        categories = this.stringColumn.categories;
      } else ({ values, categories } = readColumn(node, rows));
      if (values.length !== ids.length) throw new Error('Observation length does not match the selected cells.');
      const colours = this.file.get(`uns/${key}_colors`);
      if (colours && categories && encoding === 'categorical') palette = readVector(colours);
    } else if (colourBy?.startsWith('var:')) {
      if (!this.header().sources.includes(source)) throw new Error('Choose an available expression source.');
      values = matrixColumn(this.file.get(source), this.markers.indexOf(colourBy.slice(4)), rows, [this.count, this.markers.length]);
    }
    let coordinates = null;
    if (embedding) {
      const node = this.file.get(`obsm/${embedding}`);
      if (node?.shape?.length !== 2 || node.shape[0] !== this.count || node.shape[1] < 2) throw new Error('Invalid embedding shape.');
      checkFilters(node);
      coordinates = new Float64Array(rows.length * 2);
      for (const [start, end, offset] of runs(rows)) coordinates.set(array(node.slice([[start, end], [0, 2]])), offset * 2);
    }
    return { rows, ids, values, categories, palette, coordinates };
  }

  cell(row) {
    if (!Number.isInteger(row) || row < 0 || row >= this.count) throw new Error('Invalid cell row.');
    const fields = [['Cell', readColumn(this.obs.get(this.obsIndex), [row]).values[0]]];
    for (const key of this.obsKeys) {
      try { fields.push([key, readColumn(this.obs.get(key), [row]).values[0]]); }
      catch (error) { fields.push([key, `Unavailable: ${error.message}`]); }
    }
    return fields;
  }

  categories(key) {
    if (!this.obsKeys.includes(key)) throw new Error('Unknown observation.');
    const node = this.obs.get(key);
    if (attribute(node, 'encoding-type') === 'categorical') {
      if (node.get('categories')?.shape?.[0] > 10000) throw new Error('This observation has more than 10,000 categories. Choose a different ordering field.');
      return readVector(node.get('categories'));
    }
    const values = new Map();
    for (let start = 0; start < this.count; start += 16384) {
      const rows = Array.from({ length: Math.min(16384, this.count - start) }, (_, i) => start + i);
      for (const value of readColumn(node, rows).values) if (value !== null) values.set(JSON.stringify(value), value);
      if (values.size > 10000) throw new Error('This observation has more than 10,000 values. Use mean observation ordering for continuous values.');
    }
    return [...values.values()].sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
  }

  populationSummary(key) {
    if (!this.obsKeys.includes(key)) throw new Error('Choose an existing observation.');
    const node = this.obs.get(key), categorical = attribute(node, 'encoding-type') === 'categorical';
    const categories = categorical ? this.categories(key) : [];
    const entries = new Map(categories.map(source => [JSON.stringify(source), { source, count: 0 }]));
    // Count in bounded slices; only the compact population table crosses to UI.
    for (let start = 0; start < this.count; start += 16384) {
      const rows = Array.from({ length: Math.min(16384, this.count - start) }, (_, i) => start + i);
      for (const value of readColumn(node, rows).values) {
        const source = value == null || (typeof value === 'number' && !Number.isFinite(value)) ? null : value;
        const token = JSON.stringify(source);
        if (!entries.has(token)) entries.set(token, { source, count: 0 });
        entries.get(token).count++;
        if (entries.size > 10000) throw new Error('This observation has more than 10,000 populations. Choose a categorical or cluster observation.');
      }
    }
    const colours = categorical && this.file.get(`uns/${key}_colors`);
    return { rows: [...entries.values()], palette: colours ? readVector(colours) : [] };
  }

  rank(spec) {
    if (!this.rois.size) throw new Error('Configure cell identities before ordering ROIs.');
    const { mode, field, value, source, direction } = spec;
    if (!['name', 'cells', 'population', 'obs', 'marker'].includes(mode) || !['asc', 'desc'].includes(direction)) throw new Error('Invalid ROI ordering.');
    const key = JSON.stringify(spec);
    if (this.rankCache.has(key)) return this.rankCache.get(key);
    const names = this.roiNames, sums = new Float64Array(names.length), counts = new Uint32Array(names.length);
    if (mode === 'population' || mode === 'obs') { if (!this.obsKeys.includes(field)) throw new Error('Choose an available observation.'); }
    if (mode === 'marker' && (!this.markers.includes(field) || !this.header().sources.includes(source))) throw new Error('Choose an available marker and expression source.');
    if (['population', 'obs', 'marker'].includes(mode)) {
      // Read one field in bounded row batches. Never materialise the full matrix.
      // CSC is already column-oriented; reading its one selected column once avoids
      // re-scanning that sparse column for every row batch.
      const matrix = mode === 'marker' ? this.file.get(source) : null;
      const csc = matrix && attribute(matrix, 'encoding-type') === 'csc_matrix'
        ? matrixColumn(matrix, this.markers.indexOf(field), Uint32Array.from({ length: this.count }, (_, i) => i), [this.count, this.markers.length]) : null;
      for (let start = 0; start < this.count; start += 16384) {
        const rows = Array.from({ length: Math.min(16384, this.count - start) }, (_, i) => start + i);
        const values = mode === 'marker' ? (csc ? csc.subarray(start, start + rows.length) : matrixColumn(matrix, this.markers.indexOf(field), rows, [this.count, this.markers.length])) : readColumn(this.obs.get(field), rows).values;
        values.forEach((item, i) => {
          const group = this.rowROI[start + i];
          if (mode === 'population') { if (JSON.stringify(item) === value) sums[group]++; }
          else if (typeof item === 'number' && Number.isFinite(item)) { sums[group] += item; counts[group]++; }
        });
      }
      if (mode !== 'population' && !counts.some(Boolean)) throw new Error('No finite numeric values are available for this ordering.');
    }
    const result = names.map((name, i) => ({ name, count: this.rois.get(name).rows.length, score: mode === 'name' ? null : mode === 'cells' ? this.rois.get(name).rows.length : mode === 'population' ? sums[i] : counts[i] ? sums[i] / counts[i] : null }));
    const alphabetical = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true });
    result.sort((a, b) => mode === 'name' ? alphabetical(a, b) * (direction === 'asc' ? 1 : -1) : a.score === null ? (b.score === null ? alphabetical(a, b) : 1) : b.score === null ? -1 : (a.score - b.score) * (direction === 'asc' ? 1 : -1) || alphabetical(a, b));
    if (this.rankCache.size >= 8) this.rankCache.delete(this.rankCache.keys().next().value);
    this.rankCache.set(key, result); return result;
  }
}
