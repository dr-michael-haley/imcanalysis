import { readColumn, checkFilters, matrixColumn } from './anndata.js';
import { readExpression } from './plot-reader.js';

export function sampledRows(count, limit) {
  if (!Number.isSafeInteger(count) || count < 0 || !Number.isSafeInteger(limit) || limit < 1) throw new Error('Invalid embedding sample size.');
  const size = Math.min(count, limit);
  if (size === count) return Array.from({ length: size }, (_, i) => i);
  // Jitter within disjoint strata avoids selecting just one ROI when rows have
  // a periodic/interleaved order. A fixed hash makes the sample reproducible.
  return Array.from({ length: size }, (_, i) => {
    let hash = Math.imul(i + 1, 0x45d9f3b); hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b); hash = (hash ^ (hash >>> 16)) >>> 0;
    const start = Math.floor(i * count / size), end = Math.floor((i + 1) * count / size);
    return start + Math.floor(hash / 4294967296 * (end - start));
  });
}
export function readEmbeddingCurrent(reader, { rows, colourBy, source }) {
  if (!Array.isArray(rows) || rows.some((r, i) => !Number.isInteger(r) || r < 0 || r >= reader.count || i > 0 && r <= rows[i - 1])) throw new Error('Invalid embedding rows.');
  if (colourBy.startsWith('obs:') && reader.obsKeys.includes(colourBy.slice(4))) {
    const data = readColumn(reader.obs.get(colourBy.slice(4)), rows), stored = reader.file.get(`uns/${colourBy.slice(4)}_colors`);
    return { ...data, palette: stored ? readColumn(stored).values : [] };
  }
  if (colourBy.startsWith('var:') && reader.header().sources.includes(source)) return { values: matrixColumn(reader.file.get(source), reader.markers.indexOf(colourBy.slice(4)), rows, [reader.count, reader.markers.length]), categories: null, palette: [] };
  throw new Error('Embedding colour feature is unavailable.');
}
// Read sequential blocks to avoid thousands of tiny random network reads. Retain
// only a deterministic context sample; the current ROI is read separately in full.
export function readEmbeddingContext(reader, request, progress = () => {}) {
  const { embedding, roiKey, idKey, colourBy = '', source = 'X', limit = 50000 } = request;
  const header = reader.header(), node = reader.file.get(`obsm/${embedding}`);
  if (!header.embeddings.includes(embedding) || !header.obsKeys.includes(roiKey) || !header.obsKeys.includes(idKey) || !Number.isInteger(limit) || limit < 1000 || limit > 200000) throw new Error('Invalid embedding context request.');
  if (colourBy && !(colourBy.startsWith('obs:') && header.obsKeys.includes(colourBy.slice(4))) && !(colourBy.startsWith('var:') && header.markers.includes(colourBy.slice(4)) && header.sources.includes(source))) throw new Error('Embedding colour feature is unavailable.');
  checkFilters(node);
  const rows = sampledRows(reader.count, limit), coordinates = new Float64Array(rows.length * 2), ids = new Uint32Array(rows.length), rois = [], values = [];
  const colourNode = colourBy.startsWith('obs:') ? reader.obs.get(colourBy.slice(4)) : null;
  let at = 0, categories = null;
  for (let start = 0; start < reader.count; start += 8192) {
    const end = Math.min(reader.count, start + 8192);
    if (rows[at] < end) {
      const block = Array.from({ length: end - start }, (_, i) => start + i);
      const xy = node.slice([[start, end], [0, 2]]), roi = readColumn(reader.obs.get(roiKey), block).values, id = readColumn(reader.obs.get(idKey), block).values;
      const colours = colourNode ? readColumn(colourNode, block) : null;
      if (colours?.categories) categories = [...new Set([...(categories ?? []), ...colours.categories])];
      while (at < rows.length && rows[at] < end) {
        const i = rows[at] - start, cell = Number(id[i]);
        if (!Number.isSafeInteger(cell) || cell < 1 || cell > 0xffffffff || roi[i] == null) throw new Error('Invalid cell identity in embedding context.');
        coordinates[at * 2] = Number(xy[i * 2]); coordinates[at * 2 + 1] = Number(xy[i * 2 + 1]); ids[at] = cell; rois.push(String(roi[i])); values.push(colours?.values[i] ?? null); at++;
      }
    }
    progress({ fraction: end / reader.count * .7, detail: `Reading embedding context · ${end.toLocaleString()} / ${reader.count.toLocaleString()} cells` });
  }
  if (colourBy.startsWith('var:')) {
    const groups = new Int32Array(reader.count).fill(-1); rows.forEach((row, i) => { groups[row] = i; values[i] = 0; });
    readExpression(reader, source, [colourBy.slice(4)], groups, (index, column, value) => { values[index] = value; }, p => progress({ ...p, fraction: .7 + p.fraction * .3 }));
  }
  const stored = colourNode && reader.file.get(`uns/${colourBy.slice(4)}_colors`);
  return { coordinates, ids, rois, values, categories, palette: stored ? readColumn(stored).values : [], sampled: rows.length < reader.count, total: reader.count };
}
