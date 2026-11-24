import { csvRows } from './normalization.js';

export const qcColumns = [
  'area_px',
  'bbox_occupancy',
  'axis_ratio',
  'solidity',
  'edge_perimeter_px',
  'touches_edge',
  'dna_mean',
  'dna_std',
  'dna_max',
  'dna_core_to_border'
];

function hullArea(points) {
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  function half(order) {
    const out = [];
    for (const p of order) {
      while (out.length > 1 && cross(out.at(-2), out.at(-1), p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  }
  const hull = half(points).concat(half(points.reverse()));
  let area = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i],
      b = hull[(i + 1) % hull.length];
    area += a[0] * b[1] - a[1] * b[0];
  }
  return Math.abs(area) / 2;
}

export function measureObjects(mask, dna, imported = null) {
  const { width, height, data } = mask;
  if (
    width !== dna.width ||
    height !== dna.height ||
    data.length !== width * height ||
    dna.data.length !== data.length
  )
    throw Error('The mask and DNA image must have identical pixel dimensions.');
  const cells = new Map();
  for (let y = 0; y < height; y++) {
    const spans = new Map();
    for (let x = 0; x < width; x++) {
      const i = y * width + x,
        id = data[i];
      if (id === 0) continue;
      if (!Number.isInteger(id) || id <= 0 || id > 0xffffffff)
        throw Error('Mask IDs must be non-negative uint32 integers.');
      let a = cells.get(id);
      if (!a) {
        a = {
          n: 0,
          x: 0,
          y: 0,
          xx: 0,
          yy: 0,
          xy: 0,
          minX: x,
          maxX: x,
          minY: y,
          maxY: y,
          perimeter: 0,
          points: [],
          edge: 0,
          count: 0,
          mean: 0,
          m2: 0,
          max: -Infinity,
          core: 0,
          coreN: 0,
          border: 0,
          borderN: 0
        };
        cells.set(id, a);
      }
      const span = spans.get(id);
      if (span) span[1] = x;
      else spans.set(id, [x, x]);
      a.n++;
      a.x += x;
      a.y += y;
      a.xx += x * x;
      a.yy += y * y;
      a.xy += x * y;
      a.minX = Math.min(a.minX, x);
      a.maxX = Math.max(a.maxX, x);
      a.maxY = y;
      const edges =
        Number(x === 0 || data[i - 1] !== id) +
        Number(x === width - 1 || data[i + 1] !== id) +
        Number(y === 0 || data[i - width] !== id) +
        Number(y === height - 1 || data[i + width] !== id);
      a.perimeter += edges;
      if (!x || !y || x === width - 1 || y === height - 1) a.edge = 1;
      const v = dna.data[i];
      if (Number.isFinite(v)) {
        a.count++;
        const delta = v - a.mean;
        a.mean += delta / a.count;
        a.m2 += delta * (v - a.mean);
        a.max = Math.max(a.max, v);
        if (edges) {
          a.border += v;
          a.borderN++;
        } else {
          a.core += v;
          a.coreN++;
        }
      }
    }
    for (const [id, [left, right]] of spans)
      cells.get(id).points.push([left, y], [right + 1, y], [left, y + 1], [right + 1, y + 1]);
  }
  const rows = [];
  for (const [id, a] of cells) {
    const xx = a.xx / a.n - (a.x / a.n) ** 2 + 1 / 12,
      yy = a.yy / a.n - (a.y / a.n) ** 2 + 1 / 12,
      xy = a.xy / a.n - (a.x * a.y) / a.n ** 2;
    const diff = Math.hypot(xx - yy, 2 * xy);
    const values = [
      a.n,
      a.n / ((a.maxX - a.minX + 1) * (a.maxY - a.minY + 1)),
      Math.sqrt((xx + yy + diff) / (xx + yy - diff)),
      a.n / hullArea(a.points),
      a.perimeter,
      a.edge,
      a.count ? a.mean : null,
      a.count ? Math.sqrt(a.m2 / a.count) : null,
      a.count ? a.max : null,
      a.coreN && a.borderN && a.border ? a.core / a.coreN / (a.border / a.borderN) : null
    ];
    if (imported) values.push(...(imported.byId[id] ?? imported.columns.map(() => null)));
    rows.push({ id, values });
  }
  return { columns: qcColumns.concat(imported?.columns ?? []), rows };
}

export function readMetrics(csv) {
  const [head, ...rows] = csvRows(csv.replace(/^\uFEFF/, ''), 'cell metrics');
  const names = (head || []).map(s => s.trim()),
    roi = names.indexOf('ROI'),
    obj = names.indexOf('ObjectNumber');
  const columns = names.filter(s => /^(mask_|dna_raw_|dna_preprocessed_)[a-z0-9_]+$/i.test(s));
  if (roi < 0 || obj < 0 || !columns.length || new Set(names).size !== names.length)
    throw Error('Metrics CSV needs ROI, ObjectNumber and uniquely named mask or DNA feature columns.');
  const indices = columns.map(s => names.indexOf(s)),
    groups = new Map();
  for (const row of rows) {
    const id = Number(row[obj]),
      key = row[roi]?.trim();
    if (!key || !Number.isInteger(id) || id <= 0 || id > 0xffffffff)
      throw Error('Invalid ROI or ObjectNumber in metrics CSV.');
    if (!groups.has(key)) groups.set(key, {});
    const group = groups.get(key);
    if (Object.hasOwn(group, id)) throw Error('Duplicate object ' + id + ' in ' + key + '.');
    group[id] = indices.map(i => {
      const s = row[i]?.trim();
      if (/^(true|false)$/i.test(s)) return Number(s.toLowerCase() === 'true');
      return s && Number.isFinite(Number(s)) ? Number(s) : null;
    });
  }
  return { columns, groups };
}

export const scoreUncertainty = score => 1 - 2 * Math.abs(score - 0.5);

// Each unlabelled, scored cell contributes once, irrespective of ROI size.
export function roiScoreSummary(scores, labels, threshold) {
  let good = 0, artefact = 0, sum = 0;
  for (const [id, score] of scores) {
    if (labels[id] || !Number.isFinite(score)) continue;
    if (score >= threshold) artefact++;
    else good++;
    sum += scoreUncertainty(score);
  }
  return { good, artefact, uncertainty: good + artefact ? sum / (good + artefact) : null };
}

export function decision(confirmed, score, threshold) {
  if (confirmed === 'good' || confirmed === 'artefact')
    return { keep: confirmed === 'good', reason: 'confirmed' };
  if (!Number.isFinite(score)) return { keep: true, reason: 'unscored' };
  return { keep: score < threshold, reason: 'model' };
}

// Baseline uncompressed unsigned TIFF. Preserve the integer label width needed
// by the original mask, including high IDs even when those objects are removed.
export function cleanedTiff(mask, remove) {
  let max = 0;
  for (const v of mask.data) max = Math.max(max, v);
  const bytes = max > 65535 ? 4 : 2,
    tags = 10,
    offset = 8 + 2 + 12 * tags + 4;
  const length = mask.data.length * bytes,
    buffer = new ArrayBuffer(offset + length),
    view = new DataView(buffer);
  view.setUint16(0, 0x4949, true);
  view.setUint16(2, 42, true);
  view.setUint32(4, 8, true);
  view.setUint16(8, tags, true);
  let pos = 10;
  for (const [tag, type, value] of [
    [256, 4, mask.width],
    [257, 4, mask.height],
    [258, 3, bytes * 8],
    [259, 3, 1],
    [262, 3, 1],
    [273, 4, offset],
    [277, 3, 1],
    [278, 4, mask.height],
    [279, 4, length],
    [339, 3, 1]
  ]) {
    view.setUint16(pos, tag, true);
    view.setUint16(pos + 2, type, true);
    view.setUint32(pos + 4, 1, true);
    if (type === 3) view.setUint16(pos + 8, value, true);
    else view.setUint32(pos + 8, value, true);
    pos += 12;
  }
  for (let i = 0; i < mask.data.length; i++) {
    const v = mask.data[i],
      kept = remove.has(v) ? 0 : v;
    if (bytes === 2) view.setUint16(offset + i * 2, kept, true);
    else view.setUint32(offset + i * 4, kept, true);
  }
  return buffer;
}
