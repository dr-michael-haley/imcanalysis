// Nimbus / NapariSBT compatibility: nimbus_normalization.py and
// _napari_imc_normalization.py are the reference for formats and image scaling.
const fieldName = value => String(value).trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
const cleanName = value => String(value).toLowerCase().replace(/[^\p{L}\p{N}_]/gu, '');
const aliasName = value => String(value).toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);

export function csvRows(text, description = 'normalisation') {
  const rows = []; let row = [], value = '', quoted = false, closed = false;
  for (let i = 0; i <= text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === undefined) throw new Error(`Unclosed quote in ${description} CSV.`);
      if (char === '"') {
        if (text[i + 1] === '"') { value += '"'; i++; }
        else { quoted = false; closed = true; }
      } else value += char;
    } else if (char === ',' || char === '\n' || char === '\r' || char === undefined) {
      row.push(value); value = ''; closed = false;
      if (char !== ',') { if (row.some(cell => cell.trim())) rows.push(row); row = []; }
      if (char === '\r' && text[i + 1] === '\n') i++;
    } else if (char === '"') {
      if (value || closed) throw new Error(`Unexpected quote in ${description} CSV.`);
      quoted = true;
    } else {
      if (closed && char.trim()) throw new Error('Unexpected text after a quoted CSV value.');
      if (!closed) value += char;
    }
  }
  return rows;
}

function numeric(value, name) {
  if (typeof value !== 'number' && !(typeof value === 'string' && /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim()))) throw new Error(`${name} must be numeric.`);
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${name} must be finite.`);
  return number;
}

export function parseNormalization(text, filename) {
  text = text.replace(/^\uFEFF/, '');
  let entries;
  if (/\.csv$/i.test(filename)) {
    const [header, ...rows] = csvRows(text);
    if (!header) throw new Error('Normalisation CSV is empty.');
    const fields = header.map(fieldName);
    if (new Set(fields).size !== fields.length) throw new Error('Duplicate normalisation CSV column names.');
    const column = names => names.map(name => fields.indexOf(name)).find(index => index >= 0);
    const marker = column(['marker', 'channel']);
    const vmax = column(['vmax', 'baseline_vmax', 'baseline', 'value']);
    const lower = column(['lower_threshold', 'lower_bound', 'lower', 'vmin']);
    if (marker === undefined || vmax === undefined) throw new Error('Normalisation CSV needs marker and vmax columns (legacy Marker,Value is also accepted).');
    entries = rows.map(row => {
      if (row.length > fields.length) throw new Error('A normalisation CSV row has extra columns. Quote marker names containing commas.');
      return [row[marker] ?? '', { vmax: row[vmax], lower_threshold: lower === undefined ? 0 : row[lower] }];
    });
  } else if (/\.json$/i.test(filename)) {
    let payload;
    try { payload = JSON.parse(text); } catch { throw new Error('Invalid normalisation JSON.'); }
    if (record(payload?.normalization_dict)) payload = payload.normalization_dict;
    if (!record(payload)) throw new Error('Normalisation JSON must map marker names to values or bounds.');
    entries = Object.entries(payload);
  } else throw new Error('Choose a normalisation .csv or .json file.');
  if (!entries.length) throw new Error('The normalisation dictionary contains no markers.');
  const result = new Map(), seen = new Set();
  for (const [rawMarker, raw] of entries) {
    const marker = rawMarker.trim(), folded = marker.toLowerCase();
    if (!marker) throw new Error('The normalisation dictionary contains an empty marker name.');
    if (seen.has(folded)) throw new Error(`Duplicate normalisation marker: ${marker}.`);
    seen.add(folded);
    const fields = record(raw) ? Object.fromEntries(Object.entries(raw).map(([key, value]) => [fieldName(key), value])) : { vmax: raw };
    const vmax = numeric(fields.vmax ?? fields.value, `${marker} Vmax`);
    const lower = fields.lower_threshold ?? fields.lower ?? fields.vmin ?? 0;
    const lower_threshold = numeric(lower === '' ? 0 : lower, `${marker} lower threshold`);
    if (vmax <= 0 || lower_threshold < 0 || lower_threshold >= vmax) throw new Error(`${marker}: require 0 ≤ lower_threshold < vmax, with positive Vmax.`);
    result.set(marker, { vmax, lower_threshold });
  }
  return result;
}

export function findNormalization(mapping, candidates, aliases = new Map()) {
  if (!mapping) return null;
  const entries = [...mapping];
  const match = matches => {
    if (matches.length > 1) throw new Error(`Ambiguous normalisation match: ${matches.map(([key]) => key).join(', ')}. Use an exact channel name.`);
    return matches.length ? { marker: matches[0][0], ...matches[0][1] } : null;
  };
  for (const candidate of candidates.filter(Boolean)) {
    const exact = match(entries.filter(([key]) => key.toLowerCase() === candidate.trim().toLowerCase()));
    if (exact) return exact;
    const clean = match(entries.filter(([key]) => cleanName(key) === cleanName(candidate)));
    if (clean) return clean;
  }
  // Use only the existing collision-aware AnnData panel aliases for extra names.
  const canonical = aliases.get(aliasName(candidates[0]));
  return canonical ? match(entries.filter(([key]) => aliases.get(aliasName(key)) === canonical)) : null;
}

export function normalizeImage(values, { bounds = null, quantile = 0.999, minimum = 0.1 } = {}) {
  if (!(quantile > 0 && quantile <= 1) || !Number.isFinite(minimum) || minimum < 0) throw new Error('Invalid normalisation fallback settings.');
  // NapariSBT thresholds the input before either dictionary or quantile scaling.
  const normalized = Float32Array.from(values, value => value > minimum ? value : 0);
  let vmax = bounds?.vmax, lower_threshold = bounds?.lower_threshold ?? 0;
  if (!bounds) {
    // Exact linear-interpolated quantile, like numpy.quantile; runs in the TIFF worker.
    const sorted = normalized.slice().sort();
    const position = (sorted.length - 1) * quantile, low = Math.floor(position), high = Math.ceil(position);
    vmax = sorted[low] + (sorted[high] - sorted[low]) * (position - low);
    if (vmax < 5) vmax = 3;
  }
  if (!Number.isFinite(vmax) || vmax <= lower_threshold) throw new Error('Cannot normalise this image: its intensity scale is not finite and positive.');
  for (let i = 0; i < normalized.length; i++) normalized[i] = Math.max(0, Math.min(1, (normalized[i] - lower_threshold) / (vmax - lower_threshold)));
  return { data: normalized, normalization: { vmax, lower_threshold, marker: bounds?.marker ?? null } };
}
