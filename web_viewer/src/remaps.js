import { colourScale } from './core.js';

export const sourceKey = value => JSON.stringify(value == null || (typeof value === 'number' && !Number.isFinite(value)) ? null : value);
export const sourceLabel = value => value === null ? '(Missing)' : typeof value === 'string' ? value || '(Empty string)' : String(value);
const hex = value => `#${value.map(channel => channel.toString(16).padStart(2, '0')).join('')}`;
const validColour = value => typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value);
function text(value, label, max, nonempty = false) {
  if (typeof value !== 'string' || value.length > max || (nonempty && !value.trim())) throw new Error(`Invalid remap ${label}.`);
  return value;
}

export function validateRemaps(input) {
  if (!Array.isArray(input) || input.length > 20) throw new Error('A workspace supports up to 20 population remaps.');
  const ids = new Set(), names = new Set();
  return input.map(plan => {
    if (!plan || !Array.isArray(plan.rows) || !plan.rows.length || plan.rows.length > 10000) throw new Error('A remap requires 1–10,000 source populations.');
    const id = text(plan.id, 'identifier', 80, true), name = text(plan.name, 'name', 120, true).trim(), source = text(plan.source, 'source observation', 16384, true);
    if (!/^[a-zA-Z0-9_-]+$/.test(id) || ids.has(id) || names.has(name)) throw new Error('Remap identifiers and names must be unique.');
    ids.add(id); names.add(name);
    const keys = new Set(), colours = new Map();
    const rows = plan.rows.map(row => {
      if (!row || (row.source !== null && !['string', 'number', 'boolean'].includes(typeof row.source)) || (typeof row.source === 'number' && !Number.isFinite(row.source))) throw new Error('Invalid remap source value.');
      if (typeof row.source === 'string') text(row.source, 'source value', 16384);
      const key = sourceKey(row.source), label = text(row.label, 'proposed name', 256, true).trim();
      if (keys.has(key)) throw new Error('Duplicate source population in remap.');
      keys.add(key);
      if (!validColour(row.colour)) throw new Error('Remap colours must use #RRGGBB.');
      const colour = row.colour.toLowerCase();
      if (colours.has(label) && colours.get(label) !== colour) throw new Error(`Merged population “${label}” has conflicting colours.`);
      colours.set(label, colour);
      if (!Number.isSafeInteger(row.count) || row.count < 0 || row.count > 5000000) throw new Error('Invalid remap cell count.');
      return { source: row.source, label, colour, notes: text(row.notes, 'notes', 4000), count: row.count };
    });
    return { id, name, source, rows };
  });
}

export function createRemap(id, name, source, summary) {
  const categories = summary.rows.map(row => row.source), colours = colourScale([], categories, summary.palette);
  const used = new Set();
  const rows = summary.rows.map(row => {
    let label = sourceLabel(row.source), suffix = 1;
    const base = label;
    // Numeric 1, string "1", null and the literal "(Missing)" stay separate
    // until the reviewer deliberately merges them.
    while (used.has(label)) label = `${base} (${typeof row.source} ${suffix++})`;
    used.add(label);
    return { ...row, label, colour: hex(colours.colour(row.source)), notes: '' };
  });
  return validateRemaps([{ id, name, source, rows }])[0];
}

export function editRemap(plan, keys, patch) {
  const selected = new Set(keys), rows = plan.rows.map(row => ({ ...row }));
  const changed = rows.filter(row => selected.has(sourceKey(row.source)));
  if (!changed.length) return plan;
  if (patch.label !== undefined) {
    const label = text(patch.label, 'proposed name', 256, true).trim();
    const colour = rows.find(row => row.label === label)?.colour ?? changed[0].colour;
    for (const row of changed) { row.label = label; row.colour = colour; }
  }
  if (patch.colour !== undefined) {
    if (!validColour(patch.colour)) throw new Error('Choose a valid population colour.');
    const labels = new Set(changed.map(row => row.label));
    for (const row of rows) if (labels.has(row.label)) row.colour = patch.colour.toLowerCase();
  }
  if (patch.notes !== undefined) for (const row of changed) row.notes = text(patch.notes, 'notes', 4000);
  return { ...plan, rows };
}

export function validateRemapSource(plan, summary) {
  const expected = new Map(summary.rows.map(row => [sourceKey(row.source), row.count]));
  if (plan.rows.length !== expected.size || plan.rows.some(row => expected.get(sourceKey(row.source)) !== row.count)) throw new Error(`Remap “${plan.name}” does not match the populations/counts in ${plan.source}. Reopen the original AnnData or create a new remap.`);
}

export function applyRemap(region, plan) {
  const rows = new Map(plan.rows.map(row => [sourceKey(row.source), row]));
  const groups = new Map(plan.rows.map(row => [row.label, row.colour]));
  const values = Array.from(region.values, value => {
    const row = rows.get(sourceKey(value));
    if (!row) throw new Error(`Remap “${plan.name}” has no entry for ${sourceLabel(value)}.`);
    return row.label;
  });
  return { ...region, values, categories: [...groups.keys()], palette: [...groups.values()] };
}

export function remapRecipe(plan) {
  return plan ? { source: plan.source, rows: plan.rows.map(({ source, label, colour }) => [source, label, colour]) } : null;
}

export function remapCSV(plan) {
  const quote = value => `"${String(value).replaceAll('"', '""')}"`;
  const rows = [['source_obs', 'source_value', 'source_value_json', 'proposed_name', 'colour', 'notes', 'remap_name'],
    ...plan.rows.map(row => [plan.source, sourceLabel(row.source), sourceKey(row.source), row.label, row.colour, row.notes, plan.name])];
  return rows.map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
}
