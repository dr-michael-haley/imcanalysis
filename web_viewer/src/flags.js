export const cellKey = (roi, id) => JSON.stringify([roi, id]);
export function validateFlags(fields = []) {
  if (!Array.isArray(fields) || fields.length > 30) throw new Error('Use at most 30 flag label fields.');
  const ids = new Set(), names = new Set();
  return fields.map(field => {
    if (!field || typeof field.id !== 'string' || !field.id || ids.has(field.id) || field.id.length > 120) throw new Error('Invalid flag field ID.');
    ids.add(field.id);
    if (typeof field.name !== 'string' || !field.name.trim() || field.name.length > 120 || names.has(field.name.trim())) throw new Error('Flag field names must be non-empty and unique.');
    names.add(field.name.trim());
    if (!Array.isArray(field.categories) || !field.categories.length || field.categories.length > 100) throw new Error('Use 1–100 categories per flag field.');
    const labels = new Set();
    const categories = field.categories.map(item => {
      if (!item || typeof item.label !== 'string' || !item.label.trim() || item.label.length > 256 || labels.has(item.label) || !/^#[0-9a-f]{6}$/i.test(item.colour)) throw new Error('Flag categories need unique names and valid colours.');
      labels.add(item.label); return { label: item.label, colour: item.colour };
    });
    if (!Array.isArray(field.assignments) || field.assignments.length > 5_000_000) throw new Error('Invalid flag assignments.');
    const cells = new Set();
    const assignments = field.assignments.map(entry => {
      if (!Array.isArray(entry) || entry.length !== 3) throw new Error('Invalid flag assignment.');
      const [roi, id, label] = entry, key = cellKey(roi, id);
      if (typeof roi !== 'string' || !roi || roi.length > 16384 || !Number.isSafeInteger(id) || id < 1 || id > 0xffffffff || !labels.has(label) || cells.has(key)) throw new Error('Invalid or duplicate cell identity in flags.');
      cells.add(key); return [roi, id, label];
    });
    return { id: field.id, name: field.name.trim(), categories, assignments };
  });
}
export function applyFlags(region, field, roi) {
  const lookup = new Map(field.assignments.filter(entry => entry[0] === roi).map(([, id, label]) => [id, label]));
  return { ...region, values: region.ids.map(id => lookup.get(id) ?? null), categories: [null, ...field.categories.map(item => item.label)], palette: ['#89949e', ...field.categories.map(item => item.colour)] };
}
export function assignFlags(field, roi, ids, label) {
  if (label !== null && !field.categories.some(item => item.label === label)) throw new Error('Choose a flag category.');
  const selected = new Set(ids), previous = field.assignments.filter(entry => entry[0] === roi && selected.has(entry[1]));
  field.assignments = field.assignments.filter(entry => entry[0] !== roi || !selected.has(entry[1]));
  if (label !== null) for (const id of selected) field.assignments.push([roi, id, label]);
  return () => { field.assignments = field.assignments.filter(entry => entry[0] !== roi || !selected.has(entry[1])).concat(previous); };
}
export function flagsCSV(field, mapping) {
  const quote = value => '"' + String(value ?? '').replaceAll('"', '""') + '"';
  const colours = new Map(field.categories.map(item => [item.label, item.colour]));
  return [['roi_column', 'roi', 'object_id_column', 'object_id', 'label_field', 'label', 'colour'], ...field.assignments.map(([roi, id, label]) => [mapping.roi, roi, mapping.id, id, field.name, label, colours.get(label)])].map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
}
export function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [a, b] = polygon[i], [c, d] = polygon[j];
    if (((b > y) !== (d > y)) && x < (c - a) * (y - b) / (d - b) + a) inside = !inside;
  }
  return inside;
}
export function cellCentres(mask, width, known) {
  const sums = new Map();
  for (let i = 0; i < mask.length; i++) {
    const id = mask[i]; if (!id || !known.has(id)) continue;
    const item = sums.get(id) ?? [0, 0, 0]; item[0] += i % width + 0.5; item[1] += Math.floor(i / width) + 0.5; item[2]++; sums.set(id, item);
  }
  return new Map([...sums].map(([id, [x, y, n]]) => [id, [x / n, y / n]]));
}
