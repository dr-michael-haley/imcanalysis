// Object IDs are only unique within an ROI. Keep both parts in every selection.
export class CellSelection {
  constructor() { this.rois = new Map(); }
  get size() { let n = 0; for (const ids of this.rois.values()) n += ids.size; return n; }
  forROI(roi) { if (!this.rois.has(roi)) this.rois.set(roi, new Set()); return this.rois.get(roi); }
  has(roi, id) { return this.rois.get(roi)?.has(id) ?? false; }
  add(roi, id) { if (typeof roi !== 'string' || !roi || !Number.isInteger(id) || id < 1 || id > 0xffffffff) throw new Error('Invalid selected cell identity.'); this.forROI(roi).add(id); }
  toggle(roi, id) { if (this.has(roi, id)) this.forROI(roi).delete(id); else this.add(roi, id); }
  clear() { this.rois.clear(); }
}
export function assignSelection(field, selection, label) {
  if (label !== null && !field.categories.some(c => c.label === label)) throw new Error('Choose an available flag category.');
  const before = field.assignments.filter(([roi, id]) => selection.has(roi, id));
  const selected = new Map([...selection.rois].map(([roi, ids]) => [roi, new Set(ids)]));
  const matches = ([roi, id]) => selected.get(roi)?.has(id);
  const next = field.assignments.filter(row => !matches(row));
  if (label !== null) for (const [roi, ids] of selected) for (const id of ids) next.push([roi, id, label]);
  if (next.length > 5000000) throw new Error('Too many flag assignments.');
  field.assignments = next;
  return () => { field.assignments = field.assignments.filter(row => !matches(row)).concat(before); };
}
