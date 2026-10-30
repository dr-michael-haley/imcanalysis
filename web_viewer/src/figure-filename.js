export const defaultFigureFilename = () => ({ label: 'figure', roi: true, views: false, timestamp: true, metadata: [] });

export function validateFigureFilename(value) {
  if (value != null && (typeof value !== 'object' || Array.isArray(value))) throw new Error('Invalid figure filename settings.');
  const s = { ...defaultFigureFilename(), ...value };
  if (typeof s.label !== 'string' || s.label.length > 100 || !['roi', 'views', 'timestamp'].every(key => typeof s[key] === 'boolean') || !Array.isArray(s.metadata) || s.metadata.length > 20 || s.metadata.some(key => typeof key !== 'string' || !key || key.length > 16384) || new Set(s.metadata).size !== s.metadata.length) throw new Error('Invalid figure filename settings.');
  return { label: s.label, roi: s.roi, views: s.views, timestamp: s.timestamp, metadata: [...s.metadata] };
}

// Keep Unicode labels, but remove path separators, control characters and Windows
// reserved filenames. The final bound leaves room for the timestamp and extension.
export function filenamePart(value) {
  let part = String(value).normalize('NFC').replace(/[\x00-\x1f\x7f<>:"/\\|?*]/g, '_').replace(/\s+/g, '_').replace(/_+/g, '_').replace(/^[. _]+|[. _]+$/g, '');
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part)) part = `_${part}`;
  return part;
}

export function figureFilename(settings, context, date = new Date()) {
  const s = validateFigureFilename(settings.filename), parts = [];
  if (s.label.trim()) parts.push(s.label);
  if (s.roi && context.roi) parts.push(context.roi);
  if (s.metadata.length && !context.metadataLoaded) throw new Error('Read ROI metadata to use the saved filename fields, or deselect those fields.');
  for (const key of s.metadata) {
    const field = context.metadata.find(item => item.name === key), pair = field?.values.find(([roi]) => roi === context.roi);
    if (!pair) throw new Error(`Filename metadata “${key}” is unavailable for this ROI. Read metadata or deselect that field.`);
    parts.push(`${key}-${pair[1] ?? 'Missing'}`);
  }
  if (s.views) parts.push(...context.viewNames);
  let base = parts.map(filenamePart).filter(Boolean).join('-') || 'figure';
  let bounded = '';
  for (const character of base) { if (bounded.length + character.length > 160) break; bounded += character; }
  base = bounded.replace(/[. _-]+$/g, '') || 'figure';
  // Apply reserved-name protection after truncation as well.
  base = filenamePart(base);
  const stamp = s.timestamp ? `-${date.toISOString().replace(/[:.]/g, '-')}` : '';
  return `${base}${stamp}.${settings.format === 'jpeg' ? 'jpg' : settings.format}`;
}
