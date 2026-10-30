// Format interpretation adapted from readimc (MIT, University of Zurich).
// See THIRD-PARTY-NOTICES.txt. Files are sliced, never read in full.
const utf16 = new TextDecoder('utf-16le');
const START = '<MCDSchema', END = '</MCDSchema>';
const signature = text => Uint8Array.from([...text].flatMap(c => [c.charCodeAt(0), 0]));
function lastIndex(bytes, needle, before = bytes.length) {
  outer: for (let i = Math.min(before - needle.length, bytes.length - needle.length); i >= 0; i--) {
    for (let j = 0; j < needle.length; j++) if (bytes[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

export async function readSchema(file, progress = () => {}, blockSize = 1024 * 1024) {
  const startTag = signature(START), endTag = signature(END);
  let cursor = file.size, end = null;
  while (cursor > 0) {
    const start = Math.max(0, cursor - blockSize);
    const bytes = new Uint8Array(await file.slice(start, Math.min(file.size, cursor + 32)).arrayBuffer());
    if (end === null) {
      const at = lastIndex(bytes, endTag);
      if (at >= 0) end = start + at + endTag.length;
    }
    if (end !== null) {
      const at = lastIndex(bytes, startTag, end - start);
      if (at >= 0) {
        if (end - start - at > 64 * 1024 * 1024) throw new Error('The MCD metadata exceeds the 64 MB limit.');
        return utf16.decode(await file.slice(start + at, end).arrayBuffer());
      }
      if (end - start > 64 * 1024 * 1024) throw new Error('Could not locate the start of the MCD metadata.');
    }
    cursor = start;
    progress({ stage: 'Finding MCD metadata', fraction: (file.size - cursor) / file.size });
  }
  throw new Error('No complete MCD metadata was found. Choose an intact .mcd file.');
}

export function parseSchema(xml, Parser = globalThis.DOMParser) {
  const doc = new Parser().parseFromString(xml, 'application/xml');
  if (doc.querySelector('parsererror') || doc.documentElement.localName !== 'MCDSchema') throw new Error('The MCD metadata contains invalid XML.');
  const elements = name => [...doc.getElementsByTagNameNS('*', name)];
  const fields = el => Object.fromEntries([...el.children].map(child => [child.localName, child.textContent.trim()]));
  const records = name => elements(name).map(fields);
  const slides = records('Slide').map(m => ({ id: m.ID, name: m.Description || m.Filename || `Slide ${m.ID}`, metadata: m }));
  const panoramas = records('Panorama').map(m => ({ id: m.ID, slideId: m.SlideID, name: m.Description || `Panorama ${m.ID}`, metadata: m }));
  const rois = new Map(records('AcquisitionROI').map(m => [m.ID, m]));
  const channels = records('AcquisitionChannel'), points = records('ROIPoint');
  const acquisitions = records('Acquisition').map(m => {
    const roi = rois.get(m.AcquisitionROIID) || {};
    const panorama = panoramas.find(p => p.id === roi.PanoramaID);
    const allChannels = channels.filter(c => c.AcquisitionID === m.ID).sort((a, b) => Number(a.OrderNumber) - Number(b.OrderNumber));
    const xy = ['X', 'Y'].map(name => allChannels.findIndex(c => c.ChannelName === name));
    const markerChannels = allChannels.map((c, index) => ({ index, name: c.ChannelName, label: c.ChannelLabel || c.ChannelName, key: `${c.ChannelName}\u0000${c.ChannelLabel || ''}` })).filter(c => !['X', 'Y', 'Z'].includes(c.name));
    return {
      id: m.ID, name: m.Description || `Acquisition ${m.ID}`, metadata: m,
      panoramaId: panorama?.id, slideId: panorama?.slideId || m.SlideID || slides[0]?.id,
      width: Number(m.MaxX), height: Number(m.MaxY), start: Number(m.DataStartOffset), end: Number(m.DataEndOffset),
      valueBytes: Number(m.ValueBytes), stride: allChannels.length, xy, channels: markerChannels,
      points: points.filter(p => p.AcquisitionROIID === m.AcquisitionROIID).sort((a, b) => Number(a.OrderNumber) - Number(b.OrderNumber)).map(p => [Number(p.SlideXPosUm), Number(p.SlideYPosUm)]),
    };
  }).sort((a, b) => Number(a.id) - Number(b.id));
  if (!acquisitions.length) throw new Error('This file contains no acquisitions to view.');
  if (new Set(acquisitions.map(a => a.id)).size !== acquisitions.length) throw new Error('The MCD contains duplicate acquisition IDs.');
  return { slides, panoramas, acquisitions };
}

export function validateAcquisition(file, acquisition, selected) {
  const { width, height, start, end, valueBytes, stride, xy } = acquisition;
  if (![start, end].every(n => Number.isSafeInteger(n) && n >= 0) || end < start || end > file.size) throw new Error('This acquisition has invalid data offsets.');
  if (valueBytes !== 4) throw new Error(`Unsupported MCD value size: ${valueBytes} bytes (this viewer supports float32).`);
  if (![width, height].every(n => Number.isSafeInteger(n) && n > 0 && n <= 16384) || width * height > 16_777_216) throw new Error('This acquisition is too large or has invalid dimensions (limit: 16 million pixels, 16,384 pixels per side).');
  if (!Number.isInteger(stride) || stride < 3 || xy.some(n => !Number.isInteger(n) || n < 0 || n >= stride)) throw new Error('The acquisition is missing its X/Y coordinate channels.');
  if (!selected.length || selected.some(i => !acquisition.channels.some(c => c.index === i))) throw new Error('Select a valid image channel.');
  if (width * height * (selected.length * 4 + 1) > 256 * 1024 * 1024) throw new Error('This channel selection needs too much memory. Select fewer channels.');
}

export async function readAcquisition(file, a, selected, progress = () => {}, chunkBytes = 4 * 1024 * 1024) {
  selected = [...new Set(selected)];
  validateAcquisition(file, a, selected);
  const recordBytes = a.stride * 4;
  let size = a.end - a.start;
  const warnings = [];
  // Some writer versions record an inclusive end offset.
  if (size % recordBytes === recordBytes - 1 && a.end < file.size) { size++; warnings.push('Read the final byte using an inclusive acquisition end offset.'); }
  if (size % recordBytes) warnings.push(`Ignored ${size % recordBytes} trailing bytes from an incomplete pixel record.`);
  const records = Math.floor(size / recordBytes);
  const planes = selected.map(index => ({ index, data: new Float32Array(a.width * a.height).fill(NaN) }));
  const occupied = new Uint8Array(a.width * a.height);
  let invalid = 0, duplicates = 0, filled = 0, nonfinite = 0;
  const batch = Math.max(1, Math.floor(chunkBytes / recordBytes));
  for (let first = 0; first < records; first += batch) {
    const count = Math.min(batch, records - first);
    const view = new DataView(await file.slice(a.start + first * recordBytes, a.start + (first + count) * recordBytes).arrayBuffer());
    for (let i = 0; i < count; i++) {
      const offset = i * recordBytes;
      const x = view.getFloat32(offset + a.xy[0] * 4, true), y = view.getFloat32(offset + a.xy[1] * 4, true);
      if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0 || x >= a.width || y >= a.height) { invalid++; continue; }
      const target = y * a.width + x;
      if (occupied[target]) duplicates++; else { occupied[target] = 1; filled++; }
      for (const plane of planes) {
        const value = view.getFloat32(offset + plane.index * 4, true);
        plane.data[target] = value;
        if (!Number.isFinite(value)) nonfinite++;
      }
    }
    progress({ stage: 'Reading acquisition', fraction: (first + count) / records });
  }
  if (invalid) warnings.push(`${invalid.toLocaleString()} pixels have invalid coordinates and were skipped.`);
  if (duplicates) warnings.push(`${duplicates.toLocaleString()} repeated pixel coordinates; the latest value is displayed.`);
  if (nonfinite) warnings.push(`${nonfinite.toLocaleString()} non-finite channel values are displayed as missing.`);
  if (filled < occupied.length) warnings.push(`${(occupied.length - filled).toLocaleString()} unacquired pixels are shown with a checker pattern.`);
  return { width: a.width, height: a.height, planes, occupied, filled, warnings };
}

export function imageRange(metadata, prefix = 'Image') {
  const start = Number(metadata[`${prefix}StartOffset`]), end = Number(metadata[`${prefix}EndOffset`]);
  return Number.isSafeInteger(start) && Number.isSafeInteger(end) && start >= 0 && end > start + 161 ? { start: start + 161, end: end - 1 } : null;
}

export async function readOpticalImage(file, range) {
  if (!range || range.start < 0 || range.end > file.size || range.end - range.start > 64 * 1024 * 1024) throw new Error('The embedded optical image has invalid offsets or exceeds 64 MB.');
  const bytes = new Uint8Array(await file.slice(range.start, range.end).arrayBuffer());
  // Inspect a small header area: image encodings differ between instrument versions.
  for (let i = 0; i < Math.min(512, bytes.length - 8); i++) {
    let type;
    if (bytes[i] === 0xff && bytes[i + 1] === 0xd8 && bytes[i + 2] === 0xff) type = 'image/jpeg';
    else if (bytes[i] === 137 && bytes[i + 1] === 80 && bytes[i + 2] === 78 && bytes[i + 3] === 71) type = 'image/png';
    else if (bytes[i] === 66 && bytes[i + 1] === 77) type = 'image/bmp';
    if (type) return new Blob([bytes.subarray(i)], { type });
  }
  throw new Error('This optical image uses an encoding the browser viewer does not support. Channel images are still available.');
}

export function contrast(data) {
  const values = [], step = Math.max(1, Math.floor(data.length / 50000));
  for (let i = 0; i < data.length; i += step) if (Number.isFinite(data[i])) values.push(data[i]);
  values.sort((a, b) => a - b);
  const low = Math.min(0, values[0] ?? 0), max = values.at(-1) ?? 1;
  const high = values[Math.floor((values.length - 1) * 0.995)] ?? 1;
  return { low, high: high > low ? high : Math.max(max, low + 1), max: Math.max(max, low + 1) };
}
