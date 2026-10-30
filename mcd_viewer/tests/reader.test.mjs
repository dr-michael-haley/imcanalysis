import test from 'node:test';
import assert from 'node:assert/strict';
import { readSchema, readAcquisition, readOpticalImage, imageRange, contrast } from '../src/mcd.js';
import { createDemoFile } from '../src/demo.js';

function utf16(text) { const bytes = new Uint8Array(text.length * 2); for (let i = 0; i < text.length; i++) { bytes[i * 2] = text.charCodeAt(i) & 255; bytes[i * 2 + 1] = text.charCodeAt(i) >> 8; } return bytes; }
function fixture(records, { width = 2, height = 2, padding = 13, trailing = 0, inclusive = false } = {}) {
  const data = new Float32Array(records.flat());
  const file = new Blob([new Uint8Array(padding), data, new Uint8Array(trailing), new Uint8Array(64)]);
  return { file, acquisition: { width, height, start: padding, end: padding + data.byteLength + trailing - Number(inclusive), valueBytes: 4, stride: 5, xy: [0, 1], channels: [{ index: 3 }, { index: 4 }] } };
}

test('finds the latest complete UTF-16 metadata across scan boundaries', async () => {
  const old = '<MCDSchema><Slide>old</Slide></MCDSchema>';
  const latest = '<MCDSchema xmlns="urn:test"><Slide>latest μ</Slide></MCDSchema>';
  for (const padding of [0, 1, 12, 37, 99]) {
    const file = new Blob([utf16(old), new Uint8Array(padding), utf16(latest), new Uint8Array(67)]);
    assert.equal(await readSchema(file, () => {}, 61), latest);
  }
});
test('rejects a non-MCD file', async () => { await assert.rejects(readSchema(new Blob(['not an MCD'])), /No complete MCD metadata/); });
test('reconstructs shuffled coordinates and selected channels across small chunks', async () => {
  const { file, acquisition } = fixture([[1, 1, 0, 40, 4], [0, 0, 0, 10, 1], [0, 1, 0, 30, 3], [1, 0, 0, 20, 2]]);
  const result = await readAcquisition(file, acquisition, [4, 3], () => {}, 21);
  assert.deepEqual([...result.planes[0].data], [1, 2, 3, 4]);
  assert.deepEqual([...result.planes[1].data], [10, 20, 30, 40]);
  assert.equal(result.filled, 4); assert.deepEqual(result.warnings, []);
});
test('distinguishes measured zero from unacquired pixels and reports truncated records', async () => {
  const { file, acquisition } = fixture([[0, 0, 0, 0, 8]], { trailing: 3 });
  const result = await readAcquisition(file, acquisition, [3]);
  assert.equal(result.planes[0].data[0], 0); assert.ok(Number.isNaN(result.planes[0].data[1]));
  assert.deepEqual([...result.occupied], [1, 0, 0, 0]);
  assert.match(result.warnings.join(' '), /3 trailing bytes/);
  assert.match(result.warnings.join(' '), /3 unacquired pixels/);
});
test('handles inclusive end offsets without dropping the last pixel', async () => {
  const { file, acquisition } = fixture([[0, 0, 0, 9, 3]], { width: 1, height: 1, inclusive: true });
  const result = await readAcquisition(file, acquisition, [3]);
  assert.equal(result.planes[0].data[0], 9); assert.equal(result.filled, 1);
});
test('reports duplicate and invalid coordinates and retains the latest measured value', async () => {
  const { file, acquisition } = fixture([[0, 0, 0, 1, 1], [0, 0, 0, 2, 2], [-1, 0, 0, 8, 8], [.5, 0, 0, 8, 8]]);
  const result = await readAcquisition(file, acquisition, [3]);
  assert.equal(result.planes[0].data[0], 2); assert.equal(result.filled, 1);
  assert.match(result.warnings.join(' '), /2 pixels have invalid coordinates/);
  assert.match(result.warnings.join(' '), /1 repeated pixel/);
});
test('validates offsets, value formats, dimensions, and selected channels before allocation', async () => {
  const { file, acquisition: a } = fixture([[0, 0, 0, 1, 2]]);
  for (const [changes, pattern] of [[{ end: file.size + 1 }, /offsets/], [{ start: -1 }, /offsets/], [{ valueBytes: 8 }, /value size/], [{ width: 999999 }, /dimensions/], [{ xy: [-1, 1] }, /coordinate/]]) {
    await assert.rejects(readAcquisition(file, { ...a, ...changes }, [3]), pattern);
  }
  await assert.rejects(readAcquisition(file, a, [99]), /valid image channel/);
  const huge = { ...a, width: 4096, height: 4096, channels: [3, 4, 5, 6].map(index => ({ index })), stride: 7 };
  await assert.rejects(readAcquisition(file, huge, [3, 4, 5, 6]), /too much memory/);
});
test('keeps offsets above 4 GiB intact when slicing the file', async () => {
  const start = 2 ** 32 + 13;
  const data = new Float32Array([0, 0, 0, 19, 7]);
  const file = { size: start + data.byteLength, slice: (a, b) => { assert.equal(a, start); assert.equal(b, start + data.byteLength); return new Blob([data]); } };
  const a = { width: 1, height: 1, start, end: file.size, stride: 5, xy: [0, 1], valueBytes: 4, channels: [{ index: 3 }] };
  assert.equal((await readAcquisition(file, a, [3])).planes[0].data[0], 19);
});
test('reads optical image signature after its instrument header', async () => {
  const png = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10, 0]);
  const file = new Blob([new Uint8Array(200), png, new Uint8Array(1)]);
  const range = imageRange({ ImageStartOffset: '39', ImageEndOffset: String(file.size) });
  const blob = await readOpticalImage(file, range);
  assert.equal(blob.type, 'image/png'); assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), png);
  assert.equal(imageRange({ ImageStartOffset: '0', ImageEndOffset: '0' }), null);
  assert.deepEqual(imageRange({ ImageStartOffset: '0', ImageEndOffset: '200' }), { start: 161, end: 199 });
});
test('contrast ignores non-finite values and does not collapse a zero image', () => {
  assert.deepEqual(contrast(Float32Array.from([0, NaN, 0])), { low: 0, high: 1, max: 1 });
});
test('synthetic demo is a readable MCD with real binary image data', async () => {
  const file = createDemoFile(); const xml = await readSchema(file);
  assert.match(xml, /Synthetic demonstration/); assert.match(xml, /<ChannelLabel>DNA<\/ChannelLabel>/);
  const a = { width: 448, height: 352, start: 512, end: 512 + 448 * 352 * 9 * 4, valueBytes: 4, stride: 9, xy: [0, 1], channels: [{ index: 3 }] };
  const result = await readAcquisition(file, a, [3]); assert.equal(result.filled, 448 * 352); assert.ok(result.planes[0].data.some(v => v > 100));
});
