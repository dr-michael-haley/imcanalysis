import test from 'node:test';
import assert from 'node:assert/strict';
import { slideImageSources } from '../src/optical-sources.js';
import { panoramaProjection } from '../src/overview.js';

const image = { ImageStartOffset: '100', ImageEndOffset: '1000' };
function panorama(id, slideId, type, width, name = `Image ${id}`) {
  return { id, slideId, name, metadata: { ...image, Type: type, SlideX1PosUm: '0', SlideY1PosUm: '25000', SlideX2PosUm: String(width), SlideY2PosUm: '25000', SlideX4PosUm: '0', SlideY4PosUm: '0' } };
}

test('finds an imported JPG-named panorama stored as PNG without any linked acquisitions', () => {
  const photo = {
    id: '1', slideId: '0', name: 'PXL_example.jpg',
    metadata: { Type: 'Imported', ImageFormat: 'PNG', ImageStartOffset: '27007', ImageEndOffset: '3839973', SlideX1PosUm: '330.578', SlideY1PosUm: '25000', SlideX2PosUm: '74669.578', SlideY2PosUm: '25000', SlideX4PosUm: '330.578', SlideY4PosUm: '0' },
  };
  const model = { slides: [{ id: '0', metadata: image }], panoramas: [photo, panorama('2', '0', 'Instrument', 6706)], acquisitions: [] };
  const sources = slideImageSources(model, '0');
  assert.deepEqual(sources.map(s => s.id), ['slide', 'slide-overview']);
  assert.equal(sources[0].name, 'PXL_example.jpg');
  assert.deepEqual(sources[0].range, { start: 27168, end: 3839972 });
  assert.equal(sources[0].projection, 'panorama');
  const project = panoramaProjection(sources[0].metadata, 2974, 1000);
  assert.deepEqual(project([330.578, 25000]).map(v => v || 0), [0, 0]);
  assert.deepEqual(project([74669.578, 0]).map(v => v || 0), [2974, 1000]);
});

test('retains all imported photos, prefers broad slide coverage, and isolates slides', () => {
  const model = { slides: [], panoramas: [panorama('2', '0', 'Imported', 3000), panorama('3', '1', 'Imported', 100000), panorama('4', '0', 'Imported', 75000), panorama('5', '0', 'Instrument', 90000)] };
  const sources = slideImageSources(model, '0');
  assert.deepEqual(sources.map(s => [s.id, s.name]), [['slide', 'Image 4'], ['slide-photo:2', 'Image 2']]);
  assert.deepEqual(slideImageSources(model, 'missing'), []);
});

test('preserves the original slide-image path when no valid imported photograph exists', () => {
  const missing = panorama('1', '0', 'Imported', 75000); missing.metadata.ImageEndOffset = '0';
  const model = { slides: [{ id: '0', metadata: image }], panoramas: [missing] };
  const sources = slideImageSources(model, '0');
  assert.equal(sources.length, 1); assert.equal(sources[0].id, 'slide'); assert.equal(sources[0].projection, 'slide');
});
