import test from 'node:test';
import assert from 'node:assert/strict';
import { slideProjection, panoramaProjection, coordinateOverview } from '../src/overview.js';

test('places whole-slide outlines using physical dimensions and a bottom-left origin', () => {
  const project = slideProjection({ WidthUm: '75000', HeightUm: '25000' }, 900, 300);
  assert.deepEqual(project([0, 0]), [0, 300]);
  assert.deepEqual(project([75000, 25000]), [900, 0]);
  assert.deepEqual(project([25000, 12500]), [300, 150]);
  for (const metadata of [{}, { WidthUm: '', HeightUm: 25000 }, { WidthUm: 0, HeightUm: 25000 }, { WidthUm: 75000, HeightUm: 'bad' }]) assert.equal(slideProjection(metadata, 900, 300), null);
});

test('keeps rotated panorama corners registered to their corresponding image corners', () => {
  const metadata = { SlideX1PosUm: 100, SlideY1PosUm: 200, SlideX2PosUm: 100, SlideY2PosUm: 300, SlideX4PosUm: 150, SlideY4PosUm: 200 };
  const project = panoramaProjection(metadata, 1000, 500);
  assert.deepEqual(project([100, 200]).map(v => v || 0), [0, 0]);
  assert.deepEqual(project([100, 300]).map(v => v || 0), [1000, 0]);
  assert.deepEqual(project([150, 200]).map(v => v || 0), [0, 500]);
  assert.equal(panoramaProjection({}, 1000, 500), null);
  assert.equal(panoramaProjection({ ...metadata, SlideY2PosUm: 200 }, 1000, 500), null);
});

test('coordinate overview keeps the same upward slide Y direction and handles missing outlines', () => {
  const bounds = coordinateOverview([{ points: [[100, 200], [300, 200], [300, 600], [100, 600]] }]);
  assert.equal(bounds.width, 200); assert.equal(bounds.height, 400);
  assert.deepEqual(bounds.project([100, 600]), [0, 0]);
  assert.deepEqual(bounds.project([300, 200]), [200, 400]);
  assert.equal(coordinateOverview([]).project, null);
});
