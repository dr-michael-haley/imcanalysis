import test from 'node:test';
import assert from 'node:assert/strict';
import { createLoadingState } from '../src/loading.js';
import { composeView, composeViewAsync } from '../src/views.js';
import { defaultLayer } from '../src/aligned-layers.js';

test('superseded ROI success or failure cannot hide or replace current progress', () => {
  const events = [], loading = createLoadingState(event => events.push(event));
  const first = loading.begin('ROI A', 'Mask');
  const next = loading.begin('ROI B', 'Mask');
  first.update('Old channels'); first.finish();
  assert.deepEqual(events.at(-1), { title: 'ROI B', detail: 'Mask' });
  next.update('Rendering view 4 of 4'); next.finish();
  assert.equal(events.at(-1), null);
  next.update('Late callback'); next.finish();
  assert.equal(events.length, 4);
});

function fixture(width = 3, height = 1) {
  const commits = [];
  const context = { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData: image => commits.push(image.data) };
  const scene = { width: 9, height: 9, getContext: () => context };
  const view = { scene, region: { ids: [1], values: ['A'] }, colours: { colour: () => [0, 255, 0] },
    imagesVisible: true, overlayVisible: true, overlay: { population: '', style: 'fill', opacity: .5 },
    channels: [{ active: true, enabled: true, low: 0, high: 1, colour: '#ff0000', raster: { data: new Float32Array(width * height).fill(1) } }],
  };
  return { view, commits, dimensions: { width, height, selected: null, mask: { data: new Uint32Array(width * height).fill(1) } } };
}

test('cooperative rendering preserves channel and overlay pixels', async () => {
  const sync = fixture(), async = fixture();
  composeView(sync.view, sync.dimensions);
  assert.equal(await composeViewAsync(async.view, async.dimensions, () => true), true);
  assert.deepEqual([...async.commits[0]], [128, 128, 0, 255, 128, 128, 0, 255, 128, 128, 0, 255]);
  assert.deepEqual(async.commits, sync.commits);
  assert.equal(async.view.scene.width, 3);
});

test('cancelling midway through a large render leaves the old frame intact', async () => {
  const { view, dimensions, commits } = fixture(1024, 1024);
  let checks = 0;
  assert.equal(await composeViewAsync(view, dimensions, () => ++checks < 5), false);
  assert.equal(commits.length, 0);
  assert.equal(view.scene.width, 9);
});

test('aligned layers obey source order and cell overlays remain on top', () => {
  const { view, dimensions, commits } = fixture(1, 1);
  const rgb = { ...defaultLayer('rgb', 'rgb'), raster: { data: new Uint8Array([0, 0, 255, 255]) } };
  const labels = { ...defaultLayer('labels', 'labels'), opacity: 1, raster: { data: new Uint16Array([1]) } };
  view.layers = [rgb, labels];
  dimensions.layers = [{ id: 'rgb', kind: 'rgb' }, { id: 'labels', kind: 'labels', mapping: [{ value: 1, color: '#ff0000' }] }];
  composeView(view, dimensions); assert.deepEqual([...commits.at(-1)], [128, 128, 0, 255]);
  dimensions.layers.reverse(); composeView(view, dimensions); assert.deepEqual([...commits.at(-1)], [0, 128, 128, 255]);
  rgb.visible = false; labels.visible = false; composeView(view, dimensions); assert.deepEqual([...commits.at(-1)], [128, 128, 0, 255]);
});

test('large rendering yields to other tasks before committing', async () => {
  const { view, dimensions, commits } = fixture(2048, 2048);
  let yields = 0;
  await composeViewAsync(view, dimensions, () => true, async () => {
    yields++; assert.equal(commits.length, 0);
    await new Promise(resolve => setTimeout(resolve, 0));
  });
  assert.ok(yields > 1, 'large renders yield more than once');
  assert.equal(commits.length, 1);
});
