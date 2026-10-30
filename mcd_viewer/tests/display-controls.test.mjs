import test from 'node:test';
import assert from 'node:assert/strict';
import { orderedChannels, adjacentChannel, CORE_COLOURS } from '../src/channel-controls.js';
import { autoBounds, moveBound, countStep } from '../src/contrast-control.js';
import { histograms } from '../src/histogram.js';
import { zoomBox } from '../src/overview-navigation.js';
import { colourTable } from '../src/colormaps.js';

test('alphabetical browsing preserves binary indices and original metal order remains available', () => {
  const channels = [{ label: 'DNA', index: 3 }, { label: 'CD20', index: 4 }, { label: 'CD3', index: 5 }];
  const sorted = orderedChannels(channels);
  assert.deepEqual(sorted.map(c => c.label), ['CD3', 'CD20', 'DNA']);
  assert.equal(adjacentChannel(sorted, 5, -1).index, 3);
  assert.equal(adjacentChannel(sorted, 3, 1).index, 5);
  assert.deepEqual(orderedChannels(channels, 'original'), channels);
  assert.deepEqual(channels.map(c => c.index), [3, 4, 5]);
});
test('contrast bounds move by tenths, stay separate, and auto bounds encompass small signals', () => {
  assert.deepEqual(autoBounds({ low: -.02, high: .04 }), { low: -.1, high: .1 });
  assert.deepEqual(autoBounds({ low: 0, high: 0 }), { low: 0, high: .1 });
  assert.equal(moveBound(0, 1, 'low', 2), .9);
  assert.equal(moveBound(0, 1, 'high', -1), .1);
  assert.equal(moveBound(0, 1, 'low', .16), .2);
  assert.equal(countStep(.2 + .1), .3);
});
test('histograms count measured zeros, ignore missing values, share bins, and include the upper edge', () => {
  const result = histograms([
    { index: 9, data: Float32Array.from([0, 0, 1, 2, 3, 4, NaN]) },
    { index: 4, data: Float32Array.from([0, 2, 4, Infinity, -Infinity]) },
  ], 4);
  assert.equal(result.low, 0); assert.equal(result.high, 4);
  assert.deepEqual([...result.curves[0].counts], [2, 1, 1, 2]);
  assert.deepEqual([...result.curves[1].counts], [1, 0, 1, 1]);
  assert.equal(result.curves[0].samples, 6); assert.equal(result.curves[0].missing, 1);
  assert.equal(result.curves[1].samples, 3); assert.equal(result.curves[1].missing, 2);
});
test('histograms handle empty, constant, negative, and small fractional signals', () => {
  assert.deepEqual(histograms([{ index: 1, data: [NaN] }]).curves, []);
  for (const data of [[0, 0], [-2, -2], [7, 7], [-.2, 0, .1]]) {
    const result = histograms([{ index: 1, data }]);
    assert.ok(result.high > result.low);
    assert.equal(result.curves[0].counts.reduce((a, b) => a + b, 0), data.length);
  }
});
test('overview zoom stays anchored under the cursor and respects zoom bounds', () => {
  const fit = { x: 0, y: 0, width: 100, height: 50 }, point = { x: 25, y: 10 };
  const zoomed = zoomBox(fit, 2, point, fit);
  assert.deepEqual(zoomed, { x: 12.5, y: 5, width: 50, height: 25 });
  assert.deepEqual(zoomBox(zoomed, .5, point, fit), fit);
  assert.equal(zoomBox(fit, 1000, point, fit).width, 100 / 64);
  assert.equal(zoomBox(fit, .0001, point, fit).width, 200);
});
test('pure colour ramps retain black/primary endpoints and scientific maps use their full colour ranges', () => {
  for (const { hex } of CORE_COLOURS) {
    const table = colourTable({ colour: hex });
    assert.deepEqual(table[0], [0, 0, 0]);
    assert.deepEqual(table[255], [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)));
  }
  assert.deepEqual(colourTable({ map: 'viridis' })[0], [68, 1, 84]);
  assert.deepEqual(colourTable({ map: 'viridis' })[255], [253, 231, 37]);
  assert.deepEqual(colourTable({ map: 'plasma' })[255], [240, 249, 33]);
});
