import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultFigure, defaultScaleBar, validateFigure, validateScaleBar, figureLayout, scaleBarGeometry, paneFurniture } from '../src/figure.js';
import { exportGeometry, figureSVG } from '../src/image-export.js';
import { figureFilename, filenamePart } from '../src/figure-filename.js';

test('IMC scale bars remain calibrated at different zooms and export resolutions', () => {
  const bar = { ...defaultScaleBar(), automatic: false, length: 50 };
  for (const outputWidth of [512, 1024, 2048]) {
    const g = exportGeometry(outputWidth, { width: 900, height: 600 }, { x: -20, y: -40, scale: 3 });
    const measure = scaleBarGeometry(bar, 1, g.zoom, g.width);
    assert.equal(measure.length, 50);
    assert.ok(Math.abs(measure.pixels - outputWidth / 6) < 1e-9);
    assert.ok(Math.abs(g.width / g.zoom - 300) < 1e-9);
    assert.ok(Math.abs(g.height / g.zoom - 200) <= .5 / g.zoom);
  }
  assert.equal(scaleBarGeometry({ ...bar, length: 1000 }, 1, 3, 900).fits, false);
});
test('physical frame overrides aspect and the preview matches export proportions', () => {
  const s = { ...defaultFigure(), physicalLock: true, physicalWidth: 300, physicalHeight: 200, aspect: '1:1' };
  const l = figureLayout(1, 1200, 1000, s);
  assert.equal(l.width / l.height, 1.5);
  assert.equal(l.outputWidth, 1024);
  assert.equal(l.outputHeight, 1024 / 1.5);
  assert.throws(() => validateFigure({ paneWidth: 100 }), /paneWidth/);
  assert.throws(() => validateScaleBar({ length: -1 }), /scale bar/);
});
test('export dimensions reject excessive allocations and retain camera translation', () => {
  assert.deepEqual(exportGeometry(1024, { width: 512, height: 256 }, { x: -10, y: 20, scale: 2 }), { width: 1024, height: 512, x: -20, y: 40, zoom: 4 });
  assert.throws(() => exportGeometry(8192, { width: 1, height: 10 }, { scale: 1 }), /too large/);
  assert.throws(() => exportGeometry(8192, { width: 1, height: 1 }, { scale: 1 }), /too large/);
  assert.throws(() => exportGeometry(1024, { width: 0, height: 10 }, { scale: 1 }), /invalid/);
});
test('SVG layers preserve editable labels and scale bars while escaping metadata', () => {
  const settings = { ...defaultFigure(), mode: 'figure' };
  const view = { name: 'A < B & C', legendVisible: true, scaleBar: { ...defaultScaleBar(), automatic: false, length: 50 } };
  const furniture = paneFurniture(view, [{ title: 'Channels', rows: [{ name: 'CD3 & CD8', colour: '#ff0000' }] }], settings, 1024, 683, 1024 / 300, 'export');
  const svg = figureSVG('data:image/png;base64,AA==', furniture, { width: 1024, height: 683 }, settings, { title: view.name, umPerPixel: 1 });
  assert.match(svg, /inkscape:label="Image"/); assert.match(svg, /inkscape:label="scale bar"/);
  assert.match(svg, /inkscape:label="legend"/); assert.match(svg, /50 µm/);
  assert.match(svg, /A &lt; B &amp; C/); assert.match(svg, /CD3 &amp; CD8/);
  assert.match(svg, /h170\.666666666666/);
});
test('optical images omit the IMC scale bar', () => {
  const text = paneFurniture({ name: 'Slide photo', legendVisible: false, scaleBar: { ...defaultScaleBar(), visible: false } }, [], { ...defaultFigure(), mode: 'figure' }, 1024, 400, 1, 'export');
  assert.doesNotMatch(text, /µm|scale-bar/);
});
test('filename options include selected acquisition metadata and protect local filenames', () => {
  const s = { ...defaultFigure(), format: 'jpeg', filename: { label: 'Figure', roi: true, views: true, timestamp: false, metadata: ['ID'] } };
  const context = { roi: 'ROI/1', viewNames: ['DNA'], metadataLoaded: true, metadata: [{ name: 'ID', values: [['ROI/1', '7']] }] };
  assert.equal(figureFilename(s, context), 'Figure-ROI_1-ID-7-DNA.jpg');
  assert.equal(filenamePart('../CON'), '_CON');
  assert.throws(() => figureFilename(s, { ...context, metadata: [] }), /unavailable/);
});
