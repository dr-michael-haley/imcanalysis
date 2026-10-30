import test from 'node:test';
import assert from 'node:assert/strict';
import { channelAliases, channelName, discoverFiles, validateMask, colourScale, isBoundary } from '../src/core.js';
import { parseNormalization, findNormalization, normalizeImage } from '../src/normalization.js';
import { readFileSync } from 'node:fs';
import { colourRange, validateColourLimits } from '../src/colour-limits.js';

test('continuous colour limits clip colours, preserve zero and missing values, and keep source data intact', () => {
  const values = [-10, 0, 5, 10, 100, null, NaN], before = [...values];
  const scale = colourScale(values, null, [], 'grey', false, { vmin: 0, vmax: 10 });
  assert.deepEqual(scale.range, [0, 10]);
  assert.deepEqual(scale.colour(-10), [0, 0, 0]); assert.deepEqual(scale.colour(0), [0, 0, 0]);
  assert.deepEqual(scale.colour(5), [128, 128, 128]); assert.deepEqual(scale.colour(100), [255, 255, 255]);
  assert.deepEqual(scale.colour(null), [137, 148, 158]); assert.deepEqual(scale.colour(NaN), [137, 148, 158]);
  assert.deepEqual(values, before);
  assert.deepEqual(colourScale(values, ['T'], ['#ff0000'], 'grey', false, { vmin: 0, vmax: 10 }).colour('T'), [255, 0, 0]);
  assert.deepEqual(colourScale(values, null, [], 'grey', true, { vmin: 0, vmax: 10 }).colour(0), [255, 255, 255]);
});

test('automatic bounds survive constant and out-of-range ROIs, while invalid manual limits fail', () => {
  assert.deepEqual(validateColourLimits({}), { vmin: null, vmax: null });
  assert.deepEqual(colourRange([-5, 10], { vmin: 0 }), [0, 10]);
  assert.deepEqual(colourRange([-5, 10], { vmax: 0 }), [-5, 0]);
  assert.deepEqual(colourRange([0, 10], { vmin: 20 }), [20, 21]);
  assert.deepEqual(colourRange([0, 10], { vmax: -20 }), [-21, -20]);
  assert.deepEqual(colourScale([0, 0], null).range, [0, 1]);
  for (const limits of [{ vmin: 1, vmax: 1 }, { vmin: 2, vmax: 1 }, { vmin: NaN }, { vmax: Infinity }, { vmin: '' }]) assert.throws(() => validateColourLimits(limits), /colour limit/i);
});

const header = { markers: ['CD3', 'CD20', 'HLA-DR'], channelNames: ['191Ir', '209Bi', '143Nd'], channelLabels: ['CD3e', 'CD20', 'HLA_DR'] };
const file = path => ({ name: path.split('/').at(-1), webkitRelativePath: path });

test('SBT isotope and sample-prefixed channel names bind to panel aliases', () => {
  const aliases = channelAliases(header);
  assert.equal(channelName(file('images/ROI_A/sample_001_191Ir_CD3e.tiff'), 'ROI_A', aliases), 'CD3');
  assert.equal(channelName(file('images/ROI_A/Nd143_HLA_DR.tiff'), 'ROI_A', aliases), 'HLA-DR');
  const ambiguous = channelAliases({ markers: ['a', 'b'], channelNames: [], channelLabels: ['shared', 'shared'] });
  assert.equal(ambiguous.has('shared'), false);
});
test('nested and flat images preserve overlapping ROI names and reject ambiguous masks', () => {
  const masks = [file('masks/ROI_A.tiff'), file('masks/ROI_A_2.tif')];
  const images = [file('images/ROI_A/CD3.tiff'), file('images/ROI_A_2_CD20.tiff')];
  const index = discoverFiles(masks, images, ['ROI_A', 'ROI_A_2'], channelAliases(header));
  assert.equal(index.get('ROI_A').images[0].name, 'CD3');
  assert.equal(index.get('ROI_A_2').images[0].name, 'CD20');
  assert.throws(() => discoverFiles([...masks, file('other/ROI_A.tif')], images, ['ROI_A'], channelAliases(header)), /Multiple masks/);
});
test('mask identities retain uint32 labels and report cohort gaps', () => {
  assert.deepEqual(validateMask(new Uint32Array([0, 300, 65537]), [300, 600]), { missing: 1, context: 1 });
  assert.throws(() => validateMask([1.5], [1]), /integer/);
  assert.throws(() => validateMask([-1], [1]), /integer/);
});
test('missing categories stay neutral and zero remains a real numeric value', () => {
  const scale = colourScale(['T', null], ['T'], ['#ff0000']);
  assert.deepEqual(scale.colour('T'), [255, 0, 0]);
  assert.deepEqual(scale.colour(null), [137, 148, 158]);
  assert.notDeepEqual(colourScale([0, 1], null).colour(0), [137, 148, 158]);
});
test('adjacent objects retain a boundary even with identical population colours', () => {
  const mask = new Uint32Array([1, 1, 2, 1, 1, 2, 1, 1, 2]);
  assert.equal(isBoundary(mask, 4, 3, 3), true);
  assert.equal(isBoundary(new Uint32Array(9).fill(1), 4, 3, 3), false);
});

test('normalisation accepts current Nimbus and legacy NapariSBT formats', () => {
  const expected = { vmax: 10, lower_threshold: 0 };
  for (const [filename, text] of [['old.json', '{"CD3":"10"}'], ['workspace.json', '{"normalization_dict":{"CD3":{"vmax":10,"lower_threshold":0}}}'], ['old.csv', '\uFEFFMarker,Value\r\nCD3,10\r\n'], ['scan.csv', 'Channel,Baseline Vmax,Lower Bound\nCD3,10,0']]) {
    assert.deepEqual(parseNormalization(text, filename).get('CD3'), expected);
  }
  assert.deepEqual(parseNormalization('marker,vmax,lower_threshold\n"CD3, clone ""A""",12,2\n', 'n.csv').get('CD3, clone "A"'), { vmax: 12, lower_threshold: 2 });
});

test('invalid or duplicate normalisation rows fail explicitly', () => {
  for (const text of ['marker,vmax\nCD3,10\ncd3,20', 'marker,vmax,lower_threshold\nCD3,10,10', 'marker,vmax\nCD3,NaN', 'marker,vmax\nCD3,0', 'marker,vmax\nCD3,', 'marker,vmax,vmax\nCD3,10,20', 'marker,vmax\n"CD3,10']) assert.throws(() => parseNormalization(text, 'n.csv'));
  for (const text of ['{}', '[]', '{"CD3":null}', '{"CD3":true}', '{"CD3":10,"cd3":20}', '{"CD3":{"vmax":10,"lower":-1}}']) assert.throws(() => parseNormalization(text, 'n.json'));
});

test('normalisation matching uses channel names, filenames and unambiguous panel aliases', () => {
  const mapping = parseNormalization('{"CD3e":10,"HLA-DR":20,"209Bi_CD20":30}', 'n.json');
  assert.equal(findNormalization(mapping, ['CD3', 'sample_001_191Ir_CD3e'], channelAliases(header)).vmax, 10);
  assert.equal(findNormalization(mapping, ['CD20', '209Bi_CD20']).vmax, 30);
  assert.equal(findNormalization(mapping, ['hladr']).vmax, 20);
  assert.equal(findNormalization(mapping, ['missing']), null);
  const ambiguous = parseNormalization('{"HLA-DR":10,"HLA.DR":20}', 'n.json');
  assert.equal(findNormalization(ambiguous, ['HLA-DR']).vmax, 10);
  assert.throws(() => findNormalization(ambiguous, ['HLADR']), /Ambiguous/);
});

test('image normalisation agrees with Python NapariSBT reference values', () => {
  const reference = JSON.parse(readFileSync(new URL('../fixtures/normalization_reference.json', import.meta.url), 'utf8'));
  for (const { values, options, expected } of reference) {
    const source = Float32Array.from(values), before = source.slice();
    const actual = normalizeImage(source, options).data;
    for (let i = 0; i < expected.length; i++) assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6, `${actual[i]} != ${expected[i]}`);
    assert.deepEqual(source, before, 'Normalisation must not mutate decoded source values.');
  }
});
