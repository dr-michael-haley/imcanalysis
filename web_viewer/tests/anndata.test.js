import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import h5wasm from 'h5wasm/node';
import { AnnDataReader } from '../src/anndata.js';

await h5wasm.ready;
const open = name => new h5wasm.File(fileURLToPath(new URL(`../fixtures/${name}.h5ad`, import.meta.url)), 'r');

for (const format of ['dense', 'csr', 'csc']) test(`${format}: real AnnData identities, sparse values, layers and embeddings`, () => {
  const file = open(format);
  try {
    const reader = new AnnDataReader(file);
    assert.deepEqual(reader.header().markers, ['CD3', 'CD20', 'DNA']);
    assert.deepEqual(reader.configure('ROI', 'ObjectNumber').map(roi => roi.count), [16, 16]);
    const region = reader.region('ROI_A', 'var:CD20', 'X', 'X_umap');
    assert.deepEqual(region.rows, Array.from({ length: 16 }, (_, i) => i * 2));
    assert.deepEqual(Array.from(region.values), region.rows.map(row => row * 3 + 1));
    assert.equal(region.coordinates.length, 32);
    assert.equal(region.ids[0], 300);
    assert.deepEqual(Array.from(reader.region('ROI_A', 'var:CD3', 'X', '').values), Array(16).fill(0));
    assert.deepEqual(Array.from(reader.region('ROI_A', 'var:CD20', 'layers/counts', '').values), region.rows.map(row => (row * 3 + 1) * 2));
    const categories = reader.region('ROI_A_2', 'obs:population', 'X', '');
    assert.equal(categories.values[1], null);
    assert.deepEqual(categories.palette, ['#ff6666', '#70f0b6', '#779cff']);
    assert.equal(new Map(reader.cell(1)).get('nullable'), null);
    const rank = (mode, field, value = '', source = 'X', direction = 'desc') => reader.rank({ mode, field, value, source, direction });
    assert.deepEqual(rank('population', 'population', JSON.stringify('B cell')).map(item => [item.name, item.score]), [['ROI_A_2', 8], ['ROI_A', 0]]);
    assert.deepEqual(rank('marker', 'CD20').map(item => [item.name, item.score]), [['ROI_A_2', 49], ['ROI_A', 46]]);
    assert.deepEqual(rank('marker', 'CD20', '', 'layers/counts').map(item => item.score), [98, 92]);
    assert.deepEqual(rank('marker', 'CD3').map(item => item.score), [0, 0], 'Implicit sparse zeros participate in means.');
    assert.deepEqual(rank('obs', 'nullable').map(item => item.score), [1, null], 'Missing values sort last, rather than becoming zero.');
    assert.deepEqual(rank('obs', 'nullable', '', 'X', 'asc').map(item => item.score), [1, null]);
    assert.deepEqual(reader.categories('population'), ['T cell', 'B cell', 'Stroma']);
    assert.deepEqual(reader.populationSummary('population').rows, [{ source: 'T cell', count: 8 }, { source: 'B cell', count: 8 }, { source: 'Stroma', count: 8 }, { source: null, count: 8 }]);
    assert.equal(reader.populationSummary('nullable').rows.reduce((sum, row) => sum + row.count, 0), 32);
    assert.ok(reader.populationSummary('nullable').rows.some(row => row.source === null));
    assert.throws(() => rank('obs', 'population'), /No finite numeric/);
    if (format === 'dense') {
      assert.equal(new Map(reader.cell(1)).get('nullable_text'), null);
      assert.deepEqual(reader.region('ROI_A', 'obs:plain_text', 'X', '').categories, reader.region('ROI_A_2', 'obs:plain_text', 'X', '').categories);
      assert.deepEqual(reader.region('ROI_A_2', 'obs:nullable_text', 'X', '').categories, ['ok']);
    }
  } finally { file.close(); }
});
test('duplicate ROI/object identities fail before a view can be constructed', () => {
  const file = open('duplicate');
  try { assert.throws(() => new AnnDataReader(file).configure('ROI', 'ObjectNumber'), /Duplicate cell identity/); }
  finally { file.close(); }
});
test('unsupported compression is rejected before entering a native HDF5 read', () => {
  const file = open('lzf');
  try { assert.throws(() => new AnnDataReader(file).configure('ROI', 'ObjectNumber'), /Unsupported HDF5 compression/); }
  finally { file.close(); }
});
