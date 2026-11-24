// Binary logistic gradient boosting over quantile-binned numeric cell features.
// Persist the bin boundaries with the trees so training and inference agree.
function bin(value, cuts, median) {
  if (!Number.isFinite(value)) value = median;
  let low = 0,
    high = cuts.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (value > cuts[mid]) low = mid + 1;
    else high = mid;
  }
  return low;
}

function leaf(tree, bins) {
  while (tree.f !== undefined) tree = bins[tree.f] <= tree.cut ? tree.left : tree.right;
  return tree.value;
}
const logistic = x => 1 / (1 + Math.exp(-x));

export function trainBinary(features, labels, columns) {
  const positives = labels.filter(y => y === 1).length;
  if (!positives || positives === labels.length || features.length !== labels.length)
    throw Error('Label both good and artefact examples before training.');
  const cuts = [],
    medians = [];
  for (let f = 0; f < columns.length; f++) {
    const values = features
      .map(row => row[f])
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    medians.push(values[Math.floor(values.length / 2)] ?? 0);
    const boundaries = [];
    const bins = Math.min(64, values.length);
    for (let i = 1; i < bins; i++) {
      const value = values[Math.ceil((i * values.length) / bins) - 1];
      if (value < values.at(-1) && value !== boundaries.at(-1)) boundaries.push(value);
    }
    cuts.push(boundaries);
  }
  if (!cuts.some(c => c.length))
    throw Error('Labelled cells have no varying features. Add examples with different measurements.');
  const rows = features.map(row => Uint8Array.from(cuts.map((c, f) => bin(row[f], c, medians[f]))));
  const n = rows.length,
    minLeaf = Math.max(1, Math.min(20, Math.floor(n / 8)));
  const weights = labels.map(y => n / (2 * (y ? positives : n - positives)));
  const logits = new Float64Array(n),
    g = new Float64Array(n),
    h = new Float64Array(n);
  const trees = [],
    rate = 0.05,
    indices = Array.from({ length: n }, (_, i) => i);
  const quality = (grad, hess) => (grad * grad) / (hess + 1);
  function build(ids, depth) {
    let G = 0,
      H = 0;
    for (const i of ids) {
      G += g[i];
      H += h[i];
    }
    const terminal = { value: G / (H + 1) };
    if (depth === 3 || ids.length < minLeaf * 2) return terminal;
    let feature = -1,
      cut = 0,
      gain = 1e-9;
    for (let f = 0; f < cuts.length; f++) {
      const size = cuts[f].length + 1;
      if (size < 2) continue;
      const count = new Uint32Array(size),
        grad = new Float64Array(size),
        hess = new Float64Array(size);
      for (const i of ids) {
        const b = rows[i][f];
        count[b]++;
        grad[b] += g[i];
        hess[b] += h[i];
      }
      let left = 0,
        lg = 0,
        lh = 0;
      for (let b = 0; b < size - 1; b++) {
        left += count[b];
        lg += grad[b];
        lh += hess[b];
        if (left < minLeaf || ids.length - left < minLeaf) continue;
        const improvement = quality(lg, lh) + quality(G - lg, H - lh) - quality(G, H);
        if (improvement > gain) {
          gain = improvement;
          feature = f;
          cut = b;
        }
      }
    }
    if (feature < 0) return terminal;
    const left = [],
      right = [];
    for (const i of ids) (rows[i][feature] <= cut ? left : right).push(i);
    return { f: feature, cut, left: build(left, depth + 1), right: build(right, depth + 1) };
  }
  for (let round = 0; round < 250; round++) {
    for (let i = 0; i < n; i++) {
      const p = logistic(logits[i]);
      g[i] = weights[i] * (labels[i] - p);
      h[i] = weights[i] * p * (1 - p);
    }
    const tree = build(indices, 0);
    if (tree.f === undefined && !trees.length)
      throw Error('These examples do not separate. Label additional cells from both classes.');
    if (tree.f === undefined && Math.abs(tree.value) < 1e-9) break;
    trees.push(tree);
    for (let i = 0; i < n; i++) logits[i] += rate * leaf(tree, rows[i]);
  }
  return { type: 'histogram-gradient-boosting', version: 1, columns, rate, cuts, medians, trees, examples: n };
}

export function predictBinary(model, values) {
  if (!model || !values.some(Number.isFinite)) return null;
  const bins = model.cuts.map((cuts, f) => bin(values[f], cuts, model.medians[f]));
  let sum = 0;
  for (const tree of model.trees) sum += model.rate * leaf(tree, bins);
  return logistic(sum);
}
