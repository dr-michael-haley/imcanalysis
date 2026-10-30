// Smooth bounded histogram aggregates rather than retaining per-cell vectors.
// Both groups use the same Gaussian bandwidth (1.5 bins). Reflection at the
// observed range boundaries retains zero-heavy distributions without plotting
// artificial negative expression. Each curve has unit area on this range.
export function densityCurves(histograms, min, max) {
  const width = max - min, bins = histograms[0]?.length;
  if (!Number.isFinite(width) || width <= 0 || !bins || histograms.some(values => values.length !== bins || values.some(n => !Number.isFinite(n) || n < 0))) throw new Error('Invalid expression distribution.');
  const steps = 256, bandwidth = 1.5 / bins;
  const positions = Array.from({ length: steps + 1 }, (_, i) => i / steps);
  const curves = histograms.map(values => {
    const count = values.reduce((a, b) => a + b, 0);
    if (!count) return positions.map(() => 0);
    const kernel = distance => Math.exp(-0.5 * (distance / bandwidth) ** 2);
    const density = positions.map(x => values.reduce((sum, n, i) => {
      if (!n) return sum;
      const centre = (i + 0.5) / bins;
      return sum + n / count * (kernel(x - centre) + kernel(x + centre) + kernel(x - (2 - centre)));
    }, 0));
    const area = density.reduce((sum, value, i) => sum + value * (i === 0 || i === steps ? 0.5 : 1), 0) / steps;
    return density.map(value => value / area / width);
  });
  return { x: positions.map(t => min + t * width), curves, maximum: Math.max(...curves.flat()) };
}

export function densityPath(values, maximum) {
  const height = 58, baseline = 64;
  return values.map((value, i) => `${i ? 'L' : 'M'}${(2 + i / (values.length - 1) * 256).toFixed(2)},${(baseline - (maximum ? value / maximum : 0) * height).toFixed(2)}`).join(' ');
}

// Quantiles use pooled cell counts, rather than equally weighting the two groups.
// Within-bin interpolation is approximate; changing the window never rereads X.
export function densityWindow(density, histograms, lower = 0, upper = 1) {
  if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower < 0 || upper > 1 || lower >= upper) throw new Error('Use 0 ≤ lower quantile < upper quantile ≤ 1.');
  const min = density.x[0], max = density.x.at(-1), bins = histograms[0].length;
  const counts = histograms[0].map((_, i) => histograms.reduce((sum, values) => sum + values[i], 0)), total = counts.reduce((a, b) => a + b, 0);
  const quantile = q => {
    if (q === 0) return min; if (q === 1 || !total) return max;
    const target = q * total; let seen = 0;
    for (let i = 0; i < bins; i++) { if (counts[i] && seen + counts[i] >= target) return min + (i + (target - seen) / counts[i]) / bins * (max - min); seen += counts[i]; }
    return max;
  };
  const low = quantile(lower), high = quantile(upper);
  const x = Array.from({ length: 257 }, (_, i) => low + i / 256 * (high - low));
  const curves = density.curves.map(values => x.map(value => {
    const position = (value - min) / (max - min) * (values.length - 1), index = Math.min(values.length - 2, Math.floor(position)), fraction = position - index;
    return values[index] * (1 - fraction) + values[index + 1] * fraction;
  }));
  return { x, curves, maximum: Math.max(...curves.flat()) };
}
