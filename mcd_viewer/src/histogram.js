// Shared raw-count bins permit direct comparisons. Missing and non-finite pixels are excluded.
export function histograms(planes, binCount = 128) {
  if (!Number.isInteger(binCount) || binCount < 2 || binCount > 1024) throw new Error('Invalid histogram bin count.');
  let low = Infinity, high = -Infinity;
  for (const { data } of planes) for (const v of data) if (Number.isFinite(v)) { low = Math.min(low, v); high = Math.max(high, v); }
  if (!Number.isFinite(low)) return { low: 0, high: 1, bins: binCount, curves: [] };
  if (low === high) { low = Math.min(0, low); high = Math.max(low + .1, high); }
  const span = high - low;
  return { low, high, bins: binCount, curves: planes.map(({ index, data }) => {
    const counts = new Uint32Array(binCount); let samples = 0;
    for (const v of data) if (Number.isFinite(v)) { counts[Math.max(0, Math.min(binCount - 1, Math.floor((v - low) / span * binCount)))]++; samples++; }
    return { index, counts, samples, missing: data.length - samples };
  }) };
}
