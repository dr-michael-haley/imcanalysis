const number = value => value === undefined || value === null || String(value).trim() === '' ? NaN : Number(value);

export function panoramaProjection(metadata, width, height) {
  const [[x, y], [x2, y2], [x4, y4]] = [1, 2, 4].map(i => [number(metadata[`SlideX${i}PosUm`]), number(metadata[`SlideY${i}PosUm`])]);
  const det = (x2 - x) * (y4 - y) - (x4 - x) * (y2 - y);
  if (![x, y, x2, y2, x4, y4, det].every(Number.isFinite) || Math.abs(det) < 1e-10) return null;
  return ([px, py]) => [((px - x) * (y4 - y) - (py - y) * (x4 - x)) / det * width, ((py - y) * (x2 - x) - (px - x) * (y2 - y)) / det * height];
}

export function slideProjection(metadata, width, height) {
  const slideWidth = number(metadata.WidthUm), slideHeight = number(metadata.HeightUm);
  if (![slideWidth, slideHeight].every(v => Number.isFinite(v) && v > 0)) return null;
  // Instrument slide coordinates have their origin at the bottom left.
  // Optical image coordinates start at the top left.
  return ([x, y]) => [x / slideWidth * width, (1 - y / slideHeight) * height];
}

export function coordinateOverview(acquisitions) {
  const points = acquisitions.flatMap(a => a.points);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of points) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
  if (!points.length) return { width: 1, height: 1, project: null };
  return { width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY), project: ([x, y]) => [x - minX, maxY - y] };
}
