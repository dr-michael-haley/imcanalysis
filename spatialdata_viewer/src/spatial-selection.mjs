import { cellKey, pointInPolygon } from "./embedding-model.mjs";
const cache = new WeakMap();
export function geometryCentre(data, i) {
	if (data.circles)
		return [data.circles.positions[0][i], data.circles.positions[1][i]];
	let xmin = Infinity,
		xmax = -Infinity,
		ymin = Infinity,
		ymax = -Infinity;
	const add = (x, y) => {
		xmin = Math.min(xmin, x);
		xmax = Math.max(xmax, x);
		ymin = Math.min(ymin, y);
		ymax = Math.max(ymax, y);
	};
	if (data.polygonBinary) {
		const { positions, startIndices } = data.polygonBinary;
		for (let k = startIndices[i]; k < startIndices[i + 1]; k++)
			add(positions[k * 2], positions[k * 2 + 1]);
	} else
		for (const ring of data.polygons?.[i] || [])
			for (const p of ring) add(p[0], p[1]);
	return [(xmin + xmax) / 2, (ymin + ymax) / 2];
}
export async function shapeCentres(element, region, system) {
	if (!cache.has(element)) cache.set(element, new Map());
	const saved = cache.get(element);
	if (!saved.has(system))
		saved.set(
			system,
			(async () => {
				const transform = element.getTransformation(system);
				if (!transform.ok) throw new Error(`No transformation for ${region}.`);
				const data = await element.loadRenderData(),
					m = transform.value.toMatrix();
				if (data.featureIds.length > 2000000)
					throw new Error(
						"Shape selection is limited to two million boundaries per layer.",
					);
				return data.featureIds
					.map((id, i) => {
						const [x, y] = geometryCentre(data, i);
						return {
							id: String(id),
							region,
							x: m[0] * x + m[4] * y + m[12],
							y: m[1] * x + m[5] * y + m[13],
						};
					})
					.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
			})().catch((error) => {
				saved.delete(system);
				throw error;
			}),
		);
	return saved.get(system);
}
export function projectPoint(point, camera, width, height) {
	const scale = 2 ** camera.zoom;
	return [
		(point.x - camera.target[0]) * scale + width / 2,
		(point.y - camera.target[1]) * scale + height / 2,
	];
}
export function selectInPane(points, camera, width, height, polygon) {
	return points.filter((point) => {
		const [x, y] = projectPoint(point, camera, width, height);
		return (
			x >= 0 &&
			y >= 0 &&
			x <= width &&
			y <= height &&
			(!polygon || pointInPolygon(x, y, polygon))
		);
	});
}
export function mergeSelection(current, points, mode = "replace") {
	if (mode === "replace") return points;
	const result = new Map(current.map((p) => [cellKey(p.region, p.id), p]));
	for (const p of points) {
		const key = cellKey(p.region, p.id);
		if (mode === "toggle" && result.has(key)) result.delete(key);
		else result.set(key, p);
	}
	return [...result.values()];
}
