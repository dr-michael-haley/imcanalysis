export const defaultEmbedding = () => ({
	table: "",
	projection: "",
	x: 0,
	y: 1,
	colourBy: "view",
	field: "",
	scale: "linear",
	colourmap: "viridis",
	reverse: false,
	vmin: null,
	vmax: null,
	square: false,
	context: false,
	limit: 50000,
});
export const cellKey = (region, id) =>
	JSON.stringify([
		String(region).replace(/^(shapes|labels)\//, ""),
		String(id),
	]);
export function sampledRows(count, limit) {
	const size = Math.min(count, limit);
	if (size === count) return Array.from({ length: size }, (_, i) => i);
	// Deterministic jitter within disjoint strata avoids periodic ROI ordering.
	return Array.from({ length: size }, (_, i) => {
		let hash = Math.imul(i + 1, 0x45d9f3b);
		hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
		hash = (hash ^ (hash >>> 16)) >>> 0;
		const start = Math.floor((i * count) / size),
			end = Math.floor(((i + 1) * count) / size);
		return start + Math.floor((hash / 4294967296) * (end - start));
	});
}
export function pointInPolygon(x, y, polygon) {
	let inside = false;
	for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
		const [xi, yi] = polygon[i],
			[xj, yj] = polygon[j];
		if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
			inside = !inside;
	}
	return inside;
}
export function plotPoints(
	data,
	{ region = "", context = false, limit = 50000 } = {},
) {
	const current = [],
		others = [];
	let invalid = 0;
	data.rows.forEach((row, i) => {
		if (
			!Number.isFinite(data.xy[i * 2]) ||
			!Number.isFinite(data.xy[i * 2 + 1])
		) {
			invalid++;
			return;
		}
		if (!region || cellKey(data.regions[row], "") === cellKey(region, ""))
			current.push(i);
		else others.push(i);
	});
	const makePoint = (i) => ({
		row: data.rows[i],
		id: data.ids[data.rows[i]],
		region: data.regions[data.rows[i]],
		x: data.xy[2 * i],
		y: data.xy[2 * i + 1],
	});
	const currentShown = Math.min(current.length, 200000);
	return {
		points: [
			...(context
				? sampledRows(others.length, limit).map((i) => ({
						...makePoint(others[i]),
						context: true,
					}))
				: []),
			...sampledRows(current.length, currentShown || 1).map((i) =>
				makePoint(current[i]),
			),
		],
		current: current.length,
		currentShown,
		others: others.length,
		invalid,
	};
}
