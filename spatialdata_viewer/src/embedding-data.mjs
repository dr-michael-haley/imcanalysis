import * as zarr from "zarrita";
import {
	getChildGroup,
	getChildNode,
	getArrayMetadata,
	getNodeAttrs,
	getArrayDtype,
} from "zarrextra";

export function embeddingCatalog(dataset, tableKey) {
	const rank = (key) =>
		/umap/i.test(key)
			? 0
			: /tsne|t_sne/i.test(key)
				? 1
				: /pca/i.test(key)
					? 2
					: key === "spatial"
						? 3
						: 4;
	const group = getChildGroup(
		dataset?.rootStore.tree,
		"tables",
		tableKey,
		"obsm",
	);
	return Object.entries(group || {})
		.flatMap(([key, node]) => {
			const metadata = getArrayMetadata(node),
				attrs = getNodeAttrs(node);
			const shape = metadata?.shape;
			if (
				shape?.length === 2 &&
				shape[1] >= 2 &&
				/^(float|int|uint)/.test(getArrayDtype(node))
			)
				return [{ key, dimensions: shape[1], rows: shape[0], node }];
			// AnnData also permits dataframe-encoded obsm entries.
			if (attrs?.["encoding-type"] === "dataframe") {
				const columns =
					attrs["column-order"]?.filter((k) =>
						/^(float|int|uint)/.test(getArrayDtype(node[k])),
					) || [];
				if (columns.length >= 2)
					return [
						{
							key,
							dimensions: columns.length,
							rows: getArrayMetadata(node[columns[0]])?.shape[0],
							node,
							columns,
						},
					];
			}
			return [];
		})
		.sort((a, b) => rank(a.key) - rank(b.key) || a.key.localeCompare(b.key));
}
export async function loadEmbedding(
	dataset,
	tableKey,
	projection,
	x,
	y,
	signal,
	progress = () => {},
) {
	const item = embeddingCatalog(dataset, tableKey).find(
		(p) => p.key === projection,
	);
	if (!item)
		throw new Error(
			"This table has no supported projection with at least two numeric dimensions.",
		);
	if (
		![x, y].every(
			(v) => Number.isInteger(v) && v >= 0 && v < item.dimensions,
		) ||
		x === y
	)
		throw new Error("Choose two different projection dimensions.");
	if (item.rows > 5000000)
		throw new Error("Embedding exceeds the five million cell read limit.");
	const table = dataset.tables[tableKey],
		keys = table.getTableKeys();
	const [ids, regions] =
		keys.instanceKey && keys.regionKey
			? await table.loadObsColumns([keys.instanceKey, keys.regionKey])
			: [await table.loadObsIndex(), Array(item.rows).fill("")];
	if (ids?.length !== item.rows || regions?.length !== item.rows)
		throw new Error(
			"Projection rows do not match the table cell IDs and regions.",
		);
	const xy = new Float64Array(item.rows * 2),
		rows = Array.from({ length: item.rows }, (_, i) => i);
	const arrays = item.columns
		? await Promise.all([x, y].map((i) => item.node[item.columns[i]].get()))
		: [await item.node.get()];
	for (let start = 0; start < item.rows; start += 8192) {
		signal?.throwIfAborted();
		const end = Math.min(item.rows, start + 8192),
			slice = zarr.slice(start, end);
		const columns = await Promise.all(
			[x, y].map((dim, i) =>
				zarr.get(
					arrays[item.columns ? i : 0],
					item.columns ? [slice] : [slice, dim],
				),
			),
		);
		for (let i = 0; i < end - start; i++) {
			xy[(start + i) * 2] = Number(columns[0].data[i]);
			xy[(start + i) * 2 + 1] = Number(columns[1].data[i]);
		}
		progress(end / item.rows);
	}
	signal?.throwIfAborted();
	// Ambiguous IDs cannot be selected by identity even when rows have coordinates.
	const seen = new Set();
	for (let i = 0; i < ids.length; i++) {
		const k = JSON.stringify([String(regions[i]), String(ids[i])]);
		if (ids[i] == null || seen.has(k))
			throw new Error(
				"Table cell IDs are missing or duplicated within a region.",
			);
		seen.add(k);
	}
	return {
		xy,
		rows,
		ids: Array.from(ids, String),
		regions: Array.from(regions, String),
		table: tableKey,
		projection,
	};
}

export async function cellCentre(dataset, region, id, system) {
	const key = region.replace(/^shapes\//, ""),
		element = dataset.shapes[key];
	if (!element || !element.coordinateSystems.includes(system)) return null;
	const data = await element.loadRenderData(),
		i = data.featureIds.indexOf(String(id));
	if (i < 0) return null;
	let x, y;
	if (data.circles) {
		x = data.circles.positions[0][i];
		y = data.circles.positions[1][i];
	} else if (data.polygonBinary) {
		const { positions, startIndices } = data.polygonBinary;
		let xmin = Infinity,
			xmax = -Infinity,
			ymin = Infinity,
			ymax = -Infinity;
		for (let k = startIndices[i]; k < startIndices[i + 1]; k++) {
			xmin = Math.min(xmin, positions[k * 2]);
			xmax = Math.max(xmax, positions[k * 2]);
			ymin = Math.min(ymin, positions[k * 2 + 1]);
			ymax = Math.max(ymax, positions[k * 2 + 1]);
		}
		x = (xmin + xmax) / 2;
		y = (ymin + ymax) / 2;
	} else if (data.polygons) {
		const points = data.polygons[i]?.flat() || [];
		x = points.reduce((s, p) => s + p[0], 0) / points.length;
		y = points.reduce((s, p) => s + p[1], 0) / points.length;
	}
	if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
	const transform = element.getTransformation(system);
	if (!transform.ok) return null;
	const m = transform.value.toMatrix();
	return [m[0] * x + m[4] * y + m[12], m[1] * x + m[5] * y + m[13], 0];
}
