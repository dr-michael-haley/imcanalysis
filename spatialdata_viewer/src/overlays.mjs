import { continuousScale, validNumber } from "./colour-scale.mjs";
import { matchGenes } from "./genes.mjs";
const colours = [
	"#81d8c1",
	"#f0a35b",
	"#9f8bea",
	"#5db5f3",
	"#ed7a96",
	"#d3c866",
	"#7bd186",
	"#c887c3",
	"#a8c9cf",
	"#ce987c",
];
const rgba = (hex) => [
	...hex
		.slice(1)
		.match(/../g)
		.map((v) => parseInt(v, 16)),
	255,
];
const missing = [110, 118, 130, 130];

// Match by (region, instance ID), retaining the original table row. Never infer a join from row order.
export function joinRows(ids, regions, elementKey, values) {
	if (regions.length !== ids.length || (values && values.length !== ids.length))
		throw new Error("Table columns have inconsistent lengths.");
	const rows = new Map();
	for (let i = 0; i < ids.length; i++) {
		if (![elementKey, `shapes/${elementKey}`].includes(String(regions[i])))
			continue;
		if (ids[i] == null || String(ids[i]) === "")
			throw new Error("The cell ID column contains a missing ID.");
		const id = String(ids[i]);
		if (rows.has(id))
			throw new Error(
				`Duplicate cell ID ${id} within this region; choose a unique ID column.`,
			);
		rows.set(id, { row: i, value: values?.[i] });
	}
	return rows;
}

export function encodeOverlay(
	rows,
	featureIds,
	mode,
	scale = "linear",
	options = {},
) {
	const present = featureIds.map((id) => rows.get(String(id))),
		values = present.map((row) => row?.value);
	const categories =
		mode === "categorical"
			? [...new Set(values.filter((v) => v != null).map(String))].sort()
			: [];
	const categoryColours = new Map(
		categories.map((v, i) => [v, rgba(colours[i % colours.length])]),
	);
	const ramp =
		mode === "continuous"
			? continuousScale(values, { ...options, scale })
			: { low: 0, high: 0, dataLow: 0, dataHigh: 0 };
	const colour = (v) =>
		mode === "categorical"
			? categoryColours.get(String(v)) || missing
			: ramp.colour(v);
	const colors = new Uint8Array(featureIds.length * 4);
	values.forEach((v, i) => colors.set(colour(v), i * 4));
	return {
		...ramp,
		colour,
		buffer: { colors, count: featureIds.length },
		matched: present.filter(Boolean).length,
		total: featureIds.length,
		missing: values.filter((v) =>
			mode === "continuous" ? !validNumber(v) : v == null,
		).length,
		categories: categories.map((label) => ({
			label,
			colour: categoryColours.get(label),
		})),
	};
}

export function defaultBinding(dataset, entry) {
	const [key, table] =
		dataset
			.getAssociatedTables("shapes", entry.source.elementKey)
			.sort(([a], [b]) => a.localeCompare(b))[0] || [];
	return {
		table: key || "",
		idColumn: table?.getTableKeys().instanceKey || "",
		source: "none",
		field: "",
		scale: "linear",
	};
}

export async function readOverlay(
	dataset,
	entry,
	binding,
	signal,
	geneLists = [],
) {
	const table = dataset.tables[binding.table];
	if (!table) throw new Error("Choose a cell table.");
	const keys = table.getTableKeys();
	if (!keys.regionKey || !binding.idColumn)
		throw new Error("This table needs region and cell ID columns.");
	const [ids, regions] = await table.loadObsColumns([
		binding.idColumn,
		keys.regionKey,
	]);
	signal?.throwIfAborted();
	if (!ids || !regions)
		throw new Error("The linking columns could not be read.");
	const { values, mode, geneSummary } = await readValues(
		table,
		binding,
		signal,
		geneLists,
	);
	signal?.throwIfAborted();
	const rows = joinRows(ids, regions, entry.source.elementKey, values);
	const featureIds =
		await dataset.shapes[entry.source.elementKey].loadFeatureIds();
	signal?.throwIfAborted();
	return { rows, featureIds, values, mode, geneSummary, tableRows: ids.length };
}

export async function readValues(table, binding, signal, geneLists = []) {
	let values,
		mode = "continuous";
	let geneSummary;
	if (binding.source === "gene" && binding.field)
		values = await table.loadExpression(binding.field, signal);
	else if (binding.source === "sum" && binding.field) {
		const list = geneLists.find((g) => g.id === binding.field);
		if (!list) throw new Error("Choose an existing gene list.");
		geneSummary = {
			...matchGenes(list.genes, await table.loadVarIndex()),
			name: list.name,
		};
		if (!geneSummary.matched.length)
			throw new Error(
				`None of the ${list.genes.length} genes in ${list.name} are present in this table.`,
			);
		// Keep only one gene vector plus the accumulator; never densify the whole matrix.
		for (const gene of geneSummary.matched) {
			signal?.throwIfAborted();
			const column = await table.loadExpression(gene, signal);
			if (!values) values = new Float64Array(column.length);
			if (column.length !== values.length)
				throw new Error("Gene columns have inconsistent lengths.");
			for (let i = 0; i < values.length; i++) values[i] += column[i];
		}
	} else if (binding.source === "obs" && binding.field) {
		[values] = await table.loadObsColumns([binding.field]);
		if (!values)
			throw new Error(
				"The selected observation column could not be read. Choose another field.",
			);
		const kind = table.getObsColumnKinds([binding.field])[0];
		mode = kind === "numeric" ? "continuous" : "categorical";
	}
	signal?.throwIfAborted();
	return { values, mode, geneSummary };
}

// Display changes recolour cached values; moving a slider never rereads X.
export function styleOverlay(data, binding) {
	const encoded = encodeOverlay(
			data.rows,
			data.featureIds,
			data.mode,
			binding.scale,
			binding,
		),
		buffers = new WeakMap();
	return {
		...data,
		...encoded,
		binding,
		label: data.geneSummary ? `Σ ${data.geneSummary.name}` : binding.field,
		resolve(ids) {
			if (!data.values || !encoded.matched) return undefined;
			if (!buffers.has(ids)) {
				const colors = new Uint8Array(ids.length * 4);
				ids.forEach((id, i) =>
					colors.set(encoded.colour(data.rows.get(String(id))?.value), i * 4),
				);
				buffers.set(ids, { colors, count: ids.length });
			}
			return buffers.get(ids);
		},
	};
}
export async function loadOverlay(
	dataset,
	entry,
	binding,
	signal,
	geneLists = [],
) {
	return styleOverlay(
		await readOverlay(dataset, entry, binding, signal, geneLists),
		binding,
	);
}
