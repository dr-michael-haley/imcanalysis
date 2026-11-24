// Read on demand, retaining only obs fields with a single value in the plotted regions.
export async function readFigureMetadata(
	dataset,
	system,
	tableKey,
	signal,
	progress = () => {},
) {
	const table = dataset.tables[tableKey];
	if (!table) throw new Error("Choose a table for filename metadata.");
	const regionKey = table.getTableKeys().regionKey;
	if (!regionKey) throw new Error("The table has no region column.");
	const regions = (await table.loadObsColumns([regionKey]))[0];
	if (regions.length > 5000000)
		throw new Error("Metadata reads are limited to five million cells.");
	const wanted = new Set(
		["shapes", "labels"].flatMap((kind) =>
			Object.entries(dataset[kind] || {})
				.filter(([, e]) => e.coordinateSystems.includes(system))
				.map(([key]) => key),
		),
	);
	const rows = Array.from(regions, (r, i) =>
		wanted.has(String(r).replace(/^(shapes|labels)\//, "")) ? i : -1,
	).filter((i) => i >= 0);
	if (!rows.length)
		throw new Error("This table has no rows in the current coordinate system.");
	const columns = table
			.getObsColumnNames()
			.filter((c) => c !== table.getTableKeys().instanceKey),
		fields = [];
	for (let i = 0; i < columns.length; i += 4) {
		signal.throwIfAborted();
		progress(
			`Reading metadata ${Math.min(i + 4, columns.length)} / ${columns.length}…`,
		);
		const names = columns.slice(i, i + 4),
			values = await table.loadObsColumns(names);
		for (let j = 0; j < names.length; j++) {
			const first = values[j]?.[rows[0]];
			if (typeof first === "object" && first !== null) continue;
			if (rows.every((r) => Object.is(values[j]?.[r], first)))
				fields.push({
					name: `${tableKey}.${names[j]}`,
					values: [[system, first]],
				});
		}
	}
	signal.throwIfAborted();
	return fields;
}
