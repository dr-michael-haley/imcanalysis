import { defaultBinding } from "./overlays.mjs";
import { cleanGenes, matchGenes } from "./genes.mjs";
import { displayedEntries } from "./view-tools.mjs";

export function propagationSource(dataset, session, paneId, entryId) {
	const pane = session.panes.find((p) => p.id === paneId);
	const entry = pane?.renderStack.entries.find((e) => e.id === entryId);
	if (!entry) throw new Error("Select a cell or Points layer first.");
	let kind, field, binding;
	if (entry.source.elementType === "shapes") {
		binding = pane.cellBindings[entry.id] || defaultBinding(dataset, entry);
		kind = binding.source;
		field = binding.field;
		if (!["gene", "sum", "obs"].includes(kind) || !field)
			throw new Error("Choose a gene, gene list or obs field first.");
	} else if (entry.source.elementType === "points") {
		kind =
			entry.props.geneListId !== undefined
				? "sum"
				: entry.props.featureNames === undefined
					? "all"
					: "genes";
		field = entry.props.geneListId;
	} else throw new Error("Select a cell or Points layer first.");
	const list =
		kind === "sum" ? session.geneLists.find((g) => g.id === field) : null;
	if (kind === "sum" && !list)
		throw new Error("Choose a workspace gene list first.");
	const genes =
		kind === "sum"
			? list.genes
			: kind === "gene"
				? [field]
				: kind === "genes"
					? cleanGenes(entry.props.featureNames)
					: [];
	if (["gene", "genes", "sum"].includes(kind) && !genes.length)
		throw new Error("Select at least one gene first.");
	return {
		paneId,
		entryId,
		kind,
		field,
		binding,
		genes,
		list,
		label: entry.source.elementKey,
	};
}

export function propagationTargets(dataset, session, source, target) {
	return session.panes.flatMap((pane) =>
		displayedEntries(pane)
			.filter(
				(entry) =>
					entry.visible &&
					entry.source.elementType ===
						(target === "cells" ? "shapes" : "points") &&
					!(pane.id === source.paneId && entry.id === source.entryId),
			)
			.map((entry) => ({
				pane,
				entry,
				binding:
					target === "cells"
						? pane.cellBindings[entry.id] || defaultBinding(dataset, entry)
						: null,
			}))
			.filter((item) => target !== "cells" || item.binding.table),
	);
}

// Camera movement and drawer navigation can continue during metadata reads.
// Changes to layers, links or gene definitions invalidate the pending copy.
export function propagationSignature(session) {
	return JSON.stringify({
		coordinateSystem: session.coordinateSystem,
		geneLists: session.geneLists,
		panes: session.panes.map((p) => ({
			id: p.id,
			cellBindings: p.cellBindings,
			entries: p.renderStack.entries.filter((e) =>
				["shapes", "points"].includes(e.source.elementType),
			),
		})),
	});
}

function cellAppearance(binding) {
	return {
		scale: binding.scale || "linear",
		colourmap: binding.colourmap || "viridis",
		reverse: binding.reverse || false,
		vmin: binding.vmin ?? null,
		vmax: binding.vmax ?? null,
	};
}

export async function preparePropagation(
	dataset,
	session,
	paneId,
	entryId,
	target,
	loadGenes = (key) => dataset.tables[key].loadVarIndex(),
) {
	if (!["cells", "points"].includes(target))
		throw new Error("Unknown propagation target.");
	const source = propagationSource(dataset, session, paneId, entryId);
	if (target === "points" && source.kind === "obs")
		throw new Error(
			"Obs fields cannot filter transcripts. Choose a gene or gene list.",
		);
	if (target === "cells" && source.kind === "all")
		throw new Error(
			"Choose selected genes or a gene list before applying to cells.",
		);
	const targets = propagationTargets(dataset, session, source, target);
	const genesCache = new Map(),
		pointsCache = new Map();
	const cached = (cache, key, read) => {
		if (!cache.has(key)) cache.set(key, Promise.resolve().then(read));
		return cache.get(key);
	};
	const outcomes = await Promise.all(
		targets.map(async ({ pane, entry, binding }) => {
			const result = {
				paneId: pane.id,
				entryId: entry.id,
				label: `${pane.name} · ${entry.source.elementKey}`,
				table: binding?.table,
				binding,
			};
			try {
				if (target === "cells" && !dataset.tables[binding.table])
					throw new Error("The linked table is unavailable.");
				if (source.kind === "obs") {
					if (
						!dataset.tables[binding.table]
							.getObsColumnNames()
							.includes(source.field)
					)
						throw new Error(
							`obs “${source.field}” is missing from ${binding.table}.`,
						);
				} else if (source.kind !== "all") {
					const available =
						target === "cells"
							? await cached(genesCache, binding.table, () =>
									loadGenes(binding.table),
								)
							: await cached(pointsCache, entry.source.elementKey, async () =>
									(
										await dataset.points[entry.source.elementKey].listFeatures()
									).entries.map((g) => g.name),
								);
					const { matched, missing } = matchGenes(source.genes, available);
					if (!matched.length)
						throw new Error(
							`None of the selected genes are present: ${missing.join(", ")}.`,
						);
					if (target === "cells") {
						const counts = new Map();
						for (const gene of available)
							counts.set(gene, (counts.get(gene) || 0) + 1);
						const duplicates = matched.filter((gene) => counts.get(gene) > 1);
						if (duplicates.length)
							throw new Error(
								`Ambiguous duplicate genes: ${duplicates.join(", ")}.`,
							);
					}
					result.missing = missing;
					result.matched = matched.length;
				}
				return { ...result, applied: true };
			} catch (e) {
				return { ...result, applied: false, reason: e.message };
			}
		}),
	);
	let list = source.list,
		newList;
	if (
		target === "cells" &&
		source.kind === "genes" &&
		source.genes.length > 1 &&
		outcomes.some((r) => r.applied)
	) {
		const wanted = new Set(source.genes);
		list = session.geneLists.find(
			(g) =>
				g.genes.length === wanted.size &&
				g.genes.every((gene) => wanted.has(gene)),
		);
		if (!list) {
			if (session.geneLists.length >= 100)
				throw new Error(
					"The workspace has 100 gene lists. Reuse or remove a list before applying multiple Points genes to cells.",
				);
			let name = `Points · ${source.label}`.slice(0, 110),
				suffix = 2;
			const base = name;
			while (session.geneLists.some((g) => g.name === name))
				name = `${base} (${suffix++})`;
			list = newList = {
				id: crypto.randomUUID(),
				name,
				genes: [...source.genes],
			};
		}
	}
	let patch;
	if (target === "points")
		patch = {
			geneListId: source.kind === "sum" ? source.field : undefined,
			featureNames:
				source.kind === "all" || source.kind === "sum"
					? undefined
					: [...source.genes],
			colorByFeature: true,
		};
	else
		patch = {
			source: source.kind === "obs" ? "obs" : list ? "sum" : "gene",
			field:
				source.kind === "obs" ? source.field : list ? list.id : source.genes[0],
			...(source.binding
				? cellAppearance(source.binding)
				: { vmin: null, vmax: null }),
		};
	return {
		signature: propagationSignature(session),
		outcomes,
		patch,
		target,
		newList,
		source: { paneId, entryId },
		listName: target === "cells" ? list?.name : undefined,
	};
}

export function applyPropagation(session, plan) {
	if (propagationSignature(session) !== plan.signature)
		throw new Error(
			"Layers or gene choices changed while checking. Apply again to use the current settings.",
		);
	const targets = new Set(
		plan.outcomes
			.filter((r) => r.applied)
			.map((r) => JSON.stringify([r.paneId, r.entryId])),
	);
	if (!targets.size) return session;
	return {
		...session,
		geneLists: plan.newList
			? [...session.geneLists, plan.newList]
			: session.geneLists,
		panes: session.panes.map((pane) => {
			const ids = pane.renderStack.entries
				.filter((e) => targets.has(JSON.stringify([pane.id, e.id])))
				.map((e) => e.id);
			if (!ids.length) return pane;
			if (plan.target === "cells") {
				const cellBindings = { ...pane.cellBindings };
				for (const id of ids) {
					// Only display fields change: each destination keeps its own join.
					const result = plan.outcomes.find(
						(r) => r.paneId === pane.id && r.entryId === id,
					);
					cellBindings[id] = {
						...result.binding,
						...cellBindings[id],
						...plan.patch,
					};
				}
				return { ...pane, cellBindings };
			}
			return {
				...pane,
				renderStack: {
					...pane.renderStack,
					entries: pane.renderStack.entries.map((entry) =>
						ids.includes(entry.id)
							? { ...entry, props: { ...entry.props, ...plan.patch } }
							: entry,
					),
				},
			};
		}),
	};
}
