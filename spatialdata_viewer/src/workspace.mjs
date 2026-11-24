import { createWorkspace, validateWorkspace, initialStack } from "./model.mjs";
import { cleanGenes } from "./genes.mjs";
import { defaultEmbedding } from "./embedding-model.mjs";
import { validateColourSettings } from "./colour-scale.mjs";
import {
	defaultFigure,
	validateFigure,
	defaultViewTools,
	validateViewTools,
	addedLayers,
} from "./view-tools.mjs";

export const uid = () => crypto.randomUUID();
export function newSession(dataset, system) {
	const renderStack = dataset
		? initialStack(dataset, system)
		: { schemaVersion: 1, entries: [] };
	if (!renderStack.entries.some((e) => e.visible) && renderStack.entries[0])
		renderStack.entries[0].visible = true;
	return {
		coordinateSystem: system,
		panes: [
			{
				...defaultViewTools(),
				id: "view-1",
				name: "View 1",
				renderStack,
				activeLayers: renderStack.entries
					.filter((e) => e.visible)
					.map((e) => e.id),
				cellBindings: {},
				viewState: null,
				selectedId: renderStack.entries.find((e) => e.visible)?.id || "",
			},
		],
		activePaneId: "view-1",
		linked: true,
		geneLists: [],
		bookmarks: [],
		embedding: defaultEmbedding(),
		figure: defaultFigure(),
		figureReturn: null,
		sidebar: { tab: dataset ? "Layers" : "Setup", hidden: false },
		drawer: { open: false, tab: "Views", height: 260 },
	};
}
export function patchPane(session, id, patch) {
	return {
		...session,
		panes: session.panes.map((p) =>
			p.id === id
				? { ...p, ...(typeof patch === "function" ? patch(p) : patch) }
				: p,
		),
	};
}
export function cameraChange(session, id, viewState) {
	const target = session.panes.find((p) => p.id === id);
	if (!target || JSON.stringify(target.viewState) === JSON.stringify(viewState))
		return session;
	return {
		...session,
		panes: session.panes.map((p) =>
			session.linked || p.id === id ? { ...p, viewState } : p,
		),
	};
}
// Relinking deliberately aligns every camera to the selected pane.
export function setNavigationLinked(session, linked) {
	const camera = session.panes.find(
		(p) => p.id === session.activePaneId,
	)?.viewState;
	return {
		...session,
		linked,
		panes:
			linked && camera
				? session.panes.map((p) => ({
						...p,
						viewState: structuredClone(camera),
					}))
				: session.panes,
	};
}
export function addIndependentPane(session) {
	if (session.panes.length >= 4) return session;
	return duplicatePane(setNavigationLinked(session, false));
}
export function duplicatePane(session, id = uid()) {
	if (session.panes.length >= 4) return session;
	const source = session.panes.find((p) => p.id === session.activePaneId);
	let number = 1;
	while (session.panes.some((p) => p.name === `View ${number}`)) number++;
	return {
		...session,
		panes: [
			...session.panes,
			{
				...structuredClone(source),
				id,
				name: `View ${number}`,
			},
		],
		activePaneId: id,
	};
}
export function removePane(session, id) {
	if (session.panes.length === 1) return session;
	const panes = session.panes.filter((p) => p.id !== id);
	return {
		...session,
		panes,
		activePaneId:
			session.activePaneId === id ? panes[0].id : session.activePaneId,
	};
}
export function snapshot(session) {
	const { coordinateSystem, panes, activePaneId, linked, geneLists } = session;
	return structuredClone({
		coordinateSystem,
		panes,
		activePaneId,
		linked,
		geneLists,
		embedding: session.embedding || defaultEmbedding(),
		figure: session.figure || defaultFigure(),
		figureReturn: session.figureReturn || null,
		sidebar: session.sidebar || { tab: "Layers", hidden: false },
	});
}
export function restoreBookmark(session, bookmark) {
	const saved = structuredClone(bookmark.viewer);
	const ids = new Set(saved.geneLists.map((g) => g.id));
	return {
		...session,
		...saved,
		geneLists: [
			...saved.geneLists,
			...session.geneLists.filter((g) => !ids.has(g.id)),
		],
	};
}
export function exportWorkspace(dataset, source, session) {
	return {
		...createWorkspace(
			dataset,
			source,
			session.coordinateSystem,
			session.panes[0].renderStack,
			session.panes[0].viewState,
		),
		version: 5,
		...session,
	};
}
const fail = (message) => {
	throw new Error(`Invalid workspace: ${message}.`);
};
const text = (v, max) => typeof v === "string" && v.length <= max;
function unique(items, max, label) {
	if (!Array.isArray(items) || items.length > max) fail(label);
	const ids = new Set();
	for (const item of items) {
		if (
			!item ||
			!text(item.id, 100) ||
			!/^[\w-]+$/.test(item.id) ||
			ids.has(item.id)
		)
			fail(`${label} IDs`);
		ids.add(item.id);
	}
}
function validateSnapshot(value, dataset, inventory) {
	validateFigure(value.figure);
	if (
		value.sidebar &&
		(!["Setup", "Layers", "Inspect"].includes(value.sidebar.tab) ||
			typeof value.sidebar.hidden !== "boolean")
	)
		fail("sidebar");
	if (
		value.figureReturn &&
		(typeof value.figureReturn !== "object" ||
			Object.values(value.figureReturn).some(
				(v) =>
					!v ||
					!Number.isFinite(v.width) ||
					v.width <= 0 ||
					!Number.isFinite(v.height) ||
					v.height <= 0 ||
					(v.camera &&
						(!Number.isFinite(v.camera.zoom) ||
							!Array.isArray(v.camera.target) ||
							v.camera.target.some((x) => !Number.isFinite(x)))),
			))
	)
		fail("figure return cameras");
	if (value.embedding) {
		const e = value.embedding;
		validateColourSettings(e);
		if (e.overrideMap !== undefined && typeof e.overrideMap !== "boolean")
			fail("embedding colour map");
		if (
			!text(e.table, 1000) ||
			(e.table && !Object.hasOwn(dataset.tables || {}, e.table)) ||
			!text(e.projection, 1000) ||
			!["view", "none", "gene", "sum", "obs"].includes(e.colourBy) ||
			!text(e.field, 1000) ||
			!["linear", "log1p"].includes(e.scale) ||
			![e.x, e.y].every((n) => Number.isInteger(n) && n >= 0 && n < 10000) ||
			![true, false].includes(e.square) ||
			![true, false].includes(e.context) ||
			![10000, 50000, 100000, 200000].includes(e.limit)
		)
			fail("embedding settings");
		if (
			e.colourBy === "sum" &&
			e.field &&
			!value.geneLists?.some((g) => g.id === e.field)
		)
			fail("embedding gene list");
	}
	unique(value.geneLists, 100, "gene lists");
	for (const list of value.geneLists) {
		if (
			!text(list.name, 120) ||
			!list.name.trim() ||
			!Array.isArray(list.genes) ||
			!list.genes.length ||
			list.genes.some((g) => !text(g, 200))
		)
			fail("gene list");
		if (JSON.stringify(cleanGenes(list.genes)) !== JSON.stringify(list.genes))
			fail("duplicate or empty genes");
	}
	unique(value.panes, 4, "views");
	if (
		!value.panes.length ||
		!value.panes.some((p) => p.id === value.activePaneId) ||
		typeof value.linked !== "boolean"
	)
		fail("active view");
	for (const pane of value.panes) {
		validateViewTools(pane);
		if (
			pane.activeLayers &&
			(!Array.isArray(pane.activeLayers) ||
				new Set(pane.activeLayers).size !== pane.activeLayers.length ||
				pane.activeLayers.some(
					(id) => !pane.renderStack.entries.some((e) => e.id === id),
				))
		)
			fail("active layers");
		if (
			!pane.cellBindings ||
			typeof pane.cellBindings !== "object" ||
			Array.isArray(pane.cellBindings)
		)
			fail("cell bindings");
		if (
			!text(pane.name, 120) ||
			!pane.name.trim() ||
			!text(pane.selectedId, 1000)
		)
			fail("view name or selection");
		validateWorkspace(
			{
				...pane,
				format: "sbt-merscope-explorer",
				version: 2,
				inventory,
				coordinateSystem: value.coordinateSystem,
			},
			dataset,
		);
		if (
			pane.selectedId &&
			!pane.renderStack.entries.some((e) => e.id === pane.selectedId)
		)
			fail("selected layer");
		for (const binding of Object.values(pane.cellBindings || {}))
			if (
				binding.source === "sum" &&
				binding.field &&
				!value.geneLists.some((g) => g.id === binding.field)
			)
				fail("missing cell gene list");
		for (const entry of pane.renderStack.entries)
			if (
				entry.props.geneListId &&
				!value.geneLists.some((g) => g.id === entry.props.geneListId)
			)
				fail("missing transcript gene list");
	}
}
export function importWorkspace(value, dataset) {
	if (
		value?.format !== "sbt-merscope-explorer" ||
		![3, 4, 5].includes(value.version)
	) {
		validateWorkspace(value, dataset);
		const session = newSession(dataset, value.coordinateSystem);
		return {
			...session,
			panes: [
				{
					...session.panes[0],
					renderStack: value.renderStack,
					activeLayers: value.renderStack.entries
						.filter((e) => e.visible)
						.map((e) => e.id),
					cellBindings: value.cellBindings || {},
					viewState: value.viewState,
				},
			],
		};
	}
	validateSnapshot(value, dataset, value.inventory);
	unique(value.bookmarks, 500, "bookmarks");
	for (const b of value.bookmarks) {
		if (
			!text(b.name, 120) ||
			!b.name.trim() ||
			!text(b.comment, 10000) ||
			!text(b.createdAt, 80)
		)
			fail("bookmark");
		validateSnapshot(b.viewer, dataset, value.inventory);
	}
	if (
		!value.drawer ||
		![
			"Views",
			"Bookmarks",
			"Gene lists",
			"Embeddings",
			"Figure",
			"Workspace",
		].includes(value.drawer.tab) ||
		typeof value.drawer.open !== "boolean" ||
		!Number.isFinite(value.drawer.height) ||
		value.drawer.height < 180 ||
		value.drawer.height > 600
	)
		fail("drawer");
	const upgrade = (s) => ({
		...snapshot(s),
		panes: s.panes.map((p) => ({
			...p,
			...validateViewTools(p),
			activeLayers: p.renderStack.entries
				.filter((e) => e.visible || addedLayers(p).includes(e.id))
				.map((e) => e.id),
		})),
	});
	return {
		...upgrade(value),
		bookmarks: value.bookmarks.map((b) => ({
			...structuredClone(b),
			viewer: upgrade(b.viewer),
		})),
		drawer: { ...value.drawer },
	};
}
