import test from "node:test";
import assert from "node:assert/strict";
import {
	addLayer,
	removeLayer,
	reorderLayer,
	defaultViewTools,
	defaultFigure,
	validateViewTools,
	validateFigure,
	figureCamera,
	exportDimensions,
	scaleBarGeometry,
} from "../src/view-tools.mjs";
import { paneOverlaySvg, customiseLegend } from "../src/pane-overlays.mjs";
import {
	shapeCentres,
	selectInPane,
	mergeSelection,
} from "../src/spatial-selection.mjs";
import { readFigureMetadata } from "../src/figure-metadata.mjs";
import { createFigureExport } from "../src/figure-export.mjs";
const entry = (id, type = "shapes", visible = true) => ({
	id,
	source: { elementType: type, elementKey: id },
	visible,
	props: { opacity: 0.5 },
});
const pane = () => ({
	...defaultViewTools(),
	id: "one",
	name: "Cells",
	activeLayers: ["a"],
	selectedId: "a",
	renderStack: {
		schemaVersion: 1,
		entries: [entry("a"), entry("b", "points", false)],
	},
	viewState: { target: [100, 200, 0], zoom: 1 },
	cellBindings: {},
});
test("added layers retain display settings through hiding, removing, readding and order changes", () => {
	const p = pane(),
		added = addLayer(p, "b");
	assert.deepEqual(added.activeLayers, ["a", "b"]);
	assert.equal(added.selectedId, "b");
	const removed = removeLayer(added, "b");
	assert.equal(removed.renderStack.entries[1].visible, false);
	assert.equal(removed.renderStack.entries[1].props.opacity, 0.5);
	const reordered = reorderLayer(addLayer(removed, "b"), "b", -1);
	assert.deepEqual(
		reordered.renderStack.entries.map((e) => e.id),
		["b", "a"],
	);
	assert.equal(p.renderStack.entries[1].visible, false);
	const reverseSource = {
		...p,
		renderStack: {
			schemaVersion: 1,
			entries: [...p.renderStack.entries].reverse(),
		},
	};
	const appended = addLayer(reverseSource, "b");
	assert.deepEqual(
		appended.renderStack.entries.filter((e) => e.visible).map((e) => e.id),
		appended.activeLayers,
	);
});
test("legends use per-view labels, preserve true colours, omit hidden entries and escape SVG text", () => {
	const p = pane(),
		groups = [
			{
				id: "a",
				title: "Cells",
				rows: [
					{ key: "t", name: "T", colour: [255, 0, 0] },
					{ key: "b", name: "B", colour: [0, 0, 255] },
				],
			},
		];
	p.legend.layers.a = {
		title: "<script>Title</script>",
		labels: { t: "CD3 & T" },
		hidden: ["b"],
	};
	const custom = customiseLegend(groups, p.legend);
	assert.equal(custom[0].rows.length, 1);
	assert.deepEqual(custom[0].rows[0].colour, [255, 0, 0]);
	const svg = paneOverlaySvg(p, groups, defaultFigure(), 800, 600, 1);
	assert.match(svg, /&lt;script&gt;Title/);
	assert.match(svg, /CD3 &amp; T/);
	assert.doesNotMatch(svg, /<script>/);
	assert.match(svg, /units/);
	assert.match(svg, /layer-names/);
});
test("physical figure size and scale bars agree with logarithmic camera zoom", () => {
	const s = {
		...defaultFigure(),
		mode: "figure",
		calibrated: true,
		umPerPixel: 0.5,
		physicalLock: true,
		physicalWidth: 200,
		physicalHeight: 100,
		paneWidth: 800,
		gap: 20,
	};
	const camera = figureCamera(pane().viewState, 800, 400, s);
	assert.equal(camera.zoom, 1);
	assert.equal((800 / 2 ** camera.zoom) * s.umPerPixel, 200);
	const dimensions = exportDimensions(4, { width: 1000, height: 700 }, s);
	assert.equal(dimensions.paneHeight, 400);
	assert.equal(dimensions.outputWidth, 1620);
	assert.equal(dimensions.outputHeight, 820);
	assert.equal(
		scaleBarGeometry(
			{ ...pane().scaleBar, automatic: false, length: 50 },
			0.5,
			2 ** camera.zoom,
			800,
		).pixels,
		200,
	);
	assert.throws(
		() =>
			exportDimensions(
				4,
				{ width: 1000, height: 700 },
				{ ...s, paneWidth: 8192 },
			),
		/exceeds/,
	);
	assert.throws(
		() => validateViewTools({ ...pane(), legend: { fontSize: -1 } }),
		/legend/,
	);
	assert.throws(
		() => validateFigure({ ...s, background: "red;url(x)" }),
		/Invalid/,
	);
	assert.throws(
		() => validateFigure({ ...s, calibrated: false }),
		/calibration/,
	);
});
test("tissue lasso uses transformed shape centres and exact IDs, while click toggles selections", async () => {
	const element = {
		loadRenderData: async () => ({
			featureIds: ["3813280800012100002", "3813280800012100001"],
			circles: {
				positions: [
					[5, 15],
					[10, 20],
				],
			},
		}),
		getTransformation: () => ({
			ok: true,
			value: {
				toMatrix: () => [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 10, 20, 0, 1],
			},
		}),
	};
	const cells = await shapeCentres(element, "a", "global");
	assert.deepEqual([cells[0].x, cells[0].y], [20, 40]);
	const selected = selectInPane(
		cells,
		{ target: [20, 40], zoom: 0 },
		100,
		100,
		[
			[45, 45],
			[55, 45],
			[55, 55],
			[45, 55],
		],
	);
	assert.equal(selected.length, 1);
	assert.equal(selected[0].id, "3813280800012100002");
	assert.equal(mergeSelection(selected, selected, "toggle").length, 0);
	assert.equal(
		mergeSelection(selected, [{ ...selected[0], region: "b" }], "add").length,
		2,
	);
});
test("filename metadata only includes obs that are constant in regions of the coordinate system", async () => {
	const values = {
		region: ["a", "a", "other"],
		id: ["1", "2", "3"],
		case: ["Case 1", "Case 1", "Case 2"],
		cluster: ["T", "B", "T"],
	};
	const dataset = {
		shapes: { a: { coordinateSystems: ["roi"] } },
		tables: {
			t: {
				getTableKeys: () => ({ regionKey: "region", instanceKey: "id" }),
				getObsColumnNames: () => Object.keys(values),
				loadObsColumns: async (cols) => cols.map((c) => values[c]),
			},
		},
	};
	const fields = await readFigureMetadata(
		dataset,
		"roi",
		"t",
		new AbortController().signal,
	);
	assert.deepEqual(fields.find((f) => f.name === "t.case").values, [
		["roi", "Case 1"],
	]);
	assert.equal(
		fields.some((f) => f.name === "t.cluster"),
		false,
	);
});
test("SVG figure export separates layers, carries spatial metadata and scales export cameras", async () => {
	const p = addLayer(pane(), "b"),
		calls = [];
	const settings = {
		...defaultFigure(),
		mode: "figure",
		format: "svg",
		paneWidth: 400,
		aspect: "1:1",
	};
	const blob = await createFigureExport({
		session: {
			figure: settings,
			panes: [p],
			geneLists: [],
			coordinateSystem: "global",
			figureReturn: { one: { width: 200, height: 200 } },
		},
		dataset: {},
		groups: {},
		overlays: {},
		rect: { width: 800, height: 600 },
		sizes: { one: { width: 400 } },
		render: async (job) => {
			calls.push(job);
			return "data:image/png;base64,AA==";
		},
		signal: new AbortController().signal,
		progress: () => {},
	});
	const svg = await blob.text();
	assert.equal(calls.length, 2);
	assert.equal(calls[0].camera.zoom, 2);
	assert.match(svg, /inkscape:label="a"/);
	assert.match(svg, /inkscape:label="b"/);
	assert.match(svg, /coordinateWidth/);
	assert.match(svg, /width="400" height="400"/);
});
