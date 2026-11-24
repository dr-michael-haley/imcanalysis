import test from "node:test";
import assert from "node:assert/strict";
import { continuousScale } from "../src/colour-scale.mjs";
import { readOverlay, styleOverlay } from "../src/overlays.mjs";
import {
	cellKey,
	plotPoints,
	pointInPolygon,
} from "../src/embedding-model.mjs";
import {
	newSession,
	exportWorkspace,
	importWorkspace,
	snapshot,
} from "../src/workspace.mjs";

test("continuous colours clip at editable limits, reverse palettes and preserve zeros", () => {
	const scale = continuousScale([0, 1, 10], {
		vmin: 0,
		vmax: 2,
		colourmap: "magma",
	});
	assert.deepEqual(scale.colour(2), scale.colour(10));
	assert.notDeepEqual(scale.colour(0), scale.colour(null));
	const reverse = continuousScale([0, 1, 10], {
		vmin: 0,
		vmax: 2,
		colourmap: "magma",
		reverse: true,
	});
	assert.deepEqual(scale.colour(0), reverse.colour(2));
	assert.equal(
		continuousScale([0, 10], { vmax: 2, scale: "log1p" }).high,
		Math.log1p(2),
	);
	assert.throws(
		() => continuousScale([0, 10], { vmin: 5, vmax: 2 }),
		/greater/,
	);
	assert.throws(
		() => continuousScale([0, 10], { colourmap: "missing" }),
		/Unknown/,
	);
	assert.doesNotThrow(() => continuousScale([0, 0]));
});
test("recolouring cached expression for a drag never rereads the matrix", async () => {
	let reads = 0;
	const dataset = {
		tables: {
			t: {
				getTableKeys: () => ({ regionKey: "region" }),
				loadObsColumns: async () => [
					["1", "2"],
					["A", "A"],
				],
				loadExpression: async () => {
					reads++;
					return [10, 60];
				},
			},
		},
		shapes: { A: { loadFeatureIds: async () => ["2", "1"] } },
	};
	const binding = {
		table: "t",
		idColumn: "id",
		source: "gene",
		field: "G",
		scale: "linear",
	};
	const data = await readOverlay(
		dataset,
		{ source: { elementKey: "A" } },
		binding,
	);
	for (let vmax = 12; vmax < 60; vmax++) {
		const result = styleOverlay(data, {
			...binding,
			vmax,
			colourmap: "plasma",
		});
		assert.equal(result.high, vmax);
		assert.equal(result.rows.get("1").value, 10);
	}
	assert.equal(reads, 1);
});
test("embedding points retain exact region IDs and original table rows, with nonfinite coordinates omitted", () => {
	const data = {
		rows: [0, 1, 2],
		xy: [0, 0, 1, 2, NaN, 2],
		ids: ["3813280800012100001", "3813280800012100001", "3"],
		regions: ["shapes/A", "B", "A"],
	};
	const result = plotPoints(data, { region: "A", context: true, limit: 10000 });
	assert.equal(result.current, 1);
	assert.equal(result.others, 1);
	assert.equal(result.invalid, 1);
	assert.equal(result.points[0].row, 1);
	assert.equal(result.points[1].row, 0);
	assert.notEqual(cellKey("A", data.ids[0]), cellKey("B", data.ids[1]));
	assert.equal(cellKey("shapes/A", "2"), cellKey("A", "2"));
	assert.equal(
		pointInPolygon(1, 1, [
			[0, 0],
			[2, 0],
			[2, 2],
			[0, 2],
		]),
		true,
	);
	assert.equal(
		pointInPolygon(3, 1, [
			[0, 0],
			[2, 0],
			[2, 2],
			[0, 2],
		]),
		false,
	);
});
test("embedding settings migrate v3 and round-trip through workspaces and bookmarks", () => {
	const dataset = {
		coordinateSystems: ["a"],
		images: { img: { coordinateSystems: ["a"] } },
		tables: {},
	};
	const session = newSession(dataset, "a");
	session.embedding = {
		...session.embedding,
		projection: "X_umap",
		vmax: 5,
		colourmap: "magma",
	};
	session.bookmarks = [
		{
			id: "b1",
			name: "Here",
			comment: "",
			createdAt: "now",
			viewer: snapshot(session),
		},
	];
	const saved = exportWorkspace(dataset, "local", session);
	assert.deepEqual(
		importWorkspace(saved, dataset).embedding,
		session.embedding,
	);
	const old = structuredClone(saved);
	old.version = 3;
	delete old.embedding;
	delete old.bookmarks[0].viewer.embedding;
	assert.equal(importWorkspace(old, dataset).embedding.colourBy, "view");
	saved.embedding.vmax = Infinity;
	assert.throws(() => importWorkspace(saved, dataset), /finite/);
});
