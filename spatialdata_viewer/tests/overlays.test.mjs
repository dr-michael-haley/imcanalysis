import test from "node:test";
import assert from "node:assert/strict";
import { joinRows, encodeOverlay, loadOverlay } from "../src/overlays.mjs";

test("gene lists sum stored X before the display transform and report missing genes", async () => {
	const read = [];
	const table = {
		getTableKeys: () => ({ regionKey: "region" }),
		loadObsColumns: async () => [
			["b", "a"],
			["A", "A"],
		],
		loadVarIndex: async () => ["G1", "G2"],
		loadExpression: async (g) => {
			read.push(g);
			return g === "G1" ? [3, 0] : [5, 2];
		},
	};
	const data = {
		tables: { t: table },
		shapes: { A: { loadFeatureIds: async () => ["a", "b"] } },
	};
	const entry = { source: { elementKey: "A" } },
		binding = {
			table: "t",
			idColumn: "id",
			source: "sum",
			field: "list",
			scale: "log1p",
		};
	const result = await loadOverlay(data, entry, binding, undefined, [
		{ id: "list", name: "Signature", genes: ["G1", "Missing", "G2"] },
	]);
	assert.deepEqual(read, ["G1", "G2"]);
	assert.equal(result.rows.get("b").value, 8);
	assert.equal(result.rows.get("a").value, 2);
	assert.equal(result.high, Math.log1p(8));
	assert.deepEqual(result.geneSummary.missing, ["Missing"]);
	await assert.rejects(
		loadOverlay(data, entry, binding, undefined, [
			{ id: "list", name: "Absent", genes: ["Missing"] },
		]),
		/None of/,
	);
	const controller = new AbortController();
	controller.abort();
	await assert.rejects(
		loadOverlay(data, entry, binding, controller.signal, []),
		/abort/i,
	);
});

test("joins shuffled rows within their region and keeps large cell IDs exact", () => {
	const rows = joinRows(
		["3813280800012100002", "3813280800012100001", "3813280800012100001"],
		["A", "B", "A"],
		"A",
		[12, 99, 4],
	);
	const ids = ["3813280800012100001", "3813280800012100002", "unpaired"];
	assert.equal(rows.get(ids[0]).row, 2);
	const result = encodeOverlay(rows, ids, "continuous");
	assert.equal(result.matched, 2);
	assert.equal(result.missing, 1);
	assert.equal(result.low, 4);
	assert.equal(result.high, 12);
	assert.notDeepEqual(
		[...result.buffer.colors.slice(0, 4)],
		[...result.buffer.colors.slice(4, 8)],
	);
	assert.throws(
		() => joinRows(["1", "1"], ["A", "A"], "A", [2, 4]),
		/Duplicate/,
	);
});
test("categorical numeric labels remain categories and missing values are distinct from zero", () => {
	const rows = joinRows(["a", "b", "c"], ["A", "A", "A"], "A", [0, 2, null]);
	const result = encodeOverlay(rows, ["a", "b", "c"], "categorical");
	assert.deepEqual(
		result.categories.map((v) => v.label),
		["0", "2"],
	);
	assert.equal(result.missing, 1);
	assert.throws(
		() =>
			encodeOverlay(
				new Map([["a", { value: -1 }]]),
				["a"],
				"continuous",
				"log1p",
			),
		/non-negative/,
	);
});
test("unmatched IDs never acquire expression by row position", async () => {
	const table = {
		getTableKeys: () => ({ regionKey: "region" }),
		loadObsColumns: async () => [
			["wrong1", "wrong2"],
			["A", "A"],
		],
		loadExpression: async () => [20, 40],
	};
	const result = await loadOverlay(
		{
			tables: { t: table },
			shapes: { A: { loadFeatureIds: async () => ["real1", "real2"] } },
		},
		{ source: { elementKey: "A" } },
		{ table: "t", idColumn: "id", source: "gene", field: "G", scale: "linear" },
	);
	assert.equal(result.matched, 0);
	assert.equal(result.resolve(["real1", "real2"]), undefined);
});
