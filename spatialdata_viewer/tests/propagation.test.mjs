import test from "node:test";
import assert from "node:assert/strict";
import { applyPropagation, preparePropagation } from "../src/propagation.mjs";

const shape = (key, visible = true) => ({
	id: `shapes:${key}`,
	source: { elementType: "shapes", elementKey: key },
	visible,
	props: { opacity: 0.4 },
});
const points = (key, props = {}, visible = true) => ({
	id: `points:${key}`,
	source: { elementType: "points", elementKey: key },
	visible,
	props: { pointSize: 3, ...props },
});
const binding = (table, field = "A") => ({
	table,
	idColumn: `${table}_id`,
	source: "gene",
	field,
	scale: "linear",
	colourmap: "magma",
	vmax: 10,
});
function fixture() {
	let reads = 0;
	const table = (genes, obs = ["type"]) => ({
		loadVarIndex: async () => {
			reads++;
			return genes;
		},
		getObsColumnNames: () => obs,
		getTableKeys: () => ({ instanceKey: "id", regionKey: "region" }),
	});
	const dataset = {
		tables: {
			full: table(["A", "B"], ["type", "score"]),
			other: table(["B", "A"]),
			partial: table(["A"]),
			absent: table(["C"]),
			duplicate: table(["A", "A"]),
		},
		points: {
			transcripts: {
				listFeatures: async () => ({ entries: [{ name: "A" }, { name: "B" }] }),
			},
			partial: { listFeatures: async () => ({ entries: [{ name: "A" }] }) },
			broken: {
				listFeatures: async () => {
					throw new Error("Unreadable feature catalog");
				},
			},
		},
		getAssociatedTables: () => [["full", table(["A", "B"])]],
	};
	const session = {
		coordinateSystem: "global",
		geneLists: [{ id: "list", name: "Panel", genes: ["A", "B"] }],
		panes: [
			{
				id: "one",
				name: "View 1",
				renderStack: {
					entries: [
						shape("first"),
						points("transcripts", { featureNames: ["B", "A"] }),
					],
				},
				cellBindings: { "shapes:first": binding("full") },
			},
			{
				id: "two",
				name: "View 2",
				renderStack: {
					entries: [
						shape("second"),
						shape("hidden", false),
						shape("default"),
						points("transcripts"),
						points("partial"),
						points("broken"),
						points("hidden", {}, false),
					],
				},
				cellBindings: {
					"shapes:second": binding("other", "B"),
					"shapes:hidden": binding("absent", "C"),
				},
			},
		],
	};
	return { dataset, session, reads: () => reads };
}

test("cell propagation copies display settings while preserving joins, geometry and hidden layers", async () => {
	const { dataset, session } = fixture();
	const before = structuredClone(session);
	const plan = await preparePropagation(
		dataset,
		session,
		"one",
		"shapes:first",
		"cells",
	);
	const result = applyPropagation(session, plan);
	assert.equal(plan.outcomes.length, 2);
	assert.deepEqual(result.panes[0], before.panes[0]);
	assert.deepEqual(result.panes[1].cellBindings["shapes:second"], {
		...binding("other"),
		field: "A",
		reverse: false,
		vmin: null,
	});
	assert.equal(result.panes[1].cellBindings["shapes:default"].idColumn, "id");
	assert.equal(result.panes[1].cellBindings["shapes:default"].table, "full");
	assert.deepEqual(
		result.panes[1].cellBindings["shapes:hidden"],
		before.panes[1].cellBindings["shapes:hidden"],
	);
	assert.equal(result.panes[1].renderStack, session.panes[1].renderStack);
	assert.deepEqual(session, before);
});

test("missing obs skips just the incompatible target and never replaces its working choice", async () => {
	const { dataset, session } = fixture();
	session.panes[0].cellBindings["shapes:first"] = {
		...binding("full"),
		source: "obs",
		field: "score",
	};
	const plan = await preparePropagation(
		dataset,
		session,
		"one",
		"shapes:first",
		"cells",
	);
	assert.match(plan.outcomes[0].reason, /score.*missing from other/);
	const result = applyPropagation(session, plan);
	assert.equal(
		result.panes[1].cellBindings["shapes:second"],
		session.panes[1].cellBindings["shapes:second"],
	);
	assert.equal(result.panes[1].cellBindings["shapes:default"].field, "score");
	await assert.rejects(
		preparePropagation(dataset, session, "one", "shapes:first", "points"),
		/Obs fields cannot/,
	);
});

test("gene lists reach Points, keep list references, and report partial and unreadable targets", async () => {
	const { dataset, session } = fixture();
	session.panes[0].cellBindings["shapes:first"] = {
		...binding("full"),
		source: "sum",
		field: "list",
	};
	const plan = await preparePropagation(
		dataset,
		session,
		"one",
		"shapes:first",
		"points",
	);
	const result = applyPropagation(session, plan);
	assert.deepEqual(
		plan.outcomes.find((r) => r.entryId === "points:partial").missing,
		["B"],
	);
	assert.match(
		plan.outcomes.find((r) => r.entryId === "points:broken").reason,
		/Unreadable/,
	);
	assert.equal(
		result.panes[1].renderStack.entries.find(
			(e) => e.id === "points:transcripts",
		).props.geneListId,
		"list",
	);
	assert.equal(
		result.panes[1].renderStack.entries.find(
			(e) => e.id === "points:transcripts",
		).props.pointSize,
		3,
	);
	assert.equal(
		result.panes[1].renderStack.entries.find((e) => e.id === "points:hidden")
			.props.geneListId,
		undefined,
	);
});

test("selected Points genes use one shared sum list, reuse equivalent lists and retain target colour maps", async () => {
	const { dataset, session, reads } = fixture();
	const plan = await preparePropagation(
		dataset,
		session,
		"one",
		"points:transcripts",
		"cells",
	);
	assert.equal(reads(), 2); // One metadata read per distinct table, across views.
	assert.equal(plan.newList, undefined);
	assert.equal(plan.patch.field, "list");
	const result = applyPropagation(session, plan);
	assert.equal(result.panes[0].cellBindings["shapes:first"].source, "sum");
	assert.equal(result.panes[0].cellBindings["shapes:first"].colourmap, "magma");
	assert.equal(result.panes[0].cellBindings["shapes:first"].vmax, null);
	session.geneLists = [];
	const created = await preparePropagation(
		dataset,
		session,
		"one",
		"points:transcripts",
		"cells",
	);
	assert.deepEqual(created.newList.genes, ["B", "A"]);
	assert.equal(applyPropagation(session, created).geneLists.length, 1);
});

test("Point-to-Point choices preserve rendering; all-genes never becomes an unintended cell sum", async () => {
	const { dataset, session } = fixture();
	const plan = await preparePropagation(
		dataset,
		session,
		"one",
		"points:transcripts",
		"points",
	);
	const result = applyPropagation(session, plan);
	assert.deepEqual(
		result.panes[1].renderStack.entries.find(
			(e) => e.id === "points:transcripts",
		).props.featureNames,
		["B", "A"],
	);
	delete session.panes[0].renderStack.entries[1].props.featureNames;
	await assert.rejects(
		preparePropagation(dataset, session, "one", "points:transcripts", "cells"),
		/Choose selected genes/,
	);
	const all = await preparePropagation(
		dataset,
		session,
		"one",
		"points:transcripts",
		"points",
	);
	assert.equal(all.patch.featureNames, undefined);
	assert.equal(all.patch.geneListId, undefined);
});

test("absent and duplicate genes are skipped; a partial sum is explicitly reported", async () => {
	const { dataset, session } = fixture();
	session.panes[1].renderStack.entries = [
		shape("absent"),
		shape("duplicate"),
		shape("partial"),
	];
	session.panes[1].cellBindings = Object.fromEntries(
		["absent", "duplicate", "partial"].map((k) => [`shapes:${k}`, binding(k)]),
	);
	const plan = await preparePropagation(
		dataset,
		session,
		"one",
		"points:transcripts",
		"cells",
	);
	assert.match(
		plan.outcomes.find((r) => r.entryId === "shapes:absent").reason,
		/None/,
	);
	assert.match(
		plan.outcomes.find((r) => r.entryId === "shapes:duplicate").reason,
		/duplicate/,
	);
	assert.deepEqual(
		plan.outcomes.find((r) => r.entryId === "shapes:partial").missing,
		["B"],
	);
	const noX = await preparePropagation(
		dataset,
		session,
		"one",
		"points:transcripts",
		"cells",
		async () => {
			throw new Error("No X matrix");
		},
	);
	assert.ok(noX.outcomes.every((r) => !r.applied));
	assert.equal(noX.newList, undefined);
	assert.equal(applyPropagation(session, noX), session);
});

test("pending copies tolerate navigation but reject changed bindings, visibility or gene definitions", async () => {
	const { dataset, session } = fixture();
	const plan = await preparePropagation(
		dataset,
		session,
		"one",
		"shapes:first",
		"cells",
	);
	const navigated = structuredClone(session);
	navigated.panes[0].viewState = { target: [1, 2, 0], zoom: 3 };
	navigated.activePaneId = "two";
	assert.equal(applyPropagation(navigated, plan).panes[0].viewState.zoom, 3);
	for (const edit of [
		(s) => {
			s.panes[1].cellBindings["shapes:second"].field = "C";
		},
		(s) => {
			s.panes[1].renderStack.entries[0].visible = false;
		},
		(s) => {
			s.geneLists[0].genes.push("C");
		},
	]) {
		const changed = structuredClone(session);
		edit(changed);
		assert.throws(
			() => applyPropagation(changed, plan),
			/changed while checking/,
		);
	}
});
