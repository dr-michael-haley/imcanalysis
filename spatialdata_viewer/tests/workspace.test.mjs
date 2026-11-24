import test from "node:test";
import assert from "node:assert/strict";
import {
	newSession,
	duplicatePane,
	removePane,
	cameraChange,
	patchPane,
	snapshot,
	restoreBookmark,
	exportWorkspace,
	importWorkspace,
} from "../src/workspace.mjs";
import { importGeneCsv, genesFromText, matchGenes } from "../src/genes.mjs";
import { createWorkspace } from "../src/model.mjs";
import { fileIdentity } from "../src/persistence.mjs";
const dataset = {
	coordinateSystems: ["a"],
	images: { img: { coordinateSystems: ["a"] } },
	tables: {},
};
test("four independent layer configurations with optional linked cameras", () => {
	let s = newSession(dataset, "a");
	s = duplicatePane(s, "v2");
	s.panes[1].renderStack.entries[0].props.opacity = 0.2;
	assert.equal(s.panes[0].renderStack.entries[0].props.opacity, 1);
	const camera = { target: [10, 20, 0], zoom: 2 };
	s = cameraChange(s, "v2", camera);
	assert.deepEqual(
		s.panes.map((p) => p.viewState),
		[camera, camera],
	);
	s = cameraChange({ ...s, linked: false }, "v2", { ...camera, zoom: 3 });
	assert.equal(s.panes[0].viewState.zoom, 2);
	assert.equal(s.panes[1].viewState.zoom, 3);
	s = duplicatePane(duplicatePane(s, "v3"), "v4");
	assert.equal(duplicatePane(s).panes.length, 4);
	s = removePane(s, "v4");
	assert.equal(s.activePaneId, "view-1");
});
test("bookmarks preserve all panes and restore gene definitions used by their overlays", () => {
	let s = duplicatePane(newSession(dataset, "a"), "v2");
	s.geneLists = [{ id: "g1", name: "Signature", genes: ["A", "B"] }];
	const b = {
		id: "b1",
		name: "Region",
		comment: "Cell boundary",
		createdAt: new Date().toISOString(),
		viewer: snapshot(s),
	};
	s.bookmarks.push(b);
	s = patchPane(s, "v2", { name: "Changed" });
	s.geneLists = [
		{ id: "g1", name: "Signature", genes: ["C"] },
		{ id: "g2", name: "Later", genes: ["D"] },
	];
	const restored = restoreBookmark(s, b);
	assert.equal(restored.panes[1].name, "View 2");
	assert.deepEqual(
		restored.geneLists.map((g) => g.genes),
		[["A", "B"], ["D"]],
	);
	assert.equal(restored.bookmarks.length, 1);
	assert.deepEqual(
		importWorkspace(
			JSON.parse(JSON.stringify(exportWorkspace(dataset, "local", restored))),
			dataset,
		),
		restored,
	);
});
test("workspace import migrates old views and rejects invalid cameras, duplicate panes and missing lists", () => {
	const s = newSession(dataset, "a");
	const old = createWorkspace(
		dataset,
		"local",
		"a",
		s.panes[0].renderStack,
		null,
	);
	assert.equal(importWorkspace(old, dataset).panes.length, 1);
	const corrupt = (fn) => {
		const v = exportWorkspace(dataset, "local", structuredClone(s));
		fn(v);
		return v;
	};
	assert.throws(
		() =>
			importWorkspace(
				corrupt((v) => v.panes.push(v.panes[0])),
				dataset,
			),
		/IDs/,
	);
	assert.throws(
		() =>
			importWorkspace(
				corrupt(
					(v) => (v.panes[0].viewState = { target: [0, 0], zoom: Infinity }),
				),
				dataset,
			),
		/camera/,
	);
	assert.throws(
		() =>
			importWorkspace(
				corrupt(
					(v) =>
						(v.panes[0].renderStack.entries[0].props.geneListId = "missing"),
				),
				dataset,
			),
		/missing transcript/,
	);
	assert.throws(
		() =>
			importWorkspace(
				corrupt((v) => (v.inventory = "foreign")),
				dataset,
			),
		/different dataset/,
	);
});
test("CSV imports quoted named groups and deduplicates exact names without changing case", () => {
	assert.deepEqual(
		importGeneCsv(
			'\uFEFFlist,gene\r\n"T, cells",CD3D\r\n"T, cells",CD3E\r\n"T, cells",CD3D\r\nMyeloid,LYZ',
		),
		[
			{ name: "T, cells", genes: ["CD3D", "CD3E"] },
			{ name: "Myeloid", genes: ["LYZ"] },
		],
	);
	assert.deepEqual(importGeneCsv("PDK4\nPDK4\nPDK1", "Metabolism"), [
		{ name: "Metabolism", genes: ["PDK4", "PDK1"] },
	]);
	assert.deepEqual(genesFromText(" A, B;A\n a\tC"), ["A", "B", "a", "C"]);
	assert.deepEqual(matchGenes(["A", "B", "a"], ["A", "B"]), {
		matched: ["A", "B"],
		missing: ["a"],
	});
	assert.throws(() => importGeneCsv("A,B"), /one gene per row/);
	assert.throws(() => importGeneCsv('gene\n"A'), /unclosed/);
});
test("local state identity is order-independent and changes when selected files change", async () => {
	const a = { webkitRelativePath: "store/a", size: 10, lastModified: 20 },
		b = { webkitRelativePath: "store/b", size: 20, lastModified: 30 };
	assert.equal(await fileIdentity([a, b]), await fileIdentity([b, a]));
	assert.notEqual(
		await fileIdentity([a, b]),
		await fileIdentity([a, { ...b, lastModified: 40 }]),
	);
});
