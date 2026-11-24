import {
	selectLayer,
	showLayer,
	toolbarAction,
	ready,
	exportPng,
} from "./ui-helpers.mjs";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
const { test, expect } = engineRequire("@playwright/test");
const fixture = JSON.parse(
	readFileSync(join(appRoot, "fixtures/fixture.json"), "utf8"),
);
async function exported(page) {
	await page.getByRole("button", { name: "Workspace", exact: true }).click();
	const wait = page.waitForEvent("download");
	await page
		.getByRole("button", { name: "Export workspace", exact: true })
		.click();
	return JSON.parse(readFileSync(await (await wait).path()));
}
const cellBinding = (state, i) =>
	state.panes[i].cellBindings["shapes:Boundaries_A"];
const pointProps = (state, i) =>
	state.panes[i].renderStack.entries.find(
		(e) => e.id === "points:Transcripts_A",
	).props;
test("visible layers share genes and obs across four views, skip incompatible targets and restore copied settings", async ({
	page,
}) => {
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	await page.goto("/");
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(appRoot, "fixtures", fixture.folder));
	await ready(page);
	const initial = await exported(page);
	const original = initial.panes[0];
	initial.panes = Array.from({ length: 4 }, (_, i) => {
		const p = structuredClone(original);
		p.id = `view-${i + 1}`;
		p.name = `View ${i + 1}`;
		p.selectedId = "shapes:Boundaries_A";
		p.renderStack.entries.forEach((e) => {
			e.visible =
				i === 3
					? e.source.elementType === "image"
					: ["shapes:Boundaries_A", "points:Transcripts_A"].includes(e.id);
		});
		p.cellBindings = {
			"shapes:Boundaries_A": {
				table:
					i === 1 ? "expression_alternative" : i === 2 ? "cells" : "expression",
				idColumn: "instance_id",
				source: "gene",
				field: i === 2 ? "DNA" : i === 0 ? "Cd3" : "GeneB",
				scale: "linear",
				colourmap: i === 0 ? "magma" : "viridis",
				vmin: 0,
				vmax: 4,
			},
		};
		return p;
	});
	initial.activePaneId = "view-1";
	await page.getByLabel("View settings", { exact: true }).setInputFiles({
		name: "comparison.json",
		mimeType: "application/json",
		buffer: Buffer.from(JSON.stringify(initial)),
	});
	await expect(page.locator(".viewer-pane canvas")).toHaveCount(4);
	const report = page.getByLabel("Apply across views", { exact: true });
	await page
		.getByRole("button", { name: "Apply to cell layers (2)", exact: true })
		.click();
	await expect(report).toContainText("Applied to 1 cell layer · 1 skipped");
	await expect(report).toContainText("View 3 · Boundaries_A");
	await expect(report).toContainText("None of the selected genes");
	let state = await exported(page);
	expect(cellBinding(state, 1)).toEqual({
		...cellBinding(initial, 1),
		field: "Cd3",
		colourmap: "magma",
		reverse: false,
	});
	expect(cellBinding(state, 2)).toEqual(cellBinding(initial, 2));
	expect(state.panes[3].cellBindings).toEqual(initial.panes[3].cellBindings);
	expect(state.panes[3].renderStack).toEqual(initial.panes[3].renderStack);
	await page
		.getByRole("button", {
			name: "Apply genes to Points layers (3)",
			exact: true,
		})
		.click();
	await expect(report).toContainText("Applied to 3 Points layers");
	state = await exported(page);
	for (let i = 0; i < 3; i++)
		expect(pointProps(state, i).featureNames).toEqual(["Cd3"]);
	expect(pointProps(state, 3)).toEqual(pointProps(initial, 3));

	await page.getByLabel("Overlay data").selectOption("obs");
	await page.getByLabel("Overlay field", { exact: true }).selectOption("score");
	await expect(
		page.getByRole("button", {
			name: "Apply genes to Points layers (3)",
			exact: true,
		}),
	).toBeDisabled();
	await page
		.getByRole("button", { name: "Apply to cell layers (2)", exact: true })
		.click();
	await expect(report).toContainText(
		"obs “score” is missing from expression_alternative",
	);
	state = await exported(page);
	expect(cellBinding(state, 1).field).toBe("Cd3");
	expect(cellBinding(state, 2).field).toBe("score");
	await page
		.getByLabel("Overlay field", { exact: true })
		.selectOption("cell_type");
	await page
		.getByRole("button", { name: "Apply to cell layers (2)", exact: true })
		.click();
	await expect(report).toContainText("Applied to 2 cell layers");
	await selectLayer(page, "Transcripts_A", "points");
	await page.getByLabel("Transcript gene Cd68", { exact: true }).check();
	await page
		.getByRole("button", {
			name: "Apply genes to Points layers (2)",
			exact: true,
		})
		.click();
	await expect(report).toContainText("Applied to 2 Points layers");
	await page
		.getByRole("button", {
			name: "Apply genes to cell layers (3)",
			exact: true,
		})
		.click();
	await expect(report).toContainText("Applied to 2 cell layers · 1 skipped");
	await expect(report).toContainText(
		"Workspace gene list: Points · Transcripts_A",
	);
	state = await exported(page);
	expect(state.geneLists).toHaveLength(1);
	expect(state.geneLists[0].genes).toEqual(["Cd3", "Cd68"]);
	for (let i = 0; i < 2; i++) {
		expect(cellBinding(state, i).field).toBe(state.geneLists[0].id);
		expect(cellBinding(state, i).source).toBe("sum");
		expect(cellBinding(state, i).vmax).toBeNull();
		expect(cellBinding(state, i).colourmap).toBe("magma");
	}
	expect(cellBinding(state, 1).table).toBe("expression_alternative");
	expect(cellBinding(state, 2).field).toBe("cell_type");
	await page
		.getByRole("button", {
			name: "Apply genes to cell layers (3)",
			exact: true,
		})
		.click();
	await expect(report).toContainText("Applied to 2 cell layers");
	expect((await exported(page)).geneLists).toHaveLength(1);
	await page.getByLabel("Transcript filter").selectOption("all");
	await expect(
		page.getByRole("button", {
			name: "Apply genes to cell layers (3)",
			exact: true,
		}),
	).toBeDisabled();
	await page
		.getByRole("button", {
			name: "Apply genes to Points layers (2)",
			exact: true,
		})
		.click();
	await expect(report).toContainText("Applied to 2 Points layers");
	state = await exported(page);
	expect(pointProps(state, 1).featureNames).toBeUndefined();
	await selectLayer(page, "Boundaries_A", "shapes");
	await page
		.getByRole("button", {
			name: "Apply genes to Points layers (3)",
			exact: true,
		})
		.click();
	await expect(report).toContainText("Applied to 3 Points layers");
	await report.scrollIntoViewIfNeeded();
	await page.screenshot({
		path: join(appRoot, "test-results/propagation.png"),
	});
	state = await exported(page);
	for (let i = 0; i < 3; i++)
		expect(pointProps(state, i).geneListId).toBe(state.geneLists[0].id);
	expect(state.panes[3].cellBindings).toEqual(initial.panes[3].cellBindings);
	await page.getByRole("button", { name: "Save locally now" }).click();
	await expect(page.locator(".drawer-bar")).toContainText("Saved locally");
	await page.reload();
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(appRoot, "fixtures", fixture.folder));
	await expect(page.locator(".viewer-pane canvas")).toHaveCount(4);
	const restored = await exported(page);
	expect(restored.geneLists).toEqual(state.geneLists);
	for (let i = 0; i < 4; i++) {
		expect(cellBinding(restored, i)).toEqual(cellBinding(state, i));
		expect(pointProps(restored, i)).toEqual(pointProps(state, i));
	}
	expect(errors).toEqual([]);
	await expect(page.locator('.viewport [role="alert"]')).toHaveCount(0);
});
