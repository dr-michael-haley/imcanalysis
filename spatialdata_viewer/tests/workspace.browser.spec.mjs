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
const fixtures = join(appRoot, "fixtures");
const fixture = JSON.parse(
	readFileSync(join(fixtures, "fixture.json"), "utf8"),
);
async function open(page) {
	await page.goto("/");
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(fixtures, fixture.folder));
	await expect(page.getByLabel("Marker for channel 1")).toBeVisible();
	await ready(page);
}
async function exported(page) {
	await page.getByRole("button", { name: "Workspace", exact: true }).click();
	const wait = page.waitForEvent("download");
	await page
		.getByRole("button", { name: "Export workspace", exact: true })
		.click();
	return JSON.parse(readFileSync(await (await wait).path()));
}
async function imported(page, value) {
	await page.getByLabel("View settings", { exact: true }).setInputFiles({
		name: "workspace.json",
		mimeType: "application/json",
		buffer: Buffer.from(JSON.stringify(value)),
	});
}
test("four views, linked navigation, bookmarks and local resume preserve independent layer settings", async ({
	page,
}) => {
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	await open(page);
	await page.getByLabel("Max for channel 1").fill("180");
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page.getByRole("button", { name: "Add view", exact: true }).click();
	await page.getByLabel("Max for channel 1").fill("90");
	await page.getByRole("button", { name: "Add view", exact: true }).click();
	await page.getByRole("button", { name: "Add view", exact: true }).click();
	await expect(page.locator(".viewer-pane canvas")).toHaveCount(4);
	await expect(
		page.getByRole("button", { name: "Add view", exact: true }),
	).toBeDisabled();
	await page
		.getByRole("button", { name: "Select View 1", exact: true })
		.click();
	await expect(page.getByLabel("Max for channel 1")).toHaveValue("180");
	await page
		.getByRole("button", { name: "Select View 2", exact: true })
		.click();
	await expect(page.getByLabel("Max for channel 1")).toHaveValue("90");
	await toolbarAction(page, "Zoom in");
	let state = await exported(page);
	expect(
		state.panes.every(
			(p) =>
				JSON.stringify(p.viewState) ===
				JSON.stringify(state.panes[0].viewState),
		),
	).toBe(true);
	const zoom = state.panes[0].viewState.zoom;
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page.getByLabel("Link navigation").uncheck();
	await toolbarAction(page, "Zoom in");
	state = await exported(page);
	expect(state.panes[0].viewState.zoom).toBe(zoom);
	expect(state.panes[1].viewState.zoom).toBe(zoom + 0.5);
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page.getByLabel("Link navigation").check();
	await page.getByRole("button", { name: "Bookmarks", exact: true }).click();
	await page.getByRole("button", { name: "Bookmark current viewer" }).click();
	await page.getByLabel("Bookmark name", { exact: true }).fill("Comparison");
	await page.getByLabel("Bookmark name", { exact: true }).press("Tab");
	await page
		.getByLabel("Comments for Comparison")
		.fill("Compare segmentation and gene expression");
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page
		.getByRole("button", { name: "Remove View 4", exact: true })
		.click();
	await page.getByLabel("Max for channel 1").fill("42");
	await toolbarAction(page, "Zoom out");
	await page.getByRole("button", { name: "Bookmarks", exact: true }).click();
	await page
		.getByRole("button", { name: "Return to bookmark Comparison" })
		.click();
	await expect(page.locator(".viewer-pane canvas")).toHaveCount(4);
	await expect(page.getByLabel("Max for channel 1")).toHaveValue("90");
	const saved = await exported(page);
	expect(saved.panes[1].viewState.zoom).toBe(zoom + 0.5);
	await page.getByRole("button", { name: "Save locally now" }).click();
	await expect(page.locator(".drawer-bar")).toContainText("Saved locally");
	await page.reload();
	await expect(page.locator(".local-resume")).toContainText(fixture.folder);
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(fixtures, fixture.folder));
	await expect(page.locator(".viewer-pane canvas")).toHaveCount(4);
	await expect(page.getByLabel("Max for channel 1")).toHaveValue("90");
	const restored = await exported(page);
	expect(restored.panes).toEqual(saved.panes);
	expect(restored.bookmarks).toEqual(saved.bookmarks);
	// Real drag events in any pane update every linked camera.
	await page.getByRole("button", { name: "Hide drawer", exact: true }).click();
	const canvas = await page.locator(".viewer-pane canvas").nth(2).boundingBox();
	await page.mouse.move(
		canvas.x + canvas.width / 2,
		canvas.y + canvas.height / 2,
	);
	await page.mouse.down();
	await page.mouse.move(
		canvas.x + canvas.width / 2 + 50,
		canvas.y + canvas.height / 2 + 25,
		{ steps: 8 },
	);
	await page.mouse.up();
	const moved = await exported(page);
	expect(moved.panes[0].viewState.target).not.toEqual(
		saved.panes[0].viewState.target,
	);
	expect(
		moved.panes.every(
			(p) =>
				JSON.stringify(p.viewState) ===
				JSON.stringify(moved.panes[0].viewState),
		),
	).toBe(true);
	await page.getByRole("button", { name: "Bookmarks", exact: true }).click();
	await page.screenshot({
		path: join(appRoot, "test-results/four-views-bookmarks.png"),
	});
	expect(errors).toEqual([]);
});
test("CSV gene lists drive dense and sparse sums, transcript filters and portable workspaces", async ({
	page,
}) => {
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	await open(page);
	await page.getByRole("button", { name: "Gene lists", exact: true }).click();
	await page.getByLabel("Gene list CSV", { exact: true }).setInputFiles({
		name: "panels.csv",
		mimeType: "text/csv",
		buffer: Buffer.from(
			"list,gene\nExpression,GeneA\nExpression,GeneB\nExpression,GeneA\nExpression,Absent\nTranscripts,Cd3\nTranscripts,Cd68\nMissing,NotInX1\nMissing,NotInX2",
		),
	});
	await expect(page.locator(".saved-gene-lists")).toContainText(
		"Expression 3 genes",
	);
	await expect(page.locator(".saved-gene-lists")).toContainText(
		"Transcripts 2 genes",
	);
	await selectLayer(page, "Boundaries_A", "shapes");
	await showLayer(page, "Boundaries_A", true);
	await page.getByLabel("Overlay data").selectOption("sum");
	await page
		.getByLabel("Overlay field", { exact: true })
		.selectOption({ label: "Expression" });
	await expect(page.locator(".legend-range")).toHaveText("1272");
	await expect(page.locator(".cell-controls")).toContainText(
		"2 genes summed · Missing from this table: Absent",
	);
	await page.getByLabel("Expression scale").selectOption("log1p");
	await expect(page.locator(".legend-range")).toHaveText("2.5654.29");
	await page
		.getByLabel("Linked cell table")
		.selectOption("expression_alternative");
	await page.getByLabel("Overlay data").selectOption("sum");
	await page
		.getByLabel("Overlay field", { exact: true })
		.selectOption({ label: "Expression" });
	await expect(page.locator(".legend-range")).toHaveText("103618");
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page.getByRole("button", { name: "Add view", exact: true }).click();
	await showLayer(page, "Boundaries_A", false);
	await showLayer(page, "Transcripts_A", true);
	await selectLayer(page, "Transcripts_A", "points");
	await page.getByLabel("Transcript filter").selectOption("list");
	await page
		.getByLabel("Transcript gene list")
		.selectOption({ label: "Transcripts" });
	await expect(page.locator(".transcript-controls")).toContainText(
		"2 / 2 list genes available",
	);
	const value = await exported(page);
	expect(value.geneLists[0].genes).toEqual(["GeneA", "GeneB", "Absent"]);
	expect(value.panes[0].cellBindings["shapes:Boundaries_A"].source).toBe("sum");
	value.panes.forEach((p) => {
		p.viewState = { target: [120, 220, 0], zoom: 1 };
	});
	await imported(page, value);
	await page.getByRole("button", { name: "Hide drawer", exact: true }).click();
	// Pick both genes from the same filtered point layer via GPU.
	for (const x of [80, 150]) {
		const box = await page.locator(".viewer-pane canvas").nth(1).boundingBox();
		await expect
			.poll(
				async () => {
					await page.mouse.click(
						box.x + box.width / 2 + (x - 120) * 2,
						box.y + box.height / 2,
						{ delay: 30 },
					);
					try {
						await expect(
							page
								.locator("dt")
								.filter({ hasText: /^Element$/ })
								.locator("+ dd"),
						).toHaveText("Transcripts_A", { timeout: 1500 });
						return true;
					} catch {
						return false;
					}
				},
				{ timeout: 10000 },
			)
			.toBe(true);
		await page.getByRole("button", { name: "Clear selection" }).click();
	}
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await expect(page.getByLabel("Transcript gene list")).toHaveValue(
		value.geneLists[1].id,
	);
	await page.getByRole("button", { name: "Gene lists", exact: true }).click();
	await page
		.getByRole("button", { name: "Expression 3 genes", exact: true })
		.click();
	await page.getByLabel("Gene list genes").fill("GeneB");
	await page
		.getByRole("button", { name: "Save gene list", exact: true })
		.click();
	await page
		.getByRole("button", { name: "Select View 1", exact: true })
		.click();
	await expect(page.locator(".legend-range")).toHaveText("318");
	await page
		.getByLabel("Overlay field", { exact: true })
		.selectOption({ label: "Missing" });
	await expect(page.locator(".cell-controls .error")).toContainText(
		"None of the 2 genes",
	);
	await imported(page, value);
	await expect(page.locator(".cell-controls .error")).toHaveCount(0);
	await page.getByRole("button", { name: "Gene lists", exact: true }).click();
	await page.screenshot({ path: join(appRoot, "test-results/gene-lists.png") });
	expect(errors).toEqual([]);
});

 test("independent regions survive bookmarks, reload and relinking", async ({ page }) => {
 await open(page);
 await page.getByRole("button", {name:"Views",exact:true}).click();
 await page.getByRole("button", {name:"Add independent view",exact:true}).click();
 await expect(page.getByLabel("View navigation",{exact:true})).toHaveValue("independent");
 const before = await exported(page);
 await page.getByRole("button", {name:"Hide drawer",exact:true}).click();
 const box = await page.locator(".viewer-pane canvas").nth(1).boundingBox();
 await page.mouse.move(box.x + box.width/2, box.y + box.height/2);
 await page.mouse.down();
 await page.mouse.move(box.x + box.width/2 + 65, box.y + box.height/2 + 35,{steps:8});
 await page.mouse.up();
 await toolbarAction(page,"Zoom in");
 const moved = await exported(page);
 expect(moved.panes[0].viewState).toEqual(before.panes[0].viewState);
 expect(moved.panes[1].viewState.target).not.toEqual(before.panes[1].viewState.target);
 expect(moved.panes[1].viewState.zoom).toBe(before.panes[1].viewState.zoom + 0.5);
 await page.getByRole("button",{name:"Bookmarks",exact:true}).click();
 await page.getByRole("button",{name:"Bookmark current viewer"}).click();
 await page.getByLabel("Bookmark name",{exact:true}).fill("Separate regions");
 await page.getByLabel("Bookmark name",{exact:true}).press("Tab");
 await toolbarAction(page,"Zoom in");
 await page.getByRole("button",{name:"Return to bookmark Separate regions"}).click();
 expect((await exported(page)).panes).toEqual(moved.panes);
 await page.getByRole("button",{name:"Save locally now"}).click();
 await expect(page.locator(".drawer-bar")).toContainText("Saved locally");
 await page.reload();
 await page.getByLabel("SpatialData folder",{exact:true}).setInputFiles(join(fixtures,fixture.folder));
 await expect(page.getByLabel("View navigation",{exact:true})).toHaveValue("independent");
 expect((await exported(page)).panes).toEqual(moved.panes);
 await page.screenshot({path:join(appRoot,"test-results/independent-regions.png")});
 await page.getByLabel("View navigation",{exact:true}).selectOption("linked");
 const linked=await exported(page);
 expect(linked.panes[0].viewState).toEqual(moved.panes[1].viewState);
 expect(linked.panes[1].viewState).toEqual(moved.panes[1].viewState);
 });
