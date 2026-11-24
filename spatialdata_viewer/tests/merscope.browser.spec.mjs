import {
	selectLayer,
	showLayer,
	toolbarAction,
	ready,
	exportPng,
} from "./ui-helpers.mjs";
import { join } from "node:path";
import { readFileSync } from "node:fs";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
const { test, expect } = engineRequire("@playwright/test");
const dataset = process.env.SBT_MERSCOPE_DATASET;

async function frameAndPick(page, target, expectedElement) {
	await page.getByRole("button", { name: "Views", exact: true }).click();
	const saved = page.waitForEvent("download");
	await page.getByRole("button", { name: "Export view settings" }).click();
	const value = JSON.parse(readFileSync(await (await saved).path()));
	value.viewState = { target, zoom: 2 };
	for (const pane of value.panes) pane.viewState = value.viewState;
	await page.getByLabel("View settings", { exact: true }).setInputFiles({
		name: "camera.json",
		mimeType: "application/json",
		buffer: Buffer.from(JSON.stringify(value)),
	});
	const box = await page.locator(".viewport canvas").boundingBox();
	await expect
		.poll(
			async () => {
				const picked = page
					.locator("dt")
					.filter({ hasText: /^Element$/ })
					.locator("+ dd");
				if ((await picked.allTextContents())[0] === expectedElement)
					return true;
				await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2, {
					delay: 30,
				});
				try {
					await expect(
						page
							.locator("dt")
							.filter({ hasText: /^Element$/ })
							.locator("+ dd"),
					).toHaveText(expectedElement, { timeout: 2500 });
					return true;
				} catch {
					return false;
				}
			},
			{ timeout: 15000, intervals: [500] },
		)
		.toBe(true);
}

test("MERSCOPE real-data integration", async ({ page }) => {
	test.skip(!dataset, "Set SBT_MERSCOPE_DATASET to a local SpatialData store.");
	test.setTimeout(180000);
	const errors = [],
		network = [];
	page.on("pageerror", (e) => errors.push(e.message));
	page.on("request", (r) => network.push({ method: r.method(), url: r.url() }));
	page.on("console", (e) => {
		if (e.type() === "error" || /preload failed|unreachable/.test(e.text()))
			errors.push(e.text());
	});
	await page.goto("/");
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(dataset);
	await ready(page);
	for (const shape of ["cellpose4_boundaries", "comseg_boundaries_cellpose4"]) {
		await selectLayer(page, shape, "shapes");
		await showLayer(page, shape, true);
		await expect(page.locator(".cell-controls")).toContainText("Unpaired:", {
			timeout: 30000,
		});
		await showLayer(page, shape, false);
	}
	for (const [shape, table, count] of [
		[
			"202604240712_AbedV2slide6repeat24042026_VMSC20702_region_Dura2_polygons",
			"table",
			16533,
		],
		["comseg_boundaries_denovo", "table_comseg_denovo", 6807],
		["comseg_boundaries_merscope", "table_comseg_merscope", 3748],
		["proseg_boundaries_cellpose4", "table_proseg_cellpose4", 3920],
		["proseg_boundaries_merscope", "table_proseg_merscope", 10758],
	]) {
		await selectLayer(page, shape, "shapes");
		await showLayer(page, shape, true);
		await page.getByLabel("Overlay data").selectOption("gene");
		await page
			.getByLabel("Overlay field", { exact: true })
			.selectOption("PDK4");
		await expect(page.getByLabel("Overlay legend")).toContainText("PDK4", {
			timeout: 30000,
		});
		await expect(page.getByLabel("Linked cell table")).toHaveValue(table);
		await expect(page.locator(".link-status")).toContainText(
			`${count.toLocaleString()} / ${count.toLocaleString()} boundaries linked`,
		);
		// Independently checked against Python/Zarr reads of this development dataset.
		await expect(page.locator(".legend-range")).toHaveText("04");
		if (shape === "proseg_boundaries_merscope") {
			await frameAndPick(
				page,
				[23216.622302201577, 8416.267314329743, 0],
				shape,
			);
			await expect(
				page
					.locator("dt")
					.filter({ hasText: /^Matched row$/ })
					.locator("+ dd"),
			).toHaveText("5943");
			await expect(
				page
					.locator("dt")
					.filter({ hasText: /^PDK4 · X$/ })
					.locator("+ dd"),
			).toHaveText("4");
			await page.getByRole("button", { name: "Layers", exact: true }).click();
			await selectLayer(page, shape, "shapes");
		}
		await page.getByLabel("Overlay data").selectOption("obs");
		await page
			.getByLabel("Overlay field", { exact: true })
			.selectOption("cell_type");
		await expect(page.getByLabel("Overlay legend")).toContainText("cell_type", {
			timeout: 30000,
		});
		if (shape === "proseg_boundaries_merscope") {
			await page.getByLabel("Overlay legend").scrollIntoViewIfNeeded();
			await expect(page.getByText(/^Refreshing layer metadata/)).toHaveCount(0);
			await page.screenshot({
				path: join(appRoot, "test-results/merscope-categories.png"),
			});
		}
		await showLayer(page, shape, false);
	}
	await selectLayer(
		page,
		"202604240712_AbedV2slide6repeat24042026_VMSC20702_region_Dura2_transcripts",
		"points",
	);
	await expect(page.locator(".transcript-controls")).toContainText(
		"transcript genes available",
		{ timeout: 30000 },
	);
	await expect(page.locator(".transcript-controls")).toContainText(
		"278,071 transcripts in this layer",
	);
	await page.getByLabel("Transcript filter").selectOption("selected");
	await page.getByLabel("Transcript gene PDK4", { exact: true }).check();
	await page.getByRole("button", { name: "Views", exact: true }).click();
	const saved = page.waitForEvent("download");
	await page.getByRole("button", { name: "Export view settings" }).click();
	const savedPath = await (await saved).path();
	await page
		.getByLabel("View settings", { exact: true })
		.setInputFiles(savedPath);
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await selectLayer(
		page,
		"202604240712_AbedV2slide6repeat24042026_VMSC20702_region_Dura2_transcripts",
		"points",
	);
	await expect(
		page.getByLabel("Transcript gene PDK4", { exact: true }),
	).toBeChecked();
	await frameAndPick(
		page,
		[39206.6009352915, 4364.348569050431, 0],
		"202604240712_AbedV2slide6repeat24042026_VMSC20702_region_Dura2_transcripts",
	);
	await expect(page.locator(".feature-data")).toContainText("PDK4");
	await page.getByRole("button", { name: "Embeddings", exact: true }).click();
	await page.getByLabel("Embedding table").selectOption("table");
	await expect(page.getByLabel("Embedding projection")).toHaveValue("spatial");
	await page.getByLabel("Embedding colour by").selectOption("gene");
	await page.getByLabel("Embedding colour field").selectOption("PDK4");
	await expect(page.locator(".embedding-status")).toContainText(
		"16,533 cells in current region",
		{ timeout: 30000 },
	);
	await expect(page.locator(".embedding-settings .legend-range")).toHaveText(
		"04",
	);
	await page
		.getByLabel("Embedding Colour map", { exact: true })
		.selectOption("magma");
	await page.getByLabel("Embedding vmax", { exact: true }).fill("2");
	await page.getByLabel("Embedding vmax", { exact: true }).press("Enter");
	await expect(page.locator(".embedding-settings .legend-range")).toHaveText(
		"02",
	);
	await expect(
		page.locator('.embedding-chart-area [role="alert"]'),
	).toHaveCount(0);
	await page.screenshot({
		path: join(appRoot, "test-results/merscope-embedding.png"),
	});
	await expect(page.locator(".notices")).toHaveCount(0);
	expect(errors).toEqual([]);
	expect(
		network.filter(
			(r) => r.method !== "GET" || !r.url.startsWith("http://127.0.0.1:8879/"),
		),
	).toEqual([]);
	await page.screenshot({
		path: join(appRoot, "test-results/merscope-real.png"),
	});
});
