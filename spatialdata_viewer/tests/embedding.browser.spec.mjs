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
test("gene availability, draggable scales and linked AnnData embeddings", async ({
	page,
}) => {
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	await page.goto("/");
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(appRoot, "fixtures", fixture.folder));
	await selectLayer(page, "Boundaries_A", "shapes");
	await showLayer(page, "Boundaries_A", true);
	await page.getByRole("button", { name: "Gene lists", exact: true }).click();
	await page.getByLabel("Gene list name").fill("Test panel");
	await page.getByLabel("Gene list genes").fill("GeneA\nGeneB\nAbsent\ngenea");
	await expect(page.getByLabel("Gene availability")).toContainText(
		"2 / 4 genes present in expression · X",
	);
	await expect(page.getByLabel("Gene availability")).toContainText(
		"Missing (2)",
	);
	await expect(page.getByLabel("Gene availability")).toContainText(
		"Absent, genea",
	);
	await page.getByLabel("Gene check table").selectOption("cells");
	await expect(page.getByLabel("Gene availability")).toContainText(
		"0 / 4 genes present in cells · X",
	);
	await page
		.getByRole("button", { name: "Create gene list", exact: true })
		.click();
	await page.getByLabel("Overlay data").selectOption("gene");
	await page.getByLabel("Overlay field", { exact: true }).selectOption("GeneA");
	await expect(page.locator(".cell-controls .legend-range")).toHaveText("1060");
	await page.getByLabel("Colour map", { exact: true }).selectOption("magma");
	await page.getByLabel("vmax", { exact: true }).fill("30");
	await page.getByLabel("vmax", { exact: true }).press("Enter");
	await expect(page.locator(".cell-controls .legend-range")).toHaveText("1030");
	const before = Number(
		await page.getByLabel("Adjust vmax", { exact: true }).inputValue(),
	);
	await page.getByLabel("Adjust vmax", { exact: true }).focus();
	await page.keyboard.press("ArrowLeft");
	expect(
		Number(await page.getByLabel("Adjust vmax", { exact: true }).inputValue()),
	).toBeLessThan(before);
	await page.getByRole("button", { name: "Embeddings", exact: true }).click();
	await expect(page.getByLabel("Embedding projection")).toHaveValue("X_umap");
	await expect(page.locator(".embedding-status")).toContainText(
		"6 cells in current region",
	);
	await expect(page.getByLabel("Linked cell embedding")).toBeVisible();
	await page.getByLabel("Embedding colour by").selectOption("gene");
	await page.getByLabel("Embedding colour field").selectOption("GeneB");
	await expect(page.locator(".embedding-settings .legend-range")).toHaveText(
		"212",
	);
	await page
		.getByLabel("Embedding Colour map", { exact: true })
		.selectOption("plasma");
	await page.getByLabel("Embedding vmax", { exact: true }).fill("6");
	await page.getByLabel("Embedding vmax", { exact: true }).press("Enter");
	await expect(page.locator(".embedding-settings .legend-range")).toHaveText(
		"26",
	);
	await expect(page.getByLabel("Colour map", { exact: true })).toHaveValue(
		"magma",
	);
	// UMAP minimum x/y maps to cell 1, row 1 in the intentionally shuffled table.
	const canvas = page.getByLabel("Linked cell embedding"),
		box = await canvas.boundingBox();
	await page.mouse.click(box.x + 24, box.y + box.height - 24, { delay: 30 });
	await expect(
		page
			.locator("dt")
			.filter({ hasText: /^Instance ID$/ })
			.locator("+ dd"),
	).toHaveText("1");
	await expect(
		page
			.locator("dt")
			.filter({ hasText: /^Matched row$/ })
			.locator("+ dd"),
	).toHaveText("1");
	await expect(page.locator(".embedding-status")).toContainText("1 selected");
	await page
		.getByRole("button", { name: "Centre selected cell", exact: true })
		.click();
	await expect(page.locator(".sidebar .error")).toHaveCount(0);
	await page.getByRole("button", { name: "Lasso cells", exact: true }).click();
	await page.mouse.move(box.x + 8, box.y + 8);
	await page.mouse.down();
	await page.mouse.move(box.x + box.width - 8, box.y + 8, { steps: 4 });
	await page.mouse.move(box.x + box.width - 8, box.y + box.height - 8, {
		steps: 4,
	});
	await page.mouse.move(box.x + 8, box.y + box.height - 8, { steps: 4 });
	await page.mouse.move(box.x + 8, box.y + 8, { steps: 4 });
	await page.mouse.up();
	await expect(page.locator(".embedding-status")).toContainText("6 selected");
	await page.getByRole("button", { name: "Clear embedding selection" }).click();
	await page.getByLabel("Embedding projection").selectOption("X_pca");
	await page.getByLabel("Embedding y dimension").selectOption("2");
	await expect(page.locator(".embedding-status")).toContainText(
		"6 cells in current region",
	);
	await page.getByLabel("Embedding colour by").selectOption("obs");
	await page.getByLabel("Embedding colour field").selectOption("cell_type");
	await expect(
		page.locator(".embedding-settings .legend-categories"),
	).toContainText("T cell");
	await page.getByLabel("Embedding colour by").selectOption("sum");
	await page
		.getByLabel("Embedding colour field")
		.selectOption({ label: "Test panel" });
	await expect(page.locator(".embedding-settings .legend-range")).toHaveText(
		"1272",
	);
	await page.getByLabel("Embedding projection").selectOption("X_umap");
	await page
		.getByLabel("Embedding table")
		.selectOption("expression_alternative");
	await expect(page.locator(".embedding-chart-area")).toContainText(
		"No stored embedding is available",
	);
	await page.getByLabel("Embedding table").selectOption("expression");
	await page.getByLabel("Embedding colour by").selectOption("obs");
	await page.getByLabel("Embedding colour field").selectOption("cell_type");
	await expect(
		page.locator(".embedding-settings .legend-categories"),
	).toContainText("T cell");
	const exported = page.waitForEvent("download");
	await page.getByRole("button", { name: "Export embedding PNG" }).click();
	expect(
		readFileSync(await (await exported).path())
			.subarray(0, 8)
			.toString("hex"),
	).toBe("89504e470d0a1a0a");
	await page.screenshot({ path: join(appRoot, "test-results/embeddings.png") });
	await page.getByRole("button", { name: "Workspace", exact: true }).click();
	await page.getByRole("button", { name: "Save locally now" }).click();
	await expect(page.locator(".drawer-bar")).toContainText("Saved locally");
	await page.reload();
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(appRoot, "fixtures", fixture.folder));
	await ready(page);
	await page.getByRole("button", { name: "Embeddings", exact: true }).click();
	await expect(page.getByLabel("Embedding colour field")).toHaveValue(
		"cell_type",
	);
	await expect(
		page.locator(".embedding-settings .legend-categories"),
	).toContainText("T cell");
	await selectLayer(page, "Boundaries_A", "shapes");
	await expect(page.getByLabel("Colour map", { exact: true })).toHaveValue(
		"magma",
	);
	expect(errors).toEqual([]);
	await expect(page.locator(".viewport canvas")).toHaveCount(1);
	await expect(page.locator('.viewport [role="alert"]')).toHaveCount(0);
});
