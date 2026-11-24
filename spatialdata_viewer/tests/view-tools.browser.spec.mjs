import { readFileSync } from "node:fs";
import { join } from "node:path";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
import { selectLayer, showLayer, exported, ready } from "./ui-helpers.mjs";
const { test, expect } = engineRequire("@playwright/test");
const fixture = JSON.parse(
	readFileSync(join(appRoot, "fixtures/fixture.json")),
);
async function open(page) {
	await page.goto("/");
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(appRoot, "fixtures", fixture.folder));
	await ready(page);
}
async function number(page, label, value) {
	await page.getByLabel(label, { exact: true }).fill(String(value));
	await page.getByLabel(label, { exact: true }).press("Enter");
}
async function importState(page, state) {
	await page.getByLabel("View settings", { exact: true }).setInputFiles({
		name: "views.json",
		mimeType: "application/json",
		buffer: Buffer.from(JSON.stringify(state)),
	});
}
async function pixels(page, buffer, mime) {
	return page.evaluate(
		async ({ data, mime }) => {
			const image = new Image();
			image.src = `data:${mime};base64,${data}`;
			await image.decode();
			const canvas = document.createElement("canvas");
			canvas.width = image.width;
			canvas.height = image.height;
			const ctx = canvas.getContext("2d");
			ctx.drawImage(image, 0, 0);
			const p = ctx.getImageData(0, 0, image.width, image.height).data;
			let coloured = 0;
			for (let i = 0; i < p.length; i += 4)
				if (
					Math.max(p[i], p[i + 1], p[i + 2]) -
						Math.min(p[i], p[i + 1], p[i + 2]) >
					30
				)
					coloured++;
			return { width: image.width, height: image.height, coloured };
		},
		{ data: buffer.toString("base64"), mime },
	);
}
test("active layers, per-view legends, calibration and shared tissue selection tools", async ({
	page,
}) => {
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	await open(page);
	await page.getByRole("button", { name: "Setup", exact: true }).click();
	await expect(
		page.getByLabel("Coordinate system", { exact: true }),
	).toBeVisible();
	await selectLayer(page, "Boundaries_A", "shapes");
	await expect(
		page.getByRole("button", { name: "Add layers", exact: true }),
	).toBeVisible();
	await expect(page.locator(".layer-picker")).toHaveCount(0);
	await page.getByLabel("Overlay data").selectOption("gene");
	await page.getByLabel("Overlay field", { exact: true }).selectOption("GeneA");
	await expect(page.locator(".legend-range")).toHaveText("1060");
	await showLayer(page, "Boundaries_A", false);
	await expect(
		page.getByRole("button", { name: "Boundaries_A shapes", exact: true }),
	).toBeVisible();
	await showLayer(page, "Boundaries_A", true);
	await page
		.getByRole("button", { name: "Remove Boundaries_A", exact: true })
		.click();
	await selectLayer(page, "Boundaries_A", "shapes");
	await expect(page.getByLabel("Overlay field", { exact: true })).toHaveValue(
		"GeneA",
	);
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page
		.getByLabel("Legend title", { exact: true })
		.fill("Immune neighbourhood");
	await page
		.getByLabel("Legend position", { exact: true })
		.selectOption("top-left");
	const editor = page
		.locator(".legend-layer-editor")
		.filter({ has: page.locator("summary", { hasText: /^Tissue_A$/ }) });
	await editor.locator("summary").click();
	await editor
		.getByLabel("Legend label for DNA", { exact: true })
		.fill("Nuclei");
	await expect(page.locator(".pane-furniture")).toContainText("Nuclei");
	await page.getByRole("button", { name: "Add view", exact: true }).click();
	await page.getByLabel("Legend title", { exact: true }).fill("Second pane");
	await page.getByLabel("Legend in View 1", { exact: true }).uncheck();
	await expect(
		page.locator(".viewer-pane").nth(0).locator('g[id$="-legend"]'),
	).toHaveCount(0);
	await expect(page.locator(".pane-furniture").nth(1)).toContainText(
		"Second pane",
	);
	await page
		.getByRole("button", { name: "Remove View 2", exact: true })
		.click();
	await page.getByLabel("Legend in View 1", { exact: true }).check();
	await page
		.locator("summary")
		.filter({ hasText: /^Scale bar settings$/ })
		.click();
	await page
		.getByRole("button", { name: "Use 1 coordinate unit = 1 µm", exact: true })
		.click();
	await expect(page.locator(".pane-furniture")).toContainText("µm");
	const saved = await exported(page);
	expect(
		saved.panes[0].legend.layers["images:Tissue_A"].labels["channel:0"],
	).toBe("Nuclei");
	expect(saved.figure.calibrated).toBe(true);
	saved.panes[0].viewState = { target: [128, 128, 0], zoom: 0 };
	await importState(page, saved);
	await page.getByRole("button", { name: "Hide drawer", exact: true }).click();
	await page
		.locator(".viewer-toolbar")
		.getByRole("button", { name: "Inspect", exact: true })
		.click();
	await page.getByRole("button", { name: /^Lasso cells Draw/ }).click();
	const box = await page
		.getByLabel("Lasso cells in View 1", { exact: true })
		.boundingBox();
	await page.mouse.move(box.x + 5, box.y + 5);
	await page.mouse.down();
	for (const [x, y] of [
		[box.width - 5, 5],
		[box.width - 5, box.height - 5],
		[5, box.height - 5],
		[5, 5],
	])
		await page.mouse.move(box.x + x, box.y + y, { steps: 5 });
	await page.mouse.up();
	await expect(page.locator(".selection-strip")).toContainText("6 selected");
	await page
		.getByRole("button", { name: "Clear selected cells", exact: true })
		.click();
	await page
		.getByRole("button", { name: "Select displayed", exact: true })
		.click();
	await expect(page.locator(".selection-strip")).toContainText("6 selected");
	await page.keyboard.press("Escape");
	await page
		.getByRole("button", { name: "Clear selected cells", exact: true })
		.click();
	await page.getByRole("button", { name: "Bookmark", exact: true }).click();
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page.screenshot({ path: join(appRoot, "test-results/view-tools.png") });
	await page.setViewportSize({ width: 1000, height: 800 });
	await page.getByRole("button", { name: "Hide sidebar", exact: true }).click();
	await expect(
		page.getByRole("button", { name: "Show sidebar", exact: true }),
	).toBeVisible();
	expect(
		await page.evaluate(
			() => document.documentElement.scrollWidth <= window.innerWidth,
		),
	).toBe(true);
	await page.screenshot({
		path: join(appRoot, "test-results/view-tools-compact.png"),
	});
	await page.getByRole("button", { name: "View", exact: true }).click();
	await page.getByText("Scale bar settings", { exact: true }).last().click();
	const popup = await page
		.getByRole("group", { name: "View options", exact: true })
		.boundingBox();
	expect(popup.y).toBeGreaterThanOrEqual(0);
	expect(popup.y + popup.height).toBeLessThanOrEqual(800);
	await page.screenshot({
		path: join(appRoot, "test-results/view-toolbar-compact.png"),
	});
	await page.keyboard.press("Escape");
	expect(errors).toEqual([]);
});

test("figure composition exports real pixels, layered SVG, physical dimensions and restores cameras", async ({
	page,
}) => {
	test.setTimeout(150000);
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	await open(page);
	await selectLayer(page, "Cells_A", "labels");
	await page.getByLabel("Colour cells by").selectOption("cell_type");
	await expect(page.locator(".pane-furniture")).toContainText("T cell");
	await selectLayer(page, "Boundaries_A", "shapes");
	await page.getByLabel("Overlay data").selectOption("gene");
	await page.getByLabel("Overlay field", { exact: true }).selectOption("GeneA");
	await expect(page.locator(".legend-range")).toHaveText("1060");
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page.getByRole("button", { name: "Add view", exact: true }).click();
	const original = await exported(page);
	await page.getByRole("button", { name: "Export", exact: true }).click();
	await page.getByRole("button", { name: "Figure mode", exact: true }).click();
	await page.getByLabel("Aspect per pane").selectOption("1:1");
	await number(page, "Pane width (px)", 400);
	await number(page, "Pane spacing (px)", 16);
	await expect(
		page.getByLabel("Lock field of view in µm", { exact: true }),
	).toBeDisabled();
	await page
		.getByRole("button", { name: "Use 1 coordinate unit = 1 µm", exact: true })
		.click();
	await page.getByLabel("Lock field of view in µm", { exact: true }).check();
	await number(page, "Width (µm)", 256);
	await number(page, "Height (µm)", 128);
	await number(page, "Figure legend text (px)", 10);
	await number(page, "View name text (px)", 14);
	await number(page, "Scale-bar text (px)", 12);
	await page.getByLabel("Figure filename prefix").fill("Comparison");
	await page.getByLabel("Timestamp (UTC)", { exact: true }).uncheck();
	await page.getByLabel("Automatically bookmark each export").check();
	await page
		.getByRole("button", {
			name: "Read coordinate-system metadata",
			exact: true,
		})
		.click();
	await expect(page.locator(".figure-panel")).toContainText(
		"fields have one value",
	);
	for (const format of ["png", "jpeg", "svg"]) {
		await page.getByLabel("Figure format").selectOption(format);
		const downloading = page.waitForEvent("download");
		await page
			.getByRole("button", { name: "Export figure", exact: true })
			.click();
		const download = await downloading,
			data = readFileSync(await download.path());
		expect(download.suggestedFilename()).toMatch(/^Comparison-roi_A\./);
		if (format === "svg") {
			const svg = data.toString();
			expect(svg.match(/<image /g)).toHaveLength(6);
			expect(svg).toContain('inkscape:label="Tissue_A"');
			expect(svg).toContain("<text");
			expect(svg).toContain("&quot;coordinateWidth&quot;:256");
			expect(svg).toContain("&quot;coordinateHeight&quot;:128");
			await download.saveAs(join(appRoot, "test-results/figure-layers.svg"));
		} else {
			const result = await pixels(page, data, `image/${format}`);
			expect(result.width).toBe(816);
			expect(result.height).toBe(200);
			expect(result.coloured).toBeGreaterThan(1500);
			await download.saveAs(
				join(appRoot, `test-results/figure-comparison.${format}`),
			);
		}
	}
	await page.screenshot({
		path: join(appRoot, "test-results/figure-panel.png"),
	});
	await page.getByRole("button", { name: "Viewer mode", exact: true }).click();
	const after = await exported(page);
	expect(after.panes.map((p) => p.viewState)).toEqual(
		original.panes.map((p) => p.viewState),
	);
	expect(after.bookmarks).toHaveLength(3);
	expect(after.figureReturn).toBe(null);
	await page
		.getByRole("button", { name: "Save locally now", exact: true })
		.click();
	await expect(page.locator(".drawer-bar")).toContainText("Saved locally");
	await page.reload();
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(appRoot, "fixtures", fixture.folder));
	await ready(page);
	const restored = await exported(page);
	expect(restored.figure).toEqual(after.figure);
	expect(
		restored.panes.map((p) => [p.legend, p.layerNames, p.scaleBar]),
	).toEqual(after.panes.map((p) => [p.legend, p.layerNames, p.scaleBar]));
	expect(errors).toEqual([]);
});
