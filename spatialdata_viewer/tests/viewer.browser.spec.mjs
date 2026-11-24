import {
	selectLayer,
	showLayer,
	toolbarAction,
	ready,
	exportPng,
} from "./ui-helpers.mjs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
const { test, expect } = engineRequire("@playwright/test");
const fixtures = join(appRoot, "fixtures");
const fixture = JSON.parse(
	readFileSync(join(fixtures, "fixture.json"), "utf8"),
);
async function pickElement(page, x, y, name) {
	await expect
		.poll(
			async () => {
				await page.mouse.click(x, y, { delay: 30 });
				try {
					await expect(
						page
							.locator("dt")
							.filter({ hasText: /^Element$/ })
							.locator("+ dd"),
					).toHaveText(name, { timeout: 750 });
					return true;
				} catch {
					return false;
				}
			},
			{ timeout: 10000, intervals: [500] },
		)
		.toBe(true);
}
function hashes(directory) {
	return Object.fromEntries(
		readdirSync(directory, { recursive: true })
			.map((name) => join(directory, name))
			.filter((path) => statSync(path).isFile())
			.map((path) => [
				path,
				createHash("sha256").update(readFileSync(path)).digest("hex"),
			]),
	);
}
test("real local SpatialData renders, links cell identities, and restores a saved view without uploads", async ({
	page,
}) => {
	const before = hashes(join(fixtures, fixture.folder));
	const errors = [],
		network = [];
	page.on("pageerror", (error) => errors.push(error.message));
	page.on("request", (request) =>
		network.push({ method: request.method(), url: request.url() }),
	);
	await page.goto("/");
	await expect(
		page.getByRole("heading", { name: "Explore the whole picture." }),
	).toBeVisible();
	await page.screenshot({ path: join(appRoot, "test-results/welcome.png") });
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(fixtures, fixture.folder));
	await expect(
		page.getByRole("heading", { name: fixture.folder, exact: true }),
	).toBeVisible();
	await expect(page.getByLabel("Marker for channel 1")).toBeVisible();
	await expect(page.getByLabel("Marker for channel 1")).toContainText("DNA");
	await expect(page.locator(".viewport canvas")).toBeVisible();
	await ready(page);
	await page.getByLabel("Max for channel 1").fill("180");
	await selectLayer(page, "Cells_A", "labels");
	await page.getByLabel("Colour cells by").selectOption("cell_type");
	await page.getByLabel("Mask style").selectOption("filled");
	await expect(page.locator(".notices")).toHaveCount(0);
	await page.screenshot({ path: join(appRoot, "test-results/dataset.png") });
	// Cell 1 is at (48, 52) in the fitted 256px fixture. Retry the real GPU
	// pick until its tiles arrive; merely mounting a canvas is not readiness.
	const canvas = page.locator(".viewport canvas");
	const bounds = await canvas.boundingBox();
	await pickElement(
		page,
		bounds.x + bounds.width * 0.305,
		bounds.y + bounds.height * 0.24,
		"Cells_A",
	);
	await expect(
		page
			.locator("dt")
			.filter({ hasText: /^Instance ID$/ })
			.locator("+ dd"),
	).toHaveText("1");
	await expect(
		page
			.locator("dt")
			.filter({ hasText: /^Table row$/ })
			.locator("+ dd"),
	).toHaveText("1");
	await expect(page.locator(".feature-data")).toContainText("T cell");
	await page.screenshot({ path: join(appRoot, "test-results/inspection.png") });
	await page.getByRole("button", { name: "Views", exact: true }).click();
	const saved = page.waitForEvent("download");
	await page.getByRole("button", { name: "Export view settings" }).click();
	const savedPath = await (await saved).path();
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await page.getByRole("button", { name: "Setup", exact: true }).click();
	await page
		.getByLabel("Coordinate system", { exact: true })
		.selectOption("roi_B");
	await expect(
		page.getByRole("button", { name: "Tissue_B image", exact: true }),
	).toBeVisible();
	await expect(
		page.getByRole("button", { name: "Tissue_A image", exact: true }),
	).toHaveCount(0);
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page
		.getByLabel("View settings", { exact: true })
		.setInputFiles(savedPath);
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await expect(
		page.getByLabel("Coordinate system", { exact: true }),
	).toHaveValue("roi_A");
	const imageSaved = page.waitForEvent("download");
	await exportPng(page);
	const png = readFileSync(await (await imageSaved).path());
	expect(png.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
	expect(png.byteLength).toBeGreaterThan(10000);
	await page.getByRole("button", { name: "Figure", exact: true }).click();
	await page.getByRole("button", { name: "Viewer mode", exact: true }).click();
	await page.getByRole("button", { name: "Setup", exact: true }).click();
	await page
		.getByRole("button", { name: "Close dataset", exact: true })
		.click();
	await page
		.getByLabel("SpatialData ZIP", { exact: true })
		.setInputFiles(join(fixtures, fixture.zip));
	await expect(page.getByLabel("Marker for channel 1")).toBeVisible();
	await expect(page.locator(".viewport canvas")).toBeVisible();
	await showLayer(page, "Spots_A", true);
	await page.getByRole("button", { name: "Views", exact: true }).click();
	const cameraDownload = page.waitForEvent("download");
	await page.getByRole("button", { name: "Export view settings" }).click();
	const camera = JSON.parse(readFileSync(await (await cameraDownload).path()))
		.panes[0].viewState;
	await page.getByRole("button", { name: "Hide drawer", exact: true }).click();
	const currentBounds = await canvas.boundingBox();
	await pickElement(
		page,
		currentBounds.x +
			currentBounds.width / 2 +
			(48 - camera.target[0]) * 2 ** camera.zoom,
		currentBounds.y +
			currentBounds.height / 2 +
			(52 - camera.target[1]) * 2 ** camera.zoom,
		"Spots_A",
	);
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await showLayer(page, "Transcripts_A", true);

	await pickElement(
		page,
		currentBounds.x +
			currentBounds.width / 2 +
			(80 - camera.target[0]) * 2 ** camera.zoom,
		currentBounds.y +
			currentBounds.height / 2 +
			(220 - camera.target[1]) * 2 ** camera.zoom,
		"Transcripts_A",
	);
	await expect(page.locator(".notices")).toHaveCount(0);
	await expect(page.locator(".canvas-error")).toHaveCount(0);
	expect(errors).toEqual([]);
	expect(
		network.filter(
			(request) =>
				request.method !== "GET" ||
				(!request.url.startsWith("http://127.0.0.1:8879/") &&
					!request.url.startsWith("blob:http://127.0.0.1:8879/")),
		),
	).toEqual([]);
	expect(hashes(join(fixtures, fixture.folder))).toEqual(before);
});
test("invalid folders fail clearly and can be replaced", async ({ page }) => {
	await page.goto("/");
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(fixtures, "invalid"));
	await expect(page.locator(".error")).toBeVisible();
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(fixtures, fixture.folder));
	await expect(page.getByLabel("Marker for channel 1")).toBeVisible();
	await expect(page.locator(".error")).toHaveCount(0);
});

test("cell overlays join shuffled table rows, switch tables, and restore gene settings", async ({
	page,
}) => {
	await page.goto("/");
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(fixtures, fixture.folder));
	await selectLayer(page, "Boundaries_A", "shapes");
	await showLayer(page, "Boundaries_A", true);
	await page.getByLabel("Overlay data").selectOption("gene");
	await page.getByLabel("Overlay field", { exact: true }).selectOption("GeneA");
	await expect(page.locator(".legend-range")).toHaveText("1060");
	const canvas = await page.locator(".viewport canvas").boundingBox();
	await pickElement(
		page,
		canvas.x + canvas.width * 0.305,
		canvas.y + canvas.height * 0.24,
		"Boundaries_A",
	);
	await expect(page.locator(".feature-data")).toContainText("10");
	await expect(
		page
			.locator("dt")
			.filter({ hasText: /^Matched row$/ })
			.locator("+ dd"),
	).toHaveText("1");
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await page
		.getByLabel("Linked cell table")
		.selectOption("expression_alternative");
	await page.getByLabel("Overlay data").selectOption("gene");
	await page.getByLabel("Overlay field", { exact: true }).selectOption("GeneA");
	await expect(page.locator(".legend-range")).toHaveText("100600");
	await page.getByLabel("Expression scale").selectOption("log1p");
	await expect(page.getByLabel("Overlay legend")).toContainText("log1p(value)");
	await page.getByRole("button", { name: "Views", exact: true }).click();
	const download = page.waitForEvent("download");
	await page.getByRole("button", { name: "Export view settings" }).click();
	const file = await (await download).path();
	await page.getByLabel("View settings", { exact: true }).setInputFiles(file);
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await selectLayer(page, "Boundaries_A", "shapes");
	await expect(page.getByLabel("Linked cell table")).toHaveValue(
		"expression_alternative",
	);
	await expect(page.getByLabel("Overlay field", { exact: true })).toHaveValue(
		"GeneA",
	);
	await expect(page.getByLabel("Expression scale")).toHaveValue("log1p");
	await page.getByLabel("Overlay data").selectOption("obs");
	await page
		.getByLabel("Overlay field", { exact: true })
		.selectOption("cell_type");
	await expect(page.locator(".legend-categories")).toContainText("T cell");
	await expect(page.locator(".notices")).toHaveCount(0);
});
