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
	await expect.poll(async () => {
		await page.mouse.click(x, y, { delay: 30 });
		try {
			await expect(page.locator('dt').filter({ hasText: /^Element$/ }).locator('+ dd')).toHaveText(name, { timeout: 750 });
			return true;
		} catch { return false; }
	}, { timeout: 10000, intervals: [500] }).toBe(true);
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
	await expect(
		page.getByRole("button", { name: "Export PNG", exact: true }),
	).toBeEnabled();
	await page.getByLabel("Max for channel 1").fill("180");
	await page.getByRole("button", { name: "Cells_A", exact: false }).click();
	await page.getByLabel("Colour cells by").selectOption("cell_type");
	await page.getByLabel("Mask style").selectOption("filled");
	await expect(page.locator(".notices")).toHaveCount(0);
	await page.screenshot({ path: join(appRoot, "test-results/dataset.png") });
	// Cell 1 is at (48, 52) in the fitted 256px fixture. Retry the real GPU
	// pick until its tiles arrive; merely mounting a canvas is not readiness.
	const canvas = page.locator(".viewport canvas");
	const bounds = await canvas.boundingBox();
	await pickElement(page, bounds.x + bounds.width * 0.305, bounds.y + bounds.height * 0.24, "Cells_A");
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
	await page.getByLabel("Coordinate system").selectOption("roi_B");
	await expect(
		page.getByRole("button", { name: "Tissue_B", exact: false }),
	).toBeVisible();
	await expect(
		page.getByRole("button", { name: "Tissue_A", exact: false }),
	).toHaveCount(0);
	await page.getByRole("button", { name: "Views", exact: true }).click();
	await page
		.getByLabel("View settings", { exact: true })
		.setInputFiles(savedPath);
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await expect(page.getByLabel("Coordinate system")).toHaveValue("roi_A");
	const imageSaved = page.waitForEvent("download");
	await page.getByRole("button", { name: "Export PNG", exact: true }).click();
	const png = readFileSync(await (await imageSaved).path());
	expect(png.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
	expect(png.byteLength).toBeGreaterThan(10000);
	await page
		.getByRole("button", { name: "Close dataset", exact: true })
		.click();
	await page
		.getByLabel("SpatialData ZIP", { exact: true })
		.setInputFiles(join(fixtures, fixture.zip));
	await expect(page.getByLabel("Marker for channel 1")).toBeVisible();
	await expect(page.locator(".viewport canvas")).toBeVisible();
	await page.getByLabel("Show Spots_A", { exact: true }).check();
	await pickElement(page, bounds.x + bounds.width * 0.305, bounds.y + bounds.height * 0.24, "Spots_A");
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	await page.getByLabel("Show Transcripts_A", { exact: true }).check();
	const camera = JSON.parse(readFileSync(savedPath)).viewState;
	await pickElement(page, bounds.x + bounds.width / 2 + (80 - camera.target[0]) * 2 ** camera.zoom, bounds.y + bounds.height / 2 + (220 - camera.target[1]) * 2 ** camera.zoom, "Transcripts_A");
	await expect(page.locator(".notices")).toHaveCount(0);
	await expect(page.locator(".canvas-error")).toHaveCount(0);
	expect(errors).toEqual([]);
	expect(
		network.filter(
			(request) =>
				request.method !== "GET" ||
				!request.url.startsWith("http://127.0.0.1:8879/"),
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
