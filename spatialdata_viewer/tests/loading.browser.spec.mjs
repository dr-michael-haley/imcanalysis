import { readFileSync } from "node:fs";
import { join } from "node:path";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
import { ready, selectLayer } from "./ui-helpers.mjs";
const { test, expect } = engineRequire("@playwright/test");
const fixtures = join(appRoot, "fixtures");
const fixture = JSON.parse(
	readFileSync(join(fixtures, "fixture.json"), "utf8"),
);

test("slow folder reads remain visible, cancel safely, and can be reopened", async ({
	page,
}, testInfo) => {
	const errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	await page.addInitScript(() => {
		const read = Blob.prototype.arrayBuffer;
		window.holdFileReads = true;
		window.releaseFileReads = [];
		Blob.prototype.arrayBuffer = async function () {
			if (window.holdFileReads)
				await new Promise((resolve) => window.releaseFileReads.push(resolve));
			return read.call(this);
		};
	});
	await page.goto("/");
	await expect(
		page.getByRole("link", { name: "Report an issue" }),
	).toHaveAttribute(
		"href",
		/mailto:michael\.haley@manchester\.ac\.uk\?subject=SBT%20MERSCOPE%20Explorer%20issue/,
	);
	const chooser = page.waitForEvent("filechooser");
	await page
		.getByRole("button", { name: "Open SpatialData folder" })
		.first()
		.click();
	await expect(page.getByTestId("dataset-loading")).toContainText(
		"browser lists its files",
	);
	await (await chooser).setFiles(join(fixtures, fixture.folder));
	const loading = page.getByTestId("dataset-loading");
	await expect(loading).toContainText("Reading SpatialData metadata");
	await expect(loading).toContainText("pending");
	await expect(loading.getByLabel("Elapsed time")).not.toHaveText("0s");
	await expect(page.getByTestId("viewport")).toHaveAttribute(
		"aria-busy",
		"true",
	);
	await page.screenshot({ path: testInfo.outputPath("loading-desktop.png") });
	await page.setViewportSize({ width: 620, height: 800 });
	await page.emulateMedia({ reducedMotion: "reduce" });
	await expect(
		loading.getByRole("button", { name: "Cancel loading" }),
	).toBeInViewport();
	await expect(loading.locator(".loading-track span")).toHaveCSS(
		"animation-name",
		"none",
	);
	await page.screenshot({ path: testInfo.outputPath("loading-narrow.png") });
	await loading.getByRole("button", { name: "Cancel loading" }).click();
	await page.evaluate(() => {
		window.holdFileReads = false;
		for (const release of window.releaseFileReads.splice(0)) release();
	});
	await expect(loading).toHaveCount(0);
	await expect(page.locator(".viewer-pane")).toHaveCount(0);
	await page.setViewportSize({ width: 1440, height: 1000 });
	await page
		.getByLabel("SpatialData folder", { exact: true })
		.setInputFiles(join(fixtures, fixture.folder));
	await ready(page);
	await expect(page.getByLabel("Marker for channel 1")).toBeVisible();
	await expect(page.getByTestId("viewport")).toHaveAttribute(
		"aria-busy",
		"false",
	);
	// A layer loaded later must keep showing real read activity too.
	await page.evaluate(() => {
		window.holdFileReads = true;
	});
	await selectLayer(page, "Transcripts_A", "points");
	await expect(page.getByTestId("read-activity")).toBeVisible();
	await expect(page.getByTestId("read-activity")).toContainText("pending");
	await page.screenshot({ path: testInfo.outputPath("loading-layer.png") });
	await page.evaluate(() => {
		window.holdFileReads = false;
		for (const release of window.releaseFileReads.splice(0)) release();
	});
	await expect(page.getByTestId("read-activity")).toHaveCount(0);
	await expect(page.locator(".transcript-controls")).toContainText(
		"transcript genes available",
	);
	expect(errors).toEqual([]);
});

test("dismissed file picker clears loading and invalid archives recover", async ({
	page,
}) => {
	await page.goto("/");
	const chooser = page.waitForEvent("filechooser");
	await page.getByRole("button", { name: "Open ZIP archive" }).click();
	await chooser;
	await expect(page.getByTestId("dataset-loading")).toContainText(
		"Waiting for ZIP selection",
	);
	// Native input cancel is dispatched by the browser when its picker is closed.
	await page
		.getByLabel("SpatialData ZIP", { exact: true })
		.dispatchEvent("cancel");
	await expect(page.getByTestId("dataset-loading")).toHaveCount(0);
	await page
		.getByLabel("SpatialData ZIP", { exact: true })
		.setInputFiles({
			name: "broken.zip",
			mimeType: "application/zip",
			buffer: Buffer.from("Invalid archive"),
		});
	await expect(page.getByRole("alert")).toBeVisible();
	await expect(page.getByTestId("dataset-loading")).toHaveCount(0);
	await page
		.getByLabel("SpatialData ZIP", { exact: true })
		.setInputFiles(join(fixtures, fixture.zip));
	await ready(page);
	await expect(page.getByRole("alert")).toHaveCount(0);
});
