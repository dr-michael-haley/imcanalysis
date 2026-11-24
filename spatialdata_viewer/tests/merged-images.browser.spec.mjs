import { readdir, readFile, stat } from "node:fs/promises";
import { basename, join, resolve, sep } from "node:path";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
import { selectLayer, exported } from "./ui-helpers.mjs";
const { test, expect } = engineRequire("@playwright/test");

test("merged dataset discovers all images and renders real image chunks", async ({
	page,
}) => {
	const dataset = process.env.SBT_MERGED_IMAGE_DATASET;
	test.skip(
		!dataset,
		"Set SBT_MERGED_IMAGE_DATASET to the merged local store.",
	);
	test.setTimeout(180000);
	const root = resolve(dataset),
		listed = [],
		reads = [],
		errors = [];
	page.on("pageerror", (e) => errors.push(e.message));
	// Read-only bridge for the native handle API. No copying or enumerating the
	// full dataset, and no data server or write operation on the source store.
	await page.exposeFunction("datasetRead", async (operation, key) => {
		const path = resolve(root, key);
		if (path !== root && !path.startsWith(root + sep))
			throw new Error("Outside dataset");
		try {
			if (operation === "stat")
				return (await stat(path)).isDirectory() ? "directory" : "file";
			if (operation === "list") {
				listed.push(key);
				return (await readdir(path, { withFileTypes: true })).map((e) => ({
					name: e.name,
					kind: e.isDirectory() ? "directory" : "file",
				}));
			}
			if (operation === "read") {
				reads.push(key);
				return (await readFile(path)).toString("base64");
			}
			throw new Error("Unsupported operation");
		} catch (e) {
			if (e.code === "ENOENT" || e.code === "ENOTDIR") return null;
			throw e;
		}
	});
	await page.addInitScript((name) => {
		function handle(path, kind, name) {
			const childPath = (child) => (path ? `${path}/${child}` : child);
			async function child(child, wanted) {
				const actual = await window.datasetRead("stat", childPath(child));
				if (!actual) throw new DOMException("Missing", "NotFoundError");
				if (actual !== wanted)
					throw new DOMException("Wrong kind", "TypeMismatchError");
				return handle(childPath(child), wanted, child);
			}
			return {
				name,
				kind,
				async *values() {
					for (const item of await window.datasetRead("list", path))
						yield handle(childPath(item.name), item.kind, item.name);
				},
				getDirectoryHandle: (name) => child(name, "directory"),
				getFileHandle: (name) => child(name, "file"),
				async getFile() {
					const data = await window.datasetRead("read", path);
					return new Blob([
						Uint8Array.from(atob(data), (c) => c.charCodeAt(0)),
					]);
				},
			};
		}
		window.showDirectoryPicker = async () => handle("", "directory", name);
	}, basename(root));
	await page.goto("/");
	const skipUsage = page.getByRole("button", { name: "Not now", exact: true });
	if (await skipUsage.isVisible()) await skipUsage.click();
	await page
		.getByRole("button", { name: "Open local / network folder" })
		.first()
		.click();
	await expect(page.getByLabel("Marker for channel 1")).toBeVisible({
		timeout: 120000,
	});
	const names = (await readdir(join(root, "images"), { withFileTypes: true }))
		.filter((e) => e.isDirectory())
		.map((e) => e.name)
		.sort();
	const workspace = await exported(page);
	expect(
		workspace.panes[0].renderStack.entries
			.filter((e) => e.source.elementType === "image")
			.map((e) => e.source.elementKey)
			.sort(),
	).toEqual(names);
	await selectLayer(page, names[0], "image");
	await expect(
		page.getByLabel("Marker for channel 1").locator("option"),
	).toHaveCount(11);
	await expect
		.poll(
			() =>
				reads.filter(
					(key) =>
						key.startsWith("images/") && !key.split("/").at(-1).startsWith("."),
				).length,
		)
		.toBeGreaterThan(0);
	// Await actual non-background pixels in the WebGL canvas, not just metadata.
	await expect
		.poll(
			async () => {
				const buffer = await page
					.locator(".viewer-pane canvas")
					.first()
					.screenshot();
				return page.evaluate(async (base64) => {
					const image = new Image();
					image.src = `data:image/png;base64,${base64}`;
					await image.decode();
					const canvas = document.createElement("canvas");
					canvas.width = image.width;
					canvas.height = image.height;
					const ctx = canvas.getContext("2d");
					ctx.drawImage(image, 0, 0);
					const pixels = ctx.getImageData(
						0,
						0,
						canvas.width,
						canvas.height,
					).data;
					let lit = 0;
					for (let i = 0; i < pixels.length; i += 4)
						if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 40) lit++;
					return lit;
				}, buffer.toString("base64"));
			},
			{ timeout: 60000 },
		)
		.toBeGreaterThan(1000);
	expect(listed.filter((key) => /^images\/[^/]+\/\d+(\/|$)/.test(key))).toEqual(
		[],
	);
	expect(errors).toEqual([]);
	await page.screenshot({
		path: join(appRoot, "test-results/merged-images.png"),
	});
});
