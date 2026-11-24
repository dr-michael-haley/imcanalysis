import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
import { selectLayer } from "./ui-helpers.mjs";
const { test, expect } = engineRequire("@playwright/test");

for (const incompleteIndex of [false, true]) {
	test(`direct folder handles render and restore ${incompleteIndex ? "with images missing from consolidation" : "without consolidation"}`, async ({
		page,
	}) => {
		const fixture = JSON.parse(
			readFileSync(join(appRoot, "fixtures/fixture.json")),
		);
		const folder = join(appRoot, "fixtures", fixture.folder);
		const files = readdirSync(folder, { recursive: true, withFileTypes: true })
			.filter((f) => f.isFile())
			.map((f) => {
				const path = join(f.parentPath, f.name);
				return [
					relative(folder, path).replaceAll("\\", "/"),
					readFileSync(path).toString("base64"),
				];
			});
		if (incompleteIndex) {
			const metadata = Object.fromEntries(
				files
					.filter(
						([key]) =>
							/\.(zattrs|zgroup|zarray)$/.test(key) &&
							!key.startsWith("images/"),
					)
					.map(([key, bytes]) => [
						key,
						JSON.parse(Buffer.from(bytes, "base64").toString()),
					]),
			);
			files.push([
				"zmetadata",
				Buffer.from(
					JSON.stringify({ zarr_consolidated_format: 1, metadata }),
				).toString("base64"),
			]);
		}
		const errors = [];
		page.on("pageerror", (e) => errors.push(e.message));
		await page.addInitScript(() => {
			// Use real browser directory handles in an isolated test context. Only the
			// native picker is replaced; file access and IndexedDB cloning remain real.
			window.showDirectoryPicker = async () =>
				(await navigator.storage.getDirectory()).getDirectoryHandle(
					"fixture.zarr",
				);
		});
		await page.goto("/");
		await page.evaluate(async (files) => {
			const root = await (
				await navigator.storage.getDirectory()
			).getDirectoryHandle("fixture.zarr", { create: true });
			const dirs = new Map([["", Promise.resolve(root)]]);
			function directory(path) {
				if (!dirs.has(path)) {
					const parts = path.split("/"),
						name = parts.pop();
					dirs.set(
						path,
						directory(parts.join("/")).then((d) =>
							d.getDirectoryHandle(name, { create: true }),
						),
					);
				}
				return dirs.get(path);
			}
			for (let start = 0; start < files.length; start += 8)
				await Promise.all(
					files.slice(start, start + 8).map(async ([path, data]) => {
						const parts = path.split("/"),
							name = parts.pop();
						const handle = await (
							await directory(parts.join("/"))
						).getFileHandle(name, { create: true });
						const writer = await handle.createWritable();
						await writer.write(
							Uint8Array.from(atob(data), (c) => c.charCodeAt(0)),
						);
						await writer.close();
					}),
				);
		}, files);
		await page
			.getByRole("button", { name: "Open local / network folder" })
			.first()
			.click();
		await expect(page.getByLabel("Marker for channel 1")).toBeVisible();
		await page.getByLabel("Max for channel 1").fill("123");
		await page.getByRole("button", { name: "Setup", exact: true }).click();
		await page
			.getByRole("button", { name: "Close dataset", exact: true })
			.click();
		await page
			.getByRole("button", { name: "Open local / network folder" })
			.first()
			.click();
		await expect(page.getByLabel("Max for channel 1")).toHaveValue("123");
		await selectLayer(page, "Transcripts_A", "points");
		await expect(page.locator(".transcript-controls")).toContainText(
			"transcript genes available",
		);
		await expect(page.locator(".viewer-pane canvas")).toBeVisible();
		expect(errors).toEqual([]);
	});
}
