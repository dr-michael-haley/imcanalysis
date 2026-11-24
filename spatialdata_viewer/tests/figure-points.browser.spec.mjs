import { readFileSync } from "node:fs";
import { join } from "node:path";
import { appRoot, engineRequire } from "../scripts/engine.mjs";
import { exported, ready, selectLayer } from "./ui-helpers.mjs";
const { test, expect } = engineRequire("@playwright/test");
const fixture = JSON.parse(
	readFileSync(join(appRoot, "fixtures/fixture.json")),
);

for (const [selection, props, expected] of [
	["all genes", {}, [true, true, true, true]],
	["selected gene", { featureNames: ["Cd3"] }, [true, true, false, false]],
	["gene list", { geneListId: "my-genes" }, [false, false, true, true]],
	["empty selection", { featureNames: [] }, [false, false, false, false]],
]) {
	test(`Figure exports transcript pixels for ${selection}`, async ({
		page,
	}) => {
		test.setTimeout(90000);
		const errors = [];
		page.on("pageerror", (error) => errors.push(error.message));
		await page.goto("/");
		await page
			.getByLabel("SpatialData folder", { exact: true })
			.setInputFiles(join(appRoot, "fixtures", fixture.folder));
		await ready(page);
		await selectLayer(page, "Transcripts_A", "points");
		const state = await exported(page);
		const pane = state.panes[0];
		for (const entry of pane.renderStack.entries) {
			entry.visible = entry.source.elementType === "points";
			if (entry.visible)
				entry.props = { ...entry.props, pointSize: 4, ...props };
		}
		pane.viewState = { target: [128, 128, 0], zoom: 0 };
		state.geneLists = [{ id: "my-genes", name: "Macrophage", genes: ["Cd68"] }];
		Object.assign(state.figure, {
			mode: "figure",
			paneWidth: 256,
			calibrated: true,
			physicalLock: true,
			umPerPixel: 1,
			physicalWidth: 256,
			physicalHeight: 256,
			showNames: false,
			showLegends: false,
			showLayerNames: false,
			showScaleBars: false,
			background: "#000000",
		});
		await page.getByLabel("View settings", { exact: true }).setInputFiles({
			name: "points.json",
			mimeType: "application/json",
			buffer: Buffer.from(JSON.stringify(state)),
		});
		await page.getByRole("button", { name: "Figure", exact: true }).click();
		for (const format of ["png", "jpeg", "svg"]) {
			await page.getByLabel("Figure format").selectOption(format);
			const downloading = page.waitForEvent("download");
			await page
				.getByRole("button", { name: "Export figure", exact: true })
				.click();
			const download = await downloading;
			const buffer = readFileSync(await download.path());
			const result = await page.evaluate(
				async ({ data, format }) => {
					const image = new Image();
					image.src = `data:image/${format === "svg" ? "svg+xml" : format};base64,${data}`;
					await image.decode();
					const canvas = document.createElement("canvas");
					canvas.width = image.width;
					canvas.height = image.height;
					const ctx = canvas.getContext("2d");
					ctx.drawImage(image, 0, 0);
					return {
						width: image.width,
						height: image.height,
						present: [30, 80, 150, 210].map((x) => {
							const p = ctx.getImageData(x - 4, 216, 9, 9).data;
							let lit = 0;
							for (let i = 0; i < p.length; i += 4)
								if (Math.max(p[i], p[i + 1], p[i + 2]) > 50) lit++;
							return lit > 5;
						}),
					};
				},
				{ data: buffer.toString("base64"), format },
			);
			await download.saveAs(
				join(
					appRoot,
					`test-results/points-${selection.replaceAll(" ", "-")}.${format}`,
				),
			);
			expect(result, `${format} transcript positions`).toEqual({
				width: 256,
				height: 256,
				present: expected,
			});
		}
		expect(errors).toEqual([]);
	});
}
