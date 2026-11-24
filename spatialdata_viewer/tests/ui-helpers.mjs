import { readFileSync } from "node:fs";
import { engineRequire } from "../scripts/engine.mjs";
const { expect } = engineRequire("@playwright/test");
export async function selectLayer(page, key, type) {
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	const button = page.getByRole("button", {
		name: `${key} ${type}`,
		exact: true,
	});
	if (!(await button.count())) {
		if (
			!(await page
				.getByRole("button", { name: `Add ${key} ${type}`, exact: true })
				.count())
		)
			await page
				.getByRole("button", { name: "Add layers", exact: true })
				.click();
		await page
			.getByRole("button", { name: `Add ${key} ${type}`, exact: true })
			.click();
		await page
			.getByRole("button", { name: "Hide layer selector", exact: true })
			.click();
	}
	await button.click();
}
export async function showLayer(page, key, visible) {
	await page.getByRole("button", { name: "Layers", exact: true }).click();
	const desired = page.getByRole("button", {
		name: `${visible ? "Hide" : "Show"} ${key}`,
		exact: true,
	});
	if (await desired.count()) return;
	const toggle = page.getByRole("button", {
		name: `${visible ? "Show" : "Hide"} ${key}`,
		exact: true,
	});
	if (await toggle.count()) await toggle.click();
	else {
		await page.getByRole("button", { name: "Add layers", exact: true }).click();
		await page
			.getByRole("button", {
				name: new RegExp(`^Add ${key} (image|labels|shapes|points)$`),
			})
			.click();
		await page
			.getByRole("button", { name: "Hide layer selector", exact: true })
			.click();
		if (!visible)
			await page
				.getByRole("button", { name: `Hide ${key}`, exact: true })
				.click();
	}
}
export async function toolbarAction(page, name) {
	await page.getByRole("button", { name: "View", exact: true }).click();
	await page.getByRole("button", { name, exact: true }).click();
	await page.keyboard.press("Escape");
}
export async function exported(page) {
	await page.getByRole("button", { name: "Workspace", exact: true }).click();
	const download = page.waitForEvent("download");
	await page
		.getByRole("button", { name: "Export workspace", exact: true })
		.click();
	return JSON.parse(readFileSync(await (await download).path()));
}
export async function ready(page) {
	await expect(page.locator(".viewer-pane canvas").first()).toBeVisible({
		timeout: 60000,
	});
}
export async function exportPng(page) {
	await page.getByRole("button", { name: "Figure", exact: true }).click();
	await page.getByRole("button", { name: "Figure mode", exact: true }).click();
	await page
		.getByRole("button", { name: "Export figure", exact: true })
		.click();
}
