import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { appRoot, engineRoot, engineRequire, checkEngine } from "./engine.mjs";

const mode = process.argv[2] || "build";
const provenance = checkEngine();
function run(entry, args, cwd = appRoot) {
	const result = spawnSync(process.execPath, [entry, ...args], {
		cwd,
		stdio: "inherit",
		env: process.env,
	});
	if (result.error) throw result.error;
	if (result.status !== 0) process.exit(result.status ?? 1);
}
const vite = join(
	dirname(engineRequire.resolve("vite/package.json")),
	"bin/vite.js",
);
if (mode === "build" || mode === "dev") {
	const tsManifest = engineRequire.resolve("typescript/package.json");
	const tsPackage = JSON.parse(readFileSync(tsManifest, "utf8"));
	const compiler = join(
		dirname(tsManifest),
		tsPackage.bin.tsc || tsPackage.bin.tsc6,
	);
	// Compile the fork's packages with its own locked toolchain. No sibling path is
	// baked into the output; Vite resolves the public package builds below.
	for (const name of [
		"zarrextra",
		"core",
		"layers",
		"avivatorish",
		"react",
		"vis",
	]) {
		const cwd = join(engineRoot, "packages", name);
		run(vite, ["build"], cwd);
		if (name === "zarrextra")
			run(vite, ["build", "--mode", "codec-worker"], cwd);
		if (mode === "build") run(compiler, ["--project", "tsconfig.json"], cwd);
	}
	run(vite, [
		mode === "build" ? "build" : "--host",
		...(mode === "dev" ? ["127.0.0.1"] : []),
		"--config",
		join(appRoot, "vite.config.mjs"),
	]);
	if (mode === "build") {
		writeFileSync(
			join(appRoot, "dist/build-info.json"),
			JSON.stringify(
				{
					application: JSON.parse(readFileSync(join(appRoot, "package.json")))
						.version,
					engine: provenance,
				},
				null,
				2,
			),
		);
	}
} else if (mode === "test:browser") {
	run(engineRequire.resolve("@playwright/test/cli"), [
		"test",
		"--config",
		join(appRoot, "playwright.config.mjs"),
	]);
} else throw new Error(`Unknown command: ${mode}`);
