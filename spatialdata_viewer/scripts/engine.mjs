import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";

export const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const engineLock = JSON.parse(
	readFileSync(join(appRoot, "engine.lock.json"), "utf8"),
);
export const engineRoot = resolve(
	process.env.SBT_SPATIALDATA_ROOT || join(appRoot, ".engine/SpatialData.js"),
);
export const engineRequire = createRequire(join(engineRoot, "package.json"));
export function checkEngine() {
	if (!existsSync(join(engineRoot, "packages/vis/package.json"))) {
		throw new Error(
			"Run npm run prepare:engine, or set SBT_SPATIALDATA_ROOT to your SpatialData.js checkout.",
		);
	}
	const revision = execFileSync("git", ["rev-parse", "HEAD"], {
		cwd: engineRoot,
		encoding: "utf8",
	}).trim();
	const dirty = Boolean(
		execFileSync("git", ["status", "--porcelain", "--untracked-files=no"], {
			cwd: engineRoot,
			encoding: "utf8",
		}).trim(),
	);
	if (
		!process.env.SBT_SPATIALDATA_ROOT &&
		(revision !== engineLock.revision || dirty)
	) {
		throw new Error(
			"The prepared engine differs from engine.lock.json. Use an explicit SBT_SPATIALDATA_ROOT for development.",
		);
	}
	return { revision, dirty, repository: engineLock.repository };
}
