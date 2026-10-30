import { existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { execFileSync } from "node:child_process";
import { engineRoot, engineLock, checkEngine } from "./engine.mjs";

if (process.env.SBT_SPATIALDATA_ROOT) {
	checkEngine();
} else {
	if (!existsSync(engineRoot)) {
		mkdirSync(dirname(engineRoot), { recursive: true });
		execFileSync(
			"git",
			["clone", "--no-checkout", engineLock.repository, engineRoot],
			{ stdio: "inherit" },
		);
		execFileSync("git", ["checkout", "--detach", engineLock.revision], {
			cwd: engineRoot,
			stdio: "inherit",
		});
	}
	checkEngine();
}
console.log(
	`Engine ready: ${engineRoot}\nInstall its locked dependencies with pnpm install --frozen-lockfile, then run npm run build here.`,
);
