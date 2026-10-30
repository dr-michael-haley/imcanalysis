import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join, parse } from "node:path";

const packages = new Map();
export function licensePlugin(engineRoot, emit = true) {
	return {
		name: "sbt-spatialdata-licenses",
		moduleParsed({ id }) {
			if (!id.includes("node_modules") && !id.includes("/packages/")) return;
			for (
				let folder = dirname(id.split("?")[0]);
				folder !== parse(folder).root;
				folder = dirname(folder)
			) {
				const manifest = join(folder, "package.json");
				if (!existsSync(manifest)) continue;
				const pkg = JSON.parse(readFileSync(manifest, "utf8"));
				if (!pkg.name) continue;
				const files = readdirSync(folder).filter(
					(name) =>
						/^(licen[sc]e|copying|notice)(\.|$)/i.test(name) &&
						statSync(join(folder, name)).isFile(),
				);
				const notices = files
					.map((name) => readFileSync(join(folder, name), "utf8"))
					.join("\n\n");
				packages.set(
					`${pkg.name}@${pkg.version}`,
					`${pkg.name} ${pkg.version}\nLicense: ${typeof pkg.license === "string" ? pkg.license : "See upstream package"}\n${notices}`,
				);
				break;
			}
		},
		generateBundle() {
			if (!emit) return;
			this.emitFile({
				type: "asset",
				fileName: "THIRD_PARTY_LICENSES.txt",
				source: [
					"SpatialData.js\n" +
						readFileSync(join(engineRoot, "LICENSE"), "utf8"),
					"parquet-wasm (vendored revision 2c23652, MIT licence)\n" +
						readFileSync(
							join(engineRoot, "packages/core/vendor/parquet-wasm/LICENSE_MIT"),
							"utf8",
						),
					...[...packages.entries()]
						.sort(([a], [b]) => a.localeCompare(b))
						.map(([, value]) => value),
				].join("\n\n----------------------------------------\n\n"),
			});
		},
	};
}
