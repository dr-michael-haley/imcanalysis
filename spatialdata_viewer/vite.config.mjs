import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import { appRoot, engineRoot, engineRequire } from "./scripts/engine.mjs";
import { licensePlugin } from "./scripts/licenses.mjs";

const { default: react } = await import(
	pathToFileURL(engineRequire.resolve("@vitejs/plugin-react")).href
);
const visRequire = createRequire(join(engineRoot, "packages/vis/package.json"));
const exact = (find, replacement) => ({
	find: new RegExp(`^${find.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`),
	replacement,
});
const aliases = [
	{
		find: "@spatialdata/core/parquet-worker",
		replacement: join(engineRoot, "packages/core/dist/parquet-worker.js"),
	},
	exact(
		"@spatialdata/core/parquet-wasm",
		join(engineRoot, "packages/core/vendor/parquet-wasm/parquet_wasm.js"),
	),
	exact(
		"zarrextra/workers",
		join(engineRoot, "packages/zarrextra/dist/workers.js"),
	),
	...["core", "layers", "avivatorish", "react", "vis"].map((name) =>
		exact(
			`@spatialdata/${name}`,
			join(engineRoot, `packages/${name}/dist/index.js`),
		),
	),
	exact("zarrextra", join(engineRoot, "packages/zarrextra/dist/index.js")),
	{
		find: "react-dom",
		replacement: dirname(visRequire.resolve("react-dom/package.json")),
	},
	{
		find: "react",
		replacement: dirname(visRequire.resolve("react/package.json")),
	},
	exact(
		"@deck.gl/core",
		visRequire.resolve("@deck.gl/core").replace(/index\.cjs$/, "index.js"),
	),
];
const version = JSON.parse(readFileSync(join(appRoot, "package.json"))).version;
export default {
	root: appRoot,
	base: "./",
	plugins: [react(), licensePlugin(engineRoot)],
	define: { __VIEWER_VERSION__: JSON.stringify(version) },
	resolve: { alias: aliases, dedupe: ["react", "react-dom"] },
	worker: { format: "es", plugins: () => [licensePlugin(engineRoot, false)] },
	assetsInclude: ["**/*.wasm"],
	build: {
		target: "es2022",
		outDir: "dist",
		assetsInlineLimit: 0,
		rolldownOptions: { output: { codeSplitting: false } },
	},
	server: { host: "127.0.0.1", port: 8879 },
};
