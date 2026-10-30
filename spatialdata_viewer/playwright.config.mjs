import { appRoot } from "./scripts/engine.mjs";
export default {
	testDir: "./tests",
	testMatch: "*.browser.spec.mjs",
	timeout: 90000,
	workers: 1,
	outputDir: "./test-results",
	use: {
		baseURL: "http://127.0.0.1:8879",
		channel: process.env.SBT_BROWSER_CHANNEL || "chrome",
		headless: true,
		viewport: { width: 1440, height: 1000 },
		launchOptions: { args: ["--enable-unsafe-swiftshader"] },
	},
	webServer: {
		command: `"${process.execPath}" scripts/serve.mjs`,
		cwd: appRoot,
		url: "http://127.0.0.1:8879",
		reuseExistingServer: true,
		timeout: 30000,
	},
};
