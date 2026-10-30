import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { resolve, extname, sep } from "node:path";
import { appRoot } from "./engine.mjs";

const root = resolve(appRoot, "dist");
const port = Number(process.env.PORT || 8879);
const mime = {
	".html": "text/html; charset=utf-8",
	".js": "text/javascript",
	".css": "text/css",
	".json": "application/json",
	".svg": "image/svg+xml",
	".png": "image/png",
	".wasm": "application/wasm",
	".txt": "text/plain; charset=utf-8",
};
// Emscripten's codec bindings generate JavaScript functions at initialization.
const policy =
	"default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; worker-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
createServer(async (req, res) => {
	res.setHeader("Content-Security-Policy", policy);
	res.setHeader("X-Content-Type-Options", "nosniff");
	res.setHeader("Referrer-Policy", "no-referrer");
	if (!["GET", "HEAD"].includes(req.method)) {
		res.writeHead(405);
		res.end();
		return;
	}
	try {
		const pathname = decodeURIComponent(
			new URL(req.url, "http://localhost").pathname,
		);
		const file = resolve(
			root,
			"." + (pathname.endsWith("/") ? pathname + "index.html" : pathname),
		);
		if (!file.startsWith(root + sep) || !(await stat(file)).isFile())
			throw new Error("Not found");
		const body = await readFile(file);
		res.writeHead(200, {
			"Content-Type": mime[extname(file)] || "application/octet-stream",
		});
		res.end(req.method === "HEAD" ? undefined : body);
	} catch {
		res.writeHead(404);
		res.end("Not found");
	}
}).listen(port, "127.0.0.1", () =>
	console.log(`SpatialData Explorer: http://127.0.0.1:${port}`),
);
