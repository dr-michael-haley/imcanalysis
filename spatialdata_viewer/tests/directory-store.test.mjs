import test from "node:test";
import assert from "node:assert/strict";
import { openDirectoryStore } from "../src/directory-store.mjs";

function fixture() {
	const reads = [],
		listed = [];
	const files = {
		"sample.zarr/.zgroup": '{"zarr_format":2}',
		"sample.zarr/.zattrs": "{}",
		"sample.zarr/images/.zgroup": "{}",
		"sample.zarr/images/plane/.zarray": '{"shape":[2,2]}',
		"sample.zarr/images/plane/0/0": "pixels",
		"sample.zarr/points/.zgroup": "{}",
		"sample.zarr/points/transcripts/.zgroup": "{}",
		"sample.zarr/points/transcripts/points.parquet/part.7.parquet":
			"PAR1abcdefPAR1",
	};
	const missing = () => new DOMException("Missing", "NotFoundError");
	function directory(path = "") {
		return {
			kind: "directory",
			name: path.split("/").at(-1) || "wrapper",
			async *values() {
				listed.push(path);
				assert.notEqual(
					path,
					"sample.zarr/images/plane",
					"must not list array chunks",
				);
				const prefix = path ? `${path}/` : "";
				const children = new Set(
					Object.keys(files)
						.filter((k) => k.startsWith(prefix))
						.map((k) => k.slice(prefix.length).split("/")[0]),
				);
				for (const name of children)
					yield files[prefix + name] === undefined
						? directory(prefix + name)
						: { kind: "file", name };
			},
			async getDirectoryHandle(name) {
				const key = path ? `${path}/${name}` : name;
				if (!Object.keys(files).some((k) => k.startsWith(`${key}/`)))
					throw missing();
				return directory(key);
			},
			async getFileHandle(name) {
				const key = path ? `${path}/${name}` : name;
				if (files[key] === undefined) throw missing();
				return {
					async getFile() {
						reads.push(key);
						return new Blob([files[key]]);
					},
				};
			},
		};
	}
	return { root: directory(), files, reads, listed };
}

test("direct directory discovery skips chunk trees and retains multipart Parquet keys", async () => {
	const f = fixture();
	const store = await openDirectoryStore(f.root);
	assert.equal(store.directoryHandle.name, "sample.zarr");
	assert.ok(
		store
			.contents()
			.some((n) => n.path === "/images/plane" && n.kind === "array"),
	);
	assert.ok(
		store
			.listKeys()
			.includes("points/transcripts/points.parquet/part.7.parquet"),
	);
	assert.ok(!f.reads.some((k) => k.endsWith("0/0") || k.endsWith(".parquet")));
	assert.equal(
		new TextDecoder().decode(await store.get("/images/plane/0/0")),
		"pixels",
	);
	assert.equal(
		new TextDecoder().decode(
			await store.getRange(
				"/points/transcripts/points.parquet/part.7.parquet",
				{ offset: 4, length: 3 },
			),
		),
		"abc",
	);
	assert.equal(
		new TextDecoder().decode(
			await store.getRange(
				"/points/transcripts/points.parquet/part.7.parquet",
				{ suffixLength: 4 },
			),
		),
		"PAR1",
	);
	assert.equal(await store.get("/missing"), undefined);
	await assert.rejects(store.get("/../outside"), /Invalid store path/);
});

test("consolidated metadata avoids probing each array while still finding Parquet parts", async () => {
	const f = fixture();
	f.files["sample.zarr/zmetadata"] = "{}";
	const contents = [
		{ path: "/", kind: "group" },
		{ path: "/images", kind: "group" },
		{ path: "/images/plane", kind: "array" },
		{ path: "/points", kind: "group" },
		{ path: "/points/transcripts", kind: "group" },
	];
	const store = await openDirectoryStore(f.root, {
		consolidate: async (raw, options) => {
			assert.equal(options.metadataKey, "zmetadata");
			await raw.get("/zmetadata");
			return { get: raw.get, contents: () => contents };
		},
	});
	assert.deepEqual(f.reads, ["sample.zarr/zmetadata"]);
	assert.deepEqual(store.contents(), contents);
	assert.ok(store.listKeys().some((k) => k.endsWith("part.7.parquet")));
});

for (const metadataKey of [".zmetadata", "zmetadata"]) {
	test(`incomplete ${metadataKey} discovers unlisted image arrays without listing chunks`, async () => {
		const f = fixture();
		f.files[`sample.zarr/${metadataKey}`] = "{}";
		const contents = [
			{ path: "/", kind: "group" },
			{ path: "/points", kind: "group" },
			{ path: "/points/transcripts", kind: "group" },
		];
		const store = await openDirectoryStore(f.root, {
			consolidate: async (raw) => ({ get: raw.get, contents: () => contents }),
		});
		assert.ok(
			store
				.contents()
				.some((n) => n.path === "/images/plane" && n.kind === "array"),
		);
		assert.ok(
			store.contents().some((n) => n.path === "/images" && n.kind === "group"),
		);
		assert.equal(
			new Set(store.contents().map((n) => n.path)).size,
			store.contents().length,
		);
		assert.ok(
			!f.reads.some((k) => k.endsWith("0/0") || k.endsWith(".parquet")),
		);
		assert.ok(store.listKeys().some((k) => k.endsWith("part.7.parquet")));
		assert.deepEqual(
			JSON.parse(
				new TextDecoder().decode(await store.get("/images/plane/.zarray")),
			),
			{ shape: [2, 2] },
		);
		assert.equal(
			new TextDecoder().decode(await store.get("/images/plane/0/0")),
			"pixels",
		);
	});
}

test("cancelled directory discovery and later reads cannot resume", async () => {
	const controller = new AbortController();
	const f = fixture();
	const store = await openDirectoryStore(f.root, { signal: controller.signal });
	controller.abort();
	await assert.rejects(store.get("/images/plane/0/0"), { name: "AbortError" });
	await assert.rejects(
		openDirectoryStore(f.root, { signal: controller.signal }),
		{ name: "AbortError" },
	);
});

test("permission failures remain visible rather than appearing to be missing data", async () => {
	const f = fixture();
	const store = await openDirectoryStore(f.root);
	store.directoryHandle.getFileHandle = async () => {
		throw new DOMException("No read permission", "NotAllowedError");
	};
	await assert.rejects(store.get("/unreadable.parquet"), {
		name: "NotAllowedError",
	});
});
