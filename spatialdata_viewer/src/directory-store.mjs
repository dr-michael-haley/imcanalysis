const metadataNames = new Set([
	".zattrs",
	".zgroup",
	".zarray",
	".zmetadata",
	"zmetadata",
	"zarr.json",
]);
const absent = (e) =>
	e?.name === "NotFoundError" || e?.name === "TypeMismatchError";
const decoder = new TextDecoder();

// A read-only File System Access store. Keep directory handles, not a snapshot of
// every File in the dataset. In particular, never enumerate array chunk trees.
export async function openDirectoryStore(
	selected,
	{ signal, onProgress = () => {}, consolidate } = {},
) {
	const check = () => signal?.throwIfAborted();
	async function entries(handle) {
		check();
		const result = [];
		for await (const entry of handle.values()) {
			check();
			result.push(entry);
		}
		return result;
	}
	let root = selected;
	let rootEntries = await entries(root);
	for (
		let depth = 0;
		depth < 8 && !rootEntries.some((e) => metadataNames.has(e.name));
		depth++
	) {
		if (rootEntries.length !== 1 || rootEntries[0].kind !== "directory") break;
		root = rootEntries[0];
		rootEntries = await entries(root);
	}
	const directories = new Map([["", Promise.resolve(root)]]);
	const metadata = new Map();
	const keys = new Set();
	const nodes = new Map();
	let visited = 0,
		lastProgress = 0;
	function parts(key) {
		const result = key.replace(/^\//, "").split("/");
		if (result.some((p) => !p || p === "." || p === ".." || p.includes("\\")))
			throw new Error(`Invalid store path: ${key}`);
		return result;
	}
	function directory(path) {
		if (!directories.has(path)) {
			const segments = path.split("/");
			const name = segments.pop();
			const promise = directory(segments.join("/")).then((parent) => {
				check();
				return parent.getDirectoryHandle(name);
			});
			directories.set(path, promise);
			promise.catch(() => directories.delete(path));
		}
		return directories.get(path);
	}
	async function file(key) {
		check();
		const segments = parts(key),
			name = segments.pop();
		try {
			const parent = await directory(segments.join("/"));
			const handle = await parent.getFileHandle(name);
			check();
			const value = await handle.getFile();
			check();
			return value;
		} catch (error) {
			if (absent(error)) return undefined;
			throw error;
		}
	}
	async function read(key, range, options) {
		check();
		options?.signal?.throwIfAborted();
		const value = await file(key);
		if (!value) return undefined;
		const blob = !range
			? value
			: "suffixLength" in range
				? value.slice(Math.max(0, value.size - range.suffixLength))
				: value.slice(range.offset, range.offset + range.length);
		const bytes = new Uint8Array(await blob.arrayBuffer());
		check();
		options?.signal?.throwIfAborted();
		return bytes;
	}
	const raw = {
		async get(key, options) {
			check();
			options?.signal?.throwIfAborted();
			if (!metadataNames.has(key.split("/").at(-1)))
				return read(key, null, options);
			if (!metadata.has(key)) {
				const promise = read(key, null, options);
				metadata.set(key, promise);
				promise.catch(() => metadata.delete(key));
			}
			return metadata.get(key);
		},
		getRange: (key, range, options) => read(key, range, options),
	};
	let consolidated;
	const rootFiles = new Set(
		rootEntries.filter((e) => e.kind === "file").map((e) => e.name),
	);
	if (consolidate) {
		for (const name of [".zmetadata", "zmetadata", "zarr.json"]) {
			if (!rootFiles.has(name)) continue;
			try {
				consolidated = await consolidate(
					raw,
					name === "zarr.json"
						? undefined
						: { format: "v2", metadataKey: name },
				);
				break;
			} catch {
				check(); /* Missing/stale consolidated metadata: discover nodes below. */
			}
		}
	}
	for (const node of consolidated?.contents() || [])
		nodes.set(node.path.replace(/^\//, ""), node);
	async function json(key) {
		const bytes = await raw.get(key);
		return bytes && JSON.parse(decoder.decode(bytes));
	}
	async function walk(handle, path, knownEntries) {
		check();
		if (nodes.get(path)?.kind === "array") return;
		visited++;
		// Consolidation may predate newly added images. Probe unlisted nodes and
		// stop at arrays, retaining the fast path for already indexed chunk trees.
		if (!nodes.has(path)) {
			const prefix = path ? `/${path}` : "";
			const v3 = await json(`${prefix}/zarr.json`);
			const kind =
				v3?.node_type ||
				((await json(`${prefix}/.zarray`))
					? "array"
					: (await json(`${prefix}/.zgroup`))
						? "group"
						: null);
			if (kind) nodes.set(path, { path: prefix || "/", kind });
			if (kind === "array") return;
		}
		if (Date.now() - lastProgress > 100) {
			lastProgress = Date.now();
			onProgress({
				directories: visited,
				nodes: nodes.size,
				path: path || root.name,
			});
		}
		for (const entry of knownEntries || (await entries(handle))) {
			check();
			const key = path ? `${path}/${entry.name}` : entry.name;
			if (entry.kind === "file") keys.add(key);
			else {
				directories.set(key, Promise.resolve(entry));
				await walk(entry, key);
			}
		}
	}
	await walk(root, "", rootEntries);
	check();
	return {
		directoryHandle: root,
		get: (consolidated || raw).get.bind(consolidated || raw),
		getRange: raw.getRange,
		contents: () => [...nodes.values()],
		// The renderer uses this listing to discover multipart Parquet files. Zarr
		// chunks intentionally stay unlisted and are resolved by key on demand.
		listKeys: () => [...keys],
	};
}
