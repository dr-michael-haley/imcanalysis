const DB = "sbt-merscope-explorer";
function open() {
	return new Promise((resolve, reject) => {
		const req = indexedDB.open(DB, 1);
		req.onupgradeneeded = () =>
			req.result.createObjectStore("workspaces", { keyPath: "key" });
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => reject(req.error);
	});
}
async function transaction(mode, action) {
	const db = await open();
	return new Promise((resolve, reject) => {
		const tx = db.transaction("workspaces", mode),
			req = action(tx.objectStore("workspaces"));
		tx.oncomplete = () => {
			db.close();
			resolve(req.result);
		};
		tx.onerror = tx.onabort = () => {
			db.close();
			reject(tx.error || new Error("Local storage unavailable."));
		};
	});
}
export const readLocal = (key) =>
	transaction("readonly", (store) => store.get(key));
export const writeLocal = (record) =>
	transaction("readwrite", (store) => store.put(record));
export async function directoryIdentity(handle) {
	const records = await transaction("readonly", (store) => store.getAll());
	for (const record of records) {
		if (!record.directoryHandle) continue;
		try {
			if (await handle.isSameEntry(record.directoryHandle)) return record.key;
		} catch {
			/* An unavailable old handle must not prevent opening another store. */
		}
	}
	return `directory:${crypto.randomUUID()}`;
}
export async function recentLocal() {
	const records = await transaction("readonly", (store) => store.getAll());
	return records
		.sort((a, b) => b.savedAt - a.savedAt)
		.slice(0, 5)
		.map(({ key, name, savedAt }) => ({ key, name, savedAt }));
}
export async function fileIdentity(files) {
	const inventory = files
		.map((file) => [
			file.webkitRelativePath || file.name,
			file.size,
			file.lastModified,
		])
		.sort(([a], [b]) => a.localeCompare(b));
	const bytes = new TextEncoder().encode(JSON.stringify(inventory));
	const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
	return [...digest].map((v) => v.toString(16).padStart(2, "0")).join("");
}
