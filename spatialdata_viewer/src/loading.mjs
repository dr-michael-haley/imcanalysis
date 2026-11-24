// Observe reads without changing the store's listing, range or caching semantics.
// Disposing releases consumers immediately; a Blob read already handed to the OS
// may finish later, but cannot publish progress or revive a closed dataset.
export function createReadTracker(publish, delay = 120) {
	const pending = new Map();
	let sequence = 0,
		reads = 0,
		bytes = 0,
		startedAt = 0;
	let timer = null,
		disposed = false;
	const cancelled = () => new DOMException("Loading cancelled", "AbortError");
	function update() {
		if (timer !== null || disposed) return;
		timer = setTimeout(() => {
			timer = null;
			if (!disposed)
				publish({
					pending: pending.size,
					reads,
					bytes,
					startedAt,
					path: pending.values().next().value?.path || "",
				});
		}, delay);
	}
	function read(method, args) {
		if (disposed) return Promise.reject(cancelled());
		if (!pending.size) startedAt = Date.now();
		const id = ++sequence;
		return new Promise((resolve, reject) => {
			pending.set(id, { path: String(args[0]), reject });
			update();
			Promise.resolve()
				.then(() => {
					if (disposed) throw cancelled();
					return method(...args);
				})
				.then(
					(value) => finish(null, value),
					(error) => finish(error),
				);
			function finish(error, value) {
				if (!pending.delete(id)) return;
				if (!error && value !== undefined) {
					reads++;
					bytes += value.byteLength || 0;
				}
				update();
				if (error) reject(error);
				else resolve(value);
			}
		});
	}
	return {
		wrap(store) {
			return new Proxy(store, {
				get(target, property) {
					const value = Reflect.get(target, property, target);
					if (typeof value !== "function") return value;
					const method = value.bind(target);
					return property === "get" || property === "getRange"
						? (...args) => read(method, args)
						: method;
				},
			});
		},
		dispose() {
			disposed = true;
			clearTimeout(timer);
			for (const item of pending.values()) item.reject(cancelled());
			pending.clear();
		},
	};
}
