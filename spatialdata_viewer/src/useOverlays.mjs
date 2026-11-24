import { useEffect, useMemo, useRef, useState } from "react";
import { defaultBinding, readOverlay, styleOverlay } from "./overlays.mjs";

// Per-pane requests survive camera changes and edits in another pane. Cancellation
// prevents an old gene/table request from repainting a newer choice.
export function useOverlays(dataset, panes, geneLists) {
	const running = useRef(new Map()),
		owner = useRef(dataset);
	const [results, setResults] = useState({});
	const specs = dataset
		? panes.flatMap((p) =>
				p.renderStack.entries
					.filter(
						(e) =>
							e.source.elementType === "shapes" &&
							(e.visible || e.id === p.selectedId),
					)
					.map((e) => {
						const binding = p.cellBindings[e.id] || defaultBinding(dataset, e);
						const list =
							binding.source === "sum"
								? geneLists.find((g) => g.id === binding.field)
								: undefined;
						return {
							id: `${p.id}/${e.id}`,
							entry: e,
							binding: {
								table: binding.table,
								idColumn: binding.idColumn,
								source: binding.source,
								field: binding.field,
							},
							list,
						};
					}),
			)
		: [];
	const signature = JSON.stringify(
		specs.map((s) => ({ ...s, entry: { source: s.entry.source } })),
	);
	const requested = useMemo(() => JSON.parse(signature), [signature]);
	useEffect(() => {
		if (owner.current !== dataset) {
			for (const task of running.current.values()) task.controller.abort();
			running.current.clear();
			owner.current = dataset;
			setResults({});
		}
		const wanted = new Set(requested.map((s) => s.id));
		setResults((previous) => {
			if (Object.keys(previous).every((id) => wanted.has(id))) return previous;
			return Object.fromEntries(
				Object.entries(previous).filter(([id]) => wanted.has(id)),
			);
		});
		for (const [id, task] of running.current)
			if (!wanted.has(id)) {
				task.controller.abort();
				running.current.delete(id);
			}
		for (const spec of requested) {
			const key = JSON.stringify(spec);
			if (running.current.get(spec.id)?.key === key) continue;
			running.current.get(spec.id)?.controller.abort();
			const controller = new AbortController();
			running.current.set(spec.id, { key, controller });
			const publish = (result) => {
				if (!controller.signal.aborted)
					setResults((prev) => ({ ...prev, [spec.id]: { key, ...result } }));
			};
			if (!spec.binding.table) {
				publish({});
				continue;
			}
			publish({ loading: true });
			readOverlay(
				dataset,
				spec.entry,
				spec.binding,
				controller.signal,
				spec.list ? [spec.list] : [],
			)
				.then(publish)
				.catch((error) => publish({ error: error.message }));
		}
	}, [dataset, requested]);
	useEffect(
		() => () => {
			for (const task of running.current.values()) task.controller.abort();
		},
		[],
	);
	const presentation = JSON.stringify(
		panes.map((p) => ({ id: p.id, cellBindings: p.cellBindings })),
	);
	return useMemo(() => {
		const settings = JSON.parse(presentation);
		return Object.fromEntries(
			requested.map((spec) => {
				const result = results[spec.id];
				if (result?.key !== JSON.stringify(spec) || owner.current !== dataset)
					return [spec.id, { loading: true }];
				if (result.loading || result.error || !result.rows)
					return [spec.id, result];
				const pane = settings.find((p) => spec.id.startsWith(p.id + "/"));
				const binding = pane?.cellBindings[
					spec.id.slice(pane.id.length + 1)
				] || { ...spec.binding, scale: "linear" };
				try {
					return [spec.id, styleOverlay(result, binding)];
				} catch (error) {
					return [spec.id, { error: error.message, dataLow: 0, dataHigh: 1 }];
				}
			}),
		);
	}, [results, requested, dataset, presentation]);
}
