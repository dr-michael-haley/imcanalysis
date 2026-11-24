import React, { useEffect, useMemo, useRef, useState } from "react";
import { embeddingCatalog, loadEmbedding } from "./embedding-data.mjs";
import {
	defaultEmbedding,
	cellKey,
	plotPoints,
	pointInPolygon,
} from "./embedding-model.mjs";
import { readValues, encodeOverlay } from "./overlays.mjs";
import { OverlayLegend } from "./CellControls.jsx";
import ColourControls from "./ColourControls.jsx";
import { download } from "./model.mjs";

export default function EmbeddingPanel({
	dataset,
	options,
	onChange,
	activeBinding,
	activeRegion,
	activeOverlay,
	geneLists,
	highlighted,
	onSelect,
	onFocus,
	selectionMode,
	onSelectionMode,
	onSelectionScope,
	onSelectionApi,
}) {
	const spec = { ...defaultEmbedding(), ...options };
	const tableKey =
		spec.table ||
		activeBinding?.table ||
		Object.keys(dataset?.tables || {})[0] ||
		"";
	const table = dataset?.tables[tableKey];
	const catalog = useMemo(
		() => embeddingCatalog(dataset, tableKey),
		[dataset, tableKey],
	);
	const projection = spec.projection || catalog[0]?.key || "",
		projectionInfo = catalog.find((p) => p.key === projection);
	const [loaded, setLoaded] = useState({}),
		[raw, setRaw] = useState({}),
		[genes, setGenes] = useState([]),
		[retry, setRetry] = useState(0),
		[localMode, setLocalMode] = useState("inspect"),
		[picked, setPicked] = useState(null),
		[hover, setHover] = useState("");
	const mode = selectionMode || localMode,
		setMode = onSelectionMode || setLocalMode;
	const job = useRef(null),
		canvas = useRef(null),
		plot = useRef([]),
		drag = useRef(null),
		paint = useRef(null);
	const change = (patch) => onChange({ ...spec, ...patch });
	const dataKey = JSON.stringify([tableKey, projection, spec.x, spec.y, retry]);
	useEffect(() => {
		const controller = new AbortController();
		job.current = controller;
		setLoaded({ loading: !!projection });
		setPicked(null);
		if (projection && dataset)
			loadEmbedding(
				dataset,
				tableKey,
				projection,
				spec.x,
				spec.y,
				controller.signal,
				(f) => {
					if (!controller.signal.aborted)
						setLoaded({ loading: true, progress: f });
				},
			)
				.then((data) => {
					if (!controller.signal.aborted) setLoaded({ data, key: dataKey });
				})
				.catch((e) => {
					if (!controller.signal.aborted) setLoaded({ error: e.message });
				});
		return () => controller.abort();
	}, [dataset, dataKey]);
	useEffect(() => {
		let live = true;
		setGenes([]);
		table
			?.loadVarIndex()
			.then((v) => {
				if (live) setGenes(v);
			})
			.catch(() => {});
		return () => {
			live = false;
		};
	}, [table]);
	const colourRequest = JSON.stringify([
		spec.colourBy,
		spec.field,
		spec.colourBy === "sum" ? geneLists.find((g) => g.id === spec.field) : null,
	]);
	useEffect(() => {
		const controller = new AbortController();
		setRaw({});
		if (
			table &&
			spec.colourBy !== "view" &&
			spec.colourBy !== "none" &&
			spec.field
		) {
			setRaw({ loading: true });
			readValues(
				table,
				{ source: spec.colourBy, field: spec.field },
				controller.signal,
				geneLists,
			)
				.then((data) => {
					if (!controller.signal.aborted) setRaw({ data });
				})
				.catch((e) => {
					if (!controller.signal.aborted) setRaw({ error: e.message });
				});
		}
		return () => controller.abort();
	}, [table, colourRequest]);
	const data = loaded.key === dataKey ? loaded.data : null;
	const display = useMemo(
		() =>
			data
				? plotPoints(data, {
						region: activeRegion || "",
						context: spec.context,
						limit: spec.limit,
					})
				: { points: [], current: 0, others: 0, invalid: 0 },
		[data, activeRegion, spec.context, spec.limit],
	);
	const presentation = JSON.stringify([
		spec.overrideMap,
		spec.scale,
		spec.colourmap,
		spec.reverse,
		spec.vmin,
		spec.vmax,
	]);
	const colourResult = useMemo(() => {
		if (!data) return {};
		try {
			let values, mode, colourBinding, label;
			if (spec.colourBy === "view") {
				if (activeBinding?.table !== tableKey)
					return {
						notice:
							"The selected view uses a different table. Select that table, or choose an independent colour field.",
					};
				if (activeOverlay?.error) return { error: activeOverlay.error };
				if (activeOverlay?.loading) return { loading: true };
				if (!activeOverlay?.values) return {};
				({ values, mode } = activeOverlay);
				colourBinding = {
					...activeBinding,
					...(spec.overrideMap
						? { colourmap: spec.colourmap, reverse: spec.reverse }
						: {}),
					vmin:
						spec.vmin ??
						activeBinding.vmin ??
						(activeOverlay.rawLow !== activeOverlay.rawHigh
							? activeOverlay.rawLow
							: null),
					vmax:
						spec.vmax ??
						activeBinding.vmax ??
						(activeOverlay.rawLow !== activeOverlay.rawHigh
							? activeOverlay.rawHigh
							: null),
				};
				label = activeOverlay.label;
			} else {
				if (raw.loading || raw.error) return raw;
				if (!raw.data?.values) return {};
				({ values, mode } = raw.data);
				colourBinding = { ...spec, table: tableKey, source: spec.colourBy };
				label = raw.data.geneSummary
					? `Σ ${raw.data.geneSummary.name}`
					: spec.field;
			}
			if (values.length !== data.ids.length)
				throw new Error(
					"Colour values and projection have different row counts.",
				);
			const rows = new Map(
				display.points.map((p) => [String(p.row), { value: values[p.row] }]),
			);
			const encoded = encodeOverlay(
				rows,
				display.points.map((p) => String(p.row)),
				mode,
				colourBinding.scale,
				colourBinding,
			);
			if (
				spec.colourBy === "view" &&
				!spec.overrideMap &&
				spec.vmin == null &&
				spec.vmax == null
			) {
				Object.assign(encoded, {
					colour: activeOverlay.colour,
					low: activeOverlay.low,
					high: activeOverlay.high,
					dataLow: activeOverlay.dataLow,
					dataHigh: activeOverlay.dataHigh,
					categories: activeOverlay.categories,
				});
			}
			return {
				...encoded,
				values,
				mode,
				binding: colourBinding,
				label,
				geneSummary: raw.data?.geneSummary,
			};
		} catch (e) {
			return { error: e.message };
		}
	}, [
		data,
		display,
		raw,
		activeOverlay,
		activeBinding,
		tableKey,
		presentation,
		spec.colourBy,
	]);

	useEffect(() => {
		const node = canvas.current;
		if (!node) return;
		const background = document.createElement("canvas");
		function draw() {
			const rect = node.getBoundingClientRect(),
				ratio = window.devicePixelRatio || 1;
			node.width = Math.max(1, Math.round(rect.width * ratio));
			node.height = Math.max(1, Math.round(rect.height * ratio));
			const ctx = node.getContext("2d");
			ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
			ctx.clearRect(0, 0, rect.width, rect.height);
			ctx.fillStyle = "#0c141a";
			ctx.fillRect(0, 0, rect.width, rect.height);
			plot.current = [];
			if (!display.points.length) return;
			let xmin = Infinity,
				xmax = -Infinity,
				ymin = Infinity,
				ymax = -Infinity;
			for (const p of display.points) {
				xmin = Math.min(xmin, p.x);
				xmax = Math.max(xmax, p.x);
				ymin = Math.min(ymin, p.y);
				ymax = Math.max(ymax, p.y);
			}
			const w = spec.square ? Math.min(rect.width, rect.height) : rect.width,
				h = spec.square ? w : rect.height,
				left = (rect.width - w) / 2,
				top = (rect.height - h) / 2;
			ctx.strokeStyle = "#3c4c57";
			ctx.strokeRect(left + 16, top + 16, w - 32, h - 32);
			for (const point of display.points) {
				const x =
						left + 24 + ((point.x - xmin) / (xmax - xmin || 1)) * (w - 48),
					y = top + h - 24 - ((point.y - ymin) / (ymax - ymin || 1)) * (h - 48);
				const selected = highlighted.has(cellKey(point.region, point.id));
				const c = colourResult.colour?.(colourResult.values?.[point.row]) || [
					129, 216, 193, 255,
				];
				ctx.globalAlpha = selected
					? 1
					: point.context
						? 0.28
						: highlighted.size
							? 0.45
							: 1;
				ctx.fillStyle = selected
					? "#ffffff"
					: `rgba(${c[0]},${c[1]},${c[2]},${c[3] / 255})`;
				ctx.beginPath();
				ctx.arc(x, y, selected ? 4 : 2.3, 0, Math.PI * 2);
				ctx.fill();
				if (selected) {
					ctx.strokeStyle = "#81d8c1";
					ctx.lineWidth = 1.5;
					ctx.stroke();
				}
				plot.current.push({ ...point, px: x, py: y });
			}
			ctx.globalAlpha = 1;
			background.width = node.width;
			background.height = node.height;
			background.getContext("2d").drawImage(node, 0, 0);
			drawGesture();
		}
		function drawGesture() {
			const ctx = node.getContext("2d"),
				ratio = window.devicePixelRatio || 1;
			ctx.setTransform(1, 0, 0, 1, 0, 0);
			ctx.clearRect(0, 0, node.width, node.height);
			ctx.drawImage(background, 0, 0);
			ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
			if (drag.current?.mode === "lasso") {
				ctx.strokeStyle = "#ffffff";
				ctx.setLineDash([4, 3]);
				ctx.beginPath();
				drag.current.points.forEach(([x, y], i) =>
					i ? ctx.lineTo(x, y) : ctx.moveTo(x, y),
				);
				ctx.stroke();
				ctx.setLineDash([]);
			}
		}
		paint.current = drawGesture;
		const observer = new ResizeObserver(draw);
		observer.observe(node);
		draw();
		return () => observer.disconnect();
	}, [display, colourResult, highlighted, spec.square]);
	const position = (e) => {
		const r = e.currentTarget.getBoundingClientRect();
		return [e.clientX - r.left, e.clientY - r.top];
	};
	const selectPoints = (
		points,
		action = mode === "click" ? "toggle" : mode === "lasso" ? "add" : "replace",
	) =>
		onSelect(
			points.map((p) => ({
				...p,
				table: tableKey,
				label: colourResult.label,
				value: colourResult.values?.[p.row],
			})),
			action,
		);
	useEffect(() => {
		onSelectionApi?.({
			kind: "embedding",
			select: () => selectPoints(plot.current, "add"),
		});
		return () => onSelectionApi?.(null);
	}, [data, display, colourResult, onSelectionApi]);
	const nearest = ([x, y]) => {
		let found,
			d = 12;
		for (const p of plot.current) {
			const n = Math.hypot(p.px - x, p.py - y);
			if (n < d) {
				d = n;
				found = p;
			}
		}
		return found;
	};
	const fields =
		spec.colourBy === "gene"
			? genes
			: spec.colourBy === "sum"
				? geneLists.map((g) => g.id)
				: table?.getObsColumnNames() || [];
	return (
		<div className="embedding-panel">
			<div className="embedding-settings">
				<div className="embedding-selectors">
					<label>
						Table
						<select
							aria-label="Embedding table"
							value={spec.table}
							onChange={(e) =>
								change({
									table: e.target.value,
									projection: "",
									x: 0,
									y: 1,
									field: "",
									vmin: null,
									vmax: null,
								})
							}
						>
							<option value="">Selected cell table</option>
							{Object.keys(dataset?.tables || {}).map((k) => (
								<option key={k}>{k}</option>
							))}
						</select>
					</label>
					<label>
						Projection
						<select
							aria-label="Embedding projection"
							value={projection}
							onChange={(e) =>
								change({ projection: e.target.value, x: 0, y: 1 })
							}
							disabled={!catalog.length}
						>
							{!catalog.length && (
								<option value="">No embeddings in this table</option>
							)}
							{catalog.map((p) => (
								<option key={p.key} value={p.key}>
									{p.key} · {p.dimensions} dimensions
								</option>
							))}
						</select>
					</label>
					<div className="two-columns">
						{["x", "y"].map((axis) => (
							<label key={axis}>
								{axis.toUpperCase()} dimension
								<select
									aria-label={`Embedding ${axis} dimension`}
									value={spec[axis]}
									onChange={(e) => change({ [axis]: Number(e.target.value) })}
								>
									{Array.from(
										{ length: projectionInfo?.dimensions || 2 },
										(_, i) => (
											<option value={i} key={i}>
												{i + 1}
											</option>
										),
									)}
								</select>
							</label>
						))}
					</div>
					<label>
						Colour by
						<select
							aria-label="Embedding colour by"
							value={spec.colourBy}
							onChange={(e) =>
								change({
									colourBy: e.target.value,
									field: "",
									vmin: null,
									vmax: null,
								})
							}
						>
							<option value="view">Selected view</option>
							<option value="gene">Gene expression · X</option>
							<option value="sum">Gene list · sum of X</option>
							<option value="obs">Cell annotation · obs</option>
							<option value="none">Single colour</option>
						</select>
					</label>
					{!["view", "none"].includes(spec.colourBy) && (
						<label>
							Field
							<select
								aria-label="Embedding colour field"
								value={spec.field}
								onChange={(e) =>
									change({ field: e.target.value, vmin: null, vmax: null })
								}
							>
								<option value="">Choose a field</option>
								{fields.map((k) => (
									<option key={k} value={k}>
										{spec.colourBy === "sum"
											? geneLists.find((g) => g.id === k)?.name
											: k}
									</option>
								))}
							</select>
						</label>
					)}
					{!["view", "none"].includes(spec.colourBy) &&
						colourResult.mode !== "categorical" && (
							<label>
								Expression scale
								<select
									aria-label="Embedding expression scale"
									value={spec.scale}
									onChange={(e) => change({ scale: e.target.value })}
								>
									<option value="linear">Linear</option>
									<option value="log1p">Log1p</option>
								</select>
							</label>
						)}
					<label className="checkbox-label">
						<input
							type="checkbox"
							checked={spec.square}
							onChange={(e) => change({ square: e.target.checked })}
						/>
						Square plot
					</label>
					<label className="checkbox-label">
						<input
							type="checkbox"
							checked={spec.context}
							onChange={(e) => change({ context: e.target.checked })}
						/>
						Other regions in this table
					</label>
					{spec.context && (
						<label>
							Context limit
							<select
								aria-label="Embedding context limit"
								value={spec.limit}
								onChange={(e) => change({ limit: Number(e.target.value) })}
							>
								{[10000, 50000, 100000, 200000].map((n) => (
									<option key={n} value={n}>
										{n.toLocaleString()}
									</option>
								))}
							</select>
						</label>
					)}
				</div>
				{colourResult.mode === "continuous" && (
					<ColourControls
						prefix="Embedding "
						value={
							spec.colourBy === "view"
								? { ...colourResult.binding, vmin: spec.vmin, vmax: spec.vmax }
								: spec
						}
						result={colourResult}
						onChange={(v) =>
							spec.colourBy === "view"
								? change({
										vmin: v.vmin,
										vmax: v.vmax,
										colourmap: v.colourmap,
										reverse: v.reverse,
										overrideMap: true,
									})
								: onChange(v)
						}
					/>
				)}
				{colourResult.mode && <OverlayLegend result={colourResult} />}
				{colourResult.geneSummary?.missing.length > 0 && (
					<p className="warning">
						Missing genes: {colourResult.geneSummary.missing.join(", ")}
					</p>
				)}
			</div>
			<div className="embedding-chart-area">
				<div className="drawer-actions">
					{["inspect", "lasso"].map((m) => (
						<button
							key={m}
							aria-pressed={mode === m}
							onClick={() => setMode(m)}
						>
							{m === "inspect" ? "Inspect point" : "Lasso cells"}
						</button>
					))}
					<button
						disabled={!highlighted.size}
						onClick={() => {
							onSelect([]);
							setPicked(null);
						}}
					>
						Clear embedding selection
					</button>
					<button disabled={!picked} onClick={() => onFocus(picked)}>
						Centre selected cell
					</button>
					<button
						disabled={!data}
						onClick={() =>
							canvas.current.toBlob(
								(b) => b && download("merscope-embedding.png", b, "image/png"),
							)
						}
					>
						Export embedding PNG
					</button>
					{loaded.loading ? (
						<button
							onClick={() => {
								job.current?.abort();
								setLoaded({
									error:
										"Embedding loading cancelled. Use Reload embedding to retry.",
								});
							}}
						>
							Cancel embedding load
						</button>
					) : (
						<button
							disabled={!projection}
							onClick={() => setRetry((n) => n + 1)}
						>
							Reload embedding
						</button>
					)}
				</div>
				{!catalog.length && (
					<p className="muted">
						No stored embedding is available in {tableKey || "this table"}.
						Choose another table. Projections such as X_umap, X_tsne or X_pca
						must already be present in AnnData obsm.
					</p>
				)}
				{projection === "spatial" && (
					<p className="muted">
						Spatial coordinates from obsm/spatial; this is not a UMAP or t-SNE
						projection.
					</p>
				)}
				{(loaded.error || colourResult.error) && (
					<p className="error" role="alert">
						{loaded.error || colourResult.error}
					</p>
				)}
				{colourResult.error && (
					<button onClick={() => change({ vmin: null, vmax: null })}>
						Reset embedding colour limits
					</button>
				)}
				{colourResult.notice && (
					<p className="warning">{colourResult.notice}</p>
				)}
				<canvas
					ref={canvas}
					className="embedding-canvas"
					aria-label="Linked cell embedding"
					tabIndex={0}
					onKeyDown={(e) => {
						if (e.key === "Escape") {
							drag.current = null;
							setMode("inspect");
							paint.current?.();
						}
					}}
					onPointerDown={(e) => {
						if (e.button !== 0) return;
						onSelectionScope?.("embedding");
						drag.current = { mode, points: [position(e)] };
						e.currentTarget.setPointerCapture(e.pointerId);
					}}
					onPointerMove={(e) => {
						const pos = position(e);
						if (drag.current?.mode === "lasso") {
							if (drag.current.points.length < 5000)
								drag.current.points.push(pos);
							paint.current?.();
						} else {
							const p = nearest(pos);
							setHover(
								p
									? `${p.region} · Cell ${p.id} · table row ${p.row}${colourResult.values ? ` · ${colourResult.label}: ${colourResult.values[p.row]}` : ""}`
									: "",
							);
						}
					}}
					onPointerUp={(e) => {
						const gesture = drag.current;
						drag.current = null;
						if (!gesture) return;
						if (gesture.mode === "lasso" && gesture.points.length > 2) {
							selectPoints(
								plot.current.filter((p) =>
									pointInPolygon(p.px, p.py, gesture.points),
								),
							);
							setPicked(null);
						} else {
							const p = nearest(position(e));
							if (p) {
								setPicked(p);
								selectPoints([p]);
							}
						}
						paint.current?.();
					}}
					onPointerCancel={() => {
						drag.current = null;
						paint.current?.();
					}}
				/>
				<p className="muted embedding-status" role="status">
					{loaded.loading
						? `Reading projection… ${Math.round((loaded.progress || 0) * 100)}%`
						: colourResult.loading
							? "Reading colours…"
							: `${display.current.toLocaleString()} cells in current region${display.currentShown < display.current ? ` (${display.currentShown.toLocaleString()} displayed; sampled)` : ""} · ${spec.context ? Math.min(display.others, spec.limit).toLocaleString() : 0} context cells · ${highlighted.size.toLocaleString()} selected`}
					{display.invalid > 0
						? ` · ${display.invalid} nonfinite coordinates omitted`
						: ""}
				</p>
				<p className="embedding-hover">
					{hover ||
						"Click a point to inspect its cell, or draw a lasso to highlight cells in the tissue views. Esc returns to inspect."}
				</p>
			</div>
		</div>
	);
}
