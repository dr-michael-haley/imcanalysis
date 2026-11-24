import React, {
	useCallback,
	useMemo,
	useState,
	useEffect,
	useRef,
} from "react";
import { SpatialCanvasViewer, SpatialFeatureTooltip } from "@spatialdata/vis";
import ChannelControls from "./ChannelControls.jsx";
import { pretty } from "./model.mjs";
import { cellKey } from "./embedding-model.mjs";
import {
	displayedEntries,
	figureCamera,
	defaultFigure,
} from "./view-tools.mjs";
import { paneOverlaySvg } from "./pane-overlays.mjs";
import PaneLegendData, { paneLegendGroups } from "./PaneLegendData.jsx";
import { shapeCentres, selectInPane } from "./spatial-selection.mjs";

export default function ViewerPane({
	pane,
	dataset,
	system,
	active,
	linked,
	overlays,
	geneLists,
	onActivate,
	onCamera,
	onPick,
	onStatus,
	channelTarget,
	onChannels,
	highlighted,
	figure = defaultFigure(),
	figureReference,
	onGroups,
	selectionMode = "inspect",
	onSelectCells,
	onSelectionScope,
	onSelectionApi,
}) {
	const [error, setError] = useState("");
	const container = useRef(null),
		drag = useRef(null),
		[size, setSize] = useState({ width: 1, height: 1 }),
		[gesture, setGesture] = useState([]),
		[loadedGroups, setLoadedGroups] = useState({});
	const isFigure = figure.mode === "figure";
	const currentDataset = useRef(dataset);
	currentDataset.current = dataset;
	useEffect(
		() => () => {
			currentDataset.current = null;
		},
		[],
	);
	useEffect(() => {
		const node = container.current;
		if (!node) return;
		const observer = new ResizeObserver(() => {
			const r = node.getBoundingClientRect();
			setSize({ width: r.width, height: r.height });
		});
		observer.observe(node);
		return () => observer.disconnect();
	}, []);
	const groupChanged = useCallback(
		(id, group) =>
			setLoadedGroups((old) => {
				if (JSON.stringify(old[id]) === JSON.stringify(group)) return old;
				const next = { ...old };
				if (group) next[id] = group;
				else delete next[id];
				return next;
			}),
		[],
	);
	const groups = useMemo(
		() => paneLegendGroups(pane, overlays, loadedGroups),
		[pane, overlays, loadedGroups],
	);
	const groupSignature = JSON.stringify(groups);
	useEffect(() => {
		onGroups?.(pane.id, groups);
	}, [pane.id, groupSignature, onGroups]);
	const camera =
		pane.viewState && isFigure
			? figureCamera(
					pane.viewState,
					size.width,
					figureReference?.width || size.width,
					figure,
				)
			: pane.viewState;
	const cameraChanged = (next) => {
		if (isFigure && pane.viewState)
			next = {
				...next,
				zoom: figure.physicalLock
					? pane.viewState.zoom
					: next.zoom -
						Math.log2(size.width / (figureReference?.width || size.width)),
			};
		onCamera(pane.id, next);
	};
	const selectCells = useCallback(
		async (polygon) => {
			if (!camera || isFigure) return;
			try {
				const entries = displayedEntries(pane).filter(
					(e) => e.source.elementType === "shapes",
				);
				if (!entries.length)
					throw new Error(
						"Add a Shapes layer to select displayed cells or use a lasso. Label masks support click selection.",
					);
				const cells = (
					await Promise.all(
						entries.map((e) =>
							shapeCentres(
								dataset.shapes[e.source.elementKey],
								e.source.elementKey,
								system,
							),
						),
					)
				).flat();
				if (currentDataset.current !== dataset) return;
				onSelectCells?.(
					selectInPane(cells, camera, size.width, size.height, polygon),
					"add",
				);
				setError("");
			} catch (e) {
				setError(e.message);
			}
		},
		[
			dataset,
			pane,
			system,
			camera?.zoom,
			camera?.target,
			size.width,
			size.height,
			isFigure,
			onSelectCells,
		],
	);
	useEffect(() => {
		if (active)
			onSelectionApi?.({ kind: "tissue", select: () => selectCells() });
	}, [active, selectCells, onSelectionApi]);
	const selected = pane.renderStack.entries.find(
		(e) => e.id === pane.selectedId,
	);
	const stack = useMemo(
		() => ({
			...pane.renderStack,
			entries: pane.renderStack.entries.map((sourceEntry) => {
				const entry = displayedEntries(pane).some(
					(e) => e.id === sourceEntry.id,
				)
					? sourceEntry
					: { ...sourceEntry, visible: false };
				if (entry.source.elementType === "labels" && highlighted.size) {
					const fillColorByFeatureId = {
						...entry.props.featureState?.fillColorByFeatureId,
					};
					for (const value of highlighted) {
						const [region, id] = JSON.parse(value);
						if (region === entry.source.elementKey)
							fillColorByFeatureId[id] = [255, 255, 255, 255];
					}
					return {
						...entry,
						props: {
							...entry.props,
							featureState: {
								...entry.props.featureState,
								fillColorByFeatureId,
							},
						},
					};
				}
				if (entry.props.geneListId === undefined) return entry;
				return {
					...entry,
					props: {
						...entry.props,
						featureNames:
							geneLists.find((g) => g.id === entry.props.geneListId)?.genes ||
							[],
					},
				};
			}),
		}),
		[pane.renderStack, pane.display, geneLists, highlighted],
	);
	const resolver = useMemo(() => {
		const buffers = new Map();
		return ({ layerId, featureIds }) => {
			if (!featureIds) return undefined; // Label rasters use pixel IDs, not shape order.
			const base = overlays[`${pane.id}/${layerId}`]?.resolve?.(featureIds);
			if (!highlighted.size) return base;
			const entry = pane.renderStack.entries.find((e) => e.id === layerId);
			if (!entry) return base;
			let cache = buffers.get(layerId);
			if (!cache) {
				cache = new WeakMap();
				buffers.set(layerId, cache);
			}
			if (!cache.has(featureIds)) {
				const colors = base
					? new Uint8Array(base.colors)
					: new Uint8Array(featureIds.length * 4);
				featureIds.forEach((id, i) => {
					if (highlighted.has(cellKey(entry.source.elementKey, id)))
						colors.set([255, 255, 255, 255], i * 4);
					else if (base)
						colors[i * 4 + 3] = Math.round(colors[i * 4 + 3] * 0.45);
					else
						colors.set(
							[...(entry.props.fillColor || [129, 216, 193]).slice(0, 3), 65],
							i * 4,
						);
				});
				cache.set(featureIds, { colors, count: featureIds.length });
			}
			return cache.get(featureIds);
		};
	}, [overlays, pane.id, pane.renderStack, highlighted]);
	const tooltip = useCallback(
		({ tooltip, clientX, clientY }) => {
			let data = tooltip;
			if (tooltip.elementType === "shapes") {
				const overlay =
					overlays[
						`${pane.id}/${tooltip.layerId || `shapes:${tooltip.elementKey}`}`
					];
				const id =
					tooltip.items.find((i) => i.label === "feature_id")?.value ??
					tooltip.title;
				const row = overlay?.rows?.get(String(id));
				const items = [{ label: "Cell ID", value: String(id) }];
				if (overlay?.binding) {
					items.push(
						{ label: "Table", value: overlay.binding.table },
						{ label: "Row", value: row ? String(row.row) : "Unpaired" },
					);
					if (overlay.binding.field)
						items.push({ label: overlay.label, value: pretty(row?.value) });
				}
				data = { ...tooltip, items };
			}
			return (
				<div className="feature-tooltip-wrap">
					<SpatialFeatureTooltip x={clientX} y={clientY} tooltip={data} />
				</div>
			);
		},
		[overlays, pane.id],
	);
	const deckProps = useMemo(
		() => ({
			style: { backgroundColor: isFigure ? figure.background : "#111111" },
			controller:
				isFigure && figure.physicalLock
					? {
							scrollZoom: false,
							doubleClickZoom: false,
							touchZoom: false,
							keyboard: false,
						}
					: true,
			onError: (e) => setError(e.message || String(e)),
		}),
		[isFigure, figure.physicalLock, figure.background],
	);
	return (
		<section
			className={`viewer-pane ${active ? "active" : ""}`}
			data-pane-id={pane.id}
			aria-label={pane.name}
			onPointerDownCapture={() => {
				onActivate(pane.id);
				onSelectionScope?.("tissue");
			}}
		>
			<button
				className="pane-heading"
				onClick={() => onActivate(pane.id)}
				aria-pressed={active}
				aria-label={`Select ${pane.name}`}
			>
				<span>{pane.name}</span>
				<small>
					{active ? "Editing this view" : "Select to edit"} ·{" "}
					{linked ? "Linked" : "Independent"}
				</small>
			</button>
			<div
				className="pane-canvas"
				ref={container}
				style={isFigure ? { background: figure.background } : undefined}
			>
				<SpatialCanvasViewer
					spatialData={dataset}
					coordinateSystem={system}
					renderStack={stack}
					autoFit={!linked || active}
					viewState={camera}
					onViewStateChange={cameraChanged}
					onFeatureClick={
						isFigure ? undefined : (event) => onPick(pane.id, event)
					}
					onEntryStatusChange={(id, value) => onStatus(pane.id, id, value)}
					hoverTooltipMode="simple"
					deckProps={deckProps}
					featureColorResolver={resolver}
					renderTooltip={tooltip}
				>
					<PaneLegendData
						pane={pane}
						dataset={dataset}
						geneLists={geneLists}
						onGroup={groupChanged}
					/>
					{active && selected?.source.elementType === "image" && (
						<ChannelControls
							entry={selected}
							target={channelTarget}
							onChange={onChannels}
						/>
					)}
				</SpatialCanvasViewer>
				{camera && (
					<svg
						className="pane-furniture"
						aria-label={`${pane.name} overlays`}
						width={size.width}
						height={size.height}
						viewBox={`0 0 ${size.width} ${size.height}`}
						dangerouslySetInnerHTML={{
							__html: paneOverlaySvg(
								pane,
								groups,
								figure,
								size.width,
								size.height,
								camera.zoom,
								pane.id,
							),
						}}
					/>
				)}
				{selectionMode === "lasso" && !isFigure && (
					<svg
						className="tissue-lasso"
						aria-label={`Lasso cells in ${pane.name}`}
						width="100%"
						height="100%"
						onPointerDown={(e) => {
							const r = e.currentTarget.getBoundingClientRect();
							drag.current = [[e.clientX - r.left, e.clientY - r.top]];
							setGesture(drag.current);
							e.currentTarget.setPointerCapture(e.pointerId);
						}}
						onPointerMove={(e) => {
							if (!drag.current) return;
							const r = e.currentTarget.getBoundingClientRect();
							drag.current = [
								...drag.current,
								[e.clientX - r.left, e.clientY - r.top],
							];
							setGesture(drag.current);
						}}
						onPointerUp={() => {
							const polygon = drag.current;
							drag.current = null;
							setGesture([]);
							if (polygon?.length > 2) void selectCells(polygon);
						}}
						onPointerCancel={() => {
							drag.current = null;
							setGesture([]);
						}}
					>
						<polyline
							points={gesture.map((p) => p.join(",")).join(" ")}
							fill="none"
							stroke="#ffffff"
							strokeWidth="1.5"
							strokeDasharray="4 3"
						/>
					</svg>
				)}
				{error && (
					<p className="canvas-error" role="alert">
						{error}
					</p>
				)}
			</div>
		</section>
	);
}
