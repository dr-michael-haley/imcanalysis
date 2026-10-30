import React, {
	Component,
	useCallback,
	useMemo,
	useRef,
	useState,
} from "react";
import { createRoot } from "react-dom/client";
import { readZarr } from "@spatialdata/core";
import { renderStackSchema } from "@spatialdata/layers";
import {
	SpatialCanvasViewer,
	ensureWorkers,
	getChunkCache,
} from "@spatialdata/vis";
import parquetWorkerUrl from "./parquet.worker.js?worker&url";
import { FolderStore, ZipStore } from "zarrextra";
import ChannelControls from "./ChannelControls.jsx";
import {
	createWorkspace,
	download,
	elementKind,
	elementKinds,
	elementsFor,
	initialStack,
	pretty,
	validateWorkspace,
	rgb,
	hex,
} from "./model.mjs";
import "./style.css";

ensureWorkers({
	parquet: { workerUrl: parquetWorkerUrl },
	codec: { chunkCacheMaxBytes: 256 * 1024 * 1024 },
});

class ViewerBoundary extends Component {
	state = { error: null };
	static getDerivedStateFromError(error) {
		return { error: error.message };
	}
	render() {
		return this.state.error ? (
			<div className="empty-state">
				<h2>Unable to render this view</h2>
				<p role="alert">{this.state.error}</p>
				<p>Close the dataset and reopen it to retry.</p>
			</div>
		) : (
			this.props.children
		);
	}
}
function Section({ number, title, children, open = true }) {
	return (
		<details className="section" open={open}>
			<summary>
				<span className="step">{number}</span>
				{title}
			</summary>
			<div className="section-body">{children}</div>
		</details>
	);
}
function App() {
	const [dataset, setDataset] = useState(null);
	const [source, setSource] = useState("");
	const [sourceInfo, setSourceInfo] = useState("");
	const [status, setStatus] = useState("");
	const [error, setError] = useState("");
	const [tab, setTab] = useState("Layers");
	const [coordinateSystem, setCoordinateSystem] = useState("");
	const [stack, setStack] = useState({ schemaVersion: 1, entries: [] });
	const [viewState, setViewState] = useState(null);
	const [selectedId, setSelectedId] = useState("");
	const [selection, setSelection] = useState(null);
	const [filter, setFilter] = useState("");
	const [channelTarget, setChannelTarget] = useState(null);
	const [layerStatuses, setLayerStatuses] = useState({});
	const [readNotices, setReadNotices] = useState([]);
	const [gpuError, setGpuError] = useState("");
	const [ready, setReady] = useState(false);
	const request = useRef(0);
	const folderInput = useRef(null);
	const zipInput = useRef(null);
	const workspaceInput = useRef(null);
	const viewport = useRef(null);
	const pendingExport = useRef(false);
	const selected = stack.entries.find((entry) => entry.id === selectedId);
	const elements = useMemo(
		() => (dataset ? elementsFor(dataset, coordinateSystem) : []),
		[dataset, coordinateSystem],
	);
	const associations =
		selected && dataset
			? dataset.getAssociatedTables(
					elementKind(selected.source.elementType),
					selected.source.elementKey,
				)
			: [];
	const annotationColumns = associations[0]?.[1].getObsColumnNames() || [];
	const enabled = stack.entries.filter((entry) => entry.visible).length;
	const updateEntry = useCallback(
		(id, patch) =>
			setStack((previous) => ({
				...previous,
				entries: previous.entries.map((entry) =>
					entry.id === id
						? {
								...entry,
								...patch,
								props: patch.props
									? { ...entry.props, ...patch.props }
									: entry.props,
							}
						: entry,
				),
			})),
		[],
	);
	const changeChannels = useCallback(
		(channels) => updateEntry(selectedId, { props: { channels } }),
		[selectedId, updateEntry],
	);
	const entryStatus = useCallback(
		(id, value) =>
			setLayerStatuses((previous) => {
				if (previous[id] === value) return previous;
				const next = { ...previous };
				if (value) next[id] = value;
				else delete next[id];
				return next;
			}),
		[],
	);
	const onFeatureClick = useCallback((event) => {
		setSelection({
			kind: event.elementKind,
			element: event.spatialElement.key,
			id: event.featureId,
			rowIndex: event.rowIndex,
			tooltip: event.tooltip,
		});
		setTab("Inspect");
	}, []);
	const deckProps = useMemo(
		() => ({
			onError: (failure) => setGpuError(failure.message || String(failure)),
			onAfterRender: () => {
				setReady(true);
				if (!pendingExport.current) return;
				pendingExport.current = false;
				const canvas = viewport.current?.querySelector("canvas");
				if (canvas)
					canvas.toBlob((blob) => {
						if (blob) download("spatialdata-view.png", blob, "image/png");
					}, "image/png");
			},
		}),
		[],
	);
	function resetView(sdata, system) {
		const next = initialStack(sdata, system);
		// Geometry-only datasets still open a useful first layer.
		if (!next.entries.some((entry) => entry.visible) && next.entries[0])
			next.entries[0].visible = true;
		setCoordinateSystem(system);
		setStack(next);
		setSelectedId(next.entries.find((entry) => entry.visible)?.id || "");
		setViewState(null);
		setSelection(null);
		setLayerStatuses({});
		setGpuError("");
		setFilter("");
		setReady(false);
	}
	async function openStore(makeStore, name, info) {
		const generation = ++request.current;
		setStatus(`Opening ${name}…`);
		setError("");
		// Release the previous rendering tree before opening a second store.
		setDataset(null);
		setSelection(null);
		setLayerStatuses({});
		setReadNotices([]);
		getChunkCache()?.clear();
		try {
			const store = await makeStore();
			if (generation !== request.current) return;
			const skipped = [];
			const sdata = await readZarr(store, undefined, (path, failure) =>
				skipped.push(`${path}: ${failure.message || failure}`),
			);
			if (generation !== request.current) return;
			if (!elementKinds.some((kind) => Object.keys(sdata[kind] || {}).length))
				throw new Error(
					"No spatial elements were found. Choose the SpatialData .zarr folder itself.",
				);
			if (!sdata.coordinateSystems.length)
				throw new Error("This dataset has no supported coordinate system.");
			setDataset(sdata);
			setSource(name);
			setSourceInfo(info);
			setTab("Layers");
			setReadNotices(skipped);
			resetView(
				sdata,
				sdata.coordinateSystems.includes("global")
					? "global"
					: sdata.coordinateSystems[0],
			);
		} catch (failure) {
			if (generation === request.current)
				setError(failure.message || String(failure));
		} finally {
			if (generation === request.current) setStatus("");
		}
	}
	function openFolder(event) {
		const files = Array.from(event.target.files || []);
		event.target.value = "";
		if (!files.length) return;
		const name = files[0].webkitRelativePath.split("/")[0] || "Local folder";
		openStore(
			() => FolderStore.fromFiles(files),
			name,
			`${files.length.toLocaleString()} local files · read on demand`,
		);
	}
	function openZip(event) {
		const file = event.target.files?.[0];
		event.target.value = "";
		if (file)
			openStore(
				() => ZipStore.open(file),
				file.name,
				`${(file.size / 1024 ** 2).toFixed(1)} MiB archive · read on demand`,
			);
	}
	function close() {
		request.current++;
		setDataset(null);
		setSource("");
		setSourceInfo("");
		setStatus("");
		setError("");
		setSelection(null);
		setLayerStatuses({});
		setReadNotices([]);
		setGpuError("");
		setReady(false);
		getChunkCache()?.clear();
	}
	async function importView(event) {
		const file = event.target.files?.[0];
		event.target.value = "";
		if (!file || !dataset) return;
		try {
			if (file.size > 2 * 1024 * 1024)
				throw new Error("View files must be smaller than 2 MiB.");
			const value = validateWorkspace(JSON.parse(await file.text()), dataset);
			const parsedStack = renderStackSchema.parse(value.renderStack);
			setCoordinateSystem(value.coordinateSystem);
			setStack(parsedStack);
			setViewState(value.viewState);
			setSelectedId(parsedStack.entries[0]?.id || "");
			setSelection(null);
			setError("");
		} catch (failure) {
			setError(failure.message || String(failure));
		}
	}
	function exportView() {
		download(
			"spatialdata-view.json",
			JSON.stringify(
				createWorkspace(dataset, source, coordinateSystem, stack, viewState),
				null,
				2,
			),
		);
	}
	function zoom(delta) {
		if (viewState)
			setViewState({
				...viewState,
				zoom: Math.max(-20, Math.min(20, viewState.zoom + delta)),
			});
	}
	function exportImage() {
		pendingExport.current = true;
		if (viewState)
			setViewState({ ...viewState, target: [...viewState.target] });
	}
	const failures = [
		...readNotices,
		...Object.entries(layerStatuses).flatMap(([id, value]) => [
			...value.errors.map((item) => `${id}: ${item.error.message}`),
			...value.notices.map(
				(item) => `${id}: ${item.message || item.code || pretty(item)}`,
			),
		]),
	];
	return (
		<main className="app-shell">
			<aside className="sidebar">
				<header className="brand-area">
					<img src="./logo.svg" alt="SpatialBiologyToolkit" />
					<div>
						<div className="brand-title">SpatialData Explorer</div>
						<div className="brand-subtitle">
							SpatialBiologyToolkit <span>v{__VIEWER_VERSION__}</span>
						</div>
					</div>
				</header>
				<div className="local-badge">
					<span className="privacy-dot" />
					Local files · browser processing
				</div>
				<Section number="01" title="Open your dataset">
					<p className="muted">
						Choose a SpatialData store from your computer. Your data stays here.
					</p>
					<button
						className="file-button primary"
						onClick={() => folderInput.current.click()}
					>
						Open SpatialData folder <span>.zarr</span>
					</button>
					<button
						className="file-button"
						onClick={() => zipInput.current.click()}
					>
						Open ZIP archive <span>.zip</span>
					</button>
					<input
						ref={(element) => {
							folderInput.current = element;
							element?.setAttribute("webkitdirectory", "");
						}}
						type="file"
						multiple
						hidden
						aria-label="SpatialData folder"
						onChange={openFolder}
					/>
					<input
						ref={zipInput}
						type="file"
						accept=".zip"
						hidden
						aria-label="SpatialData ZIP"
						onChange={openZip}
					/>
					{status && (
						<p className="status" role="status">
							{status}
						</p>
					)}
					{source && dataset && (
						<div className="source-card">
							<strong>{source}</strong>
							<span>{sourceInfo}</span>
							<button className="text-button" onClick={close}>
								Close dataset
							</button>
						</div>
					)}
					{(status || error) && (
						<button className="text-button" onClick={close}>
							Cancel / clear
						</button>
					)}
					{error && (
						<p className="error" role="alert">
							{error}
						</p>
					)}
				</Section>
				<nav className="sidebar-tabs" aria-label="Viewer panels">
					{["Layers", "Inspect", "Views"].map((name) => (
						<button
							key={name}
							aria-pressed={tab === name}
							onClick={() => setTab(name)}
						>
							{name}
						</button>
					))}
				</nav>
				{tab === "Layers" && (
					<>
						<Section number="02" title="Coordinate system">
							<label>
								Show elements in
								<select
									aria-label="Coordinate system"
									value={coordinateSystem}
									disabled={!dataset}
									onChange={(event) => resetView(dataset, event.target.value)}
								>
									{!dataset && <option>Open a dataset first</option>}
									{dataset?.coordinateSystems.map((name) => (
										<option key={name}>{name}</option>
									))}
								</select>
							</label>
							<p className="muted">
								Images and annotations use the transforms stored in your
								dataset.
							</p>
						</Section>
						<Section number="03" title="Layers">
							<input
								type="search"
								aria-label="Filter layers"
								placeholder="Find an image, mask or geometry…"
								value={filter}
								onChange={(event) => setFilter(event.target.value)}
								disabled={!dataset}
							/>
							<div className="layer-list">
								{elements
									.filter((element) =>
										element.key.toLowerCase().includes(filter.toLowerCase()),
									)
									.map(({ id, type, key }) => {
										const entry = stack.entries.find((item) => item.id === id);
										return (
											<div
												className={`layer-row ${selectedId === id ? "selected" : ""}`}
												key={id}
											>
												<input
													type="checkbox"
													aria-label={`Show ${key}`}
													checked={entry?.visible || false}
													onChange={(event) =>
														updateEntry(id, { visible: event.target.checked })
													}
												/>
												<button
													onClick={() => setSelectedId(id)}
													aria-pressed={selectedId === id}
												>
													<span>{key}</span>
													<small>{type}</small>
												</button>
											</div>
										);
									})}
							</div>
							{!dataset && (
								<p className="muted">
									Your images, labels, points and shapes will appear here.
								</p>
							)}
						</Section>
						{selected && (
							<Section number="04" title="Display settings">
								<div className="selected-title">
									{selected.source.elementKey}
								</div>
								<label>
									Opacity{" "}
									<span className="value">
										{Math.round(selected.props.opacity * 100)}%
									</span>
									<input
										type="range"
										min="0"
										max="1"
										step="0.05"
										aria-label="Layer opacity"
										value={selected.props.opacity}
										onChange={(event) =>
											updateEntry(selectedId, {
												props: { opacity: Number(event.target.value) },
											})
										}
									/>
								</label>
								{selected.source.elementType === "image" && (
									<div ref={setChannelTarget} />
								)}
								{["labels", "shapes"].includes(selected.source.elementType) && (
									<>
										<label>
											Colour cells by
											<select
												aria-label="Colour cells by"
												value={
													selected.props.fillColorByColumn?.columnName || ""
												}
												onChange={(event) => {
													const column = event.target.value;
													updateEntry(selectedId, {
														props: {
															fillColorByColumn: column
																? { columnName: column, mode: "auto" }
																: undefined,
															tooltipFields: column ? [column] : [],
														},
													});
												}}
											>
												<option value="">Single colour</option>
												{annotationColumns.map((column) => (
													<option key={column}>{column}</option>
												))}
											</select>
										</label>
										{selected.source.elementType === "labels" && (
											<label>
												Mask style
												<select
													aria-label="Mask style"
													value={
														selected.props.channels?.channelsFilled?.[0]
															? "filled"
															: "outline"
													}
													onChange={(event) =>
														updateEntry(selectedId, {
															props: {
																channels: {
																	...selected.props.channels,
																	channelsFilled: [
																		event.target.value === "filled",
																	],
																},
															},
														})
													}
												>
													<option value="outline">Outlines</option>
													<option value="filled">Filled cells</option>
												</select>
											</label>
										)}
										<label>
											Layer colour
											<input
												type="color"
												aria-label="Layer colour"
												value={hex(
													selected.source.elementType === "labels"
														? selected.props.channels?.colors?.[0] || [
																129, 216, 193,
															]
														: selected.props.fillColor || [129, 216, 193],
												)}
												onChange={(event) =>
													updateEntry(selectedId, {
														props:
															selected.source.elementType === "labels"
																? {
																		channels: {
																			...selected.props.channels,
																			colors: [rgb(event.target.value)],
																		},
																	}
																: {
																		fillColor: [
																			...rgb(event.target.value),
																			180,
																		],
																	},
													})
												}
											/>
										</label>
										<p className="muted">
											{associations.length
												? `Linked table: ${associations[0][0]}`
												: "No associated annotation table."}
										</p>
									</>
								)}
								{selected.source.elementType === "points" && (
									<>
										<label>
											Point radius
											<input
												type="range"
												min="0.25"
												max="8"
												step="0.25"
												value={selected.props.pointSize || 1}
												onChange={(event) =>
													updateEntry(selectedId, {
														props: { pointSize: Number(event.target.value) },
													})
												}
											/>
										</label>
										<label className="checkbox-label">
											<input
												type="checkbox"
												checked={selected.props.colorByFeature || false}
												onChange={(event) =>
													updateEntry(selectedId, {
														props: { colorByFeature: event.target.checked },
													})
												}
											/>
											Colour by feature
										</label>
										<p className="muted">
											Large point clouds may show a bounded subset. Check layer
											notices below.
										</p>
									</>
								)}
								<div className="two-columns">
									<button onClick={() => setViewState(null)}>
										Fit visible layers
									</button>
									<button
										onClick={() => {
											setStack((previous) => {
												const entries = [...previous.entries];
												const index = entries.findIndex(
													(entry) => entry.id === selectedId,
												);
												if (index < entries.length - 1)
													[entries[index], entries[index + 1]] = [
														entries[index + 1],
														entries[index],
													];
												return { ...previous, entries };
											});
										}}
									>
										Bring forward
									</button>
								</div>
							</Section>
						)}
					</>
				)}
				{tab === "Inspect" && (
					<>
						<Section number="02" title="Selected cell / feature">
							{selection ? (
								<>
									<dl>
										<dt>Element</dt>
										<dd>{selection.element}</dd>
										<dt>Instance ID</dt>
										<dd>{selection.id}</dd>
										{selection.rowIndex !== undefined && (
											<>
												<dt>Table row</dt>
												<dd>{selection.rowIndex}</dd>
											</>
										)}
									</dl>
									<TooltipContent tooltip={selection.tooltip} />
									<button onClick={() => setSelection(null)}>
										Clear selection
									</button>
								</>
							) : (
								<p className="muted">
									Click a cell, shape or point to inspect its identity and
									annotations.
								</p>
							)}
						</Section>
						<Section number="03" title="Dataset contents">
							{dataset ? (
								<>
									<div className="inventory">
										{[...elementKinds, "tables"].map((kind) => (
											<div key={kind}>
												<strong>
													{Object.keys(dataset[kind] || {}).length}
												</strong>
												<span>{kind}</span>
											</div>
										))}
									</div>
									{Object.entries(dataset.tables || {}).map(([name, table]) => (
										<details className="table-details" key={name}>
											<summary>{name}</summary>
											<p className="muted">
												{table.getObsColumnNames().join(" · ") ||
													"No observation columns"}
											</p>
										</details>
									))}
								</>
							) : (
								<p className="muted">
									Open a SpatialData store to inspect its contents.
								</p>
							)}
						</Section>
					</>
				)}
				{tab === "Views" && (
					<Section number="02" title="Save a view">
						<p className="muted">
							Save channels, layers and framing as a small JSON file. Reopen the
							dataset before restoring a view.
						</p>
						<button
							className="file-button"
							disabled={!dataset}
							onClick={exportView}
						>
							Export view settings <span>.json</span>
						</button>
						<button
							className="file-button"
							disabled={!dataset}
							onClick={() => workspaceInput.current.click()}
						>
							Import view settings <span>.json</span>
						</button>
						<input
							ref={workspaceInput}
							type="file"
							accept=".json"
							hidden
							aria-label="View settings"
							onChange={importView}
						/>
						<button
							className="file-button"
							disabled={!ready || !viewState}
							onClick={exportImage}
						>
							Export current image <span>.png</span>
						</button>
						<p className="muted">
							PNG exports the visible canvas at screen resolution.
						</p>
					</Section>
				)}
				{failures.length > 0 && (
					<div className="notices" role="alert">
						{failures.map((message, i) => (
							<p key={i}>{message}</p>
						))}
					</div>
				)}
				<footer className="sidebar-footer">
					Powered by SpatialData.js<p>Read-only data · no uploads</p>
					<a href="./THIRD_PARTY_LICENSES.txt" target="_blank" rel="noreferrer">
						Open-source licences
					</a>
				</footer>
			</aside>
			<section className="workspace">
				<header className="workspace-header">
					<div>
						<div className="eyebrow">SPATIALBIOLOGYTOOLKIT</div>
						<h1>{dataset ? source : "SpatialData Explorer"}</h1>
						<p>
							{dataset
								? `${coordinateSystem} · ${enabled} visible ${enabled === 1 ? "layer" : "layers"}`
								: "Images, cells and annotations. Together, on your computer."}
						</p>
					</div>
					<div className="header-badge">
						<span className="privacy-dot" />
						Browser local
					</div>
				</header>
				<div className="canvas-toolbar">
					<span>{dataset ? "Spatial view" : "Your workspace"}</span>
					<div>
						<button
							disabled={!viewState}
							onClick={() => zoom(-0.5)}
							aria-label="Zoom out"
						>
							−
						</button>
						<button
							disabled={!viewState}
							onClick={() => zoom(0.5)}
							aria-label="Zoom in"
						>
							+
						</button>
						<button disabled={!dataset} onClick={() => setViewState(null)}>
							Fit view
						</button>
						<button
							disabled={!dataset}
							onClick={() => viewport.current?.requestFullscreen?.()}
						>
							Fullscreen
						</button>
						<button disabled={!ready || !viewState} onClick={exportImage}>
							Export PNG
						</button>
					</div>
				</div>
				<div className="viewport" ref={viewport} data-testid="viewport">
					{dataset ? (
						<ViewerBoundary key={source + ":" + request.current}>
							<SpatialCanvasViewer
								spatialData={dataset}
								coordinateSystem={coordinateSystem}
								renderStack={stack}
								viewState={viewState}
								onViewStateChange={setViewState}
								onFeatureClick={onFeatureClick}
								onEntryStatusChange={entryStatus}
								hoverTooltipMode="simple"
								deckProps={deckProps}
							>
								{selected?.source.elementType === "image" && (
									<ChannelControls
										entry={selected}
										target={channelTarget}
										onChange={changeChannels}
									/>
								)}
							</SpatialCanvasViewer>
						</ViewerBoundary>
					) : (
						<div className="empty-state">
							<div className="empty-mark">
								<span />
								<span />
								<span />
							</div>
							<div className="eyebrow">A CLOSER LOOK AT YOUR TISSUE</div>
							<h2>Explore the whole picture.</h2>
							<p>
								Bring your images, segmentation and cell annotations
								<br className="wide-only" /> into one spatial view. Everything
								stays on your computer.
							</p>
							<button
								className="primary"
								onClick={() => folderInput.current.click()}
							>
								Open SpatialData folder <span>↗</span>
							</button>
							<div className="empty-formats">
								<span>Multichannel images</span>
								<span>Cell masks</span>
								<span>Points & shapes</span>
							</div>
						</div>
					)}
					{gpuError && (
						<div className="canvas-error" role="alert">
							Rendering error: {gpuError}
						</div>
					)}
				</div>
				<footer className="statusbar">
					<span>
						<span className="privacy-dot" />
						{status ||
							(dataset
								? "Local store connected"
								: "Ready for a local SpatialData store")}
					</span>
					<span>Scroll to zoom · drag to pan · click to inspect</span>
				</footer>
			</section>
		</main>
	);
}
function TooltipContent({ tooltip }) {
	if (!tooltip)
		return (
			<p className="muted">
				Choose a table column under Display settings to include its annotation.
			</p>
		);
	const sections = tooltip.sections || [tooltip];
	return (
		<div className="feature-data">
			{sections.map((section, index) => (
				<div key={index}>
					{section.title && <h3>{section.title}</h3>}
					<dl>
						{(section.items || []).map((item, i) => (
							<React.Fragment key={i}>
								<dt>{item.label || item.key}</dt>
								<dd>{pretty(item.value)}</dd>
							</React.Fragment>
						))}
					</dl>
				</div>
			))}
		</div>
	);
}
createRoot(document.getElementById("root")).render(<App />);
