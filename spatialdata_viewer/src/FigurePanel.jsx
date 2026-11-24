import React, { useEffect, useRef, useState } from "react";
import { Check, NumberField } from "./ui-controls.jsx";
import { CalibrationControls } from "./ViewOverlayControls.jsx";
import { viewTools, exportDimensions, figureFilename } from "./view-tools.mjs";
import { readFigureMetadata } from "./figure-metadata.mjs";
export default function FigurePanel({
	session,
	dataset,
	onChange,
	onPane,
	onMode,
	onExport,
	onCancel,
	busy,
	progress,
	rect,
}) {
	const settings = session.figure,
		active = settings.mode === "figure";
	const [tableKey, setTableKey] = useState(
			Object.keys(dataset?.tables || {})[0] || "",
		),
		[metadata, setMetadata] = useState([]),
		[metadataLoaded, setMetadataLoaded] = useState(false),
		[reading, setReading] = useState(false),
		[message, setMessage] = useState("");
	const job = useRef(null);
	useEffect(() => {
		job.current?.abort();
		job.current = null;
		setMetadata([]);
		setMetadataLoaded(false);
		setReading(false);
		setMessage("");
		return () => job.current?.abort();
	}, [dataset, session.coordinateSystem, tableKey]);
	const change = (patch) => onChange({ ...settings, ...patch });
	const filename = (patch) =>
		change({ filename: { ...settings.filename, ...patch } });
	let dimensions,
		filenamePreview,
		error = "";
	const context = {
		roi: session.coordinateSystem,
		viewNames: session.panes.map((p) => p.name),
		metadata,
		metadataLoaded,
	};
	try {
		dimensions = exportDimensions(session.panes.length, rect, settings);
		filenamePreview = figureFilename(settings, context);
	} catch (e) {
		error = e.message;
	}
	async function readMetadata() {
		const controller = new AbortController();
		job.current = controller;
		setReading(true);
		setMessage("Reading table metadata…");
		try {
			const fields = await readFigureMetadata(
				dataset,
				session.coordinateSystem,
				tableKey,
				controller.signal,
				setMessage,
			);
			if (job.current !== controller || controller.signal.aborted) return;
			setMetadata(fields);
			setMetadataLoaded(true);
			setMessage(
				`${fields.length} fields have one value across the plotted regions.`,
			);
		} catch (e) {
			if (job.current === controller)
				setMessage(
					controller.signal.aborted ? "Metadata read cancelled." : e.message,
				);
		} finally {
			if (job.current === controller) setReading(false);
		}
	}
	return (
		<div className="figure-panel">
			<div className="figure-mode-buttons">
				<button
					aria-pressed={!active}
					disabled={busy}
					onClick={() => onMode("viewer")}
				>
					Viewer mode
				</button>
				<button
					aria-pressed={active}
					disabled={!dataset || busy}
					onClick={() => onMode("figure")}
				>
					Figure mode
				</button>
				<span className="muted">
					{active
						? "Compose panes and overlays. Returning to Viewer restores your camera positions."
						: "Switch to Figure mode to compose and export."}
				</span>
			</div>
			<div className="figure-settings">
				<fieldset disabled={!active || busy}>
					<legend>Frame &amp; physical size</legend>
					<label>
						Aspect per pane
						<select
							aria-label="Aspect per pane"
							value={settings.aspect}
							disabled={settings.physicalLock}
							onChange={(e) => change({ aspect: e.target.value })}
						>
							{["free", "1:1", "4:3", "3:2", "16:9"].map((v) => (
								<option key={v} value={v}>
									{v === "free" ? "Available space" : v}
								</option>
							))}
						</select>
					</label>
					<CalibrationControls figure={settings} onChange={onChange} />
					<Check
						label="Lock field of view in µm"
						disabled={!settings.calibrated}
						value={settings.physicalLock}
						onChange={(v) => change({ physicalLock: v })}
					/>
					<div className="two-columns">
						<NumberField
							label="Width (µm)"
							value={settings.physicalWidth}
							min={0.001}
							max={1e7}
							disabled={!settings.physicalLock}
							onChange={(v) => change({ physicalWidth: v })}
						/>
						<NumberField
							label="Height (µm)"
							value={settings.physicalHeight}
							min={0.001}
							max={1e7}
							disabled={!settings.physicalLock}
							onChange={(v) => change({ physicalHeight: v })}
						/>
					</div>
					<p className="muted">
						Pan remains available. A physical lock fixes the field of view and
						disables zoom.
					</p>
				</fieldset>
				<fieldset disabled={!active || busy}>
					<legend>Figure overlays</legend>
					<label>
						Colour scheme
						<select
							aria-label="Figure colour scheme"
							value={
								settings.background === "#000000" &&
								settings.foreground === "#ffffff"
									? "dark"
									: settings.background === "#ffffff" &&
											settings.foreground === "#000000"
										? "light"
										: "custom"
							}
							onChange={(e) => {
								if (e.target.value !== "custom")
									change({
										background:
											e.target.value === "light" ? "#ffffff" : "#000000",
										foreground:
											e.target.value === "light" ? "#000000" : "#ffffff",
									});
							}}
						>
							<option value="dark">White on black</option>
							<option value="light">Black on white</option>
							<option value="custom">Custom</option>
						</select>
					</label>
					<div className="two-columns">
						<label>
							Background
							<input
								aria-label="Figure background"
								type="color"
								value={settings.background}
								onChange={(e) => change({ background: e.target.value })}
							/>
						</label>
						<label>
							Text and scale bars
							<input
								aria-label="Figure foreground"
								type="color"
								value={settings.foreground}
								onChange={(e) => change({ foreground: e.target.value })}
							/>
						</label>
					</div>
					{[
						["showNames", "View names"],
						["showLegends", "Legends for enabled views"],
						["showScaleBars", "Scale bars for enabled views"],
						["showLayerNames", "SpatialData layer names"],
						["overlayBackground", "Background behind labels"],
					].map(([key, label]) => (
						<Check
							key={key}
							label={label}
							value={settings[key]}
							onChange={(v) => change({ [key]: v })}
						/>
					))}
					<div className="two-columns">
						{[
							["fontSize", "View name text (px)", 96],
							["scaleFontSize", "Scale-bar text (px)", 96],
							["legendSize", "Figure legend text (px)", 72],
							["legendRows", "Figure legend rows", 100],
						].map(([key, label, max]) => (
							<NumberField
								key={key}
								label={label}
								value={settings[key]}
								min={key === "legendRows" ? 1 : 8}
								max={max}
								step={1}
								onChange={(v) => change({ [key]: v })}
							/>
						))}
					</div>
					<details>
						<summary>Scale bars · choose panes</summary>
						<div className="drawer-actions">
							<button
								onClick={() =>
									session.panes.forEach((p) =>
										onPane(p.id, {
											scaleBar: { ...viewTools(p).scaleBar, visible: true },
										}),
									)
								}
							>
								All panes
							</button>
							<button
								onClick={() =>
									session.panes.forEach((p) =>
										onPane(p.id, {
											scaleBar: { ...viewTools(p).scaleBar, visible: false },
										}),
									)
								}
							>
								No panes
							</button>
						</div>
						{session.panes.map((p) => {
							const bar = viewTools(p).scaleBar;
							const edit = (v) => onPane(p.id, { scaleBar: { ...bar, ...v } });
							return (
								<div className="figure-scale-row" key={p.id}>
									<Check
										label={`${p.name} scale bar`}
										value={bar.visible}
										onChange={(v) => edit({ visible: v })}
									/>
									<Check
										label={`${p.name} automatic length`}
										value={bar.automatic}
										onChange={(v) => edit({ automatic: v })}
									/>
									<NumberField
										label={`${p.name} scale length`}
										value={bar.length}
										min={0.001}
										max={1e7}
										disabled={bar.automatic}
										onChange={(v) => edit({ length: v })}
									/>
									<NumberField
										label={`${p.name} line width`}
										value={bar.thickness}
										min={1}
										max={20}
										onChange={(v) => edit({ thickness: v })}
									/>
								</div>
							);
						})}
					</details>
					<p className="muted">
						Custom legend labels and overlay positions are in Views.
					</p>
				</fieldset>
				<fieldset disabled={!active || busy}>
					<legend>Export</legend>
					<div className="two-columns">
						<NumberField
							label="Pane width (px)"
							value={settings.paneWidth}
							min={128}
							max={8192}
							step={1}
							onChange={(v) => change({ paneWidth: v })}
						/>
						<NumberField
							label="Pane spacing (px)"
							value={settings.gap}
							min={0}
							max={500}
							step={1}
							onChange={(v) => change({ gap: v })}
						/>
					</div>
					<label>
						Format
						<select
							aria-label="Figure format"
							value={settings.format}
							onChange={(e) => change({ format: e.target.value })}
						>
							<option value="png">PNG</option>
							<option value="jpeg">JPEG</option>
							<option value="svg">SVG · separate SpatialData layers</option>
						</select>
					</label>
					<Check
						label="Automatically bookmark each export"
						value={settings.autoBookmark}
						onChange={(v) => change({ autoBookmark: v })}
					/>
					<details open>
						<summary>File name</summary>
						<label>
							Label / prefix
							<input
								aria-label="Figure filename prefix"
								value={settings.filename.label}
								maxLength={100}
								onChange={(e) => filename({ label: e.target.value })}
							/>
						</label>
						{[
							["roi", "Coordinate-system name"],
							["views", "Include view names"],
							["timestamp", "Timestamp (UTC)"],
						].map(([key, label]) => (
							<Check
								key={key}
								label={label}
								value={settings.filename[key]}
								onChange={(v) => filename({ [key]: v })}
							/>
						))}
						<label>
							Metadata table
							<select
								value={tableKey}
								onChange={(e) => setTableKey(e.target.value)}
							>
								{Object.keys(dataset?.tables || {}).map((key) => (
									<option key={key}>{key}</option>
								))}
							</select>
						</label>
						<button disabled={!tableKey || reading} onClick={readMetadata}>
							{reading
								? "Reading metadata…"
								: "Read coordinate-system metadata"}
						</button>
						{reading && (
							<button onClick={() => job.current?.abort()}>
								Cancel metadata read
							</button>
						)}
						<p className="muted" role="status">
							{message ||
								"Optional: use obs fields with one value across the regions in this coordinate system."}
						</p>
						<label>
							Filename metadata
							<select
								aria-label="Filename metadata"
								multiple
								size={3}
								value={settings.filename.metadata}
								onChange={(e) =>
									filename({
										metadata: [...e.target.selectedOptions].map((o) => o.value),
									})
								}
							>
								{[
									...new Set([
										...metadata.map((m) => m.name),
										...settings.filename.metadata,
									]),
								].map((name) => (
									<option key={name}>{name}</option>
								))}
							</select>
						</label>
						<button onClick={() => filename({ metadata: [] })}>
							Clear metadata fields
						</button>
					</details>
					<output className="filename-preview">{filenamePreview}</output>
					{dimensions && (
						<p className="muted">
							{dimensions.outputWidth} × {dimensions.outputHeight} pixels ·{" "}
							{session.panes.length} panes
						</p>
					)}
					<button
						className="primary"
						disabled={!dataset || !!error || reading}
						onClick={() => onExport(context)}
					>
						Export figure
					</button>
					<p className="muted">
						SVG keeps each SpatialData layer as an embedded raster, with
						editable vector legends and scale bars. PNG/JPEG preserve the
						composed appearance.
					</p>
				</fieldset>
			</div>
			{error && (
				<p className="error" role="alert">
					{error}
				</p>
			)}
			<p role="status">{progress}</p>
			{busy && <button onClick={onCancel}>Cancel figure export</button>}
		</div>
	);
}
