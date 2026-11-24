import React, { useState } from "react";
import { viewTools } from "./view-tools.mjs";
import { Check, NumberField, PositionField } from "./ui-controls.jsx";
export function CalibrationControls({ figure, onChange }) {
	return (
		<div className="calibration-controls">
			<NumberField
				label="µm per coordinate unit"
				value={figure.umPerPixel}
				min={0.000001}
				max={1000000}
				onChange={(umPerPixel) =>
					onChange({ ...figure, umPerPixel, calibrated: true })
				}
			/>
			<p className="muted">
				Uses the current coordinate system after SpatialData transforms, not the
				image pixel size.{" "}
				{figure.calibrated
					? "Calibration set for this workspace."
					: "Units are unconfirmed; scale bars show coordinate units."}
			</p>
			{!figure.calibrated && (
				<button onClick={() => onChange({ ...figure, calibrated: true })}>
					Use 1 coordinate unit = 1 µm
				</button>
			)}
		</div>
	);
}
export default function ViewOverlayControls({
	pane,
	groups = [],
	onChange,
	figure,
	onFigure,
	compact = false,
}) {
	const tools = viewTools(pane),
		[query, setQuery] = useState("");
	const patch = (key, value) =>
		onChange({ [key]: { ...tools[key], ...value } });
	const layerPatch = (id, value) =>
		patch("legend", {
			layers: {
				...tools.legend.layers,
				[id]: { ...tools.legend.layers[id], ...value },
			},
		});
	return (
		<div className={`view-overlay-controls ${compact ? "compact" : ""}`}>
			{compact && (
				<div className="display-toggles">
					{[
						["images", "Image channels"],
						["cells", "Cell layers"],
						["points", "Transcript points"],
					].map(([key, label]) => (
						<Check
							key={key}
							label={label}
							value={tools.display[key]}
							onChange={(v) => patch("display", { [key]: v })}
						/>
					))}
				</div>
			)}
			<div className="overlay-switches">
				<Check
					label="Show legend"
					value={tools.legend.visible}
					onChange={(v) => patch("legend", { visible: v })}
				/>
				<Check
					label="Show scale bar"
					value={tools.scaleBar.visible}
					onChange={(v) => patch("scaleBar", { visible: v })}
				/>
				<Check
					label="Show layer names"
					value={tools.layerNames.visible}
					onChange={(v) => patch("layerNames", { visible: v })}
				/>
			</div>
			<details open={!compact}>
				<summary>Legend · {pane.name}</summary>
				<label>
					Legend title
					<input
						aria-label="Legend title"
						maxLength={120}
						value={tools.legend.title}
						placeholder="Optional title"
						onChange={(e) => patch("legend", { title: e.target.value })}
					/>
				</label>
				<div className="two-columns">
					<PositionField
						label="Legend position"
						value={tools.legend.position}
						onChange={(v) => patch("legend", { position: v })}
					/>
					<NumberField
						label="Legend text size"
						value={tools.legend.fontSize}
						min={8}
						max={36}
						onChange={(v) => patch("legend", { fontSize: v })}
					/>
				</div>
				<NumberField
					label="Maximum legend rows"
					value={tools.legend.maxRows}
					min={1}
					max={100}
					step={1}
					onChange={(v) => patch("legend", { maxRows: v })}
				/>
				{!compact && (
					<>
						<p className="muted">
							Edit labels for this view. Colours and ranges follow the displayed
							data.
						</p>
						<input
							type="search"
							aria-label="Find legend entries"
							placeholder="Find a channel, gene or category…"
							value={query}
							onChange={(e) => setQuery(e.target.value)}
						/>
						{groups.map((g) => {
							const custom = tools.legend.layers[g.id] || {};
							return (
								<details key={g.id} className="legend-layer-editor">
									<summary>{g.title}</summary>
									<Check
										label={`Include ${g.id} in legend`}
										value={custom.show !== false}
										onChange={(v) => layerPatch(g.id, { show: v })}
									/>
									<label>
										Layer legend title
										<input
											maxLength={160}
											value={custom.title || ""}
											placeholder={g.title}
											onChange={(e) =>
												layerPatch(g.id, { title: e.target.value })
											}
										/>
									</label>
									{g.rows
										?.filter((r) =>
											`${r.name} ${custom.labels?.[r.key] || ""}`
												.toLowerCase()
												.includes(query.toLowerCase()),
										)
										.slice(0, 60)
										.map((r) => (
											<div className="legend-entry-editor" key={r.key}>
												<Check
													label={r.name}
													value={!custom.hidden?.includes(r.key)}
													onChange={(v) =>
														layerPatch(g.id, {
															hidden: v
																? (custom.hidden || []).filter(
																		(key) => key !== r.key,
																	)
																: [...(custom.hidden || []), r.key],
														})
													}
												/>
												<input
													aria-label={`Legend label for ${r.name}`}
													maxLength={160}
													placeholder={r.name}
													value={custom.labels?.[r.key] || ""}
													onChange={(e) =>
														layerPatch(g.id, {
															labels: {
																...custom.labels,
																[r.key]: e.target.value,
															},
														})
													}
												/>
											</div>
										))}
									{g.rows?.length > 60 && (
										<p className="muted">
											Use search to find entries beyond the first 60.
										</p>
									)}
								</details>
							);
						})}
						<button onClick={() => patch("legend", { title: "", layers: {} })}>
							Reset custom legend labels
						</button>
					</>
				)}
			</details>
			<details>
				<summary>Scale bar settings</summary>
				<CalibrationControls figure={figure} onChange={onFigure} />
				<Check
					label="Automatic scale length"
					value={tools.scaleBar.automatic}
					onChange={(v) => patch("scaleBar", { automatic: v })}
				/>
				<NumberField
					label={`Scale length (${figure.calibrated ? "µm" : "units"})`}
					value={tools.scaleBar.length}
					disabled={tools.scaleBar.automatic}
					min={0.001}
					max={1e7}
					onChange={(v) => patch("scaleBar", { length: v })}
				/>
				<div className="two-columns">
					<NumberField
						label="Scale line width"
						value={tools.scaleBar.thickness}
						min={1}
						max={20}
						onChange={(v) => patch("scaleBar", { thickness: v })}
					/>
					<NumberField
						label="Scale text size"
						value={tools.scaleBar.fontSize}
						min={8}
						max={48}
						onChange={(v) => patch("scaleBar", { fontSize: v })}
					/>
				</div>
				<label>
					Scale colour
					<input
						type="color"
						value={tools.scaleBar.colour}
						onChange={(e) => patch("scaleBar", { colour: e.target.value })}
					/>
				</label>
				<PositionField
					label="Scale position"
					bottom
					value={tools.scaleBar.position}
					onChange={(v) => patch("scaleBar", { position: v })}
				/>
			</details>
			<details>
				<summary>Layer-name settings</summary>
				<PositionField
					label="Layer names position"
					value={tools.layerNames.position}
					onChange={(v) => patch("layerNames", { position: v })}
				/>
				<NumberField
					label="Layer names text size"
					value={tools.layerNames.fontSize}
					min={8}
					max={36}
					onChange={(v) => patch("layerNames", { fontSize: v })}
				/>
				<p className="muted">
					Shows the original SpatialData names of the layers currently
					displayed.
				</p>
			</details>
		</div>
	);
}
