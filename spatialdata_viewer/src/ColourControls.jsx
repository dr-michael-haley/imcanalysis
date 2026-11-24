import React, { useState, useEffect } from "react";
import {
	COLOURMAPS,
	colourmapGradient,
	continuousScale,
} from "./colour-scale.mjs";
export default function ColourControls({
	value,
	onChange,
	result,
	prefix = "",
}) {
	const [error, setError] = useState("");
	useEffect(() => setError(""), [value]);
	function limit(key, input) {
		const next = { ...value, [key]: input === "" ? null : Number(input) };
		try {
			continuousScale([result.dataLow, result.dataHigh], next);
			setError("");
			onChange(next);
		} catch (e) {
			setError(e.message);
		}
	}
	const lower = value.vmin ?? result.dataLow,
		span = Math.max(
			result.dataHigh - result.dataLow,
			Math.abs(lower) * 0.01,
			0.01,
		);
	const min = lower + span / 1000,
		max = Math.max(result.dataHigh, value.vmax ?? -Infinity, lower + span);
	return (
		<div className="colour-controls">
			<label>
				Colour map
				<select
					aria-label={`${prefix}Colour map`}
					value={value.colourmap || "viridis"}
					onChange={(e) => onChange({ ...value, colourmap: e.target.value })}
				>
					{Object.entries(COLOURMAPS).map(([id, [name]]) => (
						<option key={id} value={id}>
							{name}
						</option>
					))}
				</select>
			</label>
			<label className="checkbox-label">
				<input
					type="checkbox"
					checked={value.reverse || false}
					aria-label={`${prefix}Reverse colour map`}
					onChange={(e) => onChange({ ...value, reverse: e.target.checked })}
				/>
				Reverse colours
			</label>
			<div className="two-columns">
				<label>
					Lower (vmin)
					<input
						type="number"
						step="any"
						aria-label={`${prefix}vmin`}
						placeholder={`Auto (${result.dataLow})`}
						key={`lo-${value.vmin}`}
						defaultValue={value.vmin ?? ""}
						onBlur={(e) => limit("vmin", e.target.value)}
						onKeyDown={(e) => {
							if (e.key === "Enter") e.currentTarget.blur();
						}}
					/>
				</label>
				<label>
					Upper (vmax)
					<input
						type="number"
						step="any"
						aria-label={`${prefix}vmax`}
						placeholder={`Auto (${result.dataHigh})`}
						key={`hi-${value.vmax}`}
						defaultValue={value.vmax ?? ""}
						onBlur={(e) => limit("vmax", e.target.value)}
						onKeyDown={(e) => {
							if (e.key === "Enter") e.currentTarget.blur();
						}}
					/>
				</label>
			</div>
			<label>
				Drag upper limit{" "}
				<span className="value">
					{Number((value.vmax ?? result.dataHigh).toPrecision(5))}
				</span>
				<input
					className="colour-slider"
					type="range"
					aria-label={`${prefix}Adjust vmax`}
					min={min}
					max={max}
					step={span / 1000}
					value={Math.max(min, value.vmax ?? result.dataHigh)}
					style={{
						background: colourmapGradient(value.colourmap, value.reverse),
					}}
					onChange={(e) => onChange({ ...value, vmax: Number(e.target.value) })}
				/>
			</label>
			<button onClick={() => onChange({ ...value, vmin: null, vmax: null })}>
				Reset {prefix ? "embedding " : ""}colour limits
			</button>
			<p className="muted">
				Limits use stored values, before log1p. Values beyond the limits use the
				end colours.
			</p>
			{error && (
				<p className="error" role="alert">
					{error}
				</p>
			)}
		</div>
	);
}
