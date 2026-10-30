import React from "react";
import { createPortal } from "react-dom";
import { useImageLayerContext, mergeLayerChannelState } from "@spatialdata/vis";
import { palette, rgb, hex } from "./model.mjs";

export default function ChannelControls({ entry, target, onChange }) {
	const context = useImageLayerContext(entry.source.elementKey);
	if (!target) return null;
	if (!context)
		return createPortal(
			<p className="muted">
				{entry.visible
					? "Loading image channels…"
					: "Show this image to load its channels."}
			</p>,
			target,
		);
	const merged = mergeLayerChannelState(
		entry.props.channels || {},
		context.defaults,
		entry.id,
	);
	const count = merged.channelCount;
	const names = context.channelNames;
	const axes = context.selectionAxisSizes || {};
	const current = {
		colors: merged.colors.slice(0, count),
		contrastLimits: merged.contrastLimits.slice(0, count),
		channelsVisible: merged.channelsVisible.slice(0, count),
		selections: merged.selections.slice(0, count),
	};
	function edit(key, index, value) {
		const list = [...current[key]];
		list[index] = value;
		onChange({ ...current, [key]: list });
	}
	function remove(index) {
		onChange(
			Object.fromEntries(
				Object.entries(current).map(([key, values]) => [
					key,
					values.filter((_, i) => i !== index),
				]),
			),
		);
	}
	function add() {
		const selection = Object.fromEntries(
			Object.keys(axes).map((axis) => [axis, 0]),
		);
		if (axes.c) selection.c = Math.min(count, axes.c - 1);
		onChange({
			colors: [...current.colors, rgb(palette[count % palette.length])],
			contrastLimits: [
				...current.contrastLimits,
				context.defaults.contrastLimits?.[0] || [0, 1],
			],
			channelsVisible: [...current.channelsVisible, true],
			selections: [...current.selections, selection],
		});
	}
	return createPortal(
		<div className="channel-controls">
			<p className="muted">
				Mix up to six channels. Contrast changes affect the display only.
			</p>
			{Array.from({ length: count }, (_, index) => {
				const selection = current.selections[index] || {};
				const limits = current.contrastLimits[index] || [0, 1];
				return (
					<div className="channel-card" key={index}>
						<div className="channel-heading">
							<input
								type="checkbox"
								aria-label={`Show channel ${index + 1}`}
								checked={current.channelsVisible[index] ?? true}
								onChange={(event) =>
									edit("channelsVisible", index, event.target.checked)
								}
							/>
							<select
								aria-label={`Marker for channel ${index + 1}`}
								value={selection.c ?? 0}
								onChange={(event) =>
									edit("selections", index, {
										...selection,
										c: Number(event.target.value),
									})
								}
							>
								{Array.from({ length: axes.c || names.length || 1 }, (_, c) => (
									<option value={c} key={c}>
										{names[c] || `Channel ${c + 1}`}
									</option>
								))}
							</select>
							<button
								className="icon-button"
								aria-label={`Remove channel ${index + 1}`}
								disabled={count <= 1}
								onClick={() => remove(index)}
							>
								×
							</button>
						</div>
						<div className="swatches">
							{palette.map((color) => (
								<button
									key={color}
									style={{ backgroundColor: color }}
									aria-label={`Channel ${index + 1} colour ${color}`}
									aria-pressed={
										hex(current.colors[index] || [255, 255, 255]) === color
									}
									onClick={() => edit("colors", index, rgb(color))}
								/>
							))}
							<input
								type="color"
								aria-label={`Custom colour for channel ${index + 1}`}
								value={hex(current.colors[index] || [255, 255, 255])}
								onChange={(event) =>
									edit("colors", index, rgb(event.target.value))
								}
							/>
						</div>
						<div className="two-columns">
							{["Min", "Max"].map((label, side) => (
								<label key={label}>
									{label}
									<input
										type="number"
										step="any"
										aria-label={`${label} for channel ${index + 1}`}
										value={limits[side]}
										onChange={(event) => {
											if (event.target.value === "") return;
											const next = [...limits];
											next[side] = Number(event.target.value);
											if (next.every(Number.isFinite) && next[0] < next[1])
												edit("contrastLimits", index, next);
										}}
									/>
								</label>
							))}
						</div>
						{["z", "t"]
							.filter((axis) => axes[axis] > 1)
							.map((axis) => (
								<label key={axis}>
									{axis.toUpperCase()} slice
									<input
										type="number"
										min="0"
										max={axes[axis] - 1}
										value={selection[axis] || 0}
										onChange={(event) =>
											edit("selections", index, {
												...selection,
												[axis]: Math.max(
													0,
													Math.min(
														axes[axis] - 1,
														Math.trunc(Number(event.target.value)),
													),
												),
											})
										}
									/>
								</label>
							))}
					</div>
				);
			})}
			<div className="two-columns">
				<button onClick={add} disabled={count >= 6}>
					+ Add channel
				</button>
				<button onClick={() => onChange({})}>Reset channels</button>
			</div>
		</div>,
		target,
	);
}
