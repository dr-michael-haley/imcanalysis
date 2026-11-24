import React, { useState } from "react";
import {
	addedLayers,
	addLayer,
	removeLayer,
	reorderLayer,
} from "./view-tools.mjs";
import Icon from "./icons.jsx";
export const layerLabel = (type) =>
	({ image: "Image", labels: "Labels", shapes: "Shapes", points: "Points" })[
		type
	] || type;
export default function LayerManager({ pane, onChange, dataset }) {
	const [picker, setPicker] = useState(false),
		[query, setQuery] = useState(""),
		[kind, setKind] = useState("all");
	const active = addedLayers(pane),
		entries = pane.renderStack.entries;
	return (
		<section className="layer-manager" aria-label="Layer manager">
			<div className="section-heading">
				<h2>Active layers</h2>
				<button
					onClick={() => setPicker(!picker)}
					aria-expanded={picker}
					disabled={!dataset}
				>
					{picker ? "Hide layer selector" : "Add layers"}
				</button>
			</div>
			<p className="muted">
				{pane.name} · Select a layer to edit. Layers lower in this list draw on
				top.
			</p>
			<div className="active-layers">
				{active.map((id, i) => {
					const e = entries.find((e) => e.id === id);
					if (!e) return null;
					const type = e.source.elementType,
						key = e.source.elementKey;
					return (
						<div
							className={`active-layer layer-${type} ${pane.selectedId === id ? "selected" : ""}`}
							key={id}
						>
							<button
								className="layer-select"
								aria-pressed={pane.selectedId === id}
								onClick={() => onChange({ ...pane, selectedId: id })}
								aria-label={`${key} ${type}`}
							>
								<span className="layer-type">
									<Icon name={type} />
									{layerLabel(type)}
								</span>
								<span className="layer-name">{key}</span>
							</button>
							<div className="layer-actions">
								<button
									aria-label={`${e.visible ? "Hide" : "Show"} ${key}`}
									title={e.visible ? "Hide layer" : "Show layer"}
									aria-pressed={e.visible}
									onClick={() =>
										onChange({
											...pane,
											renderStack: {
												...pane.renderStack,
												entries: entries.map((x) =>
													x.id === id ? { ...x, visible: !x.visible } : x,
												),
											},
										})
									}
								>
									<Icon name={e.visible ? "eye" : "eyeOff"} />
								</button>
								<button
									aria-label={`Move ${key} up`}
									disabled={!i}
									onClick={() => onChange(reorderLayer(pane, id, -1))}
								>
									<Icon name="up" />
								</button>
								<button
									aria-label={`Move ${key} down`}
									disabled={i === active.length - 1}
									onClick={() => onChange(reorderLayer(pane, id, 1))}
								>
									<Icon name="down" />
								</button>
								<button
									aria-label={`Remove ${key}`}
									onClick={() => onChange(removeLayer(pane, id))}
								>
									<Icon name="close" />
								</button>
							</div>
						</div>
					);
				})}
				{!active.length && (
					<p className="muted">
						Add images, shapes or transcript points to build this view.
					</p>
				)}
			</div>
			{picker && (
				<div className="layer-picker" aria-label="Dataset layer selector">
					<label>
						Search SpatialData layers
						<input
							type="search"
							aria-label="Filter layers"
							placeholder="Search layer names…"
							value={query}
							onChange={(e) => setQuery(e.target.value)}
						/>
					</label>
					<label>
						Layer type
						<select value={kind} onChange={(e) => setKind(e.target.value)}>
							<option value="all">All types</option>
							{["image", "shapes", "points", "labels"].map((type) => (
								<option key={type} value={type}>
									{layerLabel(type)}
								</option>
							))}
						</select>
					</label>
					<div className="available-layers">
						{entries
							.filter(
								(e) =>
									!active.includes(e.id) &&
									(kind === "all" || kind === e.source.elementType) &&
									e.source.elementKey
										.toLowerCase()
										.includes(query.toLowerCase()),
							)
							.map((e) => (
								<button
									className={`layer-option layer-${e.source.elementType}`}
									key={e.id}
									aria-label={`Add ${e.source.elementKey} ${e.source.elementType}`}
									onClick={() => onChange(addLayer(pane, e.id))}
								>
									<span className="layer-type">
										<Icon name={e.source.elementType} />
										{layerLabel(e.source.elementType)}
									</span>
									<span>{e.source.elementKey}</span>
									<Icon name="plus" />
								</button>
							))}
					</div>
					<p className="muted">
						Available in the current coordinate system. Removing a layer keeps
						its settings for when you add it again.
					</p>
				</div>
			)}
		</section>
	);
}
