import React, { useEffect, useState } from "react";
import ColourControls from "./ColourControls.jsx";
import { colourmapGradient } from "./colour-scale.mjs";

export default function CellControls({
	dataset,
	entry,
	binding,
	onChange,
	result,
	geneLists = [],
	propagation,
}) {
	const table = dataset.tables[binding.table];
	const [genes, setGenes] = useState([]),
		[query, setQuery] = useState(""),
		[error, setError] = useState("");
	useEffect(() => {
		let live = true;
		setGenes([]);
		setError("");
		setQuery("");
		if (table)
			table
				.loadVarIndex()
				.then((v) => {
					if (live) setGenes(v);
				})
				.catch((e) => {
					if (live) setError(e.message);
				});
		return () => {
			live = false;
		};
	}, [table]);
	const columns = table?.getObsColumnNames() || [];
	const fieldName = (v) =>
		binding.source === "sum" ? geneLists.find((g) => g.id === v)?.name || v : v;
	const available =
		binding.source === "sum"
			? geneLists.map((g) => g.id)
			: binding.source === "gene"
				? genes
				: columns;
	const filtered = available.filter((v) =>
		fieldName(v).toLowerCase().includes(query.toLowerCase()),
	);
	return (
		<div className="cell-controls">
			<label>
				Linked cell table
				<select
					aria-label="Linked cell table"
					value={binding.table}
					onChange={(e) => {
						const next = dataset.tables[e.target.value];
						onChange({
							table: e.target.value,
							idColumn: next?.getTableKeys().instanceKey || "",
							source: "none",
							field: "",
							scale: "linear",
						});
					}}
				>
					<option value="">Choose a table</option>
					{Object.keys(dataset.tables || {}).map((k) => (
						<option key={k}>{k}</option>
					))}
				</select>
			</label>
			{table && (
				<>
					<details>
						<summary>Cell linking</summary>
						<p className="muted">
							Matches this shapes element using the table’s region and cell IDs.
							Row order is never used.
						</p>
						<label>
							Cell ID column
							<select
								aria-label="Cell ID column"
								value={binding.idColumn}
								onChange={(e) =>
									onChange({ ...binding, idColumn: e.target.value })
								}
							>
								{columns.map((k) => (
									<option key={k}>{k}</option>
								))}
							</select>
						</label>
						<p className="muted">
							Region column: {table.getTableKeys().regionKey || "Missing"}
						</p>
					</details>
					{result?.loading && <p role="status">Reading cell table…</p>}
					{result?.error && (
						<p className="error" role="alert">
							{result.error}
						</p>
					)}
					{result && !result.loading && !result.error && (
						<p
							className={result.matched ? "link-status" : "error"}
							role="status"
						>
							{result.matched.toLocaleString()} /{" "}
							{result.total.toLocaleString()} boundaries linked ·{" "}
							{result.tableRows.toLocaleString()} table rows
							{!result.matched && (
								<>
									<br />
									Unpaired: cell IDs or regions do not match. Boundaries remain
									visible; table colours are unavailable.
								</>
							)}
						</p>
					)}
					<label>
						Overlay data
						<select
							aria-label="Overlay data"
							value={binding.source}
							onChange={(e) => {
								setQuery("");
								onChange({
									...binding,
									source: e.target.value,
									field: "",
									scale: "linear",
									vmin: null,
									vmax: null,
								});
							}}
						>
							<option value="none">Single colour</option>
							<option value="gene">Gene expression · X</option>
							<option value="sum">Gene list · sum of X</option>
							<option value="obs">Cell annotation · obs</option>
						</select>
					</label>
					{binding.source !== "none" && (
						<>
							<input
								aria-label="Search overlay fields"
								type="search"
								placeholder={
									binding.source === "sum"
										? "Search gene lists…"
										: binding.source === "gene"
											? "Search genes…"
											: "Search annotations…"
								}
								value={query}
								onChange={(e) => setQuery(e.target.value)}
							/>
							<label>
								{binding.source === "sum"
									? "Gene list"
									: binding.source === "gene"
										? `Gene (${genes.length})`
										: "Annotation"}
								<select
									aria-label="Overlay field"
									value={binding.field}
									onChange={(e) =>
										onChange({
											...binding,
											field: e.target.value,
											vmin: null,
											vmax: null,
										})
									}
								>
									<option value="">
										Choose{" "}
										{binding.source === "sum"
											? "a gene list"
											: binding.source === "gene"
												? "a gene"
												: "an annotation"}
									</option>
									{binding.field && !filtered.includes(binding.field) && (
										<option value={binding.field}>
											{fieldName(binding.field)}
										</option>
									)}
									{filtered.map((k) => (
										<option key={k} value={k}>
											{fieldName(k)}
										</option>
									))}
								</select>
							</label>
							{(["gene", "sum"].includes(binding.source) ||
								result?.mode === "continuous") && (
								<label>
									Expression scale
									<select
										aria-label="Expression scale"
										value={binding.scale}
										onChange={(e) =>
											onChange({ ...binding, scale: e.target.value })
										}
									>
										<option value="linear">Linear · stored values</option>
										<option value="log1p">Log1p · display only</option>
									</select>
								</label>
							)}
							<p className="muted">
								{binding.source === "sum"
									? "Add stored X values for each gene, then apply the display scale. Create or import lists in the Gene lists drawer."
									: binding.source === "gene"
										? "X values are shown as stored. No normalisation is applied to your dataset."
										: "Categorical annotations use discrete colours; numeric annotations use a continuous scale."}
							</p>
						</>
					)}
					{propagation}
					{error && <p className="error">{error}</p>}
					{binding.field && result?.mode === "continuous" && (
						<ColourControls
							value={binding}
							onChange={onChange}
							result={result}
						/>
					)}
					{result?.error && (binding.vmin != null || binding.vmax != null) && (
						<button
							onClick={() => onChange({ ...binding, vmin: null, vmax: null })}
						>
							Reset colour limits
						</button>
					)}
					{result?.geneSummary && (
						<p className="muted" role="status">
							{result.geneSummary.matched.length} genes summed
							{result.geneSummary.missing.length > 0 && (
								<>
									{" "}
									· Missing from this table:{" "}
									{result.geneSummary.missing.join(", ")}
								</>
							)}
						</p>
					)}
					{binding.field && result?.matched > 0 && (
						<OverlayLegend result={result} />
					)}
				</>
			)}
		</div>
	);
}

export function OverlayLegend({ result }) {
	return (
		<div className="overlay-legend" aria-label="Overlay legend">
			<strong>
				{result.binding.table} · {result.label || result.binding.field}
			</strong>
			{result.mode === "categorical" ? (
				<div className="legend-categories">
					{result.categories.map(({ label, colour }) => (
						<div key={label}>
							<i
								style={{ background: `rgb(${colour.slice(0, 3).join(",")})` }}
							/>
							{label}
						</div>
					))}
				</div>
			) : (
				<>
					<div
						className="legend-ramp"
						style={{
							background: colourmapGradient(
								result.binding.colourmap,
								result.binding.reverse,
							),
						}}
					/>
					<div className="legend-range">
						<span>
							{result.low.toLocaleString(undefined, {
								maximumFractionDigits: 3,
							})}
						</span>
						<span>
							{result.high.toLocaleString(undefined, {
								maximumFractionDigits: 3,
							})}
						</span>
					</div>
					<small>
						{result.binding.scale === "log1p" ? "log1p(value)" : "Stored value"}
					</small>
				</>
			)}
			{result.missing > 0 && (
				<small>
					{result.missing.toLocaleString()} boundaries have no value (grey)
				</small>
			)}
		</div>
	);
}
