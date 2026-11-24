import React, { useEffect, useState } from "react";
import { tableGenes } from "./table-genes.mjs";
import { genesFromText, matchGenes } from "./genes.mjs";
export default function GeneAvailability({ dataset, activeTable, text }) {
	const [chosen, setChosen] = useState(""),
		[state, setState] = useState({});
	const tableKey =
		chosen || activeTable || Object.keys(dataset?.tables || {})[0] || "";
	const table = dataset?.tables[tableKey];
	useEffect(() => {
		let live = true;
		setState({ loading: true });
		if (!table) {
			setState({});
			return;
		}
		tableGenes(dataset, tableKey)
			.then((names) => {
				if (live) setState({ names });
			})
			.catch((e) => {
				if (live) setState({ error: e.message });
			});
		return () => {
			live = false;
		};
	}, [dataset, table, tableKey]);
	let genes = [],
		error = "";
	try {
		genes = genesFromText(text);
	} catch (e) {
		error = e.message;
	}
	const match = state.names && matchGenes(genes, state.names);
	const duplicates = new Set(),
		seen = new Set();
	for (const name of state.names || []) {
		if (seen.has(name)) duplicates.add(name);
		seen.add(name);
	}
	const ambiguous = match?.matched.filter((g) => duplicates.has(g)) || [];
	return (
		<section className="gene-availability" aria-label="Gene availability">
			<label>
				Check against table
				<select
					aria-label="Gene check table"
					value={chosen}
					onChange={(e) => setChosen(e.target.value)}
					disabled={!dataset}
				>
					<option value="">
						Selected cell table{activeTable ? ` · ${activeTable}` : ""}
					</option>
					{Object.keys(dataset?.tables || {}).map((k) => (
						<option key={k}>{k}</option>
					))}
				</select>
			</label>
			{!table && (
				<p className="muted">
					Open a dataset with an AnnData table to check genes.
				</p>
			)}
			{state.loading && <p role="status">Reading table genes…</p>}
			{(state.error || error) && (
				<p className="error" role="alert">
					{state.error || error}
				</p>
			)}
			{match && !state.error && !error && (
				<>
					<p
						role="status"
						className={
							match.missing.length || ambiguous.length
								? "warning"
								: "link-status"
						}
					>
						{match.matched.length} / {genes.length} genes present in {tableKey}{" "}
						· X
					</p>
					<p className="muted">
						Checks gene names, including case, against the table’s var index.
						Presence does not imply nonzero expression.
					</p>
					{match.missing.length > 0 && (
						<div className="warning">
							<strong>Missing ({match.missing.length})</strong>
							<p>{match.missing.join(", ")}</p>
						</div>
					)}
					{ambiguous.length > 0 && (
						<p className="warning">
							Ambiguous duplicate names in table: {ambiguous.join(", ")}.
							Expression reads cannot resolve these genes.
						</p>
					)}
					{match.matched.length > 0 && (
						<details>
							<summary>Present ({match.matched.length})</summary>
							<p className="gene-match-names">{match.matched.join(", ")}</p>
						</details>
					)}
				</>
			)}
		</section>
	);
}
