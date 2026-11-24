import React, { useEffect, useState } from "react";
import { matchGenes } from "./genes.mjs";
export default function TranscriptControls({
	element,
	entry,
	onChange,
	geneLists = [],
	propagation,
}) {
	const [catalog, setCatalog] = useState(null),
		[error, setError] = useState(""),
		[query, setQuery] = useState(""),
		[count, setCount] = useState(null);
	useEffect(() => {
		let live = true;
		setCatalog(null);
		setError("");
		setCount(null);
		element
			.listFeatures()
			.then((c) => {
				if (live) setCatalog(c);
			})
			.catch((e) => {
				if (live) setError(e.message);
			});
		element
			.getParquetRowCount()
			.then((n) => {
				if (live) setCount(n);
			})
			.catch(() => {});
		return () => {
			live = false;
		};
	}, [element]);
	const names = entry.props.featureNames;
	const listId = entry.props.geneListId;
	const list = geneLists.find((g) => g.id === listId);
	const matches =
		list && catalog
			? matchGenes(
					list.genes,
					catalog.entries.map((v) => v.name),
				)
			: null;
	return (
		<div className="transcript-controls">
			<label>
				Transcript genes
				<select
					aria-label="Transcript filter"
					value={
						listId !== undefined
							? "list"
							: names === undefined
								? "all"
								: "selected"
					}
					onChange={(e) =>
						onChange({
							geneListId:
								e.target.value === "list" ? geneLists[0]?.id || "" : undefined,
							featureNames: e.target.value === "all" ? undefined : [],
							colorByFeature: true,
						})
					}
				>
					<option value="all">All genes</option>
					<option value="selected">Selected genes</option>
					<option value="list">Workspace gene list</option>
				</select>
			</label>
			{error && <p className="error">{error}</p>}
			{!catalog && !error && <p role="status">Reading transcript genes…</p>}
			{listId !== undefined && (
				<>
					<label>
						Gene list
						<select
							aria-label="Transcript gene list"
							value={listId}
							onChange={(e) =>
								onChange({
									geneListId: e.target.value,
									featureNames: undefined,
									colorByFeature: true,
								})
							}
						>
							<option value="">Choose a gene list</option>
							{geneLists.map((g) => (
								<option key={g.id} value={g.id}>
									{g.name}
								</option>
							))}
						</select>
					</label>
					<p className="muted">
						Create or import lists in the Gene lists drawer.
					</p>
					{matches && (
						<p className="muted" role="status">
							{matches.matched.length} / {list.genes.length} list genes
							available
							{matches.missing.length > 0 && (
								<> · Missing from points: {matches.missing.join(", ")}</>
							)}
						</p>
					)}
				</>
			)}
			{listId === undefined && names !== undefined && (
				<>
					<input
						type="search"
						aria-label="Search transcript genes"
						placeholder="Find a transcript gene…"
						value={query}
						onChange={(e) => setQuery(e.target.value)}
					/>
					<p className="muted">{names.length} selected</p>
					<div className="gene-list">
						{catalog?.entries
							.filter((v) => v.name.toLowerCase().includes(query.toLowerCase()))
							.map((v) => (
								<label className="checkbox-label" key={v.name}>
									<input
										type="checkbox"
										aria-label={`Transcript gene ${v.name}`}
										checked={names.includes(v.name)}
										onChange={(e) =>
											onChange({
												featureNames: e.target.checked
													? [...names, v.name]
													: names.filter((n) => n !== v.name),
											})
										}
									/>
									{v.name}
								</label>
							))}
					</div>
				</>
			)}
			{propagation}
			{catalog && (
				<p className="muted">
					{catalog.entries.length.toLocaleString()} transcript genes available
				</p>
			)}
			{count !== null && (
				<p className="muted">
					{count.toLocaleString()} transcripts in this layer
				</p>
			)}
		</div>
	);
}
