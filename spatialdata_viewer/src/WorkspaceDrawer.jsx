import React, { useEffect, useRef, useState } from "react";
import {
	duplicatePane,
	addIndependentPane,
	setNavigationLinked,
	removePane,
	patchPane,
	snapshot,
	restoreBookmark,
	uid,
} from "./workspace.mjs";
import { genesFromText, importGeneCsv } from "./genes.mjs";

import GeneAvailability from "./GeneAvailability.jsx";
import EmbeddingPanel from "./EmbeddingPanel.jsx";
import FigurePanel from "./FigurePanel.jsx";
import ViewOverlayControls from "./ViewOverlayControls.jsx";
import { viewTools } from "./view-tools.mjs";
const tabs = [
	"Views",
	"Bookmarks",
	"Gene lists",
	"Embeddings",
	"Figure",
	"Workspace",
];
export default function WorkspaceDrawer({
	session,
	setSession,
	dataset,
	onExport,
	onImport,
	saveStatus,
	onSave,
	activeBinding,
	activeRegion,
	activeOverlay,
	highlighted,
	onEmbeddingSelect,
	onEmbeddingFocus,
	paneGroups,
	figureProps,
	selectionMode,
	onSelectionMode,
	onSelectionScope,
	onSelectionApi,
}) {
	const { drawer } = session;
	const [listId, setListId] = useState(""),
		[name, setName] = useState(""),
		[genes, setGenes] = useState(""),
		[error, setError] = useState("");
	const csv = useRef(null),
		file = useRef(null),
		drag = useRef(null);
	const updateDrawer = (patch) =>
		setSession((s) => ({ ...s, drawer: { ...s.drawer, ...patch } }));
	function editList(list) {
		setListId(list?.id || "");
		setName(list?.name || "");
		setGenes(list?.genes.join("\n") || "");
		setError("");
	}
	// Workspace imports and bookmark returns can replace a list's definition.
	useEffect(() => {
		if (listId) editList(session.geneLists.find((g) => g.id === listId));
	}, [session.geneLists]);
	function saveList() {
		try {
			const list = {
				id: listId || uid(),
				name: name.trim(),
				genes: genesFromText(genes),
			};
			if (!list.name || !list.genes.length)
				throw new Error("Enter a list name and at least one gene.");
			if (!listId && session.geneLists.length >= 100)
				throw new Error("Use at most 100 gene lists.");
			setSession((s) => ({
				...s,
				geneLists: listId
					? s.geneLists.map((g) => (g.id === listId ? list : g))
					: [...s.geneLists, list],
			}));
			editList(list);
		} catch (e) {
			setError(e.message);
		}
	}
	async function importCsv(event) {
		const input = event.target.files?.[0];
		event.target.value = "";
		if (!input) return;
		try {
			if (input.size > 2 * 1024 ** 2)
				throw new Error("Gene CSV files must be smaller than 2 MiB.");
			const lists = importGeneCsv(
				await input.text(),
				input.name.replace(/\.csv$/i, ""),
			).map((g) => ({ ...g, id: uid() }));
			if (session.geneLists.length + lists.length > 100)
				throw new Error("Use at most 100 gene lists.");
			setSession((s) => ({ ...s, geneLists: [...s.geneLists, ...lists] }));
			editList(lists[0]);
		} catch (e) {
			setError(e.message);
		}
	}
	function deleteList() {
		setSession((s) => ({
			...s,
			geneLists: s.geneLists.filter((g) => g.id !== listId),
			embedding:
				s.embedding?.colourBy === "sum" && s.embedding.field === listId
					? { ...s.embedding, colourBy: "none", field: "" }
					: s.embedding,
			panes: s.panes.map((p) => ({
				...p,
				cellBindings: Object.fromEntries(
					Object.entries(p.cellBindings).map(([id, b]) => [
						id,
						b.source === "sum" && b.field === listId
							? { ...b, source: "none", field: "" }
							: b,
					]),
				),
				renderStack: {
					...p.renderStack,
					entries: p.renderStack.entries.map((e) =>
						e.props.geneListId === listId
							? {
									...e,
									props: {
										...e.props,
										geneListId: undefined,
										featureNames: [],
									},
								}
							: e,
					),
				},
			})),
		}));
		editList(null);
	}
	function bookmark() {
		if (session.bookmarks.length >= 500) {
			setError("Use at most 500 bookmarks.");
			return;
		}
		const item = {
			id: uid(),
			name: `Bookmark ${session.bookmarks.length + 1}`,
			comment: "",
			createdAt: new Date().toISOString(),
			viewer: snapshot(session),
		};
		setSession((s) => ({ ...s, bookmarks: [...s.bookmarks, item] }));
	}
	const updateBookmark = (id, patch) =>
		setSession((s) => ({
			...s,
			bookmarks: s.bookmarks.map((b) => (b.id === id ? { ...b, ...patch } : b)),
		}));
	return (
		<section
			className={`workspace-drawer ${drawer.open ? "open" : ""}`}
			aria-label="Workspace tools"
		>
			<div className="drawer-bar">
				<button
					aria-expanded={drawer.open}
					aria-label={drawer.open ? "Hide drawer" : "Show drawer"}
					onClick={() => updateDrawer({ open: !drawer.open })}
				>
					{drawer.open ? "▾" : "▴"} Workspace tools
				</button>
				<nav aria-label="Workspace panels">
					{tabs.map((tab) => (
						<button
							key={tab}
							aria-pressed={drawer.open && drawer.tab === tab}
							onClick={() =>
								updateDrawer({
									open: true,
									tab,
									height: [
										"Embeddings",
										"Gene lists",
										"Figure",
										"Views",
									].includes(tab)
										? Math.max(drawer.height, 380)
										: drawer.height,
								})
							}
						>
							{tab}
						</button>
					))}
				</nav>
				<span className="muted" role="status">
					{saveStatus}
				</span>
			</div>
			{drawer.open && (
				<>
					<div
						className="drawer-resize"
						role="separator"
						aria-label="Drawer height"
						aria-orientation="horizontal"
						tabIndex={0}
						aria-valuenow={drawer.height}
						aria-valuemin={180}
						aria-valuemax={600}
						onPointerDown={(e) => {
							drag.current = { y: e.clientY, height: drawer.height };
							e.currentTarget.setPointerCapture(e.pointerId);
						}}
						onPointerMove={(e) => {
							if (drag.current)
								updateDrawer({
									height: Math.max(
										180,
										Math.min(
											600,
											drag.current.height + drag.current.y - e.clientY,
										),
									),
								});
						}}
						onPointerUp={() => {
							drag.current = null;
						}}
						onPointerCancel={() => {
							drag.current = null;
						}}
						onKeyDown={(e) => {
							if (["ArrowUp", "ArrowDown"].includes(e.key)) {
								e.preventDefault();
								updateDrawer({
									height: Math.max(
										180,
										Math.min(
											600,
											drawer.height + (e.key === "ArrowUp" ? 30 : -30),
										),
									),
								});
							}
						}}
					/>
					<div className="drawer-content" style={{ height: drawer.height }}>
						{drawer.tab === "Views" && (
							<div className="drawer-views">
								<div className="drawer-actions">
									<button
										disabled={!dataset || session.panes.length >= 4}
										onClick={() => setSession(duplicatePane(session))}
									>
										Add view
									</button>
									<button
										disabled={!dataset || session.panes.length >= 4}
										onClick={() => setSession(addIndependentPane)}
										title="Add a copy of this view and let all panes pan and zoom independently"
									>
										Add independent view
									</button>
									<label className="checkbox-label">
										<input
											type="checkbox"
											aria-label="Link navigation"
											checked={session.linked}
											onChange={(e) =>
												setSession((s) =>
													setNavigationLinked(s, e.target.checked),
												)
											}
										/>
										Link pan and zoom
									</label>
									<span className="muted">
										Explore different regions independently, or link pan and
										zoom. Linking aligns all views to the selected pane.
									</span>
								</div>
								<div className="pane-cards">
									{session.panes.map((pane, i) => (
										<div
											className={`pane-card ${pane.id === session.activePaneId ? "active" : ""}`}
											key={pane.id}
										>
											<button
												onClick={() =>
													setSession((s) => ({ ...s, activePaneId: pane.id }))
												}
												aria-label={`Edit ${pane.name}`}
											>
												{pane.name}
											</button>
											<input
												aria-label={`Name for view ${i + 1}`}
												defaultValue={pane.name}
												key={pane.name}
												maxLength={120}
												onBlur={(e) =>
													setSession((s) =>
														patchPane(s, pane.id, {
															name: e.target.value.trim() || pane.name,
														}),
													)
												}
											/>
											<div className="pane-overlay-toggles">
												{[
													["legend", "Legend"],
													["scaleBar", "Scale bar"],
													["layerNames", "Layer names"],
												].map(([key, label]) => (
													<label className="checkbox-label" key={key}>
														<input
															type="checkbox"
															aria-label={`${label} in ${pane.name}`}
															checked={viewTools(pane)[key].visible}
															onChange={(e) =>
																setSession((s) =>
																	patchPane(s, pane.id, {
																		[key]: {
																			...viewTools(pane)[key],
																			visible: e.target.checked,
																		},
																	}),
																)
															}
														/>
														{label}
													</label>
												))}
											</div>
											<small>
												{
													pane.renderStack.entries.filter((e) => e.visible)
														.length
												}{" "}
												visible layers
											</small>
											<button
												disabled={session.panes.length === 1}
												aria-label={`Remove ${pane.name}`}
												onClick={() =>
													setSession((s) => removePane(s, pane.id))
												}
											>
												Remove
											</button>
										</div>
									))}
								</div>
								{dataset && (
									<details className="overlay-editor" open>
										<summary>
											Customise overlays ·{" "}
											{
												session.panes.find((p) => p.id === session.activePaneId)
													.name
											}
										</summary>
										<ViewOverlayControls
											pane={session.panes.find(
												(p) => p.id === session.activePaneId,
											)}
											groups={paneGroups?.[session.activePaneId] || []}
											onChange={(patch) =>
												setSession((s) => patchPane(s, s.activePaneId, patch))
											}
											figure={session.figure}
											onFigure={figureProps.onChange}
										/>
									</details>
								)}
								<div className="drawer-actions">
									<button disabled={!dataset} onClick={onExport}>
										Export view settings
									</button>
									<button
										disabled={!dataset}
										onClick={() => file.current.click()}
									>
										Import view settings
									</button>
								</div>
							</div>
						)}
						{drawer.tab === "Bookmarks" && (
							<>
								<div className="drawer-actions">
									<button disabled={!dataset} onClick={bookmark}>
										Bookmark current viewer
									</button>
									<span className="muted">
										Capture all panes, their camera positions and display
										settings.
									</span>
								</div>
								{!session.bookmarks.length && (
									<p className="muted">No bookmarks yet.</p>
								)}
								<table className="bookmark-table">
									<tbody>
										{session.bookmarks.map((b) => (
											<tr key={b.id}>
												<td>
													<button
														aria-label={`Return to bookmark ${b.name}`}
														onClick={() =>
															setSession((s) => restoreBookmark(s, b))
														}
													>
														↩ Return
													</button>
												</td>
												<td>
													<input
														aria-label="Bookmark name"
														defaultValue={b.name}
														maxLength={120}
														onBlur={(e) =>
															updateBookmark(b.id, {
																name: e.target.value.trim() || b.name,
															})
														}
													/>
													<small>
														{b.viewer.coordinateSystem} ·{" "}
														{b.viewer.panes.length} views
													</small>
												</td>
												<td>
													<textarea
														aria-label={`Comments for ${b.name}`}
														value={b.comment}
														maxLength={10000}
														onChange={(e) =>
															updateBookmark(b.id, { comment: e.target.value })
														}
													/>
												</td>
												<td>
													<button
														aria-label={`Remove bookmark ${b.name}`}
														onClick={() =>
															setSession((s) => ({
																...s,
																bookmarks: s.bookmarks.filter(
																	(v) => v.id !== b.id,
																),
															}))
														}
													>
														Remove
													</button>
												</td>
											</tr>
										))}
									</tbody>
								</table>
							</>
						)}
						{drawer.tab === "Figure" && (
							<FigurePanel
								session={session}
								dataset={dataset}
								{...figureProps}
							/>
						)}
						{drawer.tab === "Gene lists" && (
							<div className="gene-editor">
								<div>
									<div className="drawer-actions">
										<button disabled={!dataset} onClick={() => editList(null)}>
											New gene list
										</button>
										<button
											disabled={!dataset}
											onClick={() => csv.current.click()}
										>
											Import gene CSV
										</button>
									</div>
									<p className="muted">
										CSV: one gene per row, or columns <code>list,gene</code>.
										Gene names match exactly; duplicates count once.
									</p>
									<div className="saved-gene-lists">
										{session.geneLists.map((list) => (
											<button
												key={list.id}
												aria-pressed={listId === list.id}
												onClick={() => editList(list)}
											>
												{list.name} <small>{list.genes.length} genes</small>
											</button>
										))}
									</div>
									<GeneAvailability
										dataset={dataset}
										activeTable={activeBinding?.table}
										text={genes}
									/>
								</div>
								<div>
									<label>
										List name
										<input
											aria-label="Gene list name"
											value={name}
											maxLength={120}
											onChange={(e) => setName(e.target.value)}
										/>
									</label>
									<label>
										Genes
										<textarea
											aria-label="Gene list genes"
											value={genes}
											onChange={(e) => setGenes(e.target.value)}
											placeholder="One gene per line, or separated by commas"
										/>
									</label>
									<div className="drawer-actions">
										<button disabled={!dataset} onClick={saveList}>
											{listId ? "Save gene list" : "Create gene list"}
										</button>
										{listId && (
											<button onClick={deleteList}>Delete gene list</button>
										)}
									</div>
								</div>
							</div>
						)}
						{drawer.tab === "Workspace" && (
							<>
								<h3>Saved on this browser</h3>
								<p>
									Views, bookmarks and gene lists save automatically. After a
									reload, reopen the same folder or ZIP to reconnect your local
									files and restore the workspace.
								</p>
								<p className="muted">
									Export a workspace JSON for a portable backup. It contains
									settings, not images, tables or transcripts.
								</p>
								<div className="drawer-actions">
									<button disabled={!dataset} onClick={onSave}>
										Save locally now
									</button>
									<button disabled={!dataset} onClick={onExport}>
										Export workspace
									</button>
									<button
										disabled={!dataset}
										onClick={() => file.current.click()}
									>
										Import workspace
									</button>
								</div>
							</>
						)}
						{drawer.tab === "Embeddings" && (
							<EmbeddingPanel
								selectionMode={selectionMode}
								onSelectionMode={onSelectionMode}
								onSelectionScope={onSelectionScope}
								onSelectionApi={onSelectionApi}
								dataset={dataset}
								options={session.embedding}
								onChange={(embedding) =>
									setSession((s) => ({ ...s, embedding }))
								}
								activeBinding={activeBinding}
								activeRegion={activeRegion}
								activeOverlay={activeOverlay}
								geneLists={session.geneLists}
								highlighted={highlighted}
								onSelect={onEmbeddingSelect}
								onFocus={onEmbeddingFocus}
							/>
						)}
						{error && (
							<p className="error" role="alert">
								{error}
							</p>
						)}
					</div>
				</>
			)}
			<input
				type="file"
				ref={csv}
				accept=".csv,text/csv"
				aria-label="Gene list CSV"
				hidden
				onChange={importCsv}
			/>
			<input
				type="file"
				ref={file}
				accept=".json"
				aria-label="View settings"
				hidden
				onChange={onImport}
			/>
		</section>
	);
}
