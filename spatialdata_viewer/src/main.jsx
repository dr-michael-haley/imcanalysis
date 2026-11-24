import React, {
	Component,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { yieldToBrowser } from "../../web_viewer/src/loading.js";
import { createReadTracker } from "./loading.mjs";
import { openDirectoryStore } from "./directory-store.mjs";
import { withConsolidatedMetadata } from "zarrita";
import LoadingIndicator from "./LoadingIndicator.jsx";
import WelcomeScreen from "./WelcomeScreen.jsx";
import { readZarr } from "@spatialdata/core";
import { renderStackSchema } from "@spatialdata/layers";
import { ensureWorkers, getChunkCache } from "@spatialdata/vis";
import parquetWorkerUrl from "./parquet.worker.js?worker&url";
import { FolderStore, ZipStore } from "zarrextra";
import ViewerPane from "./ViewerPane.jsx";
import WorkspaceDrawer from "./WorkspaceDrawer.jsx";
import {
	newSession,
	setNavigationLinked,
	patchPane,
	cameraChange,
	exportWorkspace,
	importWorkspace,
	snapshot,
	uid,
} from "./workspace.mjs";
import {
	fileIdentity,
	directoryIdentity,
	readLocal,
	writeLocal,
	recentLocal,
} from "./persistence.mjs";
import { useOverlays } from "./useOverlays.mjs";
import CellControls from "./CellControls.jsx";
import TranscriptControls from "./TranscriptControls.jsx";
import PropagationControls from "./PropagationControls.jsx";
import { usePropagation } from "./usePropagation.mjs";
import { defaultBinding } from "./overlays.mjs";
import { cellKey } from "./embedding-model.mjs";
import { cellCentre } from "./embedding-data.mjs";
import LayerManager, { layerLabel } from "./LayerManager.jsx";
import Icon from "./icons.jsx";
import ViewerToolbar from "./ViewerToolbar.jsx";
import FigureRenderer from "./FigureRenderer.jsx";
import { figureLayout, validateFigure, figureFilename } from "./view-tools.mjs";
import { createFigureExport } from "./figure-export.mjs";
import { mergeSelection } from "./spatial-selection.mjs";
import {
	download,
	elementKind,
	elementKinds,
	pretty,
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
function Section({ title, children, open = true, className = "" }) {
	return (
		<details className={`section ${className}`} open={open}>
			<summary>{title}</summary>
			<div className="section-body">{children}</div>
		</details>
	);
}
function App() {
	const [dataset, setDataset] = useState(null);
	const [source, setSource] = useState("");
	const [sourceInfo, setSourceInfo] = useState("");
	const [loading, setLoading] = useState(null);
	const [readActivity, setReadActivity] = useState(null);
	const status = loading?.title || "";
	const [error, setError] = useState("");
	const [session, setSession] = useState(() => newSession(null, ""));
	const tab = session.sidebar?.tab || "Setup";
	const setTab = useCallback(
		(tab) => setSession((s) => ({ ...s, sidebar: { ...s.sidebar, tab } })),
		[],
	);
	const [selectionMode, setSelectionMode] = useState("inspect"),
		[selectionScope, setSelectionScope] = useState("tissue"),
		[paneGroups, setPaneGroups] = useState({}),
		[figureJob, setFigureJob] = useState(null),
		[figureBusy, setFigureBusy] = useState(false),
		[figureStatus, setFigureStatus] = useState(""),
		[viewportSize, setViewportSize] = useState({ width: 1000, height: 700 });
	const figureAbort = useRef(null),
		selectionApis = useRef({});
	const tissueSelectionApi = useCallback((api) => {
		selectionApis.current.tissue = api;
	}, []);
	const embeddingSelectionApi = useCallback((api) => {
		selectionApis.current.embedding = api;
	}, []);
	const onGroups = useCallback(
		(id, groups) =>
			setPaneGroups((old) =>
				JSON.stringify(old[id]) === JSON.stringify(groups)
					? old
					: { ...old, [id]: groups },
			),
		[],
	);
	const changePane = useCallback(
		(id, patch) => setSession((s) => patchPane(s, id, patch)),
		[],
	);
	const changeFigure = (value) => {
		try {
			const next = validateFigure(value);
			setSession((s) => ({ ...s, figure: next }));
			setFigureStatus("");
		} catch (e) {
			setFigureStatus(e.message);
		}
	};
	const propagation = usePropagation(dataset, session, setSession);
	const activeId = session.activePaneId;
	const pane = session.panes.find((p) => p.id === activeId);
	const { coordinateSystem, geneLists } = session;
	const { renderStack: stack, cellBindings, viewState, selectedId } = pane;
	const setStack = useCallback(
		(value) =>
			setSession((s) =>
				patchPane(s, s.activePaneId, (p) => ({
					renderStack:
						typeof value === "function" ? value(p.renderStack) : value,
				})),
			),
		[],
	);
	const setCellBindings = useCallback(
		(value) =>
			setSession((s) =>
				patchPane(s, s.activePaneId, (p) => ({
					cellBindings:
						typeof value === "function" ? value(p.cellBindings) : value,
				})),
			),
		[],
	);
	const setSelectedId = useCallback(
		(id) => setSession((s) => patchPane(s, s.activePaneId, { selectedId: id })),
		[],
	);
	const setViewState = useCallback(
		(camera) => setSession((s) => cameraChange(s, s.activePaneId, camera)),
		[],
	);
	const onCamera = useCallback(
		(id, camera) => setSession((s) => cameraChange(s, id, camera)),
		[],
	);
	const onActivate = useCallback((id) => {
		setSession((s) => (s.activePaneId === id ? s : { ...s, activePaneId: id }));
		setSelection((old) => (old?.paneId === id ? old : null));
	}, []);
	const [localKey, setLocalKey] = useState(""),
		[saveStatus, setSaveStatus] = useState(""),
		[recent, setRecent] = useState([]);
	const saveRecord = useRef(null),
		sourceDirectory = useRef(null),
		pendingSave = useRef(Promise.resolve());
	const flushSave = useCallback(() => {
		const record = saveRecord.current;
		if (!record) return Promise.resolve();
		const write = pendingSave.current
			.catch(() => {})
			.then(() => writeLocal(record));
		pendingSave.current = write;
		return write
			.then(() => {
				if (saveRecord.current === record) setSaveStatus("Saved locally");
			})
			.catch((e) => {
				setSaveStatus(
					`Local save failed: ${e.message}. Export a workspace backup.`,
				);
			});
	}, []);
	useEffect(() => {
		recentLocal()
			.then(setRecent)
			.catch(() => {});
	}, []);
	useEffect(() => {
		if (!dataset || !localKey) return;
		saveRecord.current = {
			key: localKey,
			name: source,
			savedAt: Date.now(),
			workspace: exportWorkspace(dataset, source, session),
			...(sourceDirectory.current
				? { directoryHandle: sourceDirectory.current }
				: {}),
		};
		setSaveStatus("Saving locally…");
		const timer = setTimeout(flushSave, 700);
		return () => clearTimeout(timer);
	}, [session, dataset, localKey, source, flushSave]);
	useEffect(() => {
		const save = () => {
			if (document.visibilityState === "hidden") void flushSave();
		};
		document.addEventListener("visibilitychange", save);
		window.addEventListener("pagehide", flushSave);
		return () => {
			document.removeEventListener("visibilitychange", save);
			window.removeEventListener("pagehide", flushSave);
		};
	}, [flushSave]);
	const [selection, setSelection] = useState(null);
	const [embeddingSelection, setEmbeddingSelection] = useState([]);
	const onSelectCells = useCallback((points, mode = "replace") => {
		setEmbeddingSelection((old) => mergeSelection(old, points, mode));
		setSelection(null);
	}, []);
	const highlighted = useMemo(
		() =>
			new Set([
				...embeddingSelection.map((p) => cellKey(p.region, p.id)),
				...(selection && ["shapes", "labels"].includes(selection.kind)
					? [cellKey(selection.element, selection.id)]
					: []),
			]),
		[embeddingSelection, selection],
	);
	useEffect(() => {
		setSelection((old) => (old?.paneId === activeId ? old : null));
	}, [activeId]);
	const [channelTarget, setChannelTarget] = useState(null);
	const [layerStatuses, setLayerStatuses] = useState({});
	const [readNotices, setReadNotices] = useState([]);
	const [gpuError, setGpuError] = useState("");
	const request = useRef(0);
	const readTracker = useRef(null);
	const openAbort = useRef(null);
	const pickerCancelled = useRef(false);
	const pickerRevision = useRef(0);
	useEffect(
		() => () => {
			readTracker.current?.dispose();
			openAbort.current?.abort();
		},
		[],
	);
	const folderInput = useRef(null);
	const zipInput = useRef(null);
	useEffect(() => {
		// React's delegated cancel event does not cover native file-picker cancel
		// consistently. Listen directly on the two persistent input elements.
		const inputs = [folderInput.current, zipInput.current];
		const cancelled = () =>
			setLoading((old) => (old?.phase === "picker" ? null : old));
		for (const input of inputs) input?.addEventListener("cancel", cancelled);
		return () => {
			for (const input of inputs)
				input?.removeEventListener("cancel", cancelled);
		};
	}, []);
	const viewport = useRef(null);
	useEffect(() => {
		const node = viewport.current;
		if (!node) return;
		const observer = new ResizeObserver(() => {
			const rect = node.getBoundingClientRect();
			setViewportSize({ width: rect.width, height: rect.height });
		});
		observer.observe(node);
		return () => observer.disconnect();
	}, []);
	useEffect(() => {
		figureAbort.current?.abort();
		setPaneGroups({});
		setSelectionMode("inspect");
		return () => figureAbort.current?.abort();
	}, [dataset]);
	const selected = stack.entries.find((entry) => entry.id === selectedId);
	const overlays = useOverlays(dataset, session.panes, geneLists);
	const binding =
		dataset && selected?.source.elementType === "shapes"
			? cellBindings[selected.id] || defaultBinding(dataset, selected)
			: null;
	const propagationControls =
		dataset && selected ? (
			<PropagationControls
				dataset={dataset}
				session={session}
				paneId={activeId}
				entryId={selected.id}
				state={propagation.state}
				onApply={propagation.propagate}
			/>
		) : null;
	const selectedOverlay =
		selection?.kind === "shapes"
			? overlays[`${selection.paneId}/shapes:${selection.element}`]
			: null;
	const selectedRow = selectedOverlay?.rows?.get(String(selection?.id));
	const activeShape =
		selected?.source.elementType === "shapes"
			? selected
			: stack.entries.find(
					(e) => e.visible && e.source.elementType === "shapes",
				);
	const activeBinding =
		dataset && activeShape
			? cellBindings[activeShape.id] || defaultBinding(dataset, activeShape)
			: null;
	const activeRegion = activeShape?.source.elementKey;
	const activeOverlay = activeShape
		? overlays[`${activeId}/${activeShape.id}`]
		: null;
	const onEmbeddingSelect = useCallback(
		(points, mode = "replace") => {
			setEmbeddingSelection((old) => mergeSelection(old, points, mode));
			setSelection(null);
			if (points.length === 1) {
				const p = points[0],
					region = p.region.replace(/^(shapes|labels)\//, "");
				const kind = dataset.shapes[region]
					? "shapes"
					: dataset.labels[region]
						? "labels"
						: "table";
				setSelection({
					paneId: activeId,
					kind,
					element: region,
					id: p.id,
					rowIndex: p.row,
					embeddingTable: p.table,
					embeddingLabel: p.label,
					embeddingValue: p.value,
					tooltip: { items: [{ label: "Table row", value: String(p.row) }] },
				});
				setTab("Inspect");
			}
		},
		[dataset, activeId],
	);
	async function onEmbeddingFocus(point) {
		const generation = request.current;
		try {
			const target = await cellCentre(
				dataset,
				point.region,
				point.id,
				coordinateSystem,
			);
			if (generation !== request.current) return;
			if (!target) {
				setError(
					"This table cell has no matching shape in the current coordinate system.",
				);
				return;
			}
			const id = `shapes:${point.region.replace(/^shapes\//, "")}`;
			setSession((s) =>
				cameraChange(
					patchPane(s, s.activePaneId, (p) => ({
						selectedId: id,
						renderStack: {
							...p.renderStack,
							entries: p.renderStack.entries.map((e) =>
								e.id === id ? { ...e, visible: true } : e,
							),
						},
					})),
					s.activePaneId,
					{
						zoom: 0,
						...s.panes.find((p) => p.id === s.activePaneId).viewState,
						target,
					},
				),
			);
			setError("");
		} catch (e) {
			setError(e.message);
		}
	}
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
	const entryStatus = useCallback((paneId, layerId, value) => {
		const id = `${paneId}/${layerId}`;
		return setLayerStatuses((previous) => {
			if (previous[id] === value) return previous;
			const next = { ...previous };
			if (value) next[id] = value;
			else delete next[id];
			return next;
		});
	}, []);
	const onFeatureClick = useCallback(
		(paneId, event) => {
			if (
				selectionMode === "click" &&
				["shapes", "labels"].includes(event.elementKind)
			) {
				onSelectCells(
					[{ id: String(event.featureId), region: event.spatialElement.key }],
					"toggle",
				);
				return;
			}
			setEmbeddingSelection([]);
			setSession((s) => ({ ...s, activePaneId: paneId }));
			setSelection({
				paneId,
				kind: event.elementKind,
				element: event.spatialElement.key,
				id: event.featureId,
				rowIndex: event.rowIndex,
				tooltip: event.tooltip,
			});
			setTab("Inspect");
		},
		[selectionMode, onSelectCells],
	);
	function resetView(sdata, system) {
		setEmbeddingSelection([]);
		setSession((old) => ({
			...newSession(sdata, system),
			geneLists: old.geneLists,
			bookmarks: old.bookmarks,
			drawer: old.drawer,
		}));
		setSelection(null);
		setLayerStatuses({});
		setGpuError("");
	}
	async function openStore(makeStore, name, info, files) {
		const generation = ++request.current;
		openAbort.current?.abort();
		const controller = new AbortController();
		openAbort.current = controller;
		readTracker.current?.dispose();
		setReadActivity(null);
		const tracker = createReadTracker((activity) => {
			if (generation === request.current) setReadActivity(activity);
		});
		readTracker.current = tracker;
		const startedAt = Date.now();
		const phase = (title, detail) =>
			setLoading({ title, detail, name, startedAt });
		phase(
			"Indexing dataset…",
			files
				? `${files.length.toLocaleString()} selected files.`
				: "Reading folder metadata without scanning image chunks.",
		);
		setEmbeddingSelection([]);
		const saved = flushSave();
		sourceDirectory.current = null;
		setError("");
		// Release the previous rendering tree before opening a second store.
		setDataset(null);
		setLocalKey("");
		saveRecord.current = null;
		setSelection(null);
		setLayerStatuses({});
		setReadNotices([]);
		getChunkCache()?.clear();
		try {
			await yieldToBrowser();
			await saved;
			if (generation !== request.current) return;
			const rawStore = await makeStore({
				signal: controller.signal,
				onProgress: ({ directories, nodes, path }) => {
					if (generation === request.current)
						phase(
							"Discovering dataset…",
							`${nodes.toLocaleString()} metadata entries · ${directories.toLocaleString()} folders checked · ${path}`,
						);
				},
			});
			if (generation !== request.current) return;
			const store = tracker.wrap(rawStore);
			phase(
				"Reading SpatialData metadata…",
				"Discovering images, shapes, points and tables.",
			);
			await yieldToBrowser();
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
			phase(
				"Restoring workspace…",
				"Checking for saved views, bookmarks and gene lists.",
			);
			await yieldToBrowser();
			if (generation !== request.current) return;
			let key;
			try {
				key = rawStore.directoryHandle
					? await directoryIdentity(rawStore.directoryHandle)
					: await fileIdentity(files);
			} catch (e) {
				if (!rawStore.directoryHandle) throw e;
				key = `directory:${crypto.randomUUID()}`;
			}
			let next = newSession(
				sdata,
				sdata.coordinateSystems.includes("global")
					? "global"
					: [...sdata.coordinateSystems].sort()[0],
			);
			let notice = "Local state enabled";
			try {
				const saved = await readLocal(key);
				if (saved) {
					next = importWorkspace(saved.workspace, sdata);
					notice = "Restored local workspace";
				}
			} catch (e) {
				notice = `Could not restore local workspace: ${e.message}`;
			}
			if (generation !== request.current) return;
			setDataset(sdata);
			window.sbtRecordAction?.("dataset_opened");
			sourceDirectory.current = rawStore.directoryHandle || null;
			setSession(next);
			setLocalKey(key);
			setSource(name);
			setSourceInfo(info);
			setTab("Layers");
			setReadNotices(skipped);
			setSaveStatus(notice);
		} catch (failure) {
			tracker.dispose();
			if (generation === request.current) {
				setReadActivity(null);
				setError(failure.message || String(failure));
				setTab("Setup");
			}
		} finally {
			if (generation === request.current) setLoading(null);
		}
	}
	function chooseFiles(kind) {
		pickerRevision.current++;
		pickerCancelled.current = false;
		flushSync(() =>
			setLoading({
				phase: "picker",
				startedAt: Date.now(),
				title:
					kind === "folder"
						? "Waiting for folder selection and file listing…"
						: "Waiting for ZIP selection…",
				detail:
					kind === "folder"
						? "After you choose a folder, the browser lists its files before passing them to the viewer. Large network folders can take a while."
						: "Choose a SpatialData ZIP archive in the file picker to continue.",
			}),
		);
		(kind === "folder" ? folderInput : zipInput).current.click();
	}
	async function chooseDirectory() {
		const ticket = ++pickerRevision.current;
		pickerCancelled.current = false;
		setError("");
		flushSync(() =>
			setLoading({
				phase: "picker",
				startedAt: Date.now(),
				title: "Choose a local or network folder…",
				detail:
					"Select the SpatialData .zarr folder. Only read access is requested; data stays on your computer.",
			}),
		);
		try {
			const handle = await window.showDirectoryPicker({
				mode: "read",
				id: "merscope-dataset",
			});
			if (pickerCancelled.current || ticket !== pickerRevision.current) return;
			await openStore(
				(options) =>
					openDirectoryStore(handle, {
						...options,
						consolidate: withConsolidatedMetadata,
					}),
				handle.name,
				"Direct folder access · read on demand",
				null,
			);
		} catch (e) {
			if (pickerCancelled.current || ticket !== pickerRevision.current) return;
			setLoading(null);
			if (e.name !== "AbortError")
				setError(
					`Could not open this folder: ${e.message}. Try the compatibility picker or a local copy.`,
				);
		}
	}
	function pickerClosed() {
		setLoading((old) => (old?.phase === "picker" ? null : old));
	}
	function cancelLoading() {
		pickerRevision.current++;
		if (loading?.phase === "picker") {
			pickerCancelled.current = true;
			setLoading(null);
		} else void close();
	}
	function openFolder(event) {
		const files = Array.from(event.target.files || []);
		event.target.value = "";
		if (pickerCancelled.current) return;
		if (!files.length) return pickerClosed();
		const name = files[0].webkitRelativePath.split("/")[0] || "Local folder";
		openStore(
			() => FolderStore.fromFiles(files).withListing(),
			name,
			`${files.length.toLocaleString()} local files · read on demand`,
			files,
		);
	}
	function openZip(event) {
		const file = event.target.files?.[0];
		event.target.value = "";
		if (pickerCancelled.current) return;
		if (!file) return pickerClosed();
		if (file)
			openStore(
				async () => (await ZipStore.open(file)).withListing(),
				file.name,
				`${(file.size / 1024 ** 2).toFixed(1)} MiB archive · read on demand`,
				[file],
			);
	}
	async function close() {
		pickerRevision.current++;
		request.current++;
		openAbort.current?.abort();
		readTracker.current?.dispose();
		readTracker.current = null;
		setReadActivity(null);
		const saved = flushSave();
		sourceDirectory.current = null;
		setLocalKey("");
		saveRecord.current = null;
		setDataset(null);
		setSource("");
		setSession(newSession(null, ""));
		setSaveStatus("");
		recentLocal()
			.then(setRecent)
			.catch(() => {});
		setSourceInfo("");
		setLoading(null);
		setError("");
		setSelection(null);
		setLayerStatuses({});
		setReadNotices([]);
		setGpuError("");
		getChunkCache()?.clear();
		await saved;
	}
	async function importView(event) {
		const file = event.target.files?.[0];
		event.target.value = "";
		if (!file || !dataset) return;
		try {
			if (file.size > 20 * 1024 ** 2)
				throw new Error("Workspace files must be smaller than 20 MiB.");
			const next = importWorkspace(JSON.parse(await file.text()), dataset);
			for (const p of next.panes)
				p.renderStack = renderStackSchema.parse(p.renderStack);
			for (const b of next.bookmarks)
				for (const p of b.viewer.panes)
					p.renderStack = renderStackSchema.parse(p.renderStack);
			setSession(next);
			setSelection(null);
			setEmbeddingSelection([]);
			setTab("Layers");

			setError("");
		} catch (failure) {
			setError(failure.message || String(failure));
		}
	}
	function exportView() {
		download(
			"merscope-workspace.json",
			JSON.stringify(exportWorkspace(dataset, source, session), null, 2),
		);
	}
	function zoom(delta) {
		if (viewState)
			setViewState({
				...viewState,
				zoom: Math.max(-20, Math.min(20, viewState.zoom + delta)),
			});
	}
	function figureMode(mode) {
		if (mode === session.figure.mode) return;
		setSelectionMode("inspect");
		if (mode === "figure") {
			const original = Object.fromEntries(
				session.panes.map((p) => {
					const rect = viewport.current
						.querySelector(`[data-pane-id="${p.id}"] .pane-canvas`)
						.getBoundingClientRect();
					return [
						p.id,
						{
							camera: structuredClone(p.viewState),
							width: rect.width,
							height: rect.height,
						},
					];
				}),
			);
			setSession((s) => ({
				...s,
				figureReturn: original,
				figure: { ...s.figure, mode },
			}));
		} else
			setSession((s) => ({
				...s,
				panes: s.panes.map((p) =>
					s.figureReturn?.[p.id]
						? { ...p, viewState: s.figureReturn[p.id].camera }
						: p,
				),
				figureReturn: null,
				figure: { ...s.figure, mode },
			}));
	}
	function bookmarkCurrent(name) {
		setSession((s) =>
			s.bookmarks.length >= 500
				? s
				: {
						...s,
						bookmarks: [
							...s.bookmarks,
							{
								id: uid(),
								name: name || `Bookmark ${s.bookmarks.length + 1}`,
								comment: "",
								createdAt: new Date().toISOString(),
								viewer: snapshot(s),
							},
						],
					},
		);
	}
	async function exportFigure(context) {
		if (figureBusy) return;
		const controller = new AbortController();
		figureAbort.current = controller;
		setFigureBusy(true);
		setFigureStatus("Preparing figure…");
		const exportSession = structuredClone(session);
		try {
			const filename = figureFilename(exportSession.figure, context);
			if (
				exportSession.figure.autoBookmark &&
				exportSession.bookmarks.length >= 500
			)
				throw new Error(
					"The workspace has 500 bookmarks. Remove one or disable automatic bookmarking before exporting.",
				);
			const sizes = Object.fromEntries(
				session.panes.map((p) => {
					const r = viewport.current
						.querySelector(`[data-pane-id="${p.id}"] .pane-canvas`)
						.getBoundingClientRect();
					return [p.id, { width: r.width, height: r.height }];
				}),
			);
			const blob = await createFigureExport({
				session: exportSession,
				dataset,
				groups: paneGroups,
				overlays,
				rect: viewportSize,
				sizes,
				signal: controller.signal,
				progress: setFigureStatus,
				render: (job) =>
					new Promise((resolve, reject) =>
						setFigureJob({ ...job, id: uid(), resolve, reject }),
					),
			});
			controller.signal.throwIfAborted();
			download(filename, blob, blob.type);
			window.sbtRecordAction?.("figure_exported");
			if (exportSession.figure.autoBookmark) {
				const item = {
					id: uid(),
					name: filename.slice(0, 120),
					comment: "Figure export",
					createdAt: new Date().toISOString(),
					viewer: snapshot(exportSession),
				};
				setSession((s) => ({ ...s, bookmarks: [...s.bookmarks, item] }));
			}
			setFigureStatus(
				`Exported ${filename}${exportSession.figure.autoBookmark ? " · bookmark saved" : ""}.`,
			);
		} catch (e) {
			setFigureStatus(
				controller.signal.aborted ? "Figure export cancelled." : e.message,
			);
		} finally {
			setFigureBusy(false);
			setFigureJob(null);
		}
	}
	const figureLayoutState =
		session.figure?.mode === "figure"
			? figureLayout(
					session.panes.length,
					viewportSize.width,
					viewportSize.height,
					session.figure,
				)
			: null;
	const failures = [
		...readNotices,
		...Object.entries(layerStatuses)
			.filter(([id]) =>
				stack.entries.some((e) => `${activeId}/${e.id}` === id && e.visible),
			)
			.flatMap(([id, value]) => [
				...value.errors.map((item) => `${id}: ${item.error.message}`),
				...value.notices.map(
					(item) => `${id}: ${item.message || item.code || pretty(item)}`,
				),
			]),
	];
	return (
		<main
			className={`app-shell ${session.sidebar?.hidden ? "sidebar-hidden" : ""} ${session.figure.mode === "figure" ? "figure-mode-active" : ""}`}
		>
			<aside className="sidebar">
				<header className="brand-area">
					<img src="./logo.svg" alt="SpatialBiologyToolkit" />
					<div>
						<div className="brand-title">MERSCOPE Explorer</div>
						<div className="brand-subtitle">
							SpatialBiologyToolkit <span>v{__VIEWER_VERSION__}</span>
						</div>
						<nav className="brand-links" aria-label="MERSCOPE Explorer help">
							<a
								href="https://imcanalysis.readthedocs.io/en/latest/browser_tools/merscope_explorer.html"
								target="_blank"
								rel="noopener noreferrer"
								title="MERSCOPE Explorer user guide (opens in a new tab)"
							>
								User guide
							</a>
							<a
								className="report-issue"
								href={`mailto:michael.haley@manchester.ac.uk?subject=${encodeURIComponent(`SBT MERSCOPE Explorer issue (v${__VIEWER_VERSION__})`)}`}
							>
								Report an issue
							</a>
						</nav>
					</div>
				</header>
				<nav className="sidebar-tabs" aria-label="Viewer panels">
					{["Setup", "Layers", "Inspect"].map((name) => (
						<button
							key={name}
							aria-pressed={tab === name}
							onClick={() => setTab(name)}
						>
							{name}
						</button>
					))}
				</nav>
				<div hidden={tab !== "Setup"}>
					<div className="local-badge">
						<span className="privacy-dot" />
						Local files · browser processing
					</div>
					<Section number="01" title="Open your dataset">
						<p className="muted">
							Choose a SpatialData store from your computer. Your data stays
							here.
						</p>
						{typeof window.showDirectoryPicker === "function" && (
							<>
								<button
									className="file-button primary"
									disabled={Boolean(loading)}
									onClick={chooseDirectory}
								>
									Open local / network folder <span>.zarr</span>
								</button>
								<p className="muted">
									Recommended for large stores and mapped network drives.
								</p>
							</>
						)}
						<button
							className="file-button"
							onClick={() => chooseFiles("folder")}
							disabled={Boolean(loading)}
						>
							Open SpatialData folder <span>compatibility</span>
						</button>
						<button
							className="file-button"
							onClick={() => chooseFiles("zip")}
							disabled={Boolean(loading)}
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
						{error && (
							<button className="text-button" onClick={close}>
								Clear error
							</button>
						)}
						{error && (
							<p className="error" role="alert">
								{error}
							</p>
						)}
					</Section>
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
							Images and annotations use the transforms stored in your dataset.
						</p>
					</Section>
				</div>

				{error && tab !== "Setup" && (
					<p className="error" role="alert">
						{error}
					</p>
				)}
				{tab === "Layers" && (
					<>
						<label className="selected-view-control">
							Editing view
							<select
								aria-label="Editing view"
								value={activeId}
								onChange={(e) => onActivate(e.target.value)}
							>
								{session.panes.map((p) => (
									<option value={p.id} key={p.id}>
										{p.name}
									</option>
								))}
							</select>
						</label>
						<LayerManager
							pane={pane}
							dataset={dataset}
							onChange={(next) => changePane(activeId, next)}
						/>

						{selected && (
							<Section
								title={`${layerLabel(selected.source.elementType)} settings`}
								className={`layer-settings layer-${selected.source.elementType}`}
							>
								<div className="selected-title">
									<span className="layer-type">
										<Icon name={selected.source.elementType} />
										{layerLabel(selected.source.elementType)}
									</span>
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
								{binding && (
									<CellControls
										dataset={dataset}
										entry={selected}
										binding={binding}
										result={overlays[`${activeId}/${selected.id}`]}
										geneLists={geneLists}
										propagation={propagationControls}
										onChange={(next) =>
											setCellBindings((previous) => ({
												...previous,
												[selected.id]: next,
											}))
										}
									/>
								)}
								{["labels", "shapes"].includes(selected.source.elementType) && (
									<>
										{selected.source.elementType === "labels" && (
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
										)}
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
										{selected.source.elementType === "labels" && (
											<p className="muted">
												{associations.length
													? `Linked table: ${associations[0][0]}`
													: "No associated annotation table."}
											</p>
										)}
									</>
								)}
								{selected.source.elementType === "points" && (
									<>
										<TranscriptControls
											element={dataset.points[selected.source.elementKey]}
											entry={selected}
											geneLists={geneLists}
											propagation={propagationControls}
											onChange={(props) => updateEntry(selected.id, { props })}
										/>
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
												checked={selected.props.colorByFeature !== false}
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
										{selection.kind !== "shapes" &&
											selection.rowIndex !== undefined && (
												<>
													<dt>Table row</dt>
													<dd>{selection.rowIndex}</dd>
												</>
											)}
									</dl>
									{selection.kind !== "shapes" && (
										<TooltipContent tooltip={selection.tooltip} />
									)}
									{selectedOverlay?.binding && (
										<div className="feature-data">
											<dl>
												<dt>Linked table</dt>
												<dd>{selectedOverlay.binding.table}</dd>
												<dt>Matched row</dt>
												<dd>{selectedRow ? selectedRow.row : "Unpaired"}</dd>
												{selectedOverlay.binding.field && (
													<>
														<dt>
															{selectedOverlay.label} ·{" "}
															{["gene", "sum"].includes(
																selectedOverlay.binding.source,
															)
																? "X"
																: "obs"}
														</dt>
														<dd>{pretty(selectedRow?.value)}</dd>
													</>
												)}
											</dl>
										</div>
									)}
									{selection.embeddingTable && (
										<dl>
											<dt>Embedding table</dt>
											<dd>{selection.embeddingTable}</dd>
											<dt>Embedding row</dt>
											<dd>{selection.rowIndex}</dd>
											{selection.embeddingLabel && (
												<>
													<dt>{selection.embeddingLabel}</dt>
													<dd>{pretty(selection.embeddingValue)}</dd>
												</>
											)}
										</dl>
									)}
									<button
										onClick={() => {
											setSelection(null);
											setEmbeddingSelection([]);
										}}
									>
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
			<section className={`workspace ${dataset ? "" : "welcome-only"}`}>
				<header className="workspace-header">
					<div>
						<h1>{dataset ? source : "MERSCOPE Explorer"}</h1>
						<p>
							{dataset
								? `${coordinateSystem} · ${enabled} visible ${enabled === 1 ? "layer" : "layers"}`
								: "Images, cells and annotations. Together, on your computer."}
						</p>
					</div>
					{dataset && session.panes.length > 1 && (
						<label className="navigation-mode">
							Navigation
							<select
								aria-label="View navigation"
								value={session.linked ? "linked" : "independent"}
								title="Linking aligns all views to the selected pane"
								onChange={(e) =>
									setSession((s) =>
										setNavigationLinked(s, e.target.value === "linked"),
									)
								}
							>
								<option value="independent">Independent views</option>
								<option value="linked">Linked pan and zoom</option>
							</select>
						</label>
					)}
					<div className="header-badge">
						<span className="privacy-dot" />
						Browser local
					</div>
				</header>
				<div
					className="viewport"
					ref={viewport}
					data-testid="viewport"
					aria-busy={Boolean(loading || readActivity?.pending)}
				>
					{dataset ? (
						<div
							className={`pane-grid panes-${session.panes.length}`}
							style={
								figureLayoutState
									? {
											width: figureLayoutState.width,
											height: figureLayoutState.height,
											gap: figureLayoutState.gap,
											gridTemplateColumns: `repeat(${figureLayoutState.columns}, ${figureLayoutState.paneWidth}px)`,
											gridTemplateRows: `repeat(${figureLayoutState.rows}, ${figureLayoutState.paneHeight}px)`,
										}
									: undefined
							}
						>
							{session.panes.map((p) => (
								<ViewerBoundary
									key={source + ":" + request.current + ":" + p.id}
								>
									<ViewerPane
										figure={session.figure}
										figureReference={session.figureReturn?.[p.id]}
										onGroups={onGroups}
										selectionMode={selectionMode}
										onSelectCells={onSelectCells}
										onSelectionScope={setSelectionScope}
										onSelectionApi={tissueSelectionApi}
										highlighted={highlighted}
										pane={p}
										dataset={dataset}
										system={coordinateSystem}
										active={p.id === activeId}
										linked={session.linked}
										overlays={overlays}
										geneLists={geneLists}
										onActivate={onActivate}
										onCamera={onCamera}
										onPick={onFeatureClick}
										onStatus={entryStatus}
										channelTarget={channelTarget}
										onChannels={changeChannels}
									/>
								</ViewerBoundary>
							))}
						</div>
					) : (
						<WelcomeScreen
							recent={recent}
							loading={Boolean(loading)}
							onOpenFolder={() =>
								typeof window.showDirectoryPicker === "function"
									? chooseDirectory()
									: chooseFiles("folder")
							}
							onOpenZip={() => chooseFiles("zip")}
						/>
					)}
					<LoadingIndicator
						loading={loading}
						activity={readActivity}
						onCancel={cancelLoading}
					/>
					{gpuError && (
						<div className="canvas-error" role="alert">
							Rendering error: {gpuError}
						</div>
					)}
				</div>
				<ViewerToolbar
					dataset={dataset}
					session={session}
					mode={selectionMode}
					onMode={setSelectionMode}
					onSystem={(system) => resetView(dataset, system)}
					onZoom={zoom}
					onFit={() => setViewState(null)}
					onFullscreen={() => viewport.current?.requestFullscreen?.()}
					onBookmark={() => bookmarkCurrent()}
					onFigure={() =>
						setSession((s) => ({
							...s,
							drawer: {
								...s.drawer,
								open: true,
								tab: "Figure",
								height: Math.max(400, s.drawer.height),
							},
						}))
					}
					onSidebar={() =>
						setSession((s) => ({
							...s,
							sidebar: { ...s.sidebar, hidden: !s.sidebar?.hidden },
						}))
					}
					onPane={changePane}
					onFigureChange={changeFigure}
					onClear={() => {
						setSelection(null);
						setEmbeddingSelection([]);
					}}
					onSelectDisplayed={() => {
						const embeddingActive =
							selectionScope === "embedding" &&
							session.drawer.open &&
							session.drawer.tab === "Embeddings";
						(embeddingActive
							? selectionApis.current.embedding
							: selectionApis.current.tissue
						)?.select();
					}}
					selectionCount={embeddingSelection.length}
					selectionScope={
						selectionScope === "embedding" &&
						session.drawer.open &&
						session.drawer.tab === "Embeddings"
							? "embedding"
							: "tissue"
					}
				/>
				<WorkspaceDrawer
					paneGroups={paneGroups}
					figureProps={{
						onChange: changeFigure,
						onPane: changePane,
						onMode: figureMode,
						onExport: exportFigure,
						onCancel: () => figureAbort.current?.abort(),
						busy: figureBusy,
						progress: figureStatus,
						rect: viewportSize,
					}}
					selectionMode={selectionMode}
					onSelectionMode={setSelectionMode}
					onSelectionScope={setSelectionScope}
					onSelectionApi={embeddingSelectionApi}
					activeBinding={activeBinding}
					activeRegion={activeRegion}
					activeOverlay={activeOverlay}
					highlighted={highlighted}
					onEmbeddingSelect={onEmbeddingSelect}
					onEmbeddingFocus={onEmbeddingFocus}
					key={localKey}
					session={session}
					setSession={setSession}
					dataset={dataset}
					onExport={exportView}
					onImport={importView}
					saveStatus={saveStatus}
					onSave={flushSave}
				/>
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
			{figureJob && <FigureRenderer key={figureJob.id} job={figureJob} />}
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
