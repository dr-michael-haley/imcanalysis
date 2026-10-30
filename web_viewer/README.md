# SBT IMC Explorer

A standalone, read-only browser application for local AnnData, segmentation
masks and IMC channel TIFFs. All scientific file reads happen on the user's
computer. The production application has no API, upload endpoint, telemetry,
external fonts, or CDN dependency.

See the [IMC Explorer user guide](https://imcanalysis.readthedocs.io/en/latest/browser_tools/imc_explorer.html)
and the [technical and format reference](../docs/source/guides/browser_viewer.md).

## Development

Requires Node.js 22.12+ and pnpm. These are developer requirements, not viewer
requirements when the built application is hosted.

```shell
cd web_viewer
pnpm install --frozen-lockfile
pnpm dev
```

## Build and run

```shell
pnpm build
cd ..
sbt gui web
```

The build goes to `SpatialBiologyToolkit/browser_viewer/static/`. The optional
`sbt gui web` launcher serves only those application files on `127.0.0.1` and
opens the browser. It never opens AnnData or images in Python. `--check` checks
for a build without launching; `--no-browser` prints the URL; `--port` selects a
port. Stop the launcher with Ctrl+C.

The same static directory can be hosted unchanged at an HTTPS root or subpath.
Users of a hosted copy need only their browser and local files. Serve the build
over HTTP/HTTPS: double-clicking `index.html` with a `file://` URL is not
supported. There is no service worker or promised offline cache in this version.

Build before creating a Python wheel if distributing the optional launcher with
bundled assets. Generated assets and node_modules are ignored by Git; the pnpm
lockfile fixes the application dependency versions. Do not serve the repository
or a dataset directory as the web root.

The build includes `THIRD_PARTY_LICENSES.md` with the runtime dependencies'
license notices, including libraries bundled into workers.

## Cloudflare hosting

The production target is `https://imc.spatialbiology.app/`, using the
`sbt-imc-explorer` Cloudflare Worker with static assets only. The checked-in
`wrangler.jsonc` specifies the account, build directory and custom domain.
`public/_headers` preserves the local launcher's security policy on Cloudflare,
including WebAssembly and local file workers. No datasets, server code, database,
telemetry or upload endpoint are deployed.

To publish an update from this directory, build and verify the viewer, authenticate
Wrangler for the configured account, inspect `wrangler whoami`, then run
`wrangler deploy`. Alternatively, upload the **contents** of the built static
directory through the existing Worker's dashboard. Include `_headers` and the
license notices; never upload the repository or a dataset folder.

The previous address, `https://explorer.spatialbiology.app/`, is maintained by
the redirect-only `sbt-tissue-explorer` Worker. Its separate configuration is
`wrangler.legacy.jsonc`; deploy it with `wrangler deploy --config wrangler.legacy.jsonc`
only after the IMC domain is verified. It returns a permanent 308 redirect,
preserving the path and query. Browser navigation retains the original fragment.
It has no static assets or dataset handling.

Users select their local files again on the new origin. Browser-only workspaces
and folder permissions from previous domains are not migrated. Previously exported
workspace JSON remains portable: import it and reconnect the local data folder.
The internal JSON format and local database identifiers retain their historical
names for compatibility. The local `sbt gui web` command and source paths are unchanged.

For deployment versions and rollback instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Verification

Generate tiny fixtures in the ignored `web_viewer/fixtures/` folder using an
existing scientific environment with anndata, h5py, numpy, scipy, pandas and
tifffile. No production files are read or changed.

```shell
python tests/make_fixtures.py
python tests/make_plot_fixtures.py
python tests/make_review_fixtures.py
python tests/make_aligned_fixtures.py
pnpm test
pnpm build
```

With `sbt gui web --no-browser` running in another terminal:

```shell
pnpm test:browser
```

Browser tests use installed Google Chrome by default; set
`SBT_BROWSER_CHANNEL=msedge` for Edge, and `SBT_VIEWER_URL` for a different local
URL. They exercise real local file pickers, workers, TIFFs, HDF5, sparse matrices,
ROI changes, linked cell selection, workspace round trips and errors, record screenshots in the ignored
`test-results/`, check that requests are GETs to the application's own origin,
and verify input-file hashes are unchanged.

An optional scale check creates a synthetic five-million-cell H5AD across 1,000
ROIs (requires additional disk space and RAM) and exercises indexing, marker
reads, the last ROI and linked cell metadata in the browser:

```shell
python tests/make_scale_fixture.py
node --test tests/browser-scale.test.js
```

The same browser channel and viewer URL settings apply. This synthetic check
does not replace performance validation on representative real datasets.

From the repository root, test the lightweight CLI and static-file boundary:

```shell
python -m pytest tests/test_browser_viewer.py tests/test_cli_lightweight.py
```

## Design and scope

- `src/anndata.js`: tested AnnData interpretation, identity checks and bounded
  matrix-column access. Dense, CSR and CSC use the same cell-row ordering.
- `src/data.worker.js`: read-only HDF5 via h5wasm/WORKERFS; the selected local
  file is not first copied wholesale into WebAssembly memory.
- `src/core.js`: SBT folder/marker conventions, mask coverage and colours.
- `src/raster.worker.js`: local TIFF decoding through GeoTIFF.js. Decoding runs
  outside the UI thread; only the selected planes are retained.
- `src/normalization.js`: Nimbus CSV / NapariSBT JSON validation, collision-aware
  channel matching and image scaling, checked against the Python NapariSBT
  helper. Dictionary and fallback scaling run in the TIFF worker.
- `src/workspace.js`: versioned JSON view settings, validation and compact local
  file references. Resolves AnnData and TIFF folders beneath one selected root;
  versions 1–4 migrate to version 5 with population proposals, review tabs, plot
  settings, flag labels, ROI filters and representative marker preferences.
- `src/remaps.js`: exact source identities, proposal validation, merges, shared
  colours, CSV export and per-ROI label previews without mutating AnnData.
- `src/remap-editor.js`: paginated editable population tables, notes, undo and
  multiple named schemas. Worker summaries count populations in bounded slices.
- `src/review-panel.js`: accessible bottom tabs and adjustable height.
- `src/flags.js`, `src/flag-panel.js`: categorical cell labels keyed by ROI and
  mask object ID, click/lasso selection, undo, CSV export and overlay preview.
- `src/review-data.js`: cancellable worker calculations for descriptive marker
  enrichment, finite-value histograms and observations constant within ROIs.
- `src/representative-panel.js`, `src/roi-metadata-panel.js`: marker shortcuts,
  sample metadata summaries and filters applied to the current ROI ranking.
- `src/colourmaps.js`: continuous marker palettes shared with the legend.
- `src/plots.js`, `src/plot-reader.js`: validated plot settings and exact batched
  aggregates, including sparse zeros, missing values and proposed remaps.
- `src/plot-panel.js`, `src/plot-svg.js`: four plot types, filters, snapshot/stale
  handling, worker cancellation, expansion and SVG/PNG/data CSV exports.
- `src/views.js`: independent pane composition and stable review recipe hashes.
  ROI rendering yields between pixel batches so the loading animation and ROI
  navigation remain responsive with multiple views.
- `src/loading.js`: loading progress for AnnData, workspace restoration and ROI
  changes. The overlay names the current stage and view; superseded requests
  cannot dismiss a newer request's indicator. Old tissue is dimmed until ready.
- `src/roi-browser.js`: ROI ordering controls, top-X shortcuts and recipe-specific
  visit colours. AnnData aggregation runs in `src/anndata.js` in the worker.
- `src/main.js`: UI, stale-result guards, bounded raster cache, linked embedding
  and tissue picking. Canvas composition is cached for pan/zoom; this first
  version recomposes full ROI planes when contrast or overlays change.
- `SpatialBiologyToolkit/browser_viewer/`: standard-library static launcher.

The viewer imports the official `images/Logo.svg` directly;
Vite includes the asset in the standalone build without an external image URL.
The browser-tab icon in `src/favicon.svg` uses the same artwork on a dark,
rounded square for contrast in light browser themes. `images/Logo.png` is the
PNG fallback. Both are bundled with content hashes for cache-safe updates.

This is an interactive review helper, not a pipeline analysis stage. It creates
no scientific assets, managed executions or reports. It has no persistent
analysis configuration, no SLURM integration and no Conda requirements. Review
settings, remaps and flags autosave to named local browser workspaces in
IndexedDB. Read-only root-directory handles can be remembered on supporting
browsers; exported JSON contains no handles or scientific file contents. Local
storage remains specific to the origin/profile and is not a cloud backup. Source
AnnData, images, masks, existing NapariSBT workflows and project state are
unchanged. Scientific equivalence and performance on representative real SBT
datasets remain release-acceptance work.

`src/workspace-library.js` implements transactional workspace records, compact
recent-workspace summaries, revision checks against stale writes, and lazy
directory traversal. `src/workspace-panel.js` manages the Setup library and
autosave lifecycle. `package.json` supplies the version displayed by Vite builds;
the Guidejar link is configured in `index.html`.

For real-browser persistence checks, copy `tests/persistence-smoke.html` as
`index.html`, `tests/persistence-smoke.js`, and `src/workspace-library.js` into
an isolated temporary static web root. Serve that folder on localhost and press
**Run persistence checks**. The checks use disposable IndexedDB/OPFS fixtures,
cover transactions, stale-write protection, directory-handle cloning and deletion,
and remove their test data afterwards. Do not include this harness in deployments.


### Viewer navigation and linked selection (0.3.0)

Workspace schema 9 retains bookmark snapshots, a separate library of single-pane
view presets with unique F-key assignments, and independent embedding controls.
`navigation.js` validates these states; `navigation-panel.js` provides the bookmark
table and preset library. Existing `view.presets` still describes active panes;
`view.savedViews` is the reusable library. Bookmarks contain display snapshots and
references to workspace labels/layers, never copies of cell data or source files.

`selection.js` stores selections by ROI and mask ID. The quick toolbar's click and
lasso tools work across tissue and embedding, independent of flag fields. Selection
is session-only; assigned flags persist. `embedding-reader.js` performs sequential,
cancellable context scans and retains a bounded, deterministic stratified sample.
The current ROI remains exact. `embedding-panel.js` caches context across ROIs and
uses a cached canvas while drawing the lasso to avoid redrawing every point on each
pointer move. `alluvial.js` conserves counts from the existing categorical
contingency-table worker and exports self-contained SVG via the plotting panel.

`tests/navigation.test.js` covers schema migration, bookmark cameras and comments,
preset validation/hotkeys, cross-ROI identity and flag undo, stratified sampling,
real dense/CSR/CSC H5AD reads, and alluvial flow conservation/escaping. Browser QA
also checks bookmark return, selected-pane hotkeys, linked selection, exports and
workspace reloads with synthetic files. Production-scale network performance is
not established by these synthetic tests.

### Grouped viewer footer (0.3.1)

`viewer-toolbar.js` owns the mouse-mode and View disclosures, keyboard navigation,
focus return, outside-click dismissal, and contextual selection feedback. The
footer keeps ROI navigation and Bookmark visible; View contains pane-specific
visibility checkboxes and linked zoom/Fit. Selection actions remain in
`flag-panel.js` and work without a flag field or an open review panel. Done/Escape
returns to Inspect without discarding selected cells; changing mode cancels any
unfinished tissue/embedding gesture. Select displayed follows the last used
space, falling back to tissue when embedding is hidden. This scope and the number
of ROIs with selected cells are shown beside the selection actions.

Menu state and pointer mode are transient; workspace schema 9 is unchanged.

### Tissue figures and asset locations (0.4.0)

Workspace schema 10 adds figure settings/calibration, per-pane scale bars,
multi-population filters and annotation outline widths. Older versions migrate
without changing the original files. `figure.js` owns validated settings,
physical geometry and shared SVG furniture; `figure-panel.js` manages preview
layout and camera restoration; `figure-export.js` exports PNG/JPEG or separate
SVG raster layers with vector labels and scale bars. Exports exclude transient
cell selections and can create bookmarks automatically. Output is capped at
32 million pixels, 16,384 pixels per side and 128 MB for SVG.

`folder-picker.js` retains read-only handles for manually chosen assets.
`workspace-paths.js` resolves them relative to the connected root using
`FileSystemDirectoryHandle.resolve`, without crawling ancestor directories.
Older basename-only paths can be repaired in Asset locations or with Locate
folder. Compatibility pickers remain available with their enumeration caveat.

`tests/figure.test.js` covers calibration/physical extents, multi-population
identity, inner annotation outlines, transparent export composition, nested
handle resolution without enumeration, SVG escaping and schema migration.
Browser QA uses synthetic data to check numeric controls, figure themes,
physical locks, exports/bookmarks and workspace reloads.

### Figure scale bars and filenames (0.4.1)

Workspace schema 11 adds independent figure scale text size and validated
filename preferences. Existing pane scale-bar visibility, physical length and
line width now have controls directly in Figure, including all/none shortcuts.
The common SVG furniture drives previews and every export format.
`figure-filename.js` composes a bounded filesystem-safe basename from selected
ROI/view names, a label, explicit ROI metadata and an optional UTC timestamp.
ROI metadata is shared with the intentional ROIs scan; selected filename fields
are preserved on reload but block export until read or deselected. Bookmark
snapshots retain naming choices, but no metadata values are copied into settings.

### Continuous colour limits (0.4.2)

Cell overlay, Embedding and continuous Plots have independent **Lower (vmin)**
and **Upper (vmax)** inputs. Blank bounds use automatic scaling; **Auto** clears
both. These controls clip colours without filtering cells or changing expression
values. Marker overlays and embeddings redraw cached values. Plot snapshots use
the chosen bounds when regenerated, including SVG/PNG legends and CSV settings.
For z-scored/normalised plots, bounds are in the displayed units.

Embedding limits can override the selected view without changing that pane.
Workspace schema 12 saves limits in panes, bookmarks, presets, embedding and plot
settings; older workspaces migrate to automatic bounds. Categorical colours are
unaffected. The header also has a larger guide link and an email issue link.

### Getting started screen (0.4.3)

The welcome screen introduces the viewer's review, plotting and export features,
explains required and optional assets, and provides a matching folder example.
It links to SBT on GitHub, the documentation, the Guidejar walkthrough and
Michael Haley's email. Open Setup and Resume a workspace take users to the
existing sidebar controls. Before AnnData is opened, unused viewer/review tools
are hidden to give the scrollable welcome screen more room. Once AnnData is
available they return, including for an AnnData-only plotting session.

### Asset reconnection (0.5.0)

`workspace-assets.js` maintains independent asset states and retains successful
checks when another asset is missing. `workspace-connection.js` renders one
compact checklist, individual file/folder location and retry controls, nested
relative paths, and sequential progress. Annotation dictionaries can be repaired
without rereading registered folders. The separate local-storage disclosure and
duplicate workspace loading overlay are removed.

Full checks remain the default on every open/switch. Opt-in Quick reconnect uses
metadata-only file inventories in a separate IndexedDB settings entry per
workspace. Missing caches fall back to full checks; asset retries force a full
check. `asset-files.js` defers image/mask getFile calls until decoding in
`raster.js`; AnnData and dictionaries are still opened immediately. Cached file
sizes are checked on use, while full checks validate the whole inventory. Caches
are excluded from exported JSON, deleted with the local workspace, and optional
if browser storage is unavailable. Workspace JSON schema remains 12.

Unit coverage includes partial recovery, exact nested paths, metadata-call
counts for quick reconnection, lazy file failures, incompatible dictionaries,
compatibility pickers and cancellation. Browser QA uses synthetic assets only.

Patch 0.5.1 constrains the expanded asset checklist to the sidebar width. Long
paths use an ellipsis (the full path remains in the tooltip and Edit path field),
and long layer names and file-picker labels wrap. Browser checks cover registered
and missing assets, expanded path editors, and the narrower sidebar breakpoint.

### Legend entries and flattened labels (0.6.0)

Each channel, cell overlay and aligned layer has a per-pane `showInLegend` flag.
Viewer and figure legends share `legendGroups`; toggling entries redraws only
the legends and preserves ROI visit signatures. Categorical layers support
`flatten`, `flatLabel` and `flatColour`. All nonzero IDs are treated as one region
at render time, without modifying or duplicating the source label raster.
Separate weak caches retain normal and flattened outline distances; holes and
background stay transparent. Setup can initialise new categorical layers with
flattening, while Views allows independent settings in each pane.

Workspace schema 13 retains these settings in panes, bookmarks and presets,
and migrates versions 1–12 with visible legend entries and flattening disabled.
Tests cover pixel preservation, uint32/unmapped IDs, background, union outlines,
exported legend filtering, visit signatures and saved-state round trips.
