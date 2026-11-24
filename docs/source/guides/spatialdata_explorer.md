# MERSCOPE Explorer

MERSCOPE Explorer opens local SpatialData Zarr folders and ZIP archives
entirely in a desktop browser. It combines multichannel images, segmentation,
points, shapes and associated cell annotations in a 2D view. Display settings
can be saved as JSON and composed figures exported as PNG, JPEG or layered SVG.

The sidebar has **Setup**, **Layers** and **Inspect** tabs. Setup contains file
opening and coordinate-system selection. In Layers, use **Add layers** to build
an active list and hide the picker when finished. Active layers can be hidden,
reordered or removed independently; adding one again restores its settings.
Image, Labels, Shapes and Points use subtle type accents in both list and controls.

Select a shapes layer to link its cell boundaries to an AnnData table. Choose
gene expression from `X` (dense, CSR or CSC) or categorical/numeric annotations
from `obs`. The viewer matches exact cell IDs within the selected region and
reports the number of matched boundaries. It leaves mismatches unpaired.
Each segmentation layer has its own table, field, scale and legend. Transcript
points can be filtered by gene. These settings are retained in saved views.

Use **Apply across views** in the cell or Points controls to copy genes and gene
lists to visible cell/Points layers across the open panes. Cell-to-cell copies
also include colour settings and can propagate an `obs` field. Each layer keeps
its own table and ID pairing. Missing obs fields or entirely absent genes leave
that target unchanged; partial gene-list matches are applied with explicit
missing-gene warnings. Multiple selected Points genes become a reusable workspace
list for summed cell expression. Hidden layers are left unchanged.

The hideable, resizable bottom drawer holds **Views**, **Bookmarks**, **Gene
lists**, **Embeddings**, **Figure** and **Workspace**. Create up to four panes with independent display
settings and linked pan/zoom, or unlink their navigation. Bookmarks save the
complete viewer with editable names and notes. All panes share the selected
coordinate system; bookmarks can return to an earlier ROI and layout.

Views has per-pane legends, scale bars and original SpatialData layer names
(bottom-left by default). Customise overlay positions, sizes, legend titles and
individual labels or omissions; colours and ranges follow the displayed layers.
The toolbar provides Inspect/pan, click selection, lasso, visibility toggles,
zoom/fit/fullscreen, coordinate navigation, bookmarks and export. Tissue lasso
and Select displayed use transformed shape centres; label masks support click
selection. Selection is shared with embeddings and remains temporary.

Figure mode composes panes at a chosen aspect ratio and export resolution, with
custom colours, overlay text sizes, per-pane scale bars and optional automatic
bookmarks. Returning to Viewer restores the original cameras. Scale bars show
coordinate units until **µm per coordinate unit** is confirmed for the transformed
coordinate system. Calibration enables a locked physical field of view with pan
but no zoom. PNG/JPEG preserve the composed appearance; SVG embeds separate
SpatialData raster layers with editable vector labels, legends and scale bars.
Filename options include region/view names, timestamp and constant table metadata.

Create named gene lists or import CSV with one gene per row, optionally using
`list,gene` columns for several lists. Select a list to display all its transcript
points, or colour shapes using the sum of those genes' stored AnnData `X` values.
Gene names match exactly and duplicates count once. Missing genes are reported;
log1p, when selected, transforms the sum for display only.

The Gene lists drawer checks draft genes against the selected table, or any table
you choose, and lists present, missing and ambiguous names. Presence means the
gene is available in `X`, not that its expression is nonzero.

Continuous cell overlays have a draggable upper colour limit, numeric vmin/vmax,
palette selection and reversal. Limits are expressed before log1p, and changes
recolour cached data. The **Embeddings** drawer reads existing numeric AnnData
`obsm` projections, including UMAP, t-SNE, PCA and spatial coordinates. Choose
dimensions, follow the tissue pane's colours or use an independent gene, gene
list or annotation, inspect points, lasso cells and centre matched shapes in the
tissue view. Embedding options persist in workspaces and bookmarks; selection
is temporary. Other regions in the same table can appear as faint context.

Workspace settings autosave locally in the browser. After reloading, reopen the
same folder or ZIP to restore views, bookmarks and gene lists. File paths, sizes
and modification times distinguish local saves. Export/import a workspace JSON
for a portable backup or a changed copy of a store; no dataset contents are
included. Local saves belong to the browser profile and site origin, and are
removed if its site storage is cleared.

The app uses SBT's IMC Explorer styling. Its reusable loading and rendering
engine comes from the pinned SBT fork of SpatialData.js. Files remain on the
user's computer; hosting requires only static HTML, JavaScript, CSS and WASM.

Build and run from `spatialdata_viewer/` with Node 24.14.1 and pnpm 11.20.0:

```shell
npm run prepare:engine
pnpm --dir .engine/SpatialData.js install --frozen-lockfile
npm run build
npm run preview
```

Open `http://127.0.0.1:8879` in Chrome or Edge, choose the `.zarr` folder itself
or its ZIP, then select a coordinate system and layers. The preview process
serves application assets only. When hosted on HTTPS, users need just a browser.

This version does not compute new embeddings, support 3D navigation or edit data.
It reads dense numeric and dataframe projections; sparse `obsm` is unsupported.
Large datasets remain subject to browser and GPU memory limits. Embedding plots
cap the current-region preview at 200,000 cells and report any sampling.
Automated browser tests cover synthetic Zarr v2 stores with shuffled cell IDs,
multiple tables and gapped Parquet partitions. An opt-in test covers the supplied
Abed MERSCOPE store, including its dense and sparse expression tables. Other
formats/codecs and larger datasets need further validation.

See the [application README](https://github.com/dr-michael-haley/imcanalysis/tree/main/spatialdata_viewer)
for development, deployment, checks and limitations.
