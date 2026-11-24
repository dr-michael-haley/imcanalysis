# MERSCOPE Explorer

A read-only, browser-local transcriptomics viewer with the dark sidebar and mint
controls used by SBT Tissue Explorer and MCD Explorer. Open a local SpatialData
`.zarr` directory or ZIP, choose a coordinate system, and combine images,
cell boundaries and transcript points. Colour boundaries by a gene in AnnData
`X` or a categorical/numeric `obs` column, compare segmentation results,
inspect cells, save views and export PNG, JPEG or layered SVG figures. The source directory remains
`spatialdata_viewer/` for compatibility with existing build scripts.

## Explore cells and transcripts

In Chrome or Edge, use **Open local / network folder** for large stores and mapped
drives. This requests read-only directory access, uses consolidated Zarr metadata
when available, and opens chunks on demand without recursively enumerating their
folders. Groups and arrays omitted from an older consolidated index are discovered
from their individual metadata files, so images added later remain available.
Parquet partitions are discovered separately. **Open SpatialData folder
(compatibility)** retains the original file-list picker for other browsers.

Direct-folder workspaces resume by directory identity when you choose the same
folder again. They are separate from the compatibility picker's existing local
saves. To transfer those settings, export the workspace from the original view
and import it after opening the folder directly. Directory handles stay in local
browser storage and are excluded from exported workspace JSON.

Opening a dataset shows its loading stage, elapsed time and file-read activity.
Compatibility folder selection includes a waiting indicator while the browser enumerates files
(particularly useful on mapped network drives). The browser supplies no file count
until enumeration finishes, so progress is indeterminate. **Cancel loading** stops
the viewer opening the store; already-issued operating-system reads may finish in
the background. Subsequent image, transcript and table reads show a compact activity
indicator without blocking navigation. Counts include successful reads, not unique
files or a percentage of the entire store; data is still read on demand.

**Report an issue** beneath the application name opens an email draft addressed to
the same support contact as Tissue Explorer, with the viewer version in its subject.
It does not attach dataset contents or send mail automatically.

1. In **Setup**, open the SpatialData folder (or its single enclosing folder)
   or a ZIP, then choose its coordinate system.
2. In **Layers**, use **Add layers** to add a shapes layer to the active list.
   Select its name to open **Shapes settings**. The picker can be hidden while
   working; search and type filters help find layers in large stores.
3. Check **Linked cell table**. The default comes from SpatialData association
   metadata; select another table explicitly when needed. **Cell linking** lets
   you choose its cell ID column. The table's region column scopes the join.
4. Select **Gene expression · X** and search for a gene, or **Cell annotation ·
   obs** and choose an annotation. The legend shows the scale or category colours.
   Linear values are read as stored; log1p is a display transform only.
5. Add/select the points layer to show transcripts. Use **Transcript genes**
   to show all genes or select a subset. Click a cell to inspect its matched row
   and current overlay value.

Joins use exact string cell IDs within the selected region, preserving large
integer identifiers. Unmatched boundaries remain visible and uncoloured by the
table. Duplicate IDs are rejected. Equal row counts never imply a valid match.
The UI reports how many boundaries match, and missing values are grey. Each
shapes layer has independent table, field and scale settings, saved in view JSON.

Active layers have separate visibility, drawing-order and removal controls.
Removing a layer from the working list retains its display settings for adding
it again. Image, Labels, Shapes and Points have matching type accents in their
list entries and settings. **Inspect** contains picked-cell details and the
dataset inventory. The toolbar can hide the sidebar to give the panes more room.

**Apply across views** in the cell and Points controls copies choices to visible
layers in all open panes, including other visible layers in the current pane.
Each destination keeps its own table, cell ID column, geometry and visibility.
Cell-to-cell copies include the field, expression scale, colour map and limits.
Genes and workspace gene lists can be copied between cell and Points layers;
obs fields apply only to cells. Missing obs fields, absent genes, ambiguous gene
names and unreadable targets are skipped with a per-layer explanation. Partial
gene-list matches are applied with a warning listing the missing genes.

Multiple individually selected Points genes become a workspace list when copied
to cells, which colour by their sum. An equivalent existing list is reused.
Points-to-cell copies retain each cell layer's palette and display transform,
and reset colour limits to auto. **All genes** can be copied between Points
layers; choose selected genes or a list before applying to cells. Hidden layers
are untouched, and copied settings are included in local saves and workspaces.

## Views, bookmarks and gene lists

The bottom **Workspace tools** drawer can be hidden or resized by dragging its
upper edge (or using arrow keys on the resize handle).

- **Views:** duplicate the selected pane, up to four panes. Click a pane header
  to edit its layers, channels and cell overlays in the sidebar. Pan and zoom
  are linked by default. Choose **Add independent view** to compare separate regions, or use **Navigation: Independent views** above the panes. Relinking aligns all cameras to the selected pane. Bookmarks and workspaces preserve independent positions.
  All panes use the selected coordinate system. Changing it starts a new view
  layout; bookmarks retain earlier layouts for returning to another ROI.
  Each pane has legend, scale-bar and original SpatialData layer-name overlays.
  **Customise overlays** edits the selected pane's positions, text sizes, legend
  title, layer titles and individual channel/gene/category labels or omissions.
  Colours and ranges follow the actual displayed data. Original layer names
  default to the bottom-left corner; scale bars default to bottom-right.
- **Bookmarks:** capture the complete viewer, including all pane settings,
  cameras and gene-list definitions. Rename bookmarks, add notes and use
  **Return** to restore them. A return restores the saved versions of lists
  and retains unrelated lists added since the bookmark.
- **Embeddings:** choose a table and an existing AnnData `obsm` projection
  (UMAP, t-SNE, PCA or another numeric projection), then choose its two
  dimensions. Colours can follow the selected tissue view or independently use
  a gene, gene-list sum or `obs` field. Click points to inspect their table rows;
  lasso cells to highlight matching boundaries in the tissue panes. **Centre
  selected cell** locates the matching shape using its spatial transformation.
  The **Square** and **Other regions** options mirror Tissue Explorer; context
  points are faint and capped at the chosen sample limit. Export the plot as PNG.
  Selection is temporary; projection and colour settings are saved in the
  workspace and bookmarks. Tables without projections show an explicit empty
  state. `obsm/spatial` is labelled as spatial coordinates, not UMAP/t-SNE.
- **Gene lists:** create named lists manually or import CSV. One gene per row
  works with or without a `gene` header. To import several lists, use
  `list,gene` columns. Quoted fields and UTF-8 BOMs are accepted. Names match
  exactly, including case; duplicate genes within a list count once.
  **Check against table** compares the draft list with the selected cell table,
  or another table you choose, and lists present and missing genes. It also
  warns about missing/inconsistent `X` and duplicate gene names in `var`.
  A present gene can still have zero expression; the check is for availability.
- For transcript points, choose **Workspace gene list** under **Transcript
  genes**. For shapes, choose **Gene list · sum of X** under **Overlay data**.
  The cell overlay adds each matched gene's stored `X` value per cell, then
  applies the optional log1p display scale. Missing genes are listed; a list
  with no matching table genes shows an error instead of a zero-valued result.

The toolbar provides coordinate-system navigation, Inspect/pan, click selection,
lasso, display toggles, zoom/fit/fullscreen, bookmarks and figure export. Selection
is shared with embeddings. **Select displayed** and tissue lasso select the
transformed centres of visible Shapes layers; label masks support click selection.
Escape returns to Inspect. Selection is temporary and is not part of exports.

## Figures and calibration

Use the toolbar's **Export** button or the **Figure** drawer, then **Figure mode**.
Choose pane aspect ratio, export width and spacing; pan and zoom to compose.
Returning to **Viewer mode** restores the cameras captured on entry. Set the
background and foreground, view-name/legend/scale text sizes, legend row limits,
overlay backgrounds and individual pane scale bars. View-specific legend labels
and positions are shared with the **Views** drawer.

SpatialData transformations do not always declare the physical unit of the
target coordinate system. Scale bars therefore show **coordinate units** until
you confirm **µm per coordinate unit**. This means the unit *after transforms*,
not the native image-pixel size. After calibration, **Lock field of view in µm**
sets each pane's physical width and height and disables zoom while preserving pan.
Changing coordinate systems resets calibration; bookmarks restore it.

PNG/JPEG render the complete composition at the requested resolution. SVG embeds
each SpatialData layer as a separate raster group and retains editable vector
legends, scale bars and labels. It does not turn cell boundaries or transcripts
into vector geometry. Exports can automatically create bookmarks. File names
can include a prefix, coordinate-system and view names, UTC timestamp, and
selected table `obs` fields constant across the plotted regions. Read metadata
again after reopening the Figure drawer to use saved metadata filename fields.
Exports are cancellable and capped at 32 million pixels, 16,384 pixels per side,
and 128 MiB of intermediate SVG. All rendering and encoding happen locally.

## Repository boundary

`imcanalysis/spatialdata_viewer` owns branding, layout, file selection, controls,
view files, build configuration and deployment. The
[SpatialData.js fork](https://github.com/dr-michael-haley/SpatialData.js) owns
reusable loaders, table joins, rendering and the embedding API. Both execute in
the browser. There is no backend service to deploy.

`engine.lock.json` pins the fork by commit. Release builds use its frozen pnpm
lockfile; they do not consume a moving branch or require an npm release of the
fork. The generated `build-info.json` records the engine revision and whether
local development changes were present. Once the engine API stabilises, versioned
package releases can replace the source checkout without moving the app.

## Build and run

Use Node 24.14.1 and pnpm 11.20.0 (the fork's packageManager version). From this
directory:

```shell
npm run prepare:engine
pnpm --dir .engine/SpatialData.js install --frozen-lockfile
npm test
npm run build
npm run preview
```

Open **http://127.0.0.1:8879** in desktop Chrome or Edge. The preview server
serves only `dist/` and accepts GET/HEAD. User datasets are opened through the
browser file picker, never through the server. No Python or npm installation is
needed on users' machines when the app is hosted on HTTPS. `file://` is not
supported because module workers and WASM need an HTTP/HTTPS origin.

To develop against the adjacent fork on this computer, use PowerShell:

```powershell
$env:SBT_SPATIALDATA_ROOT = 'D:/Github_repos/SpatialData.js'
pnpm --dir $env:SBT_SPATIALDATA_ROOT install --frozen-lockfile
npm run build
npm run preview
```

`npm run dev` rebuilds the engine packages once, then starts the app dev server.
After changing engine source, restart it to rebuild the engine. Clear
`SBT_SPATIALDATA_ROOT` before making a release build. The prepared checkout must
match the pin and have no tracked modifications; development overrides are
explicitly recorded in build provenance. Build scripts resolve the fork's
locked tools and public package entry points, so the app needs no second copy
of React or deck.gl.

## Deploy

Publish the **contents of `dist/`** to a static host. Keep workers and WASM at
their generated paths, serve `.wasm` as `application/wasm`, and preserve
`_headers` on Cloudflare. All runtime dependencies are bundled locally; there
are no CDN scripts, remote datasets or fonts. The same-origin connection policy
allows loading the app's own workers/WASM. The codec bindings need
`script-src 'unsafe-eval'` during initialization; removing it breaks Blosc
decoding. The app does not evaluate imported view files as code.

`wrangler.jsonc` is a separate static-assets deployment named
`sbt-spatialdata-explorer`. After authenticating to the intended Cloudflare
account, `wrangler deploy` from this directory publishes the built assets.
No account or custom domain is preselected, and this does not update the Tissue
Explorer or MCD Explorer sites. A dedicated subdomain such as
`spatialdata.spatialbiology.app` can be attached when the deployment is ready.

The manual **Build SpatialData Explorer** GitHub Actions workflow builds the
pinned engine and uploads a static-site artifact. It does not deploy a site.

## Validation

```shell
npm test
python tests/make_fixture.py
npm run test:browser
```

Fixture generation needs the scientific SBT environment, including spatialdata,
AnnData and GeoPandas. It writes a fresh synthetic store and archive under the
ignored `fixtures/` directory and records their names in `fixture.json`. The
fixture uses Zarr v2, two coordinate systems, multiscale Blosc-compressed images
and masks, Parquet shapes/points, and shuffled table rows with repeated IDs
across regions. A process-local adapter handles the installed Dask/OME-Zarr
writer API mismatch without changing the environment.

Browser tests use desktop Chrome by default (`SBT_BROWSER_CHANNEL` can select
another installed Playwright channel). They check loading, channel controls,
region-aware annotation picking, saved views, PNG output, invalid-file recovery,
same-origin GET-only requests and unchanged input-file hashes.

## Current limits

- Initial 2D viewer. Image z/t slice controls are available when axes exist;
  there is no 3D scene navigation, linked UMAP, expression plots or editing.
- The first associated table supplies colouring and tooltips when several
  tables annotate an element. View files match element names and coordinate
  systems, not a content checksum; reopen the correct dataset before importing.
- Chunk decoding has a 256 MiB cache ceiling, not a whole-browser memory limit.
  Tables, geometry, GPU textures and large unchunked arrays need additional
  memory. Large point sets follow the engine's limits and report notices.
- Format/codec support follows the pinned SpatialData.js engine. Automated
  browser coverage currently uses the synthetic Zarr v2 fixture; large real
  stores, Zarr v3 and other codec combinations still need representative testing.
- PNG exports screen-resolution canvas pixels. Scientific figure composition,
  editable legends, calibration controls and high-resolution export are not
  implemented here.

Bundled dependencies retain their licences in `THIRD_PARTY_LICENSES.txt`.
