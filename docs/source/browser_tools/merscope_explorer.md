# MERSCOPE Explorer

[Open MERSCOPE Explorer](https://merscope.spatialbiology.app/)

Use MERSCOPE Explorer to view images, cell boundaries and transcript locations
together in a **SpatialData** store. You can colour cells by gene expression or
annotations, compare segmentations, explore existing embeddings, and export
figures. Files are read locally; the viewer does not upload or edit the store.

## Prepare your data

Open a complete SpatialData **`.zarr` folder** or a ZIP containing that store.
Preserve its internal directories, metadata and chunks. A folder of raw
MERSCOPE instrument exports is not automatically a SpatialData store; ask the
person preparing the dataset for the exported store, or consult the
[SpatialData preparation guide](../guides/spatialdata.md).

The contents determine what you can display:

| SpatialData element | Use in the viewer |
| --- | --- |
| Images | Multichannel tissue images, with z/t slices where available |
| Labels | Raster segmentation masks |
| Shapes | Cell boundaries or other geometry, optionally linked to cell tables |
| Points | Transcript locations, with gene filtering where the data provides it |
| Tables | Cell observations, gene expression and existing embeddings |

The store should include coordinate-system transformations and the table
associations needed to place elements and join cells correctly. Equal cell
counts alone do not establish a valid table-to-segmentation match.

## Your first view

1. Open the app in desktop Chrome or Edge.
2. In **Setup**, open the `.zarr` folder or choose **Open ZIP archive**. Where
   available, **Open local / network folder** reads the directory on demand;
   **Open SpatialData folder (compatibility)** uses a file-list chooser.
3. Choose a **Coordinate system**. This determines which elements can be
   displayed together and how they are positioned.
4. Open **Layers → Add layers**, add an image and a shapes or labels layer,
   then select each layer's name to open its settings. Begin with a small
   selection rather than all layers in a large store.
5. Choose image channels and contrast. For shapes, check the linked cell table
   and choose a gene or annotation as described below.
6. Add a Points layer to show transcripts, then choose the genes of interest.
7. Drag to pan, scroll to zoom, and use Fit to recover the view. Click a cell
   and open **Inspect** to read its matched table row.

Loading indicators show the current stage, elapsed time and file-read activity.
A compatibility folder chooser may enumerate files before a count is available,
especially on a network drive. **Cancel loading** stops opening the store;
operating-system reads already issued may still finish in the background.
Subsequent layer/table reads happen on demand rather than loading the entire
store at once.

## Manage layers and coordinate systems

The sidebar contains **Setup**, **Layers** and **Inspect**. In Layers, search
or filter the picker by element type, add what you need, then hide the picker
to leave more room for the active list.

Active layers have independent visibility, drawing-order and removal controls.
Removing one from the active list does not delete data, and adding it again
restores its display settings. Image, Labels, Shapes and Points have different
controls; choose the layer name before adjusting them.

All panes share a coordinate system. Changing it starts a new view layout;
bookmark the existing layout first if you want to return to it. Elements that
do not belong to the selected system should not be assumed to overlap simply
because they came from the same experiment.

## Link cell boundaries to measurements

1. Select a **Shapes** layer and open **Shapes settings**.
2. Check **Linked cell table**. The default uses the store's association
   metadata; select another table explicitly when comparing segmentations.
3. Under **Cell linking**, check the cell-ID column and the reported number
   of matched boundaries. The table's region column scopes the join.
4. Under **Overlay data**, choose **Gene expression · X** and search for a
   gene, or choose **Cell annotation · obs** and select an observation.

Matches use exact cell IDs within the region. Duplicate IDs are rejected;
unmatched boundaries remain visible without a table-derived colour. Missing
values are grey. Different ID prefixes or the wrong segmentation/table pairing
must be resolved in the source preparation, not by assuming the rows have the
same order.

Each shapes layer retains its own table, field, colours and limits. Label-mask
colouring uses its associated table's observations; the explicit table and `X`
gene controls described above apply to Shapes.

### Interpret expression colours

`X` values are read as stored. Their meaning depends on the upstream analysis;
the viewer does not silently normalise them. Linear and optional log1p scaling
are display choices.

Use **Colour map**, **Reverse colours**, **vmin/vmax** or the upper-limit slider
to adjust the display. Bounds are in stored-value units **before log1p**.
Values outside the range clip to the palette ends. Reset the limits to return
to automatic scaling. Set comparable limits when comparing regions or panes.

## Show transcripts and gene lists

Select a **Points** layer and use **Transcript genes** to show all genes, a
selected subset or a workspace gene list. Start with a few genes for a large
transcript dataset. Filtering the display does not remove points from the store.

The bottom **Workspace tools → Gene lists** panel lets you create a named list
or import a CSV. A single `gene` column is sufficient; use `list,gene` for
multiple lists:

```text
list,gene
T cells,CD3D
T cells,CD3E
Myeloid,LYZ
Myeloid,CTSS
```

Gene names match exactly, including case. Duplicates within one list count
once. **Check against table** reports present, missing or ambiguous names for
the chosen cell table. Presence means the gene is available in `X`, not that
its expression is nonzero.

- For transcripts, choose **Workspace gene list** under Transcript genes.
- For shapes, choose **Gene list · sum of X** under Overlay data. Each cell's
  value is the sum of the matched genes' stored values; log1p, if selected, is
  applied to that sum for display.
- Missing genes are reported. A list with no matching genes produces an error,
  rather than a misleading all-zero expression overlay.

## Compare panes and copy settings

In **Workspace tools → Views**, duplicate the selected pane, up to four panes.
Click a pane header to edit its layers and colours. Pan and zoom are linked
by default; untick **Link pan and zoom** for independent navigation.

**Apply across views** copies a gene, list or supported cell annotation to
visible cell/Points layers in open panes. Each destination keeps its own table,
cell IDs, geometry and visibility. Read the per-layer result: unavailable or
ambiguous fields are skipped, and partial gene-list matches report omissions.
Hidden layers are not changed.

Cell-to-cell copies can include colour settings. Copying several selected
Points genes to cells creates/reuses a workspace list and displays their sum;
the destination cell palette and transform are retained and limits reset to
automatic. To copy from transcripts to cells, choose specific genes or a list
rather than **All genes**.

Use **Customise overlays** to adjust each pane's legend, original layer names,
scale bar, text size and positions. Individual legend labels can be renamed or
omitted without changing the underlying data.

## Explore an existing embedding

1. Open **Workspace tools → Embeddings** and choose a table.
2. Select an existing `obsm` projection and its two dimensions. UMAP, t-SNE,
   PCA and other numeric projections can be used; `obsm/spatial` is labelled
   as spatial coordinates rather than a newly computed embedding.
3. Follow the tissue pane's colours, or select an independent gene, list or
   annotation and colour limits.
4. Click a point to inspect its table row. Use lasso to highlight matching cells
   in tissue, or **Centre selected cell** to locate a matched shape.

**Other regions** adds faint context from the same table, subject to the chosen
sample limit. Large plots report sampling. Lasso selects displayed points,
not unplotted cells. Tissue lasso and **Select displayed** use the transformed
centres of visible Shapes; label masks support click selection. **Esc** returns
to Inspect. The selection is temporary, while projection and colour settings
are saved in workspaces and bookmarks. Export the embedding as PNG if needed.

## Bookmark and save your work

**Bookmarks** capture the complete viewer: coordinate system, panes, cameras,
gene lists and display settings. Rename them and add notes, then use **Return**
to revisit a comparison. A bookmark restores the versions of its gene lists
while retaining unrelated lists added later.

Workspace settings autosave in this browser. After reloading, reopen the same
store to resume. The compatibility folder/ZIP method identifies a store from
file paths, sizes and modification times; direct-folder opening uses directory
identity. Their local saves are separate. A modified or repackaged store may
therefore start a fresh workspace.

For a portable backup, choose **Workspace → Export workspace**. To restore or
transfer it, open the matching dataset first, then **Import workspace**. This
also lets you transfer settings between folder-opening methods. The JSON
contains settings, not dataset contents or permission to read local files.

Local saves are specific to the website address, browser profile and computer.
Export before clearing site storage or changing devices. Keep the original
store and check that imported settings match the intended layers and regions.

## Export a composed figure

1. Choose **Export** in the toolbar, or open the **Figure** drawer, then switch
   to **Figure mode**.
2. Set pane aspect ratio, export width and spacing; pan/zoom to compose.
3. Set background/foreground colours, names, legends and per-pane scale bars.
4. Confirm physical calibration if a scale in micrometres is required.
5. Choose PNG, JPEG or SVG, check the filename preview, and export. Allow
   selected transcript genes to finish loading before capture.

**Scale bars initially show coordinate units.** SpatialData does not always
declare the physical unit of the transformed system. Set **µm per coordinate
unit** only when you know that conversion. It refers to coordinates **after
transforms**, not necessarily the native image-pixel size. Changing coordinate
system resets calibration; a bookmark can restore it.

After calibration, **Lock field of view in µm** fixes each pane's physical
width and height, disables zoom and retains pan. Returning to **Viewer mode**
restores the cameras from before Figure mode.

PNG/JPEG render the composition as pixels. SVG embeds each SpatialData layer
as a separate raster group, with editable vector labels, legends and scale
bars. It does not export cell boundaries or transcripts as vector geometry.
Filename options can include region/view names, a UTC timestamp and selected
constant table metadata. Read metadata again after reopening the Figure drawer
if those filename fields are needed. Exports can create bookmarks automatically.

## Supported inputs and limits

- This is a 2D viewer. Image z/t slice controls appear where available, but
  there is no 3D scene navigation, dataset editing or new embedding computation.
- Shapes expression supports dense, CSR and CSC `X`. `raw.X` and alternative
  expression layers are not exposed in these controls.
- Existing dense numeric/dataframe projections are supported; sparse `obsm`
  projections are not. Projection reads are capped at five million rows;
  previews show at most 200,000 current-region cells plus the selected context.
- Transcript displays and large tables remain subject to memory/render caps.
  Read any sampling or truncation notice before interpreting a preview as a
  complete population. More panes require additional GPU memory.
- Zarr v2 is covered by the current validation. Zarr v3 and other codec
  combinations may require a compatible export from your data provider.
- Exports are capped at 32 million pixels, 16,384 pixels per side and 128 MiB
  of intermediate SVG. Reduce dimensions if an export exceeds these bounds.

For slow network folders, missing matches, workspace recovery and connection
problems, see [Troubleshooting](troubleshooting.md).
