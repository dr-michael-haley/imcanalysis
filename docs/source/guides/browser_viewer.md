# SBT IMC Explorer: browser review

For the hosted application, start with the
[IMC Explorer user guide](../browser_tools/imc_explorer.md) in the
dedicated [Browser-based tools](../browser_tools/index.md) section. It covers
opening data, review controls, exports and workspaces without local build steps.
This page retains the detailed reference and local-launch notes for existing links.

SBT IMC Explorer is an initial read-only browser viewer for IMC analysis
results. Open a local `.h5ad`, a mask folder and an image folder, then navigate
ROIs, mix channels, colour cells, inspect observations and view existing
embeddings. The files remain on your computer and are never uploaded.

## Open the viewer

The welcome screen includes a feature overview, asset requirements, a folder
layout example and links to SBT's GitHub, documentation, Guidejar tutorial and
Michael Haley's contact email. **Open Setup** takes you to the file controls;
**Resume a workspace** takes you to your saved workspace choices. The screen
scrolls when needed. Review tools appear after AnnData is opened, even if image
or mask folders have not yet been selected.

An already hosted static copy needs only a browser. In a development checkout,
build the application once, then use the SBT launcher:

```shell
cd web_viewer
pnpm install --frozen-lockfile
pnpm build
cd ..
sbt gui web
```

Node.js 22.12+ and pnpm are needed to build, not to view a hosted copy. The
optional Python launcher only serves application assets on localhost. It does
not process scientific data, require Napari, or create a project. Use
`sbt gui web --check` to check for a build, `--port 8877` to change the port,
or `--no-browser` to print the URL. Press Ctrl+C in its terminal to stop it.

This initial version is tested in desktop Chrome and Edge. Other browsers and
large production datasets still need acceptance testing. Do not open the HTML
directly using `file://`; workers need HTTP/HTTPS. No offline cache is promised.

## Review a dataset

The sidebar has three tabs: **Setup** contains dataset loading, workspaces and
image normalisation; **ROIs** contains navigation, ranking and visit tracking;
**Views** contains the named panes and their image/overlay controls. Numbered
sections restart within each tab and remain individually collapsible.

While AnnData or an ROI is loading, a central indicator shows the current stage
and which view is being prepared. Previous tissue images are dimmed until the
new ROI is ready. You can still move to a different ROI during loading; the
indicator follows the latest selection. Rendering yields between pixel batches
to keep the interface responsive with several views.

1. Choose the **AnnData** file, **Masks** folder and **Images** folder. Their
   order does not matter. Folder selection grants read access to those files;
   no data is sent to the application host.
2. Check **Cell identity columns**. Defaults are `ROI` and `ObjectNumber`.
   Custom column names can be selected explicitly. Each `(ROI, object ID)` pair
   must be unique. IDs must be positive integers fitting uint32; background is
   zero. Missing identities and duplicates stop loading.
3. Choose an ROI or use the previous/next buttons. Missing masks and cells are
   reported. Mask objects without an AnnData row are grey context and cannot
   acquire another cell's metadata.
4. Start with three image channels; use **Add channel** for up to six. Defaults
   are pure red, green, blue, cyan, magenta and yellow; each colour is editable.
   Untick a channel to hide it while keeping its selection and contrast, or use
   its **×** button to remove the slot. Without a dictionary, low/high values are
   in the original TIFF intensity units; **Auto** uses the sampled minimum and
   99.5th percentile. With a dictionary, contrast is on 0–1 and **Reset** restores
   that range. Low and high each have a slider and an exact numeric input.
   Display changes never change the data. New selections of multipage scalar
   images use their first TIFF plane; older saved plane settings still restore.
5. Choose a categorical/numeric observation or expression marker under
   **Colour by**. Marker values use the selected `X` or `layers/...` source with
   no additional normalization. Use **Show population** to restrict the overlay,
   and choose outlines, filled cells or hidden overlays. Marker overlays offer
   Viridis, Plasma, Magma, Inferno, Cividis, greyscale, pure RGB ramps,
   blue–white–red and the original palette, with a reverse option. The legend
   follows the selected palette; missing values remain grey.
6. Drag to pan, scroll to zoom, or use **Fit view**. Click a segmented cell to
   read its observations. The existing embedding shows the current ROI's cells;
   clicking a point selects the same cell in the tissue view. No embedding is
   recomputed. Image scale is shown in pixels, not unverified physical units.

Click any numbered section heading on the left to collapse or reopen its
controls without changing their values. **Collapse all / Expand all** controls
all sections together. The eye beside **Image channels** or **Cell overlay**
temporarily hides or shows that entire category, independently of expanding the
section. Individual channel visibility, contrast, colours, overlay appearance
and the selected cell are retained. Hiding the overlay also hides its selected-cell
outline. The **Review tools** bar folds away the bottom panel. Its tabs show
**Cell details**, **Embedding**, **Population remapping**, **Plots**, **Flagging**, or **Bookmarks**, one at a time.
Drag the divider above the tabs to change the panel height, or focus it and use
the up/down arrow keys. **Reset height** returns to the default. The active tab
and chosen height are saved in the workspace. A fitted image refits to
the available space; a manually zoomed or panned view keeps its scale and centre.
The tissue view extends to the top edge with no header bars. Zoom and **Fit view**
float inside its bottom-right corner; **Close dataset** is in **Open your dataset**.
The floating icons also provide previous/next ROI and image/overlay visibility
for the selected pane. Tooltips name their actions and the affected view.

## Multiple views and ROI review

Under **Views**, use **Add view** to duplicate the selected view, up to four
active panes. Edit **View name**, then choose that pane's six-channel mix and
cell overlay. Select a pane by clicking its image, its name badge, or the
**Selected view** list. A mint outline marks the selected pane. **Remove view**
removes that pane from the current workspace; at least one pane remains.
The inspector and embedding follow the selected pane. Pan and zoom in any pane
move all panes together, using the same image centre and pixel scale.

ROI navigation retains every pane's channel names, visibility, colours, plane,
contrast and overlay settings. An unavailable channel is reported as missing;
it is not substituted and reconnects when available in a later ROI. Automatic
contrast is calculated when a channel is first chosen or **Auto** is pressed,
then retained between ROIs. Normalisation is shared across all panes.

In **ROIs**, order the full list by name, total cell count, the count of a chosen
population/observation value, mean numeric observation, or mean marker intensity.
Marker ranking uses the chosen `X` or `layers/...` values, independent of image
normalisation. Means exclude missing/non-finite values and include implicit
sparse zeros. ROIs without finite values sort last. Population counts include
zero-count ROIs. Select ascending or descending, then **Apply order**; the
status states the applied ordering. **Reset order** restores ascending names.
Reordering preserves the current ROI. The configurable first-X buttons (1–100)
show ranked ROIs and scores; previous/next follows the complete ordered list.

Shortcut colours track visits against each pane's channel, contrast,
normalisation and overlay recipe: green is unvisited, amber means some current
recipes have been visited, and grey means all have. An ROI is recorded after its
view has rendered and remained stable for 0.7 seconds. Missing required images
do not count as reviewed. Changing settings creates a different review recipe;
renaming panes, zooming and selecting cells do not. **Reset visits for current
views** clears those recipes' history. Visits indicate display, not an explicit
scientific sign-off. Names, pane presets, ordering and visit history are saved
in the workspace JSON.

**Close dataset** reloads the app. Wait for **Saved locally** to retain the view
and population proposals; unsaved edits trigger a browser warning. The viewer
does not write modified AnnData.

## Propose population names, merges and colours

Open **Population remapping** in the bottom panel. Choose an existing observation
(for example `leiden` or `population`) and select **Create remap**. Each original
value gets one row, its total cell count across all ROIs, an editable proposed
name, a colour, and notes. Missing values have their own row when present;
unused categorical levels are retained with count zero. Up to 20 named remaps,
with at most 10,000 original populations each, can be kept in a workspace.

Edit a proposed name directly, or select rows and use **Rename / merge selected**
to give them a shared name. Matching proposed names form a merge. Every member
of a merge shares one colour: recolouring any member updates the others. Renaming
into an existing group adopts that group's colour. Notes remain attached to the
original population, so merging does not discard separate observations. **Undo**
reverts recent edits during the current session. Use search and page controls
for long population lists; selection can span pages.

**Use in selected view** previews the proposal in that pane. Remaps also appear
as **Remap · name** in **Cell overlay → Colour by**. The population filter,
legend, cell details and embedding use those proposed labels and colours.
Keep another pane on the original observation to compare the two. ROI changes
retain the chosen remap. Edits refresh all panes using it. Changes to proposed
names or colours create a new visit recipe; notes and the remap's title do not.
Removing a remap returns panes using it to cell outlines.

Every remap autosaves locally, including notes, colours and source values.
Opening a workspace restores the proposals after the original inputs are relinked,
and checks source populations and counts against the AnnData. No new observation
or colour array is written to the source file. These are review proposals;
subclustering, cell-level reassignment and AnnData export are outside this tool.

### Export a schema for Python

**Export CSV** downloads the selected remap. Columns are `source_obs`,
`source_value`, `source_value_json`, `proposed_name`, `colour`, `notes`, and
`remap_name`. There is one row per original population, including every member
of a merge. UTF-8 CSV quoting preserves commas, quotes and multiline notes.
`source_value` is for display; use `source_value_json` to preserve exact types,
including numeric versus string labels, empty strings and missing values.
Source values are not trimmed or silently combined.

For example, with an AnnData already loaded as `adata`, this creates proposed
labels in memory without modifying the AnnData:

```python
import json
import math
import numbers
import numpy as np
import pandas as pd

schema = pd.read_csv("Immune_review.remap.csv", dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
assert schema["source_obs"].nunique() == 1
source = schema["source_obs"].iloc[0]

def source_key(value):
    if pd.isna(value):
        return ("null", None)
    if isinstance(value, (bool, np.bool_)):
        return ("boolean", bool(value))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, numbers.Real):
        return ("number", value) if math.isfinite(value) else ("null", None)
    raise TypeError(f"Unsupported population value: {value!r}")

keys = [source_key(json.loads(value)) for value in schema["source_value_json"]]
assert len(keys) == len(set(keys)), "Duplicate source rows"
lookup = dict(zip(keys, schema["proposed_name"]))
cell_keys = [source_key(value) for value in adata.obs[source].array]
assert all(key in lookup for key in cell_keys), "Schema does not cover this AnnData"
labels = pd.Categorical([lookup[key] for key in cell_keys],
                        categories=list(dict.fromkeys(schema["proposed_name"])))
palette = schema.groupby("proposed_name", sort=False)["colour"]
assert palette.nunique().max() == 1, "Conflicting merged colours"
colours = palette.first().reindex(labels.categories).tolist()
```

Apply `labels` to a chosen new `adata.obs` column and `colours` to its matching
`adata.uns["<new_column>_colors"]` only when ready to accept the proposal.
This CSV is a browser-review schema, not the positional CSV template consumed
directly by the pipeline's `sbt` remap stage; select/convert the relevant columns
if using that stage.

## Bookmarks and reusable view presets

Use **Bookmark** in the viewer footer or **Bookmark current viewer**
in the **Bookmarks** bottom tab. Each row has an editable name, optional comments,
the ROI and view names, a **Return** button and a remove button. A bookmark captures
all active panes, their channels, contrast, overlay and aligned-layer settings,
legend settings, selected pane, ROI, selected inspector cell, normalisation,
embedding settings and the shared camera's scale and image centre. It does not
copy images, expression data or transient multi-cell selections.

**Return** restores the captured state. If ROI filters exclude the bookmark's
ROI, they are cleared to make it reachable. Bookmarks refer to the workspace's
current remaps, flag fields and aligned assets; they do not freeze copies of
those annotations. Removing a referenced field or layer makes that saved view
unavailable until the dependency is restored or the bookmark is replaced.

In **Views → Saved view presets**, enter a name and **Save current view**. A preset
captures one pane's channel/overlay/layer/legend settings. Clicking its name or
assigned F-key applies it only to the selected pane, preserving its name, the
ROI and shared camera. Each F-key can belong to only one preset. F5, F11 and F12
remain browser shortcuts; typing in an input disables preset shortcuts. Some
keyboards require Fn with an F-key. Up to 500 bookmarks and 100 presets can be
stored, subject to the workspace size limit. Both libraries autosave and are
included in portable workspace JSON.

## Linked embedding and cell selection

The **Embedding** tab can follow the selected view's colours or independently
colour by an observation or marker from `X` or a layer, with a chosen colour map.
**Square** centres a square plotting area. The current ROI is always read in full;
its displayed population filter follows the selected tissue pane.

Enable **Other ROIs** for faint context points spanning the dataset projection.
Context is a reproducible stratified sample, capped at 50,000 dataset cells by
default; choose 10,000, 100,000 or 200,000 when appropriate. Smaller datasets show
all cells. Current-ROI cells are plotted at full opacity without duplicate
context points. The status explicitly reports sampling and point scope.

Colours/context are read in a cancellable worker. The first context read scans
coordinate and identity blocks sequentially; expression colours read one marker,
never the complete expression matrix. Context is cached across ROI changes while
the projection, feature, source and sample limit are unchanged. **Update** retries
or rereads these data; **Cancel** stops the read. Large network datasets still
need time for the initial context scan. Presets and bookmarks never upload data.

Use the footer's mouse-mode menu to choose **Select cells** or **Lasso** in tissue
and embedding. **Inspect & pan** restores normal inspection; **Esc** also returns
to Inspect and cancels an unfinished lasso. Both spaces
highlight the same ROI/object-ID selections. Lassoing context selects only the
plotted sample, not every cell in that area of the full dataset. In inspect mode,
clicking a context point navigates to its ROI and opens that cell's details.

## Plot populations and markers

Open **Plots** in the bottom panel. Choose a plot type, population observation
or proposed remap, and cell scope, then select **Generate / update**. The five
available plots are:

- **Marker matrix plot:** population means for selected markers in `X` or a
  layer. Unscaled means retain stored intensities. Marker-wise z-scores and
  0–1 scaling operate on the population means for each marker, without changing
  AnnData or applying the image normalisation dictionary.
- **Population composition bars:** stacked counts or percentages grouped by
  an observation such as ROI, patient or condition. Remap colours are retained.
- **Population composition heatmap:** the same population-by-group counts or
  percentages in a compact matrix.
- **Compare labels:** cross-tabulate the comparison labels (rows) against the
  primary labels (columns). Either can be an observation or a proposed remap.
  Choose counts, within-row percentages, or within-column percentages. Column
  percentages are useful for seeing the contributors to a proposed merge.
- **Categorical alluvial plot:** connect two observations or proposed remaps,
  with primary labels on the left and comparison labels on the right. Ribbon
  widths represent exact cell counts after filtering; hover to see counts and
  percentages. Source colours are retained, and missing labels remain explicit.
  Export SVG, PNG or the exact contingency table as Data CSV.

Scope can be **Current ROI**, **Selected ROIs**, or **All cells / ROIs**. Use
Ctrl/Cmd-click to select multiple markers or ROIs. Under **Population & metadata
filters**, load population choices or values from an additional observation,
then select the values to include. An empty selection includes every value.
All filters intersect. Missing values have an explicit `(Missing)` category;
source numbers, strings, booleans and missing values remain distinct.

Composition percentages are normalized within each sample/group **after all
filters**, including population selection. Matrix means exclude non-finite
values and include implicit sparse zeros. Merged means are computed over the
contributing cells, not by averaging the original population means. These five
plots use all selected cells; no sampling is applied.

**Ordering & appearance** provides dataset/name/count ordering, an optional
population used to sort composition samples by count, colour maps, heatmap axis
swapping and a custom title. Limits are 100 markers, 200 represented groups per
axis and 20,000 heatmap cells. Use filters for larger category sets. `raw` is not
an expression source in this release.

Plot calculations use a separate read-only worker. A progress indicator reports
the scan and **Cancel** stops it without affecting tissue navigation or the
previous plot. Dense and CSR matrices are read in batches. CSC processing holds
one marker column at a time (up to 40 MB for five million cells), alongside a
20 MB row-to-population index for marker plots; the whole expression matrix is
never materialized. Actual speed depends on HDF5 compression/layout, marker count
and the device. Five-million-cell plotting still needs production acceptance
testing.

Plots are snapshots. Changing the ROI, settings or a used remap marks the plot
as needing an update. Changing an unrelated ROI does not invalidate an all-cell
plot. The existing figure remains visible until a new calculation succeeds.
**Expand** enlarges the plot panel; **Restore size**, Escape, or another review
tab returns it to the bottom panel.

**SVG** exports a self-contained vector figure. **PNG** exports a raster figure
with bounded dimensions for browser memory. **Data CSV** exports displayed
values, counts, and—for marker plots—unscaled means and finite-value counts.
Typed JSON row/column identities preserve exact source values. The first data
row's `settings_json` contains the filters and remap snapshot used for that
figure; subsequent rows leave it empty. Exports always describe the displayed
snapshot, even if controls have since changed. No source files are modified.

Plot settings are saved in the workspace. After restoration, select **Generate /
update** to recreate the plot; figures and cell-level data are not embedded in
the workspace JSON.

## Representative markers

In **Views → 03 Representative markers**, select one observation or remap
population in **02 Cell overlay**. The panel automatically ranks up to six
positively enriched markers by default, with a configurable limit of 1–30.
It compares the population mean with the mean of all other cells, divided by
the standard deviation across finite cells in the chosen scope. These are
descriptive scores, not p-values or formal differential-expression results.
Constant markers and markers with no positive score are omitted; both groups
must contain cells. Missing expression is excluded and sparse zeros are included.

**Current ROI only** defaults to checked. Uncheck it to compare across the full
AnnData; the calculation runs in a separate worker and can be cancelled.
Choose `X` or an available layer. Each marker displays two overlaid expression
density curves on shared axes: the selected population (solid teal) and all
other cells in the calculation scope (dashed amber). Each curve has unit area,
so unequal group sizes do not dominate the comparison; the legend reports
finite cell counts for each group. Curves are approximate distributions made
from 64-bin summaries, with the same Gaussian smoothing bandwidth (1.5 bins)
and reflected boundaries at the shared observed range. This keeps memory
bounded without retaining every expression value. **R**, **G**, and **B** assign that marker's
image to slots 1, 2 or 3 of the selected view, respectively. Buttons are disabled
if the current ROI has no matching image. Marker rankings are recomputed after
workspace restoration; the settings are saved.

The controls at the top set lower and upper **X-axis quantiles**, from 0 to 1
(for example 0.01–0.99). For each marker, these are estimated from the pooled
histogram counts of both groups. They crop the displayed range; densities retain
their full-range units rather than being renormalised to the crop. Quantile changes
redraw cached curves without reading AnnData. RGB assignments and image-only
refreshes also retain the calculation. Changes to the population, expression
source, ROI scope or ranking settings invalidate it.

## Flagging cells

The **Flagging** bottom tab creates categorical label fields without changing
AnnData. Create a field, name it, then add or rename categories and choose their
colours. The viewer footer groups ROI navigation, mouse mode, **View**, and
**Bookmark**. The mouse-mode menu provides **Inspect & pan**, **Select cells**,
and **Lasso**, with a short explanation of each. A contextual selection strip
shows the cell count, number of selected ROIs, and the space used by
**Select displayed**. **Done** or **Esc** returns to Inspect while keeping
the selection; **Clear** removes it across all ROIs.
These tools work with any bottom tab, including when the bottom panel is closed
or no flag field exists. Tissue lasso uses segmentation centroids and the selected
pane's population filter. Mask objects without AnnData rows are excluded.
Click mode still allows dragging to pan the tissue. Embedding lasso selects
plotted points using their ROI and object ID, including plotted context cells.

Selections remain across ROI changes and are highlighted in both spaces.
**Select displayed cells** acts on the last interacted tissue/embedding space;
when the embedding is hidden it selects the current tissue ROI's displayed
population. **Clear selected cells** clears all ROIs. Selection itself is
transient; **Apply to selected cells** saves the category for every selected
ROI/object-ID pair. The assigned labels persist in the workspace.

**Unassign selected** removes labels and **Undo** reverses recent edits (up to
30 actions in the current session). **Use in selected view** previews a flag
field as a categorical cell overlay. Unassigned cells appear grey. Up to 30
fields with 100 categories each are supported, within the workspace's 10 MB
size limit. Source identity columns cannot be changed while assigned flags
exist; save or remove the fields first.

All fields autosave in the local workspace. **Export CSV** downloads the selected
field's assigned cells, with columns `roi_column`, `roi`, `object_id_column`,
`object_id`, `label_field`, `label`, and `colour`. Join externally using both
ROI and object ID; an object number alone is not unique across ROIs. Unassigned
cells are absent from this CSV. Restore checks that every saved identity exists.
Flag labels are available in cell overlays; plots currently use observations
and population remaps.

## ROI metadata and filtering

The **ROIs** tab has **01 Navigation**, **02 Filtering**, and **03 Sorting**.
Navigation shows the current ROI and its sample metadata. Choose **Read metadata**
in Filtering to populate metadata intentionally. Opening the tab, changing ROIs
and restoring a workspace do not start this expensive read. The scan identifies observations whose values are constant
within every ROI, including numeric and missing values. This scan can be
cancelled and retried. Per-cell columns that vary inside an ROI are excluded.

Filtering accepts one or more values per field, with OR within a field and AND
between fields. Applied filters are shown explicitly and can be removed
individually or reset together. Navigation arrows and the top-X shortcuts use
the filtered list, while sorting retains the requested ranking. A zero-match
filter clears the tissue display and reports the empty result. Resetting the
filters makes the complete ranked list available again. Filter choices persist
in the workspace. After reopening, saved filters are shown as pending and the full
ROI list remains available until **Read metadata** is chosen. Unavailable saved
fields are reported, rather than silently hiding ROIs.

## Aligned reference and annotation layers

In **Setup → 03 Aligned layers**, choose a type and add one folder per named
layer. Except for additional IMC folders, each image must be named exactly after its ROI, for example `ROI_A.png`
or `ROI_A.tiff`. Subfolders are searched; duplicate matches are rejected.
Images must have the same pixel dimensions and alignment as the tissue mask
and channels. No registration or resizing is applied.

- **Colour / reference image** supports RGB/RGBA TIFF (unsigned 8/16-bit), PNG,
  JPEG, BMP, GIF and WebP. This is suitable for aligned H&E or reference images.
  Animated formats display their first frame.
- **Scalar image** supports single-plane TIFF intensity or probability maps,
  with colourmap, exact low/high contrast and automatic contrast controls.
  It does not apply the IMC normalisation dictionary.
- **Categorical labels** supports non-negative integer TIFFs, including region
  annotations and additional masks. Choose an individual annotation set folder
  containing its ROI TIFFs and, optionally, NapariSBT's `label_mapping.csv` with
  `value,label,color` columns. Colours use `#RRGGBB` or `transparent`; zero is
  always transparent. With a mapping, unmapped IDs are transparent. Without one,
  colours follow label IDs consistently across ROIs. The legend shows up to
  100 labels. Labels are visual context; they do not change AnnData or cell picking.
- **Additional IMC image folder** accepts the same layout as the original Images
  folder: channel TIFFs in ROI subfolders, or flat ROI-prefixed filenames.
  Channels need not exist in AnnData. Select them in **Views → 01 Image channels**,
  where the source folder name distinguishes them from original channels with
  the same name. They share the six slots per pane, and preserve their source,
  contrast and colour across ROIs and workspace restoration.
  The dropdowns group channels by source; channels present in other ROIs remain
  selectable and say **unavailable in this ROI**. **Choose image channels** in
  Setup opens these controls directly.

For categorical layers, **Load dictionary CSV** can select a separate file with
any filename. Supported defaults include `value,label,color`, `original_num,name,label`
and `label_value,label_name`. Choose the ID, name and optional colour columns in
Setup. `original_num` is preferred over a merged `label` column when both exist;
ambiguous repeated IDs with different names are rejected. Without a colour column,
colours are assigned consistently by ID. Place the CSV anywhere beneath the data
root used for restoration. Its reference and column choices are saved; the file
is reread when restoring the workspace.

**Views → 04 Aligned layers** appears only after a reference, scalar or categorical layer is added. Its cards
show only controls relevant to loaded types. Each pane has independent visibility,
opacity, blending, scalar contrast/colourmap, and label fill/outline settings.
Categorical outlines have an **Outline width (px)** control from 1–20 source
image pixels. The outline grows inside each region, without spilling into its
neighbours. Width is retained independently for each view.
Layer order follows Setup's list (later entries draw on top); **Lower / Raise**
changes that order across all panes. Image channels sit below these layers and
cell overlays remain above them. Missing ROIs show an explicit notice and keep
their settings for the next matching ROI. At most 12 extra layers are supported.

This follows the legacy IMC explorer's extra-image sources and annotation-set
format; newer NapariSBT extra image folders also supply scalar channel assets.
All new sources and per-view settings are included in browser workspaces.

Each view also has a floating legend for enabled image channels, loaded aligned
layers and its cell overlay. The footer's **View** menu names the selected pane
and provides visibility checkboxes for its image channels, cell overlay, and
legend, and scale bar. **Scale bar settings** offers automatic or fixed length,
position, colour, thickness and text size. The shared calibration defaults to
**1 image pixel = 1 µm**; change it to match the acquisition pixel size. Zoom and **Fit** in the same menu apply to the linked camera across all
panes. Small **− / +** buttons on each legend change
its text, swatches and width from 70% to 200%. Visibility and size are saved
independently per preset; resizing does not rerender tissue or reset ROI visits.
Legends follow channel colours, dictionaries, population filters and continuous
colourmaps; long category lists are capped at 24 entries per group, with the
sidebar retaining the fuller legend.

**02 Cell overlay → Show populations** supports several selected categories.
Use Ctrl/Cmd-click to add or remove categories, or Shift-click for a range.
**Show all** clears the filter. Choices persist between ROIs, including categories
absent from the current ROI. Representative markers requires exactly one category.

## Exporting tissue figures

Choose **Export** in the viewer footer to open the bottom **Figure** tab, then
switch from **Viewer** to **Figure** mode. The preview uses simplified names,
legends and scale bars. Choose white on black, black on white, or custom colours;
adjust label sizes, legend entries and optional backgrounds behind labels.
**Scale-bar text (px)** is independent of the view-name and legend text sizes.
Expand **Scale bars · choose panes** to select individual panes (or use **All
panes / No panes**). Each pane has automatic/fixed length in µm and line width
in export pixels. Length, line width and visibility are shared with that pane's
View-menu settings; figure text size is separate from viewer text size.
Per-view legend and scale-bar visibility still apply. Selection highlights and
interface controls are excluded from exports.

Set a per-pane aspect ratio, or enable **Lock field of view in µm** and enter a
width and height, such as 500 × 500 µm. The lock uses the calibration in **View →
Scale bar settings**, disables zoom/Fit and keeps panning available. All panes
show the same physical extent. Returning to **Viewer** restores the previous
camera and unconstrained layout. Figure settings persist for the next export.

Choose the pane width in output pixels and the spacing between panes. Export
supports PNG, JPEG and layered SVG. SVG embeds tissue, each annotation layer and
cell overlays as separate raster layers; names, legends and scale bars remain
editable vectors. Annotation layers with additive blending use SVG `plus-lighter`,
whose support varies between editors; PNG/JPEG preserve the composed pixels.
The maximum export is 32 million pixels and 16,384 pixels on either side.

Under **Export → File name**, choose a label/prefix and whether to include the
ROI name, all exported view names and a UTC timestamp. Select ROI metadata
fields to append `field-value` components. **Read ROI metadata** performs the
same intentional scan as the ROIs tab and makes observations constant within
each ROI available. It also applies any pending ROI filters. Saved filename
field choices remain selected after reload, but exporting with them requires
reading metadata again; no expensive scan starts automatically. Missing values
are labelled `Missing`, and unavailable fields must be read or deselected.
The live filename preview shows sanitised characters and any shortening needed
for filesystem limits. The timestamp, when enabled, and format extension remain
intact. Naming settings persist in workspaces and export bookmarks.

**Automatically bookmark each export** is on by default. A successful export
adds a bookmark containing the ROI, all panes, camera and figure settings. Add
comments or return to that state from **Bookmarks**. Workspaces save figure
settings, calibration, scale bars and these bookmarks, but not exported pixels.

## Persistent workspaces

The **Workspaces** library is at the top of **Setup**. The active workspace saves
automatically to this browser's IndexedDB after edits, including remaps, notes,
flags, views and navigation. Wait for **Saved locally** before closing the page.
On your next visit, select a saved workspace and choose **Open** beside it.
Use its name field to rename it, **+ New** for another dataset, or
**Duplicate** for an independent review of the same data. **Delete**
requires an inline confirmation and removes only that browser workspace, including
its annotations; it never removes source files. There are no recovery snapshots.

On browsers supporting directory handles (including current Chrome/Edge),
**Connect folder** associates the active workspace with one common root
folder. The **Choose data folder** button during restoration remembers that root too.
Only read access is requested. A later **Open** checks the stored
permission, asks again if needed, then reads the original assets from that root.
It opens the saved file and folder paths directly, without enumerating the whole
root or searching unrelated datasets. Only the selected asset folders are listed
to verify their saved file counts, paths and sizes. **Assets & connection** contains
paths relative to the connected root. The primary asset pickers retain file and
folder handles, so connecting their common root resolves the full relative path
without searching, including paths such as `HE_annotations/niches`. New assets
added after connecting the root are resolved when saved. For older workspaces
that stored only a basename, correct the path once in **Assets & connection**, or
use **Locate folder** after choosing the root. The compatibility asset pickers
cannot reveal ancestors outside the selected folder, so those initial paths may
also need this correction. Missing
paths fail explicitly; they never trigger a recursive search elsewhere.
If access is revoked, the folder moves, or a network share is disconnected,
reconnect using **Choose data folder**. The alternate folder input works in
browsers without this API, but its access cannot be remembered between sessions.
This **Compatibility chooser** is explicitly labelled as listing the entire
folder: the browser performs that enumeration before the viewer receives files.
Prefer the primary chooser for large network roots.

Local workspaces belong to the website origin, browser profile and computer.
They are not cloud-synchronised; localhost and the hosted website have separate
libraries. The viewer requests persistent browser storage, which the browser may
decline. Clearing site data still deletes the library, and private browsing may
remove it on exit. Storage failures are shown explicitly and block switching
away from unsaved edits. If two tabs edit the same workspace, the second stale
write is rejected; export its current view before reopening the saved version.

The secondary **Import / export** section provides **Import JSON**,
**Export JSON**, and, where supported, **Save to file…**. The last option lets you
choose a JSON file and update that same file in the current session. File export
is a portable copy; it does not replace automatic local saving. Choose a new
output file after switching or duplicating workspaces. A workspace contains:

- The AnnData and folder references, cell identity mapping and current ROI.
- Aligned layer names, types, folder references and stacking order, plus each
  pane's visibility, opacity, blending, colourmap, contrast and label appearance.
- Additional IMC source identities, separate label CSV references/column choices,
  representative-curve quantile bounds and each pane's floating legend visibility and size.
- Up to four named view presets, each with six channel slots, visibility,
  colours, TIFF planes, contrast and independent overlay settings.
- The normalisation dictionary itself, its fallback quantile and minimum counts.
- Overlay visibility, annotation or marker, expression source, population filter,
  appearance and opacity; the embedding and selected cell.
- Pan/zoom and whether the view was fitted, plus expanded/collapsed panel states.
- Population remaps with proposed names, merge colours, source values and notes;
  the active remap editor and bottom-panel tab/height.
- The selected pane, ROI ordering, top-X limit and visit history.
- Flag fields, categories, colours and assignments keyed by ROI + object ID;
  sample-metadata ROI filters and representative-marker preferences.
- Plot type, expression source, selected markers, scope, filters, ordering and
  appearance (the generated figure itself is not saved).

Choose **Import JSON**, select the JSON, then use **Choose data folder** in
**Assets & connection**. Select one common folder containing the AnnData,
masks, images, aligned layers and any separate CSV dictionaries. The import
creates a separate named entry in your local library.

The checklist checks each asset in turn and shows **Waiting**, **Checking**,
**Registered**, **Cached list** or **Needs attention**. Missing items move to the
top and expand to show **Locate file/folder**, **Retry** and **Edit path**.
Registered items collapse but can be expanded for details. Fix a missing path
or locate its file, then retry only that asset; successful checks are retained.
The view opens automatically once all assets are registered. Nested locations
such as `HE_annotations/niches` are retained relative to the connected root.
Native Locate controls update that path automatically. With compatibility
pickers, check **Edit path** when the parent folder cannot be determined.
Image normalisation dictionaries are embedded in the workspace; separate
annotation CSV dictionaries are listed and can be located individually.

**On open** defaults to **Full asset check**, including on workspace switches.
This lists only the saved asset folders and verifies their file names and sizes.
**Quick reconnect** reuses file lists saved by previous successful checks in this
browser. It checks the named folders, opens AnnData and dictionaries, and opens
image/mask files as they are needed. Cached folders are clearly labelled; new
or changed locations without a matching cache receive a full check. Use full
checking after changing files, or when validating a shared dataset. Cached lists
contain paths and sizes, not pixels, and are not included in workspace JSON.
Quick reconnect reduces network directory operations; it does not skip AnnData
opening, identity indexing or saved annotation validation.

One connection area reports asset checks, AnnData opening and rendering, without
an additional loading card over the viewer. **Cancel** stops the connection.
The compatibility root chooser is still available, but the browser enumerates
its whole folder before application progress is available. It always performs
full checking; prefer the standard chooser for large network roots.

JSON cannot grant a browser access to local files. It contains no AnnData matrix,
image pixels, absolute file paths or browser permissions, and source data is
never uploaded or modified. Keep the source files alongside your saved workspace.
Stored directory handles stay inside the browser and are excluded from JSON exports.
AnnData matching uses its filename and byte size. Folder matching uses a digest
of supported asset paths relative to the selected folder and their byte sizes; folder roots
can move while their internal paths and contents stay the same. If a folder is
renamed, update its relative path in **Assets & connection**. These are
metadata checks, not checksums of file contents, so retain the original datasets
for reproducible review. Inputs absent from a saved workspace are cleared on load.

The JSON uses format `sbt-tissue-explorer-workspace`, version `11`, with `sources`
and `view` objects. Version-1 JSON files migrate to a single **View 1** preset;
version-2 files retain their named presets. Both receive empty remaps and default
review-panel settings. Version-1–3 files receive default plot settings. Version-1–4
files receive empty flags/ROI filters and default representative-marker settings;
their continuous overlays retain the original palette. Versions 1–5 receive empty
aligned-layer lists. Versions 1–6 default to full-range representative curves
and visible floating legends. Versions 1–7 default to 100% legend size; relative
asset paths are saved after reconnecting or setting Asset locations. Versions
1–8 receive empty bookmark/preset libraries and current-ROI embedding defaults.
Versions 1–9 receive default figure/scale-bar settings and 1 px annotation
outlines; an existing single-population filter becomes a one-item selection.
Versions 1–10 receive default filename choices; scale-bar text initially uses
the previous figure text size to preserve the saved appearance.
Files larger than 10 MB, unsupported versions and malformed
settings are rejected before changing the current view. Workspaces are display
settings for this browser viewer, not NapariSBT workspace bundles or analysis
configuration. A manual zoom restores its pixel scale and image centre; a fitted
view adapts to the current browser size.

The top-left branding displays the viewer's application version (independent of
the workspace format version) and a **How to guide** link, which opens the
Guidejar guide in a separate tab.

## Image normalisation dictionaries

Open **Image normalisation** and choose a local Nimbus CSV or NapariSBT JSON.
The dictionary can be selected before or after the dataset. Preferred CSV:

```csv
marker,vmax,lower_threshold
CD3,120,2
CD20,80,0
```

Legacy `Marker,Value` CSV and marker-to-number JSON are accepted with a zero
lower threshold. Structured JSON may map markers to `{"vmax":120,
"lower_threshold":2}` directly or wrap the mapping in `"normalization_dict"`,
as in a NapariSBT workspace's `display/normalization.json`. Nimbus CSV aliases
such as `channel`, `baseline_vmax`, and `lower_bound` are also supported.
Vmax must be finite and positive, with `0 <= lower_threshold < vmax`. Duplicate
marker names ignoring case and invalid bounds are rejected. An invalid
replacement leaves the previous dictionary and display settings unchanged.

The viewer follows NapariSBT's image-display transform:
`clip((image - lower_threshold) / (vmax - lower_threshold), 0, 1)`.
Matching checks channel names, TIFF stems and unambiguous AnnData panel aliases;
case and punctuation variants are recognized. Ambiguous matches produce a
visible error instead of choosing a marker arbitrarily. Each channel reports
whether it uses dictionary bounds or a fallback; the section shows match counts.

With a dictionary loaded, unmatched images use the **Fallback settings**:
the exact per-plane quantile (default 0.999), after values at or below the
minimum pixel count (default 0.1) are set to zero. NapariSBT's fallback maximum
of 3 is used when the computed quantile is below 5. The minimum-count threshold
also applies before dictionary scaling. Calculation runs in the TIFF worker.

Display contrast is 0–1. Manual contrast survives ROI navigation for the same
channel; **Reset** restores 0–1. Loading a new dictionary resets contrast to
avoid mixing raw and normalised units. **Remove dictionary** returns to raw
intensities and automatic contrast. The dictionary affects channel images only:
AnnData expression, observation values and segmentation IDs are unchanged, and
none of the selected files is modified or uploaded.

## Supported inputs

Masks: single-plane scalar TIFFs named exactly `<ROI>.tif` or `<ROI>.tiff`.
The image and mask dimensions must agree; the viewer does not resize or
register misaligned data.

Images: scalar TIFFs in either layout:

```text
masks/ROI_A.tiff
images/ROI_A/CD3.tiff
images/ROI_A/191Ir_CD3e.tiff

# Alternatively:
images/ROI_A_CD3.tiff
images/ROI_A_CD20.tiff
```

`images/<ROI>.tiff` is also recognized. New multipage scalar selections display
the first plane. Matching is case-sensitive for ROI names. Marker aliases use
`var_names`, `var['channel_name']` and `var['channel_label']`, including isotope
prefixes and SBT sample-prefixed filenames. Ambiguous panel aliases are not
assigned to a marker; duplicate image names retain their file paths. Duplicate
mask matches are an error.
Channels absent from AnnData use the filename stem after the last underscore,
or the full stem when there is no underscore: `44_90_Ir193_DNA3.tiff` and
`55_20_Ir193_DNA3.tiff` both become **DNA3**. This stable name retains channel
colour, visibility and contrast when moving between ROIs.

AnnData support includes numeric and string observation columns, categoricals
(including missing codes), nullable integer/boolean/string columns, `var`, dense
and CSR/CSC `X` and layers, existing two-dimensional projections from `obsm`,
and categorical `uns['<observation>_colors']` palettes using six-digit hex colours.
String labels without a categorical encoding receive a generated palette.

Uncompressed and gzip-compressed HDF5 are tested. **LZF, Blosc and other HDF5
plugin filters are not enabled in this version**: they produce an explicit
unsupported-compression error before decoding. The current upstream LZF plugin
failed to link against the HDF5 WebAssembly runtime during validation. An
uncompressed or gzip-compressed viewing copy can be prepared separately in
Python if needed; this application never rewrites the source file.

## Initial limits

- Up to 5 million AnnData rows and 16 megapixels per TIFF plane, with at most
  16384 pixels on either side. These are allocation guards, not performance
  guarantees. Full-ROI compositing and metadata still use local RAM/CPU.
  ROI and object-ID indexing spans all cells; expression and embeddings are
  read on demand for the current view. This total-row limit does not imply
  interactive rendering of five million cells in a single ROI.
- Only the current mask, selected channel planes and visible aligned layers are
  cached, with a maximum of 37 decoded raster entries (six channels per pane,
  one shared mask and 12 extra layers). Identical planes are shared across panes. An estimated 512 MB budget
  covers retained raster arrays and pane composition buffers; this is not a cap
  on total browser memory. Use fewer channels or panes if this guard is reached.
  Superseded TIFF decodes are cancelled when
  changing selections. Browser garbage collection still determines when memory
  from discarded arrays is released.
- ROI ordering runs in the HDF5 worker, scanning one chosen field in bounded
  batches. CSC matrices read one selected sparse column once. Ordering a large
  dataset can take time; it does not load the complete expression matrix.
- RGB TIFFs are supported as aligned reference layers. PNG/JPEG segmentation
  masks, TIFF stacks for extra layers, OME dimension interpretation, pyramid
  streaming, SpatialData-Zarr and arbitrary coordinate transforms are not yet
  supported. Very large images need a future tiled rendering path.
- Expression comes from `X` or `layers`; `raw.X`, nested/compound columns,
  external HDF5 links and plugin-compressed matrices are outside this version's
  supported contract.
- Embedding context uses a bounded sample, so lassoing it does not label unseen
  cells. Full-dataset analysis, model training and AnnData write-back remain
  external operations. The viewer exports proposed annotations and plots.

## Development and verification

The source and build instructions are in the repository's `web_viewer/` folder.
Tests use generated synthetic H5AD and TIFF files, exercise local file selection
in an actual browser, verify no uploads or external application requests, and
check source hashes after review. Normalisation values are checked against
reference outputs generated by the existing NapariSBT Python helper. Real dataset visual agreement and scale
benchmarks are still required before replacing an established NapariSBT setup.

## Set continuous colour limits

In **Views → Cell overlay**, choose a marker or numeric observation to reveal
**Lower (vmin)** and **Upper (vmax)**. Each bound can be a number or left blank
for automatic scaling. **Auto** clears both. Limits persist while moving between
ROIs, and each pane retains its own settings. Values beyond the limits use the
end colours; no cells or expression values are removed or changed.

The **Embedding** tab has independent colour limits. When colouring by
**Selected view**, blank bounds follow that pane's limits; explicit embedding
bounds override them only in the embedding. Categorical colours do not use limits.

**Plots** provides these controls for marker matrices, composition heatmaps and
label-comparison heatmaps. Enter limits in the displayed units (including
z-scores or percentages), then **Generate / update**. Exported SVG/PNG colour
keys use the same limits; data CSV retains the original plotted values and
records the limits in its settings. Workspaces save all of these choices.

Use the header's **How to guide** link for the walkthrough, or **Report an issue**
to open a message to Michael Haley in your email application.

## Choose legend entries and flatten annotation layers

In **Views**, each image channel, the cell overlay, and each aligned layer has
a **Show in legend** checkbox. This controls the selected pane's floating legend
and exported figure legend without changing which images or cells are displayed.
The toolbar's Legend switch still shows or hides the entire legend.

For **Categorical labels**, enable **Flatten labels on import** in Setup to
start with a single foreground label. You can also turn flattening on or off for
each pane in **Views → Aligned layers**, and choose its label name and colour.
Every non-zero image ID is included, even if it was unmapped or transparent in
the dictionary. Zero remains background; holes are retained. Outlines follow
the combined foreground region, removing boundaries between the original IDs.
Original files and label pixels are unchanged, so turning flattening off restores
the original categories and dictionary colours.

These settings follow ROI changes and are included in workspaces, bookmarks,
and saved view presets. Existing workspaces default to showing all legend
entries and keeping the original label categories.
