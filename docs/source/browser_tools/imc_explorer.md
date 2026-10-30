(tissue-explorer)=
# IMC Explorer

[Open IMC Explorer](https://imc.spatialbiology.app/)

The application now uses `imc.spatialbiology.app`; the previous address redirects
here. Browser-saved workspaces and folder permissions do not transfer between
addresses. You can still import previously exported workspace JSON and reconnect
the original files.

Use IMC Explorer to review analysed tissue images alongside segmented cells
and their AnnData measurements. You can compare channel combinations, inspect
populations and embeddings, propose annotation changes, and export figures.
Your selected files stay on your computer and are not rewritten by the viewer.

## Prepare your data

You normally need:

| Input | What it provides |
| --- | --- |
| One `.h5ad` file | Cell observations, ROI and object identities, expression values and any existing embeddings |
| A mask folder | One labelled, single-plane TIFF per ROI; zero is background and positive integers identify cells |
| An image folder | Scalar TIFF channel images, organised by ROI |

For example:

```text
my_review/
  cells.h5ad
  masks/
    ROI_A.tiff
    ROI_B.tiff
  images/
    ROI_A/
      CD3.tiff
      CD20.tiff
    ROI_B/
      CD3.tiff
      CD20.tiff
```

Flat image names such as `ROI_A_CD3.tiff` and `ROI_A_CD20.tiff` are also
supported. ROI names are case-sensitive. Images and masks must already have
matching dimensions and alignment: the viewer does not register or resize
misaligned data.

In AnnData, the default identity columns are **ROI** and **ObjectNumber**;
you can choose other columns explicitly. Each `(ROI, object ID)` pair must be
unique, and object IDs must be positive integers that match the mask labels.
An object number alone is not a unique identity across different ROIs.

AnnData can be opened before selecting images or masks. Review tools become
available, but tissue context requires the corresponding assets.

## Your first tissue view

1. Open the app in desktop Chrome or Edge and choose **Open Setup** on the
   welcome screen.
2. In **Setup**, select the **AnnData** file, **Masks** folder and **Images**
   folder. Their selection order does not matter.
3. Check **Cell identity columns**, then choose an ROI in **ROIs**. Resolve any
   missing or duplicate identity errors before interpreting the overlays.
4. In **Views → Image channels**, choose a few markers and their colours.
   Choose **Auto** for an initial contrast range, then adjust low/high bounds.
5. In **Cell overlay → Colour by**, choose an observation, such as a population
   label, or a marker expression value. Choose filled cells, outlines or a
   hidden overlay according to the question you are reviewing.
6. Drag to pan, scroll to zoom, and use **Fit view** to see the full region.
   Click a segmented cell to see its observations in **Cell details**.

The loading indicator identifies the current stage or view. Previous images
are dimmed while a new ROI is being prepared. A grey mask object can be tissue
context without a matching AnnData row; it must not be interpreted as a cell
with zero expression or assigned another cell's metadata.

## Find your way around

| Area | Main tasks |
| --- | --- |
| **Setup** | Open data, manage workspaces, connect assets, add aligned layers and image normalisation |
| **ROIs** | Navigate, filter and sort regions; track which view recipes you have visited |
| **Views** | Choose the active pane, channels, cell overlays and additional layer appearance |
| Bottom **Review tools** | Cell details, embeddings, population remapping, plots, flagging, bookmarks and figure export |

Collapse individual sidebar sections when you do not need them. Hiding image
channels or a cell overlay is separate from collapsing its controls: the eye
or visibility control changes the display while retaining the settings.
The bottom panel can be folded away or resized by dragging its divider.

## Channels, expression and colour limits

Each pane supports up to **six image channels**. Add or remove slots, change
their colours, or hide a slot without losing its marker and contrast settings.
Image channels are TIFF intensities; cell expression overlays come from the
selected AnnData **X** or **layers** source. The two need not have identical
units or preprocessing.

Without an image normalisation dictionary, channel bounds use the original
TIFF intensity units. **Auto** uses the sampled minimum and 99.5th percentile.
Contrast is retained between ROIs until you change it or request Auto again.
If a selected channel is absent in an ROI, it is reported as missing rather
than replaced by another marker.

For a marker or numeric cell observation, choose a colour map and optional
reversal. **Lower (vmin)** and **Upper (vmax)** accept exact values; leave them
blank for automatic scaling or use **Auto** to clear both. Values outside the
range clip to the endpoint colours. Missing values remain grey.

**Show populations** filters the cell overlay. Use Ctrl/Cmd-click for several
categories, Shift-click for a range, and **Show all** to remove the filter.
Filters and colour limits change the view, not the AnnData values.

### Use an image normalisation dictionary

In **Setup → Image normalisation**, load a Nimbus CSV or a compatible
NapariSBT JSON. A typical CSV is:

```text
marker,vmax,lower_threshold
CD3,120,2
CD20,80,0
```

The displayed channel intensity is clipped to 0–1 after subtracting the lower
threshold and dividing by the range to `vmax`. Channel controls report whether
they matched the dictionary or use fallback settings. Resolve ambiguous marker
matches rather than assuming the intended marker was chosen.

With a dictionary, contrast controls use 0–1 and **Reset** restores that range.
Loading a replacement dictionary resets contrast to avoid mixing units.
**Remove dictionary** returns to raw intensity display. The dictionary affects
channel images only, not AnnData expression, cell IDs or source files.

## Compare views and review ROIs

Use **Views → Add view** to duplicate the selected pane, up to four panes.
Name them for the comparison, for example “Immune markers” and “Population
labels”. Each pane has its own channel mix, colours, overlay and aligned-layer
settings. Click its image/name or use **Selected view** to make it active.
Pan and zoom remain linked across the panes.

In **ROIs**, sort by name, cell count, a population count, a numeric observation
or a marker mean. Choose the direction and **Apply order**. Navigation follows
the resulting order while retaining each pane's display recipe.

Use **Filtering → Read metadata** to find observations constant within each
ROI, such as sample or condition. Select values to include: alternatives
within a field use OR, while separate fields intersect. If no ROI matches,
reset or remove filters. After reopening a workspace, saved filters remain
pending until you read metadata again.

Visit colours describe the current set of view recipes: green is unvisited,
amber means some recipes have been visited, and grey means all have. Changing
channels, contrast or overlays creates a new recipe. A visit records display,
**not scientific approval or completion of QC**.

Save a **view preset** to reuse one pane's settings on another ROI. Use a
**bookmark** to capture the complete viewer state, including ROI, panes,
camera and display settings. Rename bookmarks, add notes and use **Return**
to revisit them. Neither copies the underlying images or expression data.

## Inspect embeddings and select cells

Open **Embedding** in the bottom panel and choose an existing projection from
AnnData. No embedding is recalculated. The plot can follow the selected
tissue pane's colours or use an independent observation/marker and limits.
Click a point to inspect the corresponding cell in tissue.

Enable **Other ROIs** for faint context from the rest of the dataset. The
status reports the context sample and its scope. Context is sampled when it
exceeds the selected limit; lassoing it selects only plotted cells, not every
unseen cell in that area of the complete embedding.

The mouse-mode menu offers **Inspect & pan**, **Select cells** and **Lasso**.
Selection is linked between tissue and embedding by ROI and object ID.
**Select displayed** uses the relevant displayed tissue/embedding scope.
**Esc** returns to Inspect; **Clear** removes the selection. The selection
itself is temporary. To retain a review annotation, apply it to a flag field.

## Review population labels and flag cells

### Propose population names or merges

1. Open **Population remapping**, choose an existing observation and select
   **Create remap**.
2. Edit proposed names, colours and notes. Give several source populations the
   same proposed name to merge them; merged entries share a colour.
3. Choose **Use in selected view** to preview the proposal. Keep another pane
   on the original observation to compare them.
4. Export the remap as CSV when you want to share or apply the proposal in
   downstream analysis.

The proposal autosaves in the workspace but does not write a new AnnData
column. Remap CSVs retain one row per original value, including members of a
merge. `source_value_json` preserves exact source types; the displayed
`source_value` alone can lose distinctions between numeric and text labels.
This CSV is a review schema, not a ready-to-run pipeline remap configuration.

### Flag individual cells

In **Flagging**, create a named field and categories, select cells, then choose
**Apply to selected cells**. **Unassign selected** removes assignments and
**Undo** reverses recent edits. **Use in selected view** previews the field;
unassigned cells remain grey.

Assignments persist in the workspace. **Export CSV** includes assigned cells
with ROI, object ID, field, label and colour. Join this output using **both ROI
and object ID**; unassigned cells are absent. Flagging does not edit the source
AnnData and is distinct from a temporary on-screen selection.

## Plot populations and markers

In **Plots**, choose the plot type, observation or proposed remap, and scope
(current ROI, selected ROIs or all cells/ROIs), then **Generate / update**.

| Plot | What it summarises |
| --- | --- |
| Marker matrix | Population mean expression for selected markers |
| Population composition bars or heatmap | Counts or percentages within sample/metadata groups |
| Compare labels | A cross-tabulation of two observations or proposed remaps |
| Categorical alluvial plot | Cell counts connecting two sets of labels |

Population and metadata filters intersect. Composition percentages are
calculated **after filtering**. Marker means include sparse zeros and exclude
non-finite values; optional z-score or 0–1 scaling affects the displayed
population means. These plots use the selected cells rather than the embedding
context sample.

Plots are snapshots. If the controls change, regenerate before interpreting
the new settings. **SVG**, **PNG** and **Data CSV** export the displayed result;
the CSV includes its calculation settings. Workspace JSON stores plot settings,
so choose Generate / update again after reopening.

For a single selected population, **Representative markers** ranks positively
enriched markers and shows expression distributions. Check whether the scope
is the current ROI or the whole AnnData and which expression source is used.
These are **descriptive rankings, not p-values or formal differential-expression
tests**. R/G/B controls can assign matching images to the selected pane.

## Add aligned reference or annotation images

In **Setup → Aligned layers**, add a named folder and choose its type:

- **Colour / reference image:** aligned H&E or another RGB image.
- **Scalar image:** a single-plane intensity or probability TIFF.
- **Categorical labels:** an integer label TIFF, optionally with a dictionary
  such as `value,label,color` CSV.
- **Additional IMC image folder:** another source of channel TIFFs using the
  usual image-folder layout; these share the pane's six channel slots.

Reference, scalar and label files must be named for their ROI and already
match the primary image/mask dimensions and alignment. In **Views → Aligned
layers**, adjust visibility, opacity, blending and the type-specific controls.
Setup's layer ordering is shared; later entries draw above earlier ones.

For categorical layers, **Flatten labels** displays all nonzero IDs as a single
foreground label, retaining holes and leaving zero as background. Turning it
off restores the original categories. **Show in legend** can omit an entry
from the legend without hiding the layer itself. Neither operation rewrites
label pixels.

## Export tissue figures

1. Choose **Export** and switch to **Figure** mode.
2. Set a per-pane aspect ratio, output width and spacing. Pan/zoom to frame the
   tissue; returning to Viewer restores the previous camera.
3. Configure view names, legend entries, backgrounds, text sizes and which
   panes have scale bars.
4. Check **View → Scale bar settings**. Calibration defaults to **1 image
   pixel = 1 µm**; change it to the correct acquisition pixel size.
5. If needed, enable **Lock field of view in µm** and set the physical width
   and height. This locks zoom while allowing pan.
6. Choose filename components and export PNG, JPEG or layered SVG.

Scale bars are only meaningful with the correct calibration. PNG/JPEG retain
the composed appearance. SVG contains separate raster image/annotation/cell
layers with editable vector labels, legends and scale bars; the cells and
images themselves remain rasterised. Selection highlights are excluded.

**Automatically bookmark each export** captures the viewer state for returning
to the composition. Filename metadata fields require **Read ROI metadata**
after reopening; the viewer does not silently perform that scan. Exports are
limited to 32 million pixels and 16,384 pixels per side.

## Save and resume your review

The **Workspaces** library in Setup autosaves named workspaces locally. Wait
for **Saved locally** before closing the page. Workspaces include remaps,
flag assignments, notes, panes, bookmarks, presets, filters and figure settings.

1. Use **Connect folder** to associate the workspace with a common root
   containing its original data and annotation assets, where supported.
2. On a later visit, choose **Resume a workspace** or open its entry in Setup.
3. If access needs renewing, choose the data folder again. Resolve items marked
   **Needs attention** in **Assets & connection** using Locate, Edit path or
   Retry. Paths are relative to the connected root.
4. Use **Import / export → Export JSON** to make a portable backup. To move
   computers or browsers, import the JSON and reconnect the matching assets.

**Full asset check** verifies the saved asset folders. **Quick reconnect**
reuses cached file lists from successful checks in this browser; use a full
check after changing files or when validating a shared dataset. These are
metadata checks, not content checksums.

**Duplicate** creates an independent review; **+ New** starts another workspace.
Deleting a workspace removes its saved review annotations, not the source
files. There are no recovery snapshots. Browser-local saves are not a backup:
clearing site storage can remove them. JSON includes review settings and
annotations, but not image pixels, expression matrices or file permissions.

## Supported inputs and limits

- AnnData supports observations, categorical palettes, dense/CSR/CSC `X` and
  layers, and existing projections. `raw.X` is not an expression source here.
- Use uncompressed or gzip-compressed HDF5. LZF, Blosc and other HDF5 plugin
  filters are unsupported; ask the data provider for a compatible viewing copy
  if the viewer reports unsupported compression.
- Primary masks must be integer, single-plane TIFFs. Primary channels are
  scalar TIFFs; new multipage selections use their first plane. RGB images
  belong in aligned reference layers, not the primary mask input.
- SpatialData Zarr belongs in [MERSCOPE Explorer](merscope_explorer.md), not
  IMC Explorer. Raw MCD belongs in [MCD Explorer](mcd_explorer.md).
- Allocation guards include five million AnnData rows, 16 megapixels per TIFF
  plane and 16,384 pixels per side. These are not performance guarantees.
  Fewer channels, panes or aligned layers can reduce memory use.
- The tool does not perform segmentation, registration, new embedding
  computation or AnnData write-back. Proposed labels must be applied separately
  in your analysis workflow.

For loading, reconnection and network problems, see
[Troubleshooting](troubleshooting.md).
