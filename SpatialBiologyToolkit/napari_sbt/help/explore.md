# Explore

Explore is the image and population-QC workspace. ROI navigation normally shows
only ROIs in the current experiment scope. A Feature Discovery Trial restricts
navigation to its representative ROIs so it is always obvious which cells are
being labelled and evaluated.

## ROI navigation and layer controls

Choose an ROI or use **Previous ROI** and **Next ROI**. **Load ROI** refreshes the
mask and available channels. Enable empty ROIs only after building the Setup
integrity index. The dimmed full-mask context is optional and does not change the
eligible classification cohort.

**Add all-cells mask** adds the complete original segmentation for the current ROI
as the non-editable `all_cells` labels layer. It reuses the mask already held in
memory, so pressing it does not reload the ROI or rescan the mask folder. The layer
updates in place during ROI navigation and can retain its visibility, opacity, and
contour width through an Explore recipe. It never modifies the source mask.

## Publication image export

Open **Publication export…** beside the ROI controls to prepare a canvas-only image
without changing the source images or masks. The window is modeless: keep it open
beside Napari while adjusting the tissue view, then use **Capture current viewport**
or a selected rectangle from a Shapes layer. A captured frame stores centre and
field-of-view coordinates rather than only Napari's raw zoom number, so it is
independent of monitor size and dock layout. **Entire ROI** calculates the frame
separately for each ROI. Aspect mismatches are cropped or padded; images are never
stretched.

Choose a named Explore recipe or the current live view. Saving a publication
preset freezes a complete recipe snapshot, even if the original Explore recipe is
later edited or removed. Publication presets are stored separately in
`explore/publication_export_presets.json`, so they do not participate in frequent
layer-visibility and ROI-review updates.

Choose one coordinated **Resolution** rather than managing raster size, DPI, and
annotation sizes separately. **Low** exports one raster pixel per source pixel at
150 DPI, **Medium** (recommended) exports at 2x linear resolution and 300 DPI, and
**High** exports at 4x linear resolution and 600 DPI. All three preserve exactly
the same centre, field of view, aspect ratio, and approximate printed dimensions.
Scale-bar thickness, text, margins, ticks, annotation text, and annotation boxes
grow with the selected resolution, so the composition keeps the same visual
proportions. Relative size controls then multiply that automatic baseline: scale
the scale-bar label, bar/ticks, margin, and box padding independently, and set
separate percentages for custom-title, ROI-name, and channel-name text. Higher
levels provide denser pixels for large figures; they cannot
create detail absent from the source images. Existing presets that used the old
manual controls open as **Custom — existing saved preset** and retain their exact
settings until you choose a new resolution. PNG is the recommended default, TIFF
is lossless, and JPEG is explicitly lossy. Filenames always retain the ROI and
channel identities, even when a custom template omits them. The default is
`{roi}__{recipe}__{channels}__{width}x{height}`.

Physical scale bars require an explicitly verified X/Y pixel calibration. Select
an automatic visually sensible length or a fixed physical length, then control
position, colour, ticks, and background box. Untick **Show physical-length text**
to export the bar and optional end ticks without a numerical label. Thickness,
font size, margin, tick size, and box padding are sized automatically from the
chosen resolution before their relative percentages are applied. Napari's
metadata detector reads OME physical-size metadata or calibrated TIFF resolution
tags from one current image, but leaves the result unconfirmed until you review it.
Napari's ordinary scale-bar overlay is hidden during capture; NapariSBT calculates bar
length from the frozen source-pixel field and verified physical size per pixel,
then composites it onto the final raster. It is therefore independent of output
DPI and remains physically correct at every resolution. Use
**Render preview** to inspect the exact final raster
before saving. ROI, channel, and custom-title annotations are optional. Their
text colour, translucent background colour, box visibility, margin, padding, and
individual relative sizes can be adjusted. **Match each channel name to its image
colour** uses the exact colormaps in the frozen Explore recipe (or its RGB/
six-colour roles for older recipes), so bulk exports remain reproducible even
when another ROI is currently displayed.

For bulk export, select one or more ROIs and run preflight. Preflight uses the fast
Setup asset index and never rescans image folders. It reports missing masks,
requested channels, and target filenames. Rendering then runs sequentially on the
GUI thread because Napari/OpenGL canvases cannot safely be moved into worker
processes. The progress display reports exported, resumed, and failed ROIs;
cancellation finishes the current atomic file safely. Matching sidecars permit
resume, while version and overwrite policies are also available.

Every raster receives a `.json` sidecar containing its frozen recipe, frame,
calibration, scale-bar settings, input file identities, software versions, and
fingerprints. Bulk folders also contain the frozen preset, a CSV result manifest,
and run provenance. The original ROI, recipe, camera, auto-reload setting, and
Napari scale bar are restored when the batch finishes or is cancelled. Automated
exports are not counted as manually viewed ROIs.

**Reapply the current Explore recipe after changing ROI** has one purpose: after
the selected ROI identity genuinely changes, it recreates the same image channels,
colours, contrast, overlays, visibility, and opacity for the new ROI. This makes
side-by-side tissue review consistent. Ticking or unticking the option does not
reload the current ROI, and changing a layer's colour, contrast, visibility, or
opacity does not trigger ROI loading. Qt display-only updates to the ROI selector,
such as its green/amber review colours, are ignored. Turn the option off when only
the base mask or classification layers are required. **Load ROI**, Population QC
view buttons, named-recipe loading, and F-key recipes remain explicit user actions
and can still refresh or replay the current view.

Data exploration and Population QC do not construct classification-cohort or
excluded-context label arrays. Classification and Labeler workflows still create
them because cell eligibility is part of those interactions.

## ROI sample metadata

NapariSBT automatically identifies `adata.obs` fields whose value is constant
within every ROI, excluding the ROI and object-identity columns themselves. These
fields are treated as sample-level metadata and shown for the current ROI near the
top of Explore. Categorical, text, Boolean, integer, floating-point, date/time,
duration, and missing scalar values are formatted appropriately.

A field is omitted if it varies within any ROI, contains a mixture of missing and
non-missing values within one ROI, is entirely missing, or stores unsupported
non-scalar objects. Detection is cached for the loaded AnnData and is not repeated
on ordinary Previous/Next navigation.

## Cell properties dock

The separate **NapariSBT Cell properties** dock passively follows left-clicks in
the tissue and shows the chosen `adata.obs` values for the matching
`(ROI, ObjectNumber)` cell. Categorical values use colours from
`adata.uns["<observation>_colors"]` when available, with the same stable fallback
palette used by Explore overlays. ROI-level metadata and the identity columns are
never offered as cell properties because they are already represented elsewhere.

Use **Settings** to choose the tracked observations, pause tracking, or add a
configurable non-editable outline around the inspected cell. The inspector has
its own transient selection state and layer: it does not change the classifier or
Labeler selection, click mode, proposed/confirmed labels, or active Napari layer.
Settings are retained in the experiment's Explore review state.

**Hide all layers** and **Show all layers** change visibility without discarding
the view. **Delete all layers** removes the displayed layers; saved reload-recipe
entries can still reconstruct managed layers on the next ROI.

## Image channels

Load channels as greyscale, RGB, or the automatic red/green/blue/cyan/yellow/
magenta sequence. All image layers start at opacity 1.0. AnnData categorical or
numeric observations, individual populations, and `adata.X` marker values can be
rendered as cell overlays. Existing categorical colours are recovered from
`adata.uns` whenever possible.

**Variable list order** is shared across NapariSBT. **AnnData order** follows
`adata.var_names` and is the default; **Alphabetical** ignores case; **Expression
similarity** uses the same hierarchical expression ordering as the matrix-plot
option. The similarity order is calculated once from the live `adata.X` and
cached. Changing this control in Explore immediately updates the matching control
and variable lists in Feature Building, Population QC, Scanpy plotting, and
Dataset Maintenance without reloading the ROI or viewer layers. Image-only
channels which cannot be matched safely to AnnData remain at the end.

Select one or more discovered channels. Use greyscale for independent contrast
control, R/G/B/C/Y/M for additive multiplex review, or RGB for one composite layer
from the first three selected channels. Image coverage reports which configured
folder supplied each ROI/channel match.

**Select feature markers** replaces the current selection with the
channel-derived markers recorded in the active built feature table. If features
have not been built yet, it uses the current synthetic-feature recipe instead; a
blank channel recipe means every compatible channel. The same shortcut is
available for the `adata.X` marker-overlay list. Shape, context, imported-table,
and embedding dimensions are not mistaken for staining markers, and markers that
are absent from the current ROI or expression source are reported rather than
guessed.

**Rank ROIs by selected marker** requires exactly one selected cell-level marker
and orders the ROI selector by descending mean `adata.X` expression. These are
cell-level values quantified inside segmented cells; raw image background and
extracellular pixels are never included. It uses mean cell expression so large
ROIs do not rank highly merely because they contain more cells. The **Overlay
scope** control determines whether ranking uses only the active workflow cell
scope or every matched cell in the AnnData object. Feature Discovery Trial ROI
restrictions are retained.

## AnnData and population overlays

Observation overlays are cohort-restricted by default. Select **Include cells
outside the classification cohort** when reviewing a curated observation
alongside every identity-matched cell in the ROI.

Choose an observation and render it as a categorical or numeric cell overlay.
Categorical overlays keep every cell's original mask ID and use the observation
only to assign its colour. A non-zero contour therefore separates touching cells
even when they belong to the same category; the source mask and AnnData are not
changed. Saved recipes retain category-level colours rather than ROI-specific
object IDs, so the same palette remains valid on another ROI.
Population controls can add selected populations as separate layers, rank ROIs by
abundance, or send one population back to Setup as a proposed cohort. Marker
overlays use cell-level values from `adata.X`, not image pixels.

Population QC can save and restore the exact verification view used for a selected
population. This is useful when several channel combinations are needed for
different cell types.

## ROI reload recipe

The active reload recipe records which images and overlays should reappear after
moving to another ROI, together with colours, visibility, opacity, contours, and
contrast limits. Several recipes can be stored under descriptive names—for
example, `T-cell verification`, `Macrophage review`, or `Segmentation QC`.

Named recipes and their viewed-ROI fingerprints are written immediately to
`explore/review_state.json` inside the workspace, so they return when the
experiment is reopened. **Export selected recipe JSON** creates a portable copy;
**Import recipe JSON** adds one from another workspace. Conflicting names receive
an imported suffix, conflicting IDs are replaced, and a conflicting F-key is left
unassigned rather than overwriting another shortcut.

Build the desired view, enter a name, optionally choose an unused F1–F12 shortcut,
then click **Save current view as new recipe**. Choose an existing recipe and use
**Update selected recipe from current view** to replace its saved layer definition,
name, or shortcut. Duplicate names and duplicate F-key assignments are rejected.
Deleting a recipe asks for confirmation, removes its shortcut, and leaves the
currently displayed Napari layers intact.

Use the recipe selector and **Load selected recipe**, or press its assigned F-key
while the Napari viewer has keyboard focus. F-key switching immediately makes the
recipe active and applies its layers for the current ROI without changing the
currently selected workflow tab. Layers whose ROI and semantic data source already
match are reused in place: only changed colour, visibility, opacity, contour, and
contrast settings are applied. Layers removed by another recipe can be restored
from a bounded cross-ROI memory cache instead of reading or calculating them
again. The least-recently-used cache holds at most 48 arrays or 512 MiB and is
cleared when source, normalization, or live AnnData inputs changeâ€”not on every ROI
switch. Image identity includes
the source path, size, and modification time, so a changed image is still reloaded.
Missing, changed, or genuinely different data are loaded or recalculated normally. The
active-status line distinguishes a saved recipe from a modified working view; a
modified view is not written back until **Update selected recipe from current
view** is clicked.

Unavailable layers are never pruned merely because the current workflow, AnnData,
or ROI cannot reconstruct them. They remain in the recipe list with an orange
warning and are retried when their source becomes available. This permits a
classification recipe to be inspected in Data exploration mode without losing
confirmed, proposed, prediction, or focus-mask display settings. Use **Delete
selected recipe items** when an absent entry should genuinely be removed.

Scalar channel images are normalized onto a 0â€“1 display scale. When a recipe
restores their saved contrast handles, the Napari slider itself remains fixed at
0â€“1. This lets you estimate the saved minimum and maximum visually without using
Napari's reset action, which would replace the recipe's handles.

**Update from current layers** captures manual display changes into the working
view. Colormap changes made in Napari are also tracked immediately, and saving or
updating a recipe takes a final snapshot of the actual continuous or label colours
visible in the layers rather than reverting to their original loading palette.
Delete selected recipe entries to stop recreating them. Reviewed-ROI
colouring applies to the complete current recipe fingerprint, so each named view
retains its own effective reviewed-ROI context. Green and amber dropdown entries
use an explicit dark foreground so their ROI names remain readable in Napari's
dark theme.

The **Live recipe tracking** toggle is available in Setup, Explore, and Population
QC; all three controls change the same session setting. It can be switched at any
time and is not frozen into the workspace. It defaults on for Data exploration
and off for Population QC. Explicit recipe actions and Population QC RGB controls
continue to work while tracking is disabled.

A simple visibility or opacity change uses an in-memory fast path. It does not
reload the ROI, reread image data, or rebuild classification layers. The recipe
value is updated immediately without rescanning the whole AnnData population
column. Opacity recipe-row text and reviewed-ROI fingerprints wait for the next
normal Explore refresh so continuous slider events remain cheap. Napari itself
may still need to repaint the layer as its alpha changes.

Visibility and opacity are treated as the hottest paths for full-size cohort,
context, focus, confirmed, proposed, predicted, and uncertainty layers. Their
recipe values are updated immediately, while named-recipe and reviewed-ROI
fingerprint displays are deferred until Explore is entered or the recipe/ROI is
otherwise refreshed. This keeps an eye-button click or opacity-slider drag from
hashing every saved colour specification.

Treat the list as an exact preview of what automatic ROI navigation will recreate
after a genuine change to a different ROI. It is not continuously reapplied while
editing layers in the current ROI.
Classifier layers are regenerated from current labels and scores, while their
visibility, opacity, contour settings, and probability display choices are replayed.
After manually changing layers, click **Update from current layers** before moving
ROIs. Then update the selected named recipe if those changes should persist beyond
the working view. Remove entries that should no longer follow navigation.
