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

Data exploration and Population QC do not construct classification-cohort or
excluded-context label arrays. Classification and Labeler workflows still create
them because cell eligibility is part of those interactions.

**Hide all layers** and **Show all layers** change visibility without discarding
the view. **Delete all layers** removes the displayed layers; saved reload-recipe
entries can still reconstruct managed layers on the next ROI.

## Image channels

Load channels as greyscale, RGB, or the automatic red/green/blue/cyan/yellow/
magenta sequence. All image layers start at opacity 1.0. AnnData categorical or
numeric observations, individual populations, and `adata.X` marker values can be
rendered as cell overlays. Existing categorical colours are recovered from
`adata.uns` whenever possible.

Select one or more discovered channels. Use greyscale for independent contrast
control, R/G/B/C/Y/M for additive multiplex review, or RGB for one composite layer
from the first three selected channels. Image coverage reports which configured
folder supplied each ROI/channel match.

## AnnData and population overlays

Observation overlays are cohort-restricted by default. Select **Include cells
outside the classification cohort** when reviewing a curated observation
alongside every identity-matched cell in the ROI.

Choose an observation and render it as a categorical or numeric cell overlay.
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
retains its own effective reviewed-ROI context.

The Setup **Live recipe tracking** toggle controls automatic capture of manual
layer changes. It defaults on for Data exploration and off for Population QC.
Explicit recipe actions and Population QC RGB controls continue to work while
tracking is disabled.

Treat the list as an exact preview of what automatic ROI navigation will recreate.
Classifier layers are regenerated from current labels and scores, while their
visibility, opacity, contour settings, and probability display choices are replayed.
After manually changing layers, click **Update from current layers** before moving
ROIs. Then update the selected named recipe if those changes should persist beyond
the working view. Remove entries that should no longer follow navigation.
