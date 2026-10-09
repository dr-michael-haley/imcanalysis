# Scanpy plotting

Scanpy plotting creates quick, regenerable views from the AnnData object loaded in
NapariSBT. It is designed for checking population identities, expression patterns,
sample representation, renames, merges, and subclusters while moving between
Population naming, Explore, and Population QC.

Plots are read-only. Opening a plot does not normalize or scale the AnnData,
recompute PCA, rebuild neighbours, change BioBatchNet output, calculate UMAP, or
run clustering. Every plot window is a snapshot of the labels and values that
existed when **Open in a new resizable window** was pressed.

## Draw and label embedding regions

Open an **Embedding** plot, then click **Annotate regions…** in its window.
The annotation popup uses the same embedding components. The plot's filtered
cell scope remains the outer limit for annotation; other cells provide grey
context. You can narrow the annotation cohort further in the popup.

1. Choose an **Output obs column** (default: `manual_population`).
2. Choose **Inherit labels from** to copy a parent population column into the
   draft. NapariSBT preselects the plotted population column. Unannotated cells
   keep their parent labels. Choose **Unassigned (no parent)** for a fresh column.
3. Under **Cohort to annotate**, choose **obs values**, **obs range**, or **X
   range**, select the field and values/bounds, then click **Add filter**. All
   filters must match. Cohort cells are coloured; other cells stay grey and cannot
   be selected. **Remove selected filter** and **Clear filters** preserve your
   existing draft annotations.
4. Choose **Lasso** and drag around cells, or **Polygon** and click vertices,
   clicking the first vertex again to finish. Turn off toolbar pan/zoom to draw.
5. Enter a category name and click **Assign label**. Optionally enable **Prefix
   new labels with parent**: assigning `Activated` inside `T cells` produces
   `T cells / Activated`. This uses each cell's original parent label, including
   when a selection spans several parents. Repeat for other categories.
   Later assignments replace earlier ones where regions overlap. **Undo
   assignment** reverses the last assignment; **Clear selection** clears the
   current outline without removing assigned labels.
6. Click **Apply to AnnData** to write a categorical observation and its colours.
   An existing column requires the explicit overwrite checkbox. The new column
   becomes available in NapariSBT's observation selectors.
7. Use **Save H5AD copy…** to persist the applied labels in a new file. The source
   H5AD is not automatically changed.

For example, add `obs[population] = T cells`, `obs[sample] = A`, and
`X[CD3] >= 2` to split only CD3-high T cells from sample A. Values within one
**obs values** filter use OR; separate filters use AND. Range limits are inclusive
and either bound can be left blank. Missing/nonfinite numeric values do not match
ranges; missing observation values can be selected explicitly.

**X range** always reads stored `adata.X` values without transforming them,
independently of any `layer` or `raw` setting used for plot colouring.

The display can be sampled, but the drawn region selects **all cells in the
cohort**, including undisplayed cells. Sampling reserves space for the cohort so
small populations remain visible. The popup shows coloured, grey, and selected counts.
Cells with missing embedding coordinates cannot be selected. Region annotations
are manual labels; the tool does not calculate clusters.

The same popup is available without Napari, masks, or images:

```bash
sbt gui annotate --anndata processed.h5ad --basis X_umap --color leiden
```

From Python or a notebook (the notebook opens a separate desktop window):

```python
from SpatialBiologyToolkit.annotation import CohortFilter, annotate_embedding

window = annotate_embedding(
    adata, basis="X_umap", color="leiden", source_obs="leiden",
    key_added="manual_population",
    filters=[
        CohortFilter("obs", "leiden", values=("3",)),
        CohortFilter("X", "CD3", minimum=2),
    ],
)
```

Optional Python arguments include `components=(0, 1)` (zero-based),
`source_obs="leiden"`, `obs_names=[...]` to restrict selectable cells,
`layer="scaled"` or `use_raw=True` for marker colouring, and `point_limit=50000`.
Use another existing `obsm` key for PCA, t-SNE, or a custom embedding.
Scripts wait for the popup to close; notebooks keep their Qt event loop active.

### Save regions and replay without a window

Click **Save recipe…** to save a JSON file, or call
`window.save_recipe("regions.json")`. Saving a recipe does not require applying
the draft to AnnData. It records the embedding key/components, output column,
parent labels, optional cell-name scope, display settings, and each assigned
region's vertices, label, parent-prefix choice, and obs/X filters. Undo removes
the last assignment from the recipe; changing the parent column starts a new
recipe. A drawn but unassigned shape is saved for reference only.

```python
from SpatialBiologyToolkit.annotation import apply_annotation_recipe

apply_annotation_recipe(adata, "regions.json")
# Optional output name or permission to replace an existing obs column:
# apply_annotation_recipe(adata, "regions.json", key_added="split", overwrite=True)
```

This updates the live AnnData without importing Qt or opening a window. Save the
AnnData separately if needed. Assigned regions run in order with their original
filters; the last matching assignment wins. Filters use the supplied object's
obs/X values, and parent labels come from its saved source column. Regions
matching no cells are skipped. Display sampling and colours do not limit replay.

Use the same embedding coordinate system. Recomputing UMAP can move cells into
different regions. Recipes with an explicit cell-name scope require those cells
to be present; recipes without one evaluate all cells in the supplied object.

For a new H5AD copy from the command line:

```bash
sbt gui annotate --anndata processed.h5ad --recipe regions.json --output annotated.h5ad
```

This uses the settings saved in the recipe and requires no desktop display.
`--overwrite-obs` permits replacing its output obs column; the output H5AD must
be a new file.
Use `block=False` when embedding the popup in an already running Qt application.
This reuses Matplotlib and Qt from the SBT environment; no inline widget backend
or `ipympl` is required. Run on a desktop or an allocated interactive session with
a display, rather than a headless/login-node analysis job.

## Choose data and cells

**Labels / populations** chooses the `adata.obs` column used to group and colour
the plot. Saved Population naming columns and labels applied by other NapariSBT
tools appear here as soon as the live AnnData selectors are refreshed. Categorical
order is preserved, and colours are taken from Scanpy's conventional
`adata.uns["<obs>_colors"]` palette whenever it is available.

**Cell scope** controls which observations contribute:

- **All cells** uses the whole AnnData, subject to any ROI selection.
- **Current classification cohort** uses only the identity-frozen cohort from the
  active experiment. It is unavailable when no cohort exists.
- **Only selected populations** uses the rows selected in the population list.

The population list does not filter **All cells** or **Current classification
cohort**. Use **Select all** and **Clear selection** when preparing the explicit
selected-population scope.

The ROI list is an optional second filter. Leave every ROI unselected to use all
available ROIs, or select one or more ROIs to restrict the plot. **Clear ROI
selection** removes this filter and returns to all ROIs. The identity and row order
of AnnData are never changed.

**Additional ROI/sample observation** provides a further metadata filter. It lists
observations that are constant within each ROI, such as patient, condition,
treatment, tissue, or acquisition batch. Select one or more values to plot only
those groups; leave the value list empty to include them all. The active filter is
included in automatic plot titles. This filter is independent of the composition
plot's **Group bars/columns by** setting, so—for example—you can restrict a plot to
one treatment while still comparing its individual ROIs.

**Expression matrix** chooses `adata.X`, `adata.raw`, or one of the matrices in
`adata.layers`. Marker choices update to match the selected source. Embedding,
composition, and label-comparison plots do not transform this matrix; the setting
is retained so moving to an expression plot remains predictable.

## Choose a plot

Choose the biological question first; box 3 then shows only the relevant controls.

- **Embedding** uses `scanpy.pl.embedding` to show populations on an existing
  matrix in `adata.obsm`, such as `X_umap`. It never calculates a new embedding.
- **Marker matrix plot** uses `scanpy.pl.matrixplot` to show population mean
  expression, optionally scaled within each marker.
- **Marker dot plot** uses `scanpy.pl.dotplot`: colour shows the mean and dot area
  shows the fraction of cells above the chosen positivity threshold.
- **Stacked violin** uses `scanpy.pl.stacked_violin` to show marker-value
  distributions within each population.
- **Population composition** compares counts or within-sample percentages across
  ROIs, patients, conditions, or another observation.
- **Compare two label columns** cross-tabulates an original observation against
  the primary labels. It is especially useful for checking Leiden renames, merges,
  and subclusters.

## Plot options

The **Variable list order** in box 1 controls both searchable expression-variable
lists and is shared with Feature Building, Explore, Population QC, and Dataset
Maintenance. **Expression similarity** here is a cached whole-`adata.X` browsing
order. It is distinct from the plot-specific **Marker ordering** option below,
which recalculates similarity from only the cells, matrix source, and markers in
the requested plot.

For embeddings, select any `obsm` matrix with at least two components and choose
the horizontal and vertical component numbers. **Colour by** can show the selected
population labels or expression values. In expression mode, choose one or more
variables from the searchable list, set the number of panel columns, and choose a
colour map. Values come from the expression matrix selected in box 1; the source
embedding and AnnData are not recalculated. The exported plot-data CSV contains
the displayed value for every selected variable.

Interactive previews default to a maximum of 50,000 points. Larger selections are
deterministically and population-stratified downsampled so repeated plots are
stable and small populations remain represented. Change the limit when a fuller
view is needed. Point size and opacity affect display only.

Expression plots use only the markers selected in the searchable list. **Select
visible** makes it easy to type a marker family and select the filtered results.
**Select feature markers** replaces that selection with markers represented by
channel-derived features in the active NapariSBT feature table. The same action is
available for expression-coloured embedding panels. It selects only markers that
exist in the chosen expression matrix and reports unavailable feature markers.
Heat maps and dot plots support at most 100 markers; distribution plots support at
most 12 to keep the result readable. Marker-wise z-scores compare populations
within each marker, marker-wise 0–1 scaling shows relative ranges, and unscaled
means retain the values from the selected matrix. These colour-scaling choices
apply to matrix and dot plots. Stacked violins always show the stored values from
the selected expression matrix so their distributions remain interpretable.

The dot-plot positivity threshold uses the stored matrix values: a cell is positive
when its value is strictly greater than the threshold. NapariSBT does not infer an
assay-specific biological cutoff.

Native expression plots also provide common Scanpy display options:

- **Colour map** selects a familiar Matplotlib/Scanpy palette. **Automatic** uses a
  diverging palette for marker-wise z-scores and viridis otherwise.
- **Side annotation** can be empty, a fresh dendrogram, or population cell totals.
  Scanpy only supports one of these side annotations at a time.
- **Fresh dendrogram** always recalculates hierarchical relationships from the
  cells, populations, expression source, and markers selected for this plot. It is
  calculated in the temporary marker-only AnnData using the chosen Pearson,
  Spearman, or Kendall correlation and complete, average, or single linkage. Any
  dendrogram in the source AnnData is ignored and the source object is not changed.
- **Population cell totals** can preserve the current population order or sort
  populations from smallest to largest or largest to smallest.
- **Marker ordering** can preserve the marker-list order or cluster the selected
  markers by cell-level expression similarity using
  `SpatialBiologyToolkit.utils.reorder_vars_by_expression`. Ordering is calculated
  from the temporary plotting AnnData, so it follows the selected cells, ROIs,
  expression source, and markers without changing the source AnnData.
- **Population colours** adds a narrow strip beside the population axis. Colours
  come from the current `adata.uns["<observation>_colors"]` palette when available,
  using the same mapping as Population QC and `matrixplot_with_row_colors`. A
  dedicated, editable **Colour/label gap** separates the strip from even long
  population labels. **Colour box width** controls the strip thickness. Both values
  use points and retain their physical size when the popup is resized.
- **Axis arrangement** can swap the marker and population axes.

Expression heat maps use the native expression **Colour map**, **Population
colours**, and freshly recalculated **Dendrogram** controls above. Composition and
label-comparison heat maps use the common heat-map colour map and population-colour
controls below. **Cell/bar edge colour** and **Edge weight** can draw boundaries
between heat-map cells or around bars; set the weight to zero for no edge.

The common controls below the plot-specific options apply to every plot:

- **Legend** shows or hides the categorical legend or continuous colour scale.
  **Legend position** controls categorical embedding and stacked-bar legends;
  native Scanpy expression colour scales retain their right-hand position.
- **Axis labels** independently show or hide the X and Y axis titles.
- **Axis ticks** independently show or hide X and Y tick marks and their text.
  On matrix, dot, violin, composition, and comparison plots, the tick text often
  contains the biologically meaningful population or marker names.
- **Title** can use the descriptive automatic title, a custom title, or no visible
  title. A hidden title does not remove the meaningful name used for the popup and
  open-window list.

A dendrogram requires at least two selected markers and three represented
populations. A clear readiness message is shown before plotting when the current
settings do not satisfy those requirements.

For composition plots, choose the observation that represents samples, ROIs,
patients, conditions, or another grouping. Percentages are normalized within each
grouping value; counts remain absolute. Bar width changes the gap between bars,
while start and end padding reserve independent space before the first and after
the last bar. Enable **Y-axis limits** to enter a fixed minimum and maximum; leave
it disabled to retain automatic scaling. Bars can be sorted ascending or descending
by the count or percentage of one chosen population. For label comparison, rows are the
original/comparison labels and columns are the primary labels from box 1. Row
percentages are usually the clearest way to see how each original Leiden cluster
was renamed, merged, or split.

## Generate and manage plots

The readiness line explains exactly what is missing or summarizes how many cells,
label groups, and ROIs will be used. **Open in a new resizable window** creates a
modeless window, so several plots can remain open while NapariSBT is used.

Resize the window normally: Scanpy's Matplotlib figure grows with it. Matrix plots,
dot plots, and stacked violins measure their rendered population labels, marker
names, titles, and separate Scanpy legend axes, then reserve the required margins
for the current window size. The fit is recalculated after the window is resized.
The initial popup also respects Scanpy's requested figure width when screen space
allows, which is particularly useful for wide marker matrices.
The toolbar offers zoom, pan, reset, and image saving. **Export plotted data
CSV…** writes the precise points or aggregate values underlying that window; it
never modifies AnnData.

The open-window list can bring a hidden plot to the front, close one plot, or close
all plots. Plot windows are snapshots. When the live AnnData is loaded again or a
population label is synchronized, existing entries are marked **Labels/data may
have changed** and their window shows **Out of date**. Generate a new plot to see
the updated state; the old window is retained for visual comparison until closed.

Population naming provides **Open these labels in Scanpy plotting**. If naming
changes are unsaved, NapariSBT offers to save and synchronize them first, then
opens this tab with the derived observation already selected.
