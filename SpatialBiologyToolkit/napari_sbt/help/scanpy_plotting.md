# Scanpy plotting

Scanpy plotting creates quick, regenerable views from the AnnData object loaded in
NapariSBT. It is designed for checking population identities, expression patterns,
sample representation, renames, merges, and subclusters while moving between
Population naming, Explore, and Population QC.

Plots are read-only. Opening a plot does not normalize or scale the AnnData,
recompute PCA, rebuild neighbours, change BioBatchNet output, calculate UMAP, or
run clustering. Every plot window is a snapshot of the labels and values that
existed when **Open in a new resizable window** was pressed.

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

For embeddings, select any `obsm` matrix with at least two components and choose
the horizontal and vertical component numbers. Interactive previews default to a
maximum of 50,000 points. Larger selections are deterministically and
population-stratified downsampled so repeated plots are stable and small
populations remain represented. Change the limit when a fuller view is needed.
Point size, opacity, and optional centroid names affect display only.

Expression plots use only the markers selected in the searchable list. **Select
visible** makes it easy to type a marker family and select the filtered results.
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
- **Axis arrangement** can swap the marker and population axes.

A dendrogram requires at least two selected markers and three represented
populations. A clear readiness message is shown before plotting when the current
settings do not satisfy those requirements.

For composition plots, choose the observation that represents samples, ROIs,
patients, conditions, or another grouping. Percentages are normalized within each
grouping value; counts remain absolute. For label comparison, rows are the
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
