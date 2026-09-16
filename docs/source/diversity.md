# Population composition and diversity in notebooks

These read-only Python APIs work with AnnData `.obs` or a pandas observation
DataFrame. They do not access `.X`, images, graphs, or mutate AnnData. They are
library helpers, not a managed pipeline stage: there are no new CLI commands,
configuration fields, SLURM wrappers, environments, or canonical asset writes.
Notebook callers own exporting returned tables and figures to their report.

```python
from SpatialBiologyToolkit import diversity, plotting

counts = diversity.population_counts(adata, "cell_type", "Case")
pooled = diversity.population_diversity(adata, "cell_type", "Case")
roi = diversity.population_diversity(adata, "cell_type", ["Case", "ROI"])
equal_roi_case_mean = roi.groupby(level="Case").mean(numeric_only=True)

# The same calls accept adata.obs instead of adata.
# Or use an existing count table (groups by populations):
metrics = diversity.diversity_from_counts(counts)
```

The default plugin estimator is Simpson dominance `D = sum(p_i**2)` and
Gini–Simpson diversity `1-D`. Larger `1-D` indicates more diverse composition.
This matches ATHENA's global Simpson calculation; no spatial graph is needed.
See [ATHENA's implementation](https://github.com/AI4SCR/ATHENA/blob/master/src/athena/metrics/heterogeneity/base_metrics.py).
The optional `estimator="unbiased"` uses `sum(n_i*(n_i-1))/(N*(N-1))` and
requires integer counts. Do not mix the two estimators without documenting it.

The helpers also return richness, inverse Simpson, Shannon entropy (base 2 by
default), and `n_total`. Empty groups have zero richness but NaN diversity;
unbiased Simpson is undefined for fewer than two observations. Missing population
labels are excluded by default; `dropna=False` assigns a configurable missing
category. Missing grouping keys raise an error, so callers must explicitly
decide whether to exclude or label them. Proportions/weights can be used with
the plugin estimator, but their sum is not a cell count.

Pooling cells per case and averaging ROI-level diversity answer different
questions. An unweighted ROI mean gives each ROI equal influence; a pooled
case estimate weights ROI contributions by their cell counts. Export both when
comparing with historical ROI-based analyses. Neither measures spatial mixing.

## Dendrogram-aligned categorical strips

```python
mp, fig = plotting.matrixplot_with_row_colors(
    adata, marker_groups=["CD3", "CD68"], groupby_key="cell_type",
    dendrogram=True,
    additional_row_colors={"Family": family_color_by_cell_type},
    row_color_labels=True,
    figsize=(10, 6),
)
```

The population strip uses `adata.uns["cell_type_colors"]` and the categorical
order in `.obs`. Additional maps use actual displayed labels after clustering,
not the input row order. Missing colour entries raise a clear error. Original
calls and the existing `secondary_colorbar` scalar/categorical forms remain
supported. The new options are keyword-only and default to the original layout.

## Multi-track case figures

`plotting.plot_stacked_graphs` accepts the case-indexed count/proportion and
clinical DataFrames. All frames are aligned to their shared index in the chosen
ordering frame. Reindex to an explicit full case list before calling it if you
must preserve cases without labels or clinical data. Optional
`bar_colorbars=False` removes the legacy continuous colourbars for compact
tracks, and `show_case_labels=True` shows bottom-axis labels. All-missing
continuous columns remain blank instead of failing or being imputed to zero.
