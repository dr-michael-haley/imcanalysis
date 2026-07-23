# Agent-guided population quality control

`SpatialBiologyToolkit.population_qc` provides small, composable tools for
reasoning across clustering structure, marker expression, biological
replication, and cell images. Functions return typed results containing both
compact summaries and the underlying data used by their plots, so a notebook
can retain an auditable evidence trail.

The toolkit may add candidate annotations to the SpatialData table in memory.
It never writes the SpatialData Zarr store or replaces the source population
column. Saving a reviewed result is a separate, explicit step.

## Start with context and broad evidence

```python
from spatialdata import read_zarr

from SpatialBiologyToolkit.population_qc import (
    MarkerExpectations,
    inspect_population_data,
    plot_marker_distributions,
    plot_population_heatmap,
    profile_population,
    summarize_population_representation,
)

sdata = read_zarr("antonn_spatialdata.zarr")
population_key = "leiden_1.0"

context = inspect_population_data(sdata, population_key)
display(context.population_counts.head())

plot_population_heatmap(
    sdata,
    population_key,
    max_cells_per_population=10_000,
)

expectations = MarkerExpectations(
    positive_markers=("CD3",),
    supportive_markers=("CD4", "CD8a"),
    negative_markers=("CD19", "CD68"),
)
expression = profile_population(
    sdata,
    population_key,
    population="18",
    expectations=expectations,
)
display(expression.strongest_markers())
plot_marker_distributions(expression)

representation = summarize_population_representation(sdata, population_key)
display(representation.for_population("18"))
```

Heatmaps show the broad landscape; effect sizes and histograms test a focused
identity hypothesis. Representation summaries show whether apparent evidence
is reproduced across animals/cases and ROIs rather than being dominated by one
image. A marker difference, structural score, or image gallery should not be
used alone to assign a label.

## Inspect deliberately selected cells

```python
from SpatialBiologyToolkit.population_qc import (
    plot_population_cell_gallery,
    select_population_cells,
)

selection = select_population_cells(
    sdata,
    population_key,
    "18",
    strategies=("typical", "marker_high", "contradictory", "random"),
    n_per_strategy=4,
    marker="CD3",
    expectations=expectations,
)

galleries = plot_population_cell_gallery(
    sdata,
    selection,
    channel=("CD3", "CD19", "CD68"),
    color=population_key,
    outline_target_only=True,
)
```

Each gallery title records its selection strategy and channel-to-RGB mapping.
Inspect unmasked local context first, then repeat with
`mask_outside_target=True` to verify that signal is inside the target mask.
Selection is deterministic for a given `random_state`, and every result retains
the exact observation name, ROI, and `ObjectNumber` for each cell.

## Test a possible split in memory

```python
from SpatialBiologyToolkit.population_qc import (
    assess_candidate_clustering,
    discard_population_qc_columns,
    subcluster_population,
)

candidate = subcluster_population(
    sdata,
    population_key,
    "18",
    markers=("CD3", "CD4", "CD8a", "CD19", "CD68"),
    resolutions=(0.3, 0.6, 1.0),
    output_prefix="qc_18",
    attach=True,
)

# Runs on candidate.adata, the copied local subset and graph.
structural_qc = assess_candidate_clustering(candidate)
display(structural_qc.cluster_summary)

# Candidate columns are present on the live SpatialData table, but not on disk.
print(candidate.columns)

# Remove rejected or superseded candidate columns after recording the evidence.
discard_population_qc_columns(sdata, candidate.columns)
```

Use `create_leiden_sweep()` when the existing full-data neighbour graph is the
scientifically appropriate feature space. Use `subcluster_population()` only
when evidence suggests a parent population contains multiple phenotypes; its
local feature choice must be recorded and biologically justified. Candidate
columns retain the original labels outside the selected parent population.

For a proposed merge or relabel, `apply_population_mapping()` creates a new
candidate annotation without overwriting the source. Pass `copy_table=True` to
the mutation functions when RAM permits and complete isolation is preferable;
on multi-million-cell tables, reversible columns on the live in-memory table
usually avoid a costly AnnData copy.

## Recommended notebook evidence record

For every proposed label, merge, or split, retain:

1. the population and exact source/candidate annotation column;
2. the biological hypothesis and positive, supportive, and negative markers;
3. structural QC and resolution-stability outputs;
4. focused expression distributions and effect sizes;
5. case/animal and ROI representation summaries;
6. typical, boundary/contradictory, and marker-extreme cell galleries;
7. the final decision, confidence, counter-evidence, and unresolved ambiguity.

The detailed function docstrings are the source of truth for parameters,
returned evidence, sampling behaviour, and agent-specific interpretation notes.
