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
    PopulationQCArtifactWriter,
    inspect_population_data,
    list_stored_population_qc,
    load_stored_population_qc,
    plot_clustering_qc_panels,
    plot_marker_distributions,
    plot_population_breakdown,
    plot_population_matrixplot,
    plot_population_umap,
    profile_population,
    summarize_population_representation,
)

sdata = read_zarr("antonn_spatialdata.zarr")
population_key = "leiden_1.0"

context = inspect_population_data(sdata, population_key)
display(context.population_counts.head())
display(list_stored_population_qc(sdata))
structural_qc = load_stored_population_qc(sdata, population_key)
structural_panels = plot_clustering_qc_panels(structural_qc)

global_umap = plot_population_umap(
    sdata,
    population_key,
    max_cells=200_000,
    random_state=1729,
)
marker_matrix = plot_population_matrixplot(
    sdata,
    population_key,
    standardization="marker_robust_zscore",
    standardization_clip=3.0,
    max_cells_per_population=10_000,
    random_state=1729,
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

representation = summarize_population_representation(
    sdata,
    population_key,
    group_keys=("animal", "ROI"),
)
display(representation.for_population("18"))
case_breakdown = plot_population_breakdown(
    representation,
    group_key="animal",
)
```

Heatmaps show the broad landscape; effect sizes and histograms test a focused
identity hypothesis. Representation summaries show whether apparent evidence
is reproduced across animals/cases and ROIs rather than being dominated by one
image. A marker difference, structural score, or image gallery should not be
used alone to assign a label.

`load_stored_population_qc()` validates that the cached result still matches
the observation identities/order, source labels, representations, graph, and
sweep columns. It never recalculates. If no compatible result is available,
run the managed `popqc` pipeline deliberately or request user direction rather
than launching the expensive analysis from an Agent notebook.

## Inspect deliberately selected cells

```python
from SpatialBiologyToolkit.population_qc import (
    plot_population_cell_gallery,
    select_population_cell_panel,
)

selection = select_population_cell_panel(
    sdata,
    population_key,
    "18",
    marker="CD3",
    expectations=expectations,
    clustering_qc=structural_qc,
    diversity_keys=("animal", "ROI"),
    random_state=1729,
)

galleries = plot_population_cell_gallery(
    sdata,
    selection,
    channel=("CD3", "CD19", "CD68"),
    color=population_key,
    outline_target_only=True,
    ncols=5,
    separate_strategies=False,
    compact_titles=True,
)
```

Each gallery title records its selection strategy and channel-to-RGB mapping.
Inspect unmasked local context first, then repeat with
`mask_outside_target=True` to verify that signal is inside the target mask.
Selection is deterministic for a given `random_state`, and every result retains
the exact observation name, ROI, and `ObjectNumber` for each cell.

The standardized panel requests 20 unique cells: four each from typical,
boundary, marker-high, contradictory, and random strategies. Supply structural
QC and expectations so every strategy can run. If fewer than 20 cells can be
selected, retain the warning and report the shortfall.

## Save complete artifacts beside staged notebooks

```python
writer = PopulationQCArtifactWriter(
    "population_assessment_files",
    stage="population",
    population="18",
)
writer.save_result_tables(expression, "expression_profile")
writer.save_plot_result(marker_matrix, "marker_matrix")
writer.save_plot_result(global_umap, "umap_context")
writer.save_table(
    selection.cells,
    "selected_cells",
    category="galleries",
)
writer.record_stage_conclusion(
    stage_id="population-18-1",
    hypothesis="Population 18 contains a coherent T-cell identity.",
    conclusion="The observed B/T mixture does not support a single T-cell call.",
    decision="unresolved",
    evidence_artifact_ids=("population:18:figure:marker_distributions",),
    notebook_path="notebooks/populations/population_18.ipynb",
)
```

The writer saves complete CSV tables, PNG figures, and JSON metadata using
stable paths and updates `manifests/artifacts.csv`. It does not modify AnnData
or SpatialData. Notebook displays can remain compact because the unabridged
values are stored outside rendered HTML.

At completion, use `save_posterior_mapping()` and
`save_execution_audit()` for the canonical unabridged decision table and the
explicit no-Zarr-write/source-column-preservation audit.

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

# Save one directly joinable row per source observation before cleanup.
candidate_writer = PopulationQCArtifactWriter(
    "population_assessment_files",
    stage="candidate",
    population="qc_18",
)
candidate_obs_path = candidate_writer.save_observation_columns(
    sdata,
    candidate.columns,
    table_name="table",
    source="subcluster_population",
)

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

The candidate observation CSV starts with the unique `obs_name` merge key and
then contains the exact proposed columns. It is not a population mapping. A
user can attach it to the corresponding AnnData table without recomputing the
candidate:

```python
import pandas as pd

candidate_obs = pd.read_csv(candidate_obs_path).set_index("obs_name")
adata.obs = adata.obs.join(candidate_obs, validate="one_to_one")
```

For `attach=False`, pass `candidate.adata` to
`save_observation_columns()`. The resulting CSV contains the candidate subset
and remains joinable by `obs_name`; observations outside that subset receive
missing values after a left join.

## Recommended notebook evidence record

For every proposed label, merge, or split, retain:

1. the population and exact source/candidate annotation column;
2. the biological hypothesis and positive, supportive, and negative markers;
3. structural QC and resolution-stability outputs;
4. focused expression distributions and effect sizes;
5. case/animal and ROI representation summaries;
6. typical, boundary/contradictory, and marker-extreme cell galleries;
7. the final decision, confidence, counter-evidence, and unresolved ambiguity.

For a multi-population assessment, keep one prior notebook, one global
clustering notebook, one notebook per population, and separate candidate
notebooks when a merge/split evidence trail becomes substantial. Use repeated
Hypothesis, Plan, Execute, and Evaluate sections, preserving the exact analysis
calls rather than display-only checkpoints.

The detailed function docstrings are the source of truth for parameters,
returned evidence, sampling behaviour, and agent-specific interpretation notes.
