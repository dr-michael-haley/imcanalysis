# Subclustering

## What this stage is for

Subclustering asks whether a broad, already annotated population contains reproducible internal structure. Instead of allowing very different lineages to dominate a graph built from the entire experiment, it restricts the analysis to one parent population at a time and partitions those cells again. This can help separate biologically meaningful states or subtypes that were compressed into a single first-pass label—for example, proliferating and non-proliferating tumour cells, naive and activated lymphocytes, or distinct macrophage phenotypes.

The output is deliberately not treated as a final biological annotation. The stage creates candidate data-driven subclusters, diagnostic figures, and an editable remapping table. A scientist reviews the evidence and decides which candidates should be retained, merged, or renamed before final labels are written back to the AnnData object.

## Why focused clustering can reveal additional structure

Graph-based clustering compares each cell with nearby cells in a selected feature space. In a global dataset, differences between major lineages—such as epithelial cells, lymphocytes, and stromal cells—are often much larger than differences within one lineage. The nearest-neighbour graph therefore spends most of its structure separating those major groups. Restricting the graph to a parent population changes the question from “which broad lineage is this cell?” to “how do cells within this lineage differ from one another?”

The stage uses Leiden community detection on a neighbour graph for each requested parent population. Leiden identifies groups of cells that are more densely connected to one another than to the rest of the subset. The configured resolution controls the granularity of that graph partition: larger values usually produce more, smaller communities, whereas smaller values usually produce fewer, broader communities. Resolution is not an expected cluster count, and the same value is not biologically or numerically equivalent across populations with different cell numbers, marker distributions, or sampling density.

A Leiden community is a statistical partition, not proof of a distinct cell type. Continuous activation, differentiation, or spatial gradients may be cut into apparently discrete groups. Conversely, a rare but real phenotype may be missed when it has too few cells or when its defining markers were not measured or selected.

## The three-checkpoint curation workflow

Subclustering is intentionally separated into checkpoints so that scientific decisions are made between computation and integration.

### Checkpoint 1: generate editable templates

On the first `generate` run, the stage inspects the configured parent-population observation and creates two reusable CSV files in `subclustering.output_subdir`:

- `sublustering_settings.csv` contains one row per current parent population, with a Leiden resolution and marker-list choice. The historical filename misspelling is retained for compatibility.
- `marker_list.csv` contains all AnnData variables and a default Boolean `markers_all` column. Additional columns whose names begin with `markers_` can define focused panels, such as `markers_tcell_state` or `markers_myeloid`.

The stage then stops. This is expected: the files are an invitation to choose which populations should be subdivided and which measured proteins are relevant to each biological question. Delete unwanted rows from the settings table, adjust resolutions, add marker panels, and set each row's `marker_list` value to the suffix of the desired column—for example, `tcell_state` selects `markers_tcell_state`.

### Checkpoint 2: generate candidate subclusters

On a subsequent `generate` run, the existing templates are read and each usable settings row is processed. The stage:

1. selects cells matching that row's parent population;
2. selects markers from the requested Boolean marker-list column;
3. constructs a nearest-neighbour graph from those marker values;
4. applies Leiden clustering at the requested resolution;
5. writes labels such as `CD8_T_cell_0` and `CD8_T_cell_1` into a new observation column;
6. produces UMAP and marker-expression diagnostics; and
7. updates the editable subcluster-to-final-population remapping table.

Cells outside the selected parent population retain their original parent label in that row's subcluster observation. This makes the column plottable across the full experiment while keeping the newly divided population visible in context.

Each successful settings row creates its own subcluster column. Multiple rows can therefore explore different resolutions or marker panels without immediately discarding alternatives. Row-level failures are logged and skipped, so always check the execution report: an output can be partial if one population failed while others completed.

### Checkpoint 3: apply a curated remap

`subcluster_to_final_population.csv` initially maps every generated subcluster to itself. Edit its `final_population` column to assign biologically meaningful names or to merge candidates that are not sufficiently distinct. Running in `apply` mode writes the reviewed labels to `subclustering.final_label_key`, which defaults to `population_final`.

By default, application is skipped until at least one `final_population` entry differs from the generated `subcluster` value. This safety check prevents an untouched computer-generated table from being mistaken for curated annotation. The stage also exports a stable cell-to-final-label table using the configured master index, or observation names when no master index is available.

If several settings rows subdivide the same parent population, a cell can have a candidate label in more than one generated column. During remapping, the first applicable mapping wins. Alternative resolutions are therefore useful for comparison, but competing rows should be removed or ordered deliberately before applying final labels.

## Marker selection and the data actually clustered

Marker choice defines the biological similarities available to the algorithm. For a focused population, include proteins capable of distinguishing the states or subtypes of interest. A T-cell panel might prioritise lineage refinement, activation, exhaustion, memory, and proliferation markers; a myeloid panel might emphasise macrophage, monocyte, antigen-presentation, phagocytic, and inflammatory programmes. Exclude DNA channels, technical channels, segmentation-only measurements, and markers that merely re-separate unrelated broad lineages unless they are relevant to the within-population question.

The selected marker columns are read from `adata.X` exactly as stored when this stage runs. Subclustering does not normalise, transform, scale, or batch-correct those values itself. Interpretation therefore depends on the upstream processing represented by `adata.X`. If it contains transformed or standardised values, the neighbour graph reflects that representation; if it contains raw-like intensities, highly abundant or high-variance proteins may dominate distance calculations.

The `use_rep` setting deserves particular care. In the current implementation it selects a preferred representation only when a missing *global QC UMAP* must be computed. The within-population neighbour graph is built from the explicitly selected markers in `adata.X`, not from `X_biobatchnet`, Harmony, PCA, or another `obsm` representation. This preserves direct control over the biological markers driving each focused partition, but it also means that batch correction contained only in an integrated embedding is not automatically used for subclustering.

The chosen marker lists are saved as Boolean columns in `adata.var` with names beginning `subclustering_`. This records which features were used and helps make the curation decision auditable.

## Diagnostic figures

For every successful settings row, the stage produces complementary views of the candidate partition.

### Combined UMAP

The combined plot colours the full dataset by the generated subcluster column. It shows where the subdivided parent population lies relative to other cells and whether candidate groups occupy distinct or overlapping regions of the global embedding.

If `adata.obsm['X_umap']` already exists, it is reused. This UMAP may have been constructed from a different feature set or an integrated representation and is not recalculated for each parent population. It is consequently a contextual display, not a direct visualisation of the exact neighbour graph used for subclustering. If no UMAP exists and `compute_umap_if_missing` is enabled, a global UMAP is computed using the preferred available representation.

### Individual highlighted UMAPs

When enabled, one plot per candidate subcluster highlights its cells against the rest of the experiment. These plots make cluster size, dispersion, and overlap easier to see than a densely coloured combined plot. A compact region can support a distinct phenotype, but visual separation alone is not biological validation; UMAP deliberately distorts global distances and can exaggerate gaps.

### Marker matrix plots

The matrix plot summarises the selected marker values across the candidate subclusters within the parent population. Use it to ask whether each group has a coherent, biologically plausible protein phenotype and whether the contrast is driven by the markers expected for the proposed label. The configured `matrixplot_vmax` is a display limit and should be adjusted to the scale of `adata.X`; saturation can hide important differences, while an excessively high limit can make genuine patterns appear faint.

## Biological review and interpretation

Before retaining a candidate subtype or state, consider several kinds of evidence together:

- **Marker coherence:** Does the group show a coordinated phenotype rather than a difference in a single noisy marker?
- **Biological plausibility:** Are co-expressed proteins compatible with known cell biology, and are mutually exclusive lineage programmes being interpreted cautiously?
- **Sampling support:** Is the group present across multiple regions of interest, specimens, or experimental batches, rather than being confined to one image or acquisition run?
- **Cell count:** Is there enough support for downstream abundance and spatial analysis? Very small communities are especially sensitive to segmentation errors and graph parameters.
- **Image evidence:** Do representative cells have credible morphology, localisation, and staining when returned to the source images?
- **Stability:** Does the broad conclusion persist under nearby resolutions or reasonable marker-panel changes?

A subcluster restricted to one biological sample may be real—for example, a treatment-specific response—but a group restricted to one staining batch, acquisition day, ROI edge, or damaged region is more likely technical. Check those covariates before assigning a biological name.

Subcluster labels also shape downstream conclusions. Splitting one parent population changes abundance denominators, interaction counts, neighbourhood composition, and network statistics. Over-splitting can create apparently specific spatial associations from small numbers of cells; over-merging can hide a rare but important niche. The final mapping should reflect the resolution at which the available markers, sample size, and intended downstream analyses can support defensible claims.

## Main inputs and outputs

The input is the annotated AnnData object plus the reusable CSV assets in `output_subdir`. Unless `output_adata_path` is set, the input AnnData is updated in place.

The principal outputs are:

- one observation column for each candidate subclustering run;
- the curated final-label observation after the apply checkpoint;
- Boolean variable annotations recording used marker lists;
- settings, marker-list, remap, and stable cell-to-label CSV files;
- combined and individually highlighted UMAPs;
- marker matrix plots; and
- provenance in `adata.uns['subclustering_pipeline']`, including paths, completed checkpoints, successful settings rows, and marker-list mappings.

## Important limitations

- Subclustering can only resolve phenotypes represented by the measured panel and retained in `adata.X`.
- The method treats cells as independent points and does not use tissue position when constructing clusters.
- Leiden resolution is exploratory and dataset-dependent; it is not a biological scale.
- A reused global UMAP is not a faithful picture of every marker-specific subclustering graph.
- Batch or ROI effects can become more visible after restricting to a lineage and can be mistaken for biological heterogeneity.
- Continuous biology can be discretised into artificial communities.
- Settings and marker names must match the current AnnData. Reusing templates after changing population labels or the marker panel requires careful review.

Subclustering is most useful as a structured hypothesis-and-curation stage: it proposes reproducible partitions, exposes the evidence needed to review them, and preserves the scientist's final decisions separately from the initial algorithmic labels.
