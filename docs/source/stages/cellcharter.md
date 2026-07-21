# CellCharter neighbourhoods

## What this stage does

CellCharter assigns every segmented cell to a **spatial cluster**, also called a
cellular niche or cellular environment. The assignment reflects two sources of
information at once:

- the cell's own molecular phenotype; and
- the molecular phenotypes of cells surrounding it at several spatial scales.

This is different from ordinary cell clustering. A conventional phenotyping
analysis groups molecularly similar cells and might identify T cells,
macrophages, endothelial cells, or tumour cells. CellCharter instead asks which
recurring tissue contexts those cells occupy. The same macrophage phenotype can
therefore be assigned to different CellCharter clusters when it lies beside a
tumour nest, in a lymphoid aggregate, or in uninvolved stroma. Conversely, several
different cell types can share one CellCharter label when together they form a
reproducible multicellular niche.

SpatialBiologyToolkit stores the resulting categorical label in
`adata.obs["spatial_cluster"]` by default. The labels `0`, `1`, and so on are
arbitrary identifiers: they are not a ranking, a cell type, or a measure of
disease severity. Biological names should be assigned only after examining the
cluster's cell-type enrichment, molecular features, spatial position, component
shapes, and consistency across samples.

## Why spatial context matters

Tissues are organized systems rather than bags of independent cells. Biological
function often depends on local arrangements: lymphocytes and antigen-presenting
cells form immune aggregates; tumour cells interact with stromal and immune
compartments; vessels define perivascular niches; and epithelial layers create
ordered boundaries. Two cells with similar marker profiles may consequently have
different roles because their neighbours, anatomical compartment, or distance
from a boundary differs.

Looking only at cell-type abundance loses this organization. Two samples can
contain the same proportions of tumour cells, neutrophils, and fibroblasts while
placing those cells in very different configurations. CellCharter encodes local
composition together with the molecular state of the focal cell, allowing those
configurations to become clusterable features.

The method is unsupervised. It is not told what a lymphoid aggregate or invasive
margin should look like, and it does not require cell-type annotations to create
clusters. Existing cell annotations become important afterwards, when the
clusters are interpreted and tested.

## How CellCharter constructs a spatial niche

### 1. Choose a molecular representation

Each cell first needs a numerical feature vector. In the published CellCharter
workflow, dimensionality reduction and batch correction are modular and chosen
for the assay: the authors used distribution-aware variational autoencoders for
transcriptomic, epigenomic, and proteomic data. This step matters because
neighbourhood aggregation cannot distinguish a technical shift from a biological
shift. If cells separate by acquisition run before spatial analysis, the inferred
niches can also become run-specific.

This pipeline normally uses the previously calculated BioBatchNet biological
embedding, `adata.obsm["X_biobatchnet"]`. It therefore does **not** train a new
CellCharter variational autoencoder by default. If that key is absent, the stage
can fall back among the pipeline's integration representations
(`X_batch_integration`, `X_biobatchnet`, `X_pca_harmony`, and `X_pca`). A specific
AnnData layer or the primary matrix `adata.X` can also be used.

An optional TRVAE mode is available. TRVAE learns a low-dimensional embedding
while using a configured batch or condition label to encourage alignment between
datasets. It should be enabled only when its input scale, reconstruction loss,
and condition variable are scientifically appropriate. In this mode the model
uses `adata.X` or `use_layer`; `use_rep` is not the TRVAE input. A loaded TRVAE can
be reused or fine-tuned, and a fitted model can be saved as a reusable project
asset.

`scale_by_sample` performs a separate z-score for every feature within every
sample. This can reduce sample-wide offsets, but it also removes between-sample
differences in the mean and variance of each feature. Those differences may be
technical, biological, or both. Scaling is therefore a modelling decision, not a
generic normalization improvement.

### 2. Build a spatial graph within each sample

The stage represents cells as nodes in a graph and connects spatially adjacent
nodes. By default, Squidpy constructs a Delaunay triangulation from each cell's
two-dimensional coordinates. A Delaunay graph joins nearby cells without
requiring a single physical distance threshold, making it useful for irregular
single-cell layouts.

Graphs are constructed with the resolved ROI/sample column as `library_key`, so
edges do not connect cells from different tissue sections. This is essential:
cells that are adjacent only because two coordinate systems overlap numerically
must never become biological neighbours.

Delaunay triangulation can create implausibly long edges across empty space or
along a tissue boundary. With the default `remove_long_links: true`, CellCharter
removes edges whose positive distance is above the configured global percentile,
99 by default. This reproduces the strategy described for single-cell imaging in
the paper, but the cutoff is not a biological interaction radius. Its suitability
depends on cell density, tissue gaps, coordinate units, and whether samples have
very different spatial resolutions.

The graph describes adjacency, not direct molecular interaction. A graph edge
does not prove that two cells touch membranes, signal to one another, or belong
to the same histological structure.

### 3. Aggregate successive rings of neighbours

For a focal cell, CellCharter finds cells one graph hop away, two hops away, and
so on. With the default `n_layers: 3`, its final vector concatenates:

1. the focal cell's own features;
2. the mean features of cells exactly one hop away;
3. the mean features of cells exactly two hops away; and
4. the mean features of cells exactly three hops away.

The rings are hop-specific rather than cumulative. The two-hop summary describes
cells reached in two shortest graph steps and excludes cells already included at
a nearer hop. Aggregation is performed separately within each sample.

The mean captures the average phenotype at each scale. The variance aggregation
can additionally capture local heterogeneity: two cells may have similar average
neighbours but differ because one sits in a homogeneous region and the other at a
mixed boundary. Multiple comma-separated aggregations concatenate multiple
summaries for every nonzero hop. If the input has *d* features, integer depth *L*,
and *J* aggregation functions, the resulting vector has
`d * (1 + L * J)` values. Greater depth and more aggregations supply richer
context, but increase dimensionality and blur increasingly distant structures.

The combined matrix is stored in `adata.obsm["X_cellcharter"]` by default. It is
useful for understanding what the clustering consumed, but its columns are
derived features rather than measured marker abundances.

### 4. Cluster the contextual vectors

CellCharter fits a Gaussian mixture model (GMM) to the aggregated vectors. A GMM
models each cluster as a multivariate distribution and assigns cells according to
the fitted mixture. The default full covariance allows each cluster to have its
own feature correlations and orientation, which is more flexible than assuming
spherical clusters but requires more data and computation.

The published CellCharter package also provides `ClusterAutoK`: it repeatedly
fits GMMs over a range of cluster numbers and uses agreement between solutions at
neighbouring values of *K*, summarized by the Fowlkes--Mallows index, to identify
stable candidates. **SpatialBiologyToolkit does not run that stability search in
this stage.** It fits one user-specified number, `n_clusters` (11 by default).
Choosing and defending that number remains part of the analysis.

There may not be one uniquely correct *K*. In the paper's lung-cancer analysis,
stable coarse, intermediate, and fine solutions represented different levels of
biological organization. A low *K* may distinguish broad anatomical
compartments; a higher *K* may split them into patient-specific niches or local
cell states. The most useful solution depends on the scientific question and must
be checked for reproducibility and interpretability.

## Evidence from the published study

The CellCharter study evaluated the framework across several spatial
technologies and biological settings. On annotated 10x Visium dorsolateral
prefrontal cortex samples, joint clustering was assessed against cortical-layer
annotations and the authors reported favourable accuracy and memory use relative
to the compared methods. The study also analysed nine CODEX mouse-spleen samples
containing 707,466 cells; CellCharter recovered known splenic anatomy and was
reported to run four times faster than STAGATE in that comparison.

The downstream analyses were demonstrated in healthy and lupus-model spleen,
spatial multiomic mouse brain, and multiple lung-cancer cohorts measured with
CosMx, MERFISH, and IMC. In lung cancer, the authors identified a spatial
association between tumour-associated neutrophils and tumour cells with hypoxia-
and migration-related states, then examined related signatures in independent
cohorts. These applications show that the framework can reveal testable tissue
organization across modalities; they do not guarantee that every returned
cluster is a real biological niche or that an observed proximity is causal.

Batch correction was important in the published joint analyses. Without it,
clusters could align with donor rather than shared anatomy. At the same time,
integration can erase genuine donor- or condition-associated biology if the
technical covariate is confounded with the phenotype of interest. Both residual
batch structure and overcorrection should therefore be evaluated before accepting
the spatial clusters.

## What SpatialBiologyToolkit runs

The CellCharter environment currently pins `cellcharter==0.3.7`. The stage:

1. loads the configured AnnData and resolves its sample and spatial-coordinate
   keys;
2. selects an existing representation or optionally computes a TRVAE latent
   representation;
3. optionally z-scores features within samples;
4. constructs one spatial graph per sample and optionally removes long links;
5. concatenates the focal and hop-specific neighbourhood features;
6. fits a fixed-*K* CellCharter GMM and records categorical cluster labels;
7. optionally calculates cell-type enrichment, cluster neighbourhood enrichment,
   differential neighbourhood enrichment, and component shapes;
8. writes tables and diagnostic figures; and
9. saves the updated AnnData, normally back to the canonical pipeline path.

If `repeat_cluster_analysis: false` and the configured cluster column contains at
least one non-null label, the implementation reuses that column and skips TRVAE,
graph construction, aggregation, and clustering. This is intended for deliberate
plotting or downstream-analysis reruns. A partially populated column also meets
that condition, so reused labels should be checked for missing cells. The other
`repeat_*_analysis` controls similarly reuse compatible results already stored in
`adata.uns`. Unset repeat flags default to recomputation; `repeat_analysis` is a
deprecated common fallback.

## Main inputs

### Cells and molecular features

Rows of the AnnData must represent the same segmented cells used by the spatial
coordinates. The selected feature matrix must have one row per cell and should
encode biologically meaningful variation on a scale suitable for a GMM. For IMC,
this is normally a batch-integrated marker embedding rather than raw image
pixels.

The representation determines what kinds of niche the model can discover. A
cell-type-dominated embedding tends to identify mixtures of broad phenotypes. An
embedding that preserves activation or functional markers can separate niches
containing the same cell types in different states. Excluding unreliable markers
upstream can be preferable to letting technical artefacts define neighbourhoods.

### Sample identity

`general.roi_obs` identifies independent ROIs or samples and prevents graph edges
from crossing them. The stage tries a small set of conventional names only if the
configured column is missing. This column should represent a physical coordinate
system, not merely a treatment arm or acquisition batch.

### Spatial coordinates

The stage uses `adata.obsm[general.spatial_key]`, retaining its first two columns,
or creates it from `general.x_coord_obs` and `general.y_coord_obs`. Coordinates
must be finite and aligned with AnnData rows. Mirroring or rotating an ROI does
not change graph topology, but anisotropic scaling, duplicated coordinates,
stitching errors, and inconsistent units can change which cells become
neighbours.

### Optional biological annotations

`general.population_obs_primary` is not needed for clustering. When present, it
is used to explain which annotated cell populations are enriched in each spatial
cluster. `general.case_obs`, `general.groupby_obs`, and associated group lists are
used for composition plots or as fallbacks for between-condition analyses.

## Outputs and where they live

The updated AnnData is the reusable scientific output. Default keys include:

| Location | Default key | Meaning |
|---|---|---|
| `adata.obs` | `spatial_cluster` | Categorical cellular-niche assignment for each cell |
| `adata.obsm` | `X_cellcharter` | Focal plus hop-specific aggregated feature matrix |
| `adata.obsm` | `X_trVAE` | Optional TRVAE latent representation |
| `adata.obsp` | `spatial_connectivities`, `spatial_distances` | Within-sample spatial graph and edge distances |
| `adata.uns` | `spatial_cluster_colors` | Stable colours for cluster displays |
| `adata.uns` | `spatial_cluster_<cell-type-key>_enrichment` | Cell-type enrichment matrices and optional P values |
| `adata.uns` | `spatial_cluster_nhood_enrichment` | Cluster-to-cluster edge enrichment |
| `adata.uns` | `spatial_cluster_<condition-key>_diff_nhood_enrichment` | Pairwise condition contrasts in edge enrichment |
| `adata.uns` | `shape_<component-key>` | Optional component boundaries and shape metrics |
| `adata.uns` | `cellcharter_pipeline` | Resolved inputs, representations, repeat policy, and downstream-analysis provenance |

The QC directory contains global and per-sample cluster counts, spatial maps, ROI
mask-style views, enrichment matrices and plots, and—when the relevant metadata
are available—UMAP and case-level composition plots. Optional shape analysis adds
per-cell component assignments, component sizes, boundary summaries, and metric
tables. These exports make review convenient; downstream code should use the
AnnData keys as the canonical machine-readable results.

## Understanding the downstream analyses

### Cell-type enrichment within spatial clusters

When a primary population annotation exists, CellCharter compares the observed
co-occurrence of each cell type and spatial cluster with that expected if the two
labels were independent. The stored enrichment is a log2 fold change. A value of
`1` means twofold enrichment, `0` means no enrichment, and `-1` means half the
expected representation. Positive enrichment supports a biological name for a
niche, but should be considered alongside absolute counts: a rare cell type can
have a large fold enrichment based on few cells.

Permutation P values are optional and are not calculated by default. If enabled,
they test label association under CellCharter's permutation scheme. They do not
by themselves account for patient-level replication, multiple testing, uncertain
cell typing, or selection of the cluster solution on the same data.

### Cluster neighbourhood enrichment

This analysis asks whether cells assigned to one spatial cluster connect to cells
in another cluster more or less often than expected from cluster abundance and
node degree. The default is asymmetric: each row is interpreted as a source
cluster, and the fraction of its edges reaching a target cluster can differ from
the reverse fraction. This is valuable when a small niche lies almost entirely
beside a large compartment while only a small fraction of the large compartment
borders that niche.

With the pipeline defaults, within-cluster edges are excluded and enrichment is
reported as observed minus expected; the diagonal is therefore absent. The
analytical expectation is fast but does not produce P values. Enabling P values
uses cluster-label permutations. Symmetric and log-fold-change alternatives are
available, but matrices generated with different definitions should not be
compared as though they share one scale.

Enrichment is an association on the constructed graph. Positive values can arise
from anatomy, shared attraction to a third structure, cell-density differences,
or graph artefacts; they are not evidence of signalling direction or physical
attraction.

### Differential neighbourhood enrichment

Differential analysis subtracts the cluster-neighbourhood enrichment matrix of
one condition from that of another. It asks whether a particular niche boundary
or adjacency is stronger in one group, not merely whether either group has a
positive enrichment in isolation.

Condition labels belong to cells, but biological replication belongs to samples.
When P values are requested, CellCharter permutes condition assignments using the
configured library/sample key. Meaningful inference therefore requires multiple
independent samples per condition and a one-to-one, scientifically valid mapping
from sample to condition. Cells are not independent replicates. Strong contrasts
from one section per condition are descriptive, regardless of how many cells the
sections contain.

### Shape characterization

A spatial cluster can occur as several disconnected tissue regions. CellCharter
first identifies same-cluster connected components above `shape_min_cells`, then
uses an iterative alpha-shape procedure to construct a polygonal boundary for
each component. Small holes relative to the component area are removed according
to `shape_min_hole_area_ratio`.

The published framework defines curl, elongation, linearity, and purity. This
pipeline wrapper currently computes **linearity and curl only**:

- linearity is the longest path through the polygon's skeleton divided by total
  skeleton length; and
- curl measures how much the region bends or twists relative to its major axis
  and inferred fibre length.

High linearity can indicate a cord, boundary, or vessel-like arrangement; high
curl can indicate a curved, circular, or irregular region. These are geometric
descriptions, not biological identities. They are sensitive to the minimum cell
count, graph connectivity, segmentation density, cut tissue edges, holes, and the
alpha-shape settings. Only components with comparable sampling and adequate size
should be compared.

## Choosing settings and interpreting a solution

The most consequential settings should be explored as a connected set:

- **Representation:** confirm that expected cell phenotypes remain separated and
  technical batches do not dominate it.
- **Graph:** overlay edges on representative dense regions, sparse regions,
  cavities, and tissue borders. Check the distribution of edge lengths before and
  after pruning.
- **Neighbourhood depth:** relate graph hops to an empirical physical distance.
  Three hops can span very different micrometres in densely and sparsely packed
  tissue.
- **Cluster number:** compare several values of *K*. Look for solutions that recur
  across random seeds and samples, split tissue in biologically coherent ways,
  and retain enough cells for downstream comparisons.
- **Composition and localization:** inspect both enrichment and raw counts, then
  view clusters in every sample rather than only on a pooled UMAP.
- **Condition association:** summarize cluster abundance and spatial relationships
  at the case level. Avoid treating thousands of cells from one case as thousands
  of biological replicates.

A good niche label usually combines composition and anatomy—for example,
"B-cell follicle," "macrophage-rich tumour boundary," or "perivascular stromal
niche"—and documents uncertainty. A label based on a single enriched marker or
one visually striking ROI is provisional.

## Common problems and limitations

- **Segmentation errors propagate.** Merged, split, or misplaced cells alter both
  molecular features and graph topology.
- **Cell centroids simplify geometry.** Delaunay neighbours need not be cells with
  touching membranes, particularly around large cells or empty spaces.
- **Density changes the biological scale of a hop.** The same `n_layers` can cover
  different physical radii across tissue compartments or platforms.
- **A joint model favours recurrent structure.** Rare or sample-specific niches
  can be absorbed into larger clusters, while technical sample signatures can
  create apparently private niches.
- **Batch and biology may be confounded.** Neither BioBatchNet nor TRVAE can infer
  which differences are technical when each condition occurs in a separate
  batch.
- **GMM assumptions are approximations.** Full-covariance Gaussian components are
  flexible, but aggregated biological features need not be Gaussian and the fit
  can vary with initialization.
- **The selected cluster number defines the level of explanation.** Different
  values may all be defensible and should not be presented as interchangeable.
- **Enrichment is not abundance.** Fold enrichment, cell count, component size,
  and prevalence across cases answer different questions.
- **Spatial association is not mechanism.** CellCharter can identify where cell
  states co-occur; perturbation or orthogonal validation is needed to establish
  recruitment, signalling, or causation.
- **P values require a valid null and replication.** Permutations do not repair
  pseudoreplication, unbalanced designs, biased ROI selection, or extensive
  exploratory multiple testing.

## Configuration reference

Every setting, default, and pipeline-specific behaviour is documented in the
[CellCharter configuration reference](../reference/configuration/sections/cellcharter.md).

## Further reading

- Varrone *et al.* (2024), [CellCharter reveals spatial cell niches associated
  with tissue remodeling and cell plasticity](https://doi.org/10.1038/s41588-023-01588-4),
  *Nature Genetics* 56, 74--84.
- [CellCharter source repository](https://github.com/CSOgroup/cellcharter).
- [CellCharter documentation](https://cellcharter.readthedocs.io/).
