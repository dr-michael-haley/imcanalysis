# Batch Integration

## What this stage does

This stage prepares cell-level IMC data from several technical batches for
joint visualization and clustering. It recomputes principal component analysis
(PCA) and then uses one of four routes:

- **Harmony** adjusts the cells' PCA coordinates to reduce systematic
  batch-associated displacement;
- **BBKNN** leaves the PCA coordinates unchanged but constructs a neighbour
  graph in which every batch contributes neighbours to every cell;
- **both** runs Harmony first and BBKNN on the Harmony-corrected coordinates;
  or
- **none** constructs an ordinary Scanpy neighbour graph from uncorrected PCA.

All four routes subsequently calculate UMAP and can run Leiden clustering at
several resolutions. The stage writes the new coordinates, graph, clusters, and
provenance into AnnData.

Batch integration does **not** change the marker values stored in `adata.X` or
its layers. Harmony changes a low-dimensional coordinate system and BBKNN
changes a graph. These derived representations influence UMAP, Leiden
clustering, CellCharter, and other graph- or embedding-based analyses, but they
must not be interpreted as corrected protein-abundance measurements.

## What a batch effect means in IMC

A batch effect is a reproducible technical difference between groups of cells
that were processed or measured under different conditions. In IMC this can
arise from staining date, antibody-lot performance, reagent ageing, instrument
tuning, detector sensitivity, ablation behaviour, slide handling, or changes in
sample processing. Even after marker normalization, these effects can shift
several markers together and make otherwise similar cells occupy different
parts of a multivariate representation.

If left unaddressed, cells can cluster by acquisition or staining batch rather
than phenotype. A T-cell state may form several batch-specific islands, or the
nearest neighbours of a cell may come almost exclusively from its own run.
UMAP and Leiden inherit this structure from the neighbour graph, so apparent
populations can reflect processing history rather than biology.

Not every difference between samples is a batch effect. Patient, treatment,
disease state, tissue compartment, ROI, and anatomical site can all represent
real biology. They may also be correlated with technical processing. Integration
is justified only for variation that the scientific design identifies as
unwanted technical variation.

## The essential design limitation: confounding

No integration algorithm can identify which part of a difference is technical
when the technical batch and the biological comparison are perfectly
confounded. If all control samples were stained in one batch and all treated
samples in another, a systematic shift could be treatment biology, staining
batch, or both. Harmony and BBKNN have no independent evidence with which to
separate them.

Applying integration in that situation can remove the effect under study or
force biologically different cells together. A mixed design is therefore more
important than the choice of integration algorithm. Each technical batch should
ideally contain biological replicates from every major study group, and the same
major cell states should be represented across batches.

Before configuring `batch_correction_obs`, ask:

- Does this column describe a technical process that should not define cell
  identity?
- Are the biological conditions distributed across its categories?
- Do its categories contain enough cells and more than one relevant phenotype?
- Are at least some comparable cell states expected in every category?
- Would removing all variation associated with this column also remove a
  scientific signal that the study needs to test?

In a typical IMC project, a staining or acquisition batch may be appropriate.
Correcting case, outcome, treatment, ROI, or tissue compartment requires a much
stronger scientific justification. Integration is not a remedy for an
unbalanced experimental design.

## The representation on which integration operates

Each cell is represented by many marker values. PCA converts those correlated
measurements into orthogonal axes that summarize major patterns of variation.
The first component captures the greatest variance that can be represented by a
linear combination of markers; later components capture progressively smaller,
independent patterns. Harmony and BBKNN operate on this compressed coordinate
system rather than directly on individual marker columns.

The active stage calls Scanpy PCA on the current AnnData object. Scanpy centres
each selected marker but does not scale every marker to unit variance by
default. If `adata.var['highly_variable']` exists, Scanpy's default PCA behaviour
uses those selected variables; otherwise it uses all markers. The stage itself
does not select markers, remove unwanted channels, normalize their distributions,
or standardize their variances before PCA.

Consequently, upstream choices remain important. Technical channels, DNA,
poor-quality antibodies, and markers with disproportionately large variance can
dominate PCA if they remain in the analysis matrix. Batch integration can then
align a representation whose main axes were biologically inappropriate. Review
the AnnData matrix and marker set before this stage rather than expecting
Harmony or BBKNN to repair upstream normalization or panel problems.

The stage recomputes PCA on every run and normally writes it to `X_pca`. With
`n_for_pca: null`, it requests one fewer component than the number of markers,
then clips that request to what the number of cells and markers permits. A
smaller component count emphasizes dominant, broad phenotypes; a larger count
retains weaker patterns but may also retain noise and residual batch structure.
The clipping calculation uses the total marker count. If a pre-existing
`highly_variable` mask selects fewer markers than the requested component count,
Scanpy can still reject the PCA request; lower `n_for_pca` or deliberately revise
that mask rather than silently changing the biological feature set.

## Harmony: correcting PCA coordinates

[Harmony](https://www.nature.com/articles/s41592-019-0619-0) was developed to
place cells from several single-cell datasets into a shared embedding while
retaining cell-type structure. This pipeline calls the Python implementation,
[`harmonypy`](https://github.com/slowkow/harmonypy), directly.

Harmony alternates between two linked operations.

### Maximum-diversity soft clustering

First, Harmony clusters cells in PCA space. The assignments are *soft*: a cell
can belong partly to several clusters rather than being irreversibly assigned to
one. This accommodates continuous cell states and uncertain boundaries.

Ordinary clustering rewards cells for lying close to a cluster centre. Harmony
adds a diversity penalty when a cluster contains a more batch-skewed composition
than expected from the dataset as a whole. The algorithm is therefore encouraged
to describe comparable states from different batches with shared clusters,
rather than creating a separate cluster for each batch.

This does not mean every final cluster must contain identical numbers from every
batch. Expected representation accounts for batch abundance, and the distance
to the cluster remains part of the objective. Nevertheless, the diversity term
encodes an assumption that batch mixing is desirable among biologically
comparable cells.

### Cluster-specific linear correction

For each soft cluster, Harmony estimates batch-specific displacement relative to
the cluster's global centre. A cell's correction is a weighted combination of
those cluster-specific adjustments, using its soft cluster memberships. Two
cells from the same technical batch can therefore receive different corrections
if they occupy different biological states.

The method alternates diversity-aware clustering and linear correction until
the assignments stabilize or the iteration limit is reached. Each correction is
estimated from the original PCA representation rather than repeatedly correcting
an already corrected result, which constrains the transformation and helps avoid
unbounded cumulative shifts.

The resulting coordinates have the same number of dimensions as the input PCA
but altered distances between cells. In this pipeline they are stored in both
`X_pca_harmony` and the canonical `X_batch_integration` representation. Scanpy
then builds an ordinary neighbour graph from `X_batch_integration`.

### Important Harmony parameters

The default `harmony_params` allows up to 30 Harmony rounds, prints progress,
and fixes the random seed at zero. Advanced parameters are passed directly to
the installed `harmonypy.run_harmony` implementation:

- `theta` controls the batch-diversity penalty. Larger values put more pressure
  on clusters to mix batches; zero removes that pressure.
- `sigma` controls the softness of cluster assignment. Smaller values approach
  hard assignment, whereas larger values let a cell contribute to more
  clusters.
- `lamb` is the ridge penalty on correction coefficients. Larger values shrink
  corrections and generally protect more of the original structure; supported
  implementations may estimate it when null.
- `nclust` controls the number of soft clusters and therefore how locally Harmony
  can model different corrections for different cell states.
- `max_iter_harmony` limits the outer correction rounds. Reaching a higher limit
  is not evidence that the biological result is more accurate.

Parameter names and defaults can change between `harmonypy` releases. The
pipeline removes null-like dictionary values before the call, so the default
`device: null` entry is not forwarded. Record the installed package version as
part of reproducibility when using advanced options.

## BBKNN: balancing the neighbour graph

[BBKNN](https://academic.oup.com/bioinformatics/article/36/3/964/5545955)
stands for batch-balanced k-nearest neighbours. It replaces the normal neighbour
search used by Scanpy. The pipeline accesses it through
[`scanpy.external.pp.bbknn`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.external.pp.bbknn.html),
which wraps the [BBKNN package](https://github.com/teichlab/bbknn).

In an ordinary k-nearest-neighbour graph, the algorithm searches the complete
cell pool and connects each cell to the closest cells regardless of batch. When
a technical shift is large, those neighbours may all come from the same batch.

BBKNN instead repeats the search separately inside every batch. For each cell it
selects a configured number of nearest neighbours from batch A, the same number
from batch B, and so on, then merges those lists into a single graph. This makes
cross-batch connections part of the graph by construction. UMAP and Leiden then
operate on that batch-balanced graph.

BBKNN does not move cells in PCA space and does not create corrected marker
values. Its primary product is the graph stored in AnnData's neighbour,
distance, and connectivity structures. `X_batch_integration` remains a copy of
the PCA coordinates in `bbknn` mode, even though the UMAP derived from the BBKNN
graph can look strongly integrated.

The method's core assumption is that at least some cells of the same type occur
across batches and that those corresponding states are more similar than
different cell types within a batch. If a genuine population occurs in only one
batch, BBKNN must still select neighbours for its cells from every other batch.
Those forced connections can blur a unique population, bridge unrelated states,
or distort a trajectory.

### Neighbour number and graph trimming

The native BBKNN parameter `neighbors_within_batch` is the number chosen from
*each* batch. The initial neighbour count therefore grows approximately as:

```{math}
k_{total} = k_{within\ batch} \times B
```

where `B` is the number of batch categories. With the library default of three
neighbours per batch and six batches, each cell starts with approximately 18
neighbours before graph symmetrization and trimming.

The pipeline's top-level `n_neighbors` has a different interpretation. When it
is set and `bbknn_params` does not explicitly contain `neighbors_within_batch`,
the stage calculates the per-batch value as the ceiling of `n_neighbors / B`.
This approximates a requested total while ensuring at least one neighbour from
each batch. When `n_neighbors` is null, BBKNN uses its own default of three per
batch.

`trim` can limit each cell to its strongest graph connections after balancing.
More aggressive trimming can make populations more distinct but may preserve
more batch separation or disconnect rare states. Approximate-neighbour options
such as Annoy or PyNNDescent improve speed on large data but can introduce small
run- and version-dependent differences in the graph.

## How the four modes differ

| `integration_method` | Coordinates in `X_batch_integration` | Final graph | Interpretation |
|---|---|---|---|
| `harmony` | Harmony-corrected PCA | Ordinary Scanpy neighbours | Batch adjustment is encoded in cell coordinates. |
| `bbknn` | Uncorrected PCA copy | BBKNN batch-balanced neighbours | Batch adjustment is encoded only in graph connections. |
| `both` | Harmony-corrected PCA | BBKNN on corrected coordinates | Coordinate correction is followed by enforced per-batch neighbour sampling. |
| `none` | Uncorrected PCA copy | Ordinary Scanpy neighbours | No batch correction, but PCA, UMAP, and optional Leiden are still recomputed. |

`both` is not an average or a choice between two independent integrations. It
first changes the PCA coordinates with Harmony and then constructs a BBKNN graph
from those changed coordinates. It combines two interventions and can
over-integrate data when either method alone is sufficient.

Harmony is often easier to reuse when a later method needs corrected continuous
coordinates. BBKNN is lightweight and directly targets the graph consumed by
UMAP and Leiden. The preferred method is the least aggressive one that removes
reproducible technical separation while preserving validated biological
structure; it cannot be selected from batch mixing alone.

## The implemented pipeline sequence

The active `bint` stage performs the following operations:

1. Load `input_adata_path`, falling back to `general.anndata_path`, through the
   pipeline's AnnData stage-state controls.
2. Validate the integration method and require `batch_correction_obs` for
   `harmony`, `bbknn`, or `both`.
3. Confirm that the configured batch column exists in `adata.obs`.
4. Recompute PCA with a valid number of components and overwrite the normal
   Scanpy PCA outputs.
5. Copy PCA into `X_batch_integration` as the initial shared representation.
6. In `harmony` or `both` mode, run `harmonypy` and replace
   `X_batch_integration` with the corrected coordinates.
7. Build either a BBKNN graph (`bbknn` or `both`) or an ordinary Scanpy graph
   (`harmony` or `none`).
8. Recompute `X_umap` from the final graph.
9. Optionally recompute Leiden labels at every configured resolution.
10. Save batch-coloured and Leiden-coloured UMAPs, write run details under
    `adata.uns['batch_integration']`, and save the updated AnnData.

The current Python module imports `harmonypy` when it starts, so that package
must be installed even for `bbknn` or `none` mode. BBKNN itself is invoked through
Scanpy's external wrapper.

By default, input and output both resolve to `general.anndata_path`. The stage
therefore updates the canonical AnnData rather than creating a second file. It
preserves marker matrices but replaces standard derived keys such as `X_pca`,
`X_umap`, the active neighbour graph, and any Leiden columns whose names match
the configured resolutions. Use `output_adata_path` when the unintegrated
derived representation must remain available in a separate AnnData file.

`pca_key` should normally remain `X_pca`: Scanpy writes newly computed PCA there.
`harmony_key` stores the Harmony-specific result, while `representation_key`
provides a stable key that downstream stages can select regardless of mode.
CellCharter and subclustering can preferentially discover
`X_batch_integration`, so a changed integration can affect later neighbourhood
or population results even though marker values were untouched.

## Selecting the batch annotation

The batch column is converted to a categorical variable for Harmony and is used
to partition neighbour searches for BBKNN. Missing, inconsistent, or overly
fine-grained labels change the scientific meaning of integration.

Inspect a cross-tabulation of the proposed batch key against case, biological
condition, tissue type, ROI, and known cell populations. Important warning signs
include:

- a condition found in only one batch;
- one batch containing a unique tissue or anatomical site;
- categories containing very few cells;
- batch labels that accidentally identify individual ROIs rather than a shared
  technical process;
- different spellings or missing values that create unintended categories; and
- large changes in panel composition or marker availability between batches.

Cells cannot be matched on a marker that was not measured comparably. Harmony
and BBKNN assume the PCA dimensions have the same meaning for every cell. They
do not harmonize antibody panels, impute missing markers, correct segmentation
bias, or distinguish a staining failure from a genuinely marker-negative state.

## How to assess integration

The stage produces UMAPs coloured by batch and by new Leiden labels. These are a
starting point, not sufficient evidence of success. UMAP is a nonlinear,
two-dimensional view of the graph; visual overlap can be produced by genuine
alignment, overcorrection, parameter choice, or projection artefact.

Evaluate at least three questions.

### 1. Has technical separation decreased?

Within a known biological population, cells from different technical batches
should no longer form reproducible batch-specific islands or graph partitions.
Compare unintegrated and integrated representations using the same marker set
and cell subset. If possible, quantify batch mixing locally rather than relying
only on the global UMAP.

### 2. Has biological identity been preserved?

Known lineages and states should retain coherent marker profiles and plausible
relationships. Inspect canonical protein expression independently of the
integrated coordinates. An integrated cluster containing cells with mutually
incompatible marker programs is not validated by having excellent batch mixing.
Rare or condition-specific states deserve particular attention because both
algorithms are encouraged to find cross-batch correspondence.

### 3. Are conclusions stable?

Compare `none`, Harmony, BBKNN, and—only if justified—`both`. Important
populations should not appear solely under one aggressive integration choice.
Assess several reasonable PCA dimensions, neighbourhood sizes, and Leiden
resolutions. Stability does not prove correctness, but major instability shows
that conclusions depend on analysis settings rather than robust biological
structure.

For a study with outcomes or treatment groups, check preservation without using
those outcomes as integration targets. Marker-level tests should use the
appropriate uncorrected or normalized quantitative values and a statistical
model that respects patients and technical batches; corrected PCs or a BBKNN
graph are not substitutes for such a model.

## Interpreting common failure patterns

### Residual batch separation

If comparable cells remain separated by batch, first check the batch labels,
marker normalization, panel consistency, and PCA-driving markers. Increasing
Harmony's diversity penalty or BBKNN connectivity may hide separation without
fixing its source. A batch-specific staining failure should normally be resolved
upstream or excluded with documented QC.

### Overcorrection

Overcorrection is suggested when distinct lineages merge, a known marker gradient
collapses, a batch-specific but biologically expected population is absorbed
into an unrelated state, or cluster identity becomes difficult to explain from
marker expression. Reduce the intervention, revisit confounding, or use an
unintegrated representation for that question.

### Forced connections from rare or absent populations

BBKNN always requests neighbours from each batch. If an equivalent state is
absent from one batch, the selected cross-batch neighbour may be biologically
wrong. The problem becomes more visible when a batch is very small, contains a
different tissue composition, or includes a unique phenotype.

### Batch-size imbalance

BBKNN balances the number of selected neighbours per batch rather than weighting
by the number of cells available in each batch. A small batch can therefore have
disproportionate influence on the graph. Harmony's expected cluster composition
accounts for batch abundance, but extreme imbalance still limits what can be
estimated reliably.

### Clusters that change with Leiden resolution

Leiden resolution controls how the final graph is partitioned; it is not an
integration-strength parameter. Higher values generally yield more communities,
but the relationship is data-dependent. Use multiple resolutions for sensitivity
analysis and assign biological meaning only after marker and spatial validation.

## Outputs

The updated AnnData normally contains:

- `adata.obsm['X_pca']`: newly computed PCA coordinates;
- `adata.obsm['X_pca_harmony']`: Harmony coordinates in `harmony` or `both`
  mode;
- `adata.obsm['X_batch_integration']`: the canonical downstream representation;
- `adata.obsp['distances']` and `adata.obsp['connectivities']`: the final Scanpy
  or BBKNN graph;
- `adata.obsm['X_umap']`: UMAP calculated from that graph;
- `adata.obs['leiden_<resolution>']`: optional graph clusters;
- `adata.uns['batch_integration']`: method, batch key, PCA dimensions,
  representation keys, graph route, method-specific parameters, Leiden keys,
  and QC path; and
- QC UMAPs in the configured batch-integration report subdirectory.

The QC figures show batch and Leiden composition but do not calculate a formal
integration score or establish biological preservation. Keep the unintegrated
representation or a reproducible pre-stage AnnData when method comparison or
auditability is important.

## Common limitations

- Harmony and BBKNN were introduced and benchmarked primarily for single-cell
  transcriptomics. Their use on IMC protein measurements is a transfer of the
  same representation principles and needs IMC-specific validation.
- Neither method can distinguish technical from biological variation under
  perfect confounding.
- Integration changes relationships between cells and can therefore change
  UMAP, Leiden, inferred populations, and downstream spatial neighbourhoods.
- Neither method corrects the quantitative marker matrix used for differential
  abundance or expression analysis.
- Batch mixing is not a sufficient objective: a representation in which all
  batches overlap but biological identities are erased is a failed integration.
- Package versions, approximate-neighbour implementations, and random seeds can
  change the result. Preserve the resolved configuration and environment with
  the AnnData provenance.

## References

- Korsunsky I, Millard N, Fan J, *et al.* [Fast, sensitive and accurate
  integration of single-cell data with Harmony](https://doi.org/10.1038/s41592-019-0619-0).
  *Nature Methods* 16, 1289–1296 (2019).
- Polański K, Young MD, Miao Z, Meyer KB, Teichmann SA, Park J-E.
  [BBKNN: fast batch alignment of single cell
  transcriptomes](https://doi.org/10.1093/bioinformatics/btz625).
  *Bioinformatics* 36, 964–965 (2020).
- [`harmonypy` source repository](https://github.com/slowkow/harmonypy).
- [BBKNN source repository](https://github.com/teichlab/bbknn).
- [Scanpy BBKNN API reference](https://scanpy.readthedocs.io/en/stable/generated/scanpy.external.pp.bbknn.html).
