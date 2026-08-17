# RAPIDS Processing

## What this stage does

This stage uses
[`rapids-singlecell`](https://github.com/scverse/rapids-singlecell) to perform a
standard single-cell representation and clustering workflow on an NVIDIA GPU.
For each normal run it can:

1. filter cells using a numeric AnnData observation;
2. calculate principal component analysis (PCA), or reuse an existing embedding;
3. optionally integrate technical batches with GPU Harmony;
4. construct a nearest-neighbour graph;
5. calculate a UMAP embedding;
6. run Leiden community detection at several resolutions; and
7. generate UMAP and marker-summary QC figures.

It can also repeat that workflow over a grid of PCA, neighbourhood, UMAP, and
Harmony settings. The processed AnnData stores the active representation, graph,
UMAP, Leiden labels, and detailed provenance.

RAPIDS is the computing platform, not a new biological model. PCA remains PCA,
UMAP remains UMAP, and Leiden remains graph community detection. The library
implements those operations with CuPy and NVIDIA's CUDA-X Data Science stack so
that arrays and computationally intensive operations run on a GPU. Acceleration
allows larger data and more parameter combinations to be explored; it does not
make an inappropriate marker set, batch definition, graph, or cluster
biologically valid.

## Why GPU acceleration is useful

Modern single-cell workflows repeatedly apply matrix decompositions, distance
calculations, graph construction, nonlinear optimization, and community
detection. These operations are highly parallel: the same arithmetic must be
performed for many cells, marker combinations, or graph edges. A GPU contains
many processing units designed to execute such operations concurrently.

[`rapids-singlecell`](https://arxiv.org/abs/2603.02402) represents dense and
sparse matrices with CuPy, uses cuML for algorithms including PCA and UMAP,
cuVS for nearest-neighbour search, and cuGraph for graph operations such as
Leiden clustering. Keeping intermediate results in GPU memory avoids repeatedly
copying large arrays between the CPU and GPU.

The supplied version 1 paper is an arXiv preprint based primarily on
single-cell transcriptomic benchmarks. In one reported workflow, preprocessing
and analysis of approximately 1.3 million cells took about 52 minutes on the
specified 32-core CPU workstation and about 25 seconds on the specified
high-end GPU, excluding input/output time. Harmony correction of an 11.4-million
cell PCA embedding completed in under 25 seconds on the tested GPUs, while the
CPU comparison was stopped after two hours. These results demonstrate potential
scaling, not the runtime expected for every IMC dataset, GPU, software version,
or configuration.

The same paper reports diminishing returns below roughly 50,000 cells because
GPU launch and CPU-to-GPU transfer overhead can approach or exceed the analysis
time. Many IMC datasets are much smaller and have far fewer features than atlas-
scale RNA sequencing datasets. For them, RAPIDS may still be useful for repeated
parameter scans or a consistent GPU workflow, but speed should be measured rather
than assumed.

## What this stage does not do

The active pipeline wrapper uses only part of the much larger
`rapids-singlecell` API. It does not normalize marker values, transform
distributions, select informative markers, scale markers to equal variance,
detect doublets, impute signal, annotate cell types, perform differential
testing, or calculate spatial neighbourhoods.

Except for optional removal of cell rows, it does not alter `adata.X` or marker
layers. Harmony changes PCA coordinates; neighbours create a graph; UMAP creates
a display; and Leiden assigns graph communities. These derived results can
strongly influence downstream population and spatial analyses even though the
underlying marker measurements remain unchanged.

Upstream quality remains decisive. Segmentation failures, missing channels,
inconsistent marker normalization, technical channels, and low-quality
antibodies can dominate the representation. GPU acceleration reproduces that
input structure efficiently; it cannot distinguish an artefact from biology.

## Optional cell filtering

Cell filtering occurs immediately after AnnData is loaded and before PCA,
Harmony, graph construction, or a parameter scan. It is disabled when both
`filter_min_value` and `filter_max_value` are null.

When either bound is set, the pipeline converts `adata.obs[filter_obs_key]` to a
numeric value. Cells below the minimum or above the maximum are removed. The
bounds themselves are inclusive. Cells with missing or non-numeric values are
also removed, and the stage stops if no cells remain.

The default filter column is `mask_area`. A lower bound can remove tiny
segmentation fragments, and an upper bound can remove grossly merged objects.
Mask area is also biological: lymphocytes are often smaller than epithelial or
malignant cells, cell size can change with activation, and multinucleated cells
may be genuinely large. A threshold selected from the pooled size distribution
can therefore alter phenotype frequencies, tissue composition, and spatial
relationships.

Review size distributions by ROI, batch, tissue compartment, and provisional
cell type before filtering. Report both the threshold and the numbers removed.
Filtering is applied once to the shared input before all parameter-scan copies,
so every scan evaluates the same retained cells.

## Choosing the input feature space

The stage has two mutually exclusive representation routes.

### Compute a new PCA

With `input_representation_key: null`, the stage transfers either `adata.X` or
the layer named in `pca_params.layer` to the GPU and calculates PCA. PCA converts
the correlated marker measurements into orthogonal axes. Early components
describe the greatest linear variation across cells; later components describe
progressively smaller patterns.

By default, PCA centres each marker but does not scale every marker to unit
variance. Markers with a large numerical range or variance can therefore
dominate. If a highly-variable-feature mask is already present, the installed
RAPIDS PCA implementation may use it by default. The stage itself does not
decide which markers are biologically suitable.

`n_for_pca` controls how many components are calculated. With null, the stage
requests one fewer component than the total marker count and clips the result to
what the total cell and marker counts permit. This clipping does not account for
a smaller pre-existing feature mask; lower `n_for_pca` if the selected feature
set contains fewer variables.

For IMC, retaining nearly every possible component is not automatically best.
Early components usually capture broad lineage and intensity structure. Later
components may retain rare states, but they can also represent noise, staining
variation, or individual problematic markers. Examine loadings and cluster
stability rather than treating the maximum valid component count as a target.

RAPIDS can perform sparse PCA without explicitly densifying the complete source
matrix. For typical IMC matrices, which contain far fewer features than RNA
sequencing, memory demand is more often driven by cell number, graph edges, and
intermediate GPU arrays than by marker count alone.

### Reuse an existing embedding

`input_representation_key` selects an existing matrix from `adata.obsm`, such as
a previously integrated or learned embedding. The stage then skips both PCA and
Harmony and proceeds directly to neighbours, UMAP, and Leiden.

This route is useful when another model already defines the feature space, but
its dimensions must have a consistent biological meaning across all cells.
`n_for_pca` and `pca_params` have no effect. `run_harmony` cannot be true because
the active implementation only applies Harmony to PCA that it computes itself.

Regardless of route, the selected coordinates are copied to
`adata.obsm['X_batch_integration']` by default. This stable key allows downstream
stages to discover the active representation without needing to know whether it
came from PCA, Harmony, or another model.

## Optional GPU Harmony integration

Harmony operates after PCA and before neighbour construction. It iterates
between diversity-aware soft clustering and cluster-specific correction of PCA
coordinates. Cells can belong partly to several soft clusters. Within those
states, Harmony estimates how technical batches are displaced and applies a
weighted cell-specific correction.

The result is a corrected embedding, not corrected marker abundance. Marker
values in `adata.X` and its layers remain unchanged. The same experimental-design
limitations described for the CPU batch-integration stage apply: Harmony cannot
distinguish technical from biological variation when they are confounded, and it
can erase real batch-associated biology if the batch key is inappropriate.

`batch_correction_obs` must identify an existing technical-batch column when
`run_harmony` is enabled. The pipeline converts it to a categorical observation.
If the column is supplied while Harmony is disabled, it is still validated and
used to colour QC UMAPs.

### Harmony 1 and Harmony 2

Current RAPIDS-singlecell releases expose two algorithm flavours:

- `harmony1` implements the original Harmony formulation from Korsunsky and
  colleagues.
- `harmony2`, the pipeline default, adds a stabilized diversity penalty, dynamic
  regularization for each cluster-batch pair, and pruning of batch-cluster
  combinations with negligible support. These changes are intended to reduce
  unstable or excessive corrections, especially when populations or batch sizes
  differ.

The flavour is always validated by the pipeline, even when `run_harmony` is
false. The stage also inspects the installed RAPIDS-singlecell Harmony API.
Modern runtimes receive `harmony1` or `harmony2` unchanged. The consolidated
RAPIDS 24.12 runtime exposes the older API: `harmony1` maps explicitly to its
original correction method, while `harmony2` is rejected rather than silently
substituted with a scientifically different algorithm. Use the newer legacy
RAPIDS environment when Harmony2 is required.

Important advanced parameters include:

- `theta`, the strength of the diversity pressure; larger values encourage more
  aggressive batch mixing;
- `tau`, which can reduce diversity pressure for small batches;
- `ridge_lambda`, the conservative ridge penalty used by `harmony1`;
- `alpha`, the scale of Harmony 2's dynamic regularization;
- `batch_prune_threshold`, below which poorly supported Harmony 2
  batch-cluster corrections are suppressed;
- `block_proportion`, the fraction of cells updated in each clustering
  sub-iteration; and
- `correction_method`, which trades memory against correction speed.

The pipeline default `harmony_params` uses up to 30 rounds, random seed zero,
verbose progress, and `float32`. RAPIDS documentation warns that 32-bit Harmony
can be numerically unstable in some cases. `float64` increases precision but
approximately doubles storage for those values and can reduce throughput. Check
convergence and biological stability before accepting the faster default.

## Nearest-neighbour graph

The active PCA, Harmony, or existing representation is a coordinate table. The
neighbour step turns it into a graph: each cell is a node, and edges connect
cells judged similar in the selected dimensions. Edge weights summarize local
connectivity and become the substrate for both UMAP and Leiden.

`n_pcs_neighbors` and `n_neighbors` answer different questions:

- `n_pcs_neighbors` selects how many coordinate dimensions contribute to the
  distance calculation. It is clipped to the width of the active
  representation.
- `n_neighbors` selects how many nearby cells define each local environment.
  Null uses the installed RAPIDS default, currently 15.

When a new PCA or Harmony representation is computed, null
`n_pcs_neighbors` uses every computed component. With an existing
`input_representation_key`, null delegates dimension selection to RAPIDS and
normally allows that representation to be used as supplied.

A small neighbour count emphasizes fine local structure and can isolate rare
states, but it also makes the graph sensitive to noise and sampling. A larger
count produces a smoother, more global topology and can bridge gradual states or
merge nearby populations. Neither choice is intrinsically more biological.

Current RAPIDS-singlecell defaults use exact brute-force neighbour search. The
library also provides approximate algorithms such as CAGRA, IVF, and
NN-descent through `neighbors_params.algorithm`. Approximation can greatly reduce
runtime and memory at large scale, but changed or missed neighbours can propagate
to UMAP and Leiden. Record the algorithm, metric, random seed, and package
version whenever results are compared.

## UMAP is a visualization, not a measurement

UMAP converts the neighbour graph into a usually two-dimensional display. It
tries to keep graph-connected cells nearby while placing weakly connected groups
apart. The displayed axes have no marker units, and distances between separated
islands are not a direct quantitative measure of biological dissimilarity.

`umap_min_dist` controls how closely neighbouring points may pack in the display.
Lower values produce tighter-looking islands; higher values spread local
neighbourhoods. It does not change the graph and therefore does not change
Leiden labels calculated by this stage. A more visually separated UMAP is not
evidence of better clustering.

UMAP optimization is stochastic. A fixed random seed improves repeatability,
but CPU and GPU implementations need not produce identical coordinates. Rotate,
reflect, or rearrange UMAP islands without changing their internal graph
relationships; interpret marker content and graph connectivity rather than axis
orientation.

## Leiden clustering

Leiden detects communities of cells that are more densely connected to one
another than to the rest of the neighbour graph. It operates on the graph, not
on the two-dimensional UMAP. Changing `umap_min_dist` alone therefore cannot
justify a different population assignment.

`leiden_resolutions_list` runs the graph partition at several granularities.
Higher resolution values generally produce more communities, but there is no
universal mapping from resolution to a biological cell type. Broad lineages may
be stable at low resolution, while activation states or technical subdivisions
appear only at higher values.

Every cluster must be interpreted from marker expression, tissue context,
replication, spatial location, and stability across reasonable settings. Leiden
optimizes a mathematical graph objective; it does not know antibody specificity,
cell lineage, or experimental outcome.

RAPIDS uses cuGraph for Leiden. `leiden_params` can control the random seed,
maximum iterations, edge weighting, numerical dtype, and refinement behaviour.
GPU parallel execution and atomic operations can alter the order of floating-
point accumulation or graph updates, so exact label identity should not be
expected across hardware and software versions.

## The implemented data flow

For a normal, non-scan run, the active `rapids` stage performs these operations:

1. Load `input_adata_path`, falling back to `general.anndata_path`, through the
   pipeline's AnnData stage-state controls.
2. Apply the optional numeric observation filter and copy retained cells when
   any are removed.
3. If no existing representation is selected, move `adata.X` or the configured
   layer to GPU memory and calculate PCA.
4. Optionally apply the selected GPU Harmony flavour to the PCA coordinates.
5. Copy the active coordinates to `representation_key`.
6. Construct the GPU neighbour graph from the configured number of dimensions.
7. Calculate GPU UMAP from that graph.
8. Optionally calculate GPU Leiden communities for every requested resolution.
9. Return GPU-backed matrices to CPU storage for Scanpy plotting and H5AD
   writing.
10. Generate UMAP and Leiden marker-summary QC, record provenance, and save the
    processed AnnData.

The stage normally updates `general.anndata_path` because both path overrides
are null. Filtering permanently removes cells from that output, and standard
derived keys are replaced. Set `output_adata_path` when the pre-RAPIDS AnnData
must remain as a separate reusable file.

Custom storage keys can preserve parallel graph or display results:

- null `neighbors_key` writes standard `adata.uns['neighbors']` plus the normal
  distance and connectivity matrices; a custom key creates a named graph;
- null `umap_key` writes `adata.obsm['X_umap']`; a custom key stores a separate
  embedding; and
- `pca_key`, `harmony_key`, and `representation_key` control the corresponding
  coordinate matrices.

Leiden labels are still named `leiden_<resolution>`, even when a custom graph or
UMAP key is used. Matching existing labels are overwritten.

## Pass-through parameter dictionaries

The five `*_params` dictionaries expose advanced RAPIDS-singlecell options
without adding a first-class configuration field for every library argument.
Null-like values are removed before each call.

Some arguments are managed by dedicated pipeline fields and are deliberately
ignored when repeated inside a dictionary:

- `pca_params`: `n_comps`, `key_added`, and `copy` are managed;
- `harmony_params`: `key`, `basis`, `adjusted_basis`, `flavor`, and
  `correction_method` are managed;
- `neighbors_params`: `n_neighbors`, `n_pcs`, `use_rep`, `key_added`, and `copy`
  are managed;
- `umap_params`: `min_dist`, `key_added`, `neighbors_key`, and `copy` are
  managed; and
- `leiden_params`: `resolution`, `key_added`, `neighbors_key`, and `copy` are
  managed.

The stage logs a warning when it discards a duplicated managed option. This
prevents a hidden dictionary value from contradicting the documented top-level
field. Other dictionary arguments are passed to the installed library, so
accepted names and behaviour remain version-dependent.

## Parameter scanning

`parameter_scan_dict` accepts lists for five fields:

- `n_neighbors`;
- `n_for_pca`;
- `umap_min_dist`;
- `run_harmony`; and
- `harmony_flavor` (`harmony_flavour` is accepted as a spelling alias).

The stage calculates the Cartesian product. Two PCA counts, three neighbour
counts, and two UMAP distances create 12 complete runs. Each run starts from a
copy of the same filtered AnnData and receives its own QC directory.

The summary CSV records settings, cell and marker counts, representation size,
method, output keys, Leiden keys, and the number of MatrixPlots. It does not
calculate a ground-truth clustering score, select a winner, or verify biological
preservation.

By default, scan mode saves only QC and
`rapids_parameter_scan_summary.csv`. It returns after the scan and does not
write the normal canonical AnnData. With `parameter_scan_save_anndata: true`,
each combination is saved to a separately suffixed H5AD; there is still no
automatically chosen canonical result. After comparing representative tissues
and settings, clear the scan dictionary and run one normal configuration to
produce the downstream asset.

Some combinations are structurally incompatible. A scan that sets
`run_harmony: true` requires `batch_correction_obs` and cannot be used with
`input_representation_key`. Scanning `n_for_pca` has no effect when an existing
representation bypasses PCA.

## QC outputs and biological interpretation

The stage generates batch-coloured UMAPs when a batch key is available and
Leiden-coloured UMAPs for every resolution. It also creates a MatrixPlot for
each Leiden result. The MatrixPlots summarize marker expression without
standard-scaling rows and cap the display at `visualization.matrixplot_vmax`;
markers are reordered by similarity where possible. Display clipping affects
the figure, not AnnData values.

Review at least the following:

- **Marker coherence:** do cells grouped together share a biologically
  interpretable protein program?
- **Technical composition:** within a validated phenotype, are clusters or graph
  islands dominated by staining batch, acquisition batch, ROI, or another
  technical variable?
- **Rare populations:** do expected small populations remain distinct across
  reasonable PCA and neighbour settings?
- **Segmentation bias:** are clusters mainly separated by mask area, DNA
  intensity, or other object-quality measurements?
- **Resolution stability:** do broad conclusions persist across nearby Leiden
  resolutions, rather than appearing at only one convenient setting?
- **Cross-method stability:** when scientific conclusions matter, are they
  supported by a CPU or independent integration route as well as one GPU run?

The paper reports that deterministic functions agreed with CPU baselines within
floating-point tolerance and that stochastic outputs retained consistent
biological interpretations in its tested transcriptomic datasets. Exact
numerical reproducibility was not expected because GPU parallelization,
floating-point precision, atomic operations, and stochastic algorithms can all
change individual values or labels. Equivalent benchmark performance does not
guarantee that an IMC clustering is biologically correct.

When comparing two runs, do not compare Leiden label numbers directly. Cluster
`3` in one run has no necessary relationship to cluster `3` in another. Match
clusters by cell membership and marker phenotype, and quantify agreement where
appropriate.

## Outputs

A normal run can add or update:

- the filtered set of rows in the saved AnnData;
- `adata.obsm[pca_key]`: GPU PCA coordinates when PCA is run;
- `adata.obsm[harmony_key]`: corrected PCA when Harmony is run;
- `adata.obsm[representation_key]`: the canonical active representation;
- standard or named neighbour distance and connectivity matrices;
- `adata.obsm[umap_key or 'X_umap']`: the RAPIDS UMAP;
- `adata.obs['leiden_<resolution>']`: optional communities;
- `adata.uns['rapids_process']`: detailed settings, source and output keys,
  dimensions, QC paths, and cell/marker counts;
- `adata.uns['batch_integration']`: a compact compatibility summary for
  downstream representation discovery; and
- UMAPs, Leiden MatrixPlots, and optional scan summaries in the active report
  directory.

The saved H5AD contains CPU-compatible arrays because the stage transfers GPU
outputs back before writing. The reusable AnnData is separate from the managed
human-facing report.

## Hardware, memory, and reproducibility limitations

- The active wrapper requests one NVIDIA GPU. Although RAPIDS-singlecell supports
  Dask, out-of-core, and multi-GPU workflows, this stage does not automatically
  configure the paper's multi-GPU setup.
- The `rapids_singlecell` environment is marked as externally managed and has no
  repository specification or lockfile. Capture its RAPIDS-singlecell, CUDA,
  CuPy, cuML, cuGraph, cuVS, AnnData, and Scanpy versions for every consequential
  analysis.
- GPU memory must hold the active data and algorithm intermediates. Host-memory
  spilling can allow an oversized analysis to continue but may remove much of
  the speed advantage.
- Sparse GPU arrays often use 32-bit indices, which can impose size limits based
  on non-zero entries as well as cells. Atlas-scale claims from a Dask workflow
  do not automatically apply to this single-GPU wrapper.
- CPU-to-GPU and GPU-to-CPU transfers are real costs. They can dominate small
  analyses or workflows that frequently fall back to CPU functions.
- Approximate neighbour search and 32-bit arithmetic trade precision for speed
  and memory. Validate conclusions rather than assuming a GPU result must match
  a CPU result exactly.
- The supplied paper benchmarks RNA-sequencing data. IMC has fewer features,
  different noise, segmentation-derived observations, and different biological
  use cases; both runtime and analytical behaviour require IMC-specific review.

## References

- Dicks S, Heumos L, May L, *et al.* [GPU-accelerated single-cell analysis at
  scale with rapids-singlecell](https://doi.org/10.48550/arXiv.2603.02402).
  arXiv version 1 preprint (2026).
- [RAPIDS-singlecell source repository](https://github.com/scverse/rapids-singlecell).
- [RAPIDS-singlecell documentation](https://rapids-singlecell.readthedocs.io/).
- Korsunsky I, Millard N, Fan J, *et al.* [Fast, sensitive and accurate
  integration of single-cell data with Harmony](https://doi.org/10.1038/s41592-019-0619-0).
  *Nature Methods* 16, 1289–1296 (2019).
- McInnes L, Healy J, Melville J. [UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction](https://doi.org/10.48550/arXiv.1802.03426).
- Traag VA, Waltman L, van Eck NJ. [From Louvain to Leiden: guaranteeing
  well-connected communities](https://doi.org/10.1038/s41598-019-41695-z).
  *Scientific Reports* 9, 5233 (2019).
