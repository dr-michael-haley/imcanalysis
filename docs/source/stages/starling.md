# STARLING segmentation-aware phenotyping

```{warning}
STARLING is a **new addition to SpatialBiologyToolkit**. Although the published
method has been evaluated on IMC and PhenoCycler data, this pipeline integration
has not yet been tested extensively enough to establish when it improves the
biological conclusions from our own datasets. Treat its labels and segmentation-
error probabilities as exploratory outputs. Compare them with the input
clustering, marker images, segmentation masks, and established phenotyping
workflow before using them for abundance or spatial inference.
```

## What this stage does

STARLING (SegmenTation AwaRe cLusterING) is a probabilistic phenotyping method
for highly multiplexed tissue images. It addresses a specific problem: the
marker profile assigned to one segmented object does not always originate from
one intact biological cell. A mask may cover parts of two cells, include signal
spilling over from a neighbour, capture a cellular projection, or intersect
several cells through the thickness of a tissue section. Conventional clustering
can mistake these mixed profiles for genuine cell states.

The stage takes an already segmented, quantified cell-by-marker matrix and an
initial clustering. It fits a model in which each observed segment can be either:

- an error-free **singlet**, represented by one latent phenotype; or
- a **segmentation error**, represented approximately as a mixture of two latent
  phenotypes.

For every observed segment, the pipeline writes:

- its most likely phenotype under the assumption that it is a singlet;
- a probability that its profile arose from a segmentation error;
- a binary error call obtained by thresholding that probability; and
- optional posterior matrices describing confidence in each singlet phenotype
  and each pair of phenotypes that could contribute to an error.

STARLING also refines the marker-expression centroid of every initialized
phenotype. The number of model components comes from the initialization; this
stage does not independently discover the appropriate number of phenotypes.

STARLING does **not** change the segmentation masks, identify which pixels belong
to which real cell, or write corrected marker values for individual cells. Its
“denoising” is at the level of inferred phenotype centroids and assignments, not
at the level of the image or the saved expression matrix.

## Why segmentation errors complicate biology

IMC and related assays reduce a tissue image to one marker vector per mask. That
summary is only a true single-cell measurement when the mask corresponds closely
to one cell. Several physical and computational effects violate this assumption:

- two adjacent cells can be merged into one instance mask;
- a boundary can include cytoplasm or membrane from a neighbouring cell;
- lateral signal spread can contaminate an otherwise correct mask;
- a thin two-dimensional section can contain overlapping parts of cells at
  different depths;
- elongated processes can enter another cell's mask; and
- imperfect nuclear or membrane contrast can shift the inferred boundary.

If the affected cells have different identities, the resulting marker vector may
contain an implausible combination such as epithelial and leukocyte lineage
markers. Ordinary clustering assumes every row is a valid cell and can therefore
create a stable “hybrid” cluster from repeated technical mixtures. Apparent rare
cell types, transitional states, and spatially restricted populations are
particularly vulnerable to this error.

Not every mixed-looking profile is technical. Real biology can produce lineage
co-expression during differentiation, activation-dependent marker changes,
cell-cell material transfer, phagocytosis, and genuinely multinucleated cells.
STARLING estimates compatibility with its statistical error model; it does not
prove that a particular biological object is an accidental doublet. Image review
and biological context remain essential.

## How the STARLING model works

### Starting phenotypes

STARLING requires an initial partition of the cells. For each starting cluster,
it calculates a marker-expression centroid and a separate variance for each
marker. These values initialize the latent phenotypes. During training, the
centroids, dispersions, relative phenotype frequencies, relative frequencies of
phenotype pairs, and overall singlet/error mixture are optimized jointly.

The starting partition is therefore influential. It determines the number of
components and gives each component its initial biological meaning. STARLING can
move centroids and reassign cells, but it does not provide a model-selection step
that establishes whether the starting resolution contains too many or too few
cell types. A poor initialization can be refined into another poor solution.

### Singlet expression model

For a segment treated as error-free, every phenotype has an expected abundance
for each selected marker and a marker-specific dispersion. Conditional on the
phenotype, marker measurements are modelled independently. The default `T`
likelihood is a Student-t distribution with two degrees of freedom. Its heavy
tails make isolated extreme measurements less influential than they would be
under a Normal likelihood. `N` instead selects a Gaussian model.

The diagonal, marker-independent likelihood is computationally practical but is
an approximation. Biological co-expression between markers is represented mainly
through their shared phenotype centroid, not through a full within-phenotype
covariance model.

### Segmentation-error expression model

For an erroneous segment, STARLING considers every possible pair of latent
phenotypes. If phenotypes *c* and *c′* contribute to an error, the expected marker
profile is modelled as the average of their two centroids; their marker-specific
dispersions are averaged in the same way. The model integrates over all phenotype
pairs to obtain the cell's overall segmentation-error probability.

This is intentionally an approximate model. Real masks can contain unequal
contributions from two cells, more than two cells, spillover affecting only some
markers, or extracellular background. STARLING assumes an equal pairwise mixture
rather than reconstructing that full physical process. Averaging across every
pair can nevertheless give non-zero support to more complicated errors, and the
paper reported useful performance on simulated mixtures of three phenotypes.

### Cell size

When `model_cell_size` is enabled, mask area supplies a second source of evidence.
Each singlet phenotype has a learned characteristic area and dispersion. A
combined segment is expected to be larger, although the area visible in a tissue
section depends on how much its constituent cells overlap in the z-plane. In the
published overlap formulation, an error's area can lie between the larger of the
two constituent areas and their sum.

Cell size can help distinguish a merged mask from a small single cell with a
mixed expression profile, but it is not neutral morphology. Lymphocytes,
macrophages, tumour cells, stromal cells, cell-cycle states, sectioning depth, and
segmentation settings can all have different area distributions. The model learns
phenotype-specific sizes to account for some of this variation; unusual size is
still evidence relative to the fitted model, not proof of an error.

```{important}
The supplied upstream STARLING 0.1.4 source contains a version-specific mismatch:
it validates `model_zplane_overlap` as a Boolean, but its posterior calculation
compares that value with the string `"Y"`. With the Boolean passed by this
wrapper, the reviewed version uses the summed-size likelihood during fitting and
prediction, although the option still changes how synthetic error areas are
generated. Treat the intended overlap model as an implementation caveat until
this upstream mismatch is resolved.
```

### Synthetic-error regularisation

Likelihood fitting alone can orient mixture components in unhelpful ways. During
training, STARLING therefore creates an auxiliary labelled dataset from the real
measurements:

- synthetic singlets are sampled from existing cell profiles; and
- synthetic errors are produced by averaging two sampled expression profiles.

When cell area is used, synthetic error areas are either summed or sampled
between the larger constituent area and the sum, depending on the overlap option.
Synthetic singlets and errors occur in equal numbers in this auxiliary dataset.

The model is trained to maximize the likelihood of the observed data while also
separating these simulated singlets and errors using binary cross-entropy. The
`model_regularizer` is the multiplier on this auxiliary loss. Larger values make
the simple synthetic-error construction more influential; smaller values favour
fit to the observed data. It is a loss weight, not a target error fraction.

The paper denotes this weight by λ and recommends `0.1` from its benchmarks.
**SpatialBiologyToolkit currently defaults to `1.0`.** That tenfold difference
should be treated as an explicit analysis choice, especially while the stage is
being evaluated. The best value can depend on data scaling, mask quality, marker
panel, and tissue.

### Singlet prior and model fitting

`singlet_prop` is documented by STARLING as the initial probability that an
observed segment is an error-free singlet. The published 0.6 setting is intended
to correspond to a 40% segmentation-error prior. It does not force a fixed final
error fraction: the mixture prior is trainable and the binary calls also depend
on `doublet_threshold`.

```{important}
In the supplied upstream STARLING 0.1.4 code, the initialized mixture vector is
`[1 - singlet_prop, singlet_prop]`, while the first branch is subsequently used
for singlets and the second for errors. The reviewed implementation therefore
initializes `singlet_prop: 0.6` as 40% singlet and 60% error, opposite to the
documented meaning. This may be corrected in another installed version; record
the exact STARLING version and verify its source before interpreting or tuning
this parameter.
```

STARLING uses Adam optimization and fixed mini-batches of 512 cells. This wrapper
seeds the random-number generators and fits one model to all selected cells,
stopping after `max_epochs` or Lightning early stopping. The paper commonly ran
several initializations and selected the fit with the lowest likelihood score;
the current pipeline does **not** perform that repeated-fit selection. Stability
across seeds and initial clusterings must therefore be assessed with separate
runs.

## What SpatialBiologyToolkit actually runs

### Expression and marker selection

The stage reads `adata.X` unless `use_layer` names an alternative `adata.layers`
matrix. `marker_include` can define an ordered subset and `marker_exclude` then
removes exact names. At least 10 cells and 10 selected features are required.

The selected matrix is converted to a dense, 64-bit floating-point array. It must
contain only finite, non-negative values. Tiny negative floating-point residuals
can be clipped to zero within `negative_value_tolerance`; larger negatives stop
the stage. Sparse input therefore becomes dense and may require substantial host
memory.

The wrapper performs no normalization, arcsinh transformation, scaling, batch
correction, or marker quality filtering. STARLING receives the numbers exactly as
stored in the chosen matrix. This is scientifically important because likelihoods
and synthetic averages depend on feature scale. Do not use a signed integrated
embedding such as Harmony coordinates as the expression input. Select a
non-negative, biologically interpretable cell-by-marker representation and record
how it was produced.

Markers should be chosen for the phenotypes the model is expected to resolve.
DNA, acquisition-control, highly technical, or uninformative channels can dominate
the likelihood without helping cell identity. Conversely, removing all lineage
markers leaves the model unable to distinguish the mixtures of interest. If
state markers are included, final components may represent activation or
functional states as well as major cell lineages.

### Initialization choices

`initial_clustering_method` supports five upstream modes:

- `User` uses a complete `adata.obs` label column. The wrapper selects
  `initial_label_obs`, falling back to `general.population_obs_primary`, and
  encodes its categories as consecutive integers.
- `KM` runs K-means with `n_clusters` components.
- `GMM` runs a diagonal-covariance Gaussian mixture with `n_clusters` components.
- `FS` runs FlowSOM with `n_clusters` meta-clusters.
- `PG` runs PhenoGraph and determines its cluster count through that method.

`User` is the most direct way to refine an existing Leiden or curated clustering.
The source labels are retained separately, and a CSV mapping records each source
category's integer initialization. The final integer label refers to the fitted
component seeded by that category, but the centroid may move during training; do
not assume that the old biological name remains correct without comparing the
initial and final centroids.

The built-in modes operate directly on the same unscaled matrix supplied to
STARLING. This can make high-range markers disproportionately important. The
upstream FlowSOM implementation also creates an intermediate `fs.csv` in the
working directory.

### Cell-size input

With cell-size modelling enabled, the wrapper first looks for
`cell_size_col_name`, which defaults to `adata.obs["mask_area"]`, and then searches
`cell_size_fallback_cols`, which defaults to `area`. The selected column must be
numeric, complete, and strictly positive. Its unit is not converted, so all cells
in a joint fit must use comparable mask-area units and segmentation conventions.

### Training controls

The current default is one seeded fit for at most 100 epochs, with early stopping
on `train_loss`. That loss is the observed-data negative log-likelihood plus the
weighted synthetic binary-cross-entropy loss. There is no held-out validation
loop in this wrapper, so early stopping detects optimization plateau rather than
generalization to unseen biological samples.

Lightning selects CPU or GPU automatically unless `trainer_accelerator` and
`trainer_devices` override it. TensorBoard logs are written below the STARLING QC
directory by default. `limit_train_batches` is mainly useful for diagnostics; a
model fitted to only part of each epoch should not be treated as a complete
biological result without deliberate validation.

## Reusable assets produced

With the default prefix `starling`, the saved AnnData receives the following
columns in `adata.obs`:

- `starling_source_label`: the original user-supplied starting category, present
  only for `User` initialization;
- `starling_init_label`: the integer component used to initialize STARLING;
- `starling_label`: the fitted component with the largest singlet-and-component
  posterior for that observed segment;
- `starling_doublet_prob`: the posterior probability of the segmentation-error
  state;
- `starling_doublet`: `1` when that probability is greater than
  `doublet_threshold`, otherwise `0`; and
- `starling_max_assign_prob`: the largest **joint** probability of being a
  singlet and belonging to one particular component.

The names use “doublet” for compatibility with upstream STARLING, but the model's
state is broader than a literal two-cell doublet. It can respond to spillover,
section overlap, projections, and other profiles resembling phenotype mixtures.

When `store_assignment_prob_matrix` is enabled,
`adata.obsm["starling_assignment_prob_matrix"]` is an *N* by *C* matrix containing
the joint probabilities `P(singlet and component c | data)`. A row sums to the
cell's singlet probability, not to one. Consequently,
`starling_max_assign_prob` combines confidence that the object is a singlet with
confidence about which phenotype it resembles. Conditional phenotype
probabilities would require dividing that row by its singlet probability.

When `store_gamma_assignment_prob_matrix` is enabled,
`adata.obsm["starling_gamma_assignment_prob_matrix"]` is an *N* by *C* by *C*
matrix of joint posterior support for every ordered phenotype pair in the error
state. Its entries sum to the segmentation-error probability. This can help ask
which two phenotypes best explain a suspicious profile, but its size grows as
*N* × *C*² and it is disabled by default.

The full-marker `adata.varm` receives:

- `starling_init_exp_centroids`;
- `starling_init_exp_variances`; and
- `starling_exp_centroids` for the fitted phenotypes.

Rows for markers excluded from the fit contain missing values. Cell-size
centroids and the resolved input, marker set, initialization, paths, and output
key mapping are recorded in `adata.uns["starling"]`.

By default only prefixed keys are written. Enabling
`write_canonical_starling_keys` also writes upstream names such as `st_label`,
`doublet_prob`, and `assignment_prob_matrix`; these generic keys can overwrite a
previous STARLING run.

If `save_model` is enabled, the complete trained Lightning object is serialized
with PyTorch. A relative `model_output_name` is resolved as a project asset. Such
objects can be sensitive to Python and package versions, so the AnnData outputs
and configuration remain the more portable scientific record.

## Human-facing outputs

With QC tables enabled, the STARLING QC directory contains:

- the source-label to initialization mapping;
- initial expression centroids and variances;
- fitted expression centroids;
- final component counts and fractions;
- binary segmentation-error counts by component;
- a one-row run summary; and
- a per-cell CSV containing available ROI, case, group, source-label, fitted-label,
  error-probability, error-call, and maximum-assignment fields.

With QC plots enabled, it also contains histograms of segmentation-error
probability and maximum joint assignment probability, plus a bar plot of fitted
component sizes. These are diagnostic summaries, not evidence that the inferred
phenotypes are biologically correct. The stage does not currently create marker
heatmaps, spatial maps of suspicious cells, image overlays, multi-seed stability
plots, or comparisons with the input clustering.

## How to interpret the results biologically

### Fitted phenotype labels

`starling_label` is STARLING's best phenotype for the measured object **under the
singlet branch of the model**, even when that object has a high error probability.
This makes it possible to retain all objects for some downstream analyses, but it
does not mean STARLING has physically unmixed an erroneous segment. Inspect the
fitted centroids and assign biological names from marker combinations, not from
the integer identifier alone.

Compare initial and fitted centroids. A useful refinement may remove implausible
lineage co-expression, sharpen lineage-defining markers, and preserve known rare
populations. A concerning fit may collapse related states, create pan-positive
components, or substantially reinterpret most cells without support in the
images.

### Segmentation-error probability

`starling_doublet_prob` is relative support for STARLING's mixture-like error
model. Values near one indicate that a phenotype-pair explanation fits the
selected expression and optional area better than any single fitted phenotype.
Values near zero favour an error-free component. They are not externally
calibrated probabilities of mask failure and should not be read as a measured
error rate.

Map high-probability objects back to their ROIs. Useful supporting evidence
includes multiple nuclei in one mask, concave or oversized masks, boundary
spillover, location at a dense heterotypic interface, and marker combinations
matching adjacent cell types. High values concentrated in one acquisition batch,
ROI, tissue compartment, or marker channel can instead reveal preprocessing or
batch artefacts.

### Two defensible downstream strategies

The STARLING paper describes two broad interpretations:

1. remove objects above a chosen error-probability threshold and analyse the
   fitted labels among the retained objects; or
2. retain every object and use its most likely singlet phenotype as a best guess.

The first is conservative about cell identity but can bias abundance and spatial
analyses if error probability varies by cell type, cell size, tissue density, or
compartment. The second preserves sampling density but treats uncertain objects
as if one phenotype were dominant. For important conclusions, report sensitivity
to both strategies and to more than one threshold rather than presenting the
default 0.5 call as a biological boundary.

## Suggested validation before routine use

Because this pipeline stage is new and not yet extensively tested, a practical
evaluation should include at least:

1. **Image-level review.** Overlay high- and low-probability objects on marker
   images and masks across several ROIs, tissues, and phenotype pairs.
2. **Centroid plausibility.** Compare initial and fitted marker centroids using
   known mutually exclusive and conditionally co-expressed lineage markers.
3. **Stability.** Repeat fits with several seeds, initialization methods or
   resolutions, and at least the paper's `model_regularizer: 0.1` alongside the
   pipeline default.
4. **Technical stratification.** Plot error probabilities and fitted labels by
   ROI, batch, acquisition, area, and segmentation-QC variables.
5. **Biological preservation.** Confirm known cell types, rare populations, and
   condition-associated states have not disappeared solely because they resemble
   the model's synthetic averages.
6. **Downstream sensitivity.** Recalculate abundance and spatial conclusions
   after excluding probable errors, retaining best-guess labels, and using the
   original phenotyping as a comparator.

These checks determine whether STARLING adds useful information for a particular
study. Good optimization loss or visually tidy clusters alone is insufficient.

## Important configuration options

A cautious first run might be configured as:

```yaml
starling:
  input_adata_path: null
  output_adata_path: null
  use_layer: null
  marker_include: null
  marker_exclude: [DNA1, DNA2]

  initial_clustering_method: User
  initial_label_obs: null
  n_clusters: null

  seed: 10
  dist_option: T
  singlet_prop: 0.6
  model_cell_size: true
  cell_size_col_name: mask_area
  cell_size_fallback_cols: [area]
  model_zplane_overlap: true

  # The paper recommends 0.1; the pipeline's current default is 1.0.
  model_regularizer: 0.1
  learning_rate: 0.001
  doublet_threshold: 0.5

  max_epochs: 100
  early_stopping: true
  tensorboard_logging: true

  output_prefix: starling
  write_canonical_starling_keys: false
  store_assignment_prob_matrix: true
  store_gamma_assignment_prob_matrix: false
  save_model: false
  save_qc_tables: true
  save_qc_plots: true
```

Change one scientifically important setting at a time and use a different
`output_prefix` or output AnnData for each comparator. In particular, record the
input representation, marker set, initialization labels and resolution, seed,
regularisation weight, cell-size setting, and threshold.

## Common problems and limitations

- **This integration is exploratory.** Published performance does not establish
  benefit for the pipeline's panels, preprocessing, Cellpose-derived masks, or
  biological questions.
- **No mask correction is performed.** A high error probability does not produce
  two replacement cells or repair spatial coordinates.
- **No per-cell expression deconvolution is produced.** `adata.X` and layers are
  unchanged, so fitted labels do not make the original marker values safe to
  interpret as pure-cell abundances.
- **The pairwise equal-mixture model is deliberately simplified.** Unequal
  mixtures, more than two cells, marker-specific spillover, and background are
  represented only approximately.
- **Spatial relationships are not model inputs.** Apart from cell area, STARLING
  does not use coordinates, neighbours, boundaries, or tissue compartments.
- **Initialization fixes model complexity.** The fitted result inherits the
  starting cluster count and can be sensitive to starting labels and random seed.
- **The wrapper performs one fit.** It does not reproduce the paper's repeated-run
  selection procedure automatically.
- **Scaling and batch effects matter.** No transformation or batch adjustment is
  performed, and technical shifts can be absorbed into phenotypes or error
  probabilities.
- **Synthetic errors are an assumption.** Averaging randomly selected real cells
  may not match the composition or frequency of errors in a particular tissue.
- **Error calls can be phenotype-dependent.** Removing all called errors can
  preferentially deplete large cells, dense compartments, interacting cell
  types, or true hybrid states.
- **Posterior arrays can be large.** Dense input, the *N* by *C* singlet matrix,
  and especially the optional *N* by *C* by *C* pair matrix require substantial
  memory.
- **Saved models are version-sensitive.** Retain the output AnnData, configuration,
  environment details, and QC tables even when the model object is serialized.

## References

- Lee, J. Y. H. *et al.* [Segmentation aware probabilistic phenotyping of
  single-cell spatial protein expression data](https://doi.org/10.1038/s41467-024-55214-w).
  *Nature Communications* **16**, 389 (2025).
- [STARLING source repository](https://github.com/camlab-bioml/starling)
- [STARLING package documentation](https://camlab-bioml.github.io/starling/)
