# BioBatchNet integration

## What this stage does

BioBatchNet learns two low-dimensional representations of every cell from its
marker-intensity profile:

- a **biological embedding**, intended to retain cell-state and cell-type signal
  while removing information that identifies the technical batch; and
- a **batch embedding**, intended to retain the variation associated with the
  supplied batch label.

SpatialBiologyToolkit stores these representations rather than replacing the
marker-expression matrix with "corrected" intensities. The biological embedding
is then used, by default, to construct a nearest-neighbour graph, a UMAP, and
Leiden clusters. This makes BioBatchNet an integration stage: its main product is
a shared coordinate system in which cells from different technical runs can be
compared.

The stage is supervised only by the batch labels. It does not need cell-type
annotations, disease labels, or spatial coordinates during training.

## Why batch correction is needed

Imaging mass cytometry (IMC) measurements can differ between staining rounds,
acquisition runs, instruments, reagent lots, sites, or processing dates. Such
technical differences may shift the measured intensity of many markers at once.
Cells can therefore group by experimental run rather than by phenotype, obscuring
shared populations and making apparent population differences difficult to
interpret.

Correction has two competing objectives:

1. remove enough technical information for comparable cells to occupy the same
   biological neighbourhoods; and
2. preserve genuine differences in cell identity, activation, tissue state, and
   abundance.

Under-correction leaves cells separated by batch. Over-correction forces genuinely
different cells together and can erase the biological effect under study. A useful
embedding is therefore not simply the one with the most visually mixed batches.
It is the one that mixes batches *within comparable biological populations* while
retaining known and independently supported biology.

## How BioBatchNet separates biological and batch variation

BioBatchNet is a dual-encoder variational autoencoder (VAE). Both encoders receive
the same per-cell marker vector, but the training objectives encourage them to
retain different information.

### Biological encoder and adversarial batch removal

The biological encoder maps each cell to a distribution in a biological latent
space. A sample from that distribution is passed to a batch discriminator, which
tries to predict the cell's batch. A gradient-reversal layer changes what the
biological encoder learns: the discriminator is trained to improve its batch
prediction, while the encoder receives the opposite gradient and is trained to
make that prediction fail. Consequently, the biological embedding is pressured
to become uninformative about batch.

The `discriminator` loss weight controls this pressure. In the method described
in the paper and implemented by the pinned dependency, a larger weight applies
stronger adversarial correction. Too little weight can leave batch separation;
too much can remove biological structure that is associated with batch. The
authors' reported sensitivity analysis illustrates this trade-off, but its best
value is dataset-specific and should not be transferred blindly.

### Batch encoder and explicit batch retention

The second encoder produces a batch-specific latent representation. A supervised
classifier is trained to predict the supplied batch label from this representation.
This gives the model somewhere to retain technical variation instead of requiring
the biological embedding to carry it.

The decoder reconstructs the original marker vector from the concatenated
biological and batch embeddings. The batch embedding is detached from the
decoder's reconstruction gradient in BioBatchNet's design, so its principal
training signal remains the explicit batch-classification objective. The
biological encoder must retain enough cell-level signal, together with the
batch-specific representation, to reconstruct the input.

### Regularisation and independence of the two embeddings

Several losses act together during training:

- **Reconstruction loss** penalises differences between observed and reconstructed
  IMC marker values. This encourages preservation of information from the input.
- **Adversarial discriminator loss** removes batch-predictive information from the
  biological embedding through gradient reversal.
- **Batch-classifier loss** makes the batch embedding predictive of batch.
- **Two KL-divergence losses** regularise the biological and batch latent
  distributions towards standard normal distributions, as in a conventional VAE.
- **Orthogonality loss** penalises cross-covariance between centred biological and
  batch embeddings. This reduces linear dependence between the two spaces.

These objectives are optimised jointly. Their weights are multipliers on losses
with different numerical scales, so the weights should not be compared as if they
were percentages. Increasing preservation pressure can limit batch removal, while
increasing adversarial pressure can sacrifice biology. The correct balance must be
assessed empirically for the dataset and scientific question.

## Evidence reported in the paper

The authors evaluated BioBatchNet on three labelled IMC datasets with different
scales and batch structures: IMMUcan (47,794 cells, 4 batches, and 10 annotated
cell types), Damond (252,059 cells, 12 batches, and 15 cell types), and Hoch
(307,931 cells, 39 batches, and 9 cell types). They compared BioBatchNet with six
other correction approaches.

The evaluation deliberately separated two objectives. Batch removal was assessed
with integration local inverse Simpson's index (iLISI), graph connectivity,
batch-average silhouette width, and principal-component regression. Biological
conservation was assessed with cell-type silhouette width and agreement between
clusters and known cell labels using adjusted Rand index (ARI) and normalised
mutual information (NMI). Across these tests, the authors reported a favourable
balance between batch mixing and preservation of annotated biology rather than
optimising mixing alone.

These results establish comparative evidence on the studied datasets, not a
guarantee for a new experiment. The benchmarks had reference cell labels that made
biological preservation measurable; many real projects do not. Differences in
panel composition, input scaling, batch imbalance, tissue, or study design can
change the result. In particular, the paper's IMC experiments used the measurements
in raw form, whereas this pipeline trains on whatever matrix the preceding stages
placed in `adata.X` unless `use_raw` is enabled.

## What SpatialBiologyToolkit actually runs

The pipeline uses the BioBatchNet API pinned in the BioBatchNet environment lock,
whose configuration names pre-date the current upstream interface. It passes a
dense cell-by-marker matrix and integer-encoded batch labels to
`correct_batch_effects`, then retains the two returned embeddings.

The BioBatchNet paper also describes a separate constrained pairwise clustering
(CPC) method that uses must-link and cannot-link pairs. **This pipeline stage does
not run CPC.** Its optional clusters are ordinary Scanpy Leiden clusters computed
from a neighbour graph in the BioBatchNet biological embedding. They should not be
described as BioBatchNet CPC clusters.

The stage does not use tissue coordinates, neighbourhood composition, or image
features. It therefore cannot distinguish a spatially restricted technical
artefact from a spatially restricted biological programme unless that distinction
is represented in the expression and batch structure supplied to the model.

## Main inputs

### Expression matrix

By default, `biobatchnet_params.use_raw` is `false`, so training uses `adata.X`
and `adata.var_names` exactly as they exist when this stage starts. BioBatchNet
does not normalise or transform that matrix in the pipeline wrapper. The matrix
should therefore be finite, numeric, and scientifically appropriate for measuring
similarity between cells. Its meaning depends on upstream processing: for example,
it may contain marker intensities, transformed intensities, or inference scores.

If `use_raw: true` and `adata.raw` exists, `adata.raw.X` and its feature names are
used instead. If raw data are requested but unavailable, the stage falls back to
`adata.X`. Record which representation was used when comparing runs.

Sparse matrices are converted to dense arrays before training. This can require a
large amount of host memory for datasets with many cells or features, independently
of GPU memory.

### Batch annotation

`batch_correction_obs` must name an existing `adata.obs` column. Its values are
converted to strings, sorted, and mapped to consecutive integer identifiers. The
mapping is saved in `adata.uns["biobatchnet"]["batch_mapping"]`.

Use a column that represents the technical unit whose effect should be removed,
such as staining batch or acquisition run. Do not casually substitute a biological
variable such as diagnosis, treatment, tissue region, or patient group. When a
biological condition occurs in only one batch, technical and biological effects
are statistically confounded: no correction method can determine from these data
alone which associated signal is unwanted. Adversarial correction may then erase
the condition of interest.

The strongest design is one in which each relevant biological condition is
represented across multiple batches and batches contain overlapping cell states.
If that is impossible, interpret correction conservatively and retain uncorrected
results as an explicit comparator.

## Reusable assets produced

The saved AnnData contains:

- `adata.obsm["X_biobatchnet"]`: the biological embedding used for downstream
  neighbours, UMAP, and Leiden;
- `adata.obsm["X_batch_integration"]`: an alias of the same biological embedding,
  used by the wider pipeline as the active integrated representation;
- `adata.obsm["X_biobatchnet_batch"]`: the batch-specific embedding, when returned
  by BioBatchNet;
- `adata.uns["biobatchnet"]`: the batch key and mapping, model dimensions, epoch
  count, selected device, and representation key; and
- `adata.uns["batch_integration"]`: standard pipeline metadata identifying
  BioBatchNet as the integration method.

When `biobatchnet_run_postprocess` is enabled, the stage also writes the Scanpy
neighbour graph and `adata.obsm["X_umap"]`. If Leiden is enabled, it adds one
`adata.obs["leiden_<resolution>"]` column per requested resolution. Existing
neighbour, UMAP, or identically named Leiden results in the output AnnData may be
replaced.

The stage does **not** alter `adata.X` into corrected marker abundances. Coordinates
in `X_biobatchnet` are abstract latent features and must not be interpreted as
individual proteins or used as corrected per-marker measurements.

Unless `output_adata_path` is set, the base run updates `general.anndata_path`.
The default API call uses a temporary checkpoint directory during fitting; the
trained model itself is not retained as a reusable pipeline asset.

## Human-facing outputs

With post-processing enabled, QC figures are written under
`<general.qc_folder>/BioBatchNet`:

- when Leiden is enabled, one two-panel UMAP for each resolution, coloured by the
  batch annotation and by the corresponding Leiden cluster; or
- when Leiden is disabled, a UMAP coloured by batch only.

Named parameter scans receive separate QC subdirectories and sibling AnnData files.
When more than one run is performed, the stage also writes
`biobatchnet_parameter_scan_summary.csv`, recording run labels, paths, and model
parameters. This summary is an execution inventory, not a scientific ranking of
the runs.

The stage does not currently report quantitative integration metrics, training-loss
curves, or plots of the batch-specific embedding. These must be added to an
analysis outside the stage if they are required for model selection.

## Important configuration options

A typical configuration is:

```yaml
biobatchnet:
  batch_correction_obs: staining_batch
  biobatchnet_params:
    data_type: imc
    latent_dim: 20
    epochs: 100
    device: null
    use_raw: false
    extra_params:
      loss_weights:
        recon_loss: 100.0
        discriminator: 0.05
        classifier: 1.0
        kl_loss_1: 0.0005
        kl_loss_2: 0.1
        ortho_loss: 0.01
  biobatchnet_run_postprocess: true
  biobatchnet_run_leiden: true
  leiden_resolutions_list: [0.3, 1.0]
  umap_min_dist: 0.1
  n_neighbors: null
```

The nested model parameters have the following roles:

| Parameter | Meaning in this pipeline |
| --- | --- |
| `data_type` | Selects the upstream reconstruction model; use `imc` for continuous IMC measurements. |
| `latent_dim` | Number of coordinates in each learned latent representation. Too few can compress away biology; more dimensions increase capacity and computational cost without guaranteeing better separation. |
| `epochs` | Maximum number of training epochs. More epochs increase runtime and can change the balance of reconstruction and correction. |
| `device` | `null` automatically selects CUDA when available and otherwise uses CPU. An unavailable requested CUDA device falls back to CPU. |
| `use_raw` | Chooses `adata.raw.X` when available instead of the default `adata.X`. |
| `extra_params` | Additional arguments for the pinned BioBatchNet API, including its legacy loss-weight mapping. |

The configured loss weights are intentionally pipeline-specific rather than the
defaults of the pinned upstream package:

| Loss key | Default | Scientific role |
| --- | ---: | --- |
| `recon_loss` | `100.0` | Preserves information needed to reconstruct marker profiles. |
| `discriminator` | `0.05` | Controls adversarial removal of batch information from the biological embedding; larger values impose stronger correction. |
| `classifier` | `1.0` | Encourages the batch embedding to retain batch identity. |
| `kl_loss_1` | `0.0005` | Regularises the biological latent distribution. |
| `kl_loss_2` | `0.1` | Regularises the batch latent distribution. |
| `ortho_loss` | `0.01` | Reduces cross-covariance between biological and batch embeddings. |

Changing one loss weight can change the effective role of the others. Treat a scan
as a sensitivity analysis, not as an automated search for a universally optimal
number.

`n_neighbors`, `umap_min_dist`, and `leiden_resolutions_list` act only after model
training. They change the graph, visual layout, and clustering but do not change
the learned BioBatchNet embeddings. `n_for_pca` is a deprecated compatibility
field and is not used by the current stage; no PCA is run here. The deprecated
flat `biobatchnet_*` options are migrated into `biobatchnet_params`, but new
configurations should use the nested form.

## Parameter scans

`biobatchnet_scan_parameter_sets` can compare explicitly chosen model settings.
Each set starts from the same input AnnData and overrides the base nested parameters.
For example:

```yaml
biobatchnet:
  biobatchnet_scan_include_base: true
  biobatchnet_scan_parameter_sets:
    - name: weaker_adversary
      extra_params:
        loss_weights:
          recon_loss: 100.0
          discriminator: 0.02
          classifier: 1.0
          kl_loss_1: 0.0005
          kl_loss_2: 0.1
          ortho_loss: 0.01
    - name: stronger_adversary
      extra_params:
        loss_weights:
          recon_loss: 100.0
          discriminator: 0.10
          classifier: 1.0
          kl_loss_1: 0.0005
          kl_loss_2: 0.1
          ortho_loss: 0.01
```

A scan's `extra_params.loss_weights` replaces the complete base loss-weight mapping,
so all six keys required by the pinned interface must be repeated. Use stable,
descriptive `name` values because they become file and directory suffixes.

BioBatchNet training and VAE sampling are stochastic, and the pipeline does not
currently expose a complete run seed. Apparent differences between parameter sets
can therefore include run-to-run variation. Important conclusions should be
checked across repeated runs or otherwise tested for stability.

## How to interpret the results

Start with paired questions rather than the batch-coloured UMAP alone:

1. **Was technical separation reduced?** Within a known cell type or state, do
   batches occupy shared neighbourhoods instead of batch-specific islands?
2. **Was biology preserved?** Are established populations, expected marker
   relationships, rare states, treatment responses, and tissue-specific programmes
   still recoverable?

Useful checks include comparing corrected and uncorrected embeddings; colouring by
batch, sample, patient, condition, known phenotype, and marker expression; checking
population frequencies per sample; and examining whether rare or batch-restricted
populations disappear after correction. Where labels are trustworthy, metrics such
as batch silhouette or iLISI can assess mixing, while cell-type silhouette, graph
connectivity, ARI, or NMI can assess biological conservation. No single metric is
sufficient, and the pipeline does not calculate these automatically.

The biological embedding should make batch difficult to predict while preserving
phenotype. Conversely, the batch-specific embedding is expected to retain batch
information; separation there is evidence that the two encoders are playing
different roles, not necessarily a failure. Inspecting both embeddings can be more
informative than inspecting `X_biobatchnet` alone.

UMAP is a nonlinear visualisation of the neighbour graph. Visual overlap is not
proof that batch effects were removed, and visual separation is not proof that a
difference is technical. Leiden clusters are graph partitions whose number and
boundaries depend on `n_neighbors`, the latent representation, and resolution.
They are analytical conveniences, not automatically biological cell types.

## Common problems and limitations

- **Confounded study design:** if condition and batch coincide, correction can
  remove the biological comparison. This cannot be repaired by parameter tuning.
- **Non-overlapping compositions:** a population present in only one batch may be
  treated as batch-specific even when it is real. Preserve such populations using
  external evidence and conservative comparisons.
- **Over-correction:** unusually complete mixing accompanied by lost marker-defined
  identities or condition responses indicates that adversarial pressure may be too
  strong.
- **Under-correction:** persistent within-population separation by technical run may
  require different weights, more appropriate input processing, or improved study
  design.
- **Memory and runtime:** the wrapper densifies the full matrix and trains neural
  networks. Large cell counts can exhaust RAM or GPU memory; CPU training may be
  slow.
- **Stochastic results:** repeated runs need not be identical, and no complete seed
  control is exposed by this stage.
- **No corrected marker matrix:** latent axes have no one-to-one marker meaning.
  Differential marker testing still requires an appropriate expression layer and
  sample-aware statistical design.
- **No spatial model:** spatial coordinates and tissue topology are not used.
- **Version-specific parameters:** the pipeline is pinned to an older BioBatchNet
  API. Use the legacy loss keys shown above rather than assuming examples from the
  current upstream repository apply unchanged.
- **Evidence status:** the BioBatchNet article supplied for this documentation is a
  bioRxiv preprint and had not undergone journal peer review at the time of posting.
  Its benchmark results support the method's promise but do not guarantee suitable
  correction for a new cohort.

## Primary references

- Gao Y, Adil A, et al. *BioBatchNet: A dual-encoder framework for robust batch
  effect correction in imaging mass cytometry data.* bioRxiv (2025),
  [doi:10.1101/2025.03.15.643447](https://doi.org/10.1101/2025.03.15.643447).
- [UoM-HealthAI/BioBatchNet](https://github.com/UoM-HealthAI/BioBatchNet), the
  upstream implementation. SpatialBiologyToolkit uses the version pinned in its
  BioBatchNet environment lock, so consult the pipeline configuration above for
  the exact supported parameter names.
