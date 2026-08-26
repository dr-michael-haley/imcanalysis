# HyPERSTAC

## What these stages do

HyPERSTAC learns patch-level image representations directly from multiplex
images, then uses Leiden clustering to discover recurrent spatial proteomic
patterns. The SBT integration is based on the
[HyPERSTAC preprint](https://www.biorxiv.org/content/10.1101/2025.10.16.682563v1.full)
and retains the IMC adaptation's patch metrics, perturbation analysis, report
galleries, optional survival integration, and cross-Leiden stability assessment.

The workflow is exposed as checkpoint jobs:

1. `hyperstac-preprocess` - background correction, robust per-channel scaling,
   and an advisory normalization preflight.
2. `hyperstac-model` - patch extraction, VICReg training, and representation
   AnnData creation.
3. `hyperstac-permutation` - channel-zeroing and pixel-shuffling sensitivity.
4. `hyperstac-visualise` - graph/UMAP/Leiden scans and interpretation reports.
5. `cox` - the separate general-purpose survival stage described in the
   [Cox guide](cox_survival.md).
6. `hyperstac-stability` - compare every Leiden setting using assignment,
   graph, cluster-support, and marker-environment evidence; overlay Cox
   directions when a compatible managed Cox report exists.

Submit the checkpointed workflow with:

```bash
sbt run hyperstac
```

Each alias can be requested directly. By default, SBT adds a conventional
upstream producer only when a blocking reusable asset is absent. Existing
validated assets permit the requested checkpoint to run directly while skipped
lineage is reported as a warning. Use `--dependency-policy all` for the full
checkpoint lineage or `--dependency-policy none` for explicit stages only. To
run everything in one GPU allocation:

```bash
sbt run hyperstac-full
```

By default this includes Cox and overlays survival evidence on the clustering
comparison. For an image-only run that still keeps preprocessing, model
training, permutation, visualisation, and parameter comparison inside one GPU
allocation, configure:

```yaml
hyperstac:
  full_include_survival: false
```

The job skips Cox but still runs the survival-independent clustering comparison
after visualisation. The `cox` stage remains independently runnable if
compatible survival metadata becomes available later.

## Why the analysis is performed

Cell segmentation and mean marker intensity intentionally compress image
structure. HyPERSTAC instead asks whether self-supervised image features capture
recurrent tissue architecture, marker localization, morphology, and
microenvironmental context without defining those patterns in advance.

The resulting Leiden labels are exploratory spatial proteomic clusters, not
automatic tissue or cell-type annotations. They should be reviewed with marker
heatmaps, patch galleries, ROI representation, perturbation sensitivity, case
prevalence, and stability across clustering settings.

## Main inputs

- An ROI/channel TIFF hierarchy:
  `{image_folder}/{ROI}/{channel}.tif[f]`.
- `hyperstac.input_images_folder`, or
  `general.denoised_images_folder` when the override is null.
- A consistent channel set and image shape within every ROI.
- Optional explicit `hyperstac.channels`; an empty list infers the alphabetical
  channel order from the first ROI and validates it against every ROI.
- For the optional Cox overlay, case outcomes and feature sources
  configured in the separate `cox` section.

Images supplied directly to `hyperstac-model` must already be scaled to
`[0, 1]`. The normal route obtains those assets from
`hyperstac-preprocess`.

## Reusable assets produced or modified

Reusable assets live below `hyperstac.asset_folder` (default `hyperstac/`):

- `normalised_images/`: background-corrected, scaled ROI/channel TIFFs and
  normalization metadata.
- `patches/` and `patch_metadata.csv`: NumPy patch arrays with spatial identity,
  signal, tissue-filter, and marker summary fields.
- `model/encoder.weights.h5` and `model/projector.weights.h5`: TensorFlow model
  weights.
- `imc_hyperstac_representations.h5ad`: one row per patch, with encoder
  representations in `.X`, patch/ROI identity in `.obs`, spatial centres in
  `.obsm["spatial"]`, and the training contract in `.uns`.
- `imc_hyperstac_patch_metrics.h5ad`: aligned handcrafted patch metrics.
- `permutation_sensitivity/imc_permutation_sensitivity.h5ad`: aligned cosine
  distance/similarity results for each perturbation.

When `hyperstac.write_clustered_adata=true`, `hyperstac-visualise` updates the
representation AnnData with namespaced graph/UMAP state and columns such as
`leiden_0.35_N100_P20`. This is a deliberate reusable-asset modification.

## Human-facing outputs produced

Managed execution reports contain:

- channel- and ROI-level normalization tables, an advisory preflight report,
  and representative source/normalized image montages;
- patch metadata and the resolved training/perturbation contracts;
- perturbation condition and per-patch summaries;
- UMAPs colored by Leiden labels, marker intensity, and perturbation scores;
- cluster marker heatmaps, ROI composition tables, optional spatial maps, TIFF
  label masks, and patch galleries;
- per-clustering visualisation summaries;
- a ranked (not prescriptive) setting scorecard, all-setting ARI/NMI matrices,
  adjacent-parameter stability, graph purity/conductance, cluster/ROI support,
  sampled silhouette metrics, and recurrent marker-environment evidence;
- CoxNet/Ridge direction comparisons, perturbation-survival overlays, and
  per-clustering HTML interpretation pages when compatible Cox output exists.

Visualisation and stability contain many interlinked files. Their full trees are
kept as managed report attachments so relative gallery and HTML links remain
valid; the execution manifest still inventories the individual files.

## Important configuration options

The IMC defaults preserve the tested local adaptation:

- `patch_size=100`, `pixel_size_um=1.0`, and a matching default stride;
- `epochs=50`, `batch_size=64`, `learning_rate=1e-4`, ResNet-50, and a
  2,048-wide projector;
- 10 px local subpatches, signal threshold `0.01`, and minimum tissue fraction
  `0.5`;
- Leiden resolutions `[0.2, 0.25, 0.3, 0.35]`, neighbours `[15, 30, 100]`, and
  PCA dimensions `[0, 10, 20, 50]`, where zero uses the complete representation;
- ten shuffle repeats and all-channel perturbations;
- spatial TIFF maps disabled by default because whole-cohort masks can be large.

These are adaptation defaults, not universal biological constants. In the
published mIF protocol, HyPERSTAC used 224 px tiles, 100 epochs, batch size 256,
a ResNet-50 encoder, a 2,048-dimensional representation and an 8,192-dimensional
projection head on an 80 GB H100. SBT does not silently impose those much larger
resource assumptions on 1 um/pixel IMC data.

The former GBM-specific marker list is deliberately not a default. Configure an
ordered list when channel order must be fixed across runs; otherwise inference
is validated before patch creation.

Normalization defaults are also assay-dependent. The preflight flags channels
where the background is large relative to raw p99, normalized p99 is very low,
saturation is high, or ROI presence is sparse. These are review prompts rather
than proof of failure. `normalisation_preview_rois_per_channel` controls the
representative montages (zero disables them). Set
`normalisation_fail_on_preflight_warning=true` only when the heuristic cutoffs
have been validated for the assay and the workflow should stop before model
training.

## How to interpret the results

Start with patch/ROI counts and the normalization QC. Next inspect the embedding
UMAP, marker heatmaps, and patch galleries for each Leiden setting. A plausible
environment should have coherent image appearance and marker evidence, appear
across more than one ROI/case, and not be driven solely by background or tissue
edges.

Permutation cosine distance asks whether removing or rearranging a marker
changes the learned representation. It is a sensitivity measure, not causal
evidence. The clustering comparison does not choose a biologically correct
resolution automatically. Its review rank prioritizes non-dominated settings
with stronger adjacent-setting agreement, matched-graph separation, cluster
support, and ROI representation. Inspect the shortlist against marker heatmaps
and patch galleries. When survival is available, CoxNet/Ridge direction is an
additional overlay and is not part of the reference-free selection score.

This parameter scan measures robustness to resolution, neighbour count, and
PCA dimension. It is not seed stability, bootstrap stability, external-label
validation, or proof that a cluster represents a biological compartment.

For a chosen clustering, the public
`SpatialBiologyToolkit.hyperstac.local_analysis` helpers reconstruct exact-size
ROI label masks, crop combined PNG/SVG galleries directly from normalized
TIFFs, recreate marker and zero/shuffle perturbation clustermaps, summarize
sample-associated environment abundance, and assign cell centres to the masks.
See [Local analysis of pre-generated HyPERSTAC artifacts](../guides/hyperstac_local_analysis.md)
for a notebook-oriented workflow and the mask-value contract.

Survival plots fitted and displayed on the same cases are optimistic. Prefer the
held-out cross-validation summaries and risk-group plots produced by `cox`.

## Common problems and limitations

- Missing, extra, or differently named channel TIFFs stop the run before GPU
  training.
- `overwrite=true` replaces assets within the configured HyPERSTAC asset folder.
  Keep `reuse_patches=false` unless the existing patch identity and image
  contract are known to match.
- Patch size is measured in pixels; biological scale changes with
  `pixel_size_um`.
- VICReg, permutation inference, and the full job require the TensorFlow/CUDA
  environment. CPU-only syntax or import checks do not validate GPU execution.
- The clustering scan can create many graphs, UMAPs, and figures. Reduce the
  grid before increasing cohort size.
- `hyperstac-stability` needs the clustered representation plus a compatible
  managed visualisation report. Cox is optional; the stage does not manufacture
  survival metadata or infer case identity from patch names.
- Normal managed runs use the shared, repository-managed `sbt-tensorflow`
  environment. The external `sbt-hyperstac` environment remains registered
  only as an explicit rollback option while representative workflow validation
  is completed.
