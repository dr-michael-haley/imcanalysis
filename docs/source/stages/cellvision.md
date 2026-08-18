# CellVision

## What this stage does

CellVision learns image-based representations of individual segmented IMC cells.
It converts selected cells and marker channels into identity-tracked 36 x 36
scPortrait images, trains a PyTorch VICReg model, extracts one embedding per
cell, fuses its morphology graph with a local graph rebuilt from the selected
cells' BioBatchNet embedding, clusters the joint graph with RAPIDS, and builds a
comparison and explanation-QC report.

Run the four checkpoint stages as separate dependent jobs with:

```bash
sbt run cellvision
```

Here `cellvision` is a mode containing `cellvision-extract`,
`cellvision-embed`, `cellvision-cluster`, and `cellvision-plot`. Each component
can also be requested directly. To keep the complete workflow in one GPU
allocation instead, run:

```bash
sbt run cellvision-full
```

The `job_cellvision_full.sh` wrapper switches from `sbt-scportrait` for
extraction/training, to `sbt-analysis` for clustering, and back to
`sbt-scportrait` for plots.

## Why it is performed

Mean marker intensity compresses each cell/channel image to one value. CellVision
tests whether spatial image features such as localization, texture, polarity,
shape, and within-cell marker relationships can separate phenotypes that are
difficult to distinguish with those conventional summaries.

This remains an exploratory representation-learning analysis. A CellVision
cluster is not automatically a biological cell type; interpret it with the
original population labels, channel galleries, cohort composition, and repeat
runs.

## Main inputs

- The source AnnData selected by `cellvision.input_adata_path` or
  `general.anndata_path`.
- One labelled mask TIFF per ROI, selected by `cellvision.masks_folder` or
  `general.masks_folder`.
- One folder per ROI containing marker TIFFs, selected by
  `cellvision.images_folder` or `general.denoised_images_folder`.
- The ROI observation column (`cellvision.roi_obs`, falling back to
  `general.roi_obs`) and integer mask-label column
  (`cellvision.object_id_obs`, default `ObjectNumber`).
- Optional `cellvision.population_obs` plus `cellvision.populations` selection.
- Optional ordered `cellvision.markers`; all discovered channels are used when
  this is unset. Each configured marker is matched case-insensitively against
  the end of a TIFF stem, immediately before `.tif`, `.tiff`, or their OME
  variants. Thus both `165Ho_CD11c` and `CD11c` can select a prefixed
  `..._165Ho_CD11c.tiff`, while `CD3` cannot select `..._CD31.tiff`.
- Optional `cellvision.normalization_dict_path` pointing to the preferred Nimbus
  `normalization_dict.csv` or a legacy JSON dictionary. CellVision reads the
  Vmax column; its own image extraction continues to use Vmax-only scaling.
  Nimbus short keys such as `CD11c` are resolved to configured full
  names such as `165Ho_CD11c` by the same suffix rule. Exact keys take priority;
  missing or ambiguous matches fail before extraction. Relative paths resolve
  from the project root.
- With the default `cellvision.fusion_enabled=true`, the source AnnData must
  contain `obsm["X_biobatchnet"]`, or the AnnData containing that representation
  must be supplied with `cellvision.fusion_intensity_adata_path`. CellVision
  selects the matching rows by observation ID and rebuilds a local intensity
  graph without recomputing or discarding the BioBatchNet correction.

Source AnnData observation names must be unique, and each `(ROI, object ID)` pair
must be unique and present in the corresponding labelled mask. These are checked
before scPortrait or GPU work starts.

## Reusable assets produced or modified

The source AnnData is read-only. Reusable outputs live below the configured
`cellvision.asset_folder` (default `scPortrait/CellVision`):

- `extraction/data/single_cells.h5sc`: exact masked cell images used for model
  training, already normalized to `[0, 1]`. H5SC mask channels are retained in
  the file but excluded from the VICReg inputs.
- `normalization_dict.json`: CellVision's compatibility copy of the supplied or
  computed Vmax-only channel scales used before scPortrait extraction.
- `cell_identity.csv`: every requested source observation, its source row,
  ROI/object ID, numeric scPortrait ID, and extraction status.
- `extraction_metadata.json`: selection/channel contract and identity
  fingerprint used to validate resumable runs.
- `vicreg_encoder.pt`: model weights, architecture, marker indices/names,
  training settings, Torch version, and identity fingerprint. VICReg does not
  fit or apply another intensity normalization.
- `cellvision_embeddings.h5ad`: encoder embeddings with the original AnnData
  observation names as row identities.
- `cellvision_clustered.h5ad`: CellVision PCA, separate morphology and
  BioBatchNet-intensity graphs, their degree-normalized joint graph, a
  random-initialized joint UMAP, and one namespaced
  `cellvision_leiden_<resolution>` column per configured resolution. When fusion
  is disabled, the stable joint keys contain the morphology-only result.

scPortrait can omit cells it cannot extract. Such cells are retained in
`cell_identity.csv` as `not_extracted_by_scportrait`; they are not silently
renumbered or projected downstream.

## Human-facing outputs produced

The combined run writes to `outputs/<execution_id>_CellVision_Full/`. Separate
jobs write their own extraction, embedding, clustering, and plotting execution
reports. Across those reports, CellVision produces:

- the weighted VICReg training objective and its unweighted invariance,
  variance, and covariance components;
- a CellVision UMAP colored by each Leiden resolution;
- the same UMAP colored by `cellvision.population_obs`, when configured;
- count and original-population-row-normalized confusion tables plus normalized
  confusion plots;
- an automatic cluster-explanation QC figure and summary table reporting the
  fraction of selected source-marker intensity variance, CellVision PCA
  morphology-proxy variance, ROI entropy, and source-population entropy
  explained by every configured CellVision Leiden resolution;
- detailed per-marker and per-CellVision-PC explained-variance tables;
- every CellVision Leiden label projected onto the original AnnData UMAP, with
  cells absent from CellVision shown in grey;
- per-cluster galleries whose rows are sampled cells and columns are the exact
  H5SC image channels used by VICReg; when there are at most three channels, an
  additional RGB composite column is included. Every channel uses the fixed
  display range `[0, 1]`; cells and channels are never individually autoscaled.

## Important configuration options

- `population_obs`, `populations`, and `markers` define the scientific scope.
  The source population column is preserved under its original name, including
  `leiden` or resolution-specific names such as `leiden_1.0`. CellVision
  clustering writes to the separate `cellvision_leiden_<resolution>` namespace,
  so source and CellVision labels can coexist in the CellVision AnnData object.
- `image_size=36` defines the fixed scPortrait crop.
- `mask_gaussian_blur=false` keeps the extraction mask binary by default for
  1 µm/pixel IMC. Enabling it restores scPortrait's sigma-1 softened mask edge.
- `normalization_dict_path` reuses reviewed Nimbus channel scales. When it is
  supplied, its keys may be exact configured marker names or unique short
  suffixes from Nimbus; resolved values remain keyed by the configured
  CellVision names downstream. If a selected marker is absent from the supplied
  dictionary, CellVision logs the marker and falls back for that channel only
  to the same default calculation used when no dictionary is supplied. Existing
  supplied marker values are retained. CellVision's default is to calculate the
  0.999 quantile of in-mask pixels per ROI, average the ROI values, and floor
  the result at 3.0. Images are divided by those values, clipped to `[0, 1]`,
  and stored that way in H5SC. VICReg validates this range and does not rescale.
- `embedding_dim`, `projector_dim`, `epochs`, `batch_size`, and the three
  `vicreg_*_weight` values control representation learning.
- Augmentation magnitudes and probabilities can be tuned independently for
  flips, right-angle rotations, zero-filled integer translations, per-marker
  intensity jitter, and Gaussian noise. Right-angle rotations avoid interpolation
  of low-resolution IMC pixels. `augmentation_noise_support=channel` preserves
  each marker's original nonzero support; `segmentation_mask` permits noise only
  inside the stored cell mask. Random crops, arbitrary-angle interpolation,
  hue/saturation operations, and channel mixing are not used.
- Marker-intensity jitter is disabled by default
  (`augmentation_intensity_jitter_probability=0`) so the two VICReg views retain
  marker amplitude. The configured jitter magnitude remains available for an
  explicitly intensity-invariant experiment.
- `n_pcs` and `n_neighbors` are single RAPIDS values; only
  `leiden_resolutions` is a list.
- `fusion_enabled=true` builds the morphology graph from `X_cellvision_pca` and
  the intensity graph directly from the selected rows of
  `fusion_intensity_representation` (default `X_biobatchnet`). Both graphs use
  `n_neighbors`, are symmetrically degree-normalized, and are combined with
  `fusion_intensity_weight=0.5`; the morphology weight is one minus that value.
  Graph fusion therefore does not require equal feature counts. Set the weight
  to 0 or 1 for modality endpoints, or disable fusion for the former
  morphology-only pipeline behavior. UMAP always uses `init_pos="random"` with
  the configured seed to avoid the pathological spectral-initialization layouts
  seen for these graphs.
- `overwrite=false` validates and reuses complete assets only when the extraction
  input manifest, cell/marker identity, VICReg training contract, and RAPIDS
  clustering contract match. Set it to `true` after deliberately changing those
  inputs or settings.

## Component scripts and SLURM checkpoints

The `cellvision` mode exposes these registered stages and package modules in
order:

1. `cellvision-extract` → `SpatialBiologyToolkit.scripts.cellvision_extract`
2. `cellvision-embed` → `SpatialBiologyToolkit.scripts.cellvision_embed`
3. `cellvision-cluster` → `SpatialBiologyToolkit.scripts.cellvision_cluster`
4. `cellvision-plot` → `SpatialBiologyToolkit.scripts.cellvision_plot`

Matching component wrappers are
`job_cellvision_extract.sh`, `job_cellvision_embed.sh`,
`job_cellvision_cluster.sh`, and `job_cellvision_plot.sh`. These are independent
planner stages. The default asset-aware policy adds the nearest upstream
CellVision producer only when a component's blocking reusable assets are absent.
When those assets already exist, the component runs directly and skipped lineage
is reported as a warning. Use `--dependency-policy all` for a deliberate full
checkpoint rerun, or `--dependency-policy none` to prohibit automatic producers.
`cellvision-full` remains the single-job alternative.

The extraction adapter is based on the earlier local `scPortrait_to_IMC` helper
behavior but is now repository-owned, multi-channel, population-aware, and tied
to the canonical SBT config/reporting system. The VICReg objective is a new
PyTorch implementation informed by the HyPERSTAC workflow; no TensorFlow patch
tiling or handcrafted patch features are retained. scPortrait documents both
[H5SC extraction](https://mannlabs.github.io/scPortrait/pages/module/pipeline/extraction.html)
and its [PyTorch-compatible single-cell datasets](https://mannlabs.github.io/scPortrait/pages/module/tools/ml.html).

## How to interpret the results

Start with the original-population UMAP and confusion matrices. A useful result
may split one conventional population reproducibly into image-defined groups or
merge conventional groups with genuinely similar spatial phenotypes. Inspect the
corresponding H5SC channel galleries to determine whether the separation reflects
cellular morphology/localization rather than acquisition artifacts, empty crops,
ROI effects, or simple global intensity.

Read the cluster-explanation QC as a confounding and interpretation aid, not a
formal significance test. Continuous fractions are eta-squared values; marker
intensity gives every selected marker equal weight in the headline mean, while
the morphology headline is variance-weighted across CellVision PCs. ROI and
source-population values are the directional fractions of those label entropies
explained by the learned clusters. The CellVision PCA is a morphology proxy, not
a guarantee of intensity-free morphology, so residual intensity can contribute
to both continuous diagnostics.

Compare multiple seeds and, where sample size permits, hold out ROIs or cases.
The current stage trains on all selected cells and is intended for representation
discovery, not a validated out-of-sample classifier.

## Common problems and limitations

- Missing or duplicated `(ROI, ObjectNumber)` values stop extraction because
  downstream cell identity would otherwise be ambiguous.
- Marker labels must resolve consistently in every selected ROI.
- Values present in a supplied normalization dictionary must be finite and
  positive. Missing selected markers use the configured per-channel in-mask
  percentile fallback and are recorded in extraction metadata and reporting;
  additional unselected Nimbus channels are ignored.
  Existing pre-change H5SC/model assets must
  be rebuilt because VICReg now requires extraction-normalized `[0, 1]` inputs.
- Very small selections are unsuitable for VICReg variance/covariance learning
  even though the hard minimum is two extracted cells.
- A 36 px crop can truncate unusually large cells; review galleries and change
  `image_size` deliberately if needed.
- RAPIDS and model training require working CUDA environments on HPC. Local
  scPortrait extraction and CPU VICReg smoke tests do not establish full GPU
  compatibility.
- Fusion requires a finite, identity-aligned batch-corrected source embedding.
  CellVision fails rather than silently substituting raw intensity when the
  configured representation is missing. If population selection leaves little
  biological overlap between batches, local graph rebuilding preserves the
  BioBatchNet coordinates but cannot create missing cross-batch support.
- The external `sbt-scportrait` environment is registered but not locked by
  this repository. RAPIDS clustering uses the repository-managed
  `sbt-analysis` runtime. Run `sbt env test scportrait` and
  `sbt env test analysis` after environment changes.
