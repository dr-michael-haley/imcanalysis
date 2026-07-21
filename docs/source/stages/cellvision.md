# CellVision

## What this stage does

CellVision learns image-based representations of individual segmented IMC cells.
It converts selected cells and marker channels into identity-tracked 36 x 36
scPortrait images, trains a PyTorch VICReg model, extracts one embedding per
cell, clusters those embeddings with RAPIDS, and builds a comparison report.

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

The `job_cellvision_full.sh` wrapper switches from `scPortrait` for
extraction/training, to `rapids_singlecell` for clustering, and back to
`scPortrait` for plots.

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
- Optional `cellvision.normalization_dict_path` pointing to the same
  marker-to-value `normalization_dict.json` format produced and reviewed by
  Nimbus. Nimbus short keys such as `CD11c` are resolved to configured full
  names such as `165Ho_CD11c` by the same suffix rule. Exact keys take priority;
  missing or ambiguous matches fail before extraction. Relative paths resolve
  from the project root.

Source AnnData observation names must be unique, and each `(ROI, object ID)` pair
must be unique and present in the corresponding labelled mask. These are checked
before scPortrait or GPU work starts.

## Reusable assets produced or modified

The source AnnData is read-only. Reusable outputs live below the configured
`cellvision.asset_folder` (default `scPortrait/CellVision`):

- `extraction/data/single_cells.h5sc`: exact masked cell images used for model
  training, already normalized to `[0, 1]`. H5SC mask channels are retained in
  the file but excluded from the VICReg inputs.
- `normalization_dict.json`: the supplied or computed Nimbus-format channel
  scales used before scPortrait extraction.
- `cell_identity.csv`: every requested source observation, its source row,
  ROI/object ID, numeric scPortrait ID, and extraction status.
- `extraction_metadata.json`: selection/channel contract and identity
  fingerprint used to validate resumable runs.
- `vicreg_encoder.pt`: model weights, architecture, marker indices/names,
  training settings, Torch version, and identity fingerprint. VICReg does not
  fit or apply another intensity normalization.
- `cellvision_embeddings.h5ad`: encoder embeddings with the original AnnData
  observation names as row identities.
- `cellvision_clustered.h5ad`: RAPIDS PCA, neighbor graph, CellVision UMAP, and
  one namespaced `cellvision_leiden_<resolution>` column per configured
  resolution.

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
- every CellVision Leiden label projected onto the original AnnData UMAP, with
  cells absent from CellVision shown in grey;
- per-cluster galleries whose rows are sampled cells and columns are the exact
  H5SC image channels used by VICReg; when there are at most three channels, an
  additional RGB composite column is included. Every channel uses the fixed
  display range `[0, 1]`; cells and channels are never individually autoscaled.

## Important configuration options

- `population_obs`, `populations`, and `markers` define the scientific scope.
  The source population column is preserved under its original name, even when
  it is named like `leiden_1.0`; CellVision clustering writes directly to the
  separate `cellvision_leiden_<resolution>` namespace.
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
- `n_pcs` and `n_neighbors` are single RAPIDS values; only
  `leiden_resolutions` is a list.
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
planner stages, so a downstream component automatically includes its upstream
CellVision dependencies. Use `--no-deps` to rerun only that component after its
required reusable assets already exist. `cellvision-full` remains the single-job
alternative.

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
- The external `scPortrait` and `rapids_singlecell` environments are registered
  but are not locked by this repository. Run `sbt env test scportrait` and
  `sbt env test rapids` after environment changes.
