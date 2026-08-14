# Nimbus quantification

## What this stage does

Nimbus converts multiplexed marker images and an existing cell-segmentation mask
into a cell-by-marker table of **marker-positivity confidence scores**. For every
ROI and every selected marker, it examines the image pixels and predicts which
segmented cells are visually positive for that marker. The resulting score lies
between 0 and 1, with larger values indicating stronger model confidence that the
cell is marker-positive.

Despite the historical module name `segmentation_nimbus`, this stage does **not**
find cell boundaries or generate segmentation masks. It requires instance masks
from an earlier segmentation stage, normally Cellpose-SAM. Nimbus uses those masks
to tell the network where cell foreground and boundaries are and to aggregate its
pixel-level predictions into one score per cell.

SpatialBiologyToolkit also performs several pipeline-specific operations around
Nimbus:

- measuring cell position, area, perimeter, circularity, and major-axis geometry
  from the masks;
- optionally measuring conventional mean image intensity inside each cell mask;
- optionally measuring mean intensity after expanding each mask;
- writing ROI and cohort-wide cell tables; and
- constructing the canonical AnnData used by downstream integration, clustering,
  phenotyping, and spatial analysis.

These additional intensity measurements are useful comparators, but they are not
part of the Nimbus neural-network prediction.

## Why image-aware marker classification is useful

The conventional way to quantify a marker is to average all of its pixel
intensities inside a segmented cell. This integrated expression is convenient,
but it discards the spatial arrangement of the signal. Several common properties
of tissue images can then make a cell's mean intensity a poor proxy for whether
that cell truly expresses the marker:

- a membrane marker may be biologically convincing but occupy only a narrow part
  of the segmented cell, producing a low mean;
- bright signal from one cell may spill across an imperfect boundary and increase
  the mean of an adjacent negative cell;
- autofluorescence, nonspecific antibody binding, or spatially varying background
  can elevate the entire local region;
- segmentation can include extracellular space or part of a neighbouring cell;
  and
- marker intensities may differ by orders of magnitude between antibodies,
  platforms, slides, or acquisition runs.

A trained observer evaluates more than the mean. They consider whether signal has
a plausible subcellular shape, whether it follows the cell boundary, its contrast
with neighbouring cells and background, and whether a bright value is likely to
be spillover. Nimbus was designed to automate this image-aware decision at scale.
It predicts marker positivity directly from the image rather than treating an
integrated mean as the complete measurement.

## How Nimbus works

### One shared model, one marker at a time

Nimbus applies a single pretrained model independently to each marker channel. It
does not receive the antibody name, the other markers in the panel, a cell-type
label, or a reference cell atlas. The same network can therefore process panels
with different marker combinations without retraining a panel-specific cell-type
classifier.

This design separates two tasks that are often conflated:

1. Nimbus asks whether the image pattern for an individual marker looks positive
   in each cell.
2. A later clustering or annotation method combines scores across markers to
   infer cell identity.

Nimbus itself does not call a cell a T cell, macrophage, endothelial cell, or
cancer cell. It supplies cleaner marker-level features from which those biological
phenotypes can be inferred.

### Image and mask inputs

For one marker in one ROI, the model receives two image planes:

- the normalized marker-intensity image; and
- a binary map of cell foreground, with the inner boundaries of adjacent instances
  removed to preserve separation between cells.

The instance label numbers are not used as semantic categories by the network.
They are retained outside the model so that the final confidence map can be
averaged separately over the pixels belonging to each cell.

The marker image gives Nimbus access to subcellular staining shape and local
contrast. The foreground map tells it which regions are cells and makes boundaries
explicit. Unlike a purely tabular model, the convolutional network can use nearby
pixels when distinguishing genuine signal from spillover or background.

### Residual U-Net and confidence scores

Nimbus uses a residual U-Net. The contracting path combines information across a
larger spatial neighbourhood, while the expanding path restores local detail. A
sigmoid output produces a confidence value between 0 and 1 for every pixel. The
per-cell Nimbus score is the mean of these pixel confidences over the adjusted
instance mask.

The output is therefore a model-derived confidence score, not the mean input
intensity and not an estimated protein concentration. A high score means that the
spatial image pattern resembles marker-positive examples learned during training.
It does not mean that 80% confidence is equivalent to a particular number of
antigen molecules or that scores are calibrated identically for every antibody.

### Resolution, tiling, and test-time augmentation

The released checkpoint expects data at a specified magnification, normally 10×.
If `dataset_magnification` differs from `model_magnification`, the toolkit rescales
the normalized image and binary mask together before inference, then returns the
confidence map to the original mask dimensions.

Large images are processed using Nimbus's tile-and-stitch inference. The toolkit
also pads ROI dimensions to a multiple of 16 before prediction, reflecting image
pixels at the outer edge and zero-padding the label mask, then crops the result
back to the original size. This avoids losing border pixels through the four U-Net
downsampling operations.

With `test_time_augmentation: true`, the model predicts rotated and flipped views,
maps them back to the original orientation, and averages their confidence maps.
This reduces orientation-specific variation at the cost of additional inference
time.

## How the model was trained and validated

The published Nimbus model was trained on the Pan-Multiplex (Pan-M) dataset. Pan-M
contains approximately 197 million cell-marker annotations from 15 million cells,
spanning 54 protein markers, 10 broad cell lineages, 4 tissue systems, and 3 imaging
platforms: MIBI-TOF, CODEX, and Vectra.

Most training targets were **silver-standard labels**. The authors took cell types
assigned in existing studies and mapped each type to markers expected to be
positive, negative, or undefined. This made it possible to create training labels
at a scale that would be impractical by manual image review. Because the original
clustering and marker-to-cell-type mappings can contain mistakes, these labels are
large and diverse but imperfect.

To reduce overfitting to label errors, Nimbus used noise-robust training. An initial
model was trained with cross-entropy loss and strong weight decay. Fine-tuning then
excluded high-loss cells and retained high-confidence examples on which the model
and silver-standard label agreed. Image augmentations included elastic deformation,
rotations, flips, brightness and contrast changes, Gaussian noise, and blurring.

For evaluation, the authors manually corrected more than one million cell-marker
annotations from approximately 90,000 cells to create a held-out gold-standard
test set. They reported precision, recall, specificity, and F1 score against those
expert labels. In the evaluated data, the general Pan-M model matched the accuracy
of the underlying study-specific silver labels and outperformed the tested
alternative approaches on the two largest dataset subsets. Nimbus scores also
separated positive and negative gold-standard cells more clearly than integrated
expression and supported accurate downstream phenotyping in independent breast
cancer datasets.

These results are strong evidence for the published domains, not a guarantee for
every panel or tissue. Performance remains dependent on staining quality,
segmentation, normalization, similarity to the training distribution, and the
biological meaning of the marker.

## What SpatialBiologyToolkit runs

The Nimbus environment currently pins `Nimbus-Inference==0.0.4`. The toolkit loads
the pretrained checkpoint and performs inference only; it does not train or
fine-tune Nimbus.

The stage proceeds as follows:

1. It reads `panel.csv` and selects channels marked `use_denoised` or `use_raw`.
2. It locates a label mask for each ROI and applies any configured boundary offset
   and cell-area filters.
3. It finds each channel image, preferring the source specified by the panel and
   optionally falling back to the other raw or denoised folder.
4. It calculates or loads one normalization Vmax per marker and produces
   normalization QC.
5. It runs the pretrained model independently for every common ROI-marker pair.
6. It averages each confidence map within cell instances and merges the scores
   with mask geometry.
7. It optionally calculates conventional and expanded-mask intensities.
8. It writes cell tables and the canonical AnnData.

The toolkit's normalization is deliberately more mask-aware than the generic
upstream helper. For each marker, it calculates the configured quantile using only
pixels inside retained cell masks for each usable ROI, then averages those per-ROI
values. The default quantile is 0.999, or the 99.9th percentile, and the computed
Vmax is not allowed to fall below `normalization_min_value` (default 3.0).
Computed dictionaries use a zero lower threshold; reviewed CSVs can set a
marker-specific non-negative cutoff, commonly around 0.2–2 when low-level
background needs removing. The shared two-bound transform clips the input to the
range 0–1.

This differs from the article's reported inference preprocessing, which used the
channel-wide 99.99th percentile. The published validation therefore supports the
model's principle and released checkpoint, while the exact toolkit scores also
reflect this pipeline-specific normalization choice.

## Main inputs

### Instance masks

`general.masks_folder` must contain one two-dimensional labelled TIFF per ROI. Zero
is background and each positive integer identifies one cell. The mask and every
channel image for an ROI must have the same height and width.

`mask_boundary_offset_pixels` changes the mask before *all* measurements. Positive
values expand labels into available background without allowing labels to overlap.
Negative values erode each cell independently and can remove small cells entirely.
`min_cell_area` and `max_cell_area` are then applied to the adjusted masks. These
options change which pixels and cells enter Nimbus, the conventional means, the
expanded means, cell geometry, and AnnData; they are not cosmetic QC settings.

### Marker images and panel selection

The stage uses `panel.csv` to select image channels:

- `use_denoised: true` makes the denoised image the preferred source;
- `use_raw: true` makes the raw image the preferred source; and
- `allow_raw_fallback: true` permits searching the other source when the preferred
  file is absent.

The historical `use_denoised_first` option is not read by the current stage. The
panel flags are authoritative. With the default naming mode, files are matched
using a `channel_name_channel_label` hint. `simple_image_names: true` instead uses
`channel_label.tiff`.

Nimbus inference is limited to channels available in every usable ROI. When
`segmentation.allow_missing_channels` is false, channels absent from one or more
ROIs are excluded from the final marker set. When it is true, missing expected
channels are represented by `NaN`; downstream tools must be able to handle those
missing values explicitly.

Non-finite image pixels are reported and replaced with zero before normalization
and measurement. This keeps inference numerically valid but does not repair the
acquisition or export problem that generated them.

### Metadata

`metadata.csv` controls which ROIs are imported and can add ROI names, dimensions,
and source-file information to `adata.obs`. An optional `dictionary.csv` can add
project-specific sample or biological annotations. These metadata do not influence
Nimbus predictions, but they are essential for sample-aware QC and downstream
statistics.

### Magnification

`dataset_magnification` must describe the actual scale of the channel images and
masks. An incorrect value changes the physical scale of structures presented to
the network. `model_magnification` normally remains 10 to match the released
checkpoint. Rescaling cannot restore spatial detail that was absent from the input.

## Normalization and its QC outputs

The stage writes the preferred `normalization_dict.csv` in `nimbus.output_dir`.
It contains one row per marker with these columns:

```text
marker,vmax,lower_threshold
CD3,20,0
FOXP3,5,0.8
```

The canonical headings are `marker`, `vmax`, and `lower_threshold`. Here `vmax`
is the selected upper normalization (baseline) value, while `lower_threshold` is
the absolute background-removal value. Both are in source-image intensity units.
Nimbus receives
`clip((image - lower_threshold) / (vmax - lower_threshold), 0, 1)`, so pixels at
or below the lower threshold become zero and pixels at or above Vmax become one.
Every lower threshold must be non-negative and strictly below that marker's
Vmax. A lower threshold of zero exactly preserves the previous `image / vmax`
normalization.

Set `normalization_dict_path` to load a reviewed CSV directly from any
project-relative or absolute path:

```yaml
nimbus:
  normalization_dict_path: metadata/reviewed_normalization.csv
```

An explicit path takes precedence and does not require
`reuse_saved_normalization: true`. It must point to a `.csv`; the source is read
without modification when it is outside `nimbus.output_dir`. Nimbus writes the
complete resolved table to
`nimbus.output_dir/normalization_dict.csv`, filling any omitted selected markers
with computed mask-aware Vmax values and a zero lower threshold.

When `normalization_dict_path` is unset, `reuse_saved_normalization: true` loads
the CSV already present in `nimbus.output_dir` and applies its reviewed bounds
consistently across reruns. A legacy scalar
`normalization_dict.json` remains readable for backwards compatibility; its
lower thresholds default to zero, and Nimbus writes an equivalent preferred CSV
without deleting the JSON.
When both files exist, CSV takes precedence. Reuse should be intentional: bounds
from a different cohort, staining protocol, or intensity scale can distort scores.

Normalization QC is written under
`<general.qc_folder>/nimbus_normalization_qc` and contains:

- `norm_hists/<marker>.png`: the distribution of per-ROI in-mask quantiles, with
  the final averaged Vmax marked;
- `cellpos_hists/<marker>.png`: the per-ROI proportion of cells whose **mean
  normalized input intensity** exceeds 1; this is an intensity diagnostic, not a
  Nimbus-positive-cell estimate. With the default QC upper clip of 1.0, the
  reported proportion is necessarily zero; and
- `channel_galleries/<marker>.png`: sampled ROIs shown as raw, normalized,
  normalized-and-masked, and clipping-diagnostic images. Red pixels reached the
  displayed upper clip and blue pixels are zero.

`normalization_subset` controls only the number of randomly sampled ROIs in the
galleries. All usable ROIs contribute to normalization, and the histograms are
still produced when galleries are disabled. The current toolkit calculation is
serial; `normalization_jobs` is retained as a compatibility option but does not
currently change execution.

Use `norm_dict_qc_only: true` to stop after this step. This is useful for reviewing
or editing normalization values before expensive inference, but it deliberately
does not create Nimbus scores, intensity tables, or AnnData.

For a model-based sensitivity assessment, run `sbt run nimbus-scan` before Nimbus.
That optional GPU stage repeats Nimbus inference marker by marker across candidate
Vmax values, reports score distributions and positive-call proportions, detects
sharp adjacent-value changes, and proposes review-only stable ranges without
creating AnnData or overwriting `normalization_dict.csv`. See
[Nimbus normalization scan](nimbus_normalization_scan.md).

## Reusable assets produced

### Cell tables

The default cohort-wide tables under `nimbus.output_dir` are:

- `nimbus_celltable.csv`: Nimbus scores plus ROI, cell label, centroid, mask area,
  perimeter, circularity, major-axis length and angle, and a master index;
- `nimbus_classic_celltable.csv`: raw mean source-image intensities measured inside
  each adjusted cell mask; and
- `nimbus_expansion_celltable.csv`: raw mean source-image intensities measured
  after independently expanding each adjusted cell mask.

ROI-level versions of the merged Nimbus table are written below
`general.celltable_folder/nimbus_cell_tables` by default.

Expanded-mask measurements should be interpreted cautiously. Each cell is dilated
independently, so expanded regions can overlap and the same pixel can contribute to
multiple cells. The measurement may include neighbouring cells, extracellular
signal, or spillover; it is not a decontaminated estimate of whole-cell abundance.

### AnnData

The canonical AnnData is written to `general.anndata_path`. The older
`nimbus.anndata_output` and `segmentation.anndata_save_path` settings are retained
for compatibility but do not override that canonical path.

The marker matrices are:

| Location | Contents |
| --- | --- |
| `adata.X` | Untransformed per-cell Nimbus confidence scores, one column per retained marker. |
| `adata.layers["nimbus_raw"]` | A copy of the same Nimbus scores stored in `adata.X`. Here, "raw" means untransformed model output, not raw image intensity. |
| `adata.layers["mean_intensities_raw"]` | Conventional mean source-image intensities inside adjusted masks, when enabled. |
| `adata.layers["mean_intensities_normalized"]` | Conventional means transformed by `segmentation.marker_normalisation`, by default a per-marker 99.9th-quantile scaling across cells. |
| `adata.layers["expansion_intensities_raw"]` | Mean source-image intensities inside independently expanded masks, when enabled. |
| `adata.layers["expansion_intensities_normalized"]` | Expanded means transformed by `segmentation.marker_normalisation`. |

The Nimbus scores in `adata.X` are not processed by
`segmentation.marker_normalisation`; they already occupy the model's 0–1 confidence
range. `adata.obs` contains cell geometry and available ROI/sample metadata, while
`adata.obsm["spatial"]` stores X/Y mask centroids.

`segmentation.remove_channels_list` removes markers whose names contain configured
substrings from all matrices, and `segmentation.remove_and_store_markers` can move
selected markers into a separate AnnData before the main object is saved.

### Optional confidence maps

`save_prediction_maps: true` writes one TIFF per ROI and marker. Pixel confidences
are scaled from 0–1 to 0–255 and saved as 8-bit images. These maps are valuable for
checking *where* the model found positive evidence, especially for cells on which
Nimbus and conventional intensity disagree. The floating-point per-cell table
remains the more precise numerical output.

## Important configuration options

A representative configuration is:

```yaml
nimbus:
  output_dir: nimbus_output
  allow_raw_fallback: true
  simple_image_names: false
  mask_extensions: [.tiff, .tif]
  mask_boundary_offset_pixels: 0
  min_cell_area: null
  max_cell_area: null

  checkpoint: latest
  device: auto
  model_magnification: 10
  dataset_magnification: 10
  test_time_augmentation: true
  batch_size: 10

  normalization_quantile: 0.999
  normalization_min_value: 3.0
  normalization_subset: 10
  normalization_dict_path: null
  reuse_saved_normalization: false
  norm_dict_qc_only: false

  save_prediction_maps: false
  allow_prediction_resize: false
  use_existing_master_celltables: false

  extract_classic_intensities: true
  extract_expansion_intensities: true
  expansion_pixels: 10
  expansion_jobs: 1
```

`checkpoint: latest` checks Hugging Face for the newest versioned checkpoint and
uses a cached local checkpoint if the check fails. This is convenient, but two runs
at different dates could use different weights. For strict reproducibility, retain
and name a specific local checkpoint and preserve the resolved configuration and
software environment with the analysis.

`allow_prediction_resize` is an emergency fallback for an unexpected prediction
shape mismatch. The toolkit normally pads, rescales, and crops explicitly so the
confidence map matches the mask. Silent resizing can change which confidence pixels
fall inside a cell; leave it false unless the mismatch has been investigated.

`use_existing_master_celltables` can avoid repeated inference and intensity
extraction. It trusts compatible CSVs found at the configured paths and is therefore
unsafe after changing input images, normalization, checkpoint, panel selection, or
segmentation. The stage automatically disables reuse when mask offsets or area
filters are configured, but it cannot detect every stale-input scenario.

## How to interpret Nimbus scores

Treat each Nimbus value as evidence for marker positivity, not as quantitative
abundance. Useful interpretation starts with the image:

1. Select high-, intermediate-, and low-scoring cells for each biologically
   important marker.
2. Inspect the original or denoised marker image, the segmentation boundary, and,
   when saved, the Nimbus confidence map.
3. Examine cells where Nimbus and conventional mask means disagree. These are often
   the most informative examples of spillover, dim membrane staining, segmentation
   error, or model failure.
4. Compare score distributions across ROIs, batches, and biological groups before
   choosing thresholds or clustering parameters.
5. Confirm that known marker combinations emerge in plausible populations after
   clustering, while remembering that Nimbus did not use those combinations itself.

A score of 0.5 is not automatically a universal biological positivity threshold.
The article primarily uses continuous scores for downstream phenotyping, and score
calibration can differ by marker and image domain. If binary calls are required,
define and validate the decision rule using appropriate positive and negative
controls or expert-reviewed examples.

Nimbus can make downstream clustering easier by producing more bimodal features,
but it does not determine the correct number of cell types, identify clusters, or
solve rare-population detection. Cluster identities still require marker knowledge,
visual validation, sample-aware QC, and biological interpretation.

Population abundance must be calculated and tested at the independent sample
level, not by treating cells as independent replicates. Better per-cell marker
classification does not remove the need for an experimental unit and an
appropriate statistical design.

## Common problems and limitations

- **Autofluorescence:** Nimbus sees image signal, not its chemical origin. Strong
  autofluorescence can be classified as marker positivity and should be corrected
  or masked using appropriate controls.
- **Nonspecific staining:** off-target antibody binding is visually real signal and
  cannot be distinguished from antigen-specific staining by Nimbus alone.
- **Low signal-to-noise ratio:** the model was trained with variable image quality,
  but sufficiently dim signal or strong noise remains ambiguous for both humans and
  algorithms.
- **Segmentation errors:** merged cells, missing cytoplasm, shifted boundaries, or
  fragments alter both the binary model input and the pixels over which confidence
  is averaged. Nimbus reduces some consequences of imperfect masks but cannot repair
  the segmentation itself.
- **Out-of-distribution images:** the published training data covered multiplexed
  protein imaging with MIBI-TOF, CODEX, and Vectra. The authors specifically caution
  against assuming performance on H&E, conventional immunohistochemistry, or
  spatial transcriptomics, and unusual tissues or staining patterns may require
  expert validation or upstream fine-tuning outside this stage.
- **Graded functional markers:** Nimbus was trained for positive-versus-negative
  classification. It is not expected to reliably distinguish medium from high
  expression, for example different levels of PD-1 within positive T cells.
- **Normalization sensitivity:** inappropriate Vmax or lower-threshold values can suppress genuine dim
  signal or amplify background. Review the galleries and histograms before trusting
  large cohorts.
- **Missing channels:** default inference retains only markers present in every ROI.
  Allowing missing channels introduces `NaN` values that can break downstream
  normalization, PCA, integration, or clustering.
- **Expanded-mask contamination:** dilation can include neighbouring cells and
  double-count pixels. Use expanded intensity as a sensitivity analysis rather than
  an unquestioned ground truth.
- **Compute and storage:** test-time augmentation multiplies inference work, large
  fields require tiled prediction, and saved per-marker confidence maps can consume
  substantial storage.
- **Checkpoint drift:** `latest` may download newer weights. A fixed checkpoint is
  preferable when exact reproducibility matters.
- **No pipeline fine-tuning:** the upstream project provides fine-tuning workflows,
  but this stage only runs pretrained inference. Applying fine-tuned weights requires
  a compatible local checkpoint and separate validation.

## Primary references

- Rumberger JL, Greenwald NF, et al. *Automated classification of cellular
  expression in multiplexed imaging data with Nimbus.* Nature Methods 22,
  2161–2170 (2025),
  [doi:10.1038/s41592-025-02826-9](https://doi.org/10.1038/s41592-025-02826-9).
- [angelolab/Nimbus-Inference](https://github.com/angelolab/Nimbus-Inference), the
  released inference and fine-tuning implementation. SpatialBiologyToolkit uses
  the dependency version pinned in `HPC_env_files/imc_segmentation/pip-extras.txt`.
