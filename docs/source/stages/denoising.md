# Denoising

## What this stage does

This stage restores selected Imaging Mass Cytometry (IMC) channel images with
[IMC-Denoise](https://github.com/PENGLU-WashU/IMC_Denoise). It addresses two
different acquisition artefacts in sequence:

1. **DIMR** (differential intensity map-based restoration) detects isolated hot
   pixels and small hot-pixel clusters, then replaces only the detected pixels
   with local median values.
2. **DeepSNiF** (deep shot-noise image filtering) learns the spatial signal
   expected for one marker from that marker's own images and suppresses
   signal-dependent shot noise without requiring clean reference images.

With `denoising.method: deep_snf`, both steps are applied: DIMR first and
DeepSNiF second. With `denoising.method: dimr`, only hot-pixel restoration is
performed. The stage processes each selected marker independently and writes a
new per-ROI TIFF set for downstream segmentation and marker quantification.

## Why it is performed

IMC measures metal-tagged antibodies by laser-ablation time-of-flight mass
spectrometry. The tissue is sampled on a roughly 1 micrometre raster, and each
channel records discrete ion counts rather than conventional camera intensity.
This produces an unusually sparse image with technical noise that is not well
described by a single Gaussian blur model.

The IMC-Denoise paper represents a raw channel as:

```{math}
R = \mathcal{P}[X + X_{\mathrm{spillover}}] + Q
```

Here, `X` is the underlying biological signal, `P` is the signal-dependent
Poisson ion-counting process, and `Q` represents hot pixels.
Hot pixels are intense, spatially abrupt events unrelated to tissue structure,
often attributed to deposited aggregates of metal-conjugated antibody. Shot
noise is different: its magnitude depends on the true signal and each pixel's
ion count is statistically independent. Both can distort cell boundaries,
inflate cell-level marker summaries, obscure weak staining, and alter manual or
automated phenotyping.

The two algorithms therefore solve different problems. DIMR removes sparse
outliers without applying a global intensity cutoff to the whole image.
DeepSNiF then estimates the locally predictable component of the remaining
Poisson-corrupted signal. Neither step performs spillover compensation, and
denoising cannot recover biological information that was never detected.

In the published human bone-marrow evaluation, DeepSNiF reduced the annotated
background standard deviation by 87% and increased contrast-to-noise ratio
5.6-fold for the reported Collagen III images. The paper also found improvements
in background classification and cell phenotyping. These are benchmark results,
not performance guarantees for a new tissue, antibody panel, or acquisition.

## How the restoration works

### DIMR: adaptive hot-pixel restoration

DIMR begins with an Anscombe transform. This stabilizes the variance of
Poisson-like count data so that local intensity differences are more comparable
across bright and dim regions. Very low-count pixels are excluded from outlier
detection because they cannot represent high-intensity hot pixels.

For every remaining centre pixel, DIMR calculates its difference from the
surrounding pixels in the configured local window. With the default 3 x 3
window there are eight neighbours. Biological staining and background are
expected to be locally continuous, so at least some neighbour differences
should resemble the centre of their image-wide difference distributions. A hot
pixel instead produces unusually large positive differences from several
neighbours.

DIMR measures how far each neighbour difference lies from the robust median of
its corresponding difference distribution, retains the configured number of
most distribution-consistent differences (`n_neighbours`), and sums them. A
kernel density estimate of these scores separates the main pixel distribution
from its extreme right tail. Pixels beyond the automatically derived tail
boundary are replaced with the median of their local window. Detection and
replacement are repeated up to `n_iter` times before the inverse Anscombe
transform returns the image to its original count scale.

This distribution-derived threshold is why DIMR can adapt across markers and
tissues with different intensity ranges. Only detected pixels are replaced;
the method is not a median filter over the complete image. The default
`n_neighbours: 4` encodes the assumption that at least half of the eight
neighbours in a 3 x 3 window resemble the local biological context, and the
published simulation supported three iterations for single hot pixels and
small consecutive clusters.

### DeepSNiF: self-supervised shot-noise filtering

Ground-truth clean IMC images cannot normally be acquired: laser ablation
destroys the sampled tissue, so the same field cannot be rescanned with a
longer exposure. DeepSNiF therefore trains without clean targets. It first uses
DIMR-corrected images from one marker to make 64 x 64 training patches. Patches
dominated by pixels below intensity 1 are excluded according to `ratio_thresh`
so that sparse foreground structures are represented, and rotations and flips
augment the retained patches.

During training, a small stratified sample of pixels is hidden by replacing
each with a nearby value. The network sees the manipulated patch but is scored
against the original value only at those masked positions. Because IMC shot
noise is pixel-independent while antibody staining is spatially structured,
the surrounding pixels contain information about the expected biological
signal but not the hidden pixel's particular noise realization. The default
`pixel_mask_percent: 0.2` means 0.2% of pixels, not 20%.

The default loss combines two terms:

- **I-divergence at masked pixels**, a Poisson-aware data-fidelity objective
  derived for ion-counting data.
- **Hessian-norm regularization over the whole prediction**, weighted by
  `lambda_HF`, which encourages locally continuous biological structure and
  reduces discontinuities that masked-pixel training alone can leave behind.

The `small` model is a compact U-Net; `normal` selects the original, much larger
residual U-Net. Both use an encoder-decoder with skip connections to combine
local detail with broader context, and a non-negative output activation for the
default I-divergence model. Training intensities are normalized using 1.1 times
the configured upper quantile (`truncated_max_rate`). After a separate model is
trained for each marker, the full DIMR-corrected image is passed through it so
that TIFFs do not acquire patch-stitching boundaries.

## Main inputs

- One folder per ROI below `general.raw_images_folder`, containing one 2D TIFF
  per marker channel.
- `metadata/panel.csv`, or the corresponding configured metadata folder. When
  `denoising.channels` is empty, rows with `to_denoise` set are selected; older
  panels fall back to `use_denoised`. The expected identifier is
  `<channel_name>_<channel_label>` and is matched case-insensitively within TIFF
  filenames.
- `metadata/metadata.csv`, used by the current implementation to identify ROIs
  smaller than the configured patch stride.
- A working TensorFlow/CUDA environment for DeepSNiF training and inference.
  DIMR is algorithmic, but this stage currently imports TensorFlow and performs
  the same GPU availability check before either configured restoration method.

Each marker's DeepSNiF model is trained from all matching ROIs. This provides
more examples of its staining morphologies than training ROI-specific models,
but it also means an ROI with a unique artefact or staining regime can influence
the model applied to every other ROI for that marker.

## Reusable assets produced

The main reusable asset is a mirrored ROI/channel TIFF hierarchy below
`general.denoised_images_folder` (default `processed`). TIFFs are written as
32-bit floating-point images and retain their original filenames. Parameter
scans instead create sibling folders named
`<denoised_images_folder>_<scan_parameter>_<scan_value>`.

DeepSNiF also writes per-channel model files named
`weights_<channel>.keras` and matching normalization-range files. They are
stored below `denoising.weights_save_directory`, or `trained_weights` in the
working directory when that setting is null. These files are required together
when `is_load_weights` is used; weights without the training normalization range
do not define a complete inference model.

## Human-facing outputs produced

- `denoised_pixel_qc.csv`, or one suffixed table per parameter-scan value,
  reports the mean of per-ROI mean, standard deviation, minimum, and maximum
  values from the central 60% of each denoised image. `low_std` flags channels
  whose mean within-ROI standard deviation is below 1.
- When `run_QC` is enabled, one raw-versus-denoised figure is written per
  channel. Up to `qc_num_rois` ROIs are sampled without replacement; null uses
  all ROIs.
- `remove_outliers_report.csv` records panel-driven pre-denoising pixel removal,
  including the threshold and number and percentage of affected pixels per ROI.
- Optional `.npz` or `.mat` training/validation loss histories are written when
  `loss_name` is configured.

The central-image pixel summary is a coarse technical diagnostic. It is not the
paper's annotated-background standard deviation or contrast-to-noise ratio, and
its `low_std` flag is not evidence that a marker has been successfully denoised.

## Important configuration options

- `method` determines whether the stage stops after DIMR or continues with
  DeepSNiF.
- `n_neighbours`, `n_iter`, and `window_size` control DIMR. The published
  defaults (4, 3, and 3) are sensible starting points; changing them alters what
  counts as an isolated local outlier and how replacement is performed.
- `patch_step_size` controls the initial spacing of fixed 64 x 64 training
  patches. With `intelligent_patch_size: true`, the stage changes this stride in
  20-pixel steps to seek at least `intelligent_patch_size_min_patches` augmented
  patches and, when configured, no more than
  `intelligent_patch_size_max_patches`. The retained
  `intelligent_patch_size_threshold` setting is not read by the current code.
- `ratio_thresh` controls the exclusion of background-dominated patches. A
  higher value admits sparser patches; a lower value demands more pixels at
  intensity 1 or above. Overly strict values may remove rare staining
  morphologies from the training set.
- `train_epochs`, `train_initial_lr`, and `train_batch_size` control model
  optimization. `val_set_percent` reserves a deterministic patch-level
  validation split, but augmented versions of the same source region are made
  before that split and should not be interpreted as an independent biological
  validation cohort.
- `loss_function: I_divergence`, `lambda_HF: 3e-6`, and `network_size: small`
  are the recommended DeepSNiF defaults. Alternative `mse` and `mse_relu`
  settings are Noise2Void-style comparison modes, not the published DeepSNiF
  objective.
- `truncated_max_rate` controls normalization. Reducing it prevents extreme
  counts from defining the whole dynamic range but clips more high pixels;
  increasing it makes the model more sensitive to extremes. Parameter scans
  are preferable to choosing this setting from one convenient ROI.
- `is_load_weights` skips patch generation and training and loads the
  automatically named per-channel model plus range file. Reuse is appropriate
  only when the marker, staining morphology, intensity scale, preprocessing,
  and network/loss architecture are compatible with the training data.

The complete field-by-field definitions are generated from
`DenoisingConfig` in the [configuration reference](../reference/configuration/sections/denoising.md).

## Data-changing preprocessing specific to this pipeline

Two operations in this repository sit outside the published DIMR/DeepSNiF
algorithm and should be reviewed before a run:

- When `remove_outliers: true`, the stage reads the optional `remove_outliers`
  rule in each panel row. An absolute rule such as `8000` or a tail-fraction
  rule such as `p0.001` is converted to a channel threshold, and pixels above it
  are set to zero **in the raw TIFFs themselves**. This is destructive and is
  different from DIMR's locally adaptive median replacement. A pre-existing
  report causes the operation to be skipped. Preserve an untouched copy of the
  raw TIFF hierarchy and review the report before interpreting denoising.
- Before restoration, ROI folders whose `height_um` or `width_um` in
  `metadata.csv` is below `patch_step_size` are deleted from the raw-image
  hierarchy. This test is based on the initial stride rather than the fixed
  64-pixel patch dimensions or adaptive minimum. Ensure small but biologically
  important ROIs have been handled deliberately.

## How to interpret the results

Interpret hot-pixel removal and shot-noise filtering separately.

For DIMR, inspect whether isolated high-count pixels disappear without loss of
true sharp structures. Pay particular attention to marker-positive cell edges,
thin vessels, stromal fibres, punctate staining, and borders between tissue and
background. A useful residual would be dominated by isolated pixels or very
small clusters rather than coherent anatomy. The current report does not save a
raw-minus-DIMR residual image, so direct TIFF inspection may be needed.

For DeepSNiF, look for lower granular background variation and more coherent
within-structure signal while preserving cell shape, boundaries, and rare
positive objects. Denoising should improve the separability of positive and
negative regions, not simply make the image visually smooth. Review several
ROIs and biological morphologies for every important marker, especially markers
used for segmentation, gating, or cell-type definitions.

The side-by-side QC panels independently set each image's upper display limit
to half of its own maximum. They are useful for morphology but do not provide a
shared quantitative intensity scale. Compare the TIFF values, pixel summaries,
and downstream per-cell distributions before concluding that intensity has been
preserved. If loss histories are saved, a stable validation curve can identify
gross training failure, but it does not demonstrate biological fidelity.

Parameter scans should be judged using pre-specified biological structures and
multiple representative ROIs. Prefer the least aggressive setting that removes
the target noise while preserving expected localization and cell-level marker
relationships. After selection, downstream segmentation and quantification
should use one consistently processed image set rather than a mixture of scan
outputs.

## Common problems and limitations

- DIMR is designed for single hot pixels and small clusters. Large contiguous
  hot clusters can resemble genuine staining and may remain, or an aggressive
  local setting may damage a real compact structure.
- DeepSNiF learns spatially predictable signal; it cannot recreate marker
  information absent from a very low-SNR acquisition. Apparent structures in a
  denoised image remain model estimates and should be checked against other
  markers and tissue morphology.
- At approximately 1 micrometre IMC resolution, structures only 1 to 2 pixels
  across may not be distinguishable from noise and may not be learned reliably.
- Self-supervision avoids fabricated clean labels but does not equal supervised
  training against true clean images. A low validation loss is not a ground
  truth accuracy measurement.
- The method assumes useful biological staining is locally continuous and shot
  noise is conditionally independent between pixels. Highly punctate biology,
  unusual acquisition artefacts, or registration/resampling artefacts may
  violate these assumptions.
- IMC-Denoise does not compensate channel spillover. A well-designed and
  titrated panel remains necessary, and spillover correction is a separate
  analytical decision.
- High-SNR channels generally benefit less than weak channels. The paper
  reported that DeepSNiF could be omitted for some markers whose positive-cell
  mean exceeded 7 counts, but this empirical observation is not a universal
  threshold across instruments, panels, or tissues.
- `skip_already_denoised` infers completed channels from filenames in the first
  existing output ROI. Incomplete or heterogeneous output folders should be
  checked manually before resuming a run.
- Training one network per marker reduces memory use and respects marker-specific
  morphology, but runtime grows with channel count and requires a compatible
  TensorFlow/CUDA installation. The compact `small` model is faster and usually
  appropriate for limited datasets; the larger `normal` model is not inherently
  more biologically accurate.

## Primary references

- Lu P, Oetjen KA, Bender DE, *et al.* [IMC-Denoise: a content aware denoising
  pipeline to enhance Imaging Mass Cytometry](https://doi.org/10.1038/s41467-023-37123-6).
  *Nature Communications* 14, 1601 (2023).
- [PENGLU-WashU/IMC_Denoise](https://github.com/PENGLU-WashU/IMC_Denoise),
  the authors' reference implementation and tutorials. Spatial Biology Toolkit
  uses a Python-compatibility-updated version of this package while preserving
  the DIMR and DeepSNiF scientific workflow described above.
