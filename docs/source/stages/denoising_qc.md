# Denoising QC

## What this stage does

Generates side-by-side denoising checks and tests consistency between panel channels and image folders.

## Why it is performed

It provides an explicit checkpoint before segmentation uses the denoised images.

## Main inputs

Raw images, denoised images, and `metadata/panel.csv`.

## Reusable assets produced

No large downstream asset is normally created.

## Human-facing outputs produced

Comparison images, panel-consistency tables, and pixel-QC summaries.

## Important configuration options

QC image selection, denoising display options, and panel/image locations.

## How to interpret the results

Investigate missing channels, unexpected image ranges, and ROIs flagged as inconsistent.

## Common problems and limitations

Pixel-level checks may be skipped when image-reading dependencies are unavailable.
