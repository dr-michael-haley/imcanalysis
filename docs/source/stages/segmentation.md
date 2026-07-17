# Segmentation

## What this stage does

Preprocesses DNA images and uses CellPose-SAM to identify individual cell masks.

## Why it is performed

Segmentation defines the cell objects used for downstream marker quantification and spatial analysis.

## Main inputs

Denoised DNA-channel images and CellPose configuration.

## Reusable assets produced

Preprocessed DNA images and per-ROI segmentation masks.

## Human-facing outputs produced

Segmentation overlays, object summaries, parameter-scan figures, and cell-morphology tables.

## Important configuration options

Cell diameter, model, thresholds, mask filtering, upscaling, and parameter-scan values.

## How to interpret the results

Check that nuclei and cells are separated without excessive fragmentation, merging, or edge artefacts.

## Common problems and limitations

Optimal thresholds vary with tissue, resolution, staining quality, and CellPose version.
