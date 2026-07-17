# Quantification

## What this stage does

Uses Nimbus to quantify marker intensities within segmentation masks and builds cell tables and AnnData.

## Why it is performed

It converts images and masks into a cell-by-feature dataset for statistical and spatial analysis.

## Main inputs

Segmentation masks, channel images, panel metadata, and sample metadata.

## Reusable assets produced

ROI cell tables, a master cell table, and the canonical AnnData object.

## Human-facing outputs produced

Normalisation histograms, channel galleries, and quantification summaries.

## Important configuration options

Image source, normalisation, marker removal, missing-channel policy, and AnnData output settings.

## How to interpret the results

Verify cell counts, marker distributions, and expected metadata before clustering or integration.

## Common problems and limitations

Mask/image mismatches and inconsistent channels can cause missing or misaligned measurements.
