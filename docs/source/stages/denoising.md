# Denoising

## What this stage does

Applies the configured image-denoising method to selected IMC channels and records objective pixel-level diagnostics.

## Why it is performed

Denoising can reduce structured acquisition noise before segmentation and quantification.

## Main inputs

Raw per-channel TIFF images and the project panel table.

## Reusable assets produced

Denoised channel images in the configured project-root folder.

## Human-facing outputs produced

Pixel summaries, outlier-removal tables, comparison figures, and parameter-scan diagnostics.

## Important configuration options

Denoising method, selected channels, training parameters, outlier removal, and scan settings.

## How to interpret the results

Compare raw and denoised images and check that signal structure is retained while unwanted noise is reduced.

## Common problems and limitations

Over-aggressive settings can suppress biological signal; GPU and model requirements depend on the selected method.
