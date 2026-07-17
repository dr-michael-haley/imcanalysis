# Visualisation

## What this stage does

Generates UMAPs, matrix plots, tissue overlays, backgating views, abundance summaries, and statistical tables.

## Why it is performed

It turns processed cell data into interpretable views for quality review and biological analysis.

## Main inputs

Processed AnnData, population labels, metadata, masks, and channel images.

## Reusable assets produced

No large computational asset is normally created; AnnData colour metadata may be reused.

## Human-facing outputs produced

Figures, legends, population galleries, abundance tables, and statistical summaries.

## Important configuration options

Input AnnData, population columns, metadata groups, plot switches, image channels, and statistical settings.

## How to interpret the results

Use several complementary views and account for sample/ROI structure when comparing populations.

## Common problems and limitations

Missing annotations or image assets can disable only the affected visualisation subsets.
