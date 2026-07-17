# Subclustering

## What this stage does

Creates editable templates, subclusters selected populations, generates diagnostic plots, and optionally applies a curated remap.

## Why it is performed

It supports focused refinement of broad cell-population annotations without rerunning the entire pipeline.

## Main inputs

AnnData plus reusable settings, marker-list, and remap CSV files in the configured subclustering asset folder.

## Reusable assets produced

Editable templates, remap tables, master-index mappings, and updated AnnData annotations.

## Human-facing outputs produced

Combined and highlighted UMAPs and marker matrix plots.

## Important configuration options

Checkpoint mode, base label, marker lists, Leiden resolution, representation, and final label key.

## How to interpret the results

Use plots to decide whether subclusters are distinct and edit the remap table before final integration.

## Common problems and limitations

Template population names and marker columns must match the current AnnData.
