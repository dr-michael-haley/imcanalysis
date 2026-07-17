# Pairwise Spatial Analysis

## What this stage does

Computes neighbourhood interactions, distance-bootstrap statistics, pair-correlation functions, and related plots.

## Why it is performed

It quantifies whether selected populations co-localise, avoid one another, or differ across groups.

## Main inputs

AnnData population labels, ROI identities, spatial coordinates, and optional group annotations.

## Reusable assets produced

Cached raw spatial statistics that can support plot-only reruns.

## Human-facing outputs produced

Pairwise matrices, source-target plots, enrichment plots, and result tables.

## Important configuration options

Population pairs, distances, permutations, PCF radii, grouping, plotting, and reload settings.

## How to interpret the results

Interpret effect sizes alongside uncertainty, cell counts, spatial scale, and multiple comparisons.

## Common problems and limitations

Sparse populations and heterogeneous ROI geometry can destabilise pairwise estimates.
