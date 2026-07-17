# NetworkX Spatial Analysis

## What this stage does

Builds per-ROI spatial graphs and computes assortativity, clustering, null distributions, and grouped summaries.

## Why it is performed

Graph metrics describe tissue organisation that is not captured by abundance alone.

## Main inputs

AnnData population labels, ROI/case metadata, and spatial coordinates.

## Reusable assets produced

Raw graph summaries and optional bootstrap samples.

## Human-facing outputs produced

Population and assortativity plots, null-comparison tables, and case/ROI summaries.

## Important configuration options

Graph construction, minimum cell counts, bootstrapping, grouping, and plotting controls.

## How to interpret the results

Compare observed metrics with their null distributions and consider ROI-level variability.

## Common problems and limitations

Metrics are sensitive to graph definition, cell density, rare populations, and coordinate scale.
