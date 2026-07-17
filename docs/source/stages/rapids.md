# RAPIDS Processing

## What this stage does

Runs GPU-accelerated filtering, PCA, optional Harmony, neighbours, UMAP, Leiden, and optional parameter scans.

## Why it is performed

It provides a scalable processing route for large cell datasets.

## Main inputs

AnnData with marker expression and required batch or filter annotations.

## Reusable assets produced

Updated AnnData embeddings, graphs, and cluster annotations.

## Human-facing outputs produced

UMAPs, matrix plots, scan summaries, and parameter-comparison figures.

## Important configuration options

Filtering, representation selection, Harmony, neighbours, UMAP, Leiden, and scan dictionaries.

## How to interpret the results

Inspect stability across settings and confirm clusters are not dominated by unwanted technical structure.

## Common problems and limitations

Requires a compatible RAPIDS/CUDA environment and sufficient GPU memory.
