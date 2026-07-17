# Batch Integration

## What this stage does

Runs Harmony and/or BBKNN integration followed by neighbours, UMAP, and Leiden processing.

## Why it is performed

It reduces technical batch structure while retaining biological variation for downstream clustering.

## Main inputs

The canonical AnnData and a configured batch annotation in `adata.obs`.

## Reusable assets produced

Updated embeddings, graphs, clusters, and AnnData provenance.

## Human-facing outputs produced

UMAPs and concise integration diagnostics.

## Important configuration options

Batch key, integration method, PCA, neighbours, UMAP, and Leiden settings.

## How to interpret the results

Compare batch mixing and biological separation; integration should not erase expected biology.

## Common problems and limitations

Sparse batches, confounded designs, or an incorrect batch key can produce misleading integration.
