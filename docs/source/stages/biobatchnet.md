# BioBatchNet Integration

## What this stage does

Applies BioBatchNet correction and then computes downstream UMAP and Leiden results.

## Why it is performed

It offers a learned batch-correction route for datasets where standard integration may be insufficient.

## Main inputs

AnnData marker expression and a valid batch annotation.

## Reusable assets produced

Corrected representations, embeddings, clusters, and updated AnnData.

## Human-facing outputs produced

UMAP figures and optional parameter-scan summaries.

## Important configuration options

Batch key, model architecture, training settings, and downstream graph parameters.

## How to interpret the results

Assess both batch mixing and preservation of known biological populations.

## Common problems and limitations

Model training is environment- and data-dependent and may require GPU resources.
