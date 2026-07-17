# STARLING Phenotyping

## What this stage does

Runs segmentation-aware probabilistic phenotyping and writes assignment probabilities and cluster labels.

## Why it is performed

It models marker expression, cell size, and overlap uncertainty when assigning cellular phenotypes.

## Main inputs

AnnData expression, optional initial labels, and configured marker and cell-size fields.

## Reusable assets produced

STARLING annotations, probabilities, updated AnnData, and optional project-root model checkpoints.

## Human-facing outputs produced

Cluster counts, expression summaries, doublet summaries, and diagnostic figures.

## Important configuration options

Initial clustering, distribution, marker selection, regularisation, training, and probability-storage settings.

## How to interpret the results

Review assignment confidence, cluster expression profiles, and doublet patterns before using labels downstream.

## Common problems and limitations

Probabilistic assignments depend on model assumptions, initialisation, and marker informativeness.
