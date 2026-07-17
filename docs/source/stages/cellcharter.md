# CellCharter Neighbourhoods

## What this stage does

Learns spatial neighbourhood representations and clusters cells into tissue-context neighbourhoods.

## Why it is performed

Cell states can have different meaning depending on their local cellular context.

## Main inputs

AnnData marker or latent representations, sample labels, and spatial coordinates.

## Reusable assets produced

Neighbourhood annotations, latent representations, optional project-root TRVAE model data, and updated AnnData.

## Human-facing outputs produced

Cluster composition, spatial maps, enrichment tables, shape summaries, and diagnostic plots.

## Important configuration options

Representation, sample key, neighbourhood scales, cluster counts, enrichment, and plotting controls.

## How to interpret the results

Relate neighbourhood composition and spatial localisation to tissue structure and experimental groups.

## Common problems and limitations

Results depend on coordinate quality, sampling density, representation choice, and cluster number.
