# Feature Building

Feature Building creates the numerical table used by the classifier. In a normal
experiment it covers the complete frozen cohort; in a Feature Discovery Trial it
covers only the selected trial ROIs. Full segmentation masks remain the authority
for boundaries, neighbours, and background exclusion.

## Imported sources

CSV, Parquet, AnnData `X`/`obs`/`obsm`, and CellVision embeddings are joined by
`(ROI, ObjectNumber)`. Validate sources before building to see identity coverage,
missing cells, and usable numeric columns. Imported rows outside the current
feature-build scope are ignored.

## Synthetic features

Select IMC channels and feature families. Distribution statistics summarize
pixel intensities; region features describe cores, borders, centroids, local
background, and contrast; gradients capture texture and edges; morphology uses
the original mask; context uses the full segmentation; cohort-relative ranks
compare eligible cells within the same ROI.

A positive offset normally uses full-mask `expand_labels`, preventing expansion
through neighbouring cells. Enable overlap explicitly if measurements should
extend into neighbouring masks. Negative offsets erode measurement regions.
Offsets change intensity regions but never the original shape features.

## Execution and progress

Local builds run in a subprocess with spawn-safe ROI workers. The progress panel
shows the orchestrator PID, heartbeats, completed/resumed/failed ROIs, recent
worker PIDs, and finalization. Cancellation preserves valid completed fragments.
Resume reuses fragments only when experiment, cohort, recipe, and input
fingerprints match. HPC builds use the same manifest through `sbt run cellfeat`.
