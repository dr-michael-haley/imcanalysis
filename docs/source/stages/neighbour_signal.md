# Neighbour-attributable signal analysis

## What this stage does

`neighsig` learns an empirical, marker-specific spatial halo from manually or
previously selected convincing positive cells, projects that halo from other
strong raw-image sources, and measures how much observed signal inside every
target cell mask is spatially explainable by those neighbours.

Run it with:

```bash
sbt run neighsig
```

The primary output is the **Neighbour-Attributable Fraction** in AnnData `X`.
It ranges from zero to one. Zero means that essentially none of the observed
background-subtracted marker signal in the target mask coincides with a
plausible neighbouring halo; one means that essentially all of it does.

The stage now retains two complementary results:

1. **How much of this target cell's marker signal is spatially explainable by
   neighbouring sources?** `NeighbourAttributableFraction` answers this in
   `X`.
2. **Which neighbouring cell(s) provide that spatial explanation?** The sparse
   source-target Parquet table retains every non-zero relationship, while
   dominant-source layers provide immediate cell-by-marker access in AnnData.

A source cell is a neighbouring cell whose projected marker halo spatially
explains signal observed inside the target cell mask. This identifies a
predicted spatial source; it does not prove that signal physically transferred
from one cell to another.

This is a spatial contamination/uncertainty metric. It is not isotopic-channel
spillover compensation, a calibrated probability, or proof that signal is
artefactual.

## Why it is performed

IMC segmentations are often deliberately conservative, for example a nucleus
plus a one-pixel expansion. Membrane and cytoplasmic markers can therefore
extend outside the source cell mask and into a nearby cell mask. Because the
spatial profile depends on marker localisation and imaging behaviour, this
stage learns each marker's halo directly from positive exemplar cells rather
than imposing one decay curve across the panel.

The input expression matrix may contain Nimbus confidence or another derived
measurement. It is not used in halo learning, source selection, background
estimation, or scoring, so it remains an independent comparison.

## Main inputs

- `general.anndata_path`: input AnnData, read without modification. Its cell and
  marker order defines the output axes.
- `adata.obs[neighbour_signal.exemplar_obs]`, default `Exemplar_stains`:
  non-null values must exactly match `adata.var_names` and identify convincing
  positive exemplars.
- `general.raw_images_folder`: the existing ROI/channel TIFF layout. Marker
  names are resolved with the same suffix-aware image discovery used by
  CellVision.
- `general.masks_folder`: one labelled TIFF mask per ROI.
- `general.roi_obs` and `neighbour_signal.object_id_obs`, defaulting to `ROI`
  and `ObjectNumber`: the exact AnnData-to-mask mapping. Every pair must be
  unique and present in its ROI mask.

Unknown exemplar values are reported and ignored. A marker is skipped when
fewer than `min_exemplars` valid exemplar profiles remain. Missing images,
masks, mapped labels, or shape agreement are input errors because pixel-level
alignment is required.

## Reusable assets produced or modified

The stage writes a separate AnnData to
`neighbour_signal.output_adata_path` (default
`neighbour_attributable_signal.h5ad`). The input AnnData is never overwritten.
The output is a copy of the input, retaining `.obs`, `.var`, `.obsm`, `.uns`,
and existing layers, with these changes:

- `X`: float32 Neighbour-Attributable Fractions, bounded in `[0, 1]`;
- `layers['original_X']`: the unmodified input expression/confidence matrix;
- `layers['classic_intensities']`: mean raw intensity inside each mask when
  enabled;
- `layers['neighbour_attributable_intensity']`: mean observed excess intensity
  per cell pixel captured by the neighbouring-halo model;
- `layers['residual_excess_intensity']`: mean per-pixel excess remaining after
  the projected halo is subtracted and clipped at zero;
- `layers['dominant_source_index']`: zero-based global AnnData row of the
  source contributing the largest attributable intensity for each target and
  marker, or `-1` when there is no attributable source;
- `layers['dominant_source_observed_fraction']`: fraction of the target's
  observed excess signal explained by its dominant source;
- `layers['dominant_source_attributable_fraction']`: fraction of all
  neighbour-attributable signal assigned to its dominant source;
- marker profile availability, exemplar count, source threshold, and effective
  extent in `.var`;
- `halo_max_score`, `halo_mean_score`, and the descriptive
  `halo_n_high_risk` summary in `.obs`;
- profiles, IQRs, exemplar statistics, backgrounds, parameters, worker
  allocation, score interpretation, and layer semantics in
  `.uns['marker_halo']`.

The stage also writes `neighbour_signal.source_target_table_path` (default
`neighbour_signal_source_target.parquet`). This sparse long-form asset contains
only non-zero target-marker-source relationships. Global zero-based AnnData row
indices and `obs_names` are authoritative; ROI and segmentation-label columns
are retained for provenance and sanity checks. Each row records total
attributable intensity, its fraction of the target's observed excess, and its
fraction of the target's total attributable component. Configured population
labels are included when available. The Parquet path, schema, relationship
count, identity semantics, and cautious interpretation are recorded in
`.uns['marker_halo']['source_target_table']`.

If the input already has an `original_X` layer, it is retained under a unique
`preexisting_original_X*` name before the current input `X` is stored.

## Human-facing outputs produced

The managed execution report contains:

- multi-panel empirical halo curves with exemplar IQR, exemplar count, and
  source threshold;
- all-marker score distributions and a CSV with median, 90th/95th percentiles,
  fractions above descriptive 0.25/0.5/0.75 thresholds, exemplar counts, and
  source thresholds;
- long-form profile, exemplar, ROI-background, skipped-marker, and unknown
  exemplar tables;
- Scanpy UMAP panels for a configured or automatically selected small marker
  set, plus `halo_max_score` and `halo_mean_score`, when `X_umap` exists;
- a Scanpy population-by-marker matrix plot when a suitable categorical
  population observation is available;
- a source-population-to-target-population marker summary and concise heatmaps
  for the most affected markers, optionally excluding same-population routes;
- a dominant-source summary reporting concentration above 50% of attributable
  signal, the median number of contributing sources, and common population
  routes;
- sampled classic-intensity versus original-`X` plots coloured by the halo
  score; and
- a concise interpretation/provenance summary linking the output AnnData and
  CPU allocation.

The 0.25, 0.5, and 0.75 report thresholds are descriptive, not validated
biological cutoffs.

## Important configuration options

```yaml
neighbour_signal:
  enabled: true
  output_adata_path: neighbour_attributable_signal.h5ad
  source_target_table_path: neighbour_signal_source_target.parquet
  exemplar_obs: Exemplar_stains
  object_id_obs: ObjectNumber
  max_halo_px: 8
  source_anchor_dilation_px: 2
  source_anchor_quantile: 0.95
  min_exemplars: 5
  source_threshold_quantile: 0.10
  halo_aggregation: max
  calculate_classic_intensities: true
  n_jobs: auto
  qc_markers: null
  max_qc_markers: 6
  population_obs: null
  source_target_qc_exclude_same_population: true
  high_risk_threshold: 0.5
```

Source strength is the raw-image anchor quantile. The anchor includes the cell
mask plus nearby unassigned pixels allocated to their nearest cell, so pixels
belonging to another segmented cell are excluded. The learned radial profile
uses only unassigned pixels outside an exemplar, subtracts a robust local
background, and divides by the exemplar's background-subtracted source
strength.

The marker source threshold is the configured quantile of valid exemplar raw
source strengths. Projected excess is scaled by
`max(source_strength - ROI_background, 0)`. Profiles are aggregated with the
median and are not forced to decrease monotonically; a membrane marker can
legitimately peak outside a conservative nucleus-centred mask.

Overlapping halos use their pixelwise maximum by default. Whenever a source
replaces the current maximum at a pixel, both predicted intensity and its
global AnnData source row are updated. Attributable pixels are then reduced by
target segmentation label and source row before the temporary pixel maps are
discarded; no second neighbour search is performed.

`sum` remains supported for the original score and is more aggressive in dense
source regions, but multiple sources contribute simultaneously to a pixel.
Source-resolved provenance is therefore disabled for `sum` with a clear
warning: the sparse table is empty and dominant-source layers use `-1`/zero
sentinels. Use the recommended `max` behavior when source identity is needed.

## Environment and resources

The stage reuses `imc_segmentation`; its existing NumPy, SciPy, tifffile,
AnnData, Scanpy, pandas, and matplotlib stack covers the analysis. No new
environment or package is required.

The initial wrapper requests 8 CPUs, 64 GB RAM, and 24 hours on the CPU
high-memory partition. `n_jobs: auto` resolves the available worker count from
the SLURM allocation, process affinity, and host CPU count. Profile extraction
and profile application are parallelized independently by ROI. Every worker
loads one mask once, reads marker TIFFs sequentially, and projects sources on
small radius-expanded bounding boxes. BLAS/OpenMP threads are fixed to one to
avoid nested oversubscription.

## How to interpret the results

Begin with the learned profile figure. A marker should have a plausible,
reasonably supported curve and adequate exemplar IQR before its cell scores are
trusted. A peak just outside the segmentation boundary can be biologically
reasonable for membrane or cytoplasmic localisation.

Next inspect the all-marker score table and UMAP/population views. A high score
means that the strongest nearby source, or the configured overlap aggregate,
could spatially explain much of the observed target-mask excess at the same
pixels. Compare classic intensity and independent input `X` to determine
whether the existing expression method already suppresses these spatially
suspicious measurements.

Use the sparse source-target table or dominant-source layers to ask which
neighbouring cell provides the spatial explanation. For example, an overall
fraction of 0.72, a dominant-source observed fraction of 0.54, and a
dominant-source attributable fraction of 0.75 mean that neighbouring sources
spatially explain 72% of the target signal, one source alone explains 54% of
the observed signal, and that source accounts for 75% of the attributable
component. These remain model-based spatial explanations, not assertions that
the source physically contaminated the target.

Use the score as QC evidence alongside image review and biological context. Do
not automatically delete, compensate, or relabel cells from this score alone.

## Common problems and limitations

- Too few or spatially unrepresentative exemplars produce a skipped marker or
  an unstable profile. Add convincing exemplars across representative ROIs.
- An exemplar with no intensity above its local background is invalid and does
  not contribute to the source threshold.
- Background and source-strength units are raw image units. Strong ROI-level
  acquisition differences can therefore affect which cells cross a global
  marker threshold and should be reviewed in the ROI-background table.
- Profiles are isotropic pixel-distance summaries. They do not model directed
  tissue structure, optical anisotropy, or cell-type priors.
- Maximum overlap asks whether the strongest source could explain a pixel. It
  deliberately avoids adding many weak sources and may understate genuinely
  additive contamination.
- Source provenance is source-resolved only for maximum overlap. Summed halos
  retain scores but cannot make an unambiguous single-source pixel assignment.
- A high fraction can occur on a low absolute signal denominator. Use the
  attributable/classic intensity layers alongside the fraction.
- Raw marker images and masks must be on the same native grid; the stage never
  silently resizes either input.
