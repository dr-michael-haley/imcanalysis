# Neighbour-attributable signal analysis

## What this stage does

`neighsig` learns an empirical, marker-specific spatial halo from convincing
positive cells selected automatically or manually, projects that halo from other
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

By default the input expression matrix supplies only the marker-positive call
used to find automatic exemplar candidates. This is normally a Nimbus inference
score. Halo values, raw-image source selection, background estimation, and final
scores do not use `X`. The original matrix is preserved for comparison, but the
selected training cells are not an independent validation of the `X` positivity
call; held-out cells and ROIs remain useful comparisons.

## Main inputs

- `general.anndata_path`: input AnnData, read without modification. Its cell and
  marker order defines the output axes.
- `adata.X`: marker-positive evidence for automatic and augment exemplar modes.
  The default candidate cutoff is 0.5, appropriate for the usual 0-1 Nimbus
  inference score, and must be changed when `X` uses another scale.
- `adata.obs[neighbour_signal.exemplar_obs]`, default `Exemplar_stains`: used in
  manual and augment modes. Non-null values must exactly match `adata.var_names`
  and identify convincing positive exemplars.
- `general.raw_images_folder`: the existing ROI/channel TIFF layout. Marker
  names are resolved with the same suffix-aware image discovery used by
  CellVision.
- `general.masks_folder`: one labelled TIFF mask per ROI.
- `general.roi_obs` and `neighbour_signal.object_id_obs`, defaulting to `ROI`
  and `ObjectNumber`: the exact AnnData-to-mask mapping. Every pair must be
  unique and present in its ROI mask.

In automatic mode, an X-positive candidate is rejected when another X-positive
cell for the same marker is closer than the configured 10-pixel mask distance.
Other segmented cells may be nearby: their pixels are excluded from the radial
average, and the candidate remains eligible when every halo bin retains enough
unassigned pixels. Eligible cells are sampled deterministically across ROIs and
thirds of the positive X-score range rather than selecting only the highest
scores. The same cell may be an exemplar for more than one marker.

Unknown manual exemplar values are reported and ignored. A marker is skipped when
fewer than `min_exemplars` valid selected profiles remain. Missing images,
masks, mapped labels, or shape agreement are input errors because pixel-level
alignment is required.

Segmentation masks may contain additional labels absent from the input AnnData,
for example small objects deliberately removed during cell-size filtering.
These mask-only objects remain occupied segmentation geometry: their pixels are
excluded from exemplar radial averages, and neighbourhoods around strong
mask-only labels are excluded from ROI background selection. They do not
project halos or appear in source provenance because every reported source must
map to an authoritative row of the output AnnData. The stage records mapped,
mask-only, projected, and strong mask-only counts for every ROI and marker
rather than treating this expected filtering pattern as an error.

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
- profiles, IQRs, complete automatic/manual candidate decisions, exemplar
  statistics, backgrounds, mapped/mask-only segmentation counts, parameters, worker
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

- one empirical halo-curve figure per marker with exemplar IQR, exemplar count,
  and source threshold; skipped markers receive an explicit unavailable-profile
  panel with the reason instead of silently disappearing;
- one score-distribution figure per marker and an all-marker CSV with median,
  90th/95th percentiles, fractions above descriptive 0.25/0.5/0.75 thresholds,
  exemplar counts, and source thresholds; all-zero distributions explicitly
  distinguish unavailable profiles from valid profiles with no attributable
  cells;
- long-form profile, selected-exemplar, automatic candidate/selection,
  ROI-background/source-mapping, skipped-marker, and unknown manual-exemplar
  tables;
- one plot per marker of input-X score versus nearest same-marker-positive
  distance, distinguishing rejected, eligible-unsampled, and selected
  candidates;
- one Scanpy UMAP per marker, plus separate `halo_max_score` and
  `halo_mean_score` UMAPs, when `X_umap` exists; point size is configurable;
- one Scanpy population matrix plot per marker when a suitable categorical
  population observation is available. Populations use a shared dendrogram
  learned from all available halo-score markers, while each marker uses its own
  native colour maximum;
- a source-population-to-target-population marker summary and one heatmap per
  marker, optionally excluding same-population routes;
- a dominant-source summary reporting concentration above 50% of attributable
  signal, the median number of contributing sources, and common population
  routes;
- bounded native-pixel contact sheets for every marker with suitable examples:
  target-source sheets compare raw signal, observed excess, projected halo,
  attributable signal, residual signal, and the pixelwise winning source;
  exemplar sheets show the source mask, unassigned radial pixels, and the
  individual profile against the marker median/IQR; automatic-decision sheets
  show selected and rejected X-positive candidates, nearby same-marker
  X-positive cells, and the unassigned pixels available in each halo bin;
- `cell_gallery_manifest.csv`, linking every displayed crop to its marker, ROI,
  authoritative AnnData row/cell identifiers, selection category, crop bounds,
  source relationship, and relevant quantitative scores;
- one sampled classic-intensity versus original-`X` plot per marker, coloured
  by the halo score; and
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
  exemplar_mode: automatic
  exemplar_obs: Exemplar_stains
  automatic_positive_threshold: 0.5
  automatic_same_marker_clearance_px: 10
  automatic_target_exemplars_per_marker: 30
  automatic_max_exemplars_per_roi: 5
  automatic_min_pixels_per_bin: 8
  object_id_obs: ObjectNumber
  max_halo_px: 8
  source_anchor_dilation_px: 2
  source_anchor_quantile: 0.95
  min_exemplars: 5
  source_threshold_quantile: 0.10
  halo_aggregation: max
  calculate_classic_intensities: true
  n_jobs: auto
  umap_point_size: null
  create_cell_galleries: true
  gallery_examples_per_marker: 6
  gallery_crop_margin_px: 8
  population_obs: null
  source_target_qc_exclude_same_population: true
  high_risk_threshold: 0.5
```

Legacy `qc_markers` and `max_qc_markers` values are still accepted so existing
configuration files validate, but they no longer restrict report generation.
Every marker on the AnnData axis receives the applicable per-marker QC output.

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

Only cells represented in the input/output AnnData are eligible to project
named halos. Strong mask-only labels are still used to protect background
estimation from their local signal, but excluding them from projection preserves
the deliberate upstream cell filter and the invariant that every source index
identifies an output AnnData row. Inspect `roi_marker_backgrounds.csv` and stage
warnings to quantify this exclusion.

`sum` remains supported for the original score and is more aggressive in dense
source regions, but multiple sources contribute simultaneously to a pixel.
Source-resolved provenance is therefore disabled for `sum` with a clear
warning: the sparse table is empty and dominant-source layers use `-1`/zero
sentinels. Use the recommended `max` behavior when source identity is needed.

## Environment and resources

The stage reuses `imc_segmentation`; its existing NumPy, SciPy, tifffile,
AnnData, Scanpy, pandas, and matplotlib stack covers the analysis. No new
environment or package is required.

The active wrapper requests 6 CPUs, 256 GB RAM, and 24 hours on the CPU
high-memory partition. `n_jobs: auto` resolves the available worker count from
the SLURM allocation, process affinity, and host CPU count. Automatic candidate
inspection, profile extraction, and profile application are parallelized
independently by ROI. Every worker
loads one mask once, reads marker TIFFs sequentially, and projects sources on
small radius-expanded bounding boxes. BLAS/OpenMP threads are fixed to one to
avoid nested oversubscription.

## How to interpret the results

Begin with the learned profile figure. A marker should have a plausible,
reasonably supported curve and adequate exemplar IQR before its cell scores are
trusted. A peak just outside the segmentation boundary can be biologically
reasonable for membrane or cytoplasmic localisation.

Next inspect the automatic exemplar selection table/plot, followed by the
cell galleries, all-marker score table, and UMAP/population views. The
target-source gallery is a direct visual decomposition of the same pixels used
for the score: cyan outlines identify the target, magenta the dominant spatial
source, and orange any other contributing sources. The winning-source panel is
categorical provenance under `halo_aggregation: max`; it should be read with
the attributable and residual panels rather than as a standalone segmentation
classification. The exemplar and automatic-decision galleries help identify
profiles learned from background structure, truncated radial support, or
unrepresentative positives before interpreting cell-level results.

Gallery examples are selected deterministically and stratified across useful
QC cases and ROIs where possible. They are not a random or exhaustive sample;
use `cell_gallery_manifest.csv` to audit the selection and join displayed cells
back to the AnnData and source-target Parquet table. Increasing
`gallery_crop_margin_px` changes only the displayed context, never the learned
profile or score.

A high score
means that the strongest nearby source, or the configured overlap aggregate,
could spatially explain much of the observed target-mask excess at the same
pixels. Compare classic intensity and preserved input `X` to determine
whether the existing expression method already suppresses these spatially
suspicious measurements. Because automatic selection uses the original `X` to
define exemplar positivity, treat agreement on selected cells as expected and
give more weight to held-out cells and ROIs when assessing Nimbus behaviour.

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

- Too few spatially eligible automatic exemplars produce a skipped marker.
  Review the candidate table, adjust the X cutoff only when justified by its
  measurement scale, increase the per-ROI cap for few-ROI datasets, or use
  `augment`/`manual` with convincing cells across representative ROIs.
- An all-zero score distribution means either that no reliable marker halo was
  learned or that a valid learned halo explained no signal in any target cell.
  Per-marker distribution figures state which case applies; confirm it with
  `halo_profile_available` and `halo_skip_reason` in the marker metadata table
  or output AnnData `.var`.
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
