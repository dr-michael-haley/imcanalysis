# Nimbus normalization scan

## What this stage does

The `nimbus-scan` stage tests how marker-specific Nimbus upper normalization
bounds (`Vmax` values) affect model-derived cell scores before cell tables, AnnData,
UMAP, clustering, or population identification exist. It runs each marker
independently over a marker-specific ROI subset spanning low-to-high input
intracellular signal, then repeats Nimbus inference across a candidate Vmax grid.

Run it as a managed GPU job:

```bash
sbt run nimbus-scan
```

The stage uses the same image discovery, adjusted masks, normalization clipping,
magnification handling, padding, checkpoint, test-time augmentation, and
per-cell confidence averaging as the normal `nimbus` stage.

## Why it is performed

Changing a normalization bound changes the image that Nimbus sees. Small Vmax
values amplify and clip more pixels; large values suppress the normalized image.
The resulting cell-score response need not be linear. A marker can be insensitive
over a broad Vmax plateau or can cross a narrow region where many cells change
their positive/negative call at once.

The older normalization-only Nimbus QC thresholds the **mean normalized input
intensity**, not the Nimbus output. It therefore cannot answer how Nimbus cell
scores or positive-call proportions change. `nimbus-scan` runs the model itself
and measures those quantities directly, without paying for downstream AnnData or
clustering work.

## Main inputs

- `general.metadata_folder/panel.csv` selects marker channels and raw versus
  denoised image sources using the same rules as Nimbus.
- `general.metadata_folder/metadata.csv` applies ROI import exclusions.
- `general.masks_folder` supplies labelled instance masks.
- `general.raw_images_folder` and `general.denoised_images_folder` supply marker
  images according to panel selection and `nimbus.allow_raw_fallback`.
- The `nimbus` config section supplies mask adjustment, model, magnification,
  checkpoint, device, batch-size, normalization-quantile, and augmentation
  settings.

No AnnData, clustering, UMAP, population column, or exemplar population is read.
Before loading the Nimbus model, the default ROI-selection pre-pass applies the
adjusted cell mask to each marker image. For every marker/ROI it takes all finite
intracellular pixels strictly above that marker's lower threshold and calculates
their mean. ROIs are ranked by this value, and approximately quantile-spaced
ranks, including the low and high ends, are selected independently per marker.
This is an ROI-level image statistic; it does not create a single-cell table.

The simplest input mode is a scan-parameter CSV:

```csv
marker,baseline_vmax,lower_threshold
CD3,20,0.5
CD20,12,0
FOXP3,5,0.8
```

Set its path in `nimbus_normalization_scan.baseline_normalization_dict_path` and
leave `markers: null`. The CSV rows then define the marker subset, baseline Vmax,
and lower threshold, so those per-marker values do not need duplicating in
`config.yaml`. Keep `vmax_factors` in config; every factor is multiplied by the
corresponding CSV baseline.

The preferred headers are `marker`, `baseline_vmax`, and `lower_threshold`.
Canonical Nimbus `marker,vmax,lower_threshold` files are also accepted, as are
the aliases `baseline` and `lower_bound`. The input is read-only. If `markers` is
set explicitly, it continues to define the scan subset and extra CSV rows are
ignored. Per-marker YAML mappings, when deliberately supplied, retain their
existing precedence over file values.

Without a baseline file, Vmax values are computed exactly like the toolkit's
Nimbus normalization: the configured in-mask quantile is calculated per usable
ROI, averaged across all usable ROIs, and floored at
`nimbus.normalization_min_value`. Legacy scalar JSON dictionaries remain readable
for baseline values and imply a zero lower threshold, but do not define the
marker subset.

To provide the current values directly in `config.yaml`, set
`nimbus_normalization_scan.marker_baseline_vmax`. Each marker's supplied value
becomes its baseline and is multiplied by every value in `vmax_factors`. Markers
omitted from this mapping fall back to the baseline dictionary path, when set,
or to the computed mask-aware quantile.

## Reusable assets produced or modified

None. The scan does not write or alter the canonical `normalization_dict.csv`,
legacy JSON, Nimbus cell tables, confidence-map folders, AnnData, images, masks,
or metadata.

The report includes `suggested_normalization_dict.csv`, but it is deliberately a
review-only human output. Copy reviewed values into the canonical Nimbus
dictionary and enable `nimbus.reuse_saved_normalization` only after examining the
marker diagnostics.

## Human-facing outputs produced

The managed execution contains:

- `figures/nimbus_normalization_scan/<marker>.png`: four-panel marker reports
  showing Nimbus score quantiles, aligned stacked histograms (one per tested
  Vmax), positive-cell fractions at each configured score threshold, in-mask
  lower-threshold removal, saturation, adjacent-value call flips, and sensitivity
  per two-fold Vmax change;
- `tables/nimbus_normalization_scan/normalization_scan_candidates.csv`: one row
  per marker/Vmax candidate with score, saturation, adjacency, and stability
  metrics;
- `normalization_scan_thresholds.csv`: positive-cell proportions at every
  configured Nimbus-score threshold;
- `normalization_scan_rois.csv`: ROI-level positive fractions, score medians, and
  saturation values, making cohort heterogeneity visible;
- `normalization_scan_recommendations.csv`: stable ranges, provisional suggested
  values, cliff locations, and explicit manual-review reasons;
- `normalization_scan_baselines.csv`: the baseline used for each marker and
  whether it came from `marker_baseline_vmax`, a dictionary file, or a computed
  quantile, plus the lower threshold and its source;
- `normalization_scan_roi_selection.csv`: every marker/ROI pre-pass statistic,
  including intracellular and above-background pixel counts, qualifying-pixel
  fraction, mean signal above background, expression rank/quantile, and whether
  that ROI was selected;
- optional compressed `cell_scores/<marker>.csv.gz` tables with every cell score;
- `summaries/nimbus_normalization_scan/normalization_scan_summary.md`; and
- `files/nimbus_normalization_scan/suggested_normalization_dict.csv`, with the
  suggested Vmax and applied lower threshold for every marker.

## Important configuration options

```yaml
nimbus_normalization_scan:
  # null uses the CSV rows as the scan marker list
  markers: null

  # null selects ROIs automatically per marker; explicit names override this
  rois: null
  roi_selection_strategy: marker_expression_range
  max_rois: 10       # per marker; 0 means all usable ROIs
  random_seed: 0     # used only by the legacy-compatible random strategy

  # CSV rows define markers, baseline Vmax, and lower threshold when markers is null
  baseline_normalization_dict_path: metadata/nimbus_scan_parameters.csv

  # half-octave grid from 0.25× to 4× baseline
  vmax_factors: [0.25, 0.353553, 0.5, 0.707107, 1, 1.414214, 2, 2.828427, 4]

  # Optional config-only alternative; omit when the CSV supplies these values
  marker_baseline_vmax: {}

  # Optional config override; omit when the CSV supplies lower thresholds
  marker_lower_thresholds: {}

  # optional absolute grids for markers needing narrower or shifted scans
  marker_vmax_values: {}

  positive_score_thresholds: [0.25, 0.5, 0.75]
  primary_positive_score_threshold: 0.5
  stability_tolerance: 0.05
  call_flip_tolerance: 0.05
  saturation_tolerance: 0.05
  cliff_tolerance: 0.15
  save_cell_scores: true
```

Each marker needs at least three distinct candidate values. The default nine-point
grid covers four octaves and includes the baseline. A narrower explicit grid is
useful after an initial broad scan finds a sharp transition.

For example, `FOXP3: 5` in `marker_baseline_vmax` with factors `[0.5, 1, 2]`
tests Vmax values `[2.5, 5, 10]`. A marker cannot appear in both
`marker_baseline_vmax` and `marker_vmax_values`: the former defines one centre
to multiply by the shared factors, while the latter defines the complete
absolute candidate grid.

The same factors applied to a CSV row `FOXP3,5,0.8` also test
`[2.5, 5, 10]`, with `0.8` held fixed as the lower threshold throughout.

For every candidate, the exact input transform is
`clip((image - lower_threshold) / (vmax - lower_threshold), 0, 1)`. Pixels at or
below the lower threshold therefore become zero. Lower thresholds are fixed
during a Vmax scan, must be non-negative, and must remain below the smallest
candidate Vmax for that marker. Omitted markers use zero, which preserves the
old `image / vmax` behavior.

`max_rois` limits only repeated model inference and is applied independently to
each marker. A computed baseline still uses all usable ROIs so it matches the
normal Nimbus stage. Explicit `rois` apply the same named subset to every marker
and are useful for known positive/negative controls. Set
`roi_selection_strategy: random` to reproduce the earlier seeded, cohort-wide
random sampling behavior.

The expression-range score is the mean of pixels above the lower threshold, not
the mean of every in-mask pixel. Review `above_background_fraction` alongside it:
a high mean supported by only a tiny fraction of pixels may represent a small
bright structure or artifact rather than broadly positive tissue.

## How to interpret the results

The primary positive-score threshold defines a diagnostic binary call. A score of
0.5 is convenient but is not a universal biological truth for every marker or
imaging domain. The additional thresholds show whether an apparent plateau is
specific to that cutoff.

For each candidate, the stage calculates:

- the Nimbus score distribution across cells;
- vertically stacked, common-bin histograms that make distribution shifts at
  every tested Vmax directly comparable;
- positive-cell proportions globally and by ROI;
- the fraction of in-mask pixels removed by the configured lower threshold;
- the fraction of in-mask pixels clipped at normalized value 1;
- the fraction of individual cells whose primary call flips relative to an
  adjacent candidate;
- the change in primary positive fraction per log2 Vmax unit; and
- the largest adjacent positive-fraction shift (a possible Vmax cliff).

A candidate is provisionally stable when its local primary-positive-fraction
sensitivity, individual-cell call-flip rate, and input saturation are all below
their configured tolerances.
Contiguous stable candidates form a stable range. The suggested value is the
candidate nearest the baseline within the nearest stable range. If no candidate
qualifies, the least-sensitive candidate is reported and the marker is flagged.

Treat the suggestion as a way to prioritize review:

1. Prefer a broad stable range over a single apparently optimal point.
2. Examine markers with cliffs, boundary suggestions, high saturation, or no
   stable range first.
3. Check ROI-level curves for batches or tissues that respond differently.
4. Review original/denoised images and appropriate positive and negative controls.
5. Re-run a narrower explicit grid around a plausible interval when needed.
6. Only then transfer reviewed values to the Nimbus normalization dictionary.

The stage cannot decide whether a stable all-negative result reflects true
biology, weak staining, a poor antibody, or excessive Vmax. Likewise, a stable
all-positive result can be biological or artifactual. Those cases are flagged as
threshold-degenerate and require image/control review.

## Common problems and limitations

- **Compute cost:** the work scales approximately with markers × Vmax candidates
  × scan ROIs. Test-time augmentation multiplies inference cost further. The
  expression-range pre-pass additionally reads every scanned marker image and
  mask once per usable ROI, but does not run Nimbus or construct cell tables.
- **ROI sampling:** expression-range sampling covers each marker's low-to-high
  ROI distribution, but it is not stratified by patient, batch, tissue type, or
  morphology. Use explicit representative/control ROIs or increase `max_rois`
  when those dimensions matter.
- **Threshold dependence:** positive-cell proportions depend on the selected
  Nimbus-score threshold. Use continuous score distributions and multiple
  thresholds rather than trusting one binary curve.
- **No biological ground truth:** stability means insensitivity to Vmax, not
  correct staining interpretation.
- **Checkpoint drift:** `nimbus.checkpoint: latest` can resolve differently over
  time. Use a fixed checkpoint for exact comparisons.
- **Input changes:** repeat the scan after changing denoising, image scale, panel
  selection, masks, mask offsets, cell-area filters, or checkpoint.
- **Range boundaries:** a suggested value at the smallest or largest scanned Vmax
  means the grid probably needs extending.

The stage requests the same segmentation environment and one GPU as Nimbus. Its
default ROI bound is intended to keep an exploratory scan smaller than a full
cohort inference run; it is not safe for a login node.
