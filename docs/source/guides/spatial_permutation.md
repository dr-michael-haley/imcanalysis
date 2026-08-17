# Population overlap with categorical region masks

`SpatialBiologyToolkit.spatial_permutation` tests whether cell-population
centres overlap categorical tissue regions more or less often than expected
from the regions' pixel areas. It is intended for rasters such as Pixie pixel
environments, histology annotations, or image-derived metaclusters.

This is a local Python/notebook API. It does not define a pipeline or SLURM
stage.

## What the null model asks

The calculation is performed separately within every ROI. For a population
with `n` analysed cells and an environment covering `K` of `N` selected tissue
pixels, the null asks:

> If the same number of population centres were placed uniformly across the
> selected tissue pixels, how many would fall in this environment?

The randomized overlap count follows a hypergeometric distribution. The API
draws directly from that distribution for each population/environment pair.
This is marginally equivalent to repeatedly sampling unique pixel coordinates
and shuffling the fixed population labels, but it avoids large temporary
coordinate tables and crosstabs.

- Positive z-score: more cell centres overlap the environment than expected.
- Negative z-score: fewer overlap than expected.
- Z-score near zero: overlap is close to the area-based null expectation.
- Missing z-score: the null standard deviation was zero, so a standardized
  score is not defined.

The test concerns area overlap at cell centres. It is not a cell-cell
interaction test and does not test whether cells cluster together inside an
environment.

## Validate alignment first

Mask lookup uses an exact pattern rather than a substring search. This avoids
mistaking an ROI such as `acq_1` for `acq_10`.

```python
import anndata as ad

from SpatialBiologyToolkit.spatial_permutation import (
    spatial_mask_alignment_qc,
)

adata = ad.read_h5ad("cells.h5ad", backed="r")
qc = spatial_mask_alignment_qc(
    adata,
    "Pixie_pixel_masks",
    pop_col="population",
    roi_col="ROI",
    x_col="X_loc",
    y_col="Y_loc",
    mask_pattern="{roi}_pixel_mask.tiff",
    tissue_exclude_values=(0, 29),
)
print(qc)
```

Inspect `n_out_of_bounds`, `n_excluded_by_tissue`, and the two coverage
fractions before running permutations. Coordinates are indexed as
`mask[y, x]`. The default `coordinate_rounding="truncate"` reproduces the
integer cast used in the legacy notebook; choose `"floor"` or `"round"`
explicitly when that matches the coordinate-generation workflow better.

## Calculate reproducible z-scores

```python
from SpatialBiologyToolkit.spatial_permutation import (
    spatial_permutation_zscores,
)

environment_names = {
    1: "Environment 1",
    2: "Environment 2",
}

results = spatial_permutation_zscores(
    adata,
    "Pixie_pixel_masks",
    pixel_dictionary=environment_names,
    n_permutations=2500,
    tissue_exclude_values=(0, 29),
    pop_col="population",
    roi_col="ROI",
    x_col="X_loc",
    y_col="Y_loc",
    mask_pattern="{roi}_pixel_mask.tiff",
    n_jobs=5,
    random_state=20260817,
)
results.to_csv("population_pixie_zscores_by_roi.csv", index=False)
```

The output retains the legacy columns `roi`, `pixel`, `population`,
`observed`, `perm_mean`, `perm_std`, and `z_score`. It also records the
population cell count, environment pixel count and fraction, valid tissue
area, analysed ROI cell count, exclusion counts, and number of permutations.
These columns make the result auditable without reopening every mask.

Use a fixed `random_state` for rerunnable notebooks. Random streams are spawned
per ROI, so results are stable when `n_jobs` changes.

## Interpretation and aggregation

Keep the per-ROI table as the primary result. When several ROIs come from the
same case, average ROI z-scores within case before calculating a cohort mean;
otherwise cases represented by more tissue cores receive more weight. Always
report the number of contributing ROIs and cases alongside summaries.

Do not transfer an environment-name dictionary from another Pixie model unless
the cluster identities are known to be shared. Numeric mask values are safer
than plausible but incorrect biological names. Likewise, exclude values such
as background or artifact only when their meaning is established for the mask
set being analysed.
