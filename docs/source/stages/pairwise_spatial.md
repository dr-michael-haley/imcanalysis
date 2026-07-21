# Pairwise Spatial Analysis

## What this stage does

Pairwise spatial analysis asks whether cells carrying one population label tend
to occur near cells carrying another label. The active stage applies three
complementary measurements to cell centroids within individual regions of
interest (ROIs):

1. **Squidpy neighbourhood enrichment** counts pairs joined in a radius-based
   spatial graph and compares those counts with shuffled population labels.
2. **Nearest-population distance** measures the distance from each source cell
   to the closest target cell and compares it with shuffled target labels.
3. **Cross-pair-correlation function (PCF)** measures the local density of a
   target population in successive distance bands around a source population,
   relative to a homogeneous spatial-randomness reference.

These analyses describe different aspects of spatial organisation. Agreement
between them can strengthen a biological hypothesis, but they are not three
replicate tests of the same quantity and their numerical values cannot be
compared directly.

| Analysis | Main question | Reference or null | Attraction-like result | Depletion-like result |
|---|---|---|---|---|
| Squidpy | Are more graph edges present between two labels than expected? | Population labels permuted on the fixed spatial graph | Positive z-score | Negative z-score |
| Nearest distance | Is the closest target nearer than expected to a source cell? | Target labels permuted among fixed cell coordinates within each ROI | Negative delta or z-score | Positive delta or z-score |
| PCF | Is target density in an annulus around source cells greater than a homogeneous random expectation? | Expected target count from global ROI density with rectangular edge correction | `g(r) > 1` | `g(r) < 1` |

The direction of the nearest-distance result is deliberately opposite to the
other two: a smaller distance means greater proximity.

```{important}
This stage detects spatial association, not mechanism. Proximity can be
consistent with recruitment, signalling, a shared anatomical niche, common
exclusion from another compartment, or an acquisition or segmentation
artefact. It cannot by itself establish direct cell contact, molecular
interaction, migration, or causality.
```

## Biological information required

The stage reads the following information from `adata.obs`:

- a population identity for each cell;
- an ROI or spatial-unit identifier;
- x and y cell-centroid coordinates;
- a unique cell identifier for the distance analysis; and
- optionally, a biological or experimental group assigned to each ROI.

The source labels may come from clustering, manual annotation, or another
classifier. Their quality is part of the spatial analysis: mixed clusters,
doublets, debris, segmentation fragments, or inconsistent annotation can all
create apparently meaningful spatial relationships.

Every calculation is confined to an ROI or configured subregion. Cells from
different images are never treated as neighbours. The settings use names ending
in `_um`, but the code does not convert coordinate units. A radius of 20 means
20 coordinate units. It represents 20 micrometres only when the supplied
centroids are already measured in micrometres.

## Test 1: Squidpy neighbourhood enrichment

### Constructing the neighbourhood graph

The pipeline normally analyses each ROI separately. Within an ROI, Squidpy
constructs an undirected spatial graph from cell centroids using the configured
radius interval. With the defaults, cells separated by more than 0 and at most
20 coordinate units are connected. The graph represents centroid proximity; it
does not use segmentation-boundary contact.

The analysis then counts graph connections between every pair of population
labels. A connection means that two centroids fall inside the selected radius,
not that the cells necessarily touch or communicate.

### The label-permutation reference

Squidpy repeatedly rearranges population labels while retaining the graph. This
keeps the observed cell positions, graph topology, and overall population
abundances but breaks the association between identity and position. For each
population pair, the shuffled results provide a null mean and standard
deviation for the number of graph connections.

The output contains two metrics:

- `count`: the observed interaction count in the spatial graph;
- `zscore`: the observed count minus the permutation mean, divided by the
  permutation standard deviation.

A positive z-score means that the pair has more radius-graph connections than
expected after label shuffling. A negative value means fewer. Values near zero
are compatible with the shuffled-label reference.

### Sensible biological interpretation

A strongly positive macrophage-to-T-cell z-score at a 20-micrometre radius is
consistent with those labelled populations sharing local neighbourhoods more
often than their abundances and the observed graph would suggest. It is not
proof that macrophages recruit T cells or that the cells physically contact one
another. Both may instead accumulate along the same tumour boundary, vessel, or
damaged anatomical structure.

The raw `count` is useful for checking support, but it should not be used alone
to compare population pairs. Common populations naturally generate more edges
than rare populations, and larger or denser ROIs generate more edges than small
or sparse ROIs. The z-score adjusts against shuffled labels, but it is still
conditional on the observed graph and population labels.

The pipeline does not calculate or save an empirical Squidpy P value. The
z-score is a standardized deviation, not automatically a normally distributed
test statistic, and no multiple-testing correction is applied across the many
population pairs.

### What changes the result

The radius defines the biological scale. A small radius focuses on immediate
neighbours, whereas a larger radius increasingly describes shared tissue
compartments. Cell density, irregular ROI borders, empty tissue, segmentation
size, and coordinate calibration all change the graph. The Squidpy calculation
does not correct graph counts for missing tissue beyond an ROI boundary.

## Test 2: nearest-population distance permutations

### The observed distance

For every cell in an ROI, the pipeline uses a k-d tree to find the Euclidean
distance to the nearest cell of each target population. If source and target are
the same population, the source cell is excluded from its own search. A
same-population distance is therefore the distance to the nearest *other* cell
of that type; it is missing when the ROI contains only one such cell.

The cell-level distances are then grouped by ROI and source population. Thus a
row such as `B cell -> macrophage` describes, among the original B cells in an
ROI, the average distance to the nearest macrophage.

`source_population_obs` can hold a separate set of source labels. When it is
unset, the same annotation column defines both source and target populations.
The measurement is directional: `A -> B` and `B -> A` need not be equal because
one abundant B cell can be the nearest neighbour of many A cells.

### What the code calls a bootstrap

For each ROI, the code repeatedly permutes the **target population labels**
among the observed cell coordinates. It does not move cells, change target
population counts, or resample patients. Original source labels are retained
when cell-level values are summarized by source population.

Although configuration and filenames use the word *bootstrap*, this is a
random-label permutation reference. It asks whether the observed association
between target identity and position differs from one produced by assigning the
same target labels randomly to the existing cells in that ROI.

### Distance output metrics

Four ROI-by-source-by-target values are saved:

- `observed`: mean observed nearest-target distance;
- `bootmean`: mean nearest-target distance across label permutations;
- `delta`: `observed - bootmean`;
- `zscore`: the cell-level observed-minus-null difference divided by its
  permutation standard deviation, then averaged within ROI and source
  population.

Interpret their signs carefully:

- negative `delta` or `zscore`: target cells are nearer than under shuffled
  labels, consistent with attraction or shared localisation;
- positive `delta` or `zscore`: target cells are farther away, consistent with
  segregation or exclusion;
- a value near zero: the observed nearest distance is close to the shuffled
  reference.

`observed` is in the coordinate units and is the most biologically tangible
quantity. `delta` reports the magnitude of departure in the same units.
`zscore` reports departure relative to permutation variability, but it is not a
P value and is not corrected for testing many population pairs.

### Sensible biological interpretation

Suppose tumour cells have an observed mean distance of 12 micrometres to the
nearest CD8 T cell, compared with a shuffled-label mean of 20 micrometres. The
delta is -8 micrometres and supports local proximity. It does not show that
every tumour cell is infiltrated: the mean may combine a highly infiltrated
margin with a large immune-excluded core. Inspect ROI points, distributions,
and source images rather than interpreting only the mean.

Nearest distance is especially sensitive to the abundance of the target
population. The label permutation preserves that abundance within each ROI,
which is useful, but rare targets still produce noisy and sometimes missing
estimates. A nearest-neighbour statistic also describes only the closest target
and ignores whether another 1 or 100 targets lie slightly farther away.

## Test 3: cross-pair-correlation function

### Relationship to SpOOx

The PCF implementation was adapted from the cross-PCF component of the
[Spatial Omics Oxford Pipeline (SpOOx)](https://github.com/Taylor-CCB-Group/SpOOx),
described in the [SpOOx lung IMC study](https://doi.org/10.1038/s41467-023-42421-0).
The paper used cross-PCF to quantify clustering or dispersal of one labelled
population around another over a range of spatial scales.

This repository does **not** reproduce the complete three-stage SpOOx
procedure. In the paper, quadrat correlation matrices first selected
co-occurring pairs, cross-PCF examined those pairs across distance, and an
adjacency cell network assessed physical contact. The active Pairwise Spatial
stage calculates PCF for all available ordered population pairs without the
SpOOx quadrat pre-screen and does not perform its segmentation-boundary contact
analysis. Its output should therefore not be described as a full SpOOx result.

### What g(r) measures

For source population A and target population B, the code centres annuli on
every A cell. Within each distance band it counts B cells and divides by the
number expected from:

- the overall density of B cells in that ROI; and
- the part of each annulus estimated to lie inside the rectangular ROI.

It then averages these normalized contributions across A cells. The resulting
cross-pair-correlation value is `g(r)`:

- `g(r) = 1`: target density around source cells matches the homogeneous random
  reference;
- `g(r) > 1`: more target cells occur in that distance band than expected,
  consistent with spatial enrichment or clustering;
- `g(r) < 1`: fewer target cells occur in that band than expected, consistent
  with spatial depletion or exclusion.

For example, `g(r) = 2` means that the estimated local target density in that
annulus is about twice its whole-ROI reference density. It does not mean that
twice as many source cells have a target neighbour, nor that the association is
statistically significant.

### Distance bins and the default 20-micrometre result

PCF is calculated in annuli with width `pcf_radius_step_um`. With the default
step of 10 and target distance of 20, the reported value counts target cells at
distances greater than 20 and at most 30 coordinate units from each source
cell. As in the SpOOx paper, the reported `r` is the lower edge of the band,
not an exact distance.

The implementation selects the available lower-radius value nearest to
`pcf_target_distance_um` and records both `requested_um` and `evaluated_um`.
Always check `evaluated_um` and `delta_um` when the requested value is not an
exact multiple of the step.

The SpOOx study focused on `g(r=20)` because 20--30 micrometres approximated
centroid separation for contacting cells in that lung IMC setting. That
interpretation is not universal. Cell size, mask expansion, tissue type, and
coordinate definition determine whether the same bin represents contact,
near-neighbour proximity, or a broader local niche in another dataset.

### Edge correction and spatial reference

An anchor near an image edge has less observable annular area than one in the
centre. The PCF code corrects for this by estimating the portion of each annulus
inside a rectangular domain. The rectangle starts at zero and extends to each
ROI's maximum x and y coordinates rounded up to the next 50 units.

This correction does not model an irregular tissue outline, holes, folds,
necrosis, lumina, or off-tissue areas inside the image rectangle. If both
populations are confined to the same tissue compartment, PCF may report
enrichment relative to a whole-rectangle homogeneous reference even without a
specific biological interaction between them.

### ROI and condition summaries

The stage saves a `g` value for every usable ROI and ordered population pair.
These ROI-level values support plots that show between-ROI variation.

It also pools anchor-cell contributions from ROIs assigned to the same
condition and calculates:

- `g_mean`: pooled condition-level mean PCF;
- `g_min`: lower grid-resampling bound;
- `g_max`: upper grid-resampling bound.

For these bounds, the current code divides rectangular image domains into
100-by-100 coordinate-unit grid cells and resamples grid cells with replacement.
This differs from the 20-micrometre square lattice described in the SpOOx paper.
It also pools spatial grid cells rather than resampling independent patients or
ROIs, so the bounds quantify spatial sampling variation under this
implementation, not uncertainty among biological subjects.

An interval entirely above one is consistent with robust enrichment under that
grid-resampling calculation, and an interval entirely below one is consistent
with depletion. An interval crossing one is inconclusive at that scale. These
bounds are not multiple-testing-adjusted confidence statements about a patient
population.

The active `crossPCF` accumulator is initialized at one before averaging over
the number of source cells. This adds `1 / number_of_source_cells` to each
estimated bin. The effect is small for abundant source populations but can
inflate `g(r)` visibly when very few anchor cells are present. Rare-population
PCF estimates should therefore be treated especially cautiously.

## Comparing the three analyses

The most useful interpretation asks whether several views of the data support
the same spatial story.

For a biologically plausible enriched pair, one might see:

- positive Squidpy z-scores at the chosen graph radius;
- negative nearest-distance deltas and z-scores;
- PCF above one at an appropriate distance band; and
- the same direction in several independent ROIs and biological cases.

Disagreement is informative rather than automatically erroneous. For example:

- PCF can show enrichment at 20--30 micrometres while nearest distance is
  unremarkable because PCF counts all targets in a band rather than only the
  closest target.
- Nearest distance can show strong proximity to a rare target while Squidpy
  counts remain small because only a few graph edges are possible.
- Squidpy can detect immediate neighbours while PCF at one selected radius
  misses an association occurring at a different scale.
- All three can report apparent association when both populations follow the
  same anatomical compartment.

Inspect the full biological context and, where possible, examine PCF curves
across radii rather than treating one distance as uniquely correct. The current
stage primarily exposes the configured target-radius PCF summaries, so a
different biological scale requires a rerun with different settings.

## Conditions, replication, and statistical inference

`groupby_obs` attaches a condition such as treatment, disease state, or tissue
class to ROI results and creates group-specific plots. Each ROI should have one
unambiguous group value. The stage does not fit a model comparing groups and
does not account for patients contributing multiple ROIs.

Likewise, barplot error bars summarize the available ROI or subregion values.
They are not a mixed-effects analysis, paired test, patient-level permutation,
or proof of a group difference. Formal biological inference should use an ROI-
or case-level table with a model appropriate to the experimental unit,
repeated-measure structure, covariates, and number of independent subjects.

Thousands of population pairs, radii, metrics, and groups can be inspected.
Selecting the largest effects after seeing all results creates a multiple-
comparison problem even when no P values are printed. Pre-specify important
pairs and distances where possible, report all evaluated comparisons, and
validate discoveries in independent material.

## Reading the plots

### Pairwise matrices

Rows are source populations and columns are targets. Distance matrices reverse
the configured colour map so that shorter values appear at the visual
enrichment end. PCF matrices are centred on one; Squidpy z-score and distance
delta/z-score matrices are centred on zero. Raw counts and observed distances
are not centred.

Hierarchical row or column clustering only reorders the display. It does not
define cell populations, discover spatial communities, or test significance.
When colour limits are calculated independently, the same colour can represent
different values in different group panels. Set
`pairwise_matrices_share_vmax_vmin: true` for direct visual comparison.

### Selected-pair and enrichment plots

`population_pairs` limits focused barplots; it does not limit the underlying
all-pairs calculations. The enrichment plots rank numerically high and low
targets for each source. For distance metrics, small values are ranked as
enriched; for Squidpy z-scores, positive and negative values define enriched
and depleted; for PCF, high and low values are ranked around the visual
reference of one.

The words *enriched* and *depleted* in these filenames and titles describe
ranking direction. The plotting step does not add another statistical test.
Keep individual ROI points visible where possible: a large mean driven by one
ROI is much less convincing than a consistent effect across independent cases.

## Important configuration choices

The most biologically consequential settings are:

- `population_obs`: the annotation being compared;
- `roi_obs`: the spatial unit within which relationships are permitted;
- `x_coord_obs` and `y_coord_obs`: calibrated centroid coordinates;
- `groupby_obs` and `groupby_obs_groups`: optional stratification and filtering;
- `squidpy_radius_min_um` and `squidpy_radius_max_um`: graph-neighbour scale;
- `distance_populations`: optional target subset for nearest-distance analysis;
- `pcf_target_distance_um`, `pcf_radius_step_um`, and `pcf_max_radius_um`: PCF
  scale and resolution;
- the permutation and bootstrap counts, which control Monte Carlo precision
  but do not create more biological replicates; and
- `population_pairs`, which selects focused plots but not tests.

`reload_saved_results` permits rapid plot-only reruns. Saved files are checked
for required columns but are not fully matched to the current population
labels, coordinates, radii, permutation counts, or other configuration. Delete
or move incompatible cached raw results, or set this option to false, after
changing analytical inputs.

## Outputs and audit trail

The stage writes its results below the Pairwise Spatial report location. Major
outputs include:

- `raw_data/squidpy_interactions/`: per-subregion count and z-score matrices,
  plus a combined long table;
- `raw_data/distance_bootstrap/`: observed, permutation-mean, delta, and z-score
  tables plus ROI/source/target long data;
- `raw_data/pcf/`: SpOOx-style input files, condition and ROI PCF summaries,
  long tables, and selected cell metadata;
- `plots/pairwise_matrices/`: all-population matrices;
- `plots/selected_pairs/`: plots requested through `population_pairs`;
- `plots/enrichment_plots/`: ranked per-source target summaries;
- `metadata/anndata_obs_snapshot.csv.gz` and ROI metadata; and
- `pairwise_spatial_run_metadata.json`: resolved analysis switches, plotting
  controls, selected pairs, and whether each result was computed or reloaded.

The AnnData is saved back to its pipeline path with a stage-completion record,
but the pairwise statistics themselves are report tables rather than new
cell-level annotations.

The observation snapshot and PCF input metadata can contain clinical or sample
information. Review configured metadata fields and access controls before
sharing the report.

## Common problems and limitations

- **Coordinate units are assumed.** No pixel-to-micrometre conversion occurs.
- **The ROI is the analysis boundary.** Cropping can remove relevant neighbours,
  and different ROI sizes or selection strategies can bias comparisons.
- **Population abundance matters.** Rare source or target populations give
  unstable, missing, or extreme values even when the null preserves counts.
- **ROIs are not necessarily independent subjects.** Treating several ROIs from
  one case as independent inflates apparent replication.
- **Tissue architecture is a confounder.** Shared localisation to epithelium,
  stroma, vessels, tumour, necrosis, or an invasive margin can resemble direct
  attraction.
- **Segmentation affects distance.** Centroid distances ignore cell shape and
  mask contact, while merged or fragmented masks alter density and labels.
- **Annotation uncertainty propagates.** A spatial statistic cannot rescue an
  incorrect or overly broad population definition.
- **The nulls are conditional.** Label shuffling asks about identity on the
  observed cellular pattern; PCF uses a homogeneous rectangular-density
  reference. Neither represents every biologically plausible null model.
- **No group-difference test is performed.** Group panels and their error bars
  are descriptive.
- **No stage-wide multiple-testing correction is performed.** Large absolute
  z-scores or PCF bounds should be interpreted with effect size, replication,
  prior hypotheses, and image review.

Before reporting a pair as biologically associated, inspect the underlying
images, verify masks and labels, show ROI-level values, confirm the coordinate
scale, consider abundance and anatomy, and repeat the observation across
independent cases.
