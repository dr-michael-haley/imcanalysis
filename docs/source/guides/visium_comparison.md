# Registered Visium–IMC comparisons

`SpatialBiologyToolkit.visium_comparison` provides read-only Python helpers for
local notebooks after cell2location mapping. It extracts reusable analysis from
the historical Visium notebook without requiring cell2location, PyTorch or a GPU.
Use the existing scientific SBT environment (NumPy, pandas, SciPy, matplotlib,
seaborn); AnnData/decoupler are only needed by caller-owned expression workflows.

## Import and align the current exports

```python
from SpatialBiologyToolkit import visium_comparison as vc

means, qc = vc.read_cell2location_tables(
    "cell2location/tables", adata_vis.obs,
    library_key="library_id", barcode_key="Barcode",
)
coarse = vc.aggregate_populations(means, annotation_mapping)
counts = vc.spot_population_counts(
    imc_obs, spot_key="VisSpot", population_key="Population"
)
keep = vc.spot_filter(counts, min_cells=5, purity=0)
result = vc.correlate_spot_features(
    vc.population_fractions(coarse),
    vc.population_fractions(counts.loc[keep]),
    groups=adata_vis.obs["Case"],
)
fig = vc.plot_correlation_heatmap(result, title="Within-case fractions")
```

The importer joins by **library and barcode**, validates unique identities, and
returns tables indexed in the requested Visium observation order. It reads
`mapping_mean_abundance.csv[.gz]` (or `mapping_q05_abundance.csv[.gz]`) together
with `mapping_spot_qc.csv`; compressed exports take precedence. A subset of the
export can be requested, but missing requested spots raise an error. No barcode
suffix guessing is performed.

Crosswalk aggregation requires exactly one coarse label for every observed
granular label. Posterior **means** are additive. Summing marginal q05 values
does not produce a coarse posterior q05; retain granular q05 only as a separate
sensitivity analysis unless joint posterior draws are available.

## Cell registration and denominators

Establish that the cell-to-spot mapping belongs to the current cell segmentation
before joining it. For tiled images, compare tile-offset coordinates and source
cell IDs; check case/library agreement and reconcile observed spot totals.
The helper cannot establish physical registration quality from a table alone.

No unobserved IMC spots are added to `spot_population_counts`. Zeros mean an
absent population in a spot with labelled cells, not unmeasured tissue. Missing
labels in assigned cells raise. Exclude artifacts explicitly and report their
frequency before calculating biological fractions.

Purity is the maximum fraction over the complete supplied IMC population level.
Apply this filter **before** selecting particular populations. `min_cells` is
inclusive; use 6 to reproduce the original notebook's `>5` threshold. There is
no global-variable dependency. Cross-resolution purity filters retain different
spots, so use no purity threshold for primary cross-resolution comparisons.

## Correlation interpretation

The result is a tidy table of feature pairs, correlation, nominal p/BH q, matched
spot count, method and within-group flag. Finite cross-table tests in each call
form the FDR family. Constant features or too few spots have undefined results.
Use `method="spearman"` for rank sensitivity. With `groups`, centring removes
case offsets; Spearman ranks within each case before centring. Grouped results
omit p-values. Cases with more spots still contribute more, so also inspect
individual-case coefficients and leave-one-case-out stability.

Spatial spots are correlated and are not independent patients. Nominal spot
p-values are exploratory only. Four cases cannot support strong population-level
claims. Fractions remove total-cell burden but introduce compositional effects;
compare fraction and count results. All-pairs associations are co-occurrence,
not evidence that unlike annotation systems identify equivalent populations.

If IMC counts informed cell2location priors, total-count agreement is not an
independent validation. An exported `registered_count` may itself contain
previously imputed values; compare it with the original non-imputed count table.
Serial sections and partial IMC coverage further limit absolute agreement.

## Clustered correlation plots

`plot_correlation_heatmap` uses a seaborn clustermap with average-linkage,
Euclidean clustering of the signed, unscaled coefficients on both axes.
Pass `row_colors` and `col_colors` as label-keyed dictionaries/Series (or
DataFrames for multiple bands); colours and stars follow the clustered labels.
Missing palette entries are grey. Undefined coefficients remain grey and disable
clustering with a warning, rather than inventing values for the dendrogram.
Singleton axes are left unclustered.

```python
fig = vc.plot_correlation_heatmap(
    selected_comparison, title="RNA vs IMC",
    row_colors=rna_palette, col_colors=imc_palette,
    vmin=-0.5, vmax=0.5, cmap="RdBu_r", stars=True,
    figsize=(4.5, 5.2), linewidths=0.3, linecolor="black",
    dendrogram_ratio=0.025, square_cells=True, label_fontsize=7,
)
```

Limits control colour saturation only; the coefficients are not rescaled.
Use `row_cluster=False`/`col_cluster=False` to retain pivot order, or specify
`method` and `metric` for the dendrogram. The default return remains a matplotlib
Figure; `return_grid=True` exposes the seaborn ClusterGrid and reordered indices.

The compact default targets 0.16-inch square cells with measured label margins,
thin black borders (`linewidths=0.3`, in points), and small dendrograms.
`dendrogram_ratio=0.025` allocates 2.5% of the available panel dimension to each
tree; supply `(row_ratio, column_ratio)` to control them separately. Zero hides
the tree without disabling clustering. Colour strips remain aligned with the
cells. Use `label_fontsize` and `title_fontsize` for text size in points.
`figsize=(width, height)` sets the canvas in inches; square cells fit inside it.
Use `square_cells=False` to fill the available rectangle, or `figsize=None` and
`cell_size` to size automatically. Long titles wrap and the colourbar has its
own space above the panel.

For notebook batches, keep explicit overrides in a leading dictionary keyed by
output filename and merge them after shared defaults:

```python
defaults = dict(vmin=-0.25, vmax=0.25, dendrogram_ratio=0.025, linewidths=0.3)
plot_settings = {
    "primary_coarse_vs_Population": dict(figsize=(4.5, 5.2)),
    "activity_neftel_ora_vs_Population": dict(figsize=(3.7, 3.6), vmin=-0.4, vmax=0.4),
}
options = {**defaults, **plot_settings[plot_name]}
fig = vc.plot_correlation_heatmap(selected_comparison, **options)
```

By default, a single star denotes `q_nominal < 0.05`. These are the supplied
BH-adjusted correlation p-values, without rounding or recalculating the FDR
family after selecting a subplot. Set `alpha`, `stars=False`, or `q_column` to
use another explicitly adjusted-p column. Raw p-values are never substituted.
When adjusted tests are unavailable (including case-centred correlations), the
plot says so and omits stars. Nominal spot-level tests remain exploratory despite
BH correction: it does not account for spatial dependence or patient replication.

## Paired population maps

`plot_paired_population_maps` makes a row for each RNA group, alongside one or
more related IMC populations. The caller explicitly chooses the correspondence;
the helper does not select pairs from correlations or silently aggregate labels.

```python
keep = vc.spot_filter(imc_counts, min_cells=5)
fig, diagnostics = vc.plot_paired_population_maps(
    vc.population_fractions(imc_counts).loc[keep],
    vc.population_fractions(rna_counts),
    coordinates.loc[case_spots],
    {"OPC-like": ["OPC-like"],
     "Vascular-associated": ["Vascular-Endo", "Vascular-SMA"]},
    spot_diameter=spatial["scalefactors"]["spot_diameter_fullres"],
    image=spatial["images"]["hires"],
    image_scale=spatial["scalefactors"]["tissue_hires_scalef"],
    imc_colors=imc_palette, rna_colors=rna_palette,
    title=case, figsize=(8.1, 6.1),
)
```

Pass cohort-wide tables and one case's full-resolution coordinates. The image
scale converts both coordinates and physical spot diameter to image pixels;
all panels share the crop and orientation. The default paired footprint is the
intersection of measured IMC, RNA and the supplied case coordinates. Missing
coverage is grey, while measured zero abundance remains at the low end of the
colour map. `matched_footprint=False` additionally displays RNA outside that
intersection, without including those spots in correlations.

Increase `spot_diameter_scale` to make circles nearly touch for readability,
choosing the factor from the observed nearest-neighbour spacing (1.5 in the paired
GBM geometry, or 1.8 for slight overlap). The default `1.0` retains their physical diameter. This changes only
the displayed circles, not their centres, values, coverage or correlation
calculations. State this enlargement in the figure caption. `figsize` controls
the complete canvas, so reducing its width makes individual maps more compact;
allow enough height for every requested row. Use `label_wrap=16` to wrap long
population labels while retaining a readable `label_fontsize`; palette swatches
remain visible on each panel. Set `swatch_size` in points to adjust the square
size independently of the map dimensions; each square is vertically centred
beside its label in a shared legend box, including wrapped labels. Abundance and missing-coverage legends use separate
lines below the maps.
Map tiles are packed with small fixed gaps. Set `annotate_correlations=False`
to omit coefficient overlays when displaying the maps beside correlation
heatmaps; the returned diagnostics still contain the complete coefficients.

By default, each population/modality is scaled independently from zero to its
99th percentile over the cohort-wide shared IMC/RNA observations; the same
population therefore retains its scale across cases. Use `scale_scope="case"`
to calculate limits from the displayed case's shared spots instead. This shows
local spatial patterns but means equal colours are not comparable across cases.
The selected scope is recorded in the diagnostics. Colour intensity is not absolute cross-modality
agreement. Change `quantile` to adjust saturation. The returned diagnostics give
each pair's limits, paired spot count, and Pearson/Spearman coefficients using
unclipped values. Constants or fewer than four paired spots have undefined
coefficients. No smoothing or significance testing is applied.

For abundance maps, pass unnormalised counts/means and change `value_label`.
For fractions, normalise using all biological populations before selecting
displayed columns. Document any data-driven pair selection and retain the same
rows across cases to make heterogeneity visible. The caller owns figure/table
exports and may group them beneath reporting's `figures` category by analysis
and case.

## Expression and pathway comparisons

Use the prepared log-normalised Visium `.X`, not cell2location's count-valued
posterior `.X`. Pass `raw=False` to decoupler MLM/ULM when `.raw` stores counts.
Save the network used, version, target coverage and scores. CytoSig-weighted MLM
is a signature activity proxy, not the original CytoSig software inference.

`gene_set_regression(expression, genes, spots, network, method="mlm")` wraps
decoupler 2 MLM/ULM and returns score and adjusted-enrichment-p-value tables.
It creates a temporary AnnData from the explicit matrix, with no `.raw`, and
handles sparse inputs without the list-to-array conversion problem in decoupler
2.1.1. The caller's expression and network remain unchanged. Imports are lazy;
only this helper requires the already available AnnData and decoupler packages.

`gene_set_ora` implements the historical decoupler 1.x one-sided hypergeometric
score, `-log10(p)`, with seeded tie breaking. Defaults select the top 5% measured
genes, require five overlapping targets, and use a background of 20,000.
Duplicate source-target edges are removed; absent targets are reported in the
coverage table. The background must cover the measured gene universe. This
is intentionally distinct from decoupler 2.x log odds, and avoids the top-gene
ranking bug in decoupler 2.1.1. Tied genes can yield small differences from older
implementations with a different random-number generator.

References: [legacy ORA definition](https://decoupler.readthedocs.io/en/v1.9.2/generated/decoupler.run_ora.html),
[upstream ORA correction](https://decoupler.readthedocs.io/en/latest/changelog.html).

## Execution and output ownership

These are in-memory library helpers, not a new planner/SLURM stage. They do not
modify AnnData, load large assets implicitly, write files, or create project
state. The caller controls memory and output lifetime; cohort-wide loading is
not a login-node task. In notebooks, retain setup, dataset-specific joins,
scientific choices and narrative in visible cells. Keep reusable computation in
this module. Return figures to the caller and route saved tables/figures through
`reporting.optional_category_output_path`, using the notebook's output folder as
the direct local fallback. Managed callers can register the same artifacts with
their existing `StageReporter`.

No persistent config, stage registry, dependency graph, CLI, SLURM wrapper,
environment mapping or canonical assets change for these notebook APIs. This
preserves the upstream cell2location contract and existing Visium-mask helper.
