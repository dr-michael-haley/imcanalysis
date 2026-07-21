# NetworkX Spatial Analysis

## What this stage does

This stage represents each ROI as a spatial graph and asks two questions about
its population labels:

1. **Whole-graph population assortativity:** do neighbouring cells tend to have
   the same population label, or do different labels tend to mix?
2. **Per-population average clustering:** do cells of one population form
   locally closed, triangle-rich arrangements among their same-population
   neighbours?

Squidpy constructs the graph from cell-centroid coordinates. The graph is then
converted to an undirected NetworkX graph, and NetworkX calculates the two
metrics. Optional label permutations compare the observed labels with a null in
which the graph remains fixed but population identities are rearranged within
each ROI.

The stage produces one assortativity result for each ROI and one average-
clustering result for every sufficiently abundant population in that ROI. When
several ROIs belong to one biological case, it also produces case-level means.

```{important}
A graph edge is an analytical definition of neighbourhood. It is not evidence
that two cells touch, signal to one another, or interact mechanistically. The
biological meaning of every result depends first on how the graph was built.
```

## From tissue image to graph

A graph contains **nodes** and **edges**. Here:

- each segmented cell is one node;
- its population annotation is a categorical node attribute; and
- an edge joins cells that the configured Squidpy graph regards as spatial
  neighbours.

Graphs are constructed independently for each value of `roi_obs`. There are no
edges between different images or ROIs. The code takes coordinates from
`adata.obs[x_coord_obs]` and `adata.obs[y_coord_obs]`, places them in a temporary
AnnData spatial matrix, and calls Squidpy's spatial-neighbour function.

After Squidpy returns a connectivity matrix, the stage:

1. removes every diagonal entry, so cells cannot be their own neighbours;
2. symmetrises connectivity by retaining an edge when it appears in either
   direction; and
3. converts every remaining non-zero connection to a NetworkX undirected edge.

The NetworkX calls in this stage are unweighted. A transformed Squidpy edge can
carry a numerical weight, but assortativity and average clustering are called
without a weight argument, so all retained edges count equally.

## Choosing the spatial graph

Graph construction is not a neutral preprocessing detail. It defines the
spatial scale and topology on which both metrics operate.

### Default: six-nearest-neighbour graph

With `graph_coord_type: generic`, `graph_delaunay: false`, and
`graph_radius: null`, Squidpy uses a k-nearest-neighbour graph. The default
`graph_n_neighs: 6` asks for six nearest candidates per cell. The stage then
symmetrises the graph, so a cell can finish with more than six neighbours when
other cells selected it.

This gives most cells a similar minimum connectivity even when physical cell
density varies. It is useful for comparing local topology across a point cloud,
but it has no single physical distance scale. In a sparse or empty region, a
cell's sixth-nearest neighbour may be biologically too distant to represent a
meaningful neighbourhood.

### Radius graph

Setting `graph_radius` with generic coordinates selects a distance-based graph.
One value is a maximum radius; two values specify a minimum and maximum edge
length. This can be easier to interpret biologically because every edge shares
an explicit coordinate scale.

The code does not convert coordinates. A radius of 20 represents 20
micrometres only if the configured x and y columns are already in micrometres.
A small radius can create isolated cells, whereas a large radius creates dense
graphs in which triangles become common by construction.

When radius mode is active, Squidpy ignores `graph_n_neighs`.

### Delaunay graph

`graph_delaunay: true` connects cells using a Delaunay triangulation. This
adapts to local geometry rather than fixing a neighbour count or physical
radius. It can describe a tissue tessellation well, but it may bridge large
gaps, lumina, folds, or off-tissue regions unless edges are pruned. In this mode
`graph_n_neighs` is ignored; a two-value radius can prune Delaunay edges after
triangulation.

### Other graph controls

`graph_percentile` can remove long generic-graph edges according to the
distribution of edge distances. This is a relative, dataset-specific cutoff,
not a fixed biological distance.

`graph_transform` forwards Squidpy's `spectral` or `cosine` connectivity
transform. The final NetworkX metrics ignore edge weights, although a transform
can still matter if it changes which entries remain non-zero.

`graph_set_diag` asks Squidpy to add self-connectivity, but the stage explicitly
removes the diagonal before conversion. It therefore does not create self-loops
in the analysed NetworkX graph.

The active implementation calls Squidpy's general
[`spatial_neighbors`](https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors.html)
interface. Current Squidpy documentation marks this interface as deprecated in
favour of mode-specific graph builders. This is a software-maintenance issue;
the resolved graph parameters recorded by this stage remain the source of truth
for interpreting an existing run.

## Metric 1: population assortativity

The stage assigns each node its population label and calls NetworkX
[`attribute_assortativity_coefficient`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.attribute_assortativity_coefficient.html).
NetworkX defines attribute assortativity from the mixing matrix of labels at
the two ends of graph edges. It summarizes whether connected nodes have similar
categorical attributes.

The result is one global value for the complete ROI graph, recorded with
`metric = assortativity` and `population = __all__`.

In general:

- a positive coefficient indicates preferential connections between cells with
  the same population label;
- a value around zero indicates little overall label preference relative to the
  edge-end label frequencies; and
- a negative coefficient indicates preferential connections between different
  population labels.

Values commonly lie between -1 and 1, although the attainable extremes depend
on the number and abundance of labels and the graph's connectivity. A value of
one requires exclusively like-with-like edges in the relevant mixing pattern.

### Sensible biological interpretation

Positive assortativity is consistent with spatial segregation into homotypic
domains: tumour nests, lymphoid aggregates, macrophage-rich regions, stromal
compartments, or clonally expanding patches can all increase it. Negative
assortativity is consistent with intermixing or interface-rich tissue, such as
immune cells dispersed among tumour or epithelial cells.

Assortativity is a whole-ROI summary. It cannot identify which population pair
caused the result. One abundant, strongly segregated population can dominate
the coefficient while rare populations mix freely. Use the Pairwise Spatial
stage, images, or an explicit mixing matrix when pair-specific interpretation
is required.

A change in assortativity between disease groups can reflect altered tissue
architecture, but it can also reflect different population abundances,
segmentation density, ROI selection, graph parameters, or annotation granularity.
It is not direct evidence for cell-cell attraction or exclusion.

## Metric 2: per-population average clustering

For each population, the stage selects that population's nodes and creates an
**induced subgraph**. This subgraph contains only those cells and only graph
edges whose two endpoints both have that population label. Cross-population
edges are removed before clustering is calculated.

The stage then calls NetworkX
[`average_clustering`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html).
For one node, the local clustering coefficient is the fraction of possible
connections among its neighbours that are actually present. In an undirected
graph this measures triangle closure: if two same-population neighbours of a
cell are also connected to one another, the three cells form a triangle.

NetworkX averages the local coefficients over every node in the induced
subgraph. The active call uses the defaults `weight=None` and
`count_zeros=True`. Consequently:

- every retained edge is treated equally;
- cells with fewer than two same-population neighbours contribute zero; and
- isolated or chain-like cells lower the population average.

The metric ranges from zero to one. A value near one means that
same-population neighbours form nearly all possible local triangles. A value
near zero means that those cells are isolated, arranged in chains or sparse
branches, or have neighbours that are not mutually connected.

`minimum_cells_per_population` controls whether this metric is attempted. It
does not guarantee adequate topology: five cells may pass the default threshold
yet contain no same-population edges or only one unstable triangle. The output
therefore also records the observed number of contributing cells and
same-population edges.

### Sensible biological interpretation

High average clustering can be consistent with compact homotypic patches, such
as a cohesive tumour nest or a tightly organized immune aggregate. Low average
clustering can be consistent with dispersed infiltration, a thin boundary,
linear arrangement along a vessel, or isolated cells.

It is not a direct measure of physical compactness, population abundance,
aggregate size, or total homotypic edge count. Two separated small cliques can
have high clustering despite not forming one large tissue compartment. A dense
radius graph also creates more possible triangles than a sparse graph, so values
from different graph definitions should not be compared as though they were on
the same biological scale.

Because the induced subgraph discards heterotypic edges, this metric answers
"how closed are the local connections among cells of this type?" It does not
describe the complete neighbourhood surrounding that population.

## The shuffled-label reference

When `run_bootstrap` is enabled, the stage repeatedly permutes population
labels among the existing nodes within each ROI. Coordinates, graph edges,
population counts, and the total number of cells remain fixed. Assortativity
and every population-induced clustering coefficient are recalculated after
each permutation.

Although the configuration uses the word *bootstrap*, this is a label-
permutation null. It does not resample cells, ROIs, cases, or patients and does
not generate new spatial point patterns.

The null question is:

> Given these exact cell positions, this exact graph, and these population
> counts, what metric values would be expected if the permitted labels were
> exchangeable across the ROI?

This is different from complete spatial randomness. It retains tissue density,
holes, boundaries, and every edge in the observed graph. It removes the
association between mutable population identity and spatial location.

### Static populations

Labels listed in `bootstrap_static_populations` remain attached to their
original nodes. All other labels are permuted only among the non-static nodes.
This creates a conditional null that can ask how mutable populations arrange
themselves around a fixed anatomical or reference population.

The choice requires care. Because a static population's nodes never change,
its induced subgraph and observed average-clustering value are identical in
every permutation. Its null standard deviation is therefore zero and its own
clustering z-score is undefined (`NaN`). The output marks such rows with
`population_static = true`. Whole-graph assortativity can still vary because
the remaining labels move.

Static labels also restrict where other labels may be placed, so null values
from a conditional permutation are not directly comparable with values from a
run that shuffled every population.

## Reading the summary statistics

Each ROI and case summary contains:

| Column | Meaning |
|---|---|
| `observed` | Metric calculated from the real population labels |
| `bootstrap_mean` | Mean metric across label permutations |
| `bootstrap_std` | Standard deviation across label permutations |
| `delta` | `observed - bootstrap_mean` |
| `zscore` | `delta / bootstrap_std` |

For both metrics, positive delta and z-score mean that the observed value is
higher than its shuffled-label expectation:

- positive assortativity z-score: more overall homotypic mixing than expected;
- negative assortativity z-score: more heterotypic mixing than expected;
- positive clustering z-score: a population forms more locally closed
  same-type triangles than expected; and
- negative clustering z-score: a population forms fewer such triangles than
  expected.

`observed` and `delta` are effect-size summaries. A z-score additionally scales
the effect by Monte Carlo variability. A large absolute z-score can arise from
a modest effect with a very narrow conditional null; it does not guarantee a
large biological difference.

The stage does not calculate empirical P values, confidence intervals, or
multiple-testing corrections. Its z-score must not automatically be converted
to a normal-distribution P value because graph statistics under constrained
label permutations need not follow a Gaussian distribution. Saving permutation
samples permits more detailed inspection, but does not create biological
replication.

When bootstrapping is disabled, `bootstrap_mean`, `bootstrap_std`, `delta`, and
`zscore` are missing. When the permutation standard deviation is zero, z-score
is also missing.

## Case-level aggregation

If `case_obs` exists, the stage averages observed ROI metrics within each case.
Every ROI with a finite metric receives equal weight; the mean is not weighted
by ROI area, cell count, or edge count.

For the case null, permutation number 1 is averaged across that case's ROIs,
then permutation number 2, and so forth. These ROI permutations use independent
random streams derived from the configured seed. The resulting case-level
delta and z-score compare the case's mean observed topology with the mean
fixed-graph label null across its ROIs.

This procedure does not resample ROIs or cases and does not estimate uncertainty
across patients. A case containing several ROIs remains one biological case,
and a case containing one ROI has no additional replication. Use case-level
tables as inputs to a separate design-aware comparison across independent
subjects.

`n_case_rois` records the number of ROIs assigned to the case, while
`n_case_rois_with_metric` records how many supplied a finite value for that
metric. Always inspect these columns when populations are rare or absent.

## Interpreting the metrics together

Assortativity and clustering describe related but distinct structure:

| Pattern | Possible interpretation |
|---|---|
| High assortativity and high clustering for several abundant populations | Broad compartmentalization with locally compact homotypic patches |
| High assortativity but low clustering for one population | Overall label segregation driven elsewhere, or that population forms sparse/linear homotypic regions rather than triangles |
| Low or negative assortativity with high clustering for one population | A locally compact population embedded in an otherwise mixed ROI |
| Low assortativity and low per-population clustering | Intermixed or dispersed tissue, or a graph scale too small/sparse to capture domains |

These are hypotheses to check against the image. Triangle-rich topology can be
created by a large radius; apparent mixing can be created by a k-nearest graph
that bridges tissue gaps; and segmentation fragments can create dense local
clusters of identically labelled nodes.

## Grouped plots are descriptive

`groupby_obs` adds experimental or biological metadata to ROI and case rows and
uses it to stratify plots. It does not alter graph calculations or perform a
statistical comparison between groups.

The default `plot_summary_level: case_if_available` plots cases when case
summaries exist and otherwise plots ROIs. This is usually preferable to treating
multiple ROIs from the same case as independent points. Barplots and boxplots
summarize the displayed observations; their appearance is not evidence of a
group difference.

Formal comparisons should operate at the biological experimental unit and
account for repeated ROIs, pairing, covariates, unequal numbers of ROIs, and
multiple metrics or populations. Show individual case values wherever possible.

## Important configuration choices

The settings with greatest scientific impact are:

- `population_obs`: the biological annotation being evaluated;
- `roi_obs`: the boundary within which edges may exist;
- `x_coord_obs` and `y_coord_obs`: the centroid coordinates;
- `graph_n_neighs`, `graph_radius`, or `graph_delaunay`: the neighbourhood
  definition;
- `graph_percentile`: optional long-edge pruning;
- `minimum_cells_per_population`: minimum support for induced-subgraph
  clustering;
- `bootstrap_static_populations`: whether the null is fully shuffled or
  conditional on fixed labels;
- `case_obs`: the biological unit used for ROI aggregation; and
- `plot_summary_level`: whether figures show cases or ROIs.

`bootstrap_n_permutations` controls Monte Carlo resolution. More permutations
stabilize the null estimate but do not add tissue samples or correct a weak
experimental design. `bootstrap_seed` makes recomputation reproducible.

`reload_saved_results` allows plot-only reruns. The loader checks expected
columns but does not confirm that saved tables match current coordinates,
population labels, graph settings, permutations, or metadata. Disable reload or
remove incompatible cached summaries whenever analytical inputs change.

## Outputs and provenance

The stage writes:

- `raw_data/networkx_roi_summary.csv`: one row per ROI and metric;
- `raw_data/networkx_case_summary.csv`: equal-weight case means when case
  metadata are available;
- `raw_data/networkx_summary_combined.csv`: the ROI and case rows together;
- `raw_data/networkx_roi_bootstrap.csv`: optional individual ROI permutation
  values;
- `raw_data/networkx_case_bootstrap.csv`: optional averaged case permutation
  values;
- `metadata/anndata_obs_snapshot.csv.gz`: a complete observation snapshot;
- `metadata/roi_metadata.csv` and `metadata/case_metadata.csv`: metadata joined
  to the summaries;
- `plots/<summary level>/`: assortativity and per-population clustering plots;
  and
- `networkx_spatial_run_metadata.json`: resolved graph, permutation, threading,
  and plotting settings.

ROI summary rows also contain `metric_n_cells_observed` and
`metric_n_edges_observed`. For assortativity these are total graph nodes and
edges; for average clustering they are the population's nodes and homotypic
induced-subgraph edges. Use them to identify unsupported estimates.

The graphs themselves and per-cell clustering coefficients are not saved. The
AnnData is saved back to its pipeline path with run metadata in
`adata.uns["networkx_spatial_pipeline"]` and a stage-completion record; the
summary metrics remain report tables rather than cell-level annotations.

The complete observation snapshot can include sample or clinical metadata.
Review data-governance requirements before sharing the output directory.

## Common problems and limitations

- **The graph defines the answer.** Metrics from different neighbour counts,
  radii, pruning thresholds, or graph modes are not directly comparable.
- **K-nearest graphs have no fixed physical scale.** They can connect cells
  across gaps in sparse regions.
- **Radius graphs depend on coordinate calibration and density.** The code does
  not convert pixels to micrometres.
- **Centroids do not establish contact.** Cell size, shape, mask boundaries, and
  intervening cells are ignored.
- **Assortativity is global.** It does not identify the responsible population
  pairs or anatomical compartments.
- **Average clustering is triangle-based.** It is not abundance, aggregate size,
  density, or a direct measure of compactness.
- **Rare populations are unstable.** The minimum cell threshold alone does not
  guarantee enough homotypic edges or triangles.
- **The permutation null assumes exchangeability across an ROI.** Anatomical
  compartments and gradients can violate that assumption.
- **Annotation and segmentation errors propagate.** Merged cells, fragments,
  doublets, and broad or inconsistent labels change graph topology and mixing.
- **ROI selection matters.** Cropping a boundary, vessel, tumour core, or immune
  aggregate can change the metric independently of the underlying disease.
- **No group-level hypothesis test or multiple-testing correction is performed.**
  Grouped figures remain exploratory until analysed at the correct biological
  unit.

Before reporting a topological difference, inspect representative graphs or
source images, verify coordinate and mask quality, show cell and edge support,
test sensitivity to a biologically reasonable graph definition, show
independent case values, and confirm that the finding is not driven by one ROI
or one abundant population.
