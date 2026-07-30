# `napari_sbt`: cohort-first IMC exploration and classification

`napari_sbt` combines the reusable parts of the IMC explorer and CellPose
active-learning viewer in one experiment-driven Napari dock. It supports both
whole-segmentation QC and the more common task of subclassifying cells from one
or more values in a categorical `AnnData.obs` column.

The original images, masks, and AnnData are read-only. An experiment freezes
the eligible `(ROI, ObjectNumber)` identities before feature calculation,
annotation, training, scoring, or export.

## Launch

From an initialized SBT project:

```bash
sbt gui napari --project /path/to/project
```

Direct launch accepts explicit paths:

```bash
napari-sbt \
  --anndata project/anndata.h5ad \
  --masks project/masks \
  --images project/denoised_images
```

The Python factory returns the viewer, controller, and dock without entering
the event loop:

```python
from SpatialBiologyToolkit.napari_sbt import launch

viewer, controller, dock = launch(
    anndata_path="anndata.h5ad",
    masks_folder="masks",
    images_folders=["denoised_images"],
)
```

## Setup and the frozen cohort

Cell scope is required. Choose either:

- **All cells**, which is the replacement for the whole-mask CellPose QC use
  case.
- **Selected adata.obs values**, selecting one categorical observation and one
  or more values. This is intended for subclassifying a broad Leiden or curated
  population.

The preview reports selected/total cells, represented ROIs, per-ROI counts,
missing masks, missing object IDs, and other labels present in the full masks.
It also renders a cohort-only mask for inspection. Confirming the preview
stores a frozen identity snapshot in the experiment. A later change in AnnData
membership does not silently change the experiment; create a revision.

The primary classification layer contains only eligible objects and preserves
their original mask labels. The optional context layer contains the rest of
the segmentation at low opacity. Annotation clicks on that context are ignored.
Normal ROI navigation includes only ROIs with eligible cells.

Multiple image folders are merged for each ROI. Supported layouts are
`<folder>/<ROI>/<channel>.tiff`, `<folder>/<ROI>.tiff`, and flat
`<folder>/<ROI>_<channel>.tiff`. Channel filenames are matched against
`adata.var_names` and, when available, `adata.var["channel_name"]` and
`adata.var["channel_label"]`. Duplicate channels from different folders remain
available with their source folder appended; unmatched composites remain
available as additional images.

Experiments contain two to eight stable, mutually exclusive classes. Each has
a display name, colour, numeric shortcut, and `keep` or `exclude` mask
disposition. The Segmentation QC template creates `good` and `artifact`
classes. Stable class and cohort semantics are locked after confirmed labels
exist.

## Feature sources and calculation

Feature work has its own **🧬 Feature Building** tab; it is not required for
ordinary image exploration. The tab separates source checking, channel
selection, recipe design, and execution:

1. Add optional imported sources. Tables use
   `source_name=/path/to/features.parquet`. AnnData and CellVision sources use
   `source_name=/path/to/data.h5ad::X`, `::obs`, or an `obsm` key such as
   `::X_cellvision`.
2. Select **Validate identities, cohort coverage, and numeric features** before
   a build. Validation runs outside the Napari process and reports whether each
   source can be read, how many frozen-cohort cells match, how many are missing,
   and how many usable numeric features it supplies. Its complete
   machine-readable report is saved as `features/source_validation.json`.
3. Refresh the channel list and select channels directly from those inferred
   from `adata.var_names` and the images available for an experiment ROI.
   Clearing the selection means “use every channel discovered by the worker.”
4. Enable feature families and tick only the individual measurements required
   for this experiment. The adjacent descriptions explain the scientific
   quantity represented by every choice.

The selectable feature families are:

- **Distribution**: fast per-channel pixel summaries such as mean, median,
  quantiles, spread, range, sum, and coefficient of variation.
- **Core/border/background/contrast**: more expensive regional measurements,
  including intensity-centroid displacement and local background comparisons.
- **Channel gradients**: the selected distribution summaries calculated over
  gradient magnitude, capturing edges and within-cell texture.
- **Original-mask morphology**: channel-independent area, perimeter,
  circularity, axes, solidity, holes, bounding-box, and edge features.
- **Full-segmentation context**: neighbour counts, nearest-cell distance, and
  ROI density calculated with every segmented cell as context.
- **Cohort-relative ROI ranks**: optional z-scores and/or percentile ranks
  among eligible cells in the same ROI.

Feature sources are joined strictly by `(ROI, ObjectNumber)`, filtered to the
frozen cohort, and namespaced before modelling. Sources may be CSV/Parquet,
AnnData `X`, numeric `obs`, a selected `obsm`, CellVision embeddings, or newly
calculated IMC features.

Synthetic extraction calculates rows only for eligible cells, but it keeps the
original full segmentation in memory:

- positive offsets block at every segmented-cell boundary by default;
- **Allow positive offsets to overlap other cells** independently expands each
  eligible measurement region through neighbouring masks. In this mode a pixel
  may contribute to multiple cells, which can be useful in dense tissue;
- negative offsets erode selected objects without changing their identities;
- background rings exclude every segmented object;
- neighbourhood features describe the full tissue segmentation;
- ranks and normalizations compare eligible cells within an ROI;
- shape features always use the original segmentation.

Stored image values are used by default. A fixed Nimbus normalization mapping
may be applied for scientific features. The quantile controls used for display
never change feature values. Image/mask shape mismatches fail rather than
resizing scientific inputs.

Local builds run in a `QProcess`; the worker uses a spawn-safe process pool with
one task per ROI. Completed Parquet fragments are reusable only when the
experiment, cohort, recipe, and input fingerprints match. Cancellation
preserves valid fragments. During a local build, the panel shows the Python
process ID and whether it is alive, heartbeat age, elapsed time, active worker
count, completed/failed/pending ROI counts, the most recent ROI result, and the
final fragment/source-combination phase. A heartbeat is emitted approximately
every two seconds while ROI work is outstanding, so a long-running ROI can be
distinguished from a dead process.

For large datasets, configure the active experiment and run:

```bash
sbt run cellfeat
```

## Multiclass review

Labels are either `proposed` or `confirmed`. Proposed labels are visible and
audited but do not train the model or become final assignments. Training
requires at least two confirmed examples per class and warns when class/ROI
coverage is weak.

HistGradientBoosting is the default model. Random Forest, XGBoost, and LightGBM
are explicit alternatives. Scores contain the predicted class, every class
probability, maximum probability, probability margin, and normalized entropy.
Cells without usable features stay visible but are marked unscorable.

The uncertainty queue ranks ambiguous unlabelled cohort cells. High-confidence
cells can be bulk-added as proposals for one predicted class. Confirmed labels
always override predictions.

## Explore, regions, layers, and export

The **Explore** tab provides cohort-aware ROI navigation, raw and extra image
loading, RGB views, categorical/numeric AnnData overlays, and abundance-ranked
population views. Previous/Next buttons follow the current ROI ordering.
**Hide all**, **Show all**, and **Delete all** act on every Napari layer; loading
the ROI again reconstructs the cohort and classification layers after deletion.
New layers have opacity 1.0 by default, except for the deliberately dimmed
full-segmentation context layer.

Selected two-dimensional image channels can be loaded as independent greyscale
layers or assigned, in displayed selection order, to repeating red, green,
blue, cyan, yellow, and magenta colormaps. The three-channel RGB composite
remains available. Image choices are stored by logical channel name rather
than path, so the same recipe can be replayed against another ROI and missing
channels are reported rather than substituted.

Categorical overlays and separate population layers first look for Scanpy's
`adata.uns["<observation>_colors"]` palette (with common mapping-style
alternatives accepted) and otherwise use a stable fallback palette. Specific
populations can be selected as individual contour layers. Variables in
`adata.var_names` can also be loaded from cell-level `adata.X` values and
mapped onto eligible mask objects as quantitative overlays.

With **Reload the current Explore view when ROI changes** enabled, navigation
reconstructs the same images, colormaps, observation overlay, population
layers, and marker overlays for the new ROI. A view fingerprint tracks which
ROIs have been rendered with exactly those settings: ROI selector entries are
green when viewed and amber when not yet viewed. Changing any component creates
a different review set.

The **Layers re-added when the ROI changes** list is the explicit reload
contract. It shows every replayable Explore layer, including its colormap,
visibility, and opacity. Select one or more entries and use **Delete/reset
selected recipe items** to remove Explore entries from both the recipe and the
current replayed view; selecting a managed classifier entry resets its display
settings to the defaults. **Update from current layers** rebuilds the recipe
from supported layers currently present in Napari, capturing manual colormap,
visibility, and opacity changes. The eligible cohort, excluded-cell context,
confirmed, proposed, predicted, and uncertainty/probability layers are also
shown explicitly. Their scientific data continue to be regenerated from masks,
labels, and scores, but their visible/hidden state and opacity are captured and
replayed. Changing the eye state in Napari updates the active recipe
immediately. Manual-region and unsupported derived layers remain separately
managed or are reported as ignored rather than being silently added to the
recipe.

**Save current view for selected population** stores a population-specific
verification recipe inside the experiment. Selecting that observation and
population later automatically retrieves the recipe; the explicit load button
does the same. Only these review recipes and their viewed-ROI sets are stored.
This is not a general Napari workspace save/load mechanism.

**Use this population as classification cohort** transfers the population
selector back to Setup and requires a new preview and confirmation.

The **Regions & Export** tab stores manual polygon regions and synchronizes
their contained cell identities. It exports:

- a cohort-only assignment table;
- an atomic annotated AnnData copy;
- optional cohort-only masks preserving original IDs;
- optional cleaned masks for classes marked `exclude`.

The annotated copy includes subclass, assignment source, confidence, and
uncertainty observations; a full probability matrix with `NaN` outside the
cohort; and an optional combined broad-population/subclass column. Original
AnnData and masks are never overwritten by default.

## Experiment layout

```text
napari_sbt/<experiment>/
├── experiment.yaml
├── cohort/eligible_cells.parquet
├── features/
│   ├── fragments/
│   ├── feature_table.parquet
│   ├── feature_dictionary.csv
│   ├── coverage_report.csv
│   ├── source_validation.json
│   └── feature_manifest.json
├── labels/labels.parquet
├── explore/review_state.json
├── models/
├── scores/scores.parquet
├── annotations/
├── cohort_masks/
└── exports/
```

There is deliberately no viewer-workspace save/load mechanism. Scientific
state is captured in explicit experiment, label, feature, model, and export
assets instead.
