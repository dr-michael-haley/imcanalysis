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

Experiments contain two to eight stable, mutually exclusive classes. Each has
a display name, colour, numeric shortcut, and `keep` or `exclude` mask
disposition. The Segmentation QC template creates `good` and `artifact`
classes. Stable class and cohort semantics are locked after confirmed labels
exist.

## Feature sources and calculation

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
preserves valid fragments. For large datasets, configure the active experiment
and run:

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
population views. **Use this population as classification cohort** transfers
the population selector back to Setup and requires a new preview and
confirmation.

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
│   └── feature_manifest.json
├── labels/labels.parquet
├── models/
├── scores/scores.parquet
├── annotations/
├── cohort_masks/
└── exports/
```

There is deliberately no viewer-workspace save/load mechanism. Scientific
state is captured in explicit experiment, label, feature, model, and export
assets instead.
