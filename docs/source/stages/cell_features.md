# Cohort-first cell features

## What this stage does

`cellfeat` reads the active `napari_sbt` experiment and its frozen cohort,
calculates synthetic IMC features only for eligible objects, joins optional
precomputed sources, and writes reusable cohort-only feature assets.

One spawn-safe worker processes each ROI. It loads the original full
segmentation once, constructs cohort measurement regions, streams selected
channel images, and writes one atomic Parquet fragment. Valid completed
fragments are reused after interruption.

## Scientific context

The output contains only eligible cells, but excluded neighbours remain part of
the scientific context. Full-mask expansion prevents overlap, local background
rings exclude every segmented cell, and neighbourhood features include the
full tissue segmentation. Shape features describe original masks. Signed
offsets change intensity measurement regions only.

Positive offsets use full-mask label expansion. Negative offsets erode eligible
objects; objects that disappear retain their feature row with missing intensity
values and a recorded warning. Within-ROI ranks compare eligible cells only.

## Inputs

- `napari_sbt.active_experiment`, resolved below
  `napari_sbt.experiment_folder` unless absolute;
- the experiment manifest and frozen identity snapshot;
- original labelled masks;
- selected ROI/channel images;
- optional imported CSV, Parquet, AnnData, or CellVision feature sources.

Images and masks must have exactly matching shapes. Inputs are never resized or
modified.

## Outputs

The experiment receives:

- resumable per-ROI Parquet fragments and fingerprint sidecars;
- a canonical namespaced cohort-only Parquet table;
- a feature dictionary;
- source coverage and failed-ROI tables;
- a provenance manifest with cohort/recipe fingerprints, timing, feature
  counts, erosion losses, and warnings.

The managed report records selected/total cells, represented and resumed ROIs,
feature count, erosion losses, failures, elapsed time, and reusable assets.

## Resources and execution

The wrapper requests 8 CPUs, 64 GB RAM, and 24 hours on the CPU high-memory
partition. It uses the segmentation environment with Parquet support and limits
OpenMP/MKL to one thread per worker.

```yaml
napari_sbt:
  experiment_folder: napari_sbt
  active_experiment: t_cell_subclasses
  worker_count: 8
  annotated_adata_path: napari_sbt_annotated.h5ad
```

Run:

```bash
sbt run cellfeat
```

There is no fixed upstream dependency and `cellfeat` is not a workflow-mode
member. Existing masks and features may originate from several valid pipeline
branches, so the experiment manifest is the explicit input contract.

## Limitations

CellPose probability, flow, flow-error, and segmentation-configuration features
can be imported when previously calculated, but ordinary IMC channels do not
contain the CellPose maps needed to derive them. Large full-mask context and
region-image recipes remain CPU- and I/O-intensive; use the managed stage when
a local QProcess build is impractical.
