# Subclustering stage (`job_subclustering.sh`)

This document explains why subclustering is used in this pipeline, and how to run it safely using the checkpointed workflow implemented in `SpatialBiologyToolkit.scripts.subclustering`.

## Why subclustering is useful

Broad populations (for example, tumor-like, vascular, myeloid) often contain biologically distinct sub-states.  
Subclustering allows you to:

- refine broad annotations into more specific populations,
- QC subcluster separation with UMAP and matrixplot outputs,
- manually curate the final labels before re-integrating them into AnnData.

The stage is deliberately split into checkpoints so users can inspect and edit intermediate files before irreversible label integration.

## Checkpointed workflow

### Checkpoint 1: template generation

On first run, the script creates a `subclustering/` folder and writes:

- `sublustering_settings.csv`
- `marker_list.csv`

`sublustering_settings.csv` columns:

- `base_label`
- `population`
- `resolution`
- `marker_list`

Rows are auto-generated from `subclustering.base_label_key` in `config.yaml`, with default resolution and marker list values from config.

`marker_list.csv`:

- index: marker names
- required marker selector columns must start with `markers_`
- includes `markers_all` by default (`True` for all markers)

You can add custom selectors, for example `markers_myeloid`, by adding a new boolean column.

Example:

```text
marker,markers_all,markers_myeloid
CD14,True,True
CD68,True,True
PanCK,True,False
PAX8,True,False
```

### Checkpoint 2: row-wise subclustering + QC outputs

If both template files already exist, each row in `sublustering_settings.csv` is executed as one subclustering task:

- subset cells by `base_label` + `population`,
- run Leiden at specified `resolution`,
- select markers from `marker_list.csv` via `marker_list` selector,
- generate QC figures.

Outputs are saved under `subclustering/figures/`:

- `combined_umap/`
- `individual_umap/` (if enabled)
- `matrixplot/`

The stage then writes/updates:

- `subclustering/subcluster_to_final_population.csv`

If this file already exists, previously edited `final_population` values are preserved where keys still match.

### Checkpoint 3: remap integration

If `subcluster_to_final_population.csv` has edited rows (`final_population != subcluster`), remapping is applied and:

- `subclustering.final_label_key` is added to `adata.obs`,
- a simple mapping CSV is exported:
  - `subclustering/master_index_to_final_population.csv`
- output AnnData is written to `subclustering.output_adata_path`.

If the remap file is unedited and `apply_remap_only_if_modified: true`, integration is skipped intentionally.

## Config block

Configure this stage in `config.yaml`:

```yaml
subclustering:
  input_adata_path: null
  output_adata_path: null
  output_subdir: subclustering
  mode: generate
  base_label_key: population
  default_resolution: 0.3
  default_marker_list: all
  use_rep: X_biobatchnet
  final_label_key: population_final
  master_index_obs: Master_Index
  apply_remap_only_if_modified: true
  save_individual_umaps: true
  figure_extension: .png
  figure_dpi: 300
```

## SLURM usage

From your dataset working directory (containing `config.yaml`):

```bash
pl subcl
```

The default `mode: generate` runs checkpoints 1 and 2. A first run stops after
creating the templates; edit them, then run `pl subcl` again to generate the
subclusters and remap table. After editing `final_population` in that table,
set `mode: apply` and run the stage once more for checkpoint 3. `mode: all`
runs checkpoints 1-3 when every required file already exists; numeric values
`1`, `2`, or `3` select one checkpoint explicitly.

or submit directly:

```bash
sbatch ~/imcanalysis/SLURM_scripts/job_subclustering.sh
```

## Recommended position in pipeline

Typical order:

1. run `bint`, `rapids`, or `bbn` first to generate processed embeddings/clusters,
2. run `subcl` to refine populations,
3. run downstream analyses (`pairsp`, `vis`, optional `cchar`) using curated labels.

## Notes

- Filename is `sublustering_settings.csv` (kept for compatibility with existing usage).
- If `Master_Index` is missing, the script falls back to `obs_names` for export mapping.
- The script logs each checkpoint clearly so users can see whether it created templates, ran subclustering, or applied remapping.
