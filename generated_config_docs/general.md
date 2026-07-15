# General

## Input folders

### `imc_files_folder`

- Type: `str`
- Default: `IMC_files`
- Level: `basic`

Folder containing raw IMC input files in MCD or TXT format.

Advice:
Use this as the primary folder for raw IMC files.

### `metadata_folder`

- Type: `str`
- Default: `metadata`
- Level: `basic`

Folder containing pipeline metadata and panel tables.

Advice:
Keep metadata.csv and panel.csv in this folder unless a stage overrides it.

## General

### `mcd_files_folder`

- Type: `str`
- Default: `MCD_files`
- Level: `advanced`

Configuration value for mcd files folder.

Advice:
No additional advice.

### `qc_folder`

- Type: `str`
- Default: `QC`
- Level: `advanced`

Configuration value for qc folder.

Advice:
No additional advice.

### `masks_folder`

- Type: `str`
- Default: `masks`
- Level: `advanced`

Configuration value for masks folder.

Advice:
No additional advice.

### `celltable_folder`

- Type: `str`
- Default: `cell_tables`
- Level: `advanced`

Configuration value for celltable folder.

Advice:
No additional advice.

### `tiff_stacks_folder`

- Type: `str`
- Default: `tiff_stacks`
- Level: `advanced`

Configuration value for tiff stacks folder.

Advice:
No additional advice.

### `raw_images_folder`

- Type: `str`
- Default: `tiffs`
- Level: `advanced`

Configuration value for raw images folder.

Advice:
No additional advice.

### `denoised_images_folder`

- Type: `str`
- Default: `processed`
- Level: `advanced`

Configuration value for denoised images folder.

Advice:
No additional advice.

### `slurm_logs_folder`

- Type: `str`
- Default: `SLURM_logs`
- Level: `advanced`

Configuration value for slurm logs folder.

Advice:
No additional advice.

## Observation columns

### `case_obs`

- Type: `Optional`
- Default: `null`
- Level: `basic`

Optional case or sample identifier column in adata.obs.

Advice:
Set this for case-level summaries and statistical comparisons.

### `roi_obs`

- Type: `str`
- Default: `ROI`
- Level: `basic`

ROI identifier column in adata.obs.

Advice:
Values should identify the imaging region associated with each cell.

### `metadata_obs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Optional metadata columns used in QC and grouped summaries.

Advice:
List stable adata.obs columns that should appear in downstream summaries.

### `master_index_obs`

- Type: `str`
- Default: `Master_Index`
- Level: `advanced`

Stable per-cell identifier column in adata.obs.

Advice:
Keep this stable across stages so cells can be matched after filtering or remapping.

## Analysis groups

### `groupby_obs`

- Type: `Optional`
- Default: `null`
- Level: `basic`

Primary adata.obs column used for cross-condition analyses.

Advice:
Choose the main experimental grouping variable, such as treatment or outcome.

### `groupby_obs_groups`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Optional ordered subset of values from groupby_obs to analyse.

Advice:
Leave unset to use all observed groups, or list groups in the desired display order.

### `groupby_obs_primary_pairwise`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Preferred two-group subset for pairwise comparisons.

Advice:
Provide two values from groupby_obs_groups when one comparison should be prioritised.

## Population annotations

### `population_obs_all`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Population or cluster annotation columns available to downstream stages.

Advice:
List adata.obs columns containing cell population or clustering labels.

### `population_obs_primary`

- Type: `Optional`
- Default: `null`
- Level: `basic`

Primary population annotation column used by downstream analyses.

Advice:
Set this to the preferred final cell population label column.

### `compartment_obs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Optional tissue-compartment annotation column in adata.obs.

Advice:
Set this when abundance or spatial outputs should be stratified by tissue compartment.

### `compartment_obs_list`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Optional ordered subset of tissue compartments to analyse separately.

Advice:
Leave unset to use all compartments or provide the desired subset and order.

## Spatial coordinates

### `spatial_key`

- Type: `str`
- Default: `spatial`
- Level: `advanced`

Canonical adata.obsm key containing XY spatial coordinates.

Advice:
Change only when coordinates are stored under a different obsm key.

### `x_coord_obs`

- Type: `str`
- Default: `X_loc`
- Level: `advanced`

Fallback adata.obs column containing X coordinates.

Advice:
Used when the configured spatial_key is unavailable.

### `y_coord_obs`

- Type: `str`
- Default: `Y_loc`
- Level: `advanced`

Fallback adata.obs column containing Y coordinates.

Advice:
Used when the configured spatial_key is unavailable.

## AnnData and execution

### `anndata_path`

- Type: `str`
- Default: `anndata.h5ad`
- Level: `basic`

Canonical AnnData file path used across pipeline stages.

Advice:
Use a path relative to the dataset working directory unless an absolute path is required.

### `anndata_stage_run_mode`

- Type: `str`
- Default: `repeat`
- Level: `advanced`

Default policy for rerunning stages recorded in AnnData.

Advice:
Use repeat, skip, or intelligent according to the desired stage rerun behaviour.

### `anndata_uns_log_key`

- Type: `str`
- Default: `pipeline_stage_log`
- Level: `expert`

AnnData.uns key used to store pipeline stage history and config snapshots.

Advice:
Keep the default unless integrating with an existing AnnData logging convention.
