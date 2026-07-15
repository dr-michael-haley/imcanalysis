# Rapids

## Rapids

### `input_adata_path`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for input adata path.

Advice:
No additional advice.

### `output_adata_path`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for output adata path.

Advice:
No additional advice.

### `batch_correction_obs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for batch correction obs.

Advice:
No additional advice.

### `run_harmony`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for run harmony.

Advice:
No additional advice.

### `harmony_flavor`

- Type: `str`
- Default: `harmony2`
- Level: `advanced`

Configuration value for harmony flavor.

Advice:
No additional advice.

### `n_for_pca`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for n for pca.

Advice:
No additional advice.

### `n_pcs_neighbors`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for n pcs neighbors.

Advice:
No additional advice.

### `leiden_resolutions_list`

- Type: `List`
- Default: `[0.3, 1.0]`
- Level: `advanced`

Configuration value for leiden resolutions list.

Advice:
No additional advice.

### `umap_min_dist`

- Type: `float`
- Default: `0.1`
- Level: `advanced`

Configuration value for umap min dist.

Advice:
No additional advice.

### `run_leiden`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for run leiden.

Advice:
No additional advice.

### `n_neighbors`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for n neighbors.

Advice:
No additional advice.

### `filter_obs_key`

- Type: `str`
- Default: `mask_area`
- Level: `advanced`

Configuration value for filter obs key.

Advice:
No additional advice.

### `filter_min_value`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for filter min value.

Advice:
No additional advice.

### `filter_max_value`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for filter max value.

Advice:
No additional advice.

### `parameter_scan_dict`

- Type: `Dict`
- Default: `{}`
- Level: `advanced`

Configuration value for parameter scan dict.

Advice:
No additional advice.

### `parameter_scan_save_anndata`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for parameter scan save anndata.

Advice:
No additional advice.

### `parameter_scan_qc_subdir`

- Type: `str`
- Default: `ParameterScan`
- Level: `advanced`

Configuration value for parameter scan qc subdir.

Advice:
No additional advice.

### `input_representation_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for input representation key.

Advice:
No additional advice.

### `pca_key`

- Type: `str`
- Default: `X_pca`
- Level: `advanced`

Configuration value for pca key.

Advice:
No additional advice.

### `harmony_key`

- Type: `str`
- Default: `X_pca_harmony`
- Level: `advanced`

Configuration value for harmony key.

Advice:
No additional advice.

### `representation_key`

- Type: `str`
- Default: `X_batch_integration`
- Level: `advanced`

Configuration value for representation key.

Advice:
No additional advice.

### `neighbors_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for neighbors key.

Advice:
No additional advice.

### `umap_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for umap key.

Advice:
No additional advice.

### `qc_output_subdir`

- Type: `str`
- Default: `RapidsProcess`
- Level: `advanced`

Configuration value for qc output subdir.

Advice:
No additional advice.

### `pca_params`

- Type: `Dict`
- Default: `{}`
- Level: `advanced`

Configuration value for pca params.

Advice:
No additional advice.

### `harmony_params`

- Type: `Dict`
- Default: `{'max_iter_harmony': 30, 'random_state': 0, 'verbose': True, 'dtype': 'float32'}`
- Level: `advanced`

Configuration value for harmony params.

Advice:
No additional advice.

### `neighbors_params`

- Type: `Dict`
- Default: `{}`
- Level: `advanced`

Configuration value for neighbors params.

Advice:
No additional advice.

### `umap_params`

- Type: `Dict`
- Default: `{}`
- Level: `advanced`

Configuration value for umap params.

Advice:
No additional advice.

### `leiden_params`

- Type: `Dict`
- Default: `{}`
- Level: `advanced`

Configuration value for leiden params.

Advice:
No additional advice.
