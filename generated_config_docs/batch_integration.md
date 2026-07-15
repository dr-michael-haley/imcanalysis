# Batch Integration

## Batch Integration

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

### `integration_method`

- Type: `str`
- Default: `harmony`
- Level: `advanced`

Configuration value for integration method.

Advice:
No additional advice.

### `batch_correction_method`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for batch correction method.

Advice:
No additional advice.

### `n_for_pca`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for n for pca.

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

### `qc_output_subdir`

- Type: `str`
- Default: `BatchIntegration`
- Level: `advanced`

Configuration value for qc output subdir.

Advice:
No additional advice.

### `harmony_params`

- Type: `Dict`
- Default: `{'max_iter_harmony': 30, 'verbose': True, 'random_state': 0, 'device': None}`
- Level: `advanced`

Configuration value for harmony params.

Advice:
No additional advice.

### `bbknn_params`

- Type: `Dict`
- Default: `{}`
- Level: `advanced`

Configuration value for bbknn params.

Advice:
No additional advice.
