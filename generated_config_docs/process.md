# Process

## Process

### `input_adata_path`

- Type: `str`
- Default: `anndata.h5ad`
- Level: `advanced`

Configuration value for input adata path.

Advice:
No additional advice.

### `output_adata_path`

- Type: `str`
- Default: `anndata_processed.h5ad`
- Level: `advanced`

Configuration value for output adata path.

Advice:
No additional advice.

## Biobatchnet

### `batch_correction_obs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for batch correction obs.

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

### `biobatchnet_params`

- Type: `Optional`
- Default: `{'data_type': 'imc', 'latent_dim': 20, 'epochs': 100, 'device': None, 'use_raw': False, 'extra_params': {'loss_weights': {'recon_loss': 100.0, 'discriminator': 0.05, 'classifier': 1.0, 'kl_loss_1': 0.0005, 'kl_loss_2': 0.1, 'ortho_loss': 0.01}}}`
- Level: `advanced`

Configuration value for biobatchnet params.

Advice:
No additional advice.

### `biobatchnet_scan_parameter_sets`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for biobatchnet scan parameter sets.

Advice:
No additional advice.

### `biobatchnet_scan_include_base`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for biobatchnet scan include base.

Advice:
No additional advice.

### `biobatchnet_run_postprocess`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for biobatchnet run postprocess.

Advice:
No additional advice.

### `biobatchnet_run_leiden`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for biobatchnet run leiden.

Advice:
No additional advice.

### `n_neighbors`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for n neighbors.

Advice:
No additional advice.

### `biobatchnet_data_type`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for biobatchnet data type.

Advice:
No additional advice.

### `biobatchnet_latent_dim`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for biobatchnet latent dim.

Advice:
No additional advice.

### `biobatchnet_epochs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for biobatchnet epochs.

Advice:
No additional advice.

### `biobatchnet_device`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for biobatchnet device.

Advice:
No additional advice.

### `biobatchnet_kwargs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for biobatchnet kwargs.

Advice:
No additional advice.

### `biobatchnet_use_raw`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for biobatchnet use raw.

Advice:
No additional advice.
