# Nimbus

## Nimbus

### `output_dir`

- Type: `str`
- Default: `nimbus_output`
- Level: `advanced`

Configuration value for output dir.

Advice:
No additional advice.

### `roi_table_subfolder`

- Type: `str`
- Default: `nimbus_cell_tables`
- Level: `advanced`

Configuration value for roi table subfolder.

Advice:
No additional advice.

### `master_celltable`

- Type: `str`
- Default: `nimbus_celltable.csv`
- Level: `advanced`

Configuration value for master celltable.

Advice:
No additional advice.

### `master_classic_celltable`

- Type: `str`
- Default: `nimbus_classic_celltable.csv`
- Level: `advanced`

Configuration value for master classic celltable.

Advice:
No additional advice.

### `master_expansion_celltable`

- Type: `str`
- Default: `nimbus_expansion_celltable.csv`
- Level: `advanced`

Configuration value for master expansion celltable.

Advice:
No additional advice.

### `anndata_output`

- Type: `str`
- Default: `anndata.h5ad`
- Level: `advanced`

Configuration value for anndata output.

Advice:
No additional advice.

### `roi_table_prefix`

- Type: `str`
- Default: `nimbus_`
- Level: `advanced`

Configuration value for roi table prefix.

Advice:
No additional advice.

### `use_denoised_first`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for use denoised first.

Advice:
No additional advice.

### `allow_raw_fallback`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for allow raw fallback.

Advice:
No additional advice.

### `simple_image_names`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for simple image names.

Advice:
No additional advice.

### `mask_extensions`

- Type: `List`
- Default: `['.tiff', '.tif']`
- Level: `advanced`

Configuration value for mask extensions.

Advice:
No additional advice.

### `mask_boundary_offset_pixels`

- Type: `int`
- Default: `0`
- Level: `advanced`

Configuration value for mask boundary offset pixels.

Advice:
No additional advice.

### `min_cell_area`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for min cell area.

Advice:
No additional advice.

### `max_cell_area`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for max cell area.

Advice:
No additional advice.

### `test_time_augmentation`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for test time augmentation.

Advice:
No additional advice.

### `batch_size`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for batch size.

Advice:
No additional advice.

### `model_magnification`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for model magnification.

Advice:
No additional advice.

### `dataset_magnification`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for dataset magnification.

Advice:
No additional advice.

### `checkpoint`

- Type: `str`
- Default: `latest`
- Level: `advanced`

Configuration value for checkpoint.

Advice:
No additional advice.

### `device`

- Type: `str`
- Default: `auto`
- Level: `advanced`

Configuration value for device.

Advice:
No additional advice.

### `normalization_quantile`

- Type: `float`
- Default: `0.999`
- Level: `advanced`

Configuration value for normalization quantile.

Advice:
No additional advice.

### `normalization_subset`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for normalization subset.

Advice:
No additional advice.

### `normalization_jobs`

- Type: `int`
- Default: `1`
- Level: `advanced`

Configuration value for normalization jobs.

Advice:
No additional advice.

### `normalization_clip`

- Type: `List`
- Default: `[0.0, 1.0]`
- Level: `advanced`

Configuration value for normalization clip.

Advice:
No additional advice.

### `normalization_min_value`

- Type: `float`
- Default: `3.0`
- Level: `advanced`

Configuration value for normalization min value.

Advice:
No additional advice.

### `reuse_saved_normalization`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for reuse saved normalization.

Advice:
No additional advice.

### `norm_dict_qc_only`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for norm dict qc only.

Advice:
No additional advice.

### `save_prediction_maps`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for save prediction maps.

Advice:
No additional advice.

### `allow_prediction_resize`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for allow prediction resize.

Advice:
No additional advice.

### `use_existing_master_celltables`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for use existing master celltables.

Advice:
No additional advice.

### `extract_classic_intensities`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for extract classic intensities.

Advice:
No additional advice.

### `extract_expansion_intensities`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for extract expansion intensities.

Advice:
No additional advice.

### `expansion_pixels`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for expansion pixels.

Advice:
No additional advice.

### `expansion_jobs`

- Type: `int`
- Default: `1`
- Level: `advanced`

Configuration value for expansion jobs.

Advice:
No additional advice.
