# Denoising

## Denoising

### `run_denoising`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for run denoising.

Advice:
No additional advice.

### `method`

- Type: `str`
- Default: `deep_snf`
- Level: `advanced`

Configuration value for method.

Advice:
No additional advice.

### `channels`

- Type: `List`
- Default: `[]`
- Level: `advanced`

Configuration value for channels.

Advice:
No additional advice.

### `n_neighbours`

- Type: `int`
- Default: `4`
- Level: `advanced`

Configuration value for n neighbours.

Advice:
No additional advice.

### `n_iter`

- Type: `int`
- Default: `3`
- Level: `advanced`

Configuration value for n iter.

Advice:
No additional advice.

### `window_size`

- Type: `int`
- Default: `3`
- Level: `advanced`

Configuration value for window size.

Advice:
No additional advice.

### `remove_outliers`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for remove outliers.

Advice:
No additional advice.

### `remove_outliers_min_threshold`

- Type: `int`
- Default: `500`
- Level: `advanced`

Configuration value for remove outliers min threshold.

Advice:
No additional advice.

### `patch_step_size`

- Type: `int`
- Default: `100`
- Level: `advanced`

Configuration value for patch step size.

Advice:
No additional advice.

### `intelligent_patch_size`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for intelligent patch size.

Advice:
No additional advice.

### `intelligent_patch_size_threshold`

- Type: `float`
- Default: `0.3`
- Level: `advanced`

Configuration value for intelligent patch size threshold.

Advice:
No additional advice.

### `intelligent_patch_size_minimum`

- Type: `int`
- Default: `40`
- Level: `advanced`

Configuration value for intelligent patch size minimum.

Advice:
No additional advice.

### `intelligent_patch_size_min_patches`

- Type: `int`
- Default: `5000`
- Level: `advanced`

Configuration value for intelligent patch size min patches.

Advice:
No additional advice.

### `intelligent_patch_size_max_patches`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for intelligent patch size max patches.

Advice:
No additional advice.

### `train_epochs`

- Type: `int`
- Default: `75`
- Level: `advanced`

Configuration value for train epochs.

Advice:
No additional advice.

### `train_initial_lr`

- Type: `float`
- Default: `0.001`
- Level: `advanced`

Configuration value for train initial lr.

Advice:
No additional advice.

### `train_batch_size`

- Type: `int`
- Default: `200`
- Level: `advanced`

Configuration value for train batch size.

Advice:
No additional advice.

### `ratio_thresh`

- Type: `float`
- Default: `0.8`
- Level: `advanced`

Configuration value for ratio thresh.

Advice:
No additional advice.

### `pixel_mask_percent`

- Type: `float`
- Default: `0.2`
- Level: `advanced`

Configuration value for pixel mask percent.

Advice:
No additional advice.

### `val_set_percent`

- Type: `float`
- Default: `0.15`
- Level: `advanced`

Configuration value for val set percent.

Advice:
No additional advice.

### `loss_function`

- Type: `str`
- Default: `I_divergence`
- Level: `advanced`

Configuration value for loss function.

Advice:
No additional advice.

### `loss_name`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for loss name.

Advice:
No additional advice.

### `weights_save_directory`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for weights save directory.

Advice:
No additional advice.

### `is_load_weights`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for is load weights.

Advice:
No additional advice.

### `lambda_HF`

- Type: `float`
- Default: `3e-06`
- Level: `advanced`

Configuration value for lambda HF.

Advice:
No additional advice.

### `network_size`

- Type: `str`
- Default: `small`
- Level: `advanced`

Configuration value for network size.

Advice:
No additional advice.

### `truncated_max_rate`

- Type: `float`
- Default: `0.99999`
- Level: `advanced`

Configuration value for truncated max rate.

Advice:
No additional advice.

### `run_parameter_scan`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for run parameter scan.

Advice:
No additional advice.

### `scan_parameter`

- Type: `Optional`
- Default: `truncated_max_rate`
- Level: `advanced`

Configuration value for scan parameter.

Advice:
No additional advice.

### `scan_values`

- Type: `Optional`
- Default: `[0.99, 0.999, 0.99999]`
- Level: `advanced`

Configuration value for scan values.

Advice:
No additional advice.

### `verbose_training`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for verbose training.

Advice:
No additional advice.

### `run_QC`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for run QC.

Advice:
No additional advice.

### `colourmap`

- Type: `str`
- Default: `jet`
- Level: `advanced`

Configuration value for colourmap.

Advice:
No additional advice.

### `dpi`

- Type: `int`
- Default: `100`
- Level: `advanced`

Configuration value for dpi.

Advice:
No additional advice.

### `qc_image_dir`

- Type: `str`
- Default: `denoising`
- Level: `advanced`

Configuration value for qc image dir.

Advice:
No additional advice.

### `qc_num_rois`

- Type: `Optional`
- Default: `10`
- Level: `advanced`

Configuration value for qc num rois.

Advice:
No additional advice.

### `skip_already_denoised`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for skip already denoised.

Advice:
No additional advice.
