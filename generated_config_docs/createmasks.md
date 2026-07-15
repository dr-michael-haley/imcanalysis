# Createmasks

## Createmasks

### `specific_rois`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for specific rois.

Advice:
No additional advice.

### `dna_image_name`

- Type: `str`
- Default: `DNA1`
- Level: `advanced`

Configuration value for dna image name.

Advice:
No additional advice.

### `dna_preprocessing_output_folder_name`

- Type: `str`
- Default: `preprocessed_dna`
- Level: `advanced`

Configuration value for dna preprocessing output folder name.

Advice:
No additional advice.

### `upscale_ratio`

- Type: `float`
- Default: `1.7`
- Level: `advanced`

Configuration value for upscale ratio.

Advice:
No additional advice.

### `expand_masks`

- Type: `int`
- Default: `1`
- Level: `advanced`

Configuration value for expand masks.

Advice:
No additional advice.

### `perform_qc`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for perform qc.

Advice:
No additional advice.

### `qc_boundary_dilation`

- Type: `int`
- Default: `0`
- Level: `advanced`

Configuration value for qc boundary dilation.

Advice:
No additional advice.

### `min_cell_area`

- Type: `Optional`
- Default: `15`
- Level: `advanced`

Configuration value for min cell area.

Advice:
No additional advice.

### `max_cell_area`

- Type: `Optional`
- Default: `200`
- Level: `advanced`

Configuration value for max cell area.

Advice:
No additional advice.

### `cell_pose_model`

- Type: `str`
- Default: `nuclei`
- Level: `advanced`

Configuration value for cell pose model.

Advice:
No additional advice.

### `cell_pose_sam_model`

- Type: `str`
- Default: `cpsam`
- Level: `advanced`

Configuration value for cell pose sam model.

Advice:
No additional advice.

### `cellprob_threshold`

- Type: `float`
- Default: `0.0`
- Level: `advanced`

Configuration value for cellprob threshold.

Advice:
No additional advice.

### `flow_threshold`

- Type: `float`
- Default: `0.4`
- Level: `advanced`

Configuration value for flow threshold.

Advice:
No additional advice.

### `run_deblur`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for run deblur.

Advice:
No additional advice.

### `run_upscale`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for run upscale.

Advice:
No additional advice.

### `image_normalise`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for image normalise.

Advice:
No additional advice.

### `image_normalise_percentile_lower`

- Type: `float`
- Default: `0.0`
- Level: `advanced`

Configuration value for image normalise percentile lower.

Advice:
No additional advice.

### `image_normalise_percentile_upper`

- Type: `float`
- Default: `99.9`
- Level: `advanced`

Configuration value for image normalise percentile upper.

Advice:
No additional advice.

### `dpi_qc_images`

- Type: `int`
- Default: `300`
- Level: `advanced`

Configuration value for dpi qc images.

Advice:
No additional advice.

### `max_size_fraction`

- Type: `float`
- Default: `0.4`
- Level: `advanced`

Configuration value for max size fraction.

Advice:
No additional advice.

### `remove_edge_masks`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for remove edge masks.

Advice:
No additional advice.

### `fill_holes`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for fill holes.

Advice:
No additional advice.

### `batch_size`

- Type: `int`
- Default: `128`
- Level: `advanced`

Configuration value for batch size.

Advice:
No additional advice.

### `resample`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for resample.

Advice:
No additional advice.

### `augment`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for augment.

Advice:
No additional advice.

### `tile_overlap`

- Type: `float`
- Default: `0.1`
- Level: `advanced`

Configuration value for tile overlap.

Advice:
No additional advice.

### `upscale_model_type`

- Type: `str`
- Default: `upsample_nuclei`
- Level: `advanced`

Configuration value for upscale model type.

Advice:
No additional advice.

### `run_parameter_scan`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for run parameter scan.

Advice:
No additional advice.

### `param_a`

- Type: `Optional`
- Default: `cellprob_threshold`
- Level: `advanced`

Configuration value for param a.

Advice:
No additional advice.

### `param_a_values`

- Type: `Optional`
- Default: `[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]`
- Level: `advanced`

Configuration value for param a values.

Advice:
No additional advice.

### `param_b`

- Type: `Optional`
- Default: `flow_threshold`
- Level: `advanced`

Configuration value for param b.

Advice:
No additional advice.

### `param_b_values`

- Type: `Optional`
- Default: `[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]`
- Level: `advanced`

Configuration value for param b values.

Advice:
No additional advice.

### `window_size`

- Type: `Optional`
- Default: `250`
- Level: `advanced`

Configuration value for window size.

Advice:
No additional advice.

### `num_rois_to_scan`

- Type: `int`
- Default: `3`
- Level: `advanced`

Configuration value for num rois to scan.

Advice:
No additional advice.

### `scan_rois`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for scan rois.

Advice:
No additional advice.

## Segmentation

### `cellpose_cell_diameter`

- Type: `float`
- Default: `10.0`
- Level: `basic`

Approximate Cellpose cell diameter in pixels.

Advice:
Increase when cells are fragmented; decrease when neighbouring cells are merged.
