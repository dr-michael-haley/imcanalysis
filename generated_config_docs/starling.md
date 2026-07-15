# Starling

## Starling

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

### `qc_output_subdir`

- Type: `str`
- Default: `Starling_QC`
- Level: `advanced`

Configuration value for qc output subdir.

Advice:
No additional advice.

### `starling_repo_path`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for starling repo path.

Advice:
No additional advice.

### `use_layer`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for use layer.

Advice:
No additional advice.

### `marker_include`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for marker include.

Advice:
No additional advice.

### `marker_exclude`

- Type: `List`
- Default: `[]`
- Level: `advanced`

Configuration value for marker exclude.

Advice:
No additional advice.

### `clip_small_negative_values`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for clip small negative values.

Advice:
No additional advice.

### `negative_value_tolerance`

- Type: `float`
- Default: `1e-08`
- Level: `advanced`

Configuration value for negative value tolerance.

Advice:
No additional advice.

### `initial_clustering_method`

- Type: `str`
- Default: `User`
- Level: `advanced`

Configuration value for initial clustering method.

Advice:
No additional advice.

### `initial_label_obs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for initial label obs.

Advice:
No additional advice.

### `n_clusters`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for n clusters.

Advice:
No additional advice.

### `seed`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for seed.

Advice:
No additional advice.

### `dist_option`

- Type: `str`
- Default: `T`
- Level: `advanced`

Configuration value for dist option.

Advice:
No additional advice.

### `singlet_prop`

- Type: `float`
- Default: `0.6`
- Level: `advanced`

Configuration value for singlet prop.

Advice:
No additional advice.

### `model_cell_size`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for model cell size.

Advice:
No additional advice.

### `cell_size_col_name`

- Type: `str`
- Default: `mask_area`
- Level: `advanced`

Configuration value for cell size col name.

Advice:
No additional advice.

### `cell_size_fallback_cols`

- Type: `List`
- Default: `['area']`
- Level: `advanced`

Configuration value for cell size fallback cols.

Advice:
No additional advice.

### `model_zplane_overlap`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for model zplane overlap.

Advice:
No additional advice.

### `model_regularizer`

- Type: `float`
- Default: `1.0`
- Level: `advanced`

Configuration value for model regularizer.

Advice:
No additional advice.

### `learning_rate`

- Type: `float`
- Default: `0.001`
- Level: `advanced`

Configuration value for learning rate.

Advice:
No additional advice.

### `doublet_threshold`

- Type: `float`
- Default: `0.5`
- Level: `advanced`

Configuration value for doublet threshold.

Advice:
No additional advice.

### `max_epochs`

- Type: `Optional`
- Default: `100`
- Level: `advanced`

Configuration value for max epochs.

Advice:
No additional advice.

### `early_stopping`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for early stopping.

Advice:
No additional advice.

### `early_stopping_monitor`

- Type: `str`
- Default: `train_loss`
- Level: `advanced`

Configuration value for early stopping monitor.

Advice:
No additional advice.

### `trainer_accelerator`

- Type: `str`
- Default: `auto`
- Level: `advanced`

Configuration value for trainer accelerator.

Advice:
No additional advice.

### `trainer_devices`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for trainer devices.

Advice:
No additional advice.

### `trainer_precision`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for trainer precision.

Advice:
No additional advice.

### `enable_checkpointing`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for enable checkpointing.

Advice:
No additional advice.

### `enable_progress_bar`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for enable progress bar.

Advice:
No additional advice.

### `log_every_n_steps`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for log every n steps.

Advice:
No additional advice.

### `limit_train_batches`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for limit train batches.

Advice:
No additional advice.

### `tensorboard_logging`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for tensorboard logging.

Advice:
No additional advice.

### `output_prefix`

- Type: `str`
- Default: `starling`
- Level: `advanced`

Configuration value for output prefix.

Advice:
No additional advice.

### `write_canonical_starling_keys`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for write canonical starling keys.

Advice:
No additional advice.

### `store_assignment_prob_matrix`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for store assignment prob matrix.

Advice:
No additional advice.

### `store_gamma_assignment_prob_matrix`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for store gamma assignment prob matrix.

Advice:
No additional advice.

### `save_model`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for save model.

Advice:
No additional advice.

### `model_output_name`

- Type: `str`
- Default: `starling_model.pt`
- Level: `advanced`

Configuration value for model output name.

Advice:
No additional advice.

### `save_qc_tables`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save qc tables.

Advice:
No additional advice.

### `save_qc_plots`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save qc plots.

Advice:
No additional advice.

### `figure_format`

- Type: `str`
- Default: `png`
- Level: `advanced`

Configuration value for figure format.

Advice:
No additional advice.
