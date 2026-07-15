# Cellcharter

## Cellcharter

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
- Default: `CellCharter_QC`
- Level: `advanced`

Configuration value for qc output subdir.

Advice:
No additional advice.

### `use_rep`

- Type: `Optional`
- Default: `X_biobatchnet`
- Level: `advanced`

Configuration value for use rep.

Advice:
No additional advice.

### `use_layer`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for use layer.

Advice:
No additional advice.

### `scale_by_sample`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for scale by sample.

Advice:
No additional advice.

### `scaled_rep_key`

- Type: `str`
- Default: `X_cellcharter_scaled`
- Level: `advanced`

Configuration value for scaled rep key.

Advice:
No additional advice.

### `use_trvae`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for use trvae.

Advice:
No additional advice.

### `trvae_latent_key`

- Type: `str`
- Default: `X_trVAE`
- Level: `advanced`

Configuration value for trvae latent key.

Advice:
No additional advice.

### `trvae_condition_key`

- Type: `Optional`
- Default: `dataset`
- Level: `advanced`

Configuration value for trvae condition key.

Advice:
No additional advice.

### `trvae_use_sample_key_fallback`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for trvae use sample key fallback.

Advice:
No additional advice.

### `trvae_constant_condition_label`

- Type: `str`
- Default: `all`
- Level: `advanced`

Configuration value for trvae constant condition label.

Advice:
No additional advice.

### `trvae_load_path`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for trvae load path.

Advice:
No additional advice.

### `trvae_save_path`

- Type: `str`
- Default: `trvae_model`
- Level: `advanced`

Configuration value for trvae save path.

Advice:
No additional advice.

### `trvae_map_location`

- Type: `str`
- Default: `gpu`
- Level: `advanced`

Configuration value for trvae map location.

Advice:
No additional advice.

### `trvae_train`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for trvae train.

Advice:
No additional advice.

### `trvae_train_early_stopping`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for trvae train early stopping.

Advice:
No additional advice.

### `trvae_train_enable_progress_bar`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for trvae train enable progress bar.

Advice:
No additional advice.

### `trvae_train_max_epochs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for trvae train max epochs.

Advice:
No additional advice.

### `trvae_hidden_layer_sizes`

- Type: `List`
- Default: `[128, 128]`
- Level: `advanced`

Configuration value for trvae hidden layer sizes.

Advice:
No additional advice.

### `trvae_latent_dim`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for trvae latent dim.

Advice:
No additional advice.

### `trvae_dr_rate`

- Type: `float`
- Default: `0.05`
- Level: `advanced`

Configuration value for trvae dr rate.

Advice:
No additional advice.

### `trvae_use_mmd`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for trvae use mmd.

Advice:
No additional advice.

### `trvae_mmd_on`

- Type: `str`
- Default: `z`
- Level: `advanced`

Configuration value for trvae mmd on.

Advice:
No additional advice.

### `trvae_mmd_boundary`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for trvae mmd boundary.

Advice:
No additional advice.

### `trvae_recon_loss`

- Type: `str`
- Default: `mse`
- Level: `advanced`

Configuration value for trvae recon loss.

Advice:
No additional advice.

### `trvae_beta`

- Type: `float`
- Default: `1.0`
- Level: `advanced`

Configuration value for trvae beta.

Advice:
No additional advice.

### `trvae_use_bn`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for trvae use bn.

Advice:
No additional advice.

### `trvae_use_ln`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for trvae use ln.

Advice:
No additional advice.

### `delaunay`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for delaunay.

Advice:
No additional advice.

### `remove_long_links`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for remove long links.

Advice:
No additional advice.

### `distance_percentile`

- Type: `float`
- Default: `99.0`
- Level: `advanced`

Configuration value for distance percentile.

Advice:
No additional advice.

### `n_layers`

- Type: `int`
- Default: `3`
- Level: `advanced`

Configuration value for n layers.

Advice:
No additional advice.

### `aggregations`

- Type: `str`
- Default: `mean`
- Level: `advanced`

Configuration value for aggregations.

Advice:
No additional advice.

### `aggregated_rep_key`

- Type: `str`
- Default: `X_cellcharter`
- Level: `advanced`

Configuration value for aggregated rep key.

Advice:
No additional advice.

### `n_clusters`

- Type: `int`
- Default: `11`
- Level: `advanced`

Configuration value for n clusters.

Advice:
No additional advice.

### `random_state`

- Type: `int`
- Default: `12345`
- Level: `advanced`

Configuration value for random state.

Advice:
No additional advice.

### `covariance_type`

- Type: `str`
- Default: `full`
- Level: `advanced`

Configuration value for covariance type.

Advice:
No additional advice.

### `batch_size`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for batch size.

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

### `trainer_max_epochs`

- Type: `int`
- Default: `100`
- Level: `advanced`

Configuration value for trainer max epochs.

Advice:
No additional advice.

### `cluster_key`

- Type: `str`
- Default: `spatial_cluster`
- Level: `advanced`

Configuration value for cluster key.

Advice:
No additional advice.

### `repeat_analysis`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for repeat analysis.

Advice:
No additional advice.

### `repeat_cluster_analysis`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for repeat cluster analysis.

Advice:
No additional advice.

### `repeat_enrichment_analysis`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for repeat enrichment analysis.

Advice:
No additional advice.

### `repeat_nhood_enrichment_analysis`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for repeat nhood enrichment analysis.

Advice:
No additional advice.

### `repeat_diff_nhood_enrichment_analysis`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for repeat diff nhood enrichment analysis.

Advice:
No additional advice.

### `repeat_shape_characterisation_analysis`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for repeat shape characterisation analysis.

Advice:
No additional advice.

### `run_enrichment`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for run enrichment.

Advice:
No additional advice.

### `enrichment_with_pvalues`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for enrichment with pvalues.

Advice:
No additional advice.

### `enrichment_n_perms`

- Type: `int`
- Default: `1000`
- Level: `advanced`

Configuration value for enrichment n perms.

Advice:
No additional advice.

### `enrichment_plot_figsize`

- Type: `List`
- Default: `[8.0, 6.0]`
- Level: `advanced`

Configuration value for enrichment plot figsize.

Advice:
No additional advice.

### `enrichment_plot_dot_scale`

- Type: `float`
- Default: `3.0`
- Level: `advanced`

Configuration value for enrichment plot dot scale.

Advice:
No additional advice.

### `enrichment_plot_show_pvalues`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for enrichment plot show pvalues.

Advice:
No additional advice.

### `enrichment_plot_significant_only`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for enrichment plot significant only.

Advice:
No additional advice.

### `run_nhood_enrichment`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for run nhood enrichment.

Advice:
No additional advice.

### `nhood_connectivity_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for nhood connectivity key.

Advice:
No additional advice.

### `nhood_log_fold_change`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for nhood log fold change.

Advice:
No additional advice.

### `nhood_only_inter`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for nhood only inter.

Advice:
No additional advice.

### `nhood_symmetric`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for nhood symmetric.

Advice:
No additional advice.

### `nhood_with_pvalues`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for nhood with pvalues.

Advice:
No additional advice.

### `nhood_n_perms`

- Type: `int`
- Default: `1000`
- Level: `advanced`

Configuration value for nhood n perms.

Advice:
No additional advice.

### `nhood_n_jobs`

- Type: `int`
- Default: `1`
- Level: `advanced`

Configuration value for nhood n jobs.

Advice:
No additional advice.

### `nhood_batch_size`

- Type: `int`
- Default: `10`
- Level: `advanced`

Configuration value for nhood batch size.

Advice:
No additional advice.

### `nhood_observed_expected`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for nhood observed expected.

Advice:
No additional advice.

### `save_nhood_enrichment_plot`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save nhood enrichment plot.

Advice:
No additional advice.

### `nhood_plot_figsize`

- Type: `List`
- Default: `[6.0, 3.0]`
- Level: `advanced`

Configuration value for nhood plot figsize.

Advice:
No additional advice.

### `nhood_enrichment_significance`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for nhood enrichment significance.

Advice:
No additional advice.

### `run_diff_nhood_enrichment`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for run diff nhood enrichment.

Advice:
No additional advice.

### `diff_nhood_condition_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for diff nhood condition key.

Advice:
No additional advice.

### `diff_nhood_condition_groups`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for diff nhood condition groups.

Advice:
No additional advice.

### `diff_nhood_connectivity_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for diff nhood connectivity key.

Advice:
No additional advice.

### `diff_nhood_log_fold_change`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for diff nhood log fold change.

Advice:
No additional advice.

### `diff_nhood_only_inter`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for diff nhood only inter.

Advice:
No additional advice.

### `diff_nhood_symmetric`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for diff nhood symmetric.

Advice:
No additional advice.

### `diff_nhood_with_pvalues`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for diff nhood with pvalues.

Advice:
No additional advice.

### `diff_nhood_library_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for diff nhood library key.

Advice:
No additional advice.

### `diff_nhood_n_perms`

- Type: `int`
- Default: `1000`
- Level: `advanced`

Configuration value for diff nhood n perms.

Advice:
No additional advice.

### `diff_nhood_n_jobs`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for diff nhood n jobs.

Advice:
No additional advice.

### `diff_nhood_plot_ncols`

- Type: `int`
- Default: `2`
- Level: `advanced`

Configuration value for diff nhood plot ncols.

Advice:
No additional advice.

### `save_diff_nhood_enrichment_plot`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save diff nhood enrichment plot.

Advice:
No additional advice.

### `run_shape_characterisation`

- Type: `bool`
- Default: `False`
- Level: `advanced`

Configuration value for run shape characterisation.

Advice:
No additional advice.

### `shape_component_key`

- Type: `str`
- Default: `component`
- Level: `advanced`

Configuration value for shape component key.

Advice:
No additional advice.

### `shape_component_cluster_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for shape component cluster key.

Advice:
No additional advice.

### `shape_connectivity_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for shape connectivity key.

Advice:
No additional advice.

### `shape_min_cells`

- Type: `int`
- Default: `250`
- Level: `advanced`

Configuration value for shape min cells.

Advice:
No additional advice.

### `shape_min_hole_area_ratio`

- Type: `float`
- Default: `0.1`
- Level: `advanced`

Configuration value for shape min hole area ratio.

Advice:
No additional advice.

### `shape_alpha_start`

- Type: `int`
- Default: `2000`
- Level: `advanced`

Configuration value for shape alpha start.

Advice:
No additional advice.

### `shape_compute_linearity`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for shape compute linearity.

Advice:
No additional advice.

### `shape_linearity_key`

- Type: `str`
- Default: `linearity`
- Level: `advanced`

Configuration value for shape linearity key.

Advice:
No additional advice.

### `shape_linearity_height`

- Type: `int`
- Default: `1000`
- Level: `advanced`

Configuration value for shape linearity height.

Advice:
No additional advice.

### `shape_linearity_min_ratio`

- Type: `float`
- Default: `0.05`
- Level: `advanced`

Configuration value for shape linearity min ratio.

Advice:
No additional advice.

### `shape_compute_curl`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for shape compute curl.

Advice:
No additional advice.

### `shape_curl_key`

- Type: `str`
- Default: `curl`
- Level: `advanced`

Configuration value for shape curl key.

Advice:
No additional advice.

### `shape_plot_metrics`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for shape plot metrics.

Advice:
No additional advice.

### `shape_metrics_condition_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for shape metrics condition key.

Advice:
No additional advice.

### `shape_metrics_condition_groups`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for shape metrics condition groups.

Advice:
No additional advice.

### `shape_metrics_cluster_key`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for shape metrics cluster key.

Advice:
No additional advice.

### `shape_metrics_cluster_groups`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for shape metrics cluster groups.

Advice:
No additional advice.

### `shape_metrics_ncols`

- Type: `int`
- Default: `2`
- Level: `advanced`

Configuration value for shape metrics ncols.

Advice:
No additional advice.

### `save_spatial_plots`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save spatial plots.

Advice:
No additional advice.

### `max_rois_for_plots`

- Type: `int`
- Default: `12`
- Level: `advanced`

Configuration value for max rois for plots.

Advice:
No additional advice.

### `point_size`

- Type: `float`
- Default: `2.0`
- Level: `advanced`

Configuration value for point size.

Advice:
No additional advice.

### `save_enrichment_heatmap`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save enrichment heatmap.

Advice:
No additional advice.

### `cluster_default_cmap`

- Type: `Optional`
- Default: `null`
- Level: `advanced`

Configuration value for cluster default cmap.

Advice:
No additional advice.

### `save_cluster_umap`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save cluster umap.

Advice:
No additional advice.

### `cluster_umap_point_size`

- Type: `float`
- Default: `10.0`
- Level: `advanced`

Configuration value for cluster umap point size.

Advice:
No additional advice.

### `cluster_umap_legend_loc`

- Type: `str`
- Default: `right margin`
- Level: `advanced`

Configuration value for cluster umap legend loc.

Advice:
No additional advice.

### `save_cluster_composition_plots`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save cluster composition plots.

Advice:
No additional advice.

### `composition_order_by_environment`

- Type: `str`
- Default: `0`
- Level: `advanced`

Configuration value for composition order by environment.

Advice:
No additional advice.

### `composition_stacked_figsize`

- Type: `List`
- Default: `[6.0, 3.0]`
- Level: `advanced`

Configuration value for composition stacked figsize.

Advice:
No additional advice.

### `composition_stacked_width_scale`

- Type: `float`
- Default: `0.3`
- Level: `advanced`

Configuration value for composition stacked width scale.

Advice:
No additional advice.

### `composition_group_barplot_figsize`

- Type: `List`
- Default: `[6.0, 3.0]`
- Level: `advanced`

Configuration value for composition group barplot figsize.

Advice:
No additional advice.

### `figure_extension`

- Type: `str`
- Default: `.png`
- Level: `advanced`

Configuration value for figure extension.

Advice:
No additional advice.

### `figure_format`

- Type: `str`
- Default: `png`
- Level: `advanced`

Configuration value for figure format.

Advice:
No additional advice.

### `save_high_res`

- Type: `bool`
- Default: `True`
- Level: `advanced`

Configuration value for save high res.

Advice:
No additional advice.
