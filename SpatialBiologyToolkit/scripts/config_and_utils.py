import os
import yaml
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict, is_dataclass
from collections.abc import MutableMapping

@dataclass
class GeneralConfig:
    imc_files_folder: str = 'IMC_files'  # Supports both .mcd and .txt files
    mcd_files_folder: str = 'MCD_files'  # Kept for backward compatibility
    metadata_folder: str = 'metadata'
    qc_folder: str = 'QC'
    masks_folder: str = 'masks'
    celltable_folder: str = 'cell_tables'
    tiff_stacks_folder: str  = 'tiff_stacks'
    raw_images_folder: str = 'tiffs'
    denoised_images_folder: str = 'processed'
    slurm_logs_folder: str = 'SLURM_logs'
    case_obs: Optional[str] = None  # Optional case/sample column in adata.obs (used for case-level summaries/stats)
    roi_obs: str = 'ROI'  # ROI identifier column in adata.obs
    metadata_obs: Optional[List[str]] = None  # Optional metadata obs columns for QC and grouped summaries
    groupby_obs: Optional[str] = None  # Primary grouping axis for cross-condition analyses
    groupby_obs_groups: Optional[List[str]] = None  # Optional ordered subset for groupby_obs
    groupby_obs_primary_pairwise: Optional[List[str]] = None  # Optional preferred 2-group subset for pairwise analyses
    population_obs_all: Optional[List[str]] = None  # Optional full list of population/cluster obs columns
    population_obs_primary: Optional[str] = None  # Optional primary population obs used by downstream analyses
    spatial_key: str = 'spatial'  # Canonical adata.obsm key for XY coordinates
    x_coord_obs: str = 'X_loc'  # Fallback X coordinate obs column when spatial_key is missing
    y_coord_obs: str = 'Y_loc'  # Fallback Y coordinate obs column when spatial_key is missing
    master_index_obs: str = 'Master_Index'  # Canonical stable per-cell index column in adata.obs
    anndata_path: str = 'anndata.h5ad'  # Canonical AnnData file path used across pipeline stages
    anndata_stage_run_mode: str = 'intelligent'  # One of: repeat, skip, intelligent
    anndata_uns_log_key: str = 'pipeline_stage_log'  # AnnData.uns key storing stage order/config snapshots

@dataclass
class PreprocessConfig:
    minimum_roi_dimensions: int = 200

@dataclass
class DenoisingConfig:
    run_denoising: bool = True
    method: str = 'deep_snf'  # Options: 'deep_snf', 'dimr'
    channels: List[str] = field(default_factory=list)
    # Parameters for both methods
    n_neighbours: int = 4
    n_iter: int = 3
    window_size: int = 3
    # Outlier removal
    remove_outliers: bool = True
    remove_outliers_min_threshold: int = 500
    # Parameters specific to 'deep_snf' method
    patch_step_size: int = 100
    intelligent_patch_size: bool = True
    intelligent_patch_size_threshold: float = 0.3  # e.g., 20%
    intelligent_patch_size_minimum: int = 40
    intelligent_patch_size_min_patches: int = 5000  # Minimum number of patches required
    intelligent_patch_size_max_patches: Optional[int] = None  # Maximum number of patches (None = no limit)
    # DeepSNIF
    train_epochs: int = 75
    train_initial_lr: float = 0.001
    train_batch_size: int = 200
    ratio_thresh: float = 0.8 # Added
    pixel_mask_percent: float = 0.2
    val_set_percent: float = 0.15
    loss_function: str = "I_divergence"
    loss_name: Optional[str] = None
    weights_save_directory: Optional[str] = None
    is_load_weights: bool = False
    lambda_HF: float = 3e-6
    network_size: str = "small"
    truncated_max_rate: float = 0.99999
    # Parameter scanning
    run_parameter_scan: bool = False
    scan_parameter: Optional[str] = 'truncated_max_rate'  # Name of parameter to scan (e.g., 'train_epochs', 'lambda_HF')
    scan_values: Optional[List[Any]] = field(default_factory=lambda: [0.99, 0.999, 0.99999])  # List of values to test for the scan parameter
    # Training verbosity
    verbose_training: bool = False  # Show detailed TensorFlow/Keras training output (progress bars, epoch details)
    # Parameters for QC images
    run_QC: bool = True
    colourmap: str = "jet"
    dpi: int = 100
    qc_image_dir: str = 'denoising'
    qc_num_rois: Optional[int] = 10  # Number of random ROIs to include in QC (None = all ROIs)
    skip_already_denoised: bool = True

@dataclass
class CreateMasksConfig:
    specific_rois: Optional[List[str]] = None
    dna_image_name: str = 'DNA1'
    dna_preprocessing_output_folder_name: str = 'preprocessed_dna'  # For DNA preprocessing output
    cellpose_cell_diameter: float = 10.0  # Works in both CellPose v3 and v4+ (behavior may differ)
    upscale_ratio: float = 1.7
    expand_masks: int = 1
    perform_qc: bool = True
    qc_boundary_dilation: int = 0
    min_cell_area: Optional[int] = 15
    max_cell_area: Optional[int] = 200
    cell_pose_model: str = 'nuclei'  # For CellPose v3 (original createmasks) - DEPRECATED in v4+
    cell_pose_sam_model: str = 'cpsam'  # For CellPose v4+ (cellpose_sam script) - only 'cpsam' or user models
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    run_deblur: bool = True
    run_upscale: bool = True
    image_normalise: bool = True
    image_normalise_percentile_lower: float = 0.0
    image_normalise_percentile_upper: float =  99.9
    dpi_qc_images: int = 300

    # CellPose-SAM mode toggle and settings - uses dna_preprocessing_output_folder_name for input, GeneralConfig.masks_folder for output
    max_size_fraction: float = 0.4              # Max cell size as fraction of image
    remove_edge_masks: bool = False         # Remove masks touching image edges  
    fill_holes: bool = True                 # Fill holes in segmented masks
    batch_size: int = 128                     # Batch size for segmentation
    resample: bool = True                   # Resample for better boundaries
    augment: bool = False                   # Use test-time augmentation
    tile_overlap: float = 0.1               # Overlap fraction for tiling

    # Upscale model configuration
    upscale_model_type: str = 'upsample_nuclei'  # 'upsample_nuclei' or 'upsample_cyto3'
    
    @property
    def upscale_target_diameter(self) -> float:
        """Get the target diameter for the upscale model."""
        if self.upscale_model_type == 'upsample_nuclei':
            return 17.0
        elif self.upscale_model_type == 'upsample_cyto3':
            return 30.0
        else:
            # Fallback to calculated ratio
            return self.cellpose_cell_diameter * self.upscale_ratio
    
    @property 
    def calculated_upscale_ratio(self) -> float:
        """Calculate the actual upscale ratio based on target diameter."""
        return self.upscale_target_diameter / self.cellpose_cell_diameter

    # Parameter scanning fields:
    run_parameter_scan: bool = False
    param_a: Optional[str] = 'cellprob_threshold'
    param_a_values: Optional[List[Any]] = field(default_factory=lambda: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    param_b: Optional[str] = 'flow_threshold'
    param_b_values: Optional[List[Any]] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    window_size: Optional[int] = 250
    num_rois_to_scan: int = 3
    scan_rois: Optional[List[str]] = None

@dataclass
class SegmentationConfig:
    celltable_output: str = 'celltable.csv'
    marker_normalisation: List[str] = field(default_factory=lambda: ["q0.999"])
    store_raw_marker_data: bool = False
    remove_channels_list: List[str] = field(default_factory=lambda: ['DNA1', 'DNA3'])
    remove_and_store_markers: List[str] = field(default_factory=list)  # Markers to remove from main AnnData and store separately
    removed_markers_anndata_path: str = 'anndata_removed.h5ad'  # Path for AnnData containing removed markers
    anndata_save_path: str = 'anndata.h5ad'
    create_roi_cell_tables: bool = True
    create_master_cell_table: bool = True
    create_anndata: bool = True
    allow_missing_channels: bool = False  # If True, fill missing channels with NaN; if False, only include channels present in all ROIs

@dataclass
class NimbusConfig:
    output_dir: str = 'nimbus_output'
    roi_table_subfolder: str = 'nimbus_cell_tables'
    master_celltable: str = 'nimbus_celltable.csv'
    master_classic_celltable: str = 'nimbus_classic_celltable.csv'
    master_expansion_celltable: str = 'nimbus_expansion_celltable.csv'
    anndata_output: str = 'anndata.h5ad'
    roi_table_prefix: str = 'nimbus_'
    use_denoised_first: bool = True
    allow_raw_fallback: bool = True
    mask_extensions: List[str] = field(default_factory=lambda: ['.tiff', '.tif'])
    test_time_augmentation: bool = True
    batch_size: int = 10
    model_magnification: int = 10
    dataset_magnification: int = 10
    checkpoint: str = 'latest'
    device: str = 'auto'
    normalization_quantile: float = 0.999
    normalization_subset: int = 10
    normalization_jobs: int = 1
    normalization_clip: List[float] = field(default_factory=lambda: [0.0, 1.0])
    normalization_min_value: float = 3.0  # Minimum normalization value to avoid background noise
    reuse_saved_normalization: bool = False  # Reuse existing normalization_dict.json if found (allows manual tweaking)
    norm_dict_qc_only: bool = False  # If True, stop after normalization dict computation and QC generation
    save_prediction_maps: bool = False
    allow_prediction_resize: bool = False  # If True, fall back to resizing predictions when shapes mismatch
    overwrite_existing_outputs: bool = True
    use_existing_master_celltables: bool = False  # If True, reuse existing master cell tables when found
    extract_classic_intensities: bool = True  # Extract classic mean intensities over masks
    extract_expansion_intensities: bool = True  # Extract mean intensities from expanded masks
    expansion_pixels: int = 2  # Number of pixels to expand masks for expansion intensities
    expansion_jobs: int = 1  # Number of parallel jobs for expansion extraction (1=sequential, -1=all CPUs)

@dataclass
class BioBatchNetConfig:
    batch_correction_obs: Optional[str] = None
    n_for_pca: Optional[int] = None
    leiden_resolutions_list: List[float] = field(default_factory=lambda: [0.3, 1.0])
    umap_min_dist: float = 0.1

    # BioBatchNet-specific parameters (nested dictionary format)
    biobatchnet_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'data_type': 'imc',
        'latent_dim': 20,
        'epochs': 100,
        'device': None,
        'use_raw': False,
        'extra_params': {
            'loss_weights': {
                'recon_loss': 100.0,
                'discriminator': 0.05,  # Batch mixing (default: 0.3 - lower = more mixing)
                'classifier': 1.0,  # Batch retention (default: 1)
                'kl_loss_1': 0.0005,  # KL divergence for bio encoder (default: 0.005)
                'kl_loss_2': 0.1,  # KL divergence for batch encoder (default: 0.1)
                'ortho_loss': 0.01,  # Orthogonality constraint (default: 0.01)
            }
        },
    })

    # BioBatchNet parameter scanning
    biobatchnet_scan_parameter_sets: Optional[List[Dict[str, Any]]] = None
    biobatchnet_scan_include_base: bool = True
    biobatchnet_run_leiden: bool = True

    # Scanpy neighbors computation
    n_neighbors: Optional[int] = None

    # Deprecated flat-style parameters (auto-migrated into biobatchnet_params)
    biobatchnet_data_type: Optional[str] = None
    biobatchnet_latent_dim: Optional[int] = None
    biobatchnet_epochs: Optional[int] = None
    biobatchnet_device: Optional[str] = None
    biobatchnet_kwargs: Optional[Dict[str, Any]] = None
    biobatchnet_use_raw: Optional[bool] = None

    def __post_init__(self):
        """Migrate deprecated flat BioBatchNet parameters into nested biobatchnet_params."""
        if self.biobatchnet_params is None:
            self.biobatchnet_params = {
                'data_type': 'imc',
                'latent_dim': 20,
                'epochs': 100,
                'device': None,
                'use_raw': True,
                'extra_params': {
                    'loss_weights': {
                        'recon_loss': 100.0,
                        'discriminator': 0.05,
                        'classifier': 1.0,
                        'kl_loss_1': 0.0005,
                        'kl_loss_2': 0.1,
                        'ortho_loss': 0.01,
                    }
                },
            }

        migrated = False
        if self.biobatchnet_data_type is not None:
            self.biobatchnet_params['data_type'] = self.biobatchnet_data_type
            migrated = True
        if self.biobatchnet_latent_dim is not None:
            self.biobatchnet_params['latent_dim'] = self.biobatchnet_latent_dim
            migrated = True
        if self.biobatchnet_epochs is not None:
            self.biobatchnet_params['epochs'] = self.biobatchnet_epochs
            migrated = True
        if self.biobatchnet_device is not None:
            self.biobatchnet_params['device'] = self.biobatchnet_device
            migrated = True
        if self.biobatchnet_kwargs is not None:
            self.biobatchnet_params['extra_params'] = self.biobatchnet_kwargs
            migrated = True
        if self.biobatchnet_use_raw is not None:
            self.biobatchnet_params['use_raw'] = self.biobatchnet_use_raw
            migrated = True

        if migrated:
            logging.warning(
                "Deprecated flat BioBatchNet parameters detected and migrated to 'biobatchnet_params'. "
                "Please update your config.yaml to use the nested format under biobatchnet.biobatchnet_params."
            )


@dataclass
class BasicProcessConfig(BioBatchNetConfig):
    """
    Legacy process config retained for backward compatibility.
    AnnData path management now belongs in GeneralConfig.
    """
    input_adata_path: str = 'anndata.h5ad'
    output_adata_path: str = 'anndata_processed.h5ad'

@dataclass
class VisualizationConfig:
    # Input data settings
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    population_columns: Optional[List[str]] = None  # Script override for population columns (None = use general.population_obs_all or auto-detect)
    metadata_columns: Optional[List[str]] = None  # Script override for metadata columns (None = use general.metadata_obs or auto-detect)
    groupby_obs: Optional[str] = None  # Script override for grouping column (None = use general.groupby_obs)
    groupby_obs_groups: Optional[List[str]] = None  # Script override for groups (None = use general pairwise/groups settings)
    
    # AI interpretation settings
    enable_ai: bool = True  # Enable AI-powered cluster interpretation
    tissue: str = "Unknown tissue"  # Tissue type for AI interpretation context
    repeat_ai_interpretation: bool = False  # Re-run AI interpretation even if labels already exist
    
    # Visualization module toggles - all default True
    create_umaps: bool = True  # Create UMAP plots for populations and markers
    create_matrix_plots: bool = True  # Create MatrixPlot summaries
    create_tissue_overlays: bool = True  # Create tissue population overlays
    create_population_analysis: bool = True  # Create population analysis across metadata
    create_backgating: bool = True  # Create backgating assessment
    create_color_legends: bool = True  # Generate color legends for categorical plots
    
    # Categorical visualization controls
    include_metadata_umaps: bool = True  # Include metadata columns in UMAP plots
    include_metadata_matrix_plots: bool = True  # Include metadata columns in MatrixPlots
    include_marker_umaps: bool = True  # Include marker expression UMAPs
    umap_plot_individual_highlights: bool = True  # For population columns, create one UMAP per category via utils.plot_umap_highlight_clusters
    max_categories: int = 50  # Maximum number of unique categories for population/metadata columns
    umap_marker_colormap: str = 'viridis'  # Colormap for marker expression UMAPs (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis')
    umap_marker_gallery_default_colorbar_label: str = 'Nimbus-Inference Score'  # Colorbar label for default-layer (adata.X) marker gallery
    
    # Backgating assessment settings
    backgating_cells_per_group: int = 50  # Number of cells to sample per population for backgating
    backgating_radius: int = 15  # Radius in pixels for cell thumbnail extraction
    backgating_output_folder: str = 'Backgating'  # Output folder for backgating results
    backgating_use_masks: bool = True  # Whether to use segmentation masks in backgating
    backgating_mask_folder: str = 'masks'  # Folder containing segmentation masks
    
    # Backgating intensity and marker settings
    backgating_minimum: float = 0.2  # Minimum intensity for backgating normalization
    backgating_max_quantile: str = 'i0.99'  # Maximum quantile method for intensity scaling
    backgating_number_top_markers: int = 2  # Number of top markers to use for RGB channels
    backgating_specify_blue: Optional[str] = 'DNA1'  # Marker to use for blue channel
    backgating_specify_red: Optional[str] = None  # Marker to use for red channel (None = auto-select)
    backgating_specify_green: Optional[str] = None  # Marker to use for green channel (None = auto-select)
    
    # Differential expression settings for backgating marker selection
    backgating_use_differential_expression: bool = True  # Use scanpy DE analysis for marker selection
    backgating_de_method: str = 'wilcoxon'  # Statistical method ('wilcoxon', 't-test', 'logreg')
    backgating_min_logfc_threshold: float = 0.2  # Minimum log fold change for quality filtering (0 to disable)
    backgating_max_pval_adj: float = 0.05  # Used for significance reporting, not filtering
    backgating_markers_exclude: Optional[List[str]] = field(default_factory=lambda: ['DNA1', 'DNA3'])  # Markers to exclude from DE analysis
    
    # Backgating execution mode control
    backgating_mode: str = 'full'  # 'full' (compute + run), 'save_markers' (compute only), 'load_markers' (load + run)
    
    # Population overlay visualization settings
    backgating_population_overlay_outline_width: int = 1  # Width of contour outlines in population overlay visualizations
    backgating_population_overlay_legend_fontsize: int = 24  # Font size for overlay legend labels
    backgating_population_overlay_crop_size: Optional[List[int]] = field(default_factory=lambda: [300, 300])  # Crop size [width, height] or None
    backgating_population_overlay_crop_origin: str = 'intelligent'  # Crop anchor: upper_left/right, lower_left/right, center, intelligent
    backgating_population_overlay_show_scale_bar: bool = True  # Whether to draw scale bar on overlays
    backgating_population_overlay_scale_bar_length: int = 50  # Scale bar length in pixels
    backgating_population_overlay_scale_bar_thickness: int = 3  # Scale bar thickness in pixels
    
    # MatrixPlot settings
    matrixplot_vmax: float = 0.5  # Maximum value for non-scaled matrix plots
    matrixplot_use_row_colors: bool = True  # Use plotting.matrixplot_with_row_colors when available for MatrixPlot generation
    
    # General visualization settings
    save_high_res: bool = True  # Save high-resolution figures (300 DPI)
    figure_format: str = 'png'  # Default figure format ('png', 'pdf', 'svg')

@dataclass
class CellCharterConfig:
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    qc_output_subdir: str = 'CellCharter_QC'

    # Spatial metadata
    sample_key: Optional[str] = None  # Optional override (None = use general.roi_obs)
    spatial_key: Optional[str] = None  # Optional override (None = use general.spatial_key)
    x_coord_col: Optional[str] = None  # Optional override (None = use general.x_coord_obs)
    y_coord_col: Optional[str] = None  # Optional override (None = use general.y_coord_obs)

    # Features
    use_rep: Optional[str] = None      # For non-TRVAE mode: adata.obsm key for neighborhood aggregation
    use_layer: Optional[str] = None    # For TRVAE or non-TRVAE mode: adata.layers key (None uses adata.X)
    scale_by_sample: bool = False       # In TRVAE mode: scale TRVAE input per sample; otherwise scale aggregation input
    scaled_rep_key: str = 'X_cellcharter_scaled'

    # TRVAE dimensionality reduction (default path, per CellCharter tutorial)
    use_trvae: bool = True
    trvae_latent_key: str = 'X_trVAE'
    trvae_condition_key: Optional[str] = 'dataset'
    trvae_use_sample_key_fallback: bool = True
    trvae_constant_condition_label: str = 'all'
    trvae_load_path: Optional[str] = None  # Optional pretrained model directory
    trvae_save_path: str = 'trvae_model'   # Relative paths are saved under QC/cellcharter.qc_output_subdir
    trvae_map_location: str = 'gpu'
    trvae_train: bool = True
    trvae_train_early_stopping: bool = False
    trvae_train_enable_progress_bar: bool = True
    trvae_train_max_epochs: Optional[int] = None
    trvae_hidden_layer_sizes: List[int] = field(default_factory=lambda: [128, 128])
    trvae_latent_dim: int = 10
    trvae_dr_rate: float = 0.05
    trvae_use_mmd: bool = True
    trvae_mmd_on: str = 'z'
    trvae_mmd_boundary: Optional[int] = None
    trvae_recon_loss: str = 'mse'
    trvae_beta: float = 1.0
    trvae_use_bn: bool = False
    trvae_use_ln: bool = True

    # Graph and neighborhood aggregation
    delaunay: bool = True
    remove_long_links: bool = True
    distance_percentile: float = 99.0
    n_layers: int = 3
    aggregations: str = 'mean'         # 'mean' or list-like string via overrides
    aggregated_rep_key: str = 'X_cellcharter'

    # Clustering
    n_clusters: int = 11
    random_state: int = 12345
    covariance_type: str = 'full'
    batch_size: Optional[int] = None
    trainer_accelerator: str = 'auto'
    trainer_devices: Optional[int] = None
    trainer_max_epochs: int = 100
    cluster_key: str = 'spatial_cluster'

    # Optional enrichment
    run_enrichment: bool = True
    enrichment_with_pvalues: bool = False
    enrichment_n_perms: int = 1000

    # Neighborhood enrichment (CellCharter graph enrichment)
    run_nhood_enrichment: bool = True
    nhood_connectivity_key: Optional[str] = None
    nhood_log_fold_change: bool = False
    nhood_only_inter: bool = True
    nhood_symmetric: bool = False
    nhood_with_pvalues: bool = False
    nhood_n_perms: int = 1000
    nhood_n_jobs: int = 1
    nhood_batch_size: int = 10
    nhood_observed_expected: bool = True
    save_nhood_enrichment_plot: bool = True
    nhood_plot_figsize: List[float] = field(default_factory=lambda: [6.0, 3.0])
    nhood_enrichment_significance: Optional[float] = None

    # Differential neighborhood enrichment by condition
    run_diff_nhood_enrichment: bool = False
    diff_nhood_condition_key: Optional[str] = None
    diff_nhood_condition_groups: Optional[List[str]] = None
    diff_nhood_connectivity_key: Optional[str] = None
    diff_nhood_log_fold_change: bool = False
    diff_nhood_only_inter: bool = True
    diff_nhood_symmetric: bool = False
    diff_nhood_with_pvalues: bool = False
    diff_nhood_library_key: Optional[str] = None
    diff_nhood_n_perms: int = 1000
    diff_nhood_n_jobs: Optional[int] = None
    diff_nhood_plot_ncols: int = 2
    save_diff_nhood_enrichment_plot: bool = True

    # Shape characterisation
    run_shape_characterisation: bool = False
    shape_component_key: str = 'component'
    shape_component_cluster_key: Optional[str] = None
    shape_connectivity_key: Optional[str] = None
    shape_min_cells: int = 250
    shape_min_hole_area_ratio: float = 0.1
    shape_alpha_start: int = 2000
    shape_compute_linearity: bool = True
    shape_linearity_key: str = 'linearity'
    shape_linearity_height: int = 1000
    shape_linearity_min_ratio: float = 0.05
    shape_compute_curl: bool = True
    shape_curl_key: str = 'curl'
    shape_plot_metrics: bool = True
    shape_metrics_condition_key: Optional[str] = None
    shape_metrics_condition_groups: Optional[List[str]] = None
    shape_metrics_cluster_key: Optional[str] = None
    shape_metrics_cluster_groups: Optional[List[str]] = None
    shape_metrics_ncols: int = 2

    # QC plotting
    save_spatial_plots: bool = True
    max_rois_for_plots: int = 12
    point_size: float = 2.0
    save_enrichment_heatmap: bool = True

@dataclass
class PairwiseSpatialConfig:
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_subdir: str = 'Pairwise_Spatial'
    reload_saved_results: bool = True  # Reuse saved raw analysis outputs when present (useful for plot-only reruns)

    # Core metadata keys
    population_obs: Optional[str] = None  # Optional override (None = use general.population_obs_primary or legacy 'population')
    groupby_obs: Optional[str] = None  # Optional override (None = use general.groupby_obs)
    roi_obs: Optional[str] = None  # Optional override (None = use general.roi_obs)
    x_coord_obs: Optional[str] = None  # Optional override (None = use general.x_coord_obs)
    y_coord_obs: Optional[str] = None  # Optional override (None = use general.y_coord_obs)
    master_index_obs: Optional[str] = None  # Optional override (None = use general.master_index_obs)
    source_population_obs: Optional[str] = None  # If None: uses population_obs

    # Metadata export controls
    include_all_obs_metadata: bool = True
    metadata_obs_columns: List[str] = field(default_factory=list)

    # Squidpy neighborhood enrichment
    run_squidpy_interactions: bool = True
    squidpy_subregion_obs: Optional[str] = None  # If None: uses roi_obs
    squidpy_subregion_suffix: str = ''
    squidpy_radius_min_um: int = 0
    squidpy_radius_max_um: int = 20
    squidpy_n_permutations: int = 1000

    # Distance bootstrap analysis
    run_distance_bootstrap: bool = True
    distance_populations: Optional[List[str]] = None
    distance_roi_ids: Optional[List[str]] = None
    distance_n_bootstraps: int = 1000
    distance_n_jobs: int = -1
    distance_ddof: int = 1

    # Pair-correlation function (PCF)
    run_pcf: bool = True
    pcf_target_distance_um: float = 20.0
    pcf_max_radius_um: float = 100.0
    pcf_radius_step_um: float = 10.0
    pcf_num_bootstrap: int = 1000
    pcf_cluster_column: str = 'cluster'
    pcf_samples: Optional[List[str]] = None

    # Optional source-target population pairs
    # Supports:
    # 1) Direct mapping: {source_pop: [target_pop, ...]}
    # 2) Nested by obs key: {population_obs: {source_pop: [target_pop, ...]}}
    # Target tokens:
    # - "ALL": all populations in population_obs
    # - "ALL_OTHERS": all populations except source_pop
    # - "MATCH_x": populations containing substring "x" (case-insensitive)
    # - "NOT_x": populations not containing substring "x" (case-insensitive)
    population_pairs: Dict[str, Any] = field(default_factory=dict)

    # Plotting
    make_matrix_plots: bool = True
    make_pair_barplots: bool = True
    heatmap_use_clustermap: bool = True
    heatmap_row_cluster: bool = True
    heatmap_col_cluster: bool = True
    heatmap_figsize: List[float] = field(default_factory=lambda: [5.0, 5.0])
    heatmap_percentile: float = 95.0
    pairwise_matrices_share_vmax_vmin: bool = False  # If True, use limits from each metric's all-data matrix for all group matrix plots
    heatmap_cmap_interactions: str = 'coolwarm'
    heatmap_cmap_distance: str = 'coolwarm'
    heatmap_cmap_pcf: str = 'coolwarm'
    heatmap_cmap_counts: str = 'viridis'
    barplot_figsize: List[float] = field(default_factory=lambda: [3.0, 3.0])
    barplot_add_points: bool = True
    figure_extension: str = '.png'
    figure_dpi: int = 300

@dataclass
class SubclusteringConfig:
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_subdir: str = 'subclustering'

    # Template/remap files
    settings_filename: str = 'sublustering_settings.csv'  # Intentionally matches existing notebook naming
    marker_list_filename: str = 'marker_list.csv'
    remap_filename: str = 'subcluster_to_final_population.csv'
    master_index_mapping_filename: str = 'master_index_to_final_population.csv'

    # Subclustering defaults
    base_label_key: Optional[str] = None  # Optional override (None = use general.population_obs_primary or legacy 'population')
    default_resolution: float = 0.3
    default_marker_list: str = 'all'  # Resolved as marker column 'markers_all'
    use_rep: Optional[str] = 'X_biobatchnet'

    # Plotting and QC
    compute_umap_if_missing: bool = True
    umap_dot_size: float = 2.0
    matrixplot_vmax: float = 0.5
    save_individual_umaps: bool = True
    figure_extension: str = '.png'
    figure_dpi: int = 300

    # Final remap integration
    final_label_key: str = 'population_final'
    master_index_obs: Optional[str] = None  # Optional override (None = use general.master_index_obs)
    apply_remap_only_if_modified: bool = True

@dataclass
class LoggingConfig:
    log_file: str = 'pipeline.log'
    level: str = 'INFO'
    to_console: bool = True
    console_only: bool = False  # Only log to console, not file (useful for SLURM jobs)
    prevent_duplicate_console: bool = True  # Prevent double console output
    use_custom_format: bool = True  # Use custom format vs basicConfig default


DEFAULT_CONFIG_CLASSES = {
    'general': GeneralConfig,
    'preprocess': PreprocessConfig,
    'denoising': DenoisingConfig,
    'createmasks': CreateMasksConfig,
    'segmentation': SegmentationConfig,
    'nimbus': NimbusConfig,
    'biobatchnet': BioBatchNetConfig,
    'process': BasicProcessConfig,
    'visualization': VisualizationConfig,
    'cellcharter': CellCharterConfig,
    'pairwise_spatial': PairwiseSpatialConfig,
    'subclustering': SubclusteringConfig,
    'logging': LoggingConfig,
}

def generate_default_config_dict() -> Dict[str, Any]:
    """
    Generate a dictionary of default configuration values from the dataclasses.
    """
    defaults = {}
    for section, cls in DEFAULT_CONFIG_CLASSES.items():
        defaults[section] = asdict(cls())
    return defaults

def filter_config_for_dataclass(config_dict: Dict[str, Any], dataclass_type) -> Dict[str, Any]:
    """
    Filter a config dictionary to only include keys that are valid for the given dataclass.
    Log warnings for any unexpected keys.
    
    Parameters:
    config_dict: Dictionary containing configuration values
    dataclass_type: The dataclass type to filter for
    
    Returns:
    Filtered dictionary with only valid keys for the dataclass
    """
    # Get the field names from the dataclass
    if hasattr(dataclass_type, '__dataclass_fields__'):
        valid_fields = set(dataclass_type.__dataclass_fields__.keys())
    else:
        # Fallback: create a temporary instance and get its attributes
        temp_instance = dataclass_type()
        valid_fields = set(temp_instance.__dict__.keys())
    
    filtered_config = {}
    dataclass_name = dataclass_type.__name__
    
    for key, value in config_dict.items():
        if key in valid_fields:
            filtered_config[key] = value
        else:
            logging.warning(f"Ignoring unrecognized config key '{key}' = {value} in {dataclass_name} configuration section. Please check if this key belongs in a different config section.")
    
    return filtered_config

def deep_merge_defaults(config: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
    """
    Recursively merge default values into config. If a key from defaults is not present in config,
    it is added. If a key is present but is a dictionary, we recurse.

    Returns True if changes were made to the config, False otherwise.
    """
    changed = False
    for key, default_value in defaults.items():
        if key not in config:
            # Key missing, add it
            config[key] = default_value
            changed = True
        else:
            # If both are dicts, recurse
            if isinstance(default_value, dict) and isinstance(config[key], dict):
                if deep_merge_defaults(config[key], default_value):
                    changed = True
            # If default_value is not a dict but config[key] is missing keys, this case is handled above
            # If config[key] is already set and not a dict, we do not overwrite existing keys
            # because we assume user config is correct. If we want to always overwrite with defaults
            # if user config is missing fields, we rely on the dictionary recursion above.
    return changed

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file, ensure all fields from the dataclasses are present.
    If the file does not exist, create it with all default values.
    If fields are missing, add them and update the file.

    Returns the fully populated config dictionary.
    """
    defaults = generate_default_config_dict()

    if not os.path.isfile(config_file):
        # File not found, create it with defaults
        with open(config_file, 'w') as f:
            yaml.safe_dump(defaults, f, default_flow_style=False)
        logging.info(f'Configuration file "{config_file}" not found. Created and saved with default values.')
        return defaults

    # If file exists, load it
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Merge defaults into config if any keys missing
    changed = deep_merge_defaults(config, defaults)

    # Save if any changes were made
    if changed:
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logging.info(f'Configuration file "{config_file}" updated with default values for missing keys.')

    return config

def setup_logging(logging_config, pipeline_stage):
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO)
    log_file = logging_config.get('log_file', 'pipeline.log')
    to_console = logging_config.get('to_console', True)
    console_only = logging_config.get('console_only', False)
    prevent_duplicate = logging_config.get('prevent_duplicate_console', True)
    use_custom_format = logging_config.get('use_custom_format', True)
    
    # Clear any existing handlers to prevent accumulation
    root_logger = logging.getLogger()
    if prevent_duplicate:
        root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        f'%(asctime)s [%(levelname)s] [{pipeline_stage}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ) if use_custom_format else logging.Formatter()
    
    if not console_only:
        # Add file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    if to_console:
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate messages if requested
    if prevent_duplicate:
        root_logger.propagate = False


def _normalize_stage_run_mode(mode: Optional[str]) -> str:
    mode_text = str(mode or "intelligent").strip().lower()
    if mode_text not in {"repeat", "skip", "intelligent"}:
        logging.warning(
            "Unknown general.anndata_stage_run_mode='%s'. Falling back to 'intelligent'.",
            mode,
        )
        return "intelligent"
    return mode_text


def _collect_slurm_context_from_env() -> Dict[str, str]:
    """
    Collect SLURM job metadata from environment variables.
    Prefer IMC_* aliases set by job scripts, with SLURM_* as fallback.
    """
    job_id = os.getenv("IMC_SLURM_JOB_ID") or os.getenv("SLURM_JOB_ID")
    job_name = os.getenv("IMC_SLURM_JOB_NAME") or os.getenv("SLURM_JOB_NAME")

    slurm: Dict[str, str] = {}
    if job_id is not None and str(job_id).strip():
        slurm["job_id"] = str(job_id).strip()
    if job_name is not None and str(job_name).strip():
        slurm["job_name"] = str(job_name).strip()
    return slurm


def _sanitize_uns_key(key: Any) -> str:
    """Sanitize dictionary keys for safe storage in AnnData .uns/HDF5."""
    key_text = str(key)
    if "/" in key_text:
        key_text = key_text.replace("/", "__slash__")
    return key_text


def _is_null_like_for_uns(value: Any) -> bool:
    """Return True for null-like values that often break cross-version AnnData I/O."""
    if value is None:
        return True

    # pandas sentinel nulls
    try:
        import pandas as pd  # local import to avoid hard dependency at module import time

        if value is pd.NA or value is pd.NaT:
            return True
    except Exception:
        pass

    # numpy masked sentinel
    try:
        import numpy as np

        if value is np.ma.masked:
            return True
    except Exception:
        pass

    return False


def _contains_null_like_object_array(value: Any) -> bool:
    """Detect object-dtype arrays containing null-like values."""
    try:
        import numpy as np

        if not isinstance(value, np.ndarray) or value.dtype != object:
            return False
        for item in value.ravel():
            if _is_null_like_for_uns(item):
                return True
    except Exception:
        return False
    return False


def _sanitize_uns_payload(
    value: Any,
    *,
    max_depth: int = 50,
    _depth: int = 0,
) -> Tuple[Any, int]:
    """
    Recursively sanitize payloads for adata.uns storage.
    Returns (cleaned_value, removed_item_count).
    """
    if _depth > max_depth:
        return value, 0

    if _is_null_like_for_uns(value):
        return None, 1

    if isinstance(value, Path):
        return str(value), 0

    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        removed = 0
        for key, item in value.items():
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out[_sanitize_uns_key(key)] = cleaned
        return out, removed

    if isinstance(value, list):
        out_list: List[Any] = []
        removed = 0
        for item in value:
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out_list.append(cleaned)
        return out_list, removed

    if isinstance(value, tuple):
        out_tuple: List[Any] = []
        removed = 0
        for item in value:
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out_tuple.append(cleaned)
        return out_tuple, removed

    if isinstance(value, set):
        out_set_as_list: List[Any] = []
        removed = 0
        for item in value:
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out_set_as_list.append(cleaned)
        return out_set_as_list, removed

    # Object arrays containing null-like values can trigger 'null' encoding on write.
    if _contains_null_like_object_array(value):
        return None, 1

    # Handle NumPy scalars without importing numpy at module import time.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item(), 0
        except Exception:
            pass

    return value, 0


def _sanitize_anndata_uns_inplace(adata: Any, *, max_depth: int = 50) -> int:
    """
    Clean adata.uns in-place to improve backward compatibility across anndata versions.
    Removes null-like values and unsupported nested payloads.
    """
    uns_obj = getattr(adata, "uns", None)
    if not isinstance(uns_obj, MutableMapping):
        return 0

    cleaned_uns, removed = _sanitize_uns_payload(dict(uns_obj), max_depth=max_depth)
    if not isinstance(cleaned_uns, dict):
        cleaned_uns = {}

    try:
        uns_obj.clear()
        uns_obj.update(cleaned_uns)
    except Exception:
        adata.uns = cleaned_uns

    return int(removed)


def _h5_attr_to_text(value: Any) -> str:
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", "ignore")
        if hasattr(value, "item") and callable(getattr(value, "item")):
            item_value = value.item()
            if isinstance(item_value, bytes):
                return item_value.decode("utf-8", "ignore")
            return str(item_value)
        return str(value)
    except Exception:
        return str(value)


def _is_h5_node_null_encoded(node: Any) -> bool:
    for attr_key in ("encoding-type", "encoding_type"):
        try:
            if attr_key in node.attrs:
                attr_val = _h5_attr_to_text(node.attrs[attr_key]).strip().lower()
                if attr_val == "null":
                    return True
        except Exception:
            continue
    return False


def _collect_null_encoded_h5_paths(group: Any, prefix: str) -> List[str]:
    paths: List[str] = []
    try:
        import h5py
    except Exception:
        return paths

    for name in list(group.keys()):
        child = group[name]
        child_path = f"{prefix}/{name}"
        if _is_h5_node_null_encoded(child):
            paths.append(child_path)
            # Whole node will be deleted, so skip recursion into this subtree.
            continue
        if isinstance(child, h5py.Group):
            paths.extend(_collect_null_encoded_h5_paths(child, child_path))
    return paths


def _remove_null_encoded_uns_entries_in_h5ad(anndata_path: Path) -> List[str]:
    """
    In-place repair for files containing 'null' encoded datasets under /uns.
    Returns a list of removed HDF5 paths.
    """
    import h5py

    removed_paths: List[str] = []
    with h5py.File(anndata_path, "r+") as handle:
        if "uns" not in handle:
            return removed_paths

        paths_to_remove = _collect_null_encoded_h5_paths(handle["uns"], "/uns")
        for path in sorted(set(paths_to_remove), key=lambda p: p.count("/"), reverse=True):
            parent_path, leaf = path.rsplit("/", 1)
            parent = handle[parent_path.lstrip("/")] if parent_path and parent_path != "/" else handle
            if leaf in parent:
                del parent[leaf]
                removed_paths.append(path)

    return removed_paths


def _looks_like_null_encoding_read_error(exc: Exception) -> bool:
    msg = str(exc)
    if "encoding_type='null'" in msg or 'encoding_type="null"' in msg:
        return True
    if "No read method registered for IOSpec" in msg and "null" in msg:
        return True
    return False


def _sanitize_for_uns(value: Any) -> Any:
    """Recursively sanitize values for safe storage in adata.uns and drop None entries."""
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _sanitize_for_uns(item)
            if cleaned is None:
                continue
            out[_sanitize_uns_key(key)] = cleaned
        return out

    if isinstance(value, (list, tuple, set)):
        out_list: List[Any] = []
        iterable = value
        if isinstance(value, set):
            # Keep set handling deterministic for stable stage snapshot comparison.
            iterable = sorted(value, key=lambda x: str(x))
        for item in iterable:
            cleaned = _sanitize_for_uns(item)
            if cleaned is None:
                continue
            out_list.append(cleaned)
        return out_list

    # Handle array-like payloads (e.g., numpy arrays, pandas index/series) by
    # converting to python-native lists recursively.
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _sanitize_for_uns(value.tolist())
        except Exception:
            pass

    # Handle NumPy scalars without importing numpy at module import time.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass

    return value


def build_uns_config_snapshot(config_obj: Any) -> Dict[str, Any]:
    """
    Build a sanitized config snapshot suitable for adata.uns.
    All None/null values are removed recursively.
    """
    cleaned = _sanitize_for_uns(config_obj)
    if cleaned is None:
        return {}
    if isinstance(cleaned, dict):
        return cleaned
    return {"value": cleaned}


def _is_nan_like(value: Any) -> bool:
    try:
        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def _safe_snapshot_equal(left: Any, right: Any) -> bool:
    """
    Robust deep equality for stage snapshots.
    Handles nested dict/list payloads and treats NaN values as equal.
    """
    if _is_nan_like(left) and _is_nan_like(right):
        return True

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_safe_snapshot_equal(left[k], right[k]) for k in left.keys())

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_safe_snapshot_equal(lv, rv) for lv, rv in zip(left, right))

    # Fallback to normalized list representation for remaining array-like objects.
    if hasattr(left, "tolist") and callable(getattr(left, "tolist")):
        try:
            left = left.tolist()
        except Exception:
            pass
    if hasattr(right, "tolist") and callable(getattr(right, "tolist")):
        try:
            right = right.tolist()
        except Exception:
            pass

    try:
        return left == right
    except Exception:
        return str(left) == str(right)


def coalesce_config_text(*values: Any, default: Optional[str] = None) -> Optional[str]:
    """
    Return the first non-empty string-like value from a list of candidates.
    """
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def coalesce_config_list(*values: Any, default: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Return the first non-null list-like value from a list of candidates as a list of strings.
    """
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        return [str(value)]
    return default


def resolve_anndata_path(
    general_config: GeneralConfig,
    override_path: Optional[str] = None,
) -> Path:
    target = override_path if override_path else general_config.anndata_path
    return Path(str(target))


def _get_stage_log_container(adata: Any, uns_key: str) -> Dict[str, Any]:
    container = adata.uns.get(uns_key)
    if not isinstance(container, dict):
        container = {}
    if not isinstance(container.get("stage_order"), list):
        container["stage_order"] = []
    run_log_raw = container.get("run_log")
    migrated: Dict[str, Any] = {}
    if isinstance(run_log_raw, dict):
        for idx, item in enumerate(run_log_raw.values(), start=1):
            run_key = f"run_{idx:06d}"
            cleaned = _sanitize_for_uns(item)
            if isinstance(cleaned, dict):
                migrated[run_key] = cleaned
            elif cleaned is not None:
                migrated[run_key] = {"value": cleaned}
    elif isinstance(run_log_raw, list):
        # Migrate legacy list-based run logs (can fail HDF5 serialization) to dict form.
        for idx, item in enumerate(run_log_raw, start=1):
            run_key = f"run_{idx:06d}"
            cleaned = _sanitize_for_uns(item)
            if isinstance(cleaned, dict):
                migrated[run_key] = cleaned
            elif cleaned is not None:
                migrated[run_key] = {"value": cleaned}
    container["run_log"] = migrated
    if not isinstance(container.get("stages"), dict):
        container["stages"] = {}
    adata.uns[uns_key] = container
    return container


def get_stage_run_record(
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
) -> Optional[Dict[str, Any]]:
    uns_key = str(general_config.anndata_uns_log_key)
    container = _get_stage_log_container(adata, uns_key)
    record = container.get("stages", {}).get(str(stage_name))
    if isinstance(record, dict):
        return record
    return None


def should_run_stage(
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
) -> Tuple[bool, str]:
    """
    Decide whether a stage should run based on general.anndata_stage_run_mode.
    """
    mode = _normalize_stage_run_mode(getattr(general_config, "anndata_stage_run_mode", "intelligent"))
    record = get_stage_run_record(adata, general_config, stage_name)
    if record is None:
        return True, f"Stage '{stage_name}' has no previous run record."

    if mode == "repeat":
        return True, "general.anndata_stage_run_mode=repeat."

    if mode == "skip":
        return False, f"Stage '{stage_name}' already recorded and mode=skip."

    current_snapshot = build_uns_config_snapshot(stage_config)
    previous_snapshot = build_uns_config_snapshot(record.get("config", {}))
    if _safe_snapshot_equal(current_snapshot, previous_snapshot):
        return (
            False,
            f"Stage '{stage_name}' already recorded with matching config and mode=intelligent.",
        )
    return (
        True,
        f"Stage '{stage_name}' config changed since last run; mode=intelligent so it will run again.",
    )


def load_pipeline_anndata(
    *,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
    override_path: Optional[str] = None,
    allow_missing: bool = False,
) -> Tuple[Optional[Any], Path, bool, str]:
    """
    Standardized AnnData loader with stage-run decision logic.

    Returns
    -------
    tuple
        (adata_or_none, resolved_path, skip_stage, decision_message)
    """
    import anndata as ad

    anndata_path = resolve_anndata_path(general_config, override_path=override_path)
    if not anndata_path.exists():
        if allow_missing:
            msg = f"AnnData not found at {anndata_path}; proceeding because allow_missing=True."
            logging.info(msg)
            return None, anndata_path, False, msg
        raise FileNotFoundError(f"AnnData file not found: {anndata_path}")

    logging.info("Loading AnnData from %s", anndata_path)
    try:
        adata = ad.read_h5ad(anndata_path)
    except Exception as exc:
        if not _looks_like_null_encoding_read_error(exc):
            raise

        logging.warning(
            "AnnData read failed due null-encoded payloads (likely from newer anndata/scanpy). "
            "Attempting in-place repair of /uns in %s.",
            anndata_path,
        )
        removed_paths = _remove_null_encoded_uns_entries_in_h5ad(anndata_path)
        if not removed_paths:
            raise

        preview = ", ".join(removed_paths[:5])
        if len(removed_paths) > 5:
            preview += ", ..."
        logging.warning(
            "Removed %d null-encoded /uns entries from %s: %s",
            len(removed_paths),
            anndata_path,
            preview,
        )
        adata = ad.read_h5ad(anndata_path)

    removed_from_uns = _sanitize_anndata_uns_inplace(adata)
    if removed_from_uns > 0:
        logging.warning(
            "Removed %d null-like entries from adata.uns after load for compatibility.",
            removed_from_uns,
        )
    should_run, reason = should_run_stage(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=stage_config,
    )
    skip_stage = not should_run
    logging.info("Stage decision for '%s': %s", stage_name, reason)
    return adata, anndata_path, skip_stage, reason


def record_stage_run_in_uns(
    *,
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
    extra_details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record a stage run in adata.uns using the configured pipeline log key.
    """
    uns_key = str(general_config.anndata_uns_log_key)
    container = _get_stage_log_container(adata, uns_key)
    stage_name = str(stage_name)
    timestamp = datetime.now(timezone.utc).isoformat()

    stage_snapshot = build_uns_config_snapshot(stage_config)
    detail_snapshot = build_uns_config_snapshot(extra_details) if extra_details is not None else {}
    slurm_snapshot = build_uns_config_snapshot(_collect_slurm_context_from_env())

    container["stage_order"].append(stage_name)
    run_event: Dict[str, Any] = {"stage": stage_name, "run_utc": timestamp}
    if stage_snapshot:
        run_event["config"] = stage_snapshot
    if detail_snapshot:
        run_event["details"] = detail_snapshot
    if slurm_snapshot:
        run_event["slurm"] = slurm_snapshot
    run_log = container.get("run_log")
    if not isinstance(run_log, dict):
        run_log = {}
        container["run_log"] = run_log
    run_idx = len(run_log) + 1
    run_key = f"run_{run_idx:06d}"
    run_log[run_key] = run_event

    entry: Dict[str, Any] = {"last_run_utc": timestamp}
    if stage_snapshot:
        entry["config"] = stage_snapshot
    if detail_snapshot:
        entry["details"] = detail_snapshot
    if slurm_snapshot:
        entry["slurm"] = slurm_snapshot
    container["stages"][stage_name] = entry
    adata.uns[uns_key] = container


def save_pipeline_anndata(
    *,
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
    override_path: Optional[str] = None,
    extra_details: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Record stage metadata in adata.uns and save AnnData to the canonical path.
    """
    target_path = resolve_anndata_path(general_config, override_path=override_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    record_stage_run_in_uns(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=stage_config,
        extra_details=extra_details,
    )

    removed_from_uns = _sanitize_anndata_uns_inplace(adata)
    if removed_from_uns > 0:
        logging.warning(
            "Removed %d null-like entries from adata.uns before save for backward compatibility.",
            removed_from_uns,
        )

    try:
        adata.write_h5ad(target_path)
    except Exception as exc:
        logging.warning(
            "Initial AnnData write failed (%s). Retrying after additional uns sanitization.",
            exc,
        )
        removed_retry = _sanitize_anndata_uns_inplace(adata, max_depth=100)
        if removed_retry > 0:
            logging.warning(
                "Removed %d additional null-like uns entries before write retry.",
                removed_retry,
            )
        adata.write_h5ad(target_path)
    logging.info("Saved AnnData to %s", target_path)
    return target_path


def get_filename(path: Path, name: str) -> str:
    """
    Retrieves a filename from the specified directory that contains a specific substring.
    """
    files = [x.name for x in path.iterdir() if name in x.name]

    if len(files) == 0:
        raise FileNotFoundError(f"No file {name} found in {path}")
    elif len(files) > 1:
        raise ValueError(f"More than one file or image in {str(path)} matches {name}")
    else:
        return files[0]

def update_config_file(config_file: str, updates: Dict[str, Any]) -> None:
    """
    Update the YAML configuration file with the given updates.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            config = {}

    config.update(updates)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    logging.info(f'Configuration file "{config_file}" updated with: {updates}')

def apply_override(config: Dict, key_path: str, value: str) -> None:
    keys = key_path.split('.')
    d = config
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]

    if ',' in value:
        value = value.split(',')
    d[keys[-1]] = value

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the pipeline with overrides.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config (default: config.yaml)')
    parser.add_argument('--override', action='append', help='Overrides in key=value format. Use dot-notation for keys.')
    return parser.parse_args()

def process_config_with_overrides():
    args = parse_arguments()

    # Load config with default merging
    config = load_config(args.config)

    # Apply overrides
    if args.override:
        for ov in args.override:
            if '=' not in ov:
                logging.warning(f"Invalid override (no '=' found): {ov}")
                continue
            key_path, value = ov.split('=', 1)
            apply_override(config, key_path.strip(), value.strip())

        # If overrides potentially added new keys not in defaults, we could re-run
        # deep_merge_defaults if desired. But since we only wanted to ensure old configs
        # get updated, this may not be necessary.

        # Save config after overrides?
        with open(args.config, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logging.info(f'Configuration file "{args.config}" updated with overrides.')

    return config

def create_config(config_class, **overrides):
    """
    Create a configuration object with defaults and optional overrides.
    
    This is useful for programmatically creating config objects when using
    individual functions from the pipeline outside of the main scripts.
    
    Parameters
    ----------
    config_class : type
        The configuration dataclass to instantiate (e.g., GeneralConfig, VisualizationConfig)
    **overrides : dict
        Keyword arguments to override default values
    
    Returns
    -------
    config object
        Instance of the specified config class with applied overrides
    
    Examples
    --------
    >>> # Create a GeneralConfig with custom masks folder
    >>> general_cfg = create_config(GeneralConfig, masks_folder='custom_masks')
    
    >>> # Create a VisualizationConfig with specific settings
    >>> viz_cfg = create_config(
    ...     VisualizationConfig,
    ...     create_umaps=True,
    ...     create_tissue_overlays=True,
    ...     save_high_res=False
    ... )
    """
    config = config_class()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logging.warning(f"Unknown config field '{key}' for {config_class.__name__}. Ignoring.")
    return config


def cleanstring(data: Any) -> str:
    """
    Helper function that returns a clean string with underscores replacing non-word characters.

    Parameters
    ----------
    data : Any
        Input data to be cleaned.

    Returns
    -------
    str
        Cleaned string with underscores instead of special characters.
    """
    import re
    data = str(data)
    # Replace sequences of non-word characters (except underscores) with single underscores
    data = re.sub(r'[^\w]+', '_', data)
    # Remove leading/trailing underscores and collapse multiple underscores
    data = re.sub(r'^_+|_+$', '', data)
    data = re.sub(r'_+', '_', data)
    return data

