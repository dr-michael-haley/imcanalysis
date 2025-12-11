import os
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

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
    overwrite_existing_outputs: bool = True
    extract_classic_intensities: bool = True  # Extract classic mean intensities over masks
    extract_expansion_intensities: bool = True  # Extract mean intensities from expanded masks
    expansion_pixels: int = 2  # Number of pixels to expand masks for expansion intensities
    expansion_jobs: int = 1  # Number of parallel jobs for expansion extraction (1=sequential, -1=all CPUs)

@dataclass
class BasicProcessConfig:
    input_adata_path: str = 'anndata.h5ad'
    output_adata_path: str = 'anndata_processed.h5ad'
    batch_correction_method: Optional[str] = None
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
        'use_raw': True,
        'extra_params': {
            'loss_weights': {
                'recon_loss': 100.0,
                'discriminator': 0.05, # Batch mixing (default: 0.3 — lower = more mixing)
                'classifier': 1.0, # Batch retention (default: 1)
                'kl_loss_1': 0.0005, # KL divergence for bio encoder (default: 0.005)
                'kl_loss_2': 0.1, # KL divergence for batch encoder (default: 0.1)
                'ortho_loss': 0.01, # Orthogonality constraint (default: 0.01)
            }
        },
    })
    
    # BioBatchNet parameter scanning
    biobatchnet_scan_parameter_sets: Optional[List[Dict[str, Any]]] = None  # Parameter overrides for scanning
    biobatchnet_scan_include_base: bool = True  # Run the base configuration alongside scans by default
    biobatchnet_run_leiden: bool = True  # Run Leiden clustering after BioBatchNet correction
    
    # Scanpy neighbors computation
    n_neighbors: Optional[int] = None  # Number of neighbors for scanpy neighbors computation (None uses scanpy default)
    
    # DEPRECATED: Old flat-style parameters (kept for backward compatibility, will be migrated to biobatchnet_params)
    biobatchnet_data_type: Optional[str] = None
    biobatchnet_latent_dim: Optional[int] = None
    biobatchnet_epochs: Optional[int] = None
    biobatchnet_device: Optional[str] = None
    biobatchnet_kwargs: Optional[Dict[str, Any]] = None
    biobatchnet_use_raw: Optional[bool] = None
    
    def __post_init__(self):
        """Migrate old flat-style parameters to nested biobatchnet_params format."""
        # If biobatchnet_params is None, initialize with defaults
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
        
        # Migrate old flat parameters if they are set
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
                "Please update your config.yaml to use the nested format under process.biobatchnet_params."
            )

@dataclass
class VisualizationConfig:
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
    max_categories: int = 50  # Maximum number of unique categories for population/metadata columns
    umap_marker_colormap: str = 'viridis'  # Colormap for marker expression UMAPs (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis')
    
    # Backgating assessment settings
    backgating_cells_per_group: int = 50  # Number of cells to sample per population for backgating
    backgating_radius: int = 15  # Radius in pixels for cell thumbnail extraction
    backgating_output_folder: str = 'Backgating'  # Output folder for backgating results
    backgating_use_masks: bool = True  # Whether to use segmentation masks in backgating
    backgating_mask_folder: str = 'masks'  # Folder containing segmentation masks
    
    # Backgating intensity and marker settings
    backgating_minimum: float = 0.3  # Minimum intensity for backgating normalization
    backgating_max_quantile: str = 'i0.96'  # Maximum quantile method for intensity scaling
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
    
    # MatrixPlot settings
    matrixplot_vmax: float = 0.5  # Maximum value for non-scaled matrix plots
    
    # General visualization settings
    save_high_res: bool = True  # Save high-resolution figures (300 DPI)
    figure_format: str = 'png'  # Default figure format ('png', 'pdf', 'svg')

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
    'process': BasicProcessConfig,
    'visualization': VisualizationConfig,
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


