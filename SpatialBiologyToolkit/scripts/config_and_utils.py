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
    network_size: str = "normal"
    # Parameters for QC images
    run_QC: bool = True
    colourmap: str = "jet"
    dpi: int = 100
    qc_image_dir: str = 'denoising'
    skip_already_denoised: bool = True

@dataclass
class CreateMasksConfig:
    specific_rois: Optional[List[str]] = None
    dna_image_name: str = 'DNA1'
    output_folder_name: str = 'preprocessed_dna'  # For DNA preprocessing output
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

    # CellPose-SAM mode toggle and settings - uses output_folder_name for input, GeneralConfig.masks_folder for output
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
    anndata_save_path: str = 'anndata.h5ad'
    create_roi_cell_tables: bool = True
    create_master_cell_table: bool = True
    create_anndata: bool = True
    allow_missing_channels: bool = False  # If True, fill missing channels with NaN; if False, only include channels present in all ROIs

@dataclass
class BasicProcessConfig:
    input_adata_path: str = 'anndata.h5ad'
    output_adata_path: str = 'anndata_processed.h5ad'
    batch_correction_method: Optional[str] = None
    batch_correction_obs: Optional[str] = None
    n_for_pca: Optional[int] = None
    leiden_resolutions_list: List[float] = field(default_factory=lambda: [0.3, 1.0])
    umap_min_dist: float = 0.5

@dataclass
class LoggingConfig:
    log_file: str = 'pipeline.log'
    level: str = 'INFO'
    to_console: bool = True


DEFAULT_CONFIG_CLASSES = {
    'general': GeneralConfig,
    'preprocess': PreprocessConfig,
    'denoising': DenoisingConfig,
    'createmasks': CreateMasksConfig,
    'segmentation': SegmentationConfig,
    'process': BasicProcessConfig,
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

    if changed:
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logging.info(f'Configuration file "{config_file}" updated with default values for missing keys.')

    return config

def setup_logging(logging_config, pipeline_stage):
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO)
    log_file = logging_config.get('log_file', 'pipeline.log')

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=log_level,
        format=f'%(asctime)s [%(levelname)s] [{pipeline_stage}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if logging_config.get('to_console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(
            f'%(asctime)s [%(levelname)s] [{pipeline_stage}] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logging.getLogger('').addHandler(console_handler)

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