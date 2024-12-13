import os
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class GeneralConfig:
    mcd_files_folder: str = 'MCD_files'
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
    raw_directory: str = 'tiffs'
    metadata_directory: str = 'metadata'
    processed_output_dir: str = 'processed'
    method: str = 'deep_snf'  # Options: 'deep_snf', 'dimr'
    channels: List[str] = field(default_factory=list)
    # Parameters for both methods
    n_neighbours: int = 4
    n_iter: int = 3
    window_size: int = 3
    # Parameters specific to 'deep_snf' method
    patch_step_size: int = 100
    train_epochs: int = 75
    train_initial_lr: float = 0.001
    train_batch_size: int = 200
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


@dataclass
class CreateMasksConfig:
    specific_rois: Optional[List[str]] = None
    dna_image_name: str = 'DNA1'
    cellpose_cell_diameter: float = 10.0
    upscale_ratio: float = 1.7
    expand_masks: int = 1
    perform_qc: bool = True
    min_cell_area: Optional[int] = 15
    max_cell_area: Optional[int] = 200

@dataclass
class SegmentationConfig:
    celltable_output: str = 'celltable.csv'
    marker_normalisation: List[str] = field(default_factory=lambda: ["q0.999"])
    store_raw_marker_data: bool = False
    remove_channels_list: List[str] = field(default_factory=lambda: ['DNA1', 'DNA3'])
    anndata_save_path: str = 'anndata.h5ad'
    extra_mask_analyses: bool = True

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


def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    """
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
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

    import yaml
    from typing import Dict, Any

def update_config_file(config_file: str, updates: Dict[str, Any]) -> None:
        """
        Update the YAML configuration file with the given updates.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file.
        updates : dict
            Dictionary of updates to apply. Keys and values in `updates`
            will overwrite or add to those in the current config.

        Example usage..
        new_updates = {
            'denoising': {
                'method': 'dimr',  # Change from 'deep_snf' to 'dimr'
                'channels': ['Channel1', 'Channel3']  # Update channel list
            }
        }

        update_config_file("config.yaml", new_updates)
        """
        # Load the existing config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}  # If file is empty or null

        # Merge updates into the config
        # For a shallow update, a simple dictionary update is sufficient.
        # If nested keys need updating, consider writing a custom deep-merge function.
        config.update(updates)

        # Write the updated config back to the file
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        # Optionally, log the update
        # Assuming you have a logging setup
        import logging
        logging.info(f'Configuration file "{config_file}" updated with: {updates}')

def load_config(config_file: str) -> Dict:
    # Example load_config function—already exists in your code
    with open(config_file, 'r') as f:
        return yaml.safe_load(f) or {}

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
    parser = argparse.ArgumentParser(description="Run the denoising step of the pipeline with overrides.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config (default: config.yaml)')
    parser.add_argument('--override', action='append', help='Overrides in key=value format. Use dot-notation for keys.')
    return parser.parse_args()

def process_config_with_overrides():
    args = parse_arguments()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.override:
        for ov in args.override:
            if '=' not in ov:
                logging.warning(f"Invalid override (no '=' found): {ov}")
                continue
            key_path, value = ov.split('=', 1)
            apply_override(config, key_path.strip(), value.strip())

    return config