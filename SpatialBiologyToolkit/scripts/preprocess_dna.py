#!/usr/bin/env python3
"""
Preprocess DNA images using Cellpose deblur and upscale models.

This script extracts DNA images from ROI folders, applies optional deblur and upscale
preprocessing, and saves the processed images to a new directory. The processed images
can then be used with CellPose-SAM segmentation in a separate environment.

The script uses the existing 'createmasks' configuration section for backwards compatibility,
focusing only on the preprocessing steps (deblur/upscale) without segmentation.

Usage:
    python preprocess_dna.py                                    # Uses default config
    python preprocess_dna.py --config custom.yaml              # Uses custom config
    python preprocess_dna.py --override createmasks.run_deblur=false  # Disable deblur
"""

# GPU acceleration imports must be first
import torch
from cellpose import denoise

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from skimage import io as skio
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List

from .config_and_utils import (
    process_config_with_overrides,
    setup_logging,
    GeneralConfig,
    CreateMasksConfig,
    get_filename,
    cleanstring
)


def load_dna_image(roi_path: Path, dna_image_name: str) -> np.ndarray:
    """
    Load DNA image from ROI folder.
    
    Parameters
    ----------
    roi_path : Path
        Path to the ROI folder.
    dna_image_name : str
        Name pattern for DNA image file.
        
    Returns
    -------
    np.ndarray
        Loaded DNA image.
    """
    dna_file = get_filename(roi_path, dna_image_name)
    img = skio.imread(roi_path / dna_file)
    return img


def preprocess_single_roi(
    roi: str,
    input_folder: Path,
    output_folder: Path,
    config: CreateMasksConfig,
    deblur_model=None,
    upscale_model=None
) -> dict:
    """
    Preprocess DNA image for a single ROI.
    
    Parameters
    ----------
    roi : str
        ROI name.
    input_folder : Path
        Input folder containing ROI subfolders.
    output_folder : Path
        Output folder for processed images.
    config : PreprocessDNAConfig
        Configuration object.
    deblur_model : cellpose.denoise.DenoiseModel, optional
        Deblur model instance.
    upscale_model : cellpose.denoise.DenoiseModel, optional
        Upscale model instance.
        
    Returns
    -------
    dict
        Processing results and statistics.
    """
    logging.info(f"Processing ROI: {roi}")
    
    roi_path = input_folder / roi
    dna_file = get_filename(roi_path, config.dna_image_name)
    original_img = skio.imread(roi_path / dna_file)
    
    # Track processing steps
    processing_steps = []
    current_img = original_img.copy()
    
    # Apply deblur if enabled
    if config.run_deblur and deblur_model is not None:
        logging.debug(f"Applying deblur to {roi}")
        current_img = deblur_model.eval(
            current_img, 
            channels=None, 
            diameter=config.cellpose_cell_diameter, 
            batch_size=16
        )
        processing_steps.append("deblur")
    
    # Apply upscale if enabled
    if config.run_upscale and upscale_model is not None:
        logging.debug(f"Applying upscale to {roi}")
        current_img = upscale_model.eval(
            current_img, 
            channels=None, 
            diameter=config.cellpose_cell_diameter, 
            batch_size=16
        )
        processing_steps.append("upscale")
    
    # Save processed image directly in output folder as {roi_name}.tiff
    output_filename = f"{roi}.tiff"
    output_path = output_folder / output_filename
    skio.imsave(output_path, current_img.astype(np.float32))
    
    # Calculate statistics
    original_stats = {
        'mean': np.mean(original_img),
        'std': np.std(original_img),
        'min': np.min(original_img),
        'max': np.max(original_img)
    }
    
    processed_stats = {
        'mean': np.mean(current_img),
        'std': np.std(current_img),
        'min': np.min(current_img),
        'max': np.max(current_img)
    }
    
    result = {
        'ROI': roi,
        'Original_DNA_file': dna_file,
        'Output_file': output_filename,
        'Output_path': str(output_path),
        'Processing_steps': ', '.join(processing_steps),
        'Run_deblur': config.run_deblur,
        'Run_upscale': config.run_upscale,
        'Cell_diameter': config.cellpose_cell_diameter,
        'Original_shape': f"{original_img.shape[0]}x{original_img.shape[1]}",
        'Processed_shape': f"{current_img.shape[0]}x{current_img.shape[1]}",
        'Original_mean': original_stats['mean'],
        'Original_std': original_stats['std'],
        'Original_min': original_stats['min'],
        'Original_max': original_stats['max'],
        'Processed_mean': processed_stats['mean'],
        'Processed_std': processed_stats['std'],
        'Processed_min': processed_stats['min'],
        'Processed_max': processed_stats['max'],
        'Size_change_factor': current_img.size / original_img.size
    }
    
    return result, original_img, current_img


def create_comparison_image(original_img: np.ndarray, processed_img: np.ndarray, roi: str, output_path: Path):
    """
    Create side-by-side comparison image for QC.
    
    Parameters
    ----------
    original_img : np.ndarray
        Original DNA image.
    processed_img : np.ndarray
        Processed DNA image.
    roi : str
        ROI name for title.
    output_path : Path
        Path to save comparison image.
    """
    import matplotlib.pyplot as plt
    
    # Resize processed image to match original if different sizes
    if processed_img.shape != original_img.shape:
        from skimage.transform import resize
        processed_img_display = resize(
            processed_img, 
            original_img.shape, 
            order=1, 
            preserve_range=True, 
            anti_aliasing=True
        )
    else:
        processed_img_display = processed_img
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    im1 = axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'{roi} - Original')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.6)
    
    # Processed image
    im2 = axes[1].imshow(processed_img_display, cmap='gray')
    axes[1].set_title(f'{roi} - Processed')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def process_all_rois(general_config: GeneralConfig, mask_config: CreateMasksConfig):
    """
    Process all ROIs with DNA preprocessing.
    
    Parameters
    ----------
    general_config : GeneralConfig
        General configuration.
    mask_config : CreateMasksConfig
        Mask creation configuration (using existing config for compatibility).
    """
    logging.info("Starting DNA preprocessing for all ROIs.")
    
    # Setup paths - use simple output folder from config
    input_folder = Path(general_config.denoised_images_folder)
    output_folder = Path(mask_config.output_folder_name)
    qc_folder = Path(general_config.qc_folder) / 'DNA_preprocessing_QC'
    
    # Create output directories
    output_folder.mkdir(parents=True, exist_ok=True)
    if mask_config.perform_qc:
        qc_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine ROIs to process
    if mask_config.specific_rois:
        rois_to_process = mask_config.specific_rois
        logging.info(f"Processing specific ROIs: {rois_to_process}")
    else:
        rois_to_process = [entry.name for entry in input_folder.iterdir() if entry.is_dir()]
        logging.info(f"Processing all ROIs found in {input_folder}")
    
    # Initialize models
    deblur_model = None
    upscale_model = None
    
    if mask_config.run_deblur:
        logging.info("Initializing deblur model...")
        deblur_model = denoise.DenoiseModel(model_type='deblur_nuclei', gpu=True)
    
    if mask_config.run_upscale:
        logging.info("Initializing upscale model...")
        upscale_model = denoise.DenoiseModel(model_type='upsample_nuclei', gpu=True)
    
    # Process each ROI
    results = []
    successful_rois = []
    failed_rois = []
    
    for roi in tqdm(rois_to_process, desc="Processing ROIs"):
        try:
            result, original_img, processed_img = preprocess_single_roi(
                roi=roi,
                input_folder=input_folder,
                output_folder=output_folder,
                config=mask_config,
                deblur_model=deblur_model,
                upscale_model=upscale_model
            )
            
            results.append(result)
            successful_rois.append(roi)
            
            # Create QC comparison image if enabled
            if mask_config.perform_qc:
                comparison_path = qc_folder / f"{roi}_comparison.png"
                create_comparison_image(original_img, processed_img, roi, comparison_path)
            
        except Exception as e:
            logging.error(f"Error processing ROI {roi}: {str(e)}", exc_info=True)
            failed_rois.append(roi)
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_path = qc_folder / 'DNA_preprocessing_results.csv'
        results_df.to_csv(results_path, index=False)
        logging.info(f"Saved processing results to {results_path}")
        
        # Print summary statistics
        logging.info(f"\nDNA Preprocessing Summary:")
        logging.info(f"Total ROIs processed: {len(successful_rois)}")
        logging.info(f"Failed ROIs: {len(failed_rois)}")
        if failed_rois:
            logging.warning(f"Failed ROI list: {failed_rois}")
        
        # Calculate average statistics
        if len(results) > 0:
            avg_size_change = np.mean([r['Size_change_factor'] for r in results])
            logging.info(f"Average size change factor: {avg_size_change:.2f}")
            
            steps_used = set()
            for r in results:
                if r['Processing_steps']:
                    steps_used.update(r['Processing_steps'].split(', '))
            logging.info(f"Processing steps applied: {', '.join(steps_used)}")
    
    logging.info("DNA preprocessing completed.")


def print_gpu_info():
    """Print GPU information for debugging."""
    logging.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU device: {torch.cuda.get_device_name()}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    # Define pipeline stage
    pipeline_stage = 'PreprocessDNA'
    
    # Load configuration
    config_data = process_config_with_overrides()
    
    # Setup logging
    setup_logging(config_data.get('logging', {}), pipeline_stage)
    
    # Print GPU info
    print_gpu_info()
    
    # Get configuration objects
    general_config = GeneralConfig(**config_data.get('general', {}))
    
    # Use existing CreateMasksConfig for backwards compatibility
    mask_config = CreateMasksConfig(**config_data.get('createmasks', {}))
    
    logging.info(f"DNA preprocessing using mask configuration: {mask_config}")
    
    # Run preprocessing
    process_all_rois(general_config, mask_config)