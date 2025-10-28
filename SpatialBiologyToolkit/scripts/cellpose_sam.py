#!/usr/bin/env python3
"""
CellPose-SAM Segmentation Script

This script performs nucleus segmentation using the CellPose-SAM model on preprocessed DNA images.
It is designed to work with images that have been preprocessed using the preprocess_dna.py script,
providing the second half of the original createmasks functionality.

CellPose-SAM (CP-SAM) combines CellPose's flow-based segmentation with SAM's attention mechanisms
for improved segmentation accuracy, particularly on challenging cell types and imaging conditions.

Usage:
    python cellpose_sam.py                                               # Uses default config
    python cellpose_sam.py --config custom.yaml                         # Uses custom config
    python cellpose_sam.py --override createmasks.cellpose_cell_diameter=15  # Override diameter
    python cellpose_sam.py --override createmasks.cellprob_threshold=-2.0    # Override threshold
"""

# GPU acceleration imports must be first
import torch
from cellpose import models

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from skimage import io as skio
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from skimage.morphology import remove_small_objects
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List
import warnings

from .config_and_utils import (
    process_config_with_overrides,
    setup_logging,
    GeneralConfig,
    CreateMasksConfig,
    cleanstring
)


def create_qc_overlay(
    image: np.ndarray,
    final_masks: np.ndarray,
    excluded_masks: np.ndarray = None,
    boundary_dilation: int = 0,
    vmin: float = 0,
    vmax_quantile: float = 0.97,
    outline_alpha: float = 0.8
) -> np.ndarray:
    """
    Create an overlay of segmentation masks on a grayscale image and return it as an RGB array.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale DNA image.
    final_masks : np.ndarray
        Final segmentation masks (kept objects).
    excluded_masks : np.ndarray, optional
        Excluded masks (filtered out objects).
    boundary_dilation : int, optional
        Number of pixels to dilate boundaries. Default is 0.
    vmin : float, optional
        Minimum intensity for normalization. Default is 0.
    vmax_quantile : float, optional
        Quantile for maximum intensity normalization. Default is 0.97.
    outline_alpha : float, optional
        Alpha transparency for outlines. Default is 0.8.
        
    Returns
    -------
    np.ndarray
        RGB overlay image.
    """
    from skimage.segmentation import find_boundaries
    from skimage.morphology import binary_dilation
    
    vmax = np.quantile(image, vmax_quantile)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(normalized_image, cmap="gray", interpolation="none")
    ax.axis("off")

    # Create mask overlays
    masks_and_colors = []
    if final_masks is not None and np.any(final_masks > 0):
        masks_and_colors.append((final_masks, 'green'))
    
    if excluded_masks is not None and np.any(excluded_masks > 0):
        masks_and_colors.append((excluded_masks, 'red'))

    for label_array, color in masks_and_colors:
        boundaries = find_boundaries(label_array, mode="outer")
        # Increase boundary thickness if needed
        for _ in range(boundary_dilation):
            boundaries = binary_dilation(boundaries)

        cmap = ListedColormap([[0, 0, 0, 0], plt.cm.colors.to_rgba(color)])
        ax.imshow(boundaries, cmap=cmap, alpha=outline_alpha, interpolation="none")

    fig.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    # Matplotlib version compatibility
    try:
        # <= 3.7.0
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    except:
        # >= 3.8.0
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[...,:3]

    plt.close(fig)
    return img_array


def load_preprocessed_image(image_path: Path) -> np.ndarray:
    """
    Load a preprocessed DNA image.
    
    Parameters
    ----------
    image_path : Path
        Path to the preprocessed image file.
        
    Returns
    -------
    np.ndarray
        Loaded image array.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Preprocessed image not found: {image_path}")
    
    img = skio.imread(image_path)
    return img


def segment_single_roi(
    roi_name: str,
    image_path: Path,
    output_dir: Path,
    qc_dir: Path,
    config: CreateMasksConfig,
    cp_sam_model=None
) -> dict:
    """
    Segment a single ROI using CellPose-SAM.
    
    Parameters
    ----------
    roi_name : str
        Name of the ROI.
    image_path : Path
        Path to the preprocessed DNA image.
    output_dir : Path
        Output directory for masks.
    qc_dir : Path
        QC directory for overlay images.
    config : CreateMasksConfig
        Configuration object.
    cp_sam_model : CellposeModel, optional
        Pre-loaded CellPose-SAM model.
        
    Returns
    -------
    dict
        Segmentation results and statistics.
    """
    logging.info(f"Segmenting ROI: {roi_name}")
    
    # Load preprocessed image
    img = load_preprocessed_image(image_path)
    original_shape = img.shape
    
    # Initialize model if not provided
    if cp_sam_model is None:
        cp_sam_model = models.CellposeModel(
            model_type='cpsam',
            gpu=True,
            pretrained_model='cpsam'
        )
    
    # Prepare normalization parameters
    normalize_params = {
        'normalize': config.image_normalise,
        'percentile': [config.image_normalise_percentile_lower, config.image_normalise_percentile_upper]
    }
    
    # Run CellPose-SAM segmentation
    # If images were upscaled during preprocessing, we need to account for this
    # by adjusting the diameter and then downscaling the masks
    diameter_for_segmentation = config.cellpose_cell_diameter
    if config.run_upscale:
        diameter_for_segmentation = config.cellpose_cell_diameter * config.upscale_ratio
    
    logging.debug(f"Running CellPose-SAM on {roi_name} with diameter={diameter_for_segmentation}")
    
    try:
        masks, flows, styles, diams = cp_sam_model.eval(
            img,
            diameter=diameter_for_segmentation,
            channels=None,  # Grayscale image
            batch_size=config.batch_size,
            normalize=normalize_params,
            cellprob_threshold=config.cellprob_threshold,
            flow_threshold=config.flow_threshold,
            min_size=config.min_cell_area or 15,
            max_size_fraction=config.max_size_fraction,
            resample=config.resample,
            augment=config.augment,
            tile_overlap=config.tile_overlap,
            compute_masks=True
        )
        
    except Exception as e:
        logging.error(f"CellPose-SAM segmentation failed for {roi_name}: {str(e)}")
        raise
    
    # If preprocessing included upscaling, we need to get the original image size
    # to downscale the masks back to the original dimensions
    # We'll estimate the original size based on the upscale ratio
    if config.run_upscale:
        original_estimated_shape = (
            int(img.shape[0] / config.upscale_ratio),
            int(img.shape[1] / config.upscale_ratio)
        )
        logging.debug(f"Downscaling masks from {img.shape} to estimated original size {original_estimated_shape}")
        masks = resize(masks, original_estimated_shape, order=0, preserve_range=True, anti_aliasing=False)
        masks = masks.astype(np.uint16)
    
    # Process masks
    if config.fill_holes:
        # Fill holes in masks
        unique_labels = np.unique(masks)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            mask_binary = (masks == label)
            mask_filled = binary_fill_holes(mask_binary)
            masks[mask_filled] = label
    
    # Remove edge masks if requested
    if config.remove_edge_masks:
        from cellpose.utils import remove_edge_masks
        masks = remove_edge_masks(masks, change_index=True)
    
    # Expand masks if requested
    if config.expand_masks > 0:
        masks = expand_labels(masks, distance=config.expand_masks)
    
    # Calculate statistics
    region_props = regionprops(masks)
    total_objects = len(region_props)
    
    # Apply size filtering based on original (downscaled) image dimensions
    final_mask = np.zeros_like(masks, dtype=np.uint16)
    excluded_mask = np.zeros_like(masks, dtype=np.uint16)
    kept_objects = 0
    
    image_area = masks.shape[0] * masks.shape[1]
    max_area = int(config.max_size_fraction * image_area)
    min_area = config.min_cell_area or 15
    
    for region in region_props:
        area = region.area
        if min_area <= area <= max_area:
            kept_objects += 1
            final_mask[masks == region.label] = region.label
        else:
            excluded_mask[masks == region.label] = region.label
    
    # Calculate density
    pixels_per_mm2 = 1e6  # Assuming 1 pixel = 1 μm
    mask_area_mm2 = image_area / pixels_per_mm2
    objects_per_mm2 = kept_objects / mask_area_mm2 if mask_area_mm2 > 0 else 0
    
    # Save final mask
    mask_path = output_dir / f"{roi_name}.tiff"
    skio.imsave(mask_path, final_mask.astype(np.uint16))
    
    # Create QC overlay if requested
    # For QC, we'll use the downscaled image if available, or the preprocessed image
    qc_image_path_str = None
    if config.perform_qc:
        qc_overlay_dir = qc_dir / 'CellposeSAM_overlay'
        qc_overlay_dir.mkdir(exist_ok=True, parents=True)
        
        # Use the appropriately sized image for QC overlay
        qc_image = img
        if config.run_upscale:
            # Downscale the QC image to match the mask size
            qc_image = resize(img, final_mask.shape, order=1, preserve_range=True, anti_aliasing=True)
        
        qc_image_array = create_qc_overlay(
            image=qc_image,
            final_masks=final_mask,
            excluded_masks=excluded_mask,
            boundary_dilation=config.qc_boundary_dilation,
            vmin=0,
            vmax_quantile=0.97,
            outline_alpha=0.8
        )
        
        qc_image_path = qc_overlay_dir / f"{roi_name}_cpsam_overlay.png"
        plt.imsave(qc_image_path, qc_image_array, dpi=config.dpi_qc_images)
        qc_image_path_str = str(qc_image_path)
    
    # Compile results
    result = {
        'ROI': roi_name,
        'Input_image': str(image_path),
        'Mask_output': str(mask_path),
        'Total_objects_detected': total_objects,
        'Objects_kept': kept_objects,
        'Objects_excluded': total_objects - kept_objects,
        'Objects_per_mm2': objects_per_mm2,
        'Image_shape_processed': f"{original_shape[0]}x{original_shape[1]}",
        'Mask_shape_final': f"{final_mask.shape[0]}x{final_mask.shape[1]}",
        'Model_type': 'cpsam',
        'Diameter_used': diameter_for_segmentation,
        'Diameter_base': config.cellpose_cell_diameter,
        'Upscale_ratio': config.upscale_ratio if config.run_upscale else 1.0,
        'Actual_diameter': diams[0] if isinstance(diams, (list, np.ndarray)) else diams,
        'CellProb_threshold': config.cellprob_threshold,
        'Flow_threshold': config.flow_threshold,
        'Min_size': min_area,
        'Max_size_fraction': config.max_size_fraction,
        'Image_normalize': config.image_normalise,
        'Expand_masks': config.expand_masks,
        'Fill_holes': config.fill_holes,
        'Remove_edge_masks': config.remove_edge_masks,
        'QC_image_path': qc_image_path_str
    }
    
    return result


def process_all_rois(general_config: GeneralConfig, mask_config: CreateMasksConfig):
    """
    Process all ROIs with CellPose-SAM segmentation.
    
    Parameters
    ----------
    general_config : GeneralConfig
        General configuration.
    mask_config : CreateMasksConfig
        Mask creation configuration with CellPose-SAM settings.
    """
    logging.info("Starting CellPose-SAM segmentation for all ROIs.")
    
    # Setup paths
    input_folder = Path(mask_config.output_folder_name)  # Use existing preprocessed DNA folder
    output_folder = Path(general_config.masks_folder)    # Use standard masks folder
    qc_folder = Path(general_config.qc_folder) / 'CellposeSAM_QC'
    
    # Create output directories
    output_folder.mkdir(parents=True, exist_ok=True)
    if mask_config.perform_qc:
        qc_folder.mkdir(parents=True, exist_ok=True)
    
    # Find available ROIs
    if mask_config.specific_rois:
        rois_to_process = mask_config.specific_rois
        logging.info(f"Processing specific ROIs: {rois_to_process}")
    else:
        # Find all .tiff files in input folder
        image_files = list(input_folder.glob("*.tiff")) + list(input_folder.glob("*.tif"))
        rois_to_process = [f.stem for f in image_files]
        logging.info(f"Found {len(rois_to_process)} ROIs in {input_folder}")
        
        if not rois_to_process:
            logging.error(f"No .tiff files found in {input_folder}")
            return
    
    # Check GPU availability
    logging.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU device: {torch.cuda.get_device_name()}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize CellPose-SAM model
    logging.info("Initializing CellPose-SAM model")
    try:
        cp_sam_model = models.CellposeModel(
            model_type='cpsam',
            gpu=True,
            pretrained_model='cpsam'
        )
        logging.info("CellPose-SAM model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load CellPose-SAM model: {str(e)}")
        raise
    
    # Process each ROI
    results = []
    successful_rois = []
    failed_rois = []
    
    for roi in tqdm(rois_to_process, desc="Segmenting ROIs"):
        try:
            # Construct image path
            image_path = input_folder / f"{roi}.tiff"
            if not image_path.exists():
                image_path = input_folder / f"{roi}.tif"
                if not image_path.exists():
                    logging.warning(f"Image file not found for ROI {roi}")
                    failed_rois.append(roi)
                    continue
            
            result = segment_single_roi(
                roi_name=roi,
                image_path=image_path,
                output_dir=output_folder,
                qc_dir=qc_folder,
                config=mask_config,
                cp_sam_model=cp_sam_model
            )
            
            results.append(result)
            successful_rois.append(roi)
            
        except Exception as e:
            logging.error(f"Error processing ROI {roi}: {str(e)}", exc_info=True)
            failed_rois.append(roi)
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_path = qc_folder / 'CellposeSAM_segmentation_results.csv'
        results_df.to_csv(results_path, index=False)
        logging.info(f"Saved segmentation results to {results_path}")
        
        # Print summary statistics
        logging.info(f"\nCellPose-SAM Segmentation Summary:")
        logging.info(f"Total ROIs processed: {len(successful_rois)}")
        logging.info(f"Failed ROIs: {len(failed_rois)}")
        if failed_rois:
            logging.warning(f"Failed ROI list: {failed_rois}")
        
        # Calculate average statistics
        if len(results) > 0:
            avg_objects = np.mean([r['Objects_kept'] for r in results])
            avg_density = np.mean([r['Objects_per_mm2'] for r in results])
            avg_diameter = np.mean([r['Actual_diameter'] for r in results])
            
            logging.info(f"Average objects per ROI: {avg_objects:.1f}")
            logging.info(f"Average density: {avg_density:.1f} objects/mm²")
            logging.info(f"Average diameter used: {avg_diameter:.1f} pixels")
    
    logging.info("CellPose-SAM segmentation completed.")


def print_model_info():
    """Print available CellPose models and system information."""
    logging.info("System Information:")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU device: {torch.cuda.get_device_name()}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        from cellpose import models
        logging.info("Available CellPose models:")
        for model_name in models.MODEL_NAMES:
            logging.info(f"  - {model_name}")
    except Exception as e:
        logging.warning(f"Could not retrieve model list: {e}")


if __name__ == "__main__":
    # Define pipeline stage
    pipeline_stage = 'CellposeSAM'
    
    # Load configuration
    config_data = process_config_with_overrides()
    
    # Setup logging
    setup_logging(config_data.get('logging', {}), pipeline_stage)
    
    # Print system and model info
    print_model_info()
    
    # Get configuration objects
    general_config = GeneralConfig(**config_data.get('general', {}))
    mask_config = CreateMasksConfig(**config_data.get('createmasks', {}))
    
    logging.info(f"CellPose-SAM configuration: {mask_config}")
    
    # Run segmentation
    process_all_rois(general_config, mask_config)