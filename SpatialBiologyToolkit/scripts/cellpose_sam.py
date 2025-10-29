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
import seaborn as sb

from .config_and_utils import (
    process_config_with_overrides,
    setup_logging,
    GeneralConfig,
    CreateMasksConfig,
    cleanstring
)


def load_cellpose_model(model_name: str, use_gpu: bool = True):
    """
    Load a CellPose model with correct parameter handling.
    
    Parameters
    ----------
    model_name : str
        Model name - 'cpsam' for CellPose-SAM, or standard model names like 'nuclei', 'cyto', 'cyto2'
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    CellposeModel
        Initialized CellPose model
    """
    if model_name == 'cpsam':
        # Use CellPose-SAM model with pretrained_model parameter
        return models.CellposeModel(
            pretrained_model='cpsam',
            gpu=use_gpu
        )
    else:
        # Use standard CellPose model with model_type parameter
        return models.CellposeModel(
            model_type=model_name,
            gpu=use_gpu
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
    denoised_folder: Path,
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
    denoised_folder : Path
        Path to the denoised images folder for overlay generation.
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
    preprocessed_shape = img.shape
    
    # Load original denoised DNA image to get true original dimensions
    from .config_and_utils import get_filename
    roi_denoised_path = denoised_folder / roi_name
    original_dna_img = None
    original_shape = None
    
    if roi_denoised_path.exists():
        try:
            dna_file = get_filename(roi_denoised_path, config.dna_image_name)
            original_dna_img = skio.imread(roi_denoised_path / dna_file)
            original_shape = original_dna_img.shape
            logging.debug(f"Loaded original denoised DNA image: {original_shape}")
        except Exception as e:
            logging.warning(f"Could not load original DNA image for {roi_name}: {e}")
            original_shape = preprocessed_shape  # Fallback
    else:
        logging.warning(f"Denoised ROI folder not found: {roi_denoised_path}")
        original_shape = preprocessed_shape  # Fallback
    
    # Initialize model if not provided
    if cp_sam_model is None:
        # Determine if GPU is available and working
        use_gpu = torch.cuda.is_available()
        cp_sam_model = load_cellpose_model(config.cell_pose_sam_model, use_gpu)
    
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
        # CellPose upscale models have fixed target diameters:
        # upsample_nuclei -> 17.0 pixels, upsample_cyto3 -> 30.0 pixels
        # Use the actual target diameter rather than our assumed ratio
        diameter_for_segmentation = config.upscale_target_diameter
    
    logging.debug(f"Running CellPose-SAM on {roi_name} with diameter={diameter_for_segmentation}")
    
    try:
        # Optimize batch size for CPU vs GPU
        batch_size = config.batch_size if torch.cuda.is_available() else 1
        
        masks, flows, styles = cp_sam_model.eval(
            img,
            diameter=diameter_for_segmentation,
            channels=None,  # Grayscale image
            batch_size=batch_size,
            normalize=normalize_params,
            cellprob_threshold=config.cellprob_threshold,
            flow_threshold=config.flow_threshold,
            min_size=config.min_cell_area or 15,
            augment=config.augment,
            compute_masks=True
        )
        
        # In CellPose v4+, diameter info might be in styles or we use the input diameter
        actual_diameter = diameter_for_segmentation  # Fallback to input diameter
        
    except Exception as e:
        logging.error(f"CellPose-SAM segmentation failed for {roi_name}: {str(e)}")
        raise
    
    # If preprocessing included upscaling, we need to downscale the masks back to original dimensions
    if config.run_upscale and original_shape is not None:
        logging.debug(f"Downscaling masks from {masks.shape} to original size {original_shape}")
        masks = resize(masks, original_shape, order=0, preserve_range=True, anti_aliasing=False)
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
    
    # Create QC overlays if requested
    qc_image_path_str = None
    qc_raw_overlay_path_str = None
    
    if config.perform_qc:
        qc_overlay_dir = qc_dir / 'CellposeSAM_overlay'
        qc_raw_overlay_dir = qc_dir / 'CellposeSAM_raw_overlay'
        qc_overlay_dir.mkdir(exist_ok=True, parents=True)
        qc_raw_overlay_dir.mkdir(exist_ok=True, parents=True)
        
        # Create overlay on processed image (resized to match final masks)
        qc_image = img
        if config.run_upscale and original_shape is not None:
            # Downscale the processed image to match the final mask size
            qc_image = resize(img, original_shape, order=1, preserve_range=True, anti_aliasing=True)
        
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
        
        # Create overlay on original denoised image if available
        if original_dna_img is not None:
            qc_raw_image_array = create_qc_overlay(
                image=original_dna_img,
                final_masks=final_mask,
                excluded_masks=excluded_mask,
                boundary_dilation=config.qc_boundary_dilation,
                vmin=0,
                vmax_quantile=0.97,
                outline_alpha=0.8
            )
            
            qc_raw_image_path = qc_raw_overlay_dir / f"{roi_name}_cpsam_raw_overlay.png"
            plt.imsave(qc_raw_image_path, qc_raw_image_array, dpi=config.dpi_qc_images)
            qc_raw_overlay_path_str = str(qc_raw_image_path)
            logging.debug(f"Created raw overlay: {qc_raw_image_path}")
        else:
            logging.warning(f"No original DNA image available for raw overlay: {roi_name}")
    
    # Compile results
    result = {
        'ROI': roi_name,
        'Input_image': str(image_path),
        'Mask_output': str(mask_path),
        'Total_objects_detected': total_objects,
        'Objects_kept': kept_objects,
        'Objects_excluded': total_objects - kept_objects,
        'Objects_per_mm2': objects_per_mm2,
        'Image_shape_preprocessed': f"{preprocessed_shape[0]}x{preprocessed_shape[1]}",
        'Image_shape_original': f"{original_shape[0]}x{original_shape[1]}" if original_shape else "unknown",
        'Mask_shape_final': f"{final_mask.shape[0]}x{final_mask.shape[1]}",
        'Model_type': config.cell_pose_sam_model,
        'Diameter_used': diameter_for_segmentation,
        'Diameter_base': config.cellpose_cell_diameter,
        'Upscale_ratio': config.calculated_upscale_ratio if config.run_upscale else 1.0,
        'Upscale_target_diameter': config.upscale_target_diameter if config.run_upscale else config.cellpose_cell_diameter,
        'Actual_diameter': actual_diameter,
        'CellProb_threshold': config.cellprob_threshold,
        'Flow_threshold': config.flow_threshold,
        'Min_size': min_area,
        'Max_size_fraction': config.max_size_fraction,
        'Image_normalize': config.image_normalise,
        'Expand_masks': config.expand_masks,
        'Fill_holes': config.fill_holes,
        'Remove_edge_masks': config.remove_edge_masks,
        'QC_image_path': qc_image_path_str,
        'QC_raw_overlay_path': qc_raw_overlay_path_str
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
    denoised_folder = Path(general_config.denoised_images_folder)  # For raw overlay generation
    
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
    gpu_available = torch.cuda.is_available()
    logging.info(f"GPU available: {gpu_available}")
    if gpu_available:
        logging.info(f"GPU device: {torch.cuda.get_device_name()}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logging.warning("GPU not available - CellPose-SAM will run on CPU which is VERY slow!")
        logging.warning("Consider using a system with CUDA support for faster processing.")
        logging.warning(f"Estimated time per ROI on CPU: ~6-8 hours (Total: ~{len(rois_to_process) * 7:.0f} hours)")
        logging.warning("RECOMMENDATION: Use 'specific_rois' to test on a small subset first!")
        
        # If running on CPU with many ROIs, suggest limiting the scope
        if len(rois_to_process) > 5:
            logging.warning(f"Processing {len(rois_to_process)} ROIs on CPU will take ~{len(rois_to_process) * 7:.0f} hours!")
            logging.warning("Consider setting 'specific_rois: [roi1, roi2, roi3]' in config for testing.")
    
    # Initialize CellPose-SAM model
    logging.info("Initializing CellPose model")
    try:
        # Determine if GPU is available and working
        use_gpu = torch.cuda.is_available()
        logging.info(f"Using GPU: {use_gpu}")
        
        cp_sam_model = load_cellpose_model(mask_config.cell_pose_sam_model, use_gpu)
        logging.info(f"CellPose model '{mask_config.cell_pose_sam_model}' loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model '{mask_config.cell_pose_sam_model}': {str(e)}")
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
                denoised_folder=denoised_folder,
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
            
            # Check for dimension consistency
            dimension_issues = sum(1 for r in results if r['Image_shape_original'] == 'unknown')
            if dimension_issues > 0:
                logging.warning(f"{dimension_issues} ROIs had dimension detection issues")
            
            # Count overlay creation success
            raw_overlays_created = sum(1 for r in results if r.get('QC_raw_overlay_path') is not None)
            logging.info(f"Raw overlays created: {raw_overlays_created}/{len(results)} ROIs")
    
    logging.info("CellPose-SAM segmentation completed.")


def parameter_scan_cpsam(general_config: GeneralConfig, mask_config: CreateMasksConfig):
    """
    Parameter scan mode for CellPose-SAM: run multiple parameter sets defined by two parameters
    (param_a, param_b), each with a list of values. Create summarizing plots comparing performance.
    
    Simplified version that processes all ROIs (no sampling) and saves masks/QC with parameter
    identifiers in folder names.
    
    Parameters
    ----------
    general_config : GeneralConfig
        General configuration.
    mask_config : CreateMasksConfig
        Mask creation configuration with parameter scan settings.
    """
    logging.info("Starting CellPose-SAM parameter scan.")
    
    param_a = mask_config.param_a
    param_a_values = mask_config.param_a_values or []
    param_b = mask_config.param_b
    param_b_values = mask_config.param_b_values or []
    
    if not param_a or not param_a_values or not param_b or not param_b_values:
        logging.error("Parameter scan requires param_a, param_a_values, param_b, and param_b_values to be set")
        return
    
    # Setup base paths
    input_folder = Path(mask_config.output_folder_name)
    denoised_folder = Path(general_config.denoised_images_folder)
    base_qc_folder = Path(general_config.qc_folder) / f'CellposeSAM_ParameterScan_{cleanstring(param_a)}_{cleanstring(param_b)}'
    base_qc_folder.mkdir(parents=True, exist_ok=True)
    
    # Find available ROIs
    if mask_config.specific_rois:
        rois_to_process = mask_config.specific_rois
        logging.info(f"Parameter scanning specific ROIs: {rois_to_process}")
    else:
        # Find all .tiff files in input folder
        image_files = list(input_folder.glob("*.tiff")) + list(input_folder.glob("*.tif"))
        rois_to_process = [f.stem for f in image_files]
        logging.info(f"Parameter scanning all {len(rois_to_process)} ROIs found in {input_folder}")
        
        if not rois_to_process:
            logging.error(f"No .tiff files found in {input_folder}")
            return
    
    # Initialize CellPose model once
    use_gpu = torch.cuda.is_available()
    logging.info(f"Initializing CellPose model (GPU: {use_gpu})")
    cp_sam_model = load_cellpose_model(mask_config.cell_pose_sam_model, use_gpu)
    
    # Construct parameter grid
    param_sets = []
    for a_val in param_a_values:
        for b_val in param_b_values:
            param_sets.append({param_a: a_val, param_b: b_val})
    
    logging.info(f"Running {len(param_sets)} parameter combinations on {len(rois_to_process)} ROIs")
    
    all_results = []
    
    # Run parameter scan
    for i, param_set in enumerate(param_sets):
        logging.info(f"Parameter set {i+1}/{len(param_sets)}: {param_set}")
        
        # Create output folders with parameter identifiers
        param_string = f"{cleanstring(param_a)}-{cleanstring(param_set[param_a])}_{cleanstring(param_b)}-{cleanstring(param_set[param_b])}"
        
        # Create temporary config with current parameters
        temp_config = CreateMasksConfig(**mask_config.__dict__)
        setattr(temp_config, param_a, param_set[param_a])
        setattr(temp_config, param_b, param_set[param_b])
        
        # Check if we need to reinitialize the model (if model type is being scanned)
        current_cp_sam_model = cp_sam_model
        if param_a == 'cell_pose_sam_model' or param_b == 'cell_pose_sam_model':
            try:
                current_cp_sam_model = load_cellpose_model(temp_config.cell_pose_sam_model, use_gpu)
                logging.info(f"Initialized model '{temp_config.cell_pose_sam_model}' for parameter scan")
            except Exception as e:
                logging.error(f"Failed to initialize model '{temp_config.cell_pose_sam_model}': {str(e)}")
                continue
        
        # Setup output folders for this parameter set
        param_masks_folder = Path(general_config.masks_folder) / f'param_{param_string}'
        param_qc_folder = base_qc_folder / f'param_{param_string}'
        param_masks_folder.mkdir(parents=True, exist_ok=True)
        param_qc_folder.mkdir(parents=True, exist_ok=True)
        
        # Process each ROI with current parameter set
        param_results = []
        for roi in tqdm(rois_to_process, desc=f"Param set {i+1}"):
            try:
                # Construct image path
                image_path = input_folder / f"{roi}.tiff"
                if not image_path.exists():
                    image_path = input_folder / f"{roi}.tif"
                    if not image_path.exists():
                        logging.warning(f"Image file not found for ROI {roi}")
                        continue
                
                result = segment_single_roi(
                    roi_name=roi,
                    image_path=image_path,
                    output_dir=param_masks_folder,
                    qc_dir=param_qc_folder,
                    config=temp_config,
                    denoised_folder=denoised_folder,
                    cp_sam_model=current_cp_sam_model
                )
                
                # Add parameter information to result
                result[f'{param_a}'] = param_set[param_a]
                result[f'{param_b}'] = param_set[param_b]
                result['Parameter_set'] = param_string
                
                param_results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing ROI {roi} with params {param_set}: {str(e)}", exc_info=True)
                continue
        
        # Save results for this parameter set
        if param_results:
            param_df = pd.DataFrame(param_results)
            param_csv_path = param_qc_folder / f'CellposeSAM_results_{param_string}.csv'
            param_df.to_csv(param_csv_path, index=False)
            logging.info(f"Saved {len(param_results)} results for parameter set {param_string}")
            
            all_results.extend(param_results)
    
    # Save combined results and create summary plots
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_csv_path = base_qc_folder / 'CellposeSAM_ParameterScan_All.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        logging.info(f"Saved combined parameter scan results to {combined_csv_path}")
        
        # Create summary plots
        create_parameter_scan_plots(combined_df, base_qc_folder, param_a, param_b, mask_config.dpi_qc_images)
        
        # Print summary statistics
        logging.info(f"\nCellPose-SAM Parameter Scan Summary:")
        logging.info(f"Total parameter combinations: {len(param_sets)}")
        logging.info(f"Total ROIs processed: {len(set(r['ROI'] for r in all_results))}")
        logging.info(f"Total segmentations: {len(all_results)}")
        
        # Calculate average statistics by parameter set
        summary_stats = combined_df.groupby(['Parameter_set']).agg({
            'Objects_kept': 'mean',
            'Objects_per_mm2': 'mean',
            'Objects_excluded': 'mean'
        }).round(2)
        
        logging.info("\nAverage statistics by parameter set:")
        for param_set, stats in summary_stats.iterrows():
            logging.info(f"{param_set}: Kept={stats['Objects_kept']:.1f}, "
                        f"Density={stats['Objects_per_mm2']:.1f}/mm², "
                        f"Excluded={stats['Objects_excluded']:.1f}")
    
    logging.info("CellPose-SAM parameter scan completed.")


def create_parameter_scan_plots(df: pd.DataFrame, output_dir: Path, param_a: str, param_b: str, dpi: int = 300):
    """
    Create summary plots for parameter scan results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined results dataframe.
    output_dir : Path
        Output directory for plots.
    param_a : str
        Name of first parameter.
    param_b : str
        Name of second parameter.
    dpi : int
        DPI for saved plots.
    """
    logging.info("Creating parameter scan summary plots")
    
    # Map parameter names to column names (maintain compatibility with original script)
    param_to_column = {
        'cellpose_cell_diameter': 'Diameter_base',
        'cellprob_threshold': 'CellProb_threshold', 
        'flow_threshold': 'Flow_threshold',
        'max_size_fraction': 'Max_size_fraction',
        'min_cell_area': 'Min_size',
        'expand_masks': 'Expand_masks',
        'batch_size': 'batch_size',  # May not be in results, use parameter name
        'cell_pose_sam_model': 'Model_type',
    }
    
    # Use column name if available, otherwise use parameter name directly
    param_a_col = param_to_column.get(param_a, param_a)
    param_b_col = param_to_column.get(param_b, param_b)
    
    # Check if columns exist in dataframe, if not use parameter value directly
    if param_a_col not in df.columns:
        param_a_col = param_a
    if param_b_col not in df.columns:
        param_b_col = param_b
    
    # Metrics to plot
    metrics = ['Objects_kept', 'Objects_per_mm2', 'Objects_excluded']
    metric_labels = ['Objects Kept', 'Objects per mm²', 'Objects Excluded']
    
    for metric, label in zip(metrics, metric_labels):
        try:
            plt.figure(figsize=(12, 8))
            
            # Create barplot
            ax = sb.barplot(
                data=df,
                y=metric,
                x=param_a_col,
                hue=param_b_col,
                palette='tab20',
                ci='sd',  # Show standard deviation
                capsize=0.1
            )
            
            # Customize plot
            ax.set_title(f'CellPose-SAM Parameter Scan: {label}', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{param_a}', fontsize=12)
            ax.set_ylabel(f'{label}', fontsize=12)
            
            # Move legend outside plot area
            sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            
            # Save plot
            plot_path = output_dir / f"ParameterScan_{cleanstring(metric)}.png"
            plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
            plt.close()
            
            logging.info(f"Saved parameter scan plot: {plot_path}")
            
        except Exception as e:
            logging.error(f"Error creating plot for {metric}: {str(e)}")
            plt.close()
            continue
    
    # Create heatmap for Objects_kept
    try:
        plt.figure(figsize=(10, 8))
        
        # Pivot data for heatmap
        pivot_data = df.groupby([param_a_col, param_b_col])['Objects_kept'].mean().unstack()
        
        # Create heatmap
        sb.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='viridis',
            cbar_kws={'label': 'Average Objects Kept'}
        )
        
        plt.title('CellPose-SAM Parameter Scan: Objects Kept Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel(f'{param_b}', fontsize=12)
        plt.ylabel(f'{param_a}', fontsize=12)
        
        # Save heatmap
        heatmap_path = output_dir / f"ParameterScan_Objects_Kept_Heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        
        logging.info(f"Saved parameter scan heatmap: {heatmap_path}")
        
    except Exception as e:
        logging.error(f"Error creating heatmap: {str(e)}")
        plt.close()


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
    
    # Decide mode based on run_parameter_scan and param fields
    if (mask_config.run_parameter_scan and
        mask_config.param_a and mask_config.param_a_values and
        mask_config.param_b and mask_config.param_b_values):
        parameter_scan_cpsam(general_config, mask_config)
    else:
        # Run normal segmentation
        process_all_rois(general_config, mask_config)