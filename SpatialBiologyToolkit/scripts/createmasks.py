# Standard library imports
import logging
from pathlib import Path

# Third-party library imports
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, expand_labels
from skimage.transform import resize
from cellpose import models, denoise
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import shared utilities and configurations
from .config_and_utils import *

def overlay_mask_and_save(
        image,
        output_path,
        masks_and_colors,
        vmin=0,
        vmax_quantile=0.97,
        outline_alpha=0.8,
):
    """
    Overlays multiple label arrays with different colors on a grayscale image and saves the result.
    Parameters
    ----------
    image : np.ndarray
        The original grayscale image.
    output_path : Path or str
        Where to save the resulting overlay image.
    masks_and_colors : list of (np.ndarray, str)
        A list of (label_array, color_name) pairs.
    vmin : float
        Minimum intensity for normalization.
    vmax_quantile : float
        Quantile to determine max intensity cutoff.
    outline_alpha : float
        Transparency for the mask outlines.
    """
    # Determine normalization factor
    vmax = np.quantile(image, vmax_quantile)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)

    # Plot the base image
    plt.figure(figsize=(10, 10))
    plt.imshow(normalized_image, cmap="gray", interpolation="none")

    # Overlay each mask's boundaries
    for label_array, color in masks_and_colors:
        boundaries = find_boundaries(label_array, mode="outer")
        cmap = ListedColormap([[0, 0, 0, 0], plt.cm.colors.to_rgba(color)])
        plt.imshow(boundaries, cmap=cmap, alpha=outline_alpha, interpolation="none")

    plt.axis("off")
    # Save the figure as a high-resolution PNG
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close()

def create_masks(general_config: GeneralConfig, mask_config: CreateMasksConfig):
    """
    Creates cell segmentation masks using Cellpose models, optionally performing deblur and upscale steps.

    Steps:
    1. Load images from denoised_images_folder (or raw images, if desired in future).
    2. Optionally run a deblur model if run_deblur=True.
    3. Optionally run an upscale model if run_upscale=True.
    4. Run the chosen Cellpose model on the resulting image.
    5. Generate final masks, filter by size, and save results.
    6. (Optional) Generate QC overlay images showing kept and excluded objects.

    Parameters
    ----------
    general_config : GeneralConfig
        Contains general pipeline paths and settings.
    mask_config : CreateMasksConfig
        Contains parameters specific to mask creation, including model usage and QC options.
    """
    logging.info("Starting mask creation.")

    # Extract paths and ensure directories exist
    image_folder = Path(general_config.denoised_images_folder)
    output_dir = Path(general_config.masks_folder)
    qc_dir = Path(general_config.qc_folder)
    output_dir.mkdir(exist_ok=True, parents=True)
    qc_dir.mkdir(exist_ok=True, parents=True)

    # Determine which ROIs to process
    specific_rois = mask_config.specific_rois
    if not specific_rois:
        # If no specific ROIs given, process all directories in image_folder
        specific_rois = [entry.name for entry in image_folder.iterdir() if entry.is_dir()]

    # Convert diameter to float (just in case)
    diameter = float(mask_config.cellpose_cell_diameter)

    # Initialize models
    # Deblur and upscale models only run if their flags are True
    # cp_model runs regardless, as this step is required
    if mask_config.run_deblur:
        deblur_model = denoise.DenoiseModel(model_type='deblur_nuclei', gpu=True)
    else:
        deblur_model = None

    if mask_config.run_upscale:
        upscale_model = denoise.DenoiseModel(model_type='upsample_nuclei', gpu=True)
    else:
        upscale_model = None

    cp_model = models.CellposeModel(model_type=mask_config.cell_pose_model, gpu=True)

    roi_data = []

    for roi in specific_rois:
        logging.info(f"Processing ROI: {roi}")

        try:
            roi_path = image_folder / roi
            # Load the specified DNA image
            DNA_image_file = get_filename(roi_path, mask_config.dna_image_name)
            img = io.imread(roi_path / DNA_image_file)

            # Start with the original image
            current_image = img

            # If run_deblur is True, deblur the image
            if mask_config.run_deblur:
                current_image = deblur_model.eval(current_image, channels=None, diameter=diameter, batch_size=16)

            # If run_upscale is True, upscale the (deblurred or original) image
            if mask_config.run_upscale:
                current_image = upscale_model.eval(current_image, channels=None, diameter=diameter, batch_size=16)

            # Run the Cellpose model on the current image (deblurred/upscaled as per config)
            mask, _, _ = cp_model.eval(current_image,
                                                 diameter=diameter * mask_config.upscale_ratio,
                                                 channels=None,
                                                 batch_size=16,
                                                 normalize={'normalize': mask_config.image_normalise, 'percentile': mask_config.image_normalise_percentile},
                                                 cellprob_threshold=mask_config.cellprob_threshold)

            # Resize the mask back to the original image shape if changed by upscale
            if mask_config.run_upscale:
                mask = resize(mask, img.shape, order=0, preserve_range=True, anti_aliasing=False)

            # Optionally expand masks
            if mask_config.expand_masks:
                mask = expand_labels(mask, distance=mask_config.expand_masks)

            # Filter objects by size
            final_mask = np.zeros_like(mask, dtype=np.uint16)
            excluded_mask = np.zeros_like(mask, dtype=np.uint16)
            region_props = regionprops(mask)
            total_objects = len(region_props)
            kept_objects = 0

            for region in region_props:
                area = region.area
                # Check if area is within acceptable range
                if ((mask_config.min_cell_area is None or area >= mask_config.min_cell_area) and
                        (mask_config.max_cell_area is None or area <= mask_config.max_cell_area)):
                    kept_objects += 1
                    final_mask[mask == region.label] = region.label
                else:
                    excluded_mask[mask == region.label] = region.label

            # Save the final mask
            mask_path = output_dir / f"{roi}.tiff"
            io.imsave(mask_path, final_mask)

            # Optionally create QC overlays
            if mask_config.perform_qc:
                qc_overlay_dir = qc_dir / 'Segmentation_overlay'
                qc_overlay_dir.mkdir(exist_ok=True, parents=True)
                overlay_mask_and_save(
                    image=img,
                    output_path=qc_overlay_dir / f"{roi}.png",
                    masks_and_colors=[(final_mask, 'green'), (excluded_mask, 'red')]
                )

            # Compute and record stats about the segmentation
            pixels_per_mm2 = 1e6
            mask_area_mm2 = final_mask.shape[0] * final_mask.shape[1] / pixels_per_mm2
            objects_per_mm2 = kept_objects / mask_area_mm2

            roi_data.append({
                'ROI': roi,
                'Total cells': total_objects,
                'Kept cells': kept_objects,
                'Excluded cells': total_objects - kept_objects,
                'Cells per mm2': objects_per_mm2
            })

        except Exception as e:
            logging.error(f"Error processing ROI: {roi}. Exception: {str(e)}", exc_info=True)
            continue

    # Save a summary CSV file with stats for all ROIs
    roi_df = pd.DataFrame(roi_data)
    roi_df_path = qc_dir / 'Segmentation_QC.csv'
    roi_df.to_csv(roi_df_path, index=False)
    logging.info(f"Saved ROI analysis to {roi_df_path}")
    logging.info("Mask creation completed.")


if __name__ == "__main__":
    # Define the pipeline stage for logging
    pipeline_stage = 'CreateMasks'

    # Load configuration and apply any command line overrides
    config = process_config_with_overrides()

    # Setup logging using the configuration
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Extract general and masking configurations
    general_config = GeneralConfig(**config.get('general', {}))
    mask_config = CreateMasksConfig(**config.get('createmasks', {}))

    # Create the segmentation masks
    create_masks(general_config, mask_config)
