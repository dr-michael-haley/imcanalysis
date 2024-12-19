# These have to be first, if not GPU acceleration won't work on some systems
import torch
from cellpose import models, denoise

import numpy as np
import pandas as pd
import random
import logging
import os
from pathlib import Path
from skimage import io as skio
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, expand_labels
from skimage.morphology import binary_dilation
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import seaborn as sb

from .config_and_utils import (
    process_config_with_overrides,
    setup_logging,
    GeneralConfig,
    CreateMasksConfig,
    get_filename,
    cleanstring
)


def create_overlay_image(
        image: np.ndarray,
        boundary_dilation: int = 0,
        masks_and_colors: list = None,
        vmin: float = 0,
        vmax_quantile: float = 0.97,
        outline_alpha: float = 0.8
) -> np.ndarray:
    """
    Create an overlay of segmentation masks on a grayscale image and return it as an RGB array.
    """
    if masks_and_colors is None:
        masks_and_colors = []

    vmax = np.quantile(image, vmax_quantile)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(normalized_image, cmap="gray", interpolation="none")
    ax.axis("off")

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
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img_array


def interesting_patch(mask, bsize=130):
    """
    ADAPTED FROM CELLPOSE

    Get patch of size bsize x bsize from the mask that contains the densest concentration of cells.
    """
    Ly, Lx = mask.shape
    m = np.float32(mask > 0)
    m = gaussian_filter(m, bsize / 2)
    y, x = np.unravel_index(np.argmax(m), m.shape)
    ycent = max(bsize // 2, min(y, Ly - bsize // 2))
    xcent = max(bsize // 2, min(x, Lx - bsize // 2))
    yinds = np.arange(ycent - bsize // 2, ycent + bsize // 2, 1, int)
    xinds = np.arange(xcent - bsize // 2, xcent + bsize // 2, 1, int)
    return (yinds, xinds)


def segment_single_roi(
    roi: str,
    image_folder: Path,
    qc_dir: Path,
    mask_config: CreateMasksConfig,
    parameter_set: dict = None,
    for_parameter_scan: bool = False,
    deblur_model=None,
    upscale_model=None,
    cp_model=None
) -> dict:
    """
    Segment a single ROI with given parameters, returning segmentation stats and QC image array.

    The entire image is always processed at full size. No window cropping occurs here.
    Window cropping (if desired) happens later in the parameter scan for display only.

    Returns
    -------
    dict:
        Dictionary with segmentation metrics, parameters, final masks, and QC image array (full image).
    """
    logging.info(f"Processing ROI: {roi}")

    # Merge parameter overrides with mask_config
    run_deblur = parameter_set.get('run_deblur', mask_config.run_deblur) if parameter_set else mask_config.run_deblur
    run_upscale = parameter_set.get('run_upscale', mask_config.run_upscale) if parameter_set else mask_config.run_upscale
    cell_pose_model = parameter_set.get('cell_pose_model', mask_config.cell_pose_model) if parameter_set else mask_config.cell_pose_model
    cellprob_threshold = parameter_set.get('cellprob_threshold', mask_config.cellprob_threshold) if parameter_set else mask_config.cellprob_threshold
    flow_threshold = parameter_set.get('flow_threshold', mask_config.flow_threshold) if parameter_set else mask_config.flow_threshold
    image_normalise = parameter_set.get('image_normalise', mask_config.image_normalise) if parameter_set else mask_config.image_normalise
    image_normalise_percentile_lower = parameter_set.get('image_normalise_percentile_lower', mask_config.image_normalise_percentile_lower) if parameter_set else mask_config.image_normalise_percentile_lower
    image_normalise_percentile_upper = parameter_set.get('image_normalise_percentile_upper', mask_config.image_normalise_percentile_upper) if parameter_set else mask_config.image_normalise_percentile_upper
    diameter = float(parameter_set.get('cellpose_cell_diameter', mask_config.cellpose_cell_diameter) if parameter_set else mask_config.cellpose_cell_diameter)

    roi_path = image_folder / roi
    DNA_image_file = get_filename(roi_path, mask_config.dna_image_name)
    img = skio.imread(roi_path / DNA_image_file)

    # Use provided models if they exist, otherwise create them here
    if run_deblur and deblur_model is None:
        deblur_model = denoise.DenoiseModel(model_type='deblur_nuclei', gpu=True)
    if run_upscale and upscale_model is None:
        upscale_model = denoise.DenoiseModel(model_type='upsample_nuclei', gpu=True)
    if cp_model is None:
        cp_model = models.CellposeModel(model_type=cell_pose_model, gpu=True)

    # Preprocessing on the full image
    current_image = img.copy()
    if run_deblur:
        current_image = deblur_model.eval(current_image, channels=None, diameter=diameter, batch_size=16)
    if run_upscale:
        current_image = upscale_model.eval(current_image, channels=None, diameter=diameter, batch_size=16)

    # Run Cellpose
    mask, _, _ = cp_model.eval(
        current_image,
        diameter=diameter * mask_config.upscale_ratio,
        channels=None,
        batch_size=16,
        normalize={'normalize': image_normalise, 'percentile': [image_normalise_percentile_lower, image_normalise_percentile_upper]},
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold
    )

    # If we upscaled, reduce mask back to original image size
    if run_upscale:
        mask = resize(mask, img.shape, order=0, preserve_range=True, anti_aliasing=False)

    if mask_config.expand_masks:
        mask = expand_labels(mask, distance=mask_config.expand_masks)

    region_props = regionprops(mask)
    total_objects = len(region_props)
    final_mask = np.zeros_like(mask, dtype=np.uint16)
    excluded_mask = np.zeros_like(mask, dtype=np.uint16)
    kept_objects = 0

    for region in region_props:
        area = region.area
        if ((mask_config.min_cell_area is None or area >= mask_config.min_cell_area) and
            (mask_config.max_cell_area is None or area <= mask_config.max_cell_area)):
            kept_objects += 1
            final_mask[mask == region.label] = region.label
        else:
            excluded_mask[mask == region.label] = region.label

    pixels_per_mm2 = 1e6
    mask_area_mm2 = final_mask.shape[0] * final_mask.shape[1] / pixels_per_mm2
    objects_per_mm2 = kept_objects / mask_area_mm2 if mask_area_mm2 > 0 else 0

    qc_image_array = None
    qc_image_path_str = None
    if mask_config.perform_qc:
        # Create full QC overlay from current_image
        qc_image_array = create_overlay_image(
            image=img,
            boundary_dilation=mask_config.qc_boundary_dilation,
            masks_and_colors=[(final_mask, 'green'), (excluded_mask, 'red')],
            vmin=0,
            vmax_quantile=0.97,
            outline_alpha=0.8
        )

        if not for_parameter_scan:
            # Save QC image if not parameter scanning
            qc_overlay_dir = qc_dir / 'Segmentation_overlay'
            qc_overlay_dir.mkdir(exist_ok=True, parents=True)
            qc_image_path = qc_overlay_dir / f"{roi}.png"
            plt.imsave(qc_image_path, qc_image_array)
            qc_image_path_str = str(qc_image_path)

    result = {
        'ROI': roi,
        'Total cells': total_objects,
        'Kept cells': kept_objects,
        'Excluded cells': total_objects - kept_objects,
        'Cells per mm2': objects_per_mm2,
        'Run Deblur': run_deblur,
        'Run Upscale': run_upscale,
        'Cell Pose Cell Diameter': diameter,
        'Cell Pose Model': cell_pose_model,
        'CellProb Threshold': cellprob_threshold,
        'Flow Threshold': flow_threshold,
        'Image Normalize': image_normalise,
        'Image Normalize Percentile Lower': image_normalise_percentile_lower,
        'Image Normalize Percentile Upper': image_normalise_percentile_upper,
        'Final Mask': final_mask,
        'Excluded Mask': excluded_mask,
        'QC Image Path': qc_image_path_str,
        'QC Image Array': qc_image_array  # full-size QC image array
    }

    return result


def process_roi_list(general_config: GeneralConfig, mask_config: CreateMasksConfig, deblur_model, upscale_model, cp_model):
    """
    Normal mode: process ROIs using the current mask_config and general_config, saving masks and QC images.
    """
    logging.info("Starting mask creation (process_roi_list).")
    image_folder = Path(general_config.denoised_images_folder)
    qc_dir = Path(general_config.qc_folder)
    output_dir = Path(general_config.masks_folder)
    output_dir.mkdir(exist_ok=True, parents=True)
    qc_dir.mkdir(exist_ok=True, parents=True)

    # Determine ROIs
    specific_rois = mask_config.specific_rois
    if not specific_rois:
        specific_rois = [entry.name for entry in image_folder.iterdir() if entry.is_dir()]

    roi_data = []
    for roi in specific_rois:
        try:
            result = segment_single_roi(
                roi=roi,
                image_folder=image_folder,
                qc_dir=qc_dir,
                mask_config=mask_config,
                parameter_set=None,
                for_parameter_scan=False,
                deblur_model=deblur_model,
                upscale_model=upscale_model,
                cp_model=cp_model
            )
            roi_data.append(result)

            # Save final mask
            final_mask = result['Final Mask']
            mask_path = output_dir / f"{roi}.tiff"
            skio.imsave(mask_path, final_mask)

        except Exception as e:
            logging.error(f"Error processing ROI: {roi}. Exception: {str(e)}", exc_info=True)
            continue

    roi_df = pd.DataFrame(roi_data)
    roi_df_path = qc_dir / 'Segmentation_QC.csv'
    roi_df.to_csv(roi_df_path, index=False)
    logging.info(f"Saved ROI analysis to {roi_df_path}")
    logging.info("Mask creation completed.")


def parameter_scan_two_params(general_config: GeneralConfig, mask_config: CreateMasksConfig, deblur_model, upscale_model, cp_model):
    """
    Parameter scan mode: run multiple parameter sets defined by two parameters (param_a, param_b),
    each with a list of values. Create a grid of QC images showing a windowed portion of the QC image
    arrays for visual comparison.

    Steps:
    1. Determine ROIs to scan.
    2. Build parameter sets from param_a_values x param_b_values.
    3. Segment full image each time.
    4. Identify a patch of high cell density from the first parameter set.
    5. Crop QC images from all parameter sets using that patch if window_size is set.
    6. Arrange QC images in a parameter grid and save results.
    """
    logging.info("Starting parameter scan with two parameters.")

    param_a = mask_config.param_a
    param_a_values = mask_config.param_a_values or []
    param_b = mask_config.param_b
    param_b_values = mask_config.param_b_values or []

    image_folder = Path(general_config.denoised_images_folder)
    qc_dir = Path(os.path.join(general_config.qc_folder, f'ParameterScan_{cleanstring(param_a)}_{cleanstring(param_b)}'))
    qc_dir.mkdir(exist_ok=True, parents=True)

    # Determine ROIs
    all_rois = [entry.name for entry in image_folder.iterdir() if entry.is_dir()]
    if mask_config.scan_rois:
        scan_rois = mask_config.scan_rois
    else:
        scan_rois = random.sample(all_rois, mask_config.num_rois_to_scan)

    # Construct parameter grid
    param_sets = []
    for a_val in param_a_values:
        for b_val in param_b_values:
            param_sets.append({param_a: a_val, param_b: b_val})

    all_results = []

    for roi in scan_rois:
        logging.info(f"Parameter scanning on ROI: {roi}")
        results_for_roi = []
        qc_images = []  # store arrays for each parameter set (full-sized arrays)

        patch_coords = None

        for i, pset in enumerate(param_sets):
            logging.info(f'Iteration {str(i)}: {str(pset)}')
            result = segment_single_roi(
                roi=roi,
                image_folder=image_folder,
                qc_dir=qc_dir,
                mask_config=mask_config,
                parameter_set=pset,
                for_parameter_scan=True,
                deblur_model=deblur_model,
                upscale_model=upscale_model,
                cp_model=cp_model
            )
            results_for_roi.append(result)

            full_qc_array = result['QC Image Array']

            if i == 0 and mask_config.window_size:
                # Determine interesting patch from the final_mask of the first result
                final_mask = result['Final Mask']
                yinds, xinds = interesting_patch(final_mask, bsize=mask_config.window_size)
                patch_coords = (yinds, xinds)

            if full_qc_array is not None:
                # If window_size is specified, crop the QC array here using patch_coords
                if mask_config.window_size and patch_coords is not None:
                    yinds, xinds = patch_coords
                    cropped = full_qc_array[yinds[:, None], xinds, :]
                    qc_images.append(cropped)
                else:
                    qc_images.append(full_qc_array)

        # Save results for this ROI
        df = pd.DataFrame(results_for_roi)
        param_scan_csv = qc_dir / f'Parameter_Scan_{roi}.csv'
        df.to_csv(param_scan_csv, index=False)
        logging.info(f"Saved parameter scan results for ROI {roi} to {param_scan_csv}")

        # Create QC image grid (rows=param_a_values, cols=param_b_values)
        M = len(param_a_values)
        N = len(param_b_values)
        if qc_images:
            fig, axes = plt.subplots(M, N, figsize=(4*N,4*M))
            if M == 1 and N == 1:
                axes = [[axes]]
            elif M == 1:
                axes = [axes]
            elif N == 1:
                axes = [[ax] for ax in axes]

            for i, a_val in enumerate(param_a_values):
                for j, b_val in enumerate(param_b_values):
                    idx = i*N + j
                    img_arr = qc_images[idx]
                    ax = axes[i][j]
                    ax.imshow(img_arr)
                    ax.axis('off')
                    ax.set_title(f"{param_a}={a_val}, {param_b}={b_val}")

            fig.tight_layout()
            qc_grid_path = qc_dir / f'Parameter_Scan_{roi}_grid.png'
            fig.savefig(qc_grid_path, dpi=mask_config.dpi_qc_images)
            plt.close(fig)
            logging.info(f"Saved parameter scan QC grid for ROI {roi} to {qc_grid_path}")

        all_results.extend(results_for_roi)

    # Combined results
    combined_df = pd.DataFrame(all_results)
    combined_csv = qc_dir / 'Parameter_Scan_All.csv'
    combined_df.to_csv(combined_csv, index=False)
    logging.info(f"Saved combined parameter scan results to {combined_csv}")

    # Save bargraph of results for number of cells captured
    param_to_column = {'cellpose_cell_diameter': 'Cell Pose Cell Diameter',
                     'cell_pose_model': 'Cell Pose Model',
                     'cellprob_threshold': 'CellProb Threshold',
                     'run_deblur': 'Run Deblur',
                     'run_upscale': 'Run Upscale',
                     'flow_threshold': 'Flow Threshold',
                     'image_normalise': 'Image Normalize',
                     'image_normalise_percentile_lower': 'Image Normalize Percentile Lower',
                     'image_normalise_percentile_upper': 'Image Normalize Percentile Upper'}

    for m in ['Cells per mm2', 'Excluded cells']:
        ax = sb.barplot(data=combined_df,
                        y=m,
                        x=param_to_column[param_a],
                        hue=param_to_column[param_b],
                        palette='tab20')

        sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        fig = ax.get_figure()
        fig.savefig(qc_dir / f"Parameterscan_{m}.png", bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    pipeline_stage = 'CreateMasks'
    config = process_config_with_overrides()
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Check GPU availability
    logging.info(f'GPU available?: {str(torch.cuda.is_available())}')

    general_config = GeneralConfig(**config.get('general', {}))
    mask_config = CreateMasksConfig(**config.get('createmasks', {}))

    # Initialize models once
    if mask_config.run_deblur:
        deblur_model = denoise.DenoiseModel(model_type='deblur_nuclei', gpu=True)
    else:
        deblur_model = None

    if mask_config.run_upscale:
        upscale_model = denoise.DenoiseModel(model_type='upsample_nuclei', gpu=True)
    else:
        upscale_model = None

    cp_model = models.CellposeModel(model_type=mask_config.cell_pose_model, gpu=True)

    # Decide mode based on run_parameter_scan and param fields
    if (mask_config.run_parameter_scan and
        mask_config.param_a and mask_config.param_a_values and
        mask_config.param_b and mask_config.param_b_values):
        parameter_scan_two_params(general_config, mask_config, deblur_model, upscale_model, cp_model)
    else:
        process_roi_list(general_config, mask_config, deblur_model, upscale_model, cp_model)