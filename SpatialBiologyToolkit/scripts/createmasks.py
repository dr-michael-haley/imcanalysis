# Standard library imports
import logging
from pathlib import Path

# Third-party library imports
import numpy as np
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
    Overlays multiple label arrays with different colors on a grayscale image.
    """
    vmax = np.quantile(image, vmax_quantile)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(normalized_image, cmap="gray", interpolation="none")

    for label_array, color in masks_and_colors:
        boundaries = find_boundaries(label_array, mode="outer")
        cmap = ListedColormap([[0, 0, 0, 0], plt.cm.colors.to_rgba(color)])
        plt.imshow(boundaries, cmap=cmap, alpha=outline_alpha, interpolation="none")

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close()

def create_masks(image_folder='processed',
                 output_dir='masks',
                 qc_dir='QC',
                 specific_rois=None,
                 dna_image_name='DNA1',
                 diameter=10.0,
                 upscale_ratio=1.7,
                 expand_masks=2,
                 perform_qc=True,
                 min_size=None,
                 max_size=None):
    """
    Creates masks of cells in the specified folder using Cellpose models.
    """
    logging.info("Starting mask creation.")

    if isinstance(diameter, int):
        diameter = float(diameter)

    image_folder = Path(image_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(exist_ok=True, parents=True)

    if not specific_rois:
        specific_rois = [entry.name for entry in image_folder.iterdir() if entry.is_dir()]

    deblur = denoise.DenoiseModel(model_type='deblur_nuclei', gpu=True)
    upscale = denoise.DenoiseModel(model_type='upsample_nuclei', gpu=True)
    cyto3 = models.CellposeModel(model_type='nuclei', gpu=True)

    roi_data = []

    for roi in specific_rois:
        logging.info(f"Processing ROI: {roi}")

        try:
            roi_path = image_folder / roi
            DNA_image_file = get_filename(roi_path, dna_image_name)
            img = io.imread(roi_path / DNA_image_file)

            img_deblur = deblur.eval(img, channels=None, diameter=diameter, batch_size=16)
            img_upscale = upscale.eval(img_deblur, channels=None, diameter=diameter, batch_size=16)
            masks_upscaled, _, _ = cyto3.eval(img_upscale,
                                              diameter=diameter * upscale_ratio,
                                              channels=None,
                                              batch_size=16,
                                              normalize={'normalize':True, 'percentile':[0,97]},
                                              cellprob_threshold=-1)
            mask = resize(masks_upscaled, img.shape, order=0, preserve_range=True, anti_aliasing=False)

            if expand_masks:
                mask = expand_labels(mask, distance=expand_masks)

            final_mask = np.zeros_like(mask, dtype=np.uint16)
            excluded_mask = np.zeros_like(mask, dtype=np.uint16)
            region_props = regionprops(mask)
            total_objects = len(region_props)
            kept_objects = 0

            for region in region_props:
                area = region.area
                if (min_size is None or area >= min_size) and (max_size is None or area <= max_size):
                    kept_objects += 1
                    final_mask[mask == region.label] = region.label
                else:
                    excluded_mask[mask == region.label] = region.label

            mask_path = output_dir / f"{roi}.tiff"
            io.imsave(mask_path, final_mask)

            if perform_qc:
                qc_overlay_dir = qc_dir / 'Segmentation_overlay'
                qc_overlay_dir.mkdir(exist_ok=True, parents=True)
                overlay_mask_and_save(
                    image=img,
                    output_path=qc_overlay_dir / f"{roi}.png",
                    masks_and_colors=[(final_mask, 'green'), (excluded_mask, 'red')]
                )

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

    roi_df = pd.DataFrame(roi_data)
    roi_df_path = qc_dir / 'Segmentation_QC.csv'
    roi_df.to_csv(roi_df_path, index=False)
    logging.info(f"Saved ROI analysis to {roi_df_path}")
    logging.info("Mask creation completed.")

if __name__ == "__main__":
    # Define the pipeline stage
    pipeline_stage = 'CreateMasks'

    # Load configuration
    config = process_config_with_overrides()

    # Setup logging
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**config.get('general', {}))
    mask_config = CreateMasksConfig(**config.get('createmasks', {}))

    # Create masks using CellPose
    create_masks(
        image_folder=general_config.denoised_images_folder,
        output_dir=general_config.masks_folder,
        qc_dir=general_config.qc_folder,
        specific_rois=mask_config.specific_rois,
        dna_image_name=mask_config.dna_image_name,
        diameter=mask_config.cellpose_cell_diameter,
        upscale_ratio=mask_config.upscale_ratio,
        expand_masks=mask_config.expand_masks,
        perform_qc=mask_config.perform_qc,
        min_size=mask_config.min_cell_area,
        max_size=mask_config.max_cell_area
    )