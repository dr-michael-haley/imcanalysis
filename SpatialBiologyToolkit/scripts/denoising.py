# denoising.py

"""
Module for denoising images using DeepSNF and DIMR methods.

This module provides functions to denoise images using the Deep Self Normalizing Flow (DeepSNF)
and the Denoising Iterative Mean Replacement (DIMR) methods. It includes utilities to load images,
perform denoising in batch mode using parameters from a configuration file, and test GPU availability.
Logging functionality is integrated to track the processing steps and errors.

Dependencies:
- IMC_Denoise package (https://github.com/PENGLU-WashU/IMC_Denoise)
- TensorFlow
- tifffile
- logging

Functions:
- load_single_img
- load_imgs_from_directory
- denoise_batch
- gpu_test
"""

# Standard library imports
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from shutil import rmtree
from glob import glob
import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Third-party library imports
import tifffile as tp
import psutil

# Import shared utilities and configurations
from .config_and_utils import *

# Import IMC_Denoise methods
from IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF

# Additional imports
import tensorflow as tf

def load_single_img(filename, quiet=True):
    """
    Load a single image from a file.
    """
    filename = str(filename)
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        img_in = tp.imread(filename).astype('float32')
        if not quiet:
            logging.debug(f'Loaded image: {filename}')
    else:
        logging.error(f'File {filename} does not end with .tiff or .tif')
        raise ValueError('Raw file should end with .tiff or .tif!')
    if img_in.ndim != 2:
        logging.error(f'Image {filename} is not 2D')
        raise ValueError('Single image should be 2D!')
    return img_in

def load_imgs_from_directory(load_directory, channel_name, quiet=True):
    """
    Load images for a specific channel from a directory.
    """
    img_collect = []
    img_folders = glob(os.path.join(str(load_directory), "*", ""))
    img_file_list = []
    matched_folders = []  # Only folders that contain the channel

    if not quiet:
        print('Loading image data from directories...\n')
        logging.info('Loading image data from directories.')

    for sub_img_folder in img_folders:
        img_list = [
            f for f in os.listdir(sub_img_folder)
            if os.path.isfile(os.path.join(sub_img_folder, f)) and (f.endswith(".tiff") or f.endswith(".tif"))
        ]
        for img_file in img_list:
            if channel_name.lower() in img_file.lower():
                img_path = os.path.join(sub_img_folder, img_file)
                img_read = load_single_img(img_path, quiet=quiet)

                if not quiet:
                    print(img_path)
                    logging.debug(f'Loaded image for channel {channel_name}: {img_path}')

                img_file_list.append(img_file)
                img_collect.append(img_read)
                matched_folders.append(sub_img_folder)  # Add folder only when channel is found
                break  # Only one image per channel per folder

    if not img_collect:
        logging.error(f'No images found for channel "{channel_name}".')
        raise ValueError(f'No images found for channel "{channel_name}". Please check the channel name!')

    if not quiet:
        print('\nImage data loading completed!')
        logging.info(f'Found {len(img_collect)} images for channel "{channel_name}" in {len(matched_folders)} ROIs.')

    return img_collect, img_file_list, matched_folders

def denoise_batch(
    general_config: GeneralConfig,
    denoise_config: DenoisingConfig
):
    """
    Denoise images in batch using the specified method from the DenoisingConfig.
    """
    # Configure TensorFlow/Keras training output verbosity
    if not denoise_config.verbose_training:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
        tf.get_logger().setLevel('ERROR')  # Only show errors
    
    logging.info('Starting denoise_batch function.')
    logging.info(f'DenoisingConfig parameters: {denoise_config}')
    logging.info(f'GeneralConfig parameters: {general_config}')

    # Extract paths from general_config
    raw_directory = general_config.raw_images_folder

    method = denoise_config.method
    channels = denoise_config.channels

    if channels == []:
        channels = load_channels_from_panel(general_config)

    # Create output directory
    processed_output_dir = Path(general_config.denoised_images_folder)
    processed_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Processed output directory set to: {processed_output_dir}')

    # Error tracking lists
    error_channels = []
    completed_channels = []

    # Boolean as to whether images exist in processed directory
    processsed_images_exist = any(f.lower().endswith('.tiff') for _, __, files in os.walk('processed') for f in files)

    if processsed_images_exist:
        logging.info(f'Previously denoised .tiff images detected in output directory {processed_output_dir}')

        # If existing images found, we can infer those channels are denoised, and so skip denoising them again
        if denoise_config.skip_already_denoised:
            # Get a list of all denoised images from first ROI

            denoised_images = os.listdir(os.path.join(processed_output_dir, os.listdir(processed_output_dir)[0]))
            already_denoised_channels = [x.split('_', maxsplit=2)[-1].replace('.tiff','') for x in denoised_images]
            skipped_channels = [x for x in channels if x in already_denoised_channels]
            logging.info(f'Denoising skipped on channels already denoised: {skipped_channels}')
            channels = [x for x in channels if x not in skipped_channels]

    print(f'\nPerforming denoising using method "{method}" on the following channels:\n{channels}\n')
    logging.info(f'Performing denoising using method "{method}" on channels: {channels}')

    # Check if parameter scanning is enabled
    if denoise_config.run_parameter_scan:
        if not denoise_config.scan_parameter or not denoise_config.scan_values:
            logging.error("Parameter scanning enabled but scan_parameter or scan_values not specified")
            raise ValueError("Must specify both scan_parameter and scan_values when run_parameter_scan=True")
        
        scan_param = denoise_config.scan_parameter
        scan_values = denoise_config.scan_values
        logging.info(f"Parameter scan enabled: varying '{scan_param}' with values {scan_values}")
        print(f"\nParameter scan mode: testing '{scan_param}' = {scan_values}\n")
    else:
        # No scanning - run once with None as scan identifier
        scan_param = None
        scan_values = [None]

    # Loop over scan values (or just once if not scanning)
    for scan_value in scan_values:
        # Create scan-specific subfolder suffix
        if scan_param is not None:
            scan_suffix = f"{scan_param}_{scan_value}"
            logging.info(f"\n{'='*60}")
            logging.info(f"Starting parameter scan iteration: {scan_param} = {scan_value}")
            logging.info(f"{'='*60}\n")
            print(f"\n{'='*60}")
            print(f"Parameter scan: {scan_param} = {scan_value}")
            print(f"{'='*60}\n")
            
            # Update the processed output directory to include scan suffix
            scan_processed_output_dir = Path(f"{general_config.denoised_images_folder}_{scan_suffix}")
            scan_processed_output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Scan output directory: {scan_processed_output_dir}")
        else:
            scan_suffix = None
            scan_processed_output_dir = processed_output_dir

        # Override the parameter value if scanning
        if scan_param is not None:
            # Create a modified config with the scan value
            setattr(denoise_config, scan_param, scan_value)
            logging.info(f"Override config: {scan_param} = {scan_value}")

        # DeepSNF parameters from denoise_config (potentially modified by scan)
        patch_step_size = denoise_config.patch_step_size
        train_epochs = denoise_config.train_epochs
        train_initial_lr = denoise_config.train_initial_lr
        train_batch_size = denoise_config.train_batch_size
        pixel_mask_percent = denoise_config.pixel_mask_percent
        val_set_percent = denoise_config.val_set_percent
        loss_function = denoise_config.loss_function
        loss_name = denoise_config.loss_name
        weights_save_directory = denoise_config.weights_save_directory
        is_load_weights = denoise_config.is_load_weights
        lambda_HF = denoise_config.lambda_HF
        n_neighbours = denoise_config.n_neighbours
        n_iter = denoise_config.n_iter
        window_size = denoise_config.window_size
        network_size = denoise_config.network_size
        ratio_thresh = denoise_config.ratio_thresh
        truncated_max_rate = denoise_config.truncated_max_rate

        for channel_name in channels:
            try:
                logging.info(f'Starting processing for channel: {channel_name}')
                # Load images for the channel
                img_collect, img_file_list, matched_folders = load_imgs_from_directory(raw_directory, channel_name)
                logging.info(f'Loaded {len(img_collect)} images for channel {channel_name}')

                if method == 'deep_snf':

                    logging.info(f'DeepSNF parameters: {denoise_config}')

                    # Training settings
                    row_step = patch_step_size
                    col_step = patch_step_size

                    # Generate patches and train model if not loading weights
                    if not is_load_weights:
                        current_step = patch_step_size
                        min_step = denoise_config.intelligent_patch_size_minimum
                        min_patches = denoise_config.intelligent_patch_size_min_patches
                        max_patches = denoise_config.intelligent_patch_size_max_patches

                        while True:
                            logging.info(f"Trying patch_step_size={current_step} with ratio_thresh={ratio_thresh}")
                            data_generator = DeepSNiF_DataGenerator(
                                channel_name=channel_name,
                                n_neighbours=n_neighbours,
                                n_iter=n_iter,
                                window_size=window_size,
                                col_step=current_step,
                                row_step=current_step,
                                ratio_thresh=ratio_thresh
                            )
                            generated_patches = data_generator.generate_patches_from_directory(load_directory=raw_directory)
                            patch_count = generated_patches.shape[0]

                            logging.info(f"Generated {patch_count} patches for {channel_name} with patch_step_size={current_step}")

                            if not denoise_config.intelligent_patch_size:
                                # Not using intelligent sizing, accept whatever we got
                                break

                            # Check if we have enough patches
                            if patch_count >= min_patches:
                                # Check if we're within max limit (if specified)
                                if max_patches is None or patch_count <= max_patches:
                                    logging.info(f"Patch count {patch_count} meets requirements (min={min_patches}, max={max_patches})")
                                    break
                                else:
                                    # Too many patches, increase step size
                                    logging.info(f"Patch count {patch_count} exceeds maximum ({max_patches}). Increasing patch_step_size.")
                                    current_step += 20
                                    continue
                            
                            # Not enough patches, try smaller step size
                            if current_step <= min_step:
                                logging.warning(
                                    f"Reached minimum patch_step_size ({min_step}). Proceeding with {patch_count} patches "
                                    f"(below minimum of {min_patches}).")
                                break

                            logging.info(
                                f"Patch count {patch_count} below minimum ({min_patches}) — reducing patch_step_size and retrying.")
                            current_step -= 20

                    weights_name = f"weights_{channel_name}.keras"
                    logging.info(f'Weights file name set to: {weights_name}')

                    # Print GPU and RAM availability
                    logging.info(f'RAM check prior to training for {channel_name}')
                    print_memory_status()

                    deepsnf = DeepSNiF(
                        train_epoches=train_epochs,
                        train_learning_rate=train_initial_lr,
                        train_batch_size=train_batch_size,
                        mask_perc_pix=pixel_mask_percent,
                        val_perc=val_set_percent,
                        loss_func=loss_function,
                        weights_name=weights_name,
                        loss_name=loss_name,
                        weights_dir=weights_save_directory,
                        is_load_weights=is_load_weights,
                        lambda_HF=lambda_HF,
                        network_size=network_size,
                        truncated_max_rate=truncated_max_rate
                    )

                    if not is_load_weights:
                        print('Starting training...')
                        logging.info('Starting model training.')
                        # Train the DeepSNF classifier
                        train_loss, val_loss = deepsnf.train(generated_patches)
                        logging.info('Model training completed.')
                    else:
                        print(f'Using weights file: {weights_name}')
                        logging.info(f'Loading existing weights from file: {weights_name}')

                    # Print GPU and RAM availability
                    logging.info(f'RAM check after training for {channel_name}')
                    print_memory_status()

                    # Process images
                    for img, img_file_name, folder in zip(img_collect, img_file_list, matched_folders):
                        # Perform denoising
                        logging.info(f'Denoising image: {img_file_name}')
                        img_denoised = deepsnf.perform_IMC_Denoise(
                            img,
                            n_neighbours=n_neighbours,
                            n_iter=n_iter,
                            window_size=window_size
                        )

                        # Get ROI folder name from the path
                        roi_folder_name = Path(folder).name

                        # Ensure output directory exists
                        output_dir = scan_processed_output_dir / roi_folder_name
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Save the denoised image
                        save_path = output_dir / img_file_name
                        tp.imwrite(save_path, img_denoised.astype('float32'))
                        logging.info(f'Saved denoised image to: {save_path}')

                elif method == 'dimr':
                    # DIMR parameters from denoise_config
                    n_neighbours = denoise_config.n_neighbours
                    n_iter = denoise_config.n_iter
                    window_size = denoise_config.window_size
                    logging.info(f'DIMR parameters: n_neighbours={n_neighbours}, n_iter={n_iter}, window_size={window_size}')

                    # Create DIMR object
                    dimr = DIMR(
                        n_neighbours=n_neighbours,
                        n_iter=n_iter,
                        window_size=window_size
                    )

                    # Process images
                    for img, img_file_name, folder in zip(img_collect, img_file_list, matched_folders):
                        logging.info(f'Denoising image: {img_file_name}')
                        # Perform DIMR denoising
                        img_denoised = dimr.perform_DIMR(img)

                        # Get ROI folder name from the path
                        roi_folder_name = Path(folder).name

                        # Ensure output directory exists
                        output_dir = scan_processed_output_dir / roi_folder_name
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Save the denoised image
                        save_path = output_dir / img_file_name
                        tp.imsave(save_path, img_denoised.astype('float32'))
                        logging.info(f'Saved denoised image to: {save_path}')
                else:
                    logging.error(f"Unknown method '{method}'.")
                    raise ValueError(f"Unknown method '{method}'. Available methods: 'deep_snf', 'dimr'.")

                completed_channels.append(channel_name)
                logging.info(f'Completed processing for channel: {channel_name}')
            except Exception as e:
                print(f"Error in channel {channel_name}: {e}")
                logging.error(f"Error in channel {channel_name}: {e}", exc_info=True)
                error_channels.append(f"{channel_name}: {e}")


        # After processing all channels, compute pixel QC metrics for this scan iteration
        logging.info("Computing QC pixel statistics for denoised images...")

        qc_records = []

        for channel_name in channels:
            try:
                # Load denoised images from scan-specific directory
                pro_imgs, pro_img_names, pro_folders = load_imgs_from_directory(scan_processed_output_dir, channel_name, quiet=True)

                stats = {'channel': channel_name, 'num_images': len(pro_imgs),
                         'mean': [], 'std': [], 'min': [], 'max': []}

                for img in pro_imgs:
                    if img.ndim != 2:
                        continue

                    h, w = img.shape
                    margin_h, margin_w = int(h * 0.2), int(w * 0.2)
                    center = img[margin_h:h - margin_h, margin_w:w - margin_w]

                    stats['mean'].append(np.mean(center))
                    stats['std'].append(np.std(center))
                    stats['min'].append(np.min(center))
                    stats['max'].append(np.max(center))

                stats_out = {
                    'channel': channel_name,
                    'num_images': stats['num_images'],
                    'mean': np.mean(stats['mean']),
                    'std': np.mean(stats['std']),
                    'min': np.mean(stats['min']),
                    'max': np.mean(stats['max']),
                    'flag': 'low_std' if np.mean(stats['std']) < 1 else ''
                }

                qc_records.append(stats_out)
            except Exception as e:
                logging.warning(f"Could not compute QC stats for {channel_name}: {e}")

        # Save CSV report with scan suffix if applicable
        qc_df = pd.DataFrame(qc_records)
        if scan_suffix:
            qc_qc_path = Path(general_config.qc_folder) / f'denoised_pixel_qc_{scan_suffix}.csv'
        else:
            qc_qc_path = Path(general_config.qc_folder) / 'denoised_pixel_qc.csv'
        qc_qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_df.to_csv(qc_qc_path, index=False)
        logging.info(f"Denoised pixel QC report saved to {qc_qc_path}")


        print("\nSuccessfully processed channels:")
        print(completed_channels)
        print("\nChannels with errors:")
        print(error_channels)

        logging.info(f'Successfully processed channels: {completed_channels}')
        if error_channels:
            logging.warning(f'Channels with errors: {error_channels}')

def gpu_test():
    """
    Test if GPU acceleration is enabled for TensorFlow.
    """
    if tf.test.is_built_with_cuda():
        print('GPU acceleration enabled.\n')
        print(tf.config.list_physical_devices('GPU'))
        logging.info('GPU acceleration enabled.')
        logging.info(f'Available GPUs: {tf.config.list_physical_devices("GPU")}')
    else:
        print('GPU not found! Check TensorFlow and CUDA setup.')
        logging.error('GPU not found! Check TensorFlow and CUDA setup.')

def remove_small_rois(general_config: GeneralConfig, denoise_config: DenoisingConfig):
    """
    Remove ROIs that are too small based on the patch_step_size.
    """
    metadata_directory = general_config.metadata_folder
    raw_directory = general_config.raw_images_folder
    patch_step_size = denoise_config.patch_step_size

    metadata_file = Path(metadata_directory) / 'metadata.csv'
    if metadata_file.exists():
        metadata_df = pd.read_csv(metadata_file)

        # Identify ROIs which are too small for denoising
        metadata_df['too_small'] = np.where(
            (metadata_df['height_um'] < patch_step_size) |
            (metadata_df['width_um'] < patch_step_size),
            True,
            False
        )

        # Remove ROIs that are too small
        for _, r in metadata_df.iterrows():
            if r['too_small']:
                roi_folder = Path(raw_directory) / r['unstacked_data_folder']
                if roi_folder.exists():
                    rmtree(roi_folder)
                    logging.warning(f"ROI {r['description']} was smaller than patch_step_size ({patch_step_size}), and was deleted from {str(raw_directory)}.")
    else:
        logging.warning(f"Metadata file '{metadata_file}' not found.")


def qc_check_side_by_side(general_config: GeneralConfig,
                          denoise_config: DenoisingConfig,
                          scan_suffix: Optional[str] = None):
    channels = denoise_config.channels

    if channels == []:
        channels = load_channels_from_panel(general_config)

    # Create folders with scan suffix if applicable
    if scan_suffix:
        save_dir = Path(os.path.join(general_config.qc_folder, f"{denoise_config.qc_image_dir}_{scan_suffix}"))
    else:
        save_dir = Path(os.path.join(general_config.qc_folder, denoise_config.qc_image_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    for channel_name in channels:

        try:

            raw_Img_collect, raw_Img_file_list, raw_img_folders = load_imgs_from_directory(general_config.raw_images_folder, channel_name,
                                                                                           quiet=True)
            # Load from scan-specific directory if scanning
            denoised_dir = f"{general_config.denoised_images_folder}_{scan_suffix}" if scan_suffix else general_config.denoised_images_folder
            pro_Img_collect, pro_Img_file_list, pro_img_folders = load_imgs_from_directory(denoised_dir,
                                                                                           channel_name, quiet=True)

            # Subsample ROIs if qc_num_rois is specified
            if denoise_config.qc_num_rois is not None and denoise_config.qc_num_rois < len(raw_Img_collect):
                num_rois = min(denoise_config.qc_num_rois, len(raw_Img_collect))
                indices = np.random.choice(len(raw_Img_collect), size=num_rois, replace=False)
                raw_Img_collect = [raw_Img_collect[i] for i in indices]
                pro_Img_collect = [pro_Img_collect[i] for i in indices]
                raw_Img_file_list = [raw_Img_file_list[i] for i in indices]
                logging.info(f'QC visualization for {channel_name}: Using {num_rois} random ROIs out of {len(raw_img_folders)} available')
            else:
                logging.info(f'QC visualization for {channel_name}: Using all {len(raw_Img_collect)} ROIs')

            fig, axs = plt.subplots(len(raw_Img_collect), 2, figsize=(10, 5 * len(raw_Img_collect)), dpi=denoise_config.dpi)

            count = 0
            for r_img, p_img, r_img_name in zip(raw_Img_collect, pro_Img_collect, raw_Img_file_list):
                im1 = axs.flat[count].imshow(r_img, vmin=0, vmax=0.5 * np.max(r_img), cmap=denoise_config.colourmap)
                divider = make_axes_locatable(axs.flat[count])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax, orientation='vertical')
                axs.flat[count].set_ylabel(str(r_img_name))
                count = count + 1

                im2 = axs.flat[count].imshow(p_img, vmin=0, vmax=0.5 * np.max(p_img), cmap=denoise_config.colourmap)
                divider = make_axes_locatable(axs.flat[count])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax, orientation='vertical')
                count = count + 1

            fig.savefig(os.path.join(save_dir, channel_name + '.png'))
            plt.close()
            logging.info(f'Denoising QC saved: {channel_name}')

        except Exception as e:
            logging.error(f'Denoising QC error: {channel_name}')
            print(f"Error in channel {channel_name}: {Exception}: {e}")


def load_channels_from_panel(general_config: GeneralConfig,):
    panel = pd.read_csv(os.path.join(general_config.metadata_folder, 'panel.csv'))
    panel['tiff_name'] = panel['channel_name'] + "_" + panel['channel_label']

    channels = panel.loc[
        panel[('to_denoise' if 'to_denoise' in panel.columns else 'use_denoised')], 'tiff_name'].tolist()

    return channels

import re  # Ensure this is at the top of your script

def remove_outliers_from_images(general_config: GeneralConfig,
                          denoise_config: DenoisingConfig):
    """
    Removes outlier pixel values from images based on thresholds defined in the panel file.
    Thresholds can be absolute (e.g., 8000) or proportion (e.g., 'p0.001').

    Saves updated images in place and generates a QC CSV report.
    Skips processing if the report already exists.
    """

    panel_path = Path(general_config.metadata_folder) / "panel.csv"
    qc_report_path = Path(general_config.qc_folder) / "remove_outliers_report.csv"

    if qc_report_path.exists():
        logging.info(f"Skipping outlier removal — report already exists at {qc_report_path}")
        return

    if not panel_path.exists():
        logging.warning(f"Panel file not found at {panel_path}. Skipping outlier removal.")
        return

    panel = pd.read_csv(panel_path, dtype={"remove_outliers": str})

    if 'remove_outliers' not in panel.columns:
        logging.info("No 'remove_outliers' column in panel.csv. Skipping outlier removal.")
        return

    report_records = []

    for _, row in panel.iterrows():
        channel = f"{row['channel_name']}_{row['channel_label']}"
        rule = str(row.get('remove_outliers')).strip().lower()

        if rule in ["", "false", "none", "nan"]:
            continue

        try:
            logging.info(f"Processing outliers for channel: {channel} with rule: {rule}")
            img_collect, img_file_list, img_folders = load_imgs_from_directory(
                general_config.raw_images_folder, channel, quiet=True
            )
            img_folders = [Path(f) for f in img_folders]  # ensure they're Path objects

            # Flatten all image pixels for thresholding
            all_pixels = np.concatenate([img.flatten() for img in img_collect])

            if rule.startswith("p"):
                rule_clean = re.sub(r"[^\d\.]", "", rule[1:])
                percentile_value = float(rule_clean)
                threshold = np.percentile(all_pixels, 100 * (1 - percentile_value))
                threshold_type = f"Percentile ({percentile_value:.7f}%)"

                # Skip channel if percentile-derived threshold is too low
                if threshold < denoise_config.remove_outliers_min_threshold:
                    logging.warning(
                        f"Skipping channel {channel} — calculated threshold {threshold:.5f} from percentile rule '{rule}' "
                        f"is below minimum allowed ({denoise_config.remove_outliers_min_threshold})"
                    )
                    continue

            else:
                threshold = float(rule)
                threshold_type = "Absolute"

            logging.info(f"Threshold for {channel}: {threshold:.7f} ({threshold_type})")

            for img, fname, folder in zip(img_collect, img_file_list, img_folders):
                mask = img > threshold
                num_outliers = np.sum(mask)
                total_pixels = img.size
                pct = 100 * num_outliers / total_pixels

                img[mask] = 0
                save_path = Path(general_config.raw_images_folder) / folder.name / fname
                tp.imwrite(save_path, img.astype('float32'))

                report_records.append({
                    'channel': channel,
                    'roi': folder.name,
                    'image': fname,
                    'threshold': threshold,
                    'threshold_type': threshold_type,
                    'outlier_count': int(num_outliers),
                    'outlier_percentage': round(pct, 10)
                })

        except Exception as e:
            logging.error(f"Error removing outliers for {channel}: {e}", exc_info=True)

    # Save report
    if report_records:
        report_df = pd.DataFrame(report_records)
        qc_report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(qc_report_path, index=False)
        logging.info(f"Outlier removal report saved to: {qc_report_path}")
    else:
        logging.info("No outlier processing performed.")

def print_memory_status():
    # GPU memory (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                info = tf.config.experimental.get_memory_info(gpu.name)
                print(f"[GPU] {gpu.name}: Used {info['current'] / 1024**2:.2f} MB | Peak {info['peak'] / 1024**2:.2f} MB")
            except:
                print(f"[GPU] {gpu.name}: Memory info not available (TF < 2.10?)")

    # RAM usage
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024**2
    print(f"[RAM] Used: {ram_mb:.2f} MB")

if __name__ == "__main__":
    # Define the pipeline stage
    pipeline_stage = 'Denoising'

    # Load configuration
    config_data = process_config_with_overrides()

    # Setup logging
    setup_logging(config_data.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**filter_config_for_dataclass(config_data.get('general', {}), GeneralConfig))
    denoise_config = DenoisingConfig(**filter_config_for_dataclass(config_data.get('denoising', {}), DenoisingConfig))

    # Remove outliers based on panel (if defined)
    if denoise_config.remove_outliers:
        remove_outliers_from_images(general_config, denoise_config)

    if denoise_config.run_denoising:

        # Remove small ROIs based on metadata
        remove_small_rois(general_config, denoise_config)

        # Check if GPU is enabled
        gpu_test()

        # Call the denoise_batch function with parameters from the config
        denoise_batch(general_config, denoise_config)

    if denoise_config.run_QC:
        # Generate QC images for each scan iteration
        if denoise_config.run_parameter_scan and denoise_config.scan_parameter and denoise_config.scan_values:
            for scan_value in denoise_config.scan_values:
                scan_suffix = f"{denoise_config.scan_parameter}_{scan_value}"
                logging.info(f"Generating QC images for scan: {scan_suffix}")
                qc_check_side_by_side(general_config, denoise_config, scan_suffix=scan_suffix)
        else:
            # Save side-by-side comparison of raw and denoised images (no scan)
            qc_check_side_by_side(general_config, denoise_config)
