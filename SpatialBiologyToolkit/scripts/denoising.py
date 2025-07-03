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

# Import shared utilities and configurations
from .config_and_utils import *

# Import IMC_Denoise methods
from IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF

# Additional imports
import tensorflow as tf

def load_single_img(filename):
    """
    Load a single image from a file.
    """
    filename = str(filename)
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        img_in = tp.imread(filename).astype('float32')
        logging.debug(f'Loaded image: {filename}')
    else:
        logging.error(f'File {filename} does not end with .tiff or .tif')
        raise ValueError('Raw file should end with .tiff or .tif!')
    if img_in.ndim != 2:
        logging.error(f'Image {filename} is not 2D')
        raise ValueError('Single image should be 2D!')
    return img_in

def load_imgs_from_directory(load_directory, channel_name, quiet=False):
    """
    Load images for a specific channel from a directory.
    """
    img_collect = []
    img_folders = glob(os.path.join(str(load_directory), "*", ""))
    img_file_list = []

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
                img_read = load_single_img(img_path)

                if not quiet:
                    print(img_path)
                logging.debug(f'Loaded image for channel {channel_name}: {img_path}')

                img_file_list.append(img_file)
                img_collect.append(img_read)
                break  # Only one image per channel per folder

    if not img_collect:
        logging.error(f'No images found for channel "{channel_name}".')
        raise ValueError(f'No images found for channel "{channel_name}". Please check the channel name!')

    if not quiet:
        print('\nImage data loading completed!')
        logging.info('Image data loading completed.')

    return img_collect, img_file_list, img_folders

def denoise_batch(
    general_config: GeneralConfig,
    denoise_config: DenoisingConfig
):
    """
    Denoise images in batch using the specified method from the DenoisingConfig.
    """
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

    # DeepSNF parameters from denoise_config
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

    for channel_name in channels:
        try:
            logging.info(f'Starting processing for channel: {channel_name}')
            # Load images for the channel
            img_collect, img_file_list, img_folders = load_imgs_from_directory(raw_directory, channel_name)
            logging.info(f'Loaded {len(img_collect)} images for channel {channel_name}')

            if method == 'deep_snf':

                logging.info(f'DeepSNF parameters: {denoise_config}')

                # Training settings
                row_step = patch_step_size
                col_step = patch_step_size

                # Generate patches and train model if not loading weights
                if not is_load_weights:
                    logging.info('Calculating maximum patch count using ratio_thresh=1.0')
                    max_data_generator = DeepSNiF_DataGenerator(
                        channel_name=channel_name,
                        n_neighbours=n_neighbours,
                        n_iter=n_iter,
                        window_size=window_size,
                        col_step=patch_step_size,
                        row_step=patch_step_size,
                        ratio_thresh=1.0
                    )
                    max_patches = max_data_generator.generate_patches_from_directory(load_directory=raw_directory)
                    max_patch_count = max_patches.shape[0]
                    logging.info(f"Max patch count for {channel_name}: {max_patch_count}")

                    current_step = patch_step_size
                    min_step = 20
                    threshold = denoise_config.intelligent_patch_size_threshold

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

                        if max_patch_count == 0:
                            percentage = 0
                        else:
                            percentage = patch_count / max_patch_count

                        logging.info(f"Generated {patch_count} patches for {channel_name} ({percentage:.2%} of max)")

                        if not denoise_config.intelligent_patch_size:
                            break

                        if percentage >= threshold:
                            break

                        if current_step <= min_step:
                            logging.warning(
                                f"Reached minimum patch_step_size ({min_step}). Proceeding with {patch_count} patches.")
                            break

                        logging.info(
                            f"Patch count below threshold ({threshold:.0%}) — reducing patch_step_size and retrying.")
                        current_step -= 20

                weights_name = f"weights_{channel_name}.keras"
                logging.info(f'Weights file name set to: {weights_name}')

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
                    network_size=network_size
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

                # Process images
                for img, img_file_name, folder in zip(img_collect, img_file_list, img_folders):
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
                    output_dir = processed_output_dir / roi_folder_name
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
                for img, img_file_name, folder in zip(img_collect, img_file_list, img_folders):
                    logging.info(f'Denoising image: {img_file_name}')
                    # Perform DIMR denoising
                    img_denoised = dimr.perform_DIMR(img)

                    # Get ROI folder name from the path
                    roi_folder_name = Path(folder).name

                    # Ensure output directory exists
                    output_dir = processed_output_dir / roi_folder_name
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


    # After processing all channels, compute pixel QC metrics
    logging.info("Computing QC pixel statistics for denoised images...")

    qc_records = []

    for channel_name in channels:
        try:
            # Load denoised images
            pro_imgs, pro_img_names, pro_folders = load_imgs_from_directory(general_config.denoised_images_folder, channel_name, quiet=True)

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

    # Save CSV report
    qc_df = pd.DataFrame(qc_records)
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
                          denoise_config: DenoisingConfig):
    channels = denoise_config.channels

    if channels == []:
        channels = load_channels_from_panel(general_config)

    # Create folders
    save_dir = Path(os.path.join(general_config.qc_folder, denoise_config.qc_image_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    for channel_name in channels:

        try:

            raw_Img_collect, raw_Img_file_list, raw_img_folders = load_imgs_from_directory(general_config.raw_images_folder, channel_name,
                                                                                           quiet=True)
            pro_Img_collect, pro_Img_file_list, pro_img_folders = load_imgs_from_directory(general_config.denoised_images_folder,
                                                                                           channel_name, quiet=True)

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

def remove_outliers_from_images(general_config):
    """
    Removes outlier pixel values from images based on thresholds defined in the panel file.
    Thresholds can be absolute values (e.g., 8000) or percentiles (e.g., 'p0.001').

    Saves updated images in place and outputs a CSV report to the QC folder.

    Parameters
    ----------
    general_config : GeneralConfig
        Configuration object with paths including raw_images_folder and qc_folder.
    """
    panel_path = Path(general_config.metadata_folder) / "panel.csv"
    qc_report_path = Path(general_config.qc_folder) / "remove_outliers_report.csv"

    if not panel_path.exists():
        logging.warning(f"Panel file not found at {panel_path}. Skipping outlier removal.")
        return

    panel = pd.read_csv(panel_path, dtype={"remove_outliers": str})

    if 'remove_outliers' not in panel.columns:
        logging.info("No 'remove_outliers' column in panel.csv. Skipping outlier removal.")
        return

    report_records = []

    for idx, row in panel.iterrows():
        channel = f"{row['channel_name']}_{row['channel_label']}"
        rule = row.get('remove_outliers')

        if pd.isna(rule) or str(rule).strip().lower() in ["", "false", "none", "nan"]:
            continue

        #try:
        logging.info(f"Processing outliers for channel: {channel} with rule: {rule}")
        img_collect, img_file_list, img_folders = load_imgs_from_directory(
            general_config.raw_images_folder, channel, quiet=True
        )

        # Determine threshold
        all_pixels = np.concatenate([img.flatten() for img in img_collect])
        if isinstance(rule, str) and rule.strip().lower().startswith("p"):
            # Extract numeric portion and ensure valid float
            rule_clean = re.sub(r"[^\d\.]", "", str(rule)[1:])  # strip 'p', remove non-numeric
            try:
                percentile_value = float(rule_clean)
                threshold = np.percentile(all_pixels, 100 - percentile_value * 100)
                threshold_type = f"Percentile ({percentile_value:.7f}%)"
            except ValueError:
                raise ValueError(f"Invalid percentile value: {rule}")
        else:
            try:
                threshold = float(str(rule).strip())
                threshold_type = "Absolute"
            except ValueError:
                raise ValueError(f"Invalid absolute threshold: {rule}")

        logging.info(f"Threshold for {channel}: {threshold:.7f} ({threshold_type})")

        for img, fname, folder in zip(img_collect, img_file_list, img_folders):
            mask = img > float(threshold)
            num_outliers = np.sum(mask)
            total_pixels = img.size
            pct = 100 * num_outliers / total_pixels

            img[mask] = 0
            save_path = Path(general_config.raw_images_folder) / Path(folder).name / fname
            tiff.imwrite(save_path, img.astype('float32'))

            report_records.append({
                'channel': channel,
                'roi': Path(folder).name,
                'image': fname,
                'threshold': threshold,
                'threshold_type': threshold_type,
                'outlier_count': int(num_outliers),
                'outlier_percentage': round(pct, 4)  # or leave as float
            })

        #except Exception as e:
        #    logging.error(f"Error removing outliers for {channel}: {e}")

    # Save report
    if report_records:
        report_df = pd.DataFrame(report_records)
        qc_report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(qc_report_path, index=False)
        logging.info(f"Outlier removal report saved to: {qc_report_path}")
    else:
        logging.info("No outlier processing performed.")



if __name__ == "__main__":
    # Define the pipeline stage
    pipeline_stage = 'Denoising'

    # Load configuration
    config_data = process_config_with_overrides()

    # Setup logging
    setup_logging(config_data.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**config_data.get('general', {}))
    denoise_config = DenoisingConfig(**config_data.get('denoising', {}))

    # Remove outliers based on panel (if defined)
    if denoise_config.remove_outliers:
        remove_outliers_from_images(general_config)

    if denoise_config.run_denoising:

        # Remove small ROIs based on metadata
        remove_small_rois(general_config, denoise_config)

        # Check if GPU is enabled
        gpu_test()

        # Call the denoise_batch function with parameters from the config
        denoise_batch(general_config, denoise_config)

    if denoise_config.run_QC:

        # Save side-by-side comparisson of raw and denoised images
        qc_check_side_by_side(general_config, denoise_config)