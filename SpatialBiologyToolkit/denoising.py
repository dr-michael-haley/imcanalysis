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

import os
import json
import logging
from glob import glob
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np
import tifffile as tp

from IMC_Denoise.DeepSNiF_utils import DeepSNiF_DataGenerator as DeepSNF_DataGenerator
from IMC_Denoise.IMC_Denoise_main import DIMR
from IMC_Denoise.IMC_Denoise_main import DeepSNiF as DeepSNF


def setup_logging(log_file='denoising.log'):
    """
    Set up logging configuration.

    Parameters
    ----------
    log_file : str, optional
        The filename for the log file. Defaults to 'denoising.log'.

    Returns
    -------
    None
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_single_img(filename):
    """
    Load a single image from a file.

    Parameters
    ----------
    filename : str or Path
        The image file name, must end with .tiff or .tif.

    Returns
    -------
    numpy.ndarray
        Loaded image data as a 2D NumPy array.

    Raises
    ------
    ValueError
        If the file does not end with .tiff or .tif.
        If the loaded image is not 2D.
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

    Parameters
    ----------
    load_directory : str or Path
        Directory containing the images.
    channel_name : str
        Name of the channel to load images for.
    quiet : bool, optional
        If True, suppress print statements. Defaults to False.

    Returns
    -------
    img_collect : list of numpy.ndarray
        List of loaded images.
    img_file_list : list of str
        List of image file names.
    img_folders : list of str
        List of image folder paths.

    Raises
    ------
    ValueError
        If no images are found for the specified channel.
    """
    img_collect = []
    img_folders = glob(join(str(load_directory), "*", ""))
    img_file_list = []

    if not quiet:
        print('Loading image data from directories...\n')
        logging.info('Loading image data from directories.')

    for sub_img_folder in img_folders:
        img_list = [
            f for f in listdir(sub_img_folder)
            if isfile(join(sub_img_folder, f)) and (f.endswith(".tiff") or f.endswith(".tif"))
        ]
        for img_file in img_list:
            if channel_name.lower() in img_file.lower():
                img_path = join(sub_img_folder, img_file)
                img_read = load_single_img(img_path)

                if not quiet:
                    print(img_path)
                logging.debug(f'Loaded image for channel {channel_name}: {img_path}')

                img_file_list.append(img_file)
                img_collect.append(img_read)
                break  # Only one image per channel per folder

    if not quiet:
        print('\nImage data loading completed!')
        logging.info('Image data loading completed.')

    if not img_collect:
        logging.error(f'No images found for channel "{channel_name}".')
        raise ValueError(f'No images found for channel "{channel_name}". Please check the channel name!')

    return img_collect, img_file_list, img_folders


def denoise_batch(
    raw_directory="tiffs",
    processed_output_dir="processed",
    method="deep_snf",
    channels=None,
    **kwargs
):
    """
    Denoise images in batch using the specified method.

    Parameters
    ----------
    raw_directory : str or Path, optional
        Input folder containing raw images. Defaults to "tiffs".
    processed_output_dir : str or Path, optional
        Output folder to save processed images. Defaults to "processed".
    method : str, optional
        Denoising method to use. Options are 'deep_snf' or 'dimr'. Defaults to 'deep_snf'.
    channels : list of str, optional
        List of specific channels to process. Defaults to None.

    Additional keyword arguments
    ----------------------------
    For 'deep_snf' method:
        patch_step_size : int, optional
            Step size for patch extraction. Defaults to 60.
        train_epochs : int, optional
            Number of training epochs. Defaults to 50.
        train_initial_lr : float, optional
            Initial learning rate. Defaults to 1e-3.
        train_batch_size : int, optional
            Training batch size. Defaults to 128.
        pixel_mask_percent : float, optional
            Percentage of masked pixels in each patch. Defaults to 0.2.
        val_set_percent : float, optional
            Percentage of validation set. Defaults to 0.15.
        loss_function : str, optional
            Loss function to use. Defaults to 'I_divergence'.
        loss_name : str, optional
            File name to save training and validation losses. Defaults to None.
        weights_save_directory : str or Path, optional
            Directory to save weights and loss files. Defaults to None.
        is_load_weights : bool, optional
            If True, use existing weights without training. Defaults to False.
        lambda_HF : float, optional
            High-frequency regularization parameter. Defaults to 3e-6.
        n_neighbours : int, optional
            Number of neighbors for DIMR. Defaults to 4.
        n_iter : int, optional
            Number of iterations for DIMR. Defaults to 3.
        window_size : int, optional
            Sliding window size for DIMR. Defaults to 3.

    For 'dimr' method:
        n_neighbours : int, optional
            Number of neighbors for DIMR. Defaults to 4.
        n_iter : int, optional
            Number of iterations for DIMR. Defaults to 3.
        window_size : int, optional
            Sliding window size for DIMR. Defaults to 3.

    Returns
    -------
    None

    Notes
    -----
    This function processes images in batch mode using either the DeepSNF or DIMR method.
    It saves the denoised images to the specified output directory.
    """
    logging.info('Starting denoise_batch function.')
    logging.info(f'Parameters: raw_directory={raw_directory}, processed_output_dir={processed_output_dir}, method={method}, channels={channels}')

    # Ensure channels is a list
    if channels is None:
        channels = []
    elif isinstance(channels, str):
        channels = [channels]

    # Create output directory
    processed_output_dir = Path(processed_output_dir)
    processed_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Processed output directory set to: {processed_output_dir}')

    # Error tracking lists
    error_channels = []
    completed_channels = []

    if not channels:
        logging.error('No channels provided to process.')
        raise ValueError("No channels provided to process.")

    print(f'\nPerforming denoising using method "{method}" on the following channels:\n{channels}\n')
    logging.info(f'Performing denoising using method "{method}" on channels: {channels}')

    for channel_name in channels:
        try:
            logging.info(f'Starting processing for channel: {channel_name}')
            # Load images for the channel
            img_collect, img_file_list, img_folders = load_imgs_from_directory(raw_directory, channel_name)
            logging.info(f'Loaded {len(img_collect)} images for channel {channel_name}')

            if method == 'deep_snf':
                # DeepSNF parameters
                patch_step_size = kwargs.get('patch_step_size', 60)
                train_epochs = kwargs.get('train_epochs', 50)
                train_initial_lr = kwargs.get('train_initial_lr', 1e-3)
                train_batch_size = kwargs.get('train_batch_size', 128)
                pixel_mask_percent = kwargs.get('pixel_mask_percent', 0.2)
                val_set_percent = kwargs.get('val_set_percent', 0.15)
                loss_function = kwargs.get('loss_function', 'I_divergence')
                loss_name = kwargs.get('loss_name', None)
                weights_save_directory = kwargs.get('weights_save_directory', None)
                is_load_weights = kwargs.get('is_load_weights', False)
                lambda_HF = kwargs.get('lambda_HF', 3e-6)
                n_neighbours = kwargs.get('n_neighbours', 4)
                n_iter = kwargs.get('n_iter', 3)
                window_size = kwargs.get('window_size', 3)

                logging.info(f'DeepSNF parameters: patch_step_size={patch_step_size}, train_epochs={train_epochs}, train_initial_lr={train_initial_lr}, train_batch_size={train_batch_size}, pixel_mask_percent={pixel_mask_percent}, val_set_percent={val_set_percent}, loss_function={loss_function}, is_load_weights={is_load_weights}, lambda_HF={lambda_HF}, n_neighbours={n_neighbours}, n_iter={n_iter}, window_size={window_size}')

                # Training settings
                row_step = patch_step_size
                col_step = patch_step_size

                # Generate patches and train model if not loading weights
                if not is_load_weights:
                    logging.info('Generating patches for training.')
                    data_generator = DeepSNF_DataGenerator(
                        channel_name=channel_name,
                        n_neighbours=n_neighbours,
                        n_iter=n_iter,
                        window_size=window_size,
                        col_step=col_step,
                        row_step=row_step
                    )

                    generated_patches = data_generator.generate_patches_from_directory(
                        load_directory=raw_directory
                    )
                    logging.info(f'Generated training patches with shape: {generated_patches.shape}')
                    print(f'The shape of the generated training set is {generated_patches.shape}.')

                weights_name = f"weights_{channel_name}.hdf5"
                logging.info(f'Weights file name set to: {weights_name}')

                deepsnf = DeepSNF(
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
                    lambda_HF=lambda_HF
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
                    tp.imsave(save_path, img_denoised.astype('float32'))
                    logging.info(f'Saved denoised image to: {save_path}')

            elif method == 'dimr':
                # DIMR parameters
                n_neighbours = kwargs.get('n_neighbours', 4)
                n_iter = kwargs.get('n_iter', 3)
                window_size = kwargs.get('window_size', 3)
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

    Prints the list of available GPUs or an error message if no GPU is found.
    """
    import tensorflow as tf

    if tf.test.is_built_with_cuda():
        print('GPU acceleration enabled.\n')
        print(tf.config.list_physical_devices('GPU'))
        logging.info('GPU acceleration enabled.')
        logging.info(f'Available GPUs: {tf.config.list_physical_devices("GPU")}')
    else:
        print('GPU not found! Check TensorFlow and CUDA setup.')
        logging.error('GPU not found! Check TensorFlow and CUDA setup.')


if __name__ == "__main__":
    # Set up logging
    setup_logging()

    # Load configuration from a local JSON file
    config_file = 'denoise_config.json'

    if not os.path.isfile(config_file):
        logging.error(f"Configuration file '{config_file}' not found.")
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, 'r') as f:
        config = json.load(f)
        logging.info(f'Loaded configuration from {config_file}')
        
    # Check if GPU is enabled
    gpu_test()

    # Extract parameters from the config
    raw_directory = config.get('raw_directory', 'tiffs')
    processed_output_dir = config.get('processed_output_dir', 'processed')
    method = config.get('method', 'deep_snf')
    channels = config.get('channels', [])

    # Extract method-specific parameters
    kwargs = config.get('parameters', {})

    # Call the denoise_batch function with parameters from the config
    denoise_batch(
        raw_directory=raw_directory,
        processed_output_dir=processed_output_dir,
        method=method,
        channels=channels,
        **kwargs
    )
