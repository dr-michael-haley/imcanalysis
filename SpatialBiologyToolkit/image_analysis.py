# Standard library imports
import os
from pathlib import Path
import math
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import logging

# Third-party imports
import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
import seaborn as sb
import skimage.io as io
from skimage.util import map_array
from skimage.measure import find_contours
from skimage.transform import resize
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def _smooth_contour(contour: np.ndarray, smoothing_factor: float = 1) -> np.ndarray:
    """
    Smooth a contour using spline interpolation.
    
    Parameters
    ----------
    contour : np.ndarray
        Contour coordinates.
    smoothing_factor : float
        Smoothing factor for spline interpolation.
    
    Returns
    -------
    np.ndarray
        Smoothed contour coordinates.
    """
    x, y = contour[:, 1], contour[:, 0]
    if len(x) < 4 or smoothing_factor == 0:
        return contour
    tck, _ = splprep([x, y], s=smoothing_factor, k=min(3, len(x)-1))
    x_new, y_new = splev(np.linspace(0, 1, len(x) * 10), tck)
    return np.vstack((y_new, x_new)).T


def save_labelled_image_as_svg(
    image: np.ndarray,
    color_mapping: dict,
    save_path: str,
    exclude_zero: bool = True,
    smoothing_factor: float = 0,
    background_color: str = None
) -> None:
    """
    Save a labelled image as an SVG file with optimized paths for contiguous pixels of the same color.

    Parameters
    ----------
    image : np.ndarray
        A 2D numpy array where each value corresponds to a label.
    color_mapping : dict
        A dictionary mapping pixel values to colors (hex code or color names).
    save_path : str
        The name of the output SVG file.
    exclude_zero : bool, optional
        Whether to exclude pixels with a value of 0.
    smoothing_factor : float, optional
        Smoothing factor for spline interpolation. Set to 0 for no smoothing.
    background_color : str, optional
        Background color for the SVG (default is None).
    """
    height, width = image.shape
    
    # Pad the image to handle contours at the edges
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    
    # Create the root element
    svg = Element('svg', width=str(width), height=str(height), version='1.1', xmlns='http://www.w3.org/2000/svg')
    
    def add_path(path_data: str, color: str) -> None:
        """
        Add a path element to the SVG.
        """
        path = SubElement(svg, 'path', d=path_data, fill=color)
    
    # Add background if specified
    if background_color:
        background_rect = SubElement(svg, 'rect', x='0', y='0', width=str(width), height=str(height), fill=background_color)
    
    for label, color in color_mapping.items():
        if exclude_zero and label == 0:
            continue
        
        # Find all contours for the current label
        mask = (padded_image == label).astype(np.uint8)
        contours = find_contours(mask, 0.5)
        
        for contour in contours:
            # Adjust contour coordinates because of padding
            contour -= 1
            smoothed_contour = _smooth_contour(contour, smoothing_factor)
            path_data = 'M ' + ' '.join(f'{point[1]},{point[0]}' for point in smoothed_contour) + ' Z'
            add_path(path_data, color)
    
    # Convert to a string
    raw_string = tostring(svg, 'utf-8')
    reparsed = minidom.parseString(raw_string)
    pretty_string = reparsed.toprettyxml(indent="  ")
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(pretty_string)

def map_pixel_values_to_colors(
    image: np.ndarray,
    cmap_name: str = 'viridis',
    min_val: float = None,
    max_val: float = None,
    quantile: float = None,
    return_norm: bool = False
) -> dict:
    """
    Map pixel values in an image to hex colors using a specified matplotlib colormap.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array of pixel values.
    cmap_name : str, optional
        The name of the colormap (default is 'viridis').
    min_val : float, optional
        Minimum value for color scaling. If None, it is set to the minimum value in the image.
    max_val : float, optional
        Maximum value for color scaling. If None, it is set to the maximum value in the image.
    quantile : float, optional
        If specified, use this quantile to set the max value for color scaling.
    return_norm : bool, optional
        Whether to return the normalization function.

    Returns
    -------
    dict
        A dictionary mapping unique pixel values to hex color codes.
    """
    cmap = plt.get_cmap(cmap_name)
    
    if min_val is None:
        min_val = np.min(image)
    if max_val is None:
        max_val = np.max(image)
    elif quantile is not None:
        max_val = np.quantile(image, quantile)
        
    norm = Normalize(vmin=min_val, vmax=max_val)
    
    unique_values = np.unique(image)
    color_mapping = {val: to_hex(cmap(norm(val))) for val in unique_values}
    
    if return_norm:
        return norm
    return color_mapping

def save_labelled_image(
    label_image: np.ndarray,
    color_mapping: dict,
    save_path: str,
    background_color: str = None,
    hide_axes: bool = False,
    hide_ticks: bool = True
) -> None:
    """
    Convert a label image to a color image using a given dictionary of pixel values to HEX colors and save the image using matplotlib.

    Parameters
    ----------
    label_image : np.ndarray
        The input label image with pixel values corresponding to the dictionary keys.
    color_mapping : dict
        A dictionary mapping pixel values (int) to HEX colors (str).
    save_path : str
        The file path to save the output color image.
    background_color : str, optional
        The background color of the saved image. If None, the background will be transparent.
    hide_axes : bool, optional
        If True, hides the axes completely (default is False).
    hide_ticks : bool, optional
        If True, hides the ticks and labels on the axes (default is True).

    Returns
    -------
    None
    """
    def hex_to_rgb(hex_color: str) -> tuple:
        return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    if background_color is not None:
        background_rgb = mcolors.to_rgb(background_color)
        background_rgb = tuple(int(c * 255) for c in background_rgb)
    else:
        background_rgb = (0, 0, 0, 0)

    color_image = np.full((label_image.shape[0], label_image.shape[1], 4 if background_color is None else 3), background_rgb, dtype=np.uint8)

    for label, hex_color in color_mapping.items():
        rgb_color = hex_to_rgb(hex_color)
        color_image[label_image == label, :3] = rgb_color
        if background_color is None:
            color_image[label_image == label, 3] = 255

    fig, ax = plt.subplots()
    ax.imshow(color_image)
    
    if hide_axes:
        ax.axis('off')
    else:
        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if background_color is not None:
        fig.patch.set_facecolor(background_color)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor=background_color)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)

    plt.close()

def transform_images(
    input_directory: str,
    output_directory: str,
    input_filter: str = "",
    transformations: list = None,
    transformations_defaults: dict = None,
    image_dataframe_file: str = 'Transformations.csv',
    input_file_extension: str = '.tif',
    output_file_extension: str = '.tif',
    source_image_as_directory: bool = False,
    source_image_in_filename: bool = False,
    matching_directory: str = None,
    output_file_prefix: str = "",
    resize_shape: tuple = None,
    x_axis: int = 1,
    y_axis: int = 0,
    channels_axis: int = None,
    channel_names_list: list = ['red', 'green', 'blue'],
    note_skipped_images: bool = True
) -> None:
    """
    Process and transform images stored in a directory according to specified transformations,
    including optional resizing based on either predefined or dynamically determined dimensions.
    
    Args:
        input_directory (str): Directory containing input images.
        output_directory (str): Directory to save transformed images.
        input_filter (str): String that must be in the name of files in the input directory, but will not be considered when matching image names.
        transformations (list): List of transformations to apply (e.g., ['flip_y', 'flip_x', 'resize', 'split_channels']).
        transformations_defaults (dict): Default values for each transformation, including optional resize dimensions.
        image_dataframe_file (str or pd.DataFrame): Path to a CSV file or a DataFrame containing transformation settings.
        input_file_extension (str): File extension of input images.
        output_file_extension (str): File extension for output images.
        source_image_as_directory (bool): If True, create a subdirectory for each image in the output directory.
        source_image_in_filename (bool): If True, include the source image name as a prefix in output filenames.
        matching_directory (str): Directory containing images used for matching dimensions.
        output_file_prefix (str): Prefix for output file names.
        resize_shape (tuple): Default dimensions to resize images to (used if matching_directory is not provided).
        y_axis (int): Y axis in image array.
        x_axis (int): X axis in image array.
        channels_axis (int): Channels axis in image array (if input images are multichannel).
        channel_names_list (list): List of channel names for split channels transformation.
        note_skipped_images (bool): If True, log a warning for images that are skipped.
    """
    if transformations is None:
        transformations = ['flip_y', 'flip_x', 'rotate_l', 'rotate_r', 'resize', 'split_channels']
        
    if 'split_channels' in transformations:
        assert transformations[-1] == 'split_channels', 'Split channels must be the last transformation'
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    if isinstance(image_dataframe_file, str):
        try:
            image_dataframe = pd.read_csv(image_dataframe_file, index_col=0)
            if 'resize_dimensions' in image_dataframe.columns:
                image_dataframe['resize_dimensions'] = image_dataframe['resize_dimensions'].apply(eval)
        except FileNotFoundError:
            logging.error(f"No CSV file found at {image_dataframe_file}. Creating a default DataFrame with initial resize dimensions.")
            image_names = [f[:-len(input_file_extension)] for f in os.listdir(input_directory) if f.endswith(input_file_extension) and input_filter in f]
            data = {t: str(transformations_defaults) for t in transformations}
            if 'resize' in transformations:
                resize_dimensions = {}
                matched_image_list = []
                for name in image_names:
                    if resize_shape:
                        resize_dimensions[name] = resize_shape
                    elif matching_directory:
                        try:
                            matching_image_path = os.path.join(matching_directory, name)
                            matching_image_name = os.listdir(matching_image_path)[0]
                            matching_image = io.imread(os.path.join(matching_image_path, matching_image_name))
                            matched_image_list.append(str(os.path.join(matching_image_path, matching_image_name)))
                            resize_dimensions[name] = matching_image.shape[:2]
                        except Exception as e:
                            logging.error(f"Error matching image {name}: {e}")
                            resize_dimensions[name] = resize_shape
                            matched_image_list.append(str(e))
                data['resize_dimensions'] = [resize_dimensions.get(name, (1000, 1000)) for name in image_names]
                if matching_directory:
                    data['matched_image'] = matched_image_list
            image_dataframe = pd.DataFrame(data, index=image_names)
            image_dataframe.to_csv(image_dataframe_file)
            return
    elif isinstance(image_dataframe_file, pd.DataFrame):
        image_dataframe = image_dataframe_file

    for file_name in os.listdir(input_directory):
        if file_name.endswith(input_file_extension):
            image_name = file_name[:-len(input_file_extension)]
            image_name = image_name.replace(input_filter, '')
            if image_name in image_dataframe.index:
                output_dir = os.path.join(output_directory, image_name) if source_image_as_directory else output_directory
                os.makedirs(output_dir, exist_ok=True)
                file_prefix = output_file_prefix + ("_" + image_name if source_image_in_filename else "")
                input_image_path = os.path.join(input_directory, file_name)
                image = io.imread(input_image_path)
                for transformation in transformations:
                    if transformation in image_dataframe.columns and image_dataframe.at[image_name, transformation]:
                        if transformation == 'resize':
                            current_resize = tuple(image_dataframe.at[image_name, 'resize_dimensions'])
                        else:
                            current_resize = resize_shape
                        image = _apply_transformation(image, transformation, image_name, matching_directory, current_resize, y_axis, x_axis, channels_axis)
                _save_transformed_image(image, transformations, image_dataframe, image_name, output_dir, file_prefix, output_file_extension, channel_names_list)
            else:
                if note_skipped_images:
                    logging.warning(f'Image {image_name} found, but not in dataframe... skipping.')

def _apply_transformation(
    image: np.ndarray,
    transformation: str,
    image_name: str,
    matching_directory: str,
    resize_shape: tuple,
    y_axis: int,
    x_axis: int,
    channels_axis: int
) -> np.ndarray:
    """
    Apply a specific transformation to an image.

    Args:
        image (np.ndarray): Input image.
        transformation (str): Transformation to apply.
        image_name (str): Name of the image.
        matching_directory (str): Directory for matching dimensions.
        resize_shape (tuple): Shape to resize the image to.
        y_axis (int): Y axis in image array.
        x_axis (int): X axis in image array.
        channels_axis (int): Channels axis in image array.

    Returns:
        np.ndarray: Transformed image.
    """
    logging.info(f'Processing image: {image_name}, transformation: {transformation}...')
    
    if transformation == 'resize':
        assert len(image.shape) <= len(resize_shape) + 1, "Resize shapes do not match, check dimensions of input images"
        if (len(image.shape) == len(resize_shape) + 1) and channels_axis is not None:
            if channels_axis < np.min((y_axis, x_axis)):
                resize_shape = tuple([image.shape[channels_axis]] + list(resize_shape))
            elif channels_axis > np.max((y_axis, x_axis)):
                resize_shape = tuple(list(resize_shape) + [image.shape[channels_axis]])
            else:
                raise Exception("Check order of y, x and channels axes")
        logging.info(f'Final resizing of image: {image_name}, resize_shape: {str(resize_shape)}, y_axis={y_axis}, x_axis={x_axis}, channels_axis={channels_axis}')   
        return resize(image, resize_shape, preserve_range=True).astype(image.dtype)
    elif transformation == 'flip_x':
        return np.flip(image, axis=x_axis)
    elif transformation == 'flip_y':
        return np.flip(image, axis=y_axis)
    elif transformation == 'rotate_l':
        return np.rot90(image, k=1, axes=(y_axis, x_axis))
    elif transformation == 'rotate_r':
        return np.rot90(image, k=-1, axes=(y_axis, x_axis))
    elif transformation == 'make_binary':
        binary = np.zeros_like(image, dtype=np.uint8)
        if channels_axis:
            for ch in range(image.shape[channels_axis]):
                binary[ch] = image[ch] > 0
        else:
            binary = image > 0
        return binary.astype(np.uint8)

def _save_transformed_image(
    image: np.ndarray,
    transformations: list,
    image_dataframe: pd.DataFrame,
    image_name: str,
    output_dir: str,
    file_prefix: str,
    output_file_extension: str,
    channel_names_list: list
) -> None:
    """
    Save the transformed image to disk.

    Args:
        image (np.ndarray): Transformed image.
        transformations (list): List of transformations applied.
        image_dataframe (pd.DataFrame): DataFrame with transformation settings.
        image_name (str): Name of the image.
        output_dir (str): Directory to save the transformed image.
        file_prefix (str): Prefix for the output file name.
        output_file_extension (str): File extension for the output file.
        channel_names_list (list): List of channel names for split channels transformation.
    """
    if 'split_channels' in transformations and image_dataframe.at[image_name, 'split_channels']:
        for channel, color in enumerate(channel_names_list):
            savepath = f'{file_prefix}{color}{output_file_extension}'
            io.imsave(os.path.join(output_dir, savepath), image[:, :, channel], check_contrast=False)
            logging.info(f'Saving image: {image_name}, output: {savepath}')
    else:
        savepath = file_prefix + image_name + output_file_extension
        io.imsave(os.path.join(output_dir, savepath), image, check_contrast=False)
        logging.info(f'Saving image: {image_name}, output: {savepath}')