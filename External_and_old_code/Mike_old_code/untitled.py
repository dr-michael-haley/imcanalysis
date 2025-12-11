# Standard Library Imports
import logging
import os

# Third-Party Imports
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def transform_images(input_directory, output_directory, input_filter="", transformations=None, 
                     transformations_defaults=False, image_dataframe_file='Transformations.csv',
                     input_file_extension='.tif', output_file_extension='.tif',
                     source_image_as_directory=False, source_image_in_filename=False,
                     matching_directory=None, output_file_prefix="", resize_shape=None, x_axis=1, y_axis=0, channels_axis=None,
                     channel_names_list=['red','green','blue']):
    """
    Process and transform images stored in a directory according to specified transformations,
    including optional resizing based on either predefined or dynamically determined dimensions.
    
    Args:
    input_directory (str): Directory containing input images.
    output_directory (str): Directory to save transformed images.
    input_filter (str): String that must be in name of files in input directory, but will not be considerd when matching image names.
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
    y_axis(int): Y axis in image array.
    x_axis(int): X axis in image array.
    channels_axis(int): Channels axis in image array (if input images are multichannel).

    Returns:
    None
    """
    if transformations is None:
        transformations = ['flip_y', 'flip_x', 'rotate_l', 'rotate_r', 'resize', 'split_channels']
        
    if 'split_channels' in transformations:
        assert transformations[-1]=='split_channels', 'Split channels must be the last transformation'
    
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
                            print(e)
                            resize_dimensions[name] = resize_shape
                            matched_image_list.append(str(e))
                            pass
                            
                data['resize_dimensions'] = [resize_dimensions.get(name, (1000, 1000)) for name in image_names]
                
                if matching_directory:
                    data['matched_image']= matched_image_list
                
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
                if source_image_as_directory:
                    output_dir = os.path.join(output_directory, image_name)
                else:
                    output_dir = output_directory
                
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
                        image = apply_transformation(image, transformation, image_name, matching_directory, current_resize, y_axis, x_axis, channels_axis)

                save_transformed_image(image, transformations, image_dataframe, image_name, output_dir, file_prefix, output_file_extension, channel_names_list)
            else:
                logging.warning(f'Image {image_name} found, but not in dataframe... skipping.')

def apply_transformation(image, transformation, image_name, matching_directory, resize_shape, y_axis, x_axis, channels_axis):
    if transformation == 'resize':
        
        assert len(image.shape) <= len(resize_shape) + 1, "Resize shapes do not match, check dimensions of input images"
        
        if (len(image.shape) == len(resize_shape) + 1) and channels_axis:
            
            # Channel axis is at the start...
            if channels_axis < np.min((y_axis, x_axis)):
                resize_shape = tuple([image.shape[channels_axis]] + list(resize_shape))
            elif channels_axis > np.max((y_axis, x_axis)):
                resize_shape = tuple(list(resize_shape) + [image.shape[channels_axis]])
            else:
                raise Exception("Check order of y, x and channels axes") 
            
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
        return np.where(image > 0, 1, 0).astype(np.uint8)

def save_transformed_image(image, transformations, image_dataframe, image_name, output_dir, file_prefix, output_file_extension, channel_names_list):
    if 'split_channels' in transformations and image_dataframe.at[image_name, 'split_channels']:
        for channel, color in enumerate(channel_names_list):
            io.imsave(os.path.join(output_dir, f'{file_prefix}{color}{output_file_extension}'), image[:, :, channel])
    else:
        io.imsave(os.path.join(output_dir, file_prefix + image_name + output_file_extension), image)
