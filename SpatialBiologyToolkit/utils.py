# Standard Library Imports
import datetime
import os
from os import getlogin
from pathlib import Path
import subprocess
from types import ModuleType
from typing import List, Union, Optional, Tuple, Any
import math
import importlib.util

# Third-Party Imports
import anndata as ad
import pandas as pd
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from skimage.measure import regionprops
import scipy
from scipy.sparse import issparse
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch


def compare_lists(L1: List[Any], L2: List[Any], L1_name: str, L2_name: str, return_error: bool = True) -> None:
    """
    Function for thoroughly comparing lists.

    Parameters
    ----------
    L1 : list
        First list to compare.
    L2 : list
        Second list to compare.
    L1_name : str
        Name of the first list.
    L2_name : str
        Name of the second list.
    return_error : bool, optional
        If True, raise a TypeError if the lists do not match (default is True).
    """
    L1_not_L2 = [x for x in L1 if x not in L2]
    L2_not_L1 = [x for x in L2 if x not in L1]

    try:
        assert L1_not_L2 == L2_not_L1, "Lists did not match:"
    except AssertionError:
        print(f'{L1_name} items NOT in {L2_name}:')
        print(L1_not_L2)
        print(f'{L2_name} items NOT in {L1_name}:')
        print(L2_not_L1)

        if return_error:
            raise TypeError(f"{L1_name} and {L2_name} should have exactly the same items")


def print_full(x: pd.DataFrame) -> None:
    """
    Prints a full pandas DataFrame.

    Parameters
    ----------
    x : pd.DataFrame
        DataFrame to be printed.
    """
    pd.set_option('display.max_rows', len(x))
    display(x)
    pd.reset_option('display.max_rows')


def adlog(
    adata: ad.AnnData,
    entry: Optional[str] = None,
    module: Optional[Union[ModuleType, str]] = None,
    module_name: Optional[str] = None,
    module_version: Optional[str] = None,
    save: bool = False,
    temp_file: str = 'adata_temp.h5ad',
    log_path: str = 'adata_logging.csv'
) -> None:
    """
    This function saves a log in the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    entry : str, optional
        Log entry description.
    module : Union[ModuleType, str], optional
        Module to log.
    module_name : str, optional
        Name of the module.
    module_version : str, optional
        Version of the module.
    save : bool, optional
        If True, saves a temporary file (default is False).
    temp_file : str, optional
        Path for the temporary file (default is 'adata_temp.h5ad').
    log_path : str, optional
        Path to the log file.
    """
    try:
        login_user = getlogin()
    except Exception:
        login_user = 'Unknown'

    now = datetime.datetime.now()
    date_now = now.strftime('%Y-%m-%d')
    time_now = now.strftime('%H:%M:%S')

    try:
        log = adata.uns['logging']
    except KeyError:
        if not os.path.isfile(log_path):
            print('No log in AnnData or saved locally, creating new log')
            log = pd.DataFrame(columns=['Date', 'Time', 'Entry', 'Module', 'Version', 'User'])
            adata.uns.update({'logging': log})
        else:
            log = pd.read_csv(log_path, index_col=0)
            adata.uns.update({'logging': log})

    if isinstance(module, ModuleType):
        module_name = module.__name__
        try:
            module_version = module.__version__
        except AttributeError:
            module_version = None

    if isinstance(module, str):
        module_name = module

    if entry:
        log.loc[len(log.index)] = [date_now, time_now, entry, module_name, module_version, login_user]

    if save:
        print(f'Saving temporary file: {temp_file}')
        log.loc[len(log.index)] = [date_now, time_now, f'Saved AnnData backup: {temp_file}', None, None, login_user]
        log.to_csv(log_path)
        del adata.uns['logging'] # Saving dataframes in .uns is unreliable, depending on h5py version
        adata.write(temp_file)


def subset(
    adata: ad.AnnData,
    obs_column: str,
    obs_values: List[Any] = [],
    include_list: bool = True
) -> ad.AnnData:
    """
    Returns a filtered AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_column : str
        Column in .obs to filter on.
    obs_values : list
        Values to include if include_list=True, or exclude if include_list=False.
    include_list : bool, optional
        If True, includes the values in the filter, else excludes them (default is True).

    Returns
    -------
    AnnData
        Filtered AnnData object.
    """
    if include_list:
        return adata[adata.obs[obs_column].isin(obs_values)]
    else:
        return adata[~adata.obs[obs_column].isin(obs_values)]


def _cleanstring(data: Any) -> str:
    """
    Helper function that returns a clean string.

    Parameters
    ----------
    data : Any
        Input data to be cleaned.

    Returns
    -------
    str
        Cleaned string.
    """
    import re
    data = str(data)
    data = re.sub(r'\W+', '', data)
    return data


def _save(path: Tuple[str], filename: str) -> Path:
    """
    Helper function that returns a save path, and makes sure all the directories are in place.

    Parameters
    ----------
    path : tuple
        Path components to create directories.
    filename : str
        Filename to be saved.

    Returns
    -------
    Path
        Full path including filename.
    """
    path_dir = Path(*path)
    os.makedirs(path_dir, exist_ok=True)
    return Path(path_dir, filename)


def _check_input_type(data: Union[ad.AnnData, pd.DataFrame, str]) -> pd.DataFrame:
    """
    Helper function that will, depending on the datatype, either access the adata.obs from an AnnData,
    take a DataFrame directly, or load a .csv if given a string.

    Parameters
    ----------
    data : Union[AnnData, pd.DataFrame, str]
        Input data to be checked.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    input_type = type(data)

    if input_type == ad._core.anndata.AnnData:
        return data.obs.copy()

    if input_type == pd.core.frame.DataFrame:
        return data.copy()

    if input_type == str:
        return pd.read_csv(data, low_memory=False)

    raise TypeError(f'Data type {str(input_type)} not recognized')


def _to_list(data: Union[pd.Index, list, Any]) -> list:
    """
    Helper function that sanitizes an input so that it's in a list format, even if just a list of one.

    Parameters
    ----------
    data : Union[pd.Index, list, Any]
        Input data to be converted to a list.

    Returns
    -------
    list
        Converted list.
    """
    if isinstance(data, pd.core.indexes.base.Index):
        return data.tolist()

    if isinstance(data, list):
        return data

    if data is None:
        return []

    return [data]


def pip_freeze_to_dataframe() -> pd.DataFrame:
    """
    Capture the output of `pip freeze` and convert it to a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing package names and versions.
    """
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)
    lines = result.stdout.split('\n')

    packages = [{'Package': name, 'Version': version} for line in lines if '==' in line for name, version in [line.split('==')]]
    
    return pd.DataFrame(packages)



def rename_folders(directory_path: str, rename_dict: dict) -> None:
    """
    Rename folders in a given directory based on a provided dictionary.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing folders to be renamed.
    rename_dict : dict
        Dictionary with keys as original folder names and values as new folder names.
    """
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path) and item in rename_dict:
            new_name = rename_dict[item]
            new_path = os.path.join(directory_path, new_name)
            os.rename(item_path, new_path)
            print(f"Renamed '{item}' to '{new_name}'")

    print("Folder renaming complete.")


def rename_tiff_files(directory_path: str) -> None:
    """
    Rename .tiff files in a given directory and all its subdirectories.
    Each .tiff file is renamed by taking the last item from its name split by '_'.

    Parameters
    ----------
    directory_path : str
        Path to the directory to search for .tiff files.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".tiff"):
                old_file_path = os.path.join(root, file)
                new_file_name = file.split('_')[-1]
                new_file_path = os.path.join(root, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{file}' to '{new_file_name}'")

    print("Renaming of .tiff files complete.")


def save_unique_tiff_names_to_csv(directory_path: str, csv_file_path: str) -> None:
    """
    Save all the unique names of .tiff files from a directory and its subdirectories to a CSV file.

    Parameters
    ----------
    directory_path : str
        Path to the directory to search for .tiff files.
    csv_file_path : str
        Path to save the CSV file.
    """
    unique_names = set()

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".tiff"):
                unique_names.add(file)

    df = pd.DataFrame(list(unique_names), columns=['FileName'])
    df.to_csv(csv_file_path, index=False)
    print(f"Unique .tiff file names saved to {csv_file_path}.")


def rename_tiff_files_according_to_dict(directory_path: str, rename_dict: dict) -> None:
    """
    Rename .tiff files in a given directory and all its subdirectories according to a provided dictionary.

    Parameters
    ----------
    directory_path : str
        Path to the directory to search for .tiff files.
    rename_dict : dict
        Dictionary with keys as original file names (without extension) and values as new file names.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".tif"):
                file_name_without_extension = os.path.splitext(file)[0]
                if file_name_without_extension in rename_dict:
                    new_file_name = rename_dict[file_name_without_extension] + ".tif"
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, new_file_name)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{file}' to '{new_file_name}'")

    print("Renaming of .tiff files complete.")


def remove_leading_string(directory_path: str, ending_string: str, extension: str = ".tif") -> None:
    """
    Rename files in a given directory if they end with a specified string.
    The specified string is removed from the file title.

    Parameters
    ----------
    directory_path : str
        Path to the directory to search for .tif files.
    ending_string : str
        The string to check for at the end of the file names.
    extension : str, optional
        File extension (default is ".tif").
    """
    for file in os.listdir(directory_path):
        if file.endswith(extension) and file.endswith(ending_string + extension):
            old_file_path = os.path.join(directory_path, file)
            new_file_name = file[:-len(ending_string + extension)] + extension
            new_file_path = os.path.join(directory_path, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{file}' to '{new_file_name}'")

    print(f"Renaming of {extension} files complete.")


def remove_tiff_extensions(series: pd.Series) -> pd.Series:
    """
    Remove .tiff or .tif extensions from the file names in a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The Series containing the file names.

    Returns
    -------
    pd.Series
        Series with the file extensions removed.
    """
    series = series.str.replace(r'\.tiff$', '', regex=True)
    series = series.str.replace(r'\.tif$', '', regex=True)
    return series


def analyze_masks(directory_path: str, file_extension: str = '.tiff') -> pd.DataFrame:
    """
    Analyze mask files in a given directory, returning a DataFrame with information about each mask.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the image files.
    file_extension : str
        File extension of the image files (default is '.tiff').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'ROI', 'x_size', 'y_size', 'Cell_count', and 'Mask_path'.
    """
    data = []

    for file in os.listdir(directory_path):
        if file.endswith(file_extension):
            file_path = os.path.join(directory_path, file)
            image = io.imread(file_path)
            y_size, x_size = image.shape[:2]
            cell_count = len(np.unique(image)) - 1  # Assuming background is labeled as 0

            data.append({
                'ROI': os.path.splitext(file)[0],
                'x_size': x_size,
                'y_size': y_size,
                'Cell_count': cell_count,
                'Mask_path': file_path
            })
            
    return pd.DataFrame(data)


def check_file_names_and_counts(directory_path: str, df: pd.DataFrame, column_name: str, check_folders: bool = False) -> Tuple[bool, set, set]:
    """
    Check that the file names (or folder names) in a given directory match the names in a column of a DataFrame,
    and that all the files (or folders) have the same number of items in each.
    Also provides lists of names that are present in the folder but not in the DataFrame, and vice versa.

    Parameters:
        directory_path (str): Path to the directory containing the files or folders.
        df (pd.DataFrame): DataFrame containing the file or folder names in one of its columns.
        column_name (str): The name of the column in the DataFrame containing the file or folder names.
        check_folders (bool): If True, check folders instead of files.

    Returns:
        bool, set, set: 
            - True if the conditions are met, False otherwise.
            - Set of names present in the folder but not in the DataFrame.
            - Set of names present in the DataFrame but not in the folder.
    """
    # Extract names from the DataFrame
    expected_names = set(df[column_name].unique())

    # Get actual names from the directory
    if check_folders:
        actual_names = set(name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name)))
    else:
        actual_names = set(os.path.splitext(name)[0] for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name)))

    # Find names that do not match
    names_not_in_df = actual_names - expected_names
    names_not_in_folder = expected_names - actual_names

    if names_not_in_df or names_not_in_folder:
        print("Names do not match the DataFrame column.")
        return False, names_not_in_df, names_not_in_folder

    # Check if all files or folders have the same number of items
    item_count = None
    for name in actual_names:
        path = os.path.join(directory_path, name)
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]) if check_folders else len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.startswith(name)])

        if item_count is None:
            item_count = count
        elif item_count != count:
            print(f"'{name}' does not have the same number of items as others.")
            return False, names_not_in_df, names_not_in_folder

    return True, names_not_in_df, names_not_in_folder
    
    
def extract_single_cell(df: pd.DataFrame, images_folder: str, save_directory: str) -> None:
    """
    Analyze images to extract single cell data and save the results in separate CSV files for each ROI.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ROI', 'Mask path' containing mask file paths.
        images_folder (str): Directory containing folders of images named after each ROI.
        save_directory (str): Directory to save the resulting CSV files.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Formula for circularity
    circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter) if r.perimeter > 0 else 0
    
    for _, row in df.iterrows():
        roi = row['ROI']
        mask_path = row['Mask_path']

        # Load the mask as a skimage label object
        mask = io.imread(mask_path)

        # Get image file paths for the current ROI
        roi_folder = os.path.join(images_folder, roi)
        image_files = [f for f in os.listdir(roi_folder) if f.endswith('.tiff')]

        # Initialize a DataFrame to store results for the current ROI
        roi_df = pd.DataFrame()
        roi_df['ROI'] = str(roi)  # Add the ROI name to the DataFrame first

        # Add label numbers and centroid locations
        props = regionprops(mask)
        roi_df['Label'] = [prop.label for prop in props]
        roi_df['X_loc'] = [prop.centroid[1] for prop in props]  # X-coordinate of the centroid
        roi_df['Y_loc'] = [prop.centroid[0] for prop in props]  # Y-coordinate of the centroid
        roi_df['mask_area'] = [prop.area for prop in props]  # Mask area
        roi_df['mask_perimeter'] = [prop.perimeter for prop in props]  # Mask perimeter
        roi_df['mask_circularity'] = [circ(prop) for prop in props]  # Mask circularity

        # Dictionary to hold mean intensities for each image
        mean_intensities_dict = {}

        for image_file in image_files:
            image_path = os.path.join(roi_folder, image_file)
            image = io.imread(image_path)

            # Calculate mean intensity for each label
            mean_intensities = [region.mean_intensity for region in regionprops(mask, image)]

            # Remove extension from image name and add to dictionary
            mean_intensities_dict[os.path.splitext(image_file)[0]] = mean_intensities

        # Add mean intensities to the DataFrame
        for image_name, intensities in mean_intensities_dict.items():
            roi_df[image_name] = intensities

        # Reorder columns to have image columns last
        # image_columns = list(mean_intensities_dict.keys())
        # non_image_columns = [col for col in roi_df.columns if col not in image_columns]
        # roi_df = roi_df[non_image_columns + image_columns]
        
        # This was getting lost for some reason
        roi_df['ROI'] = str(roi)

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(save_directory, f"{roi}.csv")
        roi_df.to_csv(csv_file_path, index=False)
        print(f"Results for {roi} saved to {csv_file_path}.")

        if not os.path.isfile('panel.csv'):
            pd.Series(image_columns, name='target').to_csv('panel.csv')


def create_celltable(input_directory: str = 'Celltables', output_file: str = 'celltable.csv') -> pd.DataFrame:
    """
    Concatenate all CSV files in the specified directory into a single DataFrame,
    make a copy of the index with the heading 'Master_Index', rename 'Label' to 'ObjectNumber',
    and save the final DataFrame as 'celltable.csv'.

    Parameters:
        input_directory (str): Directory containing the CSV files.
        output_file (str): Path to save the final concatenated DataFrame.

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    # List to store individual DataFrames
    dfs = []

    # Iterate through each CSV file in the directory and append to the list
    for file in os.listdir(input_directory):
        if file.endswith('.csv'):
            file_path = os.path.join(input_directory, file)
            df = pd.read_csv(file_path)

            # Rename 'Label' column to 'ObjectNumber'
            df.rename(columns={'Label': 'ObjectNumber'}, inplace=True)

            dfs.append(df)

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dfs)

    # Add 'Master_Index' as a copy of the index
    concatenated_df.reset_index(drop=True, inplace=True)
    concatenated_df['Master_Index'] = list(concatenated_df.index)

    # Save the final DataFrame as 'celltable.csv'
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated DataFrame saved to {output_file}.")
    
    return concatenated_df


def _intensity_mode(mask: np.ndarray, intensity_image: np.ndarray) -> np.ndarray:
    """
    Calculate the mode of pixel intensities for a given mask.

    Parameters:
        mask (np.ndarray): Mask array.
        intensity_image (np.ndarray): Intensity image array.

    Returns:
        np.ndarray: Mode of pixel intensities.
    """
    return scipy.stats.mode(intensity_image[mask], axis=None).mode


def extract_pixel_environment(df: pd.DataFrame, 
                              pixel_folder: str = 'pixel_masks', 
                              save_directory: str = 'celltables_pixel') -> None:
    """
    Extract the most common pixel environment that a cell is in within a labelled image.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ROI', 'Mask path' containing mask file paths.
        pixel_folder (str): Directory containing pixel mask images.
        save_directory (str): Directory to save the resulting CSV files.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    for _, row in df.iterrows():
        roi = row['ROI']
        mask_path = row['Mask_path']

        # Load the mask as a skimage label object
        mask = io.imread(mask_path)

        # Initialize a DataFrame to store results for the current ROI
        roi_df = pd.DataFrame()

        # Add label numbers and centroid locations
        props = regionprops(mask)
        roi_df['Label'] = [prop.label for prop in props]
        roi_df['X_loc'] = [prop.centroid[1] for prop in props]  # X-coordinate of the centroid
        roi_df['Y_loc'] = [prop.centroid[0] for prop in props]  # Y-coordinate of the centroid

        image_path = os.path.join(pixel_folder, roi + "_pixel_mask.tiff")
        image = io.imread(image_path)

        # Calculate mean intensity for each label
        props = regionprops(mask, image, extra_properties=(_intensity_mode,))
        roi_df['pixel_env'] = [prop._intensity_mode for prop in props]

        # This was getting lost for some reason
        roi_df['ROI'] = str(roi)

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(save_directory, f"{roi}.csv")
        roi_df.to_csv(csv_file_path, index=False)
        print(f"Results for {roi} saved to {csv_file_path}.")
        
def get_module_path(module_name: str) -> str:
    """
    Get the file path of a Python module or package.

    Parameters
    ----------
    module_name : str
        The name of the module or package.

    Returns
    -------
    str
        The path to the module file or package directory.
    """
    try:
        # Import the module dynamically
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ImportError(f"Module {module_name} could not be found.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the module's file path
        module_path = module.__file__

        # Check if the module is a package (i.e., a directory)
        if os.path.isdir(module_path):
            # It's a package
            module_path = os.path.dirname(module_path)

        return module_path

    except Exception as e:
        raise ImportError(f"An error occurred while locating the module: {e}")

# Helper function to identify boolean columns saved as objects, and convert them to true boolean columns
def _convert_to_boolean(df):
    for col in df.select_dtypes(include=['object']).columns:
        # Check if all unique values are boolean-like
        unique_vals = pd.Series(df[col].dropna().unique().astype(str))  # Convert to Pandas Series
        unique_vals = [x.lower() for x in unique_vals]  # Lower case
        if all(val in {"true", "false", "yes", "no", "1", "0"} for val in unique_vals):
            # Convert to boolean
            df[col] = df[col].astype("boolean")
    return df


def update_sample_metadata(adata, dictionary_path="metadata/dictionary.csv"):
    # Process dictionary for additional metadata
    dictionary_path = Path(dictionary_path)

    if dictionary_path.exists():
        dictionary_file = pd.read_csv(dictionary_path, index_col='ROI')

        # Automatically convert boolean-like columns
        dictionary_file = _convert_to_boolean(dictionary_file)

        # Get list of columns/metadata from the dictionary file
        cols = [x for x in dictionary_file.columns if 'Example' not in x and 'description' not in x]

        if len(cols) > 0:
            print(f'Dictionary file found with the following columns: {str(cols)}')

            # Ensure `adata.obs` is not a view
            adata.obs = adata.obs.copy()

            for c in cols:
                # Map the data from the dictionary to the adata.obs
                mapped_data = adata.obs['ROI'].map(dictionary_file[c].to_dict())

                # Convert to the appropriate type
                adata.obs[c] = mapped_data.astype(dictionary_file[c].dtype)

            # Make sure boolean columns properly converted
            adata.obs = _convert_to_boolean(adata.obs)

        else:
            print(
                f'Dictionary file found but was empty. Edit dictionary file ({str(dictionary_path)}) to add extra sample-level metadata!'
            )
    else:
        print(f'No dictionary file found, creating a blank file from AnnData at location {str(dictionary_path)}).')
        dictionary_file = adata.obs[~adata.obs.ROI.duplicated()][['ROI', 'ROI_name']].sort_values('ROI').set_index(
            'ROI', drop=True).rename(columns={'ROI_name': 'description'}).copy()
        dictionary_file['Example_1'] = 'Example_info'
        dictionary_file['Example_2'] = 1
        dictionary_file['Example_3'] = True
        dictionary_file.to_csv(dictionary_path)


def reorder_vars_by_expression(
    adata: ad.AnnData,
    vars_of_interest: List[str],
    distance_metric: str = 'euclidean',
    linkage_method: str = 'ward'
) -> List[str]:
    """
    Reorder variables (genes/markers) by hierarchical clustering based on expression patterns.
    
    This function performs hierarchical clustering on a subset of variables to determine
    their optimal ordering based on expression similarity across cells. Useful for
    organizing heatmaps and visualizations.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data.
    vars_of_interest : list of str
        List of variable names (genes/markers) to cluster and reorder.
    distance_metric : str, optional
        Distance metric for clustering. Options include 'euclidean', 'correlation',
        'cosine', etc. See scipy.spatial.distance.pdist for all options.
        Default is 'euclidean'.
    linkage_method : str, optional
        Linkage method for hierarchical clustering. Options include 'ward', 'single',
        'complete', 'average', etc. See scipy.cluster.hierarchy.linkage for all options.
        Default is 'ward'.
    
    Returns
    -------
    list of str
        Ordered list of variable names based on hierarchical clustering.
    
    Examples
    --------
    >>> markers = ['CD3', 'CD4', 'CD8', 'CD20', 'CD68']
    >>> ordered_markers = reorder_vars_by_expression(adata, markers)
    >>> # Use ordered markers for plotting
    >>> sc.pl.heatmap(adata, ordered_markers, groupby='leiden')
    """
    # Subset the data to include only the vars of interest
    adata_subset = adata[:, vars_of_interest]

    # Extract the expression matrix for the vars of interest
    expression_matrix = adata_subset.X
    if issparse(expression_matrix):
        expression_matrix = expression_matrix.toarray()

    # Calculate the distance matrix
    distance_matrix = ssd.pdist(expression_matrix.T, metric=distance_metric)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(distance_matrix, method=linkage_method)

    # Get the order of the vars based on the clustering
    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
    ordered_var_indices = dendrogram['leaves']
    ordered_vars = adata_subset.var_names[ordered_var_indices]

    return ordered_vars.tolist()


def leiden_on_subset(
    adata: ad.AnnData,
    restrict_to: Optional[Tuple[str, List[Any]]] = None,
    *,
    genes: Optional[List[str]] = None,
    gene_layer: Optional[str] = None,
    subset_key_name: str = "leiden_subset",
    base_label_key: Optional[str] = None,
    leiden_resolution: float = 1.0,
    neighbors_kwargs: Optional[dict] = None,
    leiden_kwargs: Optional[dict] = None,
    use_rep: Optional[str] = None,
    copy: bool = False,
) -> ad.AnnData:
    """
    Run Leiden clustering on a cell subset and/or gene subset and merge labels back.
    
    This function allows you to perform Leiden clustering on a specific subset of cells
    and/or using a specific subset of genes, then integrates the results back into the
    full dataset. Cells outside the subset can retain existing labels or be marked separately.

    Parameters
    ----------
    adata : AnnData
        Full AnnData object containing the complete dataset.
    restrict_to : tuple of (str, list), optional
        Tuple of (obs_key, values) defining the cell subset to analyze.
        For example, ('cell_type', ['T cells', 'B cells']) to restrict analysis
        to only T cells and B cells. Default is None (use all cells).
    genes : list of str, optional
        List of gene/marker names to subset for analysis (adata[:, genes]).
        If provided, only these variables will be used for neighborhood graph
        construction. Default is None (use all genes).
    gene_layer : str, optional
        Layer name to use instead of X when genes are provided.
        Useful for using raw counts or other alternative data representations.
        Default is None (use X).
    subset_key_name : str, optional
        Name for the new obs column containing the merged labels.
        Default is 'leiden_subset'.
    base_label_key : str, optional
        Existing obs column with labels for non-subset cells.
        If None, non-subset cells are labeled as 'outside_subset'.
        Default is None.
    leiden_resolution : float, optional
        Resolution parameter for Leiden clustering. Higher values produce
        more clusters. Default is 1.0.
    neighbors_kwargs : dict, optional
        Additional keyword arguments passed to sc.pp.neighbors().
        Default is None.
    leiden_kwargs : dict, optional
        Additional keyword arguments passed to sc.tl.leiden().
        Default is None.
    use_rep : str, optional
        Representation to use for neighbors computation (e.g., 'X_pca').
        MUST be None or 'X' if genes parameter is not None.
        Default is None.
    copy : bool, optional
        If True, return a copy of the AnnData object. If False, modify in place.
        Default is False.
    
    Returns
    -------
    AnnData
        Modified AnnData object with new clustering labels in obs[subset_key_name].
        Subset cells are labeled as 'subset_0', 'subset_1', etc.
        Non-subset cells retain their base labels or are marked as 'outside_subset'.
    
    Raises
    ------
    ValueError
        If the cell subset is empty after applying restrictions.
    KeyError
        If specified genes are not found in the dataset.
    
    Examples
    --------
    >>> # Cluster only T cells using all genes
    >>> leiden_on_subset(adata, restrict_to=('cell_type', ['T cells']))
    
    >>> # Cluster all cells but only using specific markers
    >>> markers = ['CD3', 'CD4', 'CD8']
    >>> leiden_on_subset(adata, genes=markers, subset_key_name='tcell_leiden')
    
    >>> # Cluster specific cells with specific genes
    >>> leiden_on_subset(
    ...     adata,
    ...     restrict_to=('tissue', ['tumor']),
    ...     genes=['CD68', 'CD163', 'HLA-DR'],
    ...     leiden_resolution=0.5,
    ...     subset_key_name='tumor_macrophage_clusters'
    ... )
    """

    if copy:
        adata = adata.copy()

    # ---- Cell mask ----
    mask = pd.Series(True, index=adata.obs_names)

    if restrict_to is not None:
        obs_key, values = restrict_to
        mask &= adata.obs[obs_key].isin(values)

    if mask.sum() == 0:
        raise ValueError("Cell subset is empty.")

    # ---- Subset cells ----
    adata_sub = adata[mask].copy()

    # ---- Subset genes ----
    if genes is not None:
        missing = set(genes) - set(adata_sub.var_names)
        if missing:
            raise KeyError(f"Genes not found: {missing}")

        adata_sub = adata_sub[:, genes].copy()
        use_rep = None  # force raw X usage

        if gene_layer is not None:
            adata_sub.X = adata_sub.layers[gene_layer]

    # ---- Neighbors ----
    sc.pp.neighbors(
        adata_sub,
        use_rep=use_rep,
        **(neighbors_kwargs or {})
    )

    # ---- Leiden ----
    sc.tl.leiden(
        adata_sub,
        resolution=leiden_resolution,
        key_added="_leiden_subset_tmp",
        **(leiden_kwargs or {})
    )

    # ---- Merge labels back ----
    merged = (
        adata.obs[base_label_key].astype(str)
        if base_label_key is not None
        else pd.Series("outside_subset", index=adata.obs_names)
    )

    merged.loc[adata_sub.obs_names] = (
        "subset_" + adata_sub.obs["_leiden_subset_tmp"].astype(str)
    ).values

    adata.obs[subset_key_name] = merged

    return adata
