# Standard Library Imports
import datetime
import os
from os import getlogin
from pathlib import Path
import subprocess
from types import ModuleType
from typing import List, Union, Optional, Tuple, Any
from collections.abc import Iterable
import math
import importlib.util
from copy import copy

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
    
    
def extract_single_cell(
    df: pd.DataFrame | None,
    images_folder: str,
    save_directory: str,
    *,
    rois: list[str] | None = None,
    masks_folder: str | None = None,
    images_to_include: list[str] | str | None = None,
    measurements: list[str] | str = "mean",
) -> None:
    """Extract single-cell intensities per ROI.

    This function saves one CSV per ROI, containing per-cell geometry from the ROI mask and
    per-cell intensities from each image in the ROI's image folder.

    Inputs
    ------
    You can provide either:
    - ``df``: a DataFrame with at least ``ROI`` and ``Mask_path`` columns, OR
    - ``rois`` + ``masks_folder``: ROI names and a folder containing ROI masks named
      ``<ROI>.tif`` or ``<ROI>.tiff`` (auto-detected).

    Parameters
    ----------
    df:
        DataFrame with ROI/mask mapping. If provided, must contain columns ``ROI`` and
        either ``Mask_path`` or ``Mask path``.
    images_folder:
        Directory containing one subfolder per ROI, with image files inside.
    save_directory:
        Output directory to write per-ROI CSV files.
    rois:
        List of ROI names (used when ``df`` is None).
    masks_folder:
        Folder containing masks named after each ROI (used when ``df`` is None).
    images_to_include:
        Optional list (or single string) of image names to include (matched by filename stem).
        If None, all images in the ROI folder are used.
    measurements:
        Intensity measurements to extract per image. Defaults to ``"mean"``.
        Supported keys: ``mean``, ``min``, ``max`` and also their regionprops equivalents
        (e.g. ``mean_intensity``).
        When a measurement is not mean, output columns are suffixed with ``_{measurement}``.
    """
    os.makedirs(save_directory, exist_ok=True)

    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
            return list(value)
        return [value]

    def _normalize_measurement(measurement: str) -> tuple[str, str, bool]:
        """Return (regionprops_attr, suffix_key, is_mean)."""
        m = str(measurement).strip().lower()
        if m in {"mean", "avg", "average", "mean_intensity"}:
            return "mean_intensity", "mean", True
        if m in {"min", "min_intensity"}:
            return "min_intensity", m, False
        if m in {"max", "max_intensity"}:
            return "max_intensity", m, False
        raise ValueError(
            f"Unsupported measurement: {measurement!r}. "
            "Use 'mean', 'min', 'max' (or 'mean_intensity', 'min_intensity', 'max_intensity')."
        )

    include_list = _as_list(images_to_include)
    include_stems = None
    if include_list is not None:
        include_stems = {os.path.splitext(str(x))[0] for x in include_list}

    measurement_list = _as_list(measurements) or ["mean"]
    measurement_specs = [_normalize_measurement(m) for m in measurement_list]

    # Formula for circularity
    circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter) if r.perimeter > 0 else 0

    roi_mask_pairs: list[tuple[str, str]] = []
    if df is not None:
        if rois is not None or masks_folder is not None:
            raise ValueError("Provide either df, or rois+masks_folder (not both).")

        if "ROI" not in df.columns:
            raise ValueError("df must contain a 'ROI' column")
        mask_col = "Mask_path" if "Mask_path" in df.columns else "Mask path" if "Mask path" in df.columns else None
        if mask_col is None:
            raise ValueError("df must contain a 'Mask_path' or 'Mask path' column")

        roi_mask_pairs = [(str(row["ROI"]), str(row[mask_col])) for _, row in df.iterrows()]
    else:
        if not rois or not masks_folder:
            raise ValueError("When df is None, you must provide both rois and masks_folder")
        for roi in rois:
            roi_str = str(roi)
            tif_path = os.path.join(masks_folder, f"{roi_str}.tif")
            tiff_path = os.path.join(masks_folder, f"{roi_str}.tiff")
            if os.path.isfile(tif_path):
                roi_mask_pairs.append((roi_str, tif_path))
            elif os.path.isfile(tiff_path):
                roi_mask_pairs.append((roi_str, tiff_path))
            else:
                raise FileNotFoundError(
                    f"No mask found for ROI {roi_str!r} in {masks_folder!r}. "
                    "Expected '<ROI>.tif' or '<ROI>.tiff'."
                )

    panel_columns: list[str] | None = None

    for roi, mask_path in roi_mask_pairs:
        # Load the mask as a skimage label object
        mask = io.imread(mask_path)

        # Get image file paths for the current ROI
        roi_folder = os.path.join(images_folder, roi)
        if not os.path.isdir(roi_folder):
            raise FileNotFoundError(f"ROI folder not found: {roi_folder!r}")

        image_files = [
            f
            for f in os.listdir(roi_folder)
            if os.path.isfile(os.path.join(roi_folder, f))
            and str(f).lower().endswith((".tif", ".tiff"))
        ]
        image_files.sort()

        if include_stems is not None:
            image_files = [f for f in image_files if os.path.splitext(f)[0] in include_stems]

        # Initialize a DataFrame to store results for the current ROI
        roi_df = pd.DataFrame()
        roi_df["ROI"] = str(roi)

        # Add label numbers and centroid locations
        props_geom = regionprops(mask)
        roi_df["Label"] = [prop.label for prop in props_geom]
        roi_df["X_loc"] = [prop.centroid[1] for prop in props_geom]  # X-coordinate of the centroid
        roi_df["Y_loc"] = [prop.centroid[0] for prop in props_geom]  # Y-coordinate of the centroid
        roi_df["mask_area"] = [prop.area for prop in props_geom]  # Mask area
        roi_df["mask_perimeter"] = [prop.perimeter for prop in props_geom]  # Mask perimeter
        roi_df["mask_circularity"] = [circ(prop) for prop in props_geom]  # Mask circularity

        # Add intensity measurements per image
        image_columns_this_roi: list[str] = []
        for image_file in image_files:
            image_path = os.path.join(roi_folder, image_file)
            image = io.imread(image_path)

            props_intensity = regionprops(mask, intensity_image=image)
            image_stem = os.path.splitext(image_file)[0]

            for attr, suffix_key, is_mean in measurement_specs:
                col_name = image_stem if is_mean else f"{image_stem}_{suffix_key}"
                if col_name in roi_df.columns:
                    raise ValueError(
                        f"Duplicate output column {col_name!r}. "
                        "This can happen if measurements contain duplicates."
                    )

                try:
                    values = [getattr(region, attr) for region in props_intensity]
                except AttributeError as e:
                    raise ValueError(
                        f"Measurement attribute {attr!r} not available on skimage RegionProperties"
                    ) from e

                roi_df[col_name] = values
                image_columns_this_roi.append(col_name)

        # This was getting lost for some reason
        roi_df["ROI"] = str(roi)

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(save_directory, f"{roi}.csv")
        roi_df.to_csv(csv_file_path, index=False)
        print(f"Results for {roi} saved to {csv_file_path}.")

        # Write panel.csv once, preserving legacy behavior (writes to CWD)
        if panel_columns is None:
            panel_columns = image_columns_this_roi
        if not os.path.isfile("panel.csv") and panel_columns is not None:
            pd.Series(panel_columns, name="target").to_csv("panel.csv", index=False)


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
    return_new_names: bool = False,
) -> Union[ad.AnnData, Tuple[ad.AnnData, List[str]]]:
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
    return_new_names : bool, optional
        If True, return a tuple of (adata, new_population_names) where new_population_names
        is a sorted list of the newly created population labels for subset cells.
        Default is False.
    
    Returns
    -------
    AnnData or tuple of (AnnData, list of str)
        Modified AnnData object with new clustering labels in obs[subset_key_name].
        Subset cells are labeled as '{base_label}_{leiden_number}', e.g., 'Tumor_0', 'Tumor_1'.
        If base_label_key is None, subset cells are labeled as 'cluster_{leiden_number}'.
        Non-subset cells retain their base labels or are marked as 'outside_subset'.
        If return_new_names=True, returns a tuple of (adata, new_population_names).
    
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
    
    >>> # Cluster all cells but only using specific markers, return new names
    >>> markers = ['CD3', 'CD4', 'CD8']
    >>> adata, new_pops = leiden_on_subset(
    ...     adata, genes=markers, subset_key_name='tcell_leiden',
    ...     return_new_names=True
    ... )
    >>> print(new_pops)  # ['cluster_0', 'cluster_1', 'cluster_2']
    
    >>> # Cluster specific cells with specific genes, preserving base labels
    >>> leiden_on_subset(
    ...     adata,
    ...     restrict_to=('tissue', ['tumor']),
    ...     genes=['CD68', 'CD163', 'HLA-DR'],
    ...     base_label_key='cell_type',
    ...     leiden_resolution=0.5,
    ...     subset_key_name='tumor_macrophage_clusters'
    ... )
    >>> # Results in labels like 'Macrophage_0', 'Macrophage_1', 'Dendritic_0', etc.
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

    # Create new labels by combining base label with leiden cluster number
    if base_label_key is not None:
        # Use existing labels as prefix
        base_labels_subset = adata.obs.loc[adata_sub.obs_names, base_label_key].astype(str)
    else:
        # Use "cluster" as default prefix when no base label exists
        base_labels_subset = pd.Series("cluster", index=adata_sub.obs_names)
    
    new_labels = (
        base_labels_subset + "_" + adata_sub.obs["_leiden_subset_tmp"].astype(str)
    )
    
    merged.loc[adata_sub.obs_names] = new_labels.values

    adata.obs[subset_key_name] = merged

    if return_new_names:
        # Get unique new population names created for the subset, sorted
        new_population_names = sorted(new_labels.unique().tolist())
        return adata, new_population_names
    
    return adata


def run_population_subclustering(
    adata,
    populations=None,
    resolutions=(0.3,),
    base_label_key="population",
    use_rep="X_biobatchnet",
    genes=None,
    show_figures=True,
    save_figures=True,
    figure_dir="Figures",
    remap_csv_path="subcluster_to_final_population.csv",
    umap_dot_size=1,
    matrixplot_vmax=0.3,
    save_individual_umaps=True,
):
    """
    Run Leiden subclustering on population subsets and generate remapping CSV.
    
    Performs Leiden clustering on specific cell populations at multiple resolutions,
    generates comprehensive visualizations (UMAPs, matrixplots), and creates a CSV
    file for mapping subclusters to final population annotations.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data and population labels.
    populations : list of str, optional
        List of populations (from base_label_key) to subcluster. If None, all unique
        populations in base_label_key will be subclustered. Default is None.
    resolutions : float or tuple of float, optional
        Leiden resolution(s) to test. Higher values produce more clusters.
        Can be a single float or iterable of floats. Default is (0.3,).
    base_label_key : str, optional
        Column in adata.obs containing base population labels. Default is 'population'.
    use_rep : str, optional
        Representation to use for neighborhood graph (e.g., 'X_pca', 'X_biobatchnet').
        Default is 'X_biobatchnet'.
    genes : list of str or dict, optional
        Gene subset(s) to use for subclustering. You can pass:
        - a list of genes applied to all populations, or
        - a dict mapping population names (values of ``base_label_key``) to gene lists.
          Include a "default" key for a fallback list; otherwise all genes are used
          when a population key is missing. If None, uses all genes.
    show_figures : bool, optional
        Whether to display figures interactively. Default is True.
    save_figures : bool, optional
        Whether to save figures to disk. Default is True.
    figure_dir : str, optional
        Directory to save figures. Created if it doesn't exist. Default is 'Figures'.
    remap_csv_path : str, optional
        Path to save the remapping CSV file. Default is 'subcluster_to_final_population.csv'.
    umap_dot_size : int or float, optional
        Point size for UMAP plots. Default is 1.
    matrixplot_vmax : float, optional
        Maximum value for matrixplot color scale. Default is 0.3.
    save_individual_umaps : bool, optional
        Whether to save individual UMAP plots highlighting each subcluster.
        Creates a subdirectory structure: figure_dir/umap_individual/subcluster_col/cluster.png.
        Each plot shows one subcluster in red with all others in grey. Default is True.
    
    Returns
    -------
    adata : AnnData
        Modified AnnData with new subcluster columns added to obs. Each column is named
        '{base_label_key}_res{resolution}_subset_{population}' and contains labels like
        '{population}_0', '{population}_1', etc.
    remap_df : pd.DataFrame
        DataFrame mapping subclusters to final populations with columns:
        - subcluster_column: Name of the obs column containing subclusters
        - parent_population: Original population that was subclustered
        - resolution: Leiden resolution used
        - subcluster: Subcluster label (e.g., 'Tumor_0')
        - final_population: Final population name (initially same as subcluster)
    new_pops_dict : dict
        Dictionary mapping subcluster column names to lists of new population labels.
        Keys are column names, values are sorted lists of population labels.
    
    Notes
    -----
    The function generates three types of visualizations:
    1. Combined UMAP showing all subclusters colored by cluster assignment
    2. Matrixplot showing marker expression across subclusters with dendrogram
    3. Individual UMAP plots (if save_individual_umaps=True) highlighting each subcluster
    
    The remapping CSV can be manually edited to assign meaningful final population names,
    then applied using apply_subcluster_remap().
    
    Examples
    --------
    >>> # Subcluster T cells at multiple resolutions
    >>> adata, remap_df, new_pops = run_population_subclustering(
    ...     adata,
    ...     populations=['T cells'],
    ...     resolutions=[0.3, 0.5, 0.7],
    ...     base_label_key='population',
    ...     use_rep='X_pca'
    ... )
    
    >>> # Subcluster all populations using specific markers
    >>> markers = ['CD3', 'CD4', 'CD8', 'CD20', 'CD68']
    >>> adata, remap_df, new_pops = run_population_subclustering(
    ...     adata,
    ...     genes=markers,
    ...     save_individual_umaps=True
    ... )
    """

    def _ensure_iterable(x):
        if isinstance(x, str):
            return [x]
        if isinstance(x, Iterable):
            return list(x)
        return [x]

    # Normalize gene input: support dict mapping population -> gene list
    genes_map = None
    if genes is None:
        default_genes = list(adata.var_names)
    elif isinstance(genes, dict):
        genes_map = {str(k): _ensure_iterable(v) for k, v in genes.items()}
        default_genes = genes_map.get("default", list(adata.var_names))
    else:
        default_genes = _ensure_iterable(genes)

    if populations is None:
        populations = adata.obs[base_label_key].unique().tolist()
    else:
        populations = _ensure_iterable(populations)

    resolutions = _ensure_iterable(resolutions)
    if save_figures:
        os.makedirs(figure_dir, exist_ok=True)

    # directory for individual subcluster UMAPs
    if save_figures and save_individual_umaps:
        indiv_umap_dir = os.path.join(figure_dir, "umap_individual")
        os.makedirs(indiv_umap_dir, exist_ok=True)

    new_pops_dict = {}
    remap_rows = []

    for pop in populations:
        print(f'Asessing population: {pop}')

        pop_key = str(pop)
        pop_genes = genes_map.get(pop_key, default_genes) if genes_map is not None else default_genes
        if pop_genes is None:
            pop_genes = list(adata.var_names)

        for resolution in resolutions:

            subcluster_col = f"{base_label_key}_res{resolution}_subset_{pop}"

            adata, new_pops = leiden_on_subset(
                adata,
                restrict_to=(base_label_key, [pop]),
                base_label_key=base_label_key,
                subset_key_name=subcluster_col,
                use_rep=use_rep,
                leiden_resolution=resolution,
                genes=pop_genes,
                return_new_names=True,
            )

            new_pops_dict[subcluster_col] = copy(new_pops)

            # ---- combined UMAP ----
            if show_figures or save_figures:
                sc.pl.umap(
                    adata,
                    color=subcluster_col,
                    s=umap_dot_size,
                    ncols=1,
                    groups=new_pops,
                    title=f"Subsetting on {pop}, res: {resolution}",
                    show=show_figures,
                    save=f"_{subcluster_col}_umap.png" if save_figures else None,
                )

                if not show_figures:
                    plt.close()
            
            # ---- individual subcluster UMAPs ----
            if save_figures and save_individual_umaps:
                sub_dir = os.path.join(indiv_umap_dir, subcluster_col)
                os.makedirs(sub_dir, exist_ok=True)
            
                for cl in new_pops:
                    # Masks
                    focus_mask = adata.obs[subcluster_col] == cl
                    background_mask = ~focus_mask
            
                    # Sort order: background first, then focus
                    sorted_adata = ad.concat(
                        [adata[background_mask], adata[focus_mask]],
                        axis=0,
                        label="__temp__",
                        keys=["background", "focus"],
                        index_unique=None
                    )
            
                    # Create color label column
                    label_col = f"__highlight_{cl}"
                    sorted_adata.obs[label_col] = sorted_adata.obs[subcluster_col].copy()
            
                    # Ensure we can assign "background" by adding it to the category list
                    if isinstance(sorted_adata.obs[label_col].dtype, pd.CategoricalDtype):
                        if "background" not in sorted_adata.obs[label_col].cat.categories:
                            sorted_adata.obs[label_col] = sorted_adata.obs[label_col].cat.add_categories(["background"])
            
                    # Safely assign background label
                    sorted_adata.obs.loc[sorted_adata.obs["__temp__"] == "background", label_col] = "background"
            
                    # Get all categories for the label column
                    categories = sorted_adata.obs[label_col].cat.categories
                    
                    # Assign red to focus cluster, grey to everything else
                    color_map = {
                        cat: ("red" if cat == cl else "#d3d3d3")
                        for cat in categories
                    }

                    # Plot
                    sc.pl.umap(
                        sorted_adata,
                        color=label_col,
                        palette=color_map,
                        s=umap_dot_size,
                        title=f"{cl}",
                        show=False,
                        legend_loc="none",  # hides the legend
                    )
            
                    plt.savefig(
                        os.path.join(sub_dir, f"{cl}.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

            # ---- dendrogram + matrixplot ----
            sc.tl.dendrogram(adata, groupby=subcluster_col)

            mp = sc.pl.matrixplot(
                adata[adata.obs[subcluster_col].isin(new_pops)],
                var_names=reorder_vars_by_expression(adata, pop_genes),
                groupby=subcluster_col,
                dendrogram=False,
                vmax=matrixplot_vmax,
                title=f"Subsetting on {pop}, res: {resolution}",
                return_fig=True,
                show=False,
            )

            mp.add_totals().style(edge_color="black")

            if save_figures:
                mp.savefig(
                    os.path.join(
                        figure_dir,
                        f"{subcluster_col}_matrixplot.png",
                    )
                )

            # ---- remap rows ----
            for cl in new_pops:
                remap_rows.append({
                    "subcluster_column": subcluster_col,
                    "parent_population": pop,
                    "resolution": resolution,
                    "subcluster": cl,
                    "final_population": cl,
                })

    remap_df = (
        pd.DataFrame(remap_rows)
        .sort_values(["parent_population", "resolution", "subcluster"])
    )

    remap_df.to_csv(remap_csv_path, index=False)

    return adata, remap_df, new_pops_dict

def apply_subcluster_remap(
    adata,
    remap_csv_path="subcluster_to_final_population.csv",
    base_label_key="population",
    new_label_key="final_population",
):
    """
    Apply subcluster-to-final-population mapping from edited CSV file.
    
    Reads a remapping CSV file (typically created and edited after running
    run_population_subclustering) and applies the final population assignments
    to create a new consolidated annotation column in adata.obs.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing subcluster columns in obs.
    remap_csv_path : str, optional
        Path to the remapping CSV file. Should contain columns:
        - subcluster_column: Name of the obs column with subclusters
        - subcluster: Original subcluster label
        - final_population: Desired final population name
        Default is 'subcluster_to_final_population.csv'.
    base_label_key : str, optional
        Column in adata.obs containing base population labels. Used as fallback
        for cells not assigned in any subcluster column. Default is 'population'.
    new_label_key : str, optional
        Name for the new column in adata.obs that will contain final population
        assignments. Default is 'final_population'.
    
    Returns
    -------
    AnnData
        Modified AnnData with new column obs[new_label_key] containing final
        population assignments based on the remapping file.
    
    Notes
    -----
    The function prioritizes assignments in the order they appear in the CSV.
    For each cell:
    1. Checks all subcluster columns in order
    2. If the cell has a non-null value in a subcluster column, looks up the
       corresponding final population from the remapping CSV
    3. If no mapping found or cell not in any subcluster, uses base_label_key value
    
    This allows you to manually curate subcluster names in the CSV before applying
    them to the dataset.
    
    Examples
    --------
    >>> # Standard workflow
    >>> adata, remap_df, _ = run_population_subclustering(adata)
    >>> # Edit subcluster_to_final_population.csv manually
    >>> # Then apply the edited mappings:
    >>> adata = apply_subcluster_remap(adata)
    >>> # Now adata.obs['final_population'] contains curated labels
    
    >>> # Custom file path and column names
    >>> adata = apply_subcluster_remap(
    ...     adata,
    ...     remap_csv_path='my_custom_remap.csv',
    ...     base_label_key='cell_type',
    ...     new_label_key='refined_cell_type'
    ... )
    """

    remap_df = pd.read_csv(remap_csv_path)

    lookup = (
        remap_df
        .set_index(["subcluster_column", "subcluster"])
        ["final_population"]
        .to_dict()
    )

    # Only consider subcluster columns that actually exist in adata.obs
    all_subcluster_cols = remap_df["subcluster_column"].unique()
    subcluster_cols = [col for col in all_subcluster_cols if col in adata.obs.columns]

    def assign_final(row):
        for col in subcluster_cols:
            v = row[col]
            if pd.notna(v):
                # Look up the final population
                final_pop = lookup.get((col, v))
                if final_pop is not None:
                    return final_pop
        # If no mapping found in any subcluster column, fall back to base label
        return row[base_label_key]

    adata.obs[new_label_key] = adata.obs.apply(assign_final, axis=1)

    return adata


def plot_umap_highlight_clusters(
    adata,
    subcluster_col,
    focus_color="red",
    background_color="#d3d3d3",
    point_size=3,
    legend_loc="none",
    show=True,
    filter_obs=None,
    filter_values=None,
    clusters=None,
):
    """
    Plot UMAPs highlighting each cluster in `subcluster_col` one at a time.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    subcluster_col : str
        Column in `adata.obs` containing cluster labels (categorical).
    focus_color : str, optional
        Color for the highlighted cluster (default: "red").
    background_color : str, optional
        Color for background points (default: light grey).
    point_size : int or float, optional
        Point size for UMAP plot (default: 3).
    legend_loc : str, optional
        Legend location passed to `sc.pl.umap` (default: "none").
    show : bool, optional
        Whether to show the plot immediately (default: True).
    filter_obs : str, optional
        Optional obs column used to restrict which cells get highlighted (e.g., ROI/case).
        Cells outside the filter remain as background.
    filter_values : Union[list, Any], optional
        Value or list of values from `filter_obs` to highlight. Required when `filter_obs` is set.
    clusters : Union[list, Any], optional
        Specific cluster label(s) in `subcluster_col` to plot. If None, plot all clusters.
    """

    if subcluster_col not in adata.obs:
        raise ValueError(f"{subcluster_col} not found in adata.obs")

    if not isinstance(adata.obs[subcluster_col].dtype, pd.CategoricalDtype):
        raise TypeError(f"{subcluster_col} must be a categorical column")

    # Optional highlighting mask: keep all cells on UMAP, only restrict highlighted ones
    if filter_obs is not None:
        if filter_obs not in adata.obs:
            raise ValueError(f"{filter_obs} not found in adata.obs")
        if filter_values is None:
            raise ValueError("filter_values must be provided when filter_obs is set")

        allowed_values = _to_list(filter_values)
        highlight_mask = adata.obs[filter_obs].isin(allowed_values)

        if highlight_mask.sum() == 0:
            raise ValueError(
                f"No cells found for {filter_obs} with values {allowed_values}"
            )
    else:
        highlight_mask = pd.Series(True, index=adata.obs_names)

    categories_to_plot = adata.obs[subcluster_col].cat.categories

    if clusters is not None:
        requested = _to_list(clusters)
        missing = [c for c in requested if c not in categories_to_plot]
        if missing:
            raise ValueError(f"Requested clusters not found in {subcluster_col}: {missing}")
        # Preserve user-requested order
        categories_to_plot = requested

    for cl in categories_to_plot:
        if ((adata.obs[subcluster_col] == cl) & highlight_mask).sum() == 0:
            continue  # Skip categories absent in the filtered subset
        # Masks
        focus_mask = (adata.obs[subcluster_col] == cl) & highlight_mask
        background_mask = ~focus_mask

        # Sort order: background first, then focus
        sorted_adata = ad.concat(
            [adata[background_mask], adata[focus_mask]],
            axis=0,
            label="__temp__",
            keys=["background", "focus"],
            index_unique=None,
        )

        # Create color label column
        label_col = f"__highlight_{cl}"
        sorted_adata.obs[label_col] = sorted_adata.obs[subcluster_col].copy()

        # Ensure "background" is a valid category
        if isinstance(sorted_adata.obs[label_col].dtype, pd.CategoricalDtype):
            if "background" not in sorted_adata.obs[label_col].cat.categories:
                sorted_adata.obs[label_col] = (
                    sorted_adata.obs[label_col]
                    .cat.add_categories(["background"])
                )

        # Assign background label
        sorted_adata.obs.loc[
            sorted_adata.obs["__temp__"] == "background", label_col
        ] = "background"

        # Color map: highlight cluster vs background
        categories = sorted_adata.obs[label_col].cat.categories
        color_map = {
            cat: (focus_color if cat == cl else background_color)
            for cat in categories
        }

        # Plot
        sc.pl.umap(
            sorted_adata,
            color=label_col,
            palette=color_map,
            s=point_size,
            title=str(cl),
            legend_loc=legend_loc,
            show=show,
        )