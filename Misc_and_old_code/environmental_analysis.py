# Standard Library Imports
import itertools
import os
from os import listdir
from os.path import abspath, exists, isfile, join
from glob import glob
from copy import copy
import pathlib
from pathlib import Path
from multiprocessing import Pool
from typing import List, Optional, Union, Tuple

# Third-Party Imports
import anndata as ad
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import tifffile as tp

from scipy.stats import kurtosis, skew

from skimage.draw import rectangle
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread, imsave
from skimage.measure import label, regionprops, regionprops_table
from skimage.util import img_as_ubyte, img_as_int, img_as_uint



def analyse_environments(adata: ad.AnnData,
                         samples_list: List[str],
                         marker_list: List[str],
                         mode: str = 'summary',
                         radius: int = 10,
                         num_cores: int = 4,
                         folder_dir: str = 'images',
                         roi_id: str = 'ROI',
                         x_loc: str = 'X_loc',
                         y_loc: str = 'Y_loc',
                         cell_index_id: str = 'Master_Index',
                         quantile: Optional[float] = 0.999,
                         parameters: Optional[List[str]] = None,
                         return_quant_table: bool = False,
                         invert_value: Optional[int] = None
                         ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Analyzes the environmental texture around cells in given images, extracting specified parameters.

    Parameters
    ----------
    adata : ad.AnnData
        Anndata object where the cell locations are stored.
    samples_list : List[str]
        List of sample identifiers to include in the analysis.
    marker_list : List[str]
        List of markers to analyze.
    mode : str, optional
        Operation mode, 'summary' by default.
    radius : int, optional
        Radius of the square area around the cell for analysis.
    num_cores : int, optional
        Number of cores to use for multiprocessing.
    folder_dir : str, optional
        Directory containing the image files.
    roi_id : str, optional
        Column in `adata` indicating the ROI.
    x_loc : str, optional
        Column in `adata` specifying the x-coordinate of the cell.
    y_loc : str, optional
        Column in `adata` specifying the y-coordinate of the cell.
    cell_index_id : str, optional
        Identifier for cells in the dataframe.
    quantile : Optional[float], optional
        Quantile to determine the maximum intensity for image scaling. If None, no normalization is applied.
        The quantile is calculated over all the images for that marker.
    parameters : Optional[List[str]], optional
        List of strings specifying which features to calculate. Features can include 'Mean', 'Median', 'Std',
        'Kurtosis', 'Skew', and texture features like 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy',
        'Correlation', 'ASM', as well as quantiles (e.g., 'Quantile_0.5'). Default parameter list is:
        ['Mean', 'Std', 'Quantile_0.1', 'Quantile_0.5', 'Quantile_0.9'].
    return_quant_table : bool, optional
        If True, returns a table of quantile values.
    invert_value : Optional[int], optional
        Pixel values for images will be subtracted from this number, if given.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        Depending on `return_quant_table`, returns either a DataFrame of results or a tuple containing
        the DataFrame and a quantile table.
    """
    master_list = []
    quant_list = []
    
    parameters = ['Mean', 'Std', 'Quantile_0.1', 'Quantile_0.5', 'Quantile_0.9'] if parameters is None else parameters

    for marker in marker_list:
        print('Processing Marker:', marker)

        img_data = _load_imgs_from_directory(folder_dir, marker, quiet=True)
        if img_data is None:
            continue
        Img_collect, Img_file_list, img_folders = img_data
        roi_list = [Path(x).stem for x in img_folders]

        # Determine quantile value for image scaling
        quant_value = np.mean([np.quantile(img, quantile) for img in Img_collect]) if quantile is not None else None
        quant_list.append(quant_value)
       
        # Capture ROI-level data for this marker
        roi_datas = []
        
        for image, img_file_name, roi in tqdm(zip(Img_collect, Img_file_list, roi_list), total=len(samples_list)):
            if roi not in samples_list:
                continue

            if invert_value:
                image = invert_value - image
            
            print('Analyzing ROI:', roi)
            adata_roi = adata.obs[adata.obs[roi_id] == roi]
            cell_coords = zip(adata_roi[y_loc], adata_roi[x_loc])

            with Pool(processes=num_cores) as pool:
                results = pool.starmap(_analyse_cell_features, [
                    (image, roi, cell_id, marker, quant_value, radius, coord, parameters, cell_index_id)
                    for coord, cell_id in zip(cell_coords, adata_roi[cell_index_id])
                ])

            roi_df = pd.concat([res for res in results if not res.empty and not res.isna().all().all()])
            roi_datas.append(roi_df.dropna().copy())
        
        master_list.append(pd.concat(roi_datas).dropna())    
        
    # Concatenate data from all markers
    final_data = pd.concat(master_list, axis=1).dropna()

    if return_quant_table:
        quant_table = pd.DataFrame(list(zip(marker_list, quant_list)), columns=['Marker', 'Max Value Images Scaled To'])
        return final_data, quant_table

    return final_data


def _analyse_cell_features(image: np.ndarray,
                          roi: str,
                          cell_id: str,
                          marker: str,
                          quant_value: Optional[float],
                          radius: int,
                          coordinates: Tuple[int, int],
                          parameters: List[str],
                          cell_index_id: str) -> pd.DataFrame:
    """
    Analyzes specified features of a cell within a given image region based on parameters.

    Parameters
    ----------
    image : np.ndarray
        The image data array.
    roi : str
        The region of interest identifier.
    cell_id : str
        Identifier for the cell within the image.
    marker : str
        Marker associated with the cell.
    quant_value : Optional[float]
        Upper limit for clipping and normalizing the image data.
    radius : int
        Radius of the square area around the cell for analysis.
    coordinates : Tuple[int, int]
        Tuple specifying the (y, x) coordinates of the cell.
    parameters : List[str]
        List of strings specifying which features to calculate.
    cell_index_id : str
        Column name to be used as index in the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains measurements for the specified features
        for the given cell, indexed by `cell_index_id`.
    """
    y, x = map(int, coordinates)
    y_min, y_max = max(0, y - radius), min(image.shape[0], y + radius)
    x_min, x_max = max(0, x - radius), min(image.shape[1], x + radius)

    if y_min == 0 and y - radius < 0 or y_max == image.shape[0] and y + radius > image.shape[0] or \
       x_min == 0 and x - radius < 0 or x_max == image.shape[1] and x + radius > image.shape[1]:
        nan_data = {f"{marker}_{param}": [np.nan] for param in parameters}
        nan_data[cell_index_id] = [cell_id]
        return pd.DataFrame(nan_data).set_index(cell_index_id)

    sub_image = image[y_min:y_max, x_min:x_max]

    if quant_value is not None:
        sub_image = np.clip(sub_image, 0, quant_value)
        sub_image /= quant_value

    return _extract_cell_features(sub_image, cell_id, marker, quant_value, cell_index_id, parameters)


def _extract_cell_features(image: np.ndarray,
                          cell_id: str,
                          marker: str,
                          quant_value: Optional[float],
                          cell_index_id: str,
                          parameters: List[str]) -> pd.DataFrame:
    """
    Processes an image to extract cellular features based on a list of specified parameters.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    cell_id : str
        Identifier for the cell within the image.
    marker : str
        Marker associated with the cell.
    quant_value : Optional[float]
        Upper limit for clipping and normalizing the image data.
    cell_index_id : str
        Column name to be used as index in the resulting DataFrame.
    parameters : List[str]
        List of strings specifying which features to calculate.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains measurements for the specified features
        for the given cell, indexed by `cell_index_id`.
    """
    img = image.clip(0, quant_value) if quant_value else image
    if quant_value:
        img /= quant_value

    measurements = {}

    if 'Mean' in parameters:
        measurements['Mean'] = np.mean(img)
    if 'Median' in parameters:
        measurements['Median'] = np.median(img)
    if 'Std' in parameters:
        measurements['Std'] = np.std(img)
    if 'Kurtosis' in parameters:
        measurements['Kurtosis'] = kurtosis(img.flat)
    if 'Skew' in parameters:
        measurements['Skew'] = skew(img.flat)

    for param in parameters:
        if 'Quantile' in param:
            quantile = float(param.split('_')[-1])
            measurements[param] = np.quantile(img, quantile)

    texture_params = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    if any(param in parameters for param in texture_params):
        img = img_as_ubyte(img / img.max() if not quant_value else img)
        glcm = graycomatrix(img, distances=[5], angles=[0], symmetric=True, normed=True)
        for param in texture_params:
            if param in parameters:
                measurements[param] = graycoprops(glcm, param.lower())[0, 0]

    results_list = [[cell_id] + [measurements[param] for param in parameters if param in measurements]]
    column_names = [cell_index_id] + [f"{marker}_{param}" for param in parameters]
    results_df = pd.DataFrame(results_list, columns=column_names).set_index(cell_index_id)

    return results_df


def _load_single_img(filename: str) -> np.ndarray:
    """
    Load a single image from the specified file.

    Parameters
    ----------
    filename : str
        The image file name, must end with .tiff or .tif.

    Returns
    -------
    np.ndarray
        Loaded image data as a float32 array.

    Raises
    ------
    ValueError
        If the file does not end with .tiff or .tif, or if the image is not 2D.
    """
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        Img_in = tp.imread(filename).astype('float32')
    else:
        raise ValueError('Raw file should end with tiff or tif!')
    if Img_in.ndim != 2:
        raise ValueError('Single image should be 2d!')
    return Img_in


def _load_imgs_from_directory(load_directory: str,
                             channel_name: str,
                             quiet: bool = False) -> Optional[Tuple[List[np.ndarray], List[str], List[str]]]:
    """
    Load images from a directory matching the specified channel name.

    Parameters
    ----------
    load_directory : str
        The directory to load images from.
    channel_name : str
        The channel name to match in the image file names.
    quiet : bool, optional
        If True, suppresses print statements.

    Returns
    -------
    Optional[Tuple[List[np.ndarray], List[str], List[str]]]
        A tuple containing a list of loaded images, a list of file names, and a list of subdirectories.

    Raises
    ------
    ValueError
        If no images are found matching the channel name.
    """
    Img_collect = []
    Img_file_list = []
    img_folders = glob(join(load_directory, "*", ""))

    if not quiet:
        print('Image data loaded from ...\n')
    
    if not img_folders:
        img_folders = [load_directory]
        
    for sub_img_folder in img_folders:
        Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) and (f.endswith(".tiff") or f.endswith(".tif"))]
        
        for Img_file in Img_list:
            if channel_name.lower() in Img_file.lower():
                Img_read = _load_single_img(join(sub_img_folder, Img_file))
                
                if not quiet:
                    print(sub_img_folder + Img_file)
                
                Img_file_list.append(Img_file)
                Img_collect.append(Img_read)
                break

    if not quiet:
        print('\nImage data loaded completed!')
    
    if not Img_collect:
        print(f'No such channel as {channel_name}. Please check the channel name again!')
        return None

    return Img_collect, Img_file_list, img_folders


def create_env_anndata(master_list: pd.DataFrame,
                       source_anndata_for_obs: Optional[ad.AnnData] = None,
                       obs_to_transfer: List[str] = [],
                       cell_index: str = 'Master_Index',
                       norm_quantile: float = 0.99,
                       drop_unmeasured_cells: bool = True) -> ad.AnnData:
    """
    Create an AnnData object for environmental analysis.

    Parameters
    ----------
    master_list : pd.DataFrame
        DataFrame containing the measurements.
    source_anndata_for_obs : Optional[ad.AnnData], optional
        Source AnnData object to extract .obs data from.
    obs_to_transfer : List[str], optional
        List of observation column names to transfer from source AnnData.
    cell_index : str, optional
        Column name to be used as index in the resulting AnnData.
    norm_quantile : float, optional
        Quantile for normalization.
    drop_unmeasured_cells : bool, optional
        If True, drop cells that couldn't be measured (sum=0).

    Returns
    -------
    ad.AnnData
        The created AnnData object.
    """
    print('Creating X...')
    X_df = master_list
    
    if X_df.index.name != cell_index:
        X_df = X_df.set_index(cell_index)        

    if drop_unmeasured_cells:
        X_df = X_df.loc[X_df.sum(axis=1) != 0, :]  # Drop rows where sum=0, as those couldn't be measured

    X_df = X_df.dropna()
    X_df = X_df / X_df.quantile(norm_quantile)
    X_df = X_df.clip(upper=1)
    
    if source_anndata_for_obs:
        print('Extracting .obs from source anndata...')
        obs_df = source_anndata_for_obs.obs[obs_to_transfer].set_index(cell_index)
        obs_df.index = obs_df.index.astype(np.int64)
        
        overlap_cells = list(set(X_df.index.tolist()) & set(obs_df.index.tolist()))
        overlap_cells = np.array(overlap_cells, dtype=np.int64)
        
        anndata = ad.AnnData(X=X_df.loc[overlap_cells, :], obs=obs_df.loc[overlap_cells, :])
        
        for c in anndata.obs.columns:
            anndata.obs[c] = anndata.obs[c].cat.remove_unused_categories()
    else:
        anndata = ad.AnnData(X=X_df)        
        
    return anndata
