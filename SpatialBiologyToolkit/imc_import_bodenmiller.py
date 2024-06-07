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

# Third-Party Imports
import anndata as ad
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import tifffile as tp

def import_bodenmiller(
    directory: str = 'cpout',
    panel_file: str = 'panel.csv',
    cell_file: str = 'cell.csv',
    image_file: str = 'image.csv',
    masks_dir: str = 'masks',
    images_dir: str = 'images',
    image_file_filename_fullstack: str = 'FileName_FullStack',
    image_file_filename_cellmask: str = 'FileName_cellmask',
    image_file_roi_name: str = 'Metadata_description',
    image_file_mcdfile_name: str = 'Metadata_acname',
    image_file_width: str = 'Width_FullStack',
    image_file_height: str = 'Height_FullStack',
    acquisition_metadata: str = 'acquisition_metadata.csv'
) -> tuple:
    """
    Import data from the Bodenmiller pipeline and return sample and panel dataframes.

    Parameters
    ----------
    directory : str
        Directory containing the Bodenmiller pipeline outputs.
    panel_file : str
        Filename of the panel file.
    cell_file : str
        Filename of the cell file.
    image_file : str
        Filename of the image file.
    masks_dir : str
        Directory containing mask files.
    images_dir : str
        Directory containing image files.
    image_file_filename_fullstack : str
        Column name for the full stack filename in the image file.
    image_file_filename_cellmask : str
        Column name for the cell mask filename in the image file.
    image_file_roi_name : str
        Column name for the ROI name in the image file.
    image_file_mcdfile_name : str
        Column name for the MCD file name in the image file.
    image_file_width : str
        Column name for the image width in the image file.
    image_file_height : str
        Column name for the image height in the image file.
    acquisition_metadata : str
        Filename of the acquisition metadata file.

    Returns
    -------
    tuple
        A tuple containing the sample dataframe and panel dataframe.
    """

    files = [panel_file, cell_file, image_file, masks_dir, images_dir]

    for file in files:
        file_path = Path(directory, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error importing outputs from Bodenmiller pipeline folder: {file} does not exist in {directory}")

    panel_df = pd.read_csv(Path(directory, panel_file), low_memory=False, index_col=0)
    image_df = pd.read_csv(Path(directory, image_file), low_memory=False)

    try:
        metadata_df = pd.read_csv(os.path.join(directory, acquisition_metadata), low_memory=False)
        print(f'Metadata file {acquisition_metadata} found and imported')
    except FileNotFoundError:
        metadata_df = None
    
    masks_dir = Path(directory, masks_dir)
    images_dir = Path(directory, images_dir)

    panel_df = panel_df.loc[panel_df['full'] == True, :].drop(columns=['full', 'ilastik'])
    panel_df['cell_table_channel'] = [f'Intensity_MeanIntensity_FullStack_c{i+1}' for i in range(len(panel_df.index))]
    panel_df.index = np.arange(1, len(panel_df) + 1)
    panel_df.rename(columns={'Label': 'Target'}, inplace=True)
    print(f'Panel dataframe imported, found {panel_df.shape[0]} channels')
    panel_df.to_csv('panel_df.csv')

    if metadata_df is None:
        sample_df_columns = [
            image_file_filename_fullstack,
            image_file_filename_cellmask,
            image_file_roi_name,
            image_file_mcdfile_name,
            image_file_width,
            image_file_height,
            'ImageNumber'
        ]

        for x in sample_df_columns:
            assert x in image_df.columns, f'Could not find column {x} in the image.csv for constructing the sample dataframe'

        sample_df = image_df.loc[:, sample_df_columns]
        sample_df = sample_df.rename(columns={
            image_file_mcdfile_name: 'MCD_File',
            image_file_roi_name: 'ROI',
            image_file_width: 'Size_x',
            image_file_height: 'Size_y'
        }).set_index('ROI')
    else:
        print('Using acquisition_metadata to setup sample dataframe...')
        sample_df = metadata_df.loc[:, ['AcSession', 'description', 'max_x', 'max_y']]
        sample_df = sample_df.rename(columns={'description': 'ROI', 'max_x': 'Size_x', 'max_y': 'Size_y'}).set_index('ROI')
        sample_df['FileName_FullStack'] = sample_df['AcSession'] + '_full.tiff'
        sample_df['FileName_cellmask'] = sample_df['AcSession'] + '_ilastik_s2_Probabilities_mask.tiff'
        sample_df['ImageNumber'] = sample_df['FileName_FullStack'].map(dict(zip(image_df['FileName_FullStack'], image_df['ImageNumber'])))

    sample_df['mm2'] = (sample_df['Size_x'] / 1000) * (sample_df['Size_y'] / 1000)
    sample_df['AcSession'] = [x[:-10] for x in sample_df['FileName_FullStack']]
    sample_df.index = sample_df.index.astype('str')
    print(f'Sample dataframe imported, found {sample_df.shape[0]} regions of interest')
    sample_df.to_csv('sample_df.csv')

    return sample_df, panel_df


def reload_dfs(
    sample_df: str = 'sample_df.csv',
    panel_df: str = 'panel_df.csv'
) -> tuple:
    """
    Reload sample and panel dataframes from disk if they are supplied as paths.

    Parameters
    ----------
    sample_df : str
        Path to the sample dataframe file.
    panel_df : str
        Path to the panel dataframe file.

    Returns
    -------
    tuple
        A tuple containing the reloaded sample dataframe and panel dataframe.
    """

    if isinstance(sample_df, str):
        sample_df = pd.read_csv(sample_df, low_memory=False, index_col=0)
        sample_df.index = sample_df.index.astype('str')
            
    if isinstance(panel_df, str):
        panel_df = pd.read_csv(panel_df, low_memory=False, index_col=0)
        
    return sample_df, panel_df


def setup_anndata(
    cell_df: str = 'cell.csv',
    directory: str = 'cpout',
    sample_df: str = 'sample_df.csv',
    panel_df: str = 'panel_df.csv',
    cell_df_x: str = 'Location_Center_X',
    cell_df_y: str = 'Location_Center_Y',
    cell_df_ROIcol: str = 'ROI',
    dictionary: str = 'dictionary.csv',
    non_cat_obs: list = [],
    cell_df_extra_columns: list = [],
    marker_normalisation: str = 'q99.9',
    panel_df_target_col: str = 'Target',
    cell_table_format: str = 'bodenmmiller',
    return_normalised_markers: bool = False
) -> ad.AnnData:
    """
    Create an AnnData object using the cell table .csv file, and panel_df and sample_df files.

    Parameters
    ----------
    cell_df : str
        Path to the cell dataframe file.
    directory : str
        Directory containing the files.
    sample_df : str
        Path to the sample dataframe file.
    panel_df : str
        Path to the panel dataframe file.
    cell_df_x : str
        Column name for the x-coordinates in the cell dataframe.
    cell_df_y : str
        Column name for the y-coordinates in the cell dataframe.
    cell_df_ROIcol : str
        Column name for the ROI in the cell dataframe.
    dictionary : str
        Path to the dictionary file.
    non_cat_obs : list
        List of observation columns not to be converted to categorical.
    cell_df_extra_columns : list
        List of extra columns from the cell dataframe to be included.
    marker_normalisation : str
        Normalisation method for the markers.
    panel_df_target_col : str
        Column name for the target in the panel dataframe.
    cell_table_format : str
        Format of the cell table ('bodenmmiller' or 'cleaned').
    return_normalised_markers : bool
        Whether to return the normalised markers along with the AnnData object.

    Returns
    -------
    ad.AnnData
        AnnData object containing the processed data.
    """
    
    pd.set_option('mode.chained_assignment', None)
    
    sample_df, panel_df = reload_dfs(sample_df, panel_df)
        
    cell_df = pd.read_csv(Path(directory, cell_df), low_memory=False)
    print(f'Loaded cell file, {cell_df.shape[0]} cells found')
    
    if cell_table_format == 'bodenmmiller':
        try:
            cell_df_intensities = cell_df[[col for col in cell_df.columns if 'Intensity_MeanIntensity_FullStack' in col]].copy()
        except KeyError:
            print('Could not find Intensity_MeanIntensity_FullStack columns in the cell table! Cell table may have been exported incorrectly from CellProfiler')
            raise
        mapping = dict(zip(panel_df['cell_table_channel'], panel_df[panel_df_target_col]))
        cell_df_intensities.rename(columns=mapping, inplace=True)    
    elif cell_table_format == 'cleaned':
        marker_cols = [col for col in cell_df.columns if col in panel_df[panel_df_target_col].values.tolist()]
        utils.compare_lists(marker_cols, panel_df[panel_df_target_col].tolist(), 'Markers from cell DF columns', 'Markers in panel file', return_error=True)
        cell_df_intensities = cell_df[marker_cols].copy()
               
    if not isinstance(marker_normalisation, list):
        marker_normalisation = [marker_normalisation]
    
    markers_normalised = cell_df_intensities
    assert markers_normalised.shape[1] == panel_df.shape[0], 'Length of panel and markers do not match!'
        
    utils.compare_lists(panel_df['Target'].tolist(), markers_normalised.columns.tolist(), 'PanelFile', 'MarkerDF')   
    markers_normalised = markers_normalised.reindex(columns=panel_df['Target'])
            
    for method in marker_normalisation:
        if method[0] == 'q':
            quantile = round(float(method[1:]) / 100, 5)
            markers_normalised = markers_normalised.div(markers_normalised.quantile(q=quantile)).clip(upper=1)        
            markers_normalised.fillna(0, inplace=True)    
            print(f'\nData normalised to {quantile} quantile')
        elif 'arcsinh' in method:
            cofactor = int(method[7:])
            markers_normalised = np.arcsinh(markers_normalised / cofactor)
            print(f'\nData Arcsinh adjusted with cofactor {cofactor}')            
        elif method == 'log2':
            markers_normalised = np.log2(markers_normalised)
            print(f'\nData Log2 adjusted')            
        elif method == 'log10':
            markers_normalised = np.log10(markers_normalised)          
            print(f'\nData Log10 adjusted')
        else:
            print(f'Normalisation method {method} not recognized')
    
    adata = sc.AnnData(markers_normalised)
    adata.var_names = markers_normalised.columns.tolist()
    
    if cell_table_format == 'bodenmmiller':
        adata.obs['ROI'] = cell_df['ImageNumber'].map(dict(zip(sample_df['ImageNumber'], sample_df.index))).values.tolist()
    elif cell_table_format == 'cleaned':
        adata.obs['ROI'] = cell_df[cell_df_ROIcol].values.tolist()
                                                        
    for c in cell_df_extra_columns:
        adata.obs[c] = cell_df[c].values.tolist()
                                                
    adata.obsm['spatial'] = cell_df[[cell_df_x, cell_df_y]].to_numpy()                                                
    adata.obs['X_loc'] = cell_df[cell_df_x].values.tolist()
    adata.obs['Y_loc'] = cell_df[cell_df_y].values.tolist()
    adata.obs['Master_Index'] = adata.obs.index.values.tolist()
                                                
    if dictionary is not None:
        try:
            try:
                obs_dict = pd.read_csv(dictionary, low_memory=False, index_col=0).to_dict()
            except FileNotFoundError:
                obs_dict = pd.read_csv(Path(directory, dictionary), low_memory=False, index_col=0).to_dict()
        except FileNotFoundError:
            print('Could not find dictionary file')
            raise
            
        for i in obs_dict.keys():
            adata.obs[i] = adata.obs['ROI'].map(obs_dict[i]).values.tolist()
            if i not in non_cat_obs:
                adata.obs[i] = adata.obs[i].astype('category')
                print(f'Obs {i} added as categorical variable with following categories:')
                print(adata.obs[i].cat.categories.tolist())
            else:
                print(f'Obs {i} NOT converted to categorical')
        
        adata.uns.update({'categorical_obs': [x for x in obs_dict if x not in non_cat_obs]})
        
        obs_nans = adata.obs.isna().sum(axis=0) / len(adata.obs) * 100
        
        if obs_nans.mean() != 0:
            print('WARNING! Some obs columns have NaNs present, which could indicate your dictionary has not been setup correctly')
            utils.print_full(pd.DataFrame(obs_nans, columns=['Percentage of NaNs']))
        
    else:
        print('No dictionary provided')
    
    print('Markers imported:')
    print(adata.var_names)
    print(adata)
    
    utils.adlog(adata, 'AnnData object created', sc)
    adata.uns.update({'sample': sample_df.copy(), 'panel': panel_df.copy()})
    
    if return_normalised_markers:
        return adata, markers_normalised
    else:
        return adata


def stacks_to_imagefolders(
    bodenmiller_folder: str = 'cpout',
    sample_df: str = 'sample_df.csv',
    panel_df: str = 'panel_df.csv',
    unstacked_output_folder: str = 'images',
    masks_output_folder: str = 'masks',
    sample_df_filename_col: str = 'FileName_FullStack',
    sample_df_mask_col: str = 'FileName_cellmask',
    panel_df_target_col: str = 'Target'
) -> None:
    """
    Create mask and image folders using the Bodenmiller outputs.

    Parameters
    ----------
    bodenmiller_folder : str
        Directory containing the Bodenmiller outputs.
    sample_df : str
        Path to the sample dataframe file.
    panel_df : str
        Path to the panel dataframe file.
    unstacked_output_folder : str
        Directory to save the unstacked images.
    masks_output_folder : str
        Directory to save the renamed masks.
    sample_df_filename_col : str
        Column name for the filename in the sample dataframe.
    sample_df_mask_col : str
        Column name for the mask filename in the sample dataframe.
    panel_df_target_col : str
        Column name for the target in the panel dataframe.

    Returns
    -------
    None
    """

    output = Path(unstacked_output_folder)
    output.mkdir(exist_ok=True)
    
    masks_folder = Path(bodenmiller_folder, 'masks')
    input_folder = Path(bodenmiller_folder, 'images') 
    
    sample_df, panel_df = reload_dfs(sample_df, panel_df)
    
    tiff_paths = list(input_folder.rglob('*.tiff'))
    print(f'Unpacking {len(tiff_paths)} ROIs...')
    
    metadata_rois = sample_df[sample_df_filename_col].tolist()
    detectedimages_rois = [os.path.basename(x) for x in tiff_paths]
    
    meta_not_actual = [x for x in metadata_rois if x not in detectedimages_rois]
    actual_not_meta = [x for x in detectedimages_rois if x not in metadata_rois]    
    
    if meta_not_actual or actual_not_meta:
        print('ROIs referred in metadata without image stacks being detected:')
        print(meta_not_actual)
        print('ROIs with images that are not referred to in metadata:')
        print(actual_not_meta)
    
    mask_paths = list(masks_folder.rglob('*.tiff'))
    
    metadata_masks_rois = sample_df[sample_df_mask_col].tolist()
    detectedmaskss_rois = [os.path.basename(x) for x in mask_paths]
    
    meta_not_actual = [x for x in metadata_masks_rois if x not in detectedmaskss_rois]
    actual_not_meta = [x for x in detectedmaskss_rois if x not in metadata_masks_rois]      
                           
    if meta_not_actual or actual_not_meta:
        print('ROIs referred in metadata without masks being detected:')
        print(meta_not_actual)
        print('Masks that are not referred to in metadata:')
        print(actual_not_meta)                           
                           
    for path in tiff_paths:
        image = tp.imread(path)
        image_filename = os.path.basename(path)
        folder_name = sample_df.loc[sample_df[sample_df_filename_col] == image_filename, sample_df_filename_col].index[0]
        output_dir = Path(unstacked_output_folder, folder_name)
        output_dir.mkdir(exist_ok=True)

        for i, channel_name in enumerate(panel_df[panel_df_target_col]):
            image_to_save = image[i]
            image_to_save[image_to_save < 0] = 0
            image_to_save = np.uint16(image_to_save)
            tp.imwrite(join(output_dir, f'{channel_name}.tiff'), image_to_save)
                   
    masks_output_folder = Path(masks_output_folder)
    masks_output_folder.mkdir(exist_ok=True)  

    for path in mask_paths:
        mask_filename = os.path.basename(path)
        roi_name = sample_df.loc[sample_df[sample_df_mask_col] == mask_filename, sample_df_mask_col].index[0]
        shutil.copy(path, os.path.join(masks_output_folder, f'{roi_name}.tiff'))


def remove_ROIs_and_markers(
    adata: ad.AnnData,
    ROI_obs: str = 'ROI',
    ROIs_to_remove: list = [],
    Markers_to_remove: list = ['DNA1', 'DNA3']
) -> ad.AnnData:
    """
    Remove unused or failed ROIs and/or markers from the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    ROI_obs : str
        Observation field containing the ROI names.
    ROIs_to_remove : list
        List of ROIs to remove.
    Markers_to_remove : list
        List of markers to remove.

    Returns
    -------
    AnnData
        AnnData object with the specified ROIs and markers removed.
    """

    if not isinstance(ROIs_to_remove, list):
        ROIs_to_remove = [ROIs_to_remove]
        
    if not isinstance(Markers_to_remove, list):
        Markers_to_remove = [Markers_to_remove]
        
    print('Removing markers:')
    print(Markers_to_remove)
    
    print('Removing ROIs:')
    print(ROIs_to_remove)

    all_markers = adata.var_names.tolist()
    markers_limited = [m for m in all_markers if m not in Markers_to_remove]
    adata.uns['sample'] = adata.uns['sample'].loc[~adata.uns['sample'].index.isin(ROIs_to_remove), :]
    adata.uns['sample'].to_csv('sample_df.csv')

    return adata[~adata.obs[ROI_obs].isin(ROIs_to_remove), markers_limited]
