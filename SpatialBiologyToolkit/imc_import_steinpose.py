# Standard Library Imports
import os
from copy import copy
from pathlib import Path

# Third-Party Imports
import anndata as ad
import numpy as np
import pandas as pd
import skimage.io

# Local Application Imports
from .utils import remove_tiff_extensions, analyze_masks, adlog, pip_freeze_to_dataframe


def _concat_csv_tables(folder: str, ROI_list: list) -> pd.DataFrame:
    """
    Concatenate .csv files in a directory, adding a column for ROI.

    Parameters
    ----------
    folder : str
        Path to the folder containing the .csv files.
    ROI_list : list
        List of ROI names corresponding to the .csv files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with a column for ROI.
    """
    csv_files = [x for x in os.listdir(folder) if '.csv' in x]
    raw_cell_table = []

    for x, ROI in zip(csv_files, ROI_list):
        cell_table_roi = pd.read_csv(Path(folder, x), low_memory=False)
        cell_table_roi['ROI'] = str(ROI)   
        raw_cell_table.append(cell_table_roi.copy())

    raw_cell_table = pd.concat(raw_cell_table)
    
    return raw_cell_table


def split_images_to_channels(input_dir: str, output_dir: str, channel_names: list) -> None:
    """
    Split images outputted from Steinpose into individual channels.

    Parameters
    ----------
    input_dir : str
        Directory containing the input images.
    output_dir : str
        Directory to save the split channel images.
    channel_names : list
        List of channel names.

    Returns
    -------
    None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".tiff"):
            file_path = os.path.join(input_dir, filename)
            try:
                image = skimage.io.imread(file_path)
                if image.shape[2] != len(channel_names):
                    image = np.transpose(image, (1, 2, 0))
                    assert image.shape[2] == len(channel_names), f"Number of channels does not match for {filename}"

                base_name = os.path.splitext(filename)[0]
                channel_dir = os.path.join(output_dir, base_name)
                if not os.path.exists(channel_dir):
                    os.makedirs(channel_dir)

                for i, channel_name in enumerate(channel_names):
                    channel_image = image[:, :, i]
                    channel_file = os.path.join(channel_dir, f"{channel_name}.tiff")
                    skimage.io.imsave(channel_file, channel_image.astype(np.uint16), check_contrast=False)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

def cellpose_setup(
    cellpose_folder: str = '',
    intensities_folder: str = 'intensities',
    masks_folder: str = 'masks',
    images_folder: str = 'img',
    image_channel_folder: str = 'Images',
    regionprops_folder: str = 'regionprops',
    regionsprops_metrics: list = ['area'],
    dictionary: str = 'dictionary.csv',
    normalisation: list = ["q0.999"],
    store_package_versions: bool = True
) -> ad.AnnData:
    """
    Takes a Steinpose directory structure and returns an AnnData object.

    Parameters
    ----------
    cellpose_folder : str
        Directory containing the Steinpose outputs.
    intensities_folder : str
        Directory containing the intensity .csv files.
    masks_folder : str
        Directory containing the mask files.
    images_folder : str
        Directory containing the image files.
    image_channel_folder : str
        Directory to save the split channel images.
    regionprops_folder : str
        Directory containing the regionprops .csv files.
    regionsprops_metrics : list
        List of regionprops metrics to include.
    dictionary : str
        Path to the dictionary file.
    normalisation : list
        List of normalisation methods.
    store_package_versions : bool
        Whether to store the package versions.

    Returns
    -------
    ad.AnnData
        AnnData object containing the processed data.
    """
    cellpose_folder = Path(cellpose_folder)
    masks_folder = Path(cellpose_folder, masks_folder)
    images_folder = Path(cellpose_folder, images_folder)
    image_channel_folder = Path(cellpose_folder, image_channel_folder)
    regionprops_folder = Path(cellpose_folder, regionprops_folder)

    images_df = pd.read_csv(Path(cellpose_folder, 'images.csv'))
    images_df['ROI'] = remove_tiff_extensions(images_df['image'])
    images_df = images_df.set_index('ROI', drop=True)
    ROI_list = images_df.index.tolist()

    panel_df = pd.read_csv(Path(cellpose_folder, 'panel.csv'))

    print('Matching files to ensure data integrity....')
    for folder, extension in zip(
        [intensities_folder, masks_folder, images_folder, regionprops_folder],
        ['.csv', '.tiff', '.tiff', '.csv']
    ):
        files = [x for x in os.listdir(folder) if extension in x]
        rois_from_filenames = [x.replace(extension, '') for x in files]
        assert ROI_list == rois_from_filenames, f'Files in folder {folder} do not match with the images. Check your data structure'
        print(f'Files in {folder} match with images.csv file')

    print('All files matched successfully')

    mask_df = analyze_masks(masks_folder)
    mask_df = mask_df.set_index('ROI', drop=True)
    assert images_df['width_px'].tolist() == mask_df['x_size'].tolist(), 'Sizes of masks do not match the sizes in the images file - check data'
    assert images_df['height_px'].tolist() == mask_df['y_size'].tolist(), 'Sizes of masks do not match the sizes in the images file - check data'
    print('Mask sizes all match with images file')
    
    print('\nROIs detected and matched:')
    print(ROI_list)

    images_df['Cell_count'] = mask_df['Cell_count']

    print(f'\nConcatenating cell tables for {len(ROI_list)} regions of interest...')
    raw_cell_table = _concat_csv_tables(intensities_folder, ROI_list)
    print('Concatenating cell complete!')

    channels = panel_df.loc[panel_df.keep == True, 'name']
    markers_raw = raw_cell_table.loc[:, channels].astype('float32')
    
    if normalisation:
        markers = normalise_markers(markers_raw, normalisation)
    else:
        markers = markers_raw
        print('Markers not normalised, raw values used')

    adata = ad.AnnData(markers, dtype=np.float32)

    for col in ['ROI', 'Object']:
        adata.obs[col] = raw_cell_table[col].values

    adata.obs['Master_Index'] = list(adata.obs.index)
    adata.obs['ROI'] = adata.obs['ROI'].astype('category')

    print(f'\nConcatenating regionprops tables for {len(ROI_list)} regions of interest...')
    regionprops_table = _concat_csv_tables(regionprops_folder, ROI_list)
    print('Concatenating regionprops tables complete!')

    regionprops_table = regionprops_table.rename(columns={'centroid-0': 'Y_loc', 'centroid-1': 'X_loc'})

    print('Adding in following entries from regionprops table to adata.obs:')
    print(regionsprops_metrics)
    for col in ['Y_loc', 'X_loc'] + regionsprops_metrics:
        adata.obs[col] = regionprops_table[col].values

    adata.obs['X_loc'] = np.int32(adata.obs['X_loc'])
    adata.obs['Y_loc'] = np.int32(adata.obs['Y_loc'])
    adata.obsm['spatial'] = adata.obs[['X_loc', 'Y_loc']].to_numpy()

    print('\nSplitting images into channels...')
    split_images_to_channels(images_folder, image_channel_folder, channels)
    print('Splitting complete into:')
    print(image_channel_folder)
    
    adata.var = pd.merge(adata.var, panel_df.set_index('name', drop=True), left_index=True, right_index=True)
    
    if dictionary:
        try:
            dictionary_path = Path(cellpose_folder, dictionary)
            dictionary = pd.read_csv(dictionary_path, low_memory=False, index_col=0)
            print('\nDictionary file found')
            for col in dictionary:
                adata.obs[col] = adata.obs['ROI'].map(dictionary[col].to_dict()).values.tolist()
                print(f'Column {col} added to adata.obs')
        except FileNotFoundError:
            print(f"\nError: The dictionary file '{dictionary_path}' does not exist. Creating a blank dictionary file you can use as a template.")
            blank_dict = pd.DataFrame(ROI_list, columns=['ROI'])
            blank_dict['Example_1'] = True
            blank_dict['Example_2'] = 'Genotype1'
            blank_dict['Example_3'] = 100
            blank_dict = blank_dict.set_index('ROI', drop=True)
            blank_dict.to_csv(Path(cellpose_folder, 'dictionary.csv'))
        except Exception as e:
            print(f"\nAn error occurred loading the dictionary file: {e}")
        
    print('\nExtra image level information stored in adata.uns.sample.\nThis includes size of images and other useful information.')
    adata.uns['sample'] = images_df.copy()
    
    if store_package_versions:
        adata.uns['packages'] = pip_freeze_to_dataframe()
    
    print('\nSuccessfully created AnnData object!')
    print(adata)
    
    adlog(adata, 'AnnData object created', ad)
    
    return adata

def normalise_markers(markers: pd.DataFrame, method_list: list) -> pd.DataFrame:
    """
    Normalises a markers DataFrame with a list of methods, in sequence.

    Parameters
    ----------
    markers : pd.DataFrame
        DataFrame containing marker intensities (cells in rows, markers in columns).
    method_list : list
        List of normalisation methods.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame.
    """
    for method in method_list:
        if method[0] == 'q':
            quantile = round(float(method[1:]), 5)
            markers = markers.div(markers.quantile(q=quantile)).clip(upper=1)
            markers.fillna(0, inplace=True)
            print(f'Data normalised to {quantile} quantile')
        elif 'arcsinh' in method:
            cofactor = int(method[7:])
            markers = np.arcsinh(markers / cofactor)
            print(f'Data Arcsinh adjusted with cofactor {cofactor}')
        elif method == 'log2':
            markers = np.log2(markers)
            print('Data Log2 adjusted')
        elif method == 'log10':
            markers = np.log10(markers)
            print('Data Log10 adjusted')
        else:
            print(f'Normalisation method {method} not recognized')
            
    return markers
