import anndata as ad
import numpy as np
import pandas as pd
import os
from copy import copy
from pathlib import Path
import skimage.io

import utils


def remove_tiff_extensions(series):
    """
    Remove .tiff or .tif extensions from the file names in a pandas Series.

    Args:
    series (pandas.Series): The Series containing the file names.

    Returns:
    pandas.Series: Series with the file extensions removed.
    """
    series = series.str.replace(r'\.tiff$', '', regex=True)
    series = series.str.replace(r'\.tif$', '', regex=True)
    return series

def analyze_masks(directory_path, file_extension='.tiff'):
    """
    Analyze masks files in a given directory, returning a DataFrame with information about each masks.

    Parameters:
    directory_path (str): Path to the directory containing the image files.
    file_extension (str): File extension of the image files (default is '.tif').

    Returns:
    pd.DataFrame: DataFrame with columns 'ROI', 'x size', 'y size', 'Cell count', and 'Full path'.
    """
    data = []

    for file in os.listdir(directory_path):
        if file.endswith(file_extension):
            file_path = os.path.join(directory_path, file)
            image = skimage.io.imread(file_path)
            y_size, x_size = image.shape[:2]
            cell_count = len(np.unique(image)) - 1  # Assuming background is labelled as 0

            data.append({
                'ROI': os.path.splitext(file)[0],
                'x_size': x_size,
                'y_size': y_size,
                'Cell_count': cell_count,
                'Mask_path': file_path
            })
            
    return pd.DataFrame(data)

def concat_csv_tables(folder, ROI_list):

    """
    Concatenates as .csv files in a directory, adding a column for ROI
    """

    # List of csv files
    csv_files = [x for x in os.listdir(folder) if '.csv' in x]

    # Create raw cell table from all data
    raw_cell_table = []

    for x, ROI in zip(csv_files, ROI_list):
        cell_table_roi = pd.read_csv(Path(folder, x), low_memory=False)
        cell_table_roi['ROI'] = str(ROI)   
        raw_cell_table.append(cell_table_roi.copy())

    raw_cell_table = pd.concat(raw_cell_table)
    
    return raw_cell_table

def split_images_to_channels(input_dir, output_dir, channel_names):
    
    '''
    Splits images outputted from Steinpose into individual channels
    '''
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".tiff"):
            file_path = os.path.join(input_dir, filename)
            try:
                # Load the image
                image = skimage.io.imread(file_path)

                # Check if number of channels matches the list length
                if image.shape[2] != len(channel_names):
                    image = np.transpose(image, (1, 2, 0))
                    assert image.shape[2] == len(channel_names), f"Number of channels does not match for {filename}"
                    
                # Process each channel
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


def cellpose_setup(cellpose_folder = '',
                    intensities_folder = 'intensities',
                    masks_folder = 'masks',
                    images_folder = 'img',
                    image_channel_folder='Images',
                    regionprops_folder = 'regionprops',
                    regionsprops_metrics = ['area'],
                    dictionary='dictionary.csv',
                    normalisation = ["q0.999"],
                    store_package_versions=True):

    '''
    Takes a Steinpose directory structure and returns an AnnData object
    '''
    
    # This initial block loads the images and panel files

    # Setup paths
    cellpose_folder = Path(cellpose_folder)
    masks_folder = Path(cellpose_folder, masks_folder)
    images_folder = Path(cellpose_folder, images_folder)
    image_channel_folder = Path(cellpose_folder, image_channel_folder)
    regionprops_folder = Path(cellpose_folder, regionprops_folder)

    # Load and setup image_df
    images_df = pd.read_csv(Path(cellpose_folder, 'images.csv'))
    images_df['ROI'] = remove_tiff_extensions(images_df['image'])
    images_df = images_df.set_index('ROI', drop=True)

    # Get list of ROIs from images_df
    ROI_list = images_df.index.tolist()

    # Load panel file
    panel_df = pd.read_csv(Path(cellpose_folder, 'panel.csv'))

    # This block checks for the integrity of the data - ie, that all the files exist and agree with one another
    
    print('Matching files to ensure data integrity....')
    for folder, extension in zip([intensities_folder, masks_folder, images_folder, regionprops_folder], ['.csv','.tiff', '.tiff', '.csv']):

        # Get a list of all files from subfolder
        files = [x for x in os.listdir(folder) if extension in x]

        # List of ROIs from names of cell tables
        rois_from_filenames = [x.replace(extension,'') for x in files]

        assert ROI_list == rois_from_filenames, f'Files in folder {folder} do not match with the images. Check your data structure'
        print(f'Files in {folder} match with images.csv file')

    print('All files matched successfully')

    mask_df = analyze_masks(masks_folder)
    mask_df = mask_df.set_index('ROI', drop=True)

    assert images_df['width_px'].tolist()==mask_df['x_size'].tolist(), 'Sizes of masks do not match the sizes in the images file - check data'
    assert images_df['height_px'].tolist()==mask_df['y_size'].tolist(), 'Sizes of masks do not match the sizes in the images file - check data'
    print('Mask sizes all match with images file')
    
    print('\nROIs detected and matched:')
    print(ROI_list)

    # Add in cell counts to images_df
    images_df['Cell_count'] = mask_df['Cell_count']

    # We now move on to actually importing the data and constructing the AnnData object

    # Concatenate all the cell tables for individual ROIs
    print(f'\nConcatenating cell tables for {str(len(ROI_list))} regions of interest...')
    raw_cell_table = concat_csv_tables(intensities_folder, ROI_list)
    print(f'Concatenating cell complete!')

    # List of channels from the panel_df
    channels = panel_df.loc[panel_df.keep==True, 'name']

    # Only data for markers where they are to be kept in panel
    markers_raw = raw_cell_table.loc[:, channels]
    markers_raw = markers_raw.astype('float32')
    
    # This can handle combinations of normalisation
    if normalisation:
        markers = normalise_markers(markers_raw, normalisation)
    else:
        markers = markers_raw
        print(f'\nMarkers not normalised, raw values used')

    # Create AnnData
    adata = ad.AnnData(markers, dtype=np.float32)

    # Add in metadata for where cells came from, and their cell ids in the masks, to the adata.obs
    for col in ['ROI','Object']:
        adata.obs[col] = raw_cell_table[col].values

    # Backup the initial index with all the cells. This is very useful if cells are removed later and the index gets reset.    
    adata.obs['Master_Index'] = list(adata.obs.index)

    adata.obs['ROI'] = adata.obs['ROI'].astype('category')

    # Get cell location and other spatial measures
     
    print(f'\nConcatenating regionprops tables for {str(len(ROI_list))} regions of interest...')
    regionprops_table = concat_csv_tables(regionprops_folder, ROI_list)
    print(f'Concatenating regionprops tables complete!')

    regionprops_table = regionprops_table.rename(columns={'centroid-0':'Y_loc', 'centroid-1':'X_loc'})

    print('Adding in following entries from regionprops table to adata.obs:')
    print(regionsprops_metrics)
    # Also add in any other measures from regionsprops
    for col in ['Y_loc', 'X_loc']+regionsprops_metrics:
        adata.obs[col] = regionprops_table[col].values

    # Whole numbers will make it easier to find the cells in images
    adata.obs['X_loc'] = np.int32(adata.obs['X_loc'])
    adata.obs['Y_loc'] = np.int32(adata.obs['Y_loc'])

    # This is the default location for spatial analyses that Squidpy will look for
    adata.obsm['spatial'] = adata.obs[['X_loc', 'Y_loc']].to_numpy()  

    # This will create image folders for each of the ROIs
    print(f'\nSplitting images into channels...')
    split_images_to_channels(images_folder, image_channel_folder, channels)
    print(f'Splitting complete into:')
    print(image_channel_folder)
    
    # Add in data from panel_df to the adata.var
    adata.var = pd.merge(adata.var, panel_df.set_index('name',drop=True), left_index=True, right_index=True)
    
    if dictionary:
        try:
            dictionary_path = Path(cellpose_folder, dictionary)
            dictionary = pd.read_csv(dictionary_path, low_memory=False, index_col=0)
            
            print('\nDictionary file found')
            for col in dictionary:
                adata.obs[col] = adata.obs['ROI'].map(dictionary[col].to_dict()).values.tolist()
                print(f'Column {col} added to adata.obs')
            
        except FileNotFoundError:
            # Handle the case where the file does not exist
            print(f"\nError: The dictionary file '{dictionary_path}' does not exist. Creating a blank dictionary file you can use as a template.")
            blank_dict=pd.DataFrame(ROI_list, columns=['ROI'])
            blank_dict['Example_1']=True
            blank_dict['Example_2']='Genotype1'
            blank_dict['Example_3']=100
            blank_dict = blank_dict.set_index('ROI',drop=True)
            blank_dict.to_csv(Path(cellpose_folder, 'dictionary.csv'))
        except Exception as e:
            # Handle all other exceptions
            print(f"\nAn error occurred loading the dictionary file: {e}")
        
    print('\nExtra image level information stored in adata.uns.sample.\nThis includes size of images and other useful information.')
    adata.uns['sample'] = images_df.copy()
    
    if store_package_versions:
        adata.uns['packages'] = utils.pip_freeze_to_dataframe()
    
    print('\nSuccessfully created AnnData object!')
    print(adata)
    
    utils.adlog(adata, 'AnnData object created', ad)

    
    return adata 
    
    

def normalise_markers(markers, method_list):
    '''
    Normalises a markers dataframe with a list of methods, in sequence.
    Default is normalisation to 0.999 quantile, clipping all values above (outliers) to 1.
    Also supports log2, log10 and arcsinh
    '''
        
    for method in method_list:
        
        if method[0]=='q':
            quantile = round(float(method[1:]),5)
            markers = markers.div(markers.quantile(q=quantile)).clip(upper=1)        
            markers.fillna(0, inplace=True)    
            print(f'\nData normalised to {str(quantile)} quantile')
        
        elif 'arcsinh' in method:
            cofactor=int(method[7:])
            markers = np.arcsinh(markers/cofactor)
            print(f'\nData Arcsinh adjusted with cofactor {str(cofactor)}')            
        
        elif method=='log2':
            markers = np.log2(markers)
            print(f'\nData Log2 adjusted')            
        
        elif method=='log10':
            markers = np.log10(markers)          
            print(f'\nData Log10 adjusted')
            
        else:
            print(f'\nNormalised method {method} no recognised')
            
    return markers