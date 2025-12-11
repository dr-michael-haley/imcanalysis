import pandas as pd
import numpy as np
from pathlib import Path
import os    
import scanpy as sc
from tqdm import tqdm
import tifffile as tp
from copy import copy
from os.path import join
import shutil

import utils


def import_bodenmiller(directory = 'cpout',
                        panel_file='panel.csv',
                        cell_file='cell.csv',
                        image_file='image.csv',
                        masks_dir='masks',
                        images_dir='images',
                        image_file_filename_fullstack= 'FileName_FullStack',
                        image_file_filename_cellmask= 'FileName_cellmask',
                        image_file_roi_name= 'Metadata_description',
                        image_file_mcdfile_name= 'Metadata_acname',
                        image_file_width='Width_FullStack', 
                        image_file_height='Height_FullStack',
                        acquisition_metadata='acquisition_metadata.csv'):
    '''
    This function returns two dataframe - sample_df and panel_df, which are created from the outputs of the Bodenmiller pipeline
    '''

    # Check the files exist
    files = [panel_file, cell_file, image_file, panel_file, masks_dir, images_dir]

    for file in files:
        file_path = Path(directory, file)
        if not os.path.exists(file_path):
            raise TypeError(f"Error importing outputs from Bodenmiller pipeline folder: {file} does not exist in {directory}")

    # Import the files
    panel_df = pd.read_csv(Path(directory,panel_file), low_memory=False, index_col=0)
    image_df = pd.read_csv(Path(directory,image_file), low_memory=False)

    # acquisition_metadata is output by some older versions of Bodeniller pipeline        
    try:
        metadata_df = pd.read_csv(os.path.join(directory,acquisition_metadata), low_memory=False)
        print(f'Metadata file {acquisition_metadata} found and imported')
    except:
        pass
        acquisition_metadata=None
    
    masks_dir = Path(directory, masks_dir)
    images_dir = Path(directory, images_dir)

    
    ### SETTING UP PANEL DATAFRAME
    
    # Filter down the panel to only those in FullStack, and then drop columns we don't need
    panel_df = panel_df.loc[panel_df['full']==True,:].drop(columns=['full','ilastik'])

    # Create a column that maps the channels with the column names all they will appear in the CellProfiler cell table
    panel_df['cell_table_channel'] = [('Intensity_MeanIntensity_FullStack_c'+str(i+1)) for i, _ in enumerate(panel_df.index)]

    # Make panel_df index start at 1 instead of 0
    panel_df.index = np.arange(1, len(panel_df) + 1)

    # Remap the column names to the proper channel names
    try:
        panel_df.rename(columns={'Label':'Target'}, inplace=True)
    except:
        pass
    
    print(f'Panel dataframe imported, found {str(panel_df.shape[0])} channels')
    panel_df.to_csv('panel_df.csv')    

    ### SETTING UP SAMPLE DATAFRAME
    
    # This is the method for the newest version of the Bodenmiller pipeline that does NOT have an acquisition_metadata file
    if acquisition_metadata==None:
                             
        sample_df_columns=[image_file_filename_fullstack,
                          image_file_filename_cellmask,
                          image_file_roi_name,
                          image_file_mcdfile_name,
                          image_file_width, 
                          image_file_height,
                          'ImageNumber'] 

        # Check that all columns are in the image file
        for x in sample_df_columns:
            assert x in image_df.columns, f'Could not find column {x} in the image.csv for constructing the sample dataframe'

        sample_df=image_df.loc[:, sample_df_columns]
        sample_df=sample_df.rename(columns={image_file_mcdfile_name:'MCD_File', 
                                            image_file_roi_name:'ROI', 
                                            image_file_width:'Size_x', 
                                            image_file_height:'Size_y'}).set_index('ROI')

                          
    # Alternatively, will use acquisition_metadata.csv from older versions of the pipeline                                             
    else:
        print('Using acquisition_metadata to setup sample dataframe...')
        
        # Setup a dataframe with ROI-level information
        sample_df=metadata_df.loc[:,['AcSession', 'description', 'max_x', 'max_y']]
        sample_df=sample_df.rename(columns={'description':'ROI', 'max_x':'Size_x', 'max_y':'Size_y'}).set_index('ROI')
        sample_df['FileName_FullStack']=sample_df['AcSession']+'_full.tiff'
        sample_df['FileName_cellmask']=sample_df['AcSession']+'_ilastik_s2_Probabilities_mask.tiff'
        sample_df['ImageNumber']=sample_df['FileName_FullStack'].map(dict(zip(image_df['FileName_FullStack'], image_df['ImageNumber'])))

    # Calculate sample size in mm2
    sample_df['mm2']=(sample_df['Size_x']/1000)*(sample_df['Size_y']/1000)
    
    # Add in AcSession
    sample_df['AcSession']=[x[:-10] for x in sample_df['FileName_FullStack']]
        
    # Make sure the ROI name is a string
    sample_df.index = sample_df.index.astype('str')    
    
    print(f'Sample dataframe imported, found {str(sample_df.shape[0])} region of interest')
    sample_df.to_csv('sample_df.csv')
    
    return sample_df, panel_df


def reload_dfs(sample_df='sample_df.csv',
                panel_df='panel_df.csv'):
    '''
    This reloads the sample and panel data frames from disk if they are supplied as paths
    '''
    
    if type(sample_df)==str:
        sample_df = pd.read_csv(sample_df, low_memory=False, index_col=0)
        sample_df.index = sample_df.index.astype('str')
            
    if type(panel_df)==str:
        panel_df = pd.read_csv(panel_df, low_memory=False, index_col=0)
        
    return sample_df, panel_df


def setup_anndata(cell_df='cell.csv',#=cell_df,
                  directory='cpout',
                  sample_df='sample_df.csv',
                  panel_df='panel_df.csv',#=panel_df, 
                  cell_df_x='Location_Center_X',
                  cell_df_y='Location_Center_Y',
                  cell_df_ROIcol='ROI',
                  dictionary='dictionary.csv',
                  non_cat_obs=[],
                  cell_df_extra_columns=[],
                  marker_normalisation='q99.9',
                  panel_df_target_col='Target',
                  cell_table_format='bodenmmiller',
                  return_normalised_markers=False):
    
    '''
    This function returns an AnnData object using the cell table .csv file, and panel_df and sample_df files
    '''

    
    #This stops a warning getting returned that we don't need to worry about
    pd.set_option('mode.chained_assignment',None)
    
    # Reload sample_df and panel_df, which allows them to be editted
    sample_df, panel_df = reload_dfs(sample_df, panel_df)
        
    # Load in the cell data
    cell_df = pd.read_csv(Path(directory, cell_df), low_memory=False)
    print(f'Loaded cell file, {str(cell_df.shape[0])} cells found')
        
    
    if cell_table_format=='bodenmmiller':
       
        # Extract only the intensities from the cell table
        try:
            cell_df_intensities = cell_df[[col for col in cell_df.columns if 'Intensity_MeanIntensity_FullStack' in col]].copy()
        except:
            print('Could not find Intensity_MeanIntensity_FullStack columns in the cell table! Cell table may have been exported incorrectly from CellProfiler')
            raise

        # Remap the column names to the proper channel names
        mapping = dict(zip(panel_df['cell_table_channel'],panel_df[panel_df_target_col]))
        cell_df_intensities.rename(columns=mapping, inplace=True)    
        
    elif cell_table_format=='cleaned':
        
        # Extract columns from those in the panel file
        marker_cols = [col for col in cell_df.columns if col in panel_df[panel_df_target_col].values.tolist()]
              
        # Check that all the markers in panel file were found in the cell table, and vice versa
        utils.compare_lists(marker_cols, panel_df[panel_df_target_col].tolist(), 'Markers from cell DF columns', 'Markers in panel file', return_error=True)

        cell_df_intensities = cell_df[marker_cols].copy()
               
    # If only a single normalisation provided, make a list anyway
    if not isinstance(marker_normalisation, list):
        marker_normalisation=[marker_normalisation]
    
    # Cell cell intensities
    markers_normalised = cell_df_intensities
    
    # Error catching
    assert markers_normalised.shape[1] == panel_df.shape[0], 'Length of panel and markers do not match!'    
        
    # Get in order that appear in panel
    utils.compare_lists(panel_df['Target'].tolist(), markers_normalised.columns.tolist(),'PanelFile','MarkerDF')   
    markers_normalised = markers_normalised.reindex(columns=panel_df['Target'])
            
    # For each method in the list, intrepret it and assess results
    for method in marker_normalisation:
        
        if method[0]=='q':
            quantile = round(float(method[1:])/100,5)
            markers_normalised = markers_normalised.div(markers_normalised.quantile(q=quantile)).clip(upper=1)        
            markers_normalised.fillna(0, inplace=True)    
            print(f'\nData normalised to {str(quantile)} quantile')
        
        elif 'arcsinh' in method:
            cofactor=int(method[7:])
            markers_normalised = np.arcsinh(markers_normalised/cofactor)
            print(f'\nData Arcsinh adjusted with cofactor {str(cofactor)}')            
        
        elif method=='log2':
            markers_normalised = np.log2(markers_normalised)
            print(f'\nData Log2 adjusted')            
        
        elif method=='log10':
            markers_normalised = np.log10(markers_normalised)          
            print(f'\nData Log10 adjusted')
            
        else:
            print(f'Normalised method {method} no recognised')
    
    # Create AnnData object using the normalised markers
    adata = sc.AnnData(markers_normalised)
    
    # Get var names from hraders of markers_normalised
    adata.var_names=markers_normalised.columns.tolist()
    
    if cell_table_format=='bodenmmiller':
        
        # Map ROI information
        adata.obs['ROI']=cell_df['ImageNumber'].map(dict(zip(sample_df['ImageNumber'],sample_df.index))).values.tolist()        
        
    elif cell_table_format=='cleaned':
        
        # Map ROI information
        adata.obs['ROI']=cell_df[cell_df_ROIcol].values.tolist()        
                                                        
    # Add in any extra columns from cell table
    for c in cell_df_extra_columns:
        adata.b[c]=cell_df[c].values.tolist()
                                                
    # Add in spatial information
    adata.obsm['spatial'] = cell_df[[cell_df_x, cell_df_y]].to_numpy()                                                
    adata.obs['X_loc'] = cell_df[cell_df_x].values.tolist()
    adata.obs['Y_loc'] = cell_df[cell_df_y].values.tolist()
    
    # Add in a master cell index
    adata.obs['Master_Index']=adata.obs.index.values.tolist()
                                                
    if not dictionary==None:
       
        #Read dictionary from file - look in this directory, and in 'cpout'   
        try:
            try:
                obs_dict = pd.read_csv(dictionary, low_memory=False, index_col=0).to_dict()
            except:
                obs_dict = pd.read_csv(Path(directory, dictionary), low_memory=False, index_col=0).to_dict()
        except:
            print('Could not find dictionary file')
            raise
            
        # Add the new columns based off the dictionary file
        for i in obs_dict.keys():
            adata.obs[i]=adata.obs['ROI'].map(obs_dict[i]).values.tolist()
            
            # By default .obs will be added as categories unless specified
            if i not in non_cat_obs:
                adata.obs[i] = adata.obs[i].astype('category')
                print(f'Obs {i} added as categorical variable with following categories:')
                print(adata.obs[i].cat.categories.tolist())
                
            else:
                print(f'Obs {i} NOT converted to categorical')
        
        # Store a list of categorical obs in the adata.uns
        adata.uns.update({'categorical_obs':[x for x in obs_dict if x not in non_cat_obs]})
            
        # Checking for NaNs in adata.obs, which could indicate the dictionary mapping has failed
        obs_nans = adata.obs.isna().sum(axis=0) / len(adata.obs) * 100
        
        if obs_nans.mean() != 0:
            print('WARNING! Some obs columns have NaNs present, which could indicate your dictionary has not been setup correctly')
            utils.print_full(pd.DataFrame(obs_nans, columns = ['Percentage of NaNs']))
        
    else:
        print('No dictionary provided')
    
    print('Markers imported:')
    print(adata.var_names)
    print(adata)
    
    utils.adlog(adata, 'AnnData object created', sc)
    
    adata.uns.update({'sample':sample_df.copy(),
                     'panel':panel_df.copy()})
    
    if return_normalised_markers:
        return adata, markers_normalised
    else:
        return adata
                  
                
            
def stacks_to_imagefolders(bodenmiller_folder='cpout',     
                            sample_df='sample_df.csv',
                            panel_df='panel_df.csv',
                            unstacked_output_folder = 'images', #The name of the folder where tiffs will be extracted
                            masks_output_folder = 'masks', #The name of the folder where renamed masks will be stored
                            sample_df_filename_col='FileName_FullStack',
                            sample_df_mask_col='FileName_cellmask',
                            panel_df_target_col='Target'):
    '''
    This function creates mask and image folders using the Bodenmiller outputs
    '''

    # Make output directories if they don't exist
    output = Path(unstacked_output_folder)
    output.mkdir(exist_ok=True)
    
    # Directs to correct subfolders for masks and images
    masks_folder = Path(bodenmiller_folder, 'masks')
    input_folder = Path(bodenmiller_folder, 'images') 
    
    # Reload sample and panel from disk unless they are passed
    sample_df, panel_df = reload_dfs(sample_df, panel_df)
    
    # Get paths for all the .tiff files in the input directory
    tiff_paths = list(input_folder.rglob('*.tiff'))
   
    print(f'Unpacking {str(len(tiff_paths))} ROIs...')
    
    metadata_rois = sample_df[sample_df_filename_col].tolist()
    detectedimages_rois = [os.path.basename(x) for x in tiff_paths]
    
    meta_not_actual = [x for x in metadata_rois if x not in detectedimages_rois]
    actual_not_meta = [x for x in detectedimages_rois if x not in metadata_rois]    
    
    try:
        assert meta_not_actual == actual_not_meta, f"Number of ROIs in image file not equal to number of full stacks found"
    except:                
        print('ROIs referred in metadata without image stacks being detected:')
        print(meta_not_actual)
        print('ROIs with images that arent referred to in metadata:')
        print(actual_not_meta)
    
    
    # Find all masks in folder
    mask_paths = list(masks_folder.rglob('*.tiff'))
    
    metadata_masks_rois = sample_df[sample_df_mask_col].tolist()
    detectedmaskss_rois = [os.path.basename(x) for x in mask_paths]
    
    meta_not_actual = [x for x in metadata_masks_rois if x not in detectedmaskss_rois]
    actual_not_meta = [x for x in detectedmaskss_rois if x not in metadata_masks_rois]      
                           
    try:
        assert meta_not_actual == actual_not_meta, f"Number of masks in image file not equal to number of mask files found"
    except:                
        print('ROIs referred in metadata without masks being detected:')
        print(meta_not_actual)
        print('Masks that arent referred to in metadata:')
        print(actual_not_meta)                           
                           
   
    for path in tiff_paths:

        # Load the image
        image = tp.imread(path)
        
        # Get the file name from the path
        image_filename = os.path.basename(path)
        
        # Get the ROI name from the sample table
        folder_name = sample_df.loc[sample_df[sample_df_filename_col]==image_filename, sample_df_filename_col].index[0]
                      
        # Make the output folder
        output_dir = Path(unstacked_output_folder,folder_name)
        output_dir.mkdir(exist_ok=True)        

        # Itterate through channels, writing an image for each
        for i, channel_name in enumerate(panel_df[panel_df_target_col]):
                              
            image_to_save = image[i]
            
            # Remove any negative values
            image_to_save[image_to_save<0] = 0
            
            # Set to 16 bit
            image_to_save = np.uint16(image_to_save)
            
            tp.imwrite(join(output_dir, (channel_name+'.tiff')), image_to_save)
                   
    # Make mask directory if doesnt exist
    masks_output_folder = Path(masks_output_folder)
    masks_output_folder.mkdir(exist_ok=True)  

    # Find all masks in folder
    mask_paths = list(masks_folder.rglob('*.tiff'))

    for path in mask_paths:

        mask_filename = os.path.basename(path)

        # Get the ROI name from the sample table
        roi_name = sample_df.loc[sample_df[sample_df_mask_col]==mask_filename, sample_df_mask_col].index[0]            

        shutil.copy(path,os.path.join(masks_output_folder, (roi_name+'.tiff')))
        

def remove_ROIs_and_markers(adata,
                        ROI_obs='ROI',
                        ROIs_to_remove=[],
                        Markers_to_remove=['DNA1', 'DNA3']):
    '''
    This function will remove unused or failed ROIs and/or markers
    '''
    
    # Check given as lists
    if not type(ROIs_to_remove)==list:
        [ROIs_to_remove]
        
    if not type(Markers_to_remove)==list:
        [Markers_to_remove]
        
    print('Removing markers:')
    print(Markers_to_remove)
    
    print('Removing ROIs:')
    print(ROIs_to_remove)

    # Make a list of all markers found
    all_markers = adata.var_names.tolist()

    #Make a new list that only has the markers we're interested in
    markers_limited = [m for m in all_markers if m not in Markers_to_remove]
    
    # Make a list of all markers found
    all_rois = adata.obs 
    
    # Fix sample dataframe in AnnData
    adata.uns['sample'] = adata.uns['sample'].loc[~adata.uns['sample'].index.isin(ROIs_to_remove),:]
    
    # Save updated sample_df
    adata.uns['sample'].to_csv('sample_df.csv')

    return adata[~adata.obs[ROI_obs].isin(ROIs_to_remove),
                            markers_limited]

        
