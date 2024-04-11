import pandas as pd
import napari
import os
import numpy as np
import itertools
import vispy
import anndata as ad
from skimage import color, io, transform
import skimage as sk
from matplotlib import colormaps
import napari
from magicgui import magicgui, widgets
from pathlib import Path


def napari_imc_explorer(masks_folder = 'Masks',
                        image_folders = ['Images'],
                        HE_folder = 'HE',
                        roi_obs='ROI',
                        adata = ad.AnnData(),
                        check_masks=False):
    '''
    This will start an interactive Napari viewer, assuming the following directories and objects are in place:
    
    adata - AnnData object as created from the pipeline
    masks_folder - Directory containing a mask for each ROI, each file named after the ROI. Masks should be uint16 .tif files.
    image_folders - Directories containing subdirectories, each named after ROIs in the AnnData. Images are named after channels (adata.var_names), uint16 .tiff files 
    HE_folder - Directory containing a colour (RGFB) .tif file, named after each ROI
    check_masks - If True, will check that all the masks match the number of cells in the AnnData object.
    '''
    
    # If only one image folder given, make it into a list of one
    if not isinstance(image_folders,list):
        image_folders = [image_folders]
    
    def check_all_masks(adata, roi_obs=roi_obs):
        '''
        This will check that the number of cells in each mask matches the number of cells in each ROI.
        '''
        
        roi_list = adata.obs[roi_obs].unique()
        print(f'Matching AnnData {roi_obs} to masks in {masks_folder}')
        print(roi_list)
        
        for roi_name in roi_list:
            mask = sk.io.imread(Path(masks_folder,f'{roi_name}.tif'))
            
            # Get a list of cells as they appear in the mask
            cell_list_from_mask = np.unique(mask.flatten())
            
            # Filter to specific ROI
            cell_list_from_anndata = adata.obs.loc[adata.obs[roi_obs]==roi_name, :]
            
            # Reset index
            cell_list_from_anndata.reset_index(drop=True, inplace=True)
            
            # Get list of cells
            cell_list_from_anndata = cell_list_from_anndata.index.to_numpy()+1
            
            assert np.all(cell_list_from_mask==cell_list_from_mask), f'Mask and cell table do not match for {roi_name}'
            
        print('All ROIs matched successfully')
        
    
    def find_tiff_files(directory):
        """
        Find all TIFF/TIF files in a specified directory.

        Parameters:
        directory (str): The path to the directory.

        Returns:
        list: A list of paths to TIFF files found in the directory.
        """
        tiff_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.tiff') or file.lower().endswith('.tif'):
                    tiff_files.append(os.path.join(root, file))

        return tiff_files

    
    def list_folders_in_directory(directory):
        """
        List all folders in a specified directory.

        Parameters:
        directory (str): The path to the directory.

        Returns:
        list: A list of folder names found in the directory.
        """
        return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]



    def load_imc_image(file, quantile=0.999, colormap=None, recolour_image=False, minimum_pixel_counts=0.1):
    
        '''
        Load a single IMC image, including removing some background and normalising to a percentile
        '''
        
        # Load raw    
        image = sk.io.imread(file)

        # Remove background
        np.where(image > minimum_pixel_counts, image, 0)

        # Norm to 99th percentile, or 3 if lower
        max_quant = np.quantile(image, quantile)

        if max_quant < 5:
            max_quant=3

        image = image / max_quant
        image = np.clip(image, 0, 1)

        # Get image name from file path
        image_name = os.path.splitext(os.path.basename(file))[0]

        if recolour_image:
            # Create a lookup table
            lut = np.array([colour * gray_value / 65535 for gray_value in range(65536)])

            # Apply the lookup table
            image = lut[(image * 255).astype(int)]

        # Add image
        viewer.add_image(image, name=image_name, blending='additive', colormap = colormap)

    
    
    def add_roi_images_raw(roi_name, 
                quantile=0.999, 
                colour_map=['r', 'g', 'b', 'c', 'm', 'y'],#colormaps['tab20'].colors,
                minimum_pixel_counts=0.1,
                recolour_image = False):
        
        '''
        This will add all images from one ROI folder, cycling through 6 colours
        '''
    
        # Load the mask and TIFFS for the ROI
        mask = sk.io.imread(Path(masks_folder,f'{roi_name}.tif'))
        
        # Find all .tiff/.tif paths in any image folders
        tiffs =[]
        for i in image_folders:
            tiff_paths = find_tiff_files(Path(i,roi_name))
            tiffs = tiffs + tiff_paths

        #viewer.add_labels(mask, name='cell_mask')

        for file, colour in zip(tiffs, itertools.cycle(colour_map)):

            load_imc_image(file,
                            quantile=quantile, 
                            colormap= vispy.color.Colormap([[0, 0, 0], colour]), 
                            recolour_image=recolour_image, 
                            minimum_pixel_counts=minimum_pixel_counts)
             
            # Hide layer
            viewer.layers[-1].visible = False
            
    def add_masks(roi_name,
                  adata,
                  pop_obs=None, #'population_broad',
                  quant=None,
                  roi_obs='ROI', 
                  adata_colormap=True,
                  colour_map=colormaps['tab20'].colors,
                  add_individual_pops=False
                 ):
                
        mask = sk.io.imread(Path(masks_folder,f'{roi_name}.tif'))
        
        # Add a mask with all cells (if not already present)
        if 'all_cells' not in viewer.layers:
        
            viewer.add_labels(mask, name='all_cells')
            viewer.layers[-1].contour=1
            viewer.layers[-1].visible = False

        # Get cell table
        adata_roi_obs = adata.obs.loc[adata.obs[roi_obs]==roi_name, :].copy()
            
        # Reset index to match the masks (hopefully!)
        adata_roi_obs.reset_index(drop=True, inplace=True)
        
        # If an adata.obs is supplied
        if pop_obs:
                 
            # Get populations
            populations = adata.obs[pop_obs].cat.categories.tolist()
            
            # Pop numbers
            populations_num = np.array(range(len(populations))) + 1    
            
            # Use AnnData colourmap, ideally
            if adata_colormap & (f'{pop_obs}_colors' in adata.uns):
                colour_map = adata.uns[f'{pop_obs}_colors']

            pop_colormap = {(x+1):y for x, y in enumerate(colour_map)}
            
            # Setup a blank mask we will add pops to in series
            all_pops_mask = np.zeros(mask.shape, dtype='uint16')
   
            for pop, pop_num in zip(populations, populations_num):
                try:
                    objects = adata_roi_obs.loc[adata_roi_obs[pop_obs]==pop, :].index.to_numpy() + 1

                    pop_mask = np.isin(mask, objects)

                    all_pops_mask = np.where(pop_mask, pop_num, all_pops_mask)
                    
                    if add_individual_pops:
                        viewer.add_labels(pop_mask, name=pop, color={1:pop_colormap[pop_num]})
                        viewer.layers[-1].contour=1
                        viewer.layers[-1].visible = False
                except:
                    print(f'Error adding group {pop} from {pop_obs}')
                
            viewer.add_labels(all_pops_mask, name=pop_obs, color=pop_colormap)    
            viewer.layers[-1].contour=1
            
        # If a quantitative value is supplied, e.g. marker or from adata.obs
        elif quant:
            
            # Label IDs
            objects = adata_roi_obs.index.to_numpy() + 1
         
            # If value from adata.obs..
            if quant in adata.obs:
            
                values = adata_roi_obs[quant]
            
            # If a marker value...
            elif quant in adata.var_names:
            
                values = adata.X[adata.obs[roi_obs]==roi_name, adata.var_names == quant].tolist()
                
            # Add fuctionality to handle adata.obs which are continuous...
            parameter_map = sk.util.map_array(np.asarray(mask), np.asarray(objects), np.asarray(values))
            
            viewer.add_image(parameter_map, name=quant, blending='additive')
            
    def add_HE(roi_name):    
        
        # Get size of ROI from mask file
        imc_image_size = sk.io.imread(Path(masks_folder,f'{roi_name}.tif')).shape
        
        # Load image
        HE_image = io.imread(Path(HE_folder,f'{roi_name}.tif'))

        # Resize the image
        HE_image = transform.resize(HE_image, imc_image_size, anti_aliasing=True)
        
        # Reverse the direction of the Y axis
        #HE_image = np.flipud(HE_image)
        
        # Reverse the direction of the X axis
        #HE_image = np.fliplr(HE_image)

        # Add to Napari
        viewer.add_image(HE_image, rgb=True, name='HE')
        
        
    ########### Button to hide all layers
    def hide_all_layers():
        for layer in viewer.layers:
            layer.visible = False
            
    hide_all_layers_button = widgets.PushButton(text='Hide all layers',
                                                name='hide_all_layers_button')

    hide_all_layers_button.clicked.connect(hide_all_layers)        
            
    ########### Button to delete all layers
    def delete_all_layers():
        viewer.layers.select_all()
        viewer.layers.remove_selected()
        
    delete_all_layers_button = widgets.PushButton(text='Delete all layers',
                                                name='Delete_all_layers_button')

    delete_all_layers_button.clicked.connect(delete_all_layers)

    ########### Button to add ROI images
    def add_roi_images():
        selected_item = roi_selector.value
        # Replace the following line with your custom action using the selected item
        # del viewer.layers[:]
        add_roi_images_raw(selected_item, quantile=quant_select.value, minimum_pixel_counts=minimum_pixel_counts_select.value)
     

    roi_list = list_folders_in_directory('Images')
    roi_selector = widgets.ComboBox(label='Select ROI:', choices=roi_list)
        
            
    add_roi_images_button = widgets.PushButton(text='Add ALL images for ROI', 
                                              name='add_roi_images_button')

    add_roi_images_button.clicked.connect(add_roi_images)
    
    add_roi_label = widgets.Label(value='Select ROI:')
    
        
    ########### Button to add masks    
    def add_mask_labels():
        selected_item = roi_selector.value
        add_masks(selected_item, adata, pop_obs=None)    

    # Create a button to add masks
    add_masks_button = widgets.PushButton(text='Add cell mask', 
                                              name='add_masks_button')
    add_masks_button.clicked.connect(add_mask_labels)    

    ########### Button to add HE images    
    def add_HE_layer():
        selected_item = roi_selector.value
        add_HE(selected_item)    

    # Create a button to add masks
    add_HE_button = widgets.PushButton(text='Add H+E image', 
                                              name='add_HE_button')
    add_HE_button.clicked.connect(add_HE_layer)

    ########### Buttons to flip HE
    def flip_HE_Y():
        viewer.layers['HE'].data = np.flipud(viewer.layers['HE'].data)
    def flip_HE_X():
        viewer.layers['HE'].data = np.fliplr(viewer.layers['HE'].data)    

    add_flip_HE_X_button = widgets.PushButton(text='Flip H+E on X axis', 
                                              name='add_flip_HE_X_button')
    add_flip_HE_X_button.clicked.connect(flip_HE_X)

    add_flip_HE_Y_button = widgets.PushButton(text='Flip H+E on Y axis', 
                                              name='add_flip_HE_Y_button')
    add_flip_HE_Y_button.clicked.connect(flip_HE_Y)


    if check_masks:
        check_all_masks(adata, roi_obs=roi_obs)
    
    viewer = napari.Viewer()

    # Add the buttons and dropdowns to the Napari viewer
    viewer.window.add_dock_widget([add_roi_label, roi_selector, hide_all_layers_button, delete_all_layers_button, 
                                   add_roi_images_button, add_masks_button, add_HE_button,
                                   add_flip_HE_Y_button, add_flip_HE_X_button], name='Controls')
                                   
    
    ############ WIDGET 2 - Image selector
     
    # This now handles being given a list of image directories, and works with .tiff and .tif files
    image_folder_dict = {}
    image_ext_dict = {}
    im_list = []
    for i in image_folders:
        
        # Not all folders may have images for all rois, so find first folder with images
        im_files = []
        for roi in roi_list:
            im_files = find_tiff_files(Path(i, roi))
            if im_files:
                break
                
        im_list_new = [os.path.basename(x).split('.')[:-1] for x in im_files]
        im_list_new = [".".join(x) for x in im_list_new]
        
        im_extensions = [os.path.basename(x).split('.')[-1] for x in im_files]
        
        for x, ext in zip(im_list_new, im_extensions):
            image_ext_dict[x] = ext
        
        for x in im_list_new:
            image_folder_dict[x]=str(i)
        
        im_list = im_list + im_list_new
        
    
    # Selector for different images
    @magicgui(x=dict(widget_type='Select', choices=im_list, label='Select images'), 
              call_button='Add images')   
    def image_selector(x: list):
        add_images_from_list(x)                        

    def add_images_from_list(selected_images):
        # Do something with the selected images
        
        for image, colour in zip(selected_images, itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y'])):

            file = Path(image_folder_dict[image], roi_selector.value, f'{image}.{image_ext_dict[image]}')
            print(file)
            
            load_imc_image(file,
                           quantile=quant_select.value, 
                           minimum_pixel_counts=minimum_pixel_counts_select.value,
                           colormap= vispy.color.Colormap([[0, 0, 0], colour]))

    quant_select_label = widgets.Label(value='Normalise intensity to quantile:')
    quant_select = widgets.FloatSpinBox(min=0, max=1, value=0.999, step=0.001)
    minimum_pixel_counts_select_label = widgets.Label(value='Minimum pixel value:')
    minimum_pixel_counts_select = widgets.FloatText(value=0.1, min=0)
    
    viewer.window.add_dock_widget([image_selector,quant_select_label,quant_select, 
                                    minimum_pixel_counts_select_label,minimum_pixel_counts_select], name='Add raw images')
    
    ############ WIDGET 3 - Population selector
    
    categorical_obs_columns = [col for col in adata.obs.columns if adata.obs[col].dtype == 'category']
       
     # Selector for different images
    @magicgui(x=dict(widget_type='Select', choices=categorical_obs_columns, label='Select categories'), 
              call_button='Add as masks')   
    def obs_selector(x: list):
        add_obs_masks(x)                        

    def add_obs_masks(obs_list):
        # Do something with the selected images
        
        for obs in obs_list:

            add_masks(roi_name = roi_selector.value,
                      adata = adata,
                      pop_obs=obs, 
                      roi_obs='ROI', 
                      adata_colormap=False,
                      colour_map=colormaps['tab20'].colors,
                      add_individual_pops=individual_pops_toggle.value)
    
    individual_pops_toggle = widgets.CheckBox(value=False, text='Add individual groups from .obs as masks')
    
    viewer.window.add_dock_widget([obs_selector, individual_pops_toggle], name='Categories as masks')   
    
    ############ WIDGET 4 - Add quantities over masks
    
    numerical_obs_columns = [col for col in adata.obs.columns if adata.obs[col].dtype in ['float32','float64','int32']]
       
    # Selector for different numeric values from adata.obs, and panel markers
    @magicgui(x=dict(widget_type='Select', choices=adata.var_names.tolist() + numerical_obs_columns, label='Select numeric'), 
              call_button='Add as overlays')   
    def quant_selector(x: list):
        add_quant_masks(x)                        

    def add_quant_masks(quant_list):
        # Do something with the selected images
        
        for quant in quant_list:

            add_masks(roi_name = roi_selector.value,
                      adata = adata,
                      pop_obs=None, 
                      quant=quant,
                      roi_obs='ROI', 
                      adata_colormap=False,
                      add_individual_pops=individual_pops_toggle.value)
        
    viewer.window.add_dock_widget(quant_selector, name='Numeric as masks')   
    

    # Start the Napari event loop
    napari.run()
    return viewer
        
