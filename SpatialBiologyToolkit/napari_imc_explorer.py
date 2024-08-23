# Standard Library Imports
import itertools
import os
from pathlib import Path

# Third-Party Imports
import anndata as ad
from magicgui import magicgui, widgets
import napari
import numpy as np
import pandas as pd
import skimage as sk
from skimage import color, io, transform
import vispy
from matplotlib import colormaps


def napari_imc_explorer(
    masks_folder: str = 'Masks',
    image_folders: list = ['Images'],
    HE_folder: str = 'HE',
    roi_obs: str = 'ROI',
    adata: ad.AnnData = ad.AnnData(),
    check_masks: bool = False,
    mask_extension: str = '.tiff'
) -> napari.Viewer:
    """
    Start an interactive Napari viewer for exploring IMC data.

    Parameters
    ----------
    masks_folder : str
        Directory containing a mask for each ROI, each file named after the ROI. Masks should be uint16 .tif files.
    image_folders : list
        Directories containing subdirectories, each named after ROIs in the AnnData. Images are named after channels (adata.var_names), uint16 .tiff files.
    HE_folder : str
        Directory containing a color (RGB) .tif file, named after each ROI.
    roi_obs : str
        Column in adata.obs indicating the ROI.
    adata : AnnData
        AnnData object as created from the pipeline.
    check_masks : bool
        If True, will check that all the masks match the number of cells in the AnnData object.
    mask_extension : str
        Extension for mask files (typically .tiff or .tif).
        
    Returns
    -------
    napari.Viewer
        The Napari viewer object.
    """

    if not isinstance(image_folders, list):
        image_folders = [image_folders]

    def _check_all_masks(adata, roi_obs=roi_obs):
        """
        Check that the number of cells in each mask matches the number of cells in each ROI.
        """
        roi_list = adata.obs[roi_obs].unique()
        print(f'Matching AnnData {roi_obs} to masks in {masks_folder}')
        print(roi_list)
        
        for roi_name in roi_list:
            mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
            cell_list_from_mask = np.unique(mask.flatten())
            cell_list_from_anndata = adata.obs.loc[adata.obs[roi_obs] == roi_name, :]
            cell_list_from_anndata.reset_index(drop=True, inplace=True)
            cell_list_from_anndata = cell_list_from_anndata.index.to_numpy() #+ 1
            assert np.all(cell_list_from_mask[0] == cell_list_from_anndata[0]), f'Mask and cell table do not match for {roi_name}'
            print(f'{roi_name} matched!')
        
        print('All ROIs matched successfully')
        
    def _find_tiff_files(directory):
        """
        Find all TIFF/TIF files in a specified directory.
        """
        tiff_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.tiff', '.tif')):
                    tiff_files.append(os.path.join(root, file))
        return tiff_files

    def _list_folders_in_directory(directory):
        """
        List all folders in a specified directory.
        """
        return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

    def _load_imc_image(file, quantile=0.999, colormap=None, recolour_image=False, minimum_pixel_counts=0.1):
        """
        Load a single IMC image, including removing some background and normalising to a percentile.
        """
        image = sk.io.imread(file)
        np.where(image > minimum_pixel_counts, image, 0)
        max_quant = np.quantile(image, quantile)
        if max_quant < 5:
            max_quant = 3
        image = image / max_quant
        image = np.clip(image, 0, 1)
        image_name = os.path.splitext(os.path.basename(file))[0]
        viewer.add_image(image, name=image_name, blending='additive', colormap=colormap)

    def _add_roi_images_raw(roi_name, quantile=0.999, colour_map=['r', 'g', 'b', 'c', 'm', 'y'], minimum_pixel_counts=0.1, recolour_image=False):
        """
        Add all images from one ROI folder, cycling through 6 colours.
        """
        mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
        tiffs = []
        for folder in image_folders:
            tiff_paths = _find_tiff_files(Path(folder, roi_name))
            tiffs.extend(tiff_paths)

        for file, colour in zip(tiffs, itertools.cycle(colour_map)):
            _load_imc_image(file, quantile=quantile, colormap=vispy.color.Colormap([[0, 0, 0], colour]), recolour_image=recolour_image, minimum_pixel_counts=minimum_pixel_counts)
            viewer.layers[-1].visible = False

    def _add_masks(roi_name, adata, pop_obs=None, quant=None, roi_obs='ROI', adata_colormap=True, colour_map=colormaps['tab20'].colors, add_individual_pops=False):
        """
        Add masks to the viewer, optionally with population or quantitative overlays.
        """
        mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
        if 'all_cells' not in viewer.layers:
            viewer.add_labels(mask, name='all_cells')
            viewer.layers[-1].contour = 1
            viewer.layers[-1].visible = False

        adata_roi_obs = adata.obs.loc[adata.obs[roi_obs] == roi_name, :].copy()
        adata_roi_obs.reset_index(drop=True, inplace=True)

        if pop_obs:
            populations = adata.obs[pop_obs].cat.categories.tolist()
            populations_num = np.array(range(len(populations))) + 1
            if adata_colormap and (f'{pop_obs}_colors' in adata.uns):
                colour_map = adata.uns[f'{pop_obs}_colors']
            pop_colormap = {(x+1): y for x, y in enumerate(colour_map)}
            all_pops_mask = np.zeros(mask.shape, dtype='uint16')

            for pop, pop_num in zip(populations, populations_num):
                try:
                    objects = adata_roi_obs.loc[adata_roi_obs[pop_obs] == pop, :].index.to_numpy() + 1
                    pop_mask = np.isin(mask, objects)
                    all_pops_mask = np.where(pop_mask, pop_num, all_pops_mask)
                    if add_individual_pops:
                        viewer.add_labels(pop_mask, name=pop, color={1: pop_colormap[pop_num]})
                        viewer.layers[-1].contour = 1
                        viewer.layers[-1].visible = False
                except:
                    print(f'Error adding group {pop} from {pop_obs}')
            viewer.add_labels(all_pops_mask, name=pop_obs, color=pop_colormap)
            viewer.layers[-1].contour = 1
        elif quant:
            objects = adata_roi_obs.index.to_numpy() + 1
            if quant in adata.obs:
                values = adata_roi_obs[quant]
            elif quant in adata.var_names:
                values = adata.X[adata.obs[roi_obs] == roi_name, adata.var_names == quant].tolist()
            parameter_map = sk.util.map_array(np.asarray(mask), np.asarray(objects), np.asarray(values))
            viewer.add_image(parameter_map, name=quant, blending='additive')

    def _add_HE(roi_name):
        """
        Add H&E image to the viewer.
        """
        imc_image_size = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}')).shape
        HE_image = io.imread(Path(HE_folder, f'{roi_name}.tif'))
        HE_image = transform.resize(HE_image, imc_image_size, anti_aliasing=True)
        viewer.add_image(HE_image, rgb=True, name='HE')

    def _hide_all_layers():
        for layer in viewer.layers:
            layer.visible = False

    hide_all_layers_button = widgets.PushButton(text='Hide all layers', name='hide_all_layers_button')
    hide_all_layers_button.clicked.connect(_hide_all_layers)

    def _delete_all_layers():
        viewer.layers.select_all()
        viewer.layers.remove_selected()

    delete_all_layers_button = widgets.PushButton(text='Delete all layers', name='delete_all_layers_button')
    delete_all_layers_button.clicked.connect(_delete_all_layers)

    def _add_roi_images():
        selected_item = roi_selector.value
        _add_roi_images_raw(selected_item, quantile=quant_select.value, minimum_pixel_counts=minimum_pixel_counts_select.value)

    roi_list = _list_folders_in_directory('Images')
    roi_selector = widgets.ComboBox(label='Select ROI:', choices=roi_list)

    add_roi_images_button = widgets.PushButton(text='Add ALL images for ROI', name='add_roi_images_button')
    add_roi_images_button.clicked.connect(_add_roi_images)

    add_roi_label = widgets.Label(value='Select ROI:')

    def _add_mask_labels():
        selected_item = roi_selector.value
        _add_masks(selected_item, adata, pop_obs=None)

    add_masks_button = widgets.PushButton(text='Add cell mask', name='add_masks_button')
    add_masks_button.clicked.connect(_add_mask_labels)

    def _add_HE_layer():
        selected_item = roi_selector.value
        _add_HE(selected_item)

    add_HE_button = widgets.PushButton(text='Add H+E image', name='add_HE_button')
    add_HE_button.clicked.connect(_add_HE_layer)

    def _flip_HE_Y():
        viewer.layers['HE'].data = np.flipud(viewer.layers['HE'].data)

    def _flip_HE_X():
        viewer.layers['HE'].data = np.fliplr(viewer.layers['HE'].data)

    add_flip_HE_X_button = widgets.PushButton(text='Flip H+E on X axis', name='add_flip_HE_X_button')
    add_flip_HE_X_button.clicked.connect(_flip_HE_X)

    add_flip_HE_Y_button = widgets.PushButton(text='Flip H+E on Y axis', name='add_flip_HE_Y_button')
    add_flip_HE_Y_button.clicked.connect(_flip_HE_Y)

    if check_masks:
        _check_all_masks(adata, roi_obs=roi_obs)
    
    viewer = napari.Viewer()
    viewer.window.add_dock_widget([add_roi_label, roi_selector, hide_all_layers_button, delete_all_layers_button, add_roi_images_button, add_masks_button, add_HE_button, add_flip_HE_Y_button, add_flip_HE_X_button], name='Controls')

    image_folder_dict = {}
    image_ext_dict = {}
    im_list = []
    for i in image_folders:
        im_files = []
        for roi in roi_list:
            im_files = _find_tiff_files(Path(i, roi))
            if im_files:
                break
        im_list_new = [os.path.basename(x).split('.')[:-1] for x in im_files]
        im_list_new = [".".join(x) for x in im_list_new]
        im_extensions = [os.path.basename(x).split('.')[-1] for x in im_files]
        for x, ext in zip(im_list_new, im_extensions):
            image_ext_dict[x] = ext
        for x in im_list_new:
            image_folder_dict[x] = str(i)
        im_list.extend(im_list_new)
    
    @magicgui(x=dict(widget_type='Select', choices=im_list, label='Select images'), call_button='Add images')
    def _image_selector(x: list):
        _add_images_from_list(x)

    def _add_images_from_list(selected_images):
        for image, colour in zip(selected_images, itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y'])):
            file = Path(image_folder_dict[image], roi_selector.value, f'{image}.{image_ext_dict[image]}')
            print(file)
            _load_imc_image(file, quantile=quant_select.value, minimum_pixel_counts=minimum_pixel_counts_select.value, colormap=vispy.color.Colormap([[0, 0, 0], colour]))

    quant_select_label = widgets.Label(value='Normalise intensity to quantile:')
    quant_select = widgets.FloatSpinBox(min=0, max=1, value=0.999, step=0.001)
    minimum_pixel_counts_select_label = widgets.Label(value='Minimum pixel value:')
    minimum_pixel_counts_select = widgets.FloatText(value=0.1, min=0)
    
    viewer.window.add_dock_widget([_image_selector, quant_select_label, quant_select, minimum_pixel_counts_select_label, minimum_pixel_counts_select], name='Add raw images')

    categorical_obs_columns = [col for col in adata.obs.columns if adata.obs[col].dtype == 'category']
    
    @magicgui(x=dict(widget_type='Select', choices=categorical_obs_columns, label='Select categories'), call_button='Add as masks')
    def _obs_selector(x: list):
        _add_obs_masks(x)

    def _add_obs_masks(obs_list):
        for obs in obs_list:
            _add_masks(roi_name=roi_selector.value, adata=adata, pop_obs=obs, roi_obs='ROI', adata_colormap=True, colour_map=colormaps['tab20'].colors, add_individual_pops=individual_pops_toggle.value)

    individual_pops_toggle = widgets.CheckBox(value=False, text='Add individual groups from .obs as masks')

    viewer.window.add_dock_widget([_obs_selector, individual_pops_toggle], name='Categories as masks')

    numerical_obs_columns = [col for col in adata.obs.columns if adata.obs[col].dtype in ['float32', 'float64', 'int32']]
    
    @magicgui(x=dict(widget_type='Select', choices=adata.var_names.tolist() + numerical_obs_columns, label='Select numeric'), call_button='Add as overlays')
    def _quant_selector(x: list):
        _add_quant_masks(x)

    def _add_quant_masks(quant_list):
        for quant in quant_list:
            _add_masks(roi_name=roi_selector.value, adata=adata, pop_obs=None, quant=quant, roi_obs='ROI', adata_colormap=False, add_individual_pops=individual_pops_toggle.value)

    viewer.window.add_dock_widget(_quant_selector, name='Numeric as masks')

    napari.run()
    return viewer
