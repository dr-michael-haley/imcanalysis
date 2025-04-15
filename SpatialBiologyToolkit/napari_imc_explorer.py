# Standard Library Imports
import itertools
import os
import pickle  # For saving and loading
from pathlib import Path

# Third-Party Imports
import anndata as ad
from magicgui import magicgui, widgets
import napari
import numpy as np
import pandas as pd
import skimage as sk
from skimage import color, io, transform, segmentation
import vispy
from matplotlib import colormaps
from qtpy.QtWidgets import QWidget, QVBoxLayout
from napari.utils.colormaps import Colormap  # For colormap reconstruction
from napari.utils import DirectLabelColormap

def napari_imc_explorer(
    masks_folder: str = 'Masks',
    image_folders: list = ['Images'],
    roi_obs: str = 'ROI',
    cell_id_in_mask_obs: str = 'ObjectNumber',
    adata: ad.AnnData = ad.AnnData(),
    check_masks: bool = True,
    mask_extension: str = None 
) -> napari.Viewer:
    """
    Start an interactive Napari viewer for exploring IMC data.

    Parameters
    ----------
    masks_folder : str
        Directory containing a mask for each ROI, each file named after the ROI. Masks should be uint16 image files.
    image_folders : list
        Directories containing subdirectories, each named after ROIs in the AnnData. Images are named after channels (`adata.var_names`), uint16 image files.
    roi_obs : str
        Column in `adata.obs` indicating the ROI.
    cell_id_in_mask_obs : str
        Column in `adata.obs` indicating the ID's in the mask file for each cell.
    adata : AnnData
        AnnData object as created from the pipeline.
    check_masks : bool
        If True, will check that all the masks match the number of cells in the AnnData object.
    mask_extension : str, optional
        Extension for mask files (e.g., `'.tiff'` or `'.tif'`). If `None`, the extension will be automatically determined from the first file found in the `masks_folder`.

    Returns
    -------
    napari.Viewer
        The Napari viewer object.
    """
    # Ensure image_folders is a list
    if not isinstance(image_folders, list):
        image_folders = [image_folders]
        
    # Automatically determine mask_extension if not provided
    if mask_extension is None:
        import glob
        mask_files = glob.glob(os.path.join(masks_folder, '*'))
        if len(mask_files) == 0:
            raise FileNotFoundError(f"No mask files found in '{masks_folder}'. Please specify 'mask_extension'.")
        else:
            # Get the extension of the first file
            first_file = os.path.basename(mask_files[0])
            _, ext = os.path.splitext(first_file)
            mask_extension = ext
            print(f"Mask extension automatically set to '{mask_extension}'.")

    # Check if mask object id exists, in which case we use that to match cells in AnnData to id's in mask
    if cell_id_in_mask_obs not in adata.obs.columns:
        print(f"Could not find {cell_id_in_mask_obs} in AnnData obs, so resorting to using index.")
        cell_id_in_mask_obs = None

    def _check_all_masks(adata, roi_obs=roi_obs):
        """
        Check that the number of cells in each mask matches the number of cells in each ROI.
        """
        roi_list = adata.obs[roi_obs].unique()
        print(f'Matching AnnData {roi_obs} to masks in {masks_folder}')
        print(roi_list)
        
        for roi_name in roi_list:
            # Load the mask image
            mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))

            # Get unique cell IDs from the mask
            cell_list_from_mask = np.trim_zeros(np.unique(mask.flatten()))

            # Retrieve cell ids from column, or use index
            if cell_id_in_mask_obs:
                cell_list_from_anndata = adata.obs.loc[adata.obs[roi_obs] == roi_name, cell_id_in_mask_obs]
                cell_list_from_anndata = cell_list_from_anndata.to_numpy()
            else:
                cell_list_from_anndata = adata.obs.loc[adata.obs[roi_obs] == roi_name, :]
                cell_list_from_anndata.reset_index(drop=True, inplace=True)
                cell_list_from_anndata = cell_list_from_anndata.index.to_numpy()

            # Check that the mask and anndata cell IDs match
            assert np.all(cell_list_from_mask == cell_list_from_anndata), f'Mask and cell table do not match for {roi_name}'
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

        Parameters
        ----------
        file : str or Path
            Path to the image file.
        quantile : float
            Quantile for normalizing image intensity.
        colormap : vispy.color.Colormap
            Colormap to use for displaying the image.
        recolour_image : bool
            If True, recolour the image.
        minimum_pixel_counts : float
            Minimum pixel value to consider.

        Returns
        -------
        None
        """
        # Load the image
        image = sk.io.imread(file)
        # Set pixels below minimum_pixel_counts to zero
        image = np.where(image > minimum_pixel_counts, image, 0)
        # Normalize image intensity to the specified quantile
        max_quant = np.quantile(image, quantile)
        if max_quant < 5:
            max_quant = 3
        image = image / max_quant
        image = np.clip(image, 0, 1)
        # Get image name from file name
        image_name = os.path.splitext(os.path.basename(file))[0]
        # Add image to the viewer
        viewer.add_image(image, name=image_name, blending='additive', colormap=colormap)

    def _add_roi_images_raw(roi_name, quantile=0.999, colour_map=['r', 'g', 'b', 'c', 'm', 'y'], minimum_pixel_counts=0.1, recolour_image=False):
        """
        Add all images from one ROI folder, cycling through specified colours.
        """
        # Load the mask to determine the image size (if needed)
        mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
        tiffs = []
        # Collect all TIFF files from the image folders for the given ROI
        for folder in image_folders:
            tiff_paths = _find_tiff_files(Path(folder, roi_name))
            tiffs.extend(tiff_paths)

        # Add each image to the viewer with cycling colours
        for file, colour in zip(tiffs, itertools.cycle(colour_map)):
            _load_imc_image(file, quantile=quant_select.value, colormap=vispy.color.Colormap([[0, 0, 0], colour]), recolour_image=recolour_image, minimum_pixel_counts=minimum_pixel_counts_select.value)
            viewer.layers[-1].visible = False  # Hide the layer by default

    def _add_masks(roi_name, adata, pop_obs=None, quant=None, roi_obs='ROI', adata_colormap=True, colour_map=colormaps['tab20'].colors, add_individual_pops=False):
        """
        Add masks to the viewer, optionally with population or quantitative overlays.
        """
        # Load the mask image
        mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
        # Add the base cell mask layer if not already added
        if 'all_cells' not in [layer.name for layer in viewer.layers]:
            viewer.add_labels(mask, name='all_cells')
            viewer.layers[-1].contour = 1
            viewer.layers[-1].visible = False

        # Get the observation data for the current ROI
        adata_roi_obs = adata.obs.loc[adata.obs[roi_obs] == roi_name, :].copy()
        adata_roi_obs.reset_index(drop=True, inplace=True)

        if pop_obs:
            # Get the unique populations/categories
            populations = adata.obs[pop_obs].cat.categories.tolist()
            populations_num = np.array(range(len(populations))) + 1
            # Use the colormap from adata if available
            if adata_colormap and (f'{pop_obs}_colors' in adata.uns):
                colour_map = adata.uns[f'{pop_obs}_colors']
            pop_colormap = {(x+1): y for x, y in enumerate(colour_map)}
            pop_colormap.update({None:'magenta'})
            all_pops_mask = np.zeros(mask.shape, dtype='uint16')

            # Create a mask for each population
            for pop, pop_num in zip(populations, populations_num):
                try:
                    if cell_id_in_mask_obs:
                        objects = adata_roi_obs.loc[adata_roi_obs[pop_obs] == pop, cell_id_in_mask_obs].to_numpy()
                    else:
                        objects = adata_roi_obs.loc[adata_roi_obs[pop_obs] == pop, :].index.to_numpy() + 1

                    pop_mask = np.isin(mask, objects)
                    all_pops_mask = np.where(pop_mask, pop_num, all_pops_mask)
                    if add_individual_pops:
                        viewer.add_labels(pop_mask, name=pop, colormap=DirectLabelColormap(color_dict={None: 'magenta', 1: pop_colormap[pop_num]}))
                        viewer.layers[-1].contour = 1
                        viewer.layers[-1].visible = False
                except Exception as e:
                    print(f'Error adding group {pop} from {pop_obs}: {e}')
            viewer.add_labels(all_pops_mask, name=pop_obs, colormap=DirectLabelColormap(color_dict=pop_colormap))
            viewer.layers[-1].contour = 1
        elif quant:
            # Add quantitative data as an overlay
            if cell_id_in_mask_obs:
                objects = adata_roi_obs.loc[:, cell_id_in_mask_obs].to_numpy()
            else:
                objects = adata_roi_obs.index.to_numpy() + 1

            if quant in adata.obs:
                values = adata_roi_obs[quant]
            elif quant in adata.var_names:
                values = adata.X[adata.obs[roi_obs] == roi_name, adata.var_names == quant].flatten()
            parameter_map = sk.util.map_array(np.asarray(mask), np.asarray(objects), np.asarray(values))
            viewer.add_image(parameter_map, name=quant, blending='additive')

    def _hide_all_layers():
        """
        Hide all layers in the viewer.
        """
        for layer in viewer.layers:
            layer.visible = False

    # Create a button to hide all layers
    hide_all_layers_button = widgets.PushButton(text='Hide all layers', name='hide_all_layers_button')
    hide_all_layers_button.clicked.connect(_hide_all_layers)

    def _delete_all_layers():
        """
        Delete all layers from the viewer.
        """
        viewer.layers.select_all()
        viewer.layers.remove_selected()

    # Create a button to delete all layers
    delete_all_layers_button = widgets.PushButton(text='Delete all layers', name='delete_all_layers_button')
    delete_all_layers_button.clicked.connect(_delete_all_layers)

    def _add_roi_images():
        """
        Add all images for the selected ROI.
        """
        selected_item = roi_selector.value
        _add_roi_images_raw(selected_item, quantile=quant_select.value, minimum_pixel_counts=minimum_pixel_counts_select.value)

    # List of available ROIs
    roi_list = _list_folders_in_directory(image_folders[0])
    # Selector widget for ROIs
    roi_selector = widgets.ComboBox(label='Select ROI:', choices=roi_list)

    # Button to add all images for the selected ROI
    add_roi_images_button = widgets.PushButton(text='Add ALL images for ROI', name='add_roi_images_button')
    add_roi_images_button.clicked.connect(_add_roi_images)

    # Label for ROI selector
    add_roi_label = widgets.Label(value='Select ROI:')

    def _add_mask_labels():
        """
        Add cell masks for the selected ROI.
        """
        selected_item = roi_selector.value
        _add_masks(selected_item, adata, pop_obs=None)

    # Button to add cell masks
    add_masks_button = widgets.PushButton(text='Add cell mask', name='add_masks_button')
    add_masks_button.clicked.connect(_add_mask_labels)

    # Check masks if required
    if check_masks:
        _check_all_masks(adata, roi_obs=roi_obs)
    
    # Create the Napari viewer
    viewer = napari.Viewer()

    # Function to get current layer names
    def get_layer_names(*args):
        return [layer.name for layer in viewer.layers]

    # --- Layer Management Widget ---
    # Create Layer Management Widget
    layer_management_widget = QWidget()
    layout = QVBoxLayout()
    layer_management_widget.setLayout(layout)

    # Set Layer Color Widget
    @magicgui(
        auto_call=False,
        color_name={'label': 'Color Name'},
        label_value={'label': 'Label Value (optional)'},
        call_button='Set Layer Color'
    )
    def set_layer_color_widget(
        color_name: str = '',
        label_value: str = ''
    ):
        color_name_str = color_name.strip()
        label_value_str = label_value.strip()
        selected_layers = viewer.layers.selection
        if not selected_layers:
            print('No layer selected.')
            return

        if not color_name_str:
            print('Please enter a color name.')
            return

        # Try to interpret label_value_str as an integer
        try:
            label_value_int = int(label_value_str)
            specific_label = True
        except ValueError:
            specific_label = False

        for layer in selected_layers:
            if isinstance(layer, napari.layers.Labels):
                # Get the existing color mapping
                current_color_mapping = dict(layer.color)
                if specific_label:
                    # Update the color for the specific label
                    current_color_mapping[label_value_int] = color_name_str
                    # Set the new color mapping
                    layer.color = current_color_mapping
                    print(f'Set color of label {label_value_int} in layer "{layer.name}" to {color_name_str}')
                else:
                    # Change all non-zero labels to the specified color
                    unique_labels = np.unique(layer.data)
                    unique_labels = unique_labels[unique_labels != 0]  # Exclude background label
                    new_color_mapping = {label: color_name_str for label in unique_labels}
                    # Update the color mapping
                    layer.color = new_color_mapping
                    print(f'Set color of all labels in layer "{layer.name}" to {color_name_str}')
            elif hasattr(layer, 'colormap'):
                try:
                    # Set the colormap of the layer
                    layer.colormap = vispy.color.Colormap([[0, 0, 0], color_name_str])
                    print(f'Set colormap of layer "{layer.name}" to {color_name_str}')
                except Exception as e:
                    print(f'Error setting colormap for layer "{layer.name}": {e}')
            else:
                print(f'Layer "{layer.name}" does not support color changes.')

    # Flip X Widget
    @magicgui(call_button='Flip X')
    def flip_x_widget():
        selected_layers = viewer.layers.selection
        if not selected_layers:
            print('No layer selected.')
            return
        for layer in selected_layers:
            layer.data = np.fliplr(layer.data)
            print(f'Flipped layer "{layer.name}" on X axis.')

    # Flip Y Widget
    @magicgui(call_button='Flip Y')
    def flip_y_widget():
        selected_layers = viewer.layers.selection
        if not selected_layers:
            print('No layer selected.')
            return
        for layer in selected_layers:
            layer.data = np.flipud(layer.data)
            print(f'Flipped layer "{layer.name}" on Y axis.')

    # Resize Layers Widget
    @magicgui(
        auto_call=False,
        target_layer={
            'label': 'Layer to Resize To',
            'choices': get_layer_names,
            'nullable': True,
            'widget_type': 'ComboBox'
        },
        call_button='Resize Layers'
    )
    def resize_layers_widget(
        target_layer: str = None
    ):
        selected_layers = viewer.layers.selection
        if not selected_layers:
            print('No layers selected for resizing.')
            return
        if not target_layer:
            print('Please select a target layer for resizing.')
            return
        if target_layer not in viewer.layers:
            print(f'Layer "{target_layer}" not found.')
            return
        target_layer_data = viewer.layers[target_layer].data
        target_shape = target_layer_data.shape

        for layer in selected_layers:
            if layer.name == target_layer:
                continue  # Skip resizing the target layer itself
            # Resize the layer data to match the target shape
            resized_data = transform.resize(
                layer.data,
                target_shape,
                preserve_range=True,
                anti_aliasing=False,
                order=0
            ).astype(layer.data.dtype)
            layer.data = resized_data
            print(f'Resized layer "{layer.name}" to match "{target_layer}".')

    # Transfer Colormap Widget
    @magicgui(
        auto_call=False,
        source_layer={
            'label': 'Source Layer',
            'choices': get_layer_names,
            'nullable': True,
            'widget_type': 'ComboBox'
        },
        call_button='Transfer Colormap'
    )
    def transfer_colormap_widget(
        source_layer: str = None
    ):
        selected_layers = viewer.layers.selection
        if not selected_layers:
            print('No layers selected for colormap transfer.')
            return
        if not source_layer:
            print('Please select a source layer for colormap transfer.')
            return
        if source_layer not in viewer.layers:
            print(f'Layer "{source_layer}" not found.')
            return
        source_layer_obj = viewer.layers[source_layer]
        for layer in selected_layers:
            if layer.name == source_layer:
                continue  # Skip transferring colormap to itself
            if hasattr(layer, 'colormap') and hasattr(source_layer_obj, 'colormap'):
                layer.colormap = source_layer_obj.colormap
                print(f'Transferred colormap from "{source_layer}" to "{layer.name}".')
            elif isinstance(layer, napari.layers.Labels) and isinstance(source_layer_obj, napari.layers.Labels):
                layer.color = dict(source_layer_obj.color)
                print(f'Transferred label colors from "{source_layer}" to "{layer.name}".')
            else:
                print(f'Cannot transfer colormap from "{source_layer}" to "{layer.name}".')

    # Expand Labels Widget
    @magicgui(
        auto_call=False,
        expand_pixels={
            'label': 'Expand Pixels',
            'widget_type': 'SpinBox',
            'min': 0,
            'max': 1000,
            'step': 1,
        },
        call_button='Expand Labels'
    )
    def expand_labels_widget(
        expand_pixels: int = 100
    ):
        selected_layers = viewer.layers.selection
        if not selected_layers:
            print('No labels layer selected for expansion.')
            return
        for layer in selected_layers:
            if isinstance(layer, napari.layers.Labels):
                expanded_data = segmentation.expand_labels(layer.data, distance=expand_pixels)
                # Create a new layer with the expanded labels
                new_layer = viewer.add_labels(
                    expanded_data,
                    name=f'{layer.name}_expanded',
                    colormap=DirectLabelColormap(color_dict=dict(layer.color))
                )
                # Copy layer properties
                new_layer.blending = layer.blending
                new_layer.opacity = layer.opacity
                new_layer.visible = layer.visible
                print(f'Created expanded labels layer "{new_layer.name}" from "{layer.name}".')
            else:
                print(f'Layer "{layer.name}" is not a Labels layer.')

    # Mask Layer Widget
    @magicgui(
        auto_call=False,
        layer_to_mask={
            'label': 'Layer to Mask',
            'choices': get_layer_names,
            'nullable': True,
            'widget_type': 'ComboBox'
        },
        call_button='Mask Layer'
    )
    def mask_layer_widget(
        layer_to_mask: str = None
    ):
        if not layer_to_mask:
            print('Please select a layer to mask.')
            return
        # Get the layer to mask
        if layer_to_mask in viewer.layers:
            all_cells_layer = viewer.layers[layer_to_mask]
        else:
            print(f"Layer '{layer_to_mask}' not found.")
            return
        # If selected_layer_names is not provided, use the currently selected layers
        selected_layers = [layer for layer in viewer.layers.selection if layer.name != layer_to_mask]
        if not selected_layers:
            print("No layers are currently selected for masking. Please select layers.")
            return
        # Get the data for all selected layers and sum them
        layer_data_sum = np.sum([layer.data for layer in selected_layers], axis=0)
        
        # Mask the 'all_cells' layer with the selected layers' data
        masked_data = np.where(layer_data_sum > 0, all_cells_layer.data, 0)
        # Copy the properties from the original layer for the new layer
        new_layer_name = layer_to_mask + '_mask'
        new_layer = viewer.add_labels(
            masked_data,
            name=new_layer_name,
            scale=all_cells_layer.scale,
            translate=all_cells_layer.translate,
            opacity=all_cells_layer.opacity,
            blending=all_cells_layer.blending,
            visible=all_cells_layer.visible,
            colormap=all_cells_layer.colormap  # Copy the colormap from the original layer
        )
        # Set the contour property after creating the new layer
        new_layer.contour = all_cells_layer.contour
        print(f"Created a new masked layer '{new_layer_name}' based on '{layer_to_mask}' with contour '{new_layer.contour}'.")

    # Save Workspace Widget
    save_workspace_folder = widgets.LineEdit(value='workspace')
    save_workspace_button = widgets.PushButton(text='Save Workspace')
    def save_workspace():
        folder_path = save_workspace_folder.value
        save_visible_layers_and_camera(viewer, folder_path)
    save_workspace_button.clicked.connect(save_workspace)

    # Load Workspace Widget
    load_workspace_folder = widgets.LineEdit(value='workspace')
    load_workspace_button = widgets.PushButton(text='Load Workspace')
    def load_workspace():
        folder_path = load_workspace_folder.value
        load_layers_and_camera_from_folder(viewer, folder_path)
    load_workspace_button.clicked.connect(load_workspace)

    # Add widgets to the Layer Management layout
    layout.addWidget(set_layer_color_widget.native)
    layout.addWidget(flip_x_widget.native)
    layout.addWidget(flip_y_widget.native)
    layout.addWidget(resize_layers_widget.native)
    layout.addWidget(transfer_colormap_widget.native)
    layout.addWidget(expand_labels_widget.native)
    layout.addWidget(mask_layer_widget.native)
    layout.addWidget(widgets.Label(value='Save Workspace Folder:').native)
    layout.addWidget(save_workspace_folder.native)
    layout.addWidget(save_workspace_button.native)
    layout.addWidget(widgets.Label(value='Load Workspace Folder:').native)
    layout.addWidget(load_workspace_folder.native)
    layout.addWidget(load_workspace_button.native)

    # --- Add 'Update Layer List' Button ---
    update_layer_list_button = widgets.PushButton(text='Update layer list')
    def update_layer_list():
        layer_names = get_layer_names()
        resize_layers_widget.target_layer.choices = layer_names
        transfer_colormap_widget.source_layer.choices = layer_names
        mask_layer_widget.layer_to_mask.choices = layer_names
        print('Layer list updated.')
    update_layer_list_button.clicked.connect(update_layer_list)
    layout.addWidget(update_layer_list_button.native)

    # Add the Layer Management widget to the viewer
    viewer.window.add_dock_widget(
        layer_management_widget,
        name='Layer management'
    )

    # Add the main control widget
    viewer.window.add_dock_widget(
        [
            add_roi_label.native,
            roi_selector.native,
            hide_all_layers_button.native,
            delete_all_layers_button.native,
            add_roi_images_button.native,
            add_masks_button.native
        ],
        name='Controls'
    )

    # Widgets for adjusting quantile normalization and minimum pixel counts
    quant_select_label = widgets.Label(value='Normalize intensity to quantile:')
    quant_select = widgets.FloatSpinBox(min=0, max=1, value=0.999, step=0.001)
    minimum_pixel_counts_select_label = widgets.Label(value='Minimum pixel value:')
    minimum_pixel_counts_select = widgets.FloatText(value=0.1, min=0)

    # Build dictionaries mapping image names to folders and extensions
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
        """
        GUI widget to select and add images.
        """
        _add_images_from_list(x)

    def _add_images_from_list(selected_images):
        """
        Add selected images to the viewer.
        """
        for image, colour in zip(selected_images, itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y'])):
            file = Path(image_folder_dict[image], roi_selector.value, f'{image}.{image_ext_dict[image]}')
            print(f'Loading image from: {file}')
            _load_imc_image(file, quantile=quant_select.value, minimum_pixel_counts=minimum_pixel_counts_select.value, colormap=vispy.color.Colormap([[0, 0, 0], colour]))

    # Add the image selector widget to the viewer
    viewer.window.add_dock_widget(
        [
            _image_selector,
            quant_select_label.native,
            quant_select.native,
            minimum_pixel_counts_select_label.native,
            minimum_pixel_counts_select.native
        ],
        name='Add raw images'
    )

    # Identify categorical observation columns
    categorical_obs_columns = [col for col in adata.obs.columns if adata.obs[col].dtype == 'category']
    
    @magicgui(x=dict(widget_type='Select', choices=categorical_obs_columns, label='Select categories'), call_button='Add as masks')
    def _obs_selector(x: list):
        """
        GUI widget to select and add categorical observations as masks.
        """
        _add_obs_masks(x)

    def _add_obs_masks(obs_list):
        """
        Add masks for selected categorical observations.
        """
        for obs in obs_list:
            _add_masks(
                roi_name=roi_selector.value,
                adata=adata,
                pop_obs=obs,
                roi_obs='ROI',
                adata_colormap=True,
                colour_map=colormaps['tab20'].colors,
                add_individual_pops=individual_pops_toggle.value
            )

    # Checkbox to toggle adding individual populations as separate masks
    individual_pops_toggle = widgets.CheckBox(value=False, text='Add individual groups from .obs as masks')

    # Add the categorical observations widget to the viewer
    viewer.window.add_dock_widget(
        [
            _obs_selector,
            individual_pops_toggle.native
        ],
        name='Categories as masks'
    )

    # Identify numerical observation columns
    numerical_obs_columns = [col for col in adata.obs.columns if adata.obs[col].dtype in ['float32', 'float64', 'int32']]

    @magicgui(x=dict(widget_type='Select', choices=adata.var_names.tolist() + numerical_obs_columns, label='Select numeric'), call_button='Add as overlays')
    def _quant_selector(x: list):
        """
        GUI widget to select and add numerical observations or variables as overlays.
        """
        _add_quant_masks(x)

    def _add_quant_masks(quant_list):
        """
        Add quantitative overlays for selected numerical observations or variables.
        """
        for quant in quant_list:
            _add_masks(
                roi_name=roi_selector.value,
                adata=adata,
                pop_obs=None,
                quant=quant,
                roi_obs='ROI',
                adata_colormap=False,
                add_individual_pops=individual_pops_toggle.value
            )

    # Add the numerical observations widget to the viewer
    viewer.window.add_dock_widget(_quant_selector, name='Numeric as masks')

    # --- Include Functions for Saving and Loading Workspace ---

    def save_serializable_layer(layer, filename):
        """
        Save a serializable version of a Napari layer to a pickle file.
        """
        # Function to serialize a colormap
        def serialize_colormap(colormap):
            if colormap is None:
                return None
            return {
                'colors': colormap.colors.tolist(),
                'name': colormap.name,
                'interpolation': colormap.interpolation.value,
                'controls': colormap.controls.tolist()
            }

        # Determine the type of the layer (e.g., 'Image', 'Points', 'Labels')
        layer_type = type(layer).__name__

        # Handle label layers differently (colormap as dictionary, contour attribute)
        if layer_type == 'Labels':
            colormap = layer.color  # For labels, color is stored as a dictionary
            contour = layer.contour  # Capture the contour attribute
        else:
            colormap = serialize_colormap(getattr(layer, 'colormap', None))
            contour = None  # No contour for non-label layers

        # Create a dictionary to store the layer attributes
        layer_data = {
            'name': layer.name,
            'type': layer_type,
            'data': layer.data if isinstance(layer.data, np.ndarray) else None,
            'scale': layer.scale,
            'translate': layer.translate,
            'opacity': layer.opacity,
            'blending': layer.blending,
            'visible': layer.visible,
            'colormap': colormap,  # Colormap or color dict for labels
            'contour': contour  # Add contour for labels
        }
        
        # Save the dictionary to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(layer_data, file)
        
        print(f"Layer '{layer.name}' saved to {filename}")

    def load_serializable_layer(filename):
        """
        Load a serializable version of a Napari layer from a pickle file.
        """
        # Load the serialized layer data from the pickle file
        with open(filename, 'rb') as file:
            layer_data = pickle.load(file)
        
        # Function to recreate colormap for non-label layers
        def recreate_colormap(colormap_data):
            if colormap_data is None:
                return None
            return Colormap(
                colors=np.array(colormap_data['colors']),
                name=colormap_data['name'],
                interpolation=colormap_data['interpolation'],
                controls=np.array(colormap_data['controls'])
            )
        
        # Recreate the colormap only if it's not a Labels layer
        if layer_data['type'] != 'Labels' and layer_data['colormap']:
            layer_data['colormap'] = recreate_colormap(layer_data['colormap'])
        
        return layer_data

    def add_layer_to_napari(viewer, layer_data):
        """
        Add a layer to the Napari viewer using the serialized layer data.
        """
        layer_type = layer_data['type']
        
        if layer_type == 'Image':
            viewer.add_image(
                layer_data['data'],
                name=layer_data['name'],
                scale=layer_data['scale'],
                translate=layer_data['translate'],
                opacity=layer_data['opacity'],
                blending=layer_data['blending'],
                visible=layer_data['visible'],
                colormap=layer_data.get('colormap', None),
                contrast_limits=layer_data.get('contrast_limits', None)
            )
            print(f"Added Image layer '{layer_data['name']}' to viewer.")
        
        elif layer_type == 'Points':
            viewer.add_points(
                layer_data['data'],
                name=layer_data['name'],
                scale=layer_data['scale'],
                translate=layer_data['translate'],
                opacity=layer_data['opacity'],
                blending=layer_data['blending'],
                visible=layer_data['visible']
            )
            print(f"Added Points layer '{layer_data['name']}' to viewer.")
        
        elif layer_type == 'Labels':
            labels_layer = viewer.add_labels(
                layer_data['data'],
                name=layer_data['name'],
                scale=layer_data['scale'],
                translate=layer_data['translate'],
                opacity=layer_data['opacity'],
                blending=layer_data['blending'],
                visible=layer_data['visible'],
                colormap=layer_data.get('colormap', None)  # Apply colormap for labels as dictionary
            )
            # Set contour after adding the layer
            labels_layer.contour = layer_data.get('contour', 0)  # Default to 0 if missing
            print(f"Added Labels layer '{layer_data['name']}' with contour '{labels_layer.contour}' to viewer.")
        
        else:
            print(f"Layer type '{layer_type}' is not currently supported for reconstruction.")

    def save_visible_layers_and_camera(viewer, folder_path):
        """
        Save all visible layers and the current camera settings in the Napari viewer to files.
        """
        os.makedirs(folder_path, exist_ok=True)
        
        # Save each visible layer to a separate pickle file
        for layer in viewer.layers:
            if layer.visible:
                file_path = os.path.join(folder_path, f"{layer.name}.pickle")
                save_serializable_layer(layer, file_path)
        
        # Save the current camera settings in a CSV file
        camera_settings = {
            'position': viewer.camera.center,
            'zoom': viewer.camera.zoom,
            'angles': viewer.camera.angles
        }
        
        camera_df = pd.DataFrame([camera_settings])
        camera_df.to_csv(os.path.join(folder_path, 'camera_settings.csv'), index=False)
        
        print(f"All visible layers and camera settings have been saved in '{folder_path}'.")

    def load_layers_and_camera_from_folder(viewer, folder_path):
        """
        Load all layers and camera settings from a folder and add them to the Napari viewer.
        """
        # Load each layer pickle file from the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pickle'):
                layer_path = os.path.join(folder_path, file_name)
                loaded_layer_data = load_serializable_layer(layer_path)
                add_layer_to_napari(viewer, loaded_layer_data)
        
        # Load and apply camera settings from the CSV file
        camera_settings_path = os.path.join(folder_path, 'camera_settings.csv')
        if os.path.exists(camera_settings_path):
            camera_df = pd.read_csv(camera_settings_path)
            if not camera_df.empty:
                camera_settings = camera_df.iloc[0]
                viewer.camera.center = eval(camera_settings['position'])
                viewer.camera.zoom = camera_settings['zoom']
                viewer.camera.angles = eval(camera_settings['angles'])
                print("Camera settings have been restored.")
        else:
            print("Camera settings file not found.")
        
        print(f"Layers and camera settings have been loaded from '{folder_path}'.")

    # Start the Napari event loop
    napari.run()
    return viewer
