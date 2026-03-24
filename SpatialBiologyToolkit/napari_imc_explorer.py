# Standard Library Imports
import itertools
import os
import pickle  # For saving and loading
import re
from pathlib import Path
from types import SimpleNamespace

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
from matplotlib.colors import to_hex
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QColorDialog,
    QDockWidget,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from napari.utils.colormaps import Colormap  # For colormap reconstruction
from napari.utils import DirectLabelColormap

def napari_imc_explorer(
    masks_folder: str = 'Masks',
    image_folders: list = ['Images'],
    annotations_folder: str = 'Annotations',
    roi_obs: str = 'ROI',
    cell_id_in_mask_obs: str = 'ObjectNumber',
    adata: ad.AnnData = ad.AnnData(),
    check_masks: bool = True,
    mask_extension: str = None,
    initial_roi_count: int = 10,
    randomize_initial_rois: bool = False,
    initial_roi_random_seed: int = 0,
) -> napari.Viewer:
    """
    Start an interactive Napari viewer for exploring IMC data.

    Parameters
    ----------
    masks_folder : str
        Directory containing a mask for each ROI, each file named after the ROI. Masks should be uint16 image files.
    image_folders : list
        Directories containing subdirectories, each named after ROIs in the AnnData. Images are named after channels (`adata.var_names`), uint16 image files.
    annotations_folder : str
        Directory containing manual annotation subfolders. Each annotation subfolder contains a label mapping CSV and one TIFF label image per ROI.
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
    initial_roi_count : int, optional
        Number of ROI buttons to show in the Population QC ROI shortcut list. If `None`, all matching ROIs are shown.
    randomize_initial_rois : bool
        If True, randomly choose/order the Population QC ROI shortcut list instead of ordering by abundance.
    initial_roi_random_seed : int
        Seed used when `randomize_initial_rois=True`.

    Returns
    -------
    napari.Viewer
        The Napari viewer object.
    """
    # Ensure image_folders is a list
    if not isinstance(image_folders, list):
        image_folders = [image_folders]

    if initial_roi_count is not None:
        initial_roi_count = int(initial_roi_count)
        if initial_roi_count <= 0:
            raise ValueError('initial_roi_count must be > 0 or None.')

    population_qc_roi_button_limit = initial_roi_count

    annotations_folder = Path(annotations_folder)
    annotations_folder.mkdir(parents=True, exist_ok=True)
        
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

    def _select_population_qc_roi_choices(ordered_rois):
        """
        Choose which ROI shortcuts appear in the Population QC panel.
        """
        ordered_rois = [str(roi_name) for roi_name in ordered_rois]
        if not ordered_rois:
            return []

        if randomize_initial_rois:
            rng = np.random.default_rng(initial_roi_random_seed)
            selected_rois = rng.permutation(ordered_rois).tolist()
        else:
            selected_rois = list(ordered_rois)

        if population_qc_roi_button_limit is not None:
            selected_rois = selected_rois[:population_qc_roi_button_limit]

        return selected_rois

    annotation_mapping_filename = 'label_mapping.csv'
    annotation_background_label = 'Unlabelled'
    annotation_default_colours = [to_hex(colour) for colour in colormaps['tab20'].colors]

    def _annotation_dir(annotation_name):
        return annotations_folder / str(annotation_name)

    def _annotation_mapping_path(annotation_name):
        return _annotation_dir(annotation_name) / annotation_mapping_filename

    def _annotation_roi_path(annotation_name, roi_name):
        return _annotation_dir(annotation_name) / f'{roi_name}.tiff'

    def _list_annotation_names():
        return sorted(
            path.name for path in annotations_folder.iterdir()
            if path.is_dir()
        )

    def _default_annotation_column(annotation_name):
        return f'annots_{annotation_name}'

    def _default_annotation_colour(index):
        return annotation_default_colours[index % len(annotation_default_colours)]

    def _normalise_annotation_colour(colour_value):
        colour = QColor(str(colour_value))
        return colour.name() if colour.isValid() else str(colour_value)

    def _load_annotation_mapping(annotation_name):
        mapping_path = _annotation_mapping_path(annotation_name)
        if not mapping_path.exists():
            raise FileNotFoundError(f'Annotation mapping not found: {mapping_path}')

        mapping_df = pd.read_csv(mapping_path)
        required_columns = {'value', 'label', 'color'}
        if not required_columns.issubset(mapping_df.columns):
            raise ValueError(
                f"Annotation mapping '{mapping_path}' must contain columns: {sorted(required_columns)}"
            )

        mapping_df = mapping_df.loc[:, ['value', 'label', 'color']].copy()
        mapping_df['value'] = mapping_df['value'].astype(int)
        mapping_df['label'] = mapping_df['label'].astype(str)
        mapping_df['color'] = mapping_df['color'].astype(str)
        mapping_df['color'] = mapping_df['color'].map(_normalise_annotation_colour)
        mapping_df = mapping_df.drop_duplicates(subset='value', keep='last').sort_values('value')

        if 0 not in mapping_df['value'].values:
            mapping_df = pd.concat(
                [
                    pd.DataFrame(
                        [{'value': 0, 'label': annotation_background_label, 'color': 'transparent'}]
                    ),
                    mapping_df,
                ],
                ignore_index=True,
            )
        else:
            mapping_df.loc[mapping_df['value'] == 0, 'label'] = annotation_background_label
            mapping_df.loc[mapping_df['value'] == 0, 'color'] = 'transparent'

        return mapping_df.sort_values('value').reset_index(drop=True)

    def _write_annotation_mapping(annotation_name, mapping_df):
        annotation_dir = _annotation_dir(annotation_name)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        mapping_df.loc[:, ['value', 'label', 'color']].sort_values('value').to_csv(
            _annotation_mapping_path(annotation_name),
            index=False,
        )

    def _blank_annotation_array(roi_name):
        mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
        return np.zeros(mask.shape, dtype='uint16')

    def _save_annotation_array(annotation_name, roi_name, annotation_array):
        annotation_array = np.asarray(annotation_array)
        if annotation_array.size > 0 and annotation_array.max() > np.iinfo(np.uint16).max:
            raise ValueError('Annotation label values must be <= 65535 to save as uint16 TIFF.')

        annotation_array = annotation_array.astype('uint16', copy=False)
        expected_shape = _blank_annotation_array(roi_name).shape
        if annotation_array.shape != expected_shape:
            raise ValueError(
                f"Annotation for ROI '{roi_name}' has shape {annotation_array.shape}, expected {expected_shape}."
            )

        annotation_dir = _annotation_dir(annotation_name)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        io.imsave(
            _annotation_roi_path(annotation_name, roi_name),
            annotation_array,
            check_contrast=False,
        )

    def _load_annotation_array(annotation_name, roi_name):
        annotation_path = _annotation_roi_path(annotation_name, roi_name)
        if annotation_path.exists():
            return sk.io.imread(annotation_path).astype('uint16', copy=False)

        blank_annotation = _blank_annotation_array(roi_name)
        _save_annotation_array(annotation_name, roi_name, blank_annotation)
        return blank_annotation

    def _build_annotation_colormap(mapping_df):
        color_dict = {None: 'transparent', 0: 'transparent'}
        for _, row in mapping_df.iterrows():
            value = int(row['value'])
            if value == 0:
                continue
            color_dict[value] = row['color']
        return DirectLabelColormap(color_dict=color_dict)

    def _dominant_annotation_value(values):
        values = np.asarray(values, dtype=int)
        if values.size == 0:
            return 0

        non_zero_values = values[values != 0]
        if non_zero_values.size == 0:
            return 0

        unique_values, counts = np.unique(non_zero_values, return_counts=True)
        return int(unique_values[np.argmax(counts)])

    def _load_imc_image(file, quantile=0.999, colormap=None, recolour_image=False, minimum_pixel_counts=0.1, layer_name=None):
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
        image_name = layer_name or os.path.splitext(os.path.basename(file))[0]
        # Add image to the viewer
        viewer.add_image(image, name=image_name, blending='additive', colormap=colormap)

    def _add_roi_images_raw(roi_name, quantile=0.999, colour_map=['r', 'g', 'b', 'c', 'm', 'y'], minimum_pixel_counts=0.1, recolour_image=False):
        """
        Add all images from one ROI folder, cycling through specified colours.
        """
        roi_image_map = _build_roi_image_map(roi_name)
        if not roi_image_map:
            print(f'No images found for ROI "{roi_name}".')
            return

        ordered_logical_names = [image_name for image_name in im_list if image_name in roi_image_map]
        if not ordered_logical_names:
            ordered_logical_names = list(roi_image_map.keys())

        for logical_name, colour in zip(ordered_logical_names, itertools.cycle(colour_map)):
            _load_imc_image(
                roi_image_map[logical_name],
                quantile=quant_select.value,
                colormap=vispy.color.Colormap([[0, 0, 0], colour]),
                recolour_image=recolour_image,
                minimum_pixel_counts=minimum_pixel_counts_select.value,
                layer_name=logical_name,
            )
            viewer.layers[-1].visible = False  # Hide the layer by default

    def _population_values(pop_obs_name):
        """
        Return ordered population labels as strings for a categorical obs column.
        """
        if pop_obs_name not in adata.obs.columns:
            return []

        obs_series = adata.obs[pop_obs_name]
        if isinstance(obs_series.dtype, pd.CategoricalDtype):
            values = obs_series.cat.categories.tolist()
        else:
            values = obs_series.dropna().unique().tolist()

        return [str(value) for value in values if not pd.isna(value)]

    def _add_masks(
        roi_name,
        adata,
        pop_obs=None,
        quant=None,
        roi_obs='ROI',
        adata_colormap=True,
        colour_map=colormaps['tab20'].colors,
        add_individual_pops=False,
        selected_populations=None,
        add_combined_mask=True,
        individual_layer_name_prefix='',
        individual_layers_visible=False,
        add_base_mask=True,
    ):
        """
        Add masks to the viewer, optionally with population or quantitative overlays.
        """
        # Load the mask image
        mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
        # Add the base cell mask layer if not already added
        if add_base_mask and 'all_cells' not in [layer.name for layer in viewer.layers]:
            viewer.add_labels(mask, name='all_cells')
            viewer.layers[-1].contour = 1
            viewer.layers[-1].visible = False

        # Get the observation data for the current ROI
        adata_roi_obs = adata.obs.loc[adata.obs[roi_obs] == roi_name, :].copy()
        adata_roi_obs.reset_index(drop=True, inplace=True)

        if pop_obs:
            populations = _population_values(pop_obs)
            if not populations:
                print(f'No populations available for "{pop_obs}".')
                return

            population_to_code = {population: index + 1 for index, population in enumerate(populations)}
            if selected_populations is None:
                populations_to_add = populations
            else:
                requested_populations = {str(population) for population in selected_populations}
                populations_to_add = [population for population in populations if population in requested_populations]

            if not populations_to_add:
                print(f'No requested populations were found in "{pop_obs}".')
                return

            # Use the colormap from adata if available
            if adata_colormap and (f'{pop_obs}_colors' in adata.uns):
                colour_map = adata.uns[f'{pop_obs}_colors']
            pop_colormap = {(x + 1): y for x, y in enumerate(colour_map)}
            pop_colormap.update({None:'magenta'})
            all_pops_mask = np.zeros(mask.shape, dtype='uint16') if add_combined_mask else None

            # Create a mask for each population
            for pop in populations_to_add:
                pop_num = population_to_code[pop]
                try:
                    if cell_id_in_mask_obs:
                        objects = adata_roi_obs.loc[
                            adata_roi_obs[pop_obs].astype(str) == pop,
                            cell_id_in_mask_obs,
                        ].to_numpy()
                    else:
                        objects = adata_roi_obs.loc[
                            adata_roi_obs[pop_obs].astype(str) == pop,
                            :,
                        ].index.to_numpy() + 1

                    pop_mask = np.isin(mask, objects)
                    if add_combined_mask:
                        all_pops_mask = np.where(pop_mask, pop_num, all_pops_mask)
                    if add_individual_pops:
                        layer_name = f'{individual_layer_name_prefix}{pop}' if individual_layer_name_prefix else pop
                        viewer.add_labels(
                            pop_mask.astype('uint8'),
                            name=layer_name,
                            colormap=DirectLabelColormap(color_dict={None: 'magenta', 1: pop_colormap[pop_num]}),
                        )
                        viewer.layers[-1].contour = 1
                        viewer.layers[-1].visible = individual_layers_visible
                except Exception as e:
                    print(f'Error adding group {pop} from {pop_obs}: {e}')
            if add_combined_mask:
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

    def _move_selected_layers_to_top():
        """
        Move the currently selected layers to the top of the Napari layer list.
        """
        selected_layers = [layer for layer in list(viewer.layers) if layer in viewer.layers.selection]
        if not selected_layers:
            print('No layers selected.')
            return

        try:
            for layer in selected_layers:
                viewer.layers.move(viewer.layers.index(layer), len(viewer.layers) - 1)
            print(f'Moved {len(selected_layers)} selected layer(s) to the top.')
        except Exception as exc:
            print(f'Could not reorder layers: {exc}')

    layers_to_top_button = widgets.PushButton(text='Layers to top', name='layers_to_top_button')
    layers_to_top_button.clicked.connect(_move_selected_layers_to_top)

    def _add_roi_images():
        """
        Add all images for the selected ROI.
        """
        selected_item = roi_selector.value
        _add_roi_images_raw(selected_item, quantile=quant_select.value, minimum_pixel_counts=minimum_pixel_counts_select.value)

    # List of all available ROIs
    all_roi_list = _list_folders_in_directory(image_folders[0])
    all_roi_set = set(all_roi_list)
    roi_list = list(all_roi_list)

    # Selector widget for ROIs
    roi_selector = widgets.ComboBox(label='Select ROI:', choices=roi_list)
    roi_selector_state = SimpleNamespace(choices=list(roi_list))

    def _get_roi_selector_choices():
        try:
            current_choices = [str(choice) for choice in roi_selector.choices]
            roi_selector_state.choices = list(current_choices)
            return current_choices
        except Exception:
            return list(roi_selector_state.choices)

    def _set_roi_selector_choices(choices, selected_roi=None):
        unique_choices = list(dict.fromkeys(str(choice) for choice in choices if str(choice)))
        if not unique_choices:
            return
        roi_selector_state.choices = list(unique_choices)
        try:
            roi_selector.choices = unique_choices
        except Exception:
            pass
        if selected_roi is not None and str(selected_roi) in unique_choices:
            roi_selector.value = str(selected_roi)

    def _ensure_roi_in_selector(roi_name):
        roi_name = str(roi_name)
        if roi_name not in all_roi_set:
            return False
        current_choices = _get_roi_selector_choices()
        if roi_name not in current_choices:
            _set_roi_selector_choices(current_choices + [roi_name], selected_roi=roi_name)
        return True

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
        _add_masks(selected_item, adata, pop_obs=None, roi_obs=roi_obs)

    # Button to add cell masks
    add_masks_button = widgets.PushButton(text='Add cell mask', name='add_masks_button')
    add_masks_button.clicked.connect(_add_mask_labels)

    # Check masks if required
    if check_masks:
        _check_all_masks(adata, roi_obs=roi_obs)
    
    # Create the Napari viewer
    viewer = napari.Viewer()

    # Avoid attaching custom attributes directly to viewer (raises ValueError in napari)

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
    def update_layer_list(silent=False):
        layer_names = get_layer_names()
        resize_layers_widget.target_layer.choices = layer_names
        transfer_colormap_widget.source_layer.choices = layer_names
        mask_layer_widget.layer_to_mask.choices = layer_names
        if not silent:
            print('Layer list updated.')
    update_layer_list_button.clicked.connect(update_layer_list)
    layout.addWidget(update_layer_list_button.native)

    # Widgets for adjusting quantile normalization and minimum pixel counts
    quant_select_label = widgets.Label(value='Normalize intensity to quantile:')
    quant_select = widgets.FloatSpinBox(min=0, max=1, value=0.999, step=0.001)
    minimum_pixel_counts_select_label = widgets.Label(value='Minimum pixel value:')
    minimum_pixel_counts_select = widgets.FloatText(value=0.1, min=0)

    def _normalise_panel_text(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        text = str(value).strip()
        return text or None

    def _clean_panel_label(value):
        text = _normalise_panel_text(value)
        if text is None:
            return None
        return re.sub(r'\W+', '', text)

    def _register_panel_alias(alias_map, ambiguous_aliases, alias_value, logical_name):
        alias_text = _normalise_panel_text(alias_value)
        if alias_text is None:
            return

        alias_key = alias_text.lower()
        existing_name = alias_map.get(alias_key)
        if existing_name is None:
            alias_map[alias_key] = logical_name
        elif existing_name != logical_name:
            ambiguous_aliases.add(alias_key)

    def _build_image_alias_map():
        alias_map = {}
        ambiguous_aliases = set()
        channel_name_values = (
            adata.var['channel_name'].tolist()
            if 'channel_name' in adata.var.columns
            else [None] * adata.n_vars
        )
        channel_label_values = (
            adata.var['channel_label'].tolist()
            if 'channel_label' in adata.var.columns
            else [None] * adata.n_vars
        )

        for logical_name, channel_name, channel_label in zip(
            [str(var_name) for var_name in adata.var_names.tolist()],
            channel_name_values,
            channel_label_values,
        ):
            clean_logical_name = _clean_panel_label(logical_name)
            clean_channel_label = _clean_panel_label(channel_label)
            channel_name_text = _normalise_panel_text(channel_name)
            channel_label_text = _normalise_panel_text(channel_label)

            for alias_value in [logical_name, clean_logical_name, channel_label_text, clean_channel_label, channel_name_text]:
                _register_panel_alias(alias_map, ambiguous_aliases, alias_value, logical_name)

            if channel_name_text and channel_label_text:
                _register_panel_alias(
                    alias_map,
                    ambiguous_aliases,
                    f'{channel_name_text}_{channel_label_text}',
                    logical_name,
                )
            if channel_name_text and clean_channel_label:
                _register_panel_alias(
                    alias_map,
                    ambiguous_aliases,
                    f'{channel_name_text}_{clean_channel_label}',
                    logical_name,
                )
            if channel_name_text and clean_logical_name:
                _register_panel_alias(
                    alias_map,
                    ambiguous_aliases,
                    f'{channel_name_text}_{clean_logical_name}',
                    logical_name,
                )

        for alias_key in ambiguous_aliases:
            alias_map.pop(alias_key, None)
        return alias_map

    def _parse_imc_image_path(image_path):
        image_path = Path(image_path)
        stem = image_path.stem
        parts = stem.split('_')
        if len(parts) >= 4:
            channel_name = parts[2]
            channel_label = '_'.join(parts[3:])
        else:
            channel_name = None
            channel_label = stem
        return {
            'path': image_path,
            'stem': stem,
            'channel_name': channel_name,
            'channel_label': channel_label,
        }

    image_alias_map = _build_image_alias_map()
    roi_image_map_cache = {}

    def _resolve_logical_image_name(image_path):
        image_info = _parse_imc_image_path(image_path)
        candidate_aliases = [
            image_info['stem'],
            image_info['channel_label'],
            _clean_panel_label(image_info['channel_label']),
            image_info['channel_name'],
        ]
        if image_info['channel_name'] and image_info['channel_label']:
            candidate_aliases.append(f"{image_info['channel_name']}_{image_info['channel_label']}")
        if image_info['channel_name'] and _clean_panel_label(image_info['channel_label']):
            candidate_aliases.append(f"{image_info['channel_name']}_{_clean_panel_label(image_info['channel_label'])}")

        for candidate_alias in candidate_aliases:
            alias_text = _normalise_panel_text(candidate_alias)
            if alias_text is None:
                continue
            logical_name = image_alias_map.get(alias_text.lower())
            if logical_name is not None:
                return logical_name

        fallback_name = _normalise_panel_text(image_info['channel_label'])
        return fallback_name or image_info['stem']

    def _build_roi_image_map(roi_name):
        roi_name = str(roi_name)
        if roi_name in roi_image_map_cache:
            return roi_image_map_cache[roi_name]

        roi_image_map = {}
        for folder in image_folders:
            roi_folder = Path(folder, roi_name)
            if not roi_folder.exists():
                continue

            for image_path in sorted(roi_folder.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in ['.tif', '.tiff']:
                    continue
                logical_name = _resolve_logical_image_name(image_path)
                roi_image_map.setdefault(logical_name, image_path)

        roi_image_map_cache[roi_name] = roi_image_map
        return roi_image_map

    def _discover_available_logical_images():
        discovered_names = []
        discovered_set = set()

        for folder in image_folders:
            for roi_name in all_roi_list:
                roi_folder = Path(folder, roi_name)
                if not roi_folder.exists():
                    continue
                roi_image_map = _build_roi_image_map(roi_name)
                if not roi_image_map:
                    continue
                for logical_name in roi_image_map.keys():
                    if logical_name not in discovered_set:
                        discovered_names.append(logical_name)
                        discovered_set.add(logical_name)
                break

        ordered_adata_names = [
            str(var_name) for var_name in adata.var_names.tolist()
            if str(var_name) in discovered_set
        ]
        remaining_names = [name for name in discovered_names if name not in ordered_adata_names]
        return ordered_adata_names + remaining_names

    im_list = _discover_available_logical_images()
    
    @magicgui(x=dict(widget_type='Select', choices=im_list, label='Select images'), call_button='Add images')
    def _image_selector(x: list):
        """
        GUI widget to select and add images.
        """
        _add_images_from_list(x)

    # Capture image selector widget for external access
    image_select_widget = _image_selector
    get_selected_images = lambda: _image_selector.x.value

    def _add_images_from_list(selected_images):
        """
        Add selected images to the viewer.
        """
        if selected_images is None:
            print('No images selected.')
            return

        if isinstance(selected_images, str):
            selected_images = [selected_images]
        else:
            selected_images = [str(image) for image in selected_images]

        selected_images = [image for image in selected_images if image]
        if not selected_images:
            print('No images selected.')
            return

        roi_image_map = _build_roi_image_map(roi_selector.value)
        if not roi_image_map:
            print(f'No images found for ROI "{roi_selector.value}".')
            return

        loaded_images = 0
        for image, colour in zip(selected_images, itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y'])):
            file = roi_image_map.get(image)
            if file is None:
                print(
                    f'Could not find image "{image}" for ROI "{roi_selector.value}" in: '
                    f'{", ".join(str(Path(folder)) for folder in image_folders)}'
                )
                continue

            print(f'Loading image from: {file}')
            try:
                _load_imc_image(
                    file,
                    quantile=quant_select.value,
                    minimum_pixel_counts=minimum_pixel_counts_select.value,
                    colormap=vispy.color.Colormap([[0, 0, 0], colour]),
                    layer_name=image,
                )
                loaded_images += 1
            except Exception as exc:
                print(f'Could not load image "{image}" from "{file}": {exc}')

        if loaded_images == 0:
            print('No images were loaded.')

    # Identify categorical observation columns
    categorical_obs_columns = [
        col for col in adata.obs.columns
        if isinstance(adata.obs[col].dtype, pd.CategoricalDtype)
    ]
    
    @magicgui(x=dict(widget_type='Select', choices=categorical_obs_columns, label='Select categories'), call_button='Add as masks')
    def _obs_selector(x: list):
        """
        GUI widget to select and add categorical observations as masks.
        """
        _add_obs_masks(x)

    # Capture categorical obs selector for external access
    obs_select_widget = _obs_selector
    get_selected_obs_categories = lambda: _obs_selector.x.value

    def _add_obs_masks(obs_list):
        """
        Add masks for selected categorical observations.
        """
        for obs in obs_list:
            _add_masks(
                roi_name=roi_selector.value,
                adata=adata,
                pop_obs=obs,
                roi_obs=roi_obs,
                adata_colormap=True,
                colour_map=colormaps['tab20'].colors,
                add_individual_pops=individual_pops_toggle.value
            )

    # Checkbox to toggle adding individual populations as separate masks
    individual_pops_toggle = widgets.CheckBox(value=False, text='Add individual groups from .obs as masks')

    # Identify numerical observation columns
    numerical_obs_columns = [col for col in adata.obs.columns if adata.obs[col].dtype in ['float32', 'float64', 'int32']]

    @magicgui(x=dict(widget_type='Select', choices=adata.var_names.tolist() + numerical_obs_columns, label='Select numeric'), call_button='Add as overlays')
    def _quant_selector(x: list):
        """
        GUI widget to select and add numerical observations or variables as overlays.
        """
        _add_quant_masks(x)

    # Capture numeric selector for external access
    quant_select_widget = _quant_selector
    get_selected_numeric = lambda: _quant_selector.x.value

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
                roi_obs=roi_obs,
                adata_colormap=False,
                add_individual_pops=individual_pops_toggle.value
            )

    annotation_state = SimpleNamespace(active_annotation=None, active_roi=None)
    annotation_creator_rows = []

    def _annotation_layer_name(annotation_name):
        return f'annotation::{annotation_name}'

    def _get_active_annotation_layer():
        for layer in reversed(list(viewer.layers)):
            if getattr(layer, 'metadata', {}).get('manual_annotation_layer'):
                return layer
        return None

    def _set_annotation_status(message):
        annotation_status_label.setText(message)
        print(message)

    def _set_colour_button_style(button, colour_value):
        colour = QColor(_normalise_annotation_colour(colour_value))
        if not colour.isValid():
            colour = QColor('#808080')
        text_colour = '#000000' if colour.lightness() > 127 else '#ffffff'
        button.setText(colour.name())
        button.setStyleSheet(
            f'background-color: {colour.name()}; color: {text_colour}; border: 1px solid #666666;'
        )
        button.setProperty('annotation_colour', colour.name())

    def _choose_annotation_colour(button):
        current_colour = QColor(button.property('annotation_colour') or '#808080')
        selected_colour = QColorDialog.getColor(current_colour, annotation_widget, 'Select annotation colour')
        if selected_colour.isValid():
            _set_colour_button_style(button, selected_colour.name())

    def _clear_qt_layout(layout_to_clear):
        while layout_to_clear.count():
            item = layout_to_clear.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child_layout is not None:
                _clear_qt_layout(child_layout)

    def _populate_annotation_mapping_table(mapping_df):
        annotation_mapping_table.setRowCount(len(mapping_df))
        for row_index, row in enumerate(mapping_df.itertuples(index=False)):
            value_item = QTableWidgetItem(str(row.value))
            label_item = QTableWidgetItem(str(row.label))
            color_item = QTableWidgetItem(str(row.color))
            color_value = QColor(str(row.color))
            if color_value.isValid() and str(row.color).lower() != 'transparent':
                color_item.setBackground(color_value)
                text_colour = QColor('#000000') if color_value.lightness() > 127 else QColor('#ffffff')
                color_item.setForeground(text_colour)
            annotation_mapping_table.setItem(row_index, 0, value_item)
            annotation_mapping_table.setItem(row_index, 1, label_item)
            annotation_mapping_table.setItem(row_index, 2, color_item)
        annotation_mapping_table.resizeRowsToContents()

    def _get_selected_annotation_name():
        annotation_name = annotation_selector_combo.currentText().strip()
        return annotation_name or None

    def _update_annotation_summary():
        selected_annotation = _get_selected_annotation_name()
        if not selected_annotation:
            selected_annotation_label.setText('Selected annotation: none')
            if annotation_state.active_annotation and annotation_state.active_roi:
                loaded_annotation_label.setText(
                    f'Loaded in viewer: {annotation_state.active_annotation} ({annotation_state.active_roi})'
                )
            else:
                loaded_annotation_label.setText('Loaded in viewer: none')
            annotation_mapping_table.setRowCount(0)
            sync_annotation_column_input.setText('')
            return

        selected_annotation_label.setText(f'Selected annotation: {selected_annotation}')
        sync_annotation_column_input.setText(_default_annotation_column(selected_annotation))

        try:
            mapping_df = _load_annotation_mapping(selected_annotation)
        except Exception as exc:
            annotation_mapping_table.setRowCount(0)
            loaded_annotation_label.setText('Loaded in viewer: none')
            _set_annotation_status(f'Could not read mapping for "{selected_annotation}": {exc}')
            return

        _populate_annotation_mapping_table(mapping_df)
        if annotation_state.active_annotation and annotation_state.active_roi:
            loaded_annotation_label.setText(
                f'Loaded in viewer: {annotation_state.active_annotation} ({annotation_state.active_roi})'
            )
        else:
            loaded_annotation_label.setText('Loaded in viewer: none')

    def _refresh_annotation_choices(selected_annotation=None):
        annotation_names = _list_annotation_names()
        annotation_selector_combo.blockSignals(True)
        annotation_selector_combo.clear()
        if annotation_names:
            annotation_selector_combo.addItems(annotation_names)
            annotation_selector_combo.setEnabled(True)
            if selected_annotation in annotation_names:
                annotation_selector_combo.setCurrentText(selected_annotation)
            else:
                annotation_selector_combo.setCurrentIndex(0)
        else:
            annotation_selector_combo.setEnabled(False)
        annotation_selector_combo.blockSignals(False)
        _update_annotation_summary()

    def _build_annotation_creator_row(row_index):
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_widget.setLayout(row_layout)

        category_label = QLabel(f'Category {row_index + 1}')
        value_spinbox = QSpinBox()
        value_spinbox.setRange(1, np.iinfo(np.uint16).max)
        value_spinbox.setValue(row_index + 1)

        label_edit = QLineEdit()
        label_edit.setPlaceholderText('Region label')

        colour_button = QPushButton()
        _set_colour_button_style(colour_button, _default_annotation_colour(row_index))
        colour_button.clicked.connect(
            lambda checked=False, button=colour_button: _choose_annotation_colour(button)
        )

        row_layout.addWidget(category_label)
        row_layout.addWidget(QLabel('Value'))
        row_layout.addWidget(value_spinbox)
        row_layout.addWidget(QLabel('Label'))
        row_layout.addWidget(label_edit)
        row_layout.addWidget(QLabel('Colour'))
        row_layout.addWidget(colour_button)

        return SimpleNamespace(
            widget=row_widget,
            value_spinbox=value_spinbox,
            label_edit=label_edit,
            colour_button=colour_button,
        )

    def _rebuild_annotation_creator_rows():
        _clear_qt_layout(annotation_creator_rows_layout)
        annotation_creator_rows.clear()
        for row_index in range(new_annotation_categories_input.value()):
            row = _build_annotation_creator_row(row_index)
            annotation_creator_rows.append(row)
            annotation_creator_rows_layout.addWidget(row.widget)

    def _load_selected_annotation_for_current_roi():
        annotation_name = _get_selected_annotation_name()
        if annotation_name is None:
            _set_annotation_status('No annotation selected.')
            return

        roi_name = str(roi_selector.value)
        try:
            mapping_df = _load_annotation_mapping(annotation_name)
            annotation_data = _load_annotation_array(annotation_name, roi_name)
        except Exception as exc:
            _set_annotation_status(f'Could not load annotation "{annotation_name}" for ROI "{roi_name}": {exc}')
            return

        active_layer = _get_active_annotation_layer()
        if active_layer is not None:
            viewer.layers.remove(active_layer)

        annotation_layer = viewer.add_labels(
            annotation_data.astype('uint16', copy=False),
            name=_annotation_layer_name(annotation_name),
            colormap=_build_annotation_colormap(mapping_df),
            opacity=0.7,
        )
        editable_values = mapping_df.loc[mapping_df['value'] != 0, 'value'].astype(int).tolist()
        if editable_values:
            annotation_layer.selected_label = editable_values[0]
        annotation_layer.mode = 'paint'
        annotation_layer.metadata = {
            'manual_annotation_layer': True,
            'annotation_name': annotation_name,
            'roi_name': roi_name,
        }

        viewer.layers.selection.clear()
        viewer.layers.selection.add(annotation_layer)

        annotation_state.active_annotation = annotation_name
        annotation_state.active_roi = roi_name
        _update_annotation_summary()
        _set_annotation_status(f'Loaded annotation "{annotation_name}" for ROI "{roi_name}".')

    def _save_selected_annotation_for_current_roi(silent=False):
        annotation_name = _get_selected_annotation_name()
        if annotation_name is None:
            if not silent:
                _set_annotation_status('No annotation selected.')
            return False

        roi_name = str(roi_selector.value)
        annotation_layer = _get_active_annotation_layer()
        if annotation_layer is None:
            if not silent:
                _set_annotation_status('No manual annotation Labels layer is currently loaded.')
            return False

        layer_annotation = annotation_layer.metadata.get('annotation_name')
        layer_roi = annotation_layer.metadata.get('roi_name')
        if layer_annotation != annotation_name or layer_roi != roi_name:
            if not silent:
                _set_annotation_status(
                    f'Loaded annotation layer is "{layer_annotation}" for ROI "{layer_roi}". '
                    'Load the selected annotation/ROI before saving.'
                )
            return False

        try:
            _save_annotation_array(annotation_name, roi_name, annotation_layer.data)
        except Exception as exc:
            if not silent:
                _set_annotation_status(f'Could not save annotation "{annotation_name}" for ROI "{roi_name}": {exc}')
            return False

        if not silent:
            _set_annotation_status(f'Saved annotation "{annotation_name}" for ROI "{roi_name}".')
        return True

    def _create_annotation_definition():
        annotation_name = new_annotation_name_input.text().strip()
        if not annotation_name:
            _set_annotation_status('Enter a name for the new annotation.')
            return
        if annotation_name != Path(annotation_name).name:
            _set_annotation_status('Annotation names cannot contain path separators.')
            return
        if _annotation_dir(annotation_name).exists():
            _set_annotation_status(f'Annotation "{annotation_name}" already exists.')
            return

        if not annotation_creator_rows:
            _rebuild_annotation_creator_rows()

        mapping_rows = [{'value': 0, 'label': annotation_background_label, 'color': 'transparent'}]
        seen_values = {0}
        seen_labels = {annotation_background_label}

        for row in annotation_creator_rows:
            value = int(row.value_spinbox.value())
            label_name = row.label_edit.text().strip()
            colour = row.colour_button.property('annotation_colour')

            if not label_name:
                _set_annotation_status('Each annotation category must have a label.')
                return
            if value in seen_values:
                _set_annotation_status('Annotation pixel values must be unique.')
                return
            if label_name in seen_labels:
                _set_annotation_status('Annotation labels must be unique.')
                return

            seen_values.add(value)
            seen_labels.add(label_name)
            mapping_rows.append({'value': value, 'label': label_name, 'color': str(colour)})

        mapping_df = pd.DataFrame(mapping_rows).sort_values('value').reset_index(drop=True)

        try:
            _write_annotation_mapping(annotation_name, mapping_df)
            for roi_name in all_roi_list:
                _save_annotation_array(annotation_name, roi_name, _blank_annotation_array(roi_name))
        except Exception as exc:
            _set_annotation_status(f'Could not create annotation "{annotation_name}": {exc}')
            return

        _refresh_annotation_choices(selected_annotation=annotation_name)
        _load_selected_annotation_for_current_roi()
        _set_annotation_status(
            f'Created annotation "{annotation_name}" with blank TIFF labels for {len(all_roi_list)} ROI(s).'
        )

    def _sync_selected_annotation_to_adata():
        annotation_name = _get_selected_annotation_name()
        if annotation_name is None:
            _set_annotation_status('No annotation selected.')
            return

        current_roi = str(roi_selector.value)
        if annotation_state.active_annotation == annotation_name and annotation_state.active_roi == current_roi:
            _save_selected_annotation_for_current_roi(silent=True)

        column_name = sync_annotation_column_input.text().strip() or _default_annotation_column(annotation_name)

        try:
            mapping_df = _load_annotation_mapping(annotation_name)
        except Exception as exc:
            _set_annotation_status(f'Could not read annotation mapping for "{annotation_name}": {exc}')
            return

        value_to_label = {
            int(row.value): str(row.label)
            for row in mapping_df.itertuples(index=False)
        }
        new_values = pd.Series(annotation_background_label, index=adata.obs.index, dtype='object')
        rois_in_adata = adata.obs[roi_obs].astype(str).unique().tolist()

        try:
            for roi_name in rois_in_adata:
                annotation_data = _load_annotation_array(annotation_name, roi_name)
                mask = sk.io.imread(Path(masks_folder, f'{roi_name}{mask_extension}'))
                if annotation_data.shape != mask.shape:
                    raise ValueError(
                        f"Annotation/mask shape mismatch for ROI '{roi_name}': "
                        f'{annotation_data.shape} vs {mask.shape}.'
                    )

                roi_index = adata.obs[adata.obs[roi_obs].astype(str) == str(roi_name)].index
                roi_obs_df = adata.obs.loc[roi_index, :].copy()

                if cell_id_in_mask_obs:
                    object_ids = pd.to_numeric(roi_obs_df[cell_id_in_mask_obs], errors='coerce').to_numpy()
                else:
                    object_ids = np.arange(len(roi_obs_df)) + 1

                roi_labels = []
                for object_id in object_ids:
                    if pd.isna(object_id):
                        dominant_value = 0
                    else:
                        dominant_value = _dominant_annotation_value(annotation_data[mask == int(object_id)])
                    roi_labels.append(value_to_label.get(dominant_value, annotation_background_label))

                new_values.loc[roi_index] = roi_labels
        except Exception as exc:
            _set_annotation_status(f'Could not sync annotation "{annotation_name}" to AnnData: {exc}')
            return

        new_values = new_values.fillna(annotation_background_label)
        changed_count = len(new_values)
        if column_name in adata.obs.columns:
            old_values = adata.obs[column_name].astype('string').fillna(annotation_background_label)
            changed_count = int((old_values != new_values.astype('string')).sum())

        mapping_df_sorted = mapping_df.sort_values('value')
        categories = mapping_df_sorted['label'].astype(str).tolist()
        adata.obs[column_name] = pd.Categorical(new_values, categories=categories)
        adata.uns[f'{column_name}_colors'] = mapping_df_sorted['color'].astype(str).tolist()

        if changed_count == 0:
            _set_annotation_status(
                f'Annotation "{annotation_name}" is already in sync with adata.obs["{column_name}"].'
            )
        else:
            _set_annotation_status(
                f'Synced annotation "{annotation_name}" to adata.obs["{column_name}"] and updated {changed_count} cell labels.'
            )

    annotation_widget = QWidget()
    annotation_layout = QVBoxLayout()
    annotation_widget.setLayout(annotation_layout)

    annotation_folder_title = QLabel('<b>Annotations folder</b>')
    annotation_folder_path = QLineEdit(str(annotations_folder))
    annotation_folder_path.setReadOnly(True)

    annotation_selector_title = QLabel('<b>Annotation selection</b>')
    annotation_selector_combo = QComboBox()
    annotation_refresh_button = QPushButton('Refresh annotation list')
    selected_annotation_label = QLabel('Selected annotation: none')
    loaded_annotation_label = QLabel('Loaded in viewer: none')

    annotation_mapping_title = QLabel('<b>Label mapping</b>')
    annotation_mapping_table = QTableWidget(0, 3)
    annotation_mapping_table.setHorizontalHeaderLabels(['Value', 'Label', 'Colour'])
    annotation_mapping_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    annotation_mapping_table.setSelectionMode(QAbstractItemView.NoSelection)
    annotation_mapping_table.verticalHeader().setVisible(False)
    annotation_mapping_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    create_annotation_title = QLabel('<b>Create new annotation</b>')
    create_annotation_note = QLabel(
        'Value 0 is reserved for background / Unlabelled. Choose the label values, names, and colours for the remaining categories.'
    )
    create_annotation_note.setWordWrap(True)
    new_annotation_name_input = QLineEdit()
    new_annotation_name_input.setPlaceholderText('Annotation name')
    new_annotation_categories_input = QSpinBox()
    new_annotation_categories_input.setRange(1, 64)
    new_annotation_categories_input.setValue(3)
    annotation_creator_rows_widget = QWidget()
    annotation_creator_rows_layout = QVBoxLayout()
    annotation_creator_rows_layout.setContentsMargins(0, 0, 0, 0)
    annotation_creator_rows_widget.setLayout(annotation_creator_rows_layout)
    create_annotation_button = QPushButton('Create annotation set')

    annotation_workflow_title = QLabel('<b>Load / save selected ROI</b>')
    annotation_workflow_note = QLabel(
        'Use the ROI selected in Controls. Load the annotation labels into napari, edit the Labels layer, then save back to disk.'
    )
    annotation_workflow_note.setWordWrap(True)
    load_annotation_button = QPushButton('Load selected ROI annotation')
    save_annotation_button = QPushButton('Save selected ROI annotation')

    sync_annotation_title = QLabel('<b>Sync to AnnData</b>')
    sync_annotation_note = QLabel(
        'Maps each cell to the dominant non-zero annotation value inside its mask and writes labels to adata.obs.'
    )
    sync_annotation_note.setWordWrap(True)
    sync_annotation_column_input = QLineEdit()
    sync_annotation_button = QPushButton('Sync selected annotation to AnnData')
    annotation_status_label = QLabel('No annotation selected.')
    annotation_status_label.setWordWrap(True)

    annotation_layout.addWidget(annotation_folder_title)
    annotation_layout.addWidget(annotation_folder_path)
    annotation_layout.addWidget(annotation_selector_title)
    annotation_layout.addWidget(annotation_selector_combo)
    annotation_layout.addWidget(annotation_refresh_button)
    annotation_layout.addWidget(selected_annotation_label)
    annotation_layout.addWidget(loaded_annotation_label)
    annotation_layout.addWidget(annotation_mapping_title)
    annotation_layout.addWidget(annotation_mapping_table)
    annotation_layout.addWidget(create_annotation_title)
    annotation_layout.addWidget(create_annotation_note)
    annotation_layout.addWidget(QLabel('New annotation name'))
    annotation_layout.addWidget(new_annotation_name_input)
    annotation_layout.addWidget(QLabel('Number of categories'))
    annotation_layout.addWidget(new_annotation_categories_input)
    annotation_layout.addWidget(annotation_creator_rows_widget)
    annotation_layout.addWidget(create_annotation_button)
    annotation_layout.addWidget(annotation_workflow_title)
    annotation_layout.addWidget(annotation_workflow_note)
    annotation_layout.addWidget(load_annotation_button)
    annotation_layout.addWidget(save_annotation_button)
    annotation_layout.addWidget(sync_annotation_title)
    annotation_layout.addWidget(sync_annotation_note)
    annotation_layout.addWidget(QLabel('AnnData obs column name'))
    annotation_layout.addWidget(sync_annotation_column_input)
    annotation_layout.addWidget(sync_annotation_button)
    annotation_layout.addWidget(annotation_status_label)

    annotation_selector_combo.currentTextChanged.connect(lambda text: _update_annotation_summary())
    annotation_refresh_button.clicked.connect(
        lambda checked=False: _refresh_annotation_choices(_get_selected_annotation_name())
    )
    new_annotation_categories_input.valueChanged.connect(lambda value: _rebuild_annotation_creator_rows())
    create_annotation_button.clicked.connect(lambda checked=False: _create_annotation_definition())
    load_annotation_button.clicked.connect(lambda checked=False: _load_selected_annotation_for_current_roi())
    save_annotation_button.clicked.connect(lambda checked=False: _save_selected_annotation_for_current_roi())
    sync_annotation_button.clicked.connect(lambda checked=False: _sync_selected_annotation_to_adata())

    _rebuild_annotation_creator_rows()
    _refresh_annotation_choices()

    population_qc_image_names = list(dict.fromkeys([str(image_name) for image_name in im_list]))
    population_qc_var_names = pd.Index([str(var_name) for var_name in adata.var_names.tolist()], dtype='object')

    def _population_qc_marker_aliases(marker_name):
        marker_text = str(marker_name).strip()
        if not marker_text:
            return []

        aliases = [marker_text]
        tokens = [token for token in re.split(r'[_\-\.\s]+', marker_text) if token]
        aliases.extend(tokens)
        aliases.extend(
            f'{tokens[index]}_{tokens[index + 1]}'
            for index in range(len(tokens) - 1)
        )
        return list(dict.fromkeys(alias.lower() for alias in aliases if alias))

    def _population_qc_display_candidate(marker_name):
        tokens = [token for token in re.split(r'[_\-\.\s]+', str(marker_name)) if token]
        return tokens[-1] if tokens else str(marker_name)

    population_qc_display_candidates = {
        image_name: _population_qc_display_candidate(image_name)
        for image_name in population_qc_image_names
    }
    population_qc_display_candidate_counts = pd.Series(
        list(population_qc_display_candidates.values()),
        dtype='object',
    ).value_counts()
    population_qc_image_name_to_display = {}
    population_qc_display_to_image_name = {}
    for image_name in population_qc_image_names:
        display_candidate = population_qc_display_candidates[image_name]
        if population_qc_display_candidate_counts.get(display_candidate, 0) == 1:
            display_name = display_candidate
        else:
            display_name = image_name
        population_qc_image_name_to_display[image_name] = display_name
        population_qc_display_to_image_name[display_name] = image_name

    population_qc_image_alias_to_display = {}
    for image_name, display_name in population_qc_image_name_to_display.items():
        for alias in _population_qc_marker_aliases(image_name):
            population_qc_image_alias_to_display.setdefault(alias, display_name)
        for alias in _population_qc_marker_aliases(display_name):
            population_qc_image_alias_to_display.setdefault(alias, display_name)

    population_qc_var_alias_to_name = {}
    for var_name in population_qc_var_names.tolist():
        for alias in _population_qc_marker_aliases(var_name):
            population_qc_var_alias_to_name.setdefault(alias, var_name)

    def _resolve_population_qc_display_marker(marker_name):
        marker_text = str(marker_name).strip()
        if not marker_text:
            return None

        if marker_text in population_qc_display_to_image_name:
            return marker_text
        if marker_text in population_qc_image_name_to_display:
            return population_qc_image_name_to_display[marker_text]

        for alias in _population_qc_marker_aliases(marker_text):
            if alias in population_qc_image_alias_to_display:
                return population_qc_image_alias_to_display[alias]
        return None

    def _resolve_population_qc_image_marker(marker_name):
        display_name = _resolve_population_qc_display_marker(marker_name)
        if display_name is None:
            return None
        return population_qc_display_to_image_name.get(display_name)

    def _resolve_population_qc_var_name(marker_name):
        marker_text = str(marker_name).strip()
        if not marker_text:
            return None

        if marker_text in population_qc_var_names:
            return marker_text

        image_name = _resolve_population_qc_image_marker(marker_text)
        if image_name is not None and image_name in population_qc_var_names:
            return image_name

        for alias in _population_qc_marker_aliases(marker_text):
            if alias in population_qc_var_alias_to_name:
                return population_qc_var_alias_to_name[alias]
        return None

    population_qc_marker_choices = list(population_qc_display_to_image_name.keys())
    population_qc_rankable_markers = []
    seen_population_qc_var_names = set()
    for display_name in population_qc_marker_choices:
        var_name = _resolve_population_qc_var_name(display_name)
        if var_name is None or var_name in seen_population_qc_var_names:
            continue
        population_qc_rankable_markers.append((display_name, var_name))
        seen_population_qc_var_names.add(var_name)

    population_qc_settings_columns = [
        'Red', 'Green', 'Blue',
        'Red_min', 'Red_max',
        'Green_min', 'Green_max',
        'Blue_min', 'Blue_max',
    ]
    population_qc_default_minimum = 0.2
    population_qc_default_maximum = 'q0.999'

    def _population_qc_clean_value(value):
        if value is None:
            return ''
        try:
            if pd.isna(value):
                return ''
        except TypeError:
            pass
        return str(value).strip()

    def _population_qc_settings_path(pop_obs_name):
        if not pop_obs_name:
            return None
        return Path.cwd() / f'backgating_settings_{pop_obs_name}.csv'

    def _read_population_qc_settings(pop_obs_name):
        settings_path = _population_qc_settings_path(pop_obs_name)
        if settings_path is None:
            return None, pd.DataFrame(columns=population_qc_settings_columns)

        if settings_path.exists():
            settings_df = pd.read_csv(settings_path, index_col=0)
        else:
            settings_df = pd.DataFrame()

        if settings_df.empty:
            settings_df = pd.DataFrame(columns=population_qc_settings_columns)
        else:
            settings_df.index = settings_df.index.map(str)

        for column_name in population_qc_settings_columns:
            if column_name not in settings_df.columns:
                settings_df[column_name] = None

        return settings_path, settings_df

    def _write_population_qc_settings(settings_path, settings_df):
        if settings_path is None:
            raise ValueError('No settings path is available for Population QC.')

        settings_df = settings_df.loc[:, population_qc_settings_columns].copy()
        settings_df.index = settings_df.index.map(str)
        settings_df.index.name = 'population'
        settings_df.sort_index().to_csv(settings_path)

    def _population_qc_top_markers(pop_obs_name, population_name, top_n=3):
        if not pop_obs_name or not population_name or pop_obs_name not in adata.obs.columns:
            return []
        if not population_qc_rankable_markers:
            return []

        population_mask = adata.obs[pop_obs_name].astype(str).to_numpy() == str(population_name)
        if int(population_mask.sum()) == 0:
            return []

        display_names = [display_name for display_name, _ in population_qc_rankable_markers]
        var_names = [var_name for _, var_name in population_qc_rankable_markers]
        var_positions = population_qc_var_names.get_indexer(var_names)
        valid_mask = var_positions >= 0
        if not np.any(valid_mask):
            return []

        valid_display_names = [display_name for display_name, is_valid in zip(display_names, valid_mask) if is_valid]
        valid_positions = var_positions[valid_mask]

        population_matrix = adata.X[population_mask, :]
        population_matrix = population_matrix[:, valid_positions]
        mean_expression = np.asarray(population_matrix.mean(axis=0)).ravel()
        marker_means = pd.Series(mean_expression, index=valid_display_names).dropna()
        return marker_means.nlargest(top_n).index.tolist()

    def _population_qc_default_row(pop_obs_name, population_name):
        top_markers = _population_qc_top_markers(pop_obs_name, population_name, top_n=3)
        row = {
            'Red': top_markers[0] if len(top_markers) > 0 else None,
            'Green': top_markers[1] if len(top_markers) > 1 else None,
            'Blue': top_markers[2] if len(top_markers) > 2 else None,
            'Red_min': population_qc_default_minimum,
            'Red_max': population_qc_default_maximum,
            'Green_min': population_qc_default_minimum,
            'Green_max': population_qc_default_maximum,
            'Blue_min': population_qc_default_minimum,
            'Blue_max': population_qc_default_maximum,
        }
        return row

    population_qc_widget = QWidget()
    population_qc_layout = QVBoxLayout()
    population_qc_widget.setLayout(population_qc_layout)

    population_qc_description = QLabel(
        'Use a population-specific RGB marker definition and load the selected ROI with only those channels plus the population mask.'
    )
    population_qc_description.setWordWrap(True)

    population_qc_obs_combo = QComboBox()
    population_qc_obs_combo.addItems(categorical_obs_columns)
    population_qc_population_combo = QComboBox()

    population_qc_settings_path_label = QLabel('Settings CSV: none')
    population_qc_settings_path_label.setWordWrap(True)

    population_qc_red_combo = QComboBox()
    population_qc_green_combo = QComboBox()
    population_qc_blue_combo = QComboBox()

    population_qc_create_button = QPushButton('Create blank settings CSV')
    population_qc_save_button = QPushButton('Save current RGB row')
    population_qc_load_button = QPushButton('Load population view')

    population_qc_top_rois_title = QLabel()
    population_qc_top_rois_widget = QWidget()
    population_qc_top_rois_layout = QVBoxLayout()
    population_qc_top_rois_layout.setContentsMargins(0, 0, 0, 0)
    population_qc_top_rois_widget.setLayout(population_qc_top_rois_layout)
    population_qc_top_rois_key = None
    population_qc_clicked_rois = set()

    population_qc_status_label = QLabel('Select a population obs and population.')
    population_qc_status_label.setWordWrap(True)

    def _set_population_qc_status(message):
        population_qc_status_label.setText(message)
        print(message)

    def _update_population_qc_top_rois_title():
        if randomize_initial_rois:
            if population_qc_roi_button_limit is None:
                title_text = 'Random ROIs'
            else:
                title_text = f'Random {population_qc_roi_button_limit} ROIs'
        else:
            if population_qc_roi_button_limit is None:
                title_text = 'ROIs by abundance'
            else:
                title_text = f'Top {population_qc_roi_button_limit} ROIs by abundance'
        population_qc_top_rois_title.setText(f'<b>{title_text}</b>')

    def _populate_population_qc_marker_combo(combo_box):
        combo_box.clear()
        combo_box.addItem('')
        combo_box.addItems(population_qc_marker_choices)

    def _set_population_qc_marker_value(combo_box, value):
        resolved_marker = _resolve_population_qc_display_marker(_population_qc_clean_value(value))
        if not resolved_marker:
            combo_box.setCurrentIndex(0)
            return _population_qc_clean_value(value) == ''

        selected_index = combo_box.findText(resolved_marker)
        if selected_index >= 0:
            combo_box.setCurrentIndex(selected_index)
            return True

        combo_box.setCurrentIndex(0)
        return False

    def _get_population_qc_selected_obs():
        value = population_qc_obs_combo.currentText().strip()
        return value or None

    def _get_population_qc_selected_population():
        value = population_qc_population_combo.currentText().strip()
        return value or None

    def _get_population_qc_marker_values():
        return [
            _population_qc_clean_value(population_qc_red_combo.currentText()),
            _population_qc_clean_value(population_qc_green_combo.currentText()),
            _population_qc_clean_value(population_qc_blue_combo.currentText()),
        ]

    def _set_population_qc_roi_button_style(button, visited=False):
        if visited:
            button.setStyleSheet(
                'background-color: #b5b5b5; color: #222222; border: 1px solid #7f7f7f;'
            )
        else:
            button.setStyleSheet(
                'background-color: #8fce8f; color: #1d1d1d; border: 1px solid #4f8a4f;'
            )

    def _set_population_qc_roi(roi_name):
        if not _ensure_roi_in_selector(roi_name):
            _set_population_qc_status(f'ROI "{roi_name}" is not available in Controls.')
            return False
        roi_selector.value = roi_name
        return True

    def _activate_population_qc_roi_button(roi_name, button):
        if not _set_population_qc_roi(roi_name):
            return
        if _load_population_qc_view():
            population_qc_clicked_rois.add(str(roi_name))
            _set_population_qc_roi_button_style(button, visited=True)

    def _update_population_qc_top_rois():
        nonlocal population_qc_top_rois_key
        _clear_qt_layout(population_qc_top_rois_layout)
        _update_population_qc_top_rois_title()

        selected_obs = _get_population_qc_selected_obs()
        selected_population = _get_population_qc_selected_population()
        current_key = (
            str(selected_obs) if selected_obs is not None else None,
            str(selected_population) if selected_population is not None else None,
        )
        if current_key != population_qc_top_rois_key:
            population_qc_top_rois_key = current_key
            population_qc_clicked_rois.clear()

        if not selected_obs or not selected_population:
            population_qc_top_rois_layout.addWidget(QLabel('Select a population to see ROI shortcuts.'))
            return

        roi_counts = (
            adata.obs.loc[
                adata.obs[selected_obs].astype(str) == str(selected_population),
                roi_obs,
            ]
            .dropna()
            .astype(str)
            .value_counts()
        )
        roi_counts = roi_counts.loc[[roi_name for roi_name in roi_counts.index if roi_name in all_roi_set]]

        if roi_counts.empty:
            population_qc_top_rois_layout.addWidget(QLabel('No ROIs contain this population.'))
            return

        display_roi_names = _select_population_qc_roi_choices(roi_counts.index.tolist())
        display_roi_items = [(roi_name, roi_counts.loc[roi_name]) for roi_name in display_roi_names]

        for roi_name, count in display_roi_items:
            button = QPushButton(f'{roi_name} ({count})')
            _set_population_qc_roi_button_style(button, visited=str(roi_name) in population_qc_clicked_rois)
            button.clicked.connect(
                lambda checked=False, roi_name=roi_name, button=button: _activate_population_qc_roi_button(roi_name, button)
            )
            population_qc_top_rois_layout.addWidget(button)

    def _load_population_qc_markers():
        selected_obs = _get_population_qc_selected_obs()
        selected_population = _get_population_qc_selected_population()

        settings_path = _population_qc_settings_path(selected_obs)
        if settings_path is None:
            population_qc_settings_path_label.setText('Settings CSV: none')
        else:
            population_qc_settings_path_label.setText(f'Settings CSV: {settings_path}')

        if not selected_obs or not selected_population:
            for combo_box in [population_qc_red_combo, population_qc_green_combo, population_qc_blue_combo]:
                combo_box.setCurrentIndex(0)
            _update_population_qc_top_rois()
            return

        settings_path, settings_df = _read_population_qc_settings(selected_obs)
        invalid_markers = []

        if settings_path.exists() and selected_population in settings_df.index:
            marker_row = settings_df.loc[selected_population]
            message = (
                f'Loaded RGB markers for "{selected_population}" from {settings_path.name}.'
            )
        else:
            marker_row = pd.Series(_population_qc_default_row(selected_obs, selected_population))
            if settings_path.exists():
                message = (
                    f'"{selected_population}" is missing from {settings_path.name}; '
                    'showing unsaved defaults from mean expression.'
                )
            else:
                message = (
                    f'No settings file found for "{selected_obs}"; showing unsaved defaults from mean expression.'
                )

        for combo_box, column_name in zip(
            [population_qc_red_combo, population_qc_green_combo, population_qc_blue_combo],
            ['Red', 'Green', 'Blue'],
        ):
            if not _set_population_qc_marker_value(combo_box, marker_row.get(column_name)):
                invalid_markers.append(_population_qc_clean_value(marker_row.get(column_name)))

        if invalid_markers:
            message = (
                f'{message} Unavailable marker(s) in settings: {", ".join(marker for marker in invalid_markers if marker)}.'
            )

        _update_population_qc_top_rois()
        _set_population_qc_status(message)

    def _refresh_population_qc_population_choices():
        selected_obs = _get_population_qc_selected_obs()
        current_population = _get_population_qc_selected_population()
        population_choices = _population_values(selected_obs) if selected_obs else []

        population_qc_population_combo.blockSignals(True)
        population_qc_population_combo.clear()
        population_qc_population_combo.addItems(population_choices)
        if population_choices:
            if current_population in population_choices:
                population_qc_population_combo.setCurrentText(current_population)
            else:
                population_qc_population_combo.setCurrentIndex(0)
            population_qc_population_combo.setEnabled(True)
        else:
            population_qc_population_combo.setEnabled(False)
        population_qc_population_combo.blockSignals(False)

        _load_population_qc_markers()

    def _create_population_qc_settings_file():
        selected_obs = _get_population_qc_selected_obs()
        if not selected_obs:
            _set_population_qc_status('Select a population obs before creating a settings CSV.')
            return
        if not population_qc_marker_choices:
            _set_population_qc_status('No image markers are available to build default Population QC settings.')
            return

        settings_path, settings_df = _read_population_qc_settings(selected_obs)
        population_choices = _population_values(selected_obs)

        for population_name in population_choices:
            default_row = _population_qc_default_row(selected_obs, population_name)
            if population_name not in settings_df.index:
                settings_df.loc[population_name, population_qc_settings_columns] = None
            for column_name in population_qc_settings_columns:
                current_value = _population_qc_clean_value(settings_df.loc[population_name, column_name])
                if not current_value:
                    settings_df.loc[population_name, column_name] = default_row[column_name]

        _write_population_qc_settings(settings_path, settings_df)
        _load_population_qc_markers()
        _set_population_qc_status(
            f'Created or updated Population QC settings at {settings_path}.'
        )

    def _save_population_qc_settings_row():
        selected_obs = _get_population_qc_selected_obs()
        selected_population = _get_population_qc_selected_population()
        if not selected_obs or not selected_population:
            _set_population_qc_status('Select a population obs and population before saving settings.')
            return

        selected_markers = _get_population_qc_marker_values()
        non_empty_markers = [marker for marker in selected_markers if marker]
        if not non_empty_markers:
            _set_population_qc_status('Choose at least one marker before saving the Population QC settings row.')
            return

        invalid_markers = [
            marker for marker in non_empty_markers
            if _resolve_population_qc_image_marker(marker) is None
        ]
        if invalid_markers:
            _set_population_qc_status(
                f'Cannot save markers that are not available as images: {", ".join(invalid_markers)}.'
            )
            return

        settings_path, settings_df = _read_population_qc_settings(selected_obs)
        if selected_population not in settings_df.index:
            settings_df.loc[selected_population, population_qc_settings_columns] = None

        default_row = _population_qc_default_row(selected_obs, selected_population)
        for column_name, marker_value in zip(['Red', 'Green', 'Blue'], selected_markers):
            settings_df.loc[selected_population, column_name] = marker_value or None
        for column_name in ['Red_min', 'Red_max', 'Green_min', 'Green_max', 'Blue_min', 'Blue_max']:
            current_value = _population_qc_clean_value(settings_df.loc[selected_population, column_name])
            if not current_value:
                settings_df.loc[selected_population, column_name] = default_row[column_name]

        _write_population_qc_settings(settings_path, settings_df)
        _set_population_qc_status(
            f'Saved RGB markers for "{selected_population}" to {settings_path.name}.'
        )

    def _load_population_qc_view():
        selected_obs = _get_population_qc_selected_obs()
        selected_population = _get_population_qc_selected_population()
        selected_roi = str(roi_selector.value)

        if not selected_obs or not selected_population:
            _set_population_qc_status('Select a population obs and population before loading a Population QC view.')
            return False

        selected_markers = _get_population_qc_marker_values()
        markers_to_load = [marker for marker in selected_markers if marker]
        if not markers_to_load:
            _set_population_qc_status('Choose at least one marker before loading a Population QC view.')
            return False
        if len(markers_to_load) != len(set(markers_to_load)):
            _set_population_qc_status('Population QC expects distinct Red, Green, and Blue markers.')
            return False

        resolved_markers_to_load = [
            _resolve_population_qc_image_marker(marker)
            for marker in markers_to_load
        ]
        invalid_markers = [
            marker for marker, resolved_marker in zip(markers_to_load, resolved_markers_to_load)
            if resolved_marker is None
        ]
        if invalid_markers:
            _set_population_qc_status(
                f'Cannot load unavailable image marker(s): {", ".join(invalid_markers)}.'
            )
            return False
        if len(resolved_markers_to_load) != len(set(resolved_markers_to_load)):
            _set_population_qc_status('Population QC expects distinct Red, Green, and Blue markers.')
            return False

        _delete_all_layers()
        _add_images_from_list(resolved_markers_to_load)
        _add_masks(
            roi_name=selected_roi,
            adata=adata,
            pop_obs=selected_obs,
            roi_obs=roi_obs,
            adata_colormap=True,
            colour_map=colormaps['tab20'].colors,
            add_individual_pops=True,
            selected_populations=[selected_population],
            add_combined_mask=False,
            individual_layer_name_prefix=f'{selected_obs}::',
            individual_layers_visible=True,
            add_base_mask=False,
        )
        _set_population_qc_status(
            f'Loaded ROI "{selected_roi}" with {", ".join(markers_to_load)} and the "{selected_population}" mask.'
        )
        return True

    for combo_box in [population_qc_red_combo, population_qc_green_combo, population_qc_blue_combo]:
        _populate_population_qc_marker_combo(combo_box)

    population_qc_layout.addWidget(population_qc_description)
    population_qc_layout.addWidget(QLabel('Population obs'))
    population_qc_layout.addWidget(population_qc_obs_combo)
    population_qc_layout.addWidget(QLabel('Population'))
    population_qc_layout.addWidget(population_qc_population_combo)
    population_qc_layout.addWidget(population_qc_settings_path_label)
    population_qc_layout.addWidget(QLabel('Red marker'))
    population_qc_layout.addWidget(population_qc_red_combo)
    population_qc_layout.addWidget(QLabel('Green marker'))
    population_qc_layout.addWidget(population_qc_green_combo)
    population_qc_layout.addWidget(QLabel('Blue marker'))
    population_qc_layout.addWidget(population_qc_blue_combo)
    population_qc_layout.addWidget(population_qc_create_button)
    population_qc_layout.addWidget(population_qc_save_button)
    population_qc_layout.addWidget(population_qc_load_button)
    population_qc_layout.addWidget(population_qc_top_rois_title)
    population_qc_layout.addWidget(population_qc_top_rois_widget)
    population_qc_layout.addWidget(population_qc_status_label)

    population_qc_obs_combo.currentTextChanged.connect(lambda text: _refresh_population_qc_population_choices())
    population_qc_population_combo.currentTextChanged.connect(lambda text: _load_population_qc_markers())
    population_qc_create_button.clicked.connect(lambda checked=False: _create_population_qc_settings_file())
    population_qc_save_button.clicked.connect(lambda checked=False: _save_population_qc_settings_row())
    population_qc_load_button.clicked.connect(lambda checked=False: _load_population_qc_view())

    if categorical_obs_columns:
        _refresh_population_qc_population_choices()
    else:
        population_qc_obs_combo.setEnabled(False)
        population_qc_population_combo.setEnabled(False)
        population_qc_create_button.setEnabled(False)
        population_qc_save_button.setEnabled(False)
        population_qc_load_button.setEnabled(False)
        _set_population_qc_status('No categorical obs columns are available for Population QC.')

    def build_dock_panel(widget_items):
        """
        Build a reusable QWidget panel from existing controls.
        """
        panel_widget = QWidget()
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_widget.setLayout(panel_layout)

        for item in widget_items:
            panel_layout.addWidget(getattr(item, 'native', item))

        return panel_widget

    dock_panels = {
        'Controls': build_dock_panel([
            add_roi_label,
            roi_selector,
            hide_all_layers_button,
            delete_all_layers_button,
            layers_to_top_button,
            add_roi_images_button,
            add_masks_button,
        ]),
        'Population QC': population_qc_widget,
        'Add raw images': build_dock_panel([
            _image_selector,
            quant_select_label,
            quant_select,
            minimum_pixel_counts_select_label,
            minimum_pixel_counts_select,
        ]),
        'Categories as masks': build_dock_panel([
            _obs_selector,
            individual_pops_toggle,
        ]),
        'Numeric as masks': build_dock_panel([
            _quant_selector,
        ]),
        'Manual annotations': annotation_widget,
        'Layer management': layer_management_widget,
    }
    dock_order = [
        'Controls',
        'Population QC',
        'Add raw images',
        'Categories as masks',
        'Numeric as masks',
        'Manual annotations',
        'Layer management',
    ]
    dock_widgets = {}

    def center_dock_on_main_window(dock_widget):
        """
        Center a floating dock on the Napari window while keeping it onscreen.
        """
        qt_window = getattr(viewer.window, '_qt_window', None)
        screen = None
        if qt_window is not None:
            screen = qt_window.screen()
            if screen is None and qt_window.windowHandle() is not None:
                screen = qt_window.windowHandle().screen()
        if screen is None:
            screen = dock_widget.screen()

        screen_geometry = screen.availableGeometry() if screen is not None else dock_widget.frameGeometry()
        target_geometry = qt_window.frameGeometry() if qt_window is not None else screen_geometry

        dock_widget.adjustSize()
        dock_width = dock_widget.frameGeometry().width() or dock_widget.sizeHint().width()
        dock_height = dock_widget.frameGeometry().height() or dock_widget.sizeHint().height()

        x_pos = target_geometry.x() + max((target_geometry.width() - dock_width) // 2, 0)
        y_pos = target_geometry.y() + max((target_geometry.height() - dock_height) // 2, 0)

        max_x = screen_geometry.x() + max(screen_geometry.width() - dock_width, 0)
        max_y = screen_geometry.y() + max(screen_geometry.height() - dock_height, 0)
        x_pos = min(max(x_pos, screen_geometry.x()), max_x)
        y_pos = min(max(y_pos, screen_geometry.y()), max_y)

        dock_widget.move(x_pos, y_pos)

    def show_dock(dock_name):
        """
        Recreate and show a dock widget so closed docked panels can be restored.
        """
        dock_widget = dock_widgets.get(dock_name)
        dock_panel = dock_panels[dock_name]
        qt_window = getattr(viewer.window, '_qt_window', None)

        if dock_widget is not None:
            current_panel = dock_widget.widget()
            if current_panel is not None:
                current_panel.setParent(None)
                dock_panel = current_panel
                dock_panels[dock_name] = current_panel
            if qt_window is not None:
                qt_window.removeDockWidget(dock_widget)
            dock_widget.deleteLater()

        dock_widget = viewer.window.add_dock_widget(
            dock_panel,
            name=dock_name,
            area='right',
        )
        dock_widgets[dock_name] = dock_widget

        dock_widget.setFloating(True)
        dock_widget.setVisible(True)
        dock_widget.show()

        toggle_action = dock_widget.toggleViewAction()
        if toggle_action is not None:
            toggle_action.setChecked(True)

        if dock_name == 'Layer management':
            update_layer_list(silent=True)
        elif dock_name == 'Population QC':
            _load_population_qc_markers()

        center_dock_on_main_window(dock_widget)
        dock_widget.activateWindow()
        dock_widget.raise_()
        return dock_widget

    def show_all_docks():
        """
        Open all registered dock widgets from the launcher.
        """
        for dock_name in dock_order:
            show_dock(dock_name)

    def sync_layer_widgets(event=None):
        """
        Keep layer-dependent selectors in sync with the current viewer state.
        """
        update_layer_list(silent=True)
        annotation_layer = _get_active_annotation_layer()
        if annotation_layer is None:
            annotation_state.active_annotation = None
            annotation_state.active_roi = None
            loaded_annotation_label.setText('Loaded in viewer: none')
        else:
            annotation_state.active_annotation = annotation_layer.metadata.get('annotation_name')
            annotation_state.active_roi = annotation_layer.metadata.get('roi_name')
            loaded_annotation_label.setText(
                f'Loaded in viewer: {annotation_state.active_annotation} ({annotation_state.active_roi})'
            )

    viewer.layers.events.inserted.connect(sync_layer_widgets)
    viewer.layers.events.removed.connect(sync_layer_widgets)

    panel_launcher_widget = QWidget()
    panel_launcher_layout = QVBoxLayout()
    panel_launcher_widget.setLayout(panel_launcher_layout)

    panel_launcher_layout.addWidget(
        widgets.Label(value='Open or restore explorer panels:').native
    )

    open_all_docks_button = widgets.PushButton(text='Open all panels')
    open_all_docks_button.clicked.connect(show_all_docks)
    panel_launcher_layout.addWidget(open_all_docks_button.native)

    dock_launcher_buttons = {}
    for dock_name in dock_order:
        button = widgets.PushButton(text=f'Open {dock_name}')
        button.clicked.connect(lambda checked=False, dock_name=dock_name: show_dock(dock_name))
        panel_launcher_layout.addWidget(button.native)
        dock_launcher_buttons[dock_name] = button

    panel_launcher_dock = viewer.window.add_dock_widget(
        panel_launcher_widget,
        name='Panels',
        area='left',
    )
    panel_launcher_dock.setFeatures(
        panel_launcher_dock.features() & ~QDockWidget.DockWidgetClosable
    )

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

    # Bundle exposed controls and helpers to avoid setting attributes on viewer
    handles = SimpleNamespace(
        roi_selector=roi_selector,
        layer_resize_selector=resize_layers_widget,
        layer_resize_target=resize_layers_widget.target_layer,
        colormap_source_selector=transfer_colormap_widget,
        colormap_source_choice=transfer_colormap_widget.source_layer,
        mask_layer_selector=mask_layer_widget,
        mask_layer_choice=mask_layer_widget.layer_to_mask,
        image_select_widget=image_select_widget,
        obs_select_widget=obs_select_widget,
        quant_select_widget=quant_select_widget,
        all_roi_list=list(all_roi_list),
        initial_roi_selector_choices=_get_roi_selector_choices,
        get_selected_roi=lambda: roi_selector.value,
        get_selected_layer_name=lambda: (list(viewer.layers.selection)[0].name if viewer.layers.selection else None),
        get_selected_images=get_selected_images,
        get_selected_obs_categories=get_selected_obs_categories,
        get_selected_numeric=get_selected_numeric,
        annotations_folder=annotations_folder,
        annotation_widget=annotation_widget,
        annotation_selector=annotation_selector_combo,
        annotation_sync_column_input=sync_annotation_column_input,
        load_annotation_for_current_roi=_load_selected_annotation_for_current_roi,
        save_annotation_for_current_roi=_save_selected_annotation_for_current_roi,
        sync_annotation_to_adata=_sync_selected_annotation_to_adata,
        population_qc_widget=population_qc_widget,
        population_qc_obs_selector=population_qc_obs_combo,
        population_qc_population_selector=population_qc_population_combo,
        population_qc_red_marker=population_qc_red_combo,
        population_qc_green_marker=population_qc_green_combo,
        population_qc_blue_marker=population_qc_blue_combo,
        population_qc_create_settings=_create_population_qc_settings_file,
        population_qc_save_settings=_save_population_qc_settings_row,
        population_qc_load_view=_load_population_qc_view,
        show_dock=show_dock,
        show_all_docks=show_all_docks,
        panel_launcher_dock=panel_launcher_dock,
        dock_widgets=dock_widgets,
        dock_launcher_buttons=dock_launcher_buttons,
    )

    # Start the Napari event loop
    napari.run()
    return viewer, handles
