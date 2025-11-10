import os
import re
from pathlib import Path
from glob import glob
from itertools import compress
from typing import List, Union, Tuple, Optional
from math import ceil
from IPython.display import display

import numpy as np
import pandas as pd
import tifffile as tp
import matplotlib.pyplot as plt
import scanpy as sc

from skimage import io, exposure, segmentation
from skimage.draw import rectangle_perimeter
from skimage.util import img_as_ubyte


def clean_text(text: str) -> str:
    """Remove special characters to produce safe filenames."""
    for ch in ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+',
               '-', '.', '!', '$', '\'', ',', ' ', '/', '"']:
        text = text.replace(ch, '')
    return text


def load_single_img(filename: str) -> np.ndarray:
    """
    Load a single 2D .tif or .tiff image as float32.

    Args:
        filename (str): Path to the image file, must end with .tiff or .tif.

    Returns:
        np.ndarray: Loaded image data (2D).
    """
    if not (filename.endswith('.tiff') or filename.endswith('.tif')):
        raise ValueError('Raw file should end with .tif or .tiff!')
    img_in = tp.imread(filename).astype('float32')

    if img_in.ndim != 2:
        raise ValueError('Single image should be 2D!')
    return img_in


def load_imgs_from_directory(
    load_directory: str,
    channel_name: str,
    quiet: bool = False
) -> Optional[Tuple[List[np.ndarray], List[str], List[str]]]:
    """
    Searches a directory (and any subfolders) for images whose filenames
    contain the given channel_name. Returns a list of those images and filenames.

    Args:
        load_directory (str): Directory or parent directory of images.
        channel_name (str): Channel name to search for in the filenames.
        quiet (bool): Whether to suppress print statements.

    Returns:
        Optional[Tuple[List[np.ndarray], List[str], List[str]]]:
            - A list of images (2D numpy arrays)
            - A list of corresponding filenames
            - A list of the subfolders from which images were loaded
              If no images are found, returns None.
    """
    img_collect = []
    img_file_list = []

    # Find any subdirectories (one level down). If none found, use load_directory itself.
    img_folders = glob(os.path.join(load_directory, "*", "")) or [load_directory]

    if not quiet:
        print(f'Loading image data for channel "{channel_name}" from ...')

    for subfolder in img_folders:
        found_files = [
            f for f in os.listdir(subfolder)
            if os.path.isfile(os.path.join(subfolder, f)) and
               (f.lower().endswith(".tiff") or f.lower().endswith(".tif"))
        ]

        for candidate_file in found_files:
            # More precise matching: check if channel_name appears as a separate word/token
            # This prevents CD45 from matching CD45RO
            filename_lower = candidate_file.lower()
            channel_lower = channel_name.lower()
            
            # Split filename by common separators and check for exact match
            import re
            # Split on common separators: underscore, dash, dot, space
            filename_tokens = re.split(r'[_\-\.\s]+', filename_lower)
            
            # Check if channel name matches any token exactly
            if channel_lower in filename_tokens:
                img_read = load_single_img(os.path.join(subfolder, candidate_file))
                if not quiet:
                    print(os.path.join(subfolder, candidate_file))
                img_file_list.append(candidate_file)
                img_collect.append(img_read)
                # Break once we find the first matching file per subfolder.
                break

    if not quiet:
        print('Image data loading completed!')

    if not img_collect:
        print(f'No files found with channel name "{channel_name}".')
        return None

    return img_collect, img_file_list, img_folders


def load_rescale_images(
    image_folder: str,
    samples_list: List[str],
    marker: str,
    minimum: float,
    max_val: Union[float, str]
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Helper function that:
      1) Loads images for a given marker across provided samples.
      2) Clips intensities using user-specified or quantile-based maxima.
      3) Rescales intensities to [0,1].

    Args:
        image_folder (str): Directory where images (and subfolders) are located.
        samples_list (List[str]): List of ROI/sample names to filter by.
        marker (str): The marker (channel name) to load from the image folder.
        minimum (float): Lower clip value.
        max_val (Union[float, str]): A numeric max or a string with prefix:
          - 'q': Mean quantile
          - 'i': Individual quantile
          - 'm': Minimum of quantiles
          - 'x': Maximum of quantiles
          Example: 'q0.97' => Use mean of the 97th percentile for all images.

    Returns:
        Tuple[List[np.ndarray], List[str]]:
            - List of images (each rescaled/clipped)
            - Matching list of ROI names
    """
    # Interpret the user-specified max_val mode
    mode = 'value'
    if isinstance(max_val, str):
        prefix = max_val[0].lower()
        try:
            max_quantile = float(max_val[1:])
        except ValueError:
            raise ValueError(f"Could not parse quantile from '{max_val}'")
        if prefix == 'q':
            mode = 'mean_quantile'
        elif prefix == 'i':
            mode = 'individual_quantile'
        elif prefix == 'm':
            mode = 'minimum_quantile'
        elif prefix == 'x':
            mode = 'max_quantile'

    # Load the images (quiet=True to suppress prints)
    loaded = load_imgs_from_directory(image_folder, marker, quiet=True)
    if not loaded:
        return [], []

    image_list, _, folder_list = loaded

    # ROI names are the last part of the subfolder path
    roi_list = [os.path.basename(Path(x)) for x in folder_list]

    # Filter out any ROIs not in our samples_list
    sample_filter = [r in samples_list for r in roi_list]
    image_list = list(compress(image_list, sample_filter))
    roi_list = list(compress(roi_list, sample_filter))

    if not image_list:
        # If nothing is loaded after filtering, return.
        print(f"No images found for marker {marker} matching {samples_list}.")
        return [], []

    # Compute maximum intensities
    if mode in ('mean_quantile', 'minimum_quantile', 'max_quantile'):
        # For each image, find the quantile, then reduce them by mean, min, or max
        all_vals = [np.quantile(im, max_quantile) for im in image_list]
        if mode == 'mean_quantile':
            max_value = float(np.mean(all_vals))
            mode_str = f'Mean of {max_quantile} quantiles'
        elif mode == 'minimum_quantile':
            max_value = float(np.min(all_vals))
            mode_str = f'Min of {max_quantile} quantiles'
        else:  # 'max_quantile'
            max_value = float(np.max(all_vals))
            mode_str = f'Max of {max_quantile} quantiles'

        print(f"Marker={marker} | Mode={mode_str} | Min={minimum:.3f} | "
              f"Calculated max={max_value:.3f}")
        image_list = [im.clip(minimum, max_value) for im in image_list]

    elif mode == 'individual_quantile':
        # Each image is clipped to its own quantile
        max_values = [np.quantile(im, max_quantile) for im in image_list]
        print(f"Marker={marker} | Mode=Individual quantile {max_quantile} | "
              f"Min={minimum} | Using image-specific maxima.")
        image_list = [
            im.clip(minimum, mv) for im, mv in zip(image_list, max_values)
        ]

    else:
        # Fixed numeric value
        print(f"Marker={marker} | Using numeric min={minimum}, max={max_val}")
        max_value = float(max_val)
        image_list = [im.clip(minimum, max_value) for im in image_list]

    # Rescale intensities to [0..1]
    image_list = [exposure.rescale_intensity(i) for i in image_list]

    return image_list, roi_list


def make_images(
    image_folder: str,
    samples_list: List[str],
    output_folder: str,
    name_prefix: str = '',
    minimum: float = 0.2,
    max_quantile: Union[float, str] = 'q0.97',
    red: Optional[str] = None,
    red_range: Optional[Tuple[float, Union[str, float]]] = None,
    green: Optional[str] = None,
    green_range: Optional[Tuple[float, Union[str, float]]] = None,
    blue: Optional[str] = None,
    blue_range: Optional[Tuple[float, Union[str, float]]] = None,
    magenta: Optional[str] = None,
    magenta_range: Optional[Tuple[float, Union[str, float]]] = None,
    cyan: Optional[str] = None,
    cyan_range: Optional[Tuple[float, Union[str, float]]] = None,
    yellow: Optional[str] = None,
    yellow_range: Optional[Tuple[float, Union[str, float]]] = None,
    white: Optional[str] = None,
    white_range: Optional[Tuple[float, Union[str, float]]] = None,
    roi_folder_save: bool = False,
    simple_file_names: bool = False,
    save_subfolder: str = ''
) -> None:
    """
    Create composite RGB images from up to seven channels. Each channel can be
    mapped onto red/green/blue/magenta/cyan/yellow/white in an additive manner
    (as done by typical multi-channel viewers).

    Args:
        image_folder (str): Folder of subfolders where each ROI is stored.
        samples_list (List[str]): List of ROI names to process.
        output_folder (str): Where to save the resulting images.
        name_prefix (str): Optional prefix for output files.
        minimum (float): Global intensity minimum for clipping (before rescale).
        max_quantile (float or str): Global intensity maximum for clipping
            (e.g., 0.97 or 'q0.97' or 'i0.97').
        {color} (str): The marker to use for that color channel.
        {color}_range (tuple): Lower and upper intensity specs, can be numeric or 'q0.95', etc.
        roi_folder_save (bool): Whether each ROI gets its own subfolder in output.
        simple_file_names (bool): If True, save images as 'ROI.png' only (otherwise includes channel info).
        save_subfolder (str): Subdirectory under output_folder for saving images.

    Returns:
        None. Saves .png images to disk.
    """

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Map the color name to the marker name and to user-specified ranges
    color_configs = {
        'red':     (red,     red_range),
        'green':   (green,   green_range),
        'blue':    (blue,    blue_range),
        'magenta': (magenta, magenta_range),
        'cyan':    (cyan,    cyan_range),
        'yellow':  (yellow,  yellow_range),
        'white':   (white,   white_range),
    }

    # For each color channel, load images & ROI lists
    loaded_images = {}  # color -> list of scaled images
    loaded_rois = {}    # color -> list of ROI names

    for color_name, (marker_name, color_range) in color_configs.items():
        if marker_name is not None:
            # Determine the min/max for this channel if provided
            if color_range is not None:
                ch_min, ch_max = color_range
            else:
                ch_min, ch_max = (minimum, max_quantile)

            imgs, rois = load_rescale_images(
                image_folder, samples_list,
                marker=marker_name,
                minimum=ch_min,
                max_val=ch_max
            )
            loaded_images[color_name] = imgs
            loaded_rois[color_name] = rois
        else:
            # This color not used
            loaded_images[color_name] = []
            loaded_rois[color_name] = []

    # Figure out how many ROIs total we have. We can unify by taking the max length across channels.
    num_rois = max(len(r) for r in loaded_rois.values()) if loaded_rois else 0
    print(f'Found {num_rois} ROIs total (across requested channels).')

    # Build the final composite images ROI by ROI
    # Note: The assumption here is that channels align in the same "ROI index" order,
    # because we used the same sample_list for each. If your data is misaligned, you'll
    # need more robust logic (e.g., matching by ROI name).
    for i in range(num_rois):
        # We pick whichever color has a valid ROI for indexing
        # and assume they are the same ROI in the same i-th position
        # in each channel's list. If any channel is missing that i-th ROI,
        # we fill with zeros.

        # A quick approach is to find a channel that has rois for index i
        # and get the actual ROI name from there. Then we try to match
        # which index that ROI is in the other channels. A more thorough
        # approach would be to unify them by dictionary, but that requires
        # more changes.

        # For simplicity, let's pick the first non-empty color:
        some_color = None
        for c in color_configs:
            if i < len(loaded_rois[c]):
                some_color = c
                break
        if some_color is None:
            continue  # no channels have i-th ROI (unlikely)

        roi_name = loaded_rois[some_color][i]

        # Now gather the images for each color, matching the ROI name
        # by index if it matches, else a blank array of the same shape.
        shape_ref = loaded_images[some_color][i].shape  # reference shape
        (h, w) = shape_ref

        # Initialize R, G, B as zeros
        channel_r = np.zeros((h, w), dtype=np.float32)
        channel_g = np.zeros((h, w), dtype=np.float32)
        channel_b = np.zeros((h, w), dtype=np.float32)

        for color_name, (marker_name, _) in color_configs.items():
            if marker_name is None:
                # Not used
                continue
            # Attempt to find the ROI in that channel
            if roi_name in loaded_rois[color_name]:
                # find the index for that ROI
                idx = loaded_rois[color_name].index(roi_name)
                this_img = loaded_images[color_name][idx]
            else:
                # no ROI found => fallback to zeros
                this_img = np.zeros((h, w), dtype=np.float32)

            if color_name == 'red':
                channel_r = np.clip(channel_r + this_img, 0, 1)
            elif color_name == 'green':
                channel_g = np.clip(channel_g + this_img, 0, 1)
            elif color_name == 'blue':
                channel_b = np.clip(channel_b + this_img, 0, 1)
            elif color_name == 'magenta':
                # Magenta = Red + Blue
                channel_r = np.clip(channel_r + this_img, 0, 1)
                channel_b = np.clip(channel_b + this_img, 0, 1)
            elif color_name == 'cyan':
                # Cyan = Green + Blue
                channel_g = np.clip(channel_g + this_img, 0, 1)
                channel_b = np.clip(channel_b + this_img, 0, 1)
            elif color_name == 'yellow':
                # Yellow = Red + Green
                channel_r = np.clip(channel_r + this_img, 0, 1)
                channel_g = np.clip(channel_g + this_img, 0, 1)
            elif color_name == 'white':
                # White = Red + Green + Blue
                channel_r = np.clip(channel_r + this_img, 0, 1)
                channel_g = np.clip(channel_g + this_img, 0, 1)
                channel_b = np.clip(channel_b + this_img, 0, 1)

        # Stack channels
        stack = np.dstack([channel_r, channel_g, channel_b])
        stack_ubyte = img_as_ubyte(stack)

        # Build filename
        if not simple_file_names:
            # Include the channels used (e.g., b_markerName_, r_markerName_, etc.)
            color_strs = []
            for color_name, (marker_name, _) in color_configs.items():
                if marker_name:
                    prefix = color_name[0].lower()  # r/g/b/m/c/y/w
                    color_strs.append(f'{prefix}_{marker_name}')
            color_part = "_".join(color_strs)
            filename = f'{name_prefix}{roi_name}_{color_part}'.rstrip('_')
        else:
            filename = roi_name

        # Possibly write to a subfolder named after ROI
        if roi_folder_save:
            roi_dir = Path(output_folder, roi_name)
            roi_dir.mkdir(parents=True, exist_ok=True)
            if save_subfolder:
                roi_dir = roi_dir / save_subfolder
                roi_dir.mkdir(parents=True, exist_ok=True)
            save_path = roi_dir / f'{filename}.png'
        else:
            out_dir = Path(output_folder)
            if save_subfolder:
                out_dir = out_dir / save_subfolder
                out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f'{filename}.png'

        io.imsave(str(save_path), stack_ubyte)


def backgating(
    adata,
    cell_index,
    radius,
    image_folder,
    # channel marker assignments
    red=None,
    red_range=None,
    green=None,
    green_range=None,
    blue=None,
    blue_range=None,
    magenta=None,
    magenta_range=None,
    cyan=None,
    cyan_range=None,
    yellow=None,
    yellow_range=None,
    white=None,
    white_range=None,
    # directories
    output_folder='Backgating',
    save_subfolder='',
    # adata.obs field names
    roi_obs='ROI',
    x_loc_obs='X_loc',
    y_loc_obs='Y_loc',
    cell_index_obs='Master_Index',
    # mask usage
    use_masks=False,
    mask_folder='masks',
    exclude_rois_without_mask=True,
    # layout options
    cells_per_row=5,
    cell_plot_spacing=(0.1, 0.1),
    overview_images=True,
    show_gallery_titles=True,
    # intensity scaling
    minimum=0.2,
    max_quantile='q0.97',
    # optional interactive training
    training=False,
):
    """
    Visualize small patches around given cell(s) on top of optionally composite images.
    Optionally overlay segmentation masks.

    Args:
        adata: AnnData object with .obs containing ROI names and x,y coords.
        cell_index: A single integer or list of integers identifying cells to visualize.
        radius: Radius (in pixels) of the square around each cell in the thumbnail.

        image_folder (str):
            Directory with subfolders, each named for an ROI, containing channel tiffs.

        {color} (str):
            Marker channel to use for that color (e.g. "CD3", "DNA1").

        {color}_range (tuple):
            (min, max) or e.g. (0.2, 'q0.97') for intensity clipping.

        output_folder (str):
            Where to save final images.

        save_subfolder (str):
            Subdirectory of output_folder for saving.

        roi_obs, x_loc_obs, y_loc_obs, cell_index_obs (str):
            Column names in adata.obs for ROI, X, Y, and unique cell index.

        use_masks (bool or str):
            - False => No masks
            - True  => Load masks from `mask_folder/<ROI>.tif(f)`
            - str   => Path to CSV with ROI->mask path

        mask_folder (str):
            Folder where <ROI>.tif or <ROI>.tiff is expected (if use_masks=True).

        exclude_rois_without_mask (bool):
            If True, drop any ROI that doesn't have a mask file.

        cells_per_row (int):
            Number of cell thumbnails per row in final “Cells.png”.

        cell_plot_spacing (tuple):
            (vertical_space, horizontal_space) for subplots_adjust.

        overview_images (bool):
            If True, saves an “overview” with bounding boxes for each cell.

        minimum, max_quantile (float or str):
            Global clipping parameters for channels, used by `make_images`.

        training (bool):
            If True, code can prompt for user input after each cell (placeholder logic).

    Returns:
        None.
        Saves “Cells.png”, optional “_overview.png” images, and a CSV of used cells.
    """
    # ----------------------------------------------------------------
    # 1) Normalize cell_index to a list
    # ----------------------------------------------------------------
    if not isinstance(cell_index, list):
        cell_index = [cell_index]

    # Subset to just the cells of interest
    adata_obs_cells = adata.obs.loc[adata.obs[cell_index_obs].isin(cell_index)].copy()
    roi_list = adata_obs_cells[roi_obs].unique().tolist()

    print(f"Backgating on {len(adata_obs_cells)} cells across {len(roi_list)} ROIs.")

    # ----------------------------------------------------------------
    # 2) Build composite images for each ROI (from your existing function)
    # ----------------------------------------------------------------
    print("Creating composite images via `make_images` ...")
    make_images(
        image_folder=image_folder,
        samples_list=roi_list,
        output_folder=output_folder,
        simple_file_names=True,
        minimum=minimum,
        max_quantile=max_quantile,
        red=red,
        red_range=red_range,
        green=green,
        green_range=green_range,
        blue=blue,
        blue_range=blue_range,
        magenta=magenta,
        magenta_range=magenta_range,
        cyan=cyan,
        cyan_range=cyan_range,
        yellow=yellow,
        yellow_range=yellow_range,
        white=white,
        white_range=white_range,
        save_subfolder=save_subfolder
    )
    print("Composite images created.")

    # ----------------------------------------------------------------
    # 3) Load those composite images from disk into a DataFrame
    # ----------------------------------------------------------------
    out_subdir = Path(output_folder) / save_subfolder
    images = []
    for roi_name in roi_list:
        roi_path = out_subdir / f"{roi_name}.png"
        if roi_path.exists():
            images.append(io.imread(str(roi_path)))
        else:
            images.append(None)

    # Create df_images with object dtype so we can store arrays in cells.
    df_images = pd.DataFrame({
        roi_obs: roi_list,
        'image': images
    }, dtype=object)

    # Drop any duplicates in case ROI names repeat
    df_images.drop_duplicates(subset=roi_obs, inplace=True)
    df_images.set_index(roi_obs, inplace=True)

    # Create mask column with object dtype
    df_images['mask'] = [None]*len(df_images)
    df_images = df_images.astype({'mask': 'object'})

    # For shapes
    df_images['y_length'] = [img.shape[0] if img is not None else 0 for img in df_images['image']]
    df_images['x_length'] = [img.shape[1] if img is not None else 0 for img in df_images['image']]

    print(f"DataFrame of images built for {len(df_images)} ROIs.")

    # ----------------------------------------------------------------
    # 4) Load segmentation masks, if requested
    # ----------------------------------------------------------------
    rois_to_exclude = []

    #display(df_images)
    #df_images.to_csv('test.csv')

    if use_masks:
        #print(f"Attempting to load masks. mask_folder='{mask_folder}'")
        if isinstance(use_masks, bool) and use_masks is True:
            # Look for <ROI>.tif or <ROI>.tiff in mask_folder
            for roi_name in df_images.index:
                potential_tif  = Path(mask_folder) / f"{roi_name}.tif"
                potential_tiff = Path(mask_folder) / f"{roi_name}.tiff"

                if potential_tif.is_file():
                    df_images.at[roi_name, 'mask'] = io.imread(str(potential_tif))
                    #print(f"  Loaded mask for ROI='{roi_name}' -> {potential_tif}")
                elif potential_tiff.is_file():
                    #print(roi_name)
                    df_images.at[roi_name, 'mask'] = io.imread(str(potential_tiff))
                    #print(f"  Loaded mask for ROI='{roi_name}' -> {potential_tiff}")
                else:
                    rois_to_exclude.append(roi_name)

        elif isinstance(use_masks, str):
            # user passed a CSV mapping ROI->mask path
            mask_csv = Path(use_masks)
            if not mask_csv.is_file():
                print(f"WARNING: mask CSV file '{mask_csv}' not found. No masks loaded.")
            else:
                print(f"Loading mask mappings from CSV: {mask_csv}")
                mask_df = pd.read_csv(mask_csv).set_index(roi_obs)
                for roi_name in df_images.index:
                    if roi_name in mask_df.index:
                        mask_path = Path(mask_df.loc[roi_name, 'mask_path'])
                        if mask_path.is_file():
                            df_images.at[roi_name, 'mask'] = io.imread(str(mask_path))
                            print(f"  Loaded mask for ROI='{roi_name}' -> {mask_path}")
                        else:
                            rois_to_exclude.append(roi_name)
                    else:
                        rois_to_exclude.append(roi_name)

        if rois_to_exclude:
            print("WARNING: The following ROIs do NOT have matching mask files:")
            for r_ in rois_to_exclude:
                print(f"    - {r_}")
            if exclude_rois_without_mask:
                print("These ROIs will be excluded due to `exclude_rois_without_mask=True`.")
                df_images.drop(rois_to_exclude, inplace=True)

    # Now filter cells in adata_obs_cells if their ROI was excluded
    valid_rois = df_images.index.tolist()
    before_count = len(adata_obs_cells)
    adata_obs_cells = adata_obs_cells[adata_obs_cells[roi_obs].isin(valid_rois)]
    after_count = len(adata_obs_cells)
    if before_count != after_count:
        print(f"Excluded {before_count - after_count} cells whose ROI lacked masks (or images).")

    # ----------------------------------------------------------------
    # 5) Filter out cells that would be out-of-bounds for the given radius
    # ----------------------------------------------------------------
    adata_obs_cells['x_max'] = adata_obs_cells[roi_obs].map(df_images['x_length'])
    adata_obs_cells['y_max'] = adata_obs_cells[roi_obs].map(df_images['y_length'])

    def in_range_func(row):
        x = row[x_loc_obs]
        y = row[y_loc_obs]
        return (
            (x - radius >= 0) and (x + radius <= row['x_max']) and
            (y - radius >= 0) and (y + radius <= row['y_max'])
        )

    adata_obs_cells['in_range'] = adata_obs_cells.apply(in_range_func, axis=1)
    adata_obs_cells_filtered = adata_obs_cells[adata_obs_cells['in_range']].copy()
    out_of_bounds_count = len(adata_obs_cells) - len(adata_obs_cells_filtered)
    print(f"{out_of_bounds_count} cells are out-of-bounds for plotting.")
    print(f"Proceeding with {len(adata_obs_cells_filtered)} cells.")

    if len(adata_obs_cells_filtered) == 0:
        print("No valid cells remain to plot. Exiting backgating.")
        return

    # ----------------------------------------------------------------
    # 6) Plot cell thumbnails in a big figure
    # ----------------------------------------------------------------
    total_cells = len(adata_obs_cells_filtered)
    rows = ceil(total_cells / cells_per_row)
    fig, axs = plt.subplots(rows, cells_per_row, figsize=(10, rows * 2), dpi=100)
    axs = axs.flatten() if (rows > 1 or cells_per_row > 1) else [axs]

    ax_idx = 0
    cell_dfs = []

    print(f"Plotting thumbnails ({cells_per_row} per row) for {total_cells} cells...")

    # Sort by ROI to group
    rois_in_data = adata_obs_cells_filtered[roi_obs].unique().tolist()

    for roi_name in rois_in_data:
        sub_cells = adata_obs_cells_filtered[adata_obs_cells_filtered[roi_obs] == roi_name]
        if sub_cells.empty:
            continue

        comp_img = df_images.loc[roi_name, 'image']
        mask_img = df_images.loc[roi_name, 'mask']

        for i, row in sub_cells.iterrows():
            if ax_idx >= len(axs):
                break
            ax = axs[ax_idx]
            ax_idx += 1

            x_cell = int(round(row[x_loc_obs]))
            y_cell = int(round(row[y_loc_obs]))

            thumb = comp_img[(y_cell - radius):(y_cell + radius),
                             (x_cell - radius):(x_cell + radius), :]

            ax.imshow(thumb)
            if show_gallery_titles:
                ax.set_title(f'{roi_name} - {i}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            # If mask exists, overlay boundary of the center cell
            if mask_img is not None:
                # same bounding region
                thumb_mask = mask_img[(y_cell - radius):(y_cell + radius),
                                      (x_cell - radius):(x_cell + radius)]
                # Ensure shape matches
                if thumb_mask.shape[:2] == thumb.shape[:2]:
                    centre_label = thumb_mask[radius, radius]
                    mask_filtered = np.where(thumb_mask == centre_label, thumb_mask, 0)
                    boundaries = segmentation.find_boundaries(mask_filtered, mode='inner')
                    boundaries = np.ma.masked_where(boundaries == 0, boundaries)
                    ax.imshow(boundaries, cmap='gray', alpha=1, vmin=0, vmax=1)

            # Optional training logic:
            # if training:
            #     answer = input("Enter label for cell, or skip: ")
            #     sub_cells.loc[i, 'training_label'] = answer

        cell_dfs.append(sub_cells)

    vspace, hspace = cell_plot_spacing
    fig.subplots_adjust(hspace=vspace, wspace=hspace)

    if show_gallery_titles:
        fig.suptitle(f"Backgating: {total_cells} cells, radius={radius}")

    out_subdir.mkdir(parents=True, exist_ok=True)

    # Save figure
    cell_fig_path = out_subdir / "Cells.png"
    fig.savefig(cell_fig_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved thumbnails to: {cell_fig_path}")

    # ----------------------------------------------------------------
    # 7) Overview images with bounding boxes
    # ----------------------------------------------------------------
    if overview_images:
        print("Creating overview images with bounding boxes...")
        for roi_name in rois_in_data:
            sub_cells = adata_obs_cells_filtered[adata_obs_cells_filtered[roi_obs] == roi_name]
            if sub_cells.empty:
                continue
            comp_img = df_images.loc[roi_name, 'image'].copy()
            for _, row in sub_cells.iterrows():
                x_cell = int(round(row[x_loc_obs]))
                y_cell = int(round(row[y_loc_obs]))
                rr, cc = rectangle_perimeter(
                    (y_cell - radius, x_cell - radius),
                    extent=(radius * 2, radius * 2),
                    shape=comp_img.shape
                )
                comp_img[rr, cc, :] = 255  # white bounding box

            overview_path = out_subdir / f"{roi_name}_overview.png"
            io.imsave(str(overview_path), img_as_ubyte(comp_img))
            #print(f"  Saved overview for ROI='{roi_name}' -> {overview_path}")

    # ----------------------------------------------------------------
    # 8) Save final CSV of included cells
    # ----------------------------------------------------------------
    cell_dfs = pd.concat(cell_dfs) if cell_dfs else pd.DataFrame()
    csv_path = out_subdir / "cells_list.csv"
    cell_dfs.to_csv(csv_path)
    print(f"Saved list of plotted cells -> {csv_path}")
    print("Backgating completed successfully.")


def get_top_columns(series: pd.Series, top_n: int = 3) -> str:
    """Return the top_n column names by descending value, joined by '__'."""
    top_markers = series.nlargest(top_n).index.tolist()
    return "__".join(top_markers)


def perform_differential_expression(
    adata,
    pop_obs: str,
    target_population: str,
    markers_exclude: Optional[List[str]] = None,
    only_use_markers: Optional[List[str]] = None,
    method: str = 'wilcoxon',
    n_top_markers: int = 3,
    min_logfc_threshold: float = 0.5,
    max_pval_adj: float = 0.05,
    verbose: bool = True
) -> List[str]:
    """
    Perform differential expression analysis to identify most discriminative markers for a population.
    
    This function prioritizes discriminative power (effect size) over statistical significance,
    making it ideal for backgating where visual contrast is more important than statistical rigor.
    
    Args:
        adata: AnnData object with expression data
        pop_obs: Column name in adata.obs containing population labels
        target_population: Population to compare against all others
        markers_exclude: List of markers to exclude from analysis
        only_use_markers: If provided, only consider these markers
        method: Statistical test method ('wilcoxon', 't-test', 'logreg')
        n_top_markers: Number of top markers to return
        min_logfc_threshold: Optional minimum log fold change for quality filtering (0 to disable)
        max_pval_adj: Used for reporting significance status, not for filtering
        verbose: Whether to print detailed results
        
    Returns:
        List of top marker names ranked by discriminative power (test statistic)
    """
    if verbose:
        import logging
        logging.info(f"  Starting differential expression analysis for {target_population}")
        print(f"Starting DE analysis for {target_population}...", flush=True)
        
    if markers_exclude is None:
        markers_exclude = []
    
    # Create a copy to avoid modifying the original
    adata_copy = adata.copy()
    
    if verbose:
        logging.info(f"  Input data: {adata_copy.n_obs} cells, {adata_copy.n_vars} markers")
        logging.info(f"  Population column: {pop_obs}")
        logging.info(f"  Target population: {target_population}")
        logging.info(f"  Markers to exclude: {markers_exclude}")
        logging.info(f"  Method: {method}")
        print(f"DE input: {adata_copy.n_obs} cells, {adata_copy.n_vars} markers", flush=True)
    
    # Filter markers if specified
    if only_use_markers:
        # Only keep specified markers
        keep_markers = [m for m in only_use_markers if m in adata_copy.var_names and m not in markers_exclude]
        adata_copy = adata_copy[:, keep_markers]
    else:
        # Remove excluded markers
        exclude_mask = ~adata_copy.var_names.isin(markers_exclude)
        adata_copy = adata_copy[:, exclude_mask]
    
    if adata_copy.n_vars == 0:
        print(f"Warning: No markers available for differential expression after filtering.")
        return []
    
    # Create binary comparison: target population vs all others
    adata_copy.obs['comparison_group'] = (adata_copy.obs[pop_obs] == target_population).astype('category')
    
    # Perform differential expression
    try:
        sc.tl.rank_genes_groups(
            adata_copy, 
            groupby='comparison_group',
            groups=[True],  # Only test the target population (True) vs others (False)
            reference='rest',
            method=method,
            use_raw=False,
            layer=None,
            pts=True  # Calculate fraction of cells expressing the gene
        )
        
        # Extract results for the target population (True group)
        result = adata_copy.uns['rank_genes_groups']
        
        # Get top markers
        markers_df = pd.DataFrame({
            'names': result['names'][str(True)],
            'scores': result['scores'][str(True)], 
            'logfoldchanges': result['logfoldchanges'][str(True)],
            'pvals': result['pvals'][str(True)],
            'pvals_adj': result['pvals_adj'][str(True)],
            'pts': result['pts'][str(True)],
            'pts_rest': result['pts_rest'][str(True)]
        })
        
        # Sort by discriminative power (score) - prioritize effect size over significance
        # We want the most discriminative markers regardless of statistical significance
        markers_df_sorted = markers_df.sort_values('scores', ascending=False)
        
        # Optionally filter out markers with very low fold changes (for quality)
        if min_logfc_threshold > 0:
            quality_markers = markers_df_sorted[
                markers_df_sorted['logfoldchanges'] > min_logfc_threshold
            ]
            # If filtering removes too many markers, use all markers
            if len(quality_markers) >= n_top_markers:
                markers_df_sorted = quality_markers
            elif verbose:
                print(f"Warning: Only {len(quality_markers)} markers meet logFC > {min_logfc_threshold} threshold. "
                      f"Using all markers ranked by discriminative power.")
        
        # Get top markers by discriminative power
        top_markers = markers_df_sorted.head(n_top_markers)['names'].tolist()
        
        # Count how many are statistically significant (for reporting)
        significant_count = len(markers_df_sorted[
            (markers_df_sorted['pvals_adj'] < max_pval_adj) & 
            (markers_df_sorted['logfoldchanges'] > min_logfc_threshold)
        ])
        
        if verbose:
            import logging
            logging.info(f"\nDifferential expression results for {target_population}:")
            logging.info(f"  Total markers tested: {len(markers_df)}")
            logging.info(f"  Markers meeting significance thresholds (padj < {max_pval_adj}, logFC > {min_logfc_threshold}): {significant_count}")
            logging.info(f"  Top {n_top_markers} most discriminative markers selected: {top_markers}")
            
            if len(top_markers) > 0:
                logging.info(f"\nTop marker details (ranked by discriminative power):")
                top_details = markers_df_sorted.head(n_top_markers)
                for _, row in top_details.iterrows():
                    sig_status = "significant" if (row['pvals_adj'] < max_pval_adj and row['logfoldchanges'] > min_logfc_threshold) else "not significant"
                    logging.info(f"    {row['names']}: score={row['scores']:.2f}, logFC={row['logfoldchanges']:.2f}, "
                          f"padj={row['pvals_adj']:.2e} ({sig_status})")
            
            # Also print to ensure immediate output in SLURM
            print(f"DE analysis for {target_population}: selected {top_markers}", flush=True)
        
        return top_markers
        
    except Exception as e:
        print(f"Error in differential expression analysis: {e}")
        print(f"Falling back to simple mean expression ranking for {target_population}")
        
        # Fallback to simple mean expression
        target_cells = adata_copy[adata_copy.obs[pop_obs] == target_population]
        if target_cells.n_obs > 0:
            mean_expression = pd.Series(
                target_cells.X.mean(axis=0), 
                index=adata_copy.var_names
            )
            return mean_expression.nlargest(n_top_markers).index.tolist()
        else:
            return []


def backgating_assessment(
    adata,
    image_folder: str,
    pop_obs: str,
    mean_expression_file: str = 'markers_mean_expression.csv',
    backgating_settings_file: str = 'backgating_settings.csv',
    pops_list = None,
    cells_per_group: int = 50,
    radius: int = 15,
    roi_obs: str = 'ROI',
    x_loc_obs: str = 'X_loc',
    y_loc_obs: str = 'Y_loc',
    cell_index_obs: str = 'Master_Index',
    # Mask parameters:
    use_masks=True,
    mask_folder='masks',
    exclude_rois_without_mask=True,
    # Subplot spacing for the final "Cells.png":
    cell_plot_spacing=(0.1, 0.1),
    show_gallery_titles=True,
    # Output folder & overview
    output_folder: str = 'Backgating',
    overview_images: bool = True,
    # Intensity scaling
    minimum: float = 0.4,
    max_quantile: str = 'q0.98',
    # Marker filtering
    markers_exclude=None,
    only_use_markers=None,
    number_top_markers: int = 3,
    # Differential expression parameters
    use_differential_expression: bool = True,
    de_method: str = 'wilcoxon',
    min_logfc_threshold: float = 0.5,
    max_pval_adj: float = 0.05,
    # Modes
    mode: str = 'full',  # 'full', 'save_markers', 'load_markers'
    specify_red=None,
    specify_green=None,
    specify_blue=None,
    specify_ranges: bool = True
):
    """
    Perform a backgating assessment on a supplied adata.obs grouping (populations).

    This version:
        1) Saves two CSV files:
           - mean_expression_file (default: markers_mean_expression.csv)
           - backgating_settings_file (default: backgating_settings.csv)
        2) Lets you specify a mask folder (mask_folder) and optionally
           exclude ROIs that have no mask file (exclude_rois_without_mask).
        3) Allows adjusting cell subplot spacing in the final figure via cell_plot_spacing.

    Args:
        adata:            AnnData object (with .obs columns for ROI, x, y, and pop_obs).
        image_folder:     Directory of image subfolders (each ROI is a subfolder).
        pop_obs:          Name of the .obs column defining the population.

        mean_expression_file: CSV with population-level means of markers
        backgating_settings_file: CSV with marker-to-channel assignments + min/max ranges

        pops_list:        Subset of population names to process; if None, uses all found in adata / files.
        cells_per_group:  Number of random cells from each pop to display.

        radius:           Pixel radius for each cell’s bounding box.
        roi_obs,x_loc_obs,y_loc_obs,cell_index_obs:
                          .obs columns for ROI, X location, Y location, and cell ID, respectively.

        use_masks:        Bool or CSV path for ROI->mask mapping (passed to backgating).
        mask_folder:      Folder where we expect <ROI>.tif or <ROI>.tiff if use_masks=True.
        exclude_rois_without_mask:
                          If True, skip all ROIs that have no mask file.

        cell_plot_spacing: (vertical_space, horizontal_space) for subplots (Cells.png).
        output_folder:    Where to store output images/files.
        overview_images:  Whether to save an ROI overview with bounding boxes.
        show_gallery_titles:    Whether to show titles of ROIs and figure in cell gallery.

        minimum, max_quantile:
                          Global intensity clipping parameters for the channel images.

        markers_exclude:  Markers to remove from top marker analysis.
        only_use_markers: If set, only consider these markers for top marker analysis.
        number_top_markers:
                          How many top markers are used (1->Red, 2->R/G, 3->R/G/B).
        
        use_differential_expression:
                          If True, use scanpy differential expression analysis to find most discriminative markers.
                          If False, fall back to simple mean expression ranking.
        de_method:        Statistical method for differential expression ('wilcoxon', 't-test', 'logreg').
        min_logfc_threshold:
                          Optional minimum log fold change for quality filtering (set to 0 to disable).
                          Markers are ranked by discriminative power, not significance.
        max_pval_adj:     Used for reporting significance status, not for filtering markers.

        mode:             One of ['full','save_markers','load_markers'].
                          - 'full': compute means + settings, then run backgating
                          - 'save_markers': compute means + settings only, no images
                          - 'load_markers': load existing settings, then run backgating

        specify_red/green/blue:
                          User overrides for which marker is used for each color channel.
        specify_ranges:   If True, tries to read e.g. 'Red_min','Red_max' etc. from the settings file
                          or fill them if missing.

    Returns:
        None. (Or returns a DataFrame if you adapt it to do so.)
        Saves output to disk: CSV files, images, etc.
    """
    if markers_exclude is None:
        markers_exclude = []
    if only_use_markers is None:
        only_use_markers = []

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_expression_path = out_dir / mean_expression_file
    backgating_settings_path = out_dir / backgating_settings_file

    # 1) Possibly compute & save population mean expression
    if mode in ['full', 'save_markers']:
        pop_categories = adata.obs[pop_obs].unique().tolist()
        mean_df = pd.DataFrame(index=pop_categories, columns=adata.var_names)

        for pop in pop_categories:
            subset = adata[adata.obs[pop_obs] == pop, :]
            mean_df.loc[pop] = subset.X.mean(axis=0)

        mean_df = mean_df.astype(float)

        # Drop excluded markers
        for m in markers_exclude:
            if m in mean_df.columns:
                mean_df.drop(columns=m, inplace=True)

        # If only_use_markers was provided, keep only those
        if only_use_markers:
            keep_cols = [c for c in only_use_markers if c in mean_df.columns]
            mean_df = mean_df[keep_cols]

        mean_df.to_csv(mean_expression_path)
        print(f"Saved population mean expression to: {mean_expression_path}")

    elif mode == 'load_markers':
        # do nothing about mean expression
        pass
    else:
        raise ValueError("mode must be one of ['full','save_markers','load_markers'].")

    # 2) Build or load backgating settings
    if backgating_settings_path.is_file():
        settings_df = pd.read_csv(backgating_settings_path, index_col=0)
        print(f"Loaded existing backgating settings from {backgating_settings_path}")
    else:
        print(f"No existing settings file found; creating a new one at {backgating_settings_path}")
        settings_df = pd.DataFrame()

    # We'll use the columns: 'Red','Green','Blue' + 'Red_min','Red_max', etc.
    needed_columns = [
        'Red','Green','Blue',
        'Red_min','Red_max','Green_min','Green_max','Blue_min','Blue_max'
    ]

    # If we computed mean_df above, we can figure out which populations are relevant
    if mode in ['full','save_markers']:
        mean_df = pd.read_csv(mean_expression_path, index_col=0)
        pop_categories = mean_df.index.tolist()
    else:
        # 'load_markers' mode => we rely on adata and/or settings
        pop_categories = sorted(list(set(adata.obs[pop_obs].unique()).union(settings_df.index)))

    # If user gave a subset of pops
    if pops_list:
        pop_categories = [p for p in pop_categories if p in pops_list]

    # Ensure each pop is in settings_df
    for pop in pop_categories:
        if pop not in settings_df.index:
            settings_df.loc[pop, :] = None

    # Ensure we have needed columns
    for col in needed_columns:
        if col not in settings_df.columns:
            settings_df[col] = None
            
    print(f"Settings DataFrame shape after initialization: {settings_df.shape}")
    print(f"Population categories to process: {pop_categories}")
    if len(pop_categories) > 0:
        print(f"Sample settings for first population ({pop_categories[0]}):")
        print(f"  Red: {settings_df.loc[pop_categories[0], 'Red']}")
        print(f"  Green: {settings_df.loc[pop_categories[0], 'Green']}")  
        print(f"  Blue: {settings_df.loc[pop_categories[0], 'Blue']}")

    # 3) Fill in missing marker assignments
    if mode in ['full','save_markers']:
        print(f"\nDetermining top markers for each population using {'differential expression' if use_differential_expression else 'mean expression'}...", flush=True)
        
        for pop in pop_categories:
            print(f"Processing population: {pop}", flush=True)
            print(f"  Current Red marker: {settings_df.loc[pop, 'Red']}", flush=True)
            print(f"  Is Red marker NaN? {pd.isna(settings_df.loc[pop, 'Red'])}", flush=True)
            
            # In 'full' mode, always recalculate unless Red/Green overrides are specified
            # In 'save_markers' mode, only calculate if missing
            # Note: Blue override (typically DNA1) doesn't prevent DE since Red/Green are what DE selects
            should_calculate = (
                pd.isna(settings_df.loc[pop, 'Red']) or 
                (mode == 'full' and specify_red is None and specify_green is None)
            )
            
            print(f"  Should calculate markers? {should_calculate}", flush=True)
            print(f"  Reason: Red is NaN: {pd.isna(settings_df.loc[pop, 'Red'])}, "
                  f"Full mode with no R/G overrides: {mode == 'full' and specify_red is None and specify_green is None}", flush=True)
            
            if should_calculate:
                print(f"  Running marker selection for {pop}...", flush=True)
                if use_differential_expression:
                    # Use differential expression analysis
                    top_markers = perform_differential_expression(
                        adata=adata,
                        pop_obs=pop_obs,
                        target_population=pop,
                        markers_exclude=markers_exclude,
                        only_use_markers=only_use_markers,
                        method=de_method,
                        n_top_markers=number_top_markers,
                        min_logfc_threshold=min_logfc_threshold,
                        max_pval_adj=max_pval_adj,
                        verbose=True
                    )
                else:
                    # Fallback to mean expression (original method)
                    mean_df = pd.read_csv(mean_expression_path, index_col=0)
                    top_str = get_top_columns(mean_df.loc[pop], number_top_markers)
                    top_markers = top_str.split('__')
                
                # Assign markers to RGB channels
                if len(top_markers) > 0:
                    settings_df.loc[pop, 'Red'] = top_markers[0]
                if len(top_markers) > 1:
                    settings_df.loc[pop, 'Green'] = top_markers[1]
                if len(top_markers) > 2:
                    settings_df.loc[pop, 'Blue'] = top_markers[2]

        # Override if user specified
        if specify_red is not None:
            for pop in pop_categories:
                settings_df.loc[pop, 'Red'] = specify_red
        if specify_green is not None:
            for pop in pop_categories:
                settings_df.loc[pop, 'Green'] = specify_green
        if specify_blue is not None:
            for pop in pop_categories:
                settings_df.loc[pop, 'Blue'] = specify_blue

    elif mode == 'load_markers':
        # Just rely on existing file. If user specified Red/Green/Blue overrides, apply them
        for pop in pop_categories:
            if specify_red is not None:
                settings_df.loc[pop, 'Red'] = specify_red
            if specify_green is not None:
                settings_df.loc[pop, 'Green'] = specify_green
            if specify_blue is not None:
                settings_df.loc[pop, 'Blue'] = specify_blue

    # 4) Fill in missing ranges if specify_ranges
    if specify_ranges:
        for pop in pop_categories:
            # For each color: if min/max is missing, fill from (minimum, max_quantile)
            for color in ['Red','Green','Blue']:
                mn_col = f"{color}_min"
                mx_col = f"{color}_max"
                if pd.isna(settings_df.loc[pop, mn_col]):
                    settings_df.loc[pop, mn_col] = minimum
                if pd.isna(settings_df.loc[pop, mx_col]):
                    settings_df.loc[pop, mx_col] = max_quantile

    # Save the updated settings to CSV
    settings_df.to_csv(backgating_settings_path)
    print(f"Saved backgating settings to: {backgating_settings_path}")

    # 5) If mode='save_markers', we stop here (no imaging).
    if mode == 'save_markers':
        print("Markers saved; no further backgating performed (mode='save_markers').")
        return

    # 6) If we get here, we either 'full' or 'load_markers' => run the actual backgating
    for pop in pop_categories:
        pop_cells = adata.obs[adata.obs[pop_obs] == pop][cell_index_obs]
        if pop_cells.empty:
            print(f"No cells found for population '{pop}'. Skipping.")
            continue

        # Subsample
        cell_sample = pop_cells.sample(min(len(pop_cells), cells_per_group))

        # Grab final channels + ranges from settings_df
        red_marker   = settings_df.loc[pop, 'Red']
        green_marker = settings_df.loc[pop, 'Green']
        blue_marker  = settings_df.loc[pop, 'Blue']

        red_range   = (settings_df.loc[pop, 'Red_min'],   settings_df.loc[pop, 'Red_max'])   if specify_ranges else None
        green_range = (settings_df.loc[pop, 'Green_min'], settings_df.loc[pop, 'Green_max']) if specify_ranges else None
        blue_range  = (settings_df.loc[pop, 'Blue_min'],  settings_df.loc[pop, 'Blue_max'])  if specify_ranges else None

        print(f"\nBackgating population: {pop}")
        print(f"  -> Red={red_marker}, range={red_range}")
        print(f"  -> Green={green_marker}, range={green_range}")
        print(f"  -> Blue={blue_marker}, range={blue_range}")

        # Call your backgating function
        # Make sure it supports the new parameters: mask_folder, exclude_rois_without_mask, cell_plot_spacing
        backgating(
            adata=adata,
            cell_index=list(cell_sample),
            radius=radius,
            image_folder=image_folder,
            # Channels & ranges:
            red=red_marker,
            red_range=red_range,
            green=green_marker,
            green_range=green_range,
            blue=blue_marker,
            blue_range=blue_range,
            # Observations:
            roi_obs=roi_obs,
            x_loc_obs=x_loc_obs,
            y_loc_obs=y_loc_obs,
            cell_index_obs=cell_index_obs,
            # Masks:
            use_masks=use_masks,
            mask_folder=mask_folder,
            exclude_rois_without_mask=exclude_rois_without_mask,
            # Figure layout
            cells_per_row=5,
            cell_plot_spacing=cell_plot_spacing,
            overview_images=overview_images,
            show_gallery_titles=show_gallery_titles,
            # Output
            output_folder=output_folder,
            save_subfolder=clean_text(pop),
            # Clipping
            minimum=minimum,
            max_quantile=max_quantile
        )

    print("\nBackgating assessment complete.\n")