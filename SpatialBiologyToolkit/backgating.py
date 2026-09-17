from __future__ import annotations

import os
import re
import logging
import json
import warnings
from pathlib import Path
from glob import glob
from typing import Dict, List, Union, Tuple, Optional
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
from skimage.measure import find_contours

# Import population overlay function from plotting module
from .plotting import (create_population_overlay, _population_cell_paths,
                       _save_population_overlay_svg, _set_svg_font_family, _svg_font_family,
                       _noninteractive_plotting)


def clean_text(text: str) -> str:
    """Remove special characters to produce safe filenames."""
    if text is None:
        return ""
    # Ensure text is a string (e.g., when population labels are ints)
    text = str(text)
    for ch in ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+',
               '-', '.', '!', '$', '\'', ',', ' ', '/', '"']:
        text = text.replace(ch, '')
    return text


def _normalize_roi_names(roi_names: Optional[List[Union[str, int]]]) -> List[str]:
    """Return ordered, de-duplicated ROI names as strings."""
    normalized = []
    seen = set()
    for roi in roi_names or []:
        if pd.isna(roi):
            continue
        roi_text = str(roi)
        if roi_text in seen:
            continue
        seen.add(roi_text)
        normalized.append(roi_text)
    return normalized


def _select_rois_to_save(
    roi_names: Optional[List[Union[str, int]]],
    max_rois_to_save: Optional[int],
) -> List[str]:
    """
    Reproducibly pseudo-randomly select up to ``max_rois_to_save`` ROI names
    while preserving the original order of the selected ROI subset.
    """
    normalized_rois = _normalize_roi_names(roi_names)
    if max_rois_to_save is None:
        return normalized_rois

    max_rois_to_save = int(max_rois_to_save)
    if max_rois_to_save < 0:
        raise ValueError("max_rois_to_save must be >= 0 or None.")
    if max_rois_to_save == 0:
        return []
    if len(normalized_rois) <= max_rois_to_save:
        return normalized_rois

    rng = np.random.default_rng(0)
    selected_set = set(rng.choice(normalized_rois, size=max_rois_to_save, replace=False).tolist())
    return [roi for roi in normalized_rois if roi in selected_set]


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


def _find_channel_files(
    load_directory: str,
    channel_name: str,
    quiet: bool = False,
    *,
    samples_list: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Find (TIFF path, ROI folder) pairs without loading image pixels."""
    matches = []

    # Find any subdirectories (one level down). If none found, use load_directory itself.
    img_folders = glob(os.path.join(load_directory, "*", "")) or [load_directory]
    if samples_list is not None:
        sample_set = set(_normalize_roi_names(samples_list))
        img_folders = [folder for folder in img_folders if Path(folder).name in sample_set]

    if not quiet:
        logging.info(f'Loading image data for channel "{channel_name}" from ...')

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
            # Split on common separators: underscore, dash, dot, space
            filename_tokens = re.split(r'[_\-\.\s]+', filename_lower)

            # Allow matching single tokens or two-token pairs joined by underscore
            token_pairs = [f"{filename_tokens[i]}_{filename_tokens[i + 1]}"
                           for i in range(len(filename_tokens) - 1)]

            # Check if channel name matches any token or token-pair exactly
            if channel_lower in filename_tokens or channel_lower in token_pairs:
                if not quiet:
                    logging.debug(os.path.join(subfolder, candidate_file))
                matches.append((os.path.join(subfolder, candidate_file), subfolder))
                # Break once we find the first matching file per subfolder.
                break

    if not quiet:
        logging.info('Image data loading completed!')

    return matches


def load_imgs_from_directory(
    load_directory: str, channel_name: str, quiet: bool = False, *,
    samples_list: Optional[List[str]] = None,
) -> Optional[Tuple[List[np.ndarray], List[str], List[str]]]:
    """Load matching TIFFs, filtering ROI folders before reading image data.

    Returns parallel image, filename and folder lists, or None when absent.
    """
    matches = _find_channel_files(load_directory, channel_name, quiet, samples_list=samples_list)
    if not matches:
        logging.warning('No files found with channel name "%s".', channel_name)
        return None
    return ([load_single_img(path) for path, _ in matches],
            [Path(path).name for path, _ in matches], [folder for _, folder in matches])


def _parse_image_maximum(max_val):
    # Resolve the effective channel maximum before deciding which TIFFs to read.
    mode = 'value'
    max_spec = max_val.strip().lower() if isinstance(max_val, str) else max_val
    if isinstance(max_spec, str) and max_spec[:1] in ('q', 'i', 'm', 'x'):
        prefix = max_spec[0]
        try:
            max_quantile = float(max_spec[1:])
        except ValueError:
            raise ValueError(f"Could not parse quantile from '{max_val}'") from None
        if not np.isfinite(max_quantile) or not 0 <= max_quantile <= 1:
            raise ValueError(f"Quantile must be between 0 and 1: '{max_val}'")
        if prefix == 'q':
            mode = 'mean_quantile'
        elif prefix == 'i':
            mode = 'individual_quantile'
        elif prefix == 'm':
            mode = 'minimum_quantile'
        elif prefix == 'x':
            mode = 'max_quantile'
    else:
        try:
            max_value = float(max_spec)
        except (TypeError, ValueError):
            raise ValueError(f"Expected a numeric maximum or q/i/m/x quantile, got {max_val!r}") from None
        if not np.isfinite(max_value):
            raise ValueError(f"Maximum must be finite, got {max_val!r}")

    return mode, max_value if mode == 'value' else max_quantile


def load_rescale_images(
    image_folder: str,
    samples_list: List[str],
    marker: str,
    minimum: float,
    max_val: Union[float, str],
    *,
    save_samples_list: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[str], List[float]]:
    """
    Helper function that:
      1) Loads images for a given marker across provided samples.
      2) Clips intensities using user-specified or quantile-based maxima.
      3) Rescales intensities to [0,1].

    Args:
        image_folder (str): Directory where images (and subfolders) are located.
        samples_list (List[str]): Eligible ROI/sample names. Cohort quantiles
            use this full scope even when only a subset is saved.
        marker (str): The marker (channel name) to load from the image folder.
        minimum (float): Lower clip value.
        max_val (Union[float, str]): A numeric max or a string with prefix:
          - 'q': Mean quantile
          - 'i': Individual quantile
          - 'm': Minimum of quantiles
          - 'x': Maximum of quantiles
          Example: 'q0.97' => Use mean of the 97th percentile for all images.
        save_samples_list (List[str] or None): Subset to return. Fixed numeric
            bounds and individual quantiles read only these ROIs; q/m/x modes
            read the full eligible scope to calculate the shared maximum.

    Returns:
        Tuple[List[np.ndarray], List[str], List[float]]:
            - List of images (each rescaled/clipped)
            - Matching list of ROI names
            - List of max values used per ROI (same order)
    """
    mode, maximum = _parse_image_maximum(max_val)
    max_value = max_quantile = maximum

    eligible_rois = _normalize_roi_names(samples_list)
    output_rois = eligible_rois
    if save_samples_list is not None:
        save_set = set(_normalize_roi_names(save_samples_list))
        output_rois = [roi for roi in eligible_rois if roi in save_set]
    if not output_rois:
        return [], [], []

    cohort_quantile = mode in ('mean_quantile', 'minimum_quantile', 'max_quantile')
    read_rois = eligible_rois if cohort_quantile else output_rois
    if cohort_quantile:
        logging.info(
            "Marker=%s | %s requires the full normalization scope (%d eligible ROIs); "
            "returning up to %d saved ROIs.", marker, max_val, len(read_rois), len(output_rois),
        )
    else:
        logging.info(
            "Marker=%s | %s; reading only %d requested ROI(s).",
            marker, "Fixed bounds (no cohort calculation)" if mode == 'value'
            else "Individual quantiles", len(read_rois),
        )
    loaded = load_imgs_from_directory(image_folder, marker, quiet=True, samples_list=read_rois)
    if not loaded:
        return [], [], []

    image_list, _, folder_list = loaded

    # ROI names are the last part of the subfolder path
    roi_list = [os.path.basename(Path(x)) for x in folder_list]

    # Compute maximum intensities
    if cohort_quantile:
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

        logging.debug(f"Marker={marker} | Mode={mode_str} | Min={minimum:.3f} | "
                      f"Calculated max={max_value:.3f}")
        # Calibration requires all eligible images, but only saved images need rescaling.
        output_set = set(output_rois)
        selected = [(roi, im) for roi, im in zip(roi_list, image_list) if roi in output_set]
        roi_list = [roi for roi, _ in selected]
        image_list = [im for _, im in selected]
        image_list = [im.clip(minimum, max_value) for im in image_list]
        max_values = [max_value] * len(image_list)

    elif mode == 'individual_quantile':
        # Each image is clipped to its own quantile
        max_values = [np.quantile(im, max_quantile) for im in image_list]
        logging.debug(f"Marker={marker} | Mode=Individual quantile {max_quantile} | "
                      f"Min={minimum} | Using image-specific maxima.")
        image_list = [
            im.clip(minimum, mv) for im, mv in zip(image_list, max_values)
        ]

    else:
        # Fixed numeric value
        logging.debug(f"Marker={marker} | Using numeric min={minimum}, max={max_val}")
        image_list = [im.clip(minimum, max_value) for im in image_list]
        max_values = [max_value] * len(image_list)

    # Rescale intensities to [0..1]
    image_list = [exposure.rescale_intensity(i) for i in image_list]

    return image_list, roi_list, max_values


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
    save_samples_list: Optional[List[str]] = None,
    save_subfolder: str = '',
    save_rescale_csv: bool = True,
    rescale_csv_name: str = 'rescale_values.csv'
) -> pd.DataFrame | None:
    """
    Create composite RGB images from up to seven channels. Each channel can be
    mapped onto red/green/blue/magenta/cyan/yellow/white in an additive manner
    (as done by typical multi-channel viewers).
    Processes one ROI and channel at a time. Cohort quantiles use a streaming
    calibration pass, then reread only the ROIs being saved.

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
        save_samples_list (List[str] or None): Optional subset of ROI names to save.
            If None, save all loaded ROIs. Only q/m/x quantile maxima need the
            full ``samples_list`` scope; fixed bounds and individual quantiles
            read only the saved subset.
        save_subfolder (str): Subdirectory under output_folder for saving images.

    Returns:
        DataFrame of rescaling values for saved ROIs if save_rescale_csv is True,
        else None.
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

    # Keep paths and scalar bounds, never a collection of full-resolution images.
    eligible_rois = _normalize_roi_names(samples_list)
    save_set = set(eligible_rois if save_samples_list is None else _normalize_roi_names(save_samples_list))
    sources = {}
    rescale_rows = []
    for color, (marker, color_range) in color_configs.items():
        if marker is None:
            continue
        ch_min, ch_max = color_range if color_range is not None else (minimum, max_quantile)
        mode, maximum = _parse_image_maximum(ch_max)
        cohort_quantile = mode in ('mean_quantile', 'minimum_quantile', 'max_quantile')
        read_scope = eligible_rois if cohort_quantile else [roi for roi in eligible_rois if roi in save_set]
        files = {Path(folder).name: path for path, folder in
                 _find_channel_files(image_folder, marker, quiet=True, samples_list=read_scope)}
        saved_files = {roi: path for roi, path in files.items() if roi in save_set}
        if not saved_files:
            continue
        if cohort_quantile:
            logging.info('Marker=%s | Calibrating %s across %d ROIs one image at a time.',
                         marker, ch_max, len(files))
            quantiles = []
            for path in files.values():
                image = load_single_img(path)
                quantiles.append(np.quantile(image, maximum))
                del image
            reduce = {'mean_quantile': np.mean, 'minimum_quantile': np.min, 'max_quantile': np.max}[mode]
            maximum = float(reduce(quantiles))
            mode = 'value'
        else:
            logging.info('Marker=%s | %s; reading only %d requested ROI(s).',
                         marker, 'Fixed bounds (no cohort calculation)' if mode == 'value'
                         else 'Individual quantiles', len(saved_files))
        sources[color] = (saved_files, ch_min, mode, maximum)

    output_rois = list(dict.fromkeys(roi for files, *_ in sources.values() for roi in files))
    logging.info('Saving composite images for %d ROI(s), one ROI at a time.', len(output_rois))
    components = {'red': (0,), 'green': (1,), 'blue': (2,), 'magenta': (0, 2),
                  'cyan': (1, 2), 'yellow': (0, 1), 'white': (0, 1, 2)}
    for roi_name in output_rois:
        stack = None
        for color, (files, ch_min, mode, maximum) in sources.items():
            if roi_name not in files:
                continue
            image = load_single_img(files[roi_name])
            max_used = float(np.quantile(image, maximum)) if mode == 'individual_quantile' else maximum
            np.clip(image, ch_min, max_used, out=image)
            image = exposure.rescale_intensity(image)
            if stack is None:
                stack = np.zeros((*image.shape, 3), dtype=np.float32)
            if image.shape != stack.shape[:2]:
                raise ValueError(f'Channel image shapes do not match for ROI {roi_name!r}.')
            for component in components[color]:
                plane = stack[:, :, component]
                np.add(plane, image, out=plane)
                np.clip(plane, 0, 1, out=plane)
            del plane, image
            rescale_rows.append(dict(roi=roi_name, channel=color,
                                     marker=str(color_configs[color][0]), min_used=ch_min, max_used=max_used))
        stack_ubyte = img_as_ubyte(stack)
        del stack

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

        # Suppress low contrast warnings when saving images per ROI
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*is a low contrast image')
            io.imsave(str(save_path), stack_ubyte)
        del stack_ubyte

    if save_rescale_csv and rescale_rows:
        rescale_df = pd.DataFrame([row for color in color_configs for row in rescale_rows
                                   if row['channel'] == color])
        rescale_dir = Path(output_folder)
        if save_subfolder:
            rescale_dir = rescale_dir / save_subfolder
            rescale_dir.mkdir(parents=True, exist_ok=True)
        rescale_df.to_csv(rescale_dir / rescale_csv_name, index=False)
        return rescale_df

    return None

def _validate_gallery_options(sampling, max_cells, umap_weight, balance_rois=True):
    if sampling not in ('random', 'intelligent'):
        raise ValueError("gallery_sampling must be 'random' or 'intelligent'.")
    if max_cells is not None and (int(max_cells) != max_cells or max_cells < 0):
        raise ValueError('max_gallery_cells must be a nonnegative integer or None.')
    if not np.isfinite(umap_weight) or not 0 <= umap_weight <= 1:
        raise ValueError('gallery_umap_weight must be between 0 and 1.')
    if not isinstance(balance_rois, (bool, np.bool_)):
        raise ValueError('gallery_balance_rois must be True or False.')


def _select_gallery_cells(
    adata, candidates, max_cells, *, sampling='random', random_state=0,
    layer=None, markers=None, umap_key='X_umap', umap_weight=0.2,
    cell_index_obs='Master_Index', roi_obs='ROI', balance_rois=True,
):
    """Select actual cells nearest the eligible pool's robust median phenotype.

    Expression distance is RMS deviation in per-marker robust scale units.
    Component percentile ranks are combined to avoid comparing expression and
    embedding units directly. Ties retain input order. Sparse matrices are read
    one column at a time, without densifying the complete population matrix.
    Intelligent selection balances ROI counts by taking the best remaining cell
    from each ROI in rounds. Scores always use the population-wide reference.
    """
    from scipy import sparse

    _validate_gallery_options(sampling, max_cells, umap_weight, balance_rois)
    scored = candidates.copy()
    n = len(scored)
    limit = n if max_cells is None else min(int(max_cells), n)
    metadata = dict(method=sampling, random_state=random_state, eligible_cells=n,
                    expression_layer=layer, requested_umap_weight=umap_weight,
                    umap_key=umap_key, effective_umap_weight=0.0,
                    balance_rois=bool(balance_rois),
                    effective_balance_rois=bool(balance_rois and sampling == 'intelligent'),
                    roi_obs=roi_obs,
                    reference='Eligible supplied cells after ROI, image, mask and crop filtering')
    for column in ('gallery_expression_distance', 'gallery_umap_distance',
                   'gallery_score', 'gallery_selection_rank', 'gallery_roi_rank'):
        scored[column] = np.nan

    def finish(selected, eligible):
        metadata['selected_cells'] = len(selected)
        if roi_obs in eligible.columns:
            for label, frame in [('eligible', eligible), ('selected', selected)]:
                counts = frame[roi_obs].dropna().astype(str).value_counts(sort=False)
                metadata[f'{label}_cells_per_roi'] = {roi: int(count) for roi, count in counts.items()}
            metadata['selected_rois'] = len(metadata['selected_cells_per_roi'])
        return selected, metadata

    if sampling == 'random' or not n or limit == 0:
        eligible = scored
        if n > limit:
            scored = scored.sample(n=limit, random_state=random_state)
        return finish(scored, eligible)

    if balance_rois:
        if roi_obs not in scored.columns:
            raise ValueError(f'ROI-balanced intelligent sampling requires candidate column {roi_obs!r}.')
        if scored[roi_obs].isna().any():
            raise ValueError(f'ROI-balanced intelligent sampling requires nonmissing {roi_obs!r} values.')

    identities = pd.Index(adata.obs[cell_index_obs])
    if not identities.is_unique:
        raise ValueError(f'Intelligent sampling requires unique adata.obs[{cell_index_obs!r}].')
    positions = identities.get_indexer(scored[cell_index_obs])
    if np.any(positions < 0):
        raise ValueError('Gallery candidates must have matching cell IDs in adata.obs.')
    if not adata.var_names.is_unique:
        raise ValueError('Intelligent sampling requires unique adata.var_names.')
    if markers is None:
        markers = list(adata.var_names)
    elif isinstance(markers, str):
        markers = [markers]
    else:
        markers = list(dict.fromkeys(markers))
    if not markers:
        raise ValueError('Intelligent sampling requires at least one expression marker.')
    marker_positions = adata.var_names.get_indexer(markers)
    if np.any(marker_positions < 0):
        missing = [marker for marker, pos in zip(markers, marker_positions) if pos < 0]
        raise ValueError(f'Gallery markers not present in adata.var_names: {missing}')
    if layer is not None and layer not in adata.layers:
        raise ValueError(f'Gallery expression layer {layer!r} is not present in adata.layers.')
    # Read one marker vector at a time. AnnData views of all candidates/markers
    # would materialize a complete candidate expression matrix here.
    matrix = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError('Intelligent sampling requires an expression matrix.')
    if sparse.issparse(matrix):
        # Keep sparse row selection efficient; never densify this block.
        matrix = matrix[positions][:, marker_positions].tocsc()
    valid = np.ones(n, dtype=bool)
    sum_squared = np.zeros(n, dtype=float)
    profiles = []
    varying = 0
    for j, marker in enumerate(markers):
        marker_pos = marker_positions[j]
        if sparse.issparse(matrix):
            values = matrix[:, j:j + 1]
        else:
            # Backed AnnData handles unsorted candidate positions via its view.
            view = adata[positions, marker_pos:marker_pos + 1]
            values = view.X if layer is None else view.layers[layer]
        if hasattr(values, 'to_memory'):
            values = values.to_memory()
        if sparse.issparse(values):
            values = values.toarray()
        values = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(values)
        valid &= finite
        if not finite.any():
            raise ValueError(f'Gallery marker {marker!r} has no finite values in eligible cells.')
        reference = values[finite]
        median = float(np.median(reference))
        scale = float(1.4826 * np.median(np.abs(reference - median)))
        scale_method = 'MAD'
        if scale == 0:
            scale = float(np.std(reference))
            scale_method = 'standard_deviation'
        if scale > 0:
            sum_squared += np.where(finite, ((values - median) / scale) ** 2, 0)
            varying += 1
        profiles.append(dict(marker=str(marker), median=median, scale=scale,
                             scale_method=scale_method if scale else 'constant'))
    if not valid.any():
        raise ValueError('No eligible cells have finite expression across all gallery markers.')
    if not valid.all():
        logging.warning('Excluding %d gallery candidates with non-finite expression.', (~valid).sum())
    distances = np.sqrt(sum_squared / max(1, varying))
    distances[~valid] = np.nan
    scored['gallery_expression_distance'] = distances
    expression_ranks = pd.Series(distances).rank(method='average', pct=True).to_numpy()
    combined = expression_ranks.copy()
    weight = 0.0
    if umap_weight > 0:
        if umap_key not in adata.obsm:
            logging.warning('Embedding %r is absent; using expression-only gallery sampling.', umap_key)
        else:
            embedding = adata.obsm[umap_key]
            embedding = embedding.iloc[positions] if isinstance(embedding, pd.DataFrame) else embedding[positions]
            if sparse.issparse(embedding):
                embedding = embedding.toarray()
            embedding = np.asarray(embedding, dtype=float)
            if embedding.ndim != 2 or embedding.shape[1] == 0:
                raise ValueError(f'Gallery embedding {umap_key!r} must be a nonempty 2D array.')
            embedding_valid = np.isfinite(embedding).all(axis=1) & valid
            if embedding_valid.any():
                center = np.median(embedding[embedding_valid], axis=0)
                embedding_distances = np.full(n, np.nan)
                embedding_distances[embedding_valid] = np.linalg.norm(embedding[embedding_valid] - center, axis=1)
                scored['gallery_umap_distance'] = embedding_distances
                # Missing embedding rows receive the worst embedding rank.
                ranks = pd.Series(embedding_distances).rank(method='average', pct=True).fillna(1).to_numpy()
                weight = float(umap_weight)
                combined = (1 - weight) * expression_ranks + weight * ranks
                metadata['umap_median'] = center.tolist()
                if np.any(valid & ~embedding_valid):
                    logging.warning('Some gallery candidates lack finite embedding coordinates; assigning worst UMAP rank.')
            else:
                logging.warning('No finite %r coordinates; using expression-only gallery sampling.', umap_key)
    scored['gallery_score'] = combined
    scored['gallery_selection_rank'] = pd.Series(combined).rank(method='first').to_numpy()
    metadata.update(effective_umap_weight=weight, marker_profiles=profiles,
                    finite_expression_cells=int(valid.sum()))
    # Stable sorting makes tied selections reproducible for the same input order.
    eligible = scored.loc[valid].sort_values('gallery_score', kind='stable')
    if balance_rois:
        # The first round contains each ROI's best cell, the second its next
        # best, etc. Within a partial round, lower global scores win. Exhausted
        # ROIs naturally give their unused places to ROIs with more candidates.
        eligible['gallery_roi_rank'] = eligible.groupby(
            roi_obs, sort=False, observed=True).cumcount() + 1
        selected = eligible.sort_values('gallery_roi_rank', kind='stable').head(limit)
        # Preserve the existing score-rank ordering of this helper's result.
        selected = selected.sort_values('gallery_score', kind='stable')
    else:
        selected = eligible.head(limit)
    return finish(selected, eligible)


@_noninteractive_plotting()
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
    max_gallery_cells: Optional[int] = None,
    # intensity scaling
    minimum=0.2,
    max_quantile='q0.97',
    # ROI scope for downstream plotting
    roi_list=None,
    # image generation scope
    image_samples_list=None,
    # optional interactive training
    training=False,
    gallery_sampling: str = 'random',
    gallery_random_state: int = 0,
    gallery_layer: Optional[str] = None,
    gallery_markers: Optional[List[str]] = None,
    gallery_umap_key: str = 'X_umap',
    gallery_umap_weight: float = 0.2,
    gallery_save_svg: bool = True,
    font_family: str = 'Arial',
    gallery_balance_rois: bool = True,
):
    """
    UPDATED
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

        max_gallery_cells (int or None):
            Optional maximum number of cells to include in the thumbnail gallery.
            When set, cells are sampled only for step 6. Overview images, mask
            overlays, and the saved cell list still use all provided cells.

        gallery_sampling (str):
            'random' (default) reproduces seeded random sampling. 'intelligent'
            selects actual cells nearest the eligible pool's median expression
            profile, with a smaller UMAP-centrality contribution. Call separately
            per population (as backgating_assessment does).
        gallery_random_state (int): Random seed for random sampling, default 0.
        gallery_layer (str or None): Expression layer for intelligent sampling;
            None uses adata.X. Supply appropriately normalized/transformed data.
        gallery_markers (list or None): Markers used for intelligent sampling;
            None uses all adata.var_names. Constant markers do not contribute.
        gallery_umap_key (str): Embedding in adata.obsm, default 'X_umap'.
        gallery_umap_weight (float): Weight on embedding distance percentile rank
            (default 0.2); the remaining weight goes to robust expression distance
            rank. Set 0 for expression only. Missing embeddings fall back to
            expression only with a warning. Cells with non-finite expression are
            excluded; missing embedding rows receive the worst embedding rank.
        gallery_balance_rois (bool): For intelligent sampling, balance counts
            across eligible ROIs by selecting their best-scoring cells in rounds
            (default True). Scores still use the whole eligible population.
            Spare slots from small ROIs are redistributed; partial rounds favour
            lower scores. False restores pooled selection. Random mode is unchanged.
        gallery_save_svg (bool): Also save Cells.svg (default True), with separate
            image, vector outline and editable title groups per thumbnail.
        font_family (str): Single font family used for gallery titles and SVG
            text, default 'Arial'. Font files are not embedded.

        overview_images (bool):
            If True, saves an “overview” with bounding boxes for each cell.

        roi_list (list or None):
            If provided, restrict downstream backgating to these ROIs.

        image_samples_list (list or None):
            Eligible normalization scope (defaults to all dataset ROIs). Only
            q/m/x quantile maxima read this full scope; fixed bounds and
            individual quantiles read only ROIs with the selected cells.

        minimum, max_quantile (float or str):
            Global clipping parameters for channels, used by `make_images`.

        training (bool):
            If True, code can prompt for user input after each cell (placeholder logic).

    Returns:
        None.
        Saves Cells.png, optionally Cells.svg, gallery_cells.csv in display order,
        gallery_sampling.json with selection settings, optional overview images,
        and cells_list.csv containing all eligible supplied cells.
    """
    # ----------------------------------------------------------------
    # 1) Normalize cell_index to a list
    # ----------------------------------------------------------------
    _validate_gallery_options(gallery_sampling, max_gallery_cells, gallery_umap_weight, gallery_balance_rois)
    _svg_font_family(font_family)
    if isinstance(cell_index, (pd.Series, np.ndarray, tuple, set)):
        cell_index = list(cell_index)
    elif not isinstance(cell_index, list):
        cell_index = [cell_index]

    cell_index = list(dict.fromkeys(cell_index))
    adata_obs_cells = adata.obs.loc[adata.obs[cell_index_obs].isin(cell_index)].copy()

    if adata_obs_cells.empty:
        logging.warning("No cells matched the supplied cell_index values. Exiting backgating.")
        return
        
    # List of specific ROIs to process that contain these cells, potentially filtered by user
    if roi_list is None:
        roi_list = adata_obs_cells[roi_obs].unique().tolist()
    else:
        adata_obs_cells = adata_obs_cells[adata_obs_cells[roi_obs].isin(roi_list)]
        roi_list = adata_obs_cells[roi_obs].unique().tolist()

    # Get all ROIs in the dataset for image generation
    all_roi_list = adata.obs[roi_obs].unique().tolist()
    if image_samples_list is None:
        image_samples_list = all_roi_list

    logging.info(f"Backgating on {len(adata_obs_cells)} cells across {len(roi_list)} ROIs.")
    logging.info(
        "Composite image scope: %d eligible ROI(s), %d ROI(s) to save. "
        "Only channels with q/m/x maxima require the full normalization scope.",
        len(_normalize_roi_names(image_samples_list)),
        len(_normalize_roi_names(roi_list)),
    )

    # ----------------------------------------------------------------
    # 2) Build composite images for each ROI (from your existing function)
    # ----------------------------------------------------------------
    logging.info("Creating composite images via `make_images` ...")

    make_images(
        image_folder=image_folder,
        samples_list=image_samples_list,
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
        save_samples_list=roi_list,
        save_subfolder=save_subfolder
    )
    logging.info("Composite images created.")

    # ----------------------------------------------------------------
    # 3) Load those composite images from disk into a DataFrame
    # ----------------------------------------------------------------
    out_subdir = Path(output_folder) / save_subfolder
    from PIL import Image
    records = []
    for roi_name in roi_list:
        roi_path = out_subdir / f"{roi_name}.png"
        if roi_path.exists():
            # PNG headers give dimensions without decoding/storing pixel arrays.
            with Image.open(roi_path) as image_header:
                width, height = image_header.size
            records.append({roi_obs: roi_name, 'image_path': roi_path,
                            'x_length': width, 'y_length': height})
        else:
            records.append({roi_obs: roi_name, 'image_path': None,
                            'x_length': 0, 'y_length': 0})
    df_images = pd.DataFrame(records, columns=[roi_obs, 'image_path', 'x_length', 'y_length'])
    df_images = df_images.drop_duplicates(subset=roi_obs).set_index(roi_obs)
    df_images['mask_path'] = pd.Series(None, index=df_images.index, dtype=object)
    logging.info('Indexed composite image paths for %d ROIs without loading pixels.', len(df_images))

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
                    df_images.at[roi_name, 'mask_path'] = potential_tif
                    #print(f"  Loaded mask for ROI='{roi_name}' -> {potential_tif}")
                elif potential_tiff.is_file():
                    #print(roi_name)
                    df_images.at[roi_name, 'mask_path'] = potential_tiff
                    #print(f"  Loaded mask for ROI='{roi_name}' -> {potential_tiff}")
                else:
                    rois_to_exclude.append(roi_name)

        elif isinstance(use_masks, str):
            # user passed a CSV mapping ROI->mask path
            mask_csv = Path(use_masks)
            if not mask_csv.is_file():
                logging.warning(f"mask CSV file '{mask_csv}' not found. No masks loaded.")
            else:
                logging.info(f"Loading mask mappings from CSV: {mask_csv}")
                mask_df = pd.read_csv(mask_csv).set_index(roi_obs)
                for roi_name in df_images.index:
                    if roi_name in mask_df.index:
                        mask_path = Path(mask_df.loc[roi_name, 'mask_path'])
                        if mask_path.is_file():
                            df_images.at[roi_name, 'mask_path'] = mask_path
                            logging.debug(f"  Loaded mask for ROI='{roi_name}' -> {mask_path}")
                        else:
                            rois_to_exclude.append(roi_name)
                    else:
                        rois_to_exclude.append(roi_name)

        if rois_to_exclude:
            logging.warning("The following ROIs do NOT have matching mask files:")
            for r_ in rois_to_exclude:
                logging.warning(f"    - {r_}")
            if exclude_rois_without_mask:
                logging.warning("These ROIs will be excluded due to `exclude_rois_without_mask=True`.")
                df_images.drop(rois_to_exclude, inplace=True)

    # Now filter cells in adata_obs_cells if their ROI was excluded
    valid_rois = df_images.index.tolist()
    before_count = len(adata_obs_cells)
    adata_obs_cells = adata_obs_cells[adata_obs_cells[roi_obs].isin(valid_rois)]
    after_count = len(adata_obs_cells)
    if before_count != after_count:
        logging.info(f"Excluded {before_count - after_count} cells whose ROI lacked masks (or images).")

    # ----------------------------------------------------------------
    # 5) Filter out cells that would be out-of-bounds for the given radius
    # ----------------------------------------------------------------
    adata_obs_cells['x_max'] = adata_obs_cells[roi_obs].map(df_images['x_length'])
    adata_obs_cells['y_max'] = adata_obs_cells[roi_obs].map(df_images['y_length'])

    x, y = adata_obs_cells[x_loc_obs], adata_obs_cells[y_loc_obs]
    adata_obs_cells['in_range'] = (
        (x - radius >= 0) & (x + radius < adata_obs_cells['x_max']) &
        (y - radius >= 0) & (y + radius < adata_obs_cells['y_max'])
    ).fillna(False)
    adata_obs_cells_filtered = adata_obs_cells[adata_obs_cells['in_range']].copy()
    out_of_bounds_count = len(adata_obs_cells) - len(adata_obs_cells_filtered)
    logging.info(f"{out_of_bounds_count} cells are out-of-bounds for plotting.")
    logging.info(f"Proceeding with {len(adata_obs_cells_filtered)} cells.")

    if len(adata_obs_cells_filtered) == 0:
        logging.warning("No valid cells remain to plot. Exiting backgating.")
        return

    # ----------------------------------------------------------------
    # 6) Plot cell thumbnails in a big figure
    # ----------------------------------------------------------------
    out_subdir.mkdir(parents=True, exist_ok=True)

    gallery_cells, gallery_metadata = _select_gallery_cells(
        adata, adata_obs_cells_filtered, max_gallery_cells,
        sampling=gallery_sampling, random_state=gallery_random_state,
        layer=gallery_layer, markers=gallery_markers,
        umap_key=gallery_umap_key, umap_weight=gallery_umap_weight,
        cell_index_obs=cell_index_obs,
        roi_obs=roi_obs, balance_rois=gallery_balance_rois,
    )

    gallery_cells = gallery_cells.sort_values([roi_obs, cell_index_obs])
    total_gallery_cells = len(gallery_cells)
    gallery_cells['gallery_display_order'] = np.arange(1, total_gallery_cells + 1)
    gallery_cells.to_csv(out_subdir / 'gallery_cells.csv')
    gallery_metadata['selected_cells'] = total_gallery_cells
    (out_subdir / 'gallery_sampling.json').write_text(
        json.dumps(gallery_metadata, indent=2), encoding='utf-8')
    logging.info('Selected %d of %d eligible gallery cells using %s sampling.',
                 total_gallery_cells, len(adata_obs_cells_filtered), gallery_sampling)

    overview_written = set()

    def write_overview(roi_name, comp_img):
        # Called only after thumbnail artists have copied their small crops.
        for _, row in adata_obs_cells_filtered.loc[adata_obs_cells_filtered[roi_obs] == roi_name].iterrows():
            x_cell, y_cell = int(round(row[x_loc_obs])), int(round(row[y_loc_obs]))
            rr, cc = rectangle_perimeter((y_cell - radius, x_cell - radius),
                                        extent=(radius * 2, radius * 2), shape=comp_img.shape)
            comp_img[rr, cc, :] = 255
        io.imsave(str(out_subdir / f'{roi_name}_overview.png'), img_as_ubyte(comp_img))
        overview_written.add(roi_name)

    if total_gallery_cells == 0:
        logging.warning("No cells selected for the thumbnail gallery; skipping Cells.png.")
    else:
        rows = ceil(total_gallery_cells / cells_per_row)
        fig, axs = plt.subplots(rows, cells_per_row, figsize=(10, rows * 2), dpi=100)
        axs = axs.flatten() if (rows > 1 or cells_per_row > 1) else [axs]

        ax_idx = 0
        gallery_layers = {}

        logging.info(
            "Plotting thumbnail gallery (%d per row) for %d cells.",
            cells_per_row,
            total_gallery_cells,
        )

        gallery_rois = gallery_cells[roi_obs].unique().tolist()

        for roi_name in gallery_rois:
            sub_cells = gallery_cells[gallery_cells[roi_obs] == roi_name]
            if sub_cells.empty:
                continue

            comp_img = io.imread(str(df_images.loc[roi_name, 'image_path']))
            mask_path = df_images.loc[roi_name, 'mask_path']
            mask_img = io.imread(str(mask_path)) if pd.notna(mask_path) else None
            thumb_mask = None

            for i, row in sub_cells.iterrows():
                if ax_idx >= len(axs):
                    break
                ax = axs[ax_idx]
                ax_idx += 1

                x_cell = int(round(row[x_loc_obs]))
                y_cell = int(round(row[y_loc_obs]))

                thumb = comp_img[(y_cell - radius):(y_cell + radius),
                                 (x_cell - radius):(x_cell + radius), :]

                image_gid = f'gallery_image_{ax_idx}'
                ax.imshow(thumb, interpolation='none', gid=image_gid)
                cell_description = f'{roi_name}: cell {row[cell_index_obs]}'
                gallery_layers[image_gid] = f'Image {ax_idx} ({cell_description})'
                if show_gallery_titles:
                    title = ax.set_title(f'{roi_name} - {i}', fontsize=8, fontfamily=font_family)
                    title_gid = f'gallery_title_{ax_idx}'
                    title.set_gid(title_gid)
                    gallery_layers[title_gid] = f'Title {ax_idx} ({cell_description})'
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
                        if centre_label != 0:
                            from matplotlib.patches import PathPatch
                            for _, path in _population_cell_paths(thumb_mask, {centre_label}):
                                outline_gid = f'gallery_outline_{ax_idx}'
                                ax.add_patch(PathPatch(path, facecolor='none', edgecolor='white',
                                                       linewidth=1, gid=outline_gid))
                                gallery_layers[outline_gid] = f'Outline {ax_idx} ({cell_description})'

                # Optional training logic:
                # if training:
                #     answer = input("Enter label for cell, or skip: ")
                #     sub_cells.loc[i, 'training_label'] = answer

            if overview_images:
                write_overview(roi_name, comp_img)
            # Crop views must not keep the full ROI or mask alive.
            del comp_img, mask_img, thumb, thumb_mask

        for ax in axs[ax_idx:]:
            ax.axis('off')

        vspace, hspace = cell_plot_spacing
        fig.subplots_adjust(hspace=vspace, wspace=hspace)

        if show_gallery_titles:
            fig.suptitle(f"Backgating: {total_gallery_cells} cells, radius={radius}",
                         gid='gallery_heading', fontfamily=font_family)
            gallery_layers['gallery_heading'] = 'Gallery heading'

        fig.canvas.draw()
        for ax in axs[:ax_idx]:
            points_per_pixel = abs(ax.transData.transform((1, 0))[0] -
                                   ax.transData.transform((0, 0))[0]) * 72 / fig.dpi
            for patch in ax.patches:
                patch.set_linewidth(points_per_pixel)

        # Save figure
        cell_fig_path = out_subdir / "Cells.png"
        try:
            fig.savefig(cell_fig_path, bbox_inches='tight', dpi=200)
            if gallery_save_svg:
                _save_population_overlay_svg(fig, out_subdir / 'Cells.svg',
                                             layers=gallery_layers, font_family=font_family,
                                             bbox_inches='tight', dpi=200)
        finally:
            plt.close(fig)
        logging.info(f"Saved thumbnails to: {cell_fig_path}")

    # ----------------------------------------------------------------
    # 7) Overview images with bounding boxes
    # ----------------------------------------------------------------
    if overview_images:
        logging.info("Creating overview images with bounding boxes...")
        for roi_name in adata_obs_cells_filtered[roi_obs].unique():
            if roi_name not in overview_written:
                comp_img = io.imread(str(df_images.loc[roi_name, 'image_path']))
                write_overview(roi_name, comp_img)
                del comp_img

    # ----------------------------------------------------------------
    # 8) Save final CSV of included cells
    # ----------------------------------------------------------------
    csv_path = out_subdir / "cells_list.csv"
    adata_obs_cells_filtered.to_csv(csv_path)
    logging.info(f"Saved list of plotted cells -> {csv_path}")
    logging.info("Backgating completed successfully.")


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
        logging.info(f"Starting differential expression analysis for {target_population}")
        logging.info(f"Starting DE analysis for {target_population}...")
        
    if markers_exclude is None:
        markers_exclude = []
    
    # Create a copy to avoid modifying the original
    adata_copy = adata.copy()
    
    if verbose:
        logging.info(f"  Input data: {adata_copy.n_obs} cells, {adata_copy.n_vars} markers")
        logging.info(f"  Population column: {pop_obs}")
        logging.info(f"  Target population: {target_population} (type: {type(target_population)})")
        logging.info(f"  Markers to exclude: {markers_exclude}")
        logging.info(f"  Method: {method}")
        
        # Debug: check population distribution and types
        pop_counts = adata_copy.obs[pop_obs].value_counts()
        logging.info(f"  Population distribution (top 10): {dict(pop_counts.head(10))}")
        
        # Check data types
        sample_values = adata_copy.obs[pop_obs].head(5).tolist()
        logging.info(f"  Sample population values: {sample_values} (types: {[type(x) for x in sample_values]})")
        
        target_count = (adata_copy.obs[pop_obs].astype(str) == str(target_population)).sum()
        logging.info(f"  Cells in target population '{target_population}': {target_count}")
        logging.info(f"DE input: {adata_copy.n_obs} cells, {adata_copy.n_vars} markers")
        logging.info(f"Target population '{target_population}' (type: {type(target_population)}) has {target_count} cells")
    
    # Filter markers if specified
    if only_use_markers:
        # Only keep specified markers
        keep_markers = [m for m in only_use_markers if m in adata_copy.var_names and m not in markers_exclude]
        adata_copy = adata_copy[:, keep_markers]
    else:
        # Remove excluded markers
        exclude_mask = ~adata_copy.var_names.isin(markers_exclude)
        adata_copy = adata_copy[:, exclude_mask]

    # Ensure we are working on a full copy (not a view) before adding columns
    adata_copy = adata_copy.copy()
    
    if adata_copy.n_vars == 0:
        logging.warning("No markers available for differential expression after filtering.")
        return []
    
    # Create binary comparison: target population vs all others
    # Convert target_population to string to match data type in obs
    target_population_str = str(target_population)
    
    # Use string labels instead of boolean for better scanpy compatibility
    labels = np.where(
        adata_copy.obs[pop_obs].astype(str) == target_population_str,
        'target',
        'rest'
    )
    comparison_series = pd.Series(
        labels,
        index=adata_copy.obs_names,
        name='comparison_group'
    ).astype('category')
    adata_copy.obs['comparison_group'] = comparison_series
    
    # Check group distribution
    group_counts = adata_copy.obs['comparison_group'].value_counts()
    if verbose:
        logging.info(f"Group distribution: {dict(group_counts)}")
    
    # Ensure we have both groups and target group has cells
    if len(group_counts) < 2:
        if verbose:
            logging.warning("Only one group found. Cannot perform differential expression.")
        raise ValueError("Insufficient groups for comparison")
    
    if 'target' not in group_counts.index or group_counts.get('target', 0) == 0:
        if verbose:
            logging.warning(f"Target population '{target_population}' has no cells. Cannot perform DE.")
        raise ValueError("Target population has no cells")
    
    # Ensure minimum cells per group for statistical power
    min_cells = 5
    if any(count < min_cells for count in group_counts.values):
        if verbose:
            logging.warning(f"Some groups have fewer than {min_cells} cells. DE may be unreliable.")
    
    # Perform differential expression
    try:
        sc.settings.verbosity = 4
        
        sc.tl.rank_genes_groups(
            adata_copy, 
            groupby='comparison_group',
            groups=['target'],  # Only test the target population vs rest
            reference='rest',
            method=method,
            use_raw=False,
            layer=None,
            pts=True  # Calculate fraction of cells expressing the gene
        )
        
        # Extract results for the target population
        result = adata_copy.uns['rank_genes_groups']
        
        # Check if results exist and are not empty
        # Handle structured arrays properly
        try:
            # Get the column names (group names) from the structured array
            if hasattr(result['names'], 'dtype') and result['names'].dtype.names:
                # This is a structured array - get available group names
                available_groups = result['names'].dtype.names
            else:
                # This might be a dictionary-like structure
                available_groups = list(result['names'].keys()) if hasattr(result['names'], 'keys') else []
            
            if 'target' not in available_groups:
                if verbose:
                    logging.warning(f"No differential expression results found for target group")
                    logging.warning(f"Available groups in results: {available_groups}")
                raise ValueError("Target group not found in DE results")
            
            # Extract data for target group from structured arrays
            names_data = result['names']['target'] if hasattr(result['names'], 'dtype') else result['names']['target']
            scores_data = result['scores']['target'] if hasattr(result['scores'], 'dtype') else result['scores']['target']
            logfc_data = result['logfoldchanges']['target'] if hasattr(result['logfoldchanges'], 'dtype') else result['logfoldchanges']['target']
            pvals_data = result['pvals']['target'] if hasattr(result['pvals'], 'dtype') else result['pvals']['target']
            pvals_adj_data = result['pvals_adj']['target'] if hasattr(result['pvals_adj'], 'dtype') else result['pvals_adj']['target']
            pts_data = result['pts']['target'] if hasattr(result['pts'], 'dtype') else result['pts']['target']
            pts_rest_data = result['pts_rest']['target'] if hasattr(result['pts_rest'], 'dtype') else result['pts_rest']['target']
            
            if len(names_data) == 0:
                if verbose:
                    logging.warning(f"No markers found in differential expression results")
                raise ValueError("No markers in DE results")
            
        except (KeyError, TypeError, AttributeError) as e:
            if verbose:
                logging.warning(f"Error accessing DE results structure: {e}")
                logging.warning(f"Result structure: {type(result)}")
                if hasattr(result, 'keys'):
                    logging.warning(f"Result keys: {list(result.keys())}")
            raise ValueError(f"Cannot access DE results: {e}")
        
        # Get top markers
        markers_df = pd.DataFrame({
            'names': names_data,
            'scores': scores_data, 
            'logfoldchanges': logfc_data,
            'pvals': pvals_data,
            'pvals_adj': pvals_adj_data,
            'pts': pts_data,
            'pts_rest': pts_rest_data
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
                logging.warning(f"Only {len(quality_markers)} markers meet logFC > {min_logfc_threshold} threshold. "
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
            
            # Log selected markers
            logging.info(f"DE analysis for {target_population}: selected {top_markers}")
        
        return top_markers
        
    except Exception as e:
        logging.error(f"Error in differential expression analysis: {e}")
        logging.info(f"Falling back to simple mean expression ranking for {target_population}")
        
        if verbose:
            import traceback
            logging.error("Full error traceback:")
            logging.error(traceback.format_exc())
            
            # Additional debugging information
            logging.info("Debug info:")
            logging.info(f"  adata_copy shape: {adata_copy.shape}")
            logging.info(f"  Target population '{target_population}' count: {(adata_copy.obs[pop_obs].astype(str) == str(target_population)).sum()}")
            logging.info(f"  Comparison group value counts: {adata_copy.obs['comparison_group'].value_counts()}")
        
        # Fallback to simple mean expression
        target_cells = adata_copy[adata_copy.obs[pop_obs].astype(str) == str(target_population)]
        if target_cells.n_obs > 0:
            mean_expression = pd.Series(
                target_cells.X.mean(axis=0), 
                index=adata_copy.var_names
            )
            fallback_markers = mean_expression.nlargest(n_top_markers).index.tolist()
            if verbose:
                logging.info(f"Fallback mean expression markers: {fallback_markers}")
            return fallback_markers
        else:
            logging.warning(f"No cells found for population {target_population}")
            return []


@_noninteractive_plotting()
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
    object_index_obs: str = 'ObjectNumber',
    # Mask parameters:
    use_masks=True,
    mask_folder='masks',
    max_rois_to_save: Optional[int] = None,
    exclude_rois_without_mask=True,
    # Subplot spacing for the final "Cells.png":
    cell_plot_spacing=(0.1, 0.1),
    show_gallery_titles=False,
    # Output folder & overview
    output_folder: str = 'Backgating',
    overview_images: bool = False,
    population_overlays: bool = True,  # New parameter for population overlay visualizations
    population_overlay_outline_width: int = 1,
    population_overlay_legend_fontsize: int = 24,
    population_overlay_show_legend: bool = True,
    population_overlay_show_label: bool = True,
    population_overlay_show_population_label: bool = True,
    population_overlay_label_text: str | dict | None = None,
    population_overlay_label_fontsize: int | None = None,
    population_overlay_crop_size: tuple[int, int] | None = (300,300),
    population_overlay_crop_origin: str = "intelligent",
    population_overlay_show_scale_bar: bool = True,
    population_overlay_scale_bar_length: int = 50,
    population_overlay_scale_bar_thickness: int = 3,
    population_overlay_scale_bar_color: str = "white",
    population_overlay_scale_bar_outline_thickness: int = 3,
    population_overlay_scale_bar_text: str | None = None,
    population_overlay_scale_bar_text_size: int = 10,
    population_overlay_extension: str = "png",
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
    verbose: bool = True,
    # Modes
    mode: str = 'full',  # 'full', 'save_markers', 'load_markers'
    specify_red=None,
    specify_green=None,
    specify_blue=None,
    specify_ranges: bool = True,
    population_overlay_save_svg: bool = False,
    gallery_sampling: str = 'random',
    gallery_random_state: int = 0,
    gallery_layer: Optional[str] = None,
    gallery_markers: Optional[List[str]] = None,
    gallery_umap_key: str = 'X_umap',
    gallery_umap_weight: float = 0.2,
    gallery_save_svg: bool = True,
    font_family: str = 'Arial',
    gallery_balance_rois: bool = True,
    population_overlay_comparison_images=None,
    population_overlay_primary_title: Optional[str] = 'IMC',
    population_overlay_title_fontsize: Optional[float] = None,
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
        cells_per_group:  Maximum number of sampled cells from each pop to show
                  in the thumbnail gallery. Other backgating outputs use
                  all supplied population cells in the selected ROIs.
        gallery_sampling: 'random' (default) or 'intelligent'. Intelligent sampling
            selects cells closest to the eligible population's median phenotype.
        gallery_random_state: Seed for random gallery sampling, default 0.
        gallery_layer: Expression layer for intelligent sampling; None uses adata.X.
        gallery_markers: Marker subset for intelligent sampling; None uses all variables.
        gallery_umap_key: Embedding key in adata.obsm, default 'X_umap'.
        gallery_umap_weight: Weight of UMAP-centrality rank, default 0.2; expression
            rank receives the remaining weight. Missing embeddings use expression only.
        gallery_balance_rois: Balance intelligent selections across eligible saved
            ROIs (default True), taking each ROI's best cells in rounds. False
            restores pooled selection. Does not change random sampling or the
            ROI subset controlled by max_rois_to_save.
        gallery_save_svg: Save layered Cells.svg alongside Cells.png (default True).
            Selection scores and settings are saved in gallery_cells.csv and
            gallery_sampling.json. See backgating for detailed selection behavior.
        font_family: Single font family for thumbnail titles and population-overlay
            labels, legends and scale-bar text, default 'Arial'. SVG text stays editable.

        radius:           Pixel radius for each cell’s bounding box.
        roi_obs,x_loc_obs,y_loc_obs,cell_index_obs:
                          .obs columns for ROI, X location, Y location, and cell ID, respectively.

        use_masks:        Bool or CSV path for ROI->mask mapping (passed to backgating).
        mask_folder:      Folder where we expect <ROI>.tif or <ROI>.tiff if use_masks=True.
        max_rois_to_save: Maximum number of ROIs to save per population.
                          If None, save all population ROIs. Only q/m/x quantile
                          maxima read all dataset ROIs for normalization; fixed
                          bounds and individual quantiles read the saved subset.
        exclude_rois_without_mask:
                          If True, skip all ROIs that have no mask file.

        cell_plot_spacing: (vertical_space, horizontal_space) for subplots (Cells.png).
        output_folder:    Where to store output images/files.
        overview_images:  Whether to save an ROI overview with bounding boxes.
        population_overlays: Whether to create population overlay visualizations showing all cells
                          of each population type with mask contours on composite images.
        population_overlay_legend_fontsize: Font size for population overlay legend labels.
        population_overlay_show_legend: Enable/disable legend on population overlay images.
        population_overlay_show_label: Enable/disable the top-left label box.
        population_overlay_show_population_label: Include the population label text in the box.
        population_overlay_label_text: Optional string or dict mapping ROI -> text.
        population_overlay_label_fontsize: Optional override for label font size (defaults to legend size).
        population_overlay_crop_size: Optional crop size (width, height) in pixels for overlays.
        population_overlay_crop_origin: Crop origin anchor: "upper_left", "upper_right",
            "lower_left", "lower_right", or "center".
        population_overlay_show_scale_bar: Whether to draw a scale bar in overlays.
        population_overlay_scale_bar_length: Scale bar length in pixels.
        population_overlay_scale_bar_thickness: Scale bar thickness in pixels.
        population_overlay_scale_bar_color: Scale bar color.
        population_overlay_scale_bar_outline_thickness: Scale bar outline thickness in pixels.
        population_overlay_scale_bar_text: Optional text displayed above the scale bar.
        population_overlay_scale_bar_text_size: Font size for scale bar text.
        population_overlay_extension: Overlay image extension (default 'png').
            'svg' saves a layered SVG plus a PNG preview for overlay galleries.
        population_overlay_save_svg: Also save a layered SVG beside each overlay.
            The source image, individual cell outlines, scale bar, scale-bar text,
            marker legend and population label are separate editable groups.
            Illustrator may show these groups beneath a single native layer.
        population_overlay_comparison_images: List of image folders or panel dicts
            with folder, title, legend (label to colour string or RGB in 0..255),
            interpolation ('bilinear' or 'nearest'), and show_cell_outlines
            (default False). ROI-matched images are resized to the IMC grid and
            share its crop; they must have the same tissue extent/orientation.
            Titles and legends remain editable in SVG. Missing or ambiguous
            matches show placeholders; a .comparisons.json sidecar records matches.
        population_overlay_primary_title: IMC panel title when comparisons are
            enabled (default 'IMC'); None or an empty string hides the title.
        population_overlay_title_fontsize: Shared font size in points for the
            IMC and comparison panel titles, independent of legend text. None
            uses population_overlay_legend_fontsize for backward compatibility.
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
                          - 'load_markers': load existing settings without rewriting
                            the source CSV, then run backgating. Overrides and
                            missing-range defaults apply only to this run.

        specify_red/green/blue:
                          User overrides for which marker is used for each color channel.
                          When a marker changes, its previous range in the same
                          population follows it; new or ambiguous markers use
                          the global defaults, never another marker's limits.
        specify_ranges:   If True, tries to read e.g. 'Red_min','Red_max' etc. from the settings file
                          or fill them if missing.

    Returns:
        None. (Or returns a DataFrame if you adapt it to do so.)
        Saves output to disk: CSV files, images, etc.
    """
    _validate_gallery_options(gallery_sampling, cells_per_group, gallery_umap_weight, gallery_balance_rois)
    _svg_font_family(font_family)
    from ._overlay_comparisons import prepare_comparison_images
    comparison_sources = prepare_comparison_images(population_overlay_comparison_images) if (
        population_overlays and mode != 'save_markers'
    ) else []
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
        # Ensure population categories are preserved as strings to avoid type conversion issues
        pop_categories = adata.obs[pop_obs].astype(str).unique().tolist()
        logging.info(f"Found population categories: {pop_categories} (types: {[type(x) for x in pop_categories]})")
        mean_df = pd.DataFrame(index=pop_categories, columns=adata.var_names)

        for pop in pop_categories:
            # Ensure consistent string comparison to avoid type mismatch issues
            subset = adata[adata.obs[pop_obs].astype(str) == str(pop), :]
            logging.debug(f"Population '{pop}': found {subset.n_obs} cells")
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
        logging.info(f"Saved population mean expression to: {mean_expression_path}")

    elif mode == 'load_markers':
        # do nothing about mean expression
        pass
    else:
        raise ValueError("mode must be one of ['full','save_markers','load_markers'].")

    # 2) Build or load backgating settings
    if backgating_settings_path.is_file():
        settings_df = pd.read_csv(backgating_settings_path, index_col=0)
        settings_df.index = settings_df.index.map(str)
        logging.info(f"Loaded existing backgating settings from {backgating_settings_path}")
        if mode == 'full':
            logging.warning(
                "mode='full' can reselect markers and overwrite %s. Use "
                "mode='load_markers' to plot with existing settings unchanged.",
                backgating_settings_path,
            )
    elif mode == 'load_markers':
        raise FileNotFoundError(
            f"Settings file not found: {backgating_settings_path}. "
            "Create it with mode='save_markers' before using mode='load_markers'."
        )
    else:
        logging.info(f"No existing settings file found; creating a new one at {backgating_settings_path}")
        settings_df = pd.DataFrame()

    # We'll use the columns: 'Red','Green','Blue' + 'Red_min','Red_max', etc.
    needed_columns = [
        'Red','Green','Blue',
        'Red_min','Red_max','Green_min','Green_max','Blue_min','Blue_max'
    ]

    # If we computed mean_df above, we can figure out which populations are relevant
    if mode in ['full','save_markers']:
        pop_categories = mean_df.index.tolist()
    else:
        # 'load_markers' mode => we rely on adata and/or settings
        # Ensure consistent string handling
        adata_pops = adata.obs[pop_obs].astype(str).unique().tolist()
        settings_pops = [str(x) for x in settings_df.index.tolist()]
        pop_categories = sorted(list(set(adata_pops).union(settings_pops)))
        logging.info(f"Load mode - found population categories: {pop_categories}")

    # If user gave a subset of pops
    if pops_list:
        requested_pops = {str(p) for p in pops_list}
        pop_categories = [p for p in pop_categories if p in requested_pops]

    if mode == 'load_markers':
        missing_pops = [pop for pop in pop_categories if pop not in settings_df.index]
        if missing_pops:
            raise ValueError(
                f"No saved backgating settings for populations: {missing_pops}. "
                "Create their settings with mode='save_markers' first, or restrict pops_list."
            )

    # Ensure we have needed columns
    for col in needed_columns:
        if col not in settings_df.columns:
            settings_df[col] = None
        # Ranges can contain numeric bounds or quantile strings.
        settings_df[col] = settings_df[col].astype(object)

    # Ensure each pop is in settings_df (after creating columns for new templates).
    for pop in pop_categories:
        if pop not in settings_df.index:
            settings_df.loc[pop, :] = None

    original_settings = settings_df.copy(deep=True)
            
    logging.info(f"Settings DataFrame shape after initialization: {settings_df.shape}")
    logging.info(f"Population categories to process: {pop_categories}")
    if len(pop_categories) > 0:
        logging.debug(f"Sample settings for first population ({pop_categories[0]}):")
        logging.debug(f"  Red: {settings_df.loc[pop_categories[0], 'Red']}")
        logging.debug(f"  Green: {settings_df.loc[pop_categories[0], 'Green']}")  
        logging.debug(f"  Blue: {settings_df.loc[pop_categories[0], 'Blue']}")

    # 3) Fill in missing marker assignments
    if mode in ['full','save_markers']:
        logging.info(f"Determining top markers for each population using {'differential expression' if use_differential_expression else 'mean expression'}...")
        
        for pop in pop_categories:
            logging.info(f"Processing population: {pop}")
            logging.debug(f"  Current Red marker: {settings_df.loc[pop, 'Red']}")
            logging.debug(f"  Is Red marker NaN? {pd.isna(settings_df.loc[pop, 'Red'])}")
            
            # In 'full' mode, always recalculate unless Red/Green overrides are specified
            # In 'save_markers' mode, only calculate if missing
            # Note: Blue override (typically DNA1) doesn't prevent DE since Red/Green are what DE selects
            should_calculate = (
                pd.isna(settings_df.loc[pop, 'Red']) or 
                (mode == 'full' and specify_red is None and specify_green is None)
            )
            
            logging.debug(f"  Should calculate markers? {should_calculate}")
            logging.debug(f"  Reason: Red is NaN: {pd.isna(settings_df.loc[pop, 'Red'])}, "
                  f"Full mode with no R/G overrides: {mode == 'full' and specify_red is None and specify_green is None}")
            
            if should_calculate:
                logging.info(f"  Running marker selection for {pop}...")
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
                        verbose=verbose
                    )
                else:
                    # Fallback to mean expression (original method)
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

    # Ranges belong to markers, not RGB positions. Selection and overrides can
    # move/replace markers; resolve against the original row so swaps are safe.
    for pop in pop_categories:
        original = original_settings.loc[pop]
        for color in ['Red', 'Green', 'Blue']:
            old_marker = original[color]
            new_marker = settings_df.loc[pop, color]
            if (pd.isna(old_marker) and pd.isna(new_marker)) or (
                pd.notna(old_marker) and pd.notna(new_marker) and old_marker == new_marker
            ):
                continue
            matching_ranges = [
                (original[f'{source}_min'], original[f'{source}_max'])
                for source in ['Red', 'Green', 'Blue']
                if pd.notna(new_marker) and pd.notna(original[source])
                and original[source] == new_marker
            ]
            unique_ranges = pd.DataFrame(matching_ranges, columns=['min', 'max']).drop_duplicates()
            if len(unique_ranges) == 1:
                new_min, new_max = unique_ranges.iloc[0]
            else:
                new_min, new_max = None, None
            settings_df.loc[pop, [f'{color}_min', f'{color}_max']] = [new_min, new_max]
            logging.info(
                "Population=%s | %s marker changed from %s to %s; %s.",
                pop, color, old_marker, new_marker,
                "using that marker's saved range" if len(unique_ranges) == 1
                else "clearing previous marker's range and using global defaults",
            )

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

    # Plotting from saved settings must not mutate the user's source CSV.
    if mode != 'load_markers':
        settings_df.to_csv(backgating_settings_path)
        logging.info(f"Saved backgating settings to: {backgating_settings_path}")
    else:
        logging.info("Using saved backgating settings without modifying %s", backgating_settings_path)

    # 5) If mode='save_markers', we stop here (no imaging).
    if mode == 'save_markers':
        logging.info("Markers saved; no further backgating performed (mode='save_markers').")
        return

    # 6) If we get here, we either 'full' or 'load_markers' => run the actual backgating
    all_rois_for_images = _normalize_roi_names(adata.obs[roi_obs].astype(str).unique().tolist())

    for pop in pop_categories:
        # Ensure consistent string comparison to handle leiden categories stored as strings
        pop_mask = adata.obs[pop_obs].astype(str) == str(pop)
        pop_rois = _normalize_roi_names(adata.obs.loc[pop_mask, roi_obs].astype(str).tolist())
        rois_to_save = _select_rois_to_save(pop_rois, max_rois_to_save)

        if not rois_to_save:
            logging.warning(
                "No ROIs selected for population '%s' after applying max_rois_to_save=%s. Skipping.",
                pop,
                max_rois_to_save,
            )
            continue

        if max_rois_to_save is None:
            logging.info(
                "Backgating population '%s': saving all %d ROI(s).",
                pop,
                len(rois_to_save),
            )
        else:
            logging.info(
                "Backgating population '%s': saving %d of %d ROI(s) after random selection.",
                pop,
                len(rois_to_save),
                len(pop_rois),
            )
            logging.debug("Selected ROIs for population '%s': %s", pop, rois_to_save)

        pop_cells = adata.obs.loc[
            pop_mask & adata.obs[roi_obs].astype(str).isin(rois_to_save),
            cell_index_obs,
        ]
        
        logging.debug(f"Population '{pop}' (type: {type(pop)}): checking {pop_mask.sum()} cells")
        if pop_cells.empty:
            logging.warning(f"No cells found for population '{pop}'. Skipping.")
            logging.debug(f"Available population values: {adata.obs[pop_obs].astype(str).unique()[:10]}")
            continue

        # Grab final channels + ranges from settings_df
        red_marker   = settings_df.loc[pop, 'Red']
        green_marker = settings_df.loc[pop, 'Green']
        blue_marker  = settings_df.loc[pop, 'Blue']

        red_range   = (settings_df.loc[pop, 'Red_min'],   settings_df.loc[pop, 'Red_max'])   if specify_ranges else None
        green_range = (settings_df.loc[pop, 'Green_min'], settings_df.loc[pop, 'Green_max']) if specify_ranges else None
        blue_range  = (settings_df.loc[pop, 'Blue_min'],  settings_df.loc[pop, 'Blue_max'])  if specify_ranges else None

        logging.info(f"Backgating population: {pop}")
        logging.info(f"  -> Cells in selected ROIs: {len(pop_cells)}")
        logging.info(f"  -> Thumbnail gallery cap: {cells_per_group}")
        logging.info(f"  -> Red={red_marker}, range={red_range}")
        logging.info(f"  -> Green={green_marker}, range={green_range}")
        logging.info(f"  -> Blue={blue_marker}, range={blue_range}")

        # Call your backgating function
        # Make sure it supports the new parameters: mask_folder, exclude_rois_without_mask, cell_plot_spacing
        backgating(
            adata=adata,
            cell_index=list(pop_cells),
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
            max_gallery_cells=cells_per_group,
            gallery_sampling=gallery_sampling,
            gallery_balance_rois=gallery_balance_rois,
            gallery_random_state=gallery_random_state,
            gallery_layer=gallery_layer,
            gallery_markers=gallery_markers,
            gallery_umap_key=gallery_umap_key,
            gallery_umap_weight=gallery_umap_weight,
            gallery_save_svg=gallery_save_svg,
            font_family=font_family,
            roi_list=rois_to_save,
            # Output
            output_folder=output_folder,
            save_subfolder=clean_text(pop),
            # Clipping
            minimum=minimum,
            max_quantile=max_quantile,
            # Image generation scope
            image_samples_list=all_rois_for_images
        )

        # Create population overlay visualizations for all cells of this type in each ROI
        if population_overlays:
            logging.info(f"Creating population overlay visualizations for '{pop}'...")
            
            logging.info(f"Population '{pop}' overlays will be saved for ROIs: {rois_to_save}")
            pop_subfolder = Path(output_folder) / clean_text(pop)
            
            # Create population overlays subdirectory
            overlay_dir = pop_subfolder / 'population_overlays'
            overlay_dir.mkdir(parents=True, exist_ok=True)

            # Build legend entries based on available channel assignments (optional)
            legend_markers = []
            legend_colors = []
            if population_overlay_show_legend:
                channel_palette = {
                    'red': (255, 0, 0),
                    'green': (0, 255, 0),
                    'blue': (0, 0, 255),
                }

                for marker_name, channel_name in [
                    (red_marker, 'red'),
                    (green_marker, 'green'),
                    (blue_marker, 'blue'),
                ]:
                    if marker_name:
                        legend_markers.append(str(marker_name))
                        legend_colors.append(channel_palette[channel_name])
            
        if population_overlays:
            for roi in rois_to_save:
                try:
                    # Path to composite image (created by earlier make_images call)
                    composite_img_path = pop_subfolder / f"{roi}.png"
                    
                    # Path to mask file (if using masks)
                    mask_path = None
                    if use_masks:
                        if isinstance(use_masks, bool) and use_masks:
                            # Look for standard mask files
                            potential_tif = Path(mask_folder) / f"{roi}.tif"
                            potential_tiff = Path(mask_folder) / f"{roi}.tiff"
                            if potential_tif.exists():
                                mask_path = str(potential_tif)
                            elif potential_tiff.exists():
                                mask_path = str(potential_tiff)
                        elif isinstance(use_masks, str):
                            # CSV mapping - would need to implement reading logic here
                            logging.warning(f"CSV mask mapping not yet implemented for population overlays")
                    
                    # Create overlay
                    overlay_output_path = overlay_dir / f"{roi}_population_overlay.{population_overlay_extension}"
                    svg_output_path = None
                    if population_overlay_save_svg or overlay_output_path.suffix.lower() == '.svg':
                        svg_output_path = overlay_output_path.with_suffix('.svg')
                    if overlay_output_path.suffix.lower() == '.svg':
                        overlay_output_path = overlay_output_path.with_suffix('.png')
                    
                    overlay_fig = create_population_overlay(
                        adata=adata,
                        population=pop,
                        pop_obs=pop_obs,
                        roi_name=roi,
                        composite_image_path=str(composite_img_path),
                        mask_path=mask_path,
                        roi_obs=roi_obs,
                        object_index_obs=object_index_obs,
                        output_path=str(overlay_output_path),
                        contour_color=(255, 255, 255),  # White contours
                        contour_width=population_overlay_outline_width,
                        verbose=False,
                        legend_markers=legend_markers if population_overlay_show_legend else None,
                        legend_colors=legend_colors if population_overlay_show_legend else None,
                        legend_fontsize=population_overlay_legend_fontsize,
                        show_label=population_overlay_show_label,
                        show_population_label=population_overlay_show_population_label,
                        population_label_text=population_overlay_label_text,
                        population_label_fontsize=population_overlay_label_fontsize or population_overlay_legend_fontsize,
                        crop_size=population_overlay_crop_size,
                        crop_origin=population_overlay_crop_origin,
                        show_scale_bar=population_overlay_show_scale_bar,
                        scale_bar_length=population_overlay_scale_bar_length,
                        scale_bar_thickness=population_overlay_scale_bar_thickness,
                        scale_bar_color=population_overlay_scale_bar_color,
                        scale_bar_outline_thickness=population_overlay_scale_bar_outline_thickness,
                        scale_bar_text=population_overlay_scale_bar_text,
                        scale_bar_text_size=population_overlay_scale_bar_text_size,
                        svg_output_path=str(svg_output_path) if svg_output_path else None,
                        font_family=font_family,
                        comparison_images=comparison_sources,
                        primary_title=population_overlay_primary_title,
                        title_fontsize=population_overlay_title_fontsize,
                    )
                    
                    if overlay_fig is not None:
                        # No caller needs the returned figure in this batch path.
                        # Release large image/artist buffers without waiting for GC.
                        overlay_fig.clear()
                        del overlay_fig

                except MemoryError:
                    logging.error(
                        "Memory exhausted creating the population overlay for ROI '%s'; "
                        "stopping the assessment instead of skipping further ROIs.", roi,
                    )
                    raise
                except Exception as e:
                    logging.warning(f"Failed to create population overlay for ROI '{roi}': {e}")
                    continue
        
        logging.info(f"Population overlay visualizations completed for '{pop}'.")

    # ----------------------------------------------------------------
    # 7) Summarize rescaling values into settings-format CSV
    # ----------------------------------------------------------------
    def _summarize_values(values):
        if len(values) == 0:
            return None
        # Normalize numeric values with tolerance
        try:
            vals = np.asarray(values, dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return None
            if np.allclose(vals, vals[0], rtol=0, atol=1e-6):
                return float(vals[0])
            return 'Variable'
        except Exception:
            unique_vals = pd.Series(values).dropna().unique()
            return unique_vals[0] if len(unique_vals) == 1 else 'Variable'

    rescale_summary = settings_df[[
        'Red', 'Green', 'Blue',
        'Red_min', 'Red_max',
        'Green_min', 'Green_max',
        'Blue_min', 'Blue_max'
    ]].copy()

    for pop in pop_categories:
        pop_key = str(pop)
        rescale_csv = Path(output_folder) / clean_text(pop_key) / 'rescale_values.csv'
        if not rescale_csv.exists():
            continue

        rescale_df = pd.read_csv(rescale_csv)
        if rescale_df.empty:
            continue

        for color in ['Red', 'Green', 'Blue']:
            marker_name = rescale_summary.loc[pop, color] if pop in rescale_summary.index else None
            if marker_name is None or (isinstance(marker_name, float) and pd.isna(marker_name)):
                continue

            channel_mask = rescale_df['channel'].astype(str).str.lower() == color.lower()
            marker_mask = rescale_df['marker'].astype(str) == str(marker_name)
            subset = rescale_df[channel_mask & marker_mask]
            if subset.empty:
                continue

            min_val = _summarize_values(subset['min_used'].tolist())
            max_val = _summarize_values(subset['max_used'].tolist())

            rescale_summary.loc[pop, f"{color}_min"] = min_val
            rescale_summary.loc[pop, f"{color}_max"] = max_val

    rescale_summary_path = Path(output_folder) / f"rescale_summary_{Path(backgating_settings_file).name}"
    rescale_summary.to_csv(rescale_summary_path)

    logging.info("Backgating assessment complete.")


def _parse_population_overlay_roi_name(path: Union[str, Path]) -> Optional[str]:
    """Extract ROI name from a backgating population overlay filename."""
    match = re.match(r"^(?P<roi>.+)_population_overlay\.[^.]+$", Path(path).name)
    if match is None:
        return None
    return match.group("roi")


def _find_population_overlay_image(
    backgating_output_folder: Union[str, Path],
    population: Union[str, int],
    roi: Union[str, int],
    *,
    source_format: str = 'auto',
    prefer_svg: bool = False,
) -> Optional[Path]:
    """Locate a saved population overlay image for one population/ROI pair."""
    overlay_dir = Path(backgating_output_folder) / clean_text(str(population)) / "population_overlays"
    if not overlay_dir.exists():
        return None

    matches = []
    roi_str = str(roi)
    extensions = ['.svg', '.png'] if prefer_svg else ['.png', '.svg']
    extensions += ['.tif', '.tiff', '.jpg', '.jpeg', '.bmp']
    if source_format != 'auto':
        extensions = [f'.{source_format}']
    for candidate in overlay_dir.glob("*_population_overlay.*"):
        if candidate.suffix.lower() not in extensions:
            continue
        parsed_roi = _parse_population_overlay_roi_name(candidate)
        if parsed_roi == roi_str:
            matches.append(candidate)

    if not matches:
        return None

    matches.sort(key=lambda p: (extensions.index(p.suffix.lower()), p.name))
    return matches[0]


def _gallery_svg_source(path, prefix, font_family='Arial'):
    """Import a self-contained SVG, isolating IDs, local references and styles."""
    from xml.etree import ElementTree as ET

    root = ET.parse(path).getroot()
    if root.tag != '{http://www.w3.org/2000/svg}svg':
        raise ValueError('Overlay is not an SVG document.')
    if 'viewBox' not in root.attrib:
        def length(value):
            match = re.fullmatch(r'\s*([\d.eE+\-]+)\s*(px|pt|pc|in|cm|mm)?\s*', value or '')
            if match is None:
                raise ValueError('SVG needs a viewBox or absolute width and height.')
            factor = {None: 1, 'px': 1, 'pt': 96 / 72, 'pc': 16,
                      'in': 96, 'cm': 96 / 2.54, 'mm': 96 / 25.4}[match[2]]
            return float(match[1]) * factor
        root.set('viewBox', f"0 0 {length(root.get('width'))} {length(root.get('height'))}")
    viewbox = [float(v) for v in re.split(r'[\s,]+', root.get('viewBox').strip())]
    if len(viewbox) != 4 or not np.isfinite(viewbox).all() or min(viewbox[2:]) <= 0:
        raise ValueError('SVG viewBox must contain four finite values with positive dimensions.')
    ids = [element.get('id') for element in root.iter() if element.get('id')]
    if len(ids) != len(set(ids)):
        raise ValueError('Source SVG contains duplicate IDs.')
    id_map = {name: f'{prefix}{name}' for name in ids}
    root_id = id_map.get(root.get('id'), f'{prefix}artwork')
    if not root.get('id'):
        while root_id in id_map.values():
            root_id += '_root'

    def references(value):
        return re.sub(r'url\(\s*([\x22\x27]?)#([^\s)\x22\x27]+)\1\s*\)',
                      lambda m: f'url(#{id_map.get(m[2], m[2])})', value)

    def css_rule(match):
        selectors, declarations = match.groups()
        if selectors.lstrip().startswith('@'):
            return match[0]
        selectors = re.sub(r'#([\w.\-]+)', lambda m: '#' + id_map.get(m[1], m[1]), selectors)
        selectors = ', '.join(f'#{root_id} {s.strip()}' for s in selectors.split(','))
        return f'{selectors} {{{references(declarations)}}}'

    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id':
                element.set(key, id_map[value])
            elif key in ('href', '{http://www.w3.org/1999/xlink}href') and value.startswith('#'):
                element.set(key, '#' + id_map.get(value[1:], value[1:]))
            else:
                element.set(key, references(value))
        if element.tag == '{http://www.w3.org/2000/svg}style' and element.text:
            element.text = re.sub(r'([^{}]+)\{([^{}]*)\}', css_rule, element.text)
    root.set('id', root_id)
    _set_svg_font_family(root, font_family)
    return root


def _compose_overlay_gallery(panels, populations, roi, ncols, nrows, panel_size,
                             row_spacing, column_spacing, population_fontsize, roi_fontsize,
                             font_family='Arial'):
    """Compose editable source trees and return SVG bytes and point-based layout."""
    import base64
    from io import BytesIO
    from xml.etree import ElementTree as ET
    from PIL import Image

    svg = 'http://www.w3.org/2000/svg'
    ink = 'http://www.inkscape.org/namespaces/inkscape'
    xlink = 'http://www.w3.org/1999/xlink'
    ET.register_namespace('', svg)
    ET.register_namespace('inkscape', ink)
    ET.register_namespace('xlink', xlink)
    panel_w, panel_h = panel_size
    title_height = 0 if population_fontsize is None else 1.5 * population_fontsize
    heading_height = 0 if roi_fontsize is None else 1.75 * roi_fontsize
    width = ncols * panel_w + (ncols - 1) * column_spacing
    height = heading_height + nrows * (panel_h + title_height) + (nrows - 1) * row_spacing
    root = ET.Element(f'{{{svg}}}svg', {'width': f'{width:g}pt', 'height': f'{height:g}pt',
                                      'viewBox': f'0 0 {width:g} {height:g}', 'version': '1.1'})

    def group(parent, gid, label):
        return ET.SubElement(parent, f'{{{svg}}}g', {'id': gid, f'{{{ink}}}groupmode': 'layer',
                                                   f'{{{ink}}}label': label})

    def text(parent, value, x, baseline, fontsize):
        element = ET.SubElement(parent, f'{{{svg}}}text', {
            'x': f'{x:g}', 'y': f'{baseline:g}', 'text-anchor': 'middle',
            'style': f'font-size:{fontsize:g}px;font-family:{_svg_font_family(font_family)};fill:black',
        })
        element.text = str(value)

    background = group(root, 'gallery_background', 'Background')
    ET.SubElement(background, f'{{{svg}}}rect', {'width': str(width), 'height': str(height), 'fill': 'white'})
    if roi_fontsize is not None:
        text(group(root, 'gallery_heading', 'ROI title'), roi, width / 2, roi_fontsize, roi_fontsize)
    layout = []
    for idx, (population, panel) in enumerate(zip(populations, panels)):
        x = (idx % ncols) * (panel_w + column_spacing)
        y = heading_height + (idx // ncols) * (panel_h + title_height + row_spacing)
        prefix = f'panel_{idx + 1:03d}'
        parent = group(root, prefix, str(population))
        if population_fontsize is not None:
            text(group(parent, f'{prefix}_title', 'Population title'), population,
                 x + panel_w / 2, y + population_fontsize, population_fontsize)
        image_y = y + title_height
        layout.append((x, image_y, panel_w, panel_h))
        path = panel['path']
        if panel['error']:
            text(parent, panel['error'], x + panel_w / 2, image_y + panel_h / 2, 12)
        elif path.suffix.lower() == '.svg':
            source = panel['svg']
            # Nested SVG establishes a panel viewport while retaining all source
            # groups, definitions, transforms, embedded pixels and clipping paths.
            source.set('x', str(x))
            source.set('y', str(image_y))
            source.set('width', str(panel_w))
            source.set('height', str(panel_h))
            source.set('preserveAspectRatio', 'xMidYMid meet')
            source.set('overflow', 'hidden')
            parent.append(source)
        else:
            with BytesIO() as buffer:
                Image.fromarray(panel['image']).save(buffer, format='PNG')
                data = base64.b64encode(buffer.getvalue()).decode('ascii')
            raster = group(parent, f'{prefix}_source_image', 'Raster overlay')
            ET.SubElement(raster, f'{{{svg}}}image', {
                'x': str(x), 'y': str(image_y), 'width': str(panel_w), 'height': str(panel_h),
                'preserveAspectRatio': 'xMidYMid meet', f'{{{xlink}}}href': 'data:image/png;base64,' + data,
            })
    return ET.tostring(root, encoding='utf-8', xml_declaration=True), (width, height, layout)


def _save_raster_overlay_gallery(panels, populations, roi, layout, output_path,
                                 dpi, population_fontsize, roi_fontsize, font_family='Arial'):
    """Render legacy raster panels at the same point-based positions as the SVG."""
    width, height, positions = layout
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=dpi)
    try:
        if roi_fontsize is not None:
            fig.text(0.5, 1, str(roi), ha='center', va='top', fontsize=roi_fontsize, fontfamily=font_family)
        for panel, population, (x, y, w, h) in zip(panels, populations, positions):
            ax = fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])
            if panel['error']:
                ax.text(.5, .5, panel['error'], ha='center', va='center', transform=ax.transAxes,
                        fontfamily=font_family)
            else:
                ax.imshow(panel['image'], interpolation='none')
            ax.axis('off')
            if population_fontsize is not None:
                fig.text((x + w / 2) / width, 1 - (y - 1.5 * population_fontsize) / height,
                         str(population), ha='center', va='top', fontsize=population_fontsize,
                         fontfamily=font_family)
        with plt.rc_context({'savefig.bbox': None}):
            fig.savefig(output_path, dpi=dpi, facecolor='white')
    finally:
        plt.close(fig)


@_noninteractive_plotting()
def create_population_overlay_galleries(
    backgating_output_folder: Union[str, Path],
    populations: List[Union[str, int]],
    ncols: int,
    nrows: int,
    output_subfolder: str = "population_overlay_galleries",
    roi_list: Optional[List[Union[str, int]]] = None,
    dpi: int = 200,
    population_title_fontsize: Optional[int] = 12,
    roi_title_fontsize: Optional[int] = 16,
    *,
    output_format: str = 'png',
    source_format: str = 'auto',
    row_spacing: float = 12,
    column_spacing: float = 12,
    panel_size: Tuple[float, float] = (288, 288),
    font_family: str = 'Arial',
) -> Path:
    """
    Create one per-ROI PNG and/or editable SVG gallery from saved overlays.

    Expected input layout is the output of ``backgating_assessment``:
      backgating_output_folder/
        <cleaned_population_name>/
          population_overlays/
            <ROI>_population_overlay.png (or .svg)

    Parameters
    ----------
    backgating_output_folder : str or Path
        Root output directory previously written by ``backgating_assessment``.
    populations : list
        Ordered list of populations as they should appear in the gallery.
    ncols, nrows : int
        Layout of the gallery grid.
    output_subfolder : str, optional
        New subdirectory inside ``backgating_output_folder`` where ROI galleries are saved.
    roi_list : list, optional
        Optional explicit ROI order/subset. If None, all ROIs found across requested populations are used.
    dpi : int, optional
        Output DPI for PNG galleries; does not affect SVG geometry or spacing.
    population_title_fontsize : int or None, optional
        Title font size for each population panel. If None, population panel
        titles are not plotted.
    roi_title_fontsize : int or None, optional
        Suptitle font size for the ROI label. If None, ROI suptitle is not plotted.
    output_format : {'png', 'svg', 'both'}, optional
        Default 'png' preserves existing output behavior. SVG output copies source
        vector objects, text and groups into a separate layer per population.
        PNG rendering of SVG inputs requires the optional CairoSVG renderer
        (install SpatialBiologyToolkit[svg]); SVG composition itself does not.
    source_format : {'auto', 'png', 'svg'}, optional
        'auto' prefers SVG for SVG/both output and PNG for PNG output, falling
        back to the other format when needed. Explicit formats do not fall back.
        PNG panels embedded in an SVG remain raster images; their annotations
        cannot be recovered as editable objects. Use self-contained SBT SVGs
        to retain the original source image and editable overlay components.
    row_spacing, column_spacing : float, optional
        Nonnegative gaps in points (72 points = 1 inch), default 12. Row gaps are
        measured between panel blocks, including any population title band.
        Column gaps are between panel viewports. Zero gives adjacent grid slots.
    panel_size : (float, float), optional
        Overlay viewport (width, height) in points, default (288, 288), i.e. 4x4
        inches. Sources retain their aspect ratio and are centered within it;
        differing aspect ratios may leave additional space inside each viewport.
    font_family : str, optional
        Single font name, default 'Arial'. Applied to gallery titles and all live
        text in imported SVG panels, replacing legacy fallback lists. Source
        files are not changed. Font files are not embedded; install the selected
        font on the rendering/editing machine. Text already outlined as paths
        and labels baked into PNGs cannot be changed this way.

    Returns
    -------
    Path
        Directory containing the saved ROI gallery images.
    """
    base_dir = Path(backgating_output_folder)
    _svg_font_family(font_family)
    if not base_dir.exists():
        raise FileNotFoundError(f"Backgating output folder not found: {base_dir}")

    if ncols <= 0 or nrows <= 0 or int(ncols) != ncols or int(nrows) != nrows:
        raise ValueError("ncols and nrows must both be positive integers.")
    ncols, nrows = int(ncols), int(nrows)
    if output_format not in ('png', 'svg', 'both'):
        raise ValueError("output_format must be 'png', 'svg' or 'both'.")
    if source_format not in ('auto', 'png', 'svg'):
        raise ValueError("source_format must be 'auto', 'png' or 'svg'.")
    if not np.isfinite([row_spacing, column_spacing]).all() or min(row_spacing, column_spacing) < 0:
        raise ValueError('row_spacing and column_spacing must be finite and nonnegative.')
    if len(panel_size) != 2 or not np.isfinite(panel_size).all() or min(panel_size) <= 0:
        raise ValueError('panel_size must contain two finite positive dimensions in points.')
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError('dpi must be positive and finite.')
    for fontsize in (population_title_fontsize, roi_title_fontsize):
        if fontsize is not None and (not np.isfinite(fontsize) or fontsize <= 0):
            raise ValueError('Title font sizes must be positive and finite, or None.')

    if not populations:
        raise ValueError("At least one population must be provided.")

    if len(populations) > int(ncols) * int(nrows):
        raise ValueError(
            f"Layout {nrows}x{ncols} cannot fit {len(populations)} populations."
        )

    output_dir = base_dir / output_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    if roi_list is None:
        discovered_rois = set()
        for population in populations:
            overlay_dir = base_dir / clean_text(str(population)) / "population_overlays"
            if not overlay_dir.exists():
                logging.warning(
                    "Population overlay directory not found for population '%s': %s",
                    population,
                    overlay_dir,
                )
                continue
            for candidate in overlay_dir.glob("*_population_overlay.*"):
                if source_format != 'auto' and candidate.suffix.lower() != f'.{source_format}':
                    continue
                if candidate.suffix.lower() not in ('.svg', '.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'):
                    continue
                roi_name = _parse_population_overlay_roi_name(candidate)
                if roi_name:
                    discovered_rois.add(roi_name)
        roi_names = sorted(discovered_rois)
    else:
        roi_names = [str(roi) for roi in roi_list]

    if not roi_names:
        raise FileNotFoundError(
            f"No population overlay images were found in {base_dir} for populations {populations}."
        )

    missing_pairs: List[str] = []
    for roi in roi_names:
        panels = []
        for idx, population in enumerate(populations):
            overlay_path = _find_population_overlay_image(
                base_dir, population, roi, source_format=source_format,
                prefer_svg=output_format in ('svg', 'both'))
            panel = {'path': overlay_path, 'error': None}
            if overlay_path is None:
                panel['error'] = 'Missing'
                missing_pairs.append(f"{population}::{roi}")
            else:
                try:
                    if overlay_path.suffix.lower() == '.svg':
                        panel['svg'] = _gallery_svg_source(overlay_path, f'panel_{idx + 1:03d}_source_', font_family)
                    else:
                        from PIL import Image
                        with Image.open(overlay_path) as image:
                            panel['image'] = np.asarray(image.convert('RGBA'))
                except Exception as exc:
                    logging.warning(
                        "Failed to load population overlay image for population '%s', ROI '%s': %s",
                        population,
                        roi,
                        exc,
                    )
                    panel['error'] = 'Load failed'
                    missing_pairs.append(f'{population}::{roi} (load failed)')
            panels.append(panel)

        svg_bytes, layout = _compose_overlay_gallery(
            panels, populations, roi, ncols, nrows, panel_size,
            row_spacing, column_spacing, population_title_fontsize, roi_title_fontsize, font_family)
        stem = output_dir / f"{clean_text(str(roi))}_population_gallery"
        # Do not use with_suffix: a cleaned ROI could still contain a period in
        # future naming rules, and the complete gallery stem must be preserved.
        if output_format in ('svg', 'both'):
            Path(f'{stem}.svg').write_bytes(svg_bytes)
        if output_format in ('png', 'both'):
            if any(panel.get('svg') is not None for panel in panels):
                try:
                    import cairosvg
                except (ImportError, OSError) as exc:
                    raise RuntimeError(
                        'PNG previews of SVG overlays require CairoSVG and its Cairo runtime. '
                        'Install SpatialBiologyToolkit[svg] or use output_format="svg". '
                        'Any requested SVG gallery has already been saved.'
                    ) from exc
                # The gallery uses point units, so physical sizing and gaps are
                # invariant while DPI controls only PNG pixel resolution.
                cairosvg.svg2png(bytestring=svg_bytes, write_to=str(stem) + '.png', dpi=dpi)
            else:
                _save_raster_overlay_gallery(
                    panels, populations, roi, layout, str(stem) + '.png', dpi,
                    population_title_fontsize, roi_title_fontsize, font_family)

    if missing_pairs:
        preview = ", ".join(missing_pairs[:10])
        if len(missing_pairs) > 10:
            preview += ", ..."
        logging.warning(
            "Population overlay galleries were created with %d missing population/ROI panels: %s",
            len(missing_pairs),
            preview,
        )
    else:
        logging.info(
            "Population overlay galleries created successfully for %d ROIs in %s.",
            len(roi_names),
            output_dir,
        )

    return output_dir


def update_settings_from_marker_dict(
    settings_path: Union[str, Path],
    marker_settings: Optional[Dict[str, Union[str, float, Tuple[float, Union[str, float]]]]] = None,
    generate_dict: bool = False,
) -> Union[pd.DataFrame, Dict[str, Union[str, float, Tuple[float, Union[str, float]]]]]:
    """
    Update backgating_settings.csv so any occurrence of a marker receives the
    provided setting, regardless of population or RGB channel.

    If generate_dict is True, return an example dictionary by scanning the
    settings file for each marker. If the marker's settings are consistent
    across all RGB channels, return that value; otherwise return "Inconsistent".

    Args:
        settings_path: Path to backgating_settings.csv.
        marker_settings: Mapping of marker name to setting.
            If a value is a ``(min, max)`` tuple, both the ``*_min`` and
            ``*_max`` columns are updated. Otherwise only ``*_max`` is updated.
        generate_dict: If True, return an example dictionary built from the
            settings file and do not modify the file.

    Returns:
        Updated settings DataFrame, or the generated example dictionary.
    """
    settings_path = Path(settings_path)
    if not settings_path.is_file():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    settings_df = pd.read_csv(settings_path, index_col=0)

    if generate_dict:
        example_dict: Dict[str, Union[str, float, Tuple[float, Union[str, float]]]] = {}
        def _normalize_scalar(value):
            if isinstance(value, np.generic):
                return value.item()
            return value

        for color in ["Red", "Green", "Blue"]:
            marker_col = color
            min_col = f"{color}_min"
            max_col = f"{color}_max"

            if marker_col not in settings_df.columns:
                continue

            marker_series = settings_df[marker_col].astype(str)
            for marker_name in marker_series.dropna().unique():
                marker_name_str = str(marker_name).strip()
                if not marker_name_str or marker_name_str.lower() == "nan":
                    continue

                mask = marker_series.str.lower() == marker_name_str.lower()
                if not mask.any():
                    continue

                min_vals = settings_df.loc[mask, min_col] if min_col in settings_df.columns else pd.Series(dtype=object)
                max_vals = settings_df.loc[mask, max_col] if max_col in settings_df.columns else pd.Series(dtype=object)

                min_unique = [_normalize_scalar(v) for v in min_vals.dropna().unique()]
                max_unique = [_normalize_scalar(v) for v in max_vals.dropna().unique()]

                if min_unique and max_unique:
                    min_val = min_unique[0] if len(min_unique) == 1 else "Inconsistent"
                    max_val = max_unique[0] if len(max_unique) == 1 else "Inconsistent"
                    setting_val: Union[str, float, Tuple[float, Union[str, float]]] = (min_val, max_val)
                else:
                    if len(max_unique) == 1:
                        setting_val = max_unique[0]
                    elif len(max_unique) == 0:
                        continue
                    else:
                        setting_val = "Inconsistent"

                if marker_name_str in example_dict and example_dict[marker_name_str] != setting_val:
                    example_dict[marker_name_str] = "Inconsistent"
                else:
                    example_dict[marker_name_str] = setting_val

        return example_dict

    if marker_settings is None:
        raise ValueError("marker_settings must be provided when generate_dict is False.")

    for color in ["Red", "Green", "Blue"]:
        marker_col = color
        min_col = f"{color}_min"
        max_col = f"{color}_max"

        if marker_col not in settings_df.columns:
            continue

        marker_series = settings_df[marker_col].astype(str)

        for marker_name, setting in marker_settings.items():
            if marker_name is None:
                continue
            marker_name_str = str(marker_name).strip().lower()
            if not marker_name_str:
                continue

            mask = marker_series.str.lower() == marker_name_str
            if not mask.any():
                continue

            if isinstance(setting, (tuple, list)) and len(setting) == 2:
                settings_df.loc[mask, min_col] = setting[0]
                settings_df.loc[mask, max_col] = setting[1]
            else:
                settings_df.loc[mask, max_col] = setting

    settings_df.to_csv(settings_path)
    logging.info(f"Updated settings saved to: {settings_path}")
    return settings_df
