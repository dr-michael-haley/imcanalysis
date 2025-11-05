# Standard library imports
import os
import re
import warnings
import json
import hashlib
from pathlib import Path
from shutil import copytree, rmtree
import logging

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from readimc import MCDFile, TXTFile
from scipy.stats import zscore
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import shared utilities and configurations
from .config_and_utils import *
from .config_and_utils import cleanstring

################ Functions

def load_single_img(filename):
    """
    Load a single image from a file.

    Parameters
    ----------
    filename : str or Path
        The image file name, must end with .tiff or .tif.

    Returns
    -------
    numpy.ndarray
        Loaded image data as a 2D NumPy array.

    Raises
    ------
    ValueError
        If the file does not end with .tiff or .tif.
        If the loaded image is not 2D.
    """
    filename = Path(filename)
    if filename.suffix.lower() in ['.tiff', '.tif']:
        img_in = tiff.imread(filename).astype('float32')
    else:
        raise ValueError(f'File {filename} should end with .tiff or .tif!')
    if img_in.ndim != 2:
        raise ValueError(f'Image {filename} should be 2D!')
    return img_in


def load_imgs_from_directory(load_directory, channel_name, quiet=False):
    """
    Load images for a specific channel from a directory.

    Parameters
    ----------
    load_directory : str or Path
        Directory containing the images, with subdirectories for each ROI.
    channel_name : str
        Name of the channel to load images for.
    quiet : bool, optional
        If True, suppress print statements. Defaults to False.

    Returns
    -------
    img_collect : list of numpy.ndarray
        List of loaded images.
    img_file_list : list of str
        List of image file names.
    img_folders : list of Path
        List of image folder paths (ROIs).

    Raises
    ------
    ValueError
        If no images are found for the specified channel.
    """
    img_collect = []
    img_file_list = []
    img_folders = []

    load_directory = Path(load_directory)

    if not quiet:
        logging.info('Loading image data from directories...\n')

    for sub_img_folder in load_directory.iterdir():
        if sub_img_folder.is_dir():
            found_image = False
            for img_file in sub_img_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.tiff', '.tif']:
                    if channel_name.lower() in img_file.name.lower():
                        img_read = load_single_img(img_file)
                        if not quiet:
                            logging.info(f"Loaded image: {img_file}")
                        img_file_list.append(img_file.name)
                        img_collect.append(img_read)
                        img_folders.append(sub_img_folder)
                        found_image = True
                        break  # Break after finding the first matching image
            if not found_image and not quiet:
                logging.info(f"No image found for channel '{channel_name}' in folder '{sub_img_folder}'.")

    if not quiet:
        logging.info('\nImage data loading completed!')

    if not img_collect:
        raise ValueError(f'No images found for channel "{channel_name}". Please check the channel name!')

    return img_collect, img_file_list, img_folders


def _get_actual_num_acquisition(acquisitions=None, path=None):
    """
    Returns the number of acquisitions in an MCD file where the number of channels is larger than zero.
    This avoids acquisitions where the stage was moved but no image was taken.
    Assumes that missing acquisitions are at the end of the list.

    Parameters
    ----------
    acquisitions : list of Acquisition, optional
        The acquisitions to check. If None, `path` must be provided.
    path : str or Path, optional
        Path to the MCD file. Required if `acquisitions` is None.

    Returns
    -------
    int
        The number of valid acquisitions in the MCD file.

    Raises
    ------
    ValueError
        If neither acquisitions nor path are provided.
    """
    if acquisitions is None:
        if path is not None:
            path = Path(path)
        else:
            raise ValueError("Either acquisitions or path must be provided.")
        if path.suffix == ".mcd":
            with MCDFile(path) as f:
                acquisitions = f.slides[0].acquisitions
        else:
            raise ValueError("File is not an MCD file.")

    # Count the number of channels in each acquisition
    num_channels_per_acquisition = [x.num_channels for x in acquisitions]

    # Keep only acquisitions where number of channels is larger than zero
    valid_acquisitions = [x for x in num_channels_per_acquisition if x > 0]

    if len(valid_acquisitions) < len(num_channels_per_acquisition):
        warnings.warn("Some acquisitions have zero channels. They will be ignored.")

    return len(valid_acquisitions)


def read_txt(
    path,
    only_metadata=False,
    fill_empty_metals=False,
):
    """
    Reads a TXT file and returns data, channels, names, and metadata.

    Parameters
    ----------
    path : str or Path
        Path to the TXT file.
    only_metadata : bool, optional
        If True, only return the metadata. Defaults to False.
    fill_empty_metals : bool, optional
        If True, fill empty names with metal names. Defaults to False.

    Returns
    -------
    data : numpy.ndarray
        The data from the TXT file.
    channels : list of str
        The channel labels from the TXT file.
    names : list of str
        The channel names from the TXT file.
    meta : pandas.DataFrame
        Metadata for the acquisition.

    Raises
    ------
    ValueError
        If the file is not a TXT file.
    """
    path = Path(path)
    if path.suffix.lower() != ".txt":
        raise ValueError("File is not a TXT file.")

    data = None
    meta = None

    with TXTFile(path) as f:
        if not only_metadata:
            # Read data for the acquisition
            data = f.read_acquisition()

        # Get channel names and labels
        names = list(f.channel_labels)
        channels = list(f.channel_names)

        # Create metadata DataFrame - TXT files don't have acquisition descriptions/dimensions
        # so we'll create minimal metadata
        clean_stem = cleanstring(path.stem)
        meta = pd.DataFrame(
            {
                "id": [0],  # TXT files have only one acquisition
                "description": [clean_stem],  # Use cleaned filename as description
                "width_um": [data.shape[2] if data is not None else None],  # Infer from data shape
                "height_um": [data.shape[1] if data is not None else None],
            }
        )

    # Fill empty target names with metal names or 'unlabeled'
    unlabel_count = 1
    for i, n in enumerate(names):
        if n == "" or n is None:
            if channels[i] == "" or channels[i] is None:
                names[i] = f"unlabeled_{unlabel_count}"
                unlabel_count += 1
            else:
                if fill_empty_metals:
                    names[i] = channels[i]
                else:
                    names[i] = None

    return data, channels, names, meta


def read_imc_file(
    path,
    acquisition_id=0,
    planes_to_load=None,
    only_metadata=False,
    fill_empty_metals=False,
):
    """
    Unified function to read either MCD or TXT IMC files.

    Parameters
    ----------
    path : str or Path
        Path to the IMC file (.mcd or .txt).
    acquisition_id : int, optional
        The acquisition ID to read. Only used for MCD files. Defaults to 0.
    planes_to_load : int or list of int, optional
        Indices of planes to load. If None, all planes are loaded.
    only_metadata : bool, optional
        If True, only return the metadata. Defaults to False.
    fill_empty_metals : bool, optional
        If True, fill empty names with metal names. Defaults to False.

    Returns
    -------
    data : numpy.ndarray
        The data from the IMC file.
    channels : list of str
        The channel labels from the IMC file.
    num_acquisitions : int
        The number of acquisitions (1 for TXT files, variable for MCD files).
    names : list of str
        The channel names from the IMC file.
    meta : pandas.DataFrame
        Metadata for the acquisition.

    Raises
    ------
    ValueError
        If the file format is not supported (.mcd or .txt).
    """
    path = Path(path)
    
    if path.suffix.lower() == ".mcd":
        return read_mcd(
            path=path,
            acquisition_id=acquisition_id,
            planes_to_load=planes_to_load,
            only_metadata=only_metadata,
            fill_empty_metals=fill_empty_metals,
        )
    elif path.suffix.lower() == ".txt":
        data, channels, names, meta = read_txt(
            path=path,
            only_metadata=only_metadata,
            fill_empty_metals=fill_empty_metals,
        )
        # TXT files only have one acquisition
        num_acquisitions = 1
        
        # Handle planes_to_load for TXT files
        if not only_metadata and planes_to_load is not None:
            if isinstance(planes_to_load, int):
                planes_to_load = [planes_to_load]
            data = data[planes_to_load]
            channels = np.array(channels)[planes_to_load].tolist()
            names = np.array(names)[planes_to_load].tolist()
        
        return data, channels, num_acquisitions, names, meta
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Only .mcd and .txt files are supported.")


def read_mcd(
    path,
    acquisition_id=0,
    planes_to_load=None,
    only_metadata=False,
    fill_empty_metals=False,
):
    """
    Reads an MCD file and returns data, channels, number of acquisitions, names, and metadata.

    Parameters
    ----------
    path : str or Path
        Path to the MCD file.
    acquisition_id : int, optional
        The acquisition ID to read. Defaults to 0.
    planes_to_load : int or list of int, optional
        Indices of planes to load. If None, all planes are loaded.
    only_metadata : bool, optional
        If True, only return the metadata. Defaults to False.
    fill_empty_metals : bool, optional
        If True, fill empty names with metal names. Defaults to False.

    Returns
    -------
    data : numpy.ndarray
        The data from the MCD file.
    channels : list of str
        The channel labels from the MCD file.
    num_acquisitions : int
        The number of acquisitions in the MCD file.
    names : list of str
        The channel names from the MCD file.
    meta : pandas.DataFrame
        Metadata (e.g., name and size) for the acquisition.

    Raises
    ------
    ValueError
        If the acquisition_id is invalid.
        If the file is not an MCD file.
    """
    if isinstance(planes_to_load, int):
        planes_to_load = [planes_to_load]

    data = None
    meta = None

    path = Path(path)
    if path.suffix == ".mcd":
        with MCDFile(path) as f:
            # Get the number of valid acquisitions
            num_acquisitions = _get_actual_num_acquisition(
                acquisitions=f.slides[0].acquisitions
            )
            if acquisition_id >= num_acquisitions:
                raise ValueError(
                    f"acquisition_id {acquisition_id} is larger than the number of acquisitions {num_acquisitions}. Possibly missing acquisition."
                )
            else:
                # Get the specified acquisition
                acquisition = f.slides[0].acquisitions[acquisition_id]
                if not only_metadata:
                    # Read data for the acquisition
                    data = f.read_acquisition(acquisition)

        # Get channel names and labels
        names = acquisition.channel_labels
        channels = acquisition.channel_names

        # Create metadata DataFrame
        meta = pd.DataFrame(
            {
                "id": [acquisition_id],
                "description": [acquisition.description],
                "width_um": [acquisition.width_um],
                "height_um": [acquisition.height_um],
            }
        )
    else:
        raise ValueError("File is not an MCD file.")

    # Fill empty target names with metal names or 'unlabeled'
    unlabel_count = 1
    for i, n in enumerate(names):
        if n == "":
            if channels[i] == "":
                names[i] = f"unlabeled_{unlabel_count}"
                unlabel_count += 1
            else:
                if fill_empty_metals:
                    names[i] = channels[i]
                else:
                    names[i] = None

    if not only_metadata:
        if planes_to_load is not None:
            # Select specified planes
            data = data[planes_to_load]
            channels = np.array(channels)[planes_to_load].tolist()
            names = np.array(names)[planes_to_load].tolist()

    return data, channels, num_acquisitions, names, meta


def export_to_tiffstack(
    path,
    export_path,
    export_panel=True,
    export_meta=True,
    export_errors=True,
    start_at=None,
    minimum_dimension=200,
):
    """
    Converts an MCD file to TIFF file stacks and metadata.

    Parameters
    ----------
    path : str or Path
        Path to the MCD file.
    export_path : str or Path
        Path to the folder where to export the TIFF files.
    export_panel : bool, optional
        If True, exports the panel file. Defaults to True.
    export_meta : bool, optional
        If True, exports the metadata file. Defaults to True.
    export_errors : bool, optional
        If True, exports the errors file if any errors are encountered. Defaults to True.
    start_at : int, optional
        Acquisition index to start at. If None, starts at 0. Defaults to None.
    minimum_dimension : int, optional
        Only extract ROIs greater than the specified size in height and width. Defaults to 200.

    Returns
    -------
    None
    """
    path = Path(path)

    # Get the number of acquisitions
    with MCDFile(path) as f:
        num_acquisitions = _get_actual_num_acquisition(
            acquisitions=f.slides[0].acquisitions
        )

    # Create a folder inside export path named after the MCD file stem
    export_dir = Path(export_path) / path.stem
    export_dir.mkdir(parents=True, exist_ok=True)

    # Lists to collect metadata and errors
    meta_list = []
    error_acq = []
    error_roiname = []
    error_list = []

    # Determine starting index
    start_index = start_at if start_at is not None else 0

    # Save each ROI within the MCD
    for i in range(start_index, num_acquisitions):
        try:
            # Read data for the acquisition
            data, channels, _, names, meta = read_mcd(path, acquisition_id=i)

            if (meta["height_um"][0] > minimum_dimension) and (meta["width_um"][0] > minimum_dimension):

                # Save image stack using tifffile
                tiff_filename = export_dir / f"{path.stem}_acq_{i}.tiff"
                tiff.imwrite(
                    tiff_filename, data, photometric="minisblack", metadata={"axes": "CYX"}
                )

                # Add metadata to list
                meta_list.append(meta.copy())

                # Export panel file
                if export_panel:
                    panel = pd.DataFrame(
                        {"channel_name": channels, "channel_label": names}
                    ).set_index("channel_name", drop=True)
                    panel_filename = export_dir / f"{path.stem}_acq_{i}.csv"
                    panel.to_csv(panel_filename)
            else:
                logging.info(f'ROI was too small, skipping: {meta["description"][0]}')
                error_acq.append(i)
                error_roiname.append(str(meta['description'][0]))
                error_list.append('ROI skipped as too small')

        except Exception as e:
            # Catch errors
            error_acq.append(i)
            if meta is not None and 'description' in meta:
                error_roiname.append(str(meta['description'][0]))
            else:
                error_roiname.append('')
            error_list.append(str(e))
            logging.error(f"Error in acquisition number {i}: {e}")

    # Save metadata
    if export_meta and meta_list:
        meta_df = pd.concat(meta_list)
        meta_df = meta_df.set_index("id", drop=True)
        meta_filename = export_dir / f"{path.stem}_meta.csv"
        meta_df.to_csv(meta_filename)

    # Save errors
    if export_errors and error_acq:
        errors_df = pd.DataFrame(
            {"id": error_acq, "description": error_roiname, "error": error_list}
        )
        errors_filename = export_dir / f"{path.stem}_errors.csv"
        errors_df.to_csv(errors_filename)
        logging.error(f"{len(error_acq)} errors encountered when unpacking {str(path.stem)}\n")


def export_txt_to_tiffstack(
    path,
    export_path,
    export_panel=True,
    export_meta=True,
):
    """
    Converts a TXT file to TIFF file stack and metadata.

    Parameters
    ----------
    path : str or Path
        Path to the TXT file.
    export_path : str or Path
        Path to the folder where to export the TIFF files.
    export_panel : bool, optional
        If True, exports the panel file. Defaults to True.
    export_meta : bool, optional
        If True, exports the metadata file. Defaults to True.

    Returns
    -------
    None
    """
    path = Path(path)
    
    # Create a folder inside export path named after the TXT file stem (cleaned)
    clean_stem = cleanstring(path.stem)
    export_dir = Path(export_path) / clean_stem
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read data from the TXT file
        data, channels, names, meta = read_txt(path)
        
        # Save image stack using tifffile (use clean stem)
        tiff_filename = export_dir / f"{clean_stem}.tiff"
        tiff.imwrite(
            tiff_filename, data, photometric="minisblack", metadata={"axes": "CYX"}
        )
        logging.info(f"Saved TIFF stack: {tiff_filename}")

        # Export panel file
        if export_panel:
            panel = pd.DataFrame(
                {"channel_name": channels, "channel_label": names}
            ).set_index("channel_name", drop=True)
            panel_filename = export_dir / f"{clean_stem}.csv"
            panel.to_csv(panel_filename)
            logging.info(f"Saved panel file: {panel_filename}")

        # Save metadata
        if export_meta:
            meta_filename = export_dir / f"{clean_stem}_meta.csv"
            meta.to_csv(meta_filename, index=False)
            logging.info(f"Saved metadata: {meta_filename}")

    except Exception as e:
        logging.error(f"Error processing TXT file {path}: {e}")
        raise


def export_imc_folder(
    path='IMC_files',
    export_path='tiff_stacks',
    merged_metadata_output_folder='metadata',
    export_panel=True,
    export_meta=True,
    export_errors=True,
    minimum_dimension=200,
):
    """
    Extracts every IMC file (.mcd or .txt) in a given directory into TIFF stacks, panel files, and a metadata table.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing IMC files. Defaults to 'IMC_files'.
    export_path : str or Path
        Path to the folder where to export the TIFF files. Defaults to 'tiff_stacks'.
    merged_metadata_output_folder : str or Path, optional
        Folder where merged metadata and panel files will be saved. Defaults to 'metadata'.
    export_panel : bool, optional
        If True, exports the panel files. Defaults to True.
    export_meta : bool, optional
        If True, exports the metadata files. Defaults to True.
    export_errors : bool, optional
        If True, exports the errors files if any errors are encountered. Defaults to True.
    minimum_dimension : int, optional
        Only extract ROIs greater than a specified size in height and width for MCD files. Defaults to 200.

    Returns
    -------
    None
    """
    path = Path(path)

    # Get a list of paths to IMC files in the directory
    mcd_paths = list(path.glob("*.mcd"))
    txt_paths = list(path.glob("*.txt"))
    
    # Check export path exists
    export_path = Path(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    if merged_metadata_output_folder:
        merged_metadata_output_folder = Path(merged_metadata_output_folder)
        merged_metadata_output_folder.mkdir(parents=True, exist_ok=True)

    # Process MCD files
    for mcd_file in mcd_paths:
        logging.info(f"Exporting MCD file: {mcd_file.stem}...")
        export_to_tiffstack(
            path=mcd_file,
            export_path=export_path,
            export_panel=export_panel,
            export_meta=export_meta,
            export_errors=export_errors,
            minimum_dimension=minimum_dimension
        )

    # Process TXT files
    for txt_file in txt_paths:
        logging.info(f"Exporting TXT file: {txt_file.stem}...")
        export_txt_to_tiffstack(
            path=txt_file,
            export_path=export_path,
            export_panel=export_panel,
            export_meta=export_meta,
        )

    # Merge metadata (same as before, but now handles both file types)
    if export_meta:
        # Concatenate all the metadata from the IMC files
        meta_data = []
        for x in export_path.iterdir():
            if x.is_dir():
                meta_file = x / f"{x.name}_meta.csv"
                if meta_file.exists():
                    m = pd.read_csv(meta_file)
                    m['source_file'] = x.name  # Directory name (already cleaned for TXT files)
                    # Determine file type from directory name patterns
                    # For MCD files, x.name matches the original stem
                    if any(mcd.stem == x.name for mcd in mcd_paths):
                        m['file_type'] = 'mcd'
                    # For TXT files, x.name matches the cleaned stem from export
                    elif any(cleanstring(txt.stem) == x.name for txt in txt_paths):
                        m['file_type'] = 'txt'
                    else:
                        m['file_type'] = 'unknown'
                    meta_data.append(m.copy())
                else:
                    warnings.warn(f"Meta file {meta_file} not found.")
        
        if meta_data:
            meta_data = pd.concat(meta_data)
            # Column which will later be used to exclude ROIs, e.g., if they are tests or had errors
            meta_data['import_data'] = True

            # Add data folder columns
            # For TXT files, we don't have acquisition IDs, so handle differently
            def create_folder_name(row):
                if row['file_type'] == 'mcd':
                    return f"{row['source_file']}_acq_{row['id']}"
                else:  # txt file - source_file should already be clean from export process
                    return row['source_file']
                    
            def create_tiff_path(row):
                if row['file_type'] == 'mcd':
                    return f"{row['source_file']}/{row['source_file']}_acq_{row['id']}.tiff"
                else:  # txt file - source_file should already be clean from export process
                    return f"{row['source_file']}/{row['source_file']}.tiff"

            meta_data['unstacked_data_folder'] = meta_data.apply(create_folder_name, axis=1)
            meta_data['tiff_stacks'] = meta_data.apply(create_tiff_path, axis=1)

            logging.info(f'Merged metadata for IMC files and ROIs saved to metadata.csv in {merged_metadata_output_folder} output folder')
            meta_data.to_csv(merged_metadata_output_folder / 'metadata.csv', index=None)

            # Create a blank dictionary if one doesn't already exist
            dictionary_file_path = merged_metadata_output_folder / 'dictionary.csv'
            if not dictionary_file_path.exists():
                logging.info('No dictionary file found, creating a blank file.')
                dictionary_file = meta_data.loc[:, ['unstacked_data_folder', 'description']].copy()
                dictionary_file.set_index('unstacked_data_folder', drop=True, inplace=True)
                dictionary_file.index.rename('ROI', inplace=True)
                dictionary_file['Example_1'] = 'Example_info'
                dictionary_file['Example_2'] = 1
                dictionary_file['Example_3'] = True
                dictionary_file.to_csv(dictionary_file_path)
            else:
                logging.info('Existing dictionary file found, not modifying.')
        else:
            logging.info('No metadata found to merge.')

    # Merge panels (enhanced with mapping functionality)
    if export_panel:
        # Load all panel files as dataframes, then get the unique ones
        panelfiles = _load_panel_files(export_path)
        panel_dfs = [pd.read_csv(file) for file in panelfiles]
        unique_panels, hash_to_files = _compare_dataframes(panel_dfs, panelfiles)

        # Add extra useful columns to the panel file(s) for use later
        for p in unique_panels.values():
            p = _panel_identify_used_channels(p, 'use_denoised')
            p['to_denoise'] = p['use_denoised']

            # Clean labels
            p['channel_label'] = [re.sub(r'\W+', '', str(x)) for x in p['channel_label']]

            # By default, don't use any raw
            p['use_raw'] = False

            # By default, don't remove outliers
            p['remove_outliers'] = False

        # Create panel mapping data
        panel_mapping_data = []
        hash_to_panel_id = {}
        
        # Save the panel(s)
        if len(unique_panels) == 1:
            logging.info("All panels are identical. Saving as 'panel.csv'.")
            unique_dataframe = next(iter(unique_panels.values()))
            unique_dataframe.to_csv(merged_metadata_output_folder / 'panel.csv', index=False)
            
            # Create mapping for single panel
            hash_val = next(iter(unique_panels.keys()))
            panel_file_name = 'panel.csv'
            hash_to_panel_id[hash_val] = panel_file_name
            
            for source_file in hash_to_files[hash_val]:
                panel_mapping_data.append({
                    'source_file': source_file,
                    'panel_file': panel_file_name
                })
        else:
            logging.warning(f"WARNING: Found {len(unique_panels)} unique panels. Saving them as 'panel_1.csv', 'panel_2.csv', etc.")
            for i, (hash_val, df) in enumerate(unique_panels.items(), start=1):
                panel_file_name = f'panel_{i}.csv'
                df.to_csv(merged_metadata_output_folder / panel_file_name, index=False)
                hash_to_panel_id[hash_val] = panel_file_name
                
                # Add mapping entries for this panel
                for source_file in hash_to_files[hash_val]:
                    panel_mapping_data.append({
                        'source_file': source_file,
                        'panel_file': panel_file_name
                    })
        
        # Save panel mapping file
        if panel_mapping_data:
            panel_mapping_df = pd.DataFrame(panel_mapping_data)
            panel_mapping_file = merged_metadata_output_folder / 'panel_mapping.csv'
            panel_mapping_df.to_csv(panel_mapping_file, index=False)
            logging.info(f'Panel mapping saved to {panel_mapping_file}')
            logging.info(f'Panel mapping shows which source files use which panel file:')
            for panel_file in panel_mapping_df['panel_file'].unique():
                source_files = panel_mapping_df[panel_mapping_df['panel_file'] == panel_file]['source_file'].tolist()
                logging.info(f'  {panel_file}: {source_files}')

    # Merge all errors (only relevant for MCD files)
    if export_errors:
        error_data = []
        for x in export_path.iterdir():
            if x.is_dir():
                error_file = x / f"{x.name}_errors.csv"
                if error_file.exists():
                    m = pd.read_csv(error_file)
                    m['source_file'] = x.name
                    error_data.append(m.copy())
        if error_data:
            error_data = pd.concat(error_data)
            logging.error(f'Merged list of import errors for IMC files and ROIs saved to errors.csv in {merged_metadata_output_folder} output folder')
            error_data.to_csv(merged_metadata_output_folder / 'errors.csv', index=None)


def _load_panel_files(directory):
    """
    Load all .csv files from a directory (including subdirectories)
    that do not have 'error' or 'meta' in their filenames.

    Parameters
    ----------
    directory : str or Path
        The directory to search for .csv files.

    Returns
    -------
    list of Path
        List of paths to the .csv files.
    """
    directory = Path(directory)
    csv_files = [f for f in directory.rglob('*.csv') if 'error' not in f.name.lower() and 'meta' not in f.name.lower()]
    return csv_files


def _compare_dataframes(dataframes, file_paths=None):
    """
    Compare all dataframes and identify unique ones.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        List of DataFrames to compare.
    file_paths : list of Path, optional
        List of file paths corresponding to the dataframes.

    Returns
    -------
    unique_dataframes : dict
        Dictionary of unique dataframes with their hash values as keys.
    hash_to_files : dict
        Dictionary mapping hash values to lists of file paths that use each panel.
    """
    unique_dataframes = {}
    hash_to_files = {}
    
    for i, df in enumerate(dataframes):
        hash_val = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        
        if hash_val not in unique_dataframes:
            unique_dataframes[hash_val] = df
            hash_to_files[hash_val] = []
        
        if file_paths is not None and i < len(file_paths):
            # Extract source file name from the panel file path
            # Panel files are named like "filename_acq_X.csv" or "filename.csv"
            panel_file = file_paths[i]
            source_name = panel_file.stem
            
            # For MCD files, remove "_acq_X" suffix to get source file name
            if '_acq_' in source_name:
                source_name = source_name.split('_acq_')[0]
            
            hash_to_files[hash_val].append(source_name)
    
    return unique_dataframes, hash_to_files


def _compare_lists_by_characters(list1, list2):
    """
    Compare two lists of strings to check pairwise if the strings in each position
    use the exact same characters.

    Parameters
    ----------
    list1 : list of str
        The first list of strings.
    list2 : list of str
        The second list of strings.

    Returns
    -------
    list of bool
        A list indicating for each pair whether the strings use the same characters.
    """
    def compare_strings(s1, s2):
        if s1 is None or s2 is None:  # Handle None values
            return False
        return sorted(str(s1)) == sorted(str(s2))

    return [compare_strings(s1, s2) for s1, s2 in zip(list1, list2)]


def _panel_identify_used_channels(df, col_to_add):
    """
    Adds a boolean column to a DataFrame indicating whether each channel contains data.

    A channel is considered to contain data if:
    - The 'channel_label' is not NaN.
    - The 'channel_label' is not simply a rearrangement of the characters in 'channel_name'.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'channel_name' and 'channel_label' columns.
    col_to_add : str
        Name of the new column to add to the DataFrame.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the new column added.
    """
    # By default, any channel with a label should have data. Blank labels should indicate no data.
    df[col_to_add] = df['channel_label'].notna()

    # Set to False any channel where the channel label and name are identical (even if rearranged), indicating a blank channel
    same_characters = _compare_lists_by_characters(df['channel_name'], df['channel_label'])
    df.loc[same_characters, col_to_add] = False

    return df


def export_mcd_folder(
    path='MCD_files',
    export_path='tiff_stacks',
    merged_metadata_output_folder='metadata',
    export_panel=True,
    export_meta=True,
    export_errors=True,
    minimum_dimension=200,
):
    """
    Legacy function for backward compatibility. Now calls export_imc_folder.
    Extracts every MCD file in a given directory into TIFF stacks, panel files, and a metadata table.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing MCD files. Defaults to 'MCD_files'.
    export_path : str or Path
        Path to the folder where to export the TIFF files. Defaults to 'tiff_stacks'.
    merged_metadata_output_folder : str or Path, optional
        Folder where merged metadata and panel files will be saved. Defaults to 'metadata'.
    export_panel : bool, optional
        If True, exports the panel files. Defaults to True.
    export_meta : bool, optional
        If True, exports the metadata files. Defaults to True.
    export_errors : bool, optional
        If True, exports the errors files if any errors are encountered. Defaults to True.
    minimum_dimension : int, optional
        Only extract ROIs greater than a specified size in height and width. Defaults to 200.

    Returns
    -------
    None
    """
    logging.info("Using legacy export_mcd_folder function. Consider using export_imc_folder for full IMC support.")
    # Just call the new unified function
    export_imc_folder(
        path=path,
        export_path=export_path,
        merged_metadata_output_folder=merged_metadata_output_folder,
        export_panel=export_panel,
        export_meta=export_meta,
        export_errors=export_errors,
        minimum_dimension=minimum_dimension,
    )


def unstack_tiffs(
    input_folder='tiff_stacks',
    unstacked_output_folder='tiffs',
    use_panel_files=True,
    use_metadata_file=True,
    save_image_data=False,
    return_dataframes=False,
):
    """
    Unpack TIFF stacks into individual channel images with sensible names.

    Parameters
    ----------
    input_folder : str or Path, optional
        Folder containing the TIFF stacks. Defaults to 'tiff_stacks'.
    unstacked_output_folder : str or Path, optional
        Folder where individual channel TIFFs will be saved. Defaults to 'tiffs'.
    use_panel_files : bool, optional
        If True, use panel files created for each ROI. Defaults to True.
    use_metadata_file : bool, optional
        If True, adds metadata for ROIs extracted from MCD files. Defaults to True.
    save_image_data : bool, optional
        If True, saves image data to CSV. Defaults to False.
    return_dataframes : bool, optional
        If True, returns DataFrames. Defaults to False.

    Returns
    -------
    channel_df : pandas.DataFrame or None
        DataFrame containing channel information.
    channels_list : list of str or None
        List of all data channel labels.
    image_data : pandas.DataFrame or None
        DataFrame containing information for all images.
    meta_data : pandas.DataFrame or None
        DataFrame containing metadata.
    """
    # Initialize variables to collect data
    image_data_list = []
    all_data_channels = []
    channel_df = pd.DataFrame()

    input_folder = Path(input_folder)
    output_folder = Path(unstacked_output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get a list of all the .tiff files in the input directory
    tiff_files = list(input_folder.rglob('*.tiff'))

    logging.info('Unpacking ROIs...')
    for roi_count, tiff_file in enumerate(tqdm(tiff_files)):
        # Read the image stack
        image_stack = tiff.imread(str(tiff_file))

        # Create output directory for the ROI
        tiff_folder_name = tiff_file.stem
        output_dir = output_folder / tiff_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_panel_files:
            # Read panel file
            panel_filename = f"{tiff_file.stem}.csv"
            panel_path = tiff_file.parent
            panel_file_path = panel_path / panel_filename

            if panel_file_path.exists():
                panel_df = pd.read_csv(panel_file_path, low_memory=False)

                # Remove awkward characters from channel names
                panel_df['channel_label'] = [re.sub(r'\W+', '', str(x)) for x in panel_df['channel_label']]

                panel_df['fullstack_path'] = str(tiff_file)
                panel_df['panel_filename'] = panel_filename
                panel_df['folder'] = tiff_folder_name
                panel_df['filename'] = ''

                # Identify non-blank channels
                panel_df = _panel_identify_used_channels(panel_df, 'data_channel')

                # Ensure the panel_df has the same number of entries as the number of channels
                if len(panel_df) != image_stack.shape[0]:
                    raise ValueError(
                        f"Panel file {panel_filename} does not match number of channels in image stack."
                    )

                # Append to image_data_list
                image_data_list.append(panel_df)
            else:
                raise FileNotFoundError(f"Panel file {panel_filename} not found.")

        for channel_count in range(image_stack.shape[0]):
            if use_panel_files:
                # Construct filename
                channel_name = panel_df.loc[channel_count, 'channel_name']
                channel_label = panel_df.loc[channel_count, 'channel_label']
                filename = (
                    f"{str(channel_count).zfill(2)}_{str(roi_count).zfill(2)}_"
                    f"{channel_name}_{channel_label}.tiff"
                )
                save_path = output_dir / filename
                tiff.imwrite(save_path, image_stack[channel_count])
                # Add filename to panel_df
                panel_df.at[channel_count, 'filename'] = filename

            else:
                file_name = f"{str(channel_count).zfill(2)}_{str(roi_count).zfill(2)}.tiff"
                save_path = output_dir / file_name
                tiff.imwrite(save_path, image_stack[channel_count])

    # Merge metadata
    if use_metadata_file:
        # Concatenate all the metadata from the MCD files
        meta_data = []
        for x in input_folder.iterdir():
            if x.is_dir():
                meta_file = x / f"{x.name}_meta.csv"
                if meta_file.exists():
                    m = pd.read_csv(meta_file)
                    m['mcd'] = x.name
                    meta_data.append(m.copy())
        if meta_data:
            meta_data = pd.concat(meta_data)
            logging.info(f'Metadata for MCD and ROIs saved to metadata.csv in {unstacked_output_folder} output folder')
            meta_data.to_csv(output_folder / 'metadata.csv', index=None)
        else:
            meta_data = None
    else:
        meta_data = None

    if use_panel_files:
        # Concatenate all image data into a single DataFrame
        image_data = pd.concat(image_data_list, ignore_index=True, sort=False)
        # Save image data to CSV
        if save_image_data:
            image_data.to_csv(output_folder / 'image_data.csv', index=False)

        # Create channel DataFrame from unique combinations of 'channel_name' and 'channel_label'
        channel_df = image_data[['channel_name', 'channel_label']].drop_duplicates()
        channel_df['channel'] = channel_df['channel_name'] + "_" + channel_df['channel_label']
        channel_df.to_csv(output_folder / 'channels_list.csv', index=False)

        # Add a column that indicates used channels
        channel_df = _panel_identify_used_channels(channel_df, 'contains_data')
        channels_list = channel_df.loc[channel_df['contains_data'], 'channel'].tolist()
        empty_channels_list = channel_df.loc[~channel_df['contains_data'], 'channel'].tolist()

        n_blank = len(empty_channels_list)
        if n_blank > 0:
            logging.info(f'The following {n_blank} empty channels were detected: {str(empty_channels_list)}')

        n_channels = len(channels_list)
        logging.info(f'The following {n_channels} channels were detected: {str(channels_list)}')

    else:
        image_data = None
        channels_list = None
        channel_df = None

    if return_dataframes:
        return channel_df, channels_list, image_data, meta_data
    else:
        return None, None, None, None


def qc_heatmap(
    directory='tiffs',
    quantile=0.95,
    save=True,
    channels=None,
    normalize=None,
    figsize=(10, 10),
    dpi=75,
    save_dir='qc_images',
    do_pca=True,
    annotate_pca=True,
    hide_figures=False,
):
    """
    Generate QC heatmaps and PCA plots to identify outliers.

    Parameters
    ----------
    directory : str or Path, optional
        Directory containing the images to analyze. Defaults to 'tiffs'.
    quantile : float, optional
        Quantile value for calculating image statistics. Defaults to 0.95.
    save : bool, optional
        If True, save the generated figures. Defaults to True.
    channels : list of str, optional
        List of specific channels to process. Defaults to None.
    normalize : str, optional
        Normalization method ('max' or 'zscore'). Defaults to None.
    figsize : tuple, optional
        Figure size for the heatmap. Defaults to (10, 10).
    dpi : int, optional
        DPI for the saved figures. Defaults to 75.
    save_dir : str or Path, optional
        Directory to save the QC images. Defaults to 'qc_images'.
    do_pca : bool, optional
        If True, perform PCA analysis. Defaults to True.
    annotate_pca : bool, optional
        If True, annotate PCA plots with ROI indices. Defaults to True.
    hide_figures : bool, optional
        If True, close figures after saving. Defaults to False.

    Returns
    -------
    None
    """
    # Validate and prepare the list of channels to process
    if channels is None:
        channels = []
    elif isinstance(channels, str):
        channels = [channels]

    # Create the directory for saving figures if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store extracted statistics
    channel_list = []
    roi_list = []
    img_max_list = []
    img_mean_list = []
    img_std_list = []
    img_quantile_list = []

    logging.info('Extracting data from images')

    # Iterate over each channel
    for channel in tqdm(channels, desc='Processing channels'):
        try:
            # Load images for the current channel
            img_collect, img_file_list, img_folders = load_imgs_from_directory(directory, channel, quiet=True)
        except ValueError as e:
            # If no images are found for the channel, skip it
            logging.info(f"Skipping channel '{channel}': {e}")
            continue

        # Iterate over each image and its corresponding folder
        for img, img_folder in zip(img_collect, img_folders):
            # Extract the ROI name from the folder path
            roi_name = img_folder.name

            # Calculate image statistics
            img_max = np.max(img)
            img_mean = np.mean(img)
            img_std = np.std(img)
            img_quantile = np.quantile(img, quantile)

            # Append statistics to the lists
            channel_list.append(channel)
            roi_list.append(roi_name)
            img_max_list.append(img_max)
            img_mean_list.append(img_mean)
            img_std_list.append(img_std)
            img_quantile_list.append(img_quantile)

    # Create a DataFrame from the collected statistics
    results_df = pd.DataFrame({
        'channel': channel_list,
        'ROI': roi_list,
        'max': img_max_list,
        'mean': img_mean_list,
        'std': img_std_list,
        'quantile': img_quantile_list
    })

    logging.info('Plotting results')

    # Define the metrics to analyze
    metrics = ['max', 'mean', 'std', 'quantile']

    # Iterate over each metric to create heatmaps and PCA plots
    for metric in metrics:
        # Create a pivot table with channels as rows and ROIs as columns
        results_pivot = results_df.pivot(index='channel', columns='ROI', values=metric)

        # Normalize the data if specified
        if normalize == 'max':
            # Normalize each row by its maximum value
            results_pivot = results_pivot.div(results_pivot.max(axis=1), axis=0)
            metric_label = f'{metric}_max_normalized'
        elif normalize == 'zscore':
            # Apply z-score normalization across the ROIs (columns)
            results_pivot = results_pivot.apply(zscore, axis=1)
            metric_label = f'{metric}_zscore'
        else:
            metric_label = metric

        # Plot the heatmap
        plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(results_pivot, xticklabels=True, yticklabels=True, cmap='viridis')
        plt.title(f'Heatmap of {metric_label}')
        plt.xlabel('ROI')
        plt.ylabel('Channel')

        # Save or display the heatmap
        if save:
            heatmap_filename = save_dir / f'{directory}_{metric_label}_heatmap.png'
            plt.savefig(heatmap_filename)
        if hide_figures:
            plt.close()
        else:
            plt.show()

        # Perform PCA if requested
        if do_pca:
            # Transpose the data to have samples as rows for PCA
            data_for_pca = results_pivot.T.fillna(0)

            # Standardize the data before applying PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_for_pca)

            # Apply PCA to reduce data to two components
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)

            # Create a scatter plot of the PCA results
            plt.figure(figsize=(10, 10))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], s=50)

            # Annotate points with ROI names if requested
            if annotate_pca:
                for idx, roi in enumerate(data_for_pca.index):
                    plt.annotate(roi, (pca_result[idx, 0], pca_result[idx, 1]))

            plt.title(f'PCA of {metric_label}')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.gca().set_aspect('equal', 'datalim')

            # Save or display the PCA plot
            if save:
                pca_filename = save_dir / f'{directory}_{metric_label}_PCA.png'
                plt.savefig(pca_filename)
            if hide_figures:
                plt.close()
            else:
                plt.show()


def combine_images(
    raw_directory="tiffs",
    processed_output_dir="processed",
    combined_dir="combined"
):
    """
    Combine raw and processed images into a new directory.

    Parameters
    ----------
    raw_directory : str or Path, optional
        Directory containing raw images. Defaults to "tiffs".
    processed_output_dir : str or Path, optional
        Directory containing processed images. Defaults to "processed".
    combined_dir : str or Path, optional
        Directory to save the combined images. Defaults to "combined".

    Returns
    -------
    None
    """
    combined_dir = Path(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Copy raw images into the combined directory
    logging.info(f'Copying original files from: {raw_directory}...')
    copytree(raw_directory, combined_dir, dirs_exist_ok=True)

    # Copy processed images over, overwriting duplicates
    logging.info(f'Adding in processed files from: {processed_output_dir}...')
    copytree(processed_output_dir, combined_dir, dirs_exist_ok=True)


def qc_check_side_by_side(
    channels=None,
    colourmap='jet',
    dpi=75,
    save=True,
    save_dir='qc_images',
    hide_images=True,
    raw_directory='tiffs',
    processed_output_dir='processed',
    quiet=True,
):
    """
    Compare raw and processed images side by side for quality control.

    Parameters
    ----------
    channels : list of str, optional
        List of channels to process. Defaults to None.
    colourmap : str, optional
        Colormap to use for images. Defaults to 'jet'.
    dpi : int, optional
        DPI for the saved figures. Defaults to 75.
    save : bool, optional
        If True, save the generated figures. Defaults to True.
    save_dir : str or Path, optional
        Directory to save the QC images. Defaults to 'qc_images'.
    hide_images : bool, optional
        If True, close figures after saving. Defaults to True.
    raw_directory : str or Path, optional
        Directory containing raw images. Defaults to 'tiffs'.
    processed_output_dir : str or Path, optional
        Directory containing processed images. Defaults to 'processed'.
    quiet : bool, optional
        If True, suppress print statements associated with loading images. Defaults to True.

    Returns
    -------
    None
    """
    if channels is None:
        channels = []
    elif isinstance(channels, str):
        channels = [channels]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    error_channels = []
    completed_channels = []

    for channel_name in channels:
        try:
            raw_img_collect, raw_img_file_list, raw_img_folders = load_imgs_from_directory(
                raw_directory, channel_name, quiet=quiet
            )
            pro_img_collect, pro_img_file_list, pro_img_folders = load_imgs_from_directory(
                processed_output_dir, channel_name, quiet=quiet
            )

            fig, axs = plt.subplots(len(raw_img_collect), 2, figsize=(10, 5 * len(raw_img_collect)), dpi=dpi)

            count = 0
            for raw_img, pro_img, raw_img_name in zip(raw_img_collect, pro_img_collect, raw_img_file_list):
                im1 = axs.flat[count].imshow(raw_img, vmin=0, vmax=0.5 * np.max(raw_img), cmap=colourmap)
                divider1 = make_axes_locatable(axs.flat[count])
                cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax1, orientation='vertical')
                axs.flat[count].set_ylabel(str(raw_img_name))
                count += 1

                im2 = axs.flat[count].imshow(pro_img, vmin=0, vmax=0.5 * np.max(pro_img), cmap=colourmap)
                divider2 = make_axes_locatable(axs.flat[count])
                cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax2, orientation='vertical')
                count += 1

            if save:
                fig.savefig(save_dir / f'{channel_name}.png')
                logging.info(f'QC images saved for {channel_name}')

            if hide_images:
                plt.close(fig)

            completed_channels.append(channel_name)

        except Exception as e:
            logging.error(f"Error in channel {channel_name}: {e}")
            error_channels.append(f"{channel_name}: {e}")

    logging.info(f"Successfully processed channels: {str(completed_channels)}")
    logging.info(f"Channels with errors: {str(error_channels)}")


def reassemble_stacks(
    restack_input_folder='combined',
    restacked_output_folder='tiffs_restacked',
    save_panel=True,
    re_order=None,
    ascending_sort_names=True
):
    """
    Reassemble individual channel TIFFs into stacks.

    Parameters
    ----------
    restack_input_folder : str or Path, optional
        Folder containing individual channel TIFFs. Defaults to 'combined'.
    restacked_output_folder : str or Path, optional
        Folder to save the reassembled stacks. Defaults to 'tiffs_restacked'.
    save_panel : bool, optional
        If True, save a CSV file detailing each channel in the new stack. Defaults to True.
    re_order : list of str, optional
        List of file names in the correct order. Defaults to None.
    ascending_sort_names : bool, optional
        If True and re_order is None, sort file names in ascending order. Defaults to True.

    Returns
    -------
    None
    """
    restack_input_folder = Path(restack_input_folder)
    output_folder = Path(restacked_output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    img_folders = [f for f in restack_input_folder.iterdir() if f.is_dir()]

    logging.info('Saving reassembled stacks...')
    for folder in tqdm(img_folders):
        tiff_files = list(folder.rglob('*.tiff'))
        file_names = [f.stem for f in tiff_files]
        file_df = pd.DataFrame({'File name': file_names, 'Path': tiff_files}).set_index('File name')

        if re_order:
            file_df = file_df.reindex(re_order)
        elif ascending_sort_names:
            file_df = file_df.sort_index(ascending=True)

        image_stack = [tiff.imread(str(file)).astype('float32') for file in file_df['Path']]
        image_stack = np.array(image_stack)

        save_path = output_folder / f"{folder.name}.tiff"
        tiff.imsave(save_path, image_stack.astype('float32'))

        if save_panel:
            panel_path = output_folder / f"{folder.name}.csv"
            file_df.to_csv(panel_path)

################ Script

if __name__ == "__main__":
    # Set up logging
    pipeline_stage = 'Preprocess'
    config = process_config_with_overrides()
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**filter_config_for_dataclass(config.get('general', {}), GeneralConfig))
    preprocess_config = PreprocessConfig(**filter_config_for_dataclass(config.get('preprocess', {}), PreprocessConfig))

    # Export .tiff stacks from IMC files (both .mcd and .txt)
    # Check if user is using new config option or legacy option
    imc_folder = getattr(general_config, 'imc_files_folder', None)
    if imc_folder and Path(imc_folder).exists():
        input_folder = imc_folder
        logging.info(f"Using IMC files folder: {input_folder}")
    else:
        input_folder = general_config.mcd_files_folder
        logging.info(f"Using legacy MCD files folder: {input_folder}")
    
    export_imc_folder(path=input_folder,
                      export_path=general_config.tiff_stacks_folder,
                      minimum_dimension=preprocess_config.minimum_roi_dimensions)

    # Unstack .tiffs into individual channel images
    unstack_tiffs(input_folder=general_config.tiff_stacks_folder,
                  unstacked_output_folder=general_config.raw_images_folder)
