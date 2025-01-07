# segmentation.py

# Standard library imports
import math
import re
from pathlib import Path

# Third-party library imports
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops
import scanpy as sc

# Import shared utilities and configurations
from .config_and_utils import *

def extract_single_cell(masks_folder='masks',
                        denoised_images_folder='processed',
                        raw_images_folder='tiffs',
                        metadata_folder='metadata',
                        save_directory='cell_tables'):
    """
    Analyze images to extract single-cell data and save the results in separate CSV files for each ROI.
    """
    # Create output directory for cell tables
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True)

    # Load panel file and clean labels
    panel = pd.read_csv(Path(metadata_folder) / 'panel.csv')
    panel['channel_label'] = [re.sub(r'\W+', '', str(x)) for x in panel['channel_label']]
    panel['filename'] = panel['channel_name'] + "_" + panel['channel_label']

    # Formula for circularity
    circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter) if r.perimeter > 0 else 0

    # Get list of ROIs from masks and images folders
    masks_folder = Path(masks_folder)
    denoised_images_folder = Path(denoised_images_folder)
    raw_images_folder = Path(raw_images_folder)

    mask_rois = sorted([x.stem for x in masks_folder.glob('*.tiff')])
    denoised_rois = sorted([x.name for x in denoised_images_folder.iterdir() if x.is_dir()])
    raw_rois = sorted([x.name for x in raw_images_folder.iterdir() if x.is_dir()])

    if not set(mask_rois) == set(denoised_rois):
        logging.warning('ROIs found in masks and image folders do not match entirely; only ROIs with masks will be used')
    else:
        logging.info('All masks have matching denoised image folders')

    for roi in [x for x in mask_rois if x in denoised_rois or x in raw_rois]:
        # Load the mask as a skimage label object
        mask_path = masks_folder / f"{roi}.tiff"
        mask = io.imread(mask_path)

        # Initialize a DataFrame to store results for the current ROI
        roi_df = pd.DataFrame()

        logging.info(f'{roi}: Extracting metrics for masks')

        # Add label numbers and centroid locations
        props = regionprops(mask)
        roi_df['Label'] = [prop.label for prop in props]
        roi_df['X_loc'] = [prop.centroid[1] for prop in props]  # X-coordinate of the centroid
        roi_df['Y_loc'] = [prop.centroid[0] for prop in props]  # Y-coordinate of the centroid
        roi_df['mask_area'] = [prop.area for prop in props]  # Mask area
        roi_df['mask_perimeter'] = [prop.perimeter for prop in props]  # Mask perimeter
        roi_df['mask_circularity'] = [circ(prop) for prop in props]  # Mask circularity
        roi_df['mask_largest_diameter'] = [prop.major_axis_length for prop in props]  # Largest diameter
        roi_df['mask_largest_diameter_angle'] = [np.degrees(prop.orientation) for prop in props]  # Angle of largest diameter

        # Dictionary to hold mean intensities for each image
        mean_intensities_dict = {}

        # Get list of channel labels to use either denoised or raw images to extract from
        channels_denoised = panel.loc[panel['use_denoised'], 'channel_label'].tolist()
        channels_denoised_filenames = panel.loc[panel['use_denoised'], 'filename'].tolist()

        channels_raw = panel.loc[panel['use_raw'], 'channel_label'].tolist()
        channels_raw_filenames = panel.loc[panel['use_raw'], 'filename'].tolist()

        logging.info(
            f'{roi}: Extracting mean cell intensities from {len(channels_denoised)} denoised images, and {len(channels_raw)} raw images')

        # Combine image folder with specific ROI folder
        for channel_list, images_folder, channel_filenames  in zip([channels_denoised, channels_raw],
                                                                  [denoised_images_folder, raw_images_folder],
                                                                  [channels_denoised_filenames, channels_raw_filenames]):

            image_path = images_folder / roi
            if not image_path.exists():
                logging.warning(f"No images found for ROI {roi} in {images_folder}")
                continue

            for channel, filename in zip(channel_list, channel_filenames):
                try:
                    # Get the best matching image from the directory
                    image_filename = get_filename(image_path, filename)

                    # Load the image
                    image = io.imread(image_path / image_filename)

                    # Calculate mean intensity for each label
                    mean_intensities = [region.mean_intensity for region in regionprops(mask, image)]

                    # Add to dictionary
                    mean_intensities_dict[channel] = mean_intensities
                except Exception as e:
                    logging.error(f"Error processing channel {channel} for ROI {roi}: {e}")

        # Add mean intensities to the DataFrame
        for image_name, intensities in mean_intensities_dict.items():
            roi_df[image_name] = intensities

        # Add the ROI name to the DataFrame, and make it first in list of columns
        roi_df['ROI'] = str(roi)
        cols = roi_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        roi_df = roi_df.loc[:, cols]

        # Save the DataFrame to a CSV file
        csv_file_path = save_directory / f"{roi}.csv"
        roi_df.to_csv(csv_file_path, index=False)
        logging.info(f'{roi}: Results saved to {csv_file_path}.')

def create_celltable(input_directory='cell_tables',
                     metadata_folder='metadata',
                     output_file='celltable.csv'):
    """
    Concatenate all CSV files in the specified directory into a single DataFrame,
    rename 'Label' to 'ObjectNumber', and save the final DataFrame.
    """
    input_directory = Path(input_directory)
    metadata_folder = Path(metadata_folder)

    # List to store individual DataFrames
    dfs = []

    # Load metadata for ROIs
    metadata = pd.read_csv(metadata_folder / 'metadata.csv', index_col='unstacked_data_folder')

    # Counters for ROIs
    included_rois = 0
    skipped_rois = 0

    # Iterate through each CSV file in the directory and append to the list
    for file in input_directory.glob('*.csv'):
        roi_name = file.stem
        if roi_name in metadata.index and metadata.loc[roi_name, 'import_data']:
            df = pd.read_csv(file)

            # Rename 'Label' column to 'ObjectNumber'
            df.rename(columns={'Label': 'ObjectNumber'}, inplace=True)

            dfs.append(df)
            logging.info(f'Imported {file.name} to cell table')
            included_rois += 1
        else:
            logging.warning(f'Skipped importing {file.name} to cell table')
            skipped_rois += 1

    logging.info(f'TOTAL: Imported {included_rois}, skipped {skipped_rois}.')

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Add 'Master_Index' as a copy of the index
    concatenated_df['Master_Index'] = concatenated_df.index

    # Save the final DataFrame
    concatenated_df.to_csv(output_file, index=False)
    logging.info(f"Concatenated DataFrame saved to {output_file}.")

    return concatenated_df

def normalise_markers(markers: pd.DataFrame, method_list: list) -> pd.DataFrame:
    """
    Normalises a markers DataFrame with a list of methods, in sequence.
    """
    for method in method_list:
        if method.startswith('q'):
            quantile = round(float(method[1:]), 5)
            markers = markers.div(markers.quantile(q=quantile)).clip(upper=1)
            markers.fillna(0, inplace=True)
            logging.info(f'Data normalised to {quantile} quantile')
        elif 'arcsinh' in method:
            cofactor = int(method[7:])
            markers = np.arcsinh(markers / cofactor)
            logging.info(f'Data Arcsinh adjusted with cofactor {cofactor}')
        elif method == 'log2':
            markers = np.log2(markers + 1)  # Avoid log(0)
            logging.info('Data Log2 adjusted')
        elif method == 'log10':
            markers = np.log10(markers + 1)  # Avoid log(0)
            logging.info('Data Log10 adjusted')
        else:
            logging.warning(f'Normalisation method {method} not recognized')

    return markers

# Helper function to identify boolean columns saved as objects, and convert them to true boolean columns
def convert_to_boolean(df):
    for col in df.select_dtypes(include=['object']).columns:
        # Check if all unique values are boolean-like
        unique_vals = pd.Series(df[col].dropna().unique().astype('string'))  # Convert to Pandas Series
        unique_vals = [x.lower() for x in unique_vals] # Lower case
        if all(val in {"true", "false", "yes", "no", "1", "0"} for val in unique_vals):
            # Convert to boolean
            df[col] = df[col].astype("boolean")
    return df

def create_anndata(celltable,
                   metadata_folder='metadata',
                   normalisation: list = ["q0.999"],
                   store_raw=False,
                   remove_channels=['DNA1', 'DNA3']):
    """
    Creates an AnnData object from the cell table and metadata.
    """
    # If a string, assume it's a path to a celltable
    if isinstance(celltable, str):
        logging.info(f'Loading cell table from path: {celltable}')
        celltable = pd.read_csv(Path(celltable), low_memory=False)

    metadata_folder = Path(metadata_folder)

    # Load panel and metadata files
    panel = pd.read_csv(metadata_folder / 'panel.csv', index_col=0)
    metadata = pd.read_csv(metadata_folder / 'metadata.csv', index_col='unstacked_data_folder')

    # Get list of channels from panel file
    panel['channel_label'] = [re.sub(r'\W+', '', str(x)) for x in panel['channel_label']]
    channels = panel.loc[panel['use_denoised'] | panel['use_raw'], 'channel_label'].tolist()
    logging.info(f'Channels found in cell table: {str(channels)}')

    # Extract raw markers
    markers_raw = celltable.loc[:, channels]

    # Marker normalisation
    if normalisation:
        markers = normalise_markers(markers_raw, normalisation)
        logging.info(f'Markers normalised: {normalisation}')
    else:
        markers = markers_raw
        logging.info('Markers not normalised, raw values used')

    # Create AnnData object using the normalised markers
    adata = sc.AnnData(markers)

    # Store raw / not normalised data
    if store_raw:
        adata.raw = adata.copy()

    # Add cellular obs from celltable
    non_channels = [x for x in celltable.columns.tolist() if x not in channels]
    logging.info(f'Non-channel data found in cell table: {str(non_channels)}')

    for col in non_channels:
        adata.obs[col] = celltable[col].tolist()

    # Add in metadata from metadata.csv
    adata.obs['ROI'] = adata.obs['ROI'].astype('category')
    adata.obs['ROI_name'] = adata.obs['ROI'].map(metadata['description'].to_dict())
    adata.obs['ROI_width'] = adata.obs['ROI'].map(metadata['width_um'].to_dict())
    adata.obs['ROI_height'] = adata.obs['ROI'].map(metadata['height_um'].to_dict())
    adata.obs['MCD_file'] = adata.obs['ROI'].map(metadata['mcd'].to_dict())

    # Add spatial coordinates
    adata.obsm['spatial'] = celltable[['X_loc', 'Y_loc']].to_numpy()

    # Process dictionary for additional metadata
    dictionary_path = metadata_folder / 'dictionary.csv'
    if dictionary_path.exists():
        dictionary_file = pd.read_csv(dictionary_path, index_col='ROI')

        # Automatically convert boolean-like columns
        dictionary_file = convert_to_boolean(dictionary_file)

        # Get list of columns/metadata from the dictionary file
        cols = [x for x in dictionary_file.columns if 'Example' not in x and 'description' not in x]

        if len(cols) > 0:
            logging.info(f'Dictionary file found with the following columns: {str(cols)}')

            # Ensure `adata.obs` is not a view
            adata.obs = adata.obs.copy()

            for c in cols:
                # Map the data from the dictionary to the adata.obs
                mapped_data = adata.obs['ROI'].map(dictionary_file[c].to_dict())

                # Convert to the appropriate type
                adata.obs[c] = mapped_data.astype(dictionary_file[c].dtype)

            # Make sure boolean columns properly converted
            adata.obs = convert_to_boolean(adata.obs)

        else:
            logging.info(
                'Dictionary file found but was empty. Edit dictionary.csv in the metadata folder to add extra sample-level metadata!'
            )
    else:
        logging.info('No dictionary file found.')

    # Remove specified channels
    if remove_channels:
        remove_channels_list = [
            channel for channel in adata.var_names
            if any(substring in channel for substring in remove_channels)
        ]
        logging.info(f'Removing channels: {str(remove_channels_list)}')
        adata = adata[:, [x for x in adata.var_names if x not in remove_channels_list]]

    logging.info('AnnData created successfully')
    return adata

if __name__ == "__main__":
    # Define the pipeline stage
    pipeline_stage = 'Segmentation'

    # Load configuration
    config = process_config_with_overrides()

    # Setup logging
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**config.get('general', {}))
    seg_config = SegmentationConfig(**config.get('segmentation', {}))

    # Extract single-cell info for each ROI
    if seg_config.create_roi_cell_tables:
        extract_single_cell(
            masks_folder=general_config.masks_folder,
            denoised_images_folder=general_config.denoised_images_folder,
            raw_images_folder=general_config.raw_images_folder,
            metadata_folder=general_config.metadata_folder,
            save_directory=general_config.celltable_folder,
        )
    else:
        logging.info(f'SKIPPING creating cell tables for ROIs...')

    # Concatenate individual ROIs into a cell table
    if seg_config.create_master_cell_table:
        celltable = create_celltable(
            input_directory=general_config.celltable_folder,
            metadata_folder=general_config.metadata_folder,
            output_file=seg_config.celltable_output
        )
    else:
        logging.info(f'SKIPPING creating master cell table...')
    
    # Create an AnnData object
    if seg_config.create_anndata:

        if 'celltable' not in locals():
            celltable = pd.read_csv(seg_config.celltable_output)
            logging.info(f'Loading master cell table: {str()}')

        adata = create_anndata(
            celltable,
            metadata_folder=general_config.metadata_folder,
            normalisation=seg_config.marker_normalisation,
            store_raw=seg_config.store_raw_marker_data,
            remove_channels=seg_config.remove_channels_list
        )

        # Save AnnData
        adata.write_h5ad(seg_config.anndata_save_path)
        logging.info(f'Saved AnnData: {seg_config.anndata_save_path}')
        
    else:
        logging.info(f'SKIPPING creating/saving AnnData...')
        
    logging.info('Segmentation pipeline finished')