import pandas as pd
import numpy as np
import json
from skimage import io, color, draw, measure
import matplotlib.pyplot as plt
import tifffile

def create_visium_mask(folder: str, show_mapping: bool = True) -> None:
    """
    Create masks for Visium spots based upon a 'spatial' directory output from Spaceranger.

    Parameters
    ----------
    folder : str
        The 'spatial' folder which is the output of Spaceranger.
    show_mapping : bool, optional
        If True, displays the mask over the tissue image (default is True).

    Output
    ------
    Saves a 16-bit mask file in the specified folder.
    """
    
    # Load the high-resolution tissue image using skimage
    tissue_image_path = f"{folder}/tissue_hires_image.png"
    tissue_image = io.imread(tissue_image_path)
    y_dim, x_dim = tissue_image.shape[:2]

    # Load the tissue positions list
    try:
        tissue_positions = pd.read_csv(f"{folder}/tissue_positions_list.csv")
    except pd.errors.EmptyDataError:
        tissue_positions = pd.read_csv(f"{folder}/tissue_positions_list.csv", header=None)

    tissue_positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

    # Load the scale factor
    with open(f"{folder}/scalefactors_json.json") as json_file:
        scale_factors = json.load(json_file)
    tissue_hires_scalef = scale_factors['tissue_hires_scalef']

    # Adjust the position coordinates and circle size
    tissue_positions['pxl_row'] = tissue_positions['pxl_row_in_fullres'] * tissue_hires_scalef
    tissue_positions['pxl_col'] = tissue_positions['pxl_col_in_fullres'] * tissue_hires_scalef
    spot_diameter = tissue_hires_scalef * scale_factors['spot_diameter_fullres']

    # Create a blank numpy array
    mask_array = np.zeros((y_dim, x_dim), dtype=np.uint8)

    # Create scaled circles in the array at specified locations
    for index, row in tissue_positions.iterrows():
        rr, cc = draw.disk((row['pxl_row'], row['pxl_col']), spot_diameter / 2)
        mask_array[rr, cc] = 255

    # Convert the numpy array into labels
    labeled_mask = measure.label(mask_array)

    # Add a column to the dataframe for label ids
    label_ids = []
    for index, row in tissue_positions.iterrows():
        label_id = labeled_mask[int(row['pxl_row']), int(row['pxl_col'])]
        label_ids.append(label_id)
    tissue_positions['label_id'] = label_ids

    # Save the dataframe and the mask file
    tissue_positions.to_csv(f"{folder}/spot_barcode_to_label.csv", index=False)
    tifffile.imwrite(f"{folder}/mask.tiff", labeled_mask.astype('uint16'), dtype='uint16')

    if show_mapping:
        # Visualize the mask over the tissue image
        mask_rgb = color.label2rgb(labeled_mask, tissue_image, colors=['red'], alpha=0.1, bg_label=0)
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_rgb)
        plt.axis('off')
        plt.show()
