# Preprocessing scripts for IMC data on University of Manchester CSF3 :test_tube: :electric_plug: :bee:
 
> [!CAUTION]
> This is all a work in progress! I will do my best to fix and bugs, but this is a fairly complex pipeline with a lot of moving parts. Getting IMC_Denoise to work is especially frustrating, requiring alternating conda and pip installs which means we can't install from a .yml file

> [!IMPORTANT]
> You will need an account on CSF3 with access to GPUs - these are availble on free accounts (free-at-point-of-access'), but you need to contact Research IT to get access. You will also need a fair amount of disk space free.


## Overview
These scripts were designed to do all the pre-processing for IMC data on the CSF3. The scripts are designed to be run in sequence via a single job file. All the scripts use a common YAML configuration file to specify  directories, pipeline behaviors, segmentation parameters, etc.


# Pipeline scripts


## `preprocesing.py` :receipt:

**Purpose:** Extracts the .TIFF images from the MCD files

**Inputs:** MCD files (`MCD_files`)

**Outputs:**
- `.tiff files (tiffs)` - Raw tiff files for each ROI and channel, named sequentially as they were imaged
- `metadata/metadata.csv` - Metadata about the ROIs (e.g names) from the MCD file
- `metadata/panel.csv` - Channels detected in MCD file.
- `metadata/dictionary.csv`  - Blank dictionary file for adding sample level metadata
- `metadata/errors.csv`  - Any errors encountered when extracting raw data, occassionally some ROIs get corrupted

**User input required:**
- Edit `panel.csv` to specify channels that will be denoised, and whether raw or denoised images should be used in segmentation
- Edit `dictionary.csv` to add sample-level metadata for the ROIs, e.g. case or treatments. This information will be incorporated into the AnnData.


## `denoising.py` :receipt:

**Purpose:** Uses IMC Denoise to denoise each channel

**Inputs:** Raw tiff files (`tiffs`), `panel.csv` for channel information

**Outputs:** 
- Denoised tiff files (`processed`)
- `QC/denoising` - Side-by-side comparisson of raw and denoised channels.


## `createmasks.py` :receipt:

**Purpose:** Uses CellPose3 to create segmentation masks based upon the DNA channel, including creating QC images. Can also do a parameter scan to fine tune the CellPose3 parameters. Nuclei are identified, followed by an expansion of a specified number of pixels (1 pixel = 1 micron in IMC data), usually 1 for a conservative approximation of cytoplasm that usually allows good cell phenotyping without excessive spillover of neighbouring cells (https://www.nature.com/articles/s41467-021-26214-x)

**Inputs:** Denoised .tiff files (`processed`), `panel.csv` for channel information

**Outputs:** 
- `masks` - Masks for each ROI
- `QC/Segmentation_overlay` - QC with sucessfully segmented cells in green, and excluded cells too small/big in red.


## `segmentation.py` :receipt:

**Purpose:** Uses the masks to segment all the denoised images, create cell tables with the raw data, which is then use to create an AnnData object.

**Inputs:** Denoised .tiff files (`processed`), `masks`, `dictionary.csv` for importing sample-level metadata

**Outputs:** 
- `celltable.csv` - Master cell table with all cell information in
- `cell_tables` - Cell tables for each ROI
- `anndata.h5ad` - Imported and normalised data, saved as AnnData.
- `QC/Segmentation_QC.csv` - QC for segmentation.


## `basic_process.py` :receipt:

**Purpose:** Performs basic pre-processing of the data, including batch correcting, calculating UMAPs, and initial leiden clustering.

**Inputs:** `anndata.h5ad`

**Outputs:** `anndata_processed.h5ad`. This can then be used for downstream analyses in my other notebooks.

> [!TIP]
> The outputs of these scripts (i.e. denoised images, `anndata_processed.h5ad`, etc) will (hopefully) dovetail into my other IMC notebooks for doing downstream spatial analyses

## Accessory scripts :ledger:
- `generate_config.py`: Creates a config.yaml file with the default parameters that can then be modified


# Installation and setup on CSF3	:computer:
1. Login to CSF3 command line.
2. Clone this repo into your home directory: `git clone https://github.com/dr-michael-haley/imcanalysis.git`
3. Install miniconda (if you don't already have it setup, anaconda should also work, but I've not tested it).
4. Create the conda environments called `segmentation` using the YML file in this diretory (`conda env create -f segmentation.yml`)
5. Create the environment for IMC Denoise. This is a very fiddly package to get working, and we can't just use a .yml file to create the environment. Instead, we have to create it manually using the following commands:
```
conda create -n 'IMC_Denoise' python=3.6.13
conda activate IMC_Denoise
conda install -c anaconda brotlipy=0.7.0 pandas=1.1.5 matplotlib==3.3.4 scipy==1.4.1 scikit-learn==0.24.2 tifffile pyyaml
pip install tensorflow==2.2.0 keras==2.3.1
conda install -c anaconda cudnn=7.6.5 cudatoolkit=10.1.243
```
6. We also need to install the IMC_Denoise package into this same environment:
```
git clone https://github.com/PENGLU-WashU/IMC_Denoise.git
cd IMC_Denoise
pip install -e .
```
8. Navigate to where you have downloaded the repo (`imcanalysis`) and install the SpatatialBiologyToolkit package into *BOTH* environments using the following command: `pip install --no-deps -e .` . This will skip the requirements, but allow easy update of the scripts with a `git pull`
9. Navigate to your `scratch` directory (or wherever you have a lot of space available), and upload the MCD files into a folder called `MCD_files`
10. Activate either `segmentation` or `IMC_Denoise` environments and create a blank `config.yaml` using the following command: `python -m SpatialBiologyToolkit.scripts.generate_config`.
11. Modify the `config.yaml` file appropriately. I will cover the various settings for this below, but this is the file in which all the configurations are stored, and the defaults are a good start.
12. Setup a job file - I've uploaded an example here (`job.txt`). By deafult all the stages are run, but I usually just comment out the steps as I've confirmed that they have run succesfully. You will also notice we use the v100 GPUs - these are free at point of access if you ask Research IT to give you access.
13. Submit the job


# Settings in Config file (`config.yaml`) :books:
All settings are stored here under various headings associated with the different parts of the pipeline. If you're unsure about YAML formatting, check online.


## General Configuration (`general`)

**Purpose**:
Specifies directories for input, output, and intermediate data. Adjust these paths to match your project's file structure, though I'd suggest just using the defaults.

**Parameters:**
- mcd_files_folder (str, default: 'MCD_files'): Directory containing MCD files (raw acquisition data).
- metadata_folder (str, default: 'metadata'): Directory where metadata files describing ROIs, imaging parameters, and other reference info will be stored.
- qc_folder (str, default: 'QC'): Where quality control (QC) outputs, such as overlay images and summary CSVs, are saved.
- masks_folder (str, default: 'masks'): Directory for saving final segmentation masks.
- celltable_folder (str, default: 'cell_tables'): Location where single-cell data tables (CSV files) are stored after segmentation.
- tiff_stacks_folder (str, default: 'tiff_stacks'): Directory for multi-channel TIFF stacks to be extracted at preprocessing step.
- raw_images_folder (str, default: 'tiffs'): Directory where raw TIFF images will be stored.
- denoised_images_folder (str, default: 'processed'): Directory for denoised/processed images will be stored, and used as input to segmentation.


## Preprocessing Configuration (`preprocess`)

**Purpose**:
Controls minimal preprocessing steps, such as validating ROI sizes before processing.

**Parameters**:
- minimum_roi_dimensions (int, default: 200): Minimum dimension required for ROIs (in pixels). ROIs smaller than this (usually test regions) are skipped and flagged.


## Denoising Configuration (`denoising`)

**Purpose**:
Handles the image denoising pipeline. You can turn denoising on/off, select methods, channels, and QC parameters.

> [!IMPORTANT]
> Most of these parameters are covered in the IMC_Denoise documentation (https://github.com/PENGLU-WashU/IMC_Denoise/tree/main)

**Parameters**:
- run_denoising (bool, default: True): Enable or disable the denoising step. This is included in case you wanted to run the QC steps without re-running any denoising.
- method (str, default: 'deep_snf'): Denoising method. Options: 'deep_snf' or 'dimr'. Adjusts which denoising algorithm is applied.
- channels (List[str], default: []): Specify which channels to denoise. Empty means all or as defined in the `panel.csv` created in the metadata directory.
- n_neighbours (int, default: 4), n_iter (int, default: 3), window_size (int, default: 3): General parameters for both denoising methods, controlling complexity and extent of denoising operations.

*For deep_snf method-specific parameters*:

- patch_step_size (int, default: 100): The patch size step used when training/denoising with DeepSNF.
- train_epochs (int, default: 75), train_initial_lr (float, default: 0.001): Number of training epochs and initial learning rate for DeepSNF model training.
- train_batch_size (int, default: 200): Batch size for DeepSNF training.
- pixel_mask_percent (float, default: 0.2): Percentage of pixels masked during training.
- val_set_percent (float, default: 0.15): Fraction of data used as validation set.
- loss_function (str, default: 'I_divergence'): Loss function for DeepSNF training.
- loss_name (Optional[str]): Optional name to save loss curves/results.
- weights_save_directory (Optional[str]): Where to save/load model weights.
- is_load_weights (bool, default: False): If True, load existing weights instead of training from scratch.
- lambda_HF (float, default: 3e-6): Regularization parameter for high-frequency details.
- network_size (str, default: 'normal'): Network size preset.

*QC parameters for denoising*:

- run_QC (bool, default: True): Whether to produce QC images post-denoising.
- colourmap (str, default: 'jet'): Colormap for QC images.
- dpi (int, default: 100): Resolution of QC images.
- qc_image_dir (str, default: 'denoising'): Directory to store QC images.
- skip_already_denoised (bool, default: True): Skip re-denoising if output files already exist.


## Create Masks Configuration (`createmasks`)
> [!IMPORTANT]
> Many of these parameters are covered in the CellPose3 documentation (https://cellpose.readthedocs.io/en/latest/settings.html)

**Purpose**:
Controls nucelar segmentation with Cellpose, including preprocessing, thresholds, parameter scanning, and QC overlay generation. n

**Parameters**:
- specific_rois (Optional[List[str]]): List of ROIs to segment. If None, all available ROIs are processed.
- dna_image_name (str, default: 'DNA1'): Name of the DNA channel image to segment.
- cellpose_cell_diameter (float, default: 10.0): Approximate cell diameter in pixels for Cellpose. **This is important to try and tweak for accurate segmentation**
- upscale_ratio (float, default: 1.7): Upscaling ratio if run_upscale is True.
- expand_masks (int, default: 1): Distance to expand nuclear segmentation masks created by CellPose. Default of 1 pixel (um) is advised as a conservative approach that usually allows good phenotyping without excessive spill over from neighbouring cells (see https://www.nature.com/articles/s41467-021-26214-x). **This is important to try and tweak for accurate segmentation, and will depend on cell density**
- perform_qc (bool, default: True): Produce QC overlay images for segmentation results.
- qc_boundary_dilation (int, default: 0): Thickness of mask outlines in QC images. Increase to make outlines appear thicker.
- min_cell_area (Optional[int], default: 15), max_cell_area (Optional[int], default: 200): Minimum and maximum acceptable cell size (in pixels). Objects outside this range are excluded.
- cell_pose_model (str, default: 'nuclei'): Cellpose model type (e.g., 'nuclei', 'cyto', etc.).
- cellprob_threshold (float, default: 0.0): Threshold for cell probability in Cellpose. **This is important to try and tweak for accurate segmentation**
- flow_threshold (float, default: 0.4): Threshold for flow error in Cellpose, controlling segmentation quality. **This is important to try and tweak for accurate segmentation**
- run_deblur (bool, default: True): If True, apply deblur preprocessing before segmentation.
- run_upscale (bool, default: True): If True, apply upscaling preprocessing before segmentation.
- image_normalise (bool, default: True): If True, normalize image intensity before Cellpose.
- image_normalise_percentile_lower (float, default: 0.0), image_normalise_percentile_upper (float, default: 99.9): Percentiles for intensity normalization range.
- dpi_qc_images (int, default: 300): DPI for saved QC images, higher values produce higher resolution overlays.

*Parameter scanning fields*:
- run_parameter_scan (bool, default: False): If True, the pipeline runs a parameter scan rather than normal segmentation.
- param_a (Optional[str], default: 'cellprob_threshold'): The first parameter to vary (X-axis of parameter grid).
- param_a_values (List[Any], default: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]): Values to test for param_a.
- param_b (Optional[str], default: 'flow_threshold'): The second parameter to vary (Y-axis of parameter grid).
- param_b_values (List[Any], default: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]): Values to test for param_b.
- window_size (Optional[int], default: 250): Size of the windowed sub-region (patch) to show in QC grids for parameter scan. This only affects displayed QC images, not the segmentation or stats on the full image.
- num_rois_to_scan (int, default: 3): If no scan_rois specified, randomly choose this many ROIs for parameter scanning.
- scan_rois (Optional[List[str]]): If provided, run parameter scan on these specific ROIs.


## Segmentation Configuration (`segmentation`)

**Purpose:**
Controls downstream data processing after segmentation, such as storing results in AnnData or adjusting marker normalization.

**Parameters:**
- celltable_output (str, default: 'celltable.csv'): Name of the CSV file to store aggregated single-cell data from segmentation.
- marker_normalisation (List[str], default: ["q0.999"]): List of normalization methods for markers (e.g., quantile normalization). The default is "q0.999", which normalises to the 99.9th percentile of intensity within each marker. This prevents normalising to outlier cells. All values are therefore normalised from 0 to 1.
- store_raw_marker_data (bool, default: False): If True, store raw marker intensities as well as normalized data.
- remove_channels_list (List[str], default: ['DNA1', 'DNA3']): Channels to remove before data analysis, often non-informative DNA stains, but could also be 
- anndata_save_path (str, default: 'anndata.h5ad'): Path to store the final single-cell data as an AnnData file.


## Basic Process Configuration (`basic_process`)

**Purpose:**
Controls additional processing steps like batch correction and clustering (e.g., PCA, UMAP, Leiden clustering).

**Parameters:**
- input_adata_path (str, default: 'anndata.h5ad'): Input AnnData file path from previous steps.
- output_adata_path (str, default: 'anndata_processed.h5ad'): Output AnnData file after processing (e.g., batch correction, or none).
- batch_correction_method (Optional[str]): Which batch correction method to use (e.g., 'bbknn', 'harmony', or none).
- batch_correction_obs (Optional[str]): Key in .obs to use for batch correction grouping.
- n_for_pca (Optional[int]): Number of principal components to compute. If None, defaults set by pipeline.
- leiden_resolutions_list (List[float], default: [0.3, 1.0]): List of resolutions for Leiden clustering, generating multiple clusterings.
- umap_min_dist (float, default: 0.5): UMAP min_dist parameter controlling embedding compactness.
