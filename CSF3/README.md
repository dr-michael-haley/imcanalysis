# Preprocessing scripts for CSF3

## Overview
These scripts were designed to do all the pre-processing for IMC data on the CSF3. The scripts are designed to be run in sequence via a single job file. All the scripts use a common YAML configuration file to specify  directories, pipeline behaviors, segmentation parameters, etc.


### Pipeline scripts
- `preprocesing.py`: Extracts the .TIFF images from the MCD files
- `denoising.py`: Uses IMC Denoise to denoise each channel
- `createmasks.py`: Uses CellPose3 to create segmentation makes based upon the DNA channel, including creating QC images. Can also do a parameter scan to fine tune the CellPose3 parameters.
- `segmentation.py`: Uses the masks to segment all the denoised images, create cell tables with the raw data, which is then use to create an AnnData object.
- `basic_process.py`: Performs basic pre-processing of the data, including batch correcting, calculating UMAPs, and initial leiden clustering.


### Accessory scripts
- `generate_config.py`: Creates a config.yaml file with the default parameters that can then be modified


## Installation and setup
1. Login to CSF3 command line.
2. Clone this repo into your home directory, you can use my access key: `git clone https://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@github.com/dr-michael-haley/imcanalysis.git`
3. Install conda or miniconda (if you don't already have it setup).
4. Create conda environments using the two YML files in this diretory (`IMC_Denoise.yml`, `segmentation.yml`). We need two because of compatability issues between packages.
5. Navigate to where you have downloaded the repo (`imcanalysis`) and install the SpatatialBiologyToolkit package into *BOTH* environments using the following command: `pip install --no-deps -e .` . This will skip the requirements, but allow easy update of the scripts with a `git pull`
6. Navigate to your `scratch` directory (or wherever you have a lot of space available), and upload the MCD files into a folder called `MCD_files`
7. Activate either `segmentation` or `IMC_Denoise` environments and create a blank `config.yaml` using the following command: `python -m SpatialBiologyToolkit.scripts.generate_config`.
8. Modify the `config.yaml` file appropriately. I will cover the various settings for this below, but this is the file in which all the configurations are stored, and the defaults are a good start.
9. Setup a job file - I've uploaded an example here (`job.txt`). By deafult all the stages are run, but I usually just comment out the steps as I've confirmed that they have run succesfully. You will also notice we use the v100 GPUs - these are free at point of access if you ask Research IT to give you access.
10. Submit the job

## Settings in Config file


### General Configuration (general)

**Purpose**:
Specifies directories for input, output, and intermediate data. Adjust these paths to match your project's file structure.


**Parameters:**
- mcd_files_folder (str, default: 'MCD_files'): Directory containing MCD files (raw acquisition data).
- metadata_folder (str, default: 'metadata'): Directory where metadata files describing ROIs, imaging parameters, and other reference info will be stored.
- qc_folder (str, default: 'QC'): Where quality control (QC) outputs, such as overlay images and summary CSVs, are saved.
- masks_folder (str, default: 'masks'): Directory for saving final segmentation masks.
- celltable_folder (str, default: 'cell_tables'): Location where single-cell data tables (CSV files) are stored after segmentation.
- tiff_stacks_folder (str, default: 'tiff_stacks'): Directory for multi-channel TIFF stacks to be extracted at preprocessing step.
- raw_images_folder (str, default: 'tiffs'): Directory where raw TIFF images will be stored.
- denoised_images_folder (str, default: 'processed'): Directory for denoised/processed images will be stored, and used as input to segmentation.

### Preprocessing Configuration (preprocess)

**Purpose**:
Controls minimal preprocessing steps, such as validating ROI sizes before processing.

**Parameters**:
- minimum_roi_dimensions (int, default: 200): Minimum dimension required for ROIs (in pixels). ROIs smaller than this (usually test regions) are skipped and flagged.

### Denoising Configuration (denoising)

**Purpose**:
Handles the image denoising pipeline. You can turn denoising on/off, select methods, channels, and QC parameters.

Most of these parameters are covered in the IMC_Denoise documentation! https://github.com/PENGLU-WashU/IMC_Denoise/tree/main


**Parameters**:
- run_denoising (bool, default: True): Enable or disable the denoising step. This is included in case you wanted to run the QC steps without re-running any denoising.
- method (str, default: 'deep_snf'): Denoising method. Options: 'deep_snf' or 'dimr'. Adjusts which denoising algorithm is applied.
- channels (List[str], default: []): Specify which channels to denoise. Empty means all or as defined in the `panel.csv` created in the metadata directory.
- n_neighbours (int, default: 4), n_iter (int, default: 3), window_size (int, default: 3): General parameters for both denoising methods, controlling complexity and extent of denoising operations.

*For deep_snf method-specific parameters*:

```python
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
```

*QC parameters for denoising*:
`
- run_QC (bool, default: True): Whether to produce QC images post-denoising.
- colourmap (str, default: 'jet'): Colormap for QC images.
- dpi (int, default: 100): Resolution of QC images.
- qc_image_dir (str, default: 'denoising'): Directory to store QC images.
- skip_already_denoised (bool, default: True): Skip re-denoising if output files already exist.

### Create Masks Configuration (createmasks)

**Purpose**:
Controls cell segmentation with Cellpose, including preprocessing, thresholds, parameter scanning, and QC overlay generation.

**Parameters**:
- specific_rois (Optional[List[str]]): List of ROIs to segment. If None, all available ROIs are processed.
- dna_image_name (str, default: 'DNA1'): Name of the DNA channel image to segment.
- cellpose_cell_diameter (float, default: 10.0): Approximate cell diameter in pixels for Cellpose.
- upscale_ratio (float, default: 1.7): Upscaling ratio if run_upscale is True.
- expand_masks (int, default: 1): Distance to expand segmentation masks.
- perform_qc (bool, default: True): Produce QC overlay images for segmentation results.
- qc_boundary_dilation (int, default: 0): Thickness of mask outlines in QC images. Increase to make outlines appear thicker.
- min_cell_area (Optional[int], default: 15), max_cell_area (Optional[int], default: 200): Minimum and maximum acceptable cell size (in pixels). Objects outside this range are excluded.
- cell_pose_model (str, default: 'nuclei'): Cellpose model type (e.g., 'nuclei', 'cyto', etc.).
- cellprob_threshold (float, default: 0.0): Threshold for cell probability in Cellpose.
- flow_threshold (float, default: 0.4): Threshold for flow error in Cellpose, controlling segmentation quality.
- run_deblur (bool, default: True): If True, apply deblur preprocessing before segmentation.
- run_upscale (bool, default: True): If True, apply upscaling preprocessing before segmentation.
- image_normalise (bool, default: True): If True, normalize image intensity before Cellpose.
- image_normalise_percentile_lower (float, default: 0.0), image_normalise_percentile_upper (float, default: 98.0): Percentiles for intensity normalization range.
- dpi_qc_images (int, default: 300): DPI for saved QC images, higher values produce higher resolution overlays.

*Parameter scanning fields*:
- run_parameter_scan (bool, default: False): If True, the pipeline runs a parameter scan rather than normal segmentation.
- param_a (Optional[str], default: 'cellprob_threshold'): The first parameter to vary (X-axis of parameter grid).
- param_a_values (List[Any], default: [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]): Values to test for param_a.
- param_b (Optional[str], default: 'flow_threshold'): The second parameter to vary (Y-axis of parameter grid).
- param_b_values (List[Any], default: [0.2, 0.3, 0.4, 0.5, 0.6]): Values to test for param_b.
- window_size (Optional[int], default: 250): Size of the windowed sub-region (patch) to show in QC grids for parameter scan. This only affects displayed QC images, not the segmentation or stats on the full image.
- num_rois_to_scan (int, default: 3): If no scan_rois specified, randomly choose this many ROIs for parameter scanning.
- scan_rois (Optional[List[str]]): If provided, run parameter scan on these specific ROIs.

### Segmentation Configuration (segmentation)

**Purpose:**
Controls downstream data processing after segmentation, such as storing results in AnnData or adjusting marker normalization.

**Parameters:**
- celltable_output (str, default: 'celltable.csv'): Name of the CSV file to store aggregated single-cell data from segmentation.
- marker_normalisation (List[str], default: ["q0.999"]): List of normalization methods for markers (e.g., quantile normalization).
- store_raw_marker_data (bool, default: False): If True, store raw marker intensities as well as normalized data.
- remove_channels_list (List[str], default: ['DNA1', 'DNA3']): Channels to remove before data analysis, often non-informative DNA stains.
- anndata_save_path (str, default: 'anndata.h5ad'): Path to store the final single-cell data as an AnnData file.

### Basic Process Configuration (basic_process)

**Purpose:**
Controls additional processing steps like batch correction and clustering (e.g., PCA, UMAP, Leiden clustering).

**Parameters:**
- input_adata_path (str, default: 'anndata.h5ad'): Input AnnData file path from previous steps.
- output_adata_path (str, default: 'anndata_processed.h5ad'): Output AnnData file after processing (e.g., batch correction).
- batch_correction_method (Optional[str]): Which batch correction method to use (e.g., 'bbknn', 'harmony').
- batch_correction_obs (Optional[str]): Key in .obs to use for batch correction grouping.
- n_for_pca (Optional[int]): Number of principal components to compute. If None, defaults set by pipeline.
- leiden_resolutions_list (List[float], default: [0.3, 1.0]): List of resolutions for Leiden clustering, generating multiple clusterings.
- umap_min_dist (float, default: 0.5): UMAP min_dist parameter controlling embedding compactness.
