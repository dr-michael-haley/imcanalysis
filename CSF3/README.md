# Preprocessing scripts for CSF3

## <font color='green'>Overview</font>
These scripts were designed to do all the pre-processing for IMC data on the CSF3. The scripts are designed to be run in sequence via a single job file. All the scripts use a common YAML configuration file to specify  directories, pipeline behaviors, segmentation parameters, etc.

### Pipeline scripts
- `preprocesing.py`: Extracts the .TIFF images from the MCD files
- `denoising.py`: Uses IMC Denoise to denoise each channel
- `createmasks.py`: Uses CellPose3 to create segmentation makes based upon the DNA channel, including creating QC images. Can also do a parameter scan to fine tune the CellPose3 parameters.
- `segmentation.py`: Uses the masks to segment all the denoised images, create cell tables with the raw data, which is then use to create an AnnData object.
- `basic_process.py`: Performs basic pre-processing of the data, including batch correcting, calculating UMAPs, and initial leiden clustering.

### Accessory scripts
- `generate_config.py`: Creates a config.yaml file with the default parameters that can then be modified

## Installation
1. 
