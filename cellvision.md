I want to implement a new analysis option in our pipeline (lets called it CellVision), however it is fairly complicated. The overall aim is to use a package called scPortrait to get images of single cells, then use an VICReg image model to get the embeddings for specific cells (adapting code from HYPERSTAC), then cluster the embeddings (using RAPIDS). My overall hypothesis is that we can use this approach to differentiate populations which are hard to differentiate using traditional metrics such as mean cell intensity of our IMC markers, because we will be using image features.

# INITIAL INPUTS
The initial input will be an IMC project folder, including an AnnData object, masks folder, image folder. The user should be able to specify a population_obs, and specific populations within it, but it should also work with all the cells. Similarly, the user should be able to specify a list of markers, but it should also work with all the markers.

# scPortrait_to_IMC
## Conda environment on my HPC for this stage: scPortrait
I have previously created helper scripts for scPortrait to create a H5SC file from the raw inputs. This is essentially an image of each individual cell, with anything outside the cell boundary (given by the cells mask), blacked out. This serves as the input for training a model
The code I was using previously can be found here, but I think I was only previously doing 1 channel, though it should be adaptable to work with as many channels as we need
D:\Github_repos\scPortrait_to_IMC
I don't think I actually changed the original code, I think I just created some helpers for IMC. The original repo is : https://github.com/MannLabs/scPortrait
scPortrait_to_IMC was setup to use scportrait_config.yaml where we specify image_size - lets set the size to be 36x36, which should capture even large cells.
Output: scPortrait folder, including single_cells.h5sc

# Train and get embeddings of VICReg model
## Conda environment on my HPC for this stage: scPortrait
I then want to try and use a VICReg model to get the embeddings of each cell. An example of the type of architecture I want to start from can be found here:
D:\Github_repos\HyPERSTAC
This method works on 100x100 patches, and assumes the entire of the patch has data in, whereas we know our data is masked by cell boundaries. This may impact the types of image transformation we do. The current code is for tensorflow, but please recreate/adapt the architecture for Torch. It also orignally used images, whereas we are using the outputs (ie, H5SC) from scPortrait. You may want to utilise dataloaders from scPortrait.
The repo above as lots of downstream analyses, but we only need to adapt the VICReg embedding portion. This was the code I was using for that:
D:\Github_repos\HyPERSTAC\imc_hyperstac_pipeline.py
In the above code we also extracted patch-level features but we don't need to do that because we already have all the cell-level data in the source AnnData object from the input.
Output: Trained VICRed model, and cell-level embeddings.

# Cluster the VICReg emeddings using RAPIDS
## Conda environment on my HPC for this stage: rapids_singlecell
This stage will just be doing clustering - we should adapt/reuse our existing rapids script for this:
imcanalysis\SpatialBiologyToolkit\scripts\basic_process_rapids.py
We shouldn't need to do a full parameter scan, instead favouring a single values for n_pc and n_neighs, but support a list of leiden resolutions
Output: UMAP/leiden for our cells based upon VICReg embeddings, associated AnnData object saved.

# Plotting
## Conda environment on my HPC for this stage: scPortrait
For each leiden resolution, we should plot a UMAP to show the groups. We should also import the population labels from the original AnnData, and plot those. We should also plot confusion plots comparing how the new VICReg generated leiden labels compared with the original ones. Finally, we should plot the new VICReg label in the original AnnData UMAP embedding space (remembering we may not have all the cells represented in the VICReg dataset).
We should intelligently adapt the function "plot_random_cell_gallery" from this notebook: D:\Programming\2025_RobMorgan\Nuclear_morphology.ipynb. Create the galleries so that cell are rows, and different channels are columns. If we have 3 or less channels, also do a column with a combined image. We should use the images from the H5SC file (ie, the images the VICReg was trained on)

# REQUIREMENTS
- It is crucially important that we track single-cells specifically throughout the pipeline, so that we can track the individual cells at all stages.
- Separate the workflow into separate scripts and into separate SLURM scripts. However, please also create a combined SLURM script - the reason being, that I can use a single GPU slurm job for the entire pipeline, rather than re-queueing!

