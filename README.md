# Contents
## Mike_IMC_Pipeline - last updated Jan 2024
This is the latest iteration of my Python pipeline for IMC analysis, which includes some basic details on how to get setup the Conda environment and get started. Start at notebook 1 and work through them. This is a work in progress, and should be treated as such.

## Mike_IMC_Denoise.ipynb
This is a Jupyter notebook that implements the IMC Denoise approach (https://github.com/PENGLU-WashU/IMC_Denoise/, https://www.biorxiv.org/content/10.1101/2022.07.21.501021v1) to marry up with the Bodenmiller IMC pipeline.

Running it will require  a computer with a good GPU / graphics card, and will also require it to be compatible with the packages that IMCDenoise uses (e.g. specific version of TensorFlow, Keras and Python, all of which are a bit out of date in the IMC Denoise repository on which I've based everything).

Underdoing testing as we speak!

## REDSEA.ipynb - last updated 4th Nov 2022

This notebook integrates the REDSEA algorithm (https://www.frontiersin.org/articles/10.3389/fimmu.2021.652631/full) converted into Python by Artem Sokolov (https://github.com/labsyspharm/redseapy) into the Bodenmiller pipeline.

This replaces the last CellProfiler step in the Bodenmiller pipeline that would normally extract the single cell expression of each of the markers and create a cell table .csv file. This notebook similarly extracts the single-cell information, but also does the REDSEA compensation for any cells in which the cells are in direct cell-to-cell contact. This is particularly useful for very dense tissues where segmentation is difficult.

This should run in the same environment as you use for other IMC analysis.
