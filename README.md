<table>
  <tr>
    <td style="padding-right: 20px;">
      <img src="Other/logo.png" alt="Logo" width="350" />
    </td>
    <td>
      <h1>Spatial Biology Toolkit</h1>
      <p>
        This is a collection of tools for analysing high-dimensional tissue data. It includes tools for analysing IMC data in the Scanpy ecosystem, but also several image-based analysis tools.
        Currently, most of the tools are designed to work with IMC data, but most should be adaptable to other modalities.
      </p>
    </td>
  </tr>
</table>


## SpatialBiologyToolkit package
All the tools I have developed are available in this package, details for installing are below. Documentation can be viewed if download the repository locally, and by opening *'index.html' in Documentation*

### Installing
- Install Anaconda or any other distribution of Python.
- Clone/download the repo locally (see the <font color="green">**<>Code**</font> button).
- Create an environment using *conda_environment.yml*, which should install all the required packages.
> conda env create -f conda_environment.yml
- Navigate to where you have downloaded the repo and install the package using:
> pip install -e .
- This will install the package in developer mode. If you would prefer a regular install, omit the *-e*
- If you haven't already, you can also install Jupyter (to view and run notebooks, in the *Tutorials* folder) using:
> conda install jupyter
- You can then start Jupyter from the Anaconda prompt using:
> jupyter lab

### Tutorials
Example tutorials can be found in the Tutorials folder. They should all work in an environment where SpatialBiologyToolkit package has been installed.

# Preprocessing scripts for IMC data on University of Manchester CSF3 :test_tube: :electric_plug: :bee:
I have added scripts for running preprocessing of IMC data on CSF3, or similar command line cloud computing platform: [Installation and usage guide](#CSF3)

## IMC_Denoise - UPDATED Nov 2024
~This is a Jupyter notebook that implements the IMC Denoise approach (https://github.com/PENGLU-WashU/IMC_Denoise/) to marry up with the Bodenmiller IMC pipeline~

This approach to denoising has been superceded by running denoising on the CSF3

## REDSEA
This has been adapted from the code originally written by Artem Sokolov (https://github.com/labsyspharm/redseapy) to help better integration into the Bodenmiller pipeline

> [!CAUTION]  
> This code has not been thoroughly tested or maintained in some time (last updated 4th Nov 2022).
