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
- Clone/download the repo locally (see the <span style="color:green">**<>Code**</span> button).
- Create an environment using *conda_environment.yml*, which should install all the required packages.
> conda env create -f conda_environment.yml
- Navigate to where you have downloaded the repo and install the package using:
> pip install . 
- If you haven't already, you can also install Jupyter (to view and run notebooks, in the *Tutorials* folder) using:
> conda install jupyter
- You can then start Jupyter from the Anaconda prompt using:
> jupyter lab

### Tutorials
Example tutorials can be found in the Tutorials folder. They should all work in an environment where SpatialBiologyToolkit package has been installed.

## IMC_Denoise
This is a Jupyter notebook that implements the IMC Denoise approach (https://github.com/PENGLU-WashU/IMC_Denoise/) to marry up with the Bodenmiller IMC pipeline.

> [!CAUTION]  
> This code requires updating, as it currently uses an old version of IMC_Denoise, which has since been updated.

## REDSEA
This has been adapted from the code originally written by Artem Sokolov (https://github.com/labsyspharm/redseapy) to help better integration into the Bodenmiller pipeline

> [!CAUTION]  
> This code has not been thoroughly tested or maintained in some time (last updated 4th Nov 2022).
