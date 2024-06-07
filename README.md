<table>
  <tr>
    <td style="vertical-align: top; padding-right: 20px;">
      <img src="Other/logo.png" alt="Logo" width="150" height="150" style="margin-right: 10px;"/>
    </td>
    <td style="vertical-align: top;">
      <h1>Spatial Biology Toolkit</h1>
      <p>
        This is a collection of tools for analysing high-dimensional tissue data. It includes tools for analysing IMC data in the Scanpy ecosystem, but also several image-based analysis tools.
        Currently, most of the tools are designed to work with IMC data, but most should be adaptable to other modalities.
      </p>
    </td>
  </tr>
</table>

## SpatialBiologyToolkit package

All the tools are in the SpatialBiologyToolkit package

Documentation can be viewed if download the repository locally, and by opening *'index.html' in Documentation*

### Installing
- Clone/download repo locally
- Create an environment using conda_environment.yml, which should install all the required packages. 
- Install the package using:
> pip install .
- You can also install Jupyter (to view notebooks) using:
> conda install jupyter

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
