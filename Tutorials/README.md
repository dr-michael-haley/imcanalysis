> [!WARNING]  
> This is all actively being updated.

# Getting started
If you are new to IMC analysis and have data to work with that has gone through pre-processing, or are interested in the analysis pipeline, started with **NEW - Population identification.ipynb**


# Old Notebooks
These are various notebooks that were previously the 'pipeline'. Some are still goood to use.

### Status
🔴 - Probably won't work, not updated in a long time
🟡 - Should work, but fairly out of date
🟢 - Recent code that should work

|**Status**| **Notebook**              | **Description** |
|----------|---------------------------|-----------------|
|🔴| **1A. ImcSegmentationPipeline Data Import** | Creating an AnnData object using the now rarely use ImcSegmentationPipeline from Bodenmiller.|
|🟡| **1B. Steinpose and Steinbock Data Import** | Creating an AnnData object using Steinpose/steinbock output files. |
|🟡| **2. Population identification** | Population identification (old). |
|🟡| **3. Neighborhoods.ipynb** | A now superceded way of calculating cell neighborhoods, similar to how originally done by Stanford MIBI-TOF groups |
|🟢| **3b. QUICHE neighbourhoods** | "Neighbourhoods" identified using [QUICHE](https://www.biorxiv.org/content/10.1101/2025.01.06.631548v1.full) |
|🟢| **4. Plotting** | Plotting and statistics |
|🟡| **5. Spatial analyses** | Spatial statistics using  the [SpOOx](https://pubmed.ncbi.nlm.nih.gov/37940670/). [Specific documentation](https://github.com/Taylor-CCB-Group/SpOOx/tree/main/src/spatialstats). |
|🟢| **6. Napari explorer** | Interactively explore an AnnData object and images in a graphical user interface. |
|🟢| **7. Subregion analysis** | Sample code for creating and analysing subregions |
|🟢| **8. Backgating and plotting populations** | Most of this code has been moved into thew population ID notebook |