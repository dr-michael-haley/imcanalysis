# Browser-based tools

Open and explore spatial-biology data in your web browser. These tools run on
your computer: you select local files, inspect them, and download figures or
review settings. You do not need to install SpatialBiologyToolkit, Python,
Node.js or a command-line environment to use the hosted applications.

The [spatialbiology.app resource hub](https://spatialbiology.app/) links to all
three explorers. These guides describe the controls and workflows for users
of those applications.

## Choose a tool

| Your data and task | Tool | Start here |
| --- | --- | --- |
| An IMC acquisition file; check regions, marker images and raw pixel counts before analysis | [MCD Explorer](https://mcd.spatialbiology.app/) | [MCD Explorer user guide](mcd_explorer.md) |
| An analysed `.h5ad` file with ROI images and cell masks; review populations, compare views and prepare figures | [IMC Explorer](https://imc.spatialbiology.app/) | [IMC Explorer user guide](imc_explorer.md) |
| A SpatialData `.zarr` folder or ZIP; explore images, cell boundaries, transcript points and gene expression | [MERSCOPE Explorer](https://merscope.spatialbiology.app/) | [MERSCOPE Explorer user guide](merscope_explorer.md) |

MCD Explorer does not require segmented cells or AnnData. IMC Explorer uses
separate AnnData, TIFF image and mask files. MERSCOPE Explorer expects a
**SpatialData store**, rather than an arbitrary folder of raw instrument
exports. None of the explorers runs the SBT processing pipeline or calculates
a new UMAP or t-SNE embedding.

## Before your first session

1. Use a current **desktop Chrome or Edge** browser. The interfaces are designed
   for a desktop display, and some folder-access features depend on the browser.
2. Open the application using its link above. There is no account to create.
3. Have the required files available on your computer or an accessible network
   drive. Make cloud-synchronised files available locally before opening them.
4. Follow the tool's first-session steps. Start with one region and a few
   channels or layers, then add more views once that display is working.

Choosing a folder gives the browser permission to read it. A browser prompt
may use the word “upload” for a file or folder picker; these applications read
the selected data locally and do not send the dataset to their host.

## What is saved?

Display controls do not change the source images, measurements or cell labels.
An image export is a new file, and a saved workspace stores review settings
rather than a replacement dataset.

| Tool | Across browser sessions | Portable output |
| --- | --- | --- |
| IMC Explorer | Named workspaces save locally; reconnect the original assets when required | Workspace JSON, figures, plot data and proposed annotation CSVs |
| MCD Explorer | Display settings and export bookmarks are limited to the current session | PNG, JPEG and layered SVG figures |
| MERSCOPE Explorer | Workspace settings save locally; reopen the same store to resume | Workspace JSON, composed figures and embedding PNGs |

Local saves belong to a particular website, browser profile and computer.
They do not automatically follow you to another device, an alternative app
address or a private-browsing window. Clearing site storage can remove them.
Export a workspace JSON before moving browsers or clearing storage, and keep
the original dataset: the JSON does not contain its images or expression data.
MCD Explorer has no equivalent portable workspace, so export the figures you
need before closing the tab.

For loading problems, missing images, memory limits and reporting an issue,
see [Troubleshooting](troubleshooting.md).

```{toctree}
:maxdepth: 2
:hidden:

imc_explorer
mcd_explorer
merscope_explorer
troubleshooting
```
