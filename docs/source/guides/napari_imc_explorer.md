# `napari_imc_explorer` guide

> This legacy explorer remains available for existing analyses. New integrated
> exploration and classification experiments should use
> [`napari_sbt`](napari_sbt.md); see the
> [migration guide](napari_sbt_migration.md).

`napari_imc_explorer` is an interactive Napari-based viewer for exploring IMC data at ROI level. It combines raw marker images, segmentation masks, cell-level metadata from `AnnData`, optional extra whole-ROI images, manual region annotation tools, and a population-focused QC view in one interface.

This guide is intended to explain what the explorer can do and how its parts fit together. It is not a click-by-click tutorial.

## What you point it to when you open it

The explorer is started by calling `napari_imc_explorer(...)`. The main inputs are:

- `adata`: the cell-level `AnnData` object. This is the table the explorer uses for per-cell metadata, categorical labels, and numeric values.
- `masks_folder`: a folder containing one segmentation mask per ROI. Each file is expected to be named after the ROI, for example `ROI_001.tiff`.
- `image_folders`: one or more folders containing ROI subfolders. Inside each ROI subfolder, the channel images for that ROI are stored as TIFF files.
- `extra_images`: optional folders that contain one image per ROI directly in the folder, such as H&E snapshots, thumbnails, or other reference images.
- `normalization_dict`: an optional Nimbus-format channel-to-maximum mapping. The
  preferred Nimbus CSV can be loaded with `load_normalization_mapping`; legacy
  JSON remains readable. The explorer uses the Vmax column only; set its existing
  minimum-pixel threshold separately if background suppression is required.
- `annotations_folder`: where manual annotation sets are stored and where new annotation files are written.
- `roi_obs`: the column in `adata.obs` that tells the explorer which ROI each cell belongs to.
- `cell_id_in_mask_obs`: the column in `adata.obs` that stores the object IDs used inside the mask image.

A minimal example looks like this:

```python
from SpatialBiologyToolkit._napari_imc_normalization import load_normalization_mapping
from SpatialBiologyToolkit.napari_imc_explorer import napari_imc_explorer

normalization_dict = load_normalization_mapping("Nimbus/normalization_dict.csv")

viewer, handles = napari_imc_explorer(
    adata=adata,
    masks_folder="Masks",
    image_folders=["Images"],
    extra_images=["Extra_images"],   # optional
    normalization_dict=normalization_dict,  # optional
    annotations_folder="Annotations",
    roi_obs="ROI",
    cell_id_in_mask_obs="ObjectNumber",
)
```

## How the explorer matches your files together

The explorer relies on consistent naming across your data sources.

- ROI names are matched between `adata.obs[roi_obs]`, mask filenames, image subfolder names, and extra-image filenames.
- Segmentation masks are expected to contain integer object IDs for cells.
- Those object IDs are matched to `adata.obs[cell_id_in_mask_obs]`. If that column is missing, the explorer falls back to row order/index-based matching.
- If `mask_extension` is not provided, the explorer tries to infer it from the first file in `masks_folder`.
- If `check_masks=True`, the explorer checks on startup that the mask IDs for each ROI match the cells listed in `adata`.

In practice, the most important requirement is simple: the same ROI names and the same cell IDs need to be used everywhere.

## How image discovery works

- The ROI selector is populated from the ROI subfolders found in the first entry of `image_folders`.
- Raw IMC images are expected inside ROI-specific subfolders.
- The raw-image selector tries to map TIFF filenames back to sensible channel names using `adata.var_names`, and, if present, `adata.var["channel_name"]` and `adata.var["channel_label"]`.
- Extra image folders are treated differently: each folder is a source, and each source should contain one image per ROI directly in the folder.
- Extra images support common image formats such as `png`, `jpg`, `tif`, `tiff`, `bmp`, `gif`, and `webp`.

This means the explorer can mix marker images with other ROI-level reference images, as long as the ROI naming is consistent.

## How the interface is organised

When the explorer starts, it creates the Napari viewer plus a small launcher panel. That launcher is the entry point to the rest of the UI.

- `Open all panels` opens every tool panel at once.
- `Open Controls`, `Open Population QC`, and the other launcher buttons open the corresponding floating panels.
- Most tools operate on the ROI currently selected in the `Controls` panel.

The main panels are:

- `Controls`
- `Population QC`
- `Add raw images`
- `Add extra images`
- `Categories as masks`
- `Numeric as masks`
- `Manual annotations`
- `Layer management`

## What each panel does

### `Controls`

This is the central panel for choosing what part of the dataset you are looking at.

- Selects the current ROI.
- Adds all raw images for the selected ROI.
- Adds the base cell mask for the selected ROI.
- Hides all layers.
- Deletes all layers.
- Moves currently selected layers to the top of the Napari layer stack.

The `Add ALL images for ROI` button loads every discovered marker image for the current ROI. These layers are added hidden by default, so the panel is useful when you want the full set available in the layer list without displaying everything at once.

The `Add cell mask` button adds the full segmentation mask as a Napari labels layer called `all_cells`.

### `Add raw images`

This panel is for loading a chosen subset of marker images rather than every channel at once.

- Lets you choose individual channels from the available logical image names.
- Applies a minimum pixel threshold before display.
- If a channel is present in `normalization_dict`, divides it by that Nimbus maximum and clips the result to the range 0 to 1.
- If no dictionary is supplied, or a channel is absent from it, normalises that image to the chosen quantile as before.
- When a non-empty normalization dictionary is supplied, shows lower and upper Nimbus display-contrast sliders from 0 to 1. Their values set the Napari contrast limits for subsequently added raw marker layers while leaving their normalized data unchanged. The controls remain ordered so the lower bound stays below the upper bound.
- Loads images as false-colour additive layers in Napari.

This is usually the main panel for looking at selected markers on top of masks or other overlays.

### `Add extra images`

This panel is for optional non-marker ROI images.

- Each entry corresponds to one folder supplied through `extra_images`.
- For the currently selected ROI, it loads the matching image from the chosen source.
- RGB images are detected and displayed as RGB; grayscale images are loaded as standard image layers.

Typical uses are pathology overviews, H&E images, stitched brightfield views, or any other one-image-per-ROI reference.

### `Categories as masks`

This panel turns categorical `adata.obs` columns into segmentation-based label overlays.

- The selector lists categorical columns from `adata.obs`.
- Each selected category can be rendered as a combined labels layer over the segmentation mask.
- Optionally, the panel can also create separate mask layers for each individual group within that category.
- If `adata.uns["<obs>_colors"]` exists, those colours are used; otherwise a default categorical palette is used.

This is useful for showing cluster labels, cell types, experimental groups, neighbourhood labels, or any other per-cell category already stored in `adata.obs`.

### `Numeric as masks`

Despite the panel name, this feature produces numeric overlays rather than label masks.

- The selector includes both numeric `adata.obs` columns and marker names from `adata.var_names`.
- For each selected value, the explorer fills every segmented cell with its corresponding numeric value.
- The result is added as an image layer, so it behaves like a quantitative heatmap over the segmentation.

This is useful for showing per-cell scores, metadata, or expression of a chosen marker across a single ROI.

### `Population QC`

This panel is a focused view for quickly inspecting one cell population across ROIs.

- You choose a categorical `adata.obs` field, such as a cell-type or cluster column.
- You then choose one population within that field.
- The panel lets you assign Red, Green, and Blue marker channels for that population.
- Marker settings are stored in a CSV file named `backgating_settings_<obs>.csv` in the current working directory.
- If no saved settings exist yet, the explorer proposes unsaved defaults based on mean expression for that population.
- The panel also shows shortcut ROI buttons, typically prioritised by how abundant that population is in each ROI.

When you click `Load population view`, the explorer clears the current viewer and loads:

- only the chosen RGB marker images for the selected ROI
- a mask for the selected population

This makes it easier to do fast visual QC or “backgating”-style inspection of a specific population without manually assembling the view each time.

### `Manual annotations`

This panel provides ROI-level manual region annotation directly inside Napari.

Its main roles are:

- managing named annotation sets
- storing label-to-colour mappings
- loading the annotation layer for the currently selected ROI
- saving edited annotations back to disk
- syncing region labels back into `adata.obs`

Each annotation set is stored as:

- a subfolder inside `annotations_folder`
- a `label_mapping.csv` file describing the annotation classes
- one TIFF label image per ROI

The panel supports three main workflows:

- **Create new annotation set**: defines the allowed annotation classes and creates blank label TIFFs for all ROIs.
- **Load / save selected ROI**: loads the annotation TIFF for the ROI selected in `Controls` as an editable Napari labels layer, then saves it back after painting/editing.
- **Sync to AnnData**: converts the annotation layer into a per-cell categorical label by asking which annotation value is dominant inside each cell mask. The result is written into a column in `adata.obs`, and matching colours are stored in `adata.uns`.

This is useful when you want to annotate tissue regions, tumour areas, structural zones, or other spatial compartments and then bring those annotations back into the analysis table.

### `Layer management`

This panel contains general Napari layer utilities that are useful when combining many views.

It can:

- recolour image layers
- recolour all labels in a labels layer, or a specific label value
- flip selected layers horizontally or vertically
- resize selected layers to match another layer
- copy a colormap from one layer to others
- expand labels by a chosen number of pixels and create a new expanded labels layer
- create a new masked labels layer based on selected layers
- save the current visible workspace
- load a previously saved workspace

The workspace save/load tools store:

- all currently visible layers as pickled layer files
- the Napari camera settings in `camera_settings.csv`

Only visible layers are saved, which makes this feature useful for preserving a curated view rather than the entire session.

## Files the explorer can create

Depending on which tools you use, the explorer can write several kinds of outputs:

- annotation TIFFs and `label_mapping.csv` files inside `annotations_folder`
- `backgating_settings_<obs>.csv` files for Population QC in the current working directory
- workspace folders containing layer `.pickle` files plus `camera_settings.csv`

So the explorer is not only a viewer; it can also produce analysis-ready annotation outputs and reusable display presets.

## Practical expectations and assumptions

- The explorer is ROI-centred. Many actions depend on the ROI selected in `Controls`.
- The segmentation mask is the bridge between image space and cell-level metadata.
- Categorical and numeric overlays only work well when mask IDs and `adata.obs` rows are correctly aligned.
- The viewer is most useful when `adata.var_names` and image filenames use a consistent channel naming scheme.
- Population QC is designed for quick focused inspection, so loading that view intentionally replaces the current layer stack.

## In one sentence

`napari_imc_explorer` is a single place to load ROI images, segmentation masks, metadata-driven overlays, extra reference images, manual region annotations, and population-specific QC views for IMC data.
