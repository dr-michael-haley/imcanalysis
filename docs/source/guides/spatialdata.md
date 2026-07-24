# Convert an IMC project to SpatialData

`SpatialBiologyToolkit.spatialdata` converts the toolkit's established IMC
layout into a scverse SpatialData object without loading the full image
collection into memory.

## Source layout

The converter expects:

- an AnnData object whose `var_names` are marker names;
- `adata.obs["ROI"]`, identifying the image for each cell;
- `adata.obs["ObjectNumber"]`, containing that cell's positive integer label
  in its ROI mask;
- one integer mask at `masks/{ROI}.tiff`;
- optionally, one image folder per ROI at `images/{ROI}/`.

Images are matched to each `var_name` by exact, case-insensitive filename stem
first. If no exact name exists, a unique bounded substring is accepted, so
`CD3` matches `152Sm_CD3.tiff` without matching `CD31.tiff`. Any ambiguity is
an error.

## Create and write the object

```python
import anndata as ad

from SpatialBiologyToolkit.spatialdata import (
    create_spatialdata,
    plan_imc_spatialdata_conversion,
    write_spatialdata,
)

adata = ad.read_h5ad("anndata.h5ad", backed="r")

# Optional preflight: resolves every file and validates table-to-mask labels.
plan = plan_imc_spatialdata_conversion(
    adata,
    images_folder="images",
    masks_folder="masks",
)
print(plan.n_rois, plan.n_image_files)

sdata = create_spatialdata(
    adata,
    images_folder="images",
    masks_folder="masks",
    roi_key="ROI",
    instance_key="ObjectNumber",
)
write_spatialdata(sdata, "spatialdata.zarr")
adata.file.close()
```

The function creates one image and one labels element per ROI. Both use an
identity transformation into an ROI-local coordinate system. A single AnnData
table annotates every labels element through two added observation columns:
`_sbt_region` and `_sbt_instance_id`. The original `ROI` and `ObjectNumber`
columns are not changed.

Extra labels in a mask are retained and recorded as unannotated instances;
this supports cells removed during filtering. A table cell whose
`ObjectNumber` is absent from its mask fails validation.

Images and masks are Dask-backed. Construction and summaries are lazy, while
writing the Zarr store materializes the rasters. Zarr v2 is the default for
broad interoperability and compatibility with mixed AnnData/Zarr dependency
sets; use `zarr_format=3` only with a dependency set known to support current
SpatialData Zarr v3 writes.

## Add named label layers

Additional ROI-aligned integer rasters can be added concurrently. Each logical
layer has its own folder, exact filename suffix, and pixel-value-to-name map:

```python
import pandas as pd

from SpatialBiologyToolkit.spatialdata import AdditionalLabelsSpec

zone_names = pd.DataFrame(
    {
        "ROI": ["ROI 1", "ROI 1", "ROI 2", "ROI 2"],
        "label_value": [1, 2, 1, 2],
        "label_name": ["Core", "Margin", "Core", "Stroma"],
    }
)

additional_labels = [
    AdditionalLabelsSpec(
        name="tissue_region",
        folder="TissueRegions",
        suffix="_regions",
        value_names={1: "Tumour", 2: "Stroma", 3: "Necrosis"},
    ),
    AdditionalLabelsSpec(
        name="analysis_zone",
        folder="AnalysisZones",
        suffix="_zones",
        value_names=zone_names,
    ),
]

sdata = create_spatialdata(
    adata,
    images_folder="images",
    masks_folder="masks",
    additional_labels=additional_labels,
)
```

For example, the first layer expects
`TissueRegions/{ROI}_regions.tif[f]`. Matching is exact and
case-insensitive; fuzzy filename matching is intentionally not used.

The TIFFs remain integer SpatialData Labels elements. Human-readable names are
stored in one linked annotation table per logical layer. A global
`{integer: name}` mapping is expanded across all ROIs; a DataFrame or CSV can
contain the source ROI column when values have ROI-specific meanings. Nested
`{ROI: {integer: name}}` mappings are also accepted. Zero is background and
is not added to the annotation table.

Generated elements and tables are named predictably:

```text
labels_tissue_region_ROI_1
tissue_region_annotations
labels_analysis_zone_ROI_1
analysis_zone_annotations
```

Use the toolkit helpers to resolve layers and names without manually joining
pixel codes:

```python
from SpatialBiologyToolkit.spatialdata import (
    get_label_annotations,
    get_roi_label_elements,
    plot_spatialdata_roi,
)

get_roi_label_elements(sdata, "ROI 1")
get_label_annotations(sdata, "tissue_region", roi="ROI 1")

plot_spatialdata_roi(
    sdata,
    roi="ROI 1",
    channel="CD3",
    label_layer="tissue_region",
)
```

For additional layers, `plot_spatialdata_roi()` colours by the semantic
`label_name` column by default. The primary cell segmentation remains the
default `label_layer="cells"` for backwards compatibility.

Preflight validation requires one layer TIFF per AnnData ROI, 2D integer data,
the same shape as the primary ROI mask, no negative values, and a name for
every positive pixel value.

## Inspect and plot

```python
from spatialdata import read_zarr

from SpatialBiologyToolkit.spatialdata import (
    plot_population_counts,
    plot_spatialdata_cells,
    plot_spatialdata_roi,
    summarize_spatialdata,
)

sdata = read_zarr("spatialdata.zarr")
summary = summarize_spatialdata(
    sdata,
    population_key="leiden_1.0",
    case_key="animal",
)

plot_population_counts(sdata, "leiden_1.0")
plot_spatialdata_roi(
    sdata,
    roi="example_roi",
    channel="CD3",
    color="leiden_1.0",
)

# Build a fixed-size gallery from unique AnnData observation names. Cells can
# come from different ROIs; each panel shows the target cell in local context.
selected_cells = sdata.tables["table"].obs_names[:8]
figure, axes = plot_spatialdata_cells(
    sdata,
    selected_cells,
    channel=["CD3", "CD4", "CD8a"],
    color="leiden_1.0",
    crop_size=64,
    ncols=4,
)

# ObjectNumber is only unique within an ROI, so supply the ROI when using it.
figure, axes = plot_spatialdata_cells(
    sdata,
    cells=[12, 35],
    cell_key="ObjectNumber",
    roi="example_roi",
    channel="CD3",
    outline_target_only=True,
    mask_outside_target=True,
    fill_alpha=0.0,  # Optional: retain the outline without a coloured fill.
    show_ax_titles=False,
)
```

`summarize_spatialdata()` does not read raster pixels. The ROI plot loads only
the selected marker image, its mask, and the relevant table annotations. The
cell gallery reads each required ROI mask and marker plane once, then extracts
fixed-size crops centred on the selected mask instances. By default, `cells`
contains unique AnnData observation names. When `cell_key` is provided, each
requested value must identify exactly one table row after applying the optional
`roi` restriction; ambiguous matches raise an error rather than silently
plotting the wrong cell. Set `outline_target_only=True` to omit neighbouring
cell boundaries, and `mask_outside_target=True` to replace everything outside
each target mask with black. The two display options are independent and both
default to `False` for backward compatibility. Set `show_ax_titles=False` to
hide the per-cell observation, ROI, instance, and annotation titles while
retaining an optional figure-level `title=`.

## Limitations

- The converter currently supports 2D, single-plane TIFF images and masks.
- Each ROI has its own pixel coordinate system; it does not infer a tissue
  mosaic or physical pixel scale.
- Marker names must be unique in `adata.var_names`.
- A single plot can combine at most three marker channels.
