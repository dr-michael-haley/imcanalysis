# Build multimodal SpatialData objects

`SpatialBiologyToolkit.spatialdata` provides declarative source adapters for
building and extending SpatialData objects. The adapters validate file
layouts, ROI relationships, channel panels, segmentation instances,
coordinates, and linked tables before creating standard SpatialData model
elements.

The construction API is intentionally split into three operations:

```python
plan = plan_spatialdata(spec)   # inspect and validate; no source mutation
sdata = create_spatialdata(spec, plan=plan)
updated = add_modality(sdata, new_modality)
```

`create_spatialdata()` accepts only a `SpatialDataSpec` or a validated
`SpatialDataPlan`. It does not have a separate legacy positional
AnnData/images/masks call.

## Modality specifications

The public specifications are source adapters rather than subclasses of
SpatialData or AnnData:

- `CellMasks`: integer cell-instance masks, one raster per ROI;
- `IMCImages`: one folder of single-channel marker images per ROI;
- `IMCAnnData`: a quantified cell table linked explicitly to one `IMCImages`
  panel and one `CellMasks` modality;
- `HistologyImages`: ROI-aligned RGB or RGBA TIFF, PNG, or JPEG images;
- `RegionLabels`: categorical integer rasters with semantic names in a linked
  annotation table;
- `MaxFuseSCRNASeq`: transcriptomes matched to a subset of cells in one
  `IMCAnnData` table.

Every modality has a stable `name`. IMC sources additionally have a
human-readable `panel_name`. Relationships refer to stable modality names, not
to generated SpatialData element names.

## Complete example

```python
import anndata as ad

from SpatialBiologyToolkit.spatialdata import (
    CellMasks,
    HistologyImages,
    IMCAnnData,
    IMCImages,
    MaxFuseSCRNASeq,
    RegionLabels,
    SpatialDataSpec,
    create_spatialdata,
    plan_spatialdata,
    write_spatialdata,
)

cells = ad.read_h5ad("immune_panel_cells.h5ad")
maxfuse = ad.read_h5ad("maxfuse_matched_transcriptomes.h5ad")

spec = SpatialDataSpec(
    modalities=[
        CellMasks(
            name="cells",
            folder="masks",
        ),
        IMCImages(
            name="immune_images",
            panel_name="Immune panel",
            folder="Images",
        ),
        IMCAnnData(
            name="immune_cells",
            panel_name="Immune panel",
            adata=cells,
            images="immune_images",
            masks="cells",
            roi_key="ROI",
            instance_key="ObjectNumber",
            x_key="X_loc",
            y_key="Y_loc",
        ),
        # This aligned panel has no cell-quantification table.
        IMCImages(
            name="extracellular_images",
            panel_name="Extracellular panel",
            folder="ExtracellularImages",
            channels=["Collagen", "Fibronectin", "Laminin"],
            reference="cells",
        ),
        HistologyImages(
            name="he",
            folder="Histology",
            reference="cells",
        ),
        RegionLabels(
            name="tissue_regions",
            folder="TissueRegions",
            suffix="_regions",
            value_names={
                1: "Tumour",
                2: "Stroma",
                3: "Necrosis",
            },
            reference="cells",
        ),
        MaxFuseSCRNASeq(
            name="atlas",
            adata=maxfuse,
            imc_table="immune_cells",
        ),
    ],
    raster_chunks=(512, 512),
)

plan = plan_spatialdata(spec)
display(plan.summary())
display(plan.report.to_frame())
plan.raise_for_errors()

sdata = create_spatialdata(spec, plan=plan)
write_spatialdata(sdata, "spatialdata.zarr")
```

Construction copies source AnnData objects before adding SpatialData table
metadata. It therefore does not add toolkit columns to the supplied objects.
Images and labels are Dask-backed and remain lazy until accessed or written.

For a very large AnnData already opened in memory or backed mode, copying may
be undesirable. Set `copy_adata=False` on `IMCAnnData` or
`MaxFuseSCRNASeq` to reuse that object. Construction will then add the formal
SpatialData region/instance columns and table metadata to the supplied
AnnData; keep a backed source open until the resulting SpatialData has been
written.

## Planning and integrity checks

`plan_spatialdata()` is the side-effect-free preflight API. It resolves all
source files and reports structured `info`, `warning`, and `error`
diagnostics.

```python
plan = plan_spatialdata(spec)

plan.ok
plan.report.errors
plan.report.warnings
plan.report.to_frame()
plan.summary()
```

The planner checks:

- unique modality, element, point, and table names;
- required AnnData columns and unique observation/variable names;
- finite coordinates and positive ROI-local instance IDs;
- unique instance IDs within each ROI;
- one unambiguous file or directory per requested ROI;
- one unambiguous image per channel;
- exact agreement between quantified AnnData variables and image channels;
- two-dimensional integer masks and region labels;
- every table instance exists in the referenced mask;
- region label names cover every positive raster value;
- identity-aligned raster shapes agree;
- MaxFuse observation names are a subset of the linked IMC table.

Mask instances absent from all linked cell tables are permitted and reported
as unannotated instances. This supports cells removed during filtering.

For stricter coordinate QC, set:

```python
IMCAnnData(
    ...,
    check_centroids_in_mask=True,
)
```

The rounded `X_loc`, `Y_loc` coordinate must then fall inside the expected
mask instance.

## IMC image matching

An `IMCImages` folder normally has this layout:

```text
Images/
  ROI 1/
    141Pr_CD3.tiff
    145Nd_CD31.tiff
  ROI 2/
    141Pr_CD3.tiff
    145Nd_CD31.tiff
```

Matching prefers an exact case-insensitive filename stem. If no exact stem is
found, `match_mode="exact_or_unique_substring"` accepts one bounded marker
match. Thus `CD3` can match `141Pr_CD3.tiff` without matching `CD31.tiff`.
More than one candidate is an error.

For strictly standardized data:

```python
IMCImages(
    ...,
    match_mode="exact",
    allow_extra_files=False,
)
```

A panel linked to `IMCAnnData` inherits the AnnData variable order. Explicit
`channels=` must contain exactly the same feature set. A standalone panel must
provide `channels=` itself.

Aligned modalities do not have to cover every reference ROI. For a standalone
IMC panel, histology collection, or region-label layer with incomplete
coverage, set `allow_partial=True` and leave `rois=None`:

```python
IMCImages(
    name="matrix_images",
    panel_name="Matrix panel",
    folder="matrix_images_aligned",
    channels=["CollagenIV", "Fibronectin", "TNC"],
    reference="cells",
    allow_partial=True,
)
```

Planning discovers the matching reference ROIs and emits one
`partial_roi_coverage` warning with included and missing counts. Explicit
`rois=` may also select a reference subset. Unknown ROI names, ambiguous
files, and missing files within an explicit selection remain errors. An image
panel quantified by `IMCAnnData` must always cover every ROI in that table.

## Coordinate systems and aligned modalities

`CellMasks` creates an independent ROI-local coordinate system per ROI.
Linked IMC images, histology, and region labels reuse those systems.

Identity alignment is used only when raster shapes agree. Different
resolutions or orientations require explicit SpatialData transformations:

```python
from spatialdata.transformations import Scale

HistologyImages(
    name="he",
    folder="Histology",
    reference="cells",
    transformations={
        "ROI 1": Scale([0.5, 0.5], axes=("y", "x")),
        "ROI 2": Scale([0.5, 0.5], axes=("y", "x")),
    },
)
```

The transformation maps source raster coordinates into the referenced
modality's coordinate system.

## Histology

Histology files are matched exactly and case-insensitively as
`{ROI}{suffix}{extension}`. TIFF, PNG, JPG, and JPEG are enabled by default.
If two enabled extensions have the same ROI stem, planning fails rather than
selecting one silently.

RGB and RGBA images receive channel coordinates `r`, `g`, `b`, and optionally
`a`. Set `drop_alpha=True` to discard an alpha channel.

## Named regions

SpatialData Labels elements remain integer rasters. Region names are stored in
a linked table because raster pixels cannot directly contain categorical
strings.

`value_names` can be:

- a global `{integer: name}` mapping;
- `{ROI: {integer: name}}` for ROI-specific meanings;
- a DataFrame;
- a CSV path.

DataFrames and CSVs use `label_value` and `label_name` by default and may
contain an `ROI` column. Alternative column names can be configured with
`value_key`, `name_key`, and `mapping_roi_key`.

```python
annotations = get_label_annotations(
    sdata,
    "tissue_regions",
    roi="ROI 1",
)
```

## MaxFuse transcriptomes

`MaxFuseSCRNASeq` stores a second AnnData table annotating the same cell masks
as its linked IMC table.

The transcriptomic AnnData must:

- have unique observation and gene names;
- use matched IMC observation names as its index;
- contain any subset of the linked IMC index.

Only matched rows are stored. An unmatched IMC cell is absent from the MaxFuse
table rather than represented as a row of expression zeros. Atlas identifiers,
match scores, populations, and other provenance in `obs` are preserved.

The builder copies each matched cell's formal SpatialData region and instance
links from the IMC table. Toolkit metadata records the linked IMC modality,
linked table, and matched fraction.

## Add modalities in memory

`add_modality()` stages and validates a complete candidate SpatialData before
changing the supplied object.

```python
from SpatialBiologyToolkit.spatialdata import add_modality

updated = add_modality(
    sdata,
    HistologyImages(
        name="he",
        folder="Histology",
        reference="cells",
    ),
)
```

The default returns a new SpatialData container while sharing the existing
lazy raster elements:

```python
assert "image_he_ROI_1" not in sdata.images
assert "image_he_ROI_1" in updated.images
```

Use `inplace=True` for an interactive notebook workflow:

```python
add_modality(sdata, new_modality, inplace=True)
```

Several related modalities can be planned and added together:

```python
addition = SpatialDataSpec([new_masks, new_images, new_table])
plan = plan_spatialdata(addition, existing=sdata)
plan.raise_for_errors()
updated = add_modality(sdata, plan)
```

Disk persistence is separate. After inspecting the updated object, write it to
a new Zarr store. This avoids presenting a sequence of per-element writes as a
transactional on-disk update.

## Interrogate and plot

```python
from SpatialBiologyToolkit.spatialdata import (
    get_roi_elements,
    get_roi_modalities,
    plot_spatialdata_cells,
    plot_spatialdata_roi,
    summarize_spatialdata,
)

get_roi_modalities(sdata, "ROI 1")

get_roi_elements(
    sdata,
    "ROI 1",
    image_modality="extracellular_images",
)

summarize_spatialdata(
    sdata,
    table_name="table_immune_cells",
    population_key="leiden_1.0",
    case_key="animal",
)

plot_spatialdata_roi(
    sdata,
    "ROI 1",
    image_modality="extracellular_images",
    channel="Collagen",
    label_layer="tissue_regions",
)

plot_spatialdata_cells(
    sdata,
    ["cell_1", "cell_2"],
    table_name="table_immune_cells",
    image_modality="immune_images",
    channel=["CD3", "CD4", "CD8"],
    outline_target_only=True,
    mask_outside_target=True,
    show_ax_titles=False,
)
```

`get_roi_modalities()` returns every image, label, point, and coordinate-system
association for an ROI. `get_roi_elements()` selects one image and labels
modality, defaulting to the primary quantified IMC panel.
