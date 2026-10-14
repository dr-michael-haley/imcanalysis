# Experimental Odon export

Convert an SBT project or standalone AnnData + TIFF folders into a portable
dataset for [Odon](https://github.com/alexcoulton/odon). This is an isolated trial
adapter, not a registered pipeline stage. It does not edit the source AnnData,
images, masks, project configuration, or execution history.

## First trial

Run in your SBT scientific Python environment. Start with one or two ROIs.
The commands below are single-line commands usable in PowerShell or Bash.

Check project defaults and input headers without writing:

```shell
python -m SpatialBiologyToolkit.odon_export --project /path/to/project --output /path/to/odon_trial --roi ROI_001 --dry-run
```

Remove `--dry-run` to export:

```shell
python -m SpatialBiologyToolkit.odon_export --project /path/to/project --output /path/to/odon_trial --roi ROI_001
```

`--project` reads `config.yaml` through SBT's existing typed config loader. It
uses `general.anndata_path`, `general.masks_folder`, and
`general.denoised_images_folder`, resolved relative to the project directory.
No `.sbt` registration is needed. There is no silent fallback from denoised to
raw images. Override these defaults with explicit paths if needed:

```shell
python -m SpatialBiologyToolkit.odon_export --anndata /data/cells.h5ad --masks /data/masks --images /data/images --output /data/odon_trial --roi ROI_001 --roi ROI_002
```

Explicit CLI paths, including `--output`, are relative to the current working
directory, not `--project`. Repeat `--images` to merge image folders. Omit
`--roi` to export every ROI represented in AnnData. Existing output directories,
even empty ones, are rejected; use a new destination for each trial.

Open Odon, choose **Import Samplesheet CSV...**, and select the exported
`samplesheet.csv`. Select an ROI and open it, or select multiple ROIs for a
mosaic. The `segpath` column links each image to its cell objects. In the object
controls, use **Color by** to select a population or measurement column.
Object workflows are more complete in single-ROI mode than mosaic mode.

## Python API

```python
from SpatialBiologyToolkit.odon_export import export_dataset

result = export_dataset(
    anndata_path="/data/cells.h5ad",
    masks_folder="/data/masks",
    images_folders=["/data/denoised_images"],
    output="/data/odon_trial",
    rois=["ROI_001", "ROI_002"],
    roi_key="ROI",
    object_id_key="ObjectNumber",
    obs_columns=["cell_type", "leiden"],
    channels=["CD3", "CD20", "DNA1"],
    # layer="counts",  # optional: default measurements come from X
)
print(result["samplesheet"])
```

The API accepts an on-disk `.h5ad`, not a live AnnData object. Save any intended
in-memory changes to a separate H5AD before exporting them.

## Inputs and cell identity

- AnnData must contain an ROI observation and positive integer segmentation
  object IDs. Defaults are `ROI` and `ObjectNumber`; override them with
  `--roi-key` and `--object-id-key`.
- `(ROI, ObjectNumber)` must be unique. Object numbers may repeat across ROIs.
  Duplicate AnnData index strings are retained as metadata; they are not used
  to join cells to masks.
- Each ROI needs exactly one 2-D integer TIFF mask, `<masks>/<ROI>.tif` or `.tiff`.
  Zero is background. Missing mask cells, negative labels, and ambiguous mask
  files cause an error. IDs must fit signed int64.
- Images must already align with the mask, with exactly the same XY shape.
  The exporter never resizes or registers level-0 images.
- Per-channel TIFF layouts supported by SBT discovery are
  `images/<ROI>/<channel>.tiff` and `images/<ROI>_<channel>.tiff`.
  AnnData channel aliases and isotope prefixes use the existing SBT matching
  rules. Multiple image folders are merged; duplicate logical names receive
  source suffixes, as in NapariSBT.
- A TIFF directly at `images/<ROI>.tiff` can be a 2-D image or a channel-first
  stack. Stack channels must be explicitly named, in plane order:

  ```shell
  python -m SpatialBiologyToolkit.odon_export --anndata /data/cells.h5ad --masks /data/masks --images /data/stacks --output /data/odon_stack_trial --stack-channel CD3 --stack-channel CD20
  ```

  The API equivalent is `stack_channels=["CD3", "CD20"]`. Do not pass a true
  Z-stack as a channel stack. RGB images, multi-series TIFFs, and general
  multidimensional microscopy files are outside this trial's scope. Select
  specific `--channel` values to exclude unrelated overview files.

Only cells present in the selected AnnData ROIs are exported as objects.
Images retain all tissue, but cells absent from AnnData have no exported object
overlay or fabricated measurements.

## What is written

```text
odon_trial/
  samplesheet.csv
  rois/
    roi_000001/
      image.ome.zarr/
      cells.parquet
    roi_000002/
      ...
```

Folder numbers are portable storage names, not analysis execution IDs. Original
ROI names are retained in the samplesheet, object table, and image metadata.
Samplesheet paths are relative, so the whole directory can be moved together.

**Images:** OME-NGFF 0.4 metadata in a Zarr v2 hierarchy, CYX axes, uint16
intensities, zlib-compressed chunks with one channel per chunk, and up to five
pyramid levels. Defaults are 512-by-512 XY chunks and 2x2 area-mean downsampling.
Odd edge pixels contribute only their actual area; images are not zero-padded.
Use `--levels` and `--chunk-size` to override these choices. Higher levels are
display derivatives; level 0 retains the original spatial resolution.

**Cells:** GeoParquet 1.0 with binary WKB geometry and primitive Arrow property
columns. Polygon geometry follows exact pixel edges (`x=column`, `y=row`, pixel
centres at half-integers) in the same level-0 coordinate system as the image.
No micrometre scale or geographic CRS is invented. Holes, border cells, and
disconnected components are preserved in the exported geometry. Current Odon
code uses exterior polygon rings for some rendering/measurement paths, so holes
may not be honoured by those viewer operations. Use the original mask and SBT
for quantitative mask-based analysis.

Every object retains `ROI`, `ObjectNumber`, `anndata_index`, and a globally
unambiguous string `id` encoding `[ROI, ObjectNumber]`. Centroids are computed
from the pixel geometry. Measurements retain their AnnData dtype and values,
including sparse-matrix zeros. No marker intensities are recomputed from images.

Measurement source is `X` by default. Use `--layer NAME` or `--use-raw` (mutually
exclusive) for other values; `raw.var_names` supplies the names for `raw.X`.
Selecting image channels does not remove AnnData measurement columns.

All scalar `obs` columns are copied by default. Use repeated `--obs-column` to
select a subset, or `--no-obs` to omit them. Categorical values become ordinary
Arrow strings/numbers so Odon can read them. Unsupported structured/date fields
fail with their column name rather than being silently dropped. Name conflicts
are resolved with `obs:` or `measurement:` prefixes and recorded in Parquet
schema metadata. Reserved identity fields cannot be overwritten by observations.

## Intensity encoding

The inspected Odon reader accepts unsigned 8/16-bit image values; this adapter
writes uint16 consistently. Its default `--intensity-mode auto`:

1. Preserves values exactly if they are integers within 0..65535, regardless of
   their source dtype.
2. Otherwise maps each ROI/channel's finite minimum and maximum to 0..65535,
   without percentile clipping. Floating values are quantized for display.
3. Records the original dtype, range, affine scale and offset in
   `image.ome.zarr/.zattrs` under `sbt_export.channels` and logs a warning.

Approximate source intensities can be recovered as
`stored * scale + offset`. Constant-valued channels use an offset with zero
stored values. These derivative images are for viewing, not re-quantification;
constant rescaled channels have no visual contrast. Because scaling can differ
between ROIs, do not interpret identical stored display values as identical
original intensities across ROIs. The cell measurement columns remain the
authoritative unmodified AnnData values. `--intensity-mode preserve` rejects
any image needing rescaling. NaN/infinite image pixels are rejected in both modes.

Image attributes and Parquet metadata also record source paths, file sizes,
modification timestamps, selected measurement source, adapter settings and the
Odon reference commit. Source paths are provenance only; loading the derivative
does not require them. This is not an annotation round-trip or a replacement
for a source-data backup.

## Execution, dependencies, and verification

This is a local CPU export. It does not submit jobs or launch Odon. On a cluster,
run inside an appropriate compute allocation, not on a login node. AnnData
observations and variable metadata are loaded; only the selected backed matrix
is accessed, one ROI at a time (dense, CSR and CSC H5AD storage). Each ROI's mask,
polygon geometry and measurement slice reside in memory, along with a channel
image/pyramid. Channel-stack TIFFs are read as whole stacks. Large datasets can
take substantial time and disk space, especially polygonizing complex masks.

The public package and CLI help import no GUI/scientific stack. Export uses
`anndata>=0.11`, `h5py`, `numpy`, `pandas`, `scipy`, `scikit-image`, `tifffile`,
`shapely`, `pyarrow`, `zarr>=2.16,<4`, and `numcodecs`, plus the existing SBT
dependencies. These are available in the tested SBT scientific environment;
no environment or lockfile is changed by this feature. Zarr v2 output is
explicit even when using the Zarr v3 Python library.

`--dry-run` checks paths, ROI identities, measurement selection and TIFF headers.
It does not read image pixels, validate mask coverage, or guarantee that every
observation can be serialized. Real export performs those checks. Completed
output is published by renaming a temporary sibling directory only after all
ROIs succeed. Failures clean up temporary export files. Existing outputs are
never replaced. A hard process termination may leave a `.sbt-odon-*` temporary
directory, which is not a completed export.

No private execution/reporting system is created. If called inside an existing
SBT reporting context, inputs, the derivative asset and ROI/cell counts are
registered with that reporter. Otherwise the CLI emits a JSON result to stdout
and progress/warnings to stderr.

The test suite uses synthetic data:

```shell
python -m pytest tests/test_odon_export.py -q
```

The interchange contract was checked against Odon commit
`b01faef010b14ea03c33e92a14e438061d257e39`, particularly its
[image reader](https://github.com/alexcoulton/odon/blob/b01faef010b14ea03c33e92a14e438061d257e39/src/data/ome.rs),
[object reader](https://github.com/alexcoulton/odon/blob/b01faef010b14ea03c33e92a14e438061d257e39/src/spatialdata/parquet_shapes.rs),
and [samplesheet format](https://github.com/alexcoulton/odon/blob/b01faef010b14ea03c33e92a14e438061d257e39/docs/data-sources/object-and-overlay-data.md).
A visual acceptance test in the user's installed Odon version is still needed.

## Removal

Delete `SpatialBiologyToolkit/odon_export/` and `tests/test_odon_export.py`.
There are no stage-registry, CLI-registry, config-schema, packaging, SLURM or
environment changes to undo. Any exported datasets are independent directories
at the destination you chose.
