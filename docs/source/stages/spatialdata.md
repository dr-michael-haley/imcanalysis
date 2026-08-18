# SpatialData assembly

## What this stage does

`spatialdata` inventories likely spatial assets, combines conservative
discovery with explicitly configured paths, runs the strict declarative
SpatialData planner, and optionally writes a new multimodal SpatialData Zarr.

Run it with:

```bash
sbt run spatialdata
```

The default `spatialdata.action: plan` is read-only. It produces the complete
selection and validation report without constructing or writing SpatialData.
After reviewing that report, set `spatialdata.action: build` and run the stage
again to create the configured Zarr.

## Why it is performed

Spatial projects commonly accumulate a quantified AnnData table, ROI-local
cell masks, one or more image panels, histology, categorical label rasters, and
matched transcriptomes in separate folders. The stage establishes explicit
relationships among those assets and validates them before they become one
SpatialData object.

The discovery layer is intentionally conservative. Explicit paths always take
priority. Ambiguous primary AnnData, cell-mask, or quantified-image candidates
stop the plan and require configuration rather than being selected from a
folder name.

## Main inputs

The discovery root is `spatialdata.root`. The primary assets may be given
explicitly:

```yaml
spatialdata:
  action: plan
  root: .
  anndata_path: anndata_scan_003.h5ad
  cell_masks_folder: Masks
  primary_images_folder: Images
  output_path: project_spatialdata.zarr
```

When any primary path is null, discovery searches within its configured
bounds and selects it only when there is one high-confidence relationship.
The primary AnnData must contain the configured ROI, instance, X, and Y
columns. The selected mask collection must contain those ROI-local instance
IDs, and the selected image panel must resolve every AnnData variable.

Additional assets can be explicit:

```yaml
spatialdata:
  additional_image_panels:
    - name: matrix_images
      folder: matrix_images_aligned
      panel_name: Matrix panel
      channels: [Collagen, Fibronectin, Laminin]
      allow_partial: true

  histology:
    - name: he
      folder: HE
      allow_partial: true

  region_labels:
    - name: tissue_regions
      folder: TissueRegions
      suffix: _regions
      mapping_path: TissueRegions/region_names.csv
      value_key: label_value
      name_key: label_name
      allow_partial: true

  maxfuse_tables:
    - name: atlas
      adata_path: maxfuse_results/maxfuse_matched_transcriptomes.h5ad
```

Categorical rasters discovered without a semantic value-to-name mapping are
reported but not imported. Mismatched resolutions or orientations require
explicit SpatialData transformations through the Python API; the configuration
stage does not guess rotations or registration transforms.

## Reusable assets produced or modified

With `spatialdata.action: plan`, no reusable project asset is changed.

With `spatialdata.action: build`, the stage creates
`spatialdata.output_path`. Existing paths are never overwritten. The source
AnnData, masks, images, label rasters, mapping tables, and MaxFuse assets are
read-only.

## Human-facing outputs produced

Every managed execution writes:

- all discovered asset candidates and content-based classifications;
- the exact selected candidate for each modality role;
- discovery warnings, informational exclusions, and blocking errors;
- every diagnostic from `plan_spatialdata()`;
- a JSON summary of modality, image, label, point, table, warning, and error
  counts.

These appear in the execution report's `tables/` and `summaries/` categories.
The planner report is the audit surface for this initial implementation.

## Important configuration options

- `action`: `plan` or `build`; defaults to the safe read-only plan.
- `root`: bounded discovery root.
- `anndata_path`, `cell_masks_folder`, `primary_images_folder`: explicit
  primary assets that bypass candidate selection.
- `roi_key`, `instance_key`, `x_key`, `y_key`: primary AnnData relationships.
- `additional_image_panels`, `histology`, `region_labels`,
  `maxfuse_tables`: explicit additional assets.
- `discover_unlisted_assets`: retain or disable generic discovery around the
  explicit assets.
- `scan_depth`, `max_scan_entries`, `sample_files`: discovery bounds.
- `raster_chunks` and `scale_factors`: construction performance and optional
  multiscale settings.

## Environment and resources

The stage uses the CPU `sbt-analysis` environment with SpatialData added
to its explicit pip requirements. The wrapper initially requests 8 CPUs,
128 GB RAM, and 24 hours on the high-memory partition. Raster construction is
lazy, but planning reads label values and mask instances, and construction
loads the primary AnnData table before writing. Review MaxRSS and elapsed time
for representative projects before reducing resources.

## How to interpret the results

Start with the candidate and selection tables. Confirm that the primary
AnnData, masks, and image panel are the intended versions. Then review
per-modality ROI coverage and all planner warnings. Partial coverage can be
intentional; ambiguous files, missing cell instances, unnamed region values,
and identity-aligned shape mismatches are blocking.

Only change `action` to `build` when the selected assets and planner report
match the intended project.

## Common problems and limitations

- Multiple processed AnnData files with the required columns require an
  explicit `anndata_path`.
- Raw and denoised copies of the same image panel may both match the primary
  table; specify `primary_images_folder`.
- A categorical raster cannot be imported until a semantic mapping is
  supplied.
- Equal raster shapes do not prove biological registration. Inspect alignment
  independently when assets were generated by different pipelines.
- Direct MCD, Visium, Xenium, MERFISH, OME-Zarr, geometry, and other unsupported
  source formats are not coerced into the current adapters.
- The stage creates a new store; it does not update an existing Zarr in place.
