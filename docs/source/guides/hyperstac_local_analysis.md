# Local analysis of pre-generated HyPERSTAC artifacts

`SpatialBiologyToolkit.hyperstac.local_analysis` provides notebook-oriented
functions for exploring an existing HyPERSTAC asset folder without rerunning
the encoder and without loading saved patch arrays. The normalized per-channel
TIFFs are sufficient for image galleries because patches can be cropped from
the spatial coordinates stored in the representation AnnData.

## Reconstruct cluster label masks

```python
import anndata as ad

from SpatialBiologyToolkit.hyperstac import reconstruct_cluster_label_masks

representation = ad.read_h5ad(
    "hyperstac/imc_hyperstac_representations.h5ad"
)
mask_result = reconstruct_cluster_label_masks(
    representation,
    "leiden_0.25_N30_P20",
    "hyperstac_local_outputs/masks",
    normalised_image_dir="hyperstac/normalised_images",
)
```

Every TIFF has the original ROI dimensions. Raster value zero means that no
accepted HyPERSTAC patch covered the pixel; positive values map to the original
Leiden strings through `mask_result.mapping_path`. Do not interpret the raster
integers as the original cluster labels. The default overlap policy stops on
overlapping patches rather than silently depending on write order.

## Build one environment gallery and marker summaries

```python
import anndata as ad

from SpatialBiologyToolkit.hyperstac import (
    hyperstac_cluster_feature_tables,
    plot_hyperstac_cluster_features,
    plot_hyperstac_environment_gallery,
)

metrics = ad.read_h5ad("hyperstac/imc_hyperstac_patch_metrics.h5ad")
permutation = ad.read_h5ad(
    "hyperstac/permutation_sensitivity/imc_permutation_sensitivity.h5ad"
)

tables = hyperstac_cluster_feature_tables(
    representation,
    metrics,
    "leiden_0.25_N30_P20",
    permutation_adata=permutation,
)
plot_hyperstac_cluster_features(tables, "hyperstac_local_outputs/features")

gallery = plot_hyperstac_environment_gallery(
    representation,
    metrics,
    "leiden_0.25_N30_P20",
    "hyperstac/normalised_images",
    "hyperstac_local_outputs/environment_gallery",
    permutation_adata=permutation,
    marker_source="zero_channel",
)
```

The feature plotting function writes fixed Leiden-order heatmaps and
row/column-clustered variants in PNG and SVG. Zeroing and shuffling report how
sensitive the learned representation is to a marker; they are not causal
effects or cell-type annotations. The gallery selects row-specific RGB markers
and records every selected patch and marker in CSV files.

## Abundance and cell relationships

```python
import pandas as pd

from SpatialBiologyToolkit.hyperstac import (
    assign_cells_to_hyperstac_masks,
    plot_environment_abundance,
    summarize_cell_environment_composition,
    summarize_environment_abundance,
)

metadata = pd.read_csv("metadata/sample_metadata.csv")
abundance = summarize_environment_abundance(
    representation,
    "leiden_0.25_N30_P20",
    metadata=metadata,
    metadata_roi_col="ROI_number",
)
plot_environment_abundance(
    abundance,
    "hyperstac_local_outputs/abundance",
    sample_group_col="Case",
    categorical_metadata=["Disease_Severity"],
    numeric_metadata=["Tumour_Size_cm3"],
)

assignments = assign_cells_to_hyperstac_masks(
    cell_adata,
    mask_result.mask_dir,
    mask_result.mapping,
)
composition = summarize_cell_environment_composition(assignments)
```

ROI abundance includes explicit zeros for environments absent from an ROI.
Cell assignment uses `mask[y, x]`, matching the toolkit spatial-permutation
coordinate convention, and reports uncovered, invalid, or out-of-bounds cells
instead of silently dropping them.

For inferential enrichment or depletion, pass the same mask directory and the
positive-value mapping to
`SpatialBiologyToolkit.spatial_permutation.spatial_permutation_zscores`, with
`tissue_exclude_values=(0,)`. Positive z-scores then mean more cell centres in
an environment than expected from its represented patch area within the ROI.

