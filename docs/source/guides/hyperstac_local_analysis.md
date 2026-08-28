# Local analysis of pre-generated HyPERSTAC artifacts

`SpatialBiologyToolkit.hyperstac.local_analysis` provides notebook-oriented
functions for exploring an existing HyPERSTAC asset folder without rerunning
the encoder and without loading saved patch arrays. The normalized per-channel
TIFFs are sufficient for image galleries because patches can be cropped from
the spatial coordinates stored in the representation AnnData.

## Reconstruct cluster label masks

```python
import anndata as ad
import pandas as pd
import seaborn as sns

from SpatialBiologyToolkit.hyperstac import reconstruct_cluster_label_masks

environment_order = [str(value) for value in range(8)]
environment_palette = dict(zip(
    environment_order,
    sns.color_palette("tab10", n_colors=len(environment_order)).as_hex(),
))
environment_colors = pd.Series(
    environment_palette,
    name="HyPERSTAC environment",
)

representation = ad.read_h5ad(
    "hyperstac/imc_hyperstac_representations.h5ad"
)
mask_result = reconstruct_cluster_label_masks(
    representation,
    "leiden_0.25_N30_P20",
    "hyperstac_local_outputs/masks",
    normalised_image_dir="hyperstac/normalised_images",
    cluster_order=environment_order,
    palette=environment_palette,
)
```

Every TIFF has the original ROI dimensions. Raster value zero means that no
accepted HyPERSTAC patch covered the pixel; positive values map to the original
Leiden strings through `mask_result.mapping_path`. Do not interpret the raster
integers as the original cluster labels. The default overlap policy stops on
overlapping patches rather than silently depending on write order.

The same call writes exact-size RGB TIFF and PNG views under
`mask_result.colorized_mask_dir`. These use the supplied palette and black for
value zero. Their paths are recorded in the manifest as
`colorized_tiff_path` and `colorized_png_path`; they are display artifacts, not
replacements for the quantitative uint16 label masks.

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

available_clusters = (
    representation.obs[cluster_col].astype(str).drop_duplicates().tolist()
)
custom_cluster_order = ["1", "0"] + [
    value for value in available_clusters if value not in {"0", "1"}
]

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

## Plot customization

All local HyPERSTAC plot functions retain publication-friendly PNG and SVG
defaults, but the output is no longer tied to those defaults. Customization is
split into three levels:

1. explicit scientific/display arguments such as environment order, marker
   selection, contrast mode, or correlation method;
2. shared `FigureSaveOptions` and `HeatmapOptions` objects;
3. low-level `*_kws` mappings forwarded to Matplotlib or Seaborn.

Define only genuinely shared visual identity—especially the environment
palette—near the start of a notebook. Put figure size, dendrogram, fonts, and
other plot-specific controls immediately beside each plotting call so figures
can be tuned independently.

### When to use an SBT wrapper and when to call Seaborn directly

The wrappers are not intended to replace Seaborn:

- Keep `plot_hyperstac_environment_gallery` and `plot_cluster_map_gallery` for
  their domain-specific work: selecting aligned patches, reading ROI TIFF
  crops, reconstructing coordinates, and composing several image panels.
- Keep `plot_hyperstac_cluster_features` and `plot_environment_abundance` for a
  quick, consistently named exploratory batch. They deliberately save and close
  several figures.
- For a final one-off heatmap, use `hyperstac_cluster_feature_tables` or
  `prepare_environment_abundance_tables` to obtain a DataFrame, then call
  `seaborn.heatmap` or `seaborn.clustermap` directly. This avoids a wrapper layer
  when exact dendrogram, color-bar, axis, and layout control matters.

In other words, SBT retains the scientific data preparation and specialized
image composition. Seaborn remains the clearest interface for a bespoke single
heatmap.

### File formats, resolution, transparency, and save metadata

```python
from SpatialBiologyToolkit.hyperstac import FigureSaveOptions

publication_files = FigureSaveOptions(
    formats=("png", "svg", "pdf"),
    dpi=400,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.05,
    metadata={"Creator": "SpatialBiologyToolkit"},
    savefig_kws={"facecolor": "none"},
)
```

Pass this object as `save_options=publication_files`. For a one-off change,
`formats=("png",)` and `dpi=300` remain accepted directly and override the
corresponding fields. Matplotlib-supported extensions are accepted rather than
being restricted to PNG/SVG.

### Heatmaps and clustermaps

```python
from SpatialBiologyToolkit.hyperstac import HeatmapOptions

common_heatmap = HeatmapOptions(
    cmap="vlag",
    center=0,
    figsize=(11, 6),
    vmin=-2.5,
    vmax=2.5,
    annot=False,
    linewidths=0.25,
    linecolor="#eeeeee",
    row_cluster=True,
    col_cluster=True,
    method="average",
    metric="correlation",
    x_tick_rotation=45,
    y_tick_rotation=0,
    tick_fontsize=8,
    cbar_kws={"label": "relative score"},
    dendrogram_ratio=(0.10, 0.18),
    colors_ratio=(0.03, 0.03),
    cbar_pos=(0.02, 0.80, 0.025, 0.15),
    clustermap_kws={
        # Environments are rows in cluster-feature tables.
        "row_colors": environment_colors,
        "tree_kws": {"linewidths": 0.7, "colors": "black"},
    },
)

plot_hyperstac_cluster_features(
    tables,
    "outputs/features",
    heatmap_options=common_heatmap,
    table_options={
        # A field mapping inherits common_heatmap.
        "mean_marker_intensity": {
            "cmap": "mako",
            "center": None,
            "annot": True,
            "fmt": ".2f",
            "figsize": (14, 7),
            "dendrogram_ratio": (0.08, 0.22),
            # Passed to sns.clustermap last for this table only.
            "clustermap_kws": {
                "tree_kws": {"linewidths": 1.0, "colors": "#333333"},
            },
        },
        # A complete object replaces it for this table.
        "relative_marker_intensity_zscore": HeatmapOptions(
            transpose=True,
            row_order=["CA9", "SMA", "Collagen1"],
            show_clustermap=False,
        ),
    },
    save_options=publication_files,
)
```

This is how clustermap arguments are passed through
`plot_hyperstac_cluster_features`: put shared arguments in
`heatmap_options.clustermap_kws`, or put table-specific arguments in that
table's `table_options[table_name]["clustermap_kws"]`. The raw dictionary is
applied last and can override wrapper defaults. Common examples are
`row_colors`, `col_colors`, `tree_kws`, `row_linkage`, and `col_linkage`.

`HeatmapOptions` also controls `robust`, `square`, masks, row/column order,
display-label mappings, transposition, automatic or explicit figure sizing,
z-scoring, standard scaling, dendrogram/color ratios, color-bar position, all
tick/axis/title/color-bar font sizes, rasterization, and whether fixed or
clustered variants are written. Mapping-style per-table overrides inherit and
merge the shared low-level dictionaries.

For one important figure, direct Seaborn is shorter and more transparent:

```python
import seaborn as sns

matrix = tables["relative_marker_intensity_zscore"]
grid = sns.clustermap(
    matrix,
    row_colors=environment_colors.reindex(matrix.index.astype(str)),
    figsize=(15, 8),
    cmap="vlag",
    center=0,
    metric="correlation",
    method="average",
    dendrogram_ratio=(0.08, 0.20),
    cbar_pos=(0.02, 0.80, 0.025, 0.15),
    tree_kws={"linewidths": 0.8, "colors": "black"},
)
grid.ax_heatmap.tick_params(axis="x", labelsize=9, rotation=45)
grid.ax_heatmap.tick_params(axis="y", labelsize=9)
grid.fig.suptitle("Relative marker intensity", fontsize=14, y=1.02)
```

The feature-table calculation itself can be customized independently:

```python
tables = hyperstac_cluster_feature_tables(
    representation,
    metrics,
    cluster_col,
    permutation_adata=permutation,
    # Ordering arguments must contain every observed cluster exactly once.
    cluster_order=custom_cluster_order,
    channel_order=["SMA", "Glut1", "CA9", "Collagen1"],
    summary_statistic="median",
    zscore_ddof=0,
    permutation_types=("zero_channel",),
    include_all_channel=False,
)
```

### UMAP

For notebooks, call Scanpy directly. Copy the selected namespaced embedding
into a lightweight plotting AnnData so `sc.pl.umap` sees the intended graph:

```python
import scanpy as sc

umap_adata = ad.AnnData(obs=representation.obs[[cluster_col]].copy())
umap_adata.obs[cluster_col] = pd.Categorical(
    umap_adata.obs[cluster_col].astype(str),
    categories=environment_order,
    ordered=True,
)
umap_adata.obsm["X_umap"] = representation.obsm["X_umap_N30_P20"].copy()

sc.pl.umap(
    umap_adata,
    color=cluster_col,
    palette=[environment_palette[value] for value in environment_order],
    size=8,
    alpha=0.65,
    title="Selected HyPERSTAC environments",
    frameon=True,
)
```

`plot_hyperstac_umap` remains as a compatibility/save adapter. Internally it
now delegates to `sc.pl.umap`; put additional Scanpy arguments in
`scanpy_kws`. Direct Scanpy is clearer for interactive or one-off figures.

### Back-gated mask maps and image overlays

```python
plot_cluster_map_gallery(
    mask_result,
    "outputs/roi_maps",
    rois=["ROI_1", "ROI_2"],
    ncols=2,
    palette="tab10",
    cluster_labels={"0": "Matrix", "1": "Vascular"},
    roi_title_template="{roi} — coverage {coverage_fraction:.1%}",
    background_image_dir="hyperstac/normalised_images",
    background_channel="SMA",
    background_cmap="gray",
    background_percentiles=(1, 99),
    show_uncovered=False,
    mask_alpha=0.45,
    show_roi_titles=True,
    roi_title_fontsize=9,
    show_title=False,
    row_spacing=0.12,
    column_spacing=0.08,
    panel_border_color="black",
    panel_border_width=0.5,
    legend_kws={"frameon": False, "ncol": 2},
    save_options=publication_files,
)
```

When ROIs are not listed explicitly, `roi_selection` can choose evenly spaced,
first, highest-coverage, or lowest-coverage examples. Panel sizes, total figure
size, interpolation, origin, uncovered color/label, legend contents, ROI/global
title visibility and font sizes, row/column spacing, panel outlines, background
and mask `imshow` keywords, and axes are configurable. Legend and title keyword
dictionaries remain available for settings that do not deserve another named
argument.

`reconstruct_cluster_label_masks` uses `palette` for the exact-size colorized
TIFF/PNG files and, by default, for optional one-ROI previews.
`preview_palette` can override only the preview. Preview figure, title, legend,
image, and save options remain available. Changing `cluster_order` changes
positive raster values, so always keep and use the newly written mapping CSV
with those masks.

### Combined environment gallery

Automatic selection remains the default, but markers and patches can be fixed
for a manuscript figure:

```python
gallery = plot_hyperstac_environment_gallery(
    representation,
    metrics,
    cluster_col,
    "hyperstac/normalised_images",
    "outputs/environment_gallery",
    permutation_adata=permutation,
    marker_source="zero_channel",
    marker_ranking="difference_from_rest",
    markers_by_cluster={
        "0": ["Panlaminin", "Nidogen2", "CA9"],
        "1": ["SMA", "Glut1", "Syndecan1"],
    },
    patch_ids_by_cluster={"0": ["patch_a", "patch_b"]},
    examples_per_cluster=6,
    selection_strategy="median",       # highest/lowest/median/quantile/random
    selection_quantile=0.75,
    unique_rois=True,
    contrast_mode="fixed",             # per_patch/per_roi/fixed/none
    fixed_vmin=0,
    fixed_vmax={"SMA": 0.8, "Glut1": 0.6},
    gamma=0.8,
    channel_weights=(1.0, 0.8, 0.8),
    invert_channels=(),
    roi_title_template="{roi} | {patch_id}",
    show_roi_titles=True,
    roi_title_fontsize=7,
    show_title=True,
    title_fontsize=13,
    cluster_label_template="Environment {cluster}",
    cluster_label_fontsize=9,
    marker_label_fontsize=8,
    boxed_marker_key=True,
    marker_key_facecolor="black",
    marker_text_colors=("red", "lime", "dodgerblue"),
    border_color="black",
    border_width=0.5,
    row_spacing=0.14,
    column_spacing=0,       # panels touch; values are fractions of patch width
    label_width=1.55,       # separate marker-key column width in inches
    image_kws={"interpolation": "bilinear"},
    save_options=publication_files,
)
```

Manual mappings may cover only selected clusters; other clusters continue to
use automatic selection. The function validates that manual markers exist and
manual patches belong to the requested cluster. Additional controls cover
cluster order, per-patch/per-ROI percentile limits, lower percentile, gamma,
channel inversion/weights, RGB label names, figure/panel/label dimensions,
global and ROI title visibility, every text size, marker-key geometry and text,
face color, borders, axes, row/column spacing, table writing, and Matplotlib
layout keywords. The patch outline now defaults to black at 0.5 points. The
default left key is a black box with marker names drawn in their red, green, or
blue channel color; set `boxed_marker_key=False` for the former plain key.
The key occupies its own GridSpec column, so its width no longer enlarges every
image cell. `column_spacing` is applied after Matplotlib preserves square image
pixels and is measured as a fraction of the rendered patch width: `0` is truly
flush, `0.05` is a 5% gap. `label_width` changes only the dedicated key column.
`show_patch_titles` and `patch_title_template` are retained as compatibility
aliases; new code should use the clearer `show_roi_titles` and
`roi_title_template` names.

### Abundance and metadata panels

```python
plot_environment_abundance(
    abundance,
    "outputs/abundance",
    environment_order=abundance["environment"].astype(str).drop_duplicates().tolist(),
    environment_labels={"0": "Matrix", "1": "Vascular", "2": "Collagen"},
    overall_sort="abundance_descending",
    overall_palette="colorblind",
    show_overall_title=True,
    overall_title_fontsize=13,
    overall_axis_label_fontsize=10,
    overall_tick_fontsize=9,
    show_bar_values=True,
    bar_value_fontsize=8,
    overall_axis_border_color="black",
    overall_axis_border_width=0.8,
    bar_kws={"edgecolor": "black", "linewidth": 0.4},
    sample_group_col="Case",
    categorical_metadata=["Disease_Severity"],
    categorical_statistic="median",
    numeric_metadata=["Tumour_Size_cm3"],
    correlation_method="pearson",
    min_numeric_observations=5,
    roi_heatmap_options={
        "show_clustermap": False,
        "clustermap_kws": {"col_colors": environment_colors},
    },
    sample_heatmap_options={
        "metric": "correlation",
        "clustermap_kws": {"col_colors": environment_colors},
    },
    categorical_table_options={
        "Disease_Severity": {
            "figsize": (11, 5),
            "dendrogram_ratio": (0.08, 0.18),
            "clustermap_kws": {"tree_kws": {"linewidths": 0.6}},
        }
    },
    numeric_heatmap_options={
        "annot": True,
        "fmt": ".2f",
        "clustermap_kws": {"row_colors": environment_colors},
    },
    save_options=publication_files,
)
```

Overall, ROI, sample-group, categorical, and numeric panels can each be
disabled. Bar title visibility, all fonts, labels, dimensions, values, ticks,
axis outline/grid, and low-level keywords are independent. Each categorical
metadata field can override the shared categorical heatmap style via
`categorical_table_options`. Categorical summaries support mean or median;
numeric panels support Spearman or Pearson correlations and a minimum
complete-observation threshold.

For bespoke abundance plots, skip the plotting wrapper:

```python
from SpatialBiologyToolkit.hyperstac import prepare_environment_abundance_tables

abundance_tables = prepare_environment_abundance_tables(
    abundance,
    sample_group_col="Case",
    categorical_metadata=["Disease_Severity"],
    numeric_metadata=["Tumour_Size_cm3"],
    correlation_method="spearman",
)

sns.clustermap(
    abundance_tables.sample_group_fraction,
    # Environments are columns in this matrix.
    col_colors=environment_colors.reindex(
        abundance_tables.sample_group_fraction.columns.astype(str)
    ),
    figsize=(14, 7),
    cmap="mako",
    dendrogram_ratio=(0.08, 0.18),
    tree_kws={"linewidths": 0.7},
)
```

### Cell-composition panels

```python
plot_cell_environment_composition(
    composition,
    "outputs/cells",
    metrics=("cell_count", "fraction_within_environment"),
    environment_order=(
        composition["environment"].astype(str).drop_duplicates().tolist()
    ),
    population_order=(
        composition["population"].astype(str).drop_duplicates().tolist()
    ),
    environment_labels={"0": "Matrix", "1": "Vascular"},
    population_labels={"Schwann cells": "Schwann"},
    heatmap_options={"show_clustermap": False, "annot": True},
    metric_options={
        "cell_count": {"cmap": "Blues", "fmt": ".0f"},
        "fraction_within_environment": {"cmap": "mako", "fmt": ".1%"},
    },
    save_options=publication_files,
)
```

Ordering and display labels do not alter the underlying exported identifiers.
Per-metric titles and colormaps can be supplied, table writing and float formats
are controllable, and every general heatmap option remains available.
