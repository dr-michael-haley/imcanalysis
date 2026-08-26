"""HyPERSTAC image-representation analysis adapted for SBT managed stages.

The stage implementations in this package were migrated from the local IMC
adaptation of the collaborator-developed workflow. Public documentation cites
the original HyPERSTAC preprint.
"""

from __future__ import annotations

from .local_analysis import (
    HyperstacGalleryResult,
    HyperstacMaskResult,
    aggregate_environment_abundance,
    assign_cells_to_hyperstac_masks,
    hyperstac_cluster_feature_tables,
    plot_cell_environment_composition,
    plot_cluster_map_gallery,
    plot_environment_abundance,
    plot_hyperstac_cluster_features,
    plot_hyperstac_environment_gallery,
    plot_hyperstac_umap,
    reconstruct_cluster_label_masks,
    summarize_cell_environment_composition,
    summarize_environment_abundance,
)

__all__ = [
    "HyperstacGalleryResult",
    "HyperstacMaskResult",
    "aggregate_environment_abundance",
    "assign_cells_to_hyperstac_masks",
    "hyperstac_cluster_feature_tables",
    "plot_cell_environment_composition",
    "plot_cluster_map_gallery",
    "plot_environment_abundance",
    "plot_hyperstac_cluster_features",
    "plot_hyperstac_environment_gallery",
    "plot_hyperstac_umap",
    "reconstruct_cluster_label_masks",
    "summarize_cell_environment_composition",
    "summarize_environment_abundance",
]
