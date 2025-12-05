"""
Reintegrate previously removed markers back into processed AnnData.

This script merges markers that were separated during segmentation (via remove_and_store_markers)
back into the processed AnnData object after batch correction and clustering are complete.
This allows suboptimal markers to be excluded from clustering while still being available
for downstream analysis and visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import scanpy as sc

from .config_and_utils import (
    BasicProcessConfig,
    GeneralConfig,
    SegmentationConfig,
    filter_config_for_dataclass,
    process_config_with_overrides,
    setup_logging,
)


def reintegrate_removed_markers(
    adata_main: ad.AnnData,
    adata_removed: ad.AnnData,
) -> ad.AnnData:
    """
    Reintegrate removed markers back into the main AnnData object.
    
    This function concatenates the removed markers back into the main AnnData along the
    variable (marker) axis, preserving all layers from both AnnData objects. All layers
    from adata_removed (e.g., nimbus_raw, mean_intensities_raw, mean_intensities_normalized)
    are merged with the corresponding layers in adata_main.
    
    Parameters
    ----------
    adata_main : ad.AnnData
        Main processed AnnData object (after batch correction/clustering)
    adata_removed : ad.AnnData
        AnnData containing the removed markers with same layer structure
    
    Returns
    -------
    ad.AnnData
        Combined AnnData with reintegrated markers in all layers
    """
    # Verify cell ordering matches
    if not adata_main.obs_names.equals(adata_removed.obs_names):
        raise ValueError(
            "Cell ordering mismatch between main and removed AnnData objects. "
            "Ensure both are derived from the same segmentation output without filtering cells."
        )
    
    n_cells_main = adata_main.n_obs
    n_cells_removed = adata_removed.n_obs
    
    if n_cells_main != n_cells_removed:
        raise ValueError(
            f"Cell count mismatch: main AnnData has {n_cells_main} cells, "
            f"removed markers AnnData has {n_cells_removed} cells. "
            "Cannot reintegrate markers from different cell populations."
        )
    
    # Check for overlapping marker names
    overlap = set(adata_main.var_names) & set(adata_removed.var_names)
    if overlap:
        logging.warning(
            f"Found {len(overlap)} overlapping markers between main and removed AnnData: {sorted(overlap)}. "
            "These will be skipped to avoid duplication."
        )
        # Filter out overlapping markers
        keep_markers = [m for m in adata_removed.var_names if m not in overlap]
        if not keep_markers:
            raise ValueError("No unique markers to reintegrate after removing overlaps.")
        adata_removed = adata_removed[:, keep_markers].copy()
    
    logging.info(
        f"Reintegrating {adata_removed.n_vars} removed markers into main AnnData "
        f"({adata_main.n_vars} existing markers, {adata_main.n_obs} cells)"
    )
    
    # Log layers to be merged
    main_layers = set(adata_main.layers.keys()) if hasattr(adata_main, 'layers') else set()
    removed_layers = set(adata_removed.layers.keys()) if hasattr(adata_removed, 'layers') else set()
    common_layers = main_layers & removed_layers
    
    if common_layers:
        logging.info(f"Will merge {len(common_layers)} common layers: {sorted(common_layers)}")
    if main_layers - removed_layers:
        logging.info(f"Main-only layers (will keep main data): {sorted(main_layers - removed_layers)}")
    if removed_layers - main_layers:
        logging.warning(
            f"Removed AnnData has layers not in main: {sorted(removed_layers - main_layers)}. "
            "These will be added with NaN for main markers."
        )
    
    # Concatenate along marker axis (add markers as new columns in .X and all layers)
    logging.info("Concatenating markers along variable axis (preserving all layers)")
    
    # Use anndata's concatenate along var axis - this handles layers automatically
    adata_combined = ad.concat(
        [adata_main, adata_removed],
        axis=1,  # Concatenate along marker (variable) axis
        join='outer',  # Include all layers from both
        merge='first',  # Use metadata from first (main) AnnData for obs/uns/obsm/obsp
    )
    
    logging.info(
        f"Reintegration complete: {adata_combined.n_obs} cells × {adata_combined.n_vars} markers "
        f"({adata_main.n_vars} original + {adata_removed.n_vars} reintegrated)"
    )
    
    # Log final layer status
    if hasattr(adata_combined, 'layers') and adata_combined.layers:
        logging.info(f"Combined AnnData has {len(adata_combined.layers)} layers: {sorted(adata_combined.layers.keys())}")
    
    return adata_combined


def main() -> None:
    pipeline_stage = "ReintegrateMarkers"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)
    
    # Load configurations
    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    seg_config = SegmentationConfig(
        **filter_config_for_dataclass(config.get("segmentation", {}), SegmentationConfig)
    )
    process_config = BasicProcessConfig(
        **filter_config_for_dataclass(config.get("process", {}), BasicProcessConfig)
    )
    
    # Determine input/output paths
    main_adata_path = Path(process_config.output_adata_path)
    removed_adata_path = Path(seg_config.removed_markers_anndata_path)
    
    # Make removed path relative to main adata's directory if not absolute
    if not removed_adata_path.is_absolute():
        removed_adata_path = main_adata_path.parent / removed_adata_path
    
    # Output path: add "_reintegrated" suffix
    output_path = main_adata_path.with_name(
        f"{main_adata_path.stem}_reintegrated{main_adata_path.suffix}"
    )
    
    # Check if files exist
    if not main_adata_path.exists():
        raise FileNotFoundError(
            f"Main processed AnnData not found at {main_adata_path}. "
            "Please run batch correction/processing first."
        )
    
    if not removed_adata_path.exists():
        raise FileNotFoundError(
            f"Removed markers AnnData not found at {removed_adata_path}. "
            "No markers were removed during segmentation, or file has been moved."
        )
    
    # Load AnnData objects
    logging.info(f"Loading main processed AnnData from {main_adata_path}")
    adata_main = ad.read_h5ad(main_adata_path)
    logging.info(
        f"Main AnnData: {adata_main.n_obs} cells × {adata_main.n_vars} markers"
    )
    
    logging.info(f"Loading removed markers AnnData from {removed_adata_path}")
    adata_removed = ad.read_h5ad(removed_adata_path)
    logging.info(
        f"Removed markers AnnData: {adata_removed.n_obs} cells × {adata_removed.n_vars} markers"
    )
    logging.info(f"Markers to reintegrate: {list(adata_removed.var_names)}")
    
    # Log layer information
    if hasattr(adata_main, 'layers') and adata_main.layers:
        logging.info(f"Main AnnData layers: {list(adata_main.layers.keys())}")
    if hasattr(adata_removed, 'layers') and adata_removed.layers:
        logging.info(f"Removed markers AnnData layers: {list(adata_removed.layers.keys())}")
    
    # Reintegrate markers (automatically handles all layers)
    adata_combined = reintegrate_removed_markers(adata_main, adata_removed)
    
    # Save combined AnnData
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving reintegrated AnnData to {output_path}")
    adata_combined.write_h5ad(output_path)
    
    logging.info("Marker reintegration complete!")
    logging.info(f"Final AnnData: {adata_combined.n_obs} cells × {adata_combined.n_vars} markers")
    logging.info(
        f"Reintegrated {adata_removed.n_vars} markers: {list(adata_removed.var_names)}"
    )
    
    # Log available clustering/embedding keys
    if hasattr(adata_combined, 'obsm') and adata_combined.obsm:
        logging.info(f"Available embeddings: {list(adata_combined.obsm.keys())}")
    
    leiden_keys = [col for col in adata_combined.obs.columns if col.startswith('leiden')]
    if leiden_keys:
        logging.info(f"Available Leiden clusterings: {leiden_keys}")


if __name__ == "__main__":
    main()
