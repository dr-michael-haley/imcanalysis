#!/usr/bin/env python3
"""
CellPose-SAM Segmentation Script

This script performs nucleus segmentation using the CellPose-SAM model on preprocessed DNA images.
It is designed to work with images that have been preprocessed using the preprocess_dna.py script,
providing the second half of the original createmasks functionality.

CellPose-SAM (CP-SAM) combines CellPose's flow-based segmentation with SAM's attention mechanisms
for improved segmentation accuracy, particularly on challenging cell types and imaging conditions.

Usage:
    python cellpose_sam.py                                               # Uses default config
    python cellpose_sam.py --config custom.yaml                         # Uses custom config
    python cellpose_sam.py --override createmasks.cellpose_cell_diameter=15  # Override diameter
    python cellpose_sam.py --override createmasks.cellprob_threshold=-2.0    # Override threshold
"""

# GPU acceleration imports must be first
import torch
from cellpose import models

import numpy as np
import pandas as pd
import logging
import re
from pathlib import Path
from skimage import io as skio
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from typing import Any, Optional
import seaborn as sb
import random

from .config_and_utils import (
    process_config_with_overrides,
    setup_logging,
    GeneralConfig,
    CreateMasksConfig,
    cleanstring,
    filter_config_for_dataclass
)
from SpatialBiologyToolkit.napari_sbt.features import (
    DISTRIBUTION_FEATURE_DESCRIPTIONS as _DISTRIBUTION_FEATURE_DESCRIPTIONS,
    REGION_IMAGE_FEATURE_DESCRIPTIONS as _REGION_IMAGE_FEATURE_DESCRIPTIONS,
)


def load_cellpose_model(model_name: str, use_gpu: bool = True):
    """
    Load a CellPose model with correct parameter handling for CellPose v4.0.1+.
    
    Parameters
    ----------
    model_name : str
        Model name - 'cpsam' for CellPose-SAM, or full path to user-trained model
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    CellposeModel
        Initialized CellPose model
        
    Notes
    -----
    CellPose v4.0.1+ only supports:
    - 'cpsam' (CellPose-SAM model)
    - Full paths to user-trained models
    Traditional models like 'nuclei', 'cyto', 'cyto2' are no longer available.
    """
    # In CellPose v4.0.1+, only pretrained_model parameter is used
    # model_type parameter is deprecated and ignored
    
    if model_name in ['nuclei', 'cyto', 'cyto2', 'livecell']:
        # Warn about deprecated models and fallback to cpsam
        logging.warning(f"Model '{model_name}' is not available in CellPose v4.0.1+. "
                       f"Traditional CellPose models are no longer supported. "
                       f"Falling back to 'cpsam' (CellPose-SAM).")
        model_name = 'cpsam'
    
    return models.CellposeModel(
        pretrained_model=model_name,
        gpu=use_gpu
    )


def create_qc_overlay(
    image: np.ndarray,
    final_masks: np.ndarray,
    excluded_masks: np.ndarray = None,
    boundary_dilation: int = 0,
    vmin: float = 0,
    vmax_quantile: float = 0.97,
    outline_alpha: float = 0.8
) -> np.ndarray:
    """
    Create an overlay of segmentation masks on a grayscale image and return it as an RGB array.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale DNA image.
    final_masks : np.ndarray
        Final segmentation masks (kept objects).
    excluded_masks : np.ndarray, optional
        Excluded masks (filtered out objects).
    boundary_dilation : int, optional
        Number of pixels to dilate boundaries. Default is 0.
    vmin : float, optional
        Minimum intensity for normalization. Default is 0.
    vmax_quantile : float, optional
        Quantile for maximum intensity normalization. Default is 0.97.
    outline_alpha : float, optional
        Alpha transparency for outlines. Default is 0.8.
        
    Returns
    -------
    np.ndarray
        RGB overlay image.
    """
    from skimage.segmentation import find_boundaries
    from skimage.morphology import binary_dilation
    
    vmax = np.quantile(image, vmax_quantile)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(normalized_image, cmap="gray", interpolation="none")
    ax.axis("off")

    # Create mask overlays
    masks_and_colors = []
    if final_masks is not None and np.any(final_masks > 0):
        masks_and_colors.append((final_masks, 'green'))
    
    if excluded_masks is not None and np.any(excluded_masks > 0):
        masks_and_colors.append((excluded_masks, 'red'))

    for label_array, color in masks_and_colors:
        boundaries = find_boundaries(label_array, mode="outer")
        # Increase boundary thickness if needed
        for _ in range(boundary_dilation):
            boundaries = binary_dilation(boundaries)

        cmap = ListedColormap([[0, 0, 0, 0], plt.cm.colors.to_rgba(color)])
        ax.imshow(boundaries, cmap=cmap, alpha=outline_alpha, interpolation="none")

    fig.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    # Matplotlib version compatibility
    try:
        # <= 3.7.0
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    except:
        # >= 3.8.0
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[...,:3]

    plt.close(fig)
    return img_array


def load_preprocessed_image(image_path: Path) -> np.ndarray:
    """
    Load a preprocessed DNA image.
    
    Parameters
    ----------
    image_path : Path
        Path to the preprocessed image file.
        
    Returns
    -------
    np.ndarray
        Loaded image array.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Preprocessed image not found: {image_path}")
    
    img = skio.imread(image_path)
    return img


def _finite_values(values: np.ndarray) -> np.ndarray:
    """Return finite values as a flat float vector."""
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or NaN if the denominator is zero/missing."""
    if denominator is None or not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
        return np.nan
    return float(numerator) / float(denominator)


def _as_2d_metric_image(
    image: Optional[np.ndarray],
    target_shape: tuple[int, int],
    *,
    order: int = 1,
    anti_aliasing: bool = True,
) -> Optional[np.ndarray]:
    """
    Coerce an image/map to a 2D float array matching the saved mask shape.
    """
    if image is None:
        return None

    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[1:] == target_shape:
            arr = arr[0]
        elif arr.shape[-1] <= 4:
            arr = arr[..., 0]
        else:
            logging.warning(f"Cannot coerce 3D metric image with shape {arr.shape} to 2D")
            return None

    if arr.ndim != 2:
        logging.warning(f"Cannot use metric image with shape {arr.shape}; expected 2D")
        return None

    if tuple(arr.shape) != tuple(target_shape):
        arr = resize(
            arr,
            target_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )

    return arr.astype(np.float32, copy=False)


def _resize_flow_vector(flow_vector: Optional[np.ndarray], target_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """
    Return a CellPose vector field as a float array with shape (2, rows, cols).
    """
    if flow_vector is None:
        return None

    arr = np.asarray(flow_vector)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        return None
    if arr.shape[0] == 2:
        arr = arr.astype(np.float32, copy=False)
    elif arr.shape[-1] == 2:
        arr = np.moveaxis(arr, -1, 0).astype(np.float32, copy=False)
    else:
        return None

    if tuple(arr.shape[1:]) == tuple(target_shape):
        return arr

    resized_components = [
        resize(
            arr[idx],
            target_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32, copy=False)
        for idx in range(2)
    ]
    return np.stack(resized_components, axis=0)


def _iter_numeric_arrays(value: Any):
    """Yield numeric numpy arrays from nested CellPose return structures."""
    if value is None:
        return

    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            yield value
        elif value.dtype == object:
            for item in value.ravel():
                yield from _iter_numeric_arrays(item)
        return

    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_numeric_arrays(item)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_numeric_arrays(item)


def _extract_cellpose_metric_maps(
    flows: Any,
    target_shape: tuple[int, int],
) -> dict[str, Optional[np.ndarray]]:
    """
    Extract CellPose cell probability and flow vectors when present.

    CellPose versions expose flow outputs with slightly different nested shapes.
    This helper deliberately detects maps by array shape rather than relying on a
    single version-specific list index.
    """
    cellprob_candidates: list[np.ndarray] = []
    flow_candidates: list[np.ndarray] = []

    for arr in _iter_numeric_arrays(flows):
        arr = np.squeeze(np.asarray(arr))
        if arr.size == 0:
            continue

        if arr.ndim == 2 and min(arr.shape) > 4:
            cellprob_candidates.append(arr)
        elif arr.ndim == 3:
            if arr.shape[0] == 2 and min(arr.shape[1:]) > 4:
                flow_candidates.append(arr)
            elif arr.shape[-1] == 2 and min(arr.shape[:2]) > 4:
                flow_candidates.append(np.moveaxis(arr, -1, 0))

    cellprob_map = None
    if cellprob_candidates:
        # Cell probability is normally the only 2D CellPose flow output. If
        # more than one appears, prefer a floating point map over label-like data.
        cellprob_candidates = sorted(
            cellprob_candidates,
            key=lambda x: 0 if np.issubdtype(np.asarray(x).dtype, np.floating) else 1,
        )
        cellprob_map = _as_2d_metric_image(cellprob_candidates[0], target_shape)

    flow_vector = None
    if flow_candidates:
        # Pixel-location maps can also be 2-channel arrays. The actual flow
        # vector field has much smaller magnitudes, so prefer that candidate.
        def _flow_candidate_score(candidate: np.ndarray) -> float:
            candidate = _resize_flow_vector(candidate, target_shape)
            if candidate is None:
                return np.inf
            magnitude = np.sqrt(np.sum(candidate.astype(np.float64) ** 2, axis=0))
            finite = _finite_values(magnitude)
            if finite.size == 0:
                return np.inf
            return float(np.nanpercentile(finite, 95))

        flow_candidates = sorted(flow_candidates, key=_flow_candidate_score)
        flow_vector = _resize_flow_vector(flow_candidates[0], target_shape)

    flow_magnitude = None
    if flow_vector is not None:
        flow_magnitude = np.sqrt(np.sum(flow_vector.astype(np.float64) ** 2, axis=0)).astype(np.float32)

    return {
        "cellprob_map": cellprob_map,
        "flow_vector": flow_vector,
        "flow_magnitude": flow_magnitude,
    }


def _calculate_cellpose_flow_errors(
    final_mask: np.ndarray,
    flow_vector: Optional[np.ndarray],
) -> dict[int, float]:
    """
    Calculate CellPose per-mask flow errors when the installed version exposes it.
    """
    if flow_vector is None or not np.any(final_mask > 0):
        return {}

    flow_error_functions = []
    for module_name in ("cellpose.dynamics", "cellpose.metrics"):
        try:
            module = __import__(module_name, fromlist=["flow_error"])
            flow_error = getattr(module, "flow_error", None)
            if flow_error is not None:
                flow_error_functions.append(flow_error)
        except Exception:
            continue

    if not flow_error_functions:
        return {}

    masks_for_error = final_mask.astype(np.int32, copy=False)
    flow_for_error = flow_vector.astype(np.float32, copy=False)
    last_error = None

    for flow_error in flow_error_functions:
        for kwargs in ({}, {"use_gpu": False}):
            try:
                result = flow_error(masks_for_error, flow_for_error, **kwargs)
                errors = result[0] if isinstance(result, tuple) else result
                errors = np.asarray(errors, dtype=np.float64).ravel()
                if errors.size == 0:
                    continue

                max_label = int(final_mask.max())
                label_errors: dict[int, float] = {}
                if errors.size == max_label:
                    for label in range(1, max_label + 1):
                        label_errors[label] = float(errors[label - 1])
                else:
                    for label in range(1, min(max_label + 1, errors.size)):
                        label_errors[label] = float(errors[label])
                return label_errors
            except Exception as exc:
                last_error = exc
                continue

    logging.debug(f"Could not calculate CellPose flow errors: {last_error}")
    return {}


def _add_distribution_features(row: dict[str, Any], values: np.ndarray, prefix: str) -> None:
    """Add shared distribution summaries with legacy underscore column names."""
    from SpatialBiologyToolkit.napari_sbt.features import add_distribution_features

    add_distribution_features(row, values, prefix, separator="_")


def _expanded_bbox_slice(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    pad: int,
) -> tuple[slice, slice]:
    min_row, min_col, max_row, max_col = bbox
    return (
        slice(max(0, min_row - pad), min(image_shape[0], max_row + pad)),
        slice(max(0, min_col - pad), min(image_shape[1], max_col + pad)),
    )


def _add_image_region_features(
    row: dict[str, Any],
    image: Optional[np.ndarray],
    final_mask: np.ndarray,
    ring_labels: np.ndarray,
    region: Any,
    prefix: str,
    *,
    ring_distance: int = 5,
) -> None:
    """Add object, core, border, and local-background summaries for one image map."""
    if image is None:
        _add_distribution_features(row, np.asarray([], dtype=np.float32), prefix)
        row[f"{prefix}_core_mean"] = np.nan
        row[f"{prefix}_border_mean"] = np.nan
        row[f"{prefix}_core_to_border_ratio"] = np.nan
        row[f"{prefix}_weighted_x"] = np.nan
        row[f"{prefix}_weighted_y"] = np.nan
        row[f"{prefix}_weighted_centroid_offset_px"] = np.nan
        row[f"{prefix}_weighted_centroid_offset_fraction_radius"] = np.nan
        row[f"{prefix}_local_bg_pixel_count"] = 0
        row[f"{prefix}_local_bg_mean"] = np.nan
        row[f"{prefix}_local_bg_std"] = np.nan
        row[f"{prefix}_foreground_to_bg_ratio"] = np.nan
        row[f"{prefix}_foreground_bg_contrast"] = np.nan
        row[f"{prefix}_foreground_bg_contrast_z"] = np.nan
        return

    object_mask = region.image.astype(bool)
    local_image = image[region.slice]
    object_values = local_image[object_mask]
    _add_distribution_features(row, object_values, prefix)

    if object_mask.size == 0:
        return

    row[f"{prefix}_weighted_x"] = np.nan
    row[f"{prefix}_weighted_y"] = np.nan
    row[f"{prefix}_weighted_centroid_offset_px"] = np.nan
    row[f"{prefix}_weighted_centroid_offset_fraction_radius"] = np.nan

    local_rows, local_cols = np.nonzero(object_mask)
    weights = np.asarray(object_values, dtype=np.float64)
    valid_weights = np.isfinite(weights)
    if np.any(valid_weights):
        valid_values = weights[valid_weights]
        shifted_weights = valid_values - float(np.min(valid_values))
        if np.sum(shifted_weights) <= np.finfo(float).eps:
            shifted_weights = np.abs(valid_values)
        if np.sum(shifted_weights) > np.finfo(float).eps:
            min_row, min_col, _, _ = region.bbox
            abs_rows = local_rows[valid_weights] + min_row
            abs_cols = local_cols[valid_weights] + min_col
            weighted_y = float(np.average(abs_rows, weights=shifted_weights))
            weighted_x = float(np.average(abs_cols, weights=shifted_weights))
            offset = float(np.hypot(weighted_y - region.centroid[0], weighted_x - region.centroid[1]))
            equivalent_radius = np.sqrt(float(region.area) / np.pi)
            row[f"{prefix}_weighted_x"] = weighted_x
            row[f"{prefix}_weighted_y"] = weighted_y
            row[f"{prefix}_weighted_centroid_offset_px"] = offset
            row[f"{prefix}_weighted_centroid_offset_fraction_radius"] = _safe_ratio(offset, equivalent_radius)

    eroded_mask = binary_erosion(object_mask, iterations=1, border_value=0)
    border_mask = object_mask & ~eroded_mask
    core_values = local_image[eroded_mask]
    border_values = local_image[border_mask]

    row[f"{prefix}_core_mean"] = float(np.mean(core_values)) if core_values.size else np.nan
    row[f"{prefix}_border_mean"] = float(np.mean(border_values)) if border_values.size else np.nan
    row[f"{prefix}_core_to_border_ratio"] = _safe_ratio(
        row[f"{prefix}_core_mean"],
        row[f"{prefix}_border_mean"],
    )

    ring_slice = _expanded_bbox_slice(region.bbox, final_mask.shape, ring_distance)
    ring_mask = (ring_labels[ring_slice] == region.label) & (final_mask[ring_slice] == 0)
    ring_values = image[ring_slice][ring_mask]

    row[f"{prefix}_local_bg_pixel_count"] = int(_finite_values(ring_values).size)
    row[f"{prefix}_local_bg_mean"] = np.nan
    row[f"{prefix}_local_bg_std"] = np.nan
    row[f"{prefix}_foreground_to_bg_ratio"] = np.nan
    row[f"{prefix}_foreground_bg_contrast"] = np.nan
    row[f"{prefix}_foreground_bg_contrast_z"] = np.nan

    ring_values = _finite_values(ring_values)
    if ring_values.size > 0:
        bg_mean = float(np.mean(ring_values))
        bg_std = float(np.std(ring_values))
        foreground_mean = row.get(f"{prefix}_mean", np.nan)
        row[f"{prefix}_local_bg_mean"] = bg_mean
        row[f"{prefix}_local_bg_std"] = bg_std
        row[f"{prefix}_foreground_to_bg_ratio"] = _safe_ratio(foreground_mean, bg_mean)
        row[f"{prefix}_foreground_bg_contrast"] = float(foreground_mean - bg_mean)
        row[f"{prefix}_foreground_bg_contrast_z"] = _safe_ratio(foreground_mean - bg_mean, bg_std)


def _gradient_magnitude(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Calculate a simple image-gradient magnitude map."""
    if image is None:
        return None
    image = image.astype(np.float32, copy=False)
    grad_y, grad_x = np.gradient(image)
    return np.sqrt(grad_y ** 2 + grad_x ** 2).astype(np.float32)


def _region_morphology_features(
    region: Any,
    image_shape: tuple[int, int],
    source_label_by_object: dict[int, int],
) -> dict[str, Any]:
    """Build morphology and positional features for one saved mask object."""
    min_row, min_col, max_row, max_col = region.bbox
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col
    bbox_area = bbox_height * bbox_width
    image_area = image_shape[0] * image_shape[1]

    area = float(region.area)
    perimeter = float(region.perimeter)
    major_axis = float(region.major_axis_length)
    minor_axis = float(region.minor_axis_length)
    convex_area = float(region.convex_area)
    filled_area = float(region.filled_area)

    equivalent_diameter = getattr(region, "equivalent_diameter_area", None)
    if equivalent_diameter is None:
        equivalent_diameter = getattr(region, "equivalent_diameter", np.nan)

    feret_diameter = getattr(region, "feret_diameter_max", np.nan)
    perimeter_crofton = getattr(region, "perimeter_crofton", np.nan)

    edge_distance = min(min_row, min_col, image_shape[0] - max_row, image_shape[1] - max_col)

    row = {
        "ObjectNumber": int(region.label),
        "SourceObjectNumber": int(source_label_by_object.get(int(region.label), region.label)),
        "X_loc": float(region.centroid[1]),
        "Y_loc": float(region.centroid[0]),
        "bbox_min_row": int(min_row),
        "bbox_min_col": int(min_col),
        "bbox_max_row": int(max_row),
        "bbox_max_col": int(max_col),
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        "bbox_area": int(bbox_area),
        "mask_area": area,
        "mask_area_fraction_roi": _safe_ratio(area, image_area),
        "mask_area_fraction_bbox": _safe_ratio(area, bbox_area),
        "mask_perimeter": perimeter,
        "mask_perimeter_crofton": float(perimeter_crofton) if np.isfinite(perimeter_crofton) else np.nan,
        "mask_circularity": _safe_ratio(4 * np.pi * area, perimeter * perimeter),
        "mask_compactness": _safe_ratio(perimeter * perimeter, 4 * np.pi * area),
        "mask_major_axis_length": major_axis,
        "mask_minor_axis_length": minor_axis,
        "mask_axis_ratio": _safe_ratio(major_axis, minor_axis),
        "mask_eccentricity": float(region.eccentricity),
        "mask_solidity": float(region.solidity),
        "mask_extent": float(region.extent),
        "mask_orientation_degrees": float(np.degrees(region.orientation)),
        "mask_equivalent_diameter": float(equivalent_diameter),
        "mask_feret_diameter_max": float(feret_diameter) if np.isfinite(feret_diameter) else np.nan,
        "mask_convex_area": convex_area,
        "mask_filled_area": filled_area,
        "mask_hole_area": float(filled_area - area),
        "mask_hole_fraction": _safe_ratio(filled_area - area, filled_area),
        "mask_convexity": _safe_ratio(area, convex_area),
        "mask_euler_number": int(region.euler_number),
        "mask_edge_touching": bool(edge_distance == 0),
        "mask_min_distance_to_edge_px": int(edge_distance),
    }

    return row


def _add_neighbor_features(
    row: dict[str, Any],
    final_mask: np.ndarray,
    region: Any,
    *,
    neighbor_distance: int = 5,
) -> None:
    """Count nearby segmented objects around a single label."""
    neighborhood_slice = _expanded_bbox_slice(region.bbox, final_mask.shape, neighbor_distance)
    local_labels = final_mask[neighborhood_slice]
    local_object = local_labels == region.label
    if not np.any(local_object):
        row["mask_neighbor_count_5px"] = 0
        row["mask_touching_neighbor_count_1px"] = 0
        return

    neighbor_area = binary_dilation(local_object, iterations=neighbor_distance) & (local_labels != region.label)
    neighbors = np.unique(local_labels[neighbor_area])
    neighbors = neighbors[neighbors > 0]
    row["mask_neighbor_count_5px"] = int(len(neighbors))

    touching_area = binary_dilation(local_object, iterations=1) & (local_labels != region.label)
    touching_neighbors = np.unique(local_labels[touching_area])
    touching_neighbors = touching_neighbors[touching_neighbors > 0]
    row["mask_touching_neighbor_count_1px"] = int(len(touching_neighbors))


def _add_flow_alignment_features(
    row: dict[str, Any],
    flow_vector: Optional[np.ndarray],
    region: Any,
) -> None:
    """Summarize how CellPose flow vectors align with directions toward the object centroid."""
    for suffix in (
        "mean",
        "median",
        "std",
        "q10",
        "q90",
        "abs_mean",
        "fraction_positive",
        "fraction_strong_inward",
    ):
        row[f"cellpose_flow_radial_alignment_{suffix}"] = np.nan

    if flow_vector is None:
        return

    object_mask = region.image.astype(bool)
    local_rows, local_cols = np.nonzero(object_mask)
    if local_rows.size == 0:
        return

    min_row, min_col, _, _ = region.bbox
    abs_rows = local_rows + min_row
    abs_cols = local_cols + min_col

    to_centroid_y = region.centroid[0] - abs_rows
    to_centroid_x = region.centroid[1] - abs_cols
    radial_norm = np.sqrt(to_centroid_y ** 2 + to_centroid_x ** 2)

    flow_y = flow_vector[0, abs_rows, abs_cols]
    flow_x = flow_vector[1, abs_rows, abs_cols]
    flow_norm = np.sqrt(flow_y ** 2 + flow_x ** 2)

    valid = (radial_norm > np.finfo(float).eps) & (flow_norm > np.finfo(float).eps)
    if not np.any(valid):
        return

    alignment = (
        (flow_y[valid] / flow_norm[valid]) * (to_centroid_y[valid] / radial_norm[valid])
        + (flow_x[valid] / flow_norm[valid]) * (to_centroid_x[valid] / radial_norm[valid])
    )
    alignment = _finite_values(alignment)
    if alignment.size == 0:
        return

    q10, q50, q90 = np.quantile(alignment, [0.10, 0.50, 0.90])
    row["cellpose_flow_radial_alignment_mean"] = float(np.mean(alignment))
    row["cellpose_flow_radial_alignment_median"] = float(q50)
    row["cellpose_flow_radial_alignment_std"] = float(np.std(alignment))
    row["cellpose_flow_radial_alignment_q10"] = float(q10)
    row["cellpose_flow_radial_alignment_q90"] = float(q90)
    row["cellpose_flow_radial_alignment_abs_mean"] = float(np.mean(np.abs(alignment)))
    row["cellpose_flow_radial_alignment_fraction_positive"] = float(np.mean(alignment > 0))
    row["cellpose_flow_radial_alignment_fraction_strong_inward"] = float(np.mean(alignment > 0.5))


def _add_roi_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-ROI z-scores and percentile ranks for high-value QC predictors."""
    if df.empty:
        return df

    rank_columns = [
        "mask_area",
        "mask_perimeter",
        "mask_circularity",
        "mask_axis_ratio",
        "dna_raw_mean",
        "dna_raw_sum",
        "dna_raw_foreground_bg_contrast",
        "dna_raw_weighted_centroid_offset_fraction_radius",
        "dna_preprocessed_mean",
        "dna_preprocessed_weighted_centroid_offset_fraction_radius",
        "cellpose_cellprob_mean",
        "cellpose_flow_magnitude_mean",
        "cellpose_flow_error",
    ]

    for col in rank_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mean_value = values.mean(skipna=True)
        std_value = values.std(skipna=True, ddof=0)
        if pd.notna(std_value) and std_value > 0:
            df[f"{col}_roi_zscore"] = (values - mean_value) / std_value
        else:
            df[f"{col}_roi_zscore"] = 0.0
        df[f"{col}_roi_percentile"] = values.rank(pct=True)

    return df


def build_cellpose_cell_metrics(
    roi_name: str,
    final_mask: np.ndarray,
    source_label_by_object: dict[int, int],
    *,
    raw_dna_image: Optional[np.ndarray],
    preprocessed_dna_image: Optional[np.ndarray],
    cellprob_map: Optional[np.ndarray],
    flow_vector: Optional[np.ndarray],
    flow_magnitude: Optional[np.ndarray],
    input_image_path: Path,
    mask_path: Path,
    config: CreateMasksConfig,
    diameter_for_segmentation: float,
    actual_diameter: float,
) -> pd.DataFrame:
    """
    Build one row per saved mask object with morphology, DNA intensity, and CellPose QC metrics.
    """
    final_mask = np.asarray(final_mask)
    if final_mask.ndim != 2:
        raise ValueError(f"Final mask must be 2D, got shape {final_mask.shape}")

    target_shape = tuple(final_mask.shape)
    raw_dna_image = _as_2d_metric_image(raw_dna_image, target_shape)
    preprocessed_dna_image = _as_2d_metric_image(preprocessed_dna_image, target_shape)
    cellprob_map = _as_2d_metric_image(cellprob_map, target_shape)
    flow_vector = _resize_flow_vector(flow_vector, target_shape)
    flow_magnitude = _as_2d_metric_image(flow_magnitude, target_shape)

    raw_dna_gradient = _gradient_magnitude(raw_dna_image)
    preprocessed_dna_gradient = _gradient_magnitude(preprocessed_dna_image)

    ring_labels = expand_labels(final_mask, distance=5) if np.any(final_mask > 0) else final_mask
    flow_errors = _calculate_cellpose_flow_errors(final_mask, flow_vector)
    props = regionprops(final_mask)
    rows: list[dict[str, Any]] = []

    for region in props:
        row = _region_morphology_features(region, target_shape, source_label_by_object)
        row.update({
            "ROI": roi_name,
            "CellID": f"{roi_name}_{int(region.label)}",
            "Input_image": str(input_image_path),
            "Mask_output": str(mask_path),
            "Image_rows": int(target_shape[0]),
            "Image_cols": int(target_shape[1]),
            "Model_type": config.cell_pose_sam_model,
            "Diameter_used": diameter_for_segmentation,
            "Actual_diameter": actual_diameter,
            "CellProb_threshold": config.cellprob_threshold,
            "Flow_threshold": config.flow_threshold,
            "Min_size": config.min_cell_area or 15,
            "Max_size_fraction": config.max_size_fraction,
            "Expand_masks": config.expand_masks,
            "Fill_holes": config.fill_holes,
            "Remove_edge_masks": config.remove_edge_masks,
            "Run_upscale": config.run_upscale,
            "CellPose_scaling_factor": 30.0 / diameter_for_segmentation,
            "Upscale_ratio": config.calculated_upscale_ratio if config.run_upscale else 1.0,
            "cellpose_flow_error": flow_errors.get(int(region.label), np.nan),
        })

        _add_neighbor_features(row, final_mask, region)

        _add_image_region_features(row, raw_dna_image, final_mask, ring_labels, region, "dna_raw")
        _add_image_region_features(row, preprocessed_dna_image, final_mask, ring_labels, region, "dna_preprocessed")
        _add_image_region_features(row, cellprob_map, final_mask, ring_labels, region, "cellpose_cellprob")
        _add_image_region_features(row, flow_magnitude, final_mask, ring_labels, region, "cellpose_flow_magnitude")

        if raw_dna_gradient is not None:
            _add_distribution_features(
                row,
                raw_dna_gradient[region.slice][region.image.astype(bool)],
                "dna_raw_gradient",
            )
        else:
            _add_distribution_features(row, np.asarray([], dtype=np.float32), "dna_raw_gradient")
        if preprocessed_dna_gradient is not None:
            _add_distribution_features(
                row,
                preprocessed_dna_gradient[region.slice][region.image.astype(bool)],
                "dna_preprocessed_gradient",
            )
        else:
            _add_distribution_features(row, np.asarray([], dtype=np.float32), "dna_preprocessed_gradient")

        _add_flow_alignment_features(row, flow_vector, region)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if len(df) > 1:
        try:
            from scipy.spatial import cKDTree

            coords = df[["Y_loc", "X_loc"]].to_numpy(dtype=float)
            tree = cKDTree(coords)
            nearest_distances, _ = tree.query(coords, k=2)
            df["nearest_centroid_distance_px"] = nearest_distances[:, 1]
            df["centroid_neighbor_count_25px"] = [
                len(tree.query_ball_point(coord, r=25)) - 1 for coord in coords
            ]
            df["centroid_neighbor_count_50px"] = [
                len(tree.query_ball_point(coord, r=50)) - 1 for coord in coords
            ]
        except Exception as exc:
            logging.debug(f"Could not calculate centroid-neighbor features for {roi_name}: {exc}")
            df["nearest_centroid_distance_px"] = np.nan
            df["centroid_neighbor_count_25px"] = np.nan
            df["centroid_neighbor_count_50px"] = np.nan
    else:
        df["nearest_centroid_distance_px"] = np.nan
        df["centroid_neighbor_count_25px"] = 0
        df["centroid_neighbor_count_50px"] = 0

    df["roi_object_count"] = len(df)
    df["roi_mask_area_fraction"] = df["mask_area"].sum() / float(final_mask.size)
    df = _add_roi_rank_features(df)

    leading_cols = [
        "ROI",
        "ObjectNumber",
        "SourceObjectNumber",
        "CellID",
        "X_loc",
        "Y_loc",
        "Input_image",
        "Mask_output",
    ]
    remaining_cols = [col for col in df.columns if col not in leading_cols]
    return df.loc[:, leading_cols + remaining_cols]


_IMAGE_FEATURE_PREFIXES = {
    "dna_raw_gradient": (
        "nucleus_stain_raw_gradient",
        "Gradient magnitude of the original denoised nucleus stain",
    ),
    "dna_preprocessed_gradient": (
        "nucleus_stain_preprocessed_gradient",
        "Gradient magnitude of the preprocessed nucleus stain used for CellPose-SAM",
    ),
    "dna_raw": (
        "nucleus_stain_raw_intensity",
        "Original denoised nucleus stain intensity",
    ),
    "dna_preprocessed": (
        "nucleus_stain_preprocessed_intensity",
        "Preprocessed nucleus stain intensity used for CellPose-SAM",
    ),
    "cellpose_cellprob": (
        "cellpose_probability",
        "CellPose-SAM cell-probability map value",
    ),
    "cellpose_flow_magnitude": (
        "cellpose_flow",
        "CellPose-SAM flow-vector magnitude",
    ),
}


_EXACT_FEATURE_DICTIONARY = {
    "ROI": ("identifier", "Region of interest that this object came from."),
    "ObjectNumber": ("identifier", "Saved object ID in the final relabeled mask for this ROI."),
    "SourceObjectNumber": ("identifier", "Original CellPose label before size filtering and final relabeling."),
    "CellID": ("identifier", "Combined ROI and ObjectNumber identifier."),
    "Input_image": ("file_metadata", "Preprocessed nucleus image path used as input to CellPose-SAM."),
    "Mask_output": ("file_metadata", "Saved final mask path for this ROI."),
    "X_loc": ("mask_position", "Object centroid X coordinate in pixels."),
    "Y_loc": ("mask_position", "Object centroid Y coordinate in pixels."),
    "Image_rows": ("roi_metadata", "Number of rows in the final mask image."),
    "Image_cols": ("roi_metadata", "Number of columns in the final mask image."),
    "bbox_min_row": ("mask_position", "Top row of the object bounding box."),
    "bbox_min_col": ("mask_position", "Left column of the object bounding box."),
    "bbox_max_row": ("mask_position", "Bottom row of the object bounding box, exclusive."),
    "bbox_max_col": ("mask_position", "Right column of the object bounding box, exclusive."),
    "bbox_width": ("mask_position", "Width of the object bounding box in pixels."),
    "bbox_height": ("mask_position", "Height of the object bounding box in pixels."),
    "bbox_area": ("mask_position", "Area of the object bounding box in pixels."),
    "mask_area": ("mask_morphology", "Object mask area in pixels."),
    "mask_area_fraction_roi": ("mask_morphology", "Fraction of the ROI covered by this object mask."),
    "mask_area_fraction_bbox": ("mask_morphology", "Fraction of the bounding box occupied by the object mask."),
    "mask_perimeter": ("mask_morphology", "Object perimeter in pixels."),
    "mask_perimeter_crofton": ("mask_morphology", "Crofton perimeter estimate for the object."),
    "mask_circularity": ("mask_morphology", "Circularity, calculated as 4*pi*area/perimeter^2."),
    "mask_compactness": ("mask_morphology", "Inverse circularity-like compactness, calculated as perimeter^2/(4*pi*area)."),
    "mask_major_axis_length": ("mask_morphology", "Major-axis length of the ellipse with matching second moments."),
    "mask_minor_axis_length": ("mask_morphology", "Minor-axis length of the ellipse with matching second moments."),
    "mask_axis_ratio": ("mask_morphology", "Major-axis length divided by minor-axis length."),
    "mask_eccentricity": ("mask_morphology", "Eccentricity of the ellipse with matching second moments."),
    "mask_solidity": ("mask_morphology", "Object area divided by convex hull area."),
    "mask_extent": ("mask_morphology", "Object area divided by bounding box area."),
    "mask_orientation_degrees": ("mask_morphology", "Orientation of the object major axis in degrees."),
    "mask_equivalent_diameter": ("mask_morphology", "Diameter of a circle with the same area as the object."),
    "mask_feret_diameter_max": ("mask_morphology", "Maximum Feret diameter of the object."),
    "mask_convex_area": ("mask_morphology", "Area of the convex hull around the object."),
    "mask_filled_area": ("mask_morphology", "Object area after filling internal holes."),
    "mask_hole_area": ("mask_morphology", "Filled area minus object area."),
    "mask_hole_fraction": ("mask_morphology", "Internal hole area divided by filled area."),
    "mask_convexity": ("mask_morphology", "Object area divided by convex area."),
    "mask_euler_number": ("mask_morphology", "Euler number of the object, reflecting holes and disconnected components."),
    "mask_edge_touching": ("mask_context", "Whether the object touches any ROI image edge."),
    "mask_min_distance_to_edge_px": ("mask_context", "Minimum distance from object bounding box to an ROI edge in pixels."),
    "mask_neighbor_count_5px": ("mask_context", "Number of other masks within five pixels of this object."),
    "mask_touching_neighbor_count_1px": ("mask_context", "Number of other masks touching this object within a one-pixel dilation."),
    "nearest_centroid_distance_px": ("mask_context", "Distance to the nearest other object centroid in pixels."),
    "centroid_neighbor_count_25px": ("mask_context", "Number of other object centroids within 25 pixels."),
    "centroid_neighbor_count_50px": ("mask_context", "Number of other object centroids within 50 pixels."),
    "roi_object_count": ("roi_context", "Number of kept objects in this ROI."),
    "roi_mask_area_fraction": ("roi_context", "Fraction of ROI pixels covered by all kept objects."),
    "Model_type": ("segmentation_config", "CellPose-SAM model identifier used for segmentation."),
    "Diameter_used": ("segmentation_config", "Diameter passed to CellPose-SAM after accounting for preprocessing."),
    "Actual_diameter": ("segmentation_config", "Recorded CellPose-SAM diameter value; currently the diameter used."),
    "CellProb_threshold": ("segmentation_config", "CellPose-SAM cell probability threshold used for mask creation."),
    "Flow_threshold": ("segmentation_config", "CellPose-SAM flow threshold used for mask creation."),
    "Min_size": ("segmentation_config", "Minimum object area threshold used for retained masks."),
    "Max_size_fraction": ("segmentation_config", "Maximum object area as a fraction of ROI area."),
    "Expand_masks": ("segmentation_config", "Configured pixel distance used to expand masks after segmentation."),
    "Fill_holes": ("segmentation_config", "Whether holes were filled in masks after segmentation."),
    "Remove_edge_masks": ("segmentation_config", "Whether CellPose edge-mask removal was applied."),
    "Run_upscale": ("segmentation_config", "Whether preprocessing upscaling was configured."),
    "CellPose_scaling_factor": ("segmentation_config", "CellPose internal scaling factor implied by the diameter."),
    "Upscale_ratio": ("segmentation_config", "Configured or inferred preprocessing upscale ratio."),
    "cellpose_flow_error": ("cellpose_quality", "CellPose per-object flow consistency error when available."),
    "cellpose_flow_radial_alignment_mean": ("cellpose_flow", "Mean alignment between CellPose flow direction and direction toward the mask centroid."),
    "cellpose_flow_radial_alignment_median": ("cellpose_flow", "Median alignment between CellPose flow direction and direction toward the mask centroid."),
    "cellpose_flow_radial_alignment_std": ("cellpose_flow", "Standard deviation of flow-to-centroid alignment values."),
    "cellpose_flow_radial_alignment_q10": ("cellpose_flow", "10th percentile of flow-to-centroid alignment values."),
    "cellpose_flow_radial_alignment_q90": ("cellpose_flow", "90th percentile of flow-to-centroid alignment values."),
    "cellpose_flow_radial_alignment_abs_mean": ("cellpose_flow", "Mean absolute flow-to-centroid alignment value."),
    "cellpose_flow_radial_alignment_fraction_positive": ("cellpose_flow", "Fraction of object pixels where flow points at least partly toward the centroid."),
    "cellpose_flow_radial_alignment_fraction_strong_inward": ("cellpose_flow", "Fraction of object pixels where flow has strong inward alignment greater than 0.5."),
}


def _describe_cell_metric_feature(feature_name: str) -> tuple[str, str]:
    """
    Return a feature type/source and plain-language description for one metrics column.
    """
    if feature_name.endswith("_roi_zscore"):
        base_feature = feature_name[:-len("_roi_zscore")]
        _, base_description = _describe_cell_metric_feature(base_feature)
        return (
            "within_roi_normalization",
            f"Within-ROI z-score of {base_feature}; base feature: {base_description}",
        )

    if feature_name.endswith("_roi_percentile"):
        base_feature = feature_name[:-len("_roi_percentile")]
        _, base_description = _describe_cell_metric_feature(base_feature)
        return (
            "within_roi_normalization",
            f"Within-ROI percentile rank of {base_feature}; base feature: {base_description}",
        )

    if feature_name in _EXACT_FEATURE_DICTIONARY:
        return _EXACT_FEATURE_DICTIONARY[feature_name]

    for prefix, (type_source, source_description) in _IMAGE_FEATURE_PREFIXES.items():
        prefix_with_separator = f"{prefix}_"
        if not feature_name.startswith(prefix_with_separator):
            continue

        suffix = feature_name[len(prefix_with_separator):]
        if suffix in _DISTRIBUTION_FEATURE_DESCRIPTIONS:
            return type_source, f"{source_description}: {_DISTRIBUTION_FEATURE_DESCRIPTIONS[suffix]}"
        if suffix in _REGION_IMAGE_FEATURE_DESCRIPTIONS:
            return type_source, f"{source_description}: {_REGION_IMAGE_FEATURE_DESCRIPTIONS[suffix]}"

    return "unclassified", "Feature generated by CellPose-SAM mask metrics extraction; no specific description is defined."


def build_cellpose_feature_dictionary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a CSV-friendly feature dictionary indexed by the metrics dataframe columns.
    """
    rows = []
    for feature_name in metrics_df.columns:
        type_source, description = _describe_cell_metric_feature(str(feature_name))
        rows.append({
            "feature": str(feature_name),
            "type/source": type_source,
            "description": description,
        })

    return pd.DataFrame(rows).set_index("feature")


def _parameter_value_slug(value: Any) -> str:
    """
    Create a filesystem-safe parameter value slug while preserving numeric signs.
    """
    value_str = str(value).strip()
    numeric_match = re.fullmatch(r'([+-]?)(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value_str)

    if numeric_match:
        sign = numeric_match.group(1)
        magnitude = value_str[1:] if sign else value_str
        value_slug = cleanstring(magnitude)
        if sign == '-':
            return f"neg_{value_slug}"
        if sign == '+':
            return f"pos_{value_slug}"
        return value_slug

    return cleanstring(value_str)


def _parameter_set_slug(param_set: dict[str, Any], param_a: str, param_b: str) -> str:
    """
    Create a unique slug for a parameter scan combination.
    """
    return (
        f"{cleanstring(param_a)}-{_parameter_value_slug(param_set[param_a])}"
        f"_{cleanstring(param_b)}-{_parameter_value_slug(param_set[param_b])}"
    )


def _validate_parameter_set_slugs(param_sets: list[dict[str, Any]], param_a: str, param_b: str) -> None:
    """
    Ensure distinct parameter sets do not collapse onto the same filesystem slug.
    """
    slug_to_param_set: dict[str, dict[str, Any]] = {}
    for param_set in param_sets:
        slug = _parameter_set_slug(param_set, param_a, param_b)
        existing = slug_to_param_set.get(slug)
        if existing is not None and existing != param_set:
            raise ValueError(
                "Parameter scan folder naming collision detected: "
                f"{existing} and {param_set} both map to '{slug}'."
            )
        slug_to_param_set[slug] = param_set


def segment_single_roi(
    roi_name: str,
    image_path: Path,
    output_dir: Path,
    qc_dir: Path,
    config: CreateMasksConfig,
    denoised_folder: Path,
    cp_sam_model=None,
    collect_cell_metrics: bool = False,
) -> dict:
    """
    Segment a single ROI using CellPose-SAM.
    
    Parameters
    ----------
    roi_name : str
        Name of the ROI.
    image_path : Path
        Path to the preprocessed DNA image.
    output_dir : Path
        Output directory for masks.
    qc_dir : Path
        QC directory for overlay images.
    config : CreateMasksConfig
        Configuration object.
    denoised_folder : Path
        Path to the denoised images folder for overlay generation.
    cp_sam_model : CellposeModel, optional
        Pre-loaded CellPose-SAM model.
    collect_cell_metrics : bool, optional
        Whether to return per-cell mask/DNA/CellPose metrics for normal
        segmentation mode.
        
    Returns
    -------
    dict
        Segmentation results and statistics.
    """
    logging.info(f"Segmenting ROI: {roi_name}")
    
    # Load preprocessed image
    img = load_preprocessed_image(image_path)
    preprocessed_shape = img.shape
    
    # Load original denoised DNA image to get true original dimensions
    from .config_and_utils import get_filename
    roi_denoised_path = denoised_folder / roi_name
    original_dna_img = None
    original_shape = None
    
    if roi_denoised_path.exists():
        try:
            dna_file = get_filename(roi_denoised_path, config.dna_image_name)
            original_dna_img = skio.imread(roi_denoised_path / dna_file)
            original_shape = original_dna_img.shape
            logging.debug(f"Loaded original denoised DNA image: {original_shape}")
        except Exception as e:
            logging.warning(f"Could not load original DNA image for {roi_name}: {e}")
            original_shape = preprocessed_shape  # Fallback
    else:
        logging.warning(f"Denoised ROI folder not found: {roi_denoised_path}")
        original_shape = preprocessed_shape  # Fallback
    
    # Initialize model if not provided
    if cp_sam_model is None:
        # Determine if GPU is available and working
        use_gpu = torch.cuda.is_available()
        cp_sam_model = load_cellpose_model(config.cell_pose_sam_model, use_gpu)
    
    # Prepare normalization parameters
    normalize_params = {
        'normalize': config.image_normalise,
        'percentile': [config.image_normalise_percentile_lower, config.image_normalise_percentile_upper]
    }
    
    # Run CellPose-SAM segmentation
    # Handle diameter parameter for both CellPose v3 and v4+ compatibility
    # In v4+, diameter is used for image scaling (30.0 / diameter)
    # If images were upscaled during preprocessing, we need to account for this
    diameter_for_segmentation = config.cellpose_cell_diameter
    if config.run_upscale:
        # CellPose upscale models have fixed target diameters:
        # upsample_nuclei -> 17.0 pixels, upsample_cyto3 -> 30.0 pixels
        # Use the actual target diameter rather than our assumed ratio
        diameter_for_segmentation = config.upscale_target_diameter
    
    logging.debug(f"Running CellPose-SAM on {roi_name} with diameter={diameter_for_segmentation}")
    logging.debug(f"Image shape for segmentation: {img.shape}")
    logging.debug(f"Expected scaling factor in CellPose: {30.0 / diameter_for_segmentation:.2f}x")
    
    try:
        # Optimize batch size for CPU vs GPU
        batch_size = config.batch_size if torch.cuda.is_available() else 1
        
        masks, flows, styles = cp_sam_model.eval(
            img,
            diameter=diameter_for_segmentation,
            channels=None,  # Grayscale image
            batch_size=batch_size,
            normalize=normalize_params,
            cellprob_threshold=config.cellprob_threshold,
            flow_threshold=config.flow_threshold,
            min_size=config.min_cell_area or 15,
            augment=config.augment,
            compute_masks=True
        )
        
        # In CellPose v4+, diameter info might be in styles or we use the input diameter
        actual_diameter = diameter_for_segmentation  # Fallback to input diameter
        
    except Exception as e:
        logging.error(f"CellPose-SAM segmentation failed for {roi_name}: {str(e)}")
        raise
    
    # If preprocessing included upscaling, we need to downscale the masks back to original dimensions
    if config.run_upscale and original_shape is not None:
        logging.debug(f"Downscaling masks from {masks.shape} to original size {original_shape}")
        masks = resize(masks, original_shape, order=0, preserve_range=True, anti_aliasing=False)
        masks = masks.astype(np.uint16)
    
    # Process masks
    if config.fill_holes:
        # Fill holes in masks
        unique_labels = np.unique(masks)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            mask_binary = (masks == label)
            mask_filled = binary_fill_holes(mask_binary)
            masks[mask_filled] = label
    
    # Remove edge masks if requested
    if config.remove_edge_masks:
        from cellpose.utils import remove_edge_masks
        masks = remove_edge_masks(masks, change_index=True)
    
    # Expand masks if requested
    if config.expand_masks > 0:
        masks = expand_labels(masks, distance=config.expand_masks)
    
    # Calculate statistics
    region_props = regionprops(masks)
    total_objects = len(region_props)
    
    # Apply size filtering based on original (downscaled) image dimensions
    final_mask = np.zeros_like(masks, dtype=np.uint16)
    excluded_mask = np.zeros_like(masks, dtype=np.uint16)
    kept_objects = 0
    source_label_by_object: dict[int, int] = {}
    
    image_area = masks.shape[0] * masks.shape[1]
    max_area = int(config.max_size_fraction * image_area)
    min_area = config.min_cell_area or 15
    
    for region in region_props:
        area = region.area
        if min_area <= area <= max_area:
            kept_objects += 1
            final_mask[masks == region.label] = kept_objects
            source_label_by_object[kept_objects] = int(region.label)
        else:
            excluded_mask[masks == region.label] = region.label

    # Calculate density
    pixels_per_mm2 = 1e6  # Assuming 1 pixel = 1 μm
    mask_area_mm2 = image_area / pixels_per_mm2
    objects_per_mm2 = kept_objects / mask_area_mm2 if mask_area_mm2 > 0 else 0
    
    # Save final mask
    mask_path = output_dir / f"{roi_name}.tiff"
    skio.imsave(mask_path, final_mask.astype(np.uint16))

    cell_metrics_df = None
    if collect_cell_metrics:
        try:
            cellpose_maps = _extract_cellpose_metric_maps(flows, final_mask.shape)
            cell_metrics_df = build_cellpose_cell_metrics(
                roi_name=roi_name,
                final_mask=final_mask,
                source_label_by_object=source_label_by_object,
                raw_dna_image=original_dna_img,
                preprocessed_dna_image=img,
                cellprob_map=cellpose_maps["cellprob_map"],
                flow_vector=cellpose_maps["flow_vector"],
                flow_magnitude=cellpose_maps["flow_magnitude"],
                input_image_path=image_path,
                mask_path=mask_path,
                config=config,
                diameter_for_segmentation=diameter_for_segmentation,
                actual_diameter=actual_diameter,
            )
            logging.debug(f"Extracted {len(cell_metrics_df)} per-cell metric rows for {roi_name}")
        except Exception as exc:
            logging.warning(f"Could not extract per-cell mask metrics for {roi_name}: {exc}", exc_info=True)
    
    # Create QC overlays if requested
    qc_image_path_str = None
    qc_raw_overlay_path_str = None
    
    if config.perform_qc:
        qc_overlay_dir = qc_dir / 'CellposeSAM_overlay'
        qc_raw_overlay_dir = qc_dir / 'CellposeSAM_raw_overlay'
        qc_overlay_dir.mkdir(exist_ok=True, parents=True)
        qc_raw_overlay_dir.mkdir(exist_ok=True, parents=True)
        
        # Create overlay on processed image (resized to match final masks)
        qc_image = img
        if config.run_upscale and original_shape is not None:
            # Downscale the processed image to match the final mask size
            qc_image = resize(img, original_shape, order=1, preserve_range=True, anti_aliasing=True)
        
        qc_image_array = create_qc_overlay(
            image=qc_image,
            final_masks=final_mask,
            excluded_masks=excluded_mask,
            boundary_dilation=config.qc_boundary_dilation,
            vmin=0,
            vmax_quantile=0.97,
            outline_alpha=0.8
        )
        
        qc_image_path = qc_overlay_dir / f"{roi_name}_cpsam_overlay.png"
        plt.imsave(qc_image_path, qc_image_array, dpi=config.dpi_qc_images)
        qc_image_path_str = str(qc_image_path)
        
        # Create overlay on original denoised image if available
        if original_dna_img is not None:
            qc_raw_image_array = create_qc_overlay(
                image=original_dna_img,
                final_masks=final_mask,
                excluded_masks=excluded_mask,
                boundary_dilation=config.qc_boundary_dilation,
                vmin=0,
                vmax_quantile=0.97,
                outline_alpha=0.8
            )
            
            qc_raw_image_path = qc_raw_overlay_dir / f"{roi_name}_cpsam_raw_overlay.png"
            plt.imsave(qc_raw_image_path, qc_raw_image_array, dpi=config.dpi_qc_images)
            qc_raw_overlay_path_str = str(qc_raw_image_path)
            logging.debug(f"Created raw overlay: {qc_raw_image_path}")
        else:
            logging.warning(f"No original DNA image available for raw overlay: {roi_name}")
    
    # Compile results
    result = {
        'ROI': roi_name,
        'Input_image': str(image_path),
        'Mask_output': str(mask_path),
        'Total_objects_detected': total_objects,
        'Objects_kept': kept_objects,
        'Objects_excluded': total_objects - kept_objects,
        'Objects_per_mm2': objects_per_mm2,
        'Image_shape_preprocessed': f"{preprocessed_shape[0]}x{preprocessed_shape[1]}",
        'Image_shape_original': f"{original_shape[0]}x{original_shape[1]}" if original_shape else "unknown",
        'Mask_shape_final': f"{final_mask.shape[0]}x{final_mask.shape[1]}",
        'Model_type': config.cell_pose_sam_model,
        'Diameter_used': diameter_for_segmentation,
        'Diameter_base': config.cellpose_cell_diameter,
        'CellPose_scaling_factor': 30.0 / diameter_for_segmentation,  # CellPose v4+ internal scaling
        'Upscale_ratio': config.calculated_upscale_ratio if config.run_upscale else 1.0,
        'Upscale_target_diameter': config.upscale_target_diameter if config.run_upscale else config.cellpose_cell_diameter,
        'Actual_diameter': actual_diameter,
        'CellProb_threshold': config.cellprob_threshold,
        'Flow_threshold': config.flow_threshold,
        'Min_size': min_area,
        'Max_size_fraction': config.max_size_fraction,
        'Image_normalize': config.image_normalise,
        'Expand_masks': config.expand_masks,
        'Fill_holes': config.fill_holes,
        'Remove_edge_masks': config.remove_edge_masks,
        'QC_image_path': qc_image_path_str,
        'QC_raw_overlay_path': qc_raw_overlay_path_str
    }

    if cell_metrics_df is not None:
        result['Cell_metrics_rows'] = len(cell_metrics_df)
        result['_cell_metrics'] = cell_metrics_df
    
    return result


def process_all_rois(general_config: GeneralConfig, mask_config: CreateMasksConfig):
    """
    Process all ROIs with CellPose-SAM segmentation.
    
    Parameters
    ----------
    general_config : GeneralConfig
        General configuration.
    mask_config : CreateMasksConfig
        Mask creation configuration with CellPose-SAM settings.
    """
    logging.info("Starting CellPose-SAM segmentation for all ROIs.")
    
    # Setup paths
    input_folder = Path(mask_config.dna_preprocessing_output_folder_name)  # Use existing preprocessed DNA folder
    output_folder = Path(general_config.masks_folder)    # Use standard masks folder
    qc_folder = Path(general_config.qc_folder) / 'CellposeSAM_QC'
    denoised_folder = Path(general_config.denoised_images_folder)  # For raw overlay generation
    
    # Create output directories
    output_folder.mkdir(parents=True, exist_ok=True)
    qc_folder.mkdir(parents=True, exist_ok=True)
    
    # Find available ROIs
    if mask_config.specific_rois:
        rois_to_process = mask_config.specific_rois
        logging.info(f"Processing specific ROIs: {rois_to_process}")
    else:
        # Find all .tiff files in input folder
        image_files = list(input_folder.glob("*.tiff")) + list(input_folder.glob("*.tif"))
        rois_to_process = [f.stem for f in image_files]
        logging.info(f"Found {len(rois_to_process)} ROIs in {input_folder}")
        
        if not rois_to_process:
            logging.error(f"No .tiff files found in {input_folder}")
            return
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    logging.info(f"GPU available: {gpu_available}")
    if gpu_available:
        logging.info(f"GPU device: {torch.cuda.get_device_name()}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logging.warning("GPU not available - CellPose-SAM will run on CPU which is VERY slow!")
        logging.warning("Consider using a system with CUDA support for faster processing.")
        logging.warning(f"Estimated time per ROI on CPU: ~6-8 hours (Total: ~{len(rois_to_process) * 7:.0f} hours)")
        logging.warning("RECOMMENDATION: Use 'specific_rois' to test on a small subset first!")
        
        # If running on CPU with many ROIs, suggest limiting the scope
        if len(rois_to_process) > 5:
            logging.warning(f"Processing {len(rois_to_process)} ROIs on CPU will take ~{len(rois_to_process) * 7:.0f} hours!")
            logging.warning("Consider setting 'specific_rois: [roi1, roi2, roi3]' in config for testing.")
    
    # Initialize CellPose-SAM model
    logging.info("Initializing CellPose model")
    try:
        # Determine if GPU is available and working
        use_gpu = torch.cuda.is_available()
        logging.info(f"Using GPU: {use_gpu}")
        
        cp_sam_model = load_cellpose_model(mask_config.cell_pose_sam_model, use_gpu)
        logging.info(f"CellPose model '{mask_config.cell_pose_sam_model}' loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model '{mask_config.cell_pose_sam_model}': {str(e)}")
        raise
    
    # Process each ROI
    results = []
    cell_metrics_frames = []
    successful_rois = []
    failed_rois = []
    
    for roi in tqdm(rois_to_process, desc="Segmenting ROIs"):
        try:
            # Construct image path
            image_path = input_folder / f"{roi}.tiff"
            if not image_path.exists():
                image_path = input_folder / f"{roi}.tif"
                if not image_path.exists():
                    logging.warning(f"Image file not found for ROI {roi}")
                    failed_rois.append(roi)
                    continue
            
            result = segment_single_roi(
                roi_name=roi,
                image_path=image_path,
                output_dir=output_folder,
                qc_dir=qc_folder,
                config=mask_config,
                denoised_folder=denoised_folder,
                cp_sam_model=cp_sam_model,
                collect_cell_metrics=True,
            )

            cell_metrics_df = result.pop('_cell_metrics', None)
            if cell_metrics_df is not None:
                cell_metrics_frames.append(cell_metrics_df)
            
            results.append(result)
            successful_rois.append(roi)
            
        except Exception as e:
            logging.error(f"Error processing ROI {roi}: {str(e)}", exc_info=True)
            failed_rois.append(roi)
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_path = qc_folder / 'CellposeSAM_segmentation_results.csv'
        results_df.to_csv(results_path, index=False)
        logging.info(f"Saved segmentation results to {results_path}")

        if cell_metrics_frames:
            cell_metrics_df = pd.concat(cell_metrics_frames, ignore_index=True)
            from SpatialBiologyToolkit.reporting import optional_category_output_path

            report_tables = optional_category_output_path("tables", output_folder)
            report_tables.mkdir(parents=True, exist_ok=True)
            cell_metrics_path = report_tables / 'CellposeSAM_cell_metrics.csv'
            cell_metrics_df.to_csv(cell_metrics_path, index=False)
            feature_dictionary_path = report_tables / 'CellposeSAM_cell_metrics_feature_dictionary.csv'
            feature_dictionary_df = build_cellpose_feature_dictionary(cell_metrics_df)
            feature_dictionary_df.to_csv(feature_dictionary_path, index_label='feature')
            logging.info(
                f"Saved per-cell CellPose-SAM mask metrics for {len(cell_metrics_df)} cells to {cell_metrics_path}"
            )
            logging.info(f"Saved CellPose-SAM metrics feature dictionary to {feature_dictionary_path}")
        
        # Print summary statistics
        logging.info(f"\nCellPose-SAM Segmentation Summary:")
        logging.info(f"Total ROIs processed: {len(successful_rois)}")
        logging.info(f"Failed ROIs: {len(failed_rois)}")
        if failed_rois:
            logging.warning(f"Failed ROI list: {failed_rois}")
        
        # Calculate average statistics
        if len(results) > 0:
            avg_objects = np.mean([r['Objects_kept'] for r in results])
            avg_density = np.mean([r['Objects_per_mm2'] for r in results])
            avg_diameter = np.mean([r['Actual_diameter'] for r in results])
            
            logging.info(f"Average objects per ROI: {avg_objects:.1f}")
            logging.info(f"Average density: {avg_density:.1f} objects/mm²")
            logging.info(f"Average diameter used: {avg_diameter:.1f} pixels")
            
            # Check for dimension consistency
            dimension_issues = sum(1 for r in results if r['Image_shape_original'] == 'unknown')
            if dimension_issues > 0:
                logging.warning(f"{dimension_issues} ROIs had dimension detection issues")
            
            # Count overlay creation success
            raw_overlays_created = sum(1 for r in results if r.get('QC_raw_overlay_path') is not None)
            logging.info(f"Raw overlays created: {raw_overlays_created}/{len(results)} ROIs")
    
    logging.info("CellPose-SAM segmentation completed.")


def parameter_scan_cpsam(general_config: GeneralConfig, mask_config: CreateMasksConfig):
    """
    Parameter scan mode for CellPose-SAM: run multiple parameter sets defined by two parameters
    (param_a, param_b), each with a list of values. Create summarizing plots comparing performance.
    
    Simplified version that processes all ROIs (no sampling) and saves masks/QC with parameter
    identifiers in folder names.
    
    Parameters
    ----------
    general_config : GeneralConfig
        General configuration.
    mask_config : CreateMasksConfig
        Mask creation configuration with parameter scan settings.
    """
    logging.info("Starting CellPose-SAM parameter scan.")
    
    param_a = mask_config.param_a
    param_a_values = mask_config.param_a_values or []
    param_b = mask_config.param_b
    param_b_values = mask_config.param_b_values or []
    
    if not param_a or not param_a_values or not param_b or not param_b_values:
        logging.error("Parameter scan requires param_a, param_a_values, param_b, and param_b_values to be set")
        return
    
    # Setup base paths
    input_folder = Path(mask_config.dna_preprocessing_output_folder_name)
    denoised_folder = Path(general_config.denoised_images_folder)
    base_qc_folder = Path(general_config.qc_folder) / f'CellposeSAM_ParameterScan_{cleanstring(param_a)}_{cleanstring(param_b)}'
    base_qc_folder.mkdir(parents=True, exist_ok=True)
    
    # Find available ROIs
    if mask_config.specific_rois:
        rois_to_process = mask_config.specific_rois
        logging.info(f"Parameter scanning specific ROIs: {rois_to_process}")
    else:
        # Find all .tiff files in input folder
        image_files = list(input_folder.glob("*.tiff")) + list(input_folder.glob("*.tif"))
        rois_to_process = [f.stem for f in image_files]
        logging.info(f"Parameter scanning discovered {len(rois_to_process)} ROIs in {input_folder}")
        
        if not rois_to_process:
            logging.error(f"No .tiff files found in {input_folder}")
            return

        # Optional ROI subsampling for faster parameter scans
        num_rois_to_scan = mask_config.num_rois_to_scan
        if isinstance(num_rois_to_scan, str):
            num_rois_to_scan = None if num_rois_to_scan.strip().lower() in ('none', 'null', '') else int(num_rois_to_scan)

        if num_rois_to_scan is not None:
            if num_rois_to_scan <= 0:
                logging.warning(
                    f"num_rois_to_scan={num_rois_to_scan} is <= 0; no ROIs selected for parameter scan."
                )
                return

            if num_rois_to_scan >= len(rois_to_process):
                logging.info(
                    f"num_rois_to_scan={num_rois_to_scan} >= available ROIs ({len(rois_to_process)}); processing all ROIs."
                )
            else:
                rois_to_process = random.sample(rois_to_process, num_rois_to_scan)
                logging.info(
                    f"Parameter scanning random subset of {len(rois_to_process)} ROIs: {rois_to_process}"
                )
        else:
            logging.info(f"num_rois_to_scan is None; parameter scanning all {len(rois_to_process)} ROIs")
    
    # Initialize CellPose model once
    use_gpu = torch.cuda.is_available()
    logging.info(f"Initializing CellPose model (GPU: {use_gpu})")
    cp_sam_model = load_cellpose_model(mask_config.cell_pose_sam_model, use_gpu)
    
    # Construct parameter grid
    param_sets = []
    for a_val in param_a_values:
        for b_val in param_b_values:
            param_sets.append({param_a: a_val, param_b: b_val})

    _validate_parameter_set_slugs(param_sets, param_a, param_b)
    
    logging.info(f"Running {len(param_sets)} parameter combinations on {len(rois_to_process)} ROIs")
    
    all_results = []
    
    # Run parameter scan
    for i, param_set in enumerate(param_sets):
        logging.info(f"Parameter set {i+1}/{len(param_sets)}: {param_set}")
        
        # Create output folders with parameter identifiers
        param_string = _parameter_set_slug(param_set, param_a, param_b)
        
        # Create temporary config with current parameters
        temp_config = CreateMasksConfig(**mask_config.__dict__)
        setattr(temp_config, param_a, param_set[param_a])
        setattr(temp_config, param_b, param_set[param_b])
        
        # Check if we need to reinitialize the model (if model type is being scanned)
        current_cp_sam_model = cp_sam_model
        if param_a == 'cell_pose_sam_model' or param_b == 'cell_pose_sam_model':
            try:
                current_cp_sam_model = load_cellpose_model(temp_config.cell_pose_sam_model, use_gpu)
                logging.info(f"Initialized model '{temp_config.cell_pose_sam_model}' for parameter scan")
            except Exception as e:
                logging.error(f"Failed to initialize model '{temp_config.cell_pose_sam_model}': {str(e)}")
                continue
        
        # Setup output folders for this parameter set
        param_masks_folder = Path(general_config.masks_folder) / f'param_{param_string}'
        param_qc_folder = base_qc_folder / f'param_{param_string}'
        param_masks_folder.mkdir(parents=True, exist_ok=True)
        param_qc_folder.mkdir(parents=True, exist_ok=True)
        
        # Process each ROI with current parameter set
        param_results = []
        for roi in tqdm(rois_to_process, desc=f"Param set {i+1}"):
            try:
                # Construct image path
                image_path = input_folder / f"{roi}.tiff"
                if not image_path.exists():
                    image_path = input_folder / f"{roi}.tif"
                    if not image_path.exists():
                        logging.warning(f"Image file not found for ROI {roi}")
                        continue
                
                result = segment_single_roi(
                    roi_name=roi,
                    image_path=image_path,
                    output_dir=param_masks_folder,
                    qc_dir=param_qc_folder,
                    config=temp_config,
                    denoised_folder=denoised_folder,
                    cp_sam_model=current_cp_sam_model
                )
                
                # Add parameter information to result
                result[f'{param_a}'] = param_set[param_a]
                result[f'{param_b}'] = param_set[param_b]
                result['Parameter_set'] = param_string
                
                param_results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing ROI {roi} with params {param_set}: {str(e)}", exc_info=True)
                continue
        
        # Save results for this parameter set
        if param_results:
            param_df = pd.DataFrame(param_results)
            param_csv_path = param_qc_folder / f'CellposeSAM_results_{param_string}.csv'
            param_df.to_csv(param_csv_path, index=False)
            logging.info(f"Saved {len(param_results)} results for parameter set {param_string}")
            
            all_results.extend(param_results)
    
    # Save combined results and create summary plots
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_csv_path = base_qc_folder / 'CellposeSAM_ParameterScan_All.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        logging.info(f"Saved combined parameter scan results to {combined_csv_path}")
        
        # Create summary plots
        create_parameter_scan_plots(combined_df, base_qc_folder, param_a, param_b, mask_config.dpi_qc_images)
        
        # Print summary statistics
        logging.info(f"\nCellPose-SAM Parameter Scan Summary:")
        logging.info(f"Total parameter combinations: {len(param_sets)}")
        logging.info(f"Total ROIs processed: {len(set(r['ROI'] for r in all_results))}")
        logging.info(f"Total segmentations: {len(all_results)}")
        
        # Calculate average statistics by parameter set
        summary_stats = combined_df.groupby(['Parameter_set']).agg({
            'Objects_kept': 'mean',
            'Objects_per_mm2': 'mean',
            'Objects_excluded': 'mean'
        }).round(2)
        
        logging.info("\nAverage statistics by parameter set:")
        for param_set, stats in summary_stats.iterrows():
            logging.info(f"{param_set}: Kept={stats['Objects_kept']:.1f}, "
                        f"Density={stats['Objects_per_mm2']:.1f}/mm², "
                        f"Excluded={stats['Objects_excluded']:.1f}")
    
    logging.info("CellPose-SAM parameter scan completed.")


def create_parameter_scan_plots(df: pd.DataFrame, output_dir: Path, param_a: str, param_b: str, dpi: int = 300):
    """
    Create summary plots for parameter scan results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined results dataframe.
    output_dir : Path
        Output directory for plots.
    param_a : str
        Name of first parameter.
    param_b : str
        Name of second parameter.
    dpi : int
        DPI for saved plots.
    """
    logging.info("Creating parameter scan summary plots")
    
    # Map parameter names to column names (maintain compatibility with original script)
    param_to_column = {
        'cellpose_cell_diameter': 'Diameter_base',
        'cellprob_threshold': 'CellProb_threshold', 
        'flow_threshold': 'Flow_threshold',
        'max_size_fraction': 'Max_size_fraction',
        'min_cell_area': 'Min_size',
        'expand_masks': 'Expand_masks',
        'batch_size': 'batch_size',  # May not be in results, use parameter name
        'cell_pose_sam_model': 'Model_type',
    }
    
    # Use column name if available, otherwise use parameter name directly
    param_a_col = param_to_column.get(param_a, param_a)
    param_b_col = param_to_column.get(param_b, param_b)
    
    # Check if columns exist in dataframe, if not use parameter value directly
    if param_a_col not in df.columns:
        param_a_col = param_a
    if param_b_col not in df.columns:
        param_b_col = param_b
    
    # Metrics to plot
    metrics = ['Objects_kept', 'Objects_per_mm2', 'Objects_excluded']
    metric_labels = ['Objects Kept', 'Objects per mm²', 'Objects Excluded']
    
    for metric, label in zip(metrics, metric_labels):
        try:
            plt.figure(figsize=(12, 8))
            
            # Create barplot
            ax = sb.barplot(
                data=df,
                y=metric,
                x=param_a_col,
                hue=param_b_col,
                palette='tab20',
                ci='sd',  # Show standard deviation
                capsize=0.1
            )
            
            # Customize plot
            ax.set_title(f'CellPose-SAM Parameter Scan: {label}', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{param_a}', fontsize=12)
            ax.set_ylabel(f'{label}', fontsize=12)
            
            # Move legend outside plot area
            sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            
            # Save plot
            plot_path = output_dir / f"ParameterScan_{cleanstring(metric)}.png"
            plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
            plt.close()
            
            logging.info(f"Saved parameter scan plot: {plot_path}")
            
        except Exception as e:
            logging.error(f"Error creating plot for {metric}: {str(e)}")
            plt.close()
            continue
    
    # Create heatmap for Objects_kept
    try:
        plt.figure(figsize=(10, 8))
        
        # Pivot data for heatmap
        pivot_data = df.groupby([param_a_col, param_b_col])['Objects_kept'].mean().unstack()
        
        # Create heatmap
        sb.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='viridis',
            cbar_kws={'label': 'Average Objects Kept'}
        )
        
        plt.title('CellPose-SAM Parameter Scan: Objects Kept Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel(f'{param_b}', fontsize=12)
        plt.ylabel(f'{param_a}', fontsize=12)
        
        # Save heatmap
        heatmap_path = output_dir / f"ParameterScan_Objects_Kept_Heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        
        logging.info(f"Saved parameter scan heatmap: {heatmap_path}")
        
    except Exception as e:
        logging.error(f"Error creating heatmap: {str(e)}")
        plt.close()


def print_model_info():
    """Print available CellPose models and system information."""
    logging.info("System Information:")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU device: {torch.cuda.get_device_name()}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        from cellpose import models
        logging.info("Available CellPose models:")
        for model_name in models.MODEL_NAMES:
            logging.info(f"  - {model_name}")
    except Exception as e:
        logging.warning(f"Could not retrieve model list: {e}")


if __name__ == "__main__":
    # Define pipeline stage
    pipeline_stage = 'CellposeSAM'
    
    # Load configuration
    config_data = process_config_with_overrides()
    
    # Setup logging
    setup_logging(config_data.get('logging', {}), pipeline_stage)
    
    # Print system and model info
    print_model_info()
    
    # Get configuration objects
    general_config = GeneralConfig(**filter_config_for_dataclass(config_data.get('general', {}), GeneralConfig))
    mask_config = CreateMasksConfig(**filter_config_for_dataclass(config_data.get('createmasks', {}), CreateMasksConfig))
    
    logging.info(f"CellPose-SAM configuration: {mask_config}")
    
    # Decide mode based on run_parameter_scan and param fields
    if (mask_config.run_parameter_scan and
        mask_config.param_a and mask_config.param_a_values and
        mask_config.param_b and mask_config.param_b_values):
        parameter_scan_cpsam(general_config, mask_config)
    else:
        # Run normal segmentation
        process_all_rois(general_config, mask_config)
