"""Feature-family catalog shared by extraction, manifests, and the GUI."""

from __future__ import annotations

DISTRIBUTION_FEATURE_DESCRIPTIONS = {
    "pixel_count": "Number of finite pixels contributing to this measurement.",
    "mean": "Mean pixel intensity inside the measurement region.",
    "median": "Median pixel intensity inside the measurement region.",
    "std": "Standard deviation of pixel intensities.",
    "min": "Minimum finite pixel intensity.",
    "max": "Maximum finite pixel intensity.",
    "sum": "Sum of finite pixel intensities.",
    "q05": "5th percentile pixel intensity.",
    "q10": "10th percentile pixel intensity.",
    "q25": "25th percentile pixel intensity.",
    "q75": "75th percentile pixel intensity.",
    "q90": "90th percentile pixel intensity.",
    "q95": "95th percentile pixel intensity.",
    "iqr": "Interquartile range (75th minus 25th percentile).",
    "range": "Maximum minus minimum pixel intensity.",
    "cv": "Coefficient of variation: standard deviation divided by absolute mean.",
}

REGION_IMAGE_FEATURE_DESCRIPTIONS = {
    "core_mean": "Mean intensity in the one-pixel-eroded cell core.",
    "border_mean": "Mean intensity on the one-pixel cell border.",
    "core_to_border_ratio": "Core mean divided by border mean.",
    "weighted_x": "X coordinate of the intensity-weighted centroid.",
    "weighted_y": "Y coordinate of the intensity-weighted centroid.",
    "weighted_centroid_offset_px": "Distance from geometric to intensity-weighted centroid.",
    "weighted_centroid_offset_fraction_radius": "Centroid offset divided by equivalent cell radius.",
    "local_bg_pixel_count": "Number of finite background-ring pixels.",
    "local_bg_mean": "Mean intensity in the local background ring.",
    "local_bg_std": "Background-ring intensity standard deviation.",
    "foreground_to_bg_ratio": "Cell mean divided by local background mean.",
    "foreground_bg_contrast": "Cell mean minus local background mean.",
    "foreground_bg_contrast_z": "Cell/background contrast divided by background standard deviation.",
}

SHAPE_FEATURE_DESCRIPTIONS = {
    "bbox_min_row": "Top row of the cell bounding box.",
    "bbox_min_col": "Left column of the cell bounding box.",
    "bbox_max_row": "Bottom row of the cell bounding box.",
    "bbox_max_col": "Right column of the cell bounding box.",
    "bbox_width": "Bounding-box width in pixels.",
    "bbox_height": "Bounding-box height in pixels.",
    "bbox_area": "Bounding-box area in pixels.",
    "mask_area": "Original segmented-cell area in pixels.",
    "mask_area_fraction_roi": "Cell area divided by ROI image area.",
    "mask_area_fraction_bbox": "Cell area divided by bounding-box area.",
    "mask_perimeter": "Original-mask perimeter.",
    "mask_perimeter_crofton": "Crofton perimeter estimate.",
    "mask_circularity": "Four pi times area divided by perimeter squared.",
    "mask_compactness": "Perimeter squared divided by four pi times area.",
    "mask_major_axis_length": "Major-axis length of the fitted ellipse.",
    "mask_minor_axis_length": "Minor-axis length of the fitted ellipse.",
    "mask_axis_ratio": "Major-axis divided by minor-axis length.",
    "mask_eccentricity": "Eccentricity of the fitted ellipse.",
    "mask_solidity": "Area divided by convex-hull area.",
    "mask_extent": "Area divided by bounding-box area.",
    "mask_orientation_degrees": "Major-axis orientation in degrees.",
    "mask_equivalent_diameter": "Diameter of a circle with the same area.",
    "mask_feret_diameter_max": "Maximum Feret diameter.",
    "mask_convex_area": "Area of the convex hull.",
    "mask_filled_area": "Area after filling internal holes.",
    "mask_hole_area": "Filled area minus original area.",
    "mask_hole_fraction": "Hole area divided by filled area.",
    "mask_convexity": "Original area divided by convex-hull area.",
    "mask_euler_number": "Objects minus holes in the cell mask.",
    "mask_edge_touching": "Whether the cell touches an ROI edge.",
    "mask_min_distance_to_edge_px": "Minimum distance from bounding box to an ROI edge.",
}

CONTEXT_FEATURE_DESCRIPTIONS = {
    "mask_neighbor_count_5px": "Number of full-mask neighbours within five pixels.",
    "mask_touching_neighbor_count_1px": "Number of full-mask neighbours touching within one pixel.",
    "nearest_centroid_distance_px": "Distance to the nearest full-mask cell centroid.",
    "centroid_neighbor_count_25px": "Full-mask centroid neighbours within 25 pixels.",
    "centroid_neighbor_count_50px": "Full-mask centroid neighbours within 50 pixels.",
    "roi_total_object_count": "Total segmented objects in the full ROI mask.",
    "roi_eligible_object_count": "Eligible cohort objects in the ROI.",
    "roi_full_mask_area_fraction": "Fraction of ROI pixels occupied by any segmented object.",
}

ROI_RANK_FEATURE_DESCRIPTIONS = {
    "zscore": "Within-eligible-cohort ROI z-score for selected base features.",
    "percentile": "Within-eligible-cohort ROI percentile rank for selected base features.",
}

FEATURE_FAMILY_DESCRIPTIONS = {
    "distribution": (
        "Per-channel summaries inside the signed-offset measurement mask. "
        "These are usually the fastest intensity features."
    ),
    "region": (
        "Per-channel core, border, intensity-centroid, local-background, and "
        "foreground/background contrast features. These are more expensive."
    ),
    "gradient": (
        "Distribution summaries of per-channel spatial gradient magnitude, "
        "capturing edges and within-cell texture."
    ),
    "shape": (
        "Channel-independent morphology and position measured from the original "
        "segmentation, regardless of the intensity-mask offset."
    ),
    "context": (
        "Full-segmentation neighbour, edge, density, and ROI context. Excluded "
        "cohort cells still contribute as tissue neighbours."
    ),
    "roi_rank": (
        "Eligible-cell-relative z-scores and percentile ranks calculated within "
        "each ROI for selected base features."
    ),
}

FEATURE_FAMILY_CATALOG = {
    "distribution": DISTRIBUTION_FEATURE_DESCRIPTIONS,
    "region": REGION_IMAGE_FEATURE_DESCRIPTIONS,
    "gradient": DISTRIBUTION_FEATURE_DESCRIPTIONS,
    "shape": SHAPE_FEATURE_DESCRIPTIONS,
    "context": CONTEXT_FEATURE_DESCRIPTIONS,
    "roi_rank": ROI_RANK_FEATURE_DESCRIPTIONS,
}

__all__ = [
    "CONTEXT_FEATURE_DESCRIPTIONS",
    "DISTRIBUTION_FEATURE_DESCRIPTIONS",
    "FEATURE_FAMILY_CATALOG",
    "FEATURE_FAMILY_DESCRIPTIONS",
    "REGION_IMAGE_FEATURE_DESCRIPTIONS",
    "ROI_RANK_FEATURE_DESCRIPTIONS",
    "SHAPE_FEATURE_DESCRIPTIONS",
]
