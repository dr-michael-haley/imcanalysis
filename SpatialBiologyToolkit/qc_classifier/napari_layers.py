"""Napari layer construction helpers for CellPose mask QC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from napari.utils import DirectLabelColormap
from napari.utils.colormaps import Colormap

from .labels import CONFIRMED_ARTIFACT, CONFIRMED_GOOD, FLAGGED_ARTIFACT, FLAGGED_GOOD, RoiLabels

LAYER_ALL_CELLS = "all_cells"
LAYER_CONFIRMED_GOOD = "confirmed_good_cells"
LAYER_CONFIRMED_ARTIFACT = "confirmed_artifact_cells"
LAYER_FLAGGED_GOOD = "flagged_good_cells"
LAYER_FLAGGED_ARTIFACT = "flagged_artifact_cells"
LAYER_SCORE = "artifact_probability"

STATE_LAYER_NAMES = {
    CONFIRMED_GOOD: LAYER_CONFIRMED_GOOD,
    CONFIRMED_ARTIFACT: LAYER_CONFIRMED_ARTIFACT,
    FLAGGED_GOOD: LAYER_FLAGGED_GOOD,
    FLAGGED_ARTIFACT: LAYER_FLAGGED_ARTIFACT,
}

STATE_COLOURS = {
    CONFIRMED_GOOD: "#006400",
    CONFIRMED_ARTIFACT: "#7f0000",
    FLAGGED_GOOD: "#00ff40",
    FLAGGED_ARTIFACT: "#ff2020",
}

DEFAULT_STATE_CONTOURS = {
    CONFIRMED_GOOD: 0,
    CONFIRMED_ARTIFACT: 0,
    FLAGGED_GOOD: 2,
    FLAGGED_ARTIFACT: 2,
}


def _has_layer(viewer, name: str) -> bool:
    return name in [layer.name for layer in viewer.layers]


def score_colormap() -> Colormap:
    """Green-to-yellow-to-red artifact-probability colormap."""

    return Colormap(
        colors=[
            [0.0, 0.35, 0.0, 0.0],
            [0.0, 0.65, 0.0, 0.65],
            [0.95, 0.85, 0.0, 0.75],
            [0.85, 0.0, 0.0, 0.85],
        ],
        controls=[0.0, 0.05, 0.5, 1.0],
        name="artifact_probability_green_red",
    )


def ids_to_binary_mask(mask: np.ndarray, object_ids: set[int]) -> np.ndarray:
    """Return a uint8 mask marking selected object IDs."""

    if not object_ids:
        return np.zeros(mask.shape, dtype=np.uint8)
    return np.isin(mask, np.asarray(sorted(object_ids), dtype=np.int64)).astype(np.uint8)


def score_map_from_scores(mask: np.ndarray, scores: pd.DataFrame) -> np.ndarray:
    """Map object-level artifact probabilities back into image space."""

    score_map = np.zeros(mask.shape, dtype=np.float32)
    if scores is None or scores.empty:
        return score_map
    if "ObjectNumber" not in scores.columns or "artifact_probability" not in scores.columns:
        return score_map

    max_label = int(np.nanmax(mask)) if mask.size else 0
    if max_label <= 0:
        return score_map

    lookup = np.zeros(max_label + 1, dtype=np.float32)
    valid_scores = scores.loc[:, ["ObjectNumber", "artifact_probability"]].dropna()
    for _, row in valid_scores.iterrows():
        object_id = int(row["ObjectNumber"])
        if 0 <= object_id <= max_label:
            lookup[object_id] = float(row["artifact_probability"])
    return lookup[np.asarray(mask, dtype=np.int64)]


def _add_or_update_labels_layer(
    viewer,
    name: str,
    data: np.ndarray,
    color: str,
    *,
    visible: bool | None = True,
    contour: int = 1,
):
    colormap = DirectLabelColormap(color_dict={None: "transparent", 0: "transparent", 1: color})
    if _has_layer(viewer, name):
        layer = viewer.layers[name]
        layer.data = data
        layer.colormap = colormap
        if visible is not None:
            layer.visible = bool(visible)
    else:
        layer = viewer.add_labels(data, name=name, colormap=colormap, visible=True if visible is None else bool(visible))
    layer.contour = int(contour)
    layer.opacity = 1.0
    return layer


def add_or_update_base_mask(viewer, mask: np.ndarray, *, visible: bool = True):
    """Add/update the original CellPose mask layer."""

    if _has_layer(viewer, LAYER_ALL_CELLS):
        layer = viewer.layers[LAYER_ALL_CELLS]
        layer.data = mask
        layer.visible = visible
    else:
        layer = viewer.add_labels(mask, name=LAYER_ALL_CELLS, visible=visible)
    layer.contour = 1
    layer.opacity = 0.7
    return layer


def add_or_update_label_state_layers(
    viewer,
    mask: np.ndarray,
    labels: RoiLabels,
    *,
    contours: dict[str, int] | None = None,
    visibilities: dict[str, bool] | None = None,
) -> None:
    """Refresh confirmed/candidate label-state layers."""

    contours = DEFAULT_STATE_CONTOURS if contours is None else contours
    for state, layer_name in STATE_LAYER_NAMES.items():
        _add_or_update_labels_layer(
            viewer,
            layer_name,
            ids_to_binary_mask(mask, labels.ids(state)),
            STATE_COLOURS[state],
            visible=None if visibilities is None else visibilities.get(state, True),
            contour=contours.get(state, DEFAULT_STATE_CONTOURS[state]),
        )


def add_or_update_score_layer(viewer, mask: np.ndarray, scores: pd.DataFrame, *, visible: bool = True):
    """Add/update the continuous classifier score image layer."""

    data = score_map_from_scores(mask, scores)
    if _has_layer(viewer, LAYER_SCORE):
        layer = viewer.layers[LAYER_SCORE]
        layer.data = data
        layer.visible = visible
    else:
        layer = viewer.add_image(
            data,
            name=LAYER_SCORE,
            blending="additive",
            opacity=0.55,
            colormap=score_colormap(),
            contrast_limits=(0.0, 1.0),
            visible=visible,
        )
    return layer
