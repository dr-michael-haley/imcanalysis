"""Pure helpers for population-focused image review in NapariSBT."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .explore import ExploreViewRecipe

POPULATION_QC_SETTINGS_COLUMNS = [
    "Red",
    "Green",
    "Blue",
    "Red_min",
    "Red_max",
    "Green_min",
    "Green_max",
    "Blue_min",
    "Blue_max",
]


def rank_population_rois(
    adata,
    *,
    observation: str,
    population: str,
    roi_obs: str,
    eligible_rois: Iterable[str],
    ordering: Literal["top", "bottom", "random"] = "top",
    limit: int | None = 10,
    random_seed: int = 0,
) -> list[tuple[str, int]]:
    """Order eligible ROIs by population abundance, including zero-count ROIs."""

    if observation not in adata.obs:
        raise KeyError(f"AnnData observation does not exist: {observation}")
    if roi_obs not in adata.obs:
        raise KeyError(f"AnnData ROI observation does not exist: {roi_obs}")
    rois = list(dict.fromkeys(str(roi) for roi in eligible_rois))
    selected = adata.obs[observation].astype("string").eq(str(population)).fillna(False)
    counts = (
        adata.obs.loc[selected, roi_obs]
        .astype("string")
        .value_counts()
        .reindex(rois, fill_value=0)
        .astype(int)
    )
    if ordering == "top":
        ordered = sorted(rois, key=lambda roi: (-int(counts[roi]), roi))
    elif ordering == "bottom":
        ordered = sorted(rois, key=lambda roi: (int(counts[roi]), roi))
    elif ordering == "random":
        ordered = list(rois)
        np.random.default_rng(int(random_seed)).shuffle(ordered)
    else:
        raise ValueError(f"Unknown Population QC ROI ordering: {ordering!r}")
    if limit is not None:
        ordered = ordered[: max(0, int(limit))]
    return [(roi, int(counts[roi])) for roi in ordered]


def top_population_markers(
    adata,
    *,
    observation: str,
    population: str,
    candidates: Sequence[tuple[str, str]],
    top_n: int = 3,
) -> list[str]:
    """Return display-channel names with the highest population mean expression."""

    if observation not in adata.obs:
        raise KeyError(f"AnnData observation does not exist: {observation}")
    selected = (
        adata.obs[observation]
        .astype("string")
        .eq(str(population))
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    if not bool(selected.any()) or not candidates:
        return []
    var_names = pd.Index(adata.var_names.astype(str))
    positions = var_names.get_indexer([var_name for _display, var_name in candidates])
    valid = positions >= 0
    if not bool(valid.any()):
        return []
    valid_candidates = [
        candidate for candidate, keep in zip(candidates, valid, strict=True) if keep
    ]
    matrix = adata.X[selected, :][:, positions[valid]]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    means = np.asarray(matrix, dtype=float).mean(axis=0).ravel()
    ranked = sorted(
        zip(valid_candidates, means, strict=True),
        key=lambda item: (-float(item[1]), item[0][0]),
    )
    return [display for (display, _var), _mean in ranked[: int(top_n)]]


def build_population_qc_recipe(
    *,
    observation: str,
    population: str,
    channels: Sequence[str],
    contrast_limits: Sequence[tuple[float, float]],
) -> ExploreViewRecipe:
    """Build an ordinary Explore recipe from simplified RGB controls."""

    cleaned_channels = [
        str(channel).strip() for channel in channels if str(channel).strip()
    ]
    if not cleaned_channels:
        raise ValueError("Choose at least one Population QC image channel.")
    if len(cleaned_channels) != len(set(cleaned_channels)):
        raise ValueError("Population QC RGB channels must be distinct.")
    if len(cleaned_channels) > 3:
        raise ValueError("Population QC supports at most three RGB channels.")
    if len(contrast_limits) != len(cleaned_channels):
        raise ValueError("Every Population QC channel requires one contrast range.")
    colours = ("red", "green", "blue")
    layer_colormaps: dict[str, str] = {}
    layer_contrast_limits: dict[str, tuple[float, float]] = {}
    layer_visibility: dict[str, bool] = {}
    layer_opacities: dict[str, float] = {}
    for channel, colour, limits in zip(
        cleaned_channels,
        colours,
        contrast_limits,
        strict=True,
    ):
        lower, upper = (float(value) for value in limits)
        if not 0 <= lower < upper <= 1:
            raise ValueError(
                f"Contrast limits for {channel!r} must satisfy "
                f"0 <= lower < upper <= 1; got {(lower, upper)!r}."
            )
        name = f"image::{channel}"
        layer_colormaps[name] = colour
        layer_contrast_limits[name] = (lower, upper)
        layer_visibility[name] = True
        layer_opacities[name] = 1.0
    population_layer = f"population::{observation}::{population}"
    layer_visibility[population_layer] = True
    layer_opacities[population_layer] = 1.0
    return ExploreViewRecipe(
        image_mode="six_colour",
        image_channels=cleaned_channels,
        population_observation=str(observation),
        populations=[str(population)],
        layer_colormaps=layer_colormaps,
        layer_visibility=layer_visibility,
        layer_opacities=layer_opacities,
        layer_contours={population_layer: 2},
        layer_contrast_limits=layer_contrast_limits,
    )


def parse_legacy_contrast(value, *, fallback: float) -> float:
    """Interpret legacy numeric or quantile-style Population QC display values."""

    text = "" if value is None else str(value).strip().casefold()
    if not text or text == "nan":
        return float(fallback)
    if text.startswith("q"):
        quantile = float(text[1:])
        if not 0 < quantile <= 1:
            raise ValueError(f"Invalid legacy quantile contrast value: {value!r}")
        return 1.0
    result = float(text)
    if not 0 <= result <= 1:
        raise ValueError(f"Contrast values must be between 0 and 1: {value!r}")
    return result


__all__ = [
    "POPULATION_QC_SETTINGS_COLUMNS",
    "build_population_qc_recipe",
    "parse_legacy_contrast",
    "rank_population_rois",
    "top_population_markers",
]
