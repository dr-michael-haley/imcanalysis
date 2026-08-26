"""Pure image-normalization helpers for :mod:`napari_imc_explorer`."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from SpatialBiologyToolkit.nimbus_normalization import (
    NimbusNormalizationParameters,
    load_normalization_file,
    normalize_nimbus_image,
    validate_normalization_parameters,
)


def prepare_normalization_parameters(
    normalization_dict: Mapping[str, Any] | None,
) -> dict[str, NimbusNormalizationParameters]:
    """Validate scalar legacy maxima or structured Nimbus channel bounds."""

    if normalization_dict is None:
        return {}
    if not isinstance(normalization_dict, Mapping):
        raise TypeError("normalization_dict must be a mapping or None.")
    if not normalization_dict:
        return {}
    payload = {
        marker: (
            {
                "vmax": value.vmax,
                "lower_threshold": value.lower_threshold,
            }
            if isinstance(value, NimbusNormalizationParameters)
            else value
        )
        for marker, value in normalization_dict.items()
    }
    return validate_normalization_parameters(payload)


def normalization_parameters_payload(
    parameters: Mapping[str, NimbusNormalizationParameters],
) -> dict[str, dict[str, float]]:
    """Return the JSON-safe marker mapping used by current Napari workspaces."""

    prepared = prepare_normalization_parameters(parameters)
    return {
        marker: {
            "vmax": float(entry.vmax),
            "lower_threshold": float(entry.lower_threshold),
        }
        for marker, entry in prepared.items()
    }


def prepare_normalization_dict(
    normalization_dict: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Return the legacy maxima-only view of Nimbus normalization data."""

    return {
        marker: entry.vmax
        for marker, entry in prepare_normalization_parameters(
            normalization_dict
        ).items()
    }


def load_normalization_parameters(
    path: str | Path,
) -> dict[str, NimbusNormalizationParameters]:
    """Load full Nimbus Vmax and lower-threshold parameters from JSON or CSV."""

    return load_normalization_file(Path(path).expanduser())


def load_normalization_mapping(path: str | Path) -> dict[str, float]:
    """Load the legacy maxima-only view of a Nimbus JSON or CSV file."""

    return {
        marker: entry.vmax
        for marker, entry in load_normalization_parameters(path).items()
    }


def find_normalization_parameters(
    normalization_dict: Mapping[str, Any],
    image_name: str,
) -> NimbusNormalizationParameters | None:
    """Find an image's full Nimbus bounds with legacy name normalization."""

    image_name = str(image_name).strip()
    if not image_name or not normalization_dict:
        return None

    folded_name = image_name.casefold()
    exact_matches = [
        (str(key), value)
        for key, value in normalization_dict.items()
        if str(key).casefold() == folded_name
    ]
    if len(exact_matches) == 1:
        key, value = exact_matches[0]
        return prepare_normalization_parameters({key: value})[key]

    clean_name = re.sub(r"\W+", "", image_name).casefold()
    clean_matches = [
        (str(key), value)
        for key, value in normalization_dict.items()
        if re.sub(r"\W+", "", str(key)).casefold() == clean_name
    ]
    if len(clean_matches) == 1:
        key, value = clean_matches[0]
        return prepare_normalization_parameters({key: value})[key]
    return None


def find_normalization_value(
    normalization_dict: Mapping[str, Any],
    image_name: str,
) -> float | None:
    """Find an image's Nimbus maximum using the backwards-compatible API."""

    parameters = find_normalization_parameters(normalization_dict, image_name)
    return None if parameters is None else float(parameters.vmax)


def normalize_imc_image(
    image: np.ndarray,
    *,
    quantile: float,
    minimum_pixel_counts: float,
    normalization_value: float | None = None,
    normalization_lower_threshold: float = 0.0,
) -> np.ndarray:
    """Apply Nimbus normalization when available, otherwise the legacy quantile path."""
    thresholded = np.where(np.asarray(image) > minimum_pixel_counts, image, 0)
    if normalization_value is None:
        maximum = float(np.quantile(thresholded, quantile))
        if maximum < 5:
            maximum = 3.0
        return np.clip(thresholded / maximum, 0, 1)
    return normalize_nimbus_image(
        thresholded,
        vmax=float(normalization_value),
        lower_threshold=float(normalization_lower_threshold),
        upper_clip=1.0,
    )


def normalized_contrast_limits(
    lower_limit: float,
    upper_limit: float,
) -> tuple[float, float]:
    """Return valid Napari limits for a normalized image display range."""
    lower_limit = float(lower_limit)
    upper_limit = float(upper_limit)
    if (
        not np.isfinite(lower_limit)
        or not np.isfinite(upper_limit)
        or not 0 <= lower_limit <= 1
        or not 0 <= upper_limit <= 1
    ):
        raise ValueError(
            "Normalized image contrast limits must both be between 0 and 1; "
            f"got {(lower_limit, upper_limit)!r}."
        )
    if lower_limit >= upper_limit:
        raise ValueError(
            "Normalized image lower contrast limit must be below the upper "
            f"limit; got {(lower_limit, upper_limit)!r}."
        )
    return lower_limit, upper_limit
