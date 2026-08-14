"""Pure image-normalization helpers for :mod:`napari_imc_explorer`."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def prepare_normalization_dict(
    normalization_dict: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Validate a Nimbus-format channel-to-maximum mapping."""
    if normalization_dict is None:
        return {}
    if not isinstance(normalization_dict, Mapping):
        raise TypeError("normalization_dict must be a mapping or None.")

    prepared: dict[str, float] = {}
    seen_keys: set[str] = set()
    for raw_key, raw_value in normalization_dict.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("normalization_dict contains an empty channel name.")

        folded_key = key.casefold()
        if folded_key in seen_keys:
            raise ValueError(
                "normalization_dict channel names must be unique ignoring case; "
                f"found a duplicate for {key!r}."
            )
        seen_keys.add(folded_key)

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Normalization value for channel {key!r} is not numeric: {raw_value!r}."
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"Normalization value for channel {key!r} must be finite and positive; "
                f"got {raw_value!r}."
            )
        prepared[key] = value
    return prepared


def load_normalization_mapping(path: str | Path) -> dict[str, float]:
    """Load and validate a Nimbus normalization mapping from JSON or CSV."""
    from SpatialBiologyToolkit.nimbus_normalization import load_normalization_file

    parameters = load_normalization_file(Path(path).expanduser())
    return prepare_normalization_dict(
        {marker: entry.vmax for marker, entry in parameters.items()}
    )


def find_normalization_value(
    normalization_dict: Mapping[str, float],
    image_name: str,
) -> float | None:
    """Find an image's Nimbus maximum, allowing case and punctuation differences."""
    image_name = str(image_name).strip()
    if not image_name or not normalization_dict:
        return None

    folded_name = image_name.casefold()
    exact_matches = [
        value
        for key, value in normalization_dict.items()
        if str(key).casefold() == folded_name
    ]
    if len(exact_matches) == 1:
        return float(exact_matches[0])

    clean_name = re.sub(r"\W+", "", image_name).casefold()
    clean_matches = [
        value
        for key, value in normalization_dict.items()
        if re.sub(r"\W+", "", str(key)).casefold() == clean_name
    ]
    if len(clean_matches) == 1:
        return float(clean_matches[0])
    return None


def normalize_imc_image(
    image: np.ndarray,
    *,
    quantile: float,
    minimum_pixel_counts: float,
    normalization_value: float | None = None,
) -> np.ndarray:
    """Apply Nimbus normalization when available, otherwise the legacy quantile path."""
    thresholded = np.where(np.asarray(image) > minimum_pixel_counts, image, 0)
    if normalization_value is None:
        maximum = float(np.quantile(thresholded, quantile))
        if maximum < 5:
            maximum = 3.0
    else:
        maximum = float(normalization_value)
    return np.clip(thresholded / maximum, 0, 1)


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
