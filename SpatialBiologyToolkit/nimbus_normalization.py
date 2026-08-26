"""Shared Nimbus normalization-table I/O and intensity transformation helpers."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PREFERRED_NORMALIZATION_FILENAME = "normalization_dict.csv"
LEGACY_NORMALIZATION_FILENAME = "normalization_dict.json"
PREFERRED_COLUMNS = ("marker", "vmax", "lower_threshold")


@dataclass(frozen=True)
class NimbusNormalizationParameters:
    """Validated upper and lower absolute intensity bounds for one marker."""

    vmax: float
    lower_threshold: float = 0.0


def _normalized_field_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _coerce_parameters(
    marker: str,
    raw_value: Any,
) -> NimbusNormalizationParameters:
    if isinstance(raw_value, Mapping):
        normalized = {
            _normalized_field_name(key): value for key, value in raw_value.items()
        }
        vmax_raw = normalized.get("vmax", normalized.get("value"))
        if vmax_raw is None:
            raise ValueError(
                f"Normalization entry for marker {marker!r} must contain vmax."
            )
        lower_raw = normalized.get(
            "lower_threshold", normalized.get("lower", normalized.get("vmin", 0.0))
        )
    else:
        vmax_raw = raw_value
        lower_raw = 0.0

    try:
        vmax = float(vmax_raw)
        lower_threshold = float(0.0 if lower_raw in (None, "") else lower_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Normalization bounds for marker {marker!r} must be numeric; "
            f"got vmax={vmax_raw!r}, lower_threshold={lower_raw!r}."
        ) from exc
    if not math.isfinite(vmax) or vmax <= 0:
        raise ValueError(
            f"Normalization vmax for marker {marker!r} must be finite and positive; "
            f"got {vmax_raw!r}."
        )
    if not math.isfinite(lower_threshold) or lower_threshold < 0:
        raise ValueError(
            f"Normalization lower_threshold for marker {marker!r} must be finite "
            f"and non-negative; got {lower_raw!r}."
        )
    if lower_threshold >= vmax:
        raise ValueError(
            f"Normalization lower_threshold for marker {marker!r} must be below "
            f"vmax; got lower_threshold={lower_threshold:g}, vmax={vmax:g}."
        )
    return NimbusNormalizationParameters(
        vmax=vmax,
        lower_threshold=lower_threshold,
    )


def validate_normalization_parameters(
    values: Mapping[str, object],
) -> dict[str, NimbusNormalizationParameters]:
    """Validate a marker mapping containing scalar or structured bounds."""

    if not isinstance(values, Mapping):
        raise TypeError("Nimbus normalization data must be a marker mapping.")
    validated: dict[str, NimbusNormalizationParameters] = {}
    folded_keys: dict[str, str] = {}
    for raw_marker, raw_value in values.items():
        marker = str(raw_marker).strip()
        if not marker:
            raise ValueError("Nimbus normalization data contains an empty marker name.")
        folded = marker.casefold()
        if folded in folded_keys:
            raise ValueError(
                "Nimbus normalization marker names must be unique ignoring case; "
                f"found {folded_keys[folded]!r} and {marker!r}."
            )
        folded_keys[folded] = marker
        validated[marker] = _coerce_parameters(marker, raw_value)
    if not validated:
        raise ValueError("Nimbus normalization data contains no marker rows.")
    return validated


def build_normalization_parameters(
    vmax_values: Mapping[str, float],
    lower_thresholds: Mapping[str, float] | None = None,
) -> dict[str, NimbusNormalizationParameters]:
    """Combine Vmax values with case-insensitive lower-threshold overrides."""

    lower_values = lower_thresholds or {}
    folded_lower: dict[str, tuple[str, float]] = {}
    for raw_marker, raw_value in lower_values.items():
        marker = str(raw_marker).strip()
        folded = marker.casefold()
        if not marker:
            raise ValueError("Lower-threshold data contains an empty marker name.")
        if folded in folded_lower:
            raise ValueError(
                "Lower-threshold marker names must be unique ignoring case; found "
                f"{folded_lower[folded][0]!r} and {marker!r}."
            )
        folded_lower[folded] = (marker, float(raw_value))

    payload: dict[str, object] = {}
    used_lower: set[str] = set()
    for raw_marker, vmax in vmax_values.items():
        marker = str(raw_marker).strip()
        folded = marker.casefold()
        lower_entry = folded_lower.get(folded)
        lower = 0.0 if lower_entry is None else lower_entry[1]
        if lower_entry is not None:
            used_lower.add(folded)
        payload[marker] = {"vmax": vmax, "lower_threshold": lower}
    unused_lower = [
        original for folded, (original, _) in folded_lower.items() if folded not in used_lower
    ]
    if unused_lower:
        raise ValueError(
            "Lower thresholds were supplied for markers without Vmax values: "
            f"{sorted(unused_lower)}."
        )
    return validate_normalization_parameters(payload)


def merge_computed_normalization_parameters(
    computed_vmax_values: Mapping[str, float],
    *,
    default_lower_threshold: float = 0.0,
    saved_parameters: Mapping[str, NimbusNormalizationParameters] | None = None,
) -> dict[str, NimbusNormalizationParameters]:
    """Combine computed defaults with saved rows, with saved rows taking precedence."""

    if not computed_vmax_values:
        if saved_parameters:
            return dict(saved_parameters)
        return build_normalization_parameters(computed_vmax_values)

    computed = build_normalization_parameters(
        computed_vmax_values,
        {marker: default_lower_threshold for marker in computed_vmax_values},
    )
    computed.update(saved_parameters or {})
    return computed


def _load_csv(path: Path) -> dict[str, NimbusNormalizationParameters]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                "Nimbus normalization CSV is empty; expected marker, vmax, and "
                "lower_threshold columns."
            )
        normalized_headers: dict[str, str] = {}
        for original in reader.fieldnames:
            normalized = _normalized_field_name(original)
            if normalized in normalized_headers:
                raise ValueError(
                    "Nimbus normalization CSV contains duplicate column names after "
                    f"normalization: {original!r}."
                )
            normalized_headers[normalized] = original

        marker_column = normalized_headers.get("marker") or normalized_headers.get(
            "channel"
        )
        vmax_column = (
            normalized_headers.get("vmax")
            or normalized_headers.get("baseline_vmax")
            or normalized_headers.get("baseline")
            or normalized_headers.get("value")
        )
        lower_column = (
            normalized_headers.get("lower_threshold")
            or normalized_headers.get("lower_bound")
            or normalized_headers.get("lower")
            or normalized_headers.get("vmin")
        )
        if marker_column is None or vmax_column is None:
            raise ValueError(
                "Nimbus normalization CSV must contain marker and vmax or "
                "baseline_vmax columns. Baseline and lower_bound aliases are also "
                "accepted, as are legacy Marker and Value columns."
            )

        payload: dict[str, object] = {}
        seen: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            marker = str(row.get(marker_column) or "").strip()
            if not marker:
                raise ValueError(
                    f"Nimbus normalization CSV row {row_number} has an empty marker."
                )
            folded = marker.casefold()
            if folded in seen:
                raise ValueError(
                    "Nimbus normalization CSV markers must be unique ignoring case; "
                    f"row {row_number} repeats {marker!r} from row {seen[folded]}."
                )
            seen[folded] = row_number
            payload[marker] = {
                "vmax": row.get(vmax_column),
                "lower_threshold": (
                    0.0 if lower_column is None else row.get(lower_column)
                ),
            }
    return validate_normalization_parameters(payload)


def load_normalization_file(
    path: str | Path,
) -> dict[str, NimbusNormalizationParameters]:
    """Load preferred CSV or legacy JSON Nimbus normalization parameters."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Nimbus normalization file not found: {source}")
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        return _load_csv(source)
    if suffix == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Nimbus normalization JSON at {source}: {exc}") from exc
        if isinstance(payload, Mapping) and isinstance(
            payload.get("normalization_dict"), Mapping
        ):
            payload = payload["normalization_dict"]
        return validate_normalization_parameters(payload)
    raise ValueError(
        "Nimbus normalization files must use .csv (preferred) or .json (legacy); "
        f"got {source.suffix or '<none>'!r}."
    )


def resolve_normalization_input_path(
    output_dir: str | Path,
    *,
    configured_path: str | Path | None = None,
    reuse_saved: bool = False,
) -> Path | None:
    """Resolve an explicit preferred CSV or the legacy output-directory fallback."""

    if configured_path is not None and str(configured_path).strip():
        source = Path(configured_path).expanduser().resolve(strict=False)
        if source.suffix.casefold() != ".csv":
            raise ValueError(
                "nimbus.normalization_dict_path must point to a .csv file in the "
                "marker,vmax,lower_threshold format."
            )
        if not source.is_file():
            raise FileNotFoundError(
                f"Configured Nimbus normalization CSV not found: {source}"
            )
        return source

    if not reuse_saved:
        return None
    root = Path(output_dir)
    preferred = root / PREFERRED_NORMALIZATION_FILENAME
    if preferred.is_file():
        return preferred
    legacy = root / LEGACY_NORMALIZATION_FILENAME
    if legacy.is_file():
        return legacy
    return None


def resolve_normalization_parameters(
    parameters: Mapping[str, NimbusNormalizationParameters],
    markers: Sequence[str],
    *,
    require_all: bool = True,
) -> dict[str, NimbusNormalizationParameters]:
    """Resolve exact or unique case-insensitive file markers to canonical names."""

    folded: dict[str, list[str]] = {}
    for marker in parameters:
        folded.setdefault(str(marker).casefold(), []).append(str(marker))
    resolved: dict[str, NimbusNormalizationParameters] = {}
    missing: list[str] = []
    for raw_marker in markers:
        marker = str(raw_marker)
        if marker in parameters:
            matched = marker
        else:
            matches = folded.get(marker.casefold(), [])
            if len(matches) == 1:
                matched = matches[0]
            elif not matches:
                missing.append(marker)
                continue
            else:
                raise ValueError(
                    f"Normalization marker {marker!r} is ambiguous: {matches}."
                )
        resolved[marker] = parameters[matched]
    if require_all and missing:
        raise ValueError(f"Nimbus normalization file is missing markers: {missing}.")
    return resolved


def write_normalization_csv(
    path: str | Path,
    vmax_values: Mapping[str, float],
    lower_thresholds: Mapping[str, float] | None = None,
) -> Path:
    """Atomically write the preferred marker/Vmax/lower-threshold CSV."""

    destination = Path(path)
    if destination.suffix.casefold() != ".csv":
        raise ValueError("Preferred Nimbus normalization output must use a .csv suffix.")
    parameters = build_normalization_parameters(vmax_values, lower_thresholds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREFERRED_COLUMNS))
        writer.writeheader()
        for marker, entry in parameters.items():
            writer.writerow(
                {
                    "marker": marker,
                    "vmax": format(entry.vmax, ".15g"),
                    "lower_threshold": format(entry.lower_threshold, ".15g"),
                }
            )
    temporary.replace(destination)
    return destination


def normalize_nimbus_image(
    image: np.ndarray,
    *,
    vmax: float,
    lower_threshold: float = 0.0,
    upper_clip: float = 1.0,
) -> np.ndarray:
    """Apply two-point Nimbus normalization in absolute input-intensity units."""

    entry = _coerce_parameters(
        "image",
        {"vmax": vmax, "lower_threshold": lower_threshold},
    )
    upper = float(upper_clip)
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError("Nimbus normalization upper_clip must be finite and positive.")
    values = np.asarray(image, dtype=np.float32)
    normalized = (values - entry.lower_threshold) / (
        entry.vmax - entry.lower_threshold
    )
    return np.clip(normalized, 0.0, upper)


__all__ = [
    "LEGACY_NORMALIZATION_FILENAME",
    "PREFERRED_COLUMNS",
    "PREFERRED_NORMALIZATION_FILENAME",
    "NimbusNormalizationParameters",
    "build_normalization_parameters",
    "load_normalization_file",
    "normalize_nimbus_image",
    "resolve_normalization_input_path",
    "resolve_normalization_parameters",
    "validate_normalization_parameters",
    "write_normalization_csv",
]
