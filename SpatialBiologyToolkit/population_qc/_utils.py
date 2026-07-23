"""Internal validation and bounded matrix helpers for population QC."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


SBT_METADATA_KEY = "spatial_biology_toolkit"


def resolve_table(data: Any, table_name: str | None = None) -> tuple[str | None, Any]:
    """Resolve an AnnData table from SpatialData or return supplied AnnData."""

    if hasattr(data, "tables"):
        tables = data.tables
        metadata = getattr(data, "attrs", {}).get(SBT_METADATA_KEY, {})
        selected = table_name
        if selected is None and isinstance(metadata, Mapping):
            selected = metadata.get("table_name")
        if selected is None:
            names = list(tables)
            if len(names) != 1:
                raise ValueError(
                    "table_name is required when SpatialData does not contain exactly one table"
                )
            selected = str(names[0])
        selected = str(selected)
        if selected not in tables:
            raise KeyError(f"SpatialData table {selected!r} was not found")
        return selected, tables[selected]
    if hasattr(data, "obs") and hasattr(data, "var_names"):
        if table_name is not None:
            raise ValueError("table_name is only valid for SpatialData input")
        return None, data
    raise TypeError("data must be a SpatialData object or AnnData table")


def metadata_for(data: Any) -> Mapping[str, Any]:
    """Return SpatialBiologyToolkit metadata or an empty mapping for AnnData."""

    metadata = getattr(data, "attrs", {}).get(SBT_METADATA_KEY, {})
    return metadata if isinstance(metadata, Mapping) else {}


def resolve_roi_key(data: Any, adata: Any, roi_key: str | None) -> str | None:
    """Resolve an ROI column from an override, toolkit metadata, or ``ROI``."""

    candidate = roi_key
    if candidate is None:
        candidate = metadata_for(data).get("roi_key")
    if candidate is None and "ROI" in adata.obs:
        candidate = "ROI"
    if candidate is not None and str(candidate) not in adata.obs:
        raise KeyError(f"ROI column {candidate!r} is missing from the table")
    return str(candidate) if candidate is not None else None


def infer_case_key(adata: Any, case_key: str | None) -> str | None:
    """Resolve an explicit case key or the first common unambiguous spelling."""

    if case_key is not None:
        if case_key not in adata.obs:
            raise KeyError(f"Case/sample column {case_key!r} is missing from the table")
        return case_key
    candidates = (
        "Case/Animal",
        "case/animal",
        "case",
        "Case",
        "animal",
        "Animal",
        "patient",
        "Patient",
        "sample",
        "Sample",
    )
    return next((value for value in candidates if value in adata.obs), None)


def validate_population(adata: Any, population_key: str, population: Any) -> pd.Series:
    """Return a Boolean mask for an existing, non-empty population."""

    if population_key not in adata.obs:
        raise KeyError(f"Population column {population_key!r} is missing from the table")
    labels = adata.obs[population_key]
    mask = labels.notna() & (labels.astype(str) == str(population))
    if not bool(mask.any()):
        available = labels.dropna().astype(str).value_counts().head(20).index.tolist()
        raise KeyError(
            f"Population {population!r} is absent from {population_key!r}; "
            f"available values include {available}"
        )
    return mask


def ordered_labels(series: pd.Series) -> list[str]:
    """Return non-missing labels, preserving categorical order when possible."""

    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(series.dropna().astype(str))
        return [str(value) for value in series.cat.categories if str(value) in observed]
    return [str(value) for value in pd.unique(series.dropna().astype(str))]


def validate_markers(adata: Any, markers: Sequence[str] | None) -> list[str]:
    """Return exact marker names and reject missing or duplicate requests."""

    if not adata.var_names.is_unique:
        raise ValueError("adata.var_names must be unique for marker-based QC")
    available = {str(value) for value in adata.var_names}
    selected = [str(value) for value in (markers or list(map(str, adata.var_names)))]
    selected = list(dict.fromkeys(selected))
    missing = [value for value in selected if value not in available]
    if missing:
        raise KeyError(
            f"Markers are absent from the table: {missing}. Marker matching is exact; "
            "provide an explicit alias mapping before QC if names differ."
        )
    if not selected:
        raise ValueError("At least one marker is required")
    return selected


def sample_positions(
    positions: np.ndarray,
    max_cells: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deterministically sample sorted row positions without replacement."""

    positions = np.asarray(positions, dtype=np.int64)
    if max_cells is None or len(positions) <= max_cells:
        return positions
    if max_cells < 1:
        raise ValueError("max_cells must be positive or None")
    return np.sort(rng.choice(positions, size=int(max_cells), replace=False))


def matrix_for_positions(
    adata: Any,
    positions: np.ndarray,
    markers: Sequence[str],
    *,
    layer: str | None = None,
) -> np.ndarray:
    """Materialise only selected rows and marker columns as a dense float array."""

    marker_index = pd.Index(adata.var_names.astype(str))
    column_positions = marker_index.get_indexer(list(markers))
    if (column_positions < 0).any():
        missing = [markers[index] for index in np.flatnonzero(column_positions < 0)]
        raise KeyError(f"Markers are absent from the table: {missing}")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} is missing from the table")
    matrix = adata.layers[layer] if layer is not None else adata.X
    selected = matrix[np.asarray(positions, dtype=np.int64), :][:, column_positions]
    if sparse.issparse(selected):
        selected = selected.toarray()
    values = np.asarray(selected, dtype=float)
    return values[:, None] if values.ndim == 1 else values


def shape_tuple(value: Any) -> tuple[int, ...]:
    """Return a JSON-friendly shape tuple for an array-like object."""

    return tuple(int(item) for item in getattr(value, "shape", ()))


def clean_key(value: Any) -> str:
    """Create a conservative observation-column fragment."""

    cleaned = "".join(
        character if character.isalnum() else "_" for character in str(value).strip()
    )
    return "_".join(part for part in cleaned.split("_") if part) or "population"


def unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    """Return unique string values in first-seen order."""

    return tuple(dict.fromkeys(str(value) for value in values))


__all__ = [
    "clean_key",
    "infer_case_key",
    "matrix_for_positions",
    "metadata_for",
    "ordered_labels",
    "resolve_roi_key",
    "resolve_table",
    "sample_positions",
    "shape_tuple",
    "unique_strings",
    "validate_markers",
    "validate_population",
]
