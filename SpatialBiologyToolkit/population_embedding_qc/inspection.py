"""AnnData validation and clustering-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class InspectionResult:
    reference_column: str
    labels: pd.Series
    valid_mask: np.ndarray
    cluster_order: list[str]
    umap: np.ndarray
    pca: np.ndarray | None
    connectivities: sparse.csr_matrix | None
    connectivities_key: str | None
    sweep: list[tuple[str, float]]
    warnings: list[str] = field(default_factory=list)
    excluded_cells: int = 0


def _ordered_labels(series: pd.Series) -> tuple[pd.Series, list[str]]:
    valid = series.notna()
    values = series.loc[valid]
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = {str(value) for value in values.astype(str)}
        order = [str(value) for value in values.cat.categories if str(value) in present]
    else:
        order = list(dict.fromkeys(str(value) for value in values.tolist()))
    converted = pd.Series(pd.NA, index=series.index, dtype="string")
    converted.loc[valid] = values.astype(str).to_numpy()
    return converted, order


def plausible_population_columns(obs: pd.DataFrame) -> list[str]:
    """Return bounded, plausible categorical columns for actionable errors."""
    candidates: list[str] = []
    n_obs = max(len(obs), 1)
    for column in obs.columns:
        series = obs[column]
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object or pd.api.types.is_string_dtype(series):
            unique = series.nunique(dropna=True)
            if 1 < unique <= min(200, max(20, n_obs // 2)):
                candidates.append(str(column))
    return candidates[:30]


def detect_sweep_columns(
    obs: pd.DataFrame,
    *,
    sweep_regex: str,
    explicit_columns: Iterable[str] | None = None,
) -> tuple[list[tuple[str, float]], list[str]]:
    """Detect numerically sorted, precomputed Leiden sweep columns."""
    try:
        pattern = re.compile(sweep_regex)
    except re.error as exc:
        raise ValueError(f"Invalid sweep regular expression: {exc}") from exc
    if "resolution" not in pattern.groupindex:
        raise ValueError("Sweep regular expression must contain a named 'resolution' group")
    columns = list(explicit_columns) if explicit_columns is not None else list(obs.columns)
    detected: list[tuple[str, float]] = []
    warnings: list[str] = []
    for column in columns:
        if column not in obs.columns:
            if explicit_columns is not None:
                raise ValueError(f"Explicit sweep column is missing from adata.obs: {column}")
            continue
        match = pattern.fullmatch(str(column))
        if match is None:
            if explicit_columns is not None:
                raise ValueError(
                    f"Explicit sweep column {column!r} does not match the configured sweep regular expression"
                )
            continue
        try:
            resolution = float(match.group("resolution"))
        except (TypeError, ValueError):
            warnings.append(f"Ignored sweep column {column!r}: invalid numerical resolution")
            continue
        if not np.isfinite(resolution):
            warnings.append(f"Ignored sweep column {column!r}: non-finite resolution")
            continue
        detected.append((str(column), resolution))
    detected.sort(key=lambda item: (item[1], item[0]))
    seen_resolution: dict[float, str] = {}
    for column, resolution in detected:
        previous = seen_resolution.get(resolution)
        if previous is not None:
            warnings.append(
                f"Sweep columns {previous!r} and {column!r} encode the same numerical resolution {resolution:g}"
            )
        seen_resolution.setdefault(resolution, column)
    for index, (left, _left_resolution) in enumerate(detected):
        for right, _right_resolution in detected[index + 1 :]:
            if obs[left].equals(obs[right]):
                warnings.append(
                    f"Duplicate clustering assignments detected in sweep columns {left!r} and {right!r}"
                )
    return detected, warnings


def select_reference_column(
    obs: pd.DataFrame,
    *,
    population_obs: str | None,
    mode: str,
    sweep: list[tuple[str, float]],
    reference_resolution: float | None,
) -> str:
    if mode not in {"auto", "single", "sweep"}:
        raise ValueError("mode must be one of: auto, single, sweep")
    if population_obs:
        if population_obs not in obs.columns:
            raise ValueError(f"Population column {population_obs!r} is missing from adata.obs")
        return population_obs
    if mode == "single":
        raise ValueError("population_obs is required in single mode")
    if mode == "auto":
        for preferred in ("population", "leiden"):
            if preferred in obs.columns:
                return preferred
    if sweep:
        if reference_resolution is not None:
            matches = [item for item in sweep if np.isclose(item[1], reference_resolution)]
            if not matches:
                available = ", ".join(f"{resolution:g}" for _, resolution in sweep)
                raise ValueError(
                    f"Reference resolution {reference_resolution:g} was not detected; available resolutions: {available}"
                )
            return matches[0][0]
        return sweep[(len(sweep) - 1) // 2][0]
    plausible = plausible_population_columns(obs)
    suffix = f" Plausible categorical columns: {', '.join(plausible)}." if plausible else ""
    raise ValueError(
        "Could not select a reference population column. Configure population_obs or provide at least two valid sweep columns."
        + suffix
    )


def _validate_representation(adata: Any, key: str, *, required: bool, dimensions: int | None = None) -> np.ndarray | None:
    if key not in adata.obsm:
        if required:
            raise ValueError(f"Required UMAP representation adata.obsm[{key!r}] is unavailable")
        return None
    values = np.asarray(adata.obsm[key])
    if values.ndim != 2 or values.shape[0] != adata.n_obs:
        raise ValueError(f"adata.obsm[{key!r}] must be a two-dimensional array with {adata.n_obs} rows")
    required_dimensions = 2 if required else 1
    if values.shape[1] < required_dimensions:
        raise ValueError(f"adata.obsm[{key!r}] must contain at least {required_dimensions} dimensions")
    use_dimensions = 2 if required else min(values.shape[1], dimensions or values.shape[1])
    selected = np.asarray(values[:, :use_dimensions], dtype=float)
    if not np.isfinite(selected).all():
        raise ValueError(f"adata.obsm[{key!r}] contains non-finite values")
    return selected


def _load_connectivities(adata: Any, explicit_key: str | None) -> tuple[sparse.csr_matrix | None, str | None, list[str]]:
    warnings: list[str] = []
    if explicit_key is not None and explicit_key not in adata.obsp:
        raise ValueError(
            f"Configured connectivity matrix adata.obsp[{explicit_key!r}] is unavailable"
        )
    recorded_key = None
    neighbors = adata.uns.get("neighbors", {})
    if isinstance(neighbors, dict):
        recorded_key = neighbors.get("connectivities_key")
    candidates = [explicit_key] if explicit_key else [recorded_key, "connectivities"]
    key = next((str(candidate) for candidate in candidates if candidate and candidate in adata.obsp), None)
    if key is None:
        warnings.append("No existing Scanpy connectivity graph was found; graph metrics were skipped")
        return None, None, warnings
    matrix = adata.obsp[key]
    if getattr(matrix, "shape", None) != (adata.n_obs, adata.n_obs):
        raise ValueError(f"adata.obsp[{key!r}] must have shape ({adata.n_obs}, {adata.n_obs})")
    graph = sparse.csr_matrix(matrix, dtype=float)
    if graph.data.size and (not np.isfinite(graph.data).all() or np.any(graph.data < 0)):
        raise ValueError(f"adata.obsp[{key!r}] contains non-finite or negative edge weights")
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph, key, warnings


def inspect_anndata(
    adata: Any,
    *,
    population_obs: str | None,
    mode: str,
    sweep_columns: list[str] | None,
    sweep_regex: str,
    reference_resolution: float | None,
    umap_key: str,
    pca_key: str,
    pca_dimensions: int,
    connectivities_key: str | None,
) -> InspectionResult:
    """Inspect every required/optional AnnData input before analysis."""
    if adata.n_obs < 2:
        raise ValueError("Population embedding QC requires at least two observations")
    sweep, warnings = detect_sweep_columns(
        adata.obs,
        sweep_regex=sweep_regex,
        explicit_columns=sweep_columns,
    )
    if mode == "single":
        sweep = []
    if mode == "sweep" and len(sweep) < 2:
        raise ValueError("Sweep mode requires at least two valid precomputed clustering columns")
    if len(sweep) == 1:
        warnings.append("Only one sweep column was detected; resolution-stability metrics were skipped")
        sweep = []
    reference = select_reference_column(
        adata.obs,
        population_obs=population_obs,
        mode=mode,
        sweep=sweep,
        reference_resolution=reference_resolution,
    )
    labels, order = _ordered_labels(adata.obs[reference])
    valid_mask = labels.notna().to_numpy()
    excluded = int((~valid_mask).sum())
    if not valid_mask.any():
        raise ValueError(f"Population column {reference!r} contains no non-missing labels")
    if len(order) < 2:
        warnings.append("The reference clustering contains fewer than two populations; separation metrics are limited")
    if excluded:
        warnings.append(f"Excluded {excluded} cells with missing labels in reference column {reference!r}")
    umap = _validate_representation(adata, umap_key, required=True)
    assert umap is not None
    pca = _validate_representation(adata, pca_key, required=False, dimensions=pca_dimensions)
    if pca is None:
        warnings.append(f"PCA representation adata.obsm[{pca_key!r}] is unavailable; PCA metrics were skipped")
    connectivities, graph_key, graph_warnings = _load_connectivities(adata, connectivities_key)
    warnings.extend(graph_warnings)
    return InspectionResult(
        reference_column=reference,
        labels=labels,
        valid_mask=valid_mask,
        cluster_order=order,
        umap=umap,
        pca=pca,
        connectivities=connectivities,
        connectivities_key=graph_key,
        sweep=sweep,
        warnings=warnings,
        excluded_cells=excluded,
    )


__all__ = [
    "InspectionResult",
    "detect_sweep_columns",
    "inspect_anndata",
    "plausible_population_columns",
    "select_reference_column",
]
