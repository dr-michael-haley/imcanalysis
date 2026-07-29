"""Strict cohort-first loading and joining of reusable feature sources."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .models import FeatureSource
from .storage import read_dataframe

IDENTITY = ["ROI", "ObjectNumber"]


@dataclass
class FeatureSourceResult:
    """One namespaced feature table and its cohort coverage."""

    source_id: str
    table: pd.DataFrame
    feature_columns: list[str]
    missing_cells: pd.DataFrame

    @property
    def covered_cell_count(self) -> int:
        return len(self.table)


@dataclass
class CombinedFeatureResult:
    """Frozen-cohort rows with all enabled feature sources joined."""

    table: pd.DataFrame
    feature_columns: list[str]
    coverage: pd.DataFrame
    missing_by_source: dict[str, pd.DataFrame]


def _normalise_identity(
    frame: pd.DataFrame,
    *,
    roi_column: str,
    object_column: str,
    source_name: str,
) -> pd.DataFrame:
    missing = [
        column for column in (roi_column, object_column) if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{source_name} is missing identity column(s): {missing}")
    result = frame.copy()
    result["ROI"] = result[roi_column].astype(str)
    object_ids = pd.to_numeric(result[object_column], errors="coerce")
    invalid = object_ids.isna() | (object_ids <= 0) | (object_ids % 1 != 0)
    if invalid.any():
        examples = result.loc[invalid, [roi_column, object_column]].head().to_dict(
            "records"
        )
        raise ValueError(
            f"{source_name} has invalid positive-integer object IDs; examples: "
            f"{examples}"
        )
    result["ObjectNumber"] = object_ids.astype(np.int64)
    result = result.drop(
        columns=[
            column
            for column in (roi_column, object_column)
            if column not in {"ROI", "ObjectNumber"}
        ]
    )
    duplicates = result.duplicated(IDENTITY, keep=False)
    if duplicates.any():
        examples = result.loc[duplicates, IDENTITY].head().to_dict("records")
        raise ValueError(
            f"{source_name} contains duplicate (ROI, ObjectNumber) identities; "
            f"examples: {examples}"
        )
    return result


def _namespace_features(
    frame: pd.DataFrame,
    *,
    source_id: str,
    selected_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    selected = [str(column) for column in selected_columns]
    if selected:
        missing = sorted(set(selected) - set(frame.columns))
        if missing:
            raise ValueError(
                f"Feature source {source_id!r} is missing selected column(s): {missing}"
            )
        candidates = selected
    else:
        candidates = [
            column
            for column in frame.columns
            if column not in {"ROI", "ObjectNumber", "obs_name", "CellID"}
        ]
    numeric = [
        column
        for column in candidates
        if pd.api.types.is_numeric_dtype(frame[column])
        or pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    if not numeric:
        raise ValueError(f"Feature source {source_id!r} contains no numeric features.")
    rename = {column: f"source::{source_id}::{column}" for column in numeric}
    result = frame.loc[:, IDENTITY + numeric].rename(columns=rename)
    for column in rename.values():
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result, list(rename.values())


def _matrix_frame(matrix, index: pd.Index, columns: list[str]) -> pd.DataFrame:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    values = np.asarray(matrix)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(index):
        raise ValueError(
            "AnnData feature representation must be a two-dimensional cell-by-feature "
            f"matrix; got {values.shape} for {len(index)} cells."
        )
    if len(columns) != values.shape[1]:
        columns = [f"feature_{index}" for index in range(values.shape[1])]
    return pd.DataFrame(values, index=index, columns=columns)


def anndata_feature_frame(
    path: str | Path,
    *,
    representation: str,
    roi_obs: str,
    object_id_obs: str,
    selected_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Read one supported AnnData representation with strict identities."""

    import anndata as ad

    adata = ad.read_h5ad(path)
    missing = [
        column
        for column in (roi_obs, object_id_obs)
        if column not in adata.obs.columns
    ]
    if missing:
        raise ValueError(f"AnnData feature source is missing observation(s): {missing}")
    representation = str(representation or "X")
    if representation == "X":
        matrix = _matrix_frame(adata.X, adata.obs_names, adata.var_names.astype(str).tolist())
    elif representation == "obs":
        columns = [str(column) for column in selected_columns]
        if not columns:
            columns = [
                str(column)
                for column in adata.obs.select_dtypes(include=[np.number]).columns
                if column not in {roi_obs, object_id_obs}
            ]
        missing_columns = sorted(set(columns) - set(adata.obs.columns))
        if missing_columns:
            raise ValueError(
                f"AnnData obs feature columns are missing: {missing_columns}"
            )
        matrix = adata.obs.loc[:, columns].copy()
    else:
        key = (
            representation.split(":", 1)[1]
            if representation.startswith("obsm:")
            else representation
        )
        if key not in adata.obsm:
            raise ValueError(f"AnnData obsm representation {key!r} is missing.")
        raw = adata.obsm[key]
        if isinstance(raw, pd.DataFrame):
            matrix = raw.copy()
        else:
            matrix = _matrix_frame(
                raw,
                adata.obs_names,
                [f"{key}_{index}" for index in range(raw.shape[1])],
            )
    matrix.index = adata.obs_names
    identity = adata.obs.loc[:, [roi_obs, object_id_obs]].copy()
    identity.index = adata.obs_names
    return pd.concat([identity, matrix], axis=1).reset_index(names="obs_name")


def load_feature_source(
    source: FeatureSource,
    cohort: pd.DataFrame,
    *,
    roi_obs: str = "ROI",
    object_id_obs: str = "ObjectNumber",
) -> FeatureSourceResult:
    """Load, identity-join, cohort-filter, and namespace one feature source."""

    cohort_identity = _normalise_identity(
        cohort,
        roi_column="ROI",
        object_column="ObjectNumber",
        source_name="frozen cohort",
    )
    if source.kind in {"table", "synthetic"}:
        if not source.path:
            raise ValueError(f"Feature source {source.source_id!r} requires a path.")
        raw = read_dataframe(source.path)
    elif source.kind == "anndata":
        raw = anndata_feature_frame(
            source.path or "",
            representation=source.representation or "X",
            roi_obs=roi_obs,
            object_id_obs=object_id_obs,
            selected_columns=source.selected_columns,
        )
    else:
        raise ValueError(f"Unsupported feature source kind: {source.kind}")

    normalised = _normalise_identity(
        raw,
        roi_column=roi_obs if roi_obs in raw.columns else "ROI",
        object_column=object_id_obs if object_id_obs in raw.columns else "ObjectNumber",
        source_name=f"feature source {source.source_id!r}",
    )
    matched = cohort_identity.loc[:, IDENTITY].merge(
        normalised,
        on=IDENTITY,
        how="inner",
        validate="one_to_one",
    )
    namespaced, features = _namespace_features(
        matched,
        source_id=source.source_id,
        selected_columns=source.selected_columns,
    )
    missing = cohort_identity.loc[:, IDENTITY].merge(
        namespaced.loc[:, IDENTITY],
        on=IDENTITY,
        how="left",
        indicator=True,
    )
    missing = missing.loc[missing["_merge"] == "left_only", IDENTITY].reset_index(
        drop=True
    )
    return FeatureSourceResult(
        source_id=source.source_id,
        table=namespaced,
        feature_columns=features,
        missing_cells=missing,
    )


def combine_feature_sources(
    cohort: pd.DataFrame,
    sources: Iterable[FeatureSource],
    *,
    roi_obs: str = "ROI",
    object_id_obs: str = "ObjectNumber",
) -> CombinedFeatureResult:
    """Left-join enabled sources onto the frozen cohort without adding rows."""

    base = _normalise_identity(
        cohort,
        roi_column="ROI",
        object_column="ObjectNumber",
        source_name="frozen cohort",
    )
    metadata = [
        column
        for column in ("obs_name", "source_population")
        if column in base.columns
    ]
    combined = base.loc[:, metadata + IDENTITY].copy()
    all_features: list[str] = []
    coverage_rows: list[dict[str, object]] = []
    missing_by_source: dict[str, pd.DataFrame] = {}
    for source in sources:
        if not source.enabled:
            continue
        result = load_feature_source(
            source,
            base,
            roi_obs=roi_obs,
            object_id_obs=object_id_obs,
        )
        combined = combined.merge(
            result.table,
            on=IDENTITY,
            how="left",
            validate="one_to_one",
        )
        all_features.extend(result.feature_columns)
        missing_by_source[source.source_id] = result.missing_cells
        coverage_rows.append(
            {
                "source_id": source.source_id,
                "eligible_cells": len(base),
                "covered_cells": result.covered_cell_count,
                "missing_cells": len(result.missing_cells),
                "feature_count": len(result.feature_columns),
            }
        )
    return CombinedFeatureResult(
        table=combined,
        feature_columns=all_features,
        coverage=pd.DataFrame(coverage_rows),
        missing_by_source=missing_by_source,
    )


__all__ = [
    "CombinedFeatureResult",
    "FeatureSourceResult",
    "anndata_feature_frame",
    "combine_feature_sources",
    "load_feature_source",
]
