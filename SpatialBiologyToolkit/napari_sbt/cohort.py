"""Identity resolution and cohort-mask construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models import CellScope
from .storage import dataframe_sha256, write_dataframe

IDENTITY_COLUMNS = ["obs_name", "ROI", "ObjectNumber"]


@dataclass
class CohortPreview:
    """Validated summary of an experiment cohort."""

    eligible_cells: pd.DataFrame
    total_cell_count: int
    per_roi_counts: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    @property
    def eligible_cell_count(self) -> int:
        return len(self.eligible_cells)

    @property
    def represented_roi_count(self) -> int:
        return int(self.eligible_cells["ROI"].nunique())

    @property
    def eligible_fraction(self) -> float:
        return self.eligible_cell_count / self.total_cell_count

    @property
    def fingerprint(self) -> str:
        return dataframe_sha256(self.eligible_cells, IDENTITY_COLUMNS)

    def scope(
        self,
        *,
        mode: str,
        obs_column: str | None,
        obs_values: Iterable[str],
        snapshot_path: str = "cohort/eligible_cells.parquet",
    ) -> CellScope:
        return CellScope(
            mode=mode,
            obs_column=obs_column,
            obs_values=[str(value) for value in obs_values],
            snapshot_path=snapshot_path,
            snapshot_sha256=self.fingerprint,
            eligible_cell_count=self.eligible_cell_count,
            total_cell_count=self.total_cell_count,
            represented_roi_count=self.represented_roi_count,
        )


def _validated_object_ids(values: pd.Series, object_id_obs: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | (numeric <= 0) | (numeric % 1 != 0)
    if invalid.any():
        examples = values.loc[invalid].head().astype(str).tolist()
        raise ValueError(
            f"adata.obs[{object_id_obs!r}] must contain positive integer mask "
            f"labels; invalid examples: {examples}"
        )
    return numeric.astype(np.int64)


def resolve_cohort(
    adata,
    *,
    roi_obs: str,
    object_id_obs: str,
    mode: str,
    obs_column: str | None = None,
    obs_values: Iterable[str] = (),
) -> CohortPreview:
    """Resolve a strict, identity-stable cohort from an AnnData-like object."""

    missing = [
        column
        for column in (roi_obs, object_id_obs)
        if column not in adata.obs.columns
    ]
    if missing:
        raise KeyError(f"AnnData is missing identity observation(s): {missing}")
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData observation names must be unique.")

    obs = adata.obs.copy()
    roi = obs[roi_obs]
    if roi.isna().any() or roi.astype(str).str.strip().eq("").any():
        raise ValueError(f"adata.obs[{roi_obs!r}] contains missing or blank ROI values.")
    object_ids = _validated_object_ids(obs[object_id_obs], object_id_obs)

    mode = str(mode)
    selected_values = [str(value) for value in obs_values]
    source_values: pd.Series | None = None
    if mode == "all_cells":
        selected = np.ones(len(obs), dtype=bool)
    elif mode == "obs_values":
        if not obs_column or obs_column not in obs.columns:
            raise KeyError(f"Cohort observation {obs_column!r} is missing from AnnData.")
        source = obs[obs_column]
        if pd.api.types.is_numeric_dtype(source.dtype) and not isinstance(
            source.dtype, pd.CategoricalDtype
        ):
            raise TypeError(
                f"adata.obs[{obs_column!r}] is numeric, not categorical. Convert "
                "it to a categorical population annotation before cohort selection."
            )
        if source.isna().all():
            raise ValueError(f"adata.obs[{obs_column!r}] contains no usable values.")
        available = set(source.dropna().astype(str))
        missing_values = sorted(set(selected_values) - available)
        if missing_values:
            raise ValueError(
                f"Selected cohort value(s) are absent from {obs_column!r}: "
                + ", ".join(missing_values)
            )
        source_values = source.astype("string")
        selected = source_values.astype(str).isin(selected_values).to_numpy()
    else:
        raise ValueError("Cohort mode must be 'all_cells' or 'obs_values'.")

    eligible = pd.DataFrame(
        {
            "obs_name": pd.Index(adata.obs_names).astype(str),
            "ROI": roi.astype(str).to_numpy(),
            "ObjectNumber": object_ids.to_numpy(),
        }
    ).loc[selected]
    if source_values is not None:
        eligible["source_population"] = source_values.loc[selected].astype(str).to_numpy()
    eligible = eligible.reset_index(drop=True)
    if eligible.empty:
        raise ValueError("The selected cohort contains no cells.")

    duplicates = eligible.duplicated(["ROI", "ObjectNumber"], keep=False)
    if duplicates.any():
        examples = eligible.loc[duplicates, IDENTITY_COLUMNS].head().to_dict("records")
        raise ValueError(
            "The pair (ROI, ObjectNumber) must uniquely identify eligible cells; "
            f"duplicate examples: {examples}"
        )

    per_roi = (
        eligible.groupby("ROI", observed=True)
        .size()
        .rename("eligible_cells")
        .reset_index()
        .sort_values(["eligible_cells", "ROI"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return CohortPreview(
        eligible_cells=eligible,
        total_cell_count=int(adata.n_obs),
        per_roi_counts=per_roi,
    )


def resolve_table_cohort(
    table: pd.DataFrame,
    *,
    roi_column: str = "ROI",
    object_id_column: str = "ObjectNumber",
) -> CohortPreview:
    """Resolve all-cells standalone compatibility from an identity-bearing table."""

    missing = [
        column
        for column in (roi_column, object_id_column)
        if column not in table.columns
    ]
    if missing:
        raise KeyError(f"Standalone feature table is missing identity column(s): {missing}")
    roi = table[roi_column]
    if roi.isna().any() or roi.astype(str).str.strip().eq("").any():
        raise ValueError("Standalone feature table contains missing or blank ROI values.")
    object_ids = _validated_object_ids(table[object_id_column], object_id_column)
    eligible = pd.DataFrame(
        {
            "obs_name": [
                f"standalone::{roi_value}::{object_id}"
                for roi_value, object_id in zip(roi.astype(str), object_ids)
            ],
            "ROI": roi.astype(str).to_numpy(),
            "ObjectNumber": object_ids.to_numpy(),
        }
    )
    duplicates = eligible.duplicated(["ROI", "ObjectNumber"], keep=False)
    if duplicates.any():
        examples = eligible.loc[duplicates, IDENTITY_COLUMNS].head().to_dict("records")
        raise ValueError(
            "Standalone identities must be unique (ROI, ObjectNumber) pairs; "
            f"examples: {examples}"
        )
    per_roi = (
        eligible.groupby("ROI", observed=True)
        .size()
        .rename("eligible_cells")
        .reset_index()
        .sort_values(["eligible_cells", "ROI"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return CohortPreview(
        eligible_cells=eligible,
        total_cell_count=len(eligible),
        per_roi_counts=per_roi,
    )


def save_cohort_snapshot(preview: CohortPreview, path: str | Path) -> Path:
    return write_dataframe(path, preview.eligible_cells)


def validate_frozen_cohort(snapshot: pd.DataFrame, scope: CellScope) -> None:
    missing = [column for column in IDENTITY_COLUMNS if column not in snapshot.columns]
    if missing:
        raise ValueError(f"Frozen cohort snapshot is missing columns: {missing}")
    observed_hash = dataframe_sha256(snapshot, IDENTITY_COLUMNS)
    if observed_hash != scope.snapshot_sha256:
        raise ValueError(
            "Frozen cohort snapshot no longer matches the experiment manifest. "
            "Create an explicit experiment revision instead of continuing."
        )
    if len(snapshot) != scope.eligible_cell_count:
        raise ValueError("Frozen cohort row count does not match the experiment manifest.")


def eligible_ids_by_roi(snapshot: pd.DataFrame) -> dict[str, set[int]]:
    return {
        str(roi): set(group["ObjectNumber"].astype(int).tolist())
        for roi, group in snapshot.groupby("ROI", observed=True)
    }


def cohort_mask(mask: np.ndarray, eligible_ids: Iterable[int]) -> np.ndarray:
    """Return a label image containing only eligible IDs."""

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional, got {mask.shape}.")
    eligible = np.fromiter((int(value) for value in eligible_ids), dtype=np.int64)
    if eligible.size == 0:
        return np.zeros_like(mask)
    return np.where(np.isin(mask, eligible), mask, 0).astype(mask.dtype, copy=False)


def validate_mask_coverage(
    mask: np.ndarray,
    eligible_ids: Iterable[int],
    *,
    roi: str,
) -> tuple[set[int], set[int]]:
    mask_ids = {int(value) for value in np.unique(mask) if int(value) > 0}
    eligible = {int(value) for value in eligible_ids}
    return eligible - mask_ids, mask_ids - eligible


__all__ = [
    "IDENTITY_COLUMNS",
    "CohortPreview",
    "cohort_mask",
    "eligible_ids_by_roi",
    "resolve_cohort",
    "resolve_table_cohort",
    "save_cohort_snapshot",
    "validate_frozen_cohort",
    "validate_mask_coverage",
]
