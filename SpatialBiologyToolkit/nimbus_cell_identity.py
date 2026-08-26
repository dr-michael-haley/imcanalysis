"""Cell-identity validation for Nimbus quantification outputs.

Nimbus scores are meaningful only when their ``(ROI, ObjectNumber)`` identities
cover the adjusted segmentation masks exactly.  These helpers keep that
invariant independent from the heavyweight Nimbus inference runtime so it can
be tested with small synthetic tables.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ("ROI", "ObjectNumber")


def _normalise_object_numbers(values: Iterable[object], *, source: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values, copy=False), errors="coerce").to_numpy(dtype=float)
    if numeric.size == 0:
        return np.asarray([], dtype=np.int64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{source} contains non-finite or non-numeric ObjectNumber values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{source} contains non-integer ObjectNumber values.")
    object_numbers = numeric.astype(np.int64)
    if np.any(object_numbers <= 0):
        raise ValueError(f"{source} contains non-positive ObjectNumber values.")
    return object_numbers


def validate_object_number_coverage(
    expected: Iterable[object],
    observed: Iterable[object],
    *,
    context: str,
    max_examples: int = 8,
) -> None:
    """Require one observed score identity for every expected positive mask label."""

    expected_ids = _normalise_object_numbers(expected, source=f"{context} expected labels")
    observed_ids = _normalise_object_numbers(observed, source=f"{context} observed labels")

    expected_unique, expected_counts = np.unique(expected_ids, return_counts=True)
    observed_unique, observed_counts = np.unique(observed_ids, return_counts=True)
    duplicate_expected = expected_unique[expected_counts > 1]
    duplicate_observed = observed_unique[observed_counts > 1]
    if duplicate_expected.size or duplicate_observed.size:
        raise ValueError(
            f"{context} contains duplicate cell identities: "
            f"expected_duplicates={duplicate_expected[:max_examples].tolist()}, "
            f"observed_duplicates={duplicate_observed[:max_examples].tolist()}."
        )

    missing = np.setdiff1d(expected_unique, observed_unique, assume_unique=True)
    unexpected = np.setdiff1d(observed_unique, expected_unique, assume_unique=True)
    if missing.size or unexpected.size:
        raise ValueError(
            f"{context} cell-identity coverage mismatch: expected {expected_unique.size} mask cells, "
            f"observed {observed_unique.size}; missing={missing.size}, unexpected={unexpected.size}; "
            f"missing_examples={missing[:max_examples].tolist()}, "
            f"unexpected_examples={unexpected[:max_examples].tolist()}."
        )


def _normalise_identity_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing_columns = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{source} is missing cell-identity column(s): {missing_columns}.")

    identities = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    if identities["ROI"].isna().any():
        raise ValueError(f"{source} contains missing ROI values.")
    identities["ROI"] = identities["ROI"].astype(str)
    identities["ObjectNumber"] = _normalise_object_numbers(
        identities["ObjectNumber"], source=source
    )

    duplicated = identities.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicated.any():
        examples = identities.loc[duplicated, list(IDENTITY_COLUMNS)].head(8).to_dict("records")
        raise ValueError(f"{source} contains duplicate cell identities; examples={examples}.")
    return identities


def validate_cell_identity_coverage(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_name: str,
    candidate_name: str,
    max_examples: int = 8,
) -> None:
    """Require exact ``(ROI, ObjectNumber)`` coverage between two cell tables."""

    reference_ids = _normalise_identity_frame(reference, source=reference_name)
    candidate_ids = _normalise_identity_frame(candidate, source=candidate_name)
    reference_index = pd.MultiIndex.from_frame(reference_ids)
    candidate_index = pd.MultiIndex.from_frame(candidate_ids)

    missing = reference_index.difference(candidate_index)
    unexpected = candidate_index.difference(reference_index)
    if len(missing) == 0 and len(unexpected) == 0:
        return

    missing_counts = (
        pd.Series(missing.get_level_values("ROI"), dtype=str).value_counts().head(8).to_dict()
        if len(missing)
        else {}
    )
    unexpected_counts = (
        pd.Series(unexpected.get_level_values("ROI"), dtype=str).value_counts().head(8).to_dict()
        if len(unexpected)
        else {}
    )
    raise ValueError(
        f"{candidate_name} cell identities do not exactly cover {reference_name}: "
        f"reference={len(reference_index)}, candidate={len(candidate_index)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}; "
        f"missing_by_roi={missing_counts}, unexpected_by_roi={unexpected_counts}; "
        f"missing_examples={list(missing[:max_examples])}, "
        f"unexpected_examples={list(unexpected[:max_examples])}."
    )
