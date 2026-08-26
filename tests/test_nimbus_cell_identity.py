from __future__ import annotations

import pandas as pd
import pytest

from SpatialBiologyToolkit.nimbus_cell_identity import (
    validate_cell_identity_coverage,
    validate_object_number_coverage,
)


def test_object_number_coverage_accepts_reordered_exact_labels() -> None:
    validate_object_number_coverage(
        [1, 2, 3, 4],
        [4, 2, 1, 3],
        context="synthetic ROI",
    )


def test_object_number_coverage_reports_trailing_missing_labels() -> None:
    with pytest.raises(ValueError, match=r"missing=3.*missing_examples=\[5, 6, 7\]"):
        validate_object_number_coverage(
            range(1, 8),
            range(1, 5),
            context="Nimbus aggregation for ROI 'roi-a'",
        )


def test_cell_identity_coverage_accepts_exact_keys_in_different_order() -> None:
    masks = pd.DataFrame({"ROI": ["a", "a", "b"], "ObjectNumber": [1, 2, 1]})
    scores = pd.DataFrame({"ROI": ["b", "a", "a"], "ObjectNumber": [1, 2, 1]})

    validate_cell_identity_coverage(
        masks,
        scores,
        reference_name="masks",
        candidate_name="scores",
    )


def test_cell_identity_coverage_reports_missing_cells_by_roi() -> None:
    masks = pd.DataFrame({"ROI": ["a", "a", "b", "b"], "ObjectNumber": [1, 2, 1, 2]})
    scores = pd.DataFrame({"ROI": ["a", "b"], "ObjectNumber": [1, 1]})

    with pytest.raises(ValueError, match=r"missing=2.*missing_by_roi=.*'a': 1.*'b': 1"):
        validate_cell_identity_coverage(
            masks,
            scores,
            reference_name="masks",
            candidate_name="scores",
        )


def test_cell_identity_coverage_rejects_duplicate_score_keys() -> None:
    masks = pd.DataFrame({"ROI": ["a"], "ObjectNumber": [1]})
    scores = pd.DataFrame({"ROI": ["a", "a"], "ObjectNumber": [1, 1]})

    with pytest.raises(ValueError, match="duplicate cell identities"):
        validate_cell_identity_coverage(
            masks,
            scores,
            reference_name="masks",
            candidate_name="scores",
        )
