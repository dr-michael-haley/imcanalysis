from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.napari_sbt.labeler import (
    LabelerClass,
    apply_labeler_to_anndata,
    build_labeler_export_table,
    empty_labeler_records,
    labeler_summary,
    remove_labeler_record,
    set_labeler_record,
    validate_labeler_records,
)

ad = pytest.importorskip("anndata")


def _definitions():
    return [
        LabelerClass(label_id="target", name="Target", color="#e11d48"),
        LabelerClass(label_id="control", name="Control", color="#2563eb"),
    ]


def _cohort():
    return pd.DataFrame(
        {
            "obs_name": ["a", "b", "c", "d"],
            "ROI": ["r1", "r1", "r2", "r3"],
            "ObjectNumber": [1, 2, 7, 4],
        }
    )


def _adata():
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["r1", "r1", "r2", "r3", "outside"]),
            "ObjectNumber": [1, 2, 7, 4, 1],
        },
        index=["a", "b", "c", "d", "e"],
    )
    return ad.AnnData(np.zeros((5, 1), dtype=np.float32), obs=obs)


def test_labeler_assignments_replace_and_clear_one_cell():
    records = set_labeler_record(
        empty_labeler_records(),
        roi="r1",
        object_number=1,
        label_id="target",
    )
    records = set_labeler_record(
        records,
        roi="r1",
        object_number=1,
        label_id="control",
    )

    assert len(records) == 1
    assert records.iloc[0]["label_id"] == "control"
    assert remove_labeler_record(records, roi="r1", object_number=1).empty


def test_labeler_summary_tracks_cells_and_roi_coverage_per_label():
    records = empty_labeler_records()
    for roi, object_number, label_id in (
        ("r1", 1, "target"),
        ("r1", 2, "target"),
        ("r2", 7, "target"),
        ("r3", 4, "control"),
    ):
        records = set_labeler_record(
            records,
            roi=roi,
            object_number=object_number,
            label_id=label_id,
        )

    summary = labeler_summary(
        records, _definitions(), eligible_rois=["r1", "r2", "r3"]
    ).set_index("label_id")

    assert summary.loc["target", "cells"] == 3
    assert summary.loc["target", "rois_sampled"] == 2
    assert summary.loc["target", "eligible_rois"] == 3
    assert summary.loc["control", "cells"] == 1
    assert summary.loc["control", "rois_sampled"] == 1


def test_labeler_rejects_cells_outside_the_frozen_cohort():
    records = set_labeler_record(
        empty_labeler_records(),
        roi="unknown",
        object_number=99,
        label_id="target",
    )
    with pytest.raises(ValueError, match="frozen experiment cohort"):
        validate_labeler_records(
            records,
            label_ids=["target", "control"],
            cohort=_cohort(),
        )


def test_labeler_export_and_live_anndata_application_preserve_identity():
    records = set_labeler_record(
        empty_labeler_records(),
        roi="r2",
        object_number=7,
        label_id="target",
    )
    records = set_labeler_record(
        records,
        roi="r1",
        object_number=2,
        label_id="control",
    )
    table = build_labeler_export_table(
        records, _definitions(), cohort=_cohort()
    )

    assert set(table["obs_name"]) == {"b", "c"}
    assert dict(zip(table["obs_name"], table["label"], strict=True)) == {
        "b": "Control",
        "c": "Target",
    }

    adata = _adata()
    apply_labeler_to_anndata(
        adata,
        records,
        _definitions(),
        cohort=_cohort(),
        obs_name="picked_cells",
    )

    assert adata.obs.loc["b", "picked_cells"] == "Control"
    assert adata.obs.loc["c", "picked_cells"] == "Target"
    assert pd.isna(adata.obs.loc["a", "picked_cells"])
    assert pd.isna(adata.obs.loc["e", "picked_cells"])
    assert list(adata.obs["picked_cells"].cat.categories) == ["Target", "Control"]

    with pytest.raises(ValueError, match="already exists"):
        apply_labeler_to_anndata(
            adata,
            records,
            _definitions(),
            cohort=_cohort(),
            obs_name="picked_cells",
        )
