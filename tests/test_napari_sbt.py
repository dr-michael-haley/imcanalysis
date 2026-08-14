from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.napari_sbt.cellpose_adapter import (
    derive_cellpose_specific_features,
    partition_cellpose_features,
)
from SpatialBiologyToolkit.napari_sbt.classifier import (
    high_confidence_queue,
    score_cohort,
    train_multiclass_classifier,
    uncertainty_queue,
)
from SpatialBiologyToolkit.napari_sbt.cohort import (
    cohort_mask,
    resolve_cohort,
    resolve_table_cohort,
)
from SpatialBiologyToolkit.napari_sbt.exports import (
    apply_assignments_to_anndata,
    build_assignment_table,
    export_annotated_anndata,
)
from SpatialBiologyToolkit.napari_sbt.feature_sources import combine_feature_sources
from SpatialBiologyToolkit.napari_sbt.features import (
    _measurement_labels,
    add_distribution_features,
    build_roi_features,
    classifier_seen_mask,
)
from SpatialBiologyToolkit.napari_sbt.labels import (
    empty_labels,
    remove_all_proposed_labels,
    remove_proposed_label,
    set_label,
)
from SpatialBiologyToolkit.napari_sbt.models import (
    CellScope,
    ExperimentManifest,
    FeatureSource,
    SyntheticFeatureRecipe,
    segmentation_qc_classes,
)
from SpatialBiologyToolkit.napari_sbt.storage import (
    dataframe_sha256,
    load_experiment,
    save_experiment,
)
from SpatialBiologyToolkit.napari_sbt.worker import (
    FeatureBuildCancelled,
    run_feature_build,
)

ad = pytest.importorskip("anndata")


def _adata():
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["r1", "r1", "r1", "r2"]),
            "ObjectNumber": [1, 2, 3, 1],
            "leiden": pd.Categorical(["0", "1", "1", "1"]),
        },
        index=["a", "b", "c", "d"],
    )
    return ad.AnnData(np.zeros((4, 2), dtype=np.float32), obs=obs)


def _mask():
    return np.array(
        [
            [0, 1, 1, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )


def test_obs_cohort_is_frozen_by_identity_and_masks_keep_original_ids():
    preview = resolve_cohort(
        _adata(),
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        mode="obs_values",
        obs_column="leiden",
        obs_values=["1"],
    )
    assert preview.eligible_cell_count == 3
    assert preview.represented_roi_count == 2
    assert set(map(tuple, preview.eligible_cells[["ROI", "ObjectNumber"]].values)) == {
        ("r1", 2),
        ("r1", 3),
        ("r2", 1),
    }
    restricted = cohort_mask(_mask(), [2])
    assert set(np.unique(restricted)) == {0, 2}


def test_numeric_observation_is_not_silently_treated_as_a_category():
    data = _adata()
    data.obs["numeric"] = [0, 1, 1, 1]
    with pytest.raises(TypeError, match="numeric, not categorical"):
        resolve_cohort(
            data,
            roi_obs="ROI",
            object_id_obs="ObjectNumber",
            mode="obs_values",
            obs_column="numeric",
            obs_values=["1"],
        )


def test_all_cells_standalone_requires_explicit_unique_identities():
    table = pd.DataFrame(
        {"sample": ["r1", "r1"], "label": [4, 9], "feature": [0.1, 0.2]}
    )
    preview = resolve_table_cohort(
        table, roi_column="sample", object_id_column="label"
    )
    assert preview.eligible_cell_count == 2
    duplicated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        resolve_table_cohort(
            duplicated, roi_column="sample", object_id_column="label"
        )


def test_positive_offset_uses_full_segmentation_and_output_is_cohort_only():
    mask = _mask()
    image = np.arange(mask.size, dtype=np.float32).reshape(mask.shape)
    result = build_roi_features(
        roi="r1",
        full_mask=mask,
        eligible_ids={1},
        channel_images={"CD3": image},
        recipe=SyntheticFeatureRecipe(
            channels=["CD3"],
            mask_offset_px=2,
            region_features=False,
            distribution_features=True,
        ),
    )
    assert result.table["ObjectNumber"].tolist() == [1]
    # Object 1 cannot expand through object 2, and context still sees every object.
    assert result.table.loc[0, "roi_total_object_count"] == 3
    assert result.table.loc[0, "channel::CD3::pixel_count"] < mask.size
    expanded, _ = _measurement_labels(mask, {1}, 3)
    assert np.array_equal(expanded[mask == 2], mask[mask == 2])


def test_classifier_seen_mask_matches_offset_overlap_and_background_recipe():
    mask = _mask()
    blocked = classifier_seen_mask(
        mask,
        {1},
        SyntheticFeatureRecipe(
            mask_offset_px=3,
            allow_positive_offset_overlap=False,
            region_features=True,
            background_ring_px=1,
        ),
    )
    overlapping = classifier_seen_mask(
        mask,
        {1},
        SyntheticFeatureRecipe(
            mask_offset_px=3,
            allow_positive_offset_overlap=True,
            region_features=True,
            background_ring_px=1,
        ),
    )

    assert blocked.dtype == bool
    assert blocked[mask == 1].all()
    assert not blocked[mask == 2].any()
    assert overlapping[mask == 2].any()
    ring_seen = classifier_seen_mask(
        mask,
        {1},
        SyntheticFeatureRecipe(
            mask_offset_px=0,
            region_features=True,
            background_ring_px=1,
        ),
    )
    assert ring_seen[2, 1]  # background-ring context is kept visible


def test_classifier_seen_mask_keeps_original_cell_when_intensity_region_erodes():
    mask = _mask()
    seen = classifier_seen_mask(
        mask,
        {1},
        SyntheticFeatureRecipe(mask_offset_px=-4, region_features=False),
    )
    assert seen[mask == 1].all()
    assert not seen[mask == 2].any()


def test_positive_offset_can_explicitly_overlap_other_cells():
    mask = _mask()
    neighbour_signal = (mask == 2).astype(np.float32)
    blocked = build_roi_features(
        roi="r1",
        full_mask=mask,
        eligible_ids={1},
        channel_images={"CD3": neighbour_signal},
        recipe=SyntheticFeatureRecipe(
            channels=["CD3"],
            mask_offset_px=2,
            region_features=False,
            distribution_features=True,
        ),
    )
    overlapping = build_roi_features(
        roi="r1",
        full_mask=mask,
        eligible_ids={1},
        channel_images={"CD3": neighbour_signal},
        recipe=SyntheticFeatureRecipe(
            channels=["CD3"],
            mask_offset_px=2,
            allow_positive_offset_overlap=True,
            region_features=False,
            distribution_features=True,
        ),
    )
    assert blocked.table.loc[0, "channel::CD3::sum"] == 0
    assert overlapping.table.loc[0, "channel::CD3::sum"] > 0
    assert not blocked.table.loc[0, "measurement_allows_cell_overlap"]
    assert overlapping.table.loc[0, "measurement_allows_cell_overlap"]
    assert overlapping.table.loc[0, "roi_total_object_count"] == 3


def test_negative_offset_records_vanished_cells_without_losing_rows():
    result = build_roi_features(
        roi="r1",
        full_mask=_mask(),
        eligible_ids={1, 2},
        channel_images={"CD3": np.ones(_mask().shape, dtype=np.float32)},
        recipe=SyntheticFeatureRecipe(mask_offset_px=-4),
    )
    assert len(result.table) == 2
    assert result.table["measurement_region_vanished"].all()
    assert result.table["channel::CD3::mean"].isna().all()
    assert set(result.vanished_object_ids) == {1, 2}


def test_image_mask_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="does not match mask shape"):
        build_roi_features(
            roi="r1",
            full_mask=_mask(),
            eligible_ids={1},
            channel_images={"CD3": np.ones((2, 2), dtype=np.float32)},
            recipe=SyntheticFeatureRecipe(),
        )


def test_shared_distribution_helper_preserves_cellpose_column_convention():
    row = {}
    add_distribution_features(
        row, np.array([1.0, 2.0, np.nan, 3.0]), "dna_raw", separator="_"
    )
    assert row["dna_raw_pixel_count"] == 3
    assert row["dna_raw_mean"] == 2.0
    assert row["dna_raw_median"] == 2.0
    assert row["dna_raw_sum"] == 6.0
    assert row["dna_raw_q95"] == pytest.approx(2.9)
    generic, cellpose_only = partition_cellpose_features(
        ["mask_area", "cellpose_flow_error", "cellprob_threshold"]
    )
    assert generic == ["mask_area"]
    assert cellpose_only == ["cellpose_flow_error", "cellprob_threshold"]
    with pytest.raises(RuntimeError, match="cannot be derived"):
        derive_cellpose_specific_features()


def test_feature_sources_filter_before_namespaced_join(tmp_path: Path):
    cohort = pd.DataFrame(
        {
            "obs_name": ["a", "b"],
            "ROI": ["r1", "r1"],
            "ObjectNumber": [1, 2],
        }
    )
    source = pd.DataFrame(
        {
            "ROI": ["r1", "r1", "r1"],
            "ObjectNumber": [1, 2, 99],
            "embedding_0": [0.1, 0.2, 100.0],
        }
    )
    source_path = tmp_path / "embedding.csv"
    source.to_csv(source_path, index=False)
    combined = combine_feature_sources(
        cohort,
        [FeatureSource(source_id="cellvision", kind="table", path=str(source_path))],
    )
    assert len(combined.table) == 2
    assert combined.feature_columns == ["source::cellvision::embedding_0"]
    assert combined.table["source::cellvision::embedding_0"].tolist() == [0.1, 0.2]


def test_cellvision_obsm_is_joined_only_to_frozen_cohort(tmp_path: Path):
    data = _adata()
    data.obsm["X_cellvision"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    source_path = tmp_path / "cellvision.h5ad"
    data.write_h5ad(source_path)
    cohort = pd.DataFrame(
        {"obs_name": ["b", "d"], "ROI": ["r1", "r2"], "ObjectNumber": [2, 1]}
    )
    combined = combine_feature_sources(
        cohort,
        [
            FeatureSource(
                source_id="cellvision",
                kind="anndata",
                path=str(source_path),
                representation="X_cellvision",
            )
        ],
    )
    assert len(combined.table) == 2
    assert combined.table["source::cellvision::X_cellvision_0"].tolist() == [3, 9]


def _training_inputs():
    cohort = pd.DataFrame(
        {
            "obs_name": [f"c{value}" for value in range(9)],
            "ROI": ["r1"] * 6 + ["r2"] * 3,
            "ObjectNumber": list(range(1, 7)) + [1, 2, 3],
        }
    )
    features = cohort[["ROI", "ObjectNumber"]].copy()
    features["f1"] = [0, 0.1, 1, 1.1, 2, 2.1, 0.2, 1.2, np.nan]
    features["f2"] = [0.1, 0, 1.1, 1, 2.1, 2, 0.1, 1.1, np.nan]
    features["all_missing"] = np.nan
    labels = empty_labels()
    for object_number, class_id in [
        (1, "a"),
        (2, "a"),
        (3, "b"),
        (4, "b"),
        (5, "c"),
        (6, "c"),
    ]:
        labels = set_label(
            labels,
            roi="r1",
            object_number=object_number,
            class_id=class_id,
            state="confirmed",
        )
    return cohort, features, labels


def test_remove_proposed_label_preserves_confirmed_labels():
    labels = set_label(
        empty_labels(),
        roi="r1",
        object_number=1,
        class_id="a",
        state="proposed",
    )
    labels = set_label(
        labels,
        roi="r1",
        object_number=2,
        class_id="b",
        state="confirmed",
    )

    labels = remove_proposed_label(labels, roi="r1", object_number=1)
    assert labels[["ROI", "ObjectNumber", "class_id", "state"]].to_dict(
        "records"
    ) == [
        {
            "ROI": "r1",
            "ObjectNumber": 2,
            "class_id": "b",
            "state": "confirmed",
        }
    ]

    unchanged = remove_proposed_label(labels, roi="r1", object_number=2)
    pd.testing.assert_frame_equal(unchanged, labels)


def test_remove_all_proposed_labels_preserves_every_confirmation():
    labels = empty_labels()
    for object_number, state in ((1, "proposed"), (2, "confirmed"), (3, "proposed")):
        labels = set_label(
            labels,
            roi="r1",
            object_number=object_number,
            class_id="a",
            state=state,
        )

    cleared = remove_all_proposed_labels(labels)

    assert cleared[["ROI", "ObjectNumber", "state"]].to_dict("records") == [
        {"ROI": "r1", "ObjectNumber": 2, "state": "confirmed"}
    ]


def test_multiclass_scores_probabilities_and_unscorable_cells():
    cohort, features, labels = _training_inputs()
    training = train_multiclass_classifier(
        features,
        labels,
        class_ids=["a", "b", "c"],
        cohort=cohort,
    )
    assert training.ok, training.errors
    scores = score_cohort(training.bundle, features)
    probabilities = scores[
        ["probability::a", "probability::b", "probability::c"]
    ]
    assert np.allclose(probabilities.iloc[:-1].sum(axis=1), 1)
    assert not scores.iloc[-1]["scorable"]
    assert pd.isna(scores.iloc[-1]["predicted_class"])
    assert not uncertainty_queue(scores, labels).empty
    predicted = scores.iloc[6]["predicted_class"]
    assert isinstance(
        high_confidence_queue(
            scores,
            labels,
            class_id=predicted,
            threshold=0,
        ),
        pd.DataFrame,
    )


def test_classifier_requires_two_confirmed_examples_per_class():
    cohort, features, labels = _training_inputs()
    labels = labels.drop(labels.index[-1]).reset_index(drop=True)
    training = train_multiclass_classifier(
        features,
        labels,
        class_ids=["a", "b", "c"],
        cohort=cohort,
    )
    assert not training.ok
    assert "at least two" in training.errors[0]


def test_confirmed_labels_override_model_assignments():
    cohort, _features, labels = _training_inputs()
    scores = pd.DataFrame(
        {
            "ROI": cohort["ROI"],
            "ObjectNumber": cohort["ObjectNumber"],
            "predicted_class": ["b"] * len(cohort),
            "maximum_probability": [0.9] * len(cohort),
            "probability_margin": [0.8] * len(cohort),
            "normalized_entropy": [0.1] * len(cohort),
            "scorable": [True] * len(cohort),
            "model_id": ["model"] * len(cohort),
            "probability::a": [0.1] * len(cohort),
            "probability::b": [0.9] * len(cohort),
            "probability::c": [0.0] * len(cohort),
        }
    )
    assignments = build_assignment_table(
        cohort, labels, scores, class_ids=["a", "b", "c"]
    )
    assert assignments.iloc[0]["class_id"] == "a"
    assert assignments.iloc[0]["assignment_source"] == "confirmed"
    assert assignments.iloc[6]["class_id"] == "b"
    assert assignments.iloc[6]["assignment_source"] == "model"


def test_final_assignment_thresholds_reject_model_without_hiding_raw_prediction():
    cohort, _features, labels = _training_inputs()
    scores = pd.DataFrame(
        {
            "ROI": cohort["ROI"],
            "ObjectNumber": cohort["ObjectNumber"],
            "predicted_class": ["b"] * len(cohort),
            "maximum_probability": [0.75] * len(cohort),
            "probability_margin": [0.25] * len(cohort),
            "normalized_entropy": [0.4] * len(cohort),
            "scorable": [True] * len(cohort),
            "model_id": ["model"] * len(cohort),
        }
    )

    assignments = build_assignment_table(
        cohort,
        labels,
        scores,
        class_ids=["a", "b", "c"],
        minimum_model_confidence=0.8,
        maximum_model_uncertainty=0.5,
        minimum_probability_margin=0.2,
    )

    assert assignments.iloc[0]["assignment_source"] == "confirmed"
    assert assignments.iloc[0]["class_id"] == "a"
    assert assignments.iloc[6]["predicted_class"] == "b"
    assert assignments.iloc[6]["assignment_source"] == "unassigned"
    assert pd.isna(assignments.iloc[6]["class_id"])
    assert (
        assignments.iloc[6]["prediction_rejection_reason"]
        == "below_minimum_confidence"
    )


def test_annotated_copy_keeps_noncohort_population_and_nan_probabilities(tmp_path: Path):
    data = _adata()
    source = tmp_path / "source.h5ad"
    destination = tmp_path / "annotated.h5ad"
    data.write_h5ad(source)
    preview = resolve_cohort(
        data,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        mode="obs_values",
        obs_column="leiden",
        obs_values=["1"],
    )
    scope = preview.scope(
        mode="obs_values", obs_column="leiden", obs_values=["1"]
    )
    manifest = ExperimentManifest(
        name="T-cell subclasses",
        anndata_path=str(source),
        masks_folder=str(tmp_path / "masks"),
        cell_scope=scope,
        classes=segmentation_qc_classes(),
    )
    assignments = preview.eligible_cells.copy()
    assignments["class_id"] = ["good", "artifact", pd.NA]
    assignments["assignment_source"] = ["confirmed", "model", "unassigned"]
    assignments["confidence"] = [1.0, 0.95, np.nan]
    assignments["uncertainty"] = [0.0, 0.1, np.nan]
    assignments["probability::good"] = [1.0, 0.05, np.nan]
    assignments["probability::artifact"] = [0.0, 0.95, np.nan]
    export_annotated_anndata(source, destination, assignments, manifest)
    exported = ad.read_h5ad(destination)
    slug = manifest.output_obs_slug
    assert pd.isna(exported.obs.loc["a", f"{slug}_subclass"])
    assert exported.obs.loc["a", f"{slug}_combined"] == "0"
    assert exported.obs.loc["b", f"{slug}_combined"] == "good"
    assert np.isnan(exported.obsm[f"{slug}_probabilities"][0]).all()


def test_apply_assignments_to_live_anndata_does_not_require_a_disk_write(tmp_path: Path):
    data = _adata()
    preview = resolve_cohort(
        data,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        mode="obs_values",
        obs_column="leiden",
        obs_values=["1"],
    )
    manifest = ExperimentManifest(
        name="Live subclasses",
        masks_folder=str(tmp_path / "masks"),
        cell_scope=preview.scope(
            mode="obs_values", obs_column="leiden", obs_values=["1"]
        ),
        classes=segmentation_qc_classes(),
    )
    assignments = preview.eligible_cells.copy()
    assignments["class_id"] = ["good", "artifact", pd.NA]
    assignments["assignment_source"] = ["confirmed", "model", "unassigned"]
    assignments["confidence"] = [1.0, 0.95, np.nan]
    assignments["uncertainty"] = [0.0, 0.1, np.nan]
    assignments["probability::good"] = [1.0, 0.05, np.nan]
    assignments["probability::artifact"] = [0.0, 0.95, np.nan]

    returned = apply_assignments_to_anndata(
        data,
        assignments,
        manifest,
        metrics={"final_identity_decision": {"minimum_model_confidence": 0.9}},
    )

    slug = manifest.output_obs_slug
    assert returned is data
    assert data.obs.loc["b", f"{slug}_subclass"] == "good"
    assert data.uns["napari_sbt"][slug]["metrics"][
        "final_identity_decision"
    ]["minimum_model_confidence"] == 0.9


def test_manifest_rejects_duplicate_shortcuts():
    scope = CellScope(
        mode="all_cells",
        snapshot_sha256="abc",
        eligible_cell_count=2,
        total_cell_count=2,
        represented_roi_count=1,
    )
    classes = segmentation_qc_classes()
    classes[1].shortcut = "1"
    with pytest.raises(ValueError, match="shortcuts"):
        ExperimentManifest(
            name="bad shortcuts",
            masks_folder="masks",
            cell_scope=scope,
            classes=classes,
        )


def test_worker_builds_and_resumes_cohort_only_roi_fragments(tmp_path: Path):
    import tifffile

    masks = tmp_path / "masks"
    images = tmp_path / "images"
    masks.mkdir()
    (images / "r1").mkdir(parents=True)
    tifffile.imwrite(masks / "r1.tiff", _mask())
    tifffile.imwrite(
        images / "r1" / "CD3.tiff",
        np.arange(_mask().size, dtype=np.float32).reshape(_mask().shape),
    )
    normalization = tmp_path / "normalization_dict.csv"
    normalization.write_text(
        "marker,vmax,lower_threshold\nCD3,10.0,0.5\n", encoding="utf-8"
    )
    preview = resolve_cohort(
        _adata(),
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        mode="obs_values",
        obs_column="leiden",
        obs_values=["1"],
    )
    # Keep one represented ROI so missing r2 assets do not obscure resume behavior.
    eligible = preview.eligible_cells.loc[
        preview.eligible_cells["ROI"].astype(str).eq("r1")
    ].reset_index(drop=True)
    preview.eligible_cells = eligible
    preview.per_roi_counts = pd.DataFrame({"ROI": ["r1"], "eligible_cells": [2]})
    root = tmp_path / "experiment"
    snapshot = root / "cohort" / "eligible_cells.parquet"
    snapshot.parent.mkdir(parents=True)
    eligible.to_parquet(snapshot, index=False)
    scope = preview.scope(
        mode="obs_values", obs_column="leiden", obs_values=["1"]
    )
    manifest = ExperimentManifest(
        name="worker test",
        anndata_path=str(tmp_path / "source.h5ad"),
        images_folders=[str(images)],
        masks_folder=str(masks),
        cell_scope=scope,
        classes=segmentation_qc_classes(),
        synthetic_features=SyntheticFeatureRecipe(
            channels=["CD3"],
            region_features=False,
            normalization_dict_path=str(normalization),
        ),
    )
    save_experiment(manifest, root)
    def request_cancel(event):
        if event["event"] == "build_started":
            (root / "logs" / "feature_build.cancel").write_text(
                "cancel\n", encoding="utf-8"
            )

    with pytest.raises(FeatureBuildCancelled):
        run_feature_build(root, workers=1, progress=request_cancel)
    first = run_feature_build(root, workers=1, progress=lambda _event: None)
    table = pd.read_parquet(first.feature_table)
    assert set(table["ObjectNumber"]) == {2, 3}
    assert table.loc[
        table["ObjectNumber"].eq(2),
        "source::imc::channel::CD3::mean",
    ].iloc[0] == pytest.approx(0.8)
    assert first.skipped_rois == 1
    second = run_feature_build(root, workers=1, progress=lambda _event: None)
    assert second.skipped_rois == 1
    reloaded, _ = load_experiment(root)
    assert reloaded.active_feature_set_id


def test_frozen_cohort_change_requires_explicit_revision(tmp_path: Path):
    first_snapshot = pd.DataFrame(
        {
            "obs_name": ["a", "b"],
            "ROI": ["r1", "r1"],
            "ObjectNumber": [1, 2],
        }
    )
    first_hash = dataframe_sha256(
        first_snapshot, ["obs_name", "ROI", "ObjectNumber"]
    )
    scope = CellScope(
        mode="all_cells",
        snapshot_sha256=first_hash,
        eligible_cell_count=2,
        total_cell_count=2,
        represented_roi_count=1,
    )
    manifest = ExperimentManifest(
        name="immutable cohort",
        masks_folder="masks",
        cell_scope=scope,
        classes=segmentation_qc_classes(),
    )
    (tmp_path / "cohort").mkdir()
    first_snapshot.to_parquet(
        tmp_path / "cohort" / "eligible_cells.parquet", index=False
    )
    save_experiment(manifest, tmp_path)
    changed = manifest.model_copy(deep=True)
    changed.cell_scope.snapshot_sha256 = "second"
    with pytest.raises(ValueError, match="explicit revision"):
        save_experiment(changed, tmp_path)
    changed.revision = 2
    second_snapshot = first_snapshot.iloc[[0]].copy()
    second_path = tmp_path / "cohort" / "eligible_cells_r2.parquet"
    second_snapshot.to_parquet(second_path, index=False)
    changed.cell_scope.snapshot_path = "cohort/eligible_cells_r2.parquet"
    changed.cell_scope.snapshot_sha256 = dataframe_sha256(
        second_snapshot, ["obs_name", "ROI", "ObjectNumber"]
    )
    changed.cell_scope.eligible_cell_count = 1
    save_experiment(changed, tmp_path)
    assert (tmp_path / "revisions" / "experiment_r1.yaml").exists()
