from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile

from SpatialBiologyToolkit import population_refinement as pr


def fixture_data():
    rows = []
    for case in range(3):
        for roi in range(case + 1):
            for cell in range(12):
                rows.append(
                    dict(
                        ROI=f"r{case}_{roi}",
                        Case=f"c{case}",
                        ObjectNumber=cell + 1,
                        parent="mixed" if cell < 10 else "outside",
                        label="A" if cell % 2 else "B",
                        score=cell / 12 - 0.3,
                    )
                )
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    obs["parent"] = pd.Categorical(obs.parent)
    data = ad.AnnData(
        np.column_stack([obs.label.eq("A"), obs.label.eq("B")]).astype(float),
        obs=obs,
        var=pd.DataFrame(index=["M1", "M2"]),
    )
    spec = pr.SplitSpec(
        population_key="parent",
        populations=("mixed",),
        classes=(
            pr.ClassSpec(class_id="a", name="A cells", source_labels=("A",)),
            pr.ClassSpec(class_id="b", name="B cells", source_labels=("B",)),
        ),
    )
    candidates = pr.MaxFuseSource(label_key="label", score_key="score").candidates(
        data, spec
    )
    sampling = pr.SamplingSpec(
        per_class=12,
        max_per_stratum=8,
        minimum_per_class=4,
        minimum_groups_per_class=2,
        seed=4,
    )
    features = candidates[["ROI", "ObjectNumber"]].copy()
    features["channel::M1::mean"] = candidates.class_id.eq("a").astype(float).to_numpy()
    features["channel::M2::mean"] = candidates.class_id.eq("b").astype(float).to_numpy()
    return data, spec, candidates, sampling, features


def test_partial_sources_join_by_id_and_keep_minority_unlabelled():
    data, spec, _, _, _ = fixture_data()
    sidecar = data.obs[["label", "score"]].iloc[::-2].copy()
    sidecar.iloc[0, 0] = "rare"
    result = pr.MaxFuseSource(
        label_key="label", score_key="score", table=sidecar
    ).candidates(data, spec)
    assert len(result) == 60
    assert not result.loc[~result.source_present, "eligible"].any()
    assert not result.loc[
        result.source_label.eq("rare").fillna(False), "eligible"
    ].any()
    present = result.loc[result.source_present].set_index("obs_name")
    pd.testing.assert_series_equal(
        present.source_score, sidecar.score.reindex(present.index), check_names=False
    )
    with pytest.raises(ValueError, match="unique"):
        pr.MaxFuseSource(
            label_key="label", score_key="score", table=pd.concat([sidecar, sidecar])
        ).candidates(data, spec)


def test_marker_roles_are_independent_and_audited():
    _, _, candidates, _, _ = fixture_data()
    evidence = pd.DataFrame({"hard": 1.0, "soft": 0.0}, index=candidates.obs_name)
    a = candidates.loc[candidates.class_id.eq("a"), "obs_name"].iloc[:2]
    evidence.loc[a.iloc[0], "hard"] = 0
    evidence.loc[a.iloc[1], "soft"] = 1
    rules = (
        pr.MarkerRule(class_id="a", feature="hard", role="required", minimum=0.5),
        pr.MarkerRule(class_id="a", feature="soft", role="supportive", minimum=0.5),
        pr.MarkerRule(class_id="b", feature="absent_image", role="feature_only"),
    )
    result, checks = pr.assess_markers(candidates, evidence, rules)
    by_id = result.set_index("obs_name")
    assert not by_id.loc[a.iloc[0], "eligible"]
    assert by_id.loc[a.iloc[1], "eligible"]
    assert by_id.loc[a.iloc[1], "support_score"] == 1
    assert by_id.loc[candidates.class_id.eq("b").to_numpy(), "eligible"].all()
    assert set(checks.feature) == {"hard", "soft"}


def test_hierarchical_sampling_balances_cases_not_roi_counts():
    _, spec, candidates, sampling, _ = fixture_data()
    result = pr.select_exemplars(
        candidates, class_ids=spec.class_ids, sampling=sampling
    )
    counts = result.examples.groupby(["class_id", "Case"]).size()
    assert set(counts) == {4}
    assert not result.examples.obs_name.duplicated().any()
    repeated = pr.select_exemplars(
        candidates.sample(frac=1, random_state=7),
        class_ids=spec.class_ids,
        sampling=sampling,
    )
    assert result.examples.obs_name.tolist() == repeated.examples.obs_name.tolist()
    assert not result.examples.human_confirmed.any()


def test_weak_or_missing_classes_fail_without_relaxing_selection():
    _, spec, candidates, sampling, features = fixture_data()
    sampling = sampling.model_copy(update={"minimum_score": 0.4})
    result = pr.select_exemplars(
        candidates, class_ids=spec.class_ids, sampling=sampling
    )
    assert result.warnings
    with pytest.raises(ValueError, match="Insufficient evidence"):
        pr.fit_refinement(
            features, result, spec, feature_columns=list(features.columns[2:])
        )


def test_class_quality_pool_does_not_recruit_weak_strata():
    _, spec, candidates, sampling, _ = fixture_data()
    candidates["source_score"] = np.where(candidates.Case.eq("c2"), 10.0, 1.0)
    strict = sampling.model_copy(update={
        "candidate_fraction": 0.4, "quality_pool_scope": "class",
    })
    selected = pr.select_exemplars(candidates, class_ids=spec.class_ids, sampling=strict)
    assert set(selected.examples.Case) == {"c2"}
    assert selected.warnings  # Representation cannot override exemplar quality.
    assert not selected.candidates.loc[selected.candidates.Case.ne("c2"), "eligible"].any()


def test_complete_assignment_retains_review_flags_and_rejects_missing_rows():
    _, spec, candidates, sampling, features = fixture_data()
    selection = pr.select_exemplars(candidates, class_ids=spec.class_ids, sampling=sampling)
    held = features.index[~candidates.obs_name.isin(selection.examples.obs_name)][0]
    features.loc[held, list(features.columns[2:])] = 100
    policy = pr.AssignmentPolicy(assignment_mode="complete")
    result = pr.fit_refinement(features, selection, spec,
                               feature_columns=list(features.columns[2:]), policy=policy)
    assert result.assignments.class_id.notna().all()
    row = result.assignments.iloc[held]
    assert row.assignment_needs_review
    assert row.assignment_source == "model_forced_review"
    assert row.prediction_rejection_reason == "outside_training_support"
    assert not row.model_prediction_accepted
    from SpatialBiologyToolkit.population_refinement.workflow import assign_predictions
    with pytest.raises(ValueError, match="Complete assignment requires"):
        assign_predictions(result.bundle, features.drop(index=held),
                           candidates[["obs_name", "ROI", "ObjectNumber", "source_population"]],
                           policy=policy)


def test_fit_rejects_unsupported_cells_and_preserves_source_labels():
    data, spec, candidates, sampling, features = fixture_data()
    selection = pr.select_exemplars(
        candidates, class_ids=spec.class_ids, sampling=sampling
    )
    held = features.index[~candidates.obs_name.isin(selection.examples.obs_name)][0]
    features.loc[held, list(features.columns[2:])] = 100
    result = pr.fit_refinement(
        features, selection, spec, feature_columns=list(features.columns[2:])
    )
    assert (
        result.assignments.loc[held, "prediction_rejection_reason"]
        == "outside_training_support"
    )
    assert set(result.assignments.assignment_source) <= {"model", "unassigned"}
    old = data.obs.copy(deep=True)
    integrated = result.integrate(data)
    pd.testing.assert_frame_equal(old, data.obs)
    retained = integrated.loc[
        ~integrated.is_classification_cohort, "refined_population"
    ]
    assert set(retained) == {"outside"}
    assert (
        integrated.set_index("obs_name").loc[
            candidates.loc[held, "obs_name"], "refined_population"
        ]
        == "mixed"
    )
    result.integrate(data, apply=True)
    assert "refined_population" in data.obs
    with pytest.raises(ValueError, match="already exists"):
        result.integrate(data, apply=True)


def test_exemplar_labels_do_not_override_a_rejected_model():
    _, spec, candidates, sampling, features = fixture_data()
    selection = pr.select_exemplars(
        candidates, class_ids=spec.class_ids, sampling=sampling
    )
    # All inputs constant: model must be uncertain even on its training examples.
    features.iloc[:, 2:] = 0
    result = pr.fit_refinement(
        features,
        selection,
        spec,
        feature_columns=list(features.columns[2:]),
        policy=pr.AssignmentPolicy(minimum_probability=0.99),
    )
    assert result.assignments.class_id.isna().all()
    assert not result.assignments.assignment_source.eq("confirmed").any()


def test_grouped_validation_keeps_cases_separate_and_exports(tmp_path):
    _, spec, candidates, sampling, features = fixture_data()
    selection = pr.select_exemplars(
        candidates, class_ids=spec.class_ids, sampling=sampling
    )
    evaluated = pr.evaluate_refinement(
        features,
        selection,
        spec,
        feature_sets={"means": list(features.columns[2:])},
        model_types=("random_forest",),
    )
    for fold in evaluated.folds.itertuples():
        assert fold.held_out_group not in fold.training_groups.split(";")
    assert evaluated.metrics.balanced_accuracy.min() == 1
    result = pr.fit_refinement(
        features, selection, spec, feature_columns=list(features.columns[2:])
    )
    writer = result.save(tmp_path)
    evaluated.save(writer)
    assert (tmp_path / "models/refinement.joblib").exists()
    assert (tmp_path / "manifests/artifacts.csv").exists()


def test_complete_validation_does_not_count_forced_labels_as_accepted():
    _, spec, candidates, sampling, features = fixture_data()
    selection = pr.select_exemplars(candidates, class_ids=spec.class_ids, sampling=sampling)
    columns = list(features.columns[2:])
    features.loc[candidates.Case.eq("c0"), columns] += 100
    evaluated = pr.evaluate_refinement(
        features, selection, spec, feature_sets={"means": columns},
        model_types=("random_forest",), held_out_groups=["c0"],
        policy=pr.AssignmentPolicy(assignment_mode="complete"),
    )
    assert evaluated.metrics.scorable_fraction.iloc[0] == 1
    assert evaluated.metrics.accepted_fraction.iloc[0] == 0


def test_manual_source_and_reference_marker_mapping():
    data, spec, _, _, _ = fixture_data()
    source = pr.TableSource(
        pd.DataFrame({"class_id": data.obs.label.map({"A": "a", "B": "b"})})
    )
    assert source.candidates(data, spec).eligible.all()
    ranked = pr.rank_reference_markers(
        data,
        spec,
        label_key="label",
        available_channels=["M1", "M2"],
        mapping=pd.DataFrame(
            {"snRNAseq": ["M1", "M2", "missing"], "IMC": ["M1", "M2", "unused"]}
        ),
    )
    assert ranked.iloc[0].channel in {"M1", "M2"}
    assert set(ranked.channel) == {"M1", "M2"}
    assert (
        ranked.query("class_id == 'a' and channel == 'M1'").mean_difference.iloc[0] > 0
    )


def test_live_anndata_feature_workspace_reuses_full_masks_and_resumes(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    data, spec, _, _, _ = fixture_data()
    for roi in data.obs.ROI.unique():
        mask = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
        tifffile.imwrite(masks / f"{roi}.tiff", mask)
        roi_images = images / roi
        roi_images.mkdir()
        tifffile.imwrite(roi_images / "M1.tiff", (mask % 2).astype(np.float32))
    recipe = pr.image_recipe(["M1"])
    recipe = recipe.model_copy(update={"region_features": False})
    experiment = pr.create_feature_experiment(
        data,
        spec,
        directory=tmp_path / "experiment",
        images=[images],
        masks=masks,
        recipe=recipe,
    )
    table = pr.build_feature_table(experiment, workers=1, progress=lambda event: None)
    assert len(table) == 60
    assert len(pr.image_feature_columns(table, means_only=True)) == 1
    assert (
        pr.create_feature_experiment(
            data,
            spec,
            directory=experiment,
            images=[images],
            masks=masks,
            recipe=recipe,
        )
        == experiment
    )
    events = []
    again = pr.build_feature_table(experiment, workers=1, progress=events.append)
    pd.testing.assert_frame_equal(table, again)
    assert events[-1]["resumed_rois"] == 6


def test_old_imports_resolve_to_shared_implementation():
    from SpatialBiologyToolkit.napari_sbt.classifier import train_from_examples
    from SpatialBiologyToolkit.cell_classification.classifier import (
        train_from_examples as shared,
    )

    assert train_from_examples is shared
    assert not any(name in sys.modules for name in ("qtpy", "napari"))


def test_nested_feature_refinement_runs_only_inside_outer_training_groups():
    _, spec, candidates, sampling, features = fixture_data()
    selection = pr.select_exemplars(
        candidates, class_ids=spec.class_ids, sampling=sampling
    )
    evaluated = pr.evaluate_refinement(
        features,
        selection,
        spec,
        feature_sets={"image": list(features.columns[2:])},
        model_types=("random_forest",),
        refine_features=True,
        recommendation_count=1,
        held_out_groups=["c0"],
    )
    assert len(evaluated.metrics) == 1
    assert evaluated.metrics.feature_count.iloc[0] == 1
    assert evaluated.folds.training_groups.iloc[0] == "c1;c2"


@pytest.mark.parametrize("strategy", ["uniform", "top_ranked", "rank_weighted"])
def test_sampling_caps_and_score_pool_never_duplicate_cells(strategy):
    _, spec, candidates, sampling, _ = fixture_data()
    sampling = sampling.model_copy(
        update={
            "strategy": strategy,
            "candidate_fraction": 0.5,
            "max_per_stratum": 1,
            "allocation": "proportional",
        }
    )
    selection = pr.select_exemplars(
        candidates, class_ids=spec.class_ids, sampling=sampling
    )
    assert selection.coverage.selected.max() <= 1
    assert not selection.examples.obs_name.duplicated().any()
    assert selection.examples.score_rank.min() >= 0.5


def test_namespaced_identity_columns_are_not_default_image_predictors():
    features = pd.DataFrame(
        columns=[
            "source::imc::X_loc",
            "source::imc::channel::IBA1::weighted_x",
            "source::imc::channel::IBA1::mean",
            "source::imc::mask_area",
        ]
    )
    assert pr.image_feature_columns(features) == [
        "source::imc::channel::IBA1::mean",
        "source::imc::mask_area",
    ]
