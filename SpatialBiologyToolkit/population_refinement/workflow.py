"""Fit, abstain, audit and integrate population refinements in a notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.cell_classification.classifier import (
    ModelBundle,
    score_cohort,
    train_from_examples,
)
from SpatialBiologyToolkit.napari_sbt.exports import (
    build_assignment_table,
    build_integrated_identity_table,
)
from SpatialBiologyToolkit.napari_sbt.labels import empty_labels
from SpatialBiologyToolkit.napari_sbt.storage import (
    dataframe_sha256,
    feature_recipe_hash,
)

from .models import AssignmentPolicy, SplitSpec
from .selection import SelectionResult

IDENTITY = ["ROI", "ObjectNumber"]


def model_features(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep scientific predictors separate from candidate and identity metadata."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Supply a non-empty, unique feature list.")
    if not set(columns).issubset(features):
        raise ValueError("Some requested model features are absent.")
    forbidden = {
        "Case",
        "ROI",
        "ObjectNumber",
        "CellID",
        "X_loc",
        "Y_loc",
        "class_id",
        "source_score",
        "score_rank",
        "support_score",
        "selected",
        "human_confirmed",
    }
    if any(set(col.split("::")) & forbidden for col in columns) or any(
        "maxfuse" in col.lower() for col in columns
    ):
        raise ValueError(
            "Identifiers and exemplar-label evidence cannot be predictors."
        )
    return features[IDENTITY + columns].copy()


def fit_model(
    features,
    examples,
    *,
    class_ids,
    feature_columns,
    model_type="random_forest",
    seed=0,
    cohort=None,
    policy=AssignmentPolicy(),
) -> ModelBundle:
    selected = model_features(features, feature_columns)
    labels = examples[IDENTITY + ["class_id", "label_origin"]].copy()
    trained = train_from_examples(
        selected,
        labels,
        class_ids=class_ids,
        feature_columns=feature_columns,
        model_type=model_type,
        random_state=seed,
        cohort=cohort,
    )
    if not trained.ok:
        raise ValueError("; ".join(trained.errors))
    bundle = trained.bundle
    if bundle is None:
        raise ValueError("Training did not produce a model.")
    training = (
        trained.training_table[bundle.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    bundle.metadata["feature_support"] = {
        "lower": training.quantile(policy.support_quantile).to_dict(),
        "upper": training.quantile(1 - policy.support_quantile).to_dict(),
        "quantile": policy.support_quantile,
    }
    bundle.metadata["warnings"] = trained.warnings
    return bundle


def assign_predictions(bundle, features, cohort, *, policy=AssignmentPolicy()):
    """Predict without exemplar overrides; flag unsupported features explicitly.

    Marginal training-range checks are a conservative diagnostic, not a general
    detector of unseen biological classes. Image review remains necessary.
    """
    selected = model_features(features, bundle.feature_columns)
    scores = score_cohort(bundle, selected)
    scores = cohort[IDENTITY].merge(
        scores, on=IDENTITY, how="left", validate="one_to_one"
    )
    values = cohort[IDENTITY].merge(
        selected, on=IDENTITY, how="left", validate="one_to_one"
    )
    values = (
        values[bundle.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    support = bundle.metadata["feature_support"]
    if support["quantile"] != policy.support_quantile:
        raise ValueError(
            "Support quantile differs from the fitted model; refit before changing it."
        )
    lower, upper = pd.Series(support["lower"]), pd.Series(support["upper"])
    finite = values.notna()
    outside = (values.lt(lower) | values.gt(upper)) & finite
    scores["finite_feature_fraction"] = finite.mean(axis=1)
    scores["outside_support_fraction"] = outside.sum(axis=1).div(
        finite.sum(axis=1).replace(0, np.nan)
    )
    coverage_bad = scores.finite_feature_fraction.lt(policy.minimum_feature_fraction)
    support_bad = scores.outside_support_fraction.gt(policy.maximum_outside_fraction)
    original_scorable = scores.scorable.fillna(False).astype(bool)
    scores["scorable"] = original_scorable & ~coverage_bad & ~support_bad
    assignments = build_assignment_table(
        cohort,
        empty_labels(),
        scores,
        class_ids=bundle.class_ids,
        minimum_model_confidence=policy.minimum_probability,
        maximum_model_uncertainty=policy.maximum_entropy,
        minimum_probability_margin=policy.minimum_margin,
    )
    assignments.loc[coverage_bad.to_numpy(), "prediction_rejection_reason"] = (
        "insufficient_features"
    )
    assignments.loc[
        (~coverage_bad & support_bad).to_numpy(), "prediction_rejection_reason"
    ] = "outside_training_support"
    assignments["finite_feature_fraction"] = scores.finite_feature_fraction.to_numpy()
    assignments["outside_support_fraction"] = scores.outside_support_fraction.to_numpy()
    assignments["assignment_needs_review"] = ~assignments.model_prediction_accepted
    if policy.assignment_mode == "complete":
        # Complete classification does not turn low-confidence predictions into
        # validated labels. Preserve every diagnostic and fail on absent data.
        if not original_scorable.all() or assignments.predicted_class.isna().any():
            raise ValueError(
                "Complete assignment requires a model prediction for every cell; "
                "repair missing features or explicitly supply a fallback model."
            )
        assignments["class_id"] = assignments.predicted_class
        assignments.loc[assignments.assignment_needs_review, "assignment_source"] = (
            "model_forced_review"
        )
    return scores, assignments


@dataclass
class RefinementResult:
    spec: SplitSpec
    selection: SelectionResult
    bundle: ModelBundle
    scores: pd.DataFrame
    assignments: pd.DataFrame
    policy: AssignmentPolicy

    def integrate(
        self, adata, *, output_key="refined_population", source_key=None, apply=False
    ):
        """Return a full-dataset label table; optionally add a new in-memory column."""
        # Reject stale identity/population alignment instead of positional writes.
        from .sources import resolve_split_cohort

        current = resolve_split_cohort(adata, self.spec)
        saved = self.assignments.set_index("obs_name")
        check = current.set_index("obs_name")
        if set(saved.index) != set(check.index) or not saved[IDENTITY].equals(
            check.reindex(saved.index)[IDENTITY]
        ):
            raise ValueError("AnnData no longer matches the frozen refinement cohort.")
        table = build_integrated_identity_table(
            adata,
            self.assignments,
            source_obs=source_key or self.spec.population_key,
            output_obs=output_key,
            class_labels={item.class_id: item.name for item in self.spec.classes},
            naming_strategy="source_and_class",
            roi_obs=self.spec.roi_key,
            object_id_obs=self.spec.object_key,
        )
        if apply:
            if output_key in adata.obs:
                raise ValueError(
                    "Output observation already exists; choose a new name."
                )
            adata.obs[output_key] = pd.Categorical(
                table.set_index("obs_name").reindex(adata.obs_names)[output_key]
            )
        return table

    def save(self, directory):
        """Persist audit tables with the shared population-QC artifact writer."""
        from pathlib import Path
        from SpatialBiologyToolkit.population_qc import PopulationQCArtifactWriter
        from SpatialBiologyToolkit.cell_classification.classifier import (
            save_model_bundle,
        )

        writer = PopulationQCArtifactWriter(directory, stage="refinement")
        for name, frame in (
            ("candidates", self.selection.candidates),
            ("exemplars", self.selection.examples),
            ("sampling_coverage", self.selection.coverage),
            ("marker_checks", self.selection.checks),
            ("scores", self.scores),
            ("assignments", self.assignments),
        ):
            writer.save_table(frame, name, source="population_refinement")
        writer.save_json(
            {
                "split": self.spec.model_dump(mode="json"),
                **self.selection.settings,
                "assignment": self.policy.model_dump(mode="json"),
                "warnings": self.selection.warnings,
                "source_write_performed": False,
            },
            "settings",
        )
        paths = save_model_bundle(
            self.bundle, Path(directory) / "models" / "refinement.joblib"
        )
        writer.save_json(
            {
                "model": str(paths[0]),
                "metadata": str(paths[1]),
                "model_id": self.bundle.model_id,
            },
            "model_assets",
        )
        return writer


def fit_refinement(
    features,
    selection: SelectionResult,
    spec: SplitSpec,
    *,
    feature_columns,
    model_type="random_forest",
    policy=AssignmentPolicy(),
    seed=0,
) -> RefinementResult:
    """Fit selected exemplars and predict the whole frozen target cohort."""
    settings = selection.settings["sampling"]
    examples = selection.examples
    for cls in spec.class_ids:
        group = examples.loc[examples.class_id.eq(cls)]
        strata = settings["strata"]
        count = group[strata[0]].nunique() if strata else int(bool(len(group)))
        if (
            len(group) < settings["minimum_per_class"]
            or count < settings["minimum_groups_per_class"]
        ):
            raise ValueError(
                f"Insufficient evidence for {cls}: {len(group)} exemplars in {count} groups."
            )
    cohort = selection.candidates[["obs_name", *IDENTITY, "source_population"]].copy()
    bundle = fit_model(
        features,
        examples,
        class_ids=spec.class_ids,
        feature_columns=feature_columns,
        model_type=model_type,
        seed=seed,
        cohort=cohort,
        policy=policy,
    )
    bundle.metadata.update(
        {
            "split_spec": spec.model_dump(mode="json"),
            "selection_settings": selection.settings,
            "assignment_policy": policy.model_dump(mode="json"),
            "cohort_fingerprint": dataframe_sha256(cohort, ["obs_name", *IDENTITY]),
            "selection_fingerprint": feature_recipe_hash(
                examples.to_json(orient="records")
            ),
            "feature_values_fingerprint": dataframe_sha256(
                model_features(features, bundle.feature_columns),
                [*IDENTITY, *bundle.feature_columns],
            ),
        }
    )
    scores, assignments = assign_predictions(bundle, features, cohort, policy=policy)
    return RefinementResult(spec, selection, bundle, scores, assignments, policy)
