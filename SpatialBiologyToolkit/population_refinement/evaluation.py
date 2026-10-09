"""Grouped development validation against exemplar labels, not biological truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from SpatialBiologyToolkit.cell_classification.feature_refinement import (
    refine_trial_features,
)
from SpatialBiologyToolkit.napari_sbt.labels import empty_labels

from .models import AssignmentPolicy
from .workflow import IDENTITY, assign_predictions, fit_model, model_features


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    per_class: pd.DataFrame
    predictions: pd.DataFrame
    confusion: pd.DataFrame
    folds: pd.DataFrame
    warnings: list[str]

    def save(self, writer):
        for name in ("metrics", "per_class", "predictions", "confusion", "folds"):
            writer.save_table(
                getattr(self, name),
                f"validation_{name}",
                source="population_refinement",
            )
        writer.save_json(
            {
                "warnings": self.warnings,
                "interpretation": "agreement with held-out exemplar labels",
            },
            "validation_notes",
        )


def evaluate_refinement(
    features,
    selection,
    spec,
    *,
    feature_sets: dict[str, list[str]],
    group_key="Case",
    model_types=("random_forest", "hist_gradient_boosting"),
    held_out_groups=None,
    refine_features=False,
    recommendation_count=20,
    policy=AssignmentPolicy(),
    seed=0,
) -> EvaluationResult:
    """Leave whole groups out; optional feature refinement is nested inside each fold.

    Pass held_out_groups to bound a development run. Selection rules/marker panels
    must be fixed before evaluating groups. Learned rules require an outer holdout.
    Model comparisons are development results, not an independent final test.
    """
    examples = selection.examples.copy()
    if group_key not in examples or examples[group_key].isna().any():
        raise ValueError(f"Validation requires non-missing {group_key!r} in examples.")
    available = sorted(examples[group_key].astype(str).unique())
    if len(available) < 2:
        raise ValueError("Validation needs at least two independent groups.")
    groups = (
        available
        if held_out_groups is None
        else [str(value) for value in held_out_groups]
    )
    if (
        not groups
        or len(set(groups)) != len(groups)
        or not set(groups).issubset(available)
    ):
        raise ValueError("Held-out groups must be distinct represented groups.")
    metrics, per_class, predictions, confusions, fold_rows, warnings = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for held_out in groups:
        is_test = examples[group_key].astype(str).eq(held_out)
        training, testing = examples.loc[~is_test], examples.loc[is_test]
        counts = training.class_id.value_counts().reindex(spec.class_ids, fill_value=0)
        if counts.lt(2).any():
            warnings.append(
                f"Skipped {held_out}: training groups lack examples for every class."
            )
            fold_rows.append(
                dict(held_out_group=held_out, status="insufficient_training_classes")
            )
            continue
        for feature_set, columns in feature_sets.items():
            fold_columns = list(columns)
            if refine_features:
                if training[group_key].nunique() < 2:
                    raise ValueError(
                        "Nested refinement requires two training groups plus an outer held-out group."
                    )
                inner_examples = training[
                    [
                        *IDENTITY,
                        "class_id",
                        *([] if group_key in IDENTITY else [group_key]),
                    ]
                ]
                inner = refine_trial_features(
                    model_features(features, columns),
                    empty_labels(),
                    examples=inner_examples,
                    class_ids=spec.class_ids,
                    group_column=group_key,
                    feature_columns=columns,
                    recommendation_count=recommendation_count,
                    random_state=seed,
                )
                fold_columns = inner.recommended_features
            test_features = testing[IDENTITY].merge(
                model_features(features, fold_columns),
                on=IDENTITY,
                how="left",
                validate="one_to_one",
            )
            for model in model_types:
                bundle = fit_model(
                    features,
                    training,
                    class_ids=spec.class_ids,
                    feature_columns=fold_columns,
                    model_type=model,
                    seed=seed,
                    policy=policy,
                )
                scores, assignments = assign_predictions(
                    bundle,
                    test_features,
                    testing[["obs_name", *IDENTITY]],
                    policy=policy,
                )
                predicted = scores.predicted_class
                scorable = predicted.notna()
                truth = testing.class_id.reset_index(drop=True)
                if not scorable.any():
                    warnings.append(
                        f"{held_out}/{feature_set}/{model}: no scorable test cells."
                    )
                    continue
                meta = dict(
                    held_out_group=held_out,
                    group_key=group_key,
                    feature_set=feature_set,
                    model=model,
                )
                # Complete assignment fills class_id even for rejected model
                # proposals; acceptance metrics must still honour diagnostics.
                accepted = assignments.model_prediction_accepted
                metrics.append(
                    {
                        **meta,
                        "train_cells": len(training),
                        "test_cells": len(testing),
                        "scorable_fraction": float(scorable.mean()),
                        "feature_count": len(bundle.feature_columns),
                        "balanced_accuracy": balanced_accuracy_score(
                            truth[scorable], predicted[scorable]
                        ),
                        "macro_f1": f1_score(
                            truth[scorable],
                            predicted[scorable],
                            labels=spec.class_ids,
                            average="macro",
                            zero_division=0,
                        ),
                        "accepted_fraction": float(accepted.mean()),
                        "accepted_accuracy": float(
                            assignments.loc[accepted, "class_id"]
                            .eq(truth[accepted])
                            .mean()
                        )
                        if accepted.any()
                        else np.nan,
                    }
                )
                precision, recall, f1, support = precision_recall_fscore_support(
                    truth[scorable],
                    predicted[scorable],
                    labels=spec.class_ids,
                    zero_division=0,
                )
                for i, cls in enumerate(spec.class_ids):
                    per_class.append(
                        {
                            **meta,
                            "class_id": cls,
                            "precision": precision[i],
                            "recall": recall[i],
                            "f1": f1[i],
                            "test_support": int(support[i]),
                        }
                    )
                matrix = confusion_matrix(
                    truth[scorable], predicted[scorable], labels=spec.class_ids
                )
                for i, actual in enumerate(spec.class_ids):
                    for j, prediction in enumerate(spec.class_ids):
                        confusions.append(
                            {
                                **meta,
                                "truth": actual,
                                "predicted": prediction,
                                "cells": int(matrix[i, j]),
                            }
                        )
                prediction_table = assignments.copy()
                prediction_table["exemplar_class"] = truth.to_numpy()
                for key, value in meta.items():
                    prediction_table[key] = value
                predictions.append(prediction_table)
                fold_rows.append(
                    {
                        **meta,
                        "status": "evaluated",
                        "features": ";".join(bundle.feature_columns),
                        "training_groups": ";".join(
                            sorted(training[group_key].astype(str).unique())
                        ),
                    }
                )
    if not metrics:
        raise ValueError("No valid grouped evaluation folds. " + " ".join(warnings))
    return EvaluationResult(
        pd.DataFrame(metrics),
        pd.DataFrame(per_class),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(confusions),
        pd.DataFrame(fold_rows),
        warnings,
    )
