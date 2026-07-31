"""ROI-grouped feature discovery for Napari SBT trial experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classifier import IDENTITY, default_feature_columns
from .feature_catalog import (
    CONTEXT_FEATURE_DESCRIPTIONS,
    DISTRIBUTION_FEATURE_DESCRIPTIONS,
    REGION_IMAGE_FEATURE_DESCRIPTIONS,
    SHAPE_FEATURE_DESCRIPTIONS,
)
from .models import SyntheticFeatureRecipe


@dataclass
class FeatureRefinementResult:
    """Tables and provenance produced by one representative-ROI trial."""

    ranking: pd.DataFrame
    fold_metrics: pd.DataFrame
    recommended_features: list[str]
    summary: dict
    warnings: list[str] = field(default_factory=list)


def compact_synthetic_recipe(
    recipe: SyntheticFeatureRecipe,
    model_features: Iterable[str],
) -> SyntheticFeatureRecipe:
    """Reduce a broad recipe to prerequisites of selected synthetic features."""

    synthetic = [
        str(feature).removeprefix("source::imc::")
        for feature in model_features
        if str(feature).startswith("source::imc::")
    ]
    if not synthetic:
        return recipe.model_copy(deep=True)
    channels: list[str] = []
    distribution: list[str] = []
    region: list[str] = []
    gradient: list[str] = []
    shape: list[str] = []
    context: list[str] = []
    rank_statistics: list[str] = []
    for feature in synthetic:
        base = feature
        for statistic in ("zscore", "percentile"):
            rank_suffix = f"::cohort_roi_{statistic}"
            if base.endswith(rank_suffix):
                rank_statistics.append(statistic)
                base = base[: -len(rank_suffix)]
                break
        if base in SHAPE_FEATURE_DESCRIPTIONS:
            shape.append(base)
            continue
        if base in CONTEXT_FEATURE_DESCRIPTIONS:
            context.append(base)
            continue
        if not base.startswith("channel::"):
            continue
        gradient_match = next(
            (
                suffix
                for suffix in DISTRIBUTION_FEATURE_DESCRIPTIONS
                if base.endswith(f"::gradient::{suffix}")
            ),
            None,
        )
        if gradient_match is not None:
            channel = base[
                len("channel::") : -len(f"::gradient::{gradient_match}")
            ]
            channels.append(channel)
            gradient.append(gradient_match)
            continue
        region_match = next(
            (
                suffix
                for suffix in REGION_IMAGE_FEATURE_DESCRIPTIONS
                if base.endswith(f"::{suffix}")
            ),
            None,
        )
        if region_match is not None:
            channel = base[len("channel::") : -len(f"::{region_match}")]
            channels.append(channel)
            region.append(region_match)
            continue
        distribution_match = next(
            (
                suffix
                for suffix in DISTRIBUTION_FEATURE_DESCRIPTIONS
                if base.endswith(f"::{suffix}")
            ),
            None,
        )
        if distribution_match is not None:
            channel = base[len("channel::") : -len(f"::{distribution_match}")]
            channels.append(channel)
            distribution.append(distribution_match)

    def unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(values))

    if not any((distribution, region, gradient, shape, context)):
        return recipe.model_copy(deep=True)

    # ROI ranks require their base features to exist in the same ROI table.
    updated = recipe.model_copy(deep=True)
    updated.channels = unique(channels)
    updated.distribution_features = bool(distribution)
    updated.distribution_feature_names = unique(distribution)
    updated.region_features = bool(region)
    updated.region_feature_names = unique(region)
    updated.gradient_features = bool(gradient)
    updated.gradient_feature_names = unique(gradient)
    updated.shape_features = bool(shape)
    updated.shape_feature_names = unique(shape)
    updated.context_features = bool(context)
    updated.context_feature_names = unique(context)
    updated.roi_rank_features = bool(rank_statistics)
    updated.roi_rank_statistics = unique(rank_statistics)
    return SyntheticFeatureRecipe.model_validate(updated.model_dump())


def _feature_source_and_family(feature: str) -> tuple[str, str]:
    source = "combined"
    base = str(feature)
    if base.startswith("source::"):
        parts = base.split("::", 2)
        if len(parts) == 3:
            source, base = parts[1], parts[2]
    if "::gradient::" in base:
        family = "channel_gradient"
    elif "cohort_roi_" in base:
        family = "cohort_roi_rank"
    elif base.startswith("channel::"):
        region_tokens = (
            "core_",
            "border_",
            "local_bg_",
            "foreground_bg_",
            "foreground_to_bg_",
            "weighted_",
        )
        family = (
            "channel_region"
            if any(token in base for token in region_tokens)
            else "channel_distribution"
        )
    elif base.startswith("mask_"):
        family = "mask_morphology"
    elif base.startswith(("centroid_", "roi_")):
        family = "spatial_context"
    else:
        family = "imported_or_embedding"
    return source, family


def _candidate_diagnostics(
    training: pd.DataFrame,
    candidates: Iterable[str],
    *,
    maximum_missing_fraction: float,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    retained = []
    for feature in candidates:
        values = pd.to_numeric(training[feature], errors="coerce")
        missing_fraction = float(values.isna().mean())
        unique_values = int(values.nunique(dropna=True))
        status = "eligible"
        if missing_fraction > maximum_missing_fraction:
            status = "high_missingness"
        elif unique_values <= 1:
            status = "constant"
        else:
            retained.append(str(feature))
        source, family = _feature_source_and_family(str(feature))
        rows.append(
            {
                "feature": str(feature),
                "source": source,
                "family": family,
                "missing_fraction": missing_fraction,
                "unique_values": unique_values,
                "screening_status": status,
            }
        )
    return retained, pd.DataFrame(rows)


def _correlation_prune(
    frame: pd.DataFrame,
    ordered_features: list[str],
    threshold: float,
) -> tuple[list[str], dict[str, str]]:
    if len(ordered_features) < 2:
        return ordered_features, {}
    numeric = frame.loc[:, ordered_features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    correlations = numeric.corr().abs()
    retained: list[str] = []
    redundant: dict[str, str] = {}
    for feature in ordered_features:
        match = next(
            (
                existing
                for existing in retained
                if pd.notna(correlations.at[feature, existing])
                and float(correlations.at[feature, existing]) >= threshold
            ),
            None,
        )
        if match is None:
            retained.append(feature)
        else:
            redundant[feature] = match
    return retained, redundant


def _model_specs(random_state: int) -> tuple[tuple[str, Pipeline], ...]:
    return (
        (
            "elastic_net",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            l1_ratio=0.5,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=300,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    )


def refine_trial_features(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    class_ids: Iterable[str],
    maximum_candidate_features: int = 150,
    recommendation_count: int = 30,
    permutation_repeats: int = 5,
    maximum_missing_fraction: float = 0.30,
    correlation_threshold: float = 0.95,
    random_state: int = 0,
    progress: Callable[[dict], None] | None = None,
) -> FeatureRefinementResult:
    """Rank features using held-out-ROI permutation importance.

    Feature screening is fitted independently inside each training fold. The
    final recommendation aggregates held-out importance and removes strongly
    correlated alternatives, favouring the more stable feature.
    """

    if maximum_candidate_features < 2:
        raise ValueError("Maximum candidate features must be at least two.")
    if recommendation_count < 1:
        raise ValueError("Recommendation count must be positive.")
    if permutation_repeats < 1:
        raise ValueError("Permutation repeats must be positive.")
    if not 0 <= maximum_missing_fraction < 1:
        raise ValueError("Maximum missing fraction must be in [0, 1).")
    if not 0.5 <= correlation_threshold < 1:
        raise ValueError("Correlation threshold must be in [0.5, 1).")

    required = set(IDENTITY)
    for name, table in (("feature table", features), ("labels", labels)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"The {name} is missing identity columns: {sorted(missing)}"
            )
    confirmed = labels.loc[labels["state"].astype(str).eq("confirmed")].copy()
    training = confirmed.merge(
        features,
        on=IDENTITY,
        how="inner",
        validate="one_to_one",
    )
    if training.empty:
        raise ValueError("No confirmed labels match the trial feature table.")
    ordered_classes = [str(value) for value in class_ids]
    counts = training["class_id"].astype(str).value_counts()
    absent = [
        class_id
        for class_id in ordered_classes
        if int(counts.get(class_id, 0)) < 2
    ]
    if absent:
        raise ValueError(
            "Feature refinement requires at least two confirmed feature-bearing "
            f"cells per class. Underfilled classes: {absent}"
        )
    rois = sorted(training["ROI"].astype(str).unique())
    if len(rois) < 2:
        raise ValueError(
            "Feature refinement requires confirmed labels in at least two ROIs; "
            "three or more are strongly recommended."
        )
    warnings: list[str] = []
    if len(rois) < 3:
        warnings.append(
            "Only two ROIs contain confirmed labels; held-out estimates are highly "
            "preliminary."
        )
    candidates = default_feature_columns(features)
    retained, diagnostics = _candidate_diagnostics(
        training,
        candidates,
        maximum_missing_fraction=maximum_missing_fraction,
    )
    if not retained:
        raise ValueError(
            "No usable features remain after missingness and variance checks."
        )

    class_to_index = {class_id: index for index, class_id in enumerate(ordered_classes)}
    y_all = training["class_id"].astype(str).map(class_to_index)
    if y_all.isna().any():
        unknown = sorted(training.loc[y_all.isna(), "class_id"].astype(str).unique())
        raise ValueError(f"Confirmed labels contain unknown classes: {unknown}")
    y_all = y_all.astype(int).to_numpy()
    importance_rows: list[dict] = []
    metric_rows: list[dict] = []
    valid_evaluations = 0
    for held_out_roi in rois:
        if progress is not None:
            progress(
                {
                    "event": "refinement_fold_started",
                    "held_out_roi": held_out_roi,
                    "completed_fold_models": valid_evaluations,
                    "total_fold_models": len(rois) * 2,
                }
            )
        train_mask = ~training["ROI"].astype(str).eq(held_out_roi)
        test_mask = ~train_mask
        y_train = y_all[train_mask.to_numpy()]
        y_test = y_all[test_mask.to_numpy()]
        if len(np.unique(y_train)) != len(ordered_classes):
            warnings.append(
                f"Skipped held-out ROI {held_out_roi!r}: its training ROIs do not "
                "contain every class."
            )
            continue
        if len(np.unique(y_test)) != len(ordered_classes):
            warnings.append(
                f"Held-out ROI {held_out_roi!r} does not contain every class; "
                "interpret its per-class performance cautiously."
            )
        X_train_all = training.loc[train_mask, retained].apply(
            pd.to_numeric, errors="coerce"
        )
        X_test_all = training.loc[test_mask, retained].apply(
            pd.to_numeric, errors="coerce"
        )
        fold_candidates = [
            feature
            for feature in retained
            if X_train_all[feature].notna().any()
            and X_train_all[feature].nunique(dropna=True) > 1
        ]
        if not fold_candidates:
            warnings.append(
                f"Skipped held-out ROI {held_out_roi!r}: no variable training "
                "features remained."
            )
            continue
        screening_imputer = SimpleImputer(strategy="median")
        X_train_screened = screening_imputer.fit_transform(
            X_train_all.loc[:, fold_candidates]
        )
        mutual_information = mutual_info_classif(
            X_train_screened,
            y_train,
            random_state=random_state,
        )
        feature_order = [
            fold_candidates[index]
            for index in np.argsort(mutual_information)[::-1][
                : min(maximum_candidate_features, len(fold_candidates))
            ]
        ]
        fold_features, _redundant = _correlation_prune(
            X_train_all,
            feature_order,
            correlation_threshold,
        )
        if not fold_features:
            continue
        for model_name, estimator in _model_specs(random_state):
            estimator.fit(X_train_all.loc[:, fold_features], y_train)
            predicted = estimator.predict(X_test_all.loc[:, fold_features])
            probabilities = estimator.predict_proba(
                X_test_all.loc[:, fold_features]
            )
            metric_rows.append(
                {
                    "held_out_roi": held_out_roi,
                    "model": model_name,
                    "train_cells": int(train_mask.sum()),
                    "test_cells": int(test_mask.sum()),
                    "feature_count": len(fold_features),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test, predicted)
                    ),
                    "macro_f1": float(
                        f1_score(
                            y_test,
                            predicted,
                            labels=list(range(len(ordered_classes))),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "log_loss": float(
                        log_loss(
                            y_test,
                            probabilities,
                            labels=list(range(len(ordered_classes))),
                        )
                    ),
                }
            )
            permutation = permutation_importance(
                estimator,
                X_test_all.loc[:, fold_features],
                y_test,
                scoring="balanced_accuracy",
                n_repeats=permutation_repeats,
                random_state=random_state,
                n_jobs=1,
            )
            valid_evaluations += 1
            if progress is not None:
                progress(
                    {
                        "event": "refinement_model_completed",
                        "held_out_roi": held_out_roi,
                        "model": model_name,
                        "feature_count": len(fold_features),
                        "balanced_accuracy": metric_rows[-1][
                            "balanced_accuracy"
                        ],
                        "completed_fold_models": valid_evaluations,
                        "total_fold_models": len(rois) * 2,
                    }
                )
            for feature, mean, std in zip(
                fold_features,
                permutation.importances_mean,
                permutation.importances_std,
                strict=True,
            ):
                importance_rows.append(
                    {
                        "held_out_roi": held_out_roi,
                        "model": model_name,
                        "feature": feature,
                        "permutation_importance": float(mean),
                        "permutation_importance_std": float(std),
                    }
                )
    if not metric_rows:
        raise ValueError(
            "No valid leave-one-ROI-out folds could be evaluated. Confirm every "
            "class in at least two trial ROIs."
        )
    importance = pd.DataFrame(importance_rows)
    aggregate = (
        importance.groupby("feature", as_index=False)
        .agg(
            mean_permutation_importance=("permutation_importance", "mean"),
            importance_std=("permutation_importance", "std"),
            evaluated_fold_models=("permutation_importance", "size"),
            useful_fold_models=(
                "permutation_importance",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    aggregate["selection_frequency"] = (
        aggregate["evaluated_fold_models"] / max(valid_evaluations, 1)
    )
    aggregate["positive_importance_frequency"] = (
        aggregate["useful_fold_models"] / aggregate["evaluated_fold_models"]
    )
    ranking = diagnostics.merge(aggregate, on="feature", how="left")
    for column in (
        "mean_permutation_importance",
        "importance_std",
        "evaluated_fold_models",
        "useful_fold_models",
        "selection_frequency",
        "positive_importance_frequency",
    ):
        ranking[column] = ranking[column].fillna(0)
    ranking = ranking.sort_values(
        [
            "mean_permutation_importance",
            "positive_importance_frequency",
            "selection_frequency",
        ],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    ranked_eligible = ranking.loc[
        ranking["screening_status"].eq("eligible"), "feature"
    ].tolist()
    recommendation_pool = ranked_eligible[
        : max(maximum_candidate_features, recommendation_count * 5)
    ]
    nonredundant, redundant = _correlation_prune(
        training,
        recommendation_pool,
        correlation_threshold,
    )
    recommended = nonredundant[: min(recommendation_count, len(nonredundant))]
    ranking["redundant_with"] = ranking["feature"].map(redundant).fillna("")
    ranking["recommended"] = ranking["feature"].isin(recommended)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    metrics = pd.DataFrame(metric_rows)
    family_summary = (
        ranking.loc[ranking["screening_status"].eq("eligible")]
        .groupby(["source", "family"], as_index=False)
        .agg(
            feature_count=("feature", "size"),
            total_permutation_importance=(
                "mean_permutation_importance",
                lambda values: float(values.clip(lower=0).sum()),
            ),
            mean_positive_importance_frequency=(
                "positive_importance_frequency",
                "mean",
            ),
            recommended_features=("recommended", "sum"),
        )
        .sort_values("total_permutation_importance", ascending=False)
        .reset_index(drop=True)
    )
    summary = {
        "schema_version": 1,
        "trial_rois": rois,
        "confirmed_cells": len(training),
        "class_counts": {key: int(value) for key, value in counts.items()},
        "candidate_features": len(candidates),
        "eligible_features": len(retained),
        "valid_fold_models": valid_evaluations,
        "mean_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
        "mean_macro_f1": float(metrics["macro_f1"].mean()),
        "mean_log_loss": float(metrics["log_loss"].mean()),
        "recommended_feature_count": len(recommended),
        "recommended_features": recommended,
        "family_importance": family_summary.to_dict("records"),
        "settings": {
            "maximum_candidate_features": maximum_candidate_features,
            "recommendation_count": recommendation_count,
            "permutation_repeats": permutation_repeats,
            "maximum_missing_fraction": maximum_missing_fraction,
            "correlation_threshold": correlation_threshold,
            "random_state": random_state,
        },
        "warnings": warnings,
    }
    return FeatureRefinementResult(
        ranking=ranking,
        fold_metrics=metrics,
        recommended_features=recommended,
        summary=summary,
        warnings=warnings,
    )


__all__ = [
    "FeatureRefinementResult",
    "compact_synthetic_recipe",
    "refine_trial_features",
]
