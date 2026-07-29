"""Cohort-restricted multiclass active-learning services."""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from SpatialBiologyToolkit.pipeline.manifests import write_json

from .labels import validate_labels
from .storage import feature_recipe_hash

IDENTITY = ["ROI", "ObjectNumber"]


def confirmed_labels_fingerprint(labels: pd.DataFrame) -> str:
    confirmed = labels.loc[
        labels["state"].eq("confirmed"),
        ["ROI", "ObjectNumber", "class_id", "timestamp"],
    ]
    return feature_recipe_hash(confirmed.to_dict("records"))


def feature_set_hash(feature_columns: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(column) for column in feature_columns).encode("utf-8")
    ).hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _estimator(model_type: str, random_state: int):
    requested = str(model_type or "hist_gradient_boosting").lower()
    if requested in {"hist_gradient_boosting", "histgradientboosting", "hgb"}:
        return (
            "sklearn.ensemble.HistGradientBoostingClassifier",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                random_state=random_state,
            ),
        )
    if requested in {"random_forest", "randomforest", "rf"}:
        return (
            "sklearn.ensemble.RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    if requested in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost was selected but is not installed in this environment."
            ) from exc
        return (
            "xgboost.XGBClassifier",
            XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if requested in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM was selected but is not installed in this environment."
            ) from exc
        return (
            "lightgbm.LGBMClassifier",
            LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    raise ValueError(f"Unsupported multiclass model type: {model_type!r}")


@dataclass
class ModelBundle:
    estimator: Pipeline
    feature_columns: list[str]
    class_ids: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return str(self.metadata.get("model_id", "unsaved_model"))


@dataclass
class TrainingResult:
    bundle: ModelBundle | None
    training_table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.bundle is not None and not self.errors


def _normalise_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in IDENTITY if column not in features.columns]
    if missing:
        raise ValueError(f"Feature table is missing identity column(s): {missing}")
    rows = features.copy()
    rows["ROI"] = rows["ROI"].astype(str)
    rows["ObjectNumber"] = pd.to_numeric(
        rows["ObjectNumber"], errors="raise"
    ).astype("int64")
    if rows.duplicated(IDENTITY).any():
        raise ValueError("Feature table contains duplicate cohort cell identities.")
    return rows


def default_feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {
        "ObjectNumber",
        "measurement_region_vanished",
        "measurement_mask_offset_px",
    }
    return [
        str(column)
        for column in features.columns
        if column not in IDENTITY
        and column not in excluded
        and pd.api.types.is_numeric_dtype(features[column])
    ]


def train_multiclass_classifier(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    class_ids: Iterable[str],
    feature_columns: Iterable[str] | None = None,
    cohort: pd.DataFrame | None = None,
    model_type: str = "hist_gradient_boosting",
    random_state: int = 0,
    cohort_fingerprint: str | None = None,
    feature_set_id: str | None = None,
) -> TrainingResult:
    """Train only on confirmed cohort labels, requiring two examples per class."""

    rows = _normalise_feature_rows(features)
    ordered_classes = [str(class_id) for class_id in class_ids]
    labels = validate_labels(labels, class_ids=ordered_classes, cohort=cohort)
    confirmed = labels.loc[labels["state"] == "confirmed"].copy()
    training = confirmed.merge(rows, on=IDENTITY, how="inner", validate="one_to_one")
    warnings: list[str] = []
    errors: list[str] = []
    if training.empty:
        errors.append("No confirmed labels match usable cohort feature rows.")
        return TrainingResult(None, training, warnings, errors)

    counts = training["class_id"].value_counts()
    underfilled = {
        class_id: int(counts.get(class_id, 0))
        for class_id in ordered_classes
        if int(counts.get(class_id, 0)) < 2
    }
    if underfilled:
        errors.append(
            "Training requires at least two confirmed examples per class; current "
            f"counts: {underfilled}"
        )
        return TrainingResult(None, training, warnings, errors)

    selected = (
        default_feature_columns(rows)
        if feature_columns is None
        else [str(column) for column in feature_columns]
    )
    missing_features = sorted(set(selected) - set(rows.columns))
    if missing_features:
        errors.append(f"Selected model features are missing: {missing_features[:10]}")
        return TrainingResult(None, training, warnings, errors)
    if not selected:
        errors.append("No numeric model features were selected.")
        return TrainingResult(None, training, warnings, errors)
    X = training.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    all_missing = [column for column in selected if X[column].notna().sum() == 0]
    if all_missing:
        if len(all_missing) == len(selected):
            errors.append(
                "Every selected feature is entirely missing in training data."
            )
            return TrainingResult(None, training, warnings, errors)
        warnings.append(
            "Dropped features that are entirely missing in confirmed training "
            "examples: " + ", ".join(all_missing[:10])
        )
    selected = [column for column in selected if column not in all_missing]
    X = X.loc[:, selected]

    class_to_index = {
        class_id: index for index, class_id in enumerate(ordered_classes)
    }
    y = training["class_id"].map(class_to_index).astype(int).to_numpy()
    if len(training) < max(20, 5 * len(ordered_classes)):
        warnings.append(
            f"Only {len(training)} confirmed examples are available; predictions "
            "may be unstable."
        )
    roi_class = pd.crosstab(training["ROI"], training["class_id"])
    missing_roi_pairs = int((roi_class == 0).sum().sum())
    if training["ROI"].nunique() > 1 and missing_roi_pairs:
        warnings.append(
            "Some represented ROIs have no confirmed examples for one or more "
            "classes; evaluate ROI generalisation carefully."
        )

    estimator_name, classifier = _estimator(model_type, random_state)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    pipeline.fit(X, y, classifier__sample_weight=sample_weight)
    now = pd.Timestamp.now(tz="UTC").isoformat()
    metadata = {
        "model_id": f"napari_sbt_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": now,
        "model_type": estimator_name,
        "class_ids": ordered_classes,
        "class_counts": {key: int(value) for key, value in counts.items()},
        "training_example_count": len(training),
        "training_roi_count": int(training["ROI"].nunique()),
        "training_rois": sorted(training["ROI"].unique().tolist()),
        "feature_columns": selected,
        "feature_set_hash": feature_set_hash(selected),
        "feature_set_id": feature_set_id,
        "cohort_fingerprint": cohort_fingerprint,
        "labels_fingerprint": confirmed_labels_fingerprint(confirmed),
        "package_versions": {
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit-learn": _version("scikit-learn"),
            "xgboost": _version("xgboost"),
            "lightgbm": _version("lightgbm"),
        },
    }
    return TrainingResult(
        ModelBundle(pipeline, selected, ordered_classes, metadata),
        training,
        warnings,
        errors,
    )


def score_cohort(
    bundle: ModelBundle,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Score only feature rows with at least one usable value."""

    rows = _normalise_feature_rows(features)
    missing = sorted(set(bundle.feature_columns) - set(rows.columns))
    if missing:
        raise ValueError(f"Scoring table is missing model features: {missing[:10]}")
    X = rows.loc[:, bundle.feature_columns].apply(pd.to_numeric, errors="coerce")
    usable = X.notna().any(axis=1)
    probability_columns = [
        f"probability::{class_id}" for class_id in bundle.class_ids
    ]
    result = rows.loc[:, IDENTITY].copy()
    for column in probability_columns:
        result[column] = np.nan
    result["predicted_class"] = pd.NA
    result["maximum_probability"] = np.nan
    result["probability_margin"] = np.nan
    result["normalized_entropy"] = np.nan
    result["scorable"] = usable.to_numpy()
    if usable.any():
        probabilities = bundle.estimator.predict_proba(X.loc[usable])
        classifier = bundle.estimator.named_steps["classifier"]
        estimator_classes = np.asarray(classifier.classes_, dtype=int)
        ordered = np.full((probabilities.shape[0], len(bundle.class_ids)), np.nan)
        ordered[:, estimator_classes] = probabilities
        result.loc[usable, probability_columns] = ordered
        winner = np.nanargmax(ordered, axis=1)
        result.loc[usable, "predicted_class"] = [
            bundle.class_ids[index] for index in winner
        ]
        sorted_probability = np.sort(ordered, axis=1)
        maximum = sorted_probability[:, -1]
        margin = maximum - sorted_probability[:, -2]
        entropy = -np.sum(
            np.where(ordered > 0, ordered * np.log(ordered), 0.0), axis=1
        ) / np.log(len(bundle.class_ids))
        result.loc[usable, "maximum_probability"] = maximum
        result.loc[usable, "probability_margin"] = margin
        result.loc[usable, "normalized_entropy"] = entropy
    result["model_id"] = bundle.model_id
    result["feature_set_hash"] = bundle.metadata.get("feature_set_hash")
    result["feature_set_id"] = bundle.metadata.get("feature_set_id")
    result["scored_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return result


def uncertainty_queue(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    limit: int = 100,
    roi: str | None = None,
    predicted_class: str | None = None,
) -> pd.DataFrame:
    """Rank unlabelled scorable cells by normalized entropy then margin."""

    labelled = labels.loc[:, IDENTITY].drop_duplicates()
    candidates = scores.merge(
        labelled.assign(_labelled=True), on=IDENTITY, how="left"
    )
    candidates = candidates.loc[
        candidates["_labelled"].isna() & candidates["scorable"].fillna(False)
    ]
    if roi is not None:
        candidates = candidates.loc[candidates["ROI"].astype(str) == str(roi)]
    if predicted_class is not None:
        candidates = candidates.loc[
            candidates["predicted_class"].astype(str) == str(predicted_class)
        ]
    return (
        candidates.sort_values(
            ["normalized_entropy", "probability_margin"],
            ascending=[False, True],
            na_position="last",
        )
        .head(int(limit))
        .drop(columns="_labelled")
        .reset_index(drop=True)
    )


def high_confidence_queue(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    class_id: str,
    threshold: float = 0.9,
    limit: int = 500,
) -> pd.DataFrame:
    probability = f"probability::{class_id}"
    if probability not in scores.columns:
        raise ValueError(f"Scores do not contain class {class_id!r}.")
    labelled = labels.loc[:, IDENTITY].drop_duplicates()
    candidates = scores.merge(
        labelled.assign(_labelled=True), on=IDENTITY, how="left"
    )
    return (
        candidates.loc[
            candidates["_labelled"].isna()
            & candidates["predicted_class"].eq(class_id)
            & candidates[probability].ge(float(threshold))
        ]
        .sort_values(probability, ascending=False)
        .head(int(limit))
        .drop(columns="_labelled")
        .reset_index(drop=True)
    )


def save_model_bundle(
    bundle: ModelBundle, path: str | Path
) -> tuple[Path, Path]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": bundle.estimator,
            "feature_columns": bundle.feature_columns,
            "class_ids": bundle.class_ids,
            "metadata": bundle.metadata,
        },
        destination,
    )
    metadata_path = destination.with_suffix(".json")
    write_json(metadata_path, bundle.metadata)
    return destination, metadata_path


def load_model_bundle(path: str | Path) -> ModelBundle:
    payload = joblib.load(path)
    return ModelBundle(
        estimator=payload["estimator"],
        feature_columns=[str(value) for value in payload["feature_columns"]],
        class_ids=[str(value) for value in payload["class_ids"]],
        metadata=dict(payload.get("metadata", {})),
    )


__all__ = [
    "ModelBundle",
    "TrainingResult",
    "confirmed_labels_fingerprint",
    "default_feature_columns",
    "high_confidence_queue",
    "load_model_bundle",
    "save_model_bundle",
    "score_cohort",
    "train_multiclass_classifier",
    "uncertainty_queue",
]
