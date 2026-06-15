"""Tabular classifier training, scoring, and persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .feature_table import FeatureTable, coerce_feature_frame
from .io import file_fingerprint, timestamp_slug, timestamp_utc, write_json
from .labels import CONFIRMED_ARTIFACT, CONFIRMED_GOOD


def feature_set_hash(feature_columns: Iterable[str]) -> str:
    """Hash an ordered feature column list."""

    payload = "\n".join(str(col) for col in feature_columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _build_estimator(model_type: str = "auto", random_state: int = 0):
    """Build the preferred available object-level classifier."""

    requested = str(model_type or "auto").lower()
    if requested in {"auto", "xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier

            return "xgboost.XGBClassifier", XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=1,
            )
        except Exception:
            if requested in {"xgboost", "xgb"}:
                raise

    if requested in {"auto", "lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier

            return "lightgbm.LGBMClassifier", LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                random_state=random_state,
                n_jobs=1,
            )
        except Exception:
            if requested in {"lightgbm", "lgbm"}:
                raise

    if requested in {"auto", "histgradientboosting", "hist_gradient_boosting", "hgb"}:
        return "sklearn.ensemble.HistGradientBoostingClassifier", HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            random_state=random_state,
        )

    if requested in {"randomforest", "random_forest", "rf"}:
        return "sklearn.ensemble.RandomForestClassifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


@dataclass
class ModelBundle:
    """Trained classifier and the metadata needed to reuse it safely."""

    estimator: Pipeline
    feature_columns: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return str(self.metadata.get("model_id", self.metadata.get("timestamp", "unsaved_model")))

    @property
    def feature_hash(self) -> str:
        return str(self.metadata.get("feature_set_hash", feature_set_hash(self.feature_columns)))


@dataclass
class TrainingResult:
    """Classifier training result plus warnings."""

    bundle: ModelBundle | None
    training_table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.bundle is not None and not self.errors


def build_training_table(
    feature_table: FeatureTable,
    confirmed_labels: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Join confirmed ROI/object labels to selected feature values."""

    feature_columns = [str(col) for col in feature_columns]
    if confirmed_labels.empty:
        return pd.DataFrame(columns=[feature_table.roi_column, feature_table.object_column, "class_label"] + feature_columns)

    labels = confirmed_labels.copy()
    if "ObjectNumber" in labels.columns and feature_table.object_column != "ObjectNumber":
        labels = labels.rename(columns={"ObjectNumber": feature_table.object_column})
    labels[feature_table.roi_column] = labels["ROI"].astype(str)
    labels[feature_table.object_column] = pd.to_numeric(labels[feature_table.object_column], errors="coerce")

    rows = feature_table.data.copy()
    rows[feature_table.object_column] = pd.to_numeric(rows[feature_table.object_column], errors="coerce")

    joined = labels.merge(
        rows,
        how="inner",
        left_on=[feature_table.roi_column, feature_table.object_column],
        right_on=[feature_table.roi_column, feature_table.object_column],
        suffixes=("_label", ""),
    )
    if "class_label" not in joined.columns:
        joined["class_label"] = joined["label_state"].map(
            {
                CONFIRMED_GOOD: 0,
                CONFIRMED_ARTIFACT: 1,
            }
        )
    joined["class_label"] = pd.to_numeric(joined["class_label"], errors="coerce").astype("Int64")
    return joined


def train_classifier(
    feature_table: FeatureTable,
    confirmed_labels: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    model_type: str = "auto",
    random_state: int = 0,
    feature_table_path: str | Path | None = None,
    dictionary_path: str | Path | None = None,
    selected_categories: Iterable[str] | None = None,
) -> TrainingResult:
    """Train a tabular classifier from confirmed labels only."""

    feature_columns = [str(col) for col in feature_columns]
    warnings = []
    errors = []
    training_table = build_training_table(feature_table, confirmed_labels, feature_columns)

    if training_table.empty:
        errors.append("No confirmed labels matched feature rows; train after confirming good and artifact examples.")
        return TrainingResult(None, training_table, warnings, errors)

    training_table = training_table.dropna(subset=["class_label"]).copy()
    class_counts = training_table["class_label"].astype(int).value_counts().to_dict()
    good_count = int(class_counts.get(0, 0))
    artifact_count = int(class_counts.get(1, 0))
    if good_count == 0 or artifact_count == 0:
        errors.append(
            f"Training requires both classes. Current confirmed examples: good={good_count}, artifact={artifact_count}."
        )
        return TrainingResult(None, training_table, warnings, errors)
    if len(training_table) < 10:
        warnings.append(f"Only {len(training_table)} confirmed examples are available; predictions may be unstable.")

    X = coerce_feature_frame(training_table, feature_columns)
    all_missing = [col for col in feature_columns if X[col].notna().sum() == 0]
    if all_missing:
        errors.append(f"Selected feature(s) are entirely missing in the training data: {', '.join(all_missing[:10])}.")
        return TrainingResult(None, training_table, warnings, errors)

    y = training_table["class_label"].astype(int).to_numpy()
    estimator_name, estimator = _build_estimator(model_type=model_type, random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ]
    )
    pipeline.fit(X, y)

    timestamp = timestamp_utc()
    model_id = f"cellpose_qc_{timestamp_slug()}"
    label_files = sorted(confirmed_labels.get("source_file", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    metadata = {
        "model_id": model_id,
        "timestamp": timestamp,
        "score_convention": "artifact_probability: 0 means likely good cell, 1 means likely artifact",
        "model_type": estimator_name,
        "model_params": estimator.get_params(),
        "feature_columns": feature_columns,
        "feature_set_hash": feature_set_hash(feature_columns),
        "feature_categories_used": None if selected_categories is None else [str(category) for category in selected_categories],
        "training_roi_count": int(training_table[feature_table.roi_column].nunique()),
        "training_rois": sorted(training_table[feature_table.roi_column].astype(str).unique().tolist()),
        "training_example_count": int(len(training_table)),
        "confirmed_good_count": good_count,
        "confirmed_artifact_count": artifact_count,
        "label_files_used": label_files,
        "label_file_fingerprints": {
            label_file: file_fingerprint(label_file) for label_file in label_files
        },
        "feature_table": file_fingerprint(feature_table_path or feature_table.path),
        "feature_dictionary": file_fingerprint(dictionary_path),
        "package_versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "xgboost": _package_version("xgboost"),
            "lightgbm": _package_version("lightgbm"),
        },
    }
    return TrainingResult(ModelBundle(pipeline, feature_columns, metadata), training_table, warnings, errors)


def predict_artifact_probability(bundle: ModelBundle, rows: pd.DataFrame) -> np.ndarray:
    """Predict artifact probabilities for rows from a feature table."""

    missing = [col for col in bundle.feature_columns if col not in rows.columns]
    if missing:
        raise ValueError(f"Rows are missing model feature column(s): {', '.join(missing[:10])}")

    X = coerce_feature_frame(rows, bundle.feature_columns)
    classifier = bundle.estimator.named_steps.get("classifier", bundle.estimator)
    if hasattr(bundle.estimator, "predict_proba"):
        probabilities = bundle.estimator.predict_proba(X)
        classes = getattr(classifier, "classes_", np.array([0, 1]))
        artifact_indices = np.where(classes == 1)[0]
        if artifact_indices.size == 0:
            return np.zeros(len(X), dtype=float)
        return probabilities[:, int(artifact_indices[0])]

    if hasattr(bundle.estimator, "decision_function"):
        scores = bundle.estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    predictions = bundle.estimator.predict(X)
    return np.asarray(predictions, dtype=float)


def score_feature_rows(
    bundle: ModelBundle,
    feature_table: FeatureTable,
    roi: str | None = None,
) -> pd.DataFrame:
    """Apply a trained model to one ROI or the whole feature table."""

    rows = feature_table.rows_for_roi(roi) if roi is not None else feature_table.data.copy()
    scores = predict_artifact_probability(bundle, rows)
    result = rows.loc[:, [feature_table.roi_column, feature_table.object_column]].copy()
    result = result.rename(columns={feature_table.roi_column: "ROI", feature_table.object_column: "ObjectNumber"})
    result["ObjectNumber"] = pd.to_numeric(result["ObjectNumber"], errors="coerce").astype("Int64")
    result["artifact_probability"] = scores.astype(float)
    result["model_id"] = bundle.model_id
    result["feature_set_hash"] = bundle.feature_hash
    result["scored_at"] = timestamp_utc()
    return result


def save_model_bundle(bundle: ModelBundle, models_folder: str | Path, *, make_latest: bool = True) -> tuple[Path, Path]:
    """Save a model bundle and metadata JSON."""

    models_folder = Path(models_folder)
    models_folder.mkdir(parents=True, exist_ok=True)
    model_id = bundle.model_id
    model_path = models_folder / f"{model_id}.joblib"
    metadata_path = models_folder / f"{model_id}_metadata.json"
    joblib.dump({"estimator": bundle.estimator, "feature_columns": bundle.feature_columns, "metadata": bundle.metadata}, model_path)
    write_json(metadata_path, bundle.metadata)

    if make_latest:
        latest_path = models_folder / "classifier_latest.joblib"
        latest_metadata_path = models_folder / "classifier_latest_metadata.json"
        joblib.dump({"estimator": bundle.estimator, "feature_columns": bundle.feature_columns, "metadata": bundle.metadata}, latest_path)
        write_json(latest_metadata_path, bundle.metadata)

    return model_path, metadata_path


def load_model_bundle(model_path: str | Path) -> ModelBundle:
    """Load a saved model bundle."""

    payload = joblib.load(model_path)
    if isinstance(payload, ModelBundle):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported model file format: {model_path}")
    return ModelBundle(
        estimator=payload["estimator"],
        feature_columns=[str(col) for col in payload["feature_columns"]],
        metadata=dict(payload.get("metadata", {})),
    )
