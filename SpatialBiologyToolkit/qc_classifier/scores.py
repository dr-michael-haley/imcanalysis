"""Per-object classifier score persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import read_json, timestamp_utc, write_json


def scores_path(scores_folder: str | Path, roi: str) -> Path:
    return Path(scores_folder) / f"{roi}_scores.csv"


def score_metadata_path(metadata_folder: str | Path, roi: str) -> Path:
    return Path(metadata_folder) / f"{roi}_scores_metadata.json"


def load_roi_scores(scores_folder: str | Path, roi: str) -> pd.DataFrame:
    """Load per-object scores for one ROI."""

    path = scores_path(scores_folder, roi)
    if not path.exists():
        return pd.DataFrame(columns=["ROI", "ObjectNumber", "artifact_probability"])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["ROI", "ObjectNumber", "artifact_probability"])


def save_roi_scores(
    scores_folder: str | Path,
    metadata_folder: str | Path,
    roi: str,
    scores: pd.DataFrame,
    *,
    model_metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Save per-object scores and linked metadata for one ROI."""

    path = scores_path(scores_folder, roi)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)

    metadata = {
        "roi": str(roi),
        "score_file": str(path),
        "saved_at": timestamp_utc(),
        "score_convention": "artifact_probability: 0 means likely good cell, 1 means likely artifact",
    }
    if model_metadata:
        metadata.update(
            {
                "model_id": model_metadata.get("model_id"),
                "feature_set_hash": model_metadata.get("feature_set_hash"),
                "feature_columns": model_metadata.get("feature_columns"),
                "model_timestamp": model_metadata.get("timestamp"),
            }
        )
    metadata_path = score_metadata_path(metadata_folder, roi)
    write_json(metadata_path, metadata)
    return path, metadata_path


def load_roi_score_metadata(metadata_folder: str | Path, roi: str) -> dict:
    """Load score metadata for one ROI."""

    return read_json(score_metadata_path(metadata_folder, roi), default={}) or {}


def score_file_warnings(score_metadata: dict, current_model_metadata: dict | None) -> list[str]:
    """Return warnings if score metadata does not match the current model."""

    warnings = []
    if not score_metadata or not current_model_metadata:
        return warnings

    score_model = score_metadata.get("model_id")
    current_model = current_model_metadata.get("model_id")
    if score_model and current_model and score_model != current_model:
        warnings.append(
            f"Score file was produced by model '{score_model}', but current model is '{current_model}'."
        )

    score_hash = score_metadata.get("feature_set_hash")
    current_hash = current_model_metadata.get("feature_set_hash")
    if score_hash and current_hash and score_hash != current_hash:
        warnings.append("Score file was produced with a different feature set hash.")

    return warnings
