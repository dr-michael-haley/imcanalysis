"""Versioned, backwards-compatible AnnData storage for population QC results."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig

from .models import MetricDefinition, PopulationEmbeddingQCResult


UNS_KEY = "population_embedding_qc"
STORAGE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1

OBS_COLUMN_MAP: dict[str, str] = {
    "graph_neighbour_purity": "embedding_qc_graph_purity",
    "umap_neighbour_purity": "embedding_qc_umap_purity",
    "umap_graph_neighbourhood_preservation": "embedding_qc_umap_graph_preservation",
    "boundary_class": "embedding_qc_boundary_class",
}

_METRIC_CONFIG_FIELDS = (
    "mode",
    "population_obs",
    "sweep_regex",
    "sweep_columns",
    "reference_resolution",
    "umap_key",
    "pca_key",
    "connectivities_key",
    "sample_obs",
    "roi_obs",
    "pca_dimensions",
    "umap_k",
    "graph_boundary_threshold",
    "core_purity_threshold",
    "high_entropy_threshold",
    "min_cluster_size",
    "min_component_size",
    "persistence_jaccard_threshold",
    "silhouette_max_cells",
    "density_max_cells_per_cluster",
    "density_grid_size",
    "metric_config_path",
    "include_optional_metrics",
    "random_seed",
)


class StoredPopulationQCError(ValueError):
    """Raised when stored population QC is missing, invalid, or incompatible."""


def _json_value(value: Any) -> Any:
    """Convert values to strict JSON-compatible primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index, pd.Series)):
        values = value.tolist() if hasattr(value, "tolist") else list(value)
        return [_json_value(item) for item in values]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="python"))
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _uns_value(value: Any) -> Any:
    """Sanitize a value for old AnnData writers, dropping null entries."""
    cleaned = _json_value(value)
    if isinstance(cleaned, dict):
        return {
            str(key): safe
            for key, item in cleaned.items()
            if (safe := _uns_value(item)) is not None
        }
    if isinstance(cleaned, list):
        return [safe for item in cleaned if (safe := _uns_value(item)) is not None]
    return cleaned


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "index": _json_value(frame.index.tolist()),
        "index_name": _json_value(frame.index.name),
        "columns": _json_value(frame.columns.tolist()),
        "columns_name": _json_value(frame.columns.name),
        "data": _json_value(frame.to_numpy(dtype=object).tolist()),
    }


def _frame_from_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(
        payload.get("data", []),
        index=payload.get("index", []),
        columns=payload.get("columns", []),
    )
    frame.index.name = payload.get("index_name")
    frame.columns.name = payload.get("columns_name")
    return frame


def _hash_text(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _sampled_array_fingerprint(value: Any, *, maximum: int = 4096) -> dict[str, Any]:
    array = np.asarray(value)
    flat = array.reshape(-1)
    if flat.size > maximum:
        positions = np.linspace(0, flat.size - 1, maximum, dtype=np.int64)
        sampled = flat[positions]
    else:
        sampled = flat
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sampled_values_sha256": _hash_text(sampled),
    }


def _sampled_sparse_fingerprint(value: Any, *, maximum: int = 4096) -> dict[str, Any]:
    matrix = sparse.csr_matrix(value)

    def sampled(values: np.ndarray) -> np.ndarray:
        if len(values) <= maximum:
            return values
        positions = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
        return values[positions]

    digest = hashlib.sha256()
    for values in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(_hash_text(sampled(np.asarray(values))).encode("ascii"))
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "nnz": int(matrix.nnz),
        "sampled_structure_sha256": digest.hexdigest(),
    }


def metric_config_snapshot(config: PopulationEmbeddingQCConfig) -> dict[str, Any]:
    """Return only settings that can change calculated QC values."""
    raw = config.model_dump(mode="python")
    snapshot = {
        field: _json_value(raw.get(field))
        for field in _METRIC_CONFIG_FIELDS
    }
    metric_path = config.metric_config_path
    if metric_path and os.path.isfile(os.fspath(metric_path)):
        with open(os.fspath(metric_path), "rb") as handle:
            snapshot["metric_config_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    return snapshot


def build_compatibility_signature(
    adata: Any,
    *,
    reference_column: str,
    run_summary: Mapping[str, Any],
    config: PopulationEmbeddingQCConfig,
) -> dict[str, Any]:
    """Build a bounded content signature used to reject stale cached results."""
    if reference_column not in adata.obs:
        raise KeyError(f"Reference population column {reference_column!r} is missing")
    signature: dict[str, Any] = {
        "signature_version": 1,
        "n_obs": int(adata.n_obs),
        "obs_names_sha256": _hash_text(adata.obs_names.astype(str)),
        "reference_column": reference_column,
        "reference_labels_sha256": _hash_text(
            adata.obs[reference_column].astype("string").fillna("<NA>").astype(str)
        ),
        "metric_configuration": metric_config_snapshot(config),
    }
    dependent_obs: dict[str, str] = {}
    for column in (config.sample_obs, config.roi_obs):
        if column is not None and str(column) in adata.obs:
            dependent_obs[str(column)] = _hash_text(
                adata.obs[str(column)].astype("string").fillna("<NA>").astype(str)
            )
    signature["dependent_obs_sha256"] = dependent_obs
    representations: dict[str, Any] = {}
    for summary_key in ("umap_key", "pca_key"):
        key = run_summary.get(summary_key)
        if key is not None and str(key) in adata.obsm:
            representations[str(key)] = _sampled_array_fingerprint(adata.obsm[str(key)])
    signature["representations"] = representations
    graph_key = run_summary.get("connectivities_key")
    if graph_key is not None and str(graph_key) in adata.obsp:
        signature["connectivities"] = {
            "key": str(graph_key),
            **_sampled_sparse_fingerprint(adata.obsp[str(graph_key)]),
        }
    sweep: dict[str, str] = {}
    for column in run_summary.get("sweep_columns", []):
        if str(column) in adata.obs:
            sweep[str(column)] = _hash_text(
                adata.obs[str(column)].astype("string").fillna("<NA>").astype(str)
            )
    signature["sweep_label_sha256"] = sweep
    signature["evidence_selection"] = {
        "pca_configured_key": config.pca_key,
        "pca_available": bool(run_summary.get("pca_available", False)),
        "connectivities_configured_key": config.connectivities_key,
        "connectivities_resolved_key": run_summary.get("connectivities_key"),
        "graph_available": bool(run_summary.get("graph_available", False)),
        "sweep_columns": list(map(str, run_summary.get("sweep_columns", []))),
    }
    return signature


def _result_payload(result: PopulationEmbeddingQCResult) -> dict[str, Any]:
    frames = {
        "cluster_metrics_raw": result.cluster_metrics_raw,
        "concern_scores": result.concern_scores,
        "threshold_flags": result.threshold_flags,
        "cluster_summary": result.cluster_summary,
        "cluster_competitors": result.cluster_competitors,
        "pairwise_graph_connectivity": result.pairwise_graph_connectivity,
        "pairwise_umap_neighbour_mixing": result.pairwise_umap_neighbour_mixing,
        "pairwise_umap_density_overlap": result.pairwise_umap_density_overlap,
        "detected_sweep_columns": result.detected_sweep_columns,
        "sweep_transition_edges": result.sweep_transition_edges,
        "sweep_best_matches": result.sweep_best_matches,
        "sweep_reference_cluster_metrics": result.sweep_reference_cluster_metrics,
        "sweep_reference_membership": result.sweep_reference_membership,
        "sweep_global_metrics": result.sweep_global_metrics,
    }
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "reference_column": result.reference_column,
        "cluster_order": result.cluster_order,
        "frames": {key: _frame_payload(frame) for key, frame in frames.items()},
        "sweep_pairwise_jaccard": {
            key: _frame_payload(frame)
            for key, frame in result.sweep_pairwise_jaccard.items()
        },
        "metric_definitions": [asdict(item) for item in result.metric_definitions],
        "run_summary": result.run_summary,
        "warnings": result.warnings,
        "per_cluster_text": result.per_cluster_text,
        "obs_column_map": OBS_COLUMN_MAP,
    }


def focused_population_summary(result: PopulationEmbeddingQCResult) -> dict[str, Any]:
    """Return the compact, decision-oriented view intended for Agents."""
    summary = result.cluster_summary
    competitors = (
        result.cluster_competitors.set_index("cluster")
        if "cluster" in result.cluster_competitors
        else pd.DataFrame()
    )
    records: list[dict[str, Any]] = []
    fields = (
        "cluster_size",
        "cluster_percentage",
        "small_cluster",
        "represented_samples",
        "represented_rois",
        "graph_separation_concern",
        "embedding_separation_concern",
        "embedding_reliability_concern",
        "resolution_stability_concern",
        "overall_concern",
        "failed_thresholds",
        "failed_metric_keys",
        "concern_rank",
    )
    for cluster in summary.index.astype(str):
        row = summary.loc[cluster]
        record = {"population": cluster}
        for field in fields:
            if field in summary:
                record[field] = row[field]
        if not competitors.empty and cluster in competitors.index:
            record["strongest_competitor"] = competitors.loc[cluster].get("competitor")
        raw = result.cluster_metrics_raw.loc[cluster]
        for source, target in (
            ("pca_silhouette_median", "feature_space_silhouette"),
            ("umap_silhouette_median", "visual_umap_silhouette"),
            (
                "umap_graph_neighbourhood_preservation",
                "umap_graph_neighbourhood_preservation",
            ),
            ("sweep_persistence_fraction", "resolution_persistence_fraction"),
        ):
            if source in result.cluster_metrics_raw:
                record[target] = raw[source]
        records.append(record)
    return {
        "reference_column": result.reference_column,
        "populations": records,
        "metric_availability": result.run_summary.get("metric_availability", {}),
        "warnings": result.warnings,
    }


def attach_focused_obs(adata: Any, result: PopulationEmbeddingQCResult) -> None:
    """Attach the established, focused cell annotations without overwriting."""
    for source, target in OBS_COLUMN_MAP.items():
        if source not in result.cell_metrics:
            continue
        if target in adata.obs:
            raise ValueError(
                f"Refusing to overwrite existing AnnData observation column {target!r}"
            )
        positions = result.cell_metrics.index.to_numpy(dtype=int)
        if source == "boundary_class":
            values = np.full(adata.n_obs, None, dtype=object)
            values[positions] = result.cell_metrics[source].astype(str).to_numpy()
            adata.obs[target] = pd.Categorical(
                values,
                categories=["core", "intermediate", "boundary"],
            )
        else:
            values = np.full(adata.n_obs, np.nan, dtype=np.float64)
            values[positions] = pd.to_numeric(
                result.cell_metrics[source], errors="coerce"
            ).to_numpy(dtype=float)
            adata.obs[target] = values


def store_population_embedding_qc(
    adata: Any,
    result: PopulationEmbeddingQCResult,
    *,
    config: PopulationEmbeddingQCConfig,
) -> str:
    """Append a complete population-level QC result under the established key."""
    existing = adata.uns.get(UNS_KEY, {})
    if existing is None:
        existing = {}
    if not isinstance(existing, Mapping):
        raise StoredPopulationQCError(
            f"AnnData .uns[{UNS_KEY!r}] exists but is not a mapping"
        )
    container = dict(existing)
    runs_raw = container.get("runs", {})
    runs = dict(runs_raw) if isinstance(runs_raw, Mapping) else {}
    next_index = 1
    while f"run_{next_index:06d}" in runs:
        next_index += 1
    run_id = f"run_{next_index:06d}"
    signature = build_compatibility_signature(
        adata,
        reference_column=result.reference_column,
        run_summary=result.run_summary,
        config=config,
    )
    run_entry = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_encoding": "application/json",
        "result_json": _json_dumps(_result_payload(result)),
        "focused_summary_json": _json_dumps(focused_population_summary(result)),
        "compatibility_signature_json": _json_dumps(signature),
    }
    environment_fields = {
        "project_id": os.environ.get("SBT_PROJECT_ID"),
        "workflow_run_id": os.environ.get("SBT_WORKFLOW_RUN_ID")
        or os.environ.get("SBT_RUN_ID"),
        "execution_id": os.environ.get("SBT_EXECUTION_ID"),
        "technical_run_id": os.environ.get("SBT_TECHNICAL_RUN_ID"),
        "slurm_job_id": os.environ.get("SBT_SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_ID"),
    }
    run_entry.update(
        {
            key: value
            for key, value in environment_fields.items()
            if value is not None and str(value)
        }
    )
    runs[run_id] = run_entry
    container["schema_version"] = STORAGE_SCHEMA_VERSION
    container["latest_run"] = run_id
    container["runs"] = runs
    for key, value in result.run_summary.items():
        cleaned = _uns_value(value)
        if cleaned is not None:
            container[str(key)] = cleaned
    adata.uns[UNS_KEY] = container
    return run_id


def _signature_mismatches(adata: Any, signature: Mapping[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if int(signature.get("n_obs", -1)) != int(adata.n_obs):
        mismatches.append("observation count changed")
    if signature.get("obs_names_sha256") != _hash_text(adata.obs_names.astype(str)):
        mismatches.append("observation names or order changed")
    reference = str(signature.get("reference_column", ""))
    if reference not in adata.obs:
        mismatches.append(f"reference column {reference!r} is missing")
    elif signature.get("reference_labels_sha256") != _hash_text(
        adata.obs[reference].astype("string").fillna("<NA>").astype(str)
    ):
        mismatches.append(f"labels in reference column {reference!r} changed")
    for key, expected in dict(signature.get("representations", {})).items():
        if key not in adata.obsm:
            mismatches.append(f"representation {key!r} is missing")
        elif expected != _sampled_array_fingerprint(adata.obsm[key]):
            mismatches.append(f"representation {key!r} changed")
    graph = signature.get("connectivities")
    if isinstance(graph, Mapping):
        key = str(graph.get("key", ""))
        current = (
            {"key": key, **_sampled_sparse_fingerprint(adata.obsp[key])}
            if key in adata.obsp
            else None
        )
        if current is None:
            mismatches.append(f"connectivity graph {key!r} is missing")
        elif dict(graph) != current:
            mismatches.append(f"connectivity graph {key!r} changed")
    for column, expected in dict(signature.get("sweep_label_sha256", {})).items():
        if column not in adata.obs:
            mismatches.append(f"sweep column {column!r} is missing")
        elif expected != _hash_text(
            adata.obs[column].astype("string").fillna("<NA>").astype(str)
        ):
            mismatches.append(f"labels in sweep column {column!r} changed")
    for column, expected in dict(signature.get("dependent_obs_sha256", {})).items():
        if column not in adata.obs:
            mismatches.append(f"dependent observation column {column!r} is missing")
        elif expected != _hash_text(
            adata.obs[column].astype("string").fillna("<NA>").astype(str)
        ):
            mismatches.append(f"values in observation column {column!r} changed")
    selection = signature.get("evidence_selection", {})
    if isinstance(selection, Mapping):
        pca_key = selection.get("pca_configured_key")
        current_pca_available = pca_key is not None and str(pca_key) in adata.obsm
        if bool(selection.get("pca_available", False)) != current_pca_available:
            mismatches.append(f"PCA availability for {pca_key!r} changed")

        configured_graph_key = selection.get("connectivities_configured_key")
        if configured_graph_key is not None:
            current_graph_key = (
                str(configured_graph_key)
                if str(configured_graph_key) in adata.obsp
                else None
            )
        else:
            neighbours = adata.uns.get("neighbors", {})
            neighbour_key = (
                neighbours.get("connectivities_key")
                if isinstance(neighbours, Mapping)
                else None
            )
            current_graph_key = (
                str(neighbour_key)
                if neighbour_key is not None and str(neighbour_key) in adata.obsp
                else ("connectivities" if "connectivities" in adata.obsp else None)
            )
        if selection.get("connectivities_resolved_key") != current_graph_key:
            mismatches.append("resolved connectivity graph selection changed")

        configuration = signature.get("metric_configuration", {})
        if isinstance(configuration, Mapping):
            from .inspection import detect_sweep_columns

            try:
                current_sweep, _ = detect_sweep_columns(
                    adata.obs,
                    sweep_regex=str(configuration.get("sweep_regex")),
                    explicit_columns=configuration.get("sweep_columns"),
                )
                if configuration.get("mode") == "single" or len(current_sweep) == 1:
                    current_sweep = []
                current_sweep_columns = [str(column) for column, _ in current_sweep]
                if current_sweep_columns != list(selection.get("sweep_columns", [])):
                    mismatches.append("detected sweep column selection changed")
            except Exception as exc:
                mismatches.append(f"sweep column discovery changed: {exc}")
    return mismatches


def _focused_cell_metrics(adata: Any, reference_column: str) -> pd.DataFrame:
    valid = adata.obs[reference_column].notna().to_numpy()
    positions = np.flatnonzero(valid)
    frame = pd.DataFrame(index=positions)
    frame.index.name = "cell_position"
    frame["cell_index"] = adata.obs_names[valid].astype(str)
    frame["reference_population"] = (
        adata.obs.loc[valid, reference_column].astype(str).to_numpy()
    )
    for result_key, obs_key in OBS_COLUMN_MAP.items():
        if obs_key in adata.obs:
            frame[result_key] = adata.obs.iloc[positions][obs_key].to_numpy()
    return frame


def _result_from_payload(adata: Any, payload: Mapping[str, Any]) -> PopulationEmbeddingQCResult:
    version = int(payload.get("result_schema_version", 0))
    if version != RESULT_SCHEMA_VERSION:
        raise StoredPopulationQCError(
            f"Unsupported population QC result schema version {version}"
        )
    frames = {
        key: _frame_from_payload(value)
        for key, value in dict(payload.get("frames", {})).items()
    }
    required = (
        "cluster_metrics_raw",
        "concern_scores",
        "threshold_flags",
        "cluster_summary",
        "cluster_competitors",
        "pairwise_graph_connectivity",
        "pairwise_umap_neighbour_mixing",
        "pairwise_umap_density_overlap",
    )
    missing = [key for key in required if key not in frames]
    if missing:
        raise StoredPopulationQCError(
            "Stored population QC result is missing tables: " + ", ".join(missing)
        )
    reference = str(payload["reference_column"])
    return PopulationEmbeddingQCResult(
        reference_column=reference,
        cluster_order=list(map(str, payload.get("cluster_order", []))),
        cluster_metrics_raw=frames["cluster_metrics_raw"],
        concern_scores=frames["concern_scores"],
        threshold_flags=frames["threshold_flags"],
        cluster_summary=frames["cluster_summary"],
        cluster_competitors=frames["cluster_competitors"],
        pairwise_graph_connectivity=frames["pairwise_graph_connectivity"],
        pairwise_umap_neighbour_mixing=frames["pairwise_umap_neighbour_mixing"],
        pairwise_umap_density_overlap=frames["pairwise_umap_density_overlap"],
        cell_metrics=_focused_cell_metrics(adata, reference),
        metric_definitions=tuple(
            MetricDefinition(**definition)
            for definition in payload.get("metric_definitions", [])
        ),
        run_summary=dict(payload.get("run_summary", {})),
        warnings=list(map(str, payload.get("warnings", []))),
        detected_sweep_columns=frames.get("detected_sweep_columns", pd.DataFrame()),
        sweep_transition_edges=frames.get("sweep_transition_edges", pd.DataFrame()),
        sweep_best_matches=frames.get("sweep_best_matches", pd.DataFrame()),
        sweep_reference_cluster_metrics=frames.get(
            "sweep_reference_cluster_metrics", pd.DataFrame()
        ),
        sweep_reference_membership=frames.get(
            "sweep_reference_membership", pd.DataFrame()
        ),
        sweep_global_metrics=frames.get("sweep_global_metrics", pd.DataFrame()),
        sweep_pairwise_jaccard={
            key: _frame_from_payload(value)
            for key, value in dict(payload.get("sweep_pairwise_jaccard", {})).items()
        },
        per_cluster_text={
            str(key): str(value)
            for key, value in dict(payload.get("per_cluster_text", {})).items()
        },
    )


def list_stored_population_qc(adata: Any) -> pd.DataFrame:
    """Inventory stored population QC runs and current compatibility."""
    container = adata.uns.get(UNS_KEY, {})
    columns = [
        "run_id",
        "created_utc",
        "reference_column",
        "compatible",
        "compatibility_message",
    ]
    if not isinstance(container, Mapping):
        return pd.DataFrame(columns=columns)
    runs = container.get("runs", {})
    if not isinstance(runs, Mapping) or not runs:
        if container:
            return pd.DataFrame(
                [
                    {
                        "run_id": "legacy_summary",
                        "created_utc": "",
                        "reference_column": str(
                            container.get("reference_column", "")
                        ),
                        "compatible": False,
                        "compatibility_message": (
                            "legacy summary only; complete cached results are unavailable"
                        ),
                    }
                ],
                columns=columns,
            )
        return pd.DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for run_id, entry in runs.items():
        if not isinstance(entry, Mapping):
            continue
        try:
            signature = json.loads(str(entry["compatibility_signature_json"]))
            mismatches = _signature_mismatches(adata, signature)
            reference = str(signature.get("reference_column", ""))
        except Exception as exc:
            mismatches = [f"invalid stored signature: {exc}"]
            reference = ""
        records.append(
            {
                "run_id": str(run_id),
                "created_utc": str(entry.get("created_utc", "")),
                "reference_column": reference,
                "compatible": not mismatches,
                "compatibility_message": "; ".join(mismatches) or "compatible",
            }
        )
    return pd.DataFrame(records, columns=columns)


def load_stored_population_qc(
    adata: Any,
    *,
    population_key: str | None = None,
    run_id: str | None = None,
    config: PopulationEmbeddingQCConfig | None = None,
    strict: bool = True,
) -> PopulationEmbeddingQCResult:
    """Load a compatible cached result without recalculating scientific metrics."""
    container = adata.uns.get(UNS_KEY)
    if not isinstance(container, Mapping):
        raise StoredPopulationQCError(
            f"No versioned population QC result is stored in adata.uns[{UNS_KEY!r}]"
        )
    runs = container.get("runs")
    if not isinstance(runs, Mapping) or not runs:
        raise StoredPopulationQCError(
            "Only legacy population QC summary metadata is stored; full results are unavailable"
        )
    selected_id = run_id or str(container.get("latest_run", ""))
    if run_id is None:
        for candidate_id in reversed(list(runs)):
            candidate_entry = runs[candidate_id]
            if not isinstance(candidate_entry, Mapping):
                continue
            try:
                candidate_signature = json.loads(
                    str(candidate_entry["compatibility_signature_json"])
                )
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if (
                population_key is not None
                and str(candidate_signature.get("reference_column", ""))
                != str(population_key)
            ):
                continue
            if _signature_mismatches(adata, candidate_signature):
                continue
            if (
                config is not None
                and candidate_signature.get("metric_configuration")
                != metric_config_snapshot(config)
            ):
                continue
            selected_id = str(candidate_id)
            break
    if selected_id not in runs:
        raise StoredPopulationQCError(f"Stored population QC run {selected_id!r} was not found")
    entry = runs[selected_id]
    if not isinstance(entry, Mapping):
        raise StoredPopulationQCError(f"Stored population QC run {selected_id!r} is invalid")
    try:
        signature = json.loads(str(entry["compatibility_signature_json"]))
        payload = json.loads(str(entry["result_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise StoredPopulationQCError(
            f"Stored population QC run {selected_id!r} cannot be decoded: {exc}"
        ) from exc
    reference = str(signature.get("reference_column", ""))
    mismatches = _signature_mismatches(adata, signature)
    if population_key is not None and str(population_key) != reference:
        mismatches.append(
            f"stored reference column is {reference!r}, not {str(population_key)!r}"
        )
    if config is not None:
        expected = metric_config_snapshot(config)
        if signature.get("metric_configuration") != expected:
            mismatches.append("metric-affecting configuration changed")
    if mismatches and strict:
        raise StoredPopulationQCError(
            f"Stored population QC run {selected_id!r} is incompatible: "
            + "; ".join(dict.fromkeys(mismatches))
        )
    result = _result_from_payload(adata, payload)
    if mismatches:
        result.warnings.append(
            "Loaded despite compatibility mismatches: "
            + "; ".join(dict.fromkeys(mismatches))
        )
    result.run_summary["loaded_from_anndata_uns"] = True
    result.run_summary["stored_run_id"] = selected_id
    return result


__all__ = [
    "OBS_COLUMN_MAP",
    "RESULT_SCHEMA_VERSION",
    "STORAGE_SCHEMA_VERSION",
    "StoredPopulationQCError",
    "UNS_KEY",
    "attach_focused_obs",
    "build_compatibility_signature",
    "focused_population_summary",
    "list_stored_population_qc",
    "load_stored_population_qc",
    "metric_config_snapshot",
    "store_population_embedding_qc",
]
