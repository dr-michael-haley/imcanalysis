"""Scientific core for one-reference MaxFuse matching.

The first modality is the scRNA-seq reference and the second modality is the
target IMC dataset.  This orientation deliberately supports ``order=(2, 1)``:
each retained target cell receives at most one best reference match, while one
reference cell may annotate several target cells.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
_TRUTHY_MAPPING_VALUES = {"1", "true", "yes", "y", "include", "included"}


@dataclass(frozen=True)
class PreparedMaxFuseInputs:
    """Dense MaxFuse arrays plus the identities needed to audit their origin."""

    reference_active: np.ndarray
    target_active: np.ndarray
    reference_shared: np.ndarray
    target_shared: np.ndarray
    reference_labels: np.ndarray | None
    target_labels: np.ndarray | None
    reference_obs_names: np.ndarray
    target_obs_names: np.ndarray
    reference_active_features: tuple[str, ...]
    target_active_features: tuple[str, ...]
    retained_mapping: pd.DataFrame
    feature_audit: pd.DataFrame


@dataclass(frozen=True)
class MaxFuseMatchResult:
    """Canonical target-unique match table and stage timings."""

    matches: pd.DataFrame
    timings_minutes: Mapping[str, float]


def safe_column_name(value: str) -> str:
    """Return a stable column-safe representation without losing readability."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return normalized or "value"


def dense_size_gib(shape: Sequence[int], *, itemsize: int = 4) -> float:
    """Estimate the dense array size in GiB."""

    size = int(itemsize)
    for value in shape:
        size *= int(value)
    return size / (1024**3)


def _matrix_for(adata: Any, layer: str | None) -> Any:
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"AnnData layer {layer!r} is missing")
    return adata.layers[layer]


def _dense_float32(matrix: Any) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(matrix):
        result = matrix.toarray().astype(np.float32, copy=False)
    else:
        result = np.asarray(matrix, dtype=np.float32)
        if not result.flags.writeable or not result.flags.c_contiguous:
            result = np.array(result, dtype=np.float32, order="C", copy=True)
        else:
            result = result.copy()
    if result.ndim != 2:
        raise ValueError(f"Expected a two-dimensional expression matrix, received {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Expression matrices must contain only finite values")
    return result


def column_mean_sd(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return numerically stable per-column means and standard deviations."""

    from scipy import sparse

    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0), dtype=np.float64).ravel()
        means_sq = np.asarray(matrix.multiply(matrix).mean(axis=0), dtype=np.float64).ravel()
        variances = np.maximum(means_sq - means**2, 0.0)
        return means, np.sqrt(variances)
    values = np.asarray(matrix)
    return (
        values.mean(axis=0, dtype=np.float64),
        values.std(axis=0, dtype=np.float64),
    )


def zscore_float32(
    values: np.ndarray,
    *,
    return_keep: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Z-score columns in float32 and remove non-varying features."""

    array = np.asarray(values, dtype=np.float32)
    means = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviations = array.std(axis=0, dtype=np.float64).astype(np.float32)
    keep = np.isfinite(standard_deviations) & (standard_deviations > 1e-8)
    if not np.any(keep):
        raise ValueError("No varying features remain after standardization")
    result = array[:, keep].copy()
    result -= means[keep]
    result /= standard_deviations[keep]
    if not np.isfinite(result).all():
        raise ValueError("Standardization produced non-finite values")
    if return_keep:
        return result, keep
    return result


def read_feature_mapping(
    path: str | Path,
    *,
    target_column: str,
    reference_column: str,
    filter_column: str | None,
) -> pd.DataFrame:
    """Read, filter, and validate a one-to-one linked-feature mapping."""

    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"MaxFuse feature mapping not found: {mapping_path}")
    mapping = pd.read_csv(mapping_path)
    missing = sorted({target_column, reference_column} - set(mapping.columns))
    if missing:
        raise ValueError(
            f"Feature mapping {mapping_path} is missing required columns: {missing}"
        )
    if filter_column and filter_column in mapping.columns:
        values = mapping[filter_column]
        if pd.api.types.is_bool_dtype(values):
            selected = values.fillna(False)
        elif pd.api.types.is_numeric_dtype(values):
            selected = values.fillna(0).astype(float).ne(0)
        else:
            selected = (
                values.fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(_TRUTHY_MAPPING_VALUES)
            )
        mapping = mapping.loc[selected].copy()
    mapping[target_column] = mapping[target_column].astype("string").str.strip()
    mapping[reference_column] = mapping[reference_column].astype("string").str.strip()
    mapping = mapping.loc[
        mapping[target_column].notna()
        & mapping[reference_column].notna()
        & mapping[target_column].ne("")
        & mapping[reference_column].ne("")
    ].copy()
    mapping = mapping.drop_duplicates(subset=[target_column, reference_column])
    duplicated_target = mapping[target_column].duplicated(keep=False)
    duplicated_reference = mapping[reference_column].duplicated(keep=False)
    if duplicated_target.any() or duplicated_reference.any():
        raise ValueError(
            "MaxFuse linked-feature mapping must be one-to-one after filtering; "
            "duplicate target or reference features were found"
        )
    if mapping.empty:
        raise ValueError("MaxFuse feature mapping contains no eligible rows")
    return mapping.reset_index(drop=True)


def _validate_adata(adata: Any, *, name: str, layer: str | None) -> None:
    if int(adata.n_obs) < 2 or int(adata.n_vars) < 2:
        raise ValueError(f"{name} AnnData must contain at least two cells and features")
    if not adata.obs_names.is_unique:
        raise ValueError(f"{name} AnnData observation names must be unique")
    if not adata.var_names.is_unique:
        raise ValueError(f"{name} AnnData feature names must be unique")
    _matrix_for(adata, layer)


def _required_obs(adata: Any, keys: Sequence[str | None], *, name: str) -> None:
    missing = sorted({key for key in keys if key and key not in adata.obs.columns})
    if missing:
        raise KeyError(f"{name} AnnData is missing observation columns: {missing}")


def _obs_labels(adata: Any, key: str | None) -> np.ndarray | None:
    if key is None:
        return None
    return (
        adata.obs[key]
        .astype(object)
        .where(adata.obs[key].notna(), "Missing")
        .astype(str)
        .to_numpy()
    )


def select_reference_active_features(
    reference: Any,
    *,
    layer: str | None,
    limit: int,
) -> list[str]:
    """Select recorded HVGs, falling back to the highest-variance genes."""

    var = reference.var
    if "highly_variable" in var.columns and var["highly_variable"].fillna(False).any():
        candidates = var.loc[var["highly_variable"].fillna(False)].copy()
        if "highly_variable_rank" in candidates.columns:
            candidates = candidates.sort_values(
                "highly_variable_rank",
                kind="stable",
                na_position="last",
            )
        selected = candidates.index.astype(str).tolist()[:limit]
    else:
        _, deviations = column_mean_sd(_matrix_for(reference, layer))
        order = np.argsort(-np.nan_to_num(deviations, nan=-np.inf), kind="stable")
        selected = reference.var_names[order[:limit]].astype(str).tolist()
    if len(selected) < 2:
        raise ValueError("At least two varying reference active features are required")
    return selected


def prepare_maxfuse_inputs(
    reference: Any,
    target: Any,
    mapping: pd.DataFrame,
    settings: Any,
) -> PreparedMaxFuseInputs:
    """Validate AnnData inputs and build the four arrays expected by MaxFuse."""

    _validate_adata(reference, name="Reference", layer=settings.reference_layer)
    _validate_adata(target, name="Target", layer=settings.target_layer)
    target_population = settings.target_population_obs
    target_smoothing = settings.target_smoothing_obs or target_population
    _required_obs(
        reference,
        [settings.reference_smoothing_obs, *settings.reference_transfer_obs],
        name="Reference",
    )
    _required_obs(
        target,
        [target_smoothing, target_population, settings.sample_obs, settings.roi_obs],
        name="Target",
    )

    target_column = settings.target_feature_column
    reference_column = settings.reference_feature_column
    audit = mapping.copy()
    target_names = pd.Index(target.var_names.astype(str))
    reference_names = pd.Index(reference.var_names.astype(str))
    audit["present_target"] = audit[target_column].isin(target_names)
    audit["present_reference"] = audit[reference_column].isin(reference_names)
    present = audit["present_target"] & audit["present_reference"]

    audit["target_mean"] = np.nan
    audit["target_sd"] = np.nan
    audit["reference_mean"] = np.nan
    audit["reference_sd"] = np.nan
    if present.any():
        target_positions = target_names.get_indexer(audit.loc[present, target_column])
        reference_positions = reference_names.get_indexer(
            audit.loc[present, reference_column]
        )
        target_mean, target_sd = column_mean_sd(
            _matrix_for(target, settings.target_layer)[:, target_positions]
        )
        reference_mean, reference_sd = column_mean_sd(
            _matrix_for(reference, settings.reference_layer)[:, reference_positions]
        )
        audit.loc[present, "target_mean"] = target_mean
        audit.loc[present, "target_sd"] = target_sd
        audit.loc[present, "reference_mean"] = reference_mean
        audit.loc[present, "reference_sd"] = reference_sd

    audit["passes_variance"] = (
        audit["target_sd"].gt(float(settings.target_shared_sd_min))
        & audit["reference_sd"].gt(float(settings.reference_shared_sd_min))
    )
    audit["decision"] = np.select(
        [
            ~audit["present_target"],
            ~audit["present_reference"],
            audit["passes_variance"].fillna(False),
        ],
        [
            "exclude: absent from target",
            "exclude: absent from reference",
            "retain",
        ],
        default="exclude: below variability threshold",
    )
    retained = audit.loc[audit["decision"].eq("retain")].copy().reset_index(drop=True)
    if len(retained) < int(settings.min_shared_features):
        raise ValueError(
            "Too few linked features remain for MaxFuse: "
            f"{len(retained)} retained, {settings.min_shared_features} required"
        )

    reference_shared_positions = reference_names.get_indexer(
        retained[reference_column]
    )
    target_shared_positions = target_names.get_indexer(retained[target_column])
    reference_shared = _dense_float32(
        _matrix_for(reference, settings.reference_layer)[:, reference_shared_positions]
    )
    target_shared = _dense_float32(
        _matrix_for(target, settings.target_layer)[:, target_shared_positions]
    )

    active_reference_names = select_reference_active_features(
        reference,
        layer=settings.reference_layer,
        limit=int(settings.reference_active_features),
    )
    reference_active_positions = reference_names.get_indexer(active_reference_names)
    reference_active = _dense_float32(
        _matrix_for(reference, settings.reference_layer)[:, reference_active_positions]
    )
    target_active_names = target_names.astype(str).tolist()
    target_active = _dense_float32(_matrix_for(target, settings.target_layer))

    if settings.zscore_reference:
        reference_active, reference_active_keep = zscore_float32(
            reference_active,
            return_keep=True,
        )
        active_reference_names = np.asarray(active_reference_names)[
            reference_active_keep
        ].tolist()
        reference_shared = cast(np.ndarray, zscore_float32(reference_shared))
    else:
        _, reference_active_sd = column_mean_sd(reference_active)
        reference_active_keep = np.isfinite(reference_active_sd) & (
            reference_active_sd > 1e-8
        )
        reference_active = reference_active[:, reference_active_keep]
        active_reference_names = np.asarray(active_reference_names)[
            reference_active_keep
        ].tolist()
    if settings.zscore_target:
        target_active, target_active_keep = zscore_float32(
            target_active,
            return_keep=True,
        )
        target_active_names = np.asarray(target_active_names)[target_active_keep].tolist()
        target_shared = cast(np.ndarray, zscore_float32(target_shared))
    else:
        _, target_active_sd = column_mean_sd(target_active)
        target_active_keep = np.isfinite(target_active_sd) & (target_active_sd > 1e-8)
        target_active = target_active[:, target_active_keep]
        target_active_names = np.asarray(target_active_names)[target_active_keep].tolist()

    component_limits = {
        "graph_svd_reference": (
            settings.graph_svd_reference,
            reference_active.shape[1],
        ),
        "graph_svd_target": (settings.graph_svd_target, target_active.shape[1]),
        "initial_svd_reference": (
            settings.initial_svd_reference,
            reference_shared.shape[1],
        ),
        "initial_svd_target": (
            settings.initial_svd_target,
            target_shared.shape[1],
        ),
        "refine_svd_reference": (
            settings.refine_svd_reference,
            reference_active.shape[1],
        ),
        "refine_svd_target": (
            settings.refine_svd_target,
            target_active.shape[1],
        ),
        "refine_cca_components": (
            settings.refine_cca_components,
            min(reference_active.shape[1], target_active.shape[1]),
        ),
    }
    invalid = {
        name: (int(value), int(limit))
        for name, (value, limit) in component_limits.items()
        if value is not None and int(value) > int(limit)
    }
    if invalid:
        details = ", ".join(
            f"{name}={value} exceeds available dimension {limit}"
            for name, (value, limit) in invalid.items()
        )
        raise ValueError(f"MaxFuse component settings are incompatible with inputs: {details}")

    LOGGER.info(
        "Prepared MaxFuse arrays: reference active=%s shared=%s; target active=%s shared=%s",
        reference_active.shape,
        reference_shared.shape,
        target_active.shape,
        target_shared.shape,
    )
    return PreparedMaxFuseInputs(
        reference_active=reference_active,
        target_active=target_active,
        reference_shared=reference_shared,
        target_shared=target_shared,
        reference_labels=_obs_labels(reference, settings.reference_smoothing_obs),
        target_labels=_obs_labels(target, target_smoothing),
        reference_obs_names=reference.obs_names.astype(str).to_numpy(),
        target_obs_names=target.obs_names.astype(str).to_numpy(),
        reference_active_features=tuple(active_reference_names),
        target_active_features=tuple(target_active_names),
        retained_mapping=retained,
        feature_audit=audit,
    )


def extract_target_unique_matching(
    fusor: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Efficiently reproduce MaxFuse 0.0.2 ``order=(2, 1)`` extraction.

    MaxFuse 0.0.2 performs NumPy membership checks in a Python loop over every
    target cell.  Its private lookup dictionaries contain the same information,
    allowing linear extraction.  A public-API fallback supports future versions.
    """

    pivot_lookup = getattr(fusor, "_pivot2_to_pivots1", None)
    propagated_lookup = getattr(fusor, "_propidx2_to_propindices1", None)
    active_target = getattr(fusor, "active_arr2", None)
    if (
        isinstance(pivot_lookup, Mapping)
        and isinstance(propagated_lookup, Mapping)
        and active_target is not None
    ):
        target_size = int(active_target.shape[0])
        best_reference: np.ndarray = np.full(target_size, -1, dtype=np.int64)
        best_score: np.ndarray = np.full(target_size, -np.inf, dtype=np.float32)
        source: np.ndarray = np.full(target_size, "", dtype=object)
        for target_index, candidates in pivot_lookup.items():
            for reference_index, score in candidates:
                if float(score) > float(best_score[int(target_index)]):
                    best_reference[int(target_index)] = int(reference_index)
                    best_score[int(target_index)] = float(score)
                    source[int(target_index)] = "pivot"
        has_pivot = best_reference >= 0
        for target_index, candidates in propagated_lookup.items():
            target_index = int(target_index)
            if has_pivot[target_index]:
                continue
            for reference_index, score in candidates:
                if float(score) > float(best_score[target_index]):
                    best_reference[target_index] = int(reference_index)
                    best_score[target_index] = float(score)
                    source[target_index] = "propagated"
        retained_target = np.flatnonzero(best_reference >= 0)
        return (
            best_reference[retained_target],
            retained_target,
            best_score[retained_target],
            source[retained_target].astype(str),
        )

    LOGGER.warning(
        "MaxFuse lookup dictionaries are unavailable; falling back to the slower public get_matching API"
    )
    public = fusor.get_matching(order=(2, 1), target="full_data")
    frame = pd.DataFrame(
        {
            "reference_index": np.asarray(public[0], dtype=np.int64),
            "target_index": np.asarray(public[1], dtype=np.int64),
            "score": np.asarray(public[2], dtype=np.float32),
        }
    )
    frame = (
        frame.sort_values("score", kind="stable")
        .drop_duplicates("target_index", keep="last")
        .sort_values("target_index", kind="stable")
    )
    return (
        frame["reference_index"].to_numpy(dtype=np.int64),
        frame["target_index"].to_numpy(dtype=np.int64),
        frame["score"].to_numpy(dtype=np.float32),
        np.full(len(frame), "unknown", dtype=str),
    )


def _match_table(
    prepared: PreparedMaxFuseInputs,
    reference: Any,
    target: Any,
    settings: Any,
    reference_indices: np.ndarray,
    target_indices: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "reference_index": reference_indices,
            "target_index": target_indices,
            "mod1_indx": reference_indices,
            "mod2_indx": target_indices,
            "score": scores.astype(np.float32),
            "match_source": sources,
            "reference_obs_name": prepared.reference_obs_names[reference_indices],
            "target_obs_name": prepared.target_obs_names[target_indices],
        }
    )
    table["rna_originalindex"] = table["reference_obs_name"]
    table["protein_originalindex"] = table["target_obs_name"]
    if settings.target_population_obs:
        table["target_population"] = (
            target.obs.iloc[target_indices][settings.target_population_obs].to_numpy()
        )
    for column in settings.reference_transfer_obs:
        output_column = f"reference_{safe_column_name(column)}"
        table[output_column] = reference.obs.iloc[reference_indices][column].to_numpy()
    if table["target_index"].duplicated().any():
        raise RuntimeError("MaxFuse target-unique extraction produced duplicate target cells")
    return table


def run_maxfuse_matching(
    prepared: PreparedMaxFuseInputs,
    reference: Any,
    target: Any,
    settings: Any,
) -> MaxFuseMatchResult:
    """Run the historical spatial-omics MaxFuse recipe."""

    try:
        import maxfuse as mf
    except ImportError as error:
        raise ImportError(
            "The dedicated MaxFuse environment is required; install maxfuse==0.0.2"
        ) from error

    timings: dict[str, float] = {}
    started = perf_counter()
    fusor = mf.model.Fusor(
        shared_arr1=prepared.reference_shared,
        shared_arr2=prepared.target_shared,
        active_arr1=prepared.reference_active,
        active_arr2=prepared.target_active,
        method="centroid_shrinkage",
        labels1=prepared.reference_labels,
        labels2=prepared.target_labels,
    )

    tick = perf_counter()
    fusor.split_into_batches(
        max_outward_size=int(settings.max_outward_size),
        matching_ratio=int(settings.matching_ratio),
        metacell_size=int(settings.metacell_size),
        batching_scheme=settings.batching_scheme,
        seed=int(settings.seed),
        verbose=True,
    )
    timings["split_into_batches"] = (perf_counter() - tick) / 60

    tick = perf_counter()
    fusor.construct_graphs(
        n_neighbors1=int(settings.n_neighbors_reference),
        n_neighbors2=int(settings.n_neighbors_target),
        svd_components1=int(settings.graph_svd_reference),
        svd_components2=settings.graph_svd_target,
        resolution1=float(settings.graph_resolution_reference),
        resolution2=float(settings.graph_resolution_target),
        resolution_tol=float(settings.graph_resolution_tolerance),
        randomized_svd=bool(settings.randomized_svd),
        svd_runs=1,
        leiden_seed=int(settings.seed),
        verbose=True,
    )
    timings["construct_graphs"] = (perf_counter() - tick) / 60

    tick = perf_counter()
    fusor.find_initial_pivots(
        wt1=float(settings.initial_weight_reference),
        wt2=float(settings.initial_weight_target),
        svd_components1=int(settings.initial_svd_reference),
        svd_components2=int(settings.initial_svd_target),
        randomized_svd=False,
        svd_runs=1,
        verbose=True,
    )
    timings["initial_pivots"] = (perf_counter() - tick) / 60

    tick = perf_counter()
    fusor.refine_pivots(
        wt1=float(settings.refine_weight_reference),
        wt2=float(settings.refine_weight_target),
        svd_components1=int(settings.refine_svd_reference),
        svd_components2=settings.refine_svd_target,
        cca_components=int(settings.refine_cca_components),
        n_iters=int(settings.refine_iterations),
        randomized_svd=bool(settings.randomized_svd),
        svd_runs=1,
        verbose=True,
    )
    fusor.filter_bad_matches(
        target="pivot",
        filter_prop=float(settings.pivot_filter_fraction),
    )
    timings["refine_and_filter_pivots"] = (perf_counter() - tick) / 60

    tick = perf_counter()
    fusor.propagate(
        svd_components1=int(settings.refine_svd_reference),
        svd_components2=settings.refine_svd_target,
        wt1=float(settings.propagation_weight_reference),
        wt2=float(settings.propagation_weight_target),
        randomized_svd=bool(settings.randomized_svd),
        svd_runs=1,
        verbose=True,
    )
    fusor.filter_bad_matches(
        target="propagated",
        filter_prop=float(settings.propagated_filter_fraction),
    )
    timings["propagate_and_filter"] = (perf_counter() - tick) / 60

    tick = perf_counter()
    reference_indices, target_indices, scores, sources = (
        extract_target_unique_matching(fusor)
    )
    timings["extract_matching"] = (perf_counter() - tick) / 60
    timings["total"] = (perf_counter() - started) / 60
    matches = _match_table(
        prepared,
        reference,
        target,
        settings,
        reference_indices,
        target_indices,
        scores,
        sources,
    )
    return MaxFuseMatchResult(matches=matches, timings_minutes=timings)


def build_transfer_anndata(
    target: Any,
    matches: pd.DataFrame,
    retained_mapping: pd.DataFrame,
    settings: Any,
    *,
    reference_path: str | Path,
) -> Any:
    """Build a target-indexed annotation-only AnnData for downstream QC."""

    import anndata as ad
    from scipy import sparse

    source = safe_column_name(settings.reference_name)
    score_column = f"{source}_maxfuse_score"
    transfer_columns = {
        column: f"{source}_{safe_column_name(column)}"
        for column in settings.reference_transfer_obs
    }
    obs = pd.DataFrame(index=target.obs_names.astype(str).copy())
    obs[score_column] = np.nan
    target_indices = matches["target_index"].to_numpy(dtype=np.int64)
    obs.iloc[target_indices, obs.columns.get_loc(score_column)] = matches[
        "score"
    ].to_numpy(dtype=np.float32)
    for original, output in transfer_columns.items():
        values = pd.Series(np.nan, index=obs.index, dtype=object)
        values.iloc[target_indices] = (
            matches[f"reference_{safe_column_name(original)}"]
            .astype(object)
            .to_numpy()
        )
        obs[output] = pd.Categorical(values)
    result = ad.AnnData(
        X=sparse.csr_matrix((len(obs), 0), dtype=np.float32),
        obs=obs,
    )
    source_manifest = {
        "name": source,
        "score_column": score_column,
        "label_columns": list(transfer_columns.values()),
        "label_roles": {
            transfer_columns[column]: role
            for column, role in settings.reference_label_roles.items()
        },
        "score_threshold": float(settings.report_score_threshold),
    }
    result.uns["maxfuse"] = {
        "schema_version": 1,
        "sources": {source: source_manifest},
        "parameters_json": json.dumps(
            settings.model_dump(mode="json"),
            sort_keys=True,
        ),
        "score_semantics": (
            "MaxFuse matching similarity; higher is better; not a calibrated probability"
        ),
    }
    result.uns["rna_reference_paths"] = {source: str(reference_path)}
    result.uns["shared_proteins"] = retained_mapping[
        settings.target_feature_column
    ].astype(str).tolist()
    result.uns["shared_genes"] = retained_mapping[
        settings.reference_feature_column
    ].astype(str).tolist()
    return result


def build_matched_transcriptomes(
    reference: Any,
    matches: pd.DataFrame,
    settings: Any,
) -> Any:
    """Materialize the optional target-indexed confident transcriptome asset."""

    selected = matches.loc[
        matches["score"].gt(float(settings.matched_transcriptome_min_score))
    ].copy()
    if selected.empty:
        raise ValueError(
            "No matches exceed matched_transcriptome_min_score; "
            "the matched-transcriptome asset was not created"
        )
    result = reference[selected["reference_index"].to_numpy(dtype=np.int64), :].copy()
    result.obs["maxfuse_reference_obs_name"] = selected[
        "reference_obs_name"
    ].to_numpy()
    result.obs["maxfuse_score"] = selected["score"].to_numpy(dtype=np.float32)
    result.obs_names = selected["target_obs_name"].astype(str).to_numpy()
    result.uns["maxfuse"] = {
        "schema_version": 1,
        "reference_name": str(settings.reference_name),
        "score_threshold": float(settings.matched_transcriptome_min_score),
        "score_semantics": (
            "MaxFuse matching similarity; higher is better; not a calibrated probability"
        ),
    }
    return result


__all__ = [
    "MaxFuseMatchResult",
    "PreparedMaxFuseInputs",
    "build_matched_transcriptomes",
    "build_transfer_anndata",
    "column_mean_sd",
    "dense_size_gib",
    "extract_target_unique_matching",
    "prepare_maxfuse_inputs",
    "read_feature_mapping",
    "run_maxfuse_matching",
    "safe_column_name",
    "select_reference_active_features",
    "zscore_float32",
]
