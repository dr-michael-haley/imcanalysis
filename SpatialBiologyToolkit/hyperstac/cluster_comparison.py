"""Reference-free comparison of a multi-parameter HyPERSTAC Leiden scan.

The comparison reads precomputed labels, graphs, embeddings, and visualisation
tables. It never recalculates clustering and never modifies the AnnData input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import combinations
import logging
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)

from SpatialBiologyToolkit.population_embedding_qc.embedding_metrics import (
    stratified_sample_indices,
)
from SpatialBiologyToolkit.population_embedding_qc.graph_metrics import (
    calculate_graph_metrics,
)


LOGGER = logging.getLogger(__name__)
SETTING_PATTERN = re.compile(
    r"^leiden_(?P<resolution>\d+(?:\.\d+)?)_N(?P<n_neighbors>\d+)_P(?P<n_pcs>\d+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ScanSetting:
    column: str
    resolution: float
    n_neighbors: int
    n_pcs: int
    representation_key: str
    neighbors_key: str
    umap_key: str


@dataclass
class ClusterComparisonResult:
    setting_summary: pd.DataFrame
    pairwise_agreement: pd.DataFrame
    parameter_transition_summary: pd.DataFrame
    cluster_summary: pd.DataFrame
    environment_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    environment_membership: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)


def _natural_key(value: object) -> list[object]:
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))
    ]


def parse_setting(column: str) -> tuple[float, int, int]:
    match = SETTING_PATTERN.fullmatch(str(column))
    if match is None:
        raise ValueError(
            f"HyPERSTAC scan column {column!r} does not match "
            "leiden_<resolution>_N<neighbors>_P<pcs>"
        )
    return (
        float(match.group("resolution")),
        int(match.group("n_neighbors")),
        int(match.group("n_pcs")),
    )


def discover_scan_settings(adata: Any) -> list[ScanSetting]:
    """Discover and validate the complete stored HyPERSTAC scan contract."""
    stored = adata.uns.get("cluster_scan", {})
    configured_columns = (
        stored.get("cluster_columns", []) if isinstance(stored, dict) else []
    )
    if isinstance(configured_columns, np.ndarray):
        configured_columns = configured_columns.reshape(-1).tolist()
    if configured_columns:
        columns = [str(column) for column in configured_columns]
    else:
        columns = [
            str(column)
            for column in adata.obs.columns
            if SETTING_PATTERN.fullmatch(str(column))
        ]
    columns = sorted(dict.fromkeys(columns), key=_natural_key)
    if len(columns) < 2:
        raise ValueError("At least two stored HyPERSTAC scan columns are required")
    missing = [column for column in columns if column not in adata.obs]
    if missing:
        raise ValueError(
            f"Stored cluster scan references missing adata.obs columns: {missing}"
        )

    umap_mapping = adata.uns.get("cluster_scan_umap_keys", {})
    neighbors_mapping = adata.uns.get("cluster_scan_neighbors_keys", {})
    if not isinstance(umap_mapping, dict):
        umap_mapping = {}
    if not isinstance(neighbors_mapping, dict):
        neighbors_mapping = {}
    settings: list[ScanSetting] = []
    for column in columns:
        resolution, n_neighbors, n_pcs = parse_setting(column)
        graph_label = f"N{n_neighbors}_P{n_pcs}"
        settings.append(
            ScanSetting(
                column=column,
                resolution=resolution,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                representation_key="X" if n_pcs == 0 else f"X_pca_{n_pcs}",
                neighbors_key=str(
                    neighbors_mapping.get(column, f"neighbors_{graph_label}")
                ),
                umap_key=str(umap_mapping.get(column, f"X_umap_{graph_label}")),
            )
        )
    return settings


def _ordered_labels(series: pd.Series) -> tuple[pd.Series, list[str]]:
    labels = series.astype("string")
    order = sorted(labels.dropna().astype(str).unique(), key=_natural_key)
    return labels, order


def _connectivity_key(adata: Any, neighbors_key: str) -> str | None:
    payload = adata.uns.get(neighbors_key, {})
    candidates: list[str] = []
    if isinstance(payload, dict) and payload.get("connectivities_key"):
        candidates.append(str(payload["connectivities_key"]))
    candidates.extend([f"{neighbors_key}_connectivities", "connectivities"])
    return next((key for key in candidates if key in adata.obsp), None)


def _representation(adata: Any, key: str) -> np.ndarray | None:
    if key == "X":
        matrix = adata.X
    elif key in adata.obsm:
        matrix = adata.obsm[key]
    else:
        return None
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != adata.n_obs:
        raise ValueError(f"Representation {key!r} must have {adata.n_obs} rows")
    return values


def _sampled_silhouette(
    coordinates: np.ndarray | None,
    labels: pd.Series,
    maximum: int,
    seed: int,
) -> tuple[float, float, int]:
    if coordinates is None:
        return np.nan, np.nan, 0
    valid = labels.notna().to_numpy()
    values = labels.loc[valid].astype(str).to_numpy()
    coordinates = np.asarray(coordinates[valid], dtype=float)
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) < 2 or np.any(counts < 2):
        return np.nan, np.nan, 0
    selected = stratified_sample_indices(values, min(maximum, len(values)), seed)
    sampled_labels = values[selected]
    if len(np.unique(sampled_labels)) < 2:
        return np.nan, np.nan, len(selected)
    scores = silhouette_samples(coordinates[selected], sampled_labels)
    return float(np.mean(scores)), float(np.median(scores)), int(len(selected))


def _label_entropy(counts: pd.Series) -> tuple[float, float]:
    probabilities = counts.to_numpy(dtype=float) / max(1.0, float(counts.sum()))
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else np.nan
    normalized = entropy / np.log(len(positive)) if len(positive) > 1 else 0.0
    return normalized, float(np.exp(entropy)) if np.isfinite(entropy) else np.nan


def calculate_setting_metrics(
    adata: Any,
    settings: Iterable[ScanSetting],
    *,
    roi_obs: str,
    min_cluster_fraction: float,
    silhouette_max_patches: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Calculate support and matched-graph metrics for every scan setting."""
    setting_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    roi_values = adata.obs[roi_obs].astype("string") if roi_obs in adata.obs else None
    if roi_values is None:
        warnings.append(
            f"ROI column {roi_obs!r} is unavailable; ROI support metrics were skipped"
        )

    for index, setting in enumerate(settings):
        labels, cluster_order = _ordered_labels(adata.obs[setting.column])
        valid_labels = labels.dropna().astype(str)
        counts = valid_labels.value_counts().reindex(cluster_order, fill_value=0)
        fractions = counts / max(1, int(counts.sum()))
        entropy, effective = _label_entropy(counts)
        cluster_frame = pd.DataFrame(
            {
                "setting": setting.column,
                "cluster": cluster_order,
                "cluster_size": counts.to_numpy(dtype=int),
                "cluster_fraction": fractions.to_numpy(dtype=float),
            }
        )
        if roi_values is not None:
            valid = labels.notna() & roi_values.notna()
            cross = pd.crosstab(
                labels.loc[valid].astype(str), roi_values.loc[valid].astype(str)
            )
            support = (cross > 0).sum(axis=1).reindex(cluster_order, fill_value=0)
            cluster_frame["represented_rois"] = support.to_numpy(dtype=int)
            cluster_frame["roi_support_fraction"] = support.to_numpy(dtype=float) / max(
                1, cross.shape[1]
            )

        graph_key = _connectivity_key(adata, setting.neighbors_key)
        graph_metrics = pd.DataFrame(index=pd.Index(cluster_order, name="cluster"))
        graph_cell = pd.DataFrame()
        if graph_key is None:
            warnings.append(
                f"{setting.column}: matching connectivity graph was not found"
            )
        else:
            graph_result = calculate_graph_metrics(
                adata.obsp[graph_key],
                labels,
                cluster_order=cluster_order,
                boundary_threshold=0.7,
                high_entropy_threshold=0.6,
                min_component_size=5,
            )
            graph_metrics = graph_result.cluster_metrics
            graph_cell = graph_result.cell_metrics
            for column in (
                "graph_purity_median",
                "graph_boundary_fraction",
                "graph_conductance",
                "graph_component_loss",
                "strongest_competitor_edge_fraction",
            ):
                cluster_frame[column] = (
                    graph_metrics[column].reindex(cluster_order).to_numpy()
                )

        representation = _representation(adata, setting.representation_key)
        if representation is None:
            warnings.append(
                f"{setting.column}: representation {setting.representation_key!r} was not found"
            )
        rep_sil_mean, rep_sil_median, rep_sample = _sampled_silhouette(
            representation,
            labels,
            silhouette_max_patches,
            random_seed + index,
        )
        umap = (
            np.asarray(adata.obsm[setting.umap_key])
            if setting.umap_key in adata.obsm
            else None
        )
        if umap is None:
            warnings.append(
                f"{setting.column}: UMAP {setting.umap_key!r} was not found"
            )
        umap_sil_mean, umap_sil_median, umap_sample = _sampled_silhouette(
            umap,
            labels,
            silhouette_max_patches,
            random_seed + 1000 + index,
        )
        cluster_rows.extend(cluster_frame.to_dict(orient="records"))
        setting_rows.append(
            {
                "setting": setting.column,
                "resolution": setting.resolution,
                "n_neighbors": setting.n_neighbors,
                "n_pcs": setting.n_pcs,
                "representation_key": setting.representation_key,
                "neighbors_key": setting.neighbors_key,
                "connectivities_key": graph_key,
                "umap_key": setting.umap_key,
                "n_patches": int(len(valid_labels)),
                "n_clusters": len(cluster_order),
                "min_cluster_fraction": float(fractions.min()),
                "median_cluster_fraction": float(fractions.median()),
                "max_cluster_fraction": float(fractions.max()),
                "clusters_below_min_fraction": int(
                    (fractions < min_cluster_fraction).sum()
                ),
                "label_entropy": entropy,
                "effective_cluster_count": effective,
                "min_roi_support_fraction": float(
                    cluster_frame["roi_support_fraction"].min()
                )
                if "roi_support_fraction" in cluster_frame
                else np.nan,
                "median_roi_support_fraction": float(
                    cluster_frame["roi_support_fraction"].median()
                )
                if "roi_support_fraction" in cluster_frame
                else np.nan,
                "graph_purity_median": float(
                    graph_cell["graph_neighbour_purity"].median()
                )
                if not graph_cell.empty
                else np.nan,
                "graph_boundary_fraction": float(
                    (graph_cell["graph_neighbour_purity"] < 0.7).mean()
                )
                if not graph_cell.empty
                else np.nan,
                "median_graph_conductance": float(
                    graph_metrics["graph_conductance"].median()
                )
                if not graph_metrics.empty
                else np.nan,
                "max_graph_conductance": float(graph_metrics["graph_conductance"].max())
                if not graph_metrics.empty
                else np.nan,
                "median_graph_component_loss": float(
                    graph_metrics["graph_component_loss"].median()
                )
                if not graph_metrics.empty
                else np.nan,
                "representation_silhouette_mean": rep_sil_mean,
                "representation_silhouette_median": rep_sil_median,
                "representation_silhouette_sample_patches": rep_sample,
                "umap_silhouette_mean": umap_sil_mean,
                "umap_silhouette_median": umap_sil_median,
                "umap_silhouette_sample_patches": umap_sample,
            }
        )
    return pd.DataFrame(setting_rows), pd.DataFrame(cluster_rows), warnings


def _mean_best_jaccard(left: pd.Series, right: pd.Series) -> float:
    contingency = pd.crosstab(left.astype(str), right.astype(str)).to_numpy(dtype=float)
    if not contingency.size:
        return np.nan
    left_sizes = contingency.sum(axis=1, keepdims=True)
    right_sizes = contingency.sum(axis=0, keepdims=True)
    union = left_sizes + right_sizes - contingency
    jaccard = np.divide(
        contingency, union, out=np.zeros_like(contingency), where=union > 0
    )
    return float(np.mean(np.concatenate([jaccard.max(axis=1), jaccard.max(axis=0)])))


def _matched_fraction(left: pd.Series, right: pd.Series) -> float:
    contingency = pd.crosstab(left.astype(str), right.astype(str)).to_numpy(dtype=float)
    if not contingency.size:
        return np.nan
    rows, columns = linear_sum_assignment(-contingency)
    return float(contingency[rows, columns].sum() / contingency.sum())


def calculate_pairwise_agreement(
    obs: pd.DataFrame,
    settings: list[ScanSetting],
) -> pd.DataFrame:
    """Compare every partition and identify one-parameter local transitions."""
    unique_values = {
        "resolution": sorted({setting.resolution for setting in settings}),
        "n_neighbors": sorted({setting.n_neighbors for setting in settings}),
        "n_pcs": sorted({setting.n_pcs for setting in settings}),
    }
    records: list[dict[str, object]] = []
    for left, right in combinations(settings, 2):
        valid = obs[left.column].notna() & obs[right.column].notna()
        left_labels = obs.loc[valid, left.column].astype(str)
        right_labels = obs.loc[valid, right.column].astype(str)
        changed = [
            name
            for name in ("resolution", "n_neighbors", "n_pcs")
            if getattr(left, name) != getattr(right, name)
        ]
        changed_parameter = changed[0] if len(changed) == 1 else "multiple"
        adjacent = False
        if len(changed) == 1:
            values = unique_values[changed_parameter]
            adjacent = (
                abs(
                    values.index(getattr(left, changed_parameter))
                    - values.index(getattr(right, changed_parameter))
                )
                == 1
            )
        records.append(
            {
                "left_setting": left.column,
                "right_setting": right.column,
                "left_resolution": left.resolution,
                "right_resolution": right.resolution,
                "left_n_neighbors": left.n_neighbors,
                "right_n_neighbors": right.n_neighbors,
                "left_n_pcs": left.n_pcs,
                "right_n_pcs": right.n_pcs,
                "changed_parameter": changed_parameter,
                "one_parameter_change": len(changed) == 1,
                "adjacent_parameter_change": adjacent,
                "n_patches": int(valid.sum()),
                "adjusted_rand_index": float(
                    adjusted_rand_score(left_labels, right_labels)
                ),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(left_labels, right_labels)
                ),
                "mean_best_cluster_jaccard": _mean_best_jaccard(
                    left_labels, right_labels
                ),
                "optimal_matched_fraction": _matched_fraction(
                    left_labels, right_labels
                ),
            }
        )
    return pd.DataFrame(records)


def add_agreement_summaries(
    setting_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    output = setting_summary.copy().set_index("setting")
    for setting in output.index:
        all_pairs = pairwise.loc[
            (pairwise["left_setting"] == setting)
            | (pairwise["right_setting"] == setting)
        ]
        local = all_pairs.loc[all_pairs["adjacent_parameter_change"]]
        output.loc[setting, "mean_all_ari"] = all_pairs["adjusted_rand_index"].mean()
        output.loc[setting, "mean_all_nmi"] = all_pairs[
            "normalized_mutual_information"
        ].mean()
        output.loc[setting, "mean_local_ari"] = local["adjusted_rand_index"].mean()
        output.loc[setting, "minimum_local_ari"] = local["adjusted_rand_index"].min()
        output.loc[setting, "mean_local_nmi"] = local[
            "normalized_mutual_information"
        ].mean()
        output.loc[setting, "mean_local_jaccard"] = local[
            "mean_best_cluster_jaccard"
        ].mean()
    return output.reset_index()


def summarize_parameter_transitions(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize sensitivity to changing each scan parameter in isolation."""
    records: list[dict[str, object]] = []
    local_pairs = pairwise.loc[pairwise["one_parameter_change"]]
    for parameter in ("resolution", "n_neighbors", "n_pcs"):
        frame = local_pairs.loc[local_pairs["changed_parameter"] == parameter]
        adjacent = frame.loc[frame["adjacent_parameter_change"]]
        records.append(
            {
                "parameter": parameter,
                "one_parameter_comparisons": len(frame),
                "adjacent_comparisons": len(adjacent),
                "mean_one_parameter_ari": float(frame["adjusted_rand_index"].mean()),
                "minimum_one_parameter_ari": float(frame["adjusted_rand_index"].min()),
                "mean_adjacent_ari": float(adjacent["adjusted_rand_index"].mean()),
                "minimum_adjacent_ari": float(adjacent["adjusted_rand_index"].min()),
                "mean_adjacent_nmi": float(
                    adjacent["normalized_mutual_information"].mean()
                ),
                "mean_adjacent_best_cluster_jaccard": float(
                    adjacent["mean_best_cluster_jaccard"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def mark_pareto_candidates(setting_summary: pd.DataFrame) -> pd.DataFrame:
    output = setting_summary.copy()
    objectives = [
        "mean_local_ari",
        "graph_purity_median",
        "representation_silhouette_median",
        "min_cluster_fraction",
        "median_roi_support_fraction",
    ]
    objectives = [
        column
        for column in objectives
        if column in output and output[column].notna().all()
    ]
    pareto = np.ones(len(output), dtype=bool)
    if objectives:
        values = output[objectives].to_numpy(dtype=float)
        for index in range(len(output)):
            dominated = np.all(values >= values[index], axis=1) & np.any(
                values > values[index], axis=1
            )
            dominated[index] = False
            pareto[index] = not bool(dominated.any())
    output["pareto_candidate"] = pareto
    order = output.sort_values(
        [
            "pareto_candidate",
            "mean_local_ari",
            "graph_purity_median",
            "min_cluster_fraction",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).index
    output["review_rank"] = pd.Series(range(1, len(output) + 1), index=order)
    output["pareto_objectives"] = ",".join(objectives)
    return output.sort_values("review_rank").reset_index(drop=True)


def _resolve_visual_dirs(visualisation_dir: Path) -> dict[str, Path]:
    summary_path = visualisation_dir / "all_cluster_visualisation_summary.csv"
    mapping: dict[str, Path] = {}
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        if {"cluster_col", "output_dir"}.issubset(summary.columns):
            for row in summary.itertuples(index=False):
                configured = Path(str(row.output_dir))
                candidate = (
                    configured
                    if configured.is_dir()
                    else visualisation_dir / configured.name
                )
                if candidate.is_dir():
                    mapping[str(row.cluster_col)] = candidate
    for child in visualisation_dir.iterdir() if visualisation_dir.is_dir() else []:
        if child.is_dir() and SETTING_PATTERN.fullmatch(child.name):
            mapping.setdefault(child.name, child)
    return mapping


def calculate_environment_evidence(
    visualisation_dir: Path | None,
    settings: list[ScanSetting],
    *,
    distance_threshold: float,
    top_markers: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Group recurrent marker signatures and add perturbation concordance."""
    if visualisation_dir is None or not visualisation_dir.is_dir():
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            ["Visualisation tree unavailable; marker-environment evidence was skipped"],
        )
    directories = _resolve_visual_dirs(visualisation_dir)
    rows: list[pd.DataFrame] = []
    membership_records: list[dict[str, object]] = []
    warnings: list[str] = []
    for setting in settings:
        directory = directories.get(setting.column)
        if directory is None:
            warnings.append(f"{setting.column}: visualisation folder was not found")
            continue
        table_path = directory / "tables" / "cluster_mean_channel_intensity_zscore.csv"
        if not table_path.is_file():
            warnings.append(f"{setting.column}: marker signature table was not found")
            continue
        signature = pd.read_csv(table_path, index_col=0)
        signature.index = [
            f"{setting.column}::{cluster}" for cluster in signature.index.astype(str)
        ]
        rows.append(signature)

        permutation_path = (
            directory / "tables" / "cluster_mean_permutation_cosine_distance.csv"
        )
        permutation = (
            pd.read_csv(permutation_path, index_col=0)
            if permutation_path.is_file()
            else pd.DataFrame()
        )
        permutation.index = permutation.index.astype(str)
        zero_columns = [
            column
            for column in permutation.columns
            if str(column).startswith("zero_channel__")
        ]
        for uid, marker_values in signature.iterrows():
            cluster = uid.rsplit("::", 1)[-1]
            enriched = (
                marker_values.sort_values(ascending=False)
                .head(top_markers)
                .index.astype(str)
                .tolist()
            )
            sensitive: list[str] = []
            if cluster in permutation.index and zero_columns:
                values = (
                    permutation.loc[cluster, zero_columns]
                    .sort_values(ascending=False)
                    .head(top_markers)
                )
                sensitive = [str(column).split("__", 1)[-1] for column in values.index]
            overlap = (
                len(set(enriched).intersection(sensitive)) / max(1, top_markers)
                if sensitive
                else np.nan
            )
            membership_records.append(
                {
                    "cluster_uid": uid,
                    "setting": setting.column,
                    "cluster": cluster,
                    "top_intensity_markers": ", ".join(enriched),
                    "top_zeroing_sensitivity_markers": ", ".join(sensitive),
                    "marker_perturbation_top_overlap": overlap,
                }
            )
    if not rows:
        return pd.DataFrame(), pd.DataFrame(membership_records), warnings
    signature = pd.concat(rows, axis=0, join="outer").fillna(0.0)
    from SpatialBiologyToolkit.hyperstac.stability import assign_environments

    environments, environment_warnings = assign_environments(
        signature, distance_threshold
    )
    warnings.extend(environment_warnings)
    membership = pd.DataFrame(membership_records)
    membership = membership.merge(
        environments.rename("environment_id").rename_axis("cluster_uid").reset_index(),
        on="cluster_uid",
        how="left",
    )
    total_settings = len(settings)
    summaries: list[dict[str, object]] = []
    for environment_id, group in membership.groupby("environment_id", sort=True):
        uids = group["cluster_uid"].tolist()
        mean_signature = (
            signature.reindex(uids).mean(axis=0).sort_values(ascending=False)
        )
        summaries.append(
            {
                "environment_id": environment_id,
                "n_clusters": len(group),
                "n_settings": int(group["setting"].nunique()),
                "setting_support_fraction": float(
                    group["setting"].nunique() / max(1, total_settings)
                ),
                "top_markers": ", ".join(
                    mean_signature.head(top_markers).index.astype(str)
                ),
                "mean_marker_perturbation_top_overlap": float(
                    group["marker_perturbation_top_overlap"].mean()
                ),
                "settings": ", ".join(
                    sorted(group["setting"].unique(), key=_natural_key)
                ),
                "clusters": ", ".join(group["cluster_uid"].astype(str)),
            }
        )
    return pd.DataFrame(summaries), membership, warnings


def _pairwise_matrix(
    pairwise: pd.DataFrame, settings: list[str], metric: str
) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(settings)), index=settings, columns=settings)
    for row in pairwise.itertuples(index=False):
        value = getattr(row, metric)
        matrix.loc[row.left_setting, row.right_setting] = value
        matrix.loc[row.right_setting, row.left_setting] = value
    return matrix


def _save_pairwise_heatmap(matrix: pd.DataFrame, output: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Clustering settings (parameter-sorted)")
    axis.set_ylabel("Clustering settings (parameter-sorted)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_scorecard(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        "mean_local_ari",
        "minimum_local_ari",
        "graph_purity_median",
        "representation_silhouette_median",
        "min_cluster_fraction",
        "median_roi_support_fraction",
    ]
    metrics = [
        column
        for column in metrics
        if column in summary and summary[column].notna().any()
    ]
    if not metrics:
        return
    values = summary.set_index("setting")[metrics].astype(float)
    scaled = values.copy()
    for column in metrics:
        minimum, maximum = values[column].min(), values[column].max()
        scaled[column] = (
            (values[column] - minimum) / (maximum - minimum)
            if maximum > minimum
            else 0.5
        )
    figure, axis = plt.subplots(figsize=(9, max(6, 0.28 * len(summary))))
    sns.heatmap(
        scaled,
        cmap="mako",
        vmin=0,
        vmax=1,
        ax=axis,
        cbar_kws={"label": "Within-scan relative support"},
    )
    axis.set_title("HyPERSTAC clustering-setting scorecard (higher is better)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.to_numpy().tolist()
    ]
    return "\n".join([header, separator, *rows])


def write_comparison_outputs(
    result: ClusterComparisonResult,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "clustering_setting_scorecard.csv": result.setting_summary,
        "clustering_pairwise_agreement.csv": result.pairwise_agreement,
        "clustering_parameter_transition_summary.csv": result.parameter_transition_summary,
        "clustering_cluster_support.csv": result.cluster_summary,
        "recurrent_environment_summary.csv": result.environment_summary,
        "recurrent_environment_membership.csv": result.environment_membership,
    }
    paths: list[Path] = []
    for name, frame in tables.items():
        if frame.empty and name.startswith("recurrent_"):
            continue
        path = output_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)
    settings = result.setting_summary.sort_values(
        ["n_pcs", "n_neighbors", "resolution"]
    )["setting"].tolist()
    for metric, title in (
        ("adjusted_rand_index", "Adjusted Rand index across HyPERSTAC settings"),
        (
            "normalized_mutual_information",
            "Normalized mutual information across HyPERSTAC settings",
        ),
    ):
        path = output_dir / f"pairwise_{metric}_heatmap.png"
        _save_pairwise_heatmap(
            _pairwise_matrix(result.pairwise_agreement, settings, metric), path, title
        )
        paths.append(path)
    scorecard = output_dir / "clustering_setting_scorecard.png"
    _save_scorecard(result.setting_summary, scorecard)
    if scorecard.is_file():
        paths.append(scorecard)

    shortlist_columns = [
        "review_rank",
        "setting",
        "pareto_candidate",
        "n_clusters",
        "mean_local_ari",
        "minimum_local_ari",
        "graph_purity_median",
        "representation_silhouette_median",
        "min_cluster_fraction",
        "median_roi_support_fraction",
    ]
    shortlist = result.setting_summary[shortlist_columns].head(10).copy()
    for column in shortlist.columns:
        if pd.api.types.is_float_dtype(shortlist[column]):
            shortlist[column] = shortlist[column].round(3)
    transitions = result.parameter_transition_summary.copy()
    for column in transitions.columns:
        if pd.api.types.is_float_dtype(transitions[column]):
            transitions[column] = transitions[column].round(3)
    lines = [
        "# HyPERSTAC clustering parameter comparison",
        "",
        f"- Settings compared: {len(result.setting_summary)}",
        f"- Pairwise comparisons: {len(result.pairwise_agreement)}",
        f"- Patches: {int(result.setting_summary['n_patches'].max())}",
        f"- Pareto candidates: {int(result.setting_summary['pareto_candidate'].sum())}",
        "",
        "The review rank prioritizes non-dominated settings and structural robustness; it is not a biological verdict. "
        "Inspect marker heatmaps and patch galleries before choosing the final granularity.",
        "",
        "## Candidate shortlist",
        "",
        _markdown_table(shortlist),
        "",
        "## Sensitivity by parameter",
        "",
        _markdown_table(transitions),
        "",
        "## Interpretation",
        "",
        "- ARI/NMI measure agreement of patch assignments across parameter settings.",
        "- Local stability uses adjacent settings where only one parameter changes.",
        "- Graph purity and conductance use the graph that generated each Leiden partition.",
        "- Representation silhouette is sampled deterministically and uses the corresponding full/PCA representation.",
        "- UMAP silhouette is descriptive only and is not used in Pareto selection.",
        "- This scan assesses hyperparameter robustness, not seed or bootstrap stability.",
        "",
    ]
    if result.warnings:
        lines.extend(
            ["## Warnings", "", *[f"- {warning}" for warning in result.warnings], ""]
        )
    report = output_dir / "clustering_parameter_scan_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    paths.append(report)
    result.output_files = paths
    return paths


def run_cluster_comparison(
    adata: Any,
    *,
    visualisation_dir: Path | None = None,
    output_dir: Path | None = None,
    roi_obs: str = "roi",
    min_cluster_fraction: float = 0.01,
    silhouette_max_patches: int = 1000,
    environment_distance_threshold: float = 0.45,
    random_seed: int = 1,
) -> ClusterComparisonResult:
    settings = discover_scan_settings(adata)
    setting_summary, cluster_summary, warnings = calculate_setting_metrics(
        adata,
        settings,
        roi_obs=roi_obs,
        min_cluster_fraction=min_cluster_fraction,
        silhouette_max_patches=silhouette_max_patches,
        random_seed=random_seed,
    )
    pairwise = calculate_pairwise_agreement(adata.obs, settings)
    setting_summary = add_agreement_summaries(setting_summary, pairwise)
    parameter_transitions = summarize_parameter_transitions(pairwise)
    environment_summary, membership, environment_warnings = (
        calculate_environment_evidence(
            visualisation_dir,
            settings,
            distance_threshold=environment_distance_threshold,
        )
    )
    warnings.extend(environment_warnings)
    if not membership.empty:
        interpretation = membership.groupby("setting")[
            "marker_perturbation_top_overlap"
        ].mean()
        setting_summary["mean_marker_perturbation_top_overlap"] = setting_summary[
            "setting"
        ].map(interpretation)
    setting_summary = mark_pareto_candidates(setting_summary)
    result = ClusterComparisonResult(
        setting_summary=setting_summary,
        pairwise_agreement=pairwise,
        parameter_transition_summary=parameter_transitions,
        cluster_summary=cluster_summary,
        environment_summary=environment_summary,
        environment_membership=membership,
        warnings=list(dict.fromkeys(warnings)),
    )
    if output_dir is not None:
        write_comparison_outputs(result, output_dir)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a precomputed multi-parameter HyPERSTAC Leiden scan."
    )
    parser.add_argument("--representation-adata", type=Path, required=True)
    parser.add_argument("--visualisation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-obs", default="roi")
    parser.add_argument("--min-cluster-fraction", type=float, default=0.01)
    parser.add_argument("--silhouette-max-patches", type=int, default=1000)
    parser.add_argument("--environment-distance-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.min_cluster_fraction <= 1:
        raise ValueError("--min-cluster-fraction must be between 0 and 1")
    if args.silhouette_max_patches < 100:
        raise ValueError("--silhouette-max-patches must be at least 100")
    if args.environment_distance_threshold < 0:
        raise ValueError("--environment-distance-threshold must be non-negative")
    if not args.representation_adata.is_file():
        raise FileNotFoundError(
            f"HyPERSTAC representation AnnData not found: {args.representation_adata}"
        )
    import anndata

    LOGGER.info(
        "Reading clustered HyPERSTAC representation: %s", args.representation_adata
    )
    adata = anndata.read_h5ad(args.representation_adata)
    result = run_cluster_comparison(
        adata,
        visualisation_dir=args.visualisation_dir,
        output_dir=args.output_dir,
        roi_obs=args.roi_obs,
        min_cluster_fraction=args.min_cluster_fraction,
        silhouette_max_patches=args.silhouette_max_patches,
        environment_distance_threshold=args.environment_distance_threshold,
        random_seed=args.seed,
    )
    LOGGER.info(
        "Compared %d settings and wrote %d files to %s",
        len(result.setting_summary),
        len(result.output_files),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ClusterComparisonResult",
    "ScanSetting",
    "calculate_pairwise_agreement",
    "calculate_setting_metrics",
    "discover_scan_settings",
    "mark_pareto_candidates",
    "parse_setting",
    "run_cluster_comparison",
    "summarize_parameter_transitions",
    "write_comparison_outputs",
]
