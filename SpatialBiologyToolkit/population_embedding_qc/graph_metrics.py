"""Sparse existing-graph metrics for population embedding QC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph


@dataclass
class GraphMetricResult:
    cluster_metrics: pd.DataFrame
    cell_metrics: pd.DataFrame
    competitors: pd.DataFrame
    pairwise_connectivity: pd.DataFrame
    graph: sparse.csr_matrix


def _fraction_below(values: np.ndarray, threshold: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite < threshold)) if finite.size else np.nan


def _normalized_entropy(weights: np.ndarray) -> float:
    weights = weights[weights > 0]
    if weights.size <= 1:
        return 0.0
    probabilities = weights / weights.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(weights.size))


def calculate_graph_metrics(
    graph: sparse.spmatrix,
    labels: pd.Series,
    *,
    cluster_order: list[str],
    boundary_threshold: float,
    high_entropy_threshold: float,
    min_component_size: int,
) -> GraphMetricResult:
    """Calculate graph metrics without densifying the cell connectivity graph.

    The input is symmetrized deterministically using the elementwise maximum,
    and self loops are removed before all calculations.
    """
    valid_mask = labels.notna().to_numpy()
    valid_positions = np.flatnonzero(valid_mask)
    valid_labels = labels.iloc[valid_positions].astype(str).to_numpy()
    graph_valid = sparse.csr_matrix(graph[valid_positions][:, valid_positions], dtype=float)
    graph_valid = graph_valid.maximum(graph_valid.T).tocsr()
    graph_valid.setdiag(0)
    graph_valid.eliminate_zeros()

    label_to_code = {label: index for index, label in enumerate(cluster_order)}
    codes = np.asarray([label_to_code[label] for label in valid_labels], dtype=np.int64)
    degrees = np.asarray(graph_valid.sum(axis=1)).ravel()
    purity_numerator = np.zeros(graph_valid.shape[0], dtype=float)
    entropy = np.full(graph_valid.shape[0], np.nan, dtype=float)

    for row in range(graph_valid.shape[0]):
        start, end = graph_valid.indptr[row], graph_valid.indptr[row + 1]
        neighbours = graph_valid.indices[start:end]
        weights = graph_valid.data[start:end]
        if weights.size == 0 or weights.sum() <= 0:
            continue
        neighbour_codes = codes[neighbours]
        purity_numerator[row] = weights[neighbour_codes == codes[row]].sum()
        totals = np.bincount(neighbour_codes, weights=weights, minlength=len(cluster_order))
        entropy[row] = _normalized_entropy(totals)
    purity = np.divide(
        purity_numerator,
        degrees,
        out=np.full_like(degrees, np.nan),
        where=degrees > 0,
    )

    coo = graph_valid.tocoo()
    pairwise = sparse.coo_matrix(
        (coo.data, (codes[coo.row], codes[coo.col])),
        shape=(len(cluster_order), len(cluster_order)),
    ).toarray()
    total_volume = float(degrees.sum())
    records: list[dict[str, object]] = []
    competitor_records: list[dict[str, object]] = []
    strongest_by_cluster: dict[str, str | None] = {}
    for code, cluster in enumerate(cluster_order):
        members = np.flatnonzero(codes == code)
        member_purity = purity[members]
        member_entropy = entropy[members]
        cluster_volume = float(degrees[members].sum())
        internal_weight = float(pairwise[code, code])
        external_weights = pairwise[code].copy()
        external_weights[code] = 0
        leaving_weight = float(external_weights.sum())
        outside_volume = max(0.0, total_volume - cluster_volume)
        denominator = min(cluster_volume, outside_volume)
        conductance = leaving_weight / denominator if denominator > 0 else (0.0 if leaving_weight == 0 else np.nan)
        competitor_code = int(np.argmax(external_weights)) if external_weights.size else -1
        competitor_weight = float(external_weights[competitor_code]) if competitor_code >= 0 else 0.0
        competitor = cluster_order[competitor_code] if competitor_weight > 0 else None
        strongest_by_cluster[cluster] = competitor

        induced = graph_valid[members][:, members]
        if len(members):
            n_components, component_labels = csgraph.connected_components(induced, directed=False)
            sizes = np.bincount(component_labels, minlength=n_components)
            largest_fraction = float(sizes.max() / len(members)) if sizes.size else np.nan
            substantial_components = int(np.sum(sizes >= min_component_size))
        else:
            n_components, largest_fraction, substantial_components = 0, np.nan, 0
        finite_purity = member_purity[np.isfinite(member_purity)]
        finite_entropy = member_entropy[np.isfinite(member_entropy)]
        records.append(
            {
                "cluster": cluster,
                "graph_purity_mean": float(np.mean(finite_purity)) if finite_purity.size else np.nan,
                "graph_purity_median": float(np.median(finite_purity)) if finite_purity.size else np.nan,
                "graph_purity_q25": float(np.quantile(finite_purity, 0.25)) if finite_purity.size else np.nan,
                "graph_purity_p10": float(np.quantile(finite_purity, 0.10)) if finite_purity.size else np.nan,
                "graph_purity_fraction_below_0_5": _fraction_below(member_purity, 0.5),
                "graph_purity_fraction_below_0_7": _fraction_below(member_purity, 0.7),
                "graph_purity_fraction_below_0_9": _fraction_below(member_purity, 0.9),
                "graph_neighbour_impurity": 1 - float(np.median(finite_purity)) if finite_purity.size else np.nan,
                "graph_boundary_fraction": _fraction_below(member_purity, boundary_threshold),
                "graph_conductance": conductance,
                "graph_internal_edge_fraction": internal_weight / cluster_volume if cluster_volume > 0 else np.nan,
                "graph_label_entropy": float(np.median(finite_entropy)) if finite_entropy.size else np.nan,
                "graph_label_entropy_q75": float(np.quantile(finite_entropy, 0.75)) if finite_entropy.size else np.nan,
                "graph_high_entropy_fraction": float(np.mean(finite_entropy > high_entropy_threshold)) if finite_entropy.size else np.nan,
                "strongest_competitor_edge_fraction": competitor_weight / cluster_volume if cluster_volume > 0 else np.nan,
                "graph_component_count": int(n_components),
                "graph_substantial_component_count": substantial_components,
                "graph_largest_component_fraction": largest_fraction,
                "graph_component_loss": 1 - largest_fraction if np.isfinite(largest_fraction) else np.nan,
                "graph_zero_degree_fraction": float(np.mean(degrees[members] <= 0)) if len(members) else np.nan,
            }
        )
        external_rank = np.argsort(-external_weights)
        rank = int(np.flatnonzero(external_rank == competitor_code)[0] + 1) if competitor_code >= 0 else 0
        reciprocal_fraction = (
            float(pairwise[competitor_code, code] / pairwise[competitor_code].sum())
            if competitor_code >= 0 and pairwise[competitor_code].sum() > 0
            else np.nan
        )
        competitor_records.append(
            {
                "cluster": cluster,
                "competitor": competitor,
                "edge_weight": competitor_weight,
                "fraction_total_cluster_connectivity": competitor_weight / cluster_volume if cluster_volume > 0 else np.nan,
                "fraction_external_connectivity": competitor_weight / leaving_weight if leaving_weight > 0 else np.nan,
                "outgoing_rank": rank if competitor is not None else pd.NA,
                "reciprocal_fraction": reciprocal_fraction,
            }
        )

    competitors = pd.DataFrame(competitor_records)
    if not competitors.empty:
        mapping = dict(zip(competitors["cluster"], competitors["competitor"]))
        competitors["reciprocal"] = [
            bool(competitor is not None and mapping.get(str(competitor)) == cluster)
            for cluster, competitor in zip(competitors["cluster"], competitors["competitor"])
        ]
        ranks: list[object] = []
        for cluster, competitor in zip(competitors["cluster"], competitors["competitor"]):
            if competitor is None:
                ranks.append(pd.NA)
                continue
            target_code = label_to_code[str(competitor)]
            source_code = label_to_code[str(cluster)]
            order = np.argsort(-np.where(np.arange(len(cluster_order)) == target_code, 0, pairwise[target_code]))
            ranks.append(int(np.flatnonzero(order == source_code)[0] + 1))
        competitors["reciprocal_rank"] = ranks

    cell = pd.DataFrame(
        {
            "cell_position": valid_positions,
            "reference_population": valid_labels,
            "graph_neighbour_purity": purity,
            "graph_label_entropy": entropy,
        }
    ).set_index("cell_position")
    pairwise_frame = pd.DataFrame(pairwise, index=cluster_order, columns=cluster_order)
    pairwise_frame.index.name = "source_cluster"
    pairwise_frame.columns.name = "target_cluster"
    return GraphMetricResult(
        cluster_metrics=pd.DataFrame(records).set_index("cluster"),
        cell_metrics=cell,
        competitors=competitors,
        pairwise_connectivity=pairwise_frame,
        graph=graph_valid,
    )


__all__ = ["GraphMetricResult", "calculate_graph_metrics"]
