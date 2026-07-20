"""Scalable metrics computed from existing UMAP and PCA coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse import csgraph
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


@dataclass
class EmbeddingMetricResult:
    cluster_metrics: pd.DataFrame
    cell_metrics: pd.DataFrame
    density_overlap: pd.DataFrame
    umap_mixing: pd.DataFrame
    umap_graph: sparse.csr_matrix
    global_metrics: dict[str, float | int | bool] = field(default_factory=dict)
    sampling: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def stratified_sample_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Return a deterministic sample that retains every small cluster."""
    n = len(labels)
    if n <= maximum:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    allocations = np.maximum(2, np.floor(maximum * counts / counts.sum()).astype(int))
    allocations = np.minimum(allocations, counts)
    preserve_limit = max(2, maximum // max(1, len(unique)))
    allocations[counts <= preserve_limit] = counts[counts <= preserve_limit]
    while allocations.sum() > maximum:
        candidates = np.flatnonzero(allocations > np.minimum(counts, 2))
        if not candidates.size:
            break
        allocations[candidates[np.argmax(allocations[candidates])]] -= 1
    while allocations.sum() < maximum:
        candidates = np.flatnonzero(allocations < counts)
        if not candidates.size:
            break
        allocations[candidates[np.argmax(counts[candidates] - allocations[candidates])]] += 1
    selected: list[np.ndarray] = []
    for label, allocation in zip(unique, allocations):
        positions = np.flatnonzero(labels == label)
        selected.append(np.sort(rng.choice(positions, size=int(allocation), replace=False)))
    return np.sort(np.concatenate(selected))


def _knn(coordinates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(coordinates) < 2:
        return np.empty((len(coordinates), 0), dtype=int), np.empty((len(coordinates), 0))
    count = min(k + 1, len(coordinates))
    distances, indices = NearestNeighbors(n_neighbors=count).fit(coordinates).kneighbors(coordinates)
    clean_indices: np.ndarray = np.empty((len(coordinates), count - 1), dtype=int)
    clean_distances: np.ndarray = np.empty((len(coordinates), count - 1), dtype=float)
    for row in range(len(coordinates)):
        keep = indices[row] != row
        clean_indices[row] = indices[row][keep][: count - 1]
        clean_distances[row] = distances[row][keep][: count - 1]
    return clean_indices, clean_distances


def _purity_and_mixing(indices: np.ndarray, codes: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    if indices.shape[1] == 0:
        return np.full(len(codes), np.nan), np.zeros((n_clusters, n_clusters))
    neighbour_codes = codes[indices]
    purity = np.mean(neighbour_codes == codes[:, None], axis=1)
    mixing: np.ndarray = np.zeros((n_clusters, n_clusters), dtype=float)
    np.add.at(mixing, (np.repeat(codes, indices.shape[1]), neighbour_codes.ravel()), 1)
    return purity, mixing


def _silhouette(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    values = np.full(len(labels), np.nan)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.any(counts < 2):
        return values, np.array([], dtype=int), False
    selected = stratified_sample_indices(labels, maximum, seed)
    if len(np.unique(labels[selected])) < 2:
        return values, selected, len(selected) < len(labels)
    values[selected] = silhouette_samples(coordinates[selected], labels[selected], metric="euclidean")
    return values, selected, len(selected) < len(labels)


def _density_overlap(
    coordinates: np.ndarray,
    codes: np.ndarray,
    cluster_order: list[str],
    *,
    maximum_per_cluster: int,
    grid_size: int,
    min_cluster_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    lower = np.min(coordinates, axis=0)
    upper = np.max(coordinates, axis=0)
    span = np.maximum(upper - lower, np.finfo(float).eps)
    lower = lower - 0.02 * span
    upper = upper + 0.02 * span
    rng = np.random.default_rng(seed)
    densities: list[np.ndarray | None] = []
    sampled: dict[str, int] = {}
    for code, cluster in enumerate(cluster_order):
        positions = np.flatnonzero(codes == code)
        if len(positions) < max(3, min_cluster_size):
            densities.append(None)
            sampled[cluster] = len(positions)
            continue
        if len(positions) > maximum_per_cluster:
            positions = np.sort(rng.choice(positions, maximum_per_cluster, replace=False))
        sampled[cluster] = len(positions)
        histogram, _, _ = np.histogram2d(
            coordinates[positions, 0],
            coordinates[positions, 1],
            bins=grid_size,
            range=[[lower[0], upper[0]], [lower[1], upper[1]]],
        )
        density = gaussian_filter(histogram.astype(float), sigma=1.0)
        total = density.sum()
        densities.append(density / total if total > 0 else None)
    overlap = np.full((len(cluster_order), len(cluster_order)), np.nan)
    for left in range(len(cluster_order)):
        if densities[left] is None:
            continue
        overlap[left, left] = 1.0
        for right in range(left + 1, len(cluster_order)):
            if densities[right] is None:
                continue
            left_density = densities[left]
            right_density = densities[right]
            assert left_density is not None and right_density is not None
            value = float(np.minimum(left_density, right_density).sum())
            overlap[left, right] = overlap[right, left] = float(np.clip(value, 0, 1))
    frame = pd.DataFrame(overlap, index=cluster_order, columns=cluster_order)
    frame.index.name = "cluster"
    return frame, sampled


def _isolation_ratio(coordinates: np.ndarray, codes: np.ndarray, n_clusters: int) -> np.ndarray:
    """Calculate exact nearest-external / robust-nearest-within ratios by cluster."""
    ratios = np.full(len(codes), np.nan)
    for code in range(n_clusters):
        members = np.flatnonzero(codes == code)
        external = np.flatnonzero(codes != code)
        if len(members) < 2 or not len(external):
            continue
        within_count = min(4, len(members))
        within_distances, within_indices = NearestNeighbors(n_neighbors=within_count).fit(
            coordinates[members]
        ).kneighbors(coordinates[members])
        robust_within = np.full(len(members), np.nan)
        for row, member in enumerate(members):
            keep = members[within_indices[row]] != member
            distances = within_distances[row][keep]
            distances = distances[distances > 0]
            if distances.size:
                robust_within[row] = float(np.median(distances[:3]))
        external_distance = NearestNeighbors(n_neighbors=1).fit(
            coordinates[external]
        ).kneighbors(coordinates[members], return_distance=True)[0][:, 0]
        ratios[members] = np.divide(
            external_distance,
            robust_within,
            out=np.full(len(members), np.nan),
            where=robust_within > 0,
        )
    return ratios


def _components(knn_indices: np.ndarray, codes: np.ndarray, cluster_order: list[str]) -> tuple[sparse.csr_matrix, dict[str, tuple[int, float]]]:
    rows = np.repeat(np.arange(len(codes)), knn_indices.shape[1])
    columns = knn_indices.ravel()
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(codes), len(codes)))
    graph = graph.maximum(graph.T).tocsr()
    results: dict[str, tuple[int, float]] = {}
    for code, cluster in enumerate(cluster_order):
        positions = np.flatnonzero(codes == code)
        count, component_codes = csgraph.connected_components(graph[positions][:, positions], directed=False)
        sizes = np.bincount(component_codes, minlength=count)
        results[cluster] = (int(count), float(sizes.max() / len(positions)) if len(positions) else np.nan)
    return graph, results


def _graph_neighbour_sets(graph: sparse.csr_matrix, k: int) -> list[set[int]]:
    sets: list[set[int]] = []
    for row in range(graph.shape[0]):
        start, end = graph.indptr[row], graph.indptr[row + 1]
        columns = graph.indices[start:end]
        weights = graph.data[start:end]
        if len(columns) > k:
            selected = np.argpartition(weights, -k)[-k:]
            columns = columns[selected]
        sets.append(set(int(item) for item in columns))
    return sets


def calculate_embedding_metrics(
    umap: np.ndarray,
    labels: pd.Series,
    *,
    cluster_order: list[str],
    graph: sparse.csr_matrix | None,
    pca: np.ndarray | None,
    umap_k: int,
    silhouette_max_cells: int,
    density_max_cells_per_cluster: int,
    density_grid_size: int,
    min_cluster_size: int,
    include_optional_metrics: bool,
    random_seed: int,
) -> EmbeddingMetricResult:
    """Calculate separation and representation-reliability metrics."""
    valid_positions = np.flatnonzero(labels.notna().to_numpy())
    label_values = labels.iloc[valid_positions].astype(str).to_numpy()
    coordinates = np.asarray(umap[valid_positions, :2], dtype=float)
    pca_values = np.asarray(pca[valid_positions], dtype=float) if pca is not None else None
    label_to_code = {label: index for index, label in enumerate(cluster_order)}
    codes = np.asarray([label_to_code[label] for label in label_values], dtype=int)
    n_clusters = len(cluster_order)

    local_indices, _local_distances = _knn(coordinates, umap_k)
    purity, mixing = _purity_and_mixing(local_indices, codes, n_clusters)
    isolation = _isolation_ratio(coordinates, codes, n_clusters)
    umap_silhouette, silhouette_sample, silhouette_sampled = _silhouette(
        coordinates, label_values, maximum=silhouette_max_cells, seed=random_seed
    )
    density_overlap, density_samples = _density_overlap(
        coordinates,
        codes,
        cluster_order,
        maximum_per_cluster=density_max_cells_per_cluster,
        grid_size=density_grid_size,
        min_cluster_size=min_cluster_size,
        seed=random_seed,
    )
    umap_graph, component_metrics = _components(local_indices, codes, cluster_order)

    preservation = np.full(len(codes), np.nan)
    recall = np.full(len(codes), np.nan)
    precision = np.full(len(codes), np.nan)
    if graph is not None:
        graph_sets = _graph_neighbour_sets(graph, min(umap_k, max(1, graph.shape[0] - 1)))
        for row in range(len(codes)):
            graph_set = graph_sets[row]
            umap_set = set(int(item) for item in local_indices[row])
            union = graph_set | umap_set
            intersection = graph_set & umap_set
            preservation[row] = len(intersection) / len(union) if union else np.nan
            recall[row] = len(intersection) / len(graph_set) if graph_set else np.nan
            precision[row] = len(intersection) / len(umap_set) if umap_set else np.nan

    pca_silhouette = np.full(len(codes), np.nan)
    pca_purity = np.full(len(codes), np.nan)
    pca_preservation = np.full(len(codes), np.nan)
    pca_sample = np.array([], dtype=int)
    if pca_values is not None:
        pca_silhouette, pca_sample, _ = _silhouette(
            pca_values, label_values, maximum=silhouette_max_cells, seed=random_seed + 1
        )
        if include_optional_metrics:
            pca_indices, _ = _knn(pca_values, umap_k)
            pca_purity, _ = _purity_and_mixing(pca_indices, codes, n_clusters)
            for row in range(len(codes)):
                left, right = set(local_indices[row]), set(pca_indices[row])
                pca_preservation[row] = len(left & right) / len(left | right) if left | right else np.nan

    cell = pd.DataFrame(
        {
            "cell_position": valid_positions,
            "reference_population": label_values,
            "umap_neighbour_purity": purity,
            "umap_isolation_ratio": isolation,
            "umap_silhouette": umap_silhouette,
            "pca_silhouette": pca_silhouette,
            "pca_neighbour_purity": pca_purity,
            "umap_graph_neighbourhood_preservation": preservation,
            "umap_graph_neighbourhood_recall": recall,
            "umap_graph_neighbourhood_precision": precision,
            "umap_pca_neighbourhood_preservation": pca_preservation,
        }
    ).set_index("cell_position")

    records: list[dict[str, object]] = []
    for code, cluster in enumerate(cluster_order):
        members = np.flatnonzero(codes == code)
        def values(array: np.ndarray) -> np.ndarray:
            return array[members][np.isfinite(array[members])]
        purity_values = values(purity)
        isolation_values = values(isolation)
        umap_silhouette_values = values(umap_silhouette)
        pca_silhouette_values = values(pca_silhouette)
        pca_purity_values = values(pca_purity)
        pca_preservation_values = values(pca_preservation)
        preservation_values = values(preservation)
        overlaps = density_overlap.loc[cluster].drop(labels=[cluster], errors="ignore").dropna()
        maximum_overlap = float(overlaps.max()) if len(overlaps) else np.nan
        overlap_competitor = str(overlaps.idxmax()) if len(overlaps) else None
        component_count, largest_component = component_metrics[cluster]
        records.append(
            {
                "cluster": cluster,
                "umap_purity_mean": float(np.mean(purity_values)) if purity_values.size else np.nan,
                "umap_purity_median": float(np.median(purity_values)) if purity_values.size else np.nan,
                "umap_purity_q25": float(np.quantile(purity_values, 0.25)) if purity_values.size else np.nan,
                "umap_purity_p10": float(np.quantile(purity_values, 0.10)) if purity_values.size else np.nan,
                "umap_purity_fraction_below_0_5": float(np.mean(purity_values < 0.5)) if purity_values.size else np.nan,
                "umap_purity_fraction_below_0_7": float(np.mean(purity_values < 0.7)) if purity_values.size else np.nan,
                "umap_purity_fraction_below_0_9": float(np.mean(purity_values < 0.9)) if purity_values.size else np.nan,
                "umap_neighbour_impurity": 1 - float(np.median(purity_values)) if purity_values.size else np.nan,
                "umap_silhouette_mean": float(np.mean(umap_silhouette_values)) if umap_silhouette_values.size else np.nan,
                "umap_silhouette_median": float(np.median(umap_silhouette_values)) if umap_silhouette_values.size else np.nan,
                "umap_silhouette_q25": float(np.quantile(umap_silhouette_values, 0.25)) if umap_silhouette_values.size else np.nan,
                "umap_silhouette_fraction_below_zero": float(np.mean(umap_silhouette_values < 0)) if umap_silhouette_values.size else np.nan,
                "pca_silhouette_mean": float(np.mean(pca_silhouette_values)) if pca_silhouette_values.size else np.nan,
                "pca_silhouette_median": float(np.median(pca_silhouette_values)) if pca_silhouette_values.size else np.nan,
                "pca_silhouette_q25": float(np.quantile(pca_silhouette_values, 0.25)) if pca_silhouette_values.size else np.nan,
                "pca_silhouette_fraction_below_zero": float(np.mean(pca_silhouette_values < 0)) if pca_silhouette_values.size else np.nan,
                "pca_neighbour_impurity": 1 - float(np.median(pca_purity_values)) if pca_purity_values.size else np.nan,
                "umap_isolation_ratio": float(np.median(isolation_values)) if isolation_values.size else np.nan,
                "umap_isolation_ratio_q25": float(np.quantile(isolation_values, 0.25)) if isolation_values.size else np.nan,
                "umap_isolation_fraction_below_1": float(np.mean(isolation_values < 1)) if isolation_values.size else np.nan,
                "umap_isolation_fraction_below_1_25": float(np.mean(isolation_values < 1.25)) if isolation_values.size else np.nan,
                "umap_max_density_overlap": maximum_overlap,
                "umap_density_overlap_competitor": overlap_competitor,
                "umap_mean_density_overlap": float(overlaps.mean()) if len(overlaps) else np.nan,
                "umap_component_count": component_count,
                "umap_largest_component_fraction": largest_component,
                "umap_component_loss": 1 - largest_component if np.isfinite(largest_component) else np.nan,
                "umap_graph_neighbourhood_preservation": float(np.median(preservation_values)) if preservation_values.size else np.nan,
                "umap_graph_neighbourhood_preservation_q25": float(np.quantile(preservation_values, 0.25)) if preservation_values.size else np.nan,
                "umap_pca_neighbourhood_preservation": float(np.median(pca_preservation_values)) if pca_preservation_values.size else np.nan,
            }
        )

    global_metrics: dict[str, float | int | bool] = {}
    if pca_values is not None and len(codes) >= 3:
        trust_sample = stratified_sample_indices(label_values, min(silhouette_max_cells, 5000), random_seed + 2)
        neighbour_count = min(umap_k, max(1, len(trust_sample) // 2 - 1))
        if len(trust_sample) > 2 * neighbour_count + 1:
            global_metrics["umap_trustworthiness"] = float(
                trustworthiness(pca_values[trust_sample], coordinates[trust_sample], n_neighbors=neighbour_count)
            )
            global_metrics["trustworthiness_sample_cells"] = int(len(trust_sample))
    sampling = {
        "umap_silhouette_sampled": silhouette_sampled,
        "umap_silhouette_sample_cells": int(len(silhouette_sample)),
        "pca_silhouette_sample_cells": int(len(pca_sample)),
        "density_cells_per_cluster": density_samples,
        "isolation_method": "nearest external cell divided by median distance to up to three nearest same-cluster cells",
    }
    mixing_frame = pd.DataFrame(mixing, index=cluster_order, columns=cluster_order)
    mixing_frame.index.name = "source_cluster"
    return EmbeddingMetricResult(
        cluster_metrics=pd.DataFrame(records).set_index("cluster"),
        cell_metrics=cell,
        density_overlap=density_overlap,
        umap_mixing=mixing_frame,
        umap_graph=umap_graph,
        global_metrics=global_metrics,
        sampling=sampling,
    )


__all__ = [
    "EmbeddingMetricResult",
    "calculate_embedding_metrics",
    "stratified_sample_indices",
]
