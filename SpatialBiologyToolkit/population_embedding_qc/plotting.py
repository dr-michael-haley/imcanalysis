"""Publication-oriented plots that consume calculated QC result tables."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from .models import PopulationEmbeddingQCResult
from .scoring import GROUP_OUTPUT_NAMES


def _save(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _metric_keys(result: PopulationEmbeddingQCResult, *, all_metrics: bool) -> list[str]:
    return [
        definition.key
        for definition in result.metric_definitions
        if definition.key in result.concern_scores
        and (all_metrics or definition.default_heatmap)
    ]


def plot_concern_heatmap(
    result: PopulationEmbeddingQCResult,
    output_base: Path,
    *,
    detailed: bool,
) -> list[Path]:
    keys = _metric_keys(result, all_metrics=detailed)
    order = result.cluster_summary.sort_values("concern_rank").index.tolist()
    values = result.concern_scores.loc[order, keys]
    mask = values.isna()
    height = max(4.5, 0.38 * len(order) + 2)
    width = max(9, 0.55 * len(keys) + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_facecolor("#d0d0d0")
    annotations = None
    if detailed:
        annotations = result.cluster_metrics_raw.reindex(index=order, columns=keys).map(
            lambda value: "NA" if pd.isna(value) else f"{float(value):.2g}"
        )
    sns.heatmap(
        values,
        mask=mask,
        cmap="magma_r",
        vmin=0,
        vmax=1,
        annot=annotations,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "QC concern score (0 low, 1 high)"},
        ax=ax,
    )
    display = {definition.key: definition.display_name for definition in result.metric_definitions}
    ax.set_xticklabels([display[key] for key in keys], rotation=55, ha="right")
    sizes = result.cluster_summary.loc[order, "cluster_size"]
    small = result.cluster_summary.loc[order, "small_cluster"]
    represented_samples = result.cluster_summary.get("represented_samples", pd.Series(index=order, dtype=float)).reindex(order)
    represented_rois = result.cluster_summary.get("represented_rois", pd.Series(index=order, dtype=float)).reindex(order)
    ax.set_yticklabels(
        [
            f"{cluster}  n={int(size)}"
            f"{' samples=' + str(int(sample_count)) if pd.notna(sample_count) else ''}"
            f"{' ROIs=' + str(int(roi_count)) if pd.notna(roi_count) else ''}"
            f"{' *' if bool(is_small) else ''}"
            for cluster, size, is_small, sample_count, roi_count in zip(
                order, sizes, small, represented_samples, represented_rois
            )
        ],
        rotation=0,
    )
    flags = result.threshold_flags.loc[order, keys]
    for row_index, cluster in enumerate(order):
        for column_index, key in enumerate(keys):
            flag = flags.at[cluster, key]
            if not pd.isna(flag) and bool(flag):
                ax.text(column_index + 0.5, row_index + 0.5, "X", ha="center", va="center", color="cyan", fontsize=11, fontweight="bold")
    groups = [next(item.group for item in result.metric_definitions if item.key == key) for key in keys]
    for index in range(1, len(groups)):
        if groups[index] != groups[index - 1]:
            ax.axvline(index, color="black", linewidth=2)
    ax.set_xlabel("Structural QC metric")
    ax.set_ylabel(f"{result.reference_column} cluster")
    ax.set_title("Population embedding QC concern heatmap" + (" (raw values shown)" if detailed else ""))
    ax.legend(
        handles=[
            Line2D([0], [0], marker="x", color="cyan", linestyle="", markeredgewidth=2, label="Raw threshold exceeded"),
            Line2D([0], [0], marker="s", color="#d0d0d0", linestyle="", label="Metric unavailable"),
            Line2D([0], [0], marker="*", color="black", linestyle="", label="Small cluster"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.16, 1),
    )
    return _save(fig, output_base)


def plot_cluster_overview(result: PopulationEmbeddingQCResult, output_base: Path) -> list[Path]:
    summary = result.cluster_summary.sort_values("concern_rank")
    columns = [*GROUP_OUTPUT_NAMES.values(), "overall_concern"]
    labels = ["Graph", "Embedding", "Reliability", "Resolution", "Overall"]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.36 * len(summary) + 1.5)))
    y = np.arange(len(summary))
    for offset, (column, label) in enumerate(zip(columns, labels)):
        values = summary[column]
        sizes = 25 + 90 * np.sqrt(summary["cluster_size"] / max(1, summary["cluster_size"].max()))
        ax.scatter(values, y + (offset - 2) * 0.08, s=sizes, alpha=0.75, label=label)
    failed = summary["failed_thresholds"]
    ax.scatter(np.minimum(1.03, summary["overall_concern"].fillna(0) + 0.035), y, marker="x", s=25 + 12 * failed, color="black", label="Threshold failures")
    ax.set_yticks(y, summary.index)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.08)
    ax.set_xlabel("QC concern score")
    ax.set_ylabel(result.reference_column)
    ax.set_title("Cluster concern overview (point size reflects cluster size)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_base)


def plot_umap_panels(
    result: PopulationEmbeddingQCResult,
    umap: np.ndarray,
    output_base: Path,
) -> list[Path]:
    cell = result.cell_metrics
    coordinates = umap[cell.index.to_numpy(dtype=int), :2]
    candidates = [
        ("reference_population", "Reference population", None),
        ("graph_neighbour_purity", "Graph neighbour impurity", "impurity"),
        ("umap_neighbour_purity", "UMAP neighbour impurity", "impurity"),
        ("boundary_class", "Core / intermediate / boundary", None),
        ("umap_silhouette", "UMAP silhouette", "value"),
        ("umap_graph_neighbourhood_preservation", "UMAP-graph preservation", "value"),
    ]
    candidates = [item for item in candidates if item[0] in cell and cell[item[0]].notna().any()]
    columns = 3
    rows = int(np.ceil(len(candidates) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.3 * rows), squeeze=False, sharex=True, sharey=True)
    point_size = max(1.0, min(12.0, 15000 / max(1, len(cell))))
    for ax, (key, title, mode) in zip(axes.ravel(), candidates):
        values = cell[key]
        if key == "reference_population" or key == "boundary_class":
            categories = list(dict.fromkeys(values.astype(str)))
            palette = sns.color_palette("tab20", n_colors=max(3, len(categories)))
            for colour, category in zip(palette, categories):
                mask = values.astype(str).to_numpy() == category
                ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=point_size, color=colour, label=category, rasterized=True)
            if len(categories) <= 15:
                ax.legend(fontsize=6, markerscale=2, loc="best")
        else:
            plotted = 1 - pd.to_numeric(values, errors="coerce") if mode == "impurity" else pd.to_numeric(values, errors="coerce")
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=plotted, s=point_size, cmap="viridis", rasterized=True)
            fig.colorbar(scatter, ax=ax, fraction=0.045)
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    for ax in axes.ravel()[len(candidates) :]:
        ax.set_visible(False)
    fig.suptitle("UMAP structural QC panel", y=1.01)
    return _save(fig, output_base)


def plot_boundary_umap(result: PopulationEmbeddingQCResult, umap: np.ndarray, output_base: Path) -> list[Path]:
    cell = result.cell_metrics
    coordinates = umap[cell.index.to_numpy(dtype=int), :2]
    colours = {"core": "#2ca25f", "intermediate": "#fec44f", "boundary": "#de2d26"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for category in ("core", "intermediate", "boundary"):
        mask = cell["boundary_class"].astype(str).to_numpy() == category
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=max(1, min(12, 15000 / len(cell))), color=colours[category], label=category, rasterized=True)
    source = "graph purity" if "graph_neighbour_purity" in cell else "UMAP-neighbour purity"
    ax.set_title(f"Core and boundary cells ({source})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    return _save(fig, output_base)


def plot_high_concern_umap(result: PopulationEmbeddingQCResult, umap: np.ndarray, output_base: Path) -> list[Path]:
    cell = result.cell_metrics
    coordinates = umap[cell.index.to_numpy(dtype=int), :2]
    high_clusters = set(
        result.cluster_summary.index[
            result.cluster_summary["failed_thresholds"] >= 2
        ].astype(str)
    )
    highlighted = cell["reference_population"].astype(str).isin(high_clusters).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coordinates[~highlighted, 0], coordinates[~highlighted, 1], s=max(1, min(10, 12000 / len(cell))), color="#cccccc", alpha=0.5, rasterized=True)
    ax.scatter(coordinates[highlighted, 0], coordinates[highlighted, 1], s=max(2, min(14, 16000 / len(cell))), color="#cb181d", label="Cluster with >=2 threshold failures", rasterized=True)
    ax.set_title("UMAP cells in multi-concern clusters")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if high_clusters:
        ax.legend()
    return _save(fig, output_base)


def plot_matrix(frame: pd.DataFrame, output_base: Path, *, title: str, cmap: str = "mako", diagonal_mask: bool = False) -> list[Path]:
    if frame.empty:
        return []
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    mask = numeric.isna().to_numpy()
    if diagonal_mask and numeric.shape[0] == numeric.shape[1]:
        mask |= np.eye(numeric.shape[0], dtype=bool)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * numeric.shape[1] + 3), max(5, 0.4 * numeric.shape[0] + 2)))
    sns.heatmap(numeric, mask=mask, cmap=cmap, ax=ax, square=numeric.shape[0] == numeric.shape[1])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=55)
    return _save(fig, output_base)


def plot_silhouettes(result: PopulationEmbeddingQCResult, output_base: Path, key: str, title: str) -> list[Path]:
    if key not in result.cell_metrics or not result.cell_metrics[key].notna().any():
        return []
    data = result.cell_metrics[["reference_population", key]].dropna()
    order = data.groupby("reference_population", observed=True)[key].median().sort_values().index
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(order) + 3), 5.5))
    sns.boxplot(data=data, x="reference_population", y=key, order=order, ax=ax, color="#74a9cf", fliersize=1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=55)
    ax.set_title(title)
    ax.set_xlabel(result.reference_column)
    ax.set_ylabel("Silhouette")
    return _save(fig, output_base)


def plot_components(result: PopulationEmbeddingQCResult, output_base: Path) -> list[Path]:
    columns = [column for column in ("graph_largest_component_fraction", "umap_largest_component_fraction") if column in result.cluster_metrics_raw]
    if not columns:
        return []
    data = result.cluster_metrics_raw[columns].reset_index().melt("cluster", var_name="source", value_name="largest_component_fraction")
    data["source"] = data["source"].str.replace("_largest_component_fraction", "", regex=False)
    fig, axes = plt.subplots(2, 1, figsize=(max(7, 0.4 * len(result.cluster_order) + 3), 8), sharex=True)
    sns.pointplot(data=data, x="cluster", y="largest_component_fraction", hue="source", ax=axes[0])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Fraction in largest within-cluster component")
    count_columns = [column for column in ("graph_substantial_component_count", "umap_component_count") if column in result.cluster_metrics_raw]
    if count_columns:
        counts = result.cluster_metrics_raw[count_columns].reset_index().melt("cluster", var_name="source", value_name="component_count")
        counts["source"] = counts["source"].str.replace("_substantial_component_count", "", regex=False).str.replace("_component_count", "", regex=False)
        sns.pointplot(data=counts, x="cluster", y="component_count", hue="source", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_title("Substantial graph / UMAP component count")
    return _save(fig, output_base)


def plot_reference_sweep_jaccard(result: PopulationEmbeddingQCResult, output_base: Path) -> list[Path]:
    matches = result.sweep_best_matches
    if matches.empty or "direction" not in matches:
        return []
    matches = matches.loc[matches["direction"] == "reference_to_sweep"]
    if matches.empty:
        return []
    matrix = matches.pivot(index="source_cluster", columns="target_resolution", values="jaccard")
    matrix = matrix.reindex(result.cluster_order)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * matrix.shape[1] + 3), max(4, 0.4 * matrix.shape[0] + 2)))
    sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if pd.notna(value) and value < 0.75:
                ax.text(column + 0.5, row + 0.5, "X", color="red", ha="center", va="center", fontweight="bold")
    ax.set_title("Reference-to-sweep best-match Jaccard")
    ax.set_xlabel("Precomputed resolution")
    ax.set_ylabel(result.reference_column)
    return _save(fig, output_base)


def plot_sweep_transitions(result: PopulationEmbeddingQCResult, output_base: Path, minimum_fraction: float) -> list[Path]:
    edges = result.sweep_transition_edges
    if edges.empty:
        return []
    resolutions = sorted(set(edges["source_resolution"]) | set(edges["target_resolution"]))
    nodes: dict[tuple[float, str], float] = {}
    for resolution in resolutions:
        labels = sorted(
            set(edges.loc[edges["source_resolution"] == resolution, "source_cluster"].astype(str))
            | set(edges.loc[edges["target_resolution"] == resolution, "target_cluster"].astype(str))
        )
        for index, label in enumerate(labels):
            nodes[(float(resolution), label)] = float(index)
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(resolutions)), 7))
    maximum_shared = max(1, int(edges["shared_cells"].max()))
    for row in edges.itertuples():
        if float(row.source_fraction) < minimum_fraction:
            continue
        edge_x = [resolutions.index(row.source_resolution), resolutions.index(row.target_resolution)]
        edge_y = [nodes[(float(row.source_resolution), str(row.source_cluster))], nodes[(float(row.target_resolution), str(row.target_cluster))]]
        ax.plot(edge_x, edge_y, color="#3182bd", alpha=0.25, linewidth=0.4 + 5 * float(row.shared_cells) / maximum_shared)
    for x_position, resolution in enumerate(resolutions):
        for (node_resolution, label), y_position in nodes.items():
            if node_resolution == float(resolution):
                ax.scatter(x_position, y_position, s=35, color="#08519c", zorder=3)
                ax.text(x_position + 0.03, y_position, label, fontsize=7, va="center")
    ax.set_xticks(range(len(resolutions)), [f"r={value:g}" for value in resolutions])
    ax.set_title("Precomputed Leiden sweep transitions")
    ax.set_yticks([])
    ax.set_xlabel("Numerically ordered resolution")
    return _save(fig, output_base)


def plot_sweep_stability(result: PopulationEmbeddingQCResult, output_base: Path) -> list[Path]:
    frame = result.sweep_reference_cluster_metrics
    keys = [key for key in ("sweep_adjacent_jaccard", "sweep_retention", "sweep_split_entropy", "sweep_merge_entropy", "sweep_persistence_fraction") if key in frame]
    if frame.empty or not keys:
        return []
    data = frame[keys].reset_index().melt("cluster", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(frame) + 4), 5.5))
    sns.pointplot(data=data, x="cluster", y="value", hue="metric", ax=ax)
    ax.axhline(0.75, color="black", linestyle="--", linewidth=0.8, label="Persistence/Jaccard starting threshold")
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="x", rotation=55)
    ax.set_title("Reference-cluster resolution stability")
    return _save(fig, output_base)


def plot_global_sweep(result: PopulationEmbeddingQCResult, output_base: Path) -> list[Path]:
    frame = result.sweep_global_metrics
    if frame.empty:
        return []
    x = np.arange(len(frame))
    labels = [f"{left:g}->{right:g}" for left, right in zip(frame["source_resolution"], frame["target_resolution"])]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.1 * len(frame) + 3), 7), sharex=True)
    for key, label in (("adjusted_rand_index", "ARI"), ("normalized_mutual_information", "NMI"), ("mean_best_jaccard", "Mean best Jaccard"), ("median_best_jaccard", "Median best Jaccard")):
        axes[0].plot(x, frame[key], marker="o", label=label)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(ncol=2)
    axes[0].set_ylabel("Agreement")
    axes[1].plot(x, frame["source_clusters"], marker="o", label="Source clusters")
    axes[1].plot(x, frame["target_clusters"], marker="o", label="Target clusters")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_ylabel("Clusters")
    axes[1].legend()
    fig.suptitle("Global precomputed sweep diagnostics")
    return _save(fig, output_base)


def create_all_plots(
    result: PopulationEmbeddingQCResult,
    umap: np.ndarray,
    figures_dir: Path,
    *,
    transition_min_fraction: float,
) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[Path] = []
    paths += plot_concern_heatmap(result, figures_dir / "Summary" / "population_qc_concern_heatmap", detailed=False)
    paths += plot_concern_heatmap(result, figures_dir / "Summary" / "population_qc_concern_heatmap_detailed", detailed=True)
    paths += plot_cluster_overview(result, figures_dir / "Summary" / "cluster_concern_overview")
    paths += plot_umap_panels(result, umap, figures_dir / "UMAP" / "umap_qc_panel")
    paths += plot_boundary_umap(result, umap, figures_dir / "UMAP" / "umap_core_boundary")
    paths += plot_high_concern_umap(result, umap, figures_dir / "UMAP" / "umap_multi_concern_clusters")
    competitor_matrix = result.pairwise_graph_connectivity if not result.pairwise_graph_connectivity.empty else result.pairwise_umap_neighbour_mixing
    competitor_title = "Directed cross-cluster graph connectivity" if not result.pairwise_graph_connectivity.empty else "UMAP-neighbour mixing (graph unavailable)"
    paths += plot_matrix(competitor_matrix, figures_dir / "Graph" / "cluster_competitor_matrix", title=competitor_title, diagonal_mask=True)
    paths += plot_silhouettes(result, figures_dir / "Separation" / "umap_silhouette_by_cluster", "umap_silhouette", "UMAP silhouette by cluster (sampled where recorded)")
    paths += plot_silhouettes(result, figures_dir / "Separation" / "pca_silhouette_by_cluster", "pca_silhouette", "PCA silhouette by cluster (sampled where recorded)")
    paths += plot_matrix(result.pairwise_umap_density_overlap, figures_dir / "Separation" / "umap_density_overlap_matrix", title="UMAP density overlap", diagonal_mask=True)
    paths += plot_components(result, figures_dir / "Separation" / "component_support")
    paths += plot_sweep_transitions(result, figures_dir / "Sweep" / "leiden_sweep_transitions", transition_min_fraction)
    for key, matrix in result.sweep_pairwise_jaccard.items():
        paths += plot_matrix(matrix, figures_dir / "Sweep" / "Jaccard" / f"sweep_jaccard_{key}", title=f"Sweep Jaccard {key}")
    paths += plot_sweep_stability(result, figures_dir / "Sweep" / "sweep_stability_summary")
    paths += plot_reference_sweep_jaccard(result, figures_dir / "Sweep" / "reference_sweep_jaccard_summary")
    paths += plot_global_sweep(result, figures_dir / "Sweep" / "sweep_global_summary")
    return paths


__all__ = ["create_all_plots", "plot_concern_heatmap"]
