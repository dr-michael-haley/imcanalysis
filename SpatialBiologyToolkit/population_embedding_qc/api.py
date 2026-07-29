"""Public orchestration API for population embedding and clustering QC."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig

from .embedding_metrics import calculate_embedding_metrics
from .graph_metrics import calculate_graph_metrics
from .inspection import inspect_anndata
from .models import PopulationEmbeddingQCResult, load_metric_definitions
from .scoring import ranked_cluster_order, score_cluster_metrics
from .sweep_metrics import calculate_sweep_metrics


LOGGER = logging.getLogger(__name__)
DEFAULT_SWEEP_REGEX = r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$"


def _choose(value: Any, default: Any, configured: Any) -> Any:
    """Prefer a non-default explicit API argument, otherwise typed config."""
    return configured if value == default else value


def _cluster_text(
    cluster: str,
    row: pd.Series,
    raw: pd.Series,
    competitors: pd.DataFrame,
    definitions: dict[str, Any],
) -> str:
    size = int(row["cluster_size"])
    failed = [key for key in definitions if key in raw and key in row.get("failed_metric_keys", "").split(", ")]
    competitor = None
    if not competitors.empty and cluster in set(competitors["cluster"].astype(str)):
        competitor = competitors.loc[competitors["cluster"].astype(str) == cluster, "competitor"].iloc[0]
    if failed:
        names = ", ".join(definitions[key].display_name.lower() for key in failed[:4])
        sentence = f"Cluster {cluster} ({size} cells) exceeds QC concern thresholds for {names}."
    else:
        sentence = f"Cluster {cluster} ({size} cells) does not exceed the available default QC concern thresholds."
    if competitor is not None and not pd.isna(competitor):
        sentence += f" Its strongest observed competitor is cluster {competitor}."
    if pd.notna(raw.get("sweep_persistence_fraction")):
        persistence = float(raw["sweep_persistence_fraction"])
        sentence += (
            " Its identity is supported across most available precomputed resolutions."
            if persistence >= 0.75
            else " Its identity has limited support across the available precomputed resolutions."
        )
    sentence += " These results describe population support, not biological validity."
    return sentence


def run_population_embedding_qc(
    adata: Any,
    population_obs: str | None = None,
    mode: str = "auto",
    sweep_columns: list[str] | None = None,
    sweep_regex: str = DEFAULT_SWEEP_REGEX,
    reference_resolution: float | None = None,
    umap_key: str = "X_umap",
    pca_key: str = "X_pca",
    connectivities_key: str | None = None,
    output_dir: str | Path | None = None,
    config: PopulationEmbeddingQCConfig | None = None,
    overwrite: bool = False,
) -> PopulationEmbeddingQCResult:
    """Assess structural population support using only existing AnnData state.

    This function never recalculates clustering, PCA, UMAP, or the Scanpy
    neighbour graph. The input AnnData is not mutated. When ``output_dir`` is
    supplied, standalone tables, plots, and a deterministic report are written.
    """
    settings = config or PopulationEmbeddingQCConfig()
    population_obs = population_obs if population_obs is not None else settings.population_obs
    mode = _choose(mode, "auto", settings.mode)
    sweep_columns = sweep_columns if sweep_columns is not None else settings.sweep_columns
    sweep_regex = _choose(sweep_regex, DEFAULT_SWEEP_REGEX, settings.sweep_regex)
    reference_resolution = reference_resolution if reference_resolution is not None else settings.reference_resolution
    umap_key = _choose(umap_key, "X_umap", settings.umap_key)
    pca_key = _choose(pca_key, "X_pca", settings.pca_key)
    connectivities_key = connectivities_key if connectivities_key is not None else settings.connectivities_key
    settings = settings.model_copy(
        update={
            "population_obs": population_obs,
            "mode": mode,
            "sweep_columns": sweep_columns,
            "sweep_regex": sweep_regex,
            "reference_resolution": reference_resolution,
            "umap_key": umap_key,
            "pca_key": pca_key,
            "connectivities_key": connectivities_key,
        }
    )

    inspection = inspect_anndata(
        adata,
        population_obs=population_obs,
        mode=mode,
        sweep_columns=sweep_columns,
        sweep_regex=sweep_regex,
        reference_resolution=reference_resolution,
        umap_key=umap_key,
        pca_key=pca_key,
        pca_dimensions=settings.pca_dimensions,
        connectivities_key=connectivities_key,
    )
    settings = settings.model_copy(
        update={"population_obs": inspection.reference_column}
    )
    LOGGER.info(
        "Population QC reference=%s clusters=%d cells=%d excluded=%d sweep_columns=%d",
        inspection.reference_column,
        len(inspection.cluster_order),
        int(inspection.valid_mask.sum()),
        inspection.excluded_cells,
        len(inspection.sweep),
    )
    labels = inspection.labels
    valid_labels = labels.loc[labels.notna()].astype(str)
    cluster_sizes = valid_labels.value_counts().reindex(inspection.cluster_order, fill_value=0)
    cluster_metrics = pd.DataFrame(index=pd.Index(inspection.cluster_order, name="cluster"))
    cluster_metrics["cluster_size"] = cluster_sizes.astype(int)
    cluster_metrics["cluster_percentage"] = 100 * cluster_sizes / max(1, int(cluster_sizes.sum()))
    cluster_metrics["small_cluster"] = cluster_sizes < settings.min_cluster_size
    for configured_column, output_column, label in (
        (settings.sample_obs, "represented_samples", "sample"),
        (settings.roi_obs, "represented_rois", "ROI"),
    ):
        if configured_column and configured_column in adata.obs:
            counts = (
                pd.DataFrame(
                    {
                        "cluster": labels.loc[labels.notna()].astype(str),
                        "value": adata.obs.loc[labels.notna(), configured_column],
                    }
                )
                .dropna()
                .groupby("cluster", observed=True)["value"]
                .nunique()
            )
            cluster_metrics[output_column] = counts.reindex(
                inspection.cluster_order, fill_value=0
            ).astype(int)
        elif configured_column:
            inspection.warnings.append(
                f"Configured {label} column {configured_column!r} is missing from adata.obs; representation counts were skipped"
            )
    cell_metrics = pd.DataFrame(index=np.flatnonzero(inspection.valid_mask))
    cell_metrics.index.name = "cell_position"
    cell_metrics["cell_index"] = adata.obs_names[inspection.valid_mask].astype(str)
    cell_metrics["reference_population"] = valid_labels.to_numpy()

    graph_result = None
    if inspection.connectivities is not None:
        LOGGER.info("Calculating sparse existing-graph metrics")
        graph_result = calculate_graph_metrics(
            inspection.connectivities,
            labels,
            cluster_order=inspection.cluster_order,
            boundary_threshold=settings.graph_boundary_threshold,
            high_entropy_threshold=settings.high_entropy_threshold,
            min_component_size=settings.min_component_size,
        )
        cluster_metrics = cluster_metrics.join(graph_result.cluster_metrics, how="left")
        cell_metrics = cell_metrics.join(graph_result.cell_metrics.drop(columns=["reference_population"]), how="left")

    LOGGER.info("Calculating UMAP%s metrics", " and PCA" if inspection.pca is not None else "")
    embedding_result = calculate_embedding_metrics(
        inspection.umap,
        labels,
        cluster_order=inspection.cluster_order,
        graph=graph_result.graph if graph_result is not None else None,
        pca=inspection.pca,
        umap_k=settings.umap_k,
        silhouette_max_cells=settings.silhouette_max_cells,
        density_max_cells_per_cluster=settings.density_max_cells_per_cluster,
        density_grid_size=settings.density_grid_size,
        min_cluster_size=settings.min_cluster_size,
        include_optional_metrics=settings.include_optional_metrics,
        random_seed=settings.random_seed,
    )
    cluster_metrics = cluster_metrics.join(embedding_result.cluster_metrics, how="left")
    cell_metrics = cell_metrics.join(embedding_result.cell_metrics.drop(columns=["reference_population"]), how="left")

    if graph_result is not None:
        purity_source = cell_metrics["graph_neighbour_purity"]
        pairwise_graph = graph_result.pairwise_connectivity
        competitors = graph_result.competitors
    else:
        purity_source = cell_metrics["umap_neighbour_purity"]
        pairwise_graph = pd.DataFrame()
        mixing = embedding_result.umap_mixing.copy()
        competitor_records: list[dict[str, object]] = []
        for cluster in inspection.cluster_order:
            row = mixing.loc[cluster].drop(labels=[cluster], errors="ignore")
            competitor = str(row.idxmax()) if len(row) and row.max() > 0 else None
            total = float(mixing.loc[cluster].sum())
            weight = float(row.max()) if competitor is not None else 0.0
            competitor_records.append(
                {
                    "cluster": cluster,
                    "competitor": competitor,
                    "edge_weight": weight,
                    "fraction_total_cluster_connectivity": weight / total if total > 0 else np.nan,
                    "fraction_external_connectivity": weight / float(row.sum()) if row.sum() > 0 else np.nan,
                    "source": "UMAP neighbours (graph unavailable)",
                }
            )
        competitors = pd.DataFrame(competitor_records)
    cell_metrics["boundary_class"] = np.select(
        [purity_source >= settings.core_purity_threshold, purity_source >= settings.graph_boundary_threshold],
        ["core", "intermediate"],
        default="boundary",
    )

    LOGGER.info("Calculating precomputed resolution-sweep metrics")
    sweep_result = calculate_sweep_metrics(
        adata.obs,
        reference_column=inspection.reference_column,
        cluster_order=inspection.cluster_order,
        sweep=inspection.sweep,
        persistence_threshold=settings.persistence_jaccard_threshold,
    )
    cluster_metrics = cluster_metrics.join(sweep_result.reference_metrics, how="left")

    metric_definitions = load_metric_definitions(settings.metric_config_path)
    concern_scores, flags, score_summary = score_cluster_metrics(cluster_metrics, metric_definitions)
    annotation_columns = [
        column
        for column in (
            "cluster_size",
            "cluster_percentage",
            "small_cluster",
            "represented_samples",
            "represented_rois",
        )
        if column in cluster_metrics
    ]
    cluster_summary = cluster_metrics[annotation_columns].join(score_summary)
    ranked_order = ranked_cluster_order(cluster_summary)
    cluster_summary["concern_rank"] = pd.Series(
        {cluster: rank for rank, cluster in enumerate(ranked_order, start=1)}
    )
    definitions_by_key = {item.key: item for item in metric_definitions}
    per_cluster_text = {
        cluster: _cluster_text(
            cluster,
            cluster_summary.loc[cluster],
            cluster_metrics.loc[cluster],
            competitors,
            definitions_by_key,
        )
        for cluster in inspection.cluster_order
    }
    warnings = [*inspection.warnings, *embedding_result.warnings]
    if (cluster_sizes < settings.min_cluster_size).any():
        warnings.append(
            f"{int((cluster_sizes < settings.min_cluster_size).sum())} clusters contain fewer than {settings.min_cluster_size} cells"
        )
    detected_sweep = pd.DataFrame(inspection.sweep, columns=["column", "resolution"])
    sweep_missing_labels = {
        column: int(adata.obs[column].isna().sum()) for column, _ in inspection.sweep
    }
    for column, missing_count in sweep_missing_labels.items():
        if missing_count:
            warnings.append(
                f"Excluded {missing_count} missing labels from sweep calculations involving {column!r}"
            )
    metric_availability = {
        definition.key: {
            "available_clusters": int(cluster_metrics.get(definition.key, pd.Series(dtype=float)).notna().sum()),
            "total_clusters": len(inspection.cluster_order),
            "minimum_data": definition.minimum_data,
        }
        for definition in metric_definitions
    }
    run_summary: dict[str, Any] = {
        "n_cells": int(adata.n_obs),
        "n_cells_analysed": int(inspection.valid_mask.sum()),
        "excluded_cells": inspection.excluded_cells,
        "reference_column": inspection.reference_column,
        "n_reference_clusters": len(inspection.cluster_order),
        "umap_key": umap_key,
        "pca_key": pca_key if inspection.pca is not None else None,
        "pca_available": inspection.pca is not None,
        "connectivities_key": inspection.connectivities_key,
        "graph_available": inspection.connectivities is not None,
        "sweep_available": len(inspection.sweep) >= 2,
        "sweep_columns": [column for column, _ in inspection.sweep],
        "sweep_resolutions": [resolution for _, resolution in inspection.sweep],
        "sweep_missing_labels": sweep_missing_labels,
        "mode": mode,
        "sampling": embedding_result.sampling,
        "global_embedding_metrics": embedding_result.global_metrics,
        "warnings": warnings,
        "configuration": settings.model_dump(mode="json"),
        "normalization_note": "Concern scores use configurable clipped linear anchors; threshold flags always use raw metric values.",
        "graph_symmetrization": "elementwise maximum of connectivity matrix and transpose; self loops removed",
        "metric_availability": metric_availability,
    }
    result = PopulationEmbeddingQCResult(
        reference_column=inspection.reference_column,
        cluster_order=inspection.cluster_order,
        cluster_metrics_raw=cluster_metrics,
        concern_scores=concern_scores,
        threshold_flags=flags,
        cluster_summary=cluster_summary,
        cluster_competitors=competitors,
        pairwise_graph_connectivity=pairwise_graph,
        pairwise_umap_neighbour_mixing=embedding_result.umap_mixing,
        pairwise_umap_density_overlap=embedding_result.density_overlap,
        cell_metrics=cell_metrics,
        metric_definitions=metric_definitions,
        run_summary=run_summary,
        warnings=warnings,
        detected_sweep_columns=detected_sweep,
        sweep_transition_edges=sweep_result.transition_edges,
        sweep_best_matches=sweep_result.best_matches,
        sweep_reference_cluster_metrics=sweep_result.reference_metrics,
        sweep_reference_membership=sweep_result.reference_membership,
        sweep_global_metrics=sweep_result.global_metrics,
        sweep_pairwise_jaccard=sweep_result.jaccard_matrices,
        per_cluster_text=per_cluster_text,
    )
    if output_dir is not None:
        from .outputs import write_standalone_outputs

        write_standalone_outputs(
            result,
            adata=adata,
            umap=inspection.umap,
            config=settings,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    return result


__all__ = ["DEFAULT_SWEEP_REGEX", "run_population_embedding_qc"]
