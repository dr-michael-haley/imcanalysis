"""Typed result and metric models for population embedding QC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd
import yaml  # type: ignore[import-untyped]


MetricGroup = Literal[
    "graph_separation",
    "embedding_separation",
    "embedding_reliability",
    "resolution_stability",
]


@dataclass(frozen=True)
class MetricDefinition:
    """One raw metric's interpretation and concern-score transformation."""

    key: str
    display_name: str
    group: MetricGroup
    description: str
    units: str
    higher_is_better: bool
    good_anchor: float
    bad_anchor: float
    concern_threshold: float
    threshold_operator: Literal[">", ">=", "<", "<="]
    default_heatmap: bool = True
    weight: float = 1.0
    minimum_data: str = "reference clustering"


def _metric(
    key: str,
    name: str,
    group: MetricGroup,
    description: str,
    anchors: tuple[float, float, float],
    operator: Literal[">", ">=", "<", "<="],
    *,
    units: str = "fraction",
    higher_is_better: bool = False,
    heatmap: bool = True,
    weight: float = 1.0,
    minimum_data: str = "reference clustering",
) -> MetricDefinition:
    return MetricDefinition(
        key=key,
        display_name=name,
        group=group,
        description=description,
        units=units,
        higher_is_better=higher_is_better,
        good_anchor=anchors[0],
        bad_anchor=anchors[1],
        concern_threshold=anchors[2],
        threshold_operator=operator,
        default_heatmap=heatmap,
        weight=weight,
        minimum_data=minimum_data,
    )


DEFAULT_METRICS: tuple[MetricDefinition, ...] = (
    _metric("graph_neighbour_impurity", "Graph neighbour impurity", "graph_separation", "One minus median weighted same-label graph-neighbour purity.", (0.05, 0.40, 0.20), ">", minimum_data="connectivity graph"),
    _metric("graph_boundary_fraction", "Graph boundary fraction", "graph_separation", "Fraction of cells below the configured graph-purity boundary threshold.", (0.05, 0.60, 0.25), ">", minimum_data="connectivity graph"),
    _metric("graph_conductance", "Graph conductance", "graph_separation", "External edge weight divided by the smaller cluster/outside weighted volume.", (0.05, 0.50, 0.25), ">", minimum_data="connectivity graph"),
    _metric("graph_label_entropy", "Graph label entropy", "graph_separation", "Median normalized entropy of graph-neighbour labels.", (0.05, 0.60, 0.30), ">", minimum_data="connectivity graph"),
    _metric("strongest_competitor_edge_fraction", "Strongest competitor edge fraction", "graph_separation", "Fraction of cluster weighted degree connected to its strongest external competitor.", (0.05, 0.40, 0.20), ">", minimum_data="connectivity graph"),
    _metric("graph_component_loss", "Graph component loss", "graph_separation", "Fraction outside the largest within-cluster graph component.", (0.00, 0.30, 0.05), ">", minimum_data="connectivity graph"),
    _metric("umap_neighbour_impurity", "UMAP neighbour impurity", "embedding_separation", "One minus median same-label UMAP-neighbour purity.", (0.05, 0.40, 0.20), ">", minimum_data="UMAP"),
    _metric("umap_silhouette_median", "UMAP silhouette median", "embedding_separation", "Median cell silhouette in the stored two-dimensional UMAP.", (0.50, -0.10, 0.10), "<", units="silhouette", higher_is_better=True, minimum_data="UMAP and at least two clusters"),
    _metric("pca_silhouette_median", "PCA silhouette median", "embedding_separation", "Median cell silhouette in the existing PCA representation.", (0.40, -0.10, 0.05), "<", units="silhouette", higher_is_better=True, minimum_data="PCA and at least two clusters"),
    _metric("umap_isolation_ratio", "UMAP isolation ratio", "embedding_separation", "Nearest external-label distance divided by robust local within-label distance.", (2.00, 1.00, 1.25), "<", units="ratio", higher_is_better=True, minimum_data="UMAP"),
    _metric("umap_max_density_overlap", "UMAP maximum density overlap", "embedding_separation", "Maximum pairwise overlap between normalized UMAP density estimates.", (0.02, 0.40, 0.20), ">", minimum_data="UMAP and sufficiently large clusters"),
    _metric("umap_graph_neighbourhood_preservation", "UMAP-graph neighbourhood preservation", "embedding_reliability", "Median Jaccard overlap between UMAP and existing graph neighbourhoods.", (0.60, 0.15, 0.30), "<", higher_is_better=True, minimum_data="UMAP and connectivity graph"),
    _metric("sweep_adjacent_jaccard", "Sweep adjacent Jaccard", "resolution_stability", "Mean best matching Jaccard support across precomputed resolutions.", (0.90, 0.50, 0.75), "<", higher_is_better=True, minimum_data="at least two sweep columns"),
    _metric("sweep_retention", "Sweep retention", "resolution_stability", "Conservative fraction of reference cells retained by their best sweep match.", (0.90, 0.50, 0.75), "<", higher_is_better=True, minimum_data="at least two sweep columns"),
    _metric("sweep_persistence_fraction", "Sweep persistence fraction", "resolution_stability", "Fraction of resolutions meeting the configured Jaccard support threshold.", (1.00, 0.40, 0.75), "<", higher_is_better=True, minimum_data="at least two sweep columns"),
    _metric("sweep_split_entropy", "Sweep split entropy", "resolution_stability", "Mean normalized entropy of reference cells across sweep labels.", (0.00, 0.80, 0.40), ">", minimum_data="at least two sweep columns"),
    _metric("sweep_merge_entropy", "Sweep merge entropy", "resolution_stability", "Mean normalized reference-label entropy in each best-matching sweep cluster.", (0.00, 0.80, 0.40), ">", minimum_data="at least two sweep columns"),
    _metric("sweep_within_cluster_consensus", "Sweep within-cluster consensus", "resolution_stability", "Probability that two reference-cluster cells remain co-assigned across resolutions.", (0.95, 0.50, 0.80), "<", higher_is_better=True, minimum_data="at least two sweep columns"),
    _metric("sweep_max_external_coassignment", "Sweep maximum external co-assignment", "resolution_stability", "Maximum mean co-assignment with another reference cluster.", (0.05, 0.60, 0.25), ">", minimum_data="at least two sweep columns"),
)


def load_metric_definitions(path: str | Path | None = None) -> tuple[MetricDefinition, ...]:
    """Load optional YAML/JSON overrides onto the single metric registry."""
    definitions = {definition.key: definition for definition in DEFAULT_METRICS}
    if path is None:
        return tuple(definitions.values())
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Metric configuration file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    raw = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if raw is None:
        return tuple(definitions.values())
    raw_metrics = raw.get("metrics", raw) if isinstance(raw, Mapping) else raw
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("Metric configuration must be a mapping or contain a 'metrics' mapping")
    for key, override in raw_metrics.items():
        if key not in definitions:
            raise ValueError(f"Unknown metric configuration key: {key}")
        if not isinstance(override, Mapping):
            raise ValueError(f"Metric override for {key} must be a mapping")
        allowed = {item.name for item in __import__("dataclasses").fields(MetricDefinition)} - {"key"}
        unknown = set(override) - allowed
        if unknown:
            raise ValueError(f"Unknown fields for metric {key}: {', '.join(sorted(unknown))}")
        definitions[key] = replace(definitions[key], **dict(override))
    return tuple(definitions.values())


@dataclass
class PopulationEmbeddingQCResult:
    """Structured, reusable output of population embedding QC."""

    reference_column: str
    cluster_order: list[str]
    cluster_metrics_raw: pd.DataFrame
    concern_scores: pd.DataFrame
    threshold_flags: pd.DataFrame
    cluster_summary: pd.DataFrame
    cluster_competitors: pd.DataFrame
    pairwise_graph_connectivity: pd.DataFrame
    pairwise_umap_neighbour_mixing: pd.DataFrame
    pairwise_umap_density_overlap: pd.DataFrame
    cell_metrics: pd.DataFrame
    metric_definitions: tuple[MetricDefinition, ...]
    run_summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    detected_sweep_columns: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep_transition_edges: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep_best_matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep_reference_cluster_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep_reference_membership: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep_global_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep_pairwise_jaccard: dict[str, pd.DataFrame] = field(default_factory=dict)
    per_cluster_text: dict[str, str] = field(default_factory=dict)
    output_files: list[Path] = field(default_factory=list)

    def metric_definition_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(item) for item in self.metric_definitions])


__all__ = [
    "DEFAULT_METRICS",
    "MetricDefinition",
    "PopulationEmbeddingQCResult",
    "load_metric_definitions",
]
