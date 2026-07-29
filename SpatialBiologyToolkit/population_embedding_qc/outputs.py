"""Table, figure, report, provenance, and optional AnnData writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig

from .models import PopulationEmbeddingQCResult
from .plotting import create_all_plots


@dataclass(frozen=True)
class OutputLayout:
    figures: Path
    tables: Path
    summaries: Path
    files: Path

    def create(self) -> None:
        for path in (self.figures, self.tables, self.summaries, self.files):
            path.mkdir(parents=True, exist_ok=True)


class _JSONEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if np.isnan(value) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        return super().default(value)


def standalone_layout(output_dir: str | Path) -> OutputLayout:
    root = Path(output_dir).expanduser().resolve(strict=False)
    return OutputLayout(
        figures=root / "Figures",
        tables=root / "Tables",
        summaries=root / "Report",
        files=root / "Run",
    )


def render_analysis_report(result: PopulationEmbeddingQCResult) -> str:
    summary = result.cluster_summary
    failures = summary["failed_thresholds"]
    no_failures = int((failures == 0).sum())
    with_failures = int((failures > 0).sum())
    group_failure_counts: list[int] = []
    for cluster in summary.index:
        failed_keys = str(summary.at[cluster, "failed_metric_keys"]).split(", ") if summary.at[cluster, "failed_metric_keys"] else []
        groups = {
            definition.group
            for definition in result.metric_definitions
            if definition.key in failed_keys
        }
        group_failure_counts.append(len(groups))
    high = summary.sort_values("concern_rank").head(5).index.astype(str).tolist()
    flag_totals = result.threshold_flags.fillna(False).sum().sort_values(ascending=False)
    most_common_key = str(flag_totals.index[0]) if len(flag_totals) and int(flag_totals.iloc[0]) else None
    definitions = {item.key: item for item in result.metric_definitions}
    lines = [
        "# Population embedding and clustering QC report",
        "",
        "This deterministic report assesses population support, structural separation, embedding separation, and resolution stability. It does not establish biological validity.",
        "",
        "## Input summary",
        "",
        f"- Cells in input: {result.run_summary['n_cells']}",
        f"- Cells analysed: {result.run_summary['n_cells_analysed']}",
        f"- Cells excluded for missing reference labels: {result.run_summary['excluded_cells']}",
        f"- Reference column: `{result.reference_column}`",
        f"- Reference clusters: {result.run_summary['n_reference_clusters']}",
        f"- UMAP key: `{result.run_summary['umap_key']}`",
        f"- PCA available: {result.run_summary['pca_available']}",
        f"- Existing graph available: {result.run_summary['graph_available']}",
        f"- Detected sweep columns: {', '.join(result.run_summary['sweep_columns']) or 'none'}",
        f"- Sampling: `{json.dumps(result.run_summary['sampling'], cls=_JSONEncoder, sort_keys=True)}`",
        "",
        "## Overall interpretation",
        "",
        f"{no_failures} clusters have no threshold failures among available metrics; {with_failures} have one or more QC concerns, and {sum(value > 1 for value in group_failure_counts)} have concerns spanning multiple metric groups.",
        f"The highest-priority clusters by deterministic concern ranking are: {', '.join(high) or 'none'}.",
    ]
    if most_common_key:
        lines.append(
            f"The most frequently exceeded starting threshold is {definitions[most_common_key].display_name.lower()} ({int(flag_totals.iloc[0])} clusters)."
        )
    else:
        lines.append("No available raw metric exceeded its configured starting threshold.")
    if result.run_summary["graph_available"]:
        graph = summary["graph_separation_concern"]
        embedding = summary["embedding_separation_concern"]
        aligned = graph.notna() & embedding.notna()
        if aligned.sum() >= 2 and graph.loc[aligned].nunique() > 1 and embedding.loc[aligned].nunique() > 1:
            correlation = graph.loc[aligned].corr(embedding.loc[aligned], method="spearman")
            lines.append(
                "Graph and embedding concern rankings are broadly concordant."
                if pd.notna(correlation) and correlation >= 0.5
                else "Graph and UMAP concern rankings differ for some clusters; inspect both evidence sources."
            )
    lines.append(
        "Resolution stability was calculated from precomputed sweep columns."
        if result.run_summary["sweep_available"]
        else "Resolution stability was unavailable because fewer than two valid precomputed sweep columns were present."
    )
    lines.extend(["", "## Per-cluster summaries", ""])
    for cluster in summary.sort_values("concern_rank").index:
        lines.extend([f"### Cluster {cluster}", "", result.per_cluster_text[str(cluster)], ""])
    lines.extend(
        [
            "## Warnings and skipped evidence",
            "",
            *(f"- {warning}" for warning in result.warnings),
            "",
            "## Limitations",
            "",
            "- UMAP separation is not proof of biological distinctness.",
            "- Leiden communities reflect the existing graph; Leiden, PCA, UMAP, and neighbours were not recalculated.",
            "- This analysis does not assess marker plausibility or differential expression.",
            "- This analysis does not detect every technical or sample-specific artefact.",
            "- Thresholds and anchors are configurable QC starting points, not universal biological cutoffs.",
            "- Small populations produce less reliable estimates and remain explicitly flagged.",
            "- Missing PCA or graph data reduces the available evidence and is not interpreted as zero concern.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_frame(frame: pd.DataFrame, path: Path, *, index: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=index, na_rep="NA")
    return path


def write_result_outputs(
    result: PopulationEmbeddingQCResult,
    *,
    umap: np.ndarray,
    config: PopulationEmbeddingQCConfig,
    layout: OutputLayout,
) -> list[Path]:
    """Write all human-facing files into an explicit category layout."""
    layout.create()
    paths: list[Path] = []
    paths.append(_write_frame(result.cluster_metrics_raw, layout.tables / "cluster_metrics_raw.tsv"))
    paths.append(_write_frame(result.concern_scores, layout.tables / "cluster_metrics_concern_scores.tsv"))
    paths.append(_write_frame(result.threshold_flags, layout.tables / "cluster_metric_threshold_flags.tsv"))
    paths.append(_write_frame(result.cluster_summary, layout.tables / "cluster_summary.tsv"))
    paths.append(_write_frame(result.cluster_competitors, layout.tables / "cluster_competitors.tsv", index=False))
    paths.append(_write_frame(result.pairwise_graph_connectivity, layout.tables / "pairwise_graph_connectivity.tsv"))
    paths.append(_write_frame(result.pairwise_umap_neighbour_mixing, layout.tables / "pairwise_umap_neighbour_mixing.tsv"))
    paths.append(_write_frame(result.pairwise_umap_density_overlap, layout.tables / "pairwise_umap_density_overlap.tsv"))
    paths.append(_write_frame(result.metric_definition_frame(), layout.tables / "metric_definitions.tsv", index=False))
    metric_config = layout.files / "metric_configuration_used.yaml"
    metric_config.write_text(
        yaml.safe_dump({"metrics": {item.key: asdict(item) for item in result.metric_definitions}}, sort_keys=False),
        encoding="utf-8",
    )
    paths.append(metric_config)
    summary_path = layout.files / "run_summary.json"
    summary_path.write_text(json.dumps(result.run_summary, indent=2, cls=_JSONEncoder), encoding="utf-8")
    paths.append(summary_path)
    report_path = layout.summaries / "analysis_report.md"
    report_path.write_text(render_analysis_report(result), encoding="utf-8")
    paths.append(report_path)

    if not result.detected_sweep_columns.empty:
        paths.append(_write_frame(result.detected_sweep_columns, layout.tables / "detected_sweep_columns.tsv", index=False))
        paths.append(_write_frame(result.sweep_transition_edges, layout.tables / "sweep_transition_edges.tsv", index=False))
        paths.append(_write_frame(result.sweep_best_matches, layout.tables / "sweep_best_matches.tsv", index=False))
        paths.append(_write_frame(result.sweep_reference_cluster_metrics, layout.tables / "sweep_reference_cluster_metrics.tsv"))
        paths.append(_write_frame(result.sweep_reference_membership, layout.tables / "sweep_reference_membership.tsv", index=False))
        paths.append(_write_frame(result.sweep_global_metrics, layout.tables / "sweep_global_metrics.tsv", index=False))
        jaccard_dir = layout.tables / "sweep_pairwise_jaccard"
        for key, frame in result.sweep_pairwise_jaccard.items():
            paths.append(_write_frame(frame, jaccard_dir / f"jaccard_{key}.tsv"))
    if config.write_per_cell_metrics:
        parquet_path = layout.tables / "cell_qc_metrics.parquet"
        try:
            result.cell_metrics.to_parquet(parquet_path, index=True)
        except (ImportError, ModuleNotFoundError) as exc:
            result.warnings.append(f"Per-cell Parquet output was skipped because no Parquet engine is installed: {exc}")
        else:
            paths.append(parquet_path)
    paths.extend(
        create_all_plots(
            result,
            umap,
            layout.figures,
            transition_min_fraction=config.transition_min_fraction,
        )
    )
    result.output_files = paths
    return paths


def annotated_copy(
    adata: Any,
    result: PopulationEmbeddingQCResult,
) -> Any:
    """Return a copy with focused cell annotations and complete cached results."""
    from .storage import attach_focused_obs, store_population_embedding_qc

    output = adata.copy()
    attach_focused_obs(output, result)
    effective_config = PopulationEmbeddingQCConfig.model_validate(
        result.run_summary.get("configuration", {})
    )
    store_population_embedding_qc(output, result, config=effective_config)
    return output


def write_standalone_outputs(
    result: PopulationEmbeddingQCResult,
    *,
    adata: Any,
    umap: np.ndarray,
    config: PopulationEmbeddingQCConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    layout = standalone_layout(output_dir)
    paths = write_result_outputs(result, umap=umap, config=config, layout=layout)
    if config.write_annotated_h5ad:
        configured = Path(config.annotated_adata_path)
        target = configured if configured.is_absolute() else Path(output_dir) / configured.name
        if target.exists() and not overwrite:
            raise FileExistsError(f"Annotated AnnData output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.h5ad")
        try:
            annotated_copy(adata, result).write_h5ad(temporary)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        paths.append(target)
    return paths


__all__ = [
    "OutputLayout",
    "annotated_copy",
    "render_analysis_report",
    "standalone_layout",
    "write_result_outputs",
    "write_standalone_outputs",
]
