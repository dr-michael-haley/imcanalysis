"""Concise scientific QC reporting for neighbour-attributable signal analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class NeighbourSignalReport:
    figures: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)
    summaries: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, int | float | str] = field(default_factory=dict)


def _dense_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def marker_score_summary(adata: Any) -> pd.DataFrame:
    """Return the requested all-marker descriptive score table."""

    scores: np.ndarray = _dense_matrix(adata.X).astype(float, copy=False)
    rows = []
    for index, marker in enumerate(adata.var_names.astype(str)):
        values = scores[:, index]
        rows.append(
            {
                "marker": marker,
                "n_exemplars": int(adata.var.iloc[index]["halo_n_exemplars"]),
                "profile_available": bool(
                    adata.var.iloc[index]["halo_profile_available"]
                ),
                "median_score": float(np.median(values)),
                "p90_score": float(np.quantile(values, 0.90)),
                "p95_score": float(np.quantile(values, 0.95)),
                "fraction_above_0.25": float(np.mean(values >= 0.25)),
                "fraction_above_0.50": float(np.mean(values >= 0.50)),
                "fraction_above_0.75": float(np.mean(values >= 0.75)),
                "source_threshold": float(
                    adata.var.iloc[index]["halo_source_threshold"]
                ),
                "effective_halo_extent_px": float(
                    adata.var.iloc[index]["halo_effective_extent_px"]
                ),
                "skip_reason": str(adata.var.iloc[index]["halo_skip_reason"]),
            }
        )
    return pd.DataFrame(rows)


def profile_values_table(adata: Any) -> pd.DataFrame:
    """Return long-form learned profile values for auditing and reuse."""

    halo = adata.uns["marker_halo"]
    edges = np.asarray(halo["distance_bin_edges_px"], dtype=float)
    raw = np.asarray(halo["raw_median_profile"], dtype=float)
    final = np.asarray(halo["final_profile"], dtype=float)
    q25 = np.asarray(halo["profile_q25"], dtype=float)
    q75 = np.asarray(halo["profile_q75"], dtype=float)
    markers = [str(value) for value in halo["marker_names"]]
    rows = []
    for marker_index, marker in enumerate(markers):
        for bin_index in range(len(edges) - 1):
            rows.append(
                {
                    "marker": marker,
                    "distance_start_px": float(edges[bin_index]),
                    "distance_end_px": float(edges[bin_index + 1]),
                    "raw_median_normalized_excess": float(raw[marker_index, bin_index]),
                    "final_normalized_excess": float(final[marker_index, bin_index]),
                    "q25_normalized_excess": float(q25[marker_index, bin_index]),
                    "q75_normalized_excess": float(q75[marker_index, bin_index]),
                }
            )
    return pd.DataFrame(rows)


def select_qc_markers(
    adata: Any,
    requested: Sequence[str] | None,
    maximum: int,
) -> tuple[list[str], list[str]]:
    """Validate explicit QC markers or choose the most affected available markers."""

    available = [str(value) for value in adata.var_names]
    warnings: list[str] = []
    if requested:
        selected = []
        for marker in requested:
            if marker not in available:
                warnings.append(
                    f"Configured QC marker {marker!r} is absent from AnnData.var_names and was skipped."
                )
            elif marker not in selected:
                selected.append(marker)
        return selected[:maximum], warnings
    scores: np.ndarray = _dense_matrix(adata.X).astype(float, copy=False)
    profile_available = adata.var["halo_profile_available"].to_numpy(dtype=bool)
    ranked = sorted(
        (
            (float(np.quantile(scores[:, index], 0.95)), available[index])
            for index in range(adata.n_vars)
            if profile_available[index]
        ),
        reverse=True,
    )
    return [marker for _score, marker in ranked[:maximum]], warnings


def _plot_profiles(adata: Any, output_dir: Path, *, per_figure: int = 12) -> list[Path]:
    import matplotlib.pyplot as plt

    halo = adata.uns["marker_halo"]
    markers = [str(value) for value in halo["marker_names"]]
    available = adata.var["halo_profile_available"].to_numpy(dtype=bool)
    selected_indices = [index for index, value in enumerate(available) if value]
    if not selected_indices:
        return []
    edges = np.asarray(halo["distance_bin_edges_px"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    final = np.asarray(halo["final_profile"], dtype=float)
    q25 = np.asarray(halo["profile_q25"], dtype=float)
    q75 = np.asarray(halo["profile_q75"], dtype=float)
    paths: list[Path] = []
    for page, start in enumerate(range(0, len(selected_indices), per_figure), start=1):
        indices = selected_indices[start : start + per_figure]
        columns = min(3, len(indices))
        rows = int(math.ceil(len(indices) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.0 * columns, 3.7 * rows),
            squeeze=False,
        )
        for axis, marker_index in zip(axes.flat, indices, strict=False):
            marker = markers[marker_index]
            axis.fill_between(
                centers,
                q25[marker_index],
                q75[marker_index],
                color="#8fb8de",
                alpha=0.45,
                label="exemplar IQR",
            )
            axis.plot(
                centers,
                final[marker_index],
                color="#1f4e79",
                linewidth=2,
                marker="o",
                markersize=3,
                label="median halo",
            )
            n_exemplars = int(adata.var.iloc[marker_index]["halo_n_exemplars"])
            threshold = float(adata.var.iloc[marker_index]["halo_source_threshold"])
            axis.set_title(f"{marker}  n={n_exemplars}, source≥{threshold:.3g}")
            axis.set_xlabel("Outward distance from source mask (px)")
            axis.set_ylabel("Normalized excess signal")
            axis.set_xlim(edges[0], edges[-1])
            axis.axhline(0, color="#777777", linewidth=0.7)
        for axis in axes.flat[len(indices) :]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(
            "Empirical marker-specific halos (not forced monotonic)",
            y=1.01,
        )
        fig.tight_layout()
        path = output_dir / f"marker_halo_profiles_{page:02d}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_score_distributions(adata: Any, summary: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    scores: np.ndarray = _dense_matrix(adata.X).astype(float, copy=False)
    order = summary.sort_values(["p95_score", "median_score"], ascending=False).index
    marker_indices = [adata.var_names.get_loc(summary.loc[index, "marker"]) for index in order]
    labels = [str(summary.loc[index, "marker"]) for index in order]
    if adata.n_obs > 50000:
        sampled = np.linspace(0, adata.n_obs - 1, 50000, dtype=int)
    else:
        sampled = np.arange(adata.n_obs)
    values = [scores[sampled, index] for index in marker_indices]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(labels) + 2)))
    ax.boxplot(
        values,
        orientation="horizontal",
        tick_labels=labels,
        showfliers=False,
        widths=0.65,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Neighbour-Attributable Fraction")
    ax.set_ylabel("Marker")
    ax.set_title("Cell score distributions by marker")
    for threshold in (0.25, 0.5, 0.75):
        ax.axvline(threshold, color="#888888", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_umap(adata: Any, markers: Sequence[str], path: Path) -> Path:
    import matplotlib.pyplot as plt
    import scanpy as sc

    colors = [*markers, "halo_max_score", "halo_mean_score"]
    figure = sc.pl.umap(
        adata,
        color=colors,
        cmap="magma",
        ncols=3,
        show=False,
        return_fig=True,
        frameon=False,
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_population_matrix(
    adata: Any,
    markers: Sequence[str],
    population_obs: str,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt
    import scanpy as sc

    plotting = adata.copy()
    plotting.obs[population_obs] = plotting.obs[population_obs].astype("category")
    matrix_plot = sc.pl.matrixplot(
        plotting,
        var_names=list(markers),
        groupby=population_obs,
        use_raw=False,
        cmap="magma",
        vmin=0,
        vmax=1,
        colorbar_title="Mean neighbour-attributable fraction",
        show=False,
        return_fig=True,
    )
    matrix_plot.savefig(path, dpi=160, bbox_inches="tight")
    plt.close("all")
    return path


def _matrix_column(matrix: Any, index: int) -> np.ndarray:
    values = matrix[:, index]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).reshape(-1).astype(float, copy=False)


def _plot_expression_comparison(
    adata: Any,
    markers: Sequence[str],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    columns = min(3, len(markers))
    rows = int(math.ceil(len(markers) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.1 * columns, 4.4 * rows),
        squeeze=False,
    )
    if adata.n_obs > 20000:
        sampled = np.linspace(0, adata.n_obs - 1, 20000, dtype=int)
    else:
        sampled = np.arange(adata.n_obs)
    for axis, marker in zip(axes.flat, markers, strict=False):
        marker_index = adata.var_names.get_loc(marker)
        classic = _matrix_column(adata.layers["classic_intensities"], marker_index)[sampled]
        original = _matrix_column(adata.layers["original_X"], marker_index)[sampled]
        halo = _matrix_column(adata.X, marker_index)[sampled]
        finite = np.isfinite(classic) & np.isfinite(original) & np.isfinite(halo)
        scatter = axis.scatter(
            np.log1p(classic[finite]),
            original[finite],
            c=halo[finite],
            cmap="magma",
            vmin=0,
            vmax=1,
            s=7,
            alpha=0.55,
            linewidths=0,
        )
        axis.set_title(marker)
        axis.set_xlabel("log1p classic raw-mask intensity")
        axis.set_ylabel("Original X expression/confidence")
    for axis in axes.flat[len(markers) :]:
        axis.axis("off")
    colorbar = fig.colorbar(scatter, ax=list(axes.flat), shrink=0.75)
    colorbar.set_label("Neighbour-Attributable Fraction")
    fig.suptitle("Raw mask intensity vs independent input expression", y=1.01)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_summary(
    adata: Any,
    score_summary: pd.DataFrame,
    selected_markers: Sequence[str],
    output_adata_path: Path,
    path: Path,
) -> Path:
    halo = adata.uns["marker_halo"]
    marker_summary = halo["marker_summary"]
    skipped = marker_summary.loc[~marker_summary["halo_profile_available"].astype(bool)]
    usage = halo["worker_usage"]
    lines = [
        "# Neighbour-attributable signal QC",
        "",
        "The Neighbour-Attributable Fraction is the fraction of background-subtracted raw marker signal inside a cell mask that spatially coincides with an empirical halo projected from a strong neighbouring cell.",
        "",
        "It is a QC/uncertainty score describing spatial explainability. It is not a calibrated probability and does not prove that signal is artefactual.",
        "",
        "## Run summary",
        "",
        f"- Output AnnData: `{output_adata_path}`",
        f"- Cells × markers: {adata.n_obs:,} × {adata.n_vars:,}",
        f"- Markers with learned profiles: {int(marker_summary['halo_profile_available'].sum()):,}",
        f"- Skipped markers: {len(skipped):,}",
        f"- ROI workers: {usage['effective']} (limit {usage['cpu_limit']} from {usage['limit_source']})",
        f"- Detailed QC markers: {', '.join(selected_markers) if selected_markers else 'none'}",
        "",
        "## Exemplar and source definition",
        "",
        f"Exemplars came from `adata.obs[{halo['parameters'].get('exemplar_obs', 'Exemplar_stains')!r}]`. Each exemplar profile was background-subtracted and divided by its robust dilated-anchor source strength. Marker source thresholds were derived only from valid exemplar source strengths.",
        "",
        "The input expression matrix was not used to learn profiles, identify source cells, estimate backgrounds, or calculate scores. It is preserved in `layers['original_X']` for comparison.",
        "",
        "## Most affected markers",
        "",
        "| Marker | Median | P95 | Fraction ≥0.5 | Exemplars |",
        "|---|---:|---:|---:|---:|",
    ]
    for _index, row in score_summary.sort_values("p95_score", ascending=False).head(10).iterrows():
        lines.append(
            f"| {row['marker']} | {row['median_score']:.3f} | {row['p95_score']:.3f} | "
            f"{row['fraction_above_0.50']:.3f} | {int(row['n_exemplars'])} |"
        )
    if len(skipped):
        lines.extend(["", "## Skipped markers", ""])
        for marker, row in skipped.iterrows():
            lines.append(f"- `{marker}`: {row['halo_skip_reason']}")
    lines.extend(
        [
            "",
            "## Layer interpretation",
            "",
            "- `classic_intensities`: mean raw intensity inside each segmentation mask (when enabled).",
            "- `neighbour_attributable_intensity`: mean observed excess per mask pixel captured by neighbouring halos.",
            "- `residual_excess_intensity`: mean excess remaining after subtracting the projected halo, clipped at zero.",
            "- `original_X`: independent input expression/confidence values preserved unchanged.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_neighbour_signal_report(
    adata: Any,
    *,
    figures_dir: Path,
    tables_dir: Path,
    summaries_dir: Path,
    output_adata_path: Path,
    qc_markers: Sequence[str] | None,
    max_qc_markers: int,
    population_obs: str | None,
) -> NeighbourSignalReport:
    """Generate compact tables and figures without altering the AnnData asset."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    report = NeighbourSignalReport()
    summary = marker_score_summary(adata)
    score_path = tables_dir / "neighbour_attributable_score_summary.csv"
    summary.to_csv(score_path, index=False)
    report.tables.append(score_path)

    profiles = profile_values_table(adata)
    profile_path = tables_dir / "marker_halo_profiles.csv"
    profiles.to_csv(profile_path, index=False)
    report.tables.append(profile_path)
    marker_metadata_path = tables_dir / "marker_halo_metadata.csv"
    adata.uns["marker_halo"]["marker_summary"].to_csv(marker_metadata_path)
    report.tables.append(marker_metadata_path)
    for key, filename in (
        ("exemplar_statistics", "exemplar_profile_statistics.csv"),
        ("roi_marker_backgrounds", "roi_marker_backgrounds.csv"),
        ("unknown_exemplar_markers", "unknown_exemplar_markers.csv"),
    ):
        table = adata.uns["marker_halo"][key]
        path = tables_dir / filename
        table.to_csv(path, index=False)
        report.tables.append(path)

    report.figures.extend(_plot_profiles(adata, figures_dir))
    distribution_path = figures_dir / "neighbour_attributable_score_distributions.png"
    report.figures.append(_plot_score_distributions(adata, summary, distribution_path))
    selected, selection_warnings = select_qc_markers(adata, qc_markers, max_qc_markers)
    report.warnings.extend(selection_warnings)

    if "X_umap" in adata.obsm:
        report.figures.append(
            _plot_umap(adata, selected, figures_dir / "scanpy_umap_halo_scores.png")
        )
    elif "X_umap" not in adata.obsm:
        report.warnings.append("Skipped Scanpy UMAP QC because adata.obsm['X_umap'] is absent.")
    if population_obs:
        if population_obs not in adata.obs.columns:
            report.warnings.append(
                f"Skipped population-by-marker QC because adata.obs[{population_obs!r}] is absent."
            )
        elif selected and adata.obs[population_obs].notna().any():
            report.figures.append(
                _plot_population_matrix(
                    adata,
                    selected,
                    population_obs,
                    figures_dir / "scanpy_population_marker_halo_matrixplot.png",
                )
            )
    else:
        report.warnings.append("Skipped population-by-marker QC because no population observation is configured.")
    if selected and "classic_intensities" in adata.layers:
        report.figures.append(
            _plot_expression_comparison(
                adata,
                selected,
                figures_dir / "classic_originalX_halo_comparison.png",
            )
        )
    elif "classic_intensities" not in adata.layers:
        report.warnings.append(
            "Skipped classic/original-X comparison because classic intensity storage is disabled."
        )

    summary_path = summaries_dir / "neighbour_signal_summary.md"
    report.summaries.append(
        _write_summary(
            adata,
            summary,
            selected,
            output_adata_path,
            summary_path,
        )
    )
    report.metrics.update(
        {
            "cells": int(adata.n_obs),
            "markers": int(adata.n_vars),
            "markers_with_profiles": int(adata.var["halo_profile_available"].sum()),
            "skipped_markers": int((~adata.var["halo_profile_available"].astype(bool)).sum()),
            "qc_markers": len(selected),
        }
    )
    return report


__all__ = [
    "NeighbourSignalReport",
    "generate_neighbour_signal_report",
    "marker_score_summary",
    "profile_values_table",
    "select_qc_markers",
]
