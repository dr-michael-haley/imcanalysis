"""Concise scientific QC reporting for neighbour-attributable signal analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage


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


def _mask_only_source_metrics(adata: Any) -> tuple[int, int]:
    """Return strong mask-only source occurrences and affected ROI-marker pairs."""

    backgrounds = adata.uns.get("marker_halo", {}).get("roi_marker_backgrounds")
    if not isinstance(backgrounds, pd.DataFrame):
        return 0, 0
    if "unmapped_strong_source_cells" not in backgrounds.columns:
        return 0, 0
    counts = pd.to_numeric(
        backgrounds["unmapped_strong_source_cells"], errors="coerce"
    ).fillna(0)
    return int(counts.sum()), int((counts > 0).sum())


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


def population_source_target_summary(source_target_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sparse relationships by spatial source/target population and marker."""

    columns = [
        "source_population",
        "target_population",
        "marker",
        "source_target_relationships",
        "unique_target_cells",
        "total_attributable_intensity",
        "mean_fraction_of_observed_signal",
        "median_fraction_of_observed_signal",
        "total_fraction_of_attributable_signal",
        "mean_fraction_of_attributable_signal",
    ]
    required = {"source_population", "target_population", "marker"}
    if source_target_table.empty or not required.issubset(source_target_table.columns):
        return pd.DataFrame(columns=columns)
    usable = source_target_table.dropna(
        subset=["source_population", "target_population"]
    )
    if usable.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        usable.groupby(
            ["source_population", "target_population", "marker"],
            observed=True,
            sort=True,
        )
        .agg(
            source_target_relationships=("source_obs_index", "size"),
            unique_target_cells=("target_obs_index", "nunique"),
            total_attributable_intensity=("attributable_intensity", "sum"),
            mean_fraction_of_observed_signal=(
                "fraction_of_observed_signal",
                "mean",
            ),
            median_fraction_of_observed_signal=(
                "fraction_of_observed_signal",
                "median",
            ),
            total_fraction_of_attributable_signal=(
                "fraction_of_attributable_signal",
                "sum",
            ),
            mean_fraction_of_attributable_signal=(
                "fraction_of_attributable_signal",
                "mean",
            ),
        )
        .reset_index()
    )
    return summary.loc[:, columns]


def dominant_source_summary(
    adata: Any,
    source_target_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize dominant-source concentration and common population routes."""

    dominant: np.ndarray = _dense_matrix(
        adata.layers["dominant_source_attributable_fraction"]
    ).astype(float, copy=False)
    rows = []
    population_columns = {
        "source_population",
        "target_population",
    }.issubset(source_target_table.columns)
    for marker_index, marker in enumerate(adata.var_names.astype(str)):
        marker_rows = source_target_table.loc[
            source_target_table["marker"].astype(str).eq(marker)
        ]
        counts = marker_rows.groupby("target_obs_index", sort=False).size()
        affected_targets = counts.index.to_numpy(dtype=np.int64)
        if len(affected_targets):
            dominant_values = dominant[affected_targets, marker_index]
            dominant_gt_half = float(np.mean(dominant_values > 0.5))
            median_sources = float(np.median(counts.to_numpy(dtype=float)))
        else:
            dominant_gt_half = 0.0
            median_sources = 0.0
        common_relationships = ""
        if population_columns and not marker_rows.empty:
            relationship_counts = (
                marker_rows.dropna(
                    subset=["source_population", "target_population"]
                )
                .groupby(
                    ["source_population", "target_population"],
                    observed=True,
                    sort=True,
                )
                .size()
                .sort_values(ascending=False)
                .head(3)
            )
            common_relationships = "; ".join(
                f"{source} → {target} ({int(count)})"
                for (source, target), count in relationship_counts.items()
            )
        rows.append(
            {
                "marker": marker,
                "affected_target_cells": int(len(affected_targets)),
                "fraction_affected_targets_dominant_source_gt_0.5": (
                    dominant_gt_half
                ),
                "median_contributing_source_cells_per_affected_target": (
                    median_sources
                ),
                "most_common_source_target_population_relationships": (
                    common_relationships
                ),
            }
        )
    return pd.DataFrame(rows)


def select_qc_markers(
    adata: Any,
    requested: Sequence[str] | None,
    maximum: int | None,
) -> tuple[list[str], list[str]]:
    """Return the complete marker axis while accepting legacy subset settings."""

    available = [str(value) for value in adata.var_names]
    warnings: list[str] = []
    if requested:
        warnings.append(
            "neighbour_signal.qc_markers is retained for config compatibility but "
            "no longer restricts report figures; every AnnData marker was plotted."
        )
    if maximum is not None:
        warnings.append(
            "neighbour_signal.max_qc_markers is retained for config compatibility but "
            "no longer limits report figures; every AnnData marker was plotted."
        )
    return available, warnings


def _select_gallery_record(
    frame: pd.DataFrame,
    eligible: pd.Series,
    *,
    category: str,
    priority: str,
    ascending: bool,
    chosen: set[int],
    used_rois: set[str],
    grouped_sources: dict[int, list[int]],
    marker: str,
    obs_names: np.ndarray,
    roi_values: np.ndarray,
    object_values: np.ndarray,
    populations: np.ndarray,
) -> dict[str, Any] | None:
    """Return the next deterministic gallery record for one QC category."""

    candidates = frame.loc[
        eligible & ~frame["target_obs_index"].isin(chosen)
    ].copy()
    if candidates.empty:
        return None
    candidates["_roi_already_used"] = candidates["target_roi"].isin(used_rois)
    candidates = candidates.sort_values(
        ["_roi_already_used", priority, "target_obs_index"],
        ascending=[True, ascending, True],
        kind="stable",
    )
    winner = candidates.iloc[0]
    target_index = int(winner["target_obs_index"])
    source_index = int(winner["dominant_source_index"])
    contributing = grouped_sources.get(target_index, [])
    record = winner.drop(labels=["_roi_already_used"]).to_dict()
    record.update(
        {
            "gallery_type": "target_source",
            "example_category": category,
            "marker": marker,
            "source_obs_index": source_index,
            "source_cell_id": (
                obs_names[source_index] if 0 <= source_index < len(obs_names) else ""
            ),
            "source_roi": (
                roi_values[source_index] if 0 <= source_index < len(obs_names) else ""
            ),
            "source_segmentation_label": (
                int(object_values[source_index])
                if 0 <= source_index < len(obs_names)
                else -1
            ),
            "source_population": (
                populations[source_index]
                if 0 <= source_index < len(obs_names)
                else pd.NA
            ),
            "contributing_source_indices": ";".join(
                str(value) for value in contributing
            ),
        }
    )
    return record


def select_cell_gallery_examples(
    adata: Any,
    source_target_table: pd.DataFrame,
    markers: Sequence[str],
    *,
    examples_per_marker: int,
    roi_obs: str,
    object_id_obs: str,
    population_obs: str | None = None,
) -> pd.DataFrame:
    """Select deterministic, stratified target-cell examples for image QC."""

    if examples_per_marker < 1:
        raise ValueError("examples_per_marker must be positive")
    missing = [column for column in (roi_obs, object_id_obs) if column not in adata.obs]
    if missing:
        raise KeyError(f"AnnData is missing gallery identity observations: {missing}")
    scores = _dense_matrix(adata.X).astype(float, copy=False)
    original = _dense_matrix(adata.layers["original_X"]).astype(float, copy=False)
    classic = (
        _dense_matrix(adata.layers["classic_intensities"]).astype(float, copy=False)
        if "classic_intensities" in adata.layers
        else np.full(scores.shape, np.nan, dtype=float)
    )
    dominant = _dense_matrix(adata.layers["dominant_source_index"]).astype(
        np.int64, copy=False
    )
    dominant_observed = _dense_matrix(
        adata.layers["dominant_source_observed_fraction"]
    ).astype(float, copy=False)
    dominant_attributable = _dense_matrix(
        adata.layers["dominant_source_attributable_fraction"]
    ).astype(float, copy=False)
    roi_values = adata.obs[roi_obs].astype(str).to_numpy()
    object_values = pd.to_numeric(
        adata.obs[object_id_obs], errors="raise"
    ).to_numpy(dtype=np.int64)
    populations = (
        adata.obs[population_obs].astype("string").to_numpy()
        if population_obs and population_obs in adata.obs
        else np.full(adata.n_obs, pd.NA, dtype=object)
    )
    obs_names = adata.obs_names.astype(str).to_numpy()
    selected_exemplars = adata.uns["marker_halo"]["exemplar_selection"]
    selected_exemplar_pairs = {
        (int(row.source_obs_index), str(row.marker))
        for row in selected_exemplars.itertuples(index=False)
        if bool(row.selected)
    }
    positive_threshold = float(
        adata.uns["marker_halo"]["parameters"].get(
            "automatic_positive_threshold", 0.5
        )
    )
    records: list[dict[str, Any]] = []
    marker_lookup = {str(marker): index for index, marker in enumerate(adata.var_names)}
    for marker in markers:
        if marker not in marker_lookup:
            continue
        marker_index = marker_lookup[marker]
        frame = pd.DataFrame(
            {
                "target_obs_index": np.arange(adata.n_obs, dtype=np.int64),
                "target_cell_id": obs_names,
                "target_roi": roi_values,
                "target_segmentation_label": object_values,
                "target_population": populations,
                "neighbour_attributable_fraction": scores[:, marker_index],
                "original_x": original[:, marker_index],
                "classic_intensity": classic[:, marker_index],
                "dominant_source_index": dominant[:, marker_index],
                "dominant_source_observed_fraction": dominant_observed[:, marker_index],
                "dominant_source_attributable_fraction": dominant_attributable[
                    :, marker_index
                ],
            }
        )
        marker_relationships = source_target_table.loc[
            source_target_table["marker"].astype(str).eq(marker)
        ]
        grouped_sources: dict[int, list[int]] = {}
        if not marker_relationships.empty:
            ordered_relationships = marker_relationships.sort_values(
                ["target_obs_index", "attributable_intensity", "source_obs_index"],
                ascending=[True, False, True],
                kind="stable",
            )
            grouped_sources = {
                int(target): [int(value) for value in group["source_obs_index"]]
                for target, group in ordered_relationships.groupby(
                    "target_obs_index", sort=False
                )
            }
        frame["contributing_source_count"] = frame["target_obs_index"].map(
            {target: len(values) for target, values in grouped_sources.items()}
        ).fillna(0).astype(int)
        score_rank = frame["neighbour_attributable_fraction"].rank(
            method="average", pct=True
        )
        x_rank = frame["original_x"].rank(method="average", pct=True)
        classic_rank = frame["classic_intensity"].rank(method="average", pct=True)
        frame["halo_x_disagreement"] = score_rank - x_rank
        frame["classic_low_halo_contrast"] = classic_rank - score_rank
        frame["is_selected_exemplar"] = [
            (int(index), marker) in selected_exemplar_pairs
            for index in frame["target_obs_index"]
        ]
        chosen: set[int] = set()
        used_rois: set[str] = set()

        affected = frame["neighbour_attributable_fraction"] > 0
        criteria = [
            (
                "high_dominant_source",
                affected
                & (frame["dominant_source_index"] >= 0)
                & (frame["dominant_source_attributable_fraction"] > 0.5),
                "neighbour_attributable_fraction",
            ),
            (
                "competing_sources",
                affected & (frame["contributing_source_count"] > 1),
                "contributing_source_count",
            ),
            ("halo_high_x_disagreement", affected, "halo_x_disagreement"),
            (
                "isolated_x_positive_low_halo",
                (frame["contributing_source_count"] == 0)
                & (frame["neighbour_attributable_fraction"] <= 0.05)
                & (frame["original_x"] >= positive_threshold),
                "original_x",
            ),
            (
                "classic_high_low_halo",
                np.isfinite(frame["classic_intensity"])
                & (frame["neighbour_attributable_fraction"] <= 0.25),
                "classic_low_halo_contrast",
            ),
            (
                "source_self_control",
                frame["is_selected_exemplar"]
                & (frame["neighbour_attributable_fraction"] <= 0.1),
                "original_x",
            ),
        ]
        for category, eligible, priority in criteria:
            if len(chosen) >= examples_per_marker:
                break
            record = _select_gallery_record(
                frame,
                eligible,
                category=category,
                priority=priority,
                ascending=False,
                chosen=chosen,
                used_rois=used_rois,
                grouped_sources=grouped_sources,
                marker=marker,
                obs_names=obs_names,
                roi_values=roi_values,
                object_values=object_values,
                populations=populations,
            )
            if record is not None:
                records.append(record)
                chosen.add(int(record["target_obs_index"]))
                used_rois.add(str(record["target_roi"]))
        while len(chosen) < examples_per_marker:
            before = len(chosen)
            record = _select_gallery_record(
                frame,
                affected,
                category="highest_remaining_attribution",
                priority="neighbour_attributable_fraction",
                ascending=False,
                chosen=chosen,
                used_rois=used_rois,
                grouped_sources=grouped_sources,
                marker=marker,
                obs_names=obs_names,
                roi_values=roi_values,
                object_values=object_values,
                populations=populations,
            )
            if record is not None:
                records.append(record)
                chosen.add(int(record["target_obs_index"]))
                used_rois.add(str(record["target_roi"]))
            if len(chosen) == before:
                break
    return pd.DataFrame(records)


def select_exemplar_gallery_examples(
    adata: Any,
    markers: Sequence[str],
    *,
    examples_per_marker: int,
) -> pd.DataFrame:
    """Choose valid exemplars spanning the raw source-strength range."""

    statistics = adata.uns["marker_halo"]["exemplar_statistics"]
    selected: list[pd.DataFrame] = []
    for marker in markers:
        frame = statistics.loc[
            statistics["marker"].astype(str).eq(marker)
            & statistics["valid"].astype(bool)
        ].sort_values(["source_strength", "roi", "object_id"], kind="stable")
        if frame.empty:
            continue
        count = min(examples_per_marker, len(frame))
        positions = np.unique(
            np.rint(np.linspace(0, len(frame) - 1, count)).astype(int)
        )
        chosen = frame.iloc[positions].copy()
        chosen["gallery_type"] = "exemplar_halo"
        chosen["example_category"] = "selected_exemplar"
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def select_automatic_decision_gallery_examples(
    adata: Any,
    markers: Sequence[str],
) -> pd.DataFrame:
    """Choose representative automatic candidate decisions for image review."""

    decisions = adata.uns["marker_halo"]["exemplar_selection"]
    decisions = decisions.loc[
        decisions["selection_origin"].astype(str).eq("automatic")
    ].copy()
    selected: list[pd.Series] = []
    for marker in markers:
        frame = decisions.loc[decisions["marker"].astype(str).eq(marker)].copy()
        if frame.empty:
            continue
        categories = (
            ("automatic_selected", frame["selected"].astype(bool)),
            (
                "rejected_same_marker_clearance",
                frame["reason"].fillna("").astype(str).str.contains(
                    "same_marker_positive_within_clearance"
                ),
            ),
            (
                "rejected_radial_coverage",
                frame["reason"].fillna("").astype(str).str.contains(
                    "insufficient_unassigned_radial_pixels"
                ),
            ),
            (
                "eligible_not_sampled",
                frame["eligible"].astype(bool) & ~frame["selected"].astype(bool),
            ),
        )
        for category, eligible in categories:
            candidates = frame.loc[eligible].sort_values(
                ["input_x_value", "source_obs_index"],
                ascending=[False, True],
                kind="stable",
            )
            if candidates.empty:
                continue
            row = candidates.iloc[0].copy()
            row["gallery_type"] = "automatic_decision"
            row["example_category"] = category
            selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame()


def _marker_plot_stem(marker_index: int, marker: str) -> str:
    """Return an ordered, collision-resistant stem for a marker QC figure."""

    return f"{marker_index + 1:03d}_{_gallery_filename(marker)}"


def _plot_profiles(adata: Any, output_dir: Path) -> list[Path]:
    """Write one empirical halo-profile figure for every marker."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    halo = adata.uns["marker_halo"]
    markers = [str(value) for value in halo["marker_names"]]
    available = adata.var["halo_profile_available"].to_numpy(dtype=bool)
    edges = np.asarray(halo["distance_bin_edges_px"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    final = np.asarray(halo["final_profile"], dtype=float)
    q25 = np.asarray(halo["profile_q25"], dtype=float)
    q75 = np.asarray(halo["profile_q75"], dtype=float)
    paths: list[Path] = []
    for marker_index, marker in enumerate(markers):
        fig, axis = plt.subplots(figsize=(6.4, 4.5))
        if available[marker_index]:
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
            axis.set_title(
                f"{marker}: empirical halo (n={n_exemplars}, source≥{threshold:.3g})"
            )
            axis.legend(frameon=False)
        else:
            reason = str(adata.var.iloc[marker_index]["halo_skip_reason"])
            axis.text(
                0.5,
                0.5,
                f"Halo profile unavailable\n{reason or 'No reliable exemplar profile was learned.'}",
                ha="center",
                va="center",
                transform=axis.transAxes,
                wrap=True,
            )
            axis.set_title(f"{marker}: halo profile unavailable")
        axis.set_xlabel("Outward distance from source mask (px)")
        axis.set_ylabel("Normalized excess signal")
        axis.set_xlim(edges[0], edges[-1])
        axis.axhline(0, color="#777777", linewidth=0.7)
        fig.tight_layout()
        path = output_dir / (
            f"marker_halo_profile_{_marker_plot_stem(marker_index, marker)}.png"
        )
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_exemplar_selection(
    adata: Any,
    markers: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    """Write one X-positive candidate/selection figure for every marker."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = adata.uns["marker_halo"]["exemplar_selection"]
    if not decisions.empty:
        decisions = decisions.loc[
            decisions["selection_origin"].astype(str).eq("automatic")
            & decisions["marker"].astype(str).isin(
                [str(marker) for marker in markers]
            )
        ].copy()
    parameters = adata.uns["marker_halo"]["parameters"]
    clearance = float(parameters.get("automatic_same_marker_clearance_px", 10.0))
    capped_distance = max(clearance * 1.25, clearance + 1.0)
    styles = (
        ("rejected", "#bdbdbd", lambda frame: ~frame["eligible"].astype(bool)),
        (
            "eligible, not sampled",
            "#3182bd",
            lambda frame: frame["eligible"].astype(bool)
            & ~frame["selected"].astype(bool),
        ),
        ("selected", "#de2d26", lambda frame: frame["selected"].astype(bool)),
    )
    paths: list[Path] = []
    for marker_index, marker in enumerate(markers):
        fig, axis = plt.subplots(figsize=(6.4, 4.5))
        frame = decisions.loc[
            decisions["marker"].astype(str).eq(str(marker))
        ] if not decisions.empty else decisions
        if frame.empty:
            axis.text(
                0.5,
                0.5,
                "No automatic exemplar-candidate records\n"
                "(manual mode or no X-positive candidates).",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            distances = pd.to_numeric(
                frame["nearest_same_marker_positive_distance_px"], errors="coerce"
            ).to_numpy(dtype=float)
            distances = np.where(np.isfinite(distances), distances, capped_distance)
            for label, color, selector in styles:
                selected_rows = selector(frame).to_numpy(dtype=bool)
                if np.any(selected_rows):
                    axis.scatter(
                        distances[selected_rows],
                        pd.to_numeric(
                            frame.loc[selected_rows, "input_x_value"]
                        ).to_numpy(dtype=float),
                        s=22,
                        alpha=0.75,
                        color=color,
                        edgecolors="none",
                        label=label,
                    )
            threshold_values = pd.to_numeric(
                frame["positive_threshold"], errors="coerce"
            ).to_numpy(dtype=float)
            finite_thresholds = threshold_values[np.isfinite(threshold_values)]
            if finite_thresholds.size:
                axis.axhline(
                    float(finite_thresholds[0]),
                    color="black",
                    linestyle=":",
                    lw=1,
                )
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels, frameon=False)
        axis.axvline(clearance, color="black", linestyle="--", lw=1)
        axis.set_title(
            f"{marker}: automatic exemplar candidates and selection"
        )
        axis.set_xlabel("Nearest same-marker X-positive cell (px)")
        axis.set_ylabel("Input AnnData.X score")
        axis.set_xlim(left=0, right=capped_distance * 1.03)
        fig.tight_layout()
        path = output_dir / (
            "automatic_exemplar_selection_"
            f"{_marker_plot_stem(marker_index, str(marker))}.png"
        )
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _gallery_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "marker"


def _crop_slice_for_labels(
    mask: np.ndarray,
    labels: Sequence[int],
    radius: int,
) -> tuple[slice, slice]:
    selected = np.isin(mask, np.asarray([int(value) for value in labels], dtype=np.int64))
    coordinates = np.argwhere(selected)
    if coordinates.size == 0:
        raise ValueError(f"Cannot crop absent segmentation label(s): {list(labels)}")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    return (
        slice(max(0, int(y0) - radius), min(mask.shape[0], int(y1) + radius)),
        slice(max(0, int(x0) - radius), min(mask.shape[1], int(x1) + radius)),
    )


def _draw_cell_outlines(
    axis: Any,
    mask: np.ndarray,
    *,
    target_label: int,
    dominant_source_label: int = -1,
    secondary_source_labels: Sequence[int] = (),
    positive_labels: Sequence[int] = (),
) -> None:
    positive_set = {int(value) for value in positive_labels}
    secondary_set = {int(value) for value in secondary_source_labels}
    for raw_label in np.unique(mask):
        label = int(raw_label)
        if label <= 0:
            continue
        if label == int(target_label):
            color, width = "#00ffff", 1.7
        elif label == int(dominant_source_label):
            color, width = "#ff00ff", 1.7
        elif label in secondary_set:
            color, width = "#ff9f1c", 1.3
        elif label in positive_set:
            color, width = "#ff00ff", 1.2
        else:
            color, width = "#c7c7c7", 0.45
        axis.contour(mask == label, levels=[0.5], colors=[color], linewidths=width)


def _shared_excess_scale(*arrays: np.ndarray) -> float:
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    positive = values[np.isfinite(values) & (values > 0)]
    if not positive.size:
        return 1.0
    return max(float(np.quantile(positive, 0.995)), np.finfo(np.float32).eps)


def _render_target_source_contact_sheet(
    marker: str,
    crops: Sequence[dict[str, Any]],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    columns = (
        "Raw marker",
        "Observed excess",
        "Projected halo",
        "Attributable",
        "Residual excess",
        "Winning source in target",
    )
    fig, axes = plt.subplots(
        len(crops),
        len(columns),
        figsize=(19.0, max(3.2, 3.15 * len(crops))),
        squeeze=False,
    )
    for row_index, crop in enumerate(crops):
        record = crop["record"]
        mask = crop["mask"]
        target_label = int(record["target_segmentation_label"])
        dominant_label = int(record.get("source_segmentation_label", -1))
        secondary_labels = [
            int(value)
            for value in crop["source_labels"]
            if int(value) != dominant_label
        ]
        scale = _shared_excess_scale(
            crop["observed_excess"],
            crop["predicted"],
            crop["attributable"],
            crop["residual"],
        )
        arrays = (
            (crop["raw"], crop["background"], crop["background"] + scale),
            (crop["observed_excess"], 0.0, scale),
            (crop["predicted"], 0.0, scale),
            (crop["attributable"], 0.0, scale),
            (crop["residual"], 0.0, scale),
        )
        for column_index, (array, vmin, vmax) in enumerate(arrays):
            axis = axes[row_index, column_index]
            axis.imshow(array, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
            _draw_cell_outlines(
                axis,
                mask,
                target_label=target_label,
                dominant_source_label=dominant_label,
                secondary_source_labels=secondary_labels,
            )
            axis.set_axis_off()
        identity_axis = axes[row_index, 5]
        identity_axis.set_facecolor("black")
        identity_axis.imshow(
            crop["observed_excess"],
            cmap="gray",
            vmin=0,
            vmax=scale,
            alpha=0.18,
            interpolation="nearest",
        )
        target_pixels = mask == target_label
        attributable_pixels = target_pixels & (crop["attributable"] > 0)
        winning_sources = sorted(
            int(value)
            for value in np.unique(crop["source_index"][attributable_pixels])
            if int(value) >= 0
        )
        categorical = np.full(mask.shape, np.nan, dtype=float)
        alpha = np.zeros(mask.shape, dtype=float)
        for source_number, source_index in enumerate(winning_sources):
            pixels = attributable_pixels & (crop["source_index"] == source_index)
            categorical[pixels] = float(source_number)
            alpha[pixels] = np.clip(
                crop["attributable"][pixels] / scale,
                0.25,
                1.0,
            )
        if winning_sources:
            identity_axis.imshow(
                categorical,
                cmap="tab10",
                vmin=0,
                vmax=max(1, len(winning_sources) - 1),
                alpha=np.where(alpha > 0, np.maximum(alpha, 0.78), 0.0),
                interpolation="nearest",
            )
            identity_axis.text(
                0.02,
                0.02,
                "sources: " + ", ".join(str(value) for value in winning_sources),
                transform=identity_axis.transAxes,
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
            )
        else:
            identity_axis.text(
                0.5,
                0.5,
                "No attributable\nsource pixels",
                transform=identity_axis.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
        _draw_cell_outlines(
            identity_axis,
            mask,
            target_label=target_label,
            dominant_source_label=dominant_label,
            secondary_source_labels=secondary_labels,
        )
        identity_axis.set_axis_off()
        axes[row_index, 0].text(
            -0.06,
            0.5,
            f"{record['example_category']}\n"
            f"target {record['target_cell_id']} ({record.get('target_population', '')})\n"
            f"source {record.get('source_cell_id', '') or 'none'} "
            f"({record.get('source_population', '')})\n"
            f"NAF={float(record['neighbour_attributable_fraction']):.2f}; "
            f"dominant obs={float(record['dominant_source_observed_fraction']):.2f}; "
            f"n sources={int(record['contributing_source_count'])}",
            transform=axes[row_index, 0].transAxes,
            fontsize=7.5,
            ha="right",
            va="center",
            clip_on=False,
        )
        if row_index == 0:
            for column_index, title in enumerate(columns):
                axes[row_index, column_index].set_title(title, fontsize=10)
    fig.suptitle(
        f"{marker}: target signal spatially explainable by neighbouring sources\n"
        "target=cyan; dominant source=magenta; secondary sources=orange",
        y=1.005,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_exemplar_contact_sheet(
    adata: Any,
    marker: str,
    crops: Sequence[dict[str, Any]],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    halo = adata.uns["marker_halo"]
    marker_index = [str(value) for value in halo["marker_names"]].index(marker)
    edges = np.asarray(halo["distance_bin_edges_px"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    median = np.asarray(halo["final_profile"], dtype=float)[marker_index]
    q25 = np.asarray(halo["profile_q25"], dtype=float)[marker_index]
    q75 = np.asarray(halo["profile_q75"], dtype=float)[marker_index]
    profile_values = halo["exemplar_profile_values"]
    fig, axes = plt.subplots(
        len(crops),
        3,
        figsize=(11.5, max(3.1, 3.0 * len(crops))),
        squeeze=False,
    )
    for row_index, crop in enumerate(crops):
        record = crop["record"]
        mask = crop["mask"]
        object_id = int(record["object_id"])
        source = mask == object_id
        distance = ndimage.distance_transform_edt(~source)
        usable = (mask == 0) & (distance > 0) & (distance <= edges[-1])
        scale = _shared_excess_scale(crop["raw"] - float(record["background"]))
        axes[row_index, 0].imshow(
            crop["raw"],
            cmap="magma",
            vmin=float(record["background"]),
            vmax=float(record["background"]) + scale,
            interpolation="nearest",
        )
        _draw_cell_outlines(axes[row_index, 0], mask, target_label=object_id)
        axes[row_index, 0].set_axis_off()
        axes[row_index, 1].imshow(crop["raw"], cmap="gray", interpolation="nearest")
        radial = np.ma.masked_where(~usable, distance)
        axes[row_index, 1].imshow(
            radial,
            cmap="turbo",
            vmin=1,
            vmax=float(edges[-1]),
            alpha=0.72,
            interpolation="nearest",
        )
        _draw_cell_outlines(axes[row_index, 1], mask, target_label=object_id)
        axes[row_index, 1].set_axis_off()
        exemplar_curve = profile_values.loc[
            profile_values["marker"].astype(str).eq(marker)
            & profile_values["roi"].astype(str).eq(str(record["roi"]))
            & (profile_values["object_id"].astype(int) == object_id)
        ].sort_values("distance_start_px")
        axes[row_index, 2].fill_between(
            centers,
            q25,
            q75,
            color="#9ecae1",
            alpha=0.45,
            label="exemplar IQR",
        )
        axes[row_index, 2].plot(centers, median, color="#08519c", lw=2, label="final median")
        axes[row_index, 2].plot(
            centers,
            exemplar_curve["normalized_excess"].to_numpy(dtype=float),
            color="#de2d26",
            marker="o",
            ms=3,
            lw=1.2,
            label="this exemplar",
        )
        axes[row_index, 2].axhline(0, color="black", lw=0.5)
        axes[row_index, 2].set_xlabel("Distance outside mask (px)")
        axes[row_index, 2].set_ylabel("Normalized excess")
        axes[row_index, 0].text(
            -0.06,
            0.5,
            f"{record['source_cell_id']} / {record['roi']}\n"
            f"{record['selection_origin']}; X={float(record['input_x_value']):.2f}\n"
            f"source={float(record['source_strength']):.2f}; "
            f"background={float(record['background']):.2f}",
            transform=axes[row_index, 0].transAxes,
            fontsize=8,
            ha="right",
            va="center",
            clip_on=False,
        )
        if row_index == 0:
            axes[row_index, 0].set_title("Raw marker and source mask")
            axes[row_index, 1].set_title("Unassigned radial pixels used")
            axes[row_index, 2].set_title("Exemplar profile vs aggregate")
            axes[row_index, 2].legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"{marker}: empirical halo-learning exemplars\n"
        "other segmented-cell pixels are excluded from radial averages",
        y=1.005,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_automatic_decision_contact_sheet(
    marker: str,
    crops: Sequence[dict[str, Any]],
    path: Path,
    *,
    max_halo_px: int,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(crops),
        2,
        figsize=(10.0, max(3.0, 2.9 * len(crops))),
        squeeze=False,
    )
    for row_index, crop in enumerate(crops):
        record = crop["record"]
        mask = crop["mask"]
        object_id = int(record["object_id"])
        source = mask == object_id
        distance = ndimage.distance_transform_edt(~source)
        usable = (mask == 0) & (distance > 0) & (distance <= float(max_halo_px))
        axes[row_index, 0].imshow(crop["raw"], cmap="magma", interpolation="nearest")
        _draw_cell_outlines(
            axes[row_index, 0],
            mask,
            target_label=object_id,
            positive_labels=crop["positive_labels"],
        )
        axes[row_index, 0].set_axis_off()
        axes[row_index, 1].imshow(crop["raw"], cmap="gray", interpolation="nearest")
        axes[row_index, 1].imshow(
            np.ma.masked_where(~usable, distance),
            cmap="turbo",
            vmin=1,
            vmax=max_halo_px,
            alpha=0.72,
            interpolation="nearest",
        )
        _draw_cell_outlines(
            axes[row_index, 1],
            mask,
            target_label=object_id,
            positive_labels=crop["positive_labels"],
        )
        axes[row_index, 1].set_axis_off()
        nearest = float(record["nearest_same_marker_positive_distance_px"])
        nearest_text = f"{nearest:.1f}px" if np.isfinite(nearest) else "> search radius"
        axes[row_index, 0].text(
            -0.06,
            0.5,
            f"{record['example_category']}\n{record['source_cell_id']} / {record['roi']}\n"
            f"X={float(record['input_x_value']):.2f}; nearest positive={nearest_text}\n"
            f"min radial pixels={int(record['min_unassigned_pixels_per_bin'])}\n"
            f"{record['reason'] or 'eligible'}",
            transform=axes[row_index, 0].transAxes,
            fontsize=7.5,
            ha="right",
            va="center",
            clip_on=False,
        )
        if row_index == 0:
            axes[row_index, 0].set_title(
                "Candidate and same-marker\nX-positive cells",
                fontsize=9,
            )
            axes[row_index, 1].set_title(
                "Usable unassigned\nhalo pixels",
                fontsize=9,
            )
    fig.suptitle(
        f"{marker}: automatic exemplar decisions\n"
        "candidate=cyan; other same-marker X-positive cells=magenta",
        y=1.005,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_cell_qc_galleries(
    adata: Any,
    source_target_table: pd.DataFrame,
    roi_inputs: Sequence[Any],
    markers: Sequence[str],
    *,
    figures_dir: Path,
    examples_per_marker: int,
    crop_margin_px: int,
    roi_obs: str,
    object_id_obs: str,
    population_obs: str | None,
) -> tuple[list[Path], pd.DataFrame, list[str]]:
    """Render bounded image galleries by reusing exact halo-application logic."""

    from tifffile import imread

    from SpatialBiologyToolkit.neighbour_signal import (
        HaloParameters,
        MarkerHaloProfile,
        calculate_marker_halo_maps,
        source_anchor_labels,
    )

    gallery_dir = figures_dir / "cell_galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    target_examples = select_cell_gallery_examples(
        adata,
        source_target_table,
        markers,
        examples_per_marker=examples_per_marker,
        roi_obs=roi_obs,
        object_id_obs=object_id_obs,
        population_obs=population_obs,
    )
    exemplar_examples = select_exemplar_gallery_examples(
        adata,
        markers,
        examples_per_marker=min(4, examples_per_marker),
    )
    decision_examples = select_automatic_decision_gallery_examples(adata, markers)
    parameters_dict = adata.uns["marker_halo"]["parameters"]
    parameters = HaloParameters(
        max_halo_px=int(parameters_dict.get("max_halo_px", 8)),
        source_anchor_dilation_px=int(
            parameters_dict.get("source_anchor_dilation_px", 2)
        ),
        source_anchor_quantile=float(
            parameters_dict.get("source_anchor_quantile", 0.95)
        ),
        min_exemplars=int(parameters_dict.get("min_exemplars", 5)),
        source_threshold_quantile=float(
            parameters_dict.get("source_threshold_quantile", 0.1)
        ),
        halo_aggregation=str(parameters_dict.get("halo_aggregation", "max")),
        exemplar_mode=str(parameters_dict.get("exemplar_mode", "manual")),
        automatic_positive_threshold=float(
            parameters_dict.get("automatic_positive_threshold", 0.5)
        ),
        automatic_same_marker_clearance_px=float(
            parameters_dict.get("automatic_same_marker_clearance_px", 10.0)
        ),
        automatic_target_exemplars_per_marker=int(
            parameters_dict.get("automatic_target_exemplars_per_marker", 30)
        ),
        automatic_max_exemplars_per_roi=int(
            parameters_dict.get("automatic_max_exemplars_per_roi", 5)
        ),
        automatic_min_pixels_per_bin=int(
            parameters_dict.get("automatic_min_pixels_per_bin", 8)
        ),
    )
    parameters.validate()
    contexts = {str(context.name): context for context in roi_inputs}
    roi_values = adata.obs[roi_obs].astype(str).to_numpy()
    object_values = pd.to_numeric(
        adata.obs[object_id_obs], errors="raise"
    ).to_numpy(dtype=np.int64)
    populations = (
        adata.obs[population_obs].astype("string").to_numpy()
        if population_obs and population_obs in adata.obs
        else np.full(adata.n_obs, pd.NA, dtype=object)
    )
    marker_names = [str(value) for value in adata.var_names]
    halo = adata.uns["marker_halo"]
    scores = _dense_matrix(adata.X).astype(float, copy=False)
    classic = (
        _dense_matrix(adata.layers["classic_intensities"]).astype(float, copy=False)
        if "classic_intensities" in adata.layers
        else None
    )
    exemplar_selection = halo["exemplar_selection"]
    target_crops: dict[str, list[dict[str, Any]]] = {marker: [] for marker in markers}
    exemplar_crops: dict[str, list[dict[str, Any]]] = {marker: [] for marker in markers}
    decision_crops: dict[str, list[dict[str, Any]]] = {marker: [] for marker in markers}
    manifest_records: list[dict[str, Any]] = []

    def rows_for(frame: pd.DataFrame, roi: str, marker: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        roi_column = "target_roi" if "target_roi" in frame.columns else "roi"
        return frame.loc[
            frame[roi_column].astype(str).eq(roi)
            & frame["marker"].astype(str).eq(marker)
        ]

    requested_rois = sorted(
        set(target_examples.get("target_roi", pd.Series(dtype=str)).astype(str))
        | set(exemplar_examples.get("roi", pd.Series(dtype=str)).astype(str))
        | set(decision_examples.get("roi", pd.Series(dtype=str)).astype(str))
    )
    for roi in requested_rois:
        context = contexts.get(roi)
        if context is None:
            warnings.append(f"Skipped gallery examples for unresolved ROI {roi!r}.")
            continue
        mask = np.asarray(np.squeeze(imread(context.mask_path)), dtype=np.int64)
        if mask.ndim != 2:
            raise ValueError(f"Gallery mask for ROI {roi!r} is not 2D: {mask.shape}")
        roi_positions = np.flatnonzero(roi_values == roi)
        source_obs_indices = {
            int(object_values[position]): int(position) for position in roi_positions
        }
        channel_paths = {
            str(marker_name): str(path)
            for marker_name, path in zip(
                context.channel_names, context.channel_files, strict=True
            )
        }
        roi_markers = [
            marker
            for marker in markers
            if not rows_for(target_examples, roi, marker).empty
            or not rows_for(exemplar_examples, roi, marker).empty
            or not rows_for(decision_examples, roi, marker).empty
        ]
        anchors: np.ndarray | None = None
        for marker in roi_markers:
            if marker not in channel_paths:
                warnings.append(
                    f"Skipped gallery examples for ROI {roi!r}, marker {marker!r}: "
                    "raw channel path is unavailable."
                )
                continue
            image = np.asarray(np.squeeze(imread(channel_paths[marker])), dtype=np.float32)
            if image.shape != mask.shape:
                raise ValueError(
                    f"Gallery image for ROI {roi!r}, marker {marker!r} has shape "
                    f"{image.shape}; expected {mask.shape}"
                )
            marker_index = marker_names.index(marker)
            marker_profile = MarkerHaloProfile(
                marker=marker,
                available=bool(adata.var.iloc[marker_index]["halo_profile_available"]),
                raw_median=np.asarray(halo["raw_median_profile"])[marker_index].astype(
                    np.float32
                ),
                final=np.asarray(halo["final_profile"])[marker_index].astype(np.float32),
                q25=np.asarray(halo["profile_q25"])[marker_index].astype(np.float32),
                q75=np.asarray(halo["profile_q75"])[marker_index].astype(np.float32),
                n_configured_exemplars=int(
                    adata.var.iloc[marker_index]["halo_n_configured_exemplars"]
                ),
                n_valid_exemplars=int(
                    adata.var.iloc[marker_index]["halo_n_exemplars"]
                ),
                source_threshold=float(
                    adata.var.iloc[marker_index]["halo_source_threshold"]
                ),
                effective_extent_px=float(
                    adata.var.iloc[marker_index]["halo_effective_extent_px"]
                ),
                skip_reason=str(adata.var.iloc[marker_index]["halo_skip_reason"]),
            )
            marker_target_rows = rows_for(target_examples, roi, marker)
            maps = None
            if not marker_target_rows.empty:
                if anchors is None:
                    anchors = source_anchor_labels(
                        mask, parameters.source_anchor_dilation_px
                    )
                maps = calculate_marker_halo_maps(
                    mask,
                    image,
                    anchors,
                    marker_profile,
                    parameters,
                    source_obs_indices,
                    roi=roi,
                    marker=marker,
                )
            for record in marker_target_rows.to_dict(orient="records"):
                source_indices = [
                    int(value)
                    for value in str(record.get("contributing_source_indices", "")).split(";")
                    if value
                ]
                source_labels = [
                    int(object_values[value])
                    for value in source_indices
                    if 0 <= value < adata.n_obs and roi_values[value] == roi
                ]
                labels = [int(record["target_segmentation_label"]), *source_labels]
                crop_slice = _crop_slice_for_labels(
                    mask,
                    labels,
                    parameters.max_halo_px + crop_margin_px,
                )
                if maps is None:
                    raise RuntimeError("Target gallery maps were not calculated")
                crop = {
                    "record": record,
                    "raw": image[crop_slice].copy(),
                    "mask": mask[crop_slice].copy(),
                    "observed_excess": maps.observed_excess[crop_slice].copy(),
                    "predicted": maps.projected.predicted[crop_slice].copy(),
                    "source_index": maps.projected.source_index[crop_slice].copy(),
                    "attributable": maps.attributable[crop_slice].copy(),
                    "residual": maps.residual[crop_slice].copy(),
                    "background": maps.background,
                    "source_labels": source_labels,
                }
                target_crops[marker].append(crop)
                dominant_label = int(record.get("source_segmentation_label", -1))
                boundary_distance = float("nan")
                if dominant_label > 0 and np.any(mask == dominant_label):
                    distance = ndimage.distance_transform_edt(mask != dominant_label)
                    target_pixels = mask == int(record["target_segmentation_label"])
                    if np.any(target_pixels):
                        boundary_distance = float(np.min(distance[target_pixels]))
                manifest_record = dict(record)
                manifest_record.update(
                    {
                        "source_target_mask_distance_px": boundary_distance,
                        "crop_y0": int(crop_slice[0].start),
                        "crop_y1": int(crop_slice[0].stop),
                        "crop_x0": int(crop_slice[1].start),
                        "crop_x1": int(crop_slice[1].stop),
                        "roi_marker_background": maps.background,
                    }
                )
                manifest_records.append(manifest_record)
            for record in rows_for(exemplar_examples, roi, marker).to_dict(
                orient="records"
            ):
                crop_slice = _crop_slice_for_labels(
                    mask,
                    [int(record["object_id"])],
                    parameters.max_halo_px + crop_margin_px,
                )
                exemplar_crops[marker].append(
                    {
                        "record": record,
                        "raw": image[crop_slice].copy(),
                        "mask": mask[crop_slice].copy(),
                    }
                )
                obs_index = int(record["source_obs_index"])
                manifest_records.append(
                    {
                        "gallery_type": "exemplar_halo",
                        "example_category": "selected_exemplar",
                        "marker": marker,
                        "target_obs_index": obs_index,
                        "target_cell_id": str(record["source_cell_id"]),
                        "target_roi": roi,
                        "target_segmentation_label": int(record["object_id"]),
                        "target_population": populations[obs_index],
                        "source_obs_index": -1,
                        "source_cell_id": "",
                        "source_roi": "",
                        "source_segmentation_label": -1,
                        "source_population": pd.NA,
                        "neighbour_attributable_fraction": float(
                            scores[obs_index, marker_index]
                        ),
                        "original_x": float(record["input_x_value"]),
                        "classic_intensity": float(
                            classic[obs_index, marker_index]
                        )
                        if classic is not None
                        else float("nan"),
                        "selection_origin": str(record["selection_origin"]),
                        "source_strength": float(record["source_strength"]),
                        "crop_y0": int(crop_slice[0].start),
                        "crop_y1": int(crop_slice[0].stop),
                        "crop_x0": int(crop_slice[1].start),
                        "crop_x1": int(crop_slice[1].stop),
                    }
                )
            marker_decisions = rows_for(decision_examples, roi, marker)
            positive_labels = (
                exemplar_selection.loc[
                    exemplar_selection["marker"].astype(str).eq(marker)
                    & exemplar_selection["roi"].astype(str).eq(roi)
                    & exemplar_selection["selection_origin"].astype(str).eq(
                        "automatic"
                    ),
                    "object_id",
                ]
                .astype(int)
                .tolist()
            )
            for record in marker_decisions.to_dict(orient="records"):
                crop_slice = _crop_slice_for_labels(
                    mask,
                    [int(record["object_id"])],
                    int(
                        math.ceil(
                            max(
                                parameters.max_halo_px,
                                parameters.automatic_same_marker_clearance_px,
                            )
                        )
                    )
                    + crop_margin_px,
                )
                visible_positive_labels = [
                    label
                    for label in positive_labels
                    if np.any(mask[crop_slice] == int(label))
                    and int(label) != int(record["object_id"])
                ]
                decision_crops[marker].append(
                    {
                        "record": record,
                        "raw": image[crop_slice].copy(),
                        "mask": mask[crop_slice].copy(),
                        "positive_labels": visible_positive_labels,
                    }
                )
                obs_index = int(record["source_obs_index"])
                manifest_records.append(
                    {
                        "gallery_type": "automatic_decision",
                        "example_category": str(record["example_category"]),
                        "marker": marker,
                        "target_obs_index": obs_index,
                        "target_cell_id": str(record["source_cell_id"]),
                        "target_roi": roi,
                        "target_segmentation_label": int(record["object_id"]),
                        "target_population": populations[obs_index],
                        "source_obs_index": -1,
                        "source_cell_id": "",
                        "source_roi": "",
                        "source_segmentation_label": -1,
                        "source_population": pd.NA,
                        "neighbour_attributable_fraction": float(
                            scores[obs_index, marker_index]
                        ),
                        "original_x": float(record["input_x_value"]),
                        "classic_intensity": float(
                            classic[obs_index, marker_index]
                        )
                        if classic is not None
                        else float("nan"),
                        "selection_origin": "automatic",
                        "selection_reason": str(record["reason"]),
                        "nearest_same_marker_positive_distance_px": float(
                            record["nearest_same_marker_positive_distance_px"]
                        ),
                        "min_unassigned_pixels_per_bin": int(
                            record["min_unassigned_pixels_per_bin"]
                        ),
                        "crop_y0": int(crop_slice[0].start),
                        "crop_y1": int(crop_slice[0].stop),
                        "crop_x0": int(crop_slice[1].start),
                        "crop_x1": int(crop_slice[1].stop),
                    }
                )

    paths: list[Path] = []
    manifest = pd.DataFrame(manifest_records)
    for marker in markers:
        safe_marker = _gallery_filename(marker)
        if target_crops[marker]:
            path = gallery_dir / f"{safe_marker}_target_source_gallery.png"
            paths.append(_render_target_source_contact_sheet(marker, target_crops[marker], path))
            if not manifest.empty:
                manifest.loc[
                    manifest["marker"].astype(str).eq(marker)
                    & manifest["gallery_type"].eq("target_source"),
                    "figure_path",
                ] = str(path)
        if exemplar_crops[marker]:
            path = gallery_dir / f"{safe_marker}_exemplar_halo_gallery.png"
            paths.append(_render_exemplar_contact_sheet(adata, marker, exemplar_crops[marker], path))
            if not manifest.empty:
                manifest.loc[
                    manifest["marker"].astype(str).eq(marker)
                    & manifest["gallery_type"].eq("exemplar_halo"),
                    "figure_path",
                ] = str(path)
        if decision_crops[marker]:
            path = gallery_dir / f"{safe_marker}_automatic_exemplar_decisions.png"
            paths.append(
                _render_automatic_decision_contact_sheet(
                    marker,
                    decision_crops[marker],
                    path,
                    max_halo_px=parameters.max_halo_px,
                )
            )
            if not manifest.empty:
                manifest.loc[
                    manifest["marker"].astype(str).eq(marker)
                    & manifest["gallery_type"].eq("automatic_decision"),
                    "figure_path",
                ] = str(path)
    if parameters.halo_aggregation != "max" and not target_examples.empty:
        warnings.append(
            "Target-cell galleries were rendered for halo_aggregation='sum', but the "
            "winning-source identity panel is unavailable because summed pixel provenance "
            "is not source-resolved."
        )
    return paths, manifest, warnings


def _plot_score_distributions(
    adata: Any,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write one explicit score-distribution figure for every marker."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    scores: np.ndarray = _dense_matrix(adata.X).astype(float, copy=False)
    summary_by_marker = summary.set_index("marker")
    if adata.n_obs > 50000:
        sampled = np.linspace(0, adata.n_obs - 1, 50000, dtype=int)
    else:
        sampled = np.arange(adata.n_obs)
    paths: list[Path] = []
    bins = np.linspace(0.0, 1.0, 41)
    for marker_index, marker_value in enumerate(adata.var_names.astype(str)):
        marker = str(marker_value)
        all_values = scores[:, marker_index]
        values = all_values[sampled]
        weights = np.full(values.shape, 100.0 / max(1, len(values)), dtype=float)
        row = summary_by_marker.loc[marker]
        profile_available = bool(adata.var.iloc[marker_index]["halo_profile_available"])
        fig, axis = plt.subplots(figsize=(6.4, 4.5))
        axis.hist(
            values,
            bins=bins,
            weights=weights,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.35,
        )
        axis.set_xlim(0, 1)
        axis.set_xlabel("Neighbour-Attributable Fraction")
        axis.set_ylabel("Cells (%)")
        axis.set_title(f"{marker}: cell score distribution")
        for threshold in (0.25, 0.5, 0.75):
            axis.axvline(threshold, color="#888888", linewidth=0.6, linestyle="--")
        status = "profile available" if profile_available else "profile unavailable"
        annotation = (
            f"{status}\nmedian={float(row['median_score']):.3f}; "
            f"p95={float(row['p95_score']):.3f}; "
            f"cells ≥0.5={float(row['fraction_above_0.50']):.1%}"
        )
        if not np.any(all_values > 0):
            if profile_available:
                annotation += "\nAll cell scores are zero: no attributable signal was found."
            else:
                reason = str(adata.var.iloc[marker_index]["halo_skip_reason"])
                annotation += f"\nAll cell scores are zero: {reason}."
        axis.text(
            0.98,
            0.96,
            annotation,
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            wrap=True,
        )
        fig.tight_layout()
        path = output_dir / (
            "neighbour_attributable_score_distribution_"
            f"{_marker_plot_stem(marker_index, marker)}.png"
        )
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_umap(
    adata: Any,
    markers: Sequence[str],
    output_dir: Path,
    *,
    point_size: float | None,
) -> list[Path]:
    """Write one Scanpy UMAP for every marker and cell-level halo summary."""

    import matplotlib.pyplot as plt
    import scanpy as sc

    output_dir.mkdir(parents=True, exist_ok=True)
    marker_list = list(markers)
    colors = [*marker_list, "halo_max_score", "halo_mean_score"]
    paths: list[Path] = []
    for color in colors:
        kwargs: dict[str, Any] = {
            "color": color,
            "cmap": "magma",
            "show": False,
            "return_fig": True,
            "frameon": False,
            "title": f"{color}: neighbour-attributable signal",
        }
        if point_size is not None:
            kwargs["size"] = float(point_size)
        figure = sc.pl.umap(adata, **kwargs)
        if color in marker_list:
            marker_index = marker_list.index(color)
            stem = _marker_plot_stem(marker_index, color)
        else:
            stem = f"summary_{_gallery_filename(color)}"
        path = output_dir / f"scanpy_umap_halo_{stem}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_population_matrix(
    adata: Any,
    markers: Sequence[str],
    population_obs: str,
    output_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Write one natively scaled matrix plot per marker with clustered populations."""

    import matplotlib.pyplot as plt
    import scanpy as sc

    output_dir.mkdir(parents=True, exist_ok=True)
    plotting = adata.copy()
    plotting.obs[population_obs] = (
        plotting.obs[population_obs].astype("category").cat.remove_unused_categories()
    )
    population_count = int(plotting.obs[population_obs].nunique(dropna=True))
    profile_markers = [
        marker
        for marker in markers
        if bool(plotting.var.loc[marker, "halo_profile_available"])
    ]
    dendrogram_markers = profile_markers or list(markers)
    warnings: list[str] = []
    dendrogram_enabled = population_count > 1 and bool(dendrogram_markers)
    if dendrogram_enabled:
        try:
            sc.tl.dendrogram(
                plotting,
                groupby=population_obs,
                var_names=dendrogram_markers,
                use_raw=False,
            )
        except (FloatingPointError, ValueError) as exc:
            dendrogram_enabled = False
            warnings.append(
                "Population dendrogram could not be calculated from neighbour-attributable "
                f"scores ({exc}); population marker plots use categorical order instead."
            )
    else:
        warnings.append(
            "Population dendrogram was skipped because fewer than two populated categories "
            "or no marker features were available."
        )
    paths: list[Path] = []
    for marker_index, marker in enumerate(markers):
        matrix_plot = sc.pl.matrixplot(
            plotting,
            var_names=[marker],
            groupby=population_obs,
            use_raw=False,
            cmap="magma",
            vmin=0,
            colorbar_title="Mean neighbour-attributable fraction",
            dendrogram=dendrogram_enabled,
            show=False,
            return_fig=True,
        )
        path = output_dir / (
            "scanpy_population_marker_halo_matrixplot_"
            f"{_marker_plot_stem(marker_index, marker)}.png"
        )
        matrix_plot.savefig(path, dpi=160, bbox_inches="tight")
        plt.close("all")
        paths.append(path)
    return paths, warnings


def _plot_source_target_population_heatmaps(
    population_summary: pd.DataFrame,
    markers: Sequence[str],
    output_dir: Path,
    *,
    exclude_same_population: bool,
) -> list[Path]:
    """Write one source-population to target-population heatmap per marker."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = population_summary.loc[
        population_summary["marker"].astype(str).isin(markers)
    ].copy()
    if exclude_same_population:
        selected = selected.loc[
            selected["source_population"].astype(str)
            != selected["target_population"].astype(str)
        ]
    paths: list[Path] = []
    for marker_index, marker in enumerate(markers):
        fig, axis = plt.subplots(figsize=(6.4, 5.2))
        marker_frame = selected.loc[selected["marker"].astype(str).eq(marker)]
        if marker_frame.empty:
            axis.axis("off")
            filter_text = (
                " after excluding same-population relationships"
                if exclude_same_population
                else ""
            )
            axis.text(
                0.5,
                0.5,
                f"No source-target population relationships{filter_text}.",
                ha="center",
                va="center",
                transform=axis.transAxes,
                wrap=True,
            )
        else:
            matrix = marker_frame.pivot_table(
                index="source_population",
                columns="target_population",
                values="total_attributable_intensity",
                aggfunc="sum",
                fill_value=0.0,
                observed=True,
            )
            values = matrix.to_numpy(dtype=float)
            image = axis.imshow(values, aspect="auto", cmap="magma")
            axis.set_xticks(np.arange(matrix.shape[1]), matrix.columns.astype(str))
            axis.set_yticks(np.arange(matrix.shape[0]), matrix.index.astype(str))
            axis.tick_params(axis="x", rotation=45, labelsize=8)
            axis.tick_params(axis="y", labelsize=8)
            axis.set_xlabel("Target population")
            axis.set_ylabel("Spatial source population")
            if not exclude_same_population:
                column_positions = {
                    str(value): index for index, value in enumerate(matrix.columns)
                }
                for row_index, source in enumerate(matrix.index.astype(str)):
                    column_index = column_positions.get(source)
                    if column_index is not None:
                        axis.add_patch(
                            Rectangle(
                                (column_index - 0.5, row_index - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="#65c7d0",
                                linewidth=1.5,
                            )
                        )
            colorbar = fig.colorbar(image, ax=axis, shrink=0.75)
            colorbar.set_label("Total attributable intensity")
        title_suffix = " (cross-population only)" if exclude_same_population else ""
        axis.set_title(
            f"{marker}: spatial source population → target population{title_suffix}"
        )
        fig.tight_layout()
        path = output_dir / (
            "source_target_population_heatmap_"
            f"{_marker_plot_stem(marker_index, marker)}.png"
        )
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _matrix_column(matrix: Any, index: int) -> np.ndarray:
    values = matrix[:, index]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).reshape(-1).astype(float, copy=False)


def _plot_expression_comparison(
    adata: Any,
    markers: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    """Write one classic/original-X/halo comparison per marker."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    if adata.n_obs > 20000:
        sampled = np.linspace(0, adata.n_obs - 1, 20000, dtype=int)
    else:
        sampled = np.arange(adata.n_obs)
    paths: list[Path] = []
    for axis_index, marker in enumerate(markers):
        fig, axis = plt.subplots(figsize=(6.4, 5.0))
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
        colorbar = fig.colorbar(scatter, ax=axis, shrink=0.75)
        colorbar.set_label("Neighbour-Attributable Fraction")
        fig.suptitle("Raw mask intensity vs preserved input expression", y=1.01)
        fig.tight_layout()
        output_path = output_dir / (
            "classic_originalX_halo_comparison_"
            f"{_marker_plot_stem(axis_index, marker)}.png"
        )
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)
    return paths


def _write_summary(
    adata: Any,
    score_summary: pd.DataFrame,
    dominant_summary: pd.DataFrame,
    selected_markers: Sequence[str],
    gallery_manifest: pd.DataFrame,
    output_adata_path: Path,
    path: Path,
) -> Path:
    halo = adata.uns["marker_halo"]
    marker_summary = halo["marker_summary"]
    skipped = marker_summary.loc[~marker_summary["halo_profile_available"].astype(bool)]
    usage = halo["worker_usage"]
    source_table = halo["source_target_table"]
    parameters = halo["parameters"]
    exemplar_mode = str(parameters.get("exemplar_mode", "manual"))
    selection_summary = halo["exemplar_selection_summary"]
    automatic_selected = int(selection_summary["automatic_selected"].sum())
    manual_selected = int(selection_summary["manual_selected"].sum())
    mask_only_source_occurrences, mask_only_source_pairs = (
        _mask_only_source_metrics(adata)
    )
    lines = [
        "# Neighbour-attributable signal QC",
        "",
        "The Neighbour-Attributable Fraction is the fraction of background-subtracted raw marker signal inside a cell mask that spatially coincides with an empirical halo projected from a strong neighbouring cell.",
        "",
        "It is a QC/uncertainty score describing spatial explainability. It is not a calibrated probability and does not prove that signal is artefactual.",
        "",
        "The sparse source-target table and dominant-source layers answer the complementary question of which neighbouring cell(s) provide that spatial explanation. A spatial source is a neighbouring cell whose projected marker halo explains observed signal inside the target mask; this wording does not assert physical transfer.",
        "",
        "## Run summary",
        "",
        f"- Output AnnData: `{output_adata_path}`",
        f"- Cells × markers: {adata.n_obs:,} × {adata.n_vars:,}",
        f"- Markers with learned profiles: {int(marker_summary['halo_profile_available'].sum()):,}",
        f"- Skipped markers: {len(skipped):,}",
        f"- ROI workers: {usage['effective']} (limit {usage['cpu_limit']} from {usage['limit_source']})",
        f"- Markers receiving per-marker QC figures: {len(selected_markers):,} "
        "(the complete AnnData marker axis)",
        f"- Cell-gallery panels: {len(gallery_manifest):,}",
        f"- Source-resolved provenance available: {bool(source_table['available'])}",
        f"- Source-target relationships: {int(source_table['relationships']):,}",
        f"- Source-target Parquet: `{source_table['path'] or 'not configured'}`",
        f"- Strong mask-only source occurrences excluded from projection: "
        f"{mask_only_source_occurrences:,} across {mask_only_source_pairs:,} "
        "ROI-marker pairs",
        "",
        "## Exemplar and source definition",
        "",
        f"Exemplar mode: `{exemplar_mode}` ({automatic_selected:,} automatic and {manual_selected:,} manual marker/cell selections).",
        "",
        (
            f"Automatic candidates required input `AnnData.X` marker score ≥ {float(parameters.get('automatic_positive_threshold', 0.5)):.3g}, no other X-positive cell for the same marker within {float(parameters.get('automatic_same_marker_clearance_px', 10.0)):.3g} pixels, and enough unassigned pixels in every halo bin. Other segmented cells were permitted nearby and their pixels were excluded from the radial averages. Eligible cells were sampled reproducibly across ROIs and X-score ranges."
            if exemplar_mode in {"automatic", "augment"}
            else f"Manual exemplars came from `adata.obs[{parameters.get('exemplar_obs', 'Exemplar_stains')!r}]`."
        ),
        "",
        "AnnData.X was used only for automatic exemplar candidate positivity. Each selected cell's halo profile, source strength, background, projected spatial sources, and final score were calculated from raw pixels and masks. The original matrix is preserved in `layers['original_X']`; comparisons involving selected training cells are therefore not independent validation of X positivity.",
        "",
        "Each exemplar profile was background-subtracted and divided by its robust dilated-anchor source strength. Marker source thresholds were derived only from valid selected exemplar source strengths.",
        "",
        "Segmentation labels absent from the input AnnData were retained as occupied geometry. Their pixels remained excluded from exemplar radial measurements, and strong mask-only source neighbourhoods remained excluded from ROI background estimation. They did not project named halos because every reported source must map to an output AnnData row; per-ROI/marker counts are available in `roi_marker_backgrounds.csv`.",
        "",
        "## Cell-based image galleries",
        "",
        "The exemplar galleries show the raw marker, unassigned radial pixels used for learning, and each exemplar curve against the aggregate profile. Target-source galleries show raw signal, observed excess, projected halo, attributable signal, residual excess, and the winning spatial source inside the target. Automatic-decision galleries show representative accepted and rejected X-positive candidates. These are targeted qualitative checks, not proof of physical signal transfer.",
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
    affected_dominant = dominant_summary.loc[
        dominant_summary["affected_target_cells"] > 0
    ].sort_values(
        "fraction_affected_targets_dominant_source_gt_0.5",
        ascending=False,
    )
    if len(affected_dominant):
        lines.extend(
            [
                "",
                "## Dominant spatial sources",
                "",
                "| Marker | Affected targets | Dominant source >50% of attributable signal | Median contributing sources | Common population routes |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for _index, row in affected_dominant.head(10).iterrows():
            lines.append(
                f"| {row['marker']} | {int(row['affected_target_cells'])} | "
                f"{row['fraction_affected_targets_dominant_source_gt_0.5']:.3f} | "
                f"{row['median_contributing_source_cells_per_affected_target']:.2f} | "
                f"{row['most_common_source_target_population_relationships'] or 'not available'} |"
            )
    lines.extend(
        [
            "",
            "## Layer interpretation",
            "",
            "- `classic_intensities`: mean raw intensity inside each segmentation mask (when enabled).",
            "- `neighbour_attributable_intensity`: mean observed excess per mask pixel captured by neighbouring halos.",
            "- `residual_excess_intensity`: mean excess remaining after subtracting the projected halo, clipped at zero.",
            "- `original_X`: input expression/confidence values preserved unchanged; automatic mode uses them only for exemplar candidate positivity.",
            "- `dominant_source_index`: global AnnData row of the largest attributable spatial source; `-1` means none.",
            "- `dominant_source_observed_fraction`: fraction of target observed excess explained by that dominant source.",
            "- `dominant_source_attributable_fraction`: fraction of all neighbour-attributable signal assigned to that dominant source.",
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
    max_qc_markers: int | None,
    umap_point_size: float | None,
    population_obs: str | None,
    source_target_table: pd.DataFrame,
    source_target_qc_exclude_same_population: bool,
    roi_inputs: Sequence[Any] | None = None,
    roi_obs: str = "ROI",
    object_id_obs: str = "ObjectNumber",
    create_cell_galleries: bool = True,
    gallery_examples_per_marker: int = 6,
    gallery_crop_margin_px: int = 8,
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
    dominant_summary = dominant_source_summary(adata, source_target_table)
    dominant_path = tables_dir / "dominant_source_summary.csv"
    dominant_summary.to_csv(dominant_path, index=False)
    report.tables.append(dominant_path)
    population_provenance = population_source_target_summary(source_target_table)
    population_provenance_path = (
        tables_dir / "source_target_population_marker_summary.csv"
    )
    population_provenance.to_csv(population_provenance_path, index=False)
    report.tables.append(population_provenance_path)

    profiles = profile_values_table(adata)
    profile_path = tables_dir / "marker_halo_profiles.csv"
    profiles.to_csv(profile_path, index=False)
    report.tables.append(profile_path)
    marker_metadata_path = tables_dir / "marker_halo_metadata.csv"
    adata.uns["marker_halo"]["marker_summary"].to_csv(marker_metadata_path)
    report.tables.append(marker_metadata_path)
    for key, filename in (
        ("exemplar_statistics", "exemplar_profile_statistics.csv"),
        ("exemplar_profile_values", "exemplar_profile_values.csv"),
        ("exemplar_selection", "exemplar_selection.csv"),
        ("exemplar_selection_summary", "exemplar_selection_summary.csv"),
        ("roi_marker_backgrounds", "roi_marker_backgrounds.csv"),
        ("unknown_exemplar_markers", "unknown_exemplar_markers.csv"),
    ):
        table = adata.uns["marker_halo"][key]
        path = tables_dir / filename
        table.to_csv(path, index=False)
        report.tables.append(path)

    selected, selection_warnings = select_qc_markers(adata, qc_markers, max_qc_markers)
    report.warnings.extend(selection_warnings)
    report.figures.extend(
        _plot_profiles(adata, figures_dir / "marker_halo_profiles")
    )
    report.figures.extend(
        _plot_score_distributions(
            adata,
            summary,
            figures_dir / "score_distributions",
        )
    )
    report.figures.extend(
        _plot_exemplar_selection(
            adata,
            selected,
            figures_dir / "automatic_exemplar_selection",
        )
    )
    gallery_manifest = pd.DataFrame()
    if create_cell_galleries:
        if roi_inputs is None:
            report.warnings.append(
                "Skipped cell-based halo galleries because resolved ROI image inputs were not provided."
            )
        elif not selected:
            report.warnings.append(
                "Skipped cell-based halo galleries because no QC markers were selected."
            )
        else:
            gallery_paths, gallery_manifest, gallery_warnings = (
                generate_cell_qc_galleries(
                    adata,
                    source_target_table,
                    roi_inputs,
                    selected,
                    figures_dir=figures_dir,
                    examples_per_marker=gallery_examples_per_marker,
                    crop_margin_px=gallery_crop_margin_px,
                    roi_obs=roi_obs,
                    object_id_obs=object_id_obs,
                    population_obs=population_obs,
                )
            )
            report.figures.extend(gallery_paths)
            report.warnings.extend(gallery_warnings)
            gallery_manifest_path = tables_dir / "cell_gallery_manifest.csv"
            gallery_manifest.to_csv(gallery_manifest_path, index=False)
            report.tables.append(gallery_manifest_path)
    if not bool(adata.uns["marker_halo"]["source_target_table"]["available"]):
        report.warnings.append(
            "Source-resolved provenance QC is unavailable for halo_aggregation='sum'; "
            "use the recommended 'max' aggregation to identify spatial source cells."
        )

    if "X_umap" in adata.obsm:
        report.figures.extend(
            _plot_umap(
                adata,
                selected,
                figures_dir / "scanpy_umap_halo_scores",
                point_size=umap_point_size,
            )
        )
    elif "X_umap" not in adata.obsm:
        report.warnings.append("Skipped Scanpy UMAP QC because adata.obsm['X_umap'] is absent.")
    if population_obs:
        if population_obs not in adata.obs.columns:
            report.warnings.append(
                f"Skipped population-by-marker QC because adata.obs[{population_obs!r}] is absent."
            )
        elif selected and adata.obs[population_obs].notna().any():
            population_paths, population_warnings = _plot_population_matrix(
                adata,
                selected,
                population_obs,
                figures_dir / "scanpy_population_marker_halo_matrixplots",
            )
            report.figures.extend(population_paths)
            report.warnings.extend(population_warnings)
            report.figures.extend(
                _plot_source_target_population_heatmaps(
                    population_provenance,
                    selected,
                    figures_dir / "source_target_population_marker_heatmaps",
                    exclude_same_population=(
                        source_target_qc_exclude_same_population
                    ),
                )
            )
    else:
        report.warnings.append("Skipped population-by-marker QC because no population observation is configured.")
    if selected and "classic_intensities" in adata.layers:
        report.figures.extend(
            _plot_expression_comparison(
                adata,
                selected,
                figures_dir / "classic_originalX_halo_comparisons",
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
            dominant_summary,
            selected,
            gallery_manifest,
            output_adata_path,
            summary_path,
        )
    )
    mask_only_occurrences, mask_only_pairs = _mask_only_source_metrics(adata)
    report.metrics.update(
        {
            "cells": int(adata.n_obs),
            "markers": int(adata.n_vars),
            "markers_with_profiles": int(adata.var["halo_profile_available"].sum()),
            "skipped_markers": int((~adata.var["halo_profile_available"].astype(bool)).sum()),
            "qc_markers": len(selected),
            "source_target_relationships": int(len(source_target_table)),
            "automatic_exemplars_selected": int(
                adata.uns["marker_halo"]["exemplar_selection_summary"][
                    "automatic_selected"
                ].sum()
            ),
            "manual_exemplars_selected": int(
                adata.uns["marker_halo"]["exemplar_selection_summary"][
                    "manual_selected"
                ].sum()
            ),
            "cell_gallery_panels": int(len(gallery_manifest)),
            "cell_gallery_figures": int(
                sum("cell_galleries" in str(path) for path in report.figures)
            ),
            "unmapped_strong_source_occurrences": mask_only_occurrences,
            "roi_marker_pairs_with_unmapped_strong_sources": mask_only_pairs,
            "affected_target_marker_pairs": int(
                source_target_table[["target_obs_index", "marker"]]
                .drop_duplicates()
                .shape[0]
            ),
        }
    )
    return report


__all__ = [
    "NeighbourSignalReport",
    "dominant_source_summary",
    "generate_cell_qc_galleries",
    "generate_neighbour_signal_report",
    "marker_score_summary",
    "population_source_target_summary",
    "profile_values_table",
    "select_automatic_decision_gallery_examples",
    "select_cell_gallery_examples",
    "select_exemplar_gallery_examples",
    "select_qc_markers",
]
