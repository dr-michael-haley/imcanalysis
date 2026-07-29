#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-reference Leiden visualisation outputs with survival outputs.

This script does not rerun HyPERSTAC, visualisation, or survival models. It
joins the per-Leiden visual marker/permutation summaries with the per-Leiden
survival outputs to help assess whether survival-associated environments are
stable across clustering resolutions.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


CLUSTER_FEATURE_PREFIX = "cluster_freq_"
FIGURE_DPI = 220


@dataclass(frozen=True)
class AnalysisPair:
    cluster_col: str
    visual_dir: Path
    survival_dir: Path


@dataclass(frozen=True)
class HtmlImageSpec:
    section: str
    title: str
    path: Path
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cross-resolution Leiden stability report by joining IMC "
            "visualisation outputs with survival analysis outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visualisation-dir", type=Path, required=True)
    parser.add_argument("--survival-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-markers", type=int, default=5)
    parser.add_argument("--max-heatmap-markers", type=int, default=50)
    parser.add_argument("--max-signature-markers", type=int, default=40)
    parser.add_argument(
        "--environment-distance-threshold",
        type=float,
        default=0.45,
        help=(
            "Hierarchical clustering cut height for grouping Leiden clusters "
            "into recurrent environments. Lower is stricter."
        ),
    )
    parser.add_argument(
        "--marker-enrichment-min-z",
        type=float,
        default=0.0,
        help=(
            "Minimum marker z-score used as a positive enrichment weight when "
            "summarising marker-survival direction."
        ),
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=1e-8,
        help="Absolute coefficient threshold used when calculating sign consistency.",
    )
    parser.add_argument(
        "--permutation-type",
        type=str,
        default="zero_channel",
        choices=["zero_channel", "shuffle_channel"],
        help="Permutation impact family to prioritise in marker summaries.",
    )
    parser.add_argument("--max-report-items", type=int, default=20)
    parser.add_argument(
        "--figure-format",
        type=str,
        default="png",
        help="Figure file format/extension, for example png, pdf, or svg.",
    )
    parser.add_argument("--figure-dpi", type=int, default=220)
    parser.add_argument(
        "--cluster-bubble-metric",
        type=str,
        default="case_prevalence",
        choices=["case_prevalence", "mean_case_frequency", "n_clusters_in_cell"],
        help="Metric used for bubble size in cluster_effect_bubble_plot.",
    )
    parser.add_argument("--cluster-bubble-size-min", type=float, default=35.0)
    parser.add_argument("--cluster-bubble-size-scale", type=float, default=650.0)
    parser.add_argument("--cluster-bubble-log-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--environment-bubble-metric",
        type=str,
        default="n_resolutions",
        choices=["n_clusters", "n_resolutions", "resolution_support_fraction", "mean_case_prevalence"],
        help="Metric used for bubble size in environment_survival_stability.",
    )
    parser.add_argument("--environment-bubble-size-min", type=float, default=35.0)
    parser.add_argument("--environment-bubble-size-scale", type=float, default=10.0)
    parser.add_argument("--environment-bubble-log-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--environment-color-metric",
        type=str,
        default="median_coxnet_coefficient",
        choices=[
            "median_coxnet_coefficient",
            "mean_abs_coxnet_coefficient",
            "coxnet_sign_consistency",
            "median_ridge_cox_coefficient",
            "mean_abs_ridge_cox_coefficient",
            "ridge_cox_sign_consistency",
            "median_image_only_ridge_cox_coefficient",
            "mean_abs_image_only_ridge_cox_coefficient",
            "resolution_support_fraction",
        ],
        help="Metric used for colour in environment_survival_stability.",
    )
    parser.add_argument(
        "--environment-color-quantile",
        type=float,
        default=0.95,
        help="Robust quantile used to scale environment colour limits.",
    )
    parser.add_argument(
        "--per-clustering-html-reports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one embedded-image HTML report per matched Leiden clustering.",
    )
    args = parser.parse_args()

    if args.top_markers <= 0:
        raise ValueError("--top-markers must be positive.")
    if args.max_heatmap_markers <= 0:
        raise ValueError("--max-heatmap-markers must be positive.")
    if args.max_signature_markers <= 0:
        raise ValueError("--max-signature-markers must be positive.")
    if args.environment_distance_threshold <= 0:
        raise ValueError("--environment-distance-threshold must be positive.")
    if args.effect_threshold < 0:
        raise ValueError("--effect-threshold must be non-negative.")
    args.figure_format = args.figure_format.lower().lstrip(".")
    if not args.figure_format:
        raise ValueError("--figure-format must not be empty.")
    if args.figure_dpi <= 0:
        raise ValueError("--figure-dpi must be positive.")
    for name in [
        "cluster_bubble_size_min",
        "cluster_bubble_size_scale",
        "environment_bubble_size_min",
        "environment_bubble_size_scale",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if not 0 < args.environment_color_quantile <= 1:
        raise ValueError("--environment-color-quantile must be in the interval (0, 1].")
    if args.output_dir is None:
        args.output_dir = args.survival_dir.parent / "leiden_stability"
    return args


def safe_filename(value: object) -> str:
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return name or "value"


def figure_path(output_dir: Path, stem: str, args: argparse.Namespace) -> Path:
    return output_dir / f"{stem}.{args.figure_format}"


def save_figure(fig: plt.Figure, output_path: Path, bbox_inches: str | None = None) -> None:
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches=bbox_inches)


def survival_safe_name(value: object) -> str:
    name = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return name or "value"


def natural_key(value: object) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def cluster_sort_key(cluster_col: str) -> tuple[int, object, object, object]:
    resolution = parse_leiden_resolution(cluster_col)
    n_neighbors = parse_leiden_n_neighbors(cluster_col)
    n_pcs = parse_leiden_n_pcs(cluster_col)
    if np.isfinite(resolution):
        return (0, resolution, n_neighbors, n_pcs)
    return (1, natural_key(cluster_col), n_neighbors, n_pcs)


def parse_leiden_resolution(cluster_col: object) -> float:
    value = str(cluster_col)
    match = re.search(r"leiden[_\-. ]*([0-9]+(?:[._][0-9]+)?)", value, flags=re.IGNORECASE)
    if not match:
        return float("nan")
    number = match.group(1).replace("_", ".")
    try:
        return float(number)
    except ValueError:
        return float("nan")


def parse_leiden_n_neighbors(cluster_col: object) -> float:
    match = re.search(r"(?:^|_)N([0-9]+)(?:_|$)", str(cluster_col), flags=re.IGNORECASE)
    if not match:
        return float("nan")
    return float(match.group(1))


def parse_leiden_n_pcs(cluster_col: object) -> float:
    match = re.search(r"(?:^|_)P([0-9]+)(?:_|$)", str(cluster_col), flags=re.IGNORECASE)
    if not match:
        return float("nan")
    return float(match.group(1))


def cluster_parameter_fields(cluster_col: object) -> dict[str, float]:
    return {
        "resolution": parse_leiden_resolution(cluster_col),
        "n_neighbors": parse_leiden_n_neighbors(cluster_col),
        "n_pcs": parse_leiden_n_pcs(cluster_col),
    }


def resolve_summary_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def read_json(path: Path) -> dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}


def read_optional_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def read_indexed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def parse_cluster_col_from_run_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Cluster column:"):
                return line.split(":", 1)[1].strip()
    return None


def discover_visualisation_dirs(visualisation_dir: Path) -> dict[str, Path]:
    summary_path = visualisation_dir / "all_cluster_visualisation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if {"cluster_col", "output_dir"}.issubset(summary.columns):
            return {
                str(row["cluster_col"]): resolve_summary_path(row["output_dir"], visualisation_dir)
                for _, row in summary.iterrows()
            }

    config = read_json(visualisation_dir / "visualisation_run_config.json")
    if config.get("resolved_cluster_col"):
        return {str(config["resolved_cluster_col"]): visualisation_dir}

    discovered: dict[str, Path] = {}
    for child in sorted(visualisation_dir.iterdir()) if visualisation_dir.exists() else []:
        if not child.is_dir():
            continue
        config = read_json(child / "visualisation_run_config.json")
        cluster_col = config.get("resolved_cluster_col")
        if cluster_col and (child / "tables").exists():
            discovered[str(cluster_col)] = child
    return discovered


def discover_survival_dirs(survival_dir: Path) -> dict[str, Path]:
    summary_path = survival_dir / "survival_all_cluster_summary.csv"
    if not summary_path.exists():
        summary_path = survival_dir / "coxnet_all_cluster_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "cluster_col" in summary.columns:
            mapping = {}
            for cluster_col in summary["cluster_col"].astype(str):
                candidate = survival_dir / survival_safe_name(cluster_col)
                mapping[cluster_col] = candidate if candidate.exists() else survival_dir / safe_filename(cluster_col)
            return mapping

    cluster_col = parse_cluster_col_from_run_summary(survival_dir / "run_summary.txt")
    if cluster_col:
        return {cluster_col: survival_dir}

    discovered: dict[str, Path] = {}
    for child in sorted(survival_dir.iterdir()) if survival_dir.exists() else []:
        if not child.is_dir():
            continue
        cluster_col = parse_cluster_col_from_run_summary(child / "run_summary.txt")
        if cluster_col:
            discovered[cluster_col] = child
    return discovered


def pair_analysis_dirs(visualisation_dir: Path, survival_dir: Path) -> tuple[list[AnalysisPair], list[str]]:
    warnings_out: list[str] = []
    visual_dirs = discover_visualisation_dirs(visualisation_dir)
    survival_dirs = discover_survival_dirs(survival_dir)

    pairs = [
        AnalysisPair(cluster_col=cluster_col, visual_dir=visual_dirs[cluster_col], survival_dir=survival_dirs[cluster_col])
        for cluster_col in sorted(set(visual_dirs).intersection(survival_dirs), key=cluster_sort_key)
    ]

    if not pairs and len(visual_dirs) == 1 and len(survival_dirs) == 1:
        visual_cluster_col, visual_path = next(iter(visual_dirs.items()))
        survival_cluster_col, survival_path = next(iter(survival_dirs.items()))
        warnings_out.append(
            "Visualisation and survival cluster column names did not match, "
            f"but each side had one analysis. Pairing {visual_cluster_col} with {survival_cluster_col}."
        )
        pairs = [AnalysisPair(visual_cluster_col, visual_path, survival_path)]

    missing_visual = sorted(set(survival_dirs).difference(visual_dirs), key=cluster_sort_key)
    missing_survival = sorted(set(visual_dirs).difference(survival_dirs), key=cluster_sort_key)
    if missing_visual:
        warnings_out.append(f"No matching visualisation folder for survival cluster columns: {missing_visual}")
    if missing_survival:
        warnings_out.append(f"No matching survival folder for visualisation cluster columns: {missing_survival}")
    return pairs, warnings_out


def feature_to_cluster(feature: object) -> str | None:
    feature = str(feature)
    if not feature.startswith(CLUSTER_FEATURE_PREFIX):
        return None
    return feature[len(CLUSTER_FEATURE_PREFIX):]


def coefficient_sign(value: float, threshold: float) -> str:
    if not np.isfinite(value) or abs(value) <= threshold:
        return "none"
    return "higher_hazard" if value > 0 else "lower_hazard"


def top_markers_from_series(values: pd.Series, n: int) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return ""
    positive = clean[clean > 0]
    source = positive if not positive.empty else clean
    return ", ".join(source.sort_values(ascending=False).head(n).index.astype(str))


def de_score_tables(de: pd.DataFrame) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    if de.empty or not {"group", "names"}.issubset(de.columns):
        return {}, {}
    score_column = "scores" if "scores" in de.columns else None
    score_lookup: dict[tuple[str, str], float] = {}
    rank_lookup: dict[tuple[str, str], int] = {}

    for group, group_df in de.groupby(de["group"].astype(str), sort=False):
        if score_column is not None:
            group_df = group_df.sort_values(score_column, ascending=False, na_position="last")
        for rank, (_, row) in enumerate(group_df.iterrows(), start=1):
            key = (str(group), str(row["names"]))
            rank_lookup[key] = rank
            score_lookup[key] = float(row[score_column]) if score_column is not None and pd.notna(row[score_column]) else np.nan
    return score_lookup, rank_lookup


def split_permutation_columns(perm: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if perm.empty:
        return {}
    result: dict[str, pd.DataFrame] = {}
    for perturbation_type in ["zero_channel", "shuffle_channel"]:
        cols = [col for col in perm.columns if str(col).startswith(f"{perturbation_type}__")]
        if not cols:
            continue
        subset = perm[cols].copy()
        subset.columns = [str(col).split("__", 1)[1] for col in cols]
        result[perturbation_type] = subset
    return result


def read_visual_data(pair: AnalysisPair, top_markers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    tables_dir = pair.visual_dir / "tables"
    zscore = read_indexed_csv(tables_dir / "cluster_mean_channel_intensity_zscore.csv")
    intensity = read_indexed_csv(tables_dir / "cluster_mean_channel_intensity.csv")
    if zscore.empty and not intensity.empty:
        zscore = (intensity - intensity.mean(axis=0)) / intensity.std(axis=0).replace(0, np.nan)
        zscore = zscore.fillna(0.0)

    de_intensity = read_optional_csv(tables_dir / "cluster_marker_differential_intensity.csv")
    intensity_score, intensity_rank = de_score_tables(de_intensity)

    de_zero = read_optional_csv(tables_dir / "cluster_marker_differential_zero_channel_impact.csv")
    zero_score, zero_rank = de_score_tables(de_zero)
    de_shuffle = read_optional_csv(tables_dir / "cluster_marker_differential_shuffle_channel_impact.csv")
    shuffle_score, shuffle_rank = de_score_tables(de_shuffle)

    perm = read_indexed_csv(tables_dir / "cluster_mean_permutation_cosine_distance.csv")
    perm_by_type = split_permutation_columns(perm)

    gallery = read_optional_csv(tables_dir / "patch_gallery_index.csv")
    gallery_markers: dict[str, set[str]] = {}
    if not gallery.empty and {"cluster", "markers"}.issubset(gallery.columns):
        for cluster, group in gallery.groupby(gallery["cluster"].astype(str)):
            markers: set[str] = set()
            for value in group["markers"].dropna().astype(str):
                markers.update(marker.strip() for marker in value.split(",") if marker.strip())
            gallery_markers[cluster] = markers

    clusters = sorted(
        set(zscore.index.astype(str))
        .union(intensity.index.astype(str))
        .union(*(df.index.astype(str) for df in perm_by_type.values())),
        key=natural_key,
    )
    markers = sorted(set(zscore.columns.astype(str)).union(intensity.columns.astype(str)), key=natural_key)
    for df in perm_by_type.values():
        markers = sorted(set(markers).union(df.columns.astype(str)), key=natural_key)

    cluster_rows = []
    marker_rows = []
    gallery_paths: dict[str, str] = {}
    for cluster in clusters:
        zrow = zscore.loc[cluster] if cluster in zscore.index else pd.Series(dtype=float)
        irow = intensity.loc[cluster] if cluster in intensity.index else pd.Series(dtype=float)
        top_intensity = top_markers_from_series(zrow, top_markers)
        gallery_path = pair.visual_dir / "galleries" / f"cluster_{safe_filename(cluster)}__patch_gallery.png"
        gallery_paths[cluster] = str(gallery_path) if gallery_path.exists() else ""
        cluster_rows.append(
            {
                "cluster_col": pair.cluster_col,
                **cluster_parameter_fields(pair.cluster_col),
                "cluster": cluster,
                "cluster_uid": f"{pair.cluster_col}::{cluster}",
                "visualisation_dir": str(pair.visual_dir),
                "top_intensity_markers": top_intensity,
                "gallery_path": gallery_paths[cluster],
            }
        )

        for marker in markers:
            key = (cluster, marker)
            row = {
                "cluster_col": pair.cluster_col,
                **cluster_parameter_fields(pair.cluster_col),
                "cluster": cluster,
                "cluster_uid": f"{pair.cluster_col}::{cluster}",
                "marker": marker,
                "intensity_zscore": float(zrow.get(marker, np.nan)),
                "mean_intensity": float(irow.get(marker, np.nan)),
                "differential_intensity_score": intensity_score.get(key, np.nan),
                "differential_intensity_rank": intensity_rank.get(key, np.nan),
                "zero_channel_differential_score": zero_score.get(key, np.nan),
                "zero_channel_differential_rank": zero_rank.get(key, np.nan),
                "shuffle_channel_differential_score": shuffle_score.get(key, np.nan),
                "shuffle_channel_differential_rank": shuffle_rank.get(key, np.nan),
                "gallery_marker": marker in gallery_markers.get(cluster, set()),
            }
            for perturbation_type, df in perm_by_type.items():
                row[f"{perturbation_type}_impact"] = (
                    float(df.loc[cluster, marker]) if cluster in df.index and marker in df.columns else np.nan
                )
            marker_rows.append(row)

    cluster_df = pd.DataFrame(cluster_rows)
    marker_df = pd.DataFrame(marker_rows)
    return cluster_df, marker_df, zscore, gallery_paths


def read_cluster_coefficients(path: Path, prefix: str) -> pd.DataFrame:
    df = read_optional_csv(path)
    columns = ["cluster", f"{prefix}_coefficient", f"{prefix}_abs_coefficient", f"{prefix}_nonzero", f"{prefix}_alpha"]
    if df.empty or "feature" not in df.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, row in df.iterrows():
        cluster = feature_to_cluster(row["feature"])
        if cluster is None:
            continue
        coefficient = float(row["coefficient"]) if "coefficient" in df.columns and pd.notna(row["coefficient"]) else np.nan
        rows.append(
            {
                "cluster": cluster,
                f"{prefix}_coefficient": coefficient,
                f"{prefix}_abs_coefficient": abs(coefficient) if np.isfinite(coefficient) else np.nan,
                f"{prefix}_nonzero": bool(row["nonzero"]) if "nonzero" in df.columns and pd.notna(row["nonzero"]) else np.nan,
                f"{prefix}_alpha": float(row["alpha"]) if "alpha" in df.columns and pd.notna(row["alpha"]) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def read_standard_cox(path: Path, prefix: str = "standard_cox") -> pd.DataFrame:
    df = read_optional_csv(path)
    columns = ["cluster", f"{prefix}_coef", f"{prefix}_hr", f"{prefix}_p"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    if "feature" not in df.columns:
        if "covariate" in df.columns:
            df = df.rename(columns={"covariate": "feature"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "feature"})
        else:
            return pd.DataFrame(columns=columns)

    rows = []
    for _, row in df.iterrows():
        cluster = feature_to_cluster(row["feature"])
        if cluster is None:
            continue
        rows.append(
            {
                "cluster": cluster,
                f"{prefix}_coef": float(row["coef"]) if "coef" in df.columns and pd.notna(row["coef"]) else np.nan,
                f"{prefix}_hr": float(row["exp(coef)"]) if "exp(coef)" in df.columns and pd.notna(row["exp(coef)"]) else np.nan,
                f"{prefix}_p": float(row["p"]) if "p" in df.columns and pd.notna(row["p"]) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def read_feature_selection(path: Path) -> pd.DataFrame:
    df = read_optional_csv(path)
    columns = ["cluster", "univariate_coef", "univariate_mean_cv_c_index", "univariate_rank"]
    if df.empty or "feature" not in df.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        cluster = feature_to_cluster(row["feature"])
        if cluster is None:
            continue
        rows.append(
            {
                "cluster": cluster,
                "univariate_coef": float(row["univariate_coef"]) if "univariate_coef" in df.columns and pd.notna(row["univariate_coef"]) else np.nan,
                "univariate_mean_cv_c_index": (
                    float(row["mean_cv_c_index"]) if "mean_cv_c_index" in df.columns and pd.notna(row["mean_cv_c_index"]) else np.nan
                ),
                "univariate_rank": rank,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def read_case_feature_support(path: Path) -> pd.DataFrame:
    df = read_optional_csv(path, index_col=0)
    columns = ["cluster", "mean_case_frequency", "median_case_frequency", "max_case_frequency", "case_prevalence"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for column in df.columns:
        cluster = feature_to_cluster(column)
        if cluster is None:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "cluster": cluster,
                "mean_case_frequency": float(values.mean()),
                "median_case_frequency": float(values.median()),
                "max_case_frequency": float(values.max()),
                "case_prevalence": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def read_survival_data(pair: AnalysisPair) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [
        read_cluster_coefficients(pair.survival_dir / "coxnet_coefficients.csv", "coxnet"),
        read_cluster_coefficients(pair.survival_dir / "coxnet_image_only" / "coxnet_coefficients.csv", "image_only_coxnet"),
        read_cluster_coefficients(pair.survival_dir / "ridge_cox_coefficients.csv", "ridge_cox"),
        read_cluster_coefficients(pair.survival_dir / "ridge_cox_image_only" / "ridge_cox_coefficients.csv", "image_only_ridge_cox"),
        read_standard_cox(pair.survival_dir / "standard_cox_results.csv"),
        read_standard_cox(pair.survival_dir / "ridge_selected_standard_cox_results.csv", "ridge_standard_cox"),
        read_feature_selection(pair.survival_dir / "feature_selection.csv"),
        read_case_feature_support(pair.survival_dir / "case_features.csv"),
    ]

    merged = pd.DataFrame({"cluster": sorted(set().union(*(set(part["cluster"].astype(str)) for part in parts if "cluster" in part.columns)), key=natural_key)})
    for part in parts:
        if not part.empty:
            part = part.copy()
            part["cluster"] = part["cluster"].astype(str)
            merged = merged.merge(part, on="cluster", how="left")
    merged["cluster_col"] = pair.cluster_col
    for key, value in cluster_parameter_fields(pair.cluster_col).items():
        merged[key] = value
    merged["cluster_uid"] = merged["cluster_col"].astype(str) + "::" + merged["cluster"].astype(str)
    merged["survival_dir"] = str(pair.survival_dir)

    validation = read_optional_csv(pair.survival_dir / "survival_model_validation_summary.csv")
    if not validation.empty:
        validation.insert(0, "cluster_col", pair.cluster_col)
        insert_at = 1
        for key, value in cluster_parameter_fields(pair.cluster_col).items():
            validation.insert(insert_at, key, value)
            insert_at += 1
        validation.insert(insert_at, "survival_dir", str(pair.survival_dir))
    return merged, validation


def merge_pair_data(pair: AnalysisPair, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    visual_clusters, marker_rows, signature, _gallery_paths = read_visual_data(pair, args.top_markers)
    survival_clusters, validation = read_survival_data(pair)

    cluster_df = visual_clusters.merge(
        survival_clusters.drop(columns=["cluster_col", "resolution", "n_neighbors", "n_pcs", "cluster_uid"], errors="ignore"),
        on="cluster",
        how="outer",
    )
    cluster_df["cluster_col"] = cluster_df["cluster_col"].fillna(pair.cluster_col)
    for key, value in cluster_parameter_fields(pair.cluster_col).items():
        cluster_df[key] = cluster_df[key].fillna(value)
    cluster_df["cluster_uid"] = cluster_df["cluster_col"].astype(str) + "::" + cluster_df["cluster"].astype(str)
    cluster_df["coxnet_direction"] = cluster_df["coxnet_coefficient"].apply(lambda value: coefficient_sign(value, args.effect_threshold))
    cluster_df["image_only_coxnet_direction"] = cluster_df.get("image_only_coxnet_coefficient", pd.Series(index=cluster_df.index, dtype=float)).apply(
        lambda value: coefficient_sign(value, args.effect_threshold)
    )
    cluster_df["ridge_cox_direction"] = cluster_df.get("ridge_cox_coefficient", pd.Series(index=cluster_df.index, dtype=float)).apply(
        lambda value: coefficient_sign(value, args.effect_threshold)
    )
    cluster_df["image_only_ridge_cox_direction"] = cluster_df.get(
        "image_only_ridge_cox_coefficient",
        pd.Series(index=cluster_df.index, dtype=float),
    ).apply(lambda value: coefficient_sign(value, args.effect_threshold))
    cluster_df["standard_cox_direction"] = cluster_df.get("standard_cox_coef", pd.Series(index=cluster_df.index, dtype=float)).apply(
        lambda value: coefficient_sign(value, args.effect_threshold)
    )

    survival_columns = [
        "cluster_uid",
        "coxnet_coefficient",
        "coxnet_nonzero",
        "image_only_coxnet_coefficient",
        "image_only_coxnet_nonzero",
        "ridge_cox_coefficient",
        "ridge_cox_nonzero",
        "image_only_ridge_cox_coefficient",
        "image_only_ridge_cox_nonzero",
        "standard_cox_coef",
        "standard_cox_p",
        "ridge_standard_cox_coef",
        "ridge_standard_cox_p",
        "univariate_coef",
        "univariate_mean_cv_c_index",
        "case_prevalence",
        "mean_case_frequency",
    ]
    available_survival_columns = [column for column in survival_columns if column in cluster_df.columns]
    marker_df = marker_rows.merge(cluster_df[available_survival_columns], on="cluster_uid", how="left")
    marker_df["positive_marker_weight"] = (
        pd.to_numeric(marker_df["intensity_zscore"], errors="coerce") - args.marker_enrichment_min_z
    ).clip(lower=0.0)
    for coefficient_column, output_column in [
        ("coxnet_coefficient", "signed_marker_coxnet_score"),
        ("image_only_coxnet_coefficient", "signed_marker_image_only_coxnet_score"),
        ("ridge_cox_coefficient", "signed_marker_ridge_cox_score"),
        ("image_only_ridge_cox_coefficient", "signed_marker_image_only_ridge_cox_score"),
        ("standard_cox_coef", "signed_marker_standard_cox_score"),
        ("ridge_standard_cox_coef", "signed_marker_ridge_standard_cox_score"),
        ("univariate_coef", "signed_marker_univariate_score"),
    ]:
        if coefficient_column in marker_df.columns:
            marker_df[output_column] = marker_df["positive_marker_weight"] * pd.to_numeric(marker_df[coefficient_column], errors="coerce")
        else:
            marker_df[output_column] = np.nan

    return cluster_df, marker_df, signature, validation


def assign_environments(signature: pd.DataFrame, threshold: float) -> tuple[pd.Series, list[str]]:
    warnings_out: list[str] = []
    if signature.empty:
        return pd.Series(dtype=str, name="environment_id"), ["No marker signature matrix was available for environment clustering."]

    matrix = signature.fillna(0.0).copy()
    if matrix.shape[0] < 2:
        return pd.Series(["E001"], index=matrix.index, name="environment_id"), warnings_out

    try:
        from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
        from scipy.spatial.distance import pdist
    except ImportError:
        warnings_out.append("scipy is not available; assigning one environment per cluster.")
        labels = np.arange(1, matrix.shape[0] + 1)
    else:
        distances = pdist(matrix.to_numpy(dtype=float), metric="correlation")
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0)
        if np.allclose(distances, 0):
            labels = np.ones(matrix.shape[0], dtype=int)
        else:
            linkage_matrix = linkage(distances, method="average")
            labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

            # Re-label environments in dendrogram order for more readable plots.
            ordered_raw = []
            for raw_label in labels[leaves_list(linkage_matrix)]:
                if raw_label not in ordered_raw:
                    ordered_raw.append(raw_label)
            remap = {raw_label: f"E{idx + 1:03d}" for idx, raw_label in enumerate(ordered_raw)}
            return pd.Series([remap[label] for label in labels], index=matrix.index, name="environment_id"), warnings_out

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    remap = {label: f"E{idx + 1:03d}" for idx, label in enumerate(unique_labels)}
    return pd.Series([remap[label] for label in labels], index=matrix.index, name="environment_id"), warnings_out


def build_global_signature(signatures: list[pd.DataFrame], cluster_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for signature, cluster_df in zip(signatures, cluster_frames):
        if signature.empty:
            continue
        uid_by_cluster = cluster_df.set_index("cluster")["cluster_uid"].to_dict()
        renamed = signature.copy()
        renamed.index = [uid_by_cluster.get(str(index), f"unknown::{index}") for index in renamed.index.astype(str)]
        rows.append(renamed)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, axis=0, join="outer").fillna(0.0)
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    return combined


def sign_consistency(values: pd.Series, threshold: float) -> tuple[str, float, int, int, int]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite[finite.abs() > threshold]
    if finite.empty:
        return "none", np.nan, 0, 0, 0
    n_positive = int((finite > 0).sum())
    n_negative = int((finite < 0).sum())
    dominant = "higher_hazard" if n_positive >= n_negative else "lower_hazard"
    consistency = max(n_positive, n_negative) / len(finite)
    return dominant, float(consistency), n_positive, n_negative, int(len(finite))


def environment_model_metrics(
    group: pd.DataFrame,
    coefficient_column: str,
    prefix: str,
    threshold: float,
) -> dict[str, object]:
    values = pd.to_numeric(group.get(coefficient_column, pd.Series(dtype=float)), errors="coerce")
    direction, consistency, n_pos, n_neg, n_effect = sign_consistency(values, threshold)
    return {
        f"median_{prefix}_coefficient": float(values.median()) if not values.dropna().empty else np.nan,
        f"mean_abs_{prefix}_coefficient": float(values.abs().mean()) if not values.dropna().empty else np.nan,
        f"dominant_{prefix}_direction": direction,
        f"{prefix}_sign_consistency": consistency,
        f"{prefix}_n_higher_hazard_clusters": n_pos,
        f"{prefix}_n_lower_hazard_clusters": n_neg,
        f"{prefix}_n_effect_clusters": n_effect,
    }


def summarise_environments(
    cluster_summary: pd.DataFrame,
    marker_table: pd.DataFrame,
    signature: pd.DataFrame,
    total_resolutions: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for environment_id, group in cluster_summary.groupby("environment_id", sort=True):
        cluster_uids = group["cluster_uid"].astype(str).tolist()
        signature_subset = signature.reindex(cluster_uids).dropna(axis=1, how="all")
        mean_signature = signature_subset.mean(axis=0) if not signature_subset.empty else pd.Series(dtype=float)
        top_markers = top_markers_from_series(mean_signature, args.top_markers)

        marker_subset = marker_table[marker_table["cluster_uid"].isin(cluster_uids)]
        permutation_column = f"{args.permutation_type}_impact"
        if permutation_column in marker_subset.columns:
            permutation_means = marker_subset.groupby("marker")[permutation_column].mean()
            top_permutation_markers = top_markers_from_series(permutation_means, args.top_markers)
        else:
            top_permutation_markers = ""

        model_metrics: dict[str, object] = {}
        model_metrics.update(environment_model_metrics(group, "coxnet_coefficient", "coxnet", args.effect_threshold))
        model_metrics.update(
            environment_model_metrics(group, "image_only_coxnet_coefficient", "image_only_coxnet", args.effect_threshold)
        )
        model_metrics.update(environment_model_metrics(group, "ridge_cox_coefficient", "ridge_cox", args.effect_threshold))
        model_metrics.update(
            environment_model_metrics(group, "image_only_ridge_cox_coefficient", "image_only_ridge_cox", args.effect_threshold)
        )
        row = {
            "environment_id": environment_id,
            "n_clusters": int(group.shape[0]),
            "n_resolutions": int(group["cluster_col"].nunique()),
            "resolution_support_fraction": float(group["cluster_col"].nunique() / total_resolutions) if total_resolutions else np.nan,
            "cluster_cols": ", ".join(sorted(group["cluster_col"].astype(str).unique(), key=cluster_sort_key)),
            "clusters": ", ".join(f"{row.cluster_col}:{row.cluster}" for row in group.itertuples()),
            "top_intensity_markers": top_markers,
            "top_permutation_markers": top_permutation_markers,
            "mean_case_prevalence": float(pd.to_numeric(group.get("case_prevalence"), errors="coerce").mean())
            if "case_prevalence" in group
            else np.nan,
            "max_case_frequency": float(pd.to_numeric(group.get("max_case_frequency"), errors="coerce").max())
            if "max_case_frequency" in group
            else np.nan,
            **model_metrics,
        }
        row.update(
            {
                "n_higher_hazard_clusters": row["coxnet_n_higher_hazard_clusters"],
                "n_lower_hazard_clusters": row["coxnet_n_lower_hazard_clusters"],
                "n_nonzero_effect_clusters": row["coxnet_n_effect_clusters"],
                "image_only_dominant_direction": row["dominant_image_only_coxnet_direction"],
                "image_only_sign_consistency": row["image_only_coxnet_sign_consistency"],
                "image_only_n_higher_hazard_clusters": row["image_only_coxnet_n_higher_hazard_clusters"],
                "image_only_n_lower_hazard_clusters": row["image_only_coxnet_n_lower_hazard_clusters"],
                "image_only_n_effect_clusters": row["image_only_coxnet_n_effect_clusters"],
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["resolution_support_fraction", "coxnet_sign_consistency", "mean_abs_coxnet_coefficient"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def summarise_marker_stability(marker_table: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (cluster_col, marker), group in marker_table.groupby(["cluster_col", "marker"], sort=False):
        weights = pd.to_numeric(group["positive_marker_weight"], errors="coerce").fillna(0.0)
        row = {
            "cluster_col": cluster_col,
            **cluster_parameter_fields(cluster_col),
            "marker": marker,
            "n_enriched_clusters": int((weights > 0).sum()),
            "mean_positive_zscore": float(pd.to_numeric(group.loc[weights > 0, "intensity_zscore"], errors="coerce").mean()) if (weights > 0).any() else np.nan,
        }
        for score_column in [
            "signed_marker_coxnet_score",
            "signed_marker_image_only_coxnet_score",
            "signed_marker_ridge_cox_score",
            "signed_marker_image_only_ridge_cox_score",
            "signed_marker_standard_cox_score",
            "signed_marker_ridge_standard_cox_score",
            "signed_marker_univariate_score",
        ]:
            values = pd.to_numeric(group[score_column], errors="coerce") if score_column in group else pd.Series(dtype=float)
            if weights.sum() > 0 and not values.dropna().empty:
                row[score_column.replace("signed_marker_", "weighted_")] = float(values.sum() / weights.sum())
            else:
                row[score_column.replace("signed_marker_", "weighted_")] = np.nan

        for permutation_type in ["zero_channel", "shuffle_channel"]:
            column = f"{permutation_type}_impact"
            if column in group.columns and weights.sum() > 0:
                impacts = pd.to_numeric(group[column], errors="coerce")
                row[f"weighted_{permutation_type}_impact"] = float((impacts * weights).sum() / weights.sum())
            else:
                row[f"weighted_{permutation_type}_impact"] = np.nan
        rows.append(row)

    by_resolution = pd.DataFrame(rows)
    if by_resolution.empty:
        return by_resolution, pd.DataFrame()

    summary_rows = []
    for marker, group in by_resolution.groupby("marker", sort=False):
        row = {
            "marker": marker,
            "n_enriched_resolution_pairs": int((group["n_enriched_clusters"] > 0).sum()),
            f"mean_{args.permutation_type}_impact": float(pd.to_numeric(group[f"weighted_{args.permutation_type}_impact"], errors="coerce").mean())
            if f"weighted_{args.permutation_type}_impact" in group
            else np.nan,
        }
        for prefix, score_column in [
            ("coxnet", "weighted_coxnet_score"),
            ("image_only_coxnet", "weighted_image_only_coxnet_score"),
            ("ridge_cox", "weighted_ridge_cox_score"),
            ("image_only_ridge_cox", "weighted_image_only_ridge_cox_score"),
            ("standard_cox", "weighted_standard_cox_score"),
            ("ridge_standard_cox", "weighted_ridge_standard_cox_score"),
            ("univariate", "weighted_univariate_score"),
        ]:
            values = pd.to_numeric(group.get(score_column, pd.Series(dtype=float)), errors="coerce")
            direction, consistency, n_pos, n_neg, n_effect = sign_consistency(values, args.effect_threshold)
            row[f"median_signed_{prefix}_score"] = float(values.median()) if not values.dropna().empty else np.nan
            row[f"mean_abs_signed_{prefix}_score"] = float(values.abs().mean()) if not values.dropna().empty else np.nan
            row[f"dominant_{prefix}_direction"] = direction
            row[f"{prefix}_sign_consistency"] = consistency
            row[f"{prefix}_n_higher_hazard_resolutions"] = n_pos
            row[f"{prefix}_n_lower_hazard_resolutions"] = n_neg
            row[f"{prefix}_n_effect_resolutions"] = n_effect

        row.update(
            {
                "n_resolutions": int(pd.to_numeric(group.get("weighted_coxnet_score", pd.Series(dtype=float)), errors="coerce").notna().sum()),
                "median_signed_coxnet_score": row["median_signed_coxnet_score"],
                "mean_abs_signed_coxnet_score": row["mean_abs_signed_coxnet_score"],
                "dominant_coxnet_direction": row["dominant_coxnet_direction"],
                "coxnet_sign_consistency": row["coxnet_sign_consistency"],
                "n_higher_hazard_resolutions": row["coxnet_n_higher_hazard_resolutions"],
                "n_lower_hazard_resolutions": row["coxnet_n_lower_hazard_resolutions"],
                "n_effect_resolutions": row["coxnet_n_effect_resolutions"],
                "median_image_only_signed_score": row["median_signed_image_only_coxnet_score"],
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return by_resolution, summary.sort_values(
        ["n_effect_resolutions", "coxnet_sign_consistency", "mean_abs_signed_coxnet_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def make_no_data_plot(output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def categorical_palette(categories: list[str]) -> dict[str, object]:
    categories = [str(category) for category in categories]
    n_categories = len(categories)
    palette = None
    try:
        import scanpy as sc

        for palette_name in ("godsnot_64", "default_102", "vega_20_scanpy"):
            candidate = getattr(sc.pl.palettes, palette_name, None)
            if candidate is not None and len(candidate) >= n_categories:
                palette = list(candidate)
                break
    except Exception:
        palette = None

    if palette is None or len(palette) < n_categories:
        palette = sns.color_palette("husl", n_categories).as_hex()
    return {category: palette[idx % len(palette)] for idx, category in enumerate(categories)}


def move_legend_outside(ax: plt.Axes, n_items: int, title: str | None = None) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    n_cols = max(1, int(math.ceil(max(n_items, 1) / 24)))
    try:
        sns.move_legend(
            ax,
            "upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=True,
            ncol=n_cols,
            fontsize=7,
            title=title,
        )
    except Exception:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=True,
            ncol=n_cols,
            fontsize=7,
            title=title,
        )


def environment_label_lookup(environment_summary: pd.DataFrame) -> dict[str, str]:
    if environment_summary.empty or "environment_id" not in environment_summary.columns:
        return {}
    labels = {}
    for row in environment_summary.itertuples():
        markers = getattr(row, "top_intensity_markers", "")
        labels[str(row.environment_id)] = f"{row.environment_id}: {markers}"[:95]
    return labels


METRIC_LABELS = {
    "case_prevalence": "fraction of cases with any cluster in cell",
    "mean_case_frequency": "mean case frequency summed across cell clusters",
    "n_clusters_in_cell": "number of clusters in cell",
    "n_clusters": "number of clusters in environment",
    "n_resolutions": "number of Leiden settings supporting environment",
    "resolution_support_fraction": "fraction of Leiden settings supporting environment",
    "mean_case_prevalence": "mean case prevalence across environment clusters",
    "median_coxnet_coefficient": "median CoxNet coefficient",
    "mean_abs_coxnet_coefficient": "mean absolute CoxNet coefficient",
    "coxnet_sign_consistency": "CoxNet sign consistency",
    "median_ridge_cox_coefficient": "median Ridge Cox coefficient",
    "mean_abs_ridge_cox_coefficient": "mean absolute Ridge Cox coefficient",
    "ridge_cox_sign_consistency": "Ridge Cox sign consistency",
    "median_image_only_ridge_cox_coefficient": "median image-only Ridge Cox coefficient",
    "mean_abs_image_only_ridge_cox_coefficient": "mean absolute image-only Ridge Cox coefficient",
}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def scale_bubble_values(values: pd.Series, minimum: float, scale: float, log_scale: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    numeric = numeric.clip(lower=0.0)
    transformed = np.log1p(numeric) if log_scale else numeric
    return minimum + scale * transformed


def format_metric_value(value: float, metric: str) -> str:
    if pd.isna(value):
        return "NA"
    if metric.endswith("fraction") or metric == "case_prevalence" or metric == "mean_case_prevalence":
        return f"{100 * float(value):.0f}%"
    if abs(float(value) - round(float(value))) < 1e-8:
        return str(int(round(float(value))))
    return f"{float(value):.3g}"


def metric_legend_values(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    numeric = numeric[numeric > 0]
    if numeric.empty:
        return [1.0]
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if np.isclose(min_value, max_value):
        return [max_value]
    return [min_value, max_value]


def colour_limits(values: pd.Series, metric: str, quantile: float) -> tuple[float, float, str, str]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return -1.0, 1.0, "coolwarm", metric_label(metric)
    if metric in {
        "median_coxnet_coefficient",
        "median_ridge_cox_coefficient",
        "median_image_only_ridge_cox_coefficient",
    }:
        abs_max = float(numeric.abs().quantile(quantile))
        if not np.isfinite(abs_max) or abs_max <= 0:
            abs_max = float(numeric.abs().max())
        if not np.isfinite(abs_max) or abs_max <= 0:
            abs_max = 1.0
        return -abs_max, abs_max, "coolwarm", f"{metric_label(metric)} (>0 = higher hazard)"
    if metric in {
        "mean_abs_coxnet_coefficient",
        "mean_abs_ridge_cox_coefficient",
        "mean_abs_image_only_ridge_cox_coefficient",
    }:
        high = float(numeric.quantile(quantile))
        if not np.isfinite(high) or high <= 0:
            high = float(numeric.max()) if np.isfinite(numeric.max()) else 1.0
        return 0.0, high if high > 0 else 1.0, "viridis", metric_label(metric)
    low = float(numeric.quantile(1 - quantile))
    high = float(numeric.quantile(quantile))
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high) or high <= low:
        high = max(float(numeric.max()), low + 1.0)
    return low, high, "viridis", metric_label(metric)


def read_case_features_cached(survival_dir: Path, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    path = survival_dir / "case_features.csv"
    if path not in cache:
        cache[path] = read_optional_csv(path, index_col=0)
    return cache[path]


def aggregate_case_support(group: pd.DataFrame, cache: dict[Path, pd.DataFrame]) -> tuple[float, float]:
    survival_dirs = [Path(value) for value in group.get("survival_dir", pd.Series(dtype=str)).dropna().astype(str).unique()]
    if survival_dirs:
        case_features = read_case_features_cached(survival_dirs[0], cache)
        cluster_columns = [
            f"{CLUSTER_FEATURE_PREFIX}{cluster}"
            for cluster in group["cluster"].astype(str)
            if f"{CLUSTER_FEATURE_PREFIX}{cluster}" in case_features.columns
        ]
        if cluster_columns:
            aggregate_frequency = case_features[cluster_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
            return float((aggregate_frequency > 0).mean()), float(aggregate_frequency.mean())

    prevalence = pd.to_numeric(group.get("case_prevalence", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    mean_frequency = pd.to_numeric(group.get("mean_case_frequency", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return float(min(prevalence.sum(), 1.0)), float(mean_frequency.sum())


def plot_marker_signed_survival_heatmap(
    by_resolution: pd.DataFrame,
    marker_summary: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    value_column: str = "weighted_coxnet_score",
    ranking_column: str = "mean_abs_signed_coxnet_score",
    model_label: str = "CoxNet",
) -> None:
    if by_resolution.empty or value_column not in by_resolution.columns:
        make_no_data_plot(output_path, f"{model_label} marker survival stability", "No marker-level survival scores were available.")
        return
    marker_order = marker_summary.copy()
    if ranking_column in marker_order.columns:
        marker_order = marker_order.sort_values(ranking_column, ascending=False, na_position="last")
    markers = marker_order.head(args.max_heatmap_markers)["marker"].astype(str).tolist()
    if not markers:
        make_no_data_plot(output_path, f"{model_label} marker survival stability", "No markers passed the stability summary step.")
        return
    data = by_resolution[by_resolution["marker"].astype(str).isin(markers)]
    pivot = data.pivot_table(index="marker", columns="cluster_col", values=value_column, aggfunc="mean")
    pivot = pivot.reindex(index=markers)
    pivot = pivot.reindex(columns=sorted(pivot.columns, key=cluster_sort_key))
    height = max(4.5, min(16.0, 0.25 * len(pivot.index) + 2.0))
    width = max(7.0, min(18.0, 0.7 * len(pivot.columns) + 4.0))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": f"Marker enrichment-weighted {model_label} coefficient"},
    )
    ax.set_xlabel("Leiden cluster column")
    ax.set_ylabel("Marker")
    ax.set_title(f"{model_label} marker-associated survival direction across Leiden settings")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_environment_survival_stability(
    environment_summary: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    coefficient_column: str = "median_coxnet_coefficient",
    sign_consistency_column: str = "coxnet_sign_consistency",
    mean_abs_column: str = "mean_abs_coxnet_coefficient",
    model_label: str = "CoxNet",
    color_metric: str | None = None,
) -> None:
    if environment_summary.empty:
        make_no_data_plot(output_path, f"{model_label} environment survival stability", "No environments were available.")
        return
    if coefficient_column not in environment_summary.columns:
        make_no_data_plot(
            output_path,
            f"{model_label} environment survival stability",
            f"No {model_label} environment coefficients were available.",
        )
        return
    sort_columns = [
        column
        for column in ["resolution_support_fraction", sign_consistency_column, mean_abs_column]
        if column in environment_summary.columns
    ]
    if not sort_columns:
        sort_columns = ["environment_id"]
    plot_df = environment_summary.copy().sort_values(
        sort_columns,
        ascending=[True] * len(sort_columns),
    )
    labels = plot_df.apply(
        lambda row: f"{row.environment_id}: {row.top_intensity_markers}"[:80],
        axis=1,
    )
    height = max(4.5, min(20.0, 0.38 * len(plot_df) + 1.5))
    fig, ax = plt.subplots(figsize=(9.5, height))
    y_positions = np.arange(len(plot_df))
    coefficients = pd.to_numeric(plot_df[coefficient_column], errors="coerce")
    bubble_values = pd.to_numeric(plot_df[args.environment_bubble_metric], errors="coerce").fillna(0.0)
    bubble_sizes = scale_bubble_values(
        bubble_values,
        args.environment_bubble_size_min,
        args.environment_bubble_size_scale,
        args.environment_bubble_log_scale,
    )
    color_metric = color_metric or args.environment_color_metric
    if color_metric not in plot_df.columns:
        color_metric = coefficient_column
    colour_values = pd.to_numeric(plot_df[color_metric], errors="coerce")
    vmin, vmax, cmap, colour_label = colour_limits(
        colour_values,
        color_metric,
        args.environment_color_quantile,
    )
    scatter = ax.scatter(
        coefficients,
        y_positions,
        s=bubble_sizes,
        c=colour_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.4,
    )
    finite_coefficients = coefficients.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_coefficients.empty:
        x_min, x_max = -1.0, 1.0
    else:
        x_min = float(finite_coefficients.min())
        x_max = float(finite_coefficients.max())
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
    x_range = x_max - x_min
    x_left = x_min - 0.15 * x_range
    x_right = x_max + 0.55 * x_range
    ax.set_xlim(x_left, x_right)

    annotation_x = x_right - 0.02 * (x_right - x_left)
    for y_pos, row in zip(y_positions, plot_df.itertuples()):
        coef = getattr(row, coefficient_column, np.nan)
        coef_text = "coef=NA" if pd.isna(coef) else f"coef={coef:.3g}"
        ax.text(
            annotation_x,
            y_pos,
            f"{coef_text}; {int(row.n_clusters)} clusters; {int(row.n_resolutions)} res",
            va="center",
            ha="right",
            fontsize=6.5,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"{metric_label(coefficient_column)} (>0 = higher hazard)")
    ax.set_ylabel("Recurrent environment")
    ax.set_title(f"Recurrent environments: {model_label} survival direction and resolution support")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(colour_label)
    size_handles = [
        plt.scatter(
            [],
            [],
            s=float(scale_bubble_values(pd.Series([value]), args.environment_bubble_size_min, args.environment_bubble_size_scale, args.environment_bubble_log_scale).iloc[0]),
            facecolor="white",
            edgecolor="black",
            label=format_metric_value(value, args.environment_bubble_metric),
        )
        for value in metric_legend_values(bubble_values)
    ]
    ax.legend(
        handles=size_handles,
        title=f"Bubble size: {metric_label(args.environment_bubble_metric)}",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(3, len(size_handles)),
        frameon=True,
        fontsize=7,
        title_fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_effect_bubble(
    cluster_summary: pd.DataFrame,
    environment_summary: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    coefficient_column: str = "coxnet_coefficient",
    nonzero_column: str = "coxnet_nonzero",
    model_label: str = "CoxNet",
) -> None:
    if cluster_summary.empty or coefficient_column not in cluster_summary.columns:
        make_no_data_plot(output_path, f"{model_label} cluster effect bubble plot", f"No cluster-level {model_label} coefficients were available.")
        return
    source_df = cluster_summary.copy()
    source_df["cluster_col"] = source_df["cluster_col"].astype(str)
    order = sorted(source_df["cluster_col"].unique(), key=cluster_sort_key)
    x_lookup = {cluster_col: idx for idx, cluster_col in enumerate(order)}
    y_order = sorted(source_df["environment_id"].dropna().unique())
    y_lookup = {environment_id: idx for idx, environment_id in enumerate(y_order)}
    label_lookup = environment_label_lookup(environment_summary)

    rows = []
    case_feature_cache: dict[Path, pd.DataFrame] = {}
    for (cluster_col, environment_id), group in source_df.groupby(["cluster_col", "environment_id"], sort=False):
        coefficients = pd.to_numeric(group[coefficient_column], errors="coerce")
        aggregate_case_prevalence, aggregate_mean_case_frequency = aggregate_case_support(group, case_feature_cache)
        if "mean_case_frequency" in group.columns:
            weights = pd.to_numeric(group["mean_case_frequency"], errors="coerce")
        elif "case_prevalence" in group.columns:
            weights = pd.to_numeric(group["case_prevalence"], errors="coerce")
        else:
            weights = pd.Series(1.0, index=group.index)

        finite = coefficients.notna()
        finite_weights = weights.where(weights > 0).fillna(0.0)
        if finite.any() and finite_weights.loc[finite].sum() > 0:
            coefficient = float((coefficients.loc[finite] * finite_weights.loc[finite]).sum() / finite_weights.loc[finite].sum())
        elif finite.any():
            coefficient = float(coefficients.loc[finite].median())
        else:
            coefficient = np.nan

        nonzero = (
            group[nonzero_column].fillna(False).astype(bool)
            if nonzero_column in group.columns
            else pd.Series(False, index=group.index)
        )
        rows.append(
            {
                "cluster_col": cluster_col,
                "environment_id": environment_id,
                "x": x_lookup[cluster_col],
                "y": y_lookup[environment_id],
                "aggregated_model_coefficient": coefficient,
                "n_clusters_in_cell": int(group.shape[0]),
                "case_prevalence": aggregate_case_prevalence,
                "mean_case_frequency": aggregate_mean_case_frequency,
                "any_nonzero_model": bool(nonzero.any()),
                "clusters": ", ".join(group["cluster"].astype(str).tolist()),
            }
        )

    plot_df = pd.DataFrame(rows)
    bubble_values = pd.to_numeric(plot_df[args.cluster_bubble_metric], errors="coerce").fillna(0.0)
    sizes = scale_bubble_values(
        bubble_values,
        args.cluster_bubble_size_min,
        args.cluster_bubble_size_scale,
        args.cluster_bubble_log_scale,
    )

    height = max(4.5, min(18.0, 0.55 * len(y_order) + 2.0))
    width = max(7.0, min(18.0, 0.75 * len(order) + 4.5))
    fig, ax = plt.subplots(figsize=(width, height))
    scatter = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=sizes,
        c=pd.to_numeric(plot_df["aggregated_model_coefficient"], errors="coerce"),
        cmap="coolwarm",
        edgecolors=np.where(plot_df["any_nonzero_model"], "black", "lightgray"),
        linewidths=np.where(plot_df["any_nonzero_model"], 0.8, 0.35),
    )
    for row in plot_df[plot_df["n_clusters_in_cell"] > 1].itertuples():
        ax.text(row.x, row.y, str(row.n_clusters_in_cell), ha="center", va="center", fontsize=6, color="black")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels([label_lookup.get(str(environment_id), str(environment_id)) for environment_id in y_order], fontsize=7)
    ax.set_xlabel("Leiden cluster column")
    ax.set_ylabel("Recurrent environment")
    ax.set_title(f"{model_label} cluster effects grouped by recurrent environment")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f"Aggregated {model_label} coefficient (>0 = higher hazard)")

    size_values = metric_legend_values(bubble_values)
    size_handles = [
        plt.scatter(
            [],
            [],
            s=float(scale_bubble_values(pd.Series([value]), args.cluster_bubble_size_min, args.cluster_bubble_size_scale, args.cluster_bubble_log_scale).iloc[0]),
            facecolor="white",
            edgecolor="black",
            label=format_metric_value(value, args.cluster_bubble_metric),
        )
        for value in size_values
    ]
    outline_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            label=f"Non-zero {model_label} coefficient",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="lightgray",
            label=f"Zero / not selected by {model_label}",
            markersize=7,
        ),
    ]
    ax.legend(
        handles=[*outline_handles, *size_handles],
        title=f"Bubble encoding\nsize: {metric_label(args.cluster_bubble_metric)}",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        fontsize=7,
        title_fontsize=8,
    )
    fig.text(
        0.01,
        0.01,
        f"One point is drawn per recurrent environment x Leiden setting. Point size = {metric_label(args.cluster_bubble_metric)}. Centre label shows how many original clusters were merged in that cell. Grey outline is not a p-value; it means all merged {model_label} coefficients are zero at the selected alpha.",
        ha="left",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_survival_overlay(
    marker_summary: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    score_column: str = "median_signed_coxnet_score",
    abs_score_column: str = "mean_abs_signed_coxnet_score",
    effect_count_column: str = "n_effect_resolutions",
    model_label: str = "CoxNet",
) -> None:
    impact_column = f"mean_{args.permutation_type}_impact"
    if marker_summary.empty or impact_column not in marker_summary.columns or score_column not in marker_summary.columns:
        make_no_data_plot(output_path, f"{model_label} permutation and survival overlay", "No marker-level permutation impact table was available.")
        return
    plot_df = marker_summary.dropna(subset=[score_column, impact_column]).copy()
    if plot_df.empty:
        make_no_data_plot(output_path, f"{model_label} permutation and survival overlay", "No markers had both survival scores and permutation impact.")
        return
    scores = pd.to_numeric(plot_df[score_column], errors="coerce")
    plot_df["median_direction"] = np.select(
        [scores > args.effect_threshold, scores < -args.effect_threshold],
        ["higher hazard median", "lower hazard median"],
        default="median zero / no consistent direction",
    )
    direction_palette = {
        "higher hazard median": "#b2182b",
        "lower hazard median": "#2166ac",
        "median zero / no consistent direction": "#9e9e9e",
    }
    colors = plot_df["median_direction"].map(direction_palette)
    if effect_count_column in plot_df.columns:
        effect_counts = pd.to_numeric(plot_df[effect_count_column], errors="coerce").fillna(0)
    else:
        effect_counts = pd.Series(0, index=plot_df.index)
    sizes = 50 + 30 * effect_counts
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        plot_df[score_column],
        plot_df[impact_column],
        s=sizes,
        c=colors,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    finite_scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_scores.empty:
        x_min = min(float(finite_scores.min()), 0.0)
        x_max = max(float(finite_scores.max()), 0.0)
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        x_range = x_max - x_min
        ax.set_xlim(x_min - 0.08 * x_range, x_max + 0.12 * x_range)
    label_df = plot_df.sort_values(abs_score_column, ascending=False, na_position="last").head(15)
    for row in label_df.itertuples():
        ax.text(getattr(row, score_column), getattr(row, impact_column), str(row.marker), fontsize=7)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            label=label,
            markersize=7,
        )
        for label, color in direction_palette.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="Median signed score",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        fontsize=7,
        title_fontsize=8,
    )
    ax.set_xlabel(f"Median marker enrichment-weighted {model_label} coefficient across Leiden settings")
    ax.set_ylabel(f"Mean {args.permutation_type} embedding impact")
    ax.set_title("Do survival-associated markers also affect the VICReg embedding?")
    fig.text(
        0.01,
        0.01,
        "Markers on x=0 have median signed score near zero. They may still have non-zero effects in a minority of Leiden settings, but not in enough settings to move the median away from zero.",
        ha="left",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_validation_by_resolution(validation: pd.DataFrame, output_path: Path) -> None:
    if validation.empty or "mean_harrell_c_index" not in validation.columns:
        make_no_data_plot(output_path, "Model validation by resolution", "No survival_model_validation_summary.csv files were found.")
        return
    plot_df = validation.copy()
    plot_df["cluster_col"] = plot_df["cluster_col"].astype(str)
    models = sorted(plot_df["model"].dropna().astype(str).unique()) if "model" in plot_df.columns else ["model"]
    if not models:
        models = ["model"]
    n_cols = min(2, len(models))
    n_rows = int(math.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.2 * n_rows), squeeze=False, sharey=True)
    order = sorted(plot_df["cluster_col"].unique(), key=cluster_sort_key)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, model_name in zip(axes.ravel(), models):
        ax.set_axis_on()
        subset = plot_df[plot_df["model"].astype(str) == model_name] if "model" in plot_df.columns else plot_df
        sns.pointplot(
            data=subset,
            x="cluster_col",
            y="mean_harrell_c_index",
            hue="feature_set" if "feature_set" in subset.columns else None,
            order=order,
            ax=ax,
            ci=None,
        )
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(str(model_name))
        ax.set_xlabel("Leiden cluster column")
        ax.set_ylabel("Held-out Harrell C-index")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Model validation across Leiden settings", y=1.0)
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)


def plot_primary_vs_image_only(
    cluster_summary: pd.DataFrame,
    output_path: Path,
    primary_column: str = "coxnet_coefficient",
    image_only_column: str = "image_only_coxnet_coefficient",
    model_label: str = "CoxNet",
) -> None:
    required = [primary_column, image_only_column]
    if cluster_summary.empty or not all(column in cluster_summary.columns for column in required):
        make_no_data_plot(output_path, f"Primary vs image-only {model_label}", f"Image-only {model_label} coefficients were not available.")
        return
    plot_df = cluster_summary.dropna(subset=required).copy()
    if plot_df.empty:
        make_no_data_plot(output_path, f"Primary vs image-only {model_label}", "No clusters had both primary and image-only coefficients.")
        return
    order = sorted(plot_df["cluster_col"].astype(str).unique(), key=cluster_sort_key)
    palette = categorical_palette(order)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    sns.scatterplot(
        data=plot_df,
        x=primary_column,
        y=image_only_column,
        hue="cluster_col",
        hue_order=order,
        palette=palette,
        size="case_prevalence" if "case_prevalence" in plot_df.columns else None,
        ax=ax,
        edgecolor="black",
        linewidth=0.3,
    )
    limit = np.nanmax(np.abs(plot_df[required].to_numpy(dtype=float)))
    if np.isfinite(limit) and limit > 0:
        ax.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(-limit * 1.1, limit * 1.1)
        ax.set_ylim(-limit * 1.1, limit * 1.1)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set_xlabel(f"Primary {model_label} coefficient")
    ax.set_ylabel(f"Image-only {model_label} coefficient")
    ax.set_title(f"Clinical-adjusted vs image-only {model_label} cluster effects")
    move_legend_outside(ax, len(order))
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_prevalence_vs_effect(
    cluster_summary: pd.DataFrame,
    output_path: Path,
    coefficient_column: str = "coxnet_coefficient",
    model_label: str = "CoxNet",
) -> None:
    if cluster_summary.empty or coefficient_column not in cluster_summary.columns or "case_prevalence" not in cluster_summary.columns:
        make_no_data_plot(output_path, f"{model_label} cluster prevalence vs effect", f"Case prevalence or {model_label} coefficients were unavailable.")
        return
    plot_df = cluster_summary.dropna(subset=[coefficient_column, "case_prevalence"]).copy()
    if plot_df.empty:
        make_no_data_plot(output_path, f"{model_label} cluster prevalence vs effect", "No clusters had both prevalence and effect estimates.")
        return
    order = sorted(plot_df["cluster_col"].astype(str).unique(), key=cluster_sort_key)
    palette = categorical_palette(order)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    sns.scatterplot(
        data=plot_df,
        x="case_prevalence",
        y=coefficient_column,
        hue="cluster_col",
        hue_order=order,
        palette=palette,
        size="mean_case_frequency" if "mean_case_frequency" in plot_df.columns else None,
        sizes=(30, 300),
        ax=ax,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fraction of cases containing the cluster")
    ax.set_ylabel(f"{model_label} coefficient (>0 = higher hazard)")
    ax.set_title(f"Check whether {model_label} survival effects are rare-cluster dominated")
    move_legend_outside(ax, len(order))
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_coefficient_scatter(
    df: pd.DataFrame,
    output_path: Path,
    x_column: str,
    y_column: str,
    hue_column: str,
    x_label: str,
    y_label: str,
    title: str,
    no_data_title: str,
) -> None:
    required = [x_column, y_column]
    if df.empty or not all(column in df.columns for column in required):
        make_no_data_plot(output_path, no_data_title, "Both model coefficient columns were not available.")
        return
    plot_df = df.dropna(subset=required).copy()
    if plot_df.empty:
        make_no_data_plot(output_path, no_data_title, "No rows had coefficients from both models.")
        return
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    if hue_column in plot_df.columns:
        order = sorted(plot_df[hue_column].astype(str).unique(), key=cluster_sort_key)
        palette = categorical_palette(order)
        sns.scatterplot(
            data=plot_df,
            x=x_column,
            y=y_column,
            hue=hue_column,
            hue_order=order,
            palette=palette,
            size="case_prevalence" if "case_prevalence" in plot_df.columns else None,
            ax=ax,
            edgecolor="black",
            linewidth=0.3,
        )
        move_legend_outside(ax, len(order))
    else:
        ax.scatter(plot_df[x_column], plot_df[y_column], s=60, alpha=0.8, edgecolor="black", linewidth=0.3)
    limit = np.nanmax(np.abs(plot_df[required].to_numpy(dtype=float)))
    if np.isfinite(limit) and limit > 0:
        ax.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(-limit * 1.1, limit * 1.1)
        ax.set_ylim(-limit * 1.1, limit * 1.1)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)


def plot_environment_signature_heatmap(signature: pd.DataFrame, cluster_summary: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    if signature.empty:
        make_no_data_plot(output_path, "Environment marker signatures", "No marker signature matrix was available.")
        return
    variance = signature.var(axis=0).sort_values(ascending=False)
    markers = variance.head(args.max_signature_markers).index.tolist()
    if not markers:
        make_no_data_plot(output_path, "Environment marker signatures", "No marker columns were available.")
        return
    order_df = cluster_summary[["cluster_uid", "environment_id", "cluster_col", "cluster"]].copy()
    order_df = order_df.sort_values(["environment_id", "cluster_col", "cluster"], key=lambda col: col.map(str))
    ordered_index = [uid for uid in order_df["cluster_uid"].astype(str) if uid in signature.index]
    plot_df = signature.loc[ordered_index, markers]
    labels = [
        f"{row.environment_id} | {row.cluster_col}:{row.cluster}"
        for row in order_df.set_index("cluster_uid").loc[ordered_index].itertuples()
    ]
    height = max(5, min(22, 0.25 * len(plot_df.index) + 2))
    width = max(8, min(18, 0.32 * len(markers) + 5))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        plot_df,
        ax=ax,
        cmap="coolwarm",
        center=0,
        yticklabels=labels,
        cbar_kws={"label": "Cluster mean marker z-score"},
    )
    ax.set_xlabel("Marker")
    ax.set_ylabel("Environment | Leiden cluster")
    ax.set_title("Marker signatures of recurrent environments")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def relative_markdown_link(path: str | Path, base_dir: Path, label: str) -> str:
    if not path:
        return ""
    try:
        rel = Path(path).resolve().relative_to(base_dir.resolve())
    except Exception:
        rel = Path(path)
    return f"[{label}]({str(rel).replace(chr(92), '/')})"


def relative_html_path(path: str | Path, base_dir: Path) -> str:
    try:
        rel = Path(path).resolve().relative_to(base_dir.resolve())
    except Exception:
        rel = Path(path)
    return str(rel).replace("\\", "/")


def image_data_uri(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def html_image_card(title: str, path: Path, base_dir: Path, description: str = "") -> str:
    title_html = html.escape(title)
    description_html = html.escape(description)
    source_html = html.escape(relative_html_path(path, base_dir))
    data_uri = image_data_uri(path)
    if data_uri is None:
        return (
            '<section class="card missing">'
            f"<h3>{title_html}</h3>"
            f"<p>{description_html}</p>"
            f"<p class=\"missing-text\">Missing image: <code>{source_html}</code></p>"
            "</section>"
        )
    return (
        '<section class="card">'
        f"<h3>{title_html}</h3>"
        f"<p>{description_html}</p>"
        f'<img src="{data_uri}" alt="{title_html}">'
        f"<p class=\"source\"><code>{source_html}</code></p>"
        "</section>"
    )


def html_link(path: Path, base_dir: Path, label: str) -> str:
    href = html.escape(relative_html_path(path, base_dir))
    return f'<a href="{href}">{html.escape(label)}</a>'


def validation_plot_filename(feature_set: str, model_name: str, suffix: str) -> str:
    safe_feature_set = "".join(ch if ch.isalnum() else "_" for ch in str(feature_set)).strip("_")
    safe_model_name = "".join(ch if ch.isalnum() else "_" for ch in str(model_name)).strip("_")
    return f"validation_{safe_feature_set}__{safe_model_name}__{suffix}.png"


def ridge_cox_image_specs(section: str, output_dir: Path, qualifier: str) -> list[HtmlImageSpec]:
    return [
        HtmlImageSpec(
            section,
            f"{qualifier} Ridge Cox alpha cross-validation",
            output_dir / "ridge_cox_alpha_cv.png",
            "Cross-validated Ridge Cox alpha selection.",
        ),
        HtmlImageSpec(
            section,
            f"{qualifier} Ridge Cox top coefficients",
            output_dir / "ridge_cox_top_coefficients.png",
            "Largest Ridge Cox coefficients by absolute standardized effect size.",
        ),
        HtmlImageSpec(
            section,
            f"{qualifier} Ridge Cox coefficient distribution",
            output_dir / "ridge_cox_coefficient_distribution.png",
            "Distribution of dense Ridge Cox coefficients across fitted features.",
        ),
        HtmlImageSpec(
            section,
            f"{qualifier} Ridge Cox predicted survival",
            output_dir / "ridge_cox_predicted_survival_curves.png",
            "Model-predicted survival curves for representative fitted-risk quantiles.",
        ),
        HtmlImageSpec(
            section,
            f"{qualifier} Ridge Cox predicted cumulative hazard",
            output_dir / "ridge_cox_predicted_cumulative_hazard.png",
            "Model-predicted cumulative hazard curves for representative fitted-risk quantiles.",
        ),
        HtmlImageSpec(
            section,
            f"{qualifier} Ridge Cox observed survival by risk group",
            output_dir / "ridge_cox_observed_km_by_risk_group.png",
            "Kaplan-Meier curves for cases grouped by fitted Ridge Cox risk score.",
        ),
        HtmlImageSpec(
            section,
            f"{qualifier} Ridge-selected standard Cox forest plot",
            output_dir / "ridge_selected_cox_forest_plot.png",
            "Standard Cox estimates for top Ridge-selected features with confidence intervals.",
        ),
    ]


def ridge_validation_image_specs(pair: AnalysisPair) -> list[HtmlImageSpec]:
    summary = read_optional_csv(pair.survival_dir / "survival_model_validation_summary.csv")
    if summary.empty or not {"model", "feature_set"}.issubset(summary.columns):
        return []

    ridge_rows = summary[summary["model"].astype(str) == "ridge_cox"]
    if ridge_rows.empty:
        return []

    specs: list[HtmlImageSpec] = []
    for feature_set in sorted(ridge_rows["feature_set"].dropna().astype(str).unique()):
        label = feature_set.replace("_", " ")
        specs.extend(
            [
                HtmlImageSpec(
                    "Held-out Ridge Cox validation",
                    f"Held-out Ridge Cox KM by risk group ({label})",
                    pair.survival_dir / validation_plot_filename(feature_set, "ridge_cox", "heldout_km_by_risk_group"),
                    "Observed survival for cases grouped by cross-validated held-out Ridge Cox risk.",
                ),
                HtmlImageSpec(
                    "Held-out Ridge Cox validation",
                    f"Held-out Ridge Cox risk distribution ({label})",
                    pair.survival_dir / validation_plot_filename(feature_set, "ridge_cox", "heldout_risk_distribution"),
                    "Held-out Ridge Cox risk-score distributions split by event status.",
                ),
                HtmlImageSpec(
                    "Held-out Ridge Cox validation",
                    f"Held-out Ridge Cox risk vs survival time ({label})",
                    pair.survival_dir / validation_plot_filename(feature_set, "ridge_cox", "heldout_risk_vs_time"),
                    "Relationship between cross-validated held-out Ridge Cox risk and observed survival time.",
                ),
            ]
        )
    return specs


def per_clustering_image_specs(pair: AnalysisPair) -> list[HtmlImageSpec]:
    visual_heatmaps = pair.visual_dir / "heatmaps"
    visual_umaps = pair.visual_dir / "umaps"
    image_survival = pair.survival_dir / "coxnet_image_only"
    specs = [
        HtmlImageSpec(
            "Visualisation QC",
            "UMAP of Leiden clusters",
            visual_umaps / f"umap_{safe_filename(pair.cluster_col)}.png",
            "Patch embedding UMAP coloured by this Leiden clustering.",
        ),
        HtmlImageSpec(
            "Visualisation QC",
            "Cluster mean channel intensity z-score",
            visual_heatmaps / "cluster_mean_channel_intensity_zscore_clustermap.png",
            "Marker enrichment by cluster, hierarchically clustered for comparison.",
        ),
        HtmlImageSpec(
            "Visualisation QC",
            "Zero-channel permutation impact",
            visual_heatmaps / "cluster_mean_permutation_zero_channel_clustermap.png",
            "Cluster-level cosine distance after zeroing each channel.",
        ),
        HtmlImageSpec(
            "Visualisation QC",
            "Shuffle-channel permutation impact",
            visual_heatmaps / "cluster_mean_permutation_shuffle_channel_clustermap.png",
            "Cluster-level cosine distance after shuffling pixels within each channel.",
        ),
        HtmlImageSpec(
            "Image-only CoxNet",
            "Image-only CoxNet best coefficients",
            image_survival / "best_model_coefficients.png",
            "Non-zero image-feature coefficients at the selected CoxNet alpha.",
        ),
        HtmlImageSpec(
            "Image-only CoxNet",
            "Image-only CoxNet path",
            image_survival / "coxnet_path_plot.png",
            "Coefficient paths across the regularisation grid.",
        ),
        HtmlImageSpec(
            "Image-only CoxNet",
            "Image-only standard Cox forest plot",
            image_survival / "cox_forest_plot.png",
            "Standard Cox estimates for selected image features with confidence intervals.",
        ),
        HtmlImageSpec(
            "Image-only CoxNet",
            "Observed survival by image-only CoxNet risk group",
            image_survival / "coxnet_observed_km_by_risk_group.png",
            "Kaplan-Meier curves for cases grouped by image-only CoxNet risk score.",
        ),
    ]
    specs.extend(ridge_cox_image_specs("Primary Ridge Cox", pair.survival_dir, "Primary"))
    specs.extend(ridge_cox_image_specs("Image-only Ridge Cox", pair.survival_dir / "ridge_cox_image_only", "Image-only"))
    specs.extend(ridge_validation_image_specs(pair))
    return specs


def core_stability_image_specs(output_dir: Path, args: argparse.Namespace) -> list[HtmlImageSpec]:
    return [
        HtmlImageSpec(
            "Core CoxNet Stability",
            "Marker signed survival heatmap",
            figure_path(output_dir, "marker_signed_survival_heatmap", args),
            "Cross-Leiden marker-level survival associations.",
        ),
        HtmlImageSpec(
            "Core CoxNet Stability",
            "Environment survival stability",
            figure_path(output_dir, "environment_survival_stability", args),
            "Cluster-environment stability across Leiden settings.",
        ),
        HtmlImageSpec(
            "Core CoxNet Stability",
            "Cluster effect bubble plot",
            figure_path(output_dir, "cluster_effect_bubble_plot", args),
            "Cluster prevalence and survival effect size across settings.",
        ),
        HtmlImageSpec(
            "Core CoxNet Stability",
            "Permutation survival overlay",
            figure_path(output_dir, "permutation_survival_overlay", args),
            "Permutation impact overlaid with survival-associated markers.",
        ),
        HtmlImageSpec(
            "Core Model Validation",
            "Model validation by resolution",
            figure_path(output_dir, "model_validation_by_resolution", args),
            "Cross-validation performance across clustering settings.",
        ),
        HtmlImageSpec(
            "Core Ridge Cox Stability",
            "Ridge marker signed survival heatmap",
            figure_path(output_dir, "ridge_marker_signed_survival_heatmap", args),
            "Cross-Leiden marker-level survival associations using Ridge Cox coefficients.",
        ),
        HtmlImageSpec(
            "Core Ridge Cox Stability",
            "Ridge environment survival stability",
            figure_path(output_dir, "ridge_environment_survival_stability", args),
            "Cluster-environment stability across Leiden settings using Ridge Cox.",
        ),
        HtmlImageSpec(
            "Core Ridge Cox Stability",
            "Ridge cluster effect bubble plot",
            figure_path(output_dir, "ridge_cluster_effect_bubble_plot", args),
            "Cluster prevalence and Ridge Cox survival effect size across settings.",
        ),
        HtmlImageSpec(
            "Core Ridge Cox Stability",
            "Ridge permutation-survival overlay",
            figure_path(output_dir, "ridge_permutation_survival_overlay", args),
            "Permutation impact overlaid with Ridge Cox survival-associated markers.",
        ),
        HtmlImageSpec(
            "Core Ridge Cox Stability",
            "Primary vs image-only Ridge Cox",
            figure_path(output_dir, "ridge_primary_vs_image_only_coefficients", args),
            "Comparison of configured-feature and image-only Ridge Cox coefficients.",
        ),
        HtmlImageSpec(
            "Core Ridge Cox Stability",
            "Ridge cluster prevalence vs effect",
            figure_path(output_dir, "ridge_cluster_prevalence_vs_effect", args),
            "Relationship between cluster prevalence and Ridge Cox survival effect.",
        ),
        HtmlImageSpec(
            "Core Model Comparison",
            "CoxNet vs Ridge cluster coefficients",
            figure_path(output_dir, "coxnet_vs_ridge_cluster_coefficients", args),
            "Cluster-level agreement between sparse CoxNet and dense Ridge Cox effects.",
        ),
        HtmlImageSpec(
            "Core Model Comparison",
            "CoxNet vs Ridge environment effects",
            figure_path(output_dir, "coxnet_vs_ridge_environment_effects", args),
            "Recurrent-environment agreement between CoxNet and Ridge Cox effects.",
        ),
    ]


def render_html_image_sections(specs: list[HtmlImageSpec], base_dir: Path) -> str:
    sections: dict[str, list[HtmlImageSpec]] = {}
    for spec in specs:
        sections.setdefault(spec.section, []).append(spec)

    parts = []
    for section, section_specs in sections.items():
        cards = "\n".join(
            html_image_card(spec.title, spec.path, base_dir, spec.description)
            for spec in section_specs
        )
        parts.append(
            f"<h2>{html.escape(section)}</h2>\n"
            f'<div class="layout">\n{cards}\n</div>'
        )
    return "\n".join(parts)


def write_single_clustering_html_report(pair: AnalysisPair, output_dir: Path, report_dir: Path, args: argparse.Namespace) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{safe_filename(pair.cluster_col)}.html"
    fields = cluster_parameter_fields(pair.cluster_col)
    parameter_items = []
    for label, value in [
        ("Resolution", fields["resolution"]),
        ("n_neighbors", fields["n_neighbors"]),
        ("n_pcs", fields["n_pcs"]),
    ]:
        text = "" if pd.isna(value) else f"{value:g}"
        parameter_items.append(f"<li><strong>{html.escape(label)}:</strong> {html.escape(text)}</li>")

    core_links = [
        html_link(spec.path, report_dir, spec.title)
        for spec in core_stability_image_specs(output_dir, args)
        if spec.path.exists()
    ]

    image_sections = render_html_image_sections(per_clustering_image_specs(pair), report_dir)
    core_links_html = "".join(f"<li>{link}</li>" for link in core_links) or "<li>No core stability figures found.</li>"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(pair.cluster_col)} report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f8fafc; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #52606d; margin-top: 0; }}
    .layout {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06); }}
    .card h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
    .card p {{ margin: 6px 0 10px 0; color: #52606d; font-size: 13px; }}
    .card img {{ width: 100%; height: auto; display: block; border: 1px solid #e4e7eb; background: white; }}
    .source {{ font-size: 11px; word-break: break-all; }}
    .missing {{ border-color: #f5c2c7; background: #fff5f5; }}
    .missing-text {{ color: #a61b1b !important; }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 3px; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>{html.escape(pair.cluster_col)}</h1>
  <p class="meta">Per-clustering visualisation, CoxNet, Ridge Cox, and held-out validation report.</p>
  <h2>Parameters</h2>
  <ul>
    {''.join(parameter_items)}
    <li><strong>Visualisation folder:</strong> <code>{html.escape(relative_html_path(pair.visual_dir, report_dir))}</code></li>
    <li><strong>Survival folder:</strong> <code>{html.escape(relative_html_path(pair.survival_dir, report_dir))}</code></li>
  </ul>
  <h2>Core Stability Figures</h2>
  <ul>{core_links_html}</ul>
  {image_sections}
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def write_per_clustering_html_reports(pairs: list[AnalysisPair], args: argparse.Namespace) -> pd.DataFrame:
    report_dir = args.output_dir / "per_clustering_reports"
    rows = []
    for pair in pairs:
        report_path = write_single_clustering_html_report(pair, args.output_dir, report_dir, args)
        missing = [
            f"{spec.section}: {spec.title}"
            for spec in per_clustering_image_specs(pair)
            if not spec.path.exists()
        ]
        rows.append(
            {
                "cluster_col": pair.cluster_col,
                "html_report": str(report_path),
                "visualisation_dir": str(pair.visual_dir),
                "survival_dir": str(pair.survival_dir),
                "missing_images": "; ".join(missing),
            }
        )

    index_rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(row['cluster_col'])}</code></td>"
        f"<td>{html_link(Path(row['html_report']), report_dir, 'open report')}</td>"
        f"<td>{html.escape(row['missing_images'])}</td>"
        "</tr>"
        for row in rows
    )
    core_sections = render_html_image_sections(core_stability_image_specs(args.output_dir, args), report_dir)
    index_path = report_dir / "index.html"
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Per-clustering Leiden reports</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    .layout {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; margin-bottom: 24px; }}
    .card {{ background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; }}
    .card h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
    .card p {{ margin: 6px 0 10px 0; color: #52606d; font-size: 13px; }}
    .card img {{ width: 100%; height: auto; display: block; border: 1px solid #e4e7eb; background: white; }}
    .source {{ font-size: 11px; word-break: break-all; }}
    .missing {{ border-color: #f5c2c7; background: #fff5f5; }}
    .missing-text {{ color: #a61b1b !important; }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Per-clustering Leiden reports</h1>
  <p>Each report embeds the key visualisation, CoxNet, Ridge Cox, and validation figures for one clustering setting.</p>
  {core_sections}
  <h2>Per-Clustering Reports</h2>
  <table>
    <thead><tr><th>Cluster column</th><th>Report</th><th>Missing images</th></tr></thead>
    <tbody>{index_rows}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(report_dir / "per_clustering_html_report_summary.csv", index=False)
    return summary


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        else:
            display[column] = display[column].fillna("")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(row) for row in rows)])


def write_markdown_report(
    output_dir: Path,
    pairs: list[AnalysisPair],
    warnings_out: list[str],
    environment_summary: pd.DataFrame,
    marker_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    validation: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Leiden Stability Report",
        "",
        "This report cross-references per-Leiden visualisation outputs with survival outputs.",
        "Positive Cox coefficients indicate higher hazard; negative coefficients indicate lower hazard.",
        "",
        "## Inputs",
        "",
        f"- Visualisation directory: `{args.visualisation_dir}`",
        f"- Survival directory: `{args.survival_dir}`",
        f"- Matched Leiden analyses: {len(pairs)}",
        "",
    ]
    if warnings_out:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings_out)
        lines.append("")

    figure_ext = args.figure_format
    lines.extend(
        [
            "## Key Figures",
            "",
            f"- [Marker signed survival heatmap](marker_signed_survival_heatmap.{figure_ext})",
            f"- [Ridge marker signed survival heatmap](ridge_marker_signed_survival_heatmap.{figure_ext})",
            f"- [Environment survival stability](environment_survival_stability.{figure_ext})",
            f"- [Ridge environment survival stability](ridge_environment_survival_stability.{figure_ext})",
            f"- [Cluster effect bubble plot](cluster_effect_bubble_plot.{figure_ext})",
            f"- [Ridge cluster effect bubble plot](ridge_cluster_effect_bubble_plot.{figure_ext})",
            f"- [Permutation-survival overlay](permutation_survival_overlay.{figure_ext})",
            f"- [Ridge permutation-survival overlay](ridge_permutation_survival_overlay.{figure_ext})",
            f"- [Model validation by resolution](model_validation_by_resolution.{figure_ext})",
            f"- [Primary vs image-only CoxNet](coxnet_primary_vs_image_only_coefficients.{figure_ext})",
            f"- [Primary vs image-only Ridge Cox](ridge_primary_vs_image_only_coefficients.{figure_ext})",
            f"- [Cluster prevalence vs effect](cluster_prevalence_vs_effect.{figure_ext})",
            f"- [Ridge cluster prevalence vs effect](ridge_cluster_prevalence_vs_effect.{figure_ext})",
            f"- [CoxNet vs Ridge cluster coefficients](coxnet_vs_ridge_cluster_coefficients.{figure_ext})",
            f"- [CoxNet vs Ridge environment effects](coxnet_vs_ridge_environment_effects.{figure_ext})",
            f"- [Environment marker signatures](environment_marker_signature_heatmap.{figure_ext})",
        ]
    )
    html_index = output_dir / "per_clustering_reports" / "index.html"
    if html_index.exists():
        lines.append("- [Per-clustering embedded HTML reports](per_clustering_reports/index.html)")
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `cluster_survival_summary.csv`: one row per Leiden cluster with survival effects, top markers, case prevalence, and environment assignment.",
            "- `cluster_survival_marker_table.csv`: one row per cluster-marker combination with intensity, permutation, and signed survival scores.",
            "- `environment_stability_summary.csv`: recurrent environment groups and their survival direction consistency.",
            "- `marker_survival_stability_by_resolution.csv`: marker survival scores per Leiden setting.",
            "- `marker_survival_stability_summary.csv`: marker-level stability summary across Leiden settings.",
            "- `model_resolution_stability_summary.csv`: held-out model validation metrics by Leiden setting, if available.",
            "",
        ]
    )

    if not environment_summary.empty:
        lines.extend(["## Most Recurrent Candidate Environments", ""])
        columns = [
            "environment_id",
            "n_clusters",
            "n_resolutions",
            "dominant_coxnet_direction",
            "coxnet_sign_consistency",
            "median_coxnet_coefficient",
            "dominant_ridge_cox_direction",
            "ridge_cox_sign_consistency",
            "median_ridge_cox_coefficient",
            "top_intensity_markers",
        ]
        columns = [column for column in columns if column in environment_summary.columns]
        lines.append(dataframe_to_markdown(environment_summary[columns].head(args.max_report_items)))
        lines.append("")

    if not marker_summary.empty:
        lines.extend(["## Marker Stability Summary", ""])
        columns = [
            "marker",
            "n_effect_resolutions",
            "dominant_coxnet_direction",
            "coxnet_sign_consistency",
            "median_signed_coxnet_score",
            "dominant_ridge_cox_direction",
            "ridge_cox_sign_consistency",
            "median_signed_ridge_cox_score",
            f"mean_{args.permutation_type}_impact",
        ]
        columns = [column for column in columns if column in marker_summary.columns]
        lines.append(dataframe_to_markdown(marker_summary[columns].head(args.max_report_items)))
        lines.append("")

    if not cluster_summary.empty and "gallery_path" in cluster_summary.columns:
        lines.extend(["## Gallery Links For Top Environments", ""])
        shown = 0
        top_environment_ids = environment_summary.head(args.max_report_items)["environment_id"].tolist() if not environment_summary.empty else []
        for environment_id in top_environment_ids:
            group = cluster_summary[cluster_summary["environment_id"] == environment_id].head(6)
            links = []
            for row in group.itertuples():
                link = relative_markdown_link(row.gallery_path, output_dir, f"{row.cluster_col}:{row.cluster}")
                if link:
                    links.append(link)
            if links:
                lines.append(f"- `{environment_id}`: " + ", ".join(links))
                shown += 1
        if shown == 0:
            lines.append("- No gallery image paths were found in the matched visualisation folders.")
        lines.append("")

    if not validation.empty:
        lines.extend(["## Validation Caveat", ""])
        lines.append(
            "The model validation plot uses held-out survival predictions from the existing survival outputs. "
            "It should be used to assess sensitivity across Leiden settings, not to select a single best "
            "setting purely by outcome performance."
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Stable environments should recur across multiple Leiden settings and have coherent marker signatures.",
            "- A survival association is more convincing when CoxNet, Ridge Cox, image-only models, standard Cox, and univariate directions agree.",
            "- Marker-survival evidence is stronger when the same marker direction appears across resolutions and the permutation report suggests the marker affects the VICReg embedding.",
            "- Rare clusters with high maximum case frequency but low case prevalence should be reviewed carefully because they may be driven by one or two cases.",
            f"- In `environment_survival_stability.{figure_ext}`, bubble size is `{args.environment_bubble_metric}` and colour is `{args.environment_color_metric}`.",
            f"- In `cluster_effect_bubble_plot.{figure_ext}`, one point is drawn per recurrent environment and Leiden setting. Bubble size is `{args.cluster_bubble_metric}`; centre labels show how many original clusters were merged into that cell.",
            f"- In `cluster_effect_bubble_plot.{figure_ext}`, grey point outlines mean all merged CoxNet coefficients are zero at the selected alpha. This is not a p-value or significance test.",
            "- Ridge Cox plots use dense L2-penalised coefficients. They are useful for ranking correlated environments, while CoxNet remains the sparse feature-selection view.",
            "",
        ]
    )

    with open(output_dir / "LEIDEN_STABILITY_REPORT.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    global FIGURE_DPI
    args = parse_args()
    FIGURE_DPI = args.figure_dpi
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    pairs, warnings_out = pair_analysis_dirs(args.visualisation_dir, args.survival_dir)
    if not pairs:
        raise ValueError(
            "No matched Leiden visualisation/survival outputs were found. "
            "Check --visualisation-dir and --survival-dir."
        )

    cluster_frames = []
    marker_frames = []
    signatures = []
    validation_frames = []
    for pair in pairs:
        print(f"Cross-referencing {pair.cluster_col}")
        print(f"  visualisation: {pair.visual_dir}")
        print(f"  survival:      {pair.survival_dir}")
        cluster_df, marker_df, signature, validation = merge_pair_data(pair, args)
        cluster_frames.append(cluster_df)
        marker_frames.append(marker_df)
        signatures.append(signature)
        if not validation.empty:
            validation_frames.append(validation)

    cluster_summary = pd.concat(cluster_frames, ignore_index=True)
    marker_table = pd.concat(marker_frames, ignore_index=True)
    validation_summary = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    global_signature = build_global_signature(signatures, cluster_frames)

    environments, env_warnings = assign_environments(global_signature, args.environment_distance_threshold)
    warnings_out.extend(env_warnings)
    if not environments.empty:
        cluster_summary = cluster_summary.merge(
            environments.rename("environment_id").reset_index().rename(columns={"index": "cluster_uid"}),
            on="cluster_uid",
            how="left",
        )
    else:
        cluster_summary["environment_id"] = "E001"
    cluster_summary["environment_id"] = cluster_summary["environment_id"].fillna("unassigned")
    marker_table = marker_table.merge(cluster_summary[["cluster_uid", "environment_id"]], on="cluster_uid", how="left")

    total_resolutions = cluster_summary["cluster_col"].nunique()
    environment_summary = summarise_environments(cluster_summary, marker_table, global_signature, total_resolutions, args)
    marker_by_resolution, marker_summary = summarise_marker_stability(marker_table, args)

    cluster_summary = cluster_summary.sort_values(["cluster_col", "cluster"], key=lambda col: col.map(str)).reset_index(drop=True)
    marker_table = marker_table.sort_values(["cluster_col", "cluster", "marker"], key=lambda col: col.map(str)).reset_index(drop=True)

    cluster_summary.to_csv(args.output_dir / "cluster_survival_summary.csv", index=False)
    marker_table.to_csv(args.output_dir / "cluster_survival_marker_table.csv", index=False)
    environment_summary.to_csv(args.output_dir / "environment_stability_summary.csv", index=False)
    marker_by_resolution.to_csv(args.output_dir / "marker_survival_stability_by_resolution.csv", index=False)
    marker_summary.to_csv(args.output_dir / "marker_survival_stability_summary.csv", index=False)
    validation_summary.to_csv(args.output_dir / "model_resolution_stability_summary.csv", index=False)
    global_signature.to_csv(args.output_dir / "environment_marker_signature_matrix.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plot_marker_signed_survival_heatmap(
            marker_by_resolution,
            marker_summary,
            figure_path(args.output_dir, "marker_signed_survival_heatmap", args),
            args,
        )
        plot_marker_signed_survival_heatmap(
            marker_by_resolution,
            marker_summary,
            figure_path(args.output_dir, "ridge_marker_signed_survival_heatmap", args),
            args,
            value_column="weighted_ridge_cox_score",
            ranking_column="mean_abs_signed_ridge_cox_score",
            model_label="Ridge Cox",
        )
        plot_environment_survival_stability(environment_summary, figure_path(args.output_dir, "environment_survival_stability", args), args)
        plot_environment_survival_stability(
            environment_summary,
            figure_path(args.output_dir, "ridge_environment_survival_stability", args),
            args,
            coefficient_column="median_ridge_cox_coefficient",
            sign_consistency_column="ridge_cox_sign_consistency",
            mean_abs_column="mean_abs_ridge_cox_coefficient",
            model_label="Ridge Cox",
            color_metric="median_ridge_cox_coefficient",
        )
        plot_cluster_effect_bubble(cluster_summary, environment_summary, figure_path(args.output_dir, "cluster_effect_bubble_plot", args), args)
        plot_cluster_effect_bubble(
            cluster_summary,
            environment_summary,
            figure_path(args.output_dir, "ridge_cluster_effect_bubble_plot", args),
            args,
            coefficient_column="ridge_cox_coefficient",
            nonzero_column="ridge_cox_nonzero",
            model_label="Ridge Cox",
        )
        plot_permutation_survival_overlay(marker_summary, figure_path(args.output_dir, "permutation_survival_overlay", args), args)
        plot_permutation_survival_overlay(
            marker_summary,
            figure_path(args.output_dir, "ridge_permutation_survival_overlay", args),
            args,
            score_column="median_signed_ridge_cox_score",
            abs_score_column="mean_abs_signed_ridge_cox_score",
            effect_count_column="ridge_cox_n_effect_resolutions",
            model_label="Ridge Cox",
        )
        plot_model_validation_by_resolution(validation_summary, figure_path(args.output_dir, "model_validation_by_resolution", args))
        plot_primary_vs_image_only(cluster_summary, figure_path(args.output_dir, "coxnet_primary_vs_image_only_coefficients", args))
        plot_primary_vs_image_only(
            cluster_summary,
            figure_path(args.output_dir, "ridge_primary_vs_image_only_coefficients", args),
            primary_column="ridge_cox_coefficient",
            image_only_column="image_only_ridge_cox_coefficient",
            model_label="Ridge Cox",
        )
        plot_cluster_prevalence_vs_effect(cluster_summary, figure_path(args.output_dir, "cluster_prevalence_vs_effect", args))
        plot_cluster_prevalence_vs_effect(
            cluster_summary,
            figure_path(args.output_dir, "ridge_cluster_prevalence_vs_effect", args),
            coefficient_column="ridge_cox_coefficient",
            model_label="Ridge Cox",
        )
        plot_model_coefficient_scatter(
            cluster_summary,
            figure_path(args.output_dir, "coxnet_vs_ridge_cluster_coefficients", args),
            x_column="coxnet_coefficient",
            y_column="ridge_cox_coefficient",
            hue_column="cluster_col",
            x_label="CoxNet coefficient",
            y_label="Ridge Cox coefficient",
            title="CoxNet vs Ridge Cox cluster effects",
            no_data_title="CoxNet vs Ridge Cox cluster effects",
        )
        plot_model_coefficient_scatter(
            environment_summary,
            figure_path(args.output_dir, "coxnet_vs_ridge_environment_effects", args),
            x_column="median_coxnet_coefficient",
            y_column="median_ridge_cox_coefficient",
            hue_column="",
            x_label="Median CoxNet coefficient",
            y_label="Median Ridge Cox coefficient",
            title="CoxNet vs Ridge Cox recurrent environment effects",
            no_data_title="CoxNet vs Ridge Cox recurrent environment effects",
        )
        plot_environment_signature_heatmap(global_signature, cluster_summary, figure_path(args.output_dir, "environment_marker_signature_heatmap", args), args)

    if args.per_clustering_html_reports:
        html_summary = write_per_clustering_html_reports(pairs, args)
        missing_count = int(html_summary["missing_images"].astype(str).ne("").sum()) if not html_summary.empty else 0
        if missing_count:
            warnings_out.append(
                f"Per-clustering HTML reports were written, but {missing_count} reports have one or more missing images. "
                "See per_clustering_reports/per_clustering_html_report_summary.csv."
            )

    if warnings_out:
        with open(args.output_dir / "stability_warnings.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(warnings_out) + "\n")

    write_markdown_report(
        output_dir=args.output_dir,
        pairs=pairs,
        warnings_out=warnings_out,
        environment_summary=environment_summary,
        marker_summary=marker_summary,
        cluster_summary=cluster_summary,
        validation=validation_summary,
        args=args,
    )

    print(f"Saved Leiden stability report to {args.output_dir}")


if __name__ == "__main__":
    main()
