"""Standalone pair-correlation analysis utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import TwoSlopeNorm

AnnDataLike = Union[ad.AnnData, str, Path]


@dataclass
class SampleSpatialData:
    sample_id: str
    df: pd.DataFrame
    points: np.ndarray
    domain_x: float
    domain_y: float
    cluster_points: Dict[Hashable, np.ndarray]
    cluster_counts: Dict[Hashable, int]


@dataclass
class PairCorrelationSampleResult:
    radii: np.ndarray
    gs: np.ndarray
    contributions: List[List[Optional[np.ndarray]]]


def run_paircorrelation_at_distance(
    adata: AnnDataLike,
    *,
    population_obs: str,
    groupby: Optional[str],
    target_distance: float,
    spoox_output_dir: Union[str, Path] = "pcf_out",
    spoox_output_summary_dir: Union[str, Path] = "pcf_out_summary",
    stats_file: Union[str, Path] = "stats.txt",
    conditions_file: Union[str, Path] = "conditions.json",
    index_obs: str = "Master_Index",
    roi_obs: str = "ROI",
    xloc_obs: str = "X_loc",
    yloc_obs: str = "Y_loc",
    cluster_column: str = "cluster",
    samples: Optional[Sequence[str]] = None,
    max_radius: float = 300.0,
    radius_step: float = 10.0,
    num_bootstrap: int = 999,
) -> pd.DataFrame:
    """Run pair-correlation analysis entirely in Python and average by condition.

    The function exports the intermediate ``stats.txt`` and ``conditions.json`` for
    reference, computes pair-correlation curves per sample, averages them by
    condition with bootstrap confidence bounds, and returns a tidy table with
    g(r) statistics evaluated at ``target_distance``.
    """

    if target_distance <= 0:
        raise ValueError("target_distance must be greater than zero.")
    if max_radius <= 0 or radius_step <= 0:
        raise ValueError("max_radius and radius_step must be greater than zero.")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be greater than zero.")

    adata = _load_adata(adata)
    print(f"[pcf] Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} variables")

    stats_path = Path(stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    spatial_stats_df, selected_samples = _create_spatial_stats_table(
        adata,
        population_obs=population_obs,
        roi_obs=roi_obs,
        xloc_obs=xloc_obs,
        yloc_obs=yloc_obs,
        index_obs=index_obs,
        cluster_column=cluster_column,
        samples=samples,
        groupby=groupby,
    )
    spatial_stats_df.to_csv(stats_path, sep="\t", index=False)
    print(f"[pcf] Spatial stats table written to {stats_path}")

    conditions = _build_conditions_mapping(
        adata, roi_obs=roi_obs, groupby=groupby, samples=selected_samples
    )
    conditions_payload: Dict[str, Union[str, Dict[str, List[str]]]] = {
        "conditions": conditions
    }
    if groupby:
        conditions_payload["name"] = groupby
    conditions_path = Path(conditions_file)
    conditions_path.parent.mkdir(parents=True, exist_ok=True)
    conditions_path.write_text(json.dumps(conditions_payload, indent=2), encoding="utf-8")

    summary_dir = Path(spoox_output_summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    Path(spoox_output_dir).mkdir(parents=True, exist_ok=True)

    annotations = _build_annotations(spatial_stats_df, cluster_column)
    datasets = _create_sample_datasets(spatial_stats_df, cluster_column)
    print(f"[pcf] Prepared per-sample datasets ({len(datasets)} total)")
    cluster_numbers = annotations["ClusterNumber"].tolist()

    per_sample_results = _compute_pair_correlations(
        datasets,
        cluster_numbers,
        max_radius=max_radius,
        radius_step=radius_step,
    )
    print("[pcf] Completed pair-correlation curves for all samples")

    condition_curves = _average_paircorrelations_by_condition(
        conditions,
        datasets,
        per_sample_results,
        annotations,
        num_bootstrap=num_bootstrap,
    )
    print("[pcf] Completed condition-level bootstrapping and averaging")

    result_df = _summarise_target_radius(condition_curves, target_distance)
    summary_path = (
        summary_dir
        / "paircorrelationfunction"
        / f"pcf_{target_distance:.1f}um_summary.tsv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(summary_path, sep="\t", index=False)
    print(f"[pcf] Saved summary table to {summary_path}")
    return result_df


def _load_adata(adata: AnnDataLike) -> ad.AnnData:
    if isinstance(adata, ad.AnnData):
        return adata
    return ad.read_h5ad(str(adata))


def _create_spatial_stats_table(
    adata: ad.AnnData,
    *,
    population_obs: str,
    roi_obs: str,
    xloc_obs: str,
    yloc_obs: str,
    index_obs: str,
    cluster_column: str,
    samples: Optional[Sequence[str]],
    groupby: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    obs = adata.obs.copy()
    if samples:
        missing = sorted(set(samples) - set(obs[roi_obs].unique()))
        if missing:
            raise ValueError(f"Samples not found in {roi_obs}: {missing}")
        obs = obs[obs[roi_obs].isin(samples)]
        sample_order = list(samples)
    else:
        sample_order = obs[roi_obs].unique().tolist()

    cols = [index_obs, roi_obs, xloc_obs, yloc_obs, population_obs]
    rename_map = {
        index_obs: "cellID",
        roi_obs: "sample_id",
        xloc_obs: "x",
        yloc_obs: "y",
        population_obs: cluster_column,
    }
    if groupby:
        cols.append(groupby)
        rename_map[groupby] = "Conditions"

    stats_df = obs[cols].rename(columns=rename_map).copy()
    stats_df["cellID"] = "ID_" + stats_df["cellID"].astype(str)

    stats_df["label"] = 0
    for sample in stats_df["sample_id"].unique():
        mask = stats_df["sample_id"] == sample
        stats_df.loc[mask, "label"] = np.arange(1, mask.sum() + 1)
    stats_df["label"] = stats_df["label"].astype(int)

    return stats_df, sample_order


def _build_conditions_mapping(
    adata: ad.AnnData,
    *,
    roi_obs: str,
    groupby: Optional[str],
    samples: Sequence[str],
) -> Dict[str, List[str]]:
    if groupby:
        df = adata.obs[[roi_obs, groupby]].drop_duplicates(subset=roi_obs)
        grouped = df.groupby(groupby)[roi_obs].apply(list).to_dict()
        return {str(cond): list(rois) for cond, rois in grouped.items()}

    return {"All": list(samples)}


def _build_annotations(stats_df: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    clusters = sorted(stats_df[cluster_column].unique())
    return pd.DataFrame(
        {
            "ClusterNumber": clusters,
            "Annotation": clusters,
        }
    )


def _create_sample_datasets(
    stats_df: pd.DataFrame, cluster_column: str
) -> Dict[str, SampleSpatialData]:
    datasets: Dict[str, SampleSpatialData] = {}
    for sample_id, sample_df in stats_df.groupby("sample_id"):
        coords = sample_df[["x", "y"]].to_numpy(dtype=float, copy=True)
        domain_x = max(50.0, float(ceil(sample_df["x"].max() / 50.0) * 50.0))
        domain_y = max(50.0, float(ceil(sample_df["y"].max() / 50.0) * 50.0))
        cluster_points = {
            cluster: sub_df[["x", "y"]].to_numpy(dtype=float, copy=True)
            for cluster, sub_df in sample_df.groupby(cluster_column, observed=True)
        }
        counts = sample_df[cluster_column].value_counts().to_dict()
        datasets[str(sample_id)] = SampleSpatialData(
            sample_id=str(sample_id),
            df=sample_df.reset_index(drop=True),
            points=coords,
            domain_x=domain_x,
            domain_y=domain_y,
            cluster_points=cluster_points,
            cluster_counts=counts,
        )
    return datasets


def _compute_pair_correlations(
    datasets: Dict[str, SampleSpatialData],
    cluster_numbers: Sequence[Hashable],
    *,
    max_radius: float,
    radius_step: float,
) -> Dict[str, PairCorrelationSampleResult]:
    results: Dict[str, PairCorrelationSampleResult] = {}
    for sample_id, dataset in datasets.items():
        results[sample_id] = _compute_pair_correlation_for_sample(
            dataset,
            cluster_numbers,
            max_radius=max_radius,
            radius_step=radius_step,
        )
    return results


def _compute_pair_correlation_for_sample(
    dataset: SampleSpatialData,
    cluster_numbers: Sequence[Hashable],
    *,
    max_radius: float,
    radius_step: float,
) -> PairCorrelationSampleResult:
    radii = np.arange(0, max_radius, radius_step)
    if len(radii) == 0:
        raise ValueError("max_radius must be greater than radius_step.")

    n_clusters = len(cluster_numbers)
    gs = np.full((n_clusters, n_clusters, len(radii)), np.nan, dtype=float)
    contributions: List[List[Optional[np.ndarray]]] = [
        [None for _ in cluster_numbers] for _ in cluster_numbers
    ]

    areas_cache: Dict[Hashable, np.ndarray] = {}
    for cluster in cluster_numbers:
        points = dataset.cluster_points.get(cluster)
        if points is None or len(points) == 0:
            continue
        areas_cache[cluster] = getAnnulusAreasAroundPoints(
            points,
            radius_step,
            max_radius,
            dataset.domain_x,
            dataset.domain_y,
        )

    for a, cluster_a in enumerate(cluster_numbers):
        points_a = dataset.cluster_points.get(cluster_a)
        if points_a is None or len(points_a) == 0:
            continue
        areas_a = areas_cache.get(cluster_a)
        if areas_a is None:
            continue

        for b, cluster_b in enumerate(cluster_numbers):
            points_b = dataset.cluster_points.get(cluster_b)
            if points_b is None or len(points_b) == 0:
                continue
            density_b = len(points_b) / (dataset.domain_x * dataset.domain_y)
            if density_b == 0:
                continue

            distances = cdist(points_a, points_b, metric="euclidean")
            radii_lower, g_vals, contrib = crossPCF(
                distances,
                areas_a,
                density_b,
                radius_step,
                max_radius,
            )
            if len(radii_lower) != len(radii):
                raise ValueError("Radii mismatch while computing PCF.")
            gs[a, b, :] = g_vals.reshape(-1)
            contributions[a][b] = contrib

    return PairCorrelationSampleResult(radii=radii, gs=gs, contributions=contributions)


def _average_paircorrelations_by_condition(
    conditions: Dict[str, List[str]],
    datasets: Dict[str, SampleSpatialData],
    per_sample_results: Dict[str, PairCorrelationSampleResult],
    annotations: pd.DataFrame,
    *,
    num_bootstrap: int,
) -> Dict[str, Dict[Tuple[Hashable, Hashable], Optional[Dict[str, np.ndarray]]]]:
    cluster_numbers = annotations["ClusterNumber"].tolist()
    results: Dict[str, Dict[Tuple[Hashable, Hashable], Optional[Dict[str, np.ndarray]]]] = {}

    for condition, rois in conditions.items():
        roi_datasets = [datasets[roi] for roi in rois if roi in datasets]
        if not roi_datasets:
            continue
        cell_totals = {cluster: 0 for cluster in cluster_numbers}
        for ds in roi_datasets:
            for cluster, count in ds.cluster_counts.items():
                cell_totals[cluster] = cell_totals.get(cluster, 0) + count
        cell_averages = {
            cluster: cell_totals.get(cluster, 0) / len(roi_datasets)
            for cluster in cluster_numbers
        }

        condition_payload: Dict[
            Tuple[Hashable, Hashable], Optional[Dict[str, np.ndarray]]
        ] = {}

        for a, cluster_a in enumerate(cluster_numbers):
            for b, cluster_b in enumerate(cluster_numbers):
                all_contribs: List[np.ndarray] = []
                rect_counts: List[int] = []
                rect_contribs: List[np.ndarray] = []
                rect_ns: List[np.ndarray] = []
                radii = None

                for ds in roi_datasets:
                    sample_result = per_sample_results.get(ds.sample_id)
                    if sample_result is None:
                        continue
                    contrib = sample_result.contributions[a][b]
                    if contrib is None:
                        continue
                    radii = sample_result.radii
                    all_contribs.append(contrib)
                    points_a = ds.cluster_points.get(cluster_a)
                    if points_a is None or len(points_a) == 0:
                        continue
                    n_rect, rect_c, rect_n = getPCFContributionsWithinGrid(
                        contrib,
                        ds.domain_x,
                        ds.domain_y,
                        points_a,
                    )
                    rect_counts.append(n_rect)
                    rect_contribs.append(rect_c)
                    rect_ns.append(rect_n)

                if not all_contribs or radii is None or not rect_counts:
                    condition_payload[(cluster_a, cluster_b)] = None
                    continue

                total_rectangles = int(np.sum(rect_counts))
                rect_contrib_array = np.concatenate(rect_contribs, axis=0)
                rect_ns_array = np.concatenate(rect_ns, axis=0)

                to_sample = np.random.choice(
                    total_rectangles,
                    size=(total_rectangles, num_bootstrap),
                )
                sample = np.sum(rect_contrib_array[to_sample, :], axis=0)
                Ns = np.sum(rect_ns_array[to_sample], axis=0)
                sample_pcfs = sample / Ns[:, np.newaxis]

                all_contribs_concat = np.concatenate(all_contribs, axis=0)
                pcf_mean = np.mean(all_contribs_concat, axis=0)
                upper_percentile = np.percentile(sample_pcfs, 97.5, axis=0)
                lower_percentile = np.percentile(sample_pcfs, 2.5, axis=0)
                pcf_min = 2 * pcf_mean - upper_percentile  # lower bound
                pcf_max = 2 * pcf_mean - lower_percentile  # upper bound

                condition_payload[(cluster_a, cluster_b)] = {
                    "radii": radii,
                    "PCF_mean": pcf_mean,
                    "PCF_max": pcf_max,
                    "PCF_min": pcf_min,
                    "cell_avg_a": cell_averages.get(cluster_a, 0.0),
                    "cell_avg_b": cell_averages.get(cluster_b, 0.0),
                }

        results[condition] = condition_payload

    return results


def _summarise_target_radius(
    condition_curves: Dict[
        str, Dict[Tuple[Hashable, Hashable], Optional[Dict[str, np.ndarray]]]
    ],
    target_distance: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Union[str, float]]] = []
    for condition, pairs in condition_curves.items():
        for (cluster_a, cluster_b), payload in pairs.items():
            if not payload:
                continue
            radii = payload["radii"]
            idx = int(np.argmin(np.abs(radii - target_distance)))
            evaluated = float(radii[idx])
            rows.append(
                {
                    "condition": condition,
                    "cell_type_1": str(cluster_a),
                    "cell_type_2": str(cluster_b),
                    "requested_um": target_distance,
                    "evaluated_um": evaluated,
                    "g_mean": float(payload["PCF_mean"][idx]),
                    "g_min": float(payload["PCF_min"][idx]),
                    "g_max": float(payload["PCF_max"][idx]),
                    "delta_um": abs(evaluated - target_distance),
                    "mean_cell_type_1": float(payload["cell_avg_a"]),
                    "mean_cell_type_2": float(payload["cell_avg_b"]),
                }
            )

    if not rows:
        raise ValueError(
            "No pair-correlation results were generated. Verify that the selected "
            "samples contain at least two populations with observations."
        )

    return pd.DataFrame(rows).sort_values(
        ["condition", "cell_type_1", "cell_type_2"]
    ).reset_index(drop=True)


def getPCFContributionsWithinGrid(
    contributions: np.ndarray, xmax: float, ymax: float, points: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    contributions = np.asarray(contributions)
    rectangle_width_x = 100
    rectangle_width_y = 100

    x_rect = np.arange(0, xmax + 1, rectangle_width_x)
    y_rect = np.arange(0, ymax + 1, rectangle_width_y)

    n_rectangles_x = len(x_rect) - 1
    n_rectangles_y = len(y_rect) - 1

    rect_ns = np.zeros(n_rectangles_x * n_rectangles_y)
    rect_contribs = np.zeros(
        (n_rectangles_x * n_rectangles_y, contributions.shape[1])
    )
    rect_id = 0
    for i in range(n_rectangles_x):
        for j in range(n_rectangles_y):
            accept = (points[:, 0] > x_rect[i]) & (points[:, 0] <= x_rect[i + 1])
            accept &= (points[:, 1] > y_rect[j]) & (points[:, 1] <= y_rect[j + 1])
            if np.sum(accept) > 0:
                rect_contribs[rect_id, :] = np.sum(contributions[accept, :], axis=0)
                rect_ns[rect_id] = np.sum(accept)
            rect_id += 1
    n_rectangles = n_rectangles_x * n_rectangles_y
    return n_rectangles, rect_contribs, rect_ns


def returnIntersectionPoints(x0, y0, r, domain_x, domain_y):
    intersection_points = []

    if x0 ** 2 + (domain_y - y0) ** 2 < r ** 2:
        intersection_points.append([0, domain_y])

    if x0 < r:
        if 0 < y0 + np.sqrt(r ** 2 - x0 ** 2) < domain_y:
            intersection_points.append([0, np.sqrt(r ** 2 - x0 ** 2) + y0])
        if 0 < y0 - np.sqrt(r ** 2 - x0 ** 2) < domain_y:
            intersection_points.append([0, y0 - np.sqrt(r ** 2 - x0 ** 2)])

    if x0 ** 2 + y0 ** 2 < r ** 2:
        intersection_points.append([0, 0])

    if y0 < r:
        if 0 < x0 - np.sqrt(r ** 2 - y0 ** 2) < domain_x:
            intersection_points.append([x0 - np.sqrt(r ** 2 - y0 ** 2), 0])
        if 0 < x0 + np.sqrt(r ** 2 - y0 ** 2) < domain_x:
            intersection_points.append([x0 + np.sqrt(r ** 2 - y0 ** 2), 0])

    if (domain_x - x0) ** 2 + y0 ** 2 < r ** 2:
        intersection_points.append([domain_x, 0])

    if domain_x - x0 < r:
        if 0 < y0 - np.sqrt(r ** 2 - (domain_x - x0) ** 2) < domain_y:
            intersection_points.append(
                [domain_x, y0 - np.sqrt(r ** 2 - (domain_x - x0) ** 2)]
            )
        if 0 < y0 + np.sqrt(r ** 2 - (domain_x - x0) ** 2) < domain_y:
            intersection_points.append(
                [domain_x, y0 + np.sqrt(r ** 2 - (domain_x - x0) ** 2)]
            )

    if (domain_x - x0) ** 2 + (domain_y - y0) ** 2 < r ** 2:
        intersection_points.append([domain_x, domain_y])

    if domain_y - y0 < r:
        if 0 < x0 + np.sqrt(r ** 2 - (domain_y - y0) ** 2) < domain_x:
            intersection_points.append(
                [x0 + np.sqrt(r ** 2 - (domain_y - y0) ** 2), domain_y]
            )
        if 0 < x0 - np.sqrt(r ** 2 - (domain_y - y0) ** 2) < domain_x:
            intersection_points.append(
                [x0 - np.sqrt(r ** 2 - (domain_y - y0) ** 2), domain_y]
            )
    return intersection_points


def returnAreaOfCircleInDomain(x0, y0, r, domain_x, domain_y):
    intersection_points = returnIntersectionPoints(x0, y0, r, domain_x, domain_y)

    if not intersection_points:
        area = np.pi * r ** 2
    else:
        intersection_points.append(intersection_points[0])
        area = 0
        for idx in range(len(intersection_points) - 1):
            a = intersection_points[idx]
            b = intersection_points[idx + 1]
            is_triangle = False

            if a[0] == b[0] or a[1] == b[1]:
                if a[0] == b[0]:
                    if a[0] == 0 and b[1] < a[1]:
                        is_triangle = True
                    elif a[0] != 0 and b[1] > a[1]:
                        is_triangle = True
                else:
                    if a[1] == 0 and b[0] > a[0]:
                        is_triangle = True
                    elif a[1] != 0 and a[0] > b[0]:
                        is_triangle = True

            if is_triangle:
                area += 0.5 * abs(
                    a[0] * (b[1] - y0) + b[0] * (y0 - a[1]) + x0 * (a[1] - b[1])
                )
            else:
                v1 = [x0 - a[0], y0 - a[1]]
                v2 = [x0 - b[0], y0 - b[1]]
                theta = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                if theta < 0:
                    theta += 2 * np.pi
                area += 0.5 * theta * r ** 2
    return area


def returnAreaOfCircleInDomainAroundPoint(
    index, points, r, domain_x, domain_y
):
    point = points[index, :]
    return returnAreaOfCircleInDomain(point[0], point[1], r, domain_x, domain_y)


def getAnnulusAreasAroundPoints(points_i, dr, maxR, domain_x, domain_y):
    vfunc = np.vectorize(
        returnAreaOfCircleInDomainAroundPoint,
        excluded=["points", "domain_x", "domain_y"],
    )
    PCF_radii_lower = np.arange(0, maxR, dr)
    PCF_radii_upper = np.arange(dr, maxR + dr, dr)
    all_areas = np.zeros((len(points_i), len(PCF_radii_lower)))

    for annulus in range(len(PCF_radii_lower)):
        inner = PCF_radii_lower[annulus]
        outer = PCF_radii_upper[annulus]
        areas_in = vfunc(
            index=np.arange(len(points_i)),
            points=points_i,
            r=inner,
            domain_x=domain_x,
            domain_y=domain_y,
        )
        areas_out = vfunc(
            index=np.arange(len(points_i)),
            points=points_i,
            r=outer,
            domain_x=domain_x,
            domain_y=domain_y,
        )
        all_areas[:, annulus] = areas_out - areas_in
    return all_areas


def crossPCF(distances_AtoB, areas_A, density_B, dr_mum, maxR_mum):
    N_A = distances_AtoB.shape[0]
    PCF_radii_lower = np.arange(0, maxR_mum, dr_mum)
    PCF_radii_upper = np.arange(dr_mum, maxR_mum + dr_mum, dr_mum)

    crossPCF_AtoB = np.ones((len(PCF_radii_lower), 1))
    contributions = np.zeros((N_A, len(PCF_radii_lower)))
    for annulus in range(len(PCF_radii_lower)):
        inner = PCF_radii_lower[annulus]
        outer = PCF_radii_upper[annulus]
        distance_mask = (distances_AtoB > inner) & (distances_AtoB <= outer)
        for i in range(N_A):
            fill_indices = np.where(distance_mask[i, :])[0]
            if areas_A[i, annulus] == 0:
                continue
            contribution = len(fill_indices) / (density_B * areas_A[i, annulus])
            crossPCF_AtoB[annulus] += contribution
            contributions[i, annulus] += contribution
        crossPCF_AtoB[annulus] = crossPCF_AtoB[annulus] / N_A
    return PCF_radii_lower, crossPCF_AtoB, contributions


def plot_paircorrelation_clustermap(
    summary: pd.DataFrame,
    *,
    condition: Optional[str] = None,
    populations: Optional[Sequence[str]] = None,
    percentile: float = 95.0,
    cmap: Union[str, "Colormap"] = "coolwarm",
    cluster: bool = True,
    figsize: Tuple[int, int] = (7, 5),
    cbar_kws: Optional[Dict[str, float]] = None,
) -> sns.matrix.ClusterGrid:
    """Plot a clustermap of g(r) with homotypic and significance annotations.

    Parameters
    ----------
    summary
        Output table from ``run_paircorrelation_at_distance`` (single condition).
    condition
        Optional condition name to filter before plotting.
    populations
        Optional ordered subset of populations to display on both axes.
    percentile
        Percentile (0-100) used to set vmin/vmax from off-diagonal g(r) values.
    cmap
        Matplotlib colormap name or object. Defaults to ``"coolwarm"``.
    cluster
        Whether to allow seaborn to cluster rows/columns.
    figsize
        Tuple passed to seaborn for the resulting figure size.
    cbar_kws
        Extra colorbar keyword arguments (defaults mimic prior heatmaps).

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The clustermap object for further customization or saving.
    """

    required_cols = {"cell_type_1", "cell_type_2", "g_mean", "g_min", "g_max"}
    missing = required_cols.difference(summary.columns)
    if missing:
        raise ValueError(f"Summary table missing columns: {missing}")

    df = summary.copy()
    if condition is not None:
        df = df[df["condition"] == condition]
        if df.empty:
            raise ValueError(f"No rows found for condition '{condition}'.")

    pivot_mean = df.pivot(index="cell_type_1", columns="cell_type_2", values="g_mean")
    pivot_min = df.pivot(index="cell_type_1", columns="cell_type_2", values="g_min")
    pivot_max = df.pivot(index="cell_type_1", columns="cell_type_2", values="g_max")

    if populations:
        order = [pop for pop in populations if pop in pivot_mean.index]
        if not order:
            raise ValueError("None of the requested populations are present.")
        pivot_mean = pivot_mean.loc[order, order]
        pivot_min = pivot_min.reindex(index=order, columns=order)
        pivot_max = pivot_max.reindex(index=order, columns=order)

    off_diag = df[df["cell_type_1"] != df["cell_type_2"]]["g_mean"].dropna()
    if off_diag.empty:
        off_diag = df["g_mean"].dropna()
    if off_diag.empty:
        raise ValueError("Cannot determine vmin/vmax; g_mean column is empty.")

    lower_pct = max(0.0, 100.0 - percentile)
    vmin = float(np.percentile(off_diag, lower_pct))
    vmax = float(np.percentile(off_diag, percentile))

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
    default_cbar = {"fraction": 0.046, "pad": 0.04}
    if cbar_kws:
        default_cbar.update(cbar_kws)

    clustergrid = sns.clustermap(
        pivot_mean,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        square=True,
        figsize=figsize,
        cbar_kws=default_cbar,
        row_cluster=cluster,
        col_cluster=cluster,
    )

    reordered_min = pivot_min.reindex(
        index=clustergrid.data2d.index, columns=clustergrid.data2d.columns
    )
    reordered_max = pivot_max.reindex(
        index=clustergrid.data2d.index, columns=clustergrid.data2d.columns
    )

    _annotate_pcf_heatmap(clustergrid.ax_heatmap, clustergrid.data2d, reordered_min, reordered_max)
    clustergrid.ax_heatmap.set_xlabel("Cell Type 2")
    clustergrid.ax_heatmap.set_ylabel("Cell Type 1")
    return clustergrid


def _annotate_pcf_heatmap(ax, mean_data, min_data, max_data):
    """Overlay homotypic and significance symbols onto a clustermap."""

    for i, row_label in enumerate(mean_data.index):
        for j, col_label in enumerate(mean_data.columns):
            mean_val = mean_data.iloc[i, j]
            min_val = min_data.iloc[i, j] if min_data is not None else np.nan
            max_val = max_data.iloc[i, j] if max_data is not None else np.nan
            if pd.isna(mean_val):
                continue

            x = j + 0.5
            y = i + 0.5

            if row_label == col_label:
                ax.text(
                    x,
                    y,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    alpha=0.75,
                )

            star_needed = False
            if mean_val < 1 and max_val < 1:
                star_needed = True
            elif mean_val > 1 and min_val > 1:
                star_needed = True

            if star_needed:
                ax.text(
                    x,
                    y,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="black",
                    fontweight="bold",
                )
