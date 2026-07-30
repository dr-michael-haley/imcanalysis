"""Auditable figures and tables for MaxFuse target-to-reference matches."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.maxfuse_matching import safe_column_name


@dataclass
class MaxFuseReportArtifacts:
    """Files, warnings, and objective metrics produced by report generation."""

    figures: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, int | float | str | bool] = field(default_factory=dict)


def natural_order(values: Iterable[Any]) -> list[str]:
    """Sort numeric-looking categories numerically, otherwise alphabetically."""

    unique = pd.Series(list(values), dtype="string").dropna().unique().tolist()
    try:
        return sorted(map(str, unique), key=float)
    except ValueError:
        return sorted(map(str, unique))


def save_figure_variants(
    figure: Any,
    base_path: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Save one Matplotlib figure in the configured publication/report formats."""

    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in formats:
        path = base_path.with_suffix(f".{extension}")
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        paths.append(path)
    return paths


def concordance_tables(
    matches: pd.DataFrame,
    *,
    target_column: str,
    reference_column: str,
    score_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return counts, row fractions, and per-target-population mean scores."""

    required = {target_column, reference_column, "score"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise KeyError(f"Match table is missing concordance columns: {missing}")
    selected = matches.loc[
        matches["score"].gt(float(score_threshold)),
        [target_column, reference_column, "score"],
    ].dropna(subset=[target_column, reference_column])
    if selected.empty:
        raise ValueError(
            f"No matches exceed the report score threshold {score_threshold:g}"
        )
    selected = selected.copy()
    selected[target_column] = selected[target_column].astype(str)
    selected[reference_column] = selected[reference_column].astype(str)
    rows = natural_order(selected[target_column])
    columns = natural_order(selected[reference_column])
    counts = pd.crosstab(
        selected[target_column],
        selected[reference_column],
    ).reindex(index=rows, columns=columns, fill_value=0)
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    scores = (
        selected.groupby(target_column, observed=True)["score"]
        .agg(["mean", "median", "count"])
        .reindex(rows)
    )
    scores.index.name = target_column
    return counts, fractions, scores


def plot_annotated_heatmap(
    values: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annotation_format: str = ".2f",
) -> Any:
    """Create a dynamically sized annotated heatmap."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    width = max(6.0, min(28.0, 4.0 + 0.62 * values.shape[1]))
    height = max(4.0, min(28.0, 2.5 + 0.48 * values.shape[0]))
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        values,
        square=values.shape[0] == values.shape[1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=annotation_format,
        linewidths=0.4,
        cbar_kws={"label": colorbar_label, "shrink": 0.75},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(False)
    figure.tight_layout()
    return figure


def best_matches_by_reference(
    matches: pd.DataFrame,
    *,
    score_threshold: float,
) -> pd.DataFrame:
    """Choose the highest-scoring target projection for each reference cell."""

    selected = matches.loc[matches["score"].gt(float(score_threshold))].copy()
    return (
        selected.sort_values("score", kind="stable")
        .drop_duplicates("reference_index", keep="last")
        .sort_values("reference_index", kind="stable")
        .reset_index(drop=True)
    )


def deterministic_stratified_indices(
    labels: Sequence[Any],
    *,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Return a reproducible proportional sample that retains small categories."""

    series = (
        pd.Series(labels, dtype="object")
        .where(pd.notna(labels), "Missing")
        .astype(str)
        .reset_index(drop=True)
    )
    if len(series) <= int(max_points):
        return np.arange(len(series), dtype=np.int64)
    counts = series.value_counts(sort=False)
    if len(counts) >= int(max_points):
        rng = np.random.default_rng(int(seed))
        selected = [
            rng.choice(np.flatnonzero(series.to_numpy() == label), size=1)[0]
            for label in counts.sort_values(ascending=False, kind="stable")
            .head(int(max_points))
            .index
        ]
        return np.sort(np.asarray(selected, dtype=np.int64))
    exact = counts / counts.sum() * int(max_points)
    allocation = np.floor(exact).astype(int)
    allocation[counts.gt(0) & allocation.eq(0)] = 1
    while int(allocation.sum()) > int(max_points):
        candidates = allocation.loc[allocation.gt(1)].sort_values(
            ascending=False,
            kind="stable",
        )
        if candidates.empty:
            break
        allocation.loc[candidates.index[0]] -= 1
    remainder = int(max_points) - int(allocation.sum())
    if remainder > 0:
        order = (exact - np.floor(exact)).sort_values(
            ascending=False,
            kind="stable",
        )
        for label in order.index:
            if remainder <= 0:
                break
            if allocation[label] < counts[label]:
                allocation[label] += 1
                remainder -= 1
    rng = np.random.default_rng(int(seed))
    values = series.to_numpy()
    selected = [
        rng.choice(
            np.flatnonzero(values == label),
            size=min(int(count), int(counts[label])),
            replace=False,
        )
        for label, count in allocation.items()
        if count > 0
    ]
    return np.sort(np.concatenate(selected).astype(np.int64))


def _categorical_embedding(
    axis: Any,
    coordinates: np.ndarray,
    values: Sequence[Any],
    *,
    title: str,
    point_size: float,
) -> None:
    series = pd.Series(values, dtype="object").where(pd.notna(values), "Missing").astype(str)
    categories = natural_order(series)
    import matplotlib.pyplot as plt

    if len(categories) <= 20:
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(categories))))
    else:
        colors = plt.get_cmap("gist_ncar")(np.linspace(0, 1, max(1, len(categories))))
    for category, color in zip(categories, colors):
        mask = series.eq(category).to_numpy()
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=point_size,
            c=[color],
            linewidths=0,
            alpha=0.75,
            label=category,
            rasterized=True,
        )
    axis.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        markerscale=3,
        fontsize=max(5, 9 - len(categories) // 15),
    )
    axis.set_title(title)


def plot_umap_triptych(
    coordinates: np.ndarray,
    *,
    first_values: Sequence[Any],
    first_title: str,
    second_values: Sequence[Any],
    second_title: str,
    scores: Sequence[float],
    score_threshold: float,
    title: str,
) -> Any:
    """Plot original labels, transferred labels, and MaxFuse score."""

    import matplotlib.pyplot as plt

    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("UMAP coordinates must contain at least two columns")
    point_size = max(0.25, min(8.0, 80_000 / max(1, len(coordinates))))
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    _categorical_embedding(
        axes[0],
        coordinates,
        first_values,
        title=first_title,
        point_size=point_size,
    )
    _categorical_embedding(
        axes[1],
        coordinates,
        second_values,
        title=second_title,
        point_size=point_size,
    )
    scatter = axes[2].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=np.asarray(scores, dtype=float),
        s=point_size,
        linewidths=0,
        vmin=float(score_threshold),
        vmax=1.0,
        cmap="inferno",
        rasterized=True,
    )
    figure.colorbar(scatter, ax=axes[2], label="MaxFuse matching score")
    axes[2].set_title("Matching score")
    for axis in axes:
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def expression_group_means(
    reference: Any,
    *,
    reference_indices: np.ndarray,
    groups: Sequence[Any],
    genes: Sequence[str],
    layer: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw and per-gene min-max scaled expression summaries."""

    from scipy import sparse

    gene_index = pd.Index(reference.var_names.astype(str))
    positions = gene_index.get_indexer(list(genes))
    if np.any(positions < 0):
        missing = np.asarray(genes)[positions < 0].tolist()
        raise KeyError(f"Reference genes missing from AnnData: {missing[:10]}")
    matrix = reference.layers[layer] if layer is not None else reference.X
    matrix = matrix[np.asarray(reference_indices, dtype=np.int64), :][:, positions]
    group_series = pd.Series(groups, dtype="object").where(pd.notna(groups), "Missing").astype(str)
    rows: list[np.ndarray] = []
    order = natural_order(group_series)
    for group in order:
        mask = group_series.eq(group).to_numpy()
        selected = matrix[mask]
        if sparse.issparse(selected):
            mean = np.asarray(selected.mean(axis=0)).ravel()
        else:
            mean = np.asarray(selected).mean(axis=0)
        rows.append(np.asarray(mean, dtype=float))
    raw = pd.DataFrame(rows, index=order, columns=list(genes))
    denominator = (raw.max(axis=0) - raw.min(axis=0)).replace(0, np.nan)
    scaled = raw.subtract(raw.min(axis=0), axis=1).div(denominator, axis=1).fillna(0)
    return raw, scaled


def order_expression_matrix(values: pd.DataFrame) -> pd.DataFrame:
    """Cluster genes and groups on their summarized expression profiles."""

    from scipy.cluster.hierarchy import leaves_list, linkage

    rows = np.arange(values.shape[0])
    columns = np.arange(values.shape[1])
    if values.shape[0] > 1:
        rows = leaves_list(linkage(values.to_numpy(), method="ward"))
    if values.shape[1] > 1:
        columns = leaves_list(linkage(values.to_numpy().T, method="ward"))
    return values.iloc[rows, columns]


def plot_expression_matrix(values: pd.DataFrame, *, title: str) -> Any:
    import matplotlib.pyplot as plt
    import seaborn as sns

    width = max(9.0, min(26.0, 4.5 + 0.36 * values.shape[1]))
    height = max(5.0, min(24.0, 2.5 + 0.42 * values.shape[0]))
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        values,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.25,
        cbar_kws={"label": "Expression scaled within gene"},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Linked reference gene")
    axis.set_ylabel("Transferred target population")
    axis.tick_params(axis="x", labelrotation=75)
    figure.tight_layout()
    return figure


def load_gene_lists(
    specifications: Sequence[Any],
    *,
    resolve_path: Callable[[str], Path],
) -> pd.DataFrame:
    """Load historical wide CSVs or explicit long gene-list tables."""

    rows: list[pd.DataFrame] = []
    for specification in specifications:
        path = resolve_path(specification.path)
        if not path.is_file():
            raise FileNotFoundError(f"MaxFuse DEG gene list not found: {path}")
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator)
        list_name = specification.name or path.stem
        file_format = specification.format
        if file_format == "auto":
            file_format = (
                "long"
                if {
                    specification.gene_column,
                    specification.group_column,
                }.issubset(frame.columns)
                else "wide"
            )
        if file_format == "long":
            missing = sorted(
                {
                    specification.gene_column,
                    specification.group_column,
                }
                - set(frame.columns)
            )
            if missing:
                raise ValueError(f"Gene list {path} is missing long-format columns: {missing}")
            normalized = frame[
                [specification.group_column, specification.gene_column]
            ].rename(
                columns={
                    specification.group_column: "group",
                    specification.gene_column: "gene",
                }
            )
        else:
            normalized = frame.melt(var_name="group", value_name="gene")
        normalized["list_name"] = str(list_name)
        normalized["gene"] = normalized["gene"].astype("string").str.strip()
        normalized["group"] = normalized["group"].astype("string").str.strip()
        normalized = normalized.dropna(subset=["gene", "group"])
        normalized = normalized.loc[
            normalized["gene"].ne("") & normalized["group"].ne("")
        ]
        rows.append(normalized[["list_name", "group", "gene"]])
    if not rows:
        return pd.DataFrame(columns=["list_name", "group", "gene"])
    return pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _gene_colors(gene_lists: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    import matplotlib.pyplot as plt

    if gene_lists.empty:
        return {}, {}
    groups = (
        gene_lists.assign(key=lambda frame: frame["list_name"] + ": " + frame["group"])
        ["key"]
        .drop_duplicates()
        .tolist()
    )
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(groups))))
    from matplotlib.colors import to_hex

    group_colors = {
        group: to_hex(color) for group, color in zip(groups, palette)
    }
    membership: dict[str, list[str]] = {}
    for row in gene_lists.itertuples(index=False):
        membership.setdefault(str(row.gene), []).append(
            f"{row.list_name}: {row.group}"
        )
    gene_colors = {
        gene: (
            group_colors[groups_for_gene[0]]
            if len(set(groups_for_gene)) == 1
            else "#000000"
        )
        for gene, groups_for_gene in membership.items()
    }
    return gene_colors, group_colors


def run_reference_degs(
    reference: Any,
    matched_reference: pd.DataFrame,
    *,
    group_column: str,
    layer: str | None,
    min_cells: int,
) -> pd.DataFrame:
    """Run Scanpy Wilcoxon DEGs on unique, thresholded matched reference cells."""

    import scanpy as sc

    groups = matched_reference[group_column].astype("string")
    counts = groups.value_counts()
    retained_groups = counts.loc[counts.ge(int(min_cells))].index
    eligible = groups.isin(retained_groups)
    selected = matched_reference.loc[eligible].copy()
    if selected[group_column].nunique(dropna=True) < 2:
        raise ValueError(
            "DEG analysis requires at least two matched target populations with "
            f"{min_cells} reference cells each"
        )
    adata = reference[
        selected["reference_index"].to_numpy(dtype=np.int64),
        :,
    ].copy()
    adata.obs["_maxfuse_target_population"] = (
        selected[group_column].astype(str).to_numpy()
    )
    adata.obs["_maxfuse_target_population"] = adata.obs[
        "_maxfuse_target_population"
    ].astype("category")
    sc.tl.rank_genes_groups(
        adata,
        groupby="_maxfuse_target_population",
        method="wilcoxon",
        key_added="maxfuse_wilcoxon",
        use_raw=False,
        layer=layer,
    )
    result = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key="maxfuse_wilcoxon",
    )
    result.insert(0, "rank", result.groupby("group", observed=True).cumcount() + 1)
    return result


def plot_ranked_genes(
    degs: pd.DataFrame,
    *,
    top_n: int,
    panel_genes: Sequence[str],
    gene_lists: pd.DataFrame,
) -> tuple[Any, Any | None]:
    """Plot ranked genes with panel markers and supplied gene-list colours."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    groups = natural_order(degs["group"])
    columns = min(4, max(1, len(groups)))
    rows = math.ceil(len(groups) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, max(4.5, 0.34 * top_n) * rows),
        squeeze=False,
    )
    gene_colors, group_colors = _gene_colors(gene_lists)
    panel_set = set(map(str, panel_genes))
    for axis, group in zip(axes.ravel(), groups):
        selected = (
            degs.loc[degs["group"].astype(str).eq(group)]
            .sort_values("rank", kind="stable")
            .head(int(top_n))
            .iloc[::-1]
        )
        labels = [
            f"{gene} #" if str(gene) in panel_set else str(gene)
            for gene in selected["names"]
        ]
        colors = [gene_colors.get(str(gene), "#777777") for gene in selected["names"]]
        axis.barh(
            np.arange(len(selected)),
            selected["scores"].to_numpy(dtype=float),
            color=colors,
            alpha=0.9,
        )
        axis.set_yticks(np.arange(len(selected)), labels=labels)
        for tick, color in zip(axis.get_yticklabels(), colors):
            tick.set_color(color)
            if color != "#777777":
                tick.set_fontweight("bold")
        axis.set_title(group)
        axis.set_xlabel("Wilcoxon score")
        axis.grid(axis="x", alpha=0.2)
    for axis in axes.ravel()[len(groups) :]:
        axis.axis("off")
    figure.suptitle("Reference RNA DEGs grouped by transferred target population")
    figure.tight_layout()

    legend = None
    if group_colors:
        handles = [
            Line2D([0], [0], marker="s", linestyle="", color=color, markersize=10)
            for color in group_colors.values()
        ]
        labels = list(group_colors)
        handles.append(
            Line2D([0], [0], marker="s", linestyle="", color="#000000", markersize=10)
        )
        labels.append("Gene in multiple supplied groups")
        legend, axis = plt.subplots(
            figsize=(max(7, 0.22 * max(map(len, labels))), 1.5 + 0.32 * len(labels))
        )
        axis.legend(handles, labels, loc="center", frameon=False, ncol=2)
        axis.axis("off")
        legend.tight_layout()
    return figure, legend


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    if len(pvalues) == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def gene_list_enrichment(
    degs: pd.DataFrame,
    gene_lists: pd.DataFrame,
    *,
    universe: Sequence[str],
    top_n: int = 100,
) -> pd.DataFrame:
    """Test top-DEG overlap with supplied lists using a hypergeometric model."""

    if gene_lists.empty:
        return pd.DataFrame()
    from scipy.stats import hypergeom

    universe_set = set(map(str, universe))
    gene_sets = {
        f"{name}: {group}": set(values.astype(str)) & universe_set
        for (name, group), values in gene_lists.groupby(
            ["list_name", "group"],
            observed=True,
        )["gene"]
    }
    rows: list[dict[str, Any]] = []
    for group, frame in degs.groupby("group", observed=True):
        top = (
            frame.sort_values("rank", kind="stable")
            .head(int(top_n))["names"]
            .astype(str)
        )
        top_set = set(top) & universe_set
        for gene_set_name, genes in gene_sets.items():
            overlap = top_set & genes
            pvalue = float(
                hypergeom.sf(
                    len(overlap) - 1,
                    len(universe_set),
                    len(genes),
                    len(top_set),
                )
            )
            rows.append(
                {
                    "population": str(group),
                    "gene_set": gene_set_name,
                    "top_genes_tested": len(top_set),
                    "gene_set_size": len(genes),
                    "overlap_count": len(overlap),
                    "overlap_genes": ";".join(sorted(overlap)),
                    "pvalue": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    result["pvalue_adjusted"] = _benjamini_hochberg(
        result["pvalue"].to_numpy(dtype=float)
    )
    return result


def plot_gene_list_enrichment(enrichment: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt
    import seaborn as sns

    values = enrichment.pivot(
        index="population",
        columns="gene_set",
        values="pvalue_adjusted",
    ).fillna(1.0)
    display = -np.log10(values.clip(lower=1e-300))
    width = max(8.0, min(28.0, 4.0 + 0.5 * display.shape[1]))
    height = max(4.0, min(24.0, 2.5 + 0.45 * display.shape[0]))
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        display,
        cmap="magma",
        linewidths=0.25,
        cbar_kws={"label": "-log10(BH-adjusted P)"},
        ax=axis,
    )
    axis.set_title("Top-DEG enrichment in supplied gene lists")
    axis.set_xlabel("Gene list")
    axis.set_ylabel("Transferred target population")
    figure.tight_layout()
    return figure


def _write_table(
    frame: pd.DataFrame,
    path: Path,
    artifacts: MaxFuseReportArtifacts,
    *,
    index: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    artifacts.tables.append(path)
    return path


def _score_and_coverage_outputs(
    matches: pd.DataFrame,
    target: Any,
    settings: Any,
    *,
    figures_dir: Path,
    tables_dir: Path,
    artifacts: MaxFuseReportArtifacts,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    thresholds = sorted(set(map(float, settings.sensitivity_thresholds)))
    sensitivity = pd.DataFrame(
        {
            "score_threshold": thresholds,
            "passing_matches": [
                int(matches["score"].gt(value).sum()) for value in thresholds
            ],
        }
    )
    sensitivity["target_cells"] = int(target.n_obs)
    sensitivity["passing_coverage"] = (
        sensitivity["passing_matches"] / max(1, int(target.n_obs))
    )
    _write_table(
        sensitivity,
        tables_dir / "score_threshold_sensitivity.csv",
        artifacts,
        index=False,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(matches["score"].dropna(), bins=80, color="#4477AA", alpha=0.9)
    axes[0].axvline(
        float(settings.report_score_threshold),
        color="#CC3311",
        linestyle="--",
        label=f"report threshold = {settings.report_score_threshold:g}",
    )
    axes[0].set_xlabel("MaxFuse matching score")
    axes[0].set_ylabel("Retained matches")
    axes[0].legend(frameon=False)
    sns.lineplot(
        data=sensitivity,
        x="score_threshold",
        y="passing_coverage",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Fraction of all target cells")
    axes[1].set_title("Post-hoc threshold sensitivity")
    figure.tight_layout()
    artifacts.figures.extend(
        save_figure_variants(
            figure,
            figures_dir / "matching_score_and_threshold_sensitivity",
            formats=settings.figure_formats,
            dpi=settings.figure_dpi,
        )
    )
    plt.close(figure)


def _population_coverage(
    matches: pd.DataFrame,
    target: Any,
    settings: Any,
    *,
    tables_dir: Path,
    figures_dir: Path,
    artifacts: MaxFuseReportArtifacts,
) -> None:
    import matplotlib.pyplot as plt

    population_column = settings.target_population_obs
    target_populations = (
        target.obs[population_column]
        .astype(object)
        .where(target.obs[population_column].notna(), "Missing")
        .astype(str)
    )
    totals = target_populations.value_counts(dropna=False)
    matched = matches.copy()
    matched["target_population"] = (
        matched["target_population"]
        .astype(object)
        .where(matched["target_population"].notna(), "Missing")
        .astype(str)
    )
    summary = matched.groupby("target_population", observed=True)["score"].agg(
        matched_cells="count",
        mean_score="mean",
        median_score="median",
    )
    passing = (
        matched.loc[
            matched["score"].gt(float(settings.report_score_threshold))
        ]["target_population"]
        .value_counts()
        .rename("passing_cells")
    )
    summary = summary.reindex(totals.index)
    summary = summary.join(passing, how="left")
    summary["matched_cells"] = summary["matched_cells"].fillna(0)
    summary["passing_cells"] = summary["passing_cells"].fillna(0)
    summary["total_target_cells"] = totals
    summary["match_coverage"] = (
        summary["matched_cells"] / summary["total_target_cells"].replace(0, np.nan)
    )
    summary["passing_coverage"] = (
        summary["passing_cells"] / summary["total_target_cells"].replace(0, np.nan)
    )
    summary = summary.reindex(natural_order(summary.index))
    _write_table(
        summary,
        tables_dir / "population_match_coverage.csv",
        artifacts,
    )
    figure, axes = plt.subplots(2, 1, figsize=(max(9, 0.45 * len(summary)), 9))
    summary[["match_coverage", "passing_coverage"]].plot.bar(ax=axes[0])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Fraction of population")
    axes[0].set_title("Target population matching coverage")
    summary[["mean_score", "median_score"]].plot.bar(ax=axes[1])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("MaxFuse matching score")
    axes[1].set_title("Target population matching scores")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=75)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    artifacts.figures.extend(
        save_figure_variants(
            figure,
            figures_dir / "population_coverage_and_scores",
            formats=settings.figure_formats,
            dpi=settings.figure_dpi,
        )
    )
    plt.close(figure)


def _stratified_group_summary(
    matches: pd.DataFrame,
    target: Any,
    *,
    obs_key: str,
    score_threshold: float,
) -> pd.DataFrame:
    lookup = pd.DataFrame(
        {
            "target_index": np.arange(target.n_obs, dtype=np.int64),
            obs_key: (
                target.obs[obs_key]
                .astype(object)
                .where(target.obs[obs_key].notna(), "Missing")
                .astype(str)
                .to_numpy()
            ),
        }
    )
    totals = lookup[obs_key].value_counts(dropna=False).rename("total_target_cells")
    aligned = matches[["target_index", "score"]].merge(
        lookup,
        on="target_index",
        how="left",
        validate="one_to_one",
    )
    summary = aligned.groupby(obs_key, observed=True)["score"].agg(
        matched_cells="count",
        mean_score="mean",
        median_score="median",
    )
    passing = (
        aligned.loc[aligned["score"].gt(float(score_threshold)), obs_key]
        .value_counts()
        .rename("passing_cells")
    )
    summary = summary.join(totals, how="outer").join(passing, how="left")
    summary["matched_cells"] = summary["matched_cells"].fillna(0)
    summary["passing_cells"] = summary["passing_cells"].fillna(0)
    summary["match_coverage"] = summary["matched_cells"] / summary[
        "total_target_cells"
    ].replace(0, np.nan)
    summary["passing_coverage"] = summary["passing_cells"] / summary[
        "total_target_cells"
    ].replace(0, np.nan)
    return summary


def _umap_outputs(
    reference: Any,
    target: Any,
    matches: pd.DataFrame,
    settings: Any,
    *,
    figures_dir: Path,
    tables_dir: Path,
    artifacts: MaxFuseReportArtifacts,
) -> None:
    import matplotlib.pyplot as plt

    threshold = float(settings.report_score_threshold)
    primary_reference_column = (
        f"reference_{safe_column_name(settings.reference_transfer_obs[0])}"
    )
    target_matches = matches.loc[matches["score"].gt(threshold)].copy()
    if settings.target_umap_key in target.obsm and not target_matches.empty:
        sample = deterministic_stratified_indices(
            target_matches["target_population"].to_numpy(),
            max_points=int(settings.max_umap_points),
            seed=int(settings.seed),
        )
        plotted = target_matches.iloc[sample]
        target_indices = plotted["target_index"].to_numpy(dtype=np.int64)
        figure = plot_umap_triptych(
            np.asarray(target.obsm[settings.target_umap_key])[target_indices, :2],
            first_values=target.obs.iloc[target_indices][
                settings.target_population_obs
            ].to_numpy(),
            first_title=f"Original {settings.target_population_obs}",
            second_values=plotted[primary_reference_column].to_numpy(),
            second_title=f"Transferred {settings.reference_transfer_obs[0]}",
            scores=plotted["score"].to_numpy(),
            score_threshold=threshold,
            title="Target IMC UMAP",
        )
        artifacts.figures.extend(
            save_figure_variants(
                figure,
                figures_dir / "umap_target_space",
                formats=settings.figure_formats,
                dpi=settings.figure_dpi,
            )
        )
        plt.close(figure)
        _write_table(
            plotted[
                [
                    "target_obs_name",
                    "reference_obs_name",
                    "score",
                    "target_population",
                    primary_reference_column,
                ]
            ],
            tables_dir / "umap_target_plotted_cells.csv.gz",
            artifacts,
            index=False,
        )
    else:
        artifacts.warnings.append(
            f"Target AnnData lacks obsm[{settings.target_umap_key!r}] or has no "
            "matches above the report threshold; skipped target UMAP."
        )

    reference_matches = best_matches_by_reference(
        matches,
        score_threshold=threshold,
    )
    if settings.reference_umap_key in reference.obsm and not reference_matches.empty:
        sample = deterministic_stratified_indices(
            reference_matches["target_population"].to_numpy(),
            max_points=int(settings.max_umap_points),
            seed=int(settings.seed),
        )
        plotted = reference_matches.iloc[sample]
        reference_indices = plotted["reference_index"].to_numpy(dtype=np.int64)
        original_reference_column = settings.reference_transfer_obs[0]
        figure = plot_umap_triptych(
            np.asarray(reference.obsm[settings.reference_umap_key])[
                reference_indices, :2
            ],
            first_values=reference.obs.iloc[reference_indices][
                original_reference_column
            ].to_numpy(),
            first_title=f"Original {original_reference_column}",
            second_values=plotted["target_population"].to_numpy(),
            second_title=f"Transferred {settings.target_population_obs}",
            scores=plotted["score"].to_numpy(),
            score_threshold=threshold,
            title="Reference RNA UMAP",
        )
        artifacts.figures.extend(
            save_figure_variants(
                figure,
                figures_dir / "umap_reference_space",
                formats=settings.figure_formats,
                dpi=settings.figure_dpi,
            )
        )
        plt.close(figure)
        _write_table(
            plotted[
                [
                    "reference_obs_name",
                    "target_obs_name",
                    "score",
                    "target_population",
                    primary_reference_column,
                ]
            ],
            tables_dir / "umap_reference_plotted_cells.csv.gz",
            artifacts,
            index=False,
        )
    else:
        artifacts.warnings.append(
            f"Reference AnnData lacks obsm[{settings.reference_umap_key!r}] or has "
            "no unique matches above the report threshold; skipped reference UMAP."
        )


def _expression_outputs(
    reference: Any,
    matches: pd.DataFrame,
    retained_mapping: pd.DataFrame,
    settings: Any,
    *,
    figures_dir: Path,
    tables_dir: Path,
    artifacts: MaxFuseReportArtifacts,
) -> tuple[pd.DataFrame, list[str]]:
    import matplotlib.pyplot as plt

    matched_reference = best_matches_by_reference(
        matches,
        score_threshold=float(settings.report_score_threshold),
    )
    if matched_reference.empty:
        artifacts.warnings.append(
            "No unique reference matches exceeded the report threshold; skipped "
            "linked-gene plots and DEGs."
        )
        return matched_reference, []
    genes = retained_mapping[settings.reference_feature_column].astype(str).tolist()
    raw, scaled = expression_group_means(
        reference,
        reference_indices=matched_reference["reference_index"].to_numpy(dtype=np.int64),
        groups=matched_reference["target_population"].to_numpy(),
        genes=genes,
        layer=settings.reference_layer,
    )
    ordered = order_expression_matrix(scaled)
    _write_table(
        raw,
        tables_dir / "linked_gene_population_means.csv",
        artifacts,
    )
    _write_table(
        ordered,
        tables_dir / "linked_gene_population_means_scaled.csv",
        artifacts,
    )
    figure = plot_expression_matrix(
        ordered,
        title="Linked reference genes grouped by transferred target population",
    )
    artifacts.figures.extend(
        save_figure_variants(
            figure,
            figures_dir / "linked_gene_matrixplot",
            formats=settings.figure_formats,
            dpi=settings.figure_dpi,
        )
    )
    plt.close(figure)

    if settings.plot_stacked_violin:
        try:
            import scanpy as sc

            sample = deterministic_stratified_indices(
                matched_reference["target_population"].to_numpy(),
                max_points=min(
                    len(matched_reference),
                    max(10_000, 2_000 * matched_reference["target_population"].nunique()),
                ),
                seed=int(settings.seed),
            )
            plotted = matched_reference.iloc[sample]
            adata = reference[
                plotted["reference_index"].to_numpy(dtype=np.int64),
                ordered.columns.tolist(),
            ].copy()
            adata.obs["_maxfuse_target_population"] = (
                plotted["target_population"].astype(str).to_numpy()
            )
            adata.obs["_maxfuse_target_population"] = adata.obs[
                "_maxfuse_target_population"
            ].astype("category")
            violin = sc.pl.stacked_violin(
                adata,
                var_names=ordered.columns.tolist(),
                groupby="_maxfuse_target_population",
                standard_scale="var",
                use_raw=False,
                layer=settings.reference_layer,
                return_fig=True,
                show=False,
            )
            for extension in settings.figure_formats:
                path = (figures_dir / "linked_gene_stacked_violin").with_suffix(
                    f".{extension}"
                )
                violin.savefig(path, dpi=int(settings.figure_dpi), bbox_inches="tight")
                artifacts.figures.append(path)
            plt.close("all")
        except (ValueError, KeyError, RuntimeError) as error:
            artifacts.warnings.append(f"Skipped linked-gene stacked violin: {error}")
    return matched_reference, genes


def _deg_outputs(
    reference: Any,
    matched_reference: pd.DataFrame,
    panel_genes: Sequence[str],
    gene_lists: pd.DataFrame,
    settings: Any,
    *,
    figures_dir: Path,
    tables_dir: Path,
    artifacts: MaxFuseReportArtifacts,
) -> None:
    import matplotlib.pyplot as plt

    if not settings.run_degs or matched_reference.empty:
        return
    try:
        degs = run_reference_degs(
            reference,
            matched_reference,
            group_column="target_population",
            layer=settings.reference_layer,
            min_cells=int(settings.deg_min_cells),
        )
    except ValueError as error:
        artifacts.warnings.append(f"Skipped MaxFuse DEG analysis: {error}")
        return
    _write_table(
        degs,
        tables_dir / "reference_degs_by_transferred_target_population.csv.gz",
        artifacts,
        index=False,
    )
    figure, legend = plot_ranked_genes(
        degs,
        top_n=int(settings.deg_top_genes),
        panel_genes=panel_genes,
        gene_lists=gene_lists,
    )
    artifacts.figures.extend(
        save_figure_variants(
            figure,
            figures_dir / "reference_degs_ranked",
            formats=settings.figure_formats,
            dpi=settings.figure_dpi,
        )
    )
    plt.close(figure)
    if legend is not None:
        artifacts.figures.extend(
            save_figure_variants(
                legend,
                figures_dir / "reference_degs_gene_list_legend",
                formats=settings.figure_formats,
                dpi=settings.figure_dpi,
            )
        )
        plt.close(legend)
    if gene_lists.empty:
        return
    enrichment = gene_list_enrichment(
        degs,
        gene_lists,
        universe=reference.var_names.astype(str),
    )
    _write_table(
        enrichment,
        tables_dir / "reference_deg_gene_list_enrichment.csv",
        artifacts,
        index=False,
    )
    enrichment_figure = plot_gene_list_enrichment(enrichment)
    artifacts.figures.extend(
        save_figure_variants(
            enrichment_figure,
            figures_dir / "reference_deg_gene_list_enrichment",
            formats=settings.figure_formats,
            dpi=settings.figure_dpi,
        )
    )
    plt.close(enrichment_figure)


def generate_maxfuse_report(
    reference: Any,
    target: Any,
    matches: pd.DataFrame,
    retained_mapping: pd.DataFrame,
    settings: Any,
    *,
    figures_dir: Path,
    tables_dir: Path,
    resolve_path: Callable[[str], Path],
) -> MaxFuseReportArtifacts:
    """Generate the default notebook-derived MaxFuse report suite."""

    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts = MaxFuseReportArtifacts()
    threshold = float(settings.report_score_threshold)
    primary_reference_column = (
        f"reference_{safe_column_name(settings.reference_transfer_obs[0])}"
    )
    try:
        counts, fractions, score_summary = concordance_tables(
            matches,
            target_column="target_population",
            reference_column=primary_reference_column,
            score_threshold=threshold,
        )
    except ValueError as error:
        artifacts.warnings.append(
            f"Skipped annotated concordance and mean-score heatmaps: {error}"
        )
    else:
        _write_table(
            counts,
            tables_dir / "population_concordance_counts.csv",
            artifacts,
        )
        _write_table(
            fractions,
            tables_dir / "population_concordance_row_fractions.csv",
            artifacts,
        )
        _write_table(
            score_summary,
            tables_dir / "mean_matching_score_by_target_population.csv",
            artifacts,
        )
        concordance = plot_annotated_heatmap(
            fractions,
            title=(
                "Target populations versus transferred reference labels "
                f"(score > {threshold:g})"
            ),
            xlabel=f"Transferred {settings.reference_transfer_obs[0]}",
            ylabel=settings.target_population_obs,
            colorbar_label="Fraction within target population",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        artifacts.figures.extend(
            save_figure_variants(
                concordance,
                figures_dir / "population_concordance_annotated",
                formats=settings.figure_formats,
                dpi=settings.figure_dpi,
            )
        )
        plt.close(concordance)
        score_heatmap = plot_annotated_heatmap(
            score_summary[["mean"]],
            title="Mean MaxFuse matching score by target population",
            xlabel="Mean matching score",
            ylabel=settings.target_population_obs,
            colorbar_label="Mean MaxFuse matching score",
            cmap="inferno",
            vmin=threshold,
            vmax=1,
        )
        artifacts.figures.extend(
            save_figure_variants(
                score_heatmap,
                figures_dir / "mean_matching_score_annotated",
                formats=settings.figure_formats,
                dpi=settings.figure_dpi,
            )
        )
        plt.close(score_heatmap)

    _score_and_coverage_outputs(
        matches,
        target,
        settings,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        artifacts=artifacts,
    )
    _population_coverage(
        matches,
        target,
        settings,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        artifacts=artifacts,
    )
    for obs_key in dict.fromkeys([settings.sample_obs, settings.roi_obs]):
        if obs_key and obs_key in target.obs.columns:
            summary = _stratified_group_summary(
                matches,
                target,
                obs_key=obs_key,
                score_threshold=threshold,
            )
            _write_table(
                summary,
                tables_dir / f"coverage_by_{safe_column_name(obs_key)}.csv",
                artifacts,
            )
    _umap_outputs(
        reference,
        target,
        matches,
        settings,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        artifacts=artifacts,
    )
    matched_reference, panel_genes = _expression_outputs(
        reference,
        matches,
        retained_mapping,
        settings,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        artifacts=artifacts,
    )
    gene_lists = load_gene_lists(settings.gene_lists, resolve_path=resolve_path)
    if not gene_lists.empty:
        _write_table(
            gene_lists,
            tables_dir / "supplied_gene_lists_normalized.csv",
            artifacts,
            index=False,
        )
    _deg_outputs(
        reference,
        matched_reference,
        panel_genes,
        gene_lists,
        settings,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        artifacts=artifacts,
    )
    artifacts.metrics.update(
        {
            "target_cells": int(target.n_obs),
            "reference_cells": int(reference.n_obs),
            "retained_matches": len(matches),
            "target_match_coverage": float(len(matches) / max(1, int(target.n_obs))),
            "matches_above_report_threshold": int(
                matches["score"].gt(threshold).sum()
            ),
            "median_match_score": float(matches["score"].median()),
            "report_figures": len(artifacts.figures),
            "report_tables": len(artifacts.tables),
        }
    )
    return artifacts


__all__ = [
    "MaxFuseReportArtifacts",
    "best_matches_by_reference",
    "concordance_tables",
    "deterministic_stratified_indices",
    "expression_group_means",
    "gene_list_enrichment",
    "generate_maxfuse_report",
    "load_gene_lists",
    "natural_order",
    "order_expression_matrix",
    "plot_annotated_heatmap",
    "plot_expression_matrix",
    "plot_gene_list_enrichment",
    "plot_ranked_genes",
    "plot_umap_triptych",
    "run_reference_degs",
    "save_figure_variants",
]
