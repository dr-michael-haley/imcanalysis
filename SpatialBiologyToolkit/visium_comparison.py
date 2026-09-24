"""Read-only, notebook-facing comparisons of registered IMC and Visium spots.

These functions consume tables rather than fitted models, with optional image
backgrounds for spatial plots. Missing IMC coverage is
never interpreted as zero cells. Correlation p-values assume independent spots;
they are descriptive diagnostics for spatial data, not patient-level inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse, stats


def _frame(frame: pd.DataFrame, *, nonnegative: bool = False) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Expected a DataFrame with unique row and column identities")
    if frame.empty or frame.index.hasnans or frame.columns.hasnans:
        raise ValueError("Tables must be nonempty and have nonmissing identities")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Table values must be finite; subset to observed spots first")
    if nonnegative and (values < 0).any():
        raise ValueError("Counts/abundances must be nonnegative")
    return frame.astype(float)


def read_cell2location_tables(
    folder: str | Path,
    observations: pd.DataFrame,
    *,
    summary: str = "mean",
    library_key: str = "library_id",
    barcode_key: str = "Barcode",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align SBT mapping tables to input observation IDs using library AND barcode.

    ``folder`` is the exported ``tables`` directory. Compressed CSV is preferred.
    Extra export spots are allowed (e.g. a requested cohort subset), but every
    requested spot must match exactly once. Returns abundance and QC tables in
    the input observation order; the original export ID is retained in QC.
    """
    if summary not in {"mean", "q05"}:
        raise ValueError("summary must be 'mean' or 'q05'")
    if not observations.index.is_unique or observations.index.hasnans or observations.empty:
        raise ValueError("Observation identities must be unique, nonmissing and nonempty")
    folder = Path(folder)
    path = folder / f"mapping_{summary}_abundance.csv.gz"
    if not path.exists():
        path = path.with_suffix("")
    abundance = _frame(pd.read_csv(path, index_col=0), nonnegative=True)
    qc = pd.read_csv(folder / "mapping_spot_qc.csv", index_col=0)
    if not qc.index.is_unique or set(abundance.index) != set(qc.index):
        raise ValueError("Abundance and QC spot identities must match one-to-one")
    source = observations[[library_key, barcode_key]]
    target = qc[["library_id", "barcode"]]
    if source.isna().any().any() or target.isna().any().any():
        raise ValueError("Library/barcode identities cannot be missing")
    source_ids = pd.MultiIndex.from_frame(source.astype(str))
    target_ids = pd.MultiIndex.from_frame(target.astype(str))
    if not source_ids.is_unique or not target_ids.is_unique:
        raise ValueError("Duplicate (library, barcode) identities")
    positions = target_ids.get_indexer(source_ids)
    if (positions < 0).any():
        raise ValueError(f"{(positions < 0).sum()} requested spots absent from cell2location export")
    aligned_qc = qc.iloc[positions].copy()
    aligned = abundance.loc[aligned_qc.index].copy()
    aligned_qc.insert(0, "export_spot_id", aligned_qc.index)
    aligned.index = aligned_qc.index = observations.index.copy()
    return aligned, aligned_qc


def aggregate_populations(
    abundance: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    source: str = "annotation_granular",
    target: str = "annotation_coarse",
) -> pd.DataFrame:
    """Sum populations using a complete many-to-one annotation crosswalk.

    Posterior means are additive. Sums of marginal quantiles are NOT quantiles
    of the aggregate posterior; use means for coarse-resolution comparisons.
    """
    abundance = _frame(abundance, nonnegative=True)
    pairs = mapping[[source, target]].drop_duplicates()
    if pairs.isna().any().any() or pairs[source].duplicated().any():
        raise ValueError("Every source annotation must map to exactly one nonmissing target")
    labels = pairs.set_index(source)[target].reindex(abundance.columns)
    if labels.isna().any():
        raise ValueError(f"Unmapped populations: {labels.index[labels.isna()].tolist()}")
    result = abundance.T.groupby(labels, sort=False, observed=True).sum().T
    result.columns.name = None
    return result


def spot_population_counts(
    observations: pd.DataFrame, *, spot_key: str, population_key: str
) -> pd.DataFrame:
    """Count labelled cells in observed spots, without inventing unobserved zeros.

    Missing spot assignments are excluded. Missing population labels in assigned
    cells raise, rather than silently altering the denominator. Input rows must
    be unique cell identities. Filter artifacts explicitly before calling.
    """
    if not observations.index.is_unique:
        raise ValueError("Cell identities must be unique")
    assigned = observations.loc[observations[spot_key].notna()]
    if assigned[population_key].isna().any():
        raise ValueError("Assigned cells have missing population labels")
    result = pd.crosstab(assigned[spot_key], assigned[population_key])
    return _frame(result, nonnegative=True)


def spot_filter(counts: pd.DataFrame, *, min_cells: int = 5, purity: float = 0) -> pd.Series:
    """Select spots with >= min_cells and maximum population fraction >= purity.

    Purity is defined over exactly the columns supplied. Use the complete label
    level before any tumour-only or selected-population subsetting. The old
    notebook used >5; pass min_cells=6 to reproduce that count threshold.
    """
    counts = _frame(counts, nonnegative=True)
    if not isinstance(min_cells, int) or min_cells < 1 or not 0 <= purity <= 1:
        raise ValueError("min_cells must be a positive integer and purity must be in [0, 1]")
    total = counts.sum(axis=1)
    return (total.ge(min_cells) & counts.max(axis=1).div(total).ge(purity)).rename("included")


def population_fractions(counts: pd.DataFrame) -> pd.DataFrame:
    """Divide by the complete per-spot total; reject zero-total rows."""
    counts = _frame(counts, nonnegative=True)
    total = counts.sum(axis=1)
    if total.le(0).any():
        raise ValueError("Fractions are undefined for zero-total spots")
    return counts.div(total, axis=0)


def correlate_spot_features(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    method: str = "pearson",
    groups: pd.Series | None = None,
    min_spots: int = 4,
) -> pd.DataFrame:
    """All cross-table correlations with explicit identity alignment and BH FDR.

    Uses the spot intersection, never row order. Constant features return NaN.
    With ``groups``, centre within each group (rank within group first for
    Spearman). This is a within-group association, not equal case weighting;
    p/q are omitted because ordinary independent-spot tests are inappropriate.
    Otherwise p/q are nominal independent-spot diagnostics, BH-corrected over
    finite cross-table tests in this call. Inspect separate cases for replication.
    """
    left, right = _frame(left), _frame(right)
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    if min_spots < 3:
        raise ValueError("min_spots must be at least 3")
    common = left.index.intersection(right.index, sort=False)
    x, y = left.loc[common], right.loc[common]
    if groups is not None:
        if not groups.index.is_unique or groups.reindex(common).isna().any():
            raise ValueError("groups must supply exactly one nonmissing group for each matched spot")
        group = groups.reindex(common)
        if method == "spearman":
            x, y = (z.groupby(group, observed=True).rank() for z in (x, y))
        x, y = (z - z.groupby(group, observed=True).transform("mean") for z in (x, y))
    elif method == "spearman":
        x, y = x.rank(), y.rank()
    n = len(common)
    shape = (left.shape[1], right.shape[1])
    r = np.full(shape, np.nan)
    if n >= min_spots:
        a, b = (z.to_numpy() - z.to_numpy().mean(axis=0) for z in (x, y))
        denom = np.sqrt(np.sum(a * a, axis=0)[:, None] * np.sum(b * b, axis=0)[None, :])
        np.divide(a.T @ b, denom, out=r, where=denom > 0)
        r = np.clip(r, -1, 1)
    p = np.full(shape, np.nan)
    valid = np.isfinite(r)
    if groups is None and n >= min_spots:
        # Pearson exact null; conventional asymptotic Spearman test.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.abs(r[valid]) * np.sqrt((n - 2) / (1 - r[valid] ** 2))
        p[valid] = 2 * stats.t.sf(t, df=n - 2)
    q = np.full(shape, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        q[finite] = stats.false_discovery_control(p[finite], method="bh")
    return pd.DataFrame({
        "left": np.repeat(left.columns.to_numpy(), right.shape[1]),
        "right": np.tile(right.columns.to_numpy(), left.shape[1]),
        "r": r.ravel(), "p_nominal": p.ravel(), "q_nominal": q.ravel(),
        "n_spots": n, "method": method, "within_group": groups is not None,
    })


def compare_by_case(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cases: pd.Series,
    *,
    method: str = "pearson",
    leave_one_out: bool = False,
) -> pd.DataFrame:
    """Return pooled, within-case and individual-case feature associations.

    Optional leave-one-case-out coefficients are centred within retained cases.
    All scopes use the same identity-matched input cohort. This is descriptive
    sensitivity analysis, not a case-level hypothesis test or meta-analysis.
    """
    left, right = _frame(left), _frame(right)
    common = left.index.intersection(right.index, sort=False)
    if not cases.index.is_unique or cases.reindex(common).isna().any() or common.empty:
        raise ValueError("Nonempty matched spots with unique, complete case identities are required")
    left, right, cases = left.loc[common], right.loc[common], cases.loc[common]
    results = []
    scopes = [("pooled", common, None), ("within_case", common, cases)]
    for case in pd.unique(cases):
        scopes.append((f"case:{case}", common[cases.eq(case)], None))
        if leave_one_out and cases.nunique() > 1:
            scopes.append((f"without:{case}", common[cases.ne(case)], cases))
    for scope, ids, groups in scopes:
        result = correlate_spot_features(left.loc[ids], right.loc[ids], method=method, groups=groups)
        result.insert(0, "scope", scope)
        results.append(result)
    return pd.concat(results, ignore_index=True)


def gene_set_ora(
    expression: Any,
    genes: pd.Index,
    spots: pd.Index,
    network: pd.DataFrame,
    *,
    top_fraction: float = 0.05,
    background: int = 20000,
    min_targets: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Historical one-sided ORA: -log10 hypergeometric upper-tail probability.

    This deliberately implements the decoupler 1.x score used by the original
    notebook, not decoupler 2.x log odds. Rank all measured genes per spot, with
    seeded tie breaking; intersect network targets with measured genes. Inputs
    are log-normalised nonnegative expression and a source/target network.
    Returns scores, unadjusted enrichment p-values, and target coverage. No
    model is fitted and the expression object is never modified.
    """
    genes, spots = pd.Index(genes), pd.Index(spots)
    if not genes.is_unique or not spots.is_unique or expression.shape != (len(spots), len(genes)):
        raise ValueError("Expression dimensions and unique gene/spot identities must agree")
    if len(genes) < 2 or not 0 < top_fraction < 1 or min_targets < 1:
        raise ValueError("Invalid gene count, top_fraction or min_targets")
    if not isinstance(background, int) or background < len(genes):
        raise ValueError("background must be an integer at least as large as the measured gene universe")
    net = network[["source", "target"]].drop_duplicates()
    if net.isna().any().any():
        raise ValueError("Network source/target identities cannot be missing")
    measured = net[net.target.isin(genes)]
    coverage = pd.concat([net.groupby("source").size().rename("n_resource"),
                          measured.groupby("source").size().rename("n_measured")], axis=1).fillna(0)
    coverage["retained"] = coverage.n_measured.ge(min_targets)
    labels = coverage.index[coverage.retained]
    if labels.empty:
        raise ValueError("No gene sets meet min_targets")
    net = measured[measured.source.isin(labels)]
    membership = sparse.csr_matrix((np.ones(len(net)),
        (genes.get_indexer(net.target), labels.get_indexer(net.source))), shape=(len(genes), len(labels)))
    n_top = max(1, int(np.ceil(top_fraction * len(genes))))
    permutation = np.random.default_rng(seed).permutation(len(genes))
    sizes = coverage.loc[labels, "n_measured"].to_numpy()
    logp = np.empty((len(spots), len(labels)))
    for start in range(0, len(spots), 128):
        block = expression[start:start + 128]
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError("ORA expression must be finite and nonnegative")
        order = np.argsort(-block[:, permutation], axis=1, kind="stable")[:, :n_top]
        selected = permutation[order]
        indicator = sparse.csr_matrix((np.ones(selected.size),
            (np.repeat(np.arange(len(block)), n_top), selected.ravel())), shape=block.shape)
        overlap = (indicator @ membership).toarray()
        logp[start:start + len(block)] = stats.hypergeom.logsf(overlap - 1, background, sizes, n_top)
    scores = pd.DataFrame(-logp / np.log(10), index=spots, columns=labels)
    pvalues = pd.DataFrame(np.exp(logp), index=spots, columns=labels)
    return scores, pvalues, coverage


def gene_set_regression(
    expression: Any,
    genes: pd.Index,
    spots: pd.Index,
    network: pd.DataFrame,
    *,
    method: str = "mlm",
    min_targets: int = 5,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a weighted network with decoupler 2 MLM/ULM on explicit expression.

    An ephemeral AnnData avoids sparse-list conversion issues in decoupler 2.1
    and makes count-valued caller ``.raw`` inaccessible. Scientific imports are
    lazy. Returns scores and decoupler's adjusted enrichment p-values; neither
    input matrix nor network is modified. Expression must already be normalised.
    """
    if method not in {"mlm", "ulm"} or min_targets < 1 or batch_size < 1:
        raise ValueError("Use mlm/ulm, positive min_targets and positive batch_size")
    genes, spots = pd.Index(genes), pd.Index(spots)
    if not genes.is_unique or not spots.is_unique or expression.shape != (len(spots), len(genes)):
        raise ValueError("Expression dimensions and unique gene/spot identities must agree")
    net = network[["source", "target", "weight"]].copy()
    if net.isna().any().any() or net.duplicated(["source", "target"]).any():
        raise ValueError("Network edges must be unique and nonmissing")
    if not np.isfinite(net.weight.to_numpy(dtype=float)).all():
        raise ValueError("Network weights must be finite")
    import anndata as ad
    import decoupler as dc

    data = ad.AnnData(X=expression, obs=pd.DataFrame(index=spots), var=pd.DataFrame(index=genes))
    repaired = getattr(dc.mt, method)(data, net, tmin=min_targets, raw=False, bsize=batch_size)
    if repaired is not None:
        data = repaired
    score, adjusted_p = data.obsm[f"score_{method}"], data.obsm[f"padj_{method}"]
    if set(score.index) != set(spots.astype(str)):
        raise ValueError("Activity inference dropped observations; check empty expression rows")
    score, adjusted_p = (frame.reindex(spots.astype(str)).copy() for frame in (score, adjusted_p))
    score.index = adjusted_p.index = spots
    _frame(score)
    _frame(adjusted_p)
    return score, adjusted_p


def plot_correlation_heatmap(
    correlations: pd.DataFrame,
    *,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    row_colors: Mapping | pd.Series | pd.DataFrame | None = None,
    col_colors: Mapping | pd.Series | pd.DataFrame | None = None,
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = "RdBu_r",
    row_cluster: bool = True,
    col_cluster: bool = True,
    method: str = "average",
    metric: str = "euclidean",
    stars: bool = True,
    q_column: str = "q_nominal",
    alpha: float = 0.05,
    linewidths: float = 0.3,
    linecolor: str = "black",
    dendrogram_ratio: float | tuple[float, float] = 0.025,
    cell_size: float = 0.16,
    square_cells: bool = True,
    label_fontsize: float = 7,
    title_fontsize: float = 9,
    return_grid: bool = False,
):
    """Cluster signed correlations with label-aligned colours and adjusted-p stars.

    Supply one selected scope/variant (one record per left/right pair). Colour
    mappings/Series use feature labels as keys; DataFrames allow multiple colour
    bands. Missing palette entries and undefined correlations are grey. If any
    correlations are undefined, clustering is disabled with a warning rather
    than imputing coefficients for the dendrogram. Singleton axes are not clustered.

    A single star marks a finite supplied adjusted p-value strictly below
    ``alpha``. Values are neither rounded nor adjusted again after subsetting.
    The default ``q_nominal`` contains BH-adjusted nominal spot-level tests;
    spatial dependence is not corrected. Group-centred correlations have no
    such tests, so their plots explicitly indicate that stars are unavailable.
    Supply a different adjusted-p column via ``q_column`` if appropriate.

    Cells have thin black outlines by default. ``dendrogram_ratio`` is a fraction
    of the available panel width/height (scalar or row/column pair). Automatic
    sizing targets ``cell_size`` inches per cell, allowing space for labels.
    Explicit ``figsize`` is in inches. ``square_cells=True`` fits square cells
    inside that canvas, keeping dendrograms and colour strips aligned; disable
    it to fill the available rectangle. Fonts and borders are in points.

    Returns a matplotlib Figure by default; ``return_grid=True`` returns the
    seaborn ClusterGrid, exposing dendrogram order and its savefig method.
    """
    import textwrap
    import warnings

    import seaborn as sns
    from matplotlib.font_manager import FontProperties

    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError("vmin and vmax must be finite, with vmin < vmax")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")
    ratios = np.broadcast_to(np.asarray(dendrogram_ratio, dtype=float), (2,))
    if not np.isfinite(ratios).all() or ((ratios < 0) | (ratios >= 0.5)).any():
        raise ValueError("dendrogram_ratio must be a scalar or pair in [0, 0.5)")
    if not np.isfinite([cell_size, label_fontsize, title_fontsize, linewidths]).all() or (
        min(cell_size, label_fontsize, title_fontsize) <= 0 or linewidths < 0
    ):
        raise ValueError("Cell/font sizes must be positive and linewidths nonnegative")
    if figsize is not None and (len(figsize) != 2 or not np.isfinite(figsize).all() or min(figsize) <= 0):
        raise ValueError("figsize must contain two positive finite dimensions")
    if correlations.empty or correlations[["left", "right"]].isna().any().any():
        raise ValueError("Nonempty correlations with nonmissing feature labels are required")
    values = correlations.pivot(index="left", columns="right", values="r")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Correlations must be finite or NaN")

    def align_colors(colors, labels):
        if colors is None:
            return None
        colors = colors.copy() if isinstance(colors, (pd.Series, pd.DataFrame)) else pd.Series(colors)
        if not colors.index.is_unique:
            raise ValueError("Colour palettes must have unique feature labels")
        return colors.reindex(labels).fillna("#dddddd")

    row_cluster = row_cluster and len(values.index) > 1
    col_cluster = col_cluster and len(values.columns) > 1
    if values.isna().any().any() and (row_cluster or col_cluster):
        warnings.warn("Undefined correlations: clustering disabled; grey cells remain undefined.",
                      UserWarning, stacklevel=2)
        row_cluster = col_cluster = False

    annotations = None
    note = ""
    if stars:
        q_values = (correlations.pivot(index="left", columns="right", values=q_column)
                    .reindex(index=values.index, columns=values.columns)
                    if q_column in correlations else pd.DataFrame(np.nan, index=values.index, columns=values.columns))
        q_array = q_values.to_numpy(dtype=float)
        if np.isinf(q_array).any() or ((q_array < 0) | (q_array > 1)).any():
            raise ValueError("Adjusted p-values must lie in [0, 1] or be NaN")
        available = q_values.notna() & values.notna()
        annotations = np.where(available & (q_values < alpha), "*", "")
        if available.any().any():
            note = f"* adjusted p < {alpha:g}"
            if q_column == "q_nominal":
                note = f"* BH q < {alpha:g} (nominal spot-level tests)"
            if (values.notna() & ~available).any().any():
                note += "; some adjusted p-values unavailable"
        else:
            note = "Adjusted p-values unavailable; no significance stars"
    if figsize is None:
        figsize = (max(3.2, cell_size * len(values.columns) + 2),
                   max(3.2, cell_size * len(values) + 2.2))
        auto_size = True
    else:
        auto_size = False
    grid = sns.clustermap(
        values, vmin=vmin, vmax=vmax, center=0, cmap=cmap,
        figsize=(max(6, figsize[0]), max(6, figsize[1])),
        row_cluster=row_cluster, col_cluster=col_cluster, method=method, metric=metric,
        row_colors=align_colors(row_colors, values.index), col_colors=align_colors(col_colors, values.columns),
        annot=annotations if stars else False, fmt="", annot_kws={"fontsize": label_fontsize},
        linewidths=linewidths, linecolor=linecolor, antialiased=True,
        xticklabels=True, yticklabels=True, dendrogram_ratio=tuple(ratios), colors_ratio=0.015,
        cbar_kws={"label": "Correlation", "orientation": "horizontal"},
    )
    grid.ax_heatmap.set_facecolor("#dddddd")
    grid.ax_heatmap.set(xlabel="IMC population", ylabel="Visium feature")
    grid.ax_heatmap.tick_params(axis="both", labelsize=label_fontsize, length=2, pad=2)
    grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), rotation=90)
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=0)
    grid.ax_heatmap.xaxis.label.set_size(label_fontsize)
    grid.ax_heatmap.yaxis.label.set_size(label_fontsize)
    for ax in (grid.ax_row_colors, grid.ax_col_colors):
        if ax is not None:
            ax.tick_params(labelsize=label_fontsize, length=0, pad=2)

    # Measure labels in inches, then position the heatmap and its aligned axes.
    # seaborn ignores square=True in clustermap; a shared explicit layout also
    # avoids leaving a tall dendrogram or colourbar over the first data rows.
    renderer = grid.fig.canvas.get_renderer()
    font = FontProperties(size=label_fontsize)
    def text_width(labels):
        return max(renderer.get_text_width_height_descent(str(label), font, False)[0]
                   for label in labels) / grid.fig.dpi

    left, right = 0.12, text_width(values.index) + 0.32
    bottom = text_width(values.columns) + 0.32
    row_bands = 0 if grid.ax_row_colors is None else (row_colors.shape[1] if isinstance(row_colors, pd.DataFrame) else 1)
    col_bands = 0 if grid.ax_col_colors is None else (col_colors.shape[1] if isinstance(col_colors, pd.DataFrame) else 1)
    row_strip, col_strip = 0.08 * row_bands, 0.08 * col_bands
    row_ratio = ratios[0] if row_cluster else 0
    col_ratio = ratios[1] if col_cluster else 0
    width, height = figsize
    if auto_size:
        width = max(3.2, left + right + (len(values.columns) * cell_size + row_strip) / (1 - row_ratio))
    wrap_width = max(24, int((width - 0.24) * 72 / (title_fontsize * 0.52)))
    heading = "\n".join(textwrap.fill(text, width=wrap_width) for text in (title, note) if text)
    heading_height = max(1, len(heading.splitlines())) * title_fontsize * 1.25 / 72
    top = heading_height + 0.62  # Dedicated title and compact horizontal colourbar.
    if auto_size:
        height = bottom + top + (len(values) * cell_size + col_strip) / (1 - col_ratio)
    grid.fig.set_size_inches(width, height)
    panel_width, panel_height = width - left - right, height - bottom - top
    row_tree, col_tree = panel_width * row_ratio, panel_height * col_ratio
    heat_width, heat_height = panel_width - row_tree - row_strip, panel_height - col_tree - col_strip
    if min(heat_width, heat_height) <= 0:
        import matplotlib.pyplot as plt
        plt.close(grid.fig)
        raise ValueError("figsize is too small for labels/title; increase it or reduce font sizes")
    if square_cells:
        side = min(heat_width / len(values.columns), heat_height / len(values))
        heat_width, heat_height = side * len(values.columns), side * len(values)
    x = left + row_tree + row_strip + (panel_width - row_tree - row_strip - heat_width) / 2
    y = bottom + (panel_height - col_tree - col_strip - heat_height) / 2

    def position(ax, x0, y0, w, h):
        if ax is not None:
            ax.set_position([x0 / width, y0 / height, w / width, h / height])

    position(grid.ax_heatmap, x, y, heat_width, heat_height)
    position(grid.ax_row_dendrogram, x - row_strip - row_tree, y, row_tree, heat_height)
    position(grid.ax_col_dendrogram, x, y + heat_height + col_strip, heat_width, col_tree)
    position(grid.ax_row_colors, x - row_strip, y, row_strip, heat_height)
    position(grid.ax_col_colors, x, y + heat_height, heat_width, col_strip)
    position(grid.cax, left, height - heading_height - 0.28, min(1.0, width * 0.3), 0.06)
    grid.cax.tick_params(labelsize=label_fontsize, length=2, pad=1)
    grid.cax.xaxis.label.set_size(label_fontsize)
    grid.cax.xaxis.labelpad = 1
    grid.fig.suptitle(heading, y=1 - 0.06 / height, fontsize=title_fontsize)
    return grid if return_grid else grid.fig


def plot_paired_population_maps(
    imc: pd.DataFrame,
    rna: pd.DataFrame,
    coordinates: pd.DataFrame,
    pairs: Mapping[str, list[str]],
    *,
    spot_diameter: float,
    title: str = "",
    image: np.ndarray | None = None,
    image_scale: float = 1.0,
    spot_diameter_scale: float = 1.0,
    imc_colors: Mapping | pd.Series | None = None,
    rna_colors: Mapping | pd.Series | None = None,
    quantile: float = 0.99,
    scale_scope: str = "cohort",
    matched_footprint: bool = True,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    image_alpha: float = 0.45,
    spot_alpha: float = 0.9,
    label_fontsize: float = 9,
    label_wrap: int | None = None,
    swatch_size: float | None = None,
    value_label: str = "Spot fraction",
    annotate_correlations: bool = True,
):
    """Show corresponding IMC populations and RNA groups over a shared tissue image.

    ``pairs`` maps each RNA label to one or more IMC labels, preserving row order.
    Coordinates and spot diameter are in full-resolution pixels; ``image_scale``
    converts both to the supplied image's pixels. All panels share the same crop,
    y orientation and spot diameter. ``spot_diameter_scale`` enlarges the displayed
    circles only (1.0 preserves physical size; choose a larger factor relative to
    the observed grid spacing to make spots nearly touch). Coordinates, coverage,
    values and correlations are unchanged.
    Missing coverage is grey, never zero.

    Pass cohort-wide value tables, with IMC already filtered for coverage/quality,
    and one case's coordinates. Each population's scale is zero to its ``quantile``
    over the cohort-wide intersection of IMC/RNA spots, so the same population has
    the same scale across cases. Scales are independent across populations and
    modalities: colour similarity describes location, not equal abundance.
    With ``scale_scope="case"``, estimate these limits using the displayed case's
    shared spots instead; intensities can then no longer be compared across cases.
    Values above the limit saturate; neither source values nor correlations are clipped.
    By default both modalities display only their shared spot footprint; setting
    ``matched_footprint=False`` additionally shows RNA outside measured IMC coverage.

    Returns ``(figure, diagnostics)`` with unclipped Pearson/Spearman coefficients,
    paired spot counts and colour limits for each displayed pair. Correlations use
    only shared spots in the supplied coordinates, require four spots, and omit
    significance tests. No smoothing, registration or pair selection is performed.
    ``annotate_correlations=False`` hides coefficients on the maps while retaining
    the complete diagnostics table. Subplots are packed with small physical gaps.
    ``label_wrap`` wraps long population labels at this character width, useful
    for compact panels without reducing their label font size.
    ``swatch_size`` sets the palette square's side in points (default: label font
    size plus four points). Label and swatch share a vertically centred legend
    box, so alignment and square size are independent of the tissue aspect ratio.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    from textwrap import fill

    imc, rna = _frame(imc, nonnegative=True), _frame(rna, nonnegative=True)
    coordinates = _frame(coordinates)
    if coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have two columns (x, y)")
    if not pairs or any(not labels or isinstance(labels, str) for labels in pairs.values()):
        raise ValueError("pairs must map RNA labels to nonempty lists of IMC labels")
    if scale_scope not in {"cohort", "case"}:
        raise ValueError("scale_scope must be 'cohort' or 'case'")
    if label_wrap is not None and label_wrap < 1:
        raise ValueError("label_wrap must be positive or None")
    if swatch_size is None:
        swatch_size = label_fontsize + 4
    if not np.isfinite(swatch_size) or swatch_size <= 0:
        raise ValueError("swatch_size must be positive and finite")
    if any(len(set(labels)) != len(labels) for labels in pairs.values()):
        raise ValueError("Each row must contain unique IMC labels")
    if set(pairs) - set(rna.columns) or {x for labels in pairs.values() for x in labels} - set(imc.columns):
        raise ValueError("Every requested population must be present in its modality table")
    if not np.isfinite([spot_diameter, image_scale, spot_diameter_scale, quantile, image_alpha, spot_alpha]).all() or (
        min(spot_diameter, image_scale, spot_diameter_scale) <= 0 or not 0 < quantile <= 1
        or not 0 <= image_alpha <= 1 or not 0 <= spot_alpha <= 1
    ):
        raise ValueError("Positive spot diameter/scale, quantile in (0, 1], and alpha in [0, 1] required")
    if image is not None and (np.asarray(image).ndim not in (2, 3) or not np.isfinite(image).all()):
        raise ValueError("image must be a finite greyscale or RGB(A) array")
    reference = imc.index.intersection(rna.index)
    if reference.empty:
        raise ValueError("IMC and RNA tables have no shared spots")
    paired = coordinates.index.intersection(reference)
    if not coordinates.index.isin(rna.index).all():
        raise ValueError("Every plotted coordinate must have an RNA observation")
    scale_reference = paired if scale_scope == "case" else reference
    if scale_reference.empty:
        raise ValueError("No paired spots available to estimate the requested colour scale")
    limits = {
        "IMC": imc.loc[scale_reference].quantile(quantile).clip(lower=1e-12),
        "RNA": rna.loc[scale_reference].quantile(quantile).clip(lower=1e-12),
    }
    xy = coordinates * image_scale
    diameter = spot_diameter * image_scale * spot_diameter_scale
    lower, upper = xy.min().to_numpy() - diameter, xy.max().to_numpy() + diameter
    nrows, n_imc = len(pairs), max(map(len, pairs.values()))
    if figsize is None:
        figsize = (2.7 * (n_imc + 1), 2.5 * nrows + 1.1)
    fig, axes = plt.subplots(nrows, n_imc + 1, figsize=figsize, squeeze=False)
    # Pack the shared-aspect maps together rather than leaving the unused width
    # of each GridSpec cell as a wide gap when the tissue is nearly square.
    width, height = figsize
    gap, margin, top_margin, bottom_margin = 0.035, 0.07, 0.75, 0.82
    aspect = (upper[1] - lower[1]) / (upper[0] - lower[0])
    panel_width = min((width - 2 * margin - n_imc * gap) / (n_imc + 1),
                      (height - top_margin - bottom_margin - (nrows - 1) * gap) / (nrows * aspect))
    panel_height = panel_width * aspect
    x0 = (width - (n_imc + 1) * panel_width - n_imc * gap) / 2
    y_top = height - top_margin
    for row in range(nrows):
        for col in range(n_imc + 1):
            axes[row, col].set_position([(x0 + col * (panel_width + gap)) / width,
                (y_top - (row + 1) * panel_height - row * gap) / height,
                panel_width / width, panel_height / height])
    diagnostics = []

    def panel(ax, frame, label, modality, palette, annotation=""):
        if image is not None:
            ax.imshow(image, origin="upper", alpha=image_alpha, interpolation="nearest")
        background = EllipseCollection(
            [diameter], [diameter], [0], units="xy", offsets=xy.to_numpy(),
            offset_transform=ax.transData, facecolors="#cccccc", edgecolors="none",
        )
        ax.add_collection(background)
        ids = paired if modality == "IMC" or matched_footprint else coordinates.index
        spots = EllipseCollection(
            [diameter], [diameter], [0], units="xy", offsets=xy.loc[ids].to_numpy(),
            offset_transform=ax.transData, cmap=cmap, norm=Normalize(0, limits[modality][label]),
            edgecolors="none", alpha=spot_alpha,
        )
        spots.set_array(frame.loc[ids, label].to_numpy())
        ax.add_collection(spots)
        ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]), aspect="equal", xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set(linewidth=0.4, color="#666666")
        color = palette.get(label) if palette is not None else None
        display_label = fill(label, width=label_wrap) if label_wrap else label
        label_box = TextArea(display_label, textprops={"color": "white", "fontsize": label_fontsize,
                                                     "multialignment": "right"})
        legend_items = [label_box]
        if color:
            swatch = DrawingArea(swatch_size, swatch_size, 0, 0)
            swatch.add_artist(Rectangle((0, 0), swatch_size, swatch_size,
                                       facecolor=color, edgecolor="black", linewidth=0.4))
            legend_items.append(swatch)
        legend = AnchoredOffsetbox(
            loc="upper right", child=HPacker(children=legend_items, align="center", pad=0, sep=3),
            pad=0.15, borderpad=0, prop={"size": label_fontsize},
            bbox_to_anchor=(0.99, 0.99), bbox_transform=ax.transAxes, frameon=True,
        )
        legend.patch.set(facecolor=(0, 0, 0, 0.8), edgecolor="none")
        ax.add_artist(legend)
        if annotation:
            ax.text(0.015, 0.015, annotation, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7, color="white",
                    bbox={"facecolor": "black", "edgecolor": "none", "pad": 2, "alpha": 0.7})

    for row, (rna_label, imc_labels) in enumerate(pairs.items()):
        for col in range(n_imc):
            if col >= len(imc_labels):
                axes[row, col].set_axis_off()
                continue
            imc_label = imc_labels[col]
            a, b = imc.loc[paired, imc_label], rna.loc[paired, rna_label]
            if len(paired) >= 4 and a.nunique() > 1 and b.nunique() > 1:
                pearson = float(stats.pearsonr(a, b).statistic)
                spearman = float(stats.spearmanr(a, b).statistic)
            else:
                pearson = spearman = float("nan")
            diagnostics.append({"rna_population": rna_label, "imc_population": imc_label,
                                "n_spots": len(paired), "pearson_r": pearson, "spearman_rho": spearman,
                                "imc_vmax": float(limits["IMC"][imc_label]),
                                "rna_vmax": float(limits["RNA"][rna_label]), "scale_scope": scale_scope})
            label = f"r = {pearson:.2f}; ρ = {spearman:.2f}" if np.isfinite(pearson) else "Correlation undefined"
            panel(axes[row, col], imc, imc_label, "IMC", imc_colors,
                  label if annotate_correlations else "")
        panel(axes[row, -1], rna, rna_label, "RNA", rna_colors)

    fig.suptitle(f"{title}\n{len(paired):,} paired / {len(coordinates):,} QC spots", fontsize=9,
                 y=1 - 0.06 / height)
    # Headers span each modality's columns, independently of the population labels.
    left_box, last_imc = axes[0, 0].get_position(), axes[0, n_imc - 1].get_position()
    rna_box = axes[0, -1].get_position()
    fig.text((left_box.x0 + last_imc.x1) / 2, 1 - 0.56 / height, "IMC · Population", ha="center", fontsize=9)
    fig.text((rna_box.x0 + rna_box.x1) / 2, 1 - 0.56 / height, "RNA · coarse\n(cell2location)",
             ha="center", va="center", fontsize=8)
    colorbar_ax = fig.add_axes([0.18, 0.61 / height, 0.64, 0.055 / height])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=colorbar_ax,
                            orientation="horizontal", ticks=[0, 1])
    scale_label = "case" if scale_scope == "case" else "pooled"
    colorbar.ax.set_xticklabels(["Low (0)", f"High ({scale_label} {quantile * 100:g}th percentile)"])
    colorbar.ax.get_xticklabels()[0].set_horizontalalignment("left")
    colorbar.ax.get_xticklabels()[1].set_horizontalalignment("right")
    colorbar.ax.tick_params(labelsize=6.5, length=0)
    fig.legend(handles=[Patch(facecolor="#cccccc", label="Outside paired coverage")],
               loc="lower center", bbox_to_anchor=(0.5, 0.22 / height), frameon=False, fontsize=6.5)
    comparison_label = "within-case scales" if scale_scope == "case" else "scales shared across cases"
    footer = f"{value_label} · {comparison_label}\nPopulations/modalities scaled separately"
    if annotate_correlations:
        footer += "\nMatched spots only for correlations; no smoothing or significance tests"
    fig.text(0.5, 0.035 / height, footer, ha="center", fontsize=6.5)
    return fig, pd.DataFrame(diagnostics)


def plot_spatial_features(
    features: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    title: str = "",
    ncols: int = 4,
    point_size: float = 4,
    limits: Mapping[str, tuple[float, float]] | None = None,
    cmap: str = "viridis",
):
    """Small multiples in source pixel coordinates; NaN/absent coverage is grey.

    Coordinates have two columns (x,y) indexed by source spot identity. Provide
    limits computed across cases for comparable scales. Images are optional
    context in caller-owned plots, not inferred from coordinate magnitudes.
    """
    import matplotlib.pyplot as plt

    if not features.index.is_unique or not coordinates.index.is_unique or coordinates.shape[1] != 2:
        raise ValueError("Unique spot identities and two coordinate columns are required")
    if features.shape[1] == 0 or ncols < 1 or not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("Nonempty features, finite coordinates and positive ncols are required")
    values = features.reindex(coordinates.index)
    fig, axes = plt.subplots(int(np.ceil(len(values.columns) / ncols)), ncols,
                             figsize=(3.6 * ncols, 3.5 * np.ceil(len(values.columns) / ncols)), squeeze=False)
    for ax, label in zip(axes.flat, values.columns):
        ax.scatter(*coordinates.to_numpy().T, s=point_size, c="#dddddd", rasterized=True)
        valid = values[label].notna()
        vmin, vmax = limits[label] if limits else (None, None)
        im = ax.scatter(*coordinates.loc[valid].to_numpy().T, c=values.loc[valid, label],
                        s=point_size, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(str(label), fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.65)
    for ax in list(axes.flat)[len(values.columns):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
