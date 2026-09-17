"""Shared-reporter QC for reference regression and Visium mapping."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.cell2location_analysis import (
    BARCODE,
    PRIOR_MEAN,
    PRIOR_RAW,
    Cell2locationResult,
)
from SpatialBiologyToolkit.config.models import Cell2locationConfig
from SpatialBiologyToolkit.reporting import StageReporter


def abundance_frame(adata, summary: str = "means") -> pd.DataFrame:
    """Return cell abundances with explicit upstream factor names and spot alignment."""
    factors = list(adata.uns["mod"]["factor_names"])
    key = f"{summary}_cell_abundance_w_sf"
    columns = [f"{summary}cell_abundance_w_sf_{f}" for f in factors]
    if key in adata.obsm:
        matrix = adata.obsm[key]
        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.loc[
                adata.obs_names, columns
            ].to_numpy()
    else:
        matrix = adata.obs[columns].to_numpy()
    return pd.DataFrame(matrix, index=adata.obs_names, columns=factors)


def report_result(
    result: Cell2locationResult,
    settings: Cell2locationConfig,
    reporter: StageReporter,
    *,
    operation: str,
) -> None:
    import matplotlib.pyplot as plt

    adata = result.adata
    figures, tables = reporter.context.figures_dir, reporter.context.tables_dir
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    reporter.add_metric(f"{operation}_observations", adata.n_obs)
    reporter.add_metric(f"{operation}_genes", adata.n_vars)
    reporter.add_metric(
        f"{operation}_cell_types", len(adata.uns["mod"]["factor_names"])
    )
    for name, history in (result.model.history or {}).items():
        # History keys originate upstream, not from user labels.
        if not str(name).replace("_", "").isalnum():
            continue
        table = tables / f"{operation}_{name}.csv"
        history.to_csv(table)
        reporter.add_file(
            "table", table, "Training loss history; assess convergence manually."
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.asarray(history).reshape(-1))
        ax.set(xlabel="Epoch", ylabel=str(name), title=f"{operation.title()} training")
        path = figures / f"{operation}_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        reporter.add_file("figure", path)
    if operation == "reference":
        table = tables / "reference_cell_type_counts.csv"
        adata.obs[settings.reference_labels_key].value_counts().rename(
            "n_cells"
        ).to_csv(table)
        reporter.add_file("table", table)
        table = tables / "reference_retained_gene_qc.csv"
        adata.var[["n_cells", "nonz_mean"]].to_csv(table)
        reporter.add_file("table", table)
        fig, ax = plt.subplots(figsize=(8, 4))
        signatures = result.signatures
        if signatures is None:
            raise ValueError("Reference reporting requires fitted signatures")
        image = ax.imshow(
            np.log1p(signatures.to_numpy().T), aspect="auto", interpolation="none"
        )
        ax.set(
            xlabel="Retained genes",
            ylabel="Reference cell types",
            title="log1p posterior mean signatures (display only)",
        )
        ax.set_yticks(range(len(signatures.columns)), signatures.columns, fontsize=7)
        fig.colorbar(image, ax=ax)
        path = figures / "reference_signatures.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        reporter.add_file("figure", path)
        return

    means, q05 = abundance_frame(adata), abundance_frame(adata, "q05")
    libraries = adata.obs[settings.library_key].astype(str)
    totals = means.sum(axis=1)
    qc = pd.DataFrame(
        {
            "library_id": libraries,
            "barcode": adata.obs[BARCODE],
            "total_counts": np.asarray(adata.X.sum(axis=1)).ravel(),
            "posterior_mean_total_cells": totals,
        }
    )
    if PRIOR_MEAN in adata.obs:
        qc["prior_mean"] = adata.obs[PRIOR_MEAN]
        qc["registered_count"] = adata.obs[PRIOR_RAW]
        qc["posterior_to_prior_ratio"] = totals / qc["prior_mean"]
        reporter.add_warning(
            "IMC counts are from a registered serial section. Prior agreement is a diagnostic, not independent validation."
        )
        for key in ["prior_floored_spots", "prior_extra_table_rows"]:
            reporter.add_metric(key, adata.uns["sbt_cell2location"][key])
    path = tables / "mapping_spot_qc.csv"
    qc.to_csv(path, index_label="spot_id")
    reporter.add_file("table", path)
    for name, frame in [("mean_abundance", means), ("q05_abundance", q05)]:
        path = tables / f"mapping_{name}.csv.gz"
        frame.to_csv(path, index_label="spot_id")
        reporter.add_file("table", path)
    summary = means.groupby(libraries, observed=True).mean()
    path = tables / "mapping_library_mean_abundance.csv"
    summary.to_csv(path)
    reporter.add_file("table", path)
    reporter.add_metric("mapping_libraries", libraries.nunique())
    for number, library in enumerate(pd.unique(libraries)):
        selected = libraries.eq(library).to_numpy()
        coords = np.asarray(adata.obsm["spatial"])[selected]
        top = means.loc[selected].mean().nlargest(settings.plot_top_cell_types).index
        ncols = min(4, len(top))
        fig, axes = plt.subplots(
            math.ceil(len(top) / ncols),
            ncols,
            figsize=(4 * ncols, 3.5 * math.ceil(len(top) / ncols)),
            squeeze=False,
        )
        for ax, label in zip(axes.flat, top):
            im = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=q05.loc[selected, label],
                s=12,
                cmap="magma",
            )
            ax.set_title(label, fontsize=9)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, shrink=0.7)
        for ax in list(axes.flat)[len(top) :]:
            ax.set_visible(False)
        fig.suptitle(f"{library}: 5% posterior quantile of cell abundance")
        path = figures / f"mapping_library_{number:03d}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        reporter.add_file(
            "figure",
            path,
            f"Library {library}; native Visium coordinates without a histology overlay.",
        )
    fig, axes = plt.subplots(
        1, 2 if PRIOR_MEAN in adata.obs else 1, figsize=(10, 4), squeeze=False
    )
    axes[0, 0].scatter(qc.total_counts, totals, s=5, alpha=0.5)
    axes[0, 0].set(xlabel="Spot RNA counts", ylabel="Posterior mean total cells")
    if PRIOR_MEAN in adata.obs:
        axes[0, 1].scatter(qc.prior_mean, totals, s=5, alpha=0.5)
        axes[0, 1].set(
            xlabel="IMC-derived prior mean", ylabel="Posterior mean total cells"
        )
    path = figures / "mapping_total_cell_qc.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    reporter.add_file("figure", path)
