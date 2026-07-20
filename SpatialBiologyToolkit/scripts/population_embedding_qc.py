"""Managed pipeline entry point for population embedding and clustering QC."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Literal
import uuid


def _atomic_annotated_write(adata, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"Annotated AnnData already exists and will not be overwritten: {target}"
        )
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.h5ad")
    try:
        adata.write_h5ad(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_pipeline() -> int:
    """Run from typed project configuration and shared report context."""
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.population_embedding_qc import run_population_embedding_qc
    from SpatialBiologyToolkit.population_embedding_qc.outputs import (
        OutputLayout,
        annotated_copy,
        write_result_outputs,
    )
    from SpatialBiologyToolkit.reporting import (
        bootstrap_stage_reporting,
        category_output_path,
        get_active_reporter,
        project_asset_path,
    )

    reporter = bootstrap_stage_reporting("popqc")
    config_path = Path(os.environ.get("SBT_CONFIG", "config.yaml")).expanduser().resolve(strict=False)
    config = load_config(config_path)
    settings = config.population_embedding_qc
    if not settings.enabled:
        logging.info("Population embedding QC is disabled in configuration; nothing to do")
        if reporter:
            reporter.add_note("Stage skipped because population_embedding_qc.enabled is false.")
        return 0
    settings = settings.model_copy(
        update={
            "population_obs": (
                settings.population_obs or config.general.population_obs_primary
            ),
            "metric_config_path": (
                str(project_asset_path(settings.metric_config_path))
                if settings.metric_config_path
                else None
            ),
            "sample_obs": settings.sample_obs or config.general.case_obs,
            "roi_obs": settings.roi_obs or config.general.roi_obs,
        }
    )
    input_path = project_asset_path(settings.input_adata_path or config.general.anndata_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Population QC input AnnData not found: {input_path}")
    import anndata
    import numpy as np

    logging.info("Reading population QC input: %s", input_path)
    adata = anndata.read_h5ad(input_path)
    result = run_population_embedding_qc(adata, config=settings)
    layout = OutputLayout(
        figures=category_output_path("figures", stage="popqc"),
        tables=category_output_path("tables", stage="popqc"),
        summaries=category_output_path("summaries", stage="popqc"),
        files=category_output_path("files", stage="popqc"),
    )
    paths = write_result_outputs(
        result,
        umap=np.asarray(adata.obsm[settings.umap_key]),
        config=settings,
        layout=layout,
    )
    active = get_active_reporter() or reporter
    if active:
        active.add_input("anndata", input_path, "AnnData inspected without in-place modification")
        for warning in result.warnings:
            active.add_warning(warning)
        active.add_metric("cells_analysed", result.run_summary["n_cells_analysed"])
        active.add_metric("reference_clusters", result.run_summary["n_reference_clusters"])
        active.add_metric("clusters_with_threshold_failures", int((result.cluster_summary["failed_thresholds"] > 0).sum()))
        for path in paths:
            category: Literal["figure", "table", "summary", "file"] = (
                "figure" if layout.figures in path.parents
                else "table" if layout.tables in path.parents
                else "summary" if layout.summaries in path.parents
                else "file"
            )
            active.add_file(category, path)
    if settings.write_annotated_h5ad:
        target = project_asset_path(settings.annotated_adata_path)
        _atomic_annotated_write(annotated_copy(adata, result), target)
        if active:
            active.add_asset("population_qc_anndata", target, "Separate AnnData copy with namespaced population embedding QC annotations")
    logging.info("Population embedding QC completed with %d files", len(paths))
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        from SpatialBiologyToolkit.population_embedding_qc.cli import main as standalone_main

        return standalone_main(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [PopulationEmbeddingQC] %(message)s")
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
