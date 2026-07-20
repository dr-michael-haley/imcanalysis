"""Standalone command-line interface for generic AnnData population QC."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig

from .api import DEFAULT_SWEEP_REGEX, run_population_embedding_qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assess population support from existing UMAP, PCA, graph, and clustering state. "
            "This command never recalculates Leiden, PCA, UMAP, or neighbours."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .h5ad file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Standalone output directory")
    parser.add_argument("--population-obs")
    parser.add_argument("--mode", choices=("auto", "single", "sweep"), default="auto")
    parser.add_argument("--sweep-regex", default=DEFAULT_SWEEP_REGEX)
    parser.add_argument("--sweep-columns", nargs="+", help="Explicit sweep columns (space or comma separated)")
    parser.add_argument("--reference-resolution", type=float)
    parser.add_argument("--umap-key", default="X_umap")
    parser.add_argument("--pca-key", default="X_pca")
    parser.add_argument("--connectivities-key")
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--umap-k", type=int, default=15)
    parser.add_argument("--pca-dimensions", type=int, default=30)
    parser.add_argument("--silhouette-max-cells", type=int, default=10000)
    parser.add_argument("--density-max-cells-per-cluster", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metric-config", type=Path)
    parser.add_argument("--write-per-cell-metrics", action="store_true")
    parser.add_argument("--write-annotated-h5ad", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of files in the exact output directory")
    return parser


def _sweep_columns(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    columns = [column.strip() for value in values for column in value.split(",") if column.strip()]
    return columns or None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    input_path = args.input.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input AnnData file not found: {input_path}")
    if input_path.suffix.lower() != ".h5ad":
        raise ValueError(f"Input must be an .h5ad file: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to replace same-named outputs."
        )
    settings = PopulationEmbeddingQCConfig(
        mode=args.mode,
        population_obs=args.population_obs,
        sweep_regex=args.sweep_regex,
        sweep_columns=_sweep_columns(args.sweep_columns),
        reference_resolution=args.reference_resolution,
        umap_key=args.umap_key,
        pca_key=args.pca_key,
        connectivities_key=args.connectivities_key,
        min_cluster_size=args.min_cluster_size,
        umap_k=args.umap_k,
        pca_dimensions=args.pca_dimensions,
        silhouette_max_cells=args.silhouette_max_cells,
        density_max_cells_per_cluster=args.density_max_cells_per_cluster,
        random_seed=args.random_seed,
        metric_config_path=str(args.metric_config) if args.metric_config else None,
        write_per_cell_metrics=args.write_per_cell_metrics,
        write_annotated_h5ad=args.write_annotated_h5ad,
    )
    import anndata

    logging.info("Reading AnnData from %s", input_path)
    adata = anndata.read_h5ad(input_path)
    result = run_population_embedding_qc(
        adata,
        population_obs=args.population_obs,
        mode=args.mode,
        sweep_columns=_sweep_columns(args.sweep_columns),
        sweep_regex=args.sweep_regex,
        reference_resolution=args.reference_resolution,
        umap_key=args.umap_key,
        pca_key=args.pca_key,
        connectivities_key=args.connectivities_key,
        output_dir=output_dir,
        config=settings,
        overwrite=args.overwrite,
    )
    logging.info("Wrote %d population QC files to %s", len(result.output_files), output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"population embedding QC failed: {exc}", file=sys.stderr)
        raise


__all__ = ["build_parser", "main"]
