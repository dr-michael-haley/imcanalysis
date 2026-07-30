"""Managed ``sbt run maxfuse`` entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import sys
import uuid
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _atomic_h5ad(adata, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.h5ad")
    try:
        adata.write_h5ad(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame, target: Path, *, index: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.csv.gz" if target.name.endswith(".csv.gz") else ".tmp.csv"
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}{suffix}")
    try:
        frame.to_csv(temporary, index=index)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime(argv: list[str] | None):
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.reporting import bootstrap_stage_reporting
    from SpatialBiologyToolkit.scripts.config_and_utils import setup_logging

    parser = argparse.ArgumentParser(
        description="Match one scRNA-seq reference to an IMC AnnData with MaxFuse."
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("SBT_CONFIG", "config.yaml"),
        help="Pipeline configuration path (default: SBT_CONFIG or ./config.yaml).",
    )
    arguments = parser.parse_args(argv)
    config_path = Path(arguments.config).expanduser().resolve(strict=False)
    os.environ["SBT_CONFIG"] = str(config_path)
    os.environ.setdefault("SBT_PROJECT_ROOT", str(config_path.parent))
    reporter = bootstrap_stage_reporting(os.environ.get("SBT_STAGE") or "maxfuse")
    config = load_config(config_path)
    setup_logging(config.logging.model_dump(mode="python"), "MaxFuse")
    settings = config.maxfuse.model_copy(
        update={
            "target_population_obs": (
                config.maxfuse.target_population_obs
                or config.general.population_obs_primary
            ),
            "sample_obs": config.maxfuse.sample_obs or config.general.case_obs,
            "roi_obs": config.maxfuse.roi_obs or config.general.roi_obs,
        }
    )
    return config, settings, reporter


def run_pipeline(argv: list[str] | None = None) -> int:
    """Run MaxFuse and create reusable assets plus an execution report."""

    import anndata as ad

    from SpatialBiologyToolkit.maxfuse_matching import (
        build_matched_transcriptomes,
        build_transfer_anndata,
        dense_size_gib,
        prepare_maxfuse_inputs,
        read_feature_mapping,
        run_maxfuse_matching,
    )
    from SpatialBiologyToolkit.maxfuse_reports import generate_maxfuse_report
    from SpatialBiologyToolkit.reporting import (
        get_active_reporter,
        optional_category_output_path,
        project_asset_path,
    )

    config, settings, reporter = _runtime(argv)
    active_reporter = get_active_reporter() or reporter
    if not settings.enabled:
        LOGGER.info("MaxFuse is disabled in configuration; nothing to do")
        if active_reporter:
            active_reporter.add_note("Stage skipped because maxfuse.enabled is false.")
        return 0

    reference_path = project_asset_path(settings.reference_adata_path)
    target_path = project_asset_path(
        settings.target_adata_path or config.general.anndata_path
    )
    mapping_path = project_asset_path(settings.feature_mapping_path)
    asset_root = project_asset_path(settings.asset_folder)
    for label, path in (
        ("reference AnnData", reference_path),
        ("target AnnData", target_path),
        ("feature mapping", mapping_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"MaxFuse {label} not found: {path}")
    if reference_path == target_path:
        raise ValueError("MaxFuse reference and target AnnData paths must differ")

    if active_reporter:
        active_reporter.add_input(
            "maxfuse_reference",
            reference_path,
            "Single scRNA-seq reference used as MaxFuse modality 1.",
        )
        active_reporter.add_input(
            "maxfuse_target",
            target_path,
            "Target IMC AnnData used as MaxFuse modality 2 without in-place modification.",
        )
        active_reporter.add_input(
            "maxfuse_feature_mapping",
            mapping_path,
            "Weakly linked target-marker to reference-gene mapping.",
        )

    LOGGER.info("Reading MaxFuse reference: %s", reference_path)
    reference = ad.read_h5ad(reference_path)
    LOGGER.info("Reading MaxFuse target: %s", target_path)
    target = ad.read_h5ad(target_path)
    estimated_reference = dense_size_gib(
        (reference.n_obs, min(reference.n_vars, settings.reference_active_features))
    )
    estimated_target = dense_size_gib((target.n_obs, target.n_vars))
    LOGGER.info(
        "Dense active-array estimate: reference %.2f GiB, target %.2f GiB",
        estimated_reference,
        estimated_target,
    )
    if active_reporter:
        active_reporter.add_metric(
            "estimated_reference_active_gib", round(estimated_reference, 3)
        )
        active_reporter.add_metric(
            "estimated_target_active_gib", round(estimated_target, 3)
        )

    mapping = read_feature_mapping(
        mapping_path,
        target_column=settings.target_feature_column,
        reference_column=settings.reference_feature_column,
        filter_column=settings.mapping_filter_column,
    )
    prepared = prepare_maxfuse_inputs(reference, target, mapping, settings)
    result = run_maxfuse_matching(prepared, reference, target, settings)

    asset_root.mkdir(parents=True, exist_ok=True)
    matches_path = asset_root / "maxfuse_matches.csv.gz"
    transfer_path = asset_root / "maxfuse_transfer.h5ad"
    mapping_used_path = asset_root / "feature_mapping_used.csv"
    _atomic_csv(result.matches, matches_path)
    _atomic_csv(prepared.retained_mapping, mapping_used_path)
    transfer = build_transfer_anndata(
        target,
        result.matches,
        prepared.retained_mapping,
        settings,
        reference_path=reference_path,
    )
    _atomic_h5ad(transfer, transfer_path)
    matched_transcriptome_path: Path | None = None
    if settings.write_matched_transcriptomes:
        matched_transcriptome_path = asset_root / "maxfuse_matched_transcriptomes.h5ad"
        matched_transcriptomes = build_matched_transcriptomes(
            reference,
            result.matches,
            settings,
        )
        _atomic_h5ad(matched_transcriptomes, matched_transcriptome_path)

    direct_root = Path("maxfuse_report")
    figures_dir = (
        optional_category_output_path("figures", direct_root / "figures") / "maxfuse"
    )
    tables_dir = (
        optional_category_output_path("tables", direct_root / "tables") / "maxfuse"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    feature_audit_path = tables_dir / "feature_mapping_audit.csv"
    timing_path = tables_dir / "maxfuse_stage_timings.csv"
    prepared.feature_audit.to_csv(feature_audit_path, index=False)
    import pandas as pd

    pd.Series(result.timings_minutes, name="minutes").rename_axis("operation").to_csv(
        timing_path
    )
    report = generate_maxfuse_report(
        reference,
        target,
        result.matches,
        prepared.retained_mapping,
        settings,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        resolve_path=lambda value: project_asset_path(value),
    )
    report.tables.extend([feature_audit_path, timing_path])

    if active_reporter:
        active_reporter.add_asset(
            "maxfuse_assets",
            asset_root,
            "Reusable target-unique match table, transfer AnnData, and exact linked-feature mapping.",
        )
        active_reporter.add_asset(
            "maxfuse_matches",
            matches_path,
            "Target-unique MaxFuse matches with stable reference and target cell identities.",
        )
        active_reporter.add_asset(
            "maxfuse_transfer",
            transfer_path,
            "Target-indexed score and transferred-label AnnData for population QC.",
        )
        if matched_transcriptome_path is not None:
            active_reporter.add_asset(
                "maxfuse_matched_transcriptomes",
                matched_transcriptome_path,
                "Optional target-indexed confident reference transcriptomes.",
            )
        active_reporter.add_file(
            "table",
            feature_audit_path,
            "Availability, variability, and retention decision for every requested weak link.",
        )
        active_reporter.add_file(
            "table",
            timing_path,
            "Elapsed minutes for each MaxFuse algorithm phase.",
        )
        for path in report.figures:
            active_reporter.add_file("figure", path)
        for path in report.tables:
            if path not in {feature_audit_path, timing_path}:
                active_reporter.add_file("table", path)
        for warning in report.warnings:
            active_reporter.add_warning(warning)
        for name, value in report.metrics.items():
            active_reporter.add_metric(name, value)
        active_reporter.add_metric(
            "retained_shared_features", len(prepared.retained_mapping)
        )
        active_reporter.add_metric(
            "reference_active_features", len(prepared.reference_active_features)
        )
        for name, value in result.timings_minutes.items():
            active_reporter.add_metric(
                f"timing_minutes.{name}",
                round(float(value), 3),
            )
        try:
            active_reporter.add_metric(
                "maxfuse_version",
                importlib.metadata.version("maxfuse"),
            )
        except importlib.metadata.PackageNotFoundError:
            active_reporter.add_warning(
                "MaxFuse package metadata was unavailable despite a successful import."
            )
        active_reporter.add_note(
            "MaxFuse scores are similarities, not calibrated probabilities. "
            "The report threshold affects figures and DEGs but not the reusable match table."
        )

    LOGGER.info(
        "MaxFuse complete: %d/%d target cells retained; assets=%s",
        len(result.matches),
        target.n_obs,
        asset_root,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    return run_pipeline(argv)


if __name__ == "__main__":
    raise SystemExit(main())
