"""Standalone or managed ``sbt run cell2location`` analysis entry point."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def run_pipeline(argv: list[str] | None = None) -> int:
    """Run with scoped reporting environment, also safe for repeated notebook calls."""
    saved = {key: value for key, value in os.environ.items() if key.startswith("SBT_")}
    try:
        return _run_pipeline(argv)
    finally:
        for key in list(os.environ):
            if key.startswith("SBT_") and key not in saved:
                del os.environ[key]
        os.environ.update(saved)


def _run_pipeline(argv: list[str] | None = None) -> int:
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.cell2location_contract import (
        input_paths,
        preflight_errors,
        resolve_path,
        signature_path,
    )
    from SpatialBiologyToolkit.reporting import StageReporter, resolve_reporting_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SBT_CONFIG", "config.yaml"))
    parser.add_argument(
        "--action",
        choices=["reference", "map", "full"],
        help="Override action for direct invocation.",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    settings = config.cell2location
    if args.action:
        settings = settings.model_copy(update={"action": args.action})
    root = Path(os.environ.get("SBT_PROJECT_ROOT", str(config_path.parent))).resolve()
    os.environ["SBT_CONFIG"] = str(config_path)
    os.environ["SBT_PROJECT_ROOT"] = str(root)
    logging.basicConfig(level=config.logging.level)
    with StageReporter(resolve_reporting_context("cell2location")) as reporter:
        errors = preflight_errors(settings, root)
        if errors:
            raise ValueError("Cell2location preflight failed:\n" + "\n".join(errors))
        reporter.add_metric("action", settings.action)
        reporter.add_note(
            "Raw inputs are preserved. Results are written only to new reference/mapping directories."
        )
        reporter.add_note(
            f"Resolved cell2location settings: {settings.model_dump_json()}"
        )
        for role, path in input_paths(settings, root).items():
            reporter.add_input(role, path)

        import anndata as ad
        import pandas as pd
        from SpatialBiologyToolkit.cell2location_analysis import (
            fit_mapping,
            fit_reference,
            prepare_visium,
            read_signatures,
            save_result,
            attach_cell_count_prior,
        )
        from SpatialBiologyToolkit.cell2location_reports import report_result

        asset_root = resolve_path(root, settings.asset_folder)
        visium = None
        counts = None
        # Validate spatial inputs and prior alignment before expensive reference fitting.
        if settings.action in {"map", "full"}:
            sources = [
                ad.read_h5ad(resolve_path(root, source.path))
                for source in settings.visium_inputs
            ]
            visium = prepare_visium(
                sources,
                settings,
                library_ids=[s.library_id for s in settings.visium_inputs],
            )
            del sources
            if settings.cell_count_prior_csv:
                counts = pd.read_csv(
                    resolve_path(root, settings.cell_count_prior_csv),
                    dtype={"library_id": str, "barcode": str},
                )
            attach_cell_count_prior(visium, settings, counts=counts)
        if settings.action in {"reference", "full"}:
            reference = ad.read_h5ad(resolve_path(root, settings.reference_adata_path))
            result = fit_reference(reference, settings)
            del reference
            destination = save_result(result, asset_root / "reference")
            reporter.add_asset("cell2location_reference_model", destination)
            report_result(result, settings, reporter, operation="reference")
            signatures = result.signatures
            del result
        else:
            signatures = read_signatures(signature_path(settings, root))
        if visium is not None:
            result = fit_mapping(visium, signatures, settings, counts=counts)
            destination = save_result(result, asset_root / "mapping")
            reporter.add_asset("cell2location_mapping_model", destination)
            report_result(result, settings, reporter, operation="mapping")
        reporter.add_asset("cell2location_assets", asset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline())
