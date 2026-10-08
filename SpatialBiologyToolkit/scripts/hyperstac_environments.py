"""Managed spatial environment discovery from saved HyPERSTAC embeddings."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from SpatialBiologyToolkit.config import load_config
from SpatialBiologyToolkit.reporting import (
    bootstrap_stage_reporting,
    category_output_path,
    get_active_reporter,
    project_asset_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SBT_CONFIG", "config.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    os.environ["SBT_CONFIG"] = str(config_path)
    os.environ.setdefault("SBT_PROJECT_ROOT", str(config_path.parent))
    bootstrap_stage_reporting("hyperstac-environments")
    reporter = get_active_reporter()
    config = load_config(config_path)
    from SpatialBiologyToolkit.hyperstac.environments import run

    try:
        output = run(
            config,
            project_asset_path(config.hyperstac_environments.output_folder),
            reporter,
        )
        # Keep full reusable tables with models; copy compact human outputs to report.
        for name in [
            "scan_scorecard.csv",
            "patient_stability.csv",
            "stability.png",
            "completed.json",
            "input_signature.json",
        ]:
            category = (
                "figures"
                if name.endswith(".png")
                else "tables"
                if name.endswith(".csv")
                else "summaries"
            )
            dest = category_output_path(category) / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / name, dest)
            reporter.add_file(
                "figure"
                if category == "figures"
                else "table"
                if category == "tables"
                else "summary",
                dest,
                name,
            )
        reporter.finalize(status="completed")
    except Exception as error:
        reporter.finalize(status="failed", error=error)
        raise


if __name__ == "__main__":
    main()
