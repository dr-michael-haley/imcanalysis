"""Run the complete HyPERSTAC workflow in one managed process/allocation."""

from __future__ import annotations

import logging

from SpatialBiologyToolkit.scripts import cox_survival
from SpatialBiologyToolkit.scripts._hyperstac_common import (
    load_runtime,
    run_model,
    run_permutation,
    run_preprocess,
    run_stability,
    run_visualisation,
)


def main() -> None:
    config = load_runtime("full")
    run_preprocess()
    run_model()
    run_permutation()
    run_visualisation()
    if not config.hyperstac.full_include_survival:
        logging.info(
            "HyPERSTAC full workflow completed after visualisation because "
            "hyperstac.full_include_survival is false."
        )
        return
    cox_survival.main()
    run_stability()


if __name__ == "__main__":
    main()
