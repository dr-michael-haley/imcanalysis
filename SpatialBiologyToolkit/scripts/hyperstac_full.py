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
    include_survival = config.hyperstac.full_include_survival
    if include_survival:
        cox_survival.main()
    else:
        logging.info(
            "Skipping the optional Cox overlay because "
            "hyperstac.full_include_survival is false; the clustering comparison still runs."
        )
    run_stability(include_survival=include_survival)


if __name__ == "__main__":
    main()
