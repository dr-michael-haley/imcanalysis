"""Run the complete HyPERSTAC workflow in one managed process/allocation."""

from __future__ import annotations

from SpatialBiologyToolkit.scripts import cox_survival
from SpatialBiologyToolkit.scripts._hyperstac_common import (
    run_model,
    run_permutation,
    run_preprocess,
    run_stability,
    run_visualisation,
)


def main() -> None:
    run_preprocess()
    run_model()
    run_permutation()
    run_visualisation()
    cox_survival.main()
    run_stability()


if __name__ == "__main__":
    main()
