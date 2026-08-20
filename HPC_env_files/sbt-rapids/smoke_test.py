"""CPU-safe API check for the inactive modern SBT RAPIDS candidate."""

from __future__ import annotations

import inspect
from importlib.metadata import version

import rapids_singlecell as rsc


def main() -> None:
    """Verify the RAPIDS-singlecell APIs used by the SBT stage are present."""

    required_paths = (
        (rsc.get, "anndata_to_GPU"),
        (rsc.get, "anndata_to_CPU"),
        (rsc.pp, "pca"),
        (rsc.pp, "harmony_integrate"),
        (rsc.pp, "neighbors"),
        (rsc.tl, "umap"),
        (rsc.tl, "leiden"),
    )
    for namespace, name in required_paths:
        assert callable(getattr(namespace, name, None)), name

    harmony_parameters = inspect.signature(rsc.pp.harmony_integrate).parameters
    assert "flavor" in harmony_parameters, tuple(harmony_parameters)

    print(
        "RAPIDS_API_SMOKE_PASS "
        f"rapids-singlecell={version('rapids-singlecell')} "
        f"cugraph={version('cugraph')}"
    )


if __name__ == "__main__":
    main()

