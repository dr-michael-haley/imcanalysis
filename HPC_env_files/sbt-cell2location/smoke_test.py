"""Tiny CPU integration check, including minibatch priors and saved-model reload.

Run only on a compute/development node. No project data or scheduler is used.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from SpatialBiologyToolkit.cell2location_analysis import (
    fit_reference,
    fit_mapping,
    prepare_visium,
    save_result,
    PRIOR_MEAN,
)
from SpatialBiologyToolkit.config import Cell2locationConfig


def main() -> None:
    from cell2location.models import Cell2location, RegressionModel
    from SpatialBiologyToolkit.cell2location_prior import SpotCellCountModel
    import pyro
    import torch

    rng = np.random.default_rng(4)
    var = pd.DataFrame(index=[f"g{i}" for i in range(24)])
    ref = ad.AnnData(
        rng.poisson(4, (40, 24)).astype("float32"),
        obs=pd.DataFrame(
            {"cell_type": ["A"] * 20 + ["B"] * 20, "sample": ["R1", "R2"] * 20},
            index=[f"r{i}" for i in range(40)],
        ),
        var=var,
    )
    vis = ad.AnnData(
        rng.poisson(30, (8, 24)).astype("float32"),
        obs=pd.DataFrame(
            {
                "library_id": ["L1"] * 4 + ["L2"] * 4,
                "imc_cells": [1, 4, 8, 12, 2, 5, 9, 15],
            },
            index=[f"s{i}" for i in range(8)],
        ),
        var=var,
    )
    vis.obsm["spatial"] = rng.uniform(0, 100, (8, 2))
    settings = Cell2locationConfig(
        reference_batch_key="sample",
        cell_count_prior_obs_key="imc_cells",
        reference_max_epochs=2,
        mapping_max_epochs=2,
        posterior_samples=3,
        mapping_batch_size=3,
        posterior_batch_size=3,
        min_shared_genes=5,
        accelerator="cpu",
    )
    fitted_ref = fit_reference(ref, settings)
    fitted = fit_mapping(
        prepare_visium([vis], settings), fitted_ref.signatures, settings
    )
    assert fitted.adata.obsm["q05_cell_abundance_w_sf"].shape == (8, 2)
    assert np.isfinite(fitted.adata.obsm["means_cell_abundance_w_sf"]).all().all()
    assert fitted.adata.obs[PRIOR_MEAN].tolist() == vis.obs.imc_cells.tolist()
    model = fitted.model.module.model
    assert isinstance(model, SpotCellCountModel)
    # A reordered minibatch, followed by a singleton, must use original spot indices.
    for indices in [torch.tensor([6, 0, 4]), torch.tensor([7])]:
        trace = pyro.poutine.trace(model).get_trace(
            torch.as_tensor(fitted.adata.X[indices.numpy()].toarray()),
            indices,
            torch.as_tensor(
                fitted.adata.obs["_scvi_batch"].to_numpy()[indices.numpy()],
                dtype=torch.long,
            ).reshape(-1, 1),
        )
        mean = trace.nodes["n_s_cells_per_location"]["fn"].mean
        torch.testing.assert_close(
            mean.flatten(), model.spot_cell_count_prior[indices].flatten()
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_result(fitted_ref, root / "reference")
        save_result(fitted, root / "mapping")
        loaded = Cell2location.load(
            str(root / "mapping" / "model"),
            adata=ad.read_h5ad(root / "mapping" / "posterior.h5ad"),
            accelerator="cpu",
        )
        assert isinstance(loaded.module.model, SpotCellCountModel)
        torch.testing.assert_close(
            loaded.module.model.spot_cell_count_prior, model.spot_cell_count_prior
        )
        RegressionModel.load(
            str(root / "reference" / "model"),
            adata=ad.read_h5ad(root / "reference" / "posterior.h5ad"),
            accelerator="cpu",
        )
    print(
        "cell2location CPU smoke passed: reference, joint mapping, indexed priors, export and reload"
    )


if __name__ == "__main__":
    main()
