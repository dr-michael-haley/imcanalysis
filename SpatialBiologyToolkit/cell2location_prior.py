"""Spot-specific soft cell-count priors for cell2location 0.1.5.

The extension substitutes only the existing ``n_s_cells_per_location`` Gamma
distribution. All other upstream model terms and the variational guide remain
unchanged. The mean is indexed by global observation indices on every forward
pass, including shuffled training batches and posterior minibatches.
"""

from __future__ import annotations

import torch
import pyro.distributions as dist
from pyro.poutine.messenger import Messenger
from cell2location.models._cell2location_module import (
    LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel as BaseLocationModel,
)


class _CellCountPrior(Messenger):
    def __init__(self, mean: torch.Tensor, ratio: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.ratio = ratio

    def _pyro_sample(self, msg):
        if msg["name"] == "n_s_cells_per_location":
            msg["fn"] = dist.Gamma(self.mean * self.ratio, self.ratio)


class SpotCellCountModel(BaseLocationModel):
    """Upstream location model with Gamma(mean * ratio, ratio) at each spot."""

    def __init__(self, *args, spot_cell_count_prior, **kwargs):
        super().__init__(*args, **kwargs)
        prior = torch.as_tensor(spot_cell_count_prior, dtype=torch.float32).reshape(
            -1, 1
        )
        if (
            prior.shape[0] != self.n_obs
            or not torch.isfinite(prior).all()
            or (prior <= 0).any()
        ):
            raise ValueError(
                "spot_cell_count_prior must contain one finite positive mean per observation"
            )
        self.register_buffer("spot_cell_count_prior", prior)

    def forward(self, x_data, idx, batch_index):
        indices = idx.reshape(-1).long()
        with _CellCountPrior(
            self.spot_cell_count_prior[indices], self.N_cells_mean_var_ratio
        ):
            return super().forward(x_data, indices, batch_index)
