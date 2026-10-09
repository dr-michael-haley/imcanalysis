"""Exercise wrapper repeatability without requiring BioBatchNet training."""

import importlib
import importlib.util
import random
import sys
from types import ModuleType

import anndata as ad
import numpy as np
import pytest
from pydantic import ValidationError

from SpatialBiologyToolkit.config.models import BioBatchNetConfig


@pytest.fixture
def stage(monkeypatch):
    if importlib.util.find_spec("biobatchnet") is None:
        dependency = ModuleType("biobatchnet")
        dependency.correct_batch_effects = None
        monkeypatch.setitem(sys.modules, "biobatchnet", dependency)
    module = importlib.import_module(
        "SpatialBiologyToolkit.scripts.basic_process_biobatchnet"
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = module.torch.get_rng_state() if module.torch is not None else None
    yield module
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    if torch_state is not None:
        module.torch.set_rng_state(torch_state)


def test_seed_restarts_initialization_for_each_fit(stage, monkeypatch):
    calls = []

    def stochastic_model(**kwargs):
        # These draws stand in for model construction, shuffling and VAE sampling.
        assert "random_state" not in kwargs
        draws = [random.random(), np.random.random()]
        if stage.torch is not None:
            draws.append(stage.torch.rand(1).item())
        calls.append(draws)
        values = np.full((len(kwargs["data"]), kwargs["latent_dim"]), sum(draws))
        return values, values.copy()

    monkeypatch.setattr(stage, "correct_batch_effects", stochastic_model)
    source = ad.AnnData(np.ones((4, 2)))
    source.obs["batch"] = ["a", "a", "b", "b"]
    results = []
    for seed in (42, 42, 7):
        candidate = source.copy()
        stage.run_biobatchnet_correction(
            candidate, "batch", latent_dim=3, epochs=1, device="cpu",
            use_raw=False, random_state=seed,
        )
        assert candidate.uns["biobatchnet"]["random_state"] == seed
        np.testing.assert_array_equal(candidate.X, source.X)
        results.append(candidate.obsm["X_biobatchnet"])
    assert calls[0] == calls[1]
    assert calls[0] != calls[2]
    np.testing.assert_array_equal(results[0], results[1])


def test_null_seed_preserves_rng_and_legacy_metadata(stage, monkeypatch):
    def forbidden_seed(*args, **kwargs):
        raise AssertionError("Unseeded calls must not reset random generators")

    monkeypatch.setattr(random, "seed", forbidden_seed)
    monkeypatch.setattr(np.random, "seed", forbidden_seed)
    if stage.torch is not None:
        monkeypatch.setattr(stage.torch, "manual_seed", forbidden_seed)
    monkeypatch.setattr(
        stage, "correct_batch_effects",
        lambda **kwargs: (np.ones((2, 3)), None),
    )
    candidate = ad.AnnData(np.ones((2, 2)))
    candidate.obs["batch"] = ["a", "b"]
    stage.run_biobatchnet_correction(candidate, "batch", device="cpu", use_raw=False)
    assert "random_state" not in candidate.uns["biobatchnet"]


def test_scan_members_receive_configured_seed(stage, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        stage, "run_biobatchnet_correction",
        lambda adata, **kwargs: seen.append(kwargs["random_state"]),
    )
    monkeypatch.setattr(stage, "save_pipeline_anndata", lambda **kw: kw["override_path"])
    config = BioBatchNetConfig(random_state=42, biobatchnet_run_postprocess=False)
    for dimension in (20, 25, 30):
        result = stage._run_single_parameter_set(
            ad.AnnData(np.ones((2, 2))), batch_key="batch",
            general_config=stage.GeneralConfig(), biobatchnet_config=config,
            stage_name="BioBatchNetProcess", base_params=config.biobatchnet_params,
            overrides={"latent_dim": dimension}, label=f"latent{dimension}",
            base_output_path=tmp_path / "bbn.h5ad", base_qc_dir=tmp_path,
        )
        assert result["random_state"] == 42
    assert seen == [42, 42, 42]


def test_seed_config_bounds():
    assert BioBatchNetConfig().random_state is None
    assert BioBatchNetConfig(random_state=0).random_state == 0
    for seed in (-1, 4294967296):
        with pytest.raises(ValidationError):
            BioBatchNetConfig(random_state=seed)
