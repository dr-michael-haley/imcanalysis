"""Run in the dedicated environment; other environments skip the modelling checks."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("cell2location")
import anndata as ad

from SpatialBiologyToolkit.scripts.cell2location_analysis import run_pipeline


def _configure_environment(monkeypatch, tmp_path):
    for key in list(__import__("os").environ):
        if key.startswith("SBT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SBT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")


def test_full_workflow_reports_assets_and_mapping_only_reuse(tmp_path, monkeypatch):
    _configure_environment(monkeypatch, tmp_path)
    rng = np.random.default_rng(19)
    genes = pd.DataFrame(index=[f"g{i}" for i in range(12)])
    ref = ad.AnnData(
        rng.poisson(3, (24, 12)).astype(np.float32),
        obs=pd.DataFrame(
            {"cell_type": ["T"] * 12 + ["B"] * 12}, index=[f"r{i}" for i in range(24)]
        ),
        var=genes,
    )
    ref.write_h5ad(tmp_path / "reference.h5ad")
    for library in ["A", "B"]:
        vis = ad.AnnData(
            rng.poisson(20, (4, 12)).astype(np.float32),
            obs=pd.DataFrame(
                {"imc": [0, 4, 8, 12]}, index=[f"bc{i}" for i in range(4)]
            ),
            var=genes,
        )
        vis.obsm["spatial"] = rng.uniform(0, 100, (4, 2))
        vis.write_h5ad(tmp_path / f"{library}.h5ad")
    settings = {
        "action": "full",
        "visium_inputs": [
            {"path": f"{lib}.h5ad", "library_id": lib} for lib in ["A", "B"]
        ],
        "min_shared_genes": 5,
        "reference_max_epochs": 1,
        "mapping_max_epochs": 1,
        "posterior_samples": 3,
        "posterior_batch_size": 3,
        "mapping_batch_size": 3,
        "accelerator": "cpu",
        "cell_count_prior_obs_key": "imc",
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"cell2location": settings}))
    files = [tmp_path / name for name in ["reference.h5ad", "A.h5ad", "B.h5ad"]]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    assert run_pipeline(["--config", str(config)]) == 0
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    mapped = ad.read_h5ad(tmp_path / "cell2location/mapping/posterior.h5ad")
    assert mapped.n_obs == 8
    assert mapped.obs_names.is_unique
    manifests = list((tmp_path / "outputs/direct").glob("*/stage_manifest.yaml"))
    assert len(manifests) == 1
    manifest = yaml.safe_load(manifests[0].read_text())
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["mapping_libraries"] == 2
    assert list(manifests[0].parent.glob("figures/mapping_library_*.png"))
    assert (manifests[0].parent / "tables/mapping_spot_qc.csv").is_file()
    # Saved signatures can be reused without the scRNA source, using the standard global prior.
    settings.update(
        action="map",
        signatures_path="cell2location/reference/signatures.csv",
        reference_adata_path="not-required.h5ad",
        asset_folder="reuse",
        cell_count_prior_obs_key=None,
        mapping_batch_size=None,
    )
    config.write_text(yaml.safe_dump({"cell2location": settings}))
    assert run_pipeline(["--config", str(config)]) == 0
    assert (tmp_path / "reuse/mapping/posterior.h5ad").is_file()
    assert not (tmp_path / "reuse/reference").exists()


def test_preflight_failure_is_reported(tmp_path, monkeypatch):
    _configure_environment(monkeypatch, tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("cell2location:\n  action: reference\n")
    with pytest.raises(ValueError, match="required file is missing"):
        run_pipeline(["--config", str(config)])
    manifest_path = next((tmp_path / "outputs/direct").glob("*/stage_manifest.yaml"))
    assert yaml.safe_load(manifest_path.read_text())["status"] == "failed"
