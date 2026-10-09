from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.scripts.config_and_utils import (
    GeneralConfig,
    get_stage_run_record,
    save_pipeline_anndata,
)

ad = pytest.importorskip("anndata")


def test_pipeline_save_nullable_indexes_and_metadata(tmp_path: Path):
    dtype = pd.StringDtype(storage="python")
    data = ad.AnnData(np.zeros((3, 2), dtype=np.float32))
    data.obs_names = pd.Index(["cell-1", "cell-2", "cell-3"], dtype=dtype)
    data.var_names = pd.Index(["CD3", "CD4"], dtype=dtype)
    data.obs["label"] = pd.array(["T", pd.NA, "B"], dtype=dtype)
    data.var["description"] = pd.array(["T cells", pd.NA], dtype=dtype)
    config = GeneralConfig(anndata_path=str(tmp_path / "cells.h5ad"))

    with ad.settings.override(allow_write_nullable_strings=False):
        path = save_pipeline_anndata(
            adata=data, general_config=config, stage_name="vis"
        )
        assert ad.settings.allow_write_nullable_strings is False

    restored = ad.read_h5ad(path)
    assert list(restored.obs_names) == list(data.obs_names)
    assert list(restored.var_names) == list(data.var_names)
    assert restored.obs["label"].iloc[0] == "T"
    assert pd.isna(restored.obs["label"].iloc[1])
    assert pd.isna(restored.var["description"].iloc[1])
    np.testing.assert_array_equal(restored.X, data.X)
    assert get_stage_run_record(restored, config, "vis") is not None


def test_pipeline_retry_restores_nullable_setting_after_failure(tmp_path: Path):
    class BrokenAnnData:
        uns = {}
        attempts = 0

        def write_h5ad(self, destination, **kwargs):
            self.attempts += 1
            assert ad.settings.allow_write_nullable_strings is True
            raise RuntimeError("simulated write failure")

    data = BrokenAnnData()
    config = GeneralConfig(anndata_path=str(tmp_path / "cells.h5ad"))
    with ad.settings.override(allow_write_nullable_strings=False):
        with pytest.raises(RuntimeError, match="simulated write failure"):
            save_pipeline_anndata(adata=data, general_config=config, stage_name="vis")
        assert ad.settings.allow_write_nullable_strings is False
    assert data.attempts == 2


def test_pipeline_save_scan_mapping_list_roundtrip(tmp_path: Path):
    from SpatialBiologyToolkit.config.models import BioBatchNetConfig
    from SpatialBiologyToolkit.scripts.config_and_utils import build_uns_config_snapshot

    scan = BioBatchNetConfig(biobatchnet_scan_parameter_sets=[
        {"name": "latent20", "latent_dim": 20, "extra_params": {"save_dir": "model20"}},
        {"name": "latent25", "latent_dim": 25},
    ])
    data = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    general = GeneralConfig(anndata_path=str(tmp_path / "scan.h5ad"))
    path = save_pipeline_anndata(adata=data, general_config=general,
                                stage_name="BioBatchNetProcess", stage_config=scan)
    restored = ad.read_h5ad(path)
    record = get_stage_run_record(restored, general, "BioBatchNetProcess")
    saved = record["config"]["biobatchnet_scan_parameter_sets"]
    assert list(saved) == ["item_000000", "item_000001"]
    assert saved["item_000000"]["extra_params"]["save_dir"] == "model20"
    assert saved["item_000001"]["latent_dim"] == 25
    assert build_uns_config_snapshot({"values": [15, 30, 50]}) == {"values": [15, 30, 50]}
