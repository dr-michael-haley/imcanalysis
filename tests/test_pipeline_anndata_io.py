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
