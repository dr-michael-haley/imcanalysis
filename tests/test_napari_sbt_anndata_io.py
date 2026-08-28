from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.napari_sbt.anndata_io import write_h5ad_compat

ad = pytest.importorskip("anndata")


def _nullable_string_adata():
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    obs["nullable_label"] = pd.array(["first", pd.NA, "third"], dtype="string")
    return ad.AnnData(
        np.zeros((3, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["CD3"]),
    )


def test_in_memory_snapshot_writes_nullable_strings_without_mutating_source(
    tmp_path: Path,
) -> None:
    data = _nullable_string_adata()
    source_dtype = data.obs["nullable_label"].dtype
    previous_setting = ad.settings.allow_write_nullable_strings
    destination = tmp_path / "experiment" / "inputs" / "anndata.h5ad"

    destination.parent.mkdir(parents=True)
    write_h5ad_compat(data, destination)
    written = destination.resolve()

    assert written == destination.resolve()
    assert data.obs["nullable_label"].dtype == source_dtype
    assert ad.settings.allow_write_nullable_strings is previous_setting
    restored = ad.read_h5ad(written)
    assert restored.obs["nullable_label"].iloc[0] == "first"
    assert pd.isna(restored.obs["nullable_label"].iloc[1])
    assert restored.obs["nullable_label"].iloc[2] == "third"


def test_nullable_string_setting_is_restored_when_write_fails(tmp_path: Path) -> None:
    previous_setting = ad.settings.allow_write_nullable_strings

    class BrokenAnnData:
        def write_h5ad(self, _destination, **_kwargs) -> None:
            assert ad.settings.allow_write_nullable_strings is True
            raise RuntimeError("simulated write failure")

    with pytest.raises(RuntimeError, match="simulated write failure"):
        write_h5ad_compat(BrokenAnnData(), tmp_path / "broken.h5ad")

    assert ad.settings.allow_write_nullable_strings is previous_setting
