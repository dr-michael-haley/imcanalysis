from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.population_qc import (
    PosteriorPublicationConfig,
    map_posterior_labels,
    publish_posterior_mapping,
)


def _adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.ones((4, 2), dtype=float),
        obs=pd.DataFrame(
            {"leiden_1.0": pd.Categorical(["0", "1", "0", "2"])},
            index=["cell-0", "cell-1", "cell-2", "cell-3"],
        ),
        var=pd.DataFrame(index=["CD3", "CD20"]),
    )


def test_map_posterior_labels_overwrites_existing_output_by_default_and_requires_complete_mapping() -> None:
    observations = _adata().obs
    observations["posterior"] = "legacy label"
    mapped = map_posterior_labels(
        observations,
        source_key="leiden_1.0",
        output_key="posterior",
        mapping={"0": "T cell", "1": "B cell", "2": "Myeloid"},
        categories=["T cell", "B cell", "Myeloid"],
    )

    assert mapped.cat.categories.tolist() == ["T cell", "B cell", "Myeloid"]
    assert mapped.astype(str).tolist() == ["T cell", "B cell", "T cell", "Myeloid"]
    with pytest.raises(ValueError, match="does not cover"):
        map_posterior_labels(
            observations,
            source_key="leiden_1.0",
            output_key="posterior",
            mapping={"0": "T cell"},
            categories=["T cell"],
        )
    with pytest.raises(ValueError, match="must differ"):
        map_posterior_labels(
            observations,
            source_key="leiden_1.0",
            output_key="leiden_1.0",
            mapping={"0": "T cell", "1": "B cell", "2": "Myeloid"},
            categories=["T cell", "B cell", "Myeloid"],
        )


def test_publish_posterior_mapping_stages_h5ad_and_writes_one_table(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _adata()
    source.obs["posterior_label_leiden_1.0"] = "legacy label"
    h5ad_path = tmp_path / "source.h5ad"
    source.write_h5ad(h5ad_path)
    mapping_path = tmp_path / "posterior_mapping.csv"
    pd.DataFrame(
        {
            "source_population": ["0", "1", "2"],
            "proposed_label": ["T cell", "B cell", "Myeloid"],
        }
    ).to_csv(mapping_path, index=False)
    zarr_path = tmp_path / "source.zarr"
    zarr_path.mkdir()

    class FakeSpatialData:
        def __init__(self) -> None:
            self.tables = {"cells": source.copy()}
            self.write_calls: list[tuple[str, bool]] = []

        def write_element(self, name: str, *, overwrite: bool) -> None:
            self.write_calls.append((name, overwrite))

    fake_sdata = FakeSpatialData()
    monkeypatch.setitem(
        sys.modules,
        "spatialdata",
        SimpleNamespace(read_zarr=lambda _path: fake_sdata),
    )
    artifact_root = tmp_path / "assessment"
    config = PosteriorPublicationConfig(
        zarr=zarr_path,
        table_name="cells",
        h5ad=h5ad_path,
        mapping_csv=mapping_path,
        source_key="leiden_1.0",
        output_key="posterior_label_leiden_1.0",
        artifact_root=artifact_root,
        heartbeat_seconds=60,
    )

    manifest = publish_posterior_mapping(config)

    assert fake_sdata.write_calls == [("cells", True)]
    assert manifest["zarr_table_write_count"] == 1
    written_h5ad = ad.read_h5ad(h5ad_path)
    assert written_h5ad.obs["posterior_label_leiden_1.0"].astype(str).tolist() == [
        "T cell",
        "B cell",
        "T cell",
        "Myeloid",
    ]
    assert fake_sdata.tables["cells"].obs["leiden_1.0"].astype(str).tolist() == [
        "0",
        "1",
        "0",
        "2",
    ]
    assert fake_sdata.tables["cells"].obs["posterior_label_leiden_1.0"].astype(str).tolist() == [
        "T cell",
        "B cell",
        "T cell",
        "Myeloid",
    ]
    finalization = json.loads(
        (artifact_root / "manifests" / "posterior_finalization.json").read_text()
    )
    assert finalization["zarr_write_count"] == 1
    assert (artifact_root / "tables" / "posterior_observation_labels.csv").exists()

    repeated = publish_posterior_mapping(config)
    assert fake_sdata.write_calls == [("cells", True)]
    assert repeated["publication_status"] == "already_published"
    assert repeated["this_invocation_zarr_write_count"] == 0
