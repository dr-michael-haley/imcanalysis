from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
napari = pytest.importorskip("napari")
ad = pytest.importorskip("anndata")
pytest.importorskip("qtpy")

from SpatialBiologyToolkit.napari_sbt.app import launch
from SpatialBiologyToolkit.napari_sbt.cohort import resolve_cohort
from SpatialBiologyToolkit.napari_sbt.models import (
    ExperimentManifest,
    segmentation_qc_classes,
)
from SpatialBiologyToolkit.napari_sbt.storage import save_experiment


def test_unified_dock_is_cohort_gated_and_rejects_context_clicks(tmp_path: Path):
    import tifffile

    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["r1", "r1", "r2"]),
            "ObjectNumber": [1, 2, 1],
            "leiden": pd.Categorical(["target", "other", "other"]),
        },
        index=["a", "b", "c"],
    )
    data = ad.AnnData(np.zeros((3, 1), dtype=np.float32), obs=obs)
    adata_path = tmp_path / "cells.h5ad"
    data.write_h5ad(adata_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    tifffile.imwrite(
        masks / "r1.tiff",
        np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32),
    )
    tifffile.imwrite(
        masks / "r2.tiff",
        np.array([[1, 1], [0, 0]], dtype=np.int32),
    )
    preview = resolve_cohort(
        data,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        mode="obs_values",
        obs_column="leiden",
        obs_values=["target"],
    )
    root = tmp_path / "experiment"
    (root / "cohort").mkdir(parents=True)
    preview.eligible_cells.to_parquet(
        root / "cohort" / "eligible_cells.parquet", index=False
    )
    manifest = ExperimentManifest(
        name="GUI cohort",
        anndata_path=str(adata_path),
        masks_folder=str(masks),
        cell_scope=preview.scope(
            mode="obs_values", obs_column="leiden", obs_values=["target"]
        ),
        classes=segmentation_qc_classes(),
    )
    save_experiment(manifest, root)

    viewer = napari.Viewer(show=False)
    try:
        _, controller, _dock = launch(viewer=viewer, experiment=root)
        assert [
            controller.tabs.tabText(index)
            for index in range(controller.tabs.count())
        ] == [
            "Setup",
            "Explore",
            "Classify",
            "Regions & Export",
            "Layers & Status",
        ]
        assert set(np.unique(viewer.layers["classification_cohort"].data)) == {0, 1}
        assert not viewer.layers["excluded_segmentation_context"].visible
        assert controller.roi_combo.count() == 1
        assert "1 eligible cells / 3 total cells" in controller.scope_label.text()

        controller._on_cohort_click(
            viewer.layers["classification_cohort"],
            SimpleNamespace(type="mouse_press", position=(0, 0)),
        )
        assert controller.current_selected_object is None
        assert "outside this experiment" in controller.selected_cell_label.text()

        controller.show_empty_rois.setChecked(True)
        controller.refresh_rois()
        assert {controller.roi_combo.itemText(index) for index in range(2)} == {
            "r1",
            "r2",
        }
    finally:
        viewer.close()
