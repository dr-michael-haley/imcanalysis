from __future__ import annotations

from pathlib import Path

from SpatialBiologyToolkit.napari_sbt.models import (
    CellScope,
    ExperimentManifest,
    segmentation_qc_classes,
)
from SpatialBiologyToolkit.napari_sbt.setup import (
    WORKFLOW_PRESENTATIONS,
    discover_workspaces,
    setup_checks,
    setup_is_ready,
    suggest_identity_columns,
    workspace_destination,
)
from SpatialBiologyToolkit.pipeline.manifests import write_yaml


def _manifest(project: Path, *, name: str = "Population review"):
    return ExperimentManifest(
        name=name,
        workflow_mode="population_qc",
        project_root=str(project),
        anndata_path=str(project / "cells.h5ad"),
        masks_folder=str(project / "masks"),
        images_folders=[str(project / "images")],
        cell_scope=CellScope(
            mode="all_cells",
            snapshot_sha256="abc",
            eligible_cell_count=12,
            total_cell_count=12,
            represented_roi_count=3,
        ),
        classes=segmentation_qc_classes(),
    )


def test_workflow_presentations_cover_every_persisted_mode():
    assert {item.mode for item in WORKFLOW_PRESENTATIONS} == {
        "data_exploration",
        "population_qc",
        "classification",
        "cell_labeling",
        "population_curation",
        "full_workspace",
    }
    assert next(
        item for item in WORKFLOW_PRESENTATIONS if item.mode == "full_workspace"
    ).advanced


def test_workspace_discovery_is_bounded_and_reports_missing_sources(tmp_path: Path):
    container = tmp_path / "napari_sbt"
    direct = container / "review"
    nested = container / "ignored" / "nested"
    direct.mkdir(parents=True)
    nested.mkdir(parents=True)
    write_yaml(direct / "experiment.yaml", _manifest(tmp_path))
    write_yaml(nested / "experiment.yaml", _manifest(tmp_path, name="Too deep"))

    summaries = discover_workspaces(container)

    assert [item.name for item in summaries] == ["Population review"]
    assert summaries[0].loadable
    assert summaries[0].level == "check"
    assert "processed cell data is missing" in summaries[0].warnings


def test_setup_readiness_requires_explicit_integrity_check(tmp_path: Path):
    adata = tmp_path / "cells.h5ad"
    adata.touch()
    masks = tmp_path / "masks"
    images = tmp_path / "images"
    masks.mkdir()
    images.mkdir()
    destination = workspace_destination(tmp_path / "napari_sbt", "My review")
    common = dict(
        workspace_name="My review",
        workspace_path=destination,
        workflow_mode="population_qc",
        anndata_path=adata,
        has_in_memory_anndata=False,
        masks_folder=masks,
        image_folders=[str(images)],
        extra_image_folders=[],
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        normalization_path=None,
    )

    unchecked = setup_checks(**common, integrity_current=False)
    checked = setup_checks(**common, integrity_current=True)

    assert not setup_is_ready(unchecked)
    assert setup_is_ready(checked)
    assert next(item for item in checked if item.key == "normalization").level == (
        "optional"
    )


def test_identity_column_suggestions_use_conventional_names_only():
    assert suggest_identity_columns(["sample", "ROI ID", "Object Number"]) == (
        "ROI ID",
        "Object Number",
    )
    assert suggest_identity_columns(["sample", "cluster"]) == (None, None)
