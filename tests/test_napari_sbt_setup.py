from __future__ import annotations

from pathlib import Path

from SpatialBiologyToolkit.napari_sbt.models import (
    CellScope,
    ExperimentManifest,
    segmentation_qc_classes,
)
from SpatialBiologyToolkit.napari_sbt.setup import (
    WORKFLOW_PRESENTATIONS,
    discover_dataset_assets,
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
        "dataset_maintenance",
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


def test_dataset_asset_discovery_is_shallow_and_finds_conventional_inputs(
    tmp_path: Path,
):
    primary = tmp_path / "anndata.h5ad"
    alternative_folder = tmp_path / "processed_data"
    alternative_folder.mkdir()
    alternative = alternative_folder / "reviewed.h5ad"
    primary.touch()
    alternative.touch()
    masks = tmp_path / "cell_masks"
    images = tmp_path / "Images"
    aligned = tmp_path / "matrix_images_alligned"
    for folder in (masks, images, aligned):
        folder.mkdir()
    hidden_workspace_input = tmp_path / "napari_sbt" / "review" / "inputs"
    hidden_workspace_input.mkdir(parents=True)
    (hidden_workspace_input / "snapshot.h5ad").touch()
    unrelated_nested = tmp_path / "results" / "nested"
    unrelated_nested.mkdir(parents=True)
    (unrelated_nested / "ignored.h5ad").touch()

    suggestions = discover_dataset_assets(tmp_path)

    assert suggestions.anndata_candidates == (primary, alternative)
    assert suggestions.masks_candidates == (masks,)
    assert suggestions.image_candidates == (images, aligned)
    assert hidden_workspace_input / "snapshot.h5ad" not in (
        suggestions.anndata_candidates
    )
    assert unrelated_nested / "ignored.h5ad" not in suggestions.anndata_candidates


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


def test_dataset_maintenance_allows_optional_mask_and_image_assets(tmp_path: Path):
    adata = tmp_path / "cells.h5ad"
    adata.touch()
    checks = setup_checks(
        workspace_name="Repair data",
        workspace_path=workspace_destination(tmp_path / "napari_sbt", "Repair data"),
        workflow_mode="dataset_maintenance",
        anndata_path=adata,
        has_in_memory_anndata=False,
        masks_folder=None,
        image_folders=[],
        extra_image_folders=[],
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        normalization_path=None,
        integrity_current=True,
    )

    assert setup_is_ready(checks)
    assert next(item for item in checks if item.key == "masks").level == "optional"
    assert next(item for item in checks if item.key == "images").level == "optional"


def test_identity_column_suggestions_use_conventional_names_only():
    assert suggest_identity_columns(["sample", "ROI ID", "Object Number"]) == (
        "ROI ID",
        "Object Number",
    )
    assert suggest_identity_columns(["sample", "cluster"]) == (None, None)
