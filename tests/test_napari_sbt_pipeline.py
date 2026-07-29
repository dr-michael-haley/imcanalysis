from __future__ import annotations

from pathlib import Path

from SpatialBiologyToolkit.config import PipelineConfig
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import MODES, STAGE_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cellfeat_registry_resources_environment_and_docs_are_aligned():
    stage = STAGE_REGISTRY["cellfeat"]
    assert stage.catalogue_order == 37
    assert stage.depends_on == []
    assert stage.groups == []
    assert stage.environment_keys == ["segmentation"]
    assert stage.config_sections == ["general", "napari_sbt"]
    assert stage.python_modules == [
        "SpatialBiologyToolkit.scripts.cell_features"
    ]
    assert stage.requires_assets == ["napari_sbt_experiments", "masks"]
    assert all("cellfeat" not in mode.stages for mode in MODES)
    assert (REPO_ROOT / stage.documentation_path).is_file()
    wrapper = (REPO_ROOT / stage.slurm_script).read_text(encoding="utf-8")
    assert "#SBATCH -t 1-0" in wrapper
    assert "#SBATCH -n 8" in wrapper
    assert "#SBATCH --mem=64G" in wrapper
    assert "#SBATCH -p himem" in wrapper
    assert "#@ENV:  imc_segmentation" in wrapper


def test_napari_sbt_config_and_asset_role_resolve_from_project(tmp_path: Path):
    context = initialize_project(tmp_path / "project")
    config = PipelineConfig()
    assert config.napari_sbt.experiment_folder == "napari_sbt"
    assert config.napari_sbt.worker_count == 8
    assets = {asset.role: asset for asset in resolve_assets(context.config, context.root)}
    assert assets["napari_sbt_experiments"].path == (
        context.root / "napari_sbt"
    ).resolve()
    assert assets["napari_sbt_experiments"].lifecycle == "human_output"


def test_cellfeat_temporary_project_plan_is_ready_without_submission(tmp_path: Path):
    context = initialize_project(tmp_path / "project")
    experiment = context.root / "napari_sbt" / "subclass"
    experiment.mkdir(parents=True)
    (experiment / "experiment.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    masks = context.root / context.config.general.masks_folder
    masks.mkdir(parents=True)
    (masks / "roi.tiff").write_bytes(b"placeholder")

    plan = build_run_plan(context, ["cellfeat"])

    assert plan.ready, plan.errors
    assert [stage.name for stage in plan.resolved_stages] == ["cellfeat"]
    assert plan.resolved_stages[0].depends_on == []
    assert "napari_sbt_experiments" in plan.resolved_stages[0].requires_assets
