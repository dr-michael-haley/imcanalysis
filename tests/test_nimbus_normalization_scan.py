from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from SpatialBiologyToolkit.config.models import (
    NimbusNormalizationScanConfig,
    PipelineConfig,
)
from SpatialBiologyToolkit.environments.registry import load_environment_registry
from SpatialBiologyToolkit.nimbus_normalization_scan import (
    analyze_normalization_scan,
    build_vmax_grid,
    plot_marker_scan,
    rank_rois_by_expression,
    resolve_marker_baseline_vmax,
    resolve_marker_lower_thresholds,
    resolve_scan_marker_inputs,
    select_scan_rois,
    select_rois_across_expression_range,
    summarize_intracellular_expression,
    write_suggested_normalization_dict,
)
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import MODES, STAGE_REGISTRY
from SpatialBiologyToolkit.scripts.nimbus_normalization_scan import (
    _build_marker_roi_selections,
    _load_baseline_normalization,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_scan(*, cliff: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows = []
    pixel_rows = []
    vmax_values = [5.0, 10.0, 20.0]
    base_scores = np.asarray([0.1] * 5 + [0.9] * 5, dtype=float)
    for vmax_index, vmax in enumerate(vmax_values):
        if cliff:
            scores = [
                np.full(10, 0.9),
                base_scores,
                np.full(10, 0.1),
            ][vmax_index]
        else:
            scores = base_scores + (vmax_index - 1) * 0.005
        for label, score in enumerate(scores, start=1):
            cell_rows.append(
                {
                    "marker": "CD3",
                    "vmax": vmax,
                    "roi": "ROI_1",
                    "label": label,
                    "nimbus_score": float(score),
                }
            )
        pixel_rows.append(
            {
                "marker": "CD3",
                "vmax": vmax,
                "roi": "ROI_1",
                "masked_pixel_count": 100,
                "saturated_pixel_fraction": 0.02,
                "lower_threshold": 1.0,
                "below_lower_threshold_fraction": 0.2,
            }
        )
    return pd.DataFrame(cell_rows), pd.DataFrame(pixel_rows)


def test_roi_sampling_is_reproducible_and_explicit_selection_preserves_order():
    rois = [f"ROI_{index}" for index in range(8)]
    first = select_scan_rois(rois, max_rois=3, random_seed=19)
    second = select_scan_rois(rois, max_rois=3, random_seed=19)
    assert first == second
    assert len(first) == 3
    assert select_scan_rois(
        rois,
        requested_rois=["ROI_5", "ROI_2"],
        max_rois=1,
    ) == ["ROI_2", "ROI_5"]
    with pytest.raises(ValueError, match="unavailable"):
        select_scan_rois(rois, requested_rois=["not-an-roi"])


def test_intracellular_expression_summary_uses_only_finite_masked_pixels_above_background():
    image = np.asarray([[0.0, 1.0, 2.0], [3.0, np.nan, 5.0]])
    mask = np.asarray([[0, 1, 1], [2, 2, 0]])
    summary = summarize_intracellular_expression(
        image, mask, background_value=1.0
    )
    assert summary.masked_pixel_count == 3
    assert summary.above_background_pixel_count == 2
    assert summary.above_background_fraction == pytest.approx(2 / 3)
    assert summary.mean_above_background == pytest.approx(2.5)

    with pytest.raises(ValueError, match="identical shapes"):
        summarize_intracellular_expression(image, mask[:, :2])
    with pytest.raises(ValueError, match="non-negative"):
        summarize_intracellular_expression(image, mask, background_value=-0.1)


def test_expression_range_selection_spans_ranked_roi_distribution():
    scores = {
        "ROI_4": 4.0,
        "ROI_0": 0.0,
        "ROI_2": 2.0,
        "ROI_5": 5.0,
        "ROI_1": 1.0,
        "ROI_3": 3.0,
    }
    assert rank_rois_by_expression(scores) == [
        "ROI_0",
        "ROI_1",
        "ROI_2",
        "ROI_3",
        "ROI_4",
        "ROI_5",
    ]
    assert select_rois_across_expression_range(scores, max_rois=3) == [
        "ROI_0",
        "ROI_2",
        "ROI_5",
    ]
    assert select_rois_across_expression_range(scores, max_rois=0) == list(scores)
    assert select_rois_across_expression_range(scores, max_rois=1) == ["ROI_2"]


def test_marker_expression_selection_can_choose_different_rois_per_marker():
    class FakeDataset:
        _values = {
            "CD3": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
            "CD20": {"A": 2.0, "B": 1.0, "C": 4.0, "D": 3.0},
        }

        def get_channel(self, roi, marker):
            return np.full((2, 2), self._values[marker][roi], dtype=float)

        def get_segmentation(self, roi):
            return np.ones((2, 2), dtype=np.uint8)

    selections, table = _build_marker_roi_selections(
        FakeDataset(),
        ["CD3", "CD20"],
        ["A", "B", "C", "D"],
        {"CD3": 0.0, "CD20": 0.0},
        requested_rois=None,
        strategy="marker_expression_range",
        max_rois=2,
        random_seed=0,
    )
    assert selections == {"CD3": ["A", "D"], "CD20": ["B", "C"]}
    assert len(table) == 8
    assert set(table.columns) >= {
        "above_background_fraction",
        "mean_above_background",
        "expression_quantile",
        "selected",
    }
    assert table.groupby("marker", observed=True)["selected"].sum().to_dict() == {
        "CD3": 2,
        "CD20": 2,
    }


def test_explicit_roi_selection_overrides_expression_prepass():
    class DatasetThatMustNotBeRead:
        def get_channel(self, roi, marker):  # pragma: no cover - failure sentinel
            raise AssertionError("Explicit selection must skip the image pre-pass")

        def get_segmentation(self, roi):  # pragma: no cover - failure sentinel
            raise AssertionError("Explicit selection must skip the mask pre-pass")

    selections, table = _build_marker_roi_selections(
        DatasetThatMustNotBeRead(),
        ["CD3"],
        ["A", "B", "C"],
        {"CD3": 0.0},
        requested_rois=["C", "A"],
        strategy="marker_expression_range",
        max_rois=1,
        random_seed=0,
    )
    assert selections == {"CD3": ["A", "C"]}
    assert table.loc[table["selected"], "roi"].tolist() == ["A", "C"]
    assert table["selection_strategy"].unique().tolist() == ["explicit"]


def test_vmax_grid_supports_relative_and_marker_specific_values():
    relative = build_vmax_grid("CD3", 10.0, factors=[0.5, 1.0, 2.0])
    assert relative == [5.0, 10.0, 20.0]
    explicit = build_vmax_grid(
        "CD3",
        10.0,
        marker_vmax_values={"cd3": [3, 7, 11, 20]},
    )
    assert explicit == [3.0, 7.0, 11.0, 20.0]
    with pytest.raises(ValueError, match="at least three"):
        build_vmax_grid("CD3", 10.0, factors=[0.5, 1.0])


def test_marker_baseline_values_are_resolved_then_multiplied_by_factors():
    baselines = resolve_marker_baseline_vmax(
        ["CD3", "FOXP3"], {"cd3": 10, "FOXP3": 5}
    )
    assert baselines == {"CD3": 10.0, "FOXP3": 5.0}
    assert build_vmax_grid(
        "FOXP3", baselines["FOXP3"], factors=[0.5, 1.0, 2.0]
    ) == [2.5, 5.0, 10.0]

    with pytest.raises(ValueError, match="does not uniquely match"):
        resolve_marker_baseline_vmax(["CD3"], {"CD20": 8})
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_marker_baseline_vmax(["CD3"], {"CD3": 0})
    lower = resolve_marker_lower_thresholds(["CD3"], {"cd3": 0.8})
    assert lower == {"CD3": 0.8}
    with pytest.raises(ValueError, match="below every"):
        build_vmax_grid(
            "CD3", 10, factors=[0.05, 0.1, 0.2], lower_threshold=0.5
        )


def test_scan_parameter_csv_defines_marker_scope_baselines_and_lower_bounds(
    tmp_path: Path,
):
    path = tmp_path / "nimbus_scan_parameters.csv"
    path.write_text(
        "marker,baseline_vmax,lower_threshold\n"
        "foxp3,5,0.8\n"
        "CD3,20,0\n",
        encoding="utf-8",
    )
    markers, baselines, lower_thresholds = resolve_scan_marker_inputs(
        ["CD3", "FOXP3", "CD20"],
        None,
        marker_parameters_path=path,
    )
    assert markers == ["FOXP3", "CD3"]
    assert baselines == {"FOXP3": 5.0, "CD3": 20.0}
    assert lower_thresholds == {"FOXP3": 0.8, "CD3": 0.0}
    assert build_vmax_grid(
        "FOXP3",
        baselines["FOXP3"],
        factors=[0.5, 1, 2],
        lower_threshold=lower_thresholds["FOXP3"],
    ) == [2.5, 5.0, 10.0]

    explicit_markers, file_baselines, file_lower = resolve_scan_marker_inputs(
        ["CD3", "FOXP3", "CD20"],
        ["CD20"],
        marker_parameters_path=path,
    )
    assert explicit_markers == ["CD20"]
    assert file_baselines == {}
    assert file_lower == {}


def test_scan_parameter_csv_rejects_unavailable_markers(tmp_path: Path):
    path = tmp_path / "nimbus_scan_parameters.csv"
    path.write_text(
        "marker,baseline_vmax,lower_threshold\nUNKNOWN,5,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not uniquely match"):
        resolve_scan_marker_inputs(
            ["CD3"],
            None,
            marker_parameters_path=path,
        )


def test_scan_analysis_finds_stable_range_nearest_baseline(tmp_path: Path):
    scores, pixels = _synthetic_scan()
    analysis = analyze_normalization_scan(
        scores,
        pixels,
        baseline_values={"CD3": 10.0},
        positive_thresholds=[0.25, 0.5, 0.75],
        primary_threshold=0.5,
    )
    recommendation = analysis.recommendations.iloc[0]
    assert recommendation["recommendation_status"] == "stable_plateau"
    assert recommendation["suggested_vmax"] == pytest.approx(10.0)
    assert recommendation["stable_vmax_min"] == pytest.approx(5.0)
    assert recommendation["stable_vmax_max"] == pytest.approx(20.0)
    assert not bool(recommendation["manual_review_required"])
    assert recommendation["lower_threshold"] == 1.0
    assert recommendation["below_lower_threshold_fraction_at_suggestion"] == 0.2
    assert set(analysis.threshold_summary["positive_score_threshold"]) == {
        0.25,
        0.5,
        0.75,
    }
    assert np.allclose(analysis.candidate_summary["primary_positive_fraction"], 0.5)

    figure = plot_marker_scan(
        marker="CD3",
        cell_scores=scores,
        analysis=analysis,
        output_path=tmp_path / "CD3.png",
    )
    assert figure.is_file() and figure.stat().st_size > 0
    suggested = write_suggested_normalization_dict(
        tmp_path / "suggested.csv",
        analysis.recommendations,
        {"CD3": 1.0},
    )
    assert suggested.read_text(encoding="utf-8").splitlines() == [
        "marker,vmax,lower_threshold",
        "CD3,10,1",
    ]
    legacy = write_suggested_normalization_dict(
        tmp_path / "suggested.json", analysis.recommendations
    )
    assert '"CD3": "10.0"' in legacy.read_text(encoding="utf-8")


def test_baseline_csv_can_supply_lower_thresholds_for_explicit_baselines(
    tmp_path: Path,
):
    path = tmp_path / "normalization_dict.csv"
    path.write_text(
        "marker,vmax,lower_threshold\nCD3,20,0.5\nFOXP3,5,0.8\n",
        encoding="utf-8",
    )
    vmax_values, lower_thresholds = _load_baseline_normalization(
        path,
        ["CD3", "MISSING"],
        require_all=False,
    )
    assert vmax_values == {"CD3": 20.0}
    assert lower_thresholds == {"CD3": 0.5}


def test_scan_analysis_flags_large_adjacent_call_shift():
    scores, pixels = _synthetic_scan(cliff=True)
    analysis = analyze_normalization_scan(
        scores,
        pixels,
        baseline_values={"CD3": 10.0},
        primary_threshold=0.5,
        cliff_tolerance=0.4,
    )
    recommendation = analysis.recommendations.iloc[0]
    assert bool(recommendation["cliff_detected"])
    assert bool(recommendation["manual_review_required"])
    assert recommendation["largest_adjacent_positive_fraction_jump"] == pytest.approx(
        0.5
    )
    assert "adjacent positive-fraction shift" in recommendation["review_reason"]


def test_scan_rejects_a_lower_threshold_that_changes_between_vmax_candidates():
    scores, pixels = _synthetic_scan()
    pixels.loc[pixels["vmax"].eq(5.0), "lower_threshold"] = 0.5
    with pytest.raises(ValueError, match="must remain fixed"):
        analyze_normalization_scan(
            scores,
            pixels,
            baseline_values={"CD3": 10.0},
        )


def test_config_validates_grids_thresholds_and_defaults():
    settings = NimbusNormalizationScanConfig()
    assert settings.max_rois == 10
    assert settings.roi_selection_strategy == "marker_expression_range"
    assert len(settings.vmax_factors) == 9
    assert settings.marker_baseline_vmax == {}
    assert settings.marker_lower_thresholds == {}
    assert settings.primary_positive_score_threshold == 0.5
    assert PipelineConfig().nimbus_normalization_scan == settings

    with pytest.raises(ValueError, match="at least three"):
        NimbusNormalizationScanConfig(vmax_factors=[0.5, 1.0])
    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        NimbusNormalizationScanConfig(positive_score_thresholds=[1.2])
    with pytest.raises(ValueError, match="unique ignoring case"):
        NimbusNormalizationScanConfig(
            marker_vmax_values={"CD3": [1, 2, 3], "cd3": [2, 3, 4]}
        )
    with pytest.raises(ValueError, match="finite and positive"):
        NimbusNormalizationScanConfig(marker_baseline_vmax={"CD3": 0})
    with pytest.raises(ValueError, match="finite and non-negative"):
        NimbusNormalizationScanConfig(marker_lower_thresholds={"CD3": -0.1})
    with pytest.raises(ValueError, match="cannot appear in both"):
        NimbusNormalizationScanConfig(
            marker_baseline_vmax={"CD3": 10},
            marker_vmax_values={"cd3": [5, 10, 20]},
        )
    marker_baselines = NimbusNormalizationScanConfig(
        marker_baseline_vmax={" CD3 ": 10, "FOXP3": 5.5},
        marker_lower_thresholds={" cd3 ": 0.5, "FOXP3": 0},
        vmax_factors=[0.5, 1, 2],
    )
    assert marker_baselines.marker_baseline_vmax == {"CD3": 10.0, "FOXP3": 5.5}
    assert marker_baselines.marker_lower_thresholds == {"cd3": 0.5, "FOXP3": 0.0}
    configured = NimbusNormalizationScanConfig(
        positive_score_thresholds=[0.25, 0.75],
        primary_positive_score_threshold=0.5,
    )
    assert configured.positive_score_thresholds == [0.25, 0.5, 0.75]


def test_stage_registry_wrapper_environment_docs_and_planner_align(tmp_path: Path):
    stage = STAGE_REGISTRY["nimbus-scan"]
    assert stage.catalogue_order == 41
    assert stage.depends_on == ["cellpose"]
    assert stage.groups == ["qc"]
    assert stage.environment_keys == ["segmentation"]
    assert stage.config_sections == [
        "general",
        "segmentation",
        "nimbus",
        "nimbus_normalization_scan",
    ]
    assert stage.requires_assets == ["masks", "metadata", "denoised_images"]
    assert stage.produces_assets == ["human_outputs"]
    assert all("nimbus-scan" not in mode.stages for mode in MODES)
    assert (REPO_ROOT / stage.documentation_path).is_file()

    wrapper = (REPO_ROOT / stage.slurm_script).read_text(encoding="utf-8")
    assert "#SBATCH -G 1" in wrapper
    assert "#SBATCH --cpus-per-task=6" in wrapper
    assert "#@ENV:  imc_segmentation" in wrapper
    assert "SpatialBiologyToolkit.scripts.nimbus_normalization_scan" in wrapper
    environments = load_environment_registry(REPO_ROOT)
    assert environments.stage_environments["nimbus-scan"] == ["segmentation"]

    context = initialize_project(tmp_path / "project")
    assets = {
        asset.role: asset for asset in resolve_assets(context.config, context.root)
    }
    assets["masks"].path.mkdir(parents=True)
    (assets["masks"].path / "ROI_1.tiff").write_bytes(b"placeholder")
    assets["metadata"].path.mkdir(parents=True, exist_ok=True)
    (assets["metadata"].path / "panel.csv").write_text(
        "channel_name,channel_label,use_denoised,use_raw\n3,CD3,true,false\n",
        encoding="utf-8",
    )
    (assets["metadata"].path / "metadata.csv").write_text(
        "unstacked_data_folder,import_data\nROI_1,true\n",
        encoding="utf-8",
    )
    (assets["denoised_images"].path / "ROI_1").mkdir(parents=True)
    (assets["denoised_images"].path / "ROI_1" / "3_CD3.tiff").write_bytes(
        b"placeholder"
    )
    plan = build_run_plan(
        context,
        ["nimbus-scan"],
        include_dependencies=False,
    )
    assert plan.ready, plan.errors
    assert [candidate.name for candidate in plan.resolved_stages] == ["nimbus-scan"]

    legacy = [
        line.split("=", 1)[0]
        for line in (REPO_ROOT / "SLURM_scripts" / "pipeline.conf")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert legacy[-1] == "nimbus-scan"
    raw_registry = yaml.safe_load(
        (REPO_ROOT / "HPC_env_files" / "environments.yaml").read_text(encoding="utf-8")
    )
    assert raw_registry["stage_environments"]["nimbus-scan"] == ["segmentation"]
