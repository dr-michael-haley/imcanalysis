"""Resolve the human-facing output tree for the visualisation stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from SpatialBiologyToolkit.reporting import optional_category_output_path


@dataclass(frozen=True)
class VisualizationReportPaths:
    root: Path
    umaps: Path
    matrixplots: Path
    color_legends: Path
    population_images: Path


def prepare_visualization_report_paths(
    fallback_root: str | Path,
) -> VisualizationReportPaths:
    """Create visualisation folders in the active report or legacy fallback."""
    root = optional_category_output_path("figures", fallback_root)
    paths = VisualizationReportPaths(
        root=root,
        umaps=root / "UMAPs",
        matrixplots=root / "Matrixplots",
        color_legends=root / "Color_legends",
        population_images=root / "Population_images",
    )
    for directory in (
        paths.umaps,
        paths.matrixplots,
        paths.color_legends,
        paths.population_images,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = ["VisualizationReportPaths", "prepare_visualization_report_paths"]
