from __future__ import annotations

import pytest

from SpatialBiologyToolkit.napari_sbt.help import (
    extract_help_section,
    load_help_markdown,
)

BOX_HELP_SECTIONS = (
    ("setup", "Start or resume"),
    ("setup", "Workflow selection"),
    ("setup", "Dataset inputs"),
    ("setup", "Image normalization and default display"),
    ("setup", "Cell scope"),
    ("setup", "Full experiment or Feature Discovery Trial"),
    ("setup", "Classes"),
    ("feature_building", "Imported sources"),
    ("feature_building", "IMC channels"),
    ("feature_building", "Synthetic features"),
    ("feature_building", "Execution and progress"),
    ("feature_refinement", "Readiness"),
    ("feature_refinement", "Evaluation"),
    ("feature_refinement", "Analysis progress"),
    ("feature_refinement", "Choosing and promoting features"),
    ("explore", "ROI reload recipe"),
    ("explore", "AnnData and population overlays"),
    ("explore", "Image channels"),
    ("population_qc", "Population selection"),
    ("population_qc", "RGB verification recipe"),
    ("population_qc", "ROI sampling"),
    ("populations", "1. Source workspace and drafts"),
    ("populations", "Subclusters"),
    ("populations", "Applying a draft"),
    ("scanpy_plotting", "Choose data and cells"),
    ("scanpy_plotting", "Choose a plot"),
    ("scanpy_plotting", "Plot options"),
    ("scanpy_plotting", "Generate and manage plots"),
    ("dataset_maintenance", "Dataset and synchronization readiness"),
    ("dataset_maintenance", "Save current AnnData"),
    ("dataset_maintenance", "Rename variables and images"),
    ("dataset_maintenance", "Remove variables"),
    ("dataset_maintenance", "Filter cells"),
    ("dataset_maintenance", "Rebuild masks and ObjectNumbers"),
    ("dataset_maintenance", "Manage observations"),
    ("dataset_maintenance", "Observation column utilities"),
    ("classify", "Cell annotation"),
    ("classify", "Models and active-learning queues"),
    ("labeler", "Define labels"),
    ("labeler", "Label cells"),
    ("labeler", "ROI sampling guidance"),
    ("labeler", "Results and export"),
    ("regions_export", "Manual tissue regions"),
    ("regions_export", "Cohort results and exports"),
    ("layers_status", "Selected-layer utilities"),
)


@pytest.mark.parametrize(("topic", "section"), BOX_HELP_SECTIONS)
def test_every_workflow_box_has_external_focused_help(topic, section):
    markdown = load_help_markdown(topic, section)

    assert markdown.startswith("#")
    assert section in markdown.splitlines()[0]
    assert len(markdown.split()) >= 20


def test_extract_help_section_includes_children_but_not_siblings():
    markdown = """# Guide

## Parent

Intro.

### Child

Details.

## Sibling

Not included.
"""

    extracted = extract_help_section(markdown, "Parent")

    assert "### Child" in extracted
    assert "Details." in extracted
    assert "Sibling" not in extracted


def test_missing_help_section_lists_available_headings():
    with pytest.raises(KeyError, match="Available headings"):
        load_help_markdown("setup", "Not a real box")
