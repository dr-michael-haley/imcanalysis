"""Shared colours for workflow boxes, Qt help, and the Sphinx user guide.

This module deliberately imports no GUI or scientific packages at module load.
Section names refer to headings in the adjacent canonical Markdown help files.
"""

from __future__ import annotations

import re

# border, title border, title background, readable title text
ACCENTS = {
    "blue": ("#60a5fa", "#3b82f6", "#dbeafe", "#1e3a8a"),
    "violet": ("#a78bfa", "#8b5cf6", "#ede9fe", "#4c1d95"),
    "teal": ("#2dd4bf", "#14b8a6", "#ccfbf1", "#134e4a"),
    "amber": ("#fbbf24", "#f59e0b", "#fef3c7", "#78350f"),
    "rose": ("#fb7185", "#f43f5e", "#ffe4e6", "#881337"),
    "cyan": ("#22d3ee", "#06b6d4", "#cffafe", "#164e63"),
}

# Explicit assignments keep existing box colours stable when help is reordered.
SECTION_ACCENTS = {
    "setup": {
        "Start or resume": "blue",
        "Workflow selection": "violet",
        "Dataset inputs": "teal",
        "Image normalization and default display": "amber",
        "Cell scope": "rose",
        "Full experiment or Feature Discovery Trial": "cyan",
        "Classes": "blue",
        "Checking data": "teal",
        "Saving a workspace": "amber",
        "Workspace summary": "blue",
    },
    "explore": {
        "ROI sample metadata": "blue",
        "ROI reload recipe": "violet",
        "AnnData and population overlays": "teal",
        "Image channels": "amber",
    },
    "population_qc": {
        "Population selection": "blue",
        "ROI sample metadata": "violet",
        "RGB verification recipe": "teal",
        "ROI sampling": "amber",
    },
    "populations": {
        "1. Source workspace and drafts": "blue",
        "Subclusters": "violet",
        "Applying a draft": "teal",
    },
    "feature_building": {
        "Feature readiness": "blue",
        "Imported sources": "violet",
        "IMC channels": "teal",
        "Synthetic features": "amber",
        "Execution and progress": "rose",
    },
    "feature_refinement": {
        "Readiness": "blue",
        "Evaluation": "violet",
        "Analysis progress": "teal",
        "Choosing and promoting features": "amber",
    },
    "classify": {
        "Cell annotation": "blue",
        "Models and active-learning queues": "violet",
        "Prediction display controls": "teal",
        "Final identities and export": "amber",
        "Integrate with existing labels": "rose",
        "Export final identities": "cyan",
    },
    "labeler": {
        "Define labels": "blue",
        "Label cells": "violet",
        "ROI sampling guidance": "teal",
        "Results and export": "amber",
    },
    "scanpy_plotting": {
        "Choose data and cells": "blue",
        "Choose a plot": "violet",
        "Plot options": "teal",
        "Generate and manage plots": "amber",
    },
    "dataset_maintenance": {
        "Dataset and synchronization readiness": "blue",
        "Save current AnnData": "violet",
        "Rename variables and images": "teal",
        "Remove variables": "amber",
        "Filter cells": "rose",
        "Rebuild masks and ObjectNumbers": "cyan",
        "Manage observations": "blue",
        "Observation column utilities": "violet",
    },
    "regions_export": {
        "Manual tissue regions": "blue",
        "Cohort results and exports": "violet",
    },
    "layers_status": {"Selected-layer utilities": "blue"},
}


def section_accent(topic: str, section: str, default: str = "blue") -> str:
    return SECTION_ACCENTS.get(topic, {}).get(section, default)


def workflow_accent_stylesheet() -> str:
    """Return Qt box styling from the same palette used by documentation."""

    return "\n".join(
        f'QGroupBox[sbtAccent="{name}"] {{ border-color: {border}; }} '
        f'QGroupBox[sbtAccent="{name}"]::title {{ '
        f'border-color: {title_border}; background-color: {background}; '
        f'color: {foreground}; }}'
        for name, (border, title_border, background, foreground) in ACCENTS.items()
    )


def populate_help_browser(browser, markdown: str, topic: str) -> None:
    """Render readable Markdown with matching, theme-independent heading colours."""

    from qtpy.QtGui import QColor, QTextCursor

    browser.setStyleSheet(
        "QTextBrowser { background: #f8fafc; color: #172033; "
        "border: 1px solid #94a3b8; border-radius: 6px; padding: 12px; }"
    )
    font = browser.font()
    font.setPointSizeF(max(13.0, font.pointSizeF()))
    browser.setFont(font)
    document = browser.document()
    document.setDefaultFont(font)
    document.setDefaultStyleSheet(
        "body { color: #172033; line-height: 1.5; } "
        "h1 { font-size: 22pt; margin-top: 8px; margin-bottom: 14px; } "
        "h2 { font-size: 18pt; margin-top: 18px; margin-bottom: 10px; } "
        "h3 { font-size: 15pt; margin-top: 14px; margin-bottom: 8px; } "
        "p, li { margin-bottom: 8px; } "
        "a { color: #1d4ed8; } code, pre { color: #334155; font-size: 12pt; } "
        "th, td { padding: 7px; }"
    )
    browser.setMarkdown(markdown)
    headings = {
        match.group(2): len(match.group(1))
        for match in re.finditer(r"^(#{1,3})\s+(.+?)\s*$", markdown, re.MULTILINE)
    }
    block = document.begin()
    while block.isValid():
        heading = block.text().strip()
        if heading in headings and block.blockFormat().headingLevel() > 0:
            accent = section_accent(topic, heading)
            _border, _title_border, background, foreground = ACCENTS[accent]
            cursor = QTextCursor(block)
            block_format = block.blockFormat()
            block_format.setBackground(QColor(background))
            block_format.setTopMargin(18)
            block_format.setBottomMargin(10)
            block_format.setLeftMargin(8)
            block_format.setRightMargin(8)
            cursor.setBlockFormat(block_format)
            cursor.select(QTextCursor.BlockUnderCursor)
            char_format = cursor.charFormat()
            char_format.setForeground(QColor(foreground))
            char_format.setFontPointSize({1: 22, 2: 18, 3: 15}[headings[heading]])
            cursor.mergeCharFormat(char_format)
        block = block.next()
