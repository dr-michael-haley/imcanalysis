"""Sphinx configuration for SpatialBiologyToolkit."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

project = "SpatialBiologyToolkit"
copyright = "2024, Michael Haley"
author = "Michael Haley"
release = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns: list[str] = []

myst_heading_anchors = 3
autodoc_member_order = "bysource"
# Read the Docs installs only the lightweight documentation requirements. Mock
# scientific/GUI runtimes while autodoc imports modules; source, signatures,
# and docstrings remain available without installing several large envs.
autodoc_mock_imports = [
    "IMC_Denoise",
    "IPython",
    "colorcet",
    "joblib",
    "lifelines",
    "magicgui",
    "napari",
    "networkx",
    "qtpy",
    "readimc",
    "sc3s",
    "scanpy",
    "seaborn",
    "shapely",
    "skimage",
    "sklearn",
    "squidpy",
    "statsmodels",
    "tensorflow",
    "tifffile",
    "tkinter",
    "torch",
    "tqdm",
    "umap",
    "vispy",
]

html_theme = "sphinx_rtd_theme"
html_title = "SpatialBiologyToolkit"
html_short_title = "SpatialBiologyToolkit"
html_logo = "_static/Logo_white.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": True,
    "logo_only": True,
    "navigation_depth": 3,
    "titles_only": True,
}
html_context = {
    "display_github": True,
    "github_user": "dr-michael-haley",
    "github_repo": "imcanalysis",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
