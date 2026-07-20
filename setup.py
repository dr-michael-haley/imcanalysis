from setuptools import setup, find_packages

setup(
    name="SpatialBiologyToolkit",
    version="0.1",
    packages=find_packages(),

    install_requires=[
        # ---- Core ----
        "numpy",
        "pandas",
        "scipy",
        "pyyaml",
        "pydantic>=2.4,<3",
        "typer>=0.12,<1",
        "tqdm",

        "anndata",
        "dask[array]",
        "scanpy",
        "bbknn",
        "harmonypy",
        "umap-learn",
        "scikit-learn",
        "statsmodels",
        "lifelines",
        "scikit-survival",
        "colorcet",
        "sc3s",
        #"ehrapy",
        "squidpy",
        "spatialdata",
        "spatialdata-plot",
        "readimc",

        "matplotlib",
        "seaborn",
        "tifffile",
        "shapely",
        "networkx",
        "scikit-image",

        # ---- Deep learning (skip if nodl) ----
        "torch; extra != 'nodl'",
        "tensorflow; extra != 'nodl'",
        "cellpose; extra != 'nodl'",
        "opencv-python; extra != 'nodl'",
        "psutil; extra != 'nodl'",
        "alpineer; extra != 'nodl'",

        # ---- Interactive / GUI (skip if headless) ----
        "napari; extra != 'headless'",
        "magicgui; extra != 'headless'",
        "qtpy; extra != 'headless'",
        "vispy; extra != 'headless'",
        "ipython; extra != 'headless'",
        "ipykernel; extra != 'headless'",

        # ---- Docs/dev (skip if nodev) ----
        "sphinx; extra != 'nodev'",
        "setuptools; extra != 'nodev'",
    ],

    extras_require={
        "headless": [],
        "nodl": [],
        "nodev": [],
        "cli": [
            "pyyaml",
            "pydantic>=2.4,<3",
            "typer>=0.12,<1",
        ],
    },

    entry_points={
        "console_scripts": [
            "sbt=SpatialBiologyToolkit.cli.main:app",
        ],
    },

    description="A tool kit for analysing high dimensional spatial data",
    author="Michael Haley",
    author_email="mrmichaelhaley@gmail.com",
    url="https://github.com/dr-michael-haley/imcanalysis",
)
