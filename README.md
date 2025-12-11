# imcanalysis

Toolkit for analysing Imaging Mass Cytometry (IMC) and other spatial-omics data. It combines a Python package (`SpatialBiologyToolkit`), CLI pipeline stages, SLURM job templates, tutorials, and HPC helper scripts.

## Start here
- New to the repo or IMC analysis? Read [`New_users.md`](New_users.md).
- Running notebooks locally? See [`Tutorials/README.md`](Tutorials/README.md) and the `NEW - Population identification.ipynb` notebook.
- Running on HPC? Follow [`README_IMC_HPC.md`](README_IMC_HPC.md) plus the SLURM helper docs below.

## Quick setup (local workstation)
1. Create the conda env: `conda env create -f Conda_environments/conda_environment.yml`.
2. Activate: `conda activate spatbiotools`.
3. Install the package editable: `pip install -e .`.
4. Copy `Tutorials/` to your analysis folder and open `jupyter lab` there.

## Components
- [`SpatialBiologyToolkit/`](SpatialBiologyToolkit/README.md): core Python package (preprocessing, denoising, clustering, spatial stats, plotting).
- [`SpatialBiologyToolkit/scripts/`](SpatialBiologyToolkit/scripts/README.md): stage-based scripts that run locally or via SLURM; consume `config.yaml` in your dataset folder.
- [`SLURM_scripts/`](SLURM_scripts/README.md): sbatch templates + stage aliases (`pipeline.conf`); used by the `pl`/`pll` helpers.
- [`Bash_scripts/`](Bash_scripts/README.md): helper CLI tools (`pl`, `pll`, `pls`, `zipqc`, `cds`, etc.). Installed by `make install` on HPC.
- [`Tutorials/`](Tutorials/README.md): Jupyter notebooks and a short guide for exploratory analysis.
- [`Conda_environments/`](Conda_environments/README.md): environment specs (lightweight current + pinned/legacy variants).
- [`install/`](install/README.md): installer/uninstaller used by `make install` for HPC logins.
- [`docs/`](docs/README.md): Sphinx sources; generated HTML lives in `Documentation/`.
- [`External_and_old_code/`](External_and_old_code/README.md): archived or vendor code; use cautiously.

## Typical workflows
- **Local notebook workflow:** set up the env, install editable, copy `Tutorials/`, and work through the population identification notebook.
- **HPC pipeline workflow:** `make install` to add the helper scripts, prepare a dataset folder with `config.yaml`, then submit stages with `pl preprocess denoise aiinter vis` (see `SLURM_scripts/README.md`).

## Issue tracking
Please report bugs or questions in GitHub issues with details on the stage/notebook, environment file used, and any overrides in `config.yaml`.
