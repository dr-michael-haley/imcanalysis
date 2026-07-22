# 🧪 SpatialBiologyToolkit – Local Setup (Workstation / Laptop)

This guide is for running **local analyses** on your own machine (typically via Jupyter notebooks), and for running/iterating on parts of the pipeline without an HPC.

If you’re completely new to Python/conda/Jupyter, start with the beginner explainers first:
- [new-user guide](beginners.md)

If you want to run the scripted pipeline on an HPC cluster, use:
- [HPC setup](hpc.md)

---

## ✅ What you will end up with

- A conda environment containing the dependencies for local analysis
- The `SpatialBiologyToolkit` package installed in editable mode
- Jupyter Lab running in a folder containing the tutorials / your analysis files

---

## 1) Install Anaconda / Miniconda

Install Anaconda (or Miniconda) for your OS:
- https://www.anaconda.com/products/distribution

---

## 2) Get the repository

Open a terminal (Anaconda Prompt on Windows) and run:

```bash
git clone <repo-url>
cd imcanalysis
```

For this repository, `<repo-url>` is
`https://github.com/dr-michael-haley/imcanalysis.git`.

If you don’t use git, you can download the repo as a ZIP from GitHub, extract it, then open a terminal in that folder.

---

## 3) Create and activate the local environment

Create the environment from the YAML. On Windows:

```bash
conda env create -f Local_envs/sbt_env.yml
```

On macOS, use the portable specification instead:

```bash
conda env create -f Local_envs/sbt_env_macos.yml
```

The macOS environment intentionally omits CUDA and Windows runtime packages,
along with the less portable R, Java/Bio-Formats, OpenCL, optional SpatialData
I/O/viewer integrations, and most pip-only analysis extras from the Windows
environment export. It includes a tested Apple Silicon combination of Napari,
PySide6, TensorFlow for macOS, and the TensorFlow Metal plug-in. This combination
requires macOS 12 or later and is not intended for Intel Macs.

Activate the environment:

```bash
# Windows
conda activate sbt

# macOS
conda activate sbt
```

---

## 4) Install the toolkit (editable)

From the repo root (i.e. the `imcanalysis` folder), install the toolkit.

On Windows:

```bash
pip install -e .
```

On macOS, use the `nodl` install option so that the package installer does not
replace the tested Apple TensorFlow packages with generic TensorFlow:

```bash
python -m pip install -e ".[nodl]"
```

Confirm that Qt, TensorFlow, and UMAP import without crashing:

```bash
python -c "from qtpy import API_NAME; import tensorflow as tf, umap; print(API_NAME, tf.__version__)"
```

---

## 5) Install and run Jupyter

Install Jupyter (if it’s not already in the environment):

```bash
conda install jupyter
```

Start Jupyter Lab in the folder where you want to work:

(You may need to use `cd` first to change into the folder where you want to save your analyses, or into a folder where you have copied the `Tutorials`.)

For example:

```bash
cd path/to/my_analysis_folder
jupyter lab
```

A good first notebook is in:
- the [tutorial index](../tutorials/index.md)

⚠️ **Important:** avoid doing your own work directly inside the repo’s `Tutorials/` folder.
If you edit those notebooks in-place and later run `git pull`, your changes may be overwritten or cause git conflicts.

Instead, copy the tutorial notebook(s) into a separate “analysis” folder **outside** the repo (e.g. `C:/imc_analysis_projects/<project_name>/`) and work on the copies.

---

## 6) Keeping your local copy up to date

From inside the repo:

```bash
git pull
```

If the Python package code changed, reinstall editable (safe to repeat):

```bash
pip install -e .
```

If the environment dependencies changed, you may need to recreate the environment.

---

## Troubleshooting

### macOS: `QtBindingsNotFoundError` or a crash while importing UMAP

For an environment created from an older version of `sbt_env_macos.yml`, repair
the Qt binding and replace generic TensorFlow with the tested Apple packages:

```bash
conda activate sbt
python -m pip uninstall -y tensorflow tensorflow-estimator
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.26,<2" "PySide6==6.9.1" "tensorflow-macos==2.16.2" "tensorflow-metal==1.2.0"
```

Restart the Jupyter kernel after changing these packages.

### Jupyter can’t import SpatialBiologyToolkit

- Confirm you activated the env: `conda activate sbt`
- Reinstall editable from the repo root: `pip install -e .`

### “command not found: conda”

You’re not in an Anaconda/Miniconda shell. On Windows, use **Anaconda Prompt**.
