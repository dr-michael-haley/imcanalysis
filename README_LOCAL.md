# 🧪 IMC Analysis – Local Setup (Workstation / Laptop)

This guide is for running **local analyses** on your own machine (typically via Jupyter notebooks), and for running/iterating on parts of the pipeline without an HPC.

If you’re completely new to Python/conda/Jupyter, start with the beginner explainers first:
- [README_NEW_USERS.md](README_NEW_USERS.md)

If you want to run the scripted pipeline on an HPC cluster, use:
- [README_IMC_HPC.md](README_IMC_HPC.md)

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

If you don’t use git, you can download the repo as a ZIP from GitHub, extract it, then open a terminal in that folder.

---

## 3) Create and activate the local environment

Create the environment from the YAML:

```bash
conda env create -f Local_envs/sbt_env.yml
```

Activate it:

```bash
conda activate sbt
```

---

## 4) Install the toolkit (editable)

From the repo root (i.e. the `imcanalysis` folder):

```bash
pip install -e .
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

```bash
jupyter lab
```

A good first notebook is in:
- [Tutorials](Tutorials/)

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

### Jupyter can’t import SpatialBiologyToolkit

- Confirm you activated the env: `conda activate spatbiotools`
- Reinstall editable from the repo root: `pip install -e .`

### “command not found: conda”

You’re not in an Anaconda/Miniconda shell. On Windows, use **Anaconda Prompt**.
