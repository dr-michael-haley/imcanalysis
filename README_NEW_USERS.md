# 🧪 IMC Analysis – Absolute Beginner Guide (Explainers)

This page explains the key ideas you’ll see throughout the repo (command line, conda, environments, notebooks, config files).

When you’re ready to actually install and run things, use one of these practical guides:

- Local setup (Jupyter, exploratory analysis): [README_LOCAL.md](README_LOCAL.md)
- HPC setup (scripted pipeline via SLURM): [README_IMC_HPC.md](README_IMC_HPC.md)

---

## 💻 The command line (Terminal)

The **command line** (also called the *terminal* or *shell*) is a text-based interface where you type commands for your computer to execute. Instead of clicking buttons in a graphical interface, you write commands like `cd` (change directory) to navigate and run programs.

It can look intimidating at first, but for most steps you’ll just copy/paste a few commands.

Even if you mainly do analysis in notebooks, **running either locally or on HPC will unavoidably involve the command line** at least a little (e.g. creating/activating conda environments, launching Jupyter, and running/submitting pipeline commands).

Common commands you’ll see:

- `cd some/folder` = change directory
- `ls` = list files (Linux/macOS)
- `pwd` = print current folder

On Windows you’ll often use **Anaconda Prompt**.

---

## 🐍 Python distributions (Anaconda / Miniconda)

To run Python and manage analysis tools, we recommend **Anaconda** (or Miniconda if you prefer a smaller install).

### *💡 What is Anaconda?*

Anaconda is a free and open-source platform that makes it easy to:
- Run Python code
- Manage software libraries and environments
- Use scientific and data analysis tools

Anaconda includes:
- Python
- Conda (a package and environment manager)
- Many pre-installed packages commonly used for data science

They include **conda**, which we use for environment management.

---

## 📦 Conda environments

A **conda environment** is an isolated set of Python + packages.

Why environments matter:
- Different projects need different package versions.
- Environments stop projects from breaking each other.

Typical workflow:

```bash
conda env create -f some_environment.yml
conda activate some_env
```

---

## 📦 Packages (and `pip` vs `conda`)

Packages are reusable libraries (like “apps” for Python).

- `conda install ...` installs packages from conda channels
- `pip install ...` installs packages from PyPI

Many environments use a mix of both.

---

## 🧩 Editable installs (`pip install -e .`)

This repo contains a Python package named `SpatialBiologyToolkit`.

When you run this from the repo root:

```bash
pip install -e .
```

You install the package in **editable** mode, meaning:
- you can update the repo (`git pull`) and your environment will immediately use the updated code
- you can run scripts like `python -m SpatialBiologyToolkit.scripts.<stage>`

---

## 🐙 GitHub repositories (what “clone” means)

This project lives in a GitHub repository (a folder of code + history).

- “Clone” means download a working copy to your machine.
- “Pull” means update your working copy when the repo changes.

You don’t need to be a git expert, but these two commands are common:

```bash
git clone <repo-url>
git pull
```

### Updating code: HPC vs local

- **On HPC (e.g. CSF3):** the pipeline launcher will try to **auto-update to the latest GitHub version** before submitting jobs (a safe `git pull --ff-only`). This helps ensure bugfixes land in your next run without you needing to remember to update.
- **Locally:** if a bug is reported and fixed in the repo, you’ll usually update your local copy by going into the repo folder and running `git pull`.

(If you ever need to disable the auto-update behaviour, you can set `PIPELINE_AUTO_UPDATE=0`.)

---

## 📓 Jupyter notebooks

Jupyter notebooks are interactive documents for analysis and exploration. They are usually the best place for:

- bespoke / novel analyses
- trying new ideas
- making figures

Notebooks in this repo live in:
- [Tutorials](Tutorials/)

### *🖥 What are Jupyter Notebooks?*

Jupyter Notebooks are interactive documents that let you:
- Run Python code step by step
- Display results instantly (tables, plots)
- Add notes and explanations (like a computational lab book)

They’re particularly useful when you move beyond the scripted pipeline into more bespoke analysis.

## 📓 Tips for using Jupyter Notebooks

Jupyter Notebooks are a great way to run Python code step by step in your browser. If you're new to them, here are some best practices and common pitfalls.

### ✅ Best Practices

- **Run cells in order (unless you know what you are doing):**
	Notebooks run code in “cells”. Always run cells from top to bottom in order. Skipping around can be useful for testing, but it can also cause confusing errors if earlier variables or imports haven’t been run.

- **Save your work often:**
	Jupyter auto-saves, but it’s still good to use `File > Save and Checkpoint` periodically.

- **Use Markdown cells for notes:**
	Markdown cells are great for keeping track of what you did and why (like an electronic lab book).

- **Use small cells:**
	Break code into manageable pieces. It’s easier to debug and easier to understand later.

- **Restart your kernel occasionally:**
	`Kernel > Restart & Run All` is a great “sanity check” that your notebook works end-to-end.

### 🧠 What is the “kernel”?

In Jupyter Notebooks, the **kernel** is the computational engine that runs your Python code.

Here’s what you should know about kernels:

- **Each notebook has its own kernel:** if you open multiple notebooks, each one runs separately (but they can use the same conda environment).
- **Restarting the kernel clears memory:** this removes all variables, functions, and imports you’ve defined. It’s like starting fresh.
- **Kernel state matters:** the order you run cells affects what the kernel knows. If you run a cell that uses a variable before it’s defined, you’ll get an error.

If things look “impossible”, restarting and running top-to-bottom often fixes it.

### ⚠️ Common pitfalls

- **Running cells out of order:** can cause `NameError` or missing variables.
- **Forgetting to activate the right environment:** if you installed packages in one environment but launch Jupyter from another, imports may fail.
- **Variables persist between cells:** deleting a cell doesn’t delete variables already in memory.
- **Not knowing where you are in the filesystem:** notebooks run relative to the folder you started Jupyter in.

### 🎓 Extra tips and shortcuts

- You can run a cell by pressing **Shift + Enter**.
- Press **Esc** then **M** to convert a code cell to Markdown.
- Use `Tab` for auto-complete and suggestions.
- Use `Shift + Tab` inside parentheses to see function hints.
- Use the **"Help"** menu to explore Python and Jupyter tutorials.
- Selecting a cell and pressing **A** will create a new cell above, and **B** will create one below.
- Selecting a cell and pressing **D twice** will delete the cell.
- You can select multiple cells by shift clicking.
- You can rearrange the order of cells by drag and dropping them.

---

## 🏢 What is an HPC cluster?

An **HPC cluster** (High Performance Computing cluster) is a shared pool of powerful computers (many CPU cores, lots of RAM, often GPUs) that many users access remotely.

In this project, the **University of Manchester CSF3** is an example of an HPC cluster you might use.

Instead of running your analysis on your laptop, you *submit a job* to the cluster, and the cluster runs it for you when resources are available.

### Why is HPC useful?

- **Speed & scale:** you can process large IMC datasets and run heavy steps (segmentation, denoising, etc.) faster than on a typical workstation.
- **Reliability for long jobs:** jobs can run for hours/days without being tied to your personal machine.
- **Shared environments:** software and compute resources are managed in a more standardized way.

### “Headless” scripts vs interactive notebooks

- On HPC, the pipeline runs as **headless scripts**: no graphical interface, no clicking buttons, and usually no live plots.
	- You submit jobs (often via a scheduler like SLURM).
	- Progress is recorded in log files and output folders.
	- This is ideal for repeatable, standard pipeline steps.

- Locally, you’ll usually work **interactively** in **Jupyter notebooks**:
	- you can run code step-by-step
	- inspect intermediate results
	- create figures and do bespoke analyses

---

## 🗓️ What is SLURM?

**SLURM** (Simple Linux Utility for Resource Management) is a **job scheduler** used on many HPC clusters (including CSF3).

Its job is to:
- accept requests to run computations ("jobs")
- decide *when* and *where* those jobs run (based on available CPUs/RAM/GPUs and queue priorities)
- track job status and record output

When you use SLURM, you typically:
- write a job script that describes the resources you need (time, memory, CPUs/GPUs)
- submit it to the queue
- read the log files it produces (stdout/stderr) to see progress and errors

In this repo, most of that job management is **handled automatically by the pipeline tooling**: you run a pipeline command, and it submits the appropriate SLURM jobs and writes logs/outputs in the right places.

This is why HPC pipelines often look “non-interactive”: you submit work, then inspect outputs and logs rather than watching a GUI.

---

## 🖥️ HPC vs local (how they fit together)

Most users will:

1. Run the scripted pipeline on HPC to generate standard outputs (e.g. AnnData + QC)
2. Switch to local notebooks for bespoke downstream analysis

Over time, useful notebook workflows get migrated into scripts.

If you want to start now:
- Local: [README_LOCAL.md](README_LOCAL.md)
- HPC: [README_IMC_HPC.md](README_IMC_HPC.md)