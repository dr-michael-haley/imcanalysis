# Complete beginner's guide

SpatialBiologyToolkit is designed around a simple division of work:

1. Store a dataset as a toolkit project on an HPC cluster.
2. Use the `sbt` command to check the project, plan a workflow, and submit jobs.
3. Let SLURM run each scientific stage in the correct Conda environment.
4. Read the reports and logs, then use the resulting data locally for interactive
   analysis when needed.

You do not need to become a Linux or Python expert before starting. The terms
below are the ones used in the setup and pipeline guides.

## Terminal and commands

A **terminal** (or shell) is a text interface to a computer. On an HPC cluster,
you normally connect to a login node and type commands there. Three useful Linux
commands are:

```bash
pwd                 # show the current directory
ls                  # list files and directories
cd path/to/project  # change directory
```

Commands are sensitive to spelling, spaces, and capital letters. A command in
the documentation can usually be copied exactly, but replace placeholders such
as `<project-name>` with your own value. Do not include the angle brackets.

The login node is for short management tasks. Scientific processing is sent to
the cluster's compute nodes through SLURM; do not run a large pipeline stage
directly on the login node.

## Repository and Git

The **repository** is the `imcanalysis` directory containing the toolkit's code,
environment definitions, and SLURM wrappers. Git downloads and updates it:

```bash
git clone https://github.com/dr-michael-haley/imcanalysis.git
git pull --ff-only
```

`clone` creates your first working copy. Later, `pull --ff-only` updates a clean
copy without silently merging local changes. Run `git status --short` first; if
it lists files you intentionally changed, preserve or commit them before asking
Git to update the checkout.

The repository and your **dataset projects** are different things. Keep the
toolkit checkout at `~/imcanalysis` on HPC. Keep each dataset in its own project
directory, usually on the cluster's project or scratch storage.

## Python, packages, and Conda environments

Python packages are reusable pieces of software. Different stages need
different, sometimes incompatible package versions, so the toolkit uses
separate **Conda environments**.

There are two kinds of environment on HPC:

- `sbt-cli` is small and contains the project-management command. It is safe to
  use on the login node.
- Scientific environments such as `imc_segmentation` and `imc_denoise` contain
  the heavier analysis software. SLURM wrappers activate them on compute nodes.

Activate the launcher environment with:

```bash
conda activate sbt-cli
```

The command prompt often shows `(sbt-cli)` while it is active. If `sbt` is not
found, first check that this environment is active.

An **editable installation** connects an environment to the code in your Git
checkout. Updating the checkout therefore updates the code used by the
environment. The supplied setup commands create these editable links for you.

## The `sbt` command

`sbt` is the supported front door to the pipeline. It can:

- create or adopt a dataset project;
- validate configuration and required files;
- list stages and reusable workflow modes;
- show a plan without submitting anything;
- submit jobs to SLURM;
- find status, reports, and logs for recorded executions.

Useful read-only commands include:

```bash
sbt --help
sbt stages list
sbt modes list
sbt project describe
sbt plan segmentation
sbt run segmentation --dry-run
```

The last command is a preview: it prints the planned SLURM submission but does
not create a run or submit a job. `sbt run segmentation` without `--dry-run`
does submit jobs.

Older commands named `pl`, `pll`, and `pls` still exist for compatibility, but
new users should use `sbt`.

## Projects and `config.yaml`

A toolkit **project** is one dataset plus its configuration, scientific assets,
and run history. A typical project contains:

```text
my_dataset/
  config.yaml
  IMC_files/
  metadata/
  .sbt/
  outputs/
```

`config.yaml` records choices and paths. Most paths are relative to the project
directory. `sbt project init` creates a new project with a compact config;
`sbt project adopt --config config.yaml` records an existing project without
moving its data.

The `.sbt/` directory contains toolkit run records. The `outputs/` directory
contains numbered, human-readable reports. Large reusable assets such as TIFFs,
masks, and AnnData remain at their configured project paths.

## HPC, SLURM, and jobs

An HPC cluster is a shared collection of computers. **SLURM** is the scheduler
that decides where and when requested work runs. A submitted unit of work is a
**job**.

This changes the rhythm of analysis:

1. Validate and preview the work on the login node.
2. Submit it to SLURM.
3. Disconnect if you wish; the job continues on a compute node.
4. Return later and inspect status, logs, and reports.

For example:

```bash
sbt run segmentation --dry-run
sbt run segmentation
sbt status latest
sbt logs latest
sbt summary
```

## Local notebooks and Napari

Jupyter notebooks and Napari are useful after scripted processing for exploring
data, reviewing images, testing ideas, and making bespoke figures. They are not
the primary way to run the full pipeline.

A Jupyter notebook runs code in a **kernel** backed by a selected environment.
Run notebook cells from top to bottom, save regularly, and use **Restart and Run
All** to check that a result does not depend on hidden state. Copy tutorial
notebooks into your own analysis directory before editing them so a later Git
update cannot conflict with your work.

## Where to go next

- Follow [HPC setup](hpc.md) for the recommended installation and first dry run.
- Use [Local analysis setup](local.md) only when you need notebooks, Napari, or
  workstation-based analysis.
- Read the [pipeline workflow](../pipeline/workflow.md) before selecting stages.
- Use the [scientific guides](../stages/index.md) to understand what each stage
  does and how to interpret its outputs.
