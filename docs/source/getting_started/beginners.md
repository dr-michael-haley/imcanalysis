# Complete beginner's guide

This page explains the ideas behind SpatialBiologyToolkit before asking you to
install or run anything. You do not need to become a Linux, Python, or HPC
expert. The practical [CSF3 setup](hpc.md) includes copy-and-paste commands and
explains what you should see after each step.

SpatialBiologyToolkit is **HPC-first**. Its usual workflow is:

1. connect to an HPC cluster;
2. use `sbt` to check a dataset project and preview a workflow;
3. let SLURM run the scientific stages on compute nodes;
4. inspect the reports and logs;
5. use notebooks or Napari locally when interactive analysis is helpful.

## The command line

The **command line**, **terminal**, and **shell** all refer to the text-based
interface where you type commands. Instead of opening folders by clicking, you
move around and run programs by typing instructions.

On Windows, open PowerShell, Windows Terminal, or Anaconda Prompt. On macOS,
open Terminal. After connecting to an HPC cluster, the commands are normally
Linux commands regardless of the computer on your desk.

These four commands are enough to start:

```bash
pwd                 # print the directory you are currently in
ls                  # list its files and directories
cd path/to/project  # change directory
mkdir new_directory # make a directory
```

The `#` text in these examples is an explanation; the shell ignores everything
after it. Commands are sensitive to spelling, spaces, and capital letters.
Replace placeholders such as `<username>` with your own value, without typing
the angle brackets.

Two useful pieces of path notation are:

- `~` or `$HOME`: your personal home directory on the current computer;
- `.`: the directory you are currently in.

For example, `cd "$HOME/imcanalysis"` enters the toolkit repository stored in
your HPC home directory.

## What is an HPC cluster?

**HPC** means **high-performance computing**. An HPC cluster is a shared group
of computers with many processors, large amounts of memory, and often GPUs.
It can process large images and cell datasets without tying the work to your
laptop.

The [University of Manchester Computational Shared Facility
(CSF3)](https://ri.itservices.manchester.ac.uk/csf3/) is one example. Other
institutions use different cluster names and login addresses, but the concepts
are similar.

An HPC cluster normally has two kinds of machine:

- a **login node**, where you connect, manage files, inspect results, and submit
  work;
- **compute nodes**, where the large scientific calculations run.

The login node is shared by many people. Do not run a heavy pipeline stage
directly there. SpatialBiologyToolkit submits that work to a compute node for
you.

### Why use HPC?

- **Scale:** large IMC datasets can need more memory or GPU capacity than a
  normal laptop.
- **Long-running jobs:** work can continue after you disconnect or close your
  laptop.
- **Repeatability:** standard scripts and environments make it easier to run
  the same analysis consistently.
- **Shared resources:** specialist hardware can be scheduled fairly between
  users.

HPC storage is not all permanent. Many clusters have fast **scratch** storage
that is automatically cleaned. Read your cluster's storage policy and keep a
separate backed-up copy of irreplaceable raw data. CSF3 users should start with
the current [CSF3 getting-started
guide](https://ri.itservices.manchester.ac.uk/csf3/getting-started/).

## Connecting with SSH

**SSH** is the secure connection used to open a terminal on an HPC login node.
The general form is:

```bash
ssh <username>@<cluster-address>
```

For CSF3, the current command is:

```bash
ssh <username>@csf3.itservices.manchester.ac.uk
```

You need a cluster account, and CSF3 requires the University VPN when connecting
from outside its network. The first connection may ask whether you trust the
host key; check it against your institution's instructions before accepting.
When you type a password, the terminal usually displays no dots or characters.
That is normal.

## What is SLURM?

**SLURM** is the job scheduler used by CSF3 and many other HPC clusters. It:

- accepts a request to run a calculation, called a **job**;
- decides when and on which compute node it can run;
- allocates the requested CPUs, memory, GPU, and time;
- tracks status and writes output and error logs.

A job may wait in a queue before it starts. Closing your SSH connection does
not cancel a submitted job. SpatialBiologyToolkit prepares and submits the
SLURM commands, so beginners do not need to write job scripts for the normal
pipeline.

The usual rhythm is:

```bash
sbt run segmentation --dry-run  # preview only
sbt run segmentation            # check environments, then submit
sbt status latest               # ask what SLURM reports
sbt logs latest                 # inspect recent log output
```

## Python, Anaconda, Miniconda, and Conda

**Python** is the programming language used by most of the toolkit.
**Packages** are reusable pieces of Python software, rather like specialist
apps or plug-ins.

**Conda** manages both packages and isolated software environments. It is
distributed through:

- **Anaconda**, a large distribution that includes many data-science packages;
- **Miniconda**, a smaller installation containing Conda and its essentials.

The HPC guide uses Miniconda because it is small and the toolkit installs only
what it needs. If your cluster already provides Conda, you may not need to
install either distribution yourself.

### What is a Conda environment?

A **Conda environment** is an isolated collection of Python and packages.
Different analyses can require incompatible versions, so environments prevent
one tool's packages from breaking another.

SpatialBiologyToolkit deliberately separates:

- `sbt-cli`: a small environment containing the lightweight `sbt` command;
- scientific environments such as `sbt-analysis` and `sbt-denoise`, used
  by the corresponding SLURM jobs.

You do **not** install every scientific environment at the beginning. When you
submit a run, `sbt` checks the environments required by that run. If their
repository specifications are valid, it offers to install any missing managed
environment and leaves unrelated ones alone. An invalid specification or a
missing external environment stops safely for maintainer or specialist setup.

Activate the launcher with:

```bash
conda activate sbt-cli
```

The prompt may then begin with `(sbt-cli)`. Activation changes which Python and
commands the shell uses; it does not start a pipeline job.

### `conda` and `pip`

Both tools install packages:

- `conda install ...` installs packages and non-Python libraries from Conda
  channels;
- `python -m pip install ...` installs Python packages from Python package
  sources.

Scientific environments can use both, but mixing arbitrary install commands
often creates hard-to-reproduce environments. Use the repository's setup and
`sbt env` commands unless a guide explicitly tells you otherwise.

### Editable installations

An **editable installation** links the installed Python package to a Git
checkout instead of copying its source elsewhere. The setup helper performs
the equivalent of:

```bash
python -m pip install --no-deps -e .
```

After a clean `git pull`, environments using that link see the updated toolkit
code. `--no-deps` prevents pip from replacing the carefully managed scientific
dependencies.

## Repository, Git, and GitHub

The toolkit lives in a **Git repository**: a directory containing code plus its
change history. GitHub hosts the shared copy.

- **Clone** downloads the repository for the first time.
- **Pull** updates an existing checkout.
- **Status** shows local changes that Git is not safe to ignore.

```bash
git clone https://github.com/dr-michael-haley/imcanalysis.git
git status --short
git pull --ff-only
```

The pipeline does not silently update the repository before a run. Update it
deliberately, and do not pull over local edits you need to keep.

The repository is software; a **dataset project** is your data, configuration,
outputs, and run history. Keep them in separate directories. On HPC, the
repository currently belongs at `~/imcanalysis` because some compatibility
wrappers still expect that location.

## Projects and `config.yaml`

A toolkit project represents one dataset. A typical project contains:

```text
my_dataset/
  config.yaml
  IMC_files/
  metadata/
  .sbt/
  outputs/
```

`config.yaml` is a text file recording analysis choices and paths. Most paths
are interpreted relative to the project directory. `sbt project init` creates
a new project with a compact config that omits values inherited unchanged from
SBT defaults; `sbt project adopt --config config.yaml` registers an existing one
without moving scientific data.

The hidden `.sbt/` directory contains technical run records. `outputs/`
contains numbered, human-readable reports. Images, masks, AnnData, and other
large reusable assets remain at the configured project paths.

## What the `sbt` command does

`sbt` is the supported front door to the pipeline. It can:

- create or adopt a project;
- validate configuration and required files;
- explain stages and reusable workflow modes;
- preview a plan without changing cluster state;
- check and, with permission, install required environments;
- submit jobs and resolve their status, reports, and logs.

Useful safe inspection commands include:

```bash
sbt --help
sbt stages list
sbt modes list
sbt project describe
sbt plan segmentation
sbt run segmentation --dry-run
```

A dry run creates no run record, installs no environment, and submits no job.
The same command without `--dry-run` can install a missing managed environment
after validating its installation specification and prompting, then submit
work. Older commands named `pl`, `pll`, and `pls` remain only for compatibility;
new users should use `sbt`.

## Headless pipeline work and interactive analysis

HPC pipeline stages are normally **headless**: there is no graphical window or
live plot. A script runs on a compute node and writes assets, logs, and reports.
This is well suited to repeatable processing.

Jupyter notebooks and Napari are interactive tools, normally used after or
alongside the standard pipeline for exploration, manual review, bespoke
analysis, and figure making.

### What is a Jupyter notebook?

A notebook combines executable Python **cells**, their displayed results, and
Markdown notes. A **kernel** is the Python process that executes those cells.
The kernel belongs to an environment, so selecting the wrong kernel can make an
installed package appear to be missing.

Good notebook habits are:

- copy a tutorial into your own analysis directory before editing it;
- run cells from top to bottom;
- use small cells and Markdown notes to record what and why;
- save regularly;
- use **Restart Kernel and Run All** to check that hidden state is not affecting
  the result.

Restarting clears variables and imports from memory. Deleting a cell does not
clear a variable it already created. If a notebook behaves impossibly, restart
the kernel and run it from the beginning.

Common shortcuts include **Shift+Enter** to run a cell, **Tab** for completion,
and **Shift+Tab** inside a function call for help. Notebook interfaces differ,
so menus and some shortcuts may vary.

## How HPC and local work fit together

Most users will run the standard, compute-heavy workflow on HPC, then copy the
resulting AnnData and selected images into a separate local analysis directory
for notebooks or Napari. Useful, repeatable notebook operations can later be
turned into scripted pipeline stages.

When these concepts make sense, continue with [CSF3 setup](hpc.md). Use
[local analysis setup](local.md) for workstation notebooks and interactive
tools, not as a replacement for the SLURM pipeline.
