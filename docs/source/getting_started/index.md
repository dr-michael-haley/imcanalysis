# Getting started

SpatialBiologyToolkit is **HPC-first**. The normal workflow is to run the
repeatable, compute-heavy pipeline on a Linux cluster with SLURM, then use the
resulting AnnData, images, and reports for interactive work on a workstation.

Choose a starting point:

1. [Complete beginner's guide](beginners.md) — learn the small set of command-line,
   Conda, Git, and SLURM concepts used by the toolkit.
2. [CSF3 setup](hpc.md) — an absolute-beginner walkthrough for University of
   Manchester users, from installing Conda to submitting and monitoring the
   first job.
3. [Local analysis setup](local.md) — prepare a Windows or macOS environment for
   notebooks, Napari, exploratory analysis, and bespoke figures after pipeline
   processing.

Experienced users on another HPC can go straight to the
[`sbt` CLI guide](../pipeline/cli.md) and
[environment-management guide](../pipeline/environments.md). Details of the
optional shell installer have moved to the
[installation helper scripts reference](../reference/installation_helpers.md).

```{toctree}
:maxdepth: 1
:hidden:

beginners
hpc
local
```
