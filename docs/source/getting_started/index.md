# Getting started

SpatialBiologyToolkit is **HPC-first**. The normal workflow is to run the
repeatable, compute-heavy pipeline on a Linux cluster with SLURM, then use the
resulting AnnData, images, and reports for interactive work on a workstation.

Choose a starting point:

1. [Complete beginner's guide](beginners.md) — learn the small set of command-line,
   Conda, Git, and SLURM concepts used by the toolkit.
2. [HPC setup](hpc.md) — install the `sbt` launcher and scientific environments,
   create or adopt a project, preview a run, and submit it to SLURM. This is the
   recommended route for the end-to-end pipeline.
3. [Local analysis setup](local.md) — prepare a Windows or macOS environment for
   notebooks, Napari, exploratory analysis, and bespoke figures after pipeline
   processing.

Experienced HPC users can go straight to [HPC setup](hpc.md). Details of the
optional shell installer have moved to the
[installation helper scripts reference](../reference/installation_helpers.md).

```{toctree}
:maxdepth: 1
:hidden:

beginners
hpc
local
```
