# 🚀 Getting Started

Choose the route that matches where you will run the analysis:

- [New users](beginners.md) explains the command line, environments, Git, and Jupyter from first principles.
- [HPC and SLURM setup](hpc.md) is the recommended route for the reproducible end-to-end pipeline.
- [Local setup](local.md) is intended for notebooks, interactive exploration, and bespoke figures.
- [Installation helpers](install_helpers.md) documents what the repository setup scripts change.

The usual workflow is to run compute-heavy pipeline stages on HPC, then copy
the resulting AnnData and selected images to a local analysis directory for
interactive work.

```{toctree}
:maxdepth: 1
:hidden:

beginners
hpc
local
install_helpers
```
