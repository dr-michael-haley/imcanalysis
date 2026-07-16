# SLURM pipeline documentation moved

- [Read the pipeline workflow](https://imcanalysis.readthedocs.io/en/latest/pipeline/workflow.html)
- [Use the project-aware `sbt` CLI](https://imcanalysis.readthedocs.io/en/latest/pipeline/cli.html)
- [Browse the generated stage reference](https://imcanalysis.readthedocs.io/en/latest/pipeline/stages/index.html)
- [Edit the workflow source](../docs/source/pipeline/workflow.md)

Stage facts are generated from the typed Python registry and the `#@` metadata
in each job wrapper. `pipeline.conf` remains an exact legacy compatibility
mirror. After changing stage mappings or metadata, run `make docs-generate`
from the repository root.
