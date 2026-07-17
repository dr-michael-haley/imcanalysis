# Pipeline

The preferred interface is the lightweight, project-aware [`sbt` CLI](cli.md).
It initializes or adopts projects, validates configured assets, plans workflows,
submits the existing stage-specific SLURM wrappers, records runs, and inspects
status and logs. The older `pl` command remains available for compatibility.

Start with the [CLI and project guide](cli.md) and the [workflow and run
order](workflow.md), then use the generated [SLURM stage
reference](stages/index.md) for exact inputs, outputs, environments, and config
sections.

The [outputs and reporting guide](reporting.md) explains sequential human-facing
executions, reusable root assets, technical records, safe removal, and explicit
legacy-layout migration. The concise [execution-layout design](execution-layout-design.md)
defines the underlying identity and locking model.
Stable scientific context for every stage is in the
[shared stage explainers](../stages/index.md).

![Core and analysis pipeline flow](../_static/pipeline_stage_flow.svg)

```{toctree}
:maxdepth: 1

cli
workflow
reporting
environments
environment-migration
execution-layout-design
stages/index
../stages/index
subclustering
python_stages
bash_helpers
```
