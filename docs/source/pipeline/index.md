# SpatialBiologyToolkit pipeline

The pipeline runs from a dataset directory containing `config.yaml`. The `pl`
command resolves short aliases, submits their SLURM wrappers in order, and
chains successful jobs with dependencies.

Start with the [workflow and run order](workflow.md), then use the generated
[SLURM stage reference](stages/index.md) for exact inputs, outputs,
environments, and config sections.

![Core and analysis pipeline flow](../_static/pipeline_stage_flow.svg)

```{toctree}
:maxdepth: 1

workflow
stages/index
subclustering
python_stages
bash_helpers
```
