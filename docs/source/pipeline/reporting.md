# Project outputs, reports, and provenance

New runs separate reusable computational assets, human-facing analysis outputs,
and technical execution records:

```text
project/
  config.yaml
  tiff_stacks/ tiffs/ processed/ masks/ cell_tables/ anndata.h5ad
  outputs/
    README.md
    001_Preprocessing/
      README.md
      <run_id>/
        README.md
        stage_manifest.yaml
        figures/ tables/ summaries/ files/
  .sbt/runs/<run_id>/
    run_manifest.yaml
    config.resolved.yaml
    submitted_jobs.yaml
    status.yaml
    logs/
    stage_events/
```

Start a project review at `outputs/README.md`. Reusable assets remain in their
configured project-root paths so later stages have one canonical source. The
numbered output folders contain material intended for human interpretation.
`.sbt/runs/` contains commands, exact configuration, scheduler state, technical
events, and SLURM logs.

## Numbered stage folders

| Order | Alias | Folder |
|---:|---|---|
| 1 | `prep` | `001_Preprocessing` |
| 2 | `denoise` | `002_Denoising` |
| 3 | `dnqc` | `003_Denoising_QC` |
| 4 | `cellpose` | `004_Segmentation` |
| 5 | `nimbus` | `005_Quantification` |
| 6 | `bint` | `006_Batch_Integration` |
| 7 | `rapids` | `007_RAPIDS_Processing` |
| 8 | `bbn` | `008_BioBatchNet_Integration` |
| 9 | `subcl` | `009_Subclustering` |
| 10 | `cchar` | `010_CellCharter_Neighbourhoods` |
| 11 | `starling` | `011_STARLING_Phenotyping` |
| 12 | `aiinter` | `012_AI_Interpretation` |
| 13 | `vis` | `013_Visualisation` |
| 14 | `pairsp` | `014_Pairwise_Spatial_Analysis` |
| 15 | `nxsp` | `015_NetworkX_Spatial_Analysis` |
| 16 | `reint` | `016_Marker_Reintegration` |
| 17 | `remap` | `017_Observation_Remapping` |
| 18 | `rebuildmeta` | `018_Metadata_Rebuild` |
| 19 | `scport` | `019_scPortrait_Export` |
| 20 | `config` | `020_Configuration_Maintenance` |
| 21 | `zipqc` | `021_Output_Archive` |
| 22 | `slogs` | `022_Legacy_SLURM_Log_Migration` |
| 23 | `debug` | `023_Environment_Diagnostics` |

The order describes the logical analysis catalogue, not a claim that every
project runs every branch. Reruns reuse the numbered folder and create a new
`<run_id>/` child.

## Stage reports

Every reported execution writes a versioned `stage_manifest.yaml` before
rendering Markdown. It records project and run identity, timing, status, job ID,
pipeline version, Git commit, inputs, reusable assets, generated files,
important parameters, metrics, warnings, and errors. The run README snapshots
the shared [stage explainer](../stages/index.md), so old reports remain
understandable after documentation changes.

Managed executions use the same run ID in `outputs/` and `.sbt/runs/`. A copy of
the stage manifest is also written to `.sbt/runs/<run_id>/stage_events/`.
Technical logs are not copied into `outputs/`; reports link to the recorded
`.sbt/runs/<run_id>/logs/<stage>_<job_id>.out` and `.err` paths.

Use `sbt run ... --reason "..."` and repeatable `--note "..."` options to add
human context without making it mandatory.

## Output routing and the legacy `QC/` folder

`general.outputs_folder` defaults to `outputs`. `general.qc_folder` remains in
the Pydantic schema for compatibility and defaults to `QC`, but it is deprecated
as a general destination.

During an active stage report, the shared legacy config adapter explicitly
routes existing `qc_folder` writes to that stage's run folder. It does not
rewrite the source or resolved YAML. Direct invocations that cannot establish a
reporting context retain the configured legacy path. Existing `QC/` data is
never deleted or moved automatically, and `sbt project validate` reports it as
legacy.

The migration audit used these classifications:

| Previous behaviour | Classification | New destination | Compatibility |
|---|---|---|---|
| TIFF stacks, channel images, masks, cell tables, AnnData | Reusable asset | Configured project-root path | Unchanged |
| `QC/...` figures and result tables | Human-facing output | Numbered stage run folder | Runtime route; old files retained |
| Subclustering templates/remaps | Reusable curation asset | Configured root `subclustering/` path | Unchanged |
| CellCharter TRVAE model | Reusable model asset | Configured root `cellcharter.trvae_save_path` | Relative paths now resolve from the project root |
| STARLING model checkpoint | Reusable model asset | Configured root `starling.model_output_name` | Relative paths now resolve from the project root |
| Subclustering figures | Human-facing output | `009_Subclustering/<run_id>/figures/` | Legacy fallback outside reports |
| CellPose morphology tables | Human-facing table | `004_Segmentation/<run_id>/tables/` | Legacy fallback outside reports |
| Panel consistency CSV | Human-facing table | `003_Denoising_QC/<run_id>/tables/` | Explicit `--save_csv` still honoured |
| Managed SLURM stdout/stderr | Technical log | `.sbt/runs/<run_id>/logs/` | Legacy direct scheduler paths remain |
| AnnData `pipeline_stage_log` | Transitional embedded provenance | Retained in AnnData | New canonical report is external |

## Direct execution

Registered `python -m SpatialBiologyToolkit.scripts...` stages infer their alias
when they load the shared config. They create a `direct-...` report and warn that
full SBT/SLURM provenance is unavailable. Managed runs provide the complete
environment and technical link. Reporting exceptions are surfaced centrally;
scientific exceptions are recorded and re-raised or preserve their non-zero
process exit status.

## Using `StageReporter`

New stage code can add richer objective summaries without depending on analysis
libraries:

```python
from SpatialBiologyToolkit.reporting import StageReporter

with StageReporter.from_environment(stage="cellpose") as report:
    report.add_input("denoised_images", "processed")
    report.add_metric("rois_processed", 12)
    report.add_warning("One ROI was skipped because its DNA channel was missing.")
```

Existing stages are integrated through the shared config/bootstrap and job-exit
hooks. Future stage-specific additions should register metrics already available
from the computation and must not invent biological conclusions.
