# Project outputs, executions, and provenance

Managed runs separate reusable computational assets, human-facing execution
reports, and permanent technical evidence:

```text
project/
  config.yaml
  tiff_stacks/ tiffs/ processed/ masks/ cell_tables/ anndata.h5ad
  outputs/
    README.md
    001_Preprocessing/
      README.md
      stage_manifest.yaml
      figures/ tables/ summaries/ files/  # only categories actually used
    002_Segmentation/
      README.md
      stage_manifest.yaml
      figures/ tables/ summaries/ files/  # only categories actually used
  .sbt/
    executions.yaml
    runs/<workflow_run_id>/
      run_manifest.yaml
      config.resolved.yaml
      submitted_jobs.yaml
      status.yaml
      logs/
      stage_events/
    audit/removals/
    audit/migrations/
```

Start a review at `outputs/README.md` or run `sbt summary`. Reusable assets stay
at their configured project-root paths so later stages have one canonical
source. The numbered folders contain material intended for human interpretation.
`.sbt/runs/` contains commands, exact configuration, scheduler state, events,
and SLURM logs.

Report category folders are created lazily. A run that produces no figures,
tables, summaries, or attachments contains only its report/provenance files;
unused empty category folders are not added.

## Numbers mean execution order

The three-digit number is allocated automatically in the order stages are
submitted in that project. It is not a permanent number assigned to a stage
type. If environment diagnostics is the first stage attempted, its folder is
`001_Environment_Diagnostics`. Running segmentation twice might produce
`002_Segmentation` and `004_Segmentation`, depending on what ran between them.

A multi-stage request receives consecutive IDs in resolved submission order.
The report is written directly inside `NNN_<stage-slug>/`; there is no extra
long-ID child directory.

Three identities deliberately remain separate:

- **Execution ID**, such as `003`, is the short, mutable project-workflow
  position shown to users and stored in `.sbt/executions.yaml`.
- **Technical execution ID**, such as `stage-...`, permanently identifies one
  stage attempt. It remains stable if visible executions are renumbered.
- **Workflow run ID** permanently identifies one `sbt run` request under
  `.sbt/runs/`. One workflow may contain several technical executions.
- **SLURM job ID** is assigned by the scheduler and is not used as either SBT
  identity.

The stage catalogue still has a documentation order, but that order never
controls project output numbers.

## Stage reports and failure states

Every accepted execution gets a typed `stage_manifest.yaml` and a Markdown
README. The manifest records all identities, timing, status, job ID, software
and Git identifiers, inputs, reusable assets, generated files, important
parameters, metrics, warnings, and errors. The README snapshots the corresponding
entry from the [Scientific Guides](../stages/index.md).

If no SLURM job is accepted, submission-failure evidence remains under `.sbt`
without creating a completed-looking output folder. Accepted jobs retain their
execution folder if they later fail or are cancelled. A downstream stage that
cannot be submitted is recorded as blocked. Folder existence therefore does
not imply success.

Technical logs remain under
`.sbt/runs/<workflow_run_id>/logs/<stage>_<slurm_job_id>.out` and `.err`.
Stage-event copies use the immutable technical execution ID. Reports link to
these records rather than copying logs.

Use `--reason` and repeatable `--note` options with `sbt run` to add human
context.

## Summary and removal

```bash
sbt summary
sbt summary --stage segmentation
sbt summary --status failed
sbt summary --latest
sbt summary --details
sbt summary --format json
```

The summary and `outputs/README.md` are generated from `.sbt/executions.yaml`,
not inferred from README prose.

`sbt remove 003` removes an execution from the visible workflow, removes its
human-facing folder, and compacts later execution numbers. Temporary names and
a project lock prevent rename collisions; manifests, technical references, and
generated links are rewritten while permanent technical IDs remain unchanged.
External bookmarks to mutable numbered paths can break after compaction.

Removal does **not** pretend to reverse scientific changes. Reusable assets are
not deleted or restored. Executions that created or modified assets, or whose
effect is unknown, require a second warning; non-interactive use requires
`--yes --accept-asset-risk`. A hidden audit under `.sbt/audit/removals/`
preserves the old ID and path, technical identity, asset classification,
reason, and renumbering map. It appears only with
`sbt summary --include-removed`.

## Existing fixed-number projects

Old projects may contain a fixed stage catalogue folder with one or more long
run-ID children. Ordinary startup never silently changes it. Preview the
explicit migration first:

```bash
sbt project migrate-execution-layout --dry-run
sbt project migrate-execution-layout
```

The plan derives chronology from structured manifest timestamps, flattens each
old child into its own sequential execution folder, preserves technical IDs,
SLURM IDs, timestamps, configs, and reusable assets, and writes a migration
audit. Ambiguous or unrecognised content aborts without moving data.

## Output routing and legacy `QC/`

`general.outputs_folder` defaults to `outputs`. `general.qc_folder` remains for
compatibility and defaults to `QC`, but is deprecated as a general destination.
During managed reporting, the shared adapter routes legacy `qc_folder` writes
to the current execution folder without rewriting YAML. Existing `QC/` data is
never moved or deleted automatically.

| Data | Location | Behaviour |
|---|---|---|
| TIFF stacks, images, masks, cell tables, AnnData | Configured project-root path | Reusable canonical asset; unchanged |
| Figures, result tables, and summaries | `outputs/<execution_id>_<stage-slug>/` | Human-facing execution output |
| Managed stdout/stderr | `.sbt/runs/<workflow_run_id>/logs/` | Permanent technical log |
| AnnData `pipeline_stage_log` | Inside AnnData | Transitional embedded provenance retained |
| Existing `QC/` files | `QC/` | Preserved legacy data |

## Direct execution

Registered `python -m SpatialBiologyToolkit.scripts...` stages create reports
under `outputs/direct/direct-..._<stage-slug>/`. They do not allocate a managed
project execution ID and warn that complete SBT/SLURM provenance is unavailable.
Scientific exceptions are recorded and retain their non-zero process exit.

## Using `StageReporter`

```python
from SpatialBiologyToolkit.reporting import StageReporter

with StageReporter.from_environment(stage="cellpose") as report:
    report.add_input("denoised_images", "processed")
    report.add_metric("rois_processed", 12)
    report.add_warning("One ROI was skipped because its DNA channel was missing.")
```

Future stage additions should register objective metrics already available from
the computation and must not invent biological conclusions.
