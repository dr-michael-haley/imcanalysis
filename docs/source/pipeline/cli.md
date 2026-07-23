# The `sbt` project and SLURM CLI

`sbt` is the lightweight command-and-control interface for
SpatialBiologyToolkit. It validates project structure and typed configuration,
plans workflows, submits the existing per-stage SLURM wrappers, records runs,
checks scheduler status, and resolves logs.

Normal CLI startup imports only lightweight packages such as Pydantic, PyYAML,
and Typer. Scientific work still runs in the stage-specific Conda environments
selected by the existing SLURM wrappers.

## Install the lightweight launcher

```bash
conda env create -f Local_envs/sbt_cli_env.yml
conda activate sbt-cli
pip install --no-deps -e .
sbt --help
```

The editable install supplies `sbt` without installing the full scientific
dependency stack into the launcher environment.

Manage the fixed scientific environments through `sbt env`, not an independent
installer implementation:

```bash
sbt env doctor
sbt env list
sbt env sync --all --dry-run
```

See the [fixed Conda environment guide](environments.md) for synchronization,
capture, drift comparison, lock maintenance, smoke tests, and stage provenance.

## SBT projects

An SBT project combines:

- a Pydantic-validated pipeline config, normally `config.yaml`;
- canonical reusable project inputs and assets at paths configured under `general`;
- automatically numbered human-facing execution reports under `outputs/`;
- `.sbt/executions.yaml`, the locked active project execution index;
- `.sbt/project.yaml`, which gives the project a stable UUID;
- `.sbt/project_notes.md`, for durable human or agent-authored context;
- `.sbt/runs/`, which contains operational run records, stage events, and logs.

Configured paths remain the source of truth. The project marker does not copy
scientific defaults or redefine folder names. Project assets may remain mutable;
run directories record operations and do not copy large images, masks, tables,
or AnnData objects.

## Initialize or adopt

Create a new minimal project:

```bash
mkdir gbm_project
cd gbm_project
sbt project init
```

This creates `config.yaml`, the configured raw IMC and metadata folders,
`.sbt/project.yaml`, `.sbt/project_notes.md`, `.sbt/runs/`, an empty execution
index, and `outputs/README.md`. Execution folders are created only as stages
are accepted for submission. Use `--config-level complete` for every
current default. Existing scientific files are not overwritten.

Adopt an existing project without moving or rewriting data:

```bash
cd existing_project
sbt project adopt --config config.yaml
```

Adoption validates the config, resolves configured paths relative to the
project root, writes the marker, and records a lightweight initial asset
inventory. Present, absent, and unexpected top-level paths are reported but
left unchanged.

After initialization or adoption, commands discover the project by walking
upwards from nested subdirectories. Use `--project /path/to/project` to select
an explicit root.

## Validate and inspect

```bash
sbt project validate
sbt project validate --stage prep
sbt project validate --stage cellpose
sbt project validate --mode segmentation
sbt project describe
sbt project assets
sbt project assets --format yaml
sbt project assets --format json
```

Validation distinguishes required initial inputs, optional inputs, generated
assets, human-facing reports, legacy output folders, and readiness for a
requested stage or mode. Inspection uses only
existence, file size, modification time, and bounded top-level counts. It does
not load AnnData or images, recurse through large trees, or calculate checksums.

Project notes can be displayed or appended explicitly:

```bash
sbt project notes
sbt project notes --add "Check panel mapping before publication."
```

## Stages and modes

```bash
sbt stages list
sbt stages explain nimbus
sbt modes list
sbt modes explain segmentation
```

In an interactive terminal, `sbt stages list` color-codes the stage,
environment, output-slug, and display-name columns. Redirected output remains
plain text, while `--format yaml` and `--format json` are always ANSI-free.

The typed Python registry contains every alias mirrored in
`SLURM_scripts/pipeline.conf`, with documentation order, an unnumbered output slug,
shared scientific explainer, wrapper paths, dependencies, workflow groups,
asset roles, expected outputs, and log patterns. `sbt stages explain` renders
the same explainer snapshot used in generated reports.

Initial modes are:

- `segmentation`: `prep`, `denoise`, `dnqc`, `cellpose`, `nimbus`;
- `integration-rapids`: the RAPIDS route;
- `integration-harmony`: the Harmony/BBKNN route;
- `integration-biobatchnet`: the BioBatchNet route;
- `spatial`: independent CellCharter, STARLING, pairwise, and NetworkX branches;
- `visualisation`: the standard visualisation stage;
- `full`: the documented segmentation, RAPIDS, and visualisation example.

Integration routes remain separate because they are alternatives rather than
three mandatory sequential stages.

## Plan and run

```bash
sbt plan segmentation
sbt plan prep denoise cellpose nimbus
sbt plan segmentation --format yaml
sbt plan segmentation --format json
```

Planning validates the config, expands modes and dependencies, checks wrappers,
simulates assets produced by earlier planned stages, and reports missing inputs
before submission.

Preview exact `sbatch` arguments, dependency order, run paths, and exported SBT
context:

```bash
sbt run segmentation --dry-run
```

Dry runs create no persistent run record and submit no jobs. Displayed run paths
are prospective only.

Submit:

```bash
sbt run segmentation
sbt run prep denoise cellpose nimbus
sbt run cellpose --reason "Repeat with a larger diameter after fragmentation."
sbt run cellpose --note "Review ROI_17 carefully." --note "Compare with run 2026..."
```

To submit only the explicitly requested stage when its upstream assets already
exist, disable dependency expansion:

```bash
sbt run cellvision-cluster --no-deps --dry-run
sbt run cellvision-cluster --no-deps
```

`--no-deps` does not relax input validation. The command fails before submission
when the selected stage's required assets are absent. Use the dry run first to
confirm that the plan contains only the intended stage. When a mode or several
stages are explicitly selected, dependencies between those selected stages are
retained; only unselected upstream stages are omitted.

Each submitted run creates:

```text
.sbt/runs/<workflow_run_id>/
  run_manifest.yaml
  run_plan.yaml
  config.user.yaml
  config.resolved.yaml
  submitted_jobs.yaml
  command.txt
  status.yaml
  project_assets.before.yaml
  logs/
  stage_events/
```

Jobs use `sbatch --parsable`, explicit stdout/stderr paths, and `afterok`
dependencies. Each receives the project, config, execution ID, immutable
technical execution ID, workflow run ID, output directory, stage, reason, and
notes. Transitional `SBT_RUN_ID` and `SBT_STAGE_OUTPUT_DIR` aliases remain for
existing wrappers.

The shared scientific-stage config parser honors `SBT_CONFIG`, so jobs read the
run's resolved config while executing from the project root. The user's source
config is not modified by normal `sbt run` operation. Submission stops on the
first `sbatch` failure and records the partial submission.

Each stage also writes:

```text
outputs/<execution_id>_<stage_slug>/
  README.md
  stage_manifest.yaml
  figures/       # only when the stage creates figures
  tables/        # only when the stage creates tables
  summaries/     # only when the stage creates summaries
  files/         # only when the stage creates attachments
```

The stage report links back to the technical run and its logs. Start a project
handover or review at `outputs/README.md`. See the
[outputs and reporting guide](reporting.md) for the complete convention.

IDs are allocated automatically in project execution order. Stage types have
no permanent number, rerunning a stage creates another sequential folder, and
a multi-stage request receives consecutive IDs. Workflow, technical execution,
and scheduler job IDs remain separate.

## Summary, status, reports, and logs

```bash
sbt summary
sbt summary --stage cellpose
sbt summary --status failed
sbt summary --format json

sbt status latest
sbt status 003
sbt status --technical-run-id stage-...
sbt status latest --format yaml
sbt status latest --format json

sbt logs latest
sbt logs 003 --stderr
sbt logs 003 --stdout --tail 100
sbt logs 003 --path-only
sbt report 003
```

Status combines active `squeue` data with `sacct` history. Missing accounting,
unknown jobs, and incomplete records are reported as uncertainty rather than
guessed states. A pending `afterok` job is recorded as `blocked` when its
upstream job has failed or been cancelled, including SLURM's
`DependencyNeverSatisfied` state. The result is written to `status.yaml` and
the project execution index, so a subsequent `sbt summary` shows the refreshed
state.

Logs are resolved from recorded paths and tailed from the end of each file;
large directory trees are not scanned.

Remove a visible execution with `sbt remove 003`. The command confirms the
identity and output path, warns again for created, modified, or unknown reusable
asset effects, and inventories asset paths created since the workflow's pre-run
snapshot. It lists unused created assets that are eligible for cleanup, assets
retained because another active stage requires or produces the same role, and
other protected paths. Pre-existing assets, paths outside the project, uncertain
ownership, and every `.h5ad` file are retained. Interactive cleanup requires
typing the full word `yes`; any other response keeps the assets. For explicit
non-interactive cleanup, use `--yes --accept-asset-risk --remove-assets`.

The command then removes the human-facing folder and active index entry and
renumbers later visible executions. Technical evidence and an asset-cleanup
audit remain under `.sbt/`; removed executions are shown only by
`sbt summary --include-removed`.

Projects using the former fixed-stage-folder layout require an explicit,
non-silent upgrade:

```bash
sbt project migrate-execution-layout --dry-run
sbt project migrate-execution-layout
```

## Legacy interfaces and provenance boundary

`pl`, `pll`, and `pls` remain available while compatibility and downstream use
are assessed. Use `sbt` for ordinary project-aware planning, submission, status,
and logs.

The canonical external reporting layer now records project/run identity,
important config fields, files, assets, objective metrics, warnings, errors,
software/Git identifiers, stage events, and narrative Markdown indexes.
Existing AnnData `pipeline_stage_log` records remain as transitional
compatibility provenance and are not removed. Expensive checksums and immutable
output versioning remain outside the current scope.

## Initial SLURM-site assumptions

- `sbatch`, `squeue`, and usually `sacct` are available on the login node.
- The existing wrappers can source `$HOME/imcanalysis/SLURM_scripts/job_env.sh`.
- Repository-managed environments have been synchronized with `sbt env`; any
  external environments exist under the fixed names in the central registry.
- The site accepts standard `--parsable`, `--chdir`, `--output`, `--error`,
  `--export`, and `afterok` options.

`SBT_TOOLKIT_ROOT` can select a different wrapper checkout for planning and
submission, but wrappers that still source `$HOME/imcanalysis` retain that
legacy site assumption. The external `scport` wrapper also retains its current
fixed `processed/` and `masks/` arguments.
