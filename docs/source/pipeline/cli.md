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

## SBT projects

An SBT project combines:

- a Pydantic-validated pipeline config, normally `config.yaml`;
- canonical project inputs and outputs at the paths configured under `general`;
- `.sbt/project.yaml`, which gives the project a stable UUID;
- `.sbt/runs/`, which contains operational run records.

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
`.sbt/project.yaml`, and `.sbt/runs/`. Use `--config-level complete` for every
current default. Existing files are not overwritten unless `--force` is used.

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
assets, and readiness for a requested stage or mode. Inspection uses only
existence, file size, modification time, and bounded top-level counts. It does
not load AnnData or images, recurse through large trees, or calculate checksums.

## Stages and modes

```bash
sbt stages list
sbt stages explain nimbus
sbt modes list
sbt modes explain segmentation
```

The typed Python registry contains every alias mirrored in
`SLURM_scripts/pipeline.conf`, with wrapper paths, dependencies, workflow
groups, asset roles, expected outputs, and log patterns.

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
```

Each submitted run creates:

```text
.sbt/runs/<run_id>/
  run_manifest.yaml
  run_plan.yaml
  config.user.yaml
  config.resolved.yaml
  submitted_jobs.yaml
  command.txt
  status.yaml
  project_assets.before.yaml
  logs/
```

Jobs use `sbatch --parsable`, explicit stdout/stderr paths, and `afterok`
dependencies. Each receives `SBT_PROJECT_ROOT`, `SBT_PROJECT_ID`, `SBT_CONFIG`,
`SBT_RUN_ID`, `SBT_RUN_DIR`, and `SBT_STAGE`.

The shared scientific-stage config parser honors `SBT_CONFIG`, so jobs read the
run's resolved config while executing from the project root. The user's source
config is not modified by normal `sbt run` operation. Submission stops on the
first `sbatch` failure and records the partial submission.

## Status and logs

```bash
sbt status latest
sbt status <run-id>
sbt status latest --format yaml
sbt status latest --format json

sbt logs latest
sbt logs latest --stage cellpose
sbt logs latest --stage cellpose --stderr
sbt logs latest --stage cellpose --stdout --tail 100
sbt logs latest --stage cellpose --path-only
```

Status combines active `squeue` data with `sacct` history. Missing accounting,
unknown jobs, and incomplete records are reported as uncertainty rather than
guessed states. The result is written to `status.yaml`.

Logs are resolved from recorded paths and tailed from the end of each file;
large directory trees are not scanned.

## Legacy interfaces and provenance boundary

`pl`, `pll`, and `pls` remain available while compatibility and downstream use
are assessed. Use `sbt` for ordinary project-aware planning, submission, status,
and logs.

This version records project identity, configs, plans, commands, available
software/Git identifiers, job submissions, status, logs, and lightweight
pre-run asset facts. It does not yet provide full scientific provenance,
post-run inventories, checksums, immutable output versioning, stage event
streams, QC summaries, or narrative reports. Existing AnnData logging remains
unchanged.

## Initial SLURM-site assumptions

- `sbatch`, `squeue`, and usually `sacct` are available on the login node.
- The existing wrappers can source `$HOME/imcanalysis/SLURM_scripts/job_env.sh`.
- The configured stage environments and `IMC_ENV_*` overrides already exist.
- The site accepts standard `--parsable`, `--chdir`, `--output`, `--error`,
  `--export`, and `afterok` options.

`SBT_TOOLKIT_ROOT` can select a different wrapper checkout for planning and
submission, but wrappers that still source `$HOME/imcanalysis` retain that
legacy site assumption. The external `scport` wrapper also retains its current
fixed `processed/` and `masks/` arguments.
