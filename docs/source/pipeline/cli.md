# The `sbt` project and SLURM CLI

`sbt` is the lightweight command-and-control interface for
SpatialBiologyToolkit. It validates project structure and typed configuration,
plans workflows, submits the existing per-stage SLURM wrappers, records runs,
checks scheduler status, and resolves logs.

Normal CLI startup imports only lightweight packages such as Pydantic, PyYAML,
and Typer. Scientific work still runs in the stage-specific Conda environments
selected by the existing SLURM wrappers.

## Install the lightweight launcher

On HPC, keep the repository at `~/imcanalysis` and use the maintained bootstrap:

```bash
cd "$HOME/imcanalysis"
bash install/bootstrap_sbt.sh
conda activate sbt-cli
sbt --help
```

The helper creates `sbt-cli` from `Local_envs/sbt_cli_env.yml` when necessary
and refreshes the editable `--no-deps` installation. The equivalent manual
commands are:

```bash
conda env create -f Local_envs/sbt_cli_env.yml
conda activate sbt-cli
python -m pip install --no-deps -e .
```

The editable installation supplies `sbt` without adding the scientific
dependency stack to the launcher environment. `conda-lock` is a separate base
environment prerequisite for `sbt env`; follow the complete
[CSF3 setup](../getting_started/hpc.md) for a fresh CSF3 installation.

## Optional project console

The lightweight graphical Project Console is a multi-project cockpit for
project-internal inspection, configuration editing, stage explanations, the
asset register, asset-aware readiness, run summaries, reports, bounded log
tails, and notes. It intentionally has no job submission, scheduler-control,
destructive, or scientific-data capability.

Install its separate Qt environment explicitly and launch it for an existing
project:

```bash
bash install/bootstrap_sbt_gui.sh
sbt gui project --project /path/to/project
```

See the complete [Project Console guide](../guides/project_console.md), including
the central project registry, CSF3 `srun-x11`, config backup/audit behavior, and
read-only mode.

## Optional NapariSBT application

NapariSBT uses its own scientific GUI environment. The lightweight launcher
uses the current interpreter when it already contains Napari and Qt; otherwise
it re-executes through the centrally registered `sbt-napari` environment:

```bash
bash install/bootstrap_napari_sbt_csf3.sh
cd /path/to/project
sbt gui napari --check
sbt gui napari
```

`--project` also accepts a registered project name, project ID, or explicit
path when launching from elsewhere.

The preflight is side-effect free and supports `--check-format json`. On CSF3,
run these commands only after entering an X11-enabled interactive allocation.
See the [CSF3 NapariSBT guide](../guides/napari_sbt_csf3.md).

## Compact a legacy configuration

Migrate a verbose legacy YAML file to the current compact style:

```bash
sbt config compact config.yaml
```

This validates the source against the current typed model and writes
`config.compact.yaml` beside it. Only canonical settings whose complete field
values differ from current defaults are retained. Deprecated aliases are
converted to their current names. Unrecognized legacy keys are preserved and
reported so migration does not silently discard them.

Choose another output path explicitly, or replace an existing output:

```bash
sbt config compact config.yaml --output config.new.yaml
sbt config compact config.yaml --output config.new.yaml --force
```

The source file is never overwritten, even with `--force`. YAML comments and
formatting are not retained in the generated file.

Manage the fixed scientific environments through `sbt env`, not an independent
installer implementation:

```bash
sbt env list
sbt env validate-spec segmentation
sbt env sync segmentation --dry-run
```

See the [fixed Conda environment guide](environments.md) for synchronization,
capture, drift comparison, lock maintenance, smoke tests, and stage provenance.
Installing all environments is not a prerequisite for the launcher or a run.

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
upwards from nested subdirectories. Every command with `--project` accepts an
initialized path, a registered project name, or a project ID. An initialized
path takes precedence; otherwise SBT resolves the value through the central
project registry. This makes registered projects usable from any working
directory, including directory-independent integrations such as SBT Gateway.

Register projects for the graphical cockpit in the SBT-managed block inside
`~/.imc_config`:

```bash
sbt project register --project /path/to/project --name "My cohort" --default
sbt project list
sbt project set-default "My cohort"
sbt project unregister "My cohort"
```

These commands preserve existing credentials and unrelated shell settings.
Unregistering never changes project files. If a project is moved, register it
again from its new root; the stable project ID replaces the old location rather
than creating a duplicate entry.

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

Project validation distinguishes structural requirements from optional source
material, generated assets, human-facing reports, legacy output folders, and
readiness for a requested stage or mode. A missing raw-IMC folder does not make
an adopted downstream-only project structurally invalid; it makes `prep` not
ready. Inspection uses only
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

Planning validates the config, expands modes, checks wrappers, simulates assets
produced by earlier planned stages, and reports missing inputs before
submission. Readiness uses the selected stage's direct blocking asset contract,
not evidence that every conventional predecessor ran.

Three concepts remain separate:

- **required assets and managed executions** are direct, blocking stage inputs;
- **advisory assets** are commonly useful context but never block execution;
- **typical upstream stages** document conventional lineage and produce warnings
  when skipped, but are not themselves readiness requirements.

The default `assets` policy schedules an upstream producer only for a missing
blocking asset. Existing external or adopted assets therefore allow the
requested stage to run directly:

```bash
sbt plan rapids --project /path/to/imported-anndata-project
sbt run rapids --project /path/to/imported-anndata-project --dry-run
```

Select another policy explicitly when needed:

```bash
sbt plan cellpose --dependency-policy assets  # default
sbt plan cellpose --dependency-policy none    # selected stages only
sbt plan cellpose --dependency-policy all     # complete conventional lineage
```

Preview exact `sbatch` arguments, dependency order, run paths, and exported SBT
context:

```bash
sbt run segmentation --dry-run
```

Dry runs create no persistent run record and submit no jobs. Displayed run paths
are prospective only.

Machine-readable dry runs also return a short-lived state-bound preview token
and action receipt:

```bash
sbt run segmentation --dry-run --format json
```

Directory-independent integrations submit the unchanged plan with
`--plan-token` and send bounded decision provenance through
`--provenance-stdin`. SBT rejects an expired token or any change to project ID,
configuration, resolved stages, dependency policy, stable asset state, or
execution index. Inventory capture timestamps are excluded from the token so
repeated inspection alone does not invalidate an unchanged preview. Reusing a
token after a successful submission returns the original run idempotently rather
than creating duplicate jobs.

Submit:

```bash
sbt run segmentation
sbt run prep denoise cellpose nimbus
sbt run cellpose --reason "Repeat with a larger diameter after fragmentation."
sbt run cellpose --note "Review ROI_17 carefully." --note "Compare with run 2026..."
```

Immediately before a real submission, `sbt run` resolves the Conda environments
mapped to the selected stages and dependencies. Missing repository-managed
environments have their installation specifications validated, then are listed
and offered for installation. If a specification is invalid, the user declines,
an external environment is missing, installation fails, or the environment is
still not visible afterward, the command stops before creating the run record
or calling `sbatch`.

Installation always requires an explicit interactive `y`; `sbt run` has no
flag that bypasses this prompt. Non-interactive workflows must prepare the
required environment beforehand with `sbt env sync <key>`. `--dry-run` remains
side-effect free: it validates and previews the workflow without checking or
installing Conda environments.

`--no-deps` remains as a compatibility alias for `--dependency-policy none`:

```bash
sbt run cellvision-cluster --no-deps --dry-run
sbt run cellvision-cluster --no-deps
```

Neither `none` nor `--no-deps` relaxes direct input validation. The command fails
before submission when a blocking asset is absent. When a mode or several stages
are explicitly selected, actual data dependencies between selected stages are
retained. Independent selected stages are not chained merely because one appears
earlier in the display; SLURM `afterok` edges are emitted only for actual plan
dependencies.

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

## Owned queue and guarded cancellation

```bash
sbt squeue
sbt squeue --job 12345678 --format json
sbt cancel 12345678 --reason "Submitted with incorrect parameters" --dry-run
sbt cancel 004 --project "My cohort" --reason "Terminal input failure" --dry-run
```

`sbt squeue` asks SLURM only for jobs owned by the current account. Its output
contains exact job IDs and bounded state/resource metadata, but never a
username. It marks jobs linked to registered SBT executions while retaining
other jobs owned by the same user so pipeline operation can account for them.

Cancellation always starts with `--dry-run`. The preview returns a five-minute
token bound to the job identity, state, project/execution association, and
reason. Pending jobs do not automatically require another confirmation;
running, completing, and stage-out jobs do. The execution call rechecks current
ownership/state before invoking `scancel` internally and writes a cancellation
audit. Without `--project`, only an exact numeric job ID (optionally an array
task) is accepted.

## Verified transfer, ZIP, upload, and backup

```bash
sbt transfer list --project "My cohort" --format json
sbt artifacts list 004 --project "My cohort" --format json
sbt transfer preview-download ITEM_ID --project "My cohort" --format json
sbt zip ITEM_ID ITEM_ID --project "My cohort" --dry-run --format json
```

Transfer commands accept stable item IDs, never caller-selected project paths.
Folders and multiple items should use `sbt zip`; bundles are ZIP64 and contain a
transfer manifest. Preparation calculates size/count, rejects links and special
files, and requires `--allow-large-transfer` above 3 GiB. SBT records the final
size and SHA-256 for SFTP verification.

Uploads use a preview, isolated staging target, and commit sequence. Metadata
accepts files or folders. A file uploaded directly to the project root must be
`.h5ad`; commit verifies its HDF5 signature. Existing destinations require an
explicit overwrite preview, are moved to a verified `.sbt/backups` record, and
only then are replaced atomically. `sbt transfer backups` lists recovery IDs;
`sbt transfer restore BACKUP --dry-run` produces the state-bound token required
for restore. Restoration retains the selected backup and backs up any displaced
current data. No transfer command deletes project files.

All gateway-facing JSON operations return an action receipt which lists each
action, its justification, outcome, evidence, warnings, and whether state
changed.

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
- The environments required by the selected stages exist under the fixed names
  in the central registry. `sbt run` can install missing repository-managed
  environments; external environments must be prepared separately.
- The site accepts standard `--parsable`, `--chdir`, `--output`, `--error`,
  `--export`, and `afterok` options.

`SBT_TOOLKIT_ROOT` can select a different wrapper checkout for planning and
submission, but wrappers that still source `$HOME/imcanalysis` retain that
legacy site assumption. The external `scport` wrapper also retains its current
fixed `processed/` and `masks/` arguments.
