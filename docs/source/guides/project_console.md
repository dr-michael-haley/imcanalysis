# SBT Project Console

The SBT Project Console is a lightweight graphical view of one existing SBT
project. It explains stages, modes, and configuration fields; edits configuration
with validation and backups; inspects the asset register and workflow readiness;
and reads durable execution records, reports, log tails, and project notes.

The console deliberately cannot submit or control jobs. It does not call
`sbatch`, `srun`, `squeue`, `sacct`, or `scancel`. It also does not load AnnData,
MCD, TIFF, Scanpy, CUDA, or Napari. Recorded statuses are labelled with the time
at which the CLI last wrote them; the GUI does not refresh scheduler state.

## Install

Keep the normal `sbt-cli` environment small and install the GUI separately. On
Linux or macOS:

```bash
cd "$HOME/imcanalysis"
bash install/bootstrap_sbt_gui.sh
```

On Windows, use Anaconda PowerShell Prompt:

```powershell
powershell -ExecutionPolicy Bypass -File install/bootstrap_sbt_gui.ps1
```

Each bootstrap creates or refreshes the fixed `sbt-gui` environment from
`Local_envs/sbt_gui_env.yml` and installs the current checkout as an editable,
no-dependency overlay. It contains Qt and YAML round-trip support, but no
scientific environment.

## Launch locally

From an existing initialized or adopted project:

```bash
sbt gui project --project /path/to/project
```

Use read-only mode when reviewing a project without permitting config or notes
writes:

```bash
sbt gui project --project /path/to/project --read-only
```

If the active launcher does not contain Qt, `sbt` uses the fixed `sbt-gui`
environment. The environment is never created or updated automatically; run the
bootstrap command explicitly when installation is required.

## Launch on CSF3

Connect with X11 enabled, then request a short interactive display session from
the login node:

```bash
srun-x11 -p interactive -t 30
conda activate sbt-gui
sbt gui project --project /path/to/project
```

`-t 30` requests 30 minutes. Save config or notes explicitly before the
interactive allocation ends or the network session is closed.

## Configuration safety

The structured editor distinguishes explicit YAML values from inherited model
defaults. Saving performs all of the following:

1. validates the complete proposed Pydantic configuration;
2. preserves unknown legacy sections and keys;
3. preserves YAML comments when `ruamel.yaml` can round-trip them;
4. refuses to overwrite an externally changed config;
5. displays a diff and requires confirmation;
6. writes the new file atomically;
7. stores the previous text under `.sbt/config-backups/`;
8. writes an audit under `.sbt/audit/config-edits/`.

Resetting a field removes its explicit YAML key so the canonical model default
is inherited. There is no autosave.

If YAML syntax or Pydantic validation is already broken, the console opens in
recovery mode. Other pages remain unavailable until the raw YAML editor validates
and saves a repaired file with the same backup and audit protections.

## Pages

- **Dashboard** summarizes project identity, validation, assets, executions, and
  the latest recorded status snapshot.
- **Stages & modes** renders the typed registry and shared Markdown explainers.
- **Configuration** provides stage/section-aware schema controls, search, level
  filters, advice, constraints, diff, backup, and audit.
- **Assets** computes paths, lifecycle, presence, producer stages, and consumer
  stages without recursively scanning the project.
- **Readiness** expands dependencies and evaluates required assets without
  constructing or running a submission command.
- **Runs** reads execution identity, provenance, stage metrics, Markdown reports,
  resolved configuration snapshots, and bounded stdout/stderr tails.
- **Notes** edits `.sbt/project_notes.md` with explicit save, concurrent-change
  detection, and a backup.

Project initialization, adoption, migration, execution removal, asset cleanup,
environment management, live scheduler refresh, and scientific exploration stay
in their existing CLI or specialist interfaces.
