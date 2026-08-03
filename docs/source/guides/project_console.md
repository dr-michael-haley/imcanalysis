# SBT Project Console

The SBT Project Console is a lightweight cockpit for initialized or adopted SBT
projects. It explains stages, modes, and configuration fields; edits
configuration with validation and backups; inspects the asset register and
asset-aware workflow readiness; and reads durable execution records, reports,
log tails, and project notes. A project selector and dedicated **Projects** page
make the same application useful across a portfolio of IMC projects.

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

## Register the project portfolio

Register each existing SBT project once:

```bash
sbt project register --project /path/to/first-project --name "First cohort" --default
sbt project register --project /path/to/second-project --name "Second cohort"
sbt project list
```

Only an initialized or adopted project containing `.sbt/project.yaml` can be
registered. Registration does not validate scientific assets, move data, or
change the project. This permits a temporarily incomplete or misconfigured
project to remain visible for recovery.

The portfolio is stored in an SBT-managed block inside `~/.imc_config` as the
shell-compatible `SBT_PROJECTS_JSON` variable. The writer:

- preserves `IMC_EMAIL`, `OPENAI_API_KEY`, comments, and all unrelated lines;
- reads the registry variable as data and never sources or executes the file;
- replaces only its own delimited block using an atomic write;
- preserves the existing file mode, using mode `600` for a new file.

Use the CLI or GUI rather than editing the compact JSON by hand:

```bash
sbt project set-default "Second cohort"
sbt project unregister "First cohort"
```

Unregistering only forgets the central reference. No project files are deleted.
Unavailable paths and project-identity mismatches remain visible in the
**Projects** page until corrected or unregistered.

## Launch locally

Launch an explicit project:

```bash
sbt gui project --project /path/to/project
```

When `--project` is omitted, project discovery first checks the current
directory and its parents. Outside a project, the console opens the available
registered default, or the first available registered project. Use the selector
above every page or the **Projects** page to switch without restarting.

Use read-only mode when reviewing without permitting config, notes, or central
registry writes. Project switching remains available:

```bash
sbt gui project --read-only
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
sbt gui project
```

`-t 30` requests 30 minutes. The application performs bounded filesystem reads
and lightweight YAML/model validation, so a short interactive job is suitable.
Save config or notes explicitly before the allocation ends or the network
session is closed.

## Readiness model

Readiness deliberately separates three different ideas:

1. **Blocking direct requirements** are the assets or managed prior reports the
   selected scientific stage actually needs. Their absence makes that stage not
   ready.
2. **Advisory context assets** are commonly present and may aid interpretation,
   but their absence never blocks execution.
3. **Typical upstream stages** describe conventional provenance. Missing stages
   are warned about, but are not proof that a direct asset is absent. Imported
   AnnData, externally prepared masks, and adopted projects are first-class.

The default **Asset-aware upstream selection** adds a conventional producer only
when a blocking direct asset is missing. If the required asset already exists,
the requested stage remains independently runnable and skipped lineage is shown
as a warning. **Explicit stages only** never adds producers. **All conventional
upstream stages** reproduces the full lineage when a deliberate rerun is wanted.

For example, an adopted project with a configured AnnData file can be ready for
`rapids`, `subcl`, or `vis` even when it has no raw MCD files and no recorded
`prep` or `nimbus` execution. Conversely, `prep` itself still requires recognised
raw MCD/TXT inputs, and `cellpose` still requires non-empty denoised images.
Stages that consume prior SBT reports, such as `hyperstac-stability`, declare
those managed execution folders as direct blocking requirements; asset-aware
planning schedules only the missing report producers.

Readiness inspection never creates a run record, checks Conda environments, or
constructs a live scheduler query.

## Configuration safety

The structured editor makes configuration provenance visible at a glance:

- **grey — inherited default:** the field is not stored in `config.yaml`; SBT is
  supplying the displayed model default;
- **blue — stored override:** the displayed value is explicitly present in the
  project `config.yaml`;
- **gold — unsaved change:** the displayed proposal differs from the file on
  disk;
- **pink — pending reset:** saving will remove the stored key and return the
  field to its inherited default.

The summary strip reports how many individual overrides and sections are
actually on disk, how many values remain inherited, and how many changes are
staged. The origin filter can show all values, only customised values, only
inherited defaults, or only unsaved changes. These controls are independent of
the existing stage/mode, expertise-level, and text-search filters.

Use **Prepare an unconfigured section** or a **Prepare stage/mode** scope to
navigate fields needed for future work. Opening a section does not write it and
does not copy all defaults into YAML. Change only the fields the project needs;
on save, those individual overrides are added while untouched values continue
to inherit canonical defaults. This keeps adopted and partially prepared
projects compact without preventing advance configuration.

Saving performs all of the following:

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

If YAML syntax or Pydantic validation is already broken, the console opens that
project in recovery mode. Other project pages are unavailable until the raw YAML
editor validates and saves a repaired file with the same backup and audit
protections. The global project selector remains available so the user is not
trapped in the broken project.

## Pages

- **Projects** shows registered, unavailable, and currently unregistered
  projects; opens, registers, defaults, or forgets portfolio entries.
- **Dashboard** summarizes project identity, structural validation, assets,
  executions, and the latest recorded status snapshot.
- **Stages & modes** renders the typed registry, direct/advisory requirements,
  typical lineage, and shared Markdown explainers.
- **Configuration** provides stage/section-aware schema controls, search, level
  and provenance filters, colour-coded stored/default/staged state, preparation
  navigation, advice, constraints, diff, backup, and audit.
- **Assets** computes paths, lifecycle, presence, producer stages, and blocking
  consumer stages without recursively scanning the project.
- **Readiness** compares the three upstream policies and displays blocking
  inputs, advisory context, actual scheduler edges, and skipped lineage.
- **Runs** reads execution identity, provenance, stage metrics, Markdown reports,
  resolved configuration snapshots, and bounded stdout/stderr tails.
- **Notes** edits `.sbt/project_notes.md` with explicit save, concurrent-change
  detection, and a backup.

Project initialization, adoption, migration, execution removal, asset cleanup,
environment management, live scheduler refresh, and scientific exploration stay
in their existing CLI or specialist interfaces.
