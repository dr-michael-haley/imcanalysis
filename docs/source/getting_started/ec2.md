# Linux and EC2 without SLURM

Use `sbt run --backend local` for managed sequential execution on Linux. It uses
the same project config, stage aliases, built-in modes, dependency planning,
numbered execution IDs, and reports as the SLURM backend. No job manager is
required. SLURM remains the default when no backend is selected.

## Run Nimbus, BioBatchNet, RAPIDS, and visualisation

Activate `sbt-cli` and adopt the project once if it does not already have
`.sbt/project.yaml`:

```bash
conda activate sbt-cli
cd /scratch/projects/GBMp2_bgnorm
sbt project adopt .

sbt run nimbus bbn rapids vis --project . --backend local --dry-run
sbt run nimbus bbn rapids vis --project . --backend local --detach
```

The default `assets` dependency policy adds upstream stages when blocking assets
are missing. To run only the stages explicitly listed, add
`--dependency-policy none`; input validation still applies. Built-in modes such
as `segmentation` work with the local backend too. There are no custom workflow
definitions or automatic retries.

The launcher selects each stage's registered Conda environment and checks that
its Python exists before allocating execution IDs. It uses `conda run` and does
not install missing environments. Stages with multiple environments switch at
the same module boundaries as their SLURM wrappers. For these four stages, the
default environment is `sbt-analysis`.

If you want to use the Conda environment already active in your shell, pass
`--use-active-env`. SBT selects its Python from `CONDA_PREFIX`, even if the `sbt`
executable itself belongs to the launcher environment:

```bash
conda activate sbt-analysis
sbt run nimbus bbn rapids vis --project . --backend local --use-active-env --detach
```

This explicitly uses the active environment for **every** selected module. It
must contain their dependencies. `--environment` retains its existing meaning:
an override for one stage that normally uses a single registered environment.

## Monitor and cancel

Without `--detach`, the command streams saved logs and waits for the workflow.
Ctrl+C cancels it and waits for process cleanup. With `--detach`, the command
returns only after the worker acknowledges startup. The worker has its own
session and survives closing the shell or disconnecting SSH.

The startup message lists the allocated execution IDs and log paths. Substitute
your actual execution ID for `001` below:

```bash
sbt status 001 --project . --details
sbt logs 001 --project . --tail 100
sbt logs 001 --project . --follow
sbt summary --project .
```

`status` shows the entire local workflow, current module process group,
environment, and selected stage's start time and elapsed time. `--details` adds
active-stage CPU percentage, summed resident memory, and GPU information when
`nvidia-smi` is available. CPU percentage can exceed 100% when several cores are
used. GPU device utilization is for the whole device; matching process IDs show
that this stage has a GPU allocation, not that it is continuously computing.

Each stage has one combined stdout/stderr log in
`.sbt/runs/<workflow>/logs/<stage>.log`. Python output is unbuffered. Both
`--stdout` and `--stderr` select this combined log locally. Worker setup errors
go to `logs/worker.log` in the same run. Following logs stops when the selected
stage finishes; Ctrl+C while following only stops viewing. `latest` means the
last allocated execution, which may still be waiting for earlier stages.

Cancellation follows the CLI's existing preview-token convention:

```bash
sbt cancel 001 --project . --reason "Stop this trial" --dry-run
# Copy the printed token into the next command.
sbt cancel 001 --project . --reason "Stop this trial" --plan-token '<token>'
```

For a local execution this stops the **whole workflow**, including the current
stage's child processes and all stages still waiting. Cancellation first sends
TERM to the active process group, then KILL after a five-second grace period
if needed. A failed stage stops the sequence and marks later stages `blocked`.
The launcher returns a nonzero exit code for a failed/cancelled foreground run.
Use `status` to inspect the result of a detached run.

One local workflow can run per project. Other projects can run independently;
SBT does not divide CPU, RAM, or GPUs between projects. Detached runs do not
survive stopping/rebooting an instance and do not resume automatically. A
missing worker is reported as `unknown`, never inferred to have succeeded.
Cancellation verifies host, boot, and process identity. It will not signal a PID
from an old instance. If a worker is forcibly killed and leaves children alive,
inspect those processes on the original host before launching another run.

Managed runs save a config snapshot and numbered reports below
`general.outputs_folder`. Existing direct-run reports remain separate. See
[local execution internals](../pipeline/local-execution.md) for the backend
boundary and future Nextflow integration.

## Keep EC2 machine locations on persistent storage

Keep the same mount path on replacement instances, for example `/scratch`, and
keep Conda, the checkout, projects, and this small environment file there:

```bash
# /scratch/sbt-machine.sh
export SBT_TOOLKIT_ROOT=/scratch/imcanalysis
export CONDA_EXE=/scratch/miniconda3/bin/conda
export SBT_CONDA_SH=/scratch/miniconda3/etc/profile.d/conda.sh
export SBT_STATE_HOME=/scratch/sbt-state
export SBT_BACKEND=local
source "$SBT_CONDA_SH"
```

Source it from each new instance's shell startup file:

```bash
source /scratch/sbt-machine.sh
conda activate sbt-cli
```

With `SBT_BACKEND=local`, `sbt run nimbus bbn rapids vis --project . --detach`
is sufficient. Override it with `--backend slurm` when needed. The project
contains its execution history; `SBT_STATE_HOME` keeps launcher/environment state
on the volume. Keep any project registry selected with `SBT_IMC_CONFIG` on the
volume too. Mount the volume before starting Conda or SBT. Existing Conda
environments and editable installs contain absolute paths, so preserving the
mount path avoids reinstalling them solely because the instance changed.

Instance-specific GPU drivers still need to be available on the replacement
machine. Mounting saved files does not transfer a running process.

## Prepare the environments and project

Use the Linux scientific environments listed in
[environment management](../pipeline/environments.md). The example uses
`sbt-analysis`, `sbt-tensorflow`, and `sbt-cellpose-sam`. SBT must be installed in
each environment that runs a stage. From the checkout, after activating each
environment:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

The lightweight `sbt-cli` environment is for inspection and configuration; it
does not contain the scientific dependencies. The Windows/macOS local environment
exports are not the Linux pipeline environments. GPU stages still need a suitable
GPU and driver; changing the launcher does not change their compute requirements.

Prepare a dataset directory containing `config.yaml` and its inputs. Review the
panel/metadata and scientific settings before running the example. You can create
an initial config with `sbt config template --output config.yaml` from that
directory. Project adoption is not required for direct Python execution.

## Optional plain Bash runner

The repository's `run_local.sh` is still available for an editable list of Python
calls with one console log. It has no managed execution IDs or cancellation
tracking. Use the CLI above for those features.

```bash
export SBT_TOOLKIT_ROOT=/mnt/software/imcanalysis
export SBT_CONDA_SH=/opt/conda/etc/profile.d/conda.sh

cp "$SBT_TOOLKIT_ROOT/run_local.sh" /mnt/data/my_project/run_local.sh
# Edit /mnt/data/my_project/run_local.sh to choose the stages you want.
bash /mnt/data/my_project/run_local.sh /mnt/data/my_project
```

The example runs preprocessing, denoising, DNA preprocessing, Cellpose-SAM, then
Nimbus segmentation/quantification. Add, remove, or reorder the literal
`python -m SpatialBiologyToolkit.scripts.<module>` lines and their Conda
activations. The [stage reference](../pipeline/stages/index.md) lists modules and
environments. Multi-module stages need all their commands; Cellpose, for example,
needs DNA preprocessing before mask generation. Rerunning the file starts at the
top, subject to each scientific stage's existing skip/repeat settings.

The script changes to the project directory before loading config or running any
stage. This matters: many existing scripts resolve relative paths against the
working directory. Use a full path for `SBT_CONFIG` if selecting another config:

```bash
export SBT_CONFIG=/mnt/data/my_project/config_trial.yaml
export SBT_LOG_DIR=/mnt/data/my_project/logs
bash /mnt/data/my_project/run_local.sh /mnt/data/my_project
```

To continue after disconnecting SSH:

```bash
nohup bash /mnt/data/my_project/run_local.sh /mnt/data/my_project >launch.log 2>&1 &
```

`launch.log` contains the selected console-log location and any early startup
error. Stage output goes to `logs/run-<timestamp>-<pid>.log` below the project by
default. Inspect it with `tail -f <log-file>`. This does not resume after an
instance shutdown.

## Plain Bash logs and reports

Both stdout and stderr are saved, including import errors and tracebacks. Python
output is unbuffered. `set -e` stops the script when a command reports failure;
there is no additional error detection or recovery.

Registered stages retain their existing direct reports under
`general.outputs_folder/direct/`. Direct runs do not allocate the numbered
executions or SLURM records created by `sbt run`; inspect the console log and
direct reports rather than expecting scheduler status or managed `sbt logs`.

Stages also append to `logging.log_file` (default `pipeline.log`). To keep just
the console capture, set:

```yaml
logging:
  console_only: true
  to_console: true
```

Some legacy stages fill missing defaults into the selected YAML when loading
it. Copy the config first if you want to retain its exact original contents.

## Machine locations

Set these in your shell or shell startup file. Defaults retain the CSF3 layout.

| Setting | Default | Used by |
| --- | --- | --- |
| `SBT_TOOLKIT_ROOT` | Shell helpers: `~/imcanalysis`; Python also discovers the installed checkout | SLURM wrappers, environment registry, legacy installer and `pl`/`pll` |
| `SBT_CONDA_SH` | `~/miniconda3/etc/profile.d/conda.sh` | Local runner and environment diagnostics |
| `SBT_CONDA_ENV_ANALYSIS` | `sbt-analysis` | Local runner's analysis stages |
| `SBT_CONDA_ENV_TENSORFLOW` | `sbt-tensorflow` | Local runner's denoising stage |
| `SBT_CONDA_ENV_CELLPOSESAM` | `sbt-cellpose-sam` | Local runner's mask stage |
| `SBT_SCPORTRAIT_CONVERTER` | `~/scPortrait_to_IMC/imc_to_single_cells.py` | External scPortrait export wrapper |
| `SBT_DATA_ROOT` | `~/scratch` | Optional legacy `cds` helper only |
| `SBT_CONFIG` | Runner: project `config.yaml` | Stage configuration |
| `SBT_LOG_DIR` | Project `logs` | Local runner's console logs |

Use full paths for machine settings. `SBT_TOOLKIT_ROOT` selects resources; it
does not install the package or replace the editable installation in each
scientific environment. Existing `PIPELINE_DIR` overrides still take precedence
in `pl`/`pll`. The local runner initializes Conda directly and does not source
SLURM wrappers. Shared wrapper setup purges environment modules only when the
`module` command exists.

Python-side Conda discovery already supports `PATH` and `CONDA_EXE`. Its state
directory can be changed with `SBT_STATE_HOME`, and the project-registry file
with `SBT_IMC_CONFIG`. Neither needs a new project config key. If you use the
optional legacy installer, its startup/credential file remains `~/.imc_config`.

## Dataset and model locations

Keep paths relative to the project where practical. Moving the whole directory
then only changes the runner's project argument. These fields already exist:

| Contents | Config field |
| --- | --- |
| Raw IMC input | `general.imc_files_folder` |
| Metadata and panel tables | `general.metadata_folder` |
| Raw channel images / denoised images | `general.raw_images_folder` / `general.denoised_images_folder` |
| Masks / cell tables / TIFF stacks | `general.masks_folder` / `general.celltable_folder` / `general.tiff_stacks_folder` |
| Main AnnData | `general.anndata_path` |
| Reports | `general.outputs_folder` |
| Denoising weights | `denoising.weights_save_directory` |
| Custom Cellpose-SAM model | `createmasks.cell_pose_sam_model` |
| HyPERSTAC encoder weights | `hyperstac.encoder_weights` |
| Optional local STARLING checkout | `starling.starling_repo_path` |
| Backgating masks | `visualization.backgating_mask_folder` (independent default: `masks`) |
| External scPortrait output | `scportrait.projects_root` (default: `scPortrait`) |

Also review any stage-specific reference AnnData, CSV, image, and model paths in
your existing config. scPortrait now reads `general.denoised_images_folder` and
`general.masks_folder` rather than fixing those arguments in its wrapper.

Use relative paths or full absolute paths in YAML: expansion of `~` is not
consistent across older stages, and `$VARIABLE` substitution is not a general
config feature. There is no need to change the default project folder names
merely because the project now lives on an EC2 volume.
