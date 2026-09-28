# Linux and EC2 without SLURM

Use the repository's `run_local.sh` as an editable list of Python stage commands.
Bash runs them in order and stops at the first nonzero exit. Each invocation saves
one console log. Edit the list to choose your workflow; there is no queue, retry
logic, dependency expansion, or background service.

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

## Copy, edit, and run

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

## Logs and reports

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
