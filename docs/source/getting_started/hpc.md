# HPC setup

This is the recommended setup for SpatialBiologyToolkit. It installs a small
`sbt` command environment on a Linux HPC login node and keeps the heavy
scientific software in separate environments used by SLURM jobs.

If the terminal, Conda, or SLURM are unfamiliar, read the
[complete beginner's guide](beginners.md) first.

## What you will set up

By the end of this guide you will have:

- the repository at `~/imcanalysis`;
- a lightweight `sbt-cli` Conda environment;
- the repository-managed scientific environments needed by the core pipeline;
- a dataset project that can be validated and previewed before submission.

The full pipeline requires Linux and SLURM. A local Windows or macOS environment
is covered separately in [local analysis setup](local.md).

## 1. Check the cluster prerequisites

Connect to your HPC login node and check for Git, Conda, and SLURM:

```bash
git --version
conda --version
sbatch --version
```

If your cluster supplies Conda through an environment module, follow the local
HPC instructions to load it. If Conda is not provided and user installations
are allowed, follow Anaconda's official
[Miniconda Linux installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install).
Restart the shell after installation and confirm that `conda --version` works.

You also need sufficient storage for the repository, environments, raw images,
and generated assets. Dataset projects usually belong on project or scratch
storage rather than in the repository.

> [!TIP]
> University of Manchester users should use the current
> [Research Infrastructure CSF help](https://ri.itservices.manchester.ac.uk/csf3/)
> for account, connection, storage, and cluster-module instructions.

## 2. Clone the toolkit

The active SLURM wrappers retain a compatibility assumption that the toolkit is
available at `~/imcanalysis`, so use that location:

```bash
cd "$HOME"
git clone https://github.com/dr-michael-haley/imcanalysis.git
cd "$HOME/imcanalysis"
```

If the repository already exists, do not clone it again. Check and update it:

```bash
cd "$HOME/imcanalysis"
git status --short
git pull --ff-only
```

If `git status --short` lists files you intentionally changed, preserve or
commit them before pulling.

## 3. Install the lightweight `sbt` launcher

From the repository root, run:

```bash
bash install/bootstrap_sbt.sh
conda activate sbt-cli
sbt --help
```

`bootstrap_sbt.sh` creates `sbt-cli` when needed and installs the toolkit there
in editable mode without the scientific dependency stack. It is safe to run
again after an update.

If shell activation is unavailable in your cluster setup, the equivalent form
is `conda run -n sbt-cli sbt --help`.

## 4. Install the environment manager

The repository's exact scientific lockfiles are installed with `conda-lock`.
Install it once in the Conda base environment:

```bash
conda install --name base --channel conda-forge conda-lock
```

The toolkit calls it as `conda run -n base conda-lock`; it does not need to be
installed in `sbt-cli` or in every scientific environment. The command follows
the official [conda-lock installation guidance](https://conda.github.io/conda-lock/getting_started/).

## 5. Validate and synchronize scientific environments

Keep `sbt-cli` active, then inspect the environment registry and preview any
changes:

```bash
sbt env doctor
sbt env list
sbt env validate-spec --all
sbt env sync --all --dry-run
```

Resolve any errors reported by `doctor` before continuing. When the preview is
correct, create or update the repository-managed environments:

```bash
sbt env sync --all
```

Synchronization installs exact Conda lockfiles, intentional pip extras, the
editable toolkit overlay, and registered smoke tests. Existing environments are
not silently replaced; use the explicit recreation options described in
[fixed Conda environment management](../pipeline/environments.md) if drift
requires a rebuild.

The managed environments currently cover segmentation, denoising,
CellPose-SAM, BioBatchNet, and CellCharter. RAPIDS, STARLING, scPortrait,
HyPERSTAC, and MaxFuse are registered external environments and are not created
by `sbt env sync --all`. You only need an external environment when selecting a
stage that uses it; `sbt env list` shows the current names and management state.

## 6. Create or adopt a dataset project

Keep dataset projects outside the repository. For a new project:

```bash
mkdir -p "$HOME/projects/my_dataset"
cd "$HOME/projects/my_dataset"
sbt project init
```

This creates a compact `config.yaml`, standard input directories, `.sbt/` run
state, and the `outputs/` report index. Put the raw inputs in the paths shown by
the config and adjust configuration values for the dataset.

For an existing project that already has a `config.yaml`:

```bash
cd /path/to/existing_project
sbt project adopt --config config.yaml
```

Adoption records the project without moving or rewriting scientific data. In
either case, inspect it before planning:

```bash
sbt project describe
sbt project assets
sbt project validate --mode segmentation
```

The validation command reports missing inputs and does not submit work.

## 7. Plan and preview the first run

List the available workflows, inspect the segmentation plan, and preview the
exact SLURM submission:

```bash
sbt modes list
sbt plan segmentation
sbt run segmentation --dry-run
```

Dry runs create no run record and submit no jobs. Fix missing-file,
configuration, wrapper, or environment errors before proceeding. See the
[pipeline workflow](../pipeline/workflow.md) for stage order and optional
branches.

## 8. Submit and monitor

Submitting changes external cluster state, so run this only when the preview is
correct:

```bash
sbt run segmentation
```

Inspect the recorded work with:

```bash
sbt status latest
sbt logs latest
sbt summary
sbt report latest
```

Human-readable stage reports are written below the project's `outputs/`
directory. Technical run records and log paths are stored below `.sbt/`.

## 9. Optional email and AI credentials

Core processing does not require an OpenAI key. Set these only if the selected
workflow needs them:

```bash
export IMC_EMAIL="your.email@example.org"
export OPENAI_API_KEY="your-key"
```

For persistent values, place the exports in a private file outside the Git
repository and restrict it with `chmod 600`. Never put credentials in
`config.yaml` or commit them to Git. The optional legacy shell installer can
create `~/.imc_config`; its exact changes are documented in the
[installation helper scripts reference](../reference/installation_helpers.md).

## 10. Update an existing installation

From a clean repository checkout:

```bash
cd "$HOME/imcanalysis"
git status --short
git pull --ff-only
bash install/bootstrap_sbt.sh
conda activate sbt-cli
sbt env doctor
sbt env sync --all --dry-run
```

If the dry run reports required environment changes, apply them with
`sbt env sync --all`. Because the toolkit is installed in editable mode,
ordinary source updates are visible immediately; rerunning the bootstrap also
refreshes command metadata safely.

## Troubleshooting

### `sbt: command not found`

Activate the launcher with `conda activate sbt-cli`, or run
`conda run -n sbt-cli sbt --help`. If the environment does not exist, rerun
`bash install/bootstrap_sbt.sh` from the repository root.

### `conda-lock` is missing

Run `conda install --name base --channel conda-forge conda-lock`, then repeat
`sbt env doctor`.

### `sbatch` is missing

Confirm that you are on a SLURM login node and follow the cluster's module or
login instructions. The end-to-end pipeline cannot be submitted from a normal
Windows or macOS terminal.

### A wrapper is not executable

From the repository root:

```bash
chmod +x SLURM_scripts/*.sh
```

### The repository is in a different directory

Planning can use `SBT_TOOLKIT_ROOT`, but some active wrappers still source files
from `$HOME/imcanalysis`. Moving the checkout is therefore an advanced site
configuration; the beginner setup should keep the documented location.

### Do I need `make install`?

No. It installs the older `cds`, `pl`, `pll`, and `pls` shell conveniences and
edits shell startup files, but it is not required for the `sbt` workflow. See
the [installation helper scripts reference](../reference/installation_helpers.md)
if you maintain a legacy setup.
