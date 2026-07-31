# HPC setup

This is the recommended installation route for SpatialBiologyToolkit. It starts
from obtaining an HPC account and installing Conda, so no previous Linux or HPC
experience is assumed. Experienced users can skip directly to the numbered
step they need.

By the end, you will have:

- a copy of the toolkit at `~/imcanalysis`;
- a small `sbt-cli` Conda environment for project and job control;
- a dataset project that can be checked and previewed;
- only the scientific environments needed by the pipeline stages you run.

The complete pipeline needs a Linux HPC cluster using SLURM. The commands below
run on the **HPC login node**, not in a terminal that is still on your Windows
or macOS computer. See the [complete beginner's guide](beginners.md) for an
explanation of terminals, HPC, SLURM, Conda, environments, and projects.

## 1. Obtain an HPC account and connect

Follow your institution's instructions to request an account. You will need:

- your HPC username;
- the cluster's SSH address;
- any required VPN or multi-factor authentication;
- the cluster's storage and acceptable-use guidance.

Open PowerShell or Windows Terminal on Windows, or Terminal on macOS/Linux, and
connect with:

```bash
ssh <username>@<cluster-address>
```

For the University of Manchester CSF3, the current command is:

```bash
ssh <username>@csf3.itservices.manchester.ac.uk
```

Replace `<username>` with your University username. CSF3 requires the
University VPN when connecting from outside its network. Its current account,
connection, file-transfer, storage, Linux, and SLURM instructions are collected
in the [CSF3 getting-started
guide](https://ri.itservices.manchester.ac.uk/csf3/getting-started/).

The first connection may ask you to confirm a host fingerprint. Compare it with
your institution's instructions. When entering a password, the terminal does
not normally display characters.

After login, practise these safe commands:

```bash
pwd
ls
echo "$HOME"
```

`$HOME` is your personal directory on the cluster. The prompt may include the
login-node name and normally ends in `$`.

> [!IMPORTANT]
> Use the login node for short commands, file management, validation, and job
> submission. Do not run segmentation, denoising, or another heavy analysis
> directly on it. `sbt run` submits those calculations to SLURM compute nodes.

## 2. Choose where data will live

Keep the small software checkout in your home directory at `~/imcanalysis`.
Keep dataset projects on storage suitable for larger data, according to your
cluster's policy.

Scratch space is often fast but temporary. On CSF3 it is subject to automatic
cleanup. Do not keep the only copy of raw data or irreplaceable results there.
Before uploading data, decide which backed-up project or research-data storage
will hold the permanent copy.

You can inspect free space with:

```bash
quota -s
df -h .
```

Not every cluster provides `quota`; if it is unavailable, use the storage
command in your cluster's documentation.

## 3. Check Git, SLURM, and Conda

Run each command separately:

```bash
git --version
sbatch --version
conda --version
```

- If Git prints a version, it is ready. If not, follow your cluster's module
  instructions or ask its support team; do not use `sudo` on a shared cluster.
- If `sbatch` prints a SLURM version, you are on a suitable login node. If not,
  check the cluster login and module instructions.
- If Conda prints a version, skip to step 5 below. If the shell says
  `conda: command not found`, continue with step 4.

Some clusters provide Conda as a module. Run `module avail` and follow the local
documentation before installing your own copy. On CSF3, a user-owned Miniconda
installation is a straightforward option.

## 4. Install Miniconda when `conda` is missing

Miniconda provides Python and the Conda environment manager without installing
a large scientific stack. These instructions are for a standard 64-bit Linux
HPC login node.

First check the processor type:

```bash
uname -m
```

If it prints `x86_64`, use the commands below. For another value, choose the
matching Linux installer from Anaconda's official [Miniconda Linux installation
guide](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)
instead of guessing.

### Download the installer

```bash
cd "$HOME"
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

If `curl` is unavailable but `wget` exists, use:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Confirm that the file exists:

```bash
ls -lh Miniconda3-latest-Linux-x86_64.sh
```

For additional assurance, calculate its checksum:

```bash
sha256sum Miniconda3-latest-Linux-x86_64.sh
```

Compare the result with the same installer in Anaconda's official [Miniconda
archive](https://repo.anaconda.com/miniconda/). Do not run an installer whose
checksum does not match.

### Run the installer

```bash
bash ./Miniconda3-latest-Linux-x86_64.sh
```

The installer asks several questions:

1. Press **Enter** to read through the licence. Press **Space** to move a page
   at a time, or `q` if the pager says that key will finish displaying it.
2. Type `yes` to accept the licence.
3. Press **Enter** to accept the default location, normally
   `/home/<username>/miniconda3`. A home-directory install does not need `sudo`.
4. Type `yes` when asked whether to initialize Miniconda/Conda.

Load the updated shell configuration:

```bash
source "$HOME/.bashrc"
```

Then verify the installation:

```bash
conda --version
conda info --base
```

The second command should show a directory in your home directory. If `conda`
is still not found, disconnect with `exit`, connect again, and repeat the two
checks. Keep the downloaded installer until the setup works; it can then be
removed if you wish.

## 5. Clone the toolkit

The active SLURM wrappers retain a compatibility assumption that the checkout
is at `~/imcanalysis`, so use that location:

```bash
cd "$HOME"
git clone https://github.com/dr-michael-haley/imcanalysis.git
cd "$HOME/imcanalysis"
```

If Git says the destination already exists, do not clone over it. Inspect and
update the existing checkout instead:

```bash
cd "$HOME/imcanalysis"
git status --short
git pull --ff-only
```

No output from `git status --short` means the checkout is clean. If it lists
files you intentionally changed, preserve or commit them before pulling.

## 6. Install the lightweight `sbt` command

From the repository root, run:

```bash
bash install/bootstrap_sbt.sh
```

This creates a small Conda environment called `sbt-cli` and links it to this
checkout. It does **not** install all scientific environments or submit work.
The first installation may take several minutes.

Activate it and verify the command:

```bash
conda activate sbt-cli
sbt --help
```

The prompt often starts with `(sbt-cli)` after activation. If your cluster does
not support shell activation, prefix commands with `conda run -n sbt-cli`, for
example:

```bash
conda run -n sbt-cli sbt --help
```

## 7. Install `conda-lock`

The repository-managed scientific environments use exact lockfiles.
Install `conda-lock` once in the Conda base environment:

```bash
conda install --name base --channel conda-forge conda-lock
```

When Conda shows a package plan and asks `Proceed ([y]/n)?`, review it and press
**Enter** or type `y`. Confirm that the command is available and inspect the
environment registry:

```bash
conda run -n base conda-lock --version
sbt env list
```

Both commands are read-only. The toolkit invokes `conda-lock` through the base
environment, in line with the official [conda-lock installation
guidance](https://conda.github.io/conda-lock/getting_started/).

## 8. Do not install every scientific environment

The environment list includes several large and specialised stacks. A user who
only runs segmentation does not need BioBatchNet, CellCharter, STARLING,
HyPERSTAC, or other unrelated environments.

Install an environment manually only when you already know you need it:

```bash
sbt env validate-spec <key>
sbt env sync <key> --dry-run
sbt env sync <key>
```

Replace `<key>` with an environment key shown by `sbt env list`. For most
beginners, continue without running these commands. `sbt run` checks
the environments used by the resolved workflow immediately before it creates a
run record or submits jobs. It first validates the installation specifications.
If a valid repository-managed environment is missing, it lists the affected
stages and asks:

```text
Install the missing environment(s) now (...) [Y/n]:
```

Press **Enter** to accept the default `Y`, or type `n` to stop. On acceptance,
`sbt` installs only those missing environments, applies the editable toolkit
link, runs their smoke tests, checks again, and then submits the workflow.

If a required installation specification is invalid, `sbt` stops before the
prompt and tells you to update the checkout or report the problem to the
maintainer. Do not work around that safeguard by installing arbitrary package
versions.

Some specialist environments are marked **external** because the repository
does not yet provide an installable lockfile. `sbt` detects a missing external
environment but cannot create it; it stops before recording or submitting the
run and directs you to `sbt env show <key>` and the relevant stage guide.

For a non-interactive script, request automatic installation of missing managed
environments explicitly:

```bash
sbt run segmentation --install-missing-envs
```

This option never installs an external environment. A dry run never checks or
installs environments because it does not execute them.

## 9. Create or adopt a dataset project

Keep dataset projects outside the repository. Substitute a location approved
for research data on your cluster. For a new project:

```bash
mkdir -p /path/to/project-storage/my_dataset
cd /path/to/project-storage/my_dataset
sbt project init
```

Do not type `/path/to/project-storage` literally. For a first CSF3 project, ask
your research group or the CSF team which project or scratch path to use.

The command creates `config.yaml`, input directories, `.sbt/` run state, and an
`outputs/` report index. Put raw inputs at the paths shown in `config.yaml` and
adjust the dataset settings.

For an existing dataset that already has `config.yaml`:

```bash
cd /real/path/to/existing_project
sbt project adopt --config config.yaml
```

Adoption records the project without moving or rewriting data. Check either
kind of project with:

```bash
sbt project describe
sbt project assets
sbt project validate --mode segmentation
```

Validation is read-only. Missing assets are reported by role and path.

## 10. Preview the first workflow

List the available modes, inspect the plan, and preview the exact SLURM
submission:

```bash
sbt modes list
sbt plan segmentation
sbt run segmentation --dry-run
```

The dry run validates configuration, assets, dependencies, and wrapper files.
It creates no run directory, installs no Conda environment, and submits no job.
Fix any reported problem and repeat it until the preview succeeds.

## 11. Submit and monitor

Only submit when the preview is correct:

```bash
sbt run segmentation
```

If required managed environments are absent, accept the installation prompt or
stop and inspect it. Environment installation can take time and produces no
SLURM job. Submission begins only after every required environment exists.

After submission:

```bash
sbt status latest
sbt logs latest
sbt summary
sbt report latest
```

It is safe to disconnect with `exit`; submitted jobs continue. Human-readable
stage reports are below the project's `outputs/` directory. Technical run
records and log paths are below `.sbt/`.

## 12. Optional email and AI credentials

Core processing does not require an OpenAI key. Set these only when a selected
workflow explicitly needs them:

```bash
export IMC_EMAIL="your.email@example.org"
export OPENAI_API_KEY="your-key"
```

For persistent values, use a private file outside the Git repository and set
mode `600`. Never put credentials in `config.yaml` or commit them. The optional
legacy installer can create `~/.imc_config`; its exact behaviour belongs in the
[installation helper scripts reference](../reference/installation_helpers.md),
not in the beginner setup.

## 13. Update an existing installation

Start from the repository and check for local changes:

```bash
cd "$HOME/imcanalysis"
git status --short
git pull --ff-only
bash install/bootstrap_sbt.sh
conda activate sbt-cli
sbt env list
```

Do not synchronize every scientific environment after every update. The next
real `sbt run` checks and offers to install any newly required managed
environment. For an already installed environment whose specification changed,
inspect it with `sbt env compare <key>` and follow the [fixed Conda environment
management guide](../pipeline/environments.md) before recreating it.

## Troubleshooting

### `conda: command not found`

If the cluster provides a Conda module, load it according to its documentation.
Otherwise repeat step 4, then run `source "$HOME/.bashrc"` or reconnect. Do not
install system-wide software with `sudo`.

### `sbt: command not found`

Run `conda activate sbt-cli`. If that environment does not exist, return to
`$HOME/imcanalysis` and rerun `bash install/bootstrap_sbt.sh`.

### `conda-lock` is missing

Run:

```bash
conda install --name base --channel conda-forge conda-lock
conda run -n base conda-lock --version
```

### `sbatch: command not found`

Confirm that SSH connected to the cluster's SLURM login node. Follow its module
instructions or contact support. A normal Windows or macOS terminal cannot
submit this pipeline directly.

### An environment is missing

Run the real `sbt run` and accept the prompt for a repository-managed
environment, or install just that environment explicitly:

```bash
sbt env show <key>
sbt env sync <key> --dry-run
sbt env sync <key>
```

Do not include the angle brackets. External environments need the setup named
in `sbt env show <key>` and their stage documentation.

### A wrapper is not executable

From the repository root:

```bash
chmod +x SLURM_scripts/*.sh
```

### The repository is elsewhere

`SBT_TOOLKIT_ROOT` can change where planning finds wrappers, but some active
wrappers still source `$HOME/imcanalysis`. Keeping the documented location is
the supported beginner setup.

### Do I need `make install` or `make envs`?

No. `make install` configures older shell shortcuts, and `make envs` installs
every repository-managed scientific environment. Neither is part of the
beginner workflow. See [installation helper scripts
reference](../reference/installation_helpers.md) only when maintaining a
legacy setup.
