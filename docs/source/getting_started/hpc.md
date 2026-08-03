# CSF3 setup

This guide will take you from logging in to the University of Manchester's
CSF3 for the first time to submitting your first SpatialBiologyToolkit job. No
previous Linux or HPC experience is assumed.

By the end, you will have:

- Miniconda installed in your CSF3 home directory;
- a copy of SpatialBiologyToolkit at `~/imcanalysis`;
- the small `sbt-cli` environment used to control the pipeline;
- a project in scratch containing your raw MCD files;
- your first analysis submitted to the CSF3 queue.

If terms such as terminal, Conda environment, HPC, or SLURM are unfamiliar,
read the [complete beginner's guide](beginners.md) first.

## 1. Obtain a CSF3 account and connect

The University of Manchester's
[CSF3 getting-started guide](https://ri.itservices.manchester.ac.uk/csf3/getting-started/)
explains how to request an account and connect from Windows, macOS, or Linux.
Research IT also runs regular CSF3 training courses if you want more help.

You log in using your central University username and password and authenticate
with Duo MFA. Once logged in, you will be on a **login node** in your home
directory (`$HOME`). This is your personal directory on the cluster.

> [!IMPORTANT]
> The login node is for short commands, moving files, checking results, and
> submitting jobs. Do not run segmentation, denoising, or another large
> analysis directly on it. `sbt run` sends that work to a compute node using
> SLURM.

## 2. Install Miniconda

Miniconda provides Python and Conda, which we use to keep the different pieces
of software needed by the pipeline separate from one another.

Download the CSF3-compatible Linux installer into your home directory:

```bash
cd "$HOME"
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run it:

```bash
bash ./Miniconda3-latest-Linux-x86_64.sh
```

The installer asks several questions:

1. Press **Enter** to read the licence. Press **Space** to move through it.
2. Type `yes` to accept the licence.
3. Press **Enter** to accept the default install location in your home
   directory. You do not need `sudo`.
4. Type `yes` when asked whether to initialize Conda.

Load the updated shell settings:

```bash
source "$HOME/.bashrc"
```

Check that Conda is working:

```bash
conda --version
```

If `conda` is not found, log out of CSF3, reconnect, and try the command again.

## 3. Clone (download) the toolkit

SpatialBiologyToolkit expects its code to live at `~/imcanalysis` on CSF3:

```bash
cd "$HOME"
git clone https://github.com/dr-michael-haley/imcanalysis.git
cd "$HOME/imcanalysis"
```

`git clone` downloads the code. You only do this once.

## 4. Install the `sbt` command

From the `imcanalysis` folder, run:

```bash
bash install/bootstrap_sbt.sh
```

This creates a small Conda environment called `sbt-cli`. It contains the
lightweight `sbt` command used to set up projects, check files, submit jobs, and
view results. It does not contain all the scientific software used by the
pipeline.

Activate it and check that `sbt` is available:

```bash
conda activate sbt-cli
sbt --help
```

Your prompt should now start with `(sbt-cli)`. You will need to run
`conda activate sbt-cli` again each time you reconnect to CSF3.

## 5. Install `conda-lock`

The toolkit uses `conda-lock` to install tested versions of scientific
software. Install it once:

```bash
conda install --name base --channel conda-forge conda-lock -y
```

You do not need to run this command again for every project or environment.

## 6. Create a project folder in scratch

Keep the toolkit code in your home directory, but keep large analysis projects
in scratch:

```bash
cd "$HOME/scratch"
mkdir HyperionProject
cd HyperionProject
```

`HyperionProject` is only an example name. You can use a name that describes
your study.

Scratch is useful for active analysis but old files are automatically deleted.
Do not keep the only copy of raw data or irreplaceable results there.

## 7. Initialize the project and add the MCD files

Initialize a SpatialBiologyToolkit project inside the new folder:

```bash
sbt project init
```

This creates the files and folders used by the pipeline, including:

- `config.yaml`, which stores the analysis settings;
- `IMC_files/`, where the raw MCD files belong;
- `.sbt/`, which stores run information;
- `outputs/`, which stores readable analysis reports.

Upload the raw MCD files into `IMC_files/`. In MobaXterm, you can drag and drop
the files into this folder, although large files may take a while to upload.

Check that the toolkit can see them:

```bash
sbt project assets
```

## 8. Understand how environments are managed

A Conda **environment** is a separate collection of software. Keeping software
in separate environments prevents the requirements for one analysis from
breaking another.

SpatialBiologyToolkit uses environments in two ways:

- `sbt-cli` is the small environment you activate yourself. It runs the `sbt`
  command on the login node.
- Scientific environments contain the larger packages used by pipeline stages.
  For example, the `prep` stage uses an environment called
  `imc_segmentation`.

You do not need to install every scientific environment at the start. When you
run a stage, `sbt` checks which environment that stage needs.

- If the environment already exists, the job is submitted normally.
- If it is missing, `sbt` warns you and asks whether it should be installed.
- If you type `y`, only the missing environments needed for that run are
  installed.
- If you type `n` or just press **Enter**, the run stops and no job is
  submitted.

An environment can be shared by several stages, so you normally install each
one only once. The environments belong to your CSF3 account and can be reused
by your other SpatialBiologyToolkit projects. SLURM activates the correct
scientific environment inside the job; you should leave `sbt-cli` active at the
command line.

## 9. Start the first workflow

The first pipeline stage is `prep`. It unpacks the images from the MCD files
and creates the initial metadata and panel tables.

First preview the run:

```bash
sbt run prep --dry-run
```

A dry run checks the project, input files, configuration, and SLURM command. It
does not install an environment, create a run record, or submit a job.

If the preview succeeds, start the real run:

```bash
sbt run prep
```

The first time, you should see a warning similar to:

```text
Required Conda environments:
  - imc_segmentation (segmentation; repository-managed): MISSING for prep
Install the missing environment(s) now (imc_segmentation)? [y/N]:
```

Type `y` and press **Enter**. Installing the environment can take several
minutes. Once it has been installed and checked, `sbt` submits the `prep` job
to SLURM.

The job joins the CSF3 queue. It may start immediately, or it may wait while the
cluster is busy. You can disconnect from CSF3 after submission; the job will
continue running.

## 10. Monitor the analysis

Use these commands to see how the analysis is progressing:

```bash
sbt status latest
sbt logs latest
sbt summary
sbt report latest
```

- `status` asks SLURM whether the latest job is waiting, running, or finished.
- `logs` shows the latest messages from the job.
- `summary` lists the analyses recorded in this project.
- `report` displays the human-readable report for an analysis.

## 11. Find other stages and workflows

List the individual analysis stages:

```bash
sbt stages list
```

List the predefined multi-stage workflows:

```bash
sbt modes list
```

Use `--help` whenever you are unsure about a command. The complete
[`sbt` CLI guide](../pipeline/cli.md) describes project checks, planning,
submission, logs, and reports in more detail.

## 12. Optional graphical project console

The Project Console is useful for switching among registered IMC projects,
explaining stages and config fields, editing config with validation and backups,
reviewing configured assets and asset-aware readiness, and reading recorded
runs, reports, log tails, and notes. It cannot submit or control jobs and does
not load scientific data.

Install its separate lightweight Qt environment once:

```bash
cd "$HOME/imcanalysis"
bash install/bootstrap_sbt_gui.sh
```

Register projects once from any shell. The registry coexists with existing
credentials in `~/.imc_config`:

```bash
sbt project register --project "$HOME/scratch/HyperionProject" --name "Hyperion" --default
sbt project list
```

From the CSF3 login node, request a short X11-enabled interactive session and
launch the cockpit:

```bash
srun-x11 -p interactive -t 30
conda activate sbt-gui
sbt gui project
```

Use `--read-only` when config, notes, and registry writes should be disabled.
See the [Project Console guide](../guides/project_console.md) for portfolio
switching, recovery mode, configuration audits, and the exact capability
boundary.

## 13. Optional NapariSBT interactive classification

NapariSBT requires a separate scientific GUI environment and must run on a
compute node. Install it once:

```bash
cd "$HOME/imcanalysis"
bash install/bootstrap_napari_sbt_csf3.sh
```

Then request an X11 session, preflight the project, and launch:

```bash
srun-x11 -p interactive -t 60 -c 4
conda activate sbt-cli
sbt gui napari --check --project "$HOME/scratch/HyperionProject"
sbt gui napari --project "$HOME/scratch/HyperionProject"
```

Use the GUI for exploration, annotation, training, scoring, and small feature
trials. Submit full feature extraction separately with `sbt run cellfeat`.
The complete [CSF3 NapariSBT guide](../guides/napari_sbt_csf3.md) explains
resource profiles, state preservation, and X11/OpenGL troubleshooting.

## Updating SpatialBiologyToolkit

From a clean toolkit checkout:

```bash
cd "$HOME/imcanalysis"
git status --short
git pull --ff-only
bash install/bootstrap_sbt.sh
conda activate sbt-cli
```

Do not reinstall every scientific environment after an update. The next
`sbt run` checks the environment needed for that run and prompts if a new one is
required.

## Troubleshooting

### `conda: command not found`

Run `source "$HOME/.bashrc"` and try again. If it is still missing, log out of
CSF3 and reconnect. If that does not help, repeat the Miniconda installation.

### `sbt: command not found`

Activate the command environment:

```bash
conda activate sbt-cli
```

If `sbt-cli` does not exist, return to `~/imcanalysis` and rerun
`bash install/bootstrap_sbt.sh`.

### The environment prompt does not appear

This normally means the required environment is already installed. The run
will continue to SLURM without asking again.

### The environment cannot be installed

Update the toolkit using the commands above and try again. If the same error
remains, keep the complete error message and contact the toolkit maintainer. Do
not try to repair it by installing arbitrary packages.

### The job is waiting for a long time

Run `sbt status latest`. A pending job is usually waiting for suitable CSF3
resources. `sbt logs latest` becomes more useful after the job starts.

## Optional email and AI credentials

Core processing does not require an OpenAI key. Set these only if a selected
workflow explicitly needs them:

```bash
export IMC_EMAIL="your.email@example.org"
export OPENAI_API_KEY="your-key"
```

Never put credentials in `config.yaml` or commit them to Git.
