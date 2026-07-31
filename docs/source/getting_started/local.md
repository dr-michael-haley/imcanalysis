# Local analysis setup

Use this setup for Jupyter notebooks, Napari, exploratory analysis, and bespoke
figures on a Windows or macOS workstation. The reproducible end-to-end pipeline
is designed for Linux HPC with SLURM. University of Manchester beginners should
start with [CSF3 setup](hpc.md); experienced HPC users can use the
[`sbt` CLI guide](../pipeline/cli.md).

## What you will set up

- a local Conda environment named `sbt`;
- an editable installation of `SpatialBiologyToolkit`;
- Jupyter Lab running from your own analysis directory.

The Windows and macOS environments are intentionally different. Use only the
file for your platform.

## 1. Install Conda and Git

Install Miniconda or Anaconda for your operating system using the official
[Conda installation guidance](https://www.anaconda.com/docs/getting-started/miniconda/install).
On Windows, run the commands below in **Anaconda Prompt**. Confirm that both
tools are available:

```bash
conda --version
git --version
```

## 2. Clone the repository

Choose a directory for software checkouts, then run:

```bash
git clone https://github.com/dr-michael-haley/imcanalysis.git
cd imcanalysis
```

Downloading a GitHub ZIP can work for a one-off read, but a Git clone is
recommended because it can be updated cleanly.

## 3. Create the local environment

On Windows:

```bash
conda env create -f Local_envs/sbt_env.yml
```

On Apple Silicon macOS:

```bash
conda env create -f Local_envs/sbt_env_macos.yml
```

Then activate the environment on either platform:

```bash
conda activate sbt
```

`Local_envs/sbt_env.yml` is a detailed Windows environment export. The portable
macOS specification is tested on Apple Silicon with macOS 12 or later. It uses
a matched Napari, PySide6, TensorFlow for macOS, and TensorFlow Metal
combination, and intentionally omits CUDA, Windows runtime packages, R,
Java/Bio-Formats, OpenCL, optional SpatialData viewer integrations, and several
less-portable extras. It is not intended for Intel Macs.

Environment solving and installation can take some time. If Conda reports a
conflict, retain the full error rather than installing arbitrary replacement
versions into the partly created environment.

## 4. Install the toolkit

The environment file already supplies the tested dependencies. From the
repository root, install only the editable toolkit link rather than asking pip
to resolve that stack again:

```bash
python -m pip install --no-deps -e .
```

Use the same command on Windows and macOS. `--no-deps` is important on macOS:
it prevents generic pip packages from replacing the matched Apple TensorFlow
and GUI packages or adding optional components deliberately omitted from the
portable environment.

Verify the package and command:

```bash
python -c "import SpatialBiologyToolkit; print(SpatialBiologyToolkit.__file__)"
sbt --help
```

On macOS, also confirm the matched GUI and numerical stack:

```bash
python -c "from qtpy import API_NAME; import tensorflow as tf, umap; print(API_NAME, tf.__version__)"
```

## 5. Start Jupyter Lab

The macOS environment includes Jupyter Lab. If `jupyter lab --version` is not
available in the Windows environment, install it into the active `sbt`
environment:

```bash
conda install --channel conda-forge jupyterlab
```

Create a separate directory for your work and start Jupyter there:

```bash
cd path/to/my_analysis_directory
jupyter lab
```

Do not edit your own analysis directly inside the repository's `Tutorials/`
directory. Copy any tutorial you want to adapt into your analysis directory;
this keeps Git updates separate from your results and notebook changes. The
[tutorial index](../tutorials/index.md) lists the maintained notebooks.

## 6. Update the local installation

Editable installation means ordinary source changes are picked up from the
checkout. Update a clean checkout with:

```bash
cd path/to/imcanalysis
git status --short
git pull --ff-only
conda activate sbt
```

If packaging metadata or command entry points changed, repeat the editable
installation from step 4. If the environment specification changed, recreate
or deliberately update the environment rather than mixing unreviewed package
versions into it.

## Troubleshooting

### `conda` is not found

On Windows, use Anaconda Prompt. On macOS, restart the terminal after installing
and initializing Conda.

### Jupyter cannot import `SpatialBiologyToolkit`

Check that the notebook kernel belongs to the `sbt` environment. In a terminal,
activate `sbt`, repeat the editable installation, then restart the notebook
kernel.

### macOS reports `QtBindingsNotFoundError` or crashes while importing UMAP

For an environment created from an older `sbt_env_macos.yml`, repair the matched
packages, then restart the Jupyter kernel:

```bash
conda activate sbt
python -m pip uninstall -y tensorflow tensorflow-estimator
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.26,<2" "PySide6==6.9.1" "tensorflow-macos==2.16.2" "tensorflow-metal==1.2.0"
```

### Can I submit the pipeline locally?

The `sbt` command can perform lightweight inspection locally, but actual
pipeline submission requires the repository's Linux SLURM environment and
scientific stage environments. University of Manchester beginners can follow
[CSF3 setup](hpc.md); experienced HPC users can use the
[`sbt` CLI guide](../pipeline/cli.md).
