
# IMC Analysis – HPC Setup (Scripted Pipeline)

This guide is for running the **scripted pipeline on an HPC cluster** (via SLURM job scripts).

An example HPC cluster is **CSF3 (University of Manchester)**, which can be accessed for free by University of Manchester staff/students and typically has all the compute resources you will need for this pipeline.

If you’re new to the command line / conda / notebooks, skim the beginner explainers first:
- [README_NEW_USERS.md](README_NEW_USERS.md)

If you want to run analyses locally (Jupyter, bespoke downstream work), use:
- [README_LOCAL.md](README_LOCAL.md)

---

## 1. Requirements

Before installing, ensure you have:

- Access to an HPC login node (e.g. a CSF3 account).

> [!TIP]
> University of Manchester users can request a free account on the CSF3 using this link - the help pages are also extremely useful and kept up to date by the Research IT Staff:
> [Request a CSF3 account](https://pages.github.com/)

- Anaconda/Miniconda available in your CSF3 account (if it’s not already installed)
	- Otherwise, install Miniconda/Anaconda into your home directory (see below).
	- Quick check (should print a version):

```
conda --version
```

### 1.1 Install Miniconda (macOS/Linux)

Follow the official instructions here:
https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation:to-download-a-different-version

Quick CLI install (silent) if you prefer the terminal installer:

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3

# Initialize your shell
source "$HOME/miniconda3/bin/activate"
conda init --all
```

### 1.2 Install conda-lock in base

After installing Miniconda, install `conda-lock` in the base environment (required for environment management):

```bash
conda install --channel=conda-forge --name=base conda-lock
```

- A clone of this repository inside your home directory:

```
git clone https://github.com/dr-michael-haley/imcanalysis.git
cd ~/imcanalysis
```

If the repository is private, use the clone URL provided by the maintainer (or configure SSH access) so you are not prompted for credentials repeatedly.

---

## 2. Install the helper commands (HPC)

Run this once on a login node:

```
make install
```

This will:

- Add `~/imcanalysis/Bash_scripts` to your PATH  
- Install convenience aliases (e.g., `cds`)  
- Create a secure config file (`~/.imc_config`)  
- Load config automatically in `.bashrc` and `.profile`  
- Make scripts executable  

> Note: `make install` does **not** create the conda environments. Use `make envs` for that.

---

## 3. Create the pipeline conda environments

Set up the pipeline environments from pinned lockfiles:

```
make envs
```

This will:

- Create (or skip) the pipeline envs from lockfiles
- Record environment names in `~/.imc_config` for the SLURM jobs to use
- Ensure SLURM job scripts are executable
- Install `SpatialBiologyToolkit` into each env (editable, no dependencies)

The following variables are written to `~/.imc_config` (defaults shown):

```
export IMC_ENV_SEGMENTATION="imc_segmentation"
export IMC_ENV_DENOISE="imc_denoise"
export IMC_ENV_CELLPOSESAM="imc_cellposesam"
export IMC_ENV_BIOBATCHNET="imc_biobatchnet"
export IMC_ENV_SCPORTRAIT="scPortrait"
```

---

## 4. Configuration file: ~/.imc_config

Generated during installation. Stores:

- SLURM notification email  
- OpenAI API key (optional)  

Permissions are restricted:

```
chmod 600 ~/.imc_config
```

Example:

```
export IMC_EMAIL="your.email@domain.com"
export OPENAI_API_KEY="sk-..."
export IMC_ENV_SEGMENTATION="imc_segmentation"
export IMC_ENV_DENOISE="imc_denoise"
```

Tip: you can edit the `IMC_ENV_*` values later if you want the SLURM jobs to use different environment names.

---

## 5. Updating the repo

When repository updates arrive:

```
cd ~/imcanalysis
git pull
make update
```

If you change environment lockfiles or want to re-apply permissions, re-run:

```bash
make envs
```

---

## 6. Uninstallation

Clean removal:

```
make uninstall
```

Removes:

- PATH entries  
- Aliases  
- Config sourcing lines  
- (Optionally) removes `~/.imc_config`  

---

## 7. Directory Structure

```
imcanalysis/
├── Bash_scripts/
├── SLURM_scripts/
├── install/
│   ├── setup.sh
│   ├── uninstall.sh
│   └── common.sh
├── Makefile
└── README.md
```

---

## 8. Quick verification

Reload environment:

```
source ~/.profile
source ~/.bashrc
```

Check PATH:

```
which cds
```

Expected:

```
/home/<user>/imcanalysis/Bash_scripts/cds
```

Check config:

```
echo $IMC_EMAIL
echo $OPENAI_API_KEY
```

List pipeline stages:

```bash
pl --list
```

---

## 9. Troubleshooting

### 9.1 `make: command not found`
Load environment modules:

```
module load tools
module load make
```

### 9.2 PATH / aliases not updating

```bash
source ~/.bashrc
source ~/.profile
```

### 9.3 Permission denied running job scripts

If you see `Permission denied` when running `pll` locally, your SLURM job scripts are not executable.

Fix it with either:

```bash
make envs
```

or:

```bash
chmod +x ~/imcanalysis/SLURM_scripts/*.txt
```

---

## 10. Example usage

```
cds mydataset
pl config preprocess denoise dnqc nimbus bbn aiinter vis reint
```

You can run a single stage locally (useful for debugging):

```bash
pll config
```

---

## 11. Summary

The `make install` system provides:

- Reproducible HPC setup  
- Automatic configuration management  
- Clean uninstall  
- Team-friendly workflows  
- No manual PATH editing  

We can extend the system with:

- Cluster modulefiles  
- Singularity containers  
- Auto-activated conda envs  

