# IMC Analysis – HPC Installation Guide

This document describes how to install, configure, and use the **IMC Analysis Toolkit** on an **HPC cluster environment**.  
The installation system uses a robust `make install` workflow that ensures consistent setup across users and sessions, without manually editing shell configuration files.

---

## 🚀 1. Requirements

Before installing, ensure you have:

- Access to an HPC login node  
- A bash-compatible shell (`bash` or `zsh`)  
- A clone of this repository inside your home directory:

```
git clone <repo-url> ~/imcanalysis
cd ~/imcanalysis
```

---

## ⚙️ 2. Installation (Recommended for HPC Users)

To install IMC analysis tools:

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

## 🐍 3. Conda Environments (Pipeline)

Set up the pipeline environments from pinned lockfiles:

```
make envs
```

This will:

- Create (or skip) the pipeline envs from lockfiles
- Record environment names in `~/.imc_config` for the SLURM jobs to use
- Ensure SLURM job scripts are executable

The following variables are written to `~/.imc_config` (defaults shown):

```
export IMC_ENV_SEGMENTATION="imc_segmentation"
export IMC_ENV_DENOISE="imc_denoise"
export IMC_ENV_CELLPOSESAM="imc_cellposesam"
export IMC_ENV_BIOBATCHNET="imc_biobatchnet"
export IMC_ENV_SCPORTRAIT="scPortrait"
```

---

## 🔐 4. Configuration File (`~/.imc_config`)

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

---

## 🔄 5. Updating IMC Analysis

When repository updates arrive:

```
cd ~/imcanalysis
git pull
make update
```

---

## 🗑️ 6. Uninstallation

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

## 📂 7. Directory Structure

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

## 🧪 8. Verify Installation

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

---

## 🧠 9. Troubleshooting

### `make: command not found`
Load environment modules:

```
module load tools
module load make
```

### PATH / aliases not updating

```
source ~/.bashrc
source ~/.profile

### Permission denied running job scripts
If you see `Permission denied` when running `pll` locally, ensure execute bits are set:

```
chmod +x ~/imcanalysis/SLURM_scripts/*.txt
```
Running `make envs` will also fix permissions.
```

---

## ☑️ 10. Example Usage

```
cds mydataset
submit_imc_job mydata.slurm
```

---

## 🎉 11. Summary

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

