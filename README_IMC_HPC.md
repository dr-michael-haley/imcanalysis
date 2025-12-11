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

---

## 🔐 3. Configuration File (`~/.imc_config`)

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
```

---

## 🔄 4. Updating IMC Analysis

When repository updates arrive:

```
cd ~/imcanalysis
git pull
make update
```

---

## 🗑️ 5. Uninstallation

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

## 📂 6. Directory Structure

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

## 🧪 7. Verify Installation

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

## 🧠 8. Troubleshooting

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
```

---

## ☑️ 9. Example Usage

```
cds mydataset
submit_imc_job mydata.slurm
```

---

## 🎉 10. Summary

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

