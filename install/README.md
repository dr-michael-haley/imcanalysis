# install scripts

Helper scripts behind `make install` for HPC setups.

- `setup.sh`: verifies the repo lives at `~/imcanalysis`, makes `Bash_scripts`/`SLURM_scripts` executable, adds `Bash_scripts` to PATH, adds the `cds` alias, creates `~/.imc_config` (email and optional OPENAI key), and ensures your shell sources it.
- `setup_envs.sh`: installs the conda environments from lockfiles, writes the environment names to `~/.imc_config`, and re-applies executable permissions to the SLURM job scripts (useful if permissions were lost after a pull).
- `uninstall.sh`: removes the PATH/alias/config entries and optionally deletes `~/.imc_config`.
- `common.sh`: shared helper functions.

Usage:
1. From the repo root run `make install` on a login node.
2. Reload your shell (`source ~/.profile && source ~/.bashrc`) or log back in.
3. Use `make uninstall` to undo the setup.

