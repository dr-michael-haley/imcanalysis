# install scripts

Helper scripts behind `make install` for HPC setups.

- `setup.sh`: verifies the repo lives at `~/imcanalysis`, makes `Bash_scripts`/`SLURM_scripts` executable, adds `Bash_scripts` to PATH, adds the `cds` alias, creates `~/.imc_config` (email and optional OPENAI key), and ensures your shell sources it.
- `bootstrap_sbt.sh`: creates/locates the lightweight launcher environment and installs the editable toolkit with `--no-deps` so `sbt` can manage the scientific environments.
- `setup_envs.sh`: deprecated compatibility wrapper for `sbt env sync --all`; it no longer contains a separate environment installer.
- `uninstall.sh`: removes the PATH/alias/config entries and optionally deletes `~/.imc_config`.
- `common.sh`: shared helper functions.

Usage:
1. From the repo root run `make install` on a login node.
2. Reload your shell (`source ~/.profile && source ~/.bashrc`) or log back in.
3. Use `make uninstall` to undo the setup.
