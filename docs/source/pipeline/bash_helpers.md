# Bash helper scripts

Shortcuts installed via `make install` (adds this directory to PATH and creates `~/.imc_config`).

- `pl`: submit a sequence of SLURM stages defined in `SLURM_scripts/pipeline.conf`; supports `--name`, `--dry-run`, and `--list` for metadata.
- `pll`: run one stage locally using the same alias resolution (handy for debugging job scripts).
- `pls`: show job IDs/status for the most recent `pl` submission (uses `sacct`/`squeue`).
- `zipqc`: zip up QC outputs using predefined folder sets.
- `cds`: jump to a scratch subfolder by prefix.
- `intnode`: request an interactive SLURM node.
- `git-all`: pull every git repo under a directory (default `$HOME`).

