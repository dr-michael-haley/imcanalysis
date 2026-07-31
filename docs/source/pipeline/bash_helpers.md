# Bash helper scripts

Optional compatibility shortcuts installed by `make install`. The exact shell
and config-file changes are listed in the
[installation helper scripts reference](../reference/installation_helpers.md).

- `pl`: legacy submission of a sequence of SLURM stages defined in `SLURM_scripts/pipeline.conf`; supports `--name`, `--dry-run`, and `--list` for metadata.
- `pll`: run one stage locally using the same alias resolution (handy for debugging job scripts).
- `pls`: show job IDs/status for the most recent `pl` submission (uses `sacct`/`squeue`).
- `zipqc`: zip up QC outputs using predefined folder sets.
- `cds`: jump to a scratch subfolder by prefix.
- `intnode`: request an interactive SLURM node.
- `git-all`: pull every git repo under a directory (default `$HOME`).

These commands are not part of the beginner installation. For new work, use
the [project-aware `sbt` CLI](cli.md) for validation,
planning, submission, run records, status, and logs. These Bash commands remain
available while compatibility and downstream usage are assessed.

