# Migrating from Make/Bash environment setup

Environment arrays, lock installation, pip extras, and editable toolkit setup
formerly lived in `install/setup_envs.sh`. Those operations are now implemented
once in the Python `sbt env` subsystem.

## New workflow

Bootstrap the lightweight launcher without requiring an existing `sbt` command:

```bash
bash install/bootstrap_sbt.sh
conda activate sbt-cli
sbt env doctor
sbt env sync --all --dry-run
sbt env sync --all
```

Those `--all` commands reproduce the old full-environment installer and are
intended for migration or administration. New users do not need them: a real
`sbt run` checks the selected workflow and offers only its missing managed
environments.

`make envs` is retained as a convenience wrapper for `sbt env sync --all`.
`install/setup_envs.sh` remains as a deprecated compatibility wrapper and no
longer contains an independent environment array or installation algorithm.

The launcher environment needs only Python, Pydantic, PyYAML, Typer, pip, and
the editable toolkit installed with `--no-deps`. `conda-lock` remains installed
once in Conda base and is invoked by `sbt` through `conda run -n base`.
Scientific dependencies remain inside their fixed stage environments.

## SLURM compatibility

Managed `sbt run` submissions now export the logical environment key and fixed
name as `SBT_ENVIRONMENT_KEY`, `SBT_ENVIRONMENT_KEYS`, `SBT_CONDA_ENV`, and
key-specific `SBT_CONDA_ENV_<KEY>` variables. Active wrappers prefer those
values.

Direct wrapper execution remains transitional: wrappers fall back to existing
`IMC_ENV_*` overrides and then the historical fixed name. Existing
`~/.imc_config` files therefore continue to work, but new environment-name
configuration belongs in `HPC_env_files/environments.yaml`.

The legacy pipeline commands `pl`, `pll`, and `pls` are unchanged compatibility
interfaces. Use `sbt` for new planning, submission, environment, status, and
reporting work.

## Existing lockfiles

Pip requirements formerly nested in some `environment.yml` files have moved to
`pip-extras.txt` without changing their requested versions. Existing generated
locks may still contain legacy pip records. On the Linux HPC, review and run:

```bash
sbt env validate-spec --all
sbt env lock --all --check
sbt env lock --all
```

No lockfile is regenerated silently during migration, capture, or comparison.
The RAPIDS, STARLING, and scPortrait environments remain explicitly external
until curated specifications are added.
