# Installation helper scripts

This page documents the shell scripts and Make targets in `install/`. New HPC
users should follow [HPC setup](../getting_started/hpc.md), which needs only the
`bootstrap_sbt.sh` helper. The remaining scripts maintain the older shell
conveniences and are not required for the project-aware `sbt` workflow.

## `bootstrap_sbt.sh`

This is the recommended launcher bootstrap:

```bash
cd "$HOME/imcanalysis"
bash install/bootstrap_sbt.sh
```

It:

- requires an existing `conda` command;
- uses the environment name `sbt-cli`, or `SBT_LAUNCHER_ENV` when explicitly set;
- creates that environment from `Local_envs/sbt_cli_env.yml` when absent;
- installs the repository in editable mode with `--no-deps`;
- leaves scientific environment management to `sbt env`.

It does not modify shell startup files, install `conda-lock`, create scientific
environments, or submit jobs. It is safe to rerun to refresh the editable
launcher installation. Scientific environments are checked and offered on
demand by a real `sbt run`.

## `setup.sh` and `make install`

`make install` runs `install/setup.sh`. This optional compatibility installer
expects the checkout at exactly `~/imcanalysis` and:

- makes files in `Bash_scripts/` and `SLURM_scripts/` executable;
- adds `~/imcanalysis/Bash_scripts` to `PATH` in `~/.profile`;
- adds the `cds` alias to `~/.bashrc`;
- creates `~/.imc_config` when absent, prompting for `IMC_EMAIL` and an optional
  `OPENAI_API_KEY`, then sets mode `600`;
- makes `~/.profile` and `~/.bashrc` source that config;
- makes `~/.bash_profile` source both startup files.

The script preserves an existing `~/.imc_config` and avoids adding exact
duplicate lines. It does not install the `sbt` launcher or create scientific
Conda environments. Its main purpose is to expose the legacy `cds`, `pl`,
`pll`, and `pls` helpers.

Because the script edits several shell startup files and prompts for sensitive
values, inspect it before use. Credentials must never be stored in the Git
checkout or committed.

## `setup_envs.sh` and `make envs`

`install/setup_envs.sh` is a deprecated compatibility wrapper. It requires
`sbt` on `PATH` and forwards its arguments to:

```bash
sbt env sync --all
```

`make envs` runs the same command directly. The canonical environment registry,
lockfiles, synchronization logic, safeguards, and smoke tests live behind
`sbt env`; the helper has no independent environment list or installation
algorithm. This intentionally installs every repository-managed environment
and is retained for legacy or administrative use; beginners should let
`sbt run` offer only the environments needed by their selected workflow.

## `make update`

`make update` runs `git pull` and then the optional `setup.sh` shell installer.
For the current `sbt` workflow, the more explicit update sequence in
[HPC setup](../getting_started/hpc.md) is preferred because it checks the Git
working tree, uses a fast-forward-only pull, refreshes the launcher, and previews
environment changes.

## `uninstall.sh` and `make uninstall`

`make uninstall` runs `install/uninstall.sh`. It removes the installer's
`Bash_scripts` `PATH` line, `cds` alias, and `.imc_config` sourcing lines from
`~/.profile` and `~/.bashrc`. It asks before deleting `~/.imc_config`.

It does not remove:

- the repository;
- `sbt-cli` or scientific Conda environments;
- dataset projects, reports, or `.sbt/` run records;
- the lines added to `~/.bash_profile` that source `~/.profile` and `~/.bashrc`.

Review the remaining shell configuration manually if a completely clean removal
is required.

## `common.sh`

`install/common.sh` contains the small line-addition and line-removal functions
shared by `setup.sh` and `uninstall.sh`. It is not intended to be run directly.
