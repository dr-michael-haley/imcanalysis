# Fixed Conda environment management

The lightweight `sbt env` command group is the canonical interface for the
pipeline's fixed-name Conda environments. It supports repository-to-HPC
synchronisation, reviewed HPC-to-repository capture, drift comparison, smoke
testing, and permanent environment evidence in stage reports.

It does not import the scientific stack and does not require shell activation.
All target-environment Python and pip operations use `conda run -n`.
All lock generation and installation operations use the copy of `conda-lock`
installed in Conda base through `conda run -n base conda-lock`; the active
scientific environment and its `PATH` do not control lock availability.

You do not need every registered environment. A real `sbt run` checks only the
environments mapped to its resolved stages, after planning and before creating
a run record. It validates the installation specification and offers to install
missing repository-managed environments. It stops safely when a specification
is invalid or a required external environment is absent. Dry runs never install
environments.

## Environment registry

[`HPC_env_files/environments.yaml`](https://github.com/dr-michael-haley/imcanalysis/blob/main/HPC_env_files/environments.yaml)
is the source of truth for logical keys, fixed Conda names, specification
directories, stage mappings, target platforms, editable-toolkit policy, and
lightweight smoke tests.

| Key | Fixed Conda name | Management |
|---|---|---|
| `analysis` | `sbt-analysis` | Repository-managed standard scientific runtime |
| `napari` | `sbt-napari` | Explicit interactive bootstrap; Linux lock pending |
| `denoise` | `sbt-denoise` | Repository lock |
| `cellposesam` | `sbt-cellpose-sam` | Repository lock |
| `starling` | `sbt-starling` | External/pre-existing |
| `scportrait` | `sbt-scportrait` | External/pre-existing |
| `hyperstac` | `sbt-hyperstac` | External/pre-existing |
| `maxfuse` | `sbt-maxfuse` | External/pre-existing |

Commands accept either the logical key or fixed name. External environments
can be listed, shown, smoke-tested, and captured into observational
compatibility bundles. `sync`, `lock`, and capture with `--write` refuse them
until repository specifications are deliberately added.

`napari` is an interactive application environment and therefore has no stage
mapping. Install or refresh it explicitly with
`bash install/bootstrap_napari_sbt_csf3.sh`. Its committed intent specification
will become a repository-managed lock contract only after the Linux solve and
GUI smoke checks have been reviewed on CSF3.

The stage mapping is also centralized:

| Environment key | Pipeline stages |
|---|---|
| `analysis` | `prep`, `vis`, `nimbus`, `nimbus-scan`, `bint`, `rapids`, `cellvision-cluster`, `cellvision-full`, `bbn`, `subcl`, `cchar`, `dnqc`, `aiinter`, `config`, `cellpose`, `reint`, `pairsp`, `nxsp`, `remap`, `slogs`, `rebuildmeta`, `popqc`, `cellfeat`, `spatialdata`, `neighsig` |
| `denoise` | `denoise`, `dnqc` |
| `cellposesam` | `cellpose` |
| `starling` | `starling` |
| `scportrait` | `scport`, `cellvision-extract`, `cellvision-embed`, `cellvision-plot`, `cellvision-full` |
| `hyperstac` | `hyperstac-preprocess`, `hyperstac-model`, `hyperstac-permutation`, `hyperstac-visualise`, `cox`, `hyperstac-stability`, `hyperstac-full` |
| `maxfuse` | `maxfuse` |

`cellpose`, `dnqc`, and `cellvision-full` intentionally use two environments;
their primary environment is listed first in the registry mapping.

The superseded segmentation, BioBatchNet, CellCharter, and standalone RAPIDS
specifications are retained only under
`image_migration/archive/retired_hpc_environments/`. They are not active
registry entries and are not discovered as automatic fallbacks.

## Per-run environment selection

A registered environment can temporarily replace the default for one resolved,
single-environment stage:

```bash
sbt run hyperstac-model --environment hyperstac --dry-run
sbt run hyperstac-model --environment sbt-hyperstac
```

The override is execution state, not project configuration: it does not alter
`stage_environments` or affect later runs. SBT checks the selected environment,
exports its logical key and fixed name to the wrapper, snapshots its committed
specification, inspects its runtime, and identifies the override in the stage
report. The selector must be a registered logical key or its current fixed
name. It does not select retired names such as `hyperstac-imc`; managed
HyPERSTAC runs now resolve `hyperstac` to `sbt-hyperstac`.

The current interface deliberately rejects a plan containing several stages,
stages without an environment, and wrappers that switch between multiple
environments. Run candidate stages separately with `--dependency-policy none`
when their blocking assets already exist.

## Specification roles

Every repository-managed environment directory contains three distinct files:

- `environment.yml` is the curated direct Conda input and contains the fixed
  name, channels, intentional constraints, and `pip` when needed.
- `conda-linux-64.lock` is generated by conda-lock and is the exact HPC Conda
  deployment. Never edit it manually.
- `pip-extras.txt` contains intentional pinned pip/VCS packages installed after
  the lock. It never contains the toolkit editable installation.

`environment.snapshot.json` is observational evidence written by an explicit
capture; it is not an installation input.

Inspect and validate without changing anything:

```bash
sbt env list
sbt env list --compare
sbt env show analysis
sbt env doctor
sbt env validate-spec --all
```

## Repository to fixed HPC environment

The normal beginner path is to let a real run offer the environments it needs:

```bash
sbt run segmentation --dry-run
sbt run segmentation
```

The first command previews without inspecting Conda. The second validates the
needed installation specifications, lists missing managed environments, and
asks permission to install them when valid. Use the commands below for
deliberate manual maintenance of one environment.

Always inspect the plan first:

```bash
sbt env sync analysis --dry-run
```

`--all` remains available for administrators or deliberate full-stack
maintenance, but it is not part of initial setup.

For an absent environment, synchronisation performs:

1. `conda run -n base conda-lock install --name <fixed-name> <lockfile>`;
2. `conda run -n <fixed-name> python -m pip install -r pip-extras.txt`;
3. `conda run -n <fixed-name> python -m pip install -e <repo> --no-deps`;
4. registered login-node-safe smoke tests;
5. an observed snapshot in the user's SBT state directory.

If a fixed environment exists and differs, `sbt` does not remove it silently.
Review the drift and request recreation explicitly:

```bash
sbt env sync analysis --recreate
# non-interactive only after review:
sbt env sync analysis --recreate --yes
```

Before removal, the live environment is written beneath
`${SBT_STATE_HOME:-${XDG_STATE_HOME:-~/.local/state}/sbt}/environment_history/`.
The resolved fixed name is checked again after removal. Missing or invalid
locks stop synchronisation; there is no silent unlocked solve.

## Drift comparison

```bash
sbt env compare analysis
sbt env compare --all
sbt env compare analysis --format yaml
sbt env compare analysis --format json
```

Comparison keeps separate layers for:

- direct `environment.yml` constraints;
- exact lock versions and builds;
- declared and unexpected pip extras;
- editable/local/VCS packages;
- the SpatialBiologyToolkit overlay, checkout path, commit, and dirty state.

Exit codes are stable:

- `0`: comparison completed and matches;
- `1`: material drift was detected;
- `2`: comparison could not be completed, including a missing environment or
  invalid specification.

A dirty toolkit checkout is reported separately as provenance and is not by
itself ordinary package drift.

## Lock maintenance

```bash
sbt env lock analysis
sbt env lock --all
sbt env lock analysis --check
```

Lock generation uses the `conda-lock` installation in Conda base and a
temporary destination. The committed lock is replaced atomically only after
conda-lock succeeds and creates a non-empty file.
`--check` compares a temporary generated lock and never replaces the repository
file. Failure preserves the previous lock.

The lockfiles that pre-date pip-extra separation can contain legacy pip records.
`sbt env validate-spec` rejects these for synchronization; regenerate them on
the Linux HPC with `sbt env lock --all` before using the cleaned separation.

## Reviewed HPC-to-repository capture

Capture is deliberately explicit because an installed environment cannot
perfectly reconstruct the original dependency intent:

```bash
sbt env capture analysis --dry-run
sbt env capture analysis --write
```

External environments use the same command without `--write`:

```bash
sbt env capture starling --dry-run --verbose
sbt env capture scportrait --dry-run --verbose
```

To capture every environment reported by the active Conda installation in one
observational pass, use:

```bash
sbt env capture --all --dry-run --verbose --accept-vcs
```

The standard spelling is `--all` with two leading hyphens. SBT uses
`conda env list --json` and captures every distinct prefix it reports,
including the base environment, prefix-based environments, and environments
absent from `HPC_env_files/environments.yaml`. Registered environments retain their SBT
identity; unregistered environments are labelled `conda:<name>` and remain
observational for manual curation. Same-named prefixes receive stable hash
suffixes so neither is silently discarded.

Each environment receives its own timestamped bundle. SBT targets exact Conda
prefixes during batch capture and records the prefix in both the snapshot and
capture plan. An environment without a working Python or pip installation can
still retain its Conda inventory, with the unavailable inspection steps called
out for manual review. Other per-environment failures are reported after the
remaining captures have been attempted; the command exits with status 2 when
the batch is incomplete. For safety, `--all` is observational only and cannot
be combined with `--write`.

This produces an observational compatibility bundle under the SBT user state
directory without changing the repository. The bundle contains a normalized
from-history `environment.yml`, separated `pip-extras.txt`, the exact Conda and
pip inventory in `environment.snapshot.json`, a self-describing
`capture-plan.json`, and a candidate target-platform lock when lock generation
succeeds. A lock-solver failure during observational capture is recorded but
does not discard the other environment evidence. VCS, editable, and local
requirements remain explicitly flagged for review; use `--accept-vcs` only
when retaining the observed VCS reference is intentional.

Capture with `--write` remains unavailable for an external environment. After
reviewing compatibility, add its repository specification deliberately and
mark it managed before using the normal write, lock, or sync workflows.

The command uses Conda's supported from-history export, normalises the fixed
name, removes machine prefixes and nested pip entries, and retains the full
installed package inventory in an observed snapshot. Pip packages not managed
by Conda are pinned and sorted separately.

SpatialBiologyToolkit is excluded from candidates and reported as its own
editable overlay. Other editable, local-path, or VCS requirements are retained
in the snapshot and marked for manual review. `--write` refuses unresolved
review requirements rather than discarding or blindly locking them.
An intentional observed VCS requirement can be retained only with the explicit
`--accept-vcs` option; local-path and non-toolkit editable requirements still
require manual candidate editing and review.

Candidate files and a generated candidate lock are stored under the SBT user
state capture directory. Repository files are written only after lock
generation succeeds. A conda-lock failure retains the candidate directory and
leaves existing repository files unchanged.

## Smoke tests and diagnostics

```bash
sbt env test analysis
sbt env test --all
sbt env doctor
```

Smoke tests execute the registry's short import probes through `conda run` and
record command, return code, output tails, and duration. They do not run GPU
workloads, datasets, or complete pipeline stages. `doctor` checks Conda,
the base-environment conda-lock installation, registry/spec paths, stage
mappings, user-state write access, and
stale installer mappings without changing anything.

## Environment evidence in every stage report

For managed stage execution, submission copies (not symlinks) the relevant
repository specifications into:

```text
outputs/003_Segmentation/environment/
  environment.yml
  conda-linux-64.lock
  pip-extras.txt
  installed.snapshot.json
  environment_manifest.yaml
```

The small specification files receive SHA-256 hashes. At stage start, the
reporting hook records Conda and pip inventories, editable packages, Python,
prefix, toolkit path/commit/dirty state, SLURM ID, visible execution ID, and
technical execution ID. Multi-environment wrappers retain additional records
under `environment/additional/<key>/`.

`stage_manifest.yaml` references the primary environment record, and the stage
README provides a concise software-environment section rather than embedding
the full package inventory.

Fixed names are convenience identifiers. Historical reproducibility comes from
the copied specification, observed runtime snapshot, toolkit commit, and run
identity stored with each execution.

See [Migrating from Make/Bash environment setup](environment-migration.md) for
the compatibility and bootstrap path.
