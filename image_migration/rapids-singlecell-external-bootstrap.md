# External RAPIDS-singlecell bootstrap

## Purpose

`rapids_singlecell` is an intentional exception to the repository-managed
`sbt-*` naming and lock convention. Its unchanged upstream name makes ownership
clear: the scverse recipe creates the scientific environment, while SBT only
adds a source overlay, verifies the runtime, and selects it for jobs after
acceptance.

SBT must not lock, recreate, repair, upgrade or remove this environment.

## Authoritative installation input

Use the immutable repository snapshot:

```text
image_migration/reference_specs/rsc_rapids_26.08_cuda13.official.yml
```

It records upstream scverse/rapids-singlecell commit
`eb8f5ae6f7cdf171a1014d9a40e0ed8c5a6b1b21` and creates the physical Conda
environment `rapids_singlecell`.

## Clean installation on CSF3

Run from the current imcanalysis checkout. Flexible channel priority is an
official RAPIDS requirement and is scoped to this command.

```bash
cd "$HOME/imcanalysis"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

CONDA_CHANNEL_PRIORITY=flexible \
conda env create \
    --file image_migration/reference_specs/rsc_rapids_26.08_cuda13.official.yml
```

Add only the lightweight modules required by SBT's configuration and command
bridge, then install the current checkout without dependency resolution:

```bash
conda run -n rapids_singlecell python -m pip install \
    "pyyaml>=6" \
    "pydantic>=2.4,<3" \
    "typer>=0.12,<1"

conda run -n rapids_singlecell python -m pip install \
    --editable "$HOME/imcanalysis" \
    --no-deps
```

Never omit `--no-deps`: SBT's complete package metadata includes scientific,
deep-learning and GUI packages that do not belong in this runtime.

## Initial verification

The lightweight CLI remains in `sbt-cli`; it invokes the tests inside the
external environment:

```bash
conda activate sbt-cli
sbt env list
sbt env validate-spec rapids
sbt env test rapids --format yaml
```

Expected registry behaviour:

```text
rapids    rapids_singlecell    external
```

`validate-spec` should report that the environment is externally managed and
has no repository lock contract. All registered tests, including `pip check`,
must pass before GPU testing or stage activation. A Pandas downgrade or other
official-recipe inconsistency is a failed acceptance result, not something SBT
should silently repair.

## Acceptance boundary

After the registered tests pass, run
`image_migration/smoke_tests/rapids_singlecell_2608_gpu_smoke.py` on a CSF3
A100. Only after direct cuGraph Leiden, the complete small RAPIDS-singlecell
workflow, and a representative managed run pass should the `rapids` and
CellVision clustering mappings move from `sbt-analysis` to this environment.
