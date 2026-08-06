# Scientific runtime consolidation tracker

This directory contains the working records for consolidating scientific
environments before the OCI/Apptainer image migration. Canonical environment
definitions remain under `HPC_env_files/` because that is where `sbt env`
loads them.

## Target catalogue

The candidate `sbt-analysis` environment is intended to replace these four
physical Conda environments after validation:

| Existing environment key | Existing Conda name | Stage mappings currently using it | Candidate destination |
|---|---|---|---|
| `segmentation` | `imc_segmentation` | `prep`, `vis`, `nimbus`, `subcl`, `dnqc` (second runtime), `aiinter`, `config`, `cellpose` (second runtime), `reint`, `remap`, `slogs`, `rebuildmeta`, `cellfeat`, `spatialdata` | `analysis` / `sbt-analysis` |
| `biobatchnet` | `imc_biobatchnet` | `bbn` | `analysis` / `sbt-analysis` |
| `cellcharter` | `imc_cellcharter` | `bint`, `cchar`, `pairsp`, `nxsp`, `popqc` | `analysis` / `sbt-analysis` |
| `starling` | `imc_starling` | `starling` | `analysis` / `sbt-analysis` |

The following runtimes are deliberately not part of this merger:

| Runtime | Reason |
|---|---|
| `imc_maxfuse` | Deferred; expected to retain a dedicated Python 3.8 environment/image. |
| `scPortrait` | Its SpatialData, AnnData, and Cellpose constraints conflict with the shared baseline. |
| `imc_cellposesam` | Requires Cellpose 4 while the shared runtime retains Cellpose 3. |
| `rapids_singlecell` | Dedicated RAPIDS/Python 3.13/CUDA runtime. |
| `hyperstac-imc` | Dedicated TensorFlow 2.15 runtime. |
| `imc_denoise` | Frozen TensorFlow 2.6/Python 3.8 legacy runtime. |
| `sbt-napari` | Interactive GUI runtime, not a pipeline compute environment. |

## Phase tracker

| Phase | Status | Codex work | Michael's work | Validation evidence | Checkpoint | Notes/blockers |
|---|---|---|---|---|---|---|
| Audit exported environments and identify compatibility families | complete | Compared registry, stage mappings, intent specifications, and latest exported snapshots. | Exported the live HPC environments. | Version matrix reviewed 2026-08-03. | Not committed | MaxFuse export absent and excluded by decision. |
| Define the joint `sbt-analysis` candidate | complete | Added the registry candidate, joint Conda/pip specifications, smoke tests, and this tracker. | Approved a full four-environment joint install. | 30 environment-management tests passed; repository validation passed with no warnings; candidate is registered with no active stage mappings. | Not committed | Linux lock intentionally pending. `validate-spec` reports only that missing required lock plus a review warning for the deliberately pinned BioBatchNet VCS requirement. |
| Generate reviewed Linux lock and perform joint install | waiting for user | Provide a small exact HPC command batch after local validation. Review the solve/install output. | Generate the linux-64 lock and create `sbt-analysis` on CSF3. | Pending. | Pending | Must not modify the four existing environments. |
| Run registered imports and GPU smoke tests | not started | Interpret smoke-test and CUDA results; adjust only the candidate if necessary. | Run the supplied test commands in an appropriate CSF3 job. | Pending. | Pending | Login-node imports do not establish GPU functionality. |
| Run scientific parity across all migrated stage families | not started | Define comparisons and assess outputs/warnings with Michael. | Run old and candidate environments on representative small data and approve parity. | Pending. | Pending | Required before any permanent stage remapping. |
| Remap validated stages to `analysis` | not started | Update the registry/wrappers as one coherent approved phase and run control-plane tests. | Confirm deployment and run selected managed workflows. | Pending. | Pending | Existing environment definitions remain as rollback. |
| Retire superseded Conda environments | not started | Mark legacy definitions deprecated only after approval. | Approve and perform any HPC environment removal. | Pending. | Pending | No removal is authorized yet. |

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-06 | Store migration trackers and working artifacts under `image_migration/`. | Keeps temporary engineering state out of canonical user documentation. |
| 2026-08-06 | Test one full joint merger rather than adding packages one family at a time. | A single resolver transaction tests the intended future combined build and avoids repeated large installs. |
| 2026-08-06 | Merge segmentation, BioBatchNet, CellCharter, and STARLING into candidate `sbt-analysis`. | They can plausibly share Python 3.11, NumPy 1.26.4, and Torch 2.9.1/CUDA 12.8. |
| 2026-08-06 | Pin CellCharter to `0.3.7`. | This is the requested current release and requires Python 3.11 or newer. |
| 2026-08-06 | Pin BioBatchNet to commit `b1d708c62f5bac70f323a36aa28c6057f63e8222`. | This is the precise VCS revision captured from the working HPC environment. |
| 2026-08-06 | Exclude MaxFuse from the merger. | It will be considered as a dedicated legacy runtime/image later. |
| 2026-08-06 | Keep all existing stage mappings unchanged until parity passes. | The four current environments remain the immediate rollback path. |

## Open validation points

- Generate the candidate's real `linux-64` lock on CSF3; a Windows solve would
  not validate the deployment platform.
- Confirm that the lock-based installation propagates the
  `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL` setting needed by the
  existing CellCharter/scArches dependency path. If it does not, adjust the
  candidate installation mechanism before changing any active environment.
- Confirm CUDA visibility and representative scientific outputs on CSF3; local
  import tests cannot establish either.

## Candidate baseline

| Component | Candidate choice | Basis |
|---|---|---|
| Python | 3.11 | Required by CellCharter 0.3.7 and SpatialData 0.7; supported by all four families. |
| NumPy | 1.26.4 | Matches working segmentation/STARLING and satisfies BioSTARLING's `<2` constraint. |
| PyTorch | 2.9.1 | Matches working segmentation; satisfies BioBatchNet, CellCharter, and BioSTARLING constraints. |
| CUDA user-space family | 12.8 via PyTorch wheels | Matches the working segmentation, CellCharter, and STARLING CUDA family. |
| SpatialData | `>=0.7,<0.8` | Preserves the existing segmentation specification and uses its required Zarr 3 generation. |
| CellCharter | 0.3.7 | Explicit migration decision. |
| BioBatchNet | pinned VCS commit | Preserves the exported known revision. |
| BioSTARLING | 0.1.4 | Preserves the working STARLING release. |

## Rollback policy

The candidate has a new Conda name and no stages point to it. A failed solve,
install, smoke test, or parity comparison therefore requires only adjustment or
removal of `sbt-analysis`; the existing environments and mappings remain intact.
