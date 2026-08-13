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
| `segmentation` | `imc_segmentation` | `prep`, `vis`, `nimbus`, `subcl`, `dnqc` (second runtime), `aiinter`, `config`, `cellpose` (second runtime), `reint`, `remap`, `slogs`, `rebuildmeta`, `cellfeat`, `spatialdata`, `neighsig` | `analysis` / `sbt-analysis` |
| `biobatchnet` | `imc_biobatchnet` | `bbn` | `analysis` / `sbt-analysis` |
| `cellcharter` | `imc_cellcharter` | `bint`, `cchar`, `pairsp`, `nxsp`, `popqc` | `analysis` / `sbt-analysis` |
| `rapids` | `rapids_singlecell` | `rapids`, `cellvision-cluster`, `cellvision-full` (middle runtime) | `analysis` / `sbt-analysis` |

The following runtimes are deliberately not part of this merger:

| Runtime | Reason |
|---|---|
| `imc_maxfuse` | Deferred; expected to retain a dedicated Python 3.8 environment/image. |
| `scPortrait` | Its SpatialData, AnnData, and Cellpose constraints conflict with the shared baseline. |
| `imc_cellposesam` | Requires Cellpose 4 while the shared runtime retains Cellpose 3. |
| `imc_starling` | Removed from this merger when RAPIDS became the higher-priority clustering runtime; retain separately. |
| `hyperstac-imc` | Dedicated TensorFlow 2.15 runtime. |
| `imc_denoise` | Frozen TensorFlow 2.6/Python 3.8 legacy runtime. |
| `sbt-napari` | Interactive GUI runtime, not a pipeline compute environment. |

## Phase tracker

| Phase | Status | Codex work | Michael's work | Validation evidence | Checkpoint | Notes/blockers |
|---|---|---|---|---|---|---|
| Audit exported environments and identify compatibility families | complete | Compared registry, stage mappings, intent specifications, and latest exported snapshots. | Exported the live HPC environments. | Version matrix reviewed 2026-08-03. | Not committed | MaxFuse export absent and excluded by decision. |
| Define the original `sbt-analysis` candidate | complete | Added the registry candidate, initial Conda/pip specifications, smoke tests, and this tracker. | Approved a full joint-install experiment. | 30 environment-management tests passed; repository validation passed with no warnings; candidate remained inactive. | Not committed | The first CSF3 pip solve exposed the NumPy/Zarr generation conflict. |
| Revise `sbt-analysis` around RAPIDS and SpatialData | complete | Replaced STARLING with RAPIDS, encoded flexible channel priority, moved to Python 3.12, retained CellCharter 0.3.7 and the pinned SpatialData 0.4/Zarr 2 spine, and expanded smoke tests. | Approved the maximal merger and the policy of splitting Nimbus only if testing proves necessary. | 31 environment-management tests passed; repository validation passed with no warnings; `git diff --check` passed. Static specification validation reports only the intentionally missing new Linux lock and the expected BioBatchNet VCS-review warning. | Not committed | No production stage mappings changed. |
| Generate a new reviewed Linux lock and perform a clean joint install | in progress | Reviewed the RAPIDS lock and first clean installation; diagnosed the shared Zarr/Numcodecs smoke-test failure and pinned the compatible boundary. | Generated the RAPIDS lock and completed a clean `sbt-analysis` installation on CSF3. Regenerate the corrected lock and resynchronize the candidate. | Conda, pip extras and the editable overlay installed. Python 3.12 and all exact-version assertions passed. Imports failed uniformly because Zarr 2.18.3 loaded Numcodecs 0.16, where its expected public Blosc helpers were removed. | Pending | Zarr 2.18.7 explicitly requires Numcodecs <0.16; the corrected specification pins Numcodecs 0.15.1. No independent RAPIDS, Nimbus, BioBatchNet or CellCharter failure has yet been observed. |
| Run registered imports and GPU smoke tests | not started | Interpret smoke-test and CUDA results; adjust only the candidate if necessary. | Run the supplied test commands in an appropriate CSF3 job. | Pending. | Pending | Login-node imports do not establish GPU functionality. |
| Run scientific parity across all migrated stage families | not started | Define comparisons and assess outputs/warnings with Michael. | Run old and candidate environments on representative small data and approve parity. | Pending. | Pending | Required before any permanent stage remapping. |
| Remap validated stages to `analysis` | not started | Update the registry/wrappers as one coherent approved phase and run control-plane tests. | Confirm deployment and run selected managed workflows. | Pending. | Pending | Existing environment definitions remain as rollback. |
| Retire superseded Conda environments | not started | Mark legacy definitions deprecated only after approval. | Approve and perform any HPC environment removal. | Pending. | Pending | No removal is authorized yet. |

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-06 | Store migration trackers and working artifacts under `image_migration/`. | Keeps temporary engineering state out of canonical user documentation. |
| 2026-08-06 | Test one full joint merger rather than adding packages one family at a time. | A single resolver transaction tests the intended future combined build and avoids repeated large installs. |
| 2026-08-06 | Initially merge segmentation, BioBatchNet, CellCharter, and STARLING into candidate `sbt-analysis`. | They appeared able to share Python 3.11, NumPy 1.26.4, and Torch 2.9.1/CUDA 12.8. This decision was superseded on 2026-08-12. |
| 2026-08-06 | Pin CellCharter to `0.3.7`. | This is the requested current release and requires Python 3.11 or newer. |
| 2026-08-06 | Pin BioBatchNet to commit `b1d708c62f5bac70f323a36aa28c6057f63e8222`. | This is the precise VCS revision captured from the working HPC environment. |
| 2026-08-06 | Exclude MaxFuse from the merger. | It will be considered as a dedicated legacy runtime/image later. |
| 2026-08-06 | Keep all existing stage mappings unchanged until parity passes. | The four current environments remain the immediate rollback path. |
| 2026-08-12 | Replace STARLING with RAPIDS in the consolidation target. | RAPIDS, segmentation and BioBatchNet together cover the principal clustering routes; STARLING remains available separately. |
| 2026-08-12 | Target Python 3.12, RAPIDS 26.04, CUDA 12.8 and `rapids-singlecell-cu12` 0.16.1. | Python 3.12 is shared by current RAPIDS-singlecell and CellCharter; RAPIDS 26.04 matches the current CUDA-12 wheel generation while CUDA 12.8 matches the exported CSF3 runtime. |
| 2026-08-12 | Keep CellCharter 0.3.7 and SpatialData 0.4/Zarr 2 in the first complete merger. | The pinned compatibility spine supplies SpatialData while retaining Squidpy and NumPy 1.26 compatibility. |
| 2026-08-12 | Attempt Nimbus on Python 3.12, but split it into a dedicated runtime if its smoke or inference tests fail. | Nimbus metadata permits Python 3.12, but its upstream installation guidance only documents Python 3.9-3.11. |
| 2026-08-12 | Make flexible Conda channel priority part of the typed environment definition. | RAPIDS explicitly does not support strict channel priority; lock generation must not depend on a user's global Conda setting. |
| 2026-08-13 | Pin Zarr 2.18.7 with Numcodecs 0.15.1. | The first clean smoke run showed Zarr 2.18.3 importing Blosc symbols removed by Numcodecs 0.16. Zarr 2.18.7 is the final Zarr-2 release and explicitly declares `numcodecs<0.16`. |

## Current compatibility boundary

The approved first attempt maximises consolidation: segmentation, BioBatchNet,
CellCharter, RAPIDS and SpatialData are specified together. The environment
uses the NumPy 1.26/SpatialData 0.4/Zarr 2 generation so that CellCharter,
Squidpy and Nimbus can participate. Nimbus on Python 3.12 is the only known
unsupported-version experiment. A failure there causes a narrow Nimbus split,
not removal of CellCharter or RAPIDS.

## Open validation points

- Generate a fresh Linux lock with flexible channel priority and verify that
  RAPIDS 26.04, CUDA 12.8 and the NumPy 1.26 baseline solve together on CSF3.
- Confirm that SpatialData 0.4 preserves the SBT builder behavior required by
  the migrated segmentation stages. Resolver compatibility alone is not
  scientific or API parity.
- Confirm that Nimbus imports and completes representative inference under
  Python 3.12; otherwise create a dedicated Nimbus runtime.
- Confirm CUDA visibility and representative scientific outputs on CSF3; local
  import tests cannot establish either.

## Candidate baseline

| Component | Candidate choice | Basis |
|---|---|---|
| Python | 3.12 | Required by RAPIDS-singlecell 0.16.1 and supported by CellCharter 0.3.7; Nimbus support must be demonstrated. |
| NumPy | 1.26.4 | Matches working segmentation, satisfies Nimbus's `<2` constraint, and is accepted by RAPIDS-singlecell. |
| PyTorch | 2.9.1 | Matches working segmentation and satisfies BioBatchNet and CellCharter constraints. |
| RAPIDS / CUDA | RAPIDS 26.04 with CUDA 12.8 and flexible channel priority | Follows the upstream CUDA-12 environment family while retaining the successfully exported CSF3 CUDA version. |
| RAPIDS-singlecell | `rapids-singlecell-cu12==0.16.1` without the `[rapids]` extra | Uses the precompiled CUDA-12 wheel on top of the Conda-provided RAPIDS stack. |
| SpatialData | 0.4.0 with multiscale-spatial-image 2.0.2, spatial-image 1.2.1, Xarray 2024.11.0, Zarr 2.18.7, and Numcodecs 0.15.1 | Provides SpatialData while retaining the NumPy-1/Zarr-2 generation needed by the maximal merger; the Numcodecs ceiling is required by Zarr 2. |
| CellCharter | 0.3.7 | Explicit migration decision. |
| BioBatchNet | pinned VCS commit | Preserves the exported known revision. |
| Nimbus | 0.0.4, provisional on Python 3.12 | Include in the first attempt; split only if import or inference validation fails. |

## Rollback policy

The candidate has a new Conda name and no stages point to it. A failed solve,
install, smoke test, or parity comparison therefore requires only adjustment or
removal of `sbt-analysis`; the existing environments and mappings remain intact.
