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

The CellCharter migration target covers the normal existing-embedding route
used by default. Optional TRVAE/scArches mode remains available only through
the legacy environment during this migration and is not a parity target.

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
| Generate a new reviewed Linux lock and perform a clean joint install | complete | Reviewed the RAPIDS lock and clean installations; diagnosed and corrected the Zarr/Numcodecs boundary. | Generated the corrected lock and completed a clean `sbt-analysis` installation on CSF3. | Corrected lock SHA-256 `dc54f3f5dc9e9119420c40e1a22048df4b8cc6017fadfca0a35d6592945e6cb6`; Conda, pip extras and editable overlay installed; Python 3.12 and all exact-version assertions passed. | `d823c79` plus CSF3 lock | The remaining registered failure was isolated to unwanted optional scArches support, not the four target runtime families. |
| Run registered imports and initial GPU smoke tests | complete | Removed optional scArches from the candidate after its AnnData-incompatible import was isolated; retained CellCharter's default existing-embedding route and defined focused CUDA checks. | Pulled the corrected specification, ran the registered import suite, and submitted the GPU smoke job on CSF3. | `sbt env test analysis --format yaml` passed all 8 registered checks. CSF3 job `18578252` completed `0:0` on an NVIDIA A100-SXM4-80GB; Torch 2.9.1+cu128, CuPy 13.6.0, cuDF/cuML/RMM 26.04.00 and RAPIDS-singlecell 0.16.1 all passed, ending with `GPU_SMOKE_PASS`. | `9c88618`; `image_migration/logs/sbt-analysis-gpu-smoke-18578252-20260813.log` on CSF3 | This smoke test did not import cuGraph/dask-cuDF or execute Leiden, so it did not establish compatibility of the complete clustering path. |
| Correct the RAPIDS/SpatialData Dask-generation conflict | waiting for user | Aligned the candidate to RAPIDS 24.12, RAPIDS-singlecell 0.12.0 and Dask/Distributed 2024.11.2; added cuGraph/dask-cuDF checks, a full GPU clustering smoke artifact, and version-aware Harmony handling that never substitutes Harmony2. | Review the local change, then archive the current lock, generate a replacement lock, recreate the candidate, and run the strengthened checks on CSF3. | A real RAPIDS job (`18675046`) reached neighbors and UMAP but failed importing cuGraph for Leiden because dask-cuDF 26.04 was combined with Dask 2024.11.2. Local environment and compatibility tests pass; replacement lock and GPU execution are pending. | Pending | RAPIDS-singlecell 0.12 supports the original Harmony algorithm, not Harmony2. Harmony2 remains available through the legacy RAPIDS environment. |
| Run targeted end-to-end acceptance | not started | Use the existing per-run `sbt run --environment analysis` selection and assess completion, outputs, warnings, reporting and provenance for a small number of representative real workflows. | Rerun the failed BioBatchNet-to-RAPIDS workflow after the corrected environment passes its strengthened smoke test, then run any other workflow needed to cover an untested runtime branch. | Pending. | Pending | Exhaustive old-versus-new numerical parity is not required by decision; acceptance must still reach real file writing and managed-run finalization before permanent remapping. |
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
| 2026-08-13 | Exclude optional scArches/TRVAE support from `sbt-analysis`. | scArches 0.6.1 fails against the candidate's current AnnData API, while CellCharter 0.3.7 declares scArches only for its optional proteomics extra. The normal existing-embedding CellCharter route imports successfully and is the required migration target. |
| 2026-08-17 | Replace exhaustive numerical parity with targeted end-to-end acceptance. | The clean installation, registered imports and GPU component tests already establish broad runtime compatibility; representative managed runs are still required to exercise complete algorithms, file writing and reporting. |
| 2026-08-17 | Align `sbt-analysis` to RAPIDS 24.12, RAPIDS-singlecell 0.12.0 and Dask/Distributed 2024.11.2. | RAPIDS 26.04 requires Dask 2026.1.1, while SpatialData 0.4 and Squidpy 1.6.5 require Dask no newer than 2024.11.2. RAPIDS 24.12 is the official generation matching that Dask version, and RAPIDS-singlecell 0.12.0 is its paired release. |
| 2026-08-17 | Keep Harmony2 in the newer legacy RAPIDS environment rather than emulate it in `sbt-analysis`. | RAPIDS-singlecell 0.12 implements original Harmony but predates Harmony2. SBT maps `harmony1` explicitly to the original method and rejects `harmony2` on the older API instead of changing the requested scientific algorithm. |

## Current compatibility boundary

The revised candidate still maximises consolidation: segmentation,
BioBatchNet, CellCharter, RAPIDS and SpatialData are specified together. Its
shared compatibility spine is Python 3.12, NumPy 1.26, SpatialData 0.4, Zarr 2,
Dask 2024.11 and RAPIDS 24.12. CellCharter's optional TRVAE/scArches route and
RAPIDS-singlecell's newer Harmony2 algorithm are outside the candidate; the
default existing-embedding CellCharter route, original Harmony, and the full
PCA/neighbors/UMAP/Leiden path remain in scope. Nimbus on Python 3.12 remains
the only unsupported-version experiment. A failure there causes a narrow
Nimbus split rather than removal of CellCharter or RAPIDS.

## Open validation points

- Confirm that SpatialData 0.4 preserves the SBT builder behavior required by
  the migrated segmentation stages. Resolver compatibility alone is not
  scientific or API parity.
- Confirm that Nimbus imports and completes representative inference under
  Python 3.12; otherwise create a dedicated Nimbus runtime.
- Generate and inspect the revised Linux lock, then confirm that cuGraph,
  dask-cuDF and RAPIDS-singlecell Leiden execute together on a CSF3 GPU.
- Complete targeted managed-run acceptance before changing permanent stage
  mappings. Exhaustive numerical comparison with each legacy environment is
  not required.

## Candidate baseline

| Component | Candidate choice | Basis |
|---|---|---|
| Python | 3.12 | Supported by RAPIDS 24.12, RAPIDS-singlecell 0.12.0 and CellCharter 0.3.7; Nimbus support must be demonstrated. |
| NumPy | 1.26.4 | Matches working segmentation, satisfies Nimbus's `<2` constraint, and is accepted by RAPIDS-singlecell. |
| PyTorch | 2.9.1 | Matches working segmentation and satisfies BioBatchNet and CellCharter constraints. |
| RAPIDS / CUDA | RAPIDS 24.12 with CUDA 12.5 and flexible channel priority | This is the last official RAPIDS generation pinned to Dask 2024.11.2, matching SpatialData 0.4 and Squidpy 1.6.5; the published 24.12 Python 3.12 runtime uses CUDA 12.5. |
| RAPIDS-singlecell | `rapids-singlecell==0.12.0` without the `[rapids12]` extra | This release is paired with RAPIDS 24.12 and exposes the PCA, original Harmony, neighbors, UMAP and cuGraph Leiden APIs used by SBT. |
| Dask stack | Dask/Distributed 2024.11.2 with dask-expr 1.1.19 | Exact shared generation declared by RAPIDS 24.12, SpatialData 0.4 and Squidpy 1.6.5. |
| SpatialData | 0.4.0 with multiscale-spatial-image 2.0.2, spatial-image 1.2.1, Xarray 2024.11.0, Zarr 2.18.7, and Numcodecs 0.15.1 | Provides SpatialData while retaining the NumPy-1/Zarr-2 generation needed by the maximal merger; the Numcodecs ceiling is required by Zarr 2. |
| CellCharter | 0.3.7 | Explicit migration decision. |
| scArches / TRVAE | Excluded | Optional CellCharter extra not required by the default SBT route; scArches 0.6.1 is incompatible with the candidate AnnData API. |
| RAPIDS Harmony2 | Legacy environment only | RAPIDS-singlecell 0.12 predates Harmony2; the candidate supports `harmony1` and rejects unsupported Harmony2 explicitly. |
| BioBatchNet | pinned VCS commit | Preserves the exported known revision. |
| Nimbus | 0.0.4, provisional on Python 3.12 | Include in the first attempt; split only if import or inference validation fails. |

## Rollback policy

The candidate has a new Conda name and no stages point to it. A failed solve,
install, smoke test, or parity comparison therefore requires only adjustment or
removal of `sbt-analysis`; the existing environments and mappings remain intact.
