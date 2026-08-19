# Scientific runtime consolidation tracker

This directory contains the working records for consolidating scientific
environments before the OCI/Apptainer image migration. Canonical environment
definitions remain under `HPC_env_files/` because that is where `sbt env`
loads them.

## Active catalogue

The `sbt-analysis` environment is now the standard registered runtime for the
four consolidated environment families:

| Retired environment key | Retired Conda name | Migrated stage mappings | Active destination |
|---|---|---|---|
| `segmentation` | `imc_segmentation` | `prep`, `vis`, `nimbus`, `subcl`, `dnqc` (second runtime), `aiinter`, `config`, `cellpose` (second runtime), `reint`, `remap`, `slogs`, `rebuildmeta`, `cellfeat`, `spatialdata`, `neighsig` | `analysis` / `sbt-analysis` |
| `biobatchnet` | `imc_biobatchnet` | `bbn` | `analysis` / `sbt-analysis` |
| `cellcharter` | `imc_cellcharter` | `bint`, `cchar`, `pairsp`, `nxsp`, `popqc` | `analysis` / `sbt-analysis` |
| `rapids` | `rapids_singlecell` | `rapids`, `cellvision-cluster`, `cellvision-full` (middle runtime) | `analysis` / `sbt-analysis` |

The CellCharter migration covers the normal existing-embedding route used by
default. Optional TRVAE/scArches mode is outside the supported consolidated
runtime and is not a parity target.

The following runtimes are deliberately not part of this merger:

| Runtime | Reason |
|---|---|
| `sbt-maxfuse` | Deferred; expected to retain a dedicated Python 3.8 environment/image. |
| `sbt-scportrait` | Its SpatialData, AnnData, and Cellpose constraints conflict with the shared baseline. |
| `sbt-cellpose-sam` | Requires Cellpose 4 while the shared runtime retains Cellpose 3. |
| `sbt-starling` | Removed from this merger when RAPIDS became the higher-priority clustering runtime; retained separately. |
| `sbt-hyperstac` | Dedicated TensorFlow 2.15 runtime. |
| `sbt-denoise` | Dedicated TensorFlow runtime; its source is being modernized from the fork's TensorFlow 2.6/Python 3.8 baseline to TensorFlow 2.15.1 on Python 3.10-3.11 before a new environment is locked. |
| `sbt-napari` | Interactive GUI runtime, not a pipeline compute environment. |

## Phase tracker

| Phase | Status | Codex work | Michael's work | Validation evidence | Checkpoint | Notes/blockers |
|---|---|---|---|---|---|---|
| Audit exported environments and identify compatibility families | complete | Compared registry, stage mappings, intent specifications, and latest exported snapshots. | Exported the live HPC environments. | Version matrix reviewed 2026-08-03. | Not committed | MaxFuse export absent and excluded by decision. |
| Define the original `sbt-analysis` candidate | complete | Added the registry candidate, initial Conda/pip specifications, smoke tests, and this tracker. | Approved a full joint-install experiment. | 30 environment-management tests passed; repository validation passed with no warnings; candidate remained inactive. | Not committed | The first CSF3 pip solve exposed the NumPy/Zarr generation conflict. |
| Revise `sbt-analysis` around RAPIDS and SpatialData | complete | Replaced STARLING with RAPIDS, encoded flexible channel priority, moved to Python 3.12, retained CellCharter 0.3.7 and the pinned SpatialData 0.4/Zarr 2 spine, and expanded smoke tests. | Approved the maximal merger and the policy of splitting Nimbus only if testing proves necessary. | 31 environment-management tests passed; repository validation passed with no warnings; `git diff --check` passed. Static specification validation reports only the intentionally missing new Linux lock and the expected BioBatchNet VCS-review warning. | Not committed | No production stage mappings changed. |
| Generate a new reviewed Linux lock and perform a clean joint install | complete | Reviewed the RAPIDS lock and clean installations; diagnosed and corrected the Zarr/Numcodecs boundary. | Generated the corrected lock and completed a clean `sbt-analysis` installation on CSF3. | Corrected lock SHA-256 `dc54f3f5dc9e9119420c40e1a22048df4b8cc6017fadfca0a35d6592945e6cb6`; Conda, pip extras and editable overlay installed; Python 3.12 and all exact-version assertions passed. | `d823c79` plus CSF3 lock | The remaining registered failure was isolated to unwanted optional scArches support, not the four target runtime families. |
| Run registered imports and initial GPU smoke tests | complete | Removed optional scArches from the candidate after its AnnData-incompatible import was isolated; retained CellCharter's default existing-embedding route and defined focused CUDA checks. | Pulled the corrected specification, ran the registered import suite, and submitted the GPU smoke job on CSF3. | `sbt env test analysis --format yaml` passed all 8 registered checks. CSF3 job `18578252` completed `0:0` on an NVIDIA A100-SXM4-80GB; Torch 2.9.1+cu128, CuPy 13.6.0, cuDF/cuML/RMM 26.04.00 and RAPIDS-singlecell 0.16.1 all passed, ending with `GPU_SMOKE_PASS`. | `9c88618`; `image_migration/logs/sbt-analysis-gpu-smoke-18578252-20260813.log` on CSF3 | This smoke test did not import cuGraph/dask-cuDF or execute Leiden, so it did not establish compatibility of the complete clustering path. |
| Correct the RAPIDS/SpatialData Dask-generation conflict | complete | Aligned the candidate to RAPIDS 24.12, RAPIDS-singlecell 0.12.0 and Dask/Distributed 2024.11.2; added cuGraph/dask-cuDF checks, a full GPU clustering smoke artifact, and version-aware Harmony handling that never substitutes Harmony2. Subsequent solver and smoke evidence aligned the candidate to Python 3.11, scikit-image 0.24.0 and Setuptools below 81, and corrected stage smoke tests to use the same process isolation as SLURM. | Generated the corrected lock, performed the clean installation, approved a controlled three-package repair, and ran the strengthened registered checks on CSF3. Michael elected to treat RAPIDS consolidation as successful while real-workflow testing continues. | Phase lock SHA-256 `9dd5d793b972dce0b69961a466b146ae1fc75ea865954475d6fd0b28305fbec6`, later superseded by the Numba-compatible lock below. In-place repair installed Panel 1.9.4, Panel Core 1.9.4 and Setuptools 80.10.2. On 2026-08-18 all 12 registered smoke tests passed, including SpatialData, CellCharter, BioBatchNet, Nimbus, RAPIDS/cuGraph/dask-cuDF, exact version assertions, and five isolated SBT stage imports. | `0e0588f`; CSF3 logs `sbt-analysis-three-package-repair-20260818-114456.log` and `sbt-analysis-post-repair-tests-20260818-114456.log` | `sbt env compare` still reports a mixture of comparison false positives and genuine package-layer drift caused by the current Conda-lock-then-pip installation model. This is an environment-management/reproducibility issue, not a failed smoke test, and must be corrected separately without perturbing the working runtime. RAPIDS-singlecell 0.12 supports original Harmony, not Harmony2. |
| Correct RAPIDS-singlecell 0.12 Numba compatibility | complete | Pinned Numba 0.60.0 in both dependency layers, strengthened registered and GPU validation, and recorded the production failure without changing the RAPIDS algorithm or CUDA generation. | Archived the prior lock, generated and published the corrected Linux lock, applied the focused three-package repair on CSF3, and ran all registered non-GPU checks. | Production job `18823078` completed neighbours and UMAP for 1,279,364 cells but segfaulted when cuGraph Leiden initialized under unsupported Numba 0.63.1. The corrected lock has canonical Linux/Git-blob SHA-256 `ad6bc0bc0ddc9f35429dbc8102745c01be027f9c79df11292d6d40be5a6d1b53`; the in-place repair installed Numba 0.60.0, llvmlite 0.43.0 and libllvm14 14.0.6; `NUMBA_REPAIR_PASS` and all 12 registered non-GPU tests passed. | `af8731a`; CSF3 logs `sbt-analysis-numba60-repair-20260819.log` and `sbt-analysis-numba60-tests-20260819.log` | The focused GPU Leiden smoke result is deferred because of queue time. Michael explicitly accepted the corrected lock provisionally so unrelated consolidation work can continue. A later GPU failure reopens RAPIDS acceptance without invalidating the lock repair or other merged runtime families. |
| Run targeted end-to-end acceptance | in progress | Keep the managed runtime and stage records suitable for representative real workflows and retain deferred validation as an explicit open item. | Continue other consolidation work while the focused GPU Leiden job waits; store its result when available and repeat the representative production RAPIDS workflow when practical. | Job `18823078` established that neighbours and UMAP run at production scale but failed at Leiden before the supported Numba repair. The corrected dependency set passes every registered non-GPU check and is provisionally accepted by explicit user decision. | Pending GPU Leiden and corrected production-run evidence | A failure under Numba 0.60.0 should be handled as a focused CUDA/cuGraph/UCX or scale investigation, not by silently changing algorithms or rolling back unrelated environment consolidation. |
| Standardise environment names and archive superseded specifications | complete | Renamed all active physical environments to the `sbt-*` convention; removed retired registry keys; remapped their stages to `analysis`; updated wrappers, diagnostics, tests and docs; archived the three repository-managed retired specifications and a pre-change registry snapshot under `image_migration/archive/retired_hpc_environments/`. | Published and deployed the corrected `sbt-analysis` lock. Dedicated external environments can be recreated under their registered `sbt-*` names when next needed; an explicit per-run environment override remains available during that user-managed transition. | 66 environment/run-control tests and 8 affected stage integration tests passed; all active shell wrappers pass `bash -n`; generated docs are current; repository validator passes; `sbt env list` exposes exactly eight standardized names; no active orchestration reference uses a retired physical Conda name or `IMC_ENV_*` runtime override. The canonical analysis lock is now committed and deployed. | `eed29bb`, `af8731a` | Existing physical environments were not deleted or automatically renamed. The denoising and other dedicated runtime rebuilds remain separate reviewed phases. |
| Repair `sbt-cellpose-sam` reproducibility | complete | Restored the exact exported Torch 2.9.1/CUDA 12.8 pip layer, required the Conda C++ runtime, strengthened model and stage imports, and made the SLURM wrapper select the environment C++ library after activation. | Archived the stale lock, regenerated and published the Linux lock, recreated `sbt-cellpose-sam`, and reran the registered acceptance checks on CSF3. | Canonical Linux/Git-blob lock SHA-256 `6e00103f5d3e3e48e461cf57b919f8d2df2264cec512d2ce18eb265847d88e35`. The recreated Python 3.10.20 environment contains Cellpose 4.0.7, Torch 2.9.1, Torchvision 0.24.1, CUDA runtime 12.8.90, Pydantic 2.13.4 and Conda libstdc++ 16.1.0. Lock generation, recreation, `cellpose.models`, the SBT stage import and all three registered tests passed. | `df578ff`, `ea55058`; CSF3 log `sbt-cellpose-sam-final-tests-20260819.log` | GPU model loading and representative image inference remain deferred scientific acceptance. A later failure there reopens Cellpose runtime/scientific validation without invalidating the reproducible environment installation. |
| Modernize `IMC_Denoise_Updated` for a maintainable Python 3.10 runtime | waiting for user | Migrated the package to TensorFlow 2.15.1 through `tf.keras`; added modern package metadata, explicit checkpoint handling, Python 3.10-3.11 constraints, compatibility tests, and README/migration documentation that distinguishes this fork from upstream. The completed feature branch was fast-forwarded into local `main`. | Push the reviewed `main` checkpoint when ready, then approve creation and testing of a clean candidate environment. Later, obtain license clarification before any public image containing IMC_Denoise is published. | 11 local tests passed and one TensorFlow-dependent test was skipped because TensorFlow is absent locally; 28 Python files parsed; 4 notebooks passed JSON validation; the lightweight package imported with Python 3.10; a version 1.1.0 wheel built with the expected metadata and package data. | `IMC_Denoise_Updated` local `main` at `0a1c936`; not pushed | TensorFlow execution, model training/loading and representative denoising remain unvalidated until the clean candidate environment exists. The inherited research-only license appears to prohibit redistribution, so public OCI publication is not currently authorized. |
| Retire superseded Conda installations | not started | Provide a reviewed removal plan only after the standardized catalogue is deployed. | Approve and perform any HPC environment removal. | Pending. | Pending | Archived specifications provide rollback inputs; no HPC environment deletion is authorized. |

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
| 2026-08-18 | Use Python 3.11 and scikit-image 0.24.0 in the RAPIDS 24.12 candidate. | The first replacement lock proved that the selected RAPIDS/cuCIM 24.12 package set cannot resolve with Python 3.12 or scikit-image 0.25. Python 3.11 is shared by the target packages and is within Nimbus's documented range; cuCIM accepts scikit-image 0.24. |
| 2026-08-18 | Pin Setuptools below 81 and test stage modules in isolated Python processes. | SpatialData 0.4 reaches xarray-schema, which imports the removed `pkg_resources` API. Normal SBT stages are separate SLURM-launched Python processes, so a smoke test must not require RAPIDS/Scanpy and SpatialData to share one interpreter when they select incompatible Dask DataFrame backends. |
| 2026-08-18 | Treat the current `sbt env compare` result as an environment-management/reproducibility defect rather than runtime failure. | Every registered import and exact-version assertion passes. The report combines false positives (compatible minor-version Conda pins and unmodelled pip transitives) with real layer drift where the pip installation replaced packages present in the Conda lock. The specification, installation model and comparison logic must be aligned separately without perturbing the working candidate. |
| 2026-08-18 | Promote `sbt-analysis` to the standard runtime for the segmentation, BioBatchNet, CellCharter and RAPIDS families. | Michael elected to assume the RAPIDS correction is successful and proceed with consolidation while representative workflow testing continues. |
| 2026-08-18 | Standardise all registered physical Conda names on the `sbt-*` prefix without automatic legacy aliases. | A clean catalogue avoids permanent compatibility overhead. Existing environments may be recreated under the new names; wrappers accept only the central `SBT_CONDA_ENV` and key-specific `SBT_CONDA_ENV_<KEY>` overrides before using standardized defaults. |
| 2026-08-18 | Archive superseded repository-managed environment definitions under `image_migration/archive/retired_hpc_environments/`. | This preserves exact rollback and investigation material without leaving retired environments active in the SBT registry. |
| 2026-08-18 | Preserve `sbt-cellpose-sam` as a dedicated Torch 2.9.1/CUDA 12.8 runtime and prefer its Conda C++ library on CSF3. | An unpinned Cellpose installation selected Torch 2.13/CUDA 13. Torch then loaded CSF3's older system libstdc++, making SciPy fail; the exact exported pip set plus the environment libstdc++ restores the known runtime generation and successful model import. |
| 2026-08-18 | Modernize `IMC_Denoise_Updated` around Python 3.10-3.11 and TensorFlow 2.15.1 rather than fold it into `sbt-analysis`. | TensorFlow 2.15.1 is the newest TensorFlow generation compatible with the existing `tf.keras` implementation and can potentially share a dedicated TensorFlow runtime with HyPERSTAC without disturbing the PyTorch/RAPIDS analysis stack. |
| 2026-08-18 | Use `.weights.h5` for newly created DeepSNiF weight checkpoints while retaining load-only support for historical `.hdf5` and `.h5` weights. | Keras requires the explicit suffix for weights-only checkpoints; accepting legacy files for loading preserves existing research artifacts without misrepresenting full-model `.keras` files as weights. |
| 2026-08-18 | Do not publish IMC_Denoise in a public OCI image without license clarification. | The inherited license permits limited research use but states that the software may not be distributed, shared or transferred. Public repository availability does not itself grant redistribution rights. |
| 2026-08-19 | Pin Numba 0.60.0 for RAPIDS-singlecell 0.12.0. | The first production-scale consolidated RAPIDS run reached cuGraph Leiden and segfaulted with Numba 0.63.1 installed. RAPIDS-singlecell 0.12.0 explicitly declares `numba<0.61.0` for its RAPIDS 24.12 extras, but omitting those extras had also omitted their upper bound from the combined environment solve. |
| 2026-08-19 | Provisionally accept the corrected `sbt-analysis` lock while GPU Leiden validation remains queued. | The exact Numba repair and all 12 registered non-GPU checks pass. Long GPU queue time should not block unrelated consolidation work, but the outstanding GPU and production-scale results remain explicit acceptance evidence and can reopen only the RAPIDS runtime decision. |
| 2026-08-19 | Accept `sbt-cellpose-sam` installation reproducibility while deferring GPU inference. | The published lock recreates the intended Python 3.10, Cellpose 4.0.7, Torch 2.9.1 and CUDA 12.8 runtime; its C++ runtime and every registered import pass. Long GPU queue time should not block work on other dedicated runtimes, while eventual model loading and representative inference remain explicit scientific acceptance evidence. |

## Current compatibility boundary

The standard analysis runtime consolidates segmentation,
BioBatchNet, CellCharter, RAPIDS and SpatialData are specified together. Its
shared compatibility spine is Python 3.11, NumPy 1.26, SpatialData 0.4, Zarr 2,
Dask 2024.11 and RAPIDS 24.12. CellCharter's optional TRVAE/scArches route and
RAPIDS-singlecell's newer Harmony2 algorithm are outside the standard runtime; the
default existing-embedding CellCharter route, original Harmony, and the full
PCA/neighbors/UMAP/Leiden path remain in scope. Nimbus is now within its
documented Python range, but representative inference still requires explicit
validation. A failure there causes a narrow Nimbus split rather than removal
of CellCharter or RAPIDS.

## Open validation points

- Run Cellpose-SAM model loading and representative small-image inference on a
  CSF3 GPU when practical, retaining mask/QC evidence and exact runtime
  provenance.
- Confirm that SpatialData 0.4 preserves the SBT builder behavior required by
  the migrated segmentation stages. Resolver compatibility alone is not
  scientific or API parity.
- Confirm that Nimbus imports and completes representative inference under
  Python 3.11; otherwise create a dedicated Nimbus runtime.
- Store the queued focused GPU result confirming that cuGraph, dask-cuDF and
  RAPIDS-singlecell Leiden execute together on CSF3 under Numba 0.60.0, then
  repeat the representative production RAPIDS workflow when practical.
- Align the Conda-lock plus pip-extras installation model and `sbt env compare`
  so that pip does not silently replace locked Conda packages and comparison
  distinguishes genuine drift from compatible pins and expected transitives.
- Preserve the published corrected `sbt-analysis` lock with canonical
  Linux/Git-blob SHA-256
  `ad6bc0bc0ddc9f35429dbc8102745c01be027f9c79df11292d6d40be5a6d1b53`.
  Windows checkouts may report a different working-tree byte hash when Git
  converts LF to CRLF; the committed blob and CSF3 file retain the canonical
  LF identity.
- Regenerate and review `HPC_env_files/sbt-denoise/conda-linux-64.lock` in a
  separate lock-maintenance phase after the Python 3.10/TensorFlow 2.15.1
  source update passes clean-environment testing; the moved historical lock
  still contains pip records that now belong in `pip-extras.txt`.
- Create a clean Python 3.10 candidate with TensorFlow 2.15.1 and run the
  TensorFlow compatibility tests, new and legacy checkpoint loading, a short
  training step, DIMR/DeepSNiF smoke execution, and representative HPC
  denoising before changing the active `sbt-denoise` specification.
- Obtain written clarification or a revised license from the upstream rights
  holder before publishing any public OCI image that contains IMC_Denoise.
- Continue targeted managed-run acceptance after activation. Exhaustive
  numerical comparison with each retired environment is not required.

## Standard analysis baseline

| Component | Standard choice | Basis |
|---|---|---|
| Python | 3.11 | Shared by the available RAPIDS 24.12 Conda builds, RAPIDS-singlecell 0.12.0, CellCharter 0.3.7, and Nimbus's documented range. |
| NumPy | 1.26.4 | Matches working segmentation, satisfies Nimbus's `<2` constraint, and is accepted by RAPIDS-singlecell. |
| PyTorch | 2.9.1 | Matches working segmentation and satisfies BioBatchNet and CellCharter constraints. |
| RAPIDS / CUDA | RAPIDS 24.12 with CUDA 12.5 and flexible channel priority | This is the last official RAPIDS generation pinned to Dask 2024.11.2, matching SpatialData 0.4 and Squidpy 1.6.5; Python 3.11 and scikit-image 0.24 satisfy the available RAPIDS/cuCIM build constraints. |
| RAPIDS-singlecell | `rapids-singlecell==0.12.0` without the `[rapids12]` extra | This release is paired with RAPIDS 24.12 and exposes the PCA, original Harmony, neighbors, UMAP and cuGraph Leiden APIs used by SBT. |
| Dask stack | Dask/Distributed 2024.11.2 with dask-expr 1.1.19 | Exact shared generation declared by RAPIDS 24.12, SpatialData 0.4 and Squidpy 1.6.5. |
| Setuptools | `<81` | Retains `pkg_resources`, required by the xarray-schema version reached through SpatialData 0.4. |
| SpatialData | 0.4.0 with multiscale-spatial-image 2.0.2, spatial-image 1.2.1, Xarray 2024.11.0, Zarr 2.18.7, and Numcodecs 0.15.1 | Provides SpatialData while retaining the NumPy-1/Zarr-2 generation needed by the maximal merger; the Numcodecs ceiling is required by Zarr 2. |
| CellCharter | 0.3.7 | Explicit migration decision. |
| scArches / TRVAE | Excluded | Optional CellCharter extra not required by the default SBT route; scArches 0.6.1 is incompatible with the candidate AnnData API. |
| RAPIDS Harmony2 | Unsupported by the standard runtime | RAPIDS-singlecell 0.12 predates Harmony2; SBT supports `harmony1` and rejects unsupported Harmony2 explicitly. |
| BioBatchNet | pinned VCS commit | Preserves the exported known revision. |
| Nimbus | 0.0.4 on Python 3.11 | Retained in the consolidated runtime; split only if representative inference demonstrates a concrete incompatibility. |

## Rollback policy

The previous repository-managed segmentation, BioBatchNet and CellCharter
specifications, their locks, and the pre-standardization registry are retained
under `image_migration/archive/retired_hpc_environments/`. If a consolidated
stage fails, restore the relevant archived specification and registry mapping
in a reviewed change or create a narrowly scoped replacement runtime. Existing
HPC Conda installations must not be deleted until the standardized catalogue
has been deployed and accepted.
