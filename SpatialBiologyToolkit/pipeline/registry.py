"""Typed stage and workflow registry for the current SLURM backend."""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from .models import ModeSpec, StageSpec


def _stage(
    name: str,
    script: str,
    description: str,
    *,
    depends_on: tuple[str, ...] = (),
    groups: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    produces: tuple[str, ...] = (),
    required_files: dict[str, list[str]] | None = None,
    outputs: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> StageSpec:
    return StageSpec(
        name=name,
        description=description,
        slurm_script=f"SLURM_scripts/{script}",
        depends_on=list(depends_on),
        groups=list(groups),
        requires_assets=list(requires),
        produces_assets=list(produces),
        required_files=required_files or {},
        expected_outputs=list(outputs),
        log_patterns=[
            f"{{run_dir}}/logs/{name}-%j.out",
            f"{{run_dir}}/logs/{name}-%j.err",
        ],
        notes=list(notes),
    )


STAGES: tuple[StageSpec, ...] = (
    _stage(
        "prep",
        "job_preprocess.sh",
        "Import IMC files, export TIFF stacks and channels, and build metadata.",
        groups=("segmentation", "full"),
        requires=("raw_imc_files",),
        produces=("tiff_stacks", "raw_images", "metadata"),
        outputs=("TIFF stacks", "raw channel TIFFs", "metadata and panel tables"),
    ),
    _stage(
        "vis",
        "job_visualisations.sh",
        "Generate UMAP, matrix, overlay, backgating, and population visualisations.",
        groups=("visualisation", "full"),
        requires=("anndata",),
        produces=("qc",),
        outputs=("QC/BasicProcess_QC visualisation outputs",),
    ),
    _stage(
        "nimbus",
        "job_nimbus.sh",
        "Quantify segmented cells with Nimbus and build cell tables and AnnData.",
        depends_on=("cellpose",),
        groups=("segmentation", "full"),
        requires=("masks", "metadata", "denoised_images"),
        produces=("cell_tables", "anndata", "qc"),
        required_files={"metadata": ["panel.csv", "metadata.csv"]},
        outputs=("Nimbus cell tables", "canonical AnnData", "Nimbus QC"),
    ),
    _stage(
        "bint",
        "job_batch_integration.sh",
        "Run Harmony and/or BBKNN batch integration and downstream clustering.",
        depends_on=("nimbus",),
        groups=("integration-harmony",),
        requires=("anndata",),
        produces=("anndata", "qc"),
        outputs=("Integrated AnnData", "batch integration QC"),
    ),
    _stage(
        "rapids",
        "job_rapids.sh",
        "Run GPU-accelerated processing, optional Harmony, UMAP, and Leiden.",
        depends_on=("nimbus",),
        groups=("integration-rapids", "full"),
        requires=("anndata",),
        produces=("anndata", "qc"),
        outputs=("Processed AnnData", "RAPIDS QC and optional parameter scan"),
    ),
    _stage(
        "bbn",
        "job_biobatchnet.sh",
        "Run BioBatchNet correction followed by UMAP and Leiden processing.",
        depends_on=("nimbus",),
        groups=("integration-biobatchnet",),
        requires=("anndata",),
        produces=("anndata", "qc"),
        outputs=("BioBatchNet-corrected AnnData", "BioBatchNet QC"),
    ),
    _stage(
        "subcl",
        "job_subclustering.sh",
        "Run checkpointed population subclustering and optional remap integration.",
        groups=("curation",),
        requires=("anndata",),
        produces=("anndata", "qc"),
        outputs=("Subclustering templates, figures, mappings, and AnnData"),
    ),
    _stage(
        "cchar",
        "job_cellcharter.sh",
        "Identify spatial neighbourhoods with CellCharter.",
        groups=("spatial",),
        requires=("anndata",),
        produces=("anndata", "qc"),
        outputs=("CellCharter annotations and QC summaries",),
    ),
    _stage(
        "starling",
        "job_starling.sh",
        "Run STARLING segmentation-aware probabilistic phenotyping.",
        groups=("spatial",),
        requires=("anndata",),
        produces=("anndata", "qc"),
        outputs=("STARLING annotations, model artifacts, and QC"),
    ),
    _stage(
        "dnqc",
        "job_denoising_qc.sh",
        "Generate denoising side-by-side QC and panel consistency checks.",
        depends_on=("denoise",),
        groups=("segmentation", "full"),
        requires=("raw_images", "denoised_images", "metadata"),
        produces=("qc",),
        required_files={"metadata": ["panel.csv"]},
        outputs=("Denoising QC images and panel consistency reports",),
    ),
    _stage(
        "aiinter",
        "job_ai.sh",
        "Apply optional AI-assisted labels to Leiden populations.",
        groups=("curation",),
        requires=("anndata",),
        produces=("anndata", "qc"),
        outputs=("Updated AnnData labels and AI interpretation QC",),
        notes=("Requires OPENAI_API_KEY when AI interpretation is enabled.",),
    ),
    _stage(
        "denoise",
        "job_denoising.sh",
        "Denoise channel TIFFs and compute denoising metrics.",
        depends_on=("prep",),
        groups=("segmentation", "full"),
        requires=("raw_images", "metadata"),
        produces=("denoised_images", "qc"),
        required_files={"metadata": ["panel.csv"]},
        outputs=("Denoised channel TIFFs", "denoising QC metrics"),
    ),
    _stage(
        "config",
        "job_config.sh",
        "Synchronise missing config defaults in the submitted run config copy.",
        groups=("maintenance",),
        outputs=("Updated run-local resolved config",),
        notes=(
            "The sbt config commands supersede this legacy maintenance stage.",
            "Under sbt, this stage never rewrites the user's source config.",
        ),
    ),
    _stage(
        "cellpose",
        "job_cellposesam.sh",
        "Preprocess DNA images and generate CellPose-SAM masks.",
        depends_on=("denoise",),
        groups=("segmentation", "full"),
        requires=("denoised_images",),
        produces=("masks", "qc"),
        outputs=("Preprocessed DNA images", "cell masks", "CellPose QC"),
    ),
    _stage(
        "reint",
        "job_reintegrate.sh",
        "Reintegrate markers previously removed from processed AnnData.",
        groups=("curation",),
        requires=("anndata",),
        produces=("anndata",),
        outputs=("Updated AnnData with reintegrated markers",),
    ),
    _stage(
        "zipqc",
        "job_zipqc.sh",
        "Zip selected QC directories for download.",
        groups=("maintenance",),
        requires=("qc",),
        outputs=("Dated QC zip archive",),
    ),
    _stage(
        "scport",
        "job_scport.sh",
        "Generate external scPortrait single-cell image outputs.",
        groups=("spatial",),
        requires=("denoised_images", "masks"),
        outputs=("scPortrait project outputs",),
        notes=("This wrapper currently uses fixed processed/ and masks/ arguments.",),
    ),
    _stage(
        "debug",
        "job_debug.sh",
        "Run SLURM wrapper environment and import diagnostics.",
        groups=("maintenance",),
        outputs=("Environment diagnostic log",),
    ),
    _stage(
        "pairsp",
        "job_pairwise_spatial.sh",
        "Run Squidpy interactions, distance bootstrap, and pair-correlation analyses.",
        groups=("spatial",),
        requires=("anndata",),
        produces=("qc",),
        outputs=("Pairwise spatial tables, matrices, and plots",),
    ),
    _stage(
        "nxsp",
        "job_networkx_spatial.sh",
        "Run per-ROI Squidpy and NetworkX spatial graph metrics.",
        groups=("spatial",),
        requires=("anndata",),
        produces=("qc",),
        outputs=("NetworkX spatial summaries, nulls, and plots",),
    ),
    _stage(
        "remap",
        "job_remap_obs.sh",
        "Apply an observation remap CSV or generate a blank remap template.",
        groups=("curation",),
        requires=("anndata",),
        produces=("anndata", "metadata"),
        outputs=("Updated AnnData and/or remap CSV",),
    ),
    _stage(
        "slogs",
        "job_slurmlogs.sh",
        "Organise legacy SLURM logs using AnnData pipeline metadata.",
        groups=("maintenance",),
        requires=("anndata",),
        produces=("slurm_logs",),
        outputs=("Organised SLURM logs and verification manifest",),
        notes=("sbt run records and sbt logs supersede most uses of this stage.",),
    ),
    _stage(
        "rebuildmeta",
        "job_rebuild_metadata.sh",
        "Rebuild metadata and panel tables from an existing AnnData file.",
        groups=("maintenance",),
        requires=("anndata",),
        produces=("metadata",),
        outputs=("Rebuilt metadata.csv, dictionary.csv, and panel.csv",),
    ),
)

STAGE_REGISTRY = {stage.name: stage for stage in STAGES}

MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        name="segmentation",
        description="Documented preprocessing-to-Nimbus segmentation workflow.",
        stages=["prep", "denoise", "dnqc", "cellpose", "nimbus"],
    ),
    ModeSpec(
        name="integration-rapids",
        description="GPU RAPIDS integration route after segmentation.",
        stages=["rapids"],
    ),
    ModeSpec(
        name="integration-harmony",
        description="Harmony/BBKNN integration route after segmentation.",
        stages=["bint"],
    ),
    ModeSpec(
        name="integration-biobatchnet",
        description="BioBatchNet integration route after segmentation.",
        stages=["bbn"],
    ),
    ModeSpec(
        name="spatial",
        description="Independent CellCharter, STARLING, pairwise, and NetworkX branches.",
        stages=["cchar", "starling", "pairsp", "nxsp"],
    ),
    ModeSpec(
        name="visualisation",
        description="Standard project visualisation and QC stage.",
        stages=["vis"],
    ),
    ModeSpec(
        name="full",
        description="Documented example route: segmentation, RAPIDS, then visualisation.",
        stages=["prep", "denoise", "dnqc", "cellpose", "nimbus", "rapids", "vis"],
    ),
)

MODE_REGISTRY = {mode.name: mode for mode in MODES}


def toolkit_root(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("SBT_TOOLKIT_ROOT"):
        candidates.append(Path(os.environ["SBT_TOOLKIT_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.append(Path.home() / "imcanalysis")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if (resolved / "SLURM_scripts").is_dir():
            return resolved
    return candidates[0].expanduser().resolve(strict=False)


def stage_script_path(
    stage: StageSpec,
    *,
    root: str | Path | None = None,
) -> Path:
    return (toolkit_root(root) / stage.slurm_script).resolve(strict=False)


def get_stage(name: str) -> StageSpec:
    try:
        return STAGE_REGISTRY[name]
    except KeyError as exc:
        matches = difflib.get_close_matches(name, STAGE_REGISTRY, n=4, cutoff=0.45)
        suggestion = f" Close matches: {', '.join(matches)}." if matches else ""
        raise KeyError(f"Unknown stage '{name}'.{suggestion}") from exc


def get_mode(name: str) -> ModeSpec:
    try:
        return MODE_REGISTRY[name]
    except KeyError as exc:
        matches = difflib.get_close_matches(name, MODE_REGISTRY, n=4, cutoff=0.45)
        suggestion = f" Close matches: {', '.join(matches)}." if matches else ""
        raise KeyError(f"Unknown mode '{name}'.{suggestion}") from exc


def registry_aliases() -> list[str]:
    return [stage.name for stage in STAGES]


__all__ = [
    "MODES",
    "MODE_REGISTRY",
    "STAGES",
    "STAGE_REGISTRY",
    "get_mode",
    "get_stage",
    "registry_aliases",
    "stage_script_path",
    "toolkit_root",
]
