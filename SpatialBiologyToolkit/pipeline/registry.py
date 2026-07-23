"""Typed stage and workflow registry for the current SLURM backend."""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from SpatialBiologyToolkit.environments.registry import environment_keys_for_stage

from .models import ModeSpec, StageSpec


STAGE_PRESENTATION: dict[str, tuple[str, int, str, str]] = {
    "prep": ("Preprocessing", 1, "Preprocessing", "preprocessing.md"),
    "denoise": ("Denoising", 2, "Denoising", "denoising.md"),
    "dnqc": ("Denoising QC", 3, "Denoising_QC", "denoising_qc.md"),
    "cellpose": ("Segmentation", 4, "Segmentation", "segmentation.md"),
    "nimbus": ("Quantification", 5, "Quantification", "quantification.md"),
    "bint": (
        "Batch Integration",
        6,
        "Batch_Integration",
        "batch_integration.md",
    ),
    "rapids": ("RAPIDS Processing", 7, "RAPIDS_Processing", "rapids.md"),
    "cellvision-extract": (
        "CellVision Extraction",
        8,
        "CellVision_Extraction",
        "cellvision.md",
    ),
    "cellvision-embed": (
        "CellVision Embedding",
        9,
        "CellVision_Embedding",
        "cellvision.md",
    ),
    "cellvision-cluster": (
        "CellVision Clustering",
        10,
        "CellVision_Clustering",
        "cellvision.md",
    ),
    "cellvision-plot": (
        "CellVision Plotting",
        11,
        "CellVision_Plotting",
        "cellvision.md",
    ),
    "cellvision-full": (
        "CellVision Full",
        12,
        "CellVision_Full",
        "cellvision.md",
    ),
    "bbn": (
        "BioBatchNet Integration",
        13,
        "BioBatchNet_Integration",
        "biobatchnet.md",
    ),
    "subcl": ("Subclustering", 14, "Subclustering", "subclustering.md"),
    "cchar": (
        "CellCharter Neighbourhoods",
        15,
        "CellCharter_Neighbourhoods",
        "cellcharter.md",
    ),
    "starling": (
        "STARLING Phenotyping",
        16,
        "STARLING_Phenotyping",
        "starling.md",
    ),
    "aiinter": (
        "AI Interpretation",
        17,
        "AI_Interpretation",
        "ai_interpretation.md",
    ),
    "vis": ("Visualisation", 18, "Visualisation", "visualisation.md"),
    "pairsp": (
        "Pairwise Spatial Analysis",
        19,
        "Pairwise_Spatial_Analysis",
        "pairwise_spatial.md",
    ),
    "nxsp": (
        "NetworkX Spatial Analysis",
        20,
        "NetworkX_Spatial_Analysis",
        "networkx_spatial.md",
    ),
    "reint": (
        "Marker Reintegration",
        21,
        "Marker_Reintegration",
        "marker_reintegration.md",
    ),
    "remap": (
        "Observation Remapping",
        22,
        "Observation_Remapping",
        "observation_remapping.md",
    ),
    "rebuildmeta": (
        "Metadata Rebuild",
        23,
        "Metadata_Rebuild",
        "metadata_rebuild.md",
    ),
    "scport": ("scPortrait Export", 24, "scPortrait_Export", "scportrait.md"),
    "config": (
        "Configuration Maintenance",
        25,
        "Configuration_Maintenance",
        "configuration_maintenance.md",
    ),
    "zipqc": ("Output Archive", 26, "Output_Archive", "output_archive.md"),
    "slogs": (
        "Legacy SLURM Log Migration",
        27,
        "Legacy_SLURM_Log_Migration",
        "slurm_log_migration.md",
    ),
    "debug": (
        "Environment Diagnostics",
        28,
        "Environment_Diagnostics",
        "environment_diagnostics.md",
    ),
    "popqc": (
        "Population Embedding QC",
        29,
        "Population_Embedding_QC",
        "population_embedding_qc.md",
    ),
}

STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "prep": ("SpatialBiologyToolkit.scripts.preprocess",),
    "vis": ("SpatialBiologyToolkit.scripts.basic_visualizations",),
    "nimbus": ("SpatialBiologyToolkit.scripts.segmentation_nimbus",),
    "bint": ("SpatialBiologyToolkit.scripts.basic_process_batch_integration",),
    "rapids": ("SpatialBiologyToolkit.scripts.basic_process_rapids",),
    "cellvision-extract": (
        "SpatialBiologyToolkit.scripts.cellvision_extract",
    ),
    "cellvision-embed": (
        "SpatialBiologyToolkit.scripts.cellvision_embed",
    ),
    "cellvision-cluster": (
        "SpatialBiologyToolkit.scripts.cellvision_cluster",
    ),
    "cellvision-plot": (
        "SpatialBiologyToolkit.scripts.cellvision_plot",
    ),
    "cellvision-full": (
        "SpatialBiologyToolkit.scripts.cellvision_extract",
        "SpatialBiologyToolkit.scripts.cellvision_embed",
        "SpatialBiologyToolkit.scripts.cellvision_cluster",
        "SpatialBiologyToolkit.scripts.cellvision_plot",
    ),
    "bbn": ("SpatialBiologyToolkit.scripts.basic_process_biobatchnet",),
    "subcl": ("SpatialBiologyToolkit.scripts.subclustering",),
    "cchar": ("SpatialBiologyToolkit.scripts.cellcharter_neighborhoods",),
    "starling": ("SpatialBiologyToolkit.scripts.starling_analysis",),
    "dnqc": (
        "SpatialBiologyToolkit.scripts.denoising_qc",
        "SpatialBiologyToolkit.scripts.check_panel_consistency",
    ),
    "aiinter": ("SpatialBiologyToolkit.scripts.ai_interpretation",),
    "denoise": ("SpatialBiologyToolkit.scripts.denoising",),
    "config": ("SpatialBiologyToolkit.scripts.update_config",),
    "cellpose": (
        "SpatialBiologyToolkit.scripts.preprocess_dna",
        "SpatialBiologyToolkit.scripts.cellpose_sam",
    ),
    "reint": ("SpatialBiologyToolkit.scripts.reintegrate_markers",),
    "pairsp": ("SpatialBiologyToolkit.scripts.pairwise_spatial",),
    "nxsp": ("SpatialBiologyToolkit.scripts.networkx_spatial",),
    "remap": ("SpatialBiologyToolkit.scripts.remap_obs",),
    "slogs": ("SpatialBiologyToolkit.scripts.slurmlogs",),
    "rebuildmeta": ("SpatialBiologyToolkit.scripts.rebuild_metadata",),
    "popqc": ("SpatialBiologyToolkit.scripts.population_embedding_qc",),
}

STAGE_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "prep": ("general", "preprocess"),
    "vis": ("general", "visualization", "process"),
    "nimbus": ("general", "segmentation", "nimbus"),
    "bint": ("general", "batch_integration"),
    "rapids": ("general", "rapids", "visualization"),
    "cellvision-extract": ("general", "cellvision"),
    "cellvision-embed": ("general", "cellvision"),
    "cellvision-cluster": ("general", "cellvision"),
    "cellvision-plot": ("general", "cellvision"),
    "cellvision-full": ("general", "cellvision"),
    "bbn": ("general", "biobatchnet"),
    "subcl": ("general", "process", "subclustering"),
    "cchar": ("general", "process", "cellcharter"),
    "starling": ("general", "starling"),
    "dnqc": ("general", "denoising"),
    "aiinter": ("general", "visualization", "process"),
    "denoise": ("general", "denoising"),
    "config": (),
    "cellpose": ("general", "createmasks"),
    "reint": ("general", "segmentation", "process"),
    "zipqc": ("general",),
    "scport": ("general",),
    "debug": (),
    "pairsp": ("general", "process", "pairwise_spatial"),
    "nxsp": ("general", "process", "networkx_spatial"),
    "remap": ("general", "remap_obs"),
    "slogs": ("general",),
    "rebuildmeta": ("general", "rebuild_metadata"),
    "popqc": ("general", "population_embedding_qc"),
}


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
    display_name, catalogue_order, output_slug, doc_name = STAGE_PRESENTATION[name]
    return StageSpec(
        name=name,
        display_name=display_name,
        catalogue_order=catalogue_order,
        output_slug=output_slug,
        documentation_path=f"docs/source/stages/{doc_name}",
        description=description,
        slurm_script=f"SLURM_scripts/{script}",
        environment_keys=environment_keys_for_stage(name),
        config_sections=list(STAGE_CONFIG_SECTIONS[name]),
        python_modules=list(STAGE_MODULES.get(name, ())),
        depends_on=list(depends_on),
        groups=list(groups),
        requires_assets=list(requires),
        produces_assets=list(produces),
        required_files=required_files or {},
        expected_outputs=list(outputs),
        log_patterns=[
            f"{{run_dir}}/logs/{name}_%j.out",
            f"{{run_dir}}/logs/{name}_%j.err",
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
        produces=("human_outputs",),
        outputs=("Stage-specific visualisation figures and tables",),
    ),
    _stage(
        "nimbus",
        "job_nimbus.sh",
        "Quantify segmented cells with Nimbus and build cell tables and AnnData.",
        depends_on=("cellpose",),
        groups=("segmentation", "full"),
        requires=("masks", "metadata", "denoised_images"),
        produces=("cell_tables", "anndata", "human_outputs"),
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
        produces=("anndata", "human_outputs"),
        outputs=("Integrated AnnData", "batch integration QC"),
    ),
    _stage(
        "rapids",
        "job_rapids.sh",
        "Run GPU-accelerated processing, optional Harmony, UMAP, and Leiden.",
        depends_on=("nimbus",),
        groups=("integration-rapids", "full"),
        requires=("anndata",),
        produces=("anndata", "human_outputs"),
        outputs=("Processed AnnData", "RAPIDS QC and optional parameter scan"),
    ),
    _stage(
        "cellvision-extract",
        "job_cellvision_extract.sh",
        "Extract normalized, identity-tracked single-cell images into H5SC.",
        depends_on=("nimbus",),
        groups=("cellvision",),
        requires=("anndata", "denoised_images", "masks"),
        produces=("cellvision_assets",),
        outputs=(
            "Identity-tracked H5SC images",
            "cell identity and extraction metadata",
            "channel normalization dictionary",
        ),
    ),
    _stage(
        "cellvision-embed",
        "job_cellvision_embed.sh",
        "Train the VICReg encoder and extract identity-aligned cell embeddings.",
        depends_on=("cellvision-extract",),
        groups=("cellvision",),
        requires=("cellvision_assets",),
        produces=("cellvision_assets", "human_outputs"),
        required_files={
            "cellvision_assets": [
                "extraction/data/single_cells.h5sc",
                "extraction_metadata.json",
            ]
        },
        outputs=(
            "Trained VICReg model",
            "cell-level embedding AnnData",
            "training diagnostics",
        ),
    ),
    _stage(
        "cellvision-cluster",
        "job_cellvision_cluster.sh",
        "Fuse CellVision morphology and BioBatchNet intensity graphs, then run RAPIDS UMAP and Leiden.",
        depends_on=("cellvision-embed",),
        groups=("cellvision",),
        requires=("anndata", "cellvision_assets"),
        produces=("cellvision_assets",),
        required_files={
            "cellvision_assets": ["cellvision_embeddings.h5ad"]
        },
        outputs=("Joint morphology/intensity graph, RAPIDS UMAP/Leiden CellVision AnnData",),
    ),
    _stage(
        "cellvision-plot",
        "job_cellvision_plot.sh",
        "Generate CellVision embedding comparisons, cluster-explanation QC, projections, and cell galleries.",
        depends_on=("cellvision-cluster",),
        groups=("cellvision",),
        requires=("anndata", "cellvision_assets"),
        produces=("human_outputs",),
        required_files={
            "cellvision_assets": [
                "extraction/data/single_cells.h5sc",
                "cellvision_clustered.h5ad",
            ]
        },
        outputs=("UMAP, explanation-QC, confusion, projection, and cell-gallery report",),
    ),
    _stage(
        "cellvision-full",
        "job_cellvision_full.sh",
        "Run extraction, VICReg embedding, RAPIDS clustering, and plotting in one GPU job.",
        depends_on=("nimbus",),
        groups=("cellvision-full",),
        requires=("anndata", "denoised_images", "masks"),
        produces=("cellvision_assets", "human_outputs"),
        outputs=(
            "Identity-tracked H5SC images",
            "trained VICReg model and cell embeddings",
            "RAPIDS UMAP/Leiden AnnData",
            "UMAP, confusion, projection, and cell-gallery report",
        ),
        notes=(
            "The combined wrapper switches between scPortrait and rapids_singlecell environments inside one GPU allocation.",
            "Use the cellvision mode or individual cellvision-* stages for separate checkpoint jobs.",
        ),
    ),
    _stage(
        "bbn",
        "job_biobatchnet.sh",
        "Run BioBatchNet correction followed by UMAP and Leiden processing.",
        depends_on=("nimbus",),
        groups=("integration-biobatchnet",),
        requires=("anndata",),
        produces=("anndata", "human_outputs"),
        outputs=("BioBatchNet-corrected AnnData", "BioBatchNet QC"),
    ),
    _stage(
        "subcl",
        "job_subclustering.sh",
        "Run checkpointed population subclustering and optional remap integration.",
        groups=("curation",),
        requires=("anndata",),
        produces=("anndata", "human_outputs"),
        outputs=("Subclustering templates, figures, mappings, and AnnData",),
    ),
    _stage(
        "cchar",
        "job_cellcharter.sh",
        "Identify spatial neighbourhoods with CellCharter.",
        groups=("spatial",),
        requires=("anndata",),
        produces=("anndata", "human_outputs"),
        outputs=("CellCharter annotations and QC summaries",),
    ),
    _stage(
        "starling",
        "job_starling.sh",
        "Run STARLING segmentation-aware probabilistic phenotyping.",
        groups=("spatial",),
        requires=("anndata",),
        produces=("anndata", "human_outputs"),
        outputs=("STARLING annotations, model artifacts, and QC",),
    ),
    _stage(
        "dnqc",
        "job_denoising_qc.sh",
        "Generate denoising side-by-side QC and panel consistency checks.",
        depends_on=("denoise",),
        groups=("segmentation", "full"),
        requires=("raw_images", "denoised_images", "metadata"),
        produces=("human_outputs",),
        required_files={"metadata": ["panel.csv"]},
        outputs=("Denoising QC images and panel consistency reports",),
    ),
    _stage(
        "aiinter",
        "job_ai.sh",
        "Apply optional AI-assisted labels to Leiden populations.",
        groups=("curation",),
        requires=("anndata",),
        produces=("anndata", "human_outputs"),
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
        produces=("denoised_images", "human_outputs"),
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
        produces=("masks", "human_outputs"),
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
        "Zip selected stage output directories for download.",
        groups=("maintenance",),
        requires=("human_outputs",),
        produces=("human_outputs",),
        outputs=("Dated project-output zip archive",),
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
        produces=("human_outputs",),
        outputs=("Pairwise spatial tables, matrices, and plots",),
    ),
    _stage(
        "nxsp",
        "job_networkx_spatial.sh",
        "Run per-ROI Squidpy and NetworkX spatial graph metrics.",
        groups=("spatial",),
        requires=("anndata",),
        produces=("human_outputs",),
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
        produces=("legacy_slurm_logs", "human_outputs"),
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
    _stage(
        "popqc",
        "job_population_embedding_qc.sh",
        "Assess population support from existing graph, UMAP, PCA, and clustering-sweep state.",
        groups=("qc",),
        requires=("anndata",),
        produces=("human_outputs",),
        outputs=(
            "Raw structural QC metrics, concern scores, and threshold flags",
            "Population QC figures and deterministic interpretation report",
            "Optional separately annotated AnnData asset",
        ),
        notes=(
            "This stage never recalculates Leiden, PCA, UMAP, or the Scanpy neighbour graph.",
            "No fixed integration-stage dependency is imposed because curated and multiple integration routes are supported.",
        ),
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
        name="cellvision",
        description="Checkpointed CellVision extraction, embedding, clustering, and plotting jobs.",
        stages=[
            "cellvision-extract",
            "cellvision-embed",
            "cellvision-cluster",
            "cellvision-plot",
        ],
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


def resolve_stage_selector(value: str) -> StageSpec:
    """Resolve an alias, output slug, or display name to one stage type."""
    normalized = "".join(character for character in value.lower() if character.isalnum())
    matches = [
        stage
        for stage in STAGES
        if normalized
        in {
            "".join(character for character in candidate.lower() if character.isalnum())
            for candidate in (stage.name, stage.output_slug, stage.display_name)
        }
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return get_stage(value)
    raise KeyError(f"Ambiguous stage selector '{value}'.")


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
    "resolve_stage_selector",
    "stage_script_path",
    "toolkit_root",
]
