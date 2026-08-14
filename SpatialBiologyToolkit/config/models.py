"""Pydantic v2 models for the complete imcanalysis pipeline configuration.

The field names, defaults, and section structure intentionally mirror the legacy
configuration dataclasses from scripts.config_and_utils. Keep compatibility
first when extending these models.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    """Shared compatibility behavior for all config sections."""

    model_config = ConfigDict(
        extra="ignore",
        coerce_numbers_to_str=True,
        validate_default=True,
    )


def config_field(
    default: Any = ...,
    *,
    description: str,
    level: str = "advanced",
    stage: str,
    ui_group: str,
    advice: str = "",
    **field_kwargs: Any,
) -> Any:
    """Create a documented Pydantic field using the supported metadata keys."""
    field_options = dict(field_kwargs)
    default_factory = field_options.pop("default_factory", None)
    common = dict(
        description=description,
        json_schema_extra={
            "level": level,
            "stage": stage,
            "ui_group": ui_group,
            "advice": advice,
        },
        **field_options,
    )
    if default_factory is not None:
        return Field(default_factory=default_factory, **common)
    return Field(default=default, **common)


def config_section(section: str):
    """Attach baseline documentation/UI metadata to every field in a section."""

    def decorate(model_class: Type[ConfigModel]) -> Type[ConfigModel]:
        ui_group = section.replace("_", " ").title()
        for name, model_field in model_class.model_fields.items():
            if model_field.description is None:
                model_field.description = (
                    f"Configuration value for {name.replace('_', ' ')}."
                )
            metadata = dict(model_field.json_schema_extra or {})
            metadata.setdefault("level", "advanced")
            metadata.setdefault("stage", section)
            metadata.setdefault("ui_group", ui_group)
            metadata.setdefault("advice", "")
            model_field.json_schema_extra = metadata
        model_class.model_rebuild(force=True)
        return model_class

    return decorate


@config_section("general")
class GeneralConfig(ConfigModel):
    imc_files_folder: str = config_field(
        "IMC_files",
        description="Folder containing raw IMC input files in MCD or TXT format.",
        level="basic",
        stage="general",
        ui_group="Input folders",
        advice="Use this as the primary folder for raw IMC files.",
    )
    mcd_files_folder: str = 'MCD_files'  # Kept for backward compatibility
    metadata_folder: str = config_field(
        "metadata",
        description="Folder containing pipeline metadata and panel tables.",
        level="basic",
        stage="general",
        ui_group="Input folders",
        advice="Keep metadata.csv and panel.csv in this folder unless a stage overrides it.",
    )
    outputs_folder: str = config_field(
        "outputs",
        description=(
            "Root folder for sequential, human-facing execution reports, figures, and tables."
        ),
        level="basic",
        stage="general",
        ui_group="Outputs and provenance",
        advice=(
            "Keep reusable assets in their dedicated root folders; use this folder "
            "for material intended for human inspection."
        ),
    )
    qc_folder: str = config_field(
        "QC",
        description=(
            "Deprecated legacy catch-all QC folder retained for existing projects "
            "and direct compatibility runs."
        ),
        level="expert",
        stage="general",
        ui_group="Outputs and provenance",
        advice=(
            "New managed runs use general.outputs_folder and sequential execution folders. "
            "Do not repurpose this field for new output layouts."
        ),
    )
    masks_folder: str = 'masks'
    celltable_folder: str = 'cell_tables'
    tiff_stacks_folder: str  = 'tiff_stacks'
    raw_images_folder: str = Field(
        default='tiffs',
        description="Folder containing one ROI subdirectory per image and its unstacked raw-channel TIFFs; Denoising QC uses these as the before-denoising comparison images.",
    )
    denoised_images_folder: str = Field(
        default='processed',
        description="Folder containing the corresponding denoised channel TIFFs organised by ROI; Denoising QC also audits these files against metadata/panel.csv.",
    )
    slurm_logs_folder: str = 'SLURM_logs'
    case_obs: Optional[str] = config_field(
        None,
        description="Optional case or sample identifier column in adata.obs.",
        level="basic",
        stage="general",
        ui_group="Observation columns",
        advice="Set this for case-level summaries and statistical comparisons.",
    )
    roi_obs: str = config_field(
        "ROI",
        description="ROI identifier column in adata.obs.",
        level="basic",
        stage="general",
        ui_group="Observation columns",
        advice="Values should identify the imaging region associated with each cell.",
    )
    metadata_obs: Optional[List[str]] = config_field(
        None,
        description="Optional metadata columns used in QC and grouped summaries.",
        stage="general",
        ui_group="Observation columns",
        advice="List stable adata.obs columns that should appear in downstream summaries.",
    )
    groupby_obs: Optional[str] = config_field(
        None,
        description="Primary adata.obs column used for cross-condition analyses.",
        level="basic",
        stage="general",
        ui_group="Analysis groups",
        advice="Choose the main experimental grouping variable, such as treatment or outcome.",
    )
    groupby_obs_groups: Optional[List[str]] = config_field(
        None,
        description="Optional ordered subset of values from groupby_obs to analyse.",
        stage="general",
        ui_group="Analysis groups",
        advice="Leave unset to use all observed groups, or list groups in the desired display order.",
    )
    groupby_obs_primary_pairwise: Optional[List[str]] = config_field(
        None,
        description="Preferred two-group subset for pairwise comparisons.",
        stage="general",
        ui_group="Analysis groups",
        advice="Provide two values from groupby_obs_groups when one comparison should be prioritised.",
    )
    population_obs_all: Optional[List[str]] = config_field(
        None,
        description="Population or cluster annotation columns available to downstream stages.",
        stage="general",
        ui_group="Population annotations",
        advice="List adata.obs columns containing cell population or clustering labels.",
    )
    population_obs_primary: Optional[str] = config_field(
        None,
        description="Primary population annotation column used by downstream analyses.",
        level="basic",
        stage="general",
        ui_group="Population annotations",
        advice="Set this to the preferred final cell population label column.",
    )
    compartment_obs: Optional[str] = config_field(
        None,
        description="Optional tissue-compartment annotation column in adata.obs.",
        stage="general",
        ui_group="Population annotations",
        advice="Set this when abundance or spatial outputs should be stratified by tissue compartment.",
    )
    compartment_obs_list: Optional[List[str]] = config_field(
        None,
        description="Optional ordered subset of tissue compartments to analyse separately.",
        stage="general",
        ui_group="Population annotations",
        advice="Leave unset to use all compartments or provide the desired subset and order.",
    )
    spatial_key: str = config_field(
        "spatial",
        description="Canonical adata.obsm key containing XY spatial coordinates.",
        stage="general",
        ui_group="Spatial coordinates",
        advice="Change only when coordinates are stored under a different obsm key.",
    )
    x_coord_obs: str = config_field(
        "X_loc",
        description="Fallback adata.obs column containing X coordinates.",
        stage="general",
        ui_group="Spatial coordinates",
        advice="Used when the configured spatial_key is unavailable.",
    )
    y_coord_obs: str = config_field(
        "Y_loc",
        description="Fallback adata.obs column containing Y coordinates.",
        stage="general",
        ui_group="Spatial coordinates",
        advice="Used when the configured spatial_key is unavailable.",
    )
    master_index_obs: str = config_field(
        "Master_Index",
        description="Stable per-cell identifier column in adata.obs.",
        stage="general",
        ui_group="Observation columns",
        advice="Keep this stable across stages so cells can be matched after filtering or remapping.",
    )
    anndata_path: str = config_field(
        "anndata.h5ad",
        description="Canonical AnnData file path used across pipeline stages.",
        level="basic",
        stage="general",
        ui_group="AnnData and execution",
        advice="Use a path relative to the dataset working directory unless an absolute path is required.",
    )
    anndata_stage_run_mode: str = config_field(
        "repeat",
        description="Default policy for rerunning stages recorded in AnnData.",
        stage="general",
        ui_group="AnnData and execution",
        advice="Use repeat, skip, or intelligent according to the desired stage rerun behaviour.",
    )
    anndata_uns_log_key: str = config_field(
        "pipeline_stage_log",
        description="AnnData.uns key used to store pipeline stage history and config snapshots.",
        level="expert",
        stage="general",
        ui_group="AnnData and execution",
        advice="Keep the default unless integrating with an existing AnnData logging convention.",
    )

@config_section("preprocess")
class PreprocessConfig(ConfigModel):
    minimum_roi_dimensions: int = Field(
        default=200,
        gt=0,
        description=(
            "Minimum MCD acquisition size: both readimc width_um and height_um must "
            "be strictly greater than this value for the ROI to be exported; the "
            "filter is not applied to TXT inputs."
        ),
        json_schema_extra={
            "level": "basic",
            "stage": "preprocess",
            "ui_group": "Input filtering",
            "advice": (
                "Keep 200 for routine acquisitions; reduce it only when small MCD "
                "ROIs are intentional and verify that the retained regions contain "
                "enough tissue for downstream segmentation and analysis."
            ),
        },
    )

@config_section("rebuild_metadata")
class RebuildMetadataConfig(ConfigModel):
    input_adata_path: Optional[str] = Field(default=None, description="Optional source AnnData override; when omitted, metadata are reconstructed from general.anndata_path.")
    output_metadata_folder: Optional[str] = Field(default=None, description="Optional destination folder override for metadata.csv, dictionary.csv, and panel.csv; defaults to general.metadata_folder.")
    include_obs_patterns: Optional[List[str]] = Field(default=None, description="Optional regular-expression allowlist applied to observation names before ROI-invariant metadata columns are selected.")
    exclude_obs: List[str] = Field(default_factory=lambda: [
        'ObjectNumber',
        'CellID',
        'cell_id',
        'Master_Index',
        'X_loc',
        'Y_loc',
    ], description="Observation names excluded from ROI-level metadata reconstruction in addition to shared ROI, coordinate, master-index, and population columns.")
    exclude_obs_contains: List[str] = Field(default_factory=lambda: [
        'population',
        'leiden',
        'cluster',
        'nhood',
        'neighborhood',
    ], description="Case-insensitive name fragments used to exclude cell-population, clustering, and neighbourhood observations from ROI-level metadata.")
    preserve_existing_import_data: bool = Field(default=True, description="Compatibility setting retained in the schema; the current rebuild implementation does not read it and always sets import_data=true for every reconstructed ROI.")
    metadata_description_obs: Optional[str] = Field(default=None, description="Optional ROI-invariant observation used for metadata.csv and dictionary.csv descriptions; defaults to an invariant 'description' column and then the ROI name.")
    include_invariant_obs_in_metadata_csv: bool = Field(default=True, description="Append selected ROI-invariant observations to metadata.csv after its required pipeline columns.")
    include_invariant_obs_in_dictionary_csv: bool = Field(default=True, description="Append selected ROI-invariant observations to the ROI-indexed dictionary.csv table.")
    panel_channel_name_var: Optional[str] = Field(default=None, description="Optional adata.var column used for panel.csv channel_name; defaults to var['channel_name'] and then var_names.")
    panel_channel_label_var: Optional[str] = Field(default=None, description="Optional adata.var column used for cleaned, unique panel.csv channel_label values; defaults to var['channel_label'] and then var_names.")
    panel_use_denoised_default: bool = Field(default=True, description="Default panel.csv use_denoised flag for channels without a preservable existing setting.")
    panel_use_raw_default: bool = Field(default=False, description="Default panel.csv use_raw flag for channels without a preservable existing setting.")
    panel_to_denoise_default: bool = Field(default=True, description="Default panel.csv to_denoise flag for channels without a preservable existing setting.")
    panel_remove_outliers_default: bool = Field(default=False, description="Default panel.csv remove_outliers flag for channels without a preservable existing setting.")
    preserve_existing_panel_flags: bool = Field(default=True, description="Preserve parseable use_denoised, to_denoise, use_raw, and remove_outliers values from an existing panel.csv when cleaned channel labels match.")

@config_section("denoising")
class DenoisingConfig(ConfigModel):
    run_denoising: bool = Field(
        default=True,
        description=(
            "Run the configured IMC-Denoise restoration method; panel-driven "
            "outlier removal and side-by-side QC are controlled separately."
        ),
    )
    method: str = Field(
        default='deep_snf',
        description=(
            "Restoration method: 'deep_snf' applies DIMR hot-pixel removal followed "
            "by self-supervised shot-noise filtering, whereas 'dimr' applies only DIMR."
        ),
    )
    channels: List[str] = Field(
        default_factory=list,
        description=(
            "Channel identifiers to denoise and display in side-by-side QC, matched "
            "case-insensitively within ROI TIFF filenames; an empty list selects panel rows "
            "marked to_denoise, or use_denoised for older panel files."
        ),
    )
    # Parameters for both methods
    n_neighbours: int = Field(
        default=4,
        description=(
            "Number of locally most distribution-consistent neighbour differences summed "
            "by DIMR when classifying a centre pixel as a hot-pixel outlier."
        ),
    )
    n_iter: int = Field(
        default=3,
        description=(
            "Maximum number of iterative DIMR detection-and-median-replacement passes; "
            "later passes can remove small adjacent hot-pixel clusters exposed by earlier passes."
        ),
    )
    window_size: int = Field(
        default=3,
        description=(
            "Odd-width local DIMR window, in pixels, used to construct neighbour differences "
            "and to median-replace detected outliers."
        ),
    )
    # Outlier removal
    remove_outliers: bool = Field(
        default=True,
        description=(
            "Before IMC-Denoise, apply each channel's optional panel.csv remove_outliers rule "
            "and overwrite above-threshold pixels with zero in the raw TIFFs; this is a "
            "pipeline-specific preprocessing step, not DIMR."
        ),
    )
    remove_outliers_min_threshold: int = Field(
        default=500,
        description=(
            "Minimum permitted intensity cutoff for percentile-based panel outlier rules; "
            "a channel is skipped when its calculated cutoff is below this guard value."
        ),
    )
    # Parameters specific to 'deep_snf' method
    patch_step_size: int = Field(
        default=70,
        description=(
            "Initial horizontal and vertical stride, in pixels, between 64 x 64 DeepSNiF "
            "training patches; the stage also removes raw ROI folders whose recorded width "
            "or height is smaller than this value."
        ),
    )
    intelligent_patch_size: bool = Field(
        default=False,
        description=(
            "Adapt the training-patch stride in 20-pixel increments until the augmented "
            "patch count reaches the configured minimum and optional maximum."
        ),
    )
    intelligent_patch_size_threshold: float = Field(
        default=0.3,
        description=(
            "Deprecated compatibility setting retained in configuration; the current "
            "denoising implementation does not read this value."
        ),
    )
    intelligent_patch_size_minimum: int = Field(
        default=40,
        description=(
            "Smallest training-patch stride, in pixels, tried by adaptive patch sampling "
            "when too few patches are available."
        ),
    )
    intelligent_patch_size_min_patches: int = Field(
        default=5000,
        description=(
            "Target minimum number of DeepSNiF training patches after rotation and flip "
            "augmentation; training proceeds with a warning if this cannot be reached."
        ),
    )
    intelligent_patch_size_max_patches: Optional[int] = Field(
        default=None,
        description=(
            "Optional target maximum number of augmented DeepSNiF training patches; when "
            "exceeded, adaptive sampling increases the patch stride."
        ),
    )
    # DeepSNIF
    train_epochs: int = Field(
        default=75,
        description="Number of complete DeepSNiF training epochs run independently for each channel.",
    )
    train_initial_lr: float = Field(
        default=0.001,
        description=(
            "Initial Adam learning rate for DeepSNiF training; the library reduces it when "
            "validation loss plateaus."
        ),
    )
    train_batch_size: int = Field(
        default=200,
        description=(
            "Number of 64 x 64 patches in each DeepSNiF training batch; lower values reduce "
            "GPU memory demand at the cost of more batch updates."
        ),
    )
    ratio_thresh: float = Field(
        default=0.9,
        description=(
            "Maximum fraction of pixels below intensity 1 permitted in a DIMR-corrected "
            "training patch; lower values retain only more signal-rich patches."
        ),
    )
    pixel_mask_percent: float = Field(
        default=0.2,
        description=(
            "Percentage of pixels per training patch replaced by nearby values for "
            "self-supervision; 0.2 means 0.2%, not a fraction of 0.2."
        ),
    )
    val_set_percent: float = Field(
        default=0.15,
        description=(
            "Fraction of generated patches held out with a fixed split for validation; "
            "0.15 reserves 15%."
        ),
    )
    loss_function: str = Field(
        default="I_divergence",
        description=(
            "Masked-pixel data-fidelity loss: 'I_divergence' gives the Poisson-aware "
            "DeepSNiF objective; 'mse' and 'mse_relu' select the library's Noise2Void-style variants."
        ),
    )
    loss_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional .npz or .mat filename for saving per-epoch training and validation "
            "losses in the weights directory."
        ),
    )
    weights_save_directory: Optional[str] = Field(
        default=None,
        description=(
            "Directory for per-channel .keras weights, normalization-range files, and optional "
            "loss histories; when unset, use trained_weights under the working directory."
        ),
    )
    is_load_weights: bool = Field(
        default=False,
        description=(
            "Load weights_<channel>.keras and its matching normalization-range file from the "
            "weights directory instead of generating patches and training a new channel model."
        ),
    )
    lambda_HF: float = Field(
        default=3e-6,
        description=(
            "Weight of Hessian-norm regularization in the DeepSNiF loss, balancing masked-pixel "
            "data fidelity against spatial continuity of the predicted biological signal."
        ),
    )
    network_size: str = Field(
        default="small",
        description=(
            "DeepSNiF U-Net capacity: 'small' uses the compact, faster network, while 'normal' "
            "uses the original larger residual U-Net."
        ),
    )
    truncated_max_rate: float = Field(
        default=0.9999,
        description=(
            "Training-pixel quantile used to define the normalization range as 1.1 times that "
            "quantile; 0.9999 corresponds to the 99.99th percentile."
        ),
    )
    # Parameter scanning
    run_parameter_scan: bool = Field(
        default=False,
        description=(
            "Repeat denoising for each configured scan value and write each result to a "
            "parameter-suffixed denoised-image and QC location."
        ),
    )
    scan_parameter: Optional[str] = Field(
        default='truncated_max_rate',
        description=(
            "Name of the single DenoisingConfig field varied during a parameter scan, such "
            "as truncated_max_rate, train_epochs, or lambda_HF."
        ),
    )
    scan_values: Optional[List[Any]] = Field(
        default_factory=lambda: [0.99, 0.999, 0.99999],
        description="Ordered values assigned to scan_parameter in separate denoising runs.",
    )
    # Training verbosity
    verbose_training: bool = Field(
        default=False,
        description=(
            "Show TensorFlow/Keras training logs and progress output instead of suppressing "
            "routine framework messages."
        ),
    )
    # Parameters for QC images
    run_QC: bool = Field(
        default=True,
        description=(
            "Generate per-channel raw-versus-denoised comparison figures at the end of the "
            "denoising stage. The standalone Denoising QC stage always makes these figures "
            "when invoked and does not consult this switch."
        ),
    )
    colourmap: str = Field(
        default="jet",
        description=(
            "Matplotlib colour map used only for raw-versus-denoised QC figures; it does not "
            "alter TIFF intensities or model inputs."
        ),
    )
    dpi: int = Field(
        default=100,
        description="Raster resolution, in dots per inch, of denoising comparison figures.",
    )
    qc_image_dir: str = Field(
        default='denoising',
        description=(
            "Subdirectory name below the active QC/report location for side-by-side comparison "
            "figures; parameter scans append the parameter and value."
        ),
    )
    qc_num_rois: Optional[int] = Field(
        default=10,
        description=(
            "Positive maximum number of randomly sampled ROIs shown per channel in comparison "
            "figures; sampling currently has no fixed seed and null includes every ROI."
        ),
    )
    skip_already_denoised: bool = Field(
        default=True,
        description=(
            "Skip requested channels whose TIFF names are already present in the first existing "
            "denoised ROI folder, rather than overwrite those channel outputs."
        ),
    )

@config_section("createmasks")
class CreateMasksConfig(ConfigModel):
    specific_rois: Optional[List[str]] = Field(
        default=None,
        description=(
            "Exact ROI folder names to process in both DNA restoration and Cellpose-SAM; use "
            "null to process every ROI found in the denoised-image folder."
        ),
    )
    dna_image_name: str = Field(
        default='DNA1',
        description=(
            "Case-sensitive substring used to identify the nuclear DNA TIFF within each ROI "
            "folder. Exactly one filename must contain this value."
        ),
    )
    dna_preprocessing_output_folder_name: str = Field(
        default='preprocessed_dna',
        description=(
            "Project-relative directory that receives one Cellpose3-restored DNA TIFF per ROI "
            "and supplies the images subsequently segmented by Cellpose-SAM."
        ),
    )
    cellpose_cell_diameter: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Estimated median nuclear diameter in pixels on the original IMC image. Cellpose3 "
            "uses it to determine restoration scale; Cellpose-SAM also uses it when upscaling is "
            "disabled."
        ),
        json_schema_extra={
            "level": "basic",
            "stage": "createmasks",
            "ui_group": "Segmentation",
            "advice": (
                "Increase when nuclei are fragmented; decrease when neighbouring "
                "nuclei are merged."
            ),
        },
    )
    upscale_ratio: float = Field(
        default=1.7,
        description=(
            "Fallback reported and segmentation target-to-input diameter ratio used only for an "
            "unrecognised custom upscale_model_type; supported nuclei and cyto3 restorers instead "
            "use fixed assumed targets of 17 and 30 pixels."
        ),
    )
    expand_masks: int = Field(
        default=1,
        description=(
            "Number of original-resolution pixels by which to expand each nuclear label after "
            "segmentation; expansion stops where neighbouring labels meet and values at or below "
            "zero disable it."
        ),
    )
    perform_qc: bool = Field(
        default=True,
        description=(
            "Generate DNA restoration comparisons and Cellpose-SAM boundary overlays in the QC "
            "directory. This does not replace visual review of every biologically distinct "
            "tissue or acquisition batch."
        ),
    )
    qc_boundary_dilation: int = Field(
        default=0,
        description=(
            "Extra dilation, in display pixels, applied only to mask outlines in QC overlays; it "
            "does not alter the saved masks or any downstream measurements."
        ),
    )
    min_cell_area: Optional[int] = Field(
        default=15,
        description=(
            "Minimum accepted nuclear-object area in pixels. Null is treated as 15; the value is "
            "used during Cellpose inference and again on masks restored to the original IMC grid."
        ),
    )
    max_cell_area: Optional[int] = Field(
        default=200,
        description=(
            "Legacy absolute maximum-area setting retained for configuration compatibility; the "
            "active Cellpose-SAM stage does not read it. Use max_size_fraction for the current "
            "upper-size filter."
        ),
    )
    cell_pose_model: str = Field(
        default='nuclei',
        description=(
            "Legacy Cellpose3 segmentation-model name retained for compatibility; the active "
            "two-step stage uses Cellpose3 only for restoration and does not read this field."
        ),
    )
    cell_pose_sam_model: str = Field(
        default='cpsam',
        description=(
            "Cellpose-SAM model identifier or path to a compatible custom model. Use cpsam for "
            "the bundled generalist model; obsolete nuclei/cyto model names fall back to cpsam."
        ),
    )
    cellprob_threshold: float = Field(
        default=0.0,
        description=(
            "Foreground logit threshold for Cellpose-SAM mask formation. Lower values generally "
            "admit more pixels and objects; higher values make foreground assignment more "
            "conservative."
        ),
    )
    flow_threshold: float = Field(
        default=0.4,
        description=(
            "Maximum Cellpose flow-consistency error accepted for a candidate mask. Lower values "
            "reject more irregular masks; higher values retain more candidates, including "
            "potential failures."
        ),
    )
    run_deblur: bool = Field(
        default=True,
        description=(
            "Apply the Cellpose3 deblur_nuclei restoration model to each DNA image before any "
            "upsampling. The result is intended to aid segmentation, not marker quantification."
        ),
    )
    run_upscale: bool = Field(
        default=True,
        description=(
            "Apply Cellpose3 learned upsampling before Cellpose-SAM, then return the resulting "
            "labels to the original IMC dimensions using nearest-neighbour resizing."
        ),
    )
    image_normalise: bool = Field(
        default=True,
        description=(
            "Percentile-normalise each restored DNA image inside Cellpose-SAM before inference; "
            "disable only when intensities have already been placed on a suitable model-input "
            "scale."
        ),
    )
    image_normalise_percentile_lower: float = Field(
        default=0.0,
        description=(
            "Lower per-image percentile mapped to the bottom of the Cellpose-SAM normalisation "
            "range when image_normalise is enabled."
        ),
    )
    image_normalise_percentile_upper: float = Field(
        default=99.9,
        description=(
            "Upper per-image percentile mapped to the top of the Cellpose-SAM normalisation "
            "range when image_normalise is enabled; reducing it increases clipping of bright DNA "
            "pixels."
        ),
    )
    dpi_qc_images: int = Field(
        default=300,
        description=(
            "Resolution in dots per inch for saved restoration and segmentation QC figures; it "
            "changes figure rendering only, not segmentation."
        ),
    )

    # Cellpose-SAM reads the restored-DNA folder and writes to general.masks_folder.
    max_size_fraction: float = Field(
        default=0.4,
        description=(
            "Largest accepted object area as a fraction of the original ROI area in the pipeline's "
            "post-filter. Cellpose-SAM also applies its own 0.4 default internally because this "
            "field is not forwarded to model inference."
        ),
    )
    remove_edge_masks: bool = Field(
        default=False,
        description=(
            "Remove every label that touches an image border after masks are returned to the "
            "original IMC grid. Enable when partial border nuclei would bias object-level "
            "measurements."
        ),
    )
    fill_holes: bool = Field(
        default=True,
        description=(
            "Fill enclosed background holes separately within each predicted nuclear label before "
            "expansion and area filtering."
        ),
    )
    batch_size: int = Field(
        default=128,
        description=(
            "Cellpose-SAM inference batch size used when a GPU is available; CPU execution is "
            "forced to a batch size of one. Reduce this value if GPU memory is exhausted."
        ),
    )
    resample: bool = Field(
        default=True,
        description=(
            "Legacy inference setting retained for compatibility; the active wrapper does not "
            "forward it, so the pinned Cellpose-SAM implementation uses its own default."
        ),
    )
    augment: bool = Field(
        default=False,
        description=(
            "Enable Cellpose-SAM test-time tiling and flip augmentation. This can change boundaries "
            "and increase runtime, so compare QC on representative tissue before adopting it."
        ),
    )
    tile_overlap: float = Field(
        default=0.1,
        description=(
            "Legacy tile-overlap setting retained for compatibility; the active wrapper does not "
            "forward it and Cellpose-SAM therefore uses its own 0.1 default."
        ),
    )

    # Upscale model configuration
    upscale_model_type: str = Field(
        default='upsample_nuclei',
        description=(
            "Cellpose3 restoration model used for learned upsampling. Use upsample_nuclei for IMC "
            "DNA (17-pixel training target); upsample_cyto3 targets 30-pixel cellular objects."
        ),
    )

    @property
    def upscale_target_diameter(self) -> float:
        """Get the target diameter for the upscale model."""
        if self.upscale_model_type == 'upsample_nuclei':
            return 17.0
        elif self.upscale_model_type == 'upsample_cyto3':
            return 30.0
        else:
            # Fallback to calculated ratio
            return self.cellpose_cell_diameter * self.upscale_ratio

    @property
    def calculated_upscale_ratio(self) -> float:
        """Calculate the actual upscale ratio based on target diameter."""
        return self.upscale_target_diameter / self.cellpose_cell_diameter

    # Parameter scanning fields:
    run_parameter_scan: bool = Field(
        default=False,
        description=(
            "Run a two-parameter grid search with separate masks and QC outputs for comparison, "
            "rather than produce the canonical masks used downstream. Rerun normally after "
            "choosing settings."
        ),
    )
    param_a: Optional[str] = Field(
        default='cellprob_threshold',
        description=(
            "Name of the first createmasks field varied by the parameter scan; normally "
            "cellprob_threshold."
        ),
    )
    param_a_values: Optional[List[Any]] = Field(
        default_factory=lambda: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
        description=(
            "Candidate values for param_a; each is crossed with every param_b value, so list "
            "length directly affects scan runtime and output volume."
        ),
    )
    param_b: Optional[str] = Field(
        default='flow_threshold',
        description=(
            "Name of the second createmasks field varied by the parameter scan; normally "
            "flow_threshold."
        ),
    )
    param_b_values: Optional[List[Any]] = Field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        description=(
            "Candidate values for param_b; each is crossed with every param_a value to form the "
            "scan grid."
        ),
    )
    window_size: Optional[int] = Field(
        default=250,
        description=(
            "Legacy parameter-scan window size retained for configuration compatibility; the "
            "active Cellpose-SAM scan does not read it."
        ),
    )
    num_rois_to_scan: int = Field(
        default=3,
        description=(
            "Number of available restored ROIs selected at random for a parameter scan when "
            "specific_rois is null; selection currently has no fixed random seed."
        ),
    )
    scan_rois: Optional[List[str]] = Field(
        default=None,
        description=(
            "Legacy scan-specific ROI list retained for configuration compatibility; the active "
            "scan does not read it, so use specific_rois to choose ROIs."
        ),
    )


@config_section("segmentation")
class SegmentationConfig(ConfigModel):
    celltable_output: str = 'celltable.csv'
    marker_normalisation: List[str] = Field(default_factory=lambda: ["q0.999"])
    store_raw_marker_data: bool = False
    remove_channels_list: List[str] = Field(default_factory=lambda: ['DNA1', 'DNA3'])
    remove_and_store_markers: List[str] = Field(
        default_factory=list,
        description="Markers copied to a separate AnnData and removed from the main feature matrix before downstream clustering or integration; they can be restored later for interpretation.",
    )
    removed_markers_anndata_path: str = Field(
        default='anndata_removed.h5ad',
        description="AnnData path used to store excluded marker values and later read them during marker reintegration; keep its cells unchanged and in the original order.",
    )
    anndata_save_path: str = 'anndata.h5ad'
    create_roi_cell_tables: bool = True
    create_master_cell_table: bool = True
    create_anndata: bool = True
    allow_missing_channels: bool = False  # If True, fill missing channels with NaN; if False, only include channels present in all ROIs

@config_section("nimbus")
class NimbusConfig(ConfigModel):
    output_dir: str = Field(
        default='nimbus_output',
        description=(
            "Directory for preferred normalization_dict.csv, master cell tables, and optional per-ROI "
            "Nimbus confidence maps. Relative paths resolve from the project working directory."
        ),
    )
    roi_table_subfolder: str = Field(
        default='nimbus_cell_tables',
        description=(
            "Subdirectory below general.celltable_folder for ROI-level Nimbus cell tables; "
            "use an empty string to write them directly into general.celltable_folder."
        ),
    )
    master_celltable: str = Field(
        default='nimbus_celltable.csv',
        description=(
            "Filename or path for the combined cell table containing mask geometry and per-marker "
            "Nimbus scores. Relative paths are placed below output_dir; an empty value falls back "
            "to segmentation.celltable_output."
        ),
    )
    master_classic_celltable: str = Field(
        default='nimbus_classic_celltable.csv',
        description=(
            "Filename or path for conventional mean image intensities measured inside each adjusted "
            "cell mask. Relative paths are placed below output_dir."
        ),
    )
    master_expansion_celltable: str = Field(
        default='nimbus_expansion_celltable.csv',
        description=(
            "Filename or path for mean image intensities measured after independently dilating each "
            "adjusted cell mask. Relative paths are placed below output_dir."
        ),
    )
    anndata_output: str = Field(
        default='anndata.h5ad',
        description=(
            "Deprecated compatibility path for the AnnData output. The current pipeline writes the "
            "canonical general.anndata_path and warns when this value differs."
        ),
    )
    roi_table_prefix: str = Field(
        default='nimbus_',
        description=(
            "Prefix added to each ROI name when writing ROI-level cell-table CSV filenames."
        ),
    )
    use_denoised_first: bool = Field(
        default=True,
        description=(
            "Deprecated compatibility setting not consulted by the current Nimbus stage. Image "
            "preference is selected per channel by panel.csv use_denoised and use_raw flags."
        ),
    )
    allow_raw_fallback: bool = Field(
        default=True,
        description=(
            "When a channel's panel-selected denoised or raw image is unavailable, also search the "
            "other image folder before declaring the channel missing."
        ),
    )
    simple_image_names: bool = Field(
        default=False,
        description=(
            "Match channel TIFFs using channel_label.tiff rather than the default "
            "channel_name_channel_label filename hint derived from panel.csv."
        ),
    )
    mask_extensions: List[str] = Field(
        default_factory=lambda: ['.tiff', '.tif'],
        description=(
            "Ordered filename extensions used to discover ROI label masks in general.masks_folder."
        ),
    )
    mask_boundary_offset_pixels: int = Field(
        default=0,
        description=(
            "Number of pixels by which to modify every cell mask before Nimbus scoring and all "
            "intensity extraction: positive values expand labels without overlap and negative "
            "values erode cells independently."
        ),
    )
    min_cell_area: Optional[int] = Field(
        default=None,
        description=(
            "Optional minimum cell area in pixels after mask-boundary adjustment; smaller labels "
            "are removed from Nimbus, cell tables, and AnnData."
        ),
    )
    max_cell_area: Optional[int] = Field(
        default=None,
        description=(
            "Optional maximum cell area in pixels after mask-boundary adjustment; larger labels "
            "are removed from Nimbus, cell tables, and AnnData."
        ),
    )
    test_time_augmentation: bool = Field(
        default=True,
        description=(
            "Average confidence maps predicted from 90-degree rotations and horizontal or vertical "
            "flips. This usually improves robustness but increases inference time."
        ),
    )
    batch_size: int = Field(
        default=10,
        description=(
            "Maximum number of image tiles processed together by tiled Nimbus inference; reduce it "
            "when accelerator memory is insufficient."
        ),
    )
    model_magnification: int = Field(
        default=10,
        description=(
            "Magnification expected by the selected Nimbus checkpoint. Inputs are rescaled from "
            "dataset_magnification to this value before inference."
        ),
    )
    dataset_magnification: int = Field(
        default=10,
        description=(
            "Magnification represented by the supplied channel images and masks. Set this to the "
            "true input scale so image and mask data are rescaled consistently for the model."
        ),
    )
    checkpoint: str = Field(
        default='latest',
        description=(
            "Nimbus model checkpoint. 'latest' checks Hugging Face for the newest V*.pt file and "
            "falls back to a cached checkpoint; any other value must name a local packaged checkpoint."
        ),
    )
    device: str = Field(
        default='auto',
        description=(
            "Torch inference device: 'auto' prefers Apple MPS, then CUDA, then CPU; explicit "
            "supported values are 'mps', 'cuda', and 'cpu'."
        ),
    )
    normalization_quantile: float = Field(
        default=0.999,
        description=(
            "Per-ROI, in-mask image quantile calculated for each channel; values are averaged across "
            "all usable ROIs to obtain the channel Vmax before Nimbus inference."
        ),
    )
    normalization_subset: int = Field(
        default=10,
        description=(
            "Maximum number of randomly sampled ROIs displayed in each normalization QC gallery. "
            "Normalization itself uses all usable ROIs; set to 0 to skip the galleries."
        ),
    )
    normalization_jobs: int = Field(
        default=1,
        description=(
            "Compatibility setting for normalization concurrency. The current toolkit wrapper "
            "calculates normalization serially, so this value does not presently change execution."
        ),
    )
    normalization_clip: List[float] = Field(
        default_factory=lambda: [0.0, 1.0],
        description=(
            "Compatibility bounds used by normalization QC, whose second value sets the displayed "
            "upper clip. The pinned Nimbus loader clips inference images to [0, 1]."
        ),
    )
    normalization_min_value: float = Field(
        default=3.0,
        description=(
            "Positive lower bound applied to computed channel Vmax values, preventing "
            "near-zero background estimates from amplifying noise."
        ),
    )
    reuse_saved_normalization: bool = Field(
        default=False,
        description=(
            "Load output_dir/normalization_dict.csv instead of recomputing channel bounds; "
            "legacy normalization_dict.json remains readable. Valid manual values are retained "
            "and normalization QC is still regenerated."
        ),
    )
    normalization_dict_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to a preferred Nimbus CSV with marker, vmax, and lower_threshold "
            "columns. Relative paths resolve from the project root, this path takes precedence "
            "over reuse_saved_normalization, and an external source is not overwritten."
        ),
    )
    norm_dict_qc_only: bool = Field(
        default=False,
        description=(
            "Stop after writing or loading the Nimbus normalization CSV (or legacy JSON) and "
            "generating normalization QC; "
            "do not run Nimbus, extract intensities, or create cell tables and AnnData."
        ),
    )
    save_prediction_maps: bool = Field(
        default=False,
        description=(
            "Save each per-pixel Nimbus confidence map as an 8-bit TIFF under an ROI subdirectory "
            "of output_dir. Per-cell floating-point scores are produced regardless."
        ),
    )
    allow_prediction_resize: bool = Field(
        default=False,
        description=(
            "On an unexpected confidence-map versus mask shape mismatch, resize the prediction to "
            "the mask instead of failing. Enable only as a diagnosed fallback because resizing can "
            "alter cell-level scores."
        ),
    )
    use_existing_master_celltables: bool = Field(
        default=False,
        description=(
            "Reuse valid existing Nimbus, classic, and expansion master CSVs where available. This "
            "is automatically disabled when mask offsets or area filters could make tables stale."
        ),
    )
    extract_classic_intensities: bool = Field(
        default=True,
        description=(
            "Also calculate conventional mean source-image intensity inside each adjusted cell mask "
            "and add raw and marker-normalized AnnData layers."
        ),
    )
    extract_expansion_intensities: bool = Field(
        default=True,
        description=(
            "Also calculate mean source-image intensity after independently dilating each adjusted "
            "cell mask and add raw and marker-normalized AnnData layers."
        ),
    )
    expansion_pixels: int = Field(
        default=10,
        description=(
            "Number of binary-dilation iterations applied independently to each cell for expansion "
            "intensity extraction; expanded regions may overlap and include neighbouring signal."
        ),
    )
    expansion_jobs: int = Field(
        default=1,
        description=(
            "ROI-level worker processes for expansion extraction: 1 is sequential, -1 requests all "
            "available CPUs, and values above 1 request that many workers subject to ROI and CPU counts."
        ),
    )

    @model_validator(mode="after")
    def validate_normalization_dict_path(self) -> "NimbusConfig":
        if self.normalization_dict_path is None:
            return self
        path = str(self.normalization_dict_path).strip()
        if not path:
            self.normalization_dict_path = None
        elif not path.casefold().endswith(".csv"):
            raise ValueError(
                "nimbus.normalization_dict_path must point to a .csv file in the "
                "marker,vmax,lower_threshold format"
            )
        else:
            self.normalization_dict_path = path
        return self


@config_section("nimbus_normalization_scan")
class NimbusNormalizationScanConfig(ConfigModel):
    """Pre-AnnData Nimbus normalization sensitivity scan."""

    markers: Optional[List[str]] = config_field(
        None,
        description="Optional marker subset to scan; null uses all rows from a baseline CSV when supplied, otherwise every channel available in all usable ROIs.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Scan scope",
        advice="Use a small scan-parameter CSV or an explicit marker list to keep repeated GPU inference focused.",
    )
    rois: Optional[List[str]] = config_field(
        None,
        description="Optional explicit ROI subset used for repeated Nimbus inference.",
        level="advanced",
        stage="nimbus-scan",
        ui_group="Scan scope",
        advice="When set, these ROI names take precedence over max_rois and must be present after Nimbus input discovery.",
    )
    roi_selection_strategy: Literal["marker_expression_range", "random"] = config_field(
        "marker_expression_range",
        description="Automatic ROI-selection strategy used independently for each marker when rois is null.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Scan scope",
        advice="marker_expression_range spreads selections from low to high mean intracellular input signal above the marker lower threshold; random preserves the earlier seeded cohort-wide sampling behavior.",
    )
    max_rois: int = config_field(
        10,
        description="Maximum ROIs selected per marker for repeated inference; zero uses every usable ROI.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Scan scope",
        advice="Normalization baselines still use all usable ROIs. With marker_expression_range, different markers can use different ROI subsets.",
        ge=0,
    )
    random_seed: int = config_field(
        0,
        description="Seed for reproducible ROI sampling when roi_selection_strategy is random.",
        stage="nimbus-scan",
        ui_group="Scan scope",
    )
    baseline_normalization_dict_path: Optional[str] = config_field(
        None,
        description="Optional CSV supplying marker, baseline Vmax, and lower-threshold rows; when markers is null, its rows also define the scan marker subset. Legacy Nimbus JSON remains readable for baseline values.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Vmax candidates",
        advice="Preferred columns are marker, baseline_vmax, and lower_threshold. Vmax factors remain in config. Relative paths resolve from the project root and the source file is never overwritten.",
    )
    vmax_factors: List[float] = config_field(
        default_factory=lambda: [
            0.25,
            0.3535533905932738,
            0.5,
            0.7071067811865476,
            1.0,
            1.4142135623730951,
            2.0,
            2.8284271247461903,
            4.0,
        ],
        description="Positive factors multiplied by each marker's baseline Vmax to form its default log-spaced scan.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Vmax candidates",
        advice="The default covers one quarter to four times baseline in half-octave steps.",
    )
    marker_baseline_vmax: Dict[str, float] = config_field(
        default_factory=dict,
        description="Optional marker-to-baseline-Vmax mapping; vmax_factors are multiplied by each supplied value.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Vmax candidates",
        advice="Use this for a current normalization dictionary written directly in config; markers omitted here use the baseline dictionary path or computed quantile.",
    )
    marker_lower_thresholds: Dict[str, float] = config_field(
        default_factory=dict,
        description="Optional marker-to-lower-threshold mapping in absolute input-intensity units; omitted markers use zero.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Vmax candidates",
        advice="Values at or below the threshold map to normalized zero. Every threshold must remain below all Vmax candidates scanned for that marker.",
    )
    marker_vmax_values: Dict[str, List[float]] = config_field(
        default_factory=dict,
        description="Optional marker-to-explicit-Vmax mapping overriding vmax_factors for selected markers.",
        level="advanced",
        stage="nimbus-scan",
        ui_group="Vmax candidates",
        advice="Each marker override needs at least three distinct positive values to estimate local sensitivity.",
    )
    positive_score_thresholds: List[float] = config_field(
        default_factory=lambda: [0.25, 0.5, 0.75],
        description="Nimbus cell-score thresholds whose positive-cell proportions are plotted across Vmax candidates.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Interpretation",
        advice="These are sensitivity views, not universal biological positivity cutoffs.",
    )
    primary_positive_score_threshold: float = config_field(
        0.5,
        description="Nimbus score threshold used for call-flip, cliff, stability, and recommendation diagnostics.",
        level="basic",
        stage="nimbus-scan",
        ui_group="Interpretation",
        advice="Validate this threshold against marker images or controls; Nimbus scores are not universally calibrated.",
        ge=0,
        le=1,
    )
    stability_tolerance: float = config_field(
        0.05,
        description="Maximum adjacent change in primary positive-cell fraction per two-fold Vmax change for a stable candidate.",
        stage="nimbus-scan",
        ui_group="Recommendation diagnostics",
        ge=0,
        le=1,
    )
    call_flip_tolerance: float = config_field(
        0.05,
        description="Maximum fraction of individual cells whose primary call may flip across either adjacent Vmax step for a stable candidate.",
        stage="nimbus-scan",
        ui_group="Recommendation diagnostics",
        advice="This prevents balanced positive-to-negative and negative-to-positive swaps from looking stable merely because the total positive proportion is unchanged.",
        ge=0,
        le=1,
    )
    saturation_tolerance: float = config_field(
        0.05,
        description="Maximum fraction of in-mask input pixels clipped at normalized value 1 for a stable candidate.",
        stage="nimbus-scan",
        ui_group="Recommendation diagnostics",
        ge=0,
        le=1,
    )
    cliff_tolerance: float = config_field(
        0.15,
        description="Adjacent primary positive-fraction shift at which a marker is flagged as having a Vmax cliff.",
        stage="nimbus-scan",
        ui_group="Recommendation diagnostics",
        ge=0,
        le=1,
    )
    save_cell_scores: bool = config_field(
        True,
        description="Write one compressed per-cell score table per marker for detailed review and reproducibility.",
        stage="nimbus-scan",
        ui_group="Outputs",
        advice="Disable to reduce report size; aggregate, ROI, recommendation, and figure outputs are always retained.",
    )

    @model_validator(mode="after")
    def validate_scan_settings(self) -> "NimbusNormalizationScanConfig":
        for field_name in ("markers", "rois"):
            values = getattr(self, field_name)
            if values is None:
                continue
            cleaned = list(
                dict.fromkeys(str(value).strip() for value in values if str(value).strip())
            )
            if not cleaned:
                raise ValueError(
                    f"nimbus_normalization_scan.{field_name} must be null or contain at least one non-empty value"
                )
            setattr(self, field_name, cleaned)

        if self.baseline_normalization_dict_path is not None:
            path = str(self.baseline_normalization_dict_path).strip()
            self.baseline_normalization_dict_path = path or None

        def validate_positive_grid(label: str, values: List[float]) -> List[float]:
            parsed = sorted(set(float(value) for value in values))
            if len(parsed) < 3 or any(
                not math.isfinite(value) or value <= 0 for value in parsed
            ):
                raise ValueError(
                    f"{label} must contain at least three distinct finite positive values"
                )
            return parsed

        self.vmax_factors = validate_positive_grid(
            "nimbus_normalization_scan.vmax_factors", self.vmax_factors
        )

        def validate_marker_keys(
            label: str, values: Dict[str, Any]
        ) -> tuple[Dict[str, Any], Dict[str, str]]:
            cleaned: Dict[str, Any] = {}
            folded_keys: Dict[str, str] = {}
            for marker, value in values.items():
                cleaned_marker = str(marker).strip()
                if not cleaned_marker:
                    raise ValueError(f"{label} keys must be non-empty marker names")
                folded_marker = cleaned_marker.casefold()
                if folded_marker in folded_keys:
                    raise ValueError(
                        f"{label} keys must be unique ignoring case: "
                        f"{folded_keys[folded_marker]!r} and {cleaned_marker!r}"
                    )
                folded_keys[folded_marker] = cleaned_marker
                cleaned[cleaned_marker] = value
            return cleaned, folded_keys

        raw_marker_baselines, baseline_keys = validate_marker_keys(
            "nimbus_normalization_scan.marker_baseline_vmax",
            self.marker_baseline_vmax,
        )
        cleaned_marker_baselines: Dict[str, float] = {}
        for marker, value in raw_marker_baselines.items():
            parsed_value = float(value)
            if not math.isfinite(parsed_value) or parsed_value <= 0:
                raise ValueError(
                    "nimbus_normalization_scan.marker_baseline_vmax values must "
                    f"be finite and positive; got {value!r} for {marker!r}"
                )
            cleaned_marker_baselines[marker] = parsed_value
        self.marker_baseline_vmax = cleaned_marker_baselines

        raw_lower_thresholds, _ = validate_marker_keys(
            "nimbus_normalization_scan.marker_lower_thresholds",
            self.marker_lower_thresholds,
        )
        cleaned_lower_thresholds: Dict[str, float] = {}
        for marker, value in raw_lower_thresholds.items():
            parsed_value = float(value)
            if not math.isfinite(parsed_value) or parsed_value < 0:
                raise ValueError(
                    "nimbus_normalization_scan.marker_lower_thresholds values "
                    f"must be finite and non-negative; got {value!r} for {marker!r}"
                )
            cleaned_lower_thresholds[marker] = parsed_value
        self.marker_lower_thresholds = cleaned_lower_thresholds

        cleaned_marker_values: Dict[str, List[float]] = {}
        raw_marker_values, explicit_keys = validate_marker_keys(
            "nimbus_normalization_scan.marker_vmax_values", self.marker_vmax_values
        )
        for marker, values in raw_marker_values.items():
            cleaned_marker_values[marker] = validate_positive_grid(
                f"nimbus_normalization_scan.marker_vmax_values[{marker!r}]", values
            )
        self.marker_vmax_values = cleaned_marker_values
        overlapping_markers = sorted(set(baseline_keys) & set(explicit_keys))
        if overlapping_markers:
            display_names = [baseline_keys[key] for key in overlapping_markers]
            raise ValueError(
                "Markers cannot appear in both "
                "nimbus_normalization_scan.marker_baseline_vmax and "
                "marker_vmax_values; choose a factor-based baseline or an explicit "
                f"candidate grid: {display_names}"
            )
        thresholds = sorted(set(float(value) for value in self.positive_score_thresholds))
        if not thresholds or any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in thresholds
        ):
            raise ValueError(
                "nimbus_normalization_scan.positive_score_thresholds must contain finite values in [0, 1]"
            )
        if self.primary_positive_score_threshold not in thresholds:
            thresholds.append(float(self.primary_positive_score_threshold))
            thresholds.sort()
        self.positive_score_thresholds = thresholds
        return self


@config_section("neighbour_signal")
class NeighbourSignalConfig(ConfigModel):
    """Empirical marker-halo and neighbour-attributable signal analysis."""

    enabled: bool = config_field(
        True,
        description="Run neighbour-attributable signal analysis when the stage is selected.",
        level="basic",
        stage="neighsig",
        ui_group="Stage control",
        advice="Disable only when a planned workflow should retain the stage without executing it.",
    )
    output_adata_path: str = config_field(
        "neighbour_attributable_signal.h5ad",
        description="Separate AnnData output whose X matrix contains neighbour-attributable fractions.",
        level="basic",
        stage="neighsig",
        ui_group="Output asset",
        advice="Use a path different from general.anndata_path; the input AnnData is never modified.",
    )
    source_target_table_path: str = config_field(
        "neighbour_signal_source_target.parquet",
        description="Sparse Parquet asset containing non-zero target-marker-source attribution relationships.",
        level="basic",
        stage="neighsig",
        ui_group="Output asset",
        advice="Global AnnData row indices and obs_names are authoritative; ROI and mask labels are retained for provenance.",
    )
    exemplar_mode: Literal["automatic", "manual", "augment"] = config_field(
        "automatic",
        description="Choose exemplars from input AnnData.X, from exemplar_obs, or by retaining manual exemplars and filling automatically.",
        level="basic",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="Automatic selection affects only which cells train the raw-pixel halo; it does not use X to calculate halo values or final scores.",
    )
    exemplar_obs: str = config_field(
        "Exemplar_stains",
        description="Categorical AnnData observation whose non-null values name convincing positive marker exemplars in manual or augment mode.",
        level="basic",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="Values must exactly match marker names on the AnnData variable axis.",
    )
    automatic_positive_threshold: float = config_field(
        0.5,
        description="Minimum input AnnData.X marker score for an automatic exemplar candidate.",
        level="basic",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="The default suits the usual 0-1 Nimbus inference score; change it when X uses another scale.",
    )
    automatic_same_marker_clearance_px: float = config_field(
        10.0,
        description="Minimum mask-to-mask pixel distance from another X-positive cell for the same marker.",
        level="basic",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="Other cells are allowed nearby because their segmented pixels are excluded from radial halo averages.",
        ge=0,
        le=256,
    )
    automatic_target_exemplars_per_marker: int = config_field(
        30,
        description="Target number of reproducibly sampled automatic exemplars per marker.",
        level="basic",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="Selection is balanced across ROIs and thirds of the positive X-score range rather than taking only the highest scores.",
        ge=1,
    )
    automatic_max_exemplars_per_roi: int = config_field(
        5,
        description="Maximum selected automatic exemplars for one marker from one ROI.",
        level="advanced",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="Increase this for datasets with few ROIs; the minimum exemplar rule still applies globally.",
        ge=1,
    )
    automatic_min_pixels_per_bin: int = config_field(
        8,
        description="Minimum unassigned pixels required in every outward halo-distance bin for an automatic candidate.",
        level="advanced",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="This permits nearby cells while rejecting candidates whose radial profile is too heavily occluded to estimate.",
        ge=1,
    )
    object_id_obs: str = config_field(
        "ObjectNumber",
        description="AnnData observation containing the positive integer label used in each ROI segmentation mask.",
        level="basic",
        stage="neighsig",
        ui_group="Cell mapping and exemplars",
        advice="Each (ROI, object ID) pair must be unique and present in the corresponding mask.",
    )
    max_halo_px: int = config_field(
        8,
        description="Maximum outward distance in pixels learned and projected from each source-cell mask.",
        level="basic",
        stage="neighsig",
        ui_group="Halo model",
        advice="Interpret this in the native pixel grid of the configured raw images and masks.",
        ge=1,
        le=128,
    )
    source_anchor_dilation_px: int = config_field(
        2,
        description="Small outward dilation used to calculate each cell's robust raw-image source strength.",
        level="basic",
        stage="neighsig",
        ui_group="Source definition",
        advice="Unassigned pixels are allocated to their nearest cell; pixels assigned to other cells are excluded.",
        ge=0,
        le=32,
    )
    source_anchor_quantile: float = config_field(
        0.95,
        description="Raw-image quantile within the source anchor used as source strength.",
        level="basic",
        stage="neighsig",
        ui_group="Source definition",
        advice="High quantiles remain robust to conservative nuclear masks for membrane or cytoplasmic markers.",
        gt=0,
        le=1,
    )
    min_exemplars: int = config_field(
        5,
        description="Minimum number of valid exemplar profiles required to model one marker.",
        level="basic",
        stage="neighsig",
        ui_group="Halo model",
        advice="Markers below this count are reported and assigned zero neighbour-attributable fractions.",
        ge=1,
    )
    source_threshold_quantile: float = config_field(
        0.10,
        description="Quantile of valid exemplar source strengths used as the marker-specific source threshold.",
        level="basic",
        stage="neighsig",
        ui_group="Source definition",
        advice="The default admits cells in the lower edge of the exemplar-positive brightness regime.",
        ge=0,
        le=1,
    )
    halo_aggregation: Literal["max", "sum"] = config_field(
        "max",
        description="Pixel-level rule for overlapping projected source halos.",
        level="basic",
        stage="neighsig",
        ui_group="Halo model",
        advice="Use max normally; sum is more aggressive when several source halos overlap.",
    )
    calculate_classic_intensities: bool = config_field(
        True,
        description="Store conventional mean raw-marker intensity within each segmentation mask.",
        level="basic",
        stage="neighsig",
        ui_group="Output asset",
        advice="Internal raw-image reductions still run when this output layer is disabled.",
    )
    n_jobs: Union[Literal["auto"], int] = config_field(
        "auto",
        description="ROI-level worker processes, or auto to follow the current SLURM/CPU allocation.",
        level="basic",
        stage="neighsig",
        ui_group="Resources",
        advice="Workers load one mask and marker images lazily; avoid nested multiprocessing.",
    )
    qc_markers: Optional[List[str]] = config_field(
        None,
        description="Deprecated compatibility setting; neighbour-signal QC now plots every marker.",
        level="advanced",
        stage="neighsig",
        ui_group="QC report",
        advice="Retained so older config files still validate; configured values no longer restrict report figures.",
    )
    max_qc_markers: Optional[int] = config_field(
        None,
        description="Deprecated compatibility limit; neighbour-signal QC now plots every marker.",
        level="advanced",
        stage="neighsig",
        ui_group="QC report",
        advice="Retained so older config files still validate; integer values are ignored by report generation.",
        ge=1,
        le=50,
    )
    umap_point_size: Optional[float] = config_field(
        None,
        description="Optional point size for every Scanpy neighbour-signal UMAP.",
        level="basic",
        stage="neighsig",
        ui_group="QC report",
        advice="This is passed to Scanpy as size (the scatter-plot s value); null uses Scanpy's dataset-size-aware default.",
        gt=0,
    )
    create_cell_galleries: bool = config_field(
        True,
        description="Create bounded exemplar, target-source, and automatic-selection image galleries for every marker.",
        level="basic",
        stage="neighsig",
        ui_group="QC report",
        advice="Galleries are qualitative model checks and do not change halo profiles or cell scores.",
    )
    gallery_examples_per_marker: int = config_field(
        6,
        description="Maximum stratified target-cell examples rendered for each marker.",
        level="basic",
        stage="neighsig",
        ui_group="QC report",
        advice="Examples cover dominant, competing-source, disagreement, isolated-positive, and self-control cases where available.",
        ge=1,
        le=12,
    )
    gallery_crop_margin_px: int = config_field(
        8,
        description="Additional native-image pixels shown around the target/source masks and modeled halo extent.",
        level="advanced",
        stage="neighsig",
        ui_group="QC report",
        advice="Increase this only when more tissue context is needed; it does not change max_halo_px.",
        ge=0,
        le=128,
    )
    population_obs: Optional[str] = config_field(
        None,
        description="Optional categorical population observation used for population-by-marker QC.",
        level="basic",
        stage="neighsig",
        ui_group="QC report",
        advice="Null falls back to general.population_obs_primary; missing annotations are skipped cleanly.",
    )
    source_target_qc_exclude_same_population: bool = config_field(
        True,
        description="Exclude same-population relationships from source-to-target population heatmaps.",
        level="advanced",
        stage="neighsig",
        ui_group="QC report",
        advice="The Parquet table and aggregate QC tables always retain same-population relationships.",
    )
    high_risk_threshold: float = config_field(
        0.5,
        description="Descriptive score threshold used only for per-cell counts and QC summaries.",
        level="advanced",
        stage="neighsig",
        ui_group="QC report",
        advice="This threshold is not a validated biological cutoff and does not alter the score.",
        ge=0,
        le=1,
    )

    @model_validator(mode="after")
    def validate_neighbour_signal(self) -> "NeighbourSignalConfig":
        if isinstance(self.n_jobs, int) and self.n_jobs < 1:
            raise ValueError("neighbour_signal.n_jobs must be 'auto' or a positive integer")
        if not math.isfinite(self.automatic_positive_threshold):
            raise ValueError(
                "neighbour_signal.automatic_positive_threshold must be finite"
            )
        if self.source_anchor_dilation_px > self.max_halo_px:
            raise ValueError(
                "neighbour_signal.source_anchor_dilation_px cannot exceed max_halo_px"
            )
        if (
            self.exemplar_mode in {"automatic", "augment"}
            and self.automatic_target_exemplars_per_marker < self.min_exemplars
        ):
            raise ValueError(
                "neighbour_signal.automatic_target_exemplars_per_marker cannot be below min_exemplars"
            )
        output = self.output_adata_path.strip()
        if not output:
            raise ValueError("neighbour_signal.output_adata_path cannot be empty")
        if not output.lower().endswith(".h5ad"):
            raise ValueError("neighbour_signal.output_adata_path must end with .h5ad")
        source_table = self.source_target_table_path.strip()
        if not source_table:
            raise ValueError("neighbour_signal.source_target_table_path cannot be empty")
        if not source_table.lower().endswith((".parquet", ".pq")):
            raise ValueError(
                "neighbour_signal.source_target_table_path must end with .parquet or .pq"
            )
        if not self.exemplar_obs.strip() or not self.object_id_obs.strip():
            raise ValueError("neighbour_signal observation column names cannot be empty")
        if self.qc_markers is not None:
            cleaned = [str(marker).strip() for marker in self.qc_markers]
            if any(not marker for marker in cleaned):
                raise ValueError("neighbour_signal.qc_markers cannot contain empty names")
            if len({marker.casefold() for marker in cleaned}) != len(cleaned):
                raise ValueError("neighbour_signal.qc_markers must be unique case-insensitively")
        return self


@config_section("batch_integration")
class BatchIntegrationConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional AnnData input path for batch integration; use null to read "
            "general.anndata_path through the pipeline's normal stage-state checks."
        ),
    )
    output_adata_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional destination for the integrated AnnData; use null to update "
            "general.anndata_path with the new PCA, integration, neighbour, UMAP, Leiden, and "
            "provenance entries."
        ),
    )

    # Core integration settings
    batch_correction_obs: Optional[str] = Field(
        default=None,
        description=(
            "AnnData obs column defining the technical batches to balance or correct. It is "
            "required for harmony, bbknn, and both, and should not be a biological variable "
            "confounded with the comparison of interest."
        ),
    )
    integration_method: str = Field(
        default='harmony',
        description=(
            "Integration strategy: harmony corrects PCA coordinates; bbknn constructs a "
            "batch-balanced neighbour graph from uncorrected PCA; both applies Harmony then "
            "BBKNN; none uses ordinary Scanpy neighbours without batch correction."
        ),
    )
    batch_correction_method: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated alias for integration_method retained for older YAML files. When set, "
            "this value overrides integration_method; new configurations should leave it null."
        ),
    )
    n_for_pca: Optional[int] = Field(
        default=None,
        description=(
            "Number of principal components recomputed from the current AnnData matrix and used "
            "for integration. Null requests markers minus one, clipped to the valid range set by "
            "cell and marker counts."
        ),
    )
    leiden_resolutions_list: List[float] = Field(
        default_factory=lambda: [0.3, 1.0],
        description=(
            "Leiden resolution values evaluated on the final neighbour graph; each value creates "
            "adata.obs['leiden_<resolution>'], with larger values generally producing more "
            "clusters."
        ),
    )
    umap_min_dist: float = Field(
        default=0.1,
        description=(
            "UMAP minimum-distance parameter used after integration; lower values permit tighter "
            "visual clusters, while higher values spread local neighbourhoods more broadly. It "
            "does not change the neighbour graph."
        ),
    )
    run_leiden: bool = Field(
        default=True,
        description=(
            "Run Leiden clustering for every configured resolution after constructing the final "
            "graph; disable to retain integration and UMAP without writing new Leiden labels."
        ),
    )
    n_neighbors: Optional[int] = Field(
        default=None,
        description=(
            "Target neighbourhood size. Harmony and none pass it to Scanpy neighbours; BBKNN "
            "converts it to ceil(n_neighbors / number_of_batches) neighbours per batch unless "
            "bbknn_params sets neighbors_within_batch. Null uses each library's default."
        ),
    )

    # Embedding storage
    pca_key: str = Field(
        default='X_pca',
        description=(
            "AnnData obsm key read as the PCA input. Keep X_pca for normal runs because the stage "
            "recomputes PCA with Scanpy and Scanpy writes the result to that key."
        ),
    )
    harmony_key: str = Field(
        default='X_pca_harmony',
        description=(
            "AnnData obsm key that receives Harmony-corrected principal-component coordinates in "
            "harmony or both mode; marker values in adata.X are not corrected."
        ),
    )
    representation_key: str = Field(
        default='X_batch_integration',
        description=(
            "Canonical AnnData obsm key used for downstream graph construction: it stores copied "
            "PCA coordinates for bbknn or none, and corrected coordinates for harmony or both."
        ),
    )
    qc_output_subdir: str = Field(
        default='BatchIntegration',
        description=(
            "Subdirectory below the active QC/report location for batch-coloured and Leiden UMAP "
            "figures. It does not affect reusable AnnData output paths."
        ),
    )

    # Method-specific parameters
    harmony_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'max_iter_harmony': 30,
            'verbose': True,
            'random_state': 0,
            'device': None,
        },
        description=(
            "Keyword arguments passed directly to harmonypy.run_harmony after null-like values "
            "are removed. Defaults allow 30 Harmony rounds, log progress, and seed stochastic "
            "steps with 0; advanced options include theta, sigma, lamb, and convergence settings."
        ),
    )
    bbknn_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed to scanpy.external.pp.bbknn after null-like values are "
            "removed. The stage supplies use_rep and n_pcs when absent; common advanced options "
            "include neighbors_within_batch, trim, metric, and approximate-neighbour settings."
        ),
    )

    @model_validator(mode="after")
    def _apply_legacy_aliases(self):
        if self.batch_correction_method is not None:
            self.integration_method = str(self.batch_correction_method)
            logging.warning(
                "Deprecated batch integration key 'batch_correction_method' detected. "
                "Please update config.yaml to use 'integration_method' under 'batch_integration'."
            )

        method = str(self.integration_method).strip().lower()
        if method in {'', 'null', 'none'}:
            method = 'none'
        self.integration_method = method


        return self


@config_section("rapids")
class RapidsProcessConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional source AnnData path for RAPIDS processing; use null to read "
            "general.anndata_path through the pipeline's normal stage-state checks."
        ),
    )
    output_adata_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional destination for the processed AnnData; use null to update "
            "general.anndata_path with any filtering plus the new representations, graph, UMAP, "
            "Leiden labels, and provenance."
        ),
    )

    # Core RAPIDS processing settings
    batch_correction_obs: Optional[str] = Field(
        default=None,
        description=(
            "AnnData obs column defining technical batches for GPU Harmony. It is required when "
            "run_harmony is true; when supplied without Harmony it is still validated and used to "
            "colour QC UMAPs."
        ),
    )
    run_harmony: bool = Field(
        default=False,
        description=(
            "Run RAPIDS-singlecell Harmony on newly computed PCA coordinates before neighbours. "
            "This changes an embedding, not marker values, and cannot be combined with "
            "input_representation_key."
        ),
    )
    harmony_flavor: str = Field(
        default='harmony2',
        description=(
            "GPU Harmony algorithm: harmony2 uses the stabilized diversity penalty, dynamic "
            "cluster-by-batch regularization, and batch pruning; harmony1 reproduces the original "
            "Harmony formulation."
        ),
    )
    n_for_pca: Optional[int] = Field(
        default=None,
        description=(
            "Number of GPU principal components computed from adata.X or pca_params.layer. Null "
            "requests markers minus one, clipped to the valid range set by cell and marker counts; "
            "ignored when using an existing representation."
        ),
    )
    n_pcs_neighbors: Optional[int] = Field(
        default=None,
        description=(
            "Number of leading columns from the active PCA, Harmony, or existing representation "
            "used to construct neighbours. Null uses every newly computed component, or lets "
            "RAPIDS choose when input_representation_key is set."
        ),
    )
    leiden_resolutions_list: List[float] = Field(
        default_factory=lambda: [0.3, 1.0],
        description=(
            "Leiden resolution values evaluated on the RAPIDS neighbour graph; each creates "
            "adata.obs['leiden_<resolution>'], with larger values generally yielding more graph "
            "communities."
        ),
    )
    umap_min_dist: float = Field(
        default=0.1,
        description=(
            "Minimum separation permitted between nearby points in the RAPIDS UMAP display. Lower "
            "values make visually tighter islands; this setting does not change the neighbour "
            "graph or Leiden labels."
        ),
    )
    run_leiden: bool = Field(
        default=True,
        description=(
            "Run GPU Leiden community detection at every configured resolution after neighbour "
            "construction; disable to retain the representation, graph, and UMAP without new "
            "cluster labels."
        ),
    )
    n_neighbors: Optional[int] = Field(
        default=None,
        description=(
            "Number of cells in each RAPIDS local neighbourhood. Smaller values emphasize fine "
            "local structure; larger values produce a more global graph. Null uses the installed "
            "library default, currently 15."
        ),
    )

    # Optional obs-based cell filter applied immediately after loading AnnData
    filter_obs_key: str = Field(
        default='mask_area',
        description=(
            "Numeric AnnData obs column used for optional pre-analysis cell filtering. The filter "
            "is disabled while both bounds are null; non-numeric or missing values are removed "
            "when it is active."
        ),
    )
    filter_min_value: Optional[float] = Field(
        default=None,
        description=(
            "Inclusive minimum accepted value in filter_obs_key; use null for no lower bound. "
            "With the default mask_area key, this can exclude small segmentation fragments but "
            "may also remove genuinely small cells."
        ),
    )
    filter_max_value: Optional[float] = Field(
        default=None,
        description=(
            "Inclusive maximum accepted value in filter_obs_key; use null for no upper bound. "
            "With mask_area, this can exclude large merged masks but may also remove genuinely "
            "large or multinucleated cells."
        ),
    )

    # Optional parameter scan. Values should be lists keyed by supported scan
    # parameters: n_neighbors, n_for_pca, umap_min_dist, run_harmony, harmony_flavor.
    parameter_scan_dict: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Lists of values expanded as a Cartesian-product scan. Supported keys are n_neighbors, "
            "n_for_pca, umap_min_dist, run_harmony, and harmony_flavor; an empty dictionary runs "
            "one normal analysis."
        ),
    )
    parameter_scan_save_anndata: bool = Field(
        default=False,
        description=(
            "Save a separately suffixed AnnData for every parameter-scan combination. When false, "
            "only QC and the scan summary are retained; scan mode never writes the normal canonical "
            "output."
        ),
    )
    parameter_scan_qc_subdir: str = Field(
        default='ParameterScan',
        description=(
            "Subdirectory below the RAPIDS QC/report directory containing one folder per scan "
            "combination and rapids_parameter_scan_summary.csv."
        ),
    )

    # Embedding / graph storage
    input_representation_key: Optional[str] = Field(
        default=None,
        description=(
            "Existing adata.obsm embedding used directly for neighbours, bypassing RAPIDS PCA and "
            "Harmony. It cannot be combined with run_harmony; n_for_pca and pca_params then have "
            "no effect."
        ),
    )
    pca_key: str = Field(
        default='X_pca',
        description=(
            "AnnData obsm key receiving newly computed GPU PCA coordinates when no existing input "
            "representation is selected."
        ),
    )
    harmony_key: str = Field(
        default='X_pca_harmony',
        description=(
            "AnnData obsm key receiving GPU Harmony-corrected PCA coordinates when run_harmony is "
            "enabled; adata.X and marker layers remain uncorrected."
        ),
    )
    representation_key: str = Field(
        default='X_batch_integration',
        description=(
            "Canonical AnnData obsm key copied from the active existing, PCA, or Harmony "
            "representation and used for RAPIDS neighbour construction and downstream stages."
        ),
    )
    neighbors_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional name for a separate RAPIDS neighbour graph and its distance/connectivity "
            "matrices; null writes the standard AnnData neighbors and obsp keys, replacing their "
            "current contents."
        ),
    )
    umap_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional AnnData obsm key for the RAPIDS UMAP embedding; null writes the standard "
            "X_umap key and replaces an existing UMAP."
        ),
    )
    qc_output_subdir: str = Field(
        default='RapidsProcess',
        description=(
            "Subdirectory below the active QC/report location for RAPIDS UMAPs, Leiden MatrixPlots, "
            "and optional parameter-scan outputs."
        ),
    )

    # RAPIDS pass-through parameters. Keys controlled by the config above
    # (e.g. n_comps, key_added, use_rep) are intentionally ignored by the script.
    pca_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional arguments for rapids_singlecell.pp.pca after null-like values are removed. "
            "Use layer to select an AnnData layer; n_comps, key_added, and copy are controlled by "
            "first-class fields and ignored here."
        ),
    )
    harmony_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'max_iter_harmony': 30,
            'random_state': 0,
            'verbose': True,
            'dtype': 'float32',
        },
        description=(
            "Additional GPU Harmony arguments after null-like values are removed. Defaults use up "
            "to 30 rounds, seed 0, progress logging, and memory-efficient float32; float64 is more "
            "numerically stable but consumes more GPU memory."
        ),
    )
    neighbors_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional arguments for rapids_singlecell.pp.neighbors, such as algorithm, metric, "
            "method, random_state, and algorithm_kwds. Dedicated fields override n_neighbors, "
            "n_pcs, use_rep, key_added, and copy."
        ),
    )
    umap_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional arguments for rapids_singlecell.tl.umap, such as spread, n_components, "
            "maxiter, init_pos, and random_state. Dedicated fields override min_dist, key_added, "
            "neighbors_key, and copy."
        ),
    )
    leiden_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional arguments for rapids_singlecell.tl.leiden, such as random_state, theta, "
            "n_iterations, use_weights, and dtype. Dedicated fields override resolution, key_added, "
            "neighbors_key, and copy."
        ),
    )


class MaxFuseGeneListConfig(ConfigModel):
    """One optional gene-list file used to annotate MaxFuse DEG results."""

    path: str = Field(min_length=1)
    name: Optional[str] = None
    format: Literal["auto", "wide", "long"] = "auto"
    gene_column: str = "gene"
    group_column: str = "group"


@config_section("maxfuse")
class MaxFuseConfig(ConfigModel):
    """Configuration for one-reference MaxFuse matching and reporting."""

    enabled: bool = config_field(
        True,
        description="Run the MaxFuse cross-modality matching stage.",
        level="basic",
        stage="maxfuse",
        ui_group="Stage control",
    )
    reference_name: str = config_field(
        "atlas",
        description="Stable short name used to prefix transferred reference annotations.",
        level="basic",
        stage="maxfuse",
        ui_group="Inputs",
        min_length=1,
    )
    reference_adata_path: str = config_field(
        "maxfuse/reference.h5ad",
        description="Project-relative AnnData path for the single scRNA-seq reference.",
        level="basic",
        stage="maxfuse",
        ui_group="Inputs",
        advice="Exactly one reference is supported per execution.",
        min_length=1,
    )
    target_adata_path: Optional[str] = config_field(
        None,
        description="Optional target IMC AnnData path; defaults to general.anndata_path.",
        level="basic",
        stage="maxfuse",
        ui_group="Inputs",
    )
    feature_mapping_path: str = config_field(
        "maxfuse/feature_mapping.csv",
        description="CSV mapping measured target features to linked reference genes.",
        level="basic",
        stage="maxfuse",
        ui_group="Inputs",
        min_length=1,
    )
    asset_folder: str = config_field(
        "maxfuse_results",
        description="Canonical project-relative folder for reusable MaxFuse result assets.",
        level="basic",
        stage="maxfuse",
        ui_group="Reusable outputs",
        advice="Execution figures and diagnostic tables are stored separately in the managed report.",
        min_length=1,
    )
    target_feature_column: str = config_field(
        "IMC",
        description="Feature-mapping column containing target IMC marker names.",
        stage="maxfuse",
        ui_group="Feature mapping",
        min_length=1,
    )
    reference_feature_column: str = config_field(
        "snRNAseq",
        description="Feature-mapping column containing linked scRNA-seq gene names.",
        stage="maxfuse",
        ui_group="Feature mapping",
        min_length=1,
    )
    mapping_filter_column: Optional[str] = config_field(
        "All",
        description="Optional mapping column whose truthy rows are eligible for matching.",
        stage="maxfuse",
        ui_group="Feature mapping",
    )
    reference_layer: Optional[str] = config_field(
        None,
        description="Optional reference AnnData layer used for matching, expression plots, and DEGs.",
        stage="maxfuse",
        ui_group="Inputs",
    )
    target_layer: Optional[str] = config_field(
        None,
        description="Optional target AnnData layer used for matching.",
        stage="maxfuse",
        ui_group="Inputs",
    )
    reference_smoothing_obs: Optional[str] = config_field(
        "annotation_level_3",
        description="Reference observation labels used only for within-modality centroid shrinkage.",
        level="basic",
        stage="maxfuse",
        ui_group="Labels",
    )
    target_smoothing_obs: Optional[str] = config_field(
        None,
        description="Target observation labels used for centroid shrinkage; defaults to target_population_obs.",
        stage="maxfuse",
        ui_group="Labels",
    )
    target_population_obs: Optional[str] = config_field(
        None,
        description="Target population column used in matching reports; defaults to general.population_obs_primary.",
        level="basic",
        stage="maxfuse",
        ui_group="Labels",
    )
    reference_transfer_obs: List[str] = config_field(
        description="Reference observation columns transferred to matched target cells.",
        level="basic",
        stage="maxfuse",
        ui_group="Labels",
        default_factory=lambda: ["annotation_level_3", "annotation_level_4"],
    )
    reference_label_roles: Dict[str, Literal["lineage", "subtype", "state", "other"]] = config_field(
        description="Optional biological roles for transferred reference label columns.",
        stage="maxfuse",
        ui_group="Labels",
        default_factory=dict,
    )
    sample_obs: Optional[str] = config_field(
        None,
        description="Optional target sample column for coverage diagnostics; defaults to general.case_obs.",
        stage="maxfuse",
        ui_group="Labels",
    )
    roi_obs: Optional[str] = config_field(
        None,
        description="Optional target ROI column for coverage diagnostics; defaults to general.roi_obs.",
        stage="maxfuse",
        ui_group="Labels",
    )
    reference_active_features: int = config_field(
        1000,
        description="Maximum number of reference active features used during refinement.",
        level="basic",
        stage="maxfuse",
        ui_group="Feature preparation",
        ge=2,
    )
    reference_shared_sd_min: float = config_field(
        0.20,
        description="Minimum reference standard deviation for a linked feature.",
        stage="maxfuse",
        ui_group="Feature preparation",
        ge=0,
    )
    target_shared_sd_min: float = config_field(
        0.025,
        description="Minimum target standard deviation for a linked feature.",
        stage="maxfuse",
        ui_group="Feature preparation",
        ge=0,
    )
    min_shared_features: int = config_field(
        20,
        description="Minimum linked features required after availability and variability filtering.",
        level="basic",
        stage="maxfuse",
        ui_group="Feature preparation",
        ge=2,
    )
    zscore_reference: bool = config_field(
        True,
        description="Z-score reference active and linked features before matching.",
        stage="maxfuse",
        ui_group="Feature preparation",
    )
    zscore_target: bool = config_field(
        False,
        description="Z-score target features; leave false for already scaled SBT IMC data.",
        stage="maxfuse",
        ui_group="Feature preparation",
    )
    batching_scheme: Literal["cyclic", "pairwise"] = config_field(
        "cyclic",
        description="MaxFuse batch pairing scheme.",
        stage="maxfuse",
        ui_group="Matching",
    )
    max_outward_size: int = config_field(
        2000,
        description="Maximum smaller-modality batch size used in assignment problems.",
        level="basic",
        stage="maxfuse",
        ui_group="Matching",
        ge=100,
    )
    matching_ratio: int = config_field(
        4,
        description="Approximate target-to-reference candidate ratio within MaxFuse batches.",
        stage="maxfuse",
        ui_group="Matching",
        ge=1,
    )
    metacell_size: int = config_field(
        2,
        description="Average reference metacell size.",
        stage="maxfuse",
        ui_group="Matching",
        ge=1,
    )
    n_neighbors_reference: int = config_field(
        15,
        description="Reference within-modality graph neighbour count.",
        stage="maxfuse",
        ui_group="Graph construction",
        ge=2,
    )
    n_neighbors_target: int = config_field(
        15,
        description="Target within-modality graph neighbour count.",
        stage="maxfuse",
        ui_group="Graph construction",
        ge=2,
    )
    graph_svd_reference: int = config_field(
        40,
        description="Reference active-space SVD components for graph construction.",
        stage="maxfuse",
        ui_group="Graph construction",
        ge=2,
    )
    graph_svd_target: Optional[int] = config_field(
        20,
        description="Target active-space SVD components for graph construction.",
        stage="maxfuse",
        ui_group="Graph construction",
        ge=2,
    )
    graph_resolution_reference: float = config_field(
        2.0,
        description="Reference graph clustering resolution.",
        stage="maxfuse",
        ui_group="Graph construction",
        gt=0,
    )
    graph_resolution_target: float = config_field(
        2.0,
        description="Target graph clustering resolution.",
        stage="maxfuse",
        ui_group="Graph construction",
        gt=0,
    )
    graph_resolution_tolerance: float = config_field(
        0.1,
        description="Resolution difference below which MaxFuse merges graph communities.",
        stage="maxfuse",
        ui_group="Graph construction",
        ge=0,
    )
    initial_weight_reference: float = config_field(
        0.3,
        description="Raw-reference weight during initial fuzzy smoothing.",
        stage="maxfuse",
        ui_group="Initial pivots",
        ge=0,
        le=1,
    )
    initial_weight_target: float = config_field(
        0.3,
        description="Raw-target weight during initial fuzzy smoothing.",
        stage="maxfuse",
        ui_group="Initial pivots",
        ge=0,
        le=1,
    )
    initial_svd_reference: int = config_field(
        20,
        description="Reference linked-space SVD components for initial pivots.",
        stage="maxfuse",
        ui_group="Initial pivots",
        ge=2,
    )
    initial_svd_target: int = config_field(
        20,
        description="Target linked-space SVD components for initial pivots.",
        stage="maxfuse",
        ui_group="Initial pivots",
        ge=2,
    )
    refine_weight_reference: float = config_field(
        0.3,
        description="Raw-reference weight during pivot refinement.",
        stage="maxfuse",
        ui_group="Pivot refinement",
        ge=0,
        le=1,
    )
    refine_weight_target: float = config_field(
        0.3,
        description="Raw-target weight during pivot refinement.",
        stage="maxfuse",
        ui_group="Pivot refinement",
        ge=0,
        le=1,
    )
    refine_svd_reference: int = config_field(
        40,
        description="Reference active-space SVD components during refinement and propagation.",
        stage="maxfuse",
        ui_group="Pivot refinement",
        ge=2,
    )
    refine_svd_target: Optional[int] = config_field(
        None,
        description="Optional target active-space SVD components during refinement.",
        stage="maxfuse",
        ui_group="Pivot refinement",
        ge=2,
    )
    refine_cca_components: int = config_field(
        25,
        description="CCA components used for pivot refinement.",
        stage="maxfuse",
        ui_group="Pivot refinement",
        ge=2,
    )
    refine_iterations: int = config_field(
        1,
        description="Pivot-refinement iterations; one is the historical spatial-omics default.",
        level="basic",
        stage="maxfuse",
        ui_group="Pivot refinement",
        ge=1,
    )
    pivot_filter_fraction: float = config_field(
        0.5,
        description="Fraction of weak refined pivots discarded.",
        stage="maxfuse",
        ui_group="Filtering and propagation",
        ge=0,
        le=1,
    )
    propagation_weight_reference: float = config_field(
        0.7,
        description="Raw-reference weight during propagation.",
        stage="maxfuse",
        ui_group="Filtering and propagation",
        ge=0,
        le=1,
    )
    propagation_weight_target: float = config_field(
        0.7,
        description="Raw-target weight during propagation.",
        stage="maxfuse",
        ui_group="Filtering and propagation",
        ge=0,
        le=1,
    )
    propagated_filter_fraction: float = config_field(
        0.3,
        description="Fraction of weak propagated matches discarded.",
        stage="maxfuse",
        ui_group="Filtering and propagation",
        ge=0,
        le=1,
    )
    randomized_svd: bool = config_field(
        True,
        description="Use randomized SVD for large active matrices.",
        stage="maxfuse",
        ui_group="Performance",
    )
    seed: int = config_field(
        100,
        description="Random seed for batching, SVD, and deterministic plotting samples.",
        level="basic",
        stage="maxfuse",
        ui_group="Performance",
        ge=0,
    )
    report_score_threshold: float = config_field(
        0.30,
        description="Minimum score used for default plots and DEG analysis; reusable matches remain unfiltered.",
        level="basic",
        stage="maxfuse",
        ui_group="Reporting",
        ge=0,
        le=1,
    )
    sensitivity_thresholds: List[float] = config_field(
        description="Post-hoc score thresholds used for coverage reporting without refitting MaxFuse.",
        stage="maxfuse",
        ui_group="Reporting",
        default_factory=lambda: [0.3, 0.5, 0.7, 0.8, 0.85, 0.9],
    )
    reference_umap_key: str = config_field(
        "X_umap",
        description="Reference embedding key used for UMAP report panels.",
        stage="maxfuse",
        ui_group="Reporting",
        min_length=1,
    )
    target_umap_key: str = config_field(
        "X_umap",
        description="Target embedding key used for UMAP report panels.",
        stage="maxfuse",
        ui_group="Reporting",
        min_length=1,
    )
    max_umap_points: int = config_field(
        200000,
        description="Maximum deterministically sampled points in each UMAP report.",
        stage="maxfuse",
        ui_group="Reporting",
        ge=1000,
    )
    figure_dpi: int = config_field(
        300,
        description="Raster resolution used for MaxFuse report figures.",
        stage="maxfuse",
        ui_group="Reporting",
        ge=72,
    )
    figure_formats: List[Literal["png", "svg"]] = config_field(
        description="Figure formats written for each MaxFuse diagnostic.",
        stage="maxfuse",
        ui_group="Reporting",
        default_factory=lambda: ["png", "svg"],
    )
    plot_stacked_violin: bool = config_field(
        True,
        description="Generate a linked-gene stacked-violin companion to the matrix plot.",
        stage="maxfuse",
        ui_group="Reporting",
    )
    run_degs: bool = config_field(
        True,
        description="Run Wilcoxon one-versus-rest DEGs on matched reference cells.",
        level="basic",
        stage="maxfuse",
        ui_group="Differential expression",
    )
    deg_top_genes: int = config_field(
        20,
        description="Top genes displayed per transferred target population.",
        stage="maxfuse",
        ui_group="Differential expression",
        ge=1,
    )
    deg_min_cells: int = config_field(
        20,
        description="Minimum unique matched reference cells required per DEG group.",
        stage="maxfuse",
        ui_group="Differential expression",
        ge=2,
    )
    gene_lists: List[MaxFuseGeneListConfig] = config_field(
        description="Optional wide or long gene-list files used to annotate and enrich DEG results.",
        level="basic",
        stage="maxfuse",
        ui_group="Differential expression",
        default_factory=list,
    )
    write_matched_transcriptomes: bool = config_field(
        False,
        description="Write an optional target-indexed matched-transcriptome AnnData asset.",
        stage="maxfuse",
        ui_group="Reusable outputs",
        advice="This can be very large because repeated reference matches are materialized.",
    )
    matched_transcriptome_min_score: float = config_field(
        0.85,
        description="Score threshold for the optional matched-transcriptome asset.",
        stage="maxfuse",
        ui_group="Reusable outputs",
        ge=0,
        le=1,
    )

    @model_validator(mode="after")
    def _validate_maxfuse(self) -> "MaxFuseConfig":
        if not self.reference_transfer_obs:
            raise ValueError("reference_transfer_obs must contain at least one column")
        if len(set(self.reference_transfer_obs)) != len(self.reference_transfer_obs):
            raise ValueError("reference_transfer_obs must not contain duplicates")
        unknown_roles = sorted(
            set(self.reference_label_roles) - set(self.reference_transfer_obs)
        )
        if unknown_roles:
            raise ValueError(
                "reference_label_roles contains columns absent from "
                f"reference_transfer_obs: {unknown_roles}"
            )
        if not self.sensitivity_thresholds:
            raise ValueError("sensitivity_thresholds must not be empty")
        if any(value < 0 or value > 1 for value in self.sensitivity_thresholds):
            raise ValueError("sensitivity_thresholds values must be between 0 and 1")
        if len(set(self.figure_formats)) != len(self.figure_formats):
            raise ValueError("figure_formats must not contain duplicates")
        return self


class SpatialDataImagePanelConfig(ConfigModel):
    """One explicitly configured standalone ROI-by-channel image panel."""

    name: str = Field(min_length=1)
    folder: str = Field(min_length=1)
    panel_name: Optional[str] = None
    channels: List[str] = Field(default_factory=list)
    reference: str = "cells"
    allow_partial: bool = True


class SpatialDataHistologyConfig(ConfigModel):
    """One explicitly configured histology image collection."""

    name: str = Field(min_length=1)
    folder: str = Field(min_length=1)
    suffix: str = ""
    reference: str = "cells"
    allow_partial: bool = True
    drop_alpha: bool = False


class SpatialDataRegionLabelsConfig(ConfigModel):
    """One categorical raster collection and its value-to-name mapping."""

    name: str = Field(min_length=1)
    folder: str = Field(min_length=1)
    mapping_path: str = Field(min_length=1)
    suffix: str = ""
    reference: str = "cells"
    allow_partial: bool = True
    value_key: str = "label_value"
    name_key: str = "label_name"
    mapping_roi_key: Optional[str] = None


class SpatialDataMaxFuseTableConfig(ConfigModel):
    """One matched transcriptome AnnData linked to the primary IMC table."""

    name: str = Field(min_length=1)
    adata_path: str = Field(min_length=1)
    imc_table: str = "cell_quantification"
    table_name: Optional[str] = None
    copy_adata: bool = True


@config_section("spatialdata")
class SpatialDataConfig(ConfigModel):
    """Configuration for discovery, planning, and SpatialData Zarr creation."""

    action: Literal["plan", "build"] = config_field(
        "plan",
        description="Run read-only discovery/planning or also create a new Zarr store.",
        level="basic",
        stage="spatialdata",
        ui_group="Stage control",
        advice="Keep plan until the selected assets and planner diagnostics have been reviewed.",
    )
    root: str = config_field(
        ".",
        description="Project-relative root searched for likely spatial assets.",
        level="basic",
        stage="spatialdata",
        ui_group="Discovery",
    )
    output_path: str = config_field(
        "spatialdata.zarr",
        description="Project-relative destination for a newly built SpatialData Zarr.",
        level="basic",
        stage="spatialdata",
        ui_group="Output",
        advice="Existing paths are never overwritten.",
        min_length=1,
    )
    anndata_path: Optional[str] = config_field(
        None,
        description="Explicit primary quantified IMC AnnData; null allows conservative discovery.",
        level="basic",
        stage="spatialdata",
        ui_group="Primary assets",
    )
    cell_masks_folder: Optional[str] = config_field(
        None,
        description="Explicit folder containing one integer cell mask per ROI.",
        level="basic",
        stage="spatialdata",
        ui_group="Primary assets",
    )
    primary_images_folder: Optional[str] = config_field(
        None,
        description="Explicit folder containing one channel-image subfolder per ROI.",
        level="basic",
        stage="spatialdata",
        ui_group="Primary assets",
    )
    primary_table_name: str = config_field(
        "cell_quantification",
        description="Stable modality name for the primary quantified cell table.",
        stage="spatialdata",
        ui_group="Naming",
        min_length=1,
    )
    primary_images_name: str = config_field(
        "cell_images",
        description="Stable modality name for the primary IMC image panel.",
        stage="spatialdata",
        ui_group="Naming",
        min_length=1,
    )
    cell_masks_name: str = config_field(
        "cells",
        description="Stable modality name for the primary cell masks.",
        stage="spatialdata",
        ui_group="Naming",
        min_length=1,
    )
    primary_panel_name: str = config_field(
        "IMC panel",
        description="Human-readable name for the quantified IMC panel.",
        stage="spatialdata",
        ui_group="Naming",
        min_length=1,
    )
    roi_key: Optional[str] = config_field(
        None,
        description="AnnData ROI column; null falls back to general.roi_obs.",
        stage="spatialdata",
        ui_group="AnnData columns",
    )
    instance_key: str = config_field(
        "ObjectNumber",
        description="ROI-local integer mask-instance column in the primary AnnData.",
        level="basic",
        stage="spatialdata",
        ui_group="AnnData columns",
        min_length=1,
    )
    x_key: Optional[str] = config_field(
        None,
        description="AnnData X-coordinate column; null falls back to general.x_coord_obs.",
        stage="spatialdata",
        ui_group="AnnData columns",
    )
    y_key: Optional[str] = config_field(
        None,
        description="AnnData Y-coordinate column; null falls back to general.y_coord_obs.",
        stage="spatialdata",
        ui_group="AnnData columns",
    )
    copy_adata: bool = config_field(
        False,
        description="Copy the primary AnnData before SpatialData table annotation.",
        stage="spatialdata",
        ui_group="Memory",
        advice="False avoids a second large in-memory copy while leaving the source H5AD unchanged.",
    )
    discover_unlisted_assets: bool = config_field(
        True,
        description="Search for additional likely assets beyond explicitly configured paths.",
        level="basic",
        stage="spatialdata",
        ui_group="Discovery",
    )
    include_discovered_image_panels: bool = config_field(
        True,
        description="Include high-confidence unquantified ROI image panels in the proposed spec.",
        stage="spatialdata",
        ui_group="Discovery",
    )
    include_discovered_histology: bool = config_field(
        True,
        description="Include high-confidence RGB/RGBA ROI collections in the proposed spec.",
        stage="spatialdata",
        ui_group="Discovery",
    )
    include_discovered_maxfuse: bool = config_field(
        True,
        description="Include high-confidence matched transcriptome AnnData candidates.",
        stage="spatialdata",
        ui_group="Discovery",
    )
    additional_image_panels: List[SpatialDataImagePanelConfig] = config_field(
        default_factory=list,
        description="Explicit standalone ROI-by-channel image panels.",
        stage="spatialdata",
        ui_group="Additional assets",
    )
    histology: List[SpatialDataHistologyConfig] = config_field(
        default_factory=list,
        description="Explicit histology image collections.",
        stage="spatialdata",
        ui_group="Additional assets",
    )
    region_labels: List[SpatialDataRegionLabelsConfig] = config_field(
        default_factory=list,
        description="Explicit categorical label rasters with semantic mapping tables.",
        stage="spatialdata",
        ui_group="Additional assets",
    )
    maxfuse_tables: List[SpatialDataMaxFuseTableConfig] = config_field(
        default_factory=list,
        description="Explicit matched-transcriptome AnnData tables.",
        stage="spatialdata",
        ui_group="Additional assets",
    )
    attrs: Dict[str, Any] = config_field(
        default_factory=dict,
        description="Small project-level attributes to place on the SpatialData object.",
        stage="spatialdata",
        ui_group="Output",
    )
    raster_chunks: Tuple[int, int] = config_field(
        (512, 512),
        description="Y/X Dask chunk dimensions used for raster elements.",
        stage="spatialdata",
        ui_group="Performance",
    )
    scale_factors: Optional[List[int]] = config_field(
        None,
        description="Optional multiscale pyramid factors passed to SpatialData raster parsers.",
        stage="spatialdata",
        ui_group="Performance",
    )
    scan_depth: int = config_field(
        3,
        description="Maximum recursive folder depth for discovery.",
        stage="spatialdata",
        ui_group="Discovery bounds",
        ge=0,
    )
    max_scan_entries: int = config_field(
        50_000,
        description="Maximum files and folders inspected during discovery.",
        stage="spatialdata",
        ui_group="Discovery bounds",
        ge=1,
    )
    sample_files: int = config_field(
        5,
        description="Representative files or ROI folders inspected per candidate collection.",
        stage="spatialdata",
        ui_group="Discovery bounds",
        ge=1,
    )

    @model_validator(mode="after")
    def _validate_spatialdata(self) -> "SpatialDataConfig":
        if not self.output_path.casefold().endswith(".zarr"):
            raise ValueError("spatialdata.output_path must end with .zarr")
        if any(value <= 0 for value in self.raster_chunks):
            raise ValueError("spatialdata.raster_chunks values must be positive")
        if self.scale_factors is not None and any(
            value <= 1 for value in self.scale_factors
        ):
            raise ValueError("spatialdata.scale_factors values must be greater than 1")
        names = [
            self.cell_masks_name,
            self.primary_images_name,
            self.primary_table_name,
            *(item.name for item in self.additional_image_panels),
            *(item.name for item in self.histology),
            *(item.name for item in self.region_labels),
            *(item.name for item in self.maxfuse_tables),
        ]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("SpatialData modality names must be unique case-insensitively")
        return self


@config_section("cellvision")
class CellVisionConfig(ConfigModel):
    """Configuration for single-cell image representation learning and clustering."""

    input_adata_path: Optional[str] = config_field(
        None,
        description="Optional CellVision source AnnData path; defaults to general.anndata_path.",
        level="basic",
        stage="cellvision",
        ui_group="Inputs and reusable assets",
        advice="Leave unset to use the project's canonical AnnData object.",
    )
    images_folder: Optional[str] = config_field(
        None,
        description="Optional ROI/channel image folder; defaults to general.denoised_images_folder.",
        level="basic",
        stage="cellvision",
        ui_group="Inputs and reusable assets",
        advice="The folder must contain one subdirectory per ROI with one TIFF per marker.",
    )
    masks_folder: Optional[str] = config_field(
        None,
        description="Optional labelled cell-mask folder; defaults to general.masks_folder.",
        level="basic",
        stage="cellvision",
        ui_group="Inputs and reusable assets",
        advice="Mask filenames must match ROI values and mask labels must match object_id_obs.",
    )
    asset_folder: str = config_field(
        "scPortrait/CellVision",
        description="Canonical project-relative folder for reusable CellVision assets.",
        level="basic",
        stage="cellvision",
        ui_group="Inputs and reusable assets",
        advice="Keep this outside general.outputs_folder; execution figures and tables are routed separately.",
        min_length=1,
    )
    roi_obs: Optional[str] = config_field(
        None,
        description="AnnData observation column containing ROI identifiers; defaults to general.roi_obs.",
        level="basic",
        stage="cellvision",
        ui_group="Cell identity and selection",
        advice="Values must match image subdirectory and mask filename stems exactly.",
    )
    object_id_obs: str = config_field(
        "ObjectNumber",
        description="AnnData observation column containing the integer label used in each ROI mask.",
        level="basic",
        stage="cellvision",
        ui_group="Cell identity and selection",
        advice="The pair (ROI, object ID) must uniquely identify every selected source cell.",
        min_length=1,
    )
    population_obs: Optional[str] = config_field(
        None,
        description="Optional source population annotation used for filtering, comparison, and plots.",
        level="basic",
        stage="cellvision",
        ui_group="Cell identity and selection",
        advice="Leave unset to analyse all cells without original-population comparison plots.",
    )
    populations: Optional[List[str]] = config_field(
        None,
        description="Optional population values to retain from population_obs.",
        level="basic",
        stage="cellvision",
        ui_group="Cell identity and selection",
        advice="Leave unset to retain every cell, even when population_obs is set for plotting.",
    )
    markers: Optional[List[str]] = config_field(
        None,
        description="Optional ordered marker/channel names to include in cell images and VICReg training.",
        level="basic",
        stage="cellvision",
        ui_group="Cell identity and selection",
        advice="Each name must match the case-insensitive suffix immediately before a TIFF extension; both 165Ho_CD11c and CD11c can match a prefixed IMC filename. Leave unset for all channels.",
    )

    image_size: int = config_field(
        36,
        description="Height and width in pixels of each extracted single-cell image.",
        level="basic",
        stage="cellvision",
        ui_group="scPortrait extraction",
        advice="The 36 px default is intended to retain even relatively large IMC cells while limiting model size.",
        ge=8,
        le=512,
    )
    extraction_threads: int = config_field(
        12,
        description="Worker processes used by scPortrait HDF5 cell extraction.",
        stage="cellvision",
        ui_group="scPortrait extraction",
        advice="Do not exceed the CPUs allocated by the CellVision SLURM wrapper.",
        ge=1,
    )
    mask_expand_px: int = config_field(
        0,
        description="Optional labelled-mask expansion distance before extracting cell portraits.",
        stage="cellvision",
        ui_group="scPortrait extraction",
        advice="Keep zero to train strictly on the segmented cell boundary.",
        ge=0,
    )
    mask_gaussian_blur: bool = config_field(
        False,
        description="Apply scPortrait's sigma-1 Gaussian blur to each extracted segmentation mask.",
        stage="cellvision",
        ui_group="scPortrait extraction",
        advice="Keep false for 1 um/pixel IMC; enable only when softened mask edges are scientifically justified.",
    )
    normalization_dict_path: Optional[str] = config_field(
        None,
        description="Optional preferred Nimbus normalization_dict.csv or legacy JSON containing one positive Vmax per selected marker or unambiguous marker suffix.",
        stage="cellvision",
        ui_group="Input normalization",
        advice="Exact keys take priority; short Nimbus keys such as CD11c match a selected 165Ho_CD11c channel, while ambiguous suffixes fail. Relative paths resolve from the project root.",
    )
    normalization_quantile: float = config_field(
        0.999,
        description="Per-ROI in-mask quantile averaged to compute each channel normalization value.",
        stage="cellvision",
        ui_group="Input normalization",
        advice="Matches the current Nimbus default and is used only when normalization_dict_path is unset.",
        gt=0.5,
        le=1.0,
    )
    normalization_min_value: float = config_field(
        3.0,
        description="Minimum computed normalization value used to avoid scaling background noise.",
        stage="cellvision",
        ui_group="Input normalization",
        advice="Matches the current Nimbus default; supplied dictionary values are preserved after positive-value validation.",
        gt=0,
    )
    normalization_clip: List[float] = config_field(
        default_factory=lambda: [0.0, 1.0],
        description="Lower and upper bounds applied after division by the channel normalization value.",
        stage="cellvision",
        ui_group="Input normalization",
        advice="CellVision H5SC training images must remain within [0, 1].",
    )
    overwrite: bool = config_field(
        False,
        description="Regenerate existing reusable CellVision assets instead of validating and reusing them.",
        stage="cellvision",
        ui_group="scPortrait extraction",
        advice="Enable only when inputs, selections, markers, or model settings have deliberately changed.",
    )

    encoder_width: int = config_field(
        32,
        description="Base channel width of the compact residual VICReg encoder.",
        stage="cellvision",
        ui_group="VICReg model",
        advice="Increase only when GPU memory and training-set size justify a larger encoder.",
        ge=8,
    )
    embedding_dim: int = config_field(
        256,
        description="Number of cell-level features emitted by the VICReg encoder.",
        level="basic",
        stage="cellvision",
        ui_group="VICReg model",
        advice="This is the representation saved for every extracted source cell.",
        ge=8,
    )
    projector_dim: int = config_field(
        512,
        description="Hidden and output width of the VICReg training projector.",
        stage="cellvision",
        ui_group="VICReg model",
        advice="The projector is used only for the self-supervised loss and is not exported as the cell embedding.",
        ge=16,
    )
    epochs: int = config_field(
        30,
        description="Number of self-supervised VICReg training epochs.",
        level="basic",
        stage="cellvision",
        ui_group="VICReg model",
        advice="Inspect the training-loss report before increasing this value.",
        ge=1,
    )
    batch_size: int = config_field(
        256,
        description="VICReg training and inference batch size.",
        level="basic",
        stage="cellvision",
        ui_group="VICReg model",
        advice="VICReg variance/covariance estimates benefit from larger batches; reduce for GPU memory pressure.",
        ge=2,
    )
    learning_rate: float = config_field(
        0.0003,
        description="Initial AdamW learning rate for VICReg training.",
        stage="cellvision",
        ui_group="VICReg model",
        advice="The schedule uses linear warmup followed by cosine decay.",
        gt=0,
    )
    weight_decay: float = config_field(
        0.000001,
        description="AdamW weight decay used during VICReg training.",
        stage="cellvision",
        ui_group="VICReg model",
        advice="Keep small so image morphology is not over-regularized.",
        ge=0,
    )
    warmup_epochs: int = config_field(
        3,
        description="Number of linear learning-rate warmup epochs before cosine decay.",
        stage="cellvision",
        ui_group="VICReg model",
        advice="Must not exceed epochs.",
        ge=0,
    )
    num_workers: int = config_field(
        4,
        description="PyTorch DataLoader worker processes for H5SC image reads.",
        stage="cellvision",
        ui_group="VICReg model",
        advice="Keep below the SLURM CPU allocation and reduce if the HDF5 filesystem is congested.",
        ge=0,
    )
    seed: int = config_field(
        0,
        description="Random seed used for selection fingerprints, augmentations, training, and galleries.",
        level="basic",
        stage="cellvision",
        ui_group="VICReg model",
        advice="Keep fixed when comparing marker sets or populations.",
        ge=0,
    )
    amp: bool = config_field(
        True,
        description="Use automatic mixed precision for CUDA VICReg training.",
        stage="cellvision",
        ui_group="VICReg model",
        advice="Disable when diagnosing numerical instability or using unsupported hardware.",
    )
    vicreg_invariance_weight: float = config_field(
        25.0,
        description="Weight of the VICReg invariance loss between augmented views.",
        stage="cellvision",
        ui_group="VICReg loss",
        advice="The default follows the standard VICReg balance.",
        gt=0,
    )
    vicreg_variance_weight: float = config_field(
        25.0,
        description="Weight of the VICReg per-dimension variance regularizer.",
        stage="cellvision",
        ui_group="VICReg loss",
        advice="The default follows the standard VICReg balance.",
        gt=0,
    )
    vicreg_covariance_weight: float = config_field(
        1.0,
        description="Weight of the VICReg off-diagonal covariance regularizer.",
        stage="cellvision",
        ui_group="VICReg loss",
        advice="The default follows the standard VICReg balance.",
        gt=0,
    )
    augmentation_translation_px: int = config_field(
        2,
        description="Maximum zero-filled integer translation applied to VICReg image views.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="Small translations preserve the complete 36 px cell crop without wraparound.",
        ge=0,
    )
    augmentation_horizontal_flip_probability: float = config_field(
        0.5,
        description="Probability of a horizontal flip for each VICReg view.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="Flips are pixel-preserving and do not interpolate low-resolution IMC data.",
        ge=0,
        le=1,
    )
    augmentation_vertical_flip_probability: float = config_field(
        0.5,
        description="Probability of a vertical flip for each VICReg view.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="Set to zero to disable this orientation augmentation.",
        ge=0,
        le=1,
    )
    augmentation_rotation_probability: float = config_field(
        1.0,
        description="Probability of applying a random 0/90/180/270-degree rotation.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="Only right-angle rotations are used, avoiding interpolation at 1 um/pixel.",
        ge=0,
        le=1,
    )
    augmentation_translation_probability: float = config_field(
        0.0,
        description="Probability of applying a zero-filled integer translation.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="Set to zero to disable translations while retaining the configured maximum distance.",
        ge=0,
        le=1,
    )
    augmentation_intensity_jitter: float = config_field(
        0.2,
        description="Maximum independent multiplicative intensity perturbation per marker channel.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="No hue, saturation, channel mixing, or artificial background is applied to multiplex IMC images.",
        ge=0,
        lt=1,
    )
    augmentation_intensity_jitter_probability: float = config_field(
        0.0,
        description="Probability of applying independent multiplicative marker jitter.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="The zero default preserves marker amplitude between views; increase deliberately to make the encoder less intensity-sensitive.",
        ge=0,
        le=1,
    )
    augmentation_noise_std: float = config_field(
        0.02,
        description="Standard deviation of Gaussian noise applied only on the configured spatial support.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="Background pixels remain exactly zero in both VICReg views.",
        ge=0,
    )
    augmentation_noise_probability: float = config_field(
        1.0,
        description="Probability of adding Gaussian noise to one augmented view.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="Set to zero to disable pixel noise for low-resolution IMC.",
        ge=0,
        le=1,
    )
    augmentation_noise_support: Literal["channel", "segmentation_mask"] = config_field(
        "channel",
        description="Spatial support on which augmentation noise may be added.",
        stage="cellvision",
        ui_group="Mask-safe augmentations",
        advice="channel preserves each marker's original nonzero support; segmentation_mask permits noise anywhere inside the extracted cell.",
    )

    n_pcs: int = config_field(
        50,
        description="Single PCA component count used before RAPIDS neighbor construction.",
        level="basic",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="The runtime value is capped by the number of cells and embedding dimensions.",
        ge=2,
    )
    n_neighbors: int = config_field(
        50,
        description="Single neighbor count used for the RAPIDS cell graph.",
        level="basic",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="The runtime value is capped below the number of embedded cells.",
        ge=2,
    )
    fusion_enabled: bool = config_field(
        True,
        description="Fuse the CellVision morphology graph with a local graph rebuilt from a batch-corrected source intensity representation.",
        level="basic",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="The default requires the configured source AnnData to contain fusion_intensity_representation; disable only for a morphology-only analysis.",
    )
    fusion_intensity_adata_path: Optional[str] = config_field(
        None,
        description="Optional AnnData supplying the batch-corrected intensity representation used for graph fusion; defaults to cellvision.input_adata_path and then general.anndata_path.",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="Set this when BioBatchNet writes to a separate output_adata_path instead of updating the CellVision source AnnData.",
    )
    fusion_intensity_representation: str = config_field(
        "X_biobatchnet",
        description="Source AnnData obsm key containing the batch-corrected per-cell intensity embedding used to build the fusion graph.",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="X_biobatchnet is written by the toolkit BioBatchNet stage; latent dimensions are used directly without another PCA.",
        min_length=1,
    )
    fusion_intensity_weight: float = config_field(
        0.5,
        description="Weight assigned to the degree-normalized intensity graph; one minus this value weights the CellVision morphology graph.",
        level="basic",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="Use 0 for the morphology endpoint, 1 for the intensity endpoint, and intermediate values for joint clustering.",
        ge=0,
        le=1,
    )
    leiden_resolutions: List[float] = config_field(
        default_factory=lambda: [0.2, 0.3, 0.5, 0.7, 1.0],
        description="Leiden resolutions evaluated on the default joint CellVision neighbor graph.",
        level="basic",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="Each value creates a namespaced cellvision_leiden_<resolution> annotation and report set.",
    )
    umap_min_dist: float = config_field(
        0.1,
        description="Minimum distance used for the CellVision RAPIDS UMAP.",
        stage="cellvision",
        ui_group="RAPIDS clustering",
        advice="Use the same value for directly comparable runs.",
        ge=0,
        le=1,
    )

    source_umap_key: str = config_field(
        "X_umap",
        description="Source AnnData obsm key on which new CellVision labels are projected.",
        stage="cellvision",
        ui_group="Plots and galleries",
        advice="Projection is skipped with a warning when this embedding is absent.",
        min_length=1,
    )
    gallery_cells_per_cluster: int = config_field(
        10,
        description="Maximum randomly sampled cells shown as rows in each Leiden gallery.",
        level="basic",
        stage="cellvision",
        ui_group="Plots and galleries",
        advice="Every selected marker is a column; a composite column is added when there are at most three markers.",
        ge=1,
    )
    gallery_max_clusters: Optional[int] = config_field(
        20,
        description="Optional maximum number of Leiden clusters receiving galleries per resolution.",
        stage="cellvision",
        ui_group="Plots and galleries",
        advice="Leave unset to generate a gallery for every discovered cluster.",
        ge=1,
    )
    figure_dpi: int = config_field(
        200,
        description="Resolution in dots per inch for CellVision raster figures.",
        stage="cellvision",
        ui_group="Plots and galleries",
        advice="Increase for publication export at the cost of larger reports.",
        ge=72,
        le=600,
    )

    @model_validator(mode="after")
    def _validate_cellvision_combinations(self):
        if self.populations is not None and self.population_obs is None:
            raise ValueError("cellvision.populations requires cellvision.population_obs")
        for field_name in ("populations", "markers"):
            values = getattr(self, field_name)
            if values is not None:
                cleaned = [str(value).strip() for value in values]
                if not cleaned or any(not value for value in cleaned):
                    raise ValueError(f"cellvision.{field_name} must contain non-empty values")
                if len(set(cleaned)) != len(cleaned):
                    raise ValueError(f"cellvision.{field_name} cannot contain duplicates")
                setattr(self, field_name, cleaned)
        if len(self.normalization_clip) != 2:
            raise ValueError("cellvision.normalization_clip must contain two values")
        lower, upper = (float(value) for value in self.normalization_clip)
        if not 0 <= lower < upper <= 1:
            raise ValueError(
                "cellvision.normalization_clip must satisfy 0 <= lower < upper <= 1"
            )
        self.normalization_clip = [lower, upper]
        if self.warmup_epochs > self.epochs:
            raise ValueError("cellvision.warmup_epochs cannot exceed cellvision.epochs")
        if not self.leiden_resolutions:
            raise ValueError("cellvision.leiden_resolutions cannot be empty")
        resolutions = [float(value) for value in self.leiden_resolutions]
        if any(value <= 0 for value in resolutions):
            raise ValueError("cellvision.leiden_resolutions must contain positive values")
        if len(set(resolutions)) != len(resolutions):
            raise ValueError("cellvision.leiden_resolutions cannot contain duplicates")
        self.leiden_resolutions = resolutions
        return self


@config_section("hyperstac")
class HyperstacConfig(ConfigModel):
    """HyPERSTAC image preprocessing, representation, and report settings."""

    asset_folder: str = config_field(
        "hyperstac",
        description="Project-relative folder for reusable HyPERSTAC images, patches, models, and AnnData assets.",
        level="basic",
        stage="hyperstac",
        ui_group="Reusable assets",
        advice="Keep this outside general.outputs_folder because downstream stages reuse these files.",
    )
    input_images_folder: Optional[str] = config_field(
        None,
        description="Optional ROI/channel TIFF folder; null uses general.denoised_images_folder.",
        level="basic",
        stage="hyperstac",
        ui_group="Reusable assets",
        advice="Each ROI must be a subfolder containing consistently named channel TIFF files.",
    )
    channels: List[str] = config_field(
        default_factory=list,
        description="Ordered channel names; an empty list infers and validates the common alphabetical channel set.",
        level="basic",
        stage="hyperstac",
        ui_group="Images",
        advice="The former GBM-specific marker list is intentionally not a package default.",
    )
    preprocess_images: bool = config_field(
        True,
        description="Create normalized channel TIFF assets before modelling.",
        level="basic",
        stage="hyperstac",
        ui_group="Images",
    )
    background_subtraction: bool = config_field(
        True,
        description="Subtract a fixed or percentile background estimate before scaling.",
        stage="hyperstac",
        ui_group="Image normalization",
    )
    background_method: Literal["fixed", "percentile"] = config_field(
        "fixed",
        description="Background estimator used for every ROI/channel image.",
        stage="hyperstac",
        ui_group="Image normalization",
    )
    background_fixed_value: float = config_field(
        0.5,
        description="Fixed background value subtracted when background_method is fixed.",
        stage="hyperstac",
        ui_group="Image normalization",
        ge=0,
    )
    background_percentile: float = config_field(
        20.0,
        description="Per-image percentile used when background_method is percentile.",
        stage="hyperstac",
        ui_group="Image normalization",
        ge=0,
        le=100,
    )
    presence_mask: bool = config_field(
        False,
        description="Restrict scale estimation to pixels passing a per-image presence threshold.",
        stage="hyperstac",
        ui_group="Image normalization",
        advice="Disabled by default to preserve the tested IMC adaptation.",
    )
    presence_percentile: float = config_field(
        95.0,
        description="Percentile used to derive a channel-presence score.",
        stage="hyperstac",
        ui_group="Image normalization",
        ge=0,
        le=100,
    )
    presence_threshold: float = config_field(
        1.0,
        description="Minimum presence score when presence masking is enabled.",
        stage="hyperstac",
        ui_group="Image normalization",
        ge=0,
    )
    scale_percentile: float = config_field(
        99.5,
        description="Robust per-channel percentile mapped to one after background correction.",
        stage="hyperstac",
        ui_group="Image normalization",
        gt=0,
        le=100,
    )
    scale_present_only: bool = config_field(
        False,
        description="Estimate scale only from presence-positive pixels.",
        stage="hyperstac",
        ui_group="Image normalization",
    )
    scale_sample_pixels: int = config_field(
        0,
        description="Maximum pixels sampled per ROI/channel for scale estimation; zero uses all pixels.",
        stage="hyperstac",
        ui_group="Image normalization",
        ge=0,
    )
    tiff_compression: Optional[str] = config_field(
        None,
        description="Optional tifffile compression codec for normalized TIFF assets.",
        stage="hyperstac",
        ui_group="Image normalization",
    )
    preprocess_overwrite: bool = config_field(
        True,
        description="Replace existing normalized TIFF outputs after validating the target path.",
        stage="hyperstac",
        ui_group="Image normalization",
    )
    patch_size: int = config_field(
        100,
        description="Square patch size in pixels.",
        level="basic",
        stage="hyperstac",
        ui_group="Patches",
        advice="At 1 um/pixel this represents a 100 um tissue window; the preprint used 224 px mIF tiles.",
        gt=0,
    )
    stride: Optional[int] = config_field(
        None,
        description="Patch stride in pixels; null uses patch_size for non-overlapping patches.",
        stage="hyperstac",
        ui_group="Patches",
        gt=0,
    )
    pixel_size_um: float = config_field(
        1.0,
        description="Micrometres represented by one source-image pixel.",
        level="basic",
        stage="hyperstac",
        ui_group="Patches",
        gt=0,
    )
    mask_channel: Optional[str] = config_field(
        None,
        description="Optional marker channel used for a direct tissue/background patch filter.",
        stage="hyperstac",
        ui_group="Patches",
    )
    mask_threshold: float = config_field(
        0.05,
        description="Mask-channel value above which a pixel is treated as tissue.",
        stage="hyperstac",
        ui_group="Patches",
        ge=0,
    )
    min_mask_fraction: float = config_field(
        0.05,
        description="Minimum mask-positive pixel fraction required for a patch.",
        stage="hyperstac",
        ui_group="Patches",
        ge=0,
        le=1,
    )
    min_patch_signal: float = config_field(
        0.0,
        description="Minimum whole-patch mean signal; zero disables this filter.",
        stage="hyperstac",
        ui_group="Patches",
        ge=0,
    )
    subpatch_size: int = config_field(
        10,
        description="Square local subpatch size used by the channel-independent tissue filter.",
        stage="hyperstac",
        ui_group="Patches",
        gt=0,
    )
    subpatch_signal_threshold: float = config_field(
        0.01,
        description="Mean all-channel signal required for a local subpatch to count as tissue.",
        stage="hyperstac",
        ui_group="Patches",
        ge=0,
    )
    min_tissue_subpatch_fraction: float = config_field(
        0.5,
        description="Minimum fraction of tissue-positive local subpatches required to retain a patch.",
        stage="hyperstac",
        ui_group="Patches",
        ge=0,
        le=1,
    )
    reuse_patches: bool = config_field(
        False,
        description="Reuse an existing validated patch set instead of re-tiling.",
        stage="hyperstac",
        ui_group="Patches",
    )
    overwrite: bool = config_field(
        True,
        description="Replace existing HyPERSTAC model assets for a deliberate rerun.",
        stage="hyperstac",
        ui_group="Reusable assets",
    )
    encoder: Literal["resnet50", "small-cnn"] = config_field(
        "resnet50",
        description="Image encoder architecture.",
        level="basic",
        stage="hyperstac",
        ui_group="VICReg training",
    )
    encoder_weights: Optional[str] = config_field(
        None,
        description="Optional existing encoder weights; when set, VICReg training is skipped.",
        stage="hyperstac",
        ui_group="VICReg training",
    )
    epochs: int = config_field(
        50,
        description="VICReg training epochs for the tested IMC adaptation.",
        level="basic",
        stage="hyperstac",
        ui_group="VICReg training",
        advice="The published mIF protocol used 100 epochs; increase after dataset-specific convergence review.",
        gt=0,
    )
    batch_size: int = config_field(
        64,
        description="VICReg training batch size.",
        stage="hyperstac",
        ui_group="VICReg training",
        advice="The preprint used 256 on an 80 GB H100; 64 is the safer portable IMC default.",
        gt=1,
    )
    representation_batch_size: Optional[int] = config_field(
        None,
        description="Optional embedding-extraction batch size; null reuses batch_size.",
        stage="hyperstac",
        ui_group="VICReg training",
        gt=0,
    )
    learning_rate: float = config_field(
        1e-4,
        description="Adam learning rate for VICReg training.",
        stage="hyperstac",
        ui_group="VICReg training",
        gt=0,
    )
    projector_output_size: int = config_field(
        2048,
        description="Width of each VICReg projection-head layer.",
        stage="hyperstac",
        ui_group="VICReg training",
        gt=0,
    )
    max_patches: Optional[int] = config_field(
        None,
        description="Optional random cap on training/embedding patches for controlled smoke runs.",
        stage="hyperstac",
        ui_group="VICReg training",
        gt=1,
    )
    permutation_channels: List[str] = config_field(
        default_factory=list,
        description="Channels perturbed in sensitivity analysis; empty means all channels.",
        stage="hyperstac",
        ui_group="Permutation sensitivity",
    )
    permutation_shuffle_repeats: int = config_field(
        10,
        description="Independent pixel-shuffle repeats per perturbation.",
        stage="hyperstac",
        ui_group="Permutation sensitivity",
        gt=0,
    )
    permutation_batch_size: int = config_field(
        128,
        description="Encoder inference batch size for permutation sensitivity.",
        stage="hyperstac",
        ui_group="Permutation sensitivity",
        gt=0,
    )
    permutation_shuffle_pixels: Literal["all", "nonzero"] = config_field(
        "all",
        description="Shuffle all pixels or only non-zero pixels.",
        stage="hyperstac",
        ui_group="Permutation sensitivity",
    )
    permutation_include_all_channels: bool = config_field(
        True,
        description="Include all-channel zero and shuffle perturbations.",
        stage="hyperstac",
        ui_group="Permutation sensitivity",
    )
    permutation_recompute_original: bool = config_field(
        False,
        description="Recompute original patch embeddings instead of reading representation AnnData.",
        stage="hyperstac",
        ui_group="Permutation sensitivity",
    )
    permutation_write_wide_csv: bool = config_field(
        True,
        description="Write optional wide compressed matrices alongside the reusable perturbation AnnData.",
        stage="hyperstac",
        ui_group="Permutation sensitivity",
    )
    run_cluster_scan: bool = config_field(
        True,
        description="Run the configured Scanpy graph/UMAP/Leiden parameter grid before visualisation.",
        level="basic",
        stage="hyperstac",
        ui_group="Clustering",
    )
    leiden_resolutions: List[float] = config_field(
        default_factory=lambda: [0.2, 0.25, 0.3, 0.35],
        description="Leiden resolution values used in the clustering grid.",
        stage="hyperstac",
        ui_group="Clustering",
        min_length=1,
    )
    n_neighbors: List[int] = config_field(
        default_factory=lambda: [15, 30, 100],
        description="Neighbour counts used in the clustering grid.",
        stage="hyperstac",
        ui_group="Clustering",
        min_length=1,
    )
    n_pcs: List[int] = config_field(
        default_factory=lambda: [0, 10, 20, 50],
        description="PCA dimensions used in the clustering grid; zero uses the complete embedding.",
        stage="hyperstac",
        ui_group="Clustering",
        min_length=1,
    )
    umap_min_dist: float = config_field(
        0.1,
        description="Scanpy UMAP minimum distance for every clustering grid graph.",
        stage="hyperstac",
        ui_group="Clustering",
        ge=0,
    )
    write_clustered_adata: bool = config_field(
        True,
        description="Persist clustering columns in the reusable representation AnnData.",
        stage="hyperstac",
        ui_group="Clustering",
    )
    replace_existing_cluster_scan: bool = config_field(
        True,
        description="Replace existing clustering-grid columns matching cluster_col_search.",
        stage="hyperstac",
        ui_group="Clustering",
    )
    cluster_col: str = config_field(
        "leiden",
        description="Exact clustering column used when cluster_col_search is empty.",
        stage="hyperstac",
        ui_group="Visualisation",
    )
    cluster_col_search: str = config_field(
        "leiden",
        description="Substring selecting all clustering columns to visualise and assess.",
        level="basic",
        stage="hyperstac",
        ui_group="Visualisation",
    )
    max_umap_channels: int = config_field(
        0,
        description="Maximum marker-intensity UMAPs; zero means all channels.",
        stage="hyperstac",
        ui_group="Visualisation",
        ge=0,
    )
    max_permutation_umaps: int = config_field(
        30,
        description="Maximum permutation-score UMAPs; zero disables these plots.",
        stage="hyperstac",
        ui_group="Visualisation",
        ge=0,
    )
    max_roi_maps: int = config_field(
        0,
        description="Maximum ROI spatial cluster maps; zero means all ROIs.",
        stage="hyperstac",
        ui_group="Visualisation",
        ge=0,
    )
    write_spatial_cluster_maps: bool = config_field(
        False,
        description="Write ROI back-gated cluster figures and TIFF label masks.",
        stage="hyperstac",
        ui_group="Visualisation",
        advice="Disabled by default because whole-cohort TIFF masks can be large.",
    )
    patches_per_cluster: int = config_field(
        20,
        description="Patch examples included in each cluster gallery.",
        stage="hyperstac",
        ui_group="Visualisation",
        gt=0,
    )
    gallery_top_markers: int = config_field(
        4,
        description="Top markers selected for each patch gallery.",
        stage="hyperstac",
        ui_group="Visualisation",
        gt=0,
    )
    gallery_columns: int = config_field(
        4,
        description="Number of gallery columns.",
        stage="hyperstac",
        ui_group="Visualisation",
        gt=0,
    )
    split_channel_gallery: bool = config_field(
        True,
        description="Show an RGB composite plus separate marker channels for selected patches.",
        stage="hyperstac",
        ui_group="Visualisation",
    )
    gallery_auto_contrast: bool = config_field(
        False,
        description="Use per-patch automatic contrast instead of a fixed intensity scale.",
        stage="hyperstac",
        ui_group="Visualisation",
    )
    gallery_vmax: float = config_field(
        1.0,
        description="Fixed gallery intensity maximum when automatic contrast is disabled.",
        stage="hyperstac",
        ui_group="Visualisation",
        gt=0,
    )
    gallery_contrast_percentile: float = config_field(
        99.5,
        description="Percentile used when automatic gallery contrast is enabled.",
        stage="hyperstac",
        ui_group="Visualisation",
        gt=0,
        le=100,
    )
    gallery_marker_source: Literal["auto", "permutation", "intensity"] = config_field(
        "auto",
        description="Evidence source used to select cluster-gallery markers.",
        stage="hyperstac",
        ui_group="Visualisation",
    )
    gallery_permutation_type: Literal["zero_channel", "shuffle_channel"] = config_field(
        "zero_channel",
        description="Permutation score used for gallery marker selection.",
        stage="hyperstac",
        ui_group="Visualisation",
    )
    differential_expression_method: Literal["wilcoxon", "t-test", "logreg"] = config_field(
        "wilcoxon",
        description="Scanpy differential-expression method used for cluster marker ranking.",
        stage="hyperstac",
        ui_group="Visualisation",
    )
    stability_top_markers: int = config_field(
        5,
        description="Top intensity markers retained per cluster in the cross-Leiden report.",
        stage="hyperstac",
        ui_group="Leiden stability",
        gt=0,
    )
    stability_max_heatmap_markers: int = config_field(
        50,
        description="Maximum markers displayed in stability heatmaps.",
        stage="hyperstac",
        ui_group="Leiden stability",
        gt=0,
    )
    stability_max_signature_markers: int = config_field(
        40,
        description="Maximum markers used to match recurrent environments.",
        stage="hyperstac",
        ui_group="Leiden stability",
        gt=0,
    )
    stability_environment_distance_threshold: float = config_field(
        0.45,
        description="Signature distance threshold for grouping clusters into recurrent environments.",
        stage="hyperstac",
        ui_group="Leiden stability",
        ge=0,
    )
    stability_marker_enrichment_min_z: float = config_field(
        0.0,
        description="Minimum marker enrichment z-score used in environment signatures.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_effect_threshold: float = config_field(
        1e-8,
        description="Absolute Cox coefficient threshold treated as a non-zero effect.",
        stage="hyperstac",
        ui_group="Leiden stability",
        ge=0,
    )
    stability_permutation_type: Literal["zero_channel", "shuffle_channel"] = config_field(
        "zero_channel",
        description="Permutation condition cross-referenced with marker-survival effects.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_max_report_items: int = config_field(
        20,
        description="Maximum rows displayed in each stability report summary.",
        stage="hyperstac",
        ui_group="Leiden stability",
        gt=0,
    )
    stability_figure_format: Literal["png", "pdf", "svg"] = config_field(
        "png",
        description="Figure format for cross-Leiden stability plots.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_figure_dpi: int = config_field(
        220,
        description="Raster resolution for cross-Leiden stability plots.",
        stage="hyperstac",
        ui_group="Leiden stability",
        gt=0,
    )
    stability_cluster_bubble_metric: Literal[
        "case_prevalence",
        "mean_case_frequency",
        "n_clusters_in_cell",
    ] = config_field(
        "case_prevalence",
        description="Metric represented by cluster-level bubble area.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_cluster_bubble_size_min: float = config_field(
        35.0,
        description="Minimum cluster-level bubble area.",
        stage="hyperstac",
        ui_group="Leiden stability",
        ge=0,
    )
    stability_cluster_bubble_size_scale: float = config_field(
        650.0,
        description="Scale factor applied to the cluster-level bubble metric.",
        stage="hyperstac",
        ui_group="Leiden stability",
        ge=0,
    )
    stability_cluster_bubble_log_scale: bool = config_field(
        False,
        description="Log-transform the cluster-level bubble metric before sizing.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_environment_bubble_metric: Literal[
        "n_clusters",
        "n_resolutions",
        "resolution_support_fraction",
        "mean_case_prevalence",
    ] = config_field(
        "n_resolutions",
        description="Metric represented by recurrent-environment bubble area.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_environment_bubble_size_min: float = config_field(
        35.0,
        description="Minimum recurrent-environment bubble area.",
        stage="hyperstac",
        ui_group="Leiden stability",
        ge=0,
    )
    stability_environment_bubble_size_scale: float = config_field(
        10.0,
        description="Scale factor applied to the recurrent-environment bubble metric.",
        stage="hyperstac",
        ui_group="Leiden stability",
        ge=0,
    )
    stability_environment_bubble_log_scale: bool = config_field(
        False,
        description="Log-transform the environment bubble metric before sizing.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_environment_color_metric: Literal[
        "median_coxnet_coefficient",
        "mean_abs_coxnet_coefficient",
        "coxnet_sign_consistency",
        "median_ridge_cox_coefficient",
        "mean_abs_ridge_cox_coefficient",
        "ridge_cox_sign_consistency",
        "median_image_only_ridge_cox_coefficient",
        "mean_abs_image_only_ridge_cox_coefficient",
        "resolution_support_fraction",
    ] = config_field(
        "median_coxnet_coefficient",
        description="Metric represented by recurrent-environment bubble colour.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    stability_environment_color_quantile: float = config_field(
        0.95,
        description="Robust quantile used to scale environment colour limits.",
        stage="hyperstac",
        ui_group="Leiden stability",
        gt=0,
        le=1,
    )
    stability_per_clustering_html: bool = config_field(
        True,
        description="Write one linked HTML interpretation page per clustering setting.",
        stage="hyperstac",
        ui_group="Leiden stability",
    )
    seed: int = config_field(
        1,
        description="Deterministic seed for preprocessing samples, patch selection, clustering, and perturbations.",
        stage="hyperstac",
        ui_group="Reproducibility",
    )

    @model_validator(mode="after")
    def validate_hyperstac_geometry(self) -> "HyperstacConfig":
        if self.subpatch_size > self.patch_size:
            raise ValueError("hyperstac.subpatch_size must not exceed hyperstac.patch_size")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("hyperstac.stride must be positive when set")
        return self


class CoxFeatureSourceConfig(ConfigModel):
    """One AnnData observation table contributing case-level Cox features."""

    name: str = Field(description="Unique short source label used as a feature prefix.")
    adata_path: str = Field(description="AnnData file containing the source observations.")
    population_obs: List[str] = Field(
        default_factory=list,
        description="Categorical observation columns aggregated to case-level fractions.",
    )
    population_obs_search: Optional[str] = Field(
        default=None,
        description="Optional substring selecting alternative population columns for a model sweep.",
    )
    continuous_obs: List[str] = Field(
        default_factory=list,
        description="Numeric observation columns aggregated to case level.",
    )
    case_obs: Optional[str] = Field(
        default=None,
        description="Source case identifier; null maps cases from clinical metadata through ROI.",
    )
    roi_obs: Optional[str] = Field(
        default="ROI",
        description="Source ROI identifier used for aggregation or clinical mapping.",
    )
    case_aggregation: Literal["weighted", "roi_mean"] = "weighted"
    normalization: Literal["fraction", "count"] = "fraction"
    include_population_counts: bool = False

    @model_validator(mode="after")
    def validate_feature_definition(self) -> "CoxFeatureSourceConfig":
        if not self.name.strip():
            raise ValueError("Cox feature source names must not be blank")
        if not self.population_obs and not self.population_obs_search and not self.continuous_obs:
            raise ValueError(
                f"Cox feature source {self.name!r} needs population_obs, "
                "population_obs_search, or continuous_obs"
            )
        return self


@config_section("cox")
class CoxConfig(ConfigModel):
    """General multi-source case-level Cox survival analysis settings."""

    feature_sources: List[CoxFeatureSourceConfig] = config_field(
        default_factory=list,
        description="AnnData observation sources combined into one case-level feature matrix.",
        level="basic",
        stage="cox",
        ui_group="Feature sources",
        advice="Use distinct names such as nimbus and hyperstac; one source may sweep matching Leiden columns.",
    )
    clinical_adata_path: Optional[str] = config_field(
        None,
        description="Optional AnnData whose obs supplies case/ROI outcomes and clinical covariates.",
        level="basic",
        stage="cox",
        ui_group="Clinical metadata",
    )
    clinical_csv_path: Optional[str] = config_field(
        None,
        description="Optional CSV supplying case/ROI outcomes and clinical covariates.",
        level="basic",
        stage="cox",
        ui_group="Clinical metadata",
        advice="Configure exactly one clinical source unless every feature AnnData already contains outcomes.",
    )
    case_col: str = config_field(
        "Case",
        description="Case identifier in clinical metadata.",
        level="basic",
        stage="cox",
        ui_group="Clinical metadata",
    )
    roi_col: Optional[str] = config_field(
        "ROI",
        description="ROI identifier used to map feature observations to cases.",
        level="basic",
        stage="cox",
        ui_group="Clinical metadata",
    )
    duration_col: str = config_field(
        "duration",
        description="Positive survival or follow-up duration column.",
        level="basic",
        stage="cox",
        ui_group="Clinical metadata",
    )
    event_col: Optional[str] = config_field(
        "event",
        description="Event indicator column; null requires assume_all_events or censored_case_ids.",
        level="basic",
        stage="cox",
        ui_group="Clinical metadata",
    )
    covariate_cols: List[str] = config_field(
        default_factory=list,
        description="Clinical covariates encoded and compared with image-derived feature sets.",
        level="basic",
        stage="cox",
        ui_group="Clinical metadata",
        advice="Configure fields such as age and sex to enable clinical-only and combined comparisons.",
    )
    censored_case_ids: List[str] = config_field(
        default_factory=list,
        description="Cases forced to event=0 when a usable event column is unavailable.",
        stage="cox",
        ui_group="Clinical metadata",
    )
    assume_all_events: bool = config_field(
        False,
        description="Treat every case as an observed event when no event column is supplied.",
        stage="cox",
        ui_group="Clinical metadata",
        advice="Enable only when censoring is genuinely absent.",
    )
    metadata_conflict: Literal["error", "first", "mode"] = config_field(
        "mode",
        description="Resolution policy for repeated non-null case metadata values.",
        stage="cox",
        ui_group="Clinical metadata",
    )
    min_observations_per_case: int = config_field(
        1,
        description="Minimum source observations required per case.",
        stage="cox",
        ui_group="Feature filtering",
        gt=0,
    )
    min_rois_per_case: int = config_field(
        1,
        description="Minimum represented ROIs required per case.",
        stage="cox",
        ui_group="Feature filtering",
        gt=0,
    )
    min_feature_prevalence: float = config_field(
        0.0,
        description="Minimum fraction of cases with a non-zero feature value.",
        stage="cox",
        ui_group="Feature filtering",
        ge=0,
        le=1,
    )
    feature_selection_top_n: int = config_field(
        25,
        description="Top univariate image features retained for multivariable models.",
        level="basic",
        stage="cox",
        ui_group="Feature filtering",
        gt=0,
    )
    coxph_max_features: int = config_field(
        20,
        description="Maximum selected features in the conventional Cox PH model.",
        stage="cox",
        ui_group="Models",
        gt=0,
    )
    ridge_max_features: int = config_field(
        30,
        description="Maximum selected image features in Ridge Cox.",
        stage="cox",
        ui_group="Models",
        gt=0,
    )
    models: List[Literal["coxph", "ridge", "coxnet"]] = config_field(
        default_factory=lambda: ["coxph", "ridge", "coxnet"],
        description="Cox model families fitted and compared.",
        level="basic",
        stage="cox",
        ui_group="Models",
        min_length=1,
    )
    feature_sets: List[Literal["image", "clinical", "clinical_image"]] = config_field(
        default_factory=lambda: ["image", "clinical", "clinical_image"],
        description="Image/clinical feature combinations compared with held-out validation.",
        level="basic",
        stage="cox",
        ui_group="Models",
        min_length=1,
    )
    standardize: bool = config_field(
        True,
        description="Standardize numeric features within every fit and training fold.",
        stage="cox",
        ui_group="Models",
    )
    coxph_penalizer: float = config_field(
        0.1,
        description="L2 penalizer for conventional lifelines Cox PH.",
        stage="cox",
        ui_group="Models",
        ge=0,
    )
    ridge_alphas: List[float] = config_field(
        default_factory=lambda: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        description="Candidate Ridge Cox alpha values evaluated by cross-validation.",
        stage="cox",
        ui_group="Models",
        min_length=1,
    )
    ridge_fixed_alpha: float = config_field(
        1.0,
        description="Ridge alpha used for feature-set comparisons when alpha scanning is not requested.",
        stage="cox",
        ui_group="Models",
        ge=0,
    )
    coxnet_l1_ratio: float = config_field(
        1.0,
        description="Elastic-net mixing parameter; one gives the interpretable lasso path used in the notebook.",
        stage="cox",
        ui_group="Models",
        gt=0,
        le=1,
    )
    coxnet_n_alphas: int = config_field(
        200,
        description="Number of CoxNet regularization values.",
        stage="cox",
        ui_group="Models",
        gt=0,
    )
    coxnet_alpha_min_ratio: float = config_field(
        0.001,
        description="Minimum-to-maximum CoxNet alpha ratio.",
        stage="cox",
        ui_group="Models",
        gt=0,
        lt=1,
    )
    coxnet_max_iter: int = config_field(
        100000,
        description="Maximum CoxNet solver iterations.",
        stage="cox",
        ui_group="Models",
        gt=0,
    )
    coxnet_tolerance: float = config_field(
        1e-7,
        description="CoxNet convergence tolerance.",
        stage="cox",
        ui_group="Models",
        gt=0,
    )
    validation_folds: int = config_field(
        5,
        description="Case-level cross-validation folds.",
        level="basic",
        stage="cox",
        ui_group="Validation",
        ge=2,
    )
    validation_repeats: int = config_field(
        10,
        description="Repeated case-level cross-validation rounds.",
        level="basic",
        stage="cox",
        ui_group="Validation",
        gt=0,
    )
    risk_group_quantiles: List[float] = config_field(
        default_factory=lambda: [0.0, 0.33, 0.67, 1.0],
        description="Quantile boundaries used for fitted and held-out risk groups.",
        stage="cox",
        ui_group="Validation",
        min_length=2,
    )
    correlation_method: Literal["pearson", "spearman", "kendall"] = config_field(
        "spearman",
        description="Case-level feature-correlation method.",
        stage="cox",
        ui_group="Diagnostics",
    )
    correlation_max_plot_features: int = config_field(
        60,
        description="Maximum features displayed in the correlation heatmap.",
        stage="cox",
        ui_group="Diagnostics",
        gt=1,
    )
    seed: int = config_field(
        1,
        description="Deterministic seed for case-level folds and model selection.",
        stage="cox",
        ui_group="Validation",
    )

    @model_validator(mode="after")
    def validate_cox_config(self) -> "CoxConfig":
        source_names = [source.name for source in self.feature_sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("cox.feature_sources names must be unique")
        if self.clinical_adata_path and self.clinical_csv_path:
            raise ValueError("Configure only one of cox.clinical_adata_path and cox.clinical_csv_path")
        quantiles = self.risk_group_quantiles
        if quantiles[0] != 0 or quantiles[-1] != 1:
            raise ValueError("cox.risk_group_quantiles must start with 0 and end with 1")
        if any(left >= right for left, right in zip(quantiles, quantiles[1:])):
            raise ValueError("cox.risk_group_quantiles must be strictly increasing")
        if any(alpha < 0 for alpha in self.ridge_alphas):
            raise ValueError("cox.ridge_alphas must contain non-negative values")
        return self


@config_section("biobatchnet")
class BioBatchNetConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description=(
            "AnnData file containing the per-cell expression matrix and batch annotation. "
            "When unset, the stage reads general.anndata_path."
        ),
    )
    output_adata_path: Optional[str] = Field(
        default=None,
        description=(
            "Destination for the AnnData with BioBatchNet embeddings and optional Scanpy results. "
            "When unset, general.anndata_path is updated in place."
        ),
    )

    batch_correction_obs: Optional[str] = Field(
        default=None,
        description=(
            "Name of the adata.obs column that identifies technical batches to remove from the "
            "biological latent space. Values are converted to strings and encoded as consecutive "
            "integers; the column must exist and should describe technical rather than biological variation."
        ),
    )
    n_for_pca: Optional[int] = Field(
        default=None,
        description=(
            "Deprecated compatibility setting retained for older configurations. The current "
            "BioBatchNet stage does not run PCA and does not use this value."
        ),
    )
    leiden_resolutions_list: List[float] = Field(
        default_factory=lambda: [0.3, 1.0],
        description=(
            "Leiden resolutions calculated from the neighbour graph of X_biobatchnet when both "
            "biobatchnet_run_postprocess and biobatchnet_run_leiden are enabled. Each result is "
            "stored in adata.obs as leiden_<resolution>."
        ),
    )
    umap_min_dist: float = Field(
        default=0.1,
        description=(
            "Scanpy UMAP min_dist used when post-processing the biological BioBatchNet embedding; "
            "smaller values produce more compact local groupings but do not change model training."
        ),
    )

    # BioBatchNet-specific parameters (nested dictionary format)
    biobatchnet_params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            'data_type': 'imc',
            'latent_dim': 20,
            'epochs': 100,
            'device': None,
            'use_raw': False,
            'extra_params': {
                'loss_weights': {
                    'recon_loss': 100.0,
                    'discriminator': 0.05,  # Adversarial removal of batch information from the biological latent
                    'classifier': 1.0,  # Retention of batch information in the batch-specific latent
                    'kl_loss_1': 0.0005,  # KL regularisation of the biological encoder
                    'kl_loss_2': 0.1,  # KL regularisation of the batch encoder
                    'ortho_loss': 0.01,  # Cross-covariance penalty between the two latent spaces
                }
            },
        },
        description=(
            "BioBatchNet training parameters passed to the pinned BioBatchNet API. The mapping "
            "controls data_type, latent_dim, epochs, device, whether to use adata.raw, and "
            "extra_params such as the six legacy loss_weights. By default the model consumes "
            "adata.X and automatically uses CUDA when available."
        ),
    )

    # BioBatchNet parameter scanning
    biobatchnet_scan_parameter_sets: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional list of BioBatchNet parameter overrides to train as separate scan runs. A "
            "set may include a name used only as its output label; other keys override the base "
            "biobatchnet_params. Supplying extra_params.loss_weights replaces the complete base "
            "loss-weight mapping, so include all six required legacy keys."
        ),
    )
    biobatchnet_scan_include_base: bool = Field(
        default=True,
        description=(
            "Also run the unmodified biobatchnet_params configuration when parameter sets are "
            "scanned. The base run writes output_adata_path; named scan runs write sibling files."
        ),
    )
    biobatchnet_run_postprocess: bool = Field(
        default=True,
        description=(
            "Compute a Scanpy neighbour graph and UMAP from X_biobatchnet after training, and "
            "optionally Leiden clusters. Disable to retain only the learned embeddings and metadata."
        ),
    )
    biobatchnet_run_leiden: bool = Field(
        default=True,
        description=(
            "Run Leiden clustering at leiden_resolutions_list during BioBatchNet post-processing. "
            "This has no effect when biobatchnet_run_postprocess is disabled."
        ),
    )

    # Scanpy neighbors computation
    n_neighbors: Optional[int] = Field(
        default=None,
        description=(
            "Number of neighbours used to build the Scanpy graph from X_biobatchnet. When unset, "
            "Scanpy's default is used; this affects UMAP and Leiden but not BioBatchNet training."
        ),
    )

    # Deprecated flat-style parameters (auto-migrated into biobatchnet_params)
    biobatchnet_data_type: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated flat alias for biobatchnet_params.data_type. Prefer the nested parameter mapping."
        ),
    )
    biobatchnet_latent_dim: Optional[int] = Field(
        default=None,
        description=(
            "Deprecated flat alias for biobatchnet_params.latent_dim. Prefer the nested parameter mapping."
        ),
    )
    biobatchnet_epochs: Optional[int] = Field(
        default=None,
        description=(
            "Deprecated flat alias for biobatchnet_params.epochs. Prefer the nested parameter mapping."
        ),
    )
    biobatchnet_device: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated flat alias for biobatchnet_params.device. Prefer the nested parameter mapping."
        ),
    )
    biobatchnet_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Deprecated flat alias that replaces biobatchnet_params.extra_params. Prefer the nested "
            "parameter mapping."
        ),
    )
    biobatchnet_use_raw: Optional[bool] = Field(
        default=None,
        description=(
            "Deprecated flat alias for biobatchnet_params.use_raw. Prefer the nested parameter mapping."
        ),
    )

    @model_validator(mode="after")
    def _apply_legacy_aliases(self):
        """Migrate deprecated flat BioBatchNet parameters into nested biobatchnet_params."""
        if self.biobatchnet_params is None:
            self.biobatchnet_params = {
                'data_type': 'imc',
                'latent_dim': 20,
                'epochs': 100,
                'device': None,
                'use_raw': True,
                'extra_params': {
                    'loss_weights': {
                        'recon_loss': 100.0,
                        'discriminator': 0.05,
                        'classifier': 1.0,
                        'kl_loss_1': 0.0005,
                        'kl_loss_2': 0.1,
                        'ortho_loss': 0.01,
                    }
                },
            }

        migrated = False
        if self.biobatchnet_data_type is not None:
            self.biobatchnet_params['data_type'] = self.biobatchnet_data_type
            migrated = True
        if self.biobatchnet_latent_dim is not None:
            self.biobatchnet_params['latent_dim'] = self.biobatchnet_latent_dim
            migrated = True
        if self.biobatchnet_epochs is not None:
            self.biobatchnet_params['epochs'] = self.biobatchnet_epochs
            migrated = True
        if self.biobatchnet_device is not None:
            self.biobatchnet_params['device'] = self.biobatchnet_device
            migrated = True
        if self.biobatchnet_kwargs is not None:
            self.biobatchnet_params['extra_params'] = self.biobatchnet_kwargs
            migrated = True
        if self.biobatchnet_use_raw is not None:
            self.biobatchnet_params['use_raw'] = self.biobatchnet_use_raw
            migrated = True

        if migrated:
            logging.warning(
                "Deprecated flat BioBatchNet parameters detected and migrated to 'biobatchnet_params'. "
                "Please update your config.yaml to use the nested format under biobatchnet.biobatchnet_params."
            )


        return self


@config_section("process")
class BasicProcessConfig(BioBatchNetConfig):
    """
    Legacy process config retained for backward compatibility.
    AnnData path management now belongs in GeneralConfig.
    """
    input_adata_path: str = 'anndata.h5ad'
    output_adata_path: str = 'anndata_processed.h5ad'

@config_section("visualization")
class VisualizationConfig(ConfigModel):
    # Input data settings
    input_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData input override; when omitted, the stage uses general.anndata_path.",
    )
    population_columns: Optional[List[str]] = Field(
        default=None,
        description="Observation columns to treat as population annotations. Defaults to general.population_obs_all, then general.population_obs_primary, then name-based auto-detection.",
    )
    metadata_columns: Optional[List[str]] = Field(
        default=None,
        description="Categorical observation columns to visualise as sample or experimental metadata. Defaults to general.metadata_obs and then dictionary/name-based auto-detection.",
    )
    groupby_obs: Optional[str] = Field(
        default=None,
        description="Observation defining biological comparison groups for abundance analysis; defaults to general.groupby_obs.",
    )
    groupby_obs_groups: Optional[List[str]] = Field(
        default=None,
        description="Ordered subset of groupby_obs categories to compare; defaults to the shared general pairwise/group selections.",
    )

    # AI interpretation settings
    enable_ai: bool = Field(
        default=True,
        description="Send summary statistics for existing Leiden clusters to the configured OpenAI call and add its provisional cluster names as *_AIlabel columns.",
    )
    tissue: str = Field(
        default="Unknown tissue",
        description="Free-text tissue context included in the AI prompt to help it interpret the marker summaries.",
    )
    repeat_ai_interpretation: bool = Field(
        default=False,
        description="Run AI interpretation again when any AnnData observation column ending in *_AIlabel already exists.",
    )

    # Visualization module toggles - all default True
    create_umaps: bool = Field(default=True, description="Create UMAP plots from the existing adata.obsm['X_umap'] coordinates.")
    create_matrix_plots: bool = Field(default=True, description="Create mean-marker matrix plots grouped by population and, optionally, metadata annotations.")
    create_tissue_overlays: bool = Field(default=True, description="Project population labels back into each labelled segmentation mask to show their tissue distribution.")
    create_population_analysis: bool = Field(default=True, description="Create population count, proportion, density, case-summary, and comparison outputs where the required metadata are available.")
    create_backgating: bool = Field(default=True, description="Create image-based backgating views that return annotated cells to their source channel images for validation.")
    create_color_legends: bool = Field(default=True, description="Save standalone category-to-colour legends for configured population and metadata columns.")

    # Categorical visualization controls
    include_metadata_umaps: bool = Field(default=True, description="Colour the existing UMAP by configured categorical metadata as well as population labels.")
    include_metadata_matrix_plots: bool = Field(default=True, description="Summarise mean marker values for groups defined by configured categorical metadata.")
    include_marker_umaps: bool = Field(default=True, description="Create per-marker UMAPs plus marker galleries for adata.X and every available AnnData layer.")
    umap_plot_individual_highlights: bool = Field(default=True, description="For each population annotation, save an additional UMAP highlighting every category separately.")
    max_categories: int = Field(default=50, description="Maximum number of unique values allowed when auto-selecting categorical population or metadata columns.")
    umap_marker_colormap: str = Field(default='viridis', description="Matplotlib colormap used for continuous marker-expression UMAPs.")
    umap_marker_gallery_default_colorbar_label: str = Field(default='Nimbus-Inference Score', description="Colour-bar label for the adata.X marker gallery; change this when adata.X contains another measurement scale.")
    umap_marker_gallery_vmax: Optional[float] = Field(default=0.8, description="Optional common upper colour limit for marker UMAP galleries; use null to scale each gallery automatically.")

    # Backgating assessment settings
    backgating_cells_per_group: int = Field(default=50, description="Maximum number of example cells sampled for each population's thumbnail gallery.")
    backgating_radius: int = Field(default=15, description="Half-width in image pixels of the square crop around each backgated cell centroid.")
    backgating_output_folder: str = Field(default='Backgating', description="Subdirectory under the visualisation output root for image-based population validation outputs.")
    backgating_use_masks: bool = Field(default=True, description="Use segmentation masks to identify cells and draw cell boundaries in backgating images.")
    backgating_mask_folder: str = Field(default='masks', description="Mask directory passed to the backgating implementation; it must correspond to the source channel images and cell identifiers.")
    backgating_pops_list: Optional[Dict[str, Any]] = Field(default=None, description="Optional populations to backgate, supplied per population-observation column or under a 'default' key; null processes all populations.")
    backgating_max_rois_to_save: Optional[int] = Field(default=None, description="Optional maximum number of randomly selected ROI image sets saved per population; intensity normalisation still uses all eligible ROIs.")

    # Backgating intensity and marker settings
    backgating_minimum: float = Field(default=0.2, description="Lower display bound used when rescaling source-channel intensities for backgating composites.")
    backgating_max_quantile: str = Field(default='i0.99', description="Upper intensity-rescaling rule for backgating images; the default uses the 99th percentile.")
    backgating_number_top_markers: int = Field(default=2, description="Number of automatically selected discriminative markers assigned to RGB channels for each population.")
    backgating_specify_blue: Optional[str] = Field(default='DNA1', description="Optional fixed blue-channel marker, normally a DNA channel for nuclear context.")
    backgating_specify_red: Optional[str] = Field(default=None, description="Optional fixed red-channel marker; null uses the first automatically selected population marker.")
    backgating_specify_green: Optional[str] = Field(default=None, description="Optional fixed green-channel marker; null uses the next automatically selected population marker.")

    # Differential expression settings for backgating marker selection
    backgating_use_differential_expression: bool = Field(default=True, description="Select RGB markers using a one-population-versus-rest Scanpy ranking instead of population mean expression alone.")
    backgating_de_method: str = Field(default='wilcoxon', description="Scanpy marker-ranking method for backgating: typically 'wilcoxon', 't-test', or 'logreg'.")
    backgating_min_logfc_threshold: float = Field(default=0.2, description="Preferred minimum log fold change for automatically selected backgating markers; ranking falls back to all markers if too few pass.")
    backgating_max_pval_adj: float = Field(default=0.05, description="Adjusted-P-value threshold reported during backgating marker selection; it does not itself exclude markers from selection.")
    backgating_markers_exclude: Optional[List[str]] = Field(default_factory=lambda: ['DNA1', 'DNA3'], description="Markers excluded from automatic backgating marker selection, usually DNA or technical channels.")

    # Backgating execution mode control
    backgating_mode: str = Field(default='full', description="Backgating workflow mode: 'full' selects markers and makes images, 'save_markers' only writes editable settings, and 'load_markers' makes images from existing settings.")

    # Population overlay visualization settings
    backgating_population_overlay_outline_width: int = Field(default=1, description="Contour width in pixels around target cells in backgating population overlays.")
    backgating_population_overlay_legend_fontsize: int = Field(default=24, description="Font size for marker and population labels on backgating overlays.")
    backgating_population_overlay_crop_size: Optional[List[int]] = Field(default_factory=lambda: [300, 300], description="Optional overlay crop size as [width, height] pixels; null retains the complete ROI.")
    backgating_population_overlay_crop_origin: str = Field(default='intelligent', description="Crop placement: a named corner, 'center', or 'intelligent' to favour a region containing target cells.")
    backgating_population_overlay_show_scale_bar: bool = Field(default=True, description="Draw a scale bar on cropped population overlays.")
    backgating_population_overlay_scale_bar_length: int = Field(default=50, description="Scale-bar length in image pixels; this is not automatically converted to micrometres.")
    backgating_population_overlay_scale_bar_thickness: int = Field(default=3, description="Scale-bar line thickness in pixels.")

    # MatrixPlot settings
    matrixplot_vmax: float = Field(default=0.5, description="Upper colour limit for unscaled mean-marker matrix plots; tune it to the measurement scale in adata.X.")
    matrixplot_use_row_colors: bool = Field(default=True, description="Use the toolkit matrix-plot helper to display group colours beside rows, falling back to Scanpy when unavailable.")

    # Population abundance plotting (create_population_abundance_analysis)
    abundance_make_all_populations_plots: bool = Field(default=True, description="Create combined abundance plots showing every population on one axis and comparison group as colour.")
    abundance_all_populations_figsize: List[float] = Field(default_factory=lambda: [4.0, 3.0], description="Minimum [width, height] in inches for combined all-population abundance plots.")
    abundance_all_populations_width_scale: float = Field(default=0.45, description="Additional plot width in inches per population used when automatically sizing combined abundance plots.")
    abundance_make_case_stacked_plots: bool = Field(default=True, description="Create stacked case-level composition plots for all cases and separately for each comparison group.")
    abundance_case_stacked_figsize: List[float] = Field(default_factory=lambda: [6.0, 3.0], description="Minimum [width, height] in inches for case-level stacked composition plots.")
    abundance_case_stacked_width_scale: float = Field(default=0.30, description="Additional plot width in inches per case used when automatically sizing stacked plots.")
    abundance_order_cases_by_population: Optional[str] = Field(default=None, description="Optional population label whose descending case abundance determines the order of stacked bars.")
    abundance_plot_style: str = Field(default='bar', description="Abundance display style: 'bar' shows summaries, while 'strip' or 'swarm' shows individual ROI or case values with mean and standard-error overlays.")
    # Y-axis scale controls for abundance barplots.
    # Accepted values: 'linear', 'log', 'intelligent'
    # Uses same flexible dictionary style as pairwise_spatial.barplot_y_scale.
    # Metrics used by abundance plots:
    # - proportions_roi_level
    # - proportions_case_average
    # - cells_per_mm2_roi_level
    # - cells_per_mm2_case_average
    abundance_barplot_y_scale: Dict[str, Any] = Field(default_factory=lambda: {
        'default': 'linear',
        'abundance': {
            'proportions_roi_level': 'linear',
            'proportions_case_average': 'linear',
            'cells_per_mm2_roi_level': 'intelligent',
            'cells_per_mm2_case_average': 'intelligent',
            'default': 'linear',
        },
    }, description="Y-axis mode ('linear', 'log', or 'intelligent') specified globally or by abundance metric.")
    abundance_barplot_y_scale_intelligent_params: Dict[str, Any] = Field(default_factory=lambda: {
        'allow_log1p': True,
        'dynamic_range_thresh': 100.0,
        'skew_improve_ratio': 0.7,
        'crush_frac_thresh': 0.7,
    }, description="Thresholds controlling when intelligent abundance scaling switches a positive, wide, skewed distribution to a logarithmic axis.")

    # General visualization settings
    save_high_res: bool = Field(default=True, description="Save most figures at 300 DPI instead of 150 DPI.")
    figure_format: str = Field(default='png', description="Output figure extension used by the visualisation suite, commonly 'png', 'pdf', or 'svg'.")


@config_section("population_embedding_qc")
class PopulationEmbeddingQCConfig(ConfigModel):
    """Configuration for population embedding and clustering structural QC."""

    enabled: bool = config_field(
        True,
        description="Enable population embedding and clustering QC when the stage is run.",
        level="basic",
        stage="population_embedding_qc",
        ui_group="Execution",
    )
    input_adata_path: Optional[str] = config_field(
        None,
        description="Optional input AnnData path; defaults to general.anndata_path.",
        level="basic",
        stage="population_embedding_qc",
        ui_group="Inputs",
    )
    mode: Literal["auto", "single", "sweep"] = config_field(
        "auto",
        description="Analysis mode: auto-detect available evidence, analyse one column, or require a Leiden sweep.",
        level="basic",
        stage="population_embedding_qc",
        ui_group="Clustering selection",
    )
    population_obs: Optional[str] = config_field(
        None,
        description="Reference population column; auto mode falls back to population, leiden, or the median sweep resolution.",
        level="basic",
        stage="population_embedding_qc",
        ui_group="Clustering selection",
    )
    sweep_regex: str = config_field(
        r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
        description="Regular expression used to detect precomputed Leiden sweep columns; it must expose a named resolution group.",
        stage="population_embedding_qc",
        ui_group="Clustering selection",
    )
    sweep_columns: Optional[List[str]] = config_field(
        None,
        description="Optional explicit, numerically ordered Leiden sweep column list instead of regex discovery.",
        stage="population_embedding_qc",
        ui_group="Clustering selection",
    )
    reference_resolution: Optional[float] = config_field(
        None,
        description="Optional sweep resolution to use as the reference clustering when no population column is supplied.",
        stage="population_embedding_qc",
        ui_group="Clustering selection",
        ge=0,
    )
    umap_key: str = config_field(
        "X_umap",
        description="Existing adata.obsm key containing the required UMAP embedding.",
        level="basic",
        stage="population_embedding_qc",
        ui_group="Representations",
    )
    pca_key: str = config_field(
        "X_pca",
        description="Existing optional adata.obsm key containing PCA coordinates; PCA is never recalculated.",
        stage="population_embedding_qc",
        ui_group="Representations",
    )
    connectivities_key: Optional[str] = config_field(
        None,
        description="Optional adata.obsp connectivity key; otherwise use neighbors.connectivities_key then connectivities.",
        stage="population_embedding_qc",
        ui_group="Representations",
    )
    sample_obs: Optional[str] = config_field(
        None,
        description="Optional sample or case obs column counted as a cluster annotation in summary heatmaps.",
        stage="population_embedding_qc",
        ui_group="Representations",
    )
    roi_obs: Optional[str] = config_field(
        None,
        description="Optional ROI obs column counted as a cluster annotation in summary heatmaps.",
        stage="population_embedding_qc",
        ui_group="Representations",
    )
    pca_dimensions: int = config_field(
        30,
        description="Maximum number of stored PCA dimensions used for distance metrics.",
        stage="population_embedding_qc",
        ui_group="Representations",
        ge=1,
    )
    umap_k: int = config_field(
        15,
        description="Number of nearest UMAP neighbours used for local separation and preservation metrics.",
        stage="population_embedding_qc",
        ui_group="Metric calculation",
        ge=2,
    )
    graph_boundary_threshold: float = config_field(
        0.7,
        description="Graph or UMAP purity below which a cell is classified as a boundary cell.",
        stage="population_embedding_qc",
        ui_group="Metric calculation",
        ge=0,
        le=1,
    )
    core_purity_threshold: float = config_field(
        0.9,
        description="Purity at or above which a cell is classified as a core cell.",
        stage="population_embedding_qc",
        ui_group="Metric calculation",
        ge=0,
        le=1,
    )
    high_entropy_threshold: float = config_field(
        0.6,
        description="Normalized local label entropy above which a cell is counted as high entropy.",
        stage="population_embedding_qc",
        ui_group="Metric calculation",
        ge=0,
        le=1,
    )
    min_cluster_size: int = config_field(
        20,
        description="Clusters below this size remain visible but are flagged and may have unavailable metrics.",
        level="basic",
        stage="population_embedding_qc",
        ui_group="Metric calculation",
        ge=2,
    )
    min_component_size: int = config_field(
        5,
        description="Minimum connected-component cell count used in substantial-component summaries.",
        stage="population_embedding_qc",
        ui_group="Metric calculation",
        ge=1,
    )
    persistence_jaccard_threshold: float = config_field(
        0.75,
        description="Minimum best-match Jaccard used to count resolution-sweep support.",
        stage="population_embedding_qc",
        ui_group="Sweep metrics",
        ge=0,
        le=1,
    )
    transition_min_fraction: float = config_field(
        0.01,
        description="Minimum source-cell fraction shown as an edge in sweep transition plots.",
        stage="population_embedding_qc",
        ui_group="Sweep metrics",
        ge=0,
        le=1,
    )
    silhouette_max_cells: int = config_field(
        10000,
        description="Maximum deterministic stratified sample used for each silhouette calculation.",
        stage="population_embedding_qc",
        ui_group="Scalability",
        ge=100,
    )
    density_max_cells_per_cluster: int = config_field(
        5000,
        description="Maximum deterministic sample per cluster used for UMAP density overlap.",
        stage="population_embedding_qc",
        ui_group="Scalability",
        ge=20,
    )
    density_grid_size: int = config_field(
        64,
        description="Number of bins per UMAP axis used by the scalable density-overlap approximation.",
        stage="population_embedding_qc",
        ui_group="Scalability",
        ge=16,
        le=512,
    )
    metric_config_path: Optional[str] = config_field(
        None,
        description="Optional YAML or JSON file overriding metric anchors, raw thresholds, inclusion, or weights.",
        stage="population_embedding_qc",
        ui_group="Scoring",
    )
    include_optional_metrics: bool = config_field(
        False,
        description="Calculate optional PCA-neighbour and UMAP-to-PCA preservation diagnostics.",
        stage="population_embedding_qc",
        ui_group="Metric calculation",
    )
    write_per_cell_metrics: bool = config_field(
        False,
        description="Write namespaced per-cell QC values as Parquet when a Parquet engine is available.",
        stage="population_embedding_qc",
        ui_group="Outputs",
    )
    write_annotated_h5ad: bool = config_field(
        True,
        description="Write a separate AnnData copy with focused cell annotations and reusable versioned population-level QC results; the input is never modified in place.",
        level="basic",
        stage="population_embedding_qc",
        ui_group="Outputs",
    )
    annotated_adata_path: str = config_field(
        "population_embedding_qc.h5ad",
        description="Configured project asset path for the optional AnnData copy containing reusable population QC results.",
        stage="population_embedding_qc",
        ui_group="Outputs",
    )
    random_seed: int = config_field(
        42,
        description="Random seed used for every deterministic sampling decision.",
        stage="population_embedding_qc",
        ui_group="Scalability",
        ge=0,
    )

    @model_validator(mode="after")
    def validate_population_qc_settings(self):
        if self.core_purity_threshold < self.graph_boundary_threshold:
            raise ValueError(
                "population_embedding_qc.core_purity_threshold must be greater than or equal to graph_boundary_threshold"
            )
        return self

@config_section("cellcharter")
class CellCharterConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData input override; when unset, the stage reads general.anndata_path.",
    )
    output_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData output override; when unset, the stage updates general.anndata_path.",
    )
    qc_output_subdir: str = Field(
        default="CellCharter_QC",
        description="Subdirectory created under general.qc_folder for CellCharter tables and figures.",
    )

    # Features
    use_rep: Optional[str] = Field(
        default="X_biobatchnet",
        description="AnnData obsm representation aggregated in non-TRVAE mode; integration-family keys can fall back to another available pipeline embedding, and null selects a layer, an automatic reduced representation, or X.",
    )
    use_layer: Optional[str] = Field(
        default=None,
        description="AnnData layer used instead of X when no non-TRVAE use_rep is selected, or used as the TRVAE input when TRVAE mode is enabled.",
    )
    scale_by_sample: bool = Field(
        default=False,
        description="Z-score every input feature within each sample before TRVAE fitting or, in non-TRVAE mode, before neighbourhood aggregation.",
    )
    scaled_rep_key: str = Field(
        default="X_cellcharter_scaled",
        description="AnnData obsm key used to store sample-scaled features or a layer copied into a matrix suitable for aggregation.",
    )

    # Optional TRVAE dimensionality reduction and batch correction
    use_trvae: bool = Field(
        default=False,
        description="Train or load a TRVAE and aggregate its latent representation instead of using the existing non-TRVAE representation.",
    )
    trvae_latent_key: str = Field(
        default="X_trVAE",
        description="AnnData obsm key receiving the TRVAE latent coordinates used for neighbourhood aggregation.",
    )
    trvae_condition_key: Optional[str] = Field(
        default="dataset",
        description="AnnData obs column supplied to TRVAE as the batch or condition covariate to be integrated.",
    )
    trvae_use_sample_key_fallback: bool = Field(
        default=True,
        description="Allow the resolved ROI/sample column to serve as the TRVAE condition when trvae_condition_key is unavailable.",
    )
    trvae_constant_condition_label: str = Field(
        default="all",
        description="Label assigned to every cell when no usable TRVAE condition column can be resolved.",
    )
    trvae_load_path: Optional[str] = Field(
        default=None,
        description="Optional directory containing a pretrained TRVAE model; a missing or invalid path causes a new model to be initialized.",
    )
    trvae_save_path: str = Field(
        default="trvae_model",
        description="Directory for the reusable fitted TRVAE model; relative paths are resolved as project assets.",
    )
    trvae_map_location: str = Field(
        default="gpu",
        description="Device location requested when loading a pretrained TRVAE model, such as gpu or cpu.",
    )
    trvae_train: bool = Field(
        default=True,
        description="Fit a newly initialized TRVAE or continue training a loaded model before computing latent coordinates.",
    )
    trvae_train_early_stopping: bool = Field(
        default=False,
        description="Enable TRVAE training early stopping when supported by the installed implementation.",
    )
    trvae_train_enable_progress_bar: bool = Field(
        default=True,
        description="Show the TRVAE training progress bar when supported by the installed implementation.",
    )
    trvae_train_max_epochs: Optional[int] = Field(
        default=None,
        description="Optional maximum TRVAE training epochs; null leaves the limit to the model's own default.",
    )
    trvae_hidden_layer_sizes: List[int] = Field(
        default_factory=lambda: [128, 128],
        description="Widths of the TRVAE encoder and decoder hidden layers.",
    )
    trvae_latent_dim: int = Field(
        default=10,
        description="Number of dimensions in the TRVAE latent representation.",
    )
    trvae_dr_rate: float = Field(
        default=0.05,
        description="Dropout rate used by the TRVAE neural network.",
    )
    trvae_use_mmd: bool = Field(
        default=True,
        description="Use maximum mean discrepancy regularization to align the configured TRVAE conditions.",
    )
    trvae_mmd_on: str = Field(
        default="z",
        description="TRVAE representation on which maximum mean discrepancy is applied, normally the latent space z.",
    )
    trvae_mmd_boundary: Optional[int] = Field(
        default=None,
        description="Optional number of conditions across which TRVAE calculates maximum mean discrepancy; null applies it across all conditions.",
    )
    trvae_recon_loss: str = Field(
        default="mse",
        description="Reconstruction loss used to fit TRVAE; it should match the scale and distribution of the input features.",
    )
    trvae_beta: float = Field(
        default=1.0,
        description="Weight applied to the TRVAE latent regularization term relative to reconstruction.",
    )
    trvae_use_bn: bool = Field(
        default=False,
        description="Enable batch normalization in the TRVAE network.",
    )
    trvae_use_ln: bool = Field(
        default=True,
        description="Enable layer normalization in the TRVAE network.",
    )

    # Graph and neighborhood aggregation
    delaunay: bool = Field(
        default=True,
        description="Construct each sample's spatial graph by Delaunay triangulation; disabling it uses Squidpy's generic-coordinate neighbour construction.",
    )
    remove_long_links: bool = Field(
        default=True,
        description="Remove unusually long spatial-graph edges after graph construction to reduce border-spanning artefacts.",
    )
    distance_percentile: float = Field(
        default=99.0,
        description="Global percentile of positive edge distances above which links are removed when remove_long_links is enabled.",
    )
    n_layers: int = Field(
        default=3,
        description="Maximum graph-hop layer to aggregate; an integer L includes the focal cell and separate summaries for hops 1 through L.",
    )
    aggregations: str = Field(
        default="mean",
        description="Neighbour feature summary, such as mean or var; comma-separated values request multiple summaries for every nonzero hop layer.",
    )
    aggregated_rep_key: str = Field(
        default="X_cellcharter",
        description="AnnData obsm key receiving the concatenated focal-cell and hop-specific neighbourhood features used for clustering.",
    )

    # Clustering
    n_clusters: int = Field(
        default=11,
        description="Fixed number of Gaussian-mixture spatial clusters to fit; this pipeline stage does not run CellCharter's automatic stability scan.",
    )
    random_state: int = Field(
        default=12345,
        description="Random seed used to initialize the CellCharter Gaussian-mixture clustering model.",
    )
    covariance_type: str = Field(
        default="full",
        description="Gaussian-mixture covariance parameterization; full allows each cluster its own unrestricted covariance matrix.",
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Optional number of cells per Gaussian-mixture fitting batch; null lets CellCharter process the full matrix according to its default.",
    )
    trainer_accelerator: str = Field(
        default="auto",
        description="Lightning accelerator used for Gaussian-mixture fitting, for example auto, cpu, gpu, or cuda where supported.",
    )
    trainer_devices: Optional[int] = Field(
        default=None,
        description="Optional number of devices supplied to the CellCharter clustering trainer.",
    )
    trainer_max_epochs: int = Field(
        default=100,
        description="Maximum training epochs for the Gaussian-mixture clustering model.",
    )
    cluster_key: str = Field(
        default="spatial_cluster",
        description="AnnData obs column receiving categorical CellCharter niche labels; the numeric labels are identifiers without intrinsic order.",
    )
    repeat_analysis: Optional[bool] = Field(
        default=None,
        description="Deprecated fallback for unset stage-specific repeat flags; null means each stage-specific flag defaults to recomputation.",
    )
    repeat_cluster_analysis: Optional[bool] = Field(
        default=None,
        description="Recompute TRVAE, graph, aggregation, and clustering; false reuses cluster_key when it contains any non-null labels, while null defaults to recomputation.",
    )
    repeat_enrichment_analysis: Optional[bool] = Field(
        default=None,
        description="Recompute cluster-by-cell-type enrichment; false reuses an existing compatible AnnData uns result, while null defaults to recomputation.",
    )
    repeat_nhood_enrichment_analysis: Optional[bool] = Field(
        default=None,
        description="Recompute cluster neighbourhood enrichment; false reuses an existing AnnData uns result, while null defaults to recomputation.",
    )
    repeat_diff_nhood_enrichment_analysis: Optional[bool] = Field(
        default=None,
        description="Recompute differential neighbourhood enrichment; false reuses an existing AnnData uns result, while null defaults to recomputation.",
    )
    repeat_shape_characterisation_analysis: Optional[bool] = Field(
        default=None,
        description="Recompute connected components, boundaries, and shape metrics; false reuses complete existing component and uns outputs, while null defaults to recomputation.",
    )

    # Optional enrichment
    run_enrichment: bool = Field(
        default=True,
        description="Calculate enrichment of general.population_obs_primary cell types within each spatial cluster when that annotation is available.",
    )
    enrichment_with_pvalues: bool = Field(
        default=False,
        description="Estimate empirical P values for cluster-by-cell-type enrichment by permutation instead of reporting enrichment alone.",
    )
    enrichment_n_perms: int = Field(
        default=1000,
        description="Number of permutations used when cluster-by-cell-type enrichment P values are enabled.",
    )
    enrichment_plot_figsize: List[float] = Field(
        default_factory=lambda: [8.0, 6.0],
        description="Width and height in inches for the CellCharter cluster-by-cell-type enrichment dot plot.",
    )
    enrichment_plot_dot_scale: float = Field(
        default=3.0,
        description="Scale factor controlling marker sizes in the cluster-by-cell-type enrichment dot plot.",
    )
    enrichment_plot_show_pvalues: bool = Field(
        default=False,
        description="Display enrichment P-value information on the CellCharter enrichment plot when permutation P values exist.",
    )
    enrichment_plot_significant_only: bool = Field(
        default=False,
        description="Restrict the CellCharter enrichment plot to statistically significant results when P values are available.",
    )

    # Neighborhood enrichment (CellCharter graph enrichment)
    run_nhood_enrichment: bool = Field(
        default=True,
        description="Quantify whether pairs of spatial clusters share more or fewer graph edges than expected from their abundance and node degree.",
    )
    nhood_connectivity_key: Optional[str] = Field(
        default=None,
        description="Optional AnnData obsp connectivity key for neighbourhood enrichment; null uses CellCharter's default spatial connectivity matrix.",
    )
    nhood_log_fold_change: bool = Field(
        default=False,
        description="Report neighbourhood enrichment as log2 observed-over-expected instead of the default observed-minus-expected difference.",
    )
    nhood_only_inter: bool = Field(
        default=True,
        description="Exclude within-cluster edges so the analysis focuses on contacts between different spatial clusters.",
    )
    nhood_symmetric: bool = Field(
        default=False,
        description="Use symmetric edge-count enrichment; false uses directional edge proportions, so source-to-target and target-to-source values can differ.",
    )
    nhood_with_pvalues: bool = Field(
        default=False,
        description="Use permutations to estimate neighbourhood-enrichment P values; false uses the faster analytical expectation without P values.",
    )
    nhood_n_perms: int = Field(
        default=1000,
        description="Number of permutations used when neighbourhood-enrichment P values are enabled.",
    )
    nhood_n_jobs: int = Field(
        default=1,
        description="Number of parallel workers used for permutation-based neighbourhood enrichment.",
    )
    nhood_batch_size: int = Field(
        default=10,
        description="Number of neighbourhood-enrichment permutations processed in each computational batch.",
    )
    nhood_observed_expected: bool = Field(
        default=True,
        description="Store the observed and expected edge matrices alongside the derived neighbourhood-enrichment matrix.",
    )
    save_nhood_enrichment_plot: bool = Field(
        default=True,
        description="Save CellCharter's neighbourhood-enrichment visualization in addition to exported matrices.",
    )
    nhood_plot_figsize: List[float] = Field(
        default_factory=lambda: [6.0, 3.0],
        description="Width and height in inches for the CellCharter neighbourhood-enrichment plot.",
    )
    nhood_enrichment_significance: Optional[float] = Field(
        default=None,
        description="Optional P-value threshold used to mark or filter significance in the neighbourhood-enrichment plot.",
    )

    # Differential neighborhood enrichment by condition
    run_diff_nhood_enrichment: bool = Field(
        default=False,
        description="Compare spatial-cluster neighbourhood-enrichment matrices between biological or experimental conditions.",
    )
    diff_nhood_condition_key: Optional[str] = Field(
        default=None,
        description="AnnData obs column defining conditions; null falls back to general.groupby_obs and then common condition columns.",
    )
    diff_nhood_condition_groups: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered subset of condition levels to compare; null uses configured general group lists or all available levels.",
    )
    diff_nhood_connectivity_key: Optional[str] = Field(
        default=None,
        description="Optional AnnData obsp connectivity key for differential neighbourhood enrichment; null uses CellCharter's default spatial graph.",
    )
    diff_nhood_log_fold_change: bool = Field(
        default=False,
        description="Express each condition's neighbourhood enrichment as log2 observed-over-expected rather than observed-minus-expected before calculating contrasts.",
    )
    diff_nhood_only_inter: bool = Field(
        default=True,
        description="Exclude within-cluster edges from differential neighbourhood-enrichment comparisons.",
    )
    diff_nhood_symmetric: bool = Field(
        default=False,
        description="Use symmetric rather than directional neighbourhood enrichment within each condition.",
    )
    diff_nhood_with_pvalues: bool = Field(
        default=False,
        description="Estimate empirical P values by resampling condition labels at the sample/library level.",
    )
    diff_nhood_library_key: Optional[str] = Field(
        default=None,
        description="AnnData obs column identifying independent samples or libraries for differential-enrichment permutations; null uses the resolved sample key.",
    )
    diff_nhood_n_perms: int = Field(
        default=1000,
        description="Number of sample-level condition permutations used for differential neighbourhood-enrichment P values.",
    )
    diff_nhood_n_jobs: Optional[int] = Field(
        default=None,
        description="Optional number of parallel workers for differential neighbourhood-enrichment permutations.",
    )
    diff_nhood_plot_ncols: int = Field(
        default=2,
        description="Number of columns in the grid of condition-pair differential neighbourhood-enrichment plots.",
    )
    save_diff_nhood_enrichment_plot: bool = Field(
        default=True,
        description="Save CellCharter's differential neighbourhood-enrichment plot in addition to exported matrices.",
    )

    # Shape characterisation
    run_shape_characterisation: bool = Field(
        default=False,
        description="Identify connected components of spatial clusters, reconstruct their boundaries, and calculate configured component-shape metrics.",
    )
    shape_component_key: str = Field(
        default="component",
        description="AnnData obs column receiving connected-component identifiers for spatially contiguous regions of a cluster.",
    )
    shape_component_cluster_key: Optional[str] = Field(
        default=None,
        description="AnnData obs labels whose connected components are characterized; null uses cluster_key.",
    )
    shape_connectivity_key: Optional[str] = Field(
        default=None,
        description="Optional AnnData obsp graph used to define connected components; null uses CellCharter's default spatial connectivity matrix.",
    )
    shape_min_cells: int = Field(
        default=250,
        description="Minimum cells required for a same-cluster connected component to be retained for shape analysis.",
    )
    shape_min_hole_area_ratio: float = Field(
        default=0.1,
        description="Minimum hole area relative to its component boundary area for the hole to be retained in the alpha-shape polygon.",
    )
    shape_alpha_start: int = Field(
        default=2000,
        description="Starting alpha value for CellCharter's iterative alpha-shape boundary reconstruction.",
    )
    shape_compute_linearity: bool = Field(
        default=True,
        description="Calculate component linearity, the dominant skeleton path length divided by total skeleton length.",
    )
    shape_linearity_key: str = Field(
        default="linearity",
        description="Key used to store the per-component linearity metric in the CellCharter shape result.",
    )
    shape_linearity_height: int = Field(
        default=1000,
        description="Raster height used when converting component polygons to skeletons for the linearity calculation.",
    )
    shape_linearity_min_ratio: float = Field(
        default=0.05,
        description="Minimum relative skeleton-branch length retained in the component linearity calculation.",
    )
    shape_compute_curl: bool = Field(
        default=True,
        description="Calculate component curl, a measure of how curved or twisted a region is relative to its major axis and fibre length.",
    )
    shape_curl_key: str = Field(
        default="curl",
        description="Key used to store the per-component curl metric in the CellCharter shape result.",
    )
    shape_plot_metrics: bool = Field(
        default=True,
        description="Plot computed shape metrics across configured condition or cluster groups when suitable metadata exists.",
    )
    shape_metrics_condition_key: Optional[str] = Field(
        default=None,
        description="AnnData obs condition used to group shape-metric plots; null falls back to the differential condition and then general.groupby_obs.",
    )
    shape_metrics_condition_groups: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered condition levels included in shape-metric plots.",
    )
    shape_metrics_cluster_key: Optional[str] = Field(
        default=None,
        description="AnnData obs cluster annotation used to stratify shape-metric plots; null uses shape_component_cluster_key or cluster_key.",
    )
    shape_metrics_cluster_groups: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered cluster levels included in shape-metric plots.",
    )
    shape_metrics_ncols: int = Field(
        default=2,
        description="Number of columns in the grid of CellCharter shape-metric plots.",
    )

    # QC plotting
    save_spatial_plots: bool = Field(
        default=True,
        description="Save per-sample spatial scatter plots of CellCharter cluster assignments.",
    )
    max_rois_for_plots: int = Field(
        default=12,
        description="Maximum number of samples or ROIs for which spatial cluster scatter plots are produced.",
    )
    point_size: float = Field(
        default=2.0,
        description="Marker size used in per-sample spatial cluster scatter plots.",
    )
    save_enrichment_heatmap: bool = Field(
        default=True,
        description="Save matrix heatmaps for cluster-by-cell-type and neighbourhood-enrichment results when available.",
    )
    cluster_default_cmap: Optional[str] = Field(
        default=None,
        description="Matplotlib colormap used to assign cluster colours; null uses Scanpy's godsnot_102 categorical palette.",
    )
    save_cluster_umap: bool = Field(
        default=True,
        description="Save a UMAP coloured by CellCharter cluster when a usable UMAP embedding can be found or computed.",
    )
    cluster_umap_point_size: float = Field(
        default=10.0,
        description="Marker size used in the CellCharter cluster UMAP.",
    )
    cluster_umap_legend_loc: str = Field(
        default="right margin",
        description="Scanpy legend-location setting for the CellCharter cluster UMAP.",
    )
    save_cluster_composition_plots: bool = Field(
        default=True,
        description="Save case-level stacked cluster compositions and grouped summaries when general.case_obs is available.",
    )
    composition_order_by_environment: str = Field(
        default="0",
        description="Spatial-cluster label whose abundance is used to order cases in stacked composition plots.",
    )
    composition_stacked_figsize: List[float] = Field(
        default_factory=lambda: [6.0, 3.0],
        description="Base width and height in inches for case-level stacked cluster-composition plots.",
    )
    composition_stacked_width_scale: float = Field(
        default=0.30,
        description="Additional automatic figure width in inches per case for stacked composition plots.",
    )
    composition_group_barplot_figsize: List[float] = Field(
        default_factory=lambda: [6.0, 3.0],
        description="Width and height in inches for grouped cluster-composition bar plots.",
    )
    figure_extension: str = Field(
        default=".png",
        description="Preferred extension for CellCharter figures, including the leading dot; a non-default value overrides the legacy figure_format.",
    )
    figure_format: str = Field(
        default="png",
        description="Legacy figure-format setting used only when figure_extension remains at its default .png value.",
    )
    save_high_res: bool = Field(
        default=True,
        description="Save supported CellCharter QC figures at high resolution, normally 300 dpi rather than 150 dpi.",
    )

@config_section("starling")
class StarlingConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData input override; null uses the pipeline-managed general AnnData path.",
    )
    output_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData output override; null updates the pipeline-managed general AnnData path.",
    )
    qc_output_subdir: str = Field(
        default='Starling_QC',
        description="Subdirectory below the active QC or managed report location for STARLING tables, plots, and training logs.",
    )

    # Optional local checkout fallback. Leave null when biostarling/starling is installed in the env.
    starling_repo_path: Optional[str] = Field(
        default=None,
        description="Optional local STARLING repository prepended to Python's import path; null imports the biostarling installation from the active environment.",
    )

    # Feature matrix. STARLING expects non-negative segmented cell-by-marker expression in adata.X.
    use_layer: Optional[str] = Field(
        default=None,
        description="Optional adata.layers key supplying the non-negative cell-by-marker matrix; null uses adata.X, and this stage applies no normalization or transformation.",
    )
    marker_include: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered marker subset used to fit STARLING; null starts from every adata.var_names feature before applying marker_exclude, and at least 10 must remain.",
    )
    marker_exclude: List[str] = Field(
        default_factory=list,
        description="Exact marker names removed from the selected STARLING feature set, commonly used to exclude DNA, control, morphology, or technically unreliable channels.",
    )
    clip_small_negative_values: bool = Field(
        default=True,
        description="Replace negative values with zero only when the matrix minimum lies within negative_value_tolerance; larger negative values always stop the stage.",
    )
    negative_value_tolerance: float = Field(
        default=1e-8,
        description="Absolute numerical tolerance below zero within which tiny floating-point residuals may be clipped when clip_small_negative_values is enabled.",
    )

    # Initial clustering.
    initial_clustering_method: str = Field(
        default='User',
        description="Cluster initialization used to seed STARLING centroids and cluster count: User labels, KM K-means, diagonal GMM, FS FlowSOM, or PG PhenoGraph.",
    )
    initial_label_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation containing complete starting labels in User mode; null falls back to general.population_obs_primary.",
    )
    n_clusters: Optional[int] = Field(
        default=None,
        description="Requested number of starting clusters for KM, GMM, or FS initialization; required for those methods and ignored for User and PG.",
    )

    # STARLING model settings.
    seed: int = Field(
        default=10,
        description="Random seed applied through Lightning, including worker seeding, before initialization, synthetic-error generation, and model fitting.",
    )
    dist_option: str = Field(
        default='T',
        description="Per-marker likelihood family: T uses a heavy-tailed Student-t distribution with 2 degrees of freedom, while N uses a Normal distribution.",
    )
    singlet_prop: float = Field(
        default=0.6,
        description="Documented initial singlet proportion, optimized rather than enforced as the final fraction; the reviewed upstream 0.1.4 code reverses this value when initializing its singlet/error branches, so verify version behavior before tuning it.",
    )
    model_cell_size: bool = Field(
        default=True,
        description="Include positive segmented-cell area as an additional phenotype-dependent signal when estimating singlet and segmentation-error probabilities.",
    )
    cell_size_col_name: str = Field(
        default='mask_area',
        description="Preferred adata.obs column containing positive cell-mask areas when model_cell_size is enabled.",
    )
    cell_size_fallback_cols: List[str] = Field(
        default_factory=lambda: ['area'],
        description="Ordered fallback adata.obs columns searched when cell_size_col_name is absent; the first complete, numeric, strictly positive column is used.",
    )
    model_zplane_overlap: bool = Field(
        default=True,
        description="Request STARLING's overlapping-section model, in which a combined segment may have an area between the larger constituent cell and their summed areas; also controls synthetic-error area generation.",
    )
    model_regularizer: float = Field(
        default=1.0,
        description="Multiplier on binary cross-entropy for on-the-fly synthetic singlet/error discrimination relative to the observed-data negative log-likelihood; the paper recommends 0.1, whereas this pipeline defaults to 1.0.",
    )
    learning_rate: float = Field(
        default=1e-3,
        description="Adam optimizer learning rate for STARLING's cluster, mixture, and segmentation-error parameters.",
    )
    doublet_threshold: float = Field(
        default=0.5,
        description="Threshold above which the inferred segmentation-error probability is written as a binary doublet/error call; it changes the call, not the fitted probabilities or phenotype label.",
    )

    # Lightning trainer settings.
    max_epochs: Optional[int] = Field(
        default=100,
        description="Maximum Lightning training epochs for the single STARLING fit; null delegates epoch selection to Lightning defaults.",
    )
    early_stopping: bool = Field(
        default=True,
        description="Stop fitting when the monitored training metric ceases to improve according to Lightning's EarlyStopping defaults.",
    )
    early_stopping_monitor: str = Field(
        default='train_loss',
        description="Lightning metric monitored in minimization mode for early stopping; STARLING logs train_loss, train_nll, and train_bce but has no validation loop here.",
    )
    trainer_accelerator: str = Field(
        default='auto',
        description="Lightning accelerator selection such as auto, cpu, gpu, or mps; auto chooses from resources visible in the Starling environment.",
    )
    trainer_devices: Optional[int] = Field(
        default=None,
        description="Optional number of devices passed to Lightning; null retains Lightning's automatic device selection.",
    )
    trainer_precision: Optional[str] = Field(
        default=None,
        description="Optional Lightning numerical-precision setting; null uses the installed Lightning default and non-default precision should be validated against STARLING's double-precision inputs.",
    )
    enable_checkpointing: bool = Field(
        default=False,
        description="Enable Lightning's automatic training checkpoints; these are separate from the explicit reusable model saved by save_model.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show Lightning's interactive training progress bar in the stage log.",
    )
    log_every_n_steps: Optional[int] = Field(
        default=None,
        description="Optional Lightning interval, in optimizer steps, for recording training metrics; null uses the installed Lightning default.",
    )
    limit_train_batches: Optional[Any] = Field(
        default=None,
        description="Optional Lightning training-data limit: an integer selects a batch count and a float from 0 to 1 selects a fraction per epoch; null uses all batches.",
    )
    tensorboard_logging: bool = Field(
        default=True,
        description="Write Lightning training metrics to a TensorBoard log below the STARLING QC directory.",
    )

    # Output controls.
    output_prefix: str = Field(
        default='starling',
        description="Prefix cleaned and applied to STARLING observation, multidimensional, centroid, metadata, table, and plot keys to avoid collisions between runs.",
    )
    write_canonical_starling_keys: bool = Field(
        default=False,
        description="Also write upstream unprefixed keys such as st_label, doublet_prob, and assignment_prob_matrix; enabling this may overwrite existing STARLING results.",
    )
    store_assignment_prob_matrix: bool = Field(
        default=True,
        description="Store the N-by-C joint singlet-and-cluster posterior matrix in adata.obsm; rows do not generally sum to one because their sum is the cell's singlet probability.",
    )
    store_gamma_assignment_prob_matrix: bool = Field(
        default=False,
        description="Store the N-by-C-by-C posterior matrix for all ordered phenotype pairs contributing to a segmentation error; memory use grows quadratically with cluster count.",
    )
    save_model: bool = Field(
        default=False,
        description="Serialize the complete trained STARLING Lightning object with torch.save for reuse or inspection; this can be version-sensitive and is not needed to read saved cell results.",
    )
    model_output_name: str = Field(
        default='starling_model.pt',
        description="Model output path used when save_model is enabled; a relative path is resolved as a reusable project asset rather than below the transient run directory.",
    )
    save_qc_tables: bool = Field(
        default=True,
        description="Write initialization mappings, initial and fitted centroids, cluster counts, error calls by cluster, run summary, and per-cell results as CSV files.",
    )
    save_qc_plots: bool = Field(
        default=True,
        description="Write histograms of segmentation-error and maximum joint assignment probabilities plus a STARLING cluster-count bar chart.",
    )
    figure_format: str = Field(
        default='png',
        description="Filename extension for STARLING QC plots, supplied without a leading dot.",
    )

@config_section("pairwise_spatial")
class PairwiseSpatialConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData input override; null uses the pipeline-managed general AnnData path.",
    )
    output_subdir: str = Field(
        default='Pairwise_Spatial',
        description="Subdirectory below the active QC or managed report location for pairwise spatial tables, plots, and metadata.",
    )
    reload_saved_results: bool = Field(
        default=True,
        description="Reuse structurally complete raw Squidpy, distance, and PCF tables when present, allowing plot-only reruns without repeating the analyses.",
    )

    # Core metadata keys
    population_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation containing the target population labels; null uses general.population_obs_primary, then the legacy population column.",
    )
    groupby_obs: Optional[str] = Field(
        default=None,
        description="Optional AnnData observation used to stratify ROI results into biological or experimental groups; null uses general.groupby_obs.",
    )
    groupby_obs_groups: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered subset of groupby_obs categories retained for every analysis and plot; null uses general.groupby_obs_groups.",
    )
    roi_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation identifying independent images or regions within which spatial relationships are calculated; null uses general.roi_obs.",
    )
    x_coord_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation containing cell-centroid x coordinates; null uses general.x_coord_obs and values are assumed to be in the units named by the distance settings.",
    )
    y_coord_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation containing cell-centroid y coordinates; null uses general.y_coord_obs and values are assumed to be in the units named by the distance settings.",
    )
    master_index_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation uniquely identifying cells when distance results are mapped back to their original source populations; null uses general.master_index_obs.",
    )
    source_population_obs: Optional[str] = Field(
        default=None,
        description="Optional separate observation used to group anchor cells in nearest-distance summaries; null uses population_obs for both source and target identities.",
    )

    # Metadata export controls
    include_all_obs_metadata: bool = Field(
        default=True,
        description="Include every AnnData observation column in the saved cell snapshot and eligible ROI metadata exports.",
    )
    metadata_obs_columns: List[str] = Field(
        default_factory=list,
        description="Additional AnnData observation columns exported with ROI-level results when include_all_obs_metadata is false; an empty list falls back to general.metadata_obs.",
    )

    # Squidpy neighborhood enrichment
    run_squidpy_interactions: bool = Field(
        default=True,
        description="Build a radius graph separately within each configured subregion and calculate Squidpy observed edge counts and label-permutation z-scores for every population pair.",
    )
    squidpy_subregion_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation whose categories are analysed separately by Squidpy; null uses roi_obs, which normally produces one graph and interaction matrix per ROI.",
    )
    squidpy_subregion_suffix: str = Field(
        default='',
        description="Suffix appended to squidpy_subregion_obs when selecting the Squidpy library key used to prevent graph edges across spatial units.",
    )
    squidpy_radius_min_um: int = Field(
        default=0,
        description="Lower distance bound passed to Squidpy radius-graph construction, in the unconverted units of the configured centroid coordinates.",
    )
    squidpy_radius_max_um: int = Field(
        default=20,
        description="Upper distance bound passed to Squidpy radius-graph construction; cells within this annular range are treated as neighbours.",
    )
    squidpy_n_permutations: int = Field(
        default=1000,
        description="Number of within-subregion population-label permutations used by Squidpy to estimate the null mean and standard deviation of interaction counts.",
    )

    # Distance bootstrap analysis
    run_distance_bootstrap: bool = Field(
        default=True,
        description="Measure each cell's nearest target-population distance and compare it with distances obtained after repeatedly permuting target labels within each ROI.",
    )
    distance_populations: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered target-population subset for nearest-distance analysis; null evaluates every observed target population.",
    )
    distance_roi_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of ROI identifiers included in nearest-distance analysis; null evaluates every ROI.",
    )
    distance_n_bootstraps: int = Field(
        default=1000,
        description="Number of within-ROI target-label permutations used to form the nearest-distance null distribution; these are permutations, not resampled biological replicates.",
    )
    distance_n_jobs: int = Field(
        default=-1,
        description="Joblib worker count for distance permutations; -1 uses all available CPU cores.",
    )
    distance_ddof: int = Field(
        default=1,
        description="Delta degrees of freedom used when calculating the standard deviation of permuted cell-level distances for distance z-scores.",
    )
    ignore_cells_without_label: bool = Field(
        default=False,
        description="Drop cells missing required source or target population labels during distance analysis; false stops with an error instead.",
    )

    # Pair-correlation function (PCF)
    run_pcf: bool = Field(
        default=True,
        description="Calculate edge-corrected cross-pair-correlation functions for every ordered population pair and report g(r) at one selected distance bin.",
    )
    pcf_target_distance_um: float = Field(
        default=20.0,
        description="Requested lower-radius value at which PCF summaries are extracted; the nearest available radius-step bin is used and reported as evaluated_um.",
    )
    pcf_max_radius_um: float = Field(
        default=100.0,
        description="Outer extent of the PCF curve in the unconverted units of the configured coordinates.",
    )
    pcf_radius_step_um: float = Field(
        default=10.0,
        description="Width of each PCF annulus and spacing between reported lower-radius values; with 10 and target 20, g is calculated for distances greater than 20 and at most 30.",
    )
    pcf_num_bootstrap: int = Field(
        default=1000,
        description="Number of 100-by-100 spatial-grid resamples used to estimate condition-level PCF confidence bounds.",
    )
    pcf_cluster_column: str = Field(
        default='cluster',
        description="Column name assigned to population labels in the exported SpOOx-style PCF statistics table; it is not an AnnData input key.",
    )
    pcf_samples: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered subset of ROI identifiers included in PCF analysis; null uses every ROI.",
    )

    # Optional source-target population pairs
    # Supports:
    # 1) Direct mapping: {source_pop: [target_pop, ...]}
    # 2) Nested by obs key: {population_obs: {source_pop: [target_pop, ...]}}
    # Target tokens:
    # - "ALL": all populations in population_obs
    # - "ALL_OTHERS": all populations except source_pop
    # - "MATCH_x": populations containing substring "x" (case-insensitive)
    # - "NOT_x": populations not containing substring "x" (case-insensitive)
    population_pairs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-to-target population selections used only for focused barplots; supports direct or observation-nested mappings plus ALL, ALL_OTHERS, MATCH_x, and NOT_x target tokens.",
    )

    # Plotting
    make_matrix_plots: bool = Field(
        default=True,
        description="Save population-by-population heatmaps or clustermaps for each available Squidpy, distance, and PCF metric.",
    )
    make_pair_barplots: bool = Field(
        default=True,
        description="Save ROI-level barplots for source-target combinations selected by population_pairs.",
    )
    heatmap_use_clustermap: bool = Field(
        default=True,
        description="Use seaborn clustermaps for pairwise matrices when possible; false uses fixed-order heatmaps.",
    )
    heatmap_row_cluster: bool = Field(
        default=True,
        description="Hierarchically reorder source-population rows when clustermap plotting is enabled.",
    )
    heatmap_col_cluster: bool = Field(
        default=True,
        description="Hierarchically reorder target-population columns when clustermap plotting is enabled.",
    )
    heatmap_figsize: List[float] = Field(
        default_factory=lambda: [5.0, 5.0],
        description="Base pairwise-matrix figure width and height in inches.",
    )
    heatmap_percentile: float = Field(
        default=95.0,
        description="Percentile of finite matrix magnitudes used to limit heatmap colour ranges and reduce domination by extreme values.",
    )
    pairwise_matrices_cbar_corner: str = Field(
        default='off_plot_right',
        description="Pairwise-matrix colourbar position: lower_right, upper_left, or off_plot_right.",
    )
    pairwise_matrices_share_vmax_vmin: bool = Field(
        default=False,
        description="Reuse each metric's all-data colour limits for its group-specific matrices so colours are directly comparable across groups.",
    )
    heatmap_cmap_interactions: str = Field(
        default='coolwarm',
        description="Matplotlib colormap for Squidpy enrichment z-score matrices.",
    )
    heatmap_cmap_distance: str = Field(
        default='coolwarm',
        description="Matplotlib colormap for nearest-distance matrices; the plotting code reverses this map so shorter distances receive the visual enrichment end.",
    )
    heatmap_cmap_pcf: str = Field(
        default='coolwarm',
        description="Matplotlib colormap for PCF matrices centred on the null reference g(r)=1.",
    )
    heatmap_cmap_counts: str = Field(
        default='viridis',
        description="Matplotlib colormap for non-negative Squidpy observed interaction-count matrices.",
    )
    barplot_figsize: List[float] = Field(
        default_factory=lambda: [3.0, 3.0],
        description="Base width and height in inches for selected source-target barplots.",
    )
    barplot_add_points: bool = Field(
        default=True,
        description="Overlay individual ROI or subregion values on selected-pair barplots so replication and heterogeneity remain visible.",
    )
    # Barplot Y-axis scale controls.
    # Accepted values: 'linear', 'log', 'intelligent'
    # Flexible structure examples:
    # barplot_y_scale: {'default': 'linear'}
    # barplot_y_scale: {'distance': {'observed': 'log', 'delta': 'linear'}, 'pcf': {'g': 'log'}}
    # barplot_y_scale: {'squidpy': {'count': 'log1p', 'zscore': 'intelligent'}, 'default': 'linear'}
    # Default is explicitly populated by analysis/metric so users can tweak directly.
    barplot_y_scale: Dict[str, Any] = Field(
        default_factory=lambda: {
            'default': 'linear',
            'squidpy': {
                'count': 'intelligent',
                'zscore': 'intelligent',
                'default': 'linear',
            },
            'distance': {
                'observed': 'intelligent',
                'bootmean': 'intelligent',
                'delta': 'intelligent',
                'zscore': 'intelligent',
                'default': 'intelligent',
            },
            'pcf': {
                'g': 'linear',
                'g_mean': 'linear',
                'default': 'linear',
            },
        },
        description="Nested per-analysis and per-metric axis-scale rules for barplots: linear, log, log1p, or intelligent automatic selection.",
    )
    barplot_y_scale_intelligent_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'allow_log1p': True,
            'dynamic_range_thresh': 100.0,
            'skew_improve_ratio': 0.7,
            'crush_frac_thresh': 0.7,
        },
        description="Thresholds controlling when intelligent barplot scaling chooses linear, log, or log1p display; these settings never transform the saved statistics.",
    )
    make_source_target_barplots: bool = Field(
        default=True,
        description="Also place all selected targets for each source population on a combined figure, with group as hue when available.",
    )
    source_target_barplot_width_scale: float = Field(
        default=0.35,
        description="Automatic combined source-target figure width in inches per selected target, subject to barplot_figsize as a minimum.",
    )
    source_target_barplot_order_group: Optional[str] = Field(
        default=None,
        description="Optional group category whose descending mean values determine target order in grouped source-to-all-target barplots.",
    )
    make_enrichment_plots: bool = Field(
        default=True,
        description="Create per-source plots of the numerically highest and lowest target-pair metrics; enrichment here is a ranking label, not an additional hypothesis test.",
    )
    enrichment_plot_figsize: List[float] = Field(
        default_factory=lambda: [5.5, 4.0],
        description="Base width and height in inches for per-source enriched/depleted target plots.",
    )
    enrichment_plot_use_barplot: bool = Field(
        default=True,
        description="Display means with error bars in enrichment plots; false displays ROI-value distributions as boxplots.",
    )
    enrichment_plot_errorbar: str = Field(
        default='ci95',
        description="Across-ROI error-bar display for enrichment barplots: ci95 for a seaborn 95% interval or se for standard error; neither performs a group-level test.",
    )
    enrichment_plot_top_n: int = Field(
        default=5,
        description="Number of targets with the strongest enrichment-direction metric retained per source population.",
    )
    enrichment_plot_bottom_n: int = Field(
        default=5,
        description="Number of targets with the strongest depletion-direction metric retained per source population.",
    )
    enrichment_plot_target_populations: Optional[List[str]] = Field(
        default=None,
        description="Optional target-population subset eligible for enriched/depleted ranking; null allows every available target.",
    )
    enrichment_plot_exclude_homotypic: bool = Field(
        default=True,
        description="Exclude source-to-same-population pairs when ranking targets for enrichment plots.",
    )
    enrichment_plot_share_x_axis_across_groups: bool = Field(
        default=True,
        description="Use a common x-axis scale for group-specific enrichment plots of the same source and metric to support visual comparison.",
    )
    enrichment_plot_color_mode: str = Field(
        default='direction',
        description="Colour enrichment-plot bars by enriched/depleted direction or by target population; accepted values are direction and population.",
    )
    enrichment_plot_label_box_width: float = Field(
        default=0.03,
        description="Width in axes coordinates of the target-population colour strip drawn beside enrichment labels.",
    )
    enrichment_plot_height_per_target: float = Field(
        default=0.25,
        description="Minimum figure height in inches allocated per displayed target population in enrichment plots.",
    )
    figure_extension: str = Field(
        default='.png',
        description="File extension, including the leading dot, used for pairwise spatial figures.",
    )
    figure_dpi: int = Field(
        default=300,
        description="Raster resolution in dots per inch for saved pairwise spatial figures.",
    )

@config_section("networkx_spatial")
class NetworkxSpatialConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData input override; null uses the pipeline-managed general AnnData path.",
    )
    output_subdir: str = Field(
        default='NetworkX_Spatial',
        description="Subdirectory below the active QC or managed report location for NetworkX spatial summaries, plots, and metadata.",
    )
    reload_saved_results: bool = Field(
        default=True,
        description="Reuse structurally complete ROI and case summary CSVs when present, allowing plot-only reruns without rebuilding graphs or null distributions.",
    )

    # Core metadata keys
    population_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation containing the categorical cell-population labels used for assortativity, induced subgraphs, and label permutations; null uses the general primary population key.",
    )
    roi_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation identifying images or regions for which separate spatial graphs are constructed; null uses general.roi_obs.",
    )
    case_obs: Optional[str] = Field(
        default=None,
        description="Optional AnnData observation identifying biological cases across which their ROI metric and matched permutation values are averaged; null uses general.case_obs.",
    )
    groupby_obs: Optional[str] = Field(
        default=None,
        description="Optional ROI or case metadata category used to colour and stratify summary plots; it does not filter cells or perform a group-comparison test.",
    )
    x_coord_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation containing cell-centroid x coordinates used to construct each ROI graph; null uses general.x_coord_obs.",
    )
    y_coord_obs: Optional[str] = Field(
        default=None,
        description="AnnData observation containing cell-centroid y coordinates used to construct each ROI graph; null uses general.y_coord_obs.",
    )
    spatial_key: Optional[str] = Field(
        default=None,
        description="Temporary AnnData obsm key under which the configured x/y coordinates are placed before Squidpy graph construction; null uses general.spatial_key.",
    )
    master_index_obs: Optional[str] = Field(
        default=None,
        description="Optional unique-cell observation carried into the analysis metadata frame for compatibility; it does not alter graph construction or the current NetworkX metrics.",
    )

    # Metadata export controls
    include_all_obs_metadata: bool = Field(
        default=True,
        description="Include all eligible AnnData observations in ROI and case metadata tables; the complete AnnData observation snapshot is written independently of this setting.",
    )
    metadata_obs_columns: List[str] = Field(
        default_factory=list,
        description="Observation columns included in ROI and case metadata tables when include_all_obs_metadata is false; an empty list falls back to general.metadata_obs.",
    )
    ignore_cells_without_label: bool = Field(
        default=False,
        description="Drop cells with null or blank population labels before constructing ROI graphs; false stops the stage when such labels are present.",
    )

    # Squidpy graph construction
    graph_coord_type: str = Field(
        default='generic',
        description="Squidpy coordinate mode used for graph construction; generic is appropriate for continuous IMC centroids, while grid is intended for lattice-based observations.",
    )
    graph_delaunay: bool = Field(
        default=False,
        description="Use a Delaunay triangulation for generic coordinates instead of the default k-nearest-neighbour graph; graph_n_neighs is ignored in Delaunay mode.",
    )
    graph_n_neighs: Optional[int] = Field(
        default=6,
        description="Nearest-neighbour count for the default generic k-nearest-neighbour graph; null lets Squidpy resolve its default, and the value is ignored when radius or Delaunay mode is active.",
    )
    graph_radius: Optional[List[float]] = Field(
        default=None,
        description="Optional spatial edge limit in unconverted coordinate units: one value gives a maximum radius and two values retain an interval; setting it selects or prunes the applicable generic graph mode.",
    )
    graph_percentile: Optional[float] = Field(
        default=None,
        description="Optional percentile of generic-graph edge distances retained by Squidpy after graph construction; null applies no percentile pruning.",
    )
    graph_transform: Optional[str] = Field(
        default=None,
        description="Optional Squidpy connectivity transform, spectral or cosine; NetworkX metrics are unweighted but any change to the nonzero edge pattern can change their result.",
    )
    graph_set_diag: bool = Field(
        default=False,
        description="Ask Squidpy to include self-connectivity before conversion; the active stage subsequently removes every diagonal entry, so self-loops never enter the NetworkX metrics.",
    )

    # Metrics
    minimum_cells_per_population: int = Field(
        default=5,
        description="Minimum number of cells of a population required within an ROI before calculating average clustering on that population's induced subgraph.",
    )

    # Bootstrapping / threading
    run_bootstrap: bool = Field(
        default=True,
        description="Compare observed graph metrics with a fixed-graph null generated by permuting population labels within each ROI.",
    )
    bootstrap_n_permutations: int = Field(
        default=1000,
        description="Number of within-ROI population-label permutations used to calculate null means, standard deviations, deltas, and z-scores.",
    )
    bootstrap_static_populations: List[str] = Field(
        default_factory=list,
        description="Population labels kept on their original nodes while all remaining labels are permuted among non-static cells within each ROI.",
    )
    bootstrap_ddof: int = Field(
        default=1,
        description="Delta degrees of freedom used for the standard deviation across permutation metric values.",
    )
    bootstrap_seed: Optional[int] = Field(
        default=12345,
        description="Base random seed from which deterministic, independent ROI permutation streams are spawned; null requests non-deterministic streams.",
    )
    n_threads: int = Field(
        default=-1,
        description="Number of ROI analyses run concurrently; -1 uses available SLURM or local CPU capacity, capped at the number of ROIs.",
    )
    save_bootstrap_samples: bool = Field(
        default=False,
        description="Save every ROI-level and derived case-level permutation metric value in long CSV tables in addition to compact summaries.",
    )

    # Plotting
    make_plots: bool = Field(
        default=True,
        description="Generate assortativity and per-population average-clustering summary figures from the saved ROI or case tables.",
    )
    plot_kind: str = Field(
        default='barplot',
        description="Summary display type: barplot for means with error bars or boxplot for distributions.",
    )
    plot_summary_level: str = Field(
        default='case_if_available',
        description="Unit plotted: case_if_available prefers case means and falls back to ROIs, while case and roi request those levels explicitly.",
    )
    plot_value_columns: List[str] = Field(
        default_factory=lambda: ['observed', 'zscore'],
        description="Summary metrics plotted from observed, bootstrap_mean, delta, and zscore; invalid names are ignored and an empty valid selection falls back to observed and zscore.",
    )
    make_all_populations_plots: bool = Field(
        default=True,
        description="Plot average-clustering results for all selected populations on one axis, using groupby_obs as hue when available.",
    )
    all_populations_plot_populations: List[str] = Field(
        default_factory=list,
        description="Optional ordered population subset shown in combined average-clustering plots; an empty list shows every population in annotation order.",
    )
    all_populations_figsize: Optional[List[float]] = Field(
        default=None,
        description="Optional fixed width and height in inches for combined population plots; null calculates width from the number of displayed populations.",
    )
    make_population_group_plots: bool = Field(
        default=True,
        description="When groupby_obs is available, save one average-clustering figure per population comparing the configured groups.",
    )
    make_assortativity_group_plots: bool = Field(
        default=True,
        description="Save whole-graph assortativity plots, stratified by groupby_obs when that metadata is available.",
    )
    barplot_figsize: List[float] = Field(
        default_factory=lambda: [4.0, 3.0],
        description="Base width and height in inches for NetworkX spatial summary plots.",
    )
    all_populations_width_scale: float = Field(
        default=0.45,
        description="Automatic combined-population plot width in inches per population, subject to barplot_figsize as a minimum.",
    )
    barplot_add_points: bool = Field(
        default=True,
        description="Overlay individual case or ROI observations on summary plots so replication and heterogeneity remain visible.",
    )
    figure_extension: str = Field(
        default='.png',
        description="File extension, including the leading dot, used for NetworkX spatial figures.",
    )
    figure_dpi: int = Field(
        default=300,
        description="Raster resolution in dots per inch for saved NetworkX spatial figures.",
    )

@config_section("remap_obs")
class RemapObsConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(default=None, description="Optional AnnData input override; when omitted, the stage uses general.anndata_path and updates that loaded object in apply mode.")
    remap_csv_path: str = Field(default='metadata/remap.csv', description="Editable CSV mapping table; relative paths are resolved from the working directory.")
    mode: str = Field(default='apply', description="Use 'generate_blank' to scaffold or refresh an editable mapping table, or 'apply' to add its curated target columns to adata.obs.")

    # Mapping behavior
    source_obs: Optional[str] = Field(default=None, description="Observation containing the source categories. Required for template generation; in apply mode it can be inferred from the CSV's first column and must match that header when explicitly set.")
    roi_obs: Optional[str] = Field(default=None, description="ROI observation used for the template evenness helper; defaults to general.roi_obs.")
    overwrite_existing_obs_columns: bool = Field(default=False, description="Allow target columns from the remap CSV to replace existing adata.obs columns; false protects existing annotations.")
    require_complete_mapping: bool = Field(default=False, description="Fail apply mode when any cell with a non-null source value lacks a non-null target mapping; otherwise those cells receive missing values.")
    set_output_as_categorical: bool = Field(default=True, description="Store applied target observations as ordered pandas categoricals using the non-null target-value order in the CSV.")
    force_string_mapping: bool = Field(default=False, description="Normalise integer-like source keys such as 1, 1.0, and '1' to the same string key; this behaviour is automatically enabled for source names containing 'leiden'.")
    ignore_csv_columns_exact: List[str] = Field(default_factory=list, description="Additional CSV columns to retain for human guidance but exclude from AnnData observation creation during apply mode.")
    ignore_csv_columns_contains: List[str] = Field(default_factory=lambda: ['notes'], description="Case-insensitive name fragments identifying human-only CSV columns that should not be applied to adata.obs.")

    # Blank-template generation
    generate_columns: List[str] = Field(default_factory=list, description="Blank target annotation columns added to a generated template; an empty list creates '<source_obs>_label'.")
    generate_note_columns: List[str] = Field(default_factory=lambda: ['notes'], description="Blank human-curation columns added to the template and ignored during application by the default ignore rule.")
    generate_include_counts: bool = Field(default=True, description="Include the total number of cells assigned to each source category as a curation aid.")
    generate_count_column_name: str = Field(default='n_cells', description="Column name for source-category cell counts in a generated template.")
    generate_include_top_markers: bool = Field(default=True, description="Include markers with the largest group-mean minus rest-mean values for each source category as descriptive naming hints.")
    generate_top_markers_n: int = Field(default=3, description="Maximum number of marker names listed for each source category in the generated template.")
    generate_top_markers_column_name: str = Field(default='top_markers', description="Column name for the generated descriptive top-marker hints.")
    generate_top_markers_use_raw: bool = Field(default=False, description="Use adata.raw.X for marker hints when available and no explicit generate_top_markers_layer is selected.")
    generate_top_markers_layer: Optional[str] = Field(default=None, description="Explicit matrix used for marker hints: 'X', 'raw', or an AnnData layer name; null uses adata.X unless generate_top_markers_use_raw is enabled.")
    generate_top_markers_var_column: Optional[str] = Field(default=None, description="Optional adata.var or adata.raw.var column supplying display marker names instead of var_names.")
    generate_top_markers_separator: str = Field(default='; ', description="Text separator placed between marker names in the generated top-marker cell.")
    generate_include_roi_distribution_evenness: bool = Field(default=True, description="Include a normalised Shannon-evenness summary of how each source category's cells are distributed across all dataset ROIs.")
    generate_roi_distribution_evenness_column_name: str = Field(default='roi_distribution_evenness', description="Column name for the generated ROI-distribution evenness helper.")
    generate_preserve_existing_values: bool = Field(default=True, description="When regenerating an existing template, carry forward edited target, note, and additional non-helper columns matched by normalised source key.")

@config_section("subclustering")
class SubclusteringConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData input override. When omitted, the stage uses general.anndata_path.",
    )
    output_adata_path: Optional[str] = Field(
        default=None,
        description="Optional AnnData output override. When omitted, the input AnnData is updated in place.",
    )
    output_subdir: str = Field(
        default='subclustering',
        description="Directory containing the reusable settings, marker-list, remap, and mapping CSV files.",
    )
    mode: Any = Field(
        default='generate',
        description="Checkpoint selection: 'generate' runs template creation or subclustering, 'apply' applies the curated remap, 'all' runs all available checkpoints, and 1, 2, or 3 selects one checkpoint directly.",
    )

    # Template/remap files
    settings_filename: str = Field(
        default='sublustering_settings.csv',
        description="Editable per-population settings table. The legacy 'sublustering' spelling is retained for compatibility.",
    )
    marker_list_filename: str = Field(
        default='marker_list.csv',
        description="Editable marker-selection table whose Boolean columns beginning with 'markers_' define alternative feature panels.",
    )
    remap_filename: str = Field(
        default='subcluster_to_final_population.csv',
        description="Editable table mapping each generated subcluster to a biologically curated final population label.",
    )
    master_index_mapping_filename: str = Field(
        default='master_index_to_final_population.csv',
        description="Output audit table linking each cell's stable master index to its final population label.",
    )

    # Subclustering defaults
    base_label_key: Optional[str] = Field(
        default=None,
        description="Parent-population observation used to define subsets. Defaults to general.population_obs_primary and then the legacy 'population' column.",
    )
    default_resolution: float = Field(
        default=0.3,
        description="Initial Leiden resolution written to newly generated settings rows; larger values usually produce finer partitions but do not specify an exact number of subclusters.",
    )
    default_marker_list: str = Field(
        default='all',
        description="Initial marker-list name written to settings rows; for example 'all' selects the marker_list.csv column 'markers_all'.",
    )
    use_rep: Optional[str] = Field(
        default='X_biobatchnet',
        description="Preferred AnnData representation for computing a missing global QC UMAP. Subcluster neighbour graphs themselves are built from the selected marker values in adata.X.",
    )

    # Plotting and QC
    compute_umap_if_missing: bool = Field(
        default=True,
        description="Compute a global UMAP for diagnostic plots when adata.obsm['X_umap'] is absent.",
    )
    umap_dot_size: float = Field(default=2.0, description="Point size used in subclustering UMAP diagnostics.")
    matrixplot_vmax: float = Field(
        default=0.5,
        description="Upper colour limit used for raw-value marker matrix plots; tune it to the scale stored in adata.X.",
    )
    save_individual_umaps: bool = Field(
        default=True,
        description="Save a separate global UMAP highlighting the cells in every generated subcluster.",
    )
    figure_extension: str = Field(
        default='.png',
        description="File extension for subclustering diagnostic figures, including the leading dot.",
    )
    figure_dpi: int = Field(default=300, description="Resolution in dots per inch for saved diagnostic figures.")

    # Final remap integration
    final_label_key: str = Field(
        default='population_final',
        description="AnnData observation column receiving the curated final population labels during the apply checkpoint.",
    )
    master_index_obs: Optional[str] = Field(
        default=None,
        description="Stable cell-identifier observation used in the exported mapping table. Defaults to general.master_index_obs and then 'Master_Index'; observation names are used if neither exists.",
    )
    apply_remap_only_if_modified: bool = Field(
        default=True,
        description="Skip remap application until at least one final_population value differs from its generated subcluster label, protecting an unreviewed template from being treated as curated annotation.",
    )


@config_section("napari_sbt")
class NapariSBTConfig(ConfigModel):
    """Project defaults for cohort-first Napari experiments and cell features."""

    experiment_folder: str = config_field(
        "napari_sbt",
        description=(
            "Project-relative folder containing versioned napari_sbt experiment "
            "manifests, frozen cohorts, labels, models, and reusable feature assets."
        ),
        level="basic",
        stage="cellfeat",
        ui_group="Experiment assets",
        advice="Keep this folder under the project root so managed and local runs agree.",
    )
    active_experiment: Optional[str] = config_field(
        None,
        description=(
            "Active experiment directory or experiment.yaml path. Relative values "
            "are resolved first below napari_sbt.experiment_folder."
        ),
        level="basic",
        stage="cellfeat",
        ui_group="Experiment selection",
        advice="Create and confirm an experiment in the Napari Setup tab before cellfeat.",
    )
    worker_count: int = config_field(
        8,
        description="Maximum local or managed ROI feature-extraction worker processes.",
        level="basic",
        stage="cellfeat",
        ui_group="Feature execution",
        advice="Each worker loads one full ROI mask plus one selected image at a time.",
        ge=1,
    )
    annotated_adata_path: str = config_field(
        "napari_sbt_annotated.h5ad",
        description=(
            "Default project-relative destination for an atomic annotated AnnData "
            "copy exported by napari_sbt."
        ),
        level="basic",
        stage="napari_sbt",
        ui_group="Classification export",
        advice="Use a different path from general.anndata_path; source AnnData is not overwritten.",
    )


@config_section("logging")
class LoggingConfig(ConfigModel):
    log_file: str = 'pipeline.log'
    level: str = 'INFO'
    to_console: bool = True
    console_only: bool = False  # Only log to console, not file (useful for SLURM jobs)
    prevent_duplicate_console: bool = True  # Prevent double console output
    use_custom_format: bool = True  # Use custom format vs basicConfig default

@config_section("pipeline")
class PipelineConfig(ConfigModel):
    """Fully resolved, typed configuration for all pipeline stages."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    rebuild_metadata: RebuildMetadataConfig = Field(default_factory=RebuildMetadataConfig)
    denoising: DenoisingConfig = Field(default_factory=DenoisingConfig)
    createmasks: CreateMasksConfig = Field(default_factory=CreateMasksConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    nimbus: NimbusConfig = Field(default_factory=NimbusConfig)
    nimbus_normalization_scan: NimbusNormalizationScanConfig = Field(
        default_factory=NimbusNormalizationScanConfig
    )
    neighbour_signal: NeighbourSignalConfig = Field(default_factory=NeighbourSignalConfig)
    batch_integration: BatchIntegrationConfig = Field(default_factory=BatchIntegrationConfig)
    rapids: RapidsProcessConfig = Field(default_factory=RapidsProcessConfig)
    maxfuse: MaxFuseConfig = Field(default_factory=MaxFuseConfig)
    spatialdata: SpatialDataConfig = Field(default_factory=SpatialDataConfig)
    cellvision: CellVisionConfig = Field(default_factory=CellVisionConfig)
    hyperstac: HyperstacConfig = Field(default_factory=HyperstacConfig)
    cox: CoxConfig = Field(default_factory=CoxConfig)
    biobatchnet: BioBatchNetConfig = Field(default_factory=BioBatchNetConfig)
    process: BasicProcessConfig = Field(default_factory=BasicProcessConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    population_embedding_qc: PopulationEmbeddingQCConfig = Field(default_factory=PopulationEmbeddingQCConfig)
    cellcharter: CellCharterConfig = Field(default_factory=CellCharterConfig)
    starling: StarlingConfig = Field(default_factory=StarlingConfig)
    pairwise_spatial: PairwiseSpatialConfig = Field(default_factory=PairwiseSpatialConfig)
    networkx_spatial: NetworkxSpatialConfig = Field(default_factory=NetworkxSpatialConfig)
    remap_obs: RemapObsConfig = Field(default_factory=RemapObsConfig)
    subclustering: SubclusteringConfig = Field(default_factory=SubclusteringConfig)
    napari_sbt: NapariSBTConfig = Field(default_factory=NapariSBTConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


DEFAULT_CONFIG_CLASSES = {
    "general": GeneralConfig,
    "preprocess": PreprocessConfig,
    "rebuild_metadata": RebuildMetadataConfig,
    "denoising": DenoisingConfig,
    "createmasks": CreateMasksConfig,
    "segmentation": SegmentationConfig,
    "nimbus": NimbusConfig,
    "nimbus_normalization_scan": NimbusNormalizationScanConfig,
    "neighbour_signal": NeighbourSignalConfig,
    "batch_integration": BatchIntegrationConfig,
    "rapids": RapidsProcessConfig,
    "maxfuse": MaxFuseConfig,
    "spatialdata": SpatialDataConfig,
    "cellvision": CellVisionConfig,
    "hyperstac": HyperstacConfig,
    "cox": CoxConfig,
    "biobatchnet": BioBatchNetConfig,
    "process": BasicProcessConfig,
    "visualization": VisualizationConfig,
    "population_embedding_qc": PopulationEmbeddingQCConfig,
    "cellcharter": CellCharterConfig,
    "starling": StarlingConfig,
    "pairwise_spatial": PairwiseSpatialConfig,
    "networkx_spatial": NetworkxSpatialConfig,
    "remap_obs": RemapObsConfig,
    "subclustering": SubclusteringConfig,
    "napari_sbt": NapariSBTConfig,
    "logging": LoggingConfig,
}


__all__ = [
    "BasicProcessConfig",
    "BatchIntegrationConfig",
    "BioBatchNetConfig",
    "CellCharterConfig",
    "ConfigModel",
    "CoxConfig",
    "CoxFeatureSourceConfig",
    "CreateMasksConfig",
    "DEFAULT_CONFIG_CLASSES",
    "DenoisingConfig",
    "GeneralConfig",
    "HyperstacConfig",
    "LoggingConfig",
    "MaxFuseConfig",
    "MaxFuseGeneListConfig",
    "NapariSBTConfig",
    "NetworkxSpatialConfig",
    "NeighbourSignalConfig",
    "NimbusConfig",
    "PairwiseSpatialConfig",
    "PipelineConfig",
    "PopulationEmbeddingQCConfig",
    "PreprocessConfig",
    "RapidsProcessConfig",
    "RebuildMetadataConfig",
    "RemapObsConfig",
    "SegmentationConfig",
    "StarlingConfig",
    "SpatialDataConfig",
    "SpatialDataHistologyConfig",
    "SpatialDataImagePanelConfig",
    "SpatialDataMaxFuseTableConfig",
    "SpatialDataRegionLabelsConfig",
    "SubclusteringConfig",
    "VisualizationConfig",
    "config_field",
    "config_section",
]
