"""Pydantic v2 models for the complete imcanalysis pipeline configuration.

The field names, defaults, and section structure intentionally mirror the legacy
configuration dataclasses from scripts.config_and_utils. Keep compatibility
first when extending these models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

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
    return Field(
        default=default,
        description=description,
        json_schema_extra={
            "level": level,
            "stage": stage,
            "ui_group": ui_group,
            "advice": advice,
        },
        **field_kwargs,
    )


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
            "Root folder for numbered, human-facing stage reports, figures, and tables."
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
            "New managed runs use general.outputs_folder and numbered stage folders. "
            "Do not repurpose this field for new output layouts."
        ),
    )
    masks_folder: str = 'masks'
    celltable_folder: str = 'cell_tables'
    tiff_stacks_folder: str  = 'tiff_stacks'
    raw_images_folder: str = 'tiffs'
    denoised_images_folder: str = 'processed'
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
        description="Minimum accepted ROI width and height in pixels.",
        json_schema_extra={
            "level": "basic",
            "stage": "preprocess",
            "ui_group": "Input filtering",
            "advice": "Reduce only when deliberately processing small ROIs.",
        },
    )

@config_section("rebuild_metadata")
class RebuildMetadataConfig(ConfigModel):
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_metadata_folder: Optional[str] = None  # Optional override (None = use general.metadata_folder)
    include_obs_patterns: Optional[List[str]] = None  # Optional regex allowlist for ROI-invariant obs columns
    exclude_obs: List[str] = Field(default_factory=lambda: [
        'ObjectNumber',
        'CellID',
        'cell_id',
        'Master_Index',
        'X_loc',
        'Y_loc',
    ])
    exclude_obs_contains: List[str] = Field(default_factory=lambda: [
        'population',
        'leiden',
        'cluster',
        'nhood',
        'neighborhood',
    ])
    preserve_existing_import_data: bool = True  # Keep prior metadata.csv import_data values where ROI names match
    metadata_description_obs: Optional[str] = None  # Optional ROI-invariant obs column used for metadata description
    include_invariant_obs_in_metadata_csv: bool = True
    include_invariant_obs_in_dictionary_csv: bool = True
    panel_channel_name_var: Optional[str] = None  # Optional adata.var column for panel channel_name values
    panel_channel_label_var: Optional[str] = None  # Optional adata.var column for panel channel_label values
    panel_use_denoised_default: bool = True
    panel_use_raw_default: bool = False
    panel_to_denoise_default: bool = True
    panel_remove_outliers_default: bool = False
    preserve_existing_panel_flags: bool = True  # Keep prior panel use_* flags when channel labels match

@config_section("denoising")
class DenoisingConfig(ConfigModel):
    run_denoising: bool = True
    method: str = 'deep_snf'  # Options: 'deep_snf', 'dimr'
    channels: List[str] = Field(default_factory=list)
    # Parameters for both methods
    n_neighbours: int = 4
    n_iter: int = 3
    window_size: int = 3
    # Outlier removal
    remove_outliers: bool = True
    remove_outliers_min_threshold: int = 500
    # Parameters specific to 'deep_snf' method
    patch_step_size: int = 100
    intelligent_patch_size: bool = True
    intelligent_patch_size_threshold: float = 0.3  # e.g., 20%
    intelligent_patch_size_minimum: int = 40
    intelligent_patch_size_min_patches: int = 5000  # Minimum number of patches required
    intelligent_patch_size_max_patches: Optional[int] = None  # Maximum number of patches (None = no limit)
    # DeepSNIF
    train_epochs: int = 75
    train_initial_lr: float = 0.001
    train_batch_size: int = 200
    ratio_thresh: float = 0.8 # Added
    pixel_mask_percent: float = 0.2
    val_set_percent: float = 0.15
    loss_function: str = "I_divergence"
    loss_name: Optional[str] = None
    weights_save_directory: Optional[str] = None
    is_load_weights: bool = False
    lambda_HF: float = 3e-6
    network_size: str = "small"
    truncated_max_rate: float = 0.99999
    # Parameter scanning
    run_parameter_scan: bool = False
    scan_parameter: Optional[str] = 'truncated_max_rate'  # Name of parameter to scan (e.g., 'train_epochs', 'lambda_HF')
    scan_values: Optional[List[Any]] = Field(default_factory=lambda: [0.99, 0.999, 0.99999])  # List of values to test for the scan parameter
    # Training verbosity
    verbose_training: bool = False  # Show detailed TensorFlow/Keras training output (progress bars, epoch details)
    # Parameters for QC images
    run_QC: bool = True
    colourmap: str = "jet"
    dpi: int = 100
    qc_image_dir: str = 'denoising'
    qc_num_rois: Optional[int] = 10  # Number of random ROIs to include in QC (None = all ROIs)
    skip_already_denoised: bool = True

@config_section("createmasks")
class CreateMasksConfig(ConfigModel):
    specific_rois: Optional[List[str]] = None
    dna_image_name: str = 'DNA1'
    dna_preprocessing_output_folder_name: str = 'preprocessed_dna'  # For DNA preprocessing output
    cellpose_cell_diameter: float = Field(
        default=10.0,
        gt=0,
        description="Approximate Cellpose cell diameter in pixels.",
        json_schema_extra={
            "level": "basic",
            "stage": "createmasks",
            "ui_group": "Segmentation",
            "advice": (
                "Increase when cells are fragmented; decrease when neighbouring "
                "cells are merged."
            ),
        },
    )  # Works in both CellPose v3 and v4+ (behavior may differ)
    upscale_ratio: float = 1.7
    expand_masks: int = 1
    perform_qc: bool = True
    qc_boundary_dilation: int = 0
    min_cell_area: Optional[int] = 15
    max_cell_area: Optional[int] = 200
    cell_pose_model: str = 'nuclei'  # For CellPose v3 (original createmasks) - DEPRECATED in v4+
    cell_pose_sam_model: str = 'cpsam'  # For CellPose v4+ (cellpose_sam script) - only 'cpsam' or user models
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    run_deblur: bool = True
    run_upscale: bool = True
    image_normalise: bool = True
    image_normalise_percentile_lower: float = 0.0
    image_normalise_percentile_upper: float =  99.9
    dpi_qc_images: int = 300

    # CellPose-SAM mode toggle and settings - uses dna_preprocessing_output_folder_name for input, GeneralConfig.masks_folder for output
    max_size_fraction: float = 0.4              # Max cell size as fraction of image
    remove_edge_masks: bool = False         # Remove masks touching image edges
    fill_holes: bool = True                 # Fill holes in segmented masks
    batch_size: int = 128                     # Batch size for segmentation
    resample: bool = True                   # Resample for better boundaries
    augment: bool = False                   # Use test-time augmentation
    tile_overlap: float = 0.1               # Overlap fraction for tiling

    # Upscale model configuration
    upscale_model_type: str = 'upsample_nuclei'  # 'upsample_nuclei' or 'upsample_cyto3'

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
    run_parameter_scan: bool = False
    param_a: Optional[str] = 'cellprob_threshold'
    param_a_values: Optional[List[Any]] = Field(default_factory=lambda: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    param_b: Optional[str] = 'flow_threshold'
    param_b_values: Optional[List[Any]] = Field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    window_size: Optional[int] = 250
    num_rois_to_scan: int = 3
    scan_rois: Optional[List[str]] = None

@config_section("segmentation")
class SegmentationConfig(ConfigModel):
    celltable_output: str = 'celltable.csv'
    marker_normalisation: List[str] = Field(default_factory=lambda: ["q0.999"])
    store_raw_marker_data: bool = False
    remove_channels_list: List[str] = Field(default_factory=lambda: ['DNA1', 'DNA3'])
    remove_and_store_markers: List[str] = Field(default_factory=list)  # Markers to remove from main AnnData and store separately
    removed_markers_anndata_path: str = 'anndata_removed.h5ad'  # Path for AnnData containing removed markers
    anndata_save_path: str = 'anndata.h5ad'
    create_roi_cell_tables: bool = True
    create_master_cell_table: bool = True
    create_anndata: bool = True
    allow_missing_channels: bool = False  # If True, fill missing channels with NaN; if False, only include channels present in all ROIs

@config_section("nimbus")
class NimbusConfig(ConfigModel):
    output_dir: str = 'nimbus_output'
    roi_table_subfolder: str = 'nimbus_cell_tables'
    master_celltable: str = 'nimbus_celltable.csv'
    master_classic_celltable: str = 'nimbus_classic_celltable.csv'
    master_expansion_celltable: str = 'nimbus_expansion_celltable.csv'
    anndata_output: str = 'anndata.h5ad'
    roi_table_prefix: str = 'nimbus_'
    use_denoised_first: bool = True
    allow_raw_fallback: bool = True
    simple_image_names: bool = False  # If True, match images by channel_label only (instead of channel_name_channel_label)
    mask_extensions: List[str] = Field(default_factory=lambda: ['.tiff', '.tif'])
    mask_boundary_offset_pixels: int = 0  # Positive expands masks; negative shrinks masks before Nimbus/cell-table extraction
    min_cell_area: Optional[int] = None  # Drop cells smaller than this post-offset mask area in pixels (None = no lower bound)
    max_cell_area: Optional[int] = None  # Drop cells larger than this post-offset mask area in pixels (None = no upper bound)
    test_time_augmentation: bool = True
    batch_size: int = 10
    model_magnification: int = 10
    dataset_magnification: int = 10
    checkpoint: str = 'latest'
    device: str = 'auto'
    normalization_quantile: float = 0.999
    normalization_subset: int = 10
    normalization_jobs: int = 1
    normalization_clip: List[float] = Field(default_factory=lambda: [0.0, 1.0])
    normalization_min_value: float = 3.0  # Minimum normalization value to avoid background noise
    reuse_saved_normalization: bool = False  # Reuse existing normalization_dict.json if found (allows manual tweaking)
    norm_dict_qc_only: bool = False  # If True, stop after normalization dict computation and QC generation
    save_prediction_maps: bool = False
    allow_prediction_resize: bool = False  # If True, fall back to resizing predictions when shapes mismatch
    use_existing_master_celltables: bool = False  # If True, reuse existing master cell tables when found
    extract_classic_intensities: bool = True  # Extract classic mean intensities over masks
    extract_expansion_intensities: bool = True  # Extract mean intensities from expanded masks
    expansion_pixels: int = 10  # Number of pixels to expand masks for expansion intensities
    expansion_jobs: int = 1  # Number of parallel jobs for expansion extraction (1=sequential, -1=all CPUs)

@config_section("batch_integration")
class BatchIntegrationConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)

    # Core integration settings
    batch_correction_obs: Optional[str] = None
    integration_method: str = 'harmony'  # Options: 'harmony', 'bbknn', 'both', 'none'
    batch_correction_method: Optional[str] = None  # Deprecated alias for integration_method
    n_for_pca: Optional[int] = None
    leiden_resolutions_list: List[float] = Field(default_factory=lambda: [0.3, 1.0])
    umap_min_dist: float = 0.1
    run_leiden: bool = True
    n_neighbors: Optional[int] = None

    # Embedding storage
    pca_key: str = 'X_pca'
    harmony_key: str = 'X_pca_harmony'
    representation_key: str = 'X_batch_integration'
    qc_output_subdir: str = 'BatchIntegration'

    # Method-specific parameters
    harmony_params: Dict[str, Any] = Field(default_factory=lambda: {
        'max_iter_harmony': 30,
        'verbose': True,
        'random_state': 0,
        'device': None,
    })
    bbknn_params: Dict[str, Any] = Field(default_factory=dict)

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
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)

    # Core RAPIDS processing settings
    batch_correction_obs: Optional[str] = None  # Required only when run_harmony=True
    run_harmony: bool = False
    harmony_flavor: str = 'harmony2'  # Options: 'harmony2' (default), 'harmony1'
    n_for_pca: Optional[int] = None
    n_pcs_neighbors: Optional[int] = None
    leiden_resolutions_list: List[float] = Field(default_factory=lambda: [0.3, 1.0])
    umap_min_dist: float = 0.1
    run_leiden: bool = True
    n_neighbors: Optional[int] = None

    # Optional obs-based cell filter applied immediately after loading AnnData
    filter_obs_key: str = 'mask_area'
    filter_min_value: Optional[float] = None
    filter_max_value: Optional[float] = None

    # Optional parameter scan. Values should be lists keyed by supported scan
    # parameters: n_neighbors, n_for_pca, umap_min_dist, run_harmony, harmony_flavor.
    parameter_scan_dict: Dict[str, Any] = Field(default_factory=dict)
    parameter_scan_save_anndata: bool = False
    parameter_scan_qc_subdir: str = 'ParameterScan'

    # Embedding / graph storage
    input_representation_key: Optional[str] = None  # Existing adata.obsm key to use instead of PCA/Harmony
    pca_key: str = 'X_pca'
    harmony_key: str = 'X_pca_harmony'
    representation_key: str = 'X_batch_integration'
    neighbors_key: Optional[str] = None  # None uses standard adata.uns['neighbors'] / obsp keys
    umap_key: Optional[str] = None  # None uses standard adata.obsm['X_umap']
    qc_output_subdir: str = 'RapidsProcess'

    # RAPIDS pass-through parameters. Keys controlled by the config above
    # (e.g. n_comps, key_added, use_rep) are intentionally ignored by the script.
    pca_params: Dict[str, Any] = Field(default_factory=dict)
    harmony_params: Dict[str, Any] = Field(default_factory=lambda: {
        'max_iter_harmony': 30,
        'random_state': 0,
        'verbose': True,
        'dtype': 'float32',
    })
    neighbors_params: Dict[str, Any] = Field(default_factory=dict)
    umap_params: Dict[str, Any] = Field(default_factory=dict)
    leiden_params: Dict[str, Any] = Field(default_factory=dict)


@config_section("biobatchnet")
class BioBatchNetConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)

    batch_correction_obs: Optional[str] = None
    n_for_pca: Optional[int] = None
    leiden_resolutions_list: List[float] = Field(default_factory=lambda: [0.3, 1.0])
    umap_min_dist: float = 0.1

    # BioBatchNet-specific parameters (nested dictionary format)
    biobatchnet_params: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        'data_type': 'imc',
        'latent_dim': 20,
        'epochs': 100,
        'device': None,
        'use_raw': False,
        'extra_params': {
            'loss_weights': {
                'recon_loss': 100.0,
                'discriminator': 0.05,  # Batch mixing (default: 0.3 - lower = more mixing)
                'classifier': 1.0,  # Batch retention (default: 1)
                'kl_loss_1': 0.0005,  # KL divergence for bio encoder (default: 0.005)
                'kl_loss_2': 0.1,  # KL divergence for batch encoder (default: 0.1)
                'ortho_loss': 0.01,  # Orthogonality constraint (default: 0.01)
            }
        },
    })

    # BioBatchNet parameter scanning
    biobatchnet_scan_parameter_sets: Optional[List[Dict[str, Any]]] = None
    biobatchnet_scan_include_base: bool = True
    biobatchnet_run_postprocess: bool = True
    biobatchnet_run_leiden: bool = True

    # Scanpy neighbors computation
    n_neighbors: Optional[int] = None

    # Deprecated flat-style parameters (auto-migrated into biobatchnet_params)
    biobatchnet_data_type: Optional[str] = None
    biobatchnet_latent_dim: Optional[int] = None
    biobatchnet_epochs: Optional[int] = None
    biobatchnet_device: Optional[str] = None
    biobatchnet_kwargs: Optional[Dict[str, Any]] = None
    biobatchnet_use_raw: Optional[bool] = None

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
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    population_columns: Optional[List[str]] = None  # Script override for population columns (None = use general.population_obs_all or auto-detect)
    metadata_columns: Optional[List[str]] = None  # Script override for metadata columns (None = use general.metadata_obs or auto-detect)
    groupby_obs: Optional[str] = None  # Script override for grouping column (None = use general.groupby_obs)
    groupby_obs_groups: Optional[List[str]] = None  # Script override for groups (None = use general pairwise/groups settings)

    # AI interpretation settings
    enable_ai: bool = True  # Enable AI-powered cluster interpretation
    tissue: str = "Unknown tissue"  # Tissue type for AI interpretation context
    repeat_ai_interpretation: bool = False  # Re-run AI interpretation even if labels already exist

    # Visualization module toggles - all default True
    create_umaps: bool = True  # Create UMAP plots for populations and markers
    create_matrix_plots: bool = True  # Create MatrixPlot summaries
    create_tissue_overlays: bool = True  # Create tissue population overlays
    create_population_analysis: bool = True  # Create population analysis across metadata
    create_backgating: bool = True  # Create backgating assessment
    create_color_legends: bool = True  # Generate color legends for categorical plots

    # Categorical visualization controls
    include_metadata_umaps: bool = True  # Include metadata columns in UMAP plots
    include_metadata_matrix_plots: bool = True  # Include metadata columns in MatrixPlots
    include_marker_umaps: bool = True  # Include marker expression UMAPs
    umap_plot_individual_highlights: bool = True  # For population columns, create one UMAP per category via utils.plot_umap_highlight_clusters
    max_categories: int = 50  # Maximum number of unique categories for population/metadata columns
    umap_marker_colormap: str = 'viridis'  # Colormap for marker expression UMAPs (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis')
    umap_marker_gallery_default_colorbar_label: str = 'Nimbus-Inference Score'  # Colorbar label for default-layer (adata.X) marker gallery
    umap_marker_gallery_vmax: Optional[float] = 0.8  # Maximum value for colorbar scaling in UMAP marker gallery

    # Backgating assessment settings
    backgating_cells_per_group: int = 50  # Number of cells to sample per population for backgating
    backgating_radius: int = 15  # Radius in pixels for cell thumbnail extraction
    backgating_output_folder: str = 'Backgating'  # Output folder for backgating results
    backgating_use_masks: bool = True  # Whether to use segmentation masks in backgating
    backgating_mask_folder: str = 'masks'  # Folder containing segmentation masks
    backgating_pops_list: Optional[Dict[str, Any]] = None  # Optional dict mapping population obs columns to population subsets for backgating; non-dict values are reused for all population obs
    backgating_max_rois_to_save: Optional[int] = None  # Maximum number of per-population ROIs to save (None = save all; normalization still uses the full ROI set)

    # Backgating intensity and marker settings
    backgating_minimum: float = 0.2  # Minimum intensity for backgating normalization
    backgating_max_quantile: str = 'i0.99'  # Maximum quantile method for intensity scaling
    backgating_number_top_markers: int = 2  # Number of top markers to use for RGB channels
    backgating_specify_blue: Optional[str] = 'DNA1'  # Marker to use for blue channel
    backgating_specify_red: Optional[str] = None  # Marker to use for red channel (None = auto-select)
    backgating_specify_green: Optional[str] = None  # Marker to use for green channel (None = auto-select)

    # Differential expression settings for backgating marker selection
    backgating_use_differential_expression: bool = True  # Use scanpy DE analysis for marker selection
    backgating_de_method: str = 'wilcoxon'  # Statistical method ('wilcoxon', 't-test', 'logreg')
    backgating_min_logfc_threshold: float = 0.2  # Minimum log fold change for quality filtering (0 to disable)
    backgating_max_pval_adj: float = 0.05  # Used for significance reporting, not filtering
    backgating_markers_exclude: Optional[List[str]] = Field(default_factory=lambda: ['DNA1', 'DNA3'])  # Markers to exclude from DE analysis

    # Backgating execution mode control
    backgating_mode: str = 'full'  # 'full' (compute + run), 'save_markers' (compute only), 'load_markers' (load + run)

    # Population overlay visualization settings
    backgating_population_overlay_outline_width: int = 1  # Width of contour outlines in population overlay visualizations
    backgating_population_overlay_legend_fontsize: int = 24  # Font size for overlay legend labels
    backgating_population_overlay_crop_size: Optional[List[int]] = Field(default_factory=lambda: [300, 300])  # Crop size [width, height] or None
    backgating_population_overlay_crop_origin: str = 'intelligent'  # Crop anchor: upper_left/right, lower_left/right, center, intelligent
    backgating_population_overlay_show_scale_bar: bool = True  # Whether to draw scale bar on overlays
    backgating_population_overlay_scale_bar_length: int = 50  # Scale bar length in pixels
    backgating_population_overlay_scale_bar_thickness: int = 3  # Scale bar thickness in pixels

    # MatrixPlot settings
    matrixplot_vmax: float = 0.5  # Maximum value for non-scaled matrix plots
    matrixplot_use_row_colors: bool = True  # Use plotting.matrixplot_with_row_colors when available for MatrixPlot generation

    # Population abundance plotting (create_population_abundance_analysis)
    abundance_make_all_populations_plots: bool = True  # Also create combined plots with all populations on one axis (hue=groupby_obs)
    abundance_all_populations_figsize: List[float] = Field(default_factory=lambda: [4.0, 3.0])  # Base [width, height]; width auto-scales with number of populations
    abundance_all_populations_width_scale: float = 0.45  # Auto width ~= max(base_width, scale * n_populations)
    abundance_make_case_stacked_plots: bool = True  # Create case-level stacked proportion plots (all cases + split by groupby_obs)
    abundance_case_stacked_figsize: List[float] = Field(default_factory=lambda: [6.0, 3.0])  # Base [width, height]; width auto-scales with number of cases
    abundance_case_stacked_width_scale: float = 0.30  # Auto width ~= max(base_width, scale * n_cases)
    abundance_order_cases_by_population: Optional[str] = None  # Optional population label to order cases by descending abundance
    abundance_plot_style: str = 'bar'  # One of: 'bar', 'strip', 'swarm'; strip/swarm show individual points with mean +/- SE overlays
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
    })
    abundance_barplot_y_scale_intelligent_params: Dict[str, Any] = Field(default_factory=lambda: {
        'allow_log1p': True,
        'dynamic_range_thresh': 100.0,
        'skew_improve_ratio': 0.7,
        'crush_frac_thresh': 0.7,
    })

    # General visualization settings
    save_high_res: bool = True  # Save high-resolution figures (300 DPI)
    figure_format: str = 'png'  # Default figure format ('png', 'pdf', 'svg')

@config_section("cellcharter")
class CellCharterConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    qc_output_subdir: str = 'CellCharter_QC'

    # Features
    use_rep: Optional[str] = 'X_biobatchnet'      # For non-TRVAE mode: adata.obsm key for neighborhood aggregation (set None to disable)
    use_layer: Optional[str] = None    # For TRVAE or non-TRVAE mode: adata.layers key (None uses adata.X)
    scale_by_sample: bool = False       # In TRVAE mode: scale TRVAE input per sample; otherwise scale aggregation input
    scaled_rep_key: str = 'X_cellcharter_scaled'

    # TRVAE dimensionality reduction (default path, per CellCharter tutorial)
    use_trvae: bool = False
    trvae_latent_key: str = 'X_trVAE'
    trvae_condition_key: Optional[str] = 'dataset'
    trvae_use_sample_key_fallback: bool = True
    trvae_constant_condition_label: str = 'all'
    trvae_load_path: Optional[str] = None  # Optional pretrained model directory
    trvae_save_path: str = 'trvae_model'   # Reusable project-root model directory when relative
    trvae_map_location: str = 'gpu'
    trvae_train: bool = True
    trvae_train_early_stopping: bool = False
    trvae_train_enable_progress_bar: bool = True
    trvae_train_max_epochs: Optional[int] = None
    trvae_hidden_layer_sizes: List[int] = Field(default_factory=lambda: [128, 128])
    trvae_latent_dim: int = 10
    trvae_dr_rate: float = 0.05
    trvae_use_mmd: bool = True
    trvae_mmd_on: str = 'z'
    trvae_mmd_boundary: Optional[int] = None
    trvae_recon_loss: str = 'mse'
    trvae_beta: float = 1.0
    trvae_use_bn: bool = False
    trvae_use_ln: bool = True

    # Graph and neighborhood aggregation
    delaunay: bool = True
    remove_long_links: bool = True
    distance_percentile: float = 99.0
    n_layers: int = 3
    aggregations: str = 'mean'         # 'mean' or list-like string via overrides
    aggregated_rep_key: str = 'X_cellcharter'

    # Clustering
    n_clusters: int = 11
    random_state: int = 12345
    covariance_type: str = 'full'
    batch_size: Optional[int] = None
    trainer_accelerator: str = 'auto'
    trainer_devices: Optional[int] = None
    trainer_max_epochs: int = 100
    cluster_key: str = 'spatial_cluster'
    repeat_analysis: Optional[bool] = None  # Deprecated fallback for stage-specific repeat flags (None = ignore, otherwise used where stage-specific flags are unset)
    repeat_cluster_analysis: Optional[bool] = None  # If False and cluster_key already exists, reuse existing cluster labels instead of rerunning TRVAE/graph/aggregation/clustering
    repeat_enrichment_analysis: Optional[bool] = None  # If False and enrichment results already exist in adata.uns, reuse them instead of recomputing
    repeat_nhood_enrichment_analysis: Optional[bool] = None  # If False and CellCharter nhood enrichment results already exist in adata.uns, reuse them
    repeat_diff_nhood_enrichment_analysis: Optional[bool] = None  # If False and differential nhood enrichment results already exist in adata.uns, reuse them
    repeat_shape_characterisation_analysis: Optional[bool] = None  # If False and shape/component outputs already exist, reuse them instead of recomputing

    # Optional enrichment
    run_enrichment: bool = True
    enrichment_with_pvalues: bool = False
    enrichment_n_perms: int = 1000
    enrichment_plot_figsize: List[float] = Field(default_factory=lambda: [8.0, 6.0])
    enrichment_plot_dot_scale: float = 3.0
    enrichment_plot_show_pvalues: bool = False
    enrichment_plot_significant_only: bool = False

    # Neighborhood enrichment (CellCharter graph enrichment)
    run_nhood_enrichment: bool = True
    nhood_connectivity_key: Optional[str] = None
    nhood_log_fold_change: bool = False
    nhood_only_inter: bool = True
    nhood_symmetric: bool = False
    nhood_with_pvalues: bool = False
    nhood_n_perms: int = 1000
    nhood_n_jobs: int = 1
    nhood_batch_size: int = 10
    nhood_observed_expected: bool = True
    save_nhood_enrichment_plot: bool = True
    nhood_plot_figsize: List[float] = Field(default_factory=lambda: [6.0, 3.0])
    nhood_enrichment_significance: Optional[float] = None

    # Differential neighborhood enrichment by condition
    run_diff_nhood_enrichment: bool = False
    diff_nhood_condition_key: Optional[str] = None
    diff_nhood_condition_groups: Optional[List[str]] = None
    diff_nhood_connectivity_key: Optional[str] = None
    diff_nhood_log_fold_change: bool = False
    diff_nhood_only_inter: bool = True
    diff_nhood_symmetric: bool = False
    diff_nhood_with_pvalues: bool = False
    diff_nhood_library_key: Optional[str] = None
    diff_nhood_n_perms: int = 1000
    diff_nhood_n_jobs: Optional[int] = None
    diff_nhood_plot_ncols: int = 2
    save_diff_nhood_enrichment_plot: bool = True

    # Shape characterisation
    run_shape_characterisation: bool = False
    shape_component_key: str = 'component'
    shape_component_cluster_key: Optional[str] = None
    shape_connectivity_key: Optional[str] = None
    shape_min_cells: int = 250
    shape_min_hole_area_ratio: float = 0.1
    shape_alpha_start: int = 2000
    shape_compute_linearity: bool = True
    shape_linearity_key: str = 'linearity'
    shape_linearity_height: int = 1000
    shape_linearity_min_ratio: float = 0.05
    shape_compute_curl: bool = True
    shape_curl_key: str = 'curl'
    shape_plot_metrics: bool = True
    shape_metrics_condition_key: Optional[str] = None
    shape_metrics_condition_groups: Optional[List[str]] = None
    shape_metrics_cluster_key: Optional[str] = None
    shape_metrics_cluster_groups: Optional[List[str]] = None
    shape_metrics_ncols: int = 2

    # QC plotting
    save_spatial_plots: bool = True
    max_rois_for_plots: int = 12
    point_size: float = 2.0
    save_enrichment_heatmap: bool = True
    cluster_default_cmap: Optional[str] = None  # If None, use scanpy's godsnot_102 palette for adata.uns['{cluster_key}_colors']
    save_cluster_umap: bool = True
    cluster_umap_point_size: float = 10.0
    cluster_umap_legend_loc: str = 'right margin'
    save_cluster_composition_plots: bool = True
    composition_order_by_environment: str = '0'  # Cluster label used to order case stacked bars by abundance
    composition_stacked_figsize: List[float] = Field(default_factory=lambda: [6.0, 3.0])
    composition_stacked_width_scale: float = 0.30  # Auto width for stacked case plots ~= max(base_width, scale * n_cases)
    composition_group_barplot_figsize: List[float] = Field(default_factory=lambda: [6.0, 3.0])
    figure_extension: str = '.png'  # Preferred image extension for all CellCharter outputs (e.g. '.png', '.pdf', '.svg')
    figure_format: str = 'png'
    save_high_res: bool = True

@config_section("starling")
class StarlingConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    qc_output_subdir: str = 'Starling_QC'

    # Optional local checkout fallback. Leave null when biostarling/starling is installed in the env.
    starling_repo_path: Optional[str] = None

    # Feature matrix. STARLING expects non-negative segmented cell-by-marker expression in adata.X.
    use_layer: Optional[str] = None  # Optional adata.layers key to use instead of adata.X
    marker_include: Optional[List[str]] = None  # Optional ordered marker subset (None = all vars)
    marker_exclude: List[str] = Field(default_factory=list)
    clip_small_negative_values: bool = True
    negative_value_tolerance: float = 1e-8

    # Initial clustering.
    initial_clustering_method: str = 'User'  # One of User, KM, GMM, FS, PG
    initial_label_obs: Optional[str] = None  # For User mode; fallback is general.population_obs_primary
    n_clusters: Optional[int] = None  # Required by KM/GMM/FS; optional/ignored for PG/User

    # STARLING model settings.
    seed: int = 10
    dist_option: str = 'T'  # T = Student-T, N = Normal
    singlet_prop: float = 0.6
    model_cell_size: bool = True
    cell_size_col_name: str = 'mask_area'
    cell_size_fallback_cols: List[str] = Field(default_factory=lambda: ['area'])
    model_zplane_overlap: bool = True
    model_regularizer: float = 1.0
    learning_rate: float = 1e-3
    doublet_threshold: float = 0.5

    # Lightning trainer settings.
    max_epochs: Optional[int] = 100
    early_stopping: bool = True
    early_stopping_monitor: str = 'train_loss'
    trainer_accelerator: str = 'auto'
    trainer_devices: Optional[int] = None
    trainer_precision: Optional[str] = None
    enable_checkpointing: bool = False
    enable_progress_bar: bool = True
    log_every_n_steps: Optional[int] = None
    limit_train_batches: Optional[Any] = None
    tensorboard_logging: bool = True

    # Output controls.
    output_prefix: str = 'starling'
    write_canonical_starling_keys: bool = False
    store_assignment_prob_matrix: bool = True
    store_gamma_assignment_prob_matrix: bool = False
    save_model: bool = False
    model_output_name: str = 'starling_model.pt'  # Reusable project-root checkpoint path when relative
    save_qc_tables: bool = True
    save_qc_plots: bool = True
    figure_format: str = 'png'

@config_section("pairwise_spatial")
class PairwiseSpatialConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_subdir: str = 'Pairwise_Spatial'
    reload_saved_results: bool = True  # Reuse saved raw analysis outputs when present (useful for plot-only reruns)

    # Core metadata keys
    population_obs: Optional[str] = None  # Optional override (None = use general.population_obs_primary or legacy 'population')
    groupby_obs: Optional[str] = None  # Optional override (None = use general.groupby_obs)
    groupby_obs_groups: Optional[List[str]] = None  # Optional ordered subset override (None = use general.groupby_obs_groups)
    roi_obs: Optional[str] = None  # Optional override (None = use general.roi_obs)
    x_coord_obs: Optional[str] = None  # Optional override (None = use general.x_coord_obs)
    y_coord_obs: Optional[str] = None  # Optional override (None = use general.y_coord_obs)
    master_index_obs: Optional[str] = None  # Optional override (None = use general.master_index_obs)
    source_population_obs: Optional[str] = None  # If None: uses population_obs

    # Metadata export controls
    include_all_obs_metadata: bool = True
    metadata_obs_columns: List[str] = Field(default_factory=list)

    # Squidpy neighborhood enrichment
    run_squidpy_interactions: bool = True
    squidpy_subregion_obs: Optional[str] = None  # If None: uses roi_obs
    squidpy_subregion_suffix: str = ''
    squidpy_radius_min_um: int = 0
    squidpy_radius_max_um: int = 20
    squidpy_n_permutations: int = 1000

    # Distance bootstrap analysis
    run_distance_bootstrap: bool = True
    distance_populations: Optional[List[str]] = None
    distance_roi_ids: Optional[List[str]] = None
    distance_n_bootstraps: int = 1000
    distance_n_jobs: int = -1
    distance_ddof: int = 1
    ignore_cells_without_label: bool = False  # If True, drop cells lacking target/source population labels for distance analysis

    # Pair-correlation function (PCF)
    run_pcf: bool = True
    pcf_target_distance_um: float = 20.0
    pcf_max_radius_um: float = 100.0
    pcf_radius_step_um: float = 10.0
    pcf_num_bootstrap: int = 1000
    pcf_cluster_column: str = 'cluster'
    pcf_samples: Optional[List[str]] = None

    # Optional source-target population pairs
    # Supports:
    # 1) Direct mapping: {source_pop: [target_pop, ...]}
    # 2) Nested by obs key: {population_obs: {source_pop: [target_pop, ...]}}
    # Target tokens:
    # - "ALL": all populations in population_obs
    # - "ALL_OTHERS": all populations except source_pop
    # - "MATCH_x": populations containing substring "x" (case-insensitive)
    # - "NOT_x": populations not containing substring "x" (case-insensitive)
    population_pairs: Dict[str, Any] = Field(default_factory=dict)

    # Plotting
    make_matrix_plots: bool = True
    make_pair_barplots: bool = True
    heatmap_use_clustermap: bool = True
    heatmap_row_cluster: bool = True
    heatmap_col_cluster: bool = True
    heatmap_figsize: List[float] = Field(default_factory=lambda: [5.0, 5.0])
    heatmap_percentile: float = 95.0
    pairwise_matrices_cbar_corner: str = 'off_plot_right'  # One of: 'lower_right', 'upper_left', 'off_plot_right'
    pairwise_matrices_share_vmax_vmin: bool = False  # If True, use limits from each metric's all-data matrix for all group matrix plots
    heatmap_cmap_interactions: str = 'coolwarm'
    heatmap_cmap_distance: str = 'coolwarm'
    heatmap_cmap_pcf: str = 'coolwarm'
    heatmap_cmap_counts: str = 'viridis'
    barplot_figsize: List[float] = Field(default_factory=lambda: [3.0, 3.0])
    barplot_add_points: bool = True
    # Barplot Y-axis scale controls.
    # Accepted values: 'linear', 'log', 'intelligent'
    # Flexible structure examples:
    # barplot_y_scale: {'default': 'linear'}
    # barplot_y_scale: {'distance': {'observed': 'log', 'delta': 'linear'}, 'pcf': {'g': 'log'}}
    # barplot_y_scale: {'squidpy': {'count': 'log1p', 'zscore': 'intelligent'}, 'default': 'linear'}
    # Default is explicitly populated by analysis/metric so users can tweak directly.
    barplot_y_scale: Dict[str, Any] = Field(default_factory=lambda: {
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
    })
    barplot_y_scale_intelligent_params: Dict[str, Any] = Field(default_factory=lambda: {
        'allow_log1p': True,
        'dynamic_range_thresh': 100.0,
        'skew_improve_ratio': 0.7,
        'crush_frac_thresh': 0.7,
    })
    make_source_target_barplots: bool = True  # Also plot all selected targets for each source on one figure (hue=group)
    source_target_barplot_width_scale: float = 0.35  # Width scaling constant for source->all-target plots (auto width ~= scale * n_targets)
    source_target_barplot_order_group: Optional[str] = None  # Optional group_col value used to order grouped source->all-target barplots by descending mean value within that group
    make_enrichment_plots: bool = True  # Create per-source enriched/depleted interaction plots
    enrichment_plot_figsize: List[float] = Field(default_factory=lambda: [5.5, 4.0])  # Base [width, height] for source enrichment plots
    enrichment_plot_use_barplot: bool = True  # If True, use horizontal barplots instead of boxplots for enrichment plots
    enrichment_plot_errorbar: str = 'ci95'  # Barplot error bars: one of 'ci95' or 'se'
    enrichment_plot_top_n: int = 5  # Number of enriched target populations to show per source
    enrichment_plot_bottom_n: int = 5  # Number of depleted target populations to show per source
    enrichment_plot_target_populations: Optional[List[str]] = None  # Optional target population subset for enrichment plots; None = use all available
    enrichment_plot_exclude_homotypic: bool = True  # If True, exclude source==target populations when ranking enriched/depleted targets
    enrichment_plot_share_x_axis_across_groups: bool = True  # If True, reuse one x-axis scale for all group-specific enrichment plots of the same source/metric combination
    enrichment_plot_color_mode: str = 'direction'  # One of: direction, population
    enrichment_plot_label_box_width: float = 0.03  # Width of the right-side population color boxes, in axes coordinates
    enrichment_plot_height_per_target: float = 0.25  # Additional figure height per displayed target population for enrichment plots
    figure_extension: str = '.png'
    figure_dpi: int = 300

@config_section("networkx_spatial")
class NetworkxSpatialConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_subdir: str = 'NetworkX_Spatial'
    reload_saved_results: bool = True  # Reuse saved summary tables when present (useful for plot-only reruns)

    # Core metadata keys
    population_obs: Optional[str] = None  # Optional override (None = use general.population_obs_primary or legacy 'population')
    roi_obs: Optional[str] = None  # Optional override (None = use general.roi_obs)
    case_obs: Optional[str] = None  # Optional override (None = use general.case_obs)
    groupby_obs: Optional[str] = None  # Optional override (None = use general.groupby_obs)
    x_coord_obs: Optional[str] = None  # Optional override (None = use general.x_coord_obs)
    y_coord_obs: Optional[str] = None  # Optional override (None = use general.y_coord_obs)
    spatial_key: Optional[str] = None  # Optional override (None = use general.spatial_key)
    master_index_obs: Optional[str] = None  # Optional override (None = use general.master_index_obs)

    # Metadata export controls
    include_all_obs_metadata: bool = True
    metadata_obs_columns: List[str] = Field(default_factory=list)
    ignore_cells_without_label: bool = False  # If True, drop cells with missing population labels before graph construction

    # Squidpy graph construction
    graph_coord_type: str = 'generic'
    graph_delaunay: bool = False
    graph_n_neighs: Optional[int] = 6  # If None, do not pass n_neighs and let Squidpy decide based on other graph settings
    graph_radius: Optional[List[float]] = None  # Optional [max] or [min, max] radius in coordinate units
    graph_percentile: Optional[float] = None
    graph_transform: Optional[str] = None
    graph_set_diag: bool = False

    # Metrics
    minimum_cells_per_population: int = 5  # Skip per-population clustering when fewer cells are present

    # Bootstrapping / threading
    run_bootstrap: bool = True
    bootstrap_n_permutations: int = 1000
    bootstrap_static_populations: List[str] = Field(default_factory=list)  # Keep these labels fixed while shuffling all others within each ROI
    bootstrap_ddof: int = 1
    bootstrap_seed: Optional[int] = 12345
    n_threads: int = -1  # One ROI per thread; -1 uses all available CPU threads
    save_bootstrap_samples: bool = False

    # Plotting
    make_plots: bool = True
    plot_kind: str = 'barplot'  # One of: barplot, boxplot
    plot_summary_level: str = 'case_if_available'  # One of: case_if_available, case, roi
    plot_value_columns: List[str] = Field(default_factory=lambda: ['observed', 'zscore'])
    make_all_populations_plots: bool = True  # Plot all populations on one axis with hue=groupby_obs when available
    all_populations_plot_populations: List[str] = Field(default_factory=list)  # Ordered subset for combined all-population plots; empty = use all observed populations
    all_populations_figsize: Optional[List[float]] = None  # Optional fixed [width, height] for combined all-population plots; None = auto width scaling
    make_population_group_plots: bool = True  # Plot one figure per population across groups
    make_assortativity_group_plots: bool = True
    barplot_figsize: List[float] = Field(default_factory=lambda: [4.0, 3.0])
    all_populations_width_scale: float = 0.45  # Auto width ~= max(base_width, scale * n_populations)
    barplot_add_points: bool = True
    figure_extension: str = '.png'
    figure_dpi: int = 300

@config_section("remap_obs")
class RemapObsConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    remap_csv_path: str = 'metadata/remap.csv'
    mode: str = 'apply'  # One of: apply, generate_blank

    # Mapping behavior
    source_obs: Optional[str] = None  # In apply mode: defaults to first CSV column header; in generate_blank mode this is required
    roi_obs: Optional[str] = None  # Optional override (None = use general.roi_obs) for ROI-based helper metrics in generate_blank mode
    overwrite_existing_obs_columns: bool = False
    require_complete_mapping: bool = False  # If True, raise if any non-null source values are missing from the remap table
    set_output_as_categorical: bool = True
    force_string_mapping: bool = False  # Auto-enabled when source_obs contains 'leiden'
    ignore_csv_columns_exact: List[str] = Field(default_factory=list)
    ignore_csv_columns_contains: List[str] = Field(default_factory=lambda: ['notes'])

    # Blank-template generation
    generate_columns: List[str] = Field(default_factory=list)  # Blank target columns to scaffold; empty -> [f'{source_obs}_label']
    generate_note_columns: List[str] = Field(default_factory=lambda: ['notes'])
    generate_include_counts: bool = True
    generate_count_column_name: str = 'n_cells'
    generate_include_top_markers: bool = True
    generate_top_markers_n: int = 3
    generate_top_markers_column_name: str = 'top_markers'
    generate_top_markers_use_raw: bool = False  # If True and adata.raw exists, use it by default for marker summaries
    generate_top_markers_layer: Optional[str] = None  # Optional explicit matrix source: 'raw', 'X', or a named adata.layers key
    generate_top_markers_var_column: Optional[str] = None  # Optional var annotation to use instead of var_names in the marker summary
    generate_top_markers_separator: str = '; '
    generate_include_roi_distribution_evenness: bool = True  # 0 = population concentrated in one ROI; 1 = evenly spread over all ROIs
    generate_roi_distribution_evenness_column_name: str = 'roi_distribution_evenness'
    generate_preserve_existing_values: bool = True

@config_section("subclustering")
class SubclusteringConfig(ConfigModel):
    # Input/output
    input_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_adata_path: Optional[str] = None  # Optional override (None = use general.anndata_path)
    output_subdir: str = 'subclustering'
    mode: Any = 'generate'  # One of: 'all', 'generate', 'apply', or integer/string stage selector 1, 2, 3

    # Template/remap files
    settings_filename: str = 'sublustering_settings.csv'  # Intentionally matches existing notebook naming
    marker_list_filename: str = 'marker_list.csv'
    remap_filename: str = 'subcluster_to_final_population.csv'
    master_index_mapping_filename: str = 'master_index_to_final_population.csv'

    # Subclustering defaults
    base_label_key: Optional[str] = None  # Optional override (None = use general.population_obs_primary or legacy 'population')
    default_resolution: float = 0.3
    default_marker_list: str = 'all'  # Resolved as marker column 'markers_all'
    use_rep: Optional[str] = 'X_biobatchnet'

    # Plotting and QC
    compute_umap_if_missing: bool = True
    umap_dot_size: float = 2.0
    matrixplot_vmax: float = 0.5
    save_individual_umaps: bool = True
    figure_extension: str = '.png'
    figure_dpi: int = 300

    # Final remap integration
    final_label_key: str = 'population_final'
    master_index_obs: Optional[str] = None  # Optional override (None = use general.master_index_obs)
    apply_remap_only_if_modified: bool = True

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
    batch_integration: BatchIntegrationConfig = Field(default_factory=BatchIntegrationConfig)
    rapids: RapidsProcessConfig = Field(default_factory=RapidsProcessConfig)
    biobatchnet: BioBatchNetConfig = Field(default_factory=BioBatchNetConfig)
    process: BasicProcessConfig = Field(default_factory=BasicProcessConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    cellcharter: CellCharterConfig = Field(default_factory=CellCharterConfig)
    starling: StarlingConfig = Field(default_factory=StarlingConfig)
    pairwise_spatial: PairwiseSpatialConfig = Field(default_factory=PairwiseSpatialConfig)
    networkx_spatial: NetworkxSpatialConfig = Field(default_factory=NetworkxSpatialConfig)
    remap_obs: RemapObsConfig = Field(default_factory=RemapObsConfig)
    subclustering: SubclusteringConfig = Field(default_factory=SubclusteringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


DEFAULT_CONFIG_CLASSES = {
    "general": GeneralConfig,
    "preprocess": PreprocessConfig,
    "rebuild_metadata": RebuildMetadataConfig,
    "denoising": DenoisingConfig,
    "createmasks": CreateMasksConfig,
    "segmentation": SegmentationConfig,
    "nimbus": NimbusConfig,
    "batch_integration": BatchIntegrationConfig,
    "rapids": RapidsProcessConfig,
    "biobatchnet": BioBatchNetConfig,
    "process": BasicProcessConfig,
    "visualization": VisualizationConfig,
    "cellcharter": CellCharterConfig,
    "starling": StarlingConfig,
    "pairwise_spatial": PairwiseSpatialConfig,
    "networkx_spatial": NetworkxSpatialConfig,
    "remap_obs": RemapObsConfig,
    "subclustering": SubclusteringConfig,
    "logging": LoggingConfig,
}


__all__ = [
    "BasicProcessConfig",
    "BatchIntegrationConfig",
    "BioBatchNetConfig",
    "CellCharterConfig",
    "ConfigModel",
    "CreateMasksConfig",
    "DEFAULT_CONFIG_CLASSES",
    "DenoisingConfig",
    "GeneralConfig",
    "LoggingConfig",
    "NetworkxSpatialConfig",
    "NimbusConfig",
    "PairwiseSpatialConfig",
    "PipelineConfig",
    "PreprocessConfig",
    "RapidsProcessConfig",
    "RebuildMetadataConfig",
    "RemapObsConfig",
    "SegmentationConfig",
    "StarlingConfig",
    "SubclusteringConfig",
    "VisualizationConfig",
    "config_field",
    "config_section",
]
