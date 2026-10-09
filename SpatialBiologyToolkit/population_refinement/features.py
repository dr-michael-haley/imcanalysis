"""Notebook setup for the existing resumable, full-mask feature worker."""

from __future__ import annotations

from pathlib import Path

from SpatialBiologyToolkit.napari_sbt.cohort import resolve_cohort, save_cohort_snapshot
from SpatialBiologyToolkit.napari_sbt.models import (
    ClassificationClass,
    ExperimentManifest,
    SyntheticFeatureRecipe,
)
from SpatialBiologyToolkit.napari_sbt.storage import load_experiment, save_experiment
from SpatialBiologyToolkit.qc_classifier.io import build_image_channel_aliases

from .models import SplitSpec


def image_recipe(
    channels: list[str], *, normalization_dict_path=None
) -> SyntheticFeatureRecipe:
    """Compact starting recipe; contextual and cohort-relative features are opt-in."""
    if not channels:
        raise ValueError("Select at least one channel explicitly.")
    return SyntheticFeatureRecipe(
        channels=channels,
        normalization_dict_path=normalization_dict_path,
        distribution_feature_names=[
            "mean",
            "median",
            "std",
            "q25",
            "q75",
            "q90",
            "iqr",
        ],
        region_feature_names=[
            "core_mean",
            "border_mean",
            "core_to_border_ratio",
            "weighted_centroid_offset_fraction_radius",
            "foreground_bg_contrast",
            "foreground_bg_contrast_z",
            "foreground_to_bg_ratio",
        ],
        shape_feature_names=[
            "mask_area",
            "mask_eccentricity",
            "mask_solidity",
            "mask_axis_ratio",
        ],
        context_features=False,
        roi_rank_features=False,
    )


def create_feature_experiment(
    adata,
    spec: SplitSpec,
    *,
    directory: str | Path,
    images: list[str | Path],
    masks: str | Path,
    recipe: SyntheticFeatureRecipe,
    name="Population refinement",
) -> Path:
    """Freeze a live AnnData cohort without saving or requiring a source H5AD.

    Existing compatible workspaces are reusable. Changed inputs require a new
    workspace, so valid fragments cannot silently acquire different semantics.
    Full target cohorts per ROI ensure ROI ranks have identical train/predict scope.
    """
    if not recipe.channels:
        raise ValueError("Refinement requires an explicit image channel selection.")
    directory = Path(directory).expanduser().resolve()
    image_paths = [str(Path(path).expanduser().resolve()) for path in images]
    mask_path = str(Path(masks).expanduser().resolve())
    if not image_paths or any(
        not Path(path).is_dir() for path in [*image_paths, mask_path]
    ):
        raise ValueError("Image and mask directories must exist.")
    preview = resolve_cohort(
        adata,
        roi_obs=spec.roi_key,
        object_id_obs=spec.object_key,
        mode="obs_values",
        obs_column=spec.population_key,
        obs_values=spec.populations,
    )
    colours = [
        "#4477aa",
        "#ee6677",
        "#228833",
        "#ccbb44",
        "#66ccee",
        "#aa3377",
        "#bbbbbb",
        "#000000",
    ]
    manifest = ExperimentManifest(
        name=name,
        masks_folder=mask_path,
        images_folders=image_paths,
        identity_source="frozen_cohort",
        roi_obs=spec.roi_key,
        object_id_obs=spec.object_key,
        channel_aliases=build_image_channel_aliases(adata.var_names, adata.var),
        cell_scope=preview.scope(
            mode="obs_values",
            obs_column=spec.population_key,
            obs_values=spec.populations,
        ),
        classes=[
            ClassificationClass(
                class_id=item.class_id,
                name=item.name,
                color=colours[i],
                shortcut=str(i + 1),
            )
            for i, item in enumerate(spec.classes)
        ],
        synthetic_features=recipe,
    )
    if (directory / "experiment.yaml").exists():
        existing, paths = load_experiment(directory)
        for key in (
            "cell_scope",
            "synthetic_features",
            "images_folders",
            "masks_folder",
            "channel_aliases",
            "classes",
        ):
            if getattr(existing, key) != getattr(manifest, key):
                raise ValueError(
                    f"Existing workspace has different {key}; select a new directory."
                )
        from SpatialBiologyToolkit.napari_sbt.cohort import validate_frozen_cohort
        from SpatialBiologyToolkit.napari_sbt.storage import read_dataframe

        validate_frozen_cohort(read_dataframe(paths.cohort), existing.cell_scope)
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("New experiment directory must be empty.")
    paths = save_experiment(manifest, directory)
    save_cohort_snapshot(preview, paths.cohort)
    return directory


def build_feature_table(experiment: str | Path, *, workers=1, progress=None):
    """Run/resume the same feature worker used by NapariSBT and sbt run cellfeat."""
    from SpatialBiologyToolkit.napari_sbt.worker import run_feature_build
    from SpatialBiologyToolkit.napari_sbt.storage import read_dataframe

    result = run_feature_build(experiment, workers=workers, progress=progress)
    return read_dataframe(result.feature_table)


def image_feature_columns(
    features, *, channels: list[str] | None = None, means_only=False
) -> list[str]:
    """Explicit image predictors, excluding identifiers and absolute coordinates."""
    shape = {"mask_area", "mask_eccentricity", "mask_solidity", "mask_axis_ratio"}
    selected = []
    for column in features:
        name = str(column)
        pieces = name.split("::")
        if "channel" in pieces:
            position = pieces.index("channel")
            if channels is not None and pieces[position + 1] not in channels:
                continue
            if means_only and pieces[position + 2 :] != ["mean"]:
                continue
            if any(
                part
                in {
                    "weighted_x",
                    "weighted_y",
                    "cohort_roi_zscore",
                    "cohort_roi_percentile",
                }
                for part in pieces
            ):
                continue
            selected.append(name)
        elif not means_only and pieces[-1] in shape:
            selected.append(name)
    if not selected:
        raise ValueError("No image features match the requested selection.")
    return selected
