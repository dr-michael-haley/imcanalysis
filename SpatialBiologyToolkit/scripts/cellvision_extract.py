"""Create identity-tracked single-cell H5SC images with scPortrait."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path


def _safe_replace_asset_folder(asset_root: Path, project_root: Path) -> None:
    resolved = asset_root.resolve(strict=False)
    project = project_root.resolve(strict=False)
    if resolved == project or project not in resolved.parents:
        raise ValueError(
            f"Refusing to replace CellVision asset folder outside the project: {resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def main() -> None:
    import anndata as ad

    from SpatialBiologyToolkit.cellvision import (
        annotate_h5sc_identity,
        assemble_scportrait_inputs,
        compute_normalization_dict,
        discover_roi_inputs,
        identity_fingerprint,
        input_file_manifest,
        load_normalization_dict,
        run_scportrait_extraction,
        select_source_cells,
        validate_existing_extraction,
        write_json,
        write_normalization_dict,
        write_scportrait_config,
    )
    from SpatialBiologyToolkit.reporting import project_asset_path
    from SpatialBiologyToolkit.scripts._cellvision_common import (
        input_paths,
        load_runtime,
        reporter,
    )

    config, paths = load_runtime("extract")
    cellvision = config.cellvision
    source_path, images_folder, masks_folder = input_paths(config)
    for path, label in (
        (source_path, "source AnnData"),
        (images_folder, "denoised image folder"),
        (masks_folder, "mask folder"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"CellVision {label} does not exist: {path}")

    source = ad.read_h5ad(source_path)
    roi_obs = cellvision.roi_obs or config.general.roi_obs
    identity = select_source_cells(
        source,
        roi_obs=roi_obs,
        object_id_obs=cellvision.object_id_obs,
        population_obs=cellvision.population_obs,
        populations=cellvision.populations,
    )
    contexts, channel_names = discover_roi_inputs(
        images_folder,
        masks_folder,
        identity,
        roi_obs=roi_obs,
        markers=cellvision.markers,
    )
    supplied_normalization_path = (
        project_asset_path(cellvision.normalization_dict_path)
        if cellvision.normalization_dict_path
        else None
    )
    if supplied_normalization_path is not None:
        normalization_values = load_normalization_dict(
            supplied_normalization_path,
            channel_names=channel_names,
        )
    elif paths.root.exists() and not cellvision.overwrite:
        if not paths.normalization_dict.is_file():
            raise FileExistsError(
                "Existing CellVision extraction predates extraction-stage normalization. "
                "Set cellvision.overwrite=true to rebuild its H5SC images."
            )
        normalization_values = load_normalization_dict(
            paths.normalization_dict,
            channel_names=channel_names,
        )
    else:
        normalization_values = compute_normalization_dict(
            contexts,
            quantile=cellvision.normalization_quantile,
            minimum_value=cellvision.normalization_min_value,
            mask_expand_px=cellvision.mask_expand_px,
        )
    source_files = [
        source_path,
        *(context.mask_path for context in contexts),
        *(channel_path for context in contexts for channel_path in context.channel_files),
    ]
    fingerprint = identity_fingerprint(
        identity,
        roi_obs=roi_obs,
        object_id_obs=cellvision.object_id_obs,
        markers=channel_names,
        image_size=cellvision.image_size,
        extraction_parameters={
            "mask_expand_px": int(cellvision.mask_expand_px),
            "mask_gaussian_blur": bool(cellvision.mask_gaussian_blur),
            "normalization_values": normalization_values,
            "normalization_quantile": float(cellvision.normalization_quantile),
            "normalization_min_value": float(cellvision.normalization_min_value),
            "normalization_clip": [float(value) for value in cellvision.normalization_clip],
        },
        input_manifest=input_file_manifest(source_files),
    )

    stage_reporter = reporter()
    if stage_reporter is not None:
        stage_reporter.add_input("anndata", source_path, "Source AnnData providing stable cell identities and labels.")
        stage_reporter.add_input("denoised_images", images_folder, "ROI/channel TIFFs used for cell portraits.")
        stage_reporter.add_input("masks", masks_folder, "Labelled ROI masks used for exact cell clipping.")
        if supplied_normalization_path is not None:
            stage_reporter.add_input(
                "normalization_dict",
                supplied_normalization_path,
                "Nimbus-format per-channel normalization values supplied by the user.",
            )

    if paths.root.exists() and not cellvision.overwrite:
        metadata = validate_existing_extraction(paths, expected_fingerprint=fingerprint)
        logging.info("Reusing validated CellVision H5SC extraction at %s", paths.h5sc)
        if stage_reporter is not None:
            stage_reporter.add_asset("cellvision_h5sc", paths.h5sc, "Validated reusable scPortrait single-cell images.")
            stage_reporter.add_asset("cellvision_identity", paths.identity, "Source-to-scPortrait cell identity table.")
            stage_reporter.add_asset(
                "cellvision_normalization_dict",
                paths.normalization_dict,
                "Nimbus-format scales used to create the stored [0, 1] H5SC marker images.",
            )
            stage_reporter.add_metric("requested_cells", metadata["n_requested_cells"])
            stage_reporter.add_metric("extracted_cells", metadata["n_extracted_cells"])
            stage_reporter.add_metric("image_channels", len(metadata["markers"]))
            stage_reporter.add_note("Reused an existing extraction with a matching identity fingerprint.")
        return

    if cellvision.overwrite:
        project_root = Path(os.environ.get("SBT_PROJECT_ROOT", Path.cwd()))
        _safe_replace_asset_folder(paths.root, project_root)
    elif paths.root.exists():
        raise FileExistsError(
            f"CellVision asset folder already exists: {paths.root}. "
            "Set cellvision.overwrite=true to replace incomplete assets."
        )
    paths.root.mkdir(parents=True, exist_ok=True)
    write_normalization_dict(paths.normalization_dict, normalization_values)

    channel_paths, combined_mask = assemble_scportrait_inputs(
        contexts,
        identity,
        roi_obs=roi_obs,
        object_id_obs=cellvision.object_id_obs,
        assembled_folder=paths.root / "assembled_channels",
        image_size=cellvision.image_size,
        mask_expand_px=cellvision.mask_expand_px,
        normalization_values=normalization_values,
        normalization_clip=cellvision.normalization_clip,
    )
    write_scportrait_config(
        paths.scportrait_config,
        image_size=cellvision.image_size,
        threads=cellvision.extraction_threads,
    )
    output = run_scportrait_extraction(
        project_folder=paths.root,
        config_path=paths.scportrait_config,
        channel_paths=channel_paths,
        channel_names=channel_names,
        combined_mask=combined_mask,
        gaussian_blur_mask=cellvision.mask_gaussian_blur,
        overwrite=True,
    )
    if output != paths.h5sc:
        raise RuntimeError(
            f"scPortrait wrote H5SC to {output}, expected canonical CellVision path {paths.h5sc}."
        )
    identity = annotate_h5sc_identity(
        paths.h5sc,
        identity,
        roi_obs=roi_obs,
        object_id_obs=cellvision.object_id_obs,
        population_obs=cellvision.population_obs,
    )
    identity.to_csv(paths.identity, index=False)
    extracted = int(identity["extraction_status"].eq("extracted").sum())
    if extracted < 2:
        raise ValueError(
            f"scPortrait extracted only {extracted} CellVision cells; VICReg requires at least two."
        )
    metadata = {
        "schema_version": 2,
        "identity_fingerprint": fingerprint,
        "source_anndata": str(source_path),
        "images_folder": str(images_folder),
        "masks_folder": str(masks_folder),
        "roi_obs": roi_obs,
        "object_id_obs": cellvision.object_id_obs,
        "population_obs": cellvision.population_obs,
        "populations": cellvision.populations,
        "markers": channel_names,
        "image_size": int(cellvision.image_size),
        "mask_gaussian_blur": bool(cellvision.mask_gaussian_blur),
        "normalization_dict": str(paths.normalization_dict),
        "normalization_values": normalization_values,
        "normalization_source": (
            str(supplied_normalization_path)
            if supplied_normalization_path is not None
            else "computed_from_all_in_mask_roi_pixels"
        ),
        "normalization_quantile": float(cellvision.normalization_quantile),
        "normalization_min_value": float(cellvision.normalization_min_value),
        "normalization_clip": [float(value) for value in cellvision.normalization_clip],
        "n_requested_cells": int(len(identity)),
        "n_extracted_cells": extracted,
        "n_not_extracted_cells": int(len(identity) - extracted),
        "n_rois": int(len(contexts)),
    }
    write_json(paths.extraction_metadata, metadata)
    logging.info(
        "CellVision extraction complete: %d/%d requested cells, %d markers, %d ROIs.",
        extracted,
        len(identity),
        len(channel_names),
        len(contexts),
    )
    if stage_reporter is not None:
        stage_reporter.add_asset("cellvision_assets", paths.root, "Reusable CellVision models and cell-level data.")
        stage_reporter.add_asset("cellvision_h5sc", paths.h5sc, "Identity-annotated scPortrait single-cell images.")
        stage_reporter.add_asset("cellvision_identity", paths.identity, "Source-to-scPortrait cell identity table including extraction status.")
        stage_reporter.add_asset("cellvision_extraction_metadata", paths.extraction_metadata, "Extraction selection and identity fingerprint.")
        stage_reporter.add_asset(
            "cellvision_normalization_dict",
            paths.normalization_dict,
            "Nimbus-format scales used to create the stored [0, 1] H5SC marker images.",
        )
        stage_reporter.add_metric("requested_cells", len(identity))
        stage_reporter.add_metric("extracted_cells", extracted)
        stage_reporter.add_metric("not_extracted_cells", len(identity) - extracted)
        stage_reporter.add_metric("image_channels", len(channel_names))
        stage_reporter.add_metric("rois", len(contexts))
        if extracted < len(identity):
            stage_reporter.add_warning(
                f"scPortrait did not extract {len(identity) - extracted} requested cells; see {paths.identity.name}."
            )


if __name__ == "__main__":
    main()
