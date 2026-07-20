"""Train the PyTorch VICReg CellVision encoder and emit cell embeddings."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _atomic_write_h5ad(adata, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    adata.write_h5ad(temporary)
    os.replace(temporary, path)


def _validate_existing(
    paths, identity_fingerprint: str, training_fingerprint: str
) -> tuple[int, int]:
    import anndata as ad

    if not paths.model.is_file() or not paths.embeddings.is_file():
        raise FileExistsError(
            "Existing CellVision model/embedding assets are incomplete. "
            "Set cellvision.overwrite=true to rebuild them."
        )
    try:
        import torch

        try:
            checkpoint = torch.load(paths.model, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch
            checkpoint = torch.load(paths.model, map_location="cpu")
    except Exception as exc:
        raise ValueError(
            "Existing CellVision model checkpoint could not be read. "
            "Set cellvision.overwrite=true to retrain."
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or str(checkpoint.get("identity_fingerprint", "")) != identity_fingerprint
        or str(checkpoint.get("training_fingerprint", "")) != training_fingerprint
    ):
        raise ValueError(
            "Existing CellVision model does not match the current extraction or VICReg "
            "configuration fingerprint. Set cellvision.overwrite=true to retrain."
        )

    embeddings = ad.read_h5ad(paths.embeddings, backed="r")
    metadata = embeddings.uns.get("cellvision", {})
    observed_identity = str(metadata.get("identity_fingerprint", ""))
    observed_training = str(metadata.get("training_fingerprint", ""))
    shape = embeddings.shape
    embeddings.file.close()
    if (
        observed_identity != identity_fingerprint
        or observed_training != training_fingerprint
    ):
        raise ValueError(
            "Existing CellVision embeddings do not match the current extraction or VICReg "
            "configuration fingerprint. "
            "Set cellvision.overwrite=true to retrain."
        )
    return int(shape[0]), int(shape[1])


def main() -> None:
    import anndata as ad
    import numpy as np
    import pandas as pd

    from SpatialBiologyToolkit.cellvision import (
        configuration_fingerprint,
        image_channel_metadata,
        read_h5sc_metadata,
        read_json,
    )
    from SpatialBiologyToolkit.cellvision_vicreg import (
        H5SCImageDataset,
        estimate_channel_scales,
        extract_embeddings,
        plot_training_history,
        save_checkpoint,
        train_vicreg,
    )
    from SpatialBiologyToolkit.reporting import category_output_path
    from SpatialBiologyToolkit.scripts._cellvision_common import load_runtime, reporter

    config, paths = load_runtime("embed")
    cellvision = config.cellvision
    if not paths.h5sc.is_file() or not paths.extraction_metadata.is_file():
        raise FileNotFoundError(
            "CellVision extraction assets are missing. Run cellvision_extract before embedding."
        )
    extraction = read_json(paths.extraction_metadata)
    fingerprint = str(extraction["identity_fingerprint"])
    h5sc_obs, h5sc_var, image_shape = read_h5sc_metadata(paths.h5sc)
    channel_indices, channel_names = image_channel_metadata(h5sc_var, image_shape)
    if channel_names != [str(value) for value in extraction["markers"]]:
        raise ValueError(
            "H5SC image-channel metadata does not match extraction metadata: "
            f"{channel_names} != {extraction['markers']}"
        )

    architecture = {
        "input_channels": len(channel_indices),
        "width": cellvision.encoder_width,
        "embedding_dim": cellvision.embedding_dim,
        "projector_dim": cellvision.projector_dim,
    }
    training_config = {
        "epochs": cellvision.epochs,
        "batch_size": cellvision.batch_size,
        "learning_rate": cellvision.learning_rate,
        "weight_decay": cellvision.weight_decay,
        "warmup_epochs": cellvision.warmup_epochs,
        "num_workers": cellvision.num_workers,
        "seed": cellvision.seed,
        "amp": cellvision.amp,
        "normalization_quantile": cellvision.normalization_quantile,
        "normalization_sample_cells": cellvision.normalization_sample_cells,
        "vicreg_invariance_weight": cellvision.vicreg_invariance_weight,
        "vicreg_variance_weight": cellvision.vicreg_variance_weight,
        "vicreg_covariance_weight": cellvision.vicreg_covariance_weight,
        "augmentation_translation_px": cellvision.augmentation_translation_px,
        "augmentation_intensity_jitter": cellvision.augmentation_intensity_jitter,
        "augmentation_noise_std": cellvision.augmentation_noise_std,
    }
    training_fingerprint = configuration_fingerprint(
        {
            "schema_version": 1,
            "identity_fingerprint": fingerprint,
            "channel_indices": channel_indices,
            "channel_names": channel_names,
            "architecture": architecture,
            "training_config": training_config,
        }
    )

    stage_reporter = reporter()
    if stage_reporter is not None:
        stage_reporter.add_input("cellvision_h5sc", paths.h5sc, "Identity-annotated H5SC images used for VICReg.")
        stage_reporter.add_input("cellvision_identity", paths.identity, "Source cell identity table.")

    if (paths.model.exists() or paths.embeddings.exists()) and not cellvision.overwrite:
        n_cells, n_embeddings = _validate_existing(
            paths, fingerprint, training_fingerprint
        )
        logging.info("Reusing validated CellVision model and embeddings.")
        if stage_reporter is not None:
            stage_reporter.add_asset("cellvision_model", paths.model, "Reusable trained VICReg checkpoint.")
            stage_reporter.add_asset("cellvision_embeddings", paths.embeddings, "Cell-level VICReg embeddings.")
            stage_reporter.add_metric("embedded_cells", n_cells)
            stage_reporter.add_metric("embedding_dimensions", n_embeddings)
            stage_reporter.add_note("Reused existing model and embeddings with a matching identity fingerprint.")
        return

    scales = estimate_channel_scales(
        paths.h5sc,
        channel_indices=channel_indices,
        quantile=cellvision.normalization_quantile,
        max_cells=cellvision.normalization_sample_cells,
        seed=cellvision.seed,
    )
    dataset = H5SCImageDataset(
        paths.h5sc,
        channel_indices=channel_indices,
        channel_scales=scales.tolist(),
    )
    try:
        model, history, device = train_vicreg(
            dataset,
            width=cellvision.encoder_width,
            embedding_dim=cellvision.embedding_dim,
            projector_dim=cellvision.projector_dim,
            epochs=cellvision.epochs,
            batch_size=cellvision.batch_size,
            learning_rate=cellvision.learning_rate,
            weight_decay=cellvision.weight_decay,
            warmup_epochs=cellvision.warmup_epochs,
            num_workers=cellvision.num_workers,
            seed=cellvision.seed,
            amp=cellvision.amp,
            invariance_weight=cellvision.vicreg_invariance_weight,
            variance_weight=cellvision.vicreg_variance_weight,
            covariance_weight=cellvision.vicreg_covariance_weight,
            translation_px=cellvision.augmentation_translation_px,
            intensity_jitter=cellvision.augmentation_intensity_jitter,
            noise_std=cellvision.augmentation_noise_std,
        )
        embeddings, rows = extract_embeddings(
            model,
            dataset,
            batch_size=cellvision.batch_size,
            num_workers=cellvision.num_workers,
            device=device,
        )
    finally:
        dataset.close()

    save_checkpoint(
        paths.model,
        model,
        architecture=architecture,
        channel_indices=channel_indices,
        channel_names=channel_names,
        channel_scales=scales.tolist(),
        identity_fingerprint=fingerprint,
        training_fingerprint=training_fingerprint,
        training_config=training_config,
    )

    if not np.array_equal(rows, np.arange(len(h5sc_obs))):
        raise RuntimeError("VICReg embedding rows do not align with H5SC observation order.")
    obs = h5sc_obs.copy()
    obs.index = obs.index.astype(str)
    if not obs.index.is_unique:
        raise ValueError("H5SC source observation IDs must be unique before embedding export.")
    var = pd.DataFrame(index=[f"embedding_{index:03d}" for index in range(embeddings.shape[1])])
    embedding_adata = ad.AnnData(X=embeddings, obs=obs, var=var)
    embedding_adata.obsm["X_cellvision"] = embeddings.copy()
    embedding_adata.uns["cellvision"] = {
        "format_version": 1,
        "identity_fingerprint": fingerprint,
        "training_fingerprint": training_fingerprint,
        "source_h5sc": str(paths.h5sc),
        "model_path": str(paths.model),
        "channel_names": channel_names,
        "channel_indices": channel_indices,
        "channel_scales": scales.astype(float).tolist(),
        "architecture": architecture,
        "training_config": training_config,
    }
    _atomic_write_h5ad(embedding_adata, paths.embeddings)

    history_path = category_output_path("tables", "vicreg_training_history.csv", stage="cellvision")
    history.to_csv(history_path, index=False)
    figure_path = category_output_path("figures", "vicreg_training_loss.png", stage="cellvision")
    plot_training_history(history, figure_path, dpi=cellvision.figure_dpi)
    logging.info("Saved %s CellVision embeddings with %s dimensions.", *embeddings.shape)
    if stage_reporter is not None:
        stage_reporter.add_asset("cellvision_model", paths.model, "Reusable trained VICReg encoder/projector checkpoint.")
        stage_reporter.add_asset("cellvision_embeddings", paths.embeddings, "Identity-aligned cell-level VICReg embeddings.")
        stage_reporter.add_file("table", history_path, "Epoch-level VICReg objective components.")
        stage_reporter.add_file("figure", figure_path, "VICReg training objective diagnostic.")
        stage_reporter.add_metric("embedded_cells", embeddings.shape[0])
        stage_reporter.add_metric("embedding_dimensions", embeddings.shape[1])
        stage_reporter.add_metric("training_epochs", len(history))
        stage_reporter.add_metric("final_vicreg_loss", float(history.iloc[-1]["loss"]))


if __name__ == "__main__":
    main()
