#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Perturbation sensitivity analysis for HyPERSTAC IMC patch representations.

The script compares each original patch embedding with embeddings from
channel-perturbed versions of the same patch. It writes an AnnData object with
the same patch index as the representation AnnData:

    .X                      cosine distance, one column per perturbation
    .layers["cosine_similarity"] cosine similarity
    .layers["perturbed_embedding_norm"] perturbed embedding L2 norm

Default perturbations:
    - set each channel to zero
    - shuffle each channel's pixels 10 times per patch
    - set all channels to zero
    - shuffle all channels independently 10 times per patch
    - shuffle all channels with a shared spatial permutation 10 times per patch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from tqdm.auto import tqdm


from .model import build_encoder, patch_obs_index


@dataclass
class PerturbationCondition:
    condition_id: str
    perturbation_type: str
    channel: str
    channel_index: int
    replicate: int
    shuffle_scope: str
    preserves_channel_histogram: bool
    preserves_cross_channel_colocalization: bool
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run channel perturbation sensitivity analysis on HyPERSTAC patch embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hyperstac-output-dir",
        type=Path,
        required=True,
        help="Existing HyPERSTAC output directory containing patches, model weights, and AnnData.",
    )
    parser.add_argument(
        "--adata-path",
        type=Path,
        default=None,
        help="Representation AnnData. Defaults to hyperstac-output-dir/imc_hyperstac_representations.h5ad.",
    )
    parser.add_argument(
        "--patch-metadata",
        type=Path,
        default=None,
        help="Patch metadata CSV. Defaults to hyperstac-output-dir/patch_metadata.csv.",
    )
    parser.add_argument(
        "--encoder-weights",
        type=Path,
        default=None,
        help="Encoder weights. Defaults to hyperstac-output-dir/model/encoder.weights.h5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to hyperstac-output-dir/permutation_sensitivity.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel names to perturb. Defaults to all channels in the representation AnnData.",
    )
    parser.add_argument(
        "--encoder",
        choices=["resnet50", "small-cnn"],
        default=None,
        help="Encoder architecture. Defaults to adata.uns['encoder'] or resnet50.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Patch size in pixels. Defaults to adata.uns['patch_size_px'].",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding prediction.",
    )
    parser.add_argument(
        "--n-shuffle-repeats",
        type=int,
        default=10,
        help="Number of independent pixel-shuffle repeats per shuffle perturbation.",
    )
    parser.add_argument(
        "--shuffle-pixels",
        choices=["all", "nonzero"],
        default="all",
        help="Shuffle all pixels, or only non-zero pixels within each affected channel/vector.",
    )
    parser.add_argument(
        "--include-all-channel-perturbations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run all-channel zero, independent shuffle, and shared-spatial shuffle perturbations.",
    )
    parser.add_argument(
        "--recompute-original",
        action="store_true",
        help="Recompute original embeddings from patches instead of using adata.X.",
    )
    parser.add_argument(
        "--write-wide-csv",
        action="store_true",
        help="Also write wide CSV.gz matrices for distance, similarity, and perturbed embedding norm.",
    )
    parser.add_argument(
        "--adata-name",
        type=str,
        default="imc_permutation_sensitivity.h5ad",
        help="Output AnnData filename.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    return parser.parse_args()


def configure_runtime(seed: int) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.hyperstac_output_dir = args.hyperstac_output_dir.resolve()
    args.adata_path = (args.adata_path or args.hyperstac_output_dir / "imc_hyperstac_representations.h5ad").resolve()
    args.patch_metadata = (args.patch_metadata or args.hyperstac_output_dir / "patch_metadata.csv").resolve()
    args.encoder_weights = (args.encoder_weights or args.hyperstac_output_dir / "model" / "encoder.weights.h5").resolve()
    args.output_dir = (args.output_dir or args.hyperstac_output_dir / "permutation_sensitivity").resolve()
    return args


def safe_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    if not name:
        name = "channel"
    if name[0].isdigit():
        name = f"ch_{name}"
    return name


def matrix_from_adata_x(adata: ad.AnnData) -> np.ndarray:
    if sparse.issparse(adata.X):
        return adata.X.toarray().astype(np.float32, copy=False)
    return np.asarray(adata.X, dtype=np.float32)


def load_channel_names(args: argparse.Namespace, adata: ad.AnnData) -> list[str]:
    if "channel_names" in adata.uns:
        return [str(channel) for channel in adata.uns["channel_names"]]

    channels_path = args.hyperstac_output_dir / "channels.json"
    if channels_path.exists():
        with open(channels_path, "r", encoding="utf-8") as handle:
            return [str(channel) for channel in json.load(handle)]

    raise ValueError("Could not find channel names in adata.uns['channel_names'] or channels.json.")


def parse_channels_to_perturb(channels_arg: str | None, channel_names: list[str]) -> list[str]:
    if channels_arg is None:
        return channel_names
    channels = [channel.strip() for channel in channels_arg.split(",") if channel.strip()]
    missing = sorted(set(channels) - set(channel_names))
    if missing:
        raise ValueError(f"Requested perturbation channels were not found: {missing}")
    return channels


def resolve_patch_path(path_value: str, hyperstac_output_dir: Path) -> str:
    path = Path(path_value)
    if path.exists():
        return str(path)
    if not path.is_absolute():
        candidate = hyperstac_output_dir / path
        if candidate.exists():
            return str(candidate)
    return str(path)


def load_aligned_metadata(args: argparse.Namespace, adata: ad.AnnData) -> pd.DataFrame:
    if "patch_path" in adata.obs:
        metadata = adata.obs.copy()
        metadata["patch_id"] = adata.obs_names
        metadata = metadata.reset_index(drop=True)
        metadata["patch_path"] = [
            resolve_patch_path(path_value, args.hyperstac_output_dir)
            for path_value in metadata["patch_path"].astype(str)
        ]
        return metadata

    if not args.patch_metadata.exists():
        raise FileNotFoundError(f"Patch metadata not found: {args.patch_metadata}")
    metadata = pd.read_csv(args.patch_metadata)
    metadata["patch_id"] = patch_obs_index(metadata)
    metadata = metadata.set_index("patch_id").loc[adata.obs_names].reset_index()
    metadata["patch_path"] = [
        resolve_patch_path(path_value, args.hyperstac_output_dir)
        for path_value in metadata["patch_path"].astype(str)
    ]
    return metadata


def validate_patch_paths(patch_paths: list[str]) -> None:
    missing = [path for path in patch_paths if not Path(path).exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Patch files are missing. First missing paths:\n{preview}")


def infer_encoder_and_patch_size(args: argparse.Namespace, adata: ad.AnnData) -> tuple[str, int]:
    encoder = args.encoder or str(adata.uns.get("encoder", "resnet50"))
    patch_size_value = args.patch_size if args.patch_size is not None else adata.uns.get("patch_size_px")
    if patch_size_value is None:
        raise ValueError("Patch size was not supplied and adata.uns['patch_size_px'] is missing.")
    patch_size = int(patch_size_value)
    return encoder, patch_size


def build_representation_model(
    encoder_name: str,
    patch_size: int,
    num_channels: int,
    weights_path: Path,
) -> tf.keras.Model:
    if not weights_path.exists():
        raise FileNotFoundError(f"Encoder weights not found: {weights_path}")
    encoder = build_encoder(encoder_name, (patch_size, patch_size, num_channels))
    encoder.load_weights(weights_path)
    return tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.GlobalAveragePooling2D(),
        ],
        name="permutation_representation_model",
    )


def condition_var_dataframe(conditions: list[PerturbationCondition]) -> pd.DataFrame:
    records = [asdict(condition) for condition in conditions]
    var = pd.DataFrame.from_records(records).set_index("condition_id")
    var.index.name = "perturbation_id"
    return var


def build_conditions(
    channel_names: list[str],
    channels_to_perturb: list[str],
    n_shuffle_repeats: int,
    include_all_channels: bool,
) -> list[PerturbationCondition]:
    if n_shuffle_repeats < 0:
        raise ValueError("--n-shuffle-repeats must be >= 0.")

    channel_index = {channel: idx for idx, channel in enumerate(channel_names)}
    conditions: list[PerturbationCondition] = []

    for channel in channels_to_perturb:
        idx = channel_index[channel]
        safe_channel = safe_name(channel)
        conditions.append(
            PerturbationCondition(
                condition_id=f"zero__{safe_channel}",
                perturbation_type="zero_channel",
                channel=channel,
                channel_index=idx,
                replicate=0,
                shuffle_scope="none",
                preserves_channel_histogram=False,
                preserves_cross_channel_colocalization=False,
                description=f"Set channel '{channel}' to zero.",
            )
        )
        for rep in range(1, n_shuffle_repeats + 1):
            conditions.append(
                PerturbationCondition(
                    condition_id=f"shuffle__{safe_channel}__rep{rep:02d}",
                    perturbation_type="shuffle_channel",
                    channel=channel,
                    channel_index=idx,
                    replicate=rep,
                    shuffle_scope="single_channel",
                    preserves_channel_histogram=True,
                    preserves_cross_channel_colocalization=False,
                    description=f"Shuffle pixel values within channel '{channel}'.",
                )
            )

    if include_all_channels:
        conditions.append(
            PerturbationCondition(
                condition_id="zero__all_channels",
                perturbation_type="zero_all_channels",
                channel="all_channels",
                channel_index=-1,
                replicate=0,
                shuffle_scope="none",
                preserves_channel_histogram=False,
                preserves_cross_channel_colocalization=False,
                description="Set all channels to zero.",
            )
        )
        for rep in range(1, n_shuffle_repeats + 1):
            conditions.append(
                PerturbationCondition(
                    condition_id=f"shuffle_all_independent__rep{rep:02d}",
                    perturbation_type="shuffle_all_channels_independent",
                    channel="all_channels",
                    channel_index=-1,
                    replicate=rep,
                    shuffle_scope="all_channels_independent",
                    preserves_channel_histogram=True,
                    preserves_cross_channel_colocalization=False,
                    description="Shuffle each channel independently within each patch.",
                )
            )
            conditions.append(
                PerturbationCondition(
                    condition_id=f"shuffle_all_shared__rep{rep:02d}",
                    perturbation_type="shuffle_all_channels_shared",
                    channel="all_channels",
                    channel_index=-1,
                    replicate=rep,
                    shuffle_scope="all_channels_shared",
                    preserves_channel_histogram=True,
                    preserves_cross_channel_colocalization=True,
                    description="Apply the same spatial pixel permutation to all channels.",
                )
            )
    return conditions


def iter_batches(values: list[str], batch_size: int):
    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        yield start, end, values[start:end]


def load_patch_batch(
    patch_paths: list[str],
    patch_size: int,
    num_channels: int,
) -> np.ndarray:
    patches = np.empty((len(patch_paths), patch_size, patch_size, num_channels), dtype=np.float32)
    for idx, patch_path in enumerate(patch_paths):
        patch = np.load(patch_path).astype(np.float32, copy=False)
        expected_shape = (patch_size, patch_size, num_channels)
        if patch.shape != expected_shape:
            raise ValueError(f"Expected patch shape {expected_shape}, got {patch.shape}: {patch_path}")
        patches[idx] = patch
    return patches


def shuffle_1d(values: np.ndarray, rng: np.random.Generator, shuffle_pixels: str) -> None:
    flat = values.reshape(-1)
    if shuffle_pixels == "all":
        flat[:] = flat[rng.permutation(flat.size)]
        return

    indices = np.flatnonzero(flat != 0)
    if indices.size > 1:
        flat[indices] = flat[indices[rng.permutation(indices.size)]]


def perturb_batch(
    batch: np.ndarray,
    condition: PerturbationCondition,
    rng: np.random.Generator,
    shuffle_pixels: str,
) -> np.ndarray:
    perturbed = batch.copy()

    if condition.perturbation_type == "zero_channel":
        perturbed[:, :, :, condition.channel_index] = 0.0
        return perturbed

    if condition.perturbation_type == "zero_all_channels":
        perturbed.fill(0.0)
        return perturbed

    if condition.perturbation_type == "shuffle_channel":
        channel_idx = condition.channel_index
        for patch in perturbed:
            shuffle_1d(patch[:, :, channel_idx], rng, shuffle_pixels)
        return perturbed

    if condition.perturbation_type == "shuffle_all_channels_independent":
        for patch in perturbed:
            for channel_idx in range(patch.shape[-1]):
                shuffle_1d(patch[:, :, channel_idx], rng, shuffle_pixels)
        return perturbed

    if condition.perturbation_type == "shuffle_all_channels_shared":
        for patch in perturbed:
            flat = patch.reshape(-1, patch.shape[-1])
            if shuffle_pixels == "all":
                flat[:] = flat[rng.permutation(flat.shape[0]), :]
            else:
                indices = np.flatnonzero(np.any(flat != 0, axis=1))
                if indices.size > 1:
                    flat[indices, :] = flat[indices[rng.permutation(indices.size)], :]
        return perturbed

    raise ValueError(f"Unknown perturbation type: {condition.perturbation_type}")


def cosine_metrics(original: np.ndarray, perturbed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if original.shape != perturbed.shape:
        raise ValueError(
            f"Original and perturbed embeddings have different shapes: {original.shape} vs {perturbed.shape}"
        )
    original = original.astype(np.float32, copy=False)
    perturbed = perturbed.astype(np.float32, copy=False)
    original_norm = np.linalg.norm(original, axis=1)
    perturbed_norm = np.linalg.norm(perturbed, axis=1)
    denominator = original_norm * perturbed_norm
    dot_product = np.sum(original * perturbed, axis=1)
    similarity = np.divide(
        dot_product,
        denominator,
        out=np.full_like(dot_product, np.nan, dtype=np.float32),
        where=denominator > 0,
    )
    similarity = np.clip(similarity, -1.0, 1.0)
    distance = 1.0 - similarity
    return distance.astype(np.float32), similarity.astype(np.float32), perturbed_norm.astype(np.float32)


def predict_original_embeddings(
    model: tf.keras.Model,
    patch_paths: list[str],
    patch_size: int,
    num_channels: int,
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    for _, _, paths in tqdm(
        iter_batches(patch_paths, batch_size),
        total=int(np.ceil(len(patch_paths) / batch_size)),
        desc="Recomputing original embeddings",
    ):
        batch = load_patch_batch(paths, patch_size, num_channels)
        embeddings.append(model.predict(batch, verbose=0))
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def run_condition(
    model: tf.keras.Model,
    condition: PerturbationCondition,
    condition_index: int,
    patch_paths: list[str],
    original_embeddings: np.ndarray,
    patch_size: int,
    num_channels: int,
    batch_size: int,
    seed: int,
    shuffle_pixels: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.empty(len(patch_paths), dtype=np.float32)
    similarities = np.empty(len(patch_paths), dtype=np.float32)
    perturbed_norms = np.empty(len(patch_paths), dtype=np.float32)
    rng = np.random.default_rng(seed + condition_index + 1)

    total_batches = int(np.ceil(len(patch_paths) / batch_size))
    batch_iter = iter_batches(patch_paths, batch_size)
    for start, end, paths in tqdm(batch_iter, total=total_batches, desc=condition.condition_id):
        batch = load_patch_batch(paths, patch_size, num_channels)
        perturbed = perturb_batch(batch, condition, rng, shuffle_pixels)
        perturbed_embeddings = model.predict(perturbed, verbose=0)
        distance, similarity, norm = cosine_metrics(original_embeddings[start:end], perturbed_embeddings)
        distances[start:end] = distance
        similarities[start:end] = similarity
        perturbed_norms[start:end] = norm

    return distances, similarities, perturbed_norms


def summarize_conditions(
    var: pd.DataFrame,
    distances: np.ndarray,
    similarities: np.ndarray,
    perturbed_norms: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for idx, condition_id in enumerate(var.index):
        distance = distances[:, idx]
        similarity = similarities[:, idx]
        norm = perturbed_norms[:, idx]
        finite = np.isfinite(distance)
        row = var.loc[condition_id].to_dict()
        row.update(
            {
                "condition_id": condition_id,
                "n_patches": int(distance.size),
                "n_valid": int(finite.sum()),
                "mean_cosine_distance": float(np.nanmean(distance)),
                "median_cosine_distance": float(np.nanmedian(distance)),
                "std_cosine_distance": float(np.nanstd(distance)),
                "p95_cosine_distance": float(np.nanpercentile(distance, 95)),
                "max_cosine_distance": float(np.nanmax(distance)),
                "mean_cosine_similarity": float(np.nanmean(similarity)),
                "mean_perturbed_embedding_norm": float(np.nanmean(norm)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_patches(obs_names: pd.Index, distances: np.ndarray, var: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(index=obs_names)
    summary.index.name = "patch_id"
    summary["mean_cosine_distance"] = np.nanmean(distances, axis=1)
    summary["median_cosine_distance"] = np.nanmedian(distances, axis=1)
    summary["max_cosine_distance"] = np.nanmax(distances, axis=1)

    for perturbation_type in sorted(var["perturbation_type"].unique()):
        columns = np.where(var["perturbation_type"].to_numpy() == perturbation_type)[0]
        if columns.size:
            summary[f"mean_distance__{perturbation_type}"] = np.nanmean(distances[:, columns], axis=1)
    return summary.reset_index()


def make_output_anndata(
    input_adata: ad.AnnData,
    distances: np.ndarray,
    similarities: np.ndarray,
    perturbed_norms: np.ndarray,
    original_embeddings: np.ndarray,
    var: pd.DataFrame,
    args: argparse.Namespace,
    channel_names: list[str],
    channels_to_perturb: list[str],
) -> ad.AnnData:
    obs = input_adata.obs.copy()
    obs["original_embedding_norm"] = np.linalg.norm(original_embeddings, axis=1).astype(np.float32)
    output = ad.AnnData(
        X=distances.astype(np.float32, copy=False),
        obs=obs,
        var=var.copy(),
    )
    output.layers["cosine_similarity"] = similarities.astype(np.float32, copy=False)
    output.layers["perturbed_embedding_norm"] = perturbed_norms.astype(np.float32, copy=False)

    for key in ["spatial", "X_umap", "X_pca"]:
        if key in input_adata.obsm:
            output.obsm[key] = input_adata.obsm[key].copy()

    output.uns["channel_names"] = channel_names
    output.uns["channels_perturbed"] = channels_to_perturb
    output.uns["hyperstac_output_dir"] = str(args.hyperstac_output_dir)
    output.uns["source_adata_path"] = str(args.adata_path)
    output.uns["encoder_weights"] = str(args.encoder_weights)
    output.uns["shuffle_pixels"] = args.shuffle_pixels
    output.uns["n_shuffle_repeats"] = args.n_shuffle_repeats
    output.uns["metric"] = "cosine_distance"
    output.uns["x_description"] = "1 - cosine_similarity(original_embedding, perturbed_embedding)"
    return output


def write_wide_csvs(
    output_dir: Path,
    obs_names: pd.Index,
    var_names: pd.Index,
    distances: np.ndarray,
    similarities: np.ndarray,
    perturbed_norms: np.ndarray,
) -> None:
    for filename, matrix in [
        ("permutation_cosine_distances.csv.gz", distances),
        ("permutation_cosine_similarities.csv.gz", similarities),
        ("permutation_perturbed_embedding_norms.csv.gz", perturbed_norms),
    ]:
        df = pd.DataFrame(matrix, index=obs_names, columns=var_names)
        df.index.name = "patch_id"
        df.to_csv(output_dir / filename)


def write_run_config(
    args: argparse.Namespace,
    output_path: Path,
    channel_names: list[str],
    channels_to_perturb: list[str],
    encoder_name: str,
    patch_size: int,
) -> None:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    config["channel_names"] = channel_names
    config["channels_to_perturb"] = channels_to_perturb
    config["resolved_encoder"] = encoder_name
    config["resolved_patch_size"] = patch_size
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = resolve_paths(parse_args())
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    configure_runtime(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.adata_path)
    channel_names = load_channel_names(args, adata)
    channels_to_perturb = parse_channels_to_perturb(args.channels, channel_names)
    metadata = load_aligned_metadata(args, adata)
    patch_paths = metadata["patch_path"].astype(str).tolist()
    validate_patch_paths(patch_paths)
    encoder_name, patch_size = infer_encoder_and_patch_size(args, adata)

    print(f"Representation AnnData: {args.adata_path}")
    print(f"Patch count: {adata.n_obs}")
    print(f"Channels: {len(channel_names)}")
    print(f"Channels perturbed: {len(channels_to_perturb)}")
    print(f"Encoder: {encoder_name}")
    print(f"Patch size: {patch_size}")
    print(f"Output dir: {args.output_dir}")

    model = build_representation_model(
        encoder_name=encoder_name,
        patch_size=patch_size,
        num_channels=len(channel_names),
        weights_path=args.encoder_weights,
    )

    if args.recompute_original:
        original_embeddings = predict_original_embeddings(
            model,
            patch_paths,
            patch_size=patch_size,
            num_channels=len(channel_names),
            batch_size=args.batch_size,
        )
    else:
        original_embeddings = matrix_from_adata_x(adata)
    if original_embeddings.shape[0] != adata.n_obs:
        raise ValueError(
            "Original embedding row count does not match AnnData observations: "
            f"{original_embeddings.shape[0]} vs {adata.n_obs}"
        )

    conditions = build_conditions(
        channel_names=channel_names,
        channels_to_perturb=channels_to_perturb,
        n_shuffle_repeats=args.n_shuffle_repeats,
        include_all_channels=args.include_all_channel_perturbations,
    )
    var = condition_var_dataframe(conditions)
    var.to_csv(args.output_dir / "permutation_conditions.csv")

    distances = np.empty((adata.n_obs, len(conditions)), dtype=np.float32)
    similarities = np.empty_like(distances)
    perturbed_norms = np.empty_like(distances)

    for condition_index, condition in enumerate(conditions):
        distance, similarity, norm = run_condition(
            model=model,
            condition=condition,
            condition_index=condition_index,
            patch_paths=patch_paths,
            original_embeddings=original_embeddings,
            patch_size=patch_size,
            num_channels=len(channel_names),
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle_pixels=args.shuffle_pixels,
        )
        distances[:, condition_index] = distance
        similarities[:, condition_index] = similarity
        perturbed_norms[:, condition_index] = norm

    out_adata = make_output_anndata(
        input_adata=adata,
        distances=distances,
        similarities=similarities,
        perturbed_norms=perturbed_norms,
        original_embeddings=original_embeddings,
        var=var,
        args=args,
        channel_names=channel_names,
        channels_to_perturb=channels_to_perturb,
    )
    out_path = args.output_dir / args.adata_name
    out_adata.write_h5ad(out_path)

    condition_summary = summarize_conditions(var, distances, similarities, perturbed_norms)
    condition_summary.to_csv(args.output_dir / "permutation_condition_summary.csv", index=False)

    patch_summary = summarize_patches(adata.obs_names, distances, var)
    patch_summary.to_csv(args.output_dir / "permutation_patch_summary.csv", index=False)

    if args.write_wide_csv:
        write_wide_csvs(
            output_dir=args.output_dir,
            obs_names=adata.obs_names,
            var_names=var.index,
            distances=distances,
            similarities=similarities,
            perturbed_norms=perturbed_norms,
        )

    write_run_config(
        args=args,
        output_path=args.output_dir / "permutation_run_config.json",
        channel_names=channel_names,
        channels_to_perturb=channels_to_perturb,
        encoder_name=encoder_name,
        patch_size=patch_size,
    )

    print(f"Saved perturbation AnnData to {out_path}")
    print(f"Saved condition summary to {args.output_dir / 'permutation_condition_summary.csv'}")
    print(f"Saved patch summary to {args.output_dir / 'permutation_patch_summary.csv'}")


if __name__ == "__main__":
    main()
