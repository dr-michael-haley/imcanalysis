"""PyTorch VICReg implementation for masked 36 x 36 CellVision images.

The encoder and augmentations are purpose-built for arbitrary-channel IMC
portraits.  They do not assume RGB semantics, introduce artificial background,
or crop already-tight cell images.
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, Dataset


H5SC_IMAGE_DATASET = "obsm/single_cell_images"


class H5SCImageDataset(Dataset):
    """Process-safe, lazy PyTorch reader for selected H5SC image channels."""

    def __init__(
        self,
        path: str | Path,
        *,
        channel_indices: Sequence[int],
        channel_scales: Sequence[float] | None = None,
    ) -> None:
        self.path = str(Path(path).expanduser().resolve(strict=True))
        self.channel_indices = [int(value) for value in channel_indices]
        if not self.channel_indices:
            raise ValueError("H5SCImageDataset requires at least one image channel.")
        with h5py.File(self.path, "r") as handle:
            if H5SC_IMAGE_DATASET not in handle:
                raise KeyError(f"H5SC image tensor not found at /{H5SC_IMAGE_DATASET}")
            shape = tuple(int(value) for value in handle[H5SC_IMAGE_DATASET].shape)
        if len(shape) != 4:
            raise ValueError(f"Expected H5SC image shape (N, C, H, W), got {shape}")
        if max(self.channel_indices) >= shape[1] or min(self.channel_indices) < 0:
            raise IndexError(
                f"Selected H5SC channel indices {self.channel_indices} exceed image shape {shape}."
            )
        self.shape = shape
        self._handle: h5py.File | None = None
        self.channel_scales: np.ndarray
        if channel_scales is None:
            self.channel_scales = np.ones(len(self.channel_indices), dtype=np.float32)
        else:
            scales = np.asarray(channel_scales, dtype=np.float32)
            if scales.shape != (len(self.channel_indices),):
                raise ValueError(
                    f"Expected {len(self.channel_indices)} channel scales, got {scales.shape}."
                )
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("All VICReg channel scales must be finite and positive.")
            self.channel_scales = scales

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _images(self) -> h5py.Dataset:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[H5SC_IMAGE_DATASET]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - interpreter cleanup
        self.close()

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = np.asarray(
            self._images()[int(index), self.channel_indices, :, :],
            dtype=np.float32,
        )
        image = np.clip(
            image / self.channel_scales[:, None, None],
            0.0,
            1.0,
        )
        return torch.from_numpy(image), int(index)


def estimate_channel_scales(
    path: str | Path,
    *,
    channel_indices: Sequence[int],
    quantile: float,
    max_cells: int,
    seed: int,
    max_values_per_channel: int = 200_000,
) -> np.ndarray:
    """Estimate positive-pixel marker scales from a bounded random cell sample."""
    rng = np.random.default_rng(int(seed))
    with h5py.File(Path(path), "r") as handle:
        images = handle[H5SC_IMAGE_DATASET]
        n_cells = int(images.shape[0])
        if n_cells < 2:
            raise ValueError("VICReg requires at least two extracted H5SC cells.")
        sample_size = min(int(max_cells), n_cells)
        rows = np.sort(rng.choice(n_cells, size=sample_size, replace=False))
        reservoirs: list[list[np.ndarray]] = [[] for _ in channel_indices]
        counts: NDArray[np.int64] = np.zeros(len(channel_indices), dtype=np.int64)
        values_per_cell = max(16, int(max_values_per_channel // max(1, sample_size)))
        for row in rows:
            cell = np.asarray(images[int(row), list(channel_indices)], dtype=np.float32)
            for channel, pixels in enumerate(cell):
                positive = pixels[pixels > 0]
                if positive.size > values_per_cell:
                    positive = rng.choice(positive, size=values_per_cell, replace=False)
                if positive.size:
                    reservoirs[channel].append(np.asarray(positive, dtype=np.float32))
                    counts[channel] += positive.size

    scales: NDArray[np.float32] = np.ones(len(channel_indices), dtype=np.float32)
    for channel, chunks in enumerate(reservoirs):
        if not chunks:
            logging.warning(
                "H5SC image channel index %d contains no positive sampled pixels; using scale 1.",
                channel_indices[channel],
            )
            continue
        values = np.concatenate(chunks)
        if values.size > max_values_per_channel:
            values = rng.choice(values, size=max_values_per_channel, replace=False)
        scale = float(np.quantile(values, float(quantile)))
        scales[channel] = max(scale, float(np.finfo(np.float32).eps))
    logging.info(
        "Estimated VICReg positive-pixel channel scales from %d cells (sampled counts=%s).",
        sample_size,
        counts.tolist(),
    )
    return scales


def _translate_zero_filled(image: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    output = torch.zeros_like(image)
    height, width = image.shape[-2:]
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    target_y0 = max(0, dy)
    target_y1 = target_y0 + max(0, source_y1 - source_y0)
    target_x0 = max(0, dx)
    target_x1 = target_x0 + max(0, source_x1 - source_x0)
    if source_y1 > source_y0 and source_x1 > source_x0:
        output[:, target_y0:target_y1, target_x0:target_x1] = image[
            :, source_y0:source_y1, source_x0:source_x1
        ]
    return output


class MaskSafeAugment:
    """Geometry/intensity augmentation that preserves exact zero background."""

    def __init__(self, *, translation_px: int, intensity_jitter: float, noise_std: float):
        self.translation_px = int(translation_px)
        self.intensity_jitter = float(intensity_jitter)
        self.noise_std = float(noise_std)

    def _one(self, image: torch.Tensor) -> torch.Tensor:
        output = image
        if torch.rand((), device=output.device) < 0.5:
            output = torch.flip(output, dims=(-1,))
        if torch.rand((), device=output.device) < 0.5:
            output = torch.flip(output, dims=(-2,))
        rotations = int(torch.randint(0, 4, (), device=output.device).item())
        output = torch.rot90(output, rotations, dims=(-2, -1))
        if self.translation_px:
            dy = int(
                torch.randint(
                    -self.translation_px,
                    self.translation_px + 1,
                    (),
                    device=output.device,
                ).item()
            )
            dx = int(
                torch.randint(
                    -self.translation_px,
                    self.translation_px + 1,
                    (),
                    device=output.device,
                ).item()
            )
            output = _translate_zero_filled(output, dy, dx)
        if self.intensity_jitter:
            lower = 1.0 - self.intensity_jitter
            upper = 1.0 + self.intensity_jitter
            scales = torch.empty(
                (output.shape[0], 1, 1), device=output.device, dtype=output.dtype
            ).uniform_(lower, upper)
            output = output * scales
        if self.noise_std:
            foreground = output.ne(0).any(dim=0, keepdim=True)
            output = output + torch.randn_like(output) * self.noise_std * foreground
        return output.clamp_(0, 1)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._one(image) for image in batch], dim=0)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.activation = nn.SiLU(inplace=True)
        self.skip: nn.Module
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(output_channels), output_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class CellVisionEncoder(nn.Module):
    """Compact residual encoder with an IMC-compatible arbitrary-channel stem."""

    def __init__(self, input_channels: int, *, width: int, embedding_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(width, width),
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, stride=2),
            ResidualBlock(width * 2, width * 2),
            ResidualBlock(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4, width * 4),
            ResidualBlock(width * 4, width * 8, stride=2),
            ResidualBlock(width * 8, width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.pool(self.features(self.stem(inputs))))


class VICRegNetwork(nn.Module):
    """CellVision encoder plus training-only VICReg projector."""

    def __init__(
        self,
        input_channels: int,
        *,
        width: int,
        embedding_dim: int,
        projector_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = CellVisionEncoder(
            input_channels,
            width=int(width),
            embedding_dim=int(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.SiLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.SiLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(inputs)
        return embedding, self.projector(embedding)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size, second = matrix.shape
    if size != second:
        raise ValueError("VICReg covariance matrix must be square.")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute standard invariance, variance, and covariance VICReg terms."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(
            f"VICReg views must be equal 2D tensors, got {first.shape} and {second.shape}."
        )
    if first.shape[0] < 2:
        raise ValueError("VICReg loss requires at least two samples per batch.")
    invariance = torch.mean((first - second) ** 2)
    centered_first = first - first.mean(dim=0)
    centered_second = second - second.mean(dim=0)
    std_first = torch.sqrt(centered_first.var(dim=0, unbiased=True) + epsilon)
    std_second = torch.sqrt(centered_second.var(dim=0, unbiased=True) + epsilon)
    variance = 0.5 * (
        torch.relu(1.0 - std_first).mean() + torch.relu(1.0 - std_second).mean()
    )
    denominator = first.shape[0] - 1
    covariance_first = centered_first.T @ centered_first / denominator
    covariance_second = centered_second.T @ centered_second / denominator
    dimensions = first.shape[1]
    covariance = (
        _off_diagonal(covariance_first).pow(2).sum()
        + _off_diagonal(covariance_second).pow(2).sum()
    ) / dimensions
    total = (
        float(invariance_weight) * invariance
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
    )
    return total, {
        "invariance_loss": invariance,
        "variance_loss": variance,
        "covariance_loss": covariance,
    }


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch workers for a reproducible CellVision run."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _learning_rate_multiplier(epoch: int, *, epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    decay_epochs = max(1, epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / decay_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA is unavailable; CellVision VICReg will run on CPU.")
    return device


def train_vicreg(
    dataset: H5SCImageDataset,
    *,
    width: int,
    embedding_dim: int,
    projector_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    num_workers: int,
    seed: int,
    amp: bool,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
    translation_px: int,
    intensity_jitter: float,
    noise_std: float,
) -> tuple[VICRegNetwork, pd.DataFrame, torch.device]:
    """Train VICReg and return the model plus epoch-level objective history."""
    if len(dataset) < 2:
        raise ValueError("CellVision VICReg requires at least two extracted cells.")
    seed_everything(seed)
    device = _device()
    effective_batch = min(int(batch_size), len(dataset))
    drop_last = len(dataset) > effective_batch and len(dataset) % effective_batch == 1
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        persistent_workers=bool(num_workers),
        worker_init_fn=_worker_seed if num_workers else None,
        generator=generator,
    )
    model = VICRegNetwork(
        len(dataset.channel_indices),
        width=int(width),
        embedding_dim=int(embedding_dim),
        projector_dim=int(projector_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _learning_rate_multiplier(
            epoch, epochs=int(epochs), warmup_epochs=int(warmup_epochs)
        ),
    )
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    augment = MaskSafeAugment(
        translation_px=int(translation_px),
        intensity_jitter=float(intensity_jitter),
        noise_std=float(noise_std),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(int(epochs)):
        model.train()
        totals = {
            "loss": 0.0,
            "invariance_loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
        }
        batches = 0
        for images, _indices in loader:
            images = images.to(device, non_blocking=True)
            first_view = augment(images)
            second_view = augment(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _first_embedding, first_projection = model(first_view)
                _second_embedding, second_projection = model(second_view)
                loss, components = vicreg_loss(
                    first_projection,
                    second_projection,
                    invariance_weight=float(invariance_weight),
                    variance_weight=float(variance_weight),
                    covariance_weight=float(covariance_weight),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach().cpu())
            for name, value in components.items():
                totals[name] += float(value.detach().cpu())
            batches += 1
        if batches == 0:
            raise RuntimeError("VICReg DataLoader produced no training batches.")
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        row.update({name: value / batches for name, value in totals.items()})
        history.append(row)
        logging.info(
            "VICReg epoch %d/%d: loss=%.6f inv=%.6f var=%.6f cov=%.6f",
            epoch + 1,
            epochs,
            row["loss"],
            row["invariance_loss"],
            row["variance_loss"],
            row["covariance_loss"],
        )
        scheduler.step()
    return model, pd.DataFrame(history), device


@torch.inference_mode()
def extract_embeddings(
    model: VICRegNetwork,
    dataset: H5SCImageDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one encoder embedding per H5SC row in deterministic order."""
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers),
    )
    model.eval()
    embeddings: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    for images, indices in loader:
        values = model.encoder(images.to(device, non_blocking=True))
        embeddings.append(values.float().cpu().numpy())
        row_indices.append(indices.numpy())
    matrix = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    rows = np.concatenate(row_indices).astype(np.int64, copy=False)
    if not np.array_equal(rows, np.arange(len(dataset), dtype=np.int64)):
        raise RuntimeError("VICReg inference did not preserve H5SC row order.")
    return matrix, rows


def save_checkpoint(
    path: Path,
    model: VICRegNetwork,
    *,
    architecture: dict[str, int],
    channel_indices: Sequence[int],
    channel_names: Sequence[str],
    channel_scales: Sequence[float],
    identity_fingerprint: str,
    training_fingerprint: str,
    training_config: dict[str, Any],
) -> None:
    """Atomically save the trained network and exact image-preprocessing contract."""
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "architecture": dict(architecture),
        "channel_indices": [int(value) for value in channel_indices],
        "channel_names": [str(value) for value in channel_names],
        "channel_scales": [float(value) for value in channel_scales],
        "identity_fingerprint": str(identity_fingerprint),
        "training_fingerprint": str(training_fingerprint),
        "training_config": dict(training_config),
        "torch_version": torch.__version__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, *, device: torch.device | None = None) -> tuple[VICRegNetwork, dict[str, Any]]:
    """Load and reconstruct one CellVision VICReg checkpoint."""
    target_device = device or _device()
    payload = torch.load(path, map_location=target_device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported CellVision checkpoint format: {path}")
    architecture = payload["architecture"]
    model = VICRegNetwork(
        int(architecture["input_channels"]),
        width=int(architecture["width"]),
        embedding_dim=int(architecture["embedding_dim"]),
        projector_dim=int(architecture["projector_dim"]),
    ).to(target_device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def plot_training_history(history: pd.DataFrame, output_path: Path, *, dpi: int) -> Path:
    """Write a compact VICReg total/component loss diagnostic."""
    import matplotlib.pyplot as plt

    required = {"epoch", "loss", "invariance_loss", "variance_loss", "covariance_loss"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"VICReg history is missing columns: {sorted(missing)}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["epoch"], history["loss"], color="black")
    axes[0].set_title("Weighted VICReg objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    for column, label in (
        ("invariance_loss", "Invariance"),
        ("variance_loss", "Variance"),
        ("covariance_loss", "Covariance"),
    ):
        axes[1].plot(history["epoch"], history[column], label=label)
    axes[1].set_title("Unweighted loss components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = [
    "CellVisionEncoder",
    "H5SCImageDataset",
    "MaskSafeAugment",
    "VICRegNetwork",
    "estimate_channel_scales",
    "extract_embeddings",
    "load_checkpoint",
    "plot_training_history",
    "save_checkpoint",
    "seed_everything",
    "train_vicreg",
    "vicreg_loss",
]
