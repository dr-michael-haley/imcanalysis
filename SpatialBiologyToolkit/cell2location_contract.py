"""Lightweight file contract shared by planning and cell2location execution."""

from __future__ import annotations

from pathlib import Path

from SpatialBiologyToolkit.config.models import Cell2locationConfig


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve(strict=False)


def signature_path(settings: Cell2locationConfig, root: Path) -> Path:
    return resolve_path(
        root,
        settings.signatures_path
        or str(Path(settings.asset_folder) / "reference" / "signatures.csv"),
    )


def input_paths(settings: Cell2locationConfig, root: Path) -> dict[str, Path]:
    paths = {}
    if settings.action in {"reference", "full"}:
        paths["cell2location_reference"] = resolve_path(
            root, settings.reference_adata_path
        )
    if settings.action == "map":
        paths["cell2location_signatures"] = signature_path(settings, root)
    if settings.action in {"map", "full"}:
        paths.update(
            {
                f"cell2location_visium_{i}": resolve_path(root, source.path)
                for i, source in enumerate(settings.visium_inputs)
            }
        )
        if settings.cell_count_prior_csv:
            paths["cell2location_prior"] = resolve_path(
                root, settings.cell_count_prior_csv
            )
    return paths


def preflight_errors(settings: Cell2locationConfig, root: Path) -> list[str]:
    """Check configuration and all configured paths without opening scientific data."""
    errors = []
    if settings.action in {"map", "full"} and not settings.visium_inputs:
        errors.append(
            "cell2location.visium_inputs must contain at least one Visium H5AD"
        )
    if settings.action == "full" and settings.signatures_path:
        errors.append(
            "signatures_path is only used with action=map; full creates a new reference"
        )
    paths = input_paths(settings, root)
    for role, path in paths.items():
        if not path.is_file():
            errors.append(f"{role}: required file is missing: {path}")
    visium_paths = [
        p for role, p in paths.items() if role.startswith("cell2location_visium_")
    ]
    if len(visium_paths) != len(set(visium_paths)):
        errors.append("The same Visium file was configured more than once")
    outputs = (
        ["reference", "mapping"]
        if settings.action == "full"
        else ["reference" if settings.action == "reference" else "mapping"]
    )
    asset_root = resolve_path(root, settings.asset_folder)
    for part in outputs:
        target = asset_root / part
        if target.exists():
            errors.append(
                f"Output already exists: {target}; choose a new cell2location.asset_folder"
            )
        for source in paths.values():
            if source == target or target in source.parents:
                errors.append(f"Input {source} is inside the proposed output {target}")
    if asset_root.exists() and not asset_root.is_dir():
        errors.append(f"asset_folder is not a directory: {asset_root}")
    return errors
