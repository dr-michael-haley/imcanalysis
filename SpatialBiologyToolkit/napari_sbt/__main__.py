# ruff: noqa: N999
"""Command-line launcher for the unified napari_sbt desktop workflow."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        help="Initialized project path, registered project name, or project ID.",
    )
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--anndata", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--images", type=Path, action="append", default=[])
    parser.add_argument("--extra-images", type=Path, action="append", default=[])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check launch prerequisites without importing Qt or opening Napari.",
    )
    parser.add_argument(
        "--check-format",
        choices=("text", "json"),
        default="text",
        help="Preflight output format.",
    )
    return parser


def _resolve_project_context(project: Path | None):
    from SpatialBiologyToolkit.pipeline.project import (
        ProjectNotFoundError,
        ProjectNotInitializedError,
        load_project,
    )

    if project is None:
        try:
            return load_project()
        except ProjectNotFoundError:
            # Preserve the standalone launcher outside an initialized project.
            return None

    try:
        return load_project(project)
    except ProjectNotInitializedError as path_error:
        from SpatialBiologyToolkit.pipeline.project_registry import (
            ProjectRegistryError,
            load_project_registry,
            resolve_registered_project,
        )

        try:
            registered = resolve_registered_project(
                load_project_registry(), str(project)
            )
        except ProjectRegistryError as registry_error:
            raise ProjectNotFoundError(
                f"Could not resolve --project {str(project)!r} as an initialized "
                "path, registered project name, or project ID. "
                f"Path lookup: {path_error} Registry lookup: {registry_error}"
            ) from registry_error
        return load_project(registered.path)


def _project_defaults(args) -> dict:
    context = _resolve_project_context(args.project)
    if context is None:
        return {}
    settings = context.config.napari_sbt
    defaults: dict = {
        "project_root": context.root,
        "anndata_path": context.root / context.config.general.anndata_path,
        "masks_folder": context.root / context.config.general.masks_folder,
        "images_folders": [
            context.root / context.config.general.denoised_images_folder
        ],
        "worker_count": settings.worker_count,
    }
    if settings.active_experiment:
        configured = Path(settings.active_experiment)
        defaults["experiment"] = (
            configured
            if configured.is_absolute()
            else context.root / settings.experiment_folder / configured
        )
    return defaults


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        defaults = _project_defaults(args)
    except Exception as exc:
        if not args.check:
            raise
        if args.check_format == "json":
            import json

            print(
                json.dumps(
                    {
                        "ready": False,
                        "checks": [
                            {
                                "name": "Project configuration",
                                "status": "error",
                                "detail": str(exc),
                            }
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print(
                "NapariSBT preflight: BLOCKED\n"
                f"[ERROR] Project configuration: {exc}"
            )
        return 2
    project_root = defaults.get("project_root")
    experiment = args.experiment or defaults.get("experiment")
    anndata_path = args.anndata or defaults.get("anndata_path")
    masks_folder = args.masks or defaults.get("masks_folder")
    images_folders = args.images or defaults.get("images_folders", [])
    if args.check:
        from .preflight import format_preflight, run_preflight

        report = run_preflight(
            project_root=project_root,
            experiment=experiment,
            anndata_path=anndata_path,
            masks_folder=masks_folder,
            images_folders=tuple(images_folders) + tuple(args.extra_images),
            worker_count=defaults.get("worker_count"),
        )
        print(format_preflight(report, args.check_format))
        return report.exit_code
    from .app import launch

    launch(
        project_root=project_root,
        experiment=experiment,
        anndata_path=anndata_path,
        masks_folder=masks_folder,
        images_folders=images_folders,
        extra_images_folders=args.extra_images,
    )
    import napari

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
