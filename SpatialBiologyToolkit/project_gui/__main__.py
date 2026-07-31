"""Command-line entry point for the SBT Project Console."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Disable configuration and project-notes writes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .app import launch

    return launch(project=args.project, read_only=args.read_only)


if __name__ == "__main__":
    raise SystemExit(main())
