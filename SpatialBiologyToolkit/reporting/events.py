"""Shell-facing reporting finalizer used by shared SLURM job hygiene."""

from __future__ import annotations

import argparse
import sys

from .reporter import StageReporter


def start_shell_stage() -> None:
    reporter = StageReporter.from_environment()
    reporter.__enter__()


def finalize_shell_stage(exit_code: int) -> None:
    reporter = StageReporter.from_environment()
    reporter.__enter__()
    if exit_code:
        error = RuntimeError(f"Stage process exited with status {exit_code}.")
        reporter.finalize(status="failed", error=error)
    else:
        reporter.finalize(status="completed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true")
    action.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    try:
        if args.start:
            start_shell_stage()
        else:
            finalize_shell_stage(int(args.exit_code))
    except Exception as exc:
        print(f"SBT reporting finalization failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
