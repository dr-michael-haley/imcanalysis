from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.napari_sbt.preflight import format_preflight, run_preflight
from SpatialBiologyToolkit.napari_sbt.resources import resolve_worker_count


class NapariSBTResourceTests(unittest.TestCase):
    def test_feature_workers_are_clamped_to_slurm_affinity(self):
        resolution = resolve_worker_count(
            8,
            environ={"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "8"},
            affinity_count=4,
        )

        self.assertEqual(resolution.requested, 8)
        self.assertEqual(resolution.effective, 4)
        self.assertTrue(resolution.adjusted)
        self.assertIn("using 4", resolution.message)

    def test_preflight_is_read_only_and_reports_machine_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anndata_path = root / "cells.h5ad"
            anndata_path.write_bytes(b"not opened by preflight")
            masks = root / "masks"
            images = root / "images"
            masks.mkdir()
            images.mkdir()
            before = sorted(path.relative_to(root) for path in root.rglob("*"))

            with patch(
                "SpatialBiologyToolkit.napari_sbt.preflight._module_available",
                return_value=True,
            ), patch.dict(
                "SpatialBiologyToolkit.napari_sbt.preflight.os.environ",
                {"SLURM_JOB_ID": "123", "DISPLAY": "localhost:10.0"},
                clear=True,
            ):
                report = run_preflight(
                    project_root=root,
                    anndata_path=anndata_path,
                    masks_folder=masks,
                    images_folders=(images,),
                    worker_count=2,
                )

            after = sorted(path.relative_to(root) for path in root.rglob("*"))
            payload = json.loads(format_preflight(report, "json"))
            self.assertTrue(payload["ready"])
            self.assertEqual(before, after)


class NapariSBTLauncherTests(unittest.TestCase):
    def test_cli_uses_registered_napari_environment_and_forwards_preflight(self):
        with patch(
            "SpatialBiologyToolkit.cli.main._napari_runtime_available",
            return_value=False,
        ), patch(
            "SpatialBiologyToolkit.cli.main.shutil.which", return_value="conda"
        ), patch(
            "SpatialBiologyToolkit.cli.main.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            result = CliRunner().invoke(
                app,
                [
                    "gui",
                    "napari",
                    "--project",
                    "example",
                    "--check",
                    "--check-format",
                    "json",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        command = run.call_args.args[0]
        self.assertEqual(
            command[:7],
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "sbt-napari",
                "python",
                "-m",
            ],
        )
        self.assertEqual(command[-3:], ["--check", "--check-format", "json"])


if __name__ == "__main__":
    unittest.main()
