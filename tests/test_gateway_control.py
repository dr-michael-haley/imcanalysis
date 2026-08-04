from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.pipeline.control import (
    make_preview_token,
    preview_run_identities,
    run_preview_snapshot,
    validate_preview_token,
)
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.runs import (
    create_run_record,
    prospective_run_record,
)
from SpatialBiologyToolkit.pipeline.scheduler import (
    cancel_job,
    list_user_jobs,
    preview_cancellation,
)
from SpatialBiologyToolkit.pipeline.transfers import (
    HDF5_SIGNATURE,
    LARGE_TRANSFER_BYTES,
    _safe_extract,
    commit_upload,
    list_backups,
    list_transfer_items,
    prepare_upload,
    preview_upload,
    restore_backup,
)


class GatewayControlTests(unittest.TestCase):
    def test_queue_is_current_user_scoped_and_never_serializes_username(self):
        calls: list[list[str]] = []

        def runner(arguments, **_kwargs):
            calls.append(arguments)
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout="12345|RUNNING|analysis|cpu|00:12|01:00|1|8|None\n",
                stderr="",
            )

        with (
            patch(
                "SpatialBiologyToolkit.pipeline.scheduler.getpass.getuser",
                return_value="private-user",
            ),
            patch(
                "SpatialBiologyToolkit.pipeline.scheduler._managed_jobs",
                return_value={},
            ),
        ):
            snapshot = list_user_jobs(runner=runner)

        self.assertIn("--user", calls[0])
        self.assertIn("private-user", calls[0])
        serialized = snapshot.model_dump_json()
        self.assertNotIn("private-user", serialized)
        self.assertEqual(snapshot.jobs[0].job_id, "12345")

    def test_cancel_preview_requires_confirmation_only_for_active_running_work(self):
        state = "PENDING"
        calls: list[list[str]] = []

        def runner(arguments, **_kwargs):
            calls.append(arguments)
            if arguments[0] == "squeue":
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=f"12345|{state}|analysis|cpu|00:00|01:00|1|8|Resources\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

        with patch(
            "SpatialBiologyToolkit.pipeline.scheduler._managed_jobs",
            return_value={},
        ), patch(
            "SpatialBiologyToolkit.pipeline.scheduler.write_json",
            return_value=Path("audit.json"),
        ):
            pending = preview_cancellation(
                "12345",
                reason="Incorrect parameters",
                runner=runner,
            )
            self.assertFalse(pending["confirmation_required"])
            result = cancel_job(
                "12345",
                reason="Incorrect parameters",
                preview_token=pending["preview_token"],
                provenance={"request_summary": "Cancel incorrect job", "decisions": [{}]},
                runner=runner,
            )
        self.assertEqual(result["outcome"], "cancellation_requested")
        self.assertEqual(calls[-1][0], "scancel")
        self.assertEqual(calls[-1][-1], "12345")

        state = "RUNNING"
        with patch(
            "SpatialBiologyToolkit.pipeline.scheduler._managed_jobs",
            return_value={},
        ):
            running = preview_cancellation(
                "12345",
                reason="Terminal input failure",
                runner=runner,
            )
        self.assertTrue(running["confirmation_required"])
        with patch(
            "SpatialBiologyToolkit.pipeline.scheduler._managed_jobs",
            return_value={},
        ), self.assertRaisesRegex(ValueError, "explicit confirmation"):
            cancel_job(
                "12345",
                reason="Terminal input failure",
                preview_token=running["preview_token"],
                provenance={"request_summary": "Cancel stalled job", "decisions": [{}]},
                runner=runner,
            )

    def test_transfer_inventory_includes_panel_and_root_h5ad(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            metadata = Path(context.config.general.metadata_folder)
            if not metadata.is_absolute():
                metadata = context.root / metadata
            metadata.mkdir(parents=True, exist_ok=True)
            (metadata / "panel.csv").write_text("name,channel\nDNA,Ir191\n", encoding="utf-8")
            (context.root / "atlas.h5ad").write_bytes(HDF5_SIGNATURE + b"atlas")

            items = list_transfer_items(context)
            relative = {item.relative_path for item in items}

            self.assertIn("atlas.h5ad", relative)
            self.assertIn(
                (metadata / "panel.csv").relative_to(context.root).as_posix(),
                relative,
            )

    def test_h5ad_overwrite_creates_backup_and_restore_retains_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            target = context.root / "atlas.h5ad"
            original = HDF5_SIGNATURE + b"original"
            replacement = HDF5_SIGNATURE + b"replacement"
            target.write_bytes(original)
            digest = hashlib.sha256(replacement).hexdigest()
            preview = preview_upload(
                context,
                name=target.name,
                destination="project-root",
                kind="file",
                size_bytes=len(replacement),
                sha256=digest,
                overwrite=True,
            )
            prepared = prepare_upload(
                context,
                name=target.name,
                destination="project-root",
                kind="file",
                size_bytes=len(replacement),
                sha256=digest,
                overwrite=True,
                preview_token=preview["preview_token"],
            )
            Path(prepared["staging_path"]).write_bytes(replacement)
            committed = commit_upload(
                context,
                prepared["transfer_id"],
                provenance={"request_summary": "Replace atlas", "decisions": [{}]},
            )

            self.assertEqual(target.read_bytes(), replacement)
            self.assertIsNotNone(committed["backup_id"])
            self.assertEqual(len(list_backups(context)), 1)

            restore_preview = restore_backup(
                context,
                committed["backup_id"],
                dry_run=True,
            )
            restored = restore_backup(
                context,
                committed["backup_id"],
                dry_run=False,
                preview_token=restore_preview["preview_token"],
                provenance={"request_summary": "Restore atlas", "decisions": [{}]},
            )
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(restored["source_backup_retained"])
            self.assertEqual(len(list_backups(context)), 2)

    def test_large_upload_needs_explicit_permission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            preview = preview_upload(
                context,
                name="large.h5ad",
                destination="project-root",
                kind="file",
                size_bytes=LARGE_TRANSFER_BYTES + 1,
                sha256="a" * 64,
                overwrite=False,
            )
            self.assertTrue(preview["requires_large_transfer_permission"])
            with self.assertRaisesRegex(ValueError, "explicit large-transfer permission"):
                prepare_upload(
                    context,
                    name="large.h5ad",
                    destination="project-root",
                    kind="file",
                    size_bytes=LARGE_TRANSFER_BYTES + 1,
                    sha256="a" * 64,
                    overwrite=False,
                    preview_token=preview["preview_token"],
                )

    def test_upload_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                _safe_extract(archive, Path(temp_dir) / "output")

    def test_cli_run_preview_returns_token_receipt_and_does_not_submit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            result = CliRunner().invoke(
                app,
                [
                    "run",
                    "debug",
                    "--project",
                    str(context.root),
                    "--dry-run",
                    "--format",
                    "json",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["preview_token"].startswith("v1."))
            self.assertEqual(payload["action_receipt"]["operation"], "preview_run")
            self.assertEqual(list((context.root / ".sbt" / "runs").iterdir()), [])

    def test_cli_run_preview_token_submits_unchanged_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            runner = CliRunner()
            preview = runner.invoke(
                app,
                [
                    "run",
                    "debug",
                    "--project",
                    str(context.root),
                    "--dry-run",
                    "--format",
                    "json",
                ],
            )
            self.assertEqual(preview.exit_code, 0, preview.stdout)
            payload = json.loads(preview.stdout)

            with (
                patch("SpatialBiologyToolkit.cli.main._ensure_run_environments"),
                patch(
                    "SpatialBiologyToolkit.cli.main.submit_run",
                    return_value=SimpleNamespace(jobs=[]),
                ) as submit_mock,
            ):
                submitted = runner.invoke(
                    app,
                    [
                        "run",
                        "debug",
                        "--project",
                        str(context.root),
                        "--plan-token",
                        payload["preview_token"],
                    ],
                )

            self.assertEqual(submitted.exit_code, 0, submitted.stdout)
            self.assertIn(payload["prospective_workflow_run_id"], submitted.stdout)
            submit_mock.assert_called_once()
            self.assertTrue(
                (context.root / ".sbt" / "runs" / payload["prospective_workflow_run_id"]).is_dir()
            )

    def test_run_token_accepts_unchanged_asset_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            plan = build_run_plan(context, ["debug"])
            token = make_preview_token(run_preview_snapshot(context, plan))

            validate_preview_token(token, run_preview_snapshot(context, plan))

    def test_guarded_preview_and_created_run_use_the_same_exact_identities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            plan = build_run_plan(context, ["debug"])
            token = make_preview_token(run_preview_snapshot(context, plan))
            workflow_id, technical_ids = preview_run_identities(
                token,
                len(plan.resolved_stages),
            )
            preview = prospective_run_record(
                context,
                plan,
                run_id=workflow_id,
                technical_run_ids=technical_ids,
            )
            created = create_run_record(
                context,
                plan,
                command="sbt run debug",
                run_id=workflow_id,
                technical_run_ids=technical_ids,
            )

            self.assertEqual(created.workflow_run_id, preview.workflow_run_id)
            self.assertEqual(
                [item.technical_run_id for item in created.executions],
                [item.technical_run_id for item in preview.executions],
            )

    def test_run_token_rejects_asset_inventory_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            plan = build_run_plan(context, ["debug"])
            token = make_preview_token(run_preview_snapshot(context, plan))
            (context.root / "IMC_files" / "new.mcd").write_bytes(b"new")

            with self.assertRaisesRegex(ValueError, "changed after preview"):
                validate_preview_token(
                    token,
                    run_preview_snapshot(context, plan),
                )


if __name__ == "__main__":
    unittest.main()
