from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.environments.manager import EnvironmentManager
from SpatialBiologyToolkit.environments.models import (
    CapturePlan,
    CondaEnvironmentRecord,
    EnvironmentCaptureTarget,
    EnvironmentRegistry,
)
from SpatialBiologyToolkit.environments.provenance import (
    snapshot_stage_environment_specifications,
)
from SpatialBiologyToolkit.environments.registry import (
    associated_stages,
    load_environment_registry,
    resolve_environment,
)
from SpatialBiologyToolkit.environments.runtime import conda_environment_records
from SpatialBiologyToolkit.environments.specification import (
    declared_conda_requirements,
    declared_pip_requirements,
    satisfies_constraint,
)
from SpatialBiologyToolkit.pipeline.registry import STAGES
from SpatialBiologyToolkit.reporting.inventory import discover_generated_files


class FakeRunner:
    def __init__(self, root: Path, *, exists: bool = True, drift: bool = False):
        self.root = root
        self.exists = exists
        self.drift = drift
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        command = [str(item) for item in command]
        self.calls.append(command)
        conda_lock_command = (
            command[5:]
            if command[:5] == ["conda", "run", "-n", "base", "conda-lock"]
            else None
        )
        stdout = ""
        stderr = ""
        return_code = 0
        if command[:4] == ["conda", "env", "list", "--json"]:
            stdout = json.dumps(
                {"envs": [str(self.root / "conda" / "envs" / "test_env")] if self.exists else []}
            )
        elif command[:3] in (
            ["conda", "list", "--name"],
            ["conda", "list", "--prefix"],
        ):
            packages = [
                {
                    "name": "python",
                    "version": "3.10" if self.drift else "3.11",
                    "build_string": "h123_0",
                    "channel": "conda-forge",
                },
                {
                    "name": "pip",
                    "version": "24.0",
                    "build_string": "pyhd8ed1ab_0",
                    "channel": "conda-forge",
                },
            ]
            stdout = json.dumps(packages)
        elif command[:3] in (
            ["conda", "run", "--name"],
            ["conda", "run", "--prefix"],
        ) and command[4] == "python":
            if "pip" in command and "list" in command and "--editable" in command:
                stdout = json.dumps(
                    [
                        {
                            "name": "SpatialBiologyToolkit",
                            "version": "0.1",
                            "editable_project_location": str(self.root),
                        }
                    ]
                )
            elif "pip" in command and "list" in command:
                stdout = json.dumps(
                    [
                        {"name": "foo", "version": "1.0"},
                        {"name": "SpatialBiologyToolkit", "version": "0.1"},
                    ]
                )
            elif "pip" in command and "freeze" in command:
                stdout = (
                    "foo==1.0\n"
                    f"-e {self.root.as_posix()}#egg=SpatialBiologyToolkit\n"
                )
            elif "importlib.util" in command[-1]:
                stdout = json.dumps(
                    {
                        "prefix": str(self.root / "conda" / "envs" / "test_env"),
                        "python": "3.11.9",
                        "toolkit_origin": str(
                            self.root / "SpatialBiologyToolkit" / "__init__.py"
                        ),
                    }
                )
        elif command == ["conda", "--version"]:
            stdout = "conda 25.1.0\n"
        elif command[:3] == ["git", "-C", str(self.root)]:
            if command[-2:] == ["rev-parse", "HEAD"]:
                stdout = "abc123\n"
            elif command[-2:] == ["status", "--porcelain"]:
                stdout = ""
        elif command[:3] == ["conda", "env", "remove"]:
            self.exists = False
        elif conda_lock_command == ["lock", "--help"]:
            stdout = "--file -f --platform -p --lockfile\n"
        elif conda_lock_command == ["install", "--help"]:
            stdout = "--name -n\n"
        elif conda_lock_command == ["--version"]:
            stdout = "conda-lock, version 3.0.4\n"
        elif conda_lock_command and conda_lock_command[0] == "install":
            self.exists = True
        elif conda_lock_command and conda_lock_command[0] == "lock":
            destination = Path(command[command.index("--lockfile") + 1])
            destination.write_text("version: 1\nmetadata:\n  platforms: [linux-64]\npackage: []\n")
        elif command[:4] in (
            ["conda", "env", "export", "--name"],
            ["conda", "env", "export", "--prefix"],
        ):
            stdout = yaml.safe_dump(
                {
                    "name": "test_env",
                    "channels": ["conda-forge"],
                    "dependencies": ["python=3.11", "pip"],
                    "prefix": "/ignored",
                },
                sort_keys=False,
            )
        return subprocess.CompletedProcess(command, return_code, stdout, stderr)


class EnvironmentFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "SpatialBiologyToolkit").mkdir()
        directory = self.root / "HPC_env_files" / "test_env"
        directory.mkdir(parents=True)
        (directory / "environment.yml").write_text(
            "name: test_env\nchannels: [conda-forge]\ndependencies:\n  - python=3.11\n  - pip\n",
            encoding="utf-8",
        )
        (directory / "conda-linux-64.lock").write_text(
            "version: 1\n"
            "metadata:\n  platforms: [linux-64]\n"
            "package:\n"
            "  - name: python\n"
            "    version: '3.11'\n"
            "    manager: conda\n"
            "    platform: linux-64\n"
            "    build: h123_0\n"
            "    url: https://conda.anaconda.org/conda-forge/linux-64/python.tar.bz2\n"
            "  - name: pip\n"
            "    version: '24.0'\n"
            "    manager: conda\n"
            "    platform: linux-64\n"
            "    build: pyhd8ed1ab_0\n"
            "    url: https://conda.anaconda.org/conda-forge/noarch/pip.tar.bz2\n",
            encoding="utf-8",
        )
        (directory / "pip-extras.txt").write_text("foo==1.0\n", encoding="utf-8")
        registry = {
            "schema_version": 1,
            "environments": {
                "test": {
                    "conda_name": "test_env",
                    "specification_directory": "HPC_env_files/test_env",
                    "platform": "linux-64",
                    "toolkit_overlay": "editable-no-deps",
                    "smoke_tests": [["python", "-c", "import SpatialBiologyToolkit"]],
                }
            },
            "stage_environments": {"prep": ["test"]},
        }
        self.registry_path = self.root / "HPC_env_files" / "environments.yaml"
        self.registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def manager(self, *, exists=True, drift=False):
        runner = FakeRunner(self.root, exists=exists, drift=drift)
        manager = EnvironmentManager(
            self.root,
            registry_path=self.registry_path,
            runner=runner,
            conda_executable="conda",
            state_root=self.root / "state",
        )
        return manager, runner


class RegistryTests(EnvironmentFixture):
    def test_conda_inventory_keeps_base_and_distinct_prefixes(self):
        root_prefix = self.root / "conda"
        scratch_prefix = root_prefix / "envs" / "scratch"

        def inventory_runner(command, **kwargs):
            payload = {
                "root_prefix": str(root_prefix),
                "platform": "linux-64",
                "envs": [
                    str(root_prefix),
                    str(scratch_prefix),
                    str(scratch_prefix),
                ],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        records = conda_environment_records("conda", runner=inventory_runner)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].name, "base")
        self.assertTrue(records[0].is_base)
        self.assertEqual(records[1].name, "scratch")
        self.assertFalse(records[1].is_base)

    def test_registry_resolves_key_and_fixed_name(self):
        registry = load_environment_registry(self.root, registry_path=self.registry_path)
        self.assertEqual(resolve_environment(registry, "test")[0], "test")
        self.assertEqual(resolve_environment(registry, "TEST_ENV")[0], "test")

    def test_duplicate_fixed_names_are_rejected(self):
        with self.assertRaises(ValidationError):
            EnvironmentRegistry.model_validate(
                {
                    "schema_version": 1,
                    "environments": {
                        "a": {"conda_name": "same", "managed": False},
                        "b": {"conda_name": "SAME", "managed": False},
                    },
                    "stage_environments": {},
                }
            )

    def test_live_stage_registry_uses_central_environment_keys(self):
        central = load_environment_registry(Path(__file__).resolve().parents[1])
        by_name = {stage.name: stage for stage in STAGES}
        for stage, keys in central.stage_environments.items():
            self.assertEqual(by_name[stage].environment_keys, keys)

    def test_joint_analysis_candidate_is_registered_but_not_active(self):
        root = Path(__file__).resolve().parents[1]
        central = load_environment_registry(root)
        definition = central.environments["analysis"]

        self.assertEqual(definition.conda_name, "sbt-analysis")
        self.assertTrue(definition.managed)
        self.assertEqual(associated_stages(central, "analysis"), [])
        self.assertIn("segmentation", " ".join(definition.notes).casefold())
        self.assertIn("biobatchnet", " ".join(definition.notes).casefold())
        self.assertIn("cellcharter", " ".join(definition.notes).casefold())
        self.assertIn("starling", " ".join(definition.notes).casefold())

        conda_requirements = declared_conda_requirements(
            root / "HPC_env_files" / "sbt-analysis" / "environment.yml"
        )
        pip_requirements = declared_pip_requirements(
            root / "HPC_env_files" / "sbt-analysis" / "pip-extras.txt"
        )
        self.assertEqual(conda_requirements["python"], "=3.11")
        self.assertEqual(conda_requirements["numpy"], "=1.26.4")
        self.assertEqual(pip_requirements["torch"].version, "2.9.1")
        self.assertEqual(pip_requirements["cellcharter"].version, "0.3.7")
        self.assertEqual(pip_requirements["biostarling"].version, "0.1.4")
        self.assertEqual(pip_requirements["biobatchnet"].source_type, "vcs")

    def test_legacy_python_environments_support_runtime_type_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        central = load_environment_registry(root)

        for key in ("denoise", "biobatchnet"):
            definition = central.environments[key]
            specification = root / definition.specification_directory
            pip_requirements = declared_pip_requirements(
                specification / "pip-extras.txt"
            )
            smoke_scripts = [
                command[-1]
                for command in definition.smoke_tests
                if len(command) >= 3 and command[:2] == ["python", "-c"]
            ]

            self.assertIn("eval-type-backport", pip_requirements)
            self.assertTrue(
                any(
                    "PipelineConfig" in script and "StageReporter" in script
                    for script in smoke_scripts
                )
            )

    def test_required_stage_environments_report_live_availability_once(self):
        manager, runner = self.manager(exists=False)

        rows = manager.required_for_stages(["prep", "prep", "unmapped"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].key, "test")
        self.assertEqual(rows[0].conda_name, "test_env")
        self.assertFalse(rows[0].exists)
        self.assertEqual(rows[0].stages, ["prep"])
        self.assertEqual(
            sum(call[:4] == ["conda", "env", "list", "--json"] for call in runner.calls),
            1,
        )

    def test_stages_without_environment_mapping_do_not_require_conda(self):
        manager = EnvironmentManager(
            self.root,
            registry_path=self.registry_path,
            runner=FakeRunner(self.root),
            conda_executable=None,
            state_root=self.root / "state",
        )
        manager.conda = None

        self.assertEqual(manager.required_for_stages(["debug"]), [])


class SpecificationTests(EnvironmentFixture):
    def test_valid_specification(self):
        manager, _ = self.manager()
        report = manager.validate("test")
        self.assertTrue(report.valid)
        self.assertEqual(report.paths.lockfile.name, "conda-linux-64.lock")

    def test_name_mismatch_is_invalid(self):
        path = self.root / "HPC_env_files" / "test_env" / "environment.yml"
        path.write_text("name: wrong\nchannels: []\ndependencies: []\n")
        manager, _ = self.manager()
        report = manager.validate("test")
        self.assertFalse(report.valid)
        self.assertIn("name_mismatch", {item.code for item in report.issues})

    def test_embedded_pip_and_toolkit_extra_are_rejected(self):
        directory = self.root / "HPC_env_files" / "test_env"
        (directory / "environment.yml").write_text(
            "name: test_env\nchannels: []\ndependencies:\n  - pip:\n      - x==1\n"
        )
        (directory / "pip-extras.txt").write_text(
            "SpatialBiologyToolkit @ file:///tmp/imcanalysis\n"
        )
        manager, _ = self.manager()
        codes = {item.code for item in manager.validate("test").issues}
        self.assertIn("embedded_pip_dependencies", codes)
        self.assertIn("toolkit_in_pip_extras", codes)

    def test_constraint_and_pip_parsing(self):
        path = self.root / "HPC_env_files" / "test_env" / "pip-extras.txt"
        records = declared_pip_requirements(path)
        self.assertEqual(records["foo"].version, "1.0")
        self.assertTrue(satisfies_constraint("3.11.9", ">=3.10,<3.12"))
        self.assertFalse(satisfies_constraint("3.9", ">=3.10"))


class ComparisonAndSyncTests(EnvironmentFixture):
    def test_doctor_is_lightweight_with_mocked_commands(self):
        manager, runner = self.manager()
        report = manager.doctor()
        self.assertFalse(report.healthy)  # the minimal fixture intentionally omits most stages
        self.assertIn("pip_through_conda_run", {check.name for check in report.checks})
        self.assertIn(
            ["conda", "run", "-n", "base", "conda-lock", "--version"],
            runner.calls,
        )
        self.assertFalse(any("scanpy" in " ".join(call) for call in runner.calls))

    def test_exact_comparison_returns_zero(self):
        manager, _ = self.manager()
        result = manager.compare("test")
        self.assertEqual(result.result, "clean")
        self.assertEqual(result.exit_code, 0)

    def test_drift_comparison_returns_one(self):
        manager, _ = self.manager(drift=True)
        result = manager.compare("test")
        self.assertEqual(result.result, "drift")
        self.assertEqual(result.exit_code, 1)
        self.assertTrue(any(item.package == "python" for item in result.drift))

    def test_missing_environment_comparison_returns_two(self):
        manager, _ = self.manager(exists=False)
        result = manager.compare("test")
        self.assertEqual(result.result, "missing")
        self.assertEqual(result.exit_code, 2)

    def test_absent_environment_has_creation_plan_and_dry_run_is_safe(self):
        manager, runner = self.manager(exists=False)
        plan = manager.sync("test", dry_run=True)
        self.assertFalse(plan.exists)
        self.assertIn("Create fixed environment", plan.actions[0])
        self.assertFalse(any("conda-lock install" in " ".join(call) for call in runner.calls))

    def test_sync_installs_lock_extras_overlay_and_tests(self):
        manager, runner = self.manager(exists=False)
        manager.sync("test")
        flattened = [" ".join(call) for call in runner.calls]
        self.assertTrue(
            any(
                "conda run -n base conda-lock install --name test_env" in call
                for call in flattened
            )
        )
        self.assertTrue(any("pip install -r" in call for call in flattened))
        overlay = next(call for call in runner.calls if "-e" in call and "--no-deps" in call)
        self.assertEqual(overlay[-1], "--no-deps")
        self.assertTrue(list((self.root / "state" / "environments" / "test_env").glob("*.json")))

    def test_drift_requires_explicit_recreation(self):
        manager, _ = self.manager(drift=True)
        with self.assertRaisesRegex(RuntimeError, "--recreate"):
            manager.sync("test")


class CaptureAndProvenanceTests(EnvironmentFixture):
    def test_capture_is_deterministic_and_excludes_toolkit(self):
        manager, _ = self.manager()
        plan = manager.capture("test")
        self.assertIn("python=3.11", plan.environment_yml)
        self.assertNotIn("prefix", plan.environment_yml)
        self.assertEqual(plan.pip_extras, "foo==1.0\n")
        self.assertIsNotNone(plan.excluded_toolkit)
        self.assertTrue((plan.candidate_directory / "conda-linux-64.lock").is_file())

    def test_discover_capture_targets_includes_unregistered_prefixes(self):
        manager, _ = self.manager()
        records = [
            CondaEnvironmentRecord(
                name="base",
                prefix=self.root / "conda",
                platform="linux-64",
                is_base=True,
            ),
            CondaEnvironmentRecord(
                name="test_env",
                prefix=self.root / "conda" / "envs" / "test_env",
                platform="linux-64",
            ),
            CondaEnvironmentRecord(
                name="scratch",
                prefix=self.root / "conda" / "envs" / "scratch",
                platform="linux-64",
            ),
            CondaEnvironmentRecord(
                name="scratch",
                prefix=self.root / "alternate" / "envs" / "scratch",
                platform="linux-64",
            ),
        ]
        manager._environment_records = Mock(return_value=records)

        targets = manager.discover_capture_targets()

        self.assertEqual(len(targets), 4)
        registered = next(item for item in targets if item.registered)
        self.assertEqual(registered.environment_key, "test")
        self.assertEqual(registered.conda_name, "test_env")
        unregistered = [item for item in targets if not item.registered]
        self.assertIn("conda:base", {item.environment_key for item in unregistered})
        scratch = [item for item in unregistered if item.conda_name == "scratch"]
        self.assertEqual(len(scratch), 2)
        self.assertEqual(len({item.environment_key for item in scratch}), 2)
        self.assertEqual(len({item.capture_directory_name for item in scratch}), 2)

    def test_capture_unregistered_target_uses_exact_prefix(self):
        manager, runner = self.manager()
        prefix = self.root / "conda" / "envs" / "test_env"
        target = EnvironmentCaptureTarget(
            environment_key="conda:test_env",
            conda_name="test_env",
            conda_prefix=prefix,
            platform="linux-64",
            registered=False,
            capture_directory_name="test_env",
        )

        plan = manager.capture_target(target, accept_vcs=True)

        self.assertFalse(plan.registered)
        self.assertFalse(plan.managed)
        self.assertEqual(plan.conda_prefix, prefix)
        self.assertTrue(
            any(call[:4] == ["conda", "list", "--prefix", str(prefix)] for call in runner.calls)
        )
        self.assertFalse(
            any(call[:4] == ["conda", "list", "--name", "test_env"] for call in runner.calls)
        )

    def test_capture_retains_conda_inventory_when_python_is_unavailable(self):
        manager, runner = self.manager()
        prefix = self.root / "conda" / "envs" / "test_env"
        target = EnvironmentCaptureTarget(
            environment_key="conda:test_env",
            conda_name="test_env",
            conda_prefix=prefix,
            platform="linux-64",
            registered=False,
            capture_directory_name="test_env",
        )

        def no_python(command, **kwargs):
            normalized = [str(item) for item in command]
            if (
                normalized[:3] == ["conda", "run", "--prefix"]
                and normalized[4] == "python"
            ):
                return subprocess.CompletedProcess(
                    normalized, 127, "", "python is unavailable"
                )
            return runner(command, **kwargs)

        manager.runner = no_python
        plan = manager.capture_target(target)
        snapshot = json.loads(
            (plan.candidate_directory / "environment.snapshot.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(snapshot["conda_packages"])
        self.assertIsNone(snapshot["python_version"])
        self.assertTrue(
            any("Python/pip inspection unavailable" in item for item in plan.review_requirements)
        )

    def test_external_capture_creates_observational_bundle_but_refuses_write(self):
        registry = yaml.safe_load(self.registry_path.read_text(encoding="utf-8"))
        registry["environments"]["test"]["managed"] = False
        registry["environments"]["test"].pop("specification_directory")
        self.registry_path.write_text(
            yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
        )
        manager, _ = self.manager()

        plan = manager.capture("test")

        self.assertFalse(plan.managed)
        self.assertTrue((plan.candidate_directory / "environment.yml").is_file())
        self.assertTrue((plan.candidate_directory / "pip-extras.txt").is_file())
        self.assertTrue(
            (plan.candidate_directory / "environment.snapshot.json").is_file()
        )
        self.assertTrue((plan.candidate_directory / "capture-plan.json").is_file())
        self.assertTrue((plan.candidate_directory / "conda-linux-64.lock").is_file())
        self.assertIsNone(plan.lock_generation_error)
        self.assertEqual(
            plan.differences["environment.yml"],
            "no repository specification to compare",
        )
        with self.assertRaisesRegex(ValueError, "without --write"):
            manager.capture("test", write=True)

    def test_external_capture_retains_inventory_when_candidate_lock_fails(self):
        registry = yaml.safe_load(self.registry_path.read_text(encoding="utf-8"))
        registry["environments"]["test"]["managed"] = False
        registry["environments"]["test"].pop("specification_directory")
        self.registry_path.write_text(
            yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
        )
        manager, _ = self.manager()

        def fail_lock(*args, **kwargs):
            raise RuntimeError("solver conflict")

        manager._generate_lock = fail_lock  # type: ignore[method-assign]
        plan = manager.capture("test")

        self.assertIsNone(plan.lockfile)
        self.assertEqual(plan.lock_generation_error, "solver conflict")
        self.assertTrue(
            (plan.candidate_directory / "environment.snapshot.json").is_file()
        )
        self.assertIn("generation failed", plan.differences["conda-linux-64.lock"])

    def test_lock_check_does_not_replace_committed_lock(self):
        manager, _ = self.manager()
        lock = self.root / "HPC_env_files" / "test_env" / "conda-linux-64.lock"
        before = lock.read_bytes()
        current, _ = manager.lock("test", check=True)
        self.assertFalse(current)
        self.assertEqual(lock.read_bytes(), before)

    def test_specification_snapshot_is_a_copy_with_hashes(self):
        output = self.root / "output"
        reference = snapshot_stage_environment_specifications(
            stage="prep", output_directory=output, repository_root=self.root
        )
        self.assertIsNotNone(reference)
        copied = output / "environment" / "environment.yml"
        source = self.root / "HPC_env_files" / "test_env" / "environment.yml"
        self.assertEqual(copied.read_bytes(), source.read_bytes())
        source.write_text("changed\n")
        self.assertNotEqual(copied.read_bytes(), source.read_bytes())
        manifest = yaml.safe_load(
            (output / "environment" / "environment_manifest.yaml").read_text()
        )
        self.assertEqual(len(manifest["specification"]["environment_yml"]["sha256"]), 64)
        self.assertEqual(discover_generated_files(output), [])


class EnvironmentCliTests(unittest.TestCase):
    def test_env_help_is_lightweight(self):
        result = CliRunner().invoke(app, ["env", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("validate-spec", result.output)
        self.assertIn("capture", result.output)

    def test_env_list_supports_json(self):
        result = CliRunner().invoke(app, ["env", "list", "--format", "json"])
        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertIn("segmentation", {item["key"] for item in payload})

    def test_capture_all_continues_after_failure_and_summarizes(self):
        manager = Mock()
        targets = [
            EnvironmentCaptureTarget(
                environment_key="conda:missing",
                conda_name="missing",
                conda_prefix=Path("/conda/envs/missing"),
                platform="linux-64",
                capture_directory_name="missing",
            ),
            EnvironmentCaptureTarget(
                environment_key="conda:external",
                conda_name="external_env",
                conda_prefix=Path("/conda/envs/external_env"),
                platform="linux-64",
                capture_directory_name="external_env",
            ),
        ]
        manager.discover_capture_targets.return_value = targets
        plan = CapturePlan(
            environment_key="conda:external",
            conda_name="external_env",
            managed=False,
            registered=False,
            conda_prefix=Path("/conda/envs/external_env"),
            candidate_directory=Path("/capture/external_env"),
            environment_yml="name: external_env\n",
            pip_extras="",
        )
        manager.capture_target.side_effect = [RuntimeError("environment is absent"), plan]

        with patch(
            "SpatialBiologyToolkit.cli.main._env_manager", return_value=manager
        ):
            result = CliRunner().invoke(
                app,
                [
                    "env",
                    "capture",
                    "--all",
                    "--dry-run",
                    "--verbose",
                    "--accept-vcs",
                ],
            )

        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("Environment: conda:external (external_env)", result.output)
        self.assertIn("Registry: unregistered", result.output)
        self.assertIn("Capture summary: 1 succeeded, 1 failed.", result.output)
        self.assertIn("conda:missing: environment is absent", result.output)
        manager.discover_capture_targets.assert_called_once_with()
        self.assertEqual(manager.capture_target.call_count, 2)
        for call in manager.capture_target.call_args_list:
            self.assertTrue(call.kwargs["accept_vcs"])
            self.assertTrue(call.kwargs["verbose"])

    def test_capture_all_rejects_repository_writes(self):
        result = CliRunner().invoke(
            app, ["env", "capture", "--all", "--write"]
        )

        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("--all is observational only", result.output)


if __name__ == "__main__":
    unittest.main()
