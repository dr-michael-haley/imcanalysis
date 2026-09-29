"""Exercise console startup without running packed interpreters or activation scripts."""

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location(
    "portable_startup", Path(__file__).parents[1] / "start_naparisbt.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "Tissue café & 20% folder"
    (root / "Scripts").mkdir(parents=True)
    (root / "Lib").mkdir()
    (root / "python.exe").touch()
    (root / "Scripts/conda-unpack-script.py").touch()
    monkeypatch.setattr(launcher, "__file__", str(root / "start_naparisbt.py"))
    monkeypatch.setattr(launcher, "setup_lock", lambda root: nullcontext())
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    return root


def test_first_launch_and_repeat_use_same_location_record(runtime, monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main([]) == 0
    marker = runtime / ".naparisbt-location.txt"
    assert marker.read_text(encoding="utf-8") == str(runtime)
    unpack, app = run.call_args_list
    assert unpack.args[0] == [str(runtime / "python.exe"), "-s",
                              str(runtime / "Scripts/conda-unpack-script.py")]
    assert unpack.kwargs["check"] is True
    assert app.args[0] == [str(runtime / "python.exe"), "-s", "-m",
                           "SpatialBiologyToolkit.napari_sbt", "--welcome"]
    assert app.kwargs["cwd"] == runtime
    assert "shell" not in app.kwargs
    # The C# launcher writes UTF-8 too; tolerate a BOM and case changes.
    marker.write_text(str(runtime).upper(), encoding="utf-8-sig")
    run.reset_mock()
    assert launcher.main([]) == 0
    assert run.call_count == 1


def test_failed_setup_never_marks_ready_and_can_retry(runtime, monkeypatch):
    run = Mock(side_effect=subprocess.CalledProcessError(11, "unpack"))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main([]) == 1
    assert not (runtime / ".naparisbt-location.txt").exists()
    assert not list(runtime.glob(".naparisbt-write-*"))
    run.side_effect = None
    run.return_value = subprocess.CompletedProcess([], 0)
    assert launcher.main(["--prepare-only"]) == 0
    assert (runtime / ".naparisbt-location.txt").is_file()
    assert run.call_count == 2  # No application launch for --prepare-only.


@pytest.mark.parametrize("problem", ["moved", "legacy", "missing_unpack", "missing_python"])
def test_unsafe_or_incomplete_setup_does_not_launch(runtime, monkeypatch, problem):
    if problem == "moved":
        (runtime / ".naparisbt-location.txt").write_text("C:\\old-folder", encoding="utf-8")
    elif problem == "legacy":
        (runtime / ".naparisbt-ready").touch()
    elif problem == "missing_unpack":
        (runtime / "Scripts/conda-unpack-script.py").unlink()
    else:
        (runtime / "python.exe").unlink()
    run = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main([]) == 1
    run.assert_not_called()


def test_environment_isolated_without_overwriting_custom_certificates(runtime, monkeypatch):
    for key in ("PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH", "CONDA_PREFIX"):
        monkeypatch.setenv(key, "unrelated-environment")
    monkeypatch.setenv("SSL_CERT_FILE", "approved-company-certificates.pem")
    env = launcher.runtime_environment(runtime)
    assert not {"PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH"} & env.keys()
    assert env["CONDA_PREFIX"] == str(runtime)
    assert env["PATH"].split(os.pathsep)[0] == str(runtime)
    assert Path(env["NUMBA_CACHE_DIR"]).is_dir()
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["SSL_CERT_FILE"] == "approved-company-certificates.pem"
    assert os.environ["PYTHONHOME"] == "unrelated-environment"


def test_application_exit_code_is_preserved(runtime, monkeypatch):
    (runtime / ".naparisbt-location.txt").write_text(str(runtime), encoding="utf-8")
    monkeypatch.setattr(launcher.subprocess, "run", Mock(
        return_value=subprocess.CompletedProcess([], 42)))
    assert launcher.main([]) == 42


@pytest.mark.skipif(os.name != "nt", reason="Windows named mutex")
def test_setup_lock_blocks_other_threads_and_releases_after_failure(tmp_path):
    def try_lock():
        with launcher.setup_lock(tmp_path):
            return True

    with pytest.raises(ValueError, match="simulated failure"):
        with launcher.setup_lock(tmp_path), ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="Another launcher"):
                pool.submit(try_lock).result(timeout=5)
            raise ValueError("simulated failure")
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(try_lock).result(timeout=5)
