"""Exercise shell launch boundaries without scientific environments or SLURM."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if BASH is None and os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    BASH = str(candidate) if candidate.is_file() else None


def clean_environment():
    return {key: value for key, value in os.environ.items() if not key.startswith("SBT_")}


@pytest.mark.skipif(BASH is None, reason="Bash is unavailable")
@pytest.mark.parametrize("fail_stage", ["", "denoising"])
def test_runner_order_logs_and_failure(tmp_path, fail_stage):
    project = tmp_path / "project with spaces"
    project.mkdir()
    config = project / "config trial.yaml"
    config.write_text("{}", encoding="utf-8")
    conda_sh = tmp_path / "fake conda.sh"
    conda_sh.write_text(
        '''conda() {
    export TEST_ACTIVE_ENV="$2"
    export CONDA_PREFIX="/fake/$2"
}
python() {
    test -f "$SBT_CONFIG" || return 99
    printf 'stage:%s env:%s\\n' "$2" "$TEST_ACTIVE_ENV"
    echo "stderr:$2" >&2
    if [[ "$2" == "SpatialBiologyToolkit.scripts.$TEST_FAIL_STAGE" ]]; then
        return 17
    fi
}
''', encoding="utf-8", newline="\n",
    )
    env = clean_environment()
    env.update(
        SBT_CONDA_SH=conda_sh.as_posix(),
        SBT_CONFIG=config.as_posix(),
        SBT_LOG_DIR=(tmp_path / "saved logs").as_posix(),
        SBT_CONDA_ENV_ANALYSIS="custom-analysis",
        TEST_FAIL_STAGE=fail_stage,
    )
    result = subprocess.run(
        [BASH, (ROOT / "run_local.sh").as_posix(), project.as_posix()],
        env=env, capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == (17 if fail_stage else 0), result.stderr
    logs = list((tmp_path / "saved logs").glob("run-*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    stages = [line for line in text.splitlines() if line.startswith("stage:")]
    expected = [
        ("preprocess", "custom-analysis"),
        ("denoising", "sbt-tensorflow"),
        ("preprocess_dna", "custom-analysis"),
        ("cellpose_sam", "sbt-cellpose-sam"),
        ("segmentation_nimbus", "custom-analysis"),
    ]
    if fail_stage:
        expected = expected[:2]
    assert stages == [
        f"stage:SpatialBiologyToolkit.scripts.{stage} env:{env_name}"
        for stage, env_name in expected
    ]
    assert "stderr:SpatialBiologyToolkit.scripts.denoising" in text
    assert "Saving console output to" in result.stdout
    assert config.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("configured", [False, True])
def test_scportrait_converter_receives_paths_and_preserves_exit(tmp_path, configured):
    wrapper = (ROOT / "SLURM_scripts/job_scport.sh").read_text(encoding="utf-8")
    code = wrapper.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    converter = tmp_path / "external converter.py"
    converter.write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(23)\n",
        encoding="utf-8",
    )
    env = clean_environment()
    env.update(PYTHONPATH=str(ROOT), SBT_SCPORTRAIT_CONVERTER=str(converter))
    if configured:
        config = tmp_path / "custom config.yaml"
        config.write_text(json.dumps({
            "general": {"denoised_images_folder": "channel images", "masks_folder": "cell masks"},
            "scportrait": {"projects_root": str(tmp_path / "portrait output")},
        }), encoding="utf-8")
        original = config.read_bytes()
        env["SBT_CONFIG"] = str(config)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 23, result.stderr
    assert json.loads(result.stdout) == [
        "--channels-dir", "channel images" if configured else "processed",
        "--mask-dir", "cell masks" if configured else "masks",
        "--projects-root", str(tmp_path / "portrait output") if configured else "scPortrait",
        "--overwrite", "--mask-expand-px", "0", "--debug",
    ]
    if configured:
        assert config.read_bytes() == original
        config.unlink()
        missing = subprocess.run(
            [sys.executable, "-c", code], cwd=tmp_path, env=env,
            capture_output=True, encoding="utf-8", timeout=30,
        )
        assert missing.returncode != 0
        assert "Configuration file not found" in missing.stderr


@pytest.mark.skipif(BASH is None, reason="Bash is unavailable")
def test_legacy_install_repeat_and_uninstall_at_custom_location(tmp_path):
    checkout = tmp_path / "toolkit checkout"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".imc_config").write_text("# Keep my settings\n", encoding="utf-8")
    (home / ".profile").write_text("# Unrelated settings\n", encoding="utf-8")
    for directory in ("install", "Bash_scripts", "SLURM_scripts"):
        (checkout / directory).mkdir(parents=True)
    for name in ("setup.sh", "common.sh", "uninstall.sh"):
        shutil.copyfile(ROOT / "install" / name, checkout / "install" / name)
    shutil.copyfile(ROOT / "Bash_scripts/cds", checkout / "Bash_scripts/cds")
    (checkout / "SLURM_scripts/job_example.sh").write_text("# example\n", encoding="utf-8")
    data = tmp_path / "data"
    (data / "Project One").mkdir(parents=True)
    env = clean_environment()
    env.update(HOME=home.as_posix(), SBT_TOOLKIT_ROOT=checkout.as_posix(), SBT_DATA_ROOT=data.as_posix())
    for _ in range(2):
        result = subprocess.run(
            [BASH, (checkout / "install/setup.sh").as_posix()],
            env=env, input="", capture_output=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0, result.stderr
    profile = (home / ".profile").read_text(encoding="utf-8")
    assert profile.count("export SBT_TOOLKIT_ROOT=") == 1
    assert profile.count("export PATH=") == 1
    use_alias = subprocess.run(
        [BASH, "-c", 'shopt -s expand_aliases; source "$HOME/.bashrc"; eval "cds proj"; pwd'],
        env=env, capture_output=True, encoding="utf-8", timeout=30,
    )
    assert use_alias.returncode == 0, use_alias.stderr
    assert use_alias.stdout.strip().endswith("/data/Project One")
    uninstall = subprocess.run(
        [BASH, (checkout / "install/uninstall.sh").as_posix()],
        env=env, input="n\n", capture_output=True, encoding="utf-8", timeout=30,
    )
    assert uninstall.returncode == 0, uninstall.stderr
    assert (home / ".profile").read_text(encoding="utf-8") == "# Unrelated settings\n"
    assert "cds=" not in (home / ".bashrc").read_text(encoding="utf-8")
    assert (home / ".imc_config").read_text(encoding="utf-8") == "# Keep my settings\n"
