from types import SimpleNamespace

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli import main as cli


def test_annotation_launcher_arguments_and_failure(monkeypatch, tmp_path):
    source = tmp_path / "input.h5ad"
    source.touch()
    commands = []
    monkeypatch.setattr(
        cli,
        "_annotation_gui_command",
        lambda: ["python", "-m", "SpatialBiologyToolkit.annotation"],
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kw: (
            commands.append(command) or SimpleNamespace(returncode=0)
        ),
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "gui",
            "annotate",
            "--anndata",
            str(source),
            "--basis",
            "X_pca",
            "--color",
            "CD3",
            "--source-obs",
            "leiden",
        ],
    )
    assert result.exit_code == 0, result.output
    assert commands[0][-4:] == ["--color", "CD3", "--source-obs", "leiden"]
    assert commands[0][commands[0].index("--basis") + 1] == "X_pca"
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=3)
    )
    result = CliRunner().invoke(cli.app, ["gui", "annotate", "--anndata", str(source)])
    assert result.exit_code == 3


def test_annotation_cli_help_and_missing_input():
    runner = CliRunner()
    assert runner.invoke(cli.app, ["gui", "annotate", "--help"]).exit_code == 0
    result = runner.invoke(
        cli.app, ["gui", "annotate", "--anndata", "does-not-exist.h5ad"]
    )
    assert result.exit_code != 0


def test_recipe_cli_launches_headless_and_requires_new_output(monkeypatch, tmp_path):
    source = tmp_path / "input.h5ad"
    recipe = tmp_path / "recipe.json"
    output = tmp_path / "output.h5ad"
    source.touch()
    recipe.write_text("{}")
    calls = []

    def command(*, headless=False):
        assert headless
        return ["python", "-m", "SpatialBiologyToolkit.annotation"]

    monkeypatch.setattr(cli, "_annotation_gui_command", command)
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0),
    )
    args = ["gui", "annotate", "--anndata", str(source), "--recipe", str(recipe)]
    runner = CliRunner()
    assert runner.invoke(cli.app, args).exit_code != 0
    result = runner.invoke(cli.app, [*args, "--output", str(output), "--overwrite-obs"])
    assert result.exit_code == 0, result.output
    assert calls[0][-5:] == [
        "--recipe", str(recipe.resolve()), "--output", str(output.resolve()),
        "--overwrite-obs",
    ]
    output.touch()
    result = runner.invoke(cli.app, [*args, "--output", str(output)])
    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output
    assert len(calls) == 1


def test_headless_launcher_does_not_require_qt(monkeypatch):
    monkeypatch.setattr(
        cli.importlib.util, "find_spec",
        lambda name: object() if name in {"anndata", "numpy", "pandas", "matplotlib"}
        else None,
    )
    assert cli._annotation_gui_command(headless=True)[1:] == [
        "-m", "SpatialBiologyToolkit.annotation"
    ]
