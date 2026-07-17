#!/usr/bin/env python3
"""Generate Sphinx reference pages from repository-owned metadata.

The generated output is committed so it can be read on GitHub as well as on
Read the Docs. Run with ``--check`` in CI or before committing to detect drift.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "docs" / "source"
GITHUB_ROOT = "https://github.com/dr-michael-haley/imcanalysis/blob/main"
GENERATED_DIRS = (
    Path("pipeline/stages"),
    Path("reference/configuration/sections"),
    Path("reference/api"),
)
GENERATED_FILES = (
    Path("reference/configuration/index.md"),
    Path("_static/config.schema.json"),
)
METADATA_PATTERN = re.compile(
    r"^#@(DESC|IN|OUT|ENV|MODULE|CONFIG):\s*(.*)$"
)


@dataclass
class StageDoc:
    """Documentation metadata for one configured SLURM stage."""

    alias: str
    script: str
    description: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    environments: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    config: list[str] = field(default_factory=list)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _reset_generated_dir(path: Path) -> None:
    resolved = path.resolve()
    source_root = path.parents[2].resolve()
    if source_root not in resolved.parents:
        raise ValueError(f"Refusing to replace generated directory outside {source_root}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_pipeline_config(config_path: Path) -> list[tuple[str, str]]:
    """Return ordered ``(alias, wrapper)`` mappings from ``pipeline.conf``."""
    mappings: list[tuple[str, str]] = []
    aliases: set[str] = set()
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{config_path}:{line_number}: expected alias=script")
        alias, script = (part.strip() for part in line.split("=", 1))
        if not alias or not script:
            raise ValueError(f"{config_path}:{line_number}: alias and script are required")
        if alias in aliases:
            raise ValueError(f"{config_path}:{line_number}: duplicate alias {alias!r}")
        aliases.add(alias)
        mappings.append((alias, script))
    return mappings


def parse_stage_metadata(alias: str, script_name: str, script_path: Path) -> StageDoc:
    """Parse ``#@`` documentation records from one SLURM wrapper."""
    if not script_path.is_file():
        raise FileNotFoundError(f"Stage {alias!r} refers to missing wrapper: {script_path}")

    stage = StageDoc(alias=alias, script=script_name)
    destinations = {
        "IN": stage.inputs,
        "OUT": stage.outputs,
        "ENV": stage.environments,
        "MODULE": stage.modules,
        "CONFIG": stage.config,
    }
    for line in script_path.read_text(encoding="utf-8").splitlines():
        match = METADATA_PATTERN.match(line.strip())
        if not match:
            continue
        key, value = match.groups()
        if key == "DESC":
            if stage.description:
                raise ValueError(f"Stage {alias!r} has more than one #@DESC record")
            stage.description = value
        else:
            destinations[key].append(value)

    if not stage.description:
        raise ValueError(f"Stage {alias!r} is missing a #@DESC record in {script_path}")
    return stage


def load_stage_docs(repo_root: Path = REPO_ROOT) -> list[StageDoc]:
    """Load stages from the Python registry and validate the legacy Bash mirror."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from SpatialBiologyToolkit.pipeline.registry import STAGES

    slurm_dir = repo_root / "SLURM_scripts"
    registry_mappings = [
        (stage.name, Path(stage.slurm_script).name)
        for stage in STAGES
    ]
    legacy_mappings = parse_pipeline_config(slurm_dir / "pipeline.conf")
    if registry_mappings != legacy_mappings:
        raise ValueError(
            "SLURM_scripts/pipeline.conf has drifted from the authoritative "
            "SpatialBiologyToolkit.pipeline.registry stage mappings."
        )
    return [
        parse_stage_metadata(alias, script, slurm_dir / script)
        for alias, script in registry_mappings
    ]


def _bullet_section(title: str, values: Iterable[str]) -> list[str]:
    values = list(values)
    if not values:
        return []
    lines = [f"## {title}", ""]
    lines.extend(f"- {value}" for value in values)
    lines.append("")
    return lines


def render_stage_page(stage: StageDoc) -> str:
    """Render one stage as a focused Markdown reference page."""
    source_url = f"{GITHUB_ROOT}/SLURM_scripts/{stage.script}"
    lines = [
        "<!-- Generated by docs/tools/generate_docs.py; do not edit directly. -->",
        "",
        f"# `{stage.alias}`",
        "",
        stage.description,
        "",
        f"- Wrapper: [`{stage.script}`]({source_url})",
    ]
    if stage.environments:
        label = "Conda environment" if len(stage.environments) == 1 else "Conda environments"
        lines.append(f"- {label}: `{', '.join(stage.environments)}`")
    if stage.modules:
        label = "Python module" if len(stage.modules) == 1 else "Python modules"
        lines.append(f"- {label}: `{', '.join(stage.modules)}`")
    lines.append("")
    lines.extend(_bullet_section("Inputs", stage.inputs))
    lines.extend(_bullet_section("Outputs", stage.outputs))
    lines.extend(_bullet_section("Configuration", stage.config))
    lines.extend(
        [
            "## Run",
            "",
            "From the dataset directory containing `config.yaml`:",
            "",
            "```bash",
            f"pl {stage.alias}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_stage_index(stages: list[StageDoc]) -> str:
    """Render the compact SLURM stage table and navigation tree."""
    lines = [
        "<!-- Generated by docs/tools/generate_docs.py; do not edit directly. -->",
        "",
        "# SLURM stage reference",
        "",
        "This reference is generated from the typed Python stage registry and the",
        "`#@` metadata in each job wrapper. The legacy `pipeline.conf` mirror is",
        "validated for compatibility. See the [pipeline workflow](../workflow.md)",
        "for ordering and usage guidance.",
        "",
        "| Alias | Purpose | Environment | Config sections |",
        "|---|---|---|---|",
    ]
    for stage in stages:
        environment = ", ".join(stage.environments) or "-"
        config = "<br>".join(stage.config) or "-"
        lines.append(
            f"| [`{stage.alias}`]({stage.alias}.md) | "
            f"{_markdown_cell(stage.description)} | "
            f"`{_markdown_cell(environment)}` | {_markdown_cell(config)} |"
        )
    lines.extend(["", "```{toctree}", ":maxdepth: 1", ":hidden:", ""])
    lines.extend(stage.alias for stage in stages)
    lines.extend(["```", ""])
    return "\n".join(lines)


def generate_slurm_docs(repo_root: Path, source_dir: Path) -> list[StageDoc]:
    """Generate the SLURM stage index and one page per configured alias."""
    stages = load_stage_docs(repo_root)
    destination = source_dir / "pipeline" / "stages"
    _reset_generated_dir(destination)
    (destination / "index.md").write_text(render_stage_index(stages), encoding="utf-8")
    for stage in stages:
        (destination / f"{stage.alias}.md").write_text(
            render_stage_page(stage), encoding="utf-8"
        )
    return stages


def generate_config_reference(repo_root: Path, source_dir: Path) -> int:
    """Generate table-based config pages plus the complete JSON Schema."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from SpatialBiologyToolkit.config import DEFAULT_CONFIG_CLASSES, write_config_docs
    from SpatialBiologyToolkit.config.schema import write_json_schema

    destination = source_dir / "reference" / "configuration" / "sections"
    _reset_generated_dir(destination)
    written = write_config_docs(destination, layout="table")
    marker = "<!-- Generated by docs/tools/generate_docs.py; do not edit directly. -->\n\n"
    for path in written:
        path.write_text(marker + path.read_text(encoding="utf-8"), encoding="utf-8")

    index_lines = [
        "<!-- Generated by docs/tools/generate_docs.py; do not edit directly. -->",
        "",
        "# Configuration reference",
        "",
        "These compact tables are generated from the Pydantic field metadata used",
        "to validate `config.yaml`. For concepts and examples, start with",
        "[using configuration](usage.md).",
        "",
        "The complete [JSON Schema](../../_static/config.schema.json) is also available",
        "for editors and other tooling.",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        ":caption: Configuration sections",
        "",
    ]
    index_lines.extend(f"sections/{section}" for section in DEFAULT_CONFIG_CLASSES)
    index_lines.extend(["```", ""])
    index_path = source_dir / "reference" / "configuration" / "index.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    write_json_schema(source_dir / "_static" / "config.schema.json")
    return len(written)


def _api_modules(repo_root: Path) -> list[tuple[str, str]]:
    package_dir = repo_root / "SpatialBiologyToolkit"
    modules = [
        (path.stem, f"SpatialBiologyToolkit.{path.stem}")
        for path in sorted(package_dir.glob("*.py"))
        if path.name != "__init__.py"
    ]
    modules.extend(
        (f"config_{path.stem}", f"SpatialBiologyToolkit.config.{path.stem}")
        for path in sorted((package_dir / "config").glob("*.py"))
        if path.name != "__init__.py"
    )
    modules.extend(
        (f"reporting_{path.stem}", f"SpatialBiologyToolkit.reporting.{path.stem}")
        for path in sorted((package_dir / "reporting").glob("*.py"))
        if path.name not in {"__init__.py", "events.py"}
    )
    modules.extend(
        (f"environments_{path.stem}", f"SpatialBiologyToolkit.environments.{path.stem}")
        for path in sorted((package_dir / "environments").glob("*.py"))
        if path.name != "__init__.py"
    )
    return modules


def generate_api_reference(repo_root: Path, source_dir: Path) -> int:
    """Generate one autodoc page per public Python module."""
    destination = source_dir / "reference" / "api"
    _reset_generated_dir(destination)
    modules = _api_modules(repo_root)

    index_lines = [
        "Python API",
        "==========",
        "",
        "This reference is split by module so the site navigation stays compact.",
        "",
        ".. automodule:: SpatialBiologyToolkit",
        "   :members:",
        "   :show-inheritance:",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    index_lines.extend(f"   {filename}" for filename, _ in modules)
    index_lines.append("")
    (destination / "index.rst").write_text("\n".join(index_lines), encoding="utf-8")

    for filename, module_name in modules:
        display_name = module_name.rsplit(".", 1)[-1]
        title_bar = "=" * len(display_name)
        content = "\n".join(
            [
                ".. Generated by docs/tools/generate_docs.py; do not edit directly.",
                "",
                display_name,
                title_bar,
                "",
                f".. automodule:: {module_name}",
                "   :members:",
                "   :undoc-members:",
                "   :show-inheritance:",
                "",
            ]
        )
        (destination / f"{filename}.rst").write_text(content, encoding="utf-8")
    return len(modules)


def generate_all(repo_root: Path = REPO_ROOT, source_dir: Path = DEFAULT_SOURCE_DIR) -> str:
    """Generate every repository-derived documentation artifact."""
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "_static").mkdir(parents=True, exist_ok=True)
    stages = generate_slurm_docs(repo_root, source_dir)
    config_count = generate_config_reference(repo_root, source_dir)
    module_count = generate_api_reference(repo_root, source_dir)
    return (
        f"Generated {len(stages)} SLURM stages, {config_count} config sections, "
        f"and {module_count} API modules."
    )


def _directory_diff(expected: Path, actual: Path) -> list[str]:
    if not actual.is_dir():
        return [f"missing generated directory: {actual}"]
    comparison = filecmp.dircmp(expected, actual)
    differences = [
        *(f"missing generated file: {actual / name}" for name in comparison.left_only),
        *(f"unexpected generated file: {actual / name}" for name in comparison.right_only),
        *(f"changed generated file: {actual / name}" for name in comparison.diff_files),
        *(f"unreadable generated file: {actual / name}" for name in comparison.funny_files),
    ]
    for name in comparison.common_dirs:
        differences.extend(_directory_diff(expected / name, actual / name))
    return differences


def check_generated_docs(repo_root: Path, source_dir: Path) -> list[str]:
    """Return descriptions of generated files that are missing or stale."""
    with tempfile.TemporaryDirectory() as temp_dir:
        expected_source = Path(temp_dir) / "source"
        generate_all(repo_root, expected_source)
        differences: list[str] = []
        for relative_path in GENERATED_DIRS:
            differences.extend(
                _directory_diff(expected_source / relative_path, source_dir / relative_path)
            )
        for relative_path in GENERATED_FILES:
            expected = expected_source / relative_path
            actual = source_dir / relative_path
            if not actual.is_file():
                differences.append(f"missing generated file: {actual}")
            elif expected.read_bytes() != actual.read_bytes():
                differences.append(f"changed generated file: {actual}")
        return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed generated documentation is missing or stale",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Sphinx source directory (primarily useful for tests)",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()

    if args.check:
        differences = check_generated_docs(REPO_ROOT, source_dir)
        if differences:
            print("Generated documentation is stale:", file=sys.stderr)
            for difference in differences:
                print(f"- {difference}", file=sys.stderr)
            print("Run: python docs/tools/generate_docs.py", file=sys.stderr)
            return 1
        print("Generated documentation is up to date.")
        return 0

    print(generate_all(REPO_ROOT, source_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
