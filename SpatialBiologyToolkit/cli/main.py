"""Typer presentation layer for the lightweight ``sbt`` control interface."""

from __future__ import annotations

import json
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from SpatialBiologyToolkit.config import load_config, write_compact_config
from SpatialBiologyToolkit.config.export import write_resolved_config
from SpatialBiologyToolkit.environments import EnvironmentManager, load_environment_registry
from SpatialBiologyToolkit.environments.models import EnvironmentSummary
from SpatialBiologyToolkit.pipeline.asset_cleanup import (
    apply_asset_cleanup,
    cleanup_audit,
    plan_asset_cleanup,
)
from SpatialBiologyToolkit.pipeline.assets import (
    count_raw_imc_files,
    resolve_assets,
)
from SpatialBiologyToolkit.pipeline.logs import resolve_run_logs, tail_text
from SpatialBiologyToolkit.pipeline.manifests import (
    format_machine_output,
    read_yaml,
    utc_now,
    write_text,
)
from SpatialBiologyToolkit.pipeline.executions import (
    execution_output_path,
    execution_summaries,
    load_execution_index,
    remove_execution,
    resolve_execution,
    resolve_technical_execution,
)
from SpatialBiologyToolkit.pipeline.models import model_data
from SpatialBiologyToolkit.pipeline.migration import (
    apply_execution_layout_migration,
    plan_execution_layout_migration,
)
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import (
    adopt_project,
    initialize_project,
    load_project,
    validate_project,
    write_config_template,
)
from SpatialBiologyToolkit.pipeline.registry import (
    MODES,
    STAGES,
    get_mode,
    get_stage,
    resolve_stage_selector,
    stage_script_path,
)
from SpatialBiologyToolkit.pipeline.runs import (
    STATUS_FILE,
    command_text,
    create_run_record,
    list_run_directories,
    prospective_run_record,
    resolve_run_directory,
)
from SpatialBiologyToolkit.pipeline.slurm import (
    SubmissionError,
    preview_submission_commands,
    submit_run,
)
from SpatialBiologyToolkit.pipeline.status import inspect_run_status


class OutputFormat(str, Enum):
    text = "text"
    yaml = "yaml"
    json = "json"


class SummaryFormat(str, Enum):
    table = "table"
    yaml = "yaml"
    json = "json"


REPOSITORY_URL = "https://github.com/dr-michael-haley/imcanalysis"
DOCUMENTATION_URL = "https://imcanalysis.readthedocs.io/en/latest/"

app = typer.Typer(
    name="sbt",
    help=(
        "Spatial Biology Toolkit project and SLURM control interface.\n\n"
        f"Repository: {REPOSITORY_URL}\n\n"
        f"Documentation: {DOCUMENTATION_URL}"
    ),
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Validate, compact, and export typed pipeline configuration.")
project_app = typer.Typer(help="Initialize, adopt, validate, and inspect SBT projects.")
stages_app = typer.Typer(help="List and explain registered pipeline stages.")
modes_app = typer.Typer(help="List and explain named workflow modes.")
env_app = typer.Typer(help="Validate, compare, capture, and synchronize fixed Conda environments.")
gui_app = typer.Typer(help="Launch optional interactive desktop applications in subprocesses.")
app.add_typer(config_app, name="config")
app.add_typer(project_app, name="project")
app.add_typer(stages_app, name="stages")
app.add_typer(modes_app, name="modes")
app.add_typer(env_app, name="env")
app.add_typer(gui_app, name="gui")


@gui_app.command("napari")
def gui_napari_command(
    project: Path | None = typer.Option(
        None, "--project", help="Optional SBT project root used to resolve config defaults."
    ),
    experiment: Path | None = typer.Option(
        None, "--experiment", help="Existing napari_sbt experiment folder or manifest."
    ),
    anndata: Path | None = typer.Option(None, "--anndata"),
    masks: Path | None = typer.Option(None, "--masks"),
    images: list[Path] | None = typer.Option(None, "--images"),
) -> None:
    """Launch cohort-first IMC exploration and classification without loading Qt here."""

    command = [sys.executable, "-m", "SpatialBiologyToolkit.napari_sbt"]
    if project is not None:
        command.extend(["--project", str(project)])
    if experiment is not None:
        command.extend(["--experiment", str(experiment)])
    if anndata is not None:
        command.extend(["--anndata", str(anndata)])
    if masks is not None:
        command.extend(["--masks", str(masks)])
    for folder in images or []:
        command.extend(["--images", str(folder)])
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


def _fail(exc: Exception | str, *, code: int = 2) -> NoReturn:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code)


def _emit_machine(
    value: BaseModel | dict[str, Any] | list[Any], output_format: OutputFormat
) -> None:
    if isinstance(value, list):
        data = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in value
        ]
        if output_format == OutputFormat.json:
            typer.echo(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            typer.echo(
                yaml.safe_dump(
                    data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                nl=False,
            )
        return
    typer.echo(format_machine_output(value, output_format.value), nl=False)


def _project(
    project: Path | None,
    config: Path | None = None,
):
    return load_project(project, config_override=config)


def _short_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _format_size(size: int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.1f} {suffix}"
        value /= 1024
    return f"{size} B"


def _print_plan(plan) -> None:
    typer.echo(f"Project: {plan.project_root}")
    typer.echo(f"Config:  {plan.config_source}")
    typer.echo(f"Backend: {plan.execution_backend}")
    typer.echo("")
    typer.echo("Submission order")
    for index, stage in enumerate(plan.resolved_stages, start=1):
        dependencies = ", ".join(stage.depends_on) or "-"
        requires = ", ".join(stage.requires_assets) or "-"
        produces = ", ".join(stage.produces_assets) or "-"
        typer.echo(
            f"  {index:>2}. {stage.name:<12} deps={dependencies} "
            f"requires={requires} produces={produces}"
        )
        typer.echo(f"      script: {stage.slurm_script}")
    if plan.warnings:
        typer.echo("")
        typer.echo("Warnings")
        for warning in plan.warnings:
            typer.echo(f"  ! {warning}")
    if plan.errors:
        typer.echo("")
        typer.echo("Errors")
        for error in plan.errors:
            typer.echo(f"  x {error}")
    typer.echo("")
    typer.echo("Plan is ready." if plan.ready else "Plan is not ready.")


def _print_validation(report) -> None:
    typer.echo(f"Project: {report.project_root}")
    typer.echo("")
    for title, items in (
        ("Required inputs", report.required_inputs),
        ("Optional inputs", report.optional_inputs),
        ("Generated assets", report.generated_assets),
        ("Reporting outputs", report.reporting_outputs),
    ):
        typer.echo(title)
        for item in items:
            symbol = {
                "ok": "+",
                "warning": "!",
                "missing": "x",
                "not_created": "-",
            }[item.status]
            path = f" [{item.path}]" if item.path else ""
            typer.echo(f"  {symbol} {item.name}{path}: {item.message}")
        typer.echo("")
    for stage, ready in report.stage_readiness.items():
        typer.echo(f"Project is {'ready' if ready else 'not ready'} for stage: {stage}")
        for message in report.readiness_messages.get(stage, []):
            typer.echo(f"  - {message}")
    if not report.stage_readiness:
        typer.echo(
            "Project structure is valid." if report.valid else "Project is invalid."
        )


@app.callback()
def main(
    context: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed package version.",
        is_eager=True,
    ),
) -> None:
    if version:
        try:
            from importlib.metadata import version as package_version

            value = package_version("SpatialBiologyToolkit")
        except Exception:
            value = "development"
        typer.echo(value)
        raise typer.Exit()
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())


@config_app.command("validate")
def config_validate(
    config: Path | None = typer.Argument(
        None,
        help="Config YAML path. Defaults to the current project's config.",
    ),
    project: Path | None = typer.Option(None, "--project"),
) -> None:
    try:
        if config is None:
            context = _project(project)
            config = context.config_path
        resolved = load_config(config)
    except Exception as exc:
        _fail(exc)
    typer.echo(f"Valid configuration: {Path(config).resolve(strict=False)}")
    typer.echo(f"Resolved sections: {len(resolved.__class__.model_fields)}")


@config_app.command("template")
def config_template(
    output: Path = typer.Option(Path("config.yaml"), "--output", "-o"),
    level: str = typer.Option("basic", "--level"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    try:
        written = write_config_template(output, config_level=level, force=force)
    except Exception as exc:
        _fail(exc)
    typer.echo(f"Wrote {level} config template: {written}")


@config_app.command("resolved")
def config_resolved(
    output: Path = typer.Option(..., "--output", "-o"),
    config: Path | None = typer.Option(None, "--config"),
    project: Path | None = typer.Option(None, "--project"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    try:
        context = _project(project, config)
        destination = output.expanduser().resolve(strict=False)
        if destination.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        write_resolved_config(context.config, destination)
    except Exception as exc:
        _fail(exc)
    typer.echo(f"Wrote resolved configuration: {destination}")


@config_app.command("compact")
def config_compact(
    config: Path | None = typer.Argument(
        None,
        help="Verbose config YAML path. Defaults to the current project's config.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="New compact YAML path. Defaults to <source>.compact.yaml.",
    ),
    project: Path | None = typer.Option(None, "--project"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing output file; the source config is never overwritten.",
    ),
) -> None:
    """Migrate verbose legacy YAML to canonical non-default settings."""
    try:
        if config is None:
            context = _project(project)
            config = context.config_path
        destination, unknown_keys = write_compact_config(
            config,
            output,
            force=force,
        )
    except Exception as exc:
        _fail(exc)
    typer.echo(f"Wrote compact configuration: {destination}")
    if unknown_keys:
        typer.echo(
            "Preserved unrecognized legacy keys: " + ", ".join(unknown_keys),
            err=True,
        )


@project_app.command("init")
def project_init(
    project: Path | None = typer.Option(None, "--project"),
    config_level: str = typer.Option("basic", "--config-level"),
    config_name: str = typer.Option("config.yaml", "--config-name"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    root = project or Path.cwd()
    try:
        context = initialize_project(
            root,
            config_name=config_name,
            config_level=config_level,
            force=force,
        )
    except Exception as exc:
        _fail(exc)
    typer.echo(f"Initialized SBT project: {context.root}")
    typer.echo(f"Project ID: {context.project_metadata.project_id}")
    typer.echo(f"Config: {context.config_path}")
    typer.echo(f"Raw inputs: {resolve_assets(context.config, context.root)[0].path}")


@project_app.command("adopt")
def project_adopt(
    project: Path | None = typer.Option(None, "--project"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    root = project or Path.cwd()
    try:
        result = adopt_project(root, config_path=config, force=force)
    except Exception as exc:
        _fail(exc)
    typer.echo(f"Adopted SBT project: {result.context.root}")
    typer.echo(f"Project ID: {result.context.project_metadata.project_id}")
    typer.echo("")
    typer.echo("Configured assets")
    for asset in result.assets:
        status = "present" if asset.exists else "missing/not created"
        typer.echo(f"  {asset.role:<18} {status:<19} {asset.path}")
    typer.echo("")
    if result.unexpected_paths:
        typer.echo("Unexpected top-level paths (left unchanged)")
        for path in result.unexpected_paths:
            typer.echo(f"  - {path}")
    else:
        typer.echo("No unexpected top-level paths identified.")


@project_app.command("validate")
def project_validate(
    project: Path | None = typer.Option(None, "--project"),
    config: Path | None = typer.Option(None, "--config"),
    stage: str | None = typer.Option(None, "--stage"),
    mode: str | None = typer.Option(None, "--mode"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    if stage and mode:
        _fail("--stage and --mode are mutually exclusive")
    try:
        context = _project(project, config)
        selected_stages = [get_stage(stage)] if stage else []
        report = validate_project(context, stages=selected_stages)
        mode_plan = build_run_plan(context, [mode]) if mode else None
    except Exception as exc:
        _fail(exc)

    if output_format != OutputFormat.text:
        data = model_data(report)
        if mode_plan is not None:
            data["mode_plan"] = mode_plan.model_dump(mode="json")
        _emit_machine(data, output_format)
    else:
        _print_validation(report)
        if mode_plan is not None:
            typer.echo("")
            typer.echo(
                f"Project is {'ready' if mode_plan.ready else 'not ready'} "
                f"for mode: {mode}"
            )
            for error in mode_plan.errors:
                typer.echo(f"  - {error}")

    ready = report.valid
    if stage:
        ready = ready and report.stage_readiness.get(stage, False)
    if mode_plan is not None:
        ready = ready and mode_plan.ready
    if not ready:
        raise typer.Exit(1)


@project_app.command("assets")
def project_assets(
    project: Path | None = typer.Option(None, "--project"),
    config: Path | None = typer.Option(None, "--config"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    try:
        context = _project(project, config)
        assets = resolve_assets(context.config, context.root)
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(assets, output_format)
        return
    for asset in assets:
        typer.echo(f"{asset.role:<20} {asset.lifecycle:<16} {asset.path}")


@project_app.command("describe")
def project_describe(
    project: Path | None = typer.Option(None, "--project"),
    config: Path | None = typer.Option(None, "--config"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    try:
        context = _project(project, config)
        assets = resolve_assets(context.config, context.root)
        runs = list_run_directories(context)
        executions = load_execution_index(context).executions
        latest = runs[-1] if runs else None
        latest_status = None
        if latest and (latest / STATUS_FILE).is_file():
            latest_status = read_yaml(latest / STATUS_FILE).get("overall_status")
        raw_count = count_raw_imc_files(assets[0].path)
    except Exception as exc:
        _fail(exc)

    data = {
        "schema_version": 1,
        "project_id": context.project_metadata.project_id,
        "project_root": str(context.root),
        "config_path": str(context.config_path),
        "raw_imc_file_count": raw_count,
        "recorded_runs": len(runs),
        "active_executions": len(executions),
        "latest_run": latest.name if latest else None,
        "latest_status": latest_status,
        "assets": [asset.model_dump(mode="json") for asset in assets],
    }
    if output_format != OutputFormat.text:
        _emit_machine(data, output_format)
        return

    typer.echo(f"Project ID: {context.project_metadata.project_id}")
    typer.echo(f"Root:       {context.root}")
    typer.echo(f"Config:     {context.config_path}")
    typer.echo(f"Raw files:  {raw_count}")
    typer.echo(f"Runs:       {len(runs)}")
    typer.echo(f"Executions: {len(executions)}")
    typer.echo(f"Latest:     {latest.name if latest else '-'}")
    typer.echo(f"Status:     {latest_status or '-'}")
    typer.echo("")
    typer.echo("Assets")
    for asset in assets:
        status = "present" if asset.exists else "absent"
        details = (
            _format_size(asset.size_bytes)
            if asset.kind == "file"
            else f"{asset.file_count or 0} top-level item(s)"
        )
        modified = (
            asset.modified_at.isoformat(timespec="seconds")
            if asset.modified_at
            else "-"
        )
        typer.echo(
            f"  {asset.role:<18} {status:<7} {details:<20} "
            f"modified={modified} {_short_path(asset.path, context.root)}"
        )


@project_app.command("notes")
def project_notes(
    project: Path | None = typer.Option(None, "--project"),
    add: str | None = typer.Option(
        None,
        "--add",
        help="Append a durable project note; omit to display the notes file.",
    ),
) -> None:
    try:
        context = _project(project)
        notes_path = context.root / context.project_metadata.notes_file
        if add:
            existing = (
                notes_path.read_text(encoding="utf-8")
                if notes_path.is_file()
                else "# Project notes\n"
            )
            entry = f"- {utc_now().isoformat()}: {add.strip()}\n"
            write_text(notes_path, existing.rstrip() + "\n\n" + entry)
        content = notes_path.read_text(encoding="utf-8")
    except Exception as exc:
        _fail(exc)
    typer.echo(content, nl=content.endswith("\n"))


@project_app.command("migrate-execution-layout")
def project_migrate_execution_layout(
    project: Path | None = typer.Option(None, "--project"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    """Explicitly migrate fixed stage folders to sequential executions."""
    try:
        context = _project(project)
        plan = plan_execution_layout_migration(context)
    except Exception as exc:
        _fail(exc)

    if output_format == OutputFormat.text:
        typer.echo(
            f"Legacy layout detected: {'yes' if plan.legacy_layout_detected else 'no'}"
        )
        typer.echo(f"Safe to apply: {'yes' if plan.safe_to_apply else 'no'}")
        typer.echo("")
        for record in plan.records:
            typer.echo(
                f"  {record.source_folder} -> "
                f"{record.execution.execution_label} {record.target_folder}"
            )
            typer.echo(
                f"    stage={record.execution.stage} "
                f"workflow={record.execution.workflow_run_id} "
                f"technical={record.execution.technical_run_id}"
            )
        for ambiguity in plan.ambiguities:
            typer.echo(f"  ! {ambiguity}")
    if dry_run:
        if output_format != OutputFormat.text:
            _emit_machine(plan, output_format)
        else:
            typer.echo("Dry run complete: no project files were changed.")
        if not plan.safe_to_apply:
            raise typer.Exit(1)
        return
    if not plan.legacy_layout_detected:
        if output_format != OutputFormat.text:
            _emit_machine(plan, output_format)
        else:
            typer.echo("No legacy execution layout was found; nothing was changed.")
        return
    if not plan.safe_to_apply:
        _fail("Migration is ambiguous; no files were changed.")
    try:
        audit = apply_execution_layout_migration(context, plan)
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(
            {
                "schema_version": 1,
                "plan": plan.model_dump(mode="json"),
                "audit": audit.model_dump(mode="json"),
            },
            output_format,
        )
        return
    typer.echo(f"Migrated {len(audit.records)} execution(s).")
    typer.echo("Technical identities and timestamps were preserved under .sbt/.")


@stages_app.command("list")
def stages_list(
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    if output_format != OutputFormat.text:
        _emit_machine(list(STAGES), output_format)
        return
    stages = sorted(STAGES, key=lambda item: (item.catalogue_order, item.name))
    stage_width = max(len("STAGE"), *(len(stage.name) for stage in stages))
    environment_width = max(
        len("ENVIRONMENT"),
        *(len(",".join(stage.environment_keys) or "-") for stage in stages),
    )
    slug_width = max(len("OUTPUT SLUG"), *(len(stage.output_slug) for stage in stages))

    typer.echo(
        typer.style(
            f"{'STAGE':<{stage_width}} "
            f"{'ENVIRONMENT':<{environment_width}} "
            f"{'OUTPUT SLUG':<{slug_width}} DISPLAY NAME",
            fg=typer.colors.BRIGHT_CYAN,
            bold=True,
        )
    )
    for stage in stages:
        environment = ",".join(stage.environment_keys) or "-"
        typer.echo(
            " ".join(
                (
                    typer.style(
                        f"{stage.name:<{stage_width}}",
                        fg=typer.colors.BRIGHT_GREEN,
                        bold=True,
                    ),
                    typer.style(
                        f"{environment:<{environment_width}}",
                        fg=typer.colors.BRIGHT_YELLOW,
                    ),
                    typer.style(
                        f"{stage.output_slug:<{slug_width}}",
                        fg=typer.colors.BRIGHT_MAGENTA,
                    ),
                    typer.style(stage.display_name, fg=typer.colors.BRIGHT_CYAN),
                )
            )
        )
        typer.secho(f"  {stage.description}", fg=typer.colors.WHITE)


@stages_app.command("explain")
def stages_explain(
    stage: str = typer.Argument(...),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    try:
        spec = get_stage(stage)
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(spec, output_format)
        return
    from SpatialBiologyToolkit.pipeline.registry import toolkit_root

    documentation = toolkit_root() / spec.documentation_path
    typer.echo(f"Stage: {spec.name} — {spec.display_name}")
    typer.echo(f"Output slug: {spec.output_slug}")
    typer.echo("Execution folder: assigned per project as NNN_<output slug>")
    typer.echo(f"SLURM script: {stage_script_path(spec)}")
    environment_registry = load_environment_registry()
    environment_names = [
        environment_registry.environments[key].conda_name
        for key in spec.environment_keys
    ]
    typer.echo(f"Environment keys: {', '.join(spec.environment_keys) or '-'}")
    typer.echo(f"Fixed Conda names: {', '.join(environment_names) or '-'}")
    typer.echo(f"Documentation: {documentation}")
    typer.echo("")
    if documentation.is_file():
        typer.echo(documentation.read_text(encoding="utf-8"))
    else:
        typer.echo(spec.description)
        typer.echo("")
        typer.echo("Shared stage explainer is missing.")


@modes_app.command("list")
def modes_list(
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    if output_format != OutputFormat.text:
        _emit_machine(list(MODES), output_format)
        return
    for mode in MODES:
        typer.echo(f"{mode.name:<24} {' -> '.join(mode.stages)}")
        typer.echo(f"  {mode.description}")


@modes_app.command("explain")
def modes_explain(
    mode: str = typer.Argument(...),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    try:
        spec = get_mode(mode)
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(spec, output_format)
        return
    typer.echo(f"Mode: {spec.name}")
    typer.echo(f"Purpose: {spec.description}")
    typer.echo(f"Stages: {' -> '.join(spec.stages)}")


def _env_manager(toolkit: Path | None) -> EnvironmentManager:
    return EnvironmentManager(toolkit)


def _env_machine(value: Any, output_format: SummaryFormat | OutputFormat) -> None:
    selected = OutputFormat.json if output_format.value == "json" else OutputFormat.yaml
    _emit_machine(value, selected)


def _required_environment_commands(rows: list[EnvironmentSummary]) -> str:
    return "\n".join(f"  sbt env sync {row.key}" for row in rows)


def _ensure_run_environments(stage_names: list[str]) -> None:
    """Stop before run creation unless every selected stage environment exists."""

    manager = _env_manager(None)
    required = manager.required_for_stages(stage_names)
    missing = [row for row in required if not row.exists]
    if not missing:
        return

    typer.echo("Required Conda environments:")
    for row in required:
        status = "available" if row.exists else "MISSING"
        management = "repository-managed" if row.managed else "external"
        typer.echo(
            f"  - {row.conda_name} ({row.key}; {management}): {status} "
            f"for {', '.join(row.stages)}"
        )

    missing_external = [row for row in missing if not row.managed]
    if missing_external:
        details = "\n".join(
            f"  - {row.conda_name} ({row.key}) for {', '.join(row.stages)}"
            for row in missing_external
        )
        raise RuntimeError(
            "The following externally managed Conda environments are missing and "
            "cannot be installed automatically:\n"
            f"{details}\n"
            "Inspect each environment with 'sbt env show <key>' and follow its "
            "stage/environment documentation. No run record was created and no jobs "
            "were submitted."
        )

    missing_managed = [row for row in missing if row.managed]
    invalid_specifications: list[str] = []
    for row in missing_managed:
        validation = manager.validate(row.key)
        if validation.valid:
            continue
        errors = [
            issue.message
            for issue in validation.issues
            if issue.severity == "error"
        ]
        detail = "; ".join(errors) or "the specification did not validate"
        invalid_specifications.append(f"  - {row.key}: {detail}")
    if invalid_specifications:
        raise RuntimeError(
            "The repository cannot safely install these missing environments "
            "because their specifications are invalid:\n"
            + "\n".join(invalid_specifications)
            + "\nUpdate the toolkit checkout and try again. If the error remains, "
            "report it to the toolkit maintainer rather than installing arbitrary "
            "packages. No run record was created and no jobs were submitted."
        )

    names = ", ".join(row.conda_name for row in missing_managed)
    if not typer.confirm(
        f"Install the missing environment(s) now ({names})?",
        default=False,
    ):
        raise RuntimeError(
            "Missing environments were not installed. Install only the environments "
            "needed by this run with:\n"
            f"{_required_environment_commands(missing_managed)}\n"
            "No run record was created and no jobs were submitted."
        )

    for row in missing_managed:
        typer.echo(f"Installing {row.conda_name} for {', '.join(row.stages)}...")
        manager.sync(row.key, verbose=True)
        typer.echo(f"Installed and smoke-tested {row.conda_name}.")

    remaining = [
        row
        for row in manager.required_for_stages(stage_names)
        if not row.exists
    ]
    if remaining:
        raise RuntimeError(
            "Environment installation finished, but these environments are still "
            "not visible to Conda: "
            + ", ".join(row.conda_name for row in remaining)
        )


@env_app.command("list")
def env_list(
    output_format: SummaryFormat = typer.Option(SummaryFormat.table, "--format"),
    compare: bool = typer.Option(False, "--compare", help="Inspect live package drift (slower)."),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """List registry environments, fixed names, availability, and stage use."""
    try:
        rows = _env_manager(toolkit).list_environments(compare=compare)
    except Exception as exc:
        _fail(exc)
    if output_format != SummaryFormat.table:
        _env_machine(rows, output_format)
        return
    typer.echo(f"{'Key':<16} {'Conda name':<22} {'Managed':<8} {'Exists':<8} {'Drift':<9} Stages")
    for row in rows:
        exists = "unknown" if row.exists is None else "yes" if row.exists else "no"
        typer.echo(
            f"{row.key:<16} {row.conda_name:<22} "
            f"{'yes' if row.managed else 'external':<8} {exists:<8} {row.drift:<9} "
            f"{', '.join(row.stages) or '-'}"
        )


@env_app.command("show")
def env_show(
    environment: str = typer.Argument(..., help="Logical key or fixed Conda name."),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Show one registry environment without importing scientific packages."""
    try:
        detail = _env_manager(toolkit).show(environment)
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(detail, output_format)
        return
    typer.echo(f"Environment: {detail['key']}")
    typer.echo(f"Conda name: {detail['conda_name']}")
    typer.echo(f"Management: {'repository' if detail['managed'] else 'external'}")
    typer.echo(f"Platform: {detail['platform']}")
    typer.echo(f"Exists: {detail['exists'] if detail['exists'] is not None else 'unknown'}")
    typer.echo(f"Prefix: {detail['prefix'] or '-'}")
    typer.echo(f"Stages: {', '.join(detail['stages']) or '-'}")
    typer.echo(f"Toolkit overlay: {detail['toolkit_overlay']}")
    typer.echo("Specification:")
    for name, path in detail["paths"].items():
        if path:
            typer.echo(f"  {name}: {path}")
    typer.echo("Smoke tests:")
    for command in detail["smoke_tests"]:
        typer.echo(f"  {' '.join(command)}")
    for note in detail["notes"]:
        typer.echo(f"Note: {note}")


@env_app.command("doctor")
def env_doctor(
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Run non-mutating launcher, registry, specification, and mapping checks."""
    try:
        report = _env_manager(toolkit).doctor()
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(report, output_format)
    else:
        for check in report.checks:
            marker = "OK" if check.status == "ok" else "WARN" if check.status == "warning" else "FAIL"
            typer.echo(f"[{marker}] {check.name}: {check.detail}")
    if not report.healthy:
        raise typer.Exit(2)


@env_app.command("validate-spec")
def env_validate_spec(
    environment: str | None = typer.Argument(None, help="Logical key or fixed Conda name."),
    all_: bool = typer.Option(False, "--all"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Validate environment.yml, lockfile, pip extras, names, and platforms."""
    try:
        manager = _env_manager(toolkit)
        selected = manager.select(environment, all_environments=all_)
        reports = [manager.validate(key) for key, _ in selected]
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(reports, output_format)
    else:
        for report in reports:
            typer.echo(
                f"{report.environment_key} ({report.conda_name}): "
                f"{'valid' if report.valid else 'INVALID'}"
            )
            for issue in report.issues:
                typer.echo(f"  {issue.severity.upper()}: {issue.message}")
    if any(not report.valid for report in reports):
        raise typer.Exit(2)


def _print_comparison(comparison) -> None:
    typer.echo(f"Environment: {comparison.environment_key} ({comparison.conda_name})")
    typer.echo(f"Exists: {'yes' if comparison.exists else 'no'}")
    if comparison.error:
        typer.echo(f"Error: {comparison.error}")
    if comparison.drift:
        for item in comparison.drift:
            marker = "!" if item.material else "~"
            typer.echo(f"  {marker} [{item.layer}] {item.message}")
    elif comparison.completed:
        typer.echo("  All declared Conda, lock, pip-extra, and toolkit checks match.")
    typer.echo(f"Result: {comparison.result.upper()}")


@env_app.command("compare")
def env_compare(
    environment: str | None = typer.Argument(None, help="Logical key or fixed Conda name."),
    all_: bool = typer.Option(False, "--all"),
    output_format: SummaryFormat = typer.Option(SummaryFormat.table, "--format"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Compare live Conda, pip extras, lock records, and toolkit overlay."""
    try:
        manager = _env_manager(toolkit)
        selected = manager.select(environment, all_environments=all_)
        comparisons = [manager.compare(key) for key, _ in selected]
    except Exception as exc:
        _fail(exc)
    if output_format != SummaryFormat.table:
        _env_machine(comparisons if all_ else comparisons[0], output_format)
    else:
        for index, comparison in enumerate(comparisons):
            if index:
                typer.echo("")
            _print_comparison(comparison)
    code = max((comparison.exit_code for comparison in comparisons), default=0)
    if code:
        raise typer.Exit(code)


@env_app.command("lock")
def env_lock(
    environment: str | None = typer.Argument(None, help="Logical key or fixed Conda name."),
    all_: bool = typer.Option(False, "--all"),
    check: bool = typer.Option(False, "--check", help="Compare a temporary generated lock only."),
    verbose: bool = typer.Option(False, "--verbose"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Generate target-platform lockfiles atomically with conda-lock."""
    try:
        manager = _env_manager(toolkit)
        selected = manager.select(environment, all_environments=all_)
        stale = False
        for key, definition in selected:
            if not definition.managed:
                typer.echo(f"Skipping external environment {key} ({definition.conda_name}).")
                continue
            current, command = manager.lock(key, check=check, verbose=verbose)
            typer.echo(
                f"{key}: {'current' if current else 'would change' if check else 'updated'}"
            )
            if verbose:
                typer.echo(f"  {' '.join(command)}")
            if check and not current:
                stale = True
        if stale:
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc)


def _print_sync_plan(plan) -> None:
    typer.echo(f"Environment: {plan.environment_key} ({plan.conda_name})")
    typer.echo(f"Exists: {'yes' if plan.exists else 'no'}")
    typer.echo(f"Drift: {plan.drift}")
    typer.echo(f"Recreation required: {'yes' if plan.recreation_required else 'no'}")
    typer.echo("Planned operations:")
    for action in plan.actions:
        typer.echo(f"  - {action}")
    if not plan.actions:
        typer.echo("  - No action required")
    typer.echo("Smoke tests:")
    for command in plan.smoke_tests:
        typer.echo(f"  - {' '.join(command)}")


@env_app.command("sync")
def env_sync(
    environment: str | None = typer.Argument(None, help="Logical key or fixed Conda name."),
    all_: bool = typer.Option(False, "--all"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    recreate: bool = typer.Option(False, "--recreate", help="Allow fixed-name recreation when drift exists."),
    yes: bool = typer.Option(False, "--yes", help="Confirm requested recreation non-interactively."),
    verbose: bool = typer.Option(False, "--verbose"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Create or safely recreate fixed Conda environments from repository locks."""
    try:
        manager = _env_manager(toolkit)
        selected = manager.select(environment, all_environments=all_)
        for key, definition in selected:
            if not definition.managed:
                typer.echo(f"Skipping external environment {key} ({definition.conda_name}).")
                continue
            plan = manager.sync_plan(key)
            _print_sync_plan(plan)
            if dry_run or not plan.actions:
                continue
            confirmed = yes
            if plan.recreation_required:
                if not recreate:
                    _fail(
                        f"{definition.conda_name} has drift; rerun with --recreate after reviewing the plan."
                    )
                if not yes:
                    confirmed = typer.confirm(
                        f"Remove and recreate fixed environment {definition.conda_name}?",
                        default=False,
                    )
                    if not confirmed:
                        raise typer.Abort()
            manager.sync(
                key,
                recreate=recreate,
                confirmed=confirmed,
                verbose=verbose,
            )
            typer.echo(f"Synchronized {definition.conda_name}.")
    except (typer.Exit, typer.Abort):
        raise
    except Exception as exc:
        _fail(exc)


@env_app.command("capture")
def env_capture(
    environment: str = typer.Argument(..., help="Logical key or fixed Conda name."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    write: bool = typer.Option(False, "--write"),
    accept_vcs: bool = typer.Option(
        False,
        "--accept-vcs",
        help="Explicitly retain observed VCS requirements in pip-extras.txt.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Capture a live environment into reviewed repository candidates."""
    if dry_run and write:
        _fail("Choose --dry-run or --write, not both.")
    try:
        plan = _env_manager(toolkit).capture(
            environment, write=write, accept_vcs=accept_vcs, verbose=verbose
        )
    except Exception as exc:
        _fail(exc)
    typer.echo(f"Environment: {plan.environment_key} ({plan.conda_name})")
    typer.echo(f"Candidate files: {plan.candidate_directory}")
    for name, difference in plan.differences.items():
        typer.echo(f"\nProposed {name} changes:")
        typer.echo(difference)
    typer.echo("\nExcluded toolkit overlay:")
    typer.echo(f"  {plan.excluded_toolkit or 'not observed'}")
    typer.echo("\nRequires manual review:")
    for item in plan.review_requirements:
        typer.echo(f"  {item}")
    if not plan.review_requirements:
        typer.echo("  none")
    typer.echo("Repository files updated." if write else "Dry run: repository files were not modified.")


@env_app.command("test")
def env_test(
    environment: str | None = typer.Argument(None, help="Logical key or fixed Conda name."),
    all_: bool = typer.Option(False, "--all"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
    verbose: bool = typer.Option(False, "--verbose"),
    toolkit: Path | None = typer.Option(None, "--toolkit-root"),
) -> None:
    """Run registered lightweight smoke tests through conda run."""
    try:
        manager = _env_manager(toolkit)
        selected = manager.select(environment, all_environments=all_)
        reports = [manager.test(key, verbose=verbose) for key, _ in selected]
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(reports, output_format)
    else:
        for report in reports:
            typer.echo(
                f"{report.environment_key} ({report.conda_name}): "
                f"{'PASS' if report.passed else 'FAIL'}"
            )
            for result in report.tests:
                typer.echo(
                    f"  {'PASS' if result.passed else 'FAIL'} "
                    f"({result.duration_seconds:.2f}s) {' '.join(result.command)}"
                )
                if result.stderr_tail and not result.passed:
                    typer.echo(f"    {result.stderr_tail}")
    if any(not report.passed for report in reports):
        raise typer.Exit(1)


@app.command(
    "plan",
    help="Validate and preview stages, dependencies, assets, and readiness.",
)
def plan_command(
    targets: list[str] = typer.Argument(..., help="Stage aliases or workflow modes."),
    project: Path | None = typer.Option(None, "--project"),
    config: Path | None = typer.Option(None, "--config"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    try:
        context = _project(project, config)
        plan = build_run_plan(context, targets)
    except Exception as exc:
        _fail(exc)
    if output_format == OutputFormat.text:
        _print_plan(plan)
    else:
        _emit_machine(plan, output_format)
    if not plan.ready:
        raise typer.Exit(1)


@app.command(
    "run",
    help="Allocate execution IDs and submit validated stages to SLURM.",
)
def run_command(
    targets: list[str] = typer.Argument(..., help="Stage aliases or workflow modes."),
    project: Path | None = typer.Option(None, "--project"),
    config: Path | None = typer.Option(None, "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    no_deps: bool = typer.Option(
        False,
        "--no-deps",
        help=(
            "Submit only explicitly selected stages; require their upstream assets "
            "to exist instead of scheduling dependency stages."
        ),
    ),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Optional human-readable purpose recorded in run and stage reports.",
    ),
    note: list[str] = typer.Option(
        [],
        "--note",
        help="Optional repeatable run note recorded in run and stage reports.",
    ),
) -> None:
    try:
        context = _project(project, config)
        plan = build_run_plan(
            context,
            targets,
            include_dependencies=not no_deps,
        )
    except Exception as exc:
        _fail(exc)
    if not plan.ready:
        _print_plan(plan)
        raise typer.Exit(1)

    command = command_text(sys.argv)
    if dry_run:
        run = prospective_run_record(
            context,
            plan,
            command=command,
            reason=reason,
            notes=note,
        )
        _print_plan(plan)
        typer.echo("")
        typer.echo(f"Prospective workflow run ID: {run.workflow_run_id}")
        typer.echo(f"Prospective technical directory: {run.run_dir}")
        typer.echo("Prospective execution IDs")
        for execution in run.executions:
            typer.echo(
                f"  {execution.execution_label} — {execution.stage_display_name} "
                f"({execution.output_folder})"
            )
        typer.echo(f"Resolved config path: {run.resolved_config_path}")
        typer.echo("")
        typer.echo("Exact submission preview")
        for arguments, exported in preview_submission_commands(context, plan, run):
            typer.echo(f"  {command_text(arguments)}")
            typer.echo(
                "    env: "
                + " ".join(f"{key}={value}" for key, value in exported.items())
            )
        typer.echo("")
        typer.echo(
            "Dry run complete: no run directory was created and no jobs were submitted."
        )
        return

    try:
        _ensure_run_environments([stage.name for stage in plan.resolved_stages])
        run = create_run_record(
            context,
            plan,
            command=command,
            reason=reason,
            notes=note,
        )
        submitted = submit_run(context, plan, run)
    except SubmissionError as exc:
        typer.echo(f"Run record: {run.run_dir}", err=True)
        _fail(exc, code=1)
    except Exception as exc:
        _fail(exc, code=1)

    typer.echo(f"Submitted workflow: {run.workflow_run_id}")
    typer.echo(f"Technical record: {run.run_dir}")
    for job in submitted.jobs:
        execution = run.execution_for_stage(job.stage)
        dependency = (
            f" afterok:{job.dependency_job_id}" if job.dependency_job_id else ""
        )
        typer.echo(
            f"  {execution.execution_label} — {execution.stage_display_name:<28} "
            f"job {job.job_id}{dependency}"
        )
    typer.echo("")
    typer.echo("Next:")
    first = run.executions[0].execution_label
    typer.echo(f"  sbt status {first} --project {context.root}")
    typer.echo(f"  sbt logs {first} --project {context.root}")
    typer.echo(f"  sbt summary --project {context.root}")


@app.command(
    "status",
    help="Refresh and show scheduler status for one project execution.",
)
def status_command(
    execution: str = typer.Argument("latest", help="Execution ID or 'latest'."),
    project: Path | None = typer.Option(None, "--project"),
    technical_run_id: str | None = typer.Option(
        None,
        "--technical-run-id",
        help="Resolve an immutable technical execution ID explicitly.",
    ),
    details: bool = typer.Option(False, "--details"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
) -> None:
    try:
        context = _project(project)
        selected = (
            resolve_technical_execution(context, technical_run_id)
            if technical_run_id
            else resolve_execution(context, execution)
        )
        run_dir = resolve_run_directory(context, selected.workflow_run_id)
        report = inspect_run_status(context, run_dir)
        selected = resolve_technical_execution(context, selected.technical_run_id)
        stage_status = next(
            (
                item
                for item in report.stages
                if item.technical_run_id == selected.technical_run_id
            ),
            None,
        )
    except Exception as exc:
        _fail(exc)
    if output_format != OutputFormat.text:
        _emit_machine(
            {
                "schema_version": 1,
                "execution": selected.model_dump(mode="json"),
                "status": stage_status.model_dump(mode="json") if stage_status else None,
            },
            output_format,
        )
        return
    typer.echo(
        f"Execution {selected.execution_label} — {selected.stage_display_name}"
    )
    typer.echo(f"Status: {selected.status}")
    typer.echo(f"SLURM job: {selected.slurm_job_id or '-'}")
    if stage_status and stage_status.detail:
        typer.echo(f"Detail: {stage_status.detail}")
    if details:
        typer.echo(f"Technical execution ID: {selected.technical_run_id}")
        typer.echo(f"Workflow run ID: {selected.workflow_run_id}")
        typer.echo(f"Technical record: {run_dir}")
    for warning in report.warnings:
        typer.echo(f"Warning: {warning}")


@app.command(
    "logs",
    help="Show or locate recorded stdout and stderr for one execution.",
)
def logs_command(
    execution: str = typer.Argument("latest", help="Execution ID or 'latest'."),
    project: Path | None = typer.Option(None, "--project"),
    technical_run_id: str | None = typer.Option(None, "--technical-run-id"),
    stage: str | None = typer.Option(
        None,
        "--stage",
        help="Compatibility filter; it must match the selected execution.",
    ),
    stdout: bool = typer.Option(False, "--stdout"),
    stderr: bool = typer.Option(False, "--stderr"),
    tail: int = typer.Option(40, "--tail", min=0),
    path_only: bool = typer.Option(False, "--path-only"),
) -> None:
    include_stdout = stdout or not stderr
    include_stderr = stderr or not stdout
    try:
        context = _project(project)
        selected = (
            resolve_technical_execution(context, technical_run_id)
            if technical_run_id
            else resolve_execution(context, execution)
        )
        if stage and resolve_stage_selector(stage).name != selected.stage:
            raise ValueError(
                f"Execution {selected.execution_label} is stage "
                f"'{selected.stage}', not '{stage}'."
            )
        run_dir = resolve_run_directory(context, selected.workflow_run_id)
        logs = resolve_run_logs(
            run_dir,
            stage=selected.stage,
            include_stdout=include_stdout,
            include_stderr=include_stderr,
        )
    except Exception as exc:
        _fail(exc)
    if not logs:
        typer.echo("No recorded logs match the selection.")
        return
    for record in logs:
        if path_only:
            typer.echo(str(record.path))
            continue
        typer.echo(
            f"[{record.stage} job={record.job_id or '-'} {record.stream}] {record.path}"
        )
        if not record.exists:
            typer.echo("  Log file has not been created.")
        elif tail:
            content = tail_text(record.path, tail)
            if content:
                typer.echo(content)
        typer.echo("")


@app.command(
    "report",
    help="Display the human-facing report for one project execution.",
)
def report_command(
    execution: str = typer.Argument("latest", help="Execution ID or 'latest'."),
    project: Path | None = typer.Option(None, "--project"),
    path_only: bool = typer.Option(False, "--path-only"),
) -> None:
    try:
        context = _project(project)
        selected = resolve_execution(context, execution)
        output = execution_output_path(context, selected)
        report = output / "README.md"
        if not report.is_file():
            raise FileNotFoundError(f"Execution report not found: {report}")
    except Exception as exc:
        _fail(exc)
    if path_only:
        typer.echo(str(report))
    else:
        typer.echo(
            f"Execution {selected.execution_label} — {selected.stage_display_name}"
        )
        typer.echo(report.read_text(encoding="utf-8"))


@app.command(
    "summary",
    help="List project executions with their statuses, assets, and outputs.",
)
def summary_command(
    project: Path | None = typer.Option(None, "--project"),
    stage: str | None = typer.Option(None, "--stage"),
    status: str | None = typer.Option(None, "--status"),
    latest: bool = typer.Option(False, "--latest"),
    details: bool = typer.Option(False, "--details"),
    assets: bool = typer.Option(False, "--assets"),
    include_removed: bool = typer.Option(False, "--include-removed"),
    output_format: SummaryFormat = typer.Option(SummaryFormat.table, "--format"),
) -> None:
    try:
        context = _project(project)
        records = execution_summaries(context, include_removed=include_removed)
        if stage:
            alias = resolve_stage_selector(stage).name
            records = [item for item in records if item.stage == alias]
        if status:
            records = [item for item in records if item.status == status.lower()]
        if latest and records:
            records = [records[-1]]
    except Exception as exc:
        _fail(exc)

    if output_format != SummaryFormat.table:
        data = {
            "schema_version": 1,
            "project_id": context.project_metadata.project_id,
            "executions": [item.model_dump(mode="json") for item in records],
        }
        _emit_machine(
            data,
            OutputFormat.json
            if output_format == SummaryFormat.json
            else OutputFormat.yaml,
        )
        return

    typer.echo(
        f"{'ID':<6} {'STAGE':<28} {'STATUS':<11} {'STARTED':<17} "
        f"{'DONE/DURATION':<17} {'ASSETS':<9} OUTPUT"
    )
    for item in records:
        started = item.started_at.strftime("%Y-%m-%d %H:%M") if item.started_at else "-"
        if item.completed_at:
            completed = item.completed_at.strftime("%Y-%m-%d %H:%M")
        elif item.duration_seconds is not None:
            completed = f"{item.duration_seconds:.0f}s"
        else:
            completed = "-"
        label = f"{item.execution_label}{'*' if item.removed else ''}"
        typer.echo(
            f"{label:<6} {item.stage_display_name:<28} {item.status:<11} "
            f"{started:<17} {completed:<17} {item.asset_effect:<9} "
            f"{item.output_folder}"
        )
        if details:
            typer.echo(
                f"       technical={item.technical_run_id} "
                f"workflow={item.workflow_run_id} slurm={item.slurm_job_id or '-'}"
            )
        if assets and item.asset_effect != "none":
            typer.echo(
                "       Reusable assets may have been created or modified; inspect "
                "the stage manifest for canonical paths."
            )
    if not records:
        typer.echo("No executions match the selection.")
    if include_removed and any(item.removed for item in records):
        typer.echo("* removed execution from the hidden technical audit")


@app.command(
    "remove",
    help="Remove a visible execution safely and renumber later executions.",
)
def remove_command(
    execution: str = typer.Argument(..., help="Active execution ID."),
    project: Path | None = typer.Option(None, "--project"),
    yes: bool = typer.Option(False, "--yes"),
    accept_asset_risk: bool = typer.Option(False, "--accept-asset-risk"),
    remove_assets: bool = typer.Option(
        False,
        "--remove-assets",
        help="With --yes, remove eligible unused assets after removing the execution.",
    ),
    reason: str | None = typer.Option(None, "--reason"),
) -> None:
    if remove_assets and not yes:
        _fail("--remove-assets requires --yes for non-interactive confirmation.")
    try:
        context = _project(project)
        selected = resolve_execution(context, execution)
        output = execution_output_path(context, selected)
        asset_plan = plan_asset_cleanup(context, selected)
    except Exception as exc:
        _fail(exc)
    risky = selected.asset_effect != "none"
    typer.echo(
        f"Remove execution {selected.execution_label} — "
        f"{selected.stage_display_name}?"
    )
    typer.echo("")
    typer.echo(f"Status: {selected.status}")
    typer.echo(f"Output folder: {output}")
    typer.echo(f"Technical execution ID: {selected.technical_run_id}")
    typer.echo(f"Reusable asset effect: {selected.asset_effect}")
    typer.echo("")
    typer.echo(
        "This removes the human-facing output folder and active workflow entry. "
        "Permanent technical evidence is retained under .sbt/."
    )
    if asset_plan.removable or asset_plan.retained:
        typer.echo("")
        typer.echo("Remaining unused assets eligible for removal:")
        if asset_plan.removable:
            for item in asset_plan.removable:
                typer.echo(f"  [{item.role}] {item.path}")
        else:
            typer.echo("  None.")

        dependent = [
            item
            for item in asset_plan.retained
            if item.reason == "used by remaining stages"
        ]
        typer.echo("")
        typer.echo("Assets retained because remaining stages depend on them:")
        if dependent:
            for item in dependent:
                stages = ", ".join(item.dependent_stages)
                typer.echo(f"  [{item.role}] {item.path}")
                typer.echo(f"       {stages}")
        else:
            typer.echo("  None.")

        protected = [
            item
            for item in asset_plan.retained
            if item.reason != "used by remaining stages"
        ]
        if protected:
            typer.echo("")
            typer.echo("Other protected assets:")
            for item in protected:
                typer.echo(f"  [{item.role}] {item.path}")
                typer.echo(f"       {item.reason}")
    if not yes and not typer.confirm("Continue?", default=False):
        raise typer.Abort()
    if risky:
        typer.echo("")
        typer.echo(
            "This execution created or modified reusable assets, or its effect is "
            "unknown. Removal does not restore those assets and downstream analyses "
            "may depend on them."
        )
        if yes and not accept_asset_risk:
            _fail(
                "Non-interactive removal requires --accept-asset-risk for this execution."
            )
        if not accept_asset_risk and not typer.confirm(
            "Remove this execution from the visible workflow anyway?",
            default=False,
        ):
            raise typer.Abort()
    remove_assets_now = remove_assets
    if asset_plan.removable and not yes:
        typer.echo("")
        response = typer.prompt(
            "Type 'yes' to remove the eligible unused assets; "
            "anything else keeps them",
            default="",
            show_default=False,
        )
        remove_assets_now = response.strip().lower() == "yes"
    try:
        audit = remove_execution(
            context,
            selected.execution_id,
            reason=reason,
            confirmation_mode="non_interactive" if yes else "interactive",
            asset_cleanup=cleanup_audit(
                asset_plan,
                offered=bool(asset_plan.removable),
                confirmed=remove_assets_now,
            ),
        )
        if remove_assets_now:
            audit = apply_asset_cleanup(context, audit, asset_plan)
    except Exception as exc:
        _fail(exc)
    typer.echo(
        f"Removed execution {audit.previous_execution.execution_label}; "
        f"renumbered {len(audit.renumbered)} later execution(s)."
    )
    cleanup = audit.asset_cleanup
    if cleanup and cleanup.confirmed:
        typer.echo(
            f"Cleaned {len(cleanup.cleaned_paths)} unused asset path(s) "
            f"({cleanup.removed_entries} filesystem entries)."
        )
        if cleanup.errors:
            typer.echo("Some asset cleanup operations failed:")
            for error in cleanup.errors:
                typer.echo(f"  {error}")
        if cleanup.retained:
            typer.echo(
                f"Retained {len(cleanup.retained)} protected or shared asset path(s)."
            )
    else:
        typer.echo("Reusable project assets were not deleted or restored.")


if __name__ == "__main__":
    app()
