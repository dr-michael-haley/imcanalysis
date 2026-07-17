"""Typer presentation layer for the lightweight ``sbt`` control interface."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from SpatialBiologyToolkit.config import load_config
from SpatialBiologyToolkit.config.export import write_resolved_config
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


app = typer.Typer(
    name="sbt",
    help="Spatial Biology Toolkit project and SLURM control interface.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Validate and export typed pipeline configuration.")
project_app = typer.Typer(help="Initialize, adopt, validate, and inspect SBT projects.")
stages_app = typer.Typer(help="List and explain registered pipeline stages.")
modes_app = typer.Typer(help="List and explain named workflow modes.")
app.add_typer(config_app, name="config")
app.add_typer(project_app, name="project")
app.add_typer(stages_app, name="stages")
app.add_typer(modes_app, name="modes")


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
    typer.echo(f"{'STAGE':<12} {'DOC ORDER':<10} {'OUTPUT SLUG':<32} DISPLAY NAME")
    for stage in sorted(STAGES, key=lambda item: (item.catalogue_order, item.name)):
        typer.echo(
            f"{stage.name:<12} {stage.catalogue_order:<10} "
            f"{stage.output_slug:<32} {stage.display_name}"
        )
        typer.echo(f"  {stage.description}")


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


@app.command("plan")
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


@app.command("run")
def run_command(
    targets: list[str] = typer.Argument(..., help="Stage aliases or workflow modes."),
    project: Path | None = typer.Option(None, "--project"),
    config: Path | None = typer.Option(None, "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
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
        plan = build_run_plan(context, targets)
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


@app.command("status")
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


@app.command("logs")
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


@app.command("report")
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


@app.command("summary")
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


@app.command("remove")
def remove_command(
    execution: str = typer.Argument(..., help="Active execution ID."),
    project: Path | None = typer.Option(None, "--project"),
    yes: bool = typer.Option(False, "--yes"),
    accept_asset_risk: bool = typer.Option(False, "--accept-asset-risk"),
    reason: str | None = typer.Option(None, "--reason"),
) -> None:
    try:
        context = _project(project)
        selected = resolve_execution(context, execution)
        output = execution_output_path(context, selected)
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
    try:
        audit = remove_execution(
            context,
            selected.execution_id,
            reason=reason,
            confirmation_mode="non_interactive" if yes else "interactive",
        )
    except Exception as exc:
        _fail(exc)
    typer.echo(
        f"Removed execution {audit.previous_execution.execution_label}; "
        f"renumbered {len(audit.renumbered)} later execution(s)."
    )
    typer.echo("Reusable project assets were not deleted or restored.")


if __name__ == "__main__":
    app()
