"""Local backend presentation, sharing the public CLI's planning/audit contract."""

from __future__ import annotations

import os

import typer

from SpatialBiologyToolkit.pipeline.control import (
    ActionRecord,
    action_receipt_payload,
    canonical_digest,
    make_preview_token,
    preview_run_identities,
    read_provenance_stdin,
    run_preview_snapshot,
    validate_preview_token,
)
from SpatialBiologyToolkit.pipeline.commands import stage_commands
from SpatialBiologyToolkit.pipeline.executions import resolve_execution
from SpatialBiologyToolkit.pipeline.local import (
    assert_no_orphan,
    launch_local,
    load_local_run,
    local_commands,
    prepare_local_run,
    request_cancel,
    wait_local,
)
from SpatialBiologyToolkit.pipeline.local_process import project_lock, require_linux
from SpatialBiologyToolkit.pipeline.manifests import (
    format_machine_output,
    utc_now,
    write_json,
)
from SpatialBiologyToolkit.pipeline.runs import (
    create_run_record,
    find_run_by_plan_token_digest,
    load_run_manifest,
    prospective_run_record,
    resolve_run_directory,
)


def emit(payload, output_format):
    typer.echo(format_machine_output(payload, output_format))


def print_resources(resources: dict) -> None:
    if not resources["available"]:
        typer.echo(f"Resources: {resources['reason']}")
        return
    typer.echo(
        f"Active stage: CPU {resources['cpu_percent']}%; "
        f"RAM (summed RSS) {resources['rss_mib']} MiB"
    )
    typer.echo("Process IDs: " + ", ".join(str(pid) for pid in resources["pids"]))
    for device in resources.get("gpu_devices", []):
        typer.echo(f"GPU (index, name, utilization %, used MiB, total MiB): {device}")
    for process in resources.get("gpu_processes", []):
        typer.echo(f"Stage GPU process (PID, device UUID, memory MiB): {process}")
    if resources.get("gpu_devices") and not resources.get("gpu_processes"):
        typer.echo("No matching stage PID currently has a reported GPU allocation.")
    if resources.get("gpu_note"):
        typer.echo(resources["gpu_note"])


def run_local_command(
    context,
    plan,
    *,
    detach,
    dry_run,
    reason,
    notes,
    plan_token,
    provenance_stdin,
    output_format,
    command,
):
    snapshot = run_snapshot(context, plan, reason, notes)
    token = plan_token
    if dry_run:
        token = make_preview_token(snapshot)
    digest = canonical_digest(token) if token else None
    workflow_id, technical_ids = (
        preview_run_identities(token, len(plan.resolved_stages))
        if token
        else (None, None)
    )

    def replay():
        existing = find_run_by_plan_token_digest(context, digest) if digest else None
        if existing is None:
            return False
        directory, manifest, _ = existing
        if manifest.execution_backend != "local":
            raise ValueError(
                "This preview token belongs to a different execution backend."
            )
        payload = {
            "schema_version": 1,
            "execution_backend": "local",
            "idempotent_replay": True,
            "workflow_run_id": manifest.run_id,
            "technical_record": str(directory),
        }
        if output_format != "text":
            emit(payload, output_format)
        else:
            typer.echo(f"Local run already exists for this preview: {manifest.run_id}")
        return True

    if not dry_run and replay():
        return
    if token and not dry_run:
        validate_preview_token(token, snapshot)
    commands = local_commands(plan, validate=not dry_run)
    if dry_run:
        run = prospective_run_record(
            context,
            plan,
            command=command,
            run_id=workflow_id,
            technical_run_ids=technical_ids,
            reason=reason,
            notes=notes,
        )
        payload = {
            "schema_version": 1,
            "preview_token": token,
            "preview_expires_in_seconds": 900,
            "plan": plan.model_dump(mode="json"),
            "prospective_workflow_run_id": workflow_id,
            "prospective_executions": [
                item.model_dump(mode="json") for item in run.executions
            ],
            "commands": {
                name: [item.model_dump(mode="json") for item in items]
                for name, items in commands.items()
            },
            "detach": detach,
        }
        if output_format != "text":
            emit(payload, output_format)
            return
        typer.echo("Local execution preview (sequential; no files created)")
        for execution in run.executions:
            typer.echo(
                f"  {execution.execution_label} {execution.stage_display_name} -> {execution.output_folder}"
            )
            for item in commands[execution.stage]:
                typer.echo(f"    {item.argv!r}  [environment: {item.environment}]")
        for warning in plan.warnings:
            typer.echo(f"Warning: {warning}")
        typer.echo("Environment paths are resolved and checked when the run starts.")
        typer.echo(f"Preview token: {token}")
        return
    require_linux()
    provenance = read_provenance_stdin() if provenance_stdin else None
    with project_lock(context.state_dir) as lock:
        if replay():
            return
        assert_no_orphan(context)
        if token:
            validate_preview_token(
                token, snapshot=run_snapshot(context, plan, reason, notes)
            )
        run = create_run_record(
            context,
            plan,
            command=command,
            run_id=workflow_id,
            reason=reason,
            notes=notes,
            technical_run_ids=technical_ids,
            plan_token_digest=digest,
            provenance_payload=provenance,
        )
        prepare_local_run(context, run, commands)
        state = launch_local(context, run, lock.fileno())
    receipt = action_receipt_payload(
        operation="run_local",
        target=context.project_metadata.project_id,
        actions=[
            ActionRecord(
                action="Started sequential local workflow",
                justification=reason or "Requested local execution",
                outcome="succeeded",
                state_changed=True,
                evidence=[f"workflow_run_id={run.run_id}"],
            )
        ],
    )
    payload = {
        "schema_version": 1,
        "execution_backend": "local",
        "idempotent_replay": False,
        "workflow_run_id": run.run_id,
        "technical_record": str(run.run_dir),
        "worker_pid": state.worker.pid,
        "detached": detach,
        "executions": [item.model_dump(mode="json") for item in run.executions],
        "action_receipt": receipt,
    }
    if output_format == "text":
        typer.echo(f"Local workflow: {run.run_id}; worker PID: {state.worker.pid}")
        for stage in state.stages:
            typer.echo(f"  {stage.execution_id:03d} {stage.stage}: {stage.log}")
        typer.echo(f"Technical record: {run.run_dir}")
        if detach:
            typer.echo("Detached: the worker continues after this terminal closes.")
    exit_code = 0
    if not detach:
        exit_code = wait_local(
            run.run_dir,
            emit=lambda text: typer.echo(text, nl=False, err=output_format != "text"),
        )
        payload["status"] = load_local_run(run.run_dir).status
        if output_format == "text":
            typer.echo(f"Local workflow: {payload['status']}")
    if output_format != "text":
        emit(payload, output_format)
    if exit_code:
        raise typer.Exit(exit_code)


def run_snapshot(context, plan, reason, notes):
    snapshot = run_preview_snapshot(context, plan, reason=reason)
    snapshot["active_prefix"] = (
        os.environ.get("CONDA_PREFIX") if plan.use_active_environment else None
    )
    snapshot["notes"] = notes
    snapshot["commands"] = {
        stage.name: [item.model_dump() for item in stage_commands(stage.name)]
        for stage in plan.resolved_stages
    }
    return snapshot


def try_cancel_local(
    context, reference, *, reason, dry_run, plan_token, provenance_stdin, output_format
):
    if context is None:
        return False
    execution = resolve_execution(context, reference)
    directory = resolve_run_directory(context, execution.workflow_run_id)
    if load_run_manifest(directory).execution_backend != "local":
        return False
    state = load_local_run(directory)
    snapshot = {
        "kind": "cancel_local",
        "project_id": context.project_metadata.project_id,
        "workflow_run_id": state.workflow_run_id,
        "worker": state.worker.model_dump() if state.worker else None,
        "reason": reason,
    }
    payload = {
        "execution_backend": "local",
        "workflow_run_id": state.workflow_run_id,
        "affected_executions": [
            item.execution_id
            for item in state.stages
            if item.status in {"pending", "running"}
        ],
        "reason": reason,
    }
    if dry_run:
        payload["preview_token"] = make_preview_token(snapshot)
    else:
        if not plan_token:
            raise ValueError(
                "Cancellation requires a preview token from sbt cancel --dry-run."
            )
        validate_preview_token(plan_token, snapshot)
        provenance = read_provenance_stdin() if provenance_stdin else None
        request_cancel(directory, reason=reason)
        payload["outcome"] = (
            "already terminal"
            if state.status in {"completed", "failed", "cancelled"}
            else "cancellation requested"
        )
        write_json(
            directory / "local_cancellation_audit.json",
            {
                **payload,
                "requested_at": utc_now().isoformat(),
                "provenance": provenance,
            },
        )
    if output_format != "text":
        emit(payload, output_format)
    elif dry_run:
        typer.echo(
            f"Cancel local workflow {state.workflow_run_id}: active stage and all remaining stages."
        )
        typer.echo(f"Affected executions: {payload['affected_executions']}")
        typer.echo(f"Preview token: {payload['preview_token']}")
    else:
        typer.echo(f"Local workflow {state.workflow_run_id}: {payload['outcome']}")
    return True
