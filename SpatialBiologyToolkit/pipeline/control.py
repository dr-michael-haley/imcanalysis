"""Preview tokens, provenance, and action receipts for guarded control operations."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .assets import inventory_assets
from .executions import load_execution_index
from .manifests import utc_now, write_json
from .models import PipelineModel, RunPlan
from .project import ProjectContext

PREVIEW_TOKEN_VERSION = "v1"
DEFAULT_PREVIEW_TTL_SECONDS = 900
PROVENANCE_FILE = "decision_provenance.json"


class ActionRecord(PipelineModel):
    action: str = Field(min_length=1, max_length=500)
    justification: str = Field(min_length=1, max_length=2000)
    outcome: Literal["planned", "succeeded", "failed", "skipped"]
    state_changed: bool = False
    evidence: list[str] = Field(default_factory=list)


class ActionReceipt(PipelineModel):
    schema_version: Literal[1] = 1
    receipt_id: str = Field(default_factory=lambda: f"receipt-{uuid.uuid4().hex}")
    operation: str
    target: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
    actions: list[ActionRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pipeline_version() -> str | None:
    try:
        return importlib.metadata.version("SpatialBiologyToolkit")
    except importlib.metadata.PackageNotFoundError:
        return None


def run_preview_snapshot(
    context: ProjectContext,
    plan: RunPlan,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return stable state which must remain unchanged between preview and submit."""

    index = load_execution_index(context)
    return {
        "kind": "run",
        "project_id": context.project_metadata.project_id,
        "pipeline_version": _pipeline_version(),
        "requested": list(plan.requested),
        "resolved_stages": [item.name for item in plan.resolved_stages],
        "dependency_policy": plan.dependency_policy,
        "ready": plan.ready,
        "errors": list(plan.errors),
        "warnings": list(plan.warnings),
        "config_digest": canonical_digest(context.config.model_dump(mode="json")),
        "asset_inventory_digest": canonical_digest(
            inventory_assets(
                project_id=context.project_metadata.project_id,
                project_root=context.root,
                config=context.config,
            ).model_dump(mode="json")
        ),
        "execution_index_digest": canonical_digest(
            [
                {
                    "execution_id": item.execution_id,
                    "technical_run_id": item.technical_run_id,
                    "status": item.status,
                }
                for item in index.executions
            ]
        ),
        "reason": reason,
    }


def make_preview_token(
    snapshot: dict[str, Any],
    *,
    ttl_seconds: int = DEFAULT_PREVIEW_TTL_SECONDS,
    now: int | None = None,
) -> str:
    issued = int(time.time() if now is None else now)
    expires = issued + ttl_seconds
    digest = canonical_digest({"expires": expires, "snapshot": snapshot})
    return f"{PREVIEW_TOKEN_VERSION}.{expires}.{digest}"


def preview_run_identities(
    token: str,
    stage_count: int,
) -> tuple[str, list[str]]:
    """Derive the exact workflow and technical IDs shown by a run preview."""

    digest = canonical_digest(token)
    workflow_run_id = f"planned-{digest[:24]}"
    technical_ids = [
        f"stage-{hashlib.sha256(f'{token}:{index}'.encode()).hexdigest()[:32]}"
        for index in range(stage_count)
    ]
    return workflow_run_id, technical_ids


def validate_preview_token(
    token: str,
    snapshot: dict[str, Any],
    *,
    now: int | None = None,
) -> None:
    try:
        version, raw_expiry, supplied_digest = token.split(".", 2)
        expires = int(raw_expiry)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid preview token format.") from exc
    if version != PREVIEW_TOKEN_VERSION:
        raise ValueError(f"Unsupported preview token version: {version}")
    current = int(time.time() if now is None else now)
    if expires < current:
        raise ValueError("The preview token has expired; create a fresh preview.")
    if expires > current + 3600:
        raise ValueError("The preview token expiry is outside the permitted window.")
    expected = canonical_digest({"expires": expires, "snapshot": snapshot})
    if not hmac.compare_digest(supplied_digest, expected):
        raise ValueError(
            "The project, configuration, execution index, or run plan changed after preview. "
            "Create and confirm a new preview."
        )


def read_provenance_stdin(*, max_bytes: int = 1_048_576) -> dict[str, Any]:
    raw = sys.stdin.buffer.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Decision provenance exceeds 1 MiB.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Decision provenance must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise TypeError("Decision provenance must be a JSON object.")
    if not payload.get("request_summary"):
        raise ValueError("Decision provenance requires request_summary.")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Decision provenance requires at least one decision.")
    return payload


def persist_provenance(run_dir: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    destination = write_json(run_dir / PROVENANCE_FILE, payload)
    return destination, canonical_digest(payload)


def action_receipt_payload(
    *,
    operation: str,
    target: str | None,
    actions: list[ActionRecord],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    receipt = ActionReceipt(
        operation=operation,
        target=target,
        actions=actions,
        warnings=warnings or [],
    )
    return receipt.model_dump(mode="json")


__all__ = [
    "DEFAULT_PREVIEW_TTL_SECONDS",
    "PROVENANCE_FILE",
    "ActionReceipt",
    "ActionRecord",
    "action_receipt_payload",
    "canonical_digest",
    "make_preview_token",
    "persist_provenance",
    "preview_run_identities",
    "read_provenance_stdin",
    "run_preview_snapshot",
    "validate_preview_token",
]
