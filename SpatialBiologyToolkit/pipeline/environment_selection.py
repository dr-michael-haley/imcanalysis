"""Resolve per-run scientific environment selections without mutating the registry."""

from __future__ import annotations

from SpatialBiologyToolkit.environments.registry import (
    load_environment_registry,
    resolve_environment,
)

from .models import RunPlan
from .registry import get_stage


def apply_environment_override(plan: RunPlan, selector: str | None) -> RunPlan:
    """Return a plan with one registered single-stage environment override."""

    if selector is None:
        return plan
    if len(plan.resolved_stages) != 1:
        raise ValueError(
            "--environment requires a run plan containing exactly one stage. "
            "Use --dependency-policy none or run the stages separately."
        )
    stage_name = plan.resolved_stages[0].name
    default_keys = get_stage(stage_name).environment_keys
    if len(default_keys) != 1:
        if not default_keys:
            detail = "does not use a registered Conda environment"
        else:
            detail = "uses multiple Conda environments"
        raise ValueError(
            f"Stage '{stage_name}' {detail}; --environment currently supports "
            "single-environment stages only."
        )
    registry = load_environment_registry()
    try:
        selected_key, _definition = resolve_environment(registry, selector)
    except KeyError as exc:
        message = str(exc.args[0]) if exc.args else str(exc)
        raise ValueError(message) from exc
    return plan.model_copy(
        update={"environment_overrides": {stage_name: selected_key}}
    )


def effective_environment_keys(plan: RunPlan, stage_name: str) -> list[str]:
    """Return the registered environment keys effective for one planned stage."""

    override = plan.environment_overrides.get(stage_name)
    if override:
        return [override]
    return list(get_stage(stage_name).environment_keys)


__all__ = ["apply_environment_override", "effective_environment_keys"]
