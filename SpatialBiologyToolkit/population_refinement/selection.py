"""Marker evidence and deterministic hierarchical exemplar sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import MarkerRule, SamplingSpec


@dataclass
class SelectionResult:
    candidates: pd.DataFrame
    examples: pd.DataFrame
    coverage: pd.DataFrame
    checks: pd.DataFrame
    warnings: list[str]
    settings: dict


def assess_markers(
    candidates: pd.DataFrame,
    evidence: pd.DataFrame | None,
    rules: tuple[MarkerRule, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply independently configured rules to candidate classes by obs_name."""
    result = candidates.copy().reset_index(drop=True)
    if result.obs_name.duplicated().any():
        raise ValueError("Candidate obs_names must be unique.")
    if evidence is not None and (
        not evidence.index.is_unique or evidence.index.hasnans
    ):
        raise ValueError("Evidence must be uniquely indexed by obs_name.")
    result["support_score"] = 0.0
    weights = pd.Series(0.0, index=result.index)
    checks = []
    for number, rule in enumerate(rules):
        if rule.role == "feature_only":
            continue
        if evidence is None or rule.feature not in evidence:
            raise ValueError(
                f"Agreement feature {rule.feature!r} is missing; supply it or use feature_only."
            )
        selected = result.class_id.eq(rule.class_id)
        positions = result.index[selected]
        values = pd.to_numeric(
            evidence[rule.feature].reindex(result.loc[selected, "obs_name"]),
            errors="coerce",
        ).to_numpy()
        finite = np.isfinite(values)
        passed = finite.copy()
        if rule.minimum is not None:
            passed &= values >= rule.minimum
        if rule.maximum is not None:
            passed &= values <= rule.maximum
        accepted = passed | (~finite & (rule.missing == "ignore"))
        if rule.role == "required":
            failed = positions[~accepted]
            result.loc[failed, "eligible"] = False
            result.loc[failed, "selection_reason"] += f";required:{rule.feature}"
        else:
            # Missing supportive measurements receive no support. 'reject' is
            # explicitly about missingness; an observed non-match is only soft.
            if rule.missing == "reject":
                failed = positions[~finite]
                result.loc[failed, "eligible"] = False
                result.loc[failed, "selection_reason"] += f";missing:{rule.feature}"
            result.loc[positions, "support_score"] += passed * rule.weight
            weights.loc[positions] += finite * rule.weight
        checks.append(
            pd.DataFrame(
                {
                    "obs_name": result.loc[selected, "obs_name"].to_numpy(),
                    "class_id": rule.class_id,
                    "rule": number,
                    "feature": rule.feature,
                    "role": rule.role,
                    "value": values,
                    "available": finite,
                    "passed": passed,
                }
            )
        )
    result["support_score"] = result.support_score.div(
        weights.where(weights.gt(0))
    ).fillna(0)
    return result, pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()


def _allocate(capacities: np.ndarray, budget: int, mode: str, rng) -> np.ndarray:
    """Capped water filling; random tie order avoids alphabetical preference."""
    allocation: np.ndarray = np.zeros(len(capacities), dtype=int)
    budget = min(budget, int(capacities.sum()))
    while budget:
        active = np.flatnonzero(capacities > allocation)
        if mode == "balanced":
            count = min(budget, len(active))
            chosen = rng.permutation(active)[:count]
            allocation[chosen] += 1
            budget -= count
        else:
            room = capacities - allocation
            target = budget * room / room.sum()
            whole = np.minimum(np.floor(target).astype(int), room)
            if not whole.any():
                chosen = rng.choice(active, size=1, p=room[active] / room[active].sum())
                whole[chosen] = 1
            allocation += whole
            budget -= int(whole.sum())
    return allocation


def _leaf_allocations(frame, strata, budget, spec, rng):
    if not strata:
        cap = min(len(frame), spec.max_per_stratum or len(frame))
        return [(frame.index, min(cap, budget))]
    groups = [group for _, group in frame.groupby(strata[0], observed=True, sort=True)]
    capacities = []
    for group in groups:
        leaves = _leaf_allocations(
            group, strata[1:], len(group), spec, np.random.default_rng(0)
        )
        capacities.append(sum(count for _, count in leaves))
    allocation = _allocate(np.asarray(capacities), budget, spec.allocation, rng)
    leaves = []
    for group, count in zip(groups, allocation, strict=True):
        leaves.extend(_leaf_allocations(group, strata[1:], int(count), spec, rng))
    return leaves


def select_exemplars(
    candidates: pd.DataFrame,
    *,
    class_ids: list[str],
    sampling: SamplingSpec = SamplingSpec(),
    evidence=None,
    rules: tuple[MarkerRule, ...] = (),
) -> SelectionResult:
    """Select examples without replacement; report all eligibility and shortages.

    Custom source adapters can supply this candidate schema. Alternative samplers
    can return SelectionResult and use the same fitting/evaluation functions.
    """
    required = {
        "obs_name",
        "ROI",
        "ObjectNumber",
        "class_id",
        "eligible",
        "source_score",
        "selection_reason",
        "label_origin",
        *sampling.strata,
    }
    if not required.issubset(candidates):
        raise ValueError(
            f"Candidate table is missing: {sorted(required - set(candidates))}"
        )
    if any(rule.class_id not in class_ids for rule in rules):
        raise ValueError("Marker rules contain an unknown class.")
    if len(set(class_ids)) != len(class_ids) or len(class_ids) < 2:
        raise ValueError("At least two distinct classes are required.")
    if candidates.loc[candidates.eligible, "class_id"].isna().any():
        raise ValueError("Eligible candidates must have a class.")
    unknown = set(candidates.class_id.dropna().astype(str)) - set(class_ids)
    if unknown:
        raise ValueError(f"Unknown candidate classes: {sorted(unknown)}")
    for key in sampling.strata:
        if candidates[key].isna().any():
            raise ValueError(f"Sampling stratum {key!r} contains missing values.")
    ordered = candidates.sort_values("obs_name", kind="stable").reset_index(drop=True)
    result, checks = assess_markers(ordered, evidence, rules)
    if result.duplicated(["ROI", "ObjectNumber"]).any():
        raise ValueError("Candidate cell identities must be unique.")
    result["source_score"] = pd.to_numeric(result.source_score, errors="coerce")
    invalid = ~np.isfinite(result.source_score)
    result.loc[invalid, "eligible"] = False
    if sampling.minimum_score is not None:
        below = result.source_score.lt(sampling.minimum_score)
        result.loc[below, "eligible"] = False
        result.loc[below, "selection_reason"] += ";below_score_floor"
    result["score_rank"] = np.nan
    group_keys = ["class_id", *sampling.strata]
    # Stable within-stratum ranks accept negative or differently scaled scores.
    eligible = result.loc[result.eligible]
    ranks = eligible.groupby(group_keys, observed=True).source_score.rank(
        pct=True, method="average"
    )
    result.loc[ranks.index, "score_rank"] = ranks
    if sampling.candidate_fraction < 1:
        quality_keys = (
            ["class_id"] if sampling.quality_pool_scope == "class" else group_keys
        )
        thresholds = eligible.groupby(quality_keys, observed=True).source_score.transform(
            lambda values: values.quantile(1 - sampling.candidate_fraction)
        )
        below = eligible.index[eligible.source_score.lt(thresholds)]
        result.loc[below, "eligible"] = False
        result.loc[below, "selection_reason"] += ";outside_score_pool"
    result["sampling_weight"] = result.score_rank.pow(sampling.score_power).fillna(
        0
    ) * (1 + sampling.support_strength * result.support_score)
    result["selected"] = False
    rng = np.random.default_rng(sampling.seed)
    warnings = []
    for cls in class_ids:
        pool = result.loc[result.eligible & result.class_id.eq(cls)]
        for positions, count in _leaf_allocations(
            pool, list(sampling.strata), sampling.per_class, sampling, rng
        ):
            if not count:
                continue
            group = result.loc[positions]
            if sampling.strategy == "top_ranked":
                # Seeded tie order; supportive evidence participates in priority.
                shuffled = group.iloc[rng.permutation(len(group))]
                selected = shuffled.sort_values(
                    "sampling_weight", ascending=False, kind="stable"
                ).index[:count]
            else:
                weights = (
                    group.sampling_weight.to_numpy()
                    if sampling.strategy == "rank_weighted"
                    else np.ones(len(group))
                )
                weights = np.maximum(weights, np.finfo(float).tiny)
                selected = rng.choice(
                    positions.to_numpy(),
                    size=count,
                    replace=False,
                    p=weights / weights.sum(),
                )
            result.loc[selected, "selected"] = True
        selected = result.loc[result.selected & result.class_id.eq(cls)]
        groups = (
            selected[sampling.strata[0]].nunique()
            if sampling.strata
            else int(bool(len(selected)))
        )
        if (
            len(selected) < sampling.minimum_per_class
            or groups < sampling.minimum_groups_per_class
        ):
            warnings.append(
                f"{cls}: insufficient exemplars ({len(selected)} cells, {groups} top-level groups)."
            )
        if len(selected) < sampling.per_class:
            warnings.append(
                f"{cls}: selected {len(selected)}/{sampling.per_class}; eligibility/caps were not relaxed."
            )
    result.loc[result.selected, "selection_reason"] += ";selected"
    result["human_confirmed"] = False
    result["training_eligible"] = result.selected
    coverage = (
        result.groupby(group_keys, observed=True, dropna=False)
        .agg(
            available=("obs_name", "size"),
            eligible=("eligible", "sum"),
            selected=("selected", "sum"),
            median_source_score=("source_score", "median"),
        )
        .reset_index()
    )
    settings = {
        "sampling": sampling.model_dump(mode="json"),
        "marker_rules": [rule.model_dump(mode="json") for rule in rules],
    }
    return SelectionResult(
        result, result.loc[result.selected].copy(), coverage, checks, warnings, settings
    )
