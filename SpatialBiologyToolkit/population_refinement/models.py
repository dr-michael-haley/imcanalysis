"""Typed, serializable notebook policies for population refinement."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ClassSpec(Policy):
    class_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    source_labels: tuple[str, ...] = ()


class SplitSpec(Policy):
    population_key: str = Field(min_length=1)
    populations: tuple[str, ...] = Field(min_length=1)
    classes: tuple[ClassSpec, ...] = Field(min_length=2, max_length=8)
    roi_key: str = "ROI"
    object_key: str = "ObjectNumber"
    metadata_keys: tuple[str, ...] = ("Case",)

    @model_validator(mode="after")
    def unique_classes(self):
        ids = [item.class_id for item in self.classes]
        names = [item.name for item in self.classes]
        labels = [label for item in self.classes for label in item.source_labels]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("Class IDs and names must be unique.")
        if len(set(labels)) != len(labels):
            raise ValueError("A source label may belong to only one class.")
        return self

    @property
    def class_ids(self) -> list[str]:
        return [item.class_id for item in self.classes]


class MarkerRule(Policy):
    """One class-specific evidence rule; features can be obs or image measurements.

    Required rules gate eligibility. Supportive rules change sampling weights.
    Feature-only rules document a predictor without requiring agreement.
    A negative marker is expressed with an upper bound, not a separate algorithm.
    """

    class_id: str
    feature: str = Field(min_length=1)
    role: Literal["required", "supportive", "feature_only"] = "supportive"
    minimum: float | None = None
    maximum: float | None = None
    weight: float = Field(default=1.0, gt=0)
    missing: Literal["reject", "ignore"] = "reject"

    @model_validator(mode="after")
    def valid_bounds(self):
        if (
            self.role != "feature_only"
            and self.minimum is None
            and self.maximum is None
        ):
            raise ValueError("An agreement rule needs at least one bound.")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum must not exceed maximum.")
        return self


class SamplingSpec(Policy):
    """Hierarchical allocation followed by sampling within each leaf stratum.

    Score ranks are calculated within class and leaf stratum. They are priorities,
    not probabilities of label correctness. No absolute score cutoff is imposed.
    """

    strata: tuple[str, ...] = ("Case", "ROI")
    per_class: int = Field(default=500, ge=2)
    max_per_stratum: int | None = Field(default=50, ge=1)
    allocation: Literal["balanced", "proportional"] = "balanced"
    strategy: Literal["rank_weighted", "top_ranked", "uniform"] = "rank_weighted"
    score_power: float = Field(default=3.0, ge=0)
    candidate_fraction: float = Field(default=1.0, gt=0, le=1)
    # Class-wide filtering happens before stratification, so weak strata cannot
    # replenish the pool merely because their local best cells rank highly.
    quality_pool_scope: Literal["stratum", "class"] = "stratum"
    minimum_score: float | None = None
    support_strength: float = Field(default=2.0, ge=0)
    minimum_per_class: int = Field(default=20, ge=2)
    minimum_groups_per_class: int = Field(default=2, ge=1)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_sampling(self):
        if self.minimum_per_class > self.per_class:
            raise ValueError("minimum_per_class must not exceed per_class.")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("Sampling strata must be distinct.")
        return self


class AssignmentPolicy(Policy):
    assignment_mode: Literal["abstain", "complete"] = "abstain"
    minimum_probability: float = Field(default=0.8, ge=0, le=1)
    maximum_entropy: float = Field(default=0.7, ge=0, le=1)
    minimum_margin: float = Field(default=0.2, ge=0, le=1)
    minimum_feature_fraction: float = Field(default=0.8, gt=0, le=1)
    # Abstain when too many features fall outside training quantile bounds.
    maximum_outside_fraction: float = Field(default=0.5, ge=0, le=1)
    support_quantile: float = Field(default=0.01, ge=0, lt=0.5)
