"""Identity-aligned source adapters. No adapter changes the input AnnData."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse

from SpatialBiologyToolkit.napari_sbt.cohort import resolve_cohort

from .models import SplitSpec


class ExemplarSource(Protocol):
    def candidates(self, adata, spec: SplitSpec) -> pd.DataFrame:
        """Return one identity-aligned candidate row per target cell."""
        ...


def resolve_split_cohort(adata, spec: SplitSpec) -> pd.DataFrame:
    preview = resolve_cohort(
        adata,
        roi_obs=spec.roi_key,
        object_id_obs=spec.object_key,
        mode="obs_values",
        obs_column=spec.population_key,
        obs_values=spec.populations,
    )
    cohort = preview.eligible_cells
    for key in spec.metadata_keys:
        if key in cohort:
            continue
        if key not in adata.obs:
            raise ValueError(f"Missing metadata column: {key}")
        cohort[key] = adata.obs.loc[cohort.obs_name, key].to_numpy()
    return cohort


def _aligned(adata, table: pd.DataFrame | None) -> pd.DataFrame:
    source = adata.obs if table is None else table
    if not source.index.is_unique or source.index.hasnans:
        raise ValueError("Source index must contain unique, non-missing obs_names.")
    source = source.copy()
    source.index = source.index.astype(str)
    if not source.index.is_unique:
        raise ValueError("Source IDs collide after string conversion.")
    if not len(source.index.intersection(adata.obs_names)):
        raise ValueError("Source and AnnData have no overlapping obs_names.")
    return source


class MaxFuseSource:
    """Consume existing transfers indexed by obs_name, including partial overlap."""

    def __init__(self, *, label_key: str, score_key: str, table=None, name="maxfuse"):
        self.label_key = label_key
        self.score_key = score_key
        self.table = table
        self.name = name

    def candidates(self, adata, spec: SplitSpec) -> pd.DataFrame:
        source = _aligned(adata, self.table)
        for key in (self.label_key, self.score_key):
            if key not in source:
                raise ValueError(f"Source is missing column {key!r}.")
        mapping = {
            label: item.class_id
            for item in spec.classes
            for label in item.source_labels
        }
        if not mapping:
            raise ValueError("MaxFuse classes require explicit source_labels.")
        result = resolve_split_cohort(adata, spec)
        aligned = source.reindex(result.obs_name)
        result["source_label"] = aligned[self.label_key].astype("string").to_numpy()
        result["source_score"] = pd.to_numeric(
            aligned[self.score_key], errors="coerce"
        ).to_numpy()
        result["class_id"] = result.source_label.map(mapping)
        result["label_origin"] = self.name
        result["source_present"] = result.obs_name.isin(source.index)
        result["eligible"] = result.class_id.notna() & np.isfinite(result.source_score)
        result["selection_reason"] = np.select(
            [
                ~result.source_present,
                result.source_label.isna(),
                result.class_id.isna(),
                ~np.isfinite(result.source_score),
            ],
            ["no_source_row", "unmatched", "outside_selected_classes", "missing_score"],
            default="candidate",
        )
        return result


class TableSource:
    """Manual/custom AnnData sampling adapter, with class IDs supplied explicitly."""

    def __init__(
        self, table: pd.DataFrame, *, class_key="class_id", score_key=None, name="table"
    ):
        self.table, self.class_key, self.score_key, self.name = (
            table,
            class_key,
            score_key,
            name,
        )

    def candidates(self, adata, spec: SplitSpec) -> pd.DataFrame:
        source = _aligned(adata, self.table)
        if self.class_key not in source:
            raise ValueError(f"Missing class column: {self.class_key}")
        unknown = set(source[self.class_key].dropna().astype(str)) - set(spec.class_ids)
        if unknown:
            raise ValueError(f"Unknown supplied class IDs: {sorted(unknown)}")
        identity_spec = spec.model_copy(
            update={
                "classes": tuple(
                    item.model_copy(update={"source_labels": (item.class_id,)})
                    for item in spec.classes
                )
            }
        )
        if self.score_key is None:
            source = source.assign(_uniform_score=1.0)
        return MaxFuseSource(
            label_key=self.class_key,
            score_key=self.score_key or "_uniform_score",
            table=source,
            name=self.name,
        ).candidates(adata, identity_spec)


def expression_evidence(
    adata, markers: list[str], *, layer: str | None = None
) -> pd.DataFrame:
    """Extract selected measured markers; use an explicit uncorrected representation."""
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("AnnData observation and variable names must be unique.")
    missing = set(markers) - set(adata.var_names)
    if missing:
        raise ValueError(f"Markers absent from AnnData: {sorted(missing)}")
    selected = adata[:, markers]
    values = selected.X if layer is None else selected.layers[layer]
    if sparse.issparse(values):
        values = values.toarray()
    return pd.DataFrame(np.asarray(values), index=adata.obs_names, columns=markers)


def rank_reference_markers(
    reference,
    spec: SplitSpec,
    *,
    label_key: str,
    mapping: pd.DataFrame,
    available_channels: list[str],
    gene_key="snRNAseq",
    channel_key="IMC",
    layer=None,
) -> pd.DataFrame:
    """Rank panel-mapped RNA contrasts between selected classes, without p-values.

    Pass unique reference cells, not repeated rows from a matched IMC/RNA table.
    This proposes markers; measured protein/image agreement remains separate.
    """
    if not reference.obs_names.is_unique:
        raise ValueError("Reference cells must be unique before marker ranking.")
    pairs = mapping[[gene_key, channel_key]].drop_duplicates()
    pairs = pairs.loc[
        pairs[gene_key].isin(reference.var_names)
        & pairs[channel_key].isin(available_channels)
    ]
    if pairs.empty:
        raise ValueError("No reference genes map to available image channels.")
    label_map = {
        label: item.class_id for item in spec.classes for label in item.source_labels
    }
    labels = reference.obs[label_key].astype("string").map(label_map)
    genes = list(dict.fromkeys(pairs[gene_key]))
    data = expression_evidence(reference, genes, layer=layer)
    means, variances, fractions, counts = {}, {}, {}, {}
    for cls in spec.class_ids:
        group = data.loc[labels.eq(cls)]
        if len(group) < 2:
            raise ValueError(f"Class {cls} needs at least two unique reference cells.")
        means[cls], variances[cls] = group.mean(), group.var()
        fractions[cls], counts[cls] = group.gt(0).mean(), len(group)
    rows = []
    for cls in spec.class_ids:
        for competitor in spec.class_ids:
            if competitor == cls:
                continue
            for gene, channel in pairs.itertuples(index=False, name=None):
                difference = means[cls][gene] - means[competitor][gene]
                scale = np.sqrt(
                    (variances[cls][gene] + variances[competitor][gene]) / 2
                )
                rows.append(
                    dict(
                        class_id=cls,
                        competitor=competitor,
                        gene=gene,
                        channel=channel,
                        mean_difference=difference,
                        standardized_difference=difference / max(scale, 1e-8),
                        detected_fraction=fractions[cls][gene],
                        reference_cells=counts[cls],
                        role="candidate_only",
                    )
                )
    return (
        pd.DataFrame(rows)
        .sort_values("standardized_difference", ascending=False)
        .reset_index(drop=True)
    )
