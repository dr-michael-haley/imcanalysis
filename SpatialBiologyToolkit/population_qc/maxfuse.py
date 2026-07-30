"""Read-only MaxFuse label-transfer evidence for population assessment.

The helpers accept transferred annotations already embedded in an AnnData
table or supplied as annotation-only H5AD/CSV sidecars. External rows are
aligned by observation identity (or explicit composite keys), never position.
They summarize match coverage separately from transferred-label consensus and
do not modify the source object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from ._utils import ordered_labels, resolve_table
from .models import (
    MaxFuseEvidenceResult,
    MaxFuseInputAudit,
    MaxFuseSourceSpec,
    PlotResult,
)


_SCORE_PATTERN = re.compile(r"^(?P<prefix>.+)_maxfuse_score$", re.IGNORECASE)
_DEFAULT_SENSITIVITY_THRESHOLDS = (0.0, 0.5, 0.7, 0.85, 0.9)
_SOURCE_COLUMNS = (
    "source",
    "input_location",
    "input_path",
    "score_column",
    "score_threshold",
    "thresholded",
    "label_column",
    "label_role",
    "reference_path",
    "shared_proteins",
    "shared_genes",
)
_ALIGNMENT_COLUMNS = (
    "source",
    "input_location",
    "input_path",
    "join_method",
    "join_keys",
    "target_cells",
    "source_rows",
    "overlap_cells",
    "target_coverage",
    "unused_source_rows",
    "same_order",
)


@dataclass
class _LoadedInput:
    location: str
    path: Path | None
    obs: pd.DataFrame
    uns: Mapping[str, Any]


def _as_path_list(
    paths: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if paths is None:
        return []
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if path in resolved:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"MaxFuse result file not found: {path}")
        resolved.append(path)
    return resolved


def _copy_uns_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_h5ad_input(path: Path) -> _LoadedInput:
    try:
        import anndata as ad
    except ImportError as error:  # pragma: no cover - scientific environment guard
        raise ImportError("anndata is required to read MaxFuse H5AD results") from error
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs.copy()
        obs.index = pd.Index(adata.obs_names.astype(str), name=adata.obs_names.name)
        uns = dict(adata.uns)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    return _LoadedInput(location="external", path=path, obs=obs, uns=uns)


def _read_csv_input(path: Path) -> _LoadedInput:
    frame = pd.read_csv(path, low_memory=False)
    index_column = next(
        (
            column
            for column in ("obs_name", "imc_obs_name")
            if column in frame.columns
        ),
        None,
    )
    if index_column is None and len(frame.columns):
        first = str(frame.columns[0])
        if first.startswith("Unnamed:"):
            index_column = frame.columns[0]
    if index_column is not None:
        frame = frame.set_index(index_column, drop=True)
        frame.index = pd.Index(frame.index.astype(str), name="obs_name")
    return _LoadedInput(location="external", path=path, obs=frame, uns={})


def _load_external(path: Path) -> _LoadedInput:
    lower = path.name.lower()
    if lower.endswith(".h5ad"):
        return _read_h5ad_input(path)
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return _read_csv_input(path)
    raise ValueError(
        "MaxFuse result paths must end in .h5ad, .csv, or .csv.gz: "
        f"{path}"
    )


def _infer_label_role(column: str) -> str:
    token = str(column).casefold()
    if any(
        value in token
        for value in (
            "state",
            "status",
            "neftel",
            "activation",
            "programme",
            "program",
        )
    ):
        return "state"
    if any(
        value in token
        for value in ("granular", "fine", "subtype", "level_4", "level4")
    ):
        return "subtype"
    if any(
        value in token
        for value in (
            "coarse",
            "broad",
            "lineage",
            "cell_type",
            "celltype",
            "level_3",
            "level3",
        )
    ):
        return "lineage"
    return "other"


def _metadata_threshold(uns: Mapping[str, Any], source: str) -> float | None:
    manifest = _copy_uns_mapping(uns.get("maxfuse"))
    sources = manifest.get("sources")
    if isinstance(sources, Mapping):
        entry = _copy_uns_mapping(sources.get(source))
        value = entry.get("score_threshold")
        if value is not None:
            return float(value)
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        for raw in sources:
            entry = _copy_uns_mapping(raw)
            if str(entry.get("name", "")) == source:
                value = entry.get("score_threshold")
                if value is not None:
                    return float(value)
    value = uns.get("score_threshold_used_for_confusion_matrices")
    return None if value is None else float(value)


def _metadata_specs(
    loaded: _LoadedInput,
    *,
    score_threshold: float | None,
) -> list[MaxFuseSourceSpec]:
    manifest = _copy_uns_mapping(loaded.uns.get("maxfuse"))
    raw_sources = manifest.get("sources")
    if isinstance(raw_sources, Mapping):
        values = [
            {"name": name, **dict(_copy_uns_mapping(value))}
            for name, value in raw_sources.items()
        ]
    elif isinstance(raw_sources, Sequence) and not isinstance(
        raw_sources, (str, bytes)
    ):
        values = list(raw_sources)
    else:
        return []
    specs: list[MaxFuseSourceSpec] = []
    for raw in values:
        value = dict(_copy_uns_mapping(raw))
        if not value:
            continue
        value.setdefault("path", None if loaded.path is None else str(loaded.path))
        if score_threshold is not None:
            value["score_threshold"] = score_threshold
        specs.append(MaxFuseSourceSpec.from_value(value))
    return specs


def _auto_specs(
    loaded: _LoadedInput,
    *,
    score_threshold: float | None,
) -> list[MaxFuseSourceSpec]:
    metadata_specs = _metadata_specs(loaded, score_threshold=score_threshold)
    if metadata_specs:
        return metadata_specs
    specs: list[MaxFuseSourceSpec] = []
    columns = list(map(str, loaded.obs.columns))
    for column in columns:
        match = _SCORE_PATTERN.match(column)
        if match is None:
            continue
        prefix = match.group("prefix")
        label_columns = [
            candidate
            for candidate in columns
            if candidate.startswith(f"{prefix}_")
            and candidate != column
            and _SCORE_PATTERN.match(candidate) is None
        ]
        if not label_columns:
            continue
        threshold = (
            score_threshold
            if score_threshold is not None
            else _metadata_threshold(loaded.uns, prefix)
        )
        specs.append(
            MaxFuseSourceSpec(
                name=prefix,
                score_column=column,
                label_columns=tuple(label_columns),
                label_roles={
                    value: _infer_label_role(value) for value in label_columns
                },
                path=None if loaded.path is None else str(loaded.path),
                score_threshold=threshold,
            )
        )
    return specs


def _reference_path(uns: Mapping[str, Any], source: str) -> str:
    paths = _copy_uns_mapping(uns.get("rna_reference_paths"))
    value = paths.get(source, "")
    return "" if value is None else str(value)


def _shared_count(uns: Mapping[str, Any], key: str) -> int | None:
    value = uns.get(key)
    if value is None:
        return None
    try:
        return int(len(value))
    except TypeError:
        return None


def _validate_columns(frame: pd.DataFrame, spec: MaxFuseSourceSpec) -> None:
    requested = [
        *(tuple() if spec.score_column is None else (spec.score_column,)),
        *spec.label_columns,
        *spec.join_keys,
    ]
    if spec.obs_name_column is not None:
        requested.append(spec.obs_name_column)
    missing = [column for column in requested if column not in frame.columns]
    if missing:
        location = spec.path or "the selected AnnData table"
        raise KeyError(
            f"MaxFuse source {spec.name!r} is missing columns in {location}: {missing}"
        )


def _key_index(frame: pd.DataFrame, keys: Sequence[str], *, label: str) -> pd.Index:
    key_frame = frame.loc[:, list(keys)]
    if key_frame.isna().any(axis=None):
        raise ValueError(f"{label} contains missing values in join keys {list(keys)}")
    tuples = pd.MultiIndex.from_frame(key_frame, names=list(keys))
    if not tuples.is_unique:
        raise ValueError(f"{label} contains duplicate composite join keys {list(keys)}")
    return tuples


def _align_source(
    target_obs: pd.DataFrame,
    target_obs_names: pd.Index,
    loaded: _LoadedInput,
    spec: MaxFuseSourceSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_columns(loaded.obs, spec)
    requested = [
        *(tuple() if spec.score_column is None else (spec.score_column,)),
        *spec.label_columns,
    ]
    source = loaded.obs.loc[:, list(dict.fromkeys(requested))].copy()
    if loaded.location == "embedded":
        aligned = source.copy()
        aligned.index = target_obs_names
        return aligned, {
            "source": spec.name,
            "input_location": loaded.location,
            "input_path": "",
            "join_method": "embedded",
            "join_keys": "",
            "target_cells": len(target_obs_names),
            "source_rows": len(source),
            "overlap_cells": len(target_obs_names),
            "target_coverage": 1.0,
            "unused_source_rows": 0,
            "same_order": True,
        }

    if spec.join_keys:
        target_index = _key_index(
            target_obs,
            spec.join_keys,
            label="Target AnnData",
        )
        source_index = _key_index(
            loaded.obs,
            spec.join_keys,
            label=f"MaxFuse source {spec.name!r}",
        )
        join_method = "composite"
        join_keys = ";".join(spec.join_keys)
    else:
        if not target_obs_names.is_unique:
            raise ValueError("Target adata.obs_names must be unique for MaxFuse alignment")
        target_index = pd.Index(target_obs_names.astype(str), name="obs_name")
        if spec.obs_name_column is not None:
            raw_source_index = loaded.obs[spec.obs_name_column]
        else:
            raw_source_index = loaded.obs.index
        source_index = pd.Index(
            pd.Series(raw_source_index, dtype=object).astype(str),
            name="obs_name",
        )
        if not source_index.is_unique:
            raise ValueError(
                f"MaxFuse source {spec.name!r} contains duplicate observation identifiers"
            )
        join_method = "obs_name"
        join_keys = spec.obs_name_column or "index"

    overlap = target_index.intersection(source_index)
    if not len(overlap):
        raise ValueError(
            f"MaxFuse source {spec.name!r} has no observation identities in common "
            "with the selected AnnData table"
        )
    same_order = bool(
        len(target_index) == len(source_index)
        and np.array_equal(target_index.to_numpy(), source_index.to_numpy())
    )
    source.index = source_index
    aligned = source.reindex(target_index)
    aligned.index = target_obs_names
    return aligned, {
        "source": spec.name,
        "input_location": loaded.location,
        "input_path": "" if loaded.path is None else str(loaded.path),
        "join_method": join_method,
        "join_keys": join_keys,
        "target_cells": len(target_index),
        "source_rows": len(source_index),
        "overlap_cells": len(overlap),
        "target_coverage": len(overlap) / max(1, len(target_index)),
        "unused_source_rows": len(source_index) - len(overlap),
        "same_order": same_order,
    }


def _deduplicate_names(
    specifications: list[tuple[MaxFuseSourceSpec, _LoadedInput]],
) -> list[tuple[MaxFuseSourceSpec, _LoadedInput]]:
    counts: dict[str, int] = {}
    output: list[tuple[MaxFuseSourceSpec, _LoadedInput]] = []
    for spec, loaded in specifications:
        count = counts.get(spec.name, 0)
        counts[spec.name] = count + 1
        if count:
            suffix = loaded.path.stem if loaded.path is not None else "embedded"
            candidate = f"{spec.name}@{suffix}"
            serial = 2
            existing = {item.name for item, _ in output}
            while candidate in existing:
                candidate = f"{spec.name}@{suffix}_{serial}"
                serial += 1
            spec = replace(spec, name=candidate)
        output.append((spec, loaded))
    return output


def _resolve_source_inputs(
    adata: Any,
    *,
    paths: str | Path | Sequence[str | Path] | None,
    source_specs: Sequence[MaxFuseSourceSpec | Mapping[str, Any]] | None,
    score_threshold: float | None,
) -> list[tuple[MaxFuseSourceSpec, _LoadedInput]]:
    embedded = _LoadedInput(
        location="embedded",
        path=None,
        obs=adata.obs,
        uns=_copy_uns_mapping(getattr(adata, "uns", {})),
    )
    external_inputs = [_load_external(path) for path in _as_path_list(paths)]
    all_inputs = [embedded, *external_inputs]
    paired_specs: list[tuple[MaxFuseSourceSpec, _LoadedInput]] = []
    if source_specs is None:
        for loaded in all_inputs:
            paired_specs.extend(
                (spec, loaded)
                for spec in _auto_specs(
                    loaded,
                    score_threshold=score_threshold,
                )
            )
        for loaded in external_inputs:
            if not any(candidate is loaded for _, candidate in paired_specs):
                raise ValueError(
                    "No MaxFuse score/label columns were detected in "
                    f"{loaded.path}; provide source_specs for non-standard columns"
                )
    else:
        normalized = [MaxFuseSourceSpec.from_value(value) for value in source_specs]
        available_by_path = {
            str(item.path): item for item in external_inputs if item.path is not None
        }
        for spec in normalized:
            if spec.path is None:
                loaded = embedded
            else:
                path = Path(spec.path).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"MaxFuse result file not found: {path}")
                loaded = available_by_path.get(str(path))
                if loaded is None:
                    loaded = _load_external(path)
                    available_by_path[str(path)] = loaded
            if score_threshold is not None:
                spec = replace(spec, score_threshold=float(score_threshold))
            paired_specs.append((spec, loaded))
    return _deduplicate_names(paired_specs)


def _inventory_rows(
    spec: MaxFuseSourceSpec,
    loaded: _LoadedInput,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for label_column in spec.label_columns:
        role = spec.label_roles.get(label_column, _infer_label_role(label_column))
        if role == "other":
            warnings.append(
                f"MaxFuse label field {label_column!r} has role 'other'; "
                "set an explicit lineage/subtype/state role before naming populations"
            )
        rows.append(
            {
                "source": spec.name,
                "input_location": loaded.location,
                "input_path": "" if loaded.path is None else str(loaded.path),
                "score_column": spec.score_column or "",
                "score_threshold": spec.score_threshold,
                "thresholded": spec.score_threshold is not None,
                "label_column": label_column,
                "label_role": role,
                "reference_path": _reference_path(loaded.uns, spec.name),
                "shared_proteins": _shared_count(loaded.uns, "shared_proteins"),
                "shared_genes": _shared_count(loaded.uns, "shared_genes"),
            }
        )
    if spec.score_threshold is None:
        warnings.append(
            f"MaxFuse source {spec.name!r} has no score threshold; summaries "
            "will use all matched cells and must be treated as unthresholded"
        )
    return rows, warnings


def inspect_maxfuse_inputs(
    data: Any,
    *,
    table_name: str | None = None,
    paths: str | Path | Sequence[str | Path] | None = None,
    source_specs: Sequence[MaxFuseSourceSpec | Mapping[str, Any]] | None = None,
    score_threshold: float | None = None,
    require: bool = False,
) -> MaxFuseInputAudit:
    """Audit MaxFuse sources and cell identity alignment without summarizing labels.

    Use this before freezing Prior v1. The returned audit exposes provenance,
    fields, thresholds, and overlap but no population-to-label associations.
    """

    selected_table, adata = resolve_table(data, table_name)
    if not adata.obs_names.is_unique:
        raise ValueError("Target adata.obs_names must be unique for MaxFuse evidence")
    if score_threshold is not None and not np.isfinite(float(score_threshold)):
        raise ValueError("score_threshold must be finite or None")
    paired_specs = _resolve_source_inputs(
        adata,
        paths=paths,
        source_specs=source_specs,
        score_threshold=score_threshold,
    )
    if not paired_specs:
        if require:
            raise ValueError(
                "No MaxFuse sources were found in the selected table and no "
                "compatible external result was supplied"
            )
        return MaxFuseInputAudit(
            sources=pd.DataFrame(columns=_SOURCE_COLUMNS),
            alignment_audit=pd.DataFrame(columns=_ALIGNMENT_COLUMNS),
            warnings=("No MaxFuse label-transfer evidence was supplied or detected",),
            parameters={
                "table_name": selected_table,
                "source_data_modified": False,
                "population_label_associations_summarized": False,
            },
        )
    target_names = pd.Index(adata.obs_names.astype(str), name="obs_name")
    source_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for spec, loaded in paired_specs:
        _, audit = _align_source(adata.obs, target_names, loaded, spec)
        alignment_rows.append(audit)
        inventory, inventory_warnings = _inventory_rows(spec, loaded)
        source_rows.extend(inventory)
        warnings.extend(inventory_warnings)
        if audit["target_coverage"] < 1.0:
            warnings.append(
                f"MaxFuse source {spec.name!r} covers "
                f"{audit['target_coverage']:.1%} of target observations"
            )
    return MaxFuseInputAudit(
        sources=pd.DataFrame(source_rows, columns=_SOURCE_COLUMNS),
        alignment_audit=pd.DataFrame(alignment_rows, columns=_ALIGNMENT_COLUMNS),
        warnings=tuple(dict.fromkeys(warnings)),
        parameters={
            "table_name": selected_table,
            "source_data_modified": False,
            "population_label_associations_summarized": False,
        },
    )


def _population_sizes(labels: pd.Series) -> tuple[list[str], dict[str, int]]:
    populations = ordered_labels(labels)
    as_strings = labels.astype("string")
    sizes = {
        population: int(as_strings.eq(population).sum()) for population in populations
    }
    return populations, sizes


def _coverage_row(
    *,
    source: str,
    population: str,
    population_cells: int,
    matched: pd.Series,
    evaluated: pd.Series,
    scores: pd.Series,
    threshold: float | None,
) -> dict[str, Any]:
    matched_cells = int(matched.sum())
    evaluated_cells = int(evaluated.sum())
    evaluated_scores = scores.loc[evaluated & scores.notna()]
    return {
        "source": source,
        "population": population,
        "population_cells": int(population_cells),
        "matched_cells": matched_cells,
        "matched_coverage": matched_cells / max(1, int(population_cells)),
        "evaluated_cells": evaluated_cells,
        "evaluated_coverage": evaluated_cells / max(1, int(population_cells)),
        "score_threshold": threshold,
        "thresholded": threshold is not None,
        "median_score": (
            float(evaluated_scores.median()) if len(evaluated_scores) else np.nan
        ),
    }


def _normalized_entropy(fractions: pd.Series) -> float:
    values = fractions.to_numpy(dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    return float(-(values * np.log(values)).sum() / np.log(len(values)))


def _label_rows(
    *,
    source: str,
    population: str,
    population_cells: int,
    label_column: str,
    label_role: str,
    labels: pd.Series,
    eligible: pd.Series,
    threshold: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = labels.loc[eligible & labels.notna()].astype(str)
    counts = selected.value_counts(sort=True)
    labelled_cells = int(counts.sum())
    fractions = counts / max(1, labelled_cells)
    distribution: list[dict[str, Any]] = []
    for rank, (label, count) in enumerate(counts.items(), start=1):
        distribution.append(
            {
                "source": source,
                "population": population,
                "label_column": label_column,
                "label_role": label_role,
                "label": str(label),
                "rank": rank,
                "cells": int(count),
                "fraction_of_labelled": float(count / max(1, labelled_cells)),
                "fraction_of_population": float(count / max(1, population_cells)),
                "population_cells": int(population_cells),
                "labelled_cells": labelled_cells,
                "score_threshold": threshold,
                "thresholded": threshold is not None,
            }
        )
    top_label = "" if counts.empty else str(counts.index[0])
    second_label = "" if len(counts) < 2 else str(counts.index[1])
    top_fraction = np.nan if counts.empty else float(fractions.iloc[0])
    second_fraction = 0.0 if len(counts) < 2 else float(fractions.iloc[1])
    summary = {
        "source": source,
        "population": population,
        "label_column": label_column,
        "label_role": label_role,
        "population_cells": int(population_cells),
        "labelled_cells": labelled_cells,
        "labelled_coverage": labelled_cells / max(1, int(population_cells)),
        "top_label": top_label,
        "top_label_cells": 0 if counts.empty else int(counts.iloc[0]),
        "top_label_fraction": top_fraction,
        "second_label": second_label,
        "second_label_fraction": second_fraction,
        "top2_margin": (
            np.nan if counts.empty else float(top_fraction - second_fraction)
        ),
        "normalized_entropy": (
            np.nan if counts.empty else _normalized_entropy(fractions)
        ),
        "score_threshold": threshold,
        "thresholded": threshold is not None,
    }
    return distribution, summary


def summarize_maxfuse_evidence(
    data: Any,
    population_key: str,
    *,
    table_name: str | None = None,
    paths: str | Path | Sequence[str | Path] | None = None,
    source_specs: Sequence[MaxFuseSourceSpec | Mapping[str, Any]] | None = None,
    score_threshold: float | None = None,
    sensitivity_thresholds: Sequence[float] = _DEFAULT_SENSITIVITY_THRESHOLDS,
    require: bool = False,
) -> MaxFuseEvidenceResult:
    """Discover, align, and summarize MaxFuse label transfers.

    Sources are auto-detected from ``<source>_maxfuse_score`` columns unless
    explicit specifications are supplied. Embedded and external sources may be
    combined. External H5AD/CSV files are aligned by observation name by
    default; use ``join_keys`` in an explicit source specification for a
    deliberate composite-key join.

    No source AnnData or SpatialData object is modified.
    """

    selected_table, adata = resolve_table(data, table_name)
    if population_key not in adata.obs:
        raise KeyError(f"Population column {population_key!r} is missing from the table")
    if not adata.obs_names.is_unique:
        raise ValueError("Target adata.obs_names must be unique for MaxFuse evidence")
    if score_threshold is not None and not np.isfinite(float(score_threshold)):
        raise ValueError("score_threshold must be finite or None")
    thresholds = sorted(
        {
            float(value)
            for value in sensitivity_thresholds
            if np.isfinite(float(value))
        }
    )
    target_obs = adata.obs
    target_names = pd.Index(adata.obs_names.astype(str), name="obs_name")
    paired_specs = _resolve_source_inputs(
        adata,
        paths=paths,
        source_specs=source_specs,
        score_threshold=score_threshold,
    )
    if not paired_specs:
        if require:
            raise ValueError(
                "No MaxFuse sources were found in the selected table and no "
                "compatible external result was supplied"
            )
        return MaxFuseEvidenceResult(
            population_key=population_key,
            sources=pd.DataFrame(columns=_SOURCE_COLUMNS),
            alignment_audit=pd.DataFrame(columns=_ALIGNMENT_COLUMNS),
            source_summary=pd.DataFrame(),
            population_summary=pd.DataFrame(),
            label_distribution=pd.DataFrame(),
            threshold_sensitivity=pd.DataFrame(),
            warnings=("No MaxFuse label-transfer evidence was supplied or detected",),
            parameters={
                "table_name": selected_table,
                "source_data_modified": False,
            },
            _cell_evidence=pd.DataFrame(index=target_names),
        )

    warnings: list[str] = []
    source_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    source_summary_rows: list[dict[str, Any]] = []
    population_summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    label_summary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    cell_evidence = pd.DataFrame(index=target_names)
    population_values = target_obs[population_key].astype("string")
    populations, population_sizes = _population_sizes(target_obs[population_key])

    for spec, loaded in paired_specs:
        aligned, audit = _align_source(target_obs, target_names, loaded, spec)
        alignment_rows.append(audit)
        if audit["target_coverage"] < 1.0:
            warnings.append(
                f"MaxFuse source {spec.name!r} covers "
                f"{audit['target_coverage']:.1%} of target observations"
            )
        threshold = spec.score_threshold
        if spec.score_column is None:
            scores = pd.Series(np.nan, index=target_names, dtype=float)
        else:
            raw_scores = aligned[spec.score_column]
            scores = pd.to_numeric(raw_scores, errors="coerce")
            invalid_scores = int(raw_scores.notna().sum() - scores.notna().sum())
            if invalid_scores:
                warnings.append(
                    f"MaxFuse source {spec.name!r} contained {invalid_scores} "
                    "non-numeric matching scores, which were treated as missing"
                )
        labels = aligned.loc[:, list(spec.label_columns)]
        matched = scores.notna() | labels.notna().any(axis=1)
        evaluated = matched if threshold is None else scores.ge(threshold)
        score_key = f"{spec.name}::score"
        cell_evidence[score_key] = scores.to_numpy()
        for label_column in spec.label_columns:
            role = spec.label_roles.get(label_column, _infer_label_role(label_column))
            cell_evidence[f"{spec.name}::{label_column}"] = aligned[
                label_column
            ].to_numpy()
        inventory, inventory_warnings = _inventory_rows(spec, loaded)
        source_rows.extend(inventory)
        warnings.extend(inventory_warnings)

        source_summary_rows.append(
            _coverage_row(
                source=spec.name,
                population="__all__",
                population_cells=len(target_names),
                matched=matched,
                evaluated=evaluated,
                scores=scores,
                threshold=threshold,
            )
        )
        for population in populations:
            population_mask = population_values.eq(population).fillna(False)
            population_summary_rows.append(
                _coverage_row(
                    source=spec.name,
                    population=population,
                    population_cells=population_sizes[population],
                    matched=matched & population_mask,
                    evaluated=evaluated & population_mask,
                    scores=scores,
                    threshold=threshold,
                )
            )
            for label_column in spec.label_columns:
                role = spec.label_roles.get(
                    label_column,
                    _infer_label_role(label_column),
                )
                distribution, label_summary = _label_rows(
                    source=spec.name,
                    population=population,
                    population_cells=population_sizes[population],
                    label_column=label_column,
                    label_role=role,
                    labels=aligned[label_column],
                    eligible=evaluated & population_mask,
                    threshold=threshold,
                )
                distribution_rows.extend(distribution)
                label_summary_rows.append(label_summary)

        if spec.score_column is not None:
            effective_thresholds = sorted(
                set(thresholds)
                | (set() if threshold is None else {float(threshold)})
            )
            groups = [("__all__", pd.Series(True, index=target_names))]
            groups.extend(
                (
                    population,
                    population_values.eq(population).fillna(False),
                )
                for population in populations
            )
            for population, group_mask in groups:
                denominator = int(group_mask.sum())
                for candidate_threshold in effective_thresholds:
                    passing = group_mask & scores.ge(candidate_threshold)
                    sensitivity_rows.append(
                        {
                            "source": spec.name,
                            "population": population,
                            "score_threshold": candidate_threshold,
                            "population_cells": denominator,
                            "passing_cells": int(passing.sum()),
                            "passing_coverage": int(passing.sum())
                            / max(1, denominator),
                            "is_primary_threshold": (
                                threshold is not None
                                and np.isclose(candidate_threshold, threshold)
                            ),
                        }
                    )

    population_summary = pd.DataFrame(population_summary_rows)
    label_summary = pd.DataFrame(label_summary_rows)
    if not label_summary.empty:
        population_summary = population_summary.merge(
            label_summary,
            on=[
                "source",
                "population",
                "population_cells",
                "score_threshold",
                "thresholded",
            ],
            how="right",
            validate="one_to_many",
        )
    return MaxFuseEvidenceResult(
        population_key=population_key,
        sources=pd.DataFrame(source_rows, columns=_SOURCE_COLUMNS),
        alignment_audit=pd.DataFrame(alignment_rows, columns=_ALIGNMENT_COLUMNS),
        source_summary=pd.DataFrame(source_summary_rows),
        population_summary=population_summary,
        label_distribution=pd.DataFrame(distribution_rows),
        threshold_sensitivity=pd.DataFrame(sensitivity_rows),
        warnings=tuple(dict.fromkeys(warnings)),
        parameters={
            "table_name": selected_table,
            "sensitivity_thresholds": thresholds,
            "source_data_modified": False,
            "score_semantics": "MaxFuse matching similarity; higher is better; not a calibrated probability",
        },
        _cell_evidence=cell_evidence,
    )


def plot_maxfuse_label_heatmap(
    result: MaxFuseEvidenceResult,
    *,
    source: str,
    label_column: str,
    metric: str = "fraction_of_labelled",
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> PlotResult:
    """Plot population-by-transferred-label composition for one reference."""

    if metric not in {"fraction_of_labelled", "fraction_of_population", "cells"}:
        raise ValueError(
            "metric must be fraction_of_labelled, fraction_of_population, or cells"
        )
    frame = result.label_distribution
    frame = frame.loc[
        frame["source"].astype(str).eq(str(source))
        & frame["label_column"].astype(str).eq(str(label_column))
    ].copy()
    if frame.empty:
        raise KeyError(
            f"No MaxFuse evidence found for source={source!r}, "
            f"label_column={label_column!r}"
        )
    matrix = frame.pivot(
        index="population",
        columns="label",
        values=metric,
    ).fillna(0.0)
    import matplotlib.pyplot as plt
    import seaborn as sns

    if figsize is None:
        figsize = (
            max(8.0, min(24.0, 0.45 * matrix.shape[1] + 5.0)),
            max(4.0, min(24.0, 0.4 * matrix.shape[0] + 2.5)),
        )
    figure, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap="viridis",
        annot=annotate,
        fmt=".2f" if metric != "cells" else ".0f",
        cbar_kws={"label": metric.replace("_", " ")},
        ax=ax,
    )
    ax.set_xlabel("Transferred label")
    ax.set_ylabel(result.population_key)
    ax.set_title(title or f"{source}: {label_column}")
    figure.tight_layout()
    return PlotResult(figure=figure, axes=ax, data=frame, display_data=matrix)


def plot_maxfuse_threshold_sensitivity(
    result: MaxFuseEvidenceResult,
    *,
    population: Any | None = None,
    sources: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    title: str | None = None,
) -> PlotResult:
    """Plot retained-cell coverage across MaxFuse score thresholds."""

    target = "__all__" if population is None else str(population)
    frame = result.threshold_sensitivity.loc[
        result.threshold_sensitivity["population"].astype(str).eq(target)
    ].copy()
    if sources is not None:
        requested = set(map(str, sources))
        frame = frame.loc[frame["source"].astype(str).isin(requested)]
    if frame.empty:
        raise ValueError("No scored MaxFuse threshold-sensitivity evidence is available")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=frame,
        x="score_threshold",
        y="passing_coverage",
        hue="source",
        marker="o",
        ax=ax,
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("MaxFuse matching-score threshold")
    ax.set_ylabel("Fraction of population retained")
    label = "all cells" if population is None else str(population)
    ax.set_title(title or f"MaxFuse threshold sensitivity: {label}")
    figure.tight_layout()
    return PlotResult(figure=figure, axes=ax, data=frame)


__all__ = [
    "inspect_maxfuse_inputs",
    "plot_maxfuse_label_heatmap",
    "plot_maxfuse_threshold_sensitivity",
    "summarize_maxfuse_evidence",
]
