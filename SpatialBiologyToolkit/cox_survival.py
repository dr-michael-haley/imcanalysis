from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import re

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTCOME_COLUMNS = {"duration", "event", "n_cells", "n_rois"}


@dataclass
class CoxFitResult:
    """Container returned by :func:`fit_cox_model`."""

    model: Any
    summary: pd.DataFrame
    feature_columns: list[str]
    fit_data: pd.DataFrame
    scaler: Any | None
    duration_col: str = "duration"
    event_col: str = "event"
    model_type: str = "coxph"
    alpha: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class CoxPipelineResult:
    """Container returned by :func:`run_cox_survival_analysis`."""

    case_table: pd.DataFrame
    feature_columns: list[str]
    univariate_results: pd.DataFrame
    selected_features: list[str]
    fit: CoxFitResult
    cv_metrics: pd.DataFrame
    cv_predictions: pd.DataFrame


def _require_lifelines():
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
        from lifelines.utils import concordance_index
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "Cox survival analysis requires lifelines. Install it with `pip install lifelines` "
            "or install SpatialBiologyToolkit from the repository requirements."
        ) from exc

    return CoxPHFitter, KaplanMeierFitter, multivariate_logrank_test, proportional_hazard_test, concordance_index


def _require_standard_scaler():
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("Feature standardisation requires scikit-learn.") from exc
    return StandardScaler


def _require_sksurv():
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored
        from sksurv.util import Surv
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "Ridge Cox and Coxnet require scikit-survival. Install it with "
            "`pip install scikit-survival` or `conda install -c conda-forge scikit-survival`."
        ) from exc

    return CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, concordance_index_censored, Surv


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _validate_obs_columns(adata: ad.AnnData, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in adata.obs.columns]
    if missing:
        raise ValueError(f"adata.obs is missing required columns: {missing}")


def _safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or "missing"


def _make_unique_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for column in columns:
        base = str(column)
        count = seen.get(base, 0)
        seen[base] = count + 1
        unique.append(base if count == 0 else f"{base}_{count + 1}")
    return unique


def _adjust_pvalues_bh(p_values: pd.Series) -> pd.Series:
    valid = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    mask = valid.notna()
    if not mask.any():
        return adjusted

    values = valid.loc[mask].to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    unsorted = np.empty_like(q)
    unsorted[order] = q
    adjusted.loc[mask] = unsorted
    return adjusted


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def normalize_event_value(value: Any) -> float:
    """Convert common event/censoring encodings to 0.0 or 1.0."""

    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "t", "yes", "y", "event", "dead", "death", "deceased"}:
            return 1.0
        if cleaned in {"0", "false", "f", "no", "n", "censored", "alive", "survived"}:
            return 0.0
    return float(value)


def collapse_case_metadata(
    obs: pd.DataFrame,
    case_obs: str,
    columns: str | Sequence[str],
    conflict: str = "error",
) -> pd.DataFrame:
    """
    Collapse repeated cell/ROI-level metadata to one row per case.

    Parameters
    ----------
    obs
        Observation table containing one or more rows per case.
    case_obs
        Case identifier column.
    columns
        Metadata columns to collapse.
    conflict
        Behaviour when a case has multiple non-null values for a column:
        ``"error"`` raises, ``"first"`` uses the first value, and ``"mode"`` uses
        the most frequent value.
    """

    columns = _as_list(columns)
    if conflict not in {"error", "first", "mode"}:
        raise ValueError("conflict must be one of 'error', 'first', or 'mode'.")

    _missing = [column for column in [case_obs, *columns] if column not in obs.columns]
    if _missing:
        raise ValueError(f"obs is missing required columns: {_missing}")

    rows: dict[Any, dict[str, Any]] = {}
    for case_id, group in obs.groupby(case_obs, dropna=False, sort=True):
        case_values: dict[str, Any] = {}
        for column in columns:
            values = group[column].dropna()
            unique_values = pd.unique(values)
            if len(unique_values) == 0:
                case_values[column] = np.nan
            elif len(unique_values) == 1:
                case_values[column] = unique_values[0]
            elif conflict == "first":
                case_values[column] = values.iloc[0]
            elif conflict == "mode":
                case_values[column] = values.mode(dropna=True).iloc[0]
            else:
                raise ValueError(
                    f"Column '{column}' has multiple values for case '{case_id}': "
                    f"{list(unique_values)}"
                )
        rows[case_id] = case_values

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "case_id"
    return result


def encode_case_covariates(
    obs: pd.DataFrame,
    case_obs: str,
    covariate_obs: str | Sequence[str] | None,
    conflict: str = "error",
    one_hot: bool = True,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Collapse clinical covariates to case level and encode categoricals."""

    covariate_obs = _as_list(covariate_obs)
    if not covariate_obs:
        index = pd.Index(sorted(obs[case_obs].dropna().astype(str).unique()), name="case_id")
        return pd.DataFrame(index=index)

    covariates = collapse_case_metadata(obs, case_obs, covariate_obs, conflict=conflict)
    covariates.index = covariates.index.astype(str)

    numeric_cols = []
    categorical_cols = []
    for column in covariates.columns:
        converted = pd.to_numeric(covariates[column], errors="coerce")
        non_missing = covariates[column].notna()
        if converted[non_missing].notna().all():
            covariates[column] = converted.astype(float)
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)

    if not one_hot or not categorical_cols:
        return covariates

    encoded = pd.get_dummies(
        covariates[categorical_cols].astype("category"),
        prefix=[_safe_name(column) for column in categorical_cols],
        drop_first=drop_first,
        dtype=float,
    )
    return pd.concat([covariates[numeric_cols], encoded], axis=1)


def _categorical_feature_table(
    obs: pd.DataFrame,
    case_obs: str,
    roi_obs: str | None,
    population_obs: str,
    case_aggregation: str,
    normalization: str,
    include_counts: bool,
) -> pd.DataFrame:
    valid = obs[[case_obs, population_obs, *([roi_obs] if roi_obs is not None else [])]].dropna()
    if valid.empty:
        return pd.DataFrame(index=pd.Index([], name="case_id"))

    valid = valid.copy()
    valid[case_obs] = valid[case_obs].astype(str)
    valid[population_obs] = valid[population_obs].astype(str)
    if roi_obs is not None:
        valid[roi_obs] = valid[roi_obs].astype(str)
        counts = pd.crosstab(
            index=[valid[case_obs], valid[roi_obs]],
            columns=valid[population_obs],
            dropna=False,
        )
    else:
        counts = pd.crosstab(index=valid[case_obs], columns=valid[population_obs], dropna=False)

    if counts.empty:
        return pd.DataFrame(index=pd.Index([], name="case_id"))

    if roi_obs is not None and case_aggregation == "roi_mean":
        roi_fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        feature_values = roi_fractions.groupby(level=0).mean()
        summed_counts = counts.groupby(level=0).sum()
    elif case_aggregation == "weighted":
        summed_counts = counts.groupby(level=0).sum() if roi_obs is not None else counts
        if normalization in {"fraction", "proportion", "frequency"}:
            feature_values = summed_counts.div(summed_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        elif normalization == "count":
            feature_values = summed_counts.astype(float)
        else:
            raise ValueError("normalization must be 'fraction', 'proportion', 'frequency', or 'count'.")
    else:
        raise ValueError("case_aggregation must be 'weighted' or 'roi_mean'.")

    metric = "frac" if normalization in {"fraction", "proportion", "frequency"} else "count"
    feature_values.columns = [
        f"{_safe_name(population_obs)}_{metric}_{_safe_name(column)}" for column in feature_values.columns
    ]
    tables = [feature_values]

    if include_counts and normalization != "count":
        count_values = summed_counts.astype(float)
        count_values.columns = [
            f"{_safe_name(population_obs)}_count_{_safe_name(column)}" for column in count_values.columns
        ]
        tables.append(count_values)

    result = pd.concat(tables, axis=1).fillna(0.0)
    result.index = result.index.astype(str)
    result.index.name = "case_id"
    result.columns = _make_unique_columns(result.columns)
    return result


def _continuous_feature_table(
    obs: pd.DataFrame,
    case_obs: str,
    roi_obs: str | None,
    continuous_obs: Sequence[str],
    continuous_agg: Sequence[str],
    case_aggregation: str,
) -> pd.DataFrame:
    if not continuous_obs:
        index = pd.Index(sorted(obs[case_obs].dropna().astype(str).unique()), name="case_id")
        return pd.DataFrame(index=index)

    frames = []
    for column in continuous_obs:
        values = obs[[case_obs, *([roi_obs] if roi_obs is not None else []), column]].copy()
        values[column] = pd.to_numeric(values[column], errors="coerce")
        values = values.dropna(subset=[case_obs, column])
        values[case_obs] = values[case_obs].astype(str)
        if roi_obs is not None:
            values = values.dropna(subset=[roi_obs])
            values[roi_obs] = values[roi_obs].astype(str)

        if values.empty:
            continue

        if roi_obs is not None and case_aggregation == "roi_mean":
            roi_level = values.groupby([case_obs, roi_obs])[column].agg(list(continuous_agg))
            case_level = roi_level.groupby(level=0).mean()
        else:
            case_level = values.groupby(case_obs)[column].agg(list(continuous_agg))

        if isinstance(case_level, pd.Series):
            case_level = case_level.to_frame(name=f"{_safe_name(column)}_{_safe_name(continuous_agg[0])}")
        else:
            case_level.columns = [f"{_safe_name(column)}_{_safe_name(agg)}" for agg in case_level.columns]
        frames.append(case_level)

    if not frames:
        index = pd.Index(sorted(obs[case_obs].dropna().astype(str).unique()), name="case_id")
        return pd.DataFrame(index=index)

    result = pd.concat(frames, axis=1)
    result.index = result.index.astype(str)
    result.index.name = "case_id"
    result.columns = _make_unique_columns(result.columns)
    return result


def build_case_level_table(
    adata: ad.AnnData,
    population_obs: str | Sequence[str] | None,
    duration_obs: str,
    event_obs: str | None = None,
    case_obs: str = "Case",
    roi_obs: str | None = "ROI",
    covariate_obs: str | Sequence[str] | None = None,
    continuous_obs: str | Sequence[str] | None = None,
    continuous_agg: Sequence[str] = ("mean",),
    case_aggregation: str = "weighted",
    normalization: str = "fraction",
    include_population_counts: bool = False,
    assume_all_events: bool = False,
    metadata_conflict: str = "error",
    one_hot_covariates: bool = True,
    drop_first_covariate_level: bool = True,
    min_cells_per_case: int = 1,
    min_rois_per_case: int = 1,
    min_feature_prevalence: float = 0.0,
    drop_constant_features: bool = True,
) -> pd.DataFrame:
    """
    Aggregate ``adata.obs`` to one row per case for Cox survival analysis.

    Cell-level categorical annotations in ``population_obs`` are aggregated by
    ROI first, then by case. With the default ``case_aggregation="weighted"``,
    ROI counts are summed within each case before fractions are calculated.
    With ``case_aggregation="roi_mean"``, each ROI contributes equally by first
    converting ROI counts to fractions and then averaging those fractions within
    each case.

    The returned table standardises outcome names to ``duration`` and ``event``
    and stores the usable model features in ``case_table.attrs["feature_columns"]``.
    """

    population_obs = _as_list(population_obs)
    continuous_obs = _as_list(continuous_obs)
    covariate_obs = _as_list(covariate_obs)
    continuous_agg = tuple(continuous_agg)
    if not continuous_agg:
        raise ValueError("continuous_agg must contain at least one aggregation name.")
    if case_aggregation not in {"weighted", "roi_mean"}:
        raise ValueError("case_aggregation must be 'weighted' or 'roi_mean'.")

    required = [case_obs, duration_obs, *population_obs, *continuous_obs, *covariate_obs]
    if event_obs is not None:
        required.append(event_obs)
    elif not assume_all_events:
        raise ValueError(
            "Survival analysis needs an event/censoring column. Pass event_obs, "
            "or set assume_all_events=True only if all cases had the event."
        )
    if roi_obs is not None:
        required.append(roi_obs)
    required = _ordered_unique(required)
    _validate_obs_columns(adata, required)

    obs = adata.obs[required].copy()
    obs[case_obs] = obs[case_obs].astype(str)
    if roi_obs is not None:
        obs[roi_obs] = obs[roi_obs].astype(str)

    outcome_columns = [duration_obs, *([event_obs] if event_obs is not None else [])]
    outcome = collapse_case_metadata(obs, case_obs, outcome_columns, conflict=metadata_conflict)
    outcome.index = outcome.index.astype(str)
    outcome["duration"] = pd.to_numeric(outcome[duration_obs], errors="coerce")
    if event_obs is None:
        outcome["event"] = True
    else:
        outcome["event"] = outcome[event_obs].map(normalize_event_value)

    n_cells = obs.groupby(case_obs).size().rename("n_cells")
    n_cells.index = n_cells.index.astype(str)
    if roi_obs is None:
        n_rois = pd.Series(1, index=n_cells.index, name="n_rois")
    else:
        n_rois = obs.groupby(case_obs)[roi_obs].nunique(dropna=True).rename("n_rois")
        n_rois.index = n_rois.index.astype(str)

    feature_tables = []
    for column in population_obs:
        feature_tables.append(
            _categorical_feature_table(
                obs=obs,
                case_obs=case_obs,
                roi_obs=roi_obs,
                population_obs=column,
                case_aggregation=case_aggregation,
                normalization=normalization,
                include_counts=include_population_counts,
            )
        )

    if continuous_obs:
        feature_tables.append(
            _continuous_feature_table(
                obs=obs,
                case_obs=case_obs,
                roi_obs=roi_obs,
                continuous_obs=continuous_obs,
                continuous_agg=continuous_agg,
                case_aggregation=case_aggregation,
            )
        )

    covariates = encode_case_covariates(
        obs=obs,
        case_obs=case_obs,
        covariate_obs=covariate_obs,
        conflict=metadata_conflict,
        one_hot=one_hot_covariates,
        drop_first=drop_first_covariate_level,
    )
    if not covariates.empty:
        feature_tables.append(covariates)

    case_index = pd.Index(sorted(obs[case_obs].dropna().astype(str).unique()), name="case_id")
    if feature_tables:
        features = pd.concat(feature_tables, axis=1).reindex(case_index)
    else:
        features = pd.DataFrame(index=case_index)
    features.columns = _make_unique_columns(features.columns)

    case_table = features.join(outcome[["duration", "event"]], how="left")
    case_table = case_table.join(n_cells, how="left").join(n_rois, how="left")
    case_table = case_table[case_table["n_cells"].fillna(0) >= min_cells_per_case]
    case_table = case_table[case_table["n_rois"].fillna(0) >= min_rois_per_case]

    case_table["duration"] = pd.to_numeric(case_table["duration"], errors="coerce")
    case_table["event"] = pd.to_numeric(case_table["event"], errors="coerce")
    case_table = case_table.dropna(subset=["duration", "event"])
    case_table = case_table[case_table["duration"] > 0].copy()
    case_table["event"] = case_table["event"].astype(bool)

    feature_columns = get_feature_columns(case_table)
    if feature_columns:
        case_table[feature_columns] = case_table[feature_columns].apply(pd.to_numeric, errors="coerce")
        case_table = case_table.dropna(subset=feature_columns)

        if min_feature_prevalence > 0:
            prevalence = (case_table[feature_columns].fillna(0.0) != 0).mean(axis=0)
            feature_columns = prevalence[prevalence >= min_feature_prevalence].index.tolist()

        if drop_constant_features and feature_columns:
            feature_columns = [
                column for column in feature_columns if case_table[column].nunique(dropna=True) > 1
            ]

        keep_columns = [*feature_columns, "duration", "event", "n_cells", "n_rois"]
        case_table = case_table[keep_columns]

    case_table.attrs["feature_columns"] = feature_columns
    return case_table


def get_feature_columns(
    case_table: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    exclude_cols: Sequence[str] | None = None,
) -> list[str]:
    """Return numeric candidate feature columns from a case-level survival table."""

    exclude = set(exclude_cols or [])
    exclude.update({duration_col, event_col, "n_cells", "n_rois"})
    return [column for column in case_table.columns if column not in exclude]


def select_top_features(
    univariate_results: pd.DataFrame,
    top_n: int = 20,
    p_value_threshold: float | None = None,
    force_include: Sequence[str] | None = None,
) -> list[str]:
    """Select features from a univariate Cox ranking table."""

    force_include = list(force_include or [])
    if univariate_results.empty:
        return force_include[:top_n]

    ranked = univariate_results.copy()
    ranked = ranked[ranked["status"] == "ok"] if "status" in ranked else ranked
    ranked = ranked.dropna(subset=["p"])
    if p_value_threshold is not None:
        ranked = ranked[ranked["p"] <= p_value_threshold]
    ranked = ranked.sort_values(["p", "coef"], ascending=[True, False])
    selected = list(dict.fromkeys([*force_include, *ranked["feature"].head(top_n).tolist()]))
    return selected[:top_n]


def fit_univariate_cox(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    penalizer: float = 0.0,
    robust: bool = True,
) -> pd.DataFrame:
    """Fit one Cox PH model per feature and return a ranked summary table."""

    CoxPHFitter, *_ = _require_lifelines()
    feature_cols = list(feature_cols or get_feature_columns(case_table, duration_col, event_col))
    rows = []

    for feature in feature_cols:
        fit_df = case_table[[feature, duration_col, event_col]].dropna().copy()
        status = "ok"
        if fit_df.empty or fit_df[feature].nunique(dropna=True) <= 1:
            rows.append(
                {
                    "feature": feature,
                    "coef": np.nan,
                    "hazard_ratio": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p": np.nan,
                    "concordance_index": np.nan,
                    "n_cases": int(len(fit_df)),
                    "n_events": int(fit_df[event_col].sum()) if event_col in fit_df else 0,
                    "status": "skipped: constant or empty feature",
                }
            )
            continue

        fit_df[event_col] = fit_df[event_col].astype(bool)
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(fit_df, duration_col=duration_col, event_col=event_col, robust=robust)
            summary = cph.summary.loc[feature]
            rows.append(
                {
                    "feature": feature,
                    "coef": float(summary["coef"]),
                    "hazard_ratio": float(summary["exp(coef)"]),
                    "ci_lower": float(summary["exp(coef) lower 95%"]),
                    "ci_upper": float(summary["exp(coef) upper 95%"]),
                    "p": float(summary["p"]),
                    "concordance_index": float(cph.concordance_index_),
                    "n_cases": int(len(fit_df)),
                    "n_events": int(fit_df[event_col].sum()),
                    "status": status,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature": feature,
                    "coef": np.nan,
                    "hazard_ratio": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p": np.nan,
                    "concordance_index": np.nan,
                    "n_cases": int(len(fit_df)),
                    "n_events": int(fit_df[event_col].sum()),
                    "status": f"failed: {exc}",
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bh"] = _adjust_pvalues_bh(result["p"])
        result = result.sort_values(["p", "concordance_index"], ascending=[True, False], na_position="last")
        result = result.reset_index(drop=True)
    return result


def _standardize_frame(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Any]:
    StandardScaler = _require_standard_scaler()
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)
    if test is None:
        return train_scaled, None, scaler
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    return train_scaled, test_scaled, scaler


def make_survival_array(
    case_table: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
):
    """Build a scikit-survival structured outcome array from a case table."""

    *_, Surv = _require_sksurv()
    return Surv.from_arrays(
        event=case_table[event_col].astype(bool).to_numpy(),
        time=case_table[duration_col].astype(float).to_numpy(),
    )


def _sksurv_c_index(y, risk: np.ndarray) -> float:
    _, _, concordance_index_censored, _ = _require_sksurv()
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


def _prepare_sksurv_inputs(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None,
    duration_col: str,
    event_col: str,
    standardize: bool,
) -> tuple[pd.DataFrame, Any, pd.DataFrame, Any | None, list[str]]:
    feature_cols = list(feature_cols or get_feature_columns(case_table, duration_col, event_col))
    if not feature_cols:
        raise ValueError("No feature columns were supplied or detected.")

    fit_table = case_table[[*feature_cols, duration_col, event_col]].dropna().copy()
    if fit_table[event_col].astype(bool).sum() == 0:
        raise ValueError("At least one observed event is required for survival fitting.")

    X = fit_table[feature_cols].astype(float)
    scaler = None
    if standardize:
        X, _, scaler = _standardize_frame(X)
    y = make_survival_array(fit_table, duration_col=duration_col, event_col=event_col)
    return X, y, fit_table, scaler, feature_cols


def _coxnet_coefficients(model: Any, feature_cols: Sequence[str], alpha: float | None) -> np.ndarray:
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 1:
        return coefficients.reshape(-1)

    alphas = np.asarray(getattr(model, "alphas_", []), dtype=float)
    if alpha is None or len(alphas) == 0:
        index = coefficients.shape[1] - 1
    else:
        index = int(np.argmin(np.abs(alphas - float(alpha))))
    return coefficients[:, index].reshape(-1)


def _penalized_summary(
    feature_cols: Sequence[str],
    coefficients: np.ndarray,
    model_type: str,
    alpha: float,
    l1_ratio: float | None = None,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "coef": coefficients.astype(float),
            "hazard_ratio": np.exp(coefficients.astype(float)),
            "abs_coef": np.abs(coefficients.astype(float)),
            "nonzero": np.abs(coefficients.astype(float)) > 1e-12,
            "model_type": model_type,
            "alpha": float(alpha),
        }
    )
    if l1_ratio is not None:
        summary["l1_ratio"] = float(l1_ratio)
    return summary.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def spread_label_positions(values: np.ndarray) -> np.ndarray:
    """Return y label positions with a small minimum separation."""

    if len(values) <= 1:
        return values.copy()

    values = values.astype(float, copy=True)
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    y_range = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
    min_gap = 0.035 * y_range

    order = np.argsort(values)
    sorted_positions = values[order].copy()
    for index in range(1, len(sorted_positions)):
        sorted_positions[index] = max(sorted_positions[index], sorted_positions[index - 1] + min_gap)

    overshoot = sorted_positions[-1] - (y_max + 0.1 * y_range)
    if overshoot > 0:
        sorted_positions -= overshoot
        for index in range(len(sorted_positions) - 2, -1, -1):
            sorted_positions[index] = min(sorted_positions[index], sorted_positions[index + 1] - min_gap)

    output = np.empty_like(values)
    output[order] = sorted_positions
    return output


def coxnet_path_coefficients(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    l1_ratio: float = 0.5,
    n_alphas: int = 100,
    alpha_min_ratio: float = 0.01,
    max_iter: int = 100000,
    tol: float = 1e-7,
    standardize: bool = True,
) -> pd.DataFrame:
    """Fit a full Coxnet path and return long-form coefficient trajectories."""

    _, CoxnetSurvivalAnalysis, *_ = _require_sksurv()
    X, y, _, _, feature_cols = _prepare_sksurv_inputs(
        case_table=case_table,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        standardize=standardize,
    )
    model = CoxnetSurvivalAnalysis(
        l1_ratio=float(l1_ratio),
        n_alphas=int(n_alphas),
        alpha_min_ratio=float(alpha_min_ratio),
        max_iter=int(max_iter),
        tol=float(tol),
    )
    model.fit(X, y)
    alphas = np.asarray(model.alphas_, dtype=float)
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(-1, 1)

    rows = []
    for alpha_index, alpha in enumerate(alphas):
        for feature, coef in zip(feature_cols, coefficients[:, alpha_index]):
            coef = float(coef)
            rows.append(
                {
                    "feature": feature,
                    "alpha": float(alpha),
                    "coef": coef,
                    "abs_coef": abs(coef),
                    "nonzero": bool(abs(coef) > 1e-12),
                    "model_type": "coxnet",
                    "l1_ratio": float(l1_ratio),
                }
            )
    return pd.DataFrame(rows)


def ridge_cox_path_coefficients(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    alphas: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Fit Ridge Cox over an alpha grid and return long-form coefficient paths.

    Ridge paths are shrinkage/stability diagnostics. They do not show sparse
    feature entry/exit in the same way that Coxnet paths do.
    """

    CoxPHSurvivalAnalysis, *_ = _require_sksurv()
    X, y, _, _, feature_cols = _prepare_sksurv_inputs(
        case_table=case_table,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        standardize=standardize,
    )
    rows = []
    for alpha in sorted({float(value) for value in alphas if float(value) > 0}):
        try:
            model = CoxPHSurvivalAnalysis(alpha=alpha)
            model.fit(X, y)
            coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
            status = "ok"
        except Exception as exc:
            coefficients = np.full(len(feature_cols), np.nan, dtype=float)
            status = f"failed: {exc}"

        for feature, coef in zip(feature_cols, coefficients):
            coef = float(coef) if np.isfinite(coef) else np.nan
            rows.append(
                {
                    "feature": feature,
                    "alpha": alpha,
                    "coef": coef,
                    "abs_coef": abs(coef) if np.isfinite(coef) else np.nan,
                    "nonzero": bool(np.isfinite(coef) and abs(coef) > 1e-12),
                    "model_type": "ridge",
                    "l1_ratio": 0.0,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def fit_ridge_cox_model(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    alpha: float = 1.0,
    standardize: bool = True,
) -> CoxFitResult:
    """Fit a scikit-survival ridge-penalised Cox model."""

    CoxPHSurvivalAnalysis, *_ = _require_sksurv()
    X, y, fit_table, scaler, feature_cols = _prepare_sksurv_inputs(
        case_table=case_table,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        standardize=standardize,
    )
    model = CoxPHSurvivalAnalysis(alpha=float(alpha))
    model.fit(X, y)
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    summary = _penalized_summary(feature_cols, coefficients, "ridge", float(alpha), l1_ratio=0.0)
    return CoxFitResult(
        model=model,
        summary=summary,
        feature_columns=feature_cols,
        fit_data=fit_table,
        scaler=scaler,
        duration_col=duration_col,
        event_col=event_col,
        model_type="ridge",
        alpha=float(alpha),
        metadata={"standardized": bool(scaler is not None)},
    )


def fit_coxnet_alpha_cv(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    l1_ratio: float = 0.5,
    n_alphas: int = 100,
    alpha_min_ratio: float = 0.01,
    max_iter: int = 100000,
    tol: float = 1e-7,
    n_splits: int = 5,
    seed: int = 1,
    standardize: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Choose a Coxnet alpha by case-level K-fold C-index."""

    _, CoxnetSurvivalAnalysis, *_ = _require_sksurv()
    try:
        from sklearn.model_selection import KFold
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("fit_coxnet_alpha_cv requires scikit-learn.") from exc

    X, y, fit_table, _, feature_cols = _prepare_sksurv_inputs(
        case_table=case_table,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        standardize=standardize,
    )
    path_model = CoxnetSurvivalAnalysis(
        l1_ratio=float(l1_ratio),
        n_alphas=int(n_alphas),
        alpha_min_ratio=float(alpha_min_ratio),
        max_iter=int(max_iter),
        tol=float(tol),
    )
    path_model.fit(X, y)
    alphas = np.asarray(path_model.alphas_, dtype=float)
    if len(alphas) == 0:
        raise ValueError("Coxnet did not produce an alpha path.")

    n_splits = min(int(n_splits), len(fit_table))
    if n_splits < 2:
        raise ValueError("At least two cases are required for Coxnet alpha CV.")

    rows = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(fit_table), start=1):
        train = fit_table.iloc[train_idx]
        test = fit_table.iloc[test_idx]
        X_train = train[feature_cols].astype(float)
        X_test = test[feature_cols].astype(float)
        if standardize:
            X_train, X_test, _ = _standardize_frame(X_train, X_test)
        y_train = make_survival_array(train, duration_col=duration_col, event_col=event_col)
        y_test = make_survival_array(test, duration_col=duration_col, event_col=event_col)

        try:
            fold_model = CoxnetSurvivalAnalysis(
                l1_ratio=float(l1_ratio),
                alphas=alphas,
                max_iter=int(max_iter),
                tol=float(tol),
            )
            fold_model.fit(X_train, y_train)
        except Exception as exc:
            for alpha in alphas:
                rows.append({"fold": fold, "alpha": float(alpha), "c_index": np.nan, "status": f"failed: {exc}"})
            continue

        for alpha in alphas:
            try:
                risk = np.asarray(fold_model.predict(X_test, alpha=float(alpha)), dtype=float).reshape(-1)
                score = _sksurv_c_index(y_test, risk)
                status = "ok"
            except Exception as exc:
                score = np.nan
                status = f"failed: {exc}"
            rows.append({"fold": fold, "alpha": float(alpha), "c_index": score, "status": status})

    scores = pd.DataFrame(rows)
    summary = (
        scores.groupby("alpha", as_index=False)
        .agg(
            mean_c_index=("c_index", "mean"),
            std_c_index=("c_index", "std"),
            n_valid_folds=("c_index", "count"),
        )
        .dropna(subset=["mean_c_index"])
    )
    if summary.empty:
        best_alpha = float(alphas[0])
        scores["selected_alpha"] = best_alpha
        return scores, best_alpha

    summary = summary.sort_values(
        ["mean_c_index", "n_valid_folds", "alpha"],
        ascending=[False, False, True],
    )
    best_alpha = float(summary.iloc[0]["alpha"])
    scores = scores.merge(summary, on="alpha", how="left")
    scores["selected_alpha"] = best_alpha
    return scores, best_alpha


def fit_coxnet_model(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    alpha: float | None = None,
    l1_ratio: float = 0.5,
    n_alphas: int = 100,
    alpha_min_ratio: float = 0.01,
    max_iter: int = 100000,
    tol: float = 1e-7,
    cv_folds: int = 5,
    seed: int = 1,
    standardize: bool = True,
) -> CoxFitResult:
    """Fit a scikit-survival elastic-net Cox model, selecting alpha if needed."""

    _, CoxnetSurvivalAnalysis, *_ = _require_sksurv()
    alpha_scores = pd.DataFrame()
    if alpha is None:
        alpha_scores, alpha = fit_coxnet_alpha_cv(
            case_table=case_table,
            feature_cols=feature_cols,
            duration_col=duration_col,
            event_col=event_col,
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=max_iter,
            tol=tol,
            n_splits=cv_folds,
            seed=seed,
            standardize=standardize,
        )

    X, y, fit_table, scaler, feature_cols = _prepare_sksurv_inputs(
        case_table=case_table,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        standardize=standardize,
    )
    model = CoxnetSurvivalAnalysis(
        l1_ratio=float(l1_ratio),
        alphas=[float(alpha)],
        max_iter=int(max_iter),
        tol=float(tol),
    )
    model.fit(X, y)
    coefficients = _coxnet_coefficients(model, feature_cols, float(alpha))
    summary = _penalized_summary(feature_cols, coefficients, "coxnet", float(alpha), l1_ratio=l1_ratio)
    return CoxFitResult(
        model=model,
        summary=summary,
        feature_columns=feature_cols,
        fit_data=fit_table,
        scaler=scaler,
        duration_col=duration_col,
        event_col=event_col,
        model_type="coxnet",
        alpha=float(alpha),
        metadata={
            "standardized": bool(scaler is not None),
            "l1_ratio": float(l1_ratio),
            "alpha_cv_scores": alpha_scores,
        },
    )


def fit_cox_model(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    robust: bool = True,
    standardize: bool = False,
    strata: Sequence[str] | None = None,
) -> CoxFitResult:
    """
    Fit a multivariable Cox proportional hazards model using lifelines.

    Set ``l1_ratio`` above zero for elastic-net style penalisation. When
    ``standardize=True``, coefficients are on the standard-deviation scale and
    the fitted scaler is stored in the returned :class:`CoxFitResult`.
    """

    CoxPHFitter, *_ = _require_lifelines()
    feature_cols = list(feature_cols or get_feature_columns(case_table, duration_col, event_col))
    if not feature_cols:
        raise ValueError("No feature columns were supplied or detected.")

    fit_df = case_table[[*feature_cols, duration_col, event_col, *(strata or [])]].dropna().copy()
    if fit_df[event_col].astype(bool).sum() == 0:
        raise ValueError("At least one observed event is required for Cox fitting.")

    scaler = None
    if standardize:
        X_scaled, _, scaler = _standardize_frame(fit_df[feature_cols])
        fit_df.loc[:, feature_cols] = X_scaled

    fit_df[event_col] = fit_df[event_col].astype(bool)
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(
        fit_df,
        duration_col=duration_col,
        event_col=event_col,
        robust=robust,
        strata=list(strata) if strata is not None else None,
    )
    summary = cph.summary.reset_index().rename(columns={"covariate": "feature"})
    summary["p_bh"] = _adjust_pvalues_bh(summary["p"])
    return CoxFitResult(
        model=cph,
        summary=summary,
        feature_columns=feature_cols,
        fit_data=fit_df,
        scaler=scaler,
        duration_col=duration_col,
        event_col=event_col,
        model_type="coxph",
        metadata={"standardized": bool(scaler is not None), "penalizer": penalizer, "l1_ratio": l1_ratio},
    )


def _predict_log_hazard(fit: CoxFitResult, case_table: pd.DataFrame) -> pd.Series:
    X = case_table[fit.feature_columns].copy()
    if fit.scaler is not None:
        X = pd.DataFrame(fit.scaler.transform(X), index=X.index, columns=X.columns)
    if fit.model_type == "coxph":
        risk = fit.model.predict_log_partial_hazard(X)
    elif fit.model_type == "coxnet":
        try:
            risk = fit.model.predict(X, alpha=fit.alpha)
        except TypeError:
            risk = fit.model.predict(X)
    elif fit.model_type == "ridge":
        risk = fit.model.predict(X)
    else:
        raise ValueError(f"Unsupported fitted Cox model type: {fit.model_type}")
    return pd.Series(np.asarray(risk, dtype=float).reshape(-1), index=X.index, name="risk_score")


def assign_risk_groups(
    risk_percentile: pd.Series,
    quantiles: Sequence[float] = (0.0, 0.33, 0.67, 1.0),
) -> pd.DataFrame:
    """Assign cases to ordered risk groups from risk-score percentiles."""

    quantiles = tuple(float(value) for value in quantiles)
    if sorted(quantiles) != list(quantiles) or quantiles[0] < 0 or quantiles[-1] > 1:
        raise ValueError("quantiles must be sorted values between 0 and 1.")
    labels = []
    n_groups = len(quantiles) - 1
    for index, (lower, upper) in enumerate(zip(quantiles[:-1], quantiles[1:])):
        if n_groups == 3:
            label = ["low", "intermediate", "high"][index]
            labels.append(f"{label} risk ({100 * lower:.0f}-{100 * upper:.0f}%)")
        else:
            labels.append(f"risk {100 * lower:.0f}-{100 * upper:.0f}%")

    groups = pd.DataFrame(index=risk_percentile.index)
    groups["risk_group"] = pd.NA
    groups["risk_group_order"] = np.nan
    for index, (lower, upper) in enumerate(zip(quantiles[:-1], quantiles[1:])):
        if index == 0:
            mask = risk_percentile <= upper
        else:
            mask = (risk_percentile > lower) & (risk_percentile <= upper)
        groups.loc[mask, "risk_group"] = labels[index]
        groups.loc[mask, "risk_group_order"] = index
    return groups


def build_risk_table(
    fit: CoxFitResult,
    case_table: pd.DataFrame,
    quantiles: Sequence[float] = (0.0, 0.33, 0.67, 1.0),
) -> pd.DataFrame:
    """Create case-level fitted risk scores and quantile risk groups."""

    risk_score = _predict_log_hazard(fit, case_table)
    risk_percentile = risk_score.rank(method="average", pct=True)
    risk_groups = assign_risk_groups(risk_percentile, quantiles=quantiles)

    risk_table = pd.DataFrame(index=risk_score.index)
    risk_table.index.name = "case_id"
    risk_table["risk_score"] = risk_score
    risk_table["risk_percentile"] = risk_percentile
    risk_table = risk_table.join(risk_groups)
    risk_table["duration"] = case_table.loc[risk_table.index, fit.duration_col].astype(float)
    risk_table["event"] = case_table.loc[risk_table.index, fit.event_col].astype(bool)
    for column in ["n_cells", "n_rois"]:
        if column in case_table:
            risk_table[column] = case_table.loc[risk_table.index, column]
    return risk_table.sort_values("risk_score")


def cross_validate_cox_model(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    n_splits: int = 5,
    repeats: int = 5,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    standardize: bool = True,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated case-level K-fold validation for a lifelines Cox model."""

    CoxPHFitter, _, _, _, concordance_index = _require_lifelines()
    try:
        from sklearn.model_selection import KFold
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("cross_validate_cox_model requires scikit-learn.") from exc

    feature_cols = list(feature_cols or get_feature_columns(case_table, duration_col, event_col))
    fit_table = case_table[[*feature_cols, duration_col, event_col]].dropna().copy()
    fit_table[event_col] = fit_table[event_col].astype(bool)
    if len(fit_table) < 2:
        raise ValueError("At least two cases are required for cross-validation.")
    if fit_table[event_col].sum() == 0:
        raise ValueError("At least one observed event is required for cross-validation.")

    n_splits = min(int(n_splits), len(fit_table))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 after accounting for the number of cases.")

    metric_rows = []
    prediction_rows = []

    for repeat in range(int(repeats)):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(fit_table), start=1):
            train = fit_table.iloc[train_idx].copy()
            test = fit_table.iloc[test_idx].copy()
            X_train = train[feature_cols]
            X_test = test[feature_cols]
            scaler = None
            if standardize:
                X_train, X_test, scaler = _standardize_frame(X_train, X_test)

            train_fit = pd.concat([X_train, train[[duration_col, event_col]]], axis=1)
            status = "ok"
            try:
                model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                model.fit(train_fit, duration_col=duration_col, event_col=event_col, robust=True)
                risk = np.asarray(model.predict_log_partial_hazard(X_test), dtype=float).reshape(-1)
                c_index = float(
                    concordance_index(
                        test[duration_col].to_numpy(dtype=float),
                        -risk,
                        test[event_col].to_numpy(dtype=bool),
                    )
                )
                train_c_index = float(model.concordance_index_)
            except Exception as exc:
                risk = np.full(len(test), np.nan, dtype=float)
                c_index = np.nan
                train_c_index = np.nan
                status = f"failed: {exc}"

            metric_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_features": int(len(feature_cols)),
                    "train_c_index": train_c_index,
                    "heldout_c_index": c_index,
                    "fit_status": status,
                    "standardized": bool(scaler is not None),
                }
            )

            for case_id, risk_score, duration, event in zip(
                test.index.astype(str),
                risk,
                test[duration_col],
                test[event_col],
            ):
                prediction_rows.append(
                    {
                        "case_id": case_id,
                        "repeat": repeat + 1,
                        "fold": fold,
                        "heldout_risk_score": float(risk_score) if np.isfinite(risk_score) else np.nan,
                        "duration": float(duration),
                        "event": bool(event),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def cross_validate_sksurv_cox_model(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    model_type: str = "ridge",
    n_splits: int = 5,
    repeats: int = 5,
    ridge_alpha: float = 1.0,
    coxnet_alpha: float | None = None,
    coxnet_l1_ratio: float = 0.5,
    coxnet_n_alphas: int = 100,
    coxnet_alpha_min_ratio: float = 0.01,
    coxnet_max_iter: int = 100000,
    coxnet_tol: float = 1e-7,
    standardize: bool = True,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated case-level K-fold validation for Ridge Cox or Coxnet."""

    CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, *_ = _require_sksurv()
    try:
        from sklearn.model_selection import KFold
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("cross_validate_sksurv_cox_model requires scikit-learn.") from exc

    model_type = str(model_type).lower()
    if model_type in {"ridge_cox", "ridge-cox"}:
        model_type = "ridge"
    if model_type not in {"ridge", "coxnet"}:
        raise ValueError("model_type must be 'ridge' or 'coxnet'.")

    feature_cols = list(feature_cols or get_feature_columns(case_table, duration_col, event_col))
    fit_table = case_table[[*feature_cols, duration_col, event_col]].dropna().copy()
    fit_table[event_col] = fit_table[event_col].astype(bool)
    if len(fit_table) < 2:
        raise ValueError("At least two cases are required for cross-validation.")
    if fit_table[event_col].sum() == 0:
        raise ValueError("At least one observed event is required for cross-validation.")

    n_splits = min(int(n_splits), len(fit_table))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 after accounting for the number of cases.")

    metric_rows = []
    prediction_rows = []

    for repeat in range(int(repeats)):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(fit_table), start=1):
            train = fit_table.iloc[train_idx].copy()
            test = fit_table.iloc[test_idx].copy()
            X_train = train[feature_cols].astype(float)
            X_test = test[feature_cols].astype(float)
            scaler = None
            if standardize:
                X_train, X_test, scaler = _standardize_frame(X_train, X_test)
            y_train = make_survival_array(train, duration_col=duration_col, event_col=event_col)
            y_test = make_survival_array(test, duration_col=duration_col, event_col=event_col)

            model_alpha = float(ridge_alpha) if model_type == "ridge" else coxnet_alpha
            status = "ok"
            try:
                if model_type == "ridge":
                    model = CoxPHSurvivalAnalysis(alpha=float(ridge_alpha))
                    model.fit(X_train, y_train)
                    risk = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
                else:
                    if model_alpha is None:
                        _, model_alpha = fit_coxnet_alpha_cv(
                            case_table=train,
                            feature_cols=feature_cols,
                            duration_col=duration_col,
                            event_col=event_col,
                            l1_ratio=coxnet_l1_ratio,
                            n_alphas=coxnet_n_alphas,
                            alpha_min_ratio=coxnet_alpha_min_ratio,
                            max_iter=coxnet_max_iter,
                            tol=coxnet_tol,
                            n_splits=min(n_splits, len(train)),
                            seed=seed + repeat + fold,
                            standardize=standardize,
                        )
                    model = CoxnetSurvivalAnalysis(
                        l1_ratio=float(coxnet_l1_ratio),
                        alphas=[float(model_alpha)],
                        max_iter=int(coxnet_max_iter),
                        tol=float(coxnet_tol),
                    )
                    model.fit(X_train, y_train)
                    risk = np.asarray(model.predict(X_test, alpha=float(model_alpha)), dtype=float).reshape(-1)
                heldout_c_index = _sksurv_c_index(y_test, risk)
                if model_type == "coxnet":
                    train_risk = np.asarray(model.predict(X_train, alpha=float(model_alpha)), dtype=float).reshape(-1)
                else:
                    train_risk = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
                train_c_index = _sksurv_c_index(y_train, train_risk)
            except Exception as exc:
                risk = np.full(len(test), np.nan, dtype=float)
                heldout_c_index = np.nan
                train_c_index = np.nan
                status = f"failed: {exc}"

            metric_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "model_type": model_type,
                    "alpha": float(model_alpha) if model_alpha is not None else np.nan,
                    "l1_ratio": float(coxnet_l1_ratio) if model_type == "coxnet" else 0.0,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_features": int(len(feature_cols)),
                    "train_c_index": train_c_index,
                    "heldout_c_index": heldout_c_index,
                    "fit_status": status,
                    "standardized": bool(scaler is not None),
                }
            )

            for case_id, risk_score, duration, event in zip(
                test.index.astype(str),
                risk,
                test[duration_col],
                test[event_col],
            ):
                prediction_rows.append(
                    {
                        "case_id": case_id,
                        "repeat": repeat + 1,
                        "fold": fold,
                        "model_type": model_type,
                        "heldout_risk_score": float(risk_score) if np.isfinite(risk_score) else np.nan,
                        "duration": float(duration),
                        "event": bool(event),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarise_cv_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise repeated-fold Cox validation metrics."""

    if metrics.empty:
        return pd.DataFrame()
    valid = metrics[metrics["fit_status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "n_folds": int(len(valid)),
                "mean_heldout_c_index": float(valid["heldout_c_index"].mean()),
                "std_heldout_c_index": float(valid["heldout_c_index"].std()),
                "mean_train_c_index": float(valid["train_c_index"].mean()),
                "std_train_c_index": float(valid["train_c_index"].std()),
                "mean_n_features": float(valid["n_features"].mean()),
            }
        ]
    )


def aggregate_cv_predictions(
    predictions: pd.DataFrame,
    quantiles: Sequence[float] = (0.0, 0.33, 0.67, 1.0),
) -> pd.DataFrame:
    """Average held-out risk scores per case and assign risk groups."""

    if predictions.empty:
        return pd.DataFrame()
    valid = predictions.dropna(subset=["heldout_risk_score"]).copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = (
        valid.groupby("case_id", as_index=False)
        .agg(
            mean_heldout_risk_score=("heldout_risk_score", "mean"),
            std_heldout_risk_score=("heldout_risk_score", "std"),
            n_heldout_predictions=("heldout_risk_score", "count"),
            duration=("duration", "first"),
            event=("event", "first"),
        )
        .set_index("case_id")
    )
    grouped["event"] = grouped["event"].astype(bool)
    grouped["heldout_risk_percentile"] = grouped["mean_heldout_risk_score"].rank(method="average", pct=True)
    grouped = grouped.join(assign_risk_groups(grouped["heldout_risk_percentile"], quantiles=quantiles))
    return grouped.sort_values("mean_heldout_risk_score")


def test_proportional_hazards(
    fit: CoxFitResult,
    time_transform: str = "rank",
) -> pd.DataFrame:
    """Run lifelines' proportional hazards test for a fitted Cox model."""

    *_, proportional_hazard_test, _ = _require_lifelines()
    result = proportional_hazard_test(fit.model, fit.fit_data, time_transform=time_transform)
    return result.summary.reset_index().rename(columns={"index": "feature"})


def _save_or_return(fig, output_path: str | Path | None):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return fig


def save_placeholder_plot(output_path: str | Path, title: str, message: str) -> None:
    """Save a small explanatory figure when a validation plot cannot be made."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axis_off()
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cox_forest(
    cox_summary: pd.DataFrame,
    output_path: str | Path | None = None,
    hazard_ratio: bool = True,
    title: str = "Cox PH feature effects",
):
    """Plot Cox coefficients or hazard ratios with 95% confidence intervals."""

    if cox_summary.empty:
        raise ValueError("cox_summary is empty.")

    df = cox_summary.copy()
    if "feature" not in df.columns:
        df = df.reset_index().rename(columns={"index": "feature", "covariate": "feature"})
    if hazard_ratio:
        x_col = "exp(coef)"
        low_col = "exp(coef) lower 95%"
        high_col = "exp(coef) upper 95%"
        null_value = 1.0
        x_label = "Hazard ratio"
    else:
        x_col = "coef"
        low_col = "coef lower 95%"
        high_col = "coef upper 95%"
        null_value = 0.0
        x_label = "Cox coefficient (log hazard ratio)"

    required = {"feature", x_col, low_col, high_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cox_summary is missing required columns for plotting: {sorted(missing)}")

    df = df.sort_values(x_col, ascending=True)
    y = np.arange(len(df))
    x = df[x_col].to_numpy(dtype=float)
    lower = df[low_col].to_numpy(dtype=float)
    upper = df[high_col].to_numpy(dtype=float)
    xerr = np.vstack([x - lower, upper - x])
    colors = np.where(x >= null_value, "#b2182b", "#2166ac")

    height = max(4.0, 0.35 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#666666", capsize=2, zorder=1)
    ax.scatter(x, y, c=colors, s=36, zorder=2)
    ax.axvline(null_value, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Feature")
    ax.set_title(title)
    if hazard_ratio:
        ax.set_xscale("log")
    if "p" in df:
        for y_pos, p_value in zip(y, df["p"]):
            ax.text(1.01, y_pos, f"p={p_value:.3g}", va="center", fontsize=8, transform=ax.get_yaxis_transform())
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_penalized_coefficients(
    coefficient_summary: pd.DataFrame,
    output_path: str | Path | None = None,
    top_n: int = 30,
    title: str = "Penalised Cox coefficients",
):
    """Plot Ridge Cox or Coxnet coefficients sorted by absolute effect size."""

    if coefficient_summary.empty:
        raise ValueError("coefficient_summary is empty.")
    required = {"feature", "coef", "abs_coef"}
    missing = required - set(coefficient_summary.columns)
    if missing:
        raise ValueError(f"coefficient_summary is missing required columns: {sorted(missing)}")

    df = coefficient_summary.copy()
    if "nonzero" in df:
        df = df[df["nonzero"].astype(bool)]
    if df.empty:
        df = coefficient_summary.copy()
    df = df.sort_values("abs_coef", ascending=False).head(top_n).sort_values("coef", ascending=True)

    colors = np.where(df["coef"].to_numpy(dtype=float) >= 0, "#b2182b", "#2166ac")
    height = max(4.0, 0.35 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(np.arange(len(df)), df["coef"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Coefficient (log hazard ratio)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_coefficient_path(
    path_coefficients: pd.DataFrame,
    output_path: str | Path | None = None,
    selected_alpha: float | None = None,
    max_features: int = 40,
    title: str = "Penalised Cox coefficient path",
):
    """Plot coefficient trajectories across a Coxnet or Ridge alpha path."""

    if path_coefficients.empty:
        raise ValueError("path_coefficients is empty.")
    required = {"feature", "alpha", "coef"}
    missing = required - set(path_coefficients.columns)
    if missing:
        raise ValueError(f"path_coefficients is missing required columns: {sorted(missing)}")

    df = path_coefficients.copy()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
    df = df.dropna(subset=["feature", "alpha", "coef"])
    df = df[df["alpha"] > 0]
    if df.empty:
        raise ValueError("No finite positive-alpha coefficients are available to plot.")

    wide = df.pivot_table(index="feature", columns="alpha", values="coef", aggfunc="first")
    wide = wide.reindex(sorted(wide.columns), axis=1)
    alphas = wide.columns.to_numpy(dtype=float)
    coefficients = wide.to_numpy(dtype=float)
    finite_mask = np.isfinite(coefficients)
    if not finite_mask.any():
        raise ValueError("No finite coefficients are available to plot.")

    abs_coefficients = np.where(finite_mask, np.abs(coefficients), 0.0)
    max_abs = abs_coefficients.max(axis=1)
    active_indices = np.where(max_abs > 1e-12)[0]
    if len(active_indices) == 0:
        active_indices = np.argsort(max_abs)[-min(max_features, len(max_abs)):]
    elif len(active_indices) > max_features:
        active_indices = active_indices[np.argsort(max_abs[active_indices])[-max_features:]]

    log_alphas = np.log10(alphas)
    x_min = float(np.nanmin(log_alphas))
    x_max = float(np.nanmax(log_alphas))
    x_range = x_max - x_min if x_max > x_min else 1.0
    label_x = x_min - 0.03 * x_range
    feature_names = wide.index.astype(str).tolist()

    label_active_indices = []
    label_anchor_indices = []
    label_targets = []
    for index in active_indices:
        finite_indices = np.where(np.isfinite(coefficients[index, :]))[0]
        if len(finite_indices) == 0:
            continue
        anchor_index = int(finite_indices[np.argmin(alphas[finite_indices])])
        label_active_indices.append(index)
        label_anchor_indices.append(anchor_index)
        label_targets.append(coefficients[index, anchor_index])
    label_positions = spread_label_positions(np.asarray(label_targets, dtype=float))
    label_lookup = {
        index: (anchor_index, label_positions[label_index])
        for label_index, (index, anchor_index) in enumerate(zip(label_active_indices, label_anchor_indices))
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.colormaps["tab20"].resampled(max(len(active_indices), 1))
    for color_index, index in enumerate(active_indices):
        color = cmap(color_index)
        ax.plot(log_alphas, coefficients[index, :], linewidth=1.4, color=color, label=feature_names[index])
        if index in label_lookup:
            anchor_index, label_y = label_lookup[index]
            end_y = coefficients[index, anchor_index]
            ax.plot(
                [log_alphas[anchor_index], label_x + 0.006 * x_range],
                [end_y, label_y],
                color=color,
                linewidth=0.8,
            )
            ax.text(
                label_x,
                label_y,
                feature_names[index],
                color=color,
                ha="right",
                va="center",
                fontsize=7 if len(active_indices) <= 25 else 6,
            )

    if selected_alpha is not None and selected_alpha > 0:
        ax.axvline(
            np.log10(float(selected_alpha)),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="selected alpha",
        )

    ax.set_xlim(label_x - 0.18 * x_range, x_max + 0.03 * x_range)
    ax.margins(y=0.08)
    ax.set_xlabel("log10(alpha; higher = stronger regularisation)")
    ax.set_ylabel("Coefficient (standardised feature scale)")
    ax.set_title(title)
    if len(active_indices) <= 25:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.tight_layout()
    _save_or_return(fig, output_path)

    if output_path is not None and len(active_indices) > 25:
        output_path = Path(output_path)
        legend_path = output_path.with_name(f"{output_path.stem}_legend{output_path.suffix}")
        legend_height = max(4.0, 0.24 * len(active_indices))
        legend_fig, legend_ax = plt.subplots(figsize=(6, legend_height))
        legend_ax.set_axis_off()
        for color_index, index in enumerate(active_indices):
            y_value = 1 - (color_index + 0.5) / len(active_indices)
            legend_ax.plot([0.02, 0.12], [y_value, y_value], color=cmap(color_index), linewidth=2.0)
            legend_ax.text(0.15, y_value, feature_names[index], va="center", fontsize=8)
        legend_fig.tight_layout()
        legend_fig.savefig(legend_path, dpi=200, bbox_inches="tight")
        plt.close(legend_fig)

    return fig


def plot_feature_correlation(
    case_table: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    method: str = "spearman",
    max_features: int = 60,
    output_path: str | Path | None = None,
    title: str = "Case-level feature correlation",
):
    """Plot a case-level feature correlation heatmap."""

    feature_cols = list(feature_cols or get_feature_columns(case_table))
    if len(feature_cols) < 2:
        raise ValueError("At least two feature columns are required for a correlation heatmap.")

    X = case_table[feature_cols].copy()
    if len(feature_cols) > max_features:
        selected = X.var(axis=0).sort_values(ascending=False).head(max_features).index.tolist()
        X = X[selected]
    corr = X.corr(method=method)
    n_features = corr.shape[0]
    fig_size = max(5.0, min(18.0, 0.32 * n_features + 3.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cmap = "vlag" if "vlag" in plt.colormaps() else "coolwarm"
    image = ax.imshow(corr.to_numpy(dtype=float), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    label_size = 6 if n_features > 25 else 8
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=label_size)
    ax.set_yticklabels(corr.index, fontsize=label_size)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=f"{method} correlation")
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_validation_summary(
    metrics: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Case-level Cox cross-validation",
):
    """Plot held-out C-index values from :func:`cross_validate_cox_model`."""

    if metrics.empty:
        raise ValueError("metrics is empty.")
    valid = metrics[metrics["fit_status"] == "ok"].copy()
    if valid.empty:
        raise ValueError("No successful validation folds were found.")

    summary = valid.groupby("repeat", as_index=False).agg(
        mean_heldout_c_index=("heldout_c_index", "mean"),
        std_heldout_c_index=("heldout_c_index", "std"),
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        summary["repeat"],
        summary["mean_heldout_c_index"],
        yerr=summary["std_heldout_c_index"].fillna(0.0),
        fmt="o-",
        color="#4c78a8",
        capsize=3,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Repeat")
    ax.set_ylabel("Held-out C-index")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_risk_distribution(
    risk_table: pd.DataFrame,
    output_path: str | Path | None = None,
    risk_col: str = "risk_score",
    title: str = "Risk score distribution",
):
    """Plot risk-score distributions split by event status."""

    if risk_table.empty:
        raise ValueError("risk_table is empty.")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    events = risk_table[risk_table["event"].astype(bool)][risk_col]
    censored = risk_table[~risk_table["event"].astype(bool)][risk_col]
    bins = min(20, max(6, int(np.sqrt(len(risk_table)))))
    ax.hist(censored, bins=bins, alpha=0.65, label=f"censored (n={len(censored)})", color="#4c78a8")
    ax.hist(events, bins=bins, alpha=0.65, label=f"event (n={len(events)})", color="#b2182b")
    ax.set_xlabel(risk_col)
    ax.set_ylabel("Cases")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_risk_vs_survival(
    risk_table: pd.DataFrame,
    output_path: str | Path | None = None,
    risk_col: str = "risk_score",
    title: str = "Risk score vs observed survival",
):
    """Plot risk score against observed survival time."""

    if risk_table.empty:
        raise ValueError("risk_table is empty.")
    fig, ax = plt.subplots(figsize=(7, 5))
    for event_value, color, marker, label in [(False, "#4c78a8", "s", "censored"), (True, "#b2182b", "o", "event")]:
        mask = risk_table["event"].astype(bool) == event_value
        ax.scatter(
            risk_table.loc[mask, risk_col],
            risk_table.loc[mask, "duration"],
            c=color,
            marker=marker,
            s=38,
            alpha=0.85,
            linewidths=0.3,
            edgecolors="black",
            label=label,
        )
    ax.set_xlabel(risk_col)
    ax.set_ylabel("Observed time")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_kaplan_meier_by_risk_group(
    risk_table: pd.DataFrame,
    output_path: str | Path | None = None,
    duration_col: str = "duration",
    event_col: str = "event",
    group_col: str = "risk_group",
    group_order_col: str = "risk_group_order",
    extreme_groups_only: bool = False,
    title: str = "Observed survival by risk group",
):
    """Plot Kaplan-Meier curves for fitted or held-out risk groups."""

    _, KaplanMeierFitter, multivariate_logrank_test, *_ = _require_lifelines()
    df = risk_table.dropna(subset=[group_col, duration_col, event_col]).copy()
    if df.empty:
        raise ValueError("risk_table has no complete risk group rows.")

    group_order = df[[group_col, group_order_col]].drop_duplicates().sort_values(group_order_col)
    if extreme_groups_only and group_order.shape[0] >= 2:
        keep_orders = {group_order[group_order_col].min(), group_order[group_order_col].max()}
        df = df[df[group_order_col].isin(keep_orders)].copy()
        group_order = group_order[group_order[group_order_col].isin(keep_orders)]
        title = f"{title}: lowest vs highest"

    if group_order.shape[0] < 2:
        raise ValueError("At least two risk groups are required for Kaplan-Meier plotting.")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = plt.colormaps["Dark2"].resampled(group_order.shape[0])
    kmf = KaplanMeierFitter()
    for color_index, (_, row) in enumerate(group_order.iterrows()):
        group_name = row[group_col]
        group_df = df[df[group_col] == group_name]
        kmf.fit(
            group_df[duration_col].astype(float),
            event_observed=group_df[event_col].astype(bool),
            label=f"{group_name} (n={len(group_df)}, events={int(group_df[event_col].sum())})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=cmap(color_index))

    try:
        logrank = multivariate_logrank_test(
            df[duration_col].astype(float),
            df[group_col],
            df[event_col].astype(bool),
        )
        subtitle = f"log-rank p={logrank.p_value:.3g}"
    except Exception:
        subtitle = "log-rank p=NA"

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{title}\n{subtitle}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_cv_validation_outputs(
    cv_metrics: pd.DataFrame,
    cv_predictions: pd.DataFrame,
    output_dir: str | Path,
    quantiles: Sequence[float] = (0.0, 0.33, 0.67, 1.0),
    prefix: str = "cox",
) -> pd.DataFrame:
    """Write standard validation plots and return aggregated held-out risk scores."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_cv_predictions(cv_predictions, quantiles=quantiles)
    if not cv_metrics.empty:
        try:
            plot_validation_summary(cv_metrics, output_dir / f"{prefix}_cv_c_index.png")
        except Exception as exc:
            save_placeholder_plot(
                output_dir / f"{prefix}_cv_c_index.png",
                "Case-level Cox cross-validation",
                f"Could not plot validation C-index: {exc}",
            )
    if not aggregate.empty:
        cv_plot_table = aggregate.rename(columns={"mean_heldout_risk_score": "risk_score"})
        for plot_func, filename, plot_title in [
            (plot_risk_distribution, f"{prefix}_cv_heldout_risk_distribution.png", "Held-out risk distribution"),
            (plot_risk_vs_survival, f"{prefix}_cv_heldout_risk_vs_survival.png", "Held-out risk vs survival"),
        ]:
            try:
                plot_func(cv_plot_table, output_dir / filename)
            except Exception as exc:
                save_placeholder_plot(output_dir / filename, plot_title, f"Could not create plot: {exc}")
        try:
            plot_kaplan_meier_by_risk_group(
                aggregate,
                output_dir / f"{prefix}_cv_heldout_km_by_risk_group.png",
                title="Observed survival by held-out Cox risk group",
            )
        except Exception as exc:
            save_placeholder_plot(
                output_dir / f"{prefix}_cv_heldout_km_by_risk_group.png",
                "Observed survival by held-out Cox risk group",
                f"Could not create Kaplan-Meier plot: {exc}",
            )
    return aggregate


def plot_schoenfeld_residuals(
    fit: CoxFitResult,
    features: Sequence[str] | None = None,
    output_path: str | Path | None = None,
):
    """Plot scaled Schoenfeld residuals for proportional hazards diagnostics."""

    features = list(features or fit.feature_columns)
    residuals = fit.model.compute_residuals(fit.fit_data, kind="scaled_schoenfeld")
    features = [feature for feature in features if feature in residuals.columns]
    if not features:
        raise ValueError("No requested features were found in the Schoenfeld residual table.")

    n_features = len(features)
    ncols = 2 if n_features > 1 else 1
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.2 * nrows), squeeze=False)
    try:
        times = fit.fit_data.loc[residuals.index, fit.duration_col].astype(float)
    except Exception:
        times = pd.Series(np.arange(len(residuals)), index=residuals.index)

    for ax, feature in zip(axes.ravel(), features):
        ax.scatter(times, residuals[feature], s=18, alpha=0.75, color="#4c78a8")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(feature)
        ax.set_xlabel("Time")
        ax.set_ylabel("Scaled Schoenfeld residual")
        ax.grid(True, alpha=0.2)
    for ax in axes.ravel()[len(features):]:
        ax.set_axis_off()
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def run_cox_survival_analysis(
    adata: ad.AnnData,
    population_obs: str | Sequence[str],
    duration_obs: str,
    event_obs: str | None = None,
    case_obs: str = "Case",
    roi_obs: str | None = "ROI",
    covariate_obs: str | Sequence[str] | None = None,
    continuous_obs: str | Sequence[str] | None = None,
    case_aggregation: str = "weighted",
    model: str = "coxph",
    max_features: int = 20,
    p_value_threshold: float | None = None,
    cox_penalizer: float = 0.1,
    cox_l1_ratio: float = 0.0,
    ridge_alpha: float = 1.0,
    coxnet_alpha: float | None = None,
    coxnet_l1_ratio: float = 0.5,
    coxnet_n_alphas: int = 100,
    coxnet_alpha_min_ratio: float = 0.01,
    coxnet_max_iter: int = 100000,
    coxnet_tol: float = 1e-7,
    ridge_path_alphas: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    standardize: bool = True,
    cv_folds: int = 5,
    cv_repeats: int = 5,
    seed: int = 1,
    output_dir: str | Path | None = None,
    create_plots: bool = True,
    create_path_plots: bool = True,
    assume_all_events: bool = False,
    **case_table_kwargs,
) -> CoxPipelineResult:
    """
    Run a complete AnnData-to-Cox workflow.

    This is a notebook-friendly wrapper around the lower-level functions:
    build a case-level table from ``adata.obs``, rank univariate associations,
    fit a multivariable Cox model, run repeated case-level cross-validation,
    and optionally write validation plots/tables. ``model`` can be ``"coxph"``
    for lifelines Cox PH, ``"ridge"``/``"ridge_cox"`` for scikit-survival Ridge
    Cox, or ``"coxnet"`` for scikit-survival elastic-net Cox.
    """

    model = str(model).lower()
    model_aliases = {
        "cox": "coxph",
        "cox_ph": "coxph",
        "coxph": "coxph",
        "ridge": "ridge",
        "ridge_cox": "ridge",
        "ridge-cox": "ridge",
        "coxnet": "coxnet",
        "cox_net": "coxnet",
        "cox-net": "coxnet",
    }
    if model not in model_aliases:
        raise ValueError("model must be one of 'coxph', 'ridge', 'ridge_cox', or 'coxnet'.")
    model = model_aliases[model]

    case_table = build_case_level_table(
        adata=adata,
        population_obs=population_obs,
        duration_obs=duration_obs,
        event_obs=event_obs,
        case_obs=case_obs,
        roi_obs=roi_obs,
        covariate_obs=covariate_obs,
        continuous_obs=continuous_obs,
        case_aggregation=case_aggregation,
        assume_all_events=assume_all_events,
        **case_table_kwargs,
    )
    feature_cols = get_feature_columns(case_table)
    if not feature_cols:
        raise ValueError("No non-constant Cox features remained after aggregation.")

    univariate = fit_univariate_cox(case_table, feature_cols=feature_cols, penalizer=0.0)
    selected = select_top_features(
        univariate,
        top_n=max_features,
        p_value_threshold=p_value_threshold,
    )
    if not selected:
        selected = feature_cols[:max_features]

    if model == "coxph":
        fit = fit_cox_model(
            case_table,
            feature_cols=selected,
            penalizer=cox_penalizer,
            l1_ratio=cox_l1_ratio,
            standardize=standardize,
        )
        cv_metrics, cv_predictions = cross_validate_cox_model(
            case_table,
            feature_cols=selected,
            n_splits=cv_folds,
            repeats=cv_repeats,
            penalizer=cox_penalizer,
            l1_ratio=cox_l1_ratio,
            standardize=standardize,
            seed=seed,
        )
    elif model == "ridge":
        fit = fit_ridge_cox_model(
            case_table,
            feature_cols=selected,
            alpha=ridge_alpha,
            standardize=standardize,
        )
        cv_metrics, cv_predictions = cross_validate_sksurv_cox_model(
            case_table,
            feature_cols=selected,
            model_type="ridge",
            n_splits=cv_folds,
            repeats=cv_repeats,
            ridge_alpha=ridge_alpha,
            standardize=standardize,
            seed=seed,
        )
    else:
        fit = fit_coxnet_model(
            case_table,
            feature_cols=selected,
            alpha=coxnet_alpha,
            l1_ratio=coxnet_l1_ratio,
            n_alphas=coxnet_n_alphas,
            alpha_min_ratio=coxnet_alpha_min_ratio,
            max_iter=coxnet_max_iter,
            tol=coxnet_tol,
            cv_folds=cv_folds,
            seed=seed,
            standardize=standardize,
        )
        cv_metrics, cv_predictions = cross_validate_sksurv_cox_model(
            case_table,
            feature_cols=selected,
            model_type="coxnet",
            n_splits=cv_folds,
            repeats=cv_repeats,
            coxnet_alpha=coxnet_alpha,
            coxnet_l1_ratio=coxnet_l1_ratio,
            coxnet_n_alphas=coxnet_n_alphas,
            coxnet_alpha_min_ratio=coxnet_alpha_min_ratio,
            coxnet_max_iter=coxnet_max_iter,
            coxnet_tol=coxnet_tol,
            standardize=standardize,
            seed=seed,
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = fit.model_type
        case_table.to_csv(output_dir / "cox_case_table.csv")
        univariate.to_csv(output_dir / "cox_univariate_results.csv", index=False)
        fit.summary.to_csv(output_dir / f"{output_prefix}_summary.csv", index=False)
        cv_metrics.to_csv(output_dir / f"{output_prefix}_cv_metrics.csv", index=False)
        cv_predictions.to_csv(output_dir / f"{output_prefix}_cv_predictions.csv", index=False)
        summarise_cv_metrics(cv_metrics).to_csv(output_dir / f"{output_prefix}_cv_summary.csv", index=False)
        alpha_scores = (fit.metadata or {}).get("alpha_cv_scores")
        if isinstance(alpha_scores, pd.DataFrame) and not alpha_scores.empty:
            alpha_scores.to_csv(output_dir / f"{output_prefix}_alpha_cv_scores.csv", index=False)

        if create_plots:
            coefficient_plot_path = output_dir / f"{output_prefix}_coefficients.png"
            try:
                if fit.model_type == "coxph":
                    coefficient_plot_path = output_dir / f"{output_prefix}_forest.png"
                    plot_cox_forest(fit.summary, coefficient_plot_path)
                else:
                    plot_penalized_coefficients(
                        fit.summary,
                        coefficient_plot_path,
                        title=f"{fit.model_type} Cox coefficients",
                    )
            except Exception as exc:
                save_placeholder_plot(
                    coefficient_plot_path,
                    "Cox feature effects",
                    f"Could not create plot: {exc}",
                )
            if len(feature_cols) > 1:
                try:
                    plot_feature_correlation(
                        case_table,
                        feature_cols,
                        output_path=output_dir / "cox_feature_correlation.png",
                    )
                except Exception as exc:
                    save_placeholder_plot(
                        output_dir / "cox_feature_correlation.png",
                        "Case-level feature correlation",
                        f"Could not create plot: {exc}",
                    )
            if create_path_plots and fit.model_type in {"ridge", "coxnet"}:
                try:
                    if fit.model_type == "coxnet":
                        path_coefficients = coxnet_path_coefficients(
                            case_table,
                            feature_cols=selected,
                            l1_ratio=coxnet_l1_ratio,
                            n_alphas=coxnet_n_alphas,
                            alpha_min_ratio=coxnet_alpha_min_ratio,
                            max_iter=coxnet_max_iter,
                            tol=coxnet_tol,
                            standardize=standardize,
                        )
                        path_title = "Coxnet coefficient path"
                    else:
                        path_coefficients = ridge_cox_path_coefficients(
                            case_table,
                            feature_cols=selected,
                            alphas=ridge_path_alphas,
                            standardize=standardize,
                        )
                        path_title = "Ridge Cox coefficient shrinkage path"

                    path_coefficients.to_csv(output_dir / f"{output_prefix}_path_coefficients.csv", index=False)
                    plot_coefficient_path(
                        path_coefficients,
                        output_dir / f"{output_prefix}_path_plot.png",
                        selected_alpha=fit.alpha,
                        title=path_title,
                    )
                except Exception as exc:
                    save_placeholder_plot(
                        output_dir / f"{output_prefix}_path_plot.png",
                        "Penalised Cox coefficient path",
                        f"Could not create path plot: {exc}",
                    )
            risk_table = build_risk_table(fit, case_table)
            risk_table.to_csv(output_dir / f"{output_prefix}_fitted_risk_table.csv")
            for plot_func, filename, plot_title in [
                (plot_risk_distribution, f"{output_prefix}_fitted_risk_distribution.png", "Fitted risk distribution"),
                (plot_risk_vs_survival, f"{output_prefix}_fitted_risk_vs_survival.png", "Fitted risk vs survival"),
                (plot_kaplan_meier_by_risk_group, f"{output_prefix}_fitted_km_by_risk_group.png", "Fitted risk Kaplan-Meier"),
            ]:
                try:
                    plot_func(risk_table, output_dir / filename)
                except Exception as exc:
                    save_placeholder_plot(output_dir / filename, plot_title, f"Could not create plot: {exc}")
            cv_risk = plot_cv_validation_outputs(cv_metrics, cv_predictions, output_dir, prefix=output_prefix)
            cv_risk.to_csv(output_dir / f"{output_prefix}_cv_heldout_risk_table.csv")
            if fit.model_type == "coxph":
                try:
                    test_proportional_hazards(fit).to_csv(
                        output_dir / "coxph_proportional_hazards_test.csv",
                        index=False,
                    )
                    plot_schoenfeld_residuals(
                        fit,
                        output_path=output_dir / "coxph_schoenfeld_residuals.png",
                    )
                except Exception as exc:
                    pd.DataFrame([{"status": f"failed: {exc}"}]).to_csv(
                        output_dir / "coxph_proportional_hazards_test.csv",
                        index=False,
                    )

    return CoxPipelineResult(
        case_table=case_table,
        feature_columns=feature_cols,
        univariate_results=univariate,
        selected_features=selected,
        fit=fit,
        cv_metrics=cv_metrics,
        cv_predictions=cv_predictions,
    )


@dataclass(frozen=True)
class CoxFeatureSource:
    """In-memory observation source used by multi-source Cox preparation."""

    name: str
    obs: pd.DataFrame
    population_obs: tuple[str, ...] = ()
    continuous_obs: tuple[str, ...] = ()
    case_obs: str | None = None
    roi_obs: str | None = "ROI"
    case_aggregation: str = "weighted"
    normalization: str = "fraction"
    include_population_counts: bool = False


def prepare_clinical_metadata(
    clinical: pd.DataFrame,
    *,
    case_col: str,
    duration_col: str,
    event_col: str | None,
    roi_col: str | None = "ROI",
    covariate_cols: Sequence[str] = (),
    censored_case_ids: Sequence[str] = (),
    assume_all_events: bool = False,
) -> pd.DataFrame:
    """Validate and normalize case/ROI survival metadata without aggregating features."""

    required = [case_col, duration_col, *covariate_cols]
    if roi_col:
        required.append(roi_col)
    if event_col:
        required.append(event_col)
    missing = [column for column in _ordered_unique(required) if column not in clinical]
    if missing:
        raise ValueError(f"Clinical metadata is missing required columns: {missing}")
    if event_col is None and not censored_case_ids and not assume_all_events:
        raise ValueError(
            "Clinical metadata needs event_col, censored_case_ids, or assume_all_events=True."
        )

    result = clinical[_ordered_unique(required)].copy()
    result[duration_col] = pd.to_numeric(result[duration_col], errors="coerce")
    if roi_col:
        result[roi_col] = result[roi_col].astype("string")
    if event_col:
        result["_cox_event"] = result[event_col].map(normalize_event_value)
    else:
        censored = {str(value) for value in censored_case_ids}
        result["_cox_event"] = (~result[case_col].astype(str).isin(censored)).astype(float)
    result["_cox_case"] = result[case_col].astype("string")
    result["_cox_duration"] = result[duration_col]
    result = result.dropna(subset=["_cox_case", "_cox_duration", "_cox_event"])
    result = result[result["_cox_duration"] > 0].copy()
    result["_cox_case"] = result["_cox_case"].astype(str)
    result["_cox_event"] = result["_cox_event"].astype(bool)
    if result.empty:
        raise ValueError("No cases with positive duration and valid event status remain.")
    return result


def _case_outcomes(
    clinical: pd.DataFrame,
    *,
    metadata_conflict: str,
) -> pd.DataFrame:
    outcomes = collapse_case_metadata(
        clinical,
        "_cox_case",
        ["_cox_duration", "_cox_event"],
        conflict=metadata_conflict,
    )
    outcomes.index = outcomes.index.astype(str)
    outcomes["_cox_duration"] = pd.to_numeric(outcomes["_cox_duration"], errors="coerce")
    outcomes["_cox_event"] = outcomes["_cox_event"].map(normalize_event_value)
    return outcomes


def _attach_source_outcomes(
    source: CoxFeatureSource,
    clinical: pd.DataFrame,
    *,
    clinical_roi_col: str | None,
    metadata_conflict: str,
) -> pd.DataFrame:
    obs = source.obs.copy()
    if source.case_obs:
        if source.case_obs not in obs:
            raise ValueError(
                f"Cox source {source.name!r} is missing configured case_obs "
                f"{source.case_obs!r}."
            )
        obs["_cox_case"] = obs[source.case_obs].astype("string")
    else:
        if not source.roi_obs or source.roi_obs not in obs:
            raise ValueError(
                f"Cox source {source.name!r} has no usable case_obs and is missing roi_obs "
                f"{source.roi_obs!r} for clinical mapping."
            )
        if not clinical_roi_col or clinical_roi_col not in clinical:
            raise ValueError(
                f"Cox source {source.name!r} needs ROI mapping, but clinical roi_col is unavailable."
            )
        roi_map = collapse_case_metadata(
            clinical,
            clinical_roi_col,
            "_cox_case",
            conflict=metadata_conflict,
        ).rename_axis(source.roi_obs)
        roi_map.index = roi_map.index.astype(str)
        obs[source.roi_obs] = obs[source.roi_obs].astype(str)
        obs = obs.join(roi_map, on=source.roi_obs, how="left")

    outcomes = _case_outcomes(clinical, metadata_conflict=metadata_conflict)
    obs["_cox_case"] = obs["_cox_case"].astype("string")
    obs = obs.join(outcomes, on="_cox_case", how="left")
    missing_case_rows = int(obs["_cox_duration"].isna().sum())
    if missing_case_rows == len(obs):
        raise ValueError(
            f"Cox source {source.name!r} did not map any observations to clinical outcomes."
        )
    return obs.dropna(subset=["_cox_case", "_cox_duration", "_cox_event"])


def build_multi_source_case_table(
    sources: Sequence[CoxFeatureSource],
    clinical: pd.DataFrame,
    *,
    clinical_case_col: str,
    clinical_duration_col: str,
    clinical_event_col: str | None,
    clinical_roi_col: str | None = "ROI",
    covariate_cols: Sequence[str] = (),
    censored_case_ids: Sequence[str] = (),
    assume_all_events: bool = False,
    metadata_conflict: str = "mode",
    min_observations_per_case: int = 1,
    min_rois_per_case: int = 1,
    min_feature_prevalence: float = 0.0,
) -> pd.DataFrame:
    """Combine case-level features from multiple AnnData observation sources.

    Sources may carry a case identifier directly or may be mapped to cases
    through their ROI column and the clinical table. Feature names are prefixed
    by source name, preventing collisions when population abundances from
    several analyses are joined.
    """

    if not sources:
        raise ValueError("At least one Cox feature source is required.")
    source_names = [source.name for source in sources]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Cox feature source names must be unique.")

    needs_roi_mapping = any(source.case_obs is None for source in sources)
    clinical_prepared = prepare_clinical_metadata(
        clinical,
        case_col=clinical_case_col,
        duration_col=clinical_duration_col,
        event_col=clinical_event_col,
        roi_col=clinical_roi_col if needs_roi_mapping else None,
        covariate_cols=covariate_cols,
        censored_case_ids=censored_case_ids,
        assume_all_events=assume_all_events,
    )
    source_tables: list[pd.DataFrame] = []
    source_feature_map: dict[str, list[str]] = {}
    for source in sources:
        mapped = _attach_source_outcomes(
            source,
            clinical_prepared,
            clinical_roi_col=clinical_roi_col,
            metadata_conflict=metadata_conflict,
        )
        mapped.index = mapped.index.astype(str)
        source_roi = source.roi_obs if source.roi_obs in mapped else None
        table = build_case_level_table(
            ad.AnnData(obs=mapped),
            population_obs=list(source.population_obs),
            continuous_obs=list(source.continuous_obs),
            duration_obs="_cox_duration",
            event_obs="_cox_event",
            case_obs="_cox_case",
            roi_obs=source_roi,
            case_aggregation=source.case_aggregation,
            normalization=source.normalization,
            include_population_counts=source.include_population_counts,
            metadata_conflict=metadata_conflict,
            min_cells_per_case=min_observations_per_case,
            min_rois_per_case=min_rois_per_case,
            min_feature_prevalence=min_feature_prevalence,
        )
        original_features = get_feature_columns(table)
        rename = {column: f"{_safe_name(source.name)}__{column}" for column in original_features}
        table = table.rename(columns=rename)
        source_feature_map[source.name] = [rename[column] for column in original_features]
        table = table.rename(
            columns={
                "n_cells": f"n_observations__{_safe_name(source.name)}",
                "n_rois": f"n_rois__{_safe_name(source.name)}",
            }
        )
        source_tables.append(table)

    combined = source_tables[0].copy()
    for source, table in zip(sources[1:], source_tables[1:]):
        comparison = combined[["duration", "event"]].join(
            table[["duration", "event"]],
            how="inner",
            lsuffix="_left",
            rsuffix="_right",
        )
        duration_match = np.isclose(
            comparison["duration_left"].astype(float),
            comparison["duration_right"].astype(float),
            equal_nan=False,
        )
        event_match = (
            comparison["event_left"].astype(bool)
            == comparison["event_right"].astype(bool)
        ).to_numpy()
        if not bool(np.all(duration_match & event_match)):
            raise ValueError(
                f"Survival outcomes disagree after joining Cox source {source.name!r}."
            )
        feature_columns = [
            column for column in table if column not in {"duration", "event"}
        ]
        combined = combined.join(table[feature_columns], how="inner")

    covariates = encode_case_covariates(
        clinical_prepared,
        "_cox_case",
        list(covariate_cols),
        conflict=metadata_conflict,
        one_hot=True,
        drop_first=True,
    )
    covariates = covariates.rename(
        columns={column: f"clinical__{_safe_name(column)}" for column in covariates}
    )
    combined = combined.join(covariates, how="left")
    clinical_features = list(covariates.columns)
    image_features = [
        feature
        for features in source_feature_map.values()
        for feature in features
        if feature in combined
    ]
    usable_clinical = [
        column
        for column in clinical_features
        if column in combined
        and pd.to_numeric(combined[column], errors="coerce").notna().all()
        and combined[column].nunique(dropna=True) > 1
    ]
    combined = combined.drop(columns=set(clinical_features) - set(usable_clinical))
    combined.attrs["image_features"] = image_features
    combined.attrs["clinical_features"] = usable_clinical
    combined.attrs["source_features"] = source_feature_map
    combined.attrs["feature_columns"] = [*image_features, *usable_clinical]
    return combined
