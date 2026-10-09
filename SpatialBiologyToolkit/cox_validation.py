"""Training-only feature screening and nested case-level Cox validation.

This scientific module is loaded lazily by the Cox API/stage, never by the CLI.
Alpha fractions for Coxnet are relative to each training fold's alpha_max;
neither validation outcomes nor validation scaling determine that grid.
"""
from __future__ import annotations

import json
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import cox_survival as cxs


class CoxTuningError(RuntimeError):
    """A failed search retaining its per-fold evidence for reporting."""

    def __init__(self, scores):
        super().__init__("No Cox penalty completed all inner validation folds successfully.")
        self.scores = scores


def select_training_features(table, image_features, clinical_features=(),
                             max_image_features=None, p_value_threshold=None,
                             duration_col="duration", event_col="event"):
    """Screen only this training table, dropping its constant columns first."""
    image = [f for f in image_features if table[f].nunique() > 1]
    clinical = [f for f in clinical_features if table[f].nunique() > 1]
    if image and (max_image_features is not None or p_value_threshold is not None):
        ranking = cxs.fit_univariate_cox(
            table, feature_cols=image, duration_col=duration_col, event_col=event_col,
            penalizer=0.0, robust=False,
        )
        image = cxs.select_top_features(
            ranking, top_n=max_image_features or len(image),
            p_value_threshold=p_value_threshold,
        )
    features = list(dict.fromkeys([*image, *clinical]))
    if not features:
        raise ValueError("No usable features remain in this training fold.")
    return features


def _fit(table, features, model_type, alpha, *, duration_col="duration",
         event_col="event", standardize=True, coxnet_l1_ratio=0.5,
         coxnet_max_iter=100000, coxnet_tol=1e-7, cox_penalizer=0.1,
         cox_l1_ratio=0.0):
    common = dict(feature_cols=features, duration_col=duration_col,
                  event_col=event_col, standardize=standardize)
    if model_type == "coxnet":
        return cxs.fit_coxnet_model(table, alpha=alpha, l1_ratio=coxnet_l1_ratio,
                                   max_iter=coxnet_max_iter, tol=coxnet_tol, **common)
    if model_type == "ridge":
        return cxs.fit_ridge_cox_model(table, alpha=alpha, **common)
    if model_type == "coxph":
        return cxs.fit_cox_model(table, penalizer=cox_penalizer,
                                l1_ratio=cox_l1_ratio, robust=False, **common)
    raise ValueError(f"Unknown Cox model: {model_type}")


def _alpha_max(table, features, options):
    X, y, *_ = cxs._prepare_sksurv_inputs(
        table, features, options["duration_col"], options["event_col"], options["standardize"])
    return cxs._coxnet_alpha_grid(
        X, y, options["coxnet_l1_ratio"], 1, 0.01,
        options["coxnet_max_iter"], options["coxnet_tol"],
    )[0]


def _clean_table(table, features, duration_col, event_col):
    result = table[list(dict.fromkeys([*features, duration_col, event_col]))].dropna().copy()
    if not result.index.is_unique:
        raise ValueError("Cox validation requires one row per unique case ID.")
    if len(result) < 3 or not result[event_col].astype(bool).any():
        raise ValueError("Cox validation requires at least three cases and an observed event.")
    if not np.isfinite(result[duration_col]).all() or (result[duration_col] <= 0).any():
        raise ValueError("Survival durations must be finite and positive.")
    result[event_col] = result[event_col].astype(bool)
    return result


def tune_cox_penalty(case_table, image_features, clinical_features=(), *,
                     model_type="coxnet", max_image_features=None,
                     p_value_threshold=None, ridge_alphas=(0.01, 0.1, 1., 10., 100.),
                     n_splits=5, seed=1, duration_col="duration", event_col="event",
                     standardize=True, coxnet_l1_ratio=0.5, coxnet_n_alphas=100,
                     coxnet_alpha_min_ratio=0.01, coxnet_max_iter=100000,
                     coxnet_tol=1e-7):
    """Tune on training-only screens/scalers; require all inner folds to succeed.

    ``alpha`` in the returned table is the actual fold-specific penalty.
    ``alpha_fraction`` is the common tuning coordinate for Coxnet. The returned
    selected alpha is rescaled to the entire supplied training table only after
    selecting a fraction. An entirely unsuccessful search raises, never falls
    back to a near-null model that could masquerade as successful validation.
    """
    table = _clean_table(case_table, [*image_features, *clinical_features], duration_col, event_col)
    options = dict(duration_col=duration_col, event_col=event_col, standardize=standardize,
                   coxnet_l1_ratio=coxnet_l1_ratio, coxnet_max_iter=coxnet_max_iter,
                   coxnet_tol=coxnet_tol)
    selection = dict(max_image_features=max_image_features, p_value_threshold=p_value_threshold,
                     duration_col=duration_col, event_col=event_col)
    if model_type == "coxnet":
        if coxnet_n_alphas < 1 or not 0 < coxnet_alpha_min_ratio < 1:
            raise ValueError("Invalid Coxnet alpha-grid parameters.")
        candidates = np.geomspace(1., coxnet_alpha_min_ratio, coxnet_n_alphas)
    elif model_type == "ridge":
        candidates = np.asarray(sorted(set(ridge_alphas), reverse=True), dtype=float)
        if not len(candidates) or not np.isfinite(candidates).all() or (candidates < 0).any():
            raise ValueError("Ridge alphas must be finite and non-negative.")
    else:
        raise ValueError("Penalty tuning supports ridge or coxnet.")
    folds = min(int(n_splits), len(table))
    if folds < 2:
        raise ValueError("At least two inner folds are required.")
    rows = []
    for fold, (itr, ite) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(table), 1):
        train, test = table.iloc[itr], table.iloc[ite]
        setup_error = None
        features, scale = [], np.nan
        try:
            features = select_training_features(train, image_features, clinical_features, **selection)
            scale = _alpha_max(train, features, options) if model_type == "coxnet" else 1.
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            setup_error = exc
        for candidate in candidates:
            alpha = float(candidate * scale)
            status, fit_warnings, score, train_score = "ok", "", np.nan, np.nan
            try:
                if setup_error is not None:
                    raise ValueError(str(setup_error))
                fit = _fit(train, features, model_type, alpha, **options)
                risk = cxs._predict_log_hazard(fit, test).to_numpy()
                if not np.isfinite(risk).all():
                    raise ValueError("Nonfinite validation predictions.")
                score = cxs._sksurv_c_index(cxs.make_survival_array(test, duration_col, event_col), risk)
                if not np.isfinite(score):
                    raise ValueError("No comparable survival pairs in validation fold.")
                train_score = cxs._sksurv_c_index(
                    cxs.make_survival_array(train, duration_col, event_col),
                    cxs._predict_log_hazard(fit, train).to_numpy())
                fit_warnings = getattr(fit.model, "sbt_fit_warnings_", "")
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                status = f"failed: {exc}"
            rows.append(dict(fold=fold, candidate=float(candidate), alpha=alpha,
                             alpha_fraction=float(candidate) if model_type == "coxnet" else np.nan,
                             c_index=score, status=status, fit_warnings=fit_warnings,
                             train_c_index=train_score, n_features=len(features),
                             selected_features=json.dumps(features)))
    scores = pd.DataFrame(rows)
    # A failed candidate must not win on an easier subset of the folds.
    valid_scores = scores.assign(valid_score=scores.c_index.where(scores.status.eq("ok")))
    summary = valid_scores.groupby("candidate", as_index=False).agg(
        mean_c_index=("valid_score", "mean"), std_c_index=("valid_score", "std"),
        n_valid_folds=("valid_score", "count"))
    eligible = summary[summary.n_valid_folds == folds].sort_values(
        ["mean_c_index", "candidate"], ascending=[False, False])
    if eligible.empty:
        raise CoxTuningError(scores.merge(summary, on="candidate", how="left").assign(eligible=False))
    chosen = float(eligible.iloc[0].candidate)
    final_features = select_training_features(table, image_features, clinical_features, **selection)
    scale = _alpha_max(table, final_features, options) if model_type == "coxnet" else 1.
    selected_alpha = chosen * scale
    scores = scores.merge(summary, on="candidate", how="left")
    scores["selected_alpha"] = selected_alpha
    scores["selected_candidate"] = chosen
    scores["eligible"] = scores.n_valid_folds == folds
    return scores, float(selected_alpha)


def cross_validate_selected_cox_model(case_table, image_features: Sequence[str],
                                      clinical_features: Sequence[str] = (), *,
                                      model_type="coxnet", max_image_features=None,
                                      p_value_threshold=None, n_splits=5, repeats=5,
                                      ridge_alphas=(1.,), coxnet_alpha=None,
                                      duration_col="duration", event_col="event",
                                      standardize=True, seed=1, cox_penalizer=0.1,
                                      cox_l1_ratio=0.0, coxnet_l1_ratio=0.5,
                                      coxnet_n_alphas=100, coxnet_alpha_min_ratio=0.01,
                                      coxnet_max_iter=100000, coxnet_tol=1e-7):
    """Repeated outer validation of screening + tuning + fitting as one process.

    Returns metrics, predictions, and inner-search audit rows. The feature list
    supplied here must be the candidate pool, not an outcome-screened full-data
    list. Fixed penalties are allowed only as prespecified values.
    """
    table = _clean_table(case_table, [*image_features, *clinical_features], duration_col, event_col)
    if n_splits < 2 or repeats < 1:
        raise ValueError("Require n_splits >= 2 and repeats >= 1.")
    options = dict(duration_col=duration_col, event_col=event_col, standardize=standardize,
                   coxnet_l1_ratio=coxnet_l1_ratio, coxnet_max_iter=coxnet_max_iter,
                   coxnet_tol=coxnet_tol)
    selection = dict(max_image_features=max_image_features, p_value_threshold=p_value_threshold,
                     duration_col=duration_col, event_col=event_col)
    metrics, predictions, inner_scores = [], [], []
    for repeat in range(repeats):
        splitter = KFold(min(n_splits, len(table)), shuffle=True, random_state=seed + repeat)
        for fold, (itr, ite) in enumerate(splitter.split(table), 1):
            train, test = table.iloc[itr], table.iloc[ite]
            features, status, fit_warnings = [], "ok", ""
            alpha, train_score, heldout_score = np.nan, np.nan, np.nan
            risk = np.full(len(test), np.nan)
            try:
                features = select_training_features(train, image_features, clinical_features, **selection)
                tune = ((model_type == "coxnet" and coxnet_alpha is None) or
                        (model_type == "ridge" and len(ridge_alphas) > 1))
                if tune:
                    search, alpha = tune_cox_penalty(
                        train, image_features, clinical_features, model_type=model_type,
                        max_image_features=max_image_features, p_value_threshold=p_value_threshold,
                        ridge_alphas=ridge_alphas, n_splits=n_splits,
                        seed=seed + repeat + fold, coxnet_n_alphas=coxnet_n_alphas,
                        coxnet_alpha_min_ratio=coxnet_alpha_min_ratio, **options)
                    inner_scores.append(search.assign(outer_repeat=repeat + 1, outer_fold=fold))
                elif model_type == "coxnet":
                    alpha = float(coxnet_alpha)
                elif model_type == "ridge":
                    alpha = float(ridge_alphas[0])
                fit = _fit(train, features, model_type, alpha, cox_penalizer=cox_penalizer,
                           cox_l1_ratio=cox_l1_ratio, **options)
                risk = cxs._predict_log_hazard(fit, test).to_numpy()
                if not np.isfinite(risk).all():
                    raise ValueError("Nonfinite held-out predictions.")
                heldout_score = cxs._sksurv_c_index(cxs.make_survival_array(test, duration_col, event_col), risk)
                train_score = cxs._sksurv_c_index(cxs.make_survival_array(train, duration_col, event_col),
                                                 cxs._predict_log_hazard(fit, train).to_numpy())
                if not np.isfinite(heldout_score):
                    raise ValueError("No comparable survival pairs in held-out fold.")
                fit_warnings = getattr(fit.model, "sbt_fit_warnings_", "")
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                if isinstance(exc, CoxTuningError):
                    inner_scores.append(exc.scores.assign(outer_repeat=repeat + 1, outer_fold=fold))
                status = f"failed: {exc}"
                risk[:] = np.nan
                train_score = heldout_score = np.nan
                logging.warning("Cox validation repeat %s fold %s: %s", repeat + 1, fold, status)
            metrics.append(dict(repeat=repeat + 1, fold=fold, model_type=model_type,
                                l1_ratio=coxnet_l1_ratio if model_type == "coxnet" else cox_l1_ratio if model_type == "coxph" else 0.,
                                alpha=alpha, n_train=len(train), n_test=len(test),
                                n_features=len(features), train_c_index=train_score,
                                heldout_c_index=heldout_score, fit_status=status,
                                standardized=standardize, validation_scheme="nested_training_only",
                                selected_features=json.dumps(features), fit_warnings=fit_warnings))
            for case_id, value, duration, event in zip(test.index.astype(str), risk,
                                                       test[duration_col], test[event_col]):
                predictions.append(dict(case_id=case_id, repeat=repeat + 1, fold=fold,
                                        model_type=model_type, heldout_risk_score=value,
                                        duration=duration, event=bool(event)))
    return (pd.DataFrame(metrics), pd.DataFrame(predictions),
            pd.concat(inner_scores, ignore_index=True) if inner_scores else pd.DataFrame())
