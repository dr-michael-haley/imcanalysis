"""Managed, general-purpose multi-source Cox survival analysis stage."""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "value"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run general multi-source Cox analysis.")
    parser.add_argument(
        "--config",
        default=os.environ.get("SBT_CONFIG", "config.yaml"),
        help="Pipeline config path (default: SBT_CONFIG or ./config.yaml).",
    )
    return parser.parse_args()


def _read_adata_obs(path: Path):
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        return data.obs.copy()
    finally:
        if getattr(data, "file", None) is not None:
            data.file.close()


def _load_clinical(config, source_obs: dict[str, Any]):
    import pandas as pd

    cox = config.cox
    if cox.clinical_csv_path:
        path = _project_path(cox.clinical_csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cox clinical CSV does not exist: {path}")
        return pd.read_csv(path), path
    if cox.clinical_adata_path:
        path = _project_path(cox.clinical_adata_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cox clinical AnnData does not exist: {path}")
        return _read_adata_obs(path), path

    general_path = _project_path(config.general.anndata_path)
    if general_path.is_file():
        return _read_adata_obs(general_path), general_path
    first_name = next(iter(source_obs))
    return source_obs[first_name].copy(), Path("<first-feature-source-obs>")


def _project_path(value: str | Path) -> Path:
    from SpatialBiologyToolkit.reporting import project_asset_path

    return project_asset_path(value)


def _resolved_sources(config):
    """Return explicit sources or useful SBT defaults for common workflows."""
    from SpatialBiologyToolkit.config.models import CoxFeatureSourceConfig

    if config.cox.feature_sources:
        return config.cox.feature_sources

    inferred = []
    general_path = _project_path(config.general.anndata_path)
    if general_path.is_file() and config.general.population_obs_primary:
        inferred.append(
            CoxFeatureSourceConfig(
                name="primary",
                adata_path=str(general_path),
                population_obs=[config.general.population_obs_primary],
                case_obs=config.general.case_obs or config.cox.case_col,
                roi_obs=config.general.roi_obs,
            )
        )
    hyperstac_path = _project_path(config.hyperstac.asset_folder) / "imc_hyperstac_representations.h5ad"
    if hyperstac_path.is_file():
        inferred.append(
            CoxFeatureSourceConfig(
                name="hyperstac",
                adata_path=str(hyperstac_path),
                population_obs_search=config.hyperstac.cluster_col_search or None,
                population_obs=(
                    []
                    if config.hyperstac.cluster_col_search
                    else [config.hyperstac.cluster_col]
                ),
                case_obs=None,
                roi_obs="roi",
            )
        )
    if not inferred:
        raise ValueError(
            "cox.feature_sources is empty and no source could be inferred. Configure at least "
            "one AnnData feature source, or set general.population_obs_primary."
        )
    return inferred


def _source_sweeps(source_configs, source_obs):
    sweep_sources = [
        source for source in source_configs if source.population_obs_search
    ]
    if len(sweep_sources) > 1:
        raise ValueError(
            "Only one cox.feature_sources entry may use population_obs_search in a run; "
            "multiple independent sweeps would create an ambiguous Cartesian product."
        )
    if not sweep_sources:
        return [(None, {})]
    source = sweep_sources[0]
    matches = [
        column
        for column in source_obs[source.name].columns
        if source.population_obs_search.lower() in str(column).lower()
    ]
    if not matches:
        raise ValueError(
            f"Cox source {source.name!r} found no obs columns matching "
            f"{source.population_obs_search!r}."
        )
    return [(column, {source.name: [column]}) for column in matches]


def _source_feature_objects(source_configs, source_obs, overrides):
    from SpatialBiologyToolkit.cox_survival import CoxFeatureSource

    return [
        CoxFeatureSource(
            name=source.name,
            obs=source_obs[source.name],
            population_obs=tuple(overrides.get(source.name, source.population_obs)),
            continuous_obs=tuple(source.continuous_obs),
            case_obs=source.case_obs,
            roi_obs=source.roi_obs,
            case_aggregation=source.case_aggregation,
            normalization=source.normalization,
            include_population_counts=source.include_population_counts,
        )
        for source in source_configs
    ]


def _selected_features(cxs, case_table, image_features, top_n):
    univariate = cxs.fit_univariate_cox(
        case_table,
        feature_cols=image_features,
        penalizer=0.0,
        robust=True,
    )
    selected = cxs.select_top_features(
        univariate,
        top_n=min(top_n, len(image_features)),
    )
    selected = [feature for feature in selected if feature in image_features]
    if not selected:
        selected = list(image_features)[: min(top_n, len(image_features))]
    return univariate, selected


def _ridge_alpha(cxs, case_table, features, config):
    import pandas as pd

    rows = []
    for alpha in config.ridge_alphas:
        metrics, _ = cxs.cross_validate_sksurv_cox_model(
            case_table,
            feature_cols=features,
            model_type="ridge",
            n_splits=config.validation_folds,
            repeats=config.validation_repeats,
            ridge_alpha=float(alpha),
            standardize=config.standardize,
            seed=config.seed,
        )
        summary = cxs.summarise_cv_metrics(metrics)
        row = {
            "alpha": float(alpha),
            "mean_heldout_c_index": float("nan"),
            "std_heldout_c_index": float("nan"),
            "n_folds": 0,
        }
        if not summary.empty:
            row.update(summary.iloc[0].to_dict())
            row["alpha"] = float(alpha)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["mean_heldout_c_index", "n_folds", "alpha"],
        ascending=[False, False, True],
        na_position="last",
    )
    usable = summary.dropna(subset=["mean_heldout_c_index"])
    if usable.empty:
        raise ValueError("Ridge Cox alpha selection produced no valid held-out C-index.")
    return float(usable.iloc[0]["alpha"]), summary


def _fit_model(cxs, case_table, model, features, config, *, select_alpha):
    if model == "coxph":
        fit = cxs.fit_cox_model(
            case_table,
            feature_cols=features,
            penalizer=config.coxph_penalizer,
            standardize=config.standardize,
            robust=True,
        )
        metrics, predictions = cxs.cross_validate_cox_model(
            case_table,
            feature_cols=features,
            n_splits=config.validation_folds,
            repeats=config.validation_repeats,
            penalizer=config.coxph_penalizer,
            standardize=config.standardize,
            seed=config.seed,
        )
        return fit, metrics, predictions, None
    if model == "ridge":
        alpha, alpha_summary = (
            _ridge_alpha(cxs, case_table, features, config)
            if select_alpha
            else (config.ridge_fixed_alpha, None)
        )
        fit = cxs.fit_ridge_cox_model(
            case_table,
            feature_cols=features,
            alpha=alpha,
            standardize=config.standardize,
        )
        metrics, predictions = cxs.cross_validate_sksurv_cox_model(
            case_table,
            feature_cols=features,
            model_type="ridge",
            n_splits=config.validation_folds,
            repeats=config.validation_repeats,
            ridge_alpha=alpha,
            standardize=config.standardize,
            seed=config.seed,
        )
        return fit, metrics, predictions, alpha_summary

    alpha_scores, alpha = cxs.fit_coxnet_alpha_cv(
        case_table,
        feature_cols=features,
        l1_ratio=config.coxnet_l1_ratio,
        n_alphas=config.coxnet_n_alphas,
        alpha_min_ratio=config.coxnet_alpha_min_ratio,
        max_iter=config.coxnet_max_iter,
        tol=config.coxnet_tolerance,
        n_splits=config.validation_folds,
        seed=config.seed,
        standardize=config.standardize,
    )
    fit = cxs.fit_coxnet_model(
        case_table,
        feature_cols=features,
        alpha=alpha,
        l1_ratio=config.coxnet_l1_ratio,
        n_alphas=config.coxnet_n_alphas,
        alpha_min_ratio=config.coxnet_alpha_min_ratio,
        max_iter=config.coxnet_max_iter,
        tol=config.coxnet_tolerance,
        cv_folds=config.validation_folds,
        seed=config.seed,
        standardize=config.standardize,
    )
    metrics, predictions = cxs.cross_validate_sksurv_cox_model(
        case_table,
        feature_cols=features,
        model_type="coxnet",
        n_splits=config.validation_folds,
        repeats=config.validation_repeats,
        coxnet_alpha=alpha,
        coxnet_l1_ratio=config.coxnet_l1_ratio,
        coxnet_n_alphas=config.coxnet_n_alphas,
        coxnet_alpha_min_ratio=config.coxnet_alpha_min_ratio,
        coxnet_max_iter=config.coxnet_max_iter,
        coxnet_tol=config.coxnet_tolerance,
        standardize=config.standardize,
        seed=config.seed,
    )
    return fit, metrics, predictions, alpha_scores


def _write_model_outputs(cxs, output, label, fit, metrics, predictions, alpha_table, config):
    output.mkdir(parents=True, exist_ok=True)
    fit.summary.to_csv(output / f"{label}_summary.csv", index=False)
    metrics.to_csv(output / f"{label}_cv_metrics.csv", index=False)
    predictions.to_csv(output / f"{label}_cv_predictions.csv", index=False)
    summary = cxs.summarise_cv_metrics(metrics)
    summary.to_csv(output / f"{label}_cv_summary.csv", index=False)
    if alpha_table is not None:
        alpha_table.to_csv(output / f"{label}_alpha_cv_scores.csv", index=False)

    if label == "coxph":
        cxs.plot_cox_forest(
            fit.summary,
            output / "coxph_forest.png",
            title="Conventional Cox PH",
        )
    else:
        cxs.plot_penalized_coefficients(
            fit.summary,
            output / f"{label}_coefficients.png",
            top_n=30,
            title=f"{label.replace('_', ' ').title()} coefficients",
        )
        if label == "ridge_cox":
            path = cxs.ridge_cox_path_coefficients(
                fit.fit_data,
                feature_cols=fit.feature_columns,
                alphas=config.ridge_alphas,
                standardize=config.standardize,
            )
        else:
            path = cxs.coxnet_path_coefficients(
                fit.fit_data,
                feature_cols=fit.feature_columns,
                l1_ratio=config.coxnet_l1_ratio,
                n_alphas=config.coxnet_n_alphas,
                alpha_min_ratio=config.coxnet_alpha_min_ratio,
                max_iter=config.coxnet_max_iter,
                tol=config.coxnet_tolerance,
                standardize=config.standardize,
            )
        path.to_csv(output / f"{label}_path_coefficients.csv", index=False)
        cxs.plot_coefficient_path(
            path,
            output / f"{label}_coefficient_path.png",
            selected_alpha=fit.alpha,
            max_features=30,
        )

    risk = cxs.build_risk_table(
        fit,
        fit.fit_data,
        quantiles=config.risk_group_quantiles,
    )
    risk.to_csv(output / f"{label}_fitted_risk_groups.csv")
    cxs.plot_risk_distribution(risk, output / f"{label}_fitted_risk_distribution.png")
    cxs.plot_risk_vs_survival(risk, output / f"{label}_fitted_risk_vs_survival.png")
    cxs.plot_kaplan_meier_by_risk_group(
        risk,
        output / f"{label}_fitted_risk_km.png",
    )
    cxs.plot_cv_validation_outputs(
        metrics,
        predictions,
        output,
        quantiles=config.risk_group_quantiles,
        prefix=label,
    )
    return summary


def _compatibility_feature_name(feature: str, dynamic_source: str | None, dynamic_col: str | None):
    if not dynamic_source or not dynamic_col:
        return feature
    source_prefix = f"{_safe_name(dynamic_source)}__"
    population_token = re.sub(r"[^0-9A-Za-z_]+", "_", dynamic_col).strip("_")
    prefix = f"{source_prefix}{population_token}_frac_"
    if feature.startswith(prefix):
        return f"cluster_freq_{feature[len(prefix):]}"
    return feature


def _write_compatibility_outputs(
    analysis_dir,
    case_table,
    univariate,
    selected,
    fits,
    dynamic_source,
    dynamic_col,
):
    def rename_feature(value: str) -> str:
        return _compatibility_feature_name(str(value), dynamic_source, dynamic_col)
    compat_case = case_table.copy().rename(columns=rename_feature)
    compat_case.to_csv(analysis_dir / "case_features.csv")
    selection = univariate.copy()
    selection["feature"] = selection["feature"].map(rename_feature)
    selection["selected"] = selection["feature"].isin({rename_feature(x) for x in selected})
    selection = selection.rename(
        columns={
            "coef": "univariate_coef",
            "concordance_index": "mean_cv_c_index",
        }
    )
    selection.to_csv(analysis_dir / "feature_selection.csv", index=False)

    primary = fits.get("clinical_image") or fits.get("image")
    if primary:
        for model, filename in (
            ("coxph", "standard_cox_results.csv"),
            ("ridge", "ridge_cox_coefficients.csv"),
            ("coxnet", "coxnet_coefficients.csv"),
        ):
            fit = primary.get(model)
            if fit is None:
                continue
            summary = fit.summary.copy()
            summary["feature"] = summary["feature"].map(rename_feature)
            summary = summary.rename(columns={"coef": "coefficient"})
            if "abs_coef" not in summary and "coefficient" in summary:
                summary["abs_coef"] = summary["coefficient"].abs()
            summary.to_csv(analysis_dir / filename, index=False)

    image = fits.get("image", {})
    for model, folder, filename in (
        ("ridge", "ridge_cox_image_only", "ridge_cox_coefficients.csv"),
        ("coxnet", "coxnet_image_only", "coxnet_coefficients.csv"),
    ):
        fit = image.get(model)
        if fit is None:
            continue
        target = analysis_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        summary = fit.summary.copy()
        summary["feature"] = summary["feature"].map(rename_feature)
        summary = summary.rename(columns={"coef": "coefficient"})
        summary.to_csv(target / filename, index=False)


def _run_analysis(cxs, case_table, output, config, dynamic_source, dynamic_col):
    import matplotlib.pyplot as plt
    import pandas as pd

    image_features = list(case_table.attrs["image_features"])
    clinical_features = list(case_table.attrs["clinical_features"])
    if not image_features:
        raise ValueError("No non-constant image-derived Cox features remain.")
    univariate, selected = _selected_features(
        cxs,
        case_table,
        image_features,
        config.feature_selection_top_n,
    )
    case_table.to_csv(output / "combined_case_level_features.csv")
    univariate.to_csv(output / "combined_image_univariate_cox.csv", index=False)
    pd.Series(selected, name="feature").to_csv(
        output / "selected_image_features.csv",
        index=False,
    )
    cxs.plot_feature_correlation(
        case_table,
        feature_cols=image_features,
        method=config.correlation_method,
        max_features=config.correlation_max_plot_features,
        output_path=output / "combined_image_feature_correlation.png",
        title="Combined image-derived case feature correlation",
    )

    feature_sets = {"image": selected}
    if clinical_features:
        feature_sets["clinical"] = clinical_features
        feature_sets["clinical_image"] = [*selected, *clinical_features]
    primary_set = "clinical_image" if "clinical_image" in feature_sets else "image"
    validation_rows = []
    fits: dict[str, dict[str, Any]] = {}
    failures = []
    for feature_set in config.feature_sets:
        features = feature_sets.get(feature_set)
        if not features:
            continue
        fits[feature_set] = {}
        for model in config.models:
            model_features = list(features)
            if model == "coxph" and feature_set != "clinical":
                image_part = [value for value in model_features if value in selected][
                    : config.coxph_max_features
                ]
                model_features = [
                    *image_part,
                    *[value for value in model_features if value in clinical_features],
                ]
            elif model == "ridge" and feature_set != "clinical":
                image_part = [value for value in model_features if value in selected][
                    : config.ridge_max_features
                ]
                model_features = [
                    *image_part,
                    *[value for value in model_features if value in clinical_features],
                ]
            try:
                fit, metrics, predictions, alpha_table = _fit_model(
                    cxs,
                    case_table,
                    model,
                    model_features,
                    config,
                    select_alpha=(model == "ridge" and feature_set == primary_set),
                )
                fits[feature_set][model] = fit
                label = "ridge_cox" if model == "ridge" else model
                model_output = output / "model_comparisons" / feature_set / label
                summary = _write_model_outputs(
                    cxs,
                    model_output,
                    label,
                    fit,
                    metrics,
                    predictions,
                    alpha_table,
                    config,
                )
                plt.close("all")
                row = {
                    "feature_set": feature_set,
                    "model": label,
                    "status": "ok",
                    "n_features": len(model_features),
                    "alpha": fit.alpha,
                }
                if not summary.empty:
                    row.update(summary.iloc[0].to_dict())
                validation_rows.append(row)
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                plt.close("all")
                logging.warning(
                    "Cox model failed for %s/%s: %s",
                    feature_set,
                    model,
                    exc,
                )
                failures.append(f"{feature_set}/{model}: {exc}")
                validation_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model,
                        "status": f"failed: {exc}",
                        "n_features": len(model_features),
                    }
                )
    if not any(row["status"] == "ok" for row in validation_rows):
        raise RuntimeError("Every requested Cox model failed: " + "; ".join(failures))

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output / "survival_model_validation_summary.csv", index=False)
    _write_compatibility_outputs(
        output,
        case_table,
        univariate,
        selected,
        fits,
        dynamic_source,
        dynamic_col,
    )
    (output / "run_summary.txt").write_text(
        "\n".join(
            [
                f"Cluster column: {dynamic_col or 'configured_sources'}",
                f"Cases: {len(case_table)}",
                f"Events: {int(case_table['event'].sum())}",
                f"Image features: {len(image_features)}",
                f"Clinical features: {len(clinical_features)}",
                f"Successful model comparisons: {sum(row['status'] == 'ok' for row in validation_rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close("all")
    return {
        "cluster_col": dynamic_col or "configured_sources",
        "n_cases": len(case_table),
        "n_events": int(case_table["event"].sum()),
        "n_image_features": len(image_features),
        "n_clinical_features": len(clinical_features),
        "successful_models": sum(row["status"] == "ok" for row in validation_rows),
    }


def main() -> None:
    import pandas as pd

    import SpatialBiologyToolkit.cox_survival as cxs
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.reporting import (
        bootstrap_stage_reporting,
        category_output_path,
        get_active_reporter,
    )
    from SpatialBiologyToolkit.scripts.config_and_utils import setup_logging

    arguments = _parse_args()
    config_path = Path(arguments.config).expanduser().resolve(strict=False)
    os.environ["SBT_CONFIG"] = str(config_path)
    os.environ.setdefault("SBT_PROJECT_ROOT", str(config_path.parent))
    bootstrap_stage_reporting(os.environ.get("SBT_STAGE") or "cox")
    config = load_config(config_path)
    setup_logging(config.logging.model_dump(mode="python"), "Cox survival")
    source_configs = _resolved_sources(config)

    source_obs = {}
    source_paths = {}
    for source in source_configs:
        path = _project_path(source.adata_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cox feature AnnData does not exist: {path}")
        source_paths[source.name] = path
        source_obs[source.name] = _read_adata_obs(path)
    clinical, clinical_path = _load_clinical(config, source_obs)
    sweeps = _source_sweeps(source_configs, source_obs)
    dynamic_source = next(
        (source.name for source in source_configs if source.population_obs_search),
        None,
    )

    output_root = category_output_path("files") / "cox"
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dynamic_col, overrides in sweeps:
        analysis_dir = (
            output_root
            if len(sweeps) == 1
            else output_root / _safe_name(dynamic_col or "configured_sources")
        )
        analysis_dir.mkdir(parents=True, exist_ok=True)
        sources = _source_feature_objects(source_configs, source_obs, overrides)
        case_table = cxs.build_multi_source_case_table(
            sources,
            clinical,
            clinical_case_col=config.cox.case_col,
            clinical_duration_col=config.cox.duration_col,
            clinical_event_col=config.cox.event_col,
            clinical_roi_col=config.cox.roi_col,
            covariate_cols=config.cox.covariate_cols,
            censored_case_ids=config.cox.censored_case_ids,
            assume_all_events=config.cox.assume_all_events,
            metadata_conflict=config.cox.metadata_conflict,
            min_observations_per_case=config.cox.min_observations_per_case,
            min_rois_per_case=config.cox.min_rois_per_case,
            min_feature_prevalence=config.cox.min_feature_prevalence,
        )
        summaries.append(
            _run_analysis(
                cxs,
                case_table,
                analysis_dir,
                config.cox,
                dynamic_source,
                dynamic_col,
            )
        )

    summary = pd.DataFrame(summaries)
    if len(summaries) > 1:
        summary.to_csv(output_root / "survival_all_cluster_summary.csv", index=False)
        summary.to_csv(output_root / "coxnet_all_cluster_summary.csv", index=False)
    summary_copy = category_output_path("tables") / "cox_analysis_summary.csv"
    summary.to_csv(summary_copy, index=False)

    reporter = get_active_reporter()
    if reporter is not None:
        for name, path in source_paths.items():
            reporter.add_input(f"cox_feature_source_{name}", path, "AnnData obs feature source.")
        if clinical_path.exists():
            reporter.add_input("cox_clinical_metadata", clinical_path, "Case outcomes and clinical covariates.")
        reporter.add_file(
            "file",
            output_root,
            "Cohesive multi-model Cox report tree with stability-compatible tables.",
        )
        reporter.add_file("table", summary_copy, "Cox analysis/model sweep summary.")
        reporter.add_metric("cox_analyses", len(summaries))
        reporter.add_metric("cox_successful_models", int(summary["successful_models"].sum()))


if __name__ == "__main__":
    main()
