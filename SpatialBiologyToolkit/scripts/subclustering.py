"""
Subclustering pipeline stage with explicit checkpoints.

Checkpoint 1:
- Create template files in `subclustering/`:
  - `sublustering_settings.csv`
  - `marker_list.csv`

Checkpoint 2:
- If templates already exist, run row-wise subclustering from `sublustering_settings.csv`
  using marker selectors defined in `marker_list.csv`.
- Save figures and `subcluster_to_final_population.csv` under `subclustering/`.
- Persist marker-list membership used for subclustering into `adata.var`.

Checkpoint 3:
- If `subcluster_to_final_population.csv` has been edited (final population differs from
  original subcluster), apply remapping and save:
  - output AnnData
  - `master_index_to_final_population.csv`
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import SpatialBiologyToolkit.utils as sbt_utils
from .config_and_utils import (
    GeneralConfig,
    SubclusteringConfig,
    cleanstring,
    coalesce_config_text,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)


def _checkpoint(title: str) -> None:
    line = "=" * 20
    logging.info("%s %s %s", line, title, line)


def _resolve_input_adata_path(
    general_config: GeneralConfig,
    subclustering_config: SubclusteringConfig,
) -> Path:
    if subclustering_config.input_adata_path:
        return Path(subclustering_config.input_adata_path)
    return Path(general_config.anndata_path)


def _resolve_use_rep(adata: ad.AnnData, requested_key: Optional[str]) -> Optional[str]:
    """Resolve the configured reduced representation with fallback across pipeline variants."""
    if requested_key is None:
        return None
    if requested_key in adata.obsm:
        return requested_key

    family = ("X_batch_integration", "X_biobatchnet", "X_pca_harmony", "X_pca")
    if requested_key not in family:
        raise KeyError(f"Configured subclustering.use_rep '{requested_key}' was not found in adata.obsm.")

    for fallback in family:
        if fallback in adata.obsm:
            logging.warning(
                "Configured subclustering.use_rep '%s' was missing; falling back to adata.obsm['%s'].",
                requested_key,
                fallback,
            )
            return fallback

    raise KeyError(
        f"Configured subclustering.use_rep '{requested_key}' was not found in adata.obsm, and no "
        "integration fallback representation was available."
    )


def _ordered_categories(series: pd.Series) -> List[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories if pd.notna(x)]
    return [str(x) for x in pd.unique(series.dropna())]


def _to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _resolve_figure_ext(ext: str) -> str:
    if not ext:
        return ".png"
    return ext if ext.startswith(".") else f".{ext}"


def _resolve_subclustering_mode(value: Any) -> Tuple[str, Tuple[int, ...]]:
    if isinstance(value, (int, np.integer)):
        stage = int(value)
        if stage in {1, 2, 3}:
            return str(stage), (stage,)
        raise ValueError("subclustering.mode integer must be one of: 1, 2, 3")

    text = str(value).strip().lower() if value is not None else "all"
    aliases = {
        "all": ("all", (1, 2, 3)),
        "full": ("all", (1, 2, 3)),
        "generate": ("generate", (1, 2)),
        "apply": ("apply", (3,)),
        "1": ("1", (1,)),
        "stage1": ("1", (1,)),
        "2": ("2", (2,)),
        "stage2": ("2", (2,)),
        "3": ("3", (3,)),
        "stage3": ("3", (3,)),
    }
    if text in aliases:
        return aliases[text]

    raise ValueError(
        "Invalid subclustering.mode={!r}. Accepted values are 'all', 'generate', 'apply', 1, 2, or 3.".format(
            value
        )
    )


def _resolve_marker_selector_column(
    marker_selector: Any,
    marker_df: pd.DataFrame,
    default_selector: str,
) -> str:
    selector = str(marker_selector).strip()
    if not selector or selector.lower() in {"nan", "none"}:
        selector = default_selector

    if selector.startswith("markers_"):
        candidate = selector
    else:
        candidate = f"markers_{selector}"

    if candidate not in marker_df.columns:
        available = ", ".join([str(c) for c in marker_df.columns])
        raise KeyError(
            f"Marker selector '{selector}' resolved to '{candidate}', but this column was not found "
            f"in marker list table. Available marker columns: {available}"
        )
    return candidate


def _build_subcluster_column(base_label: str, population: str, resolution: float) -> str:
    res_text = f"{float(resolution):g}"
    return (
        f"{cleanstring(base_label)}"
        f"_res{res_text}"
        f"_subset_{cleanstring(population)}"
    )


def _ensure_templates(
    adata: ad.AnnData,
    subclustering_config: SubclusteringConfig,
    output_dir: Path,
    base_label_key: str,
) -> Tuple[Path, Path, bool]:
    settings_path = output_dir / subclustering_config.settings_filename
    marker_list_path = output_dir / subclustering_config.marker_list_filename

    created_any = False

    if not settings_path.exists():
        if base_label_key not in adata.obs.columns:
            raise KeyError(
                f"Configured base_label_key '{base_label_key}' "
                "is missing in AnnData.obs, so template settings cannot be generated."
            )

        populations = _ordered_categories(adata.obs[base_label_key])
        settings_df = pd.DataFrame(
            {
                "base_label": [base_label_key] * len(populations),
                "population": populations,
                "resolution": [float(subclustering_config.default_resolution)] * len(populations),
                "marker_list": [str(subclustering_config.default_marker_list)] * len(populations),
            }
        )
        settings_df.to_csv(settings_path, index=False)
        created_any = True
        logging.info(
            "Created template settings file with %d rows: %s",
            len(settings_df),
            settings_path,
        )
    else:
        logging.info("Found existing subclustering settings file: %s", settings_path)

    if not marker_list_path.exists():
        marker_df = pd.DataFrame(index=[str(x) for x in adata.var_names])
        marker_df.index.name = "marker"
        marker_df["markers_all"] = True
        marker_df.to_csv(marker_list_path)
        created_any = True
        logging.info(
            "Created template marker list file with %d markers: %s",
            marker_df.shape[0],
            marker_list_path,
        )
    else:
        logging.info("Found existing marker list file: %s", marker_list_path)

    return settings_path, marker_list_path, created_any


def _load_settings_table(settings_path: Path) -> pd.DataFrame:
    settings = pd.read_csv(settings_path)
    required = {"base_label", "population", "resolution", "marker_list"}
    missing = sorted(required.difference(set(settings.columns)))
    if missing:
        raise ValueError(
            f"Subclustering settings file is missing required columns: {missing}. "
            f"Path: {settings_path}"
        )

    settings = settings.dropna(subset=["base_label", "population", "resolution"]).copy()
    if settings.empty:
        raise ValueError(
            f"No usable rows found in {settings_path}. Ensure base_label, population, and resolution are populated."
        )
    return settings


def _load_marker_table(marker_list_path: Path) -> pd.DataFrame:
    marker_df = pd.read_csv(marker_list_path, index_col=0)
    marker_df.index = marker_df.index.astype(str)
    marker_cols = [c for c in marker_df.columns if str(c).startswith("markers_")]
    if not marker_cols:
        raise ValueError(
            f"{marker_list_path} must include at least one marker column beginning with 'markers_'."
        )

    for col in marker_cols:
        marker_df[col] = marker_df[col].map(_to_bool)

    return marker_df[marker_cols]


def _build_marker_list_var_column(marker_col: str) -> str:
    cleaned = cleanstring(marker_col)
    if not cleaned:
        cleaned = "marker_list"
    return f"subclustering_{cleaned}"


def _persist_used_marker_lists_to_var(
    adata: ad.AnnData,
    marker_df: pd.DataFrame,
    used_marker_cols: Sequence[str],
) -> Dict[str, str]:
    used = sorted({str(col) for col in used_marker_cols if str(col) in marker_df.columns})
    if not used:
        logging.warning(
            "No marker selectors were recorded as used; skipping marker-list persistence to adata.var."
        )
        return {}

    marker_df = marker_df.reindex([str(v) for v in adata.var_names]).fillna(False)
    existing_cols = set([str(c) for c in adata.var.columns])
    name_map: Dict[str, str] = {}

    for marker_col in used:
        base_name = _build_marker_list_var_column(marker_col)
        var_col = base_name
        suffix = 2
        while var_col in name_map.values():
            var_col = f"{base_name}_{suffix}"
            suffix += 1

        if var_col in existing_cols:
            logging.info(
                "Overwriting existing adata.var column '%s' with current marker list '%s'.",
                var_col,
                marker_col,
            )
        adata.var[var_col] = marker_df[marker_col].astype(bool).values
        name_map[marker_col] = var_col
        logging.info(
            "Stored marker list '%s' in adata.var['%s'] (%d/%d markers selected).",
            marker_col,
            var_col,
            int(marker_df[marker_col].sum()),
            int(adata.n_vars),
        )

    return name_map


def _ensure_umap_for_plots(
    adata: ad.AnnData,
    use_rep: Optional[str],
    enabled: bool,
) -> None:
    if "X_umap" in adata.obsm:
        return

    if not enabled:
        logging.warning(
            "UMAP coordinates (adata.obsm['X_umap']) are missing and compute_umap_if_missing=False. "
            "UMAP QC plots will be skipped."
        )
        return

    _checkpoint("UMAP fallback")
    if use_rep and use_rep in adata.obsm:
        logging.info("Computing UMAP from representation '%s' because X_umap was missing.", use_rep)
        sc.pp.neighbors(adata, use_rep=use_rep)
    else:
        if use_rep:
            logging.warning(
                "Configured use_rep '%s' not found in adata.obsm. Falling back to default neighbors input.",
                use_rep,
            )
        sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    logging.info("Computed fallback UMAP coordinates.")


def _save_combined_umap(
    adata: ad.AnnData,
    subcluster_col: str,
    title: str,
    out_path: Path,
    point_size: float,
    dpi: int,
) -> None:
    if "X_umap" not in adata.obsm:
        return

    sc.pl.umap(
        adata,
        color=subcluster_col,
        s=float(point_size),
        ncols=1,
        title=title,
        show=False,
    )
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close()


def _save_matrixplot(
    adata: ad.AnnData,
    subcluster_col: str,
    markers: Sequence[str],
    title: str,
    out_path: Path,
    vmax: float,
) -> None:
    valid_markers = [m for m in markers if m in adata.var_names]
    if not valid_markers:
        valid_markers = [str(x) for x in adata.var_names]

    # Dendrogram is intentionally disabled to avoid storing large/fragile objects in adata.uns
    # for columns with special characters.
    ordered = (
        sbt_utils.reorder_vars_by_expression(adata, valid_markers)
        if len(valid_markers) > 1
        else valid_markers
    )
    mp = sc.pl.matrixplot(
        adata,
        var_names=ordered,
        groupby=subcluster_col,
        dendrogram=False,
        vmax=float(vmax),
        title=title,
        return_fig=True,
        show=False,
    )
    mp.add_totals().style(edge_color="black")
    mp.savefig(out_path)
    plt.close("all")


def _merge_existing_remap(
    remap_df: pd.DataFrame,
    remap_path: Path,
) -> pd.DataFrame:
    if not remap_path.exists():
        return remap_df

    try:
        existing = pd.read_csv(remap_path)
    except Exception as exc:
        logging.warning("Could not read existing remap file at %s: %s", remap_path, exc)
        return remap_df

    required = {"subcluster_column", "subcluster", "final_population"}
    if not required.issubset(existing.columns):
        logging.warning(
            "Existing remap file '%s' is missing required columns for merge. Replacing with new template.",
            remap_path,
        )
        return remap_df

    lookup = (
        existing[["subcluster_column", "subcluster", "final_population"]]
        .dropna(subset=["subcluster_column", "subcluster"])
        .set_index(["subcluster_column", "subcluster"])["final_population"]
        .to_dict()
    )
    key_index = list(zip(remap_df["subcluster_column"], remap_df["subcluster"]))
    preserved = 0
    final_values: List[Any] = []
    for key, default_value in zip(key_index, remap_df["final_population"]):
        if key in lookup and pd.notna(lookup[key]):
            final_values.append(lookup[key])
            if str(lookup[key]) != str(default_value):
                preserved += 1
        else:
            final_values.append(default_value)
    remap_df = remap_df.copy()
    remap_df["final_population"] = final_values

    if preserved > 0:
        logging.info(
            "Preserved %d edited final_population entries from existing remap file.",
            preserved,
        )
    return remap_df


def _is_remap_modified(remap_df: pd.DataFrame) -> bool:
    if remap_df.empty:
        return False
    left = remap_df["subcluster"].astype(str).fillna("")
    right = remap_df["final_population"].astype(str).fillna("")
    return bool((left != right).any())


def _load_existing_remap_table(remap_path: Path) -> pd.DataFrame:
    if not remap_path.exists():
        raise FileNotFoundError(
            f"Remap file not found: {remap_path}. Run subclustering mode 'generate' or stage 2 first."
        )

    remap_df = pd.read_csv(remap_path)
    required = {"subcluster_column", "subcluster", "final_population"}
    missing = sorted(required.difference(set(remap_df.columns)))
    if missing:
        raise ValueError(
            f"Remap file is missing required columns {missing}: {remap_path}"
        )
    return remap_df


def _apply_subcluster_remap_adapted(
    adata: ad.AnnData,
    remap_df: pd.DataFrame,
    fallback_base_labels: Sequence[str],
    new_label_key: str,
) -> ad.AnnData:
    required = {"subcluster_column", "subcluster", "final_population"}
    missing = sorted(required.difference(set(remap_df.columns)))
    if missing:
        raise ValueError(
            f"Remap file is missing required columns for application: {missing}"
        )

    lookup = (
        remap_df[["subcluster_column", "subcluster", "final_population"]]
        .set_index(["subcluster_column", "subcluster"])["final_population"]
        .to_dict()
    )
    subcluster_cols = [
        str(col)
        for col in pd.unique(remap_df["subcluster_column"])
        if str(col) in adata.obs.columns
    ]
    if not subcluster_cols:
        raise ValueError(
            "No subcluster columns referenced in remap file were found in adata.obs."
        )

    fallback_cols = [str(col) for col in fallback_base_labels if str(col) in adata.obs.columns]
    if not fallback_cols:
        raise KeyError(
            "None of the fallback base label columns were found in adata.obs: "
            + ", ".join([str(x) for x in fallback_base_labels])
        )

    def assign_final(row: pd.Series) -> Any:
        for col in subcluster_cols:
            value = row[col]
            if pd.notna(value):
                mapped = lookup.get((col, value))
                if mapped is not None and pd.notna(mapped):
                    return mapped
        for col in fallback_cols:
            value = row[col]
            if pd.notna(value):
                return value
        return np.nan

    adata.obs[new_label_key] = adata.obs.apply(assign_final, axis=1)
    adata.obs[new_label_key] = adata.obs[new_label_key].astype("category")
    return adata


def run_subclustering_stage(
    general_config: GeneralConfig,
    subclustering_config: SubclusteringConfig,
    stage_name: str = "Subclustering",
) -> Optional[Path]:
    subclustering_config.base_label_key = coalesce_config_text(
        subclustering_config.base_label_key,
        general_config.population_obs_primary,
        default="population",
    )
    subclustering_config.master_index_obs = coalesce_config_text(
        subclustering_config.master_index_obs,
        general_config.master_index_obs,
        default="Master_Index",
    )
    resolved_base_label_key = subclustering_config.base_label_key
    resolved_master_index_obs = subclustering_config.master_index_obs
    mode_label, selected_stages = _resolve_subclustering_mode(getattr(subclustering_config, "mode", "all"))

    _checkpoint("Load Input")
    input_path = _resolve_input_adata_path(general_config, subclustering_config)
    adata, _, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=stage_name,
        stage_config=subclustering_config,
        override_path=str(input_path),
    )
    if skip_stage:
        logging.info("Skipping subclustering stage based on AnnData stage policy.")
        return input_path
    if adata is None:
        raise FileNotFoundError(f"AnnData could not be loaded for subclustering stage: {input_path}")
    logging.info("Loaded AnnData: %d cells x %d markers", adata.n_obs, adata.n_vars)

    logging.info(
        "Resolved Subclustering obs keys: base_label_key='%s', master_index_obs='%s'.",
        resolved_base_label_key,
        resolved_master_index_obs,
    )
    logging.info(
        "Resolved subclustering mode '%s' to checkpoints %s.",
        mode_label,
        list(selected_stages),
    )

    output_dir = Path(subclustering_config.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Subclustering output directory: %s", output_dir)

    settings_path = output_dir / subclustering_config.settings_filename
    marker_list_path = output_dir / subclustering_config.marker_list_filename
    remap_path = output_dir / subclustering_config.remap_filename

    if 1 in selected_stages:
        _checkpoint("Checkpoint 1 - Template files")
        settings_path, marker_list_path, created_any = _ensure_templates(
            adata=adata,
            subclustering_config=subclustering_config,
            output_dir=output_dir,
            base_label_key=resolved_base_label_key,
        )
        if created_any:
            logging.info(
                "Template files were created. Edit '%s' and '%s', then rerun this stage.",
                settings_path.name,
                marker_list_path.name,
            )
            return None
        if selected_stages == (1,):
            logging.info("Subclustering mode '%s' completed after checkpoint 1.", mode_label)
            return None

    if 2 in selected_stages and 1 not in selected_stages:
        missing_inputs = [path.name for path in [settings_path, marker_list_path] if not path.exists()]
        if missing_inputs:
            raise FileNotFoundError(
                "Checkpoint 2 requires existing template files: {}. Run subclustering mode 'generate' or stage 1 first.".format(
                    ", ".join(missing_inputs)
                )
            )

    settings_df: Optional[pd.DataFrame] = None
    remap_df: Optional[pd.DataFrame] = None
    existing_pipeline = adata.uns.get("subclustering_pipeline", {})
    if not isinstance(existing_pipeline, dict):
        existing_pipeline = {}
    marker_list_var_columns: Dict[str, str] = {}
    existing_marker_list_var_columns = existing_pipeline.get("marker_list_var_columns", {})
    if isinstance(existing_marker_list_var_columns, dict):
        marker_list_var_columns = {str(k): str(v) for k, v in existing_marker_list_var_columns.items()}
    used_marker_lists_for_uns = [
        str(x) for x in existing_pipeline.get("used_marker_lists", []) if pd.notna(x)
    ]

    if 2 in selected_stages:
        _checkpoint("Checkpoint 2 - Run subclustering")
        settings_df = _load_settings_table(settings_path)
        marker_df = _load_marker_table(marker_list_path)
        logging.info("Loaded %d subclustering row(s) from settings.", settings_df.shape[0])
        resolved_use_rep = _resolve_use_rep(adata, subclustering_config.use_rep)

        fig_ext = _resolve_figure_ext(subclustering_config.figure_extension)
        figures_dir = output_dir / "figures"
        combined_umap_dir = figures_dir / "combined_umap"
        matrixplot_dir = figures_dir / "matrixplot"
        individual_umap_dir = figures_dir / "individual_umap"
        combined_umap_dir.mkdir(parents=True, exist_ok=True)
        matrixplot_dir.mkdir(parents=True, exist_ok=True)
        if subclustering_config.save_individual_umaps:
            individual_umap_dir.mkdir(parents=True, exist_ok=True)

        _ensure_umap_for_plots(
            adata=adata,
            use_rep=resolved_use_rep,
            enabled=bool(subclustering_config.compute_umap_if_missing),
        )

        remap_rows: List[Dict[str, Any]] = []
        seen_subcluster_cols: set[str] = set()
        used_marker_cols: set[str] = set()

        for row_idx, row in settings_df.reset_index(drop=True).iterrows():
            try:
                base_label = str(row["base_label"]).strip()
                population = str(row["population"]).strip()
                resolution = float(row["resolution"])
                marker_selector = row.get("marker_list", subclustering_config.default_marker_list)

                if base_label not in adata.obs.columns:
                    logging.warning(
                        "Row %d skipped: base_label '%s' not found in adata.obs.",
                        row_idx,
                        base_label,
                    )
                    continue

                pop_values = set(adata.obs[base_label].astype(str).dropna().tolist())
                if population not in pop_values:
                    logging.warning(
                        "Row %d skipped: population '%s' not found in adata.obs['%s'].",
                        row_idx,
                        population,
                        base_label,
                    )
                    continue

                marker_col = _resolve_marker_selector_column(
                    marker_selector=marker_selector,
                    marker_df=marker_df,
                    default_selector=subclustering_config.default_marker_list,
                )
                selected_markers = marker_df.index[marker_df[marker_col]].tolist()
                selected_markers = [m for m in selected_markers if m in adata.var_names]
                if not selected_markers:
                    logging.warning(
                        "Row %d (%s/%s) had zero selected markers in '%s'. Falling back to all markers.",
                        row_idx,
                        base_label,
                        population,
                        marker_col,
                    )
                    selected_markers = [str(x) for x in adata.var_names]
                used_marker_cols.add(marker_col)

                subcluster_col = _build_subcluster_column(
                    base_label=base_label,
                    population=population,
                    resolution=resolution,
                )
                if subcluster_col in seen_subcluster_cols or subcluster_col in adata.obs.columns:
                    subcluster_col = f"{subcluster_col}_r{row_idx+1}"
                seen_subcluster_cols.add(subcluster_col)

                logging.info(
                    "Row %d: subclustering base_label='%s', population='%s', resolution=%s, markers='%s' (%d markers).",
                    row_idx,
                    base_label,
                    population,
                    f"{resolution:g}",
                    marker_col,
                    len(selected_markers),
                )

                adata, new_pops = sbt_utils.leiden_on_subset(
                    adata=adata,
                    restrict_to=(base_label, [population]),
                    genes=selected_markers,
                    subset_key_name=subcluster_col,
                    base_label_key=base_label,
                    leiden_resolution=resolution,
                    use_rep=resolved_use_rep,
                    return_new_names=True,
                )
                new_pops = [str(x) for x in new_pops]
                logging.info(
                    "Row %d produced %d subcluster(s): %s",
                    row_idx,
                    len(new_pops),
                    ", ".join(new_pops) if new_pops else "none",
                )

                if "X_umap" in adata.obsm:
                    combined_path = combined_umap_dir / f"{cleanstring(subcluster_col)}_umap{fig_ext}"
                    _save_combined_umap(
                        adata=adata,
                        subcluster_col=subcluster_col,
                        title=f"{population} (res={resolution:g})",
                        out_path=combined_path,
                        point_size=float(subclustering_config.umap_dot_size),
                        dpi=int(subclustering_config.figure_dpi),
                    )
                    logging.info("Saved combined UMAP: %s", combined_path)

                    if subclustering_config.save_individual_umaps and new_pops:
                        subcluster_indiv_dir = individual_umap_dir / cleanstring(subcluster_col)
                        subcluster_indiv_dir.mkdir(parents=True, exist_ok=True)
                        sbt_utils.plot_umap_highlight_clusters(
                            adata=adata,
                            subcluster_col=subcluster_col,
                            point_size=float(subclustering_config.umap_dot_size),
                            legend_loc="none",
                            show=False,
                            clusters=new_pops,
                            save_dir=str(subcluster_indiv_dir),
                            save_dpi=int(subclustering_config.figure_dpi),
                        )
                        logging.info("Saved individual UMAP highlights in: %s", subcluster_indiv_dir)

                matrixplot_path = matrixplot_dir / f"{cleanstring(subcluster_col)}_matrixplot{fig_ext}"
                _save_matrixplot(
                    adata=adata[adata.obs[subcluster_col].isin(new_pops)].copy(),
                    subcluster_col=subcluster_col,
                    markers=selected_markers,
                    title=f"{population} (res={resolution:g})",
                    out_path=matrixplot_path,
                    vmax=float(subclustering_config.matrixplot_vmax),
                )
                logging.info("Saved matrixplot: %s", matrixplot_path)

                for subcluster in new_pops:
                    remap_rows.append(
                        {
                            "subcluster_column": subcluster_col,
                            "parent_population": population,
                            "resolution": resolution,
                            "subcluster": subcluster,
                            "final_population": subcluster,
                        }
                    )
            except Exception as exc:
                logging.exception("Row %d failed during subclustering and was skipped: %s", row_idx, exc)
                continue

        if not remap_rows:
            raise RuntimeError(
                "Subclustering did not generate any rows. Check subclustering settings and population names."
            )

        remap_df = (
            pd.DataFrame(remap_rows)
            .sort_values(["parent_population", "resolution", "subcluster"])
            .reset_index(drop=True)
        )
        remap_df = _merge_existing_remap(remap_df=remap_df, remap_path=remap_path)
        remap_df.to_csv(remap_path, index=False)
        logging.info("Saved remap table: %s", remap_path)

        marker_list_var_columns = _persist_used_marker_lists_to_var(
            adata=adata,
            marker_df=marker_df,
            used_marker_cols=sorted(used_marker_cols),
        )
        used_marker_lists_for_uns = sorted([str(x) for x in used_marker_cols])

    if 3 in selected_stages:
        if settings_df is None:
            if not settings_path.exists():
                raise FileNotFoundError(
                    f"Settings file not found: {settings_path}. Run subclustering mode 'generate' or stage 1 first."
                )
            settings_df = _load_settings_table(settings_path)
        if remap_df is None:
            remap_df = _load_existing_remap_table(remap_path)

        _checkpoint("Checkpoint 3 - Apply edited remap")
        remap_modified = _is_remap_modified(remap_df)
        if subclustering_config.apply_remap_only_if_modified and not remap_modified:
            logging.info(
                "Remap file has no edits (final_population matches subcluster for all rows). "
                "Skipping remap application; edit '%s' then rerun to integrate final labels.",
                remap_path.name,
            )
        else:
            fallback_base_labels = [resolved_base_label_key]
            fallback_base_labels.extend(
                [str(x) for x in pd.unique(settings_df["base_label"].dropna()) if str(x) not in fallback_base_labels]
            )
            adata = _apply_subcluster_remap_adapted(
                adata=adata,
                remap_df=remap_df,
                fallback_base_labels=fallback_base_labels,
                new_label_key=subclustering_config.final_label_key,
            )
            logging.info(
                "Applied subcluster remap and populated obs['%s'].",
                subclustering_config.final_label_key,
            )

            master_col = resolved_master_index_obs
            if master_col in adata.obs.columns:
                master_values = adata.obs[master_col].tolist()
            else:
                logging.warning(
                    "Master index column '%s' not found in adata.obs. Using obs_names instead.",
                    master_col,
                )
                master_values = adata.obs_names.astype(str).tolist()

            mapping_path = output_dir / subclustering_config.master_index_mapping_filename
            mapping_df = pd.DataFrame(
                {
                    master_col: master_values,
                    subclustering_config.final_label_key: adata.obs[subclustering_config.final_label_key].astype(str).tolist(),
                }
            )
            mapping_df.to_csv(mapping_path, index=False)
            logging.info("Saved %s to final population mapping: %s", master_col, mapping_path)

    if settings_df is None:
        settings_df = _load_settings_table(settings_path)
    if remap_df is None:
        remap_df = _load_existing_remap_table(remap_path)

    adata.uns["subclustering_pipeline"] = {
        "input_adata_path": str(input_path),
        "settings_path": str(settings_path),
        "marker_list_path": str(marker_list_path),
        "remap_path": str(remap_path),
        "mode": mode_label,
        "executed_stages": list(selected_stages),
        "base_label_key": resolved_base_label_key,
        "final_label_key": subclustering_config.final_label_key,
        "n_settings_rows": int(settings_df.shape[0]),
        "n_remap_rows": int(remap_df.shape[0]),
        "used_marker_lists": used_marker_lists_for_uns,
        "marker_list_var_columns": marker_list_var_columns,
    }

    output_override = subclustering_config.output_adata_path or str(input_path)
    output_path = save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=subclustering_config,
        override_path=output_override,
        extra_details={"subclustering_output_dir": str(output_dir)},
    )
    logging.info("Saved subclustering AnnData output to %s", output_path)
    return output_path


if __name__ == "__main__":
    pipeline_stage = "Subclustering"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    subclustering_config = SubclusteringConfig(
        **filter_config_for_dataclass(config.get("subclustering", {}), SubclusteringConfig)
    )

    output = run_subclustering_stage(
        general_config=general_config,
        subclustering_config=subclustering_config,
        stage_name=pipeline_stage,
    )
    if output is None:
        logging.info("Subclustering stage finished without saving an AnnData output.")
