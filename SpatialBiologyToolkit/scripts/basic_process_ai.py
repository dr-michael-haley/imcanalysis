"""
Basic preprocessing and AI interpretation.

Core processing pipeline including:
- Batch correction and neighbors computation
- UMAP computation
- Leiden clustering at multiple resolutions
- AI interpretation of clusters (optional)
"""

# Standard library imports
import logging
from pathlib import Path

# Third-party library imports
import scanpy as sc
import anndata as ad
import scanpy.external as sce  # Needed for Harmony and BBKNN
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot/scanpy that plots

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- AI interpretation: extra imports for IMC marker-based labeling ---
import os
import json
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional


# -------------------- IMC AI Interpretation Helpers --------------------
def _percent_positive(x: np.ndarray, thresh: float = None) -> float:
    """Share of cells above a per-marker threshold (robust to IMC dynamic ranges)."""
    if thresh is None:
        nz = x[x > 0]
        thresh = float(np.percentile(nz, 10)) if nz.size else 0.0
    return float((x > thresh).mean())

def summarize_clusters_imc(
    adata,
    leiden_key: str,
    top_n: int = 8,
    roi_key: Optional[str] = "ROI",
) -> Dict[str, dict]:
    """Summarize each Leiden cluster for IMC (marker-level, not genes)."""
    assert leiden_key in adata.obs.columns, f"{leiden_key} not in adata.obs"
    markers = list(map(str, adata.var_names))
    # Build dense DataFrame from X
    df = pd.DataFrame(adata.X.A if hasattr(adata.X, "A") else adata.X, columns=markers, index=adata.obs_names)
    clusters = adata.obs[leiden_key].astype(str)
    summaries: Dict[str, dict] = {}

    global_mean = df.mean(axis=0) + 1e-9
    global_std = df.std(axis=0) + 1e-9

    for c in sorted(clusters.unique(), key=lambda x: (len(x), x)):
        idx = clusters == c
        dfc = df.loc[idx]
        mean_c = dfc.mean(axis=0)
        z = (mean_c - global_mean) / global_std
        rank = z.sort_values(ascending=False)
        pct_pos = dfc.apply(lambda col: _percent_positive(col.values), axis=0)
        fc = (mean_c + 1e-9) / global_mean

        top = rank.head(top_n).index.tolist()

        roi_mix = None
        if roi_key and roi_key in adata.obs.columns:
            roi_counts = adata.obs.loc[idx, roi_key].value_counts(normalize=True).sort_values(ascending=False)
            roi_mix = roi_counts.head(10).round(3).to_dict()

        summaries[c] = {
            "n_cells": int(idx.sum()),
            "top_markers_by_z": top,
            "markers": {
                m: {
                    "mean": float(mean_c[m]),
                    "pct_positive": float(pct_pos[m]),
                    "fold_change_vs_global": float(fc[m]),
                    "zscore_mean": float(z[m]),
                } for m in markers
            },
            "roi_composition": roi_mix,
        }
    return summaries

def build_prompt_imc(
    tissue: str,
    panel_markers: List[str],
    leiden_key: str,
    cluster_summaries: Dict[str, dict],
    guidance: Optional[str] = None,
) -> str:
    guidance = guidance or (
        "Interpret IMC Leiden clusters using protein markers. "
        "For each cluster, infer likely cell identity, list the key discriminative markers, "
        "call out ambiguities (e.g., macrophage vs dendritic), and provide confidence 0–1. "
        "If uncertain, provide multiple hypotheses."
    )
    compact = {}
    for c, s in cluster_summaries.items():
        keep = set(s["top_markers_by_z"])
        slim = {m: s["markers"][m] for m in keep if m in s["markers"]}
        compact[c] = {
            "n_cells": s["n_cells"],
            "roi_composition": s["roi_composition"],
            "top_markers_by_z": s["top_markers_by_z"],
            "marker_stats_top": slim,
        }
    prompt = (
        f"You are assisting with imaging mass cytometry (IMC) analysis.\n"
        f"Tissue: {tissue}\n"
        f"Panel markers (~{len(panel_markers)}): {', '.join(sorted(map(str, panel_markers)))}\n\n"
        f"Task: {guidance}\n\n"
        f"Clustering key: {leiden_key}\n"
        f"Summaries (JSON):\n" + json.dumps(compact, separators=(',',':')) + "\n\n"
        "Return a JSON array of objects with fields: cluster, label, confidence, rationale, alt_labels."
    )
    return prompt

def annotate_leiden_imc(
    adata,
    tissue: str,
    panel_markers: Optional[List[str]] = None,
    resolutions: Optional[List[float]] = None,
    roi_key: Optional[str] = "ROI",
    output_dir: Path = Path("AI_Interpretation"),
    llm_call: Optional[Callable[[str], str]] = None,
    top_n: int = 8,
) -> Dict[str, pd.DataFrame]:
    """Run LLM-based interpretation for each Leiden resolution; attach *_AIlabel to .obs and save TSVs."""
    if panel_markers is None:
        panel_markers = list(map(str, adata.var_names))
    if resolutions is None:
        resolutions = [float(k.split("leiden_")[1]) for k in adata.obs.columns if k.startswith("leiden_")]
    output_dir.mkdir(parents=True, exist_ok=True)
    if llm_call is None:
        raise ValueError("Please provide llm_call(prompt:str)->str to call your LLM provider.")
    results: Dict[str, pd.DataFrame] = {}

    for r in sorted(resolutions):
        key = f"leiden_{r}"
        if key not in adata.obs.columns:
            continue
        summaries = summarize_clusters_imc(adata, key, top_n=top_n, roi_key=roi_key)
        prompt = build_prompt_imc(
            tissue=tissue,
            panel_markers=panel_markers,
            leiden_key=key,
            cluster_summaries=summaries,
        )
        raw = llm_call(prompt)

        # Parse JSON robustly
        try:
            parsed = json.loads(raw)
        except Exception:
            start, end = raw.find('['), raw.rfind(']') + 1
            parsed = json.loads(raw[start:end]) if (start != -1 and end > start) else []

        rows = []
        for item in (parsed if isinstance(parsed, list) else []):
            rows.append({
                "cluster": str(item.get("cluster")),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "rationale": item.get("rationale"),
                "alt_labels": ", ".join(item.get("alt_labels", []) or []),
            })
        df = pd.DataFrame(rows).sort_values("cluster", kind="stable")
        (output_dir / f"{key}_prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / f"{key}_raw.json").write_text(raw, encoding="utf-8")
        df.to_csv(output_dir / f"{key}_interpretation.tsv", sep="\t", index=False)

        if not df.empty:
            mapping = dict(zip(df["cluster"].astype(str), df["label"].astype(str)))
            adata.obs[f"{key}_AIlabel"] = adata.obs[key].astype(str).map(mapping).astype("category")

        results[key] = df
    return results

def openai_adapter_chat(prompt: str) -> str:
    """Lightweight adapter for OpenAI Chat Completions (set OPENAI_API_KEY)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are an expert in immunology and histopathology for IMC data."},
                {"role": "user", "content": prompt},
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")
# -----------------------------------------------------------------------


# Import shared utilities and configurations
from .config_and_utils import *

# Core processing only - visualization moved to separate module

# Color legend functionality moved to basic_visualizations.py

def batch_neighbors(
        adata,
        correction_method = None, #: str = 'bbknn',
        batch_correction_obs = None, #: str = 'Case',
        n_for_pca: int = None
) -> None:
    """
    Perform batch correction and preprocessing on an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    correction_method : str, optional
        Method for batch correction. Options are 'bbknn', 'harmony', 'both', or None.
        Default is 'bbknn'.
    batch_correction_obs : str, optional
        Observation key for batch correction.
        Default is 'Case'.
    n_for_pca : int, optional
        Number of principal components to use. If None, it defaults to one less than the number of markers.
        Default is None.

    Returns
    -------
    None

    Notes
    -----
    - If `n_for_pca` is not specified, it is set to one less than the number of markers in `adata.var_names`.
    - The function performs PCA followed by the specified batch correction method.
    """
    if n_for_pca is None:
        # Define the number of PCA dimensions to work with - one less than number of markers.
        n_for_pca = len(adata.var_names) - 1

    logging.info(f'Calculating PCA with {n_for_pca} dimensions.')
    sc.tl.pca(adata, n_comps=n_for_pca)

    # Ensure the batch correction observation is categorical
    if batch_correction_obs:
        adata.obs[batch_correction_obs] = adata.obs[batch_correction_obs].astype('category')

    # Apply the specified batch correction method
    if correction_method == 'bbknn':
        logging.info('Starting BBKNN calculations.')
        sc.external.pp.bbknn(adata, batch_key=batch_correction_obs, n_pcs=n_for_pca)
        logging.info(f'Finished BBKNN batch correction with obs: {batch_correction_obs}.')

    elif correction_method == 'harmony':
        logging.info('Starting Harmony calculations.')
        sce.pp.harmony_integrate(adata, key=batch_correction_obs, basis='X_pca', adjusted_basis='X_pca')
        logging.info(f'Finished Harmony batch correction with obs: {batch_correction_obs}.')
        logging.info('Calculating neighbors using adjusted PCA.')
        sc.pp.neighbors(adata, use_rep='X_pca')
        logging.info('Finished calculating neighbors.')

    elif correction_method == 'both':
        logging.info('Starting Harmony calculations.')
        sce.pp.harmony_integrate(adata, key=batch_correction_obs, basis='X_pca', adjusted_basis='X_pca')
        logging.info(f'Finished Harmony batch correction with obs: {batch_correction_obs}.')
        logging.info('Starting BBKNN calculations.')
        sc.external.pp.bbknn(adata, batch_key=batch_correction_obs, n_pcs=n_for_pca)
        logging.info(f'Finished BBKNN batch correction with obs: {batch_correction_obs}.')

    else:
        logging.info('No batch correction performed. Calculating neighbors using PCA.')
        sc.pp.neighbors(adata, use_rep='X_pca')
        logging.info('Finished calculating neighbors.')

    logging.info('Finished PCA and batch correction.')

if __name__ == "__main__":
    # Set up logging
    pipeline_stage = 'Process'
    config = process_config_with_overrides()
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**filter_config_for_dataclass(config.get('general', {}), GeneralConfig))
    process_config = BasicProcessConfig(**filter_config_for_dataclass(config.get('process', {}), BasicProcessConfig))
    viz_config = VisualizationConfig(**filter_config_for_dataclass(config.get('visualization', {}), VisualizationConfig))

    # Load saved AnnData
    logging.info(f'Loading AnnData from {process_config.input_adata_path}.')
    adata = ad.read_h5ad(process_config.input_adata_path)
    logging.info('AnnData loaded successfully.')

    # Batch correction and neighbors computation
    batch_neighbors(
        adata=adata,
        correction_method=process_config.batch_correction_method,
        batch_correction_obs=process_config.batch_correction_obs,
        n_for_pca=process_config.n_for_pca
    )

    # UMAP computation
    logging.info('Starting UMAP calculations.')
    sc.tl.umap(adata, min_dist=process_config.umap_min_dist)
    logging.info('Finished UMAP calculations.')

    # Leiden clustering
    lr_list = process_config.leiden_resolutions_list

    # AI mode flag from config
    ai_mode = bool(getattr(viz_config, "enable_ai", False))
    if ai_mode:
        logging.info("AI mode enabled via config (visualization.enable_ai=True).")
    else:
        logging.info("AI mode disabled (set visualization.enable_ai=True to enable).")

    if lr_list:
        if not isinstance(lr_list, list):
            lr_list = [lr_list]

        for r in lr_list:
            logging.info(f'Starting Leiden clustering for resolution {r}.')
            sc.tl.leiden(adata, resolution=r, key_added=f'leiden_{r}')
            logging.info(f'Finished Leiden clustering for resolution {r}.')

    # Core processing complete - generate processed AnnData

    # ---------------- AI Interpretation (IMC) ----------------
    try:
        if ai_mode:
            if os.getenv("OPENAI_API_KEY"):
                logging.info("Starting AI interpretation of Leiden clusters (IMC).")
                tissue_label = getattr(viz_config, "tissue", "Unknown tissue")
                panel_markers = adata.var_names.tolist()
                ai_dir = Path(general_config.output_folder) / "AI_Interpretation"
                ai_dir.mkdir(parents=True, exist_ok=True)
                _ = annotate_leiden_imc(
                    adata=adata,
                    tissue=tissue_label,
                    panel_markers=panel_markers,
                    resolutions=lr_list,
                    roi_key="ROI" if "ROI" in adata.obs.columns else None,
                    output_dir=ai_dir,
                    llm_call=openai_adapter_chat,
                    top_n=8
                )
                logging.info("AI interpretation complete. Labels saved under *_AIlabel; TSVs in AI_Interpretation/.")
            else:
                logging.warning("AI mode requested but OPENAI_API_KEY is not set. Skipping AI interpretation.")
        else:
            logging.info("AI mode disabled; skipping AI interpretation.")
    except Exception as e:
        logging.warning(f"AI interpretation step failed: {e}")
    # ---------------------------------------------------------
    
    logging.info("Core processing pipeline complete. Data ready for visualization.")

    # Save the processed AnnData object
    adata.write_h5ad(process_config.output_adata_path)
    logging.info(f'Saved processed AnnData to {process_config.output_adata_path}.')