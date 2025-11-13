"""
AI-powered interpretation module for IMC data analysis.

This module provides AI-based cluster interpretation functionality using large language models
to automatically annotate Leiden clustering results based on marker expression patterns.
Currently supports OpenAI GPT models for cell type identification in imaging mass cytometry data.

Key Features:
- Automatically detects existing AI labels and skips re-interpretation unless configured
- Saves results by overwriting the original AnnData file by default
- Configurable via visualization.repeat_ai_interpretation setting
- Comprehensive logging of interpretation decisions and progress
"""

# Standard library imports
import os
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Third-party library imports
import numpy as np
import pandas as pd
import anndata as ad

# Import shared utilities and configurations
from .config_and_utils import (
    GeneralConfig,
    VisualizationConfig,
    BasicProcessConfig,
    filter_config_for_dataclass,
    process_config_with_overrides,
    setup_logging,
)


def _percent_positive(x: np.ndarray, thresh: float = None) -> float:
    """
    Calculate the share of cells above a per-marker threshold.
    
    Robust to IMC dynamic ranges by using a percentile-based threshold
    when no explicit threshold is provided.
    
    Parameters
    ----------
    x : np.ndarray
        Array of marker expression values
    thresh : float, optional
        Threshold value. If None, uses 10th percentile of non-zero values
        
    Returns
    -------
    float
        Fraction of cells above threshold (0.0 to 1.0)
    """
    if thresh is None:
        nz = x[x > 0]
        thresh = float(np.percentile(nz, 10)) if nz.size else 0.0
    return float((x > thresh).mean())


def summarize_clusters_imc(
    adata: ad.AnnData,
    leiden_key: str,
    top_n: int = 8,
    roi_key: Optional[str] = "ROI",
) -> Dict[str, dict]:
    """
    Summarize each Leiden cluster for IMC data analysis.
    
    Creates comprehensive statistical summaries of marker expression patterns
    for each cluster, including z-scores, fold changes, and ROI composition.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with expression data and cluster assignments
    leiden_key : str
        Key in adata.obs containing Leiden cluster assignments
    top_n : int, optional
        Number of top markers to include in summary (default: 8)
    roi_key : str, optional
        Key in adata.obs containing ROI information (default: "ROI")
        
    Returns
    -------
    Dict[str, dict]
        Dictionary mapping cluster IDs to their statistical summaries
        
    Raises
    ------
    AssertionError
        If leiden_key is not found in adata.obs
    """
    assert leiden_key in adata.obs.columns, f"{leiden_key} not in adata.obs"
    
    markers = list(map(str, adata.var_names))
    # Build dense DataFrame from X (handle both sparse and dense matrices)
    if hasattr(adata.X, "A"):
        # Sparse matrix
        df = pd.DataFrame(adata.X.A, columns=markers, index=adata.obs_names)
    else:
        # Dense matrix
        df = pd.DataFrame(adata.X, columns=markers, index=adata.obs_names)
    
    clusters = adata.obs[leiden_key].astype(str)
    summaries: Dict[str, dict] = {}

    # Global statistics for normalization
    global_mean = df.mean(axis=0) + 1e-9  # Add small epsilon to avoid division by zero
    global_std = df.std(axis=0) + 1e-9

    for c in sorted(clusters.unique(), key=lambda x: (len(x), x)):
        idx = clusters == c
        dfc = df.loc[idx]
        
        # Calculate cluster-specific statistics
        mean_c = dfc.mean(axis=0)
        z = (mean_c - global_mean) / global_std  # Z-score vs global mean
        rank = z.sort_values(ascending=False)
        pct_pos = dfc.apply(lambda col: _percent_positive(col.values), axis=0)
        fc = (mean_c + 1e-9) / global_mean  # Fold change vs global mean

        top = rank.head(top_n).index.tolist()

        # ROI composition analysis
        roi_mix = None
        if roi_key and roi_key in adata.obs.columns:
            roi_counts = adata.obs.loc[idx, roi_key].value_counts(normalize=True).sort_values(ascending=False)
            roi_mix = roi_counts.head(10).round(3).to_dict()

        # Store comprehensive cluster summary
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
    """
    Build an LLM prompt for IMC cluster interpretation.
    
    Creates a structured prompt containing tissue context, marker panel information,
    and cluster statistics to guide AI-powered cell type identification.
    
    Parameters
    ----------
    tissue : str
        Tissue type being analyzed (e.g., "breast cancer", "lung tissue")
    panel_markers : List[str]
        List of all markers in the IMC panel
    leiden_key : str
        Name of the Leiden clustering resolution being analyzed
    cluster_summaries : Dict[str, dict]
        Statistical summaries from summarize_clusters_imc()
    guidance : str, optional
        Custom guidance text for the LLM. If None, uses default IMC guidance
        
    Returns
    -------
    str
        Formatted prompt ready for LLM submission
    """
    if guidance is None:
        guidance = (
            "Interpret IMC Leiden clusters using protein markers. "
            "For each cluster, infer likely cell identity, list the key discriminative markers, "
            "call out ambiguities (e.g., macrophage vs dendritic), and provide confidence 0–1. "
            "If uncertain, provide multiple hypotheses."
        )
    
    # Create compact version of summaries for prompt efficiency
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
        f"Summaries (JSON):\n" + json.dumps(compact, separators=(',', ':')) + "\n\n"
        "Return a JSON array of objects with fields: cluster, label, confidence, rationale, alt_labels."
    )
    
    return prompt


def annotate_leiden_imc(
    adata: ad.AnnData,
    tissue: str,
    panel_markers: Optional[List[str]] = None,
    resolutions: Optional[List[float]] = None,
    roi_key: Optional[str] = "ROI",
    output_dir: Path = Path("AI_Interpretation"),
    llm_call: Optional[Callable[[str], str]] = None,
    top_n: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    Run LLM-based interpretation for each Leiden resolution.
    
    Automatically generates cell type annotations for Leiden clustering results
    using AI interpretation of marker expression patterns. Results are attached
    to adata.obs as *_AIlabel columns and saved as TSV files.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with Leiden clustering results
    tissue : str
        Tissue type for contextualized interpretation
    panel_markers : List[str], optional
        List of markers to include. If None, uses all adata.var_names
    resolutions : List[float], optional
        Leiden resolutions to interpret. If None, auto-detects from adata.obs
    roi_key : str, optional
        Key for ROI information in adata.obs (default: "ROI")
    output_dir : Path, optional
        Directory to save interpretation results (default: "AI_Interpretation")
    llm_call : Callable[[str], str], optional
        Function to call LLM with prompt and return response
    top_n : int, optional
        Number of top markers to include in analysis (default: 8)
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping leiden keys to interpretation DataFrames
        
    Raises
    ------
    ValueError
        If llm_call is not provided
    """
    # Set defaults
    if panel_markers is None:
        panel_markers = list(map(str, adata.var_names))
    
    if resolutions is None:
        # Auto-detect Leiden resolutions from adata.obs
        resolutions = [
            float(k.split("leiden_")[1]) 
            for k in adata.obs.columns 
            if k.startswith("leiden_")
        ]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if llm_call is None:
        raise ValueError("Please provide llm_call(prompt:str)->str to call your LLM provider.")
    
    results: Dict[str, pd.DataFrame] = {}

    for r in sorted(resolutions):
        key = f"leiden_{r}"
        if key not in adata.obs.columns:
            logging.warning(f"Leiden resolution {r} not found in adata.obs, skipping.")
            continue
        
        logging.info(f"Processing AI interpretation for {key}")
        
        # Generate cluster summaries
        summaries = summarize_clusters_imc(adata, key, top_n=top_n, roi_key=roi_key)
        
        # Build LLM prompt
        prompt = build_prompt_imc(
            tissue=tissue,
            panel_markers=panel_markers,
            leiden_key=key,
            cluster_summaries=summaries,
        )
        
        # Call LLM
        try:
            raw = llm_call(prompt)
        except Exception as e:
            logging.error(f"LLM call failed for {key}: {e}")
            continue

        # Parse JSON response robustly
        try:
            parsed = json.loads(raw)
        except Exception as e:
            logging.warning(f"Failed to parse full JSON response for {key}: {e}")
            # Try to extract JSON array from response
            start, end = raw.find('['), raw.rfind(']') + 1
            if start != -1 and end > start:
                try:
                    parsed = json.loads(raw[start:end])
                except Exception:
                    logging.error(f"Failed to parse extracted JSON for {key}")
                    parsed = []
            else:
                parsed = []

        # Convert to DataFrame
        rows = []
        for item in (parsed if isinstance(parsed, list) else []):
            rows.append({
                "cluster": str(item.get("cluster", "")),
                "label": item.get("label", "Unknown"),
                "confidence": item.get("confidence", 0.0),
                "rationale": item.get("rationale", ""),
                "alt_labels": ", ".join(item.get("alt_labels", []) or []),
            })
        
        df = pd.DataFrame(rows).sort_values("cluster", kind="stable")
        
        # Save outputs
        (output_dir / f"{key}_prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / f"{key}_raw.json").write_text(raw, encoding="utf-8")
        df.to_csv(output_dir / f"{key}_interpretation.tsv", sep="\t", index=False)

        # Add AI labels to adata.obs
        if not df.empty:
            mapping = dict(zip(df["cluster"].astype(str), df["label"].astype(str)))
            adata.obs[f"{key}_AIlabel"] = (
                adata.obs[key].astype(str).map(mapping).astype("category")
            )
            logging.info(f"Added {key}_AIlabel to adata.obs with {len(mapping)} labels")

        results[key] = df

    return results


def openai_adapter_chat(prompt: str) -> str:
    """
    Lightweight adapter for OpenAI Chat Completions API.
    
    Requires OPENAI_API_KEY environment variable to be set.
    Uses GPT-4o-mini model with conservative temperature for consistent results.
    
    Parameters
    ----------
    prompt : str
        The prompt to send to the OpenAI API
        
    Returns
    -------
    str
        The response from the OpenAI API
        
    Raises
    ------
    RuntimeError
        If the OpenAI API call fails or OPENAI_API_KEY is not set
    """
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,  # Low temperature for consistency
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert in immunology and histopathology for IMC data."
                },
                {
                    "role": "user", 
                    "content": prompt
                },
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")


def should_run_ai_interpretation(adata: ad.AnnData, viz_config: VisualizationConfig) -> bool:
    """
    Check if AI interpretation should be run based on configuration and existing labels.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to check for existing AI labels
    viz_config : VisualizationConfig
        Visualization configuration containing AI settings
        
    Returns
    -------
    bool
        True if AI interpretation should be run, False otherwise
    """
    if not viz_config.enable_ai:
        return False
    
    existing_ai_columns = [col for col in adata.obs.columns if col.endswith('_AIlabel')]
    return not existing_ai_columns or viz_config.repeat_ai_interpretation


def run_ai_interpretation(
    adata: ad.AnnData,
    config: dict,
    output_base_dir: Path = Path("."),
    save_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Main function to run AI interpretation using configuration settings.
    
    This function integrates with the configuration system to automatically
    run AI interpretation when enabled in the visualization config.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Processed AnnData object with Leiden clustering results
    config : dict
        Full configuration dictionary from config.yaml
    output_base_dir : Path, optional
        Base directory for outputs (default: current directory)
    save_path : str, optional
        Path to save the updated AnnData. If None, saves over input file
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping leiden keys to interpretation DataFrames
        
    Raises
    ------
    RuntimeError
        If AI is enabled but OpenAI API key is not available
    """
    # Parse configuration
    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get('general', {}), GeneralConfig)
    )
    viz_config = VisualizationConfig(
        **filter_config_for_dataclass(config.get('visualization', {}), VisualizationConfig)
    )
    
    # Check if AI interpretation should be run
    existing_ai_columns = [col for col in adata.obs.columns if col.endswith('_AIlabel')]
    
    if not viz_config.enable_ai:
        logging.info("AI interpretation disabled in configuration (visualization.enable_ai=False)")
        return {}
    
    if existing_ai_columns and not viz_config.repeat_ai_interpretation:
        logging.info(f"AI interpretation labels already exist: {existing_ai_columns}")
        logging.info("Skipping AI interpretation (set visualization.repeat_ai_interpretation=True to re-run)")
        return {}
    elif existing_ai_columns and viz_config.repeat_ai_interpretation:
        logging.info(f"AI interpretation labels already exist: {existing_ai_columns}")
        logging.info("Re-running AI interpretation because visualization.repeat_ai_interpretation=True")
    else:
        logging.info("No existing AI labels found, proceeding with AI interpretation")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "AI interpretation enabled but OPENAI_API_KEY environment variable not set"
        )
    
    # Set up parameters
    tissue_label = getattr(viz_config, "tissue", "Unknown tissue")
    panel_markers = adata.var_names.tolist()
    
    # Auto-detect Leiden resolutions
    resolutions = [
        float(k.split("leiden_")[1]) 
        for k in adata.obs.columns 
        if k.startswith("leiden_")
    ]
    
    if not resolutions:
        logging.warning("No Leiden clustering results found in adata.obs")
        return {}
    
    # Set up output directory
    qc_folder = getattr(general_config, 'qc_folder', 'QC')
    ai_dir = output_base_dir / qc_folder / "AI_Interpretation"
    
    logging.info(f"Starting AI interpretation for {len(resolutions)} Leiden resolutions")
    logging.info(f"Tissue: {tissue_label}")
    logging.info(f"Panel: {len(panel_markers)} markers")
    
    # Run interpretation
    results = annotate_leiden_imc(
        adata=adata,
        tissue=tissue_label,
        panel_markers=panel_markers,
        resolutions=resolutions,
        roi_key="ROI" if "ROI" in adata.obs.columns else None,
        output_dir=ai_dir,
        llm_call=openai_adapter_chat,
        top_n=8
    )
    
    logging.info(
        f"AI interpretation complete. "
        f"Labels saved under *_AIlabel columns; "
        f"TSV files in {ai_dir}"
    )
    
    # Save the updated AnnData if save_path is provided
    if save_path and results:
        adata.write_h5ad(save_path)
        logging.info(f"Updated AnnData with AI labels saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    """
    Standalone execution for AI interpretation of existing processed data.
    
    This allows running AI interpretation separately from the main processing pipeline,
    useful for re-interpreting data with different parameters or tissue contexts.
    """
    # Set up logging
    pipeline_stage = 'AI_Interpretation'
    config = process_config_with_overrides()
    setup_logging(config.get('logging', {}), pipeline_stage)
    
    # Get configuration
    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get('general', {}), GeneralConfig)
    )
    viz_config = VisualizationConfig(
        **filter_config_for_dataclass(config.get('visualization', {}), VisualizationConfig)
    )
    
    # Load processed AnnData (should have Leiden clustering results)
    # Try to get input path from config, fallback to default
    try:
        process_config = BasicProcessConfig(
            **filter_config_for_dataclass(config.get('process', {}), BasicProcessConfig)
        )
        input_path = process_config.output_adata_path
    except:
        input_path = "anndata_processed.h5ad"  # Fallback default
    
    if not Path(input_path).exists():
        logging.error(f"Processed AnnData file not found at {input_path}")
        logging.error("Please run basic processing first to generate Leiden clustering results")
        exit(1)
    
    logging.info(f'Loading processed AnnData from {input_path}')
    adata = ad.read_h5ad(input_path)
    logging.info('AnnData loaded successfully.')
    
    # Run AI interpretation
    try:
        results = run_ai_interpretation(adata, config, Path("."), save_path=input_path)
        
        if results:
            logging.info(f"Successfully interpreted {len(results)} Leiden resolutions")
            # Save updated adata with AI labels (over original file by default)
            adata.write_h5ad(input_path)
            logging.info(f"Saved AnnData with AI labels to {input_path}")
        else:
            logging.info("No AI interpretation results generated - no file updates needed")
            
    except Exception as e:
        logging.error(f"AI interpretation failed: {e}")
        exit(1)