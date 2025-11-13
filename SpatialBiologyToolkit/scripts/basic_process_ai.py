"""
Basic preprocessing and AI interpretation (LEGACY - Combined Pipeline).

This module combines core processing with AI interpretation.
For modular workflows, use:
1. basic_process.py - Core processing (batch correction, UMAP, Leiden)
2. ai_interpretation.py - AI-powered cluster labeling
3. basic_visualizations.py - Comprehensive visualizations

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

# Import shared utilities and configurations
from .config_and_utils import *

# Import AI interpretation functionality from dedicated module
from .ai_interpretation import (
    annotate_leiden_imc,
    openai_adapter_chat,
    should_run_ai_interpretation
)

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
    pipeline_stage = 'Process_AI'
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

    if lr_list:
        if not isinstance(lr_list, list):
            lr_list = [lr_list]

        for r in lr_list:
            logging.info(f'Starting Leiden clustering for resolution {r}.')
            sc.tl.leiden(adata, resolution=r, key_added=f'leiden_{r}')
            logging.info(f'Finished Leiden clustering for resolution {r}.')

    # Core processing complete - generate processed AnnData
    logging.info("Core processing complete.")

    # ---------------- AI Interpretation (IMC) ----------------
    import os
    
    if should_run_ai_interpretation(adata, viz_config):
        if not os.getenv("OPENAI_API_KEY"):
            logging.warning("AI mode requested but OPENAI_API_KEY is not set. Skipping AI interpretation.")
        else:
            try:
                logging.info("Starting AI interpretation of Leiden clusters (IMC).")
                tissue_label = getattr(viz_config, "tissue", "Unknown tissue")
                panel_markers = adata.var_names.tolist()
                
                # Set up output directory
                qc_folder = getattr(general_config, 'qc_folder', 'QC')
                ai_dir = Path(qc_folder) / "AI_Interpretation"
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
            except Exception as e:
                logging.warning(f"AI interpretation step failed: {e}")
    else:
        logging.info("AI interpretation skipped (disabled or already completed).")
    # ---------------------------------------------------------
    
    logging.info("Combined processing and AI interpretation pipeline complete.")

    # Save the processed AnnData object
    adata.write_h5ad(process_config.output_adata_path)
    logging.info(f'Saved processed AnnData to {process_config.output_adata_path}.')