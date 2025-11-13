"""
Basic preprocessing and visualization.

Adds basic visualizations of initial Leiden clustering:
- UMAPs colored by Leiden clusters
- MatrixPlots grouped by Leiden clusters
- Tissue overlays of populations per ROI using segmentation masks
"""

# Standard library imports
import logging
from pathlib import Path

# Third-party library imports
import scanpy as sc
import anndata as ad
import scanpy.external as sce  # Needed for Harmony and BBKNN

# Import shared utilities and configurations
from .config_and_utils import *

# Try to import plotting utilities for tissue visualization
try:
    # Preferred absolute import if package is available
    from SpatialBiologyToolkit import plotting as sbt_plotting
except Exception:
    try:
        # Fallback to relative import if run as module inside package
        from .. import plotting as sbt_plotting  # type: ignore
    except Exception:
        sbt_plotting = None  # Will guard usage at runtime

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

    # Set up QC output folders for figures
    qc_base = Path(general_config.qc_folder) / 'BasicProcess_QC'
    qc_umap_dir = qc_base / 'UMAPs'
    qc_matrix_dir = qc_base / 'Matrixplots'
    qc_pop_dir = qc_base / 'Population_images'
    for p in [qc_umap_dir, qc_matrix_dir, qc_pop_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Create UMAP plots colored by Leiden clusters
    try:
        for r in lr_list or []:
            key = f'leiden_{r}'
            if key in adata.obs.columns:
                logging.info(f'Saving UMAP colored by {key}.')
                try:
                    fig = sc.pl.umap(
                        adata,
                        color=key,
                        size=10,
                        legend_loc='right margin',
                        return_fig=True
                    )
                    fig_path = qc_umap_dir / f'UMAP_{key}.png'
                    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
                except Exception as e:
                    logging.warning(f'Failed to save UMAP for {key}: {e}')
            else:
                logging.warning(f'{key} not found in adata.obs; skipping UMAP.')
    except Exception as e:
        logging.error(f'UMAP visualization step failed: {e}')

    # Create UMAP plots colored by marker expression
    try:
        markers = adata.var_names.tolist()
        logging.info(f'Creating UMAP plots for {len(markers)} markers.')
        for marker in markers:
            if marker in adata.var_names:
                logging.info(f'Saving UMAP colored by marker {marker}.')
                try:
                    fig = sc.pl.umap(
                        adata,
                        color=marker,
                        size=10,
                        use_raw=False,  # Use processed data
                        return_fig=True
                    )
                    fig_path = qc_umap_dir / f'UMAP_marker_{marker}.png'
                    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
                except Exception as e:
                    logging.warning(f'Failed to save UMAP for marker {marker}: {e}')
            else:
                logging.warning(f'Marker {marker} not found in adata.var_names; skipping UMAP.')
    except Exception as e:
        logging.error(f'Marker UMAP visualization step failed: {e}')

    # Create MatrixPlot summaries grouped by Leiden clusters
    try:
        for r in lr_list or []:
            key = f'leiden_{r}'
            if key in adata.obs.columns:
                logging.info(f'Saving MatrixPlot grouped by {key}.')
                try:
                    fig = sc.pl.matrixplot(
                        adata,
                        var_names=adata.var_names.tolist(),
                        groupby=key,
                        standard_scale='var',
                        dendrogram=True,
                        show=False,
                        return_fig=True
                    )
                    fig_path = qc_matrix_dir / f'Matrixplot_{key}.png'
                    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
                    logging.info(f'MatrixPlot saved to {fig_path}')
                except Exception as e:
                    logging.warning(f'Failed to save MatrixPlot for {key}: {e}')
            else:
                logging.warning(f'{key} not found in adata.obs; skipping MatrixPlot.')
    except Exception as e:
        logging.error(f'MatrixPlot visualization step failed: {e}')

    # Visualize populations in tissue by mapping clusters back to masks
    try:
        if sbt_plotting is None:
            logging.warning('plotting module unavailable; skipping tissue visualization.')
        elif 'ROI' not in adata.obs.columns:
            logging.warning('ROI column not found in adata.obs; skipping tissue visualization.')
        else:
            rois = sorted(adata.obs['ROI'].astype(str).unique().tolist())
            if not rois:
                logging.warning('No ROIs found in adata.obs; skipping tissue visualization.')
            for r in lr_list or []:
                key = f'leiden_{r}'
                if key not in adata.obs.columns:
                    continue
                out_dir = qc_pop_dir / f'{key}'
                out_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f'Creating tissue overlays for {key} across {len(rois)} ROIs.')
                for roi in rois:
                    try:
                        save_path = out_dir / f'{roi}.png'
                        sbt_plotting.obs_to_mask(
                            adata=adata,
                            roi=roi,
                            roi_obs='ROI',
                            cat_obs=key,
                            masks_folder=general_config.masks_folder,
                            save_path=str(save_path),
                            background_color='white',
                            separator_color='black'
                        )
                    except Exception as e:
                        logging.warning(f'Failed tissue overlay for ROI {roi} ({key}): {e}')
    except Exception as e:
        logging.error(f'Tissue visualization step failed: {e}')

    # Save the processed AnnData object
    adata.write_h5ad(process_config.output_adata_path)
    logging.info(f'Saved processed AnnData to {process_config.output_adata_path}.')
