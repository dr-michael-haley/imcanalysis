# Standard library imports
import logging

# Third-party library imports
import scanpy as sc
import anndata as ad
import scanpy.external as sce  # Needed for Harmony and BBKNN

# Import shared utilities and configurations
from .config_and_utils import *

def batch_neighbors(
        adata,
        correction_method: str = 'bbknn',
        batch_correction_obs: str = 'Case',
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
    general_config = GeneralConfig(**config.get('general', {}))
    process_config = BasicProcessConfig(**config.get('process', {}))

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

    # Save the processed AnnData object
    adata.write_h5ad(process_config.output_adata_path)
    logging.info(f'Saved processed AnnData to {process_config.output_adata_path}.')
