"""
Basic visualization module for IMC data analysis.

Creates comprehensive visualizations for processed AnnData objects including:
- UMAPs colored by leiden clusters and AI labels
- MatrixPlots grouped by populations  
- Tissue overlays of populations per ROI using segmentation masks
- Population analysis across metadata categories
- Backgating assessment for population validation
- Color legends for categorical data
"""

# Standard library imports
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party library imports
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
# Note: Backend is set to "Agg" only when run as main script (see __main__ section)
# This allows interactive plotting when importing functions from this module

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# Try to import backgating utilities for cell validation
try:
    # Preferred absolute import if package is available
    from SpatialBiologyToolkit import backgating as sbt_backgating
except Exception:
    try:
        # Fallback to relative import if run as module inside package
        from .. import backgating as sbt_backgating  # type: ignore
    except Exception:
        sbt_backgating = None  # Will guard usage at runtime


from SpatialBiologyToolkit import utils as sbt_utils
 

def log_detailed_error(error, context="", logger=None):
    """
    Log detailed error information including traceback and context.
    
    Parameters
    ----------
    error : Exception
        The exception that occurred
    context : str, optional
        Additional context information about when/where the error occurred
    logger : logging.Logger, optional
        Logger to use. If None, uses the root logger
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Get the full traceback
    tb_str = traceback.format_exc()
    
    # Extract line number from traceback if possible
    tb_lines = tb_str.strip().split('\n')
    line_info = "Line info not available"
    for line in tb_lines:
        if 'File "' in line and 'line' in line:
            line_info = line.strip()
            break
    
    # Format the error message with all details
    error_msg = f"Error in {context}:\n"
    error_msg += f"  Exception: {type(error).__name__}: {str(error)}\n"
    error_msg += f"  Location: {line_info}\n"
    error_msg += f"  Full traceback:\n{tb_str}"
    
    logger.error(error_msg)


def find_population_columns(adata, max_categories=50):
    """
    Intelligently find population/clustering columns in adata.obs.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    max_categories : int, optional
        Maximum number of unique categories allowed (default: 50)
        
    Returns
    -------
    list
        List of population column names found in adata.obs
    """
    population_columns = []
    
    # Look for common population column patterns
    for col in adata.obs.columns:
        # Leiden clustering columns (including AI labels)
        if 'leiden' in col.lower():
            population_columns.append(col)
        # Louvain clustering columns
        elif 'louvain' in col.lower():
            population_columns.append(col)
        elif 'population' in col.lower():
            population_columns.append(col)
        elif 'cluster' in col.lower():
            population_columns.append(col)
        # Phenotype or cell type columns
        elif any(term in col.lower() for term in ['phenotype', 'celltype', 'cell_type', 'celltypes', 'cell_types']):
            population_columns.append(col)
        # Manual annotation columns
        elif any(term in col.lower() for term in ['annotation', 'annotations', 'manual', 'annotated']):
            population_columns.append(col)
    
    # Filter out columns with too many or too few unique values
    filtered_columns = []
    for col in population_columns:
        n_unique = adata.obs[col].nunique()
        # Reasonable range: 2 to max_categories unique populations
        if 2 <= n_unique <= max_categories:
            filtered_columns.append(col)
        else:
            logging.info(f"Excluding {col} from population analysis: {n_unique} unique values (outside range 2-{max_categories})")
    
    if filtered_columns:
        logging.info(f"Found population columns: {filtered_columns}")
    else:
        logging.warning("No suitable population columns found in adata.obs")
    
    return filtered_columns


def find_metadata_columns(adata, population_columns=None, metadata_folder='metadata', max_categories=50):
    """
    Intelligently find metadata/categorical columns in adata.obs using dictionary.csv.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    population_columns : list, optional
        List of population columns to exclude from metadata
    metadata_folder : str or Path, optional
        Path to metadata folder containing dictionary.csv
    max_categories : int, optional
        Maximum number of unique categories allowed (default: 50)
        
    Returns
    -------
    list
        List of metadata column names found in adata.obs
    """
    if population_columns is None:
        population_columns = []
    
    # Define columns to always exclude from metadata analysis
    exclude_columns = {
        'X_loc', 'Y_loc', 'Master_Index', 'ObjectNumber', 
        'ROI_name', 'ROI_width', 'ROI_height', 'MCD_file', 'Source_file', 'File_type',
        'mask_area', 'mask_perimeter', 'mask_circularity', 'mask_largest_diameter', 'mask_largest_diameter_angle'
    }
    exclude_columns.update(population_columns)  # Exclude population columns
    
    metadata_columns = []
    
    # First, try to find metadata columns from dictionary.csv
    metadata_folder = Path(metadata_folder)
    dictionary_path = metadata_folder / 'dictionary.csv'
    
    dictionary_columns = []
    if dictionary_path.exists():
        try:
            import pandas as pd
            dictionary_df = pd.read_csv(dictionary_path, index_col='ROI')
            
            # Get columns from dictionary file (excluding description/example columns)
            dictionary_columns = [col for col in dictionary_df.columns 
                                if 'example' not in col.lower() and 'description' not in col.lower()]
            
            logging.info(f"Found {len(dictionary_columns)} potential metadata columns in dictionary.csv: {dictionary_columns}")
            
        except Exception as e:
            logging.warning(f"Could not read dictionary.csv: {e}")
    
    # Check which dictionary columns are actually present in adata.obs and suitable for visualization
    for col in dictionary_columns:
        if col in adata.obs.columns and col not in exclude_columns:
            n_unique = adata.obs[col].nunique()
            if 2 <= n_unique <= max_categories:  # Reasonable range for visualization
                metadata_columns.append(col)
            else:
                logging.info(f"Excluding dictionary column {col}: {n_unique} unique values (outside range 2-{max_categories})")
    
    # Also check for ROI column and other common metadata patterns not in dictionary
    for col in adata.obs.columns:
        if col in exclude_columns or col in metadata_columns:
            continue
            
        n_unique = adata.obs[col].nunique()
        is_metadata = False
        
        # ROI information (commonly used in IMC)
        if col.upper() == 'ROI' or 'roi' in col.lower():
            is_metadata = True
        # Sample/Patient/Batch identifiers (common patterns not in dictionary)
        elif any(term in col.lower() for term in ['sample', 'patient', 'batch', 'replicate', 'condition', 'treatment', 'group']):
            is_metadata = True
        # General categorical data with reasonable categories (fallback)
        elif 2 <= n_unique <= min(20, max_categories):  # Stricter range for non-dictionary columns
            # Check if it's not obviously continuous data
            try:
                # If all values can be converted to float and show continuous distribution, skip
                numeric_values = adata.obs[col].dropna().astype(float)
                if len(numeric_values.unique()) > n_unique * 0.8:  # Likely continuous
                    continue
            except (ValueError, TypeError):
                # Not numeric, likely categorical
                is_metadata = True
        
        if is_metadata and 2 <= n_unique <= max_categories:
            metadata_columns.append(col)
        elif is_metadata:
            logging.info(f"Excluding {col} from metadata analysis: {n_unique} unique values (outside range 2-{max_categories})")
    
    if metadata_columns:
        logging.info(f"Found metadata columns: {metadata_columns}")
    else:
        logging.warning("No suitable metadata columns found in adata.obs")
    
    return metadata_columns


def create_color_legend(adata, obs_key: str, save_path: Path, title: str = None):
    """
    Create a simple color legend showing how categories map to colors.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    obs_key : str
        Key in adata.obs containing categorical data
    save_path : Path
        Path to save the legend image
    title : str, optional
        Title for the legend
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt
    
    if obs_key not in adata.obs.columns:
        logging.warning(f"Observation key '{obs_key}' not found in adata.obs")
        return
    
    # Get unique categories
    categories = adata.obs[obs_key].cat.categories if hasattr(adata.obs[obs_key], 'cat') else sorted(adata.obs[obs_key].unique())
    
    # Try to get colors from adata.uns, otherwise use default colormap
    colors = None
    color_key = f"{obs_key}_colors"
    if color_key in adata.uns:
        colors = adata.uns[color_key]
    else:
        # Use matplotlib's tab20 colormap as default (same as scanpy default)
        cmap = matplotlib.colormaps['tab20']
        colors = [cmap(i / len(categories)) for i in range(len(categories))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, max(3, len(categories) * 0.3)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(categories))
    
    # Create legend patches
    patches = []
    for i, (category, color) in enumerate(zip(categories, colors)):
        # Convert color to matplotlib format if needed
        if isinstance(color, str):
            patch_color = color
        else:
            patch_color = color
        
        patch = mpatches.Rectangle((0.1, len(categories) - i - 0.8), 0.1, 0.6, 
                                 facecolor=patch_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(patch)
        
        # Add text label
        ax.text(0.25, len(categories) - i - 0.5, str(category), 
               verticalalignment='center', fontsize=10)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    else:
        ax.set_title(f"Color Legend: {obs_key}", fontsize=12, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Color legend saved to {save_path}")


def create_categorical_umaps(adata, categorical_columns, qc_umap_dir, qc_legend_dir, viz_config, category_type="categorical"):
    """
    Create UMAP plots colored by categorical columns (populations or metadata).
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    categorical_columns : list
        List of categorical column names
    qc_umap_dir : Path
        Directory to save UMAP plots
    qc_legend_dir : Path
        Directory to save color legends
    viz_config : VisualizationConfig
        Visualization configuration
    category_type : str
        Type of categories for logging ('population', 'metadata', etc.)
    """
    try:
        make_individual_highlights = getattr(viz_config, 'umap_plot_individual_highlights', True)
        for cat_col in categorical_columns:
            if cat_col in adata.obs.columns:
                logging.info(f'Creating UMAP for {category_type} column: {cat_col}')
                try:
                    fig = sc.pl.umap(
                        adata,
                        color=cat_col,
                        size=10,
                        legend_loc='right margin',
                        return_fig=True
                    )
                    fig_path = qc_umap_dir / f'UMAP_{cat_col}.{viz_config.figure_format}'
                    fig.savefig(fig_path, bbox_inches='tight', dpi=300 if viz_config.save_high_res else 150)
                    plt.close(fig)

                    # Optional: for population columns, create one highlighted UMAP per category
                    if make_individual_highlights and category_type == "population":
                        try:
                            if not isinstance(adata.obs[cat_col].dtype, pd.CategoricalDtype):
                                logging.info(
                                    f"Column '{cat_col}' is not categorical; converting to categorical for highlighted UMAPs."
                                )
                                adata.obs[cat_col] = adata.obs[cat_col].astype("category")

                            highlight_dir = qc_umap_dir / 'Individual_Highlights' / cleanstring(cat_col)
                            try:
                                sbt_utils.plot_umap_highlight_clusters(
                                    adata=adata,
                                    subcluster_col=cat_col,
                                    point_size=10,
                                    legend_loc='none',
                                    show=False,
                                    save_dir=str(highlight_dir),
                                    save_dpi=300 if viz_config.save_high_res else 150,
                                )
                            except AttributeError as attr_err:
                                if "Can only use .cat accessor with a 'category' dtype" not in str(attr_err):
                                    raise
                                logging.warning(
                                    f"Caught non-categorical .cat accessor error for '{cat_col}'. "
                                    "Converting to categorical and retrying highlighted UMAPs."
                                )
                                adata.obs[cat_col] = adata.obs[cat_col].astype("category")
                                sbt_utils.plot_umap_highlight_clusters(
                                    adata=adata,
                                    subcluster_col=cat_col,
                                    point_size=10,
                                    legend_loc='none',
                                    show=False,
                                    save_dir=str(highlight_dir),
                                    save_dpi=300 if viz_config.save_high_res else 150,
                                )
                            logging.info(f'Individual highlighted UMAPs saved to {highlight_dir}')
                        except Exception as e:
                            log_detailed_error(e, f"creating individual highlighted UMAPs for '{cat_col}'")
                    
                except Exception as e:
                    log_detailed_error(e, f"creating UMAP for {category_type} column '{cat_col}'")
            else:
                logging.warning(f'{cat_col} not found in adata.obs; skipping UMAP.')
    except Exception as e:
        log_detailed_error(e, f"{category_type.title()} UMAP visualization step")


def create_categorical_matrix_plots(adata, categorical_columns, qc_matrix_dir, viz_config, category_type="categorical", remove_markers_list=None):
    """
    Create MatrixPlot summaries grouped by categorical columns (populations or metadata).
    Creates both standard-scaled and vmax-capped versions.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    categorical_columns : list
        List of categorical column names
    qc_matrix_dir : Path
        Directory to save matrix plots
    viz_config : VisualizationConfig
        Visualization configuration
    category_type : str
        Type of categories for logging ('population', 'metadata', etc.)
    remove_markers_list : list, optional
        List of markers to exclude when creating filtered matrix plots
    """
    try:
        use_row_color_matrixplot = getattr(viz_config, 'matrixplot_use_row_colors', True)
        if use_row_color_matrixplot:
            if sbt_plotting is None or not hasattr(sbt_plotting, 'matrixplot_with_row_colors'):
                logging.warning(
                    "matrixplot_use_row_colors=True but plotting.matrixplot_with_row_colors is unavailable. "
                    "Falling back to scanpy.pl.matrixplot."
                )
                use_row_color_matrixplot = False

        save_dpi = 300 if viz_config.save_high_res else 150

        def _create_and_save_matrixplot(
            var_names,
            groupby,
            out_path,
            standard_scale=None,
            vmax=None,
        ):
            if use_row_color_matrixplot:
                _, fig = sbt_plotting.matrixplot_with_row_colors(
                    adata,
                    marker_groups=var_names,
                    groupby_key=groupby,
                    out_path=str(out_path),
                    reorder_var_by_expression=False,
                    standard_scale=standard_scale,
                    vmax=vmax,
                    dendrogram=True,
                    save_dpi=save_dpi,
                )
                plt.close(fig)
            else:
                matrixplot_obj = sc.pl.matrixplot(
                    adata,
                    var_names=var_names,
                    groupby=groupby,
                    standard_scale=standard_scale,
                    dendrogram=True,
                    vmax=vmax,
                    show=False,
                    return_fig=True,
                )
                matrixplot_obj.savefig(out_path, bbox_inches='tight', dpi=save_dpi)
                plt.close()

        markers_to_plot = adata.var_names.tolist()
        for cat_col in categorical_columns:
            if cat_col in adata.obs.columns:
                logging.info(f'Creating MatrixPlots for {category_type} column: {cat_col}')
                try:
                    # Pre-compute dendrogram to avoid warning
                    sc.tl.dendrogram(adata, groupby=cat_col)
                    
                    # 1. Create standard-scaled matrixplot
                    logging.info(f'Creating standard-scaled MatrixPlot for {cat_col}')
                    ordered_markers = sbt_utils.reorder_vars_by_expression(adata, markers_to_plot)
                    fig_path_scaled = qc_matrix_dir / f'Matrixplot_{cat_col}_scaled.{viz_config.figure_format}'
                    _create_and_save_matrixplot(
                        var_names=ordered_markers,
                        groupby=cat_col,
                        out_path=fig_path_scaled,
                        standard_scale='var',
                        vmax=None,
                    )
                    logging.info(f'Standard-scaled MatrixPlot saved to {fig_path_scaled}')
                    
                    # 2. Create non-scaled matrixplot with vmax
                    logging.info(f'Creating vmax-capped MatrixPlot for {cat_col} (vmax={viz_config.matrixplot_vmax})')
                    fig_path_vmax = qc_matrix_dir / f'Matrixplot_{cat_col}_vmax.{viz_config.figure_format}'
                    _create_and_save_matrixplot(
                        var_names=ordered_markers,
                        groupby=cat_col,
                        out_path=fig_path_vmax,
                        standard_scale=None,
                        vmax=viz_config.matrixplot_vmax,
                    )
                    logging.info(f'Vmax-capped MatrixPlot saved to {fig_path_vmax}')
                    
                    # 3. Create filtered matrix plots if remove_markers_list is provided
                    if remove_markers_list and len(remove_markers_list) > 0:
                        # Filter out the markers to remove
                        filtered_markers = [m for m in markers_to_plot if m not in remove_markers_list]
                        
                        if filtered_markers:
                            logging.info(f'Creating filtered MatrixPlots for {cat_col} (excluding {len(remove_markers_list)} markers: {remove_markers_list})')
                            
                            # 3a. Create filtered standard-scaled matrixplot
                            logging.info(f'Creating filtered standard-scaled MatrixPlot for {cat_col}')
                            ordered_filtered_markers = sbt_utils.reorder_vars_by_expression(adata, filtered_markers)
                            fig_path_scaled_filtered = qc_matrix_dir / f'Matrixplot_{cat_col}_scaled_filtered.{viz_config.figure_format}'
                            _create_and_save_matrixplot(
                                var_names=ordered_filtered_markers,
                                groupby=cat_col,
                                out_path=fig_path_scaled_filtered,
                                standard_scale='var',
                                vmax=None,
                            )
                            logging.info(f'Filtered standard-scaled MatrixPlot saved to {fig_path_scaled_filtered}')
                            
                            # 3b. Create filtered non-scaled matrixplot with vmax
                            logging.info(f'Creating filtered vmax-capped MatrixPlot for {cat_col} (vmax={viz_config.matrixplot_vmax})')
                            fig_path_vmax_filtered = qc_matrix_dir / f'Matrixplot_{cat_col}_vmax_filtered.{viz_config.figure_format}'
                            _create_and_save_matrixplot(
                                var_names=ordered_filtered_markers,
                                groupby=cat_col,
                                out_path=fig_path_vmax_filtered,
                                standard_scale=None,
                                vmax=viz_config.matrixplot_vmax,
                            )
                            logging.info(f'Filtered vmax-capped MatrixPlot saved to {fig_path_vmax_filtered}')
                        else:
                            logging.warning(f'All markers would be filtered out for {cat_col}; skipping filtered plots.')
                    
                except Exception as e:
                    log_detailed_error(e, f"creating MatrixPlot for {category_type} column '{cat_col}'")
            else:
                logging.warning(f'{cat_col} not found in adata.obs; skipping MatrixPlot.')
    except Exception as e:
        log_detailed_error(e, f"{category_type.title()} MatrixPlot visualization step")


def create_marker_umaps(adata, qc_umap_dir, viz_config):
    """
    Create UMAP plots colored by marker expression.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    qc_umap_dir : Path
        Directory to save UMAP plots
    viz_config : VisualizationConfig
        Visualization configuration
    """
    try:
        markers = adata.var_names.tolist()
        colormap = getattr(viz_config, 'umap_marker_colormap', 'viridis')
        logging.info(f'Creating UMAP plots for {len(markers)} markers using colormap: {colormap}')

        # Also create gallery views of all markers (X and any available layers), if available
        if sbt_plotting is not None and hasattr(sbt_plotting, 'umap_marker_gallery'):
            try:
                dpi = 300 if viz_config.save_high_res else 150
                default_gallery_colorbar_label = getattr(
                    viz_config,
                    'umap_marker_gallery_default_colorbar_label',
                    'Nimbus-Inference Score'
                )
                gallery_vmax = getattr(viz_config, 'umap_marker_gallery_vmax', None)
                if isinstance(gallery_vmax, str) and gallery_vmax.strip().lower() in {'', 'none', 'null'}:
                    gallery_vmax = None

                # 1) Gallery for default matrix (adata.X)
                gallery_path = qc_umap_dir / f'UMAP_marker_gallery.{viz_config.figure_format}'
                gallery_fig = sbt_plotting.umap_marker_gallery(
                    adata,
                    markers=markers,
                    cmap=colormap,
                    vmax=gallery_vmax,
                    add_colorbar=True,
                    colorbar_label=default_gallery_colorbar_label,
                    show=False,
                    save=str(gallery_path),
                    dpi=dpi,
                )
                plt.close(gallery_fig)
                logging.info(f'Marker UMAP gallery saved to {gallery_path}')

                # 2) Galleries for each available layer
                for layer_name in adata.layers.keys():
                    safe_layer = cleanstring(layer_name)
                    layer_gallery_path = qc_umap_dir / f'UMAP_marker_gallery_layer_{safe_layer}.{viz_config.figure_format}'
                    layer_fig = sbt_plotting.umap_marker_gallery(
                        adata,
                        markers=markers,
                        cmap=colormap,
                        layer=layer_name,
                        vmax=gallery_vmax,
                        add_colorbar=True,
                        colorbar_label=f'Expression ({layer_name})',
                        show=False,
                        save=str(layer_gallery_path),
                        dpi=dpi,
                    )
                    plt.close(layer_fig)
                    logging.info(f'Marker UMAP gallery for layer {layer_name} saved to {layer_gallery_path}')
            except Exception as e:
                log_detailed_error(e, "creating marker UMAP gallery")
        else:
            logging.warning('plotting.umap_marker_gallery unavailable; skipping marker gallery plot.')

        for marker in markers:
            if marker in adata.var_names:
                logging.info(f'Creating UMAP for marker: {marker}')
                try:
                    fig = sc.pl.umap(
                        adata,
                        color=marker,
                        size=10,
                        use_raw=False,  # Use processed data
                        cmap=colormap,
                        return_fig=True
                    )
                    fig_path = qc_umap_dir / f'UMAP_marker_{marker}.{viz_config.figure_format}'
                    fig.savefig(fig_path, bbox_inches='tight', dpi=300 if viz_config.save_high_res else 150)
                    plt.close(fig)
                except Exception as e:
                    log_detailed_error(e, f"creating UMAP for marker '{marker}'")
            else:
                logging.warning(f'Marker {marker} not found in adata.var_names; skipping UMAP.')
    except Exception as e:
        log_detailed_error(e, "marker UMAP visualization step")


def create_population_tissue_overlays(adata, population_columns, qc_pop_dir, general_config):
    """
    Create tissue population overlays by mapping populations back to masks.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    population_columns : list
        List of population column names
    qc_pop_dir : Path
        Directory to save population images
    general_config : GeneralConfig
        General configuration for masks folder
    """
    try:
        if sbt_plotting is None:
            logging.warning('plotting module unavailable; skipping tissue visualization.')
            return

        roi_obs = getattr(general_config, 'roi_obs', 'ROI')
        if roi_obs not in adata.obs.columns:
            logging.warning("ROI column '%s' not found in adata.obs; skipping tissue visualization.", roi_obs)
            return

        rois = sorted(adata.obs[roi_obs].astype(str).unique().tolist())
        if not rois:
            logging.warning('No ROIs found in adata.obs; skipping tissue visualization.')
            return
            
        for pop_col in population_columns:
            if pop_col not in adata.obs.columns:
                continue
                
            out_dir = qc_pop_dir / f'{pop_col}'
            out_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f'Creating tissue overlays for {pop_col} across {len(rois)} ROIs.')
            
            for roi in rois:
                try:
                    save_path = out_dir / f'{roi}.png'
                    sbt_plotting.obs_to_mask(
                        adata=adata,
                        roi=roi,
                        roi_obs=roi_obs,
                        cat_obs=pop_col,
                        masks_folder=general_config.masks_folder,
                        save_path=str(save_path),
                        background_color='white',
                        separator_color='black'
                    )
                except Exception as e:
                    log_detailed_error(e, f"creating tissue overlay for ROI '{roi}', population column '{pop_col}'")
    except Exception as e:
        log_detailed_error(e, "tissue visualization step")


def create_backgating_assessment(adata, population_columns, viz_config, general_config, qc_base):
    """
    Create backgating assessment for populations.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    population_columns : list
        List of population column names
    viz_config : VisualizationConfig
        Visualization configuration
    general_config : GeneralConfig
        General configuration
    qc_base : Path
        Base QC directory
    """
    try:
        if sbt_backgating is None:
            logging.warning('backgating module unavailable; skipping backgating assessment.')
            return
            
        logging.info("Starting backgating assessment for populations.")
        
        # Check if we have image folder available
        image_folder = getattr(general_config, 'denoised_images_folder', 'images')
        if not Path(image_folder).exists():
            logging.warning(f'Image folder {image_folder} not found; skipping backgating assessment.')
            return
            
        for pop_col in population_columns:
            if pop_col in adata.obs.columns:
                logging.info(f"Running backgating assessment for {pop_col}")
                
                # Set up output folder for this population column
                backgating_output = qc_base / viz_config.backgating_output_folder / f'{pop_col}'
                
                try:
                    # Debug configuration being used
                    use_de = getattr(viz_config, 'backgating_use_differential_expression', True)
                    mode = getattr(viz_config, 'backgating_mode', 'full')
                    pops_list = _resolve_backgating_pops_list(viz_config, pop_col)
                    logging.info(f"Backgating config - use_differential_expression: {use_de}, mode: {mode}")
                    logging.info(
                        "Backgating populations for '%s': %s",
                        pop_col,
                        pops_list if pops_list is not None else "all",
                    )
                    logging.info(f"Specify overrides - red: {viz_config.backgating_specify_red}, "
                                f"green: {viz_config.backgating_specify_green}, blue: {viz_config.backgating_specify_blue}")
                    
                    roi_obs = getattr(general_config, 'roi_obs', 'ROI')
                    sbt_backgating.backgating_assessment(
                        adata=adata,
                        image_folder=image_folder,
                        pop_obs=pop_col,
                        mean_expression_file=f'markers_mean_expression_{pop_col}.csv',
                        backgating_settings_file=f'backgating_settings_{pop_col}.csv',
                        pops_list=pops_list,
                        cells_per_group=viz_config.backgating_cells_per_group,
                        radius=viz_config.backgating_radius,
                        roi_obs=roi_obs,
                        x_loc_obs='X_loc',
                        y_loc_obs='Y_loc',
                        cell_index_obs='Master_Index',
                        object_index_obs='ObjectNumber',
                        # Mask parameters
                        use_masks=viz_config.backgating_use_masks,
                        mask_folder=viz_config.backgating_mask_folder,
                        max_rois_to_save=getattr(viz_config, 'backgating_max_rois_to_save', None),
                        exclude_rois_without_mask=True,
                        # Output settings
                        output_folder=str(backgating_output),
                        # Intensity scaling
                        minimum=viz_config.backgating_minimum,
                        max_quantile=viz_config.backgating_max_quantile,
                        # Population overview setttings
                        population_overlay_outline_width=viz_config.backgating_population_overlay_outline_width,
                        population_overlay_legend_fontsize=viz_config.backgating_population_overlay_legend_fontsize,
                        population_overlay_crop_size=tuple(viz_config.backgating_population_overlay_crop_size) if viz_config.backgating_population_overlay_crop_size is not None else None,
                        population_overlay_crop_origin=viz_config.backgating_population_overlay_crop_origin,
                        population_overlay_show_scale_bar=viz_config.backgating_population_overlay_show_scale_bar,
                        population_overlay_scale_bar_length=viz_config.backgating_population_overlay_scale_bar_length,
                        population_overlay_scale_bar_thickness=viz_config.backgating_population_overlay_scale_bar_thickness,
                        # Marker selection and differential expression
                        markers_exclude=getattr(viz_config, 'backgating_markers_exclude', ['DNA1', 'DNA3']),
                        use_differential_expression=use_de,
                        de_method=getattr(viz_config, 'backgating_de_method', 'wilcoxon'),
                        min_logfc_threshold=getattr(viz_config, 'backgating_min_logfc_threshold', 0.2),
                        max_pval_adj=getattr(viz_config, 'backgating_max_pval_adj', 0.05),
                        mode=mode,  # Control execution mode
                        number_top_markers=viz_config.backgating_number_top_markers,
                        specify_blue=viz_config.backgating_specify_blue,
                        specify_red=viz_config.backgating_specify_red,
                        specify_green=viz_config.backgating_specify_green
                    )
                    logging.info(f"Backgating assessment completed for {pop_col}. Results saved to {backgating_output}")
                    
                except Exception as e:
                    log_detailed_error(e, f"backgating assessment for population column '{pop_col}'")
            else:
                logging.warning(f"{pop_col} not found in adata.obs; skipping backgating assessment.")
                
        logging.info("Backgating assessment for all populations completed.")
    except Exception as e:
        log_detailed_error(e, "backgating assessment step")


def _resolve_backgating_pops_list(viz_config, pop_col: str):
    """
    Resolve an optional backgating population subset for one population obs column.

    Supports either:
    - a dict mapping population obs column -> list of population labels
    - a flat list/scalar reused for all population obs columns
    """
    configured = getattr(viz_config, 'backgating_pops_list', None)
    if configured is None:
        return None

    if isinstance(configured, dict):
        resolved_value = None
        for key in (pop_col, str(pop_col), 'default'):
            if key in configured:
                resolved_value = configured[key]
                break
        if resolved_value is None:
            return None
        return coalesce_config_list(resolved_value, default=None)

    return coalesce_config_list(configured, default=None)


def create_population_metadata_analysis(adata, population_columns, metadata_columns, qc_base, max_categories=20):
    """
    Create population analysis across metadata categories.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    population_columns : list
        List of population column names
    metadata_columns : list
        List of metadata column names
    qc_base : Path
        Base QC directory
    max_categories : int, optional
        Maximum number of categories to plot (default: 20, stricter for plotting)
    """
    logging.info("Starting population analysis across metadata categories...")
    
    try:
        if sbt_plotting is None:
            logging.warning('plotting module unavailable; skipping population metadata analysis.')
            return
            
        # Create output directory for population analysis figures
        population_analysis_dir = qc_base / 'Population_Analysis_Figures'
        population_analysis_dir.mkdir(exist_ok=True)
        
        if not population_columns:
            logging.warning("No population columns found for population analysis")
            return
        
        if not metadata_columns:
            logging.warning("No metadata columns found for population analysis")
            return
            
        # Generate plots for each combination of population column and metadata column
        for population_col in population_columns:
            logging.info(f"Analyzing population column: {population_col} (N populations: {adata.obs[population_col].nunique()})")
            
            # Create subdirectory for this population column
            pop_analysis_subdir = population_analysis_dir / population_col
            pop_analysis_subdir.mkdir(exist_ok=True)
            
            for metadata_col in metadata_columns:
                logging.info(f"Analyzing {population_col} by {metadata_col}")
                
                # Check if this metadata column has reasonable number of categories
                n_categories = adata.obs[metadata_col].nunique()
                logging.info(f"Number of categories in {metadata_col}: {n_categories}")
                
                if n_categories > max_categories:
                    logging.warning(f"Skipping {metadata_col} - too many categories ({n_categories}, max: {max_categories})")
                    continue
                
                try:
                    # Ensure subdirectory exists before saving
                    pop_analysis_subdir.mkdir(parents=True, exist_ok=True)
                    
                    # 1. Raw counts plot
                    logging.info(f"Creating raw counts plot for {population_col} by {metadata_col}...")
                    raw_counts_graph = pop_analysis_subdir / f"{metadata_col}_raw_counts.png"
                    raw_counts_table = pop_analysis_subdir / f"{metadata_col}_raw_counts.csv"
                    raw_counts_graph.parent.mkdir(parents=True, exist_ok=True)
                    
                    sbt_plotting.grouped_graph(
                        adata,
                        group_by_obs=population_col,
                        x_axis=metadata_col,
                        proportions=False,
                        log_scale=True,
                        fig_size=(max(8, n_categories * 0.8), 6),
                        display_tables=False,
                        save_graph=str(raw_counts_graph),
                        save_table=str(raw_counts_table)
                    )
                    
                    # 2. Proportions plot  
                    logging.info(f"Creating proportions plot for {population_col} by {metadata_col}...")
                    proportions_graph = pop_analysis_subdir / f"{metadata_col}_proportions.png"
                    proportions_table = pop_analysis_subdir / f"{metadata_col}_proportions.csv"
                    proportions_graph.parent.mkdir(parents=True, exist_ok=True)
                    
                    sbt_plotting.grouped_graph(
                        adata,
                        group_by_obs=population_col,
                        x_axis=metadata_col,
                        proportions=True,
                        log_scale=False,
                        fig_size=(max(8, n_categories * 0.8), 6),
                        display_tables=False,
                        save_graph=str(proportions_graph),
                        save_table=str(proportions_table)
                    )
                    
                    # 3. Stacked plot for better comparison (proportions, bars add up to 1)
                    logging.info(f"Creating stacked plot for {population_col} by {metadata_col}...")
                    stacked_graph = pop_analysis_subdir / f"{metadata_col}_stacked.png"
                    stacked_table = pop_analysis_subdir / f"{metadata_col}_stacked.csv"
                    stacked_graph.parent.mkdir(parents=True, exist_ok=True)
                    
                    sbt_plotting.grouped_graph(
                        adata,
                        group_by_obs=population_col,
                        x_axis=metadata_col,
                        proportions=True,
                        stacked=True,
                        log_scale=False,
                        fig_size=(max(8, n_categories * 0.8), 6),
                        display_tables=False,
                        save_graph=str(stacked_graph),
                        save_table=str(stacked_table)
                    )
                    
                except Exception as e:
                    log_detailed_error(e, f"creating plots for population column '{population_col}' by metadata column '{metadata_col}'")
        
        logging.info(f"Population analysis completed. Figures saved to: {population_analysis_dir}")
        
    except Exception as e:
        log_detailed_error(e, "population analysis step")


def _ordered_groups(series: pd.Series, configured_groups=None):
    if configured_groups is not None and len(configured_groups) > 0:
        observed_groups = {str(x) for x in series.dropna().astype(str).unique().tolist()}
        configured = [str(x) for x in configured_groups]
        selected = [x for x in configured if x in observed_groups]
        missing = [x for x in configured if x not in observed_groups]
        if missing:
            logging.warning("Configured groups not found in '%s': %s", series.name, missing)
        if selected:
            return selected

    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories if pd.notna(x)]
    return sorted([str(x) for x in series.dropna().astype(str).unique().tolist()])


def _population_order(adata: ad.AnnData, pop_col: str):
    if pop_col not in adata.obs.columns:
        return []
    series = adata.obs[pop_col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories if pd.notna(x)]
    return sorted([str(x) for x in series.dropna().astype(str).unique().tolist()])


def _population_palette(adata: ad.AnnData, pop_col: str, population_order):
    default_color = '#4C72B0'
    palette = {str(pop): default_color for pop in population_order}
    color_key = f'{pop_col}_colors'
    if color_key not in adata.uns:
        return palette

    colors = list(adata.uns[color_key])
    categories = _population_order(adata, pop_col)
    for idx, pop in enumerate(categories):
        if idx < len(colors):
            palette[str(pop)] = colors[idx]
    return palette


def _build_roi_area_mm2_map(
    adata: ad.AnnData,
    general_config,
    roi_obs=None,
    roi_col=None,
):
    # Backward-compatible arg handling: accept either roi_obs or roi_col.
    if roi_obs is None and roi_col is not None:
        roi_obs = roi_col
    elif roi_obs is not None and roi_col is not None and roi_obs != roi_col:
        logging.warning(
            "Both roi_obs ('%s') and roi_col ('%s') were provided; using roi_obs.",
            roi_obs,
            roi_col,
        )
    if roi_obs is None:
        roi_obs = getattr(general_config, 'roi_obs', 'ROI')

    roi_area = {}
    valid_rois = None
    if roi_obs in adata.obs.columns:
        valid_rois = set(adata.obs[roi_obs].dropna().astype(str).unique().tolist())

    sample_uns = adata.uns.get('sample', None)
    if isinstance(sample_uns, pd.DataFrame) and 'mm2' in sample_uns.columns:
        for roi, mm2 in sample_uns['mm2'].items():
            try:
                roi_key = str(roi)
                if valid_rois is None or roi_key in valid_rois:
                    roi_area[roi_key] = float(mm2)
            except Exception:
                continue
    elif isinstance(sample_uns, dict):
        mm2_data = sample_uns.get('mm2', None)
        if isinstance(mm2_data, pd.Series):
            for roi, mm2 in mm2_data.items():
                try:
                    roi_key = str(roi)
                    if valid_rois is None or roi_key in valid_rois:
                        roi_area[roi_key] = float(mm2)
                except Exception:
                    continue
        elif isinstance(mm2_data, dict):
            for roi, mm2 in mm2_data.items():
                try:
                    roi_key = str(roi)
                    if valid_rois is None or roi_key in valid_rois:
                        roi_area[roi_key] = float(mm2)
                except Exception:
                    continue

    if roi_area:
        logging.info("Loaded %d ROI areas from adata.uns['sample']['mm2'].", len(roi_area))
        return roi_area

    masks_folder = Path(getattr(general_config, 'masks_folder', 'masks'))
    if not masks_folder.exists():
        logging.warning("Masks folder '%s' not found. Cells/mm2 plots will be skipped.", masks_folder)
        return roi_area

    for mask_path in sorted(masks_folder.glob('*.tif*')):
        roi_name = mask_path.stem
        if valid_rois is not None and roi_name not in valid_rois:
            continue
        mask = None
        try:
            import tifffile as tiff
            mask = tiff.imread(mask_path)
        except Exception:
            try:
                import imageio.v3 as iio
                mask = iio.imread(mask_path)
            except Exception as err:
                logging.warning("Could not read mask '%s': %s", mask_path, err)
                continue

        try:
            area_um2 = float(mask.shape[0]) * float(mask.shape[1])
            roi_area[str(roi_name)] = area_um2 / 1e6
        except Exception:
            continue

    if not roi_area:
        logging.warning(
            "No ROI area map could be built from masks. "
            "Cells/mm2 plotting and stats will be skipped."
        )
    else:
        logging.info("Built ROI area map from masks for %d ROIs.", len(roi_area))
    return roi_area


def _save_df(df: pd.DataFrame, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)


def _normalise_barplot_scale_mode(value: Any, *, context: str = "") -> Optional[str]:
    text = str(value).strip().lower()
    if text in {"linear", "log", "intelligent"}:
        return text
    if context:
        logging.warning(
            "Invalid barplot y-scale '%s' for %s. Valid options are: linear, log, intelligent.",
            value,
            context,
        )
    return None


def _resolve_barplot_scale_mode(
    scale_cfg: Any,
    *,
    analysis: str,
    metric: str,
) -> str:
    analysis_key = str(analysis)
    metric_key = str(metric)

    if isinstance(scale_cfg, dict):
        direct_key = f"{analysis_key}.{metric_key}"
        if direct_key in scale_cfg:
            mode = _normalise_barplot_scale_mode(
                scale_cfg.get(direct_key), context=f"{analysis_key}.{metric_key}"
            )
            if mode:
                return mode

        analysis_cfg = scale_cfg.get(analysis_key)
        if isinstance(analysis_cfg, dict):
            for key in [metric_key, "default", "*"]:
                if key in analysis_cfg:
                    mode = _normalise_barplot_scale_mode(
                        analysis_cfg.get(key), context=f"{analysis_key}.{key}"
                    )
                    if mode:
                        return mode
        elif analysis_cfg is not None:
            mode = _normalise_barplot_scale_mode(analysis_cfg, context=analysis_key)
            if mode:
                return mode

        if metric_key in scale_cfg and not isinstance(scale_cfg.get(metric_key), dict):
            mode = _normalise_barplot_scale_mode(scale_cfg.get(metric_key), context=metric_key)
            if mode:
                return mode

        for key in ["default", "*"]:
            if key in scale_cfg:
                mode = _normalise_barplot_scale_mode(scale_cfg.get(key), context=key)
                if mode:
                    return mode
        return "linear"

    mode = _normalise_barplot_scale_mode(scale_cfg, context="abundance_barplot_y_scale")
    return mode or "linear"


def _resolve_intelligent_scale_params(params: Any) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "allow_log1p": True,
        "dynamic_range_thresh": 100.0,
        "skew_improve_ratio": 0.7,
        "crush_frac_thresh": 0.7,
    }
    if not isinstance(params, dict):
        return defaults

    resolved = defaults.copy()
    if "allow_log1p" in params:
        resolved["allow_log1p"] = bool(params.get("allow_log1p"))
    for key in ["dynamic_range_thresh", "skew_improve_ratio", "crush_frac_thresh"]:
        if key in params:
            try:
                resolved[key] = float(params.get(key))
            except Exception:
                logging.warning(
                    "Invalid abundance_barplot_y_scale_intelligent_params['%s']=%s. Using default %s.",
                    key,
                    params.get(key),
                    defaults[key],
                )
    return resolved


def _choose_scale_1d(
    x: Any,
    *,
    allow_log1p: bool = True,
    dynamic_range_thresh: float = 100.0,
    skew_improve_ratio: float = 0.7,
    crush_frac_thresh: float = 0.7,
) -> str:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return "linear"

    if np.any(arr <= 0):
        return "linear"

    min_pos = float(np.min(arr))
    max_pos = float(np.max(arr))
    if not np.isfinite(min_pos) or not np.isfinite(max_pos) or max_pos <= 0:
        return "linear"

    dyn = max_pos / min_pos if min_pos > 0 else np.inf

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(hi, lo):
        return "linear"
    crush_frac = float(np.mean(arr <= lo + 0.05 * (hi - lo)))

    def _skew(values: np.ndarray) -> float:
        m = float(np.mean(values))
        s = float(np.std(values, ddof=0))
        if np.isclose(s, 0.0):
            return 0.0
        z = (values - m) / s
        return float(np.mean(z**3))

    skew_lin = abs(_skew(arr))
    if allow_log1p:
        transformed = np.log1p(arr)
    else:
        transformed = np.log(arr)
    skew_log = abs(_skew(transformed))

    log_helpful = (dyn >= float(dynamic_range_thresh)) and (
        (skew_log <= skew_lin * float(skew_improve_ratio))
        or (crush_frac >= float(crush_frac_thresh))
    )
    return "log" if log_helpful else "linear"


def _apply_barplot_y_scale(
    ax: Any,
    values: pd.Series,
    *,
    requested_mode: str,
    analysis: str,
    metric: str,
    intelligent_params: Dict[str, Any],
) -> str:
    mode = str(requested_mode).lower()
    if mode == "intelligent":
        mode = _choose_scale_1d(values.to_numpy(dtype=float), **intelligent_params)

    if mode == "log":
        finite = values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0 or np.any(finite <= 0):
            logging.warning(
                "Requested log y-scale for %s/%s barplot, but values include non-positive entries. Using linear scale.",
                analysis,
                metric,
            )
            return "linear"

        min_pos = float(np.min(finite))
        max_val = float(np.max(finite))
        ax.set_yscale("log")
        if np.isfinite(min_pos) and np.isfinite(max_val) and max_val > min_pos:
            lower = max(min_pos * 0.9, np.finfo(float).tiny)
            upper = max_val * 1.1
            if upper > lower:
                ax.set_ylim(bottom=lower, top=upper)
        return "log"

    return "linear"


def _plot_population_bar(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    order,
    color,
    ylabel: str,
    title: str,
    save_path: Path,
    save_high_res: bool,
    y_scale_mode: str = "linear",
    y_scale_intelligent_params: Optional[Dict[str, Any]] = None,
    scale_metric_name: str = "",
):
    if data.empty:
        return
    if y_scale_intelligent_params is None:
        y_scale_intelligent_params = _resolve_intelligent_scale_params(None)

    fig_width = 1.5 if len(order) <= 2 else 2.0
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    sns.barplot(
        data=data,
        x=x_col,
        y=y_col,
        order=order,
        color=color,
        edgecolor='black',
        linewidth=0.8,
        errorbar='se',
        err_kws={'linewidth': 2},
        capsize=0.2,
        ax=ax
    )
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10, rotation=90)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=10)
    _apply_barplot_y_scale(
        ax,
        data[y_col],
        requested_mode=y_scale_mode,
        analysis="abundance",
        metric=scale_metric_name or y_col,
        intelligent_params=y_scale_intelligent_params,
    )
    ax.grid(False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=300 if save_high_res else 150)
    plt.close(fig)


def _dedupe_legend(ax: Any, *, title: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen = set()
    unique_h = []
    unique_l = []
    for handle, label in zip(handles, labels):
        name = str(label)
        if not name or name.startswith("_") or name in seen:
            continue
        seen.add(name)
        unique_h.append(handle)
        unique_l.append(name)
    if unique_h:
        ax.legend(
            unique_h,
            unique_l,
            title=title,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def _majority_label(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return ""
    modes = values.mode()
    if not modes.empty:
        return str(modes.iloc[0])
    return str(values.iloc[0])


def _plot_all_populations_by_group(
    data: pd.DataFrame,
    *,
    pop_col: str,
    group_col: str,
    value_col: str,
    pop_order,
    group_order,
    group_palette: Dict[str, str],
    ylabel: str,
    title: str,
    save_path: Path,
    save_high_res: bool,
    base_figsize,
    width_scale: float,
    y_scale_mode: str = "linear",
    y_scale_intelligent_params: Optional[Dict[str, Any]] = None,
    scale_metric_name: str = "",
):
    if data.empty:
        return
    if y_scale_intelligent_params is None:
        y_scale_intelligent_params = _resolve_intelligent_scale_params(None)

    plot_data = data[[pop_col, group_col, value_col]].dropna().copy()
    if plot_data.empty:
        return

    plot_data[pop_col] = plot_data[pop_col].astype(str)
    plot_data[group_col] = plot_data[group_col].astype(str)
    pop_order = [str(x) for x in pop_order if str(x) in plot_data[pop_col].unique().tolist()]
    group_order = [str(x) for x in group_order if str(x) in plot_data[group_col].unique().tolist()]
    if not pop_order or not group_order:
        return

    base_width = float(base_figsize[0]) if len(base_figsize) > 0 else 4.0
    base_height = float(base_figsize[1]) if len(base_figsize) > 1 else 3.0
    width_scale = max(0.05, float(width_scale))
    plot_width = max(base_width, width_scale * max(1, len(pop_order)))

    fallback = sns.color_palette("tab10", n_colors=max(1, len(group_order))).as_hex()
    palette = {g: str(group_palette.get(g, fallback[i])) for i, g in enumerate(group_order)}

    fig, ax = plt.subplots(figsize=(plot_width, base_height))
    sns.barplot(
        data=plot_data,
        x=pop_col,
        y=value_col,
        hue=group_col,
        order=pop_order,
        hue_order=group_order,
        errorbar="se" if len(plot_data) > 1 else None,
        palette=palette,
        edgecolor='black',
        linewidth=0.6,
        capsize=0.2,
        ax=ax,
    )

    ax.tick_params(axis='x', labelsize=10, rotation=90)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(pop_col, fontsize=10)
    ax.set_title(title, fontsize=10)
    _apply_barplot_y_scale(
        ax,
        plot_data[value_col],
        requested_mode=y_scale_mode,
        analysis="abundance",
        metric=scale_metric_name or value_col,
        intelligent_params=y_scale_intelligent_params,
    )
    _dedupe_legend(ax, title=group_col)
    ax.grid(False)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=300 if save_high_res else 150)
    plt.close(fig)


def _plot_case_stacked_proportions(
    case_props: pd.DataFrame,
    *,
    case_col: str,
    pop_col: str,
    value_col: str,
    pop_order,
    palette: Dict[str, str],
    title: str,
    save_path: Path,
    save_high_res: bool,
    base_figsize,
    width_scale: float,
    order_by_population: Optional[str] = None,
    wide_csv_path: Optional[Path] = None,
):
    if case_props.empty:
        return

    wide = case_props.pivot_table(
        index=case_col,
        columns=pop_col,
        values=value_col,
        aggfunc="mean",
        fill_value=0.0,
    )
    if wide.empty:
        return

    ordered_cols = [str(p) for p in pop_order if str(p) in wide.columns]
    if ordered_cols:
        wide = wide.reindex(columns=ordered_cols, fill_value=0.0)

    order_label = str(order_by_population) if order_by_population is not None else ""
    if order_label and order_label in wide.columns:
        wide = wide.sort_values(by=order_label, ascending=False)
    elif order_label:
        logging.info(
            "abundance_order_cases_by_population '%s' is not present for '%s'; keeping default case order.",
            order_label,
            pop_col,
        )

    if wide_csv_path is not None:
        wide_csv_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(wide_csv_path, index=True)

    base_width = float(base_figsize[0]) if len(base_figsize) > 0 else 6.0
    base_height = float(base_figsize[1]) if len(base_figsize) > 1 else 3.0
    width_scale = max(0.05, float(width_scale))
    plot_width = max(base_width, width_scale * max(1, wide.shape[0]))

    colors = [palette.get(str(col), "#808080") for col in wide.columns]
    fig, ax = plt.subplots(figsize=(plot_width, base_height))
    wide.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.9)
    ax.set_xlabel(case_col, fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=10, rotation=90)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.margins(y=0.0)
    ax.grid(False)
    ax.legend(
        title=pop_col,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300 if save_high_res else 150)
    plt.close(fig)


def _run_mlm_stats_and_save(
    data: pd.DataFrame,
    pop_col: str,
    group_col: str,
    case_col: str,
    roi_col: str,
    value_col: str,
    average_cases: bool,
    stats_dir: Path,
    out_prefix: str,
):
    if sbt_plotting is None or not hasattr(sbt_plotting, 'mlm_stats'):
        logging.warning("plotting.mlm_stats unavailable; skipping stats for %s.", out_prefix)
        return

    required_cols = [pop_col, group_col, case_col, roi_col, value_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logging.warning("Skipping stats for %s; missing required columns: %s", out_prefix, missing_cols)
        return

    stats_input = data.dropna(subset=required_cols).copy()
    if stats_input.empty:
        logging.warning("Skipping stats for %s; no rows after filtering.", out_prefix)
        return

    stats_input[pop_col] = stats_input[pop_col].astype(str)
    stats_input[group_col] = stats_input[group_col].astype(str)
    stats_input[case_col] = stats_input[case_col].astype(str)
    stats_input[roi_col] = stats_input[roi_col].astype(str)

    n_groups = stats_input[group_col].nunique()
    if n_groups != 2:
        logging.warning(
            "Skipping stats for %s; mlm_stats requires exactly 2 groups in '%s', found %d.",
            out_prefix,
            group_col,
            n_groups,
        )
        return

    suffix = 'case_avg' if average_cases else 'roi_level'
    try:
        results_df = sbt_plotting.mlm_stats(
            stats_input,
            pop_col=pop_col,
            value_col=value_col,
            case_col=case_col,
            group_col=group_col,
            roi_col=roi_col,
            method='fdr_bh',
            average_cases=average_cases,
        )
    except Exception as e:
        log_detailed_error(e, f"running mlm_stats for {out_prefix} ({suffix})")
        return

    stats_dir.mkdir(parents=True, exist_ok=True)
    full_path = stats_dir / f"{out_prefix}_{suffix}_full.csv"
    results_df.to_csv(full_path, index=False)

    if average_cases:
        preferred_cols = [
            pop_col,
            't_test_p_value',
            'mannwhitneyu_p_value',
            't_test_p_value_corrected',
            'mannwhitneyu_p_value_corrected',
        ]
        sort_col = 'mannwhitneyu_p_value' if 'mannwhitneyu_p_value' in results_df.columns else None
    else:
        preferred_cols = [
            pop_col,
            'mlm_p_value',
            'mlm_p_value_corrected',
            'mlm_warnings',
            't_test_p_value',
            'mannwhitneyu_p_value',
            't_test_p_value_corrected',
            'mannwhitneyu_p_value_corrected',
        ]
        sort_col = 'mlm_p_value' if 'mlm_p_value' in results_df.columns else None

    summary_cols = [col for col in preferred_cols if col in results_df.columns]
    summary_df = results_df[summary_cols].copy()
    if sort_col is not None:
        summary_df = summary_df.sort_values(by=sort_col)
    summary_path = stats_dir / f"{out_prefix}_{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved stats: %s and %s", summary_path, full_path)


def _resolve_population_abundance_scopes(
    adata: ad.AnnData,
    general_config,
):
    scopes = [
        {
            'label': 'all',
            'folder_name': 'all',
            'mask': None,
            'allow_mm2': True,
        }
    ]

    compartment_col = coalesce_config_text(getattr(general_config, 'compartment_obs', None))
    if not compartment_col:
        return scopes

    if compartment_col not in adata.obs.columns:
        logging.warning(
            "Compartment column '%s' not found in adata.obs; running abundance analysis for all cells only.",
            compartment_col,
        )
        return scopes

    configured_compartments = coalesce_config_list(getattr(general_config, 'compartment_obs_list', None))
    if not configured_compartments:
        logging.warning(
            "Compartment column '%s' is configured but compartment_obs_list is empty; "
            "running abundance analysis for all cells only.",
            compartment_col,
        )
        return scopes

    compartment_values = adata.obs[compartment_col].astype("string")
    seen_labels = set()
    used_folder_names = {'all'}

    for idx, raw_label in enumerate(configured_compartments, start=1):
        label = str(raw_label)
        if label in seen_labels:
            logging.warning("Duplicate compartment '%s' in compartment_obs_list; skipping repeated entry.", label)
            continue
        seen_labels.add(label)

        scope_mask = compartment_values.eq(label).fillna(False).astype(bool)
        n_cells = int(scope_mask.sum())
        if n_cells == 0:
            logging.warning(
                "Configured compartment '%s' was not found in adata.obs['%s']; skipping it.",
                label,
                compartment_col,
            )
            continue

        folder_name = cleanstring(label) or f'compartment_{idx}'
        if folder_name in used_folder_names:
            suffix = 2
            base_name = folder_name
            while folder_name in used_folder_names:
                folder_name = f'{base_name}_{suffix}'
                suffix += 1
        used_folder_names.add(folder_name)

        scopes.append(
            {
                'label': label,
                'folder_name': folder_name,
                'mask': scope_mask,
                'allow_mm2': False,
            }
        )

    if len(scopes) > 1:
        logging.info(
            "Compartment-specific abundance analysis enabled using '%s' for compartments: %s",
            compartment_col,
            [scope['label'] for scope in scopes[1:]],
        )

    return scopes


def _run_population_abundance_analysis_scope(
    adata: ad.AnnData,
    population_columns,
    viz_config,
    *,
    analysis_root: Path,
    scope_label: str,
    scope_mask,
    allow_mm2: bool,
    group_col: str,
    base_group_order,
    roi_col: str,
    case_col: Optional[str],
    roi_area_map: Dict[str, float],
):
    raw_root = analysis_root / 'Raw_Data'
    plot_root = analysis_root / 'Plots'
    stats_root = analysis_root / 'Stats'
    for out_dir in [analysis_root, raw_root, plot_root, stats_root]:
        out_dir.mkdir(parents=True, exist_ok=True)

    scope_obs = adata.obs if scope_mask is None else adata.obs.loc[scope_mask]
    if scope_obs.empty:
        logging.warning("Abundance analysis scope '%s' has no cells after filtering; skipping.", scope_label)
        return

    observed_groups = set(scope_obs[group_col].dropna().astype(str).unique().tolist())
    group_order = [str(group) for group in base_group_order if str(group) in observed_groups]
    if not group_order:
        group_order = _ordered_groups(scope_obs[group_col], None)
    if not group_order:
        logging.warning(
            "Abundance analysis scope '%s' has no valid groups for '%s'; skipping.",
            scope_label,
            group_col,
        )
        return

    has_roi_areas = allow_mm2 and len(roi_area_map) > 0
    if not allow_mm2:
        logging.info(
            "Abundance analysis scope '%s' is compartment-specific; skipping cells/mm2 outputs and statistics.",
            scope_label,
        )

    make_all_pop_plots = bool(getattr(viz_config, 'abundance_make_all_populations_plots', True))
    raw_all_pop_figsize = getattr(viz_config, 'abundance_all_populations_figsize', [4.0, 3.0])
    if not isinstance(raw_all_pop_figsize, (list, tuple)) or len(raw_all_pop_figsize) < 2:
        all_pop_figsize = (4.0, 3.0)
    else:
        all_pop_figsize = (float(raw_all_pop_figsize[0]), float(raw_all_pop_figsize[1]))
    all_pop_width_scale = max(
        0.05,
        float(getattr(viz_config, 'abundance_all_populations_width_scale', 0.45)),
    )
    make_case_stacked_plots = bool(getattr(viz_config, 'abundance_make_case_stacked_plots', True))
    raw_case_stacked_figsize = getattr(viz_config, 'abundance_case_stacked_figsize', [6.0, 3.0])
    if not isinstance(raw_case_stacked_figsize, (list, tuple)) or len(raw_case_stacked_figsize) < 2:
        case_stacked_figsize = (6.0, 3.0)
    else:
        case_stacked_figsize = (
            float(raw_case_stacked_figsize[0]),
            float(raw_case_stacked_figsize[1]),
        )
    case_stacked_width_scale = max(
        0.05,
        float(getattr(viz_config, 'abundance_case_stacked_width_scale', 0.30)),
    )
    order_cases_by_population = getattr(viz_config, 'abundance_order_cases_by_population', None)
    if order_cases_by_population is not None:
        order_cases_by_population = str(order_cases_by_population)

    abundance_scale_cfg = getattr(viz_config, 'abundance_barplot_y_scale', {'default': 'linear'})
    abundance_scale_intelligent_params = _resolve_intelligent_scale_params(
        getattr(viz_config, 'abundance_barplot_y_scale_intelligent_params', None)
    )
    mode_proportions_roi = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="proportions_roi_level",
    )
    mode_proportions_case = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="proportions_case_average",
    )
    mode_mm2_roi = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="cells_per_mm2_roi_level",
    )
    mode_mm2_case = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="cells_per_mm2_case_average",
    )
    group_palette = _population_palette(adata, group_col, group_order)

    logging.info(
        "Running abundance analysis for scope '%s' with %d cells and groups %s.",
        scope_label,
        int(scope_obs.shape[0]),
        group_order,
    )

    for pop_col in population_columns:
        if pop_col not in adata.obs.columns:
            logging.warning("Population column '%s' not found in adata.obs; skipping.", pop_col)
            continue

        pop_name = cleanstring(str(pop_col))
        pop_root = analysis_root / pop_name
        pop_raw_root = raw_root / pop_name
        pop_plot_root = plot_root / pop_name
        pop_stats_root = stats_root / pop_name
        for out_dir in [pop_root, pop_raw_root, pop_plot_root, pop_stats_root]:
            out_dir.mkdir(parents=True, exist_ok=True)

        required_cols = [pop_col, group_col, roi_col]
        if case_col:
            required_cols.append(case_col)

        obs = scope_obs[required_cols].copy().dropna(subset=required_cols)
        if obs.empty:
            logging.warning("No non-null rows available for population column '%s'; skipping.", pop_col)
            continue

        for col in required_cols:
            obs[col] = obs[col].astype(str)
        obs = obs[obs[group_col].isin(group_order)].copy()
        if obs.empty:
            logging.warning(
                "No rows in '%s' after filtering '%s' to groups %s for scope '%s'; skipping.",
                pop_col,
                group_col,
                group_order,
                scope_label,
            )
            continue

        _save_df(obs, pop_raw_root / 'cell_level_filtered.csv')

        observed_populations = set(obs[pop_col].astype(str).unique().tolist())
        pop_order = [pop for pop in _population_order(adata, pop_col) if pop in observed_populations]
        if not pop_order:
            pop_order = sorted(observed_populations)
        palette = _population_palette(adata, pop_col, pop_order)

        count_group_cols = [pop_col, group_col]
        if case_col:
            count_group_cols.append(case_col)
        count_group_cols.append(roi_col)
        counts = obs.groupby(count_group_cols, observed=True).size().reset_index(name='n_cells')
        counts['n_cells'] = counts['n_cells'].astype(float)
        _save_df(counts, pop_raw_root / 'counts_by_population_group_roi.csv')

        total_group_cols = [group_col, roi_col]
        if case_col:
            total_group_cols.append(case_col)
        totals = obs.groupby(total_group_cols, observed=True).size().reset_index(name='total_cells')
        _save_df(totals, pop_raw_root / 'totals_by_group_roi.csv')

        proportions = counts.merge(totals, on=total_group_cols, how='left')
        proportions['prop_cells'] = proportions['n_cells'] / proportions['total_cells']
        proportions = proportions.replace([np.inf, -np.inf], np.nan).dropna(subset=['prop_cells'])
        _save_df(proportions, pop_raw_root / 'proportions_roi_level.csv')

        counts_mm2 = pd.DataFrame()
        if allow_mm2 and has_roi_areas:
            counts_mm2 = counts.copy()
            counts_mm2['area_mm2'] = counts_mm2[roi_col].map(roi_area_map)
            missing_area = int(counts_mm2['area_mm2'].isna().sum())
            if missing_area > 0:
                logging.warning(
                    "Population column '%s': %d rows missing ROI area in masks/sample table; "
                    "those rows are excluded from cells/mm2.",
                    pop_col,
                    missing_area,
                )
            counts_mm2 = counts_mm2.dropna(subset=['area_mm2']).copy()
            if not counts_mm2.empty:
                counts_mm2['cells_per_mm2'] = counts_mm2['n_cells'] / counts_mm2['area_mm2']
                counts_mm2 = counts_mm2.replace([np.inf, -np.inf], np.nan).dropna(subset=['cells_per_mm2'])
                _save_df(counts_mm2, pop_raw_root / 'cells_per_mm2_roi_level.csv')
        elif allow_mm2:
            logging.warning(
                "Population column '%s': no ROI area map available; skipping cells/mm2 plots and stats.",
                pop_col,
            )

        case_proportions = pd.DataFrame()
        if case_col:
            case_proportions = (
                proportions.groupby([pop_col, group_col, case_col], observed=True, as_index=False)['prop_cells']
                .mean()
            )
            _save_df(case_proportions, pop_raw_root / 'proportions_case_average.csv')

        case_mm2 = pd.DataFrame()
        if case_col and allow_mm2 and has_roi_areas and not counts_mm2.empty:
            case_mm2 = (
                counts_mm2.groupby([pop_col, group_col, case_col], observed=True, as_index=False)['cells_per_mm2']
                .mean()
            )
            _save_df(case_mm2, pop_raw_root / 'cells_per_mm2_case_average.csv')

        case_level_props = pd.DataFrame()
        if case_col:
            case_counts = obs.groupby([case_col, pop_col], observed=True).size().reset_index(name='n_cells')
            case_totals = case_counts.groupby(case_col, observed=True)['n_cells'].transform('sum')
            case_counts['case_proportion'] = case_counts['n_cells'] / case_totals.replace(0, np.nan)
            case_level_props = case_counts[[case_col, pop_col, 'case_proportion']].copy()
            case_level_props = case_level_props.replace([np.inf, -np.inf], np.nan).dropna(subset=['case_proportion'])

            case_group_nunique = obs.groupby(case_col, observed=True)[group_col].nunique(dropna=True)
            ambiguous_cases = case_group_nunique[case_group_nunique > 1].index.tolist()
            if ambiguous_cases:
                logging.warning(
                    "Population '%s': some cases map to multiple '%s' values; using majority assignment for stacked plots.",
                    pop_col,
                    group_col,
                )
            case_group_map = obs.groupby(case_col, observed=True)[group_col].agg(_majority_label).reset_index()
            case_level_props = case_level_props.merge(case_group_map, on=case_col, how='left')
            _save_df(case_level_props, pop_raw_root / 'case_level_proportions.csv')

        if make_case_stacked_plots and case_col and not case_level_props.empty:
            stacked_plot_root = pop_plot_root / 'case_stacked_proportions'
            stacked_raw_root = pop_raw_root / 'case_stacked_proportions'
            stacked_plot_root.mkdir(parents=True, exist_ok=True)
            stacked_raw_root.mkdir(parents=True, exist_ok=True)

            _plot_case_stacked_proportions(
                case_level_props,
                case_col=case_col,
                pop_col=pop_col,
                value_col='case_proportion',
                pop_order=pop_order,
                palette=palette,
                title=f"Case-level {pop_col} proportions",
                save_path=stacked_plot_root / f'stacked_case_proportion_all.{viz_config.figure_format}',
                save_high_res=viz_config.save_high_res,
                base_figsize=case_stacked_figsize,
                width_scale=case_stacked_width_scale,
                order_by_population=order_cases_by_population,
                wide_csv_path=stacked_raw_root / 'stacked_case_proportion_all.csv',
            )

            group_values = case_level_props[group_col].astype(str).unique().tolist()
            available_groups = [group for group in group_order if group in group_values]
            for group in available_groups:
                group_subset = case_level_props[case_level_props[group_col].astype(str) == str(group)].copy()
                if group_subset.empty:
                    continue
                group_safe = cleanstring(str(group))
                _plot_case_stacked_proportions(
                    group_subset,
                    case_col=case_col,
                    pop_col=pop_col,
                    value_col='case_proportion',
                    pop_order=pop_order,
                    palette=palette,
                    title=f"Case-level {pop_col} proportions ({group_col}={group})",
                    save_path=stacked_plot_root / f'stacked_case_proportion_{group_safe}.{viz_config.figure_format}',
                    save_high_res=viz_config.save_high_res,
                    base_figsize=case_stacked_figsize,
                    width_scale=case_stacked_width_scale,
                    order_by_population=order_cases_by_population,
                    wide_csv_path=stacked_raw_root / f'stacked_case_proportion_{group_safe}.csv',
                )

        for pop in pop_order:
            pop_label = str(pop)
            pop_safe = cleanstring(pop_label)
            pop_color = palette.get(pop_label, '#4C72B0')

            pop_prop = proportions[(proportions[pop_col] == pop_label) & (proportions[group_col].isin(group_order))].copy()
            if not pop_prop.empty:
                _save_df(pop_prop, pop_raw_root / 'per_population' / 'proportions_roi_level' / f'{pop_safe}.csv')
                _plot_population_bar(
                    data=pop_prop,
                    x_col=group_col,
                    y_col='prop_cells',
                    order=group_order,
                    color=pop_color,
                    ylabel='Proportion of cells\n(ROI level)',
                    title=pop_label,
                    save_path=pop_plot_root / 'proportions_roi_level' / f'{pop_safe}.{viz_config.figure_format}',
                    save_high_res=viz_config.save_high_res,
                    y_scale_mode=mode_proportions_roi,
                    y_scale_intelligent_params=abundance_scale_intelligent_params,
                    scale_metric_name='proportions_roi_level',
                )

            if case_col and not case_proportions.empty:
                pop_prop_case = case_proportions[
                    (case_proportions[pop_col] == pop_label) & (case_proportions[group_col].isin(group_order))
                ].copy()
                if not pop_prop_case.empty:
                    _save_df(pop_prop_case, pop_raw_root / 'per_population' / 'proportions_case_average' / f'{pop_safe}.csv')
                    _plot_population_bar(
                        data=pop_prop_case,
                        x_col=group_col,
                        y_col='prop_cells',
                        order=group_order,
                        color=pop_color,
                        ylabel='Proportion of cells\n(Case average)',
                        title=pop_label,
                        save_path=pop_plot_root / 'proportions_case_average' / f'{pop_safe}.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        y_scale_mode=mode_proportions_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='proportions_case_average',
                    )

            if allow_mm2 and has_roi_areas and not counts_mm2.empty:
                pop_mm2 = counts_mm2[
                    (counts_mm2[pop_col] == pop_label) & (counts_mm2[group_col].isin(group_order))
                ].copy()
                if not pop_mm2.empty:
                    _save_df(pop_mm2, pop_raw_root / 'per_population' / 'cells_per_mm2_roi_level' / f'{pop_safe}.csv')
                    _plot_population_bar(
                        data=pop_mm2,
                        x_col=group_col,
                        y_col='cells_per_mm2',
                        order=group_order,
                        color=pop_color,
                        ylabel='Cells per mm$^2$\n(ROI level)',
                        title=pop_label,
                        save_path=pop_plot_root / 'cells_per_mm2_roi_level' / f'{pop_safe}.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        y_scale_mode=mode_mm2_roi,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_roi_level',
                    )

            if case_col and allow_mm2 and has_roi_areas and not case_mm2.empty:
                pop_mm2_case = case_mm2[
                    (case_mm2[pop_col] == pop_label) & (case_mm2[group_col].isin(group_order))
                ].copy()
                if not pop_mm2_case.empty:
                    _save_df(pop_mm2_case, pop_raw_root / 'per_population' / 'cells_per_mm2_case_average' / f'{pop_safe}.csv')
                    _plot_population_bar(
                        data=pop_mm2_case,
                        x_col=group_col,
                        y_col='cells_per_mm2',
                        order=group_order,
                        color=pop_color,
                        ylabel='Cells per mm$^2$\n(Case average)',
                        title=pop_label,
                        save_path=pop_plot_root / 'cells_per_mm2_case_average' / f'{pop_safe}.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        y_scale_mode=mode_mm2_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_case_average',
                    )

        if make_all_pop_plots:
            all_raw_root = pop_raw_root / 'all_populations'
            all_plot_root = pop_plot_root / 'all_populations'
            all_raw_root.mkdir(parents=True, exist_ok=True)
            all_plot_root.mkdir(parents=True, exist_ok=True)

            all_prop = proportions[proportions[group_col].isin(group_order)].copy()
            if not all_prop.empty:
                _save_df(all_prop, all_raw_root / 'proportions_roi_level.csv')
                _plot_all_populations_by_group(
                    all_prop,
                    pop_col=pop_col,
                    group_col=group_col,
                    value_col='prop_cells',
                    pop_order=pop_order,
                    group_order=group_order,
                    group_palette=group_palette,
                    ylabel='Proportion of cells\n(ROI level)',
                    title=f"{pop_col}: all populations by {group_col} (ROI level)",
                    save_path=all_plot_root / f'proportions_roi_level.{viz_config.figure_format}',
                    save_high_res=viz_config.save_high_res,
                    base_figsize=all_pop_figsize,
                    width_scale=all_pop_width_scale,
                    y_scale_mode=mode_proportions_roi,
                    y_scale_intelligent_params=abundance_scale_intelligent_params,
                    scale_metric_name='proportions_roi_level',
                )

            if case_col and not case_proportions.empty:
                all_prop_case = case_proportions[case_proportions[group_col].isin(group_order)].copy()
                if not all_prop_case.empty:
                    _save_df(all_prop_case, all_raw_root / 'proportions_case_average.csv')
                    _plot_all_populations_by_group(
                        all_prop_case,
                        pop_col=pop_col,
                        group_col=group_col,
                        value_col='prop_cells',
                        pop_order=pop_order,
                        group_order=group_order,
                        group_palette=group_palette,
                        ylabel='Proportion of cells\n(Case average)',
                        title=f"{pop_col}: all populations by {group_col} (Case average)",
                        save_path=all_plot_root / f'proportions_case_average.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        base_figsize=all_pop_figsize,
                        width_scale=all_pop_width_scale,
                        y_scale_mode=mode_proportions_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='proportions_case_average',
                    )

            if allow_mm2 and has_roi_areas and not counts_mm2.empty:
                all_mm2 = counts_mm2[counts_mm2[group_col].isin(group_order)].copy()
                if not all_mm2.empty:
                    _save_df(all_mm2, all_raw_root / 'cells_per_mm2_roi_level.csv')
                    _plot_all_populations_by_group(
                        all_mm2,
                        pop_col=pop_col,
                        group_col=group_col,
                        value_col='cells_per_mm2',
                        pop_order=pop_order,
                        group_order=group_order,
                        group_palette=group_palette,
                        ylabel='Cells per mm$^2$\n(ROI level)',
                        title=f"{pop_col}: all populations by {group_col} (ROI level)",
                        save_path=all_plot_root / f'cells_per_mm2_roi_level.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        base_figsize=all_pop_figsize,
                        width_scale=all_pop_width_scale,
                        y_scale_mode=mode_mm2_roi,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_roi_level',
                    )

            if case_col and allow_mm2 and has_roi_areas and not case_mm2.empty:
                all_mm2_case = case_mm2[case_mm2[group_col].isin(group_order)].copy()
                if not all_mm2_case.empty:
                    _save_df(all_mm2_case, all_raw_root / 'cells_per_mm2_case_average.csv')
                    _plot_all_populations_by_group(
                        all_mm2_case,
                        pop_col=pop_col,
                        group_col=group_col,
                        value_col='cells_per_mm2',
                        pop_order=pop_order,
                        group_order=group_order,
                        group_palette=group_palette,
                        ylabel='Cells per mm$^2$\n(Case average)',
                        title=f"{pop_col}: all populations by {group_col} (Case average)",
                        save_path=all_plot_root / f'cells_per_mm2_case_average.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        base_figsize=all_pop_figsize,
                        width_scale=all_pop_width_scale,
                        y_scale_mode=mode_mm2_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_case_average',
                    )

        if case_col:
            stats_prefix_base = f"{cleanstring(str(pop_col))}_{cleanstring(str(group_col))}"
            _run_mlm_stats_and_save(
                data=proportions,
                pop_col=pop_col,
                group_col=group_col,
                case_col=case_col,
                roi_col=roi_col,
                value_col='prop_cells',
                average_cases=False,
                stats_dir=pop_stats_root,
                out_prefix=f'{stats_prefix_base}_proportions',
            )
            _run_mlm_stats_and_save(
                data=proportions,
                pop_col=pop_col,
                group_col=group_col,
                case_col=case_col,
                roi_col=roi_col,
                value_col='prop_cells',
                average_cases=True,
                stats_dir=pop_stats_root,
                out_prefix=f'{stats_prefix_base}_proportions',
            )

            if allow_mm2 and has_roi_areas and not counts_mm2.empty:
                _run_mlm_stats_and_save(
                    data=counts_mm2,
                    pop_col=pop_col,
                    group_col=group_col,
                    case_col=case_col,
                    roi_col=roi_col,
                    value_col='cells_per_mm2',
                    average_cases=False,
                    stats_dir=pop_stats_root,
                    out_prefix=f'{stats_prefix_base}_cells_per_mm2',
                )
                _run_mlm_stats_and_save(
                    data=counts_mm2,
                    pop_col=pop_col,
                    group_col=group_col,
                    case_col=case_col,
                    roi_col=roi_col,
                    value_col='cells_per_mm2',
                    average_cases=True,
                    stats_dir=pop_stats_root,
                    out_prefix=f'{stats_prefix_base}_cells_per_mm2',
                )
        else:
            logging.info(
                "Population column '%s': case_obs not configured or missing, so MLM stats and case-average "
                "plots are skipped.",
                pop_col,
            )

    logging.info("Population abundance analysis scope '%s' completed. Outputs saved to: %s", scope_label, analysis_root)


def create_population_abundance_analysis(
    adata: ad.AnnData,
    population_columns,
    viz_config,
    general_config,
    qc_base: Path,
):
    """
    Create notebook-style per-population abundance plots and stats for one grouping variable.

    Grouping resolution order:
    1) visualization.groupby_obs / visualization.groupby_obs_groups
    2) general.groupby_obs / general.groupby_obs_primary_pairwise / general.groupby_obs_groups
    ROI and case identifiers are controlled by general.roi_obs and general.case_obs.
    If general.compartment_obs/general.compartment_obs_list are configured, the analysis is
    repeated for 'all' cells plus each configured compartment, with compartment-specific runs
    limited to proportion-based outputs.
    """
    group_col = coalesce_config_text(
        getattr(viz_config, 'groupby_obs', None),
        getattr(general_config, 'groupby_obs', None),
    )
    if not group_col:
        logging.warning("No grouping column configured (visualization.groupby_obs/general.groupby_obs); skipping abundance analysis.")
        return
    if group_col not in adata.obs.columns:
        logging.warning("Grouping column '%s' not found in adata.obs; skipping abundance analysis.", group_col)
        return

    roi_col = getattr(general_config, 'roi_obs', 'ROI')
    if roi_col not in adata.obs.columns:
        logging.warning("ROI column '%s' not found in adata.obs; skipping abundance analysis.", roi_col)
        return

    case_col = getattr(general_config, 'case_obs', None)
    if case_col is not None and case_col not in adata.obs.columns:
        logging.warning(
            "Case column '%s' not found in adata.obs; case-average plots/stats will be skipped.",
            case_col,
        )
        case_col = None

    configured_groups = coalesce_config_list(
        getattr(viz_config, 'groupby_obs_groups', None),
        getattr(general_config, 'groupby_obs_primary_pairwise', None),
        getattr(general_config, 'groupby_obs_groups', None),
    )
    base_group_order = _ordered_groups(adata.obs[group_col], configured_groups)
    if not base_group_order:
        logging.warning("No valid groups found for '%s'; skipping abundance analysis.", group_col)
        return

    logging.info(
        "Running abundance analysis for group '%s' (groups=%s) with ROI column '%s'%s.",
        group_col,
        base_group_order,
        roi_col,
        f" and case column '{case_col}'" if case_col else "",
    )

    analysis_base_root = qc_base / 'Population_Analysis_Figures'
    analysis_base_root.mkdir(parents=True, exist_ok=True)
    analysis_name = f"Abundance_by_{cleanstring(str(group_col))}"
    scopes = _resolve_population_abundance_scopes(adata, general_config)
    use_scope_folders = len(scopes) > 1
    roi_area_map = _build_roi_area_mm2_map(adata, general_config, roi_col=roi_col)

    for scope in scopes:
        if use_scope_folders:
            analysis_root = analysis_base_root / scope['folder_name'] / analysis_name
        else:
            analysis_root = analysis_base_root / analysis_name

        _run_population_abundance_analysis_scope(
            adata=adata,
            population_columns=population_columns,
            viz_config=viz_config,
            analysis_root=analysis_root,
            scope_label=str(scope['label']),
            scope_mask=scope['mask'],
            allow_mm2=bool(scope['allow_mm2']),
            group_col=group_col,
            base_group_order=base_group_order,
            roi_col=roi_col,
            case_col=case_col,
            roi_area_map=roi_area_map,
        )

    logging.info("Population abundance analysis completed. Outputs saved under: %s", analysis_base_root)
    return

    analysis_root = qc_base / 'Population_Analysis_Figures' / f"Abundance_by_{cleanstring(str(group_col))}"
    raw_root = analysis_root / 'Raw_Data'
    plot_root = analysis_root / 'Plots'
    stats_root = analysis_root / 'Stats'
    for out_dir in [analysis_root, raw_root, plot_root, stats_root]:
        out_dir.mkdir(parents=True, exist_ok=True)

    roi_area_map = _build_roi_area_mm2_map(adata, general_config, roi_col=roi_col)
    has_roi_areas = len(roi_area_map) > 0
    make_all_pop_plots = bool(getattr(viz_config, 'abundance_make_all_populations_plots', True))
    raw_all_pop_figsize = getattr(viz_config, 'abundance_all_populations_figsize', [4.0, 3.0])
    if (
        not isinstance(raw_all_pop_figsize, (list, tuple))
        or len(raw_all_pop_figsize) < 2
    ):
        all_pop_figsize = (4.0, 3.0)
    else:
        all_pop_figsize = (float(raw_all_pop_figsize[0]), float(raw_all_pop_figsize[1]))
    all_pop_width_scale = max(
        0.05,
        float(getattr(viz_config, 'abundance_all_populations_width_scale', 0.45)),
    )
    make_case_stacked_plots = bool(getattr(viz_config, 'abundance_make_case_stacked_plots', True))
    raw_case_stacked_figsize = getattr(viz_config, 'abundance_case_stacked_figsize', [6.0, 3.0])
    if (
        not isinstance(raw_case_stacked_figsize, (list, tuple))
        or len(raw_case_stacked_figsize) < 2
    ):
        case_stacked_figsize = (6.0, 3.0)
    else:
        case_stacked_figsize = (
            float(raw_case_stacked_figsize[0]),
            float(raw_case_stacked_figsize[1]),
        )
    case_stacked_width_scale = max(
        0.05,
        float(getattr(viz_config, 'abundance_case_stacked_width_scale', 0.30)),
    )
    order_cases_by_population = getattr(viz_config, 'abundance_order_cases_by_population', None)
    if order_cases_by_population is not None:
        order_cases_by_population = str(order_cases_by_population)

    abundance_scale_cfg = getattr(viz_config, 'abundance_barplot_y_scale', {'default': 'linear'})
    abundance_scale_intelligent_params = _resolve_intelligent_scale_params(
        getattr(viz_config, 'abundance_barplot_y_scale_intelligent_params', None)
    )
    mode_proportions_roi = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="proportions_roi_level",
    )
    mode_proportions_case = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="proportions_case_average",
    )
    mode_mm2_roi = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="cells_per_mm2_roi_level",
    )
    mode_mm2_case = _resolve_barplot_scale_mode(
        abundance_scale_cfg,
        analysis="abundance",
        metric="cells_per_mm2_case_average",
    )
    group_palette = _population_palette(adata, group_col, group_order)

    for pop_col in population_columns:
        if pop_col not in adata.obs.columns:
            logging.warning("Population column '%s' not found in adata.obs; skipping.", pop_col)
            continue

        pop_name = cleanstring(str(pop_col))
        pop_root = analysis_root / pop_name
        pop_raw_root = raw_root / pop_name
        pop_plot_root = plot_root / pop_name
        pop_stats_root = stats_root / pop_name
        for out_dir in [pop_root, pop_raw_root, pop_plot_root, pop_stats_root]:
            out_dir.mkdir(parents=True, exist_ok=True)

        required_cols = [pop_col, group_col, roi_col]
        if case_col:
            required_cols.append(case_col)

        obs = adata.obs[required_cols].copy()
        obs = obs.dropna(subset=required_cols)
        if obs.empty:
            logging.warning("No non-null rows available for population column '%s'; skipping.", pop_col)
            continue

        for col in required_cols:
            obs[col] = obs[col].astype(str)
        obs = obs[obs[group_col].isin(group_order)].copy()
        if obs.empty:
            logging.warning(
                "No rows in '%s' after filtering '%s' to groups %s; skipping.",
                pop_col,
                group_col,
                group_order,
            )
            continue

        _save_df(obs, pop_raw_root / 'cell_level_filtered.csv')

        pop_order = _population_order(adata, pop_col)
        if not pop_order:
            pop_order = sorted(obs[pop_col].astype(str).unique().tolist())
        palette = _population_palette(adata, pop_col, pop_order)

        count_group_cols = [pop_col, group_col]
        if case_col:
            count_group_cols.append(case_col)
        count_group_cols.append(roi_col)
        counts = (
            obs.groupby(count_group_cols, observed=True)
            .size()
            .reset_index(name='n_cells')
        )
        counts['n_cells'] = counts['n_cells'].astype(float)
        _save_df(counts, pop_raw_root / 'counts_by_population_group_roi.csv')

        total_group_cols = [group_col, roi_col]
        if case_col:
            total_group_cols.append(case_col)
        totals = (
            obs.groupby(total_group_cols, observed=True)
            .size()
            .reset_index(name='total_cells')
        )
        _save_df(totals, pop_raw_root / 'totals_by_group_roi.csv')

        # Proportions
        proportions = counts.merge(totals, on=total_group_cols, how='left')
        proportions['prop_cells'] = proportions['n_cells'] / proportions['total_cells']
        proportions = proportions.replace([np.inf, -np.inf], np.nan).dropna(subset=['prop_cells'])
        _save_df(proportions, pop_raw_root / 'proportions_roi_level.csv')

        # Cells/mm2
        counts_mm2 = counts.copy()
        if has_roi_areas:
            counts_mm2['area_mm2'] = counts_mm2[roi_col].map(roi_area_map)
            missing_area = int(counts_mm2['area_mm2'].isna().sum())
            if missing_area > 0:
                logging.warning(
                    "Population column '%s': %d rows missing ROI area in masks/sample table; "
                    "those rows are excluded from cells/mm2.",
                    pop_col,
                    missing_area,
                )
            counts_mm2 = counts_mm2.dropna(subset=['area_mm2']).copy()
            if not counts_mm2.empty:
                counts_mm2['cells_per_mm2'] = counts_mm2['n_cells'] / counts_mm2['area_mm2']
                counts_mm2 = counts_mm2.replace([np.inf, -np.inf], np.nan).dropna(subset=['cells_per_mm2'])
                _save_df(counts_mm2, pop_raw_root / 'cells_per_mm2_roi_level.csv')
        else:
            logging.warning(
                "Population column '%s': no ROI area map available; skipping cells/mm2 plots and stats.",
                pop_col,
            )

        # Optional case-averaged raw outputs
        case_proportions = pd.DataFrame()
        if case_col:
            case_proportions = (
                proportions.groupby([pop_col, group_col, case_col], observed=True, as_index=False)['prop_cells']
                .mean()
            )
            _save_df(case_proportions, pop_raw_root / 'proportions_case_average.csv')

        case_mm2 = pd.DataFrame()
        if case_col and has_roi_areas and not counts_mm2.empty:
            case_mm2 = (
                counts_mm2.groupby([pop_col, group_col, case_col], observed=True, as_index=False)['cells_per_mm2']
                .mean()
            )
            _save_df(case_mm2, pop_raw_root / 'cells_per_mm2_case_average.csv')

        # Case-level composition from direct case counts (no ROI pre-averaging),
        # used for stacked composition plots similar to CellCharter outputs.
        case_level_props = pd.DataFrame()
        if case_col:
            case_counts = (
                obs.groupby([case_col, pop_col], observed=True)
                .size()
                .reset_index(name='n_cells')
            )
            case_totals = case_counts.groupby(case_col, observed=True)['n_cells'].transform('sum')
            case_counts['case_proportion'] = case_counts['n_cells'] / case_totals.replace(0, np.nan)
            case_level_props = case_counts[[case_col, pop_col, 'case_proportion']].copy()
            case_level_props = (
                case_level_props
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=['case_proportion'])
            )

            case_group_nunique = (
                obs.groupby(case_col, observed=True)[group_col]
                .nunique(dropna=True)
            )
            ambiguous_cases = case_group_nunique[case_group_nunique > 1].index.tolist()
            if ambiguous_cases:
                logging.warning(
                    "Population '%s': some cases map to multiple '%s' values; using majority assignment for stacked plots.",
                    pop_col,
                    group_col,
                )
            case_group_map = (
                obs.groupby(case_col, observed=True)[group_col]
                .agg(_majority_label)
                .reset_index()
            )
            case_level_props = case_level_props.merge(case_group_map, on=case_col, how='left')
            _save_df(case_level_props, pop_raw_root / 'case_level_proportions.csv')

        if make_case_stacked_plots and case_col and not case_level_props.empty:
            stacked_plot_root = pop_plot_root / 'case_stacked_proportions'
            stacked_raw_root = pop_raw_root / 'case_stacked_proportions'
            stacked_plot_root.mkdir(parents=True, exist_ok=True)
            stacked_raw_root.mkdir(parents=True, exist_ok=True)

            _plot_case_stacked_proportions(
                case_level_props,
                case_col=case_col,
                pop_col=pop_col,
                value_col='case_proportion',
                pop_order=pop_order,
                palette=palette,
                title=f"Case-level {pop_col} proportions",
                save_path=stacked_plot_root / f'stacked_case_proportion_all.{viz_config.figure_format}',
                save_high_res=viz_config.save_high_res,
                base_figsize=case_stacked_figsize,
                width_scale=case_stacked_width_scale,
                order_by_population=order_cases_by_population,
                wide_csv_path=stacked_raw_root / 'stacked_case_proportion_all.csv',
            )

            available_groups = _ordered_groups(case_level_props[group_col], configured_groups)
            for group in available_groups:
                group_subset = case_level_props[
                    case_level_props[group_col].astype(str) == str(group)
                ].copy()
                if group_subset.empty:
                    continue
                group_safe = cleanstring(str(group))
                _plot_case_stacked_proportions(
                    group_subset,
                    case_col=case_col,
                    pop_col=pop_col,
                    value_col='case_proportion',
                    pop_order=pop_order,
                    palette=palette,
                    title=f"Case-level {pop_col} proportions ({group_col}={group})",
                    save_path=stacked_plot_root / f'stacked_case_proportion_{group_safe}.{viz_config.figure_format}',
                    save_high_res=viz_config.save_high_res,
                    base_figsize=case_stacked_figsize,
                    width_scale=case_stacked_width_scale,
                    order_by_population=order_cases_by_population,
                    wide_csv_path=stacked_raw_root / f'stacked_case_proportion_{group_safe}.csv',
                )

        # Per-population plots
        for pop in pop_order:
            pop_label = str(pop)
            pop_safe = cleanstring(pop_label)
            pop_color = palette.get(pop_label, '#4C72B0')

            pop_prop = proportions[(proportions[pop_col] == pop_label) & (proportions[group_col].isin(group_order))].copy()
            if not pop_prop.empty:
                _save_df(pop_prop, pop_raw_root / 'per_population' / 'proportions_roi_level' / f'{pop_safe}.csv')
                _plot_population_bar(
                    data=pop_prop,
                    x_col=group_col,
                    y_col='prop_cells',
                    order=group_order,
                    color=pop_color,
                    ylabel='Proportion of cells\n(ROI level)',
                    title=pop_label,
                    save_path=pop_plot_root / 'proportions_roi_level' / f'{pop_safe}.{viz_config.figure_format}',
                    save_high_res=viz_config.save_high_res,
                    y_scale_mode=mode_proportions_roi,
                    y_scale_intelligent_params=abundance_scale_intelligent_params,
                    scale_metric_name='proportions_roi_level',
                )

            if case_col and not case_proportions.empty:
                pop_prop_case = case_proportions[
                    (case_proportions[pop_col] == pop_label) & (case_proportions[group_col].isin(group_order))
                ].copy()
                if not pop_prop_case.empty:
                    _save_df(pop_prop_case, pop_raw_root / 'per_population' / 'proportions_case_average' / f'{pop_safe}.csv')
                    _plot_population_bar(
                        data=pop_prop_case,
                        x_col=group_col,
                        y_col='prop_cells',
                        order=group_order,
                        color=pop_color,
                        ylabel='Proportion of cells\n(Case average)',
                        title=pop_label,
                        save_path=pop_plot_root / 'proportions_case_average' / f'{pop_safe}.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        y_scale_mode=mode_proportions_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='proportions_case_average',
                    )

            if has_roi_areas and not counts_mm2.empty:
                pop_mm2 = counts_mm2[
                    (counts_mm2[pop_col] == pop_label) & (counts_mm2[group_col].isin(group_order))
                ].copy()
                if not pop_mm2.empty:
                    _save_df(pop_mm2, pop_raw_root / 'per_population' / 'cells_per_mm2_roi_level' / f'{pop_safe}.csv')
                    _plot_population_bar(
                        data=pop_mm2,
                        x_col=group_col,
                        y_col='cells_per_mm2',
                        order=group_order,
                        color=pop_color,
                        ylabel='Cells per mm$^2$\n(ROI level)',
                        title=pop_label,
                        save_path=pop_plot_root / 'cells_per_mm2_roi_level' / f'{pop_safe}.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        y_scale_mode=mode_mm2_roi,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_roi_level',
                    )

            if case_col and has_roi_areas and not case_mm2.empty:
                pop_mm2_case = case_mm2[
                    (case_mm2[pop_col] == pop_label) & (case_mm2[group_col].isin(group_order))
                ].copy()
                if not pop_mm2_case.empty:
                    _save_df(pop_mm2_case, pop_raw_root / 'per_population' / 'cells_per_mm2_case_average' / f'{pop_safe}.csv')
                    _plot_population_bar(
                        data=pop_mm2_case,
                        x_col=group_col,
                        y_col='cells_per_mm2',
                        order=group_order,
                        color=pop_color,
                        ylabel='Cells per mm$^2$\n(Case average)',
                        title=pop_label,
                        save_path=pop_plot_root / 'cells_per_mm2_case_average' / f'{pop_safe}.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        y_scale_mode=mode_mm2_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_case_average',
                    )

        # Combined all-populations plots (x=population, hue=group)
        if make_all_pop_plots:
            all_raw_root = pop_raw_root / 'all_populations'
            all_plot_root = pop_plot_root / 'all_populations'
            all_raw_root.mkdir(parents=True, exist_ok=True)
            all_plot_root.mkdir(parents=True, exist_ok=True)

            all_prop = proportions[proportions[group_col].isin(group_order)].copy()
            if not all_prop.empty:
                _save_df(all_prop, all_raw_root / 'proportions_roi_level.csv')
                _plot_all_populations_by_group(
                    all_prop,
                    pop_col=pop_col,
                    group_col=group_col,
                    value_col='prop_cells',
                    pop_order=pop_order,
                    group_order=group_order,
                    group_palette=group_palette,
                    ylabel='Proportion of cells\n(ROI level)',
                    title=f"{pop_col}: all populations by {group_col} (ROI level)",
                    save_path=all_plot_root / f'proportions_roi_level.{viz_config.figure_format}',
                    save_high_res=viz_config.save_high_res,
                    base_figsize=all_pop_figsize,
                    width_scale=all_pop_width_scale,
                    y_scale_mode=mode_proportions_roi,
                    y_scale_intelligent_params=abundance_scale_intelligent_params,
                    scale_metric_name='proportions_roi_level',
                )

            if case_col and not case_proportions.empty:
                all_prop_case = case_proportions[case_proportions[group_col].isin(group_order)].copy()
                if not all_prop_case.empty:
                    _save_df(all_prop_case, all_raw_root / 'proportions_case_average.csv')
                    _plot_all_populations_by_group(
                        all_prop_case,
                        pop_col=pop_col,
                        group_col=group_col,
                        value_col='prop_cells',
                        pop_order=pop_order,
                        group_order=group_order,
                        group_palette=group_palette,
                        ylabel='Proportion of cells\n(Case average)',
                        title=f"{pop_col}: all populations by {group_col} (Case average)",
                        save_path=all_plot_root / f'proportions_case_average.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        base_figsize=all_pop_figsize,
                        width_scale=all_pop_width_scale,
                        y_scale_mode=mode_proportions_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='proportions_case_average',
                    )

            if has_roi_areas and not counts_mm2.empty:
                all_mm2 = counts_mm2[counts_mm2[group_col].isin(group_order)].copy()
                if not all_mm2.empty:
                    _save_df(all_mm2, all_raw_root / 'cells_per_mm2_roi_level.csv')
                    _plot_all_populations_by_group(
                        all_mm2,
                        pop_col=pop_col,
                        group_col=group_col,
                        value_col='cells_per_mm2',
                        pop_order=pop_order,
                        group_order=group_order,
                        group_palette=group_palette,
                        ylabel='Cells per mm$^2$\n(ROI level)',
                        title=f"{pop_col}: all populations by {group_col} (ROI level)",
                        save_path=all_plot_root / f'cells_per_mm2_roi_level.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        base_figsize=all_pop_figsize,
                        width_scale=all_pop_width_scale,
                        y_scale_mode=mode_mm2_roi,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_roi_level',
                    )

            if case_col and has_roi_areas and not case_mm2.empty:
                all_mm2_case = case_mm2[case_mm2[group_col].isin(group_order)].copy()
                if not all_mm2_case.empty:
                    _save_df(all_mm2_case, all_raw_root / 'cells_per_mm2_case_average.csv')
                    _plot_all_populations_by_group(
                        all_mm2_case,
                        pop_col=pop_col,
                        group_col=group_col,
                        value_col='cells_per_mm2',
                        pop_order=pop_order,
                        group_order=group_order,
                        group_palette=group_palette,
                        ylabel='Cells per mm$^2$\n(Case average)',
                        title=f"{pop_col}: all populations by {group_col} (Case average)",
                        save_path=all_plot_root / f'cells_per_mm2_case_average.{viz_config.figure_format}',
                        save_high_res=viz_config.save_high_res,
                        base_figsize=all_pop_figsize,
                        width_scale=all_pop_width_scale,
                        y_scale_mode=mode_mm2_case,
                        y_scale_intelligent_params=abundance_scale_intelligent_params,
                        scale_metric_name='cells_per_mm2_case_average',
                    )

        # Stats (requires case column and exactly two groups for mlm_stats)
        if case_col:
            stats_prefix_base = f"{cleanstring(str(pop_col))}_{cleanstring(str(group_col))}"
            _run_mlm_stats_and_save(
                data=proportions,
                pop_col=pop_col,
                group_col=group_col,
                case_col=case_col,
                roi_col=roi_col,
                value_col='prop_cells',
                average_cases=False,
                stats_dir=pop_stats_root,
                out_prefix=f'{stats_prefix_base}_proportions',
            )
            _run_mlm_stats_and_save(
                data=proportions,
                pop_col=pop_col,
                group_col=group_col,
                case_col=case_col,
                roi_col=roi_col,
                value_col='prop_cells',
                average_cases=True,
                stats_dir=pop_stats_root,
                out_prefix=f'{stats_prefix_base}_proportions',
            )

            if has_roi_areas and not counts_mm2.empty:
                _run_mlm_stats_and_save(
                    data=counts_mm2,
                    pop_col=pop_col,
                    group_col=group_col,
                    case_col=case_col,
                    roi_col=roi_col,
                    value_col='cells_per_mm2',
                    average_cases=False,
                    stats_dir=pop_stats_root,
                    out_prefix=f'{stats_prefix_base}_cells_per_mm2',
                )
                _run_mlm_stats_and_save(
                    data=counts_mm2,
                    pop_col=pop_col,
                    group_col=group_col,
                    case_col=case_col,
                    roi_col=roi_col,
                    value_col='cells_per_mm2',
                    average_cases=True,
                    stats_dir=pop_stats_root,
                    out_prefix=f'{stats_prefix_base}_cells_per_mm2',
                )
        else:
            logging.info(
                "Population column '%s': case_obs not configured or missing, so MLM stats and case-average "
                "plots are skipped.",
                pop_col,
            )

    logging.info("Population abundance analysis completed. Outputs saved to: %s", analysis_root)


def create_population_analysis(
    adata,
    population_columns,
    metadata_columns,
    qc_base,
    viz_config,
    general_config,
    max_categories=20,
):
    """
    Dispatch population analysis:
    - If a grouping column is configured in visualization/group general settings: run abundance-focused analysis.
    - Otherwise: run legacy metadata-by-population analysis.
    """
    effective_groupby = coalesce_config_text(
        getattr(viz_config, 'groupby_obs', None),
        getattr(general_config, 'groupby_obs', None),
    )
    if effective_groupby:
        create_population_abundance_analysis(
            adata=adata,
            population_columns=population_columns,
            viz_config=viz_config,
            general_config=general_config,
            qc_base=qc_base,
        )
    else:
        create_population_metadata_analysis(
            adata=adata,
            population_columns=population_columns,
            metadata_columns=metadata_columns,
            qc_base=qc_base,
            max_categories=max_categories,
        )


if __name__ == "__main__":
    # Set matplotlib to non-interactive backend for batch processing
    matplotlib.use("Agg")
    
    # Set up logging
    pipeline_stage = 'Visualizations'
    config = process_config_with_overrides()
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**filter_config_for_dataclass(config.get('general', {}), GeneralConfig))
    viz_config = VisualizationConfig(**filter_config_for_dataclass(config.get('visualization', {}), VisualizationConfig))
    segmentation_config = SegmentationConfig(**filter_config_for_dataclass(config.get('segmentation', {}), SegmentationConfig))

    # Promote shared General obs defaults into visualization config when script overrides are not set.
    if viz_config.population_columns is None:
        if getattr(general_config, 'population_obs_all', None) is not None:
            viz_config.population_columns = [str(c) for c in general_config.population_obs_all]
        elif getattr(general_config, 'population_obs_primary', None):
            viz_config.population_columns = [str(general_config.population_obs_primary)]
    if viz_config.metadata_columns is None and getattr(general_config, 'metadata_obs', None) is not None:
        viz_config.metadata_columns = [str(c) for c in general_config.metadata_obs]
    if viz_config.groupby_obs is None:
        viz_config.groupby_obs = coalesce_config_text(getattr(general_config, 'groupby_obs', None))
    if viz_config.groupby_obs_groups is None:
        viz_config.groupby_obs_groups = coalesce_config_list(
            getattr(general_config, 'groupby_obs_primary_pairwise', None),
            getattr(general_config, 'groupby_obs_groups', None),
        )

    adata, adata_path, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=viz_config,
        override_path=viz_config.input_adata_path,
    )
    if skip_stage:
        logging.info("Skipping visualization stage based on AnnData stage policy.")
        exit(0)
    if adata is None:
        raise FileNotFoundError(f"AnnData could not be loaded for visualization stage: {adata_path}")
    logging.info('AnnData loaded successfully.')

    # Set up QC output folder
    qc_base = Path(general_config.qc_folder) / 'BasicProcess_QC'
    
    # Set up output directories
    qc_umap_dir = qc_base / 'UMAPs'
    qc_matrix_dir = qc_base / 'Matrixplots'
    qc_legend_dir = qc_base / 'Color_legends'
    qc_pop_dir = qc_base / 'Population_images'
    
    for p in [qc_umap_dir, qc_matrix_dir, qc_legend_dir, qc_pop_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Resolve population columns (visualization/general resolved config > auto-detect)
    if viz_config.population_columns is not None:
        population_columns = [str(c) for c in viz_config.population_columns]
        logging.info(f"Using population columns from resolved config: {population_columns}")
    else:
        population_columns = find_population_columns(adata, max_categories=viz_config.max_categories)

    # Resolve metadata columns (visualization/general resolved config > auto-detect)
    if viz_config.metadata_columns is not None:
        metadata_columns = [str(c) for c in viz_config.metadata_columns]
        logging.info(f"Using metadata columns from resolved config: {metadata_columns}")
    else:
        metadata_columns = find_metadata_columns(adata, population_columns, general_config.metadata_folder, max_categories=viz_config.max_categories)
    
    logging.info("Starting comprehensive visualization suite...")
    
    # Create UMAPs for populations, metadata, and markers
    if viz_config.create_umaps:
        logging.info("Creating UMAP visualizations...")
        # Population UMAPs
        create_categorical_umaps(adata, population_columns, qc_umap_dir, qc_legend_dir, viz_config, "population")
        
        # Metadata UMAPs (optional)
        if viz_config.include_metadata_umaps:
            create_categorical_umaps(adata, metadata_columns, qc_umap_dir, qc_legend_dir, viz_config, "metadata")
        
        # Marker UMAPs (optional)
        if viz_config.include_marker_umaps:
            create_marker_umaps(adata, qc_umap_dir, viz_config)
    
    # Create matrix plots for populations and metadata
    if viz_config.create_matrix_plots:
        logging.info("Creating MatrixPlot visualizations...")
        # Get remove_and_store_markers list from segmentation config
        remove_markers_list = segmentation_config.remove_and_store_markers if segmentation_config.remove_and_store_markers else None
        
        # Population matrix plots
        create_categorical_matrix_plots(adata, population_columns, qc_matrix_dir, viz_config, "population", remove_markers_list)
        
        # Metadata matrix plots (optional)
        if viz_config.include_metadata_matrix_plots:
            create_categorical_matrix_plots(adata, metadata_columns, qc_matrix_dir, viz_config, "metadata", remove_markers_list)
    
    # Create tissue overlays for populations (metadata overlays would be similar but populations are more relevant)
    if viz_config.create_tissue_overlays:
        logging.info("Creating tissue overlay visualizations...")
        create_population_tissue_overlays(adata, population_columns, qc_pop_dir, general_config)
    
    # Create population analysis across metadata
    if viz_config.create_population_analysis:
        logging.info("Creating population analysis...")
        create_population_analysis(
            adata=adata,
            population_columns=population_columns,
            metadata_columns=metadata_columns,
            qc_base=qc_base,
            viz_config=viz_config,
            general_config=general_config,
            max_categories=min(20, viz_config.max_categories),
        )
    
    # Create backgating assessment for populations
    if viz_config.create_backgating:
        logging.info("Creating backgating assessment...")
        create_backgating_assessment(adata, population_columns, viz_config, general_config, qc_base)
    
    # Create color legends for all categorical columns (independent of other visualizations)
    if viz_config.create_color_legends:
        logging.info("Creating color legends...")
        # Create legends for population columns
        for pop_col in population_columns:
            if pop_col in adata.obs.columns:
                try:
                    create_color_legend(adata, pop_col, 
                                      qc_legend_dir / f'{pop_col}_legend.{viz_config.figure_format}',
                                      title=f"Population: {pop_col}")
                except Exception as e:
                    log_detailed_error(e, f"creating color legend for population column '{pop_col}'")
        
        # Create legends for metadata columns  
        for meta_col in metadata_columns:
            if meta_col in adata.obs.columns:
                try:
                    create_color_legend(adata, meta_col,
                                      qc_legend_dir / f'{meta_col}_legend.{viz_config.figure_format}',
                                      title=f"Metadata: {meta_col}")
                except Exception as e:
                    log_detailed_error(e, f"creating color legend for metadata column '{meta_col}'")
    
    save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=viz_config,
        override_path=str(adata_path),
        extra_details={"qc_output_root": str(qc_base)},
    )
    logging.info('Visualization pipeline completed successfully!')
