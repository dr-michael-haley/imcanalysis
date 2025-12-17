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

# Third-party library imports
import scanpy as sc
import anndata as ad
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
                    
                except Exception as e:
                    log_detailed_error(e, f"creating UMAP for {category_type} column '{cat_col}'")
            else:
                logging.warning(f'{cat_col} not found in adata.obs; skipping UMAP.')
    except Exception as e:
        log_detailed_error(e, f"{category_type.title()} UMAP visualization step")


def create_categorical_matrix_plots(adata, categorical_columns, qc_matrix_dir, viz_config, category_type="categorical"):
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
    """
    try:
        markers_to_plot = adata.var_names.tolist()
        for cat_col in categorical_columns:
            if cat_col in adata.obs.columns:
                logging.info(f'Creating MatrixPlots for {category_type} column: {cat_col}')
                try:
                    # Pre-compute dendrogram to avoid warning
                    sc.tl.dendrogram(adata, groupby=cat_col)
                    
                    # 1. Create standard-scaled matrixplot
                    logging.info(f'  Creating standard-scaled MatrixPlot for {cat_col}')
                    matrixplot_scaled = sc.pl.matrixplot(
                        adata,
                        var_names=markers_to_plot,
                        groupby=cat_col,
                        standard_scale='var',
                        dendrogram=True,
                        show=False,
                        return_fig=True
                    )
                    fig_path_scaled = qc_matrix_dir / f'Matrixplot_{cat_col}_scaled.{viz_config.figure_format}'
                    matrixplot_scaled.savefig(fig_path_scaled, bbox_inches='tight', dpi=300 if viz_config.save_high_res else 150)
                    plt.close()
                    logging.info(f'  Standard-scaled MatrixPlot saved to {fig_path_scaled}')
                    
                    # 2. Create non-scaled matrixplot with vmax
                    logging.info(f'  Creating vmax-capped MatrixPlot for {cat_col} (vmax={viz_config.matrixplot_vmax})')
                    matrixplot_vmax = sc.pl.matrixplot(
                        adata,
                        var_names=markers_to_plot,
                        groupby=cat_col,
                        standard_scale=None,
                        dendrogram=True,
                        vmax=viz_config.matrixplot_vmax,
                        show=False,
                        return_fig=True
                    )
                    fig_path_vmax = qc_matrix_dir / f'Matrixplot_{cat_col}_vmax.{viz_config.figure_format}'
                    matrixplot_vmax.savefig(fig_path_vmax, bbox_inches='tight', dpi=300 if viz_config.save_high_res else 150)
                    plt.close()
                    logging.info(f'  Vmax-capped MatrixPlot saved to {fig_path_vmax}')
                    
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
            
        if 'ROI' not in adata.obs.columns:
            logging.warning('ROI column not found in adata.obs; skipping tissue visualization.')
            return
            
        rois = sorted(adata.obs['ROI'].astype(str).unique().tolist())
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
                        roi_obs='ROI',
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
                    logging.info(f"Backgating config - use_differential_expression: {use_de}, mode: {mode}")
                    logging.info(f"Specify overrides - red: {viz_config.backgating_specify_red}, "
                                f"green: {viz_config.backgating_specify_green}, blue: {viz_config.backgating_specify_blue}")
                    
                    sbt_backgating.backgating_assessment(
                        adata=adata,
                        image_folder=image_folder,
                        pop_obs=pop_col,
                        mean_expression_file=f'markers_mean_expression_{pop_col}.csv',
                        backgating_settings_file=f'backgating_settings_{pop_col}.csv',
                        pops_list=None,  # Use all populations
                        cells_per_group=viz_config.backgating_cells_per_group,
                        radius=viz_config.backgating_radius,
                        roi_obs='ROI',
                        x_loc_obs='X_loc',
                        y_loc_obs='Y_loc',
                        cell_index_obs='Master_Index',
                        object_index_obs='ObjectNumber',
                        # Mask parameters
                        use_masks=viz_config.backgating_use_masks,
                        mask_folder=viz_config.backgating_mask_folder,
                        exclude_rois_without_mask=True,
                        # Output settings
                        output_folder=str(backgating_output),
                        overview_images=True,
                        # Intensity scaling
                        minimum=viz_config.backgating_minimum,
                        max_quantile=viz_config.backgating_max_quantile,
                        # Population overview setttings
                        population_overlay_outline_width=viz_config.backgating_population_overlay_outline_width,
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


def create_population_analysis(adata, population_columns, metadata_columns, qc_base, max_categories=20):
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


if __name__ == "__main__":
    # Set matplotlib to non-interactive backend for batch processing
    matplotlib.use("Agg")
    
    # Set up logging
    pipeline_stage = 'Visualizations'
    config = process_config_with_overrides()
    setup_logging(config.get('logging', {}), pipeline_stage)

    # Get parameters from config
    general_config = GeneralConfig(**filter_config_for_dataclass(config.get('general', {}), GeneralConfig))
    process_config = BasicProcessConfig(**filter_config_for_dataclass(config.get('process', {}), BasicProcessConfig))
    viz_config = VisualizationConfig(**filter_config_for_dataclass(config.get('visualization', {}), VisualizationConfig))

    # Determine which AnnData to load
    adata_path = viz_config.input_adata_path if viz_config.input_adata_path is not None else process_config.output_adata_path
    
    # Load processed AnnData
    logging.info(f'Loading processed AnnData from {adata_path}.')
    adata = ad.read_h5ad(adata_path)
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

    # Find all population and metadata columns intelligently
    if viz_config.population_columns is not None:
        population_columns = viz_config.population_columns
        logging.info(f"Using population columns from config: {population_columns}")
    else:
        population_columns = find_population_columns(adata, max_categories=viz_config.max_categories)
    
    if viz_config.metadata_columns is not None:
        metadata_columns = viz_config.metadata_columns
        logging.info(f"Using metadata columns from config: {metadata_columns}")
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
        # Population matrix plots
        create_categorical_matrix_plots(adata, population_columns, qc_matrix_dir, viz_config, "population")
        
        # Metadata matrix plots (optional)
        if viz_config.include_metadata_matrix_plots:
            create_categorical_matrix_plots(adata, metadata_columns, qc_matrix_dir, viz_config, "metadata")
    
    # Create tissue overlays for populations (metadata overlays would be similar but populations are more relevant)
    if viz_config.create_tissue_overlays:
        logging.info("Creating tissue overlay visualizations...")
        create_population_tissue_overlays(adata, population_columns, qc_pop_dir, general_config)
    
    # Create population analysis across metadata
    if viz_config.create_population_analysis:
        logging.info("Creating population analysis...")
        create_population_analysis(adata, population_columns, metadata_columns, qc_base, max_categories=min(20, viz_config.max_categories))
    
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
    
    logging.info('Visualization pipeline completed successfully!')