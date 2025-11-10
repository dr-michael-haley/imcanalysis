import os
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import seaborn as sb
import skimage.io as io
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
from scipy import stats
from shapely.geometry import MultiPoint, Point, Polygon
from skimage.util import map_array
from skimage.transform import resize
import sklearn as sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
import umap
from scipy.spatial import Voronoi
import seaborn as sns
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display, HTML

from .image_analysis import save_labelled_image, save_labelled_image_as_svg, map_pixel_values_to_colors
from .utils import (_cleanstring, _save, _check_input_type, _to_list, subset,
                    adlog, print_full, compare_lists)


def _count_summary(data: ad.AnnData | pd.DataFrame | str,
                 pop_col: str = None,
                 levels: list = ['ROI'],
                 mean_over: list = ['ROI'],
                 crosstab_normalize: bool = False,
                 mode: str = 'population_counts') -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Summarize data into long and wide formats for plotting.

    Parameters
    ----------
    data : AnnData, DataFrame, or str
        The input data.
    pop_col : str, optional
        Column identifying a population.
    levels : list, optional
        Levels at which the data is structured.
    mean_over : list, optional
        Levels at which to calculate a mean.
    crosstab_normalize : bool, optional
        Whether, and how, to normalize crosstab results.
    mode : str, optional
        'population_counts' for counts, 'numeric' for numeric data summarization.

    Returns
    -------
    tuple
        Wide and long form data.
    '''
    from pandas.api.types import is_numeric_dtype

    data = _check_input_type(data)
    levels = _to_list(levels)
    mean_over = _to_list(mean_over)
    wide, long = None, None

    if mean_over:
        assert all([x in levels for x in mean_over]), 'Observation to mean over should also be in levels list'

    if mode == 'numeric':
        if pop_col:
            levels.append(pop_col)
            mean_over.append(pop_col)
        long = data.groupby(levels, observed=True).mean(numeric_only=True)
        if mean_over:
            long = long.groupby(mean_over, observed=True).mean(numeric_only=True)
            levels = mean_over
        if pop_col:
            levels.remove(pop_col)
            wide = pd.pivot(data=long.reset_index(), index=levels, columns=pop_col)
    elif mode == 'population_counts':
        assert not is_numeric_dtype(data[pop_col]), 'Column is numeric, cannot calculate population-level counts'
        crosstab_df = pd.crosstab([data[x] for x in levels], columns=data[pop_col], normalize=crosstab_normalize)
        wide = crosstab_df
        long = crosstab_df.reset_index().melt(id_vars=levels)
        if mean_over:
            wide = wide.groupby(mean_over).mean(numeric_only=True)
        long = long.groupby(mean_over + [pop_col], observed=True).mean(numeric_only=True).reset_index()

    return wide, long

def bargraph(data: ad.AnnData | pd.DataFrame | str,
             pop_col: str = None,
             value_col: str = None,
             hue: str = None,
             hue_order: list = None,
             specify_populations: list = [],
             levels: list = None,
             mean_over: list = None,
             confidence_interval: int = 68,
             figsize: tuple = (5, 5),
             crosstab_normalize: bool = False,
             cells_per_mm: bool = False,
             palette: dict = None,
             hide_grid: bool = False,
             legend: bool = True,
             save_data: bool = True,
             save_figure: bool = False,
             return_data: bool = True,
             rotate_x_labels: int = 90,
             case_col_name: str = 'Case',
             ROI_col_name: str = 'ROI') -> pd.DataFrame | plt.Figure:
    '''
    Plot bar graphs for population abundances or measured values.

    Parameters
    ----------
    data : AnnData, DataFrame, or str
        The input data.
    pop_col : str, optional
        Column identifying a population.
    value_col : str, optional
        Column identifying the measured values.
    hue : str, optional
        Column to subgroup the graph.
    hue_order : list, optional
        Order of hues.
    specify_populations : list, optional
        List of specific populations to plot.
    levels : list, optional
        Levels at which the data is structured.
    mean_over : list, optional
        Levels at which to calculate a mean.
    confidence_interval : int, optional
        Confidence interval for error bars.
    figsize : tuple, optional
        Size of the figure.
    crosstab_normalize : bool, optional
        Whether, and how, to normalize crosstab results.
    cells_per_mm : bool, optional
        Normalize values by mm².
    palette : dict, optional
        Color palette.
    hide_grid : bool, optional
        Whether to hide the grid.
    legend : bool, optional
        Whether to display the legend.
    save_data : bool, optional
        Whether to save the raw data.
    save_figure : bool, optional
        Whether to save the figure.
    return_data : bool, optional
        Whether to return the data.
    rotate_x_labels : int, optional
        Angle of x labels.
    case_col_name : str, optional
        Name of the case column.
    ROI_col_name : str, optional
        Name of the ROI column.

    Returns
    -------
    DataFrame or Figure
        The data used to create the figure or the figure itself.
    '''
    from pandas.api.types import is_numeric_dtype

    data = _check_input_type(data)
    if not levels and not mean_over:
        assert ROI_col_name in data.columns, f'{ROI_col_name} column not found in data'
        if case_col_name in data.columns:
            levels = ['Case', 'ROI']
            mean_over = ['Case', 'ROI']
        else:
            levels = ['ROI']
            mean_over = ['ROI']
    levels = _to_list(levels)
    mean_over = _to_list(mean_over)
    specify_populations = _to_list(specify_populations)

    if specify_populations:
        data = data[data[pop_col].isin(specify_populations)]

    if hue and hue not in levels:
        levels.append(hue)
    if hue and hue not in mean_over:
        mean_over.append(hue)

    if not palette:
        if hue:
            try:
                palette = adata.uns[f'{hue}_colormap']
            except:
                pass
        else:
            try:
                palette = adata.uns[f'{pop_col}_colormap']
            except:
                pass

    if pop_col and value_col:
        _, long_form_data = _count_summary(data, pop_col, levels, mean_over, crosstab_normalize, 'numeric')
        plot_data = long_form_data.reset_index()
        y_plot, x_plot = value_col, pop_col
    elif not pop_col and value_col:
        _, long_form_data = _count_summary(data, pop_col, levels, mean_over, crosstab_normalize, 'numeric')
        plot_data = long_form_data.reset_index()
        y_plot, x_plot = value_col, levels[-1]
    elif pop_col and not value_col:
        _, long_form_data = _count_summary(data, pop_col, levels, mean_over, crosstab_normalize)
        long_form_data.rename(columns={'value': 'Cells'}, inplace=True)
        plot_data = long_form_data
        y_plot, x_plot = 'Cells', pop_col

    if cells_per_mm:
        try:
            size_dict = adata.uns['sample']['mm2'].to_dict()
            plot_data['mm2'] = plot_data['ROI'].map(size_dict)
            new_y = f'{y_plot} per mm²'
            plot_data[new_y] = plot_data[y_plot] / plot_data['mm2']
            y_plot = new_y
        except:
            pass

    fig, ax = plt.subplots(figsize=figsize)
    sb.barplot(data=plot_data, y=y_plot, x=x_plot, hue=hue, hue_order=hue_order, palette=palette, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_x_labels)

    if hide_grid:
        ax.grid(False)
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1))
    else:
        try:
            ax.get_legend().remove()
        except:
            pass

    filename = f'Bargraph_{_cleanstring(pop_col)}_{_cleanstring(levels)}_{_cleanstring(mean_over)}'
    if save_data:
        file_path = _save(('Figures', 'Barcharts', 'Raw'), f'{filename}.csv')
        long_form_data.to_csv(file_path)
    if save_figure:
        for ext in ['.png', '.svg']:
            file_path = _save(('Figures', 'Barcharts'), f'{filename}{ext}')
            fig.savefig(file_path, bbox_inches='tight', dpi=300)

    if return_data:
        return plot_data
    return fig

def mlm_stats(data: pd.DataFrame, 
              pop_col: str, 
              group_col: str, 
              case_col: str = 'Case', 
              value_col: str = 'Cells', 
              roi_col: str = 'ROI', 
              method: str = 'holm-sidak', 
              average_cases: bool = False, 
              show_t_values: bool = False, 
              run_t_tests: bool = True) -> pd.DataFrame:
    """
    Conduct mixed linear model analysis and optional t-test on data.

    Parameters
    ----------
    data : pd.DataFrame
        The input data.
    pop_col : str
        Column identifying different populations.
    group_col : str
        Column identifying 2 groups.
    case_col : str, optional
        Column identifying cases.
    value_col : str, optional
        Column containing the values to be analyzed.
    roi_col : str, optional
        Column identifying ROIs.
    method : str, optional
        Method to correct for multiple comparisons.
    average_cases : bool, optional
        Average results over cases before performing t-test.
    show_t_values : bool, optional
        Include t-values in the output.
    run_t_tests : bool, optional
        Run t-tests.

    Returns
    -------
    pd.DataFrame
        P-values of the mixed linear model analysis and t-tests.
    """
    assert set([pop_col, group_col, case_col, value_col, roi_col]).issubset(data.columns), "Some required columns are missing from the input DataFrame."
    assert data[group_col].nunique() == 2, "'group_col' must identify exactly two groups."

    data[pop_col] = [_cleanstring(x) for x in data[pop_col]]
    pop_list = data[pop_col].unique().tolist()
    results = []

    for i in pop_list:
        subset = data[data[pop_col] == i]
        formula = f"{value_col} ~ {group_col}"
        roi_counts = subset.groupby(case_col)[roi_col].nunique()
        case_roi_counts = subset.groupby([case_col, roi_col], observed=True).size()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if (case_roi_counts > 1).any():
                vc_formula = {roi_col: f'0 + C({roi_col})'}
                re_formula = '1'
                md = smf.mixedlm(formula=formula, data=subset, groups=subset[case_col], vc_formula=vc_formula, re_formula=re_formula)
            else:
                md = smf.mixedlm(formula=formula, data=subset, groups=subset[case_col])
            mdf = md.fit()
        warning_messages = [str(warn.message) for warn in w]
        result = {pop_col: i, 'mlm_p_value': mdf.pvalues[1], 'mlm_warnings': str(warning_messages)}
        if run_t_tests:
            if average_cases:
                subset = subset.groupby([group_col, case_col], observed=True).mean(numeric_only=True).reset_index()
            group1 = subset[subset[group_col] == subset[group_col].unique()[0]][value_col]
            group2 = subset[subset[group_col] == subset[group_col].unique()[1]][value_col]
            t_stat, t_pval = stats.ttest_ind(group1, group2)
            u_stat, mwu_pval = stats.mannwhitneyu(group1, group2)
            result['t_test_p_value'] = t_pval
            result['mannwhitneyu_p_value'] = mwu_pval
            if show_t_values:
                result['t_value'] = t_stat
                result['u_value'] = u_stat
        results.append(result)

    results_df = pd.DataFrame(results)
    reject_mlm, pvals_corrected_mlm, _, _ = multipletests(results_df['mlm_p_value'], method=method)
    results_df['mlm_p_value_corrected'] = pvals_corrected_mlm

    if 't_test_p_value' in results_df.columns:
        _, pvals_corrected_ttest, _, _ = multipletests(results_df['t_test_p_value'], method=method)
        results_df['t_test_p_value_corrected'] = pvals_corrected_ttest
        _, pvals_corrected_mwu, _, _ = multipletests(results_df['mannwhitneyu_p_value'], method=method)
        results_df['mannwhitneyu_p_value_corrected'] = pvals_corrected_mwu

    if average_cases:
        results_df = results_df.drop(columns=['mlm_p_value', 'mlm_warnings', 'mlm_p_value_corrected'])

    return results_df

def cellabundance_UMAP(adata: ad.AnnData, 
                       ROI_id: str, 
                       population: str, 
                       colour_by: str = None, 
                       annotate: bool = True, 
                       save: str = None, 
                       normalize: bool = False, 
                       dim_red: str = 'UMAP', 
                       ax: plt.Axes = None, 
                       cmap: str = 'tab20', 
                       figsize: tuple = (3, 3), 
                       point_size: int = 50) -> plt.Figure | plt.Axes:
    """
    Visualize cell population abundance within ROIs using UMAP, PCA, or tSNE with Seaborn's scatterplot.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing single-cell data.
    ROI_id : str
        Column in adata.obs that contains ROI identifiers.
    population : str
        Column in adata.obs that contains cell population identifiers.
    colour_by : str, optional
        Column to color points by. If None, points are colored by ROI_id.
    annotate : bool, optional
        Annotate points with ROI identifiers.
    save : str, optional
        File path to save the plot.
    normalize : bool, optional
        Normalize cell counts across ROIs.
    dim_red : str, optional
        Dimensionality reduction technique to use ('UMAP', 'PCA', 'tSNE').
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None, a new figure and axes object are created.
    cmap : str, optional
        Matplotlib colormap to use.
    figsize : tuple, optional
        Size of the figure to create. Ignored if 'ax' is not None.
    point_size : int, optional
        Size of the points in the scatter plot.

    Returns
    -------
    plt.Figure or plt.Axes
        The figure or axes object containing the plot.
    """
    if colour_by:
        cells = pd.crosstab([adata.obs[ROI_id], adata.obs[colour_by]], adata.obs[population], normalize=normalize).reset_index().copy()
    else:
        cells = pd.crosstab(adata.obs[ROI_id], adata.obs[population], normalize=normalize).reset_index().copy()

    summary_data = cells.iloc[:, 2:] if colour_by else cells.iloc[:, 1:]
    scaled_summary_data = sklearn.preprocessing.StandardScaler().fit_transform(summary_data)

    if dim_red == 'UMAP':
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(scaled_summary_data)
    elif dim_red == 'PCA':
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(scaled_summary_data)
    elif dim_red == 'tSNE':
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        embedding = tsne.fit_transform(scaled_summary_data)

    plot_data = pd.DataFrame(embedding, columns=[f'{dim_red} Dimension 1', f'{dim_red} Dimension 2'])
    plot_data['color'] = cells[colour_by] if colour_by else cells[ROI_id]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_new_fig = True
    else:
        created_new_fig = False

    sb.scatterplot(data=plot_data, x=f'{dim_red} Dimension 1', y=f'{dim_red} Dimension 2', hue='color', ax=ax, palette=cmap, s=point_size, legend="full")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if annotate:
        for _, row in plot_data.iterrows():
            ax.text(row[f'{dim_red} Dimension 1'], row[f'{dim_red} Dimension 2'], row['color'], ha='right', size='small')

    if save:
        fig.savefig(save, bbox_inches='tight')

    if created_new_fig:
        plt.show()
        return fig
    return ax


def overlayed_heatmaps(dfs: list[pd.DataFrame], 
                       cmaps: list[str] = ['Reds', 'Greens', 'Greens', 'Purples', 'Oranges', 'Greys'],
                       mode: str = 'horizontal', 
                       vmin_list: list[float] = None, 
                       vmax_list: list[float] = None, 
                       figsize: tuple[int, int] = (10, 10), 
                       grid_color: str = 'black', 
                       grid_linewidth: int = 1, 
                       save_path: str = None, 
                       colorbar_plot_list: list[bool] = None, 
                       colorbar_params: dict = {'size': "2%", 'pad': 0.25, 'spacing': 0.25}, 
                       colorbar_labels: list[str] = None, 
                       nan_color: str = 'lightgray',
                       return_fig=False,
                       show=True,
                       **heatmap_kwargs) -> None:
    """
    Overlay multiple heatmaps on the same plot.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames to plot for 'horizontal' mode; exactly two for 'diagonal' mode. DataFrames must be the same size.
    cmaps : list of str, optional
        List of colormaps for each DataFrame.
    mode : str, optional
        'horizontal' for horizontal bars; 'diagonal' for a diagonal split of the second DataFrame.
    vmin_list : list of float, optional
        List of vmin values. Default is to calculate automatically.
    vmax_list : list of float, optional
        List of vmax values. Default is to calculate automatically.
    figsize : tuple of int, optional
        Tuple for the figure size.
    grid_color : str, optional
        Color of the manually plotted grid lines.
    grid_linewidth : int, optional
        Width of the manually plotted grid lines.
    save_path : str, optional
        Path to save the output image. If None, the image is not saved.
    colorbar_plot_list : list of bool, optional
        By default all will be plotted, can alternatively provide a list of True/False entries to indicate which should be plotted.
    colorbar_params : dict, optional
        Dictionary to customize the size and placement of colour bars (size, pad, spacing).
    colorbar_labels : list of str, optional
        List of labels to associate with colorbars.
    nan_color : str
        Named matplotlib color to replace NaN values.
    **heatmap_kwargs
        Additional keyword arguments for sns.heatmap.
    """
    assert all(d.shape == dfs[0].shape for d in dfs), 'Data frames are not all the same shape' 

    if mode == 'diagonal' and len(dfs) != 2:
        raise ValueError("Diagonal mode requires exactly two DataFrames.")
    if mode not in ['horizontal', 'diagonal']:
        raise ValueError("Mode must be either 'horizontal' or 'diagonal'.")
        
    if not vmin_list:
        vmin_list = [df.min().min() for df in dfs]
    if not vmax_list:
        vmax_list = [df.max().max() for df in dfs]
        
    if not colorbar_plot_list:
        colorbar_plot_list = [True for _ in range(len(dfs))]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dfs[0], cmap=cmaps[0], cbar=False, ax=ax, **heatmap_kwargs)

    mappables = [ScalarMappable(Normalize(vmin=vmin_list[0], vmax=vmax_list[0]), cmap=cmaps[0])]
    
    if mode == 'horizontal':
        cell_height = 1.0 / len(dfs)
        for idx, df in enumerate(dfs[1:], start=1):
            norm = Normalize(vmin=vmin_list[idx], vmax=vmax_list[idx])
            mappable = ScalarMappable(norm=norm, cmap=cmaps[idx])
            mappables.append(mappable)

            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    bottom = i + cell_height * (len(dfs) - idx - 1)
                    
                    value = df.iloc[i, j]
                    
                    if np.isnan(value):
                        color = nan_color
                    else:
                        color = mappable.to_rgba(value)
                    
                    rect = plt.Rectangle((j, bottom), 1, cell_height, color=color)
                    ax.add_patch(rect)
    elif mode == 'diagonal':
        df = dfs[1]
        norm = Normalize(vmin=vmin_list[1], vmax=vmax_list[1])
        mappable = ScalarMappable(norm=norm, cmap=cmaps[1])
        mappables.append(mappable)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                
                value = df.iloc[i, j]
                    
                if np.isnan(value):
                    color = nan_color
                else:
                    color = mappable.to_rgba(value)
                
                triangle = plt.Polygon([(j, i), (j + 1, i), (j, i + 1)], color=color)
                ax.add_patch(triangle)

    for i in range(dfs[0].shape[0] + 1):
        ax.axhline(i, color=grid_color, linewidth=grid_linewidth)
    for j in range(dfs[0].shape[1] + 1):
        ax.axvline(j, color=grid_color, linewidth=grid_linewidth)

    ax.set_xlim([0, dfs[0].shape[1]])
    ax.set_ylim([0, dfs[0].shape[0]])

    divider = make_axes_locatable(ax)
    for idx, mappable in enumerate(mappables, start=0):
        if colorbar_plot_list[idx]:
            cax = divider.append_axes("right", size=colorbar_params.get('size', "2%"), pad=colorbar_params.get('pad', 0.02) + colorbar_params.get('spacing', 0.1) * idx)
            plt.colorbar(mappable, cax=cax)

            if colorbar_labels:
                cax.set_ylabel(colorbar_labels[idx])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
        
    if return_fig:
        return fig
    


def plot_colorbar(
    array: np.ndarray = None,
    vmin: float = None,
    vmax: float = None,
    cmap: str = 'viridis',
    orientation: str = 'horizontal',
    figsize: tuple = None,
    dpi: int = 100,
    aspect: int = 10,
    fontsize: int = None,
    hide_ticks: bool = False,
    label: str = '',
    tick_interval: float = None,
    tick_values: list = None,
    save: str = None
) -> plt.Figure:
    """
    A unified colorbar plotting function that can:
      - Create a colorbar from a provided array (old style), or
      - Create a "legend-only" style colorbar from vmin/vmax (no array),
    while allowing custom ticks, orientation, label, etc.

    Parameters
    ----------
    array : np.ndarray, optional
        If provided, we'll plot an image of this array and attach a colorbar
        (like your original function 'plot_colorbar').
        - If 1D, it is repeated to form a 2D array of shape (N,100).
        - If None, we skip the image approach and directly create a colorbar
          from vmin..vmax using a ScalarMappable (like a legend).
    vmin : float, optional
        Minimum data value for the color scale. If array is provided and vmin is None,
        we default to array's min. If array is None and vmin is None, default is 0.
    vmax : float, optional
        Maximum data value for the color scale. If array is provided and vmax is None,
        we default to array's max. If array is None and vmax is None, default is 1.
    cmap : str, optional
        Colormap name (or may be a Colormap object) used for the colorbar. (Default 'viridis')
    orientation : str, optional
        'horizontal' or 'vertical' colorbar. (Default 'horizontal')
    figsize : tuple, optional
        Figure size in inches. If None, a default is chosen based on orientation.
    dpi : int, optional
        Figure DPI (default 100).
    aspect : int, optional
        Aspect ratio for the colorbar when using the image approach. (Default 10)
    fontsize : int, optional
        Font size for colorbar tick labels. If None, uses default.
    hide_ticks : bool, optional
        If True, hide the colorbar ticks entirely. (Default False)
    label : str, optional
        If non-empty, will set this string as the colorbar label. (Default '')
    tick_interval : float, optional
        If provided, we create ticks at multiples of this interval from vmin..vmax.
        Ignored if tick_values is given.
    tick_values : list, optional
        If provided, explicitly set colorbar ticks to these positions. (Overrides tick_interval)
    save : str, optional
        File path to save the colorbar image (e.g., 'mycolorbar.png'). If provided,
        we save and do not return a figure. If None, return the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if 'save' is not provided. Otherwise, returns None
        after saving to the specified file.
    """
    # --------------------------------------------------------------------------------
    # Step 1: Determine vmin, vmax if array is given (or if not given)
    # --------------------------------------------------------------------------------
    if array is not None:
        # If array is 1D, expand it to 2D
        if array.ndim == 1:
            array = np.repeat(array[:, np.newaxis], 100, axis=1)

        # If vmin/vmax not given, get from the array
        if vmin is None:
            vmin = np.nanmin(array)
        if vmax is None:
            vmax = np.nanmax(array)
    else:
        # No array => we rely on user-provided vmin/vmax or default to (0..1)
        if vmin is None:
            vmin = 0.
        if vmax is None:
            vmax = 1.

    # --------------------------------------------------------------------------------
    # Step 2: Create the figure
    # Default figsize if none given
    # --------------------------------------------------------------------------------
    if figsize is None:
        if orientation == 'horizontal':
            figsize = (4, 1)
        else:
            figsize = (1, 4)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # --------------------------------------------------------------------------------
    # Step 3: Two modes:
    #   (A) array-based image with colorbar
    #   (B) direct "legend-only" colorbar from vmin..vmax
    # --------------------------------------------------------------------------------
    if array is not None:
        # (A) Array approach
        ax = fig.add_subplot(111)
        pos = ax.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        cbar = fig.colorbar(pos, ax=ax, orientation=orientation, aspect=aspect)

        # Hide the image axis
        ax.set_axis_off()

    else:
        # (B) Legend-only approach
        # We'll manually add an Axes for the colorbar that fills most of the figure
        if orientation == 'horizontal':
            # left=0.05, bottom=0.2, width=0.9, height=0.6
            cax = fig.add_axes([0.05, 0.3, 0.9, 0.4])
        else:
            # left=0.2, bottom=0.05, width=0.6, height=0.9
            cax = fig.add_axes([0.25, 0.05, 0.5, 0.9])

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # Convert str colormap to actual object if needed
        if isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap)
        else:
            cmap_obj = cmap

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])  # dummy

        cbar = fig.colorbar(sm, cax=cax, orientation=orientation)

    # --------------------------------------------------------------------------------
    # Step 4: Ticks + label + font size, etc.
    # --------------------------------------------------------------------------------
    # 4a) Possibly define explicit tick positions
    if tick_values is not None:
        cbar.set_ticks(tick_values)
    elif tick_interval is not None:
        # generate ticks from vmin..vmax
        ticks = np.arange(vmin, vmax + tick_interval*0.5, tick_interval)
        # clamp
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        cbar.set_ticks(ticks)

    # 4b) Label
    if label:
        cbar.set_label(label)

    # 4c) Hide ticks
    if hide_ticks:
        cbar.ax.set_yticks([]) if orientation == 'vertical' else cbar.ax.set_xticks([])

    # 4d) Fontsize
    if fontsize:
        cbar.ax.tick_params(labelsize=fontsize)

    # --------------------------------------------------------------------------------
    # Step 5: Save or return
    # --------------------------------------------------------------------------------
    if save:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig


def obs_to_mask_simple(
    adata: ad.AnnData,
    roi: str,
    roi_obs: str = 'ROI',
    cat_obs: str = None,
    cat_colour_map: str = 'tab20',
    cat_obs_groups: list = None,
    quant_obs: str = None,
    quant_colour_map: str = 'viridis',
    adata_colormap: bool = True,
    masks_folder: str = 'Masks',
    masks_ext: str = 'tiff',
    min_val: float = None,
    max_val: float = None,
    quantile: float = None,
    save_path: str = None,
    background_color: str = None,
    hide_axes: bool = False,
    hide_ticks: bool = True,
    svg_smoothing_factor: int = 0,
    label_obs: str = 'ObjectNumber'
) -> tuple:
    """
    Map values from an AnnData object to a mask and generate a color map, with options to save the resulting image.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    roi : str
        Region of interest identifier.
    roi_obs : str, optional
        Column in adata.obs that contains ROI information (default is 'ROI').
    cat_obs : str, optional
        Categorical observation to map.
    cat_colour_map : str, list or dict, optional
        Colormap for categorical data (default is 'tab20').
    cat_obs_groups : list or str, optional
        Specific groups to include.
    quant_obs : str, optional
        Quantitative observation to map.
    quant_colour_map : str, optional
        Colormap for quantitative data (default is 'viridis').
    adata_colormap : bool, optional
        Whether to use colormap from AnnData if available (default is True).
    masks_folder : str, optional
        Folder containing mask files (default is 'Masks').
    masks_ext : str, optional
        File extension for mask files (default is 'tiff').
    min_val : float, optional
        Minimum value for quantitative color scaling.
    max_val : float, optional
        Maximum value for quantitative color scaling.
    quantile : float, optional
        Quantile to set the maximum value for quantitative color scaling.
    save_path : str, optional
        File path to save the output image. Supports PNG and SVG formats.
    background_color : str, optional
        Background color of the saved image. If None, the background will be transparent for PNG files.
    hide_axes : bool, optional
        If True, hides the axes completely in the saved image (default is False).
    hide_ticks : bool, optional
        If True, hides the ticks and labels on the axes in the saved image (default is True).
    svg_smoothing_factor : int, optional
        Smoothing factor for SVG output (default is 0).
    label_obs : string, optional
        Identifier for cell labels in mask (default 'ObjectNumber')

    Returns
    -------
    tuple
        Mapped mask, pixel colormap, and category dictionary (if applicable).
    """
    # Read in mask
    mask = io.imread(Path(masks_folder, f'{roi}.{masks_ext}'))

    # Get cell table
    adata_roi_obs = adata.obs.loc[adata.obs[roi_obs] == roi, :].copy()   
    adata_roi_obs.reset_index(drop=True, inplace=True)

    # Check number of cells match
    cells_in_roi = adata_roi_obs.shape[0]
    cells_in_mask = len(np.unique(mask)) - 1
    assert cells_in_roi == cells_in_mask, f'Number of cells in mask ({cells_in_mask}) does not match cells in AnnData ({cells_in_roi}) for ROI {roi}'

    # Check that label obs exists
    if label_obs:
        assert label_obs in adata.obs.columns, f"{label_obs} not found in adata.obs. Set labels_obs=None"

    # Setup a blank mask we will map values into
    mapped_mask = np.zeros(mask.shape, dtype='uint16')    
    
    # If an adata.obs is categorical
    if cat_obs:
        # Get categories
        cats = adata.obs[cat_obs].cat.categories.tolist()
        
        # Use AnnData colormap, if available
        if adata_colormap and f'{cat_obs}_colormap' in adata.uns:
            cat_colour_map = adata.uns[f'{cat_obs}_colormap']

        ### Generate category colormap
        # If dict is supplied, use directly
        if isinstance(cat_colour_map, dict):
            cat_cmap = cat_colour_map
        #If a list, then create a dictionary
        elif isinstance(cat_colour_map,list):
            assert len(cats) < len(cat_colour_map), 'Colourmap list isnt long enough for number of populations'
            cat_cmap = {x:v for (x,v) in zip(cats, cat_colour_map)}
        # If a string, then get matplotlib cmap.
        elif isinstance(cat_colour_map,str):
            cat_colour_map = cm.get_cmap(cat_colour_map).colors
            assert len(cats) < len(cat_colour_map), 'Matplot ib colournap isnt long enough for number of populations'
            cat_cmap = {x:v for (x,v) in zip(cats, cat_colour_map)}
        else:
            raise 'Categorical colour map (cat_colour_map) must be a dictionary, list of colours, or name of a matplotlib colormap'
        
        # Filter specific observations, if specified
        if cat_obs_groups:
            if not isinstance(cat_obs_groups, list):
                cat_obs_groups = [cat_obs_groups]
            cats = [x for x in cats if x in cat_obs_groups]

        # Cat numbers
        cat_num = np.array(range(len(cats))) + 1    
   
        # Initialize dictionaries for mapping
        cat_dict = {}
        pixel_colormap = {}
        
        for cat, num in zip(cats, cat_num):
            #try:
                if not label_obs:
                    objects = adata_roi_obs.loc[adata_roi_obs[cat_obs] == cat, :].index.to_numpy() + 1
                else:
                    objects = adata_roi_obs.loc[adata_roi_obs[cat_obs] == cat, label_obs]
                cat_mask = np.isin(mask, objects)
                mapped_mask = np.where(cat_mask, num, mapped_mask)
                cat_dict[num] = str(cat)
                pixel_colormap[num] = cat_cmap[cat]
            #except:
            #    print(f'Error adding group {cat} from {cat_obs}')

    # If a quantitative value is supplied
    if quant_obs:
        # Label IDs
        if not label_obs:
            objects = adata_roi_obs.index.to_numpy() + 1
        else:
            objects = adata_roi_obs[label_obs]

        # Get values from adata.obs or adata.var_names
        if quant_obs in adata.obs:
            values = adata_roi_obs[quant_obs]
        elif quant_obs in adata.var_names:
            values = adata.X[adata.obs[roi_obs] == roi, adata.var_names == quant_obs].tolist()
        
        # Map array values
        quant_mask = map_array(np.asarray(mask), np.asarray(objects), np.asarray(values))        
        
        # If supplied, then use the masks calculated above to only show specific pops
        if cat_obs_groups != None and cat_obs != None:
            mapped_mask = np.where(mapped_mask != 0, quant_mask, 0)
        else:
            mapped_mask = quant_mask
        
        # Map pixel values to hex colours
        pixel_colormap = map_pixel_values_to_colors(mapped_mask, cmap_name=quant_colour_map, min_val=min_val, max_val=max_val, quantile=quantile)
        cat_dict = None
        
    if not save_path:
        return mapped_mask, pixel_colormap, cat_dict
    else:

        # Create folder for saving
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if save_path.split('.')[-1] == 'svg':
            save_labelled_image_as_svg(mapped_mask, pixel_colormap, save_path, exclude_zero=True, smoothing_factor=svg_smoothing_factor, background_color=background_color)
        else:
            save_labelled_image(mapped_mask, pixel_colormap, save_path, background_color=background_color, hide_axes=hide_axes, hide_ticks=hide_ticks)
            
            
# Voronoi plot functions from Nolan lab (https://github.com/nolanlab)


def _voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram.
    radius : float, optional
        Distance to 'points at infinity'. Default is None.

    Returns
    -------
    list of tuples
        Indices of vertices in each revised Voronoi region.
    list of tuples
        Coordinates for revised Voronoi vertices.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def _plot_voronoi(
    points: np.ndarray,
    colors: list,
    invert_y: bool = True,
    edge_color: str = 'facecolor',
    line_width: float = 0.1,
    alpha: float = 1,
    size_max: float = np.inf
) -> list:
    """
    Plot a Voronoi diagram.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of points to plot.
    colors : list
        List of colors for the Voronoi regions.
    invert_y : bool, optional
        If True, invert the Y-axis. Default is True.
    edge_color : str, optional
        Color of the edges. Default is 'facecolor'.
    line_width : float, optional
        Width of the edges. Default is 0.1.
    alpha : float, optional
        Alpha value for the Voronoi regions. Default is 1.
    size_max : float, optional
        Maximum size of the Voronoi regions. Default is np.inf.

    Returns
    -------
    list
        List of areas of the Voronoi regions.
    """
    if invert_y:
        points[:, 1] = max(points[:, 1]) - points[:, 1]
    vor = Voronoi(points)
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    pts = MultiPoint([Point(i) for i in points])
    mask = pts.convex_hull
    new_vertices = []
    if not isinstance(alpha, list):
        alpha = [alpha] * len(points)
    areas = []
    for i, (region, alph) in enumerate(zip(regions, alpha)):
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        areas.append(p.area)
        if p.area < size_max:
            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            new_vertices.append(poly)
            if edge_color == 'facecolor':
                plt.fill(*zip(*poly), alpha=alph, edgecolor=colors[i], linewidth=line_width, facecolor=colors[i])
            else:
                plt.fill(*zip(*poly), alpha=alph, edgecolor=edge_color, linewidth=line_width, facecolor=colors[i])
    return areas


def draw_voronoi_scatter(
    spot: pd.DataFrame,
    c: pd.DataFrame,
    voronoi_palette=sb.color_palette('bright'),
    scatter_palette='voronoi',
    X='X:X',
    Y='Y:Y',
    voronoi_hue='neighborhood10',
    scatter_hue='ClusterName',
    figsize=(5, 5),
    voronoi_kwargs={},
    scatter_kwargs={}
) -> list:
    """
    Plot Voronoi of a region and overlay the location of specific cell types.

    Parameters
    ----------
    spot : pd.DataFrame
        Cells that are used for Voronoi diagram.
    c : pd.DataFrame
        Cells that are plotted over Voronoi.
    voronoi_palette : list, optional
        Color palette used for coloring neighborhoods. Default is sb.color_palette('bright').
    scatter_palette : str or list, optional
        Color palette for scatter plot. Default is 'voronoi'.
    X : str, optional
        Column name used for X locations. Default is 'X:X'.
    Y : str, optional
        Column name used for Y locations. Default is 'Y:Y'.
    voronoi_hue : str, optional
        Column name used for neighborhood allocation. Default is 'neighborhood10'.
    scatter_hue : str, optional
        Column name used for scatter plot. Default is 'ClusterName'.
    figsize : tuple, optional
        Size of figure. Default is (5, 5).
    voronoi_kwargs : dict, optional
        Arguments passed to plot_voronoi function. Default is {}.
    scatter_kwargs : dict, optional
        Arguments passed to plt.scatter(). Default is {}.

    Returns
    -------
    list
        Sizes of each Voronoi region.
    """
    if scatter_palette == 'voronoi':
        scatter_palette = voronoi_palette
        scatter_hue = voronoi_hue

    if len(c) > 0:
        neigh_alpha = 0.3
    else:
        neigh_alpha = 1

    voronoi_kwargs = {**{'alpha': neigh_alpha}, **voronoi_kwargs}
    scatter_kwargs = {**{'s': 50, 'alpha': 1, 'marker': '.'}, **scatter_kwargs}

    plt.figure(figsize=figsize)
    colors = [voronoi_palette[i] for i in spot[voronoi_hue]]
    areas = _plot_voronoi(spot[[X, Y]].values, colors, **voronoi_kwargs)

    if len(c) > 0:
        if 'c' not in scatter_kwargs:
            colors = [scatter_palette[i] for i in c[scatter_hue]]
            scatter_kwargs['c'] = colors

        plt.scatter(
            x=c[X],
            y=(max(spot[Y]) - c[Y].values),
            **scatter_kwargs
        )

    plt.axis('off')
    return areas


def plot_stacked_graphs(dataframes, color_maps, plot_types, hide_axes=False, create_legends=True, order_by=0, ax_limits=None, y_labels=None, y_labels_rotations='vertical', height_ratios=None, figsize=(6, 2.5), graph_spacing=0.05):
    """
    Plot stacked graphs for given dataframes with various options.

    Parameters:
    - dataframes: List of DataFrames to be plotted
    - color_maps: List of color mappings for each DataFrame (can be a dict for 'stacked_bar' or a colormap str for 'bar')
    - plot_types: List of plot types for each DataFrame ('bar', 'stacked_bar')
    - hide_axes: List of booleans indicating whether to hide axes for each DataFrame
    - create_legends: List of booleans indicating whether to create a legend for each DataFrame
    - order_by: Index of the DataFrame to use for ordering all the dataframes
    - ax_limits: List of tuples specifying y-axis limits for each plot
    - y_labels: List of y-axis labels for each plot
    - y_labels_rotations: List of y-axis label rotations for each plot
    - height_ratios: List of height ratios for the subplots
    - figsize: Tuple representing the figure size
    - graph_spacing: Float representing the spacing between graphs

    Returns:
    - matplotlib.figure.Figure object
    """
    
    def process_list(input_list, default_value, num_plots):
        if input_list is None:
            return [default_value] * num_plots
        elif isinstance(input_list, list) and len(input_list) == 1:
            return input_list * num_plots
        elif isinstance(input_list, list):
            return input_list
        else:
            return [input_list] * num_plots

    num_plots = len(dataframes)

    color_maps = process_list(color_maps, 'viridis', num_plots)
    plot_types = process_list(plot_types, 'bar', num_plots)
    hide_axes = process_list(hide_axes, False, num_plots)
    create_legends = process_list(create_legends, True, num_plots)
    ax_limits = process_list(ax_limits, None, num_plots)
    y_labels = process_list(y_labels, None, num_plots)
    y_labels_rotations = process_list(y_labels_rotations, 'vertical', num_plots)
    height_ratios = process_list(height_ratios, 1, num_plots)
    
    # Ensure all dataframes have matching indices
    common_index = dataframes[order_by].index
    for df in dataframes:
        common_index = common_index.intersection(df.index)
    
    # Filter and order dataframes by the specified dataframe
    ordered_dataframes = [df.loc[common_index, :] for df in dataframes]
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_plots, 1, height_ratios=height_ratios)
    
    for i, (df, color_map, plot_type, hide_axis, create_legend, ax_limit, y_label, y_label_rotation) in enumerate(zip(ordered_dataframes, color_maps, plot_types, hide_axes, create_legends, ax_limits, y_labels, y_labels_rotations)):
        ax = plt.subplot(gs[i])
        
        # Plot based on the specified type
        if plot_type == 'bar':
            norm = plt.Normalize(df.dropna().values.min(), df.dropna().values.max())
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            for col in df.columns:
                df[col].plot(kind='bar', width=1, ax=ax, color=sm.to_rgba(df[col]), legend=False)
            
            
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.3)
            cbar.set_label('Values')
        elif plot_type == 'stacked_bar':
            colors = [color_map.get(col, '#333333') for col in df.columns]
            df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=1)
        
        # Hide axes if specified
        if hide_axis:
            ax.tick_params(axis='y', length=0, labelleft=False)
            ax.set_ylabel('')

        if ax_limit:
            ax.set_ylim(ax_limit)
        else:
            ax.autoscale(enable=True, axis='y', tight=True)
            
        if y_label:
            if y_label_rotation[0].lower() == 'h':
                ax.set_ylabel(y_label, rotation=y_label_rotation, ha='right', va='center')
            else:
                ax.set_ylabel(y_label)
        
        # Create legend if specified
        if create_legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small', labelspacing=0.1)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
                
        ax.tick_params(axis='x', length=0, labelbottom=False)
        ax.set_xlabel('')
        ax.autoscale(enable=True, axis='x', tight=True)

    plt.subplots_adjust(hspace=graph_spacing)
    return fig

def catobs_to_stacked(adata, cat_obs, groupby='Case', fillna=False, return_long=False):
    
    # Extract a long data frame
    cat_df = adata.obs[~adata.obs[groupby].duplicated()][[groupby,cat_obs]].set_index(groupby,drop=True)
    
    if fillna:
        cat_df = cat_df.fillna(fillna)
    
    # Expand into a dummies dataframe which we can use for plotting
    dummy_df = pd.get_dummies(cat_df, columns=[cat_obs], prefix='', prefix_sep='')
    
    if return_long:
        return cat_df, dummy_df,
    else:
        return dummy_df
    
def reorder_within_categories_by_continuous(df_cat, df_cont, cont_col, cat_columns=None, ascending=True):

    new_index = []
    
    if not cat_columns:
        cat_columns = df_cat.columns
    
    for c in cat_columns:
        data = df_cat.loc[df_cat[c] != 0, c]

        assert all(x == data.iloc[0] for x in data), "Not all values are the same within each category"

        sorted_index = df_cont.loc[data.index, :].sort_values(cont_col, ascending=ascending).index

        new_index += sorted_index.tolist()
        
    return df_cat.loc[new_index, :], df_cont.loc[new_index, :]
    
    
def norm_by_row(df):
    return df.div(df.sum(axis=1), axis=0)

def norm_by_col(df):
    return df.div(df.sum(axis=0), axis=1)

def no_norm(df):
    return df


import numpy as np
import pandas as pd
from pathlib import Path

from skimage import io, segmentation, morphology
import matplotlib.cm as mpl_cm

##################################################
# Pseudocode imports for your existing utility functions.
# Adjust as needed for your actual code environment.
##################################################
# from spatialbiotools.utils import (
#     map_array,
#     map_pixel_values_to_colors,
#     save_labelled_image,
#     save_labelled_image_as_svgf
# )

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from skimage import io, segmentation, morphology

# If you have your own utility functions, import them or define inline.
# E.g.:
# from SpatialBiologyToolkit.utils import (
#     map_array,
#     map_pixel_values_to_colors,
#     save_labelled_image,
#     save_labelled_image_as_svg,
# )

def obs_to_mask(
    adata,
    roi: str,
    roi_obs: str = 'ROI',
    check_cell_numbers: bool = False,
    # Inner fill (cat/quant) params
    cat_obs: str = None,
    cat_colour_map='tab20',
    cat_obs_groups=None,
    quant_obs: str = None,
    quant_colour_map: str = 'viridis',
    quant_exclude_background: bool = True,
    adata_colormap: bool = True,
    # Mask file location
    masks_folder: str = 'Masks',
    masks_ext: str = 'tiff',
    # Numeric scaling for quant_obs
    min_val: float = None,
    max_val: float = None,
    quantile: float = None,
    # Output
    save_path: str = None,
    background_color: str = None,
    hide_axes: bool = False,
    hide_ticks: bool = True,
    svg_smoothing_factor: int = 0,
    dpi: int = 300,
    # Label handling (DEFAULT CHANGED)
    label_obs: str = 'ObjectNumber',
    # Separator ring
    separator_color=None,
    separator_thickness: int = 1,
    separator_mode='inner',
    separator_connectivity=1,
    # Outline from second categorical
    outline_cat_obs: str = None,
    outline_cat_colour_map='tab20',
    outline_thickness: int = 1,
    outline_mode='inner',
    outline_connectivity=1,
    # NEW: The order in which layers are plotted
    layers_order=None
) -> tuple:
    """
    Map values from an AnnData object to a mask and generate a color map, with options
    to save the result. The labeling is done in up to three layers:

      1) "Inner" fill from cat_obs or quant_obs (optional).
      2) "Separator" ring of uniform color around each cell (optional).
      3) "Outline" from a second categorical observation (optional).

    If `label_obs` is provided (default: 'ObjectNumber'), all mapping uses those
    label IDs to match AnnData rows to mask labels. This allows correct mapping
    even if some cells were removed from AnnData or if mask labels are not 1..N.
    """

    # Default layer order if user didn't provide any
    if layers_order is None:
        layers_order = ['inner', 'separator', 'outline']

    # ------------------------------------------------------------
    # 1) Load the base segmentation mask and subset AnnData to ROI
    # ------------------------------------------------------------
    mask_path = Path(masks_folder) / f"{roi}.{masks_ext}"
    base_mask = io.imread(mask_path)
    if base_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {base_mask.shape} for ROI '{roi}'.")

    adata_roi_obs = adata.obs.loc[adata.obs[roi_obs] == roi].copy()
    adata_roi_obs.reset_index(drop=True, inplace=True)

    # ---- NEW/CHANGED: validate label_obs and define cells_in_adata safely
    cells_in_adata = len(adata_roi_obs)
    if label_obs is not None:
        assert label_obs in adata.obs.columns, (
            f"{label_obs} not found in adata.obs. "
            f"Set label_obs=None to fall back to index+1 mapping."
        )

    if check_cell_numbers:
        cells_in_mask = len(np.unique(base_mask)) - 1  # assume 0 is background
        if cells_in_adata != cells_in_mask:
            raise ValueError(
                f"Mask has {cells_in_mask} cells, AnnData has {cells_in_adata} for ROI='{roi}'."
            )

    # The final label image to fill in
    mapped_mask = np.zeros_like(base_mask, dtype=np.uint16)

    # We'll store an integer->color mapping
    pixel_colormap = {}
    cat_dict = None  # store category info if using cat_obs

    # ------------------------------------------------------------
    # HELPER: build a cat colormap from user input or adata.uns
    # ------------------------------------------------------------
    def _build_cat_cmap(cmap_in, cat_obs_in, use_adata_cmap=True):
        
        # If a listed colour map exists in AnnData
        if use_adata_cmap and f"{cat_obs_in}_colormap" in adata.uns:
            return adata.uns[f"{cat_obs_in}_colormap"]

        # If just a list of colors exists in AnnData
        if use_adata_cmap and f"{cat_obs_in}_colors" in adata.uns:
            colors = adata.uns[f"{cat_obs_in}_colors"]
            cats_all = adata.obs[cat_obs_in].cat.categories
            return {c: color for c, color in zip(cats_all, colors)}

        if isinstance(cmap_in, dict):
            return cmap_in
        elif isinstance(cmap_in, list):
            cats_all = adata.obs[cat_obs_in].cat.categories
            return {c: color for c, color in zip(cats_all, cmap_in)}
        elif isinstance(cmap_in, str):
            col = mpl_cm.get_cmap(cmap_in).colors
            cats_all = adata.obs[cat_obs_in].cat.categories
            return {c: cval for c, cval in zip(cats_all, col)}
        else:
            raise ValueError(
                "cat_colour_map must be a dict, list, or valid matplotlib colormap name."
            )

    # ------------------------------------------------------------
    # FUNCTION 1: do the "inner" fill
    # ------------------------------------------------------------
    def apply_inner_fill():
        """
        Fills the 'inner' layer with either:
          - a categorical observation (cat_obs), or
          - a quantitative observation (quant_obs).

        When `label_obs` is provided, mapping uses those label IDs so removed/missing
        cells in AnnData do not corrupt the mapping.
        """
        nonlocal mapped_mask, pixel_colormap, cat_dict, cat_obs_groups, label_obs

        # ---------------------------
        # 1) Categorical observation
        # ---------------------------
        if cat_obs:
            cats_all = adata.obs[cat_obs].cat.categories.tolist()

            # Possibly subset categories
            if cat_obs_groups:
                if not isinstance(cat_obs_groups, list):
                    cat_obs_groups = [cat_obs_groups]
                cats_all = [c for c in cats_all if c in cat_obs_groups]

            # Build colormap
            cat_cmap = _build_cat_cmap(cat_colour_map, cat_obs, adata_colormap)

            cat_dict = {}
            for i, cat_val in enumerate(cats_all, start=1):
                # Use label_obs if provided; else fall back to index+1
                if label_obs:
                    cell_ids = adata_roi_obs.loc[adata_roi_obs[cat_obs] == cat_val, label_obs].values
                else:
                    cell_ids = (
                        adata_roi_obs.loc[adata_roi_obs[cat_obs] == cat_val].index + 1
                    ).to_numpy()

                cat_mask = np.isin(base_mask, cell_ids)
                mapped_mask[cat_mask] = i

                cat_dict[i] = str(cat_val)
                pixel_colormap[i] = cat_cmap.get(cat_val, (0.5, 0.5, 0.5))

        # --------------------------
        # 2) Quantitative observation
        # --------------------------
        elif quant_obs:
            # ---- CHANGED: always prefer label_obs for the objects -> values mapping
            if label_obs:
                objects = adata_roi_obs[label_obs].values
            else:
                objects = (np.arange(cells_in_adata) + 1).astype(int)

            # Values from obs or var
            if quant_obs in adata_roi_obs.columns:
                values = adata_roi_obs[quant_obs].values
            elif quant_obs in adata.var_names:
                subX = adata[adata.obs[roi_obs] == roi, [quant_obs]].X
                values = np.asarray(subX).ravel()
            else:
                raise ValueError(f"Can't find '{quant_obs}' in adata.obs or var_names.")

            # Map values to pixels using the explicit objects->values dictionary
            quant_mask = map_array(np.asarray(base_mask), np.asarray(objects), np.asarray(values))

            if quant_exclude_background:
                nonzero_locs = (base_mask != 0)
                mapped_mask[nonzero_locs] = quant_mask[nonzero_locs]
            else:
                mapped_mask = quant_mask

            # Build colormap from numeric data
            colormap = map_pixel_values_to_colors(
                mapped_mask,
                cmap_name=quant_colour_map,
                min_val=min_val,
                max_val=max_val,
                quantile=quantile
            )

            if quant_exclude_background and 0 in colormap:
                del colormap[0]

            pixel_colormap.update(colormap)

        else:
            # If neither cat_obs nor quant_obs is given, do nothing for inner fill
            pass

    # ------------------------------------------------------------
    # FUNCTION 2: draw a "separator" ring (uniform color)
    # ------------------------------------------------------------
    def apply_separator():
        nonlocal mapped_mask, pixel_colormap

        if separator_color is None:
            return
        separator_id = np.max(mapped_mask) + 1

        # Use the base mask for boundaries
        any_cell = base_mask > 0

        boundary = segmentation.find_boundaries(
            any_cell, connectivity=separator_connectivity, mode=separator_mode
        )
        if separator_thickness > 1:
            boundary = morphology.binary_dilation(boundary, morphology.disk(separator_thickness))

        mapped_mask[boundary] = separator_id
        pixel_colormap[separator_id] = separator_color

    # ------------------------------------------------------------
    # FUNCTION 3: draw an "outline" from a second categorical obs
    # ------------------------------------------------------------
    def apply_outline():
        nonlocal mapped_mask, pixel_colormap

        if outline_cat_obs is None:
            return

        outline_cats_all = adata.obs[outline_cat_obs].cat.categories.tolist()
        outline_cmap_dict = _build_cat_cmap(outline_cat_colour_map, outline_cat_obs, adata_colormap)

        start_id = int(np.max(mapped_mask)) + 1

        for idx, cat_val in enumerate(outline_cats_all, start=start_id):
            if label_obs:
                cell_ids = adata_roi_obs.loc[
                    adata_roi_obs[outline_cat_obs] == cat_val,
                    label_obs
                ].values
            else:
                cell_ids = (
                    adata_roi_obs.loc[adata_roi_obs[outline_cat_obs] == cat_val].index + 1
                ).to_numpy()

            cat_mask = np.isin(base_mask, cell_ids)

            cat_boundary = segmentation.find_boundaries(
                cat_mask, connectivity=outline_connectivity, mode=outline_mode
            )

            if outline_thickness > 1:
                cat_boundary = morphology.binary_dilation(
                    cat_boundary, morphology.disk(outline_thickness)
                )

            mapped_mask[cat_boundary] = idx
            pixel_colormap[idx] = outline_cmap_dict.get(cat_val, (0, 0, 0))

    # ------------------------------------------------------------
    # 2) Run the steps in the user-specified order
    # ------------------------------------------------------------
    layer_functions = {
        'inner': apply_inner_fill,
        'separator': apply_separator,
        'outline': apply_outline
    }

    for layer in layers_order:
        func = layer_functions.get(layer)
        if func is not None:
            func()
        else:
            raise ValueError(
                f"Invalid layer '{layer}' in layers_order. Must be one of {list(layer_functions.keys())}."
            )

    # ------------------------------------------------------------
    # 3) Possibly save the final image
    # ------------------------------------------------------------
    if not save_path:
        return mapped_mask, pixel_colormap, cat_dict

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    ext = save_path_obj.suffix.lower()
    if ext == '.svg':
        save_labelled_image_as_svg(
            mapped_mask,
            pixel_colormap,
            str(save_path_obj),
            exclude_zero=True,
            smoothing_factor=svg_smoothing_factor,
            background_color=background_color,
            dpi=dpi
        )
    else:
        save_labelled_image(
            mapped_mask,
            pixel_colormap,
            str(save_path_obj),
            background_color=background_color,
            hide_axes=hide_axes,
            hide_ticks=hide_ticks,
            dpi=dpi
        )

    return mapped_mask, pixel_colormap, cat_dict


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from skimage import io, segmentation, morphology
from pathlib import Path

# from SpatialBiologyToolkit.utils import (
#     map_array,
#     map_pixel_values_to_colors,
#     save_labelled_image,
#     save_labelled_image_as_svg,
# )


def grouped_graph(
        adata,
        group_by_obs,
        x_axis,
        ROI_id='ROI',
        display_tables=True,
        fig_size=(6, 5),
        errorbar='se',
        save_graph=False,
        save_table=False,
        log_scale=True,
        order=None,
        scale_factor=False,
        proportions=False,
        stacked=False,
        width=0.8,
        ylabel=None,
        sort_by_population=None,
        use_adata_colormap=True
):
    """
    Plot a grouped bar graph of cell counts or proportions from an AnnData object.

    Parameters:
    - adata: AnnData object
    - group_by_obs: str, obs column used for bar color/grouping
    - x_axis: str, obs column for x-axis categories
    - ROI_id: str, ROI column name (used for cross-tabulation)
    - display_tables: bool, whether to display raw crosstab table
    - fig_size: tuple, matplotlib figure size
    - errorbar: Error bars, e.g se (standard error) or sd (standard deviation)
    - save_graph: str or False, path to save the graph (must end in .png, .jpg, .svg, etc)
    - save_table: str or False, path to save the table (must end in .csv)
    - log_scale: bool, use log scale for y-axis
    - order: list or None, custom order for x-axis
    - scale_factor: number or False, normalize values by this factor
    - proportions: bool, normalize crosstab output (proportions)
    - stacked: bool, if True, use pandas stacked bar plot instead of seaborn
    - sort_by_population: str or None, if given, sort x_axis by abundance of this group
    - use_adata_colormap: bool, if True, use colormap from AnnData.uns if available (default True)
    """

    crosstab_norm = 'index' if proportions else False

    # Crosstab
    if x_axis != ROI_id:
        # Create crosstab with ROI level first (always start with raw counts)
        table_raw = pd.crosstab(
            [adata.obs[group_by_obs], adata.obs[ROI_id]],
            adata.obs[x_axis],
            normalize=False
        )
        table_raw.columns = table_raw.columns.astype(str)
        
        if proportions:
            # For proportions: aggregate across ROIs first, then normalize by column (x_axis category)
            table_agg = table_raw.groupby(level=0).sum()  # Sum across ROIs for each population
            table = table_agg.div(table_agg.sum(axis=0), axis=1)  # Normalize columns to sum to 1
        else:
            # For counts: keep raw table with ROI structure
            table = table_raw
        
        # Create long format data
        if proportions:
            # For proportions, we work with the aggregated table (no ROI level)
            data_long = table.reset_index().melt(id_vars=[group_by_obs])
        else:
            # For counts, keep ROI level
            data_long = table.reset_index().melt(id_vars=[group_by_obs, ROI_id])
    else:
        table = pd.crosstab(
            adata.obs[group_by_obs],
            adata.obs[x_axis],
            normalize=crosstab_norm
        )
        table.columns = table.columns.astype(str)
        data_long = table.reset_index().melt(id_vars=group_by_obs)

    # Optional scaling (e.g. to cells/mm²)
    if scale_factor:
        data_long['value'] = data_long['value'] / scale_factor

    # Optional sorting
    if sort_by_population:
        sort_df = data_long[data_long[group_by_obs] == sort_by_population]
        sort_order = sort_df.groupby(x_axis)['value'].sum().sort_values(ascending=False).index.tolist()
        order = sort_order if order is None else order

    # Extract colors from AnnData if available
    palette = None
    if use_adata_colormap:
        # Check for colormap in AnnData.uns (try multiple possible keys)
        color_key_options = [
            f"{group_by_obs}_colors",
            f"{group_by_obs}_colormap", 
            f"{group_by_obs}_colour_map"
        ]
        
        for color_key in color_key_options:
            if color_key in adata.uns:
                colors = adata.uns[color_key]
                
                # Get the categories for this observation
                if hasattr(adata.obs[group_by_obs], 'cat'):
                    categories = adata.obs[group_by_obs].cat.categories.tolist()
                else:
                    categories = sorted(adata.obs[group_by_obs].unique().tolist())
                
                # Create palette dict or list depending on color format
                if isinstance(colors, dict):
                    palette = colors
                elif isinstance(colors, (list, tuple, np.ndarray)):
                    # Map categories to colors
                    palette = {cat: color for cat, color in zip(categories, colors[:len(categories)])}
                
                break  # Use first found colormap
    
    # Plotting
    fig, ax = plt.subplots(figsize=fig_size)

    if stacked:
        pivot_table = data_long.pivot_table(
            index=x_axis,
            columns=group_by_obs,
            values='value',
            aggfunc='sum',
            fill_value=0
        )
        if order:
            pivot_table = pivot_table.loc[order]
        
        # Apply colors to stacked plot if palette is available
        if palette:
            # Create color list in the order of pivot_table columns
            color_list = [palette.get(col, '#1f77b4') for col in pivot_table.columns]
            pivot_table.plot(kind='bar', stacked=True, ax=ax, width=width, color=color_list)
        else:
            pivot_table.plot(kind='bar', stacked=True, ax=ax, width=width)
    else:
        sb.barplot(
            data=data_long,
            y="value",
            x=x_axis,
            hue=group_by_obs,
            errorbar=errorbar,
            order=order,
            ax=ax,
            width=width,
            palette=palette
        )

    # Formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_xlabel(x_axis)
    ax.grid(False)

    if not ylabel:
        ylabel = 'Cells'

    if proportions:
        ylabel = 'Proportion'
        ax.set_ylim(0, 1)

    ax.set_ylabel(ylabel)

    if log_scale and not proportions:
        ax.set_yscale("log")

    ax.legend(bbox_to_anchor=(1.01, 1), title=group_by_obs)

    if save_graph:
        fig.savefig(save_graph, bbox_inches='tight', dpi=200)

    if save_table:
        table.to_csv(save_table)

    if display_tables:
        print('Raw data:')
        display(HTML(table.to_html()))