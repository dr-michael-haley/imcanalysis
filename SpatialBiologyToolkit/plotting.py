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
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy import stats
from shapely.geometry import MultiPoint, Point, Polygon
from skimage.util import map_array, img_as_ubyte
from skimage.transform import resize
from skimage.measure import find_contours
from skimage import exposure
from glob import glob
from itertools import compress
import re
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
            wide = wide.groupby(mean_over, observed=True).mean(numeric_only=True)
        long = long.groupby(mean_over + [pop_col], observed=True).mean(numeric_only=True).reset_index()

    return wide, long

def bargraph(data: ad.AnnData | pd.DataFrame | str,
             pop_col: str = None,
             value_col: str = None,
             hue: str = None,
             hue_order: list = None,
             specify_populations: list | None = None,
             levels: list = None,
             mean_over: list = None,
             confidence_interval: int = 68,
             figsize: tuple = (5, 5),
             ax: plt.Axes | None = None,
             crosstab_normalize: bool = False,
             cells_per_mm: bool = False,
             palette: dict = None,
             x_palette: dict = None,
             bar_edgecolor: str | None = None,
             bar_linewidth: float | None = None,
             hide_grid: bool = False,
             despine: bool = True,
             legend: bool = True,
             log_scale: bool = False,
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
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If provided, figsize is ignored.
    crosstab_normalize : bool, optional
        Whether, and how, to normalize crosstab results.
    cells_per_mm : bool, optional
        Normalize values by mm².
    palette : dict, optional
        Color palette.
    x_palette : dict, optional
        Dictionary mapping x-axis values to colors, applied after plotting.
    bar_edgecolor : str, optional
        Edge color for bars (e.g., "black").
    bar_linewidth : float, optional
        Edge line width for bars.
    hide_grid : bool, optional
        Whether to hide the grid.
    despine : bool, optional
        Whether to remove top/right axes spines.
    legend : bool, optional
        Whether to display the legend.
    log_scale : bool, optional
        Whether to use log scale for the y-axis.
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
    adata = data if isinstance(data, ad.AnnData) else None
    data = _check_input_type(data)
    if not levels and not mean_over:
        assert ROI_col_name in data.columns, f'{ROI_col_name} column not found in data'
        if case_col_name in data.columns:
            levels = [case_col_name, ROI_col_name]
            mean_over = [case_col_name, ROI_col_name]
        else:
            levels = [ROI_col_name]
            mean_over = [ROI_col_name]
    levels = _to_list(levels)
    mean_over = _to_list(mean_over)
    specify_populations = _to_list(specify_populations) if specify_populations else []

    if specify_populations:
        data = data[data[pop_col].isin(specify_populations)]

    if hue and hue not in levels:
        levels.append(hue)
    if hue and hue not in mean_over:
        mean_over.append(hue)

    if not palette and adata is not None:
        color_key = f'{hue}_colormap' if hue else f'{pop_col}_colormap'
        palette = adata.uns.get(color_key, palette)

    if value_col:
        _, long_form_data = _count_summary(data, pop_col, levels, mean_over, crosstab_normalize, 'numeric')
        plot_data = long_form_data.reset_index()
        y_plot = value_col
        x_plot = pop_col if pop_col else levels[-1]
    else:
        _, long_form_data = _count_summary(data, pop_col, levels, mean_over, crosstab_normalize)
        long_form_data.rename(columns={'value': 'Cells'}, inplace=True)
        plot_data = long_form_data
        y_plot, x_plot = 'Cells', pop_col

    if cells_per_mm and adata is not None:
        try:
            size_dict = adata.uns['sample']['mm2'].to_dict()
            plot_data['mm2'] = plot_data[ROI_col_name].map(size_dict)
            new_y = f'{y_plot} per mm²'
            plot_data[new_y] = plot_data[y_plot] / plot_data['mm2']
            y_plot = new_y
        except Exception:
            pass

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False
    sb.barplot(
        data=plot_data,
        y=y_plot,
        x=x_plot,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        errorbar=("ci", confidence_interval),
    )

    if x_palette:
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        x_ticks = np.asarray(ax.get_xticks())
        for patch in ax.patches:
            if len(x_ticks) == 0:
                continue
            x_center = patch.get_x() + patch.get_width() / 2
            idx = int(np.argmin(np.abs(x_ticks - x_center)))
            if 0 <= idx < len(x_labels):
                x_val = x_labels[idx]
                if x_val in x_palette:
                    patch.set_facecolor(x_palette[x_val])

    if bar_edgecolor is not None or bar_linewidth is not None:
        for patch in ax.patches:
            if bar_edgecolor is not None:
                patch.set_edgecolor(bar_edgecolor)
            if bar_linewidth is not None:
                patch.set_linewidth(bar_linewidth)
                    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_x_labels)

    if log_scale:
        ax.set_yscale('log')
    
    if hide_grid:
        ax.grid(False)
    if despine:
        sb.despine(ax=ax)
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
    return fig if created_fig else ax

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
        roi_counts = subset.groupby(case_col, observed=True)[roi_col].nunique()
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
        result = {pop_col: i, 'mlm_p_value': mdf.pvalues.iloc[1], 'mlm_warnings': str(warning_messages)}
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
    scatter_kwargs={},
    show=True,
    close_fig=False
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
    show : bool, optional
        Whether to display the plot. Default is True.
    close_fig : bool, optional
        Whether to close the figure to prevent memory leaks. Default is False.

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
    
    fig = plt.gcf()  # Get current figure
    
    if show:
        plt.show()
    
    if close_fig:
        plt.close(fig)
    
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
            table_agg = table_raw.groupby(level=0, observed=True).sum()  # Sum across ROIs for each population
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
        sort_order = sort_df.groupby(x_axis, observed=True)['value'].sum().sort_values(ascending=False).index.tolist()
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
            fill_value=0,
            observed=True
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
    ax.tick_params(axis='x', rotation=90, labelsize=10)
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
    
    # Close the figure to prevent memory leaks
    plt.close(fig)


def load_single_img(filename: str) -> np.ndarray:
    """
    Load a single 2D .tif or .tiff image as float32.

    Args:
        filename (str): Path to the image file, must end with .tiff or .tif.

    Returns:
        np.ndarray: Loaded image data (2D).
    """
    import tifffile as tp
    
    if not (filename.endswith('.tiff') or filename.endswith('.tif')):
        raise ValueError('Raw file should end with .tif or .tiff!')
    img_in = tp.imread(filename).astype('float32')

    if img_in.ndim != 2:
        raise ValueError('Single image should be 2D!')
    return img_in


def load_imgs_from_directory(
    load_directory: str,
    channel_name: str,
    quiet: bool = False
) -> tuple[list[np.ndarray], list[str], list[str]] | None:
    """
    Searches a directory (and any subfolders) for images whose filenames
    contain the given channel_name. Returns a list of those images and filenames.

    Args:
        load_directory (str): Directory or parent directory of images.
        channel_name (str): Channel name to search for in the filenames.
        quiet (bool): Whether to suppress print statements.

    Returns:
        Optional[Tuple[List[np.ndarray], List[str], List[str]]]:
            - A list of images (2D numpy arrays)
            - A list of corresponding filenames
            - A list of the subfolders from which images were loaded
              If no images are found, returns None.
    """
    img_collect = []
    img_file_list = []

    # Find any subdirectories (one level down). If none found, use load_directory itself.
    img_folders = glob(os.path.join(load_directory, "*", "")) or [load_directory]

    if not quiet:
        print(f'Loading image data for channel "{channel_name}" from ...')

    for subfolder in img_folders:
        found_files = [
            f for f in os.listdir(subfolder)
            if os.path.isfile(os.path.join(subfolder, f)) and
               (f.lower().endswith(".tiff") or f.lower().endswith(".tif"))
        ]

        for candidate_file in found_files:
            # More precise matching: check if channel_name appears as a separate word/token
            # This prevents CD45 from matching CD45RO
            filename_lower = candidate_file.lower()
            channel_lower = channel_name.lower()
            
            # Split on common separators: underscore, dash, dot, space
            filename_tokens = re.split(r'[_\-\.\s]+', filename_lower)
            
            # Check if channel name matches any token exactly
            if channel_lower in filename_tokens:
                img_read = load_single_img(os.path.join(subfolder, candidate_file))
                if not quiet:
                    print(os.path.join(subfolder, candidate_file))
                img_file_list.append(candidate_file)
                img_collect.append(img_read)
                # Break once we find the first matching file per subfolder.
                break

    if not quiet:
        print('Image data loading completed!')

    if not img_collect:
        print(f'No files found with channel name "{channel_name}".')
        return None

    return img_collect, img_file_list, img_folders


def load_rescale_images(
    image_folder: str,
    samples_list: list[str],
    marker: str,
    minimum: float,
    max_val: float | str
) -> tuple[list[np.ndarray], list[str]]:
    """
    Helper function that:
      1) Loads images for a given marker across provided samples.
      2) Clips intensities using user-specified or quantile-based maxima.
      3) Rescales intensities to [0,1].

    Args:
        image_folder (str): Directory where images (and subfolders) are located.
        samples_list (List[str]): List of ROI/sample names to filter by.
        marker (str): The marker (channel name) to load from the image folder.
        minimum (float): Lower clip value.
        max_val (Union[float, str]): A numeric max or a string with prefix:
          - 'q': Mean quantile
          - 'i': Individual quantile
          - 'm': Minimum of quantiles
          - 'x': Maximum of quantiles
          Example: 'q0.97' => Use mean of the 97th percentile for all images.

    Returns:
        Tuple[List[np.ndarray], List[str]]:
            - List of images (each rescaled/clipped)
            - Matching list of ROI names
    """
    # Interpret the user-specified max_val mode
    mode = 'value'
    if isinstance(max_val, str):
        prefix = max_val[0].lower()
        try:
            max_quantile = float(max_val[1:])
        except ValueError:
            raise ValueError(f"Could not parse quantile from '{max_val}'")
        if prefix == 'q':
            mode = 'mean_quantile'
        elif prefix == 'i':
            mode = 'individual_quantile'
        elif prefix == 'm':
            mode = 'minimum_quantile'
        elif prefix == 'x':
            mode = 'max_quantile'

    # Load the images (quiet=True to suppress prints)
    loaded = load_imgs_from_directory(image_folder, marker, quiet=True)
    if not loaded:
        return [], []

    image_list, _, folder_list = loaded

    # ROI names are the last part of the subfolder path
    roi_list = [os.path.basename(Path(x)) for x in folder_list]

    # Filter out any ROIs not in our samples_list
    sample_filter = [r in samples_list for r in roi_list]
    image_list = list(compress(image_list, sample_filter))
    roi_list = list(compress(roi_list, sample_filter))

    if not image_list:
        # If nothing is loaded after filtering, return.
        print(f"No images found for marker {marker} matching {samples_list}.")
        return [], []

    # Compute maximum intensities
    if mode in ('mean_quantile', 'minimum_quantile', 'max_quantile'):
        # For each image, find the quantile, then reduce them by mean, min, or max
        all_vals = [np.quantile(im, max_quantile) for im in image_list]
        if mode == 'mean_quantile':
            max_value = float(np.mean(all_vals))
            mode_str = f'Mean of {max_quantile} quantiles'
        elif mode == 'minimum_quantile':
            max_value = float(np.min(all_vals))
            mode_str = f'Min of {max_quantile} quantiles'
        else:  # 'max_quantile'
            max_value = float(np.max(all_vals))
            mode_str = f'Max of {max_quantile} quantiles'

        print(f"Marker={marker} | Mode={mode_str} | Min={minimum:.3f} | "
              f"Calculated max={max_value:.3f}")
        image_list = [im.clip(minimum, max_value) for im in image_list]

    elif mode == 'individual_quantile':
        # Each image is clipped to its own quantile
        max_values = [np.quantile(im, max_quantile) for im in image_list]
        print(f"Marker={marker} | Mode=Individual quantile {max_quantile} | "
              f"Min={minimum} | Using image-specific maxima.")
        image_list = [
            im.clip(minimum, mv) for im, mv in zip(image_list, max_values)
        ]

    else:
        # Fixed numeric value
        print(f"Marker={marker} | Using numeric min={minimum}, max={max_val}")
        max_value = float(max_val)
        image_list = [im.clip(minimum, max_value) for im in image_list]

    # Rescale intensities to [0..1]
    image_list = [exposure.rescale_intensity(i) for i in image_list]

    return image_list, roi_list


def make_images(
    image_folder: str,
    samples_list: list[str],
    output_folder: str,
    name_prefix: str = '',
    minimum: float = 0.2,
    max_quantile: float | str = 'q0.97',
    red: str | None = None,
    red_range: tuple[float, float | str] | None = None,
    green: str | None = None,
    green_range: tuple[float, float | str] | None = None,
    blue: str | None = None,
    blue_range: tuple[float, float | str] | None = None,
    magenta: str | None = None,
    magenta_range: tuple[float, float | str] | None = None,
    cyan: str | None = None,
    cyan_range: tuple[float, float | str] | None = None,
    yellow: str | None = None,
    yellow_range: tuple[float, float | str] | None = None,
    white: str | None = None,
    white_range: tuple[float, float | str] | None = None,
    roi_folder_save: bool = False,
    simple_file_names: bool = False,
    save_subfolder: str = ''
) -> None:
    """
    Create composite RGB images from up to seven channels. Each channel can be
    mapped onto red/green/blue/magenta/cyan/yellow/white in an additive manner
    (as done by typical multi-channel viewers).

    Args:
        image_folder (str): Folder of subfolders where each ROI is stored.
        samples_list (List[str]): List of ROI names to process.
        output_folder (str): Where to save the resulting images.
        name_prefix (str): Optional prefix for output files.
        minimum (float): Global intensity minimum for clipping (before rescale).
        max_quantile (float or str): Global intensity maximum for clipping
            (e.g., 0.97 or 'q0.97' or 'i0.97').
        {color} (str): The marker to use for that color channel.
        {color}_range (tuple): Lower and upper intensity specs, can be numeric or 'q0.95', etc.
        roi_folder_save (bool): Whether each ROI gets its own subfolder in output.
        simple_file_names (bool): If True, save images as 'ROI.png' only (otherwise includes channel info).
        save_subfolder (str): Subdirectory under output_folder for saving images.

    Returns:
        None. Saves .png images to disk.
    """

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Map the color name to the marker name and to user-specified ranges
    color_configs = {
        'red':     (red,     red_range),
        'green':   (green,   green_range),
        'blue':    (blue,    blue_range),
        'magenta': (magenta, magenta_range),
        'cyan':    (cyan,    cyan_range),
        'yellow':  (yellow,  yellow_range),
        'white':   (white,   white_range),
    }

    # For each color channel, load images & ROI lists
    loaded_images = {}  # color -> list of scaled images
    loaded_rois = {}    # color -> list of ROI names

    for color_name, (marker_name, color_range) in color_configs.items():
        if marker_name is not None:
            # Determine the min/max for this channel if provided
            if color_range is not None:
                ch_min, ch_max = color_range
            else:
                ch_min, ch_max = (minimum, max_quantile)

            imgs, rois = load_rescale_images(
                image_folder, samples_list,
                marker=marker_name,
                minimum=ch_min,
                max_val=ch_max
            )
            loaded_images[color_name] = imgs
            loaded_rois[color_name] = rois
        else:
            # This color not used
            loaded_images[color_name] = []
            loaded_rois[color_name] = []

    # Figure out how many ROIs total we have. We can unify by taking the max length across channels.
    num_rois = max(len(r) for r in loaded_rois.values()) if loaded_rois else 0
    print(f'Found {num_rois} ROIs total (across requested channels).')

    # Build the final composite images ROI by ROI
    # Note: The assumption here is that channels align in the same "ROI index" order,
    # because we used the same sample_list for each. If your data is misaligned, you'll
    # need more robust logic (e.g., matching by ROI name).
    for i in range(num_rois):
        # We pick whichever color has a valid ROI for indexing
        # and assume they are the same ROI in the same i-th position
        # in each channel's list. If any channel is missing that i-th ROI,
        # we fill with zeros.

        # A quick approach is to find a channel that has rois for index i
        # and get the actual ROI name from there. Then we try to match
        # which index that ROI is in the other channels. A more thorough
        # approach would be to unify them by dictionary, but that requires
        # more changes.

        # For simplicity, let's pick the first non-empty color:
        some_color = None
        for c in color_configs:
            if i < len(loaded_rois[c]):
                some_color = c
                break
        if some_color is None:
            continue  # no channels have i-th ROI (unlikely)

        roi_name = loaded_rois[some_color][i]

        # Now gather the images for each color, matching the ROI name
        # by index if it matches, else a blank array of the same shape.
        shape_ref = loaded_images[some_color][i].shape  # reference shape
        (h, w) = shape_ref

        # Initialize R, G, B as zeros
        channel_r = np.zeros((h, w), dtype=np.float32)
        channel_g = np.zeros((h, w), dtype=np.float32)
        channel_b = np.zeros((h, w), dtype=np.float32)

        for color_name, (marker_name, _) in color_configs.items():
            if marker_name is None:
                # Not used
                continue
            # Attempt to find the ROI in that channel
            if roi_name in loaded_rois[color_name]:
                # find the index for that ROI
                idx = loaded_rois[color_name].index(roi_name)
                this_img = loaded_images[color_name][idx]
            else:
                # no ROI found => fallback to zeros
                this_img = np.zeros((h, w), dtype=np.float32)

            if color_name == 'red':
                channel_r = np.clip(channel_r + this_img, 0, 1)
            elif color_name == 'green':
                channel_g = np.clip(channel_g + this_img, 0, 1)
            elif color_name == 'blue':
                channel_b = np.clip(channel_b + this_img, 0, 1)
            elif color_name == 'magenta':
                # Magenta = Red + Blue
                channel_r = np.clip(channel_r + this_img, 0, 1)
                channel_b = np.clip(channel_b + this_img, 0, 1)
            elif color_name == 'cyan':
                # Cyan = Green + Blue
                channel_g = np.clip(channel_g + this_img, 0, 1)
                channel_b = np.clip(channel_b + this_img, 0, 1)
            elif color_name == 'yellow':
                # Yellow = Red + Green
                channel_r = np.clip(channel_r + this_img, 0, 1)
                channel_g = np.clip(channel_g + this_img, 0, 1)
            elif color_name == 'white':
                # White = Red + Green + Blue
                channel_r = np.clip(channel_r + this_img, 0, 1)
                channel_g = np.clip(channel_g + this_img, 0, 1)
                channel_b = np.clip(channel_b + this_img, 0, 1)

        # Stack channels
        stack = np.dstack([channel_r, channel_g, channel_b])
        stack_ubyte = img_as_ubyte(stack)

        # Build filename
        if not simple_file_names:
            # Include the channels used (e.g., b_markerName_, r_markerName_, etc.)
            color_strs = []
            for color_name, (marker_name, _) in color_configs.items():
                if marker_name:
                    prefix = color_name[0].lower()  # r/g/b/m/c/y/w
                    color_strs.append(f'{prefix}_{marker_name}')
            color_part = "_".join(color_strs)
            filename = f'{name_prefix}{roi_name}_{color_part}'.rstrip('_')
        else:
            filename = roi_name

        # Possibly write to a subfolder named after ROI
        if roi_folder_save:
            roi_dir = Path(output_folder, roi_name)
            roi_dir.mkdir(parents=True, exist_ok=True)
            if save_subfolder:
                roi_dir = roi_dir / save_subfolder
                roi_dir.mkdir(parents=True, exist_ok=True)
            save_path = roi_dir / f'{filename}.png'
        else:
            out_dir = Path(output_folder)
            if save_subfolder:
                out_dir = out_dir / save_subfolder
                out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f'{filename}.png'

        io.imsave(str(save_path), stack_ubyte)


def create_population_overlay(
    adata,
    population: str,
    pop_obs: str,
    roi_name: str,
    composite_image_path: str,
    mask_path: str = None,
    roi_obs: str = 'ROI',
    object_index_obs: str = 'ObjectNumber',
    output_path: str = None,
    contour_color: tuple = (255, 255, 255),  # White contours
    contour_width: int = 2,
    verbose: bool = True,
    legend_markers: list[str] | None = None,
    legend_colors: list[tuple[int, int, int] | str] | None = None,
    legend_fontsize: int = 10,
    legend_box_size: tuple[float, float] = (0.25, 0.18),
    show_label: bool = True,
    show_population_label: bool = True,
    population_label_text: str | dict | None = None,
    population_label_fontsize: int | None = None,
    crop_size: tuple[int, int] | None = None,
    crop_origin: str = "center",
    show_scale_bar: bool = False,
    scale_bar_length: int = 25,
    scale_bar_thickness: int = 3,
    scale_bar_color: str = "white",
    scale_bar_outline_thickness: int = 2,
    scale_bar_text: str | None = None,
    scale_bar_text_size: int = 10
):
    """
    Create an overlay visualization showing all cells of a specific population
    with mask contours on a composite RGB image.
    
    Args:
        adata: AnnData object with cell data
        population: Population name to visualize
        pop_obs: Column name containing population labels
        roi_name: Name of the ROI to process
        composite_image_path: Path to the composite RGB image
        mask_path: Path to the segmentation mask (optional)
        roi_obs: Column name for ROI identifiers
        object_index_obs: Column name for cell indices
        output_path: Where to save the overlay image
        contour_color: RGB color for cell contours (default: yellow)
        contour_width: Width of contour lines in pixels
        verbose: Whether to print status messages (default: True)
        legend_markers: Optional list of marker names to show in a legend
        legend_colors: Optional list of RGB tuples (0-255) matching legend_markers
        legend_fontsize: Font size for legend text
        legend_box_size: (width, height) as fraction of axes for the legend inset
        show_label: Whether to show the top-left label box.
        show_population_label: Whether to include the population label in the text.
        population_label_text: Optional string or dict mapping ROI -> text.
        crop_size: Optional crop size (width, height) in pixels.
        crop_origin: Crop origin anchor: "upper_left", "upper_right",
            "lower_left", "lower_right", or "center".
        show_scale_bar: Whether to draw a scale bar in the bottom-right.
        scale_bar_length: Length of the scale bar in pixels.
        scale_bar_thickness: Line thickness in pixels.
        scale_bar_color: Scale bar color (matplotlib color string).
        scale_bar_outline_thickness: Black outline thickness in pixels.
        scale_bar_text: Optional text displayed above the scale bar.
        scale_bar_text_size: Font size for scale bar text.
        
    Returns:
        None. Saves overlay image to output_path if provided.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation
    
    # Load composite image
    if not Path(composite_image_path).exists():
        if verbose:
            print(f"Warning: Composite image not found: {composite_image_path}")
        return
        
    composite_img = io.imread(composite_image_path)
    
    # Load mask if provided
    mask = None
    if mask_path and Path(mask_path).exists():
        mask = io.imread(mask_path)
    elif mask_path:
        if verbose:
            print(f"Warning: Mask file not found: {mask_path}")
    
    # Get cells of this population in this ROI
    roi_cells = adata.obs[
        (adata.obs[roi_obs] == roi_name) & 
        (adata.obs[pop_obs].astype(str) == str(population))
    ]
    
    if len(roi_cells) == 0:
        if verbose:
            print(f"No cells of population '{population}' found in ROI '{roi_name}'")
        return
        
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Display composite image as background
    base_image = ax.imshow(composite_img)
    
    # If we have a mask, draw contours for cells of this population
    if mask is not None:
        # Get unique cell labels for this population
        if object_index_obs in roi_cells.columns:
            target_cell_ids = set(roi_cells[object_index_obs].astype(int))
        else:
            # Fallback: use mask values at cell locations
            target_cell_ids = set()
            for _, cell in roi_cells.iterrows():
                if 'X_loc' in cell and 'Y_loc' in cell:
                    x, y = int(cell['X_loc']), int(cell['Y_loc'])
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                        target_cell_ids.add(mask[y, x])
        
        # Remove background (typically 0)
        target_cell_ids.discard(0)
        
        if target_cell_ids:
            target_mask = np.isin(mask, list(target_cell_ids))
            if not target_mask.any():
                if verbose:
                    print("Warning: No matching labels found in mask for selected population.")
            else:
                boundaries = find_boundaries(target_mask, mode='inner')
                if contour_width > 1:
                    iterations = max(1, contour_width // 2)
                    boundaries = binary_dilation(boundaries, iterations=iterations)

                overlay = np.zeros((*boundaries.shape, 4), dtype=float)
                overlay[..., :3] = np.array(contour_color) / 255.0
                overlay[..., 3] = boundaries.astype(float)
                ax.imshow(overlay)
                if verbose:
                    print(f"Drew contours for {len(target_cell_ids)} cells")
        else:
            if verbose:
                print("Warning: No target cell IDs found; skipping contour overlay.")
        
    else:
        # If no mask, just plot cell centers as points
        if verbose:
            print(f"No mask available, plotting cell centers as points")
        if 'X_loc' in roi_cells.columns and 'Y_loc' in roi_cells.columns:
            ax.scatter(roi_cells['X_loc'], roi_cells['Y_loc'], 
                      c=[np.array(contour_color)/255], s=20, alpha=0.8, marker='o')
    
    # Apply optional central crop AFTER overlays but BEFORE legends/labels
    if crop_size is not None and composite_img is not None:
        crop_w, crop_h = crop_size
        h, w = composite_img.shape[:2]
        crop_w = max(1, min(int(crop_w), w))
        crop_h = max(1, min(int(crop_h), h))

        origin = (crop_origin or "center").lower()
        if origin == "upper_left":
            x_min, y_min = 0, 0
        elif origin == "upper_right":
            x_min, y_min = w - crop_w, 0
        elif origin == "lower_left":
            x_min, y_min = 0, h - crop_h
        elif origin == "lower_right":
            x_min, y_min = w - crop_w, h - crop_h
        elif origin == "center":
            x_min = (w - crop_w) // 2
            y_min = (h - crop_h) // 2
        else:
            raise ValueError(
                "crop_origin must be one of 'upper_left', 'upper_right', "
                "'lower_left', 'lower_right', or 'center'."
            )

        x_min = max(0, min(x_min, w - crop_w))
        y_min = max(0, min(y_min, h - crop_h))
        x_max = x_min + crop_w
        y_max = y_min + crop_h

        # Maintain image orientation (origin='upper')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    # Optional scale bar drawn into pixels AFTER crop (positioned bottom-left of cropped view)
    scale_bar_text_pos = None
    if show_scale_bar and composite_img is not None:
        from matplotlib.colors import to_rgb

        img_h, img_w = composite_img.shape[:2]
        margin_px = 20
        pad_px = max(0, int(scale_bar_outline_thickness))
        bar_len = max(1, int(scale_bar_length))
        bar_thick = max(1, int(scale_bar_thickness))

        # Use current view limits to determine crop window
        x_min, x_max = ax.get_xlim()
        y_top, y_bottom = ax.get_ylim()  # origin='upper' -> y_top > y_bottom
        x_min_i = int(round(min(x_min, x_max)))
        x_max_i = int(round(max(x_min, x_max)))
        y_min_i = int(round(min(y_top, y_bottom)))
        y_max_i = int(round(max(y_top, y_bottom)))

        # Clamp to image bounds
        x_min_i = max(0, min(x_min_i, img_w - 1))
        x_max_i = max(0, min(x_max_i, img_w))
        y_min_i = max(0, min(y_min_i, img_h - 1))
        y_max_i = max(0, min(y_max_i, img_h))

        x0 = x_min_i + margin_px
        x1 = min(x0 + bar_len - 1, x_max_i - margin_px - 1)
        y1 = y_max_i - margin_px - 1
        y0 = y1 - bar_thick + 1

        x0p = max(0, x0 - pad_px)
        x1p = min(img_w - 1, x1 + pad_px)
        y0p = max(0, y0 - pad_px)
        y1p = min(img_h - 1, y1 + pad_px)

        rgb_bar = tuple(int(round(c * 255)) for c in to_rgb(scale_bar_color))
        rgb_pad = (0, 0, 0)

        def _apply_color(img, ys, xs, rgb):
            if img.ndim == 2:
                val = int(round(sum(rgb) / 3))
                img[ys, xs] = np.clip(val, 0, 255)
            else:
                img[ys, xs, 0] = np.clip(rgb[0], 0, 255)
                img[ys, xs, 1] = np.clip(rgb[1], 0, 255)
                img[ys, xs, 2] = np.clip(rgb[2], 0, 255)
            return img

        if x1 > x0 and y1 > y0:
            composite_img = _apply_color(composite_img, slice(y0p, y1p + 1), slice(x0p, x1p + 1), rgb_pad)
            composite_img = _apply_color(composite_img, slice(y0, y1 + 1), slice(x0, x1 + 1), rgb_bar)

            # Update the base image shown (keep current limits)
            base_image.set_data(composite_img)

            if scale_bar_text:
                scale_bar_text_pos = ((x0 + x1) / 2, y0 - 6)

    renderer = None
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    # Optional scale bar text (data coords, will crop naturally)
    if show_scale_bar and scale_bar_text and scale_bar_text_pos is not None:
        ax.text(
            scale_bar_text_pos[0],
            scale_bar_text_pos[1],
            str(scale_bar_text),
            color=scale_bar_color,
            fontsize=scale_bar_text_size,
            ha='center',
            va='bottom'
        )

    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    if renderer is None:
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:
            renderer = None

    # Optional legend showing marker colors used in the composite
    if legend_markers and legend_colors and len(legend_markers) == len(legend_colors):
        # Normalize legend colors to RGB tuples (0-255), allowing matplotlib color strings
        from matplotlib.colors import to_rgb

        normalized_colors = []
        for color in legend_colors:
            if isinstance(color, str):
                rgb_float = to_rgb(color)
                normalized_colors.append(tuple(int(round(c * 255)) for c in rgb_float))
            else:
                normalized_colors.append(color)
        from matplotlib.transforms import Bbox

        try:
            from matplotlib import font_manager as fm

            if renderer is None:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

            padding_px = 15
            margin_px = 20
            gap_px = 15

            labels = [str(l).strip() for l in legend_markers]
            fontprops = fm.FontProperties(size=legend_fontsize)
            sizes = [renderer.get_text_width_height_descent(lbl, fontprops, False)[:2] for lbl in labels]

            if sizes:
                max_w = max(w for w, _ in sizes)
                total_h = sum(h for _, h in sizes) + gap_px * (len(sizes) - 1)
            else:
                max_w = total_h = 0

            box_w_px = max_w + 2 * padding_px
            box_h_px = total_h + 2 * padding_px

            fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi

            width_frac = max(box_w_px / fig_w_px, 0.02)
            height_frac = max(box_h_px / fig_h_px, 0.02)

            margin_x = margin_px / fig_w_px
            margin_y = margin_px / fig_h_px

            ax_pos = ax.get_position()
            pos_x = max(0.0, ax_pos.x1 - width_frac - margin_x)
            pos_y = max(0.0, ax_pos.y1 - height_frac - margin_y)

            inset_ax = fig.add_axes([pos_x, pos_y, width_frac, height_frac])
            inset_ax.set_facecolor((0, 0, 0, 1))
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlim(0, 1)
            inset_ax.set_ylim(0, 1)
            inset_ax.set_zorder(ax.get_zorder() + 1)

            inset_w_px = box_w_px
            inset_h_px = box_h_px
            y_cursor = inset_h_px - padding_px
            for (lbl, rgb), (w, h) in zip(zip(labels, normalized_colors), sizes):
                color = tuple(np.array(rgb) / 255.0)
                y_cursor -= h / 2
                inset_ax.text(padding_px / inset_w_px, y_cursor / inset_h_px, lbl,
                              color=color, fontsize=legend_fontsize,
                              va='center', ha='left')
                y_cursor -= h / 2 + gap_px

            for spine in inset_ax.spines.values():
                spine.set_color('white')
                spine.set_linewidth(0.5)
        except Exception:
            # Fallback: simple inset in axes coords
            default_w, default_h = legend_box_size
            inset_ax = ax.inset_axes([1 - default_w - 0.02, 1 - default_h - 0.02, default_w, default_h])
            inset_ax.set_facecolor((0, 0, 0, 1))
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlim(0, 1)
            inset_ax.set_ylim(0, 1)
            for (lbl, rgb) in zip(legend_markers, normalized_colors):
                color = tuple(np.array(rgb) / 255.0)
                inset_ax.text(0.05, 0.95 - 0.1 * legend_markers.index(lbl), str(lbl).strip(),
                              color=color, fontsize=legend_fontsize, va='top', ha='left')

    # Optional label in top-left
    label_text = None
    if show_label:
        pop_label = str(population).strip() if show_population_label else None
        extra_label = None
        if isinstance(population_label_text, dict):
            extra_label = population_label_text.get(roi_name)
        elif population_label_text is not None:
            extra_label = population_label_text

        if extra_label is not None:
            extra_label = str(extra_label).strip()
            if extra_label == "":
                extra_label = None

        label_parts = []
        if pop_label:
            label_parts.append(pop_label)
        if extra_label:
            label_parts.append(extra_label)

        if label_parts:
            label_text = ". ".join(label_parts)

    if label_text:
        try:
            from matplotlib import font_manager as fm

            if renderer is None:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

            padding_px = 15
            margin_px = 20

            fontprops = fm.FontProperties(size=population_label_fontsize or legend_fontsize)
            w, h = renderer.get_text_width_height_descent(label_text, fontprops, False)[:2]

            box_w_px = w + 2 * padding_px
            box_h_px = h + 2 * padding_px

            fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi

            width_frac = max(box_w_px / fig_w_px, 0.02)
            height_frac = max(box_h_px / fig_h_px, 0.02)

            margin_x = margin_px / fig_w_px
            margin_y = margin_px / fig_h_px

            ax_pos = ax.get_position()
            pos_x = max(0.0, ax_pos.x0 + margin_x)
            pos_y = max(0.0, ax_pos.y1 - height_frac - margin_y)

            inset_ax = fig.add_axes([pos_x, pos_y, width_frac, height_frac])
            inset_ax.set_facecolor((0, 0, 0, 1))
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlim(0, 1)
            inset_ax.set_ylim(0, 1)
            inset_ax.set_zorder(ax.get_zorder() + 1)

            inset_ax.text(padding_px / box_w_px, 1 - padding_px / box_h_px,
                          label_text, color='white', fontsize=population_label_fontsize or legend_fontsize,
                          va='top', ha='left')

            for spine in inset_ax.spines.values():
                spine.set_color('white')
                spine.set_linewidth(0.5)
        except Exception:
            # Fallback: simple inset in axes coords (top-left)
            default_w, default_h = legend_box_size
            inset_ax = ax.inset_axes([0.02, 1 - default_h - 0.02, default_w, default_h])
            inset_ax.set_facecolor((0, 0, 0, 1))
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlim(0, 1)
            inset_ax.set_ylim(0, 1)
            inset_ax.text(0.05, 0.95, label_text,
                          color='white', fontsize=population_label_fontsize or legend_fontsize,
                          va='top', ha='left')
    
    # Save if output path provided
    if output_path:
        # Save tightly around the axes with no padding to avoid a white border
        fig.savefig(
            output_path,
            bbox_inches='tight',
            pad_inches=0,
            dpi=200,
            transparent=False,
        )
        plt.close(fig)
    else:
        plt.show()
        
    return fig


def population_backgating(
    adata,
    population: str,
    pop_obs: str,
    image_folder: str,
    roi_list: list[str] = None,
    roi_obs: str = 'ROI',
    object_index_obs: str = 'ObjectNumber',
    output_folder: str = 'Population_Backgating',
    # Marker assignments for composite images
    red: str | None = None,
    red_range: tuple[float, float | str] | None = None,
    green: str | None = None,
    green_range: tuple[float, float | str] | None = None,
    blue: str | None = None,
    blue_range: tuple[float, float | str] | None = None,
    magenta: str | None = None,
    magenta_range: tuple[float, float | str] | None = None,
    cyan: str | None = None,
    cyan_range: tuple[float, float | str] | None = None,
    yellow: str | None = None,
    yellow_range: tuple[float, float | str] | None = None,
    white: str | None = None,
    white_range: tuple[float, float | str] | None = None,
    # Image processing parameters
    minimum: float = 0.2,
    max_quantile: float | str = 'q0.97',
    # Mask parameters
    use_masks: bool = True,
    mask_folder: str = 'masks',
    mask_extension: str = 'tiff',
    # Overlay appearance
    contour_color: tuple = (255, 255, 255),  # White contours by default
    contour_width: int = 2,
    # Output options
    save_composite_images: bool = True,
    save_overlays: bool = True,
    create_summary_figure: bool = True
) -> dict:
    """
    Streamlined backgating function that creates composite images and population overlays.
    
    This function combines make_images() and create_population_overlay() to provide
    an easy way to visualize specific cell populations with custom marker channels.
    
    Args:
        adata: AnnData object with cell data
        population: Population name to visualize
        pop_obs: Column name containing population labels
        image_folder: Directory containing image subfolders for each ROI
        roi_list: List of ROI names to process. If None, uses all ROIs with the population
        roi_obs: Column name for ROI identifiers
        object_index_obs: Column name for cell indices
        output_folder: Directory to save output images
        
        # Color channel assignments (same as make_images)
        red, green, blue, etc.: Marker names for each color channel
        red_range, green_range, etc.: Intensity ranges for each channel
        
        # Image processing
        minimum: Global intensity minimum for clipping
        max_quantile: Global intensity maximum for clipping
        
        # Mask parameters
        use_masks: Whether to use segmentation masks for overlays
        mask_folder: Directory containing mask files
        mask_extension: File extension for mask files
        
        # Overlay appearance
        contour_color: RGB color for population contours
        contour_width: Width of contour lines
        
        # Output control
        save_composite_images: Whether to save composite RGB images
        save_overlays: Whether to save population overlay images
        create_summary_figure: Whether to create a summary figure with all ROIs
        
    Returns:
        dict: Summary information including file paths and cell counts
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Create output directories
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    composite_dir = output_path / 'composite_images'
    overlay_dir = output_path / 'population_overlays'
    
    if save_composite_images:
        composite_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Get ROIs that contain the target population
    population_cells = adata.obs[adata.obs[pop_obs].astype(str) == str(population)]
    
    if len(population_cells) == 0:
        print(f"No cells found for population '{population}'")
        return {'error': f'No cells found for population {population}'}
    
    available_rois = population_cells[roi_obs].unique().tolist()
    
    if roi_list is None:
        roi_list = available_rois
    else:
        # Filter to only ROIs that have the population
        roi_list = [roi for roi in roi_list if roi in available_rois]
    
    if len(roi_list) == 0:
        print(f"No ROIs found containing population '{population}'")
        return {'error': f'No ROIs found containing population {population}'}
    
    print(f"Processing {len(roi_list)} ROIs for population '{population}'")
    
    # Step 1: Create composite images
    composite_paths = {}
    if save_composite_images:
        print("Creating composite images...")
        make_images(
            image_folder=image_folder,
            samples_list=roi_list,
            output_folder=str(composite_dir),
            name_prefix=f'{population}_',
            minimum=minimum,
            max_quantile=max_quantile,
            red=red,
            red_range=red_range,
            green=green,
            green_range=green_range,
            blue=blue,
            blue_range=blue_range,
            magenta=magenta,
            magenta_range=magenta_range,
            cyan=cyan,
            cyan_range=cyan_range,
            yellow=yellow,
            yellow_range=yellow_range,
            white=white,
            white_range=white_range,
            roi_folder_save=False,
            simple_file_names=True
        )
        
        # Build dictionary of composite image paths
        for roi in roi_list:
            composite_paths[roi] = composite_dir / f'{population}_{roi}.png'
    
    # Step 2: Create population overlays
    overlay_paths = {}
    cell_counts = {}
    successful_overlays = 0
    
    if save_overlays:
        print("Creating population overlays...")
        
        for roi in roi_list:
            # Get cell count for this ROI and population
            roi_pop_cells = population_cells[population_cells[roi_obs] == roi]
            cell_counts[roi] = len(roi_pop_cells)
            
            if cell_counts[roi] == 0:
                print(f"  Skipping {roi}: no {population} cells found")
                continue
            
            # Determine paths
            if save_composite_images and composite_paths[roi].exists():
                composite_img_path = str(composite_paths[roi])
            else:
                # Try to find existing composite or create a minimal one
                print(f"  Warning: No composite image found for {roi}, creating overlay without background")
                composite_img_path = None
            
            mask_path = None
            if use_masks:
                for ext in [mask_extension, 'tif', 'tiff']:
                    potential_mask = Path(mask_folder) / f'{roi}.{ext}'
                    if potential_mask.exists():
                        mask_path = str(potential_mask)
                        break
                
                if not mask_path:
                    print(f"  Warning: No mask found for {roi}")
            
            # Create overlay
            overlay_output_path = overlay_dir / f'{population}_{roi}_overlay.png'
            overlay_paths[roi] = overlay_output_path
            
            try:
                if composite_img_path and Path(composite_img_path).exists():
                    create_population_overlay(
                        adata=adata,
                        population=population,
                        pop_obs=pop_obs,
                        roi_name=roi,
                        composite_image_path=composite_img_path,
                        mask_path=mask_path,
                        roi_obs=roi_obs,
                        object_index_obs=object_index_obs,
                        output_path=str(overlay_output_path),
                        contour_color=contour_color,
                        contour_width=contour_width
                    )
                    successful_overlays += 1
                    print(f"  Created overlay for {roi}: {cell_counts[roi]} {population} cells")
                else:
                    print(f"  Skipped overlay for {roi}: no composite image available")
            
            except Exception as e:
                print(f"  Error creating overlay for {roi}: {e}")
                continue
    
    # Step 3: Create summary figure (optional)
    summary_info = {
        'population': population,
        'total_rois_processed': len(roi_list),
        'total_cells': sum(cell_counts.values()),
        'successful_overlays': successful_overlays,
        'composite_paths': composite_paths,
        'overlay_paths': overlay_paths,
        'cell_counts_by_roi': cell_counts,
        'output_folder': str(output_path)
    }
    
    if create_summary_figure and successful_overlays > 0:
        print("Creating summary figure...")
        try:
            # Create a grid layout for summary
            n_rois = len([roi for roi in roi_list if roi in overlay_paths and overlay_paths[roi].exists()])
            if n_rois > 0:
                cols = min(3, n_rois)  # Max 3 columns
                rows = (n_rois + cols - 1) // cols  # Ceiling division
                
                fig = plt.figure(figsize=(cols * 4, rows * 4))
                gs = GridSpec(rows, cols, figure=fig)
                
                roi_idx = 0
                for roi in roi_list:
                    if roi not in overlay_paths or not overlay_paths[roi].exists():
                        continue
                    
                    row = roi_idx // cols
                    col = roi_idx % cols
                    ax = fig.add_subplot(gs[row, col])
                    
                    # Load and display overlay image
                    overlay_img = io.imread(str(overlay_paths[roi]))
                    ax.imshow(overlay_img)
                    ax.set_title(f'{roi}\n{cell_counts[roi]} {population} cells')
                    ax.axis('off')
                    
                    roi_idx += 1
                
                plt.suptitle(f'Population: {population}', fontsize=16)
                plt.tight_layout()
                
                summary_path = output_path / f'{population}_summary.png'
                plt.savefig(summary_path, bbox_inches='tight', dpi=200, facecolor='white')
                plt.close(fig)
                
                summary_info['summary_figure'] = str(summary_path)
                print(f"Summary figure saved to: {summary_path}")
        
        except Exception as e:
            print(f"Error creating summary figure: {e}")
    
    # Save summary information
    summary_csv = output_path / f'{population}_summary.csv'
    summary_df = pd.DataFrame([
        {'ROI': roi, 'Population': population, 'Cell_Count': cell_counts.get(roi, 0)}
        for roi in roi_list
    ])
    summary_df.to_csv(summary_csv, index=False)
    summary_info['summary_csv'] = str(summary_csv)
    
    print(f"\nPopulation backgating complete!")
    print(f"Total ROIs: {len(roi_list)}")
    print(f"Total {population} cells: {sum(cell_counts.values())}")
    print(f"Successful overlays: {successful_overlays}")
    print(f"Output folder: {output_path}")
    
    return summary_info


def umap_marker_gallery(
    adata,
    markers=None,
    ncols=5,
    point_size=2,
    panel_scale=1.3,
    title_size=6,
    tight_pad=0.25,
    vmax=None,
    cmap="viridis",
    add_colorbar=False,
    colorbar_label=None,
    colorbar_rect=(0.85, 0.5, 0.015, 0.4),  # (x, y, width, height)
    colorbar_orientation="vertical",
    colorbar_tick_size=None,
    colorbar_label_size=None,
    show=True,
    save=None,
    layer=None,
    dpi=300
):
    """
    Plot a gallery of UMAPs coloured by marker expression, with an optional
    single shared colourbar.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing UMAP embedding.
    markers : list[str] or None
        Markers to plot. If None, uses adata.var_names.
    ncols : int
        Number of columns in the gallery.
    point_size : float
        Marker size for UMAP points.
    panel_scale : float
        Inches per panel (controls overall figure size).
    title_size : int
        Font size for each panel title.
    tight_pad : float
        Padding for tight_layout.
    vmax : float or None
        Maximum value for colour scaling (shared across panels).
    cmap : str
        Matplotlib colormap.
    add_colorbar : bool
        Whether to add a single shared colourbar.
    colorbar_label : str or None
        Label for the colourbar.
    colorbar_rect : tuple
        (x, y, width, height) in figure fraction coordinates.
    colorbar_orientation : {"vertical", "horizontal"}
        Orientation of the colourbar.
    colorbar_tick_size : int or None
        Tick label font size for colourbar.
    colorbar_label_size : int or None
        Label font size for colourbar.
    show : bool
        Whether to show the plot.
    save : str
        Save path.
    dpi : int
        DPI for saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    if markers is None:
        markers = adata.var_names.tolist()

    nrows = int(np.ceil(len(markers) / ncols))

    # ---- Plot first (Scanpy creates the figure)
    sc.pl.umap(
        adata,
        color=markers,
        ncols=ncols,
        s=point_size,
        vmax=vmax,
        cmap=cmap,
        colorbar_loc=None,  # disable Scanpy colourbars
        show=False,
        layer=layer
    )

    # ---- Grab Scanpy-created figure
    fig = plt.gcf()

    # ---- Resize figure AFTER creation
    fig.set_size_inches(
        ncols * panel_scale,
        nrows * panel_scale,
    )

    # ---- Clean axes + titles
    for ax, marker in zip(fig.axes, markers):
        ax.set_axis_off()
        ax.set_title(marker, fontsize=title_size)

    # ---- Optional shared colourbar
    if add_colorbar:
        norm = Normalize(vmin=0, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cax = fig.add_axes(colorbar_rect)

        cbar = fig.colorbar(
            sm,
            cax=cax,
            orientation=colorbar_orientation,
        )

        # Tick font size
        if colorbar_tick_size is not None:
            cbar.ax.tick_params(labelsize=colorbar_tick_size)

        # Label
        if colorbar_label is not None:
            cbar.set_label(
                colorbar_label,
                fontsize=colorbar_label_size or title_size,
            )

    # ---- Tight layout (leave room for colourbar if present)
    right = 0.9 if add_colorbar else 1.0
    plt.tight_layout(pad=tight_pad, rect=[0, 0, right, 1])

    if show:
        plt.show()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig


import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

import seaborn as sns
from copy import copy
import os
from collections.abc import Iterable
from pathlib import Path
import matplotlib.pyplot as plt
# plt.switch_backend('module://ipympl.backend_nbagg')
sc.settings.verbosity = 3 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def roi_counts(adata,
               obs = ['Case', 'Group'],
               roi_obs = 'ROI'):

    roi_counts = (
        adata.obs
        .groupby(obs, observed=True)[roi_obs]
        .nunique()
        .reset_index(name='n_ROIs')
    )
    
    return roi_counts


def barplot_by_subgroup_roi_case_averaged(
    adata,
    marker,
    layer=None,
    subgroup_key=None,
    case_key='Case',
    roi_key='ROI',
    subgroup_values=None,
    subgroup_filter=None,
    estimator='mean',          # 'mean' or 'median'
    error='sem',               # 'sem' or 'std'
    average_over_roi=True,
    average_over_case=True,
    figsize=(5, 4),
    palette='tab10',
    ylim=None,
    title=None,
    ylabel=None,
    rotate_xticks=30,
    order=None,
    return_df = False
):
    """
    Bar plot of marker intensity with optional averaging over ROI and Case.
    """

    # Subset data
    adata_sub = adata
    if subgroup_filter is not None:
        adata_sub = adata_sub[
            adata_sub.obs[subgroup_key].str.contains(subgroup_filter)
        ]

    marker_idx = adata_sub.var_names.tolist().index(marker)

    # Build tidy dataframe at cell level
    df = pd.DataFrame({
        subgroup_key: adata_sub.obs[subgroup_key].values,
        case_key: adata_sub.obs[case_key].values,
        roi_key: adata_sub.obs[roi_key].values,
        'value': np.asarray(
            adata_sub.layers[layer][:, marker_idx]
        ).flatten()
    }).dropna()

    # Enforce subgroup order early
    if subgroup_values is not None:
        df[subgroup_key] = pd.Categorical(
            df[subgroup_key],
            categories=subgroup_values,
            ordered=True
        )

    # Helper for aggregation
    def agg_fn(x):
        return x.mean() if estimator == 'mean' else x.median()

    # === Optional averaging steps ===

    # 1) Average within ROI
    if average_over_roi:
        df = (
            df.groupby([subgroup_key, case_key, roi_key], observed=True)
              ['value']
              .apply(agg_fn)
              .reset_index()
        )

    # 2) Average ROIs within each Case
    if average_over_case:
        df = (
            df.groupby([subgroup_key, case_key], observed=True)
              ['value']
              .mean()
              .reset_index()
        )

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x=subgroup_key,
        y='value',
        palette=palette,
        errorbar=error,
        capsize=0.15,
        order=order
    )

    # Labels
    if ylabel is None:
        if average_over_roi and average_over_case:
            ylabel = f'{marker} (ROI → Case averaged)'
        elif average_over_roi:
            ylabel = f'{marker} (ROI averaged)'
        elif average_over_case:
            ylabel = f'{marker} (Case averaged)'
        else:
            ylabel = f'{marker} (cell level)'
    ax.set_ylabel(ylabel)

    if title is None:
        title = f'{marker} by {subgroup_key}'
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel('')
    plt.xticks(rotation=rotate_xticks, ha='right')
    sns.despine()

    plt.tight_layout()
    
    if not return_df:
        return a
    else:
        return ax, df

def stacked_kde_by_subgroup(
    adata,
    marker,
    layer=None,
    subgroup_key=None,
    subgroup_values=None,
    subgroup_filter=None,
    figsize=(6, 8),
    palette='tab10',
    fill=False,
    alpha=0.3,
    linewidth=2,
    bw_adjust=1.0,
    sharex=True,
    xlim=None,
    title=None,
    xlabel=None,
    ylabel='Density',
    ylim=None,
    despine=True,
):
    """
    Plot stacked KDEs (one axis per subgroup) with a shared x-axis.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data.
    marker : str
        Marker / feature name (must be in adata.var_names).
    layer : str
        Layer name to pull values from.
    subgroup_key : str
        obs column defining subgroups.
    subgroup_values : list or None
        Explicit order of subgroup values. If None, inferred from data.
    subgroup_filter : str or None
        If provided, only subgroups containing this string are used.
    figsize : tuple
        Figure size.
    palette : str or list
        Seaborn palette name or list of colors.
    fill : bool
        Whether to fill KDE curves.
    alpha : float
        Transparency for filled KDEs.
    linewidth : float
        Line width of KDE curves.
    bw_adjust : float
        Bandwidth adjustment for KDE.
    sharex : bool
        Share x-axis across subplots.
    xlim : tuple or None
        Explicit x-axis limits (min, max).
    title : str or None
        Figure title. Defaults to "<marker> KDE by subgroup".
    xlabel : str or None
        X-axis label. Defaults to "<marker> (normalized intensity)".
    ylabel : str
        Y-axis label (applied to each subplot).
    despine : bool
        Remove top/right spines.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """

    # Subset data
    adata_sub = adata
    if subgroup_filter is not None:
        adata_sub = adata_sub[
            adata_sub.obs[subgroup_key].str.contains(subgroup_filter)
        ]

    # Marker index
    marker_idx = adata_sub.var_names.tolist().index(marker)

    # Determine subgroup order
    if subgroup_values is None:
        subgroup_values = (
            adata_sub.obs[subgroup_key]
            .dropna()
            .unique()
            .tolist()
        )

    # Colors
    colors = sns.color_palette(palette, n_colors=len(subgroup_values))

    # Create figure
    fig, axes = plt.subplots(
        nrows=len(subgroup_values),
        ncols=1,
        figsize=figsize,
        sharex=sharex
    )

    if len(subgroup_values) == 1:
        axes = [axes]

    # Global x-limits if not provided
    if xlim is None:
        all_vals = np.asarray(
            adata_sub.layers[layer][:, marker_idx]
        ).flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        xlim = (all_vals.min(), all_vals.max())

    # Plot
    for ax, sg, color in zip(axes, subgroup_values, colors):
        mask = adata_sub.obs[subgroup_key] == sg
        
        if layer:
            values = np.asarray(
                adata_sub[mask].layers[layer][:, marker_idx]
            ).flatten()
        else:
            values = np.asarray(
                adata_sub[mask].X[:, marker_idx]
            ).flatten()            
        
        values = values[~np.isnan(values)]

        if len(values) > 1:
            sns.kdeplot(
                values,
                ax=ax,
                color=color,
                fill=fill,
                alpha=alpha if fill else None,
                linewidth=linewidth,
                bw_adjust=bw_adjust,
            )

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(sg, fontsize=9, loc='left')
        ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)
            
        if despine:
            sns.despine(ax=ax)

    # Labels and title
    if xlabel is None:
        xlabel = f'{marker} (normalized intensity)'
    axes[-1].set_xlabel(xlabel, fontsize=10)

    if title is None:
        title = f'{marker} KDE by {subgroup_key}'
    fig.suptitle(title, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, axes

def barplot_thresholded_cell_counts(
    adata,
    thresholds,
    layer=None,
    subgroup_key=None,
    case_key='Case',
    roi_key='ROI',
    subgroup_filter=None,
    plot_populations=('pos',),
    average_over_roi=True,
    average_over_case=True,
    plot_type='bar',          # 'bar' or 'scatter'
    scatter_level='case',     # 'case' or 'roi'
    error='se',
    figsize=(4, 3),
    palette='tab10',
    order=None,
    title=None,
    ylabel=None,
    jitter=0.15,
):
    """
    Plot thresholded cell counts as bar or scatter plots.
    """

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # --- Build base dataframe ---
    df = adata.obs[[subgroup_key, case_key, roi_key]].copy()

    for m, rule in thresholds.items():
        idx = adata.var_names.tolist().index(m)
        vals = np.asarray(adata.layers[layer][:, idx]).flatten()

        if 'pos' in rule:
            df[f'{m}_pos'] = vals > rule['pos']
        if 'neg' in rule:
            df[f'{m}_neg'] = vals < rule['neg']

    # Combine marker rules (AND logic)
    pop_mask = np.ones(len(df), dtype=bool)
    for col in df.columns:
        if col.endswith(('_pos', '_neg')):
            pop_mask &= df[col]

    df['population'] = np.where(pop_mask, 'pos', 'neg')
    df = df[df['population'].isin(plot_populations)]

    if subgroup_filter is not None:
        df = df[df[subgroup_key].str.contains(subgroup_filter)]

    # --- Count cells per ROI ---
    df_counts = (
        df.groupby([subgroup_key, case_key, roi_key, 'population'], observed=True)
          .size()
          .reset_index(name='n_cells')
    )

    # --- Aggregate ---
    if average_over_roi:
        df_counts = (
            df_counts.groupby([subgroup_key, case_key, 'population'], observed=True)
                     ['n_cells']
                     .mean()
                     .reset_index()
        )

    if average_over_case:
        df_counts = (
            df_counts.groupby([subgroup_key, 'population'], observed=True)
                     ['n_cells']
                     .mean()
                     .reset_index()
        )

    # --- Plot ---
    plt.figure(figsize=figsize)

    if plot_type == 'bar':
        ax = sns.barplot(
            data=df_counts,
            x=subgroup_key,
            y='n_cells',
            hue='population' if len(plot_populations) > 1 else None,
            palette=palette,
            errorbar=error,
            order=order,
            capsize=0.15
        )

    elif plot_type == 'scatter':
        # Choose level for scatter
        if scatter_level == 'case' and not average_over_case:
            raise ValueError("scatter_level='case' requires average_over_case=True")
        if scatter_level == 'roi' and not average_over_roi:
            raise ValueError("scatter_level='roi' requires average_over_roi=True")

        ax = sns.stripplot(
            data=df_counts,
            x=subgroup_key,
            y='n_cells',
            hue='population' if len(plot_populations) > 1 else None,
            palette=palette,
            order=order,
            dodge=len(plot_populations) > 1,
            jitter=jitter,
            alpha=0.8
        )

    else:
        raise ValueError("plot_type must be 'bar' or 'scatter'")

    if ylabel is None:
        ylabel = 'Cell count'
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('')
    sns.despine()
    plt.tight_layout()

    return ax, df_counts


# Build color map with alphabetical categories and optional abundance-ordered colors
def set_adata_categorical_colors(
    adata,
    obs_key='population',
    uns_key=None,
    order_colors_by_abundance=False,
    palette=None,
    reuse_existing_colors=False
 ):
    """
    Set categorical order and colors for an AnnData obs column.

    Categories are always set to alphabetical order. Colors can optionally be
    assigned by abundance order, then re-mapped onto alphabetical categories.

    Args:
        adata: AnnData object to modify in-place.
        obs_key: Column in ``adata.obs`` containing the categories.
        uns_key: Key in ``adata.uns`` to store colors (defaults to
            f"{obs_key}_colors").
        order_colors_by_abundance: If True, assign colors by descending
            abundance, then map them onto alphabetical categories; otherwise,
            assign colors in alphabetical order.
        palette: Optional list of colors to use instead of Scanpy defaults.
        reuse_existing_colors: If True, map existing colors (if present) to the
            new alphabetical order; otherwise refresh colors from the chosen palette.

    Returns:
        The input ``adata`` (modified in-place).
    """
    if uns_key is None:
        uns_key = f"{obs_key}_colors"

    # Keep original categories/colors if present
    old_cats = None
    old_colors = None
    if isinstance(adata.obs[obs_key].dtype, pd.CategoricalDtype):
        old_cats = adata.obs[obs_key].cat.categories.astype(str).tolist()
        old_colors = list(adata.uns.get(uns_key, []))
        if len(old_colors) != len(old_cats):
            old_colors = None

    # Convert to str and set alphabetical category order
    series = adata.obs[obs_key].astype(str)
    cat_order = sorted(series.unique().tolist())
    adata.obs[obs_key] = pd.Categorical(series, categories=cat_order, ordered=True)

    # Determine color assignment order
    if order_colors_by_abundance:
        color_order = series.value_counts().index.tolist()
    else:
        color_order = cat_order

    # Choose palette
    if palette is not None:
        colors = palette
    else:
        n = len(color_order)
        if n <= 20:
            colors = sc.pl.palettes.default_20[:n]
        elif n <= 28:
            colors = sc.pl.palettes.default_28[:n]
        else:
            colors = sc.pl.palettes.default_102[:n]

    # Optionally reuse old colors where possible
    if reuse_existing_colors and old_cats is not None and old_colors is not None:
        color_map = dict(zip(old_cats, old_colors))
        colors = [color_map.get(cat, colors[i]) for i, cat in enumerate(color_order)]

    # Map colors to alphabetical categories
    color_map = dict(zip(color_order, colors))
    adata.uns[uns_key] = [color_map.get(cat) for cat in cat_order]
    return adata