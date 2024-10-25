# Standard Library Imports
import os
from copy import copy
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser
import warnings
from warnings import simplefilter
import re
from typing import List, Optional, Tuple, Union

# Third-Party Imports
import anndata as ad
import colorcet as cc
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sb
import scipy as sp
from matplotlib import cm, colors, rcParams
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import sc3s
from statsmodels.stats.multitest import multipletests
from IPython.display import display, HTML

# Local Application Imports
from .utils import adlog

# Set up Scanpy settings
sc.set_figure_params(figsize=(5, 5))

# Set up output figure settings
plt.rcParams['figure.figsize'] = (5, 5)  # rescale figures, increase size here

# Set up Scanpy settings
sc.settings.verbosity = 3
sc.set_figure_params(dpi=100, dpi_save=200)  # Increase DPI for better resolution figures


def batch_neighbors(
    adata,
    correction_method: str = 'bbknn',
    batch_correction_obs: str = 'Case',
    n_for_pca: int = None,
    save: bool = True
) -> None:
    """
    Perform batch correction and preprocessing on an AnnData object.

    Parameters
    ----------
    adata : AnnData
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
    save : bool, optional
        Whether to save the results.
        Default is True.

    Returns
    -------
    None

    Notes
    -----
    - If `n_for_pca` is not specified, it is set to one less than the number of markers in `adata.var_names`.
    - The function performs PCA followed by the specified batch correction method.
    - The function logs progress and results using `adlog`.
    """
    
    if not n_for_pca:
        # Define the number of PCA dimensions to work with - one less than number of markers.
        n_for_pca = len(adata.var_names) - 1
    
    print(f'Calculating PCA with {n_for_pca} dimensions')
    sc.tl.pca(adata, n_comps=n_for_pca)
    adlog(adata, f'PCA with {n_for_pca} dimensions', sc)
    
    # Apply the specified batch correction method
    if correction_method == 'bbknn':
        adlog(adata, 'Starting BBKNN calculations', sc)               
        sc.external.pp.bbknn(adata, batch_key=batch_correction_obs, n_pcs=n_for_pca)
        adlog(adata, f'Finished BBKNN batch correction with obs: {batch_correction_obs}', sc)
    
    elif correction_method == 'harmony':
        import scanpy.external as sce
        adlog(adata, 'Starting Harmony calculations', sc)        
        sce.pp.harmony_integrate(adata, key=batch_correction_obs, basis='X_pca', adjusted_basis='X_pca')
        adlog(adata, f'Finished Harmony batch correction with obs: {batch_correction_obs}', sc)
        adlog(adata, 'Starting calculating neighbors', sc)        
        sc.pp.neighbors(adata, use_rep='X_pca')
        adlog(adata, 'Finished calculating neighbors', sc)
        
    elif correction_method == 'both':
        import scanpy.external as sce
        adlog(adata, 'Starting Harmony calculations', sc)        
        sce.pp.harmony_integrate(adata, key=batch_correction_obs, basis='X_pca', adjusted_basis='X_pca')
        adlog(adata, f'Finished Harmony batch correction with obs: {batch_correction_obs}', sc)    
        adlog(adata, 'Starting BBKNN calculations', sc)               
        sc.external.pp.bbknn(adata, batch_key=batch_correction_obs, n_pcs=n_for_pca)
        adlog(adata, f'Finished BBKNN batch correction with obs: {batch_correction_obs}', sc)   
    
    else:
        print('No batch correction performed, using scanpy.pp.neighbors')
        adlog(adata, 'Starting calculating neighbors', sc)        
        sc.pp.neighbors(adata, use_rep='X_pca')
        adlog(adata, 'Finished calculating neighbors', sc)
        
    adlog(adata, 'Finished PCA and batch correction', save=save)   

        
def leiden(
    adata,
    resolution: float = 0.3,
    leiden_obs_name: str = None,
    restrict_to_existing_leiden: str = None,
    existing_leiden_groups: list = None
) -> None:
    """
    Perform Leiden clustering on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    resolution : float, optional
        Resolution parameter for the Leiden algorithm.
        Default is 0.3.
    leiden_obs_name : str, optional
        Name for the observation field to store the Leiden clustering results.
        Default is 'leiden_<resolution>'.
    restrict_to_existing_leiden : str, optional
        Restrict clustering to specific existing Leiden groups.
        Default is None.
    existing_leiden_groups : list, optional
        Specific groups to restrict clustering to. If not a list, it will be converted to a list.
        Default is None.

    Returns
    -------
    None

    Notes
    -----
    - If `leiden_obs_name` is not specified, it defaults to 'leiden_<resolution>'.
    - If `existing_leiden_groups` is not a list, it will be converted to a list.
    - If `restrict_to_existing_leiden` is specified, clustering is restricted to the specified groups.
    - The function logs progress and results using `adlog`.
    """

    if not leiden_obs_name:
        leiden_obs_name = f'leiden_{resolution}'
        
    if not isinstance(existing_leiden_groups, list):
        existing_leiden_groups = [existing_leiden_groups]
    
    # Check if the Leiden clustering already exists in the AnnData object
    try:
        adata.obs[leiden_obs_name]
        response = input('That Leiden clustering already exists in the AnnData. Respond "yes" to continue and overwrite the old results: ')
        if response.lower() != 'yes':
            print('Aborting')
            return
        else:
            adlog(adata, f'Existing Leiden obs {leiden_obs_name} to be overwritten', sc)
    except KeyError:
        pass
    
    # Setup restriction to specific groups
    restrict_to = None
    if restrict_to_existing_leiden:
        adlog(adata, f'Clustering restricted to groups {restrict_to_existing_leiden} from {existing_leiden_groups}')
        restrict_to = (restrict_to_existing_leiden, existing_leiden_groups)
        
    adlog(adata, 'Starting Leiden calculations', sc)
    
    # Perform the Leiden clustering
    sc.tl.leiden(adata, resolution=resolution, key_added=leiden_obs_name, restrict_to=restrict_to)
        
    adlog(adata, f'Finished Leiden. Key added: {leiden_obs_name}', sc, save=True)



def consensus(
    adata,
    n_clusters: list = [3, 5, 10],
    n_runs: int = 25,
    d_range: list = None,
    save: bool = True
) -> None:
    """
    Run consensus clustering to cluster cells in an AnnData object.

    This function requires the user to perform dimensionality
    reduction using PCA (`scanpy.tl.pca`) first.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_clusters : list, optional
        Number of clusters. Default is [3, 5, 10].
    n_runs : int, optional
        Number of realizations to perform for the consensus. Default is 25.
    d_range : list, optional
        Number of PCs. Default is 25, or the number of PCs in the 
        AnnData object, whichever is lower. Can accept a list
        (e.g. [15, 20, 25]).
    save : bool, optional
        Whether to save the results. Default is True.

    Returns
    -------
    None

    Notes
    -----
    - If `n_clusters` is not a list, it will be converted to a list.
    - The function logs progress and results using `adlog`.
    """

    try:
        import sc3s
    except ImportError:
        print("Install sc3s with 'pip install sc3s'")
        return

    # Ensure n_clusters is a list
    if not isinstance(n_clusters, list):
        n_clusters = [n_clusters]

    # Check if clusters already exist
    for o in [f'sc3s_{x}' for x in n_clusters]:
        try:
            adata.obs[o]
            response = input(f'That clustering already exists in the AnnData. Respond "yes" to continue and overwrite the old results: ')
            if response.lower() != 'yes':
                print('Aborting')
                return
            else:
                adlog(adata, f'Existing obs {o} to be overwritten', sc)    
        except KeyError:
            pass
    
    adlog(adata, 'Starting SC3s consensus clustering', sc3s)
    
    start_obs = adata.obs.columns.tolist()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        sc3s.tl.consensus(
            adata, 
            n_clusters=n_clusters,
            n_runs=n_runs
        )        
    
    # Remove unnecessary key to allow saving
    del adata.uns['sc3s_trials']
    
    new_obs = [x for x in adata.obs.columns.tolist() if x not in start_obs]
    new_obs_str = ', '.join(new_obs)
    print(f'New obs added: {new_obs_str}')

    adlog(adata, f'SC3 clustering. n_clusters: {n_clusters}. n_runs: {n_runs}', sc3s, save=save)
 

    
def population_summary(
    adata,
    groupby_obs: str,
    markers: list = None,
    categorical_obs: list = [],
    heatmap_vmax: float = None,
    graph_log_scale: bool = True,
    umap_point_size: float = None,
    display_tables: bool = False
) -> None:
    """
    Produces graphs to summarize clustering performance.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby_obs : str
        Observation field to group by.
    markers : list, optional
        List of markers to use. If not specified, all markers are used.
    categorical_obs : list or str, optional
        List of categorical observations. Default is an empty list.
    heatmap_vmax : float, optional
        Maximum value for the heatmap.
    graph_log_scale : bool, optional
        Whether to use logarithmic scale for graphs. Default is True.
    umap_point_size : float, optional
        Size of points in UMAP plot.
    display_tables : bool, optional
        Whether to display tables. Default is False.

    Returns
    -------
    None

    Notes
    -----
    - If `markers` is not specified, all markers in `adata.var_names` are used.
    - Logs and visualizations are generated using `scanpy` plotting functions.
    """

    # Ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=UserWarning)

    # Ensure categorical_obs is a list and prepend 'ROI'
    if isinstance(categorical_obs, list):
        categorical_obs = ['ROI'] + categorical_obs
    elif isinstance(categorical_obs, str):
        categorical_obs = ['ROI'] + [categorical_obs]
    
    # Use all markers if not specified
    if markers is None:
        markers = adata.var_names.tolist()
        
    # Plot UMAP with specified grouping
    sc.pl.umap(adata, color=groupby_obs, size=umap_point_size)
    
    # Run and plot dendrogram
    sc.tl.dendrogram(adata, groupby_obs, use_rep='X_pca')
    sc.pl.matrixplot(
        adata,
        markers,
        groupby=groupby_obs,
        var_group_rotation=0,
        vmax=heatmap_vmax,
        dendrogram=True
    )
    
    for c in categorical_obs:
        grouped_graph(
            adata,
            group_by_obs=groupby_obs,
            x_axis=c,
            log_scale=graph_log_scale,
            fig_size=(10, 3),
            display_tables=display_tables
        )
        plt.show()

def grouped_graph(adata_plotting, 
                  group_by_obs, 
                  x_axis, 
                  ROI_id='ROI', 
                  display_tables=True, 
                  fig_size=(5,5), 
                  confidence_interval=68, 
                  save=False, 
                  log_scale=True, 
                  order=False,
                  scale_factor=False,
                  crosstab_norm=False):
    
    '''
    TO UDPATE
    Old function for plotting graphs for populations
    '''

    # Create cells table    
    
    print(x_axis)
    if not x_axis==ROI_id:
        cells = pd.crosstab([adata_plotting.obs[group_by_obs], adata_plotting.obs[ROI_id]],adata_plotting.obs[x_axis],normalize=crosstab_norm)
        cells.columns=cells.columns.astype('str')        
        cells_long = cells.reset_index().melt(id_vars=[group_by_obs,ROI_id])
    else:    
        cells = pd.crosstab(adata_plotting.obs[group_by_obs],adata_plotting.obs[x_axis],normalize=crosstab_norm)
        cells.columns=cells.columns.astype('str')   
        cells_long = cells.reset_index().melt(id_vars=group_by_obs)

    grouped_graph.cells = cells  
    grouped_graph.cellslong = cells_long

    if scale_factor:
        cells_long['value'] = cells_long['value'] / scale_factor
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    if order:
        sb.barplot(data = cells_long, y = "value", x = x_axis, hue = group_by_obs, ci=confidence_interval, order=order, ax=ax)
    else:
        sb.barplot(data = cells_long, y = "value", x = x_axis, hue = group_by_obs, ci=confidence_interval, ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 10)
    
    if scale_factor:
        ax.set_ylabel('Cells/mm2')
    else:
        ax.set_ylabel('Cells')        
                  
    if log_scale:
        ax.set_yscale("log")
        
    ax.set_xlabel(x_axis)
    ax.legend(bbox_to_anchor=(1.01, 1))

    #fig = ax.get_figure()

    if save:
        fig.savefig(save, bbox_inches='tight',dpi=200)

    col_names = adata_plotting.obs[group_by_obs].unique().tolist()

    data_frame = cells.reset_index()

    celltype = []
    ttest = []
    mw = []

    for i in cells.columns.tolist():
        celltype.append(i)
        ttest.append(sp.stats.ttest_ind(cells.loc[col_names[0]][i], cells.loc[col_names[1]][i]).pvalue) 
        mw.append(sp.stats.mannwhitneyu(cells.loc[col_names[0]][i], cells.loc[col_names[1]][i]).pvalue)

    stats = pd.DataFrame(list(zip(celltype,ttest,mw)),columns = ['Cell Type','T test','Mann-Whitney'])

    import statsmodels as sm

    #Multiple comparissons correction
    for stat_column in ['T test','Mann-Whitney']:
        corrected_stats = multipletests(stats[stat_column],alpha=0.05,method='holm-sidak')
        stats[(stat_column+' Reject null?')]=corrected_stats[0]
        stats[(stat_column+' Corrected Pval')]=corrected_stats[1]

    if display_tables:
        print('Raw data:')
        display(HTML(cells.to_html()))

        print('Statistics:')
        display(HTML(stats.to_html()))
                               
            
def prune_leiden_using_dendrogram(
    adata: ad.AnnData,
    leiden_obs: str,
    new_obs: str = 'leiden_merged',
    mode: str = 'max',
    max_leiden: int = None,
    minimum_pop_size: int = None
) -> dict:
    """
    Use the results of a dendrogram to reduce the number of Leiden populations by merging them into larger and more robust populations.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    leiden_obs : str
        Observation identifying the Leiden clusters to use.
    new_obs : str, optional
        New observation to be created in the AnnData. Default is 'leiden_merged'.
    mode : str, optional
        Mode for merging clusters. Options are 'max', 'size', or 'percent'. Default is 'max'.
    max_leiden : int, optional
        Largest Leiden population number to keep when mode is 'max'. Default is None.
    minimum_pop_size : int, optional
        Minimum population size to keep when mode is 'size' or 'percent'. Default is None.

    Returns
    -------
    dict
        Dictionary mapping old clusters to new clusters.
    """
    try:
        Z = adata.uns[f'dendrogram_{leiden_obs}']['linkage']
    except KeyError:
        print('No dendrogram has been run, running with defaults')
        sc.tl.dendrogram(adata, groupby=leiden_obs, n_pcs=adata.varm['PCs'].shape[1])
        Z = adata.uns[f'dendrogram_{leiden_obs}']['linkage']

    n = len(adata.obs[leiden_obs].cat.categories)
    clusters = {i: str(i) for i in range(n)}

    for i, z in enumerate(Z.astype(int)):
        cluster_num = n + i
        cluster_names = [clusters[z[0]], clusters[z[1]]]
        clusters[cluster_num] = ','.join(cluster_names)

    if mode == 'max':
        clusters_rmv = [str(x) for x in range(max_leiden + 1, int(adata.obs[leiden_obs].cat.categories[-1]) + 1)]
        percent_removed = round(adata[adata.obs[leiden_obs].isin(clusters_rmv)].shape[0] / adata.shape[0], 3)
        print(f'{percent_removed * 100}% of cells will be remapped by setting upper cluster at {max_leiden}, which has a size of {adata.obs[leiden_obs].value_counts()[max_leiden]}')
    elif mode == 'size':
        clusters_rmv = adata.obs[leiden_obs].value_counts()[adata.obs[leiden_obs].value_counts() < minimum_pop_size].index.tolist()
    elif mode == 'percent':
        minimum_pop_size = adata.obs[leiden_obs].value_counts().sum() * minimum_pop_size / 100
        clusters_rmv = adata.obs[leiden_obs].value_counts()[adata.obs[leiden_obs].value_counts() < minimum_pop_size].index.tolist()
    elif isinstance(mode, list):
        clusters_rmv = mode
    else:
        print('Mode not recognized')
        return None

    adata.obs[new_obs] = adata.obs[leiden_obs].astype('str')
    remap_dict = {}
    cluster_list = [x.split(',') for x in clusters.values()]

    for cr in clusters_rmv:
        target_forks = [x for x in cluster_list if cr in x][1:]
        target_forks_keep = [x for x in target_forks if any(~pd.Series(x).isin(clusters_rmv))]
        target_leiden = [x for x in target_forks_keep[0] if x not in clusters_rmv]
        remap_dict[cr] = target_leiden[0]

    adata.obs[new_obs] = np.where(adata.obs[new_obs].isin(clusters_rmv), adata.obs[new_obs].map(remap_dict), adata.obs[new_obs])
    adata.obs[new_obs] = adata.obs[new_obs].astype('category')

    return remap_dict



def transfer_populations(
    adata_source,
    adata_source_populations_obs: str,
    adata_target,
    adata_target_populations_obs: str,
    common_cell_index: str = 'Master_Index',
    pop_prefix: str = ''
) -> None:
    """
    Transfer populations from a source AnnData object to a target AnnData object based on a common cell index.

    Parameters
    ----------
    adata_source : AnnData
        Source AnnData object containing the populations to be transferred.
    adata_source_populations_obs : str
        Observation field in the source AnnData object that contains the population information.
    adata_target : AnnData
        Target AnnData object where the populations will be transferred to.
    adata_target_populations_obs : str
        Observation field in the target AnnData object where the population information will be stored.
    common_cell_index : str, optional
        Common cell index used to match cells between the source and target AnnData objects.
        Default is 'Master_Index'.
    pop_prefix : str, optional
        Prefix to add to the population names in the target AnnData object.
        Default is an empty string.

    Returns
    -------
    None
    """

    # Create a mapping dictionary from the source populations
    remap_dict = dict(zip(adata_source.obs[common_cell_index], adata_source.obs[adata_source_populations_obs]))

    # Map the new population data to the target AnnData object
    new_col_data = adata_target.obs[common_cell_index].map(remap_dict)

    # Add the prefix to the new population data
    new_col_data = pop_prefix + '_' + new_col_data.astype(str)

    # Transfer the population data to the target AnnData object
    adata_target.obs[adata_target_populations_obs] = np.where(
        new_col_data.isna(),
        adata_target.obs[adata_target_populations_obs],
        new_col_data
    )
 
        

def create_remapping(
    adata,
    obs_column: str,
    groups: list = ['population', 'population_broad', 'hierarchy'],
    file: str = None
) -> None:
    """
    Creates a mapping file for renaming populations.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_column : str
        Observation column containing the populations to be remapped.
    groups : list, optional
        List of group names for the mapping. Default is ['population', 'population_broad', 'hierarchy'].
    file : str, optional
        File name to save the mapping file. If not provided, a default name based on `obs_column` is used.

    Returns
    -------
    None
    """

    # Create a DataFrame with unique values from the observation column
    unique_values = adata.obs[obs_column].unique().tolist()
    df = pd.DataFrame(data=None, index=unique_values, columns=groups)

    # Rename the index to match the observation column
    df.index.rename(name=obs_column, inplace=True)

    # Determine the file name if not provided
    if not file:
        file = 'remapping_' + re.sub(r'\W+', '', obs_column) + '.csv'
    
    # Save the DataFrame to a CSV file
    df.to_csv(file)
    print(f'Saved population map file: {file}')

    

def read_remapping(
    adata,
    obs_column: str,
    file: str = None
) -> pd.DataFrame:
    """
    Reads a mapping file for renaming populations and adds the details of the population to `adata.uns`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_column : str
        Observation column containing the populations to be remapped.
    file : str, optional
        File name to read the mapping from. If not provided, a default name based on `obs_column` is used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the remapping information.
    """

    import re
    import pandas as pd

    if not file:
        file = 'remapping_' + re.sub(r'\W+', '', obs_column) + '.csv'
        
    df = pd.read_csv(file, index_col=0)
    
    for c in df.columns.tolist():
        if df[c].isnull().values.any():
            print(f'Column {c} contains null or empty values and will not be used')
            df.drop(columns=c, inplace=True)
        else:
            print(f'Found {c}')
    
    # Ensure all values are strings, as Leidens may look like numbers
    df = df.astype(str)
    df.index = df.index.astype(str)
    
    df_dict = df.to_dict()
    
    for p in df_dict.keys():
        adata.obs[p] = adata.obs[obs_column].map(df[p])
        adata.obs[p] = adata.obs[p].astype('category')
        
    # Add the list of population observations to AnnData
    adata.uns.update({'population_obs': list(df_dict.keys())})
    
    new_pops = ', '.join(list(df_dict.keys()))
    entry = f'Populations remapped from obs: {obs_column}. New populations: {new_pops}'
    print(entry)
    
    adlog(adata, entry, save=True)
    
    return df

def _choose_colors(color_dict: dict, item: str, result_labels: dict) -> None:
    """
    Allow the user to choose a color for a specific item.

    Parameters
    ----------
    color_dict : dict
        Dictionary containing the current colors.
    item : str
        The item for which the color is being chosen.
    result_labels : dict
        Dictionary containing label widgets to update the background color.
    """
    color = colorchooser.askcolor(title=f"Choose color for {item}", initialcolor=color_dict[item])[1]
    color_dict[item] = color
    result_labels[item].config(bg=color)

def _show_colors(color_dict: dict) -> None:
    """
    Display a window to show and choose colors for items.

    Parameters
    ----------
    color_dict : dict
        Dictionary containing the current colors.
    """
    result = tk.Tk()
    result.title("Selected colors")
    result_labels = {}
    label_frame = tk.Frame(result)  # Create a frame to hold the labels
    button_frame = tk.Frame(result)  # Create a frame to hold the buttons
    
    for item in color_dict.keys():
        result_labels[item] = tk.Label(label_frame, text=f"{item}:", bg=color_dict[item])
        result_labels[item].pack(side=tk.TOP, anchor=tk.W)  # Place labels in a vertical column aligned to the left
    
    label_frame.pack(side=tk.LEFT)  # Pack the label frame on the left side
    
    tk.Button(button_frame, text="OK", command=result.destroy, bg="white").pack(side=tk.TOP)  # Place the OK button in a vertical column on top
    for item in color_dict.keys():
        tk.Button(button_frame, text=item, command=lambda item=item: _choose_colors(color_dict, item, result_labels)).pack(side=tk.TOP)  # Place buttons in a vertical column
    
    button_frame.pack(side=tk.LEFT, padx=10)  # Pack the button frame on the left side with some padding
    
    result.mainloop()


def recolor_population(
    adata,
    population_obs: str,
    save: bool = True
) -> None:
    """
    Allow the user to recolor populations in an AnnData object using a GUI.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    population_obs : str
        Observation field containing the populations to be recolored.
    save : bool, optional
        Whether to save the AnnData object before and after recoloring. Default is True.

    Returns
    -------
    None
    """
    
    # Save a backup of AnnData
    if save:
        adlog(adata, None, save=True)
    
    global color_dict

    # Try to get the existing color map, if not create default colors according to Scanpy
    try:
        adata.uns[f'{population_obs}_colors']
    except KeyError:
        print('Could not retrieve color map from AnnData, creating a default color map')
        sc.plotting._utils._set_default_colors_for_categorical_obs(adata, population_obs)
    
    pop_list = adata.obs[population_obs].cat.categories.tolist()
    color_list = adata.uns[f'{population_obs}_colors']
        
    color_dict = dict(zip(pop_list, color_list))

    root = tk.Tk()
    root.title("Color chooser")
    tk.Button(root, text="Choose colors", command=lambda: _show_colors(color_dict)).pack()
    root.mainloop()
    
    color_list = [color_dict[x] for x in pop_list]
    
    adata.uns.update({f'{population_obs}_colors': color_list})
    adata.uns.update({f'{population_obs}_colormap': color_dict})
    
    print('New color map: \n')
    print(pop_list)
    display(ListedColormap(adata.uns[f'{population_obs}_colors'], name=population_obs))
    
    if save:
        adlog(adata, None, save=True)