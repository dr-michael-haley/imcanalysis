


#### From pop_id.py


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

    import seaborn as sb
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
    import scipy as sp
    import matplotlib.pyplot as plt 

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
        display(cells)

        print('Statistics:')
        display(stats)
        
    grouped_graph.cells = cells     
    grouped_graph.cellslong = cells_long
    grouped_graph.stats = stats  
    
    
def adata_subclustering(adata,
                        population_obs,
                        populations,
                        marker_list,
                        umap_categories=['ROI','Case'],
                        batch_correct='bbknn',
                        batch_correct_obs='Case',
                        clustering=True,
                        clustering_resolutions=[0.3],
                        close_plots=True):
    '''
    OLD CODE, needs updating
    '''

    import scanpy as sc
    import pandas as pd
    import os
    from pathlib import Path
    from copy import copy
    
    # Make populations into a list if only one given
    if not isinstance(populations, list):
        populations=[populations]
        
    pops_str='_'.join(populations)
            
    # Make save directories
    figure_dir=Path('Figures',f'{population_obs}_{pops_str}_Subclustering')
    os.makedirs(figure_dir, exist_ok=True)

    if not isinstance(clustering_resolutions, list):
        clustering_resolutions=[clustering_resolutions]
    
    # Filter AnnData down to specific population 
    adata_new = adata[adata.obs[population_obs].isin(populations), marker_list].copy()
                                                     
    n_for_pca = len(adata_new.var_names)-1
    
    # Batch correction
    if batch_correct=='bbknn':

        # Define the 'obs' which defines the different cases
        batch_correction_obs = batch_correct_obs

        # Calculate PCA, this must be done before BBKNN
        sc.tl.pca(adata_new, n_comps=n_for_pca)

        # BBKNN - it is used in place of the scanpy 'neighbors' command that calculates nearest neighbours in the feature space
        sc.external.pp.bbknn(adata_new, batch_key=batch_correct_obs, n_pcs=n_for_pca)

    else:
        sc.pp.neighbors(adata_new, n_neighbors=10, n_pcs=n_for_pca)
                                                     
    sc.tl.umap(adata_new)
                                                 
    new_pops = []
    
    if clustering:
        
        for c in clustering_resolutions:
        
            pop_key = f'leiden_{str(c)}'
            
            sc.tl.leiden(adata_new, resolution=c, key_added = pop_key)

            new_pops.append(copy(pop_key))

            try:
                del adata.uns[f'dendrogram_{pop_key}']
            except:
                pass
                                    
            fig = sc.pl.matrixplot(adata_new,
                                   adata_new.var_names.tolist(), 
                                   groupby=pop_key, 
                                   dendrogram=True,
                                   return_fig=True)

            fig.savefig(Path(figure_dir, f'Heatmap_{pop_key}_{population_obs}_subanalyses.png'), bbox_inches='tight', dpi=150)

    # Plot UMAPs coloured by list above
    fig = sc.pl.umap(adata_new, color=(umap_categories+new_pops), size=3, return_fig=True)
    fig.savefig(Path(figure_dir, f'Categories_{population_obs}_subanalyses.png'), bbox_inches='tight', dpi=150)

    # This will plot a UMAP for each of the individual markers
    fig = sc.pl.umap(adata_new, color=adata_new.var_names.tolist(), color_map='plasma', ncols=4, return_fig=True)
    fig.savefig(Path(figure_dir, f'Markers_{population_obs}_subanalyses.png'), bbox_inches='tight', dpi=150)
    
    if close_plots:
        plt.close('all')
    
    return adata_new
    
    
def _show_colors_2(color_dict: dict) -> None:
    """
    Display a window to show and choose colors for items (alternative layout).

    Parameters
    ----------
    color_dict : dict
        Dictionary containing the current colors.
    """
    result = tk.Tk()
    result.title("Selected colors")
    result_labels = {}
    for item in color_dict.keys():
        result_labels[item] = tk.Label(result, text=f"{item}:", bg=color_dict[item])
        result_labels[item].pack()
    tk.Button(result, text="OK", command=result.destroy).pack()
    for item in color_dict.keys():
        tk.Button(result, text=item, command=lambda item=item: _choose_colors(color_dict, item, result_labels)).pack()
    result.mainloop()
    
    
def population_connectivity_new(
    nodes: np.ndarray,
    cells: pd.DataFrame,
    X: str,
    Y: str,
    radius: float,
    cluster_col: str,
    population_list: list
) -> np.ndarray:
    """
    Create a network of the window of cells, then return a connectivity score for each population.

    Parameters
    ----------
    nodes : np.ndarray
        Array of node indices.
    cells : pd.DataFrame
        DataFrame containing cell data.
    X : str
        Column name for X coordinates.
    Y : str
        Column name for Y coordinates.
    radius : float
        Radius for neighbors graph.
    cluster_col : str
        Column name for cell cluster information.
    population_list : list
        List of populations.

    Returns
    -------
    np.ndarray
        Array of connectivity scores for each population.
    """
    coords = [(cells.loc[n, X], cells.loc[n, Y]) for n in nodes]
    ndata = pd.DataFrame.from_records(coords, index=nodes)

    df_pops = cells.loc[ndata.index, cluster_col]

    adj = radius_neighbors_graph(
        ndata.to_numpy(), radius=radius, n_jobs=-1, include_self=False
    )

    df = pd.DataFrame(adj.A, index=df_pops, columns=df_pops)
    total_edges_by_pop = (
        df.reset_index(names="population")
        .groupby("population")
        .sum()
        .T.reset_index(names="population")
        .groupby("population")
        .sum()
        .sum(axis=1)
    )
    total_cells_per_pop = df_pops.value_counts()

    average_edges = np.float16(
        [
            0 if x not in total_edges_by_pop.index else total_edges_by_pop[x] / total_cells_per_pop[x]
            for x in population_list
        ]
    )

    return average_edges
