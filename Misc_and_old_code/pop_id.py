import pandas as pd
import numpy as np
import scanpy as sc

from pathlib import Path
import os    
from copy import copy

from utils import adlog

# Matplotlib and seaborn for plotting
import matplotlib
from matplotlib import rcParams
from matplotlib import colors
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sb

# For access to good colour maps
import colorcet as cc

# Colour remapping
import tkinter as tk
from tkinter import colorchooser


sc.set_figure_params(figsize=(5, 5))

# Set up output figure settings
plt.rcParams['figure.figsize']=(5,5) #rescale figures, increase sizehere

# Set up scanpy settings
sc.settings.verbosity = 3
sc.set_figure_params(dpi=100, dpi_save=200) #Increase DPI for better resolution figures



def batch_neighbors(adata,
                    correction_method = 'bbknn',
                    batch_correction_obs='Case',
                    n_for_pca=None,
                    save=True):
    
    '''
    This function does batch correction and preprocessing
    '''
    
    if not n_for_pca:
        # Define the number of PCA dimensions to work with - one less than number of markers. Without this, it usually defaults to 50, which we don't have enough markers for.
        n_for_pca = len(adata.var_names)-1
    
    print(f'Calculating PCA with {str(n_for_pca)} dimensions')
    sc.tl.pca(adata, n_comps=n_for_pca)
    adlog(adata, f'PCA with {str(n_for_pca)} dimensions', sc)
    
    # If we dont use batch correction, use this instead
    if correction_method == 'bbknn':
        
        adlog(adata, f'Starting BBKNN calculations', sc)               
        # BBKNN - it is used in place of the scanpy 'neighbors' command that calculates nearest neighbours in the feature space
        sc.external.pp.bbknn(adata, batch_key=batch_correction_obs, n_pcs=n_for_pca)
        adlog(adata, f'Finished BBKNN batch correction with obs: {batch_correction_obs}', sc)
    
    elif correction_method == 'harmony':
        
        import scanpy.external as sce
        
        adlog(adata, f'Starting Harmony calculations', sc)        
        # Compute Harmony correction
        sce.pp.harmony_integrate(adata, key=batch_correction_obs, basis='X_pca', adjusted_basis='X_pca')
        adlog(adata, f'Finished Harmony batch correction with obs: {batch_correction_obs}', sc)
        
        adlog(adata, f'Starting calculating neighbors', sc)        
        sc.pp.neighbors(adata, use_rep = 'X_pca')
        adlog(adata, f'Finished calculating neighbors', sc)
        
        
    elif correction_method == 'both':

        import scanpy.external as sce

        adlog(adata, f'Starting Harmony calculations', sc)        
        # Compute Harmony correction
        sce.pp.harmony_integrate(adata, key=batch_correction_obs, basis='X_pca', adjusted_basis='X_pca')
        adlog(adata, f'Finished Harmony batch correction with obs: {batch_correction_obs}', sc)    
        
        adlog(adata, f'Starting BBKNN calculations', sc)               
        # BBKNN - it is used in place of the scanpy 'neighbors' command that calculates nearest neighbours in the feature space
        sc.external.pp.bbknn(adata, batch_key=batch_correction_obs, n_pcs=n_for_pca)
        adlog(adata, f'Finished BBKNN batch correction with obs: {batch_correction_obs}', sc)   
    
    else:
        print('No batch correction performed, using scanpy.pp.neighbors')
        adlog(adata, f'Starting calculating neighbors', sc)        
        sc.pp.neighbors(adata, use_rep = 'X_pca')
        adlog(adata, f'Finished calculating neighbors', sc)
        
    adlog(adata, f'Finished PCA and batch correction', save=save)        

        
def leiden(adata, 
           resolution=0.3, 
           leiden_obs_name=None,
           restrict_to_existing_leiden=None,
           existing_leiden_groups=None):
    '''
    Wrapper for performing leiden clustering
    '''
    
    if not leiden_obs_name:
        leiden_obs_name = 'leiden_'+str(resolution)
        
    if not type(existing_leiden_groups)==list:
        existing_leiden_groups = [existing_leiden_groups]
    
    # Catch when someone over writes their old leiden
    
    try:
        adata.obs[leiden_obs_name]
        r = input('That leiden clustering already exists in the AnnData, so any new clustering will overwrite the old results. Respond yes to continue')
        if r != 'yes':
            print('Aborting')
            return None
        else:
            adlog(adata, f'Existing leiden obs {leiden_obs_name} to be overwritten ', sc)
    
    except:
        pass
       
    # Setup restricting to specific groups
    if restrict_to_existing_leiden:
        if not type(existing_leiden_groups)==list:
            existing_leiden_groups = [existing_leiden_groups]
        
        adlog(adata, f'Clustering restricted to groups {str(restrict_to_existing_leiden)} from {existing_leiden_groups}')
        restrict_to = (restrict_to_existing_leiden, existing_leiden_groups)
    else:
        restrict_to = None
        
    adlog(adata, f'Starting Leiden calculations', sc)
   
    # This will perform the clustering, then add an 'obs' with name specified above, e.g leiden_0.35
    sc.tl.leiden(adata, resolution=resolution, key_added = leiden_obs_name, restrict_to=restrict_to)
        
    adlog(adata, f'Finished Leiden. Key added: {leiden_obs_name}', sc, save=True)


def consensus(adata, 
              n_clusters=[3, 5, 10], 
              n_runs=25, 
              d_range=None,
              save=True):
    '''
    Run consensus clustering to cluster cells in an AnnData object.

    This function requires the user to perform dimensionality
    reduction using PCA (`scanpy.tl.pca`) first.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_clusters
        Number of clusters. Default: [3,5,10]
    d_range
        Number of PCs. Default is 25, or the number of PCs in the 
        AnnData object, whichever is lower. Can accept a list
        (e.g. `[15, 20, 25`]).
    n_runs
        Number of realisations to perform for the consensus.
        Default is 5, recommended > 1.
    '''
    
    try:
        import sc3s
    except:
        print('Install sc3s with \'pip install sc3s\'')
        return None
    
    # Make clusters a list
    if not type(n_clusters)==list:
        n_clusters = [n_clusters]
    
    # Check if clusters already exist
    for o in [('sc3s_'+str(x)) for x in n_clusters]:
        try:
            adata.obs[o]
            r = input('That clustering already exists in the AnnData, so any new clustering will overwrite the old results. Respond yes to continue')
            if r != 'yes':
                print('Aborting')
                return None
            else:
                adlog(adata, f'Existing obs {o} to be overwritten ', sc)    
        except:
            pass
    
    import warnings

    adlog(adata, f'Starting SC3s consensus clustering', sc3s)
    
    start_obs = adata.obs.columns.tolist()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        sc3s.tl.consensus(adata, 
                          n_clusters=n_clusters,
                          n_runs=n_runs)         
    
    del adata.uns['sc3s_trials'] #This is deleted because prevents saving
    
    new_obs= [x for x in adata.obs.columns.tolist() if x not in start_obs]
    new_obs = ', '.join(new_obs)
    print(f'New obs added: {new_obs}')

    adlog(adata, f'SC3 clustering. n_clusters: {str(n_clusters)}. n_runs: {str(n_runs)}', sc3s, save=save)    
    

    
    
    
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
    
    
def population_summary(adata,
                       groupby_obs,
                       markers=None,
                       categorical_obs=[],
                       heatmap_vmax=None,
                       graph_log_scale=True,
                       umap_point_size=None,
                       display_tables=False):
    '''
    Produces graphs to summmarise clustering performance 
    '''
    
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=UserWarning)

    
    if type(categorical_obs)==list:
        categorical_obs = ['ROI'] + categorical_obs
    elif type(categorical_obs)==str:
        categorical_obs = ['ROI'] + [categorical_obs]
    
    # Uses all markers if not specified
    if not markers:
        markers = adata.var_names.tolist()
        
    sc.pl.umap(adata, color=groupby_obs, size=umap_point_size)
    
    # Run dendrogram
    sc.tl.dendrogram(adata, groupby_obs, use_rep='X_pca')

    # Plot a heatmap
    sc.pl.matrixplot(adata,
                     markers, 
                     groupby=groupby_obs, 
                     var_group_rotation = 0,
                     vmax=heatmap_vmax,
                     dendrogram=True)
    
    for c in categorical_obs:
        grouped_graph(adata, 
                      group_by_obs=groupby_obs, 
                      x_axis=c,
                      log_scale=graph_log_scale,
                      fig_size=(10,3),
                      display_tables=display_tables)
        
        plt.show()
                       
            
def prune_leiden_using_dendrogram(adata,
                                  leiden_obs,
                                  new_obs='leiden_merged',
                                  mode='max',
                                  max_leiden=None,
                                  minimum_pop_size=None,
                                  save=True):
    '''
    This function will use the results of a dendrogram to reduce the number of leiden populations by merging them into larger and more robust populations
    
    adata = AnnData object
    leiden_obs = Obs that identifies the leiden to use
    new_obs = New obs that will be created in the AnnData
    mode = 'max', 'size' or 'percent'
    
    If max, then 'max_leiden' indicates the largest leiden population number to keep, all others above are targetted for merging
    If 'size', then 'minimum_pop_size' should be a number indicating the mimimum population size to be kept, with any smaller being merged
    If 'percent', then 'minimum_pop_size' is instead a percentage, with any population being a smaller percentage of total number of cells being merged
    
    '''        
    try:
        Z = adata.uns[f'dendrogram_{leiden_obs}']['linkage']
    except:
        print('No dendrogram has been run, running with defaults')
        sc.tl.dendrogram(adata, groupby=leiden_obs, n_pcs=adata.varm['PCs'].shape[1])
        Z = adata.uns[f'dendrogram_{leiden_obs}']['linkage']
        
    # Number of groups in dendrogram
    n = len(adata.obs[leiden_obs].cat.categories)
        
    # clusters of single elements
    clusters = {i: str(i) for i in range(int(n))}
    # loop through Z matrix
    for i, z in enumerate(Z.astype(int)):
        # cluster number
        cluster_num = int(n+i)
        # elements in clusters
        cluster_names = [clusters[z[0]], clusters[z[1]]]
        cluster_elements = [str(i) for i in cluster_names]
        # update the dictionary
        clusters.update({cluster_num: ','.join(cluster_elements)})

    if mode=='max':
        # Create a list of clusters to remove, with any leiden higher than the one we want to keep being removed
        clusters_rmv = [str(x) for x in range(max_leiden + 1, int(adata.obs[leiden_obs].cat.categories[-1])+1)]
        
        percent_removed = round(adata[adata.obs[leiden_obs].isin(clusters_rmv)].shape[0] / adata.shape[0], 3)
        print(f'{str(percent_removed * 100)}% of cells will be remapped by setting upper cluster at {str(max_leiden)}, which has a size of {str(adata.obs[leiden_obs].value_counts()[max_leiden])}')
    
    elif mode=='size':
        # Remove all clusters with a size less than a specified number
        print(f'Removing populations smaller than {str(minimum_pop_size)}')
        clusters_rmv = adata.obs[leiden_obs].value_counts()[adata.obs[leiden_obs].value_counts() < minimum_pop_size].index.tolist()
            
    elif mode=='percent':
        # Remove all clusters with a size less than a specified percent of all cells
        minimum_pop_size = adata.obs[leiden_obs].value_counts().sum() * minimum_pop_size/100
        print(f'Removing populations smaller than {str(minimum_pop_size)}')
        clusters_rmv = adata.obs[leiden_obs].value_counts()[adata.obs[leiden_obs].value_counts() < minimum_pop_size].index.tolist()
        
    elif type(mode)==list:
        # Use a supplied list of clusters
        clusters_rmv=mode
        
    else:
        print('Mode not recognized')
        return None

    # Create new obs to receive data
    adata.obs[new_obs]=adata.obs[leiden_obs].astype('str')

    # We will add to this as we loop through
    remap_dict= {}

    # Make a list of list of leiden clusters in each dendrogram cluster/fork
    cluster_list = [x.split(',') for (_,x) in clusters.items()]

    # For each cluster to remove
    for cr in clusters_rmv:

        # List of forks in which the cluster to remove is found, skipping the first as that will be the group by itself
        target_forks = [x for x in cluster_list if cr in x][1:]

        # List of remaining forks that also contain a cluster to keep
        target_forks_keep = [x for x in target_forks if any(~pd.Series(x).isin(clusters_rmv))]

        # Extract the cluster to keep from the nearest fork
        target_leiden = [x for x in target_forks_keep[0] if x not in clusters_rmv]

        remap_dict.update({cr:target_leiden[0]})

    # Only remap population to be remapped, leave others in their existing groups
    adata.obs[new_obs] = np.where(adata.obs[new_obs].isin(clusters_rmv), 
                                 adata.obs[new_obs].map(remap_dict),
                                 adata.obs[new_obs])
    
    adata.obs[new_obs] = adata.obs[new_obs].astype('category')
    
    entry = f'Pruning population {str(leiden_obs)}, new obs: {str(new_obs)}. Mode: {str(mode)}. Max_leiden: {str(max_leiden)}. Minimum_pop_size: {str(minimum_pop_size)}'
    print(entry)
    
    adlog(adata, entry, save=save)

    return remap_dict


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


def transfer_populations(adata_source,
                         adata_source_populations_obs,
                         adata_target,
                         adata_target_populations_obs,
                         common_cell_index='Master_Index',
                         pop_prefix=''):
    '''
    OLD CODE, needs updating
    '''
    
    remap_dict = dict(zip(adata_source.obs[common_cell_index], adata_source.obs[adata_source_populations_obs]))
    
    new_col_data = adata_target.obs[common_cell_index].map(remap_dict)
    
    new_col_data = pop_prefix + '_' + new_col_data
                      
    try:
        adata_target.obs[adata_target_populations_obs]=np.where(new_col_data.isna(), 
                                                                adata_target.obs[adata_target_populations_obs],
                                                                new_col_data)
    except:
        adata_target.obs[adata_target_populations_obs]=new_col_data        
        
    


def create_remapping(adata,
                     obs_column,
                     groups=['population', 'population_broad', 'hierarchy'],
                     file=None):
    '''
    Creates a mapping file for renaming populations
    '''
    import re
    import pandas as pd
    
    df = pd.DataFrame(data=None, index=adata.obs[obs_column].unique().tolist(), columns=groups)

    df.index.rename(name=obs_column, inplace=True)

    if not file:
        file  = 'remapping_' + re.sub(r'\W+', '', obs_column) + '.csv'
    
    df.to_csv(file)
    print(f'Saved population map file: {file}')
    

    
def read_remapping(adata,
                   obs_column,
                   file=None):
    '''
    Reads a mapping file for renaming populations, and adds the details of the population to the Adata.uns
    '''

    import re
    
    if not file:
        file  = 'remapping_' + re.sub(r'\W+', '', obs_column) + '.csv'
        
    df = pd.read_csv(file, index_col=0)
    
    for c in df.columns.tolist():
        if df[c].isnull().values.any():
            print(f'Column {c} contained null or empty values, will not be used')
            df.drop(columns=c, inplace=True)
        else:
            print(f'Found {c}')
    
    # Make sure everything is a string, as leidens look like numbers
    df = df.astype(str)
    df.index = df.index.astype(str)
    
    df_dict = df.to_dict()
    
    for p in df_dict.keys():
        adata.obs[p] = adata.obs[obs_column].map(df[p])
        adata.obs[p]=adata.obs[p].astype('category')
        
    # Add the liste of pop obs to the anndata
    adata.uns.update({'population_obs':list(df_dict.keys())})
    
    new_pops =', '.join(list(df_dict.keys()))
    
    entry = f'Populations remapped from obs: {obs_column}. New populations: {new_pops}'
    print(entry)
    
    adlog(adata, entry, save=True)
    
    return df

''' Colour remapping functions '''

def choose_colors(color_dict, item, result_labels):
    color = colorchooser.askcolor(title=f"Choose color for {item}", initialcolor=color_dict[item])[1]
    color_dict[item] = color
    result_labels[item].config(bg=color)

def show_colors_2(color_dict):
    result = tk.Tk()
    result.title("Selected colors")
    result_labels = {}
    for item in color_dict.keys():
        result_labels[item] = tk.Label(result, text=f"{item}:", bg=color_dict[item])
        result_labels[item].pack()
    tk.Button(result, text="OK", command=result.destroy).pack()
    for item in color_dict.keys():
        tk.Button(result, text=item, command=lambda item=item: choose_colors(color_dict, item, result_labels)).pack()
    result.mainloop()

def show_colors(color_dict):
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
        tk.Button(button_frame, text=item, command=lambda item=item: choose_colors(color_dict, item, result_labels)).pack(side=tk.TOP)  # Place buttons in a vertical column
    
    button_frame.pack(side=tk.LEFT, padx=10)  # Pack the button frame on the left side with some padding
    
    result.mainloop()
    
    
def recolour_population(adata,
                        population_obs,
                        save=True):
    
    from matplotlib.colors import ListedColormap, Normalize
    import tkinter as tk
    from tkinter import colorchooser
    import scanpy as sc
    
    # Save a backup of anndata
    if save:
        adlog(adata, None, save=True)
    
    global color_dict

    # Try get the existing colour map, if not create default colours according to Scanpy
    try:
        adata.uns[f'{population_obs}_colors']
    except:
        print('Could not retrieve colourmap from AnnData, creating a default colour map')
        sc.plotting._utils._set_default_colors_for_categorical_obs(adata, population_obs)
    
    pop_list = adata.obs[population_obs].cat.categories.tolist()
    color_list = adata.uns[f'{population_obs}_colors']
        
    color_dict = dict(zip(pop_list, color_list))

    root = tk.Tk()
    root.title("Color chooser")
    tk.Button(root, text="Choose colors", command=lambda: show_colors(color_dict)).pack()
    root.mainloop()
    
    color_list = [color_dict[x] for x in pop_list]
    
    adata.uns.update({f'{population_obs}_colors' : color_list})
    adata.uns.update({f'{population_obs}_colormap' : color_dict})
    
    print('New colour map: \n')
    print(pop_list)
    display(ListedColormap(adata.uns[f'{population_obs}_colors'], name=population_obs))
    
    if save:
        adlog(adata, None, save=True)