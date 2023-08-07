import pandas as pd
import numpy as np
import seaborn as sns
import anndata as ad
import os
import shutil

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.pyplot import get_cmap

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def run_spoox(adata, 
             population_obs, 
             groupby=None, 
             samples=None,
             specify_functions=None,
             spoox_output_dir='spooxout', 
             spoox_output_summary_dir='spooxout_summary', 
             output_file='stats.txt',
             index_obs='Master_Index',
             roi_obs='ROI',
             xloc_obs='X_loc',
             yloc_obs='Y_loc',
             masks_source_directory='masks',
             masks_destination_directory='spoox_mask',
             run_analysis=True,
             analyse_samples_together=True,
             summary=True):
    
    '''
    Run the SpOOx analyis pipeline from an AnnData object.
    More information can be found here: https://github.com/Taylor-CCB-Group/SpOOx/tree/main/src/spatialstats
    
    Any errors in running functions will be saved in errors.csv

    Parameters
    ----------
    adata
        AnnData object, or path (string) to a saved h5ad file.
    population_obs
        The .obs that defines the population for each cell
    groupby
        If specifed, should be a .obs that identifies different groups in the data.
        In the summary step, it will then compare 
    samples
        Specify a list of samples, if None will process all samples
    specify_functions
        By default will run the follow functions from the Spoox pipeline: paircorrelationfunction morueta-holme networkstatistics 
        This is a complete list that will be run if 'all' is used: 
            paircorrelationfunction
            localclusteringheatmaps
            celllocationmap
            contourplots
            quadratcounts
            quadratcelldistributions
            morueta-holme
            networkstatistics
    run_analysis
        Whether or not to run the analyis, or just create the spatialstats file
    analyse_samples_together
        Whether to analyse all samples together
    summary
        Whether to run summary script
        
    Returns
    ----------
    Creates two folders with outputs of SpOOx pipeline
        
    '''
    
    # Load from file if given a string
    if type(adata) == str: 
        adata = ad.read_h5ad(adata)
        
    if not samples:
        samples = adata.obs[roi_obs].unique().tolist()
        print(f'Following samples found in {roi_obs}:')
        print(samples)
    else:
        print(f'Only analysing these samples from {roi_obs}:')
        print(samples)
        
    # Specify functions to run, by default will run all
    if specify_functions=='all':
        functions=''
    elif specify_functions:
        functions=f' -f {specify_functions}'    
    else:
        functions=' -f paircorrelationfunction morueta-holme networkstatistics'

    # Copy over masks into correct format
    if os.path.isdir(masks_destination_directory):
        print('Spoox_masks directory already exists')
    else:
    
        # Create a copy of the original directory
        shutil.copytree(masks_source_directory, masks_destination_directory)

        # Get the list of files in the copied directory
        files = os.listdir(masks_destination_directory)

        for filename in files:
            if os.path.isfile(os.path.join(masks_destination_directory, filename)):
                # Create a subdirectory with the same name as the file
                subdir = os.path.join(masks_destination_directory, os.path.splitext(filename)[0])
                os.makedirs(subdir, exist_ok=True)

                # Move the file to the subdirectory and rename it
                new_filepath = os.path.join(subdir, "deepcell.tif")
                shutil.move(os.path.join(masks_destination_directory, filename), new_filepath)
                #print(f"Moved file: {filename} -> {new_filepath}")
        
        print(f'Created spoox compatible masks in directory: {masks_destination_directory}')
    
    # Create output folders
    os.makedirs(spoox_output_dir, exist_ok = True)
    os.makedirs(spoox_output_summary_dir, exist_ok = True)

    spatial_stats = adata.obs.copy()
    
    # Create spatial stats dataframe from adata.obs
    cols = [index_obs, roi_obs,xloc_obs, yloc_obs, population_obs]
    cols_rename = {index_obs:'cellID',roi_obs:'sample_id', xloc_obs:'x', yloc_obs:'y', population_obs:'cluster'}
    #print(cols)
    
    if groupby:
        cols.append(str(groupby))
        cols_rename.update({groupby: 'Conditions'})
    
   # print(cols)
    spatial_stats = spatial_stats[cols]
    spatial_stats = spatial_stats.rename(columns=cols_rename)
        

    # This may not be neded, just 'label'
    #spatial_stats.cellID = spatial_stats.cellID.astype('int') + 1

    spatial_stats.cellID = 'ID_' + spatial_stats.cellID

    # Add a label column that will match cell numbers to their labels in the mask files (hopefully!)
    for i in spatial_stats.sample_id.unique().tolist():

        df_location = spatial_stats.loc[spatial_stats.sample_id==i, 'cellID']
        spatial_stats.loc[spatial_stats.sample_id==i, 'label'] = [int(x+1) for x in range(0,df_location.shape[0])]

    spatial_stats['label'] = spatial_stats['label'].astype('int')

    spatial_stats.to_csv(output_file, sep='\t')
    print(f'Saved to file: {output_file}')

    ### Create conditions file
    ################

    if groupby:
    
        result = {}

        for roi, niches in zip(adata.obs[~adata.obs.duplicated(roi_obs)][roi_obs], adata.obs[~adata.obs.duplicated(roi_obs)][groupby]):
            if niches in result:
                result[niches].append(roi)
            else:
                result[niches] = [roi]

        formatted_result = {"conditions": result}
        print(formatted_result)


        import json

        with open('conditions.json', 'w') as json_file:
            json.dump(formatted_result, json_file, indent=1)              
    
    
    
    ### Run analysis
    ################
    
    if run_analysis:
    
        if not analyse_samples_together:
        
            # Run each sample individually
            for s in samples:

                spatial_stats_sample = spatial_stats[spatial_stats.sample_id==s]
                spatial_stats_sample.to_csv('sample.txt', sep='\t')
                print(f'Running sample {s}...')

                #command = "python spatialstats\\spatialstats.py -i stats.txt -o spooxout -d nf2_masks -cl cluster -f quadratcounts quadratcelldistributions paircorrelationfunction morueta-holme networkstatistics"
                command = f"python spatialstats\\spatialstats.py -i sample.txt -o {spoox_output_dir} -d {masks_destination_directory} -cl cluster{functions}"

                os.system(command)
        else:
            
            #Run all samples together
            spatial_stats_sample = spatial_stats[spatial_stats.sample_id.isin(samples)]
            spatial_stats_sample.to_csv('sample.txt', sep='\t')
            print(f'Running {str(len(samples))} samples...')

            #command = "python spatialstats\\spatialstats.py -i stats.txt -o spooxout -d nf2_masks -cl cluster -f quadratcounts quadratcelldistributions paircorrelationfunction morueta-holme networkstatistics"
            command = f"python spatialstats\\spatialstats.py -i sample.txt -o {spoox_output_dir} -d {masks_destination_directory} -cl cluster{functions}"

            os.system(command)            
                      
        
        if groupby:

            print('Calculating group averages...')

            command = f"python spatialstats\\average_by_condition.py -i stats.txt -p {spoox_output_dir} -o {spoox_output_summary_dir} -cl cluster -j conditions.json"

            os.system(command)
            
            
        if summary:
        
            print('Summarising data...')

            command = f'python spatialstats\\summary.py -p {spoox_output_dir}'

            os.system(command)
            
            
            import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

def _validate_inputs(data, cols_of_interest):
    """
    Validate the input data to make sure it contains the required columns.

    Parameters:
        data (DataFrame): Input data to validate.
        cols_of_interest (list): The columns that should be present in the data.

    Returns:
        None. Raises a ValueError if a column is missing.
    """
    for col in cols_of_interest:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in input data.")
    # If the function completes without raising an error, the input data is valid.

def _create_heatmap(data, states, col, vmin, vmax, norm, cluster_mh, cmap, figsize, save_folder, save_extension):
    """
    Create a heatmap for a specific column.

    Parameters:
        data (DataFrame): The data to create the heatmap from.
        states (list): List of unique states in the data.
        col (str): The column to create the heatmap for.
        vmin (float): The minimum value for the colormap.
        vmax (float): The maximum value for the colormap.
        norm (Normalize): The normalizer for the colormap.
        cluster_mh (bool): Whether to cluster the 'Morueta-Holme' column.
        cmap (Colormap): The colormap to use for the heatmap.
        figsize (tuple): The size of the figure for the heatmap.
        save_folder (str): The folder to save the heatmap in.
        save_extension (str): The file extension to use for the saved heatmap.

    Returns:
        None. The heatmap is displayed and saved to a file.
    """
    fig, axs = plt.subplots(1, len(states), figsize=(len(states)*figsize, figsize))
    
    # In case only one state is found
    if not hasattr(axs, '__iter__'):
        axs = [axs]
    
    fig.suptitle(f"Heatmaps for analysis: {col}", fontsize=16, y=1.02)

    # Fill NaN values if clustering is enabled.
    if cluster_mh:
        data[col] = data[col].fillna(0)
    
    # If the column is not 'Morueta-Holme' or if clustering is enabled, create a clustermap.
    if 'Morueta-Holme' in col or cluster_mh:
        first_state_data = data[data['state'] == states[0]]
        heatmap_data_first_state = first_state_data.pivot(index='Cell Type 1', columns='Cell Type 2', values=col)
        g = sns.clustermap(heatmap_data_first_state, cmap=cmap, robust=True, figsize=(10, 10))
        plt.close(g.fig)

        # Get the order of rows and columns from the clustermap.
        row_order = g.dendrogram_row.reordered_ind
        col_order = g.dendrogram_col.reordered_ind

    # Create a heatmap for each state.
    for ax, state in zip(axs, states):
        state_data = data[data['state'] == state]
        heatmap_data = state_data.pivot(index='Cell Type 1', columns='Cell Type 2', values=col)

        # Reorder the rows and columns according to the clustermap if the column is not 'MH' or if clustering is enabled.
        if 'Morueta-Holme' not in col or cluster_mh:
            heatmap_data = heatmap_data.iloc[row_order, col_order]

        cbar_kws = {'fraction':0.046, 'pad':0.04}

        # Generate the heatmap.
        if norm:
            sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, robust=True, square=True, vmin=vmin, vmax=vmax, norm=norm, cbar_kws=cbar_kws)
        else:
            sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, robust=True, square=True, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws)

        ax.set_title(state)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'heatmap_{col}{save_extension}'), bbox_inches='tight', dpi=400)
    plt.show()


def create_spoox_heatmaps(data_input, percentile=95, sig_threshold=0.05, cluster_mh=True, save_folder='spoox_figures', save_extension='.png', figsize=10):
    """
    Creates heatmaps from the SpOOx sumary data
    
    Parameters:
        data_input (DataFrame): Pandas dataframe of loaded SpOOx summary data.
        percentile (float): Percentile to determine vmax for heatmap color scaling. Default is 95.
        sig_threshold (float): Threshold for significance. Default is 0.05.
        cluster_mh (bool): If True, the Morueta-Holme column is clustered. Default is True.
        save_folder (str): Folder to save the generated heatmaps. Default is 'spoox_figures'.
        save_extension (str): File extension for the saved heatmaps. Default is '.png'.
        figsize (int): Size of the figure for the heatmaps. Default is 10.
        
    Returns:
        None. The function saves heatmap figures in the specified folder.
    """
    cols_of_interest = ['gr10 PCF lower', 'gr20 PCF lower', 'Morueta-Holme_Significant', 'Morueta-Holme_All', 'contacts', '%contacts', 'Network', 'Network(%)']

    # Create output folder if it doesn't exist.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    data = data_input.copy()
    data['Morueta-Holme_Significant'] = np.where(data['MH_FDR']<sig_threshold, data['MH_SES'], np.nan)
    data['Morueta-Holme_All'] = data['MH_SES']
    
    # Validate the input data.
    _validate_inputs(data, cols_of_interest)    
    
    states = data['state'].unique()

    for col in cols_of_interest:
        
        cmap = get_cmap("Reds")
        cmap.set_under("darkgrey")
        
        vmax = np.percentile(data[col].dropna(), percentile)

        if col in ['gr10 PCF lower', 'gr20 PCF lower']:
            vmin = 1
            norm = None
            
        elif 'Morueta-Holme' in col:
            vmax = np.percentile(data[col].dropna(), percentile)
            vmin = np.percentile(data[col].dropna(), (100-percentile))
            cmap = get_cmap("coolwarm")
            
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            vmin = 0
            norm = None
        
        # Call the function to create the heatmap for the current column.
        _create_heatmap(data, states, col, vmin, vmax, norm, cluster_mh, cmap, figsize, save_folder, save_extension)

        print(f"Saved heatmap for column '{col}' in folder '{save_folder}'")


def _apply_filters(dataframe, filters):
    """
    Apply filters to the dataframe.

    Args:
        dataframe: A pandas DataFrame containing the cell interaction data output.
        filters: A dictionary with the filters to apply.

    Returns:
        The filtered dataframe.
    """
    for column, value in filters.items():
        if isinstance(value, tuple) or isinstance(value, list):
            dataframe = dataframe[(dataframe[column] >= value[0]) & (dataframe[column] <= value[1])]
        elif column.endswith('_greater'):
            dataframe = dataframe[dataframe[column.replace('_greater', '')] >= value]
        elif column.endswith('_less'):
            dataframe = dataframe[dataframe[column.replace('_less', '')] <= value]

    return dataframe


def _create_color_map(dataframe, color_map):
    """
    Create a color map for cell types.

    Args:
        dataframe: A pandas DataFrame containing the cell interaction data output.
        color_map: A dictionary mapping cell types to colors. If None, a default color map is generated.

    Returns:
        A dictionary mapping cell types to colors.
    """
    if color_map is None:
        unique_cell_types = pd.unique(dataframe[['Cell Type 1', 'Cell Type 2']].values.ravel())
        cmap = plt.get_cmap('tab20')
        color_map = {cell_type: cmap(i % cmap.N) for i, cell_type in enumerate(unique_cell_types)}

    return color_map


def _generate_graph(state_data, node_color_map, layout_scale, layout_type, edge_weight_column, edge_color_column, node_scale, node_area_scale, center_cell_population, force_centre, edge_scale):
    """
    Generate a network graph for a given state.

    Args:
        state_data: A pandas DataFrame containing the cell interaction data for a specific state.
        node_color_map: A dictionary mapping cell types to colors.
        layout_scale: A scale factor for the node layout.
        layout_type: The type of layout to use for the graph.
        edge_weight_column: The column from the dataframe that defines the edge weights.
        edge_color_column: The column from the dataframe that defines the edge colors.
        node_scale: A value to scale the node sizes by.
        node_area_scale: Scale node sizes so that the area correlates with population abundance, rather than node diameter.
        center_cell_population: If specified, only nodes immediately connected to this node are visualised.
        force_centre: If True, force the center_cell_population node to be at the centre of the graph.
        edge_scale: A value to scale the edge weights by.

    Returns:
        A networkx Graph object and the positions of the nodes in the graph.
    """
    # Map layout type to corresponding networkx function
    layout_func_map = {
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spring': nx.spring_layout
    }

    # Get the layout function
    layout_func = layout_func_map.get(layout_type, nx.spring_layout)

    # Create graph
    G = nx.Graph()

    # Map cell types to their mean cell 1 number for node size
    node_sizes = state_data.groupby('Cell Type 1')['mean cell 1 number'].first().to_dict()

    # Add nodes from both 'Cell Type 1' and 'Cell Type 2' columns
    for cell_type in pd.unique(state_data[['Cell Type 1', 'Cell Type 2']].values.ravel()):
        size = node_sizes.get(cell_type, 0)  # Use size 0 for cell types that do not appear in the 'Cell Type 1' column
        
        if node_area_scale:
            # This transforms a radius into an area, so that the areas (rather than radius) correlate with pop abundances
            size = np.sqrt(size/np.pi) * 20
        
        G.add_node(cell_type, color=node_color_map[cell_type], size=size)

    # Add edges
    for _, row in state_data.iterrows():
        G.add_edge(row['Cell Type 1'], row['Cell Type 2'], weight=row[edge_weight_column], contacts=row[edge_color_column])

    # If a center cell population was specified, prune graph to only include nodes connected to the center node
    if center_cell_population is not None:
        if center_cell_population not in G.nodes:
            print('Specified population not present, skipping')
            return None, None
        else:
            connected_nodes = list(G.neighbors(center_cell_population))
            connected_nodes.append(center_cell_population)
            G = G.subgraph(connected_nodes)

    # Draw graph
    pos = layout_func(G, scale=layout_scale)
    
    if center_cell_population is not None and force_centre:
        pos[center_cell_population] = np.array([np.mean([x[1][0] for x in pos.items()]),
                                                np.mean([x[1][1] for x in pos.items()])])

    return G, pos
    
    
def _draw_graph(G, fig_size, pos, center_cell_population, draw_labels, node_scale, edge_scale, edge_color_map, node_outline, add_legend, legend_bbox_to_anchor, state, figure_showtitle, figure_padding, figure_box, output_folder, save_extension, node_color_map, edge_color_min, edge_color_max, edge_color_column, edge_weight_column):
    """
    Draw a graph with various visual customizations.

    Args:
        G: A networkx Graph object.
        fig_size: The size of the figure.
        pos: The positions of the nodes in the graph.
        center_cell_population: If specified, only nodes immediately connected to this node are visualised.
        draw_labels: Whether to draw labels on the nodes.
        node_scale: A value to scale the node sizes by.
        edge_scale: A value to scale the edge weights by.
        edge_color_map: The colormap to use for the edge colors.
        node_outline: Whether to draw a black outline around the nodes.
        add_legend: Whether to add a legend to the graph.
        legend_bbox_to_anchor: The location to place the legend in bbox_to_anchor format.
        state: The state for which the graph is generated.
        figure_showtitle: Whether to show a title over each figure.
        figure_padding: The padding around a figure, adjust if nodes are overlapping the edge.
        figure_box: Whether to show a bounding box around the figure.
        output_folder: The folder where the graphs will be saved.
        save_extension: Extension for saving file.
        node_color_map: A dictionary mapping cell types to colors.
        edge_color_min: The minimum value for the colormap used for plotting the edge colors. Defaults to None.
        edge_color_max: The maximum value for the colormap used for plotting the edge colors. Defaults to None.

    Returns:
        None. The graph is drawn and saved in the specified output folder.
    """
    plt.figure(figsize=fig_size)

    edges = G.edges()
    weights = [G[u][v]['weight']*edge_scale for u, v in edges]
    contacts = [G[u][v]['contacts'] for u, v in edges]
    
    if edge_color_min is not None and edge_color_max is not None:
        contacts = [(contact - edge_color_min) / (edge_color_max - edge_color_min) for contact in contacts]

    edge_collection = nx.draw_networkx_edges(G, pos, edge_cmap=plt.get_cmap(edge_color_map), edge_color=contacts, width=weights, alpha=1)

    if node_outline:
        node_collection = nx.draw_networkx_nodes(G, pos, node_color=[node[1]['color'] for node in G.nodes(data=True)], node_size=[node[1]['size']*node_scale for node in G.nodes(data=True)], alpha=1, edgecolors='black')
    else:
        node_collection = nx.draw_networkx_nodes(G, pos, node_color=[node[1]['color'] for node in G.nodes(data=True)], node_size=[node[1]['size']*node_scale for node in G.nodes(data=True)], alpha=1)

    if draw_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)

    # Add legend
    if add_legend:
        plt.colorbar(edge_collection, label=edge_color_column)

        legend_elements = [Line2D([0], [0], color='k', lw=4, label=f'Edge width scaled by {edge_weight_column}')]
        for cell_type, color in node_color_map.items():
            legend_elements.append(Patch(facecolor=color, edgecolor=color, label=cell_type))

        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=legend_bbox_to_anchor)

    if figure_showtitle:
        plt.title(state)
    
    plt.subplots_adjust(left=figure_padding, right=1-figure_padding, bottom=figure_padding, top=1-figure_padding)
    
    if not figure_box:
        plt.box(False)
    
    plt.savefig(os.path.join(output_folder, f'{state.replace(" ", "_")}{save_extension}'), bbox_inches='tight', dpi=400)
    
    plt.show()

    
    
def create_network_graphs(
    dataframe, 
    output_folder='spoox_figures',
    fig_size=(5,5), 
    edge_color_map='Reds',
    edge_color_min=None,
    edge_color_max=None,
    node_color_map=None,
    filters={'gr20_greater': 1, 'gr20 PCF lower_greater': 1, 'MH_FDR_less': 0.05, 'MH_SES': (0, 100)},
    edge_weight_column='gr20',
    edge_color_column='%contacts',
    edge_scale=1, 
    node_scale=1,
    node_area_scale=True,
    layout_scale=1,
    layout_type='circular',
    center_cell_population=None,
    force_centre=True,
    draw_labels=True,
    node_outline=True,
    add_legend=True,
    legend_bbox_to_anchor=(1.35, 0.5),
    figure_box=True,
    figure_padding=0.1,
    figure_showtitle=True,
    save_extension='.png'
):
    """
    Generates a adjaceny network graph that summarises how populations interact using the SpOOx output

    Args:
        dataframe: A pandas DataFrame containing the cell interaction data output from the SpOOx pipeline.
        output_folder: The folder where the graphs will be saved. Defaults to 'spoox_figures'.
        fig_size: The size of the figure. Defaults to (10,10).
        edge_color_map: The colormap to use for the edge colors. Defaults to 'Reds'.
        edge_color_map_min/max: If defined, will set the min and max values on the edge colour.
        node_color_map: A dictionary mapping cell types to colors. If None, a default color map is generated. Defaults to None.
        filters: A dictionary with the filters to apply on the SpOOx output. By default it is the following:
            gr20 > 1
            gr20 lower bound of 95% CI > 1 (ie, statistically significant) 
            Morueta-Holme false discovery rate < 0.05 (ie, statistically significant) 
            Morueta-Holme standard effect size from 0 to 100 (ie, positive associations only)
        edge_weight_column: The column from the dataframe that defines the edge weights. Defaults to 'gr20'.
        edge_color_column: The column from the dataframe that defines the edge colors. Defaults to '%contacts'.
        edge_scale: A value to scale the edge weights by. Defaults to 1.
        node_scale: A value to scale the node sizes by. Defaults to 1.
        node_area_scale: Scale node sizes so that the area correlates with population abundance, rather than node diameter. Default to True.
        layout_scale: A scale factor for the node layout. A larger value spreads the nodes further apart. Defaults to 1.
        layout_type: The type of layout to use for the graph. Options are 'circular', 'kamada_kawai', or 'spring'. Defaults to 'circular'.
        center_cell_population: If specified, only nodes immediately connected to this node are visualised. Defaults to None.
        force_centre: If True, force the center_cell_population node to be at the centre of the graph.
        draw_labels: Whether to draw labels on the nodes. Defaults to True.
        node_outline: Whether to draw a black outline around the nodes. Defaults to True.
        add_legend: Whether to add a legend to the graph. Defaults to True.
        legend_bbox_to_anchor: The location to place the legend in bbox_to_anchor format. Defaults to (1.35, 0.5).
        figure_box: Whether to show a bounding box around the figure. Defaults to True.
        figure_padding: The padding around a figure, adjust if nodes are overlapping the edge. Defaults to 0.1.
        figure_showtitle: Whether to show a title over each figure. Defaults to True.
        save_extension: Extension for saving file. Defaults to .png
        
    Output:
        Saved one graph per state/tissue type detailed in the 'state' colun
    """
    # Check if the output folder exists. If not, create it.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Apply filters to the dataframe
    if filters is not None:
        dataframe = _apply_filters(dataframe, filters)
    
    # Create a color map for cell types
    node_color_map = _create_color_map(dataframe, node_color_map)
    
    # Generate one graph per state
    for state, state_data in dataframe.groupby('state'):
        G, pos = _generate_graph(state_data, node_color_map, layout_scale, layout_type, edge_weight_column, edge_color_column, node_scale, node_area_scale, center_cell_population, force_centre, edge_scale)
        
        if G is not None and pos is not None:
            _draw_graph(G, fig_size, pos, center_cell_population, draw_labels, node_scale, edge_scale, edge_color_map, node_outline, add_legend, legend_bbox_to_anchor, state, figure_showtitle, figure_padding, figure_box, output_folder, save_extension, node_color_map, edge_color_min, edge_color_max, edge_color_column, edge_weight_column)
