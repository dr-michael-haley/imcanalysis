import time
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import anndata as ad
import scanpy as sc
from tqdm import tqdm
import networkx as nx

from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt


def get_windows(job,n_neighbors):
    '''
    This is code taken from the Schurch repository!
    
    For each region and each individual cell in dataset, return the indices of the nearest neighbors.
    
    'job:  meta data containing the start time,index of region, region name, indices of region in original dataframe
    n_neighbors:  the number of neighbors to find for each cell
    '''

    tissue_group = Neighborhood_Identification.tissue_group    
    exps = Neighborhood_Identification.exps
    X = Neighborhood_Identification.X
    Y = Neighborhood_Identification.Y
    
    start_time,idx,tissue_name,indices = job
    job_start = time.time()
    
    #print ("Starting:", str(idx+1)+'/'+str(len(exps)),': ' + exps[idx])

    tissue = tissue_group.get_group(tissue_name)
    to_fit = tissue.loc[indices][[X,Y]].values

#     fit = NearestNeighbors(n_neighbors=n_neighbors+1).fit(tissue[[X,Y]].values)
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X,Y]].values)
    m = fit.kneighbors(to_fit)
#     m = m[0][:,1:], m[1][:,1:]
    m = m[0], m[1]
    

    #sort_neighbors
    args = m[0].argsort(axis = 1)
    add = np.arange(m[1].shape[0])*m[1].shape[1]
    sorted_indices = m[1].flatten()[args+add[:,None]]

    neighbors = tissue.index.values[sorted_indices]
   
    end_time = time.time()
   
    #print ("Finishing:", str(idx+1)+"/"+str(len(exps)),": "+ exps[idx],end_time-job_start,end_time-start_time)
    return neighbors.astype(np.int32)



def population_connectivity(nodes, cells, X, Y, radius, cluster_col, population_list):
    '''
    This is a helper function for neighbourhood detection. It creates a network of the window of cells, then returns a connectivity score for each population
    '''    
    
    coords = [(cells.loc[n, X], cells.loc[n, Y]) for n in nodes]
    ndata = pd.DataFrame.from_records(coords, index=nodes)
    
    # Solve problem with remapping?
    ndata.index = ndata.index.astype('str')
    
    adj = radius_neighbors_graph(ndata.to_numpy(), 
                                 radius=radius, 
                                 n_jobs=-1,
                                 include_self=True)
    
    df = pd.DataFrame(adj.A, index=ndata.index, columns=ndata.index)
    
    population_connectivity.adj = adj
    population_connectivity.df = df
    
    graph = nx.from_pandas_adjacency(df)
    
    node_pop_dict = dict(zip(cells.loc[nodes, :].index.astype(str),
                             cells.loc[nodes, cluster_col]))
    
    #nx.set_node_attributes(graph, cells.loc[nodes, cluster_col].T.to_dict(), 'pop')
    nx.set_node_attributes(graph, node_pop_dict, 'pop')

    
   # aver = calculate_average_edges_by_population(graph, 'population', population_list)
    
    population_edges = {population: 0 for population in population_list}
    population_counts = {population: 0 for population in population_list}
    
    for node in graph.nodes():
        if 'pop' in graph.nodes[node]:
            population = graph.nodes[node]['pop']
            if population in population_edges:
                population_edges[population] += graph.degree(node)
                population_counts[population] += 1

    average_edges = {}
    for population in population_edges:
        if population_counts[population] > 0:
            average_edges[population] = population_edges[population] / population_counts[population]
        else:
            average_edges[population] = 0

    # Convert back into a numpy array of length equal to the population list
    average_edges = [np.float16(average_edges[x]) for x in population_list]
    
    return average_edges




def population_connectivity_new(nodes, cells, X, Y, radius, cluster_col, population_list):
    '''
    This is an ALTERNATIVE helper function for neighbourhood detection that doesnt used Networkx. It creates a network of the window of cells, then returns a connectivity score for each population
    '''    
    
    
    coords = [(cells.loc[n, X], cells.loc[n, Y]) for n in nodes]
    ndata = pd.DataFrame.from_records(coords, index=nodes)
    
    # List of population identities for columns
    df_pops = cells.loc[ndata.index, cluster_col]
           
    # Get adjacency matrix using Scipy function
    adj = radius_neighbors_graph(ndata.to_numpy(), 
                                 radius=radius, 
                                 n_jobs=-1,
                                 include_self=False)
    
    # Make into a dataframe
    df = pd.DataFrame(adj.A, index=df_pops, columns=df_pops)
    
    # Get total edges by population by summing over both axis
    total_edges_by_pop = df.reset_index(names='population').groupby('population').sum().T.reset_index(names='population').groupby('population').sum().sum(axis=1)
    
    # Number of cells per population
    total_cells_per_pop = df_pops.value_counts()
    
    # Convert back into a numpy array of length equal to the population list
    average_edges = np.float16([0 if x not in total_edges_by_pop.index else total_edges_by_pop[x]/total_cells_per_pop[x] for x in population_list])
    
    return average_edges    
    

    
def Neighborhood_Identification(data,
                                cluster_col,
                                ks = [20],
                                keep_cols = 'all',
                                radius=20,
                                X = 'X_loc',
                                Y = 'Y_loc',
                                reg = 'ROI',
                                connect_suffix='_CON',
                                return_raw=False):
    '''
    This has been developed from the Schurch data.
    
    data = Dataframe, AnnData (in which case .obs will be used), or path to a .csv file.
    cluster_col = Column which defines the populations
    ks = A list of the different window sizes to try - default of 20 seems to work well
    keep_cols = Columns from the cell tabe that will be returned in metadata
    radius = Radius at which cells are considered connected/interacting.
    X = Column defining X location
    Y = Column defining Y location
    reg = Column defining each separate ROI
    
    '''    


    # Make accessible
    Neighborhood_Identification.cluster_col = cluster_col
    Neighborhood_Identification.X = X
    Neighborhood_Identification.Y = Y
    Neighborhood_Identification.reg = reg
    
    
    #read in data and do some quick data rearrangement
    n_neighbors = max(ks)

    
    if type(data) == pd.core.frame.DataFrame:
        cells = data
    elif type(data) == ad._core.anndata.AnnData:
        cells = data.obs.copy()
        cells.index = cells.index.astype(int)
    elif type(data) == str: 
        cells = pd.read_csv(data)
    else:
        print(f'Input data of type {str(type(data))} not recognised as input')
        return None
    
    # Remove categorical columns    
    for col in [x for x in cells.columns.tolist() if x not in [reg]]:
        if cells[col].dtype == 'category':
            cells[col] = cells[col].astype(cells[col].cat.categories.to_numpy().dtype)
    
    if keep_cols=='all':
        keep_cols = cells.columns.tolist()
    else:
        keep_cols = [reg,cluster_col,X,Y] + keep_cols    
    
    cells = pd.concat([cells,pd.get_dummies(cells[cluster_col])],1)

    sum_cols = cells[cluster_col].unique()
    values = cells[sum_cols].values

    #find windows for each cell in each tissue region
    tissue_group = cells[[X,Y,reg]].groupby(reg)
    exps = list(cells[reg].unique())
    
    #Save into variables accessible for get_window function
    Neighborhood_Identification.tissue_group = tissue_group
    Neighborhood_Identification.exps = exps
    
    #print(exps)
    tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 
    tissues = [get_windows(job,n_neighbors) for job in tissue_chunks]   
    
    #for each cell and its nearest neighbors, reshape and count the number of each cell type in those neighbors.
    out_dict = {}
    out_dict_connectivity = {}
    
    counter=0
    for k in ks:
        for neighbors,job in zip(tissues,tissue_chunks):
           
            chunk = np.arange(len(neighbors))#indices
            tissue_name = job[2]
            indices = job[3]
            
            counter +=1
            print(f'{str(counter)} of {str(len(tissues))} - Calculating for region {str(tissue_name)}')
          
            ############## ABUNDANCY  
            
            window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(sum_cols)).sum(axis = 1)
            out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)
            #window_itter = window.copy()
            
            ########### CONNECTIVITY
            
            window_connectivity = [population_connectivity(nodes = n.tolist()[:k],
                                    cells = cells,
                                    X = X,
                                    Y = Y,
                                    radius=radius,
                                    cluster_col=cluster_col,
                                    population_list =  sum_cols) for n in tqdm(neighbors, position=0, leave=True)]
        
            window_connectivity = np.array(window_connectivity, dtype=np.float16)
            out_dict_connectivity[(tissue_name,k)] = (window_connectivity.astype(np.float16),indices)

    #concatenate the summed windows and combine into one dataframe for each window size tested.
    windows = {}
    windows_connect = {}

    for k in ks:

        window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = sum_cols) for exp in exps],0)
        window = window.loc[cells.index.values]
        #window = pd.concat([cells[keep_cols],window],1)
        windows[k] = window
        
        window_connect = pd.concat([pd.DataFrame(out_dict_connectivity[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = sum_cols) for exp in exps],0)
        window_connect = window_connect.loc[cells.index.values]
        #window_connect = pd.concat([cells[keep_cols],window_connect],1)
        
        # Add suffix onto connectivity column names
        if connect_suffix:
            window_connect.columns = window_connect.columns + connect_suffix
        
        windows_connect[k] = window_connect
    
    metadata = cells[keep_cols]
    
    # These are simplty here for diagnosing errors
    Neighborhood_Identification.tissues = tissues
    Neighborhood_Identification.tissue_chunks = tissue_chunks
    Neighborhood_Identification.neighbors = neighbors
    Neighborhood_Identification.job = job
    Neighborhood_Identification.cells = cells
    Neighborhood_Identification.sum_cols = sum_cols
    Neighborhood_Identification.values = values
    Neighborhood_Identification.window = window
    Neighborhood_Identification.chunk = chunk
    Neighborhood_Identification.out_dict = out_dict
    
    
    if return_raw:
        return windows, windows_connect, metadata
    else:
        
        adatas = {}

        for k in ks:

            combined_data = pd.merge(left = windows[k],
                     right= windows_connect[k],
                     left_index=True,
                     right_index=True)

            scaler = StandardScaler()

            scaled_data = pd.DataFrame(scaler.fit_transform(combined_data), columns = combined_data.columns)

            # Create AnnData with normalised data
            adata = ad.AnnData(scaled_data)

            # Add in obs
            adata.obs = metadata

            adatas.update({k:adata.copy()})
        
        if len(ks)==1:
            return adatas[ks[0]]
        else:
            return adatas
            

            
def prune_leiden_using_dendrogram(adata,
                                  leiden_obs,
                                  new_obs='leiden_merged',
                                  mode='max',
                                  max_leiden=None,
                                  minimum_pop_size=None):
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

    return remap_dict

                                  
                                  
                           
#########################################################
''' The following functions and code for voronoi plots have been taken entirely from the Nolan lab (https://github.com/nolanlab) '''
#########################################################
    
from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    adapted from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647 3.18.2019
    
    
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuplesy
    
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_voronoi(points,colors,invert_y = True,edge_color = 'facecolor',line_width = .1,alpha = 1,size_max=np.inf):
    
# spot_samp = spot#.sample#(n=100,random_state = 0)
# points = spot_samp[['X:X','Y:Y']].values
# colors = [sns.color_palette('bright')[i] for i in spot_samp['neighborhood10']]

    if invert_y:
        points[:,1] = max(points[:,1])-points[:,1]
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    pts = MultiPoint([Point(i) for i in points])
    mask = pts.convex_hull
    new_vertices = []
    if type(alpha)!=list:
        alpha = [alpha]*len(points)
    areas = []
    for i,(region,alph) in enumerate(zip(regions,alpha)):
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        areas+=[p.area]
        if p.area <size_max:
            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            new_vertices.append(poly)
            if edge_color == 'facecolor':
                plt.fill(*zip(*poly), alpha=alph,edgecolor=  colors[i],linewidth = line_width , facecolor = colors[i])
            else:
                plt.fill(*zip(*poly), alpha=alph,edgecolor=  edge_color,linewidth = line_width, facecolor = colors[i])
        # else:

        #     plt.scatter(np.mean(p.boundary.xy[0]),np.mean(p.boundary.xy[1]),c = colors[i])
    return areas


def draw_voronoi_scatter(spot,c,voronoi_palette = sns.color_palette('bright'),scatter_palette = 'voronoi',X = 'X:X', Y = 'Y:Y',voronoi_hue = 'neighborhood10',scatter_hue = 'ClusterName',
        figsize = (5,5),
         voronoi_kwargs = {},
         scatter_kwargs = {}):
    
    if scatter_palette=='voronoi':
        scatter_palette = voronoi_palette
        scatter_hue = voronoi_hue
    '''
    plot voronoi of a region and overlay the location of specific cell types onto this
    
    spot:  cells that are used for voronoi diagram
    c:  cells that are plotted over voronoi
    palette:  color palette used for coloring neighborhoods
    X/Y:  column name used for X/Y locations
    hue:  column name used for neighborhood allocation
    figsize:  size of figure
    voronoi_kwargs:  arguments passed to plot_vornoi function
    scatter_kwargs:  arguments passed to plt.scatter()

    returns sizes of each voronoi to make it easier to pick a size_max argument if necessary
    '''
    if len(c)>0:
        neigh_alpha = .3
    else:
        neigh_alpha = 1
        
    voronoi_kwargs = {**{'alpha':neigh_alpha},**voronoi_kwargs}
    scatter_kwargs = {**{'s':50,'alpha':1,'marker':'.'},**scatter_kwargs}
    
    plt.figure(figsize = figsize)
    colors  = [voronoi_palette[i] for i in spot[voronoi_hue]]
    a = plot_voronoi(spot[[X,Y]].values,
                 colors,#[{0:'white',1:'red',2:'purple'}[i] for i in spot['color']],
                 **voronoi_kwargs)
    
    if len(c)>0:
        if 'c' not in scatter_kwargs:
            colors  = [scatter_palette[i] for i in c[scatter_hue]]
            scatter_kwargs['c'] = colors
            
        plt.scatter(x = c[X],y = (max(spot[Y])-c[Y].values),
                  **scatter_kwargs
                   )
    plt.axis('off');
    return a