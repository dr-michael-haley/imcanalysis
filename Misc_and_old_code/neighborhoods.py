# Standard Library Imports
import sys
import time

# Third-Party Imports
import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi

def _get_windows(job, n_neighbors: int) -> np.ndarray:
    """
    For each region and each individual cell in dataset, return the indices of the nearest neighbors.

    Parameters
    ----------
    job : tuple
        Meta data containing the start time, index of region, region name, indices of region in original dataframe.
    n_neighbors : int
        The number of neighbors to find for each cell.

    Returns
    -------
    np.ndarray
        Indices of the nearest neighbors for each cell.
    """
    tissue_group = Neighborhood_Identification.tissue_group
    exps = Neighborhood_Identification.exps
    X = Neighborhood_Identification.X
    Y = Neighborhood_Identification.Y

    start_time, idx, tissue_name, indices = job
    job_start = time.time()

    tissue = tissue_group.get_group(tissue_name)
    to_fit = tissue.loc[indices][[X, Y]].values

    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X, Y]].values)
    m = fit.kneighbors(to_fit)
    m = m[0], m[1]

    args = m[0].argsort(axis=1)
    add = np.arange(m[1].shape[0]) * m[1].shape[1]
    sorted_indices = m[1].flatten()[args + add[:, None]]

    neighbors = tissue.index.values[sorted_indices]

    end_time = time.time()
    return neighbors.astype(np.int32)


def _population_connectivity(
    nodes: np.ndarray,
    cells: pd.DataFrame,
    X: str,
    Y: str,
    radius: float,
    cluster_col: str,
    population_list: list,
    bootstrap: int = None,
    connectivity_modes: list = ['conn_all', 'conn_self']
) -> dict:
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
    bootstrap : int, optional
        Number of bootstrap iterations. Default is None.
    connectivity_modes : list, optional
        List of connectivity modes to calculate. Default is ['conn_all', 'conn_self'].

    Returns
    -------
    dict
        Dictionary with connectivity scores for each population.
    """
    coords = [(cells.loc[n, X], cells.loc[n, Y]) for n in nodes]
    ndata = pd.DataFrame.from_records(coords, index=nodes)
    ndata.index = ndata.index.astype(str)

    adj = radius_neighbors_graph(
        ndata.to_numpy(), radius=radius, n_jobs=-1, include_self=True
    )

    df = pd.DataFrame(adj.A, index=ndata.index, columns=ndata.index)

    graph = nx.from_pandas_adjacency(df)
    node_pop_dict = dict(
        zip(cells.loc[nodes, :].index.astype(str), cells.loc[nodes, cluster_col])
    )
    nx.set_node_attributes(graph, node_pop_dict, "pop")

    observed = {}
    predicted = {}
    output = {}

    for m in connectivity_modes:
        if m == "conn_all":
            observed[m] = _average_connections_per_pop(graph, population_list=population_list, attr="pop")
        if m == "conn_self":
            observed[m] = _proportion_samepop_interactions_per_pop(graph, population_list=population_list, attr="pop")

    if bootstrap:
        for m in connectivity_modes:
            predicted[m] = []

        for n in range(bootstrap):
            graph = _randomise_graph(graph, attr="pop")

            for m in connectivity_modes:
                if m == "conn_all":
                    predicted[m].append(_average_connections_per_pop(graph, population_list=population_list, attr="pop"))
                if m == "conn_self":
                    predicted[m].append(_proportion_samepop_interactions_per_pop(graph, population_list=population_list, attr="pop"))

        for m in connectivity_modes:
            predicted[m] = np.mean(np.array(predicted[m]), axis=0)
            output[m] = observed[m] - predicted[m]
    else:
        for m in connectivity_modes:
            output[m] = observed[m]

    return output


def _average_connections_per_pop(graph, population_list, attr="pop"):
    """
    Calculate the average connections per population.

    Parameters
    ----------
    graph : networkx.Graph
        The graph object.
    population_list : list
        List of populations.
    attr : str, optional
        Attribute name for population. Default is 'pop'.

    Returns
    -------
    list
        List of average connections per population.
    """
    population_edges = {population: 0 for population in population_list}
    population_counts = {population: 0 for population in population_list}

    for node in graph.nodes():
        if attr in graph.nodes[node]:
            population = graph.nodes[node][attr]
            if population in population_edges:
                population_edges[population] += graph.degree(node)
                population_counts[population] += 1

    average_edges = {}
    for population in population_edges:
        if population_counts[population] > 0:
            average_edges[population] = population_edges[population] / population_counts[population]
        else:
            average_edges[population] = 0

    average_edges = [np.float16(average_edges[x]) for x in population_list]

    return average_edges


def _proportion_samepop_interactions_per_pop(graph, population_list, attr="pop"):
    """
    Calculate the proportion of same population interactions per population.

    Parameters
    ----------
    graph : networkx.Graph
        The graph object.
    population_list : list
        List of populations.
    attr : str, optional
        Attribute name for population. Default is 'pop'.

    Returns
    -------
    list
        List of proportions of same population interactions per population.
    """
    population_conn_prop = {population: 0 for population in population_list}
    population_counts = {population: 0 for population in population_list}

    for node in graph.nodes():
        population = graph.nodes[node][attr]
        total_connections = graph.degree[node]

        if total_connections != 0:
            connections_same_pop = sum(1 for neighbor in graph.neighbors(node) if graph.nodes[neighbor][attr] == population)
            proportion_same_pop = connections_same_pop / total_connections
        else:
            proportion_same_pop = 0

        population_conn_prop[population] += proportion_same_pop
        population_counts[population] += 1

    average_proportion = {population: 0 for population in population_list}

    for population in population_list:
        if population_counts[population] > 0:
            average_proportion[population] = population_conn_prop[population] / population_counts[population]
        else:
            average_proportion[population] = 0

    average_proportion = [np.float16(average_proportion[x]) for x in population_list]

    return average_proportion


def _randomise_graph(graph, attr):
    """
    Randomize the node attributes in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph object.
    attr : str
        Attribute name to randomize.

    Returns
    -------
    networkx.Graph
        The randomized graph.
    """
    import random

    graph_perm = graph.copy()
    attr_list = [graph_perm.nodes[x][attr] for x in graph_perm.nodes()]
    random.shuffle(attr_list)

    for a, n in zip(attr_list, graph_perm.nodes()):
        graph_perm.nodes[n].update({attr: a})

    return graph_perm




def Neighborhood_Identification(
    data,
    cluster_col: str,
    ks: list = [20],
    keep_cols: str = 'all',
    radius: float = 20,
    X: str = 'X_loc',
    Y: str = 'Y_loc',
    reg: str = 'ROI',
    modes: list = ['abundancy', 'connectivity'],
    connect_suffix: bool = True,
    return_raw: bool = False,
    bootstrap: int = 75,
    connectivity_modes: list = ['conn_all', 'conn_self'],
    reset_index: bool = True
):
    """
    Identify neighborhoods in spatial data.

    Parameters
    ----------
    data : DataFrame, AnnData, or str
        Input data. Can be a DataFrame, AnnData object, or path to a CSV file.
    cluster_col : str
        Column defining the populations.
    ks : list, optional
        List of window sizes to try. Default is [20].
    keep_cols : str, optional
        Columns to keep in the output metadata. Default is 'all'.
    radius : float, optional
        Radius at which cells are considered connected/interacting. Default is 20.
    X : str, optional
        Column defining X location. Default is 'X_loc'.
    Y : str, optional
        Column defining Y location. Default is 'Y_loc'.
    reg : str, optional
        Column defining each separate ROI. Default is 'ROI'.
    modes : list, optional
        List of modes for neighborhood identification. Default is ['abundancy', 'connectivity'].
    connect_suffix : bool, optional
        If True, add suffix to connectivity column names. Default is True.
    return_raw : bool, optional
        If True, return raw data. Default is False.
    bootstrap : int, optional
        Number of bootstrap iterations. Default is 75.
    connectivity_modes : list, optional
        List of connectivity modes to calculate. Default is ['conn_all', 'conn_self'].
    reset_index : bool, optional
        If True, reset the index of the input data. Default is True.

    Returns
    -------
    dict or AnnData
        Dictionary of modalities if return_raw is True, else AnnData object.
    """
    from copy import copy

    # Make accessible
    Neighborhood_Identification.cluster_col = cluster_col
    Neighborhood_Identification.X = X
    Neighborhood_Identification.Y = Y
    Neighborhood_Identification.reg = reg

    n_neighbors = max(ks)

    if isinstance(data, pd.core.frame.DataFrame):
        cells = data.copy()
    elif isinstance(data, ad._core.anndata.AnnData):
        cells = data.obs.copy()
    elif isinstance(data, str):
        cells = pd.read_csv(data)
    else:
        print(f'Input data of type {str(type(data))} not recognized as input')
        return None

    if keep_cols == 'all':
        keep_cols = cells.columns.tolist()
    else:
        keep_cols = [reg, cluster_col, X, Y] + keep_cols

    if reset_index:
        cells.reset_index(drop=True, inplace=True)

    cells = pd.concat([cells, pd.get_dummies(cells[cluster_col])], axis=1)
    sum_cols = cells[cluster_col].unique()
    values = cells[sum_cols].values

    tissue_group = cells[[X, Y, reg]].groupby(reg)
    exps = list(cells[reg].unique())

    Neighborhood_Identification.tissue_group = tissue_group
    Neighborhood_Identification.exps = exps

    tissue_chunks = [
        (time.time(), exps.index(t), t, a)
        for t, indices in tissue_group.groups.items()
        for a in np.array_split(indices, 1)
    ]
    tissues = [_get_windows(job, n_neighbors) for job in tissue_chunks]

    modalities = copy(modes)
    if 'connectivity' in modalities:
        modalities.remove('connectivity')
        modalities += connectivity_modes

    out_dict_all = {m: {} for m in modalities}
    counter = 0

    for k in ks:
        for neighbors, job in zip(tissues, tissue_chunks):
            chunk = np.arange(len(neighbors))
            tissue_name = job[2]
            indices = job[3]

            counter += 1
            print(f'{counter} of {len(tissues)} - Calculating for region {tissue_name}')

            if 'abundancy' in modes:
                m = 'abundancy'
                window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
                out_dict_all[m][(tissue_name, k)] = (window.astype(np.float16), indices)

            if 'connectivity' in modes:
                window = [
                    _population_connectivity(
                        nodes=n.tolist()[:k],
                        cells=cells,
                        X=X,
                        Y=Y,
                        radius=radius,
                        cluster_col=cluster_col,
                        population_list=sum_cols,
                        bootstrap=bootstrap
                    )
                    for n in tqdm(neighbors, position=0, leave=True)
                ]

                window_connectivity = {}

                for m in connectivity_modes:
                    window_connectivity[m] = [window[x][m] for x in range(len(window))]
                    window_connectivity[m] = np.array(window_connectivity[m], dtype=np.float16)
                    out_dict_all[m][(tissue_name, k)] = (window_connectivity[m].astype(np.float16), indices)

    modalities_output = {}
    modalities_output.update(dict([(k, {}) for k in ks]))

    for k in ks:
        for m in modalities:
            out_dict = out_dict_all[m]
            window = pd.concat(
                [
                    pd.DataFrame(out_dict[(exp, k)][0], index=out_dict[(exp, k)][1].astype(int), columns=sum_cols)
                    for exp in exps
                ],
                axis=0
            )
            window = window.loc[cells.index.values]
            if connect_suffix:
                window.columns = window.columns.astype(str) + '_' + str(m)
            modalities_output[k].update({str(m): copy(window)})

    metadata = cells[keep_cols]

    if return_raw:
        return modalities_output, metadata
    else:
        adatas = {}

        for k in ks:
            combined_data = pd.concat([modalities_output[k][x] for x in modalities], axis=1)
            scaler = StandardScaler()
            scaled_data = pd.DataFrame(scaler.fit_transform(combined_data), columns=combined_data.columns)
            adata = ad.AnnData(scaled_data)
            adata.obs = metadata
            adatas.update({k: adata.copy()})

        if len(ks) == 1:
            return adatas[ks[0]]
        else:
            return adatas


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
    voronoi_palette=sns.color_palette('bright'),
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
        Color palette used for coloring neighborhoods. Default is sns.color_palette('bright').
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
