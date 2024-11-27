# Standard Library Imports
import itertools
import json
import os
import pathlib
import random
import shutil
import time
from copy import copy
from glob import glob
from multiprocessing import Pool
from os import listdir
from os.path import abspath, exists, isfile, join
from typing import List, Optional, Tuple, Union, Dict
from collections import Counter
from pathlib import Path

# Third-Party Imports
import anndata as ad
from anndata import AnnData
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
import tifffile as tp
from IPython.display import display
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as patches
from matplotlib.pyplot import get_cmap
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew
from scipy import stats
from shapely.geometry import MultiPoint, Point, Polygon
from skimage.draw import rectangle
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread, imsave
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import find_boundaries
from skimage.transform import rescale
from skimage.util import img_as_ubyte
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, notebook

from .plotting import overlayed_heatmaps, obs_to_mask

# Functions Definitions
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


def calculate_boundary_matrix(labeled_image: np.ndarray, 
                              normalize: bool = False, 
                              normalize_to_all_boundaries: bool = False) -> pd.DataFrame:
    """
    Calculate a boundary matrix showing the number of boundary pixels shared between each pair of labels in a labeled image.

    Parameters
    ----------
    labeled_image : np.ndarray
        A 2D labeled image where each unique integer represents a different label.
    normalize : bool, optional
        If True, normalize the boundary counts by the total number of non-zero pixels.
    normalize_to_all_boundaries : bool, optional
        If True, normalize the boundary counts by the total number of boundary pixels.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows and columns represent unique labels, and values represent the number of boundary pixels shared.
    """
    unique_labels = np.unique(labeled_image)
    boundary_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    boundaries = find_boundaries(labeled_image, mode='inner')
    total_non_zero_pixels = np.sum(labeled_image != 0)
    
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i >= j:
                continue
            
            mask = ((labeled_image == label1) | (labeled_image == label2)) & boundaries
            boundary_count = np.sum(mask)
            
            if normalize:
                boundary_count = boundary_count / total_non_zero_pixels
            
            boundary_matrix[i, j] = boundary_count
            boundary_matrix[j, i] = boundary_count
    
    if normalize_to_all_boundaries:
        total_boundary_pixels = np.sum(boundary_matrix)
        if total_boundary_pixels > 0:
            boundary_matrix = boundary_matrix / total_boundary_pixels
    
    boundary_df = pd.DataFrame(boundary_matrix, index=unique_labels, columns=unique_labels)
    return boundary_df


def average_boundary_matrices(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Average a list of boundary matrices.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames to average.

    Returns
    -------
    pd.DataFrame
        The averaged DataFrame.
    """
    concatenated_df = pd.concat(dataframes, axis=0)
    average_df = concatenated_df.groupby(concatenated_df.index).mean().groupby(level=0, axis=1).mean()
    return average_df


def count_label_pixels(labeled_image: np.ndarray) -> pd.DataFrame:
    """
    Count the number of pixels for each unique label in a labeled image.

    Parameters
    ----------
    labeled_image : np.ndarray
        A 2D labeled image where each unique integer represents a different label.

    Returns
    -------
    pd.DataFrame
        A DataFrame with labels as the index and their respective pixel counts.
    """
    unique_labels, counts = np.unique(labeled_image, return_counts=True)
    label_counts_df = pd.DataFrame({'label': unique_labels, 'count': counts})
    label_counts_df.set_index('label', inplace=True)
    return label_counts_df.T


def measure_label_objects(labeled_image: np.ndarray, metric: str = 'count') -> pd.DataFrame:
    """
    Measure specified properties of labeled objects in a labeled image.

    Parameters
    ----------
    labeled_image : np.ndarray
        A 2D array where each unique integer represents a different label.
    metric : str, optional
        The property to measure for each labeled object. If 'count', counts the number of objects for each label.
        Otherwise, any property available in skimage.measure.regionprops can be specified.

    Returns
    -------
    pd.DataFrame
        A DataFrame with labels as the index and the specified metric as the column.
    """
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]
    
    label_metrics = {lbl: [] for lbl in unique_labels}
    
    for lbl in unique_labels:
        binary_image = (labeled_image == lbl)
        labeled_objects = label(binary_image)
        props = regionprops(labeled_objects)
        
        if metric == 'count':
            label_metrics[lbl] = len(props)
        else:
            for prop in props:
                label_metrics[lbl].append(getattr(prop, metric))
    
    if metric != 'count':
        average_metrics = {lbl: np.mean(values) if values else 0 for lbl, values in label_metrics.items()}
    else:
        average_metrics = label_metrics
    
    metrics_df = pd.DataFrame.from_dict(average_metrics, orient='index', columns=[metric]).T
    return metrics_df


def calculate_overlap_matrix(label_image1: np.ndarray, 
                             label_image2: np.ndarray, 
                             ignore_zeros: bool = False, 
                             normalize: bool = False) -> pd.DataFrame:
    """
    Calculate the overlap matrix between two label images.

    This function calculates the overlap between the unique labels in two label images of equal size.
    It can ignore zero values and normalize the overlap counts by the total number of overlapping pixels.

    Parameters
    ----------
    label_image1 : np.ndarray
        The first labeled image.
    label_image2 : np.ndarray
        The second labeled image.
    ignore_zeros : bool, optional
        If True, ignores the zero values in both images during calculation.
    normalize : bool, optional
        If True, normalizes the overlap counts by the total number of overlapping pixels.

    Returns
    -------
    pd.DataFrame
        A DataFrame where the rows represent unique labels from the first image,
        the columns represent unique labels from the second image, and the values
        represent the number of overlapping pixels between each pair of labels.
        If normalization is applied, the values are normalized accordingly.

    Raises
    ------
    ValueError
        If the input images do not have the same dimensions.
    """
    if label_image1.shape != label_image2.shape:
        raise ValueError("The two label images must have the same dimensions.")
    
    unique_labels1 = np.unique(label_image1)
    unique_labels2 = np.unique(label_image2)
    
    overlap_matrix = np.zeros((len(unique_labels1), len(unique_labels2)), dtype=int)
    
    for i, label1 in enumerate(unique_labels1):
        for j, label2 in enumerate(unique_labels2):
            mask1 = (label_image1 == label1)
            mask2 = (label_image2 == label2)
            overlap_count = np.sum(mask1 & mask2)
            overlap_matrix[i, j] = overlap_count
    
    overlap_df = pd.DataFrame(overlap_matrix, index=unique_labels1, columns=unique_labels2)
    
    if ignore_zeros:
        overlap_df = overlap_df.loc[overlap_df.index != 0, overlap_df.columns != 0]
    
    if normalize:
        total_overlap = np.sum(overlap_df.values)
        if total_overlap > 0:
            overlap_df = overlap_df / total_overlap
    
    return overlap_df


def lisa_clustering_image(image1: np.ndarray, image2: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """
    Perform LISA clustering on two images where each pixel represents a different feature,
    and return a binary clustering result in the shape of the original images. Optionally,
    images can be downscaled to reduce computation and memory requirements.

    This updated version applies the spatial weights matrix to both features when computing
    the local indicators, providing a more accurate assessment of local spatial autocorrelation.

    Parameters
    ----------
    image1 : np.ndarray
        2D numpy array representing the first feature image.
    image2 : np.ndarray
        2D numpy array representing the second feature image.
    scale_factor : float, optional
        Scaling factor for downsizing the images. Default is 1.0 (no scaling).
        Values less than 1.0 will downscale the images.

    Returns
    -------
    np.ndarray
        2D numpy array of clustering results where:
        
        - 1 represents 'High' clustering status, indicating all the following conditions are met:
            * The standardized value of the first feature at the location is positive.
            * The weighted sum of the neighboring standardized values for the first feature is positive.
            * The standardized value of the second feature at the location is positive.
            * The weighted sum of the neighboring standardized values for the second feature is positive.
        - 0 represents 'Low' clustering status, indicating that one or more of the conditions are not met.

    Raises
    ------
    ValueError
        If the input images do not have the same dimensions.
        If the standard deviation of a feature is zero, preventing standardization.

    Notes
    -----
    This function performs a LISA (Local Indicators of Spatial Association) analysis by
    considering both the value at each location and its spatial lag (weighted average of neighboring values)
    for both features. The spatial weights are calculated using inverse distance weighting.

    The function identifies areas where both features exhibit high values locally and in their
    neighborhood, which can be interpreted as 'hot spots' of spatial association between the features.
    """
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")

    # Downscale images if scale_factor is different from 1.0
    if scale_factor != 1.0:
        image1 = rescale(image1, scale_factor, anti_aliasing=False, preserve_range=True)
        image2 = rescale(image2, scale_factor, anti_aliasing=False, preserve_range=True)
    
    # Ensure images are 2D arrays
    if image1.ndim != 2 or image2.ndim != 2:
        raise ValueError("Input images must be 2D arrays.")
    
    nrows, ncols = image1.shape

    # Flatten images and prepare coordinate arrays
    feature1 = image1.flatten()
    feature2 = image2.flatten()
    row_indices, col_indices = np.indices((nrows, ncols))
    row_indices = row_indices.flatten()
    col_indices = col_indices.flatten()

    # Set negative values to zero
    feature1[feature1 < 0] = 0
    feature2[feature2 < 0] = 0

    # Construct spatial weights matrix based on inverse distance
    coords = np.column_stack((row_indices, col_indices))
    distances = distance.pdist(coords)
    distances[distances == 0] = np.finfo(float).eps  # Replace zero distances to avoid division by zero
    Wij = distance.squareform(1 / distances)
    np.fill_diagonal(Wij, 0)

    # Normalize weights
    Wij /= Wij.sum(axis=1, keepdims=True)

    # Standardize features
    x_mean = np.mean(feature1)
    y_mean = np.mean(feature2)
    x_std = np.std(feature1)
    y_std = np.std(feature2)
    if x_std == 0 or y_std == 0:
        raise ValueError("Standard deviation of a feature is zero; cannot standardize features.")
    x = (feature1 - x_mean) / x_std
    y = (feature2 - y_mean) / y_std

    # Handle NaN values
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    # Compute spatially weighted features (spatial lags)
    x_weighted = Wij.dot(x)
    y_weighted = Wij.dot(y)

    # Determine positive standardized values and positive spatial lags
    x_positive = x > 0
    x_weighted_positive = x_weighted > 0
    y_positive = y > 0
    y_weighted_positive = y_weighted > 0

    # Clustering logic: 'High' clustering if all conditions are met
    lisa_clust = np.array([
        1 if xp and xwp and yp and ywp else 0
        for xp, xwp, yp, ywp in zip(x_positive, x_weighted_positive, y_positive, y_weighted_positive)
    ])

    return lisa_clust.reshape(nrows, ncols)


def analyse_environments(adata: ad.AnnData,
                         samples_list: List[str],
                         marker_list: List[str],
                         mode: str = 'summary',
                         radius: int = 10,
                         num_cores: int = 4,
                         folder_dir: str = 'images',
                         roi_id: str = 'ROI',
                         x_loc: str = 'X_loc',
                         y_loc: str = 'Y_loc',
                         cell_index_id: str = 'Master_Index',
                         quantile: Optional[float] = 0.999,
                         parameters: Optional[List[str]] = None,
                         return_quant_table: bool = False,
                         invert_value: Optional[int] = None
                         ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Analyzes the environmental texture around cells in given images, extracting specified parameters.

    Parameters
    ----------
    adata : ad.AnnData
        Anndata object where the cell locations are stored.
    samples_list : List[str]
        List of sample identifiers to include in the analysis.
    marker_list : List[str]
        List of markers to analyze.
    mode : str, optional
        Operation mode, 'summary' by default.
    radius : int, optional
        Radius of the square area around the cell for analysis.
    num_cores : int, optional
        Number of cores to use for multiprocessing.
    folder_dir : str, optional
        Directory containing the image files.
    roi_id : str, optional
        Column in `adata` indicating the ROI.
    x_loc : str, optional
        Column in `adata` specifying the x-coordinate of the cell.
    y_loc : str, optional
        Column in `adata` specifying the y-coordinate of the cell.
    cell_index_id : str, optional
        Identifier for cells in the dataframe.
    quantile : Optional[float], optional
        Quantile to determine the maximum intensity for image scaling. If None, no normalization is applied.
        The quantile is calculated over all the images for that marker.
    parameters : Optional[List[str]], optional
        List of strings specifying which features to calculate. Features can include 'Mean', 'Median', 'Std',
        'Kurtosis', 'Skew', and texture features like 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy',
        'Correlation', 'ASM', as well as quantiles (e.g., 'Quantile_0.5'). Default parameter list is:
        ['Mean', 'Std', 'Quantile_0.1', 'Quantile_0.5', 'Quantile_0.9'].
    return_quant_table : bool, optional
        If True, returns a table of quantile values.
    invert_value : Optional[int], optional
        Pixel values for images will be subtracted from this number, if given.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        Depending on `return_quant_table`, returns either a DataFrame of results or a tuple containing
        the DataFrame and a quantile table.
    """
    master_list = []
    quant_list = []
    
    parameters = ['Mean', 'Std', 'Quantile_0.1', 'Quantile_0.5', 'Quantile_0.9'] if parameters is None else parameters

    for marker in marker_list:
        print('Processing Marker:', marker)

        img_data = _load_imgs_from_directory(folder_dir, marker, quiet=True)
        if img_data is None:
            continue
        Img_collect, Img_file_list, img_folders = img_data
        roi_list = [Path(x).stem for x in img_folders]

        # Determine quantile value for image scaling
        quant_value = np.mean([np.quantile(img, quantile) for img in Img_collect]) if quantile is not None else None
        quant_list.append(quant_value)
       
        # Capture ROI-level data for this marker
        roi_datas = []
        
        for image, img_file_name, roi in tqdm(zip(Img_collect, Img_file_list, roi_list), total=len(samples_list)):
            if roi not in samples_list:
                continue

            if invert_value:
                image = invert_value - image
            
            print('Analyzing ROI:', roi)
            adata_roi = adata.obs[adata.obs[roi_id] == roi]
            cell_coords = zip(adata_roi[y_loc], adata_roi[x_loc])

            with Pool(processes=num_cores) as pool:
                results = pool.starmap(_analyse_cell_features, [
                    (image, roi, cell_id, marker, quant_value, radius, coord, parameters, cell_index_id)
                    for coord, cell_id in zip(cell_coords, adata_roi[cell_index_id])
                ])

            roi_df = pd.concat([res for res in results if not res.empty and not res.isna().all().all()])
            roi_datas.append(roi_df.dropna().copy())
        
        master_list.append(pd.concat(roi_datas).dropna())    
        
    # Concatenate data from all markers
    final_data = pd.concat(master_list, axis=1).dropna()

    if return_quant_table:
        quant_table = pd.DataFrame(list(zip(marker_list, quant_list)), columns=['Marker', 'Max Value Images Scaled To'])
        return final_data, quant_table

    return final_data


def _analyse_cell_features(image: np.ndarray,
                          roi: str,
                          cell_id: str,
                          marker: str,
                          quant_value: Optional[float],
                          radius: int,
                          coordinates: Tuple[int, int],
                          parameters: List[str],
                          cell_index_id: str) -> pd.DataFrame:
    """
    Analyzes specified features of a cell within a given image region based on parameters.

    Parameters
    ----------
    image : np.ndarray
        The image data array.
    roi : str
        The region of interest identifier.
    cell_id : str
        Identifier for the cell within the image.
    marker : str
        Marker associated with the cell.
    quant_value : Optional[float]
        Upper limit for clipping and normalizing the image data.
    radius : int
        Radius of the square area around the cell for analysis.
    coordinates : Tuple[int, int]
        Tuple specifying the (y, x) coordinates of the cell.
    parameters : List[str]
        List of strings specifying which features to calculate.
    cell_index_id : str
        Column name to be used as index in the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains measurements for the specified features
        for the given cell, indexed by `cell_index_id`.
    """
    y, x = map(int, coordinates)
    y_min, y_max = max(0, y - radius), min(image.shape[0], y + radius)
    x_min, x_max = max(0, x - radius), min(image.shape[1], x + radius)

    if y_min == 0 and y - radius < 0 or y_max == image.shape[0] and y + radius > image.shape[0] or \
       x_min == 0 and x - radius < 0 or x_max == image.shape[1] and x + radius > image.shape[1]:
        nan_data = {f"{marker}_{param}": [np.nan] for param in parameters}
        nan_data[cell_index_id] = [cell_id]
        return pd.DataFrame(nan_data).set_index(cell_index_id)

    sub_image = image[y_min:y_max, x_min:x_max]

    if quant_value is not None:
        sub_image = np.clip(sub_image, 0, quant_value)
        sub_image /= quant_value

    return _extract_cell_features(sub_image, cell_id, marker, quant_value, cell_index_id, parameters)


def _extract_cell_features(image: np.ndarray,
                          cell_id: str,
                          marker: str,
                          quant_value: Optional[float],
                          cell_index_id: str,
                          parameters: List[str]) -> pd.DataFrame:
    """
    Processes an image to extract cellular features based on a list of specified parameters.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    cell_id : str
        Identifier for the cell within the image.
    marker : str
        Marker associated with the cell.
    quant_value : Optional[float]
        Upper limit for clipping and normalizing the image data.
    cell_index_id : str
        Column name to be used as index in the resulting DataFrame.
    parameters : List[str]
        List of strings specifying which features to calculate.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains measurements for the specified features
        for the given cell, indexed by `cell_index_id`.
    """
    img = image.clip(0, quant_value) if quant_value else image
    if quant_value:
        img /= quant_value

    measurements = {}

    if 'Mean' in parameters:
        measurements['Mean'] = np.mean(img)
    if 'Median' in parameters:
        measurements['Median'] = np.median(img)
    if 'Std' in parameters:
        measurements['Std'] = np.std(img)
    if 'Kurtosis' in parameters:
        measurements['Kurtosis'] = kurtosis(img.flat)
    if 'Skew' in parameters:
        measurements['Skew'] = skew(img.flat)

    for param in parameters:
        if 'Quantile' in param:
            quantile = float(param.split('_')[-1])
            measurements[param] = np.quantile(img, quantile)

    texture_params = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    if any(param in parameters for param in texture_params):
        img = img_as_ubyte(img / img.max() if not quant_value else img)
        glcm = graycomatrix(img, distances=[5], angles=[0], symmetric=True, normed=True)
        for param in texture_params:
            if param in parameters:
                measurements[param] = graycoprops(glcm, param.lower())[0, 0]

    results_list = [[cell_id] + [measurements[param] for param in parameters if param in measurements]]
    column_names = [cell_index_id] + [f"{marker}_{param}" for param in parameters]
    results_df = pd.DataFrame(results_list, columns=column_names).set_index(cell_index_id)

    return results_df


def _load_single_img(filename: str) -> np.ndarray:
    """
    Load a single image from the specified file.

    Parameters
    ----------
    filename : str
        The image file name, must end with .tiff or .tif.

    Returns
    -------
    np.ndarray
        Loaded image data as a float32 array.

    Raises
    ------
    ValueError
        If the file does not end with .tiff or .tif, or if the image is not 2D.
    """
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        Img_in = tp.imread(filename).astype('float32')
    else:
        raise ValueError('Raw file should end with tiff or tif!')
    if Img_in.ndim != 2:
        raise ValueError('Single image should be 2d!')
    return Img_in


def _load_imgs_from_directory(load_directory: str,
                             channel_name: str,
                             quiet: bool = False) -> Optional[Tuple[List[np.ndarray], List[str], List[str]]]:
    """
    Load images from a directory matching the specified channel name.

    Parameters
    ----------
    load_directory : str
        The directory to load images from.
    channel_name : str
        The channel name to match in the image file names.
    quiet : bool, optional
        If True, suppresses print statements.

    Returns
    -------
    Optional[Tuple[List[np.ndarray], List[str], List[str]]]
        A tuple containing a list of loaded images, a list of file names, and a list of subdirectories.

    Raises
    ------
    ValueError
        If no images are found matching the channel name.
    """
    Img_collect = []
    Img_file_list = []
    img_folders = glob(join(load_directory, "*", ""))

    if not quiet:
        print('Image data loaded from ...\n')
    
    if not img_folders:
        img_folders = [load_directory]
        
    for sub_img_folder in img_folders:
        Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) and (f.endswith(".tiff") or f.endswith(".tif"))]
        
        for Img_file in Img_list:
            if channel_name.lower() in Img_file.lower():
                Img_read = _load_single_img(join(sub_img_folder, Img_file))
                
                if not quiet:
                    print(sub_img_folder + Img_file)
                
                Img_file_list.append(Img_file)
                Img_collect.append(Img_read)
                break

    if not quiet:
        print('\nImage data loaded completed!')
    
    if not Img_collect:
        print(f'No such channel as {channel_name}. Please check the channel name again!')
        return None

    return Img_collect, Img_file_list, img_folders


def create_env_anndata(master_list: pd.DataFrame,
                       source_anndata_for_obs: Optional[ad.AnnData] = None,
                       obs_to_transfer: List[str] = [],
                       cell_index: str = 'Master_Index',
                       norm_quantile: float = 0.99,
                       drop_unmeasured_cells: bool = True) -> ad.AnnData:
    """
    Create an AnnData object for environmental analysis.

    Parameters
    ----------
    master_list : pd.DataFrame
        DataFrame containing the measurements.
    source_anndata_for_obs : Optional[ad.AnnData], optional
        Source AnnData object to extract .obs data from.
    obs_to_transfer : List[str], optional
        List of observation column names to transfer from source AnnData.
    cell_index : str, optional
        Column name to be used as index in the resulting AnnData.
    norm_quantile : float, optional
        Quantile for normalization.
    drop_unmeasured_cells : bool, optional
        If True, drop cells that couldn't be measured (sum=0).

    Returns
    -------
    ad.AnnData
        The created AnnData object.
    """
    print('Creating X...')
    X_df = master_list
    
    if X_df.index.name != cell_index:
        X_df = X_df.set_index(cell_index)        

    if drop_unmeasured_cells:
        X_df = X_df.loc[X_df.sum(axis=1) != 0, :]  # Drop rows where sum=0, as those couldn't be measured

    X_df = X_df.dropna()
    X_df = X_df / X_df.quantile(norm_quantile)
    X_df = X_df.clip(upper=1)
    
    if source_anndata_for_obs:
        print('Extracting .obs from source anndata...')
        obs_df = source_anndata_for_obs.obs[obs_to_transfer].set_index(cell_index)
        obs_df.index = obs_df.index.astype(np.int64)
        
        overlap_cells = list(set(X_df.index.tolist()) & set(obs_df.index.tolist()))
        overlap_cells = np.array(overlap_cells, dtype=np.int64)
        
        anndata = ad.AnnData(X=X_df.loc[overlap_cells, :], obs=obs_df.loc[overlap_cells, :])
        
        for c in anndata.obs.columns:
            anndata.obs[c] = anndata.obs[c].cat.remove_unused_categories()
    else:
        anndata = ad.AnnData(X=X_df)        
        
    return anndata

def run_spoox(adata: Union[ad.AnnData, str], 
             population_obs: str, 
             groupby: Optional[str] = None, 
             samples: Optional[List[str]] = None,
             specify_functions: Optional[str] = None,
             spoox_output_dir: str = 'spooxout', 
             spoox_output_summary_dir: str = 'spooxout_summary', 
             output_file: str = 'stats.txt',
             index_obs: str = 'Master_Index',
             roi_obs: str = 'ROI',
             xloc_obs: str = 'X_loc',
             yloc_obs: str = 'Y_loc',
             masks_source_directory: str = 'masks',
             masks_destination_directory: str = 'spoox_mask',
             run_analysis: bool = True,
             analyse_samples_together: bool = True,
             summary: bool = True) -> None:
    
    """
    Run the SpOOx analysis pipeline from an AnnData object.
    More information can be found here: https://github.com/Taylor-CCB-Group/SpOOx/tree/main/src/spatialstats
    
    Any errors in running functions will be saved in errors.csv

    Parameters
    ----------
    adata : Union[ad.AnnData, str]
        AnnData object, or path (string) to a saved h5ad file.
    population_obs : str
        The .obs that defines the population for each cell.
    groupby : Optional[str]
        If specifed, should be a .obs that identifies different groups in the data.
        In the summary step, it will then compare.
    samples : Optional[List[str]]
        Specify a list of samples, if None will process all samples.
    specify_functions : Optional[str]
        By default will run the follow functions from the Spoox pipeline: paircorrelationfunction morueta-holme networkstatistics.
        This is a complete list that will be run if 'all' is used: 
        - paircorrelationfunction
        - localclusteringheatmaps
        - celllocationmap
        - contourplots
        - quadratcounts
        - quadratcelldistributions
        - morueta-holme
        - networkstatistics.
    run_analysis : bool
        Whether or not to run the analyis, or just create the spatialstats file.
    analyse_samples_together : bool
        Whether to analyse all samples together.
    summary : bool
        Whether to run summary script.
        
    Returns
    ----------
    None
        Creates two folders with outputs of SpOOx pipeline.
    """
    
    # Check for 'spatialstats' folder in the working directory
    if not os.path.isdir('spatialstats'):
        print("The 'spatialstats' folder is not found in the current working directory.")
        print("Copying 'spatialstats' from the package directory...")
        package_path = utils.get_module_path('SpatialBiologyToolkit')
        source_spatialstats_path = os.path.join(package_path, 'spatialstats')
        
        if os.path.isdir(source_spatialstats_path):
            shutil.copytree(source_spatialstats_path, 'spatialstats')
            print("'spatialstats' folder has been copied to the current working directory.")
        else:
            raise FileNotFoundError(f"Could not find 'spatialstats' in the package directory: {source_spatialstats_path}")
    
    
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


def _create_heatmap(data_input, states, col, vmin, vmax, norm, cluster_mh, cmap, figsize, save_folder, save_extension, sig_annot=None, specify_row_order=None, specify_col_order=None, cmap_ticks=None):
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
        sig_annot - Column in data that has annotations
        specify_row_order - Specify a row order
        specify_col_order - Specify a column order
        cmap_ticks - Can provide of a list of where ticks should appear on the colourmap

    Returns:
        None. The heatmap is displayed and saved to a file.
    """
    
    data = data_input.copy()
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
        
        # Retrieve annotations from the given column in the raw data
        if sig_annot:
            annotations = state_data.pivot(index='Cell Type 1', columns='Cell Type 2', values=sig_annot)
         
        # Reorder the rows and columns according to the clustermap if the column is not 'MH' or if clustering is enabled.
        if 'Morueta-Holme' not in col or cluster_mh:
            heatmap_data = heatmap_data.iloc[row_order, col_order]
            
            # Reorder is needed
            if sig_annot:
                annotations = annotations.iloc[row_order, col_order]

        # Overwrite column orders if given
        if specify_row_order:
            heatmap_data = heatmap_data.loc[specify_row_order, specify_col_order]
            if sig_annot:
                annotations = annotations.loc[specify_row_order, specify_col_order]           
        
        if type(cmap_ticks) != list and cmap_ticks:
            cmap_ticks = [vmin, 1, vmax]
        
        cbar_kws = {'fraction':0.046, 'pad':0.04, 'ticks':cmap_ticks}
        
        #if sig_annot:
        #    print(col)
        #    display(heatmap_data)
        #   display(annotations)

        # Generate the heatmap.
        if not sig_annot:
            if norm:
                sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, robust=True, square=True, vmin=vmin, vmax=vmax, norm=norm, cbar_kws=cbar_kws)
            else:
                sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, robust=True, square=True, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws)
        else:
            
            annot_kws={'fontsize':'x-large', 'fontweight':'extra bold','va':'center','ha':'center'}

            if norm:
                sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, robust=True, square=True, vmin=vmin, vmax=vmax, norm=norm, cbar_kws=cbar_kws, annot=annotations.to_numpy(), annot_kws=annot_kws, fmt="")
            else:
                sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, robust=True, square=True, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws, annot=annotations.to_numpy(), annot_kws=annot_kws, fmt="")        

        ax.set_title(state)

        
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'heatmap_{col}{save_extension}'), bbox_inches='tight', dpi=400)
    plt.show()


def create_spoox_heatmaps(data_input: pd.DataFrame, 
                          percentile: float = 95, 
                          sig_threshold: float = 0.05, 
                          cluster_mh: bool = True, 
                          save_folder: str = 'spoox_figures', 
                          save_extension: str = '.png', 
                          figsize: int = 10, 
                          cell_type_1_list: Optional[List[str]] = None, 
                          cell_type_2_list: Optional[List[str]] = None, 
                          annotate_signficance: bool = True, 
                          specify_row_order: Optional[List[str]] = None, 
                          specify_col_order: Optional[List[str]] = None, 
                          cmap_ticks: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Creates heatmaps from the SpOOx sumary data
    
    Parameters:
        data_input (pd.DataFrame): Pandas dataframe of loaded SpOOx summary data.
        percentile (float): Percentile to determine vmax for heatmap color scaling. Default is 95.
        sig_threshold (float): Threshold for significance. Default is 0.05.
        cluster_mh (bool): If True, the Morueta-Holme column is clustered. Default is True.
        save_folder (str): Folder to save the generated heatmaps. Default is 'spoox_figures'.
        save_extension (str): File extension for the saved heatmaps. Default is '.png'.
        figsize (int): Size of the figure for the heatmaps. Default is 10.
        cell_type_1_list (Optional[List[str]]): Populations to filter to in cell type 1 (rows).
        cell_type_2_list (Optional[List[str]]): Populations to filter to in cell type 1 (columns).
        annotate_signficance (bool): Whether to annotate significant values or not. Default is True.
        specify_row_order (Optional[List[str]]): Specify a row order. Default is None.
        specify_col_order (Optional[List[str]]): Specify a column order. Default is None.
        cmap_ticks (Optional[List[float]]): List of where ticks should appear on the color map. Default is None.
       
    Returns:
        pd.DataFrame: The function saves heatmap figures in the specified folder.
    """
    # These are the columns from the SpOOx output
    cols_of_interest = ['gr10 PCF lower', 'gr20', 'gr20 PCF lower', 'gr20 PCF upper', 'gr20 PCF combined', 'Morueta-Holme_Significant', 'Morueta-Holme_All', 'contacts', '%contacts', 'Network', 'Network(%)']

    # Create output folder if it doesn't exist.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Copy data
    data = data_input.copy()
    
    # PCF combined
    data['gr20 PCF combined'] = data.apply(_pcf_combined, axis=1)
    
    # Warn if any 'No data' rows are detected, and filter them out.
    no_data_count = sum(data['gr10 PCF lower']=='ND')
    if  no_data_count != 0:
        print(f'WARNING: {str(no_data_count)} instances of no data detected, which is where a cell interaction was never found in that state. These will be excluded.')
        data = data[data['gr10 PCF lower']!='ND']
        
        for c in ['gr20', 'gr10 PCF lower', 'gr20 PCF lower', 'gr20 PCF upper', 'MH_PC', 'MH_SES', 'MH_FDR', 'contacts', '%contacts', 'Network', 'Network(%)']:
            data[c] = data[c].astype('float64')
    
    # Add column names with more meaningful titles
    data['Morueta-Holme_Significant'] = np.where(data['MH_FDR']<sig_threshold, data['MH_SES'], np.nan)
    data['Morueta-Holme_All'] = data['MH_SES']
    
    if annotate_signficance:
        data['Morueta-Holme_Annotation'] = np.where(data['MH_FDR']<sig_threshold, "*", "")
        data['gr20_Annotation'] = copy(np.where(~data['gr20 PCF combined'].isna(), "*", ""))
    
    # Filter to only specific cells on axes
    if cell_type_1_list:
        data = data[data['Cell Type 1'].isin(cell_type_1_list)]
  
    if cell_type_2_list:
        data = data[data['Cell Type 2'].isin(cell_type_2_list)]   
    
    # Validate the input data.
    _validate_inputs(data, cols_of_interest)    
    
    # Get a list of unique states for plotting
    states = data['state'].unique()
    

    for col in cols_of_interest:
        
        sig_annot = None
        
        cmap = get_cmap("Reds")
        cmap.set_under("darkgrey")
        
        vmax = np.percentile(data[col].dropna(), percentile)

        if col in ['gr10 PCF lower', 'gr20 PCF lower']:
            vmin = 1
            norm = None
            
        elif col == 'gr20 PCF upper':
            cmap = get_cmap("Blues_r")
            cmap.set_over("darkgrey")
            vmax=1
            vmin=None
            
        elif col == 'gr20 PCF combined':
            
            #vmax = np.percentile(data.dropna().loc[data[col] != 0, col], percentile)
            vmax = np.percentile(data[col].dropna(), percentile)

            print(vmax)
            #vmin = np.percentile(data[col].dropna(), (100-percentile))
            vmin = np.min(data[col].dropna())
            print(vmin)
            
            cmap = get_cmap("coolwarm")
            
            #cmap.set_over("darkgrey")
            cmap.set_under("darkgrey")
            #cmap.set_bad('darkgrey')
            
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
            
            if annotate_signficance:
                sig_annot = 'gr20_Annotation'
                
         
        elif col == 'gr20':
            
            vmax = np.percentile(data[col].dropna(), percentile)
            vmin = np.percentile(data[col].dropna(), (100-percentile))
            cmap = get_cmap("coolwarm")
            
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
            
            if annotate_signficance:
                sig_annot = 'gr20_Annotation'
                       
        elif 'Morueta-Holme' in col:
            vmax = np.percentile(data[col].dropna(), percentile)
            vmin = np.percentile(data[col].dropna(), (100-percentile))
            cmap = get_cmap("coolwarm")
            
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            
            if annotate_signficance:
                sig_annot = 'Morueta-Holme_Annotation'
                             
        else:
            vmin = 0
            norm = None
        
        # Call the function to create the heatmap for the current column.
        _create_heatmap(data, states, col, vmin, vmax, norm, cluster_mh, cmap, figsize, save_folder, save_extension, sig_annot, specify_row_order, specify_col_order, cmap_ticks)

        print(f"Saved heatmap for column '{col}' in folder '{save_folder}'")
        
        
    return data


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
            print('Specified population not present, skipping. This could be only significant interactions are visualised, and this population has none with the current settings of filters')
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
    data, 
    output_folder='spoox_figures',
    fig_size=(5,5), 
    edge_color_map='Reds',
    edge_color_min=None,
    edge_color_max=None,
    node_color_map=None,
    filters={'gr20_greater': 1, 'gr20 PCF lower_greater': 1, 'MH_FDR_less': 0.05, 'MH_SES': (0, 100)},
    cell_type_1_list=None,
    cell_type_2_list=None,
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
        cell_type_1_list (list, strs): Populations to filter to in cell type 1.
        cell_type_2_list (list, strs): Populations to filter to in cell type 2.
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
    
    dataframe = data.copy()
    
    # Check if the output folder exists. If not, create it.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Apply filters to the dataframe
    if filters is not None:
        dataframe = _apply_filters(dataframe, filters)
        
    # Filter to only specific cells on axes
    if cell_type_1_list:
        dataframe = dataframe[dataframe['Cell Type 1'].isin(cell_type_1_list)]
  
    if cell_type_2_list:
        dataframe = dataframe[dataframe['Cell Type 2'].isin(cell_type_2_list)]   
    
    # Create a color map for cell types
    node_color_map = _create_color_map(dataframe, node_color_map)
    
    # Generate one graph per state
    for state, state_data in dataframe.groupby('state'):
        G, pos = _generate_graph(state_data, node_color_map, layout_scale, layout_type, edge_weight_column, edge_color_column, node_scale, node_area_scale, center_cell_population, force_centre, edge_scale)
        
        if G is not None and pos is not None:
            _draw_graph(G, fig_size, pos, center_cell_population, draw_labels, node_scale, edge_scale, edge_color_map, node_outline, add_legend, legend_bbox_to_anchor, state, figure_showtitle, figure_padding, figure_box, output_folder, save_extension, node_color_map, edge_color_min, edge_color_max, edge_color_column, edge_weight_column)


# Define a function to apply the logic
def _pcf_combined(row):
    if row['gr20 PCF lower'] > 1:
        return row['gr20 PCF lower']
    elif row['gr20 PCF upper'] < 1:
        return row['gr20 PCF upper']
    else:
        return None


def create_subregions(data: Union[pd.DataFrame, AnnData],
                      group_col: str,
                      x_col: str = 'X_loc',
                      y_col: str = 'Y_loc',
                      roi_col: str = 'ROI',
                      max_distance: float = 100,
                      min_cells: int = 10,
                      inplace: bool = True,
                      return_noise_as_nan: bool = True,
                      cluster_col_name: str = 'subregion_cluster',
                      unique_label_col_name: str = 'subregion') -> Optional[pd.DataFrame]:
    """
    Apply DBSCAN clustering to spatial data within specified ROI and group columns,
    adding unique labels for spatially-separated clusters of labels, which we call subregions.

    Parameters
    ----------
    data : Union[pd.DataFrame, AnnData]
        The input DataFrame or an AnnData object (AnnData.obs is used as the DataFrame).
    group_col : str
        Column name that defines cells belonging to different groups (e.g., the manual labels).
    x_col : str, optional
        Column name for the x spatial coordinates. Default is 'X_loc'.
    y_col : str, optional
        Column name for the y spatial coordinates. Default is 'Y_loc'.
    roi_col : str, optional
        Column name for the region of interest (ROI). Default is 'ROI'.
    max_distance : float, optional
        The maximum distance between two cells for them to be considered as in the same subregion. Default is 100.
    min_cells : int, optional
        The number of cells in a neighborhood to be established as a subregion. Default is 10.
    inplace : bool, optional
        If True, modifies the DataFrame in place; otherwise, returns a modified copy. Default is True.
    return_noise_as_nan : bool, optional
        If True, cells identified as noise by DBSCAN are returned as NaN in the new columns. Default is True.
    cluster_col_name : str, optional
        Custom name for the DBSCAN cluster column. Default is 'subregion_cluster'.
    unique_label_col_name : str, optional
        Custom name for the unique label column. Default is 'subregion'.

    Returns
    -------
    Optional[pd.DataFrame]
        If inplace=False, returns a new DataFrame with the clustering results. Otherwise, modifies the input DataFrame in place.
    """
    # Check if the input is an AnnData object
    if isinstance(data, AnnData):
        df = data.obs.copy() if not inplace else data.obs
    else:
        df = data.copy() if not inplace else data

    results = []

    grouped = df.groupby([roi_col, group_col])
    for (roi, group_name), group in grouped:
        coords = group[[x_col, y_col]].values
        db = DBSCAN(eps=max_distance, min_samples=min_cells).fit(coords)
        labels = db.labels_
        
        group[cluster_col_name] = labels
        group[unique_label_col_name] = group.apply(lambda row: f"{roi}_{group_name}_{row[cluster_col_name]}", axis=1)
        
        if return_noise_as_nan:
            group.loc[labels == -1, unique_label_col_name] = np.nan
        
        results.append(group)

    clustered_data = pd.concat(results)

    if inplace:
        data.loc[clustered_data.index, f'{group_col}_{unique_label_col_name}'] = clustered_data[unique_label_col_name].copy()
        data.loc[clustered_data.index, f'{group_col}_{cluster_col_name}'] = clustered_data[cluster_col_name].copy()
        print(f'Columns added to adata.obs: {group_col}_{unique_label_col_name}, {group_col}_{cluster_col_name}')
    else:
        return clustered_data
        
def create_population_anndata(adata: ad.AnnData, 
                              pop_obs: str, 
                              groupby: Optional[str] = None, 
                              roi_obs: str = 'ROI', 
                              crosstab_normalize: Union[bool, str] = False) -> ad.AnnData:
    """
    Create an AnnData object summarizing populations based on specified observations.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object.
    pop_obs : str
        Obs that identifies the population.
    groupby : str, optional
        If supplied, Obs which identifies the level at which to summarize the populations, i.e., the subregion column.
        If None, populations will be summarized at the ROI level.
    roi_obs : str, optional
        Obs that identifies ROIs. Default is 'ROI'.
    crosstab_normalize : Union[bool, str], optional
        Whether to normalize counts. 'index' will normalize within each ROI/subregion, 
        whereas 'columns' will normalize within each population. Default is False.

    Returns
    -------
    ad.AnnData
        An AnnData object summarizing the populations.
    """
    if groupby:
        pop_data = pd.crosstab([adata.obs[roi_obs], adata.obs[groupby]], adata.obs[pop_obs], normalize=crosstab_normalize)
        adata_pops = ad.AnnData(X=pop_data.values, obs=pop_data.reset_index()[[roi_obs, groupby]], var=pop_data.columns.tolist())
    else:
        pop_data = pd.crosstab(adata.obs[roi_obs], adata.obs[pop_obs], normalize=crosstab_normalize)
        adata_pops = ad.AnnData(X=pop_data.values, obs=pop_data.reset_index()[[roi_obs]], var=pop_data.columns.tolist())  
    
    adata_pops.var.set_index(0, inplace=True)
                       
    return adata_pops
    
   
def squidpy_subregion_interactions(
    adata: AnnData,
    population_obs: str,
    subregion: str,
    radius: Tuple[int, int] = (0, 20),
    n_permutations: int = 1000
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate neighborhood enrichment for subregions using Squidpy.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    population_obs : str
        Observation key for population annotation.
    subregion : str
        Key for the subregion annotation.
    radius : tuple, optional
        Radius for spatial neighbors. Default is (0, 20).
    n_permutations : int, optional
        Number of permutations for neighborhood enrichment. Default is 1000.

    Returns
    -------
    dict
        Dictionary containing neighborhood enrichment results for each subregion.
    """
    # If unique subregions haven't already been calculated, then do it now
    if f'{subregion}_subregion' not in adata.obs:
        create_subregions(adata.obs, group_col='histannot_niches')
    else:
        print(f'{subregion}_subregion found in AnnData.obs - assuming subregions already calculated')

    subregion_list = adata.obs[subregion].unique().dropna().tolist()

    # Setup empty dicts to capture results
    results = {'count': {}, 'zscore': {}}

    for s in subregion_list:
        print(f'Calculating for {s}...')

        # Remove cells which were not in a subregion
        adata_sub = adata[~adata.obs[f'{subregion}_subregion'].isna()].copy()

        # Only look at specific subregion
        adata_sub = adata_sub[adata_sub.obs[subregion] == s]

        # Make sure they are categorical
        adata_sub.obs[f'{subregion}_subregion'] = adata_sub.obs[f'{subregion}_subregion'].astype('category')

        sq.gr.spatial_neighbors(adata_sub, library_key=f'{subregion}_subregion', coord_type='generic', radius=radius)
        sq.gr.nhood_enrichment(adata_sub, cluster_key=population_obs, n_perms=n_permutations)

        #results[str(s)] = adata_sub.uns[f'{population_obs}_nhood_enrichment'].copy()

        pops = pd.Categorical(adata_sub.obs[population_obs].cat.categories)

        for x in ['zscore', 'count']:
            array = adata_sub.uns[f'{population_obs}_nhood_enrichment'][x]
            df = pd.DataFrame(data=array, index=pops, columns=pops, dtype=array.dtype)
            results[x][s] = df.copy()

    return results


def plot_subregion_interactions(
    interactions_results: Dict[str, Dict[str, pd.DataFrame]],
    log_counts: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'viridis',
    save_path_counts: Optional[str] = 'subregions_counts.svg',
    save_path_zscore: Optional[str] = 'subregions_zscore.svg',
    **heatmap_kwargs
) -> None:
    """
    Plot neighborhood enrichment results for subregions.

    Parameters
    ----------
    interactions_results : dict
        Dictionary containing neighborhood enrichment results.
    log_counts : bool, optional
        Whether to log-transform the count data. Default is True.
    figsize : tuple, optional
        Size of the figure. Default is (10, 10).
    cmap : str, optional
        Colormap for heatmaps. Default is 'viridis'.
    save_path_counts : str, optional
        Path to save the count heatmap. Default is 'subregions_counts.svg'.
    save_path_zscore : str, optional
        Path to save the z-score heatmap. Default is 'subregions_zscore.svg'.
    **heatmap_kwargs
        Additional keyword arguments for heatmap plotting.

    Returns
    -------
    None
    """
    # Get sublists from results structure
    subregion_list = list(interactions_results['count'].keys())

    # Log counts
    count_dfs = [
        np.log1p(interactions_results['count'][x]) if log_counts else interactions_results['count'][x]
        for x in subregion_list
    ]
    
    zscore_dfs = [interactions_results['zscore'][x] for x in subregion_list]

    overlayed_heatmaps(
        count_dfs,
        cmaps=[cmap] * len(subregion_list),
        figsize=figsize,
        save_path=save_path_counts,
        **heatmap_kwargs
    )

    overlayed_heatmaps(
        zscore_dfs,
        cmaps=[cmap] * len(subregion_list),
        figsize=figsize,
        grid_color='black',
        save_path=save_path_zscore,
        **heatmap_kwargs
    )

def simpleplot_subregion_interactions(interactions_results, zscore_minmax=35, zscore_cmap='coolwarm'):
    '''
    TO DO - Docstring

    This is a simple function to plot and sense-check the results of the interactions analyiss
    '''
    
    envs = list(interactions_results['count'].keys())

    for i in envs:
    
        print(i + "  - " + str(interactions_results['count'][i].shape))
        
        fig, axs = plt.subplots(1, 2, figsize=(16,5))
        sns.heatmap(interactions_results['zscore'][i], vmax=zscore_minmax, vmin=-zscore_minmax, ax=axs[0], cmap=zscore_cmap)
        sns.heatmap(interactions_results['count'][i], ax=axs[1])
        plt.subplots_adjust(wspace=0.6)
        plt.show()


def differing_interactions_in_subregions(interactions_results, metric='zscore', num_lower_extremes=5, num_upper_extremes=5, figsize=(15, 10), ncols=2, xlim=None, bar_width=0.8, wspace=0.5, hspace=0.5, filter_list=None, include_diagonal=False, color_dict=None, line_thickness=1, add_vertical_line=False, add_horizontal_line=True, include_means=False, log_scale=False, remove_y_ticks=False, save=None, title_size='large'):
    """
    TO DO - Docstring

    
    Plots the highest and lowest values from each heatmap in a dictionary, showing unique interactions.

    Parameters:
    heatmap_dict (dict): Dictionary with keys as titles and values as pandas DataFrames representing heatmaps.
    num_lower_extremes (int): Number of lowest values to plot from each heatmap.
    num_upper_extremes (int): Number of highest values to plot from each heatmap.
    figsize (tuple): Figure size for the plot.
    ncols (int): Number of columns for the subplot arrangement.
    xlim (tuple): Tuple specifying the min and max values for the x-axis.
    bar_width (float): Width of the bars in the plot.
    wspace (float): Width space between subplots.
    hspace (float): Height space between subplots.
    filter_list (list): List of items to filter the heatmap dataframes by.
    include_diagonal (bool): Whether to include diagonal elements in the analysis.
    color_dict (dict): Dictionary mapping columns/indices to colors.
    line_thickness (int): Thickness of the black box around each rectangle.
    add_vertical_line (bool): Whether to add a vertical dotted line at x=0.
    add_horizontal_line (bool): Whether to add horizontal dotted lines separating the upper and lower extremes and above the mean bars.
    include_means (bool): Whether to include mean bars for positive and negative values.
    log_scale (bool): Whether to use a logarithmic scale for the x-axis.
    remove_y_ticks (bool): Whether to remove all ticks and labels from the y-axis.
    save (str): Path to save the figure.
    """
    
    heatmap_dict = interactions_results[metric]
    
    heatmap_titles = list(heatmap_dict.keys())
    heatmaps = list(heatmap_dict.values())
    nrows = int(np.ceil(len(heatmaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case we have more than one row/column

    for i, (title, heatmap) in enumerate(heatmap_dict.items()):
        if filter_list is not None:
            heatmap = heatmap.loc[[x for x in filter_list if x in heatmap.index], [x for x in filter_list if x in heatmap.columns]]

        # Mask the upper triangle, include diagonal based on the include_diagonal parameter
        mask = np.triu(np.ones(heatmap.shape, dtype=bool), k=1 if include_diagonal else 0)
        masked_heatmap = heatmap.mask(mask)

        # Convert the DataFrame to a NumPy array and get values - used for mean later
        flat_heatmap = masked_heatmap.values.flatten()
        flat_heatmap = flat_heatmap[~np.isnan(flat_heatmap)]
        
        # Transform data into longform and drop nan
        data = masked_heatmap.reset_index(names='Row').melt(id_vars='Row', var_name='Column', value_name='Value').dropna().sort_values(by='Value')
        data['Label'] = data['Row'].astype(str) + ", " + data['Column'].astype(str)

        # Only use upper/lower extremes
        data = pd.concat([data.head(num_lower_extremes), data.tail(num_upper_extremes)])
        
        if include_means:
            # Calculate means and confidence intervals for positive and negative values
            positive_values = flat_heatmap[flat_heatmap > 0]
            negative_values = flat_heatmap[flat_heatmap < 0]

            if len(positive_values) > 0:
                mean_pos = positive_values.mean()
                ci_pos = stats.t.interval(0.95, len(positive_values)-1, loc=mean_pos, scale=stats.sem(positive_values)) if len(positive_values) > 1 else (mean_pos, mean_pos)
                mean_pos_data = pd.DataFrame({'Value': [mean_pos], 'Row': ['Mean'], 'Column': ['Positive'], 'Label': [r'$\bf{Mean \ Positive \pm 95\% \ CI}$']})
                data = pd.concat([data, mean_pos_data], ignore_index=True)
                data = data.reset_index(drop=True)

            if len(negative_values) > 0:
                mean_neg = negative_values.mean()
                ci_neg = stats.t.interval(0.95, len(negative_values)-1, loc=mean_neg, scale=stats.sem(negative_values)) if len(negative_values) > 1 else (mean_neg, mean_neg)
                mean_neg_data = pd.DataFrame({'Value': [mean_neg], 'Row': ['Mean'], 'Column': ['Negative'], 'Label': [r'$\bf{Mean \ Negative \pm 95\% \ CI}$']})
                data = pd.concat([data, mean_neg_data], ignore_index=True)
                data = data.reset_index(drop=True)

        # Plot the extremes using seaborn
        sns.barplot(x='Value', y='Label', data=data, ax=axes[i], width=bar_width)
        
        axes[i].set_title(title, fontdict={'fontsize': title_size})


        # Set x-axis limits if specified
        if xlim:
            axes[i].set_xlim(xlim)

        # Set x-axis to log scale if specified
        if log_scale:
            axes[i].set_xscale('log')

        # Remove padding by setting y-axis limits
        axes[i].set_ylim(-0.5, len(data) - 0.5)

        # Remove y ticks and labels if specified
        if remove_y_ticks:
            axes[i].set_yticks([])
            axes[i].set_ylabel('')
            axes[i].set_yticklabels([])

        # Add vertical dotted line at x=0 if specified
        if add_vertical_line and not log_scale:
            axes[i].axvline(x=0, color='black', linestyle='dotted')
            
        if add_horizontal_line:
            if num_lower_extremes > 0:
                axes[i].axhline(y=num_lower_extremes - 0.5, color='black', linestyle='dotted')
            if include_means:
                axes[i].axhline(y=len(data) - 2.5, color='black', linestyle='dotted')

        # Modify each bar to be split horizontally into two colors
        for bar, (value, row_label, col_label) in zip(axes[i].patches, zip(data['Value'], data['Row'], data['Column'])):
            # Get the position and dimensions of the bar
            x, y, width, height = bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()

            if row_label == 'Mean' and col_label in ['Positive', 'Negative']:
                # Add the new rectangle for the mean
                edge_color = 'blue' if col_label == 'Positive' else 'red'
                rect = patches.Rectangle((x, y), width, height, color='white', edgecolor=edge_color, hatch='/', linewidth=line_thickness)
                axes[i].add_patch(rect)
                if len(positive_values) > 1 and col_label == 'Positive':
                    axes[i].errorbar(x=value, y=y + height / 2, xerr=[[value-ci_pos[0]], [ci_pos[1]-value]], fmt='none', color='blue', capsize=5)
                if len(negative_values) > 1 and col_label == 'Negative':
                    axes[i].errorbar(x=value, y=y + height / 2, xerr=[[value-ci_neg[0]], [ci_neg[1]-value]], fmt='none', color='red', capsize=5)
                    
                bbox = patches.Rectangle((x, y), width, height, linewidth=line_thickness, edgecolor=edge_color, facecolor='none')
                axes[i].add_patch(bbox)
            else:
                # Remove the original bar
                bar.remove()

                # Create two new rectangles
                color1 = color_dict.get(row_label, 'gray')
                color2 = color_dict.get(col_label, 'gray')

                rect1 = patches.Rectangle((x, y + height / 2), width, height / 2, color=color1, edgecolor='none')
                rect2 = patches.Rectangle((x, y), width, height / 2, color=color2, edgecolor='none')

                # Add the new rectangles to the axes
                axes[i].add_patch(rect1)
                axes[i].add_patch(rect2)

                # Add a black box around the entire rectangle
                bbox = patches.Rectangle((x, y), width, height, linewidth=line_thickness, edgecolor='black', facecolor='none')
                axes[i].add_patch(bbox)

    # Remove empty subplots
    for j in range(len(axes)):
        if j >= len(heatmap_dict):
            fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.tight_layout()
    
    if save:
        fig.savefig(save, bbox_inches='tight', dpi=300)
    
    plt.show()


def _transponse_to_populations(data_dict):
    '''
    This will transform the interactions data to focus it on specific populations/interactions so we can see how they differ between subregions.
    '''
    
    # Find the largest DataFrame based on the number of elements
    largest_df_key = max(data_dict, key=lambda k: data_dict[k].shape[0] * data_dict[k].shape[1])
    largest_df = data_dict[largest_df_key]
    
    # Get the indices and columns from the largest DataFrame
    all_indices = largest_df.index
    all_columns = largest_df.columns
    
    # Create a new dictionary to hold the transformed data
    transformed_dict = {index: pd.DataFrame(index=data_dict.keys(), columns=all_columns) for index in all_indices}
    
    # Populate the new dictionary with the appropriate values
    for title, df in data_dict.items():
        for index in all_indices:
            if index in df.index:
                transformed_dict[index].loc[title] = df.loc[index].reindex(all_columns)
            else:
                # If the index is not in the current DataFrame, fill with NaN
                transformed_dict[index].loc[title] = np.nan
    
    # Create the final dictionary to hold the series
    series_dict = {}
    
    # Extracting series for each index and column
    for index in all_indices:
        for col in all_columns:
            # Creating the series with original DataFrame titles as indices
            series_dict[(index, col)] = pd.Series({title: transformed_dict[index].loc[title, col] for title in data_dict.keys()})
    
    return series_dict


def specific_pops_interactions_in_subregions(interactions_results, metric='zscore', figsize=(15, 10), ncols=2, xlim=None, bar_width=0.7, wspace=0.5, hspace=0.5, color_dict=None, line_thickness=1, add_vertical_line=False, color_by='title', filter_first=None, filter_second=None, sort_values=False, include_average=False, plot_items=None, remove_y_ticks=False, title_size='large'):
    """
    Plots series data from a dictionary, with bars optionally split and colored by title or index.

    Parameters:
    series_dict (dict): Dictionary with keys as tuples representing the titles and values as pandas Series.
    figsize (tuple): Figure size for the plot.
    ncols (int): Number of columns for the subplot arrangement.
    xlim (tuple): Tuple specifying the min and max values for the x-axis.
    wspace (float): Width space between subplots.
    hspace (float): Height space between subplots.
    color_dict (dict): Dictionary mapping titles or indices to colors.
    line_thickness (int): Thickness of the black box around each rectangle.
    add_vertical_line (bool): Whether to add a vertical dotted line at x=0.
    color_by (str): Whether to color bars 'by title' or 'by index'.
    filter_first (list): List of items to filter the first value in the tuple titles.
    filter_second (list): List of items to filter the second value in the tuple titles.
    sort_values (bool): Whether to sort the values before plotting.
    include_average (bool): Whether to include an average bar with a hatched pattern and confidence interval bars.
    plot_items (list): List of items to be plotted from each series.
    """
    
    series_dict = _transponse_to_populations(interactions_results[metric])
    
    
    # Filter the series based on the first and second values in the tuple titles
    filtered_series_dict = {k: v for k, v in series_dict.items() if 
                            (filter_first is None or k[0] in filter_first) and 
                            (filter_second is None or k[1] in filter_second)}

    series_titles = list(filtered_series_dict.keys())
    series_data = list(filtered_series_dict.values())
    nrows = int(np.ceil(len(series_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case we have more than one row/column

    for i, (title, series) in enumerate(filtered_series_dict.items()):
        data = series.reset_index()
        data.columns = ['Index', 'Value']

        if plot_items is not None:
            data = data[data['Index'].isin(plot_items)]

        if sort_values:
            data = data.sort_values(by='Value', ascending=False)

        # Plot the data using seaborn
        sns.barplot(x='Value', y='Index', data=data, ax=axes[i], width=bar_width, palette=[color_dict.get(idx, 'gray') for idx in data['Index']] if color_by == 'index' else ['gray']*len(data))
        axes[i].set_title(f'{title[0]} to {title[1]}', fontdict={'fontsize': title_size})

        # Set x-axis limits if specified
        if xlim:
            axes[i].set_xlim(xlim)

        # Add vertical dotted line at x=0 if specified
        if add_vertical_line:
            axes[i].axvline(x=0, color='black', linestyle='dotted')
            
        if remove_y_ticks:
            axes[i].set_yticks([])
            axes[i].set_ylabel('')
            axes[i].set_yticklabels([])


        for bar in axes[i].patches:
            # Get the position and dimensions of the bar
            x, y, width, height = bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()

            if color_by == 'title':
                # Remove the original bar
                bar.remove()

                # Create two new rectangles
                color1 = color_dict.get(title[0], 'gray')
                color2 = color_dict.get(title[1], 'gray')

                rect1 = patches.Rectangle((x, y + height / 2), width, height / 2, color=color1, edgecolor='none')
                rect2 = patches.Rectangle((x, y), width, height / 2, color=color2, edgecolor='none')

                # Add the new rectangles to the axes
                axes[i].add_patch(rect1)
                axes[i].add_patch(rect2)

            # Add a black box around the entire rectangle
            bbox = patches.Rectangle((x, y), width, height, linewidth=line_thickness, edgecolor='black', facecolor='none')
            axes[i].add_patch(bbox)

        if include_average:
            mean_val = data['Value'].dropna().mean()
            if len(data) > 1:
                ci = stats.t.interval(0.95, len(data['Value'].dropna())-1, loc=mean_val, scale=stats.sem(data['Value'].dropna()))
            else:
                ci = (mean_val, mean_val)  # Default CI if not enough data
            y_position = len(data)  # Position for the mean bar
            axes[i].barh(y=y_position, width=mean_val, height=height, color='none', edgecolor='black', label='Mean ± 95% CI', linewidth=line_thickness)
            if not np.isnan(ci).any():
                axes[i].errorbar(x=mean_val, y=y_position, xerr=[[mean_val-ci[0]], [ci[1]-mean_val]], fmt='none', color='black', capsize=3, linewidth=1)

            # Update y-ticks and labels to include the mean
            current_ticks = axes[i].get_yticks()
            current_labels = [item.get_text() for item in axes[i].get_yticklabels()]
            new_ticks = np.append(current_ticks, y_position)
            new_labels = np.append(current_labels, r'$\bf{Mean \pm 95\% \ CI}$')
            axes[i].set_yticks(new_ticks)
            axes[i].set_yticklabels(new_labels)

    # Remove empty subplots
    for j in range(len(filtered_series_dict), len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.tight_layout()
    plt.show()


















    
    
def simpsons_diversity_index(image):
    '''
    TO DO - Docstring
    '''
    
    # Flatten the image to a 1D array
    pixels = image.flatten()
    
    # Count the frequency of each label
    counts = Counter(pixels)
    
    # Total number of pixels
    total_pixels = len(pixels)
    
    # Calculate the proportion of each label
    proportions = [count / total_pixels for count in counts.values()]
    
    # Calculate the Simpson's Diversity Index
    sdi = 1 - sum(p ** 2 for p in proportions)
    
    return sdi


def lisaclust_catobs_img_overlap(adata, cat_obs, image_folder, roi_obs='ROI', masks_folder='Masks', masks_ext='tif', image_suffix='.tiff', scale_factor=0.05):
    '''
    TO DO - Docstring
    '''   
    
    # Get list of images
    images = os.listdir(image_folder)
    
    roi_list = []
    img1_list = []
    img2_list = []
    prop_sig_list = []

    #for img in tqdm(os.listdir(image_folder)[:8]):
    for img in tqdm(os.listdir(image_folder)):
        
        try:
        
            roi_name = img.replace(image_suffix, '')

            img_array = imread(Path(image_folder, img))

            img_unique = [x for x in np.unique(img_array) if x !=0]

            adata_roi_obs = adata.obs.loc[adata.obs[roi_obs]==roi_name, :]

            cat_unique = adata_roi_obs[cat_obs].unique().tolist()

            for i in img_unique:
                for c in cat_unique:

                    pop_mask, _, _ = obs_to_mask(adata, roi = roi_name, cat_obs = cat_obs, cat_obs_groups=c)

                    clustering_results = lisa_clustering_image(image1 = pop_mask, 
                                                               image2 = np.where(img_array == i, 1, 0), 
                                                               scale_factor=scale_factor)

                    prop_sig = clustering_results.sum() / np.sum(~np.isnan(clustering_results))

                    roi_list.append(str(roi_name))
                    img1_list.append(str(c))
                    img2_list.append(str(i))
                    prop_sig_list.append(np.float32(prop_sig))
                    
        except Exception as e:
            print(e)
            pass
        
    results = pd.DataFrame(zip(roi_list, img1_list, img2_list, prop_sig_list), columns=['ROI','Image1','Image2','PropSig'])
    
    return results
