import os
import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Sequence
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from joblib import Parallel, delayed


def get_annulus_proportions_func(
    adata,
    roi_id,
    master_index,
    roi_col='ROI',
    master_index_col='Master_Index',
    inner_radius=0.0,
    outer_radius=50.0,
    x_col='X_loc',
    y_col='Y_loc',
    cell_type_col='cell_type'
):
    """
    Compute the proportions of cell types within a specified annulus
    around a reference cell (by master index) in a particular ROI.

    Parameters
    ----------
    adata : AnnData
        An AnnData object whose .obs contains columns for:
        - roi_col: ROI/region of interest
        - master_index_col: a unique ID for each cell across all ROIs
        - cell_type_col: the cell-type annotation
        - x_col, y_col: the 2D coordinates
    roi_id : str or int
        The identifier of the ROI you want to analyze.
    roi_col : str
        The column name in adata.obs that indicates the ROI identity.
    master_index : int
        The master index for the reference cell (from master_index_col).
    master_index_col : str
        The column name in adata.obs that holds the master index.
    inner_radius : float
        Inner radius of the annulus (default = 0).
    outer_radius : float
        Outer radius of the annulus (default = 50).
    x_col : str
        Column name in adata.obs for the x-coordinate.
    y_col : str
        Column name in adata.obs for the y-coordinate.
    cell_type_col : str
        Column name in adata.obs for the cell-type annotation.

    Returns
    -------
    pd.Series
        A pandas Series where each index is a cell type, and each value
        is the proportion of cells of that type in the annulus.
    """
    # 1. Filter cells to the specified ROI
    roi_mask = adata.obs[roi_col] == roi_id
    roi_df = adata.obs.loc[roi_mask]
    
    # 2. Identify the reference cell based on master_index
    if master_index not in roi_df[master_index_col].values:
        raise ValueError(f"Master index {master_index} not found in ROI {roi_id}.")
    
    ref_row = roi_df.loc[roi_df[master_index_col] == master_index].iloc[0]

    # 3. Compute distances from the reference cell to each cell in this ROI
    dx = roi_df[x_col] - ref_row[x_col]
    dy = roi_df[y_col] - ref_row[y_col]
    dist = np.sqrt(dx**2 + dy**2)

    # 4. Identify cells within [inner_radius, outer_radius]
    annulus_mask = (dist >= inner_radius) & (dist <= outer_radius)
    annulus_cells = roi_df.loc[annulus_mask]

    # 5. Compute proportions of each cell type in the annulus
    proportions = annulus_cells[cell_type_col].value_counts(normalize=True)

    return proportions


def get_nearest_population_distances_func(
    adata,
    roi_id,
    master_index,
    roi_col='ROI',
    master_index_col='Master_Index',
    x_col='X_loc',
    y_col='Y_loc',
    cell_type_col='cell_type',
    populations: Optional[Sequence[str]] = None
):
    """
    Compute the distance from a reference cell to the nearest member of each population
    within the specified ROI.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose .obs contains ROI, master index, cell type, and x/y coordinates.
    roi_id : str or int
        ROI identifier to analyze.
    master_index : int
        Master index for the reference cell (from master_index_col).
    roi_col : str
        Column in adata.obs indicating ROI identity.
    master_index_col : str
        Column in adata.obs holding the master index.
    x_col : str
        Column name for x-coordinate.
    y_col : str
        Column name for y-coordinate.
    cell_type_col : str
        Column name for population/cell-type annotation.

    Returns
    -------
    pd.Series
        Series where each index is a population and each value is the nearest
        distance to that population (NaN if none found).
    """
    # Filter cells to the specified ROI
    roi_mask = adata.obs[roi_col] == roi_id
    roi_df = adata.obs.loc[roi_mask]

    # Identify the reference cell
    if master_index not in roi_df[master_index_col].values:
        raise ValueError(f"Master index {master_index} not found in ROI {roi_id}.")

    distances_df = _compute_nearest_population_distances_matrix(
        roi_df,
        roi_id=roi_id,
        master_index_col=master_index_col,
        cell_type_col=cell_type_col,
        x_col=x_col,
        y_col=y_col,
        populations=populations
    )

    row = distances_df.loc[
        distances_df["master_index"] == master_index
    ].iloc[0]
    return row.drop(labels=["master_index", "roi"])


def compute_nearest_population_distances_for_all_cells(
    adata,
    roi_id,
    roi_col,
    master_index_col,
    cell_type_col,
    x_col,
    y_col,
    populations: Optional[Sequence[str]] = None
):
    """
    For each cell in the specified ROI, compute the nearest distance
    to each population in that ROI.

    Returns
    -------
    pd.DataFrame
        Columns: [master_index, roi, ...population distances...]
    """
    adata_filt = adata[adata.obs[roi_col] == roi_id]
    roi_df = adata_filt.obs.copy()

    return _compute_nearest_population_distances_matrix(
        roi_df,
        roi_id=roi_id,
        master_index_col=master_index_col,
        cell_type_col=cell_type_col,
        x_col=x_col,
        y_col=y_col,
        populations=populations
    )


def _compute_nearest_population_distances_matrix(
    roi_df: pd.DataFrame,
    *,
    roi_id,
    master_index_col: str,
    cell_type_col: str,
    x_col: str,
    y_col: str,
    populations: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Vectorized nearest-distance computation using KDTree, per population.
    """
    coords_all = roi_df[[x_col, y_col]].to_numpy(dtype=float, copy=True)
    pops_all = roi_df[cell_type_col].to_numpy()

    if populations is None:
        populations = pd.unique(pops_all).tolist()

    data = {
        'master_index': roi_df[master_index_col].to_numpy(),
        'roi': np.full(len(roi_df), roi_id)
    }

    for pop in populations:
        pop_mask = pops_all == pop
        pop_coords = coords_all[pop_mask]

        if pop_coords.size == 0:
            data[str(pop)] = np.full(len(roi_df), np.nan)
            continue

        tree = cKDTree(pop_coords)
        # Default: distance to nearest member of this population
        dists, _ = tree.query(coords_all, k=1)

        # If reference cell is in same population, exclude itself
        if pop_coords.shape[0] > 1 and np.any(pop_mask):
            same_coords = coords_all[pop_mask]
            dists_same, _ = tree.query(same_coords, k=2)
            dists[pop_mask] = dists_same[:, 1]
        elif pop_coords.shape[0] == 1 and np.any(pop_mask):
            # No other members to compare against
            dists[pop_mask] = np.nan

        data[str(pop)] = dists

    return pd.DataFrame(data)

import numpy as np

def shuffle_cell_types_in_roi(
    adata,
    roi_id,
    roi_col,
    cell_type_col,
    inplace=False
):
    """
    Randomly shuffle the cell-type labels for cells in the specified ROI.
    This preserves the total distribution of cell types but breaks
    any spatial arrangement.

    Returns:
    --------
    AnnData
        Either the same object (if inplace=True) or a copy with permuted cell types.
    """
    if inplace:
        adata_shuffled = adata
    else:
        adata_shuffled = adata.copy()
    
    # Extract mask for the ROI
    mask = adata_shuffled.obs[roi_col] == roi_id
    
    # Get the cell types for this ROI
    cell_types_roi = adata_shuffled.obs.loc[mask, cell_type_col]
    
    # Permute the cell types
    shuffled_cell_types = np.random.permutation(cell_types_roi.values)
    
    # Assign them back
    adata_shuffled.obs.loc[mask, cell_type_col] = shuffled_cell_types
    
    return adata_shuffled


def bootstrap_annulus_proportions_parallel(
    adata,
    roi_id,
    roi_col,
    master_index_col,
    cell_type_col,
    x_col,
    y_col,
    annulus_ranges,
    get_annulus_proportions_func,
    n_bootstraps=100,
    n_jobs=-1
):
    """
    1) Compute observed annulus proportions for all cells in `roi_id`.
    2) Perform bootstrap by shuffling cell-type labels in `roi_id` in parallel
       and re-computing annulus proportions for each shuffle.
    3) Return the observed DataFrame plus a list of bootstrapped DataFrames.

    Parameters
    ----------
    adata : AnnData
    roi_id : str or int
        ROI identifier.
    roi_col : str
        Column in adata.obs that indicates the ROI/region.
    master_index_col : str
        Column in adata.obs that has a unique cell index.
    cell_type_col : str
        Column in adata.obs with the cell types to be shuffled.
    x_col, y_col : str
        Column names for the x and y coordinates.
    annulus_ranges : list of (float, float)
        List of (inner_radius, outer_radius) tuples.
    get_annulus_proportions_func : callable
        Function that computes annulus proportions for a single cell (like `get_annulus_proportions`).
    n_bootstraps : int
        Number of bootstrap shuffles.
    n_jobs : int
        Number of cores to use in parallel. -1 uses all available cores.

    Returns
    -------
    observed_df : pd.DataFrame
        The observed annulus proportions (one row per cell + annulus).
    bootstrap_dfs : list of pd.DataFrame
        A list of length `n_bootstraps`, each a DataFrame with the
        annulus proportions under one random shuffle.
    """
    # --- Step 1: Observed proportions
    observed_df = compute_annulus_proportions_for_all_cells(
        adata=adata,
        roi_id=roi_id,
        roi_col=roi_col,
        master_index_col=master_index_col,
        cell_type_col=cell_type_col,
        x_col=x_col,
        y_col=y_col,
        annulus_ranges=annulus_ranges,
        get_annulus_proportions_func=get_annulus_proportions_func
    )

    # --- Step 2: Bootstrap in parallel
    bootstrap_dfs = Parallel(n_jobs=n_jobs)(
        delayed(compute_shuffled_proportions_for_roi)(
            adata=adata,
            roi_id=roi_id,
            roi_col=roi_col,
            master_index_col=master_index_col,
            cell_type_col=cell_type_col,
            x_col=x_col,
            y_col=y_col,
            annulus_ranges=annulus_ranges,
            get_annulus_proportions_func=get_annulus_proportions_func
        )
        for i in range(n_bootstraps)
    )

    return observed_df, bootstrap_dfs


def bootstrap_nearest_population_distances_parallel(
    adata,
    roi_id,
    roi_col,
    master_index_col,
    cell_type_col,
    x_col,
    y_col,
    populations: Optional[Sequence[str]] = None,
    n_bootstraps=100,
    n_jobs=-1
):
    """
    1) Compute observed nearest-population distances for all cells in `roi_id`.
    2) Perform bootstrap by shuffling cell-type labels in `roi_id` in parallel
       and re-computing nearest-population distances for each shuffle.
    3) Return the observed DataFrame plus a list of bootstrapped DataFrames.

    Parameters
    ----------
    adata : AnnData
    roi_id : str or int
        ROI identifier.
    roi_col : str
        Column in adata.obs that indicates the ROI/region.
    master_index_col : str
        Column in adata.obs that has a unique cell index.
    cell_type_col : str
        Column in adata.obs with the cell types to be shuffled.
    x_col, y_col : str
        Column names for the x and y coordinates.
    get_nearest_population_distances_func : callable
        Function that computes nearest distances for a single cell.
    n_bootstraps : int
        Number of bootstrap shuffles.
    n_jobs : int
        Number of cores to use in parallel. -1 uses all available cores.

    Returns
    -------
    observed_df : pd.DataFrame
        Observed nearest-population distances (one row per cell).
    bootstrap_dfs : list of pd.DataFrame
        A list of length `n_bootstraps`, each a DataFrame with the
        nearest-population distances under one random shuffle.
    """
    observed_df = compute_nearest_population_distances_for_all_cells(
        adata=adata,
        roi_id=roi_id,
        roi_col=roi_col,
        master_index_col=master_index_col,
        cell_type_col=cell_type_col,
        x_col=x_col,
        y_col=y_col,
        populations=populations
    )

    bootstrap_dfs = Parallel(n_jobs=n_jobs)(
        delayed(compute_shuffled_nearest_population_distances_for_roi)(
            adata=adata,
            roi_id=roi_id,
            roi_col=roi_col,
            master_index_col=master_index_col,
            cell_type_col=cell_type_col,
            x_col=x_col,
            y_col=y_col,
            populations=populations
        )
        for _ in range(n_bootstraps)
    )

    return observed_df, bootstrap_dfs


def summarize_bootstrap_results(
    observed_df: pd.DataFrame,
    bootstrap_dfs: Sequence[pd.DataFrame],
    *,
    adata: Optional[ad.AnnData] = None,
    source_population_col: str = "cell_type",
    master_index_col: str = "Master_Index",
    id_cols: Sequence[str] = ("master_index", "roi"),
    ddof: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summarize bootstrap results by computing bootstrap means, deltas, and z-scores,
    then aggregate by source population (from AnnData) to return ROI x population matrices.

    Parameters
    ----------
    observed_df : pd.DataFrame
        Observed results (one row per cell).
    bootstrap_dfs : Sequence[pd.DataFrame]
        List of bootstrap DataFrames with matching columns/rows.
    adata : AnnData, optional
        AnnData containing the source population labels in `.obs`.
    source_population_col : str
        Column in `adata.obs` with source population labels.
    master_index_col : str
        Column in `adata.obs` with master index values that match `id_cols`.
    id_cols : Sequence[str]
        Identifier columns to exclude from statistics.
    ddof : int
        Delta degrees of freedom for bootstrap std.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (observed_roi_mat, bootstrap_mean_roi_mat, delta_roi_mat, zscore_roi_mat)
        Each is a DataFrame with ROI rows and source-population columns.
    """
    if not bootstrap_dfs:
        raise ValueError("bootstrap_dfs is empty.")

    if "roi" not in id_cols:
        raise ValueError("'roi' must be included in id_cols to compute ROI-level summaries.")

    if adata is None:
        raise ValueError("adata is required to map master_index to source populations.")

    if master_index_col not in adata.obs.columns:
        raise ValueError(f"{master_index_col!r} not found in adata.obs.")

    if source_population_col not in adata.obs.columns:
        raise ValueError(f"{source_population_col!r} not found in adata.obs.")

    value_cols = [c for c in observed_df.columns if c not in id_cols]

    # Align and stack bootstrap arrays
    aligned_bootstraps = [df[value_cols].to_numpy() for df in bootstrap_dfs]
    boot_stack = np.stack(aligned_bootstraps, axis=0)

    boot_mean = np.nanmean(boot_stack, axis=0)
    boot_std = np.nanstd(boot_stack, axis=0, ddof=ddof)

    observed_values = observed_df[value_cols].to_numpy()
    delta_values = observed_values - boot_mean
    zscore_values = np.divide(
        delta_values,
        boot_std,
        out=np.full_like(delta_values, np.nan, dtype=float),
        where=boot_std != 0,
    )

    bootstrap_mean_df = observed_df[list(id_cols)].copy()
    bootstrap_mean_df[value_cols] = boot_mean

    delta_df = observed_df[list(id_cols)].copy()
    delta_df[value_cols] = delta_values

    zscore_df = observed_df[list(id_cols)].copy()
    zscore_df[value_cols] = zscore_values

    master_to_pop = (
        adata.obs[[master_index_col, source_population_col]]
        .drop_duplicates()
        .set_index(master_index_col)[source_population_col]
        .to_dict()
    )

    master_index_df_col = (
        master_index_col if master_index_col in observed_df.columns else id_cols[0]
    )

    def _roi_source_pop_matrix(df: pd.DataFrame) -> pd.DataFrame:
        out = df[list(id_cols)].copy()
        out["source_population"] = out[master_index_df_col].map(master_to_pop)

        if out["source_population"].isna().any():
            missing = out.loc[out["source_population"].isna(), master_index_df_col].unique()
            raise ValueError(
                "Missing source population labels for master_index values: "
                + ", ".join(map(str, missing))
            )

        values = df[value_cols].copy()
        values["roi"] = out["roi"].values
        values["source_population"] = out["source_population"].values

        roi_source_mean = (
            values.groupby(["roi", "source_population"], observed=True)[value_cols]
            .mean()
        )

        return roi_source_mean

    observed_roi_mat = _roi_source_pop_matrix(observed_df)
    bootstrap_mean_roi_mat = _roi_source_pop_matrix(bootstrap_mean_df)
    delta_roi_mat = _roi_source_pop_matrix(delta_df)
    zscore_roi_mat = _roi_source_pop_matrix(zscore_df)

    return observed_roi_mat, bootstrap_mean_roi_mat, delta_roi_mat, zscore_roi_mat


def bootstrap_nearest_population_distances_all_rois(
    adata,
    *,
    roi_col: str = "ROI",
    master_index_col: str = "Master_Index",
    cell_type_col: str = "cell_type",
    x_col: str = "X_loc",
    y_col: str = "Y_loc",
    populations: Optional[Sequence[str]] = None,
    roi_ids: Optional[Sequence] = None,
    n_bootstraps: int = 100,
    n_jobs: int = -1,
    source_population_col: str = "cell_type",
    ddof: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run nearest-population distance bootstraps for each ROI and return
    summary matrices for all ROIs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (observed_all, bootstrap_mean_all, delta_all, zscore_all)
        Each DataFrame has a MultiIndex (roi, source_population) and
        target populations as columns.
    """
    if roi_ids is None:
        roi_ids = pd.unique(adata.obs[roi_col]).tolist()

    observed_list: list[pd.DataFrame] = []
    bootmean_list: list[pd.DataFrame] = []
    delta_list: list[pd.DataFrame] = []
    zscore_list: list[pd.DataFrame] = []

    for roi_id in tqdm(roi_ids, desc="ROIs"):
        observed_df, bootstrap_dfs = bootstrap_nearest_population_distances_parallel(
            adata=adata,
            roi_id=roi_id,
            roi_col=roi_col,
            master_index_col=master_index_col,
            cell_type_col=cell_type_col,
            x_col=x_col,
            y_col=y_col,
            populations=populations,
            n_bootstraps=n_bootstraps,
            n_jobs=n_jobs
        )

        observed_roi, bootmean_roi, delta_roi, zscore_roi = summarize_bootstrap_results(
            observed_df=observed_df,
            bootstrap_dfs=bootstrap_dfs,
            adata=adata,
            source_population_col=source_population_col,
            master_index_col=master_index_col,
            id_cols=("master_index", "roi"),
            ddof=ddof
        )

        observed_list.append(observed_roi)
        bootmean_list.append(bootmean_roi)
        delta_list.append(delta_roi)
        zscore_list.append(zscore_roi)

    observed_all = pd.concat(observed_list, axis=0)
    bootmean_all = pd.concat(bootmean_list, axis=0)
    delta_all = pd.concat(delta_list, axis=0)
    zscore_all = pd.concat(zscore_list, axis=0)

    return observed_all, bootmean_all, delta_all, zscore_all

def compute_shuffled_proportions_for_roi(
    adata,
    roi_id,
    roi_col,
    master_index_col,
    cell_type_col,
    x_col,
    y_col,
    annulus_ranges,
    get_annulus_proportions_func
):
    """
    1. Shuffle the cell types within `roi_id` in a copy of `adata`.
    2. Compute annulus proportions for all cells in that ROI.
    3. Return the resulting DataFrame.
    """
    # Shuffle
    adata_shuffled = shuffle_cell_types_in_roi(
        adata=adata,
        roi_id=roi_id,
        roi_col=roi_col,
        cell_type_col=cell_type_col,
        inplace=False  # important to keep original adata intact
    )
    # Recompute annulus proportions on the shuffled data
    random_df = compute_annulus_proportions_for_all_cells(
        adata=adata_shuffled,
        roi_id=roi_id,
        roi_col=roi_col,
        master_index_col=master_index_col,
        cell_type_col=cell_type_col,
        x_col=x_col,
        y_col=y_col,
        annulus_ranges=annulus_ranges,
        get_annulus_proportions_func=get_annulus_proportions_func
    )
    return random_df


def compute_shuffled_nearest_population_distances_for_roi(
    adata,
    roi_id,
    roi_col,
    master_index_col,
    cell_type_col,
    x_col,
    y_col,
    populations: Optional[Sequence[str]] = None
):
    """
    1. Shuffle the cell types within `roi_id` in a copy of `adata`.
    2. Compute nearest-population distances for all cells in that ROI.
    3. Return the resulting DataFrame.
    """
    adata_shuffled = shuffle_cell_types_in_roi(
        adata=adata,
        roi_id=roi_id,
        roi_col=roi_col,
        cell_type_col=cell_type_col,
        inplace=False
    )

    random_df = compute_nearest_population_distances_for_all_cells(
        adata=adata_shuffled,
        roi_id=roi_id,
        roi_col=roi_col,
        master_index_col=master_index_col,
        cell_type_col=cell_type_col,
        x_col=x_col,
        y_col=y_col,
        populations=populations
    )
    return random_df