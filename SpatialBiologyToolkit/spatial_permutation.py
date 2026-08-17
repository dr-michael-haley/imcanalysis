"""Population overlap tests for categorical spatial-region masks.

The public functions in this module compare cell-population centres with a
categorical raster such as a Pixie pixel-environment mask.  The null model
preserves the number of cells in every population within each ROI and samples
their centres uniformly from the pixels selected as tissue.

For a population containing ``n`` cells and an environment occupying ``K`` of
``N`` valid pixels, its randomized overlap count has a hypergeometric
distribution.  Sampling that marginal distribution is equivalent to the
coordinate-sampling and label-shuffling implementation previously used in
notebooks, but avoids materializing randomized coordinates and crosstabs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import tifffile


LOGGER = logging.getLogger(__name__)

CoordinateRounding = Literal["truncate", "floor", "round"]


def resolve_spatial_mask_paths(
    mask_folder: str | Path,
    rois: Sequence[Any],
    *,
    mask_pattern: str = "{roi}_pixel_mask.tiff",
) -> dict[str, Path]:
    """Resolve exactly one categorical mask path for each requested ROI.

    Parameters
    ----------
    mask_folder
        Folder containing the masks.
    rois
        ROI identifiers.  Their string representations are inserted into
        ``mask_pattern``.
    mask_pattern
        Relative filename pattern containing ``{roi}``.  Exact formatting is
        deliberately used instead of substring matching so, for example,
        ``acq_1`` cannot silently select the mask for ``acq_10``.

    Returns
    -------
    dict
        Mapping from string ROI identifiers to resolved mask paths.

    Raises
    ------
    FileNotFoundError
        If the mask folder or one or more expected masks do not exist.
    ValueError
        If the ROI list or pattern is invalid, contains duplicates, or would
        resolve outside ``mask_folder``.
    """

    folder = Path(mask_folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Mask folder does not exist: {folder}")
    if "{roi}" not in mask_pattern:
        raise ValueError("mask_pattern must contain the '{roi}' placeholder")

    roi_keys = [str(roi) for roi in rois]
    if not roi_keys:
        raise ValueError("At least one ROI is required")
    duplicates = sorted({roi for roi in roi_keys if roi_keys.count(roi) > 1})
    if duplicates:
        raise ValueError(f"Duplicate ROI identifiers after string conversion: {duplicates}")

    folder_resolved = folder.resolve()
    resolved: dict[str, Path] = {}
    missing: list[Path] = []
    for roi in roi_keys:
        relative = Path(mask_pattern.format(roi=roi))
        if relative.is_absolute():
            raise ValueError("mask_pattern must resolve relative to mask_folder")
        candidate = (folder / relative).resolve()
        if not candidate.is_relative_to(folder_resolved):
            raise ValueError(
                f"Mask pattern for ROI {roi!r} resolves outside mask_folder: {candidate}"
            )
        if not candidate.is_file():
            missing.append(candidate)
        resolved[roi] = candidate

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        remainder = len(missing) - 10
        if remainder:
            preview += f"\n  - ... and {remainder} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(roi_keys)} expected ROI masks:\n{preview}"
        )
    return resolved


def spatial_mask_alignment_qc(
    adata_or_obs: Any,
    mask_folder: str | Path,
    *,
    rois: Sequence[Any] | None = None,
    tissue_include_values: Sequence[Any] | None = None,
    tissue_exclude_values: Sequence[Any] | None = (0,),
    x_col: str = "X_loc",
    y_col: str = "Y_loc",
    pop_col: str = "population",
    roi_col: str = "ROI",
    mask_pattern: str = "{roi}_pixel_mask.tiff",
    coordinate_rounding: CoordinateRounding = "truncate",
) -> pd.DataFrame:
    """Check ROI/mask matching and cell-coordinate coverage before analysis.

    The returned table has one row per ROI and reports mask dimensions, mask
    labels, missing population labels, non-finite coordinates, out-of-bounds
    cells, and cells excluded by the tissue rule.  ``tissue_include_values``
    takes precedence when both tissue arguments are supplied.
    """

    obs, roi_keys = _prepare_obs(
        adata_or_obs,
        rois=rois,
        required_columns=(roi_col, pop_col, x_col, y_col),
        roi_col=roi_col,
    )
    mask_paths = resolve_spatial_mask_paths(
        mask_folder, roi_keys, mask_pattern=mask_pattern
    )
    grouped = _group_obs_by_roi(obs)

    rows: list[dict[str, Any]] = []
    for roi in roi_keys:
        frame = grouped[roi]
        mask = _read_label_mask(mask_paths[roi])
        tissue_mask = _select_tissue_pixels(
            mask,
            tissue_include_values=tissue_include_values,
            tissue_exclude_values=tissue_exclude_values,
        )

        coordinates = frame[[x_col, y_col]].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(coordinates).all(axis=1)
        x_indices, y_indices = _coordinate_indices(
            coordinates[finite], coordinate_rounding=coordinate_rounding
        )
        in_bounds_finite = (
            (x_indices >= 0)
            & (x_indices < mask.shape[1])
            & (y_indices >= 0)
            & (y_indices < mask.shape[0])
        )
        n_in_bounds = int(in_bounds_finite.sum())
        if n_in_bounds:
            in_tissue = tissue_mask[
                y_indices[in_bounds_finite], x_indices[in_bounds_finite]
            ]
            n_in_tissue = int(in_tissue.sum())
        else:
            n_in_tissue = 0

        n_cells = len(frame)
        valid_labels = np.unique(mask[tissue_mask])
        rows.append(
            {
                "roi": roi,
                "mask_path": str(mask_paths[roi]),
                "mask_height": int(mask.shape[0]),
                "mask_width": int(mask.shape[1]),
                "environment_values": ";".join(map(str, valid_labels.tolist())),
                "n_valid_tissue_pixels": int(tissue_mask.sum()),
                "n_cells": int(n_cells),
                "n_missing_population": int(frame[pop_col].isna().sum()),
                "n_nonfinite_coordinates": int((~finite).sum()),
                "n_out_of_bounds": int(finite.sum() - n_in_bounds),
                "n_in_bounds": n_in_bounds,
                "n_excluded_by_tissue": int(n_in_bounds - n_in_tissue),
                "n_in_tissue": n_in_tissue,
                "fraction_in_bounds": n_in_bounds / n_cells if n_cells else np.nan,
                "fraction_in_tissue": n_in_tissue / n_cells if n_cells else np.nan,
            }
        )
    return pd.DataFrame(rows)


def spatial_permutation_zscores(
    adata_or_obs: Any,
    mask_folder: str | Path,
    *,
    pixel_dictionary: Mapping[Any, Any] | None = None,
    rois: Sequence[Any] | None = None,
    n_permutations: int = 1000,
    tissue_include_values: Sequence[Any] | None = None,
    tissue_exclude_values: Sequence[Any] | None = (0,),
    x_col: str = "X_loc",
    y_col: str = "Y_loc",
    pop_col: str = "population",
    roi_col: str = "ROI",
    mask_pattern: str = "{roi}_pixel_mask.tiff",
    coordinate_rounding: CoordinateRounding = "truncate",
    n_jobs: int = 1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Calculate per-ROI population/environment permutation z-scores.

    Cell centres are assigned their observed mask value using ``mask[y, x]``.
    Within each ROI, the null model preserves each population's analysed cell
    count and samples locations uniformly without replacement from valid
    tissue pixels.  The randomized count for each population/environment pair
    is therefore sampled directly from its equivalent hypergeometric marginal
    distribution.

    Parameters
    ----------
    adata_or_obs
        An AnnData-like object with ``.obs``, or an observation DataFrame.
    mask_folder
        Folder containing one categorical mask per ROI.
    pixel_dictionary
        Optional mapping from raster values to readable environment names.
        When supplied, a ``pixel_mapped`` result column is added.
    rois
        Optional ROI subset.  By default all non-null ROI values are used in
        their order of appearance.
    n_permutations
        Number of hypergeometric draws used to estimate the null mean and
        sample standard deviation.  Must be at least two.
    tissue_include_values, tissue_exclude_values
        Raster values defining the valid sampling space.  Include values take
        precedence; with neither argument all mask pixels are valid.
    x_col, y_col, pop_col, roi_col
        Observation-column names.
    mask_pattern
        Exact relative mask filename pattern containing ``{roi}``.
    coordinate_rounding
        How floating cell centres become pixel indices.  ``"truncate"``
        reproduces NumPy integer casting used by the legacy notebook.
    n_jobs
        Number of ROI-level worker threads understood by Joblib; ``-1`` uses
        all available CPUs.  Random results remain stable across worker counts.
    random_state
        Seed for reproducible permutations.

    Returns
    -------
    pandas.DataFrame
        Long-form rows for every observed ROI population and valid mask value.
        Core columns are ``roi``, ``pixel``, ``population``, ``observed``,
        ``perm_mean``, ``perm_std``, and ``z_score``.  Additional columns record
        the population count, environment area, analysed-cell count, and null
        settings needed to audit the result.

    Notes
    -----
    A positive z-score means more population centres lie in the environment
    than expected from that environment's tissue area.  A negative value means
    fewer than expected.  This tests area overlap, not cell-cell interaction or
    within-environment spatial clustering.
    """

    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 2:
        raise ValueError("n_permutations must be an integer of at least 2")
    if not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    _validate_coordinate_rounding(coordinate_rounding)

    obs, roi_keys = _prepare_obs(
        adata_or_obs,
        rois=rois,
        required_columns=(roi_col, pop_col, x_col, y_col),
        roi_col=roi_col,
    )
    mask_paths = resolve_spatial_mask_paths(
        mask_folder, roi_keys, mask_pattern=mask_pattern
    )
    grouped = _group_obs_by_roi(obs)
    roi_seeds = np.random.SeedSequence(random_state).spawn(len(roi_keys))

    def run_one(position: int) -> pd.DataFrame:
        roi = roi_keys[position]
        return _spatial_permutation_roi(
            roi,
            grouped[roi],
            mask_paths[roi],
            n_permutations=int(n_permutations),
            tissue_include_values=tissue_include_values,
            tissue_exclude_values=tissue_exclude_values,
            x_col=x_col,
            y_col=y_col,
            pop_col=pop_col,
            coordinate_rounding=coordinate_rounding,
            seed=roi_seeds[position],
        )

    LOGGER.info(
        "Calculating spatial permutation z-scores for %d ROIs with %d permutations",
        len(roi_keys),
        n_permutations,
    )
    if n_jobs == 1:
        frames = [run_one(position) for position in range(len(roi_keys))]
    else:
        from joblib import Parallel, delayed

        frames = Parallel(n_jobs=int(n_jobs), prefer="threads")(
            delayed(run_one)(position) for position in range(len(roi_keys))
        )

    result = pd.concat(frames, ignore_index=True)
    result["random_state"] = random_state
    if pixel_dictionary is not None:
        result["pixel_mapped"] = result["pixel"].map(pixel_dictionary)
    return result


def _spatial_permutation_roi(
    roi: str,
    frame: pd.DataFrame,
    mask_path: Path,
    *,
    n_permutations: int,
    tissue_include_values: Sequence[Any] | None,
    tissue_exclude_values: Sequence[Any] | None,
    x_col: str,
    y_col: str,
    pop_col: str,
    coordinate_rounding: CoordinateRounding,
    seed: np.random.SeedSequence,
) -> pd.DataFrame:
    mask = _read_label_mask(mask_path)
    tissue_mask = _select_tissue_pixels(
        mask,
        tissue_include_values=tissue_include_values,
        tissue_exclude_values=tissue_exclude_values,
    )
    environment_values, environment_pixels = np.unique(
        mask[tissue_mask], return_counts=True
    )
    n_valid_pixels = int(environment_pixels.sum())

    coordinates = frame[[x_col, y_col]].to_numpy(dtype=float, copy=True)
    finite = np.isfinite(coordinates).all(axis=1)
    has_population = frame[pop_col].notna().to_numpy()
    eligible = finite & has_population
    x_indices, y_indices = _coordinate_indices(
        coordinates[eligible], coordinate_rounding=coordinate_rounding
    )
    in_bounds = (
        (x_indices >= 0)
        & (x_indices < mask.shape[1])
        & (y_indices >= 0)
        & (y_indices < mask.shape[0])
    )
    eligible_positions = np.flatnonzero(eligible)
    bounded_positions = eligible_positions[in_bounds]
    bounded_x = x_indices[in_bounds]
    bounded_y = y_indices[in_bounds]
    in_tissue = tissue_mask[bounded_y, bounded_x]
    analysed_positions = bounded_positions[in_tissue]
    analysed_x = bounded_x[in_tissue]
    analysed_y = bounded_y[in_tissue]

    if not len(analysed_positions):
        raise ValueError(f"ROI {roi!r} has no cells with valid labels inside selected tissue")
    if len(analysed_positions) > n_valid_pixels:
        raise ValueError(
            f"ROI {roi!r} has {len(analysed_positions)} analysed cells but only "
            f"{n_valid_pixels} valid tissue pixels; sampling without replacement is impossible"
        )

    populations = frame.iloc[analysed_positions][pop_col].astype("object").to_numpy()
    observed_pixels = mask[analysed_y, analysed_x]
    population_values = sorted(pd.unique(populations).tolist(), key=str)
    population_counts = pd.Series(populations).value_counts(sort=False)
    observed_counts = (
        pd.DataFrame({"pixel": observed_pixels, "population": populations})
        .groupby(["pixel", "population"], observed=True)
        .size()
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for pixel, n_environment_pixels in zip(environment_values, environment_pixels):
        environment_count = int(n_environment_pixels)
        for population in population_values:
            n_population_cells = int(population_counts.loc[population])
            draws = rng.hypergeometric(
                environment_count,
                n_valid_pixels - environment_count,
                n_population_cells,
                size=n_permutations,
            )
            perm_mean = float(draws.mean())
            perm_std = float(draws.std(ddof=1))
            observed = int(observed_counts.get((pixel, population), 0))
            z_score = (
                (observed - perm_mean) / perm_std if perm_std > 0 else np.nan
            )
            rows.append(
                {
                    "roi": roi,
                    "pixel": pixel,
                    "population": population,
                    "observed": observed,
                    "perm_mean": perm_mean,
                    "perm_std": perm_std,
                    "z_score": z_score,
                    "n_population_cells": n_population_cells,
                    "n_environment_pixels": environment_count,
                    "environment_fraction": environment_count / n_valid_pixels,
                    "n_valid_tissue_pixels": n_valid_pixels,
                    "n_roi_cells": int(len(frame)),
                    "n_analysed_cells": int(len(analysed_positions)),
                    "n_out_of_bounds_cells": int(in_bounds.size - in_bounds.sum()),
                    "n_excluded_tissue_cells": int(in_tissue.size - in_tissue.sum()),
                    "n_missing_population_cells": int((~has_population).sum()),
                    "n_permutations": n_permutations,
                }
            )
    return pd.DataFrame(rows)


def _prepare_obs(
    adata_or_obs: Any,
    *,
    rois: Sequence[Any] | None,
    required_columns: Sequence[str],
    roi_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    if isinstance(adata_or_obs, pd.DataFrame):
        source = adata_or_obs
    elif hasattr(adata_or_obs, "obs"):
        source = adata_or_obs.obs
    else:
        raise TypeError("adata_or_obs must be a pandas DataFrame or expose an .obs DataFrame")
    if not isinstance(source, pd.DataFrame):
        raise TypeError("adata_or_obs.obs must be a pandas DataFrame")

    missing_columns = [column for column in required_columns if column not in source]
    if missing_columns:
        raise KeyError(f"Missing required observation columns: {missing_columns}")
    obs = source.loc[:, list(dict.fromkeys(required_columns))].copy()
    obs["_sbt_roi_key"] = obs[roi_col].astype("string")
    available = obs["_sbt_roi_key"].dropna().drop_duplicates().astype(str).tolist()
    if rois is None:
        roi_keys = available
    else:
        roi_keys = [str(roi) for roi in rois]
        if not roi_keys:
            raise ValueError("rois cannot be empty")
        duplicates = sorted({roi for roi in roi_keys if roi_keys.count(roi) > 1})
        if duplicates:
            raise ValueError(f"Duplicate requested ROI identifiers: {duplicates}")
        missing_rois = [roi for roi in roi_keys if roi not in set(available)]
        if missing_rois:
            raise ValueError(f"Requested ROIs not present in {roi_col!r}: {missing_rois}")
    if not roi_keys:
        raise ValueError(f"No non-null ROI values found in observation column {roi_col!r}")
    return obs, roi_keys


def _group_obs_by_roi(obs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(roi): frame.drop(columns="_sbt_roi_key")
        for roi, frame in obs.groupby("_sbt_roi_key", observed=True, sort=False)
    }


def _read_label_mask(path: Path) -> np.ndarray:
    mask = np.asarray(tifffile.imread(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D categorical mask at {path}, found shape {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        if not np.all(np.isfinite(mask)) or not np.all(mask == np.trunc(mask)):
            raise ValueError(f"Mask values must be finite integer labels: {path}")
    return mask


def _select_tissue_pixels(
    mask: np.ndarray,
    *,
    tissue_include_values: Sequence[Any] | None,
    tissue_exclude_values: Sequence[Any] | None,
) -> np.ndarray:
    if tissue_include_values is not None:
        tissue_mask = np.isin(mask, list(tissue_include_values))
    elif tissue_exclude_values is not None:
        tissue_mask = ~np.isin(mask, list(tissue_exclude_values))
    else:
        tissue_mask = np.ones(mask.shape, dtype=bool)
    if not tissue_mask.any():
        raise ValueError("The tissue selection contains no valid mask pixels")
    return tissue_mask


def _coordinate_indices(
    coordinates: np.ndarray,
    *,
    coordinate_rounding: CoordinateRounding,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_coordinate_rounding(coordinate_rounding)
    if coordinate_rounding == "truncate":
        rounded = np.trunc(coordinates)
    elif coordinate_rounding == "floor":
        rounded = np.floor(coordinates)
    else:
        rounded = np.rint(coordinates)
    return rounded[:, 0].astype(np.int64), rounded[:, 1].astype(np.int64)


def _validate_coordinate_rounding(value: str) -> None:
    if value not in {"truncate", "floor", "round"}:
        raise ValueError(
            "coordinate_rounding must be one of 'truncate', 'floor', or 'round'"
        )
