"""Reusable scalar calibration, separate from image rendering."""
import numpy as np


def quantile_bounds(values, quantiles):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError('Cannot calibrate an array with no finite values.')
    return tuple(float(v) for v in np.quantile(finite, quantiles))


def calibrate(scale, rois, loader, *, progress=None, cancelled=None):
    if scale.mode == 'fixed':
        return tuple(scale.limits)
    if scale.mode == 'roi_quantile':
        return None
    scope = scale.rois if scale.rois is not None else rois
    if not scope:
        raise ValueError('Cohort normalisation needs at least one reference ROI.')
    bounds = []
    pooled = []
    for roi in scope:
        if cancelled and cancelled():
            raise InterruptedError('Figure calibration cancelled.')
        array = loader(roi)
        if scale.mode == 'pooled_quantile':
            flat = np.asarray(array).ravel()
            pooled.append(flat[np.isfinite(flat)])
        else:
            bounds.append(quantile_bounds(array, scale.quantiles))
        del array
        if progress:
            progress({'stage': 'calibration', 'roi': roi})
    if scale.mode == 'pooled_quantile':
        return quantile_bounds(np.concatenate(pooled), scale.quantiles)
    result = getattr(np, scale.reduction)(np.asarray(bounds), axis=0)
    return tuple(float(v) for v in result)


def resolve_bounds(scale, calibrated, values):
    return calibrated if calibrated is not None else quantile_bounds(values, scale.quantiles)


def scale_pixels(values, bounds, gamma=1):
    """Fixed endpoints really are fixed, even if the image misses both extremes."""
    lo, hi = bounds
    values = np.asarray(values, dtype=np.float32)
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    output = (values - lo) / (hi - lo)
    np.nan_to_num(output, copy=False, nan=0, posinf=1, neginf=0)
    np.clip(output, 0, 1, out=output)
    if gamma != 1:
        np.power(output, 1 / gamma, out=output)
    return output
