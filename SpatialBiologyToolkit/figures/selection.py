"""Deterministic shared views, with O(width + cells) crop-search working arrays."""
import numpy as np
from .values import conditions, numeric_values, cell_ids


def _validate_bounds(bounds, shape):
    x, y, w, h = map(int, bounds)
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > shape[1] or y + h > shape[0]:
        raise ValueError(f'Crop {bounds} lies outside reference shape {shape}.')
    return x, y, w, h


def select_view(context, crop):
    height, width = context.shape
    roi, data = context.roi, context.dataset
    if roi in crop.roi_bounds:
        bounds = _validate_bounds(crop.roi_bounds[roi], context.shape)
        return dict(bounds=list(bounds), mode='override', score=None)
    if crop.mode == 'full':
        return dict(bounds=[0, 0, width, height], mode='full', score=None)
    if crop.mode == 'bounds':
        if crop.bounds is None:
            raise ValueError(f'No saved crop bounds for ROI {roi}.')
        return dict(bounds=list(_validate_bounds(crop.bounds, context.shape)), mode='bounds', score=None)
    cw, ch = min(crop.size[0], width), min(crop.size[1], height)
    x, y = (width - cw) // 2, (height - ch) // 2
    if crop.mode == 'hotspot':
        result = _hotspot(context, crop, cw, ch)
        if result is not None:
            return result
        if crop.fallback == 'error':
            raise ValueError(f'No eligible hotspot window for ROI {roi}.')
        return dict(bounds=[x, y, cw, ch], mode='center_fallback', score=None)
    if crop.mode in ('coordinate', 'cell'):
        center = crop.center
        if crop.mode == 'cell':
            frame = data.observations(roi)
            found = np.flatnonzero(cell_ids(data, roi) == crop.cell_id)
            if not len(found):
                raise ValueError(f'Cell {crop.cell_id} not found in ROI {roi}.')
            center = frame.iloc[found[0]][[data.x_obs, data.y_obs]].to_numpy(dtype=float)
        if not np.isfinite(center).all() or not (0 <= center[0] < width and 0 <= center[1] < height):
            raise ValueError('Crop center must lie within the reference image.')
        x = int(np.clip(round(center[0] - cw / 2), 0, width - cw))
        y = int(np.clip(round(center[1] - ch / 2), 0, height - ch))
    elif crop.mode != 'center':
        x = 0 if crop.mode.endswith('left') else width - cw
        y = 0 if crop.mode.startswith('upper') else height - ch
    return dict(bounds=[x, y, cw, ch], mode=crop.mode, score=None)


def _window_sums(columns, width):
    prefix = np.empty(len(columns) + 1, dtype=np.float64)
    prefix[0] = 0
    np.cumsum(columns, out=prefix[1:])
    return prefix[width:] - prefix[:-width]


def _hotspot(context, crop, cw, ch):
    data, roi = context.dataset, context.roi
    height, width = context.shape
    area_only = bool(crop.mask_source and crop.score is None and not crop.where and not crop.denominator)
    region_mask = context.mask((0, 0, width, height), crop.mask_source) if crop.mask_source else None
    if not area_only:
        frame = data.observations(roi)
        xs = frame[data.x_obs].to_numpy(dtype=float)
        ys = frame[data.y_obs].to_numpy(dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        denominator = conditions(data, roi, crop.denominator) & valid
        selected = conditions(data, roi, crop.where) & denominator
        weights = np.ones(len(frame))
        if crop.score is not None:
            weights = numeric_values(data, roi, crop.score)
            selected &= np.isfinite(weights)
        if crop.reducer in ('count', 'fraction'):
            weights = np.ones(len(frame))
        # Mean/sum min_cells refers to contributing finite cells, not background cells.
        counts = denominator if crop.reducer == 'fraction' else selected
        xi = np.clip(np.rint(xs[valid]).astype(int), 0, width - 1)
        yi = np.clip(np.rint(ys[valid]).astype(int), 0, height - 1)
        order = np.argsort(yi, kind='stable')
        xi, yi = xi[order], yi[order]
        weights = np.where(selected, weights, 0)[valid][order]
        count_values = counts[valid][order].astype(float)
        starts = np.searchsorted(yi, np.arange(height + 1))
        center = (float(xs[selected].mean()), float(ys[selected].mean())) if selected.any() else (width / 2, height / 2)
    else:
        xi, weights, count_values = np.array([], dtype=int), np.array([]), np.array([])
        starts = np.zeros(height + 1, dtype=int)
        center = (width / 2, height / 2)
    sums, counts = np.zeros(width), np.zeros(width)
    coverage = np.zeros(width)
    previous_y = 0
    best = None
    columns = np.unique(np.r_[np.arange(0, width - cw + 1, crop.stride), width - cw]).astype(int)
    rows = np.unique(np.r_[np.arange(0, height - ch + 1, crop.stride), height - ch]).astype(int)
    for iteration, y in enumerate(rows):
        if iteration % 32 == 0 and getattr(context, 'cancelled', None) and context.cancelled():
            raise InterruptedError('Crop selection cancelled.')
        if iteration == 0:
            entering = slice(starts[0], starts[ch])
        else:
            leaving = slice(starts[previous_y], starts[min(y, previous_y + ch)])
            np.add.at(sums, xi[leaving], -weights[leaving])
            np.add.at(counts, xi[leaving], -count_values[leaving])
            entering = slice(starts[max(y, previous_y + ch)], starts[y + ch])
        np.add.at(sums, xi[entering], weights[entering])
        np.add.at(counts, xi[entering], count_values[entering])
        if region_mask is not None:
            if iteration == 0:
                coverage += np.isin(region_mask[:ch], crop.mask_labels).sum(axis=0)
            else:
                coverage -= np.isin(region_mask[previous_y:min(y, previous_y + ch)], crop.mask_labels).sum(axis=0)
                coverage += np.isin(region_mask[max(y, previous_y + ch):y + ch], crop.mask_labels).sum(axis=0)
        previous_y = y
        n = _window_sums(counts, cw)[columns]
        totals = _window_sums(sums, cw)[columns]
        covered = _window_sums(coverage, cw)[columns]
        if area_only:
            scores = covered / (cw * ch) if crop.reducer == 'fraction' else covered
            eligible = covered > 0
        else:
            scores = totals / np.maximum(n, 1) if crop.reducer in ('mean', 'fraction') else totals
            eligible = n >= crop.min_cells
        if region_mask is not None:
            eligible &= covered >= crop.min_coverage * cw * ch
        if not eligible.any():
            continue
        # Only the row maximum can win globally; avoid a Python loop over pixels.
        candidates = np.flatnonzero(eligible & (scores == scores[eligible].max()))
        distances = (columns[candidates] + cw / 2 - center[0])**2 + (y + ch / 2 - center[1])**2
        for idx in candidates[np.argsort(distances, kind='stable')[:1]]:
            x = int(columns[idx])
            distance = (x + cw / 2 - center[0])**2 + (y + ch / 2 - center[1])**2
            key = (float(scores[idx]), -distance, -int(y), -x)
            if best is None or key > best[0]:
                best = (key, dict(bounds=[x, int(y), cw, ch], mode='hotspot',
                                  score=float(scores[idx]), cell_count=int(round(n[idx])),
                                  mask_coverage=float(covered[idx] / (cw * ch)) if region_mask is not None else None,
                                  reducer=crop.reducer, stride=crop.stride,
                                  selection_basis='mask_area' if area_only else 'cell_centers'))
    return best[1] if best is not None else None
