"""ROI-scoped AnnData access; never densify an entire expression matrix."""
import numpy as np


def get_values(dataset, roi, reference):
    rows = dataset.rows(roi)
    if reference.kind == 'obs':
        if reference.key not in dataset.obs:
            raise ValueError(f'Unknown obs column {reference.key!r}.')
        return dataset.obs[reference.key].iloc[rows].to_numpy()
    adata = dataset.adata
    if not adata.var_names.is_unique:
        raise ValueError('var_names must be unique for expression mapping.')
    if reference.key not in adata.var_names:
        raise ValueError(f'Unknown expression marker {reference.key!r}.')
    view = adata[rows, [reference.key]]
    matrix = view.X if reference.layer is None else view.layers[reference.layer]
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    return np.asarray(matrix).ravel()


def numeric_values(dataset, roi, reference):
    import pandas as pd
    return pd.to_numeric(get_values(dataset, roi, reference), errors='raise').astype(float)


def conditions(dataset, roi, rules):
    keep = np.ones(len(dataset.rows(roi)), dtype=bool)
    for rule in rules:
        values = get_values(dataset, roi, rule.value)
        if rule.op in ('in', 'eq'):
            import pandas as pd
            selected = pd.Series(values).isin(rule.values).to_numpy()
        else:
            values = np.asarray(values, dtype=float)
            compare = {'gt': np.greater, 'ge': np.greater_equal, 'lt': np.less, 'le': np.less_equal}[rule.op]
            selected = compare(values, rule.values[0]) & np.isfinite(values)
        keep &= selected
    return keep


def cell_ids(dataset, roi):
    frame = dataset.observations(roi)
    if dataset.label_obs not in frame:
        raise ValueError(f'Missing cell ID column {dataset.label_obs!r}.')
    values = frame[dataset.label_obs].to_numpy()
    if values.dtype.kind not in 'iu':
        numeric = np.asarray(values, dtype=float)
        if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
            raise ValueError('Cell IDs must be finite positive integers.')
        values = numeric.astype(np.int64)
    if np.any(values <= 0) or len(np.unique(values)) != len(values):
        raise ValueError(f'Cell IDs must be unique and positive within ROI {roi}.')
    return values


def categorical_palette(dataset, column, explicit):
    """Stable cohort palette, independent of the ROI subset and row order."""
    import pandas as pd
    import matplotlib as mpl
    from matplotlib.colors import to_hex
    series = dataset.obs[column]
    categories = (list(series.cat.categories) if isinstance(series.dtype, pd.CategoricalDtype)
                  else sorted(series.dropna().astype(str).unique()))
    labels = [str(c) for c in categories]
    if len(set(labels)) != len(labels):
        raise ValueError('Distinct population categories collapse to the same string.')
    stored = dataset.adata.uns.get(f'{column}_colormap', {})
    colors = dataset.adata.uns.get(f'{column}_colors', [])
    # Preserve the first twenty legacy colours, extend before repeating.
    palette = [to_hex(c) for name in ('tab20', 'tab20b', 'tab20c') for c in mpl.colormaps[name].colors]
    return {label: to_hex(explicit.get(label, stored.get(label, colors[i] if i < len(colors)
                         else palette[i % len(palette)]))) for i, label in enumerate(labels)}
