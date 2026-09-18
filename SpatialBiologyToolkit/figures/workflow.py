"""Notebook conveniences shared with file-based figure workflows.

Scientific imports stay here, outside the lightweight recipe schema.
"""
from pathlib import Path
import hashlib


def file_record(path, *, digest=False):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    record = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if digest:
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def read_h5ad(path):
    """Load in memory, with read-only compatibility for newer nullable encodings.

    Older SBT environments lack these readers. Never repair the user's file in
    place. Prefer the installed native readers whenever they are available.
    """
    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    from anndata._io.specs import _REGISTRY, IOSpec
    string_spec = IOSpec('nullable-string-array', '0.1.0')
    if (h5py.Group, string_spec, frozenset()) not in _REGISTRY.read:
        @_REGISTRY.register_read(h5py.Group, string_spec)
        def nullable_string(group, _reader):
            values = _reader.read_elem(group['values'])
            mask = np.asarray(_reader.read_elem(group['mask']), dtype=bool)
            return pd.array(np.where(mask, pd.NA, values), dtype='string')
    null_spec = IOSpec('null', '0.1.0')
    if (h5py.Dataset, null_spec, frozenset()) not in _REGISTRY.read:
        @_REGISTRY.register_read(h5py.Dataset, null_spec)
        def null_value(element, _reader):
            return None
    return ad.read_h5ad(path)


def map_observations(dataset, path, *, source, columns, key, overwrite):
    import pandas as pd
    if dataset.obs is None or source not in dataset.obs:
        raise ValueError(f'Missing observation source {source!r}.')
    if isinstance(columns, str):
        columns = [columns]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Supply distinct output columns to map.')
    table = pd.read_csv(path, dtype={key: str})
    if not {key, *columns} <= set(table.columns):
        raise ValueError(f'Mapping CSV needs {key!r} and {columns!r}.')
    if table[key].isna().any() or table[key].duplicated().any():
        raise ValueError('Mapping keys must be unique and nonmissing.')
    if set(columns) & {dataset.roi_obs, dataset.label_obs, dataset.x_obs, dataset.y_obs}:
        raise ValueError('Annotation mapping cannot replace spatial identity columns.')
    if not overwrite and set(columns) & set(dataset.obs.columns):
        raise ValueError('Annotation columns already exist; use overwrite=True explicitly.')
    if dataset.obs[source].isna().any():
        raise ValueError(f'Missing values in mapping source {source!r}.')
    table = table.set_index(key)
    keys = dataset.obs[source].astype(str)
    missing = sorted(set(keys) - set(table.index))
    if missing:
        raise ValueError(f'Unmapped observation values: {missing}')
    annotations = {}
    for column in columns:
        values = keys.map(table[column])
        if values.isna().any():
            raise ValueError(f'Missing mapped labels for {column!r}.')
        annotations[column] = values.astype('category')
    # Commit only after every output column validates. Expression data stays shared.
    dataset.obs = dataset.obs.assign(**annotations)
    dataset.provenance.setdefault('mappings', []).append(dict(
        **file_record(path, digest=True), source=source, key=key, columns=list(columns)))


def image_info(path):
    """Read headers only; the preflight never decompresses full image arrays."""
    if Path(path).suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            return tif.series[0].shape, tif.series[0].dtype.kind
    from PIL import Image
    with Image.open(path) as image:
        channels = len(image.getbands())
        return (image.height, image.width) + ((channels,) if channels > 1 else ()), 'u'


class PreflightReport:
    def __init__(self, table, dataset):
        self.table, self.dataset = table, dataset

    @property
    def eligible_rois(self):
        return self.table.index[self.table['eligible']].tolist()

    def _repr_html_(self):
        return self.table._repr_html_()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(path)
        return path

    def select(self, n=1, *, method='first', seed=0, balance_by=None):
        """Deterministic ROI selection, optionally round-robin across cases/samples."""
        import numpy as np
        if method not in ('first', 'most_cells', 'random') or n < 1:
            raise ValueError('Use a positive n and first, most_cells, or random selection.')
        candidates = self.table.loc[self.eligible_rois]
        if candidates.empty:
            raise ValueError('No eligible ROIs; inspect the preflight report.')
        if method == 'most_cells':
            candidates = candidates.sort_values('selected_cells', ascending=False, kind='stable')
        elif method == 'random':
            candidates = candidates.iloc[np.random.default_rng(seed).permutation(len(candidates))]
        names = candidates.index.tolist()
        if balance_by is not None:
            if self.dataset.obs is None or balance_by not in self.dataset.obs:
                raise ValueError(f'Unknown balancing column {balance_by!r}.')
            buckets = {}
            for roi in names:
                values = self.dataset.observations(roi)[balance_by]
                if values.isna().any() or values.nunique() != 1:
                    raise ValueError(f'{balance_by!r} must identify one nonmissing group per ROI ({roi}).')
                buckets.setdefault(str(values.iloc[0]), []).append(roi)
            names = []
            while any(buckets.values()):
                for bucket in buckets.values():
                    if bucket:
                        names.append(bucket.pop(0))
        return names[:n]


def preflight(recipe, dataset, *, rois=None):
    """Report missing inputs; raise on ambiguous matches or malformed data.

    Cell counts screen obviously impossible hotspots. Full spatial feasibility
    is still evaluated by the selector at render time.
    """
    import numpy as np
    import pandas as pd
    from .specs import IMC, Image, LabelMask, Populations, Values
    from .values import conditions, numeric_values, cell_ids
    recipe = recipe.resolved()
    rois = sorted(dataset.rois) if rois is None else [str(r) for r in rois]
    if not rois or len(set(rois)) != len(rois) or set(rois) - set(dataset.rois):
        raise ValueError('Supply nonempty, distinct, known ROIs.')
    layers = [layer for panel in recipe.panels for layer in panel.layers]
    cell_layers = any(isinstance(layer, (Populations, Values)) for layer in layers)
    crop = recipe.crop
    area_only = crop.mask_source and not crop.where and crop.score is None and not crop.denominator
    if (crop.mode == 'hotspot' and not area_only) or crop.mode == 'cell':
        if dataset.obs is None:
            raise ValueError('This cell-based crop requires AnnData.')
        for column in (dataset.x_obs, dataset.y_obs):
            if column not in dataset.obs:
                raise ValueError(f'Missing coordinate column {column!r}.')
    refs = [rule.value for rule in crop.where + crop.denominator]
    refs += [crop.score] if crop.score is not None else []
    refs += [layer.value for layer in layers if isinstance(layer, Values)]
    for ref in refs:
        if dataset.adata is None:
            raise ValueError('Cell references require AnnData.')
        collection = dataset.obs.columns if ref.kind == 'obs' else dataset.adata.var_names
        if ref.key not in collection:
            raise ValueError(f'Unknown {ref.kind} reference {ref.key!r}.')
        if ref.layer is not None and ref.layer not in dataset.adata.layers:
            raise ValueError(f'Unknown expression layer {ref.layer!r}.')
    for panel in recipe.panels:
        if panel.scale_bar and panel.scale_bar.unit == 'um' and dataset.pixel_size_um is None:
            raise ValueError('A micrometre scale bar requires pixel_size_um.')
    for layer in layers:
        if isinstance(layer, IMC) and any(channel.scale.mode == 'pooled_quantile' for channel in layer.channels):
            raise ValueError('Pooled quantiles are supported for cell values, not image cohorts.')
        if isinstance(layer, Populations):
            if dataset.obs is None or layer.obs not in dataset.obs:
                raise ValueError(f'Unknown population column {layer.obs!r}.')
            categories = set(dataset.obs[layer.obs].dropna().astype(str))
            if layer.groups and set(layer.groups) - categories:
                raise ValueError(f'Unknown populations: {set(layer.groups) - categories}')
        if isinstance(layer, Image) and layer.source not in dataset.images:
            raise ValueError(f'Unknown image source {layer.source!r}.')
        if isinstance(layer, LabelMask) and layer.source not in dataset.labels:
            raise ValueError(f'Unknown label source {layer.source!r}.')
    if crop.mask_source and crop.mask_source not in dataset.labels:
        raise ValueError(f'Unknown crop label source {crop.mask_source!r}.')
    if cell_layers and (dataset.adata is None or dataset.masks is None):
        raise ValueError('Cell layers require AnnData and cell masks.')
    records = []
    for roi in rois:
        record = dict(ROI=roi, eligible=False, reason='', selected_cells=0, resized_sources=[])
        try:
            reference = dataset.reference_path(roi)
            reference_shape, _ = image_info(reference)
            shape = reference_shape[:2]
            record.update(reference=str(reference), width=shape[1], height=shape[0])
            if roi in crop.roi_bounds or crop.mode == 'bounds':
                from .selection import _validate_bounds
                bounds = crop.roi_bounds.get(roi, crop.bounds)
                if bounds is None:
                    record['reason'] = 'No saved crop bounds for this ROI'
                else:
                    _validate_bounds(bounds, shape)
            checks = []
            if cell_layers:
                cell_ids(dataset, roi)
                checks.append((dataset.masks.match(roi), 'labels'))
            for layer in layers:
                if isinstance(layer, IMC):
                    checks += [(dataset.channel_path(roi, channel.marker), 'imc') for channel in layer.channels]
                elif isinstance(layer, Image):
                    checks.append((dataset.images[layer.source].match(roi), 'image'))
                elif isinstance(layer, LabelMask):
                    checks.append((dataset.labels[layer.source].match(roi), 'labels'))
            if crop.mask_source:
                checks.append((dataset.labels[crop.mask_source].match(roi), 'labels'))
            for path, kind in dict.fromkeys(checks):
                dims, dtype = image_info(path)
                if kind == 'imc' and (len(dims) != 2 or dims != shape):
                    raise ValueError(f'{roi}: IMC image {path} must match the 2D reference grid.')
                if kind == 'labels' and (len(dims) != 2 or dtype not in 'ui'):
                    raise ValueError(f'{roi}: label mask {path} must be a 2D integer TIFF.')
                if kind == 'image' and not (len(dims) == 2 or (len(dims) == 3 and dims[-1] in (3, 4))):
                    raise ValueError(f'{roi}: select a 2D plane or RGB/RGBA image for {path}.')
                if dims[:2] != shape:
                    record['resized_sources'].append(str(path))
            if crop.size and (crop.size[0] > shape[1] or crop.size[1] > shape[0]):
                record['reason'] = 'Reference image smaller than crop'
            elif dataset.adata is not None:
                selected = conditions(dataset, roi, crop.where) & conditions(dataset, roi, crop.denominator)
                frame = dataset.observations(roi)
                if crop.mode == 'hotspot':
                    xs, ys = frame[dataset.x_obs].to_numpy(float), frame[dataset.y_obs].to_numpy(float)
                    valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (ys >= 0) & (xs < shape[1]) & (ys < shape[0])
                    selected &= valid
                    if crop.score is not None:
                        selected &= np.isfinite(numeric_values(dataset, roi, crop.score))
                record['selected_cells'] = int(selected.sum())
                count = record['selected_cells']
                if crop.mode == 'hotspot' and crop.reducer == 'fraction':
                    count = int((conditions(dataset, roi, crop.denominator) & valid).sum())
                if crop.mode == 'hotspot' and not area_only and count < crop.min_cells:
                    record['reason'] = 'Too few qualifying cells for hotspot'
            record['eligible'] = not record['reason']
        except FileNotFoundError as error:
            record['reason'] = str(error)
        records.append(record)
    return PreflightReport(pd.DataFrame(records).set_index('ROI'), dataset)
