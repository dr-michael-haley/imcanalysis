"""ROI discovery, bounded per-ROI loading, and reference-grid resampling."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re

import numpy as np


def roi_tokens(name):
    return tuple(str(int(t)) if t.isdigit() else t
                 for t in re.findall(r'[a-z]+|\d+', str(name).casefold()))


def image_stem(path):
    stem = path.stem
    return stem[:-4] if stem.casefold().endswith('.ome') else stem


def roi_matches(files, roi):
    """Shared ranking used by new figures and legacy comparison overlays."""
    tokens = roi_tokens(roi)
    exact = [p for p, stem, _ in files if stem.casefold() == str(roi).casefold()]
    normalized = [p for p, _, parts in files if tokens and parts == tokens]
    decorated = [p for p, _, parts in files if tokens and any(
        parts[i:i + len(tokens)] == tokens for i in range(len(parts) - len(tokens) + 1))]
    return exact or normalized or decorated


def match_roi(files, roi):
    """Rank exact, separator-normalized, then decorated names; never guess ties."""
    matches = roi_matches(files, roi)
    if not matches:
        raise FileNotFoundError(f'No image matched ROI {roi!r}.')
    if len(matches) != 1:
        raise ValueError(f'Ambiguous image match for ROI {roi!r}: {matches}')
    return matches[0]


def match_channel(files, marker):
    """Marker tokens differ from ROI numbers: CD45 must never match CD45RO."""
    tokens = tuple(re.split(r'[_\-.\s]+', marker.casefold()))
    exact, equivalent, decorated = [], [], []
    for path, stem, _ in files:
        parts = tuple(re.split(r'[_\-.\s]+', stem.casefold()))
        if stem.casefold() == marker.casefold():
            exact.append(path)
        if parts == tokens:
            equivalent.append(path)
        if any(parts[i:i + len(tokens)] == tokens for i in range(len(parts) - len(tokens) + 1)):
            decorated.append(path)
    matches = exact or equivalent or decorated
    if not matches:
        raise FileNotFoundError(f'No channel image matched marker {marker!r}.')
    if len(matches) > 1:
        raise ValueError(f'Ambiguous channel match for {marker!r}: {matches}')
    return matches[0]


class ImageIndex:
    def __init__(self, folder, *, labels=False):
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(self.folder)
        extensions = {'.tif', '.tiff'} if labels else {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}
        self.files = tuple((p, image_stem(p), roi_tokens(image_stem(p)))
                           for p in sorted(self.folder.rglob('*')) if p.is_file() and p.suffix.lower() in extensions)
        self._matches = {}

    def match(self, roi):
        if roi not in self._matches:
            self._matches[roi] = match_roi(self.files, roi)
        return self._matches[roi]


def read_image(path):
    """Use TIFF memory mapping when possible; compressed images decode on demand."""
    if Path(path).suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        try:
            return tifffile.memmap(path, mode='r')
        except (ValueError, OSError):
            return tifffile.imread(path)
    from PIL import Image
    with Image.open(path) as image:
        if image.mode == 'P':
            image = image.convert('RGBA' if 'transparency' in image.info else 'RGB')
        return np.array(image)


def resample_crop(image, shape, bounds, *, nearest=False):
    """Map reference pixel centers to source centers; allocate only the crop.

    Unlike full-image resize followed by crop this also bounds the destination
    allocation for very large images. No registration/orientation is inferred.
    """
    x, y, width, height = bounds
    if image.shape[:2] == tuple(shape):
        return np.array(image[y:y + height, x:x + width], copy=True)
    from scipy.ndimage import map_coordinates
    ys = (np.arange(y, y + height) + .5) * image.shape[0] / shape[0] - .5
    xs = (np.arange(x, x + width) + .5) * image.shape[1] / shape[1] - .5
    if nearest:
        # Integer indexing preserves even uint64 label IDs without a float cast.
        yi = np.clip(np.floor(ys + .5).astype(int), 0, image.shape[0] - 1)
        xi = np.clip(np.floor(xs + .5).astype(int), 0, image.shape[1] - 1)
        return image[yi[:, None], xi[None, :]].copy()
    coordinates = np.meshgrid(ys, xs, indexing='ij')
    if image.ndim == 2:
        return map_coordinates(image, coordinates, order=1, mode='nearest', prefilter=False)
    return np.stack([map_coordinates(image[..., i], coordinates, order=1, mode='nearest', prefilter=False)
                     for i in range(image.shape[-1])], axis=-1)


class Dataset:
    """Runtime data binding, kept separate from serialisable figure recipes.

    A reference image folder can be supplied for image-only figures. Otherwise
    reference dimensions come from the cell mask, IMC, or first image source.
    """
    def __init__(self, adata=None, *, imc_folder=None, mask_folder=None, image_folders=None,
                 label_folders=None, reference_folder=None, roi_obs='ROI', label_obs='ObjectNumber',
                 x_obs='X_loc', y_obs='Y_loc', pixel_size_um=None, cache_bytes=128 * 1024**2):
        self.adata = adata
        self.obs = None if adata is None else adata.obs
        self.provenance = {}
        self.roi_obs, self.label_obs, self.x_obs, self.y_obs = roi_obs, label_obs, x_obs, y_obs
        self.pixel_size_um = pixel_size_um
        if pixel_size_um is not None and (not np.isfinite(pixel_size_um) or pixel_size_um <= 0):
            raise ValueError('pixel_size_um must be a finite positive reference-pixel size.')
        if cache_bytes < 0:
            raise ValueError('cache_bytes must be nonnegative.')
        self.cache_bytes = cache_bytes
        self.images = {name: ImageIndex(folder) for name, folder in (image_folders or {}).items()}
        self.labels = {name: ImageIndex(folder, labels=True) for name, folder in (label_folders or {}).items()}
        self.masks = ImageIndex(mask_folder, labels=True) if mask_folder is not None else None
        self.reference = ImageIndex(reference_folder) if reference_folder is not None else None
        self.imc = {}
        self._channel_paths = {}
        self.imc_folder = Path(imc_folder).resolve() if imc_folder is not None else None
        if self.imc_folder is not None:
            if not self.imc_folder.is_dir():
                raise FileNotFoundError(self.imc_folder)
            folders = sorted(p for p in self.imc_folder.iterdir() if p.is_dir()) or [self.imc_folder]
            self.imc = {p.name: ImageIndex(p, labels=True) for p in folders}
        self._row_indices = {}
        if adata is not None:
            if roi_obs not in adata.obs:
                raise ValueError(f'AnnData is missing ROI column {roi_obs!r}.')
            series = adata.obs[roi_obs]
            if series.isna().any():
                raise ValueError('ROI identifiers must not be missing.')
            # Convert once, not on every panel; retain only index positions.
            names = series.astype(str)
            if len(set(series.unique())) != names.nunique():
                raise ValueError('Distinct ROI identifiers collapse to the same string.')
            self._row_indices = {str(k): np.asarray(v) for k, v in
                                 names.groupby(names, observed=True, sort=False).indices.items()}
        if self._row_indices:
            self.rois = list(self._row_indices)
        elif self.imc:
            self.rois = list(self.imc)
        else:
            source = self.reference or self.masks or next(iter(self.images.values()), None) or next(iter(self.labels.values()), None)
            self.rois = list(dict.fromkeys(stem for _, stem, _ in source.files)) if source else []
        if not self.rois:
            raise ValueError('No ROIs discovered; supply AnnData or an image/mask source.')

    def rows(self, roi):
        if self.adata is None:
            raise ValueError('This layer/selection requires AnnData.')
        if str(roi) not in self._row_indices:
            raise ValueError(f'No AnnData cells for ROI {roi!r}.')
        return self._row_indices[str(roi)]

    def observations(self, roi):
        return self.obs.iloc[self.rows(roi)]

    @classmethod
    def from_h5ad(cls, path, **kwargs):
        """Read ordinary in-memory AnnData, retaining X, layers and original identities."""
        from .workflow import read_h5ad, file_record
        data = cls(read_h5ad(path), **kwargs)
        data.provenance['anndata'] = file_record(path)
        return data

    def map_obs(self, path, *, source, columns, key='source_population', overwrite=False):
        """Attach categorical annotations to this binding; never edit the source AnnData."""
        from .workflow import map_observations
        map_observations(self, path, source=source, columns=columns, key=key, overwrite=overwrite)
        return self

    def binding(self):
        """JSON-safe source and mapping provenance, shared by notebook and batch exports."""
        def folder(index):
            return None if index is None else str(index.folder)
        return dict(imc_folder=None if self.imc_folder is None else str(self.imc_folder),
                    mask_folder=folder(self.masks), reference_folder=folder(self.reference),
                    image_folders={k: folder(v) for k, v in self.images.items()},
                    label_folders={k: folder(v) for k, v in self.labels.items()},
                    roi_obs=self.roi_obs, label_obs=self.label_obs, x_obs=self.x_obs, y_obs=self.y_obs,
                    pixel_size_um=self.pixel_size_um, provenance=self.provenance)

    def channel_path(self, roi, marker):
        key = (str(roi), marker)
        if key in self._channel_paths:
            return self._channel_paths[key]
        names = [(Path(name), name, roi_tokens(name)) for name in self.imc]
        name = match_roi(names, str(roi)).name
        self._channel_paths[key] = match_channel(self.imc[name].files, marker)
        return self._channel_paths[key]

    def reference_path(self, roi):
        if self.reference:
            return self.reference.match(roi)
        if self.masks:
            return self.masks.match(roi)
        if self.imc:
            name = match_roi([(Path(k), k, roi_tokens(k)) for k in self.imc], roi).name
            files = self.imc[name].files
            if not files:
                raise FileNotFoundError(f'No IMC images for ROI {roi}.')
            return files[0][0]
        source = next(iter(self.images.values()), None) or next(iter(self.labels.values()))
        return source.match(roi)

    def describe(self):
        """Small discovery payload for notebooks and a future recipe editor."""
        return dict(rois=self.rois, image_sources=list(self.images), label_sources=list(self.labels),
                    channels=sorted({stem for source in self.imc.values() for _, stem, _ in source.files}),
                    obs=[] if self.obs is None else list(self.obs.columns),
                    var=[] if self.adata is None else list(self.adata.var_names),
                    layers=[] if self.adata is None else list(self.adata.layers),
                    pixel_size_um=self.pixel_size_um)


class ROIContext:
    """Small bounded read cache; all entries die with this ROI render."""
    def __init__(self, dataset, roi):
        self.dataset, self.roi = dataset, str(roi)
        self.cache, self.sources = OrderedDict(), {}
        reference = dataset.reference_path(self.roi)
        array = self.read(reference)
        self.shape = array.shape[:2]
        if len(self.shape) != 2:
            raise ValueError('Reference image must have two spatial axes.')

    def read(self, path):
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        image = read_image(path)
        stat = Path(path).stat()
        self.sources[key] = dict(shape=list(image.shape), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        size = sum(a.nbytes for a in self.cache.values())
        while self.cache and size + image.nbytes > self.dataset.cache_bytes:
            _, removed = self.cache.popitem(last=False)
            size -= removed.nbytes
        if image.nbytes <= self.dataset.cache_bytes:
            self.cache[key] = image
        return image

    def mask(self, bounds, source=None):
        index = self.dataset.masks if source is None else self.dataset.labels.get(source)
        if index is None:
            raise ValueError(f'Mask source unavailable: {source or "cell masks"}')
        image = self.read(index.match(self.roi))
        if image.ndim != 2 or image.dtype.kind not in 'ui' or (image.dtype.kind == 'i' and image.min() < 0):
            raise ValueError('Label masks must be nonnegative 2D integer images.')
        if source is None and image.shape != self.shape:
            raise ValueError('Cell mask dimensions must match the reference grid.')
        return resample_crop(image, self.shape, bounds, nearest=True)

    def close(self):
        self.cache.clear()
