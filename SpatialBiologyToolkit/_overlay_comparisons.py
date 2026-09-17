"""ROI-matched comparison images for population overlays.

Files are indexed once per assessment; only one source image is decoded at a
time. Resizing assumes corresponding full image extents, not image registration.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
import re

import numpy as np


def _roi_tokens(name):
    return tuple(str(int(token)) if token.isdigit() else token
                 for token in re.findall(r'[a-z]+|\d+', str(name).casefold()))


def _image_stem(path):
    stem = path.stem
    return stem[:-4] if stem.casefold().endswith('.ome') else stem


@dataclass
class _ComparisonSource:
    folder: Path
    title: str
    legend: dict
    interpolation: str
    show_cell_outlines: bool
    files: tuple

    def match(self, roi):
        """Exact names win; token matching preserves numeric ROI boundaries."""
        roi = str(roi)
        exact = [p for p, stem, _ in self.files if stem.casefold() == roi.casefold()]
        tokens = _roi_tokens(roi)
        normalized = [p for p, _, parts in self.files if tokens and parts == tokens]
        decorated = [p for p, _, parts in self.files if tokens and any(
            parts[i:i + len(tokens)] == tokens for i in range(len(parts) - len(tokens) + 1))]
        matches = exact or normalized or decorated
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, 'No matching image'
        return None, 'Ambiguous match: ' + ', '.join(str(p.relative_to(self.folder)) for p in matches)


def prepare_comparison_images(comparison_images):
    """Validate folder shortcuts/settings and build reusable file indexes."""
    from .config.models import BackgatingComparisonImageConfig
    from matplotlib.colors import to_rgb

    if comparison_images is None:
        return []
    if isinstance(comparison_images, (str, Path, dict)):
        raise ValueError('comparison_images must be a list of folders or panel settings.')
    prepared = []
    extensions = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}
    for item in comparison_images:
        if isinstance(item, _ComparisonSource):
            prepared.append(item)
            continue
        if isinstance(item, (str, Path)):
            item = {'folder': str(item)}
        elif isinstance(item, dict):
            item = dict(item)
            if isinstance(item.get('folder'), Path):
                item['folder'] = str(item['folder'])
        config = BackgatingComparisonImageConfig.model_validate(item)
        folder = Path(config.folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f'Comparison image folder not found: {folder}')
        legend = {}
        for label, color in config.legend.items():
            if isinstance(color, str):
                rgb = to_rgb(color)
            else:
                rgb = np.asarray(color, dtype=float) / 255.0
                if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
                    raise ValueError(f'Legend color for {label!r} must be an RGB triple in 0..255.')
            legend[str(label)] = tuple(rgb)
        files = tuple((p, _image_stem(p), _roi_tokens(_image_stem(p)))
                      for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix.lower() in extensions)
        prepared.append(_ComparisonSource(folder, folder.name if config.title is None else config.title,
                                          legend, config.interpolation, config.show_cell_outlines, files))
    return prepared


def _load_comparison_crop(path, shape, bounds, interpolation):
    """Resize the whole extent onto the IMC grid, then retain only visible pixels."""
    from skimage import io
    from skimage.transform import resize
    from PIL import Image

    image = io.imread(str(path))
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[-1] in (3, 4)):
        raise ValueError(f'Expected a 2D grayscale or RGB/RGBA image, got shape {image.shape}. '
                         'Export a single plane for multi-channel or multi-page TIFFs.')
    original_shape = image.shape[:2]
    if image.ndim == 3 and image.dtype != np.uint8:
        # Matplotlib interprets integer RGB as 0..255; high-bit-depth colour TIFFs
        # need dtype-aware conversion, not clipping everything above 255.
        from skimage.util import img_as_ubyte
        image = img_as_ubyte(image)
    h, w = shape
    if original_shape != (h, w):
        if image.dtype == np.uint8:
            method = Image.Resampling.NEAREST if interpolation == 'nearest' else Image.Resampling.BILINEAR
            image = np.asarray(Image.fromarray(image).resize((w, h), resample=method))
        else:
            target = (h, w) if image.ndim == 2 else (h, w, image.shape[-1])
            image = resize(image, target, order=0 if interpolation == 'nearest' else 1,
                           preserve_range=True, anti_aliasing=False).astype(image.dtype)
    x0, y0, x1, y1 = bounds
    # Preserve full-image grayscale contrast even though only the crop is retained.
    intensity_limits = (float(np.nanmin(image)), float(np.nanmax(image))) if image.ndim == 2 else None
    return image[y0:y1, x0:x1].copy(), original_shape, intensity_limits


def add_comparison_panels(fig, primary_ax, sources, roi, image_shape, *, primary_title,
                          font_family, fontsize, contour_artists, title_fontsize=None):
    """Append equally sized panels, sharing exact primary-axis view coordinates."""
    from matplotlib.patches import Patch, PathPatch

    title_fontsize = fontsize if title_fontsize is None else title_fontsize
    primary_position = primary_ax.get_position().frozen()
    positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
    old_width, old_height = fig.get_size_inches()
    panel_step = primary_position.width + 0.03
    width_factor = 1 + len(sources) * panel_step
    fig.set_size_inches(old_width * width_factor, old_height)
    for ax, pos in positions:
        ax.set_position([pos.x0 / width_factor, pos.y0, pos.width / width_factor, pos.height])
    if primary_title:
        primary_ax.set_title(primary_title, fontsize=title_fontsize, fontfamily=font_family,
                             gid='primary_panel_title')
    xlim, ylim = primary_ax.get_xlim(), primary_ax.get_ylim()
    h, w = image_shape
    x0, x1 = max(0, int(np.floor(min(xlim) + 0.5))), min(w, int(np.ceil(max(xlim) + 0.5)))
    y0, y1 = max(0, int(np.floor(min(ylim) + 0.5))), min(h, int(np.ceil(max(ylim) + 0.5)))
    metadata = []
    for i, source in enumerate(sources, 1):
        gid = f'comparison_{i}'
        ax = fig.add_axes([(primary_position.x0 + i * panel_step) / width_factor, primary_position.y0,
                           primary_position.width / width_factor, primary_position.height])
        ax.set_gid(gid + '_panel')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        if source.title:
            ax.set_title(source.title, fontsize=title_fontsize, fontfamily=font_family, gid=gid + '_title')
        path, error = source.match(roi)
        record = dict(title=source.title, folder=str(source.folder),
                      path=str(path) if path else None, target_shape=[h, w],
                      xlim=list(xlim), ylim=list(ylim), interpolation=source.interpolation,
                      crop_pixel_bounds=[x0, y0, x1, y1], show_cell_outlines=source.show_cell_outlines)
        if path is not None:
            try:
                crop, original_shape, limits = _load_comparison_crop(
                    path, (h, w), (x0, y0, x1, y1), source.interpolation)
                kwargs = dict(cmap='gray', vmin=limits[0], vmax=limits[1]) if limits is not None else {}
                ax.imshow(crop, extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5),
                          origin='upper', interpolation='none', gid=gid + '_image', **kwargs)
                del crop
                record['original_shape'] = list(original_shape)
                if source.show_cell_outlines:
                    for artist in contour_artists:
                        ax.add_patch(PathPatch(artist.get_path(), facecolor='none',
                                               edgecolor=artist.get_edgecolor(), linewidth=artist.get_linewidth(),
                                               zorder=2, gid=gid + '_' + artist.get_gid()))
                    for collection in primary_ax.collections:
                        if collection.get_gid() == 'cell_centers':
                            points = collection.get_offsets()
                            ax.scatter(points[:, 0], points[:, 1], c=collection.get_facecolors(),
                                       s=collection.get_sizes(), alpha=collection.get_alpha(),
                                       gid=gid + '_cell_centers')
            except MemoryError:
                raise
            except (OSError, ValueError) as exc:
                error = str(exc)
        if error:
            logging.warning('Comparison panel %r, ROI %s: %s', source.title, roi, error)
            ax.text(0.5, 0.5, 'Ambiguous image match' if error.startswith('Ambiguous') else 'Image unavailable',
                    transform=ax.transAxes, ha='center', va='center', color='white',
                    fontsize=fontsize, fontfamily=font_family, gid=gid + '_missing')
        elif source.legend:
            # Legend's renderer emits its gid; OffsetBox does not, which would
            # leave its text ungrouped in the layered SVG.
            handles = [Patch(visible=False, label=label) for label in source.legend]
            box = ax.legend(handles=handles, loc='upper right', handlelength=0, handletextpad=0,
                            labelcolor=list(source.legend.values()),
                            prop={'family': font_family, 'size': fontsize},
                            facecolor='black', edgecolor='white', framealpha=1, fancybox=False)
            box.set_gid(gid + '_legend')
        record['status'] = error or 'matched'
        metadata.append(record)
    return metadata
