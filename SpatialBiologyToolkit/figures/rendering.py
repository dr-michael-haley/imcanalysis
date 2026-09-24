"""Render a resolved recipe without pyplot or intermediate image files."""
from __future__ import annotations

import numpy as np

from .specs import Figure, IMC, Image, Populations, Values, LabelMask
from .sources import ROIContext, read_image, resample_crop
from .values import cell_ids, get_values, numeric_values, categorical_palette
from .normalization import calibrate, resolve_bounds, scale_pixels
from .selection import select_view


class PreparedFigure:
    """Frozen recipe snapshot plus scalar calibration; no retained image arrays."""
    def __init__(self, recipe, dataset, *, rois=None, progress=None, cancelled=None):
        from matplotlib.colors import to_rgba
        import matplotlib as mpl
        from .svg import _svg_font_family
        self.recipe = recipe.resolved()
        self.dataset = dataset
        self.rois = [str(r) for r in (dataset.rois if rois is None else rois)]
        if not self.rois or len(set(self.rois)) != len(self.rois):
            raise ValueError('Supply a nonempty list of unique ROIs.')
        unknown = set(self.rois) - set(dataset.rois)
        if unknown:
            raise ValueError(f'Unknown ROIs: {sorted(unknown)}')
        self.bounds, self.palettes = {}, {}
        self.progress, self.cancelled = progress, cancelled
        _svg_font_family(self.recipe.style.font_family)
        to_rgba(self.recipe.style.background)
        to_rgba(self.recipe.style.panel_background)
        # Validate references before any potentially expensive calibration.
        for panel in self.recipe.panels:
            if panel.scale_bar:
                to_rgba(panel.scale_bar.color)
                if panel.scale_bar.unit == 'um' and dataset.pixel_size_um is None:
                    raise ValueError('A micrometre scale bar needs Dataset(pixel_size_um=...).')
            for layer in panel.layers:
                if isinstance(layer, Image) and not panel.legend_only and layer.source not in dataset.images:
                    raise ValueError(f'Unknown image source {layer.source!r}.')
                if isinstance(layer, LabelMask) and not panel.legend_only and layer.source not in dataset.labels:
                    raise ValueError(f'Unknown label-mask source {layer.source!r}.')
                if isinstance(layer, (Populations, Values)) and (dataset.adata is None or (not panel.legend_only and dataset.masks is None)):
                    raise ValueError('Cell layers require AnnData and a cell-mask folder.')
                if isinstance(layer, IMC) and not panel.legend_only and not dataset.imc:
                    raise ValueError('IMC layers require an IMC image folder.')
                if isinstance(layer, Populations):
                    palette = categorical_palette(dataset, layer.obs, layer.colors)
                    if layer.groups and set(layer.groups) - set(palette):
                        raise ValueError(f'Unknown populations in {layer.obs!r}: {set(layer.groups) - set(palette)}')
                    self.palettes[layer.id] = palette
                if isinstance(layer, Values):
                    mpl.colormaps[layer.cmap]
                    to_rgba(layer.missing_color)
                    # Metadata-only checks; do not materialize expression to validate.
                    ref = layer.value
                    if dataset.adata is not None:
                        collection = dataset.obs.columns if ref.kind == 'obs' else dataset.adata.var_names
                        if ref.key not in collection:
                            raise ValueError(f'Unknown {ref.kind} value {ref.key!r}.')
                        if ref.layer is not None and ref.layer not in dataset.adata.layers:
                            raise ValueError(f'Unknown expression layer {ref.layer!r}.')
                for color in getattr(layer, 'colors', {}).values():
                    to_rgba(color)
                if getattr(layer, 'edgecolor', None):
                    to_rgba(layer.edgecolor)
                if isinstance(layer, IMC):
                    for channel in layer.channels:
                        to_rgba(channel.color)
                        if not panel.legend_only and channel.scale.mode == 'pooled_quantile':
                            raise ValueError('Pooled quantiles are supported for cell values, not whole image cohorts.')
        calibrated = {}
        for panel in self.recipe.panels:
            if panel.legend_only:
                continue
            for layer in panel.layers:
                items = [(str(i), channel.scale, ('image', channel.marker),
                          lambda roi, marker=channel.marker: read_image(dataset.channel_path(roi, marker)))
                         for i, channel in enumerate(layer.channels)] if isinstance(layer, IMC) else []
                if isinstance(layer, Values):
                    items = [('values', layer.scale, ('value', layer.value.model_dump_json()),
                              lambda roi, ref=layer.value: numeric_values(dataset, roi, ref))]
                for name, scale, source, loader in items:
                    if scale.rois and set(scale.rois) - set(dataset.rois):
                        raise ValueError('Normalisation references unknown ROIs.')
                    key = (source, scale.model_dump_json())
                    if key not in calibrated:
                        # Cohort is the dataset unless scale.rois explicitly narrows it.
                        calibrated[key] = calibrate(scale, dataset.rois, loader,
                                                    progress=progress, cancelled=cancelled)
                    self.bounds[(layer.id, name)] = calibrated[key]

    def render(self, roi, *, dpi=None):
        from matplotlib.figure import Figure as MplFigure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if self.cancelled and self.cancelled():
            raise InterruptedError('Figure rendering cancelled.')
        if str(roi) not in self.rois:
            raise ValueError(f'ROI {roi!r} is outside this prepared selection.')
        context = ROIContext(self.dataset, roi)
        context.cancelled = self.cancelled
        fig = None
        try:
            view = select_view(context, self.recipe.crop)
            style = self.recipe.style
            rows, cols = self.recipe.layout
            title_height = max(style.title_height_mm, style.title_fontsize * 25.4 / 72 * 1.4)
            annotated_rows = {panel.row + panel.rowspan - 1 for panel in self.recipe.panels
                              if any(isinstance(layer, Values) and layer.colorbar for layer in panel.layers)}
            annotation_height = max(style.annotation_height_mm, style.legend_fontsize * 25.4 / 72 * 2 + 6)
            row_heights = [title_height + style.panel_height_mm + (annotation_height if row in annotated_rows else 0)
                           for row in range(rows)]
            total_width = cols * style.panel_width_mm + (cols - 1) * style.column_spacing_mm
            total_height = sum(row_heights) + (rows - 1) * style.row_spacing_mm
            fig = MplFigure(figsize=(total_width / 25.4, total_height / 25.4),
                            dpi=dpi or style.dpi, facecolor=style.background)
            FigureCanvasAgg(fig)
            state = _RenderState(self, context, fig, view)
            for panel in self.recipe.panels:
                if self.cancelled and self.cancelled():
                    raise InterruptedError('Figure rendering cancelled.')
                x_mm = panel.col * (style.panel_width_mm + style.column_spacing_mm)
                y_top = sum(row_heights[:panel.row]) + panel.row * style.row_spacing_mm
                pw = panel.colspan * style.panel_width_mm + (panel.colspan - 1) * style.column_spacing_mm
                end_row = panel.row + panel.rowspan - 1
                ph = sum(row_heights[panel.row:end_row + 1]) + (panel.rowspan - 1) * style.row_spacing_mm
                top = total_height - y_top - title_height
                bottom = total_height - y_top - ph + (annotation_height if end_row in annotated_rows else 0)
                ax = fig.add_axes([x_mm / total_width, bottom / total_height, pw / total_width, (top - bottom) / total_height])
                ax.set_gid(panel.id)
                state.groups[panel.id] = panel.title or panel.id
                ax.set_facecolor(style.panel_background)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                x, y, w, h = view['bounds']
                ax.set_xlim(x - .5, x + w - .5)
                ax.set_ylim(y + h - .5, y - .5)
                ax.set_aspect('equal')
                legends, colorbars = [], []
                for layer_index, layer in enumerate(panel.layers):
                    state.zorder = 1 + layer_index * 3
                    entries, bar = (state.legend_entries(layer), None) if panel.legend_only else state.draw_layer(ax, layer)
                    if layer.legend:
                        legends.extend(entries)
                    if bar is not None:
                        colorbars.append((layer, bar))
                if panel.legend_only:
                    if not legends:
                        raise ValueError(f'Legend-only panel {panel.title or panel.id!r} has no categorical legend entries.')
                    ax.set_facecolor(style.background)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_aspect('auto')
                state.decorate(ax, panel, legends, colorbars, x_mm, bottom, pw,
                               top, total_width, total_height, title_height)
            metadata = dict(roi=str(roi), reference_shape=list(context.shape),
                            pixel_size_um=self.dataset.pixel_size_um, view=view,
                            sources=context.sources, scaling=state.scaling,
                            palettes=self.palettes, dimensions_mm=[total_width, total_height],
                            normalization_scope='dataset unless overridden by Scale.rois',
                            dataset=self.dataset.binding(),
                            recipe=self.recipe.model_dump(mode='json'))
            from .export import RenderedFigure
            return RenderedFigure(fig, metadata, state.groups, state.prefixes, style.font_family)
        except BaseException:
            if fig is not None:
                fig.clear()
            raise
        finally:
            context.close()


class _RenderState:
    def __init__(self, prepared, context, figure, view):
        self.prepared, self.context, self.fig, self.view = prepared, context, figure, view
        self.groups, self.prefixes, self.scaling, self.paths = {}, {}, {}, {}
        self.zorder = 1

    def legend_entries(self, layer):
        """Categorical keys use metadata only; no image/mask reads or geometry."""
        if isinstance(layer, IMC):
            return [(channel.marker, channel.color) for channel in layer.channels]
        if isinstance(layer, Image):
            return list(layer.colors.items())
        if isinstance(layer, LabelMask):
            palette = self.label_palette(layer)
            return [(name, palette[ident]) for ident, name in layer.labels.items()]
        if isinstance(layer, Populations):
            palette = self.prepared.palettes[layer.id]
            groups = list(palette) if layer.groups is None else layer.groups
            return [(group, palette[group]) for group in groups]
        return []

    @staticmethod
    def label_palette(layer):
        import matplotlib as mpl
        from matplotlib.colors import to_hex
        return {ident: layer.colors.get(ident, to_hex(mpl.colormaps['tab20'](i % 20)))
                for i, ident in enumerate(layer.labels)}

    def image(self, ax, array, gid, **kwargs):
        x, y, w, h = self.view['bounds']
        self.groups[gid] = gid
        return ax.imshow(array, extent=(x - .5, x + w - .5, y + h - .5, y - .5),
                         origin='upper', interpolation='none', gid=gid, zorder=self.zorder, **kwargs)

    def geometry(self, source=None, *, include_zero=False):
        from .geometry import _population_cell_paths
        key = (source, include_zero)
        if key not in self.paths:
            x, y, w, h = self.view['bounds']
            x0, y0 = max(0, x - 1), max(0, y - 1)
            x1, y1 = min(self.context.shape[1], x + w + 1), min(self.context.shape[0], y + h + 1)
            mask = self.context.mask((x0, y0, x1 - x0, y1 - y0), source)
            paths = {}
            for ident, path in _population_cell_paths(mask, np.unique(mask)):
                path.vertices += [x0, y0]
                paths[ident] = path
            if include_zero:
                for _, path in _population_cell_paths((mask == 0).astype(np.uint8), {1}):
                    path.vertices += [x0, y0]
                    paths[0] = path
            self.paths[key] = paths
        return self.paths[key]

    def draw_cells(self, ax, layer, colors, source=None):
        from matplotlib.patches import PathPatch
        from matplotlib.colors import to_rgba
        if layer.rendering == 'raster':
            from skimage.segmentation import find_boundaries
            mask = self.context.mask(self.view['bounds'], source)
            image = np.zeros((*mask.shape, 4), dtype=np.float32)
            ids = np.array(sorted(colors))
            if len(ids):
                positions = np.searchsorted(ids, mask)
                valid = (positions < len(ids)) & (ids[np.minimum(positions, len(ids) - 1)] == mask)
                palette = np.array([to_rgba(colors[k]) for k in ids])
                image[valid] = palette[positions[valid]]
                boundary = find_boundaries(mask, mode='inner') & valid
                if layer.mode == 'outline':
                    image[~boundary] = 0
                if layer.edgecolor and layer.mode != 'fill':
                    image[boundary] = to_rgba(layer.edgecolor)
            self.image(ax, image, layer.id + '_raster', alpha=layer.opacity)
            return
        paths = self.geometry(source, include_zero=0 in colors)
        for mode in ('fill', 'outline'):
            if layer.mode not in (mode, 'both'):
                continue
            group = layer.id + '_' + mode
            self.groups[group] = f'{layer.kind}: {mode}'
            self.prefixes[group + '_object_'] = group
            for ident, color in colors.items():
                if ident not in paths:
                    continue
                ax.add_patch(PathPatch(paths[ident],
                    facecolor=color if mode == 'fill' else 'none',
                    edgecolor=(layer.edgecolor or color) if mode == 'outline' else 'none',
                    linewidth=layer.linewidth, alpha=layer.opacity, zorder=self.zorder + (0.1 if mode == 'outline' else 0),
                    gid=f'{group}_object_{ident}'))

    def draw_layer(self, ax, layer):
        from matplotlib.colors import to_rgb, to_hex, Normalize
        from matplotlib.cm import ScalarMappable
        import matplotlib as mpl
        data, roi = self.context.dataset, self.context.roi
        if isinstance(layer, IMC):
            _, _, w, h = self.view['bounds']
            composite = np.zeros((h, w, 3), dtype=np.float32)
            for i, channel in enumerate(layer.channels):
                raw = self.context.read(data.channel_path(roi, channel.marker))
                if raw.ndim != 2 or raw.shape != self.context.shape:
                    raise ValueError('IMC channels must be 2D and match the reference grid.')
                bounds = resolve_bounds(channel.scale, self.prepared.bounds[(layer.id, str(i))], raw)
                crop = resample_crop(raw, self.context.shape, self.view['bounds'])
                scaled = scale_pixels(crop, bounds, channel.gamma)
                for component, weight in enumerate(to_rgb(channel.color)):
                    composite[..., component] += scaled * weight
                self.scaling[f'{layer.id}/{channel.marker}'] = dict(limits=list(bounds),
                    scope=channel.scale.rois or data.rois if channel.scale.mode == 'cohort_quantile' else [roi],
                    mode=channel.scale.mode)
                del crop, scaled, raw
            np.clip(composite, 0, 1, out=composite)
            self.image(ax, composite, layer.id + '_image', alpha=layer.opacity)
            return self.legend_entries(layer), None
        if isinstance(layer, Image):
            raw = self.context.read(data.images[layer.source].match(roi))
            if raw.ndim != 2 and not (raw.ndim == 3 and raw.shape[-1] in (3, 4)):
                raise ValueError('External images must be 2D grayscale or RGB/RGBA; export/select a plane first.')
            crop = resample_crop(raw, self.context.shape, self.view['bounds'], nearest=layer.interpolation == 'nearest')
            if crop.ndim == 3 and crop.dtype != np.uint8:
                from skimage.util import img_as_float32
                crop = img_as_float32(crop)
                if not np.isfinite(crop).all() or crop.min() < 0 or crop.max() > 1:
                    raise ValueError('Floating RGB images must use intensities in [0, 1].')
            kwargs = dict(cmap='gray', vmin=float(np.nanmin(raw)), vmax=float(np.nanmax(raw))) if crop.ndim == 2 else {}
            self.image(ax, crop, layer.id + '_image', alpha=layer.opacity, **kwargs)
            return self.legend_entries(layer), None
        if isinstance(layer, LabelMask):
            palette = self.label_palette(layer)
            self.draw_cells(ax, layer, palette, layer.source)
            return self.legend_entries(layer), None
        ids = cell_ids(data, roi)
        if isinstance(layer, Populations):
            palette = self.prepared.palettes[layer.id]
            groups = list(palette) if layer.groups is None else layer.groups
            values = data.observations(roi)[layer.obs]
            colors = {int(ident): palette[str(value)] for ident, value in zip(ids, values)
                      if str(value) in groups}
            self.draw_cells(ax, layer, colors)
            return self.legend_entries(layer), None
        values = numeric_values(data, roi, layer.value)
        bounds = resolve_bounds(layer.scale, self.prepared.bounds[(layer.id, 'values')], values)
        # Equal-valued data has a defined display colour and a nondegenerate bar.
        lo, hi = bounds
        norm = Normalize(lo, hi if hi > lo else lo + 1, clip=True)
        mapper = ScalarMappable(norm=norm, cmap=layer.cmap)
        colors = {int(ident): mapper.to_rgba(value) if np.isfinite(value) else layer.missing_color
                  for ident, value in zip(ids, values)}
        self.draw_cells(ax, layer, colors)
        self.scaling[layer.id] = dict(limits=list(bounds), mode=layer.scale.mode,
            scope=(layer.scale.rois or data.rois) if layer.scale.mode in ('cohort_quantile', 'pooled_quantile') else [roi])
        return [], mapper if layer.colorbar else None

    def decorate(self, ax, panel, legends, colorbars, x_mm, bottom, pw, top, total_width, total_height, title_height):
        from matplotlib.patches import Patch
        import matplotlib.patheffects as effects
        style = self.prepared.recipe.style
        font = style.font_family
        from matplotlib.colors import to_rgb
        light_background = np.dot(to_rgb(style.background), [0.299, 0.587, 0.114]) > .5
        legend_foreground = 'black' if light_background else 'white'
        if panel.title:
            gid = panel.id + '_title'
            self.groups[gid] = 'Panel title'
            self.fig.text((x_mm + pw / 2) / total_width,
                          (top + title_height / 2) / total_height, panel.title,
                          ha='center', va='center', fontsize=style.title_fontsize, fontfamily=font, gid=gid)
        if panel.letter:
            gid = panel.id + '_letter'
            self.groups[gid] = 'Panel letter'
            ax.text(.02, .98, panel.letter, transform=ax.transAxes, va='top',
                    color=legend_foreground if panel.legend_only else 'white',
                    fontweight='bold', fontsize=style.letter_fontsize or style.title_fontsize,
                    fontfamily=font, gid=gid, zorder=10001)
        if panel.legend and legends:
            # Keep different colours with the same label visible; remove exact duplicates.
            legends = list(dict.fromkeys((label, str(color)) for label, color in legends))
            handles = [Patch(facecolor=color, edgecolor='none', label=label) for label, color in legends]
            legend = ax.legend(handles=handles, loc='center' if panel.legend_only else 'upper right',
                               ncol=panel.legend_ncols,
                               prop={'family': font, 'size': panel.legend_fontsize or style.legend_fontsize},
                               facecolor=style.background if panel.legend_only else 'black',
                               edgecolor='white', framealpha=1, frameon=not panel.legend_only,
                               labelcolor=legend_foreground if panel.legend_only else 'white', fancybox=False)
            legend.set_gid(panel.id + '_legend')
            legend.set_zorder(10000)
            self.groups[panel.id + '_legend'] = 'Colour legend'
        for i, (layer, mapper) in enumerate(colorbars):
            fraction = 1 / len(colorbars)
            cax = self.fig.add_axes([(x_mm + pw * (i * fraction + .1 * fraction)) / total_width,
                                    (bottom - 3) / total_height, pw * .8 * fraction / total_width, 1.5 / total_height])
            cax.set_gid(layer.id + '_colorbar')
            self.groups[layer.id + '_colorbar'] = f'Colour bar: {layer.value.key}'
            bar = self.fig.colorbar(mapper, cax=cax, orientation='horizontal')
            bar.set_label(layer.value.key, fontsize=style.legend_fontsize, fontfamily=font)
            bar.ax.tick_params(labelsize=style.legend_fontsize, pad=1)
            for tick in bar.ax.get_xticklabels():
                tick.set_fontfamily(font)
        if panel.scale_bar:
            spec = panel.scale_bar
            x, y, w, h = self.view['bounds']
            unit = ('um' if self.context.dataset.pixel_size_um else 'px') if spec.unit == 'auto' else spec.unit
            pixel_size = self.context.dataset.pixel_size_um if unit == 'um' else 1
            physical_length = spec.length
            if physical_length is None:
                target = w * pixel_size / 5
                base = 10 ** np.floor(np.log10(target))
                physical_length = max(n * base for n in (1, 2, 5, 10) if n * base <= target)
            length = physical_length / pixel_size
            if length > .9 * w:
                raise ValueError('Scale bar is wider than the crop; reduce its length.')
            x0, y0 = x + .05 * w, y + .91 * h
            gid = panel.id + '_scale_bar'
            line, = ax.plot([x0, x0 + length], [y0, y0], color=spec.color,
                            linewidth=spec.linewidth, solid_capstyle='butt', gid=gid, zorder=10000)
            line.set_path_effects([effects.Stroke(linewidth=spec.linewidth + 1, foreground='black'), effects.Normal()])
            self.groups[gid] = 'Scale bar'
            gid += '_text'
            ax.text(x0 + length / 2, y0 - .025 * h, spec.label or f'{physical_length:g} {"µm" if unit == "um" else "px"}',
                    ha='center', va='bottom', color=spec.color,
                    fontsize=style.scale_bar_fontsize or style.legend_fontsize,
                    fontfamily=font, gid=gid, zorder=10000)
            self.groups[gid] = 'Scale bar text'
