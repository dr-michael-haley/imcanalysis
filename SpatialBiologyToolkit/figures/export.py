"""Final-artifact export, reproducible manifests, and lightweight batch review."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote


class RenderedFigure:
    """Owns one Matplotlib figure. Use a context manager or close() after use."""
    def __init__(self, figure, metadata, groups, prefixes, font_family):
        self.figure, self.metadata = figure, metadata
        self.groups, self.prefixes, self.font_family = groups, prefixes, font_family
        self.closed = False

    def save(self, path, *, dpi=None, metadata=True):
        if self.closed:
            raise ValueError('This rendered figure has been closed.')
        path = Path(path)
        if path.suffix.lower() not in ('.png', '.svg', '.pdf', '.tif', '.tiff'):
            raise ValueError('Export supports PNG, SVG, PDF or TIFF.')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never tight-crop: publication dimensions and grid spacing are explicit.
        kwargs = dict(dpi=dpi or self.figure.dpi, facecolor=self.figure.get_facecolor())
        if path.suffix.lower() == '.svg':
            from .svg import _save_population_overlay_svg
            _save_population_overlay_svg(self.figure, path, layers=self.groups,
                                         group_prefixes=self.prefixes, font_family=self.font_family, **kwargs)
        else:
            import matplotlib as mpl
            with mpl.rc_context({'pdf.fonttype': 42}):
                self.figure.savefig(path, **kwargs)
        if metadata:
            path.with_suffix('.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        return path

    def _repr_png_(self):
        from io import BytesIO
        if self.closed:
            return None
        with BytesIO() as buffer:
            self.figure.savefig(buffer, format='png')
            return buffer.getvalue()

    def freeze(self):
        """Reproduce this ROI with resolved bounds, colours and a fixed crop."""
        from .specs import Figure, Crop, Scale, IMC, Values, Populations
        recipe = Figure.model_validate(self.metadata['recipe'])
        for panel in recipe.panels:
            for layer in panel.layers:
                if isinstance(layer, IMC) and not panel.legend_only:
                    for channel in layer.channels:
                        bounds = self.metadata['scaling'][f'{layer.id}/{channel.marker}']['limits']
                        channel.scale = Scale(mode='fixed', limits=bounds)
                elif isinstance(layer, Values):
                    low, high = self.metadata['scaling'][layer.id]['limits']
                    layer.scale = Scale(mode='fixed', limits=(low, high))
                elif isinstance(layer, Populations):
                    layer.colors = self.metadata['palettes'][layer.id]
        return recipe.with_crop(Crop(mode='bounds', roi_bounds={
            self.metadata['roi']: tuple(self.metadata['view']['bounds'])}))

    def save_bundle(self, output_folder, *, formats=('png', 'svg'), dpi=300):
        """Save final images, metadata, source binding and recipes together."""
        from .specs import Figure
        formats = tuple(str(f).lower().lstrip('.') for f in formats)
        if not formats or set(formats) - {'png', 'svg', 'pdf', 'tif', 'tiff'}:
            raise ValueError('Export supports PNG, SVG, PDF or TIFF.')
        if self.closed:
            raise ValueError('This rendered figure has been closed.')
        output = Path(output_folder)
        output.mkdir(parents=True, exist_ok=True)
        basename = _output_name(self.metadata['roi'])
        files = [self.save(output / f'{basename}.{format}', dpi=dpi, metadata=False)
                 for format in dict.fromkeys(formats)]
        (output / f'{basename}.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        Figure.model_validate(self.metadata['recipe']).save(output / 'recipe.json')
        self.freeze().save(output / f'{basename}_frozen.json')
        (output / 'dataset_binding.json').write_text(json.dumps(self.metadata.get('dataset', {}), indent=2), encoding='utf-8')
        return files

    def close(self):
        if self.closed:
            return
        self.figure.clear()
        if hasattr(self.figure.canvas, 'renderer'):
            self.figure.canvas.renderer = None
        self.figure = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _output_name(roi):
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', str(roi)).strip('_')[:80] or 'ROI'
    # Avoid path traversal, Windows reserved names, and normalisation collisions.
    digest = hashlib.sha256(str(roi).encode()).hexdigest()[:10]
    return f'roi_{slug}_{digest}'


def export_rois(recipe, dataset, output_folder, *, rois=None, formats=('png', 'svg'),
                on_error='raise', progress=None, cancelled=None):
    formats = tuple(str(f).lower().lstrip('.') for f in formats)
    if not formats or set(formats) - {'png', 'svg', 'pdf', 'tif', 'tiff'}:
        raise ValueError('Specify one or more of png, svg, pdf, tif, tiff.')
    if on_error not in ('raise', 'skip'):
        raise ValueError('on_error must be raise or skip.')
    prepared = recipe.prepare(dataset, rois=rois, progress=progress, cancelled=cancelled)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    recipe.save(output / 'recipe.json')
    (output / 'dataset_binding.json').write_text(json.dumps(dataset.binding(), indent=2), encoding='utf-8')
    manifest = dict(schema_version=1, rois=prepared.rois, status='running', results=[])
    manifest_path = output / 'manifest.json'

    def record():
        temp = output / 'manifest.json.tmp'
        temp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temp.replace(manifest_path)

    record()
    try:
        for i, roi in enumerate(prepared.rois):
            if cancelled and cancelled():
                raise InterruptedError('Figure batch cancelled.')
            entry = dict(roi=roi, files=[])
            manifest['results'].append(entry)
            try:
                with prepared.render(roi) as result:
                    basename = _output_name(roi)
                    for format in dict.fromkeys(formats):
                        file = result.save(output / f'{basename}.{format}', metadata=False)
                        entry['files'].append(file.name)
                    metadata_path = output / f'{basename}.json'
                    metadata_path.write_text(json.dumps(result.metadata, indent=2), encoding='utf-8')
                    result.freeze().save(output / f'{basename}_frozen.json')
                    entry.update(status='ok', metadata=metadata_path.name, view=result.metadata['view'])
                del result
            except (MemoryError, InterruptedError):
                entry.update(status='aborted')
                raise
            except Exception as error:
                entry.update(status='error', error=str(error))
                if on_error == 'raise':
                    raise
            finally:
                record()
            if progress:
                progress(dict(stage='export', roi=roi, completed=i + 1, total=len(prepared.rois)))
        manifest['status'] = 'completed_with_errors' if any(r['status'] != 'ok' for r in manifest['results']) else 'completed'
    except BaseException as error:
        manifest['status'] = 'cancelled' if isinstance(error, InterruptedError) else 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        record()
        _review_index(output, manifest)
    return manifest


def _review_index(output, manifest):
    entries = []
    for item in manifest['results']:
        preview = next((f for f in item['files'] if f.endswith('.png')), None)
        preview = preview or next((f for f in item['files'] if f.endswith('.svg')), None)
        links = ' '.join(f'<a href="{quote(f)}">{html.escape(Path(f).suffix[1:])}</a>' for f in item['files'])
        image = f'<img loading="lazy" src="{quote(preview)}" alt="{html.escape(item["roi"], quote=True)}">' if preview else ''
        description = item.get('error') or json.dumps(item.get('view', {}))
        entries.append(f'<article><h2>{html.escape(item["roi"])}</h2>{image}<p>{links}</p><pre>{html.escape(description)}</pre></article>')
    page = ('<!doctype html><meta charset="utf-8"><title>ROI figure review</title>'
            '<style>body{font:16px Arial;margin:24px;background:#eee}article{background:white;padding:16px;margin:16px 0}'
            'img{max-width:100%;max-height:700px}pre{white-space:pre-wrap}a{margin-right:12px}</style>'
            '<h1>ROI figure review</h1><p>Hotspot crops show concentrated events, not necessarily representative tissue.</p>'
            + ''.join(entries))
    (output / 'index.html').write_text(page, encoding='utf-8')
