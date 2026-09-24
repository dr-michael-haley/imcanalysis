"""Versioned, GUI-ready figure recipes. Imports only Pydantic and the stdlib."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Spec(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True, allow_inf_nan=False)


class Value(Spec):
    """Unambiguous reference to an obs column or a var in X/a named layer."""
    kind: Literal['obs', 'var'] = 'obs'
    key: str
    layer: str | None = None

    @model_validator(mode='after')
    def check_layer(self):
        if self.kind == 'obs' and self.layer is not None:
            raise ValueError('An obs reference cannot specify an expression layer.')
        return self


def obs(key: str) -> Value:
    return Value(key=key)


def var(key: str, *, layer: str | None = None) -> Value:
    return Value(kind='var', key=key, layer=layer)


class Condition(Spec):
    value: Value
    op: Literal['in', 'eq', 'gt', 'ge', 'lt', 'le'] = 'in'
    values: list[str | float | bool] = Field(min_length=1)

    @model_validator(mode='after')
    def check_values(self):
        if self.op != 'in' and len(self.values) != 1:
            raise ValueError('Comparison conditions need exactly one value.')
        return self


class Scale(Spec):
    """Display bounds. Cohort quantiles are reductions of per-ROI quantiles."""
    mode: Literal['fixed', 'roi_quantile', 'cohort_quantile', 'pooled_quantile'] = 'cohort_quantile'
    limits: tuple[float, float] | None = None
    quantiles: tuple[float, float] = (0.0, 0.99)
    reduction: Literal['mean', 'min', 'max'] = 'mean'
    rois: list[str] | None = None

    @model_validator(mode='after')
    def check_bounds(self):
        lo, hi = self.quantiles
        if not 0 <= lo < hi <= 1:
            raise ValueError('quantiles must satisfy 0 <= low < high <= 1.')
        if self.mode == 'fixed' and (self.limits is None or self.limits[0] > self.limits[1]):
            raise ValueError('Fixed scaling requires ordered limits (low <= high).')
        if self.mode != 'fixed' and self.limits is not None:
            raise ValueError('Use Scale(mode="fixed", limits=(low, high)) for explicit bounds.')
        return self


class Intensities(Spec):
    """Shared image-pixel bounds, stored in the recipe rather than a mutable CSV."""
    limits: dict[str, tuple[float, float]]
    source: str | None = None
    source_sha256: str | None = None

    @model_validator(mode='after')
    def check_limits(self):
        for marker, bounds in self.limits.items():
            if not marker or bounds[0] >= bounds[1]:
                raise ValueError(f'Invalid intensity bounds for {marker!r}.')
        return self

    def scale_max(self, factor: float) -> 'Intensities':
        """Return a new object with maxima multiplied by factor and minima unchanged.

        The original bounds and source CSV are untouched. Source metadata still
        identifies the original input file; the returned limits contain the
        adjusted values. Reject nonpositive/nonfinite factors and any resulting
        maximum that is not greater than its minimum.
        """
        import math
        if isinstance(factor, bool):
            raise ValueError('factor must be a finite positive number.')
        try:
            factor = float(factor)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('factor must be a finite positive number.') from error
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError('factor must be a finite positive number.')
        payload = self.model_dump()
        payload['limits'] = {marker: (low, high * factor)
                             for marker, (low, high) in self.limits.items()}
        return type(self).model_validate(payload)

    @classmethod
    def from_csv(cls, path, *, marker='marker', minimum='lower_threshold', maximum='vmax'):
        import csv
        import hashlib
        path = Path(path).resolve()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {marker, minimum, maximum} <= set(reader.fieldnames or []):
                raise ValueError(f'Intensity CSV needs {marker}, {minimum}, {maximum} columns.')
            limits = {}
            for row in reader:
                name = row[marker].strip()
                if name in limits:
                    raise ValueError(f'Duplicate marker {name!r} in intensity CSV.')
                limits[name] = (float(row[minimum]), float(row[maximum]))
        return cls(limits=limits, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class Channel(Spec):
    marker: str
    color: str = 'white'
    scale: Scale | None = None  # inherit Figure.intensities; otherwise legacy cohort quantiles
    gamma: float = Field(default=1, gt=0)

    def __init__(self, marker=None, **data):
        if marker is not None:
            data['marker'] = marker
        # Notebook convenience; persisted recipes always contain a Scale.
        limits = data.pop('limits', None)
        if limits is not None:
            if 'scale' in data:
                raise ValueError('Supply limits or scale, not both.')
            data['scale'] = Scale(mode='fixed', limits=limits)
        super().__init__(**data)


class Layer(Spec):
    id: str = Field(default_factory=lambda: 'layer_' + uuid4().hex, pattern=r'^[A-Za-z][A-Za-z0-9_-]*$')
    opacity: float = Field(default=1, ge=0, le=1)
    legend: bool = True


class IMC(Layer):
    kind: Literal['imc'] = 'imc'
    channels: list[Channel] = Field(min_length=1)

    @classmethod
    def rgb(cls, red, green, blue, **kwargs):
        return cls(channels=[Channel(marker, color=color) for marker, color in
                             zip((red, green, blue), ('red', 'lime', 'blue'))], **kwargs)


class Image(Layer):
    kind: Literal['image'] = 'image'
    source: str
    interpolation: Literal['bilinear', 'nearest'] = 'bilinear'
    colors: dict[str, str] = Field(default_factory=dict)


class CellStyle(Layer):
    mode: Literal['outline', 'fill', 'both'] = 'outline'
    linewidth: float = Field(default=0.5, ge=0)
    edgecolor: str | None = None
    rendering: Literal['vector', 'raster'] = 'vector'


class Populations(CellStyle):
    kind: Literal['populations'] = 'populations'
    obs: str
    groups: list[str] | None = None
    colors: dict[str, str] = Field(default_factory=dict)


class Values(CellStyle):
    kind: Literal['values'] = 'values'
    mode: Literal['outline', 'fill', 'both'] = 'fill'
    value: Value
    scale: Scale = Field(default_factory=Scale)
    cmap: str = 'viridis'
    colorbar: bool = True
    missing_color: str = '#808080'


class LabelMask(CellStyle):
    kind: Literal['labels'] = 'labels'
    mode: Literal['outline', 'fill', 'both'] = 'fill'
    source: str
    labels: dict[int, str]
    colors: dict[int, str] = Field(default_factory=dict)
    background: int = 0

    @model_validator(mode='after')
    def check_labels(self):
        if self.background < 0 or any(label < 0 for label in self.labels):
            raise ValueError('Label IDs must be nonnegative integers.')
        if self.background in self.labels:
            raise ValueError('The background label must not also be an annotated class.')
        return self


LayerSpec = Annotated[Union[IMC, Image, Populations, Values, LabelMask], Field(discriminator='kind')]


class ScaleBar(Spec):
    length: float | None = Field(default=None, gt=0)
    unit: Literal['px', 'um', 'auto'] = 'px'
    color: str = 'white'
    linewidth: float = Field(default=2, gt=0)
    label: str | None = None

    @classmethod
    def auto(cls, **kwargs):
        return cls(unit='auto', **kwargs)


class Panel(Spec):
    id: str = Field(default_factory=lambda: 'panel_' + uuid4().hex, pattern=r'^[A-Za-z][A-Za-z0-9_-]*$')
    row: int = Field(default=0, ge=0)
    col: int = Field(default=0, ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    title: str = ''
    letter: str | None = None
    layers: list[LayerSpec] = Field(default_factory=list)
    scale_bar: ScaleBar | None = None
    legend: bool = True
    legend_only: bool = False
    legend_ncols: int = Field(default=1, ge=1, strict=True)
    legend_fontsize: float | None = Field(default=None, gt=0, description='Legend text size in points; None inherits Style.legend_fontsize.')

    @model_validator(mode='after')
    def check_legend_only(self):
        if self.legend_only:
            if not self.legend or not any(layer.legend for layer in self.layers):
                raise ValueError('A legend-only panel needs an enabled legend and at least one legend layer.')
            if self.scale_bar is not None:
                raise ValueError('A legend-only panel cannot have a spatial scale bar.')
            if any(isinstance(layer, Values) for layer in self.layers):
                raise ValueError('legend_only supports categorical legends, not continuous Values colour bars.')
        return self


class Crop(Spec):
    mode: Literal['full', 'bounds', 'center', 'upper_left', 'upper_right', 'lower_left',
                  'lower_right', 'coordinate', 'cell', 'hotspot'] = 'full'
    size: tuple[int, int] | None = None  # width, height, reference pixels
    bounds: tuple[int, int, int, int] | None = None  # x, y, width, height
    roi_bounds: dict[str, tuple[int, int, int, int]] = Field(default_factory=dict)
    center: tuple[float, float] | None = None
    cell_id: int | None = None
    score: Value | None = None
    where: list[Condition] = Field(default_factory=list)
    denominator: list[Condition] = Field(default_factory=list)
    reducer: Literal['count', 'sum', 'mean', 'fraction'] = 'count'
    min_cells: int = Field(default=1, ge=1)
    stride: int = Field(default=1, ge=1)
    mask_source: str | None = None
    mask_labels: list[int] = Field(default_factory=list)
    min_coverage: float = Field(default=0, ge=0, le=1)
    fallback: Literal['center', 'error'] = 'error'

    @classmethod
    def hotspot(cls, **kwargs):
        return cls(mode='hotspot', **kwargs)

    @model_validator(mode='after')
    def check_selection(self):
        if self.size is not None and min(self.size) <= 0:
            raise ValueError('Crop size must be positive.')
        if self.mode not in ('full', 'bounds') and self.size is None:
            raise ValueError('This crop mode needs size=(width, height).')
        if self.mode == 'bounds' and self.bounds is None and not self.roi_bounds:
            raise ValueError('Bounds mode needs bounds or roi_bounds.')
        if self.mode == 'coordinate' and self.center is None:
            raise ValueError('Coordinate crop needs a center.')
        if self.mode == 'cell' and self.cell_id is None:
            raise ValueError('Cell crop needs a cell_id.')
        if self.mode == 'hotspot' and self.reducer in ('sum', 'mean') and self.score is None:
            raise ValueError('Sum/mean selection needs a numeric score.')
        if self.mode == 'hotspot' and self.reducer in ('count', 'fraction') and self.score is not None:
            raise ValueError('Use where conditions to threshold count/fraction selections, not score.')
        if self.mask_source and not self.mask_labels:
            raise ValueError('Mask selection needs mask_labels.')
        if self.min_coverage and not self.mask_source:
            raise ValueError('min_coverage needs a mask_source.')
        if self.mode == 'hotspot' and self.mask_source and self.score is None and not self.where:
            if self.reducer not in ('count', 'fraction'):
                raise ValueError('Mask-only selection supports count or fraction.')
        return self


class Style(Spec):
    font_family: str = 'Arial'
    title_fontsize: float = Field(default=12, gt=0)
    legend_fontsize: float = Field(default=8, gt=0)
    letter_fontsize: float | None = Field(default=None, gt=0, description='Panel-letter size in points; None inherits title_fontsize.')
    scale_bar_fontsize: float | None = Field(default=None, gt=0, description='Scale-bar text size in points; None inherits legend_fontsize.')
    panel_width_mm: float = Field(default=60, gt=0)
    panel_height_mm: float = Field(default=60, gt=0)
    row_spacing_mm: float = Field(default=5, ge=0)
    column_spacing_mm: float = Field(default=4, ge=0)
    title_height_mm: float = Field(default=7, ge=0)
    annotation_height_mm: float = Field(default=12, ge=0)
    background: str = 'white'
    panel_background: str = 'black'
    dpi: int = Field(default=300, ge=30)


class Figure(Spec):
    schema_version: Literal[1] = 1
    layout: tuple[int, int] = (1, 1)  # rows, columns
    panels: list[Panel] = Field(min_length=1)
    crop: Crop = Field(default_factory=Crop)
    style: Style = Field(default_factory=Style)
    intensities: Intensities | None = None

    @model_validator(mode='before')
    @classmethod
    def place_reused_layers(cls, data):
        # Reuse Python objects naturally. Malformed persisted duplicate IDs still fail.
        if isinstance(data, dict):
            seen, panels = set(), []
            for panel in data.get('panels', []):
                if isinstance(panel, Panel):
                    placed = panel.model_copy(deep=True)
                    for original, layer in zip(panel.layers, placed.layers):
                        if id(original) in seen:
                            layer.id = 'layer_' + uuid4().hex
                        seen.add(id(original))
                    panels.append(placed)
                else:
                    panels.append(panel)
            data = {**data, 'panels': panels}
        return data

    @classmethod
    def grid(cls, rows, *, letters=True, scale_bar=True, **kwargs):
        """Place a rectangular nested list of Panels; add letters and one scale bar."""
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError('Supply a nonempty rectangular panel grid.')
        panels = []
        for r, row in enumerate(rows):
            for c, panel in enumerate(row):
                placed = panel.model_copy(update={'row': r, 'col': c, 'id': 'panel_' + uuid4().hex})
                if letters and placed.letter is None:
                    number, letter = len(panels) + 1, ''
                    while number:
                        number, digit = divmod(number - 1, 26)
                        letter = chr(65 + digit) + letter
                    placed = placed.model_copy(update={'letter': letter})
                panels.append(placed)
        if scale_bar and not any(p.scale_bar for p in panels):
            first_image = next((i for i, p in enumerate(panels) if not p.legend_only), None)
            if first_image is not None:
                panels[first_image] = panels[first_image].model_copy(update={'scale_bar': ScaleBar.auto()})
        return cls(layout=(len(rows), len(rows[0])), panels=panels, **kwargs)

    def with_crop(self, crop):
        return self.model_validate({**self.model_dump(mode='json'), 'crop': crop.model_dump(mode='json')})

    def with_style(self, **changes):
        return self.model_validate({**self.model_dump(mode='json'),
                                   'style': {**self.style.model_dump(mode='json'), **changes}})

    def resolved(self):
        """Resolve shared channel settings without image reads or changing this recipe."""
        result = self.model_validate(self.model_dump(mode='json'))
        for panel in result.panels:
            if panel.legend_only:
                continue  # A categorical legend needs colours/names, not intensity calibration.
            for layer in panel.layers:
                if isinstance(layer, IMC):
                    for channel in layer.channels:
                        if channel.scale is None:
                            if result.intensities is not None:
                                if channel.marker not in result.intensities.limits:
                                    raise ValueError(f'No supplied intensity limits for {channel.marker!r}; provide bounds or an explicit Scale.')
                                channel.scale = Scale(mode='fixed', limits=result.intensities.limits[channel.marker])
                            else:
                                channel.scale = Scale()
        return result

    def preflight(self, dataset, *, rois=None):
        from .workflow import preflight
        return preflight(self, dataset, rois=rois)

    def preview(self, dataset, roi, *, dpi=120):
        """Return an in-memory preview; use as a context manager to release it."""
        return self.prepare(dataset, rois=[roi]).render(roi, dpi=dpi)

    def export(self, dataset, output_folder, *, roi, formats=('png', 'svg')):
        return self.export_rois(dataset, output_folder, rois=[roi], formats=formats)

    @model_validator(mode='after')
    def check_layout(self):
        if min(self.layout) <= 0:
            raise ValueError('Layout dimensions must be positive.')
        ids, occupied = set(), set()
        for panel in self.panels:
            for ident in [panel.id] + [layer.id for layer in panel.layers]:
                if ident in ids:
                    raise ValueError(f'Duplicate panel/layer ID: {ident}')
                ids.add(ident)
            for row in range(panel.row, panel.row + panel.rowspan):
                for col in range(panel.col, panel.col + panel.colspan):
                    if row >= self.layout[0] or col >= self.layout[1] or (row, col) in occupied:
                        raise ValueError('Panel spans overlap or lie outside the figure grid.')
                    occupied.add((row, col))
        return self

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            text = yaml.safe_dump(self.model_dump(mode='json'), sort_keys=False)
        else:
            text = self.model_dump_json(indent=2)
        path.write_text(text, encoding='utf-8')

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            return cls.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
        return cls.model_validate_json(path.read_text(encoding='utf-8'))

    def prepare(self, dataset, *, rois=None, progress=None, cancelled=None):
        from .rendering import PreparedFigure
        return PreparedFigure(self, dataset, rois=rois, progress=progress, cancelled=cancelled)

    def render(self, dataset, roi, *, dpi=None):
        # Explicitly scoped cohort calibration: use prepare() to reuse across renders.
        return self.prepare(dataset).render(roi, dpi=dpi)

    def export_rois(self, dataset, output_folder, *, rois=None, formats=('png', 'svg'),
                    on_error='raise', progress=None, cancelled=None):
        from .export import export_rois
        return export_rois(self, dataset, output_folder, rois=rois, formats=formats,
                           on_error=on_error, progress=progress, cancelled=cancelled)
