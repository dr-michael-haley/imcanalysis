"""SVG composition preserves objects, references and physical gallery spacing."""
import base64
from io import BytesIO
from pathlib import Path
import re
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from SpatialBiologyToolkit.backgating import create_population_overlay_galleries


NS = {'s': 'http://www.w3.org/2000/svg'}
INK = 'http://www.inkscape.org/namespaces/inkscape'


def write_source(base, population, color='red', *, roi='R1', size=(100, 80)):
    folder = base / population / 'population_overlays'
    folder.mkdir(parents=True, exist_ok=True)
    with BytesIO() as buffer:
        Image.new('RGB', (10, 8), color).save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    source = f'''<svg xmlns="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:inkscape="{INK}"
      width="100pt" height="80pt" viewBox="0 0 {size[0]} {size[1]}">
      <defs>
        <clipPath id="clip"><rect width="100" height="80"/></clipPath>
        <linearGradient id="gradient"><stop stop-color="white"/></linearGradient>
        <path id="marker" d="M1 1L5 5"/>
        <style>* {{stroke-linejoin:round}} #cell {{stroke:{color}}}</style>
      </defs>
      <g id="source_image" inkscape:groupmode="layer" inkscape:label="Source image">
        <image width="100" height="80" xlink:href="data:image/png;base64,{encoded}"/>
      </g>
      <g id="cell_outlines" inkscape:groupmode="layer" inkscape:label="Cell outlines" clip-path="url(#clip)">
        <path id="cell" d="M10 10L90 10L90 70L10 70Z" fill="none" stroke="white"/>
        <use xlink:href="#marker"/>
      </g>
      <g id="scale_bar"><path d="M10 65L35 65" stroke="white"/></g>
      <g id="population_label"><text x="10" y="25">Cell &amp; label</text></g>
    </svg>'''
    path = folder / f'{roi}_population_overlay.svg'
    path.write_text(source, encoding='utf-8')
    Image.new('RGB', size, color).save(path.with_suffix('.png'))
    return path


def svg_path(base):
    return base / 'population_overlay_galleries' / 'R1_population_gallery.svg'


def test_sources_remain_editable_with_isolated_ids_and_references(tmp_path):
    first = write_source(tmp_path, 'A')
    second = write_source(tmp_path, 'B', 'blue')
    before = [first.read_bytes(), second.read_bytes()]
    output = create_population_overlay_galleries(tmp_path, ['A', 'B'], 2, 1, output_format='svg')
    root = ET.parse(output / 'R1_population_gallery.svg').getroot()
    ids = [e.get('id') for e in root.iter() if e.get('id')]
    assert len(ids) == len(set(ids))
    assert len(root.findall('.//s:image', NS)) == 2
    for i in (1, 2):
        prefix = f'panel_{i:03d}_source_'
        for original_id in ('source_image', 'cell_outlines', 'scale_bar', 'population_label'):
            assert root.find(f'.//s:g[@id="{prefix}{original_id}"]', NS) is not None
        assert root.find(f'.//s:g[@id="panel_{i:03d}"]', NS).get(f'{{{INK}}}groupmode') == 'layer'
    for element in root.iter():
        for key, value in element.attrib.items():
            for ref in re.findall(r'url\(#([^)]*)\)', value):
                assert ref in ids
            if key.endswith('href') and value.startswith('#'):
                assert value[1:] in ids
    styles = [node.text for node in root.findall('.//s:style', NS)]
    assert '#panel_001_source_artwork #panel_001_source_cell' in styles[0]
    assert '#panel_002_source_artwork #panel_002_source_cell' in styles[1]
    assert sum('Cell & label' == t.text for t in root.findall('.//s:text', NS)) == 2
    assert [first.read_bytes(), second.read_bytes()] == before


def test_exact_spacing_dimensions_zero_gaps_and_different_aspect_ratios(tmp_path):
    for population in ('A', 'B', 'C', 'D'):
        write_source(tmp_path, population, size=(200, 50))
    create_population_overlay_galleries(
        tmp_path, ['A', 'B', 'C', 'D'], 2, 2, output_format='svg', panel_size=(120, 80),
        row_spacing=7, column_spacing=11, population_title_fontsize=None, roi_title_fontsize=None)
    root = ET.parse(svg_path(tmp_path)).getroot()
    assert root.get('viewBox') == '0 0 251 167'
    sources = root.findall('./s:g/s:svg', NS)
    assert [(float(s.get('x')), float(s.get('y'))) for s in sources] == [(0, 0), (131, 0), (0, 87), (131, 87)]
    assert all(s.get('preserveAspectRatio') == 'xMidYMid meet' for s in sources)
    assert all(s.get('viewBox') == '0 0 200 50' for s in sources)
    create_population_overlay_galleries(
        tmp_path, ['A', 'B'], 2, 1, output_format='svg', panel_size=(120, 80),
        row_spacing=0, column_spacing=0, population_title_fontsize=None, roi_title_fontsize=None)
    assert ET.parse(svg_path(tmp_path)).getroot().get('viewBox') == '0 0 240 80'


def test_auto_svg_only_and_mixed_raster_panels(tmp_path):
    source = write_source(tmp_path, 'A')
    source.with_suffix('.png').unlink()
    source = write_source(tmp_path, 'B')
    source.unlink()
    create_population_overlay_galleries(tmp_path, ['A', 'B'], 2, 1, output_format='svg')
    root = ET.parse(svg_path(tmp_path)).getroot()
    assert root.find('.//s:g[@id="panel_001_source_cell_outlines"]', NS) is not None
    assert root.find('.//s:g[@id="panel_002_source_image"]', NS) is not None
    assert len(root.findall('.//s:image', NS)) == 2


def test_png_default_and_explicit_png_source_stay_raster(tmp_path):
    write_source(tmp_path, 'A')
    output = create_population_overlay_galleries(tmp_path, ['A'], 1, 1, panel_size=(72, 72),
                                                population_title_fontsize=None, roi_title_fontsize=None, dpi=100)
    assert Image.open(output / 'R1_population_gallery.png').size == (100, 100)
    assert not svg_path(tmp_path).exists()
    create_population_overlay_galleries(tmp_path, ['A'], 1, 1, output_format='svg', source_format='png')
    root = ET.parse(svg_path(tmp_path)).getroot()
    assert not root.findall('.//s:path', NS)
    assert len(root.findall('.//s:image', NS)) == 1


def test_svg_preview_renders_actual_svg_not_stale_png_and_retains_gaps(tmp_path):
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError):
        pytest.skip('Optional CairoSVG/Cairo renderer is unavailable')
    first = write_source(tmp_path, 'A', 'red')
    second = write_source(tmp_path, 'B', 'blue')
    # Stale previews must not replace the source SVGs when exporting both.
    Image.new('RGB', (100, 80), 'green').save(first.with_suffix('.png'))
    Image.new('RGB', (100, 80), 'green').save(second.with_suffix('.png'))
    output = create_population_overlay_galleries(
        tmp_path, ['A', 'B'], 2, 1, output_format='both', panel_size=(100, 80),
        column_spacing=10, population_title_fontsize=None, roi_title_fontsize=None, dpi=72)
    pixels = np.asarray(Image.open(output / 'R1_population_gallery.png').convert('RGB'))
    assert pixels.shape[:2] == (80, 210)
    np.testing.assert_array_equal(pixels[40, 50], [255, 0, 0])
    np.testing.assert_array_equal(pixels[40, 160], [0, 0, 255])
    np.testing.assert_array_equal(pixels[:, 101:109], np.full((80, 8, 3), 255))


def test_missing_corrupt_panels_and_hidden_titles(tmp_path, caplog):
    source = write_source(tmp_path, 'A')
    source.write_text('<svg malformed', encoding='utf-8')
    create_population_overlay_galleries(tmp_path, ['A', 'B'], 2, 1, output_format='svg',
                                        population_title_fontsize=None, roi_title_fontsize=None)
    root = ET.parse(svg_path(tmp_path)).getroot()
    assert [t.text for t in root.findall('.//s:text', NS)] == ['Load failed', 'Missing']
    assert 'load failed' in caplog.text
    assert root.find('.//s:g[@id="gallery_heading"]', NS) is None


@pytest.mark.parametrize('kwargs', [
    {'row_spacing': -1}, {'column_spacing': np.nan}, {'panel_size': (0, 1)},
    {'ncols': 1.5}, {'output_format': 'pdf'}, {'source_format': 'jpg'}, {'dpi': 0},
])
def test_invalid_layout_options(tmp_path, kwargs):
    options = dict(ncols=1, nrows=1, **{'output_format': 'svg'})
    options.update(kwargs)
    with pytest.raises(ValueError):
        create_population_overlay_galleries(tmp_path, ['A'], **options)
    assert not (tmp_path / 'population_overlay_galleries').exists()


def test_absolute_dimensions_without_viewbox_and_title_bands(tmp_path):
    source = write_source(tmp_path, 'A')
    source.write_text(source.read_text().replace('viewBox="0 0 100 80"', ''), encoding='utf-8')
    create_population_overlay_galleries(tmp_path, ['A'], 1, 1, output_format='svg',
                                        panel_size=(120, 80), population_title_fontsize=10, roi_title_fontsize=20)
    root = ET.parse(svg_path(tmp_path)).getroot()
    assert root.get('viewBox') == '0 0 120 130'
    source = root.find('./s:g/s:svg', NS)
    np.testing.assert_allclose([float(v) for v in source.get('viewBox').split()], [0, 0, 100 * 96 / 72, 80 * 96 / 72])
    assert source.get('y') == '50.0'


def test_missing_optional_renderer_still_saves_svg(tmp_path, monkeypatch):
    import sys
    write_source(tmp_path, 'A')
    monkeypatch.setitem(sys.modules, 'cairosvg', None)
    with pytest.raises(RuntimeError, match='CairoSVG'):
        create_population_overlay_galleries(tmp_path, ['A'], 1, 1, output_format='both')
    assert svg_path(tmp_path).is_file()
    # SVG-only mode has no dependency on the preview renderer.
    create_population_overlay_galleries(tmp_path, ['A'], 1, 1, output_format='svg')


def test_actual_sbt_overlay_components_survive_composition(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from skimage import io
    from SpatialBiologyToolkit.plotting import create_population_overlay

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[..., 2] = np.arange(120, dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint16)
    mask[20:55, 20:50] = 1
    mask[20:55, 50:80] = 2
    io.imsave(tmp_path / 'source.png', image, check_contrast=False)
    io.imsave(tmp_path / 'mask.tif', mask, check_contrast=False)
    data = SimpleNamespace(obs=pd.DataFrame({
        'population': ['A', 'B'], 'ROI': ['R1', 'R1'], 'ObjectNumber': [1, 2],
    }))
    for population in ('A', 'B'):
        folder = tmp_path / population / 'population_overlays'
        folder.mkdir(parents=True)
        create_population_overlay(
            data, population, 'population', 'R1', tmp_path / 'source.png',
            mask_path=tmp_path / 'mask.tif', output_path=folder / 'R1_population_overlay.svg',
            show_scale_bar=True, scale_bar_length=25, scale_bar_text='25 px',
            legend_markers=['CD3'], legend_colors=['cyan'], verbose=False,
        )
    create_population_overlay_galleries(tmp_path, ['A', 'B'], 2, 1, output_format='svg',
                                        row_spacing=0, column_spacing=9, panel_size=(216, 180))
    root = ET.parse(svg_path(tmp_path)).getroot()
    for i in (1, 2):
        for name in ('source_image', 'cell_outlines', 'scale_bar', 'scale_bar_text',
                     'marker_legend', 'population_label'):
            assert root.find(f'.//s:g[@id="panel_{i:03d}_source_{name}"]', NS) is not None
    assert len(root.findall('.//s:image', NS)) == 2
    assert sum(t.text == '25 px' for t in root.findall('.//s:text', NS)) == 2


@pytest.mark.parametrize('family', ['Arial', 'Liberation Sans'])
def test_gallery_normalizes_legacy_font_lists_and_shorthand(tmp_path, family):
    source = write_source(tmp_path, 'A')
    legacy = source.read_text().replace(
        '<text x="10" y="25">',
        '<text x="10" y="25" style="font-size:10px;font-family:DejaVu Sans, Bitstream Vera Sans, sans-serif">')
    legacy = legacy.replace('</style>', "text {font: italic 700 11px/1.2 'Computer Modern Sans Serif', sans-serif;}</style>")
    source.write_text(legacy, encoding='utf-8')
    create_population_overlay_galleries(tmp_path, ['A'], 1, 1, output_format='svg', font_family=family)
    root = ET.parse(svg_path(tmp_path)).getroot()
    output = svg_path(tmp_path).read_text()
    for old in ('DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', ', sans-serif'):
        assert old not in output
    assert f"font: italic 700 11px/1.2 '{family}'" in output
    texts = root.findall('.//s:text', NS)
    assert len(texts) == 3  # ROI title, population title, and original live text.
    for text in texts:
        assert family in (text.get('style', '') + text.get('font-family', ''))
    assert source.read_text() == legacy


def test_svg_font_rewrite_does_not_change_label_content_or_geometry():
    from SpatialBiologyToolkit.plotting import _set_svg_font_family
    root = ET.fromstring('''<svg xmlns="http://www.w3.org/2000/svg">
        <text x="3" y="4" transform="rotate(20)" font-family="Geneva"
        style="font: bold 12px 'Geneva',sans-serif;fill:red">DejaVu Sans<tspan
        style="font-family:Lucid;font-size:8px"> Arial</tspan></text>
        <path d="M1 1L3 4"/></svg>''')
    _set_svg_font_family(root, 'Arial')
    text = root.find('s:text', NS)
    assert text.text == 'DejaVu Sans'
    assert text.get('transform') == 'rotate(20)'
    assert text.get('style') == "font: bold 12px 'Arial';fill:red"
    assert text.get('font-family') == 'Arial'
    assert text.find('s:tspan', NS).get('font-family') == 'Arial'
    assert root.find('s:path', NS).get('d') == 'M1 1L3 4'
