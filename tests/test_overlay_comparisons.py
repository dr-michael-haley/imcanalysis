"""Registered modality panels: matching, shared crops, and editable export."""
import json
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
import pytest
from skimage import io

from SpatialBiologyToolkit._overlay_comparisons import prepare_comparison_images, _load_comparison_crop
from SpatialBiologyToolkit.config.models import BackgatingComparisonImageConfig, VisualizationConfig
from test_population_overlay import scene, render, SVG


def test_roi_matching_is_ranked_and_never_guesses_ambiguities(tmp_path):
    names = ['ROI_1.png', 'HE-ROI-001-registered.tif', 'ROI_10.jpg',
             'nested/case02_roi003_HE.ome.tif', 'A-ROI_4.png', 'B-ROI_4.png']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    source, = prepare_comparison_images([tmp_path])
    assert source.match('roi_1')[0].name == 'ROI_1.png'
    assert source.match('ROI 001')[0].name == 'ROI_1.png'
    assert source.match('case2_roi3')[0].name == 'case02_roi003_HE.ome.tif'
    assert source.match('ROI_10')[0].name == 'ROI_10.jpg'
    assert source.match('ROI_4')[0] is None
    assert source.match('ROI_4')[1].startswith('Ambiguous')
    assert source.match('ROI_100') == (None, 'No matching image')
    # A prepared source remains reusable without scanning folders again.
    assert prepare_comparison_images([source])[0] is source


@pytest.mark.parametrize('origin', ['center', 'upper_left', 'lower_right', 'intelligent'])
def test_scaled_panels_share_crop_pixels_and_coordinates(tmp_path, scene, origin):
    folder = tmp_path / 'HE'
    folder.mkdir()
    pixels = np.zeros((50, 60, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(60) * 4
    pixels[..., 1] = np.arange(50)[:, None] * 5
    Image.fromarray(pixels).save(folder / 'he_R01_registered.png')
    path = tmp_path / 'comparison.svg'
    fig = render(tmp_path, scene, output_path=path, crop_size=(60, 50), crop_origin=origin,
                 comparison_images=[dict(folder=folder, title='H&E', interpolation='nearest',
                                         legend={'Tumour': 'red'}, show_cell_outlines=True)],
                 show_scale_bar=True, scale_bar_length=15)
    primary = fig.axes[0]
    extra = next(ax for ax in fig.axes if ax.get_gid() == 'comparison_1_panel')
    np.testing.assert_allclose(extra.get_xlim(), primary.get_xlim())
    np.testing.assert_allclose(extra.get_ylim(), primary.get_ylim())
    # Equal scale in physical space, including after adding titles and legends.
    np.testing.assert_allclose(extra.transData.get_matrix()[:2, :2], primary.transData.get_matrix()[:2, :2])
    record, = json.loads(path.with_suffix('.comparisons.json').read_text())
    assert record['status'] == 'matched'
    assert record['original_shape'] == [50, 60]
    assert record['target_shape'] == [100, 120]
    x0, y0, x1, y1 = record['crop_pixel_bounds']
    expected = np.asarray(Image.fromarray(pixels).resize((120, 100), Image.Resampling.NEAREST))
    np.testing.assert_array_equal(extra.images[0].get_array(), expected[y0:y1, x0:x1])
    np.testing.assert_array_equal(extra.patches[0].get_path().vertices, primary.patches[0].get_path().vertices)
    root = ET.parse(path).getroot()
    ids = [el.get('id') for el in root.iter() if el.get('id')]
    assert len(ids) == len(set(ids))
    assert ids.count('scale_bar') == 1
    assert len(root.findall('.//s:image', SVG)) == 2
    for gid in ['primary_panel_title', 'comparison_1_image', 'comparison_1_title',
                'comparison_1_legend', 'comparison_1_cell_outlines']:
        group = root.find(f'.//s:g[@id="{gid}"]', SVG)
        assert group is not None, gid
        assert group.get('{http://www.inkscape.org/namespaces/inkscape}groupmode') == 'layer'
    texts = root.findall('.//s:text', SVG)
    assert {'IMC', 'H&E', 'Tumour'} <= {''.join(t.itertext()) for t in texts}
    assert all(t.get('font-family') == 'Arial' for t in texts)


def test_missing_and_ambiguous_images_keep_placeholders(tmp_path, scene):
    empty = tmp_path / 'empty'
    ambiguous = tmp_path / 'ambiguous'
    empty.mkdir()
    ambiguous.mkdir()
    for name in ('a_R1.png', 'b_R1.png'):
        Image.fromarray(scene[1]).save(ambiguous / name)
    path = tmp_path / 'missing.svg'
    fig = render(tmp_path, scene, output_path=path, show_label=False,
                 comparison_images=[empty, ambiguous])
    assert len(fig.axes) == 3
    assert len(fig.axes[0].images) == 1
    assert not fig.axes[1].images and not fig.axes[2].images
    records = json.loads(path.with_suffix('.comparisons.json').read_text())
    assert records[0]['status'] == 'No matching image'
    assert records[1]['status'].startswith('Ambiguous match')
    assert all(record['path'] is None for record in records)


def test_resize_bilinear_and_high_bit_depth(tmp_path):
    pixels = np.array([[[0, 65535, 32768], [65535, 0, 32768]]], dtype=np.uint16)
    path = tmp_path / 'rgb.tif'
    io.imsave(path, pixels, check_contrast=False)
    crop, original_shape, limits = _load_comparison_crop(path, (2, 4), (0, 0, 4, 2), 'bilinear')
    assert original_shape == (1, 2)
    assert crop.dtype == np.uint8 and limits is None
    assert crop[0, 0].tolist() == [0, 255, 128]
    assert 0 < crop[0, 1, 0] < 255


def test_typed_config_validation_and_shortcuts(tmp_path):
    config = VisualizationConfig(backgating_population_overlay_comparison_images=[
        str(tmp_path), dict(folder=str(tmp_path), title='Labels', legend={'A': [255, 0, 128]},
                            interpolation='nearest')])
    sources = prepare_comparison_images(config.backgating_population_overlay_comparison_images)
    assert len(sources) == 2
    assert sources[1].legend['A'] == (1, 0, 128 / 255)
    assert VisualizationConfig().backgating_population_overlay_comparison_images == []
    for bad in ({'legend': {'bad': [256, 1, 0]}}, {'interpolation': 'bicubic'}, {'typo': True}):
        with pytest.raises(ValueError):
            BackgatingComparisonImageConfig(folder=str(tmp_path), **bad)
    with pytest.raises(FileNotFoundError):
        prepare_comparison_images([tmp_path / 'missing'])
    with pytest.raises(ValueError, match='list'):
        prepare_comparison_images(str(tmp_path))
