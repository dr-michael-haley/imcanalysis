"""Synthetic segmentation and editable export regressions (no project assets)."""
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from skimage import io

from SpatialBiologyToolkit.plotting import (
    _population_cell_paths,
    create_population_overlay,
)


SVG = {'s': 'http://www.w3.org/2000/svg'}


@pytest.fixture
def scene(tmp_path):
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[..., 0] = np.arange(120, dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint32)
    mask[10:60, 10:45] = 1
    mask[10:60, 45:80] = 2  # Shared edge, same population.
    mask[60:85, 80:105] = 3  # Other population, must not be outlined.
    io.imsave(tmp_path / 'source.png', image, check_contrast=False)
    io.imsave(tmp_path / 'mask.tif', mask, check_contrast=False)
    obs = pd.DataFrame({
        'ROI': ['R1'] * 3,
        'population': ['T & B', 'T & B', 'Other'],
        'ObjectNumber': [1, 2, 3],
        'Master_Index': [1, 2, 3],
        'X_loc': [25, 60, 90], 'Y_loc': [35, 35, 70],
    })
    return SimpleNamespace(obs=obs), image, mask


def render(tmp_path, scene, **kwargs):
    return create_population_overlay(
        scene[0], 'T & B', 'population', 'R1', str(tmp_path / 'source.png'),
        mask_path=str(tmp_path / 'mask.tif'), verbose=False, **kwargs,
    )


def test_touching_cells_have_individual_closed_paths(scene):
    paths = dict(_population_cell_paths(scene[2], {0, 1, 2, 999}))
    assert set(paths) == {1, 2}
    assert paths[1].vertices[:, 0].max() == 44.5
    assert paths[2].vertices[:, 0].min() == 44.5
    for path in paths.values():
        np.testing.assert_array_equal(path.vertices[0], path.vertices[-1])


def test_sparse_ids_holes_disconnected_regions_and_image_edges():
    mask = np.zeros((12, 12), dtype=np.int64)
    mask[:6, :6] = 10**9
    mask[2:4, 2:4] = 0  # Hole within the cell.
    mask[9:11, 9:11] = 10**9  # Disconnected part of the same label.
    paths = dict(_population_cell_paths(mask, {10**9}))
    path = paths[10**9]
    assert np.count_nonzero(path.codes == path.MOVETO) == 3
    assert np.count_nonzero(path.codes == path.CLOSEPOLY) == 3
    assert path.vertices.min() == -0.5
    # Missing IDs must not wrap around the mask dtype and select another cell.
    assert list(_population_cell_paths(np.array([[1]], dtype=np.uint8), {257})) == []


@pytest.mark.parametrize('crop_size', [None, (90, 80)])
def test_svg_has_separate_vectors_text_and_unmodified_source(tmp_path, scene, crop_size):
    svg_path = tmp_path / 'overlay.svg'
    fig = render(
        tmp_path, scene, output_path=tmp_path / 'overlay.png', svg_output_path=svg_path,
        crop_size=crop_size, crop_origin='upper_left', contour_width=1,
        show_scale_bar=True, scale_bar_length=25, scale_bar_thickness=1,
        scale_bar_text='25 px', legend_markers=['CD3 & CD4'], legend_colors=['red'],
    )
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), scene[1])
    assert len(fig.axes[0].images) == 1
    root = ET.parse(svg_path).getroot()
    ids = [e.get('id') for e in root.iter() if e.get('id')]
    assert len(ids) == len(set(ids))
    for group in ('source_image', 'cell_outlines', 'scale_bar', 'scale_bar_text',
                  'marker_legend', 'population_label'):
        element = root.find(f'.//s:g[@id="{group}"]', SVG)
        assert element is not None, group
        assert element.get('{http://www.inkscape.org/namespaces/inkscape}groupmode') == 'layer'
    assert root.find('.//s:g[@id="cell_outline_1"]/s:path', SVG) is not None
    assert root.find('.//s:g[@id="cell_outline_2"]/s:path', SVG) is not None
    assert 'cell_outline_3' not in ids
    assert len(root.findall('.//s:g[@id="scale_bar"]//s:path', SVG)) == 2
    texts = [''.join(t.itertext()) for t in root.findall('.//s:text', SVG)]
    assert all(t.get('font-family') == 'Arial' for t in root.findall('.//s:text', SVG))
    assert 'DejaVu Sans' not in svg_path.read_text()
    assert {'25 px', 'CD3 & CD4', 'T & B'} <= set(texts)
    images = root.findall('.//s:image', SVG)
    assert len(images) == 1
    href = images[0].get('{http://www.w3.org/1999/xlink}href')
    assert href.startswith('data:image/png;base64,')
    embedded = Image.open(BytesIO(base64.b64decode(href.split(',', 1)[1]))).convert('RGB')
    # The SVG backend may omit pixels outside the view, but retained pixels
    # must be unchanged (including where the separate scale bar is drawn).
    expected = scene[1][:embedded.height, :embedded.width]
    np.testing.assert_array_equal(np.asarray(embedded), expected)
    if crop_size is None:
        assert embedded.size == (scene[1].shape[1], scene[1].shape[0])
    assert (tmp_path / 'overlay.png').is_file()
    assert not plt.fignum_exists(fig.number)


def test_png_contains_shared_boundary(tmp_path, scene):
    fig = render(tmp_path, scene, output_path=tmp_path / 'overlay.png', show_label=False)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())
    x, y = np.rint(fig.axes[0].transData.transform((44.5, 35))).astype(int)
    assert pixels[pixels.shape[0] - y, x, :3].min() > 200


def test_no_mask_exports_vector_centers_without_disabled_layers(tmp_path, scene):
    svg_path = tmp_path / 'centers.svg'
    create_population_overlay(
        scene[0], 'T & B', 'population', 'R1', tmp_path / 'source.png',
        output_path=svg_path, show_label=False, verbose=False,
    )
    root = ET.parse(svg_path).getroot()
    assert root.find('.//s:g[@id="cell_centers"]', SVG) is not None
    for group in ('cell_outlines', 'scale_bar', 'marker_legend', 'population_label'):
        assert root.find(f'.//s:g[@id="{group}"]', SVG) is None


def test_coordinate_fallback_excludes_background(tmp_path, scene):
    scene[0].obs = scene[0].obs.drop(columns='ObjectNumber')
    fig = render(tmp_path, scene, output_path=tmp_path / 'fallback.svg', show_label=False)
    assert {p.get_gid() for p in fig.axes[0].patches} == {'cell_outline_1', 'cell_outline_2'}


def test_invalid_mask_shape_does_not_leak_figure(tmp_path, scene):
    io.imsave(tmp_path / 'mask.tif', np.ones((10, 10), dtype=np.uint8), check_contrast=False)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match='match the composite'):
        render(tmp_path, scene)
    assert plt.get_fignums() == before


@pytest.mark.parametrize('extension,save_svg', [('png', True), ('svg', False), ('png', False)])
def test_backgating_export_and_gallery_companion(tmp_path, scene, monkeypatch, extension, save_svg):
    from SpatialBiologyToolkit import backgating

    scene[0].obs['population'] = ['T', 'T', 'Other']
    settings = pd.DataFrame({
        'Red': ['CD3'], 'Green': ['CD4'], 'Blue': ['DNA'],
        'Red_min': [0], 'Green_min': [0], 'Blue_min': [0],
        'Red_max': [1], 'Green_max': [1], 'Blue_max': [1],
    }, index=['T'])
    settings.to_csv(tmp_path / 'backgating_settings.csv')

    def fake_backgating(**kwargs):
        assert kwargs['font_family'] == 'Arial'
        assert kwargs['gallery_sampling'] == 'intelligent'
        assert kwargs['gallery_markers'] == ['CD3']
        assert kwargs['gallery_umap_weight'] == 0.1
        assert kwargs['gallery_save_svg'] is True
        folder = Path(kwargs['output_folder']) / kwargs['save_subfolder']
        folder.mkdir(exist_ok=True)
        io.imsave(folder / 'R1.png', scene[1], check_contrast=False)

    monkeypatch.setattr(backgating, 'backgating', fake_backgating)
    backgating.backgating_assessment(
        scene[0], str(tmp_path), 'population', output_folder=str(tmp_path),
        mode='load_markers', pops_list=['T'], use_masks=False,
        population_overlay_extension=extension, population_overlay_save_svg=save_svg,
        gallery_sampling='intelligent', gallery_markers=['CD3'], gallery_umap_weight=0.1,
    )
    folder = tmp_path / 'T' / 'population_overlays'
    assert (folder / 'R1_population_overlay.png').is_file()
    assert (folder / 'R1_population_overlay.svg').is_file() == (save_svg or extension == 'svg')
    assert backgating._find_population_overlay_image(tmp_path, 'T', 'R1').suffix == '.png'
    gallery = backgating.create_population_overlay_galleries(tmp_path, ['T'], 1, 1)
    assert (gallery / 'R1_population_gallery.png').is_file()
