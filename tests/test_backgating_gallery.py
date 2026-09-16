"""Representative thumbnail selection and layered gallery export tests."""
import base64
import json
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import anndata as ad
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import sparse
from skimage import io

from SpatialBiologyToolkit import backgating as bg
from SpatialBiologyToolkit.config.models import VisualizationConfig


def make_adata(matrix):
    matrix = np.asarray(matrix, dtype=float)
    obs = pd.DataFrame({
        'Master_Index': np.arange(len(matrix)) + 100,
        'ROI': ['R1'] * len(matrix),
        'X_loc': [40] * len(matrix), 'Y_loc': [40] * len(matrix),
    }, index=[f'cell{i}' for i in range(len(matrix))])
    return ad.AnnData(matrix, obs=obs,
                      var=pd.DataFrame(index=[f'M{i}' for i in range(matrix.shape[1])]))


def test_default_matches_existing_seeded_random_sampling():
    data = make_adata(np.arange(20).reshape(-1, 1))
    selected, metadata = bg._select_gallery_cells(data, data.obs, 5)
    assert selected.index.tolist() == data.obs.sample(5, random_state=0).index.tolist()
    assert metadata['method'] == 'random'
    assert data.obs.columns.tolist() == ['Master_Index', 'ROI', 'X_loc', 'Y_loc']
    assert not any(column.startswith('gallery_') for column in data.obs)


@pytest.mark.parametrize('sparse_matrix', [False, True])
def test_intelligent_selects_median_cells_and_rejects_outliers(sparse_matrix):
    data = make_adata([[-100, 2], [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [100, 2]])
    if sparse_matrix:
        data.X = sparse.csr_matrix(data.X)
    selected, metadata = bg._select_gallery_cells(
        data, data.obs, 3, sampling='intelligent', umap_weight=0)
    assert selected['Master_Index'].tolist() == [103, 102, 104]
    assert selected['gallery_expression_distance'].iloc[0] == 0
    assert metadata['marker_profiles'][1]['scale_method'] == 'constant'
    assert metadata['marker_profiles'][0]['median'] == 0


def test_layer_marker_subset_and_alignment_with_reordered_candidates():
    data = make_adata([[10, 100], [0, 1], [20, 2], [30, 3]])
    data.layers['normalized'] = sparse.csc_matrix([[0, 0], [20, 0], [1, 0], [2, 0]])
    candidates = data.obs.iloc[[3, 0, 2]]
    selected, _ = bg._select_gallery_cells(
        data, candidates, 1, sampling='intelligent', layer='normalized', markers=['M0'], umap_weight=0)
    assert selected['Master_Index'].tolist() == [102]
    assert selected['gallery_expression_distance'].tolist() == [0]


def test_marker_scale_invariance_and_zero_mad_fallback():
    data = make_adata([[0, 1], [0, 2], [0, 3], [1, 4], [10, 50]])
    first, metadata = bg._select_gallery_cells(data, data.obs, 3, sampling='intelligent', umap_weight=0)
    assert metadata['marker_profiles'][0]['scale_method'] == 'standard_deviation'
    data.X[:, 1] *= 1e6
    second, _ = bg._select_gallery_cells(data, data.obs, 3, sampling='intelligent', umap_weight=0)
    assert first.index.tolist() == second.index.tolist()
    np.testing.assert_allclose(first['gallery_score'], second['gallery_score'])


def test_umap_breaks_expression_tie_and_missing_umap_falls_back(caplog):
    data = make_adata([[-10], [-2], [-1], [1], [2], [10]])
    # Expression distances for -1 and +1 are tied. Only -1 is UMAP-central.
    data.obsm['X_umap'] = np.array([[0, 0], [0, 0], [0, 0], [8, 8], [0, 0], [0, 0]])
    selected, metadata = bg._select_gallery_cells(data, data.obs, 1, sampling='intelligent')
    assert selected['Master_Index'].tolist() == [102]
    assert metadata['effective_umap_weight'] == 0.2
    del data.obsm['X_umap']
    selected, metadata = bg._select_gallery_cells(data, data.obs, 1, sampling='intelligent')
    assert metadata['effective_umap_weight'] == 0
    assert 'expression-only' in caplog.text
    np.testing.assert_array_equal(data.X.ravel(), [-10, -2, -1, 1, 2, 10])


def test_nonfinite_expression_excluded_and_missing_embedding_penalized():
    data = make_adata([[-1], [0], [1], [np.nan]])
    data.obsm['X_umap'] = np.array([[0, 0], [np.nan, np.nan], [0, 0], [0, 0]])
    selected, metadata = bg._select_gallery_cells(data, data.obs, 4, sampling='intelligent')
    assert len(selected) == 3
    assert 103 not in selected['Master_Index'].values
    assert selected['gallery_score'].notna().all()
    assert metadata['finite_expression_cells'] == 3
    assert np.isnan(selected.loc['cell1', 'gallery_umap_distance'])


@pytest.mark.parametrize('kwargs,match', [
    ({'sampling': 'bad'}, 'gallery_sampling'),
    ({'umap_weight': 1.1}, 'gallery_umap_weight'),
    ({'balance_rois': 'yes'}, 'gallery_balance_rois'),
    ({'markers': ['missing']}, 'markers not present'),
    ({'markers': []}, 'at least one'),
    ({'layer': 'missing'}, 'expression layer'),
])
def test_actionable_invalid_options(kwargs, match):
    data = make_adata([[0], [1]])
    kwargs = {'sampling': 'intelligent', **kwargs}
    with pytest.raises(ValueError, match=match):
        bg._select_gallery_cells(data, data.obs, 1, **kwargs)


def test_empty_limit_single_cell_and_duplicate_identifiers():
    data = make_adata([[1]])
    selected, _ = bg._select_gallery_cells(data, data.obs, 1, sampling='intelligent', umap_weight=0)
    assert selected['gallery_expression_distance'].tolist() == [0]
    selected, _ = bg._select_gallery_cells(data, data.obs, 0, sampling='intelligent')
    assert selected.empty
    data = make_adata([[0], [1]])
    data.obs['Master_Index'] = 100
    with pytest.raises(ValueError, match='requires unique'):
        bg._select_gallery_cells(data, data.obs, 1, sampling='intelligent')


@pytest.mark.parametrize('save_svg,show_titles', [(True, True), (True, False), (False, False)])
def test_gallery_images_vectors_selection_audit_and_full_cell_list(tmp_path, monkeypatch, save_svg, show_titles):
    data = make_adata([[-10], [-1], [0], [1], [10]])
    # The expression-median cell cannot supply a complete crop and must be excluded.
    data.obs.loc['cell2', 'X_loc'] = 1
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[..., 0] = np.arange(80, dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint16)
    mask[35:45, 35:45] = 7
    io.imsave(tmp_path / 'R1.tif', mask, check_contrast=False)

    def fake_make_images(**kwargs):
        io.imsave(tmp_path / 'R1.png', image, check_contrast=False)

    monkeypatch.setattr(bg, 'make_images', fake_make_images)
    bg.backgating(
        data, list(data.obs.Master_Index), 10, str(tmp_path), output_folder=str(tmp_path),
        use_masks=True, mask_folder=str(tmp_path), overview_images=False,
        cells_per_row=2, max_gallery_cells=2, gallery_sampling='intelligent',
        gallery_umap_weight=0, gallery_save_svg=save_svg, show_gallery_titles=show_titles,
    )
    assert (tmp_path / 'Cells.png').is_file()
    assert (tmp_path / 'Cells.svg').is_file() == save_svg
    selected = pd.read_csv(tmp_path / 'gallery_cells.csv', index_col=0)
    assert selected['Master_Index'].tolist() == [101, 103]
    assert selected['gallery_display_order'].tolist() == [1, 2]
    assert len(pd.read_csv(tmp_path / 'cells_list.csv')) == 4
    metadata = json.loads((tmp_path / 'gallery_sampling.json').read_text())
    assert metadata['eligible_cells'] == 4
    assert metadata['selected_cells'] == 2
    assert data.obs.columns.tolist() == ['Master_Index', 'ROI', 'X_loc', 'Y_loc']
    if save_svg:
        ns = {'s': 'http://www.w3.org/2000/svg'}
        root = ET.parse(tmp_path / 'Cells.svg').getroot()
        ids = [e.get('id') for e in root.iter() if e.get('id')]
        assert len(ids) == len(set(ids))
        for i in (1, 2):
            assert root.find(f'.//s:g[@id="gallery_outline_{i}"]//s:path', ns) is not None
            element = root.find(f'.//s:g[@id="gallery_image_{i}"]//s:image', ns)
            href = element.get('{http://www.w3.org/1999/xlink}href')
            thumbnail = np.asarray(Image.open(BytesIO(base64.b64decode(href.split(',', 1)[1]))).convert('RGB'))
            np.testing.assert_array_equal(thumbnail, image[30:50, 30:50])
        assert len(root.findall('.//s:image', ns)) == 2
        assert bool(root.findall('.//s:text', ns)) == show_titles
        assert all(t.get('font-family') == 'Arial' for t in root.findall('.//s:text', ns))
    assert not plt.get_fignums()


def test_gallery_config_defaults_and_validation():
    config = VisualizationConfig()
    assert config.backgating_gallery_sampling == 'random'
    assert config.backgating_gallery_balance_rois is True
    assert VisualizationConfig(backgating_gallery_balance_rois=False).backgating_gallery_balance_rois is False
    assert config.backgating_gallery_save_svg is True
    assert config.backgating_font_family == 'Arial'
    with pytest.raises(ValueError):
        VisualizationConfig(backgating_gallery_umap_weight=-0.1)
    with pytest.raises(ValueError):
        VisualizationConfig(backgating_gallery_sampling='other')


def test_intelligent_balances_rois_while_preserving_population_median_scores():
    data = make_adata([[0]] * 20 + [[1], [2], [3], [4], [5], [6]])
    data.obs['ROI'] = ['large'] * 20 + ['small1'] * 3 + ['small2'] * 3
    balanced, metadata = bg._select_gallery_cells(
        data, data.obs, 9, sampling='intelligent', umap_weight=0)
    pooled, pooled_meta = bg._select_gallery_cells(
        data, data.obs, 9, sampling='intelligent', umap_weight=0, balance_rois=False)
    assert balanced['ROI'].value_counts().to_dict() == {'large': 3, 'small1': 3, 'small2': 3}
    assert pooled['ROI'].unique().tolist() == ['large']
    assert metadata['marker_profiles'] == pooled_meta['marker_profiles']
    assert metadata['marker_profiles'][0]['median'] == 0
    assert metadata['eligible_cells_per_roi'] == {'large': 20, 'small1': 3, 'small2': 3}
    assert metadata['selected_cells_per_roi'] == {'large': 3, 'small1': 3, 'small2': 3}
    assert metadata['selected_rois'] == 3
    assert metadata['effective_balance_rois'] is True
    all_cells, _ = bg._select_gallery_cells(
        data, data.obs, None, sampling='intelligent', umap_weight=0, balance_rois=False)
    np.testing.assert_array_equal(balanced['gallery_score'], all_cells.loc[balanced.index, 'gallery_score'])
    for _, group in balanced.groupby('ROI'):
        assert group['gallery_roi_rank'].tolist() == [1, 2, 3]
    again, _ = bg._select_gallery_cells(data, data.obs, 9, sampling='intelligent', umap_weight=0)
    pd.testing.assert_frame_equal(balanced, again)


def test_roi_balancing_redistributes_shortfalls_and_excludes_invalid_cells():
    data = make_adata([[0]] * 20 + [[1], [2], [3], [np.nan]])
    data.obs['ROI'] = ['large'] * 20 + ['small1'] + ['small2'] * 2 + ['invalid']
    selected, metadata = bg._select_gallery_cells(
        data, data.obs, 10, sampling='intelligent', umap_weight=0)
    assert selected['ROI'].value_counts().to_dict() == {'large': 7, 'small2': 2, 'small1': 1}
    assert len(selected['Master_Index'].unique()) == 10
    assert 'invalid' not in metadata['eligible_cells_per_roi']


@pytest.mark.parametrize('limit', [0, 2, 4, 100, None])
def test_roi_balancing_partial_rounds_and_limits(limit):
    data = make_adata([[0], [0], [0], [1], [2], [3]])
    data.obs['ROI'] = pd.Categorical(['A', 'A', 'A', 'B', 'B', 'C'], categories=['A', 'B', 'C', 'unused'])
    selected, metadata = bg._select_gallery_cells(
        data, data.obs, limit, sampling='intelligent', umap_weight=0)
    assert len(selected) == (6 if limit is None else min(limit, 6))
    if limit == 2:
        assert selected['Master_Index'].tolist() == [100, 103]
    elif limit == 4:
        assert selected['ROI'].value_counts().to_dict() == {'A': 2, 'B': 1, 'C': 1, 'unused': 0}
    assert 'unused' not in metadata['selected_cells_per_roi']


def test_roi_balancing_does_not_change_random_sampling():
    data = make_adata(np.arange(30).reshape(-1, 1))
    data.obs['ROI'] = ['large'] * 25 + ['small'] * 5
    with_balance, metadata = bg._select_gallery_cells(data, data.obs, 10, balance_rois=True)
    without_balance, _ = bg._select_gallery_cells(data, data.obs, 10, balance_rois=False)
    pd.testing.assert_frame_equal(with_balance, without_balance)
    assert metadata['effective_balance_rois'] is False


@pytest.mark.parametrize('missing_column', [False, True])
def test_roi_balancing_requires_roi_identity(missing_column):
    data = make_adata([[0], [1]])
    if missing_column:
        del data.obs['ROI']
    else:
        data.obs.loc['cell0', 'ROI'] = None
    with pytest.raises(ValueError, match='ROI-balanced'):
        bg._select_gallery_cells(data, data.obs, 1, sampling='intelligent')
    selected, _ = bg._select_gallery_cells(
        data, data.obs, 1, sampling='intelligent', balance_rois=False, umap_weight=0)
    assert len(selected) == 1


def test_backgating_balances_custom_roi_column_and_saves_audit(tmp_path, monkeypatch):
    data = make_adata([[0], [1], [2], [3], [10], [20]])
    data.obs = data.obs.rename(columns={'ROI': 'region'})
    data.obs['region'] = ['A'] * 4 + ['B'] * 2

    def fake_images(**kwargs):
        for roi in ['A', 'B']:
            io.imsave(tmp_path / f'{roi}.png', np.zeros((80, 80, 3), dtype=np.uint8), check_contrast=False)

    monkeypatch.setattr(bg, 'make_images', fake_images)
    bg.backgating(data, list(data.obs.Master_Index), 10, str(tmp_path), output_folder=str(tmp_path),
                  roi_obs='region', use_masks=False, overview_images=False,
                  max_gallery_cells=4, gallery_sampling='intelligent', gallery_balance_rois=True,
                  gallery_umap_weight=0, gallery_save_svg=False)
    selected = pd.read_csv(tmp_path / 'gallery_cells.csv')
    assert selected['region'].value_counts().to_dict() == {'A': 2, 'B': 2}
    metadata = json.loads((tmp_path / 'gallery_sampling.json').read_text())
    assert metadata['roi_obs'] == 'region'
    assert metadata['selected_cells_per_roi'] == {'A': 2, 'B': 2}
    assert len(pd.read_csv(tmp_path / 'cells_list.csv')) == 6
