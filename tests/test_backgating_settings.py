"""Saved marker ranges remain attached to markers throughout backgating."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit import backgating as bg


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    data = ad.AnnData(
        np.array([[2., 3., 4., 1.], [2., 3., 4., 1.]]),
        obs=pd.DataFrame({'Specific': ['P', 'Q'], 'ROI': ['R1', 'R2'],
                          'Master_Index': [1, 2]}, index=['cell1', 'cell2']),
        var=pd.DataFrame(index=['A', 'B', 'C', 'D']),
    )
    settings = pd.DataFrame({
        'Red': ['A', 'A'], 'Green': ['B', 'B'], 'Blue': ['C', 'C'],
        'Red_min': [1., 10.], 'Red_max': [20., 200.],
        'Green_min': [3., 30.], 'Green_max': [40., 400.],
        'Blue_min': [5., 50.], 'Blue_max': [60., 600.],
    }, index=['P', 'Q'])
    path = tmp_path / 'settings.csv'
    settings.to_csv(path)
    calls = []
    monkeypatch.setattr(bg, 'backgating', lambda **kwargs: calls.append(kwargs))

    def run(**kwargs):
        options = dict(adata=data, image_folder=tmp_path / 'images', pop_obs='Specific',
                       output_folder=tmp_path, backgating_settings_file='settings.csv',
                       use_differential_expression=False, population_overlays=False,
                       minimum=0.5, max_quantile='q0.97', pops_list=['P'])
        options.update(kwargs)
        bg.backgating_assessment(**options)
        return pd.read_csv(path, index_col=0)

    return run, path, calls, data


def test_norm_dict_then_load_markers_preserves_csv_and_ranges(workflow, monkeypatch):
    run, path, calls, _ = workflow
    bg.update_settings_from_marker_dict(path, {'A': (2., 62.5), 'B': (2., 60.), 'C': (2., 20.)})
    before = path.read_bytes(), path.stat().st_mtime_ns

    def forbidden_selection(*args, **kwargs):
        pytest.fail('Loading saved settings must not select markers')

    monkeypatch.setattr(bg, 'get_top_columns', forbidden_selection)
    run(mode='load_markers')
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert (calls[0]['red'], calls[0]['green'], calls[0]['blue']) == ('A', 'B', 'C')
    assert calls[0]['red_range'] == (2., 62.5)
    assert calls[0]['blue_range'] == (2., 20.)
    assert not (path.parent / 'markers_mean_expression.csv').exists()


def test_full_reselection_moves_ranges_with_markers_and_preserves_other_rows(workflow, caplog):
    run, path, calls, _ = workflow
    before = pd.read_csv(path, index_col=0)
    result = run(mode='full')
    assert result.loc['P', ['Red', 'Green', 'Blue']].tolist() == ['C', 'B', 'A']
    assert calls[0]['red_range'] == (5., 60.)
    assert calls[0]['green_range'] == (3., 40.)
    assert calls[0]['blue_range'] == (1., 20.)
    pd.testing.assert_series_equal(result.loc['Q'], before.loc['Q'])
    assert "mode='load_markers'" in caplog.text


@pytest.mark.parametrize('mode', ['full', 'save_markers', 'load_markers'])
def test_new_marker_override_cannot_inherit_old_marker_range(workflow, mode):
    run, path, calls, _ = workflow
    before = path.read_bytes()
    result = run(mode=mode, specify_red='D')
    if mode == 'load_markers':
        assert path.read_bytes() == before
    else:
        assert result.loc['P', 'Red'] == 'D'
        assert result.loc['P', 'Red_min'] == 0.5
        assert result.loc['P', 'Red_max'] == 'q0.97'
    if mode != 'save_markers':
        assert calls[0]['red'] == 'D'
        assert calls[0]['red_range'] == (0.5, 'q0.97')


def test_swapped_overrides_use_original_ranges_without_overwriting_source(workflow):
    run, path, calls, _ = workflow
    before = path.read_bytes()
    run(mode='load_markers', specify_red='B', specify_green='A')
    assert calls[0]['red_range'] == (3., 40.)
    assert calls[0]['green_range'] == (1., 20.)
    assert path.read_bytes() == before


def test_missing_ranges_fill_only_in_memory_when_loading(workflow):
    run, path, calls, _ = workflow
    settings = pd.read_csv(path, index_col=0).drop(columns=['Red_min', 'Red_max'])
    settings.to_csv(path)
    before = path.read_bytes()
    run(mode='load_markers')
    assert calls[0]['red_range'] == (0.5, 'q0.97')
    assert path.read_bytes() == before


def test_ambiguous_saved_marker_ranges_fall_back_to_defaults(workflow):
    run, path, calls, _ = workflow
    settings = pd.read_csv(path, index_col=0)
    settings.loc['P', 'Blue'] = 'B'
    settings.to_csv(path)
    run(mode='load_markers', specify_red='B')
    assert calls[0]['red_range'] == (0.5, 'q0.97')


def test_save_markers_preserves_existing_assignments_and_limits(workflow):
    run, path, calls, _ = workflow
    before = pd.read_csv(path, index_col=0)
    result = run(mode='save_markers')
    pd.testing.assert_frame_equal(result, before)
    assert calls == []


def test_load_markers_requires_existing_file_without_creating_one(workflow):
    run, path, calls, _ = workflow
    path.unlink()
    with pytest.raises(FileNotFoundError, match='save_markers'):
        run(mode='load_markers')
    assert not path.exists()
    assert calls == []


def test_load_markers_missing_population_does_not_rewrite_settings(workflow):
    run, path, calls, data = workflow
    data.obs.loc['cell1', 'Specific'] = 'missing'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='No saved backgating settings'):
        run(mode='load_markers', pops_list=['missing'])
    assert path.read_bytes() == before
    assert calls == []


def test_new_template_and_numeric_population_labels(workflow):
    run, path, calls, data = workflow
    path.unlink()
    data.obs['Specific'] = [0, 1]
    result = run(mode='save_markers', pops_list=[0])
    assert result.index.tolist() == [0]
    run(mode='load_markers', pops_list=[0])
    assert len(calls) == 1
    assert calls[0]['red'] == 'C'


@pytest.mark.parametrize('balance', [True, False])
def test_assessment_forwards_roi_balancing_option(workflow, balance):
    run, _, calls, _ = workflow
    run(mode='load_markers', gallery_sampling='intelligent', gallery_balance_rois=balance)
    assert calls[0]['gallery_balance_rois'] is balance
