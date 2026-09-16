"""Backgating reads only images required by each channel's intensity bounds."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import tifffile
from skimage import exposure

from SpatialBiologyToolkit import backgating as bg


@pytest.fixture
def images(tmp_path, monkeypatch):
    root = tmp_path / 'images'
    arrays = {}
    for roi, scale in [('R1', 1), ('R2', 10), ('R3', 100), ('outside', 1000)]:
        folder = root / roi
        folder.mkdir(parents=True)
        arrays[roi] = np.array([[0, 1], [2, 8]], dtype=np.float32) * scale
        for marker in ('CD14', 'MCT4'):
            tifffile.imwrite(folder / f'{marker}.tiff', arrays[roi])
    reads = []
    original = bg.load_single_img

    def record_read(path):
        path = Path(path)
        reads.append((path.parent.name, path.stem))
        return original(str(path))

    monkeypatch.setattr(bg, 'load_single_img', record_read)
    return root, arrays, reads


@pytest.mark.parametrize('maximum', [32.5, np.float64(32.5), '32.5', '1'])
def test_fixed_bounds_skip_other_rois_and_all_quantile_calculations(images, monkeypatch, maximum):
    root, arrays, reads = images

    def forbidden_quantile(*args, **kwargs):
        pytest.fail('Fixed bounds must not compute quantiles')

    monkeypatch.setattr(bg.np, 'quantile', forbidden_quantile)
    scaled, rois, maxima = bg.load_rescale_images(
        root, ['R1', 'R2', 'R3'], 'CD14', np.float64(0.5), maximum,
        save_samples_list=['R2'])
    assert reads == [('R2', 'CD14')]
    assert rois == ['R2']
    assert maxima == [float(maximum)]
    np.testing.assert_array_equal(
        scaled[0], exposure.rescale_intensity(arrays['R2'].clip(0.5, float(maximum))))


@pytest.mark.parametrize('maximum,reduce', [('q0.5', np.mean), ('m0.5', np.min), ('x0.5', np.max)])
def test_cohort_quantile_uses_full_eligible_scope_but_only_rescales_saved_roi(
        images, monkeypatch, maximum, reduce):
    root, arrays, reads = images
    rescaled = []
    original = bg.exposure.rescale_intensity

    def record_rescale(im):
        rescaled.append(im)
        return original(im)

    monkeypatch.setattr(bg.exposure, 'rescale_intensity', record_rescale)
    scaled, rois, maxima = bg.load_rescale_images(
        root, ['R1', 'R2', 'R3'], 'CD14', 0, maximum, save_samples_list=['R2'])
    expected_max = float(reduce([np.quantile(arrays[roi], 0.5) for roi in ['R1', 'R2', 'R3']]))
    assert set(reads) == {(roi, 'CD14') for roi in ['R1', 'R2', 'R3']}
    assert len(reads) == 3
    assert len(rescaled) == 1
    assert rois == ['R2']
    assert maxima == [expected_max]
    np.testing.assert_array_equal(scaled[0], original(arrays['R2'].clip(0, expected_max)))


def test_individual_quantile_reads_only_saved_roi(images):
    root, arrays, reads = images
    scaled, rois, maxima = bg.load_rescale_images(
        root, ['R1', 'R2', 'R3'], 'CD14', 0, 'i0.5', save_samples_list=['R2'])
    assert reads == [('R2', 'CD14')]
    assert rois == ['R2']
    assert maxima == [np.quantile(arrays['R2'], 0.5)]
    np.testing.assert_array_equal(scaled[0], exposure.rescale_intensity(arrays['R2'].clip(0, maxima[0])))


@pytest.mark.parametrize('maximum', [32.5, 'i0.97', 'q0.97', 'm0.97', 'x0.97'])
def test_saved_subset_matches_full_run_pixels_and_bounds(images, maximum):
    root, _, _ = images
    full, rois, maxima = bg.load_rescale_images(root, ['R1', 'R2', 'R3'], 'CD14', 0.5, maximum)
    subset, subset_rois, subset_maxima = bg.load_rescale_images(
        root, ['R1', 'R2', 'R3'], 'CD14', 0.5, maximum, save_samples_list=['R2'])
    idx = rois.index('R2')
    assert subset_rois == ['R2']
    assert subset_maxima == [maxima[idx]]
    np.testing.assert_array_equal(subset[0], full[idx])


@pytest.mark.parametrize('maximum', [32.5, 'q0.97', 'i0.97'])
@pytest.mark.parametrize('saved', [[], ['outside']])
def test_empty_output_scope_does_not_read_any_images(images, maximum, saved):
    root, _, reads = images
    assert bg.load_rescale_images(root, ['R1'], 'CD14', 0, maximum, save_samples_list=saved) == ([], [], [])
    assert reads == []


def test_directory_filter_applied_before_read_and_missing_markers_stay_aligned(images):
    root, arrays, reads = images
    (root / 'R1' / 'CD14.tiff').unlink()
    loaded, filenames, folders = bg.load_imgs_from_directory(
        root, 'CD14', samples_list=['R1', 'R2'])
    assert reads == [('R2', 'CD14')]
    assert filenames == ['CD14.tiff']
    assert [Path(folder).name for folder in folders] == ['R2']
    np.testing.assert_array_equal(loaded[0], arrays['R2'])


@pytest.mark.parametrize('green_max', [50.0, 'q0.5'])
def test_make_images_decides_read_scope_from_each_effective_channel_range(images, tmp_path, green_max):
    root, arrays, reads = images
    output = tmp_path / 'output'
    df = bg.make_images(
        root, ['R1', 'R2', 'R3'], output, simple_file_names=True,
        minimum=0.5, max_quantile='q0.97',
        red='CD14', red_range=(np.float64(1), np.float64(32.5)),
        green='MCT4', green_range=(0.5, green_max), save_samples_list=['R2'])
    assert [roi for roi, marker in reads if marker == 'CD14'] == ['R2']
    expected_green = ['R2'] if isinstance(green_max, float) else ['R1', 'R2', 'R3']
    assert sorted(roi for roi, marker in reads if marker == 'MCT4') == expected_green
    assert {file.name for file in output.glob('*.png')} == {'R2.png'}
    assert df['roi'].tolist() == ['R2', 'R2']
    assert df['min_used'].tolist() == [1, 0.5]
    expected_max = green_max if isinstance(green_max, float) else np.mean([
        np.quantile(arrays[roi], 0.5) for roi in ['R1', 'R2', 'R3']])
    assert df['max_used'].tolist() == [32.5, expected_max]


def test_numeric_global_defaults_also_skip_unselected_rois(images, tmp_path):
    root, _, reads = images
    bg.make_images(root, ['R1', 'R2', 'R3'], tmp_path / 'output',
                   minimum=0.5, max_quantile=32.5, red='CD14', save_samples_list=['R2'])
    assert reads == [('R2', 'CD14')]


def test_missing_channels_do_not_drop_or_mislabel_output_rois(images, tmp_path):
    root, _, reads = images
    (root / 'R1' / 'CD14.tiff').unlink()
    (root / 'R2' / 'MCT4.tiff').unlink()
    output = tmp_path / 'output'
    df = bg.make_images(root, ['R1', 'R2'], output, simple_file_names=True,
                        red='CD14', green='MCT4', max_quantile=32.5)
    assert set(reads) == {('R1', 'MCT4'), ('R2', 'CD14')}
    assert {file.name for file in output.glob('*.png')} == {'R1.png', 'R2.png'}
    assert set(zip(df['roi'], df['marker'])) == {('R1', 'MCT4'), ('R2', 'CD14')}
    r1 = bg.io.imread(output / 'R1.png')
    r2 = bg.io.imread(output / 'R2.png')
    assert not r1[:, :, 0].any() and r1[:, :, 1].any()
    assert r2[:, :, 0].any() and not r2[:, :, 1].any()


@pytest.mark.parametrize('maximum', ['', 'bad', 'q2', 'i-1', 'mnan', 'xinf', float('nan')])
def test_invalid_maximum_fails_before_image_reads(images, maximum):
    root, _, reads = images
    with pytest.raises(ValueError):
        bg.load_rescale_images(root, ['R1'], 'CD14', 0, maximum)
    assert reads == []
