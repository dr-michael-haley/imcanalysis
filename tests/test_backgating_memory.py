"""Bounded-memory paths must retain the previous scientific/visual results."""
import weakref

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
import tifffile
from skimage import exposure, io
from skimage.draw import rectangle_perimeter
from skimage.util import img_as_ubyte

from SpatialBiologyToolkit import backgating as bg
from SpatialBiologyToolkit.plotting import _intelligent_crop_origin


def original_crop(xs, ys, w, h, cw, ch):
    """Previous full-image integral algorithm, retained as a result oracle."""
    counts = np.zeros((h, w), dtype=np.uint16)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    np.add.at(counts, (yi, xi), 1)
    integral = np.pad(counts.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    sums = integral[ch:, cw:] - integral[:-ch, cw:] - integral[ch:, :-cw] + integral[:-ch, :-cw]
    y, x = np.where(sums == sums.max())
    best = np.argmin((x + cw / 2 - xs.mean()) ** 2 + (y + ch / 2 - ys.mean()) ** 2)
    return int(x[best]), int(y[best])


@pytest.mark.parametrize('shape,crop', [((30, 40), (1, 1)), ((30, 40), (40, 30)),
                                     ((30, 40), (13, 17)), ((1, 40), (7, 1))])
def test_sweep_crop_exactly_matches_integral_windows_and_ties(shape, crop):
    h, w = shape
    cw, ch = crop
    rng = np.random.default_rng(42)
    for n in [1, 5, 100, 1000]:
        for _ in range(5):
            xs = rng.uniform(-2, w + 2, n)
            ys = rng.uniform(-2, h + 2, n)
            assert _intelligent_crop_origin(xs, ys, w, h, cw, ch) == original_crop(xs, ys, w, h, cw, ch)
    assert _intelligent_crop_origin([], [], w, h, cw, ch) == ((w - cw) // 2, (h - ch) // 2)


@pytest.mark.parametrize('maximum', [25., 'i0.7', 'q0.7', 'm0.7', 'x0.7'])
def test_composite_streaming_matches_pixels_and_releases_raw_arrays(tmp_path, monkeypatch, maximum):
    root, out = tmp_path / 'images', tmp_path / 'output'
    rng = np.random.default_rng(1)
    arrays = {}
    colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'white']
    component_ids = [(0,), (1,), (2,), (0, 2), (1, 2), (0, 1), (0, 1, 2)]
    for roi in ['R1', 'R2', 'R3']:
        (root / roi).mkdir(parents=True)
        for j, color in enumerate(colors):
            image = rng.uniform(-2, (j + 1) * 10, (10, 12)).astype(np.float32)
            arrays[roi, color] = image
            tifffile.imwrite(root / roi / f'{color}.tiff', image)
    original_load = bg.load_single_img
    refs = []

    def tracked_load(path):
        assert not any(ref() is not None for ref in refs), 'Previous raw ROI/channel image is still retained'
        image = original_load(path)
        refs.append(weakref.ref(image))
        return image

    monkeypatch.setattr(bg, 'load_single_img', tracked_load)
    bg.make_images(root, ['R1', 'R2', 'R3'], out, simple_file_names=True, minimum=0.5,
                   max_quantile=maximum, save_samples_list=['R1', 'R2'], **dict(zip(colors, colors)))
    for roi in ['R1', 'R2']:
        expected = np.zeros((10, 12, 3), dtype=np.float32)
        for color, components in zip(colors, component_ids):
            if isinstance(maximum, float):
                upper = maximum
            elif maximum[0] == 'i':
                upper = np.quantile(arrays[roi, color], 0.7)
            else:
                reducer = {'q': np.mean, 'm': np.min, 'x': np.max}[maximum[0]]
                upper = reducer([np.quantile(arrays[r, color], 0.7) for r in ['R1', 'R2', 'R3']])
            scaled = exposure.rescale_intensity(arrays[roi, color].clip(0.5, upper))
            for c in components:
                expected[:, :, c] = np.clip(expected[:, :, c] + scaled, 0, 1)
        np.testing.assert_array_equal(io.imread(out / f'{roi}.png'), img_as_ubyte(expected))
    assert not (out / 'R3.png').exists()


def test_thumbnail_and_overview_stream_each_roi_once(tmp_path, monkeypatch):
    obs = pd.DataFrame({'Master_Index': [1, 2, 3], 'ROI': ['R1', 'R2', 'R3'],
                        'X_loc': [30]*3, 'Y_loc': [30]*3}, index=['a', 'b', 'c'])
    data = ad.AnnData(np.ones((3, 1)), obs=obs, var=pd.DataFrame(index=['marker']))
    source = np.full((60, 60, 3), 70, dtype=np.uint8)
    for roi in obs.ROI:
        io.imsave(tmp_path / f'{roi}.png', source, check_contrast=False)
        mask = np.zeros((60, 60), dtype=np.uint16)
        mask[25:35, 25:35] = 1
        tifffile.imwrite(tmp_path / f'{roi}.tif', mask)
    monkeypatch.setattr(bg, 'make_images', lambda **kwargs: None)
    original = bg.io.imread
    refs, reads = [], []

    def read(path, *args, **kwargs):
        from pathlib import Path
        roi = Path(path).stem
        assert all(other == roi or ref() is None for other, ref in refs)
        image = original(path, *args, **kwargs)
        refs.append((roi, weakref.ref(image)))
        reads.append(Path(path).name)
        return image

    monkeypatch.setattr(bg.io, 'imread', read)
    bg.backgating(data, [1, 2, 3], 10, tmp_path, output_folder=tmp_path,
                  use_masks=True, mask_folder=tmp_path, overview_images=True,
                  max_gallery_cells=2, gallery_sampling='intelligent', gallery_umap_weight=0)
    assert sorted(reads) == ['R1.png', 'R1.tif', 'R2.png', 'R2.tif', 'R3.png']
    expected_overview = source.copy()
    rr, cc = rectangle_perimeter((20, 20), extent=(20, 20), shape=source.shape)
    expected_overview[rr, cc, :] = 255
    for roi in obs.ROI:
        overview = original(tmp_path / f'{roi}_overview.png')
        np.testing.assert_array_equal(overview, expected_overview)
        np.testing.assert_array_equal(original(tmp_path / f'{roi}.png'), source)
    assert len(pd.read_csv(tmp_path / 'cells_list.csv')) == 3


@pytest.mark.parametrize('backed', [False, True])
def test_intelligent_dense_and_backed_reads_only_one_marker_at_a_time(tmp_path, monkeypatch, backed):
    data = ad.AnnData(np.arange(40, dtype=np.float32).reshape(10, 4),
                     obs=pd.DataFrame({'Master_Index': range(10), 'ROI': ['R1']*10},
                                      index=[str(x) for x in range(10)]))
    if backed:
        path = tmp_path / 'test.h5ad'
        data.write_h5ad(path)
        data = ad.read_h5ad(path, backed='r')
    original = ad.AnnData.X

    def getter(self):
        if self.is_view:
            assert self.n_vars == 1, 'Materialized a multi-marker candidate expression matrix'
        return original.fget(self)

    monkeypatch.setattr(ad.AnnData, 'X', property(getter, original.fset, original.fdel))
    try:
        selected, _ = bg._select_gallery_cells(data, data.obs.iloc[[8, 1, 5, 4]], 2,
                                              sampling='intelligent', umap_weight=0)
        assert selected.Master_Index.tolist() == [5, 4]
    finally:
        if backed:
            data.file.close()
