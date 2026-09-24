"""Small scientific scenes exercise the figures contract without project data."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import weakref
from xml.etree import ElementTree as ET

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import pytest
from scipy.sparse import csr_matrix
import tifffile

from SpatialBiologyToolkit import figures as F
from SpatialBiologyToolkit.figures.sources import ROIContext, resample_crop
from SpatialBiologyToolkit.figures.selection import select_view


@pytest.fixture
def dataset(tmp_path):
    for name in ('imc', 'masks', 'he', 'labels'):
        (tmp_path / name).mkdir()
    labels = np.zeros((24, 32), dtype=np.uint32)
    labels[3:10, 3:10] = 1
    labels[3:10, 10:17] = 2  # Adjacent cells of the same population.
    labels[15:21, 23:29] = 90000000
    labels[12:15, 3:7] = 4
    regions = np.zeros_like(labels)
    regions[:, 16:] = 1
    regions[10:15, 18:23] = 0  # Hole must remain a hole in SVG.
    records, expression = [], []
    for roi_index, roi in enumerate(('R1', 'R2')):
        folder = tmp_path / 'imc' / roi
        folder.mkdir()
        raw = np.arange(24 * 32, dtype=np.float32).reshape(24, 32) + roi_index * 100
        tifffile.imwrite(folder / 'CD3.tif', raw)
        tifffile.imwrite(folder / 'DNA1.tif', raw / 2)
        tifffile.imwrite(tmp_path / 'masks' / f'{roi}.tif', labels)
        tifffile.imwrite(tmp_path / 'labels' / f'annotated_{roi}.tif', regions)
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(16) * 15
        rgb[..., 1] = np.arange(12)[:, None] * 20
        PILImage.fromarray(rgb).save(tmp_path / 'he' / f'HE_{roi}_registered.png')
        for ident, pop, x, y, value in [(1,'A',6,6,-1.5),(2,'A',13,6,0.25),
                                      (90000000,'B',26,18,1.75),(4,'B',5,13,np.nan)]:
            records.append(dict(ROI=roi, ObjectNumber=ident, population=pop,
                                X_loc=x, Y_loc=y, score=value))
            expression.append([value if np.isfinite(value) else 0, 3])
    obs = pd.DataFrame(records, index=[f'c{i}' for i in range(len(records))])
    obs['population'] = pd.Categorical(obs.population, categories=['B', 'A'])
    data = ad.AnnData(csr_matrix(expression), obs=obs, var=pd.DataFrame(index=['M','N']))
    data.layers['scaled'] = data.X * 2
    return F.Dataset(data, imc_folder=tmp_path/'imc', mask_folder=tmp_path/'masks',
                     image_folders={'he':tmp_path/'he'}, label_folders={'regions':tmp_path/'labels'},
                     pixel_size_um=.5), tmp_path


def recipe():
    return F.Figure(layout=(2, 3), crop=F.Crop(mode='center', size=(24, 20)),
        style=F.Style(dpi=80, panel_width_mm=40, panel_height_mm=36), panels=[
        F.Panel(row=0,col=0,title='IMC',letter='A',scale_bar=F.ScaleBar(length=5,unit='um'), layers=[
            F.IMC(channels=[F.Channel('CD3',color='red',limits=(0,1000)), F.Channel('DNA1',color='blue',limits=(0,500))])]),
        F.Panel(row=0,col=1,title='Populations',layers=[F.Populations(obs='population', mode='both')]),
        F.Panel(row=0,col=2,title='H&E',layers=[F.Image(source='he')]),
        F.Panel(row=1,col=0,title='Score',layers=[F.Values(value=F.obs('score'),scale=F.Scale(mode='fixed',limits=(-2,2)))]),
        F.Panel(row=1,col=1,title='Annotation',layers=[F.Image(source='he'), F.LabelMask(source='regions',labels={1:'Tumour'},opacity=.4, mode='both')]),
        F.Panel(row=1,col=2,title='Expression',layers=[F.Values(value=F.var('M',layer='scaled'))])])


def test_shared_intensities_reuse_and_grid(dataset, monkeypatch):
    data, root = dataset
    path = root / 'limits.csv'
    path.write_text('marker,vmax,lower_threshold\nCD3,1000,0\nDNA1,500,0\n')
    limits = F.Intensities.from_csv(path)
    layer = F.IMC(channels=[F.Channel('CD3'), F.Channel('DNA1')])
    figure = F.Figure.grid([[F.Panel(layers=[layer]), F.Panel(layers=[layer])]], intensities=limits)
    assert figure.panels[0].layers[0].id != figure.panels[1].layers[0].id
    assert layer.channels[0].scale is None
    assert [p.letter for p in figure.panels] == ['A', 'B']
    assert sum(p.scale_bar is not None for p in figure.panels) == 1
    import SpatialBiologyToolkit.figures.rendering as rendering
    monkeypatch.setattr(rendering, 'read_image', lambda *args: pytest.fail('Fixed bounds must not load images to calibrate'))
    prepared = figure.prepare(data, rois=['R1'])
    assert set(prepared.bounds.values()) == {(0, 1000), (0, 500)}
    assert F.Figure.model_validate_json(figure.model_dump_json()).model_dump() == figure.model_dump()
    figure.intensities = F.Intensities(limits={'CD3': (0, 1)})
    with pytest.raises(ValueError, match='DNA1'):
        figure.prepare(data)
    layer.channels[1].scale = F.Scale(mode='fixed', limits=(0, 500))
    override = F.Figure(panels=[F.Panel(layers=[layer])], intensities=figure.intensities)
    assert override.prepare(data).bounds


def test_intensity_csv_rejects_duplicates_and_nonfinite(tmp_path):
    path = tmp_path / 'limits.csv'
    path.write_text('marker,vmax,lower_threshold\nA,10,1\nA,20,2\n')
    with pytest.raises(ValueError, match='Duplicate'):
        F.Intensities.from_csv(path)
    with pytest.raises(ValueError):
        F.Intensities(limits={'A': (0, float('nan'))})


def test_scale_max_preserves_minima_original_and_source(tmp_path):
    path = tmp_path / 'normalization.csv'
    path.write_text('marker,vmax,lower_threshold\nA,10,1\nB,20,2\n')
    before = path.read_bytes()
    original = F.Intensities.from_csv(path)
    scaled = original.scale_max(0.5)
    assert scaled.limits == {'A': (1, 5), 'B': (2, 10)}
    assert original.limits == {'A': (1, 10), 'B': (2, 20)}
    assert scaled.limits is not original.limits
    assert scaled.source == original.source
    assert scaled.source_sha256 == original.source_sha256
    assert scaled.scale_max(2).limits == original.limits
    assert path.read_bytes() == before
    assert F.Intensities.model_validate_json(scaled.model_dump_json()) == scaled


@pytest.mark.parametrize('factor', [0, -1, float('nan'), float('inf'), None, 'bad', True])
def test_scale_max_rejects_invalid_factors(factor):
    with pytest.raises(ValueError, match='finite positive'):
        F.Intensities(limits={'A': (1, 10)}).scale_max(factor)


def test_scale_max_rejects_invalid_result_without_mutation():
    original = F.Intensities(limits={'A': (4, 6)})
    with pytest.raises(ValueError, match='A'):
        original.scale_max(0.5)
    assert original.limits == {'A': (4, 6)}
    with pytest.raises(ValueError):
        F.Intensities(limits={'A': (0, 1e308)}).scale_max(10)


def test_readonly_mapping_and_in_memory_h5ad(dataset):
    data, root = dataset
    path = root / 'annotations.csv'
    path.write_text('source_population,Broad\nA,Tumour\nB,Myeloid\n')
    data.map_obs(path, source='population', columns=['Broad'])
    assert 'Broad' not in data.adata.obs
    assert data.observations('R1')['Broad'].tolist() == ['Tumour', 'Tumour', 'Myeloid', 'Myeloid']
    with F.Figure(panels=[F.Panel(layers=[F.Populations(obs='Broad')])]).preview(data, 'R1') as result:
        assert result.metadata['palettes']
    before = data.obs
    path.write_text('source_population,Bad\nA,Tumour\n')
    with pytest.raises(ValueError, match='Unmapped'):
        data.map_obs(path, source='population', columns=['Bad'])
    assert data.obs is before
    h5 = root / 'test.h5ad'
    data.adata.write_h5ad(h5)
    contents = h5.read_bytes()
    loaded = F.Dataset.from_h5ad(h5, mask_folder=root/'masks')
    assert loaded.adata.shape == data.adata.shape
    assert list(loaded.adata.obs_names) == list(data.adata.obs_names)
    np.testing.assert_array_equal(loaded.adata.layers['scaled'].toarray(), data.adata.layers['scaled'].toarray())
    assert h5.read_bytes() == contents


def test_preflight_missing_inputs_and_ambiguity(dataset):
    data, root = dataset
    figure = recipe()
    report = figure.preflight(data)
    assert report.eligible_rois == ['R1', 'R2']
    assert report.table.loc['R1', 'resized_sources']
    assert report.select(2, method='random', seed=13) == report.select(2, method='random', seed=13)
    (root/'he'/'HE_R2_registered.png').unlink()
    report = figure.preflight(data)
    assert report.eligible_rois == ['R1']
    assert report.table.loc['R2', 'reason']
    # New binding reindexes files and sees an ambiguous match, which must raise.
    PILImage.new('RGB', (16, 12)).save(root/'he'/'other_R1_registered.png')
    other = F.Dataset(data.adata, imc_folder=root/'imc', mask_folder=root/'masks',
                      image_folders={'he': root/'he'}, label_folders={'regions': root/'labels'}, pixel_size_um=.5)
    with pytest.raises(ValueError, match='Ambiguous'):
        figure.preflight(other)


def test_case_balanced_preflight_selection(dataset):
    data, root = dataset
    data.obs = data.obs.assign(Case=['one'] * 4 + ['two'] * 4)
    report = recipe().preflight(data)
    assert report.select(2, method='most_cells', balance_by='Case') == ['R1', 'R2']
    with pytest.raises(ValueError, match='one nonmissing group'):
        report.select(2, balance_by='population')


def test_pooled_quantiles_are_cell_weighted():
    from SpatialBiologyToolkit.figures.normalization import calibrate
    arrays = {'R1': np.array([0]), 'R2': np.full(9, 10)}
    pooled = calibrate(F.Scale(mode='pooled_quantile', quantiles=(.5, .99)), list(arrays), arrays.get)
    cohort = calibrate(F.Scale(quantiles=(.5, .99)), list(arrays), arrays.get)
    assert pooled == (10, 10)
    assert cohort == (5, 5)


def test_preview_bundle_freezes_crop_bounds_and_palette(dataset):
    data, root = dataset
    figure = recipe().with_style(title_fontsize=15)
    assert figure.style.title_fontsize == 15
    with figure.preview(data, 'R1', dpi=80) as result:
        files = result.save_bundle(root/'bundle', dpi=80)
        frozen = result.freeze()
        assert len(files) == 2 and all(p.exists() for p in files)
        assert frozen.crop.roi_bounds['R1'] == tuple(result.metadata['view']['bounds'])
        assert (root/'bundle'/'dataset_binding.json').exists()
        for panel in frozen.panels:
            for layer in panel.layers:
                if isinstance(layer, F.Values):
                    assert layer.scale.mode == 'fixed'
        with frozen.preview(data, 'R1', dpi=80) as repeat:
            assert repeat.metadata['view']['bounds'] == result.metadata['view']['bounds']


def test_automatic_scale_bar_uses_calibration(dataset):
    data, root = dataset
    figure = F.Figure.grid([[F.Panel(layers=[F.Image(source='he')])]])
    with figure.preview(data, 'R1') as result:
        texts = [t.get_text() for axis in result.figure.axes for t in axis.texts]
        assert '2 µm' in texts
    data.pixel_size_um = None
    with figure.preview(data, 'R1') as result:
        texts = [t.get_text() for axis in result.figure.axes for t in axis.texts]
        assert '5 px' in texts


def test_legend_only_reuses_population_palette_and_exports(dataset):
    from matplotlib.colors import to_hex
    data, root = dataset
    layer = F.Populations(obs='population', mode='fill', colors={'A': '#ff0000', 'B': '#00ff00'})
    figure = F.Figure.grid([[
        F.Panel(title='Cells', layers=[layer], legend=False),
        F.Panel(title='Population key', layers=[layer], legend_only=True, legend_ncols=2, legend_fontsize=9),
    ]], scale_bar=False)
    assert figure.preflight(data).eligible_rois == ['R1', 'R2']
    with figure.preview(data, 'R1') as result:
        cells, key = result.figure.axes
        assert cells.get_legend() is None
        assert cells.patches and not key.images and not key.patches
        legend = key.get_legend()
        assert legend._ncols == 2
        assert [text.get_text() for text in legend.get_texts()] == ['B', 'A']
        assert all(text.get_fontsize() == 9 for text in legend.get_texts())
        assert [to_hex(handle.get_facecolor()) for handle in legend.legend_handles] == ['#00ff00', '#ff0000']
        result.figure.canvas.draw()
        outer, inner = key.get_window_extent(), legend.get_window_extent()
        assert outer.contains(inner.x0, inner.y0) and outer.contains(inner.x1, inner.y1)
        files = result.save_bundle(root/'legend_bundle', dpi=80)
        svg = ET.parse(next(p for p in files if p.suffix == '.svg')).getroot()
        legend_id = figure.panels[1].id + '_legend'
        group = next(el for el in svg.iter() if el.get('id') == legend_id)
        assert group.findall('.//{http://www.w3.org/2000/svg}text')
        frozen = result.freeze()
        assert frozen.panels[1].legend_only and frozen.panels[1].legend_ncols == 2
        assert frozen.panels[0].layers[0].colors == frozen.panels[1].layers[0].colors
        with frozen.preview(data, 'R1') as repeat:
            assert repeat.figure.axes[1].get_legend()._ncols == 2


def test_legend_only_has_no_mask_channel_or_calibration_dependency(dataset, monkeypatch):
    import SpatialBiologyToolkit.figures.rendering as rendering
    from SpatialBiologyToolkit.figures.sources import ROIContext
    original, root = dataset
    data = F.Dataset(original.adata, image_folders={'he': root/'he'})
    monkeypatch.setattr(rendering, 'read_image', lambda *args: pytest.fail('Legend must not calibrate channel images'))
    monkeypatch.setattr(ROIContext, 'mask', lambda *args: pytest.fail('Legend must not load segmentation masks'))
    figure = F.Figure.grid([[
        F.Panel(legend_only=True, legend_ncols=2, layers=[
            F.Populations(obs='population', groups=['A']),
            F.IMC.rgb('unavailable_red', 'unavailable_green', 'unavailable_blue'),
            F.LabelMask(source='unavailable_masks', labels={1: 'Region'}),
            F.Image(source='unavailable_image', colors={'Modality': 'pink'}),
        ]),
        F.Panel(layers=[F.Image(source='he')]),
    ]], intensities=F.Intensities(limits={}))
    assert figure.panels[0].scale_bar is None
    assert figure.panels[1].scale_bar is not None
    assert figure.preflight(data).eligible_rois == ['R1', 'R2']
    with figure.preview(data, 'R1') as result:
        texts = [text.get_text() for text in result.figure.axes[0].get_legend().get_texts()]
        assert texts == ['A', 'unavailable_red', 'unavailable_green', 'unavailable_blue', 'Region', 'Modality']
        assert result.metadata['scaling'] == {}
        result.save_bundle(root/'metadata_only_legend', dpi=80)


def test_legend_options_validation_and_existing_overlay_columns(dataset):
    data, _ = dataset
    layer = F.Populations(obs='population')
    for kwargs in [dict(legend=False), dict(scale_bar=F.ScaleBar.auto()), dict(legend_ncols=0)]:
        with pytest.raises(ValueError):
            F.Panel(layers=[layer], legend_only=True, **kwargs)
    with pytest.raises(ValueError, match='continuous'):
        F.Panel(layers=[F.Values(value=F.obs('score'))], legend_only=True)
    figure = F.Figure(panels=[F.Panel(layers=[layer], legend_ncols=2)])
    with figure.preview(data, 'R1') as result:
        assert result.figure.axes[0].get_legend()._ncols == 2
        assert result.figure.axes[0].patches
    empty = F.Figure(panels=[F.Panel(layers=[F.Image(source='he')], legend_only=True)])
    with pytest.raises(ValueError, match='no categorical legend entries'):
        empty.preview(data, 'R1')


@pytest.mark.parametrize('letter_size,scale_size,expected_letter,expected_scale', [
    (None, None, 13, 7),
    (19, 11, 19, 11),
])
def test_independent_letter_and_scale_bar_fonts(dataset, letter_size, scale_size,
                                              expected_letter, expected_scale):
    data, root = dataset
    figure = F.Figure.grid([[F.Panel(title='Histology', layers=[
        F.Image(source='he', colors={'Tissue': 'pink'})])]],
        style=F.Style(title_fontsize=13, legend_fontsize=7,
                      letter_fontsize=letter_size, scale_bar_fontsize=scale_size))
    figure = F.Figure.model_validate_json(figure.model_dump_json())
    panel_id = figure.panels[0].id
    with figure.preview(data, 'R1') as result:
        axis = result.figure.axes[0]
        labels = {text.get_gid(): text for text in axis.texts}
        assert labels[panel_id + '_letter'].get_fontsize() == expected_letter
        assert labels[panel_id + '_scale_bar_text'].get_fontsize() == expected_scale
        assert result.figure.texts[0].get_fontsize() == 13
        assert axis.get_legend().get_texts()[0].get_fontsize() == 7
        path = result.save(root / 'font_sizes.svg')
        svg = ET.parse(path).getroot()
        texts = svg.findall('.//{http://www.w3.org/2000/svg}text')
        letter = next(text for text in texts if text.text == 'A')
        scale = next(text for text in texts if text.text == '2 µm')
        assert f'font-size: {expected_letter}px' in letter.get('style')
        assert f'font-size: {expected_scale}px' in scale.get('style')


def test_freeze_preserves_degenerate_image_bounds(dataset):
    data, root = dataset
    raw = np.zeros((24, 32), dtype=np.float32)
    raw[5, 5] = 100  # Both display quantiles are zero despite this outlier.
    tifffile.imwrite(root/'imc'/'R1'/'CD3.tif', raw)
    figure = F.Figure(panels=[F.Panel(layers=[F.IMC(channels=[
        F.Channel('CD3', scale=F.Scale(mode='roi_quantile'))])])])
    with figure.preview(data, 'R1') as result:
        expected = np.asarray(result.figure.axes[0].images[0].get_array())
        frozen = result.freeze()
        assert frozen.panels[0].layers[0].channels[0].scale.limits == (0, 0)
        with frozen.preview(data, 'R1') as repeat:
            np.testing.assert_array_equal(expected, repeat.figure.axes[0].images[0].get_array())
    report = frozen.preflight(data)
    assert report.eligible_rois == ['R1']


def test_recipe_roundtrip_validation_and_lightweight_import(tmp_path):
    figure = recipe()
    for suffix in ('json','yaml'):
        path = tmp_path / f'recipe.{suffix}'
        figure.save(path)
        assert F.Figure.load(path).model_dump() == figure.model_dump()
    assert 'discriminator' in json.dumps(F.Figure.model_json_schema())
    with pytest.raises(ValueError,match='overlap'):
        F.Figure(panels=[F.Panel(row=0,col=0),F.Panel(row=0,col=0)])
    with pytest.raises(ValueError):
        F.Image(source='he', arbitrary_typo=True)
    result = subprocess.run([sys.executable,'-c',
        'from SpatialBiologyToolkit import figures; import sys; '
        'assert not {"numpy","matplotlib","scanpy"}.intersection(sys.modules)'], capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_marker_matching_preserves_biological_names():
    from SpatialBiologyToolkit.figures.sources import match_channel, roi_tokens
    names=['169Tm_CD45RO.tif','CD3e.tif','170Er_HLA_DRA.tif']
    files=[(Path(name),Path(name).stem,roi_tokens(Path(name).stem)) for name in names]
    with pytest.raises(FileNotFoundError):
        match_channel(files,'CD45')
    with pytest.raises(FileNotFoundError):
        match_channel(files,'CD3')
    assert match_channel(files,'HLA_DRA').name=='170Er_HLA_DRA.tif'
    assert match_channel(files,'CD45RO').name=='169Tm_CD45RO.tif'


def test_all_layers_aligned_editable_and_no_intermediate_files(dataset):
    data, root = dataset
    before = set(root.rglob('*'))
    figure = recipe()
    with figure.render(data,'R1') as result:
        assert set(root.rglob('*')) == before
        panels = [ax for ax in result.figure.axes if ax.get_gid() in {p.id for p in figure.panels}]
        assert len(panels) == 6
        for ax in panels[1:]:
            np.testing.assert_allclose(ax.get_xlim(),panels[0].get_xlim())
            np.testing.assert_allclose(ax.get_ylim(),panels[0].get_ylim())
        assert panels[4].images[0].get_zorder() < panels[4].patches[0].get_zorder()
        outlines = [p for p in panels[1].patches if '_outline_' in p.get_gid()]
        assert len(outlines) >= 3
        path = result.save(root/'result.svg')
        result.save(root/'result.png')
        tree = ET.parse(path)
        ns = {'s':'http://www.w3.org/2000/svg'}
        ids = [e.get('id') for e in tree.iter() if e.get('id')]
        assert len(ids)==len(set(ids))
        assert all(t.get('font-family')=='Arial' for t in tree.findall('.//s:text',ns))
        for panel in figure.panels:
            assert tree.find(f'.//s:g[@id="{panel.id}"]',ns) is not None
        assert tree.find(f'.//s:g[@id="{figure.panels[1].layers[0].id}_outline"]',ns) is not None
        assert result.metadata['view']['bounds'] == [4,2,24,20]
        # Numeric values are mapped by ID, never quantised to uint16.
        numeric = panels[3].patches
        colors = {int(p.get_gid().split('_')[-1]):p.get_facecolor() for p in numeric}
        assert colors[1] != colors[2] != colors[90000000]
    assert result.closed


def test_fixed_bounds_no_calibration_io_and_cohort_deduplicates(dataset,monkeypatch):
    data, root = dataset
    from SpatialBiologyToolkit.figures import rendering
    original = rendering.read_image
    reads=[]
    def read(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(rendering,'read_image',read)
    fixed=F.Figure(panels=[F.Panel(row=0,col=0,layers=[F.IMC(channels=[F.Channel('CD3',limits=(0,1000))])])])
    fixed.prepare(data)
    assert reads==[]
    cohort=F.Figure(layout=(1,2),panels=[F.Panel(row=0,col=i,layers=[F.IMC(channels=[F.Channel('CD3')])]) for i in range(2)])
    prepared=cohort.prepare(data,rois=['R1'])
    assert len(reads)==2  # one calibration read per cohort ROI, reused in both panels
    assert len(set(prepared.bounds.values()))==1
    with fixed.render(data,'R1') as output:
        actual=output.figure.axes[0].images[0].get_array()
        assert actual[1,1,0]==pytest.approx(33/1000)


@pytest.mark.parametrize('reducer',['count','sum','mean','fraction'])
def test_hotspot_scores_match_brute_force(dataset,reducer):
    data,_=dataset
    selected=F.Condition(value=F.obs('population'),values=['A'])
    crop=F.Crop.hotspot(size=(12,10), reducer=reducer, where=[selected],
                        score=F.obs('score') if reducer in ('sum','mean') else None)
    context=ROIContext(data,'R1')
    view=select_view(context,crop)
    frame=data.observations('R1')
    scores=[]
    for y in range(24-10+1):
        for x in range(32-12+1):
            inside=(frame.X_loc>=x)&(frame.X_loc<x+12)&(frame.Y_loc>=y)&(frame.Y_loc<y+10)
            match=inside&(frame.population=='A')
            n=inside.sum() if reducer=='fraction' else match.sum()
            if n<1:
                continue
            if reducer in ('count','fraction'):
                total=match.sum()
            else:
                total=frame.loc[match,'score'].sum()
            scores.append(total/n if reducer in ('mean','fraction') else total)
    assert view['score']==pytest.approx(max(scores))
    assert select_view(context,crop)==view
    context.close()


def test_region_selection_resizing_and_saved_bounds(dataset):
    data,_=dataset
    context=ROIContext(data,'R1')
    view=select_view(context,F.Crop.hotspot(size=(8,8),mask_source='regions',mask_labels=[1],reducer='fraction'))
    assert view['score']==1
    assert view['selection_basis']=='mask_area'
    repeated=select_view(context,F.Crop(mode='bounds',roi_bounds={'R1':view['bounds']}))
    assert repeated['bounds']==view['bounds']
    ids=np.array([[0,2**32+1],[5,6]],dtype=np.uint64)
    resized=resample_crop(ids,(8,8),(0,0,8,8),nearest=True)
    assert resized.dtype==ids.dtype
    assert set(np.unique(resized))==set(np.unique(ids))
    assert resized[0,-1]==2**32+1
    context.close()


def test_batch_outputs_and_lifetime(dataset,monkeypatch):
    data,root=dataset
    from SpatialBiologyToolkit.figures.rendering import PreparedFigure
    original=PreparedFigure.render
    references=[]
    def render(self,*args,**kwargs):
        gc.collect()
        assert not any(ref() is not None for ref in references)
        result=original(self,*args,**kwargs)
        references.append(weakref.ref(result.figure))
        return result
    monkeypatch.setattr(PreparedFigure,'render',render)
    report=recipe().export_rois(data,root/'output')
    assert report['status']=='completed'
    assert len(report['results'])==2
    assert (root/'output'/'index.html').is_file()
    assert (root/'output'/'manifest.json').is_file()
    assert len(list((root/'output').glob('*.png')))==2
    assert F.Figure.load(root/'output'/'recipe.json').schema_version==1


def test_pipeline_job_binding(dataset):
    data,root=dataset
    from SpatialBiologyToolkit.config.models import FigureJobConfig
    from SpatialBiologyToolkit.figures.pipeline import run_figure_jobs
    recipe().save(root/'recipe.json')
    job=FigureJobConfig(name='test',recipe=str(root/'recipe.json'),imc_folder=str(root/'imc'),
        mask_folder=str(root/'masks'),image_folders={'he':str(root/'he')},
        label_folders={'regions':str(root/'labels')},pixel_size_um=.5,rois=['R1'],formats=['svg'])
    outputs=run_figure_jobs(data.adata,[job],root/'pipeline')
    assert outputs['test']['status']=='completed'


def test_batch_skip_failure_and_cancellation_record_status(dataset,monkeypatch):
    data,root=dataset
    from SpatialBiologyToolkit.figures.rendering import PreparedFigure
    original=PreparedFigure.render
    def broken(self,roi,**kwargs):
        if roi=='R2':
            raise FileNotFoundError('Synthetic missing modality')
        return original(self,roi,**kwargs)
    monkeypatch.setattr(PreparedFigure,'render',broken)
    report=recipe().export_rois(data,root/'skip',formats=['svg'],on_error='skip')
    assert report['status']=='completed_with_errors'
    assert report['results'][1]['status']=='error'
    def exhausted(self,roi,**kwargs):
        raise MemoryError('Synthetic allocation failure')
    monkeypatch.setattr(PreparedFigure,'render',exhausted)
    with pytest.raises(MemoryError):
        recipe().export_rois(data,root/'failed',formats=['svg'],on_error='skip')
    assert json.loads((root/'failed'/'manifest.json').read_text())['status']=='failed'
    monkeypatch.setattr(PreparedFigure,'render',original)
    state={'cancel':False}
    def progress(event):
        if event['stage']=='export':
            state['cancel']=True
    with pytest.raises(InterruptedError):
        recipe().export_rois(data,root/'cancelled',formats=['svg'],
                            progress=progress,cancelled=lambda:state['cancel'])
    report=json.loads((root/'cancelled'/'manifest.json').read_text())
    assert report['status']=='cancelled'
    assert len(report['results'])==1


def test_strided_hotspot_empty_fallback_and_nonfinite_values(dataset):
    data,_=dataset
    context=ROIContext(data,'R1')
    # Stride larger than the crop also exercises disjoint vertical windows.
    crop=F.Crop.hotspot(size=(8,5),where=[F.Condition(value=F.obs('population'),values=['B'])],stride=8)
    view=select_view(context,crop)
    x,y,w,h=view['bounds']
    assert view['score']==1
    assert x in (0,8,16,24) and y in (0,8,16,19)
    absent=F.Condition(value=F.obs('population'),values=['absent'])
    with pytest.raises(ValueError,match='No eligible'):
        select_view(context,F.Crop.hotspot(size=(8,8),where=[absent]))
    fallback=select_view(context,F.Crop.hotspot(size=(8,8),where=[absent],fallback='center'))
    assert fallback['mode']=='center_fallback'
    context.close()


def test_image_only_annotation_masks_and_safe_render_ids(dataset):
    _,root=dataset
    data=F.Dataset(label_folders={'regions':root/'labels'})
    figure=F.Figure(panels=[F.Panel(row=0,col=0,layers=[F.LabelMask(source='regions',labels={1:'Tissue'},rendering='raster')])])
    with figure.render(data,'annotated_R1') as result:
        image=result.figure.axes[0].images[0].get_array()
        assert image[0,0,3]==0
        assert image[0,-1,3]==1
    from SpatialBiologyToolkit.figures.export import _output_name
    assert '/' not in _output_name('../../CON')
    assert _output_name('ROI/1')!=_output_name('ROI_1')
    zero=F.Figure(panels=[F.Panel(row=0,col=0,layers=[F.LabelMask(source='regions',labels={0:'Outside'},background=1)])])
    with zero.render(data,'annotated_R1') as result:
        assert any(p.get_gid().endswith('_object_0') for p in result.figure.axes[0].patches)


def test_duplicate_cell_ids_and_absent_physical_calibration_fail(dataset):
    data,_=dataset
    data.adata.obs.loc['c1','ObjectNumber']=1
    with pytest.raises(ValueError,match='unique'):
        recipe().render(data,'R1')
    data.pixel_size_um=None
    with pytest.raises(ValueError,match='pixel_size_um'):
        recipe().prepare(data)


def test_preview_uses_same_view_and_does_not_queue_notebook_figures(dataset):
    data,_=dataset
    import matplotlib.pyplot as plt
    previous=plt.isinteractive()
    plt.ion()
    caller=plt.figure()
    try:
        prepared=recipe().prepare(data)
        with prepared.render('R1',dpi=60) as preview, prepared.render('R1',dpi=300) as final:
            assert preview.metadata['view']==final.metadata['view']
            assert preview.metadata['scaling']==final.metadata['scaling']
            assert plt.get_fignums()==[caller.number]
            assert plt.isinteractive()
    finally:
        plt.close(caller)
        plt.interactive(previous)
