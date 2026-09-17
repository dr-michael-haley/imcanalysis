"""Exercise notebook display queues, which the Agg-only export tests cannot see."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from types import SimpleNamespace

from SpatialBiologyToolkit import plotting, backgating as bg


def test_notebook_batch_figures_do_not_accumulate_in_inline_display_queue(tmp_path):
    script = textwrap.dedent('''
        import gc
        import sys
        import weakref
        from pathlib import Path
        import matplotlib
        matplotlib.use('module://matplotlib_inline.backend_inline', force=True)
        import matplotlib.pyplot as plt
        from matplotlib_inline.backend_inline import show
        import anndata as ad
        import numpy as np
        import pandas as pd
        from PIL import Image
        from SpatialBiologyToolkit import backgating as bg, plotting

        root = Path(sys.argv[1])
        Image.fromarray(np.full((60,60,3), 70, dtype=np.uint8)).save(root/'R.png')
        obs = pd.DataFrame({'ROI':['R'], 'population':['P'], 'Master_Index':[1],
                            'X_loc':[30], 'Y_loc':[30]}, index=['cell'])
        data = ad.AnnData(np.ones((1,1)), obs=obs, var=pd.DataFrame(index=['M']))
        plt.ion()
        caller = plt.figure(figsize=(1,1))
        original_queue = list(show._to_draw)
        assert original_queue == [caller]
        refs = []
        for i in range(6):
            fig = plotting.create_population_overlay(
                data, 'P', 'population', 'R', root/'R.png', output_path=root/'overlay.png',
                svg_output_path=root/'overlay.svg', show_label=False, verbose=False,
                comparison_images=[root] if i % 2 else None)
            refs.append(weakref.ref(fig))
            fig.clear()
            del fig
            gc.collect()
            assert show._to_draw == original_queue, 'Saved overlays retained by notebook queue'
            assert not any(ref() is not None for ref in refs), 'Overlay figure not released'
            assert plt.isinteractive()
            assert plt.get_fignums() == [caller.number]
        assert (root/'overlay.svg').is_file()

        bg.make_images = lambda **kwargs: None
        bg.backgating(data, [1], 10, root, output_folder=root, overview_images=False)
        assert (root/'Cells.svg').is_file()
        assert show._to_draw == original_queue
        assert plt.isinteractive()

        # Assessment and gallery APIs also run inside the protected context.
        pd.DataFrame({'Red':['M'],'Green':[None],'Blue':[None]}, index=['P']).to_csv(root/'settings.csv')
        def fake_backgating(**kwargs):
            assert not plt.isinteractive()
        bg.backgating = fake_backgating
        bg.backgating_assessment(data, root, 'population', output_folder=root,
            backgating_settings_file='settings.csv', mode='load_markers', population_overlays=False)
        assert plt.isinteractive() and show._to_draw == original_queue

        folder = root/'P'/'population_overlays'
        folder.mkdir(parents=True)
        Image.fromarray(np.zeros((40,40,3), dtype=np.uint8)).save(folder/'R_population_overlay.png')
        bg.create_population_overlay_galleries(root, ['P'], 1, 1, source_format='png', output_format='png')
        assert plt.isinteractive() and show._to_draw == original_queue

        # Display-only calls retain their interactive notebook behaviour.
        old_show = plt.show
        plt.show = lambda: None
        interactive_fig = plotting.create_population_overlay(data,'P','population','R',root/'R.png',
                                                              show_label=False, verbose=False)
        assert interactive_fig in show._to_draw
        plt.show = old_show
        plt.close('all')
        show._to_draw.clear()
        print('Inline queue unchanged across saved overlays, thumbnails, assessment and galleries.')
    ''')
    env = os.environ.copy()
    env['NUMBA_DISABLE_JIT'] = '1'
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_saved_overlay_failure_closes_own_figures_and_restores_interactivity(tmp_path, monkeypatch):
    Image.fromarray(np.zeros((40,40,3), dtype=np.uint8)).save(tmp_path/'source.png')
    data = SimpleNamespace(obs=pd.DataFrame({'ROI':['R'],'population':['P'],'X_loc':[20],'Y_loc':[20]}))
    caller = plt.figure()
    before = plt.get_fignums()
    original_mode = plt.isinteractive()
    plt.ion()

    def fail(*args, **kwargs):
        raise MemoryError('synthetic renderer allocation failure')

    monkeypatch.setattr(plt.Axes, 'imshow', fail)
    try:
        with pytest.raises(MemoryError):
            plotting.create_population_overlay(data,'P','population','R',tmp_path/'source.png',
                                                output_path=tmp_path/'out.png')
        assert plt.get_fignums() == before
        assert plt.isinteractive()
    finally:
        plt.close(caller)
        plt.interactive(original_mode)


def test_overlay_converts_population_labels_only_after_roi_filtering(tmp_path, monkeypatch):
    Image.fromarray(np.zeros((40,40,3), dtype=np.uint8)).save(tmp_path/'source.png')
    data = SimpleNamespace(obs=pd.DataFrame({'ROI':['R']*2 + ['other']*1000,
        'population':pd.Categorical([1]*1002),'X_loc':[20]*1002,'Y_loc':[20]*1002}))
    original = pd.Series.astype

    def convert(self, dtype, *args, **kwargs):
        if self.name == 'population' and dtype is str:
            assert len(self) == 2
        return original(self, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'astype', convert)
    fig = plotting.create_population_overlay(data,'1','population','R',tmp_path/'source.png',
                                              output_path=tmp_path/'out.png', show_label=False)
    assert len(fig.axes[0].collections[0].get_offsets()) == 2


def test_assessment_stops_on_memory_error_instead_of_silently_skipping(tmp_path, monkeypatch):
    obs = pd.DataFrame({'ROI':['R1','R2'],'population':['P','P'], 'Master_Index':[1,2]})
    data = SimpleNamespace(obs=obs)
    pd.DataFrame({'Red':['M'],'Green':[None],'Blue':[None]}, index=['P']).to_csv(tmp_path/'settings.csv')
    folder = tmp_path/'P'
    folder.mkdir()
    for roi in obs.ROI:
        Image.fromarray(np.zeros((20,20,3),dtype=np.uint8)).save(folder/f'{roi}.png')
    monkeypatch.setattr(bg, 'backgating', lambda **kwargs: None)
    calls = []

    def fail(**kwargs):
        calls.append(kwargs['roi_name'])
        raise MemoryError('synthetic failure')

    monkeypatch.setattr(bg, 'create_population_overlay', fail)
    with pytest.raises(MemoryError):
        bg.backgating_assessment(data, tmp_path, 'population', output_folder=tmp_path,
                                 backgating_settings_file='settings.csv', mode='load_markers')
    assert calls == ['R1']
