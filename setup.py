from setuptools import setup, find_packages

setup(
    name='SpatialBiologyToolkit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Core numerical stack
        'numpy',            # array maths
        'pandas',           # tabular data handling
        'scipy',            # scientific routines and stats helpers
        'pyyaml',           # config file parsing
        'tqdm',             # progress bars

        # Single-cell / spatial omics
        'anndata',          # AnnData container
        'scanpy',           # single-cell workflows
        'bbknn',            # batch correction via BBKNN
        'harmonypy',        # Harmony batch correction backend
        'umap-learn',       # UMAP embeddings
        'scikit-learn',     # machine learning utilities
        'statsmodels',      # statistical models and tests
        'lifelines',        # survival analysis
        'colorcet',         # colour maps
        'sc3s',             # clustering utilities
        'ehrapy',           # healthcare/scRNA helpers
        'squidpy',          # spatial omics tooling
        'readimc',          # IMC file readers

        # Plotting and geometry
        'matplotlib',       # plotting backend
        'seaborn',          # statistical plots
        'tifffile',         # TIFF IO
        'shapely',          # geometric operations
        'networkx',         # graph utilities
        'scikit-image',     # image processing
        'ipython',          # rich display utilities

        # Deep learning / segmentation / denoising
        'torch',            # PyTorch for segmentation
        'tensorflow',       # TensorFlow for denoising models
        'cellpose',         # cell segmentation
        'opencv-python',    # computer vision utilities
        'psutil',           # system resource monitoring
        'alpineer',         # Nimbus segmentation helpers

        # Interactive viewers and GUI layers
        'napari',           # interactive image viewer
        'magicgui',         # GUI building for napari plugins
        'qtpy',             # Qt abstraction layer
        'vispy',            # GPU-accelerated visualization

        # Optional/IPython helpers
        'ipykernel',        # Jupyter kernel support

        # Docs/build tooling
        'sphinx',           # documentation generator
        'setuptools'        # packaging utilities
    ],
    description='A tool kit for analysing high dimensional spatial data',
    author='Michael Haley',
    author_email='mrmichaelhaley@gmail.com',
    url='https://github.com/dr-michael-haley/imcanalysis',
)
