# BioBatchNet

[![PyPI version](https://badge.fury.io/py/biobatchnet.svg)](https://badge.fury.io/py/biobatchnet)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BioBatchNet is a VAE framework for batch effect correction in biological data, supporting both **single-cell RNA-seq (scRNA-seq)** and **Imaging Mass Cytometry (IMC)** data.

---

## Features

- **Multi-modal Support**: Works with both scRNA-seq and IMC data
- **Easy-to-Use API**: One-line batch correction with `correct_batch_effects()`
- **Flexible Architecture**: Customizable neural network parameters
- **Adaptive Loss Weights**: Automatically adjusts based on dataset characteristics
- **Comprehensive Documentation**: Detailed usage examples and interactive tutorials

---

## Installation

### Create Environment (Required for All Users)

```bash
conda env create -f environment.yml
conda activate biobatchnet
```

### Install BioBatchNet

**For Users (Recommended):**
```bash
pip install biobatchnet
```

**For Development:**
```bash
git clone https://github.com/UoM-HealthAI/BioBatchNet
cd BioBatchNet
pip install -e .
```

---

## Usage

### Python API (Recommended for Users)

The simplest way to use BioBatchNet is through the high-level API:

```python
import pandas as pd
import numpy as np
import anndata as ad
from biobatchnet import correct_batch_effects

# Load your data
adata = ad.read_h5ad('your_data.h5ad')
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

# Prepare batch labels (must be integers)
unique_batches = np.unique(adata.obs['BATCH'].values)
batch_to_int = {batch: i for i, batch in enumerate(unique_batches)}
batch_labels = np.array([batch_to_int[b] for b in adata.obs['BATCH'].values])

# Correct batch effects
bio_embeddings, batch_embeddings = correct_batch_effects(
    data=pd.DataFrame(X),
    batch_info=pd.DataFrame({'BATCH': batch_labels}),
    batch_key='BATCH',
    data_type='imc',        # 'imc' or 'scrna'
    latent_dim=20,
    epochs=100,
    device='cuda'           # or 'cpu'
)

# Add embeddings to AnnData
adata.obsm['X_biobatchnet'] = bio_embeddings
```
**For detailed documentation and examples:**  
(Note: PyPI pages do not support previewing relative links from the repository, so absolute links and nbviewer previews are provided here. If you cannot open the original relative links on PyPI, this is expected.)

- 📖 **USAGE Documentation:** [GitHub](https://github.com/UoM-HealthAI/BioBatchNet/blob/main/USAGE.md)
- 📓 **Tutorial Notebook:** [GitHub](https://github.com/UoM-HealthAI/BioBatchNet/blob/main/tutorial.ipynb) | [nbviewer preview](https://nbviewer.org/github/UoM-HealthAI/BioBatchNet/blob/main/tutorial.ipynb)

### Config-based Processing in SpatialBiologyToolkit

When using BioBatchNet as part of the SpatialBiologyToolkit pipeline, configure it in your `config.yaml`:

#### Single Parameter Set (No Scanning)

```yaml
process:
  batch_correction_obs: 'ROI'  # Column in adata.obs with batch information
  
  # BioBatchNet parameters (nested dictionary format)
  biobatchnet_params:
    data_type: 'imc'           # 'imc' or 'scrnaseq'
    latent_dim: 20             # Latent dimension size
    epochs: 100                # Training epochs
    device: null               # 'cpu', 'cuda', or null for auto-detect
    use_raw: true              # Use adata.raw.X instead of adata.X
    extra_params:              # Additional parameters passed to BioBatchNet
      lambda_bio: 1.0          # Weight for biological loss
      lambda_batch: 0.5        # Weight for batch loss
      learning_rate: 0.001     # Learning rate
  
  biobatchnet_run_leiden: true  # Run Leiden clustering after correction
  leiden_resolutions_list: [0.3, 1.0]
  n_neighbors: 15              # Number of neighbors for scanpy
```

#### Parameter Scanning (Multiple Sets)

To test multiple parameter combinations:

```yaml
process:
  batch_correction_obs: 'ROI'
  
  # Base parameters (used if scan_include_base: true)
  biobatchnet_params:
    data_type: 'imc'
    latent_dim: 20
    epochs: 100
    device: null
    use_raw: true
  
  # Parameter scan sets
  biobatchnet_scan_parameter_sets:
    - name: 'high_latent'
      latent_dim: 40
      epochs: 150
      
    - name: 'custom_weights'
      extra_params:
        lambda_bio: 2.0
        lambda_batch: 0.3
        
    - name: 'low_latent_long'
      latent_dim: 10
      epochs: 200
      extra_params:
        learning_rate: 0.0005
  
  biobatchnet_scan_include_base: true  # Also run base parameters
```

Each parameter set will generate a separate output file and QC directory with results.

### Config-based Training (For Development/Research)

For reproducing research results or training with specific configurations:

```bash
# For IMC data
python biobatchnet/IMC.py --config biobatchnet/config/IMC/IMMUcan.yaml

# For scRNA-seq data
python biobatchnet/Gene.py --config biobatchnet/config/scRNA/pancreas.yaml
```

**Configuration files:**
- IMC datasets: `biobatchnet/config/IMC/`
- scRNA-seq datasets: `biobatchnet/config/scRNA/`

These scripts expect datasets under `Data/` directory (see YAML files for exact paths).

---

## CPC Usage

To use CPC, ensure you are running in the same environment as BioBatchNet.

All experiment results can be found in the following directory:

```bash
cd CPC/IMC_experiment
```

**✅ Key Notes:**
- CPC requires embeddings from BioBatchNet as input
- Sample data includes batch-corrected IMMUcan IMC embeddings
- Ensure the same computational environment as BioBatchNet before running CPC

---

## Data

**Download scRNA-seq Data:**
- Available on Google Drive: [Download Link](https://drive.google.com/drive/folders/1m4AkNc_KMadp7J_lL4jOQj9DdyKutEZ5?usp=drive_link)

**Download IMC Data:**

The IMC dataset can be accessed from the Bodenmiller Group IMC datasets repository. Visit the link below to explore and download the datasets:

🔗 [IMC Datasets - Bodenmiller Group](https://github.com/BodenmillerGroup/imcdatasets)

---

## Citation

If you use BioBatchNet in your research, please cite:

```
Liu H, Zhang S, Mao S, et al. BioBatchNet: A Dual-Encoder Framework for Robust Batch Effect Correction in Imaging Mass Cytometry[J]. bioRxiv, 2025: 2025.03.15.643447.
```

---

## License

MIT License