#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 12

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Train the PyTorch CellVision VICReg encoder and extract identity-aligned cell embeddings
#@IN:   cellvision.asset_folder/extraction/data/single_cells.h5sc and extraction metadata
#@OUT:  CellVision VICReg checkpoint, embedding AnnData, and training diagnostics
#@ENV:  sbt-scportrait
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_embed
#@CONFIG: general, cellvision, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_SCPORTRAIT:-sbt-scportrait}"
python -m SpatialBiologyToolkit.scripts.cellvision_embed
