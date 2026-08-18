#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 12

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Extract selected identity-tracked IMC cells into one 36 px H5SC dataset with scPortrait
#@IN:   CellVision AnnData, ROI/channel TIFF folders, and labelled masks from config.yaml
#@OUT:  cellvision.asset_folder H5SC, cell identity table, and extraction metadata
#@ENV:  sbt-scportrait
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_extract
#@CONFIG: general, cellvision, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_SCPORTRAIT:-sbt-scportrait}"
python -m SpatialBiologyToolkit.scripts.cellvision_extract
