#! /bin/bash --login
#SBATCH -p multicore
#SBATCH -t 0-8
#SBATCH -n 4
#SBATCH --mem=32G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Plot CellVision UMAPs, original-label confusion, source-UMAP projections, and H5SC channel galleries
#@IN:   CellVision clustered AnnData, source AnnData, and exact training H5SC images
#@OUT:  CellVision figures and comparison tables in the active execution report
#@ENV:  scPortrait
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_plot
#@CONFIG: general, cellvision, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_SCPORTRAIT:-${IMC_ENV_SCPORTRAIT:-scPortrait}}"
python -m SpatialBiologyToolkit.scripts.cellvision_plot
