#! /bin/bash --login
#SBATCH -p multicore
#SBATCH -t 0-8
#SBATCH -n 4
#SBATCH --mem=32G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Plot CellVision UMAPs, cluster-explanation QC, original-label confusion, source projections, and H5SC galleries
#@IN:   CellVision clustered AnnData, source AnnData, and exact training H5SC images
#@OUT:  CellVision figures, cluster-explanation QC, and comparison tables in the active execution report
#@ENV:  sbt-scportrait
#@MODULE:  SpatialBiologyToolkit.scripts.cellvision_plot
#@CONFIG: general, cellvision, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_SCPORTRAIT:-sbt-scportrait}"
python -m SpatialBiologyToolkit.scripts.cellvision_plot
