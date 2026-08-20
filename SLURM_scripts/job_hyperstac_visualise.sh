#! /bin/bash --login
#SBATCH -p himem
#SBATCH -t 1-0
#SBATCH -n 12
#SBATCH --mem=96G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Run HyPERSTAC clustering scans and create embedding, marker, spatial, and gallery reports
#@IN:   HyPERSTAC representation/metric AnnData, patches, and optional permutation AnnData
#@OUT:  managed HyPERSTAC visualisation report and optional clustered representation AnnData update
#@ENV:  sbt-tensorflow
#@MODULE:  SpatialBiologyToolkit.scripts.hyperstac_visualise
#@CONFIG: general, hyperstac, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_TENSORFLOW:-sbt-tensorflow}}"
python -m SpatialBiologyToolkit.scripts.hyperstac_visualise
