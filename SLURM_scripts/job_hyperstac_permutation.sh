#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 12
#SBATCH --mem=64G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Quantify HyPERSTAC embedding sensitivity to channel zeroing and pixel shuffling
#@IN:   HyPERSTAC patch metadata, patch arrays, representation AnnData, and encoder weights
#@OUT:  hyperstac.asset_folder/permutation_sensitivity AnnData and managed sensitivity tables
#@ENV:  sbt-tensorflow
#@MODULE:  SpatialBiologyToolkit.scripts.hyperstac_permutation
#@CONFIG: general, hyperstac, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV:-${SBT_CONDA_ENV_TENSORFLOW:-sbt-tensorflow}}"
python -m SpatialBiologyToolkit.scripts.hyperstac_permutation
