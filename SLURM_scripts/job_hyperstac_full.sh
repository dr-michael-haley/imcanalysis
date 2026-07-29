#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -n 12
#SBATCH --mem=96G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Run HyPERSTAC preprocessing, representation, sensitivity, visualisation, Cox, and stability in one GPU job
#@IN:   configured ROI/channel TIFFs, Cox feature sources, and clinical survival metadata
#@OUT:  reusable HyPERSTAC assets plus managed visualisation, Cox, and cross-Leiden stability reports
#@ENV:  hyperstac
#@MODULE:  SpatialBiologyToolkit.scripts.hyperstac_full
#@CONFIG: general, hyperstac, cox, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_HYPERSTAC:-${IMC_ENV_HYPERSTAC:-hyperstac-imc}}"

python -m SpatialBiologyToolkit.scripts.hyperstac_full
