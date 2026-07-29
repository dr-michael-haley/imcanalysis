#! /bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 12
#SBATCH --mem=96G

#SBATCH --mail-user=${IMC_EMAIL}
#SBATCH --mail-type=ALL

set -e

#@DESC: Tile normalized IMC images, train HyPERSTAC VICReg, and extract patch representations
#@IN:   hyperstac.asset_folder/normalised_images or configured source ROI/channel TIFFs
#@OUT:  hyperstac.asset_folder patches, model weights, metadata, and representation/metric AnnData
#@ENV:  hyperstac
#@MODULE:  SpatialBiologyToolkit.scripts.hyperstac_model
#@CONFIG: general, hyperstac, logging

source "$HOME/imcanalysis/SLURM_scripts/job_env.sh"
conda activate "${SBT_CONDA_ENV_HYPERSTAC:-${IMC_ENV_HYPERSTAC:-hyperstac-imc}}"
python -m SpatialBiologyToolkit.scripts.hyperstac_model
